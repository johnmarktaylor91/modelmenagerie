"""Bounded per-leg terminal-review backlog sweep: the pure halves.

The requeue-grant lane admits a re-drive only over a REVIEW-HELD terminal
(``status.human_review.required`` true -- ``driver_admission`` grant
reconciliation, hold reason ``unreviewed-terminal``).  The marking itself is
produced by exactly one authority: the terminal-disposition checker lane,
whose only driver-side caller is ``CrawlerDriver._route_terminal_author_result``
-- reached solely when an author session IN THE CURRENT LEG returns a typed
terminal recommendation.  A failed terminal head recorded in an earlier leg is
never scheduled again (WS-1: a raw head re-enters only through a bindable
grant, and the grant is exactly what the missing review holds), so the review
lane structurally cannot sweep its own backlog: grants wait on a review, the
review waits on scheduling, and scheduling waits on the grants.

This module holds the pure halves of the per-leg sweep that breaks that cycle;
the effectful half (rehydration, checker sessions, gate appends, the review
revision) lives on the driver
(``CrawlerDriver._review_terminal_backlog``) and reuses the fresh-terminal
lane verbatim -- same envelope build, same spend admission and calibration
gate, same gate-ledger append, same ``validate_terminal_disposition_gate``
decision, and the same ``human_review = not decision.accepted`` marking rule.

The verdict is applied in the CONTAINMENT shape, not the fresh-mint shape: a
fresh terminal that fails its gate is minted ``failed:accuracy-gate``, but a
backlog head is an already-settled record whose facts stay true whatever the
review concludes.  The precedent is the terminal containment ladder
(``_contain_terminal_finalization_refusal``), which re-appends the ORIGINAL
status code with ``human_review=True`` -- the exact review-held shape operator
grants bind (``_grant_stage_binds_terminal`` keys the grant's stage against
the head's recorded stage, so rewriting the stage would convert the
``unreviewed-terminal`` hold into a permanent ``stage-mismatch`` hold).

This module also holds the pure halves of the ORPHAN-CITATION ADOPTION pass
(effectful half: ``CrawlerDriver._adopt_orphan_citations``).  The partition
proof refuses any in-window head whose citation names a run that never joined
the open generation ("position is not provenance") -- the correct tripwire,
seen live on 2026-08-27 when eight out-of-band review-hold appends (built
through this module's candidate builder under a context whose run never
joined any generation, and inheriting their superseded heads' retired
``crawler-run`` citation) refused ordinary leg 2 at admission.  Adoption is
the lawful head-continuity remedy: a review-held failure terminal whose row
is provably byte-equal to the production containment marking over sealed
pre-window history is re-appended by the leg under its OWN joined run, the
orphan row stays disclosed in a durable census, and everything failing the
closed rule keeps refusing exactly as before.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final, Mapping, Optional

from menagerie.crawler.models import JsonObject

#: Named per-leg bound on backlog terminal reviews.  Generous by design: the
#: measured queue is ~124 heads and a sol terminal session costs ~2-4 minutes,
#: so a healthy checker lane may clear the whole backlog in one leg; the
#: brakes are the lane's existing valves (per-head checker-lane deferral
#: parks, the leg session/wall budgets in ``_admit_checker_spend``), never a
#: silent skip -- heads beyond the bound are disclosed in the leg census.
TERMINAL_REVIEW_BACKLOG_PER_LEG: Final[int] = 128

#: Operational census disposition for one leg's backlog sweep.
TERMINAL_REVIEW_CENSUS_DISPOSITION: Final[str] = "terminal-review-backlog"

#: Operational disposition the sweep appends per CONFIRMED backlog review.
#: The ``already_adjudicated`` pre-filter keys on these driver-owned rows
#: (plus a still-current gate identity): the fresh-mint terminal gate alone
#: must never satisfy the filter, because every review-free settled head HAS
#: such a gate -- that is exactly why it is in the backlog -- and skipping on
#: it would mean no current-identity claim could ever buy the independent
#: re-review an operator grant waits on.
TERMINAL_REVIEW_ADJUDICATED_DISPOSITION: Final[str] = "terminal-review-adjudicated"

#: Exact ``source_resolution.decision`` wording `_assemble_terminal_model`
#: stamps on a typed terminal verdict whose disposition gate was never
#: obtained (the ``terminal_gate_obtained=False`` arm).  The sweep's closed
#: classification rule keys on this exact string -- a byte-equal producer
#: constant, pinned by test against ``driver_models`` -- so a claim-bearing
#: head that cannot be rehydrated from ledger-bound evidence is never
#: mislabeled as claim-free.
TERMINAL_UNADJUDICATED_DECISION: Final[str] = (
    "author published a typed terminal verdict; its disposition gate "
    "was never obtained, so the verdict is unadjudicated"
)

#: Maximum stable IDs disclosed per census class (counts are always exact).
TERMINAL_REVIEW_CENSUS_ID_CAP: Final[int] = 32

#: Operational census disposition for one leg's orphan-citation adoption pass
#: (:meth:`CrawlerDriver._adopt_orphan_citations`).  Per-occurrence identity,
#: exactly like the backlog-sweep census.
ORPHAN_CITATION_ADOPTION_DISPOSITION: Final[str] = "orphan-citation-adoption"

#: Per-member operational census disposition, appended BEFORE the adoption
#: re-append it discloses (Opus C5): a leg dying between the member event and
#: the leg's summary census can therefore never leave an adopted row without
#: a durable disclosure.  A member event with no matching adopted revision is
#: coherent (the leg died before its append landed; the member is still
#: orphaned and the next leg retries and re-discloses it).
ORPHAN_CITATION_ADOPTION_MEMBER_DISPOSITION: Final[str] = (
    "orphan-citation-adoption-member"
)

#: Named per-leg bound on adoption re-appends (Opus C3: the adjacent backlog
#: sweep is bounded and shutdown-guarded per head; the adoption pass matches
#: that discipline).  Members beyond the bound are disclosed ``beyond_bound``
#: on the leg census and stay orphaned -- the partition proof then refuses the
#: leg exactly as for any unadoptable member, and the NEXT leg adopts the next
#: tranche, so progress is monotonic and every append burst is bounded and
#: disclosed.
ORPHAN_ADOPTION_PER_LEG: Final[int] = 128

#: Fields the ledger assigns per appended row; stripped on every re-append and
#: excluded from every verbatim comparison below.
_LEDGER_ASSIGNED_FIELDS: Final[frozenset[str]] = frozenset(
    {"record_seq", "record_revision", "ledger_seq", "payload_sha256"}
)

#: The EXACT (closed) set of dependency-vector axes that legitimately move
#: between two appends of the SAME content: the identity axes
#: ``prepare_model`` re-derives from LIVE reducer context (the policy /
#: prompt / schema / dispatcher closures stamped from ``AuthorityContext``
#: in ``derive_dependency_vector``, plus the re-derived status-proof
#: identity).  Everything ELSE in the closed 22-field vector is PROVENANCE,
#: not rotation, and must survive byte-exact in every comparison below:
#: the snapshot and item pins (``intake_snapshot_id``,
#: ``intake_snapshot_sha256``, ``intake_item_sha256`` -- read as authority
#: by the partition proof's ``recorded_under`` refusal, the register door's
#: recorded-history proof, and the close's exact-pin staleness cohort) and
#: the per-work evidence bindings (attempt ids, artifact transaction and
#: claims, recipe and representative revisions, resolved result/gate/
#: manifest/proposal references, the proof-attested environment
#: generation).  A blanket ``dependency_vector.`` exclusion let adoption
#: re-stamp the pins with zero adjudication (Opus C1, 2026-08-27); this
#: enumeration is what the rule actually adjudicates.  Widening it is a new
#: adoption law and needs its own deliberated, reviewed change (the same
#: bar as :data:`ORPHAN_HOLD_ALLOWED_DIFF`).
ORPHAN_ROTATABLE_VECTOR_AXES: Final[frozenset[str]] = frozenset(
    {
        "dependency_vector.author_result_schema_identity",
        "dependency_vector.author_dispatcher_identity",
        "dependency_vector.author_prompt_identity",
        "dependency_vector.checker_prompt_identity",
        "dependency_vector.terminal_rule_identity",
        "dependency_vector.status_proof_identity",
        "dependency_vector.runner_identity",
        "dependency_vector.award_closure_identity",
        "dependency_vector.publication_policy_identity",
    }
)

#: The EXACT (closed) path set on which a lawful orphan review-hold row may
#: differ from its below-floor parent: the containment marking
#: (:func:`terminal_review_hold_candidate`) plus ledger parentage.  Anything
#: else differing means the out-of-band row granted itself content beyond the
#: production hold shape, and adoption refuses.  Widening this set is a new
#: adoption law and needs its own deliberated, reviewed change (the same bar
#: as the four-part waiver doctrine's closed proof rules).
ORPHAN_HOLD_ALLOWED_DIFF: Final[frozenset[str]] = (
    frozenset(
        {
            "parent_revision",
            "status.supersedes_revision",
            "status.human_review.required",
            "status.human_review.queue",
            "status.human_review.requested_at",
        }
    )
    | _LEDGER_ASSIGNED_FIELDS
)

#: The EXACT (closed) path set on which the prepared adoption re-append may
#: differ from the orphan head it supersedes: ledger parentage plus the
#: re-citation itself.  ``provenance.producer_run_id`` / ``machine_id`` name
#: the run that APPENDS the row ("position is provenance"); every content
#: field -- author model, checker identity, status facts, budget, evidence
#: bindings -- must survive byte-verbatim or the adoption refuses.
ORPHAN_ADOPTION_ALLOWED_DIFF: Final[frozenset[str]] = (
    frozenset(
        {
            "parent_revision",
            "status.supersedes_revision",
            "provenance.producer_run_id",
            "provenance.machine_id",
        }
    )
    | _LEDGER_ASSIGNED_FIELDS
)

#: The exact containment hold block queue label (`_assemble_terminal_model`
#: and :func:`terminal_review_hold_candidate` both stamp it).
_HOLD_QUEUE: Final[str] = "crawler-human-review"


def _flatten_record(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten one record to dot-joined leaf paths.

    Non-empty mappings recurse; empty mappings, sequences, and scalars are
    leaves (sequences compare whole, so a reordered list is a difference).
    """

    flat: dict[str, Any] = {}
    for key, value in record.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(_flatten_record(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _disallowed_diff_paths(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    allowed_exact: frozenset[str],
) -> list[str]:
    """Return sorted differing leaf paths outside the allowed exact-path set."""

    missing = object()
    flat_left = _flatten_record(left)
    flat_right = _flatten_record(right)
    return [
        path
        for path in sorted(set(flat_left) | set(flat_right))
        if flat_left.get(path, missing) != flat_right.get(path, missing)
        and path not in allowed_exact
    ]


def orphan_hold_adoption_refusal(
    head: Mapping[str, Any],
    parent: Mapping[str, Any],
    *,
    parent_row_index: int,
    window_floor: int,
) -> Optional[str]:
    """Classify one orphan-cited head for adoption; ``None`` means adoptable.

    The closed proof rule for the leg's orphan-citation adoption pass
    (2026-08-27, ordinary-leg-2 admission refusal: 8 out-of-band review-hold
    appends citing a run that never joined the generation).  Adoption grants
    NOTHING positive -- it re-appends an already NEGATIVE disposition (a
    review-held failure terminal) under the adopting run's own citation, so
    the partition proof's "position is not provenance" tripwire stays armed
    verbatim.  Eligibility is exhaustively enumerated and replay-derived:

    * the head is a review-held failure terminal carrying exactly the
      containment hold block (required/reason/queue/requested_at);
    * its parent (the revision it superseded) sits STRICTLY BELOW the open
      generation's window floor -- sealed pre-generation history the leg
      genesis already attested, so a forged two-row chain (fake parent plus
      hold) can never launder content through adoption; and
    * the head differs from that parent ONLY on :data:`ORPHAN_HOLD_ALLOWED_DIFF`
      (the production containment marking) plus the enumerated rotatable
      identity axes :data:`ORPHAN_ROTATABLE_VECTOR_AXES` -- i.e. the orphan
      row is byte-equal to the containment MARKING the sweep applies
      (:func:`terminal_review_hold_candidate` over the same parent).  The
      dependency-vector snapshot/item pins and evidence bindings are
      ADJUDICATED, never excluded: a pin or evidence move refuses typed.
      Adoption never consults ``validate_terminal_disposition_gate``, so
      this proves marking-shape equality, never the sweep's adjudicated
      verdict -- fail-safe, because a hold is a negative disposition the
      sweep could only have RELAXED to ``confirmed``, never worsened.

    Anything failing any clause is NOT adopted; the partition proof keeps
    refusing the leg exactly as before (fail-closed, disclosed, never
    silently normalized).

    Parameters
    ----------
    head:
        The current in-window head whose citation is orphaned.
    parent:
        The ledger row ``head["parent_revision"]`` resolves to.
    parent_row_index, window_floor:
        The parent's position among the source's non-empty rows and the open
        generation's baseline row count for that source.

    Returns
    -------
    str | None
        ``None`` when adoptable, else the exact typed refusal reason.
    """

    status = head.get("status")
    if not isinstance(status, Mapping):
        return "head carries no status block"
    if status.get("kind") != "failed":
        return (
            f"head status kind is {status.get('kind')!r}, not 'failed': only "
            "review-held failure terminals are adoptable"
        )
    review = status.get("human_review")
    if not isinstance(review, Mapping) or review.get("required") is not True:
        return (
            "head is not review-held: unreviewed heads belong to the "
            "terminal-review sweep, never to citation adoption"
        )
    if review.get("reason") is not None:
        return (
            f"hold block carries reason {review.get('reason')!r}; the "
            "containment marking stamps None"
        )
    if review.get("queue") != _HOLD_QUEUE:
        return (
            f"hold block queue is {review.get('queue')!r}, not "
            f"{_HOLD_QUEUE!r}: not the containment marking"
        )
    requested_at = review.get("requested_at")
    if not isinstance(requested_at, str) or not requested_at:
        return "hold block requested_at is not a non-empty string"
    if int(parent_row_index) >= int(window_floor):
        return (
            f"parent row index {parent_row_index} is inside the open window "
            f"(floor {window_floor}): adoption requires sealed pre-generation "
            "history, and an in-window parent is itself unprovenanced"
        )
    if head.get("parent_revision") != parent.get("record_revision"):
        return "parent_revision does not bind the supplied parent row"
    if status.get("supersedes_revision") != parent.get("record_revision"):
        return "status.supersedes_revision does not bind the parent head revision"
    offending = _disallowed_diff_paths(
        head, parent, ORPHAN_HOLD_ALLOWED_DIFF | ORPHAN_ROTATABLE_VECTOR_AXES
    )
    if offending:
        shown = ", ".join(offending[:5])
        suffix = "" if len(offending) <= 5 else f" (+{len(offending) - 5} more)"
        return (
            "head differs from its parent outside the containment marking: "
            f"{shown}{suffix}"
        )
    return None


def orphan_adoption_drift(
    prepared: Mapping[str, Any], head: Mapping[str, Any]
) -> Optional[str]:
    """Refuse an adoption re-append that changed anything beyond re-citation.

    Runs AFTER ``prepare_model`` so reducer-state drift between the orphan
    append and the adopting leg (a rotated identity axis re-deriving a status
    projection differently, a moved artifact authority) refuses typed instead
    of appending silently altered content.  Only the enumerated rotatable
    identity axes (:data:`ORPHAN_ROTATABLE_VECTOR_AXES`) may move inside the
    dependency vector; ``prepare_model`` stamps the adopting leg's ACTIVE
    snapshot/item pins, so this comparison additionally proves the orphan
    row's recorded pins already equal the leg's active values -- old work
    whose recorded provenance the adopting leg cannot honestly re-derive
    refuses here instead of silently moving its recorded snapshot, un-staling
    its item pin, and rebinding the close's stale-excluded cohort (Opus C1).

    Returns
    -------
    str | None
        ``None`` when the prepared candidate is the head verbatim modulo
        :data:`ORPHAN_ADOPTION_ALLOWED_DIFF`, else the typed drift reason.
    """

    offending = _disallowed_diff_paths(
        prepared, head, ORPHAN_ADOPTION_ALLOWED_DIFF | ORPHAN_ROTATABLE_VECTOR_AXES
    )
    if offending:
        shown = ", ".join(offending[:5])
        suffix = "" if len(offending) <= 5 else f" (+{len(offending) - 5} more)"
        return (
            "adoption re-append drifted beyond re-citation after "
            f"prepare_model: {shown}{suffix}"
        )
    return None

#: Closed census classes for the sweep disclosure (no silent outcomes).
TERMINAL_REVIEW_CENSUS_CLASSES: Final[tuple[str, ...]] = (
    # A live checker session (or a rotated-identity re-adjudication) accepted
    # the recorded terminal claim; the gate record is the durable adjudication.
    "confirmed",
    # The review refused or could not verify the claim; the head was re-marked
    # review-held and its outstanding grant may now bind.
    "review_held",
    # A prior SWEEP review already adjudicated this head's terminal work
    # generation (a durable ``terminal-review-adjudicated`` disclosure) and a
    # current-identity terminal-disposition gate still covers it; nothing new
    # to buy.  The fresh-mint gate alone never qualifies -- every review-free
    # settled head has one.
    "already_adjudicated",
    # The head's terminal was minted by the driver from infrastructure facts
    # (effort exhaustion, session crash, protocol violation, ...): no typed
    # author recommendation exists, so the terminal-disposition lane has no
    # claim to adjudicate.  Fabricating one is forbidden; the head is
    # disclosed and stays operator property.
    "no_consumer",
    # The head DOES assert a typed terminal claim (the exact
    # ``TERMINAL_UNADJUDICATED_DECISION`` producer wording), but its work
    # generation has no finalized artifact authority to rehydrate the claim
    # from -- the raw recommendation survives only in unattested work-dir
    # files, and adjudicating producer-forgeable bytes is forbidden.
    # Disclosed distinctly from ``no_consumer`` so the honest remainder is
    # countable; releasing it requires a staged-transaction rehydration
    # authority that does not exist yet.
    "claim_unrehydratable",
    # The review refused the claim but the review-held supersession itself
    # refused to record (a reducer refusal is one head's fact, never the
    # sweep's); the head stays eligible next leg and the refusal text is
    # disclosed on the census event.
    "revision_refused",
    # The member is not in the active intake snapshot (dormant history).
    "dormant",
    # A live valve row withholds the member, or the checker lane failed and
    # parked it this leg (the lane's own brake).
    "parked",
    # Eligible but beyond TERMINAL_REVIEW_BACKLOG_PER_LEG this leg.
    "beyond_bound",
)


def unreviewed_failed_terminal_heads(
    raw_heads: Mapping[str, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return failed terminal heads lacking a human-review adjudication.

    Eligibility is exactly the grant lane's ``unreviewed-terminal`` hold
    predicate read from the other side: a raw head whose status kind is
    ``failed`` and whose ``status.human_review.required`` is not true.  Order
    is deterministic (stable ID) so the per-leg bound is reproducible.

    Parameters
    ----------
    raw_heads:
        The reducer's raw settlement authority
        (:attr:`~menagerie.crawler.reducer.CanonicalReducer.raw_head_view`).

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Eligible heads in stable-ID order (shared references; read-only).
    """

    eligible = [
        head
        for head in raw_heads.values()
        if isinstance(head.get("status"), Mapping)
        and head["status"].get("kind") == "failed"
        and head["status"].get("human_review", {}).get("required") is not True
    ]
    eligible.sort(key=lambda head: str(head.get("stable_id")))
    return tuple(eligible)


def current_identity_terminal_gate(
    gate_records: Any,
    *,
    stable_id: str,
    work_id: str,
    checker_identity: Optional[Any],
) -> Optional[JsonObject]:
    """Return the current-identity terminal gate already covering one head.

    A scheduling pre-filter, never a verdict authority: it grants nothing and
    can only conclude "this head needs no new session this leg".  The verdict
    itself was applied when that gate landed (fresh path) or when a sweep
    consumed it; re-running the sweep over it every leg would re-record reuse
    events for a settled adjudication.  The authoritative reuse decision for
    anything this filter admits remains
    :func:`menagerie.crawler.gate_reuse.reusable_terminal_gate`, which
    additionally re-verifies the exact artifact binding.

    Parameters
    ----------
    gate_records:
        Persisted gate rows, newest last (``CanonicalReducer.gate_records``).
    stable_id, work_id:
        The head's identity and its recorded terminal work generation.
    checker_identity:
        The live lane adjudication identity
        (:func:`menagerie.crawler.gate_reuse.lane_adjudication_identity`);
        ``None`` disables the filter (fail-open to a live review decision,
        which itself fails closed).

    Returns
    -------
    dict[str, Any] | None
        The newest matching gate row, or ``None`` when a live review (or
        reuse) decision is required.
    """

    if checker_identity is None:
        return None
    from menagerie.crawler.gate_reuse import _gate_adjudication_matches

    for gate in reversed(list(gate_records)):
        if gate.get("gate_kind") != "terminal_disposition":
            continue
        items = gate.get("items")
        if not isinstance(items, list):
            continue
        mine = [
            item
            for item in items
            if isinstance(item, Mapping)
            and item.get("stable_id") == stable_id
            and item.get("work_id") == work_id
        ]
        if not mine:
            continue
        # Newest row containing this head decides, exactly like gate reuse:
        # an identity mismatch on the deciding row refuses to a live review
        # rather than falling back into superseded history.
        if not _gate_adjudication_matches(
            gate, kind="terminal_disposition", checker=checker_identity
        ):
            return None
        return dict(gate)
    return None


def terminal_review_hold_candidate(
    head: Mapping[str, Any],
    *,
    requested_at: str,
    producer_run_id: str,
    machine_id: str,
) -> JsonObject:
    """Build the review-held supersession candidate for one refused head.

    The candidate is the head verbatim -- every recorded CONTENT fact, gate
    binding, budget line, and authorship provenance field survives -- with
    exactly the containment ladder's marking applied: ``status.human_review``
    flips to the required review-held block, and the supersession parentage
    binds the exact head revision.  Ledger-assigned fields are stripped for
    re-assignment.  Keeping the ORIGINAL status code and stage is what lets
    the outstanding grant bind
    (:func:`~menagerie.crawler.driver_admission._grant_stage_binds_terminal`);
    the reviewing checker's identity and verdict live on the appended gate
    record, not on this revision.

    The CITATION is the one deliberate exception to verbatim: position is
    provenance, so the re-append must name the run (and machine) that appends
    it.  Copying the superseded head's stale ``producer_run_id`` minted rows
    whose in-window citation named a run that never joined the open
    generation, and the partition proof refused the whole leg
    (``PartitionProofError``, ordinary leg 2, 2026-08-27).  Both stamps are
    REQUIRED keywords precisely so no caller -- production or out-of-band --
    can mint a hold without naming the appending run.

    Parameters
    ----------
    head:
        Raw ledger head being re-marked (never mutated).
    requested_at:
        Driver clock reading for ``human_review.requested_at``.
    producer_run_id, machine_id:
        The APPENDING run's identity (``config.run_id`` / ``config.machine_id``
        of the leg performing the re-append).

    Returns
    -------
    dict[str, Any]
        Append-ready model revision candidate.
    """

    candidate: JsonObject = {
        key: deepcopy(value)
        for key, value in head.items()
        if key not in _LEDGER_ASSIGNED_FIELDS
    }
    candidate["parent_revision"] = head["record_revision"]
    status = candidate.get("status")
    if not isinstance(status, dict):
        raise ValueError("terminal review candidate requires a status block")
    status["supersedes_revision"] = head["record_revision"]
    status["human_review"] = {
        "required": True,
        "reason": None,
        "queue": _HOLD_QUEUE,
        "requested_at": requested_at,
    }
    provenance = candidate.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("terminal review candidate requires a provenance block")
    provenance["producer_run_id"] = producer_run_id
    provenance["machine_id"] = machine_id
    return candidate
