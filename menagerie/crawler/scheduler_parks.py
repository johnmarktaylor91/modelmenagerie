"""Scheduler-fault quarantine parking: one member's failure is THAT member's.

BL-8 (a)/(b): the pipelined scheduler used to have exactly one response to a
per-member fault that escaped the seams' own containment arms -- kill the whole
leg (`DriverIntegrationError("scheduler made no durable routing progress")`,
or an exception propagating out of a batch seam). At 20k-member legs that
converts one member's unroutable refusal into the loss of everyone else's
progress and a full-restart crash loop.

This module is the durable valve for that outcome class, mirroring the sibling
valves (:mod:`menagerie.crawler.checker_lane`,
:mod:`menagerie.crawler.host_capacity`):

* **It refuses to mint canonical terminals.** A quarantined member holds no
  verdict: it is withheld from the rest of this leg, recorded here, and
  requeued on the next run. No model record asserts a failure the model never
  earned -- the fault may be the scheduler's own routing, not the model.
* **It is accounted, never silent.** The terminal partition dereferences these
  rows exactly like the admission-valve rows (every in-scope model terminal
  XOR machine-deferral-claimed), so a quarantined member can never vanish.
* **It never re-claims a model by itself.** The row has no
  ``_deferral_row_fact_still_holds`` arm. A v2 row earns a BOUNDED number of
  fresh-leg retries (:data:`SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT`, counted
  since the last budget-restoring release) and then holds until an operator
  release: a bare void clears the row but restores no budget, a remedied void
  (fixing sha + RED-pinned test node + passing canary, the DX-69 shape)
  restores it, and a disclosed machine drain (W12) resets the counter and
  binds next-leg admission to the drain's fixed-in tree sha. Quarantine is
  bounded fault isolation, never a standing verdict and never an unbounded
  retry loop.

Campaign-level signals (pauses, shutdowns, retryable infrastructure, provider
backoff) never reach this valve: those stay loud and campaign-scoped in the
driver.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    row_content_address,
    stamp_row_digest,
)
from menagerie.crawler.completion_line import RelativeImportUnbindableError
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import read_with_tail_recovery

SCHEDULER_QUARANTINE_SCHEMA_VERSION_V1 = "menagerie.crawler.scheduler-quarantine.v1"
"""Historical row: unconditional next-run release and no bounded lifecycle."""

SCHEDULER_QUARANTINE_SCHEMA_VERSION_V2 = "menagerie.crawler.scheduler-quarantine.v2"
"""Historical row: bounded retry, no typed settlement_class tripwire."""

SCHEDULER_QUARANTINE_SCHEMA_VERSION = "menagerie.crawler.scheduler-quarantine.v3"
"""Current row: bounded retry + the REQUIRED redundant ``settlement_class``
tripwire validated against the clause-1 kind registry (RULING W7 clause 11)."""

SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION = (
    "menagerie.crawler.scheduler-quarantine-void.v1"
)
"""BARE operator void: clears the live row, restores NO retry budget."""

SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2 = (
    "menagerie.crawler.scheduler-quarantine-void.v2"
)
"""REMEDIED operator void (W12/SD-6): carries the complete DX-69 remedy shape
(fixing_sha + RED-pinned test node + passing canary) and is budget-restoring."""

SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION = (
    "menagerie.crawler.scheduler-quarantine-drain.v1"
)
"""Disclosed machine drain (W12/RE-11): appendable only after the bounded
retries are exhausted, requires the fixed-in tree sha, resets the counter, and
pins next-leg admission to that sha."""

SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSIONS = frozenset(
    {
        SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION,
        SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2,
    }
)
"""Every readable void schema (append-only history stays valid)."""

SCHEDULER_QUARANTINE_RELEASE_SCHEMA_VERSIONS = frozenset(
    {
        *SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSIONS,
        SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION,
    }
)
"""Every row kind that removes a park from the live set."""

SCHEDULER_QUARANTINE_SCHEMA_VERSIONS = frozenset(
    {
        SCHEDULER_QUARANTINE_SCHEMA_VERSION_V1,
        SCHEDULER_QUARANTINE_SCHEMA_VERSION_V2,
        SCHEDULER_QUARANTINE_SCHEMA_VERSION,
    }
)
"""Every readable scheduler-quarantine schema (append-only history stays valid)."""

SCHEDULER_QUARANTINE_PARK_EVENT_DISPOSITION = "scheduler-member-quarantined"
"""Operational-event disposition that CHARGES the recidivism budget."""

SCHEDULER_QUARANTINE_REUSED_DISPOSITION = "scheduler-member-quarantine-reused"
"""Disclosure for an idempotent re-park observation: charges NOTHING.

The 28-events-vs-22-rows class: the row append is idempotent per work
generation, but the event append fired unconditionally with a fresh clock,
so observations drifted above rows and the budget counter (which counts
events) charged the member for parks that minted no row."""

SCHEDULER_DRAIN_CENSUS_CLASSES: tuple[str, ...] = (
    "retry-eligible",
    "held-awaiting-operator-drain",
    "machine-drained",
    "remedied-void-released",
)
"""Closed classes for :func:`scheduler_drain_census` (waiver doctrine part 4)."""

SCHEDULER_QUARANTINE_DISPOSITION = "deferred:scheduler-fault-quarantined"
"""Greppable disposition for a member quarantined by scheduler fault isolation."""

SCHEDULER_QUARANTINE_REMEDY_V1 = "scheduler-fault-quarantine"
"""Historical remedy token promising unconditional next-run release."""

SCHEDULER_QUARANTINE_REMEDY = "scheduler-fault-operator-drain"
"""Current remedy: one bounded retry, then list and explicitly drain the park."""

SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT = 2
"""Parks of one work generation allowed before its unchanged row holds."""

SCHEDULER_QUARANTINE_RELATIVE_PATH = Path("intake-extensions") / "scheduler-quarantines.jsonl"
"""Ledger location below the canonical records root."""

SCHEDULER_QUARANTINE_REASONS = frozenset(
    {
        "member-fault",
        "non-model-fault",
        "award-unroutable",
        "no-durable-routing-progress",
        "invalidation-nonconvergent",
        "terminal-proof-generation-exhausted",
        "award-blocked:spent-chain",
    }
)
"""Closed vocabulary for WHY a member was quarantined.

``member-fault`` is a per-member exception the seam's own containment arms did
not catch (the blast-radius census class). ``non-model-fault`` is a host- or
machinery-owned execution failure (BL-12 e): the attempt row carries the typed
evidence and no model terminal may be minted. ``award-unroutable`` is the
recorded-attempt award dead end -- the gate round accepted but the recorded
triage attempts can never satisfy the execution policy under the unchanged
identity, so no routing step exists. ``no-durable-routing-progress`` is the
scheduler's (or the metadata gate round's) no-progress tripwire firing on this
member: quarantining it is what lets the tripwire keep protecting the leg
without killing it. ``invalidation-nonconvergent`` is W9's N-repeat fuse: the
third same-key identity-stale cache retirement for one member proves the
invalidation is not converging, so the member parks instead of re-buying the
same authorship loop (R16); the ordinal is replay-derived from the operational
ledger's own retirement rows on every leg.
``terminal-proof-generation-exhausted`` is AT-5's pre-spend retirement: the
member's decisive attempts span environment generations and no ONE generation
carries a complete satisfying set, so no coherent terminal proof exists --
the member parks BEFORE any publication (the m12646 arming refusal, moved
pre-spend) and re-execution under the current generation may heal it.
``award-blocked:spent-chain`` is AT-4's post-spend containment: the reducer's
typed append refusal manifested only AFTER the artifact chain was spent (the
pre-flight passed against staged authority), so the member parks with its
seam disclosure instead of minting a failed terminal under a published final
event (m12646's ``published_with_failure_head`` incoherence).
"""

#: CLOSED exception types the env-turn member-fault catch-all reads as the
#: crawler's OWN machinery (item 9 / author-uplift R-7, R10 class).
#: ``RelativeImportUnbindableError``: the static import-closure derivation
#: cannot bind a vendored package's relative import (R6b m243) -- raised
#: BEFORE any execution row exists, so the ``non-model-fault`` routing marker
#: is lawful under RULING W7 clause 3 (the same custody shape as the
#: sandbox-unavailable park). The corpus-index ``LedgerConflictError`` class is
#: deliberately NOT here: its two raise sites are classified where they fire
#: (``_author_lane_failure`` -> machinery-owned park at accept time;
#: ``_promote_award_corpus`` -> post-commit disclosure), and a member reaching
#: a catch-all with published artifact rows can never carry a routing marker.
#: Widening this tuple is a reviewed change with its own RED test.
MACHINERY_FAULT_QUARANTINE_TYPES: tuple[type[BaseException], ...] = (
    RelativeImportUnbindableError,
)


def member_quarantine_reason(fault: BaseException) -> str:
    """Return the closed quarantine reason for one escaped per-member fault.

    Parameters
    ----------
    fault:
        The exception a member-containment arm caught.

    Returns
    -------
    str
        ``non-model-fault`` for the closed machinery types (a routing marker:
        no verdict about the model may be minted), ``member-fault`` otherwise
        (the ambiguous blast-radius class, resolved per row by custody
        corroboration as before).
    """

    if isinstance(fault, MACHINERY_FAULT_QUARANTINE_TYPES):
        return "non-model-fault"
    return "member-fault"


ROUTING_MARKER_REASONS = frozenset(
    {
        "non-model-fault",
        "no-durable-routing-progress",
        "invalidation-nonconvergent",
    }
)
"""Reasons whose rows are ROUTING MARKERS (RULING W7 clause 1, PD-1 kernel).

A routing marker states that the LEG's machinery, not the model, ended this
member's turn: it defers work, renders no verdict, settles nothing, and (under
proof schema v4) is EXCLUDED from the head-contradiction predicate by closed
membership -- when, and only when, the member's custody window is replay-silent
(clause 3's corroboration conjunct)."""

VERDICT_BEARING_QUARANTINE_REASONS = frozenset(
    {
        "award-blocked:spent-chain",
        "terminal-proof-generation-exhausted",
    }
)
"""Reasons whose rows GUARD CUSTODY (RULING W7 clause 1; OPEN-2 default).

The narrow, fail-closed census: a contested code may not enter the privileged
exclusion set. Widening later is a lawful amendment; narrowing later is a
laundering post-mortem. These rows keep every refusal, retyped
``head-beside-live-custody-withholding``."""

AMBIGUOUS_QUARANTINE_REASONS = frozenset(
    {
        "member-fault",
        "award-unroutable",
    }
)
"""Reasons resolved PER ROW by clause 3's custody corroboration (O R-2).

Custody-silent rows corroborate as routing; custody-active rows are
verdict-bearing, fail-closed. Their static ``settlement_class`` is the
fail-closed ``verdict-bearing`` -- no row field may claim the routing side."""

SETTLEMENT_CLASS_ROUTING_MARKER = "routing-marker"
SETTLEMENT_CLASS_VERDICT_BEARING = "verdict-bearing"

SCHEDULER_QUARANTINE_SETTLEMENT_CLASSES = frozenset(
    {SETTLEMENT_CLASS_ROUTING_MARKER, SETTLEMENT_CLASS_VERDICT_BEARING}
)
"""Closed vocabulary for the v3 row's redundant settlement tripwire."""


def quarantine_reason_kind(reason_code: str) -> str:
    """Return one reason's closed semantic kind, refusing the unclassified.

    The future-kind fail-closed rule (RULING W7 clause 1, unanimous): kind is
    a total function of the closed registry -- ``routing``, ``verdict-bearing``,
    or ``ambiguous`` -- and an unknown or unclassified reason REFUSES, at build
    and at read, with NO default arm. Adding a reason to
    :data:`SCHEDULER_QUARANTINE_REASONS` without classifying it here is a RED
    census test, never a silent inheritance.
    """

    if reason_code in ROUTING_MARKER_REASONS:
        return "routing"
    if reason_code in VERDICT_BEARING_QUARANTINE_REASONS:
        return "verdict-bearing"
    if reason_code in AMBIGUOUS_QUARANTINE_REASONS:
        return "ambiguous"
    raise SchedulerQuarantineError(
        f"scheduler quarantine reason {reason_code!r} has no classified semantic "
        "kind: an unclassified reason refuses fail-closed (RULING W7 clause 1)"
    )


def expected_settlement_class(reason_code: str) -> str:
    """Return the ONE lawful ``settlement_class`` for a reason (clause 11).

    Ambiguous reasons carry the fail-closed static class (verdict-bearing);
    only clause 3's replay-derived custody corroboration -- never a row field
    -- may treat such a row as routing at close.
    """

    kind = quarantine_reason_kind(reason_code)
    if kind == "routing":
        return SETTLEMENT_CLASS_ROUTING_MARKER
    return SETTLEMENT_CLASS_VERDICT_BEARING


def _assert_reason_census_partition() -> None:
    """Refuse AT BUILD when the kind census stops partitioning the reasons."""

    union = (
        ROUTING_MARKER_REASONS
        | VERDICT_BEARING_QUARANTINE_REASONS
        | AMBIGUOUS_QUARANTINE_REASONS
    )
    overlaps = (
        (ROUTING_MARKER_REASONS & VERDICT_BEARING_QUARANTINE_REASONS)
        | (ROUTING_MARKER_REASONS & AMBIGUOUS_QUARANTINE_REASONS)
        | (VERDICT_BEARING_QUARANTINE_REASONS & AMBIGUOUS_QUARANTINE_REASONS)
    )
    if union != SCHEDULER_QUARANTINE_REASONS or overlaps:
        raise SchedulerQuarantineError(
            "the quarantine reason kind census no longer partitions "
            "SCHEDULER_QUARANTINE_REASONS exactly "
            f"(unclassified: {sorted(SCHEDULER_QUARANTINE_REASONS - union)}, "
            f"foreign: {sorted(union - SCHEDULER_QUARANTINE_REASONS)}, "
            f"overlapping: {sorted(overlaps)}): classify every reason in the "
            "same reviewed change (RULING W7 clause 1)"
        )


SCHEDULER_QUARANTINE_SITES = frozenset(
    {
        "author-wave",
        "admission",
        "run-anchors",
        "env-turn",
        "gate-round",
        "award",
        "scheduler-round",
    }
)
"""Closed vocabulary for WHERE the quarantine decision was made."""

SCHEDULER_QUARANTINE_EVIDENCE_MAX_CHARS = 1_500
"""Head-bounded fault evidence, matching the sibling valves' STDIO tail bound."""


class SchedulerQuarantineError(ValueError):
    """A scheduler-quarantine row is malformed or does not record a fault."""


# Refuse at import (the earliest "build" there is) when the kind census stops
# partitioning the closed reason vocabulary; the set-equality census test is
# the reviewed-change tripwire, this is the runtime one (RULING W7 clause 1).
_assert_reason_census_partition()


def _park_event_key(event: Mapping[str, Any]) -> Optional[tuple[str, str, str]]:
    """Return ``(stable_id, work_id, created_at)`` for one charging park event.

    Only the exact :data:`SCHEDULER_QUARANTINE_PARK_EVENT_DISPOSITION` charges
    the budget; the ``-reused`` disclosure and every other disposition charge
    nothing (PD-11's void-counting composition).
    """

    if not isinstance(event, Mapping):
        return None
    details = event.get("details")
    if not isinstance(details, Mapping):
        return None
    if details.get("disposition") != SCHEDULER_QUARANTINE_PARK_EVENT_DISPOSITION:
        return None
    return (
        str(details.get("stable_id")),
        str(details.get("work_id")),
        str(event.get("created_at", "")),
    )


def budget_restoring_boundaries(
    ledger_rows: Sequence[Mapping[str, Any]],
    operational_events: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], str]:
    """Derive each work generation's LAST budget-restoring release instant.

    PD-11's closed set of budget-restoring events, recomputed from the ledger
    on every read with zero producer input (waiver doctrine parts 2 and 3):

    - a REMEDIED void (schema v2): always restores -- its validator already
      refused any incomplete remedy at build and at read;
    - a MACHINE DRAIN: restores ONLY when, at its position, the generation's
      charged park events since the previous boundary had reached
      :data:`SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT`. An ineligible drain row
      -- however it got appended -- is not budget-restoring, so bypassing the
      tool's precondition forges nothing.

    A bare v1 void, a restart, a new leg, and a tree change without declared
    remedy restore nothing (S-C5).

    Parameters
    ----------
    ledger_rows:
        Every validated quarantine ledger row, parks and releases alike
        (:func:`load_scheduler_quarantine_ledger`).
    operational_events:
        Durable campaign-health events used to count charged parks.

    Returns
    -------
    dict[tuple[str, str], str]
        ``(stable_id, work_id) -> created_at`` of the latest restoring event.
    """

    park_owner: dict[str, tuple[str, str]] = {}
    for row in ledger_rows:
        if row.get("schema_version") in SCHEDULER_QUARANTINE_SCHEMA_VERSIONS:
            park_owner[str(row["row_sha256"])] = (
                str(row["stable_id"]),
                str(row["work_id"]),
            )

    charged: dict[tuple[str, str], list[str]] = {}
    for event in operational_events:
        key = _park_event_key(event)
        if key is None:
            continue
        charged.setdefault((key[0], key[1]), []).append(key[2])
    for stamps in charged.values():
        stamps.sort()

    candidates: list[tuple[str, str, tuple[str, str]]] = []
    for row in ledger_rows:
        schema = row.get("schema_version")
        if schema not in (
            SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2,
            SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION,
        ):
            continue
        owner = park_owner.get(str(row.get("void_of")))
        if owner is None:
            # Fail closed: a release whose park this ledger cannot resolve
            # restores nothing.
            continue
        candidates.append((str(row.get("created_at", "")), str(schema), owner))
    candidates.sort()

    boundaries: dict[tuple[str, str], str] = {}
    for created_at, schema, owner in candidates:
        if schema == SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2:
            boundaries[owner] = created_at
            continue
        floor = boundaries.get(owner, "")
        charged_before = [
            stamp for stamp in charged.get(owner, ()) if floor < stamp <= created_at
        ]
        if len(charged_before) >= SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT:
            boundaries[owner] = created_at
    return boundaries


def scheduler_quarantine_released(
    row: Mapping[str, Any],
    operational_events: Sequence[Mapping[str, Any]],
    *,
    ledger_rows: Sequence[Mapping[str, Any]] = (),
) -> bool:
    """Return whether a quarantine row releases toward a bounded retry.

    Current v2 rows release while the exact work generation has recorded fewer
    than :data:`SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT` charged park events
    SINCE THE LAST BUDGET-RESTORING EVENT (a remedied void or an eligible
    machine drain -- :func:`budget_restoring_boundaries`). Counting all of
    history was the m12646 livelock: 8 parks / 8 voids and the operator's
    drain bought zero retries. Legacy v1 rows retain their historical
    unconditional next-run release.

    Parameters
    ----------
    row:
        Validated scheduler-quarantine park row.
    operational_events:
        Durable campaign-health events used to count exact-generation parks.
    ledger_rows:
        Every quarantine ledger row (parks and releases), used to derive the
        budget boundaries. Empty means no boundary exists and every charged
        park in history counts (the conservative, holding direction).

    Returns
    -------
    bool
        Whether the row is released toward attempting this run.
    """

    if row.get("schema_version") == SCHEDULER_QUARANTINE_SCHEMA_VERSION_V1:
        return True
    stable_id = str(row.get("stable_id"))
    work_id = str(row.get("work_id"))
    boundary = budget_restoring_boundaries(ledger_rows, operational_events).get(
        (stable_id, work_id), ""
    )
    parks = 0
    for event in operational_events:
        key = _park_event_key(event)
        if key is None or key[0] != stable_id or key[1] != work_id:
            continue
        if key[2] > boundary:
            parks += 1
    return parks < SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT


def scheduler_drain_census(
    ledger_rows: Sequence[Mapping[str, Any]],
    operational_events: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Segregated release census over the quarantine ledger (doctrine part 4).

    Live rows split into ``retry-eligible`` (budget remains) and
    ``held-awaiting-operator-drain`` (the operator-owed quarantine debt the
    close must disclose); historical restoring releases are disclosed as
    ``machine-drained`` and ``remedied-void-released``, never mixed with rows
    that released on their own budget.

    Returns
    -------
    dict[str, tuple[str, ...]]
        :data:`SCHEDULER_DRAIN_CENSUS_CLASSES` -> sorted ``stable_id::work_id``
        member keys.
    """

    census: dict[str, set[str]] = {name: set() for name in SCHEDULER_DRAIN_CENSUS_CLASSES}
    for row in current_scheduler_quarantine_rows(ledger_rows):
        key = f"{row['stable_id']}::{row['work_id']}"
        if scheduler_quarantine_released(row, operational_events, ledger_rows=ledger_rows):
            census["retry-eligible"].add(key)
        else:
            census["held-awaiting-operator-drain"].add(key)

    park_owner = {
        str(row["row_sha256"]): f"{row['stable_id']}::{row['work_id']}"
        for row in ledger_rows
        if row.get("schema_version") in SCHEDULER_QUARANTINE_SCHEMA_VERSIONS
    }
    boundaries = budget_restoring_boundaries(ledger_rows, operational_events)
    restored = {f"{owner[0]}::{owner[1]}" for owner in boundaries}
    for release in ledger_rows:
        owner = park_owner.get(str(release.get("void_of", "")))
        if owner is None or owner not in restored:
            continue
        if release.get("schema_version") == SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION:
            census["machine-drained"].add(owner)
        elif release.get("schema_version") == SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2:
            census["remedied-void-released"].add(owner)
    return {name: tuple(sorted(members)) for name, members in census.items()}


def unsatisfied_drain_tree_pins(
    ledger_rows: Sequence[Mapping[str, Any]], repo_root: Path
) -> tuple[JsonObject, ...]:
    """Return every drain whose fixed-in sha the running tree does not contain.

    W12: a machine drain binds the NEXT leg to the tree that carries the
    declared fix -- admission refuses while ``fixed_in_tree_sha`` is not an
    ancestor of the running ``HEAD``. Fails CLOSED: outside a git repository
    (or on any git failure) every drain pin reads unsatisfied, because a
    drain declared a tree obligation nothing can verify.

    Parameters
    ----------
    ledger_rows:
        Every validated quarantine ledger row.
    repo_root:
        Running repository root.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One record per unsatisfied drain:
        ``{"fixed_in_tree_sha", "void_of", "reason", "created_at"}``.
    """

    import subprocess

    verdicts: dict[str, bool] = {}

    def _contains(sha: str) -> bool:
        if sha not in verdicts:
            try:
                completed = subprocess.run(
                    ("git", "-C", str(repo_root), "merge-base", "--is-ancestor", sha, "HEAD"),
                    capture_output=True,
                    timeout=10,
                )
                verdicts[sha] = completed.returncode == 0
            except (OSError, subprocess.SubprocessError):
                verdicts[sha] = False
        return verdicts[sha]

    unsatisfied: list[JsonObject] = []
    for row in ledger_rows:
        if row.get("schema_version") != SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION:
            continue
        sha = str(row.get("fixed_in_tree_sha", ""))
        if not _contains(sha):
            unsatisfied.append(
                {
                    "fixed_in_tree_sha": sha,
                    "void_of": str(row.get("void_of", "")),
                    "reason": str(row.get("reason", "")),
                    "created_at": str(row.get("created_at", "")),
                }
            )
    return tuple(unsatisfied)


def scheduler_quarantine_path(records_root: Path) -> Path:
    """Return the canonical scheduler-quarantine ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only scheduler-quarantine JSONL path.
    """

    return records_root / SCHEDULER_QUARANTINE_RELATIVE_PATH


def build_scheduler_quarantine_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    site: str,
    reason_code: str,
    fault_evidence: str,
    exception_type: Optional[str] = None,
) -> JsonObject:
    """Assemble one complete, self-describing scheduler-quarantine row.

    The row states WHERE the leg isolated the member, WHY, and the
    head-bounded fault evidence, so a later reviewer can re-derive the
    decision without the code that made it.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the active scheduled work generation.
    campaign_id, run_id, machine_id:
        Campaign and host provenance for the quarantine decision.
    created_at:
        UTC timestamp of the decision.
    site:
        One of :data:`SCHEDULER_QUARANTINE_SITES`.
    reason_code:
        One of :data:`SCHEDULER_QUARANTINE_REASONS`.
    fault_evidence:
        Diagnosis text from the isolating decision; head-truncated.
    exception_type:
        Fully qualified exception type when an exception drove the decision.

    Returns
    -------
    dict[str, Any]
        Content-addressed quarantine row.

    Raises
    ------
    SchedulerQuarantineError
        If the site or reason is outside its closed vocabulary.
    """

    if not campaign_id or campaign_id == str(None):
        # T2-12 parity with the sibling valves: a stringified absence must
        # refuse at BUILD time.
        raise ValueError("campaign_id must name a real campaign, never a stringified absence")
    if site not in SCHEDULER_QUARANTINE_SITES:
        raise SchedulerQuarantineError(
            f"scheduler quarantine names unknown site {site!r}"
        )
    if reason_code not in SCHEDULER_QUARANTINE_REASONS:
        raise SchedulerQuarantineError(
            f"scheduler quarantine names unknown reason {reason_code!r}"
        )
    evidence = fault_evidence.strip()
    if len(evidence) > SCHEDULER_QUARANTINE_EVIDENCE_MAX_CHARS:
        evidence = evidence[:SCHEDULER_QUARANTINE_EVIDENCE_MAX_CHARS] + "...<truncated>"
    quarantine: JsonObject = {
        "site": site,
        "reason_code": reason_code,
        "fault_evidence": evidence,
        "exception_type": exception_type,
        "remedy": SCHEDULER_QUARANTINE_REMEDY,
        # RULING W7 clause 11: a REDUNDANT typed tripwire, derived from the
        # clause-1 registry at build and re-validated against it at read. The
        # registry stays authoritative; this field can never steer a verdict,
        # it can only refuse a row whose prose and vocabulary disagree.
        "settlement_class": expected_settlement_class(reason_code),
    }
    row: JsonObject = {
        "schema_version": SCHEDULER_QUARANTINE_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": SCHEDULER_QUARANTINE_DISPOSITION,
        "scheduler_quarantine": quarantine,
        "recheck_hint": (
            f"{SCHEDULER_QUARANTINE_REMEDY}: this member's fault was isolated so the "
            "leg could continue; no verdict was rendered. One bounded fresh-leg retry "
            "is automatic; repeated failure then holds this work generation until an "
            "operator drain is appended with tools/scheduler_quarantines.py"
        ),
    }
    row = stamp_row_digest(row)
    return row


def validate_scheduler_quarantine_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one scheduler-quarantine row and return its canonical copy.

    Parameters
    ----------
    row:
        Candidate quarantine row.

    Returns
    -------
    dict[str, Any]
        Validated row.

    Raises
    ------
    SchedulerQuarantineError
        If a required field is missing, mistyped, or the digest disagrees.
    """

    required = (
        "schema_version",
        "stable_id",
        "work_id",
        "name",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
        "disposition",
        "scheduler_quarantine",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise SchedulerQuarantineError(
            f"scheduler quarantine row is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] not in SCHEDULER_QUARANTINE_SCHEMA_VERSIONS:
        raise SchedulerQuarantineError(
            f"scheduler quarantine row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] != SCHEDULER_QUARANTINE_DISPOSITION:
        raise SchedulerQuarantineError(
            f"scheduler quarantine row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise SchedulerQuarantineError(f"scheduler quarantine row has an empty {field}")
    quarantine = row["scheduler_quarantine"]
    if not isinstance(quarantine, Mapping):
        raise SchedulerQuarantineError("scheduler quarantine row must describe the fault")
    if quarantine.get("site") not in SCHEDULER_QUARANTINE_SITES:
        raise SchedulerQuarantineError(
            f"scheduler quarantine row names unknown site {quarantine.get('site')!r}"
        )
    if quarantine.get("reason_code") not in SCHEDULER_QUARANTINE_REASONS:
        raise SchedulerQuarantineError(
            f"scheduler quarantine row names unknown reason {quarantine.get('reason_code')!r}"
        )
    expected_remedy = (
        SCHEDULER_QUARANTINE_REMEDY_V1
        if row["schema_version"] == SCHEDULER_QUARANTINE_SCHEMA_VERSION_V1
        else SCHEDULER_QUARANTINE_REMEDY
    )
    if quarantine.get("remedy") != expected_remedy:
        raise SchedulerQuarantineError(
            f"scheduler quarantine row names unknown remedy {quarantine.get('remedy')!r}"
        )
    if row["schema_version"] == SCHEDULER_QUARANTINE_SCHEMA_VERSION:
        declared = quarantine.get("settlement_class")
        lawful = expected_settlement_class(str(quarantine.get("reason_code")))
        if declared != lawful:
            raise SchedulerQuarantineError(
                f"scheduler quarantine row declares settlement_class {declared!r} "
                f"but the clause-1 registry classifies {quarantine.get('reason_code')!r} "
                f"as {lawful!r}: no row field is authoritative for its own kind "
                "(RULING W7 clause 11)"
            )
    elif "settlement_class" in quarantine:
        raise SchedulerQuarantineError(
            "scheduler quarantine row smuggles a settlement_class into a "
            "pre-v3 schema: the field exists only where the validator "
            "cross-checks it (RULING W7 clause 11)"
        )
    evidence = quarantine.get("fault_evidence")
    if not isinstance(evidence, str) or not evidence:
        raise SchedulerQuarantineError(
            "scheduler quarantine row must carry non-empty fault evidence"
        )
    exception_type = quarantine.get("exception_type")
    if exception_type is not None and (not isinstance(exception_type, str) or not exception_type):
        raise SchedulerQuarantineError(
            "scheduler quarantine row exception_type must be a non-empty string or null"
        )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise SchedulerQuarantineError(
            f"scheduler park row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise SchedulerQuarantineError(
            "scheduler quarantine row digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def build_scheduler_quarantine_void(
    *, void_of: str, reason: str, voided_by: str, created_at: str
) -> JsonObject:
    """Build one append-only operator drain for a live scheduler park.

    Parameters
    ----------
    void_of:
        Exact ``row_sha256`` of the live quarantine row.
    reason:
        Operator justification for retrying the unchanged work generation.
    voided_by:
        Operator identity making the decision.
    created_at:
        UTC timestamp of the decision.

    Returns
    -------
    dict[str, Any]
        Validated content-addressed void row.
    """

    row: JsonObject = {
        "schema_version": SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION,
        "void_of": void_of,
        "reason": reason,
        "voided_by": voided_by,
        "created_at": created_at,
    }
    row = stamp_row_digest(row)
    return validate_scheduler_quarantine_void(row)


def build_scheduler_quarantine_remedied_void(
    *,
    void_of: str,
    reason: str,
    voided_by: str,
    created_at: str,
    fixing_sha: str,
    test_node: str,
    canary: str,
) -> JsonObject:
    """Build one BUDGET-RESTORING remedied void (W12/SD-6, the DX-69 shape).

    Unlike the bare void, this one declares the actual repair: the fixing
    commit, the RED-pinned test node that reproduces the fault, and the
    passing canary evidence. Only this complete shape restores the work
    generation's retry budget -- a bare void, a restart, or a tree change
    alone never does (S-C5).

    Parameters
    ----------
    void_of:
        Exact ``row_sha256`` of the live quarantine row.
    reason:
        Operator justification.
    voided_by:
        Operator identity making the decision.
    created_at:
        UTC timestamp of the decision.
    fixing_sha, test_node, canary:
        The declared remedy: fixing commit, RED-pinned test node, and the
        passing-canary evidence. All three are required and non-empty.

    Returns
    -------
    dict[str, Any]
        Validated content-addressed remedied-void row.
    """

    row: JsonObject = {
        "schema_version": SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2,
        "void_of": void_of,
        "reason": reason,
        "voided_by": voided_by,
        "created_at": created_at,
        "remedy": {
            "fixing_sha": fixing_sha,
            "test_node": test_node,
            "canary": canary,
        },
    }
    row = stamp_row_digest(row)
    return validate_scheduler_quarantine_void(row)


def build_scheduler_quarantine_drain(
    *,
    void_of: str,
    reason: str,
    drained_by: str,
    created_at: str,
    fixed_in_tree_sha: str,
) -> JsonObject:
    """Build one disclosed machine drain (W12/RE-11).

    The drain is appendable only against a live row (the tool additionally
    refuses before the bounded retries are exhausted), resets the generation's
    retry budget, and binds next-leg admission to the tree carrying
    ``fixed_in_tree_sha`` (:func:`unsatisfied_drain_tree_pins`).

    Parameters
    ----------
    void_of:
        Exact ``row_sha256`` of the live quarantine row.
    reason:
        Operator justification for the drain.
    drained_by:
        Operator identity making the decision.
    created_at:
        UTC timestamp of the decision.
    fixed_in_tree_sha:
        Commit the running tree must contain before the next leg may open.

    Returns
    -------
    dict[str, Any]
        Validated content-addressed drain row.
    """

    row: JsonObject = {
        "schema_version": SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION,
        "void_of": void_of,
        "reason": reason,
        "drained_by": drained_by,
        "created_at": created_at,
        "fixed_in_tree_sha": fixed_in_tree_sha,
    }
    row = stamp_row_digest(row)
    return validate_scheduler_quarantine_drain(row)


def validate_scheduler_quarantine_drain(row: Mapping[str, Any]) -> JsonObject:
    """Validate one machine-drain row.

    Raises
    ------
    SchedulerQuarantineError
        If required authority, remedy, or content-address fields are invalid.
    """

    required = (
        "schema_version",
        "void_of",
        "reason",
        "drained_by",
        "created_at",
        "fixed_in_tree_sha",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise SchedulerQuarantineError(
            f"scheduler quarantine drain is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION:
        raise SchedulerQuarantineError("scheduler quarantine drain declares unknown schema")
    for field in ("void_of", "reason", "drained_by", "created_at", "fixed_in_tree_sha"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise SchedulerQuarantineError(f"scheduler quarantine drain has an empty {field}")
    return _finish_content_addressed(row, kind="drain")


def _finish_content_addressed(row: Mapping[str, Any], *, kind: str) -> JsonObject:
    """Canonicalize, refuse foreign content addresses, and re-derive the digest."""

    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise SchedulerQuarantineError(
            f"scheduler {kind} row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise SchedulerQuarantineError(
            f"scheduler quarantine {kind} digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def validate_scheduler_quarantine_void(row: Mapping[str, Any]) -> JsonObject:
    """Validate one scheduler-quarantine operator void (bare v1 or remedied v2).

    Parameters
    ----------
    row:
        Candidate void row.

    Returns
    -------
    dict[str, Any]
        Canonical validated void row.

    Raises
    ------
    SchedulerQuarantineError
        If required authority, remedy, or content-address fields are invalid.
    """

    schema = row.get("schema_version")
    if schema not in SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSIONS:
        raise SchedulerQuarantineError("scheduler quarantine void declares unknown schema")
    required = ["schema_version", "void_of", "reason", "voided_by", "created_at", "row_sha256"]
    if schema == SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2:
        required.append("remedy")
    missing = [field for field in required if field not in row]
    if missing:
        raise SchedulerQuarantineError(
            f"scheduler quarantine void is missing {', '.join(sorted(missing))}"
        )
    for field in ("void_of", "reason", "voided_by", "created_at"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise SchedulerQuarantineError(
                f"scheduler quarantine void has an empty {field}"
            )
    if schema == SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSION_V2:
        remedy = row.get("remedy")
        if not isinstance(remedy, Mapping) or set(remedy) != {
            "fixing_sha",
            "test_node",
            "canary",
        }:
            raise SchedulerQuarantineError(
                "scheduler quarantine remedied void must declare exactly the "
                "DX-69 remedy shape: fixing_sha, test_node, canary"
            )
        for field in ("fixing_sha", "test_node", "canary"):
            value = remedy.get(field)
            if not isinstance(value, str) or not value.strip():
                raise SchedulerQuarantineError(
                    f"scheduler quarantine remedied void has an empty remedy {field}: "
                    "a partial remedy restores nothing"
                )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise SchedulerQuarantineError(
            f"scheduler park row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise SchedulerQuarantineError(
            "scheduler quarantine void digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def validate_scheduler_quarantine_ledger_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one ledger line as a quarantine park or operator void.

    Parameters
    ----------
    row:
        Decoded JSONL object.

    Returns
    -------
    dict[str, Any]
        Validated park or void row.
    """

    schema = row.get("schema_version")
    if schema in SCHEDULER_QUARANTINE_VOID_SCHEMA_VERSIONS:
        return validate_scheduler_quarantine_void(row)
    if schema == SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION:
        return validate_scheduler_quarantine_drain(row)
    return validate_scheduler_quarantine_row(row)


def _split_scheduler_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonObject], set[str]]:
    """Split validated scheduler ledger rows into parks and released digests.

    Every release kind -- bare void, remedied void, machine drain -- removes
    its park from the live set; only the LAST TWO restore retry budget
    (:func:`budget_restoring_boundaries`).
    """

    parks: list[JsonObject] = []
    voided: set[str] = set()
    for row in rows:
        if row.get("schema_version") in SCHEDULER_QUARANTINE_RELEASE_SCHEMA_VERSIONS:
            voided.add(str(row["void_of"]))
        else:
            parks.append(dict(row))
    return parks, voided


def current_scheduler_quarantine_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[JsonObject, ...]:
    """Return EVERY unvoided park, per row -- never latest-per-key (clause 9).

    RULING W7 clause 9, from the defect O A3 verified by execution: withholding
    liveness is derived PER ROW, not per (stable_id, work_id)-latest. Under the
    old latest-per-key map, one lawful void of the LATER of two unvoided parks
    for a work generation silently erased the EARLIER unvoided row too -- even
    a verdict-bearing one. Every unreleased park is live; a release removes
    exactly the row it names.

    Parameters
    ----------
    rows:
        Validated scheduler park and void rows in ledger order.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Live rows ordered by ``(stable_id, work_id, ledger position)``.
    """

    parks, voided = _split_scheduler_rows(rows)
    live = [
        (str(row["stable_id"]), str(row["work_id"]), position, row)
        for position, row in enumerate(parks)
        if str(row["row_sha256"]) not in voided
    ]
    return tuple(row for _sid, _wid, _position, row in sorted(live, key=lambda item: item[:3]))


def append_scheduler_quarantine_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one scheduler quarantine durably and idempotently.

    Parameters
    ----------
    path:
        Destination scheduler-quarantine JSONL.
    row:
        Complete typed quarantine row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    SchedulerQuarantineError
        If the row is malformed.

    Notes
    -----
    The FIRST quarantine recorded for a work generation is the record, and a
    later run that re-quarantines the same generation is a no-op:
    ``created_at`` and ``run_id`` necessarily differ on every resume, so
    treating a re-derived quarantine as a conflict would abort the driver on
    the second fault over the same model. A member the scheduler eventually
    routes normally writes nothing here, so a stale row can only ever be
    superseded by the model completing normally.
    """

    validated = validate_scheduler_quarantine_row(row)
    key = (validated["stable_id"], validated["work_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_scheduler_quarantine_lines(_ledger_bytes(path, handle), path)
        matching = [
            item
            for item in current_scheduler_quarantine_rows(existing)
            if (item["stable_id"], item["work_id"]) == key
        ]
        if matching:
            return matching[0]
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def append_scheduler_quarantine_void(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one operator drain durably and idempotently.

    Parameters
    ----------
    path:
        Scheduler-quarantine ledger holding the target park.
    row:
        Complete typed void row.

    Returns
    -------
    dict[str, Any]
        Persisted void row.

    Raises
    ------
    SchedulerQuarantineError
        If the target park does not exist or the void is malformed.
    """

    schema = row.get("schema_version")
    if schema == SCHEDULER_QUARANTINE_DRAIN_SCHEMA_VERSION:
        validated = validate_scheduler_quarantine_drain(row)
    else:
        validated = validate_scheduler_quarantine_void(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_scheduler_quarantine_lines(_ledger_bytes(path, handle), path)
        parks, _voided = _split_scheduler_rows(existing)
        target = str(validated["void_of"])
        if not any(str(item["row_sha256"]) == target for item in parks):
            raise SchedulerQuarantineError(
                f"scheduler quarantine void references an unknown row: {target}"
            )
        prior = [
            item
            for item in existing
            if item.get("schema_version") in SCHEDULER_QUARANTINE_RELEASE_SCHEMA_VERSIONS
            and str(item["void_of"]) == target
        ]
        if prior:
            if str(prior[0]["row_sha256"]) == str(validated["row_sha256"]):
                # Byte-identical re-release: idempotent.
                return dict(prior[0])
            # SD-7: the prior release stands. Silently accepting a SECOND,
            # different release against the same park would forge a budget
            # restoration (a drain "of" a bare-voided row drained nothing).
            raise SchedulerQuarantineError(
                f"scheduler quarantine row {target} was already released by "
                f"{prior[0]['schema_version']}; the prior release stands"
            )
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def _ledger_bytes(path: Path, handle: Optional[BinaryIO] = None) -> bytes:
    """Read the whole ledger, recovering only a torn final line with evidence.

    A complete malformed line stays this valve's typed hard refusal: the
    recovery's prefix check is this module's own strict decoder.
    """

    return read_with_tail_recovery(
        path,
        prefix_check=lambda retained: _decode_scheduler_quarantine_lines(
            retained.read_bytes(), path
        ),
        locked_handle=handle,
    )


def load_scheduler_quarantine_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load validated scheduler quarantines in ledger order.

    The one validated, tail-recovering read path for status and preflight
    consumers, exactly as the sibling valves expose.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Validated rows in ledger order.
    """

    rows: list[JsonObject] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.extend(_decode_scheduler_quarantine_lines(_ledger_bytes(path), path))
    return current_scheduler_quarantine_rows(rows)


def load_scheduler_quarantine_ledger(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load EVERY validated quarantine ledger row -- parks and releases alike.

    The reduced :func:`load_scheduler_quarantine_rows` view deliberately drops
    released rows, which is exactly why the budget derivation cannot use it:
    "when was this generation last drained" lives only in the full history.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Every validated row in ledger order.
    """

    rows: list[JsonObject] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.extend(_decode_scheduler_quarantine_lines(_ledger_bytes(path), path))
    return tuple(rows)


def _decode_scheduler_quarantine_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one quarantine ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SchedulerQuarantineError(
                f"scheduler quarantine ledger {path} has an unreadable row at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise SchedulerQuarantineError(
                f"scheduler quarantine ledger {path} has a non-object row at line {index}"
            )
        rows.append(validate_scheduler_quarantine_ledger_row(decoded))
    return tuple(rows)
