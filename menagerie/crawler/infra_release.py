"""Infra-terminal grant release: the second four-part waiver (KERNEL_DECISION 2026-08-21).

Requeue grants bind only review-held terminals (``status.human_review.required``):
the terminal-review lane adjudicates AUTHOR CLAIMS (SKIP / DEFER / BLOCKED
recommendations), and a grant on an unreviewed terminal holds forever as
``unreviewed-terminal``. A claim-less machine-infrastructure death -- a provider
storm, a worker crash before any receipt, an effort exhaustion whose sessions the
host censored -- has NOTHING the review lane can adjudicate, so its hold is a
review requirement with no consumer: exactly the shape the four-part waiver
doctrine (CLAUDE.md, LOCKED) permits to be absent, and only that shape.

The kernel decision of 2026-08-21 authorizes a grant to bind WITHOUT human review
on a terminal that is PROVABLY machine infrastructure. This module is the closed
proof rule. The four parts, explicitly:

1. **Disposition-invariant.** Eligibility releases an operator-authorized
   re-drive only: the grant row itself is still operator-appended
   (``tools/requeue.py``), the terminal's verdict is untouched, nothing publishes,
   nothing upgrades. A hold that stands loses nothing that existed before.
2. **Closed proof rule.** Infra-ness is decided exhaustively over the terminal's
   WITNESSING ATTEMPTS (``status.attempt_ids``) and their whole work generation:
   every failing attempt in the generation must match one enumerated machine
   classification row below -- exact typed exception identities, the closed
   :data:`~menagerie.crawler.constants.EFFORT_EXHAUSTION_REASON_CODES` vocabulary,
   machine-observed supervisor exits, or the TRI-A session-failure fields
   (``error.failure_class`` / ``error.failure_class_basis``) restricted to the
   MACHINE-classified bases -- never message text, never a reason code alone.
   One non-infra witnessing attempt = NOT eligible = the hold stands. Widening
   any set here is a new waiver and needs its own deliberated, reviewed change.
3. **Replay-derived, zero producer input.** :func:`derive_infra_terminal_release`
   is recomputed on every leg from ledger-bound records: the terminal head and
   the canonical attempts ledger, each attempt re-authenticated through
   :func:`~menagerie.crawler.authority.load_current_attempt_proof` (parent-attested
   proof replay). No persisted flag steers it -- the ``release_authority``
   disclosure on a consumption event is written for the census and never read
   back as authority. Historical pre-vocabulary terminals qualify by the same
   recomputation over the same fields.
4. **Disclosed.** Released-via-infra-waiver grants appear in the derived
   :func:`~menagerie.crawler.status.grant_release_census` class
   ``infra-waiver-released``, segregated from ``review-released``; the selection
   census counts ``granted_infra_waiver`` beside ``granted``; waived is never
   counted as reviewed.

First instance of the doctrine: ``CanonicalReducer._fidelity_gate_has_no_consumer``
(tri-lab KERNEL_DECISION of 2026-08-11). This is the second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.authority import AuthorityDerivationError, load_current_attempt_proof
from menagerie.crawler.constants import (
    BLOCKED_PREREQUISITE_REASON_CODE,
    EFFORT_EXHAUSTION_REASON_CODES,
    TERMINAL_DISPOSITION_FAILURE_REASONS,
)
from menagerie.crawler.release_vocabulary import (
    CACHE_INVALIDATION_FAULT_EXCEPTION_TYPES,
    MACHINE_RELEASE_CLASS_BLOCKED_PREREQUISITE,
    MACHINE_RELEASE_CLASS_CORPUS_CONFLICT,
    MACHINE_RELEASE_CLASS_ENGINE_CACHE,
    MACHINE_RELEASE_CLASS_QUOTA_DEFERRAL,
    ReleaseVocabularyError,
    SIGNED_MACHINE_RELEASE_CLASSES,
    is_blocked_prerequisite_reason,
    validate_blocked_prerequisite_record,
    validate_cache_invalidation_fault_event,
    validate_prerequisite_provision_witness,
)

__all__ = [
    "GRANT_RELEASE_AUTHORITIES",
    "INFRA_EXHAUSTION_EXCEPTION_TYPES",
    "INFRA_SESSION_CRASH_EXCEPTION_TYPES",
    "InfraTerminalRelease",
    "MACHINE_CLASS_RELEASE_AUTHORITY_PREFIX",
    "MACHINE_SESSION_FAILURE_CLASSIFICATIONS",
    "MachineClassRelease",
    "RELEASE_AUTHORITY_AWARD_NAMING",
    "RELEASE_AUTHORITY_HUMAN_REVIEW",
    "RELEASE_AUTHORITY_INFRA_WAIVER",
    "TERMINAL_CLAIM_REASON_CODES",
    "derive_infra_terminal_release",
    "derive_machine_class_release",
    "machine_infrastructure_witness_basis",
]

#: The closed release-authority vocabulary a bound grant discloses. ``human-review``
#: is the historical lane (a review-held terminal); ``award-naming`` is AW-1's
#: explicit award reopen; ``infra-waiver`` is this module's lane. The census and
#: the consumption-event disclosure both speak only these values.
RELEASE_AUTHORITY_HUMAN_REVIEW = "human-review"
RELEASE_AUTHORITY_AWARD_NAMING = "award-naming"
RELEASE_AUTHORITY_INFRA_WAIVER = "infra-waiver"
GRANT_RELEASE_AUTHORITIES = frozenset(
    {
        RELEASE_AUTHORITY_HUMAN_REVIEW,
        RELEASE_AUTHORITY_AWARD_NAMING,
        RELEASE_AUTHORITY_INFRA_WAIVER,
    }
)

#: Closed driver-raised exception identities for an effort exhaustion the driver's
#: own arithmetic tripped. The cap trip is machine fact: no session text, checker
#: verdict, or author claim participates in ``AuthorEffortCapExceeded`` (the
#: per-generation session allowance) or ``AuthorEffortExhaustionClaim`` (the
#: executor's own wall-clock cap). ``AuthorSessionRetryExhausted`` is deliberately
#: ABSENT: a retry exhaustion wraps the underlying session conduct (a refusal
#: cohort exhausts retries too), so it may qualify only through the TRI-A
#: machine-classified session-failure arm below.
INFRA_EXHAUSTION_EXCEPTION_TYPES = frozenset(
    {
        "menagerie.crawler.driver_contracts.AuthorEffortCapExceeded",
        "menagerie.crawler.author_dispatch.AuthorEffortExhaustionClaim",
    }
)

#: Closed driver-raised TorchLens-internal exception identities for a session the
#: crawler's own machinery killed. A ledger write conflict is crawler
#: infrastructure by construction -- the session's content never participates.
INFRA_SESSION_CRASH_EXCEPTION_TYPES = frozenset(
    {
        "menagerie.crawler.recordio.LedgerConflictError",
    }
)

#: TRI-A's exact attempt fields, restricted to the MACHINE-classified pairs.
#: ``error.failure_class`` alone is a diagnostic label whose basis may be
#: quarantined session text (``diagnostic-text:*`` bases are producer-controlled
#: prose and NEVER qualify); only the two structured harness-envelope bases the
#: executor derives without reading any text are admissible, each pinned to the
#: one class that basis can honestly produce.
MACHINE_SESSION_FAILURE_CLASSIFICATIONS = frozenset(
    {
        ("provider-overloaded", "api_error_status:529"),
        ("auth-unavailable", "error.type:authentication_error"),
    }
)

#: Closed terminal reason codes that record an AUTHOR CLAIM crossing (BLOCKED
#: prerequisites, tier requests, malformed published results, and refused or
#: unverifiable terminal dispositions). A claim-bearing terminal always has a
#: consumer for review, so the waiver refuses it outright -- the review lane,
#: not this rule, adjudicates claims.
TERMINAL_CLAIM_REASON_CODES = frozenset(
    {
        BLOCKED_PREREQUISITE_REASON_CODE,
        "needs-higher-tier",
        "malformed-result",
        *TERMINAL_DISPOSITION_FAILURE_REASONS.values(),
    }
)

#: The one gate kind that records a terminal author claim's adjudication.
_TERMINAL_DISPOSITION_GATE_KIND = "terminal_disposition"


@dataclass(frozen=True)
class InfraTerminalRelease:
    """One terminal's replay-derived machine-infrastructure release eligibility.

    Attributes
    ----------
    work_id:
        The exact witnessed work generation (part 2's identity binding).
    witness_bases:
        Sorted closed classification bases, one per failing attempt in the
        generation -- the census-facing disclosure of WHY the hold had no
        review consumer.
    """

    work_id: str
    witness_bases: tuple[str, ...]


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` when it is a mapping, else an empty one (fail-closed reads)."""

    return value if isinstance(value, Mapping) else {}


def machine_infrastructure_witness_basis(attempt: Mapping[str, Any]) -> Optional[str]:
    """Classify one failing attempt against the closed machine-infrastructure rows.

    Every row is an exact conjunction of parent-written typed fields: actor,
    stage, driver observation, exception identity, closed reason vocabulary,
    supervisor exit, receipt absence, or the TRI-A machine-classified
    session-failure pair. Message text, quarantined diagnostics, and bare
    reason codes never decide (doctrine part 2).

    Parameters
    ----------
    attempt:
        Ledger-bound canonical attempt row (authenticate it BEFORE classifying).

    Returns
    -------
    str | None
        Closed classification basis, or ``None`` in the conservative
        direction -- an unclassified failing attempt holds the terminal.
    """

    if attempt.get("result") != "failed":
        return None
    error = attempt.get("error")
    if not isinstance(error, Mapping):
        return None
    stage = attempt.get("stage")
    if error.get("native_crash") is not False or error.get("stage") != stage:
        return None
    details = _mapping(error.get("details"))
    actor = attempt.get("actor")
    exception_type = error.get("exception_type")
    reason_code = error.get("reason_code")

    # Row 1: the driver's own effort arithmetic tripped (exhaustion from
    # censored sessions -- the generation bought its allowance and died).
    if (
        actor == "driver"
        and stage == "author"
        and details.get("driver_observed") is True
        and exception_type in INFRA_EXHAUSTION_EXCEPTION_TYPES
        and reason_code in EFFORT_EXHAUSTION_REASON_CODES
    ):
        return f"driver-effort-exhaustion:{exception_type}"

    # Row 2: the worker process died before writing ANY receipt; the
    # supervisor (parent) watched the non-zero exit itself.
    supervisor = _mapping(attempt.get("supervisor_observation"))
    receipt = _mapping(attempt.get("worker_receipt"))
    exit_code = supervisor.get("exit_code")
    if (
        actor == "worker"
        and stage == "runner"
        and exception_type is None
        and reason_code == "missing-receipt"
        and receipt.get("present") is False
        and isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and exit_code != 0
    ):
        return f"worker-crash-missing-receipt:exit:{exit_code}"

    # Row 3: the crawler's own machinery killed the session (typed
    # TorchLens-internal exception, driver-observed).
    if (
        actor == "driver"
        and stage == "author"
        and details.get("driver_observed") is True
        and exception_type in INFRA_SESSION_CRASH_EXCEPTION_TYPES
        and reason_code == "session-crashed"
    ):
        return f"driver-session-crash:{exception_type}"

    # Row 4: TRI-A's machine-classified session death -- the exact
    # (failure_class, failure_class_basis) pair, structured bases only.
    classification = (error.get("failure_class"), error.get("failure_class_basis"))
    if (
        stage == "author"
        and details.get("driver_observed") is True
        and classification in MACHINE_SESSION_FAILURE_CLASSIFICATIONS
    ):
        return f"machine-session-failure:{classification[1]}"

    return None


def _authenticated(attempt: Mapping[str, Any]) -> bool:
    """Return whether one attempt row replays its parent-attested v3 proof."""

    try:
        load_current_attempt_proof(attempt)
    except AuthorityDerivationError:
        return False
    return True


def derive_infra_terminal_release(
    record: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> Optional[InfraTerminalRelease]:
    """Recompute one unreviewed terminal's machine-infrastructure release eligibility.

    The closed proof rule (doctrine part 2), recomputed on every leg from
    ledger-bound evidence with zero producer input (part 3). Any deviation --
    a claim surface, a dangling or foreign witness, an unauthenticated row, an
    unclassified failing attempt, an ambiguous generation -- returns ``None``
    and the ``unreviewed-terminal`` hold stands unchanged (part 1).

    Parameters
    ----------
    record:
        Persisted supersession-lineage terminal head (the same authority
        grant binding reads).
    attempts:
        Canonical attempts ledger rows.
    gates:
        Canonical gates ledger rows (the terminal-claim surface).

    Returns
    -------
    InfraTerminalRelease | None
        Eligibility with its witnessed generation and disclosure bases, or
        ``None`` when the hold must stand.
    """

    status = _mapping(record.get("status"))
    if status.get("kind") != "failed":
        return None
    if _mapping(status.get("human_review")).get("required") is True:
        # A review-held terminal HAS a consumer; the review lane owns it.
        return None
    if status.get("reason_code") in TERMINAL_CLAIM_REASON_CODES or is_blocked_prerequisite_reason(
        status.get("reason_code")
    ):
        return None
    if _mapping(record.get("accuracy_gate")).get("gate_id"):
        return None

    resolved = _resolve_witnessed_generation(record, attempts)
    if resolved is None:
        return None
    stable_id, work_id, generation = resolved

    # Part 2 quantifies over the WHOLE generation, not just the referenced
    # witnesses: a failing attempt the record omitted still holds the terminal.
    witness_bases: list[str] = []
    for row in generation:
        if row.get("defer_evidence") is not None:
            # A defer probe is claim material; the review lane owns it.
            return None
        if row.get("result") != "failed":
            continue
        basis = machine_infrastructure_witness_basis(row)
        if basis is None:
            return None
        witness_bases.append(basis)
    if not witness_bases:
        return None

    if _adjudicated_claim_exists(gates, stable_id, work_id):
        return None

    return InfraTerminalRelease(work_id=work_id, witness_bases=tuple(sorted(witness_bases)))


def _resolve_witnessed_generation(
    record: Mapping[str, Any], attempts: Sequence[Mapping[str, Any]]
) -> Optional[tuple[str, str, tuple[Mapping[str, Any], ...]]]:
    """Resolve one terminal head's exact witnessed work generation, fail-closed.

    The shared identity-binding half of every release derivation (the original
    waiver and each machine class): the head must name real, distinct witness
    attempt ids; every witness must resolve to exactly one ledger row of the
    head's ``stable_id``; the witnesses must agree on ONE ``work_id``; and every
    row of that generation must replay its parent-attested proof. Any deviation
    returns ``None`` and the hold stands.

    Returns
    -------
    tuple[str, str, tuple[Mapping[str, Any], ...]] | None
        ``(stable_id, work_id, generation_rows)`` -- generation rows are EVERY
        authenticated attempt of the witnessed ``(stable_id, work_id)``, not
        just the referenced witnesses.
    """

    status = _mapping(record.get("status"))
    stable_id = record.get("stable_id")
    if not isinstance(stable_id, str) or not stable_id:
        return None

    attempt_ids = status.get("attempt_ids")
    if (
        not isinstance(attempt_ids, list)
        or not attempt_ids
        or not all(isinstance(value, str) and value for value in attempt_ids)
        or len(set(attempt_ids)) != len(attempt_ids)
    ):
        return None

    rows_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in attempts:
        row_id = row.get("attempt_id")
        if isinstance(row_id, str) and row_id:
            rows_by_id.setdefault(row_id, []).append(row)

    witnesses: list[Mapping[str, Any]] = []
    for attempt_id in attempt_ids:
        candidates = rows_by_id.get(attempt_id, [])
        if len(candidates) != 1:
            # Dangling or ambiguous witness reference: fail closed.
            return None
        witnesses.append(candidates[0])

    if any(row.get("stable_id") != stable_id for row in witnesses):
        return None
    work_ids = {row.get("work_id") for row in witnesses}
    if len(work_ids) != 1:
        return None
    work_id = work_ids.pop()
    if not isinstance(work_id, str) or not work_id:
        return None

    generation: list[Mapping[str, Any]] = []
    for row in attempts:
        if row.get("stable_id") != stable_id or row.get("work_id") != work_id:
            continue
        if not _authenticated(row):
            return None
        generation.append(row)
    return (stable_id, work_id, tuple(generation))


def _adjudicated_claim_exists(
    gates: Sequence[Mapping[str, Any]], stable_id: str, work_id: str
) -> bool:
    """Return whether a terminal-disposition adjudication binds this generation."""

    for gate in gates:
        if gate.get("gate_kind") != _TERMINAL_DISPOSITION_GATE_KIND:
            continue
        for item in gate.get("items") or ():
            item_map = _mapping(item)
            if item_map.get("stable_id") == stable_id and item_map.get("work_id") == work_id:
                return True
    return False


# --------------------------------------------------------------------------- #
# W14: machine-classified release for the four signed transient classes
# --------------------------------------------------------------------------- #

#: Release-authority disclosure prefix for machine-class releases. A consumption
#: event discloses ``machine-class:<class>``; the census never reads it back as
#: authority (part 3) -- it recomputes the class from records.
MACHINE_CLASS_RELEASE_AUTHORITY_PREFIX = "machine-class:"

#: Class-local closed attempt row for the corpus-conflict class: the crawler's
#: own ledger-conflict machinery killed the run mid-write, at either paid lane.
#: The 16 pilot rows all share exactly this parent-written shape (actor=driver,
#: driver_observed, native_crash false, error.stage==stage); the reason set is
#: the exact observed closed triple, never message text.
_CORPUS_CONFLICT_EXCEPTION_TYPE = "menagerie.crawler.recordio.LedgerConflictError"
_CORPUS_CONFLICT_REASON_CODES = frozenset(
    {"session-crashed", "internal-error", "protocol-violation"}
)
_CORPUS_CONFLICT_STAGES = frozenset({"author", "runner"})

#: Class-local closed attempt row for the blocked-prerequisite class: the
#: author's checked BLOCKED verdict carried as the attempt's typed cause
#: (driver_contracts.AuthorBlockedPrerequisite is never raised; it names the
#: witness so the attempt does not borrow an unrelated exception type).
_BLOCKED_CLAIM_EXCEPTION_TYPE = "menagerie.crawler.driver_contracts.AuthorBlockedPrerequisite"


@dataclass(frozen=True)
class MachineClassRelease:
    """One terminal's replay-derived machine-CLASS release eligibility.

    Attributes
    ----------
    release_class:
        The closed :data:`~menagerie.crawler.release_vocabulary.MACHINE_RELEASE_CLASSES`
        member whose proof rule derived.
    work_id:
        The exact witnessed work generation.
    witness_bases:
        Sorted closed classification bases, one per failing attempt -- the
        census-facing disclosure of the machine evidence.
    """

    release_class: str
    work_id: str
    witness_bases: tuple[str, ...]


def _class_witness_bases(
    generation: Sequence[Mapping[str, Any]],
    extra_basis: Any,
) -> Optional[tuple[str, ...]]:
    """Classify every failing attempt of one generation, with a class-local row.

    The O-B2 HARD CONSTRAINT, shared by all four classes: every failing attempt
    must carry a closed machine classification -- the base waiver rows or the
    class's own closed row (``extra_basis``). ONE unclassified failing attempt
    (a model-owned reason) refuses the whole class. A defer probe is claim
    material and refuses (the review lane owns it), exactly as in the base rule.
    """

    bases: list[str] = []
    for row in generation:
        if row.get("defer_evidence") is not None:
            return None
        if row.get("result") != "failed":
            continue
        basis = machine_infrastructure_witness_basis(row)
        if basis is None and extra_basis is not None:
            basis = extra_basis(row)
        if basis is None:
            return None
        bases.append(basis)
    if not bases:
        return None
    return tuple(sorted(bases))


def _corpus_conflict_witness_basis(attempt: Mapping[str, Any]) -> Optional[str]:
    """Classify one failing attempt against the corpus-conflict closed row."""

    if attempt.get("result") != "failed":
        return None
    error = attempt.get("error")
    if not isinstance(error, Mapping):
        return None
    stage = attempt.get("stage")
    if error.get("native_crash") is not False or error.get("stage") != stage:
        return None
    details = _mapping(error.get("details"))
    if (
        attempt.get("actor") == "driver"
        and stage in _CORPUS_CONFLICT_STAGES
        and details.get("driver_observed") is True
        and error.get("exception_type") == _CORPUS_CONFLICT_EXCEPTION_TYPE
        and error.get("reason_code") in _CORPUS_CONFLICT_REASON_CODES
    ):
        return f"corpus-conflict:{error.get('reason_code')}"
    return None


def _blocked_claim_witness_basis(attempt: Mapping[str, Any]) -> Optional[str]:
    """Classify one failing attempt against the blocked-claim closed row."""

    if attempt.get("result") != "failed":
        return None
    error = attempt.get("error")
    if not isinstance(error, Mapping):
        return None
    stage = attempt.get("stage")
    if error.get("native_crash") is not False or error.get("stage") != stage:
        return None
    details = _mapping(error.get("details"))
    if (
        details.get("driver_observed") is True
        and error.get("exception_type") == _BLOCKED_CLAIM_EXCEPTION_TYPE
    ):
        return "blocked-prerequisite-claim"
    return None


def _cache_fault_witness_basis(attempt: Mapping[str, Any]) -> Optional[str]:
    """Classify one failing attempt against the engine-cache fault closed row."""

    if attempt.get("result") != "failed":
        return None
    error = attempt.get("error")
    if not isinstance(error, Mapping):
        return None
    stage = attempt.get("stage")
    if error.get("native_crash") is not False or error.get("stage") != stage:
        return None
    details = _mapping(error.get("details"))
    if (
        attempt.get("actor") == "driver"
        and details.get("driver_observed") is True
        and error.get("exception_type") in CACHE_INVALIDATION_FAULT_EXCEPTION_TYPES
    ):
        return f"engine-cache-fault:{error.get('exception_type')}"
    return None


def _unreviewed_claim_free(
    record: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    work_id: str,
) -> bool:
    """Return the base rule's consumer-absence preconditions (parts 1 and 2)."""

    status = _mapping(record.get("status"))
    if _mapping(status.get("human_review")).get("required") is True:
        return False
    if status.get("reason_code") in TERMINAL_CLAIM_REASON_CODES or is_blocked_prerequisite_reason(
        status.get("reason_code")
    ):
        return False
    if _mapping(record.get("accuracy_gate")).get("gate_id"):
        return False
    return not _adjudicated_claim_exists(gates, stable_id, work_id)


def _quota_deferral_release(
    record: Mapping[str, Any],
    generation: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    work_id: str,
    valve_rows: Sequence[Mapping[str, Any]],
) -> Optional[tuple[str, ...]]:
    """QUOTA-deferral class proof rule (signed C.8 arm; four parts below).

    1. **Disposition-invariant.** The terminal (or valve withholding) is already
       the member's negative disposition; eligibility funds only the
       fresh-generation retry a reviewed release would grant. The PD-3 recidivist
       park is checked FIRST: a member past its deferral bound holds for a human
       regardless of any other evidence, so this arm can never upgrade a
       recidivist.
    2. **Closed proof rule.** Head reason in the closed
       :data:`~menagerie.crawler.constants.EFFORT_EXHAUSTION_REASON_CODES`
       vocabulary (or a validated quota-deferral valve row binding this exact
       generation); every failing attempt machine-classified by the BASE waiver
       rows (no class-local widening: an exhaustion generation's failures are
       driver-arithmetic rows by construction); at least one
       ``driver-effort-exhaustion:`` basis so a non-exhaustion terminal can
       never ride this class. Checker-lane exhaustion binds only through valve
       rows minted going forward -- 200a has no closed checker-side attempt
       evidence, a DELIBERATE narrowness (class audit finding 2), never a
       silent one.
    3. **Replay-derived.** Head, attempts, gates, and valve rows are all
       ledger-bound parent-written records re-validated here; the valve rows
       revalidate through the quota-deferral family validator on every call.
    4. **Disclosed.** Census arm ``machine-released:quota-deferral``
       (:mod:`menagerie.crawler.status`), segregated from ``review-released``
       and from the base ``infra-waiver-released``.

    The invisible-victim assumption (class audit finding 3) is stated here: this
    class binds terminals and valve rows ONLY; an exhaustion path that fails to
    terminalize or defer leaves nothing to bind, and W1's event-emission
    invariant is the tripwire that keeps that path extinct.
    """

    binding_valves: list[Mapping[str, Any]] = []
    for row in valve_rows:
        try:
            from menagerie.crawler.quota_deferral import validate_quota_deferral_row

            validated = validate_quota_deferral_row(row)
        except Exception:
            # An invalid or foreign valve row is no evidence; fail toward hold.
            continue
        if validated.get("stable_id") == stable_id and validated.get("work_id") == work_id:
            binding_valves.append(validated)
    if any(
        row.get("disposition") == "review-held:storm-recidivist" for row in binding_valves
    ):
        return None

    status = _mapping(record.get("status"))
    exhaustion_head = status.get("reason_code") in EFFORT_EXHAUSTION_REASON_CODES
    deferral_valve = any(
        row.get("disposition") == "machine-deferred:quota-storm" for row in binding_valves
    )
    if not exhaustion_head and not deferral_valve:
        return None
    if not _unreviewed_claim_free(record, gates, stable_id, work_id):
        return None
    bases = _class_witness_bases(generation, None)
    if bases is None:
        return None
    if not any(basis.startswith("driver-effort-exhaustion:") for basis in bases):
        return None
    return bases


def _corpus_conflict_release(
    record: Mapping[str, Any],
    generation: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    work_id: str,
) -> Optional[tuple[str, ...]]:
    """Corpus-conflict class proof rule (signed C.8 arm; four parts below).

    1. **Disposition-invariant.** Binds only an already-failed terminal; funds
       only the fresh-generation retry. The W10 idempotent-replay fix removed
       the cause; nothing about the dead generation upgrades.
    2. **Closed proof rule.** At least one failing attempt carries the exact
       parent-written conflict shape (driver actor, driver-observed, typed
       ``recordio.LedgerConflictError``, the observed closed reason triple);
       every OTHER failing attempt is machine-classified by the base rows or
       the same conflict row. The conflict is crawler infrastructure by
       construction: no session content participates in a ledger write race.
    3. **Replay-derived.** Typed fields of authenticated attempt rows only;
       no message text, no producer field.
    4. **Disclosed.** Census arm ``machine-released:corpus-conflict``.

    200a precedent note (class audit finding 5): no 200a grant consumed on this
    shape; the class is signed on the fault evidence (16 typed conflict rows),
    forward-looking by design.
    """

    if _mapping(record.get("status")).get("kind") != "failed":
        return None
    if not _unreviewed_claim_free(record, gates, stable_id, work_id):
        return None
    bases = _class_witness_bases(generation, _corpus_conflict_witness_basis)
    if bases is None:
        return None
    if not any(basis.startswith("corpus-conflict:") for basis in bases):
        return None
    return bases


def _blocked_prerequisite_release(
    record: Mapping[str, Any],
    generation: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    work_id: str,
    blocked_prerequisite_rows: Sequence[Mapping[str, Any]],
    provision_witnesses: Sequence[Mapping[str, Any]],
) -> Optional[tuple[str, ...]]:
    """Blocked-prerequisite-post-rotation proof rule (signed C.8 arm; four parts).

    This is the ONE class that binds a claim-bearing terminal, and its
    disposition-invariance argument is exactly why that is lawful:

    1. **Disposition-invariant.** The author's BLOCKED claim was ALREADY
       adjudicated ACCURATE by the terminal-disposition gate this rule requires
       and re-binds by content address -- the review the claim was owed has been
       consumed. The only question left ("is the named prerequisite provided
       NOW?") has a closed machine answer: an attested environment rotation
       witness per token, REGISTERED AFTER the claim. The grant funds only the
       fresh-generation retry a reviewer reading the same rotation registry
       would grant; the terminal, the gate, and the claim are untouched.
    2. **Closed proof rule.** A validated typed seam record (the AL-6/EN-6
       vocabulary: grammar-closed prerequisite tokens, gate + author-result
       content addresses) for exactly this generation; the adjudicating gate
       item present with ``kind=BLOCKED``, ``predicate=blocked-prerequisite``,
       ``verdict=accepted``, and the SAME ``author_result_sha256`` the seam
       record binds; EVERY token carrying a valid rotation-provision witness
       registered strictly after the seam record's mint; every failing attempt
       machine-classified by the base rows or the typed blocked-claim row.
       ERA + WIDTH (class audit finding 4): 200a terminals have no typed seam
       record -- their prerequisite names exist only in redacted producer text,
       which part 3 forbids reading -- so they refuse here and stay on the
       human lane; a code-fix narrative is likewise not a witness shape. The
       class is deliberately NARROW, disclosed, never silently widened.
    3. **Replay-derived.** Seam records, gate rows, and witnesses are all
       parent-written and re-validated by content address on every call; the
       tokens were grammar-validated at driver mint time and are re-validated
       here. Producer text cannot mint a seam record, a gate acceptance, or a
       rotation witness (B2-X-V29).
    4. **Disclosed.** Census arm
       ``machine-released:blocked-prerequisite-post-rotation``.
    """

    status = _mapping(record.get("status"))
    if status.get("kind") != "failed":
        return None
    if not is_blocked_prerequisite_reason(status.get("reason_code")):
        return None

    seam_rows: list[Mapping[str, Any]] = []
    for row in blocked_prerequisite_rows:
        try:
            validated = validate_blocked_prerequisite_record(row)
        except ReleaseVocabularyError:
            continue
        if validated["stable_id"] == stable_id and validated["work_id"] == work_id:
            seam_rows.append(validated)
    if len(seam_rows) != 1:
        # Zero (era: 200a rows) or ambiguous typed naming: hold for a human.
        return None
    seam = seam_rows[0]

    adjudicated = False
    for gate in gates:
        if gate.get("gate_kind") != _TERMINAL_DISPOSITION_GATE_KIND:
            continue
        if gate.get("gate_id") != seam["gate_id"]:
            continue
        for item in gate.get("items") or ():
            item_map = _mapping(item)
            if item_map.get("stable_id") != stable_id or item_map.get("work_id") != work_id:
                continue
            disposition = _mapping(item_map.get("terminal_disposition"))
            if (
                disposition.get("kind") == "BLOCKED"
                and disposition.get("predicate") == BLOCKED_PREREQUISITE_REASON_CODE
                and disposition.get("verdict") == "accepted"
                and disposition.get("author_result_sha256") == seam["author_result_sha256"]
            ):
                adjudicated = True
    if not adjudicated:
        return None

    witnesses_by_token: dict[str, list[Mapping[str, Any]]] = {}
    for row in provision_witnesses:
        try:
            validated = validate_prerequisite_provision_witness(row)
        except ReleaseVocabularyError:
            continue
        witnesses_by_token.setdefault(str(validated["token"]), []).append(validated)
    for token in seam["prerequisite_tokens"]:
        provided = [
            witness
            for witness in witnesses_by_token.get(token, ())
            if str(witness["registered_at"]) > str(seam["created_at"])
        ]
        if not provided:
            return None

    bases = _class_witness_bases(generation, _blocked_claim_witness_basis)
    if bases is None:
        return None
    if "blocked-prerequisite-claim" not in bases:
        return None
    return bases


def _engine_cache_release(
    record: Mapping[str, Any],
    generation: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    stable_id: str,
    work_id: str,
    invalidation_events: Sequence[Mapping[str, Any]],
) -> Optional[tuple[str, ...]]:
    """Engine-invalidated-artifact-cache proof rule (AT-14; signed C.8 arm).

    1. **Disposition-invariant.** Binds only an already-failed terminal whose
       committed artifact the engine's OWN machinery destroyed; funds only the
       fresh-generation retry. Nothing publishes; the fault evidence is already
       durable either way.
    2. **Closed proof rule.** A validated engine-fault cache-invalidation event
       (closed kind, closed ``engine-fault`` reason, closed TorchLens-internal
       fault exception identity, parent-written by the artifact-transaction
       machinery) binding exactly this generation; every failing attempt
       machine-classified by the base rows or the typed engine-fault row.
       200a precedent note (class audit finding 5): all 21 candidate 200a
       grants were voided and resolved on the human supersession lane; the
       class is signed on fault evidence, forward-looking by design.
    3. **Replay-derived.** Typed event and attempt fields only; a session never
       writes an invalidation event.
    4. **Disclosed.** Census arm
       ``machine-released:engine-invalidated-artifact-cache``.
    """

    if _mapping(record.get("status")).get("kind") != "failed":
        return None
    if not _unreviewed_claim_free(record, gates, stable_id, work_id):
        return None
    bound = False
    for event in invalidation_events:
        try:
            validated = validate_cache_invalidation_fault_event(event)
        except ReleaseVocabularyError:
            continue
        details = _mapping(validated.get("details"))
        if details.get("stable_id") == stable_id and details.get("work_id") == work_id:
            bound = True
    if not bound:
        return None
    bases = _class_witness_bases(generation, _cache_fault_witness_basis)
    if bases is None:
        return None
    return bases


def derive_machine_class_release(
    record: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    *,
    valve_rows: Sequence[Mapping[str, Any]] = (),
    blocked_prerequisite_rows: Sequence[Mapping[str, Any]] = (),
    provision_witnesses: Sequence[Mapping[str, Any]] = (),
    invalidation_events: Sequence[Mapping[str, Any]] = (),
) -> Optional[MachineClassRelease]:
    """Recompute one terminal's machine-CLASS release eligibility (W14 engine).

    The four class rules are each a JMT-signed widening of this registered
    four-part waiver instance (#2), evaluated in a fixed order over the SAME
    fail-closed identity binding as the base rule: head witnesses resolve
    exactly, the whole generation authenticates, and one deviation anywhere
    returns ``None``. Evidence sequences default to empty, so a caller without
    the class's evidence CANNOT derive that class -- absence fails closed, and
    era rows (200a) hold for humans exactly as before this engine existed.

    The classes evaluate most-evidence-specific first; each derives only while
    it is a member of the signed included-class set
    (:data:`~menagerie.crawler.release_vocabulary.SIGNED_MACHINE_RELEASE_CLASSES`
    -- the Wave-2 merge receipt's bound set).

    Parameters
    ----------
    record, attempts, gates:
        Exactly as :func:`derive_infra_terminal_release`.
    valve_rows:
        Quota-deferral valve family rows (W1), revalidated here.
    blocked_prerequisite_rows:
        Typed AL-6 seam records, revalidated here.
    provision_witnesses:
        EN-6 rotation-provision witnesses, revalidated here.
    invalidation_events:
        W9/AT-14 engine-fault cache-invalidation events, revalidated here.

    Returns
    -------
    MachineClassRelease | None
        The first class whose proof rule derives, or ``None`` -- the hold
        stands.
    """

    if _mapping(record.get("status")).get("kind") != "failed":
        return None
    resolved = _resolve_witnessed_generation(record, attempts)
    if resolved is None:
        return None
    stable_id, work_id, generation = resolved

    if MACHINE_RELEASE_CLASS_BLOCKED_PREREQUISITE in SIGNED_MACHINE_RELEASE_CLASSES:
        bases = _blocked_prerequisite_release(
            record,
            generation,
            gates,
            stable_id,
            work_id,
            blocked_prerequisite_rows,
            provision_witnesses,
        )
        if bases is not None:
            return MachineClassRelease(
                release_class=MACHINE_RELEASE_CLASS_BLOCKED_PREREQUISITE,
                work_id=work_id,
                witness_bases=bases,
            )
    if MACHINE_RELEASE_CLASS_ENGINE_CACHE in SIGNED_MACHINE_RELEASE_CLASSES:
        bases = _engine_cache_release(
            record, generation, gates, stable_id, work_id, invalidation_events
        )
        if bases is not None:
            return MachineClassRelease(
                release_class=MACHINE_RELEASE_CLASS_ENGINE_CACHE,
                work_id=work_id,
                witness_bases=bases,
            )
    if MACHINE_RELEASE_CLASS_CORPUS_CONFLICT in SIGNED_MACHINE_RELEASE_CLASSES:
        bases = _corpus_conflict_release(record, generation, gates, stable_id, work_id)
        if bases is not None:
            return MachineClassRelease(
                release_class=MACHINE_RELEASE_CLASS_CORPUS_CONFLICT,
                work_id=work_id,
                witness_bases=bases,
            )
    if MACHINE_RELEASE_CLASS_QUOTA_DEFERRAL in SIGNED_MACHINE_RELEASE_CLASSES:
        bases = _quota_deferral_release(
            record, generation, gates, stable_id, work_id, valve_rows
        )
        if bases is not None:
            return MachineClassRelease(
                release_class=MACHINE_RELEASE_CLASS_QUOTA_DEFERRAL,
                work_id=work_id,
                witness_bases=bases,
            )
    return None
