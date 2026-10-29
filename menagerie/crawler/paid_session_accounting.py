"""CONCERN A -- paid-session ACCOUNTING: pure budget truth over durable rows.

This module is the ONE place a paid-session money number is defined, and it is
deliberately a pure library: it opens no file, takes no lock, spawns no process,
reads no environment variable, and imports nothing from the custody or
durability concerns.  Every budget is a pure function of already-decoded rows,
so a change to how bytes reach disk (concern D) or to how a process is proven
dead (concern B) cannot alter a single charge.

The instrument set is CLOSED and each instrument names exactly one axis:

``QUOTA``
    Cumulative, per unit-of-work, never resets.  Axis: ``campaign_id`` plus the
    row's own ``work_generations``.
``RUNAWAY_BREAKER``
    Epoch-scoped, resets only on durable evidence the epoch ended.  Axis:
    ``campaign_id`` plus the row's own ``generation_id``.

Every money row stamps BOTH coordinates, so a consumer is windowed by
construction and cannot invent a window from process state, invocation labels,
or checker identity.  There is no mutable spend counter anywhere and no
hydration entry point: :func:`quota_spend` and friends take rows and nothing
else.

The settlement wall rule (:func:`charged_wall_seconds`) exists exactly once and
is applied twice -- by the writer when it mints a settlement, and by the
validator when it re-derives that settlement on every replay -- so a charge is
checkable evidence rather than a self-describing label.  The same
re-derive-what-the-writer-declared discipline covers each row's own content
address (:func:`validate_row_digest`), because a digest nothing recomputes is a
label the row asserts about itself.  That re-derivation now runs the SHARED
derivation the tamper-evident chain itself uses
(:func:`menagerie.crawler.identity.row_content_address`), so the money authority
and the chain address are equal by construction rather than by two modules
agreeing about an exclusion set, and the row's top-level vocabulary is closed
(:func:`validate_row_vocabulary`) so a foreign digest field is unrepresentable as
well as harmless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    SHADOW_REHEARSAL_MARKER_FIELD,
    canonical_json_bytes,
    foreign_content_address_fields,
    row_content_address,
    stable_hash,
)
from menagerie.crawler.ledger_chain import LedgerChainError, extend_chain_state

PAID_SESSION_CHAIN_SCOPE = "paid.sessions"
#: Census slot name for the ledger's external extent anchor.  The anchor is ONE
#: atomically replaced object rather than a chain, so nothing links rows in this
#: scope; the name exists so the durable-fsync census still has a slot to hold
#: the anchor writer's durability duty against.
PAID_SESSION_ANCHOR_CHAIN_SCOPE = "paid.sessions.anchor"
PAID_SESSION_RESERVATION_SCHEMA_V1 = "menagerie.crawler.paid-session-reservation.v1"
#: v2 adds ``attempt_group``: the spend id of the attempt's FIRST reservation
#: when this reservation is a LATER guard session of the same author attempt,
#: or ``None`` when this reservation heads a new attempt.  The field makes the
#: ATTEMPT a durable, replay-derived unit: the author allowance counts distinct
#: attempt groups, never raw guard sessions (PSLRETRY round-2 F1 -- pricing the
#: allowance in session units let one work generation buy ``2*A`` source-only
#: failed attempts and desynced ``author_sessions_used/max``).
PAID_SESSION_RESERVATION_SCHEMA = "menagerie.crawler.paid-session-reservation.v2"
PAID_SESSION_SETTLEMENT_SCHEMA_V1 = "menagerie.crawler.paid-session-settlement.v1"
PAID_SESSION_SETTLEMENT_SCHEMA_V2 = "menagerie.crawler.paid-session-settlement.v2"
PAID_SESSION_SETTLEMENT_SCHEMA_V3 = "menagerie.crawler.paid-session-settlement.v3"
PAID_SESSION_SETTLEMENT_SCHEMA_V4 = "menagerie.crawler.paid-session-settlement.v4"
#: v5 (D-1 rider): the provider block gains ``adapter_identity`` -- the
#: answering reconciler's own name/version stamp, already digest-covered by
#: ``receipt_sha256`` since Wave-1 -- so a settlement ROW alone names WHICH
#: adapter answered (200a's two deployed adapters were observationally
#: identical on every ledger row). ``None`` is the disclosed in-process-rig
#: spelling; a command adapter always populates it.
PAID_SESSION_SETTLEMENT_SCHEMA = "menagerie.crawler.paid-session-settlement.v5"
PAID_SESSION_CUSTODY_SCHEMA_V1 = "menagerie.crawler.paid-session-custody.v1"
PAID_SESSION_CUSTODY_SCHEMA = "menagerie.crawler.paid-session-custody.v2"
PAID_SESSION_CUSTODY_RELEASE_SCHEMA = "menagerie.crawler.paid-session-custody-release.v1"
PAID_SESSION_GENESIS_SCHEMA = "menagerie.crawler.paid-session-genesis.v1"
#: The rollover genesis for a campaign whose previous paid ledger the fold
#: REFUSES (the 2026-08-20 duplicate-custody residue).  The row EMBEDS the
#: poisoned ledger's chain-verified rows verbatim, so every replay recomputes
#: eligibility (the carried rows must still refuse the fold) and the
#: conservative opening charges from digest-bound evidence -- never from a
#: field the rollover tool wrote.  See the four-part waiver doctrine: the row
#: grants nothing positive (the new authority starts MORE charged, never
#: less), eligibility is a closed structural rule, the decision replays from
#: parent-attested rows, and the charges are disclosed in the folded state.
PAID_SESSION_ROLLOVER_SCHEMA = "menagerie.crawler.paid-session-rollover.v1"

#: Settlement schemas that predate the machine-floor wall rule.  They describe the
#: separately pinned legacy operational prefix; they are never written, and they
#: are never admissible for a row of THIS ledger.  That "historical prefix only"
#: claim is now STRUCTURAL rather than declared: the cutover
#: ``accounting_genesis`` row must be the ledger's first row and pins the legacy
#: prefix as another object entirely, so every row here is post-cutover by
#: construction and :meth:`_StateBuilder._admit_settlement_schema` refuses a legacy
#: schema whatever any row declares.  Nothing a producer writes can grant it
#: (PSLRE5 acct F2: the declaration was mintable by the same producer, and a
#: uniform downgrade never moved below the monotone ladder's own rank).
LEGACY_SETTLEMENT_SCHEMAS = frozenset(
    {
        PAID_SESSION_SETTLEMENT_SCHEMA_V1,
        PAID_SESSION_SETTLEMENT_SCHEMA_V2,
        PAID_SESSION_SETTLEMENT_SCHEMA_V3,
    }
)

#: The CLOSED settlement-schema ladder, oldest first.  Position is the only
#: ordering; a schema outside it is refused by :func:`validate_settlement` before
#: anything consults the ladder.
SETTLEMENT_SCHEMA_LADDER = (
    PAID_SESSION_SETTLEMENT_SCHEMA_V1,
    PAID_SESSION_SETTLEMENT_SCHEMA_V2,
    PAID_SESSION_SETTLEMENT_SCHEMA_V3,
    PAID_SESSION_SETTLEMENT_SCHEMA_V4,
    PAID_SESSION_SETTLEMENT_SCHEMA,
)

#: The post-cutover machine-floor family: every rung from v4 up shares the wall
#: re-derivation, refund semantics, and exact machine/provider truth blocks.
#: Versioning to v5 must never weaken what already held for v4 rows, so every
#: "current schema" check that guards those rules quantifies over THIS set --
#: keying any of them on the single current constant would silently exempt
#: honest v4 history from its own era's checks.
MACHINE_FLOOR_SETTLEMENT_SCHEMAS = frozenset(
    {PAID_SESSION_SETTLEMENT_SCHEMA_V4, PAID_SESSION_SETTLEMENT_SCHEMA}
)

INSTRUMENTS = {
    "QUOTA": frozenset({"campaign_id", "work_generations"}),
    "RUNAWAY_BREAKER": frozenset({"campaign_id", "generation_id"}),
}
REQUIRED_AXIS_FIELDS = frozenset({"campaign_id", "generation_id", "work_generations"})
ACCOUNTING_AXIS_PROVENANCE = {
    "campaign_id": "bound_campaign_id",
    "generation_id": "leg_generation.generation_id",
    "work_generations": "intake_and_work_generation_authority",
}
MONEY_EVENTS = frozenset({"session_reserved", "session_settled"})
LIFECYCLE_EVENTS = frozenset(
    {
        "accounting_genesis",
        "accounting_rollover",
        "session_reserved",
        "session_settled",
        "session_custody_bound",
        "session_custody_released",
    }
)
SETTLEMENT_EVIDENCE = frozenset({"receipt", "reservation", "provider-query"})
CUSTODY_RELEASE_OUTCOMES = frozenset(
    {"verified-exit", "verified-reap", "stale-boot", "operator-drain"}
)
#: Closed raw machine outcomes that refund only the ATTEMPT projection.  Each
#: value names evidence the driver can observe without consulting author/model
#: text.  They never affect QUOTA or RUNAWAY session/wall charges: those keep
#: using provider acceptance and the machine wall floor.  A new outcome is
#: chargeable until this reviewed vocabulary and the deriving driver arm grow
#: together, so ambiguity fails in the conservative direction.
#:
#: ``failed-checker-session-wall`` is stamped only by the checker lane's own
#: external wall kill (``CommandCheckerLane._run``): the driver killed the
#: whole guarded process group at the named wall x kill factor, so the session
#: never rendered a verdict for reasons no checker text can influence.
#: ``failed-custody-reaped`` is stamped by the driver's own custody machinery
#: and nowhere else: the mid-leg custody deadline reaper
#: (``_reap_expired_dead_custody``, past ``deadline_at`` plus the named grace)
#: and -- since item 9 (recovery F1.2) -- leg-start reconciliation
#: (``_reconcile_paid_custody_lease``) for a bound custodian the closed custody
#: ladder proves dead (``already-dead``: flock FREE and group EMPTY;
#: ``stale-boot``: another boot) REGARDLESS of deadline. Both kernel oracles
#: proved the session died without ever settling -- infrastructure, never
#: model conduct -- and the deadline was only ever the probe schedule.
#: ``failed-executor-typed-outage`` (item 9 / F1.1) is stamped only by
#: ``classify_author_exit`` from the closed proof rule
#: ``author_dispatch.executor_typed_outage_outcome``: the executor exited with
#: its typed 75/78 code, printed its own PRE-SESSION typed line, published no
#: payload at the envelope output path and emitted no harness notice -- the
#: 82 pilot / 742 R6b sub-10 s deaths that were charged as model attempts.
ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES = frozenset(
    {
        "campaign-teardown",
        "failed-author-queue-stalled",
        "failed-checker-session-wall",
        "failed-custody-reaped",
        "failed-executor-typed-outage",
        "failed-harness-aborted-streaming-sigterm",
        "failed-harness-api-error",
        "failed-machine-spawn",
    }
)
#: ``author-blocked-prerequisite`` (AL-6, JMT-signed 2026-08-24) is stamped only
#: by the author-lane success settle
#: (``driver_admission._author_blocked_prerequisite_settlement_outcome`` inside
#: ``_retry_infrastructure_call``): the session completed and returned a
#: schema-validated ``BLOCKED`` terminal recommendation -- the typed
#: recommendation-kind enum on the result envelope, the same trust level the
#: DEFER needs-cuda routing arm already relies on; no model text participates.
#: An honest checked prerequisite verdict is the author DOING ITS JOB, so it
#: must not consume the model's attempt allowance the way a crash does. The
#: refund upgrades nothing: a spammed BLOCKED still terminalizes its model and
#: never earns a retry, sessions/wall stay fully charged, and the member is
#: deliberately OUTSIDE the infrastructure subset above so the provider-storm
#: fuse and the ``effort-exhausted:provider-storm`` split never count an author
#: verdict as machine weather.
AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME = "author-blocked-prerequisite"
ATTEMPT_REFUND_LOCAL_OUTCOMES = ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES | {
    AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME
}
#: The closed set of machine-observed local dispositions a settlement discloses.
#: Most values remain accounting-inert.  The exact raw infrastructure outcomes
#: above affect only :func:`settled_attempt_weight`, never money/fuse weights.
LOCAL_OUTCOMES = frozenset(
    {"completed", "failed", "unknown", *ATTEMPT_REFUND_LOCAL_OUTCOMES}
)
#: The closed provenance set for the machine-side wall floor.
MACHINE_WALL_SOURCES = frozenset(
    {"writer-monotonic", "writer-reservation-age", "writer-unobserved"}
)
LANES = frozenset({"author", "checker"})

#: CLOSED per-lane bound on paid guard sessions in ONE attempt group, the fold's
#: own vocabulary (PSLRETRY round-4 F1).  The join rule was otherwise open-ended:
#: once every member settled, another reservation could join the same head
#: indefinitely, laundering unlimited paid sessions into ONE disclosed attempt
#: while every join bypassed the new-attempt valve.  An author attempt is the
#: source-request session plus the author session (2); a checker attempt is one
#: session, so a checker join is unrepresentable.  A lane absent from this
#: mapping refuses joins entirely -- a new lane must declare its bound here, in
#: one reviewed change, before its attempts may span sessions.  The production
#: QUOTA factor (:data:`menagerie.crawler.constants.AUTHOR_PAID_SESSIONS_PER_ATTEMPT`)
#: mirrors the author bound and is pinned equal in-test.
LANE_ATTEMPT_SESSION_BOUNDS: Mapping[str, int] = {"author": 2, "checker": 1}

#: CLOSED bounds on the nested rollover walk (PSLRETRY round-4 F2b).  Every
#: replay of a rollover row re-folds and re-walks its embedded rows, and a
#: nested rollover embeds its predecessor's rollover row in turn, so without a
#: bound a producer-shaped row could drive unbounded recursion and work
#: (``RecursionError`` is not the eligibility ``ValueError`` the fold catches).
#: The constants suffice by construction: a legitimate campaign rolls over only
#: when its ledger is durably poisoned (the pilot needs depth 1; a twice-rolled
#: campaign needs 2; 3 leaves one reviewed spare), and one generation's rows are
#: bounded by the RUNAWAY session budget (2000 sessions x <=4 rows each plus
#: genesis rows sits far below 50k even across several generations between
#: rollovers).  Exceeding either bound refuses the rollover row outright.
ROLLOVER_MAX_DEPTH = 3
ROLLOVER_MAX_CARRIED_ROWS = 50_000

#: Permanent scale-cost instrumentation.  Every row this module validates is
#: counted exactly once here, so the scale census can assert that validation
#: work over N appends stays LINEAR in N rather than re-verifying the whole
#: ledger per append.  Counting is the only mutable module state, it is
#: write-only from the accounting path, and no projection ever reads it.
ACCOUNTING_STATS: dict[str, int] = {"rows_validated": 0, "folds": 0}


def reset_accounting_stats() -> None:
    """Zero the scale-cost counters before one measurement window."""

    for key in ACCOUNTING_STATS:
        ACCOUNTING_STATS[key] = 0


class PaidSessionError(RuntimeError):
    """Base class for fail-closed paid-session lifecycle errors."""


class PaidSessionBudgetExceeded(PaidSessionError):
    """Raised when reserve refuses one named paid-session instrument."""

    def __init__(
        self,
        instrument: str,
        detail: str,
        *,
        work_generations: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Store the closed instrument name and refusal detail."""

        super().__init__(f"{instrument}: {detail}")
        self.instrument = instrument
        self.work_generations = work_generations


@dataclass(frozen=True)
class Weight:
    """Derived charge for one or more paid-session rows."""

    wall_seconds: float = 0.0
    sessions: int = 0

    def __add__(self, other: object) -> Weight:
        """Return the elementwise sum of two charge weights."""

        if not isinstance(other, Weight):
            return NotImplemented
        return Weight(
            wall_seconds=self.wall_seconds + other.wall_seconds,
            sessions=self.sessions + other.sessions,
        )


@dataclass(frozen=True)
class BudgetPolicy:
    """Limits for the QUOTA and RUNAWAY_BREAKER projections."""

    quota_sessions: int
    quota_wall_seconds: float
    runaway_sessions: int
    runaway_wall_seconds: float
    reservation_wall_seconds: float

    def __post_init__(self) -> None:
        """Reject non-finite or negative machine-owned budget values."""

        integer_values = (self.quota_sessions, self.runaway_sessions)
        wall_values = (
            self.quota_wall_seconds,
            self.runaway_wall_seconds,
            self.reservation_wall_seconds,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_values
        ):
            raise ValueError("paid-session session budgets must be non-negative integers")
        if any(not math.isfinite(value) or value < 0.0 for value in wall_values):
            raise ValueError("paid-session wall budgets must be finite and non-negative")


def finite_nonnegative(value: object, *, field: str) -> float:
    """Return one finite non-negative numeric field or refuse the ledger.

    Parameters
    ----------
    value:
        Decoded ledger value.
    field:
        Dotted field name used in the refusal.

    Returns
    -------
    float
        The validated value.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def charged_wall_seconds(
    *,
    reserved_wall_seconds: float,
    machine_wall_seconds: Optional[float],
    provider_wall_seconds: Optional[float],
) -> float:
    """Return THE settled wall charge for one paid session.

    A runaway is a LOCAL phenomenon, so the machine's own measurement is a
    FLOOR that a remote answer can only raise, never delete: a provider that
    declines the session, meters nothing, or under-reports cannot erase the
    seconds this host observably spent.  A provider that over-reports cannot
    charge past the authorization the reservation already durably bought.  Both
    directions are therefore bounded by evidence rather than by a label, and
    because the rule is a pure function of three persisted numbers, a replay
    re-derives the exact charge and a clamp is visible instead of silent.

    Parameters
    ----------
    reserved_wall_seconds:
        The reservation's own authorized ceiling.
    machine_wall_seconds:
        Writer-measured elapsed seconds, or ``None`` when this host observed
        none (a pre-settlement crash reconciled by a later leg).
    provider_wall_seconds:
        The provider's attested billing wall, or ``None`` when the provider
        does not meter wall clock at all.

    Returns
    -------
    float
        ``min(reserved, max(machine, provider))``.
    """

    floor = 0.0 if machine_wall_seconds is None else max(0.0, float(machine_wall_seconds))
    attested = 0.0 if provider_wall_seconds is None else max(0.0, float(provider_wall_seconds))
    return min(float(reserved_wall_seconds), max(floor, attested))


def work_generations(value: object) -> tuple[tuple[str, str], ...]:
    """Validate and canonicalize a nonempty work-generation coordinate.

    Parameters
    ----------
    value:
        Sequence of ``(stable_id, work_id)`` pairs or mappings.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Sorted, de-duplicated coordinate.
    """

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("work_generations must be a nonempty sequence")
    parsed: set[tuple[str, str]] = set()
    for item in value:
        if isinstance(item, Mapping):
            stable_id = item.get("stable_id")
            work_id = item.get("work_id")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            stable_id, work_id = item
        else:
            raise ValueError("work_generations entries must name stable_id and work_id")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError("work generation stable_id must be nonempty")
        if not isinstance(work_id, str) or not work_id:
            raise ValueError("work generation work_id must be nonempty")
        parsed.add((stable_id, work_id))
    return tuple(sorted(parsed))


# ---------------------------------------------------------------------------
# Per-row validation.  Every row is validated exactly once, when it first
# enters an accounting state, and never re-validated on a later read.
# ---------------------------------------------------------------------------

def validate_row_digest(row: Mapping[str, Any]) -> None:
    """Fail closed unless a row's declared content digest re-derives from its body.

    Every paid-session row carries a ``row_sha256`` its writer stamped over the
    row body with :func:`menagerie.crawler.identity.stamp_row_digest`, and the
    tamper-evident chain now DERIVES its content address from that same body
    (:func:`menagerie.crawler.ledger_chain.row_content_digest`) instead of reading
    a field the row declares about itself.  Three holes lived in that reading:

    * a length-preserving in-place edit of any money row moved no digest, no
      chain link, and no byte count, so it rewrote a charge with load, integrity,
      and anchor all green (PSLRE4 D-1);
    * a forged out-of-band row could satisfy an OPEN anchor intent by COPYING the
      intended row's digest instead of finding a preimage for it (PSLRE4 D-2);
    * a row carrying ONE extra field named ``close_sha256`` pinned the chain to an
      arbitrary string, so both of the above came straight back with strictly less
      work than the original attack needed (PSLRE5 acct F1 / dur P5-D-1).

    Re-deriving here through the SHARED derivation keeps the accounting authority
    and the chain's address exactly equal by construction, in the one place every
    row of every paid-session schema already passes through, and as a pure function
    of the row -- so this cannot become a durability or custody dependency.  A row
    carrying a FOREIGN content-address field refuses outright: a money row owns
    exactly one content address and has no reason to bind another family's.

    Parameters
    ----------
    row:
        Decoded paid-session lifecycle row.

    Raises
    ------
    ValueError
        When the digest is absent, malformed, foreign, or does not re-derive.
    """

    declared = row.get("row_sha256")
    if not isinstance(declared, str) or not declared:
        raise ValueError("paid-session row requires its own declared content digest")
    foreign = foreign_content_address_fields(row, own="row_sha256")
    if foreign:
        raise ValueError(
            "paid-session row carries a foreign content-address field "
            f"{', '.join(foreign)}: a money row owns exactly one content address"
        )
    if row_content_address(row) != declared:
        raise ValueError(
            "paid-session row digest does not re-derive from the row body: the row "
            "was edited, substituted, or stamped with a digest it does not own"
        )


#: The CLOSED top-level key vocabulary of every paid-session row, keyed by the
#: ``(event, schema_version)`` pair its writer mints.  Defense in depth behind the
#: body-derived content address: the address already makes a foreign digest field
#: harmless, and this makes it unrepresentable.  A NEW field -- digest-named or
#: not -- must be added HERE and, if it is a digest, to
#: :data:`menagerie.crawler.identity.ROW_CONTENT_ADDRESS_EXCLUDED` and the chain's
#: fused certification, in ONE reviewed change.
#:
#: The check is exact set EQUALITY (never a superset test), it runs at the single
#: fold chokepoint BEFORE any event dispatch, and the union over every admissible
#: pair contains no content-address name other than each row's own ``row_sha256``
#: -- all three asserted in ``tests/test_checker_wall_breaker.py``, because a row
#: that could choose WHICH vocabulary judges it would be judging itself.
_SETTLEMENT_LEGACY_KEYS = frozenset(
    {
        "schema_version",
        "event",
        "spend_id",
        "observed",
        "evidence",
        "custody_outcome",
        "created_at",
        "row_sha256",
    }
)
PAID_SESSION_ROW_VOCABULARY: Mapping[tuple[str, str], frozenset[str]] = {
    ("accounting_genesis", PAID_SESSION_GENESIS_SCHEMA): frozenset(
        {
            "schema_version",
            "event",
            "legacy_operational_path",
            "legacy_operational_sha256",
            "legacy_operational_bytes",
            "legacy_settlement_schemas",
            "created_at",
            "row_sha256",
        }
    ),
    ("session_reserved", PAID_SESSION_RESERVATION_SCHEMA_V1): frozenset(
        {
            "schema_version",
            "event",
            "spend_id",
            "provider_idempotency_key",
            "lane",
            "dispatch_kind",
            "axis",
            "reserved",
            "reservation_evidence",
            "gate_kind",
            "checker_identity",
            "run_id",
            "machine_id",
            "created_at",
            "custody",
            "row_sha256",
        }
    ),
    ("session_reserved", PAID_SESSION_RESERVATION_SCHEMA): frozenset(
        {
            "schema_version",
            "event",
            "spend_id",
            "provider_idempotency_key",
            "lane",
            "dispatch_kind",
            "attempt_group",
            "axis",
            "reserved",
            "reservation_evidence",
            "gate_kind",
            "checker_identity",
            "run_id",
            "machine_id",
            "created_at",
            "custody",
            "row_sha256",
        }
    ),
    ("accounting_rollover", PAID_SESSION_ROLLOVER_SCHEMA): frozenset(
        {
            "schema_version",
            "event",
            "poisoned_ledger_name",
            "poisoned_ledger_sha256",
            "poisoned_ledger_bytes",
            "poisoned_chain_head",
            "poisoned_row_count",
            "poisoned_anchor",
            "carried_rows",
            "created_at",
            "row_sha256",
        }
    ),
    ("session_settled", PAID_SESSION_SETTLEMENT_SCHEMA_V1): _SETTLEMENT_LEGACY_KEYS,
    ("session_settled", PAID_SESSION_SETTLEMENT_SCHEMA_V2): _SETTLEMENT_LEGACY_KEYS,
    ("session_settled", PAID_SESSION_SETTLEMENT_SCHEMA_V3): (
        _SETTLEMENT_LEGACY_KEYS | {"provider"}
    ),
    # v4 and v5 share the top-level vocabulary; v5's adapter_identity lives
    # INSIDE the provider block, whose exact per-schema key set
    # :func:`validate_settlement` owns.
    ("session_settled", PAID_SESSION_SETTLEMENT_SCHEMA_V4): (
        _SETTLEMENT_LEGACY_KEYS | {"provider", "machine", "local_outcome"}
    ),
    ("session_settled", PAID_SESSION_SETTLEMENT_SCHEMA): (
        _SETTLEMENT_LEGACY_KEYS | {"provider", "machine", "local_outcome"}
    ),
    ("session_custody_bound", PAID_SESSION_CUSTODY_SCHEMA_V1): frozenset(
        {
            "schema_version",
            "event",
            "spend_id",
            "lease_id",
            "boot_id",
            "host_id",
            "lock_path",
            "leader_pid",
            "leader_start_token",
            "leader_pgid",
            "created_at",
            "row_sha256",
        }
    ),
    ("session_custody_bound", PAID_SESSION_CUSTODY_SCHEMA): frozenset(
        {
            "schema_version",
            "event",
            "spend_id",
            "lease_id",
            "boot_id",
            "host_id",
            "lock_path",
            "lock_device",
            "lock_inode",
            "leader_pid",
            "leader_start_token",
            "leader_pgid",
            "created_at",
            "row_sha256",
        }
    ),
    ("session_custody_released", PAID_SESSION_CUSTODY_RELEASE_SCHEMA): frozenset(
        {
            "schema_version",
            "event",
            "spend_id",
            "lease_id",
            "outcome",
            "operator",
            "reason",
            "created_at",
            "row_sha256",
        }
    ),
}


def validate_row_vocabulary(row: Mapping[str, Any]) -> None:
    """Fail closed unless a row's top-level keys are EXACTLY its schema's.

    The row declares its own ``(event, schema_version)`` pair, so this cannot be
    the load-bearing control -- it is the closed vocabulary that makes an injected
    field unrepresentable, behind an address that already makes one harmless.  The
    pair must be admissible and the key set exact in both directions: an EXTRA key
    is the injection vector, and a MISSING key is a row that dodges a validator by
    omission.

    Parameters
    ----------
    row:
        Decoded paid-session lifecycle row.

    Raises
    ------
    ValueError
        When the pair is unknown or the top-level vocabulary is not exact.
    """

    pair = (str(row.get("event")), str(row.get("schema_version")))
    expected = PAID_SESSION_ROW_VOCABULARY.get(pair)
    if expected is None:
        raise ValueError(
            f"paid-session row declares the unknown event/schema pair {pair[0]!r}/"
            f"{pair[1]!r}: the lifecycle vocabulary is closed"
        )
    # W3 schema widening (reviewed change): the OPTIONAL shadow-rehearsal
    # marker is admissible on every row kind -- the write-side producer stamps
    # it on everything a rehearsal leg writes, inside ``row_sha256``. When
    # present it must be the typed namespace string; carrying it grants
    # nothing (marked rows are excluded from authority, never blessed by it).
    marker = row.get(SHADOW_REHEARSAL_MARKER_FIELD)
    if marker is not None and (not isinstance(marker, str) or not marker):
        raise ValueError(
            f"paid-session row {SHADOW_REHEARSAL_MARKER_FIELD} marker must be the "
            "non-empty rehearsal generation namespace string"
        )
    present = {str(key) for key in row} - {"chain", SHADOW_REHEARSAL_MARKER_FIELD}
    if present != expected:
        extra = sorted(present - expected)
        missing = sorted(expected - present)
        raise ValueError(
            f"paid-session {pair[0]} row vocabulary is not exact: unexpected "
            f"{extra}, absent {missing}"
        )


_RESERVATION_CUSTODY_FIELDS = frozenset(
    {
        "lease_id",
        "boot_id",
        "host_id",
        "lock_path",
        "driver_pid",
        "driver_start_token",
        "leader_pid",
        "leader_start_token",
        "leader_pgid",
        "deadline_at",
    }
)


def validate_reservation(row: Mapping[str, Any]) -> None:
    """Fail closed unless a reservation carries the complete accounting census."""

    schema = row.get("schema_version")
    if schema not in {PAID_SESSION_RESERVATION_SCHEMA, PAID_SESSION_RESERVATION_SCHEMA_V1}:
        raise ValueError("paid-session reservation schema mismatch")
    if row.get("event") != "session_reserved":
        raise ValueError("paid-session reservation event mismatch")
    if schema == PAID_SESSION_RESERVATION_SCHEMA:
        attempt_group = row.get("attempt_group")
        if attempt_group is not None and (
            not isinstance(attempt_group, str) or not attempt_group
        ):
            raise ValueError("paid-session reservation attempt_group must be null or a spend id")
        if attempt_group == row.get("spend_id"):
            raise ValueError("paid-session reservation attempt_group cannot name itself")
    for name in ("spend_id", "provider_idempotency_key", "lane", "dispatch_kind"):
        if not isinstance(row.get(name), str) or not row.get(name):
            raise ValueError(f"paid-session reservation requires {name}")
    if row["lane"] not in LANES:
        raise ValueError("paid-session reservation lane is outside the closed set")
    axis = row.get("axis")
    if not isinstance(axis, Mapping) or set(axis) != REQUIRED_AXIS_FIELDS:
        raise ValueError("paid-session reservation axis census is incomplete")
    if not isinstance(axis.get("campaign_id"), str) or not axis.get("campaign_id"):
        raise ValueError("paid-session reservation campaign_id is invalid")
    if not isinstance(axis.get("generation_id"), str) or not axis.get("generation_id"):
        raise ValueError("paid-session reservation generation_id is invalid")
    work_generations(axis.get("work_generations"))
    reserved = row.get("reserved")
    if not isinstance(reserved, Mapping) or reserved.get("sessions") != 1:
        raise ValueError("paid-session reservation requires exactly one session")
    finite_nonnegative(reserved.get("wall_seconds"), field="reserved.wall_seconds")
    custody = row.get("custody")
    if not isinstance(custody, Mapping) or set(custody) != _RESERVATION_CUSTODY_FIELDS:
        raise ValueError("paid-session reservation custody topology is incomplete")


def reservation_attempt_group(row: Mapping[str, Any]) -> Optional[str]:
    """Return the attempt-group head a reservation declares, if any.

    A v1 reservation predates the field and is its own attempt's head; a v2
    reservation either heads a new attempt (``None``) or joins the group headed
    by the named spend id.
    """

    value = row.get("attempt_group")
    return str(value) if isinstance(value, str) and value else None


def attempt_join_refusal(
    row: Mapping[str, Any],
    *,
    head: Optional[Mapping[str, Any]],
    group_members: Sequence[str],
    settled_ids: Any,
    live_members: int,
) -> Optional[str]:
    """Return THE closed attempt-group join rule's refusal, or ``None``.

    One rule, two consumers: the live fold raises on it before a join can
    steer the allowance, and the rollover opening projection re-runs it over
    carried rows so an invalid predecessor topology can never group sessions
    the live fold would have refused to group (PSLRETRY round-4 R1).  The
    closed clauses mirror the production seam exactly: a later guard session
    of ONE attempt joins the attempt's FIRST reservation, on the same lane and
    the same accounting axis, only after every earlier member settled, only
    within the lane's declared per-attempt session bound, and never into a
    group whose attempt already refunded.

    Parameters
    ----------
    row:
        The validated joining reservation (its ``attempt_group`` names the head).
    head:
        The head reservation the join names, or ``None`` when absent.
    group_members:
        Spend ids already in the head's group, head first.
    settled_ids:
        Container answering ``spend_id in settled_ids``.
    live_members:
        Members of the group whose current charge still weighs a session.
    """

    spend_id = str(row["spend_id"])
    head_id = reservation_attempt_group(row)
    if head is None:
        return f"paid-session reservation {spend_id} joins absent attempt group {head_id}"
    if reservation_attempt_group(head) is not None:
        return (
            f"paid-session reservation {spend_id} may only join a group HEAD; "
            f"{head_id} is itself a group member"
        )
    lane = str(head["lane"])
    if lane != str(row["lane"]):
        return (
            f"paid-session reservation {spend_id} joins attempt group {head_id} "
            "across lanes"
        )
    bound = LANE_ATTEMPT_SESSION_BOUNDS.get(lane)
    if bound is None:
        return (
            f"paid-session reservation {spend_id} joins attempt group {head_id} on "
            f"lane {lane}, which declares no per-attempt session bound: joins refuse "
            "entirely until the closed bound vocabulary names the lane"
        )
    head_axis, row_axis = head["axis"], row["axis"]
    if (
        str(head_axis["campaign_id"]) != str(row_axis["campaign_id"])
        or str(head_axis["generation_id"]) != str(row_axis["generation_id"])
        or work_generations(head_axis["work_generations"])
        != work_generations(row_axis["work_generations"])
    ):
        return (
            f"paid-session reservation {spend_id} joins attempt group {head_id} "
            "across accounting axes"
        )
    unsettled = [member for member in group_members if member not in settled_ids]
    if unsettled:
        return (
            f"paid-session reservation {spend_id} joins attempt group {head_id} "
            f"while members are unsettled: {', '.join(unsettled)}"
        )
    if len(group_members) + 1 > bound:
        return (
            f"paid-session reservation {spend_id} would be paid session "
            f"{len(group_members) + 1} of attempt group {head_id}: lane {lane} "
            f"bounds one attempt at {bound} paid sessions"
        )
    if live_members == 0:
        return (
            f"paid-session reservation {spend_id} joins refunded attempt group "
            f"{head_id}: a fully provider-declined attempt is closed, and a fresh "
            "session must head a NEW attempt through the admission valve"
        )
    return None


def validate_settlement(row: Mapping[str, Any]) -> None:
    """Fail closed unless a settlement is exact, bounded, and evidenced.

    The cross-row half of the rule (the charge equals
    :func:`charged_wall_seconds` over this settlement's own reservation) lives
    in :meth:`AccountingState.extend`, which is the only place a reservation is
    in scope.
    """

    schema = row.get("schema_version")
    if schema not in {*MACHINE_FLOOR_SETTLEMENT_SCHEMAS, *LEGACY_SETTLEMENT_SCHEMAS}:
        raise ValueError("paid-session settlement schema mismatch")
    if row.get("event") != "session_settled":
        raise ValueError("paid-session settlement event mismatch")
    if not isinstance(row.get("spend_id"), str) or not row.get("spend_id"):
        raise ValueError("paid-session settlement requires spend_id")
    observed = row.get("observed")
    if not isinstance(observed, Mapping):
        raise ValueError("paid-session settlement requires observed weight")
    finite_nonnegative(observed.get("wall_seconds"), field="observed.wall_seconds")
    if row.get("evidence") not in SETTLEMENT_EVIDENCE:
        raise ValueError("paid-session settlement evidence is outside the closed set")
    if not isinstance(row.get("custody_outcome"), str) or not row.get("custody_outcome"):
        raise ValueError("paid-session settlement requires custody_outcome")
    if schema == PAID_SESSION_SETTLEMENT_SCHEMA_V2:
        sessions = observed.get("sessions")
        if isinstance(sessions, bool) or sessions not in {0, 1}:
            raise ValueError("paid-session settlement sessions must be zero or one")
    if schema == PAID_SESSION_SETTLEMENT_SCHEMA_V3:
        _validate_legacy_v3_provider(row)
    if schema not in MACHINE_FLOOR_SETTLEMENT_SCHEMAS:
        return
    if row.get("evidence") != "provider-query":
        raise ValueError("current paid-session settlement requires provider-query evidence")
    if row.get("local_outcome") not in LOCAL_OUTCOMES:
        raise ValueError("paid-session settlement local outcome is outside the closed set")
    provider = row.get("provider")
    # Each rung's provider truth is EXACT on its own original terms: v4 rows
    # never carry adapter_identity, v5 rows always do (D-1 row legibility).
    expected_provider_keys = {"accepted", "attested_wall_seconds", "receipt_sha256"}
    if schema == PAID_SESSION_SETTLEMENT_SCHEMA:
        expected_provider_keys = expected_provider_keys | {"adapter_identity"}
    if not isinstance(provider, Mapping) or set(provider) != expected_provider_keys:
        raise ValueError("current paid-session settlement requires exact provider truth")
    if schema == PAID_SESSION_SETTLEMENT_SCHEMA:
        adapter_identity = provider.get("adapter_identity")
        if adapter_identity is not None and (
            not isinstance(adapter_identity, str) or not adapter_identity.strip()
        ):
            raise ValueError(
                "paid-session provider adapter identity must be a nonempty string "
                "or the disclosed in-process null"
            )
    if not isinstance(provider.get("accepted"), bool):
        raise ValueError("paid-session provider acceptance must be boolean")
    if provider.get("attested_wall_seconds") is not None:
        finite_nonnegative(
            provider.get("attested_wall_seconds"),
            field="provider.attested_wall_seconds",
        )
    if not isinstance(provider.get("receipt_sha256"), str) or not provider.get("receipt_sha256"):
        raise ValueError("paid-session provider truth requires a receipt digest")
    machine = row.get("machine")
    if not isinstance(machine, Mapping) or set(machine) != {"wall_seconds", "source"}:
        raise ValueError("current paid-session settlement requires an exact machine floor")
    if machine.get("source") not in MACHINE_WALL_SOURCES:
        raise ValueError("paid-session machine wall source is outside the closed set")
    if machine.get("wall_seconds") is not None:
        finite_nonnegative(machine.get("wall_seconds"), field="machine.wall_seconds")
    elif machine.get("source") != "writer-unobserved":
        raise ValueError("an absent machine wall floor must declare writer-unobserved")


def _validate_legacy_v3_provider(row: Mapping[str, Any]) -> None:
    """Validate the historical v3 provider block on its own original terms."""

    provider = row.get("provider")
    if not isinstance(provider, Mapping) or set(provider) != {
        "accepted",
        "observed_wall_seconds",
        "receipt_sha256",
    }:
        raise ValueError("legacy paid-session settlement requires exact provider truth")
    if not isinstance(provider.get("accepted"), bool):
        raise ValueError("paid-session provider acceptance must be boolean")
    provider_wall = finite_nonnegative(
        provider.get("observed_wall_seconds"), field="provider.observed_wall_seconds"
    )
    if provider_wall != float(row["observed"]["wall_seconds"]):
        raise ValueError("paid-session observed wall must equal provider-attested wall")
    if not isinstance(provider.get("receipt_sha256"), str) or not provider.get("receipt_sha256"):
        raise ValueError("paid-session provider truth requires a receipt digest")


def settled_session_weight(row: Mapping[str, Any]) -> int:
    """Return the finalized consummated-session weight of one settlement.

    Parameters
    ----------
    row:
        Validated ``session_settled`` row.

    Returns
    -------
    int
        ``1`` exactly when the provider attests it consummated the session.
    """

    schema = row.get("schema_version")
    if schema in {*MACHINE_FLOOR_SETTLEMENT_SCHEMAS, PAID_SESSION_SETTLEMENT_SCHEMA_V3}:
        provider = row.get("provider")
        if not isinstance(provider, Mapping) or not isinstance(provider.get("accepted"), bool):
            raise ValueError("current settlement lacks provider-attested acceptance")
        return int(provider["accepted"])
    if schema == PAID_SESSION_SETTLEMENT_SCHEMA_V2:
        return int(row["observed"]["sessions"])
    return 0 if row.get("custody_outcome") == "reconciled-provider-not-accepted" else 1


def settled_attempt_weight(row: Mapping[str, Any]) -> int:
    """Return one settlement's replay-derived model-attempt weight.

    Provider decline keeps the R4 refund.  A provider-accepted session also
    weighs zero only when the current-schema row carries one exact raw
    machine-owned infrastructure outcome.  The provider session remains fully
    charged by :func:`settled_session_weight`; this helper is used solely by
    the effort-discipline attempt-group projection.

    Parameters
    ----------
    row:
        Validated ``session_settled`` row.

    Returns
    -------
    int
        Zero for provider decline or closed machine infrastructure evidence;
        one for every other settlement.
    """

    if settled_session_weight(row) == 0:
        return 0
    if (
        row.get("schema_version") in MACHINE_FLOOR_SETTLEMENT_SCHEMAS
        and row.get("local_outcome") in ATTEMPT_REFUND_LOCAL_OUTCOMES
    ):
        return 0
    return 1


def settlement_machine_wall(row: Mapping[str, Any]) -> Optional[float]:
    """Return the machine-side wall floor a settlement recorded, if any."""

    machine = row.get("machine")
    if not isinstance(machine, Mapping):
        return None
    value = machine.get("wall_seconds")
    return None if value is None else float(value)


def settlement_provider_wall(row: Mapping[str, Any]) -> Optional[float]:
    """Return the provider's raw attested wall, unclamped, if any."""

    provider = row.get("provider")
    if not isinstance(provider, Mapping):
        return None
    if "attested_wall_seconds" in provider:
        value = provider["attested_wall_seconds"]
        return None if value is None else float(value)
    value = provider.get("observed_wall_seconds")
    return None if value is None else float(value)


def validate_custody_binding(row: Mapping[str, Any]) -> None:
    """Fail closed unless a post-spawn custody binding owns a verified group."""

    schema = row.get("schema_version")
    if schema not in {PAID_SESSION_CUSTODY_SCHEMA, PAID_SESSION_CUSTODY_SCHEMA_V1}:
        raise ValueError("paid-session custody schema mismatch")
    if row.get("event") != "session_custody_bound":
        raise ValueError("paid-session custody event mismatch")
    for name in ("spend_id", "lease_id", "boot_id", "host_id", "leader_start_token"):
        if not isinstance(row.get(name), str) or not row.get(name):
            raise ValueError(f"paid-session custody binding requires {name}")
    for name in ("leader_pid", "leader_pgid"):
        if isinstance(row.get(name), bool) or not isinstance(row.get(name), int):
            raise ValueError(f"paid-session custody binding requires integer {name}")
    if row["leader_pid"] != row["leader_pgid"]:
        raise ValueError("paid-session custody leader must own its process group")
    if schema != PAID_SESSION_CUSTODY_SCHEMA:
        return
    for name in ("lock_device", "lock_inode"):
        if isinstance(row.get(name), bool) or not isinstance(row.get(name), int):
            raise ValueError(f"paid-session custody binding requires integer {name}")


def validate_custody_release(row: Mapping[str, Any]) -> None:
    """Fail closed unless a custody release carries a closed machine outcome."""

    if row.get("schema_version") != PAID_SESSION_CUSTODY_RELEASE_SCHEMA:
        raise ValueError("paid-session custody release schema mismatch")
    if row.get("event") != "session_custody_released":
        raise ValueError("paid-session custody release event mismatch")
    for name in ("spend_id", "lease_id"):
        if not isinstance(row.get(name), str) or not row.get(name):
            raise ValueError(f"paid-session custody release requires {name}")
    if row.get("outcome") not in CUSTODY_RELEASE_OUTCOMES:
        raise ValueError("paid-session custody release outcome is outside the closed set")
    operator = row.get("operator")
    reason = row.get("reason")
    if row.get("outcome") == "operator-drain":
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError("operator custody drain requires a named operator")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("operator custody drain requires a justification")
    elif operator is not None or reason is not None:
        raise ValueError("machine custody releases cannot carry operator drain fields")


def validate_accounting_genesis(row: Mapping[str, Any]) -> None:
    """Validate the quiescent cutover anchor for the legacy operational prefix."""

    if row.get("schema_version") != PAID_SESSION_GENESIS_SCHEMA:
        raise ValueError("paid-session accounting genesis schema mismatch")
    if row.get("event") != "accounting_genesis":
        raise ValueError("paid-session accounting genesis event mismatch")
    if not isinstance(row.get("legacy_operational_sha256"), str):
        raise ValueError("paid-session accounting genesis requires legacy prefix digest")
    if isinstance(row.get("legacy_operational_bytes"), bool) or not isinstance(
        row.get("legacy_operational_bytes"), int
    ):
        raise ValueError("paid-session accounting genesis requires legacy prefix size")
    declared_legacy_settlement_schemas(row)


def declared_legacy_settlement_schemas(row: Mapping[str, Any]) -> frozenset[str]:
    """Validate and return one cutover genesis row's legacy-schema DISCLOSURE.

    The cutover row is the durable place a campaign says "my operational prefix
    predates the machine-floor wall rule, and these exact legacy schemas appear in
    it".  It is DISCLOSURE about the separately pinned prefix, and it licenses
    NOTHING on this ledger: the genesis row must be this ledger's first row, so
    every row here is post-cutover by construction and a legacy settlement refuses
    whatever the declaration says (:meth:`_StateBuilder._admit_settlement_schema`).

    That is the PSLRE5 acct F2 correction.  Treating the declaration as authority
    was the "self-describing label" pattern this module's own docstring condemns
    one level up: the genesis row is written by the same producer, with the same
    unkeyed authority, as the settlements it would license, so minting it cost one
    inserted row -- and because the ladder is monotone rather than strictly
    increasing, a UNIFORM downgrade of every settlement stayed at rank 0 and zeroed
    a whole ledger.  Validation of the declaration remains (a malformed disclosure
    is still a refusal), but no decision reads its value.

    Parameters
    ----------
    row:
        Decoded ``accounting_genesis`` row.

    Returns
    -------
    frozenset[str]
        The declared subset of :data:`LEGACY_SETTLEMENT_SCHEMAS`, possibly empty.
    """

    declared = row.get("legacy_settlement_schemas")
    if declared is None:
        return frozenset()
    if not isinstance(declared, (list, tuple)):
        raise ValueError(
            "paid-session accounting genesis legacy schema declaration must be a sequence"
        )
    named: set[str] = set()
    for value in declared:
        if not isinstance(value, str) or value not in LEGACY_SETTLEMENT_SCHEMAS:
            raise ValueError(
                "paid-session accounting genesis may only declare legacy settlement schemas"
            )
        if value in named:
            raise ValueError(
                "paid-session accounting genesis declares a legacy settlement schema twice"
            )
        named.add(value)
    return frozenset(named)


_ANCHOR_PIN_FIELDS = frozenset({"ledger_identity", "rows", "bytes", "chain_head"})


def validate_accounting_rollover(row: Mapping[str, Any]) -> None:
    """Shape-validate one rollover genesis row, bounding its nested walk.

    The heavy verification -- per-carried-row digests, chain linkage to the
    declared head, the whole-file byte binding, the on-disk predecessor
    authentication, the fold-refusal eligibility rule, and the conservative
    charge derivation -- lives in :meth:`_StateBuilder._apply_rollover`, where
    every replay re-runs it.  The depth/size bounds run HERE, iteratively, so
    even shape validation never recurses through producer-shaped nesting.
    """

    if row.get("schema_version") != PAID_SESSION_ROLLOVER_SCHEMA:
        raise ValueError("paid-session rollover schema mismatch")
    if row.get("event") != "accounting_rollover":
        raise ValueError("paid-session rollover event mismatch")
    for name in ("poisoned_ledger_name", "poisoned_ledger_sha256", "poisoned_chain_head"):
        if not isinstance(row.get(name), str) or not row.get(name):
            raise ValueError(f"paid-session rollover requires {name}")
    sibling_name = str(row["poisoned_ledger_name"])
    if "/" in sibling_name or "\\" in sibling_name or sibling_name in {".", ".."}:
        raise ValueError(
            "paid-session rollover predecessor name must be a plain sibling filename"
        )
    for name in ("poisoned_ledger_bytes", "poisoned_row_count"):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"paid-session rollover requires positive integer {name}")
    pin = row.get("poisoned_anchor")
    if not isinstance(pin, Mapping) or set(pin) != _ANCHOR_PIN_FIELDS:
        raise ValueError(
            "paid-session rollover requires the predecessor's exact anchor pin "
            "(ledger_identity, rows, bytes, chain_head)"
        )
    for name in ("ledger_identity", "chain_head"):
        if not isinstance(pin.get(name), str) or not pin.get(name):
            raise ValueError(f"paid-session rollover anchor pin requires {name}")
    for name in ("rows", "bytes"):
        value = pin.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"paid-session rollover anchor pin requires positive integer {name}"
            )
    if (
        int(pin["rows"]) != int(row["poisoned_row_count"])
        or int(pin["bytes"]) != int(row["poisoned_ledger_bytes"])
        or str(pin["chain_head"]) != str(row["poisoned_chain_head"])
    ):
        raise ValueError(
            "paid-session rollover anchor pin disagrees with the declared "
            "predecessor extent and head"
        )
    carried = row.get("carried_rows")
    if not isinstance(carried, (list, tuple)) or not carried:
        raise ValueError("paid-session rollover requires the carried poisoned rows")
    if len(carried) != row["poisoned_row_count"]:
        raise ValueError("paid-session rollover carried rows disagree with the declared count")
    if any(not isinstance(member, Mapping) for member in carried):
        raise ValueError("paid-session rollover carried rows must be decoded row objects")
    _scan_rollover_extent(carried)


def _scan_rollover_extent(carried: Sequence[Mapping[str, Any]]) -> None:
    """Iteratively refuse a nested rollover walk beyond the closed bounds.

    The scan is an explicit stack, never Python recursion, so the bound check
    itself cannot be driven into ``RecursionError`` by the very nesting it
    exists to refuse.
    """

    total = 0
    stack: list[tuple[Sequence[Any], int]] = [(carried, 1)]
    while stack:
        rows, depth = stack.pop()
        for member in rows:
            total += 1
            if total > ROLLOVER_MAX_CARRIED_ROWS:
                raise ValueError(
                    "paid-session rollover walks more than "
                    f"{ROLLOVER_MAX_CARRIED_ROWS} embedded rows: the nested walk "
                    "is bounded and this row refuses"
                )
            if isinstance(member, Mapping) and member.get("event") == "accounting_rollover":
                if depth + 1 > ROLLOVER_MAX_DEPTH:
                    raise ValueError(
                        f"paid-session rollover nests deeper than {ROLLOVER_MAX_DEPTH} "
                        "generations: the nested walk is bounded and this row refuses"
                    )
                nested = member.get("carried_rows")
                if isinstance(nested, (list, tuple)):
                    stack.append((nested, depth + 1))


# ---------------------------------------------------------------------------
# The incremental accounting state.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountingState:
    """One validated paid-session state machine and its budget projections.

    The state is EXTENDED by appended rows and never rebuilt, so validating and
    indexing a ledger costs one visit per row for the life of the process
    rather than one whole-ledger pass per append.
    """

    #: How many rows this state covers.  The decoded rows themselves belong to
    #: the durability concern's snapshot, so an accounting state never holds a
    #: second copy of the ledger.
    row_count: int = 0
    reservations: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    settlements: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    custody_bindings: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    custody_releases: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    provider_keys: frozenset[str] = frozenset()
    genesis_seen: bool = False
    #: Attempt-group membership: head spend id -> every member spend id, head
    #: first.  A v1 reservation or a v2 head opens its own group; a v2 join
    #: extends the named head's group.  The ATTEMPT is the durable author
    #: budget unit; guard sessions are the paid provider unit.
    attempt_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: Per-group liveness: positive while the attempt still weighs one, ZERO
    #: once the attempt refunded and closed.  Members settle strictly in
    #: sequence, so one member settled at zero attempt weight (provider
    #: decline, or exact machine infrastructure evidence) is the attempt's
    #: TERMINAL disposition: it refunds the whole group -- completed earlier
    #: guard sessions included -- and a refunded group refuses joins.  Raw
    #: paid-session QUOTA/RUNAWAY weights remain separately and fully charged.
    attempt_group_live: Mapping[str, int] = field(default_factory=dict)
    #: Chargeable attempt-group count per ``(campaign_id, stable_id, work_id,
    #: lane)`` -- the projection the author allowance valve reads, and the one
    #: rollover opening charges extend so a rolled-over generation cannot
    #: rebuy the attempts its poisoned predecessor already recorded.
    quota_attempt_groups: Mapping[tuple[str, str, str, str], int] = field(
        default_factory=dict
    )
    #: Disclosure of the rollover genesis, when this ledger carries one:
    #: poisoned-ledger identity, the recomputed refusal, and the conservative
    #: ``rollover_opening_charges`` -- segregated from live charges so a census
    #: never counts a carried reservation as an audited live one.
    rollover: Optional[Mapping[str, Any]] = None
    #: Spend ids the rollover already charged.  A new reservation reusing one
    #: refuses, so a carried identity can never be re-minted on this ledger.
    rollover_spend_ids: frozenset[str] = frozenset()
    #: The ledger directory's own predecessor authenticator, injected by the
    #: durability-aware seam (:func:`menagerie.crawler.paid_session_durability.
    #: preserved_lineage_verifier` via the lifecycle seed).  A rollover row is
    #: admissible ONLY through a fold that holds one: the verifier re-reads the
    #: preserved ``.poisoned-*`` sibling and its anchor from disk on EVERY
    #: replay, so a synthetic chain with no anchored on-disk predecessor is
    #: unmintable (PSLRETRY round-4 R1).  This module stays pure -- it defines
    #: the obligation and calls the injected callable, and a fold WITHOUT one
    #: (the bare pure projections) refuses rollover rows fail-closed.
    lineage_verifier: Optional[Callable[..., None]] = field(
        default=None, compare=False, repr=False
    )
    #: There is deliberately NO field holding the genesis row's legacy-schema
    #: declaration: it licenses nothing (every row of this ledger is post-cutover
    #: by construction), and a decision cannot be steered by state that does not
    #: exist.  The declaration survives only as validated disclosure ABOUT the
    #: separately pinned legacy prefix (PSLRE5 acct F2).
    #: Ladder position of the highest settlement schema already folded, so the
    #: settlement schema can only ever move FORWARD along the closed ladder.
    settlement_schema_rank: int = 0
    charges: Mapping[str, Weight] = field(default_factory=dict)
    quota_all: Mapping[tuple[str, str, str], Weight] = field(default_factory=dict)
    quota_by_lane: Mapping[tuple[str, str, str, str], Weight] = field(default_factory=dict)
    runaway_all: Mapping[tuple[str, str], Weight] = field(default_factory=dict)
    runaway_by_lane: Mapping[tuple[str, str, str], Weight] = field(default_factory=dict)
    campaign: Mapping[str, Weight] = field(default_factory=dict)

    def extend(self, appended: Iterable[Mapping[str, Any]]) -> AccountingState:
        """Return this state extended by newly appended, not-yet-seen rows.

        Parameters
        ----------
        appended:
            Decoded rows that follow :attr:`rows`, in ledger order.

        Returns
        -------
        AccountingState
            A new state; ``self`` is never mutated.
        """

        materialized = list(appended)
        if not materialized:
            return self
        ACCOUNTING_STATS["folds"] += 1
        ACCOUNTING_STATS["rows_validated"] += len(materialized)
        builder = _StateBuilder(self)
        for row in materialized:
            builder.apply(row)
        return builder.finish(len(materialized))


class _StateBuilder:
    """Mutable scratch that folds appended rows into one new immutable state."""

    def __init__(self, base: AccountingState) -> None:
        """Copy the base aggregates once per fold, never once per row."""

        self.reservations = dict(base.reservations)
        self.settlements = dict(base.settlements)
        self.custody_bindings = dict(base.custody_bindings)
        self.custody_releases = dict(base.custody_releases)
        self.provider_keys = set(base.provider_keys)
        self.genesis_seen = base.genesis_seen
        self.attempt_groups = dict(base.attempt_groups)
        self.attempt_group_live = dict(base.attempt_group_live)
        self.quota_attempt_groups = dict(base.quota_attempt_groups)
        self.rollover = base.rollover
        self.rollover_spend_ids = set(base.rollover_spend_ids)
        self.lineage_verifier = base.lineage_verifier
        self.settlement_schema_rank = base.settlement_schema_rank
        self.charges = dict(base.charges)
        self.quota_all = dict(base.quota_all)
        self.quota_by_lane = dict(base.quota_by_lane)
        self.runaway_all = dict(base.runaway_all)
        self.runaway_by_lane = dict(base.runaway_by_lane)
        self.campaign = dict(base.campaign)
        self.base_row_count = base.row_count

    def apply(self, row: Mapping[str, Any]) -> None:
        """Validate and fold exactly one appended lifecycle row."""

        # EVERY schema, EVERY event: the row's own content address is evidence the
        # validator re-derives before anything else reads a field from the body,
        # and its top-level vocabulary is closed before any dispatch chooses which
        # validator judges it.
        validate_row_digest(row)
        validate_row_vocabulary(row)
        event = row.get("event")
        if event == "accounting_genesis":
            validate_accounting_genesis(row)
            if (
                self.genesis_seen
                or self.base_row_count
                or self.reservations
                or self.settlements
                or self.rollover is not None
            ):
                raise ValueError("paid-session accounting genesis must be the first row")
            self.genesis_seen = True
        elif event == "accounting_rollover":
            self._apply_rollover(row)
        elif event == "session_reserved":
            self._apply_reservation(row)
        elif event == "session_settled":
            self._apply_settlement(row)
        elif event == "session_custody_bound":
            self._apply_binding(row)
        elif event == "session_custody_released":
            self._apply_release(row)
        else:
            raise ValueError(f"unknown paid-session lifecycle event {event!r}")

    def _apply_reservation(self, row: Mapping[str, Any]) -> None:
        """Fold one reservation at its conservative pre-settlement weight."""

        validate_reservation(row)
        spend_id = str(row["spend_id"])
        if spend_id in self.reservations or spend_id in self.rollover_spend_ids:
            raise ValueError(f"duplicate paid-session reservation {spend_id}")
        key = str(row["provider_idempotency_key"])
        if key in self.provider_keys:
            raise ValueError(f"paid-session reservation {spend_id} reuses a provider key")
        head_id = reservation_attempt_group(row)
        if head_id is not None:
            self._validate_attempt_group_join(spend_id, head_id, row)
        self.provider_keys.add(key)
        self.reservations[spend_id] = row
        self._charge(
            spend_id,
            Weight(float(row["reserved"]["wall_seconds"]), 1),
        )
        if head_id is None:
            self.attempt_groups[spend_id] = (spend_id,)
            self.attempt_group_live[spend_id] = 1
            self._count_attempt_group(row, +1)
        else:
            # The join rule already refused a refunded (zero-live) group, so a
            # join always extends a group that is still chargeable: the revival
            # a zero-live group previously allowed bypassed the new-attempt
            # valve, and the refund is now FINAL (PSLRETRY round-4 F1).
            self.attempt_groups[head_id] = (*self.attempt_groups[head_id], spend_id)
            self.attempt_group_live[head_id] = self.attempt_group_live[head_id] + 1

    def _validate_attempt_group_join(
        self, spend_id: str, head_id: str, row: Mapping[str, Any]
    ) -> None:
        """Refuse a join whose group cannot be the same author attempt."""

        refusal = attempt_join_refusal(
            row,
            head=self.reservations.get(head_id),
            group_members=self.attempt_groups.get(head_id, ()),
            settled_ids=self.settlements,
            live_members=self.attempt_group_live.get(head_id, 0),
        )
        if refusal is not None:
            raise ValueError(refusal)

    def _count_attempt_group(self, head_row: Mapping[str, Any], delta: int) -> None:
        """Move one group's chargeable-attempt count on every axis it names."""

        axis = head_row["axis"]
        campaign_id = str(axis["campaign_id"])
        lane = str(head_row["lane"])
        for stable_id, work_id in work_generations(axis["work_generations"]):
            key = (campaign_id, stable_id, work_id, lane)
            self.quota_attempt_groups[key] = self.quota_attempt_groups.get(key, 0) + delta

    def _apply_settlement(self, row: Mapping[str, Any]) -> None:
        """Fold one settlement, re-deriving its charge from its own evidence."""

        validate_settlement(row)
        self._admit_settlement_schema(str(row["schema_version"]))
        spend_id = str(row["spend_id"])
        reservation = self.reservations.get(spend_id)
        if reservation is None:
            raise ValueError(f"settlement precedes reservation {spend_id}")
        if spend_id in self.settlements:
            raise ValueError(f"paid-session reservation {spend_id} is already settled")
        reserved = finite_nonnegative(
            reservation["reserved"]["wall_seconds"], field="reserved.wall_seconds"
        )
        observed = finite_nonnegative(
            row["observed"]["wall_seconds"], field="observed.wall_seconds"
        )
        if observed > reserved:
            raise ValueError(f"settlement for {spend_id} exceeds its reservation")
        if row.get("schema_version") in MACHINE_FLOOR_SETTLEMENT_SCHEMAS:
            expected = charged_wall_seconds(
                reserved_wall_seconds=reserved,
                machine_wall_seconds=settlement_machine_wall(row),
                provider_wall_seconds=settlement_provider_wall(row),
            )
            if observed != expected:
                raise ValueError(
                    f"settlement for {spend_id} does not re-derive its own wall charge "
                    "from the recorded machine floor and provider attestation"
                )
        self.settlements[spend_id] = row
        sessions = settled_session_weight(row)
        self._recharge(spend_id, Weight(observed, sessions))
        attempt_weight = settled_attempt_weight(row)
        if attempt_weight == 0:
            # A provider-declined or exactly machine-infrastructure settlement
            # refunds the WHOLE attempt group, not just its own member.  Group
            # members settle strictly in sequence (a join is admitted only
            # after every earlier member settled), so a zero-weight settlement
            # IS the attempt's terminal disposition: the 2026-08-20 storm's
            # two-guard attempts (completed source-request guard, then a 529
            # api_error author/supplement death) consumed their allowance
            # because the completed guard kept the group live forever.  The
            # provider sessions and walls remain charged above, so
            # QUOTA/RUNAWAY still halt an infra storm.
            head_id = reservation_attempt_group(reservation) or spend_id
            live = self.attempt_group_live.get(head_id, 0)
            if live > 0:
                self.attempt_group_live[head_id] = 0
                self._count_attempt_group(self.reservations[head_id], -1)

    def _admit_settlement_schema(self, schema: str) -> None:
        """Refuse a legacy or backward settlement schema on this exact ledger.

        Two independent gates, both structural:

        * a LEGACY schema is never admissible for a row of THIS ledger, because
          every row of this ledger is post-cutover BY CONSTRUCTION: the cutover
          ``accounting_genesis`` row must be the ledger's first row, and it pins
          the pre-machine-floor prefix as a SEPARATE object (path, digest, byte
          count) that this fold never ingests.  So the legacy schemas govern rows
          that are not here, and a settlement on this ledger must be current;
        * the ladder is MONOTONE, so a settlement can never fold below the
          highest schema this ledger already settled with -- defense in depth that
          keeps holding when the ladder grows a new current schema.

        The finding this closes (PSLRE4 F2) rewrote one honest v4 settlement as a
        v1/v2 row: the current-schema wall re-derivation and the provider-attested
        session weight are both keyed on the schema string, so a downgrade skipped
        BOTH and read the charge from a self-describing label instead of evidence.

        PSLRE5 acct F2 then showed why the DECLARATION could not be the gate: the
        genesis row is written by the same producer, with the same unkeyed
        authority, as the rows it would license -- so minting it cost one inserted
        row, and a UNIFORM downgrade of every settlement never moved below the
        monotone ladder's own starting rank, zeroing a whole ledger.  The decision
        is now derived from the ledger's STRUCTURE (a row of this ledger is
        post-cutover) with zero producer input: the declaration grants nothing at
        all, so forging it is worthless, and a uniform downgrade refuses on its
        FIRST row instead of its second.
        """

        rank = SETTLEMENT_SCHEMA_LADDER.index(schema)
        if schema in LEGACY_SETTLEMENT_SCHEMAS:
            raise ValueError(
                f"paid-session settlement schema {schema} predates the machine-floor "
                "wall rule, and every row of this ledger is post-cutover by "
                "construction: a legacy schema can only ever describe the separately "
                "pinned legacy operational prefix, never a row of this ledger"
            )
        if rank < self.settlement_schema_rank:
            raise ValueError(
                f"paid-session settlement schema {schema} is below "
                f"{SETTLEMENT_SCHEMA_LADDER[self.settlement_schema_rank]}, which this "
                "ledger has already settled with: a legacy schema can only ever be a "
                "historical prefix, never a later downgrade"
            )
        self.settlement_schema_rank = rank

    def _apply_binding(self, row: Mapping[str, Any]) -> None:
        """Fold one post-spawn custody binding."""

        validate_custody_binding(row)
        spend_id = str(row["spend_id"])
        reservation = self.reservations.get(spend_id)
        if reservation is None:
            raise ValueError(f"custody binding precedes reservation {spend_id}")
        if spend_id in self.custody_bindings:
            raise ValueError(f"paid-session reservation {spend_id} has duplicate custody")
        if spend_id in self.settlements:
            raise ValueError(f"custody binding follows settlement for {spend_id}")
        if row["lease_id"] != reservation["custody"]["lease_id"]:
            raise ValueError(f"custody lease mismatch for {spend_id}")
        self.custody_bindings[spend_id] = row

    def _apply_release(self, row: Mapping[str, Any]) -> None:
        """Fold one machine-verified or operator-drained custody release."""

        validate_custody_release(row)
        spend_id = str(row["spend_id"])
        binding = self.custody_bindings.get(spend_id)
        if binding is None:
            raise ValueError(f"custody release precedes binding {spend_id}")
        if spend_id in self.custody_releases:
            raise ValueError(
                f"paid-session reservation {spend_id} has duplicate custody release"
            )
        if row["lease_id"] != binding["lease_id"]:
            raise ValueError(f"custody release lease mismatch for {spend_id}")
        self.custody_releases[spend_id] = row

    def _apply_rollover(self, row: Mapping[str, Any]) -> None:
        """Fold the rollover genesis for a fold-refused predecessor ledger.

        Every clause replays from the embedded, digest-bound rows with zero
        producer input, per the four-part waiver doctrine, and the
        predecessor's IDENTITY is anchor-derived rather than declared
        (PSLRETRY round-4 R1 -- a synthetic chain-valid-but-fold-refused row
        sequence could previously mint a fresh, under-charged authority): the
        named whole-file digest must re-derive byte-for-byte from the embedded
        rows through the ledger's own canonical line encoding, the named
        anchor pin must re-derive from the embedded chain, and the injected
        lineage verifier must re-read the preserved ``.poisoned-*`` sibling
        and its commit/intent anchor from disk ON EVERY REPLAY and confirm the
        same digest, extent, and head.  A fold without a lineage verifier (the
        bare pure projections) refuses rollover rows outright.

        1. *Disposition-invariant*: the only effect is CHARGES -- the new
           authority starts more charged, never less, and no reservation,
           custody, or settlement capability is granted.
        2. *Closed proof rule*: eligibility is exactly "the carried rows are an
           authentic chain the accounting fold REFUSES".  A predecessor the
           fold accepts refuses rollover here, on every replay -- so laundering
           a healthy ledger through this row is unrepresentable.
        3. *Replay-derived*: digests, linkage to the declared head, the
           refusal, and the conservative charges are all recomputed from the
           carried rows on every fold; no persisted verdict is read.
        4. *Disclosed*: the poisoned identity, the recomputed refusal, and the
           ``rollover_opening_charges`` are segregated in :attr:`rollover`.

        The charges are the constant-accept conservative reading: every
        carried reservation (settled or not, nested rollovers included) is
        charged as consummated at its full reserved wall, at its own recorded
        axes -- so the cumulative QUOTA and RUNAWAY instruments lose nothing
        across the rollover.  The opening ATTEMPT count re-runs the SAME
        closed join rule the live fold enforces: a carried reservation whose
        join topology is invalid charges as its OWN attempt head, so an
        invalid predecessor topology can only ever raise the opening attempt
        count, never lower it below what its sessions imply.
        """

        validate_accounting_rollover(row)
        if (
            self.genesis_seen
            or self.base_row_count
            or self.reservations
            or self.settlements
            or self.rollover is not None
        ):
            raise ValueError("paid-session accounting rollover must be the first row")
        carried = [dict(member) for member in row["carried_rows"]]
        for member in carried:
            validate_row_digest(member)
            validate_row_vocabulary(member)
        try:
            chain = extend_chain_state(
                carried, scope=PAID_SESSION_CHAIN_SCOPE, require_links=True
            )
        except LedgerChainError as exc:
            raise ValueError(
                f"paid-session rollover carried rows do not chain-verify: {exc}"
            ) from exc
        if chain.head_sha256 != str(row["poisoned_chain_head"]) or chain.rows != int(
            row["poisoned_row_count"]
        ):
            raise ValueError(
                "paid-session rollover carried rows do not reach the declared "
                "poisoned chain head"
            )
        # The whole-file binding is PURE: the ledger writes exactly one
        # canonical JSON line per row, so the preserved file's bytes re-derive
        # from the embedded rows and a declared digest naming any OTHER byte
        # sequence refuses here, before any disk read.
        payload = b"".join(canonical_json_bytes(member) + b"\n" for member in carried)
        if len(payload) != int(row["poisoned_ledger_bytes"]) or stable_hash(
            {"bytes_hex": payload.hex()}
        ) != str(row["poisoned_ledger_sha256"]):
            raise ValueError(
                "paid-session rollover whole-file digest does not re-derive from "
                "the embedded rows: the named predecessor is not the embedded one"
            )
        pin = row["poisoned_anchor"]
        if (
            str(pin["ledger_identity"]) != chain.heads[0]
            or int(pin["rows"]) != chain.rows
            or int(pin["bytes"]) != len(payload)
            or str(pin["chain_head"]) != chain.head_sha256
        ):
            raise ValueError(
                "paid-session rollover anchor pin does not re-derive from the "
                "embedded rows' own chain"
            )
        if self.lineage_verifier is None:
            raise ValueError(
                "paid-session rollover requires the ledger directory's own lineage "
                "authority: this fold has no verifier for the preserved predecessor, "
                "so the rollover row refuses fail-closed"
            )
        try:
            self.lineage_verifier(
                sibling_name=str(row["poisoned_ledger_name"]),
                ledger_sha256=str(row["poisoned_ledger_sha256"]),
                ledger_bytes=int(row["poisoned_ledger_bytes"]),
                rows=chain.rows,
                chain_head=chain.head_sha256,
                ledger_identity=chain.heads[0],
            )
        except ValueError as exc:
            raise ValueError(
                f"paid-session rollover predecessor lineage does not authenticate: {exc}"
            ) from exc
        refusal: Optional[str] = None
        try:
            AccountingState(lineage_verifier=self.lineage_verifier).extend(carried)
        except ValueError as exc:
            refusal = str(exc)
        if refusal is None:
            raise ValueError(
                "paid-session rollover requires a fold-refused predecessor: the "
                "carried rows fold cleanly, so the existing ledger remains the "
                "accounting authority and nothing may reauthor it"
            )
        opening: dict[str, dict[str, Any]] = {}
        for reservation_row in _carried_reservation_rows(carried):
            validate_reservation(reservation_row)
            spend_id = str(reservation_row["spend_id"])
            axis = reservation_row["axis"]
            campaign_id = str(axis["campaign_id"])
            generation_id = str(axis["generation_id"])
            lane = str(reservation_row["lane"])
            weight = Weight(
                finite_nonnegative(
                    reservation_row["reserved"]["wall_seconds"], field="reserved.wall_seconds"
                ),
                1,
            )
            _accumulate(self.campaign, campaign_id, weight)
            _accumulate(self.runaway_all, (campaign_id, generation_id), weight)
            _accumulate(self.runaway_by_lane, (campaign_id, generation_id, lane), weight)
            for stable_id, work_id in work_generations(axis["work_generations"]):
                _accumulate(self.quota_all, (campaign_id, stable_id, work_id), weight)
                _accumulate(
                    self.quota_by_lane, (campaign_id, stable_id, work_id, lane), weight
                )
            charged = opening.setdefault(spend_id, {"wall_seconds": 0.0, "sessions": 0})
            charged["wall_seconds"] += weight.wall_seconds
            charged["sessions"] += weight.sessions
            self.rollover_spend_ids.add(spend_id)
            self.provider_keys.add(str(reservation_row["provider_idempotency_key"]))
        attempt_heads = _conservative_opening_attempt_heads(carried)
        for head_row in attempt_heads:
            self._count_attempt_group(head_row, +1)
        self.rollover = {
            "poisoned_ledger_name": str(row["poisoned_ledger_name"]),
            "poisoned_ledger_sha256": str(row["poisoned_ledger_sha256"]),
            "poisoned_ledger_bytes": int(row["poisoned_ledger_bytes"]),
            "poisoned_chain_head": str(row["poisoned_chain_head"]),
            "poisoned_row_count": int(row["poisoned_row_count"]),
            "poisoned_anchor": dict(pin),
            "fold_refusal": refusal,
            "carried_reservations": len(opening),
            "carried_attempt_groups": len(attempt_heads),
            "rollover_opening_charges": opening,
        }

    def _charge(self, spend_id: str, weight: Weight) -> None:
        """Add one charge to every instrument the reservation's axes name."""

        self.charges[spend_id] = weight
        axis = self.reservations[spend_id]["axis"]
        campaign_id = str(axis["campaign_id"])
        generation_id = str(axis["generation_id"])
        lane = str(self.reservations[spend_id]["lane"])
        _accumulate(self.campaign, campaign_id, weight)
        _accumulate(self.runaway_all, (campaign_id, generation_id), weight)
        _accumulate(self.runaway_by_lane, (campaign_id, generation_id, lane), weight)
        for stable_id, work_id in work_generations(axis["work_generations"]):
            _accumulate(self.quota_all, (campaign_id, stable_id, work_id), weight)
            _accumulate(self.quota_by_lane, (campaign_id, stable_id, work_id, lane), weight)

    def _recharge(self, spend_id: str, weight: Weight) -> None:
        """Replace a reservation's conservative charge with its settled one.

        This is arithmetic on a DERIVED projection, not a subtractive money
        operation: the durable row sequence is append-only and no row is ever
        voided.  Recomputing the same instrument from the same rows produces
        the same number, which is what makes the projection a pure function.
        """

        previous = self.charges[spend_id]
        delta = Weight(
            weight.wall_seconds - previous.wall_seconds,
            weight.sessions - previous.sessions,
        )
        self.charges[spend_id] = weight
        axis = self.reservations[spend_id]["axis"]
        campaign_id = str(axis["campaign_id"])
        generation_id = str(axis["generation_id"])
        lane = str(self.reservations[spend_id]["lane"])
        _accumulate(self.campaign, campaign_id, delta)
        _accumulate(self.runaway_all, (campaign_id, generation_id), delta)
        _accumulate(self.runaway_by_lane, (campaign_id, generation_id, lane), delta)
        for stable_id, work_id in work_generations(axis["work_generations"]):
            _accumulate(self.quota_all, (campaign_id, stable_id, work_id), delta)
            _accumulate(self.quota_by_lane, (campaign_id, stable_id, work_id, lane), delta)

    def finish(self, appended: int) -> AccountingState:
        """Freeze the folded scratch into the next immutable state."""

        return AccountingState(
            row_count=self.base_row_count + appended,
            reservations=self.reservations,
            settlements=self.settlements,
            custody_bindings=self.custody_bindings,
            custody_releases=self.custody_releases,
            provider_keys=frozenset(self.provider_keys),
            genesis_seen=self.genesis_seen,
            attempt_groups=self.attempt_groups,
            attempt_group_live=self.attempt_group_live,
            quota_attempt_groups=self.quota_attempt_groups,
            rollover=self.rollover,
            rollover_spend_ids=frozenset(self.rollover_spend_ids),
            lineage_verifier=self.lineage_verifier,
            settlement_schema_rank=self.settlement_schema_rank,
            charges=self.charges,
            quota_all=self.quota_all,
            quota_by_lane=self.quota_by_lane,
            runaway_all=self.runaway_all,
            runaway_by_lane=self.runaway_by_lane,
            campaign=self.campaign,
        )


def _accumulate(index: dict[Any, Weight], key: Any, weight: Weight) -> None:
    """Add one signed charge to an aggregate index in place."""

    index[key] = index.get(key, Weight()) + weight


def _carried_reservation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Iterable[Mapping[str, Any]]:
    """Yield every reservation a rollover carries, nested rollovers included.

    A twice-rolled-over campaign embeds its first rollover row inside the
    second's carried rows; walking through it keeps EVERY historical
    reservation charged, so repeated rollover can never shed cumulative spend.
    """

    for member in rows:
        if member.get("event") == "session_reserved":
            yield member
        elif member.get("event") == "accounting_rollover":
            nested = member.get("carried_rows")
            if isinstance(nested, (list, tuple)):
                yield from _carried_reservation_rows(
                    [entry for entry in nested if isinstance(entry, Mapping)]
                )


def _conservative_opening_attempt_heads(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return one head row per OPENING attempt, via the closed join rule.

    The rollover opening projection must never charge fewer attempts than the
    live fold's own join rule implies (PSLRETRY round-4 R1: grouping carried
    reservations by their raw ``attempt_group`` field let two sessions joining
    an ABSENT head open as one attempt).  This walk re-runs
    :func:`attempt_join_refusal` over the carried rows in ledger order: a join
    the rule accepts groups exactly as the live fold would have grouped it,
    and a join the rule refuses charges as its OWN attempt head -- strictly
    conservative, because a refused join could never have reduced the live
    count either.  Decline refunds deliberately do NOT reduce the opening
    count (the constant-accept conservative reading), but declines are still
    tracked so the refund-finality clause groups exactly as the live rule
    does.  Recursion into nested rollovers is bounded by the extent scan that
    every rollover row already passed.
    """

    heads: list[Mapping[str, Any]] = []
    reservations: dict[str, Mapping[str, Any]] = {}
    groups: dict[str, list[str]] = {}
    live: dict[str, int] = {}
    membership: dict[str, str] = {}
    settled: set[str] = set()
    for member in rows:
        event = member.get("event")
        if event == "accounting_rollover":
            nested = member.get("carried_rows")
            if isinstance(nested, (list, tuple)):
                heads.extend(
                    _conservative_opening_attempt_heads(
                        [entry for entry in nested if isinstance(entry, Mapping)]
                    )
                )
        elif event == "session_reserved":
            validate_reservation(member)
            spend_id = str(member["spend_id"])
            head_id = reservation_attempt_group(member)
            joined = False
            if head_id is not None and spend_id not in reservations:
                refusal = attempt_join_refusal(
                    member,
                    head=reservations.get(head_id),
                    group_members=tuple(groups.get(head_id, ())),
                    settled_ids=settled,
                    live_members=live.get(head_id, 0),
                )
                if refusal is None:
                    groups[head_id].append(spend_id)
                    live[head_id] += 1
                    membership[spend_id] = head_id
                    joined = True
            if not joined:
                heads.append(member)
                if head_id is None and spend_id not in reservations:
                    groups[spend_id] = [spend_id]
                    live[spend_id] = 1
                    membership[spend_id] = spend_id
            reservations.setdefault(spend_id, member)
        elif event == "session_settled":
            try:
                validate_settlement(member)
                weight = settled_attempt_weight(member)
            except ValueError:
                # A malformed settlement neither settles a member (so a later
                # join own-heads: MORE attempts) nor refunds one.
                continue
            spend_id = str(member["spend_id"])
            if spend_id in reservations and spend_id not in settled:
                settled.add(spend_id)
                if weight == 0:
                    # Mirror the live fold exactly: a zero-weight settlement is
                    # the attempt's TERMINAL disposition and closes the whole
                    # group, so the refund-finality join clause groups carried
                    # rows precisely as the live rule would have grouped them.
                    owner = membership.get(spend_id)
                    if owner is not None and live.get(owner, 0) > 0:
                        live[owner] = 0
    return heads


def index_rows(rows: Sequence[Mapping[str, Any]]) -> AccountingState:
    """Validate and index a complete row sequence from scratch.

    Parameters
    ----------
    rows:
        Decoded paid-session rows in ledger order.

    Returns
    -------
    AccountingState
        The fully folded state.
    """

    return AccountingState().extend(rows)


# ---------------------------------------------------------------------------
# The pure projections.  Each takes rows (or a folded state) and nothing else.
# ---------------------------------------------------------------------------


def charge(rows: Sequence[Mapping[str, Any]], spend_id: str) -> Weight:
    """Purely return one charge from an already-validated row sequence."""

    weight = index_rows(rows).charges.get(spend_id)
    if weight is None:
        raise ValueError(f"unknown paid-session reservation {spend_id}")
    return weight


def quota_spend(
    rows: Sequence[Mapping[str, Any]],
    campaign_id: str,
    stable_id: str,
    work_id: str,
    lane: Optional[str] = None,
) -> Weight:
    """Return cumulative QUOTA spend for one work generation and lane kind."""

    state = index_rows(rows)
    if lane is None:
        return state.quota_all.get((campaign_id, stable_id, work_id), Weight())
    return state.quota_by_lane.get((campaign_id, stable_id, work_id, lane), Weight())


def runaway_spend(
    rows: Sequence[Mapping[str, Any]],
    campaign_id: str,
    generation_id: str,
    lane: Optional[str] = None,
) -> Weight:
    """Return RUNAWAY_BREAKER spend for one open generation and lane kind."""

    state = index_rows(rows)
    if lane is None:
        return state.runaway_all.get((campaign_id, generation_id), Weight())
    return state.runaway_by_lane.get((campaign_id, generation_id, lane), Weight())


def campaign_spend(rows: Sequence[Mapping[str, Any]], campaign_id: str) -> Weight:
    """Return cumulative reporting spend for one campaign without an instrument."""

    return index_rows(rows).campaign.get(campaign_id, Weight())


def quota_attempt_groups(
    rows: Sequence[Mapping[str, Any]],
    campaign_id: str,
    stable_id: str,
    work_id: str,
    lane: str = "author",
) -> int:
    """Return chargeable ATTEMPT GROUPS for one work generation and lane.

    The durable author budget unit: one attempt is one group of guard-session
    reservations (a head plus its ``attempt_group`` joins), chargeable while
    any member's charge still weighs a session.  Rollover opening charges are
    included, so a rolled-over generation keeps its consumed attempts.
    """

    return index_rows(rows).quota_attempt_groups.get(
        (campaign_id, stable_id, work_id, lane), 0
    )


def authorize(
    state: AccountingState,
    *,
    campaign_id: str,
    generation_id: str,
    lane: str,
    members: Sequence[tuple[str, str]],
    policy: BudgetPolicy,
) -> None:
    """Refuse one proposed paid session under the closed instrument set.

    Parameters
    ----------
    state:
        Folded state over the durable rows the caller has already verified.
    campaign_id, generation_id, lane:
        The proposed row's own accounting coordinates.
    members:
        The proposed row's work generations.
    policy:
        Machine-owned limits.

    Raises
    ------
    PaidSessionBudgetExceeded
        Naming the exact instrument and, for QUOTA, the exact coordinate.
    """

    proposed = Weight(policy.reservation_wall_seconds, 1)
    for stable_id, work_id in members:
        spent = state.quota_by_lane.get((campaign_id, stable_id, work_id, lane), Weight())
        if (
            spent.sessions + proposed.sessions > policy.quota_sessions
            or spent.wall_seconds + proposed.wall_seconds > policy.quota_wall_seconds
        ):
            raise PaidSessionBudgetExceeded(
                "QUOTA",
                f"{stable_id}/{work_id} has no remaining cumulative capacity",
                work_generations=((stable_id, work_id),),
            )
    runaway = state.runaway_by_lane.get((campaign_id, generation_id, lane), Weight())
    if (
        runaway.sessions + proposed.sessions > policy.runaway_sessions
        or runaway.wall_seconds + proposed.wall_seconds > policy.runaway_wall_seconds
    ):
        raise PaidSessionBudgetExceeded(
            "RUNAWAY_BREAKER", f"generation {generation_id} has no remaining capacity"
        )
