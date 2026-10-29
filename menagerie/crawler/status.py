"""Terminal partition enforcement and queryable crawler funnel reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.backend import Framework
from menagerie.crawler.constants import (
    ACCESS_BARRIER_REJECTION_CLASS,
    ACCESS_BLOCKED_STATUS_CODE,
    BACKEND_PARITY_DEFERRAL_STATUS_CODE,
    BLOCKED_ARM_STAGES,
    BLOCKED_DISPOSITION_REVIEW_REASON_CODES,
    BLOCKED_PREREQUISITE_REASON_CODE,
    CHECKER_SESSION_EXHAUSTION_REASON_CODE,
    ENVIRONMENT_ADJACENT_STATUS_CODE,
    FailureStage,
    NO_RUNG_SELECTED,
    NO_RUNG_SELECTED_STATUS_KINDS,
    SKIPPED_STATUS_CODES,
    TERMINAL_STATUS_CODES,
    WORKFLOW_STATES,
    OperationalEventKind,
    StatusKind,
)
from menagerie.crawler.checker_lane import (
    CHECKER_LANE_DEFERRAL_DISPOSITION,
    CHECKER_LANE_DEFERRAL_RELATIVE_PATH,
    load_checker_lane_deferral_rows,
)
from menagerie.crawler.disk_floor import (
    DISK_FLOOR_DEFERRAL_DISPOSITION,
    DISK_FLOOR_DEFERRAL_RELATIVE_PATH,
    load_disk_floor_deferral_rows,
)
from menagerie.crawler.environment_coverage import (
    COVERAGE_DEFERRAL_DISPOSITION,
    COVERAGE_DEFERRAL_RELATIVE_PATH,
    load_coverage_deferral_rows,
)
from menagerie.crawler.environment_rebuild import (
    REBUILD_DEFERRAL_DISPOSITION,
    REBUILD_DEFERRAL_RELATIVE_PATH,
    load_rebuild_deferral_rows,
)
from menagerie.crawler.family_templates import family_variant_currency_error
from menagerie.crawler.grant_parks import (
    AWARD_PARK_DISPOSITION,
    GRANT_PARK_RELATIVE_PATH,
    GRANT_PENDING_DISPOSITION,
    SELECTION_EXCLUSION_DISPOSITION,
    grant_park_released,
    load_grant_park_rows,
)
from menagerie.crawler.host_capacity import (
    CAPACITY_DEFERRAL_DISPOSITION,
    CAPACITY_DEFERRAL_RELATIVE_PATH,
    load_capacity_deferral_rows,
)
from menagerie.crawler.scheduler_parks import (
    SCHEDULER_DRAIN_CENSUS_CLASSES,
    SCHEDULER_QUARANTINE_DISPOSITION,
    SCHEDULER_QUARANTINE_RELATIVE_PATH,
    load_scheduler_quarantine_rows,
)
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_DISPOSITION,
    REFUSAL_PARK_RELATIVE_PATH,
    load_refusal_park_rows,
    refusal_park_released,
)
from menagerie.crawler.identity import canonical_json_bytes, row_content_address
from menagerie.crawler.infra_release import (
    derive_infra_terminal_release,
    derive_machine_class_release,
)
from menagerie.crawler.quota_deferral import (
    QUOTA_DEFERRAL_DISPOSITION,
    QUOTA_DEFERRAL_RELATIVE_PATH,
    STORM_RECIDIVIST_DISPOSITION,
    load_quota_deferral_rows,
)
from menagerie.crawler.release_vocabulary import MACHINE_RELEASE_CLASSES
from menagerie.crawler.models import (
    CompletenessReport,
    JsonObject,
    PartitionReport,
)
from menagerie.crawler.work_selection import _award_naming_grant
from menagerie.crawler.wakeup import reduce_wake_episodes, render_wakeup_status

MACHINE_DEFERRAL_DISPOSITIONS = frozenset(
    {
        COVERAGE_DEFERRAL_DISPOSITION,
        CAPACITY_DEFERRAL_DISPOSITION,
        CHECKER_LANE_DEFERRAL_DISPOSITION,
        REBUILD_DEFERRAL_DISPOSITION,
        DISK_FLOOR_DEFERRAL_DISPOSITION,
        GRANT_PENDING_DISPOSITION,
        AWARD_PARK_DISPOSITION,
        SELECTION_EXCLUSION_DISPOSITION,
        REFUSAL_PARK_DISPOSITION,
        SCHEDULER_QUARANTINE_DISPOSITION,
        QUOTA_DEFERRAL_DISPOSITION,
        STORM_RECIDIVIST_DISPOSITION,
    }
)
"""The closed set of machine-derived deferral dispositions the partition accounts.

These are the two deliberate pre-gate admission valves (environment coverage and
host capacity), the checker-lane parking valve, the environment-quarantine
parking valve, and the host-disk-floor parking valve. All five refuse to mint
canonical terminals by design -- a
canonical ``deferred:*`` record requires checker adjudication they do not have
(for the checker-lane valve, adjudication is exactly the thing that failed) --
so their durable intake-extension ledger rows are the ONLY truthful statement
about a withheld model. The terminal partition dereferences those rows: every
in-scope model must be terminal XOR machine-deferral-claimed, and a model with
neither stays a loud :class:`PartitionError` (the tripwire's real target).

The sixth member is different in kind and accounted revision-aware: a
grant-pending park row records a COMPLETED supersession that only an operator
grant may publish over a review-held terminal. When that terminal is
dependency-stale (absent from the projection) the row claims the model exactly
like the valves; when the terminal is still projection-current the row stays a
LIVE pending supersession alongside it; and once ANY newer revision lands the
row reads as consumed/superseded.

The seventh member accounts the models the RECORDING PATH withheld: a
terminal-refusal park row records a terminal the reducer refused to record
(intended append AND minimal fallback both refused), so no canonical record
exists at all. It is accounted revision-aware exactly like the grant park --
the row claims an absent model, coexists LIVE beside the still-current
predecessor a refused supersession recorded, and reads as consumed once any
newer revision lands (any record at all, for a parent-less park). A live
refusal park always forces a never-"complete" close: an operator decision is
owed."""


@dataclass(frozen=True)
class MachineDeferralValve:
    """One machine-deferral valve ledger the terminal partition accounts.

    The registry row binds the valve's short name, the closed disposition set
    its rows may carry, the owning module's canonical relative ledger path, and
    that module's validated reader. Every consumer that enumerates valve
    ledgers (the launch monitor, rung stratification/rung_report, the scheduler
    shadow audit) derives from :data:`MACHINE_DEFERRAL_VALVES` -- four
    hand-maintained copies of this list disagreed in four distinct ways at
    main ``289445e6``, and the omissions made both completion detectors
    provably wrong (a live ``environment-rebuild`` claim was invisible to the
    monitor, so m10502 read as forever in-flight).
    """

    short_name: str
    dispositions: frozenset[str]
    relative_path: str
    load_rows: Callable[[Sequence[Path]], tuple[JsonObject, ...]]

    @property
    def family(self) -> str:
        """``"park"`` for review-held ledgers, ``"deferral"`` otherwise."""

        if all(value.startswith("review-held:") for value in self.dispositions):
            return "park"
        return "deferral"


def _load_quota_deferral_valve_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Adapt the quota-deferral single-path reader to the registry's shape."""

    return tuple(row for path in paths for row in load_quota_deferral_rows(path))


MACHINE_DEFERRAL_VALVES: tuple[MachineDeferralValve, ...] = (
    MachineDeferralValve(
        short_name="environment-coverage",
        dispositions=frozenset({COVERAGE_DEFERRAL_DISPOSITION}),
        relative_path=COVERAGE_DEFERRAL_RELATIVE_PATH.as_posix(),
        load_rows=load_coverage_deferral_rows,
    ),
    MachineDeferralValve(
        short_name="host-capacity",
        dispositions=frozenset({CAPACITY_DEFERRAL_DISPOSITION}),
        relative_path=CAPACITY_DEFERRAL_RELATIVE_PATH.as_posix(),
        load_rows=load_capacity_deferral_rows,
    ),
    MachineDeferralValve(
        short_name="environment-rebuild",
        dispositions=frozenset({REBUILD_DEFERRAL_DISPOSITION}),
        relative_path=REBUILD_DEFERRAL_RELATIVE_PATH.as_posix(),
        load_rows=load_rebuild_deferral_rows,
    ),
    MachineDeferralValve(
        short_name="disk-floor",
        dispositions=frozenset({DISK_FLOOR_DEFERRAL_DISPOSITION}),
        relative_path=DISK_FLOOR_DEFERRAL_RELATIVE_PATH.as_posix(),
        load_rows=load_disk_floor_deferral_rows,
    ),
    MachineDeferralValve(
        short_name="checker-lane",
        dispositions=frozenset({CHECKER_LANE_DEFERRAL_DISPOSITION}),
        relative_path=CHECKER_LANE_DEFERRAL_RELATIVE_PATH.as_posix(),
        load_rows=load_checker_lane_deferral_rows,
    ),
    MachineDeferralValve(
        short_name="grant-parks",
        # R2-1: the selection-exclusion park shares this ledger — the same
        # loader, liveness, and disclosure spine, one more closed disposition.
        dispositions=frozenset(
            {
                GRANT_PENDING_DISPOSITION,
                AWARD_PARK_DISPOSITION,
                SELECTION_EXCLUSION_DISPOSITION,
            }
        ),
        relative_path=GRANT_PARK_RELATIVE_PATH.as_posix(),
        load_rows=load_grant_park_rows,
    ),
    MachineDeferralValve(
        short_name="refusal-parks",
        dispositions=frozenset({REFUSAL_PARK_DISPOSITION}),
        relative_path=REFUSAL_PARK_RELATIVE_PATH.as_posix(),
        load_rows=load_refusal_park_rows,
    ),
    MachineDeferralValve(
        short_name="scheduler-quarantines",
        dispositions=frozenset({SCHEDULER_QUARANTINE_DISPOSITION}),
        relative_path=SCHEDULER_QUARANTINE_RELATIVE_PATH.as_posix(),
        load_rows=load_scheduler_quarantine_rows,
    ),
    MachineDeferralValve(
        short_name="quota-deferrals",
        dispositions=frozenset(
            {QUOTA_DEFERRAL_DISPOSITION, STORM_RECIDIVIST_DISPOSITION}
        ),
        relative_path=QUOTA_DEFERRAL_RELATIVE_PATH.as_posix(),
        load_rows=_load_quota_deferral_valve_rows,
    ),
)
"""THE machine-deferral valve registry: one row per park ledger, all consumers derive."""


def machine_deferral_valve_paths(records_root: Path) -> dict[str, Path]:
    """Return every valve ledger path below one records root, by short name."""

    return {
        valve.short_name: records_root / valve.relative_path
        for valve in MACHINE_DEFERRAL_VALVES
    }


#: Closed stale-axis vocabulary for reasons that are NOT a named dependency
#: axis, mapped from the stable prefix :func:`~menagerie.crawler.authority.validate_currency`
#: and the replay arms emit. Every classified reason must land in exactly one
#: bucket, and an unrecognized reason lands in ``unclassified`` -- never
#: silently dropped, never invented as an axis a reader would take as
#: understood. Widening this table is a deliberate reviewed change.
_STALE_REASON_PREFIXES: tuple[tuple[str, str], ...] = (
    ("dependency-vector: stale ", "<axis>"),
    ("dependency-vector:", "dependency-vector"),
    ("intake:", "intake"),
    ("status-proof:", "status-proof"),
    ("family-authority:", "family-authority"),
    ("legacy-untrusted:", "legacy-untrusted"),
    ("replay-refused:", "replay-refused"),
)

#: The reducer's derivation-mismatch refusal. This -- not
#: ``validate_currency``'s "stale <axis>" -- is the arm a rotation actually
#: trips first, because the replay re-derives the vector from live authority
#: before the currency check runs. It names the differing axes inline as
#: ``<axis>: recorded=... derived=...`` (``reducer._validate_derived_model_authority``,
#: sorted), so the axis is recoverable exactly and never guessed.
_DERIVATION_MISMATCH_MARKER = "dependency vector contradicts reducer derivation"
_DERIVATION_AXIS_PATTERN = re.compile(r"(\w+): recorded=")

#: The bucket a stale reason lands in when the closed table above does not
#: recognize it (a replay ``ReductionError``, a contradicted evidence re-pass,
#: a ``KeyError`` from a missing reference). It fails the partition exactly
#: like a named axis: "we could not classify why this head is not current" is
#: never weaker evidence than "this head's prompt identity moved".
UNCLASSIFIED_STALE_AXIS = "unclassified"

#: Stale bucket for a head the per-record currency check calls CURRENT while
#: the full replay still refused it. The replay is the stronger authority (it
#: re-passes referenced evidence and family currency), so the head stays stale;
#: this label says exactly which authority refused, instead of pretending an
#: identity axis moved. Producers prefix the reason with ``"replay-refused: "``
#: so :func:`stale_axis` classifies it back to this exact bucket (the table
#: above carries the matching prefix).
REPLAY_REFUSED_STALE_AXIS = "replay-refused"


def dependency_axes() -> frozenset[str]:
    """Return the closed dependency-vector axis names.

    Derived from the frozen :class:`~menagerie.crawler.authority.DependencyVector`
    dataclass rather than a hand-copied literal: a new axis cannot exist
    without appearing here, and a retired one cannot linger.

    Returns
    -------
    frozenset[str]
        Every dependency-vector axis name.
    """

    from dataclasses import fields  # noqa: PLC0415 -- lazy: keeps status import-light

    from menagerie.crawler.authority import DependencyVector  # noqa: PLC0415

    return frozenset(field_.name for field_ in fields(DependencyVector))


def closed_stale_classes() -> frozenset[str]:
    """Return the complete closed vocabulary :func:`stale_axis` can emit.

    The union of every dependency-vector axis name and every closed non-axis
    bucket (the prefix table's labels plus :data:`UNCLASSIFIED_STALE_AXIS`).
    Consumers that must bind a stale classification durably (the D-47
    stale-excluded acknowledgment digest) validate against exactly this set,
    so a free-form reason string -- which legitimately embeds revision hashes
    and derived identities -- can never leak into a digest preimage.

    Returns
    -------
    frozenset[str]
        Every value :func:`stale_axis` can return.
    """

    return (
        dependency_axes()
        | frozenset(label for _, label in _STALE_REASON_PREFIXES if label != "<axis>")
        | frozenset((UNCLASSIFIED_STALE_AXIS,))
    )


def stale_axis(reason: str) -> str:
    """Classify one projection stale reason into the closed axis vocabulary.

    Parameters
    ----------
    reason:
        Stale reason string the dependency-currency projection recorded.

    Returns
    -------
    str
        A dependency-vector axis name, a closed non-axis category, or
        :data:`UNCLASSIFIED_STALE_AXIS`.
    """

    text = str(reason).strip()
    for prefix, label in _STALE_REASON_PREFIXES:
        if not text.startswith(prefix):
            continue
        if label != "<axis>":
            return label
        axis = text[len(prefix) :].strip().split(" ", 1)[0]
        return axis if axis in dependency_axes() else UNCLASSIFIED_STALE_AXIS
    if _DERIVATION_MISMATCH_MARKER in text:
        axes = dependency_axes()
        named = sorted(axis for axis in _DERIVATION_AXIS_PATTERN.findall(text) if axis in axes)
        # The message lists every differing axis; the census buckets on the
        # first, and the verbatim reason (which the boundary view prints beside
        # the census) keeps the rest visible. An unparseable mismatch stays in
        # the vector CLASS rather than inventing an axis.
        return named[0] if named else "dependency-vector"
    return UNCLASSIFIED_STALE_AXIS


def stale_census(
    stale_reasons: Mapping[str, str],
    *,
    scope: Optional[Iterable[str]] = None,
) -> dict[str, frozenset[str]]:
    """Bucket stale stable IDs by their closed axis.

    Parameters
    ----------
    stale_reasons:
        Stable-ID keyed projection stale reasons.
    scope:
        Optional membership restriction (in-scope intake IDs).

    Returns
    -------
    dict[str, frozenset[str]]
        Axis name to stale stable IDs, ordered by axis.
    """

    in_scope = None if scope is None else frozenset(scope)
    buckets: dict[str, set[str]] = defaultdict(set)
    for stable_id, reason in stale_reasons.items():
        if in_scope is not None and stable_id not in in_scope:
            continue
        buckets[stale_axis(reason)].add(str(stable_id))
    return {axis: frozenset(stable_ids) for axis, stable_ids in sorted(buckets.items())}


def awards_stale_on_live_basis(
    ledger_heads: Mapping[str, Mapping[str, Any]],
    stale_ids: Iterable[str],
) -> frozenset[str]:
    """Return the ``runs`` ledger heads the live projection cannot vouch for.

    The one census line (T6-opus-13) that makes award staleness visible BEFORE
    a relaunch re-drives the awards: at the step-100 close the campaign's live
    view could vouch for 25 of 67 settled heads and 0 of 15 awards, and nothing
    printed the divergence.

    Parameters
    ----------
    ledger_heads:
        Stable-ID keyed raw ledger heads.
    stale_ids:
        Stable IDs the projection excluded.

    Returns
    -------
    frozenset[str]
        Stale members whose ledger head holds a ``runs`` award.
    """

    stale = frozenset(str(stable_id) for stable_id in stale_ids)
    return frozenset(
        stable_id
        for stable_id in stale
        if str((ledger_heads.get(stable_id) or {}).get("status", {}).get("kind")) == "runs"
    )


class PartitionError(ValueError):
    """Raised when current terminal records do not exactly partition intake."""


class StatusCompletenessError(ValueError):
    """Raised when a terminal status omits its closed required evidence."""


@dataclass(frozen=True)
class _StatusTransition:
    """Completeness policy applied after a code/kind pair is validated."""

    requires_failure_evidence: bool
    forbids_failure_fields: bool
    expects_terminal_issue: bool
    requires_run_completeness: bool


_STATUS_TRANSITIONS: dict[StatusKind, _StatusTransition] = {
    StatusKind.RUNS: _StatusTransition(
        requires_failure_evidence=False,
        forbids_failure_fields=True,
        expects_terminal_issue=False,
        requires_run_completeness=True,
    ),
    StatusKind.DEFERRED: _StatusTransition(
        requires_failure_evidence=False,
        forbids_failure_fields=True,
        expects_terminal_issue=True,
        requires_run_completeness=False,
    ),
    StatusKind.SKIPPED: _StatusTransition(
        requires_failure_evidence=False,
        forbids_failure_fields=True,
        expects_terminal_issue=True,
        requires_run_completeness=False,
    ),
    StatusKind.FAILED: _StatusTransition(
        requires_failure_evidence=True,
        forbids_failure_fields=False,
        expects_terminal_issue=True,
        requires_run_completeness=False,
    ),
    # W13: the disposition-review hold. It carries a closed reason (so failure
    # fields are not forbidden) but no failure stage and no failure-evidence
    # obligation; its own completeness arm below enforces the exact shape.
    StatusKind.BLOCKED: _StatusTransition(
        requires_failure_evidence=False,
        forbids_failure_fields=False,
        expects_terminal_issue=True,
        requires_run_completeness=False,
    ),
}
if set(_STATUS_TRANSITIONS) != set(StatusKind):
    raise RuntimeError("status transition table is not exhaustive")


def wakeup_status(
    events: Sequence[Mapping[str, Any]], *, now: Optional[str] = None
) -> JsonObject:
    """Render usage-pause episode state from canonical operational events.

    Parameters
    ----------
    events:
        Canonical operational events in ledger order.
    now:
        Exact UTC instant to measure liveness against. Omitting it renders every
        liveness field ``unknown`` rather than defaulting an unmeasured window to
        ``0``/healthy (BL-6 (f)).

    Returns
    -------
    dict[str, Any]
        Active/history wake status from the shared episode reducer.
    """

    return render_wakeup_status(reduce_wake_episodes(events), now=now, events=events)


def _records(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Normalize current-record mappings and iterables.

    Parameters
    ----------
    current_records:
        Stable-ID mapping or record iterable.

    Returns
    -------
    list[Mapping[str, Any]]
        Current record values.
    """

    if isinstance(current_records, Mapping):
        return list(current_records.values())
    return list(current_records)


def machine_deferral_rejection(stable_id: str, row: Any) -> Optional[str]:
    """Explain why one candidate valve row cannot account a model, or return None.

    The partition arm dereferences the ACTUAL deferral ledger row rather than
    trusting its caller: the row must carry the model's identity, one of the
    closed machine-derived dispositions, a nonempty recheck hint naming the
    remedy, and the content digest the ledger persisted it under. A row failing
    any of these accounts for nothing -- fail-closed, the model stays missing.

    Parameters
    ----------
    stable_id:
        The model the row is offered as accounting for.
    row:
        Candidate deferral ledger row.

    Returns
    -------
    str | None
        Human-readable rejection, or ``None`` for a row that accounts.
    """

    if not isinstance(row, Mapping):
        return "deferral row is not a mapping"
    if str(row.get("stable_id")) != stable_id:
        return f"deferral row names {row.get('stable_id')!r}, not {stable_id!r}"
    disposition = row.get("disposition")
    if disposition not in MACHINE_DEFERRAL_DISPOSITIONS:
        return f"disposition {disposition!r} is not a machine-derived deferral"
    hint = row.get("recheck_hint")
    if not isinstance(hint, str) or not hint.strip():
        return "deferral row names no recheck remedy"
    digest = row.get("row_sha256")
    if not isinstance(digest, str) or not digest:
        return "deferral row carries no ledger content digest"
    try:
        canonical = dict(json.loads(canonical_json_bytes(dict(row)).decode("utf-8")))
    except (TypeError, ValueError):
        return "deferral row is not canonically serializable"
    # The re-derivation routes through the ONE shared helper rather than spelling its
    # own preimage (PSLRE6 acct F2): a validator with a private exclusion set agrees
    # with the chain only while the body carries no other content-address name, and
    # nothing pinned that. The canonical round-trip above stays, because "not
    # serializable" is a distinct finding from "digest does not match".
    if row_content_address(canonical) != digest:
        return "deferral row digest does not match its payload"
    return None


def _machine_deferral_accounting(
    intake: frozenset[str],
    present: Mapping[str, Sequence[Mapping[str, Any]]],
    machine_deferrals: Optional[Mapping[str, Mapping[str, Any]]],
    stale_ids: Optional[frozenset[str]] = None,
) -> tuple[
    dict[str, frozenset[str]],
    frozenset[str],
    frozenset[str],
    frozenset[str],
    dict[str, frozenset[str]],
    dict[str, str],
]:
    """Split candidate valve rows into claimed buckets, superseded IDs, and rejections.

    Parameters
    ----------
    intake:
        Trusted intake stable IDs.
    present:
        Stable IDs holding a canonical LEDGER HEAD, mapped to every record seen
        for the ID (used for the park-row revision comparisons; more than one
        record is already a duplicate-coverage violation, and consulting them
        ALL keeps the park diagnostics stable under record permutation).

        BD-1: this is the RAW ledger head, not the dependency-currency
        projection. Release was keyed on the projection, so a model whose head
        went stale skipped the release test entirely and its park resurrected
        as live accounting -- 18 of 72 members at the step-100 close, including
        5 refusal parks whose models had terminalized hours earlier and 8
        awards presented as ``deferred:*``. A recorded revision is a fact about
        the ledger, and park release is a fact about that revision.
    machine_deferrals:
        Stable-ID-keyed validated deferral ledger rows, or ``None``.
    stale_ids:
        Members whose ledger head the live projection excluded. A valve row on
        a member whose head is CURRENT is superseded (the model completed
        normally after an earlier deferral); a valve row on a member whose head
        is STALE is neither -- the record stands in history and the row is a
        live statement that the RE-DRIVE is withheld (the environment-rebuild
        deferral after a checker-digest rotation is the worked example). Those
        land in the returned ``stale_deferred`` bucket: still disclosed, still
        counted as a deferral by the close, and no longer a coverage CLAIM.

    Returns
    -------
    tuple
        Disposition-keyed claimed IDs, superseded IDs, LIVE grant-pending IDs,
        LIVE terminal-refusal-parked IDs, disposition-keyed stale-deferred IDs,
        and rejection reasons.
    """

    claimed: dict[str, set[str]] = defaultdict(set)
    stale_deferred: dict[str, set[str]] = defaultdict(set)
    superseded: set[str] = set()
    grant_pending: set[str] = set()
    refusal_parked: set[str] = set()
    rejected: dict[str, str] = {}
    stale = frozenset(stale_ids or ())
    park_kinds = {
        GRANT_PENDING_DISPOSITION: (grant_park_released, grant_pending),
        # An award-supersession park is a review hold (AW-1): the runs head is
        # operator property and the refused re-adjudication waits on an
        # award-naming grant, so it buckets with the grant-pending holds and
        # forces the distinct ``-with-review-holds`` close.
        AWARD_PARK_DISPOSITION: (grant_park_released, grant_pending),
        REFUSAL_PARK_DISPOSITION: (refusal_park_released, refusal_parked),
    }
    for stable_id, row in (machine_deferrals or {}).items():
        rejection = machine_deferral_rejection(stable_id, row)
        if rejection is not None:
            rejected[stable_id] = rejection
            continue
        park_kind = park_kinds.get(str(row.get("disposition")))
        if stable_id in present:
            if park_kind is not None and not any(
                park_kind[0](row, record) for record in present[stable_id]
            ):
                # The park is a PENDING decision, not a stale row: the very
                # revision it recorded still governs, so the pair coexists --
                # the terminal satisfies the partition and the park stays live
                # until a grant, a landed revision, or a void releases it.
                park_kind[1].add(stable_id)
                continue
            if park_kind is None and stable_id in stale:
                # The head stands in history but the live authority cannot
                # vouch for it, so this row is not stale -- it is the live
                # accounting for the withheld re-drive.
                stale_deferred[str(row["disposition"])].add(stable_id)
                continue
            # The canonical record wins: a model that completed normally after
            # an earlier deferral holds a stale row, and treating the pair as a
            # conflict would crash every legitimately recovered model.
            superseded.add(stable_id)
        elif stable_id in intake:
            claimed[str(row["disposition"])].add(stable_id)
            if park_kind is not None:
                park_kind[1].add(stable_id)
    return (
        {disposition: frozenset(ids) for disposition, ids in sorted(claimed.items())},
        frozenset(superseded),
        frozenset(grant_pending),
        frozenset(refusal_parked),
        {disposition: frozenset(ids) for disposition, ids in sorted(stale_deferred.items())},
        rejected,
    )


def partition_report(
    intake_ids: Iterable[str],
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    machine_deferrals: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ledger_heads: Optional[Mapping[str, Mapping[str, Any]]] = None,
    stale_reasons: Optional[Mapping[str, str]] = None,
    stale_accounted_ids: Optional[Iterable[str]] = None,
) -> PartitionReport:
    """Compute exact, pairwise-disjoint current terminal status buckets.

    Parameters
    ----------
    intake_ids:
        Stable IDs in trusted intake.
    current_records:
        Materialized dependency-current model revisions. Duplicate stable IDs
        remain visible.
    machine_deferrals:
        Stable-ID-keyed deferral valve ledger rows for models this run withheld.
        A model with no canonical record at all that a valid row claims is
        accounted as machine-deferred rather than missing; a model HOLDING A
        RECORD and a row is terminal, with the row reported as superseded or,
        for a park the record does not release, as a live pending decision.
    ledger_heads:
        Stable-ID-keyed RAW ledger heads -- the existence authority for
        coverage, park release, and deferral claims (BD-1). Omitting it falls
        back to ``current_records``, which is correct only for callers whose
        records already ARE the raw view; every boundary caller passes the raw
        heads, because the dependency-currency projection is routing data and
        was never an existence authority: a stale head read as NONEXISTENT and
        the step-100 boundary gate reported 24 recorded terminals (7 of them
        awards) as coverage holes.
    stale_reasons:
        Stable-ID-keyed reasons the projection excluded a ledger head. In-scope
        entries become the ``stale:<axis>`` census: coverage counts the member
        (the record exists) and this bucket carries the cause. Every stale
        member NOTHING accounts invalidates the partition.
    stale_accounted_ids:
        Members whose staleness a caller-known DURABLE row already accounts
        (the driver's persisted park/valve rows, which are deliberately not
        run-scoped claims -- see
        :func:`~menagerie.crawler.driver._live_park_rows`). This suppresses the
        stale REFUSAL only; it can never satisfy coverage, claim a member, or
        remove anything from the disclosed census.

    Returns
    -------
    PartitionReport
        Coverage, extras, duplicates, terminal buckets, deferral accounting,
        and the per-axis staleness census.
    """

    intake = frozenset(intake_ids)
    records = _records(current_records)
    buckets_mutable: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    extra: set[str] = set()
    for record in records:
        stable_id = str(record.get("stable_id"))
        code = record.get("status", {}).get("code")
        occurrences[stable_id] += 1
        if stable_id not in intake:
            extra.add(stable_id)
        if code in SKIPPED_STATUS_CODES:
            buckets_mutable[str(code)].add(stable_id)
        elif code not in TERMINAL_STATUS_CODES:
            buckets_mutable[f"invalid:{code}"].add(stable_id)
        else:
            buckets_mutable[str(code)].add(stable_id)
    present_records: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        present_records.setdefault(str(record.get("stable_id")), []).append(record)
    head_records: dict[str, list[Mapping[str, Any]]] = (
        {str(stable_id): [record] for stable_id, record in ledger_heads.items()}
        if ledger_heads is not None
        else present_records
    )
    # Extras stay derived from the projected records exactly as before: the raw
    # head view is an EXISTENCE authority for IN-SCOPE members, never a second
    # foreign-record detector. A subset rung legitimately reads a superset
    # history (the T4-T1 membership carve-out), and the reducer's own
    # construction-time tripwire owns unknown stable IDs.
    present = frozenset(head_records)
    duplicates = frozenset(stable_id for stable_id, count in occurrences.items() if count != 1)
    buckets = {code: frozenset(stable_ids) for code, stable_ids in sorted(buckets_mutable.items())}
    stale_by_axis = stale_census(stale_reasons or {}, scope=intake)
    stale_ids = frozenset(
        stable_id for stable_ids in stale_by_axis.values() for stable_id in stable_ids
    )
    claimed, superseded, grant_pending, refusal_parked, stale_deferred, _rejected = (
        _machine_deferral_accounting(intake, head_records, machine_deferrals, stale_ids)
    )
    claimed_ids = frozenset(
        stable_id for stable_ids in claimed.values() for stable_id in stable_ids
    )
    return PartitionReport(
        intake_ids=intake,
        buckets=buckets,
        missing_ids=intake - present - claimed_ids,
        extra_ids=frozenset(extra),
        duplicate_ids=duplicates,
        machine_deferred=claimed,
        superseded_deferral_ids=superseded,
        grant_pending_ids=grant_pending,
        refusal_parked_ids=refusal_parked,
        stale_by_axis=stale_by_axis,
        stale_deferred=stale_deferred,
        externally_accounted_stale_ids=frozenset(
            str(stable_id) for stable_id in (stale_accounted_ids or ()) if stable_id in stale_ids
        ),
    )


def assert_partition(
    intake_ids: Iterable[str],
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    unrecordable_diagnostics: Optional[Mapping[str, str]] = None,
    machine_deferrals: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ledger_heads: Optional[Mapping[str, Mapping[str, Any]]] = None,
    stale_reasons: Optional[Mapping[str, str]] = None,
    stale_accounted_ids: Optional[Iterable[str]] = None,
) -> PartitionReport:
    """Assert every in-scope model is terminal XOR machine-deferral-claimed.

    Parameters
    ----------
    intake_ids:
        Trusted intake stable IDs.
    current_records:
        Materialized current model revisions.
    unrecordable_diagnostics:
        Optional stable-ID-keyed ``ExcType: message`` summaries for terminals whose
        own append refused. A missing ID is USUALLY not a coverage hole at all but a
        terminal the driver tried and failed to write, and the exception that stopped
        it is the actual defect. Naming it here puts the cause in the failure a human
        reads instead of only in the ``terminal-unrecordable`` operational report.
        Coverage itself is never relaxed by supplying, or omitting, this argument.
    machine_deferrals:
        Stable-ID-keyed deferral valve ledger rows for models this run withheld
        (see :func:`partition_report`). The two admission valves refuse to mint
        canonical terminals BY DESIGN, so their durable rows are the partition's
        only truthful accounting for a withheld model. A model with NO terminal
        and NO valid valve row still raises -- that hole is the tripwire's
        target, and supplying this argument never relaxes it.
    ledger_heads:
        Raw ledger heads: the coverage existence authority (see
        :func:`partition_report`).
    stale_reasons:
        Projection stale reasons. A member whose recorded head the live
        projection excluded is COVERED and STALE: it leaves ``missing`` and
        enters ``stale:<axis>``, and the assertion still REFUSES, naming the
        axis. Supplying this argument can never make a partition pass that
        would otherwise fail -- it only replaces the phantom coverage hole with
        the real cause.

    Returns
    -------
    PartitionReport
        Valid partition report.

    Raises
    ------
    PartitionError
        If coverage is missing/extra/duplicated, a status is not terminal, or a
        member's recorded head is not dependency-current.
    """

    records = _records(current_records)
    report = partition_report(
        intake_ids,
        records,
        machine_deferrals=machine_deferrals,
        ledger_heads=ledger_heads,
        stale_reasons=stale_reasons,
        stale_accounted_ids=stale_accounted_ids,
    )
    invalid = sorted(code for code in report.buckets if code.startswith("invalid:"))
    if not report.valid or invalid:
        unaccounted = report.unaccounted_stale_ids
        stale = ", ".join(
            f"stale:{axis}={sorted(stable_ids & unaccounted)}"
            for axis, stable_ids in sorted(report.stale_by_axis.items())
            if stable_ids & unaccounted
        )
        message = (
            "terminal partition invalid: "
            f"missing={sorted(report.missing_ids)}, extra={sorted(report.extra_ids)}, "
            f"duplicates={sorted(report.duplicate_ids)}, invalid={invalid}"
        )
        if stale:
            message += (
                "; these members hold a RECORDED head the live projection cannot "
                f"vouch for -- {stale}. A recorded terminal is never a coverage "
                "hole: re-derive the boundary under the authority that recorded "
                "it, or register the rotation, then re-close"
            )
        explained = tuple(
            f"{stable_id}: {(unrecordable_diagnostics or {})[stable_id]}"
            for stable_id in sorted(report.missing_ids)
            if stable_id in (unrecordable_diagnostics or {})
        )
        if explained:
            message += "; missing because their terminal append itself failed -- " + "; ".join(
                explained
            )
        rejected = tuple(
            f"{stable_id}: {machine_deferral_rejection(stable_id, (machine_deferrals or {})[stable_id])}"
            for stable_id in sorted(report.missing_ids)
            if stable_id in (machine_deferrals or {})
        )
        if rejected:
            message += (
                "; a machine-deferral row was offered for these but rejected -- "
                + "; ".join(rejected)
            )
        raise PartitionError(message)
    return report


def assert_status_completeness(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> None:
    """Assert that every represented terminal status is internally complete.

    Parameters
    ----------
    current_records:
        Current terminal model revisions.

    Raises
    ------
    StatusCompletenessError
        If a status is not closed or omits required failure evidence.
    """

    failures = _status_completeness_failures(_records(current_records))
    if failures:
        raise StatusCompletenessError(
            f"terminal status completeness invalid: {', '.join(sorted(failures))}"
        )


def _status_completeness_failures(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return closed-status evidence failures without raising.

    Parameters
    ----------
    records:
        Current terminal model revisions.

    Returns
    -------
    list[str]
        Stable-ID-qualified completeness failures.
    """

    failures: list[str] = []
    for record in records:
        stable_id = str(record.get("stable_id"))
        status = record.get("status", {})
        code = status.get("code")
        kind = status.get("kind")
        if code not in TERMINAL_STATUS_CODES or kind != str(code).split(":", 1)[0]:
            failures.append(f"{stable_id}:invalid-code-kind")
            continue
        transition = _STATUS_TRANSITIONS[StatusKind(kind)]
        if transition.requires_failure_evidence:
            stage = str(code).split(":", 1)[1]
            if status.get("stage") != stage or not status.get("reason_code"):
                failures.append(f"{stable_id}:missing-stage-reason")
            if status.get("traceback") is None and status.get("no_traceback_reason") is None:
                failures.append(f"{stable_id}:missing-traceback-disposition")
            if status.get("root_cause_fingerprint") is None:
                failures.append(f"{stable_id}:missing-root-cause")
            if code == "failed:accuracy-gate" and not status.get("human_review", {}).get(
                "required"
            ):
                failures.append(f"{stable_id}:missing-human-review")
        elif transition.forbids_failure_fields and (
            status.get("stage") is not None or status.get("reason_code") is not None
        ):
            failures.append(f"{stable_id}:nonfailure-has-failure-fields")
        elif kind == StatusKind.BLOCKED.value:
            # W13: the disposition-review hold's exact shape -- stage-less, a
            # closed reason from the disposition pair, and a mandatory human
            # review (the hold IS the review request).
            if status.get("stage") is not None:
                failures.append(f"{stable_id}:blocked-carries-failure-stage")
            if status.get("reason_code") not in BLOCKED_DISPOSITION_REVIEW_REASON_CODES:
                failures.append(f"{stable_id}:blocked-reason-not-closed")
            if not status.get("human_review", {}).get("required"):
                failures.append(f"{stable_id}:missing-human-review")
        # "No rung was selected" is only coherent where work genuinely stopped short.
        # A ``runs`` record necessarily selected a rung to be runnable, and every
        # ``skipped:*`` code is an epistemic claim that must carry a checked R5_SKIP with
        # its certified predicate -- so the sentinel appearing under either is a real
        # defect, not a labelling nicety.
        rung = record.get("source_resolution", {}).get("rung")
        if rung == NO_RUNG_SELECTED and kind not in NO_RUNG_SELECTED_STATUS_KINDS:
            failures.append(f"{stable_id}:no-rung-selected-under-{kind}")
        # The access deferral and its evidence are one fact, so they bind in BOTH
        # directions. Omission would let the campaign's only recoverable-by-access
        # terminal be claimed with nothing naming what was unreachable -- unusable for
        # the later access pass the code exists to enable. Over-claim is the more
        # dangerous half: a record carrying a locator the author says it could not read,
        # while terminalising as "no description exists", is the exact false statement
        # this whole change removes. Neither passes.
        # A `failed:*` record asserts nothing about the world -- it records that WE
        # broke, frequently before any conclusion was reached -- so it is outside this
        # check's contract, exactly as it is outside the mandatory-source-link one.
        # Binding it here would also make the failure fallback itself unrecordable,
        # turning one bad model into a second-order failure.
        if kind != StatusKind.FAILED.value and bool(access_barrier_probes(record)) != (
            code == ACCESS_BLOCKED_STATUS_CODE
        ):
            failures.append(
                f"{stable_id}:missing-access-barrier-evidence"
                if code == ACCESS_BLOCKED_STATUS_CODE
                else f"{stable_id}:access-barrier-evidence-under-{code}"
            )
    return failures


def access_barrier_probes(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the candidate probes whose locator the author reports it could not read.

    The barrier set keys on the AUTHOR'S CLAIM, not on the machine's HTTP status. The
    claim is what the terminal asserts, so the claim is what the terminal is bound to.
    The machine's ``probe_outcome``/``http_status`` sit beside it in the same row for a
    checker to compare -- a claimed ``access-barrier`` that probed ``fetched`` is a
    finding -- but they deliberately do not decide this predicate. Bot-walls return 403
    to automated probes constantly, and letting an incidental 403 hard-refuse an
    otherwise honest skip would convert a common, harmless observation into a lost model.

    Parameters
    ----------
    record:
        Current terminal model revision.

    Returns
    -------
    list[Mapping[str, Any]]
        Probe rows the author classified as an access barrier.
    """

    probes = record.get("discovery_probes")
    if not isinstance(probes, list):
        return []
    return [
        probe
        for probe in probes
        if isinstance(probe, Mapping)
        and probe.get("author_claimed_class") == ACCESS_BARRIER_REJECTION_CLASS
    ]


def funnel_counts(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> Mapping[str, int]:
    """Aggregate queryable current-record funnel dimensions.

    Parameters
    ----------
    current_records:
        Current model revisions.

    Returns
    -------
    Mapping[str, int]
        Namespaced count keys for status, rung, framework, metadata, and modes.
    """

    from menagerie.crawler.reducer import (  # noqa: PLC0415 -- lazy: keeps status import-light
        FIDELITY_WAIVER_EXECUTION_STAGES,
        PRE_FIDELITY_FAILURE_STAGES,
    )

    counts: Counter[str] = Counter()
    for record in _records(current_records):
        counts["models:total"] += 1
        counts[f"status:{record['status']['code']}"] += 1
        # W13 (R24 census honesty): a historical failed:accuracy-gate head
        # carrying a terminal-disposition reason is the pre-split mislabel --
        # the accuracy gate never rendered a verdict for it. Derived on every
        # census so the 27 known 200a heads are disclosed, never silently
        # normal; fresh mints of this shape are refused at reduction.
        if (
            record["status"]["code"] == "failed:accuracy-gate"
            and record["status"].get("reason_code") in BLOCKED_DISPOSITION_REVIEW_REASON_CODES
        ):
            counts["accuracy_gate_disposition_mislabel"] += 1
        elif (
            record["status"]["code"] == "failed:accuracy-gate"
            and record.get("accuracy_gate", {}).get("verdict") is None
            and record["status"].get("reason_code") != CHECKER_SESSION_EXHAUSTION_REASON_CODE
        ):
            # Any OTHER accuracy-gate failure with no rendered verdict beyond
            # the closed machine-witnessed exhaustion class: an anomaly the
            # census must surface, never fold into the healthy status counts.
            counts["accuracy_gate_failure_without_rendered_verdict"] += 1
        counts[f"metadata:{record['authored_metadata_state']}"] += 1
        counts[f"rung:{record['source_resolution']['rung']}"] += 1
        counts[f"framework:{record['implementation']['run_framework']}"] += 1
        for mode in record["modes"]["meaningful_modes"]:
            counts[f"mode:{mode}"] += 1
        if (
            record["status"]["kind"] == "runs"
            and not record["completeness"]["source_read_fields_complete"]
        ):
            counts["runs:metadata-pending"] += 1
        # AW-3 census segregation: an operator-restored award is DERIVED from
        # the record's own restoration scar on every census -- no persisted
        # flag -- and is never silently counted among earned awards.
        if record["status"]["kind"] == "runs" and any(
            str(scar).startswith("award-restored-by-operator:")
            for scar in record.get("scar_history", ())
        ):
            counts["runs:restored"] += 1
        if (
            record["status"]["kind"] != "runs"
            and record["fidelity"]["required"]
            and not record["fidelity"]["current"]
            and record["execution"]["accepted_attempt_ids"]
        ):
            counts["forward_observed_but_blocked"] += 1
        # KERNEL_DECISION condition 8: segregate fidelity-waived failure terminals
        # from fidelity-audited ones. Both counters are DERIVED from the record
        # shape on every census -- no persisted waiver field exists. A waived
        # R3/R4 implementation is never counted as audited, and release
        # eligibility (record_is_release_eligible) keeps excluding it.
        fidelity_required = bool(
            record["fidelity"]["required"]
            or record["source_resolution"]["rung"] in {"R3_PORT", "R4_REIMPLEMENT"}
        )
        if record["status"]["kind"] == "failed" and fidelity_required:
            if (
                record["status"]["code"] == "failed:fidelity"
                or record["fidelity"]["gate_id"] is not None
                or record["fidelity"]["verdict"] is not None
            ):
                counts["fidelity_audited_failure"] += 1
            elif (
                record["status"]["stage"] in FIDELITY_WAIVER_EXECUTION_STAGES
                and not record["execution"]["accepted_attempt_ids"]
                and not record["execution"]["current"]
            ):
                counts["fidelity_waived_machine_failure"] += 1
            elif (
                record["status"]["stage"] in PRE_FIDELITY_FAILURE_STAGES
                and not record["execution"]["accepted_attempt_ids"]
                and not record["execution"]["current"]
            ):
                # W8/PD-2 four-part waiver part 4: the pre-execution
                # no-consumer arm's SEGREGATED disclosure. Derived from the
                # record shape on every census -- no persisted waiver field --
                # and never counted beside fidelity-audited outcomes; release
                # eligibility keeps excluding it (non-``runs``).
                counts["fidelity_no_consumer_pre_execution"] += 1
        # BL-13 census segregation, on the same derived-not-persisted rule: a model whose
        # framework this campaign cannot execute is a MACHINERY gap, and counting it beside
        # audited outcomes is what let m7960's working paddle forward read as a model defect.
        # The two counters below are derived from the record on every census.
        if record["status"]["code"] == BACKEND_PARITY_DEFERRAL_STATUS_CODE:
            counts["backend_parity_deferred"] += 1
            declared = record["implementation"]["run_framework"]
            counts[f"backend_parity_deferred:{declared}"] += 1
        elif (
            record["status"]["kind"] == "failed"
            and record["status"]["stage"] == FailureStage.FORWARD.value
            and not record["execution"]["accepted_attempt_ids"]
            and record["implementation"]["run_framework"] != Framework.TORCH.value
        ):
            # The BK-1/B7 shape, counted so it can never be silently normal again: a
            # non-torch model published as a FORWARD failure with no accepted attempt. The
            # honest terminal for a lane this campaign cannot serve is the deferral above.
            counts["non_torch_forward_failure_without_award"] += 1
    return dict(sorted(counts.items()))


def backend_parity_census(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> Mapping[str, frozenset[str]]:
    """Disclose every backend-parity deferral, keyed by the framework that has no lane.

    Part four of the waiver doctrine applied to a deferral: a withheld model is visible in a
    DERIVED census, segregated from records that passed the gate, and release eligibility
    keeps excluding it (``record_is_release_eligible`` already requires
    ``status.kind == "runs"``). The census is recomputed from the records on every call and
    reads no persisted flag, so nothing a producer writes can hide a member.

    Parameters
    ----------
    current_records:
        Current model revisions.

    Returns
    -------
    Mapping[str, frozenset[str]]
        Declared run framework -> stable IDs deferred for backend parity.
    """

    by_framework: dict[str, set[str]] = defaultdict(set)
    for record in _records(current_records):
        if record.get("status", {}).get("code") != BACKEND_PARITY_DEFERRAL_STATUS_CODE:
            continue
        framework = str(record.get("implementation", {}).get("run_framework", "unknown"))
        by_framework[framework].add(str(record.get("stable_id")))
    return {
        framework: frozenset(stable_ids) for framework, stable_ids in sorted(by_framework.items())
    }


#: Closed release-authority census classes for consumed requeue grants.
#: ``award-released`` is AW-1's explicit award reopen; ``review-released`` bound a
#: review-held terminal; ``infra-waiver-released`` bound through the infra-terminal
#: waiver (KERNEL_DECISION 2026-08-21); the four ``machine-released:<class>`` arms
#: are the W14 signed class extensions, one segregated arm per class (doctrine
#: part 4 -- a class release is never counted as reviewed); ``unattributed`` is
#: the fail-closed remainder a recomputation cannot place -- disclosed, never
#: hidden.
GRANT_RELEASE_CENSUS_CLASSES = (
    "award-released",
    *(f"machine-released:{name}" for name in sorted(MACHINE_RELEASE_CLASSES)),
    "review-released",
    "infra-waiver-released",
    "unattributed",
)


def grant_release_census(
    *,
    model_revisions: Sequence[Mapping[str, Any]],
    operational_events: Iterable[Mapping[str, Any]],
    attempt_records: Sequence[Mapping[str, Any]],
    gate_records: Sequence[Mapping[str, Any]],
    valve_rows: Sequence[Mapping[str, Any]] = (),
    blocked_prerequisite_rows: Sequence[Mapping[str, Any]] = (),
    provision_witnesses: Sequence[Mapping[str, Any]] = (),
    invalidation_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, tuple[str, ...]]:
    """Disclose every consumed requeue grant, keyed by its RECOMPUTED release class.

    Part four of the waiver doctrine applied to grant release: an
    infra-waiver-released grant is visible in a DERIVED census class, segregated
    from review-released grants, so a waived release can never be counted as
    reviewed. Part three: the class is recomputed on every call from the bound
    terminal revision, the attempts ledger, and the gates ledger -- the
    ``release_authority`` field a consumption event happens to carry is
    deliberately never read, so nothing a producer writes can steer the census.

    The four W14 machine classes census the same way, one segregated
    ``machine-released:<class>`` arm each, and their precedence is deliberately
    CONSERVATIVE: a consumption whose terminal derives a machine class counts as
    machine-released even when a human review also stood -- counting a reviewed
    release as waived is lawful (disclosed, never upgraded); counting a waived
    release as reviewed is the exact laundering part 4 exists to prevent.

    Parameters
    ----------
    model_revisions:
        Append-only model ledger rows (every revision, not just heads: a
        consumption binds the exact superseded parent revision).
    operational_events:
        Operational events carrying ``REQUEUE_GRANT_CONSUMED`` receipts.
    attempt_records:
        Canonical attempts ledger rows.
    gate_records:
        Canonical gates ledger rows.
    valve_rows, blocked_prerequisite_rows, provision_witnesses, invalidation_events:
        The W14 class-evidence streams (quota-deferral valve rows, AL-6 typed
        seam records, EN-6 rotation-provision witnesses, W9 engine-fault
        invalidation events). Defaults are EMPTY: a caller without a class's
        evidence cannot census that class -- absence fails closed.

    Returns
    -------
    dict[str, tuple[str, ...]]
        :data:`GRANT_RELEASE_CENSUS_CLASSES` -> sorted consumed grant IDs.
    """

    revisions_by_binding: dict[tuple[str, str], Mapping[str, Any]] = {}
    for revision in model_revisions:
        stable_id = str(revision.get("stable_id"))
        record_revision = str(revision.get("record_revision"))
        revisions_by_binding[(stable_id, record_revision)] = revision

    classes: dict[str, set[str]] = {name: set() for name in GRANT_RELEASE_CENSUS_CLASSES}
    for event in operational_events:
        if event.get("event_kind") != OperationalEventKind.REQUEUE_GRANT_CONSUMED.value:
            continue
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        grant_id = str(details.get("grant_id"))
        stable_id = str(details.get("stable_id"))
        source_revision = str(details.get("source_record_revision"))
        bound = revisions_by_binding.get((stable_id, source_revision))
        if bound is None:
            classes["unattributed"].add(grant_id)
            continue
        if _award_naming_grant(details, bound):
            classes["award-released"].add(grant_id)
            continue
        machine_class = derive_machine_class_release(
            bound,
            attempt_records,
            gate_records,
            valve_rows=valve_rows,
            blocked_prerequisite_rows=blocked_prerequisite_rows,
            provision_witnesses=provision_witnesses,
            invalidation_events=invalidation_events,
        )
        if machine_class is not None:
            classes[f"machine-released:{machine_class.release_class}"].add(grant_id)
        elif bound.get("status", {}).get("human_review", {}).get("required") is True:
            classes["review-released"].add(grant_id)
        elif derive_infra_terminal_release(bound, attempt_records, gate_records) is not None:
            classes["infra-waiver-released"].add(grant_id)
        else:
            classes["unattributed"].add(grant_id)
    return {name: tuple(sorted(members)) for name, members in classes.items()}


def scheduler_drain_census_from_ledgers(ledgers: Any) -> dict[str, tuple[str, ...]]:
    """Derive the scheduler-quarantine release census from the ledgers (W12).

    Waiver-doctrine part 4 for the machine drain: the segregated census --
    ``retry-eligible`` / ``held-awaiting-operator-drain`` (the operator-owed
    quarantine debt the close must disclose) / ``machine-drained`` /
    ``remedied-void-released`` -- recomputed from the quarantine ledger and
    the operational events on every call, and read by BOTH production
    consumers (the driver's close currency block and ``crawl status``).

    Parameters
    ----------
    ledgers:
        Ledger paths (``models``/...); the quarantine ledger and operational
        ledger are derived exactly as every canonical consumer derives them.

    Returns
    -------
    dict[str, tuple[str, ...]]
        :data:`~menagerie.crawler.scheduler_parks.SCHEDULER_DRAIN_CENSUS_CLASSES`
        -> sorted ``stable_id::work_id`` member keys.
    """

    from menagerie.crawler.checkpoint import (  # noqa: PLC0415 -- lazy: checkpoint imports status
        canonical_operational_ledger_path,
    )
    from menagerie.crawler.recordio import scan_jsonl  # noqa: PLC0415 -- lazy: import-light
    from menagerie.crawler.scheduler_parks import (  # noqa: PLC0415 -- lazy: sibling valve
        load_scheduler_quarantine_ledger,
        scheduler_drain_census,
        scheduler_quarantine_path,
    )

    records_root = ledgers.models.parent.parent
    operational_path = canonical_operational_ledger_path(ledgers.models)
    events = tuple(scan_jsonl(operational_path)) if operational_path.exists() else ()
    rows = load_scheduler_quarantine_ledger([scheduler_quarantine_path(records_root)])
    return scheduler_drain_census(rows, events)


def grant_release_census_from_ledgers(ledgers: Any) -> dict[str, tuple[str, ...]]:
    """Derive the grant-release census from a campaign's evidence ledgers.

    D-19 (LC-6): :func:`grant_release_census` satisfied waiver-doctrine part 4
    on paper while its ONLY caller was a test -- no production reader ever
    computed it, so the registered infra-terminal waiver had no standing
    disclosure and the doctrine text was false as shipped. This is the one
    ledger-reading assembly both production consumers (the driver's close
    currency block and ``crawl status``) call, so the disclosure can never
    again be "reachable" only from the suite.

    Parameters
    ----------
    ledgers:
        Ledger paths (``models``/``attempts``/``gates``); the operational
        ledger is derived from the models path exactly as every canonical
        consumer derives it.

    Returns
    -------
    dict[str, tuple[str, ...]]
        :data:`GRANT_RELEASE_CENSUS_CLASSES` -> sorted consumed grant IDs.
    """

    from menagerie.crawler.checkpoint import (  # noqa: PLC0415 -- lazy: checkpoint imports status
        canonical_operational_ledger_path,
    )
    from menagerie.crawler.recordio import scan_jsonl  # noqa: PLC0415 -- lazy: import-light

    def rows(path: Path) -> tuple[Mapping[str, Any], ...]:
        return tuple(scan_jsonl(path)) if path.exists() else ()

    operational_path = canonical_operational_ledger_path(ledgers.models)
    records_root = Path(ledgers.models).parent.parent
    valve_path = records_root / QUOTA_DEFERRAL_RELATIVE_PATH
    from menagerie.crawler.release_vocabulary import (  # noqa: PLC0415 -- lazy: import-light
        blocked_prerequisite_record_path,
        blocked_prerequisite_records,
    )

    seam_path = blocked_prerequisite_record_path(records_root)
    # Provision witnesses (EN-6) and engine-fault invalidation events (W9) gain
    # production readers when their emitters land; absent evidence fails closed
    # and those two classes stay human-released, disclosed as such.
    return grant_release_census(
        model_revisions=rows(ledgers.models),
        operational_events=rows(operational_path),
        attempt_records=rows(ledgers.attempts),
        gate_records=rows(ledgers.gates),
        valve_rows=load_quota_deferral_rows(valve_path),
        blocked_prerequisite_rows=blocked_prerequisite_records(seam_path),
    )


#: W31 grant-aging alarm threshold. A live unconsumed grant is an OPEN operator
#: authorization; one that nobody consumed for a week has outlived the leg
#: cadence it was issued into (campaign legs turn over in hours-to-days) and is
#: far more likely stale -- the exact residue that later surfaces as a surprise
#: work generation. Seven days keeps a weekend-spanning issuance quiet while
#: catching the forgotten class; the census is a pure derivation an operator or
#: status assembly calls, never a daemon.
GRANT_AGING_ALARM_SECONDS: int = 7 * 24 * 60 * 60

#: Closed grant-aging census classes. ``aged`` is live, unconsumed, and older
#: than the threshold; ``fresh`` is live and younger; ``age-unknown`` is live
#: but pre-v2 (no ``created_at`` was ever stamped) -- disclosed, never hidden,
#: because an undatable open authorization is at least as suspicious as an old
#: one. Voided and consumed grants are spent history and never listed.
GRANT_AGING_CENSUS_CLASSES = ("aged", "fresh", "age-unknown")


def grant_aging_census(
    grant_ledger_rows: Sequence[Mapping[str, Any]],
    operational_events: Iterable[Mapping[str, Any]],
    *,
    now: Any,
    threshold_seconds: int = GRANT_AGING_ALARM_SECONDS,
) -> dict[str, tuple[str, ...]]:
    """Disclose every LIVE unconsumed requeue grant by age class (W31 alarm).

    A pure replay-derived census: liveness comes from the grant ledger's own
    void rows plus recorded consumption events, age from each row's stamped
    ``created_at``. Nothing here mutates a ledger and no persisted field can
    upgrade a class -- an unparseable timestamp reads as ``age-unknown``, never
    as fresh.

    Parameters
    ----------
    grant_ledger_rows:
        Complete requeue-grant ledger rows (grants AND voids) in append order.
    operational_events:
        Operational events carrying ``REQUEUE_GRANT_CONSUMED`` receipts.
    now:
        Timezone-aware ``datetime`` the ages are measured against.
    threshold_seconds:
        Age past which a live grant discloses as ``aged``.

    Returns
    -------
    dict[str, tuple[str, ...]]
        :data:`GRANT_AGING_CENSUS_CLASSES` -> sorted live grant IDs.
    """

    from datetime import datetime  # noqa: PLC0415 -- lazy: the only datetime use here

    voided = {
        str(row["void_of"]) for row in grant_ledger_rows if "void_of" in row
    }
    consumed = {
        str(event.get("details", {}).get("grant_id"))
        for event in operational_events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    }
    classes: dict[str, set[str]] = {name: set() for name in GRANT_AGING_CENSUS_CLASSES}
    for row in grant_ledger_rows:
        if "void_of" in row:
            continue
        grant_id = str(row.get("grant_id"))
        if grant_id in voided or grant_id in consumed:
            continue
        stamp = row.get("created_at")
        age_seconds: Optional[float] = None
        if isinstance(stamp, str) and stamp.strip():
            try:
                issued = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                issued = None
            if issued is not None and issued.tzinfo is not None:
                age_seconds = (now - issued).total_seconds()
        if age_seconds is None:
            classes["age-unknown"].add(grant_id)
        elif age_seconds > threshold_seconds:
            classes["aged"].add(grant_id)
        else:
            classes["fresh"].add(grant_id)
    return {name: tuple(sorted(members)) for name, members in classes.items()}


def grant_aging_census_from_ledgers(
    ledgers: Any,
    *,
    now: Any,
    threshold_seconds: int = GRANT_AGING_ALARM_SECONDS,
) -> dict[str, tuple[str, ...]]:
    """Derive the grant-aging census from a campaign's canonical ledgers.

    The same one-call assembly shape as :func:`grant_release_census_from_ledgers`
    so production status consumers can surface the alarm without re-deriving
    paths.

    Parameters
    ----------
    ledgers:
        Ledger paths (``models``/...); grant and operational ledgers are
        derived exactly as every canonical consumer derives them.
    now:
        Timezone-aware ``datetime`` the ages are measured against.
    threshold_seconds:
        Age past which a live grant discloses as ``aged``.

    Returns
    -------
    dict[str, tuple[str, ...]]
        :data:`GRANT_AGING_CENSUS_CLASSES` -> sorted live grant IDs.
    """

    from menagerie.crawler.checkpoint import (  # noqa: PLC0415 -- lazy: checkpoint imports status
        canonical_operational_ledger_path,
        canonical_requeue_grants_path,
    )
    from menagerie.crawler.recordio import scan_jsonl  # noqa: PLC0415 -- lazy: import-light

    grants_path = canonical_requeue_grants_path(ledgers.models)
    operational_path = canonical_operational_ledger_path(ledgers.models)
    return grant_aging_census(
        tuple(scan_jsonl(grants_path, validate=False)) if grants_path.exists() else (),
        tuple(scan_jsonl(operational_path)) if operational_path.exists() else (),
        now=now,
        threshold_seconds=threshold_seconds,
    )


class RungRosterError(ValueError):
    """A rung roster cannot be verified against its registered identity."""


@dataclass(frozen=True)
class VerifiedRungRoster:
    """One rung roster verified fail-closed against its registration.

    D-34 (OB-M6): every consumer that answers "how many members does this
    rung have, and who are they" must do so through THIS derivation. The
    tolerant read the launch monitor used (raw ``items.jsonl`` lines,
    unreadable lines skipped) silently yielded 209 of 210 members on a torn
    file -- a wrong denominator in every downstream census, with no signal.
    Here the snapshot loads through the verifying intake reader (manifest
    equality, source digests, name binding) and the registration's own
    ``member_count`` and ``snapshot_sha256`` must both agree, or the read
    refuses typed.
    """

    campaign_id: str
    label: str
    snapshot_id: str
    snapshot_sha256: str
    member_count: int
    member_ids: frozenset[str]


def blocked_prerequisite_injectivity_census(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Disclose every blocked-prerequisite row whose pair loses the blocking stage.

    AL-6 invariant: on every NEW-era mint the ``(status_code, reason_code)``
    pair is injective back to the blocking stage -- ``failed:<stage>`` carries
    ``blocked-prerequisite:<stage>`` with the SAME stage, and
    ``failed:environment-adjacent`` carries ``blocked-prerequisite:<stage>``
    for a real blocked-arm stage. Recomputed from the records on every call;
    no persisted flag participates.

    Parameters
    ----------
    current_records:
        Current model revisions.

    Returns
    -------
    dict[str, tuple[str, ...]]
        ``flagged``: sorted stable IDs whose pair disagrees (a new-era mint
        regression -- the suffixed family did not exist before 2026-08-24);
        ``legacy``: sorted stable IDs still carrying the bare read-side member
        (historical rows, disclosed and never flagged).
    """

    flagged: set[str] = set()
    legacy: set[str] = set()
    suffix_prefix = f"{BLOCKED_PREREQUISITE_REASON_CODE}:"
    for record in _records(current_records):
        status = record.get("status", {})
        reason = status.get("reason_code")
        if not isinstance(reason, str) or not reason.startswith(
            BLOCKED_PREREQUISITE_REASON_CODE
        ):
            continue
        stable_id = str(record.get("stable_id"))
        if reason == BLOCKED_PREREQUISITE_REASON_CODE:
            legacy.add(stable_id)
            continue
        if not reason.startswith(suffix_prefix):
            flagged.add(stable_id)
            continue
        suffix = reason[len(suffix_prefix) :]
        code = status.get("code")
        if code == ENVIRONMENT_ADJACENT_STATUS_CODE:
            if suffix not in BLOCKED_ARM_STAGES:
                flagged.add(stable_id)
        elif code != f"failed:{suffix}":
            flagged.add(stable_id)
    return {"flagged": tuple(sorted(flagged)), "legacy": tuple(sorted(legacy))}


#: The exact persisted spelling of the typed cause a checked BLOCKED verdict
#: stamps on its witnessing attempt (never raised; see ``driver_contracts``).
_AUTHOR_BLOCKED_PREREQUISITE_EXCEPTION_TYPE = (
    "menagerie.crawler.driver_contracts.AuthorBlockedPrerequisite"
)


def author_blocked_prerequisite_terminal_census(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    attempt_records: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Disclose every ``failed:author`` head witnessed by a BLOCKED-verdict cause.

    The AL-6 leg-close tripwire: after the vocabulary change, NO new terminal
    may collapse a checked BLOCKED verdict onto the generic
    ``failed:author``/``malformed-result`` label (pilot: 22 rows did). Run over
    a post-AL-6 campaign ledger, ``flagged`` must be empty; over a historical
    ledger it DISCLOSES the pre-era cohort without rewriting it (era-honest:
    rows are never edited, only counted).

    Parameters
    ----------
    current_records:
        Current model revisions.
    attempt_records:
        Canonical attempts ledger rows (the witnessing causes live here).

    Returns
    -------
    dict[str, tuple[str, ...]]
        ``flagged``: ``failed:author`` heads with a generic reason whose
        witnessing attempt carries the typed ``AuthorBlockedPrerequisite``
        cause; ``escalation``: the legitimate top-tier ``needs-higher-tier``
        arm (its witness carries the cause by design); ``stage_bound``: heads
        whose reason is a blocked-prerequisite family member bound to the
        author stage (honest same-stage terminals).
    """

    witnessed: dict[tuple[str, str], bool] = {}
    for attempt in attempt_records:
        error = attempt.get("error")
        if (
            isinstance(error, Mapping)
            and error.get("exception_type") == _AUTHOR_BLOCKED_PREREQUISITE_EXCEPTION_TYPE
        ):
            witnessed[(str(attempt.get("stable_id")), str(attempt.get("work_id")))] = True
    flagged: set[str] = set()
    escalation: set[str] = set()
    stage_bound: set[str] = set()
    for record in _records(current_records):
        status = record.get("status", {})
        if status.get("code") != "failed:author":
            continue
        key = (str(record.get("stable_id")), str(record.get("work_id")))
        if not witnessed.get(key):
            continue
        stable_id = str(record.get("stable_id"))
        reason = status.get("reason_code")
        if reason == "needs-higher-tier":
            escalation.add(stable_id)
        elif isinstance(reason, str) and reason.startswith(BLOCKED_PREREQUISITE_REASON_CODE):
            stage_bound.add(stable_id)
        else:
            flagged.add(stable_id)
    return {
        "flagged": tuple(sorted(flagged)),
        "escalation": tuple(sorted(escalation)),
        "stage_bound": tuple(sorted(stage_bound)),
    }


def verified_rung_roster(
    records_root: Path,
    *,
    label: Optional[str] = None,
    campaign_id: Optional[str] = None,
) -> VerifiedRungRoster:
    """Load the newest matching registered rung roster, or refuse typed.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    label:
        Optional rung label filter (``step-200``); newest row wins.
    campaign_id:
        Optional campaign filter.

    Returns
    -------
    VerifiedRungRoster
        Verified membership with its attested count and digest.

    Raises
    ------
    RungRosterError
        If no registration matches, the snapshot fails intake verification,
        the snapshot digest disagrees with the registration, or the member
        count disagrees with the registration (a torn ``items.jsonl`` lands
        here, never in a silently smaller roster).
    """

    from menagerie.crawler.intake import (  # noqa: PLC0415 -- lazy: keeps status import-light
        IntakeError,
        load_intake_snapshot,
    )
    from menagerie.crawler.rung_staging import (  # noqa: PLC0415 -- lazy: avoids an import cycle
        load_rung_registrations,
    )

    rows = [
        row
        for row in load_rung_registrations(records_root)
        if (label is None or row.label == label)
        and (campaign_id is None or row.campaign_id == campaign_id)
    ]
    if not rows:
        raise RungRosterError(
            f"no registered rung matches label={label!r} campaign={campaign_id!r}"
        )
    registration = rows[-1]
    try:
        snapshot = load_intake_snapshot(records_root / "intake" / registration.snapshot_id)
    except (KeyError, OSError, ValueError, IntakeError) as exc:
        raise RungRosterError(
            f"rung {registration.label!r} snapshot {registration.snapshot_id!r} "
            f"fails intake verification: {exc}"
        ) from exc
    if snapshot.snapshot_sha256 != registration.snapshot_sha256:
        raise RungRosterError(
            f"rung {registration.label!r} snapshot digest "
            f"{snapshot.snapshot_sha256!r} contradicts its registration "
            f"{registration.snapshot_sha256!r}"
        )
    member_ids = frozenset(item.stable_id for item in snapshot.items)
    if len(member_ids) != int(registration.member_count) or len(snapshot.items) != int(
        registration.member_count
    ):
        raise RungRosterError(
            f"rung {registration.label!r} roster holds {len(snapshot.items)} members "
            f"({len(member_ids)} distinct) but its registration attests "
            f"{registration.member_count}: a torn or rewritten roster must refuse, "
            "never shrink a census denominator"
        )
    return VerifiedRungRoster(
        campaign_id=registration.campaign_id,
        label=registration.label,
        snapshot_id=registration.snapshot_id,
        snapshot_sha256=registration.snapshot_sha256,
        member_count=int(registration.member_count),
        member_ids=member_ids,
    )


@dataclass(frozen=True)
class WaiverDisclosure:
    """One registered waiver's part-4 disclosure binding (LOCKED doctrine).

    Every four-part waiver must be visible in a DERIVED census, segregated
    from records that passed the check, through a PRODUCTION reader -- a
    census only a test computes is not a disclosure (the exact gap D-19
    closed). Each row binds a registered waiver's closed proof rule to its
    derived census and to the non-test entry points that compute it, and the
    reachability tripwire (test_waiver_disclosures) recompiles the binding
    from source on every run.
    """

    waiver_id: str
    proof_rule: str
    census_callable: str
    census_keys: tuple[str, ...]
    production_entry_points: tuple[str, ...]


#: The CLOSED waiver registry: exactly the two instances the doctrine names
#: (torchlens CLAUDE.md, four-part waiver doctrine). Registering a third
#: waiver -- or removing a disclosure binding -- is a deliberate reviewed
#: change that must pass all four doctrine parts; the tripwire pins this
#: tuple by set equality.
WAIVER_DISCLOSURES: tuple[WaiverDisclosure, ...] = (
    WaiverDisclosure(
        waiver_id="fidelity-gate-no-consumer",
        proof_rule=(
            "menagerie.crawler.reducer.CanonicalReducer._fidelity_gate_has_no_consumer"
        ),
        census_callable="menagerie.crawler.status.funnel_counts",
        # W8/PD-2: the signed pre-worker widening added its own SEGREGATED
        # census key beside the execution-failure arm's pair -- three keys,
        # one registered instance, every arm disclosed apart from audited.
        census_keys=(
            "fidelity_waived_machine_failure",
            "fidelity_audited_failure",
            "fidelity_no_consumer_pre_execution",
        ),
        production_entry_points=("menagerie.crawler.cli._status_command",),
    ),
    WaiverDisclosure(
        waiver_id="infra-terminal-grant-release",
        proof_rule="menagerie.crawler.infra_release.derive_infra_terminal_release",
        census_callable="menagerie.crawler.status.grant_release_census_from_ledgers",
        census_keys=GRANT_RELEASE_CENSUS_CLASSES,
        production_entry_points=(
            "menagerie.crawler.cli._status_command",
            "menagerie.crawler.driver.CrawlerDriver._run_locked",
        ),
    ),
    # Third instance (W12, Wave-2 rotation batch): the scheduler-quarantine
    # machine drain. Disposition-invariant (releases toward a re-drive, mints
    # no verdict, the park's history stands); closed proof rule (exact event
    # vocabulary + exhaustion recomputed at the drain's position); replay-
    # derived (budget_restoring_boundaries recomputes from ledger rows on
    # every read -- an ineligible drain row restores nothing no matter who
    # appended it); disclosed (this census, read by both production entry
    # points, with held-awaiting-operator-drain as the debt the close owes).
    WaiverDisclosure(
        waiver_id="scheduler-quarantine-machine-drain",
        proof_rule="menagerie.crawler.scheduler_parks.budget_restoring_boundaries",
        census_callable="menagerie.crawler.status.scheduler_drain_census_from_ledgers",
        census_keys=SCHEDULER_DRAIN_CENSUS_CLASSES,
        production_entry_points=(
            "menagerie.crawler.cli._status_command",
            "menagerie.crawler.driver.CrawlerDriver._run_locked",
        ),
    ),
)


def completeness_report(
    intake_ids: Iterable[str],
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    workflow_states: Sequence[str] = (),
    machine_deferrals: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ledger_heads: Optional[Mapping[str, Mapping[str, Any]]] = None,
    stale_reasons: Optional[Mapping[str, str]] = None,
    stale_accounted_ids: Optional[Iterable[str]] = None,
    needs_recheck: Iterable[str] = (),
) -> CompletenessReport:
    """Compute the final-release partition and crawl-completion gates.

    Parameters
    ----------
    intake_ids:
        Trusted intake stable IDs.
    current_records:
        Current terminal revisions.
    workflow_states:
        Scheduler/operational states still present.
    machine_deferrals:
        Stable-ID-keyed deferral valve ledger rows (see :func:`partition_report`).
        ``complete`` keeps its represented-record meaning; deferred models hold
        no records, so a run whose only shortfall is deferrals reads complete
        here and the caller reports the deferrals alongside.
    ledger_heads, stale_reasons:
        Raw ledger heads and projection stale reasons (see
        :func:`partition_report`). A member holding a head the projection
        excluded is covered-and-stale, so ``complete`` stays False -- the close
        may not read complete over a head the machine cannot vouch for.
    needs_recheck:
        RC-2 recheck census members: acceptances recorded before a soundness
        rotation. They are current and covered, and release-INELIGIBLE until
        re-checked, so they surface as a ``needs_recheck`` issue here.

    Returns
    -------
    CompletenessReport
        Exact partition, issue membership, workflow counts, and funnel totals.
    """

    records = [dict(record) for record in _records(current_records)]
    partition = partition_report(
        intake_ids,
        records,
        machine_deferrals=machine_deferrals,
        ledger_heads=ledger_heads,
        stale_reasons=stale_reasons,
        stale_accounted_ids=stale_accounted_ids,
    )
    issues = _record_completion_issues(records, needs_recheck=needs_recheck)
    workflow_counts = Counter(workflow_states)
    unknown_workflows = set(workflow_counts) - WORKFLOW_STATES
    for workflow in unknown_workflows:
        issues[f"unknown_workflow:{workflow}"].append("<campaign>")
    complete = partition.valid and not issues and not workflow_counts
    return CompletenessReport(
        partition=partition,
        incomplete_by_issue={key: tuple(sorted(value)) for key, value in sorted(issues.items())},
        workflow_counts=dict(sorted(workflow_counts.items())),
        funnel_counts=funnel_counts(records),
        complete=complete,
    )


def checkpoint_consistency_report(
    intake_ids: Iterable[str],
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> CompletenessReport:
    """Validate the represented terminal prefix without requiring crawl completion.

    Missing intake IDs are valid progress at per-environment and daily checkpoint
    cadence. Represented IDs must still be unique, in intake, terminal, and
    internally complete for their terminal status.

    Parameters
    ----------
    intake_ids:
        Trusted full intake stable IDs.
    current_records:
        Current terminal revisions in the represented prefix.

    Returns
    -------
    CompletenessReport
        Prefix partition diagnostics; ``complete`` means checkpoint-consistent,
        not final-release complete.
    """

    records = [dict(record) for record in _records(current_records)]
    partition = partition_report(intake_ids, records)
    issues = _record_completion_issues(records)
    invalid = sorted(code for code in partition.buckets if code.startswith("invalid:"))
    for code in invalid:
        issues[f"partition:{code}"].extend(sorted(partition.buckets[code]))
    consistent = not (
        partition.extra_ids or partition.duplicate_ids or invalid or any(issues.values())
    )
    return CompletenessReport(
        partition=partition,
        incomplete_by_issue={key: tuple(sorted(value)) for key, value in sorted(issues.items())},
        workflow_counts={},
        funnel_counts=funnel_counts(records),
        complete=consistent,
    )


def release_view_members(
    current: Iterable[Mapping[str, Any]],
    current_by_id: Mapping[str, Mapping[str, Any]],
    *,
    needs_recheck: Iterable[str] = (),
) -> list[Mapping[str, Any]]:
    """Return the release view: eligible records minus the recheck census members.

    RC-2 (D.2.3): a ``needs_recheck`` member stays an award and stays current;
    it leaves every release view until re-checked under the live authority.
    """

    excluded = frozenset(str(item) for item in needs_recheck)
    return [
        record
        for record in current
        if str(record.get("stable_id")) not in excluded
        and record_is_release_eligible(record, current_by_id)
    ]


def record_is_release_eligible(
    record: Mapping[str, Any], current_records: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Derive release eligibility without trusting the persisted release flag.

    RC-2: the recheck census's ``needs_recheck`` members are excluded from
    release VIEWS by their callers (:func:`release_view_members`,
    :func:`_record_completion_issues`) -- the exclusion is a census fact, not a
    property of the record, and it never touches currency.

    Parameters
    ----------
    record:
        Candidate current canonical model revision.
    current_records:
        Independently selected current revisions keyed by stable ID.

    Returns
    -------
    bool
        True only when current metadata, gates, execution, and family dependencies hold.
    """

    projected = current_records.get(str(record.get("stable_id")))
    if (
        not isinstance(projected, Mapping)
        or projected.get("record_revision") != record.get("record_revision")
        or record.get("status", {}).get("kind") != "runs"
        or record.get("authored_metadata_state") != "accepted"
        or record.get("status", {}).get("human_review", {}).get("required") is True
    ):
        return False
    source_fields = (
        "taxonomy",
        "external_metadata",
        "website",
        "people_and_origin",
        "dates",
        "citation",
        "licenses",
    )
    if any(record.get(field) is None for field in source_fields):
        return False
    coverage = record.get("evidence", {}).get("coverage", {})
    accuracy = record.get("accuracy_gate", {})
    fidelity = record.get("fidelity", {})
    execution = record.get("execution", {})
    meaningful = set(record.get("modes", {}).get("meaningful_modes", []))
    outcomes = set(record.get("modes", {}).get("per_mode_run", {}))
    return bool(
        coverage.get("all_agent_fields_have_support") is True
        and coverage.get("family_grounding_complete") is True
        and coverage.get("missing_support") == []
        and accuracy.get("current") is True
        and accuracy.get("verdict") == "accurate"
        and accuracy.get("gate_id")
        and accuracy.get("vet_identity")
        and (not fidelity.get("required") or fidelity.get("current") is True)
        and execution.get("current") is True
        and meaningful
        and meaningful == outcomes
        and family_variant_currency_error(record, current_records) is None
    )


def _record_completion_issues(
    records: Sequence[Mapping[str, Any]], *, needs_recheck: Iterable[str] = ()
) -> dict[str, list[str]]:
    """Return status-appropriate completeness issues for represented records.

    ``runs`` records must satisfy every runnable-release gate. Honest terminal
    failures, skips, and deferrals are complete as terminal outcomes even though
    they intentionally remain ineligible for the runnable release view.

    Parameters
    ----------
    records:
        Current terminal model revisions.
    needs_recheck:
        RC-2 recheck census members (see :func:`record_is_release_eligible`).

    Returns
    -------
    dict[str, list[str]]
        Issue name to affected stable IDs.
    """

    issues: dict[str, list[str]] = defaultdict(list)
    base_required = ("schema_valid", "mandatory_source_present")
    run_required = (
        "source_read_fields_complete",
        "evidence_coverage_complete",
        "accuracy_gate_current",
        "required_fidelity_current",
        "execution_current",
        "family_template_valid",
        "release_eligible",
    )
    current_by_id = {str(record.get("stable_id")): record for record in records}
    # RC-2: a needs_recheck member is current and covered, and release-
    # INELIGIBLE until re-checked under the live authority; disclosed under
    # its own issue key, segregated from derived ineligibility.
    awaiting_recheck = sorted(frozenset(str(item) for item in needs_recheck) & set(current_by_id))
    if awaiting_recheck:
        # Only create the key when it has members: ``complete`` is
        # ``not issues``, and an empty key would read every clean close as
        # incomplete (the 2026-09-01 ``terminal-partition-complete`` slip).
        issues["needs_recheck"].extend(awaiting_recheck)
    for failure in _status_completeness_failures(records):
        stable_id, issue = failure.split(":", 1)
        issues[f"status:{issue}"].append(stable_id)
    for record in records:
        stable_id = str(record["stable_id"])
        status = record.get("status", {})
        status_code = str(status.get("code"))
        kind = status.get("kind")
        try:
            transition = _STATUS_TRANSITIONS[StatusKind(kind)]
        except (TypeError, ValueError):
            transition = None
        if status.get("human_review", {}).get("required") is True:
            issues["human_review_pending"].append(stable_id)
        completeness = record.get("completeness", {})
        required_true = (
            (*base_required, *run_required)
            if transition is not None and transition.requires_run_completeness
            else base_required
        )
        for field in required_true:
            if not completeness.get(field, False):
                issues[field].append(stable_id)
        expected_terminal_issues = (
            {status_code} if transition is not None and transition.expects_terminal_issue else set()
        )
        for issue in completeness.get("issues", []):
            if str(issue) not in expected_terminal_issues:
                issues[f"record:{issue}"].append(stable_id)
        if transition is not None and transition.requires_run_completeness:
            meaningful = set(record.get("modes", {}).get("meaningful_modes", []))
            outcomes = set(record.get("modes", {}).get("per_mode_run", {}))
            if meaningful != outcomes:
                issues["meaningful_mode_outcomes_incomplete"].append(stable_id)
            family_error = family_variant_currency_error(record, current_by_id)
            if family_error is not None:
                issues["stale_family_variant"].append(stable_id)
            elif not record_is_release_eligible(record, current_by_id):
                issues["derived_release_ineligible"].append(stable_id)
    return issues
