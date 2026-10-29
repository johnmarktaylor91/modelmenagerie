"""Stratified rung composition: tail-class salting for every ladder rung.

JMT directive (2026-08-11): every ladder rung from step-200 onward is
STRATIFIED -- deliberately salted with tail-class representatives so surprise
classes surface at rung width, never at full-run width. This module composes
one rung's member list from the campaign's records state, read-only:

* the campaign partition (scheduling order, via the registered root snapshot);
* the models ledger heads (current revisions through the real parentage
  selector);
* the machine-deferral valve rows and the TK1 attempt refusal fields; and
* the durable rung registry (which zoos prior rungs already sampled).

The output is a deterministic member list in the ``--stable-ids-file`` shape
:mod:`menagerie.crawler.tools.stage_rung` consumes, plus a JSON report naming
every member's stratum and the evidence that put it there. The report carries a
COMPOSITION ATTESTATION (B-03) -- campaign, member/stratum pairs, and the
snapshot identity those members compose to -- which the staging door re-derives
from the stage and refuses on mismatch, so a member list edited after
composition, or a snapshot sha carried forward by hand, cannot be staged.

Rung membership is CUMULATIVE (C-2): a successor derives from the campaign's
registry tip and strictly widens it, so the composition CARRIES the tip's
members forward as the disclosed ``carried-tip`` stratum and ``--size`` buys the
NEW width on top of it. Carrying is mandatory, so ``carried-tip`` takes no share
and cannot be acknowledged away; it is never backfilled from ladder bulk (bulk
means "next FRESH partition row" and still does); and the report names the tip,
its members, and every award authority the staging door will demand for them --
a carried award is operator property exactly like a freshly staged one.

Strata (shares configurable; defaults below):

* ``ladder-bulk`` (~70%) -- the next fresh partition rows in scheduling order,
  deduplicated against already-terminal runs-holders and all prior staged
  state (a converted model is never re-crawled).
* ``r2-vendored`` -- the vendored-closure class (recbole-with-shims style),
  detected from heads whose ``source_resolution`` selected or attempted
  ``R2_VENDOR``, plus fresh members of zoos that produced such resolutions.
* ``oddball-intent`` -- models routing to the oddball/lockless environment
  intents, included to PROVE the coverage valves park them honestly at width.
* ``cuda-deferred`` -- prior ``deferred:needs-cuda`` heads.
* ``content-refusal`` -- models whose attempt history carries the TK1
  ``error.failure_class == "content-refusal"`` classification; these measure
  the author-model refusal rate rung over rung.
* ``unsampled-zoo`` -- zoos never sampled by a prior registered rung,
  round-robined across distinct zoos to maximize zoo diversity.
* ``hard-tail`` -- prior ``failed:author`` / ``failed:import`` heads, the
  hard-tail difficulty sample.

A stratum with a positive share and ZERO candidates REFUSES with a typed note
rather than silently reweighting: surprise-avoidance is the whole point, and a
vanished stratum is information. A short (non-empty) stratum takes what exists
and the gap backfills from ladder bulk, disclosed in the report. Selection is
deterministic given (records state, seed, size).
"""

from __future__ import annotations

import fcntl
import hashlib
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    HOST_CAPABILITY_DEFERRAL_STATUS_CODE,
    framework_from_markers,
)
from menagerie.crawler.intake import IntakeItem, load_intake_snapshot
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import LedgerCorruptionError, TornTailError, scan_jsonl
from menagerie.crawler.routing import route_model
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    fsync_directory,
    row_content_address,
    stamp_row_digest,
)
from menagerie.crawler.partitioner import CampaignPartitionError, load_campaign_bindings
from menagerie.crawler.rung_staging import (
    CARRIED_TIP_PROVENANCE,
    CARRIED_TIP_STRATUM,
    NO_CARRIED_TIP_PROVENANCE,
    RungCompositionAttestation,
    RungRegistration,
    RungStagingError,
    build_composition_attestation,
    lineage_tip_membership,
    load_rung_registrations,
    partition_member_stable_ids,
    planned_subset_identity,
    required_award_authorities,
    resolve_verified_root,
    rung_registry_path,
)
from menagerie.crawler.state import _select_current
from menagerie.crawler.status import (
    machine_deferral_rejection,
    machine_deferral_valve_paths,
)

#: v3 adds the REQUIRED ``composition_attestation`` (B-03): the report now
#: carries what the composer attested -- members, strata, and the snapshot
#: identity they compose to -- so ``stage_rung`` can refuse a stage the
#: composition does not describe instead of trusting a hand-carried sha.
#: v4 adds the REQUIRED ``carried_tip`` and ``award_authorities_required``
#: blocks (C-2): a composition is CUMULATIVE, so it carries the lineage tip's
#: members explicitly and discloses both the carry and the award authorities the
#: staging door will demand for it.
STRATIFIED_RUNG_PLAN_SCHEMA_VERSION = "menagerie.crawler.stratified-rung-plan.v4"

LADDER_BULK_STRATUM = "ladder-bulk"

#: Tail strata in ASSIGNMENT order: a model matching several classes belongs to
#: the first one here, so the report never double-counts a representative.
TAIL_STRATA: tuple[str, ...] = (
    "r2-vendored",
    "oddball-intent",
    "cuda-deferred",
    "content-refusal",
    "unsampled-zoo",
    "hard-tail",
)

#: Default stratum shares per the 2026-08-11 directive: ~70% ladder bulk, the
#: remainder spread evenly across the six tail classes.
DEFAULT_SHARES: Mapping[str, float] = {
    LADDER_BULK_STRATUM: 0.70,
    "r2-vendored": 0.05,
    "oddball-intent": 0.05,
    "cuda-deferred": 0.05,
    "content-refusal": 0.05,
    "unsampled-zoo": 0.05,
    "hard-tail": 0.05,
}

#: Environment intents whose members are salted to prove the coverage valves
#: park them honestly at rung width (a few per rung, never many).
#:
#: Re-adjudicated 2026-08-19: the step-200 rotations landed real locked families
#: for ``mmlab`` and ``oddballs`` (commit 7aa4bfe5 rotated both through the real
#: env lane with staged members), so they GRADUATED out of the salted set -- a
#: locked intent's members ride the ordinary environment path and no longer need
#: salting to prove the valves park them. The set stays exactly the on-disk
#: lockless intents (``envs/<intent>/locks/<target>.lock`` absent), which
#: ``test_rung_boundary_protocol.py::
#: test_salted_oddball_intents_match_the_lockless_intents_on_disk`` pins.
DEFAULT_ODDBALL_INTENTS: frozenset[str] = frozenset(
    {"tf-keras-arm64", "jax-flax-arm64"}
)

_R2_RUNG = "R2_VENDOR"
_RUNS_STATUS = "runs"
_CUDA_DEFERRED_STATUS = "deferred:needs-cuda"

HOST_CAPABILITY_DEFERRED_STATUS = HOST_CAPABILITY_DEFERRAL_STATUS_CODE
"""The machine-capability deferral a bigger-RAM host discharges.

Exposed so step-200 composition can select the cohort EXPLICITLY -- by pinned member
list or ``--force-member`` -- and see it in the strata report through
:meth:`_StratumPools.host_capability_deferred_pool`.

It deliberately does NOT become a share-bearing tail stratum here. Adding one would
change ``DEFAULT_SHARES`` (and therefore the composition of every future rung) as a side
effect of a settlement fix, and would make ``EmptyStratumError`` fire on every rung
composed while no host-capability head exists. Which rung first spends width on this
cohort, and how much, is a step-200 composition decision with an operator-declared share.
"""
_HARD_TAIL_STATUSES = frozenset({"failed:author", "failed:import"})
_REFUSAL_FAILURE_CLASS = "content-refusal"


class StratificationError(ValueError):
    """Raised when a stratified rung cannot be composed from the records state."""


class EmptyStratumError(StratificationError):
    """Raised when a stratum with a positive share has zero candidates.

    Silently reweighting a vanished stratum would hide exactly the surprise
    the salting exists to surface, so composition refuses instead. An operator
    who has investigated acknowledges explicitly with a zero share.
    """

    def __init__(self, stratum: str) -> None:
        """Name the vanished stratum in a typed, actionable refusal.

        The remedy names the ONE gesture that actually works. ``--share
        <stratum>=0`` alone does not: shares must still sum to one, so zeroing a
        stratum without re-weighting the rest earns a second refusal.
        """

        self.stratum = stratum
        super().__init__(
            f"stratum {stratum!r} has a positive share but zero candidates in the "
            "current records state. A vanished stratum is information -- "
            "investigate what emptied it, or acknowledge it explicitly with "
            f"--acknowledge-empty-stratum {stratum} (which zeroes its share and "
            "reassigns the reclaimed share to ladder-bulk, disclosed in the "
            f"report). Passing --share {stratum}=0 alone is NOT the "
            "acknowledgement: the shares must still sum to 1.0."
        )


class StratumNotEmptyError(StratificationError):
    """Raised when a live stratum is offered as an empty-stratum acknowledgement.

    Acknowledging a stratum that still holds candidates would quietly drop a
    live tail class under cover of the refusal remedy, which is the opposite of
    what the salting exists to do. Re-weighting a live stratum stays an explicit
    ``shares`` decision.
    """

    def __init__(self, stratum: str, pool_size: int) -> None:
        """Name the stratum and the candidate count that disqualifies it."""

        self.stratum = stratum
        self.pool_size = pool_size
        super().__init__(
            f"stratum {stratum!r} still holds {pool_size} candidates, so it cannot "
            "be acknowledged as empty. --acknowledge-empty-stratum only applies to "
            "a stratum with zero candidates; to deliberately shrink a live "
            "stratum, set its share explicitly and re-weight the rest."
        )


class MembershipReconciliationError(StratificationError):
    """Raised when canonical records exist outside the staging universe.

    Two censuses guard rung composition (C-ERNIE / T2-14). ``foreign``:
    stable ids with canonical records (model head, attempt, valve claim, or
    prior-rung membership) that belong to NO frozen campaign's universe --
    such a model can never be staged again, so its records silently fall out
    of every future rung. The four campaigns share one records root by design,
    so a record-holder belonging to a SIBLING campaign is accounted (disclosed
    as ``records_outside_this_campaign``), never foreign: the first rung of
    every campaign after the first used to refuse outright, listing every
    model ever crawled (CAT-02/F4), which is a census reduced to noise at the
    exact moment a REAL foreign id would hide in the flood.
    ``unregistered``: THIS campaign's partition members holding canonical
    records while belonging to no registered rung -- their history is
    invisible to prior-rung dedup and lineage-aware reads. Both refusing
    censuses are reconciliation decisions, not composition inputs: the
    operator either repairs the universe (roster/partition addition, legacy
    snapshot registration) or acknowledges each id explicitly, which only
    lets composition proceed with the census disclosed -- it never selects
    the id. Acknowledgments are DURABLE (an append-only ledger read by every
    composition); the CLI flag is the write path.
    """

    def __init__(
        self,
        *,
        foreign: Sequence[str] = (),
        unregistered: Sequence[str] = (),
        unmatched_acknowledgments: Sequence[str] = (),
    ) -> None:
        """Name every unreconciled id in a typed, actionable refusal."""

        self.foreign = tuple(foreign)
        self.unregistered = tuple(unregistered)
        self.unmatched_acknowledgments = tuple(unmatched_acknowledgments)
        parts: list[str] = []
        if foreign:
            parts.append(
                "canonical records exist for ids foreign to the campaign partition "
                f"universe: {sorted(foreign)}"
            )
        if unregistered:
            parts.append(
                "canonical records exist for partition members outside every "
                f"registered rung: {sorted(unregistered)}"
            )
        if unmatched_acknowledgments:
            parts.append(
                "acknowledgments match no censused id (stale or mistyped): "
                f"{sorted(unmatched_acknowledgments)}"
            )
        super().__init__(
            "membership reconciliation required -- "
            + "; ".join(parts)
            + ". Reconcile the universe (catalog/roster addition or "
            "stage_rung register of the legacy snapshot) or acknowledge each id "
            "explicitly with --acknowledge-membership <stable_id> (the "
            "acknowledgment is written durably to "
            "intake-extensions/membership-acknowledgments.jsonl and read by "
            "every later composition)."
        )


@dataclass(frozen=True)
class StratifiedMember:
    """One selected rung member with the evidence for its stratum.

    Parameters
    ----------
    stable_id, name, zoo:
        Intake identity of the member.
    stratum:
        Owning stratum (first match in assignment order).
    evidence:
        Human-readable record fact that placed the member in its stratum.
    """

    stable_id: str
    name: str
    zoo: str
    stratum: str
    evidence: str

    def to_row(self) -> JsonObject:
        """Return the report row for this member."""

        return {
            "stable_id": self.stable_id,
            "name": self.name,
            "zoo": self.zoo,
            "stratum": self.stratum,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class StratifiedRungPlan:
    """One composed rung: members, per-stratum accounting, records identity.

    Parameters
    ----------
    campaign_id, size, seed:
        Composition inputs. ``size`` is the NEWLY COMPOSED width: the carried
        lineage tip is added on top of it, so a rung's total membership is
        ``size`` plus the carried members no composed stratum already selected.
    shares, targets, pool_sizes, selected_counts:
        Per-stratum shares, apportioned targets, candidate-pool sizes, and
        final selections. ``carried-tip`` appears in ``selected_counts`` only:
        carrying is mandatory, so it has no share and no apportioned target.
    shortfalls:
        Strata that held fewer candidates than their target, with both counts.
    backfilled_from_bulk:
        Members ladder bulk contributed beyond its own target to cover
        disclosed shortfalls.
    members:
        Selected members in emission order (ladder bulk first, then the tail
        strata in assignment order).
    records_state:
        Content digests of every records input the composition read.
    membership_reconciliation:
        The C-ERNIE censuses (``foreign_to_partition``,
        ``records_outside_this_campaign``, ``outside_registered_lineage``)
        plus the acknowledged ids; always present, empty lists in a healthy
        store. ``records_outside_this_campaign`` is informational -- a sibling
        campaign's record-holders in the shared root are accounted, never
        foreign (C-27).
    durable_membership_acknowledgments:
        The campaign's durable acknowledgment rows applied to this
        composition, verbatim (CAT-02: acknowledgments are written once,
        durably, and read by every composition).
    acknowledged_empty_strata:
        Strata the operator acknowledged as vanished; each was PROVEN to hold
        zero candidates before its share was reclaimed by ladder bulk.
    composition_attestation:
        What this composition attests (B-03): the campaign, every member with
        its stratum, and the ``(snapshot_id, snapshot_sha256)`` these members
        compose to. ``stage_rung`` re-derives all three from the stage and
        refuses a mismatch, so a stale or hand-edited snapshot sha can never be
        staged. Required from schema v3 on. The carry labels ride INSIDE its
        digest, which is what makes the carried stratum attested rather than
        merely reported.
    carried_tip:
        The C-2 carry disclosure: the lineage tip's identity and label, its full
        membership with a count, which of those members this composition
        labelled ``carried-tip`` (``carried_forward``) versus which a tail
        stratum selected on its own evidence
        (``also_selected_on_own_evidence``), and the ``lineage-tip`` provenance
        string. A campaign with no registered rung discloses the
        ``no-registered-lineage-tip`` shape with an empty membership. Required
        from schema v4 on.
    award_authorities_required:
        ``stable_id -> exact current raw runs revision`` for every composed
        member (carried ones included) the staging door will demand an AW-2
        ``--award-authority`` for. Disclosure only: the door re-derives it.
    """

    campaign_id: str
    size: int
    seed: int
    shares: Mapping[str, float]
    targets: Mapping[str, int]
    pool_sizes: Mapping[str, int]
    selected_counts: Mapping[str, int]
    shortfalls: Mapping[str, Mapping[str, int]]
    backfilled_from_bulk: int
    members: tuple[StratifiedMember, ...]
    records_state: Mapping[str, Any]
    membership_reconciliation: Mapping[str, list[str]]
    composition_attestation: RungCompositionAttestation
    carried_tip: Mapping[str, Any]
    award_authorities_required: Mapping[str, str]
    acknowledged_empty_strata: tuple[str, ...] = ()
    durable_membership_acknowledgments: tuple[JsonObject, ...] = ()

    def member_ids(self) -> tuple[str, ...]:
        """Return the member stable ids in emission order."""

        return tuple(member.stable_id for member in self.members)

    def to_report(self) -> JsonObject:
        """Return the complete JSON strata report."""

        return {
            "schema_version": STRATIFIED_RUNG_PLAN_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "size": self.size,
            "seed": self.seed,
            "shares": dict(self.shares),
            "targets": dict(self.targets),
            "pool_sizes": dict(self.pool_sizes),
            "selected_counts": dict(self.selected_counts),
            "shortfalls": {name: dict(gap) for name, gap in self.shortfalls.items()},
            "backfilled_from_bulk": self.backfilled_from_bulk,
            "acknowledged_empty_strata": list(self.acknowledged_empty_strata),
            "composition_attestation": self.composition_attestation.to_row(),
            "carried_tip": dict(self.carried_tip),
            "award_authorities_required": dict(self.award_authorities_required),
            "records_state": dict(self.records_state),
            "membership_reconciliation": {
                name: list(ids) for name, ids in self.membership_reconciliation.items()
            },
            "durable_membership_acknowledgments": [
                dict(row) for row in self.durable_membership_acknowledgments
            ],
            "members": [member.to_row() for member in self.members],
        }


def _validated_shares(shares: Optional[Mapping[str, float]]) -> dict[str, float]:
    """Validate one share mapping against the closed stratum vocabulary.

    Parameters
    ----------
    shares:
        Requested shares, or ``None`` for :data:`DEFAULT_SHARES`.

    Returns
    -------
    dict[str, float]
        Complete per-stratum shares summing to one.

    Raises
    ------
    StratificationError
        If a share is unknown, negative, or the total is not one.
    """

    if shares is None:
        return dict(DEFAULT_SHARES)
    known = {LADDER_BULK_STRATUM, *TAIL_STRATA}
    unknown = sorted(set(shares) - known)
    if unknown:
        raise StratificationError(
            f"unknown strata {unknown}; the closed vocabulary is {sorted(known)}"
        )
    merged = {name: float(shares.get(name, 0.0)) for name in known}
    if any(value < 0.0 for value in merged.values()):
        raise StratificationError("stratum shares must be non-negative")
    total = sum(merged.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise StratificationError(f"stratum shares must sum to 1.0, got {total!r}")
    return merged


def _apportion(size: int, shares: Mapping[str, float]) -> dict[str, int]:
    """Apportion the rung size across strata by largest remainder.

    Parameters
    ----------
    size:
        Total rung size.
    shares:
        Validated per-stratum shares.

    Returns
    -------
    dict[str, int]
        Integer targets summing exactly to ``size``.
    """

    quotas = {name: share * size for name, share in shares.items()}
    targets = {name: math.floor(quota) for name, quota in quotas.items()}
    remainder = size - sum(targets.values())
    by_fraction = sorted(
        quotas,
        key=lambda name: (-(quotas[name] - targets[name]), name),
    )
    for name in by_fraction[:remainder]:
        targets[name] += 1
    return targets


def _file_digest(path: Path) -> str:
    """Return one records input's content digest, or ``absent``."""

    if not path.is_file():
        return "absent"
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_rows(path: Path, *, label: str) -> list[JsonObject]:
    """Scan one ledger read-only, translating corruption into a typed refusal.

    Parameters
    ----------
    path:
        Ledger path; a missing file is an empty ledger.
    label:
        Input name for the diagnostic.

    Returns
    -------
    list[dict[str, Any]]
        Parsed rows in byte order.

    Raises
    ------
    StratificationError
        If the ledger is torn or corrupt -- staging happens between rungs, so
        an unreadable input is a surprise to surface, not to skip.
    """

    try:
        return scan_jsonl(path, validate=False)
    except (TornTailError, LedgerCorruptionError) as exc:
        raise StratificationError(f"cannot read {label} ledger {path}: {exc}") from exc


def _current_heads(records_root: Path) -> dict[str, JsonObject]:
    """Load current model revisions keyed by stable id."""

    path = records_root / "models" / "current-shard.jsonl"
    rows = _scan_rows(path, label="models")
    try:
        return _select_current(rows)
    except (LedgerCorruptionError, KeyError) as exc:
        raise StratificationError(f"cannot select current heads from {path}: {exc}") from exc


def _refusal_stable_ids(records_root: Path) -> frozenset[str]:
    """Return every stable id with a content-refusal attempt classification."""

    rows = _scan_rows(records_root / "attempts" / "local.jsonl", label="attempts")
    refused: set[str] = set()
    for row in rows:
        error = row.get("error")
        if (
            isinstance(error, Mapping)
            and error.get("failure_class") == _REFUSAL_FAILURE_CLASS
            and isinstance(row.get("stable_id"), str)
        ):
            refused.add(str(row["stable_id"]))
    return frozenset(refused)


def _attempt_stable_ids(records_root: Path) -> frozenset[str]:
    """Return every stable id that is the subject of any attempt row."""

    rows = _scan_rows(records_root / "attempts" / "local.jsonl", label="attempts")
    return frozenset(str(row["stable_id"]) for row in rows if isinstance(row.get("stable_id"), str))


def _reconcile_membership(
    *,
    universe: frozenset[str],
    sibling_universe: frozenset[str],
    heads: Mapping[str, JsonObject],
    attempt_subjects: frozenset[str],
    valves: Mapping[str, JsonObject],
    prior_members: frozenset[str],
    acknowledged: frozenset[str],
    durable_acknowledged: frozenset[str] = frozenset(),
) -> dict[str, list[str]]:
    """Census canonical-record ids against the staging universe, or refuse.

    Recomputed from the ledgers on every composition -- no persisted flag
    steers a DISPOSITION (the durable acknowledgment ledger only discloses).
    The census is campaign-scoped over a SHARED records root (C-27):
    ``foreign`` means "in no frozen campaign's universe", while a sibling
    campaign's record-holder is accounted informationally
    (``records_outside_this_campaign``) and never refuses. Acknowledgment is
    disposition-invariant: it discloses, it never selects.

    Raises
    ------
    MembershipReconciliationError
        If any refusing censused id lacks an acknowledgment, or an
        acknowledgment matches no censused id.
    """

    record_holders = frozenset(heads) | attempt_subjects | frozenset(valves) | prior_members
    outside_this_campaign = record_holders - universe
    foreign = outside_this_campaign - sibling_universe
    sibling_records = outside_this_campaign & sibling_universe
    unregistered = ((frozenset(heads) | attempt_subjects) & universe) - prior_members
    censused = foreign | unregistered
    # Only EXPLICIT (per-invocation) acknowledgments refuse when they match
    # nothing: a mistyped flag must surface. A durable ledger row whose id is
    # no longer censused (the books were since reconciled) is spent history --
    # it discloses nothing and must never brick a later composition.
    effective = (acknowledged | (durable_acknowledged & censused)) & censused
    unmatched = acknowledged - censused
    unacknowledged_foreign = foreign - effective
    unacknowledged_unregistered = unregistered - effective
    if unmatched or unacknowledged_foreign or unacknowledged_unregistered:
        raise MembershipReconciliationError(
            foreign=sorted(unacknowledged_foreign),
            unregistered=sorted(unacknowledged_unregistered),
            unmatched_acknowledgments=sorted(unmatched),
        )
    return {
        "foreign_to_partition": sorted(foreign),
        "records_outside_this_campaign": sorted(sibling_records),
        "outside_registered_lineage": sorted(unregistered),
        "acknowledged": sorted(effective),
    }


def _sibling_universe(records_root: Path, campaign_id: str) -> frozenset[str]:
    """Return the union of every OTHER frozen campaign's verified membership.

    The C-ERNIE census's ``foreign`` must mean "in NO campaign's universe"
    (CAT-02 / F4): the four campaigns share one records root by design, so a
    record-holder belonging to a sibling campaign is accounted, not foreign.
    Each sibling root must load and reproduce its frozen binding before it may
    account anything -- unverifiable sibling bytes account nothing, and the
    unaccounted ids stay censused (fail-closed: tampered sibling bytes can
    never launder a truly foreign id).

    Parameters
    ----------
    records_root:
        Shared campaign canonical records directory.
    campaign_id:
        The campaign being composed (excluded from the union).

    Returns
    -------
    frozenset[str]
        Stable ids of every verified sibling campaign's root membership.
    """

    manifest_path = records_root / "campaigns.json"
    if not manifest_path.is_file():
        return frozenset()
    try:
        bindings = load_campaign_bindings(manifest_path)
    except CampaignPartitionError:
        return frozenset()
    members: set[str] = set()
    for binding in bindings:
        if binding.spec.campaign_id == campaign_id:
            continue
        try:
            snapshot = load_intake_snapshot(records_root / binding.intake_path)
        except (OSError, ValueError):
            continue
        if (
            snapshot.snapshot_id != binding.intake_snapshot_id
            or snapshot.snapshot_sha256 != binding.intake_snapshot_sha256
        ):
            continue
        members.update(item.stable_id for item in snapshot.items)
    return frozenset(members)


MEMBERSHIP_ACKNOWLEDGMENT_SCHEMA_VERSION = "menagerie.crawler.membership-acknowledgment.v1"

MEMBERSHIP_ACKNOWLEDGMENT_RELATIVE_PATH = (
    "intake-extensions",
    "membership-acknowledgments.jsonl",
)


def membership_acknowledgment_path(records_root: Path) -> Path:
    """Return the durable membership-acknowledgment ledger below one root."""

    return records_root.joinpath(*MEMBERSHIP_ACKNOWLEDGMENT_RELATIVE_PATH)


def _validated_acknowledgment_row(row: JsonObject, path: Path, index: int) -> JsonObject:
    """Validate one durable acknowledgment row, or refuse typed."""

    for field in ("schema_version", "campaign_id", "stable_id", "reason", "acknowledged_by"):
        if not isinstance(row.get(field), str) or not str(row[field]).strip():
            raise StratificationError(
                f"membership-acknowledgment ledger {path} row {index} has an "
                f"empty or missing {field}"
            )
    if row["schema_version"] != MEMBERSHIP_ACKNOWLEDGMENT_SCHEMA_VERSION:
        raise StratificationError(
            f"membership-acknowledgment ledger {path} row {index} declares "
            f"unknown schema {row['schema_version']!r}"
        )
    if row.get("row_sha256") != row_content_address(row):
        raise StratificationError(
            f"membership-acknowledgment ledger {path} row {index} does not "
            "reproduce its content address: acknowledgments are disclosure "
            "rows and may never be hand-edited in place"
        )
    return dict(row)


def load_membership_acknowledgments(
    records_root: Path, campaign_id: str
) -> tuple[JsonObject, ...]:
    """Load one campaign's durable membership acknowledgments, validated.

    Parameters
    ----------
    records_root:
        Shared campaign canonical records directory.
    campaign_id:
        Campaign whose composition is consuming the acknowledgments; rows are
        campaign-scoped (C-27: nothing in a shared root applies across
        campaigns by default).

    Returns
    -------
    tuple[dict[str, Any], ...]
        Validated rows for this campaign, in ledger order.

    Raises
    ------
    StratificationError
        If the ledger is torn, corrupt, or a row fails validation.
    """

    path = membership_acknowledgment_path(records_root)
    rows: list[JsonObject] = []
    for index, row in enumerate(_scan_rows(path, label="membership-acknowledgment")):
        validated = _validated_acknowledgment_row(row, path, index)
        if validated["campaign_id"] == campaign_id:
            rows.append(validated)
    return tuple(rows)


def append_membership_acknowledgment(
    records_root: Path,
    *,
    campaign_id: str,
    stable_id: str,
    reason: str,
    acknowledged_by: str,
    created_at: str,
) -> JsonObject:
    """Append one durable membership acknowledgment, idempotently.

    Written once per ``(campaign_id, stable_id)``: a later append for the same
    pair returns the FIRST persisted row unchanged, so re-running a
    composition command never inflates the ledger. Acknowledgment is
    disposition-invariant -- the row discloses a reconciliation decision; it
    never selects the id into any stratum and never applies to a sibling
    campaign.

    Parameters
    ----------
    records_root:
        Shared campaign canonical records directory.
    campaign_id, stable_id:
        The campaign-scoped censused id being acknowledged.
    reason, acknowledged_by, created_at:
        Operator provenance.

    Returns
    -------
    dict[str, Any]
        The persisted (or already-persisted) row.
    """

    row = stamp_row_digest(
        {
            "schema_version": MEMBERSHIP_ACKNOWLEDGMENT_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "stable_id": stable_id,
            "reason": reason,
            "acknowledged_by": acknowledged_by,
            "created_at": created_at,
        }
    )
    path = membership_acknowledgment_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        for index, existing in enumerate(_scan_rows(path, label="membership-acknowledgment")):
            validated = _validated_acknowledgment_row(existing, path, index)
            if validated["campaign_id"] == campaign_id and validated["stable_id"] == stable_id:
                return validated
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return row


def _valve_paths(records_root: Path) -> dict[str, Path]:
    """Return every machine-deferral valve ledger path by short name.

    Derived from the single registry beside the partition authority
    (:data:`menagerie.crawler.status.MACHINE_DEFERRAL_VALVES`); the previous
    hand-maintained copy omitted the scheduler-quarantine valve, so
    ``rung_report`` and ``stratify_rung`` under-counted valve claims at a
    close boundary (OB-M1).
    """

    return machine_deferral_valve_paths(records_root)


def _valve_claims(records_root: Path) -> dict[str, JsonObject]:
    """Load accountable machine-deferral valve rows keyed by stable id.

    Rows are filtered through the partition arm's own row check
    (:func:`menagerie.crawler.status.machine_deferral_rejection`), so a row
    that could not account a model in the terminal partition cannot mark it
    as prior staged state here either.
    """

    claims: dict[str, JsonObject] = {}
    for label, path in _valve_paths(records_root).items():
        for row in _scan_rows(path, label=f"{label} valve"):
            stable_id = row.get("stable_id")
            if isinstance(stable_id, str) and machine_deferral_rejection(stable_id, row) is None:
                claims[stable_id] = row
    return claims


def _prior_rung_members(
    records_root: Path, campaign_id: str
) -> tuple[frozenset[str], frozenset[str]]:
    """Return prior registered rung member ids and their sampled zoos.

    Every prior snapshot must load and reproduce its registered identity; a
    missing or tampered prior rung is a refusal, not a silent shrink of the
    dedup set.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Campaign whose lineage is read.

    Returns
    -------
    tuple[frozenset[str], frozenset[str]]
        Member stable ids and member zoos across all prior registered rungs.

    Raises
    ------
    StratificationError
        If a registered prior rung snapshot cannot be loaded and verified.
    """

    members: set[str] = set()
    zoos: set[str] = set()
    for registration in load_rung_registrations(records_root):
        if registration.campaign_id != campaign_id:
            continue
        snapshot_root = records_root / "intake" / registration.snapshot_id
        try:
            snapshot = load_intake_snapshot(snapshot_root)
        except (OSError, ValueError) as exc:
            raise StratificationError(
                f"registered rung {registration.snapshot_id} ({registration.label}) "
                f"cannot be loaded from {snapshot_root}: {exc}"
            ) from exc
        if (
            snapshot.snapshot_id != registration.snapshot_id
            or snapshot.snapshot_sha256 != registration.snapshot_sha256
        ):
            raise StratificationError(
                f"registered rung {registration.snapshot_id} does not reproduce its "
                "registered identity; refusing to compose over unverifiable lineage"
            )
        for item in snapshot.items:
            members.add(item.stable_id)
            zoos.add(item.zoo)
    return frozenset(members), frozenset(zoos)


def _routed_intent(item: IntakeItem) -> str:
    """Return the environment intent the driver would route one item to."""

    return route_model(
        item.to_model_requirements(framework_from_markers(item.zoo, item.name))
    ).intent


def _head_status(head: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return one head's status code, if any."""

    if head is None:
        return None
    status = head.get("status")
    if isinstance(status, Mapping):
        code = status.get("code")
        return str(code) if code is not None else None
    return None


def _own_r2_evidence(head: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return the record fact showing one head selected or attempted R2."""

    if head is None:
        return None
    resolution = head.get("source_resolution")
    if not isinstance(resolution, Mapping):
        return None
    status = _head_status(head)
    if resolution.get("rung") == _R2_RUNG:
        return f"head source_resolution.rung={_R2_RUNG} (status {status})"
    attempted = resolution.get("attempted_rungs")
    if isinstance(attempted, Sequence):
        for entry in attempted:
            if isinstance(entry, Mapping) and entry.get("rung") == _R2_RUNG:
                return f"head attempted_rungs contains {_R2_RUNG} (status {status})"
    return None


def _shuffled(candidates: Iterable[str], rng: random.Random) -> list[str]:
    """Return one seeded permutation of a candidate list."""

    ordered = list(candidates)
    rng.shuffle(ordered)
    return ordered


class _Composer:
    """One deterministic composition pass over the loaded records state."""

    def __init__(
        self,
        *,
        campaign_id: str,
        seed: int,
        partition_order: Sequence[str],
        items: Mapping[str, IntakeItem],
        heads: Mapping[str, JsonObject],
        valves: Mapping[str, JsonObject],
        refusals: frozenset[str],
        prior_members: frozenset[str],
        sampled_zoos: frozenset[str],
        oddball_intents: frozenset[str],
    ) -> None:
        """Bind the loaded records state for one composition."""

        self._campaign_id = campaign_id
        self._seed = seed
        self._order = list(partition_order)
        self._items = items
        self._heads = heads
        self._valves = valves
        self._refusals = refusals
        self._prior_members = prior_members
        self._sampled_zoos = sampled_zoos
        self._oddball_intents = oddball_intents
        self._selected: dict[str, StratifiedMember] = {}

    def _rng(self, stratum: str) -> random.Random:
        """Return the stratum-scoped deterministic generator."""

        return random.Random(f"{self._seed}:{self._campaign_id}:{stratum}")

    def _is_runs_holder(self, stable_id: str) -> bool:
        """Return whether one model already converted (and never re-crawls)."""

        return _head_status(self._heads.get(stable_id)) == _RUNS_STATUS

    def _member(self, stable_id: str, stratum: str, evidence: str) -> StratifiedMember:
        """Materialize one selected member row."""

        item = self._items[stable_id]
        return StratifiedMember(
            stable_id=stable_id,
            name=item.name,
            zoo=item.zoo,
            stratum=stratum,
            evidence=evidence,
        )

    def pool(self, stratum: str) -> list[tuple[str, str]]:
        """Return one stratum's full candidate pool as (id, evidence) pairs.

        Pools are computed before cross-stratum dedup so the report's pool
        sizes describe the records state, not the selection order.
        """

        if stratum == "r2-vendored":
            own, propagated = self._r2_tiers()
            return own + propagated
        if stratum == "oddball-intent":
            return self._oddball_pool()
        if stratum == "cuda-deferred":
            return self._status_pool(
                lambda code: code == _CUDA_DEFERRED_STATUS,
                lambda code: f"head status {code}",
            )
        if stratum == "content-refusal":
            return [
                (
                    stable_id,
                    "attempt error.failure_class=content-refusal "
                    f"(head status {_head_status(self._heads.get(stable_id))})",
                )
                for stable_id in self._order
                if stable_id in self._refusals and not self._is_runs_holder(stable_id)
            ]
        if stratum == "unsampled-zoo":
            return [
                (pair[0], pair[1])
                for zoo_members in self._unsampled_zoo_pool().values()
                for pair in zoo_members
            ]
        if stratum == "hard-tail":
            return self._status_pool(
                lambda code: code in _HARD_TAIL_STATUSES,
                lambda code: f"head status {code}",
            )
        raise StratificationError(f"unknown stratum {stratum!r}")

    def host_capability_deferred_pool(self) -> list[tuple[str, str]]:
        """Return the members a bigger-RAM host would discharge, with their evidence.

        Diagnostic and composition input, not a share-bearing stratum: see
        :data:`HOST_CAPABILITY_DEFERRED_STATUS` for why that boundary is deliberate.

        Returns
        -------
        list[tuple[str, str]]
            ``(stable_id, evidence)`` pairs in partition order.
        """

        return self._status_pool(
            lambda code: code == HOST_CAPABILITY_DEFERRED_STATUS,
            lambda code: f"head status {code}",
        )

    def _status_pool(self, predicate, evidence) -> list[tuple[str, str]]:
        """Return partition members whose head status satisfies one predicate."""

        pool: list[tuple[str, str]] = []
        for stable_id in self._order:
            code = _head_status(self._heads.get(stable_id))
            if code is not None and code != _RUNS_STATUS and predicate(code):
                pool.append((stable_id, evidence(code)))
        return pool

    def _r2_tiers(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Return the vendored-closure class: own R2 evidence, then zoo mates.

        The two tiers stay separate so selection exhausts the models that
        RECORDED an R2 resolution before reaching for fresh zoo mates.
        """

        own: list[tuple[str, str]] = []
        own_ids: set[str] = set()
        r2_zoos: dict[str, str] = {}
        for stable_id in self._order:
            evidence = _own_r2_evidence(self._heads.get(stable_id))
            if evidence is None:
                continue
            r2_zoos.setdefault(self._items[stable_id].zoo, stable_id)
            if not self._is_runs_holder(stable_id):
                own.append((stable_id, evidence))
                own_ids.add(stable_id)
        propagated = [
            (
                stable_id,
                f"zoo {self._items[stable_id].zoo!r} carries a prior {_R2_RUNG} "
                f"resolution (via {r2_zoos[self._items[stable_id].zoo]})",
            )
            for stable_id in self._order
            if stable_id not in own_ids
            and self._items[stable_id].zoo in r2_zoos
            and not self._is_runs_holder(stable_id)
        ]
        return own, propagated

    def _oddball_pool(self) -> list[tuple[str, str]]:
        """Return members routing to the oddball/lockless environment intents."""

        pool: list[tuple[str, str]] = []
        for stable_id in self._order:
            if self._is_runs_holder(stable_id):
                continue
            intent = _routed_intent(self._items[stable_id])
            if intent not in self._oddball_intents:
                continue
            evidence = f"routes to environment intent {intent!r}"
            valve = self._valves.get(stable_id)
            if valve is not None:
                evidence += f"; valve-parked as {valve.get('disposition')}"
            pool.append((stable_id, evidence))
        return pool

    def _unsampled_zoo_pool(self) -> dict[str, list[tuple[str, str]]]:
        """Return fresh members of never-sampled zoos, grouped by zoo."""

        by_zoo: dict[str, list[tuple[str, str]]] = {}
        for stable_id in self._order:
            zoo = self._items[stable_id].zoo
            if (
                zoo in self._sampled_zoos
                or stable_id in self._heads
                or stable_id in self._valves
                or stable_id in self._prior_members
            ):
                continue
            by_zoo.setdefault(zoo, []).append(
                (stable_id, f"zoo {zoo!r} never sampled by a prior registered rung")
            )
        return by_zoo

    def take(self, stratum: str, target: int) -> tuple[list[StratifiedMember], int]:
        """Select up to ``target`` members for one tail stratum.

        Parameters
        ----------
        stratum:
            Tail stratum name.
        target:
            Apportioned member count.

        Returns
        -------
        tuple[list[StratifiedMember], int]
            Selected members and the full (pre-dedup) pool size.

        Raises
        ------
        EmptyStratumError
            If the target is positive and the pool holds zero candidates.
        """

        pool = self.pool(stratum)
        if target > 0 and not pool:
            raise EmptyStratumError(stratum)
        if target == 0:
            return [], len(pool)
        evidence_by_id = dict(pool)
        if stratum == "unsampled-zoo":
            ordered = self._zoo_round_robin(evidence_by_id)
        elif stratum == "r2-vendored":
            own, propagated = self._r2_tiers()
            rng = self._rng(stratum)
            ordered = _shuffled((stable_id for stable_id, _ in own), rng) + _shuffled(
                (stable_id for stable_id, _ in propagated), rng
            )
        else:
            ordered = _shuffled((stable_id for stable_id, _ in pool), self._rng(stratum))
        selected: list[StratifiedMember] = []
        for stable_id in ordered:
            if len(selected) >= target or stable_id in self._selected:
                continue
            member = self._member(stable_id, stratum, evidence_by_id[stable_id])
            self._selected[stable_id] = member
            selected.append(member)
        return selected, len(pool)

    def _zoo_round_robin(self, evidence_by_id: Mapping[str, str]) -> list[str]:
        """Order never-sampled-zoo candidates one zoo at a time.

        Zoos rotate in seeded order and members keep partition order within
        each zoo, so a width of N reaches N distinct zoos whenever N distinct
        zoos exist -- maximum zoo diversity per rung.
        """

        by_zoo = self._unsampled_zoo_pool()
        rng = self._rng("unsampled-zoo")
        zoo_order = _shuffled(sorted(by_zoo), rng)
        ordered: list[str] = []
        depth = 0
        while any(depth < len(by_zoo[zoo]) for zoo in zoo_order):
            for zoo in zoo_order:
                if depth < len(by_zoo[zoo]):
                    stable_id = by_zoo[zoo][depth][0]
                    if stable_id in evidence_by_id:
                        ordered.append(stable_id)
            depth += 1
        return ordered

    def bulk_eligible(self) -> list[str]:
        """Return fresh next-partition rows in scheduling order.

        Fresh means: never a prior rung member, holding no head record and no
        valve claim, and not already selected by a tail stratum. Runs-holders
        are excluded by construction (a runs head is a head record).
        """

        return [
            stable_id
            for stable_id in self._order
            if stable_id not in self._selected
            and stable_id not in self._prior_members
            and stable_id not in self._heads
            and stable_id not in self._valves
        ]

    def carry_lineage_tip(
        self, tip_label: str, tip_snapshot_id: str, tip_members: frozenset[str]
    ) -> list[StratifiedMember]:
        """Carry every lineage-tip member the composition has not already selected.

        Called LAST, after the tail strata and ladder bulk have spent ``--size``,
        so carrying changes no pool, no target, and no shortfall: a tip member a
        tail stratum picked on its own evidence keeps that stratum and that
        evidence, and the residue is carried explicitly. What the rung staged
        becomes cumulative either way, which is what the door's widen rule needs.

        Parameters
        ----------
        tip_label, tip_snapshot_id:
            Identity of the lineage tip, recorded as each carried member's
            evidence so the report says WHY the member is in the rung.
        tip_members:
            The tip's re-verified durable membership.

        Returns
        -------
        list[StratifiedMember]
            Carried members: partition order first, then any tip member the
            partition no longer schedules (still a campaign member, so still
            carried -- dropping it would silently un-widen the tip).

        Raises
        ------
        StratificationError
            If the tip holds a member the campaign root snapshot does not, which
            would make the tip uncarryable and the successor unstageable.
        """

        missing = sorted(tip_members - set(self._items))
        if missing:
            raise StratificationError(
                f"the lineage tip {tip_snapshot_id} ({tip_label}) holds "
                f"{len(missing)} member(s) the campaign root snapshot does not "
                f"contain: {missing[:5]}. A successor must carry its tip, so this "
                "is a records-state repair, not a composition input"
            )
        evidence = (
            f"carried from lineage tip {tip_snapshot_id} ({tip_label}): rung "
            "membership is cumulative"
        )
        scheduled = [stable_id for stable_id in self._order if stable_id in tip_members]
        unscheduled = sorted(tip_members - set(self._order))
        carried: list[StratifiedMember] = []
        for stable_id in (*scheduled, *unscheduled):
            if stable_id in self._selected:
                continue
            member = self._member(stable_id, CARRIED_TIP_STRATUM, evidence)
            self._selected[stable_id] = member
            carried.append(member)
        return carried

    def take_bulk(self, count: int) -> list[StratifiedMember]:
        """Select the leading ``count`` fresh partition rows as ladder bulk.

        Raises
        ------
        EmptyStratumError
            If bulk members are needed and no fresh partition row remains.
        StratificationError
            If fewer fresh rows remain than the rung still needs.
        """

        eligible = self.bulk_eligible()
        if count > 0 and not eligible:
            raise EmptyStratumError(LADDER_BULK_STRATUM)
        if len(eligible) < count:
            raise StratificationError(
                f"the campaign partition holds only {len(eligible)} fresh rows but the "
                f"rung still needs {count}; reduce --size or investigate the partition"
            )
        selected: list[StratifiedMember] = []
        for stable_id in eligible[:count]:
            member = self._member(stable_id, LADDER_BULK_STRATUM, "next fresh partition row")
            self._selected[stable_id] = member
            selected.append(member)
        return selected


def _acknowledged_strata(acknowledged: Optional[Iterable[str]]) -> tuple[str, ...]:
    """Validate empty-stratum acknowledgements against the tail vocabulary.

    Parameters
    ----------
    acknowledged:
        Requested acknowledgements, or ``None``.

    Returns
    -------
    tuple[str, ...]
        De-duplicated names in :data:`TAIL_STRATA` assignment order.

    Raises
    ------
    StratificationError
        If a name is not a tail stratum. ``ladder-bulk`` is deliberately not
        acknowledgeable: an exhausted partition is a rung-sizing refusal, not a
        vanished tail class.
    """

    if not acknowledged:
        return ()
    requested = dict.fromkeys(str(name) for name in acknowledged)
    unknown = sorted(set(requested) - set(TAIL_STRATA))
    if unknown:
        raise StratificationError(
            f"cannot acknowledge {unknown} as empty strata; only the tail strata "
            f"{sorted(TAIL_STRATA)} are acknowledgeable "
            f"({LADDER_BULK_STRATUM} exhaustion is a rung-sizing refusal)"
        )
    return tuple(name for name in TAIL_STRATA if name in requested)


def _shares_after_acknowledgements(
    shares: Mapping[str, float], acknowledged: Sequence[str]
) -> dict[str, float]:
    """Zero every acknowledged stratum and reclaim its share for ladder bulk.

    Parameters
    ----------
    shares:
        Validated per-stratum shares summing to one.
    acknowledged:
        Validated acknowledged tail strata.

    Returns
    -------
    dict[str, float]
        Shares that still sum to one, with the reclaimed share on ladder bulk.
    """

    resolved = dict(shares)
    reclaimed = sum(resolved[name] for name in acknowledged)
    for name in acknowledged:
        resolved[name] = 0.0
    resolved[LADDER_BULK_STRATUM] += reclaimed
    return resolved


def _carried_tip_disclosure(
    tip: Optional[RungRegistration],
    tip_members: frozenset[str],
    members: Sequence[StratifiedMember],
) -> dict[str, Any]:
    """Return the report's carry block for one composition (C-2).

    The carry has to be readable without re-deriving anything: which rung was
    carried, what it contained, how many members that is, which of them this
    composition labelled ``carried-tip``, and which a tail stratum had already
    selected on its own evidence (a re-attempt of the hard tail keeps its own
    evidence -- it is carried either way, and pretending otherwise would hide
    why it is in the rung).

    Parameters
    ----------
    tip:
        The campaign's registry-tip registration, or ``None`` when none exists.
    tip_members:
        The tip's re-verified membership.
    members:
        Every selected member of this composition.

    Returns
    -------
    dict[str, Any]
        Canonical carry disclosure, JSON-ready.
    """

    by_id = {member.stable_id: member.stratum for member in members}
    carried_forward = sorted(
        stable_id for stable_id in tip_members if by_id.get(stable_id) == CARRIED_TIP_STRATUM
    )
    also_selected = {
        stable_id: by_id[stable_id]
        for stable_id in sorted(tip_members)
        if by_id.get(stable_id) not in (None, CARRIED_TIP_STRATUM)
    }
    return {
        "provenance": NO_CARRIED_TIP_PROVENANCE if tip is None else CARRIED_TIP_PROVENANCE,
        "snapshot_id": None if tip is None else tip.snapshot_id,
        "snapshot_sha256": None if tip is None else tip.snapshot_sha256,
        "label": None if tip is None else tip.label,
        "member_count": len(tip_members),
        "member_ids": sorted(tip_members),
        "carried_forward": carried_forward,
        "also_selected_on_own_evidence": also_selected,
    }


def stratify_rung_members(
    records_root: Path,
    *,
    campaign_id: str,
    size: int,
    seed: int,
    shares: Optional[Mapping[str, float]] = None,
    oddball_intents: Optional[Iterable[str]] = None,
    acknowledged_empty_strata: Optional[Iterable[str]] = None,
    acknowledged_membership: Optional[Iterable[str]] = None,
    pd7_disposition: Optional[Mapping[str, str]] = None,
) -> StratifiedRungPlan:
    """Compose one stratified rung member list from the records state.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory (read-only).
    campaign_id:
        Frozen tier campaign whose partition supplies the membership universe.
    size:
        NEWLY COMPOSED rung width. The campaign's registry tip is carried on top
        of it (C-2: membership is cumulative), so the staged rung holds ``size``
        composed members plus every tip member no stratum already selected.
    seed:
        Determinism input: the same records state, seed, and size always
        compose the same rung.
    shares:
        Optional per-stratum share overrides; must cover the closed stratum
        vocabulary and sum to one. ``None`` uses :data:`DEFAULT_SHARES`.
    oddball_intents:
        Optional override of the oddball/lockless intent set.
    acknowledged_empty_strata:
        Tail strata the operator acknowledges as vanished. Each is PROVEN empty
        against this records state before its share is zeroed and reclaimed by
        ladder bulk, and every acknowledgement is disclosed in the report.
    acknowledged_membership:
        Stable ids whose membership-reconciliation census rows the operator
        explicitly acknowledges for THIS invocation, unioned with the
        campaign's durable acknowledgment ledger
        (``intake-extensions/membership-acknowledgments.jsonl``), which every
        composition reads. Acknowledgment discloses; it never selects the id
        into any stratum.
    pd7_disposition:
        PD-7 slop-family-routes disposition to bind into the composition
        attestation (SOL-02): ``{"status": ..., "source_attestation_sha256":
        ...}``. ``None`` emits a v1 attestation, which no successor-rung
        stage accepts.

    Returns
    -------
    StratifiedRungPlan
        Members (carried tip first, then ladder bulk, then the tail strata in
        assignment order), per-stratum accounting, the carry disclosure, and
        records-state digests.

    Raises
    ------
    EmptyStratumError
        If a stratum with a positive share has zero candidates.
    StratumNotEmptyError
        If an acknowledged stratum still holds candidates.
    MembershipReconciliationError
        If canonical records exist outside the staging universe without an
        explicit acknowledgment, or an acknowledgment matches nothing.
    StratificationError
        If the inputs are invalid, the records state cannot compose a rung, or
        the composition adds nothing beyond the carried lineage tip (a rung that
        is only its predecessor is not a rung, and the door would refuse it).
    """

    if size < 1:
        raise StratificationError("rung size must be positive")
    acknowledged = _acknowledged_strata(acknowledged_empty_strata)
    validated_shares = _shares_after_acknowledgements(_validated_shares(shares), acknowledged)
    targets = _apportion(size, validated_shares)
    try:
        root_binding, root_snapshot = resolve_verified_root(records_root, campaign_id)
        partition_order = partition_member_stable_ids(records_root, root_binding, root_snapshot)
    except RungStagingError as exc:
        raise StratificationError(str(exc)) from exc
    items = {item.stable_id: item for item in root_snapshot.items}
    heads = _current_heads(records_root)
    refusals = _refusal_stable_ids(records_root)
    valves = _valve_claims(records_root)
    attempt_subjects = _attempt_stable_ids(records_root)
    prior_members, sampled_zoos = _prior_rung_members(records_root, campaign_id)
    # The lineage tip a successor must carry, through the staging door's OWN
    # spelling of "the tip and its membership" -- so the composition can never
    # carry a different set than the door recomputes and enforces.
    try:
        tip_membership = lineage_tip_membership(records_root, campaign_id)
    except RungStagingError as exc:
        raise StratificationError(str(exc)) from exc
    tip, tip_members = tip_membership if tip_membership is not None else (None, frozenset())
    universe = frozenset(partition_order)
    record_holders = frozenset(heads) | attempt_subjects | frozenset(valves) | prior_members
    # The sibling universes are consulted only when a record-holder falls
    # outside THIS campaign's universe: loading (and re-verifying) three more
    # root snapshots is pointless on the overwhelmingly common healthy path.
    sibling_universe = (
        _sibling_universe(records_root, campaign_id)
        if record_holders - universe
        else frozenset()
    )
    durable_acknowledgments = load_membership_acknowledgments(records_root, campaign_id)
    membership_reconciliation = _reconcile_membership(
        universe=universe,
        sibling_universe=sibling_universe,
        heads=heads,
        attempt_subjects=attempt_subjects,
        valves=valves,
        prior_members=prior_members,
        acknowledged=frozenset(acknowledged_membership or ()),
        durable_acknowledged=frozenset(row["stable_id"] for row in durable_acknowledgments),
    )
    composer = _Composer(
        campaign_id=campaign_id,
        seed=seed,
        partition_order=partition_order,
        items=items,
        heads={k: v for k, v in heads.items() if k in items},
        valves={k: v for k, v in valves.items() if k in items},
        refusals=refusals,
        prior_members=prior_members,
        sampled_zoos=sampled_zoos,
        oddball_intents=frozenset(
            oddball_intents if oddball_intents is not None else DEFAULT_ODDBALL_INTENTS
        ),
    )
    tail_members: list[StratifiedMember] = []
    pool_sizes: dict[str, int] = {}
    shortfalls: dict[str, dict[str, int]] = {}
    for stratum in TAIL_STRATA:
        selected, pool_size = composer.take(stratum, targets[stratum])
        pool_sizes[stratum] = pool_size
        # An acknowledgement is a claim about the records state, so prove it here
        # rather than trusting it: a stratum that still holds candidates cannot
        # be dropped under cover of the empty-stratum remedy.
        if stratum in acknowledged and pool_size:
            raise StratumNotEmptyError(stratum, pool_size)
        tail_members.extend(selected)
        if len(selected) < targets[stratum]:
            shortfalls[stratum] = {
                "target": targets[stratum],
                "available": len(selected),
            }
    backfill = sum(gap["target"] - gap["available"] for gap in shortfalls.values())
    bulk_members = composer.take_bulk(targets[LADDER_BULK_STRATUM] + backfill)
    pool_sizes[LADDER_BULK_STRATUM] = len(bulk_members) + len(composer.bulk_eligible())
    composed = (*bulk_members, *tail_members)
    # C-2: the rung is a cumulative successor, so it carries its lineage tip.
    # Carried LAST (nothing above it changes) and emitted FIRST (the inherited
    # base of the rung, before the width this composition bought).
    carried_members = (
        composer.carry_lineage_tip(tip.label, tip.snapshot_id, tip_members)
        if tip is not None
        else []
    )
    if tip is not None and not {member.stable_id for member in composed} - tip_members:
        raise StratificationError(
            f"this composition adds nothing beyond the carried lineage tip "
            f"{tip.snapshot_id} ({tip.label}): all {len(composed)} composed member(s) "
            "are already tip members, so the rung would not strictly widen it. Widen "
            "--size, or re-weight the shares toward ladder-bulk"
        )
    members = (*carried_members, *composed)
    selected_counts: dict[str, int] = {name: 0 for name in targets}
    selected_counts[CARRIED_TIP_STRATUM] = 0
    for member in members:
        selected_counts[member.stratum] += 1
    valve_digests = {
        f"valve_{label}_sha256": _file_digest(path)
        for label, path in sorted(_valve_paths(records_root).items())
    }
    records_state: dict[str, Any] = {
        "records_root": records_root.as_posix(),
        "models_ledger_sha256": _file_digest(records_root / "models" / "current-shard.jsonl"),
        "attempts_ledger_sha256": _file_digest(records_root / "attempts" / "local.jsonl"),
        "registered_rungs_sha256": _file_digest(rung_registry_path(records_root)),
        "membership_acknowledgments_sha256": _file_digest(
            membership_acknowledgment_path(records_root)
        ),
        "partition_sha256": root_binding.partition_sha256,
        "root_snapshot_id": root_binding.intake_snapshot_id,
        **valve_digests,
    }
    # B-03: attest the composition against the SAME records state it was
    # composed from, using the staging door's own identity spelling. A later
    # stage recomputes this identity from the members it is actually staging,
    # so a stale attestation (root snapshot moved) and a hand-edited sha both
    # refuse there.
    # Both halves are staging-side calls, so both are re-raised as composition
    # refusals: the CLI catches StratificationError, and an operator tool that
    # tracebacks instead of refusing is indistinguishable from a crash.
    try:
        attested_snapshot_id, attested_snapshot_sha256 = planned_subset_identity(
            records_root, campaign_id, [member.stable_id for member in members]
        )
        attestation = build_composition_attestation(
            campaign_id=campaign_id,
            size=size,
            seed=seed,
            member_strata={member.stable_id: member.stratum for member in members},
            snapshot_id=attested_snapshot_id,
            snapshot_sha256=attested_snapshot_sha256,
            # SOL-02: the PD-7 disposition the operator handed the composer is
            # bound INTO the attestation digest, so the staging door consumes
            # the adjudication itself. Absent -> a v1 attestation, which no
            # successor-rung stage accepts.
            pd7_disposition=pd7_disposition,
        )
        # The AW-2 demand the stage will make, derived by the DOOR's predicate:
        # disclosure only, so the operator never hand-tallies which carried
        # award holders need an --award-authority.
        award_authorities = required_award_authorities(
            records_root, [member.stable_id for member in members]
        )
    except RungStagingError as exc:
        raise StratificationError(str(exc)) from exc
    carried_tip = _carried_tip_disclosure(tip, tip_members, members)
    return StratifiedRungPlan(
        campaign_id=campaign_id,
        size=size,
        seed=seed,
        shares=validated_shares,
        targets=targets,
        pool_sizes=pool_sizes,
        selected_counts=selected_counts,
        shortfalls=shortfalls,
        backfilled_from_bulk=backfill,
        members=members,
        records_state=records_state,
        membership_reconciliation=membership_reconciliation,
        composition_attestation=attestation,
        carried_tip=carried_tip,
        award_authorities_required=award_authorities,
        acknowledged_empty_strata=acknowledged,
        durable_membership_acknowledgments=durable_acknowledgments,
    )
