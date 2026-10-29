"""The persisted stratified crawl ORDER (overhaul RC-1, RULING D.1 behavior 2).

Under the continuous crawl there are no rungs. A campaign has ONE durable,
content-addressed member order, ``records/intake/<campaign>/order.jsonl``,
composed once from the campaign's full partition snapshot with a recorded seed
and never re-derived. ``crawl --limit N`` admits the first ``N`` order rows
that the WS-1 predicate says are work; widening is raising ``N`` (an
operational event), never a new composition.

Composition reuses the existing stratifier exactly once, as an ordering:

* the campaign's registered lineage tip (frozen history) is carried FIRST, in
  partition order -- so a live crawl continues the cohort the last rung
  adjudicated instead of re-shuffling it;
* the next ``stratified_width`` rows are the stratifier's own composition
  (ladder bulk plus the six tail strata, the same shares and seeded shuffles
  a rung would have used), so the first widening step is a stratified sample;
* every remaining partition member follows in the campaign's frozen partition
  order (``partition-order``), so the order is TOTAL over the partition and
  ``--limit`` eventually reaches every member.

The file is append-once: the writer creates it with ``O_EXCL`` and never
rewrites it; every later invocation loads and re-verifies the seal digest,
the ordinal chain, and the member universe against the bound root snapshot.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    fsync_directory,
    stable_hash,
)
from menagerie.crawler.models import JsonObject

__all__ = [
    "CRAWL_ORDER_SCHEMA_VERSION",
    "DEFAULT_ORDER_SEED",
    "DEFAULT_STRATIFIED_WIDTH",
    "PARTITION_ORDER_STRATUM",
    "CrawlOrder",
    "CrawlOrderError",
    "build_crawl_order",
    "crawl_scope",
    "ensure_crawl_order",
    "load_crawl_order",
    "order_path",
]

CRAWL_ORDER_SCHEMA_VERSION = "menagerie.crawler.crawl-order.v1"

#: The recorded default seed. The seed is provenance, not policy: it is written
#: into the order header and the order is never re-derived, so changing this
#: constant changes nothing about an existing campaign order.
DEFAULT_ORDER_SEED = 20260901

#: Width of the ONE stratified composition after the carried tip. The
#: historical rung step was 100; the first widening step of a crawl is a
#: stratified sample of exactly that width.
DEFAULT_STRATIFIED_WIDTH = 100

#: Stratum label for every member after the stratified composition.
PARTITION_ORDER_STRATUM = "partition-order"

#: Stratum label for a snapshot member the frozen partition does not schedule
#: (a records-state oddity, disclosed rather than dropped).
SNAPSHOT_RESIDUE_STRATUM = "snapshot-residue"

_HEADER_KIND = "header"
_MEMBER_KIND = "member"
_SEAL_KIND = "seal"


class CrawlOrderError(ValueError):
    """The order cannot be composed, persisted, or trusted."""


@dataclass(frozen=True)
class CrawlOrder:
    """One loaded (or freshly composed) campaign crawl order.

    Parameters
    ----------
    header:
        The header row: schema, campaign and snapshot binding, seed, widths,
        acknowledged strata, records-state digests, composition disclosure.
    members:
        ``(stable_id, stratum, evidence)`` in order, ordinal = index + 1.
    order_sha256:
        Content address over the header and every member row.
    """

    header: Mapping[str, Any]
    members: tuple[tuple[str, str, str], ...]
    order_sha256: str

    @property
    def campaign_id(self) -> str:
        """Return the bound campaign."""

        return str(self.header["campaign_id"])

    @property
    def stable_ids(self) -> tuple[str, ...]:
        """Return every member id in order."""

        return tuple(stable_id for stable_id, _stratum, _evidence in self.members)

    def rows(self) -> list[JsonObject]:
        """Return the member rows exactly as persisted."""

        return [
            {
                "kind": _MEMBER_KIND,
                "ordinal": ordinal,
                "stable_id": stable_id,
                "stratum": stratum,
                "evidence": evidence,
            }
            for ordinal, (stable_id, stratum, evidence) in enumerate(self.members, start=1)
        ]


def order_path(records_root: Path, campaign_id: str) -> Path:
    """Return the campaign's durable order path."""

    return records_root / "intake" / campaign_id / "order.jsonl"


def _order_digest(header: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    """Content-address one header plus its member rows."""

    return stable_hash({"header": dict(header), "members": [dict(row) for row in rows]})


def crawl_scope(order: CrawlOrder, limit: Optional[int]) -> tuple[str, ...]:
    """Return the first ``limit`` order members (all of them when ``None``).

    The scope is the order prefix, NOT the first ``limit`` work items: the WS-1
    predicate decides work inside the prefix, so raising ``limit`` is the only
    way a member enters the crawl (RULING D.1 behavior 2).
    """

    if limit is None:
        return order.stable_ids
    if limit < 1:
        raise CrawlOrderError(f"crawl limit must be positive, not {limit}")
    return order.stable_ids[:limit]


def _compose_stratified(
    records_root: Path,
    *,
    campaign_id: str,
    seed: int,
    stratified_width: int,
) -> tuple[list[tuple[str, str, str]], list[str], dict[str, Any], str, int]:
    """Run the stratifier once as an ordering, acknowledging proven-empty strata.

    Returns
    -------
    tuple
        ``(composed members, acknowledged empty strata, records-state digests,
        composition disclosure, effective width)``. A partition with no fresh
        row left composes nothing and discloses ``partition-order-only``.
    """

    from menagerie.crawler.rung_stratification import (  # noqa: PLC0415 -- composition-only import
        EmptyStratumError,
        MembershipReconciliationError,
        StratificationError,
        TAIL_STRATA,
        stratify_rung_members,
    )

    acknowledged: list[str] = []
    width = stratified_width
    while width > 0:
        try:
            plan = stratify_rung_members(
                records_root,
                campaign_id=campaign_id,
                size=width,
                seed=seed,
                acknowledged_empty_strata=tuple(acknowledged) or None,
            )
        except EmptyStratumError as exc:
            if exc.stratum in TAIL_STRATA and exc.stratum not in acknowledged:
                # Proven empty by the stratifier itself; acknowledged and
                # disclosed in the header, exactly the remedy the rung
                # composer offered its operator.
                acknowledged.append(exc.stratum)
                continue
            # No fresh partition row remains.
            return [], acknowledged, {}, "partition-order-only", 0
        except MembershipReconciliationError as exc:
            raise CrawlOrderError(
                f"membership reconciliation refuses composition: {exc}"
            ) from exc
        except StratificationError:
            # Fewer fresh rows than the requested width: halve until the
            # stratifier can compose, disclosed through the effective width.
            width //= 2
            continue
        members = [(m.stable_id, m.stratum, m.evidence) for m in plan.members]
        return members, acknowledged, dict(plan.records_state), "stratified", width
    return [], acknowledged, {}, "partition-order-only", 0


def _assemble_members(
    composed: Sequence[tuple[str, str, str]],
    *,
    partition_order: Sequence[str],
    snapshot_member_ids: Sequence[str],
    tip_members: frozenset[str],
    tip_evidence: str,
    carried_tip_stratum: str,
) -> list[tuple[str, str, str]]:
    """Order = lineage tip (partition order), then the composition, then the rest.

    A tip member a tail stratum picked keeps that stratum's evidence but not
    its later slot: a live crawl at ``--limit <= |tip|`` continues exactly the
    cohort the last rung adjudicated, and the stratified step starts at row
    ``|tip| + 1``. Every snapshot member is placed exactly once.
    """

    composed_by_id = {sid: (stratum, evidence) for sid, stratum, evidence in composed}
    members: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    def place(stable_id: str, stratum: str, evidence: str) -> None:
        if stable_id in seen:
            return
        seen.add(stable_id)
        members.append((stable_id, stratum, evidence))

    for stable_id in (*partition_order, *sorted(tip_members)):
        if stable_id in tip_members:
            stratum, evidence = composed_by_id.get(stable_id, (carried_tip_stratum, tip_evidence))
            place(stable_id, stratum, evidence)
    for stable_id, stratum, evidence in composed:
        place(stable_id, stratum, evidence)
    for stable_id in partition_order:
        place(stable_id, PARTITION_ORDER_STRATUM, "campaign partition order")
    for stable_id in snapshot_member_ids:
        place(stable_id, SNAPSHOT_RESIDUE_STRATUM, "root snapshot member absent from partition")
    return members


def build_crawl_order(
    records_root: Path,
    *,
    campaign_id: str,
    seed: int = DEFAULT_ORDER_SEED,
    stratified_width: int = DEFAULT_STRATIFIED_WIDTH,
    created_at: str,
) -> CrawlOrder:
    """Compose the campaign order from the records state (no write).

    Parameters
    ----------
    records_root:
        Campaign canonical records directory (read here).
    campaign_id:
        Frozen tier campaign whose root partition supplies the universe.
    seed, stratified_width:
        The recorded composition inputs.
    created_at:
        Composition timestamp, written into the header.

    Raises
    ------
    CrawlOrderError
        If the campaign binding cannot be verified or the records state
        refuses composition for a reason the operator must repair.
    """

    # Lazy: the staging reader is a rung-era module this order REPLACES; it is
    # consulted once, at composition, never on the steady-state invocation
    # path (which only loads the file).
    from menagerie.crawler.rung_staging import (  # noqa: PLC0415 -- composition-only import
        CARRIED_TIP_STRATUM,
        RungStagingError,
        lineage_tip_membership,
        partition_member_stable_ids,
        resolve_verified_root,
    )

    if stratified_width < 0:
        raise CrawlOrderError("stratified_width cannot be negative")
    try:
        binding, snapshot = resolve_verified_root(records_root, campaign_id)
        partition_order = partition_member_stable_ids(records_root, binding, snapshot)
        tip_membership = lineage_tip_membership(records_root, campaign_id)
    except RungStagingError as exc:
        raise CrawlOrderError(f"campaign binding refused: {exc}") from exc
    tip, tip_members = tip_membership if tip_membership is not None else (None, frozenset())
    composed, acknowledged, records_state, composition, effective_width = _compose_stratified(
        records_root, campaign_id=campaign_id, seed=seed, stratified_width=stratified_width
    )
    members = _assemble_members(
        composed,
        partition_order=partition_order,
        snapshot_member_ids=[item.stable_id for item in snapshot.items],
        tip_members=tip_members,
        tip_evidence=(
            f"carried from lineage tip {tip.snapshot_id} ({tip.label})" if tip is not None else ""
        ),
        carried_tip_stratum=CARRIED_TIP_STRATUM,
    )
    strata_counts: dict[str, int] = {}
    for _stable_id, stratum, _evidence in members:
        strata_counts[stratum] = strata_counts.get(stratum, 0) + 1
    header: JsonObject = {
        "kind": _HEADER_KIND,
        "schema_version": CRAWL_ORDER_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "intake_snapshot_id": snapshot.snapshot_id,
        "intake_snapshot_sha256": snapshot.snapshot_sha256,
        "partition_sha256": binding.partition_sha256,
        "seed": int(seed),
        "stratified_width_requested": int(stratified_width),
        "stratified_width_effective": int(effective_width),
        "composition": composition,
        "acknowledged_empty_strata": list(acknowledged),
        "records_state": records_state,
        "member_count": len(members),
        "lineage_tip": (
            None
            if tip is None
            else {"snapshot_id": tip.snapshot_id, "label": tip.label, "members": len(tip_members)}
        ),
        "strata_counts": strata_counts,
        "created_at": created_at,
    }
    order = CrawlOrder(header=header, members=tuple(members), order_sha256="")
    digest = _order_digest(header, order.rows())
    return CrawlOrder(header=header, members=tuple(members), order_sha256=digest)


def _persist(path: Path, order: CrawlOrder) -> None:
    """Write the order exactly once (``O_EXCL``), fsynced, or refuse."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [canonical_json_bytes(dict(order.header)) + b"\n"]
    lines.extend(canonical_json_bytes(row) + b"\n" for row in order.rows())
    lines.append(
        canonical_json_bytes({"kind": _SEAL_KIND, "order_sha256": order.order_sha256}) + b"\n"
    )
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as exc:
        raise CrawlOrderError(f"order already exists and is append-once: {path}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(b"".join(lines))
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)


def _read_order_lines(path: Path) -> tuple[JsonObject, list[JsonObject], str]:
    """Parse the persisted file into ``(header, member rows, seal digest)``."""

    by_kind: dict[str, list[JsonObject]] = {_HEADER_KIND: [], _MEMBER_KIND: [], _SEAL_KIND: []}
    for line_number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CrawlOrderError(f"{path}:{line_number} is not JSON: {exc}") from exc
        bucket = by_kind.get(str(value.get("kind"))) if isinstance(value, dict) else None
        if bucket is None:
            raise CrawlOrderError(f"{path}:{line_number} is not an order row")
        bucket.append(value)
    if len(by_kind[_HEADER_KIND]) != 1 or len(by_kind[_SEAL_KIND]) != 1:
        raise CrawlOrderError(f"{path} must carry exactly one header and one seal; the order is torn")
    header = by_kind[_HEADER_KIND][0]
    seal = str(by_kind[_SEAL_KIND][0].get("order_sha256"))
    return header, by_kind[_MEMBER_KIND], seal


def _members_from_rows(path: Path, rows: Sequence[JsonObject]) -> tuple[tuple[str, str, str], ...]:
    """Verify the ordinal chain and member uniqueness; return the member tuples."""

    members: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for expected_ordinal, row in enumerate(rows, start=1):
        if row.get("ordinal") != expected_ordinal:
            raise CrawlOrderError(
                f"{path} ordinal chain broken at {expected_ordinal}: row says {row.get('ordinal')!r}"
            )
        stable_id = row.get("stable_id")
        if not isinstance(stable_id, str) or not stable_id or stable_id in seen:
            raise CrawlOrderError(f"{path} row {expected_ordinal} lacks a unique stable_id")
        seen.add(stable_id)
        members.append((stable_id, str(row.get("stratum")), str(row.get("evidence"))))
    return tuple(members)


def _verify_binding(
    path: Path,
    order: CrawlOrder,
    *,
    expected_member_ids: Optional[frozenset[str]],
    expected_snapshot_sha256: Optional[str],
) -> None:
    """Refuse an order whose universe or snapshot differs from the bound root."""

    header = order.header
    if (
        expected_snapshot_sha256 is not None
        and header.get("intake_snapshot_sha256") != expected_snapshot_sha256
    ):
        raise CrawlOrderError(
            f"{path} was composed over snapshot {header.get('intake_snapshot_sha256')}, "
            f"not the bound root {expected_snapshot_sha256}"
        )
    if expected_member_ids is not None and set(order.stable_ids) != set(expected_member_ids):
        present = set(order.stable_ids)
        missing = sorted(set(expected_member_ids) - present)[:5]
        extra = sorted(present - set(expected_member_ids))[:5]
        raise CrawlOrderError(
            f"{path} member universe differs from the bound snapshot "
            f"(missing sample {missing}, extra sample {extra}); an order is never re-derived, "
            "so this is a records-state repair"
        )


def load_crawl_order(
    records_root: Path,
    campaign_id: str,
    *,
    expected_member_ids: Optional[frozenset[str]] = None,
    expected_snapshot_sha256: Optional[str] = None,
) -> Optional[CrawlOrder]:
    """Load and verify the persisted order, or return ``None`` when absent.

    Raises
    ------
    CrawlOrderError
        On any structural defect: bad header, broken ordinal chain, duplicate
        member, seal mismatch, campaign mismatch, or a member universe that no
        longer equals the bound snapshot's.
    """

    path = order_path(records_root, campaign_id)
    if not path.is_file():
        return None
    header, rows, seal = _read_order_lines(path)
    if header.get("schema_version") != CRAWL_ORDER_SCHEMA_VERSION:
        raise CrawlOrderError(
            f"{path} has schema {header.get('schema_version')!r}, not {CRAWL_ORDER_SCHEMA_VERSION!r}"
        )
    if header.get("campaign_id") != campaign_id:
        raise CrawlOrderError(
            f"{path} binds campaign {header.get('campaign_id')!r}, not {campaign_id!r}"
        )
    members = _members_from_rows(path, rows)
    if header.get("member_count") != len(members):
        raise CrawlOrderError(
            f"{path} header claims {header.get('member_count')} members; {len(members)} present"
        )
    order = CrawlOrder(header=header, members=members, order_sha256=seal)
    recomputed = _order_digest(header, order.rows())
    if recomputed != seal:
        raise CrawlOrderError(f"{path} seal {seal} does not match its content {recomputed}")
    _verify_binding(
        path,
        order,
        expected_member_ids=expected_member_ids,
        expected_snapshot_sha256=expected_snapshot_sha256,
    )
    return order


def ensure_crawl_order(
    records_root: Path,
    *,
    campaign_id: str,
    seed: int = DEFAULT_ORDER_SEED,
    stratified_width: int = DEFAULT_STRATIFIED_WIDTH,
    created_at: str,
    expected_member_ids: Optional[frozenset[str]] = None,
    expected_snapshot_sha256: Optional[str] = None,
) -> tuple[CrawlOrder, bool]:
    """Load the persisted order, or compose and persist it on first use.

    Returns
    -------
    tuple[CrawlOrder, bool]
        The order and whether THIS call composed it.
    """

    existing = load_crawl_order(
        records_root,
        campaign_id,
        expected_member_ids=expected_member_ids,
        expected_snapshot_sha256=expected_snapshot_sha256,
    )
    if existing is not None:
        return existing, False
    order = build_crawl_order(
        records_root,
        campaign_id=campaign_id,
        seed=seed,
        stratified_width=stratified_width,
        created_at=created_at,
    )
    if expected_member_ids is not None and set(order.stable_ids) != set(expected_member_ids):
        raise CrawlOrderError(
            "composed order does not cover the bound snapshot's member universe"
        )
    _persist(order_path(records_root, campaign_id), order)
    return order, True
