"""Operator work-queue derivation, park dossier/aging, and the shadow operator census.

The failure class (W19, PLAN_v6 WS10): the operator census was HAND-COPIED and
wrong in composition -- 12 members reported as 14 rows, and the dossier's
"42-head review queue" prose went stale while the live ledger held 55
review-held heads. Every number here is therefore RE-DERIVED at execution from
the ledgers, with MEMBERS (stable_ids whose current head is review-held)
counted distinctly from ROWS (live park/grant rows those members carry). No
persisted status field steers any count.

Four derivations, one membership authority:

* :func:`derive_operator_work_queue` -- THE queue derivation. Members are the
  heads with ``status.human_review.required is True``; each member's rows are
  its live requeue grants plus its live grant-park and refusal-park rows.
  ``members_total`` and ``rows_total`` are separate, and the with/without
  live-grant split is derived from the same pass.
* :func:`park_aging_census` -- ages every LIVE park row against an EXPLICIT
  ``as_of`` (never wall-clock inside the derivation), buckets them into the
  count-by-age table a close names, and names SLA breaches and the oldest row.
* :func:`park_dossier` -- the per-member join of head status, live park rows,
  and live grants (default membership: the queue derivation above).
* :func:`shadow_operator_census` -- B2-X-V30: recompute one close's withheld
  set THREE ways and diff each against the set the close REPORTED, so a
  reported set that silently drops a member is caught, not trusted.

Shadow-census independence, disclosed honestly: (a) the STATUS view is derived
in this module from head status plus the two park ledgers with its own inline
release logic; (b) the DRIVER-STATE view is
``partition_proof._live_withholding_rows`` (the driver's own resolver -- the
in-package private reuse pattern, over the same underlying park-row loaders but
with the resolver's full valve universe and gauntlet); (c) the CLOSE-PROOF view
is ``withheld_here(load_partition_proof(...))``. The three share ledger bytes,
never the reported set: the census exists to catch REPORTING divergence, and
each view recomputes from evidence with zero producer-reported input.

This tool is the standing shadow operator census: the orchestrator runs the
``shadow`` verb on its ~15-minute pulse against the previous rung's close.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.close_object import close_object_path, load_close_objects
from menagerie.crawler.grant_parks import (
    grant_park_path,
    grant_park_released,
    load_grant_park_rows,
)
from menagerie.crawler.leg_generation import current_model_heads, model_head_citations
from menagerie.crawler.models import JsonObject
from menagerie.crawler.refusal_parks import (
    load_refusal_park_rows,
    refusal_park_path,
    refusal_park_released,
)

#: Closed age buckets, in ascending order: the count-by-age table a close names.
AGE_BUCKETS = ("<1d", "1-3d", "3-7d", ">=7d")

_AGE_BUCKET_FLOORS = ((7.0, ">=7d"), (3.0, "3-7d"), (1.0, "1-3d"), (0.0, "<1d"))


class OperatorWorkQueueError(ValueError):
    """Raised when a derivation input is malformed or a named authority is absent."""


def _parse_utc(value: str, *, field: str) -> datetime:
    """Parse one explicit ISO-8601 UTC timestamp, refusing anything else typed.

    Parameters
    ----------
    value:
        Timestamp text, ``Z`` suffix or explicit offset.
    field:
        Field name for the refusal message.

    Returns
    -------
    datetime.datetime
        Timezone-aware UTC instant.

    Raises
    ------
    OperatorWorkQueueError
        If the value is not an ISO-8601 timestamp with a timezone.
    """

    if not isinstance(value, str) or not value.strip():
        raise OperatorWorkQueueError(f"{field} must be an explicit ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OperatorWorkQueueError(
            f"{field} is not an ISO-8601 timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise OperatorWorkQueueError(f"{field} must carry an explicit timezone: {value!r}")
    return parsed.astimezone(timezone.utc)


def _age_days(created_at: str, as_of: datetime) -> float:
    """Return one row's age in days at an explicit instant, rounded to 1e-6."""

    created = _parse_utc(created_at, field="created_at")
    return round((as_of - created).total_seconds() / 86400.0, 6)


def _age_bucket(age_days: float) -> str:
    """Classify one age into the closed bucket vocabulary."""

    for floor, bucket in _AGE_BUCKET_FLOORS:
        if age_days >= floor:
            return bucket
    return "<1d"


def _live_park_rows(
    records_root: Path, heads: Mapping[str, Mapping[str, Any]]
) -> list[JsonObject]:
    """Return every LIVE park row (both families) tagged with its source family.

    Live means unvoided per the park modules' own per-row liveness AND
    unreleased against the member's RAW ledger head (BD-1: release is a fact
    about what landed).
    """

    live: list[JsonObject] = []
    for row in load_grant_park_rows([grant_park_path(records_root)]):
        if not grant_park_released(row, heads.get(str(row.get("stable_id")))):
            live.append({**row, "park_family": "park.grant"})
    for row in load_refusal_park_rows([refusal_park_path(records_root)]):
        if not refusal_park_released(row, heads.get(str(row.get("stable_id")))):
            live.append({**row, "park_family": "park.refusal"})
    return live


def _live_grants_by_member(records_root: Path) -> dict[str, list[JsonObject]]:
    """Return each member's live requeue grants: unvoided AND unconsumed.

    Liveness reuses the checkpoint module's own readers (the requeue tool's
    semantics): a grant row is live when no void row names its ``grant_id``
    and no ``requeue-grant-consumed`` operational event spent it.
    """

    from menagerie.crawler.checkpoint import (  # noqa: PLC2701 -- in-package reuse
        _recorded_grant_consumptions,
        _tail_recovered_requeue_rows,
    )

    ledger = records_root / "operational" / "requeue-grants.jsonl"
    if not ledger.is_file():
        return {}
    rows = _tail_recovered_requeue_rows(ledger)
    voided = {str(row["void_of"]) for row in rows if "void_of" in row}
    consumed = _recorded_grant_consumptions(ledger)
    live: dict[str, list[JsonObject]] = {}
    for row in rows:
        if "void_of" in row:
            continue
        grant_id = row.get("grant_id")
        stable_id = row.get("stable_id")
        if not isinstance(grant_id, str) or not isinstance(stable_id, str):
            continue
        if grant_id in voided or grant_id in consumed:
            continue
        live.setdefault(stable_id, []).append(dict(row))
    return live


def _review_required(head: Mapping[str, Any]) -> bool:
    """Return whether one raw head is review-held (the queue membership test)."""

    status = head.get("status")
    if not isinstance(status, Mapping):
        return False
    review = status.get("human_review")
    return isinstance(review, Mapping) and review.get("required") is True


def _head_field(head: Optional[Mapping[str, Any]], *path: str) -> Any:
    """Return one nested head field, or ``None`` where the shape ends early."""

    cursor: Any = head
    for key in path:
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(key)
    return cursor


def derive_operator_work_queue(records_root: Path, *, as_of: str) -> JsonObject:
    """Derive the operator work queue: THE one membership and row counting path.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    as_of:
        Explicit ISO-8601 UTC derivation instant (echoed; membership itself is
        instant-free -- it is the current ledger state).

    Returns
    -------
    dict[str, Any]
        ``members_total`` (distinct review-held stable_ids), ``rows_total``
        (live park + grant rows across members -- the 12-members/14-rows
        composition distinction), the with/without live-grant split, and the
        member table sorted by stable_id.
    """

    _parse_utc(as_of, field="as_of")
    heads = current_model_heads(records_root, require_ledger=True)
    members = sorted(stable_id for stable_id, head in heads.items() if _review_required(head))
    grants = _live_grants_by_member(records_root)
    parks_by_member: dict[str, list[JsonObject]] = {}
    for row in _live_park_rows(records_root, heads):
        parks_by_member.setdefault(str(row.get("stable_id")), []).append(row)

    table: list[JsonObject] = []
    rows_total = 0
    with_live_grants = 0
    for stable_id in members:
        head = heads.get(stable_id)
        live_grant_ids = sorted(str(row["grant_id"]) for row in grants.get(stable_id, []))
        park_row_shas = sorted(
            str(row["row_sha256"]) for row in parks_by_member.get(stable_id, [])
        )
        rows = len(live_grant_ids) + len(park_row_shas)
        rows_total += rows
        if live_grant_ids:
            with_live_grants += 1
        table.append(
            {
                "stable_id": stable_id,
                "status_code": _head_field(head, "status", "code"),
                "requested_at": _head_field(head, "status", "human_review", "requested_at"),
                "live_grant_ids": live_grant_ids,
                "park_row_shas": park_row_shas,
                "rows": rows,
            }
        )
    return {
        "as_of": as_of,
        "members_total": len(members),
        "rows_total": rows_total,
        "with_live_grants": with_live_grants,
        "without_live_grants": len(members) - with_live_grants,
        "members": table,
    }


def park_aging_census(records_root: Path, *, as_of: str, sla_days: float) -> JsonObject:
    """Age every live park row against an explicit instant; name SLA breaches.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    as_of:
        Explicit ISO-8601 UTC instant -- ages are NEVER derived from wall-clock
        inside the derivation (the CLI may default its flag to now).
    sla_days:
        Age (days) beyond which a live park row is an SLA breach.

    Returns
    -------
    dict[str, Any]
        Per-row ages, the closed ``count_by_age`` table, ``sla_breaches``
        (oldest first), and the ``oldest`` live row named, or ``None``.
    """

    instant = _parse_utc(as_of, field="as_of")
    if not isinstance(sla_days, (int, float)) or isinstance(sla_days, bool) or sla_days < 0:
        raise OperatorWorkQueueError("sla_days must be a non-negative number")
    heads = current_model_heads(records_root, require_ledger=True)
    rows: list[JsonObject] = []
    for row in _live_park_rows(records_root, heads):
        age = _age_days(str(row.get("created_at")), instant)
        rows.append(
            {
                "stable_id": str(row.get("stable_id")),
                "family": str(row.get("park_family")),
                "row_sha256": str(row.get("row_sha256")),
                "created_at": str(row.get("created_at")),
                "age_days": age,
            }
        )
    rows.sort(key=lambda row: (str(row["created_at"]), str(row["row_sha256"])))
    count_by_age = {bucket: 0 for bucket in AGE_BUCKETS}
    for row in rows:
        count_by_age[_age_bucket(float(row["age_days"]))] += 1
    breaches = [row for row in rows if float(row["age_days"]) > float(sla_days)]
    return {
        "as_of": as_of,
        "sla_days": float(sla_days),
        "rows_total": len(rows),
        "rows": rows,
        "count_by_age": count_by_age,
        "sla_breaches": breaches,
        "oldest": dict(rows[0]) if rows else None,
    }


def park_dossier(
    records_root: Path,
    *,
    stable_ids: Optional[Sequence[str]] = None,
    as_of: str,
) -> JsonObject:
    """Join head status, live park rows, and live grants per queue member.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    stable_ids:
        Explicit member scope; ``None`` derives the scope from the ONE queue
        membership derivation (review-held heads).
    as_of:
        Explicit ISO-8601 UTC instant used only for row ages.

    Returns
    -------
    dict[str, Any]
        Per-member dossier rows, sorted by stable_id.
    """

    instant = _parse_utc(as_of, field="as_of")
    heads = current_model_heads(records_root, require_ledger=True)
    if stable_ids is None:
        members = sorted(
            stable_id for stable_id, head in heads.items() if _review_required(head)
        )
    else:
        members = sorted(dict.fromkeys(str(stable_id) for stable_id in stable_ids))
    grants = _live_grants_by_member(records_root)
    parks_by_member: dict[str, list[JsonObject]] = {}
    for row in _live_park_rows(records_root, heads):
        parks_by_member.setdefault(str(row.get("stable_id")), []).append(row)

    table: list[JsonObject] = []
    for stable_id in members:
        head = heads.get(stable_id)
        parks = [
            {
                "family": str(row.get("park_family")),
                "disposition": str(row.get("disposition")),
                "refusal_class": _head_field(row, "refusal", "refusal_class"),
                "created_at": str(row.get("created_at")),
                "age_days": _age_days(str(row.get("created_at")), instant),
                "row_sha256": str(row.get("row_sha256")),
            }
            for row in parks_by_member.get(stable_id, [])
        ]
        parks.sort(key=lambda row: (str(row["created_at"]), str(row["row_sha256"])))
        live_grants = [
            {
                "grant_id": str(row.get("grant_id")),
                "stage": row.get("stage"),
                "reason": row.get("reason"),
                "created_at": row.get("created_at"),
            }
            for row in grants.get(stable_id, [])
        ]
        live_grants.sort(key=lambda row: str(row["grant_id"]))
        table.append(
            {
                "stable_id": stable_id,
                "status_code": _head_field(head, "status", "code"),
                "requested_at": _head_field(head, "status", "human_review", "requested_at"),
                "live_grants": live_grants,
                "parks": parks,
            }
        )
    return {"as_of": as_of, "members_total": len(members), "members": table}


def _status_derived_withheld(
    records_root: Path, campaign_id: str, heads: Mapping[str, Mapping[str, Any]]
) -> frozenset[str]:
    """STATUS view: members withheld by a live park row, per head status.

    Inline release logic (independent of the resolver's): a grant park is
    released when the member's head revision moved past the blocked parent; a
    refusal park is released by ANY landed head when parent-less, else by a
    moved revision. Rows outside ``campaign_id`` never count.
    """

    withheld: set[str] = set()
    for row in load_grant_park_rows([grant_park_path(records_root)]):
        if str(row.get("campaign_id")) != campaign_id:
            continue
        stable_id = str(row.get("stable_id"))
        revision = _head_field(heads.get(stable_id), "record_revision")
        blocked = _head_field(row, "pending", "blocked_parent_revision")
        if isinstance(revision, str) and revision != blocked:
            continue
        withheld.add(stable_id)
    for row in load_refusal_park_rows([refusal_park_path(records_root)]):
        if str(row.get("campaign_id")) != campaign_id:
            continue
        stable_id = str(row.get("stable_id"))
        revision = _head_field(heads.get(stable_id), "record_revision")
        blocked = _head_field(row, "refusal", "blocked_parent_revision")
        if isinstance(revision, str) and (blocked is None or revision != blocked):
            continue
        withheld.add(stable_id)
    return frozenset(withheld)


def shadow_operator_census(
    records_root: Path,
    *,
    campaign_id: str,
    close: Optional[Mapping[str, Any]] = None,
    close_id: Optional[str] = None,
) -> JsonObject:
    """B2-X-V30: recompute one close's withheld set three ways and diff reported.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Campaign whose ledgers corroborate the withholding.
    close:
        The close object itself, or ``None`` to resolve ``close_id``.
    close_id:
        ``close_sha256`` of the close under audit, when ``close`` is absent.

    Returns
    -------
    dict[str, Any]
        Per-source sorted sets, per-source diffs against the reported set,
        every divergence named, and ``status`` of ``consistent`` or
        ``divergent``.

    Raises
    ------
    OperatorWorkQueueError
        If neither a close nor a resolvable ``close_id`` is supplied, or the
        close discloses no ``withheld_here`` arm.
    menagerie.crawler.partition_proof.PartitionProofError
        If the close's bound proof is absent or fails its content address --
        an absent authority never reads as an empty withheld set.
    """

    from menagerie.crawler.partition_proof import (  # noqa: PLC2701 -- in-package reuse
        _live_withholding_rows,
        load_partition_proof,
        withheld_here,
    )

    if close is None:
        if not close_id:
            raise OperatorWorkQueueError("shadow census needs a close object or a close_id")
        matches = [
            row
            for row in load_close_objects([close_object_path(records_root)])
            if str(row.get("close_sha256")) == str(close_id)
        ]
        if not matches:
            raise OperatorWorkQueueError(
                f"no close with close_sha256 {close_id!r} exists under {records_root}"
            )
        close = matches[-1]
    reported_arm = _head_field(close, "partition", "withheld_here")
    if not isinstance(reported_arm, Sequence) or isinstance(reported_arm, (str, bytes)):
        raise OperatorWorkQueueError(
            "close discloses no withheld_here arm: nothing to audit against"
        )
    reported = frozenset(str(member) for member in reported_arm)

    heads = current_model_heads(records_root, require_ledger=True)
    citations = model_head_citations(records_root, require_ledger=True)
    status_view = _status_derived_withheld(records_root, campaign_id, heads)
    driver_view = frozenset(
        _live_withholding_rows(
            records_root, campaign_id, citations, tolerate_pilot_campaign=False
        )
    )
    proof = load_partition_proof(records_root, str(close["partition_proof_sha256"]))
    proof_view = withheld_here(proof)

    sources = {
        "status_derived": sorted(status_view),
        "driver_state": sorted(driver_view),
        "close_proof": sorted(proof_view),
    }
    diffs: dict[str, JsonObject] = {}
    divergences: list[str] = []
    for name, view in (
        ("status_derived", status_view),
        ("driver_state", driver_view),
        ("close_proof", proof_view),
    ):
        missing = sorted(view - reported)
        extra = sorted(reported - view)
        diffs[f"{name}_vs_reported"] = {
            "missing_from_reported": missing,
            "extra_in_reported": extra,
        }
        for member in missing:
            divergences.append(
                f"{name} withholds {member} but the close's reported set omits it"
            )
        for member in extra:
            divergences.append(
                f"the close reports {member} withheld but {name} does not derive it"
            )
    for left, right in (
        ("status_derived", "driver_state"),
        ("status_derived", "close_proof"),
        ("driver_state", "close_proof"),
    ):
        for member in sorted(frozenset(sources[left]) ^ frozenset(sources[right])):
            divergences.append(f"{left} and {right} disagree about {member}")
    return {
        "campaign_id": campaign_id,
        "close_sha256": str(close.get("close_sha256")),
        "partition_proof_sha256": str(close.get("partition_proof_sha256")),
        "reported": sorted(reported),
        "sources": sources,
        "diffs": diffs,
        "divergences": divergences,
        "status": "divergent" if divergences else "consistent",
    }


def _utc_now() -> str:
    """Return now in the ledger timestamp format (CLI default only)."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    """Build the operator work-queue tool argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Typed parser with the ``queue``, ``aging``, ``dossier``, and
        ``shadow`` verbs.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    verbs = parser.add_subparsers(dest="verb", required=True)

    def _common(verb: argparse.ArgumentParser) -> None:
        verb.add_argument("--repo-root", type=Path, default=Path.cwd())
        verb.add_argument("--records-root", type=Path)

    queue = verbs.add_parser("queue", help="derive the operator work queue")
    _common(queue)
    queue.add_argument("--as-of", help="explicit ISO-8601 UTC instant (default: now)")

    aging = verbs.add_parser("aging", help="park SLA + aging census")
    _common(aging)
    aging.add_argument("--as-of", help="explicit ISO-8601 UTC instant (default: now)")
    aging.add_argument("--sla-days", type=float, default=3.0)

    dossier = verbs.add_parser("dossier", help="per-member park dossier")
    _common(dossier)
    dossier.add_argument("--as-of", help="explicit ISO-8601 UTC instant (default: now)")
    dossier.add_argument(
        "--stable-id",
        dest="stable_ids",
        action="append",
        help="explicit member scope (repeatable; default: all queue members)",
    )

    shadow = verbs.add_parser("shadow", help="B2-X-V30 shadow operator census")
    _common(shadow)
    shadow.add_argument("--campaign", required=True)
    shadow.add_argument("--close-id", required=True, metavar="CLOSE_SHA256")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one derivation verb and print its JSON payload.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero on success; non-zero on a typed refusal or a DIVERGENT shadow
        census (the census's whole point is a loud nonzero on divergence).
    """

    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    records_root = args.records_root or (repo_root / "menagerie" / "crawler" / "records")
    try:
        if args.verb == "queue":
            payload = derive_operator_work_queue(records_root, as_of=args.as_of or _utc_now())
        elif args.verb == "aging":
            payload = park_aging_census(
                records_root, as_of=args.as_of or _utc_now(), sla_days=args.sla_days
            )
        elif args.verb == "dossier":
            payload = park_dossier(
                records_root,
                stable_ids=args.stable_ids,
                as_of=args.as_of or _utc_now(),
            )
        else:
            payload = shadow_operator_census(
                records_root, campaign_id=args.campaign, close_id=args.close_id
            )
    except (OSError, ValueError) as exc:
        print(f"operator-work-queue failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    if payload.get("status") == "divergent":
        print(
            "shadow operator census DIVERGENT: " + "; ".join(payload["divergences"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
