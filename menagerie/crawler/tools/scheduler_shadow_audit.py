"""Projection-only shadow audit for scheduler cutover/rollback readiness.

D1's cutover step 4 (Sol list; M4 territory): before switching scheduler mode
mid-campaign, "run the planner in projection-only mode and require each intake
ID to have exactly one explainable next state". This tool is that audit. It
reads ONLY durable state -- the intake snapshot, the canonical ledgers, the
deferral/park valve ledgers, and the requeue-grant ledger -- buys no session,
executes nothing, and appends nothing. Every intake ID is classified into
exactly one state from the closed vocabulary below; any ID that cannot be
explained fails the audit (exit 2).

Classification vocabulary (closed):

- ``settled-runs``           current ``runs`` record, no catch-up owed.
- ``settled-terminal``       current non-``runs`` record (terminal classes).
- ``pending-runs-catchup``   ``runs`` record owing its R1/R2 catch-up gate.
- ``requeue-armed``          terminal record with an unconsumed operator grant;
                             the next leg re-drives it.
- ``stale-record-redrive``   a ledger head exists but is dependency-stale
                             (identity/environment rotation); next leg re-runs.
- ``deferral-row-present``   no record; a durable machine-deferral valve row
                             names it (the next leg's valve re-adjudicates the
                             claim -- an audit never claims from a blind read).
- ``park-row-present``       no record; a durable grant-pending or refusal
                             park row names it (same re-adjudication rule).
- ``awaiting-representative``no record; family variant whose representative
                             has not settled usably yet.
- ``awaiting-author``        no record; the author lane is its next state.

Usage::

    python -m menagerie.crawler.tools.scheduler_shadow_audit \
        --records-root RECORDS --intake-root INTAKE_SNAPSHOT_DIR \
        --runtime-root RUNTIME [--author-model ...] [--checker-model ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.authority import build_authority_context
from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.driver_admission import restored_environment_generations
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.intake import IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer, default_ledger_paths
from menagerie.crawler.status import MACHINE_DEFERRAL_VALVES
from menagerie.crawler.rung_staging import (
    registered_prior_intake_members,
    registered_prior_intake_snapshots,
)

#: Closed classification vocabulary; anything else is an audit failure.
AUDIT_STATES = (
    "settled-runs",
    "settled-terminal",
    "pending-runs-catchup",
    "requeue-armed",
    "stale-record-redrive",
    "deferral-row-present",
    "park-row-present",
    "awaiting-representative",
    "awaiting-author",
)


def _pending_metadata_readmission(record: Mapping[str, Any]) -> bool:
    """Mirror the driver's shared R1/R2 catch-up predicate for offline use."""

    from menagerie.crawler.driver import _pending_metadata_readmission as shared

    return shared(record)


def _rows_by_stable_id(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return the stable IDs named by a sequence of valve/park rows."""

    return {
        str(row["stable_id"]) for row in rows if isinstance(row, Mapping) and row.get("stable_id")
    }


def _unconsumed_grant_ids(
    grants_paths: Sequence[Path], operational_records: Sequence[Mapping[str, Any]]
) -> set[str]:
    """Return stable IDs holding at least one unconsumed requeue grant.

    Both grant spellings are consulted: the runtime-root ledger the driver
    reads (``DriverPaths.requeue_grants``) and the canonical
    ``records/operational`` ledger the operator tools write.
    """

    consumed = {
        str(event.get("details", {}).get("grant_id"))
        for event in operational_records
        if event.get("event_kind") == "requeue-grant-consumed"
    }
    armed: set[str] = set()
    for grants_path in grants_paths:
        if not grants_path.exists():
            continue
        for grant in scan_jsonl(grants_path):
            if str(grant.get("grant_id")) not in consumed:
                armed.add(str(grant.get("stable_id")))
    return armed


def resolve_ledger_paths(records_root: Path) -> Any:
    """Detect the campaign's ledger layout below one records root.

    Parameters
    ----------
    records_root:
        Canonical records directory in either the sharded production layout
        (``models/current-shard.jsonl``) or the flat layout
        (``models.jsonl``) the driver rigs use.

    Returns
    -------
    LedgerPaths
        Resolved models/attempts/gates paths.
    """

    from menagerie.crawler.leg_generation import close_ledger_sources
    from menagerie.crawler.models import LedgerPaths

    sources = close_ledger_sources(records_root)
    if sources["models.shard"].is_file():
        return default_ledger_paths(records_root)
    if sources["models.flat"].is_file():
        return LedgerPaths(
            sources["models.flat"],
            sources["attempts.flat"],
            sources["gates.flat"],
        )
    raise FileNotFoundError(
        f"no models ledger found under {records_root} (checked sharded and flat layouts)"
    )


def shadow_audit(
    snapshot: IntakeSnapshot,
    ledgers: Any,
    runtime_root: Path,
    *,
    author_model: Optional[str] = None,
    checker_model: Optional[str] = None,
) -> dict[str, Any]:
    """Classify every intake ID into exactly one explainable next state.

    Parameters
    ----------
    snapshot:
        Verified intake snapshot the campaign runs against.
    ledgers:
        Resolved canonical ``LedgerPaths``.
    runtime_root:
        Campaign runtime directory (requeue grants live here).
    author_model, checker_model:
        Authority identities for the dependency-currency projection; default
        to the driver's configured defaults.

    Returns
    -------
    dict[str, Any]
        ``{"states": {stable_id: state}, "counts": {state: n},
        "unexplained": [stable_id, ...]}``.
    """

    config = DriverConfig()
    # Valve/park ledgers hang off the same root the driver derives them from.
    valve_root = ledgers.models.parent.parent
    operational_path = canonical_operational_ledger_path(ledgers.models)
    operational_records = tuple(scan_jsonl(operational_path)) if operational_path.exists() else ()
    base_context = build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=author_model or config.author_model,
        author_version=config.author_version,
        checker_model=checker_model or config.checker_model,
        checker_version=config.checker_version,
        prior_intake_snapshots=registered_prior_intake_snapshots(
            snapshot.root, snapshot.snapshot_id
        ),
        prior_intake_member_ids=registered_prior_intake_members(
            snapshot.root, snapshot.snapshot_id
        ),
    )
    from dataclasses import replace

    context = replace(
        base_context,
        environment_generations=restored_environment_generations(
            base_context.environment_generations,
            ledgers.attempts,
            operational_records,
        ),
    )
    with CanonicalReducer(ledgers, context) as reducer:
        current = dict(reducer.current_records)

    heads: dict[str, Mapping[str, Any]] = {}
    if ledgers.models.exists():
        for record in scan_jsonl(ledgers.models):
            heads[str(record["stable_id"])] = record

    # Every machine-deferral valve ledger, from the single registry beside the
    # partition authority (OB-M1): the previous hand-picked tuples covered five
    # of the eight ledgers, so a host-capacity park, a coverage deferral, or a
    # scheduler quarantine read as unexplained here.
    deferral_ids: set[str] = set()
    park_ids: set[str] = set()
    for valve in MACHINE_DEFERRAL_VALVES:
        path = valve_root / valve.relative_path
        if not path.exists():
            continue
        ids = _rows_by_stable_id(valve.load_rows([path]))
        if valve.family == "park":
            park_ids |= ids
        else:
            deferral_ids |= ids

    from menagerie.crawler.checkpoint import canonical_requeue_grants_path

    armed = _unconsumed_grant_ids(
        (
            runtime_root / "requeue-grants.jsonl",
            canonical_requeue_grants_path(ledgers.models),
        ),
        operational_records,
    )

    settled_usable: set[str] = {
        stable_id
        for stable_id, record in current.items()
        if not _pending_metadata_readmission(record) and stable_id not in armed
    }

    states: dict[str, str] = {}
    unexplained: list[str] = []
    for item in snapshot.items:
        stable_id = item.stable_id
        record = current.get(stable_id)
        if record is not None:
            kind = str(record.get("status", {}).get("kind", ""))
            if _pending_metadata_readmission(record):
                state = "pending-runs-catchup"
            elif stable_id in armed:
                state = "requeue-armed"
            elif kind == "runs":
                state = "settled-runs"
            elif kind:
                state = "settled-terminal"
            else:
                unexplained.append(stable_id)
                continue
        elif stable_id in heads:
            state = "stale-record-redrive"
        elif stable_id in deferral_ids:
            state = "deferral-row-present"
        elif stable_id in park_ids:
            state = "park-row-present"
        elif (
            item.declares_family_variant
            and item.family_representative_id != stable_id
            and item.family_representative_id not in settled_usable
        ):
            state = "awaiting-representative"
        else:
            state = "awaiting-author"
        states[stable_id] = state

    counts: dict[str, int] = {state: 0 for state in AUDIT_STATES}
    for state in states.values():
        counts[state] += 1
    return {
        "states": states,
        "counts": {state: count for state, count in counts.items() if count},
        "unexplained": unexplained,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the audit and refuse (exit 2) on any unexplainable intake ID."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--intake-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--author-model", default=None)
    parser.add_argument("--checker-model", default=None)
    parser.add_argument("--json", action="store_true", help="emit the full JSON report")
    args = parser.parse_args(argv)

    snapshot = load_intake_snapshot(args.intake_root)
    report = shadow_audit(
        snapshot,
        resolve_ledger_paths(args.records_root),
        args.runtime_root,
        author_model=args.author_model,
        checker_model=args.checker_model,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for state, count in sorted(report["counts"].items()):
            print(f"{state:24s} {count}")
        if report["unexplained"]:
            print(f"UNEXPLAINED: {', '.join(report['unexplained'])}")
    if report["unexplained"]:
        print(
            "shadow audit REFUSED: intake IDs without one explainable next state",
            file=sys.stderr,
        )
        return 2
    print(f"shadow audit OK: {len(report['states'])} intake IDs explainable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
