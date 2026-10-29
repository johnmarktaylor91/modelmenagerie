"""Restore one operator-adjudicated eaten award, or ratify a leak re-award (AW-3).

Two append-only operator dispositions over the award-eating incident:

- ``restore``: the m5273 shape. The current head is a failure carrying the
  typed ``award-superseded-without-grant:<target>`` scar; the reducer proves
  the five acceptance conditions from ledger bytes (ancestor, naming scar,
  evidence disjointness, exact content, recorded-basis replay) and re-appends
  the awarded content as a new head carrying the
  ``award-restored-by-operator:<event_id>`` scar. ``--dry-run`` proves and
  reports without appending anything.
- ``ratify``: the m4334/m7546 shape. The model already re-won ``runs`` through
  the full gated path (the work-selection leak's re-drive); no restoration
  append is owed. The disposition names the EATEN revision and EVERY leak
  award in the chain (m7546 was re-awarded three times: seq 232, 265, 288;
  m4334 twice: seq 227, 262 -- naming only the final head would misreport how
  many re-drives fired), ratifying the current head as the governing award.

Restored awards are segregated in the derived census as ``restored``; they are
never silently counted among earned awards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.driver import DriverLock, DriverLockError
from menagerie.crawler.identity import utc_now
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger, LedgerError, scan_jsonl
from menagerie.crawler.reducer import (
    AWARD_RESTORATION_EVENT_DISPOSITION,
    CanonicalReducer,
    ReductionError,
)


def _campaign_event(
    *,
    disposition: str,
    details: dict,
    run_id: str,
    machine_id: str,
    status: str,
    created_at: str,
    identity_payload: dict,
) -> dict:
    """Assemble one semantically-keyed campaign-health disposition event.

    The identity is the INCIDENT's fact key (disposition, stable id, exact
    revisions), never the clock: a retried disposition replays into the
    already-persisted row instead of double-appending (H1 F4), and a retry
    that contradicts the persisted details refuses as a conflicting replay.
    """

    event = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": created_at,
        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
        "status": status,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": run_id,
        "machine_id": machine_id,
        "details": {"disposition": disposition, **details},
    }
    event = mint_operational_event(
        event, identity_payload={"disposition": disposition, **identity_payload}
    )
    return event


def build_parser() -> argparse.ArgumentParser:
    """Build the restore/ratify tool argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("stable_id")
        subparser.add_argument("--reason", required=True)
        subparser.add_argument("--granted-by", default="operator")
        subparser.add_argument("--repo-root", type=Path, default=Path.cwd())
        subparser.add_argument("--records-root", type=Path)

    restore = actions.add_parser("restore", help="restore one scarred eaten award")
    add_common(restore)
    restore.add_argument(
        "--intake",
        type=Path,
        required=True,
        help="intake snapshot directory (the same authority the driver ran under)",
    )
    restore.add_argument("--restored-award-revision", required=True)
    restore.add_argument("--superseding-revision", required=True)
    restore.add_argument(
        "--dry-run",
        action="store_true",
        help="prove every acceptance condition and report; append nothing",
    )

    ratify = actions.add_parser(
        "ratify", help="ratify a leak re-award chain as the governing award"
    )
    add_common(ratify)
    ratify.add_argument("--eaten-revision", required=True)
    ratify.add_argument(
        "--leak-award-revision",
        dest="leak_award_revisions",
        action="append",
        required=True,
        metavar="RECORD_REVISION",
        help=(
            "every runs revision the re-drive leak minted for this model, in "
            "chain order, ending with the current governing head; repeatable"
        ),
    )
    return parser


def _records_root(args: argparse.Namespace) -> Path:
    if args.records_root is not None:
        return args.records_root.resolve()
    return args.repo_root.resolve() / "menagerie" / "crawler" / "records"


def _ledger_paths(records_root: Path) -> LedgerPaths:
    return LedgerPaths(
        models=records_root / "models" / "current-shard.jsonl",
        attempts=records_root / "attempts" / "local.jsonl",
        gates=records_root / "gates" / "current-shard.jsonl",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Append one restoration (event then revision) or one ratification event."""

    args = build_parser().parse_args(argv)
    records_root = _records_root(args)
    repo_root = args.repo_root.resolve()
    events_path = records_root / "operational" / "events.jsonl"
    owner = {"pid": os.getpid(), "run_id": "restore-superseded-award", "target": None}
    created_at = utc_now()
    try:
        with DriverLock(repo_root / ".crawl-local" / "locks" / "driver.lock", owner):
            if args.action == "ratify":
                # H1 F1: the governing head comes from the SAME union-of-layouts
                # parentage-validated projection the reducer/status lane and
                # `discharge_generation` use (the rt3 B' third-private-resolver
                # lesson) -- a raw last-row read of one layout attested a
                # SUPERSEDED award as governing whenever the true head lived in
                # models.jsonl.
                from menagerie.crawler.leg_generation import models_source_paths
                from menagerie.crawler.state import _select_current

                model_rows = [
                    dict(row)
                    for path in models_source_paths(records_root)
                    for row in scan_jsonl(path, validate=False)
                    if row.get("stable_id") == args.stable_id
                ]
                if not model_rows:
                    raise ReductionError(f"no model rows exist for {args.stable_id}")
                head = _select_current(model_rows).get(args.stable_id)
                if head is None:
                    raise ReductionError(f"no current head exists for {args.stable_id}")
                if head.get("status", {}).get("kind") != "runs":
                    raise ReductionError(
                        "ratification requires a governing runs head; "
                        f"{args.stable_id} holds {head.get('status', {}).get('code')!r}"
                    )
                if str(head.get("record_revision")) != args.leak_award_revisions[-1]:
                    raise ReductionError(
                        "the last named leak award must be the current governing "
                        f"head: head={head.get('record_revision')}"
                    )
                known = {str(row.get("record_revision")) for row in model_rows}
                missing = [
                    revision
                    for revision in [args.eaten_revision, *args.leak_award_revisions]
                    if revision not in known
                ]
                if missing:
                    raise ReductionError(
                        f"named revisions are not persisted for {args.stable_id}: {missing}"
                    )
                event = _campaign_event(
                    disposition="award-ratified",
                    details={
                        "stable_id": args.stable_id,
                        "eaten_revision": args.eaten_revision,
                        "leak_award_revisions": list(args.leak_award_revisions),
                        "governing_revision": str(head.get("record_revision")),
                        "reason": args.reason,
                        "granted_by": args.granted_by,
                    },
                    run_id="restore-superseded-award",
                    machine_id="operator",
                    status=OperationalEventStatus.DEGRADED.value,
                    created_at=created_at,
                    identity_payload={
                        "stable_id": args.stable_id,
                        "eaten_revision": args.eaten_revision,
                        "leak_award_revisions": list(args.leak_award_revisions),
                    },
                )
                with JsonlLedger(events_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
                    persisted = operational.append(event).record
                print(json.dumps(persisted, sort_keys=True))
                return 0

            # restore -- the projection authority is the F12 pair the status
            # boundary uses: snapshot lineage plus ledger-restored per-family
            # environment generations.
            from dataclasses import replace as dc_replace

            from menagerie.crawler.cli import (
                _snapshot_authority_context,
                _snapshot_driver_config,
            )
            from menagerie.crawler.driver_admission import (
                restored_environment_generations,
            )
            from menagerie.crawler.intake import load_intake_snapshot

            snapshot = load_intake_snapshot(args.intake)
            context = _snapshot_authority_context(
                snapshot, _snapshot_driver_config(repo_root, snapshot)
            )
            context = dc_replace(
                context,
                environment_generations=restored_environment_generations(
                    context.environment_generations,
                    records_root / "attempts" / "local.jsonl",
                    tuple(scan_jsonl(events_path)) if events_path.is_file() else (),
                ),
            )
            event = _campaign_event(
                disposition=AWARD_RESTORATION_EVENT_DISPOSITION,
                details={
                    "stable_id": args.stable_id,
                    "restored_award_revision": args.restored_award_revision,
                    "superseding_revision": args.superseding_revision,
                    "reason": args.reason,
                    "granted_by": args.granted_by,
                },
                run_id="restore-superseded-award",
                machine_id="operator",
                status=OperationalEventStatus.DEGRADED.value,
                created_at=created_at,
                identity_payload={
                    "stable_id": args.stable_id,
                    "restored_award_revision": args.restored_award_revision,
                    "superseding_revision": args.superseding_revision,
                },
            )
            with CanonicalReducer(_ledger_paths(records_root), context) as reducer:
                if args.dry_run:
                    report = reducer.append_award_restoration(
                        stable_id=args.stable_id,
                        restored_award_revision=args.restored_award_revision,
                        superseding_revision=args.superseding_revision,
                        restoration_event_id=event["event_id"],
                        dry_run=True,
                    )
                    print(json.dumps(report, sort_keys=True))
                    return 0
                # Event first, then the revision (section 2.3's append order):
                # the decision survives even if the revision append refuses.
                with JsonlLedger(events_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
                    operational.append(event)
                report = reducer.append_award_restoration(
                    stable_id=args.stable_id,
                    restored_award_revision=args.restored_award_revision,
                    superseding_revision=args.superseding_revision,
                    restoration_event_id=event["event_id"],
                )
            print(json.dumps(report, sort_keys=True))
    except (DriverLockError, LedgerError, OSError, ReductionError, ValueError) as exc:
        print(f"restore-superseded-award failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
