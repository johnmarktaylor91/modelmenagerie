"""List live scheduler quarantines, or append one explicit release.

Three release modes (W12/PD-11), all append-only against a LIVE row:

- ``--void ROW --reason R`` -- BARE void: clears the row, restores NO retry
  budget (the next fault on the same work generation holds immediately).
- ``--void ROW --reason R --fixing-sha S --test-node N --canary C`` --
  REMEDIED void (the DX-69 shape): declares the actual repair and restores
  the budget for exactly one fresh retry.
- ``--drain ROW --reason R --fixed-in-tree SHA`` -- disclosed MACHINE DRAIN:
  refused until the bounded fresh-leg retries are exhausted, resets the
  counter, and binds next-leg admission to the tree carrying the fix
  (admission refuses while ``HEAD`` lacks the sha).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

from menagerie.crawler.driver import DriverLock, DriverLockError
from menagerie.crawler.recordio import operational_events_path, scan_jsonl
from menagerie.crawler.scheduler_parks import (
    SchedulerQuarantineError,
    append_scheduler_quarantine_void,
    build_scheduler_quarantine_drain,
    build_scheduler_quarantine_remedied_void,
    build_scheduler_quarantine_void,
    load_scheduler_quarantine_ledger,
    load_scheduler_quarantine_rows,
    scheduler_quarantine_path,
    scheduler_quarantine_released,
)


def _utc_now() -> str:
    """Return the current UTC timestamp in the ledger's format."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    """Build the scheduler-quarantine lister/release parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for listing live rows or appending one release decision.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--void",
        dest="void_of",
        metavar="ROW_SHA256",
        help=(
            "append an operator void releasing one live scheduler quarantine; "
            "bare unless the full remedy triple is declared"
        ),
    )
    parser.add_argument(
        "--drain",
        dest="drain_of",
        metavar="ROW_SHA256",
        help=(
            "append a disclosed machine drain (requires --fixed-in-tree and "
            "exhausted retries); resets the retry budget"
        ),
    )
    parser.add_argument("--reason", help="operator justification (required with any release)")
    parser.add_argument("--voided-by", default="operator")
    parser.add_argument(
        "--fixing-sha", help="remedied void: the fixing commit sha (DX-69 shape)"
    )
    parser.add_argument(
        "--test-node", help="remedied void: the RED-pinned test node reproducing the fault"
    )
    parser.add_argument(
        "--canary", help="remedied void: the passing-canary evidence"
    )
    parser.add_argument(
        "--fixed-in-tree",
        dest="fixed_in_tree",
        metavar="SHA",
        help="machine drain: commit the running tree must contain before the next leg",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--records-root", type=Path)
    return parser


def _build_release(args: argparse.Namespace, ledger: Path, records_root: Path) -> dict:
    """Assemble and precondition-check the requested release row."""

    if args.void_of is not None and args.drain_of is not None:
        raise ValueError("--void and --drain are separate decisions; pass exactly one")
    if not args.reason:
        raise ValueError("a release requires --reason: it is an operator decision")
    target = args.void_of if args.void_of is not None else args.drain_of
    live = {str(row["row_sha256"]): row for row in load_scheduler_quarantine_rows([ledger])}
    if target not in live:
        raise ValueError(f"no LIVE scheduler quarantine carries row_sha256 {target!r}")

    remedy_parts = (args.fixing_sha, args.test_node, args.canary)
    if args.drain_of is not None:
        if any(part is not None for part in remedy_parts):
            raise ValueError(
                "--fixing-sha/--test-node/--canary belong to the remedied --void; "
                "a --drain declares --fixed-in-tree instead"
            )
        if not args.fixed_in_tree:
            raise ValueError(
                "--drain requires --fixed-in-tree: the drain binds next-leg "
                "admission to the tree that carries the fix"
            )
        rows = load_scheduler_quarantine_ledger([ledger])
        operational = operational_events_path(records_root)
        events = scan_jsonl(operational, validate=False) if operational.is_file() else []
        if scheduler_quarantine_released(live[target], events, ledger_rows=rows):
            raise ValueError(
                "machine drain refused: the bounded fresh-leg retries are not "
                "exhausted for this work generation; let the automatic retry run "
                "(or declare a remedied --void)"
            )
        return build_scheduler_quarantine_drain(
            void_of=target,
            reason=args.reason,
            drained_by=args.voided_by,
            created_at=_utc_now(),
            fixed_in_tree_sha=args.fixed_in_tree,
        )

    if args.fixed_in_tree is not None:
        raise ValueError("--fixed-in-tree belongs to --drain")
    declared = [part for part in remedy_parts if part is not None]
    if declared and len(declared) != 3:
        raise ValueError(
            "a remedied void declares the COMPLETE remedy shape: --fixing-sha, "
            "--test-node, and --canary together (a partial remedy restores nothing)"
        )
    if declared:
        return build_scheduler_quarantine_remedied_void(
            void_of=target,
            reason=args.reason,
            voided_by=args.voided_by,
            created_at=_utc_now(),
            fixing_sha=str(args.fixing_sha),
            test_node=str(args.test_node),
            canary=str(args.canary),
        )
    return build_scheduler_quarantine_void(
        void_of=target,
        reason=args.reason,
        voided_by=args.voided_by,
        created_at=_utc_now(),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """List live scheduler quarantines, or append one explicit release.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero after a listing or durable release, otherwise non-zero.
    """

    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    records_root = args.records_root or (repo_root / "menagerie" / "crawler" / "records")
    ledger = scheduler_quarantine_path(records_root)
    try:
        if args.void_of is None and args.drain_of is None:
            rows = load_scheduler_quarantine_rows([ledger])
            print(json.dumps({"scheduler_quarantines": list(rows)}, sort_keys=True))
            return 0
        owner = {
            "pid": os.getpid(),
            "run_id": "scheduler-quarantines-tool",
            "target": None,
        }
        with DriverLock(repo_root / ".crawl-local" / "locks" / "driver.lock", owner):
            payload = append_scheduler_quarantine_void(
                ledger, _build_release(args, ledger, records_root)
            )
        print(json.dumps(payload, sort_keys=True))
    except (DriverLockError, SchedulerQuarantineError, OSError, ValueError) as exc:
        print(f"scheduler-quarantines failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
