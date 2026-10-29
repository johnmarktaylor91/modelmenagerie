"""List paid custody refusals or drain one after operator-verified resolution."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

from menagerie.crawler.driver import DriverLock, DriverLockError
from menagerie.crawler.identity import utc_now
from menagerie.crawler.paid_session_custody import (
    ALIVE,
    FREE,
    PaidCustodyLease,
    leader_is_provably_gone,
    process_group_liveness,
    subtree_liveness,
)
from menagerie.crawler.paid_session_lifecycle import PaidSessionAuthority, PaidSessionError


def build_parser() -> argparse.ArgumentParser:
    """Build the paid-custody listing and operator-drain parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser requiring an explicit lease confirmation for every drain.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drain", metavar="LEASE_ID")
    parser.add_argument("--confirm-resolved", metavar="LEASE_ID")
    parser.add_argument("--reason")
    parser.add_argument("--drained-by", default="operator")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    return parser


def _is_drainable(lease: PaidCustodyLease) -> bool:
    """Return whether one lease is MACHINE-provably resolved.

    The lock oracle alone is not enough: a live subtree holds its ``flock`` on an
    inode, so unlinking the (nonsecret, runtime-root) lock pathname used to make
    the oracle report a live orphan dead and let an operator retire it -- the one
    guard the drain had.  Draining therefore requires the lock to be genuinely
    FREE on the inode the binding pinned, the paid process group to be empty,
    AND the bound leader to be provably gone.  An UNANSWERABLE lock (pathname
    deleted or replaced) is never sufficient by itself; only the independent
    leader-identity proof can carry that case.
    """

    lock_state = subtree_liveness(lease.lock_path, expected_identity=lease.lock_identity)
    if lock_state == ALIVE:
        return False
    if process_group_liveness(lease.leader_pgid) == ALIVE:
        return False
    if lock_state == FREE:
        return True
    return leader_is_provably_gone(lease)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """List unresolved leases or append one explicit operator custody drain.

    Parameters
    ----------
    argv:
        Optional command arguments excluding the executable name.

    Returns
    -------
    int
        Zero after a listing or durable drain, otherwise one.
    """

    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    records_root = args.records_root or (repo_root / "menagerie" / "crawler" / "records")
    runtime_root = args.runtime_root or (repo_root / ".crawl-local")
    authority = PaidSessionAuthority(
        records_root / "operational" / "spend.jsonl",
        runtime_root=runtime_root,
        clock=utc_now,
    )
    try:
        leases = authority.outstanding_custody_leases()
        if args.drain is None:
            payload = []
            for lease in leases:
                row = asdict(lease)
                row["lock_path"] = str(lease.lock_path)
                row["kernel_lock"] = subtree_liveness(
                    lease.lock_path, expected_identity=lease.lock_identity
                )
                row["process_group"] = process_group_liveness(lease.leader_pgid)
                row["drainable"] = _is_drainable(lease)
                payload.append(row)
            print(json.dumps({"paid_custody_leases": payload}, sort_keys=True))
            return 0
        if args.confirm_resolved != args.drain:
            raise ValueError(
                "--confirm-resolved must repeat the exact --drain lease after the orphan "
                "has been resolved"
            )
        if not isinstance(args.reason, str) or not args.reason.strip():
            raise ValueError("--drain requires --reason")
        matches = [lease for lease in leases if lease.lease_id == args.drain]
        if len(matches) != 1:
            raise ValueError(f"no unique outstanding paid custody lease {args.drain!r}")
        lease = matches[0]
        if not _is_drainable(lease):
            raise ValueError(
                "paid custody is not provably resolved; the subtree's lock, its process "
                "group, or its bound leader identity still shows life. Resolve the live "
                "subtree before draining -- deleting its lock file does not resolve it."
            )
        owner = {
            "pid": os.getpid(),
            "run_id": "paid-session-custody-tool",
            "target": lease.lease_id,
        }
        with DriverLock(runtime_root / "locks" / "driver.lock", owner):
            authority.drain_custody(
                lease,
                operator=str(args.drained_by),
                reason=args.reason,
            )
        print(
            json.dumps(
                {"drained": lease.lease_id, "spend_id": lease.spend_id},
                sort_keys=True,
            )
        )
    except (DriverLockError, OSError, PaidSessionError, ValueError) as exc:
        print(f"paid-session-custody failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
