"""List grant-pending supersession parks, or void one, without rewriting history."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.driver import DriverLock, DriverLockError
from menagerie.crawler.grant_parks import (
    GrantParkError,
    append_grant_park_row,
    append_grant_park_void,
    build_grant_park_void,
    build_selection_exclusion_park_row,
    grant_park_path,
    load_grant_park_rows,
)


def _utc_now() -> str:
    """Return the current UTC timestamp in the ledger's format."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    """Build the grant-park tool argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Typed parser for listing live parks or appending one operator void.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--void",
        dest="void_of",
        metavar="ROW_SHA256",
        help="append an operator void event releasing one live park row",
    )
    parser.add_argument(
        "--park-exclusion",
        dest="park_exclusion",
        metavar="STABLE_ID",
        help=(
            "append one SELECTION-EXCLUSION park for a member whose current "
            "head is a review-held failure terminal (P5 R2-1): the park reads "
            "the hold (live while that exact revision governs), selection "
            "excludes the member absolutely in every mode, and no requeue "
            "grant releases it. The held revision is derived from the LIVE "
            "models ledger, never typed by hand"
        ),
    )
    parser.add_argument(
        "--decision",
        help=(
            "frozen decision the exclusion executes, e.g. "
            "PD-7-slop-family-routes (required with --park-exclusion)"
        ),
    )
    parser.add_argument(
        "--campaign",
        help="frozen tier campaign id the member belongs to (required with --park-exclusion)",
    )
    parser.add_argument(
        "--reason", help="operator justification (required with --void / --park-exclusion)"
    )
    parser.add_argument("--voided-by", default="operator")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--records-root", type=Path)
    return parser


def _held_head(records_root: Path, stable_id: str) -> dict:
    """Return one member's LIVE review-held failure-terminal head, or refuse.

    The exclusion park binds the exact revision the LIVE ledger governs with
    (the registry-identity-derivation rule: identities come from live records,
    never from operator-typed strings), and the head must actually carry the
    hold the park reads — a hold-less or non-failed head has nothing for the
    exclusion's live predicate to hold on to.
    """

    from menagerie.crawler.recordio import scan_jsonl
    from menagerie.crawler.reducer import _select_current

    models_path = records_root / "models" / "current-shard.jsonl"
    head = _select_current(scan_jsonl(models_path)).get(stable_id)
    if head is None:
        raise GrantParkError(f"{stable_id} holds no revision in {models_path}")
    status = head.get("status") or {}
    if status.get("kind") != "failed":
        raise GrantParkError(
            f"{stable_id}'s current head is {status.get('kind')!r}, not a failure "
            "terminal; a selection exclusion parks review-held failure terminals only"
        )
    if not (status.get("human_review") or {}).get("required"):
        raise GrantParkError(
            f"{stable_id}'s current head carries no review hold; the exclusion park "
            "READS the hold — append the review hold through the terminal-review "
            "lane first (and let the next leg adopt it), then park"
        )
    return dict(head)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """List live grant-pending parks, or durably void one.

    Parameters
    ----------
    argv:
        Optional command arguments, excluding the executable name.

    Returns
    -------
    int
        Zero after a listing or durable void append, otherwise non-zero.
    """

    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    records_root = args.records_root or (repo_root / "menagerie" / "crawler" / "records")
    ledger = grant_park_path(records_root)
    try:
        if args.park_exclusion is not None:
            if not args.reason:
                raise ValueError(
                    "--park-exclusion requires --reason: an exclusion is an operator decision"
                )
            if not args.decision:
                raise ValueError(
                    "--park-exclusion requires --decision naming the frozen decision "
                    "it executes"
                )
            if not args.campaign:
                raise ValueError("--park-exclusion requires --campaign")
            head = _held_head(records_root, args.park_exclusion)
            owner = {
                "pid": os.getpid(),
                "run_id": "grant-parks-tool",
                "target": None,
            }
            with DriverLock(repo_root / ".crawl-local" / "locks" / "driver.lock", owner):
                payload = append_grant_park_row(
                    ledger,
                    build_selection_exclusion_park_row(
                        stable_id=args.park_exclusion,
                        name=str(head.get("name") or args.park_exclusion),
                        campaign_id=args.campaign,
                        machine_id=os.uname().nodename,
                        created_at=_utc_now(),
                        held_revision=str(head["record_revision"]),
                        held_status_code=str((head.get("status") or {}).get("code")),
                        decision=args.decision,
                        reason=args.reason,
                    ),
                )
            print(json.dumps(payload, sort_keys=True))
            return 0
        if args.void_of is None:
            rows = load_grant_park_rows([ledger])
            print(json.dumps({"review_holds": list(rows)}, sort_keys=True))
            return 0
        if not args.reason:
            raise ValueError("--void requires --reason: a void is an operator decision")
        owner = {
            "pid": os.getpid(),
            "run_id": "grant-parks-tool",
            "target": None,
        }
        with DriverLock(repo_root / ".crawl-local" / "locks" / "driver.lock", owner):
            payload = append_grant_park_void(
                ledger,
                build_grant_park_void(
                    void_of=args.void_of,
                    reason=args.reason,
                    voided_by=args.voided_by,
                    created_at=_utc_now(),
                ),
            )
        print(json.dumps(payload, sort_keys=True))
    except (DriverLockError, GrantParkError, OSError, ValueError) as exc:
        print(f"grant-parks failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
