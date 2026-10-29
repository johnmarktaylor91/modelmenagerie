"""List terminal-refusal parks, or void one, without rewriting history."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.driver import DriverLock, DriverLockError
from menagerie.crawler.refusal_parks import (
    RefusalParkError,
    append_refusal_park_void,
    build_refusal_park_void,
    load_refusal_park_rows,
    refusal_park_path,
)


def _utc_now() -> str:
    """Return the current UTC timestamp in the ledger's format."""

    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_parser() -> argparse.ArgumentParser:
    """Build the refusal-park tool argument parser.

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
    parser.add_argument("--reason", help="operator justification (required with --void)")
    parser.add_argument("--voided-by", default="operator")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--records-root", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """List live terminal-refusal parks, or durably void one.

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
    ledger = refusal_park_path(records_root)
    try:
        if args.void_of is None:
            rows = load_refusal_park_rows([ledger])
            print(json.dumps({"refusal_parks": list(rows)}, sort_keys=True))
            return 0
        if not args.reason:
            raise ValueError("--void requires --reason: a void is an operator decision")
        owner = {
            "pid": os.getpid(),
            "run_id": "refusal-parks-tool",
            "target": None,
        }
        with DriverLock(repo_root / ".crawl-local" / "locks" / "driver.lock", owner):
            payload = append_refusal_park_void(
                ledger,
                build_refusal_park_void(
                    void_of=args.void_of,
                    reason=args.reason,
                    voided_by=args.voided_by,
                    created_at=_utc_now(),
                ),
            )
        print(json.dumps(payload, sort_keys=True))
    except (DriverLockError, RefusalParkError, OSError, ValueError) as exc:
        print(f"refusal-parks failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
