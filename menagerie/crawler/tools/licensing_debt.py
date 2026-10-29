"""Leg-start licensing-debt census + the explicit operator ack arm (SD-2).

``status`` prints the exact census the leg-open refusal derives -- same
implementation, read-only. ``ack`` durably acknowledges ONE census by its
content digest; the digest is recomputed from evidence at every leg open, so
an acknowledgment can never cover debt it has not seen.

Exit codes for ``status``: 0 = no debt; 2 = outstanding debt (census printed);
3 = the census itself could not run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.leg_licensing_guard import (
    acknowledge_licensing_debt,
    licensing_debt_census,
    read_licensing_debt_ack,
)

STATUS_CLEAN = 0
STATUS_IN_DEBT = 2
STATUS_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    """Return the licensing-debt tool's argument parser."""

    parser = argparse.ArgumentParser(prog="licensing-debt", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="derive and print the debt census")
    status.add_argument("--records-root", type=Path, required=True)
    status.add_argument("--campaign-id", required=True)
    # Item 11: no ``--repo-root``. The award-closure axis is derived through
    # the register tool's live identity, never hashed natively from a tree.
    ack = commands.add_parser("ack", help="durably acknowledge one exact census")
    ack.add_argument("--records-root", type=Path, required=True)
    ack.add_argument("--debt-sha256", required=True)
    ack.add_argument("--reason", required=True)
    ack.add_argument("--acked-by", default="operator")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the tool."""

    args = build_parser().parse_args(argv)
    if args.command == "status":
        try:
            census = licensing_debt_census(args.records_root, campaign_id=args.campaign_id)
        except Exception as exc:  # noqa: BLE001 -- tool boundary: exit code 3, never a stack trace
            print(f"licensing-debt census could not run: {exc}", file=sys.stderr)
            return STATUS_ERROR
        census["ack"] = read_licensing_debt_ack(args.records_root)
        print(json.dumps(census, indent=2, sort_keys=True))
        return STATUS_IN_DEBT if census["in_debt"] else STATUS_CLEAN
    payload = acknowledge_licensing_debt(
        args.records_root,
        debt_sha256=args.debt_sha256,
        reason=args.reason,
        acked_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        acked_by=args.acked_by,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover -- CLI shim
    raise SystemExit(main())
