"""LC-11 alarm surface: one read-only invocation for the monitor cadence.

Runs both LC-11 derivations (generation span + continuation budget) against a
records root and exits 0 (quiet), 2 (alarm -- payload names every breach), or
3 (the alarm itself could not run). WS9's monitor lane owns the schedule;
this tool is one invocation, exactly like the DX-24 shadow-close probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.continuation_alarm import (
    LC11_GENERATION_SPAN_ALARM_HOURS,
    LC11_GENERATION_SPAN_ALARM_RUNS,
    continuation_budget_alarm,
    generation_span_alarm,
)

ALARM_QUIET = 0
ALARM_RAISED = 2
ALARM_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    """Return the alarm tool's argument parser."""

    parser = argparse.ArgumentParser(prog="continuation-alarm", description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument(
        "--max-span-hours", type=float, default=LC11_GENERATION_SPAN_ALARM_HOURS
    )
    parser.add_argument("--max-runs", type=int, default=LC11_GENERATION_SPAN_ALARM_RUNS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one read-only LC-11 alarm pass."""

    args = build_parser().parse_args(argv)
    try:
        span = generation_span_alarm(
            args.records_root,
            now=datetime.now(timezone.utc),
            max_span_hours=args.max_span_hours,
            max_runs=args.max_runs,
        )
        budget = continuation_budget_alarm(args.records_root)
    except Exception as exc:  # noqa: BLE001 -- a broken alarm is a fact, never a clean exit
        print(f"continuation-alarm failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return ALARM_ERROR
    payload = {
        "alarm": "lc11-continuation",
        "generation_span": span,
        "continuation_budget": budget,
    }
    print(json.dumps(payload, sort_keys=True))
    if span["alarms"] or budget["alarms"]:
        print(
            "LC-11 alarm: continuation loop / generation span breach -- act "
            "before rung end",
            file=sys.stderr,
        )
        return ALARM_RAISED
    return ALARM_QUIET


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
