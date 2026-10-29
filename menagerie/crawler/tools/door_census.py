"""Door census (AB-10): first-reachability disclosure per production stage.

A production stage that has never executed is a door 200b opens first — the
class behind the day-4 first-fire incidents. This census counts REAL
executions per stage from the attempt ledger rows and DISCLOSES every stage
whose lifetime execution count is below the floor. It is a disclosure
instrument, deliberately additional to (never a replacement for) the
blindness-class manifest the certificate enforces: the manifest proves an
instrument exists per blindness class; this census says which doors have
actually swung.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from menagerie.crawler.constants import FAILURE_REASON_CODES

CENSUS_SCHEMA_VERSION = "menagerie.crawler.door-census.v1"

#: The closed production stage vocabulary: the stage-keyed failure-reason map
#: is the one authority every attempt row's ``stage`` must belong to.
PRODUCTION_STAGES = frozenset(FAILURE_REASON_CODES)


def door_census(attempt_rows: Iterable[Mapping[str, Any]], *, floor: int = 1) -> dict[str, Any]:
    """Count executions per stage; disclose dark and foreign stages.

    Parameters
    ----------
    attempt_rows:
        Attempt ledger rows carrying a ``stage`` field.
    floor:
        Minimum lifetime execution count below which a stage is disclosed as
        a first-reachability door (default 1: never-executed).
    """

    counts: Counter[str] = Counter()
    foreign: Counter[str] = Counter()
    for row in attempt_rows:
        stage = str(row.get("stage"))
        if stage in PRODUCTION_STAGES:
            counts[stage] += 1
        else:
            foreign[stage] += 1
    dark = sorted(stage for stage in PRODUCTION_STAGES if counts[stage] < floor)
    return {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "floor": floor,
        "executions_by_stage": {stage: counts[stage] for stage in sorted(PRODUCTION_STAGES)},
        "dark_stages": dark,
        "foreign_stage_rows": dict(sorted(foreign.items())),
        "status": "all-doors-exercised" if not dark else "dark-doors-disclosed",
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load one JSONL ledger, skipping blank lines, refusing non-objects."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(dict(value))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Render the door census for one attempt ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", type=Path, required=True, help="attempt ledger JSONL")
    parser.add_argument("--floor", type=int, default=1)
    parser.add_argument(
        "--refuse-dark",
        action="store_true",
        help="exit nonzero when any stage is below the floor (gate composition)",
    )
    args = parser.parse_args(argv)
    census = door_census(_load_jsonl(args.attempts), floor=args.floor)
    print(json.dumps(census, indent=2, sort_keys=True))
    if args.refuse_dark and census["dark_stages"]:
        print(
            f"door-census: dark stages below floor {args.floor}: {census['dark_stages']}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
