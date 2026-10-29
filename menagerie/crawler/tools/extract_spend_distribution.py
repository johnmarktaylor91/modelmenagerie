"""Extract the measured paid-session wall distribution from a spend ledger (AB-12).

The W17 virtual-time scheduler rehearsal must replay the MEASURED 200a wall
distribution, not an invented one: every throughput bug of the launch was
invisible at toy scale with toy durations. This tool distills a campaign's
``spend.jsonl`` into the committed replay artifact
(``menagerie/crawler/tests/spend_wall_distribution_200a.json``): per-lane
sorted machine wall vectors plus provenance (source path, source sha256, row
counts) and the summary quantiles the plan quotes (median 243s / p90 1663s /
max 2666s), so drift between the artifact and its source is mechanically
visible.

Usage::

    python -m menagerie.crawler.tools.extract_spend_distribution \
        --spend /path/to/records/operational/spend.jsonl \
        [--out menagerie/crawler/tests/spend_wall_distribution_200a.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

SCHEMA_VERSION = "menagerie.crawler.spend-wall-distribution.v1"
_DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "spend_wall_distribution_200a.json"
)


def _quantiles(walls: list[float]) -> dict[str, float]:
    """Return the summary quantiles the rehearsal plan quotes."""

    ordered = sorted(walls)
    return {
        "median": statistics.median(ordered),
        "p90": ordered[int(len(ordered) * 0.9)],
        "max": ordered[-1],
    }


def extract(spend_path: Path) -> dict:
    """Distill one spend ledger into the committed distribution artifact."""

    raw = spend_path.read_bytes()
    walls_by_lane: dict[str, list[float]] = {}
    settled = 0
    reserved_lane_by_spend: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event = row.get("event")
        if event == "session_reserved":
            reserved_lane_by_spend[str(row.get("spend_id"))] = str(row.get("lane"))
        if event != "session_settled":
            continue
        machine = row.get("machine")
        if not isinstance(machine, dict):
            continue
        wall = machine.get("wall_seconds")
        if not isinstance(wall, (int, float)):
            continue
        settled += 1
        lane = reserved_lane_by_spend.get(str(row.get("spend_id")), "unknown")
        walls_by_lane.setdefault(lane, []).append(float(wall))
    if not walls_by_lane:
        raise ValueError(f"{spend_path} carries no settled machine walls")
    all_walls = [wall for walls in walls_by_lane.values() for wall in walls]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": str(spend_path),
        "source_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "settled_sessions": settled,
        "quantiles": _quantiles(all_walls),
        "lanes": {
            lane: {
                "count": len(walls),
                "quantiles": _quantiles(walls),
                "wall_seconds": sorted(walls),
            }
            for lane, walls in sorted(walls_by_lane.items())
        },
    }


def main(argv: list[str]) -> int:
    """Extract and write the distribution artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spend", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args(argv[1:])
    artifact = extract(args.spend)
    args.out.write_text(
        json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    quantiles = artifact["quantiles"]
    print(
        f"extract_spend_distribution: wrote {args.out} — "
        f"{artifact['settled_sessions']} settled, median {quantiles['median']:.0f}s / "
        f"p90 {quantiles['p90']:.0f}s / max {quantiles['max']:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
