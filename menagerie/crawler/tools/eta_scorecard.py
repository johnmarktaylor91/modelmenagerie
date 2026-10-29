"""OB-M11b: the ETA scorecard -- prediction ledger + post-run scorer (SHADOW).

200a launched on a modeled "3-8h" estimate; actual wall was 89.7h. The
monitor's terminal-rate ETA was n/a or misleading exactly when the operator
asked (author-heavy phases, calibration withhold), and no gate ever asked "is
this ETA CORRECT?" because nothing scored one against ground truth. This
module makes ETA quality a measured, per-rung regression:

* every tick appends a prediction row (funnel vector, rates, reachability,
  eta_p50/p90 with basis) to a SIDECAR ledger -- never under the records root
  (OB-Q4: monitor-state sidecar first; records-root promotion only after the
  floors pass);
* the bottleneck ETA is Little's law per stage: hours-to-drain = input /
  hourly flow; a stage with input and NO flow makes the campaign
  ``blocked:<stage>`` -- an honest reachability state instead of an
  extrapolated zero;
* the post-run scorer replays the ledger against the actual completion
  instant and emits the error table (the 200a backtest is the baseline the
  method must beat).

SHADOW-FIRST (B2-X-O34/B2-X-O15): nothing here sends. The monitor's delivered
text keeps the M5 net-rate ETA with its closed basis vocabulary; a scorecard
ETA reaches a text only after its conservation/false-alarm/accuracy floors
pass on a real rung and JMT confirms the p50<=25%-after-25%-completion target.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import durable_fsync
from menagerie.crawler.tools.launch_monitor import Snapshot

SCORECARD_SCHEMA_VERSION = "menagerie.crawler.eta-scorecard-row.v1"

#: Closed reachability vocabulary for one prediction row.
REACHABILITY_STATES = frozenset(
    {"complete", "flowing", "blocked", "insufficient-evidence"}
)

#: Prior spread factor for p90 until per-rung scoring replaces it (disclosed
#: in every row as its basis).
DEFAULT_P90_FACTOR = 2.0


def predict(snapshot: Snapshot) -> dict[str, Any]:
    """Build one prediction row from a snapshot (pure; never sends).

    The ETA is the max of the net-terminal-rate drain and the slowest stage's
    Little's-law drain; a starved stage (input>0, flow=0) yields
    ``blocked:<stage>`` with NO eta -- a blocked pipeline has no honest finite
    estimate, and 200a proved an extrapolated one drives bad interventions.
    """

    remaining = snapshot.in_flight
    if remaining == 0:
        reachability = "complete"
        eta_p50: Optional[float] = 0.0
        bottleneck: Optional[str] = None
    elif snapshot.starved_stages:
        reachability = "blocked"
        eta_p50 = None
        bottleneck = snapshot.starved_stages[0]
    elif snapshot.net_rate_per_hour and snapshot.net_rate_per_hour > 0:
        reachability = "flowing"
        drains: dict[str, float] = {
            "terminal-rate": remaining / snapshot.net_rate_per_hour
        }
        window_hours = max(snapshot.flow_window_minutes, 1.0) / 60.0
        for stage in ("author", "exec", "gate", "terminal"):
            flow = snapshot.flows.get(
                {
                    "author": "author-sessions",
                    "exec": "exec-attempts",
                    "gate": "gate-adjudications",
                    "terminal": "net-terminals",
                }[stage],
                0,
            )
            queue = snapshot.stage_inputs.get(stage, 0)
            if queue > 0 and flow > 0:
                drains[stage] = queue / (flow / window_hours)
        bottleneck = max(drains, key=lambda key: drains[key])
        eta_p50 = drains[bottleneck] * 3600.0
    else:
        reachability = "insufficient-evidence"
        eta_p50 = None
        bottleneck = None
    return {
        "schema_version": SCORECARD_SCHEMA_VERSION,
        "ts": snapshot.generated_at,
        "remaining": remaining,
        "net_terminals": snapshot.net_terminals,
        "net_rate_per_hour": snapshot.net_rate_per_hour,
        "flows": dict(snapshot.flows),
        "stage_inputs": dict(snapshot.stage_inputs),
        "starved_stages": list(snapshot.starved_stages),
        "effective_width": snapshot.effective_width,
        "configured_width": snapshot.configured_width,
        "leg_cycling": snapshot.leg_cycling,
        "reachability": reachability,
        "bottleneck": bottleneck,
        "eta_p50_seconds": eta_p50,
        "eta_p90_seconds": None if eta_p50 is None else eta_p50 * DEFAULT_P90_FACTOR,
        "p90_basis": f"prior-factor-{DEFAULT_P90_FACTOR}",
    }


def append_row(ledger_path: Path, row: Mapping[str, Any]) -> None:
    """Append one prediction row durably to the sidecar ledger."""

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "ab") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True).encode("utf-8") + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())


def read_rows(ledger_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        payload = ledger_path.read_bytes()
    except OSError:
        return rows
    for line in payload.split(b"\n"):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _parse(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def score(
    rows: Sequence[Mapping[str, Any]], actual_completion_at: datetime
) -> dict[str, Any]:
    """Score a prediction ledger against the actual completion instant.

    Emits the per-rung error table: median absolute relative error overall
    and after 25% completion (the B2-X-O15 target's denominator), plus the
    blocked/insufficient coverage so an ETA that hid behind "blocked" for the
    whole rung cannot read as accurate. Never raises on an empty ledger --
    it reports n=0 and no error, which no floor may read as a pass.
    """

    total = max((row.get("remaining", 0) + row.get("net_terminals", 0)) for row in rows) if rows else 0
    errors: list[tuple[float, float]] = []  # (completion_fraction, rel_error)
    states: dict[str, int] = {}
    for row in rows:
        states[str(row.get("reachability"))] = states.get(str(row.get("reachability")), 0) + 1
        ts = _parse(row.get("ts"))
        eta = row.get("eta_p50_seconds")
        if ts is None or not isinstance(eta, (int, float)) or eta is None:
            continue
        actual_seconds = (actual_completion_at - ts).total_seconds()
        if actual_seconds <= 0:
            continue
        fraction = (
            row.get("net_terminals", 0) / total if total else 0.0
        )
        errors.append((fraction, abs(eta - actual_seconds) / actual_seconds))

    def _median(values: list[float]) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    late = [error for fraction, error in errors if fraction >= 0.25]
    return {
        "n_rows": len(rows),
        "n_scored": len(errors),
        "reachability_census": dict(sorted(states.items())),
        "p50_abs_rel_error": _median([error for _fraction, error in errors]),
        "p50_abs_rel_error_after_25pct": _median(late),
        "n_after_25pct": len(late),
        "actual_completion_at": actual_completion_at.isoformat(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.tools.eta_scorecard",
        description="Score one shadow ETA prediction ledger against ground truth.",
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--actual-completion-at", required=True)
    args = parser.parse_args(argv)
    actual = _parse(args.actual_completion_at)
    if actual is None:
        print("unparseable --actual-completion-at", file=sys.stderr)
        return 64
    print(json.dumps(score(read_rows(args.ledger), actual), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
