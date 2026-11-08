"""The tick's lane-stall alarm: no lane of any kind started for ``STALL_S`` while work waits.

Every lane writes its launch event under ``lanes/<run id>/`` in the work root, and fleet-run run
ids are UUIDv7, whose first 48 bits are the start in Unix milliseconds, so the newest id is the
last start (hub lanes, worker lanes and reviews alike; other folder names are ignored). The
reasons are read from the same pass's ``top_up`` and ``describes`` results. One FACT per stall:
the state remembers the last start it reported (``lane_stall_sent``).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STALL_S = 3 * 3600
UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}")
HELD = "held: "


def last_start(lanes: Path) -> datetime | None:
    """The newest lane's start, from its run id; ``None`` with no lane folder."""

    names = [n for n in os.listdir(lanes) if UUID7.fullmatch(n)] if lanes.is_dir() else []
    if not names:
        return None
    return datetime.fromtimestamp(int(max(names).replace("-", "")[:12], 16) / 1000, timezone.utc)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _parts(results: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """This pass's ``top_up`` result and the describe stage's last run."""

    return _dict(results.get("top_up")), _dict(_dict(results.get("describes")).get("last"))


def waiting(results: Mapping[str, Any]) -> dict[str, int]:
    """Work that waits for a lane, by kind, from this pass's results (absent kinds left out)."""

    top, last = _parts(results)
    found = {"2b": top.get("waiting_2b"), "2a": top.get("waiting_2a"),
             "describe": _dict(last.get("verdicts")).get("waiting")}
    return {k: int(v) for k, v in found.items() if isinstance(v, int) and v > 0}


def reasons(results: Mapping[str, Any]) -> list[str]:
    """Why this pass started nothing: each hold with its count, commonest first."""

    top, last = _parts(results)
    held = Counter(str(r)[len(HELD):] for r in top.get("runs") or () if str(r).startswith(HELD))
    for key in ("held", "skipped", "2b_held"):
        if top.get(key):
            held[f"top_up {key}: {top[key]}"] += 1
    if top.get("2b_repair_seat"):
        held["repair families kept off the Claude-only workers"] += len(top["2b_repair_seat"])
    for why, n in _dict(last.get("held")).items():
        held[f"describe {why}"] += int(n)
    return [f"{why} ({n})" for why, n in held.most_common()]


def check(state: dict[str, Any], results: Mapping[str, Any], lanes: Path,
          now: datetime) -> str | None:
    """The FACT to send, or ``None``: no lane started for ``STALL_S`` while work waits, and this
    stall (its last start) not reported yet. Records the report in ``state``."""

    start, work = last_start(lanes), waiting(results)
    if start is None or not work or (now - start).total_seconds() < STALL_S:
        return None
    if state.get("lane_stall_sent") == start.isoformat():
        return None
    state["lane_stall_sent"] = start.isoformat()
    hours = (now - start).total_seconds() / 3600
    queue = ", ".join(f"{k} {v}" for k, v in work.items())
    why = "; ".join(reasons(results)[:6]) or "no hold recorded this pass"
    return (f"no lane of any kind has started for {hours:.1f} h (last start "
            f"{start:%Y-%m-%d %H:%M} UTC) while work waits ({queue}). This pass: {why}.")
