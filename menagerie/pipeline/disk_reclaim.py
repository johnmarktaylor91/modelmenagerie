"""Reclaim worker disk before the guard pauses new work (the tick's ``reclaim`` step).

When the worker's free space (the ``disk`` step's reading) is under ``below_gib``, at most once
every ``every_min`` minutes, the tick starts one detached shell on the hub that runs each owner's
own cleanup: run-on's reaper for this project with the failed-tree grace shortened to
``failed_hours`` (retention tightens under pressure; a tree still goes only once its outputs are
recorded), then one worker job running ``prune_overlays_job.sh`` (overlays and their uv cache,
keeping every overlay a queued or claimed request names). Nothing here deletes a file itself.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULTS = {"below_gib": 75.0, "every_min": 60, "failed_hours": 2}
JOB = "menagerie/pipeline/prune_overlays_job.sh"

Launch = Callable[[str, Path, Path], int]


def _launch(script: str, log: Path, repo: Path) -> int:
    """Start ``script`` under bash in ``repo``, detached from the tick; return its pid."""

    with open(log, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(["bash", "-c", script], stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=repo, start_new_session=True)
    return proc.pid


def script(host: str, *, failed_hours: int, ref: str = "main") -> str:
    """The reclaim commands, run from the repository on the hub."""

    reap = ["run-on", "reap", "--host", host, "--project", "menagerie", "--ttl-hours", "0",
            "--failed-hours", str(failed_hours), "--apply"]
    prune = ["run-on", "submit", "--project", "menagerie", "--host", host, "--os", "linux",
             "--ref", ref, "--cpus", "1", "--memory-gib", "2", "--admit-wait", "60",
             "--no-outputs", "--", "bash", JOB]
    return (f"echo \"$(date -u +%FT%TZ) reclaim start\"; {shlex.join(reap)}; "
            f"{shlex.join(prune)}; echo \"$(date -u +%FT%TZ) reclaim submitted\"")


def step(state: Mapping[str, Any] | None, free_gib: float | None, *, now: datetime,
         cfg: Mapping[str, Any] | None, repo: Path, log: Path,
         launch: Launch = _launch) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decide and, when due, start one reclaim. Returns (new state, result for the tick log)."""

    conf = {**DEFAULTS, **(cfg or {})}
    last = dict(state or {})
    if free_gib is None or free_gib >= float(conf["below_gib"]):
        return last, {"skipped": f"free {free_gib} GiB, line {conf['below_gib']:g}"}
    host = compute_host(repo)
    if not host:
        return last, {"skipped": "no worker host in .compute-host"}
    started = datetime.fromisoformat(last["at"]) if last.get("at") else None
    if started is not None and now - started < timedelta(minutes=int(conf["every_min"])):
        return last, {"skipped": f"last reclaim at {last['at']}"}
    pid = launch(script(host, failed_hours=int(conf["failed_hours"])), log, repo)
    new = {"at": now.isoformat(timespec="seconds"), "free_gib": free_gib, "pid": pid}
    return new, {"started": new}


def compute_host(repo: Path) -> str | None:
    """The project's default worker (the untracked ``.compute-host``), or None."""

    path = repo / ".compute-host"
    name = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    return name or None
