"""Remove a finished 2b lane's run-on checkout once the hub holds what it made (the tick's
``lane_reap`` step).

A lane's checkout is about a third of a gigabyte on its worker. Everything it made reaches the
hub: record writes go through the hub relay's verbs, and the relay pulls ``outputs/lane/``
(drafts, result, agent log) at ingest. run-on's own timer reaps a lane that exited 0 at once but
keeps a failed one (most often a lane that wrote no result) for its 24-hour failed grace, which
on a 12-lane worker holds several gigabytes for no use. This step asks run-on to reap one lane's
checkout, with no grace, once:

- the relay has ingested its outputs (``ingested.json``, any status but ``pull-failed``);
- its lease is gone (``admission`` releases an ended mini lease only after ingest);
- none of its families has a lane result file the ``lanes`` step has not committed yet;
- run-on's reaper holds its checkout only for the failed grace (``run-on-held.json``, the reaper's
  own record of the trees it keeps). A lane that exited 0 is gone at the timer's next run, and one
  held for unverified outputs is not this step's to release, so neither costs a reap call.

run-on still decides: it removes a checkout only when the job has stopped and every pulled file
still matches its hub copy, and it keeps the tree otherwise. Nothing here deletes a file itself.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

WINDOW_H = 48  # a lane ingested longer ago than this is left to run-on's own timer
LIMIT = 8  # reap calls one pass makes at most (each takes about 6 s)
SPARE_S = 60.0  # no call starts when the pass has less than this left
TIMEOUT_S = 90
DONE = "reaped.json"  # in the lane's relay state folder: run-on's answer, so it is asked once
BUSY = "another reap is running"
# run-on verdicts that may change on a later pass; any other answer is final for this step.
RETRY = ("KEEP:running", "KEEP:pull-in-progress", "KEEP:pull-before-end",
         "KEEP:outputs-changed-since-pull", "KEEP:bare-repo-missing", "KEEP:remove-failed",
         "KEEP:timeout")

# run-on's reasons for holding a tree that no grace means it may go once its pull is verified.
GRACE = ("failed-retention", "empty-pull-retention")

Reap = Callable[[str], tuple[int, str]]  # label: (exit code, output)
Held = Callable[[], "set[str] | None"]


def held_by_run_on() -> set[str] | None:
    """The menagerie jobs run-on's reaper keeps only for their failed grace, on any host; ``None``
    when its record does not read."""

    state = Path(os.environ.get("FLEET_STATE_DIR") or Path.home() / ".local/state/fleet")
    row = _read(state / "run-on-held.json")
    hosts = row.get("hosts")
    if not isinstance(hosts, Mapping):
        return None
    return {job for jobs in hosts.values() if isinstance(jobs, Mapping)
            for job, held in jobs.items()
            if isinstance(held, Mapping) and held.get("project") == "menagerie"
            and held.get("reason") in GRACE}


def run_on_reap(label: str, tool: str = "run-on") -> tuple[int, str]:
    """run-on's reaper for this one job, with no success or failure grace."""

    argv = [tool, "reap", "--job", label, "--project", "menagerie", "--ttl-hours", "0",
            "--failed-hours", "0", "--quick-failed-hours", "0", "--apply"]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        return 124, f"KEEP:timeout {label}"
    return done.returncode, done.stdout + done.stderr


def uncommitted_keys(repo: Path, data_rel: str, changed: Iterable[str]) -> set[str]:
    """The lease keys (families and ids) of the changed lane result files under the data root."""

    from menagerie.pipeline import lane_commit

    keys: set[str] = set()
    for path in changed:
        if not path.startswith(f"{data_rel}/"):
            continue
        target = repo / path
        parsed = lane_commit.check(target)[0] if target.is_file() else None
        keys.update(lane_commit.keys_of(path[len(data_rel) + 1:], parsed))
    return keys


def _read(path: Path) -> Mapping[str, Any]:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return row if isinstance(row, Mapping) else {}


def families(folder: Path, briefs: Path) -> list[str]:
    """The families a lane worked on: its staged folders directly under the 2b work root."""

    staged = _read(folder / "staged.json").get("paths") or {}
    return sorted(Path(k).name for k in staged if Path(k).parent == briefs)


def candidates(root: Path, leases: Path, briefs: Path, pending: set[str], held: set[str],
               now: datetime, window_h: float = WINDOW_H) -> tuple[list[str], dict[str, int]]:
    """Held labels ready to reap, newest ingest first, and counts of the ones waiting and why."""

    since = (now - timedelta(hours=window_h)).timestamp()
    ready: list[tuple[float, str]] = []
    waiting = {"lease": 0, "uncommitted": 0}
    for marker in root.glob("2b-mini-*/ingested.json"):
        folder = marker.parent
        if folder.name not in held:
            continue
        try:
            mtime = marker.stat().st_mtime
        except OSError:
            continue
        if mtime < since or (folder / DONE).exists():
            continue
        if _read(marker).get("status") in (None, "pull-failed"):
            continue
        if (leases / f"{folder.name}.json").exists():
            waiting["lease"] += 1
        elif pending.intersection(families(folder, briefs)):
            waiting["uncommitted"] += 1
        else:
            ready.append((mtime, folder.name))
    return [label for _, label in sorted(ready, reverse=True)], waiting


def verdict(label: str, output: str) -> str:
    """``reaped``, ``busy``, ``retry`` or the final ``KEEP:`` reason run-on gave."""

    if BUSY in output:
        return "busy"
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == label:
            if parts[0] == "REAPED":
                return "reaped"
            if parts[0].startswith("KEEP:"):
                return "retry" if parts[0] in RETRY else parts[0]
    return "gone"  # "(no jobs)": nothing left to reap


def step(root: Path, leases: Path, briefs: Path, pending: set[str], *, now: datetime,
         reap: Reap, held: set[str] | None, deadline: float, dry_run: bool = False,
         limit: int = LIMIT, window_h: float = WINDOW_H) -> dict[str, Any]:
    """One pass: reap up to ``limit`` ready lanes; a busy reaper ends the pass's calls."""

    if held is None:
        return {"skipped": "run-on's held record (run-on-held.json) does not read"}
    ready, waiting = candidates(root, leases, briefs, pending, held, now, window_h)
    out: dict[str, Any] = {"ready": len(ready), "waiting": waiting}
    if dry_run:
        return {**out, "would_reap": ready[:20]}
    tally: dict[str, int] = {}
    for label in ready[:limit]:
        if time.monotonic() + SPARE_S > deadline:
            break
        rc, output = reap(label)
        found = verdict(label, output)
        tally[found] = tally.get(found, 0) + 1
        if found == "busy":
            break
        if found != "retry":
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            (root / label / DONE).write_text(json.dumps({"verdict": found, "rc": rc, "at": stamp})
                                             + "\n", encoding="utf-8")
    return {**out, **tally}
