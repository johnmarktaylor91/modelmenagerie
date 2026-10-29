"""Describe seats on the worker runtimes: 2b first, the hub the fallback (``describes.workers``).

With ``describes.workers`` true the describe stage passes ``WORKERS`` (the most describe seats
that may run on workers at once: ``describes.worker_seats``, default the describe seat cap) and
``describe_write._launch`` tries a worker before the hub. A worker seat runs through the 2b
lanes' machinery: ``mini_placement.start_describe`` (worker checks, runtime, identity, lease,
run-on) and ``mini_relay``, which stages the batch folder's ``families/`` and ``out/`` with the
translated brief and ingests ``out/<family>.json`` and the result file back into the hub
folder. It takes no hub slot, so the hub gate does not apply to it; the seat cap counts both
placements. Nothing the lane does reaches the hub except through that ingest.

2b comes first by one rule. Each tick pass that plans 2b lanes (full or refill) records the
describe room per worker (``ROOM``): the worker slots the pass left free after placing every 2b
lane it wanted, plus the describe seats already there. When the pass filled every free worker
slot with 2b (it may want more) or 2b is held, there is no room anywhere. A describe seat starts
on a worker only while that worker's describe seats are under its room, so it never takes a slot
the last pass gave 2b; a slot 2b wants later goes back to 2b when the describe lane in it ends
(at most one describe lane's run). A room older than ``ROOM_FRESH_S`` is none, so describe stays
off the workers whenever the tick is not planning. The trace host's disk guard, runner share and
night rules hold a describe seat exactly as a 2b lane (``mini_placement._admissible``).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from menagerie.paths import work_dir
from menagerie.pipeline import admission, mini_placement, mini_relay, now

WORKERS = "MENAGERIE_DESCRIBE_WORKERS"  # describe seats that may run on workers; unset or 0: none
ROOM = "describe-room.json"  # in the tick folder: the last 2b plan's describe room per worker
ROOM_FRESH_S = 1200  # two full tick passes; the refill pass rewrites it more often


def seats() -> int:
    """The worker seat cap the describe stage passed (``WORKERS``); 0 is the hub only."""

    return max(0, int(os.environ.get(WORKERS) or 0))


def _placed() -> Counter[str]:
    """Describe leases placed on each worker, read without the lease lock (``_reserve``'s slot
    check under it is the last word)."""

    count: Counter[str] = Counter()
    for path in work_dir("leases").glob("*.json"):
        try:
            lease = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if lease.get("kind") == "describe" and lease.get("placement"):
            count[lease["placement"]] += 1
    return count


def record_room(free: Sequence[str], placements: Sequence[str | None], *, held: bool
                ) -> dict[str, int]:
    """Write and return this pass's describe room per worker (module doc). ``free`` is the pass's
    free worker slots (``mini_placement.slots``), ``placements`` where its 2b lanes go."""

    spare = Counter(free) - Counter(p for p in placements if p)  # keeps positive counts only
    room: dict[str, int] = {}
    if spare and not held:
        live = _placed()
        room = {h: spare[h] + live[h] for h in sorted(set(spare) | set(live))}
    path = work_dir("tick") / ROOM
    path.parent.mkdir(parents=True, exist_ok=True)
    admission._write_atomic(path, json.dumps({"at": now(), "room": room}) + "\n")
    return room


def room() -> dict[str, int]:
    """The last pass's describe room per worker; none when missing, unreadable or stale."""

    try:
        saved = json.loads((work_dir("tick") / ROOM).read_text(encoding="utf-8"))
        at = datetime.fromisoformat(str(saved["at"]).replace("Z", "+00:00"))
        given = {str(h): int(n) for h, n in saved["room"].items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}
    at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - at).total_seconds() > ROOM_FRESH_S:
        return {}
    return given


def hosts(cap: int) -> list[str]:
    """Workers a describe seat may start on now, most room left first: fewer than ``cap``
    describe seats on workers in all and fewer than the worker's room on it."""

    live = _placed()
    if sum(live.values()) >= cap:
        return []
    left = {h: n - live[h] for h, n in room().items() if n - live[h] > 0}
    return sorted(left, key=lambda h: (-left[h], h))


def landed(lane: Mapping[str, object]) -> bool:
    """Whether an ended describe lane's outputs are on the hub (``describe_write._batch_rows``):
    a hub lane's always; a worker lane's once the relay ingested it or its lease is gone
    (expired), so a lost worker lane is settled like a failed hub lane."""

    label = str(lane.get("label") or "")
    if not lane.get("placement"):
        return True
    return mini_relay.ingested(label) or not (work_dir("leases") / f"{label}.json").is_file()


def launch(brief: Path, families: Sequence[str], *, role: str) -> tuple[str, str, str] | None:
    """Start a describe lane on the first worker with room that admits it: its run, worker and
    lease label. None when no worker has room or each refused (the caller falls back to the
    hub). A start whose outcome is unproven keeps its lease and raises ``Refused``
    (``worker_uncertain``), so the batch waits instead of starting twice."""

    from menagerie.pipeline import tick

    if seats() <= 0:
        return None
    cfg = tick.config()
    for host in hosts(seats()):
        try:
            started = mini_placement.start_describe(
                families, brief, role=role, host=host, cfg=cfg,
                config=(cfg.get("mini_workers") or {}).get(host) or {})
            return mini_placement.finish(started), host, started.lease["label"]
        except admission.Refused:
            continue
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise admission.Refused("worker_uncertain", f"{host}: {exc}"[:300]) from exc
    return None
