"""Start the store trim when the worker store is under pressure (the tick's ``store_pressure`` step).

When the worker's local store holds more than its target (``store_ceiling.TARGET_GIB``) or its
free space (the ``disk`` step's reading) is under the guard's resume line, at most once every
``every_hours`` hours, the tick starts the store-box daily unit: its worker step boxes the blobs
no sealed batch holds, it waits for the collector to verify them, then runs the store trim with
its ceiling step (``store_box.daily``). A big day trims itself whenever it happens, not only at
the daily timer's hour. The unit is the one path that trims, so the timer and this trigger never
run two trims at once (systemd runs one instance; the daily run holds its own lock).

It starts only when the trim can free at least ``min_free_gib``: the ceiling's own candidates
(trace saves and extracted PDFs past its minimum age, outside the working set, not yet sent to
the box; boxed already or boxed by the same run), capped at what the store holds over its
target (``freeable``). Otherwise a heavy day, whose young blobs the ceiling keeps, would start a
box batch and its verification restore every few hours and free nothing. The estimate costs an
index read and a worker listing, so it is taken at most once every ``check_minutes``.

Under the resume line the target dips by the shortfall (``dip_target``; decisions.md 2026-11-03):
with free 8 GiB short of the resume line, a 20 GiB target becomes 12, never under ``floor_gib``.
The step hands the dipped target to the trim through ``store_box.request_ceiling``; the ceiling
still offloads only blobs a full-restore-verified box batch holds.
"""

from __future__ import annotations

import gzip
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any

from menagerie.pipeline import DataLayout, store_ceiling
from menagerie.pipeline.store_ceiling import MIN_AGE_DAYS, TARGET_GIB

GIB = 2**30
DEFAULTS = {"every_hours": 4.0, "target_gib": TARGET_GIB, "min_free_gib": 1.0, "check_minutes": 60,
            "floor_gib": 12.0}
UNIT = "menagerie-store-box-daily.service"
# The ceiling's working set, as ``store_trim_job.sh`` passes it (under the worker's $HOME).
WORKING_SET = ("scratch/menagerie/spool-*/new/*.json", "scratch/menagerie/spool-*/claimed/*.json",
               "scratch/menagerie/spool-*/stage3/new/*.json",
               "scratch/menagerie/spool-*/stage3/claimed/*.json",
               "scratch/menagerie/axon/*/attempts/*/result.json")
SHA = re.compile(r"[0-9a-f]{64}")


def ssh_size(host: str, path: str, *, timeout: float = 20.0) -> Callable[[], float]:
    """A reader of the worker store's allocated size in GiB (``du``, under a second at 90k blobs)."""

    def read() -> float:
        if not host:
            raise OSError("MENAGERIE_RUNNER_HOST is not set; it names the worker's ssh alias")
        done = subprocess.run(["ssh", host, f"du -sB1 {shlex.quote(path)}"], capture_output=True,
                              text=True, check=False, timeout=timeout)
        if done.returncode:
            raise OSError(f"store size read on {host} failed: {done.stderr.strip()[-200:]}")
        return int(done.stdout.split()[0]) / GIB

    return read


def worker_listing(host: str, path: str, *, timeout: float = 60.0) -> tuple[dict[str, int], set[str]]:
    """Allocated bytes of every visible blob in the worker store, by sha, and the shas the
    working set names; one ssh call (a gzipped ``find``, about a second)."""

    if not host:
        raise OSError("MENAGERIE_RUNNER_HOST is not set; it names the worker's ssh alias")
    globs = " ".join(WORKING_SET)
    command = (f"cd {shlex.quote(path)}/sha256 && find . -type f -printf '%P %b\\n'; "
               f"cd && cat {globs} 2>/dev/null | grep -oE '[0-9a-f]{{64}}' | sed 's/^/keep /'")
    done = subprocess.run(["ssh", host, f"({command}) | gzip -1"], capture_output=True, check=False,
                          timeout=timeout)
    if done.returncode:
        raise OSError(f"store listing on {host} failed: {done.stderr.decode(errors='replace')[-200:]}")
    return parse_listing(gzip.decompress(done.stdout).decode())


def parse_listing(text: str) -> tuple[dict[str, int], set[str]]:
    local: dict[str, int] = {}
    keep: set[str] = set()
    for line in text.splitlines():
        head, _, tail = line.partition(" ")
        if head == "keep":
            keep.add(tail)
            continue
        name = head.split("/")[1] if head.count("/") >= 1 else ""
        sha = None if name.startswith(".") else store_ceiling._sha_of(name)
        if sha and tail.isdigit():
            local[sha] = local.get(sha, 0) + int(tail) * 512
    return local, keep


def aged_gib(local: Mapping[str, int], keep: set[str], index: store_ceiling.Index, *, now: float,
             min_age: float = MIN_AGE_DAYS * 86400) -> float:
    """GiB the ceiling may offload: its kinds, past ``min_age`` since placement, outside the
    working set, never sent to the box (a fetched-back blob is left out: it is rare)."""

    total = 0
    for sha, disk in local.items():
        if sha in keep or sha in index.offloaded or sha not in index.first:
            continue
        if store_ceiling._kind(sha, index) and now - store_ceiling._epoch(index.first[sha].fetched) >= min_age:
            total += disk
    return total / GIB


def freeable(host: str, path: str, data: DataLayout) -> Callable[[], float]:
    """A reader of the GiB the ceiling could offload now (before the cap at the target)."""

    def read() -> float:
        local, keep = worker_listing(host, path)
        return aged_gib(local, keep, store_ceiling.load_index(data), now=time.time())

    return read


def start_unit(unit: str) -> None:
    """Start a user unit without waiting for it."""

    subprocess.run(["systemctl", "--user", "start", "--no-block", unit], capture_output=True,
                   text=True, check=True, timeout=30)


def _reasons(store_gib: float | None, free_gib: float | None, resume_gib: float | None,
             target: float) -> list[str]:
    why = []
    if store_gib is not None and store_gib > target:
        why.append(f"store {store_gib} GiB over its {target:g} GiB target")
    if free_gib is not None and resume_gib is not None and free_gib < resume_gib:
        why.append(f"free {free_gib} GiB under the resume line {resume_gib:g}")
    return why


def dip_target(target: float, free_gib: float | None, resume_gib: float | None,
               floor: float) -> float:
    """The store target for this trim: ``target`` less what free disk is short of the resume
    line, never under ``floor`` (and never above ``target``)."""

    if free_gib is None or resume_gib is None or free_gib >= resume_gib:
        return target
    return min(target, max(floor, target - (resume_gib - free_gib)))


def _can_free(last: Mapping[str, Any], store_gib: float, target: float, *, now: datetime,
              conf: Mapping[str, Any], read_aged: Callable[[], float]) -> dict[str, Any]:
    """What the trim could free (``aged`` capped at the store's excess over its target), reusing
    an estimate taken in the last ``check_minutes``."""

    checked = last.get("checked") or {}
    fresh = checked.get("at") and now - datetime.fromisoformat(checked["at"]) < timedelta(
        minutes=float(conf["check_minutes"]))
    aged = checked["aged_gib"] if fresh else round(read_aged(), 1)
    return {"at": checked["at"] if fresh else now.isoformat(timespec="seconds"), "aged_gib": aged,
            "can_gib": round(max(0.0, min(aged, store_gib - target)), 1)}


def step(state: Mapping[str, Any] | None, free_gib: float | None, resume_gib: float | None, *,
         now: datetime, cfg: Mapping[str, Any] | None, read_store: Callable[[], float],
         read_aged: Callable[[], float], active: Callable[[str], bool],
         start: Callable[[str], None],
         request: Callable[[float], None] = lambda target_gib: None,
         ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Decide and, when due, start the store-box daily unit; ``request`` hands the trim a target
    dipped under the configured one. Returns (new state, tick log result)."""

    conf = {**DEFAULTS, **(cfg or {})}
    last = dict(state or {})
    began = datetime.fromisoformat(last["at"]) if last.get("at") else None
    if began is not None and now - began < timedelta(hours=float(conf["every_hours"])):
        return last, {"skipped": f"last trim start at {last['at']}"}
    if active(UNIT):
        return last, {"skipped": f"{UNIT} is running"}
    try:
        store_gib: float | None = round(read_store(), 1)
        error = None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        store_gib, error = None, str(exc)[:200]
    target = float(conf["target_gib"])
    why = _reasons(store_gib, free_gib, resume_gib, target)
    if not why or store_gib is None:
        return last, {"skipped": f"store {store_gib} GiB, free {free_gib} GiB",
                      **({"read_error": error} if error else {})}
    trim_to = round(dip_target(target, free_gib, resume_gib, float(conf["floor_gib"])), 1)
    try:
        checked = _can_free(last, store_gib, trim_to, now=now, conf=conf, read_aged=read_aged)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        return last, {"skipped": f"no estimate of what the trim could free: {str(exc)[:200]}"}
    last = {**last, "checked": checked}
    if checked["can_gib"] < float(conf["min_free_gib"]):
        return last, {"skipped": f"{'; '.join(why)}, but the trim could free only "
                                 f"{checked['can_gib']} GiB to a {trim_to:g} GiB target (aged "
                                 f"{checked['aged_gib']} GiB, estimated {checked['at']})"}
    if trim_to < target:
        request(trim_to)
    start(UNIT)
    new = {"at": now.isoformat(timespec="seconds"), "why": why, "store_gib": store_gib,
           "free_gib": free_gib, "target_gib": trim_to, "checked": checked}
    return new, {"started": new}
