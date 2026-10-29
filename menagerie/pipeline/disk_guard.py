"""The worker-disk guard the tick reads once per pass (lane r3-tick; STORE-GATE section 4).

The trace store lives on the worker's root disk, which also holds every other job's scratch. The
guard keeps rung 3 from filling it: ``decide`` is a pure state machine over one free-space reading
(``ok``, ``paused``, ``critical`` or ``unknown``), and the tick skips the steps a state holds.

- The floor is the larger of the config floor (45.7 GiB, the fleet's worker-disk floor) and 10
  percent of the disk. The config's critical, pause and resume lines (50, 62 and 72 GiB) move up
  with it, so their gaps stay as written.
- Below the pause line, top-up, stage-3 queueing, new runner jobs and new AXON and GPU batches
  hold; running rows finish. Below the critical line the staging drain holds too, and Crawler gets
  one escalation a day. Leaving a held state needs the resume line (10 GiB of hysteresis).
- A failed reading keeps the last state for 30 minutes, then becomes ``unknown``, which holds
  like a pause.
- Crawler gets one FACT on each entry to paused, critical or unknown; a restart reads the saved
  state, so it never repeats one. Resuming is a log line only.

The worker side (``claim_ok``) idles ``stage3 serve`` and ``runner serve`` below the pause line
the runner job was started with (``MENAGERIE_DISK_PAUSE_GIB``); a job without it never idles.
Inside a running attempt, ``attempt_watch`` kills it under ``free_and_floor``'s floor plus a
margin, or when the attempt's own output passes its cap.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

GIB = 1024 ** 3
UNKNOWN_AFTER = timedelta(minutes=30)
HELD = {  # what each state holds; everything else (collect, commits, status, ...) runs
    "ok": frozenset(),
    "paused": frozenset({"top_up", "stage3_queue", "runner_submit", "cluster"}),
    "unknown": frozenset({"top_up", "stage3_queue", "runner_submit", "cluster"}),
    "critical": frozenset({"top_up", "stage3_queue", "runner_submit", "cluster", "drain"}),
}
PAUSE_ENV = "MENAGERIE_DISK_PAUSE_GIB"


class GuardConfigError(ValueError):
    """The thresholds are out of order; the tick refuses to start."""


@dataclass(frozen=True)
class Config:
    """The guard's lines in GiB (STORE-GATE section 4, accepted by Crawler at the check-in)."""

    host: str = field(default_factory=lambda: os.environ.get("MENAGERIE_RUNNER_HOST", ""))
    path: str = "data/menagerie/store"  # the store root, relative to the worker home
    floor_gib: float = 45.7
    floor_fraction: float = 0.10
    critical_gib: float = 50.0
    pause_gib: float = 62.0
    resume_gib: float = 72.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> Config:
        known = {k: v for k, v in (values or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known).checked()

    def checked(self) -> Config:
        if not self.floor_gib < self.critical_gib < self.pause_gib < self.resume_gib:
            raise GuardConfigError(
                "disk guard lines must rise floor < critical < pause < resume, got "
                f"{self.floor_gib} < {self.critical_gib} < {self.pause_gib} < {self.resume_gib}")
        return self

    def lines(self, total_gib: float | None) -> dict[str, float]:
        """The effective floor and lines: all shifted up when 10 percent of the disk is larger."""

        floor = max(self.floor_gib, self.floor_fraction * total_gib if total_gib else 0.0)
        up = floor - self.floor_gib
        return {"floor_gib": round(floor, 1), "critical_below_gib": round(self.critical_gib + up, 1),
                "pause_below_gib": round(self.pause_gib + up, 1),
                "resume_at_gib": round(self.resume_gib + up, 1)}


@dataclass(frozen=True)
class Reading:
    free_bytes: int
    total_bytes: int


@dataclass(frozen=True)
class State:
    """The guard's saved state (the tick's state file and ``status.md``)."""

    state: str = "ok"
    free_gib: float | None = None
    total_gib: float | None = None
    floor_gib: float | None = None
    pause_below_gib: float | None = None
    resume_at_gib: float | None = None
    since: str | None = None
    read_at: str | None = None
    fact_sent_since: str | None = None
    escalated_on: str | None = None  # the date of the last critical escalation
    reader_failures: int = 0

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> State:
        return cls(**{k: v for k, v in (values or {}).items() if k in cls.__dataclass_fields__})

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def held(self, step: str) -> bool:
        return step in HELD[self.state]


def _level(free_gib: float, prev: str, lines: dict[str, float]) -> str:
    if free_gib < lines["critical_below_gib"]:
        return "critical"
    if free_gib < lines["pause_below_gib"]:
        return "paused"
    if prev != "ok" and free_gib < lines["resume_at_gib"]:
        return "paused"  # hysteresis: a held state leaves only at the resume line
    return "ok"


def _message(state: State, cfg: Config) -> str:
    where = "worker"
    if state.state == "unknown":
        return (f"{where} disk unknown: no free-space reading for over 30 minutes "
                f"({state.reader_failures} failed reads); top-up, queueing, new runner and AXON jobs "
                "held as for a pause; running rows finish")
    tail = (f"free {state.free_gib} GiB < {state.pause_below_gib:g} (floor {state.floor_gib:g}); "
            "top-up, queueing, new runner and AXON jobs held; running rows finish; resumes at "
            f"{state.resume_at_gib:g} GiB")
    if state.state == "critical":
        return (f"{where} disk critical: {tail}; the staging drain is held too. Recommend stopping "
                "runner claiming (touch the spool's STOP) until space is freed")
    return f"{where} disk paused: {tail}"


def _failed(prev: State, now: datetime) -> State:
    """A failed reading: the last state for 30 minutes after the last good one, then unknown."""

    new = replace(prev, reader_failures=prev.reader_failures + 1)
    last = datetime.fromisoformat(prev.read_at) if prev.read_at else None
    if (last is not None and now - last <= UNKNOWN_AFTER) or prev.state == "unknown":
        return new
    return replace(new, state="unknown", since=now.isoformat(timespec="seconds"))


def _read(prev: State, reading: Reading, now: datetime, cfg: Config) -> State:
    total = reading.total_bytes / GIB
    free = round(reading.free_bytes / GIB, 1)
    lines = cfg.lines(total)
    level = _level(free, prev.state, lines)
    new = replace(prev, free_gib=free, total_gib=round(total, 1),
                  read_at=now.isoformat(timespec="seconds"), reader_failures=0,
                  floor_gib=lines["floor_gib"], pause_below_gib=lines["pause_below_gib"],
                  resume_at_gib=lines["resume_at_gib"])
    return new if level == prev.state else replace(new, state=level, since=new.read_at)


def _facts(prev: State, new: State, today: str, cfg: Config) -> tuple[State, list[str]]:
    """One FACT per entry to a held state (none when easing from critical to paused), and one
    escalation a day while critical."""

    if new.state == "ok":
        return replace(new, fact_sent_since=None), []
    if prev.state == "critical" and new.state == "paused" and new.since != prev.since:
        return replace(new, fact_sent_since=new.since), []
    entered = new.fact_sent_since != new.since
    if not entered and (new.state != "critical" or new.escalated_on == today):
        return new, []
    new = replace(new, fact_sent_since=new.since)
    if new.state == "critical":
        new = replace(new, escalated_on=today)
    return new, [_message(new, cfg)]


def decide(prev: State, reading: Reading | None, *, now: datetime,
           cfg: Config) -> tuple[State, list[str]]:
    """The next guard state and the FACTs to send, from the last state and one reading.

    ``reading`` is ``None`` when the reader failed. Pure: the caller saves the state and sends
    the messages. Returns: the new state and zero or more message texts for Crawler.
    """

    new = _failed(prev, now) if reading is None else _read(prev, reading, now, cfg)
    return _facts(prev, new, now.date().isoformat(), cfg)


def log_line(state: State, cfg: Config) -> str:
    del cfg
    where = "worker"
    free = "?" if state.free_gib is None else f"{state.free_gib}GiB"
    return (f"disk {where} free={free} floor={state.floor_gib} pause<{state.pause_below_gib:g} "
            f"resume>={state.resume_at_gib:g} state={state.state} since={state.since}"
            if state.pause_below_gib is not None else
            f"disk {where} free={free} state={state.state} since={state.since}")


def ssh_reader(cfg: Config, *, timeout: float = 20.0) -> Callable[[], Reading]:
    """A reader that asks the worker once: ``stat -f`` gives df's Avail (``%a*%S``) and size."""

    def read() -> Reading:
        if not cfg.host:
            raise OSError("MENAGERIE_RUNNER_HOST is not set; it names the worker's ssh alias")
        argv = ["ssh", cfg.host, f"stat -f -c '%a %S %b' {shlex.quote(cfg.path)}"]
        done = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=timeout)
        if done.returncode:
            raise OSError(f"disk read on {cfg.host} failed: {done.stderr.strip()[-200:]}")
        avail, size, blocks = (int(x) for x in done.stdout.split())
        return Reading(free_bytes=avail * size, total_bytes=blocks * size)

    return read


def read_once(reader: Callable[[], Reading]) -> tuple[Reading | None, str | None]:
    """One reading, or ``None`` and the error text; a reader never stops the tick."""

    try:
        return reader(), None
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"[:300]


def free_and_floor(path: Path | str, *,
                   statvfs: Callable[[str], Any] = os.statvfs) -> tuple[float, float]:
    """Free GiB on ``path``'s filesystem and the fleet floor there (the config floor or its
    fraction of the disk, whichever is larger)."""

    st, config = statvfs(str(path)), Config(host="")
    total = st.f_blocks * st.f_frsize / GIB
    return st.f_bavail * st.f_frsize / GIB, max(config.floor_gib, config.floor_fraction * total)


def claim_ok(path: Path | str, *, statvfs: Callable[[str], Any] = os.statvfs) -> bool:
    """Worker side: may ``stage3 serve`` or ``runner serve`` claim? False below
    ``MENAGERIE_DISK_PAUSE_GIB``.

    The runner job sets that variable from the tick's effective pause line; without it the
    worker never idles (jobs started before the guard).
    """

    pause = os.environ.get(PAUSE_ENV)
    if not pause:
        return True
    st = statvfs(str(path))
    return st.f_bavail * st.f_frsize / GIB >= float(pause)
