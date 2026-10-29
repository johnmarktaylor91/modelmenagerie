"""``runner ensure``: keep exactly one live ``stage3 serve`` job per spool (lane r3-tick).

A runner is a ``run-on --restartable`` job on the worker running ``runner_job.sh`` (a ``runner
serve`` and ``stage3 serve`` pair on one spool). Each pass, for every configured runner:

- RUNNING with a fresh spool heartbeat: nothing to do, unless a runner with a size ladder has
  wanted another size two passes running and its spool is idle (nothing claimed): then it is
  cancelled (``resize``) and the next pass starts it at the wanted size. A busy runner is
  drained once a smaller size has been wanted ``RESIZE_AFTER`` passes, or a larger one
  ``GROW_DRAIN_AFTER`` passes: ``<spool>/stage3/DRAIN`` stops its ``stage3 serve`` claiming new
  jobs (its runner keeps serving the claimed ones), so it idles within one attempt; the drain
  ends when the size is no longer wanted or the successor job starts (the job script deletes
  the file). A refilled stage 3 queue otherwise never lets B idle, so G never started
  (2026-10-21) and B stayed at its night size all day (2026-10-22). The night runner G is
  drained the same way before its morning stop.
- RUNNING with the ``stage3 serve`` heartbeat stale over 15 minutes (and the job past its 30-minute
  start-up, which builds environments): cancelled; a later pass starts its successor.
- DONE (it saw the spool's ``STOP``), or any dead job while ``STOP`` exists: left stopped.
- FAILED, INTERRUPTED or MISSING: a new job from the template, unless another live job already
  serves the spool (adopted), the heartbeat is still fresh (someone serves it: never two
  ``stage3 serve`` on one spool, since a starting server re-queues every claimed job), the disk
  guard holds new runner jobs, or three restarts already happened today (then one escalation a
  day; a resize is not a restart).
- REFUSED (run-on's admission turned the job down for capacity; it never ran): its submission is
  not a restart, and the next pass tries again at once, each size behind ``run-on probe`` so a
  refused try starts nothing (``held-capacity``; Crawler, 28 Oct: a backoff lost every race to
  ad hoc jobs). The tick also runs this step in its refill passes while a runner is refused
  (``refused``). A full worker never uses up the day's restarts or escalates.
- UNKNOWN or HOST_UNREACHABLE: nothing (never act on an unverified state).

Size ladder (runner A, JMT 2026-10-08 5:06 PM PT: it grows to match 2b tries inside the worker
share): ``sizes`` lists (workers, CPUs, GiB) smallest first and ``demand`` names the lane kind it
serves; the wanted size has one worker per ``lanes_per_worker`` live lanes. A start tries the
wanted size, then each smaller one, so run-on's admission keeps it inside the share.

The job is found by its ``MENAGERIE_TICK_RUNNER`` marker, or for runner B by the catalog pass's
own job script (``SPOOL=$HOME/scratch/menagerie/spool-B``), so the live catalog runner is adopted,
never replaced. Runners come from ``<work root>/tick/runners.json`` when it exists, else
``DEFAULT_RUNNERS``; their recorded jobs and restarts live in the tick state.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from menagerie.paths import work_dir

STALE_S = 15 * 60  # a RUNNING job whose stage3 heartbeat is older than this is cancelled
SERVING_S = 120  # a heartbeat younger than this means some server is live on the spool
STARTUP = timedelta(minutes=30)  # env build and sync before the first heartbeat
RESTARTS_A_DAY = 3  # starts run-on admitted that later died; capacity refusals never count
CAPACITY = ("NO_CAPACITY", "PROBE_TIMEOUT")  # run-on's reasons for deferring a job it never ran
# ``run-on probe`` exits: would be admitted now, refused (with the submit's reason), could not tell
PROBE_ADMIT, PROBE_REFUSED, PROBE_UNKNOWN = 0, 6, 7
RESIZE_AFTER = 2  # passes in a row wanting another size
# passes in a row wanting a larger size before a busy runner is drained for it: B's growth want
# at night follows G on and off (streaks of 2 to 4 passes, 2026-10-22), its day want holds for
# hours, so a growth drain waits an hour and a flickering want never restarts B twice
GROW_DRAIN_AFTER = 6
DRAIN = "stage3/DRAIN"  # under a spool: its stage3 serve claims no new job while this exists
LIVE = "RUNNING"
# REFUSED: never admitted for capacity, never ran (not a restart); REJECTED: refused for another
# reason (a policy or placement refusal): it counts as a restart, so a stuck one escalates
DEAD = ("FAILED", "INTERRUPTED", "MISSING", "REFUSED", "REJECTED")
DISCOVER_DAYS = 14
# Readings that never change for a job id: ``RunOn(ended=...)`` remembers them across passes, so
# each pass asks run-on only about jobs that may still move (discovery lists 14 days of jobs).
ENDED = ("DONE", "FAILED", "REFUSED", "REJECTED")
ENDED_JOBS = "ended-jobs.json"  # under the tick's work folder
TEMPLATE = "menagerie/pipeline/runner_job.sh"
NIGHT_ZONE = ZoneInfo("America/Los_Angeles")
BIG_LEAD_IN = timedelta(minutes=30)  # trace-host lanes stop starting this long before the night
SHARE_CPUS, SHARE_GIB = 18, 112
# B's (workers, CPUs, GiB) under the big-runner policy: its night size beside G, then its day size,
# which is also its declaration in ``DEFAULT_RUNNERS``. Eight workers share the 6 CPUs and 36 GiB two
# had: a stage-3 request's default cap stays the declaration over ``runner.DEFAULT_WORKERS``
# (12 GiB) and ``runner serve`` starts a request only while the running caps fit the declaration,
# so more workers pledge no more memory; they fill the room the 4 GiB per-row trace floor leaves
# (``flow.ROW_FLOOR_GIB``; Crawler, 28 Oct: B used under one of its 6 CPUs with four workers).
# Sizes are told apart by CPUs and memory alone (``_observed``).
# The third is B's day boost (Lead, 28 Oct: stage 3 is the limit), off unless the tick config's
# ``b_boost`` is true: on its first day the default worker refused 12 lane starts for RAM beside it. With
# it on, it is wanted only while it fits the share beside A's wanted size (``_b_day_workers``)
# and run-on would admit ``BOOST_MARGIN_GIB`` more (``_boost_room``); reached only when ``run-on
# probe`` admits it (``_resize``, ``_submit``); never counted in ``lane_room``, so lanes keep their
# room and B stays at its day size when the probe refuses. Its workers stay above the day size's, so
# ``Runner.wanted`` never picks it for the day size's count.
B_SIZES = ((1, 3, 16), (8, 6, 36), (10, 9, 46))
B_DAY = 1  # B's day size in B_SIZES, its declaration in DEFAULT_RUNNERS
# B holds its boost only while run-on would admit this much more beside it, so new lanes and the
# day's other jobs keep room (a 1 GiB 2b lane was refused 7 minutes after B first grew, 28 Oct)
BOOST_MARGIN_GIB = 10


class SubmitRefused(OSError):
    """run-on refused the job outright (no job id was issued): a smaller size may be tried."""

    @property
    def capacity(self) -> bool:
        """Refused for lack of room (nothing ran), not for a policy reason."""

        return any(code in str(self) for code in CAPACITY)


class NoCapacity(OSError):
    """run-on refused every size for capacity: nothing ran, so it is no restart; back off."""


@dataclass(frozen=True)
class Runner:
    """One spool's runner job: its spool (relative to the worker home) and its declaration."""

    name: str
    spool: str
    cpus: int
    memory_gib: int
    workers: int
    work_root: str = "scratch/menagerie/r2-run/work"
    legacy: str | None = None  # a pre-tick job script's spool line, for adoption
    sizes: tuple[tuple[int, int, int], ...] = ()  # (workers, cpus, GiB), smallest first
    demand: str | None = None  # the lane kind whose live count sizes it
    lanes_per_worker: int = 5
    modes: tuple[str, ...] = ("collect", "full")  # the tick modes it runs in

    def sized(self, index: int | None) -> Runner:
        if index is None or not self.sizes:
            return self
        workers, cpus, gib = self.sizes[index]
        return replace(self, workers=workers, cpus=cpus, memory_gib=gib)

    def wanted(self, demand: int) -> int | None:
        """The size index for ``demand`` live lanes (one worker per ``lanes_per_worker``)."""

        if not self.sizes:
            return None
        workers = max(1, -(-demand // self.lanes_per_worker))
        fits = [i for i, size in enumerate(self.sizes) if size[0] <= workers]
        return fits[-1] if fits else 0


# Menagerie's worker share is 18 CPUs and 112 GiB at any hour (JMT 2026-10-08 5:31 PM PT,
# share-grants.json): B keeps 6 and 36; A (2b tries through the bridge, rung 3 stage 3; full mode
# only) grows from 3 CPUs to the remaining 12 and 76 with live 2b lanes. run-on admission is the
# final word on what fits. The opt-in night runner G is 6 CPUs and 40 GiB, so the night's steady
# share is 12 CPUs and 76 GiB (A 3/20, B 3/16, G 6/40): run-on pledges 1.25 times a job's declared
# memory and on the science-compute worker could not admit G at 76 or 52 GiB beside A and B
# (2026-10-12). flow.BIG_GIB follows G's memory.
DEFAULT_RUNNERS = (
    Runner("B", "scratch/menagerie/spool-B", cpus=6, memory_gib=36, workers=8,
           legacy="SPOOL=$HOME/scratch/menagerie/spool-B"),
    Runner("A", "scratch/menagerie/spool-A", cpus=3, memory_gib=20, workers=1,
           work_root="scratch/menagerie/runner-A/work",
           sizes=((1, 3, 20), (2, 6, 38), (3, 9, 56), (4, 12, 76)), demand="2b",
           modes=("full",)),
    Runner("G", "scratch/menagerie/spool-G", cpus=6, memory_gib=40, workers=1,
           work_root="scratch/menagerie/runner-G/work", modes=("full",)),
)


@dataclass(frozen=True)
class Record:
    """What the tick remembers about one runner between passes."""

    job: str | None = None
    submitted_at: str | None = None
    restarts: tuple[str, ...] = ()  # submit times after a failure, ISO; refunded if REFUSED
    cancelled: str | None = None  # the job this tick cancelled, until its successor starts
    cancelled_for: str | None = None  # "stale" or "resize"
    escalated_on: str | None = None
    size: int | None = None  # the size index the job was started at
    resize_streak: int = 0
    refused_at: str | None = None  # the last capacity refusal, ISO: starts back off from it
    draining: bool = False  # the tick wrote the spool's DRAIN; cleared when it removes it
    resize_to: int | None = None  # the size a resize cancel was for, until its successor starts
    hold_growth_until: str | None = None  # ISO, set by hand: no growth resize before it (gotchas.md)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> Record:
        values = dict(values or {})
        values["restarts"] = tuple(values.get("restarts", ()))
        return cls(**{k: v for k, v in values.items() if k in cls.__dataclass_fields__})

    def to_mapping(self) -> dict[str, Any]:
        return {**asdict(self), "restarts": list(self.restarts)}


@dataclass
class Outcome:
    """One runner's pass: the action taken (or planned, in a dry run) and the new record."""

    runner: str
    action: str
    detail: str
    record: Record
    job_state: str | None = None
    heartbeat_age_s: float | None = None
    messages: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"runner": self.runner, "action": self.action, "detail": self.detail,
                "job": self.record.job, "size": self.record.size, "job_state": self.job_state,
                "heartbeat_age_s": self.heartbeat_age_s}


class Ops(Protocol):
    """The worker and run-on calls ``ensure`` makes; tests pass fakes."""

    def job_state(self, job: str) -> str: ...
    def heartbeat_age(self, runner: Runner) -> float | None: ...
    def stopped(self, runner: Runner) -> bool: ...
    def idle(self, runner: Runner) -> bool: ...
    def drain(self, runner: Runner, on: bool) -> None: ...
    def discover(self, runner: Runner) -> list[tuple[str, str | None]]: ...
    def submit(self, runner: Runner, env: Mapping[str, str]) -> str: ...
    def cancel(self, job: str) -> None: ...
    def allocation(self, job: str) -> tuple[int, int] | None: ...
    def probe(self, runner: Runner, replacing: str | None) -> tuple[int, str]: ...


@dataclass(frozen=True)
class Pass:
    """One runner's inputs for a pass."""

    runner: Runner
    ops: Ops
    now: datetime
    hold_submit: bool
    pause_gib: float | None
    demand: int
    dry_run: bool


def runners(path: Path | None = None) -> tuple[Runner, ...]:
    """The configured runners: ``runners.json`` (a list of Runner fields) or the default."""

    path = path or work_dir("tick") / "runners.json"
    if not path.is_file():
        return DEFAULT_RUNNERS
    items = json.loads(path.read_text(encoding="utf-8"))
    return tuple(Runner(**{**item, "sizes": tuple(tuple(s) for s in item.get("sizes", ())),
                           "modes": tuple(item.get("modes", ("collect", "full")))})
                 for item in items)


def _today(now: datetime) -> str:
    return now.date().isoformat()


def _restarts_today(record: Record, now: datetime) -> int:
    return sum(1 for at in record.restarts if at[:10] == _today(now))


def _young(record: Record, now: datetime) -> bool:
    if not record.submitted_at:
        return False
    return now - datetime.fromisoformat(record.submitted_at) < STARTUP


def _adopt(p: Pass, record: Record, out: Outcome) -> tuple[Record | None, str]:
    """Another job serving the spool: its record when exactly one is live, else why not.

    Returns ``(record, "adopted")`` (with the job's own start time, so its start-up is not cut
    short), ``(None, "conflict")`` for several live jobs (reported, never fixed),
    ``(None, "unverified")`` when a candidate's state cannot be read or it is still pending (a
    new job then waits: it might be the one serving), or ``(None, "")``.
    """

    found = [(j, at) for j, at in p.ops.discover(p.runner) if j != record.job]
    states = {j: p.ops.job_state(j) for j, _ in found}
    live = [(j, at) for j, at in found if states[j] == LIVE]
    if len(live) > 1:
        jobs = ", ".join(j for j, _ in live)
        out.messages.append(f"runner {p.runner.name}: {len(live)} live jobs serve {p.runner.spool} "
                            f"({jobs}); the tick changes nothing until one remains")
        return None, "conflict"
    if live:
        job, at = live[0]
        return Record(job=job, submitted_at=at, restarts=record.restarts,
                      escalated_on=record.escalated_on,
                      hold_growth_until=record.hold_growth_until), "adopted"
    unsure = sorted(j for j, state in states.items() if state not in (*DEAD, "DONE"))
    if unsure:
        out.detail = f"cannot verify {', '.join(unsure)} ({', '.join(states[j] for j in unsure)})"
        return None, "unverified"
    return None, ""


def _cancel(p: Pass, out: Outcome, why: str, detail: str) -> Outcome:
    out.action, out.detail = f"cancel-{why}" if why == "stale" else why, detail
    if not p.dry_run:
        p.ops.cancel(out.record.job)
        out.record = replace(out.record, cancelled=out.record.job, cancelled_for=why,
                             resize_streak=0)
    return out


def _set_drain(p: Pass, record: Record, on: bool) -> Record:
    """Write or remove the spool's DRAIN when the record says otherwise (never in a dry run)."""

    if record.draining == on or p.dry_run:
        return record
    p.ops.drain(p.runner, on)
    return replace(record, draining=on)


def _live(p: Pass, out: Outcome, age: float | None) -> Outcome:
    """A RUNNING job: cancel it when its heartbeat is stale or (idle) it should change size."""

    record = out.record
    if (age is None or age > STALE_S) and not _young(record, p.now):
        stale = "missing" if age is None else f"{age:.0f} s old"
        return _cancel(p, out, "stale", f"{record.job} RUNNING, stage3 heartbeat {stale}")
    want = p.runner.wanted(p.demand)
    if want is None or record.size is None or want == record.size:
        out.record = _set_drain(p, replace(record, resize_streak=0), False)
        return out
    return _resize(p, out, want)


def _resize(p: Pass, out: Outcome, want: int) -> Outcome:
    """Another size is wanted: cancel once it has been ``RESIZE_AFTER`` passes and the spool is
    idle; a busy runner is drained so it idles within one attempt (growth waits longer). Growth
    waits while the record's ``hold_growth_until`` is in the future (a shrink never does). Before
    draining or cancelling, ``run-on probe --replacing`` the job must admit the new size; any
    other answer keeps the job (and its claims) and asks again next pass."""

    record = out.record
    grow = want > record.size
    if grow and record.hold_growth_until and p.now < datetime.fromisoformat(record.hold_growth_until):
        out.record = _set_drain(p, replace(record, resize_streak=0), False)
        out.detail = f"wants size {want} (now {record.size}); growth held until {record.hold_growth_until}"
        return out
    streak = record.resize_streak + 1
    out.record = replace(record, resize_streak=streak)
    if streak < RESIZE_AFTER or (grow and p.hold_submit):
        out.record = _set_drain(p, out.record, False)
        out.detail = f"wants size {want} (now {record.size}), {streak} passes"
        return out
    # Ask run-on before draining or cancelling: a successor it would refuse leaves the spool unserved.
    code, why = _probe(p, p.runner.sized(want), record.job)
    if code != PROBE_ADMIT:
        out.record = _set_drain(p, out.record, False)
        verdict = "refused" if code == PROBE_REFUSED else "unknown"
        out.detail = (f"wants size {want} (now {record.size}), {streak} passes; run-on probe {verdict}, "
                      f"keeps {record.job}: {why[:160]}")
        return out
    if not p.ops.idle(p.runner):
        out.record = _set_drain(p, out.record, streak >= (GROW_DRAIN_AFTER if grow else RESIZE_AFTER))
        drained = "; draining" if out.record.draining else ""
        out.detail = f"wants size {want} (now {record.size}), {streak} passes{drained}"
        return out
    out = _cancel(p, out, "resize", f"{record.job} idle; size {record.size} to {want} "
                                    f"for {p.demand} live {p.runner.demand} lanes")
    if not p.dry_run:
        out.record = replace(out.record, resize_to=want)
    return out


def _probe(p: Pass, sized: Runner, replacing: str | None) -> tuple[int, str]:
    """run-on's check-only admission answer; ops without a probe (older fakes) admit."""

    probe = getattr(p.ops, "probe", None)
    return probe(sized, replacing) if probe is not None else (PROBE_ADMIT, "")


def _submit(p: Pass, record: Record) -> tuple[str, int | None]:
    """Start the job at the wanted size, else each smaller one (run-on admits inside the share),
    skipping sizes ``run-on probe`` refuses for capacity; the submit's own refusal still counts.

    After a resize cancel, at least the size the resize was for: 2b launches wait while A is down,
    so the live lanes the restart pass counts are its own dip, and sizing from them made A shrink
    back and grow again every few passes (26 Oct)."""

    want = p.runner.wanted(p.demand)
    if want is not None and record.cancelled_for == "resize" and record.resize_to is not None:
        want = max(want, record.resize_to)
    tries = list(range(want, -1, -1)) if want is not None else [None]
    errors: list[SubmitRefused] = []
    for index in tries:
        sized = p.runner.sized(index)
        # a size the probe refuses for capacity is skipped; an unknown answer leaves it to the submit
        code, why = _probe(p, sized, None)
        if code == PROBE_REFUSED and any(c in why for c in CAPACITY):
            errors.append(SubmitRefused(f"run-on probe for runner {p.runner.name}: {why[:160]}"))
            continue
        try:
            return p.ops.submit(sized, job_env(sized, pause_gib=p.pause_gib)), index
        except SubmitRefused as exc:  # an unclear failure is raised: it may have started a job
            errors.append(exc)
    text = f"runner {p.runner.name}: no size admitted: {'; '.join(map(str, errors))[-400:]}"
    if all(exc.capacity for exc in errors):
        raise NoCapacity(text)
    raise OSError(text)


def _later(*stamps: str | None) -> str | None:
    """The latest of the ISO stamps given (None when there are none)."""

    known = [s for s in stamps if s]
    return max(known, key=datetime.fromisoformat) if known else None


def _refund(record: Record, state: str) -> Record:
    """A REFUSED job never ran: take its submission off the restarts and note the refusal."""

    if state != "REFUSED":
        return record
    restarts = tuple(at for at in record.restarts if at != record.submitted_at)
    return replace(record, restarts=restarts,
                   refused_at=_later(record.refused_at, record.submitted_at))


def refused(record: Record) -> bool:
    """Whether the runner's last start was refused for capacity and none has started since."""

    return bool(record.refused_at) and not (
        record.submitted_at and record.submitted_at > record.refused_at)


def _escalate(p: Pass, out: Outcome, state: str) -> Outcome:
    record = out.record
    out.action = "escalate"
    out.detail = f"{record.job} {state}; {RESTARTS_A_DAY} restarts today already"
    if record.escalated_on != _today(p.now):
        out.messages.append(f"runner {p.runner.name} on {p.runner.spool}: job {record.job} is "
                            f"{state} after {RESTARTS_A_DAY} restarts today; the tick stops "
                            "restarting it until tomorrow. Look at its outputs/stage3.log")
        out.record = replace(record, escalated_on=_today(p.now))
    return out


def _dead(p: Pass, out: Outcome, state: str, age: float | None) -> Outcome:
    """A dead job: adopt another live server, wait, hold, escalate, or start its successor."""

    record = out.record = _refund(out.record, state)
    if state == "DONE" or p.ops.stopped(p.runner):
        out.action, out.detail = "stopped", f"{record.job} {state}; STOP on the spool, left stopped"
        return out
    adopted, how = _adopt(p, record, out)
    if how:
        out.action = how
        if adopted:
            out.detail = f"{record.job} {state}; live job {adopted.job} serves"
            out.record = adopted
        elif how == "conflict":
            out.detail = "several live jobs"
        return out
    if age is not None and age < SERVING_S:
        out.action, out.detail = "wait-heartbeat", f"{record.job} {state} but the spool heartbeat is {age:.0f} s old"
        return out
    if p.hold_submit:
        out.action, out.detail = "held-disk", f"{record.job} {state}; the disk guard holds new runner jobs"
        return out
    resize = record.cancelled_for == "resize" and record.cancelled == record.job
    if not resize and _restarts_today(record, p.now) >= RESTARTS_A_DAY:
        return _escalate(p, out, state)
    out.action, out.detail = "restart", f"{record.job} {state}; new job from {TEMPLATE}"
    if p.dry_run:
        return out
    stamp = p.now.isoformat(timespec="seconds")
    try:
        new, size = _submit(p, record)
    except NoCapacity as exc:  # nothing ran: no restart is counted and no job is recorded
        out.action, out.detail = "held-capacity", f"{record.job} {state}; {exc}"
        out.record = replace(record, refused_at=stamp)
        return out
    restarts = record.restarts if resize else (*record.restarts[-9:], stamp)
    # the new job's script deletes the spool's DRAIN before it serves
    out.record = replace(record, job=new, submitted_at=stamp, cancelled=None, cancelled_for=None,
                         restarts=restarts, size=size, resize_streak=0, refused_at=None,
                         draining=False, resize_to=None)
    out.detail += f": {new}" + (f" at size {size}" if size is not None else "")
    return out


def ensure_one(runner: Runner, record: Record, ops: Ops, *, now: datetime, hold_submit: bool,
               pause_gib: float | None = None, demand: int = 0, dry_run: bool = False) -> Outcome:
    """One runner's pass (see the module docstring). Never touches another runner's spool."""

    p = Pass(runner, ops, now, hold_submit, pause_gib, demand, dry_run)
    out = Outcome(runner.name, "ok", "", record)
    if record.job is None:
        adopted, how = _adopt(p, record, out)
        if how in ("conflict", "unverified"):
            out.action = how
            return out
        if adopted:
            out.record = adopted
            out.detail = f"adopted live job {adopted.job}"
    state = ops.job_state(out.record.job) if out.record.job else "MISSING"
    out.job_state = state
    out.heartbeat_age_s = age = ops.heartbeat_age(runner)
    if state == LIVE:
        return _live(p, out, age)
    if state not in DEAD and state != "DONE":
        out.action, out.detail = "unverified", f"{out.record.job} state {state}; nothing done"
        return out
    return _dead(p, out, state, age)


def job_env(runner: Runner, *, pause_gib: float | None) -> dict[str, str]:
    """The job's environment: the declaration, the spool and the disk guard's pause line."""

    env = {"MENAGERIE_MEMORY_GB": str(runner.memory_gib), "MENAGERIE_CPUS": str(runner.cpus),
           "MENAGERIE_TICK_RUNNER": runner.name, "RUNNER_SPOOL": runner.spool,
           "RUNNER_WORKERS": str(runner.workers), "RUNNER_WORK_ROOT": runner.work_root}
    if pause_gib is not None:
        env["MENAGERIE_DISK_PAUSE_GIB"] = f"{pause_gib:g}"
    return env


# The runner that serves the bridge's 2b tries (spool-A): a 2b lane's request caps follow its size.
TRIES_RUNNER = "A"


def tries_declaration(summary: Mapping[str, Any] | None, ops: Ops | None = None
                      ) -> tuple[dict[str, str] | None, str]:
    """The 2b lanes' declaration from this pass's summary of ``TRIES_RUNNER``: its live job's
    size (the size it was started at, else its admitted allocation), never its largest size or
    the hub's. ``(None, why)`` when the runner is not running or its size is unknown."""

    name = TRIES_RUNNER
    if not summary:
        return None, f"runner {name} not read this pass"
    if summary.get("job_state") != LIVE or summary.get("action") in ("cancel-stale", "resize"):
        return None, (f"runner {name} not running ({summary.get('job')}, "
                      f"{summary.get('job_state')}, {summary.get('action')})")
    runner = next((r for r in runners() if r.name == name), None)
    if runner is None:
        return None, f"runner {name} not configured"
    size = summary.get("size")
    if isinstance(size, int) and 0 <= size < len(runner.sizes):
        sized = runner.sized(size)
        cpus, gib = sized.cpus, sized.memory_gib
    else:
        allocation = getattr(ops or RunOn(), "allocation", lambda job: None)(summary.get("job"))
        if allocation is None:
            return None, f"runner {name} job {summary.get('job')} size unknown"
        cpus, gib = allocation
    return {"MENAGERIE_CPUS": str(cpus), "MENAGERIE_MEMORY_GB": str(gib)}, ""


# --- the real calls (run-on and ssh) ---------------------------------------------------------


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _jobs_mirror() -> Path:
    base = os.environ.get("FLEET_STATE_DIR") or str(Path.home() / ".local/state/fleet")
    return Path(base) / "jobs"


class RunOn:
    """``Ops`` through ``run-on`` (status, submit, cancel) and one ssh per spool read."""

    def __init__(self, *, timeout: float = 60.0, ref: str = "main",
                 ended: Path | None = None) -> None:
        self.timeout, self.ref, self.ended = timeout, ref, ended
        self._ended: dict[str, str] | None = None

    def _run(self, argv: Sequence[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(list(argv), capture_output=True, text=True, check=False, cwd=_repo(),
                              timeout=timeout or self.timeout)

    def _ended_jobs(self) -> dict[str, str]:
        if self._ended is None:
            try:
                self._ended = json.loads(self.ended.read_text(encoding="utf-8")) if self.ended else {}
            except (OSError, ValueError):
                self._ended = {}  # unreadable: every job is asked again and the file rewritten
        return self._ended

    def job_state(self, job: str) -> str:
        """``_job_state``, remembered in ``ended`` (when given) once it is one of ``ENDED``."""

        known = self._ended_jobs().get(job)
        if known in ENDED:
            return known
        state = self._job_state(job)
        if state in ENDED and self.ended is not None:
            self._ended_jobs()[job] = state
            tmp = self.ended.with_name(f".{self.ended.name}.tmp")
            tmp.write_text(json.dumps(self._ended, indent=0, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, self.ended)
        return state

    def _job_state(self, job: str) -> str:
        """PENDING or REFUSED from admission; else the worker's own reading (RUNNING, DONE, ...).

        run-on's ``admission_timeout`` verdict (its admission wait ran out with no room; the lane is over and
        the job never launched) and a ``refused`` one for capacity read as REFUSED; a refusal for
        any other reason reads as REJECTED. The worker has no record of either.
        """

        admitted = self._run(["run-on", "state", job, "--json"])
        try:
            reading = json.loads(admitted.stdout)
            verdict, reason = reading.get("verdict"), str(reading.get("reason") or "")
        except (ValueError, AttributeError):
            verdict, reason = None, ""
        if verdict == "pending":
            return "PENDING"
        if verdict == "admission_timeout" or (verdict == "refused" and reason.startswith(CAPACITY)):
            return "REFUSED"
        if verdict == "refused":
            return "REJECTED"
        done = self._run(["run-on", "status", job])
        if done.returncode == 3 and "HOST_UNREACHABLE" in done.stdout:
            return "HOST_UNREACHABLE"
        fields = done.stdout.split()
        if done.returncode or len(fields) < 3:
            return "UNKNOWN"
        return fields[2].rstrip(":")

    def _stat(self, runner: Runner, script: str) -> str:
        host = os.environ.get("MENAGERIE_RUNNER_HOST")
        if not host:
            raise OSError("MENAGERIE_RUNNER_HOST is not set; it names the worker's ssh alias")
        done = subprocess.run(["ssh", host, script], capture_output=True, text=True, check=False, timeout=30)
        if done.returncode:
            raise OSError(f"ssh {host}: {done.stderr.strip()[-200:]}")
        return done.stdout.strip()

    def heartbeat_age(self, runner: Runner) -> float | None:
        beat = shlex.quote(f"{runner.spool}/stage3/serving")
        out = self._stat(runner, f"if [ -f {beat} ]; then echo $(( $(date +%s) - $(stat -c %Y {beat}) )); fi")
        return float(out) if out else None

    def stopped(self, runner: Runner) -> bool:
        return self._stat(runner, f"[ -f {shlex.quote(runner.spool + '/STOP')} ] && echo y || true") == "y"

    def idle(self, runner: Runner) -> bool:
        """Nothing claimed on the spool (runner requests or stage-3 jobs): safe to restart."""

        dirs = " ".join(shlex.quote(f"{runner.spool}/{d}") for d in ("claimed", "stage3/claimed"))
        out = self._stat(runner, f"find {dirs} -maxdepth 1 -name '*.json' 2>/dev/null | wc -l")
        return out == "0"

    def drain(self, runner: Runner, on: bool) -> None:
        path = shlex.quote(f"{runner.spool}/{DRAIN}")
        self._stat(runner, f"mkdir -p $(dirname {path}) && touch {path}" if on else f"rm -f {path}")

    def discover(self, runner: Runner) -> list[tuple[str, str | None]]:
        """Recent mirrored menagerie jobs serving this runner's spool, with start times, newest first."""

        marker = re.compile(rf"MENAGERIE_TICK_RUNNER={re.escape(runner.name)}(?![\w-])")
        legacy = re.compile(re.escape(runner.legacy) + r"(?:\\n|\s|;|$)") if runner.legacy else None
        cutoff = datetime.now(timezone.utc) - timedelta(days=DISCOVER_DAYS)
        found = []
        for manifest in _jobs_mirror().glob("menagerie-*/manifest.json"):
            try:
                meta = json.loads(manifest.read_text(encoding="utf-8"))
                started = datetime.fromisoformat(meta["started"].replace("Z", "+00:00"))
            except (OSError, ValueError, KeyError):
                continue
            cmd = meta.get("cmd", "")
            if started >= cutoff and (marker.search(cmd) or (legacy and legacy.search(cmd))):
                found.append((meta["started"], meta["job"]))
        return [(job, at.replace("Z", "+00:00")) for at, job in sorted(found, reverse=True)]

    def submit(self, runner: Runner, env: Mapping[str, str]) -> str:
        # the host is the project's default compute machine (.compute-host): the store is there
        argv = ["run-on", "submit", "--project", "menagerie", "--os", "linux",
                "--ref", self.ref, "--cpus", str(runner.cpus), "--memory-gib", str(runner.memory_gib),
                "--no-fallback", "--restartable", "--pull", "outputs/*", "--admit-wait", "90", "--",
                "env", *(f"{k}={v}" for k, v in env.items()), "bash", TEMPLATE]
        done = self._run(argv, timeout=180)
        text = done.stdout + done.stderr
        launched = re.search(r"^job: +(\S+)", text, re.MULTILINE)
        if launched:
            return launched.group(1)
        # admission ended without a launch (``admission_timeout: NO_CAPACITY ...`` once the admission wait runs
        # out, or ``refused: ...``): nothing ran, even when "awaiting admission" was printed first
        why = re.search(r"(?:admission_timeout|waiting|refused): [A-Z_]+[^;\n]{0,160}", text)
        if why:
            raise SubmitRefused(f"run-on submit for runner {runner.name}: {why.group(0)}")
        # still awaiting admission when the front end stopped waiting: the job may yet start, so
        # its id is recorded and the next pass reads its state
        job = re.search(r"^run-on: (menagerie-\S+) awaiting admission", text, re.MULTILINE)
        if job:
            return job.group(1)
        raise OSError(f"run-on submit for runner {runner.name}: rc {done.returncode}, no job id: "
                      f"{text.strip()[-200:]}")

    def probe(self, runner: Runner, replacing: str | None) -> tuple[int, str]:
        """Whether run-on would admit ``runner``'s declaration on the compute host now, as if
        ``replacing`` had ended: ``(exit, reason)``, with every failure to ask read as unknown."""

        host_file = _repo() / ".compute-host"
        host = host_file.read_text(encoding="utf-8").strip() if host_file.is_file() else ""
        if not host:
            return PROBE_UNKNOWN, "no .compute-host"
        argv = ["run-on", "probe", host, "--project", "menagerie", "--os", "linux",
                "--cpus", str(runner.cpus), "--memory-gib", str(runner.memory_gib),
                *(("--replacing", replacing) if replacing else ()), "--json"]
        try:
            done = self._run(argv, timeout=120)
        except subprocess.TimeoutExpired:
            return PROBE_UNKNOWN, "run-on probe timed out"
        try:
            reason = str(json.loads(done.stdout).get("reason") or "")
        except (ValueError, AttributeError):
            reason = (done.stdout + done.stderr).strip()[-200:]
        code = done.returncode if done.returncode in (PROBE_ADMIT, PROBE_REFUSED) else PROBE_UNKNOWN
        return code, reason

    def cancel(self, job: str) -> None:
        done = self._run(["run-on", "cancel", job])
        if done.returncode:
            raise OSError(f"run-on cancel {job}: {done.stderr.strip()[-200:]}")

    def allocation(self, job: str) -> tuple[int, int] | None:
        """The job's admitted declaration, including adopted legacy jobs; unknown holds G."""

        path = _jobs_mirror() / job / "manifest.json"
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                return None
            admitted = meta.get("memory_admitted_gib")
            cpus = float(meta["cpus"])
            gib = float(meta["memory_gib"] if admitted is None else admitted)
            # Never truncate an admitted declaration: an undercount could permit growth
            # past the share. Noncanonical declarations require conductor recovery.
            if not (cpus.is_integer() and gib.is_integer() and cpus > 0 and gib > 0):
                return None
            return int(cpus), int(gib)
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            return None


def lane_room(cap_2b: int, *, big_runner: bool = False,
              path: Path | None = None) -> tuple[int, int]:
    """CPUs and GiB of the worker share left for agent lanes on the runner host.

    The runners keep their largest footprint the 2b cap can call for: A at the size ``cap_2b``
    live lanes want, every other day runner at its largest size, and with the night big runner
    the larger of that and the night layout (A and B smallest, G as declared). Lanes get only
    what is left, so a lane never takes room a runner start or resize needs.
    """

    by_name = {r.name: r for r in runners(path)}
    day = [r.sized(r.wanted(cap_2b)) if r.demand == "2b" else r.sized(len(r.sizes) - 1 if r.sizes else None)
           for name, r in by_name.items() if name != "G"]
    used = (sum(r.cpus for r in day), sum(r.memory_gib for r in day))
    if big_runner and {"A", "B", "G"} <= by_name.keys():
        night = (by_name["A"].sized(0), replace(by_name["B"], sizes=B_SIZES).sized(0), by_name["G"])
        used = (max(used[0], sum(r.cpus for r in night)), max(used[1], sum(r.memory_gib for r in night)))
    return max(0, SHARE_CPUS - used[0]), max(0, SHARE_GIB - used[1])


def live_lanes(kinds: Sequence[str]) -> dict[str, int]:
    """Live lanes by kind (fleet-run reads), only for the kinds a runner's size follows."""

    if not kinds:
        return {}
    from menagerie.pipeline import DataLayout, admission

    counts: dict[str, int] = {}
    for lane in admission.lanes(data=DataLayout.from_env()):
        counts[lane["kind"]] = counts.get(lane["kind"], 0) + 1
    return {k: counts.get(k, 0) for k in kinds}


def ensure(*, spool: str | None = None, dry_run: bool = False, ops: Ops | None = None,
           state: Mapping[str, Any] | None = None, hold_submit: bool = False,
           pause_gib: float | None = None, demand: Mapping[str, int] | None = None,
           now: datetime | None = None, mode: str | None = None) -> list[dict[str, Any]]:
    """Run ``ensure_one`` for every configured runner (or the one on ``spool``).

    Standalone (the ``runner ensure`` verb) it takes the tick's lock and reads and writes the
    tick's runner records itself; the tick passes ``state`` and saves the returned records.
    ``demand`` gives live lanes by kind (default: read when a runner's size follows a kind);
    ``mode`` picks the runners that run in it (default: the tick folder's mode).
    Returns: one summary per runner, each with its ``record`` and any ``messages`` for Crawler.
    """

    from menagerie.pipeline import tick

    cfg = tick.config()
    chosen = [r for r in runners() if (mode or tick.read_mode()) in r.modes
              and (not spool or spool.rstrip("/") in (r.spool, f"~/{r.spool}"))]
    if state is None and not dry_run:  # standalone: never beside a running tick
        with tick.locked() as mine:
            if not mine:
                return [{"skipped": "a tick holds the lock; it runs runner ensure itself"}]
            saved = tick.load_state()
            out = _policy_pass(chosen, ops, saved, hold_submit, pause_gib, demand, now, dry_run, cfg)
            records = {**saved.get("runners", {}), **{r["runner"]: r["record"] for r in out}}
            tick.save_state({**saved, "runners": records})
            return out
    saved = state if state is not None else tick.load_state()
    return _policy_pass(chosen, ops, saved, hold_submit, pause_gib, demand, now, dry_run, cfg)


def _ensure_all(chosen: Sequence[Runner], ops: Ops | None, saved: Mapping[str, Any], hold_submit: bool,
                pause_gib: float | None, demand: Mapping[str, int] | None,
                now: datetime | None, dry_run: bool) -> list[dict[str, Any]]:
    ops = ops or RunOn()
    now = now or datetime.now(timezone.utc)
    records = dict(saved.get("runners", {}))
    if demand is None:
        demand = live_lanes(sorted({r.demand for r in chosen if r.demand}))
    out = []
    for runner in chosen:
        result = ensure_one(runner, Record.from_mapping(records.get(runner.name)), ops, now=now,
                            hold_submit=hold_submit, pause_gib=pause_gib,
                            demand=demand.get(runner.demand or "", 0), dry_run=dry_run)
        out.append({**result.summary(), "record": result.record.to_mapping(),
                    "messages": result.messages})
    return out


def big_work(state: Mapping[str, Any]) -> bool:
    """Whether the night big runner has rows to run: the tick's last count of reserved (18, 40]
    GiB rows plus spool-G's waiting jobs (``big_work`` in its state). An unknown count keeps G on;
    a zero count leaves G off (or drains it), so A, B and the night boost keep the night."""

    count = state.get("big_work")
    return count is None or int(count) > 0


def big_hold_open(now: datetime | None = None) -> bool:
    """The night window with a 30-minute lead-in (9:30 PM to 6 AM Pacific): lanes on the trace
    host stop starting early, so their memory pledges have ended when G asks for admission."""

    now = now or datetime.now(timezone.utc)
    return night_open(now) or night_open(now + BIG_LEAD_IN)


def night_open(now: datetime | None = None) -> bool:
    """The 10 PM inclusive to 6 AM exclusive Pacific window, including DST."""

    hour = (now or datetime.now(timezone.utc)).astimezone(NIGHT_ZONE).hour
    return hour >= 22 or hour < 6


def _observed(runner: Runner, record: Record, ops: Ops) -> tuple[Record, str, tuple[int, int] | None]:
    """Resolve adoption and declaration without changing jobs. Ambiguity holds the transition."""

    jobs = {job: at for job, at in ops.discover(runner)}
    if record.job:
        jobs[record.job] = record.submitted_at
    states = {job: ops.job_state(job) for job in jobs}
    active = [(job, at, states[job]) for job, at in jobs.items()
              if states[job] not in (*DEAD, "DONE")]
    if not active:
        age = ops.heartbeat_age(runner)
        if runner.name == "G" and age is not None and age < SERVING_S:
            return record, "unverified", None
        return record, "inactive", (0, 0)
    if len(active) != 1:
        return record, "unverified", None
    job, at, status = active[0]
    allocation = getattr(ops, "allocation", lambda job: None)(job)
    size = next((i for i, (_, cpus, gib) in enumerate(runner.sizes)
                 if allocation == (cpus, gib)), None)
    return replace(record, job=job, submitted_at=at, size=size), status, allocation


def _summary(out: Outcome) -> dict[str, Any]:
    return {**out.summary(), "record": out.record.to_mapping(), "messages": out.messages}


def _hold(runner: Runner, record: Record, why: str, state: str | None = None) -> dict[str, Any]:
    """A held transition; ``state`` is the job's observed state, so a live runner held from
    growing still reads as live (``tries_declaration``)."""

    return _summary(Outcome(runner.name, "held-big", why, record, job_state=state))


def _drain_big(runner: Runner, record: Record, ops: Ops, status: str, *,
               dry_run: bool) -> dict[str, Any]:
    if status == "inactive":
        return _summary(Outcome(runner.name, "off", "night runner inactive",
                                replace(record, job=None, draining=False)))
    if status != LIVE:
        return _hold(runner, record, "night runner draining or unverified; A/B growth held")
    if not ops.idle(runner):
        if not (record.draining or dry_run):
            ops.drain(runner, True)
            record = replace(record, draining=True)
        return _hold(runner, record, "night runner draining (claims no new job); A/B growth held")
    out = Outcome(runner.name, "resize", "drain G before restoring A/B", record)
    if not dry_run:
        ops.cancel(record.job)
        out.record = replace(record, cancelled=record.job, cancelled_for="resize")
    return _summary(out)


def _policy_pass(chosen: Sequence[Runner], ops: Ops | None, saved: Mapping[str, Any],
                 hold_submit: bool, pause_gib: float | None, demand: Mapping[str, int] | None,
                 now: datetime | None, dry_run: bool, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = saved.get("runners", {})
    enabled = cfg.get("big_runner", False) and big_work(saved)
    managed = enabled or records.get("G", {}).get("job") or records.get("B", {}).get("size") == 0
    if not managed:
        # Preserve the original path exactly: no probes, resize ladder or submissions for G.
        return _ensure_all([r for r in chosen if r.name != "G"], ops, saved, hold_submit,
                           pause_gib, demand, now, dry_run)
    by_name = {r.name: r for r in chosen}
    if not _big_layout(by_name):
        return [_hold(r, Record.from_mapping(records.get(r.name)),
                      "big-runner transitions require canonical A/B/G declarations in full mode") for r in chosen]
    ops, now = ops or RunOn(), now or datetime.now(timezone.utc)
    by_name["B"] = replace(by_name["B"], sizes=B_SIZES, demand="big-B", lanes_per_worker=1)
    out = _big_pass(by_name, records, ops, now, enabled=enabled, hold_submit=hold_submit,
                    pause_gib=pause_gib, demand=demand, dry_run=dry_run,
                    b_gives_way=bool(cfg.get("g_replaces_b", False)) and enabled and night_open(now),
                    b_boost=bool(cfg.get("b_boost", False)))
    return _with_share(out, ops)


def _big_layout(by_name: Mapping[str, Runner]) -> bool:
    expected = {r.name: r for r in DEFAULT_RUNNERS}
    if by_name.keys() != expected.keys():
        return False
    def declarations(r: Runner) -> tuple[int, int, int, str]:
        return r.cpus, r.memory_gib, r.workers, r.spool
    return all(declarations(r) == declarations(expected[n]) for n, r in by_name.items()) and \
        by_name["A"].sizes == expected["A"].sizes


def _share(allocations: Sequence[tuple[int, int] | None]) -> dict[str, int] | None:
    if any(a is None for a in allocations):
        return None
    return {"cpus": sum(a[0] for a in allocations), "memory_gib": sum(a[1] for a in allocations)}


def _with_share(out: Sequence[dict[str, Any]], ops: Ops) -> list[dict[str, Any]]:
    jobs = [r["record"].get("job") for r in out]
    active = [job for job in jobs if job and ops.job_state(job) not in (*DEAD, "DONE")]
    share = _share([ops.allocation(job) for job in active])
    return [{**r, "share": share} for r in out]


def _small_pass(by_name: Mapping[str, Runner], observed: Mapping[str, Any], ops: Ops, now: datetime, *,
                night: bool, hold_submit: bool, pause_gib: float | None,
                demand: Mapping[str, int] | None, dry_run: bool,
                held: Mapping[str, str] | None = None, b_gives_way: bool = False,
                b_boost: bool = False) -> list[dict[str, Any]]:
    demands = dict(demand) if demand is not None else live_lanes(["2b"])
    out = []
    for name in ("B", "A"):
        runner = by_name[name]
        record, status, allocation = observed[name]
        if name in (held or {}):
            out.append(_hold(runner, record, held[name], status))
            continue
        if name == "B" and b_gives_way:
            out.append(_give_way(runner, record, ops, status, dry_run=dry_run))
            continue
        if status != "inactive" and (allocation is None or record.size is None):
            out.append(_hold(runner, record, "unknown admitted size; transition held", status))
            continue
        if night and status == "inactive" and not _restart_fits(runner, observed):
            out.append(_hold(runner, record, "small restart would exceed or cannot verify the share",
                             status))
            continue
        count = 0 if night else (_b_day(by_name, record, status, ops, demands.get("2b", 0), b_boost)
                                 if name == "B" else demands.get("2b", 0))
        out.append(_summary(ensure_one(runner, record, ops, now=now, hold_submit=hold_submit,
                                      pause_gib=pause_gib, demand=count, dry_run=dry_run)))
    return out


def _b_day(by_name: Mapping[str, Runner], record: Record, status: str, ops: Ops, live_2b: int,
           boost: bool) -> int:
    """B's day want: with ``boost`` on, the boost when the share allows it and ``_boost_room``
    holds; else its day size."""

    if not boost:
        return B_SIZES[B_DAY][0]
    workers = _b_day_workers(by_name["A"], live_2b)
    job = record.job if status == LIVE else None
    if workers == B_SIZES[-1][0] and not _boost_room(by_name["B"], job, ops):
        return B_SIZES[B_DAY][0]
    return workers


def _boost_room(b: Runner, job: str | None, ops: Ops) -> bool:
    """Whether ``run-on probe`` (replacing B's job) admits the boost plus ``BOOST_MARGIN_GIB``; an
    unknown answer counts as no room. Ops without a probe (older fakes) admit."""

    probe = getattr(ops, "probe", None)
    if probe is None:
        return True
    boost = b.sized(len(B_SIZES) - 1)
    code, _ = probe(replace(boost, memory_gib=boost.memory_gib + BOOST_MARGIN_GIB), job)
    return code == PROBE_ADMIT


def _b_day_workers(a: Runner, live_2b: int) -> int:
    """B's day want (in workers): its boost while it fits the share beside the size A's live 2b
    lanes want, else its day size, so a growing A is never refused for B's boost."""

    wanted = a.sized(a.wanted(live_2b))
    _, cpus, gib = B_SIZES[-1]
    fits = wanted.cpus + cpus <= SHARE_CPUS and wanted.memory_gib + gib <= SHARE_GIB
    return B_SIZES[-1 if fits else B_DAY][0]


READY_WAIT = {False: "A/B must be confirmed at 3 CPUs/20 GiB and 3 CPUs/16 GiB",
              True: "A must be confirmed at 3 CPUs/20 GiB and B stopped"}


def _give_way(runner: Runner, record: Record, ops: Ops, status: str, *, dry_run: bool) -> dict[str, Any]:
    """B at night while G has big rows (``g_replaces_b``; Crawler, 28 Oct): drained, never
    interrupted, then cancelled when idle, so G starts in its room; it comes back at 6 AM or once
    G's big rows run out (the day path)."""

    if status == "inactive":
        return _summary(Outcome(runner.name, "off", "gives way to G at night (big rows wait)",
                                replace(record, draining=False)))
    if status != LIVE:
        return _hold(runner, record, "giving way to G: B unverified", status)
    if not ops.idle(runner):
        if not (record.draining or dry_run):
            ops.drain(runner, True)
            record = replace(record, draining=True)
        return _hold(runner, record, "draining to give way to G (claims no new job)", status)
    out = Outcome(runner.name, "resize", "B gives way to G at night", record)
    if not dry_run:
        ops.cancel(record.job)
        out.record = replace(record, cancelled=record.job, cancelled_for="g")
    return _summary(out)


def _restart_fits(runner: Runner, observed: Mapping[str, Any]) -> bool:
    share = _share([v[2] for v in observed.values()])
    _, cpus, gib = runner.sizes[0]
    return share is not None and share["cpus"] + cpus <= SHARE_CPUS and \
        share["memory_gib"] + gib <= SHARE_GIB


def _a_beside_g(by_name: Mapping[str, Runner], observed: Mapping[str, Any], ops: Ops, *,
                night: bool) -> bool | None:
    """At night with A down and G present: does run-on admit A's night size beside G now?
    None when there is no such contest. Asked every pass, with no refusal back-off: 2b tries
    (A) outrank G's big rows (Crawler, 26 Oct), so A returns the moment the host has room."""

    if not night or observed["A"][1] != "inactive" or observed["G"][1] == "inactive":
        return None
    probe = getattr(ops, "probe", None)
    return probe is None or probe(by_name["A"].sized(0), None)[0] == PROBE_ADMIT


def _big_ready(by_name: Mapping[str, Runner], observed: Mapping[str, Any],
               out: Sequence[Mapping[str, Any]], *, b_gives_way: bool = False) -> bool:
    want = {n: (LIVE, by_name[n].sizes[0][1:]) for n in ("A", "B")}
    if b_gives_way:
        want["B"] = ("inactive", (0, 0))  # gone, not shrunk (``_give_way``)
    small = all(observed[n][1:] == w for n, w in want.items())
    steady = all(r["action"] in ("ok", "off") for r in out)
    share = _share([v[2] for v in observed.values()])
    return small and steady and share is not None and \
        share["cpus"] <= SHARE_CPUS and share["memory_gib"] <= SHARE_GIB



def _draining_small(by_name: Mapping[str, Runner], observed: Mapping[str, Any], ops: Ops, now: datetime, *,
                    hold_submit: bool, pause_gib: float | None, demand: Mapping[str, int] | None,
                    dry_run: bool) -> list[dict[str, Any]]:
    if observed["G"][1:] == (LIVE, (by_name["G"].cpus, by_name["G"].memory_gib)):
        return _small_pass(by_name, observed, ops, now, night=True, hold_submit=hold_submit,
                           pause_gib=pause_gib, demand=demand, dry_run=dry_run)
    return [_hold(by_name[n], observed[n][0], "G must be confirmed inactive before growth", observed[n][1])
            for n in ("B", "A")]


def _big_pass(by_name: Mapping[str, Runner], records: Mapping[str, Any], ops: Ops, now: datetime, *,
              enabled: bool, hold_submit: bool, pause_gib: float | None,
              demand: Mapping[str, int] | None, dry_run: bool,
              b_gives_way: bool = False, b_boost: bool = False) -> list[dict[str, Any]]:
    """Downsize, verify, start G; drain, verify, restore. No grow and G-start in one pass. With
    ``b_gives_way`` B is drained off at night instead of shrunk, and G starts beside A alone;
    with ``b_boost`` (config) B may want its day boost."""

    observed = {name: _observed(r, Record.from_mapping(records.get(name)), ops)
                for name, r in by_name.items()}
    night = enabled and night_open(now)
    g_record, g_status, _ = observed["G"]
    if not night and g_status != "inactive":
        small = _draining_small(by_name, observed, ops, now, hold_submit=hold_submit,
                               pause_gib=pause_gib, demand=demand, dry_run=dry_run)
        return [_drain_big(by_name["G"], g_record, ops, g_status, dry_run=dry_run),
                *small]
    if not night and _share([v[2] for v in observed.values()]) is None:
        return [_hold(by_name[n], observed[n][0], "unknown declaration; restoration held", observed[n][1])
                for n in ("B", "A", "G")]
    room = _a_beside_g(by_name, observed, ops, night=night)
    if room:
        a_record, a_status, a_allocation = observed["A"]
        observed = {**observed, "A": (replace(a_record, refused_at=None), a_status, a_allocation)}
    held = {"A": "no room beside G: G finishes its row, drains, then A returns"} if room is False else None
    out = _small_pass(by_name, observed, ops, now, night=night, hold_submit=hold_submit,
                      pause_gib=pause_gib, demand=demand, dry_run=dry_run, held=held,
                      b_gives_way=b_gives_way, b_boost=b_boost)
    if not night or room is False:
        return [*out, _drain_big(by_name["G"], g_record, ops, g_status, dry_run=dry_run)]
    share = _share([v[2] for v in observed.values()])
    g = by_name["G"]
    if g_record.draining and g_status == LIVE and not dry_run:
        ops.drain(g, False)  # A came back beside it: G claims rows again
        g_record = replace(g_record, draining=False)
    if not _big_ready(by_name, observed, out, b_gives_way=b_gives_way):
        out.append(_hold(g, g_record, READY_WAIT[b_gives_way]))
    else:
        result = ensure_one(g, g_record, ops, now=now, hold_submit=hold_submit,
                            pause_gib=pause_gib, dry_run=dry_run)
        out.append(_summary(result))
        if result.action == "restart":
            share = {"cpus": 6 + g.cpus, "memory_gib": 36 + g.memory_gib}
    return [{**r, "planned_share": share} for r in out]


def big(*, dry_run: bool = False) -> list[dict[str, Any]]:
    """Run the config-gated, share-checked night policy through the usual tick lock."""

    return ensure(dry_run=dry_run)
