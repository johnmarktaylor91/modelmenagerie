"""``tick``: one idempotent driver pass under ``flock -n`` (BUILD-PLAN r3 section 2, lane r3-tick).

Run every 10 minutes on the hub (the coordinating machine) by the persistent user timer ``menagerie-tick.timer``. A pass
reads the worker-disk guard first, then runs each step in its own ``try``, so one failing step
never stops the others, and stops starting steps once its 8-minute bound has passed:

- both modes: ``disk`` (the guard), ``runners`` (``runner ensure``), ``collect`` (finished stage-3
  results from each spool onto the data checkout, committed by name; files another writer left
  dirty are skipped), ``drain`` (hub staging to the worker store), ``env_missing`` (ids whose latest
  trace row of the last day found no built environment) and ``status``;
- ``full`` mode adds ``requeue`` (queued lanes), ``stage3_queue`` (worklist rows whose next step
  is a forward pass or trace, up to the spool's depth), ``top_up`` (2a and 2b lanes for worklist
  rows, up to the lane limits), ``trickle`` (``paper harvest --limit``, its packets, citation
  and store-index lines committed by the tick), ``reviews`` (config ``reviews.enabled``: seat
  calibration at the prompt in force, verdicts ended review lanes left unrecorded, and review
  lanes for worklist rows at ``review`` up to ``reviews.lanes``) and ``describes`` (config
  ``describes.enabled``: one detached ``describe write`` run at a time, ``describe_stage``);
- both modes end with ``lanes``: what lanes wrote to the data checkout (packets, adapters, model
  records, roster, id, citation, ledger and store-index files) committed by name once no live or
  queued lane holds its family and it reads whole (``lane_commit``; config ``commit_lanes``),
  then ``lane_reap``: run-on removes the worker checkout of each 2b lane whose outputs the hub
  relay has ingested and whose results are committed (config ``reap_lanes``).
- ``refill`` (``menagerie-refill.timer``, minutes 1, 3, 5 and 7 of each ten, only while
  ``refill.want`` exists: the relay touches it when a lane ends, and every pass while a runner's
  last start was refused for capacity): ``disk`` and a 2b-only ``top_up`` under the same lock,
  bounded to end before the next full pass; it leaves the runners to the full pass unless one was
  refused, then retries it as the full pass would (runner A's declaration comes from its saved
  record and live job state), reads
  the last full pass's cohort verdicts and holds when they are over 15 minutes old, and writes a
  ``tick.log`` line but no ``status.md``, heartbeat, message or error streak.

The tick commits only what it wrote. A file another writer left dirty no longer blocks the tick's
own lines in it: ``tick_commit`` commits HEAD's content plus exactly the tick's lines, and holds
and logs a file whose other edits it cannot separate (``pending_owned`` in the state, tried again
each pass). A dry run writes nothing.

The disk guard holds top-up, queueing and new runner jobs while paused, and the drain while
critical. Crawler is woken only by a FACT: a disk guard entry, a runner that will not stay up,
rows typed ``infra:env_missing`` (``env_missing``) or tick errors two passes in a row (each
condition at most once a day).

Everything the tick keeps is in the work root's ``tick/`` folder, never the repository:
``state.json`` (the typed state), ``tick.log`` (one JSON line and one disk line a pass),
``status.md``, ``heartbeat``, and optional inputs ``mode`` (``collect`` or ``full``; the timer
passes no mode), ``config.json`` (guard lines, spools, lane limits, worklist) and ``runners.json``.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import (
    DataLayout,
    disk_guard,
    disk_reclaim,
    lane_reap,
    mini_relay,
    runner_job,
    tick_commit,
    written,
)

BOUND_S = 8 * 60  # steps stop starting 30 s before this; the alarm fires at it
# top_up starts no lane this close to the pass's step deadline: the last start, the admission
# reads after it and the steps after top_up (trickle, lanes) fit in what is left.
LAUNCH_RESERVE_S = 120
FINISH_S = 520  # state, log and messages are done by then (the unit's TimeoutStartSec is 540)
SEND_S = 30  # at most this long per message
MODES = ("collect", "full")
# ``refill``: the light pass the refill timer runs between full passes while ``REFILL_WANT`` exists
# (a lane ended): only ``disk`` and a 2b-only ``top_up``, finished before the next full pass.
REFILL = "refill"
REFILL_WANT = mini_relay.REFILL_WANT
TICK_PERIOD_S = 600  # menagerie-tick.timer's cadence; a refill ends REFILL_MARGIN_S before its next pass
REFILL_MARGIN_S = 20
REFILL_MIN_S = 75  # a refill with less time than this before the next full pass waits for it
REFILL_RESERVE_S = 30  # a refill has no step after top_up: only the admission reads follow the last start
TRIPWIRES_FRESH_S = 15 * 60  # a refill holds 2b when no full pass checked the tripwires this recently
BIG_SPOOL = "scratch/menagerie/spool-G"
DEFAULTS: dict[str, Any] = {
    "collect_spools": ["scratch/menagerie/spool-B", "scratch/menagerie/spool-Bn",
                       "scratch/menagerie/spool-A"],
    # one collector per spool: while one of these units is active the tick leaves the spools alone
    "exclusive_with": ["menagerie-catalog-collect.timer"],
    "lane_limits": {},  # lowers admission.lane_cap per kind (2b 14, 2a 4 by default); never raises
    "families_per_2a_lane": 8,
    # Speed-up switches (BUILD-PLAN section 1, item 3); these values keep today's behaviour:
    # one family a 2b lane (batches of 4 to 6 after lane A's switch), no per-cohort 2b shares
    # (``{"r4-github": 0.4}`` after lane C's), and no night big-row runner (lane G's).
    "families_per_2b_lane": 1,
    # families 2a keeps ready for 2b whether or not a 2b slot is free this pass;
    # None means two full sets of 2b lanes (2 x the 2b cap x families_per_2b_lane)
    "2b_buffer_families": None,
    "mini_workers": {},  # per-worker cap defaults to 0; positive caps require measured peaks
    # 2b lanes run model code: workers only, hub slots kept for 2a; true restores hub 2b and fallback
    "2b_on_hub": False,
    # true holds the Google seats (the 2b fallback and the third reviewer) while their role is broken
    "third_lab_hold": False,
    "cohort_share": {},
    "big_runner": False,
    "queue_spool": "scratch/menagerie/spool-B",
    "queue_depth": 40,
    "trickle_limit": 20,
    "disk": {},
    "reclaim": {},  # disk_reclaim.DEFAULTS: below 75 GiB free, once an hour, failed trees after 2 h
    "commit_lanes": True,  # false leaves lane results uncommitted (the ``lanes`` step skips)
    "lane_commit_max": 2000,  # lane result files one pass commits at most
    "reap_lanes": True,  # false keeps finished 2b lane checkouts for run-on's own timer
    "axon": {},  # axon_feed.CONFIG: the cluster feeder for rows past G (off unless enabled)
}


class TickTimeout(Exception):
    """The pass reached its bound; the steps not started wait for the next pass."""


def _tick_dir() -> Path:
    return work_dir("tick")


def _iso(when: datetime) -> str:
    return when.astimezone(timezone.utc).isoformat(timespec="seconds")


def load_state() -> dict[str, Any]:
    path = _tick_dir() / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save_state(state: Mapping[str, Any]) -> None:
    path = _tick_dir() / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def config() -> dict[str, Any]:
    path = _tick_dir() / "config.json"
    given = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {**DEFAULTS, **given}


def read_mode() -> str:
    path = _tick_dir() / "mode"
    mode = path.read_text(encoding="utf-8").strip() if path.is_file() else "collect"
    if mode not in MODES:
        raise ValueError(f"{path} says {mode!r}; expected one of {MODES}")
    return mode


@contextlib.contextmanager
def locked(name: str = "tick.lock") -> Iterator[bool]:
    """``flock -n`` on a file in the tick folder: yields whether this process holds it."""

    path = _tick_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# --- the calls a pass makes (tests replace them) ---------------------------------------------


def _repo(data: DataLayout) -> Path:
    top = subprocess.run(["git", "-C", str(data.root), "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=False, timeout=30)
    if top.returncode:
        raise OSError(f"{data.root} is not in a git checkout: {top.stderr.strip()}")
    return Path(top.stdout.strip())


def _unit_active(unit: str) -> bool:
    done = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True,
                          text=True, check=False, timeout=30)
    return done.stdout.strip() in ("active", "activating", "reloading")


def _send(text: str, timeout: float = SEND_S) -> None:
    from menagerie.pipeline.admission import _tool

    done = subprocess.run([_tool("agent-send"), "Crawler", "--project", "menagerie", "--from", "tick",
                           "--tag", "FACT", text], capture_output=True, text=True, check=False,
                          timeout=timeout)
    if done.returncode:
        raise OSError(f"agent-send failed: {done.stderr.strip()[-200:]}")


def _collect_spool(spool: str, data: DataLayout) -> dict[str, Any]:
    from menagerie.pipeline import stage3

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        ids = stage3.collect(data=data, spool=spool)
    lines = [json.loads(x) for x in printed.getvalue().splitlines() if x.startswith("{")]
    total = next((x for x in reversed(lines) if "collected" in x), {})
    return {"spool": spool, "ids": len(ids), "collected": total.get("collected", 0),
            "added": total.get("added", {})}


def _digest(repo: Path, path: str) -> str | None:
    target = repo / path
    return hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None


def _lane_keys(data: DataLayout) -> set[str]:
    """The family and id keys every live or queued lane's lease holds (``admission``)."""

    from menagerie.pipeline import admission

    lanes = [*admission.lanes(data=data), *admission.queued()]
    return {str(k) for x in lanes for k in x.get("keys", ())}


def _finished(spool: str) -> int:
    from menagerie.pipeline import stage3

    return len(stage3.Spool.named(spool).finished())


def _commit(repo: Path, paths: Sequence[str], message: str, deadline: float) -> None:
    from menagerie.pipeline.admission import _tool

    tool = _tool("git-commit-paths")
    while True:
        done = subprocess.run([tool, "-C", str(repo), "-m", message, "--", *paths],
                              capture_output=True, text=True, check=False, timeout=120)
        if done.returncode == 0:
            return
        if "index.lock" not in done.stderr or time.monotonic() + 10 > deadline:
            raise OSError(f"git-commit-paths rc {done.returncode}: {done.stderr.strip()[-300:]}")
        time.sleep(10)  # another session's index lock: wait it out, never remove it


def _drain(data: DataLayout, dry_run: bool) -> dict[str, Any]:
    from menagerie.pipeline import StoreLayout, store

    if not os.environ.get("MENAGERIE_STORE_REMOTE"):
        return {"skipped": "MENAGERIE_STORE_REMOTE is not set"}
    staging = StoreLayout.from_env()
    if dry_run:
        return {"staged_bytes": store.staged_bytes(staging)}
    return {"moved": len(store.drain(store=staging, data=data))}


@dataclass
class Deps:
    """Every outside call the tick makes, so tests can run a pass on fakes."""

    disk_reader: Callable[[], disk_guard.Reading] | None = None
    runner_ops: runner_job.Ops | None = None
    axon_ops: Any = None  # axon_feed.Ops; tests stand in for the cluster and the worker
    unit_active: Callable[[str], bool] = _unit_active
    collect_spool: Callable[[str, DataLayout], dict[str, Any]] = _collect_spool
    finished: Callable[[str], int] = _finished
    changed: Callable[[DataLayout], list[str]] | None = None
    lane_changed: Callable[[DataLayout], list[str]] | None = None  # None: ``changed``
    lane_keys: Callable[[DataLayout], set[str]] = _lane_keys
    commit: Callable[[Path, Sequence[str], str, float], None] = _commit
    commit_owned: Callable[[Path, tick_commit.Owned, str, float], tick_commit.Outcome] = (
        tick_commit.commit_owned)
    repo: Callable[[DataLayout], Path] = _repo
    digest: Callable[[Path, str], str | None] = _digest
    reclaim_launch: disk_reclaim.Launch = disk_reclaim._launch
    # ``lane_held`` comes before ``lane_reap``: from that field on, the name is no longer the module
    lane_held: lane_reap.Held = lane_reap.held_by_run_on
    lane_reap: lane_reap.Reap = lane_reap.run_on_reap
    drain: Callable[[DataLayout, bool], dict[str, Any]] = _drain
    send: Callable[..., None] = _send
    full: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    demand: Callable[[Sequence[str]], dict[str, int]] = runner_job.live_lanes
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)


# --- one pass --------------------------------------------------------------------------------


@dataclass
class Pass:
    """What one pass carries between its steps."""

    mode: str
    dry_run: bool
    data: DataLayout
    deps: Deps
    cfg: dict[str, Any]
    state: dict[str, Any]
    started: datetime
    deadline: float
    guard: disk_guard.State = field(default_factory=disk_guard.State)
    messages: list[tuple[str, str]] = field(default_factory=list)  # (key, text)
    results: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    t0: float = field(default_factory=time.monotonic)
    step_seconds: dict[str, float] = field(default_factory=dict)  # wall time per step, for tick.log


def _step_disk(p: Pass) -> dict[str, Any]:
    gcfg = disk_guard.Config.from_mapping(p.cfg.get("disk"))
    reader = p.deps.disk_reader or disk_guard.ssh_reader(gcfg)
    reading, error = disk_guard.read_once(reader)
    prev = disk_guard.State.from_mapping(p.state.get("disk_guard"))
    p.guard, texts = disk_guard.decide(prev, reading, now=p.started, cfg=gcfg)
    p.messages += [("disk", t) for t in texts]
    p.state["disk_guard"] = p.guard.to_mapping()
    return {"line": disk_guard.log_line(p.guard, gcfg), "read_error": error}


def _runner_refused(state: Mapping[str, Any]) -> bool:
    """Whether some runner's last start was refused for capacity and none has started since."""

    records = (state.get("runners") or {}).values()
    return any(runner_job.refused(runner_job.Record.from_mapping(r)) for r in records)


def _step_reclaim(p: Pass) -> dict[str, Any]:
    """Start the worker's own cleanups when its free disk is under the reclaim line (disk_reclaim)."""

    launch = (lambda *_: 0) if p.dry_run else p.deps.reclaim_launch
    new, result = disk_reclaim.step(
        p.state.get("disk_reclaim"), p.guard.free_gib, now=p.started, cfg=p.cfg.get("reclaim"),
        repo=p.deps.repo(p.data), log=work_dir("tick") / "reclaim.log", launch=launch)
    if not p.dry_run:
        p.state["disk_reclaim"] = new
    return result


def _step_runners(p: Pass) -> list[dict[str, Any]]:
    """``runner ensure`` for this pass's runners. A refill pass runs it as the timer's pass would,
    and only while a runner's last start was refused for capacity (``_runner_refused``), so a
    refused runner is retried on every pass and is not beaten to the room by jobs started between
    full passes."""

    mode = p.mode
    if mode == REFILL:
        if not _runner_refused(p.state):
            return []
        mode = read_mode()
    kinds = sorted({r.demand for r in runner_job.runners() if r.demand and mode in r.modes})
    ops = p.deps.runner_ops or runner_job.RunOn(ended=work_dir("tick") / runner_job.ENDED_JOBS)
    out = runner_job.ensure(dry_run=p.dry_run, ops=ops, state=p.state,
                            hold_submit=p.guard.held("runner_submit"),
                            pause_gib=p.guard.pause_below_gib, now=p.started, mode=mode,
                            demand=p.deps.demand(kinds))
    p.state["runners"] = {**p.state.get("runners", {}), **{r["runner"]: r.pop("record") for r in out}}
    for r in out:
        p.messages += [(f"runner-{r['runner']}", t) for t in r.pop("messages")]
    return out


def _changed(p: Pass) -> set[str]:
    from menagerie.pipeline import drive

    lister = p.deps.changed or (lambda data: drive.collect(data=data, commit=False))
    return set(lister(p.data))


def _spool_names(cfg: Mapping[str, Any]) -> list[str]:
    """The spools the tick collects: ``collect_spools``, plus spool-G while the big runner is on
    (its results are collected by day too)."""
    names = list(cfg["collect_spools"])
    return [*names, BIG_SPOOL] if cfg.get("big_runner") and BIG_SPOOL not in names else names


def _collect_spools(p: Pass) -> list[dict[str, Any]]:
    """Collect each spool under its own lock: one collector per spool."""

    spools = []
    for spool in _spool_names(p.cfg):
        name = spool.rstrip("/").rsplit("/", 1)[-1]
        with locked(f"collect-{name}.lock") as mine:
            spools.append(p.deps.collect_spool(spool, p.data) if mine
                          else {"spool": spool, "skipped": "another collector holds its lock"})
    return spools


def _commit_ours(p: Pass, repo: Path, ours: Sequence[str]) -> None:
    """Commit this pass's (and an earlier failed pass's) files by name.

    On failure each file's content hash is kept: the next pass commits it only if nobody has
    changed it since (else it is another writer's now, and skipped).
    """

    if ours:
        message = f"data: tick collected stage 3 rows ({len(ours)} ledger and store-index files)"
        try:
            p.deps.commit(repo, ours, message, p.deadline)
        except OSError:
            p.state["pending_commit"] = {x: p.deps.digest(repo, x) for x in ours}
            raise
    p.state["pending_commit"] = {}


def _shards(p: Pass, repo: Path, spools: Sequence[Mapping[str, Any]]) -> set[str]:
    """The data files collect appended to, from its ``added`` counts (``ledger/2026-10`` ...)."""

    rel = Path(p.data.root).resolve().relative_to(repo.resolve()).as_posix()
    return {f"{rel}/{shard}.jsonl" for r in spools
            for shard, n in (r.get("added") or {}).items() if n}


def _in_data(p: Pass, repo: Path) -> Callable[[str], bool]:
    rel = Path(p.data.root).resolve().relative_to(repo.resolve()).as_posix()
    return lambda path: path.startswith(f"{rel}/")


@contextlib.contextmanager
def _owning(p: Pass, repo: Path, keep: Callable[[str], bool], *,
            lines_only: bool = False) -> Iterator[written.Writes]:
    """Record what the block writes inside the data checkout and, however the block ends, add it
    to the state's ``pending_owned`` (what the tick still has to commit)."""

    with written.recording() as rec:
        try:
            yield rec
        finally:
            mine = tick_commit.Owned.from_writes(rec, repo, keep=keep)
            mine = tick_commit.Owned(mine.lines, {}) if lines_only else mine
            held = tick_commit.Owned.from_mapping(p.state.get("pending_owned"))
            p.state["pending_owned"] = held.merged(mine).to_mapping()


def _commit_owned(p: Pass, repo: Path, message: str) -> dict[str, Any]:
    """Commit what the state's ``pending_owned`` holds, exactly (``tick_commit``).

    What is held stays in the state for the next pass; on a failure everything does.
    """

    todo = tick_commit.Owned.from_mapping(p.state.get("pending_owned"))
    if not todo:
        return {}
    out = p.deps.commit_owned(repo, todo, message, p.deadline)
    p.state["pending_owned"] = out.pending.to_mapping()
    return out.to_json()


def _step_collect(p: Pass) -> dict[str, Any]:
    """Collect each spool, then commit exactly the lines collect appended (``tick_commit``), so
    another writer's lines in the same shard, from before or during the pass, stay uncommitted.

    A shard whose lines were not recorded (a stand-in collector) falls back to a whole-file commit
    by name when collect alone dirtied it, and is skipped when another writer had; so is a file an
    earlier failed pass left whose content has changed since.
    """

    owners = [u for u in p.cfg["exclusive_with"] if p.deps.unit_active(u)]
    owned = f"another collector owns the spools: {', '.join(owners)} is active" if owners else None
    if p.dry_run:
        return {"would_collect": {s: p.deps.finished(s) for s in _spool_names(p.cfg)},
                "dirty_now": sorted(_changed(p))[:20], "skipped": owned}
    if owned:
        return {"skipped": owned}
    repo = p.deps.repo(p.data)
    pending = p.state.get("pending_commit") or {}
    pending = pending if isinstance(pending, dict) else {}
    kept = {x for x, sha in pending.items() if sha and p.deps.digest(repo, x) == sha}
    foreign = _changed(p) - kept  # dirty before this pass and not left by an earlier one
    with _owning(p, repo, keep=_shard_path(p, repo), lines_only=True) as rec:
        spools = _collect_spools(p)
    touched = _shards(p, repo, spools)
    recorded = set(tick_commit.Owned.from_writes(rec, repo, keep=_shard_path(p, repo)).lines)
    after = _changed(p)
    ours = sorted((((touched - foreign) | kept) & after) - recorded)
    skipped = sorted(((touched & foreign) - recorded) | (set(pending) - kept))
    _commit_ours(p, repo, ours)
    lines = _commit_owned(p, repo, f"data: tick collected stage 3 rows ({len(touched)} ledger and "
                          "store-index files, its own lines only)")
    return {"spools": spools, "commit": ours, "skipped_foreign_dirty": skipped,
            **({"own_lines": lines} if lines else {})}


def _shard_path(p: Pass, repo: Path) -> Callable[[str], bool]:
    rel = Path(p.data.root).resolve().relative_to(repo.resolve()).as_posix()
    return lambda path: path.startswith((f"{rel}/ledger/", f"{rel}/store-index/"))


def _step_drain(p: Pass) -> dict[str, Any]:
    if p.guard.held("drain"):
        return {"held": f"disk {p.guard.state}"}
    return p.deps.drain(p.data, p.dry_run)


def _full_call(p: Pass, name: str, held: str | None) -> Any:
    if name in ("top_up", "stage3_queue") and "tripwires" in p.errors:
        return {"held": "cohort tripwires could not be checked"}
    if held and p.guard.held(held):
        return {"held": f"disk {p.guard.state}"}
    from menagerie.pipeline import tick_full

    call = p.deps.full.get(name) or getattr(tick_full, name)
    return call(p.data, p.cfg, dry_run=p.dry_run)


def _tries_summary(p: Pass) -> dict[str, Any] | None:
    """Runner A's summary: from this pass's runners step, or in a refill (which leaves the runners
    to the full pass) from its saved record and its job's live state; a pending resize or drain
    reads as ``resize``, so the refill holds 2b as the full pass would."""

    if p.mode != REFILL:
        return next((r for r in p.results.get("runners") or []
                     if r.get("runner") == runner_job.TRIES_RUNNER), None)
    record = (p.state.get("runners") or {}).get(runner_job.TRIES_RUNNER) or {}
    if not record.get("job"):
        return None
    ops = p.deps.runner_ops or runner_job.RunOn(ended=work_dir("tick") / runner_job.ENDED_JOBS)
    pending = record.get("resize_to") is not None or record.get("draining")
    return {"runner": runner_job.TRIES_RUNNER, "job": record["job"], "size": record.get("size"),
            "job_state": ops.job_state(record["job"]), "action": "resize" if pending else None}


def _tripwires_stale(p: Pass) -> bool:
    """Whether no full pass has checked the cohort tripwires in the last ``TRIPWIRES_FRESH_S``."""
    at = p.state.get("tripwires_at")
    return not at or (p.started - datetime.fromisoformat(at)).total_seconds() > TRIPWIRES_FRESH_S


def _step_top_up(p: Pass) -> Any:
    """``top_up`` with the 2b lanes' declaration (``runner_job.tries_declaration``) from this
    pass's runners step; with none, top_up holds 2b launches and says why.

    A full pass clears ``REFILL_WANT`` first: the lanes that ended before it are its to refill.
    A refill tops up 2b only (2a capped at zero), with the last full pass's cohort verdicts
    (``cohort_states``), and holds when that pass could not check them."""

    if p.mode == REFILL and _tripwires_stale(p):
        return {"held": "no full pass checked the cohort tripwires in the last 15 minutes"}
    if not p.dry_run:
        (_tick_dir() / REFILL_WANT).unlink(missing_ok=True)
    env, why = runner_job.tries_declaration(_tries_summary(p), p.deps.runner_ops)
    p.cfg["runner_declaration"] = env or {"held": why}
    p.cfg["launch_by"] = p.deadline - (REFILL_RESERVE_S if p.mode == REFILL else LAUNCH_RESERVE_S)
    if p.mode == REFILL:
        p.cfg["lane_limits"] = {**(p.cfg.get("lane_limits") or {}), "2a": 0}
    # Worker lanes on the trace host follow this pass's reading; a failed disk step is unknown.
    p.cfg["disk_state"] = "unknown" if "disk" in p.errors else p.guard.state
    return _full_call(p, "top_up", "top_up")


def _step_tripwires(p: Pass) -> Any:
    from menagerie.pipeline import tick_full

    call = p.deps.full.get("tripwires")
    if call is None and not p.cfg.get("tripwires", False):
        p.cfg["stopped_cohorts"] = [c for c, v in p.state.get("cohort_states", {}).items()
                                    if v.get("stopped")]
        p.state["tripwires_at"] = _iso(p.started)
        return {"skipped": "cohort tripwires disabled"}
    result = (call(p.data, p.cfg, dry_run=p.dry_run) if call else
              tick_full.tripwires(p.data, p.cfg, dry_run=p.dry_run, state=p.state))
    p.cfg["stopped_cohorts"] = result.get("stopped", [])
    p.state["tripwires_at"] = _iso(p.started)  # what a refill's top_up reads (_tripwires_stale)
    p.messages += [(f"cohort-{c}", text) for c, text in result.get("facts", {}).items()]
    return result


def _step_trickle(p: Pass) -> Any:
    """``paper harvest --limit``, then commit exactly what it wrote: packets, citation lines and
    store-index lines. What a harvest stopped partway wrote is kept for the next commit."""

    if p.dry_run:
        return _full_call(p, "trickle", None)
    repo = p.deps.repo(p.data)
    with _owning(p, repo, keep=_in_data(p, repo)):
        harvested = _full_call(p, "trickle", None)
    rows = harvested if isinstance(harvested, list) else []
    message = (f"data: tick paper harvest trickle ({len(rows)} packets tried; packets, citation "
               "and store-index lines it wrote)")
    commit = _commit_owned(p, repo, message)
    return {"harvest": harvested, **({"commit": commit} if commit else {})}


def _latest_traces(ledger_dir: Path, since: str) -> dict[str, dict[str, Any]]:
    """Each id's latest trace row at or after ``since`` (UTC, to the second) in the last two
    shards. Only lines naming a trace are parsed: ``ledger.read_all`` would parse every row of
    every shard each pass."""

    latest: dict[str, dict[str, Any]] = {}
    for shard in sorted(ledger_dir.glob("*.jsonl"))[-2:]:
        for line in shard.read_text(encoding="utf-8").splitlines():
            row = json.loads(line) if '"trace"' in line else {}
            at = str(row.get("at", ""))[:19]
            if row.get("stage") == "trace" and at >= since \
                    and at >= str(latest.get(row["id"], {}).get("at", ""))[:19]:
                latest[row["id"]] = row
    return latest


def _step_env_missing(p: Pass) -> dict[str, list[str]]:
    """Ids whose latest trace row of the last day is ``infra:env_missing``, by environment: a lock
    committed on main that the worker has not built. One FACT a day per set of environments."""

    since = (p.started - timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    found: dict[str, list[str]] = {}
    for model_id, row in sorted(_latest_traces(p.data.root / "ledger", since).items()):
        if row.get("class") == "infra:env_missing":
            found.setdefault(str(row.get("env")), []).append(model_id)
    if found:
        listed = ", ".join(f"{env} ({len(ids)} ids)" for env, ids in found.items())
        text = (f"latest trace found no built environment: {listed}. Runner jobs build missing "
                "environments after they start (outputs/envs.log); full mode queues worklist "
                "rows again, other rows need `stage3 queue --committed`.")
        p.messages.append(("env-missing", text))
    return found


def _step_stage3_queue(p: Pass) -> Any:
    """``stage3_queue``; with the big runner on, it also keeps ``big_work`` (rows reserved for G
    plus spool-G's waiting jobs) in the state, which decides whether G runs tonight."""
    out = _full_call(p, "stage3_queue", "stage3_queue")
    big = out.get("big") if isinstance(out, dict) else None
    if isinstance(big, dict) and "held" not in big and "reserved" in big:
        p.state["big_work"] = len(big["reserved"]) + int(big.get("waiting") or 0)
    return out


def _step_lanes(p: Pass) -> dict[str, Any]:
    """Commit the lane results no live or queued lane holds (``lane_commit``); a file left
    unreadable or refused by the commit hooks is one FACT a day."""

    if not p.cfg.get("commit_lanes", True):
        return {"skipped": "commit_lanes is off"}
    from menagerie.pipeline import lane_commit

    repo = p.deps.repo(p.data)
    rel = Path(p.data.root).resolve().relative_to(repo.resolve()).as_posix()
    changed = p.deps.lane_changed(p.data) if p.deps.lane_changed else _changed(p)
    out = lane_commit.step(repo, rel, changed, p.deps.lane_keys(p.data), p.state,
                           commit=p.deps.commit, digest=p.deps.digest, deadline=p.deadline,
                           dry_run=p.dry_run, limit=int(p.cfg["lane_commit_max"]))
    stuck = len(out.get("unreadable", {})) + len(out.get("refused", []))
    if stuck and not p.dry_run:
        p.messages.append(("lane-commit", f"{stuck} lane result files stay uncommitted: they do "
                           "not read whole or the commit hooks refused them (tick status.md, "
                           "lanes step)."))
    return out


def _step_lane_reap(p: Pass) -> dict[str, Any]:
    """Reap the run-on checkouts of 2b lanes whose outputs the hub has ingested, whose lane
    results the ``lanes`` step has committed and which run-on holds for its failed grace
    (``lane_reap``)."""

    if not p.cfg.get("reap_lanes", True):
        return {"skipped": "reap_lanes is off"}
    from menagerie.pipeline import mini_relay

    repo = p.deps.repo(p.data)
    rel = Path(p.data.root).resolve().relative_to(repo.resolve()).as_posix()
    changed = p.deps.lane_changed(p.data) if p.deps.lane_changed else _changed(p)
    return lane_reap.step(mini_relay.state_dir(), work_dir("leases"), work_dir("2b"),
                          lane_reap.uncommitted_keys(repo, rel, changed), now=p.started,
                          reap=p.deps.lane_reap, held=p.deps.lane_held(), deadline=p.deadline,
                          dry_run=p.dry_run)


def _step_axon(p: Pass) -> dict[str, Any]:
    """The axon feeder (``axon_feed``): collect the cluster batch and its renders in any mode,
    submit the next only in a full pass with the disk guard clear; commit exactly the lines it
    appended (``tick_commit``)."""

    from menagerie.pipeline import axon_feed

    cfg, state = axon_feed.config(p.cfg), p.state.get("axon") or {}
    if not cfg["enabled"] and not state.get("active") and not state.get("renders"):
        return {"skipped": "the axon feeder is off"}
    if p.dry_run:
        return {"active": state.get("active"), "renders": state.get("renders"),
                "ready": state.get("ready")}
    repo = p.deps.repo(p.data)
    held = f"disk {p.guard.state}" if p.guard.held("cluster") else \
        "" if p.mode == "full" else f"{p.mode} mode"
    feed = axon_feed.Feed(state, p.data, cfg, p.deps.axon_ops or axon_feed.Ops(), repo,
                          submit=not held, deadline=p.deadline, held=held)
    try:
        with _owning(p, repo, keep=_shard_path(p, repo), lines_only=True):
            out = axon_feed.step(feed)
    finally:
        p.state["axon"] = feed.state
    lines = _commit_owned(p, repo, "data: tick collected axon rows (ledger and store-index "
                          "lines, its own lines only)")
    return {**out, **({"own_lines": lines} if lines else {})}


STEPS: tuple[tuple[str, tuple[str, ...], Callable[[Pass], Any]], ...] = (
    ("disk", (*MODES, REFILL), _step_disk),
    ("reclaim", MODES, _step_reclaim),
    ("runners", (*MODES, REFILL), _step_runners),
    ("collect", MODES, _step_collect),
    ("drain", MODES, _step_drain),
    ("env_missing", MODES, _step_env_missing),
    ("tripwires", ("full",), _step_tripwires),
    ("requeue", ("full",), lambda p: _full_call(p, "requeue", "top_up")),
    ("stage3_queue", ("full",), _step_stage3_queue),
    ("top_up", ("full", REFILL), _step_top_up),
    ("trickle", ("full",), _step_trickle),
    ("reviews", ("full",), lambda p: _full_call(p, "reviews", None)),
    ("describes", ("full",), lambda p: _full_call(p, "describes", None)),
    ("lanes", MODES, _step_lanes),
    ("lane_reap", MODES, _step_lane_reap),
    ("axon", MODES, _step_axon),
)


def _run_steps(p: Pass) -> None:
    for name, modes, step in STEPS:
        if p.mode not in modes:
            continue
        if time.monotonic() > p.deadline:
            p.results[name] = {"skipped": "the pass reached its bound"}
            continue
        began = time.monotonic()
        try:
            p.results[name] = step(p)
        except NotImplementedError as exc:
            p.results[name] = {"waiting": str(exc)}
        except TickTimeout:
            p.errors[name] = "the pass reached its bound inside this step"
            raise
        except Exception as exc:  # noqa: BLE001 -- one step's failure never stops the others
            p.errors[name] = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            p.step_seconds[name] = round(time.monotonic() - began, 1)


def _outbox(p: Pass) -> list[list[str]]:
    """This pass's messages plus any an earlier pass could not send, each (key, text) once a day.

    Errors two passes in a row add one message a day. The outbox is saved with the state before
    anything is sent, so a pass killed while sending loses nothing.
    """

    # a refill's errors are in tick.log; the streak counts the timer's passes only
    streak = p.state.get("error_streak", 0)
    if p.mode != REFILL:
        streak = streak + 1 if p.errors else 0
    p.state["error_streak"] = streak
    today = p.started.date().isoformat()
    if streak >= 2 and p.state.get("errors_escalated_on") != today:
        steps = ", ".join(f"{k} ({v[:120]})" for k, v in sorted(p.errors.items()))
        text = f"tick errors {streak} passes in a row: {steps}; log {_tick_dir() / 'tick.log'}"
        p.messages.append(("errors", text))
        p.state["errors_escalated_on"] = today
    seen = {k: v for k, v in (p.state.get("sent_on") or {}).items() if v == today}
    box = [list(m) for m in p.state.get("outbox", [])]
    for key, text in p.messages:
        mark = f"{key}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        if mark not in seen and [key, text] not in box:
            box.append([key, text])
            seen[mark] = today
    p.state["sent_on"] = seen
    p.state["outbox"] = box
    return box


def _send_outbox(p: Pass, finish_by: float) -> list[str]:
    """Send what fits before ``finish_by``; what does not stays in the outbox for the next pass."""

    sent, left = [], []
    for key, text in p.state.get("outbox", []):
        room = finish_by - time.monotonic()
        if room < 5:
            left.append([key, text])
            continue
        try:
            p.deps.send(text, timeout=min(SEND_S, room))
            sent.append(key)
        except (OSError, subprocess.TimeoutExpired) as exc:
            p.errors[f"send-{key}"] = str(exc)[:200]
            left.append([key, text])
    p.state["outbox"] = left
    return sent


def _where_2b(p: Pass) -> list[str]:
    """``status.md``'s line on where this pass's 2b lanes went and why (none when top_up held)."""
    top = p.results.get("top_up")
    return [f"- 2b: {top['2b_where']}"] if isinstance(top, dict) and top.get("2b_where") else []


def _status_md(p: Pass) -> str:
    gline = p.results.get("disk", {}).get("line", "disk: not read")
    rows = [f"# Menagerie tick: {_iso(p.started)} ({p.mode}{', dry run' if p.dry_run else ''})", "",
            f"- {gline}", *_where_2b(p)]
    for r in p.results.get("runners", []) or []:
        rows.append(f"- runner {r['runner']}: {r['action']} ({r['job']}, {r['job_state']}, "
                    f"heartbeat {r['heartbeat_age_s']} s) {r['detail']}")
    for name, value in p.results.items():
        if name not in ("disk", "runners"):
            rows.append(f"- {name}: {json.dumps(value, default=str)[:600]}")
    for cohort, counts in p.results.get("tripwires", {}).get("cohorts", {}).items():
        stopped = cohort in p.results["tripwires"].get("stopped", [])
        rows.append(f"- cohort {cohort}: {counts['validated']}/{counts['settled']} validated/settled "
                    f"({counts.get('retired', 0)} retired), "
                    f"{counts['research_gap']}/{counts['completed_2a']} research gaps/completed 2a, "
                    f"{'stopped' if stopped else 'open'}")
    rows += [f"- ERROR {k}: {v}" for k, v in sorted(p.errors.items())]
    return "\n".join(rows) + "\n"


def _write_out(p: Pass, sent: Sequence[str]) -> None:
    folder = _tick_dir()
    folder.mkdir(parents=True, exist_ok=True)
    save_state(p.state)
    if _runner_refused(p.state):  # the refill timer's next pass retries the refused runner
        (folder / REFILL_WANT).touch()
    line = {"at": _iso(p.started), "mode": p.mode, "seconds": round(time.monotonic() - p.t0, 1),
            "step_seconds": p.step_seconds, "results": p.results, "errors": p.errors, "sent": list(sent),
            "unsent": len(p.state.get("outbox", []))}
    with open(folder / "tick.log", "a", encoding="utf-8") as log:
        log.write(json.dumps(line, default=str) + "\n")
        log.write(p.results.get("disk", {}).get("line", "disk: not read") + "\n")
    if p.mode == REFILL:  # status.md and the heartbeat describe the timer's passes
        return
    tmp = folder / ".status.md.tmp"
    tmp.write_text(_status_md(p), encoding="utf-8")
    os.replace(tmp, folder / "status.md")
    (folder / "heartbeat").write_text(_iso(p.deps.now()) + "\n", encoding="utf-8")


@contextlib.contextmanager
def _bound(seconds: float | None) -> Iterator[None]:
    """Raise ``TickTimeout`` in the main thread after ``seconds`` (a subprocess wait included)."""

    if not seconds or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(*_: Any) -> None:
        raise TickTimeout(f"tick bound of {seconds:.0f} s reached")

    signal.signal(signal.SIGALRM, _raise)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)


def _finish(p: Pass, bound_s: float | None) -> list[str]:
    """Save the state and outbox, send what fits (a refill's messages wait for the next full
    pass), then write the log line and, for the timer's passes, ``status.md``; returns what went."""

    _outbox(p)
    save_state(p.state)  # runner records and the outbox are safe before any message goes
    sent: list[str] = []
    if p.mode != REFILL:
        sent = _send_outbox(p, p.t0 + (FINISH_S if bound_s else BOUND_S + FINISH_S))
        p.results["status"] = {"written": str(_tick_dir() / "status.md")}
    _write_out(p, sent)
    return sent


def _refill_room(now: datetime) -> float:
    """Seconds a refill may run: until ``REFILL_MARGIN_S`` before the timer's next full pass."""
    return TICK_PERIOD_S - now.timestamp() % TICK_PERIOD_S - REFILL_MARGIN_S


def tick(mode: str | None = None, *, dry_run: bool = False, data: DataLayout,
         deps: Deps | None = None, bound_s: float | None = BOUND_S) -> dict[str, Any]:
    """Run one tick pass; return what it did (or would do, with ``dry_run``).

    Inputs: the mode (``None`` reads the tick folder's ``mode`` file, else ``collect``; ``refill``
    is the refill timer's light pass, bounded to end before the next full pass), the data
    checkout and, for tests, the outside calls. A dry run reads the disk, the runner jobs and the
    spools and changes nothing: no lock, state, log, commit, job or message. Returns: the step
    results, errors and messages. Side effects (not dry): see the module docstring.
    """

    mode = mode or read_mode()
    if mode not in (*MODES, REFILL):
        raise ValueError(f"mode {mode!r}; expected one of {(*MODES, REFILL)}")
    deps = deps or Deps()
    room = _refill_room(deps.now()) if mode == REFILL else BOUND_S
    if room < REFILL_MIN_S:
        return {"mode": mode, "skipped": f"the next full pass is {room + REFILL_MARGIN_S:.0f} s away"}
    bound_s = min(bound_s, room) if bound_s else None
    cfg = config()
    disk_guard.Config.from_mapping(cfg.get("disk"))  # refuse to start on misordered lines
    with contextlib.ExitStack() as stack:
        if not dry_run and not stack.enter_context(locked()):
            return {"mode": mode, "skipped": "another tick holds the lock"}
        state = load_state()
        started = deps.now()
        budget = bound_s or room
        p = Pass(mode, dry_run, data, deps, cfg, state, started, time.monotonic() + budget - 30)
        try:
            with _bound(None if dry_run else bound_s):
                _run_steps(p)
        except TickTimeout as exc:
            p.errors.setdefault("bound", str(exc))
        if dry_run:
            return {"mode": mode, "dry_run": True, "results": p.results, "errors": p.errors,
                    "would_send": [t for _, t in p.messages]}
        sent = _finish(p, bound_s)
        return {"mode": mode, "results": p.results, "errors": p.errors, "sent": sent}

