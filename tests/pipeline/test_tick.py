"""Lane r3-tick: the disk guard, ``runner ensure`` and one tick pass, all on fakes and temp folders.

No test reaches the worker, run-on, git-commit-paths or agent-send: the runner jobs are fake
states, the spools are temporary folders and every message is captured.
"""

from __future__ import annotations

import fcntl
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.pipeline import (
    DataLayout, append_jsonl, cli, disk_guard, runner_job, stage3, tick, tick_commit, tick_full,
    write_json,
)
from menagerie.pipeline.records import StoreIndexLine, dumps
from menagerie.pipeline.disk_guard import GIB, Config, Reading, State, decide

T0 = datetime(2026, 10, 9, 18, 0, tzinfo=timezone.utc)
TOTAL = 400 * GIB  # 10 percent is 40 GiB, under the 45.7 floor


def _gib(free: float, total: int = TOTAL) -> Reading:
    return Reading(int(free * GIB), total)


def _run(readings: list[Reading | None], *, start: State | None = None,
         step: timedelta = timedelta(minutes=10)) -> tuple[list[State], list[list[str]]]:
    state, states, sent = start or State(), [], []
    for n, reading in enumerate(readings):
        state, msgs = decide(state, reading, now=T0 + n * step, cfg=Config())
        states.append(state)
        sent.append(msgs)
    return states, sent


# --- the disk guard ----------------------------------------------------------------------------


def test_well_above_the_line_everything_runs() -> None:
    states, sent = _run([_gib(88.4)])
    assert states[0].state == "ok" and not sent[0]
    assert not any(states[0].held(s) for s in ("top_up", "stage3_queue", "runner_submit", "drain"))
    assert (states[0].floor_gib, states[0].pause_below_gib, states[0].resume_at_gib) == (45.7, 62, 72)


def test_crossing_down_pauses_with_one_fact_and_five_low_ticks_send_no_second() -> None:
    states, sent = _run([_gib(70), _gib(61.9), *[_gib(58)] * 5])
    assert [s.state for s in states] == ["ok"] + ["paused"] * 6
    assert len(sent[1]) == 1 and "disk paused: free 61.9 GiB < 62" in sent[1][0]
    assert not any(sent[2:])
    held = states[-1]
    assert held.held("top_up") and held.held("runner_submit") and not held.held("drain")


def test_hysteresis_holds_between_pause_and_resume_then_resumes_at_72_without_a_fact() -> None:
    states, sent = _run([_gib(60), _gib(65), _gib(71.9), _gib(72)])
    assert [s.state for s in states] == ["paused", "paused", "paused", "ok"]
    assert [len(m) for m in sent] == [1, 0, 0, 0]
    assert states[-1].fact_sent_since is None


def test_critical_holds_the_drain_and_escalates_once_a_day() -> None:
    states, sent = _run([_gib(49), _gib(48), _gib(48)], step=timedelta(hours=2))
    assert all(s.state == "critical" and s.held("drain") for s in states)
    assert len(sent[0]) == 1 and "critical" in sent[0][0] and "STOP" in sent[0][0]
    assert sent[1:] == [[], []]  # 18:00, 20:00 and 22:00 UTC: one day
    _, more = _run([_gib(48)], start=states[-1], step=timedelta(0))
    assert more == [[]]
    next_day, msgs = decide(states[-1], _gib(48), now=T0 + timedelta(days=1), cfg=Config())
    assert len(msgs) == 1 and next_day.escalated_on == "2026-10-10"
    eased, quiet = decide(next_day, _gib(55), now=T0 + timedelta(days=1, hours=1), cfg=Config())
    assert eased.state == "paused" and quiet == []  # easing off sends nothing


def test_a_failed_reader_keeps_the_last_state_then_goes_unknown_with_one_fact() -> None:
    first, _ = decide(State(), _gib(80), now=T0, cfg=Config())
    states, sent = _run([None, None, None, None], start=first, step=timedelta(minutes=10))
    # reads at +0, +10, +20, +30 min after the last good one at T0: held until 30 minutes pass
    assert [s.state for s in states] == ["ok", "ok", "ok", "ok"] and not any(sent)
    late, msgs = decide(states[-1], None, now=T0 + timedelta(minutes=31), cfg=Config())
    assert late.state == "unknown" and late.held("stage3_queue") and len(msgs) == 1
    again, quiet = decide(late, None, now=T0 + timedelta(minutes=41), cfg=Config())
    assert again.state == "unknown" and quiet == [] and again.reader_failures == 6


def test_a_restart_while_paused_sends_no_repeated_fact() -> None:
    paused, sent = decide(State(), _gib(55), now=T0, cfg=Config())
    assert len(sent) == 1
    reloaded = State.from_mapping(json.loads(json.dumps(paused.to_mapping())))
    again, quiet = decide(reloaded, _gib(55), now=T0 + timedelta(minutes=10), cfg=Config())
    assert again.state == "paused" and quiet == []


def test_misordered_lines_are_refused() -> None:
    with pytest.raises(disk_guard.GuardConfigError):
        Config.from_mapping({"pause_gib": 49})
    with pytest.raises(disk_guard.GuardConfigError):
        Config.from_mapping({"resume_gib": 60})


def test_ten_percent_of_a_large_disk_raises_every_line() -> None:
    big = 1000 * GIB  # 10 percent is 100 GiB, above 45.7
    state, _ = decide(State(), _gib(150, big), now=T0, cfg=Config())
    assert state.floor_gib == 100.0 and state.pause_below_gib == pytest.approx(116.3)
    assert state.resume_at_gib == pytest.approx(126.3) and state.state == "ok"
    low, msgs = decide(state, _gib(110, big), now=T0, cfg=Config())
    assert low.state == "paused" and msgs


def test_the_worker_claim_check_idles_below_the_job_pause_line(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(free: float):
        return lambda _: SimpleNamespace(f_bavail=int(free * GIB) // 4096, f_frsize=4096)

    monkeypatch.delenv(disk_guard.PAUSE_ENV, raising=False)
    assert disk_guard.claim_ok("/x", statvfs=fake(1))  # a job started before the guard
    monkeypatch.setenv(disk_guard.PAUSE_ENV, "62")
    assert not disk_guard.claim_ok("/x", statvfs=fake(61.5))
    assert disk_guard.claim_ok("/x", statvfs=fake(62.5))


def test_log_line_matches_the_store_gate_format() -> None:
    state, _ = decide(State(), _gib(58.3), now=T0, cfg=Config())
    line = disk_guard.log_line(state, Config())
    assert line.startswith("disk worker free=58.3GiB floor=45.7 pause<62 resume>=72 state=paused")


# --- runner ensure -----------------------------------------------------------------------------


@dataclass
class FakeOps:
    """Scripted job states and heartbeats; records every submit and cancel."""

    states: dict[str, str] = field(default_factory=dict)
    age: float | None = 5.0
    stop: bool = False
    is_idle: bool = True
    refuse_above: int | None = None  # run-on refuses jobs with more workers than this
    found: list[str] = field(default_factory=list)
    started: dict[str, str] = field(default_factory=dict)
    ambiguous: bool = False  # submit fails with no job id and no refusal
    new_state: str = "RUNNING"  # what a submitted job reads as on the next pass
    submitted: list[Mapping[str, str]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    drains: list[bool] = field(default_factory=list)

    def job_state(self, job: str) -> str:
        return self.states.get(job, "MISSING")

    def heartbeat_age(self, runner: runner_job.Runner) -> float | None:
        return self.age

    def stopped(self, runner: runner_job.Runner) -> bool:
        return self.stop

    def idle(self, runner: runner_job.Runner) -> bool:
        return self.is_idle

    def drain(self, runner: runner_job.Runner, on: bool) -> None:
        self.drains.append(on)

    def discover(self, runner: runner_job.Runner) -> list[tuple[str, str | None]]:
        return [(j, self.started.get(j)) for j in self.found]

    def submit(self, runner: runner_job.Runner, env: Mapping[str, str]) -> str:
        if self.ambiguous:
            self.submitted.append({"ambiguous": "1"})
            raise OSError("run-on submit: rc 255, no job id: connection closed")
        if self.refuse_above is not None and runner.workers > self.refuse_above:
            raise runner_job.SubmitRefused(f"refused: NO_CAPACITY at {runner.workers} workers")
        self.submitted.append(dict(env))
        job = f"menagerie-new-{len(self.submitted)}"
        self.states[job] = self.new_state
        return job

    def cancel(self, job: str) -> None:
        self.cancelled.append(job)
        self.states[job] = "INTERRUPTED"


B = runner_job.DEFAULT_RUNNERS[0]
A = runner_job.Runner("A", "scratch/menagerie/spool-A", cpus=3, memory_gib=20, workers=1,
                      sizes=((1, 3, 20), (2, 6, 36), (3, 9, 54)), demand="2b")


def _ensure(ops: FakeOps, record: runner_job.Record | None = None, *, now: datetime = T0,
            hold: bool = False, dry_run: bool = False, runner: runner_job.Runner = B,
            demand: int = 0) -> runner_job.Outcome:
    return runner_job.ensure_one(runner, record or runner_job.Record(), ops, now=now,
                                 hold_submit=hold, pause_gib=62, demand=demand, dry_run=dry_run)


def test_the_live_catalog_runner_is_adopted_never_replaced() -> None:
    ops = FakeOps(states={"menagerie-b": "RUNNING", "menagerie-old": "FAILED"},
                  found=["menagerie-b", "menagerie-old"])
    out = _ensure(ops)
    assert (out.action, out.record.job) == ("ok", "menagerie-b") and "adopted" in out.detail
    assert not ops.submitted and not ops.cancelled


def test_a_stale_running_job_is_cancelled_then_restarted_on_the_next_pass() -> None:
    ops = FakeOps(states={"menagerie-b": "RUNNING"}, age=20 * 60)
    first = _ensure(ops, runner_job.Record(job="menagerie-b"))
    assert first.action == "cancel-stale" and ops.cancelled == ["menagerie-b"] and not ops.submitted
    second = _ensure(ops, first.record, now=T0 + timedelta(minutes=10))
    assert second.action == "restart" and second.record.job == "menagerie-new-1"
    assert ops.submitted[0]["RUNNER_SPOOL"] == "scratch/menagerie/spool-B"
    assert ops.submitted[0]["MENAGERIE_DISK_PAUSE_GIB"] == "62"
    assert ops.submitted[0]["MENAGERIE_TICK_RUNNER"] == "B" and len(second.record.restarts) == 1


def test_a_young_job_without_a_heartbeat_is_left_to_start() -> None:
    ops = FakeOps(states={"j": "RUNNING"}, age=None)
    out = _ensure(ops, runner_job.Record(job="j", submitted_at=(T0 - timedelta(minutes=5)).isoformat()))
    assert out.action == "ok" and not ops.cancelled


@pytest.mark.parametrize("dead", ["FAILED", "INTERRUPTED", "MISSING"])
def test_a_dead_job_with_a_fresh_heartbeat_waits_so_two_serves_never_share_a_spool(dead: str) -> None:
    ops = FakeOps(states={"j": dead}, age=30)
    out = _ensure(ops, runner_job.Record(job="j"))
    assert out.action == "wait-heartbeat" and not ops.submitted


def test_a_dead_job_adopts_another_live_server_instead_of_starting_one() -> None:
    ops = FakeOps(states={"j": "FAILED", "k": "RUNNING"}, age=900, found=["k", "j"])
    out = _ensure(ops, runner_job.Record(job="j"))
    assert (out.action, out.record.job) == ("adopted", "k") and not ops.submitted


def test_an_adopted_job_keeps_its_own_start_time_so_its_start_up_is_not_cut_short() -> None:
    ops = FakeOps(states={"k": "RUNNING"}, age=None, found=["k"],
                  started={"k": (T0 - timedelta(minutes=5)).isoformat()})
    out = _ensure(ops)
    assert (out.action, out.record.job) == ("ok", "k") and not ops.cancelled
    late = _ensure(ops, out.record, now=T0 + timedelta(minutes=40))
    assert late.action == "cancel-stale"  # past start-up with no heartbeat


@pytest.mark.parametrize("unsure", ["UNKNOWN", "HOST_UNREACHABLE", "PENDING"])
def test_a_candidate_whose_state_cannot_be_read_blocks_a_new_job(unsure: str) -> None:
    ops = FakeOps(states={"j": "FAILED", "k": unsure}, age=None, found=["k", "j"])
    out = _ensure(ops, runner_job.Record(job="j"))
    assert out.action == "unverified" and "cannot verify k" in out.detail and not ops.submitted
    first = _ensure(ops)  # no recorded job yet
    assert first.action == "unverified" and first.record.job is None and not ops.submitted


def test_an_unclear_submit_failure_is_never_retried_at_another_size() -> None:
    ops = FakeOps(states={"a": "MISSING"}, age=None, ambiguous=True)
    with pytest.raises(OSError, match="no job id") as caught:
        _ensure(ops, runner_job.Record(job="a"), runner=A, demand=14)
    assert not isinstance(caught.value, runner_job.SubmitRefused) and len(ops.submitted) == 1


def _done(out: str, rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], rc, out, "")


def test_run_on_submit_records_a_pending_job_and_tells_refusal_from_failure(
        monkeypatch: pytest.MonkeyPatch) -> None:
    ops = runner_job.RunOn()
    env = runner_job.job_env(B, pause_gib=None)
    monkeypatch.setattr(ops, "_run", lambda argv, timeout=None: _done(
        "run-on: menagerie-x awaiting admission\n", 75))
    assert ops.submit(B, env) == "menagerie-x"
    monkeypatch.setattr(ops, "_run", lambda argv, timeout=None: _done("job:   menagerie-y\n"))
    assert ops.submit(B, env) == "menagerie-y"
    monkeypatch.setattr(ops, "_run", lambda argv, timeout=None: _done(
        "run-on: refused: NO_CAPACITY needs 6 CPUs\n", 75))
    with pytest.raises(runner_job.SubmitRefused):
        ops.submit(B, env)
    monkeypatch.setattr(ops, "_run", lambda argv, timeout=None: _done(
        "run-on: menagerie-z awaiting admission\nrun-on: menagerie-z admission_timeout: NO_CAPACITY: "
        "worker-c: all projects together stay within the 18 cores\n", 75))
    with pytest.raises(runner_job.SubmitRefused) as refused:
        ops.submit(B, env)  # the admission wait ran out with no room: nothing ran, no job id
    assert refused.value.capacity
    monkeypatch.setattr(ops, "_run", lambda argv, timeout=None: _done("ssh: closed\n", 255))
    with pytest.raises(OSError) as caught:
        ops.submit(B, env)
    assert not isinstance(caught.value, runner_job.SubmitRefused)


def test_run_on_job_state_reads_admission_before_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    ops = runner_job.RunOn()
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[1] == "state":
            return _done(json.dumps({"verdict": verdict, "reason": reason}))
        return _done("menagerie-x  worker  INTERRUPTED (no result.json, no live unit)\n")

    monkeypatch.setattr(ops, "_run", run)
    verdict, reason = "pending", ""
    assert ops.job_state("menagerie-x") == "PENDING" and len(calls) == 1
    verdict = "admission_timeout"  # the admission wait ran out with no room: the job never launched
    assert ops.job_state("menagerie-x") == "REFUSED" and len(calls) == 2
    verdict, reason = "refused", "NO_CAPACITY: worker-c: 18 of 18 cores declared"
    assert ops.job_state("menagerie-x") == "REFUSED"
    verdict, reason = "refused", "NOT-STARTED: payload guard"  # not capacity: counts as a restart
    assert ops.job_state("menagerie-x") == "REJECTED" and len(calls) == 4
    verdict, reason = "admitted", ""
    assert ops.job_state("menagerie-x") == "INTERRUPTED"


def test_run_on_remembers_ended_job_states_across_passes(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    worker = {"menagerie-done": "DONE", "menagerie-live": "RUNNING", "menagerie-cut": "INTERRUPTED"}

    def run(argv: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        calls.append(argv[2])
        if argv[1] == "state":
            return _done(json.dumps({"verdict": "admitted", "reason": ""}))
        return _done(f"{argv[2]}  worker  {worker[argv[2]]} (detail)\n")

    ended = tmp_path / runner_job.ENDED_JOBS
    for _ in range(2):  # two passes, each with its own ops
        ops = runner_job.RunOn(ended=ended)
        monkeypatch.setattr(ops, "_run", run)
        assert [ops.job_state(j) for j in worker] == ["DONE", "RUNNING", "INTERRUPTED"]
    # The DONE job is asked once; a running or interrupted (restartable) one every pass.
    assert calls.count("menagerie-done") == 2 and calls.count("menagerie-live") == 4
    assert calls.count("menagerie-cut") == 4
    assert json.loads(ended.read_text()) == {"menagerie-done": "DONE"}
    plain = runner_job.RunOn()  # outside the tick nothing is remembered
    monkeypatch.setattr(plain, "_run", run)
    assert plain.job_state("menagerie-done") == "DONE" and calls.count("menagerie-done") == 4


def test_two_live_servers_are_reported_and_left_alone() -> None:
    ops = FakeOps(states={"k": "RUNNING", "l": "RUNNING"}, found=["k", "l"])
    out = _ensure(ops)
    assert out.action == "conflict" and len(out.messages) == 1 and not ops.cancelled


def test_a_stopped_spool_stays_stopped() -> None:
    assert _ensure(FakeOps(states={"j": "DONE"}, age=None), runner_job.Record(job="j")).action == "stopped"
    ops = FakeOps(states={"j": "FAILED"}, age=None, stop=True)
    assert _ensure(ops, runner_job.Record(job="j")).action == "stopped" and not ops.submitted


def test_the_disk_guard_holds_new_runner_jobs_but_not_the_stale_cancel() -> None:
    ops = FakeOps(states={"j": "FAILED"}, age=None)
    assert _ensure(ops, runner_job.Record(job="j"), hold=True).action == "held-disk"
    ops = FakeOps(states={"j": "RUNNING"}, age=3600)
    assert _ensure(ops, runner_job.Record(job="j"), hold=True).action == "cancel-stale"


def test_three_restarts_a_day_then_one_escalation_a_day() -> None:
    today = [(T0 - timedelta(hours=h)).isoformat() for h in (1, 2, 3)]
    ops = FakeOps(states={"j": "FAILED"}, age=None)
    first = _ensure(ops, runner_job.Record(job="j", restarts=tuple(today)))
    assert first.action == "escalate" and len(first.messages) == 1 and not ops.submitted
    second = _ensure(ops, first.record, now=T0 + timedelta(minutes=10))
    assert second.action == "escalate" and not second.messages
    tomorrow = _ensure(ops, second.record, now=T0 + timedelta(days=1))
    assert tomorrow.action == "restart"


def test_unverified_states_and_dry_runs_change_nothing() -> None:
    ops = FakeOps(states={"j": "HOST_UNREACHABLE"}, age=None)
    assert _ensure(ops, runner_job.Record(job="j")).action == "unverified"
    ops = FakeOps(states={"j": "FAILED"}, age=None)
    out = _ensure(ops, runner_job.Record(job="j"), dry_run=True)
    assert out.action == "restart" and out.record.job == "j" and not ops.submitted
    ops = FakeOps(states={"j": "RUNNING"}, age=3600)
    out = _ensure(ops, runner_job.Record(job="j"), dry_run=True)
    assert out.action == "cancel-stale" and not ops.cancelled and out.record.cancelled is None


def _manifest(root: Path, job: str, cmd: str, started: str = "2026-10-07T23:42:12Z") -> None:
    (root / job).mkdir(parents=True)
    (root / job / "manifest.json").write_text(json.dumps({"job": job, "cmd": cmd, "started": started}))


def test_discovery_finds_the_spool_b_job_and_never_the_night_side_spool(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = tmp_path / "jobs"
    legacy = "env X=1 bash -c $'# Runner B\\nSPOOL=$HOME/scratch/menagerie/spool-B\\nmkdir'"
    night = "bash -c $'SRC=$HOME/scratch/menagerie/spool-B; SPOOL=$HOME/scratch/menagerie/spool-Bn; W=2'"
    _manifest(jobs, "menagerie-b", legacy)
    _manifest(jobs, "menagerie-night", night)
    _manifest(jobs, "menagerie-tick", "env MENAGERIE_TICK_RUNNER=B bash x", "2026-10-08T23:00:00Z")
    _manifest(jobs, "menagerie-other", "env MENAGERIE_TICK_RUNNER=B2 bash x")
    monkeypatch.setenv("FLEET_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(runner_job, "DISCOVER_DAYS", 10_000)
    found = runner_job.RunOn().discover(B)
    assert [j for j, _ in found] == ["menagerie-tick", "menagerie-b"]
    assert found[0][1] == "2026-10-08T23:00:00+00:00"


def test_the_template_names_its_inputs_and_keeps_the_stop_file() -> None:
    text = (Path(runner_job.__file__).parent / "runner_job.sh").read_text()
    assert "RUNNER_SPOOL:?" in text and "MENAGERIE_TICK_RUNNER:?" in text and '"$SPOOL/STOP"' in text
    assert "build_rest &" in text and "wait $serves" in text  # the job never waits on a build



def _serve_python(tmp_path: Path, *, built: bool) -> subprocess.CompletedProcess[str]:
    """Run the template's ``PY=`` lookup alone with a stub ``uv`` that runs this interpreter."""

    import sys

    from menagerie.pipeline import envs

    text = (Path(runner_job.__file__).parent / "runner_job.sh").read_text()
    start = text.index("PY=$(")
    lookup = text[start:text.index("exit 1; }\n", start) + len("exit 1; }")]
    bin_dir, root = tmp_path / "bin", tmp_path / "envs"
    for path in (bin_dir, root, tmp_path / "outputs"):
        path.mkdir()
    (bin_dir / "uv").write_text(f'#!/bin/bash\nshift 5\nexec {sys.executable} "$@"\n')
    (bin_dir / "uv").chmod(0o755)
    if built:
        target = envs.prefix("torch-main", root=root)
        target.mkdir(parents=True)
        (target / envs.BUILT_MARKER).write_text(envs.lock_id("torch-main") + "\n")
    repo = Path(runner_job.__file__).resolve().parents[2]
    logs = tmp_path / "outputs"
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "MENAGERIE_ENVS": str(root)}
    script = f"cd {repo}\n{lookup.replace('outputs/', f'{logs}/')}\necho \"$PY\""
    return subprocess.run(["bash", "-uc", script], env=env, capture_output=True, text=True,
                          check=False)


def test_the_template_serves_from_the_shared_torch_main_build(tmp_path: Path) -> None:
    # No venv is synced into the job's checkout (1.4 GiB that stops sharing uv's cache).
    from menagerie.pipeline import envs

    text = (Path(runner_job.__file__).parent / "runner_job.sh").read_text()
    assert "uv sync" not in text and ".venv" not in text
    done = _serve_python(tmp_path, built=True)
    expected = envs.prefix("torch-main", root=tmp_path / "envs") / "bin" / "python"
    assert done.returncode == 0 and done.stdout.strip() == str(expected)


def test_the_template_stops_when_torch_main_is_not_built(tmp_path: Path) -> None:
    done = _serve_python(tmp_path, built=False)
    assert done.returncode == 1 and "torch-main is not built under" in done.stdout

def _build_rest(tmp_path: Path, *, free_gib: int, stop: bool = False) -> tuple[list[str], str]:
    """Run the template's ``build_rest`` alone with stub ``uv`` and ``df``: ``envs missing``
    names jax and tf, and the tf build fails. Returns the builds tried and envs.log."""

    text = (Path(runner_job.__file__).parent / "runner_job.sh").read_text()
    start = text.index("build_rest() {")
    func = text[start:text.index("\n}\n", start) + 3]
    bin_dir, spool = tmp_path / "bin", tmp_path / "spool"
    for path in (bin_dir, spool, tmp_path / "envs", tmp_path / "outputs"):
        path.mkdir()
    (bin_dir / "uv").write_text(
        '#!/bin/bash\ncase "$*" in *"envs missing"*) printf "jax\\ntf\\n" ;;\n'
        f'*"envs build"*) echo "${{@: -1}}" >> {tmp_path}/built; [ "${{@: -1}}" != tf ] ;;\nesac\n')
    (bin_dir / "df").write_text(f"#!/bin/bash\nprintf 'Avail\\n {free_gib}G\\n'\n")
    for stub in ("uv", "df"):
        (bin_dir / stub).chmod(0o755)
    if stop:
        (spool / "STOP").write_text("")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "SPOOL": str(spool),
           "MENAGERIE_ENVS": str(tmp_path / "envs"), "MENAGERIE_DISK_PAUSE_GIB": "58"}
    subprocess.run(["bash", "-uc", f"{func}\nbuild_rest"], cwd=tmp_path, env=env, check=True)
    built = tmp_path / "built"
    tried = built.read_text().split() if built.exists() else []
    return tried, (tmp_path / "outputs" / "envs.log").read_text()


def test_the_template_builds_every_missing_environment_and_never_stops_on_a_failure(
        tmp_path: Path) -> None:
    built, log = _build_rest(tmp_path, free_gib=100)
    assert built == ["jax", "tf"] and "envs build tf failed" in log and "jax failed" not in log


@pytest.mark.parametrize(("free_gib", "stop", "why"),
                         [(63, False, "skipped: 63 GiB free"), (100, True, "")])
def test_the_template_skips_builds_near_the_pause_line_and_after_stop(
        tmp_path: Path, free_gib: int, stop: bool, why: str) -> None:
    built, log = _build_rest(tmp_path, free_gib=free_gib, stop=stop)
    assert built == [] and why in log
    done = subprocess.run(["bash", "-n", str(Path(runner_job.__file__).parent / "runner_job.sh")],
                          check=False)
    assert done.returncode == 0


# --- one tick pass -----------------------------------------------------------------------------


@pytest.fixture
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    return tmp_path / "work" / "tick"


@dataclass
class World:
    """Fake outside calls for a pass; ``dirty`` lists before and after collect."""

    ops: FakeOps = field(default_factory=lambda: FakeOps(states={"j": "RUNNING"}))
    free: list[float | None] = field(default_factory=lambda: [88.0])
    before: set[str] = field(default_factory=set)
    after: set[str] = field(default_factory=set)
    active: set[str] = field(default_factory=set)
    sent: list[str] = field(default_factory=list)
    commits: list[list[str]] = field(default_factory=list)
    commit_fails: bool = False
    collected: list[str] = field(default_factory=list)
    added: dict[str, int] = field(default_factory=lambda: {"ledger/2026-10": 1})
    digests: list[str] = field(default_factory=lambda: ["h1"])
    send_fails: int = 0
    drained: int = 0
    full_calls: list[str] = field(default_factory=list)
    calls: int = 0
    lane_dirty: set[str] = field(default_factory=set)  # what the ``lanes`` step sees changed
    leased: set[str] = field(default_factory=set)
    launched: list[tuple[str, Path]] = field(default_factory=list)  # reclaim scripts started
    reaped: list[str] = field(default_factory=list)  # lane checkouts run-on was asked to reap
    held: set[str] = field(default_factory=set)  # jobs run-on keeps for their failed grace

    def read(self) -> Reading:
        value = self.free.pop(0) if len(self.free) > 1 else self.free[0]
        if value is None:
            raise OSError("ssh: timed out")
        return _gib(value)

    def changed(self, data: DataLayout) -> list[str]:
        self.calls += 1
        return sorted(self.before if self.calls % 2 else self.after)

    def commit(self, repo: Path, paths: Any, message: str, deadline: float) -> None:
        if self.commit_fails:
            raise OSError("index.lock held")
        self.commits.append(list(paths))

    def collect(self, spool: str, data: DataLayout) -> dict[str, Any]:
        self.collected.append(spool)
        return {"spool": spool, "collected": 1, "added": dict(self.added)}

    def digest(self, repo: Path, path: str) -> str:
        return self.digests.pop(0) if len(self.digests) > 1 else self.digests[0]

    def send(self, text: str, timeout: float | None = None) -> None:
        if self.send_fails:
            self.send_fails -= 1
            raise OSError("agent-send failed: busy")
        self.sent.append(text)

    def drain(self, data: DataLayout, dry_run: bool) -> dict[str, Any]:
        self.drained += 1
        return {"moved": 0}

    def full(self, name: str) -> Any:
        def call(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> Any:
            self.full_calls.append(name)
            if name == "trickle":
                raise NotImplementedError("rung 3: the r3-papers lane fills this in")
            return {"ran": name}
        return call

    def deps(self, now: datetime = T0) -> tick.Deps:
        return tick.Deps(disk_reader=self.read, runner_ops=self.ops, unit_active=self.active.__contains__,
                         collect_spool=self.collect, finished=lambda s: 3, changed=self.changed,
                         commit=self.commit, repo=lambda d: Path("/repo"), drain=self.drain,
                         digest=self.digest, send=self.send,
                         lane_changed=lambda d: sorted(self.lane_dirty),
                         lane_keys=lambda d: set(self.leased),
                         full={n: self.full(n) for n in ("tripwires", "requeue", "stage3_queue", "top_up", "trickle")},
                         demand=lambda kinds: dict.fromkeys(kinds, 0),
                         reclaim_launch=lambda script, log, repo: self.launched.append((script, repo)) or 1,
                         lane_reap=lambda label: self.reaped.append(label) or (0, f"REAPED\t{label}\t1\tfailed-ttl"),
                         lane_held=lambda: set(self.held),
                         now=lambda: now)


DATA = DataLayout(Path("/repo/menagerie/data"))


def _tick(world: World, mode: str | None = "collect", *, now: datetime = T0, dry_run: bool = False,
          state: dict[str, Any] | None = None) -> dict[str, Any]:
    if state is not None:
        tick.save_state(state)
    return tick.tick(mode, dry_run=dry_run, data=DATA, deps=world.deps(now), bound_s=None)


def _state_with_job(job: str = "j") -> dict[str, Any]:
    return {"runners": {"B": runner_job.Record(job=job).to_mapping()}}


def test_a_collect_pass_runs_its_steps_and_writes_state_log_status_and_heartbeat(work: Path) -> None:
    world = World(after={"menagerie/data/ledger/2026-10.jsonl", "menagerie/data/models/m1.json"})
    out = _tick(world, state=_state_with_job())
    assert set(out["results"]) == {"disk", "reclaim", "runners", "collect", "drain", "env_missing",
                                   "lanes", "lane_reap", "axon", "status"}
    assert not out["errors"] and out["results"]["env_missing"] == {}
    assert world.collected == ["scratch/menagerie/spool-B", "scratch/menagerie/spool-Bn",
                               "scratch/menagerie/spool-A"]
    assert world.commits == [["menagerie/data/ledger/2026-10.jsonl"]]  # models/ is not the tick's
    assert world.full_calls == [] and world.drained == 1 and world.sent == []
    for name in ("state.json", "tick.log", "status.md", "heartbeat"):
        assert (work / name).is_file()
    log = (work / "tick.log").read_text().splitlines()
    assert json.loads(log[0])["mode"] == "collect" and log[1].startswith("disk worker free=88.0GiB")
    assert set(json.loads(log[0])["step_seconds"]) == {"disk", "reclaim", "runners", "collect", "drain",
                                                       "env_missing", "lanes", "lane_reap", "axon"}
    assert "runner B: ok" in (work / "status.md").read_text()


def _trace(model_id: str, at: datetime, cls: str | None, env: str = "paddle@d485c924") -> str:
    row = {"schema": "ledger/1", "id": model_id, "stage": "trace",
           "at": at.strftime("%Y-%m-%dT%H:%M:%SZ"), "env": env,
           "outcome": "infra" if cls else "pass", "class": cls}
    return json.dumps(row, separators=(",", ":"))


def test_rows_without_a_built_environment_get_one_fact_a_day(work: Path, tmp_path: Path) -> None:
    data = DataLayout(tmp_path / "data")
    (data.root / "ledger").mkdir(parents=True)
    missing = "infra:env_missing"
    (data.root / "ledger" / "2026-10.jsonl").write_text("\n".join([
        _trace("m1", T0 - timedelta(hours=2), missing),
        _trace("m2", T0 - timedelta(hours=3), missing, env="jax@a740df71"),
        _trace("m2", T0 - timedelta(hours=1), None, env="jax@a740df71"),  # built and traced since
        _trace("m3", T0 - timedelta(days=2), missing, env="tf@fcc82f70"),  # older than a day
    ]) + "\n")
    world = World()
    for minutes in (0, 10):
        deps = world.deps(T0 + timedelta(minutes=minutes))
        out = tick.tick("collect", data=data, deps=deps, bound_s=None)
        assert out["results"]["env_missing"] == {"paddle@d485c924": ["m1"]}
    facts = [t for t in world.sent if t.startswith("latest trace found no built environment")]
    assert len(facts) == 1 and "paddle@d485c924 (1 ids)" in facts[0]  # beside collect's error FACT
    assert "env_missing" in (work / "status.md").read_text()


def test_files_another_writer_left_dirty_are_skipped_and_a_failed_commit_is_retried(work: Path) -> None:
    foreign, mine = "menagerie/data/ledger/2026-09.jsonl", "menagerie/data/store-index/2026-10.jsonl"
    world = World(before={foreign}, after={foreign, mine}, commit_fails=True,
                  added={"ledger/2026-09": 2, "store-index/2026-10": 1})
    out = _tick(world, state=_state_with_job())
    assert "collect" in out["errors"] and tick.load_state()["pending_commit"] == {mine: "h1"}
    world.commit_fails, world.calls = False, 0
    world.before = {foreign, mine}  # still dirty from the failed commit
    out = _tick(world, now=T0 + timedelta(minutes=10))
    assert world.commits == [[mine]] and out["results"]["collect"]["skipped_foreign_dirty"] == [foreign]
    assert tick.load_state()["pending_commit"] == {}


def test_a_left_file_another_writer_changed_since_is_skipped(work: Path) -> None:
    mine = "menagerie/data/store-index/2026-10.jsonl"
    world = World(after={mine}, commit_fails=True, added={"store-index/2026-10": 1},
                  digests=["h1", "h2"])
    _tick(world, state=_state_with_job())
    world.commit_fails, world.calls, world.before, world.added = False, 0, {mine}, {}
    out = _tick(world, now=T0 + timedelta(minutes=10))
    assert world.commits == [] and out["results"]["collect"]["skipped_foreign_dirty"] == [mine]


def test_a_file_collect_did_not_touch_is_never_committed(work: Path) -> None:
    other = "menagerie/data/ledger/2026-10.jsonl"  # dirtied mid-pass by another writer
    world = World(after={other}, added={})
    out = _tick(world, state=_state_with_job())
    assert world.commits == [] and out["results"]["collect"]["commit"] == []


def test_one_collector_per_spool(work: Path) -> None:
    world = World(active={"menagerie-catalog-collect.timer"})
    out = _tick(world, state=_state_with_job())
    assert "another collector owns" in out["results"]["collect"]["skipped"] and not world.collected
    world.active.clear()
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "collect-spool-Bn.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = _tick(world, now=T0 + timedelta(minutes=10))
    assert world.collected == ["scratch/menagerie/spool-B", "scratch/menagerie/spool-A"]
    assert out["results"]["collect"]["spools"][1]["skipped"] == "another collector holds its lock"


def test_a_paused_disk_holds_queueing_top_up_and_new_runners_while_the_rest_runs(work: Path) -> None:
    world = World(ops=FakeOps(states={"j": "FAILED"}, age=None), free=[58.0])
    out = _tick(world, "full", state=_state_with_job())
    assert out["results"]["runners"][0]["action"] == "held-disk" and not world.ops.submitted
    for held in ("requeue", "stage3_queue", "top_up"):
        assert out["results"][held] == {"held": "disk paused"}
    assert world.full_calls == ["tripwires", "trickle"] and out["results"]["trickle"]["waiting"].startswith("rung 3")
    assert world.drained == 1 and len(world.sent) == 1 and "disk paused" in world.sent[0]
    _tick(world, "full", now=T0 + timedelta(minutes=10))
    assert len(world.sent) == 1  # no second FACT


def test_a_critical_disk_holds_the_drain(work: Path) -> None:
    world = World(free=[49.0])
    out = _tick(world, state=_state_with_job())
    assert out["results"]["drain"] == {"held": "disk critical"} and world.drained == 0
    assert len(world.sent) == 1 and "critical" in world.sent[0]


def test_full_mode_runs_its_steps_when_the_disk_is_clear(work: Path) -> None:
    world = World()
    out = _tick(world, "full", state=_state_with_job())
    assert world.full_calls == ["tripwires", "requeue", "stage3_queue", "top_up", "trickle"]
    assert out["results"]["top_up"] == {"ran": "top_up"}


def test_the_mode_file_chooses_the_mode_and_a_bad_one_is_refused(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "mode").write_text("full\n")
    assert _tick(World(), None, state=_state_with_job())["mode"] == "full"
    (work / "mode").write_text("fast\n")
    with pytest.raises(ValueError, match="fast"):
        _tick(World(), None)


def test_misordered_disk_lines_stop_the_tick_before_any_step(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "config.json").write_text(json.dumps({"disk": {"pause_gib": 40}}))
    world = World()
    with pytest.raises(disk_guard.GuardConfigError):
        _tick(world)
    assert world.collected == [] and not (work / "tick.log").exists()


def test_a_held_lock_skips_the_pass(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    with open(work / "tick.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = _tick(World())
    assert out["skipped"] == "another tick holds the lock"
    with open(work / "tick.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = runner_job.ensure(ops=FakeOps(), mode="collect")
        assert out[0]["skipped"].startswith("a tick holds the lock")


def test_a_dry_run_changes_nothing_and_sends_nothing(work: Path) -> None:
    world = World(ops=FakeOps(states={"j": "FAILED"}, age=None), free=[55.0],
                  active={"menagerie-catalog-collect.timer"})
    out = _tick(world, dry_run=True, state=_state_with_job())
    before = (work / "state.json").read_text()
    assert out["dry_run"] and out["results"]["collect"]["would_collect"]["scratch/menagerie/spool-B"] == 3
    assert "another collector owns" in out["results"]["collect"]["skipped"]
    assert out["results"]["runners"][0]["action"] == "held-disk" and len(out["would_send"]) == 1
    assert world.sent == [] and world.commits == [] and world.drained == 1  # drain's own dry run
    assert (work / "state.json").read_text() == before and not (work / "tick.log").exists()


REFILL_AT = T0 + timedelta(minutes=5)  # the next full pass is five minutes away


def _refill(world: World, work: Path, *, now: datetime = REFILL_AT, a: dict[str, Any] | None = None,
            checked: datetime = T0, want: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """A refill pass with runner A's saved record ``a``; returns its result and top_up's cfg."""
    work.mkdir(parents=True, exist_ok=True)
    if want:
        (work / tick.REFILL_WANT).touch()
    tick.save_state({"runners": {"A": a or runner_job.Record(job="j", size=0).to_mapping()},
                     "tripwires_at": tick._iso(checked), "error_streak": 1})
    seen: dict[str, Any] = {}
    deps = world.deps(now)
    deps.full["top_up"] = lambda data, cfg, dry_run: seen.update(cfg) or {"ran": "top_up"}
    return tick.tick("refill", data=DATA, deps=deps, bound_s=None), seen


def test_a_refill_tops_up_2b_only_with_runner_a_s_saved_declaration_and_clears_the_want(work: Path) -> None:
    world = World()
    out, cfg = _refill(world, work)
    assert set(out["results"]) == {"disk", "runners", "top_up"} and not out["errors"]
    assert out["results"]["runners"] == [] and world.ops.submitted == []  # no runner was refused
    assert world.full_calls == [] and world.collected == [] and world.drained == 0  # no tripwires either
    assert cfg["lane_limits"] == {"2a": 0} and cfg["runner_declaration"] == runner_job.tries_declaration(
        {"job": "j", "job_state": runner_job.LIVE, "size": 0, "action": None})[0]
    assert not (work / tick.REFILL_WANT).exists()
    assert json.loads((work / "tick.log").read_text().splitlines()[0])["mode"] == "refill"
    assert not (work / "status.md").exists() and not (work / "heartbeat").exists()
    assert tick.load_state()["error_streak"] == 1 and world.sent == []  # the timer's passes own both


@pytest.mark.parametrize("room", [False, True], ids=["still-full", "room-at-last"])
def test_a_refill_retries_a_runner_refused_for_capacity_and_asks_for_the_next_refill(
        work: Path, room: bool) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "mode").write_text("full\n")
    world = World(ops=FakeOps(states={"j": "REFUSED"}, age=None, refuse_above=None if room else 0))
    at = tick._iso(REFILL_AT - timedelta(minutes=9))
    out, _ = _refill(world, work, a=runner_job.Record(job="j", size=0, submitted_at=at,
                                                      refused_at=at).to_mapping())
    a = next(r for r in out["results"]["runners"] if r["runner"] == "A")
    assert a["action"] == ("restart" if room else "held-capacity") and not out["errors"]
    assert (work / tick.REFILL_WANT).exists() is not room  # refused again: the next refill retries
    assert runner_job.refused(runner_job.Record.from_mapping(tick.load_state()["runners"]["A"])) is not room


def test_a_refill_holds_2b_while_runner_a_waits_on_a_resize(work: Path) -> None:
    _, cfg = _refill(World(), work, a=runner_job.Record(job="j", size=0, resize_to=1).to_mapping())
    assert "held" in cfg["runner_declaration"]


def test_a_refill_holds_when_no_full_pass_checked_the_tripwires_lately(work: Path) -> None:
    out, cfg = _refill(World(), work, checked=T0 - timedelta(minutes=20))
    assert out["results"]["top_up"]["held"].startswith("no full pass checked") and cfg == {}
    assert (work / tick.REFILL_WANT).exists()  # the next refill tries again


def test_a_refill_too_close_to_the_next_full_pass_leaves_it_the_want(work: Path) -> None:
    out, cfg = _refill(World(), work, now=T0 + timedelta(minutes=9))
    assert out == {"mode": "refill", "skipped": "the next full pass is 60 s away"} and cfg == {}
    assert (work / tick.REFILL_WANT).exists() and not (work / "tick.log").exists()


def test_a_full_pass_takes_the_want_and_stamps_the_tripwire_check(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / tick.REFILL_WANT).touch()
    _tick(World(), "full", state=_state_with_job())
    assert not (work / tick.REFILL_WANT).exists() and tick.load_state()["tripwires_at"] == tick._iso(T0)


def test_errors_two_passes_in_a_row_escalate_once_a_day(work: Path) -> None:
    world = World(commit_fails=True, after={"menagerie/data/ledger/2026-10.jsonl"})
    _tick(world, state=_state_with_job())
    assert world.sent == []
    for n in (1, 2, 3):
        world.calls = 0
        _tick(world, now=T0 + timedelta(minutes=10 * n))
    assert len(world.sent) == 1 and world.sent[0].startswith("tick errors 2 passes in a row: collect")


def test_a_failed_send_waits_in_the_outbox_and_a_repeated_message_goes_once_a_day(work: Path) -> None:
    ops = FakeOps(states={"k": "RUNNING", "l": "RUNNING"}, found=["k", "l"])
    world = World(ops=ops, send_fails=1)
    out = _tick(world)
    assert out["sent"] == [] and len(tick.load_state()["outbox"]) == 1
    for n in (1, 2, 3):
        _tick(world, now=T0 + timedelta(minutes=10 * n))
    assert len(world.sent) == 1 and "2 live jobs serve" in world.sent[0]
    assert tick.load_state()["outbox"] == []
    _tick(world, now=T0 + timedelta(days=1))
    assert len(world.sent) == 2  # the next day, once more


def test_a_real_temp_spool_is_collected_once(work: Path, tmp_path: Path) -> None:
    spool = tmp_path / "spool-T"
    (spool / "stage3" / "done").mkdir(parents=True)
    name = "m9001-aaaaaaaaaaaa-20261008t000000z-0000"
    index = {"sha256": "c" * 64, "bytes": 1, "mime": "application/json",
             "fetched": "2026-10-08T00:00:00Z", "license_class": "derived"}
    (spool / "stage3" / "done" / f"{name}.json").write_text(json.dumps({
        "schema": stage3.DONE_SCHEMA, "name": name, "id": "m9001", "rev": "a" * 12,
        "outcome": "done", "store_index": [json.dumps(index)]}))
    data = DataLayout(tmp_path / "data")
    first = tick._collect_spool(str(spool), data)
    assert (first["collected"], first["added"]) == (1, {"store-index/2026-10": 1})
    assert tick._collect_spool(str(spool), data)["collected"] == 0
    assert len(data.store_index("2026-10").read_text().splitlines()) == 1


# --- the tick commits what it writes, and only that (lane tick-trickle) -----------------------

def _shard_line(char: str) -> str:
    return json.dumps({"sha256": char * 64, "bytes": 1, "mime": "application/json",
                       "fetched": "2026-10-01T00:00:00Z", "license_class": "derived"}) + "\n"


BASE_LINE = _shard_line("e")  # store-index lines: collect reads every shard's sha256
FOREIGN_LINE = _shard_line("f")
SHARD = "menagerie/data/store-index/2026-10.jsonl"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout


def _index_line(n: int) -> StoreIndexLine:
    return StoreIndexLine(sha256=f"{n:064x}", bytes=n, mime="application/json",
                          fetched="2026-10-10T00:00:00Z", license_class="third-party")


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DataLayout:
    """A real git checkout whose store-index shard holds another writer's uncommitted line."""

    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = Tick Test\n\temail = tick@example.invalid\n"
                   "[init]\n\tdefaultBranch = main\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    shard = root / SHARD
    shard.parent.mkdir(parents=True)
    shard.write_text(BASE_LINE)
    _git(root, "add", SHARD)
    _git(root, "commit", "-qm", "base")
    with shard.open("a") as handle:
        handle.write(FOREIGN_LINE)
    return DataLayout(root.resolve() / "menagerie" / "data")


def _writing_trickle(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> Any:
    write_json(_index_line(1), data.packet("f-a"))  # stands in for a packet amendment
    append_jsonl([_index_line(2)], data.cite("2026-10"))
    append_jsonl([_index_line(3)], data.store_index("2026-10"))
    return [{"family": "f-a", "status": "unmatched"}]


def _full_deps(world: World, repo: Path, **changes: Any) -> tick.Deps:
    deps = world.deps()
    return replace(deps, repo=lambda d: repo, full={**deps.full, "trickle": _writing_trickle},
                   **changes)


def test_the_trickle_output_is_committed_without_another_writers_lines(work: Path,
                                                                        checkout: DataLayout) -> None:
    repo = checkout.root.parents[1]
    tick.save_state(_state_with_job())
    out = tick.tick("full", data=checkout, deps=_full_deps(World(added={}), repo), bound_s=None)
    assert not {"collect", "trickle", "index"} & set(out["errors"])
    assert out["results"]["trickle"]["harvest"][0]["family"] == "f-a"
    packet = checkout.packet("f-a").relative_to(repo).as_posix()
    assert out["results"]["trickle"]["commit"]["paths"] == sorted(
        [packet, "menagerie/data/cite/2026-10.jsonl", SHARD])
    head = _git(repo, "show", f"HEAD:{SHARD}")
    assert head == BASE_LINE + dumps(_index_line(3)) + "\n"  # the tick's line, not the other's
    assert FOREIGN_LINE in (repo / SHARD).read_text()
    assert _git(repo, "status", "--porcelain").splitlines() == [f" M {SHARD}"]
    assert tick.load_state()["pending_owned"] == {}


def test_a_failed_trickle_commit_is_kept_and_made_by_the_next_pass(work: Path,
                                                                  checkout: DataLayout) -> None:
    repo = checkout.root.parents[1]
    tick.save_state(_state_with_job())

    def refuse(*args: Any) -> Any:
        raise OSError("index.lock held")

    out = tick.tick("full", data=checkout, deps=_full_deps(World(added={}), repo, commit_owned=refuse),
                    bound_s=None)
    assert "trickle" in out["errors"] and SHARD in tick.load_state()["pending_owned"]["lines"]
    deps = replace(_full_deps(World(added={}), repo), full={**World().deps().full,
                                                           "trickle": lambda d, c, dry_run: []})
    out = tick.tick("full", data=checkout, deps=deps, bound_s=None)
    assert not {"collect", "trickle", "index"} & set(out["errors"])
    assert tick.load_state()["pending_owned"] == {}
    assert dumps(_index_line(3)) in _git(repo, "show", f"HEAD:{SHARD}")
    assert FOREIGN_LINE not in _git(repo, "show", f"HEAD:{SHARD}")


def test_collect_commits_its_own_lines_in_a_store_index_another_writer_left_dirty(
        work: Path, checkout: DataLayout, tmp_path: Path) -> None:
    repo = checkout.root.parents[1]
    spool = tmp_path / "spool-T"
    (spool / "stage3" / "done").mkdir(parents=True)
    name = "m9001-aaaaaaaaaaaa-20261008t000000z-0000"
    index = json.dumps({"sha256": "c" * 64, "bytes": 1, "mime": "application/json",
                        "fetched": "2026-10-08T00:00:00Z", "license_class": "derived"})
    (spool / "stage3" / "done" / f"{name}.json").write_text(json.dumps({
        "schema": stage3.DONE_SCHEMA, "name": name, "id": "m9001", "rev": "a" * 12,
        "outcome": "done", "store_index": [index]}))
    work.mkdir(parents=True, exist_ok=True)
    (work / "config.json").write_text(json.dumps({"collect_spools": [str(spool)]}))
    tick.save_state(_state_with_job())
    deps = replace(World().deps(), repo=lambda d: repo, collect_spool=tick._collect_spool,
                   changed=None)
    out = tick.tick("collect", data=checkout, deps=deps, bound_s=None)
    assert not {"collect", "index"} & set(out["errors"]), out["errors"]
    collect = out["results"]["collect"]
    assert collect["skipped_foreign_dirty"] == [] and collect["commit"] == []
    assert collect["own_lines"]["paths"] == [SHARD]
    assert _git(repo, "show", f"HEAD:{SHARD}") == BASE_LINE + index + "\n"
    assert (repo / SHARD).read_text() == BASE_LINE + FOREIGN_LINE + index + "\n"


def test_collect_never_commits_a_line_another_writer_appends_during_the_pass(
        work: Path, checkout: DataLayout, tmp_path: Path) -> None:
    repo = checkout.root.parents[1]
    spool = tmp_path / "spool-T"
    (spool / "stage3" / "done").mkdir(parents=True)
    name = "m9002-aaaaaaaaaaaa-20260908t000000z-0000"
    index = json.dumps({"sha256": "d" * 64, "bytes": 1, "mime": "application/json",
                        "fetched": "2026-09-08T00:00:00Z", "license_class": "derived"})
    (spool / "stage3" / "done" / f"{name}.json").write_text(json.dumps({
        "schema": stage3.DONE_SCHEMA, "name": name, "id": "m9002", "rev": "a" * 12,
        "outcome": "done", "store_index": [index]}))
    work.mkdir(parents=True, exist_ok=True)
    (work / "config.json").write_text(json.dumps({"collect_spools": [str(spool)]}))
    tick.save_state(_state_with_job())
    shard = checkout.store_index("2026-09")  # clean (absent) before the pass

    def collect_then_another_writer(spool_name: str, data: DataLayout) -> dict[str, Any]:
        got = tick._collect_spool(spool_name, data)
        if got["collected"]:
            with shard.open("a") as handle:
                handle.write(FOREIGN_LINE)
        return got

    def refuse(*args: Any) -> Any:
        raise OSError("index.lock held")

    deps = replace(World().deps(), repo=lambda d: repo, collect_spool=collect_then_another_writer,
                   changed=None, commit_owned=refuse)
    out = tick.tick("collect", data=checkout, deps=deps, bound_s=None)
    rel = shard.relative_to(repo).as_posix()
    assert "collect" in out["errors"] and tick.load_state()["pending_owned"]["lines"][rel] == [index + "\n"]
    deps = replace(deps, commit_owned=tick_commit.commit_owned)
    out = tick.tick("collect", data=checkout, deps=deps, bound_s=None)
    assert not {"collect", "index"} & set(out["errors"]), out["errors"]
    assert _git(repo, "show", f"HEAD:{rel}") == index + "\n"
    assert shard.read_text() == index + "\n" + FOREIGN_LINE


def test_a_dry_run_trickle_lists_due_packets_and_writes_nothing(work: Path, tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import bib_harvest, fetch
    from menagerie.pipeline.records import (Availability, Citation, Identity, Member, Packet,
                                            Search, Source)

    data = DataLayout(tmp_path / "data")
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    write_json(Packet(family="f-a", written="2026-10-08T20:00:00Z", by="research",
                      members=(Member(id="m1"),), identity=Identity(canonical="Example"),
                      sources=(Source(sid="s1", role="page", url="https://example.org/model"),),
                      availability=Availability(paper="not_found", code="not_found",
                                                license="not_found", description="insufficient"),
                      search=Search(places=("official README",)),
                      citation=Citation(title="Example", arxiv="2103.14899")), data.packet("f-a"))
    files = {p: p.read_bytes() for p in data.root.rglob("*") if p.is_file()}

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dry run must not fetch")

    monkeypatch.setattr(bib_harvest, "fill", boom)
    monkeypatch.setattr(fetch, "bib_records", boom)
    world = World()
    deps = world.deps()
    deps = replace(deps, full={k: v for k, v in deps.full.items() if k != "trickle"})
    tick.save_state(_state_with_job())
    out = tick.tick("full", dry_run=True, data=data, deps=deps, bound_s=None)
    assert out["results"]["trickle"] == [{"family": "f-a", "status": "due", "ref": "arXiv:2103.14899"}]
    assert {p: p.read_bytes() for p in data.root.rglob("*") if p.is_file()} == files
    assert not (tmp_path / "store").exists()


# --- full-mode steps ---------------------------------------------------------------------------


def test_the_worklist_reads_ids_in_order_without_comments_or_repeats(work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "worklist.txt").write_text("# residual\nm2\nm1  # first\n\nm2\n")
    assert tick_full.worklist({}) == ["m2", "m1"]


def test_stage3_queue_fills_the_spool_to_its_depth(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import drive

    work.mkdir(parents=True, exist_ok=True)
    (work / "worklist.txt").write_text("\n".join(f"m{i}" for i in range(10)))
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "forward" if int(i[1:]) % 2 else "2b"} for i in ids])
    monkeypatch.setattr(tick_full, "_waiting", lambda spool: 37)
    monkeypatch.setattr(tick_full, "_not_pending", lambda ids, data: [i for i in ids if i != "m1"])
    queued: list[list[str]] = []
    monkeypatch.setattr(stage3, "queue", lambda lane, data, ids, spool: queued.append(list(ids)) or ids)
    cfg = {**tick.DEFAULTS}
    out = tick_full.stage3_queue(DATA, cfg, dry_run=False)
    assert queued == [["m3", "m5", "m7"]] and out["eligible"] == 5  # depth 40, 37 waiting
    assert tick_full.stage3_queue(DATA, cfg, dry_run=True)["would_queue"] == ["m3", "m5", "m7"]


def _worklist_world(work: Path, monkeypatch: pytest.MonkeyPatch, slots: int) -> list[Any]:
    from collections import Counter

    from menagerie.pipeline import drive

    monkeypatch.delenv("MENAGERIE_BRIDGE_SPOOL", raising=False)
    spool = work.parent / "hubspool"
    spool.mkdir(parents=True, exist_ok=True)
    (spool / "serving").touch()
    work.mkdir(parents=True, exist_ok=True)
    (work / "worklist.txt").write_text("\n".join(f"m{i}" for i in range(1, 13)))
    monkeypatch.setattr(drive, "next_steps", lambda ids, data: [
        {"id": i, "step": "2b" if int(i[1:]) <= 5 else "2a"} for i in ids])
    monkeypatch.setattr(drive, "families_of", lambda ids, data: {
        i: (f"f{i}" if int(i[1:]) <= 5 else None) for i in ids})
    monkeypatch.setattr(tick_full, "_busy", lambda: (Counter({"2b": 1}), {"fm1"}))
    monkeypatch.setattr(tick_full, "hub_slots", lambda: slots)
    launched: list[Any] = []
    monkeypatch.setattr(tick_full, "_launch_2b", lambda fams, data: launched.append(fams) or ["r"])
    monkeypatch.setattr(tick_full, "_launch_2a", lambda chunks, data: launched.append(chunks) or ["s"])
    return launched


def test_top_up_respects_the_2b_cap_held_families_and_keeps_2b_a_tick_ahead(
        work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launched = _worklist_world(work, monkeypatch, slots=10)
    monkeypatch.setenv("MENAGERIE_LANES_2B", "3")
    cfg = {**tick.DEFAULTS, "2b_on_hub": True}
    out = tick_full.top_up(DATA, cfg, dry_run=False)
    assert out["caps"] == {"2a": 4, "2b": 3}
    assert out["2b"] == ["fm2", "fm3"]  # 3 allowed, 1 busy; fm1 held by a lane
    # fm4 and fm5 still wait, short of the default buffer of 6 (2 x cap 3): one 2a lane of up to 8 rows
    assert out["2a"] == [["m6", "m7", "m8", "m9", "m10", "m11", "m12"]]
    assert launched == [["fm2", "fm3"], out["2a"]] and out["runs"] == ["r", "s"]
    assert tick_full.top_up(DATA, cfg, dry_run=True).get("runs") is None


def test_top_up_at_the_default_cap_and_with_no_hub_room(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launched = _worklist_world(work, monkeypatch, slots=3)
    cfg = {**tick.DEFAULTS, "2b_on_hub": True}
    out = tick_full.top_up(DATA, cfg, dry_run=False)
    assert out["caps"]["2b"] == 14 and out["2b"] == ["fm2", "fm3", "fm4"] and out["2a"] == []
    launched.clear()
    monkeypatch.setattr(tick_full, "hub_slots", lambda: 0)
    out = tick_full.top_up(DATA, cfg, dry_run=False)
    assert out["2b"] == [] and out["2a"] == [] and launched == [[], []]


def test_lane_caps_default_to_14_2b_lanes_and_never_pass_16(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import admission, drive

    monkeypatch.delenv("MENAGERIE_LANES_2B", raising=False)
    assert admission.lane_cap("2b") == 14 and drive.lane_cap is admission.lane_cap
    monkeypatch.setenv("MENAGERIE_LANES_2B", "20")
    assert admission.lane_cap("2b") == 16
    monkeypatch.setenv("MENAGERIE_LANES_2B", "12")
    assert admission.lane_cap("2b") == 12
    assert tick_full._caps({"lane_limits": {"2b": 40, "2a": 1}}) == {"2a": 1, "2b": 12}


def test_2a_lanes_start_only_when_the_2b_queue_runs_short() -> None:
    need = [f"m{i}" for i in range(40)]
    assert tick_full._plan_2a(need, ready_2b=14, busy_2a=0, target=14, cap_2a=4, per=8) == []
    assert len(tick_full._plan_2a(need, ready_2b=0, busy_2a=0, target=14, cap_2a=4, per=8)) == 2
    assert tick_full._plan_2a(need, ready_2b=0, busy_2a=2, target=14, cap_2a=4, per=8) == []
    assert len(tick_full._plan_2a(need, ready_2b=0, busy_2a=0, target=40, cap_2a=4, per=8)) == 4




def test_runner_a_wants_one_worker_per_five_live_2b_lanes() -> None:
    assert [A.wanted(n) for n in (0, 5, 6, 10, 11, 14, 40)] == [0, 0, 1, 1, 2, 2, 2]
    assert B.wanted(14) is None and A.sized(2).cpus == 9


def test_runner_a_starts_at_the_largest_size_run_on_admits() -> None:
    ops = FakeOps(states={"a": "MISSING"}, age=None, refuse_above=2)
    out = _ensure(ops, runner_job.Record(job="a"), runner=A, demand=14)
    assert out.action == "restart" and out.record.size == 1
    assert ops.submitted[0]["RUNNER_WORKERS"] == "2" and ops.submitted[0]["MENAGERIE_CPUS"] == "6"
    ops = FakeOps(states={"a": "MISSING"}, age=None, refuse_above=0)
    out = _ensure(ops, runner_job.Record(job="a"), runner=A, demand=14)
    assert out.action == "held-capacity" and "no size admitted" in out.detail
    assert out.record.restarts == () and out.record.job == "a" and not ops.submitted


def test_a_non_capacity_refusal_at_every_size_is_still_an_error() -> None:
    class Policy(FakeOps):
        def submit(self, runner: runner_job.Runner, env: Mapping[str, str]) -> str:
            raise runner_job.SubmitRefused("refused: PLACEMENT host not cleared")

    with pytest.raises(OSError, match="no size admitted") as caught:
        _ensure(Policy(states={"a": "MISSING"}, age=None), runner_job.Record(job="a"), runner=A)
    assert not isinstance(caught.value, runner_job.NoCapacity)


@dataclass
class ProbedOps(FakeOps):
    """FakeOps with ``run-on probe``: answers by worker count, admitting when none is given."""

    answers: dict[int, tuple[int, str]] = field(default_factory=dict)
    probes: list[tuple[int, str | None]] = field(default_factory=list)

    def probe(self, runner: runner_job.Runner, replacing: str | None) -> tuple[int, str]:
        self.probes.append((runner.workers, replacing))
        return self.answers.get(runner.workers, (runner_job.PROBE_ADMIT, ""))


FULL = (runner_job.PROBE_REFUSED,
        "NO_CAPACITY: worker-c: Worker budget refused: available RAM cannot cover commitments and reserve")


@pytest.mark.parametrize("answer", [FULL, (runner_job.PROBE_UNKNOWN, "run-on probe timed out")])
def test_a_resize_keeps_the_job_until_run_on_probe_admits_its_successor(answer: tuple[int, str]) -> None:
    ops, record = ProbedOps(states={"a": "RUNNING"}, answers={2: answer}), runner_job.Record(job="a", size=0)
    for n in range(runner_job.GROW_DRAIN_AFTER + 2):
        out = _ensure(ops, record, runner=A, demand=10, now=T0 + timedelta(minutes=10 * n))
        record = out.record
    assert not ops.cancelled and not ops.submitted and not any(ops.drains)
    assert "keeps a" in out.detail and ops.probes[-1] == (2, "a")
    ops.answers.clear()  # the room is back: the next pass cancels for the resize
    out = _ensure(ops, record, runner=A, demand=10, now=T0 + timedelta(hours=2))
    assert out.action == "resize" and ops.cancelled == ["a"]


def test_a_restart_starts_at_the_largest_size_run_on_probe_admits() -> None:
    ops = ProbedOps(states={"a": "MISSING"}, age=None, answers={3: FULL, 2: FULL})
    out = _ensure(ops, runner_job.Record(job="a"), runner=A, demand=14)
    assert out.action == "restart" and out.record.size == 0
    assert [e["RUNNER_WORKERS"] for e in ops.submitted] == ["1"]
    assert ops.probes == [(3, None), (2, None), (1, None)]


@pytest.mark.parametrize("ops", [ProbedOps(states={"a": "MISSING"}, age=None, answers={3: FULL, 2: FULL, 1: FULL}),
                                 ProbedOps(states={"a": "MISSING"}, age=None, refuse_above=0)],
                         ids=["probe-refuses-all", "probe-admits-submit-refuses"])
def test_a_restart_with_no_room_is_held_for_capacity_whichever_refuses(ops: ProbedOps) -> None:
    out = _ensure(ops, runner_job.Record(job="a"), runner=A, demand=14)
    assert out.action == "held-capacity" and "no size admitted" in out.detail
    assert out.record.restarts == () and out.record.job == "a" and not ops.submitted


@pytest.mark.parametrize("rc,stdout,expected", [
    (6, '{"verdict": "refused", "reason": "NO_CAPACITY: worker-c: full"}', (6, "NO_CAPACITY: worker-c: full")),
    (0, '{"verdict": "admitted"}', (0, "")),
    (7, '{"verdict": "unknown", "reason": "unreachable"}', (7, "unreachable")),
    (2, "usage: run-on probe ...", (7, "usage: run-on probe ...")),
])
def test_run_on_probe_reads_the_exit_code_and_reason(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rc: int, stdout: str, expected: tuple[int, str]) -> None:
    (tmp_path / ".compute-host").write_text("worker-c\n", encoding="utf-8")
    monkeypatch.setattr(runner_job, "_repo", lambda: tmp_path)
    calls = []

    def run(self: runner_job.RunOn, argv: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, rc, stdout, "")

    monkeypatch.setattr(runner_job.RunOn, "_run", run)
    assert runner_job.RunOn().probe(A, "menagerie-old") == expected
    assert calls[0][:3] == ["run-on", "probe", "worker-c"] and calls[0][-3:] == ["--replacing", "menagerie-old", "--json"]


@pytest.mark.parametrize("how", ["refused-at-submit", "refused-after-pending"])
def test_a_full_worker_never_uses_up_runner_a_restarts(how: str) -> None:
    # every 10 minutes for 6 hours run-on finds no room: outright, or after "awaiting admission"
    if how == "refused-at-submit":
        ops = FakeOps(age=None, refuse_above=0)
    else:
        ops = FakeOps(age=None, new_state="PENDING")
    record, actions = runner_job.Record(), []
    for n in range(36):
        out = _ensure(ops, record, runner=A, now=T0 + timedelta(minutes=10 * n))
        assert not out.messages and out.action != "escalate"
        actions.append(out.action)
        record = out.record
        for job, state in ops.states.items():  # a pending job's admission ends: no room
            ops.states[job] = "REFUSED" if state == "PENDING" else state
    assert len(record.restarts) <= 1
    if how == "refused-after-pending":  # no backoff: every pass tries again
        assert set(actions) == {"restart"} and len(ops.submitted) == 36
    else:
        assert set(actions) == {"held-capacity"} and record.refused_at is not None
    ops.refuse_above, ops.new_state = None, "RUNNING"  # room at last
    later = T0 + timedelta(hours=7)
    started = _ensure(ops, record, runner=A, now=later)
    assert started.action == "restart" and len(started.record.restarts) == 1
    assert started.record.refused_at is None


def test_a_refused_job_is_refunded_and_retried_at_once_and_a_start_that_dies_still_counts() -> None:
    submitted = (T0 - timedelta(minutes=10)).isoformat(timespec="seconds")
    earlier = (T0 - timedelta(hours=2)).isoformat(timespec="seconds")
    ops = FakeOps(states={"a": "REFUSED"}, age=None, refuse_above=0)
    record = runner_job.Record(job="a", submitted_at=submitted, restarts=(earlier, submitted))
    held = _ensure(ops, record, runner=A)
    assert held.action == "held-capacity" and not ops.submitted
    assert held.record.restarts == (earlier,) and runner_job.refused(held.record)
    ops.refuse_above = None  # room on the very next pass: no backoff
    again = _ensure(ops, held.record, runner=A, now=T0 + timedelta(minutes=2))
    assert again.action == "restart" and len(again.record.restarts) == 2
    assert not runner_job.refused(again.record)
    ops.states[again.record.job] = "FAILED"  # it ran, then died: that start stays counted
    dead = _ensure(ops, again.record, runner=A, now=T0 + timedelta(minutes=20))
    assert dead.action == "restart" and len(dead.record.restarts) == 3


def test_runner_a_resizes_only_when_idle_after_two_passes_and_it_is_not_a_restart() -> None:
    ops = FakeOps(states={"a": "RUNNING"}, age=3)
    record = runner_job.Record(job="a", size=0)
    first = _ensure(ops, record, runner=A, demand=14)
    assert first.action == "ok" and first.record.resize_streak == 1 and not ops.cancelled
    ops.is_idle = False
    busy = _ensure(ops, first.record, runner=A, demand=14)
    assert busy.action == "ok" and not ops.cancelled
    ops.is_idle = True
    held = _ensure(ops, busy.record, runner=A, demand=14, hold=True)
    assert held.action == "ok" and not ops.cancelled  # growing waits while the disk guard holds
    resize = _ensure(ops, held.record, runner=A, demand=14)
    assert resize.action == "resize" and ops.cancelled == ["a"]
    ops.age = 4000
    # 2b waited while A was down, so the restart pass counts fewer lanes: A still takes the size
    # the resize was for, and shrinks only on a later want.
    again = _ensure(ops, resize.record, runner=A, demand=3, now=T0 + timedelta(minutes=10))
    assert again.action == "restart" and again.record.size == 2 and again.record.restarts == ()
    assert again.record.resize_to is None
    steady = _ensure(ops, again.record, runner=A, demand=3, now=T0 + timedelta(minutes=20))
    assert steady.action == "ok" and steady.record.resize_streak == 1  # wants size 0 now


def test_a_busy_runner_drains_to_shrink_after_two_passes_and_to_grow_after_six() -> None:
    ops = FakeOps(states={"a": "RUNNING"}, age=3, is_idle=False)
    record = runner_job.Record(job="a", size=2)
    first = _ensure(ops, record, runner=A, demand=0)
    assert first.record.resize_streak == 1 and not ops.drains  # one pass is not a want yet
    busy = _ensure(ops, first.record, runner=A, demand=0)
    assert busy.action == "ok" and busy.record.draining and ops.drains == [True]
    assert busy.detail.endswith("draining") and not ops.cancelled
    again = _ensure(ops, busy.record, runner=A, demand=0)
    assert ops.drains == [True]  # written once
    dry = _ensure(FakeOps(states={"a": "RUNNING"}, age=3, is_idle=False),
                  replace(again.record, draining=False), runner=A, demand=0, dry_run=True)
    assert not dry.record.draining
    ops.is_idle = True
    resize = _ensure(ops, again.record, runner=A, demand=0)
    assert resize.action == "resize" and ops.cancelled == ["a"]
    ops.age = 4000
    restart = _ensure(ops, resize.record, runner=A, demand=0, now=T0 + timedelta(minutes=10))
    assert restart.action == "restart" and restart.record.size == 0 and not restart.record.draining
    grow, record = FakeOps(states={"a": "RUNNING"}, age=3, is_idle=False), runner_job.Record(job="a", size=0)
    for _ in range(runner_job.GROW_DRAIN_AFTER - 1):
        record = _ensure(grow, record, runner=A, demand=14).record
    assert not grow.drains and not record.draining
    held = _ensure(grow, record, runner=A, demand=14, hold=True)  # growth held: no drain for it
    assert not grow.drains and not held.record.draining
    drained = _ensure(grow, record, runner=A, demand=14)
    assert grow.drains == [True] and drained.detail.endswith("passes; draining")
    gone = _ensure(grow, drained.record, runner=A, demand=0)  # back to its own size: drain ends
    assert grow.drains == [True, False] and not gone.record.draining and not grow.cancelled


def test_no_worklist_means_nothing_is_queued_or_launched(work: Path) -> None:
    cfg = dict(tick.DEFAULTS)
    assert tick_full.stage3_queue(DATA, cfg, dry_run=False) == {"skipped": "no worklist"}
    assert tick_full.top_up(DATA, cfg, dry_run=False) == {"skipped": "no worklist"}


def test_runner_a_runs_only_in_full_mode_and_fits_the_share_beside_b(work: Path) -> None:
    b, a = runner_job.DEFAULT_RUNNERS[:2]
    assert a.modes == ("full",) and a.spool.endswith("spool-A")
    assert b.cpus + a.sizes[-1][1] == 18 and b.memory_gib + a.sizes[-1][2] == 112
    collect = _tick(World(), "collect", state=_state_with_job())
    assert [r["runner"] for r in collect["results"]["runners"]] == ["B"]
    full = _tick(World(), "full", now=T0 + timedelta(minutes=10))
    assert [r["runner"] for r in full["results"]["runners"]] == ["B", "A"]


def test_the_cli_reads_the_mode_from_the_tick_folder_by_default() -> None:
    args = cli.build_parser().parse_args(["tick", "--dry-run"])
    assert args.mode is None and args.dry_run


def test_top_up_counts_a_five_family_batch_as_one_slot(work: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import implement

    launched = _worklist_world(work, monkeypatch, slots=1)
    monkeypatch.setattr(implement, "source_case", lambda family, *, data: "saved-pytorch")
    cfg = {**tick.DEFAULTS, "families_per_2b_lane": 5, "2b_on_hub": True}
    out = tick_full.top_up(DATA, cfg, dry_run=False)
    assert out["2b"] == ["fm2", "fm3", "fm4", "fm5"]
    assert out["2b_batches"] == [["fm2", "fm3", "fm4", "fm5"]]
    assert out["2a"] == [] and launched == [out["2b_batches"], []]
    assert tick.DEFAULTS["families_per_2b_lane"] == 1


def _tries_pass(work: Path, monkeypatch: pytest.MonkeyPatch, runners: dict[str, runner_job.Record],
                on_hub: bool = True) -> tuple[dict[str, Any], list[Any]]:
    """One full pass with the real top_up; each 2b launch writes its verb wrapper's text."""

    from menagerie.pipeline import implement

    launched = _worklist_world(work, monkeypatch, slots=10)
    (work / "config.json").write_text(json.dumps({"2b_on_hub": on_hub}))
    for key in implement.DECLARATION:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tick_full, "_launch_2b", lambda fams, data: [implement._wrapper(data) for _ in fams])
    world = World(ops=FakeOps(states={"j": "RUNNING", "a": "RUNNING"}))
    deps = world.deps()
    deps = replace(deps, full={k: v for k, v in deps.full.items() if k != "top_up"})
    tick.save_state({"runners": {k: v.to_mapping() for k, v in runners.items()}})
    return tick.tick("full", data=DATA, deps=deps, bound_s=None), launched


def test_top_up_gives_2b_lanes_runner_a_s_live_size(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    out, _ = _tries_pass(work, monkeypatch, {"B": runner_job.Record(job="j"),
                                             "A": runner_job.Record(job="a", size=2)})
    top = out["results"]["top_up"]
    assert "top_up" not in out["errors"] and "2b_held" not in top and top["2b"]
    wrappers = top["runs"][:len(top["2b"])]
    # size 2 is 9 CPUs and 56 GiB, not the largest size (12, 76) and not the hub's
    assert all("export MENAGERIE_CPUS='9'" in w and "export MENAGERIE_MEMORY_GB='56'" in w
               for w in wrappers)
    assert "MENAGERIE_CPUS" not in os.environ and "MENAGERIE_MEMORY_GB" not in os.environ


def test_top_up_holds_2b_lanes_while_runner_a_is_not_running(work: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    out, launched = _tries_pass(work, monkeypatch, {"B": runner_job.Record(job="j")})
    top = out["results"]["top_up"]
    assert "top_up" not in out["errors"] and top["2b"] == []
    assert top["2b_held"].startswith("runner A not running (") and "MISSING" in top["2b_held"]
    assert top["runs"] == ["s"] and launched == [top["2a"]]  # 2a lanes still start
    assert "runner A not running" in (work / "status.md").read_text()


def test_tick_keeps_2b_off_the_hub_by_default_and_says_so(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out, launched = _tries_pass(work, monkeypatch, {"B": runner_job.Record(job="j"),
                                                    "A": runner_job.Record(job="a", size=2)},
                                on_hub=False)
    top = out["results"]["top_up"]
    assert "top_up" not in out["errors"] and "2b_held" not in top
    assert top["2b"] == [] and top["waiting_2b"] == 4 and top["2a"]  # no workers configured
    assert top["runs"] == ["s"] and launched == [top["2a"]]
    status = (work / "status.md").read_text()
    assert "- 2b: workers only (2b_on_hub false: 2b runs model code; hub slots are kept for 2a)" in status


def test_the_tries_declaration_follows_the_live_job_or_says_why_not() -> None:
    live = {"runner": "A", "action": "ok", "job": "a", "size": 1, "job_state": "RUNNING"}
    assert runner_job.tries_declaration(live) == ({"MENAGERIE_CPUS": "6", "MENAGERIE_MEMORY_GB": "38"}, "")
    admitted = SimpleNamespace(allocation=lambda job: (9, 56))
    assert runner_job.tries_declaration({**live, "size": None}, admitted)[0] == {
        "MENAGERIE_CPUS": "9", "MENAGERIE_MEMORY_GB": "56"}
    unknown = SimpleNamespace(allocation=lambda job: None)
    assert runner_job.tries_declaration({**live, "size": None}, unknown) == (
        None, "runner A job a size unknown")
    for summary in ({**live, "action": "cancel-stale"}, {**live, "action": "resize"},
                    {**live, "job_state": "PENDING"}):
        env, why = runner_job.tries_declaration(summary, unknown)
        assert env is None and why.startswith("runner A not running")
    assert runner_job.tries_declaration(None) == (None, "runner A not read this pass")


def test_batch_launch_dispatch_and_held_keys(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import admission, implement

    calls = []
    monkeypatch.setattr(implement, "launch_batch", lambda families, **kw:
                        calls.append(families) or "r-batch")
    assert tick_full._launch_2b([["f-a", "f-b"]], DATA) == ["r-batch"]
    assert calls == [["f-a", "f-b"]]
    monkeypatch.setattr(admission, "lanes", lambda **kw: [
        {"kind": "2b", "keys": ["f-a", "f-b", "f-c", "f-d", "f-e"]}])
    monkeypatch.setattr(admission, "queued", list)
    count, held = tick_full._busy()
    assert count["2b"] == 1 and held == {"f-a", "f-b", "f-c", "f-d", "f-e"}


def test_tripwire_failure_holds_launching_but_keeps_the_rest_of_the_tick(work: Path) -> None:
    world = World()
    def broken(data: DataLayout, cfg: Mapping[str, Any], *, dry_run: bool) -> None:
        raise ValueError("bad cohort input")
    deps = world.deps()
    deps.full = {**deps.full, "tripwires": broken}
    tick.save_state(_state_with_job())
    out = tick.tick("full", dry_run=True, data=DATA, deps=deps, bound_s=None)
    assert out["results"]["top_up"] == {"held": "cohort tripwires could not be checked"}
    assert "tripwires" in out["errors"] and "trickle" in out["results"]
    assert out["results"]["stage3_queue"] == {"held": "cohort tripwires could not be checked"}


def test_tripwire_fact_survives_send_failure_and_status_lists_cohort_counts(
        work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"rows": 100, "settled": 100, "validated": 36, "completed_2a": 100, "research_gap": 0}
    monkeypatch.setattr(tick_full, "_cohort_counts", lambda data, cfg: {"r4-github": counts})
    world = World(send_fails=1)
    deps = world.deps()
    deps.full = {k: v for k, v in deps.full.items() if k != "tripwires"}
    tick.save_state(_state_with_job())
    (work / "config.json").write_text(json.dumps({"tripwires": True}))
    out = tick.tick("full", data=DATA, deps=deps, bound_s=None)
    assert not out["sent"]
    state = tick.load_state()
    assert len(state["outbox"]) == 1 and state["cohort_states"]["r4-github"]["stopped"]
    assert "cohort r4-github: 36/100 validated/settled" in (work / "status.md").read_text()
    out = tick.tick("full", data=DATA, deps=deps, bound_s=None)
    assert out["sent"] == ["cohort-r4-github"] and tick.load_state()["outbox"] == []
    tick.tick("full", data=DATA, deps=deps, bound_s=None)
    assert len(world.sent) == 1 and len(tick.load_state()["cohort_events"]) == 1


def test_tripwires_default_off_preserves_live_behavior(work: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("production tripwires were enabled without the switch")
    monkeypatch.setattr(tick_full, "tripwires", forbidden)
    deps = World().deps()
    deps.full = {k: v for k, v in deps.full.items() if k != "tripwires"}
    out = tick.tick("full", data=DATA, deps=deps, bound_s=None)
    assert out["results"]["tripwires"] == {"skipped": "cohort tripwires disabled"}
    assert "cohort_states" not in tick.load_state()


def test_top_up_reads_this_pass_s_disk_state(work: Path) -> None:
    """Worker lanes on the trace host follow the reading this pass took, never a stale one."""
    seen = []
    world = World()
    deps = world.deps()
    deps = replace(deps, full={**deps.full, "top_up": lambda data, cfg, dry_run: seen.append(cfg["disk_state"])})
    tick.tick("full", dry_run=False, data=DATA, deps=deps, bound_s=None)
    assert seen == ["ok"]


def test_a_failed_disk_reading_gives_worker_lanes_an_unknown_disk(work: Path) -> None:
    seen = []
    world = World()
    deps = world.deps()

    def broken() -> Reading:
        raise RuntimeError("no reading")

    deps = replace(deps, disk_reader=broken,
                   full={**deps.full, "top_up": lambda data, cfg, dry_run: seen.append(cfg["disk_state"])})
    out = tick.tick("full", dry_run=False, data=DATA, deps=deps, bound_s=None)
    assert seen == ["unknown"], (seen, out.get("errors"))


def test_big_runner_adds_spool_g_to_collection_and_keeps_big_work(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    @contextmanager
    def mine(name: str = "tick.lock"):
        yield True

    monkeypatch.setattr(tick, "locked", mine)
    seen: list[str] = []
    p = SimpleNamespace(cfg={"collect_spools": ["scratch/menagerie/spool-B"], "big_runner": True},
                        data=None, deps=SimpleNamespace(collect_spool=lambda s, d: seen.append(s) or {"spool": s}),
                        state={})
    tick._collect_spools(p)
    assert seen == ["scratch/menagerie/spool-B", "scratch/menagerie/spool-G"]
    seen.clear()
    p.cfg["big_runner"] = False
    tick._collect_spools(p)
    assert seen == ["scratch/menagerie/spool-B"]
    answers = iter([{"big": {"reserved": ["m1", "m2"], "waiting": 3}},
                    {"big": {"reserved": [], "held": "unavailable"}},
                    {"skipped": "no worklist"}])
    monkeypatch.setattr(tick, "_full_call", lambda p, name, held: next(answers))
    tick._step_stage3_queue(p)
    assert p.state["big_work"] == 5
    tick._step_stage3_queue(p)  # a held count keeps the last one
    tick._step_stage3_queue(p)
    assert p.state["big_work"] == 5


def test_reclaim_starts_below_its_line_once_an_hour_and_never_in_a_dry_run(work: Path, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".compute-host").write_text("worker-c\n")
    world = World(free=[70.0])
    deps = replace(world.deps(), repo=lambda d: repo)
    tick.tick("collect", dry_run=True, data=DATA, deps=deps, bound_s=None)
    assert world.launched == []
    out = tick.tick("collect", dry_run=False, data=DATA, deps=deps, bound_s=None)
    assert out["results"]["reclaim"]["started"]["free_gib"] == 70.0
    script, where = world.launched[0]
    assert where == repo and "run-on reap --host worker-c --project menagerie" in script
    assert "--failed-hours 2 --apply" in script and "bash menagerie/pipeline/prune_overlays_job.sh" in script
    later = replace(world.deps(T0 + timedelta(minutes=30)), repo=lambda d: repo)
    assert "skipped" in tick.tick("collect", dry_run=False, data=DATA, deps=later, bound_s=None)["results"]["reclaim"]
    assert len(world.launched) == 1


def test_reclaim_waits_above_its_line(work: Path, tmp_path: Path) -> None:
    (tmp_path / ".compute-host").write_text("worker-c\n")
    world = World(free=[80.0])
    out = tick.tick("collect", dry_run=False, data=DATA, deps=replace(world.deps(), repo=lambda d: tmp_path),
                    bound_s=None)
    assert out["results"]["reclaim"] == {"skipped": "free 80.0 GiB, line 75"} and world.launched == []



def _lane(work_root: Path, label: str, families: list[str], status: str | None = "no-result",
          age_h: float = 1.0, lease: bool = False) -> Path:
    import os

    folder = work_root / "minis" / label
    folder.mkdir(parents=True)
    paths = {str(work_root / "2b" / f): f"/w/{label}/.mini-lane/work/{f}" for f in families}
    paths[str(work_root / "lanes" / "run" / "src" / families[0])] = "/w/src"  # not a family folder
    (folder / "staged.json").write_text(json.dumps({"paths": paths}))
    if status:
        marker = folder / "ingested.json"
        marker.write_text(json.dumps({"status": status}))
        when = (T0 - timedelta(hours=age_h)).timestamp()
        os.utime(marker, (when, when))
    if lease:
        (work_root / "leases").mkdir(exist_ok=True)
        (work_root / "leases" / f"{label}.json").write_text("{}")
    return folder


def test_lane_reap_takes_ingested_committed_lanes_once_and_never_in_a_dry_run(work: Path) -> None:
    root = work.parent
    done = _lane(root, "2b-mini-a", ["f-a"])
    _lane(root, "2b-mini-b", ["f-b"], lease=True)  # its lease is still held: not yet released
    _lane(root, "2b-mini-c", ["f-c", "f-d"])  # f-d has a result the lanes step has not committed
    _lane(root, "2b-mini-e", ["f-e"], status="pull-failed")
    _lane(root, "2b-mini-g", ["f-g"], status=None)  # still running or not yet ingested
    _lane(root, "2b-mini-h", ["f-h"], age_h=60)  # left to run-on's own timer
    _lane(root, "2b-mini-k", ["f-k"])  # run-on does not hold it: already reaped, or unverified
    world = World(lane_dirty={"menagerie/data/adapters/f-d.py"}, leased={"f-d"},
                  held={"2b-mini-a", "2b-mini-b", "2b-mini-c", "2b-mini-e", "2b-mini-g", "2b-mini-h"})
    dry = _tick(world, dry_run=True, state=_state_with_job())["results"]["lane_reap"]
    assert dry["would_reap"] == ["2b-mini-a"] and world.reaped == []
    assert dry["waiting"] == {"lease": 1, "uncommitted": 1}
    out = _tick(world)["results"]["lane_reap"]
    assert world.reaped == ["2b-mini-a"] and out["reaped"] == 1
    assert json.loads((done / "reaped.json").read_text())["verdict"] == "reaped"
    _tick(world)
    assert world.reaped == ["2b-mini-a"]  # asked once
    world.lane_dirty, world.leased = set(), set()
    _tick(world)
    assert world.reaped == ["2b-mini-a", "2b-mini-c"]


def test_lane_reap_is_off_by_config(work: Path) -> None:
    _lane(work.parent, "2b-mini-a", ["f-a"])
    work.mkdir(parents=True, exist_ok=True)
    (work / "config.json").write_text(json.dumps({"reap_lanes": False}))
    world = World()
    assert _tick(world, state=_state_with_job())["results"]["lane_reap"] == {"skipped": "reap_lanes is off"}
    assert world.reaped == []
