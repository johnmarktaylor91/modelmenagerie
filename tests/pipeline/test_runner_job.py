"""Opt-in night runner transitions and projected-peak routing, with no live operations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from menagerie.pipeline import runner_job, tick

PT = ZoneInfo("America/Los_Angeles")


def clock(hour: int) -> datetime:
    return datetime(2026, 10, 9, hour, tzinfo=PT)


@dataclass
class Ops:
    states: dict[str, str] = field(default_factory=lambda: {"a": "RUNNING", "b": "RUNNING"})
    sizes: dict[str, tuple[int, int]] = field(default_factory=lambda: {"a": (12, 76), "b": (6, 36)})
    spools: dict[str, str] = field(default_factory=lambda: {"a": "A", "b": "B"})
    submitted: list[runner_job.Runner] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    busy: set[str] = field(default_factory=set)
    pending: bool = False
    fresh_dead: bool = False
    fed: set[str] = field(default_factory=set)  # runners whose queue refills: busy unless drained
    drained: dict[str, bool] = field(default_factory=dict)

    def job_state(self, job: str) -> str:
        return self.states.get(job, "MISSING")

    def allocation(self, job: str) -> tuple[int, int] | None:
        return self.sizes.get(job)

    def heartbeat_age(self, runner: runner_job.Runner) -> float | None:
        if self.fresh_dead or any(self.job_state(j) == "RUNNING" for j, n in self.spools.items()
                                  if n == runner.name):
            return 5.0
        return None

    def discover(self, runner: runner_job.Runner) -> list[tuple[str, str | None]]:
        return [(j, None) for j, n in self.spools.items() if n == runner.name]

    def stopped(self, runner: runner_job.Runner) -> bool:
        return False

    def idle(self, runner: runner_job.Runner) -> bool:
        fed = runner.name in self.fed and not self.drained.get(runner.name)
        return runner.name not in self.busy and not fed

    def drain(self, runner: runner_job.Runner, on: bool) -> None:
        self.drained[runner.name] = on

    def cancel(self, job: str) -> None:
        self.cancelled.append(job)
        self.states[job] = "INTERRUPTED"

    def submit(self, runner: runner_job.Runner, env: dict[str, str]) -> str:
        assert env["RUNNER_SPOOL"] == runner.spool
        self.submitted.append(runner)
        self.drained.pop(runner.name, None)  # the job script deletes the spool's DRAIN
        job = f"new-{runner.name}-{len(self.submitted)}"
        self.states[job] = "PENDING" if self.pending else "RUNNING"
        self.sizes[job] = (runner.cpus, runner.memory_gib)
        self.spools[job] = runner.name
        self.check_share()
        return job

    def check_share(self) -> None:
        active = [self.sizes[j] for j, s in self.states.items() if s in ("RUNNING", "PENDING")]
        assert sum(c for c, _ in active) <= 18
        assert sum(g for _, g in active) <= 112


@pytest.fixture
def policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    cfg = {"big_runner": True}
    monkeypatch.setattr(tick, "config", lambda: cfg)
    return cfg


def run(ops: Ops, state: dict[str, Any], hour: int, **kw: Any) -> list[dict[str, Any]]:
    result = runner_job.ensure(ops=ops, state=state, mode="full", demand={"2b": 20},
                               now=clock(hour), **kw)
    state["runners"] = {r["runner"]: r["record"] for r in result}
    ops.check_share()
    for r in result:
        if r.get("share") is not None:
            assert r["share"]["cpus"] <= 18 and r["share"]["memory_gib"] <= 112
    return result


@pytest.mark.parametrize("hour,opened", [(21, False), (22, True), (23, True),
                                         (0, True), (5, True), (6, False)])
def test_window(hour: int, opened: bool) -> None:
    assert runner_job.night_open(clock(hour)) is opened


def test_switch_off_preserves_live_runners_and_spools(policy: dict[str, Any]) -> None:
    policy["big_runner"] = False
    ops = Ops()
    state = {"runners": {"A": runner_job.Record(job="a", size=3).to_mapping(),
                         "B": runner_job.Record(job="b").to_mapping()}}
    for hour in (22, 6, 12):
        result = run(ops, state, hour)
        assert [r["runner"] for r in result] == ["B", "A"]
    assert not ops.submitted and not ops.cancelled
    assert ops.spools == {"a": "A", "b": "B"}


def test_downsize_confirm_start_and_drain_before_restoring(policy: dict[str, Any]) -> None:
    ops, state = Ops(), {}
    run(ops, state, 22)  # first demand pass
    run(ops, state, 22)  # cancel idle A/B; do not start G
    assert set(ops.cancelled) == {"a", "b"} and not ops.submitted
    run(ops, state, 22)  # submit small A/B
    assert [r.name for r in ops.submitted] == ["B", "A"]
    run(ops, state, 22)  # confirmed small A/B: start G
    assert ops.submitted[-1].name == "G"
    assert ops.submitted[-1].workers == 1
    assert (ops.submitted[-1].cpus, ops.submitted[-1].memory_gib) == (6, 40)
    steady = run(ops, state, 22)
    assert all(r["planned_share"] == {"cpus": 12, "memory_gib": 76} for r in steady)
    ops.busy.add("G")
    run(ops, state, 6)
    assert len(ops.cancelled) == 2  # drain, do not interrupt a row
    ops.busy.clear()
    run(ops, state, 6)
    assert len(ops.cancelled) == 3  # G cancelled, no A/B growth on this pass
    run(ops, state, 6)
    run(ops, state, 6)
    run(ops, state, 6)
    assert ops.submitted[-2].name == "B" and ops.submitted[-2].memory_gib == 36
    assert ops.submitted[-1].name == "A" and ops.submitted[-1].memory_gib == 76


def test_g_replaces_b_at_night_and_b_returns_in_the_morning(policy: dict[str, Any]) -> None:
    policy["g_replaces_b"] = True
    ops, state = Ops(fed={"B"}), {}
    first = run(ops, state, 22)  # B busy: drained, never interrupted
    assert ops.drained.get("B") is True and "b" not in ops.cancelled
    assert "give way" in next(r for r in first if r["runner"] == "B")["detail"]
    for _ in range(4):
        run(ops, state, 22)
    assert "b" in ops.cancelled and state["runners"]["B"]["cancelled_for"] == "g"
    assert [r.name for r in ops.submitted] == ["A", "G"]  # A at its night size, then G; no B
    assert (ops.submitted[0].memory_gib, ops.submitted[1].memory_gib) == (20, 40)
    steady = run(ops, state, 23)
    assert [r.name for r in ops.submitted] == ["A", "G"]  # B stays off while G runs
    assert next(r for r in steady if r["runner"] == "B")["action"] == "off"
    for _ in range(3):
        run(ops, state, 6)  # morning: G drains and stops, and B comes back
    assert state["runners"]["G"]["cancelled"] and ops.submitted[-1].name == "B"


def test_g_replaces_b_only_while_big_rows_wait(policy: dict[str, Any]) -> None:
    policy["g_replaces_b"] = True
    ops, state = Ops(), {"big_work": 0}
    for _ in range(3):
        run(ops, state, 23)
    assert "b" not in ops.cancelled and not any(r.name == "G" for r in ops.submitted)


@dataclass
class CrowdedOps(Ops):
    """Another project's run-on claim leaves room for A or G beside B, not both (worker-c, 26 Oct)."""

    room: bool = True

    def probe(self, runner: runner_job.Runner, replacing: str | None) -> tuple[int, str]:
        other = {"A": "G", "G": "A"}.get(runner.name)
        if other and not self.room and any(self.states.get(j) == "RUNNING" for j, n in self.spools.items()
                                           if n == other):
            return runner_job.PROBE_REFUSED, "NO_CAPACITY: memory declarations exceed usable RAM"
        return runner_job.PROBE_ADMIT, ""


def started_g_with_a_down(ops: CrowdedOps, state: dict[str, Any]) -> None:
    for _ in range(4):  # small A/B, then G beside them
        run(ops, state, 22)
    assert ops.submitted[-1].name == "G"
    ops.states[state["runners"]["A"]["job"]] = "INTERRUPTED"  # A is down with G running


def test_a_down_with_no_room_beside_g_g_finishes_its_row_drains_and_a_returns(
        policy: dict[str, Any]) -> None:
    ops, state = CrowdedOps(), {}
    started_g_with_a_down(ops, state)
    ops.room, g_job = False, state["runners"]["G"]["job"]
    ops.busy.add("G")
    first = run(ops, state, 23)  # G has a row in progress: drained, not cancelled
    assert ops.drained["G"] is True and g_job not in ops.cancelled
    assert "no room beside G" in next(r for r in first if r["runner"] == "A")["detail"]
    ops.busy.clear()
    run(ops, state, 23)  # G idle: cancelled, A still held this pass
    assert ops.cancelled[-1] == g_job and ops.submitted[-1].name == "G"
    run(ops, state, 23)  # A returns at its night size
    assert ops.submitted[-1].name == "A" and ops.submitted[-1].memory_gib == 20
    a_job = state["runners"]["A"]["job"]
    for _ in range(4):  # G cannot fit beside A: G waits, A is never cancelled for it
        run(ops, state, 0)
    assert a_job not in ops.cancelled and state["runners"]["A"]["job"] == a_job
    assert ops.submitted[-1].name == "A"


def test_a_returns_beside_g_the_pass_room_appears_and_g_claims_rows_again(policy: dict[str, Any]) -> None:
    ops, state = CrowdedOps(), {}
    started_g_with_a_down(ops, state)
    ops.room, g_job = False, state["runners"]["G"]["job"]
    ops.busy.add("G")
    run(ops, state, 23)
    assert ops.drained["G"] is True
    ops.room = True  # the other project's job ended
    run(ops, state, 23)  # A back at once, no back-off; G keeps its row
    assert ops.submitted[-1].name == "A" and g_job not in ops.cancelled
    run(ops, state, 23)
    assert ops.drained["G"] is False and state["runners"]["G"]["draining"] is False
    assert state["runners"]["G"]["job"] == g_job


def test_a_busy_b_whose_queue_refills_is_drained_so_g_starts(policy: dict[str, Any]) -> None:
    ops, state = Ops(fed={"B"}), {}
    run(ops, state, 22)  # first demand pass
    second = run(ops, state, 22)  # A idle: cancelled; B busy: drained, not cancelled
    assert ops.cancelled == ["a"] and ops.drained == {"B": True}
    assert "draining" in next(r for r in second if r["runner"] == "B")["detail"]
    run(ops, state, 22)  # drained B idles: cancelled for the resize
    assert ops.cancelled == ["a", "b"]
    for _ in range(3):
        run(ops, state, 22)
    assert [r.name for r in ops.submitted] == ["A", "B", "G"]
    assert ops.submitted[1].memory_gib == 16 and not ops.drained
    assert state["runners"]["B"]["draining"] is False


def test_a_drain_ends_when_the_shrink_is_no_longer_wanted(policy: dict[str, Any]) -> None:
    ops, state = Ops(fed={"B"}), {}
    run(ops, state, 22)
    run(ops, state, 22)
    assert ops.drained == {"B": True} and state["runners"]["B"]["draining"] is True
    run(ops, state, 12)  # day again before B idled: size 1 is wanted, B claims again
    assert ops.drained == {"B": False} and state["runners"]["B"]["draining"] is False
    assert "b" not in ops.cancelled


def test_g_is_drained_before_its_morning_stop(policy: dict[str, Any]) -> None:
    ops, state = Ops(), {}
    for _ in range(4):
        run(ops, state, 22)
    assert ops.submitted[-1].name == "G"
    ops.fed.add("G")
    held = run(ops, state, 6)
    assert ops.drained == {"G": True} and len(ops.cancelled) == 2
    assert next(r for r in held if r["runner"] == "G")["record"]["draining"] is True
    run(ops, state, 6)  # drained G idles: cancelled
    assert len(ops.cancelled) == 3


def test_a_flickering_night_want_never_restarts_b_and_the_day_want_drains_it_to_grow(
        policy: dict[str, Any]) -> None:
    ops, state = Ops(), {}
    for _ in range(4):
        run(ops, state, 22)
    assert ops.submitted[-1].name == "G"
    ops.fed.add("B")  # B's stage 3 queue never empties by itself
    cancelled = len(ops.cancelled)
    for _ in range(3):  # G off and on with the big rows, as on 2026-10-22
        state["big_work"] = 0
        for _ in range(5):
            run(ops, state, 23)
        state["big_work"] = 2
        for _ in range(3):
            run(ops, state, 23)
    assert not any(ops.drained.values())
    assert [j for j in ops.cancelled[cancelled:] if ops.spools[j] == "B"] == []
    details = [next(r for r in run(ops, state, 7) if r["runner"] == "B")["detail"]
               for _ in range(runner_job.GROW_DRAIN_AFTER + 3)]
    drained = [d for d in details if d.endswith("draining")]
    assert drained == [f"wants size 1 (now 0), {runner_job.GROW_DRAIN_AFTER} passes; draining"]
    assert any("idle; size 0 to 1" in d for d in details)  # drained B idles: resized
    assert ops.submitted[-1].name == "B" and ops.submitted[-1].memory_gib == 36


def test_a_growth_hold_keeps_b_small_until_its_time_then_the_ladder_grows_it(
        policy: dict[str, Any]) -> None:
    ops, state = Ops(), {}
    for _ in range(4):
        run(ops, state, 22)
    assert ops.submitted[-1].name == "G"
    state["big_work"] = 0  # G drains and stops in the morning; the day wants B at size 1
    ops.fed.add("B")
    state["runners"]["B"]["hold_growth_until"] = clock(9).isoformat()
    submitted = len(ops.submitted)
    cancelled = [j for j in ops.cancelled if ops.spools[j] == "B"]
    details = [next(r for r in run(ops, state, 7) if r["runner"] == "B")["detail"]
               for _ in range(runner_job.GROW_DRAIN_AFTER + 6)]
    assert details[-1] == f"wants size 1 (now 0); growth held until {clock(9).isoformat()}"
    assert [j for j in ops.cancelled if ops.spools[j] == "B"] == cancelled
    assert not ops.drained.get("B") and "B" not in [r.name for r in ops.submitted[submitted:]]
    details = [next(r for r in run(ops, state, 9) if r["runner"] == "B")["detail"]
               for _ in range(runner_job.GROW_DRAIN_AFTER + 3)]
    assert any("idle; size 0 to 1" in d for d in details)  # the hold ended: drained, then resized
    assert ops.submitted[-1].name == "B" and ops.submitted[-1].memory_gib == 36


def test_unknown_adopted_size_and_pending_small_runner_hold_g(policy: dict[str, Any]) -> None:
    ops, state = Ops(), {}
    ops.sizes.pop("a")
    result = runner_job.ensure(ops=ops, state=state, mode="full", demand={"2b": 20}, now=clock(22))
    assert result[1]["action"] == "held-big"
    assert not ops.submitted
    ops = Ops(pending=True)
    for _ in range(5):
        run(ops, state, 22)
    assert all(r.name != "G" for r in ops.submitted)


def test_a_live_a_held_from_growth_still_gives_2b_lanes_its_size(policy: dict[str, Any]) -> None:
    ops = Ops(sizes={"a": (6, 38), "b": (6, 36), "g": (6, 40)})
    ops.states["g"], ops.spools["g"] = "PENDING", "G"  # daytime, G unresolved: A/B growth held
    result = runner_job.ensure(ops=ops, state={}, mode="full", demand={"2b": 20}, now=clock(12))
    a = next(r for r in result if r["runner"] == "A")
    assert a["action"] == "held-big" and a["job_state"] == "RUNNING" and a["size"] == 1
    assert runner_job.tries_declaration(a, ops) == ({"MENAGERIE_CPUS": "6", "MENAGERIE_MEMORY_GB": "38"}, "")
    ops.states["a"] = "PENDING"
    a = next(r for r in runner_job.ensure(ops=ops, state={}, mode="full", demand={"2b": 20},
                                          now=clock(12)) if r["runner"] == "A")
    assert runner_job.tries_declaration(a, ops)[0] is None


def test_a_busy_large_runner_never_allows_g(policy: dict[str, Any]) -> None:
    ops, state = Ops(busy={"A"}), {}
    for _ in range(6):
        run(ops, state, 22)
    assert "a" not in ops.cancelled
    assert all(r.name != "G" for r in ops.submitted)


def test_dry_run_never_mutates_jobs(policy: dict[str, Any]) -> None:
    ops, state = Ops(), {}
    run(ops, state, 22, dry_run=True)
    result = run(ops, state, 22, dry_run=True)
    assert [r["action"] for r in result[:2]] == ["resize", "resize"]
    assert not ops.cancelled and not ops.submitted


def test_fresh_g_heartbeat_after_dead_job_holds_restoration(policy: dict[str, Any]) -> None:
    ops, state = Ops(), {}
    for _ in range(4):
        run(ops, state, 22)
    run(ops, state, 6)  # cancel G
    ops.fresh_dead = True
    result = run(ops, state, 6)
    assert all(r["action"] == "held-big" for r in result)
    assert len(ops.cancelled) == 3


def test_g_declaration_is_six_cpus_40_gib_and_bounds_its_spool() -> None:
    from menagerie.pipeline import flow

    b, a, g = runner_job.DEFAULT_RUNNERS
    assert (g.cpus, g.memory_gib, g.workers) == (6, 40, 1)
    assert a.sizes[-1] == (4, 12, 76)  # A's own ladder is not G's
    assert flow.BIG_GIB == g.memory_gib
    night = [b.sizes or runner_job.B_SIZES[0], a.sizes[0], (1, g.cpus, g.memory_gib)]
    assert sum(s[1] for s in night) == 12 and sum(s[2] for s in night) == 76


def test_shell_and_job_env_leave_b_unchanged() -> None:
    b, _, g = runner_job.DEFAULT_RUNNERS
    assert b.sizes == () and (b.cpus, b.memory_gib, b.workers) == (6, 36, 8)
    assert runner_job.job_env(g, pause_gib=None)["RUNNER_SPOOL"].endswith("spool-G")


def test_b_runs_eight_workers_by_day_in_the_cpus_and_memory_two_had(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import runner

    b = runner_job.DEFAULT_RUNNERS[0]
    assert (b.workers, b.cpus, b.memory_gib) == runner_job.B_SIZES[runner_job.B_DAY] == (8, 6, 36)
    env = runner_job.job_env(b, pause_gib=None)
    assert env["RUNNER_WORKERS"] == "8"
    for key in ("MENAGERIE_CPUS", "MENAGERIE_MEMORY_GB"):
        monkeypatch.setenv(key, env[key])
    assert runner.default_caps("trace").rss_mb == 12 * 1024  # a request's default cap is unchanged
    sizes = [(cpus, gib) for _, cpus, gib in runner_job.B_SIZES]
    assert len(set(sizes)) == len(sizes)  # _observed tells sizes apart by CPUs and memory


@dataclass
class BoostOps(Ops):
    """run-on refuses B's 46 GiB boost while ``refuse_boost`` is set."""

    refuse_boost: bool = False
    tight: bool = False  # no room for the boost's margin beside it
    probes: list[tuple[str, int]] = field(default_factory=list)

    def probe(self, runner: runner_job.Runner, replacing: str | None) -> tuple[int, str]:
        self.probes.append((runner.name, runner.memory_gib))
        if self.tight and runner.name == "B" and runner.memory_gib > 46:
            return runner_job.PROBE_REFUSED, "NO_CAPACITY: available RAM cannot cover commitments"
        if self.refuse_boost and runner.name == "B" and runner.memory_gib == 46:
            return runner_job.PROBE_REFUSED, "NO_CAPACITY: available RAM cannot cover commitments"
        return runner_job.PROBE_ADMIT, ""


def day(ops: Ops, state: dict[str, Any], live_2b: int, passes: int) -> list[str]:
    details = []
    for _ in range(passes):
        result = runner_job.ensure(ops=ops, state=state, mode="full", demand={"2b": live_2b},
                                   now=clock(12))
        state["runners"] = {r["runner"]: r["record"] for r in result}
        ops.check_share()
        details.append(next(r for r in result if r["runner"] == "B")["detail"])
    return details


def test_b_keeps_its_day_size_while_the_boost_is_switched_off(policy: dict[str, Any]) -> None:
    ops = BoostOps(sizes={"a": (3, 20), "b": (6, 36)})
    details = day(ops, {"big_work": 1}, 0, runner_job.GROW_DRAIN_AFTER + 2)
    assert not ops.cancelled and not ops.drained.get("B") and "wants" not in details[-1]
    assert not any(name == "B" for name, _ in ops.probes)


def test_b_grows_to_its_day_boost_by_a_drain_only_when_the_probe_admits(
        policy: dict[str, Any]) -> None:
    policy["b_boost"] = True
    ops = BoostOps(sizes={"a": (3, 20), "b": (6, 36)}, fed={"B"}, refuse_boost=True)
    state: dict[str, Any] = {"big_work": 1}
    details = day(ops, state, 0, runner_job.GROW_DRAIN_AFTER + 2)
    assert "run-on probe refused, keeps b" in details[-1]
    assert not ops.drained.get("B") and "b" not in ops.cancelled  # B keeps its day size
    ops.refuse_boost = False
    details = day(ops, state, 0, 3)
    assert details[0].endswith("; draining")  # drained first, never interrupted mid-row
    assert any("idle; size 1 to 2" in d for d in details) and ops.cancelled == ["b"]
    b = ops.submitted[-1]  # the probe admitted the boost
    assert (b.name, b.workers, b.cpus, b.memory_gib) == ("B", 10, 9, 46)
    assert runner_job.job_env(b, pause_gib=None)["RUNNER_WORKERS"] == "10"


def test_b_gives_its_boost_back_when_a_wants_its_largest_size(policy: dict[str, Any]) -> None:
    policy["b_boost"] = True
    ops = BoostOps(sizes={"a": (3, 20), "b": (9, 46)})
    state: dict[str, Any] = {"big_work": 1}
    assert runner_job._b_day_workers(runner_job.DEFAULT_RUNNERS[1], 15) == 10  # A at 9 CPUs fits
    day(ops, state, 0, 1)
    assert state["runners"]["B"]["size"] == 2 and not ops.cancelled
    details = day(ops, state, 20, runner_job.RESIZE_AFTER)  # A wants 12 CPUs: B back to 6
    assert "b" in ops.cancelled and "size 2 to 1" in details[-1]
    day(ops, state, 20, 1)
    assert [(r.name, r.cpus) for r in ops.submitted] == [("B", 6), ("A", 12)]
    # lanes keep the room B's day size leaves: the boost is never counted against them
    assert runner_job.lane_room(0) == (18 - 3 - 6, 112 - 20 - 36)


def test_a_boosted_b_gives_its_room_back_when_lanes_would_have_none(policy: dict[str, Any]) -> None:
    policy["b_boost"] = True
    ops = BoostOps(sizes={"a": (3, 20), "b": (9, 46)})
    state: dict[str, Any] = {"big_work": 1}
    day(ops, state, 0, 2)
    assert ("B", 46 + runner_job.BOOST_MARGIN_GIB) in ops.probes and not ops.cancelled
    ops.tight = True  # run-on would admit no 10 GiB more beside B: lanes could be refused
    details = day(ops, state, 0, runner_job.RESIZE_AFTER + 1)
    assert ops.cancelled == ["b"] and any("size 2 to 1" in d for d in details)
    assert (ops.submitted[-1].cpus, ops.submitted[-1].memory_gib) == (6, 36)
    day(ops, state, 0, runner_job.GROW_DRAIN_AFTER + 2)  # still tight: B stays at its day size
    assert (ops.submitted[-1].cpus, ops.submitted[-1].memory_gib) == (6, 36)


def test_dead_small_runner_restarts_at_minimum_during_g_drain(policy: dict[str, Any]) -> None:
    ops, state = Ops(), {}
    for _ in range(4):
        run(ops, state, 22)
    ops.busy.add("G")
    ops.states[state["runners"]["A"]["job"]] = "INTERRUPTED"
    result = run(ops, state, 6)
    assert next(r for r in result if r["runner"] == "A")["action"] == "restart"
    assert ops.submitted[-1].name == "A" and ops.submitted[-1].memory_gib == 20
    assert len(ops.cancelled) == 2


def test_switch_off_after_night_restores_legacy_spool_selection(policy: dict[str, Any]) -> None:
    ops, state = Ops(), {}
    for _ in range(4):
        run(ops, state, 22)
    policy["big_runner"] = False
    for _ in range(5):
        run(ops, state, 6)
    before = len(ops.submitted), len(ops.cancelled)
    result = runner_job.ensure(ops=ops, state=state, mode="collect", spool="scratch/menagerie/spool-B",
                               now=clock(6))
    assert [r["runner"] for r in result] == ["B"] and result[0]["action"] == "ok"
    assert (len(ops.submitted), len(ops.cancelled)) == before


def test_unknown_declaration_holds_daytime_restoration(
    policy: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops, state = Ops(), {}
    allocation = ops.allocation
    monkeypatch.setattr(ops, "allocation", lambda job: None if job == "a" else allocation(job))
    ops.sizes["b"] = (3, 16)
    for _ in range(4):
        result = run(ops, state, 6)
        assert all(r["action"] == "held-big" for r in result)
    assert not ops.cancelled and not ops.submitted


@pytest.mark.parametrize("admitted,expected", [
    (76, (6, 76)), (None, (6, 76)), (76.5, None), (0, None), (-1, None),
])
def test_manifest_allocation_never_truncates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, admitted: float | None,
    expected: tuple[int, int] | None,
) -> None:
    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps(
        {"cpus": 6, "memory_gib": 76, "memory_admitted_gib": admitted}))
    monkeypatch.setattr(runner_job, "_jobs_mirror", lambda: tmp_path)
    assert runner_job.RunOn().allocation("job") == expected


@pytest.mark.parametrize("value", [None, [], 76, "unknown"])
def test_nonobject_manifest_holds_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: object,
) -> None:
    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps(value))
    monkeypatch.setattr(runner_job, "_jobs_mirror", lambda: tmp_path)
    assert runner_job.RunOn().allocation("job") is None


def test_big_work_and_lead_in_window() -> None:
    assert runner_job.big_work({}) is True  # unknown count: G stays available
    assert runner_job.big_work({"big_work": 0}) is False
    assert runner_job.big_work({"big_work": 3}) is True
    assert not runner_job.big_hold_open(datetime(2026, 10, 9, 21, 29, tzinfo=PT))
    assert runner_job.big_hold_open(datetime(2026, 10, 9, 21, 30, tzinfo=PT))
    assert runner_job.big_hold_open(datetime(2026, 10, 10, 5, 59, tzinfo=PT))
    assert not runner_job.big_hold_open(datetime(2026, 10, 10, 6, 0, tzinfo=PT))


def test_a_night_with_no_big_rows_keeps_a_and_b_at_day_width(policy: dict[str, Any]) -> None:
    ops, state = Ops(), {"big_work": 0}
    for _ in range(5):
        run(ops, state, 22)
    assert not ops.cancelled and not ops.submitted


def test_g_drains_when_its_rows_run_out_before_morning(policy: dict[str, Any]) -> None:
    ops, state = Ops(), {"big_work": 4}
    for _ in range(4):
        run(ops, state, 22)
    assert ops.submitted[-1].name == "G"
    state["big_work"] = 0
    ops.busy.add("G")
    run(ops, state, 23)
    assert len(ops.cancelled) == 2  # G still on a row: drain, do not interrupt it
    ops.busy.clear()
    run(ops, state, 23)
    assert len(ops.cancelled) == 3  # G cancelled; A/B grow on later passes
    for _ in range(3):
        run(ops, state, 23)
    assert ops.submitted[-1].name == "A" and ops.submitted[-1].memory_gib == 76
