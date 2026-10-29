"""The tick's describe stage: its knobs, one detached run at a time, the run's record, the switch."""

from __future__ import annotations

import json
import subprocess

import pytest

from menagerie.pipeline import DataLayout, describe_stage, describe_write, tick_full


class _Systemd:
    """Stands in for ``systemctl --user is-active`` and ``systemd-run``."""

    def __init__(self, active: bool = False, rc: int = 0) -> None:
        self.active, self.rc, self.calls = active, rc, []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[0] == "systemctl":
            return subprocess.CompletedProcess(argv, 0 if self.active else 3, "", "")
        return subprocess.CompletedProcess(argv, self.rc, "", "unit exists" if self.rc else "")

    def started(self) -> list[list[str]]:
        return [c for c in self.calls if c[0] == "systemd-run"]


@pytest.fixture(autouse=True)
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path / "work"))
    monkeypatch.setenv("MENAGERIE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    monkeypatch.delenv("MENAGERIE_LANES_DESCRIBE", raising=False)


def test_knobs_default_and_never_exceed_the_describe_maximum():
    assert describe_stage.knobs({}) == {
        "MENAGERIE_LANES_DESCRIBE": "4", describe_write.HUB_RESERVE: "6",
        "MENAGERIE_DESCRIBE_MAX_PCT": "60", describe_write.EARLIER_PINS: "1",
        describe_write.ABSTRACTS: ""}
    words = describe_stage.knobs({"lanes": 9, "hub_reserve": 2, "earlier_pins": False})
    assert words["MENAGERIE_LANES_DESCRIBE"] == "4" and words[describe_write.HUB_RESERVE] == "2"
    assert words[describe_write.EARLIER_PINS] == "0"
    with pytest.raises(TypeError, match="earlier_pins"):
        describe_stage.knobs({"earlier_pins": "yes"})
    assert describe_stage.knobs({"abstracts": True})[describe_write.ABSTRACTS] == "all"
    assert describe_stage.knobs({"abstracts": ["f-a", "f-b"]})[describe_write.ABSTRACTS] == "f-a,f-b"
    with pytest.raises(TypeError, match="abstracts"):
        describe_stage.knobs({"abstracts": "f-a,f-b"})


def test_a_pass_starts_one_unit_with_its_knobs_and_the_tick_environment(monkeypatch):
    monkeypatch.setenv("MENAGERIE_DATA", "/data")
    monkeypatch.setenv("UNRELATED_SECRET_LOOKING", "x")
    systemd = _Systemd()
    out = describe_stage.stage({"lanes": 2}, dry_run=False, run=systemd)
    [argv] = systemd.started()
    assert out["started"] == describe_stage.UNIT and f"--unit={describe_stage.UNIT}" in argv
    assert "--setenv=MENAGERIE_LANES_DESCRIBE=2" in argv and "--setenv=MENAGERIE_DATA=/data" in argv
    assert not any("UNRELATED" in a for a in argv)
    assert argv[-1] == "exec python3 -m menagerie.pipeline.describe_stage"


def test_an_active_unit_a_dry_run_and_a_refused_start_start_nothing():
    active = _Systemd(active=True)
    assert describe_stage.stage({}, dry_run=False, run=active, slots=lambda: 20)["running"] == describe_stage.UNIT
    assert not active.started()
    idle = _Systemd()
    assert "would_start" in describe_stage.stage({}, dry_run=True, run=idle, slots=lambda: 20)
    assert not idle.started()
    refused = _Systemd(rc=1)
    assert describe_stage.stage({}, dry_run=False, run=refused, slots=lambda: 20)["held"].startswith("systemd-run rc 1")


def test_a_run_records_its_verdicts_and_a_failure(monkeypatch):
    rows = [{"family": "f-a", "verdict": "launched", "run": "r1"},
            {"family": "f-b", "verdict": "launched", "run": "r1"},
            {"family": "f-c", "verdict": "grounded", "written": True},
            {"family": "f-d", "verdict": "waiting"},
            {"family": "f-e", "verdict": "waiting", "why": {"result": "hub_full", "why": "no free describe slot"}}]
    monkeypatch.setattr(describe_write, "write", lambda families, **kw: rows)
    assert describe_stage.main() == 0
    last = describe_stage._last()
    assert last["verdicts"] == {"launched": 2, "grounded": 1, "waiting": 2}
    assert last["held"] == {"hub_full: no free describe slot": 1}
    assert last["launched"] == ["r1"] and last["written"] == ["f-c"]

    def broken(families, **kw):
        raise OSError("store unreachable")

    monkeypatch.setattr(describe_write, "write", broken)
    assert describe_stage.main() == 1
    assert describe_stage._last()["error"] == "OSError: store unreachable"
    history = describe_stage._state().with_suffix(".jsonl").read_text().splitlines()
    assert [("error" in json.loads(line)) for line in history] == [False, True]
    assert "last" in describe_stage.stage({}, dry_run=True, run=_Systemd(), slots=lambda: 20)


def test_the_tick_step_runs_only_when_switched_on(tmp_path, monkeypatch):
    data = DataLayout(tmp_path / "data")
    assert "skipped" in tick_full.describes(data, {}, dry_run=False)
    assert "skipped" in tick_full.describes(data, {"describes": {"enabled": False}}, dry_run=False)
    seen = []
    monkeypatch.setattr(describe_stage, "stage", lambda given, dry_run: seen.append(given) or {})
    tick_full.describes(data, {"describes": {"enabled": True, "lanes": 3}}, dry_run=True)
    assert seen == [{"lanes": 3}]


def test_no_run_starts_while_every_seat_is_staged_and_the_hub_is_full():
    root = describe_write.work_dir(describe_write.KIND)
    for n in range(2):
        (root / f"seat{n}").mkdir(parents=True)
        (root / f"seat{n}" / "families.json").write_text("[]")
        (root / f"seat{n}" / "ready.json").write_text("{}")
    systemd = _Systemd()
    out = describe_stage.stage({"lanes": 2}, dry_run=False, run=systemd, slots=lambda: 6)
    assert out["held"].startswith("every seat is staged") and not systemd.started()
    assert "started" in describe_stage.stage({"lanes": 2}, dry_run=False, run=systemd, slots=lambda: 7)
    # a seat still to reserve, or a lane to start or collect: always a run
    assert "started" in describe_stage.stage({"lanes": 3}, dry_run=False, run=_Systemd(), slots=lambda: 0)
    (root / "seat0" / "lane.json").write_text("{}")
    assert "started" in describe_stage.stage({"lanes": 2}, dry_run=False, run=_Systemd(), slots=lambda: 0)


def test_the_worker_switch_adds_its_seat_cap_and_off_changes_nothing():
    from menagerie.pipeline import describe_workers

    off = describe_stage.knobs({})
    assert describe_stage.knobs({"workers": False, "worker_seats": 2}) == off
    assert describe_workers.WORKERS not in off
    assert describe_stage.knobs({"workers": True})[describe_workers.WORKERS] == "4"
    assert describe_stage.knobs({"workers": True, "worker_seats": 2})[describe_workers.WORKERS] == "2"
    assert describe_stage.knobs({"workers": True, "lanes": 3, "worker_seats": 9})[describe_workers.WORKERS] == "3"
    with pytest.raises(TypeError, match="workers"):
        describe_stage.knobs({"workers": "yes"})


def test_worker_room_starts_a_run_while_the_hub_is_full():
    from menagerie.pipeline import describe_workers

    root = describe_write.work_dir(describe_write.KIND)
    for n in range(2):
        (root / f"seat{n}").mkdir(parents=True)
        (root / f"seat{n}" / "families.json").write_text("[]")
        (root / f"seat{n}" / "ready.json").write_text("{}")
    describe_write.work_dir("leases").mkdir(parents=True)
    describe_workers.record_room(["zm"], [], held=False)
    held = describe_stage.stage({"lanes": 2}, dry_run=False, run=_Systemd(), slots=lambda: 0)
    assert held["held"].startswith("every seat is staged")  # off: the hub alone decides
    out = describe_stage.stage({"lanes": 2, "workers": True}, dry_run=False, run=_Systemd(), slots=lambda: 0)
    assert "started" in out
    describe_workers.record_room(["zm"], ["zm"], held=False)  # 2b took the slot
    out = describe_stage.stage({"lanes": 2, "workers": True}, dry_run=False, run=_Systemd(), slots=lambda: 0)
    assert out["held"].startswith("every seat is staged")
