from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from menagerie.pipeline import lane_stall, tick

START = datetime(2026, 10, 30, 9, 50, tzinfo=timezone.utc)


def _uuid7(at: datetime, tail: str = "7000-8000-000000000000") -> str:
    hexed = f"{int(at.timestamp() * 1000):012x}"
    return f"{hexed[:8]}-{hexed[8:]}-{tail}"


def _lanes(tmp_path: Path, *starts: datetime) -> Path:
    lanes = tmp_path / "lanes"
    for n, at in enumerate(starts):
        (lanes / _uuid7(at, f"7{n:03x}-8000-000000000000")).mkdir(parents=True)
    (lanes / "manual").mkdir(parents=True, exist_ok=True)  # not a run id: ignored
    return lanes


RESULTS = {
    "top_up": {"waiting_2b": 210, "waiting_2a": 289, "2b_repair_seat": ["f-van", "f-gin"],
               "runs": ["held: account-usage gate: no pool account has known fresh headroom"] * 13
               + ["held: phone builds first: the mini's phone lock is busy"]},
    "describes": {"last": {"verdicts": {"waiting": 3237},
                           "held": {"no_admission: account-usage gate": 40}}},
}


def test_the_last_start_is_the_newest_run_id(tmp_path: Path) -> None:
    lanes = _lanes(tmp_path, START - timedelta(hours=5), START)
    assert lane_stall.last_start(lanes) == START
    assert lane_stall.last_start(tmp_path / "absent") is None


def test_a_stall_sends_one_fact_with_its_reasons_and_waits_for_the_next_stall(
        tmp_path: Path) -> None:
    lanes, state = _lanes(tmp_path, START), {}
    assert lane_stall.check(state, RESULTS, lanes, START + timedelta(hours=2, minutes=59)) is None
    text = lane_stall.check(state, RESULTS, lanes, START + timedelta(hours=3, minutes=10))
    assert text is not None and "for 3.2 h (last start 2026-10-30 09:50 UTC)" in text
    assert "(2b 210, 2a 289, describe 3237)" in text
    assert "account-usage gate: no pool account has known fresh headroom (13)" in text
    assert "describe no_admission: account-usage gate (40)" in text
    assert "repair families kept off the Claude-only workers (2)" in text
    assert lane_stall.check(state, RESULTS, lanes, START + timedelta(hours=6)) is None  # same stall
    later = START + timedelta(hours=7)
    lanes = _lanes(tmp_path, later)  # a lane started, then work stalled again
    assert lane_stall.check(state, RESULTS, lanes, later + timedelta(hours=3)) is not None


def test_no_fact_without_waiting_work(tmp_path: Path) -> None:
    lanes = _lanes(tmp_path, START)
    idle = {"top_up": {"waiting_2b": 0, "waiting_2a": 0}, "describes": {"skipped": "off"}}
    assert lane_stall.check({}, idle, lanes, START + timedelta(hours=5)) is None


def test_the_alarm_runs_in_full_passes_only() -> None:
    (modes,) = [m for name, m, _ in tick.STEPS if name == "lane_stall"]
    assert modes == ("full",)
