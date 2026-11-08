"""A big one-off job on the runner host waits for runner B's day size (``oneoff``)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import oneoff

NOON = datetime(2026, 10, 29, 19, 0, tzinfo=timezone.utc)  # 12 PM PT
HOST = oneoff.RUNNER_HOST
DAY = {"runners": {"B": {"job": "b", "size": 1, "submitted_at": "2026-10-29T17:40:00+00:00"}}}


def _b(**record: Any) -> dict[str, Any]:
    return {"runners": {"B": {"job": "b", "size": 1, "submitted_at": "2026-10-29T17:40:00+00:00", **record}}}


@pytest.mark.parametrize("state, why", [
    (_b(size=0), "below its day size"),
    (_b(resize_to=1), "resize pending"),
    (_b(draining=True), "resize pending"),
    (_b(cancelled="b", cancelled_for="resize"), "stopping to change size"),
    (_b(cancelled="b", cancelled_for="g"), "give way to G"),
    (_b(refused_at="2026-10-29T18:00:00+00:00"), "waits for run-on capacity"),
    ({"runners": {}}, "not running"),
])
def test_a_big_job_waits_while_b_is_short_of_its_day_size(state: dict[str, Any], why: str) -> None:
    answer = oneoff.verdict(24, HOST, state, NOON, NOON)
    assert not answer.ok and why in answer.why and "next tick pass, about 12:15 PM PT" in answer.why


def test_small_jobs_other_hosts_and_a_settled_b_pass() -> None:
    short = _b(size=0)
    assert oneoff.verdict(8, HOST, short, NOON, NOON).ok
    assert oneoff.verdict(24, "otherbox", short, NOON, NOON).ok
    assert not oneoff.verdict(24, "any", short, NOON, NOON).ok  # run-on may pick the runner host
    assert oneoff.verdict(24, HOST, DAY, NOON, NOON).ok


def test_stale_state_refuses_and_night_says_six_am() -> None:
    assert "40 minutes old" in oneoff.verdict(24, HOST, DAY, NOON, NOON - timedelta(minutes=40)).why
    night = datetime(2026, 10, 29, 7, 0, tzinfo=timezone.utc)  # midnight PT
    answer = oneoff.verdict(24, HOST, {"runners": {}}, night, night)
    assert not answer.ok and "after 6 AM PT" in answer.why


def test_submit_options_read_the_declaration_before_the_payload() -> None:
    args = ["--project", "menagerie", "--memory-gib", "24", "--host=trace-host", "--", "x", "--memory-gib", "1"]
    assert oneoff.submit_options(args) == (24.0, "trace-host")
    assert oneoff.submit_options(["--project", "menagerie", "--", "x"]) == (0.0, None)


@pytest.fixture
def tick_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MENAGERIE_WORK_ROOT", str(tmp_path))
    folder = oneoff.work_dir("tick")
    folder.mkdir(parents=True)
    return folder


def test_main_submits_refuses_and_logs_an_override(tick_dir: Path, tmp_path: Path) -> None:
    (tmp_path / ".compute-host").write_text(f"{HOST}\n")
    (tick_dir / "state.json").write_text(json.dumps(_b(size=0)))
    now = datetime.now(timezone.utc)
    ran: list[list[str]] = []
    run = lambda argv: ran.append(argv) or 0  # noqa: E731
    args = ["--", "--project", "menagerie", "--memory-gib", "24", "--", "true"]
    assert oneoff.main(args, repo=tmp_path, run=run, now=now) == oneoff.REFUSED and ran == []
    assert oneoff.main(["--because", "JMT asked", *args], repo=tmp_path, run=run, now=now) == 0
    assert ran == [["run-on", "submit", "--project", "menagerie", "--memory-gib", "24", "--", "true"]]
    logged = json.loads((tick_dir / "oneoff-overrides.jsonl").read_text())
    assert logged["because"] == "JMT asked" and "below its day size" in logged["refusal"]
    (tick_dir / "state.json").write_text(json.dumps(DAY))
    assert oneoff.main(args, repo=tmp_path, run=run, now=now) == 0 and len(ran) == 2
    assert oneoff.main(["--check", "--memory-gib", "24"], repo=tmp_path, run=run, now=now) == 0
