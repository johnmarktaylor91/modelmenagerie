"""The bridge's remote polling: paced ssh calls, batched result fetches, stale claims.

No ssh: a fake link runs each remote command with ``sh`` against a local folder and counts the
calls, and a simulated clock drives the serve loop's passes, so the rates are exact.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.pipeline import bridge
from menagerie.pipeline.implement import POLL_S


class _Clock:
    def __init__(self) -> None:
        self.at = time.time()

    def now(self) -> float:
        return self.at


class _FakeLink:
    """Runs remote commands locally with ``sh``; ``calls`` counts them (one per ssh)."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, command: str, *, data: bytes | None = None) -> bytes:
        self.calls += 1
        done = subprocess.run(["sh", "-c", command], input=data, capture_output=True,
                              timeout=30, check=True)
        return done.stdout


@pytest.fixture
def spools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, _Clock]:
    local, remote = tmp_path / "hubspool", tmp_path / "remote"
    for name in ("new", "claimed", "done", "rejected"):
        (local / name).mkdir(parents=True)
    (remote / "done").mkdir(parents=True)
    (remote / "serving").touch()
    clock = _Clock()
    monkeypatch.setattr(bridge, "time", SimpleNamespace(monotonic=clock.now, time=clock.now))
    return local, remote, clock


def _claim(local: Path, name: str, age_s: float, clock: _Clock) -> None:
    path = local / "claimed" / f"{name}.json"
    path.write_text("{}\n")
    os.utime(path, (clock.at - age_s, clock.at - age_s))


def _drive(link: _FakeLink, local: Path, remote: Path, clock: _Clock, seconds: float,
           last: float = float("-inf")) -> float:
    """The serve loop's passes, ``POLL_S`` apart, over ``seconds`` of simulated time."""

    for _ in range(int(seconds / POLL_S)):
        last = bridge._bridge_once(link, local, remote.as_posix(), last)
        clock.at += POLL_S
    return last


def test_idle_bridge_polls_every_idle_interval(spools: tuple[Path, Path, _Clock]) -> None:
    local, remote, clock = spools
    link = _FakeLink()
    _drive(link, local, remote, clock, 600)
    assert link.calls == 600 / bridge.IDLE_POLL_S  # 3 a minute
    assert (local / "serving").is_file()  # touched on every poll while the runner beats


def test_busy_bridge_polls_every_fast_interval(spools: tuple[Path, Path, _Clock]) -> None:
    local, remote, clock = spools
    _claim(local, "m1-forward-eval-a", 0, clock)
    link = _FakeLink()
    _drive(link, local, remote, clock, 600)
    assert link.calls == 600 / bridge.FAST_POLL_S  # 12 a minute
    assert link.calls * 60 / 600 < 15


def test_results_come_back_in_one_call(spools: tuple[Path, Path, _Clock]) -> None:
    local, remote, clock = spools
    names = ["m1-forward-eval-a", "m2-forward-eval-b", "m3-forward-eval-c", "m4-forward-eval-d"]
    for name in names:
        _claim(local, name, 0, clock)
    for name in names[:3]:
        (remote / "done" / f"{name}.json").write_text(f'{{"request": "{name}",\n "x": 1}}\n')
    (remote / "done" / "m9-forward-eval-other.json").write_text("{}\n")  # not ours
    link = _FakeLink()
    bridge._bridge_once(link, local, remote.as_posix(), float("-inf"))
    assert link.calls == 1
    for name in names[:3]:
        assert (local / "done" / f"{name}.json").read_text() == \
            f'{{"request": "{name}",\n "x": 1}}\n'
        assert not (local / "claimed" / f"{name}.json").exists()
    assert (local / "claimed" / f"{names[3]}.json").exists()
    assert not (local / "done" / "m9-forward-eval-other.json").exists()


def test_stale_runner_heartbeat_leaves_serving_alone(spools: tuple[Path, Path, _Clock]) -> None:
    local, remote, clock = spools
    old = time.time() - 600
    os.utime(remote / "serving", (old, old))
    bridge._bridge_once(_FakeLink(), local, remote.as_posix(), float("-inf"))
    assert not (local / "serving").exists()


def test_stale_claim_drops_to_idle_rate_but_is_still_collected(
        spools: tuple[Path, Path, _Clock]) -> None:
    local, remote, clock = spools
    name = "m1-forward-eval-old"
    _claim(local, name, bridge.CLAIM_FRESH_S + 60, clock)
    link = _FakeLink()
    last = _drive(link, local, remote, clock, 600)
    assert link.calls == 600 / bridge.IDLE_POLL_S
    (remote / "done" / f"{name}.json").write_text("{}\n")
    _drive(link, local, remote, clock, bridge.IDLE_POLL_S, last)
    assert (local / "done" / f"{name}.json").is_file()
    assert not (local / "claimed" / f"{name}.json").exists()


def test_a_new_claim_brings_the_next_poll_forward(spools: tuple[Path, Path, _Clock]) -> None:
    local, remote, clock = spools
    link = _FakeLink()
    last = _drive(link, local, remote, clock, 1)  # one idle poll, the next due in 20 s
    _claim(local, "m1-forward-eval-a", 0, clock)
    (remote / "done" / "m1-forward-eval-a.json").write_text("{}\n")
    _drive(link, local, remote, clock, bridge.FAST_POLL_S, last)
    assert link.calls == 2
    assert (local / "done" / "m1-forward-eval-a.json").is_file()



def _code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
          idle: list[bool]) -> tuple[bridge.LoadedCode, Path, list[float]]:
    """A LoadedCode rooted at a temporary package holding one imported module."""
    source = tmp_path / "menagerie" / "pipeline" / "fake.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n")
    monkeypatch.setitem(sys.modules, "menagerie_fake_loaded", SimpleNamespace(__file__=str(source)))
    now = [0.0]
    code = bridge.LoadedCode(idle=lambda: idle[0], clock=lambda: now[0], root=tmp_path / "menagerie")
    return code, source, now

def test_a_changed_module_restarts_the_bridge_once_the_relay_is_idle(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    idle = [False]
    code, source, now = _code(tmp_path, monkeypatch, idle)
    now[0] = bridge.STALE_CHECK_S
    assert not code.restart_due()  # nothing changed
    os.utime(source, (time.time() + 5, time.time() + 5))
    now[0] += 10
    assert not code.restart_due()  # checked at most once a minute
    now[0] += bridge.STALE_CHECK_S
    assert not code.restart_due() and code.changed == [str(source)]  # a verb is in progress
    idle[0] = True
    now[0] += bridge.STALE_CHECK_S
    assert code.restart_due()


def test_a_stale_bridge_restarts_after_its_cap_even_while_busy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code, source, now = _code(tmp_path, monkeypatch, [False])
    os.utime(source, (time.time() + 5, time.time() + 5))
    now[0] = bridge.STALE_CHECK_S
    assert not code.restart_due()
    now[0] += bridge.STALE_MAX_S
    assert code.restart_due()


def test_relay_idle_reads_its_lanes(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import mini_relay

    monkeypatch.setattr(mini_relay, "_serving", None)
    assert mini_relay.idle()
    busy = [False]
    lane = SimpleNamespace(busy=lambda: busy[0])
    monkeypatch.setattr(mini_relay, "_serving", SimpleNamespace(lanes={"2b-mini-a": lane}))
    assert mini_relay.idle()
    busy[0] = True
    assert not mini_relay.idle()


def test_the_2b_prompt_keeps_verbs_in_the_foreground() -> None:
    from menagerie.pipeline import implement

    text = implement.PROMPT.read_text(encoding="utf-8")
    assert "Run every verb in the foreground" in text and "silent for 30 minutes is stopped" in text
