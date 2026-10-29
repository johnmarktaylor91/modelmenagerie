"""DX-64: the driver holds its OWN sleep immunity and records the state.

200a's sleep immunity was a STRAY host ``caffeinate`` (pid 3763) nothing owned
or monitored: a reboot, logout, or kill of that unrelated process would let the
host sleep mid-leg with paid sessions in flight. The driver now spawns and
holds its own assertion for its exact lifetime (``caffeinate -w <driver pid>``
so even a SIGKILLed driver releases it), and the assertion state is recorded in
the driver-lock owner record so the leg-open census can see an unarmed guard.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.crawler import power_assertion
from menagerie.crawler.power_assertion import (
    PowerAssertionState,
    driver_power_assertion,
)


class _FakeChild:
    """Stand-in caffeinate child process."""

    def __init__(self, alive: bool = True) -> None:
        self.pid = 4242
        self._alive = alive
        self.terminated = False

    def poll(self) -> int | None:
        return None if self._alive else 3

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.mark.smoke
def test_darwin_assertion_spawns_pid_bound_caffeinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The assertion is caffeinate -w <driver pid>: dies with the driver."""

    spawned: dict[str, object] = {}
    child = _FakeChild()

    def fake_popen(argv, **kwargs):  # noqa: ANN001, ANN202
        spawned["argv"] = tuple(argv)
        spawned["kwargs"] = kwargs
        return child

    monkeypatch.setattr(power_assertion, "subprocess", SimpleNamespace(
        Popen=fake_popen, DEVNULL=subprocess.DEVNULL, SubprocessError=subprocess.SubprocessError
    ))
    monkeypatch.setattr(power_assertion.sys, "platform", "darwin")
    monkeypatch.setattr(power_assertion.Path, "is_file", lambda self: True)

    with driver_power_assertion(watched_pid=999) as state:
        assert state.held is True
        assert state.mechanism == "caffeinate"
        assert state.pid == child.pid
        argv = spawned["argv"]
        assert argv[0] == power_assertion.CAFFEINATE_EXECUTABLE
        assert "-w" in argv
        assert argv[argv.index("-w") + 1] == "999"
    assert child.terminated is True


@pytest.mark.smoke
def test_non_darwin_platform_records_not_applicable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux legs record the guard as not-applicable, never silently absent."""

    monkeypatch.setattr(power_assertion.sys, "platform", "linux")
    with driver_power_assertion(watched_pid=1) as state:
        assert state.held is False
        assert state.mechanism == "not-applicable"


@pytest.mark.smoke
def test_spawn_failure_records_unarmed_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed spawn is a RECORDED unarmed guard, not a dead leg."""

    def failing_popen(argv, **kwargs):  # noqa: ANN001, ANN202
        raise OSError("caffeinate unavailable")

    monkeypatch.setattr(power_assertion, "subprocess", SimpleNamespace(
        Popen=failing_popen, DEVNULL=subprocess.DEVNULL,
        SubprocessError=subprocess.SubprocessError,
    ))
    monkeypatch.setattr(power_assertion.sys, "platform", "darwin")
    monkeypatch.setattr(power_assertion.Path, "is_file", lambda self: True)

    with driver_power_assertion(watched_pid=1) as state:
        assert state.held is False
        assert state.mechanism == "caffeinate"
        assert "caffeinate unavailable" in (state.detail or "")


@pytest.mark.smoke
def test_dead_on_arrival_child_records_unarmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that exits immediately is an unarmed guard, recorded."""

    child = _FakeChild(alive=False)
    monkeypatch.setattr(power_assertion, "subprocess", SimpleNamespace(
        Popen=lambda argv, **kwargs: child, DEVNULL=subprocess.DEVNULL,
        SubprocessError=subprocess.SubprocessError,
    ))
    monkeypatch.setattr(power_assertion.sys, "platform", "darwin")
    monkeypatch.setattr(power_assertion.Path, "is_file", lambda self: True)

    with driver_power_assertion(watched_pid=1) as state:
        assert state.held is False


@pytest.mark.smoke
def test_state_serializes_for_the_owner_record() -> None:
    """The recorded mapping is exactly the four census facts."""

    state = PowerAssertionState(
        mechanism="caffeinate", held=True, pid=7, detail=None
    )
    assert state.to_dict() == {
        "mechanism": "caffeinate",
        "held": True,
        "pid": 7,
        "detail": None,
    }


@pytest.mark.smoke
def test_driver_startup_records_the_assertion_in_the_lock_owner() -> None:
    """driver.run() holds the assertion and publishes its state (census hook)."""

    source = (
        Path(power_assertion.__file__).resolve().parent / "driver.py"
    ).read_text(encoding="utf-8")
    assert "driver_power_assertion(" in source
    assert '"power_assertion": ' in source
