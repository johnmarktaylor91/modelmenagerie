"""T5-5 RED set: the env lane's busy guard under scheduler concurrency.

The env lane is a host singleton (T3-T8). `SequentialEnvironmentLifecycle`
historically guarded that with a check-then-act on `_active` -- safe only
while every `run()` call came from one thread. The scheduler makes the guard
load-bearing, so it must be a real mutex: a concurrent `run()` observes a
deterministic typed `EnvironmentBusyError`, never a race through the
validation window, and a builder death releases the claim so a later turn
can retry.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from menagerie.crawler.effort import EffortTracker, StageCap
from menagerie.crawler.env_lifecycle import (
    EnvironmentBusyError,
    EnvironmentLifecycleError,
    ProbeResult,
    SequentialEnvironmentLifecycle,
)
from menagerie.crawler.envs import load_environment_registry
from menagerie.crawler.tests.test_slice_c_envs_lifecycle import (
    FakeEnvironmentBackend,
    _copy_env_specs,
    _swap_in_fake_family,
)


def _lifecycle(
    tmp_path: Path, events: list[str], *, attempts: int = 10
) -> tuple[SequentialEnvironmentLifecycle, object]:
    """Build a lifecycle over the fake backend with one buildable intent.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    events:
        Backend event sink.
    attempts:
        Environment effort-cap attempts for the test budget.

    Returns
    -------
    tuple[SequentialEnvironmentLifecycle, object]
        The lifecycle and the committed `core` intent.
    """

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend(events),
        EffortTracker({"environment": StageCap(attempts=attempts, seconds=600, bytes=10**9)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: 10**12,
        minimum_free_bytes=0,
    )
    return lifecycle, intent


@pytest.mark.heavy
def test_concurrent_run_observes_typed_busy_error_never_a_race(tmp_path: Path) -> None:
    """While one turn holds the activation, every concurrent entry refuses typed.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    events: list[str] = []
    lifecycle, intent = _lifecycle(tmp_path, events)
    entered = threading.Event()
    release = threading.Event()
    winner_error: list[BaseException] = []

    def holding_use(_prefix: Path, _probes: tuple[ProbeResult, ...]) -> None:
        entered.set()
        assert release.wait(timeout=30.0), "test release never arrived"

    def winner() -> None:
        try:
            lifecycle.run(intent, use=holding_use)
        except BaseException as exc:  # noqa: BLE001 - surfaced to the main thread
            winner_error.append(exc)
            entered.set()

    winner_thread = threading.Thread(target=winner)
    winner_thread.start()
    assert entered.wait(timeout=30.0)
    assert not winner_error, f"winner failed before contention: {winner_error}"

    contender_errors: list[BaseException | None] = []
    contender_lock = threading.Lock()

    def contend() -> None:
        try:
            lifecycle.run(intent, use=lambda _p, _r: None)
        except BaseException as exc:  # noqa: BLE001 - the exception IS the assertion
            with contender_lock:
                contender_errors.append(exc)
        else:
            with contender_lock:
                contender_errors.append(None)

    contenders = [threading.Thread(target=contend) for _ in range(6)]
    for thread in contenders:
        thread.start()
    for thread in contenders:
        thread.join(timeout=30.0)
    release.set()
    winner_thread.join(timeout=30.0)
    assert not winner_thread.is_alive()
    assert not winner_error

    assert len(contender_errors) == 6
    for error in contender_errors:
        assert isinstance(error, EnvironmentBusyError)
    # Exactly one prefix was ever created while the winner held the claim.
    assert events.count("create") == 1
    # The claim released cleanly: a fresh turn succeeds afterwards.
    lifecycle.run(intent, use=lambda _p, _r: None)
    assert events.count("create") == 2


@pytest.mark.smoke
def test_builder_death_releases_the_claim_for_a_retry(tmp_path: Path) -> None:
    """A create failure releases the activation so a later turn can retry.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    events: list[str] = []

    class DyingOnceBackend(FakeEnvironmentBackend):
        def __init__(self) -> None:
            super().__init__(events)
            self.deaths = 0

        def create(self, lock_file: Path, prefix: Path) -> bytes:
            if self.deaths == 0:
                self.deaths += 1
                raise OSError("builder died mid-create")
            return super().create(lock_file, prefix)

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    lifecycle = SequentialEnvironmentLifecycle(
        DyingOnceBackend(),
        EffortTracker({"environment": StageCap(attempts=10, seconds=600, bytes=10**9)}),
        env_root=tmp_path / "active",
        disk_free=lambda _path: 10**12,
        minimum_free_bytes=0,
    )
    with pytest.raises((EnvironmentLifecycleError, OSError)):
        lifecycle.run(intent, use=lambda _p, _r: None)
    # The waiter's retry is an ordinary next turn, not a wedged singleton.
    lifecycle.run(intent, use=lambda _p, _r: None)
    assert events.count("create") == 1
