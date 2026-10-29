"""R0 instrument self-tests: the injectable monotonic clock (AB-11/AB-13).

The virtual-time seam is the prerequisite instrument for every scheduler,
pacing, and economics rehearsal (W17): driver-side duration and deadline
arithmetic must read ONE injectable monotonic clock so a rehearsal can run
hours of scheduling in milliseconds of wall. These tests keep the seam
provably non-vacuous -- each one injects a controlled clock and asserts the
machinery actually READ it, because a seam the driver silently bypasses would
make every later virtual-time rehearsal green without testing anything.

Deliberately outside the seam (they bound REAL child processes or REAL kernel
locks, where virtual time would spin or hang): ``_run_operator_command``,
``_close_paid_custody``, ``_reconcile_paid_custody_lease``'s reap wait, and
the real-wall slicing loop inside ``_wait_before_infrastructure_retry``.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.crawler.driver_admission import (
    _AuthorWavePool,
    _CheckerSessionPool,
    _RetryLadderState,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot


class VirtualMonotonic:
    """A controllable monotonic clock that counts every consultation."""

    def __init__(self, start: float = 1000.0, step: float = 0.0) -> None:
        """Start at ``start`` and advance ``step`` seconds per read."""

        self.now = start
        self.step = step
        self.reads = 0

    def __call__(self) -> float:
        """Return the virtual instant, then advance by the configured step."""

        self.reads += 1
        current = self.now
        self.now += self.step
        return current

    def advance(self, seconds: float) -> None:
        """Move virtual time forward explicitly."""

        self.now += seconds


@pytest.mark.smoke
def test_driver_dependencies_carry_an_injectable_monotonic_clock(tmp_path: Path) -> None:
    """The dependency dataclass owns the seam and defaults to the real clock."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    assert driver.dependencies.monotonic is time.monotonic
    clock = VirtualMonotonic()
    driver.dependencies = replace(driver.dependencies, monotonic=clock)
    assert driver.dependencies.monotonic is clock


@pytest.mark.smoke
def test_parked_backoff_deadline_is_virtual_clock_arithmetic(tmp_path: Path) -> None:
    """A capped ladder parks its retry wait at exactly virtual-now + backoff.

    The parked deadline rides the ladder continuation and is consumed against
    the SAME clock on resume, so both sides must read the injected seam. The
    asserted value is exact virtual arithmetic: on the real clock this test
    cannot pass by accident.
    """

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    clock = VirtualMonotonic(start=5000.0)
    driver.dependencies = replace(driver.dependencies, monotonic=clock)
    ladder = _RetryLadderState(work_id="work-virtual", per_round_cap=1)
    ladder.attempts_this_round = 1
    driver._wait_or_park_infrastructure_backoff(ladder, 30.0)  # noqa: SLF001
    assert ladder.pending_backoff_deadline == pytest.approx(5030.0)
    assert ladder.observed_backoffs == [30.0]
    assert clock.reads == 1


@pytest.mark.smoke
def test_author_wave_pool_paces_on_the_injected_clock(capsys: pytest.CaptureFixture) -> None:
    """The pool's starvation-note throttle advances only with virtual time."""

    clock = VirtualMonotonic(start=100.0)
    pool = _AuthorWavePool(
        1,
        lambda _item, _request_admission: None,  # type: ignore[arg-type, return-value]
        lambda _item, _attempt_group: None,  # type: ignore[arg-type, return-value]
        monotonic=clock,
    )
    pool._note_relay_wait("m-one", 45.0)  # noqa: SLF001
    clock.advance(59.0)
    pool._note_relay_wait("m-two", 45.0)  # noqa: SLF001
    clock.advance(2.0)
    pool._note_relay_wait("m-three", 45.0)  # noqa: SLF001
    notes = [
        line for line in capsys.readouterr().err.splitlines() if "author-admission relay" in line
    ]
    assert len(notes) == 2
    assert "m-one" in notes[0] and "m-three" in notes[1]


@pytest.mark.smoke
def test_checker_session_pool_accepts_the_injected_clock() -> None:
    """The checker pool binds the injected clock for its paid span measurement."""

    clock = VirtualMonotonic()
    pool = _CheckerSessionPool(
        1,
        lambda _scope: None,  # type: ignore[arg-type, return-value]
        monotonic=clock,
    )
    assert pool._monotonic is clock  # noqa: SLF001


@pytest.mark.heavy
def test_pipelined_leg_consults_the_injected_clock_end_to_end(tmp_path: Path) -> None:
    """A full fake pipelined leg reads the injected seam, not the real clock.

    This is the wiring proof from ``DriverDependencies`` down through the
    scheduler's readiness windows and the admission machinery: the leg must
    complete AND the injected clock must have been consulted. Paid settlement
    spans measured on a constant virtual clock are exactly zero -- a nonzero
    machine wall in the spend ledger would mean some span origin still read
    the real clock.
    """

    snapshot = _snapshot(tmp_path, count=3)
    driver = _driver(
        tmp_path,
        snapshot,
        scheduler="pipelined",
        author_concurrency=2,
        checker_concurrency=2,
    )
    clock = VirtualMonotonic(start=7000.0, step=0.0)
    driver.dependencies = replace(driver.dependencies, monotonic=clock)
    result = driver.run()
    assert result.status == "complete"
    assert clock.reads > 0
    spend_path = (
        driver.paths.ledgers.models.parent.parent / "operational" / "spend.jsonl"
    )
    settled_walls = [
        row["machine"]["wall_seconds"]
        for row in map(json.loads, spend_path.read_text(encoding="utf-8").splitlines())
        if row.get("event") == "session_settled"
        and row.get("machine", {}).get("source") == "writer-monotonic"
    ]
    assert settled_walls, "the fake leg settled no paid sessions"
    assert all(wall == 0.0 for wall in settled_walls)
