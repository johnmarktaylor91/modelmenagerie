"""Bounded concurrent checker sessions with a single serial canonical writer.

The checker lane is ~15 hours of wall per pilot leg and every session is a
multi-minute Codex round trip, dispatched strictly one at a time before this
pool existed. ``_CheckerSessionPool`` is the checker-lane port of the author
wave pool: only the *sessions* overlap. Envelope identity, gate normalization,
routing, every ledger append, and every park/terminal decision stay on the
single canonical writer, in ladder order.

This is the T5-5 feed-forward rig for the WAVE driver: overlap-to-bound plus
the reverse-completion-order determinism oracle, ported from
``test_author_wave_concurrency``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pytest

from menagerie.crawler.checker_dispatch import CheckerPartialResultError
from menagerie.crawler.constants import (
    DEFAULT_CHECKER_WAVE_CONCURRENCY,
    MAX_CHECKER_WAVE_CONCURRENCY,
    METADATA_BATCH_MAX,
)
from menagerie.crawler.driver import DriverConfig
from menagerie.crawler.driver_contracts import CheckerOutcome
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_author_wave_concurrency import _canonical_projection
from menagerie.crawler.tests.test_slice_f_driver import (
    FakePauseScheduler,
    FidelityAuthor,
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)

#: Per-test tiering (T5-7): rendezvous cells are heavy; classification and
#: configuration cells stay smoke.


@dataclass
class _Interval:
    """One checker session's observed wall-clock occupancy."""

    key: str
    started: float
    finished: float


class ObservedChecker(ScriptedChecker):
    """Synthetic checker that records each session's occupancy window.

    ``hold`` runs while the session is "in flight", so it is the exact window a
    concurrent lane is supposed to overlap. Every mutation is lock-guarded:
    unlike the writer-owned ledger state, this fake IS called from pool
    threads, and unsynchronized counters would make the evidence itself racy.
    """

    def __init__(
        self,
        *,
        hold: Optional[Callable[[str], None]] = None,
        after: Optional[Callable[[str], None]] = None,
        fidelity_hold: Optional[Callable[[str], None]] = None,
        **kwargs: Any,
    ) -> None:
        """Bind the in-session holds and the observation state."""

        super().__init__(**kwargs)
        self._hold = hold or (lambda _key: None)
        self._after = after or (lambda _key: None)
        self._fidelity_hold = fidelity_hold or (lambda _key: None)
        self._observer_lock = threading.Lock()
        self._live = 0
        self._fidelity_live = 0
        self._seen_metadata_ids: set[str] = set()
        self._seen_fidelity_ids: set[str] = set()
        self.intervals: list[_Interval] = []
        self.completed: list[str] = []
        self.observed_batch_sizes: list[int] = []
        self.peak_concurrency = 0
        self.fidelity_peak_concurrency = 0

    @staticmethod
    def _batch_key(artifacts: Sequence[Any]) -> str:
        """Name one batch by its member stable IDs."""

        return "+".join(str(artifact.proposal["stable_id"]) for artifact in artifacts)

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Adjudicate one batch while recording its occupancy window.

        Observation covers PRIMARY (first-adjudication) sessions only: the
        H-5 duplicate-accept confirmations are deliberately serial writer
        buys over already-adjudicated members, so they are a different lane
        fact than the pool overlap these rigs measure -- and a barrier hold
        applied to a serial confirmation would deadlock by construction.
        """

        member_ids = [str(artifact.proposal["stable_id"]) for artifact in artifacts]
        key = self._batch_key(artifacts)
        started = time.monotonic()
        with self._observer_lock:
            primary = any(member not in self._seen_metadata_ids for member in member_ids)
            self._seen_metadata_ids.update(member_ids)
            if primary:
                self._live += 1
                self.peak_concurrency = max(self.peak_concurrency, self._live)
                self.observed_batch_sizes.append(len(artifacts))
        if not primary:
            with self._observer_lock:
                return super().check_metadata(artifacts, work_root, config)
        try:
            self._hold(key)
            with self._observer_lock:
                outcome = super().check_metadata(artifacts, work_root, config)
            return outcome
        finally:
            with self._observer_lock:
                self._live -= 1
                self.intervals.append(_Interval(key, started, time.monotonic()))
                self.completed.append(key)
            self._after(key)

    def check_fidelity(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Adjudicate one fidelity request while recording PRIMARY overlap."""

        key = str(artifact.proposal["stable_id"])
        with self._observer_lock:
            primary = key not in self._seen_fidelity_ids
            self._seen_fidelity_ids.add(key)
            if primary:
                self._fidelity_live += 1
                self.fidelity_peak_concurrency = max(
                    self.fidelity_peak_concurrency, self._fidelity_live
                )
        if not primary:
            with self._observer_lock:
                return super().check_fidelity(artifact, work_root, config)
        try:
            self._fidelity_hold(key)
            with self._observer_lock:
                return super().check_fidelity(artifact, work_root, config)
        finally:
            with self._observer_lock:
                self._fidelity_live -= 1


def _max_overlap(intervals: list[_Interval]) -> int:
    """Return the largest number of intervals that were open simultaneously."""

    edges = sorted(
        [(interval.started, 1) for interval in intervals]
        + [(interval.finished, -1) for interval in intervals]
    )
    live = 0
    peak = 0
    for _instant, delta in edges:
        live += delta
        peak = max(peak, live)
    return peak


# ---------------------------------------------------------------------------
# 1. sessions genuinely overlap, within the configured bound
# ---------------------------------------------------------------------------


@pytest.mark.heavy
def test_checker_sessions_overlap_up_to_the_configured_bound(tmp_path: Path) -> None:
    """Two width-5 batches must be able to rendezvous inside one gate round.

    The hold is a barrier of width two. Under the historical serial lane the
    first batch waits for a sibling that cannot exist and the barrier times
    out, so this test cannot pass without real session overlap. The measured
    interval overlap is asserted independently and must not exceed the bound.
    """

    degree = 2
    barrier = threading.Barrier(degree, timeout=30.0)
    snapshot = _snapshot(tmp_path, count=10)
    checker = ObservedChecker(hold=lambda _key: barrier.wait())

    result = _driver(
        tmp_path,
        snapshot,
        checker=checker,
        checker_concurrency=degree,
    ).run()

    assert result.status == "complete"
    assert checker.peak_concurrency == degree
    assert _max_overlap(checker.intervals) == degree
    assert checker.peak_concurrency <= degree


@pytest.mark.heavy
def test_serial_checker_configuration_keeps_exactly_one_session_in_flight(
    tmp_path: Path,
) -> None:
    """``checker_concurrency=1`` must restore the historical fully serial lane."""

    snapshot = _snapshot(tmp_path, count=10)
    checker = ObservedChecker(hold=lambda _key: time.sleep(0.02))

    result = _driver(tmp_path, snapshot, checker=checker, checker_concurrency=1).run()

    assert result.status == "complete"
    assert checker.peak_concurrency == 1
    assert _max_overlap(checker.intervals) == 1


@pytest.mark.smoke
def test_checker_concurrency_is_a_validated_closed_range() -> None:
    """The fan-out bound must be an explicit, validated number, never unbounded."""

    assert DriverConfig().checker_concurrency == DEFAULT_CHECKER_WAVE_CONCURRENCY
    assert DEFAULT_CHECKER_WAVE_CONCURRENCY == 24
    with pytest.raises(ValueError, match="checker_concurrency"):
        DriverConfig(checker_concurrency=0)
    with pytest.raises(ValueError, match="checker_concurrency"):
        DriverConfig(checker_concurrency=MAX_CHECKER_WAVE_CONCURRENCY + 1)


@pytest.mark.smoke
def test_concurrency_parallelizes_sessions_without_widening_batches(
    tmp_path: Path,
) -> None:
    """The calibrated batch-width pin survives the pool untouched.

    The pool parallelizes SESSIONS; the batch width is a calibrated
    measurement (constants.py) and never a concurrency knob.
    """

    snapshot = _snapshot(tmp_path, count=10)
    checker = ObservedChecker()

    result = _driver(tmp_path, snapshot, checker=checker, checker_concurrency=4).run()

    assert result.status == "complete"
    assert checker.observed_batch_sizes
    assert all(size <= METADATA_BATCH_MAX for size in checker.observed_batch_sizes)
    # Ten models divide exactly at the calibrated width: every batch is full.
    assert sorted(checker.observed_batch_sizes) == [METADATA_BATCH_MAX] * (
        10 // METADATA_BATCH_MAX
    )


# ---------------------------------------------------------------------------
# 2. failure and pause isolation
# ---------------------------------------------------------------------------


class OneBatchPoisonedChecker(ObservedChecker):
    """Refuse full-width batches containing one poisoned model; answer halves."""

    def __init__(self, poisoned_id: str, **kwargs: Any) -> None:
        """Bind the model whose full-width batch draws a partial-result refusal."""

        super().__init__(**kwargs)
        self._poisoned_id = poisoned_id

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Fail the poisoned full batch exactly once per width."""

        ids = {str(artifact.proposal["stable_id"]) for artifact in artifacts}
        if self._poisoned_id in ids and len(artifacts) == METADATA_BATCH_MAX:
            raise CheckerPartialResultError("checker result is partial or has extra items")
        return super().check_metadata(artifacts, work_root, config)


@pytest.mark.heavy
@pytest.mark.skipif(
    METADATA_BATCH_MAX < 2,
    reason=(
        "the halving downshift ladder has no consumer at the calibrated "
        "width-1 production config: a single-item batch cannot split, so a "
        "poisoned batch parks its one member instead (the park path keeps its "
        "own tests); re-arms with any future width increase"
    ),
)
def test_one_failing_batch_downshifts_while_its_sibling_batch_commits(
    tmp_path: Path,
) -> None:
    """A lane failure in one batch never discards its sibling's finished session.

    The poisoned batch rides the halving ladder exactly as under the serial
    lane; the healthy sibling batch's concurrently bought session still
    commits, and every model still gates.
    """

    snapshot = _snapshot(tmp_path, count=10)
    poisoned_id = snapshot.items[0].stable_id
    checker = OneBatchPoisonedChecker(poisoned_id)

    result = _driver(tmp_path, snapshot, checker=checker, checker_concurrency=2).run()

    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    current = {model["stable_id"] for model in scan_jsonl(paths.ledgers.models)}
    assert current == {item.stable_id for item in snapshot.items}
    # The poisoned full batch failed once and was re-asked in halves.
    assert METADATA_BATCH_MAX in checker.observed_batch_sizes
    assert any(size < METADATA_BATCH_MAX for size in checker.observed_batch_sizes)


@pytest.mark.smoke
def test_quota_pause_with_a_sibling_batch_in_flight_records_one_pause(
    tmp_path: Path,
) -> None:
    """A quota pause reaches the driver once; the drained sibling corrupts nothing.

    Drain-never-cancel: the sibling batch's session was already bought when the
    pause landed, so it runs to completion on its pool thread, but no gate is
    appended for it and the pause is recorded exactly once, from the first
    batch in ladder order.
    """

    snapshot = _snapshot(tmp_path, count=10)
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)
    checker = ObservedChecker(quota=True)

    result = _driver(
        tmp_path,
        snapshot,
        checker=checker,
        checker_concurrency=2,
        pause_scheduler=scheduler,
    ).run()

    assert result.status == "paused:usage-limit"
    assert scheduler.calls == 1
    # Neither the paused batch nor the drained sibling minted a gate; models
    # legitimately sit at runs/pending (execution precedes gating).
    assert scan_jsonl(paths.ledgers.gates) == []
    pending = scan_jsonl(paths.ledgers.models)
    assert all(model["authored_metadata_state"] == "pending" for model in pending)


# ---------------------------------------------------------------------------
# 3. determinism: identical to serial regardless of completion order
# ---------------------------------------------------------------------------


class _ReverseCompletionChain:
    """Force the wave's checker sessions to complete in exactly reverse order.

    Identical construction to the author rig's chain: session ``i`` blocks
    until session ``i + 1`` has recorded its completion, so completion order is
    reverse ladder order by construction. The chain is a concurrency assertion
    in its own right -- a serial lane deadlocks here loudly (timeout) rather
    than passing quietly.
    """

    def __init__(self, order: dict[str, int], *, timeout: float = 60.0) -> None:
        """Bind the ladder order and the per-link deadline."""

        self._order = order
        self._timeout = timeout
        self._done = [threading.Event() for _ in order]

    def hold(self, key: str) -> None:
        """Block until every later-scheduled batch has already completed."""

        successor = self._order[key] + 1
        if successor >= len(self._done):
            return
        if not self._done[successor].wait(timeout=self._timeout):
            raise AssertionError(
                f"batch {self._order[key]} waited {self._timeout:g}s for its successor: "
                "the lane is not running its checker sessions concurrently"
            )

    def after(self, key: str) -> None:
        """Release the predecessor once this batch's completion is recorded."""

        self._done[self._order[key]].set()


@pytest.mark.heavy
def test_reverse_completion_order_yields_the_serial_result_exactly(tmp_path: Path) -> None:
    """A wave whose checker sessions finish backwards records the serial result.

    If any canonical effect were driven by completion order rather than ladder
    order -- gate append order, gate rounds, record revisions -- the canonical
    projection would diverge from the serial baseline.
    """

    serial_root = tmp_path / "serial"
    concurrent_root = tmp_path / "concurrent"
    serial_root.mkdir()
    concurrent_root.mkdir()

    serial_snapshot = _snapshot(serial_root, count=10)
    concurrent_snapshot = _snapshot(concurrent_root, count=10)
    scheduled = [item.stable_id for item in concurrent_snapshot.items]
    order = {
        "+".join(scheduled[offset : offset + METADATA_BATCH_MAX]): index
        for index, offset in enumerate(range(0, len(scheduled), METADATA_BATCH_MAX))
    }

    serial = _driver(
        serial_root,
        serial_snapshot,
        checker=ObservedChecker(),
        checker_concurrency=1,
    ).run()
    chain = _ReverseCompletionChain(order)
    concurrent_checker = ObservedChecker(hold=chain.hold, after=chain.after)
    concurrent = _driver(
        concurrent_root,
        concurrent_snapshot,
        checker=concurrent_checker,
        # Every batch must be in flight at once for the reverse chain to
        # release (each waits on its successor), so the pool must admit the
        # whole wave whatever the calibrated width makes of ten models.
        checker_concurrency=len(order),
    ).run()

    assert serial.status == concurrent.status == "complete"
    # Guaranteed by construction; asserted so a broken chain can never silently
    # weaken the equality below into a same-order comparison.
    assert concurrent_checker.completed == sorted(order, key=lambda key: -order[key])

    assert _canonical_projection(
        _paths(concurrent_root, concurrent_snapshot), concurrent_root
    ) == _canonical_projection(_paths(serial_root, serial_snapshot), serial_root)


@pytest.mark.heavy
def test_fidelity_sessions_overlap_up_to_the_configured_bound(tmp_path: Path) -> None:
    """Per-model fidelity sessions rendezvous under the same lane bound."""

    degree = 2
    barrier = threading.Barrier(degree, timeout=30.0)
    snapshot = _snapshot(tmp_path, count=4)
    checker = ObservedChecker(fidelity_hold=lambda _key: barrier.wait())

    result = _driver(
        tmp_path,
        snapshot,
        author=FidelityAuthor(),
        checker=checker,
        checker_concurrency=degree,
    ).run()

    assert result.status == "complete"
    assert checker.fidelity_peak_concurrency == degree
    # Four primary sessions plus four serial H-5 blind confirmations.
    assert checker.fidelity_calls == 8
