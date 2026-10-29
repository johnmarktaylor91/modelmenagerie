"""Virtual-time scheduler rehearsal rig core (AB-12/AB-13, the W17 instrument).

The rig drives the REAL driver -- real ``run_pipelined_leg``, real admission,
real wave pools, real ledger appends -- with stub sessions that replay the
MEASURED 200a wall distribution in VIRTUAL time: a session "runs" by blocking
on a discrete-event clock until every in-flight session is blocked, at which
point the clock jumps to the earliest pending wake. Hours of session wall
simulate in seconds, thread overlap is real (the wave pools' worker threads
genuinely interleave), and the concurrency ledger the rig emits is measured,
not modeled.

Composes with AB-11: the same :class:`VirtualClock` is injected as
``DriverDependencies.monotonic``, so the driver's own deadline and pacing
arithmetic runs on rehearsal time.

Vacuousness guards (the AB-13 doctrine): the clock refuses to hang (a bounded
real-wall watchdog raises instead of deadlocking a broken rehearsal), replay
lanes record every session they simulate, and the self-tests in
``test_scheduler_throughput_floor.py`` prove the injectors fire before any
floor number is trusted.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from menagerie.crawler.identity import stable_hash

_DISTRIBUTION_PATH = Path(__file__).resolve().parent / "spend_wall_distribution_200a.json"
DISTRIBUTION_SCHEMA_VERSION = "menagerie.crawler.spend-wall-distribution.v1"

#: Real-wall seconds the virtual clock may sit with no progress before the
#: rehearsal declares itself wedged. A harness that can hang is worse than a
#: harness that fails: nothing it guards can be trusted from a stuck run.
_WEDGE_WATCHDOG_SECONDS = 120.0

#: Real-wall seconds the quiescent state (every in-flight session asleep) must
#: HOLD before virtual time advances. The production writer services queued
#: admission buys on ~100ms real slices; advancing the instant the last
#: started session sleeps would leap over walls that queued siblings should
#: have overlapped, and the rig would measure serial execution regardless of
#: the scheduler under test (the exact vacuousness AB-13 forbids -- caught by
#: this rig's own self-test, which plants a known-concurrent load and refuses
#: a serial measurement).
_QUIESCENT_GRACE_SECONDS = 0.25


def load_measured_wall_distribution() -> dict[str, Any]:
    """Load the committed 200a wall-distribution artifact."""

    payload = json.loads(_DISTRIBUTION_PATH.read_text(encoding="utf-8"))
    if payload["schema_version"] != DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError(
            f"unknown distribution artifact schema: {payload['schema_version']}"
        )
    return payload


def measured_wall_for(stable_id: str, lane: str) -> float:
    """Return one deterministic measured wall for a session, by identity hash.

    The same stable id always replays the same measured duration, the mapping
    spans the whole sorted vector (hash-uniform), and no randomness source is
    consulted -- reruns are byte-reproducible.
    """

    lanes = load_measured_wall_distribution()["lanes"]
    walls = lanes[lane]["wall_seconds"]
    digest = stable_hash({"stable_id": stable_id, "lane": lane}).removeprefix("sha256:")
    return float(walls[int(digest[:12], 16) % len(walls)])


class VirtualClock:
    """Discrete-event monotonic clock for whole-driver rehearsals.

    Injected as ``DriverDependencies.monotonic`` AND used by the replay lanes'
    ``sleep_until``. Time advances only when every session that declared
    itself in flight is blocked in :meth:`sleep_until`; it then jumps to the
    earliest pending wake. Threads doing real work (the writer committing
    ledger rows, a pool dispatching) hold time still -- their CPU work is
    instantaneous in rehearsal time, exactly the virtual-time contract.
    """

    def __init__(self, start: float = 0.0) -> None:
        """Start the clock; no wall entropy is consulted."""

        self._now = float(start)
        self._condition = threading.Condition()
        self._sleepers: dict[int, float] = {}
        self._in_flight = 0
        self._quiescent_since: Optional[float] = None
        self.reads = 0

    def __call__(self) -> float:
        """Return the current virtual instant (the ``monotonic`` seam)."""

        with self._condition:
            self.reads += 1
            return self._now

    def begin_session(self) -> None:
        """Declare one session in flight; its sleeps gate time advancement."""

        with self._condition:
            self._in_flight += 1
            self._quiescent_since = None

    def end_session(self) -> None:
        """Retire one in-flight session and let waiting sleepers re-evaluate."""

        with self._condition:
            self._in_flight -= 1
            self._quiescent_since = None
            self._condition.notify_all()

    def sleep_until(self, wake_at: float) -> None:
        """Block in virtual time until ``wake_at``; advance once quiescence holds.

        Virtual time advances to the earliest pending wake only after EVERY
        in-flight session has been asleep for a continuous real-wall grace, so
        the production writer's real-time slices (relay admission service,
        commit work) get their chance to start queued siblings first -- the
        difference between measuring the scheduler and measuring the rig.

        Raises
        ------
        RuntimeError
            When no virtual progress happens for the real-wall watchdog bound:
            a wedged rehearsal fails loudly instead of hanging its suite.
        """

        ident = threading.get_ident()
        deadline = time.monotonic() + _WEDGE_WATCHDOG_SECONDS
        with self._condition:
            self._sleepers[ident] = float(wake_at)
            self._quiescent_since = None
            self._condition.notify_all()
            try:
                while self._now < wake_at:
                    self._condition.wait(timeout=0.05)
                    if self._now >= wake_at:
                        break
                    self._advance_if_quiescence_held()
                    if self._now >= wake_at:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "virtual clock made no progress within the wedge "
                            f"watchdog ({_WEDGE_WATCHDOG_SECONDS:.0f}s wall): "
                            f"now={self._now} wake_at={wake_at} "
                            f"in_flight={self._in_flight} sleepers={len(self._sleepers)}"
                        )
            finally:
                self._sleepers.pop(ident, None)
                self._quiescent_since = None
                self._condition.notify_all()

    def _advance_if_quiescence_held(self) -> None:
        """Jump to the earliest pending wake after the quiescent grace held.

        Called with the condition lock held, from a sleeper's poll loop.
        """

        if not self._sleepers or len(self._sleepers) < self._in_flight:
            self._quiescent_since = None
            return
        stamp = self._quiescent_since
        if stamp is None:
            self._quiescent_since = time.monotonic()
            return
        if time.monotonic() - stamp < _QUIESCENT_GRACE_SECONDS:
            return
        earliest = min(self._sleepers.values())
        self._quiescent_since = None
        if earliest > self._now:
            self._now = earliest
            self._condition.notify_all()


@dataclass
class SessionSpan:
    """One simulated session's measured virtual span."""

    stable_id: str
    lane: str
    started_at: float
    ended_at: float

    @property
    def wall_seconds(self) -> float:
        """Return the replayed session wall."""

        return self.ended_at - self.started_at


@dataclass
class SessionLedger:
    """Thread-safe record of every session the rig simulated."""

    spans: list[SessionSpan] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, span: SessionSpan) -> None:
        """Append one completed session span."""

        with self._lock:
            self.spans.append(span)

    def lane_spans(self, lane: str) -> tuple[SessionSpan, ...]:
        """Return the recorded spans for one lane."""

        with self._lock:
            return tuple(span for span in self.spans if span.lane == lane)

    def mean_concurrency(self, lane: str) -> float:
        """Return measured busy-time over elapsed-time for one lane.

        The TE-6 metric: total replayed session seconds divided by the virtual
        span from the lane's first session start to its last session end. A
        perfectly pipelined lane at width W measures ~W; a barriered lane
        collapses toward its tail width.
        """

        spans = self.lane_spans(lane)
        if not spans:
            raise ValueError(f"no {lane} sessions were simulated")
        busy = sum(span.wall_seconds for span in spans)
        elapsed = max(span.ended_at for span in spans) - min(
            span.started_at for span in spans
        )
        if elapsed <= 0:
            raise ValueError(f"{lane} sessions span no virtual time")
        return busy / elapsed

    def first_started_at(self, lane: str) -> float:
        """Return the lane's earliest virtual session start."""

        return min(span.started_at for span in self.lane_spans(lane))


class MeasuredReplaySession:
    """Mixin replaying one measured session wall on the shared virtual clock."""

    def __init__(self, clock: VirtualClock, ledger: SessionLedger, lane: str) -> None:
        """Bind the shared clock, the span ledger, and this lane's name."""

        self._clock = clock
        self._ledger = ledger
        self._lane = lane

    def replay_session(self, stable_id: str) -> None:
        """Consume one measured session wall in virtual time and record it."""

        wall = measured_wall_for(stable_id, self._lane)
        self._clock.begin_session()
        try:
            started = self._clock()
            self._clock.sleep_until(started + wall)
            self._ledger.record(
                SessionSpan(
                    stable_id=stable_id,
                    lane=self._lane,
                    started_at=started,
                    ended_at=self._clock(),
                )
            )
        finally:
            self._clock.end_session()


def build_rig_driver(
    tmp_path: Path,
    *,
    members: int,
    author_concurrency: int,
    checker_concurrency: int = 5,
    scheduler: str = "pipelined",
) -> tuple[Any, VirtualClock, SessionLedger]:
    """Build the real driver over N synthetic members with measured replay lanes.

    Returns the driver, the shared virtual clock (already injected as the
    driver's ``monotonic`` seam), and the session ledger the replay lanes
    record into.
    """

    from dataclasses import replace

    from menagerie.crawler.tests.test_slice_f_driver import (
        FakeAuthor,
        FakeChecker,
        _driver,
        _snapshot,
    )

    clock = VirtualClock()
    ledger = SessionLedger()

    class ReplayAuthor(FakeAuthor):
        """The canned author behind one measured virtual-time session."""

        def __init__(self) -> None:
            super().__init__()
            self._replay = MeasuredReplaySession(clock, ledger, "author")

        def author(self, item: Any, work_root: Path, config: Any, context: Any) -> Any:
            self._replay.replay_session(item.stable_id)
            return super().author(item, work_root, config, context)

    class ReplayChecker(FakeChecker):
        """The canned checker behind measured virtual-time gate sessions."""

        def __init__(self) -> None:
            super().__init__()
            self._replay = MeasuredReplaySession(clock, ledger, "checker")

        def check_metadata(self, artifacts: Sequence[Any], work_root: Path, config: Any) -> Any:
            batch_id = "+".join(
                str(artifact.proposal["stable_id"]) for artifact in artifacts
            )
            self._replay.replay_session(f"metadata:{batch_id}")
            return super().check_metadata(artifacts, work_root, config)

        def check_fidelity(self, artifact: Any, work_root: Path, config: Any) -> Any:
            self._replay.replay_session(
                f"fidelity:{artifact.proposal['stable_id']}"
            )
            return super().check_fidelity(artifact, work_root, config)

    snapshot = _snapshot(tmp_path, count=members)
    driver = _driver(
        tmp_path,
        snapshot,
        author=ReplayAuthor(),
        checker=ReplayChecker(),
        scheduler=scheduler,
        author_concurrency=author_concurrency,
        checker_concurrency=checker_concurrency,
    )
    driver.dependencies = replace(driver.dependencies, monotonic=clock)
    return driver, clock, ledger


def production_shaped_asserts(
    *, status: str, members: int, ledger: SessionLedger
) -> dict[str, bool]:
    """Compute the W17 production-shaped invariant guard over one rehearsal leg.

    The three invariants the scale-time-shaped-blindness class names (the
    step-200 class), computed exactly as the Wave-4 full-scale rehearsal
    computed them (r4-w17-scheduler-rehearsal.json): the driver completed,
    every member's author session ran (no starvation), and none ran twice
    (no double-author). Fail-closed: an empty author ledger reads as total
    starvation, never as vacuous green.
    """

    counts: dict[str, int] = {}
    for span in ledger.lane_spans("author"):
        counts[span.stable_id] = counts.get(span.stable_id, 0) + 1
    return {
        "driver_completed": status == "complete",
        "no_starvation_every_member_authored": len(counts) == members,
        "no_member_authored_twice": all(n == 1 for n in counts.values()),
    }


@dataclass(frozen=True)
class RigResult:
    """One rehearsal run's outcome and measured lane numbers."""

    status: str
    virtual_elapsed: float
    author_mean_concurrency: float
    author_sessions: int
    first_gate_at: Optional[float]


def run_rig(
    tmp_path: Path,
    *,
    members: int,
    author_concurrency: int,
    checker_concurrency: int = 5,
    scheduler: str = "pipelined",
) -> RigResult:
    """Run one virtual-time rehearsal leg and return its measured numbers."""

    driver, clock, ledger = build_rig_driver(
        tmp_path,
        members=members,
        author_concurrency=author_concurrency,
        checker_concurrency=checker_concurrency,
        scheduler=scheduler,
    )
    result = driver.run()
    checker_spans = ledger.lane_spans("checker")
    return RigResult(
        status=result.status,
        virtual_elapsed=clock(),
        author_mean_concurrency=ledger.mean_concurrency("author"),
        author_sessions=len(ledger.lane_spans("author")),
        first_gate_at=(
            min(span.started_at for span in checker_spans) if checker_spans else None
        ),
    )
