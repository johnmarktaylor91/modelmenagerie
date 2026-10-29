"""Terminal-review sweep concurrency: pooled sessions, serial canonical result.

DX-54: the terminal-review backlog sweep was the last serial multi-minute
checker lane in the leg -- 114 gates x ~146 s ran strictly one at a time in
200a (4.61 h of pre-selection wall; leg 37 burned a 1.55 h sweep burst with
zero authors running). The sweep now buys its ``check_terminal`` sessions
through a sweep-scoped ``_CheckerSessionPool`` while the single canonical
writer keeps every classification, validation, ledger append, and review-held
supersession in stable-ID order.

The load-bearing invariant (the FT-(c) ordering proof) is POSITIONAL: the
sweep is one synchronous call before the WS-1 selection, joining every session
before it returns, so a hold released by the sweep still binds in the same leg
-- selection re-reads the durable models ledger afterwards. These tests pin
the two halves the proof demands:

* the sessions genuinely overlap up to the configured bound (a serial lane
  deadlocks the rendezvous loudly instead of passing quietly);
* completion order steers NOTHING -- a sweep whose sessions finish backwards
  records the byte-identical canonical result of the fully serial sweep, and
  the review-held supersessions land in stable-ID order at every width.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import pytest

from menagerie.crawler.driver import CommandAuthorLane
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    CheckerOutcome,
    DriverConfig,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_author_wave_concurrency import _canonical_projection
from menagerie.crawler.tests.test_slice_f_driver import (
    _BLOCKED_SOURCE_PAYLOAD,
    CheckerScript,
    _driver,
    _install_executor_lane,
    _paths,
    _snapshot,
)
from menagerie.crawler.tests.test_terminal_review_backlog_e2e import (
    RecordingTerminalChecker,
    _backlog_census_events,
)


class ObservedTerminalChecker(RecordingTerminalChecker):
    """Terminal checker recording per-session occupancy from pool threads.

    ``hold`` runs while the session is "in flight" -- the exact window a
    concurrent sweep must overlap. Mutations are lock-guarded because the
    sweep pool genuinely calls this fake from worker threads.
    """

    def __init__(
        self,
        tag: str,
        *,
        script: Optional[CheckerScript] = None,
        hold=None,
        after=None,
    ) -> None:
        super().__init__(tag, script=script)
        self._hold = hold or (lambda _key: None)
        self._after = after or (lambda _key: None)
        self._observer_lock = threading.Lock()
        self._live = 0
        self.peak_concurrency = 0
        self.completed: list[str] = []

    def check_terminal(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        key = str(artifact.author_result.binding.stable_id)
        with self._observer_lock:
            self._live += 1
            self.peak_concurrency = max(self.peak_concurrency, self._live)
        try:
            self._hold(key)
            with self._observer_lock:
                return super().check_terminal(artifact, work_root, config)
        finally:
            with self._observer_lock:
                self._live -= 1
                self.completed.append(key)
            self._after(key)


class _ReverseCompletionChain:
    """Force the sweep's sessions to complete in exactly reverse stable order.

    Session ``i`` blocks until session ``i + 1`` has recorded its completion,
    so completion order is reverse sweep order by construction. The chain is a
    concurrency assertion in its own right: a serial sweep deadlocks here
    loudly (timeout) rather than passing quietly.
    """

    def __init__(self, order: dict[str, int], *, timeout: float = 60.0) -> None:
        self._order = order
        self._timeout = timeout
        self._done = [threading.Event() for _ in order]

    def hold(self, key: str) -> None:
        successor = self._order[key] + 1
        if successor >= len(self._done):
            return
        if not self._done[successor].wait(timeout=self._timeout):
            raise AssertionError(
                f"sweep session {self._order[key]} waited {self._timeout:g}s for its "
                "successor: the sweep is not running its terminal sessions concurrently"
            )

    def after(self, key: str) -> None:
        self._done[self._order[key]].set()


def _settle_unreviewed_blocked_terminals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int
):
    """Leg 1: settle ``count`` review-free failed terminals (the sweep backlog)."""

    snapshot = _snapshot(tmp_path, count=count)
    _install_executor_lane(monkeypatch, _BLOCKED_SOURCE_PAYLOAD)
    result = _driver(tmp_path, snapshot, author=CommandAuthorLane(("fake-author",))).run()
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    assert _backlog_census_events(paths) == []
    return snapshot, paths


@pytest.mark.heavy
def test_terminal_review_sessions_overlap_up_to_the_configured_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two backlog sessions must rendezvous inside one sweep.

    The hold is a barrier of width two: under the historical serial sweep the
    first session waits for a sibling that cannot exist and the barrier times
    out, so this test cannot pass without real session overlap.
    """

    snapshot, paths = _settle_unreviewed_blocked_terminals(tmp_path, monkeypatch, 2)
    barrier = threading.Barrier(2, timeout=30.0)
    checker = ObservedTerminalChecker("sweep", hold=lambda _key: barrier.wait())

    result = _driver(tmp_path, snapshot, checker=checker, checker_concurrency=2).run()

    assert result.status == "complete"
    assert sorted(checker.terminal_stable_ids) == sorted(
        item.stable_id for item in snapshot.items
    )
    assert checker.peak_concurrency == 2
    census = _backlog_census_events(paths)
    assert len(census) == 1
    details = census[-1]["details"]
    assert details["dispatched"] == 2
    assert details["classes"]["confirmed"]["count"] == 2


@pytest.mark.heavy
def test_serial_sweep_configuration_keeps_exactly_one_session_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``checker_concurrency=1`` must restore the historical fully serial sweep."""

    snapshot, paths = _settle_unreviewed_blocked_terminals(tmp_path, monkeypatch, 3)
    checker = ObservedTerminalChecker("sweep")

    result = _driver(tmp_path, snapshot, checker=checker, checker_concurrency=1).run()

    assert result.status == "complete"
    assert checker.peak_concurrency == 1
    details = _backlog_census_events(paths)[-1]["details"]
    assert details["dispatched"] == 3
    assert details["classes"]["confirmed"]["count"] == 3


@pytest.mark.heavy
def test_reverse_completion_order_yields_the_serial_sweep_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep whose sessions finish backwards records the serial result.

    The REFUSED verdict makes this the strongest determinism probe the sweep
    owns: every head earns a review-held model supersession, so a commit
    driven by completion order rather than sweep order would reorder model
    ``record_seq`` assignment, gate append order, and the census lists -- and
    the canonical projection would diverge from the serial baseline. This is
    the DX-54 stable-order acceptance at N > 1.
    """

    serial_root = tmp_path / "serial"
    concurrent_root = tmp_path / "concurrent"
    serial_root.mkdir()
    concurrent_root.mkdir()

    serial_snapshot, serial_paths = _settle_unreviewed_blocked_terminals(
        serial_root, monkeypatch, 3
    )
    concurrent_snapshot, concurrent_paths = _settle_unreviewed_blocked_terminals(
        concurrent_root, monkeypatch, 3
    )

    sweep_order = sorted(item.stable_id for item in concurrent_snapshot.items)
    chain = _ReverseCompletionChain({key: index for index, key in enumerate(sweep_order)})

    serial = _driver(
        serial_root,
        serial_snapshot,
        checker=ObservedTerminalChecker(
            "sweep", script=CheckerScript(terminal_verdict="rejected")
        ),
        checker_concurrency=1,
    ).run()
    concurrent_checker = ObservedTerminalChecker(
        "sweep",
        script=CheckerScript(terminal_verdict="rejected"),
        hold=chain.hold,
        after=chain.after,
    )
    concurrent = _driver(
        concurrent_root,
        concurrent_snapshot,
        checker=concurrent_checker,
        checker_concurrency=3,
    ).run()

    assert serial.status == concurrent.status
    # Guaranteed by construction; asserted so a broken chain can never silently
    # weaken the equality below into a same-order comparison.
    assert concurrent_checker.completed == list(reversed(sweep_order))

    # The review-held supersessions landed in stable-ID sweep order, not
    # completion order: the models ledger's append order IS the sweep order.
    held_order = [
        record["stable_id"]
        for record in scan_jsonl(concurrent_paths.ledgers.models)
        if record["status"].get("human_review", {}).get("required") is True
    ]
    assert held_order == sweep_order

    assert _canonical_projection(concurrent_paths, concurrent_root) == (
        _canonical_projection(serial_paths, serial_root)
    )
