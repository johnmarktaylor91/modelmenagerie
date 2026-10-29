"""The author admission relay must not starve while the writer works.

Pilot evidence (step-200, 2026-08-21, ``operational/spend.jsonl``): with 12
pool slots the reserve->custody-bind hop took 49ms at median, yet consecutive
author reserves gapped 15.8s at median with 41 gaps of 15-60 minutes, and
bound concurrency averaged 6.17 of 12 over the last eight hours. Every worker
session start blocks on the single canonical writer servicing its
``_AuthorAdmissionRequest`` (the durable reserve), and the writer serviced the
queue ONLY while blocked inside ``pool.claim``/``pool.drain`` -- an inline
terminal-disposition checker session, the non-pool-certain inline author
fallback, or a retry backoff on the writer starved every queued session start
for minutes to an hour.

The fix is a writer-context relay drain: ``_ensure_authors`` binds
``pool.service_admissions`` into the CANONICAL WRITER's own context for the
wave's lifetime, and the writer's other long stretches (the operator-command
outcome poll, infrastructure-retry backoffs, per-item commit boundaries) drain
the queue at bounded points. These tests are the fix's evidence:

* with 18 slots and slow sessions, a writer pinned inside one inline operator
  command still lets >= 15 sibling sessions become CONCURRENTLY CUSTODY-BOUND
  (real reserve + real guard + real durable bind rows);
* the old code's collapse is pinned as the causality control: without the
  writer-context binding, the same inline stretch services NOTHING;
* the reserve site cannot move off the writer: the binding is thread-scoped
  even under a context-propagating runtime, and the pool's public drain
  refuses any other thread typed;
* W1 is unweakened: a sealed window still refuses queued and racing buys.
"""

from __future__ import annotations

import contextvars
import inspect
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, cast

import pytest

from menagerie.crawler import driver_admission as driver_admission_module
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    AuthorUsagePause,
    DriverIntegrationError,
    WorkItem,
)
from menagerie.crawler.paid_session_lifecycle import (
    Reservation,
    authorized_paid_dispatch,
    load_paid_session_rows,
)
from menagerie.crawler.tests.test_paid_session_retry_custody import (
    _ProductionAdmissionStub,
    _authority,
    _ledger,
    _work_item,
)

#: The pilot's configured slot count for the release wave.
_SLOTS = 18

#: The headline threshold: at least this many WORKER sessions concurrently
#: custody-bound while the writer never leaves its inline operator command.
_CONCURRENT_BOUND_FLOOR = 15


def _relay_items(count: int = _SLOTS) -> list[WorkItem]:
    """Return ``count`` distinct author work items for one wave."""

    return [
        _work_item(
            stable_id=f"m_relay_{index:02d}",
            active_work_id=f"work-relay-{index:02d}",
            granted_attempts=None,
        )
        for index in range(count)
    ]


def _guard_command(request_path: Path, *, hold_flag: Optional[Path]) -> tuple[str, ...]:
    """Return one real paid-guard child argv, optionally holding on a flag file.

    ``request_path`` must stay the LAST argument: the dispatch chokepoint
    derives its teardown-record directory from ``argv[-1]``.
    """

    if hold_flag is None:
        body = "import time; time.sleep(120.0)"
        return (sys.executable, "-c", body, str(request_path))
    body = (
        "import os, sys, time\n"
        "while not os.path.exists(sys.argv[1]):\n"
        "    time.sleep(0.05)\n"
    )
    return (sys.executable, "-c", body, str(hold_flag), str(request_path))


def _request_path(tmp_path: Path, name: str) -> Path:
    """Materialize one request.json path for a guard child."""

    path = tmp_path / name / "request.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def _worker_reserves(authority_path: Path, work_ids: set[str]) -> list[Any]:
    """Return every reserved row whose accounting axis names one of ``work_ids``."""

    if not authority_path.exists():
        return []
    matched: list[Any] = []
    for row in load_paid_session_rows(authority_path):
        if row["event"] != "session_reserved":
            continue
        axis = row.get("axis") or {}
        generations = axis.get("work_generations") or []
        if any(generation.get("work_id") in work_ids for generation in generations):
            matched.append(row)
    return matched


def _bound_not_released(authority_path: Path) -> int:
    """Return the ledger's current count of custody-bound, unreleased sessions."""

    if not authority_path.exists():
        return 0
    bound = 0
    for row in load_paid_session_rows(authority_path):
        if row["event"] == "session_custody_bound":
            bound += 1
        elif row["event"] == "session_custody_released":
            bound -= 1
    return bound


def _wave_pool(
    stub: _ProductionAdmissionStub,
    dispatch: Callable[[WorkItem, Callable[[Optional[str]], Reservation]], Any],
    operational: Any,
) -> "driver_admission_module._AuthorWavePool":
    """Return one production pool wired to the REAL admission valve."""

    return driver_admission_module._AuthorWavePool(  # noqa: SLF001
        _SLOTS,
        cast(
            Callable[[WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact],
            dispatch,
        ),
        lambda item, attempt_group: stub._admit_author_spend(  # noqa: SLF001
            item,
            operational,
            attempt_group=attempt_group,
        ),
    )


# ---------------------------------------------------------------------------
# 1. the headline: 18 slots widen past 15 concurrent binds under the fix
# ---------------------------------------------------------------------------


@pytest.mark.heavy
def test_eighteen_slots_reach_fifteen_concurrent_binds_during_an_inline_session(
    tmp_path: Path,
) -> None:
    """A writer pinned in one inline operator command no longer starves starts.

    The writer thread mints its own reservation and enters ONE real inline
    ``_run_operator_command`` (the pilot's worst starvation window: a terminal
    checker session or non-pool-certain author fallback) whose child exits only
    once >= 16 sessions are custody-bound (the writer's own plus >= 15
    workers). Under the pre-fix code this deadlocks-by-starvation -- the relay
    is serviced nowhere -- so the child's exit condition is itself the proof
    that the writer-context drain admitted, reserved, and custody-bound the
    sibling wave while the writer never reached ``pool.claim``.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    operational = _ledger(records=())
    items = _relay_items()
    hold_flag = tmp_path / "release-the-inline-command.flag"

    def dispatch(item: WorkItem, request_admission: Any) -> str:
        """One pooled worker: relay buy, then a real paid guard that holds."""

        command = _guard_command(
            _request_path(tmp_path, item.stable_id), hold_flag=None
        )
        return stub._retry_infrastructure_call(  # noqa: SLF001
            lambda: driver_admission_module._run_operator_command(  # noqa: SLF001
                command,
                timeout_seconds=180.0,
                env=None,
            ).stdout,
            admission=("author", item),
            operational=operational,
            author_admission=request_admission,
        )

    pool = _wave_pool(stub, dispatch, operational)
    peak_bound = {"value": 0}
    observer_stop = threading.Event()

    def observe() -> None:
        """Touch the flag once >= 16 sessions are concurrently custody-bound."""

        while not observer_stop.is_set():
            bound = _bound_not_released(authority.path)
            peak_bound["value"] = max(peak_bound["value"], bound)
            if bound >= _CONCURRENT_BOUND_FLOOR + 1 and not hold_flag.exists():
                hold_flag.write_text("released\n", encoding="utf-8")
            observer_stop.wait(0.1)

    observer = threading.Thread(target=observe, name="ledger-observer", daemon=True)
    observer.start()
    try:
        for item in items:
            pool.submit(item)
        inline_item = _work_item(
            stable_id="m_inline_terminal",
            active_work_id="work-inline",
            granted_attempts=None,
        )
        inline_reservation = stub._admit_author_spend(  # noqa: SLF001
            inline_item, operational
        )
        inline_command = _guard_command(
            _request_path(tmp_path, inline_item.stable_id), hold_flag=hold_flag
        )
        inline_started = time.monotonic()
        with (
            authorized_paid_dispatch(inline_reservation),
            driver_admission_module._author_admission_relay_service(  # noqa: SLF001
                pool.service_admissions
            ),
        ):
            completed = driver_admission_module._run_operator_command(  # noqa: SLF001
                inline_command,
                timeout_seconds=120.0,
                env=None,
            )
        inline_elapsed = time.monotonic() - inline_started
        assert completed.returncode == 0
        stub._settle_paid_session(  # noqa: SLF001
            inline_reservation,
            elapsed_seconds=inline_elapsed,
            completed=True,
        )
    finally:
        observer_stop.set()
        observer.join(timeout=5.0)
        # Tear the held sibling wave down exactly as a campaign halt would.
        pool.halt(trigger="campaign-halt", reason="relay headline teardown")
        pool.drain()
        pool.close()

    assert peak_bound["value"] >= _CONCURRENT_BOUND_FLOOR + 1, (
        "the inline operator command returned without the sibling wave "
        f"reaching {_CONCURRENT_BOUND_FLOOR} concurrent custody-bound sessions "
        f"(peak observed: {peak_bound['value']})"
    )


@pytest.mark.heavy
def test_without_the_writer_binding_an_inline_session_services_nothing(
    tmp_path: Path,
) -> None:
    """The old code's collapse, pinned: no binding means no serviced buys.

    Identical rig, identical inline operator command on the writer -- but the
    writer-context binding is deliberately absent, which IS the pre-fix code
    (the poll loop's drain reads ``None`` and skips). Eighteen queued session
    starts must still be queued, with zero reserves minted for them, when the
    inline command finishes. The sealed-window unwind then refuses every one
    typed (W1), so the collapse cleanup buys nothing either.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    operational = _ledger(records=())
    items = _relay_items()

    def dispatch(item: WorkItem, request_admission: Any) -> str:
        """One pooled worker whose head buy must starve in this arm."""

        command = _guard_command(_request_path(tmp_path, item.stable_id), hold_flag=None)
        return stub._retry_infrastructure_call(  # noqa: SLF001
            lambda: driver_admission_module._run_operator_command(  # noqa: SLF001
                command,
                timeout_seconds=30.0,
                env=None,
            ).stdout,
            admission=("author", item),
            operational=operational,
            author_admission=request_admission,
        )

    pool = _wave_pool(stub, dispatch, operational)
    for item in items:
        pool.submit(item)
    deadline = time.monotonic() + 30.0
    while True:
        with pool._admission_lock:  # noqa: SLF001
            queued = len(pool._admission_requests)  # noqa: SLF001
        if queued == _SLOTS:
            break
        assert time.monotonic() < deadline, "all 18 workers must reach the relay queue"
        time.sleep(0.01)

    inline_item = _work_item(
        stable_id="m_inline_unserviced",
        active_work_id="work-inline-unserviced",
        granted_attempts=None,
    )
    inline_reservation = stub._admit_author_spend(inline_item, operational)  # noqa: SLF001
    inline_command = (
        sys.executable,
        "-c",
        "import time; time.sleep(2.0)",
        str(_request_path(tmp_path, inline_item.stable_id)),
    )
    inline_started = time.monotonic()
    with authorized_paid_dispatch(inline_reservation):
        completed = driver_admission_module._run_operator_command(  # noqa: SLF001
            inline_command,
            timeout_seconds=60.0,
            env=None,
        )
    assert completed.returncode == 0
    stub._settle_paid_session(  # noqa: SLF001
        inline_reservation,
        elapsed_seconds=time.monotonic() - inline_started,
        completed=True,
    )

    worker_work_ids = {item.active_work_id for item in items}
    assert _worker_reserves(authority.path, worker_work_ids) == [], (
        "pre-fix behavior must starve every queued buy while the writer is "
        "inside an inline operator command"
    )
    with pool._admission_lock:  # noqa: SLF001
        still_queued = len(pool._admission_requests)  # noqa: SLF001
    assert still_queued == _SLOTS

    # W1 cleanup: sealing refuses all 18 queued buys typed; nothing is bought.
    pool.halt(trigger="campaign-halt", reason="relay collapse-pin teardown")
    drained = dict(pool.drain())
    pool.close()
    assert len(drained) == _SLOTS
    assert all(
        isinstance(outcome.error, AuthorUsagePause) for outcome in drained.values()
    )
    assert _worker_reserves(authority.path, worker_work_ids) == []


# ---------------------------------------------------------------------------
# 2. the backoff seam: a writer retry sleep drains the relay
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_a_writer_backoff_wait_services_queued_admissions(tmp_path: Path) -> None:
    """``_wait_before_infrastructure_retry`` drains the wave when it owns it.

    Six workers queue their head buys; the writer then sits in one real
    (sleeper-free) backoff wait with the binding installed. Every buy must be
    committed before the wait returns. The same wait WITHOUT the binding is
    the pre-fix behavior and must service nothing -- that arm is this test's
    collapse control at the backoff seam.
    """

    class _BackoffStub(driver_admission_module.AdmissionEnvironmentMixin):
        """Real backoff wait; everything else inert."""

        def __init__(self) -> None:
            from types import SimpleNamespace

            self.dependencies = SimpleNamespace(sleeper=None)
            self._shutdown_event = threading.Event()

    stub = _BackoffStub()
    serviced: list[str] = []
    pool = driver_admission_module._AuthorWavePool(  # noqa: SLF001
        6,
        cast(
            Callable[[WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact],
            lambda item, request_admission: request_admission(None),
        ),
        lambda item, _attempt_group: serviced.append(item.stable_id)  # type: ignore[func-returns-value]
        or cast(Reservation, object()),
    )

    def enqueue(index: int) -> None:
        pool._request_admission(  # noqa: SLF001
            _work_item(
                stable_id=f"m_backoff_{index}",
                active_work_id=f"work-backoff-{index}",
                granted_attempts=None,
            )
        )

    threads = [
        threading.Thread(target=enqueue, args=(index,), daemon=True) for index in range(6)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 10.0
    while True:
        with pool._admission_lock:  # noqa: SLF001
            if len(pool._admission_requests) == 6:  # noqa: SLF001
                break
        assert time.monotonic() < deadline
        time.sleep(0.005)

    # Collapse control first: the plain pre-fix wait must service nothing.
    stub._wait_before_infrastructure_retry(0.15)  # noqa: SLF001
    assert serviced == []

    with driver_admission_module._author_admission_relay_service(  # noqa: SLF001
        pool.service_admissions
    ):
        stub._wait_before_infrastructure_retry(0.15)  # noqa: SLF001
    assert sorted(serviced) == sorted(f"m_backoff_{index}" for index in range(6))
    for thread in threads:
        thread.join(timeout=5.0)
    pool.close()


# ---------------------------------------------------------------------------
# 3. the reserve site cannot move off the writer
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_the_relay_binding_is_thread_scoped_even_when_the_context_propagates() -> None:
    """A propagated writer context must not let another thread drain the relay."""

    calls: list[int] = []
    with driver_admission_module._author_admission_relay_service(  # noqa: SLF001
        lambda: calls.append(threading.get_ident())
    ):
        driver_admission_module._service_author_admission_relay()  # noqa: SLF001
        assert calls == [threading.get_ident()]
        # Hand the EXACT writer context to a foreign thread -- the strongest
        # propagation any runtime could perform -- and require the no-op.
        context = contextvars.copy_context()
        foreign = threading.Thread(
            target=lambda: context.run(
                driver_admission_module._service_author_admission_relay  # noqa: SLF001
            ),
            daemon=True,
        )
        foreign.start()
        foreign.join(timeout=5.0)
        assert calls == [threading.get_ident()]
    # Outside the binding the helper is inert everywhere.
    driver_admission_module._service_author_admission_relay()  # noqa: SLF001
    assert len(calls) == 1


@pytest.mark.smoke
def test_the_public_drain_refuses_any_thread_but_the_writer() -> None:
    """``service_admissions`` is a typed tripwire off the constructing thread."""

    pool = driver_admission_module._AuthorWavePool(  # noqa: SLF001
        2,
        cast(
            Callable[[WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact],
            lambda item, request_admission: None,
        ),
        lambda _item, _attempt_group: cast(Reservation, object()),
    )
    pool.service_admissions()  # the writer's own drain of an empty queue is a no-op
    raised: list[BaseException] = []

    def off_writer() -> None:
        try:
            pool.service_admissions()
        except BaseException as exc:  # noqa: BLE001 -- the assertion target
            raised.append(exc)

    thread = threading.Thread(target=off_writer, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    assert len(raised) == 1
    assert isinstance(raised[0], DriverIntegrationError)
    assert "canonical writer" in str(raised[0])
    pool.close()


@pytest.mark.smoke
def test_a_sealed_window_still_refuses_buys_at_every_drain_point() -> None:
    """W1 through the new drain: sealed windows refuse, never commit."""

    bought: list[str] = []
    pool = driver_admission_module._AuthorWavePool(  # noqa: SLF001
        2,
        cast(
            Callable[[WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact],
            lambda item, request_admission: None,
        ),
        lambda item, _attempt_group: bought.append(item.stable_id)  # type: ignore[func-returns-value]
        or cast(Reservation, object()),
    )
    item = _work_item(
        stable_id="m_sealed", active_work_id="work-sealed", granted_attempts=None
    )
    raced = driver_admission_module._AuthorAdmissionRequest(item, None)  # noqa: SLF001
    pool.cancel_pending()
    with pool._admission_lock:  # noqa: SLF001
        pool._admission_requests.append(raced)  # noqa: SLF001
    pool.service_admissions()
    assert raced.completed.is_set()
    assert isinstance(raced.error, AuthorUsagePause)
    assert bought == []
    pool.close()


@pytest.mark.smoke
def test_a_failed_starvation_note_never_strands_a_popped_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead stderr must lose the diagnostics line, never the session start.

    Adversarial-review finding (relay round 1): the starvation note runs
    between a request's ``popleft()`` and the ``try/finally`` that guarantees
    its completion signal. If the stderr write raised (ENOSPC on the redirected
    leg log, a broken pipe), the popped worker would block forever on an
    unbounded ``Event.wait`` that no seal can reach -- the untyped hang W1
    forbids. The note is therefore best-effort by contract.
    """

    class _DeadStderr:
        def write(self, _text: str) -> int:
            raise OSError(28, "No space left on device")

        def flush(self) -> None:
            raise OSError(28, "No space left on device")

    pool = driver_admission_module._AuthorWavePool(  # noqa: SLF001
        2,
        cast(
            Callable[[WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact],
            lambda item, request_admission: None,
        ),
        lambda _item, _attempt_group: cast(Reservation, object()),
    )
    item = _work_item(
        stable_id="m_dead_stderr", active_work_id="work-dead-stderr", granted_attempts=None
    )
    starved = driver_admission_module._AuthorAdmissionRequest(item, None)  # noqa: SLF001
    starved.enqueued_at = time.monotonic() - 120.0  # over the 30 s note threshold
    with pool._admission_lock:  # noqa: SLF001
        pool._admission_requests.append(starved)  # noqa: SLF001
    monkeypatch.setattr(driver_admission_module.sys, "stderr", _DeadStderr())
    pool.service_admissions()
    assert starved.completed.is_set()
    assert starved.error is None
    assert starved.reservation is not None
    pool.close()


# ---------------------------------------------------------------------------
# 4. the production install and drain sites stay where the fix put them
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_the_wave_installs_the_binding_and_the_writer_stretches_drain_it() -> None:
    """Source pins for the install seam and every writer-side drain point."""

    ensure_authors = inspect.getsource(
        driver_admission_module.AdmissionEnvironmentMixin._ensure_authors
    )
    assert "_author_admission_relay_service(pool.service_admissions)" in ensure_authors
    assert ensure_authors.index(
        "_author_admission_relay_service(pool.service_admissions)"
    ) < ensure_authors.index("_commit_author_wave(")

    commit_wave = inspect.getsource(
        driver_admission_module.AdmissionEnvironmentMixin._commit_author_wave
    )
    assert commit_wave.count("pool.service_admissions()") >= 2

    operator_command = inspect.getsource(driver_admission_module._run_operator_command)
    assert operator_command.index("_service_author_admission_relay()") < (
        operator_command.index("process.communicate(timeout=min(0.1, remaining))")
    )

    backoff_wait = inspect.getsource(
        driver_admission_module.AdmissionEnvironmentMixin._wait_before_infrastructure_retry
    )
    assert "_AUTHOR_ADMISSION_RELAY_SERVICE.get()" in backoff_wait


def test_a_nonwhitelisted_diagnostic_exception_never_strands_a_popped_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ANY diagnostic raise loses the line, never the session start.

    Sol adversarial review, finding 1 (2026-08-21): the round-1 fix
    whitelisted anticipated stderr exception types (OSError, ValueError), so a
    RuntimeError from a re-entrant or wrapper-backed stderr escaped between
    ``popleft()`` and the completion guard and stranded the dequeued worker
    forever. The completion guarantee must begin the instant the request
    leaves the queue, with diagnostics inside it and best-effort by
    construction -- no exception-type list.
    """

    class _HostileStderr:
        def write(self, _text: str) -> int:
            raise RuntimeError("re-entrant stderr wrapper failure")

        def flush(self) -> None:
            raise RuntimeError("re-entrant stderr wrapper failure")

    pool = driver_admission_module._AuthorWavePool(  # noqa: SLF001
        2,
        cast(
            Callable[[WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact],
            lambda item, request_admission: None,
        ),
        lambda _item, _attempt_group: cast(Reservation, object()),
    )
    item = _work_item(
        stable_id="m_hostile_stderr",
        active_work_id="work-hostile-stderr",
        granted_attempts=None,
    )
    starved = driver_admission_module._AuthorAdmissionRequest(item, None)  # noqa: SLF001
    starved.enqueued_at = time.monotonic() - 120.0
    with pool._admission_lock:  # noqa: SLF001
        pool._admission_requests.append(starved)  # noqa: SLF001
    monkeypatch.setattr(driver_admission_module.sys, "stderr", _HostileStderr())
    pool.service_admissions()
    assert starved.completed.is_set(), "hostile diagnostic stranded the dequeued request"
    assert starved.reservation is not None or starved.error is not None
