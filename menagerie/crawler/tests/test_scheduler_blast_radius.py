"""BL-8 permanent tripwires: scheduler fault isolation and bounded admission.

Four fail-closed contracts, each of which used to fail open:

* **Blast radius (a).** Any single member's fault -- an unroutable award dead
  end, a host-owned execution refusal, an escaped per-member raise -- costs
  exactly THAT member (a durable typed quarantine park); the leg CONTINUES
  and every sibling keeps its progress.
* **No-progress quarantine (b).** The no-progress tripwires (scheduler round
  and metadata gate round) quarantine the stuck member; they never kill the
  leg.
* **Bounded admission (d).** The wave pools admit at most the declared
  running+lookahead window into their executors; an early usage pause
  executes no more sessions than that window and CANCELS everything held or
  queued instead of buying it.
* **Containment pass-through.** Campaign-level control flow and
  malformed-PROGRAM errors are a CLOSED set that always re-raises; the
  per-member containment can never absorb a pause, a shutdown, or a code bug
  into a member quarantine.
"""

from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, cast

import pytest

import sys

from menagerie.crawler import scheduler as scheduler_module
from menagerie.crawler.driver_admission import (
    CommandEnvironmentBackend,
    _AuthorWavePool,
    _CheckerSessionPool,
    _CheckerSpendScope,
    _observed_interpreter_facts,
)
from menagerie.crawler.env_lifecycle import EnvironmentBackendUnavailableError
from menagerie.crawler.driver_contracts import (
    MEMBER_CONTAINMENT_PASSTHROUGH,
    AuthorCampaignTeardown,
    AuthorArtifact,
    AuthorBackoffError,
    AuthorUsagePause,
    DriverPaused,
    DriverShutdown,
    EnvironmentBinding,
    RetryableOperatorError,
    WorkItem,
)
from menagerie.crawler.policy import SandboxUnavailableError
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    Reservation,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.scheduler_parks import (
    SCHEDULER_QUARANTINE_DISPOSITION,
    load_scheduler_quarantine_rows,
    scheduler_quarantine_path,
)
from menagerie.crawler.tests.test_author_wave_concurrency import (
    ObservedAuthor,
    PausingAuthor,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakePauseScheduler,
    ScriptedForward,
    _driver,
    _paths,
    _snapshot,
)


def _quarantine_rows(paths: Any) -> tuple[Mapping[str, Any], ...]:
    """Load the durable scheduler-quarantine rows below one driver rig."""

    records_root = paths.ledgers.models.parent.parent
    return load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)])


# ---------------------------------------------------------------------------
# (a) blast radius: member X's unroutable refusal leaves the leg RUNNING
# ---------------------------------------------------------------------------


class OneMemberSandboxDeadForward(ScriptedForward):
    """Raise the host-owned sandbox refusal for exactly one member."""

    def __init__(self, dead_id: str) -> None:
        super().__init__()
        self._dead_id = dead_id

    def forward(  # type: ignore[override]
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Refuse the designated member; run every sibling normally."""

        if str(artifact.proposal["stable_id"]) == self._dead_id:
            raise SandboxUnavailableError("failed:sandbox-unavailable")
        return super().forward(artifact, environment, cold_runs, work_root)


@pytest.mark.smoke
@pytest.mark.parametrize("scheduler", ["pipelined", "waves"])
def test_member_unroutable_refusal_leaves_the_leg_running(
    tmp_path: Path, scheduler: str
) -> None:
    """The BL-8 (a) headline: one member's host fault costs that member only.

    Member X's execution hits a host-owned refusal (``sandbox-unavailable-v1``:
    a park, never a verdict). Before BL-8 that member became the
    ``_award_recorded_run -> False`` unroutable dead end and the no-progress
    tripwire killed the WHOLE leg. Now: X is quarantined behind a durable
    typed park, the leg runs to a close, and every sibling holds its ``runs``
    award.
    """

    snapshot = _snapshot(tmp_path, count=4)
    dead_id = snapshot.items[1].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        forward=OneMemberSandboxDeadForward(dead_id),
        scheduler=scheduler,
    ).run()

    # The leg RAN and CLOSED; one member's refusal never became a leg death.
    assert result.status == "complete-with-deferrals"
    paths = _paths(tmp_path, snapshot)
    heads: dict[str, Mapping[str, Any]] = {}
    for record in scan_jsonl(paths.ledgers.models):
        heads[str(record["stable_id"])] = record
    # Every sibling kept its progress; the faulted member holds NO record.
    assert dead_id not in heads
    for item in snapshot.items:
        if item.stable_id != dead_id:
            assert heads[item.stable_id]["status"]["code"] == "runs"
    # The member is accounted by the durable typed park, not silence.
    rows = _quarantine_rows(paths)
    assert {row["stable_id"] for row in rows} == {dead_id}
    assert rows[0]["disposition"] == SCHEDULER_QUARANTINE_DISPOSITION
    assert rows[0]["scheduler_quarantine"]["reason_code"] == "non-model-fault"
    events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "scheduler-member-quarantined"
    ]
    assert {event["details"]["stable_id"] for event in events} == {dead_id}


@pytest.mark.smoke
def test_no_progress_tripwire_quarantines_the_member_not_the_leg(tmp_path: Path) -> None:
    """BL-8 (b): a member the leg can never route settles as a quarantine.

    Member X is filtered out of host-capacity admission every round, so it can
    never author its way to settlement. The old tripwire raised
    ``DriverIntegrationError("scheduler made no durable routing progress")``
    and voided every sibling's leg; now X quarantines with the typed
    ``no-durable-routing-progress`` reason and the leg closes.
    """

    snapshot = _snapshot(tmp_path, count=3)
    stuck_id = snapshot.items[0].stable_id
    driver = _driver(tmp_path, snapshot, scheduler="pipelined")
    real_admit = driver._admit_within_host_capacity  # noqa: SLF001

    def filtering_admit(work, artifacts, operational, **kwargs):  # noqa: ANN001, ANN202
        admitted = real_admit(work, artifacts, operational, **kwargs)
        return tuple(item for item in admitted if item.stable_id != stuck_id)

    driver._admit_within_host_capacity = filtering_admit  # type: ignore[method-assign]  # noqa: SLF001

    result = driver.run()
    assert result.status == "complete-with-deferrals"
    paths = _paths(tmp_path, snapshot)
    heads = {str(r["stable_id"]): r for r in scan_jsonl(paths.ledgers.models)}
    assert stuck_id not in heads
    for item in snapshot.items[1:]:
        assert heads[item.stable_id]["status"]["code"] == "runs"
    rows = _quarantine_rows(paths)
    assert {row["stable_id"] for row in rows} == {stuck_id}
    assert rows[0]["scheduler_quarantine"]["reason_code"] == "no-durable-routing-progress"
    assert rows[0]["scheduler_quarantine"]["site"] == "scheduler-round"


@pytest.mark.smoke
def test_containment_passthrough_is_the_closed_campaign_set() -> None:
    """The pass-through set is CLOSED: exactly campaign flow + program bugs.

    Narrowing it silently absorbs a pause into a member park (a usage pause
    becomes 20k quarantines); widening it with data-shaped errors
    (``KeyError``/``ValueError``) re-arms the leg-killer for ordinary member
    facts. Either drift must be a reviewed change that edits this census.
    """

    assert MEMBER_CONTAINMENT_PASSTHROUGH == (
        KeyboardInterrupt,
        SystemExit,
        GeneratorExit,
        AuthorCampaignTeardown,
        DriverShutdown,
        DriverPaused,
        AuthorBackoffError,
        AuthorUsagePause,
        RetryableOperatorError,
        NameError,
        AttributeError,
        ImportError,
        RecursionError,
        MemoryError,
        SystemError,
    )
    for excluded in (KeyError, IndexError, TypeError, ValueError):
        assert not issubclass(excluded, MEMBER_CONTAINMENT_PASSTHROUGH), (
            "data-shaped errors ARE member facts and must be containable"
        )


@pytest.mark.smoke
def test_blast_radius_census_every_containment_site_is_wired() -> None:
    """Source census over the four per-member containment sites (BL-8 a/b).

    The blast-radius census enumerated every ``raise`` reachable from the
    per-member path; the fix routes them through ``_record_scheduler_quarantine``
    at four sites. Removing any site (or its pass-through re-raise) re-opens
    the whole-leg blast radius, so their presence is pinned structurally.
    """

    from menagerie.crawler import driver_admission, driver_receipts

    scheduler_source = inspect.getsource(scheduler_module.run_pipelined_leg)
    # Site 1: the award loop -- member faults AND the unroutable dead end.
    assert 'site="award"' in scheduler_source
    assert 'reason_code="award-unroutable"' in scheduler_source
    # Site 2: the scheduler no-progress tripwire quarantines, never raises.
    assert 'site="scheduler-round"' in scheduler_source
    assert 'reason_code="no-durable-routing-progress"' in scheduler_source
    assert "raise DriverIntegrationError(\n            \"scheduler made no" not in (
        scheduler_source
    ), "the leg-killing no-progress raise came back"
    assert "except MEMBER_CONTAINMENT_PASSTHROUGH" in scheduler_source

    admission_source = inspect.getsource(driver_admission)
    # Sites 3 and 4: the env turn's per-member loop and the gate round.
    assert 'site="env-turn"' in admission_source
    assert 'site="gate-round"' in admission_source
    # Site 5 (W5): the author-wave lane-health boundary's member containment.
    # Declared-but-producerless until W5 -- the boundary terminalized machine
    # failed:author instead; its containment now produces the park.
    assert 'site="author-wave"' in admission_source
    assert admission_source.count("except MEMBER_CONTAINMENT_PASSTHROUGH") >= 1

    receipts_source = inspect.getsource(driver_receipts)
    # Site 4: run anchors + the non-model-fault park settles via quarantine.
    assert 'site="run-anchors"' in receipts_source
    assert 'reason_code="non-model-fault"' in receipts_source


# ---------------------------------------------------------------------------
# (d) bounded admission: window + cancel, pool-level (deterministic)
# ---------------------------------------------------------------------------


def _work_items(count: int) -> list[Any]:
    """Return minimal stable-id-bearing stand-ins for pool admission."""

    return [SimpleNamespace(stable_id=f"m{index:02d}") for index in range(count)]


@pytest.mark.smoke
def test_author_pool_admission_is_windowed_and_pause_cancels_queued(tmp_path: Path) -> None:
    """The author pool admits at most 2*degree; cancel drops the rest unrun.

    Sol R3 W1 sharpened the cancel arm: the two workers blocked on the
    worker-to-writer admission relay when ``cancel_pending()`` lands are
    refused typed (``AuthorUsagePause``) instead of being serviced by a later
    ``drain()`` -- a sealed window converts NO blocked buy into spend, so the
    money ledger stays empty and no dispatch passes the admission barrier.
    """

    degree = 2
    release = threading.Event()
    both_blocked = threading.Event()
    entered: list[str] = []
    dispatched: list[str] = []
    lock = threading.Lock()
    admissions: list[str] = []
    authority = PaidSessionAuthority(
        tmp_path / "author-spend.jsonl",
        runtime_root=tmp_path / "runtime",
        clock=lambda: "2026-08-18T12:00:00Z",
    )

    def admit(item: Any, attempt_group: str | None = None) -> Reservation:
        """Record the writer-owned durable admission before dispatch."""

        admissions.append(item.stable_id)
        return authority.reserve(
            scope={
                "campaign_id": "c1-mech",
                "generation_id": "generation-open",
                "lane": "author",
                "dispatch_kind": "author_source",
                "work_generations": ((item.stable_id, f"work-{item.stable_id}"),),
                "run_id": "run-test",
                "machine_id": "host-test",
            },
            policy=BudgetPolicy(100, 100_000.0, 100, 100_000.0, 1.0),
        )

    def dispatch(item: Any, request_admission: Any) -> Any:
        """Request the durable buy before exposing the paid-session start."""

        with lock:
            entered.append(item.stable_id)
            if len(entered) == degree:
                both_blocked.set()
        request_admission()
        with lock:
            dispatched.append(item.stable_id)
        assert release.wait(timeout=30.0)
        return SimpleNamespace(stable_id=item.stable_id)

    pool = _AuthorWavePool(degree, dispatch, admit)
    items = _work_items(12)
    try:
        for item in items:
            pool.submit(item)
        window = 2 * degree
        # Structural bound: the executor NEVER sees more than the window; the
        # remaining eight members are held back in plain Python, unpurchased.
        assert len(pool._futures) == window  # noqa: SLF001
        assert [held.stable_id for held in pool._held] == [  # noqa: SLF001
            item.stable_id for item in items[window:]
        ]
        # Deterministic W1 interleaving: both running workers are blocked on
        # the admission relay (their buys queued, unserviced) when the seal lands.
        assert both_blocked.wait(timeout=30.0)
        cancelled = pool.cancel_pending()
        # Every held member cancels; both queued-but-unstarted futures cancel
        # (the two workers are still blocked inside the first two sessions).
        assert set(cancelled) >= {item.stable_id for item in items[window:]}
        assert len(set(cancelled)) >= len(items) - degree
        release.set()
        drained = pool.drain()
    finally:
        release.set()
        pool.close()
    # W1: a sealed window converts NO blocked buy into spend -- the writer
    # committed nothing and no dispatch passed the admission barrier.
    assert admissions == []
    assert dispatched == []
    started = {stable_id for stable_id, _outcome in drained}
    assert started == set(entered) == {item.stable_id for item in items[:degree]}
    for stable_id, outcome in drained:
        assert isinstance(outcome.error, AuthorUsagePause), (
            f"{stable_id} must surface the typed sealed-window refusal, "
            f"got {outcome.error!r}"
        )
    assert started.isdisjoint(set(cancelled))
    # Post-cancel admission is closed: a late submit buys nothing.
    pool_after = len(pool._futures)  # noqa: SLF001
    pool.submit(cast(WorkItem, SimpleNamespace(stable_id="late")))
    assert len(pool._futures) == pool_after  # noqa: SLF001


@pytest.mark.smoke
def test_checker_pool_admission_is_windowed_and_pause_cancels_queued(tmp_path: Path) -> None:
    """The checker pool mirrors the author pool's window and cancel contract."""

    degree = 2
    release = threading.Event()
    dispatched: list[str] = []
    lock = threading.Lock()
    admissions: list[str] = []
    authority = PaidSessionAuthority(
        tmp_path / "checker-spend.jsonl",
        runtime_root=tmp_path / "runtime",
        clock=lambda: "2026-08-18T12:00:00Z",
    )

    def admit(scope: _CheckerSpendScope) -> Reservation:
        """Record the writer-owned buy before allowing dispatch."""

        spend_id = f"spend-{scope.work_generations[0][0]}"
        admissions.append(spend_id)
        return authority.reserve(
            scope={
                "campaign_id": "c1-mech",
                "generation_id": "generation-open",
                "lane": "checker",
                "dispatch_kind": scope.gate_kind,
                "work_generations": scope.work_generations,
                "gate_kind": scope.gate_kind,
                "run_id": "run-test",
                "machine_id": "host-test",
            },
            policy=BudgetPolicy(100, 100_000.0, 100, 100_000.0, 1.0),
        )

    def make_dispatch(key: str):  # noqa: ANN202
        def dispatch() -> Any:
            with lock:
                dispatched.append(key)
            assert release.wait(timeout=30.0)
            return SimpleNamespace(key=key)

        return dispatch

    pool = _CheckerSessionPool(degree, admit)
    keys = [f"metadata:batch-{index:02d}" for index in range(12)]
    try:
        for index, key in enumerate(keys):
            pool.submit(
                key,
                make_dispatch(key),
                _CheckerSpendScope(
                    gate_kind="metadata_batch",
                    work_generations=((f"m_{index}", f"work-m_{index}"),),
                    checker_identity=None,
                ),
            )
        window = 2 * degree
        assert len(pool._futures) == window  # noqa: SLF001
        assert [held_key for held_key, _d, _s in pool._held] == keys[window:]  # noqa: SLF001
        cancelled = pool.cancel_pending()
        assert set(cancelled) >= set(keys[window:])
        assert len(set(cancelled)) >= len(keys) - degree
        release.set()
        pool.drain()
    finally:
        release.set()
        pool.close()
    assert len(dispatched) <= degree
    assert len(admissions) == len(dispatched)
    assert set(dispatched).isdisjoint(set(cancelled))


# ---------------------------------------------------------------------------
# (f) deterministic node-set: the leg-scoped prompt pin opens fresh per test
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_prompt_pin_scope_is_per_test_so_node_sets_are_order_independent() -> None:
    """BL-8 (f): the driver-slice failing node-set is order-INDEPENDENT.

    The checker-prompt pin is process-global state with a leg-scoped
    contract; before BL-8 whichever test pinned first decided which LATER
    tests exploded with ``CheckerPromptRotatedError`` -- a failure set that
    was a function of collection order, which no count can gate. The corpus
    conftest now opens a fresh pin scope around every test (and the driver
    opens one at every locked leg start), so this test observing an UNPINNED
    scope -- regardless of what ran before it -- is the order-independence
    witness.
    """

    from menagerie.crawler import checker_dispatch

    assert checker_dispatch._LEG_CHECKER_PROMPT_PIN is None  # noqa: SLF001
    pinned = checker_dispatch.leg_checker_prompt_sha256()
    assert checker_dispatch._LEG_CHECKER_PROMPT_PIN == pinned  # noqa: SLF001
    # The driver opens the scope at each locked leg start, so an in-process
    # relaunch behaves exactly like the fresh process the contract assumes.
    from menagerie.crawler import driver as driver_module_live

    assert "begin_checker_prompt_leg()" in inspect.getsource(
        driver_module_live.CrawlerDriver._run_locked  # noqa: SLF001
    )


# ---------------------------------------------------------------------------
# (e) hung env wrapper: the parent-owned deadline holds, group-killed, bounded
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_wrapper_parent_deadline_group_kills_and_stays_typed() -> None:
    """BL-8 (e): a wedged wrapper is killed as a GROUP under a bounded reap.

    The pre-BL-8 expiry path killed only the wrapper root and then waited
    UNBOUNDED for the corpse, so one D-state wrapper still pinned the writer
    thread, every ledger flock, and the completion partition forever. The
    deadline must fire typed (the same retryable host condition BL-9e routes
    to a park) and return promptly.
    """

    backend = cast(CommandEnvironmentBackend, SimpleNamespace(_POST_KILL_REAP_SECONDS=5.0))
    started = time.monotonic()
    with pytest.raises(EnvironmentBackendUnavailableError, match="parent-owned"):
        CommandEnvironmentBackend._run_wrapper_with_parent_deadline(  # noqa: SLF001
            backend,
            [sys.executable, "-c", "import time; time.sleep(30)"],
            action="probe",
            deadline=0.3,
            environment=None,
        )
    assert time.monotonic() - started < 10.0
    # The healthy path is unchanged: exit status and captured text survive.
    completed = CommandEnvironmentBackend._run_wrapper_with_parent_deadline(  # noqa: SLF001
        backend,
        [sys.executable, "-c", "print('ok')"],
        action="probe",
        deadline=30.0,
        environment=None,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == "ok"


@pytest.mark.smoke
def test_interpreter_facts_probe_reuses_the_parent_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BL-8 (e): the env-python facts probe is bounded by BL-9e's probe row.

    ``_observed_interpreter_facts`` executes the just-built prefix's own
    python on the writer thread; it was the one remaining unbounded wait on
    the env-turn path (a hung store symlink wedged the leg forever). It now
    reuses the parent-owned deadline table's ``probe`` entry and refuses as
    retryable infrastructure, never hanging.
    """

    hung = tmp_path / "hung-python"
    hung.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    hung.chmod(0o755)
    monkeypatch.setattr(
        CommandEnvironmentBackend, "_ACTION_DEADLINE_SECONDS", {"probe": 0.3}
    )
    started = time.monotonic()
    with pytest.raises(EnvironmentBackendUnavailableError, match="probe deadline"):
        _observed_interpreter_facts(hung)
    assert time.monotonic() - started < 10.0


# ---------------------------------------------------------------------------
# (d) bounded admission: the early-pause fixture, end to end
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_early_pause_executes_no_more_than_the_admission_window(tmp_path: Path) -> None:
    """An early usage pause buys at most the window; the rest is cancelled.

    Twelve members, concurrency two (window four), the FIRST member pauses.
    Historically "bounded concurrency" submitted the whole cohort and the
    pause drained it to completion -- eleven more sessions of spend after the
    decision to stop spending. Now the non-pausing spend is bounded by the
    declared window, cancelled members hold no cache, and the resume
    re-authors exactly the members whose sessions never ran.
    """

    count, concurrency = 12, 2
    window = 2 * concurrency
    snapshot = _snapshot(tmp_path, count=count)
    pausing_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)
    pause_started = threading.Event()

    def hold(stable_id: str) -> None:
        """Keep sibling sessions in flight until the pause has fired."""

        if stable_id != pausing_id:
            pause_started.wait(timeout=30.0)

    author = PausingAuthor(pausing_id, hold=hold)
    real_author = author.author

    def pausing_author(item, work_root, config, context):  # noqa: ANN001, ANN202
        if item.stable_id == pausing_id:
            time.sleep(0.05)
            pause_started.set()
        return real_author(item, work_root, config, context)

    author.author = pausing_author  # type: ignore[method-assign]

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=concurrency,
        pause_scheduler=scheduler,
    ).run()
    assert result.status == "paused:usage-limit"
    assert scheduler.calls == 1

    executed = [stable_id for stable_id in author.completed if stable_id != pausing_id]
    # The declared bound: a pause never buys more than the admission window.
    assert len(executed) <= window
    # Everything past the window plus one completed-slot feed was never handed
    # to the executor at all, let alone run.
    admissible = {item.stable_id for item in snapshot.items[: window + 1]}
    assert set(author.completed) <= admissible
    # Executed siblings keep their preserved sessions; cancelled members hold
    # no cache and are re-authored on resume -- nothing is silently lost.
    for stable_id in executed:
        assert (paths.work_root / stable_id / "driver-author-artifact.json").is_file()
    cancelled_ids = {
        item.stable_id
        for item in snapshot.items
        if item.stable_id != pausing_id and item.stable_id not in set(executed)
    }
    for stable_id in cancelled_ids:
        assert not (paths.work_root / stable_id / "driver-author-artifact.json").exists()
    resumed_author = ObservedAuthor()
    resumed = _driver(
        tmp_path,
        snapshot,
        author=resumed_author,
        author_concurrency=concurrency,
    ).run()
    assert resumed.status == "complete"
    assert cancelled_ids <= set(resumed_author.calls)
    for stable_id in executed:
        assert stable_id not in resumed_author.calls
