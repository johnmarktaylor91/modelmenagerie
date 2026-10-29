"""BL-9 tripwires: env-turn effort economics, durable admission, bounded parks.

The pilot bug this suite pins closed: the per-intent-generation environment
effort cap (attempts=2) was spent by NORMAL successful scheduling, and its
exhaustion mass-parked whole intents behind a human-only release -- six
firings during the pilot, one of which deferred 46/60 (65%) of a rung. The
fixes under test:

* (a) a successful environment turn resets the budget; only genuine failed
  turns consume it;
* (b) env-turn admission is durable: outcomes persist in an append-only
  ledger and are rederived by replay on resume, so a supervisor/wake restart
  cannot re-permit full-prefix env-build churn;
* (c) an intent-setup park is clearable while a leg runs (receipts land in
  the operator sidecar, no driver lock), and clearance also restores the
  turn budget;
* (d) effort exhaustion parks AT MOST ONE representative member behind the
  human-only release; the cohort is withheld with self-releasing rows;
* park lifecycle census: every machine-deferral park kind names BOTH its
  release predicate and its drain tool.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.driver_admission import (
    ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION,
    ENVIRONMENT_SETUP_FAILURE_PARKED_DISPOSITION,
    _deferral_row_fact_still_holds,
    _quarantine_is_cleared,
    append_environment_clearance_event,
    environment_clearance_events,
    environment_clearance_path,
)
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.effort import EffortCapExceeded, StageCap
from menagerie.crawler.env_lifecycle import (
    PerIntentGenerationEffort,
    SequentialEnvironmentLifecycle,
    environment_turn_effort_path,
    replay_environment_turn_attempts,
)
from menagerie.crawler.environment_rebuild import (
    load_rebuild_deferral_rows,
    rebuild_deferral_path,
)
from menagerie.crawler.envs import load_environment_registry
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.status import MACHINE_DEFERRAL_DISPOSITIONS
from menagerie.crawler.tests.test_env_failure_routing import (
    _effort_cap,
    _FailingEnvironments,
)
from menagerie.crawler.tests.test_slice_c_envs_lifecycle import (
    CreateFailingBackend,
    FakeEnvironmentBackend,
    _copy_env_specs,
    _swap_in_fake_family,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

pytestmark = pytest.mark.smoke


def _core_lifecycle(
    tmp_path: Path,
    *,
    backend: FakeEnvironmentBackend,
    effort: PerIntentGenerationEffort,
):
    """Build one real lifecycle over the committed fake core family."""

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    lifecycle = SequentialEnvironmentLifecycle(
        backend,
        effort,
        env_root=tmp_path / "active",
        disk_free=lambda _path: 10**9,
        minimum_free_bytes=0,
    )
    return intent, lifecycle


def test_successful_env_turn_does_not_consume_retry_budget(tmp_path: Path) -> None:
    """Normal scheduling never spends the cap; only genuine retries consume it.

    Pre-fix, three successful turns of the same intent-generation exhausted
    the attempts=2 cap and parked the intent behind a human-only release.
    """

    effort = PerIntentGenerationEffort(StageCap(attempts=2, seconds=3600, bytes=10**12))
    backend = CreateFailingBackend([])
    backend.fail = False
    intent, lifecycle = _core_lifecycle(tmp_path, backend=backend, effort=effort)

    # Far past the pilot's six firings: every successful turn resets.
    for _turn in range(7):
        lifecycle.run(intent, use=lambda _prefix, _probes: None)
        assert effort(intent).usage("environment").attempts == 0

    # Genuine retries still consume, and the cap still trips fail-closed.
    backend.fail = True
    for _failed_turn in range(2):
        with pytest.raises(Exception, match="synthetic build failure"):
            lifecycle.run(intent, use=lambda _prefix, _probes: None)
    assert effort(intent).usage("environment").attempts == 2
    with pytest.raises(EffortCapExceeded):
        lifecycle.run(intent, use=lambda _prefix, _probes: None)


def test_env_turn_admission_survives_resume(tmp_path: Path) -> None:
    """Admission state is replay-derived from the ledger, never process memory.

    Persist two failed turns, simulate a supervisor/wake restart (a brand new
    provider and lifecycle over the same ledger), and require the rederived
    state to refuse admission BEFORE the backend runs -- no full-prefix
    env-build rechurn.
    """

    records_root = tmp_path / "records"
    ledger = environment_turn_effort_path(records_root)
    cap = StageCap(attempts=2, seconds=3600, bytes=10**12)

    first_backend = CreateFailingBackend([])
    intent, first_lifecycle = _core_lifecycle(
        tmp_path, backend=first_backend, effort=PerIntentGenerationEffort(cap, ledger_path=ledger)
    )
    for _failed_turn in range(2):
        with pytest.raises(Exception, match="synthetic build failure"):
            first_lifecycle.run(intent, use=lambda _prefix, _probes: None)
    # Each failed turn reached the backend once (the trailing "remove" is the
    # lifecycle's unconditional teardown of the partial prefix).
    assert first_backend.events.count("create-failed") == 2

    # Simulated restart: everything in-memory is gone; only the ledger remains.
    resumed_effort = PerIntentGenerationEffort(cap, ledger_path=ledger)
    key = (intent.name, intent.lock.target, intent.generation)
    assert replay_environment_turn_attempts(ledger, key) == 2
    assert resumed_effort(intent).usage("environment").attempts == 2

    resumed_backend = CreateFailingBackend([])
    resumed_lifecycle = SequentialEnvironmentLifecycle(
        resumed_backend,
        resumed_effort,
        env_root=tmp_path / "active-resumed",
        disk_free=lambda _path: 10**9,
        minimum_free_bytes=0,
    )
    with pytest.raises(EffortCapExceeded):
        resumed_lifecycle.run(intent, use=lambda _prefix, _probes: None)
    # THE churn guard: the refused admission never reached the backend.
    assert resumed_backend.events == []

    # A recorded SUCCESS is equally durable: it zeroes the replayed count.
    success_root = tmp_path / "records-success"
    success_ledger = environment_turn_effort_path(success_root)
    healthy_backend = CreateFailingBackend([])
    healthy_backend.fail = False
    healthy_effort = PerIntentGenerationEffort(cap, ledger_path=success_ledger)
    healthy_lifecycle = SequentialEnvironmentLifecycle(
        healthy_backend,
        healthy_effort,
        env_root=tmp_path / "active-success",
        disk_free=lambda _path: 10**9,
        minimum_free_bytes=0,
    )
    healthy_lifecycle.run(intent, use=lambda _prefix, _probes: None)
    assert replay_environment_turn_attempts(success_ledger, key) == 0
    restarted = PerIntentGenerationEffort(cap, ledger_path=success_ledger)
    assert restarted(intent).usage("environment").attempts == 0


def _run_effort_exhausted_leg(tmp_path: Path, count: int):
    """Run one real driver leg whose environment lane reports cap exhaustion."""

    snapshot = _snapshot(tmp_path, count=count)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(
        tmp_path,
        snapshot,
        environments=_FailingEnvironments(tmp_path / "fake-envs", _effort_cap()),
    )
    result = driver.run()
    return snapshot, paths, driver, result


def test_cap_exhaustion_parks_at_most_one_member(tmp_path: Path) -> None:
    """Effort exhaustion seats ONE human-release park; the cohort self-releases."""

    snapshot, paths, driver, result = _run_effort_exhausted_leg(tmp_path, count=4)
    assert result.status == "complete-with-deferrals"
    records_root = paths.ledgers.models.parent.parent
    rows = load_rebuild_deferral_rows([rebuild_deferral_path(records_root)])
    assert {row["stable_id"] for row in rows} == {item.stable_id for item in snapshot.items}

    by_scope: dict[str, list[dict]] = {}
    for row in rows:
        by_scope.setdefault(str(row["quarantine"]["condition_scope"]), []).append(dict(row))
    # THE bound: at most one member behind the human-only release.
    assert len(by_scope.get("intent-setup", ())) == 1
    assert len(by_scope.get("intent-setup-cohort", ())) == len(snapshot.items) - 1
    park_event_ids = {str(row["quarantine"]["event_id"]) for row in rows}
    assert len(park_event_ids) == 1

    events = scan_jsonl(canonical_operational_ledger_path(paths.ledgers.models))
    park_events = [
        event
        for event in events
        if event.get("details", {}).get("disposition")
        == ENVIRONMENT_SETUP_FAILURE_PARKED_DISPOSITION
        and event.get("details", {}).get("representative_stable_id") is not None
        and "stable_id" not in event.get("details", {})
    ]
    assert len(park_events) == 1
    details = park_events[0]["details"]
    representative = by_scope["intent-setup"][0]["stable_id"]
    assert details["representative_stable_id"] == representative
    assert details["cohort_size"] == len(snapshot.items) - 1

    # Cohort rows never re-claim a member by themselves; the representative's
    # row holds on unchanged setup identity (the human-visible fact carrier).
    for cohort_row in by_scope["intent-setup-cohort"]:
        item = cast(
            WorkItem,
            SimpleNamespace(
                route=SimpleNamespace(intent=str(cohort_row["quarantine"]["intent"]))
            ),
        )
        assert _deferral_row_fact_still_holds(driver, item, cohort_row) is False
    representative_row = by_scope["intent-setup"][0]
    rep_item = cast(
        WorkItem,
        SimpleNamespace(
            route=SimpleNamespace(intent=str(representative_row["quarantine"]["intent"]))
        ),
    )
    assert _deferral_row_fact_still_holds(driver, rep_item, representative_row) is True


def test_env_generation_blast_radius_is_bounded(tmp_path: Path) -> None:
    """Repeated exhaustion legs never accrete representatives, and ONE
    operator action (a clearance receipt) releases the entire cohort.

    The pilot's amplifier: one ``core`` env-generation event deferred 46/60 =
    65% of a rung behind a human-only release. Now the human-parked set is
    bounded at one member no matter how many legs observe the exhaustion, and
    releasing it is one receipt, not 46.
    """

    snapshot, paths, driver, _result = _run_effort_exhausted_leg(tmp_path, count=5)
    records_root = paths.ledgers.models.parent.parent

    # Leg 2 over the same standing exhaustion: the cohort re-admitted (its
    # rows self-release) and re-parked -- WITHOUT seating a second
    # representative.
    second = _driver(
        tmp_path,
        snapshot,
        environments=_FailingEnvironments(tmp_path / "fake-envs", _effort_cap()),
    )
    second.run()
    rows = load_rebuild_deferral_rows([rebuild_deferral_path(records_root)])
    setup_rows = [
        row for row in rows if row["quarantine"]["condition_scope"] == "intent-setup"
    ]
    assert len({str(row["stable_id"]) for row in setup_rows}) == 1
    follow_up_events = [
        event
        for event in scan_jsonl(canonical_operational_ledger_path(paths.ledgers.models))
        if event.get("details", {}).get("disposition")
        == ENVIRONMENT_SETUP_FAILURE_PARKED_DISPOSITION
        and "prior_park_event_id" in event.get("details", {})
    ]
    assert follow_up_events, "the second leg must hang its cohort off the standing park"
    assert all(
        event["details"]["representative_stable_id"] is None for event in follow_up_events
    )

    # ONE clearance receipt -- recordable while a leg runs -- releases the
    # representative too; nothing else in the rung needs a human.
    representative_row = setup_rows[0]
    park_event_id = str(representative_row["quarantine"]["event_id"])
    append_environment_clearance_event(
        records_root,
        {
            "event_kind": "campaign-health",
            "event_id": "sha256:" + "ab" * 32,
            "details": {
                "disposition": ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION,
                "quarantine_event_id": park_event_id,
                "evidence": "synthetic repair for the blast-radius tripwire",
            },
        },
    )
    rep_item = cast(
        WorkItem,
        SimpleNamespace(
            route=SimpleNamespace(intent=str(representative_row["quarantine"]["intent"]))
        ),
    )
    assert _deferral_row_fact_still_holds(driver, rep_item, representative_row) is False


def test_clearance_recordable_while_leg_runs(tmp_path: Path) -> None:
    """The clearance tool writes the operator sidecar without the driver lock.

    Pre-fix the tool refused with ``another driver owns .../driver.lock``
    whenever a leg was running, so recovering a parked intent always cost a
    full leg restart. The receipt now lands in the sidecar the running driver
    only reads, and the merged release view honors it.
    """

    from menagerie.crawler.checkpoint import append_canonical_operational_event
    from menagerie.crawler.constants import (
        OPERATIONAL_EVENT_SCHEMA_VERSION,
        OperationalEventKind,
        OperationalEventStatus,
    )
    from menagerie.crawler.driver_contracts import DriverLock
    from menagerie.crawler.operational_identity import mint_operational_event
    from menagerie.crawler.tools import clear_environment_quarantine as tool

    repo_root = tmp_path / "repo"
    records_root = repo_root / "menagerie" / "crawler" / "records"
    models_path = records_root / "models" / "current-shard.jsonl"
    ledger_path = canonical_operational_ledger_path(models_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    park_event = mint_operational_event(
        {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": "2026-08-17T00:00:00Z",
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.RUNNER_FAILED.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 3},
            "current_environment": "core",
            "run_id": "run-test",
            "machine_id": "machine-test",
            "details": {
                "disposition": ENVIRONMENT_SETUP_FAILURE_PARKED_DISPOSITION,
                "intent": "core",
                "target": "osx-arm64",
                "failure_class": "environment-effort-exhausted",
                "condition_scope": "intent-setup",
            },
        }
    )
    append_canonical_operational_event(ledger_path, park_event)

    # A live leg holds the driver lock for its whole duration.
    lock_path = repo_root / ".crawl-local" / "locks" / "driver.lock"
    with DriverLock(lock_path, {"pid": 0, "run_id": "live-leg", "target": None}):
        exit_code = tool.main(
            [
                str(park_event["event_id"]),
                "--evidence",
                "backend repaired out of band (synthetic)",
                "--repo-root",
                str(repo_root),
            ]
        )
    assert exit_code == 0

    sidecar = environment_clearance_path(records_root)
    assert sidecar.is_file()
    receipts = environment_clearance_events(records_root)
    assert len(receipts) == 1
    merged = (*scan_jsonl(ledger_path), *receipts)
    assert _quarantine_is_cleared(merged, str(park_event["event_id"]))

    # Double clears stay refused across BOTH ledgers.
    assert (
        tool.main(
            [
                str(park_event["event_id"]),
                "--evidence",
                "second receipt must refuse",
                "--repo-root",
                str(repo_root),
            ]
        )
        == 1
    )
    assert len(environment_clearance_events(records_root)) == 1


def test_clearance_restores_the_turn_budget_idempotently(tmp_path: Path) -> None:
    """A clearance resets the durable budget exactly once per park event."""

    records_root = tmp_path / "records"
    ledger = environment_turn_effort_path(records_root)
    cap = StageCap(attempts=2, seconds=3600, bytes=10**12)
    backend = CreateFailingBackend([])
    intent, lifecycle = _core_lifecycle(
        tmp_path, backend=backend, effort=PerIntentGenerationEffort(cap, ledger_path=ledger)
    )
    for _failed_turn in range(2):
        with pytest.raises(Exception, match="synthetic build failure"):
            lifecycle.run(intent, use=lambda _prefix, _probes: None)

    park_event_id = "sha256:" + "cd" * 32
    assert lifecycle.reset_effort_after_clearance(intent, park_event_id) is True
    # The reset is durable AND idempotent per park event: replaying the same
    # receipt on a later leg cannot erase a genuine post-clearance attempt.
    with pytest.raises(Exception, match="synthetic build failure"):
        lifecycle.run(intent, use=lambda _prefix, _probes: None)
    assert lifecycle.reset_effort_after_clearance(intent, park_event_id) is False
    key = (intent.name, intent.lock.target, intent.generation)
    assert replay_environment_turn_attempts(ledger, key) == 1

    # After the reset the cleared intent admits and a healthy turn completes,
    # which itself re-zeroes the durable budget.
    backend.fail = False
    lifecycle.run(intent, use=lambda _prefix, _probes: None)
    assert replay_environment_turn_attempts(ledger, key) == 0


#: BL-9 park-lifecycle census: EVERY machine-deferral park kind must name BOTH
#: its release predicate (module attribute, verified importable and callable)
#: and at least one drain tool (repo path, verified to exist). A park kind
#: missing either is an un-drainable trap and fails this closed census.
_PARK_LIFECYCLE_CENSUS: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    "deferred:needs-environment-coverage": {
        "release_predicates": (
            ("menagerie.crawler.driver_admission", "_deferral_row_fact_still_holds"),
        ),
        "drain_tools": (
            ("menagerie/crawler/tools/rotate_setup_artifacts.py", "declare the distribution"),
            ("menagerie/crawler/tools/requeue.py", "new work generation escapes any row"),
        ),
    },
    "deferred:needs-more-memory": {
        "release_predicates": (
            ("menagerie.crawler.driver_admission", "_deferral_row_fact_still_holds"),
        ),
        "drain_tools": (
            ("menagerie/crawler/tools/requeue.py", "--capability host-capability-routing"),
        ),
    },
    "deferred:checker-lane-contract-failure": {
        "release_predicates": (
            ("menagerie.crawler.driver_admission", "_deferral_row_fact_still_holds"),
        ),
        "drain_tools": (
            ("menagerie/crawler/tools/requeue.py", "checker identity rotation or requeue"),
        ),
    },
    "deferred:environment-rebuild-required": {
        "release_predicates": (
            ("menagerie.crawler.driver_admission", "_deferral_row_fact_still_holds"),
        ),
        "drain_tools": (
            ("menagerie/crawler/tools/clear_environment_quarantine.py", "clearance receipt"),
            ("menagerie/crawler/tools/rotate_setup_artifacts.py", "identity rotation"),
        ),
    },
    "deferred:needs-host-disk": {
        "release_predicates": (
            ("menagerie.crawler.driver_admission", "_deferral_row_fact_still_holds"),
        ),
        "drain_tools": (
            ("menagerie/crawler/tools/reap_disk.py", "free disk on the recording host"),
        ),
    },
    "review-held:supersession-pending-grant": {
        "release_predicates": (("menagerie.crawler.grant_parks", "grant_park_released"),),
        "drain_tools": (
            ("menagerie/crawler/tools/grant_parks.py", "--void"),
            ("menagerie/crawler/tools/requeue.py", "--release-park"),
        ),
    },
    "review-held:award-supersession-pending-grant": {
        "release_predicates": (("menagerie.crawler.grant_parks", "grant_park_released"),),
        "drain_tools": (
            ("menagerie/crawler/tools/requeue.py", "--release-park / award revision grant"),
            ("menagerie/crawler/tools/grant_parks.py", "--void"),
        ),
    },
    # R2-1: the selection-exclusion park. Release is the same head-bound
    # predicate; the ONLY drain is the explicit operator void (no grant of any
    # spelling releases it -- its stage sentinel is outside the grant
    # vocabulary by design).
    "review-held:selection-exclusion": {
        "release_predicates": (("menagerie.crawler.grant_parks", "grant_park_released"),),
        "drain_tools": (("menagerie/crawler/tools/grant_parks.py", "--void"),),
    },
    "review-held:terminal-recording-refused": {
        "release_predicates": (("menagerie.crawler.refusal_parks", "refusal_park_released"),),
        "drain_tools": (("menagerie/crawler/tools/refusal_parks.py", "--void"),),
    },
    # W1: a storm-classified exhaustion is withheld transiently; the W14
    # quota-deferral class (or a human grant) funds the fresh generation.
    "machine-deferred:quota-storm": {
        "release_predicates": (
            ("menagerie.crawler.infra_release", "derive_machine_class_release"),
        ),
        "drain_tools": (
            ("menagerie/crawler/tools/requeue.py", "new work generation escapes any row"),
        ),
    },
    # W1/PD-3: the deferral bound is spent; the hold is generation-scoped, so
    # ONLY a human-issued grant minting a fresh generation escapes it.
    "review-held:storm-recidivist": {
        "release_predicates": (
            ("menagerie.crawler.quota_deferral", "storm_recidivist_hold"),
        ),
        "drain_tools": (
            ("menagerie/crawler/tools/requeue.py", "new work generation escapes any row"),
        ),
    },
    # BL-8 (a)/(b), completed by N2: scheduler faults earn a bounded fresh-leg
    # retry, then hold until an explicit append-only operator drain.
    "deferred:scheduler-fault-quarantined": {
        "release_predicates": (
            ("menagerie.crawler.scheduler_parks", "scheduler_quarantine_released"),
        ),
        "drain_tools": (
            (
                "menagerie/crawler/tools/scheduler_quarantines.py",
                "list live scheduler quarantines, append a bare/remedied "
                "operator void, or a budget-restoring machine drain (W12)",
            ),
        ),
    },
}


def test_park_lifecycle_inventory_names_release_and_drain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Closed census: every park kind names a real predicate AND a real tool.

    Fail-closed in both directions: a disposition added to
    ``MACHINE_DEFERRAL_DISPOSITIONS`` without a census entry fails here until
    its release predicate and drain tool are named in the same reviewed
    change, and a census entry whose disposition, predicate, or tool no
    longer exists fails equally.
    """

    import importlib

    assert set(_PARK_LIFECYCLE_CENSUS) == set(MACHINE_DEFERRAL_DISPOSITIONS), (
        "park-lifecycle census and MACHINE_DEFERRAL_DISPOSITIONS diverged; a park "
        "kind must ship with BOTH its release predicate and its drain tool"
    )
    repo_root = Path(__file__).resolve().parents[3]
    for disposition, entry in _PARK_LIFECYCLE_CENSUS.items():
        predicates = entry["release_predicates"]
        tools = entry["drain_tools"]
        assert predicates, f"{disposition} names no release predicate"
        assert tools, f"{disposition} names no drain tool"
        for module_name, attribute in predicates:
            module = importlib.import_module(module_name)
            predicate = getattr(module, attribute, None)
            assert callable(predicate), (
                f"{disposition} release predicate {module_name}.{attribute} is not callable"
            )
        for tool_path, _how in tools:
            assert (repo_root / tool_path).is_file(), (
                f"{disposition} drain tool {tool_path} does not exist"
            )

    # N2's behavioral half, REWRITTEN to the W12 composed contract (never
    # deleted): one stochastic scheduler fault earns a bounded retry counted
    # SINCE THE LAST BUDGET-RESTORING RELEASE. The old form of this block
    # ENCODED the m12646 livelock -- it counted all of history with no void
    # cutoff, so an operator drain bought zero retries -- and it exercised the
    # bare void without ever asserting whether the drain actually bought one.
    from menagerie.crawler.scheduler_parks import (
        SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT,
        SCHEDULER_QUARANTINE_REMEDY,
        append_scheduler_quarantine_row,
        build_scheduler_quarantine_row,
        load_scheduler_quarantine_ledger,
        load_scheduler_quarantine_rows,
        scheduler_quarantine_path,
        scheduler_quarantine_released,
    )

    def _park(created_at: str) -> dict:
        return build_scheduler_quarantine_row(
            stable_id="m_scheduler_parked",
            work_id="work-m_scheduler_parked",
            name="Scheduler Parked",
            campaign_id="c1-mech",
            run_id="run-1",
            machine_id="machine-1",
            created_at=created_at,
            site="scheduler-round",
            reason_code="no-durable-routing-progress",
            fault_evidence="scheduler made no durable routing progress",
        )

    row = _park("2026-08-18T00:00:00Z")
    events = tuple(
        {
            "created_at": f"2026-08-18T00:0{index}:00Z",
            "details": {
                "disposition": "scheduler-member-quarantined",
                "stable_id": row["stable_id"],
                "work_id": row["work_id"],
            },
        }
        for index in range(SCHEDULER_QUARANTINE_RECIDIVISM_LIMIT)
    )
    records_root = tmp_path / "records"
    ledger = scheduler_quarantine_path(records_root)
    append_scheduler_quarantine_row(ledger, row)
    ledger_rows = load_scheduler_quarantine_ledger([ledger])
    assert scheduler_quarantine_released(row, events[:-1], ledger_rows=ledger_rows) is True
    assert scheduler_quarantine_released(row, events, ledger_rows=ledger_rows) is False
    assert SCHEDULER_QUARANTINE_REMEDY == "scheduler-fault-operator-drain"
    assert "operator drain" in row["recheck_hint"]
    assert [item["row_sha256"] for item in load_scheduler_quarantine_rows([ledger])] == [
        row["row_sha256"]
    ]
    from menagerie.crawler.tools.scheduler_quarantines import main as drain_scheduler_park

    base = ("--repo-root", str(tmp_path), "--records-root", str(records_root))
    assert (
        drain_scheduler_park(
            (*base, "--void", str(row["row_sha256"]), "--reason",
             "operator cleared the row without a remedy")
        )
        == 0
    )
    capsys.readouterr()
    assert load_scheduler_quarantine_rows([ledger]) == ()

    # The bare void cleared the row but restored NO budget: the next fault
    # re-parks and holds immediately.
    reparked = append_scheduler_quarantine_row(ledger, _park("2026-08-18T01:00:00Z"))
    reparked_events = events + (
        {
            "created_at": "2026-08-18T01:00:00Z",
            "details": {
                "disposition": "scheduler-member-quarantined",
                "stable_id": row["stable_id"],
                "work_id": row["work_id"],
            },
        },
    )
    ledger_rows = load_scheduler_quarantine_ledger([ledger])
    assert (
        scheduler_quarantine_released(reparked, reparked_events, ledger_rows=ledger_rows)
        is False
    ), "a bare void must not buy a retry"

    # A machine drain (retries exhausted, fixed-in tree declared) DOES buy
    # the bounded retry back -- the previously un-asserted gap. The tool
    # recomputes exhaustion from the durable operational ledger, so the
    # charged park events must exist on disk.
    import json as _json

    from menagerie.crawler.recordio import operational_events_path

    operational = operational_events_path(records_root)
    operational.parent.mkdir(parents=True, exist_ok=True)
    with operational.open("w", encoding="utf-8") as handle:
        for event in reparked_events:
            handle.write(_json.dumps(event) + "\n")
    assert (
        drain_scheduler_park(
            (*base, "--drain", str(reparked["row_sha256"]), "--reason",
             "machine drain after exhausted retries", "--fixed-in-tree", "a" * 40)
        )
        == 0
    )
    capsys.readouterr()
    assert load_scheduler_quarantine_rows([ledger]) == ()
    final = append_scheduler_quarantine_row(ledger, _park("2026-08-18T02:00:00Z"))
    final_events = reparked_events + (
        {
            "created_at": "2026-08-18T02:00:00Z",
            "details": {
                "disposition": "scheduler-member-quarantined",
                "stable_id": row["stable_id"],
                "work_id": row["work_id"],
            },
        },
    )
    ledger_rows = load_scheduler_quarantine_ledger([ledger])
    assert (
        scheduler_quarantine_released(final, final_events, ledger_rows=ledger_rows) is True
    ), "the disclosed machine drain resets the retry budget"
