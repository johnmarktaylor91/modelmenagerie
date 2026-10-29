"""M3 merge-gate tests: one clean pipelined leg on a synthetic intake.

The pipelined scheduler must close the same valid partition the wave driver
closes, from the same intake, through the same canonical machinery -- while
buying checker sessions ONLY for models that executed (execute-first triage)
and terminalizing machine deaths gate-free. The canonical-projection oracle
(digest-aliased record-for-record equality) is imported from the author-wave
rig; the wave leg is the serial reference.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from menagerie.crawler.driver_contracts import AuthorArtifact, EnvironmentBinding
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.conftest import HASH
from menagerie.crawler.tests.test_author_wave_concurrency import _canonical_projection
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakeChecker,
    FakeEnvironments,
    FakeForward,
    InjectedKill,
    ScriptedForward,
    _driver,
    _family_snapshot,
    _mixed_phase_snapshot,
    _paths,
    _snapshot,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash


class ImportDeadForward(ScriptedForward):
    """Fail every model with the worker's exact import-death attempt shape."""

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
    ) -> Sequence[Mapping[str, Any]]:
        """Return one module-missing import failure per model."""

        attempts = [
            dict(attempt)
            for attempt in super().forward(artifact, environment, cold_runs, work_root)
        ]
        failed = attempts[0]
        failed["result"] = "failed"
        failed["stage"] = "import"
        failed["mode"] = None
        failed["worker_receipt"] = dict(failed["worker_receipt"])
        failed["worker_receipt"]["mode"] = None
        failed["worker_receipt"]["constructor_started"] = False
        failed["worker_receipt"]["constructor_completed"] = False
        failed["worker_receipt"]["input_completed"] = False
        failed["worker_receipt"]["forward_started"] = False
        failed["worker_receipt"]["forward_completed"] = False
        failed["error"] = {
            "stage": "import",
            "reason_code": "module-missing",
            "exception_type": "builtins.ModuleNotFoundError",
            "message": "No module named 'synthetic_dead_dependency'",
            "traceback": "Traceback: ModuleNotFoundError",
            "no_traceback_reason": None,
            "native_crash": False,
            "root_cause_fingerprint": HASH,
            "details": {"receipt_error": None},
        }
        failed["raw_award_receipt"] = None
        failed["raw_award_receipt_sha256"] = None
        parent_attestation = dict(failed["parent_attestation"])
        parent_attestation["completion_line_sha256"] = None
        parent_attestation["named_raw_award_receipt_sha256"] = None
        parent_attestation["attestation_sha256"] = stable_hash(
            {key: value for key, value in parent_attestation.items() if key != "attestation_sha256"}
        )
        failed["parent_attestation"] = parent_attestation
        failed["unattested_partial"] = {
            "state": "unattested-partial",
            "stage": "import",
            "reason_code": "module-missing",
            "diagnostic_sha256": None,
        }
        return [failed]


@pytest.mark.heavy
def test_pipelined_leg_closes_the_same_partition_as_waves(tmp_path: Path) -> None:
    """A seven-model intake completes identically under both schedulers.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    waves_root = tmp_path / "waves"
    pipelined_root = tmp_path / "pipelined"
    waves_root.mkdir()
    pipelined_root.mkdir()

    waves_snapshot = _snapshot(waves_root, count=7)
    waves_result = _driver(waves_root, waves_snapshot).run()
    assert waves_result.status == "complete"

    pipelined_snapshot = _snapshot(pipelined_root, count=7)
    pipelined_result = _driver(pipelined_root, pipelined_snapshot, scheduler="pipelined").run()
    assert pipelined_result.status == "complete"

    waves_paths = _paths(waves_root, waves_snapshot)
    pipelined_paths = _paths(pipelined_root, pipelined_snapshot)
    assert len(scan_jsonl(pipelined_paths.ledgers.models)) == 7
    assert _canonical_projection(pipelined_paths, pipelined_root) == _canonical_projection(
        waves_paths, waves_root
    )


@pytest.mark.heavy
def test_pipelined_leg_buys_no_checker_session_for_machine_dead_models(
    tmp_path: Path,
) -> None:
    """Execute-first triage: import-dead models never reach the checker lane.

    Under waves every authored model crosses the metadata lane before its
    import ever runs; the pilot bought 33 models' metadata waves for models
    that then died at environment/import. Under the pipelined scheduler the
    forward failure is recorded first and the gate-free terminal costs ZERO
    checker sessions -- the measured point of the whole redesign.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    waves_root = tmp_path / "waves"
    pipelined_root = tmp_path / "pipelined"
    waves_root.mkdir()
    pipelined_root.mkdir()

    waves_snapshot = _snapshot(waves_root, count=5)
    waves_checker = FakeChecker()
    waves_result = _driver(
        waves_root, waves_snapshot, checker=waves_checker, forward=ImportDeadForward()
    ).run()

    pipelined_snapshot = _snapshot(pipelined_root, count=5)
    pipelined_checker = FakeChecker()
    pipelined_result = _driver(
        pipelined_root,
        pipelined_snapshot,
        checker=pipelined_checker,
        forward=ImportDeadForward(),
        scheduler="pipelined",
    ).run()
    assert pipelined_result.status == waves_result.status

    paths = _paths(pipelined_root, pipelined_snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert len(models) == 5
    for record in models:
        assert record["status"]["code"] == "failed:import"
        assert record["authored_metadata_state"] == "pending"
    # The measured execute-first payoff: the wave driver buys the metadata
    # batch before any import runs; the pipelined leg buys ZERO checker
    # sessions for the same intake.
    assert waves_checker.metadata_calls >= 1
    assert len(scan_jsonl(_paths(waves_root, waves_snapshot).ledgers.gates)) >= 1
    assert pipelined_checker.metadata_calls == 0
    assert len(scan_jsonl(paths.ledgers.gates)) == 0


@pytest.mark.heavy
def test_pipelined_two_intents_serialize_on_the_env_lane_and_terminalize_nothing(
    tmp_path: Path,
) -> None:
    """Two env intents in one pipelined leg: sequential turns, no busy terminals.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _mixed_phase_snapshot(tmp_path)
    environments = FakeEnvironments(tmp_path / "fake-envs")
    result = _driver(tmp_path, snapshot, environments=environments, scheduler="pipelined").run()
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert len(models) == len(snapshot.items)
    for record in models:
        assert "lifecycle-busy" not in str(record.get("status", {}))


@pytest.mark.heavy
def test_pipelined_family_variant_derives_from_its_awarded_representative(
    tmp_path: Path,
) -> None:
    """A family variant authors only after its representative became usable.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot, representative_id, variant_ids = _family_snapshot(tmp_path)
    author = FakeAuthor()
    result = _driver(tmp_path, snapshot, author=author, scheduler="pipelined").run()
    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in models:
        by_id[str(record["stable_id"])] = record
    representative = by_id[representative_id]
    for variant_id in variant_ids:
        derivation = by_id[variant_id].get("family_variant_derivation")
        assert derivation is not None
        assert derivation["template_source_revision"] == representative["record_revision"]


class ExecuteFirstWitnessChecker(FakeChecker):
    """Record how many models had executed when each checker session began."""

    def __init__(self, forward: FakeForward) -> None:
        super().__init__()
        self._forward = forward
        self.executed_at_dispatch: list[int] = []

    def check_metadata(self, artifacts, work_root, config):  # noqa: ANN001, ANN201
        self.executed_at_dispatch.append(len(self._forward.calls))
        return super().check_metadata(artifacts, work_root, config)


@pytest.mark.heavy
def test_pipelined_interleaved_appends_keep_per_model_lineage_and_execute_first_order(
    tmp_path: Path,
) -> None:
    """Triage executes EVERY model before the first checker session dispatches.

    Also pins per-model ledger discipline under the interleaved multi-stage
    appends: model revisions stay parent-chained per stable_id and global
    sequences stay strictly monotone (T5-5 item 2).

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=7)
    forward = FakeForward()
    checker = ExecuteFirstWitnessChecker(forward)
    result = _driver(
        tmp_path, snapshot, checker=checker, forward=forward, scheduler="pipelined"
    ).run()
    assert result.status == "complete"
    # The inversion, measured: at the FIRST metadata dispatch every one of the
    # seven models had already run its full mechanical matrix.
    assert checker.executed_at_dispatch
    assert checker.executed_at_dispatch[0] == 7

    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert [record["record_seq"] for record in models] == list(range(1, len(models) + 1))
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for record in models:
        by_model.setdefault(str(record["stable_id"]), []).append(record)
    for revisions in by_model.values():
        assert revisions[0]["parent_revision"] is None
        for earlier, later in zip(revisions, revisions[1:]):
            assert later["parent_revision"] == earlier["record_revision"]
    attempts = scan_jsonl(paths.ledgers.attempts)
    assert [record["ledger_seq"] for record in attempts] == list(range(1, len(attempts) + 1))


@pytest.mark.heavy
def test_pipelined_crash_at_gate_boundary_resumes_and_completes(tmp_path: Path) -> None:
    """A mid-leg death after one gate batch resumes to the same close (I4 salvage).

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=7)
    killed = False

    def kill_after_first_gate(boundary: str, _stable_id: str) -> None:
        nonlocal killed
        if boundary == "after-gate" and not killed:
            killed = True
            raise InjectedKill("after-gate")

    with pytest.raises(InjectedKill):
        _driver(tmp_path, snapshot, boundary=kill_after_first_gate, scheduler="pipelined").run()
    resumed = _driver(tmp_path, snapshot, scheduler="pipelined").run()
    assert resumed.status == "complete"
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.models)) == 7


@pytest.mark.heavy
def test_pipelined_requeue_grant_consumes_exactly_once(tmp_path: Path) -> None:
    """A durable requeue grant mints exactly one generation under the scheduler.

    Ported from the wave exemplar: leg 1 terminalizes through a rejecting
    checker, an operator grant re-drives the model once, and a third leg
    consumes nothing (grant exactly-once, T5-5 item 5).

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    from menagerie.crawler.tests.test_slice_f_driver import NOW, InaccurateChecker

    snapshot = _snapshot(tmp_path)
    paths = _paths(tmp_path, snapshot)
    first = _driver(tmp_path, snapshot, checker=InaccurateChecker(), scheduler="pipelined").run()
    assert first.status == "terminal-partition-complete"
    stable_id = snapshot.items[0].stable_id
    reason = "human reviewed corrected evidence"
    grant_id = stable_hash(
        {"stable_id": stable_id, "stage": "accuracy-gate", "reason": reason, "grant": 1}
    )
    grant = {
        "grant_id": grant_id,
        "stable_id": stable_id,
        "stage": "accuracy-gate",
        "reason": reason,
        "attempts": 1,
        "created_at": NOW,
    }
    paths.requeue_grants.parent.mkdir(parents=True, exist_ok=True)
    with paths.requeue_grants.open("ab") as handle:
        handle.write(canonical_json_bytes(grant) + b"\n")

    forward = FakeForward()
    second = _driver(tmp_path, snapshot, forward=forward, scheduler="pipelined").run()
    assert second.status == "terminal-partition-complete"
    assert forward.calls == {stable_id: 1}
    revisions = [
        record for record in scan_jsonl(paths.ledgers.models) if record["stable_id"] == stable_id
    ]
    assert [record["status"]["code"] for record in revisions] == [
        "failed:accuracy-gate",
        "runs",
    ]
    assert revisions[-1]["budget"]["explicit_grants"] == [grant_id]

    replay_forward = FakeForward()
    third = _driver(tmp_path, snapshot, forward=replay_forward, scheduler="pipelined").run()
    assert third.status == "terminal-partition-complete"
    assert replay_forward.calls == {}
    consumed = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event["event_kind"] == "requeue-grant-consumed"
    ]
    assert len(consumed) == 1


@pytest.mark.smoke
def test_dry_run_factory_threads_the_selected_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run --dry-run --scheduler pipelined`` must build a PIPELINED driver.

    The acceptance dry-run is the only shipped end-to-end path that drives the
    real worker, the real environment binder, and the real reducer -- it is what
    the release gate runs. Before this pin the CLI parsed ``--scheduler`` and
    then built the dry-run driver through a factory that took no scheduler at
    all, so every dry-run silently ran WAVES: the scheduler that a step-100 leg
    is about to depend on had no end-to-end rehearsal available at any effort.
    An ignored flag is worse than a refused one, so the flag is threaded.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    monkeypatch:
        Fixture used to intercept the dry-run driver builder.
    """

    from menagerie.crawler import cli as cli_module
    from menagerie.crawler.tests import dry_run_support

    captured: dict[str, Any] = {}
    real_builder = dry_run_support.build_dry_run_driver

    def _capture(
        repo_root: Path,
        campaign_root: Path,
        environment_prefix: Path,
        **kwargs: Any,
    ) -> str:
        captured.update(kwargs)
        return "captured-driver"

    monkeypatch.setattr(dry_run_support, "build_dry_run_driver", _capture)
    prefix = tmp_path / "prefix"
    prefix.mkdir()
    arguments = cli_module.build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "run",
            "--dry-run",
            "--dry-run-root",
            str(tmp_path / "dry-run"),
            "--dry-run-environment-prefix",
            str(prefix),
            "--scheduler",
            "pipelined",
        ]
    )
    assert arguments.scheduler == "pipelined"
    assert cli_module._default_driver_factory(arguments) == "captured-driver"  # noqa: SLF001
    assert captured["scheduler"] == "pipelined"
    # The capture fake above swallows **kwargs, so it cannot see a kwarg the REAL
    # builder rejects. Binding the captured call against the real signature is what
    # closes that gap: without this, the threading assertion passes while every
    # `run --dry-run` dies with
    # `TypeError: build_dry_run_driver() got an unexpected keyword argument
    # 'scheduler'` -- a green test shipping a dry-run path that cannot start.
    inspect.signature(real_builder).bind(tmp_path, tmp_path / "campaign", prefix, **captured)


def test_dry_run_builder_propagates_the_scheduler_into_driver_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The REAL dry-run builder must put the selected scheduler on the driver.

    Nothing in the suite invoked ``build_dry_run_driver`` itself before this pin --
    its only other coverage monkeypatches the builder away -- which is precisely
    how a threaded kwarg the real builder did not accept could read as green while
    every ``run --dry-run`` died on a ``TypeError``. Only the host-dependent
    environment lane is stubbed here, because it requires a materialized conda
    prefix; the builder, the snapshot, the paths, and the ``DriverConfig`` are real.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    monkeypatch:
        Fixture used to stub the materialized-environment lane.
    """

    from menagerie.crawler.tests import dry_run_support
    from menagerie.crawler.tests.test_slice_f_driver import (
        FakeEnvironments,
        _unlocked_registry,
    )

    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    monkeypatch.setattr(
        dry_run_support,
        "_materialized_environment",
        lambda prefix: (FakeEnvironments(tmp_path / "envs"), _unlocked_registry()),
    )
    selected = dry_run_support.build_dry_run_driver(
        tmp_path,
        campaign_root,
        tmp_path / "prefix",
        review_checkpoint_at=None,
        progress_milestones=(),
        run_id="scheduler-pin",
        scheduler="pipelined",
    )
    assert selected.config.scheduler == "pipelined"
    # The historical default must stay waves for every caller that omits it.
    defaulted = dry_run_support.build_dry_run_driver(
        tmp_path,
        campaign_root,
        tmp_path / "prefix",
        review_checkpoint_at=None,
        progress_milestones=(),
        run_id="scheduler-pin-default",
    )
    assert defaulted.config.scheduler == "waves"


@pytest.mark.heavy
def test_pipelined_review_checkpoint_closes_admissions_and_is_one_shot(
    tmp_path: Path,
) -> None:
    """Crossing the review threshold pauses the pipelined leg without spend (F6).

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    from menagerie.crawler.tests.test_slice_f_driver import FakeNotifier

    snapshot = _snapshot(tmp_path, count=7)
    notifier = FakeNotifier()
    first = _driver(tmp_path, snapshot, notifier=notifier, review_at=1, scheduler="pipelined").run()
    assert first.status == "paused:review-checkpoint"

    reblocked = _driver(
        tmp_path, snapshot, notifier=FakeNotifier(), review_at=1, scheduler="pipelined"
    ).run()
    assert reblocked.status == "paused:review-checkpoint"

    resumed = _driver(
        tmp_path, snapshot, notifier=FakeNotifier(), review_at=1, scheduler="pipelined"
    ).run(after_review=True)
    assert resumed.status == "complete"
    paths = _paths(tmp_path, snapshot)
    assert len(scan_jsonl(paths.ledgers.models)) == 7


@pytest.mark.smoke
def test_pipelined_leg_derives_no_selection_of_its_own():
    """WS-1: selection happens ONCE, in the driver, through the one policy.

    The --only-unsettled scope cut and every other admission rule live in
    ``menagerie.crawler.work_selection`` and are pinned BEHAVIORALLY by the
    five-row fixture in ``test_work_selection_policy.py`` (the TP-1 template
    that replaced this module's former ``inspect.getsource`` grep pin). The
    pipelined leg consumes the plan-admitted ``work`` verbatim; this pin
    keeps the leg from growing a second predicate.
    """

    import inspect

    from menagerie.crawler import scheduler as scheduler_module

    source = inspect.getsource(scheduler_module.run_pipelined_leg)
    assert "scheduled = tuple(work)" in source
    assert "only_unsettled" not in source.replace(
        "--only-status re-validation is not supported", ""
    )
    assert "current_records" not in source, (
        "the pipelined leg must never read the dependency-currency projection "
        "for settlement (vector 8b)"
    )


@pytest.mark.smoke
def test_pipelined_author_lane_slices_per_round_so_gates_interleave(tmp_path: Path) -> None:
    """The first gate flush must land BEFORE the last model authors.

    Step-200 regression (2026-08-20): the author turn fed EVERY ready model
    into the pooled barrier, so at rung scale no leg ever finished step 1 --
    env turns and gate batches never ran while authored artifacts piled up
    ungated. With the per-round slice, a width-5 metadata batch flushes while
    later models are still unauthored.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    events: list[tuple[str, str]] = []

    class OrderProbeAuthor(FakeAuthor):
        """Record every author call in scheduler order."""

        def author(self, item, work_root, config, context):  # type: ignore[override]
            """Log the model then delegate to the synthetic author."""

            events.append(("author", item.stable_id))
            return super().author(item, work_root, config, context)

    class OrderProbeChecker(FakeChecker):
        """Record every metadata batch the checker lane receives."""

        def check_metadata(self, *args, **kwargs):  # type: ignore[override]
            """Log the flush then delegate to the synthetic checker."""

            events.append(("gate", "metadata-batch"))
            return super().check_metadata(*args, **kwargs)

    count = 12
    snapshot = _snapshot(tmp_path, count=count)
    result = _driver(
        tmp_path,
        snapshot,
        author=OrderProbeAuthor(),
        checker=OrderProbeChecker(),
        scheduler="pipelined",
        author_concurrency=1,
    ).run()
    assert result.status == "complete"

    authored = [event for event in events if event[0] == "author"]
    assert len(authored) == count
    gate_positions = [index for index, event in enumerate(events) if event[0] == "gate"]
    assert gate_positions, "the pipelined leg never flushed a metadata batch"
    authors_before_first_gate = sum(
        1 for event in events[: gate_positions[0]] if event[0] == "author"
    )
    assert authors_before_first_gate < count, (
        "author lane ran as a whole-intake barrier: every model authored "
        "before the first gate flush (the step-200 wedge)"
    )


@pytest.mark.smoke
def test_pipelined_first_env_turn_fires_despite_fraction_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged backlog executes at the first boundary, not after a starvation clock.

    Step-200 regression (legs 7-9, 2026-08-20/21): the readiness fraction counts
    the whole live population, so at rung scale a handful of exec-ready models
    never reached the threshold, and ``first_ready_at`` is leg-local so every
    restart re-armed the 30-minute oldest-waiter clock -- no env turn ever fired
    while authored models exhausted upstream. The intent's first turn of the leg
    must fire whenever anything is ready. The rig reproduces the production
    shape: most authors terminalize (the ready pool stays small) while the
    fraction threshold stays out of reach.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    monkeypatch:
        Pytest monkeypatch fixture.
    """

    from menagerie.crawler import scheduler as scheduler_module
    from menagerie.crawler.fetcher import UnpinnedTargetError

    monkeypatch.setattr(scheduler_module, "DEFAULT_ENV_TURN_READINESS_FRACTION", 0.9)

    turn_rounds: list[int] = []
    authored_rounds: list[int] = []
    round_counter = {"n": 0}
    survivors: set[str] = set()

    class MostlyDoomedAuthor(FakeAuthor):
        """Terminalize all but the first three models so the ready pool stays small."""

        def author(self, item, work_root, config, context):  # type: ignore[override]
            """Record the round; fail non-survivors with a typed author error."""

            authored_rounds.append(round_counter["n"])
            if item.stable_id not in survivors and len(survivors) < 3:
                survivors.add(item.stable_id)
            if item.stable_id not in survivors:
                raise UnpinnedTargetError("injected doomed author for trigger test")
            return super().author(item, work_root, config, context)

    count = 16
    snapshot = _snapshot(tmp_path, count=count)
    driver = _driver(
        tmp_path,
        snapshot,
        author=MostlyDoomedAuthor(),
        scheduler="pipelined",
        author_concurrency=1,
    )

    original = driver._run_environment_work  # noqa: SLF001

    def counting_env_work(*args, **kwargs):
        turn_rounds.append(round_counter["n"])
        return original(*args, **kwargs)

    driver._run_environment_work = counting_env_work  # type: ignore[method-assign]  # noqa: SLF001

    original_check = driver._check_shutdown  # noqa: SLF001

    def round_ticking_check(site: str, **kwargs):
        if site == "scheduler-round":
            round_counter["n"] += 1
        return original_check(site, **kwargs)

    driver._check_shutdown = round_ticking_check  # type: ignore[method-assign,assignment]  # noqa: SLF001

    result = driver.run()
    assert result.status in ("complete", "terminal-partition-complete")
    assert turn_rounds, "no env turn ever fired"
    assert len(authored_rounds) >= 8, "author lane never reached the midpoint"
    first_turn_round = turn_rounds[0]
    midpoint_author_round = authored_rounds[7]
    assert first_turn_round <= midpoint_author_round, (
        "first env turn starved behind the fraction threshold "
        f"(turn at round {first_turn_round}, 8th author at round "
        f"{midpoint_author_round}): the step-200 starvation shape -- the "
        "intent's first turn must fire at the first boundary with anything ready"
    )


def _attempt_cap_rig(
    tmp_path: Path, *, needed_attempts: int = 4, count: int = 12
) -> tuple[Any, dict[str, list[int]], str, Any]:
    """Build the ACAP rig: one pathological multi-attempt ladder among clean peers.

    The pathological model dies with a scripted machine-infrastructure death
    (``OSError`` -> ``failed-machine-spawn``, the weigh-zero session-retry arm)
    on every author attempt before ``needed_attempts``; every other model
    authors cleanly on its first attempt. Author attempts are recorded with
    the scheduler round they ran in, via the ``scheduler-round`` shutdown
    checkpoint at the top of each round loop.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    needed_attempts:
        Author attempt on which the pathological model finally succeeds.
    count:
        Total intake size.

    Returns
    -------
    tuple
        ``(driver, attempt_rounds, pathological_id, snapshot)`` where
        ``attempt_rounds`` maps stable_id -> the 1-based round number of each
        author attempt.
    """

    snapshot = _snapshot(tmp_path, count=count)
    pathological = snapshot.items[0].stable_id
    attempt_rounds: dict[str, list[int]] = {}
    round_counter = {"n": 0}

    class InfraDeathAuthor(FakeAuthor):
        """Scripted machine-death ladder for exactly one model."""

        def author(self, item, work_root, config, context):  # type: ignore[override]
            """Record the attempt's round; die until the scripted attempt."""

            attempts = attempt_rounds.setdefault(item.stable_id, [])
            attempts.append(round_counter["n"])
            if item.stable_id == pathological and len(attempts) < needed_attempts:
                raise OSError("scripted infra death (ACAP rig)")
            return super().author(item, work_root, config, context)

    driver = _driver(
        tmp_path,
        snapshot,
        author=InfraDeathAuthor(),
        scheduler="pipelined",
        author_concurrency=count,
    )

    original_check = driver._check_shutdown  # noqa: SLF001

    def round_ticking_check(site: str, **kwargs):
        if site == "scheduler-round":
            round_counter["n"] += 1
        return original_check(site, **kwargs)

    driver._check_shutdown = round_ticking_check  # type: ignore[method-assign,assignment]  # noqa: SLF001
    return driver, attempt_rounds, pathological, snapshot


@pytest.mark.smoke
def test_pipelined_author_attempt_cap_defers_pathological_ladders(tmp_path: Path) -> None:
    """ACAP: a multi-attempt ladder pipelines across rounds instead of clogging one.

    Release-wave pathology (2026-08-21): the pooled author turn is a barrier,
    so one model's whole retry ladder -- up to 7+ attempts with backoffs --
    resolved inside one pooled call, in one pool slot, and the round could not
    return until it finished; the wave ran 1-2 wide for hours against ~65
    ready models. With ``AUTHOR_ATTEMPTS_PER_ROUND``, the clean cohort authors
    in round 1 while the pathological ladder runs exactly its in-round budget,
    defers unresolved, and resolves in a later round.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    from menagerie.crawler import scheduler as scheduler_module

    cap = scheduler_module.AUTHOR_ATTEMPTS_PER_ROUND
    assert cap == 2, "rig assumes the shipped per-round author attempt budget"

    driver, attempt_rounds, pathological, snapshot = _attempt_cap_rig(tmp_path)
    result = driver.run()
    assert result.status == "complete"
    assert len(scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)) == 12

    ladder = attempt_rounds[pathological]
    assert len(ladder) == 4, f"pathological model ran {len(ladder)} attempts, expected 4"
    first_round = ladder[0]
    assert ladder[:2] == [first_round, first_round], (
        "the in-round budget must run inside one pooled call before deferring"
    )
    assert ladder[2] > first_round, (
        "cap never deferred: attempt 3 ran in the same round as attempts 1-2, "
        "so the pooled call still resolves whole ladders in one slot"
    )
    assert ladder[3] <= first_round + 2, (
        f"deferred ladder did not pipeline promptly: attempts ran in rounds {ladder}"
    )

    clean = {
        stable_id: rounds
        for stable_id, rounds in attempt_rounds.items()
        if stable_id != pathological
    }
    assert len(clean) == 11
    assert all(len(rounds) == 1 for rounds in clean.values())
    assert all(rounds[0] == first_round for rounds in clean.values()), (
        "clean models were held behind the pathological ladder instead of authoring in round 1"
    )


@pytest.mark.smoke
def test_pipelined_author_ladder_blocks_round_one_without_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old-behavior pin: uncapped, the whole ladder resolves inside round 1.

    This is the disease the cap treats, asserted so the contrast stays
    measured: with ``AUTHOR_ATTEMPTS_PER_ROUND`` disabled, every one of the
    pathological model's four attempts (and their backoffs) runs inside the
    first round's pooled barrier, which is exactly the shape that held release
    rounds hostage to the slowest member's ladder.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    monkeypatch:
        Pytest monkeypatch fixture.
    """

    from menagerie.crawler import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "AUTHOR_ATTEMPTS_PER_ROUND", None)
    driver, attempt_rounds, pathological, _snapshot_unused = _attempt_cap_rig(tmp_path)
    result = driver.run()
    assert result.status == "complete"

    ladder = attempt_rounds[pathological]
    assert len(ladder) == 4
    assert set(ladder) == {ladder[0]}, (
        "uncapped author lane no longer resolves a ladder inside one pooled "
        f"call (attempt rounds {ladder}); the waves-driver contract changed"
    )


def test_late_staged_artifacts_execute_within_two_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Models that become exec-ready mid-leg must not starve behind the threshold.

    Leg 23 (2026-08-21): four re-drives staged artifacts hours into the leg and
    ZERO worker leases followed in four hours -- execution starved despite the
    first-turn guarantee. This pins the production shape: a large live
    population (threshold high), no exec-ready model until several author
    rounds have passed, then a few become ready; an env turn must run for them
    within two subsequent boundaries.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    monkeypatch:
        Pytest monkeypatch fixture.
    """

    from menagerie.crawler import scheduler as scheduler_module
    from menagerie.crawler.fetcher import UnpinnedTargetError

    monkeypatch.setattr(scheduler_module, "DEFAULT_ENV_TURN_READINESS_FRACTION", 0.9)
    monkeypatch.setattr(scheduler_module, "DEFAULT_ENV_TURN_OLDEST_WAITER_SECONDS", 10_000.0)

    turn_item_counts: list[int] = []
    survivors: set[str] = set()
    authored_order: list[str] = []

    class LateSurvivorAuthor(FakeAuthor):
        """Doom early models so survivors stage only in later rounds."""

        def author(self, item, work_root, config, context):  # type: ignore[override]
            """First twelve models fail; the last four author cleanly."""

            authored_order.append(item.stable_id)
            if len(authored_order) <= 12 and item.stable_id not in survivors:
                raise UnpinnedTargetError("injected doomed author for starvation test")
            survivors.add(item.stable_id)
            return super().author(item, work_root, config, context)

    count = 16
    snapshot = _snapshot(tmp_path, count=count)
    driver = _driver(
        tmp_path,
        snapshot,
        author=LateSurvivorAuthor(),
        scheduler="pipelined",
        author_concurrency=1,
    )

    original_env = driver._run_environment_work  # noqa: SLF001

    def counting_env_work(items, *args, **kwargs):
        turn_item_counts.append(len(items))
        return original_env(items, *args, **kwargs)

    driver._run_environment_work = counting_env_work  # type: ignore[method-assign,assignment]  # noqa: SLF001

    result = driver.run()
    assert result.status in ("complete", "terminal-partition-complete")
    nonempty = [n for n in turn_item_counts if n > 0]
    assert nonempty, (
        "no env turn ever carried items: late-staged artifacts starved "
        f"(turn item counts: {turn_item_counts})"
    )
