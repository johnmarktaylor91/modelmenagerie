"""Checker-lane failure scoping: downshift, park, and requeue -- never terminalize.

(One deliberate exception, pinned in ``test_checker_quota_park_terminalization``:
the closed machine-classified QUOTA/budget-exhaustion class terminalizes at the
floor instead of parking, because generation-scoped exhaustion is permanent and
a run-scoped park would re-enter and re-park forever. Every TRANSIENT lane
failure keeps the park semantics this file pins.)

Rung 10 (head50, 2026-08-06) proved the outcome class this file pins: the
first-ever >=10-item metadata batches exceeded the checker's output budget,
every response was refused by the operator wrapper (exit 64,
``permanent-contract-rejection``), and the driver terminalized all 41 batch
members ``failed:runner`` / ``protocol-violation`` -- models charged for a lane
that never rendered a verdict on any of them (m4334, the rung-9b runs-record
holder, died of batch membership alone).

The failure route under test:

1. a checker-lane BATCH failure downshifts along the halving ladder (shared
   mechanism with the batch-economics seam, floor one) and the halves are
   re-asked -- a healthy checker then gates every member;
2. a floor-one item whose gate still cannot be delivered PARKS on the durable
   checker-lane deferral valve: no terminal, accounted by the terminal
   partition, requeued on the next run;
3. genuinely campaign-scoped checker weather (typed retryable transport,
   provider backoff) stays loud and is never absorbed into the ladder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from menagerie.crawler.checker_lane import (
    CHECKER_LANE_DEFERRAL_DISPOSITION,
    CheckerLaneDeferralError,
    append_checker_lane_deferral_row,
    build_checker_lane_deferral_row,
    checker_lane_deferral_path,
    validate_checker_lane_deferral_row,
)
from menagerie.crawler.cli import _status_machine_deferrals
from menagerie.crawler.constants import METADATA_BATCH_MAX
from menagerie.crawler.driver_admission import CheckerPartialResultError
from menagerie.crawler.driver_contracts import (
    CheckerOutcome,
    DriverConfig,
    DriverIntegrationError,
    RetryableOperatorError,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.status import (
    machine_deferral_rejection,
    partition_report,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeForward,
    FidelityAuthor,
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke

#: The operator wrapper's frozen rung-10 exit-64 refusal, verbatim from
#: ``rung10-7777c75b-head50/work/checker/metadata-9843d87089e1fa90/operator-status.json``
#: through ``_raise_for_checker_exit``'s typed message.
RUNG10_EXIT64_MESSAGE = (
    "checker violated the gate contract and its result was refused by the operator "
    "wrapper (exit 64): checker result is partial or has extra items "
    "[permanent-contract-rejection]"
)


class AlwaysPartialChecker(ScriptedChecker):
    """Emit a partial result for every metadata batch, at every size."""

    def __init__(self) -> None:
        """Track the exact batch sizes the driver asked for."""

        super().__init__()
        self.batch_sizes: list[int] = []

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Refuse the batch with the typed partial-result seam error."""

        del work_root, config
        self.batch_sizes.append(len(artifacts))
        raise CheckerPartialResultError("checker result is partial or has extra items")


class OutputBudgetChecker(ScriptedChecker):
    """Refuse any batch wider than the lane's output budget, like rung 10 did."""

    def __init__(self, max_completable: int) -> None:
        """Bind the widest batch the scripted output budget can complete."""

        super().__init__()
        self.max_completable = max_completable
        self.batch_sizes: list[int] = []

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Answer small batches; refuse wide ones with the frozen exit-64 message."""

        self.batch_sizes.append(len(artifacts))
        if len(artifacts) > self.max_completable:
            raise DriverIntegrationError(RUNG10_EXIT64_MESSAGE)
        return super().check_metadata(artifacts, work_root, config)


class StormyChecker(ScriptedChecker):
    """Fail every metadata batch with the campaign-scoped retryable transport error."""

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Raise typed campaign-level weather, never a lane contract failure."""

        del artifacts, work_root, config
        raise RetryableOperatorError("checker command failed (exit 78): model at capacity")


def _valve_rows(tmp_path: Path, snapshot: Any) -> list[Mapping[str, Any]]:
    """Read the checker-lane deferral ledger rows for one campaign root."""

    records_root = _paths(tmp_path, snapshot).ledgers.models.parent.parent
    ledger = checker_lane_deferral_path(records_root)
    return scan_jsonl(ledger) if ledger.is_file() else []


def _expected_partial_park_shape() -> tuple[list[int] | None, list[int] | None]:
    """Expected (overall batch sizes, per-member ladder) for a 4-model roster.

    Multi-item width: the halving ladder is walked to the floor
    (4 -> 2,2 -> 1,1,1,1). At width 1 every batch already IS the floor, so
    each member parks off its own singleton refusal. A middle width has no
    exact pin here (``None``): the floor-reached property below still binds.
    """

    if METADATA_BATCH_MAX >= 4:
        return [4, 2, 2, 1, 1, 1, 1], [4, 2, 1]
    if METADATA_BATCH_MAX == 1:
        return [1, 1, 1, 1], [1]
    return None, None


def test_partial_output_checker_parks_all_members_gateless(tmp_path: Path) -> None:
    """A checker that can never complete a batch parks every member; none terminalizes.

    This is the rung-10 fault shape: partial output at every size. The models
    end PARKED (typed valve rows, partition accounted, requeue-ready) with zero
    fabricated terminals, zero gates, zero fidelity calls, and zero forwards.
    """

    snapshot = _snapshot(tmp_path, count=4)
    checker = AlwaysPartialChecker()
    forward = FakeForward()
    result = _driver(
        tmp_path, snapshot, author=FidelityAuthor(), checker=checker, forward=forward
    ).run()

    assert result.status == "complete-with-deferrals"
    paths = _paths(tmp_path, snapshot)
    assert scan_jsonl(paths.ledgers.models) == []
    assert scan_jsonl(paths.ledgers.gates) == []
    assert checker.fidelity_calls == 0, "a parked model must never reach the fidelity lane"
    assert forward.calls == {}, "a parked model must never reach the worker"
    expected_sizes, expected_ladder = _expected_partial_park_shape()
    assert all(size <= METADATA_BATCH_MAX for size in checker.batch_sizes)
    if expected_sizes is not None:
        assert sorted(checker.batch_sizes, reverse=True) == expected_sizes
    rows = _valve_rows(tmp_path, snapshot)
    assert {row["stable_id"] for row in rows} == {item.stable_id for item in snapshot.items}
    for row in rows:
        assert row["disposition"] == CHECKER_LANE_DEFERRAL_DISPOSITION
        assert row["checker_lane"]["failure_class"] == "partial-result"
        if expected_ladder is not None:
            assert row["checker_lane"]["attempted_batch_sizes"] == expected_ladder
        assert row["checker_lane"]["attempted_batch_sizes"][-1] == 1
        assert row["checker_lane"]["remedy"] == "checker-lane-contract-failure"
    # The completion state names every parked model for the resume tooling.
    state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    assert state["status"] == "complete-with-deferrals"
    deferred = {entry["stable_id"] for entry in state["machine_deferrals"]["deferred"]}
    assert deferred == {item.stable_id for item in snapshot.items}


@pytest.mark.skipif(
    METADATA_BATCH_MAX < 4,
    reason=(
        "the 4 -> 2,2 halving shape needs a batch cap of at least four; at the "
        "calibrated width-1 production config no batch can refuse on width and "
        "the ladder has no consumer (re-arms with a future width increase)"
    ),
)
def test_healthy_checker_after_downshift_gates_every_member(tmp_path: Path) -> None:
    """When smaller batches fit the budget, downshift ends in verdicts, not parks."""

    snapshot = _snapshot(tmp_path, count=4)
    checker = OutputBudgetChecker(max_completable=2)
    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert {record["status"]["code"] for record in models.values()} == {"runs"}
    assert set(models) == {item.stable_id for item in snapshot.items}
    # 4 refused once, then both halves of size 2 completed.
    assert checker.batch_sizes == [4, 2, 2]
    assert _valve_rows(tmp_path, snapshot) == []
    events = scan_jsonl(paths.operational_ledger)
    downshifts = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "checker-batch-downshift"
    ]
    assert len(downshifts) == 1
    assert downshifts[0]["details"]["from_size"] == 4
    assert downshifts[0]["details"]["to_sizes"] == [2, 2]
    assert not [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "checker-lane-parked"
    ]


def test_exit64_wrapper_refusal_parks_with_its_own_classification(tmp_path: Path) -> None:
    """The frozen rung-10 exit-64 refusal parks as ``permanent-contract-rejection``."""

    snapshot = _snapshot(tmp_path, count=2)
    checker = OutputBudgetChecker(max_completable=0)
    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "complete-with-deferrals"
    rows = _valve_rows(tmp_path, snapshot)
    assert {row["checker_lane"]["failure_class"] for row in rows} == {
        "permanent-contract-rejection"
    }
    assert {row["stable_id"] for row in rows} == {item.stable_id for item in snapshot.items}
    for row in rows:
        assert RUNG10_EXIT64_MESSAGE[:60] in row["checker_lane"]["failure_evidence"]


def test_campaign_scoped_checker_weather_stays_loud(tmp_path: Path) -> None:
    """Typed retryable transport is never absorbed: no downshift, no park, loud exit."""

    snapshot = _snapshot(tmp_path, count=3)
    with pytest.raises(RetryableOperatorError):
        _driver(tmp_path, snapshot, checker=StormyChecker()).run()

    paths = _paths(tmp_path, snapshot)
    assert _valve_rows(tmp_path, snapshot) == []
    events = scan_jsonl(paths.operational_ledger)
    assert not [
        event
        for event in events
        if event.get("details", {}).get("disposition")
        in {"checker-batch-downshift", "checker-lane-parked"}
    ]
    models = scan_jsonl(paths.ledgers.models)
    assert all(record["status"]["code"] != "failed:runner" for record in models)


def test_parked_models_requeue_and_gate_on_the_next_run(tmp_path: Path) -> None:
    """A parked model is not terminal: the next run re-gates it to completion."""

    snapshot = _snapshot(tmp_path, count=3)
    first = _driver(tmp_path, snapshot, checker=AlwaysPartialChecker()).run()
    assert first.status == "complete-with-deferrals"
    assert scan_jsonl(_paths(tmp_path, snapshot).ledgers.models) == []

    second = _driver(tmp_path, snapshot).run()
    assert second.status == "complete"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert set(models) == {item.stable_id for item in snapshot.items}
    assert {record["status"]["code"] for record in models.values()} == {"runs"}


def test_floor_one_gate_routing_failure_gets_one_reask_before_parking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checker-authored routing violation earns ONE fresh re-ask, then parks.

    The operator wrapper's bounded re-asks cover only its own dispatch-side
    validation; a ``GateRoutingError`` raised by the driver's routing validators
    has spent none. The checker is stochastic -- one flubbed identity echo among
    clean siblings -- so a single-item routing refusal is re-asked exactly once
    (the batch-7 precedent) and gates when the fresh response routes cleanly.
    """

    from menagerie.crawler import driver_admission as driver_admission_module
    from menagerie.crawler.gates import GateRoutingError

    snapshot = _snapshot(tmp_path, count=1)
    real = driver_admission_module.emit_gate_records
    refusals = {"count": 0}

    def refuse_once(gate: Mapping[str, Any]) -> Any:
        """Refuse the first routing exactly like a checker-authored flub."""

        if refusals["count"] == 0:
            refusals["count"] += 1
            raise GateRoutingError("terminal disposition source IDs do not match")
        return real(gate)

    monkeypatch.setattr(driver_admission_module, "emit_gate_records", refuse_once)
    result = _driver(tmp_path, snapshot).run()

    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models)
    assert [record["status"]["code"] for record in models] == ["runs"]
    assert _valve_rows(tmp_path, snapshot) == []
    events = scan_jsonl(paths.operational_ledger)
    reasks = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "checker-gate-routing-reask"
    ]
    assert len(reasks) == 1
    assert reasks[0]["details"]["stable_id"] == snapshot.items[0].stable_id


def test_floor_one_gate_routing_failure_parks_after_its_single_reask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A routing violation that recurs on the re-ask parks; the re-ask is bounded."""

    from menagerie.crawler import driver_admission as driver_admission_module
    from menagerie.crawler.gates import GateRoutingError

    snapshot = _snapshot(tmp_path, count=1)

    def refuse(gate: Mapping[str, Any]) -> Any:
        """Refuse every routing, so the bounded re-ask is provably spent."""

        del gate
        raise GateRoutingError("terminal disposition source IDs do not match")

    monkeypatch.setattr(driver_admission_module, "emit_gate_records", refuse)
    result = _driver(tmp_path, snapshot).run()

    assert result.status == "complete-with-deferrals"
    rows = _valve_rows(tmp_path, snapshot)
    assert [row["stable_id"] for row in rows] == [snapshot.items[0].stable_id]
    assert rows[0]["checker_lane"]["failure_class"] == "invalid-gate-result"
    # The trail shows the floor was asked twice: the original and its one re-ask.
    assert rows[0]["checker_lane"]["attempted_batch_sizes"] == [1, 1]


def test_checker_lane_valve_row_accounts_in_the_partition() -> None:
    """A valid checker-lane row claims its model; a terminal supersedes a stale one."""

    row = build_checker_lane_deferral_row(
        stable_id="m4334",
        work_id="work-1",
        name="MobileOne-S3",
        campaign_id="c1-mech",
        run_id="run-10",
        machine_id="machine-test",
        created_at="2026-08-06T07:21:11Z",
        gate_kind="metadata_batch",
        failure_class="permanent-contract-rejection",
        failure_evidence=RUNG10_EXIT64_MESSAGE,
        attempted_batch_sizes=(20, 10, 5, 3, 2, 1),
    )
    assert machine_deferral_rejection("m4334", row) is None
    report = partition_report(["m4334"], [], machine_deferrals={"m4334": row})
    assert report.valid
    assert report.machine_deferred == {CHECKER_LANE_DEFERRAL_DISPOSITION: frozenset({"m4334"})}

    terminal = {"stable_id": "m4334", "status": {"code": "runs", "kind": "runs"}}
    superseded = partition_report(["m4334"], [terminal], machine_deferrals={"m4334": row})
    assert superseded.superseded_deferral_ids == frozenset({"m4334"})

    tampered = dict(row)
    tampered["recheck_hint"] = "forged"
    assert machine_deferral_rejection("m4334", tampered) is not None


def test_status_view_loads_checker_lane_deferrals(tmp_path: Path) -> None:
    """The CLI partition view accounts parked models from the durable valve."""

    records_root = tmp_path / "records"
    row = build_checker_lane_deferral_row(
        stable_id="m4334",
        work_id="work-1",
        name="MobileOne-S3",
        campaign_id="c1-mech",
        run_id="run-10",
        machine_id="machine-test",
        created_at="2026-08-06T07:21:11Z",
        gate_kind="metadata_batch",
        failure_class="permanent-contract-rejection",
        failure_evidence=RUNG10_EXIT64_MESSAGE,
        attempted_batch_sizes=(5, 3, 2, 1),
    )
    append_checker_lane_deferral_row(checker_lane_deferral_path(records_root), row)

    report = partition_report(
        ["m4334"],
        [],
        machine_deferrals=_status_machine_deferrals(records_root),
    )

    assert report.valid
    assert report.missing_ids == frozenset()


def test_checker_lane_row_vocabulary_is_closed(tmp_path: Path) -> None:
    """Unknown failure classes and ladders that never reach the floor refuse."""

    kwargs: dict[str, Any] = {
        "stable_id": "m1",
        "work_id": "w1",
        "name": "Example",
        "campaign_id": "c1-mech",
        "run_id": "run-test",
        "machine_id": "machine-test",
        "created_at": "2026-08-06T00:00:00Z",
        "gate_kind": "metadata_batch",
        "failure_class": "permanent-contract-rejection",
        "failure_evidence": "refused",
        "attempted_batch_sizes": (2, 1),
    }
    row = build_checker_lane_deferral_row(**kwargs)
    assert validate_checker_lane_deferral_row(row) == row

    with pytest.raises(CheckerLaneDeferralError):
        build_checker_lane_deferral_row(**{**kwargs, "failure_class": "sad-checker"})
    with pytest.raises(CheckerLaneDeferralError):
        build_checker_lane_deferral_row(**{**kwargs, "attempted_batch_sizes": (4, 2)})

    ledger = tmp_path / "checker-lane-deferrals.jsonl"
    persisted = append_checker_lane_deferral_row(ledger, row)
    assert persisted == row
    # Idempotent per work generation: a resume re-park returns the FIRST row.
    later = build_checker_lane_deferral_row(
        **{**kwargs, "created_at": "2026-08-07T00:00:00Z", "run_id": "run-resume"}
    )
    assert append_checker_lane_deferral_row(ledger, later) == row
    assert len(scan_jsonl(ledger)) == 1
