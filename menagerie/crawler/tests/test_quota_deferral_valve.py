"""W1: QUOTA storms mint transient deferral valve rows, never exhaustion terminals.

The valve family (build/validate/mint, PD-3 bound, recidivist park), the
two-direction census invariant (``effort-exhausted:author-sessions`` RESERVED for
used == max; the valve RESERVED for settlement-witnessed storms), the storm-split
arm's behavior, and the driver wiring pins (the PD-13 event-emission invariant at
every exhaustion arm; the W26 storm-halt composition). The production-shaped
storm-pack e2e rides R4's V4 rehearsal arm, exactly like the W26 fuse.
"""

from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler import driver_admission as da
from menagerie.crawler.driver_admission import AdmissionEnvironmentMixin
from menagerie.crawler.paid_session_accounting import MACHINE_FLOOR_SETTLEMENT_SCHEMAS
from menagerie.crawler.quota_deferral import (
    QUOTA_DEFERRAL_DISPOSITION,
    QUOTA_DEFERRAL_MAX_GRANTS,
    STORM_RECIDIVIST_DISPOSITION,
    QuotaDeferralError,
    load_quota_deferral_rows,
    mint_quota_deferral,
    quota_exhaustion_census,
    storm_recidivist_hold,
    validate_quota_deferral_row,
)

#: Load-bearing money-vocabulary tests: sub-second, in the mandatory smoke gate.
pytestmark = pytest.mark.smoke

STABLE_ID = "m_storm_victim"
WORK_ID = "work-gen-1"
NOW = "2026-08-24T12:00:00+00:00"
SCHEMA = sorted(MACHINE_FLOOR_SETTLEMENT_SCHEMAS)[0]


def _mint(path: Path, *, run_id: str = "run-1", stable_id: str = STABLE_ID) -> dict[str, Any]:
    return mint_quota_deferral(
        path,
        stable_id=stable_id,
        work_id=WORK_ID,
        campaign_id="campaign-test",
        run_id=run_id,
        machine_id="machine-1",
        created_at=NOW,
        evidence={"witness": "author-sessions-exhaustion-classifier"},
    )


# ---------------------------------------------------------------------------
# The valve family: mint, idempotency, PD-3 bound, recidivist park


def test_mint_appends_one_valve_row_and_resumes_idempotently(tmp_path: Path) -> None:
    ledger = tmp_path / "quota-deferrals.jsonl"
    first = _mint(ledger)
    assert first["disposition"] == QUOTA_DEFERRAL_DISPOSITION
    assert first["deferral_ordinal"] == 1
    replay = _mint(ledger)  # same run re-derivation
    assert replay == first
    assert len(load_quota_deferral_rows(ledger)) == 1


def test_pd3_bound_ladder_ends_in_one_recidivist_park(tmp_path: Path) -> None:
    ledger = tmp_path / "quota-deferrals.jsonl"
    rows = [_mint(ledger, run_id=f"run-{n}") for n in range(QUOTA_DEFERRAL_MAX_GRANTS + 2)]
    valves = [row for row in rows if row["disposition"] == QUOTA_DEFERRAL_DISPOSITION]
    parks = [row for row in rows if row["disposition"] == STORM_RECIDIVIST_DISPOSITION]
    assert len(valves) == QUOTA_DEFERRAL_MAX_GRANTS
    assert [row["deferral_ordinal"] for row in valves] == [1, 2, 3]
    # The 4th storm mints the ONE park; the 5th returns it unchanged.
    assert parks and all(p == parks[0] for p in parks)
    assert parks[0]["deferral_ordinal"] == QUOTA_DEFERRAL_MAX_GRANTS + 1
    persisted = load_quota_deferral_rows(ledger)
    assert len(persisted) == QUOTA_DEFERRAL_MAX_GRANTS + 1
    assert storm_recidivist_hold(persisted, STABLE_ID, WORK_ID)
    assert not storm_recidivist_hold(persisted, "m_other", WORK_ID)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.__setitem__("disposition", "machine-deferred:weather"),
        lambda row: row.__setitem__("deferral_ordinal", 0),
        lambda row: row.__setitem__("deferral_ordinal", QUOTA_DEFERRAL_MAX_GRANTS + 1),
        lambda row: row.__setitem__("campaign_id", "None"),
        lambda row: row["quota_deferral"].__setitem__("dimension", "author-sessions"),
        lambda row: row.__setitem__("recheck_hint", row["recheck_hint"] + " tampered"),
    ],
)
def test_valve_row_validator_refuses_every_broken_shape(tmp_path: Path, mutate: Any) -> None:
    row = dict(_mint(tmp_path / "quota-deferrals.jsonl"))
    row["quota_deferral"] = dict(row["quota_deferral"])
    mutate(row)
    with pytest.raises(QuotaDeferralError):
        validate_quota_deferral_row(row)


# ---------------------------------------------------------------------------
# The two-direction census invariant


def _head(reason: str = "effort-exhausted:author-sessions") -> dict[str, Any]:
    return {
        "stable_id": STABLE_ID,
        "work_id": WORK_ID,
        "status": {"kind": "failed", "reason_code": reason, "work_id": WORK_ID},
    }


def _spend_events(*, storm: bool) -> list[dict[str, Any]]:
    """One reserved+settled author session; storm settlements refund the attempt."""

    events: list[dict[str, Any]] = []
    for index in range(3):
        spend_id = f"spend-{index}"
        events.append(
            {
                "event": "session_reserved",
                "lane": "author",
                "spend_id": spend_id,
                "axis": {
                    "work_generations": [{"stable_id": STABLE_ID, "work_id": WORK_ID}]
                },
            }
        )
        events.append(
            {
                "event": "session_settled",
                "spend_id": spend_id,
                "schema_version": SCHEMA,
                "provider": {"accepted": True},
                "local_outcome": "failed-harness-api-error" if storm else "completed",
            }
        )
    return events


def test_census_direction_one_flags_the_understated_head(tmp_path: Path) -> None:
    """The m697 class: an exhaustion head whose replay-derived used < allowance."""

    census = quota_exhaustion_census([_head()], [], [], _spend_events(storm=True))
    assert census.exhausted_heads
    assert census.understated_heads == census.exhausted_heads
    assert not census.clean


def test_census_direction_two_flags_the_unsupported_valve(tmp_path: Path) -> None:
    """A valve row whose generation's settlements do not classify provider-storm."""

    valve = _mint(tmp_path / "quota-deferrals.jsonl")
    census = quota_exhaustion_census([], [valve], [], _spend_events(storm=False))
    assert census.storm_valves == ((STABLE_ID, WORK_ID),)
    assert census.unsupported_valves == ((STABLE_ID, WORK_ID),)
    assert not census.clean


def test_census_passes_on_a_storm_supported_valve(tmp_path: Path) -> None:
    valve = _mint(tmp_path / "quota-deferrals.jsonl")
    census = quota_exhaustion_census([], [valve], [], _spend_events(storm=True))
    assert census.storm_valves == ((STABLE_ID, WORK_ID),)
    assert census.clean


# ---------------------------------------------------------------------------
# The storm-split arm (functional, over the REAL method)


class _RecordingReducer:
    def __init__(self) -> None:
        self.appended: list[Any] = []

    def append_attempt(self, attempt: Any) -> Any:
        self.appended.append(attempt)
        return SimpleNamespace(record=attempt)


class _RecordingLedger:
    def __init__(self) -> None:
        self.records: list[Any] = []

    def append(self, event: Any) -> None:
        self.records.append(event)


def _stub_driver(tmp_path: Path) -> Any:
    records_root = tmp_path / "records"
    models = records_root / "models" / "current-shard.jsonl"
    stub = SimpleNamespace(
        config=SimpleNamespace(
            run_id="run-1", machine_id="machine-1", campaign_id="campaign-test"
        ),
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=models), work_root=tmp_path / "work"
        ),
        dependencies=SimpleNamespace(clock=lambda: NOW),
        _terminalize_calls=[],
    )
    stub._repair_failure_stage_and_reason = (
        lambda exc: AdmissionEnvironmentMixin._repair_failure_stage_and_reason(exc)
    )
    stub._terminalize = lambda *args, **kwargs: stub._terminalize_calls.append(args)
    stub._quota_deferral_ledger_path = (
        lambda: AdmissionEnvironmentMixin._quota_deferral_ledger_path(stub)
    )
    stub._mint_quota_storm_deferral = (
        lambda item, evidence: AdmissionEnvironmentMixin._mint_quota_storm_deferral(
            stub, item, evidence
        )
    )
    return stub


def _fake_item() -> Any:
    return SimpleNamespace(stable_id=STABLE_ID, active_work_id=WORK_ID)


def _run_split(tmp_path: Path, monkeypatch: Any, *, dimension: str) -> tuple[Any, Any, Any]:
    stub = _stub_driver(tmp_path)
    reducer = _RecordingReducer()
    operational = _RecordingLedger()
    monkeypatch.setattr(
        da, "_driver_failure_attempt", lambda *a, **k: {"attempt": "row"}
    )
    monkeypatch.setattr(da, "_diagnostics_root_for_work_root", lambda root: root)
    exc = da.AuthorEffortCapExceeded(
        "allowance spent", stage="author", dimension=dimension
    )
    AdmissionEnvironmentMixin._terminalize_effort_exhaustion(
        stub, _fake_item(), None, exc, reducer, operational, {}
    )
    return stub, reducer, operational


def test_storm_classified_exhaustion_defers_and_never_terminalizes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    stub, reducer, operational = _run_split(tmp_path, monkeypatch, dimension="provider-storm")
    assert stub._terminalize_calls == []
    assert operational.records == []  # no model-lane-failed: no terminal minted
    assert reducer.appended  # the failing attempt still records the spend truth
    rows = load_quota_deferral_rows(
        AdmissionEnvironmentMixin._quota_deferral_ledger_path(stub)
    )
    assert [row["disposition"] for row in rows] == [QUOTA_DEFERRAL_DISPOSITION]
    # The valve row is stashed for the completion partition, like every valve.
    assert stub._machine_deferral_rows[STABLE_ID]["disposition"] == QUOTA_DEFERRAL_DISPOSITION


def test_plain_exhaustion_terminalizes_with_the_cause_event_first(
    tmp_path: Path, monkeypatch: Any
) -> None:
    stub, reducer, operational = _run_split(tmp_path, monkeypatch, dimension="author-sessions")
    assert len(stub._terminalize_calls) == 1
    assert len(operational.records) == 1
    event = operational.records[0]
    assert event["event_kind"] == "model-lane-failed"
    assert event["details"]["reason_code"] == "effort-exhausted:author-sessions"
    assert event["details"]["stable_id"] == STABLE_ID


# ---------------------------------------------------------------------------
# Wiring pins (the W26 idiom: the e2e storm pack rides R4's V4 arm)


def test_every_admission_exhaustion_arm_routes_through_the_shared_split() -> None:
    """No admission arm may terminalize exhaustion outside the evented split."""

    source = Path(da.__file__).read_text(encoding="utf-8")
    assert source.count("self._terminalize_effort_exhaustion(") >= 3


def test_receipts_exhaustion_arms_carry_event_and_storm_split() -> None:
    from menagerie.crawler import driver_receipts as dr

    source = Path(dr.__file__).read_text(encoding="utf-8")
    arms = source.count("except (AuthorEffortCapExceeded, AuthorEffortExhaustionClaim)")
    assert arms == 2
    assert source.count('getattr(exc, "dimension", None) == "provider-storm"') == 2
    assert source.count("_model_lane_failure_event(") >= 2


def test_quota_arm_runs_the_settlement_classifier() -> None:
    """The raw-session QUOTA trip derives its dimension, never hardcodes it."""

    source = textwrap.dedent(inspect.getsource(AdmissionEnvironmentMixin._admit_author_spend))
    quota_arm = source.find("except PaidSessionBudgetExceeded")
    assert quota_arm != -1
    assert "author_sessions_exhaustion_dimension(" in source[quota_arm:]
    assert 'dimension="author-sessions"' not in source[quota_arm:]


def test_storm_halt_mints_the_valve_row_for_the_stranded_member() -> None:
    """W1/W26 composition: the halt leaves the refused member deferral-accounted."""

    source = textwrap.dedent(inspect.getsource(AdmissionEnvironmentMixin._admit_author_spend))
    halt = source.find("CrossModelProviderStorm(")
    mint = source.find("_mint_quota_storm_deferral(")
    assert mint != -1 and halt != -1
    assert mint < halt  # the durable row precedes the typed re-raise
    assert "storm_recidivist_hold(" in source


def test_the_partition_vocabulary_accounts_both_valve_dispositions() -> None:
    from menagerie.crawler.status import MACHINE_DEFERRAL_DISPOSITIONS

    assert QUOTA_DEFERRAL_DISPOSITION in MACHINE_DEFERRAL_DISPOSITIONS
    assert STORM_RECIDIVIST_DISPOSITION in MACHINE_DEFERRAL_DISPOSITIONS
