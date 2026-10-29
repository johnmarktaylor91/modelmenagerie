"""Driver-side pre-admission capacity gate tests.

Exercises ``CrawlerDriver._admit_within_host_capacity`` directly against the
real Mixtral recipe: an oversized model must be withheld from the environment
lane, recorded in the deferral ledger, and announced on the operational ledger,
while every other model passes through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler.constants import (
    EnvironmentPhase,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.driver_contracts import DriverIntegrationError, WorkItem
from menagerie.crawler.driver_receipts import _worker_observed_rss_deferral_row
from menagerie.crawler.host_capacity import (
    CAPACITY_DEFERRAL_DISPOSITION,
    HOST_MEMORY_ENV_VAR,
    capacity_deferral_path,
    load_capacity_deferral_rows,
    validate_capacity_deferral_row,
)
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.tests.test_host_capacity import (
    MIXTRAL_RECIPE,
    NAMED_SYMBOL_RECIPE,
)


@dataclass
class _StubArtifact:
    """Minimal stand-in exposing only the proposal the capacity gate reads."""

    recipe: Any

    @property
    def proposal(self) -> dict[str, Any]:
        """Return a proposal carrying exactly the authored implementation facts."""

        return {"proposed_facts": {"implementation": self.recipe}}


class _StubLedger:
    """Operational ledger stub capturing appended events."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Record one operational event."""

        self.records.append(event)
        return event


def _work_item(stable_id: str, name: str) -> WorkItem:
    """Return a scheduled work item for one synthetic roster row."""

    return WorkItem(
        intake=IntakeItem(
            stable_id=stable_id,
            name=name,
            zoo="discovered-pytorch",
            variant="",
            discovery_source="master_catalog",
            legacy_row_sha256=f"sha256:{'0' * 64}",
            preserved_legacy_flags=(),
        ),
        route=IntentRoute(
            stable_id=stable_id,
            intent="torch-core",
            phase=EnvironmentPhase.PYTORCH,
        ),
    )


def _driver(records_root: Path) -> Any:
    """Return a duck-typed driver exposing exactly what the gate touches."""

    return SimpleNamespace(
        dependencies=SimpleNamespace(clock=lambda: "2026-08-04T12:00:00.000000Z"),
        paths=SimpleNamespace(
            ledgers=SimpleNamespace(models=records_root / "models" / "current-shard.jsonl")
        ),
        config=SimpleNamespace(
            campaign_id="c1-mech",
            run_id="crawler-run",
            machine_id="worker-b",
        ),
    )


def _rss_cap_attempt(
    attempt_id: str,
    mode: str,
    peak_rss_bytes: int,
    *,
    work_id: str = "",
) -> dict[str, Any]:
    """Return one frozen-rung-9b-shaped RSS-cap worker attempt."""

    return {
        "attempt_id": attempt_id,
        "work_id": work_id,
        "result": "failed",
        "error": {"stage": "resource", "reason_code": "rss-cap"},
        "host": {
            "machine_id": "worker-a",
            "architecture": "arm64",
            "ram_bytes": 17_179_869_184,
        },
        "invocation": {
            "mode": mode,
            "rss_limit_bytes": 12_884_901_888,
        },
        "supervisor_observation": {
            "signal": 9,
            "peak_rss_bytes": peak_rss_bytes,
        },
    }


@pytest.fixture(autouse=True)
def _small_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the host to 16 GiB so the gate's decision is deterministic."""

    monkeypatch.setenv(HOST_MEMORY_ENV_VAR, str(16 * 2**30))


def test_oversized_model_is_withheld_and_recorded(tmp_path: Path) -> None:
    """Mixtral never reaches the environment lane, and the refusal is durable."""

    records_root = tmp_path / "records"
    driver = _driver(records_root)
    mixtral = _work_item("m5915", "Mixtral 8x7B")
    dla = _work_item("m4334", "DLA-60")
    artifacts: Any = {
        "m5915": _StubArtifact({"library_recipe": MIXTRAL_RECIPE}),
        "m4334": _StubArtifact({"library_recipe": NAMED_SYMBOL_RECIPE}),
    }
    operational: Any = _StubLedger()

    admitted = CrawlerDriver._admit_within_host_capacity(
        driver, (mixtral, dla), artifacts, operational
    )

    assert [item.stable_id for item in admitted] == ["m4334"]
    rows = load_capacity_deferral_rows([capacity_deferral_path(records_root)])
    assert len(rows) == 1
    assert rows[0]["stable_id"] == "m5915"
    assert rows[0]["disposition"] == CAPACITY_DEFERRAL_DISPOSITION
    assert rows[0]["capacity"]["estimate"]["parameter_count_lower_bound"] > 8.59e9
    assert rows[0]["capacity"]["threshold"]["host_physical_memory_bytes"] == 16 * 2**30

    assert len(operational.records) == 1
    event = operational.records[0]
    assert event["event_kind"] == OperationalEventKind.CAMPAIGN_HEALTH.value
    assert event["status"] == OperationalEventStatus.HEALTHY.value
    assert event["details"]["disposition"] == "host-capacity-deferred"
    assert event["details"]["deferred"][0]["stable_id"] == "m5915"
    assert event["event_id"].startswith("campaign-health-")


def test_a_wave_with_nothing_oversized_is_untouched(tmp_path: Path) -> None:
    """No ledger and no event when every model fits."""

    records_root = tmp_path / "records"
    driver = _driver(records_root)
    dla = _work_item("m4334", "DLA-60")
    artifacts: Any = {"m4334": _StubArtifact({"library_recipe": NAMED_SYMBOL_RECIPE})}
    operational: Any = _StubLedger()

    admitted = CrawlerDriver._admit_within_host_capacity(driver, (dla,), artifacts, operational)

    assert admitted == (dla,)
    assert not capacity_deferral_path(records_root).exists()
    assert operational.records == []


def test_a_bigger_host_admits_the_same_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog becomes runnable on larger hardware with no code change."""

    monkeypatch.setenv(HOST_MEMORY_ENV_VAR, str(512 * 2**30))
    records_root = tmp_path / "records"
    driver = _driver(records_root)
    mixtral = _work_item("m5915", "Mixtral 8x7B")
    artifacts: Any = {"m5915": _StubArtifact({"library_recipe": MIXTRAL_RECIPE})}
    operational: Any = _StubLedger()

    admitted = CrawlerDriver._admit_within_host_capacity(driver, (mixtral,), artifacts, operational)

    assert admitted == (mixtral,)
    assert not capacity_deferral_path(records_root).exists()
    assert operational.records == []


class _TerminalArtifact:
    """Terminal author result: it has no executable proposal to size."""

    @property
    def proposal(self) -> dict[str, Any]:
        """Raise exactly as a terminal author result does."""

        raise DriverIntegrationError("terminal author result has no executable proposal")


def test_a_terminal_author_result_passes_through(tmp_path: Path) -> None:
    """A skip/blocked/defer arm has nothing to size and must not crash the gate.

    The gate now runs before the checker, where terminal author results are
    still in the wave, so this arm is reached in normal operation.
    """

    records_root = tmp_path / "records"
    driver = _driver(records_root)
    blocked = _work_item("m3671", "BindsNET DiehlAndCook2015")
    artifacts: Any = {"m3671": _TerminalArtifact()}
    operational: Any = _StubLedger()

    admitted = CrawlerDriver._admit_within_host_capacity(driver, (blocked,), artifacts, operational)

    assert admitted == (blocked,)
    assert not capacity_deferral_path(records_root).exists()
    assert operational.records == []


def test_an_empty_wave_never_reads_the_host(tmp_path: Path) -> None:
    """The gate is free when there is no work to admit."""

    driver = _driver(tmp_path / "records")
    operational: Any = _StubLedger()
    assert CrawlerDriver._admit_within_host_capacity(driver, (), {}, operational) == ()
    assert operational.records == []


@pytest.mark.smoke
def test_m7637_observed_rss_becomes_an_honest_capacity_deferral() -> None:
    """Replay m7637's two exact RSS peaks after a non-derivable estimate."""

    item = _work_item("m7637", "NemotronH")
    recipe = {
        "library_recipe": {
            "distribution": "transformers",
            "version": "5.14.1",
            "module": "transformers",
            "symbol": "NemotronHForCausalLM",
            "kwargs": {
                "config": {
                    "__construct__": {
                        "module": "transformers",
                        "symbol": "NemotronHConfig",
                        "kwargs": {
                            "hidden_size": 4096,
                            "vocab_size": 131072,
                            "hybrid_override_pattern": (
                                "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
                            ),
                        },
                    }
                }
            },
        }
    }
    attempts = (
        _rss_cap_attempt(
            "sha256:06aabc82", "train", 12_893_874_304, work_id=item.active_work_id
        ),
        _rss_cap_attempt(
            "sha256:2e8adb0f", "eval", 12_908_324_872, work_id=item.active_work_id
        ),
    )

    row = _worker_observed_rss_deferral_row(
        item,
        _StubArtifact(recipe),
        attempts,
        campaign_id="c1-mech",
        run_id="rung9b-replay",
        machine_id="worker-a",
        created_at="2026-08-05T23:40:50Z",
    )

    assert row is not None
    validated = validate_capacity_deferral_row(row)
    assert validated["disposition"] == "deferred:needs-more-memory"
    estimate = validated["capacity"]["estimate"]
    assert estimate["origin"] == "worker-observed-rss"
    assert estimate["max_peak_rss_bytes"] == 12_908_324_872
    assert estimate["observed_peak_basis"] == "lower-bound"
    assert [entry["peak_rss_bytes"] for entry in estimate["attempts"]] == [
        12_893_874_304,
        12_908_324_872,
    ]
    assert validated["capacity"]["static_assessment"]["verdict"] == "not-derivable"
    assert "12908324872" in validated["recheck_hint"]
    assert "lower bound" in validated["recheck_hint"]


@pytest.mark.smoke
def test_m12627_class_derivable_estimate_rss_overrun_now_defers() -> None:
    """DX-6/RE-6: the deferral keys on the OBSERVATION, not the static verdict.

    m12627/m12704/m12733 were admitted on a derivable declared-config estimate
    and then RSS-killed by live supervision; the old NOT_DERIVABLE gate turned
    each into a permanent ``failed:resource``. The parent's own observation is
    stronger evidence than the estimate that admitted them: the row now builds,
    with the static verdict RECORDED as the refusal predicate instead of
    consulted as a gate.
    """

    item = _work_item("m-estimator-surprise", "Estimator surprise")
    row = _worker_observed_rss_deferral_row(
        item,
        _StubArtifact({"library_recipe": MIXTRAL_RECIPE}),
        (
            _rss_cap_attempt(
                "attempt-1", "eval", 12_908_324_872, work_id=item.active_work_id
            ),
        ),
        campaign_id="c1-mech",
        run_id="rung9b-replay",
        machine_id="worker-a",
        created_at="2026-08-05T23:40:50Z",
    )

    assert row is not None
    validated = validate_capacity_deferral_row(row)
    assert validated["disposition"] == "deferred:needs-more-memory"
    static = validated["capacity"]["static_assessment"]
    assert static["verdict"] != "not-derivable", "the refusal predicate is recorded"
    assert validated["capacity"]["estimate"]["observed_peak_basis"] == "lower-bound"


@pytest.mark.smoke
def test_rss_deferral_scopes_mixed_history_to_the_active_generation() -> None:
    """A stale non-RSS failure from an earlier generation cannot block the valve."""

    item = _work_item("m-mixed-history", "Mixed history")
    stale_import_failure = {
        "attempt_id": "attempt-old",
        "work_id": "work-earlier-generation",
        "result": "failed",
        "error": {"stage": "import", "reason_code": "import-exception"},
    }
    active_rss = _rss_cap_attempt(
        "attempt-new", "eval", 12_908_324_872, work_id=item.active_work_id
    )
    row = _worker_observed_rss_deferral_row(
        item,
        _StubArtifact({"library_recipe": MIXTRAL_RECIPE}),
        (stale_import_failure, active_rss),
        campaign_id="c1-mech",
        run_id="rung9b-replay",
        machine_id="worker-a",
        created_at="2026-08-05T23:40:50Z",
    )
    assert row is not None

    # A non-RSS failure IN the active generation still blocks it: the decisive
    # verdict belongs to that failure, not to a capacity valve.
    active_import_failure = dict(stale_import_failure)
    active_import_failure["work_id"] = item.active_work_id
    blocked = _worker_observed_rss_deferral_row(
        item,
        _StubArtifact({"library_recipe": MIXTRAL_RECIPE}),
        (active_import_failure, active_rss),
        campaign_id="c1-mech",
        run_id="rung9b-replay",
        machine_id="worker-a",
        created_at="2026-08-05T23:40:50Z",
    )
    assert blocked is None


@pytest.mark.smoke
def test_rss_deferral_still_requires_complete_observation_evidence() -> None:
    """A kill without a recorded peak cannot build the row (machinery park path)."""

    item = _work_item("m-no-peak", "No peak recorded")
    attempt = _rss_cap_attempt("attempt-1", "eval", 1, work_id=item.active_work_id)
    attempt["supervisor_observation"]["peak_rss_bytes"] = 0
    row = _worker_observed_rss_deferral_row(
        item,
        _StubArtifact({"library_recipe": MIXTRAL_RECIPE}),
        (attempt,),
        campaign_id="c1-mech",
        run_id="rung9b-replay",
        machine_id="worker-a",
        created_at="2026-08-05T23:40:50Z",
    )
    assert row is None
