"""W3 (calibration-withholding half): calibration mode WITHHOLDS awards.

The 200a audits proved the docstring claim "the calibration run cannot publish
awards" was FALSE at the code level: ``MENAGERIE_CHECKER_CALIBRATION_RUN=1``
bypassed :func:`require_checker_calibration` and nothing downstream withheld
anything -- a calibration leg could issue ``runs`` award heads and mark
records release-eligible exactly like a production leg (risk R1: the bypass
renders verdicts unattested and does NOT withhold awards). These tests pin
the semantics-made-TRUE fix (PLAN_v6 W3, O-B3 fork deletion):

* a live ``runs`` award append refuses typed while calibration mode is set --
  whether or not the award would be release-eligible;
* a record cannot be MARKED release-eligible while calibration mode is set;
* award RESTORATION (the second runs-award mint path) refuses too;
* semantic replay of already-issued history stays exempt: the withhold
  governs ISSUANCE, never re-admission of issued records;
* non-award terminals still append (a calibration leg records its work);
* every leg attests calibration mode plus the complete verdict-steering
  env-read inventory in its leg-config row (TE-3);
* the PD-9 bypass census arm discloses receiptless-era verdicts and a
  retroactively minted receipt can never bless them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checker_calibration import (
    CALIBRATION_BYPASS_CENSUS_ARM,
    CHECKER_CALIBRATION_RECEIPT_ENV,
    CHECKER_CALIBRATION_RUN_ENV,
    CHECKER_GOLDEN_CORPUS_ENV,
    SHADOW_REHEARSAL_RUN_ENV,
    CheckerCalibrationWithheldError,
    calibration_bypass_census,
    calibration_environment_snapshot,
)
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import project_dependency_current
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_attempt,
    make_authority_context,
    make_failed_attempt,
    make_gate,
    make_model,
)
from menagerie.crawler.tests.test_reducer import CanonicalReducer

pytestmark = pytest.mark.smoke


def _paths(tmp_path: Path) -> LedgerPaths:
    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _award_scenario(reducer: CanonicalReducer, *, accepted: bool = True) -> dict[str, Any]:
    """Stage the gate/attempt evidence and return a runs-award candidate."""

    reducer.append_gate(make_gate(["m_example"]))
    reducer.append_attempt(make_attempt())
    return make_model(accepted=accepted)


def test_runs_award_append_refuses_while_calibration_run_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live release-eligible award issuance refuses typed in calibration mode."""

    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        candidate = _award_scenario(reducer, accepted=True)
        with pytest.raises(CheckerCalibrationWithheldError, match="withh"):
            reducer.append_model(candidate)


def test_non_release_eligible_award_also_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The withhold keys on the award, not only on the release-eligible mark."""

    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        candidate = _award_scenario(reducer, accepted=False)
        with pytest.raises(CheckerCalibrationWithheldError):
            reducer.append_model(candidate)


def test_award_restoration_refuses_while_calibration_run_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The AW-3 restoration mint path is withheld before any ledger scan."""

    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        with pytest.raises(CheckerCalibrationWithheldError):
            reducer.append_award_restoration(
                stable_id="m_example",
                restored_award_revision="sha256:" + "a" * 64,
                superseding_revision="sha256:" + "b" * 64,
                restoration_event_id="restoration-1",
            )


def test_non_award_terminal_still_appends_in_calibration_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A calibration leg still records non-award terminals -- only awards withhold."""

    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    attempt = make_failed_attempt(stage="forward", reason_code="exception")
    model = make_model(status_code="failed:forward")
    model["status"]["reason_code"] = "exception"
    bind_terminal_attempts(model, [attempt])
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        head = reducer.append_model(model).record
    assert head["status"]["code"] == "failed:forward"
    assert head["completeness"]["release_eligible"] is False


def test_semantic_replay_readmits_issued_awards_under_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """History replay is exempt: an issued award re-admits during calibration."""

    paths = _paths(tmp_path)
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        head = reducer.append_model(_award_scenario(reducer, accepted=True)).record
    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    projection = project_dependency_current(
        paths, context=make_authority_context(["m_example"])
    )
    replayed = projection.current_records["m_example"]
    assert replayed["record_revision"] == head["record_revision"]
    assert replayed["status"]["kind"] == "runs"


def test_calibration_environment_snapshot_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The snapshot names exactly the four verdict-steering env reads."""

    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    monkeypatch.delenv(SHADOW_REHEARSAL_RUN_ENV, raising=False)
    monkeypatch.delenv(CHECKER_CALIBRATION_RECEIPT_ENV, raising=False)
    monkeypatch.delenv(CHECKER_GOLDEN_CORPUS_ENV, raising=False)
    assert calibration_environment_snapshot() == {
        "calibration_run_active": False,
        "shadow_rehearsal_run": None,
        "calibration_receipt_override": None,
        "golden_corpus_override": None,
    }
    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, "/tmp/receipt.json")
    monkeypatch.setenv(CHECKER_GOLDEN_CORPUS_ENV, "/tmp/corpus")
    assert calibration_environment_snapshot() == {
        "calibration_run_active": True,
        "shadow_rehearsal_run": None,
        "calibration_receipt_override": "/tmp/receipt.json",
        "golden_corpus_override": "/tmp/corpus",
    }
    # TOTAL under the ambiguous both-declared mode: attestation records the
    # raw declarations; the direct mode readers are where ambiguity refuses.
    monkeypatch.setenv(SHADOW_REHEARSAL_RUN_ENV, "r6b-gen-1")
    assert calibration_environment_snapshot() == {
        "calibration_run_active": True,
        "shadow_rehearsal_run": "r6b-gen-1",
        "calibration_receipt_override": "/tmp/receipt.json",
        "golden_corpus_override": "/tmp/corpus",
    }


def test_leg_config_attests_calibration_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TE-3: the B-09 leg-config row carries the calibration env snapshot."""

    from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
    from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
    from menagerie.crawler.tests.test_slice_f_driver import (
        _driver,
        _paths as _driver_paths,
        _snapshot,
    )

    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    monkeypatch.delenv(SHADOW_REHEARSAL_RUN_ENV, raising=False)
    monkeypatch.delenv(CHECKER_CALIBRATION_RECEIPT_ENV, raising=False)
    monkeypatch.setenv(CHECKER_GOLDEN_CORPUS_ENV, "/tmp/corpus")
    snapshot = _snapshot(tmp_path)
    paths = _driver_paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    details = driver._leg_config_details()  # noqa: SLF001 -- the unit under test
    assert details["checker_calibration"] == {
        "calibration_run_active": True,
        "shadow_rehearsal_run": None,
        "calibration_receipt_override": None,
        "golden_corpus_override": "/tmp/corpus",
    }
    # The attestation is digest-covered and schema-legal end to end: append
    # through the REAL ledger writer, which validates the B-09 contract.
    with JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        driver._record_leg_config(operational)  # noqa: SLF001
    events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("event_kind") == "leg-config"
    ]
    assert events[-1]["details"]["checker_calibration"]["calibration_run_active"] is True


def _gate_row(gate_id: str, model: str, finished_at: str) -> dict[str, Any]:
    return {"gate_id": gate_id, "checker": {"model": model, "finished_at": finished_at}}


def test_bypass_census_discloses_receiptless_history_fail_closed() -> None:
    """No valid receipt => every verdict is bypass-adjudicated, none attested."""

    rows = [
        _gate_row("gate-1", "gpt-5.6-sol", "2026-08-20T00:00:00Z"),
        _gate_row("gate-2", "gpt-5.6-terra", "2026-08-07T00:00:00Z"),
        {"gate_id": "gate-3"},  # no checker block at all: fail-closed bypass
    ]
    census = calibration_bypass_census(rows, receipt=None)
    assert census["census_arm"] == CALIBRATION_BYPASS_CENSUS_ARM
    assert census["gates_adjudicated_under_calibration_bypass"] == 3
    assert census["receipt_attested"] == 0
    assert census["bypass_gate_ids"] == ["gate-1", "gate-2", "gate-3"]


def test_retroactive_receipt_never_blesses_earlier_verdicts() -> None:
    """Era rule: only rows finishing at/after the receipt capture attest."""

    receipt = {"checker_model": "gpt-5.6-sol", "captured_at": "2026-08-25T00:00:00Z"}
    rows = [
        _gate_row("gate-early", "gpt-5.6-sol", "2026-08-20T00:00:00Z"),
        _gate_row("gate-late", "gpt-5.6-sol", "2026-08-26T00:00:00Z"),
        _gate_row("gate-wrong-model", "gpt-5.6-terra", "2026-08-26T00:00:00Z"),
    ]
    census = calibration_bypass_census(rows, receipt=receipt)
    assert census["gates_adjudicated_under_calibration_bypass"] == 2
    assert census["receipt_attested"] == 1
    assert census["bypass_gate_ids"] == ["gate-early", "gate-wrong-model"]
