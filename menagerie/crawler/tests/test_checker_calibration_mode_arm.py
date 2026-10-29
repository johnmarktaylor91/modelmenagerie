"""W2's receipt-XOR preflight arm: exactly one sanctioned calibration mode.

A verdict-rendering launch holds a valid receipt (with golden corroboration)
covering every pinned lane XOR the declared calibration capture mode -- both
is ambiguous, neither is the uncalibrated launch BL-10 prevents, and a
verdict-free leg (closing licensing leg, final-sweep step 3) passes vacuously
with the declaration disclosed (F-O19). The arm lands in the launch preflight
through the W4 registry (``checker-calibration-mode`` in
``EXPECTED_PREFLIGHT_ARMS``), and W3's shadow-rehearsal third mode is
RECEIPT-BEARING: it must extend the ``calibration_run_active`` disjunct of
this arm, never add a receiptless one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checker_calibration import (
    CHECKER_ACTIVATION_ENV,
    CHECKER_CALIBRATION_MODE_ARM,
    CHECKER_CALIBRATION_MODE_UNPROVEN,
    CHECKER_CALIBRATION_RECEIPT_ENV,
    CHECKER_CALIBRATION_RUN_ENV,
    CHECKER_GOLDEN_CORPUS_ENV,
    evaluate_checker_calibration_mode,
)
from menagerie.crawler.constants import CheckerSeatRole, GateKind
from menagerie.crawler.operator_checker import required_checker_model
from menagerie.crawler.tests.conftest import _calibration_receipt
from menagerie.crawler.tools.launch_preflight import EXPECTED_PREFLIGHT_ARMS

pytestmark = pytest.mark.smoke

#: The RECEIPT door under test pins the sol seat (the K=2 confirmation seat
#: since the 2026-09-01 swap): receipt + goldens are sol's attestation. The
#: Fable primary is activated by the committed activation matrix instead
#: (test_checker_seat_swap.py), so these tests point the activation record at
#: nothing to exercise the receipt side alone.
PINNED = {
    kind.value: required_checker_model(kind.value, CheckerSeatRole.CONFIRMATION)
    for kind in GateKind
}


@pytest.fixture(autouse=True)
def _no_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHECKER_ACTIVATION_ENV, "/nonexistent/checker-activation.json")


def _calibrated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(_calibration_receipt()), encoding="utf-8")
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(receipt_path))
    corpus = tmp_path / "goldens"
    corpus.mkdir(exist_ok=True)
    for name in ("metadata_x", "fidelity_x", "terminal_x"):
        (corpus / f"{name}.json").write_text(
            json.dumps({"checker_model": "gpt-5.6-sol"}), encoding="utf-8"
        )
    monkeypatch.setenv(CHECKER_GOLDEN_CORPUS_ENV, str(corpus))


def _evaluate(**kwargs: Any) -> dict[str, Any]:
    return evaluate_checker_calibration_mode(pinned_models=PINNED, **kwargs)


def test_the_arm_is_in_the_closed_preflight_census() -> None:
    assert CHECKER_CALIBRATION_MODE_ARM in EXPECTED_PREFLIGHT_ARMS


def test_neither_receipt_nor_mode_refuses_a_verdict_rendering_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, "/nonexistent/receipt.json")
    result = _evaluate(renders_gate_verdicts=True)
    assert result["ok"] is False and result["vacuous"] is False
    assert len(result["refusals"]) == 1
    assert result["refusals"][0].startswith(CHECKER_CALIBRATION_MODE_UNPROVEN)
    # Every pinned lane is disclosed individually.
    assert {row["gate_kind"] for row in result["lanes"]} == set(PINNED)
    assert all(row["receipt"] == "refused" for row in result["lanes"])


def test_valid_receipt_alone_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    _calibrated_env(tmp_path, monkeypatch)
    result = _evaluate(renders_gate_verdicts=True)
    assert result["ok"] is True and result["refusals"] == []
    assert all(row["receipt"] == "attested" for row in result["lanes"])


def test_declared_calibration_mode_alone_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, "/nonexistent/receipt.json")
    result = _evaluate(renders_gate_verdicts=True)
    assert result["ok"] is True and result["calibration_run_active"] is True


def test_both_receipt_and_mode_is_an_ambiguous_mode_and_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The XOR: a calibration run under a live covering receipt is ambiguous
    -- the receipt-side evaluation deliberately IGNORES the bypass flag, so
    the flag cannot make the receipt side unsatisfiable and dodge the XOR."""

    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    _calibrated_env(tmp_path, monkeypatch)
    result = _evaluate(renders_gate_verdicts=True)
    assert result["ok"] is False
    assert "ambiguous" in result["refusals"][0]


def test_verdict_free_leg_passes_vacuously_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, "/nonexistent/receipt.json")
    result = _evaluate(renders_gate_verdicts=False)
    assert result["ok"] is True
    assert result["vacuous"] is True, "F-O19: the vacuous pass must be DISCLOSED"


def test_a_lane_whose_goldens_lag_the_receipt_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receipt + goldens travel together: a receipt without a re-recorded
    golden for one lane leaves the XOR's receipt side unsatisfied."""

    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    _calibrated_env(tmp_path, monkeypatch)
    (tmp_path / "goldens" / "terminal_x.json").unlink()
    result = _evaluate(renders_gate_verdicts=True)
    assert result["ok"] is False
    lanes = {row["gate_kind"]: row["receipt"] for row in result["lanes"]}
    assert lanes["terminal_disposition"] == "refused"
    assert lanes["metadata_batch"] == "attested"
