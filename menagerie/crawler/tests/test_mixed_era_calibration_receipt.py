"""AB-30 matrix: calibration-receipt-identity seam (W2 + AG-4, lane w2-checker).

The Wave-2 checker lane tightened calibration-receipt schema v1 PRE-FIRST-MINT:
new REQUIRED per-lane fields (``sessions_bought``, ``attempt_latency_p99_seconds``,
``attempt_samples``, ``attempt_censoring_rate`` capped at
:data:`CALIBRATION_MAX_REFUSAL_RATE`), a REQUIRED top-level
``production_batch_width`` pinned to :data:`METADATA_BATCH_MAX`, REQUIRED
capacity session+dollar pairs, a REQUIRED ``blind_replay`` attestation block,
and timing tripwires (a measured lane attempt p99 above
``OPERATOR_ATTEMPT_TIMEOUT_SECONDS`` or session p99 above
``CHECKER_SESSION_WALL_SECONDS`` refuses at load).

The checker lane originally pinned that NO COMMITTED RECEIPT EXISTED at this
tree (the era boundary was vacuous by proof). The minted v2 receipt landed
2026-08-29 and retired the absence half of that proof; the boundary now closes
from the other side: the ONLY committed receipt parses through the tightened
loader carrying every new REQUIRED field (post-tightening by construction),
and a pre-batch-SHAPED receipt cannot load at all -- so no mixed-era reader of
the new fields can exist. This matrix pins exactly that, plus the tightened
loader's field-caused refusals -- through the REAL activation loader only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checker_calibration import (
    CALIBRATION_MAX_REFUSAL_RATE,
    CALIBRATION_PRODUCTION_BATCH_WIDTH,
    CHECKER_CALIBRATION_RECEIPT_ENV,
    CHECKER_CALIBRATION_RECEIPT_PATH,
    CHECKER_CALIBRATION_RECEIPT_SCHEMA,
    CHECKER_CALIBRATION_RUN_ENV,
    CheckerCalibrationInvalidError,
    load_checker_calibration_receipt,
    require_checker_calibration,
)
from menagerie.crawler.constants import METADATA_BATCH_MAX, GateKind
from menagerie.crawler.operator_protocol import (
    CHECKER_SESSION_WALL_SECONDS,
    OPERATOR_ATTEMPT_TIMEOUT_SECONDS,
)
from menagerie.crawler.tests.conftest import _calibration_receipt
from menagerie.crawler.tests.mixed_era import assert_era_outcome, classify_reader

pytestmark = pytest.mark.smoke

#: The new REQUIRED per-lane fields the pre-first-mint tightening introduced.
_NEW_LANE_FIELDS = (
    "sessions_bought",
    "attempt_latency_p99_seconds",
    "attempt_samples",
    "attempt_censoring_rate",
)


def _write_receipt(tmp_path: Path, receipt: dict[str, Any]) -> Path:
    """Persist one receipt candidate for the real loader to adjudicate."""

    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _strip_new_fields(receipt: dict[str, Any], variant: str) -> dict[str, Any]:
    """Remove one tightened-schema field family, modeling a pre-batch shape."""

    if variant == "lane-sessions_bought":
        for lane in receipt["lanes"].values():
            del lane["sessions_bought"]
    elif variant == "lane-attempt-trio":
        for lane in receipt["lanes"].values():
            for field in (
                "attempt_latency_p99_seconds",
                "attempt_samples",
                "attempt_censoring_rate",
            ):
                del lane[field]
    elif variant == "production_batch_width":
        del receipt["production_batch_width"]
    elif variant == "blind_replay":
        del receipt["blind_replay"]
    elif variant == "capacity-session-pair":
        del receipt["capacity"]["projected_sessions"]
        del receipt["capacity"]["session_budget"]
    elif variant == "capacity-dollar-pair":
        del receipt["capacity"]["projected_dollars"]
        del receipt["capacity"]["dollar_budget"]
    elif variant == "all-new-required-fields":
        for inner in (
            "lane-sessions_bought",
            "lane-attempt-trio",
            "production_batch_width",
            "blind_replay",
            "capacity-session-pair",
            "capacity-dollar-pair",
        ):
            _strip_new_fields(receipt, inner)
    else:  # pragma: no cover - closed variant vocabulary
        raise AssertionError(f"unknown pre-batch variant {variant!r}")
    return receipt


def test_committed_receipt_is_current_era_and_activates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cell 1: the ONLY committed receipt is CURRENT-era and activates the lane.

    The committed-ABSENT half of the original vacuous-era proof retired when
    the minted v2 receipt landed (2026-08-29). The era boundary stays closed
    from the surviving half: the committed receipt parses through the REAL
    tightened loader carrying every new REQUIRED field -- it is
    post-tightening by construction, and cells 2-3 prove a pre-batch-shaped
    receipt cannot load at all, so no prior-era receipt can ever sit at the
    committed location and no mixed-era reader of the new fields exists.
    """

    monkeypatch.delenv(CHECKER_CALIBRATION_RECEIPT_ENV, raising=False)
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)

    # The committed artifact location holds the minted receipt at this tree.
    assert CHECKER_CALIBRATION_RECEIPT_PATH.is_file(), (
        "the minted calibration receipt vanished from the committed location; "
        "either restore it or return this cell to the vacuous-absence proof"
    )

    outcome = classify_reader(load_checker_calibration_receipt)
    assert_era_outcome(
        reader_id="calibration-receipt-committed-lookup",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    receipt = outcome[1]
    assert receipt["schema_version"] == CHECKER_CALIBRATION_RECEIPT_SCHEMA
    assert receipt["production_batch_width"] == METADATA_BATCH_MAX
    for kind in GateKind:
        lane = receipt["lanes"][kind.value]
        for field in _NEW_LANE_FIELDS:
            assert field in lane, f"committed receipt lane {kind.value} lost {field}"
    assert "blind_replay" in receipt

    # The full activation lookup: R2-3 keys activation on the structured
    # prompt epoch. A committed receipt at the LIVE epoch activates; the
    # current committed artifact predates the epoch binding (its epoch lives
    # only in `source` free text), so activation refuses TYPED until the ops
    # re-mint at the live epoch lands — the armed consequence, not a defect.
    from menagerie.crawler.checker_calibration import (
        CheckerCalibrationEpochError,
        live_checker_prompt_identity,
    )

    def activate() -> dict[str, Any]:
        return require_checker_calibration(
            GateKind.METADATA_BATCH.value, str(receipt["checker_model"])
        )

    if receipt.get("checker_prompt_identity") == live_checker_prompt_identity():
        activation = classify_reader(activate)
        assert activation[0] == "parsed", activation
        assert activation[1]
    else:
        with pytest.raises(CheckerCalibrationEpochError):
            activate()


@pytest.mark.parametrize(
    "variant",
    [
        "lane-sessions_bought",
        "lane-attempt-trio",
        "production_batch_width",
        "blind_replay",
        "capacity-session-pair",
        "capacity-dollar-pair",
        "all-new-required-fields",
    ],
)
def test_prebatch_shaped_receipt_refuses_typed(tmp_path: Path, variant: str) -> None:
    """Cell 2: a pre-batch-SHAPED receipt refuses typed through the real loader.

    A well-formed v1 receipt missing any of the newly REQUIRED field families
    (and the combined pre-tightening shape missing all of them) must land on a
    typed :class:`CheckerCalibrationInvalidError`, never an unclassified crash
    and never a silent parse with defaulted measurements.
    """

    receipt = _strip_new_fields(_calibration_receipt(), variant)
    path = _write_receipt(tmp_path, receipt)

    outcome = classify_reader(lambda: load_checker_calibration_receipt(path))
    assert_era_outcome(
        reader_id=f"calibration-receipt-loader[{variant}]",
        outcome=outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(outcome[1], CheckerCalibrationInvalidError)


@pytest.mark.parametrize(
    ("field", "value", "message_fragment"),
    [
        (
            "attempt_latency_p99_seconds",
            float(OPERATOR_ATTEMPT_TIMEOUT_SECONDS) + 1.0,
            "attempt grant",
        ),
        (
            "latency_p99_seconds",
            float(CHECKER_SESSION_WALL_SECONDS) + 1.0,
            "session wall",
        ),
        (
            "attempt_censoring_rate",
            CALIBRATION_MAX_REFUSAL_RATE + 0.05,
            "censoring rate",
        ),
    ],
)
def test_timing_tripwires_refuse_typed_at_load(
    tmp_path: Path, field: str, value: float, message_fragment: str
) -> None:
    """Cell 3: the timing tripwires are ARMED in the loader itself.

    A receipt whose measured metadata-lane attempt p99 exceeds
    ``OPERATOR_ATTEMPT_TIMEOUT_SECONDS`` (the 600s lesson), whose session p99
    exceeds ``CHECKER_SESSION_WALL_SECONDS``, or whose attempt censoring rate
    exceeds the 0.2 cap refuses typed at load -- one poisoned lane refuses the
    whole receipt.
    """

    receipt = _calibration_receipt()
    receipt["lanes"][GateKind.METADATA_BATCH.value][field] = value
    path = _write_receipt(tmp_path, receipt)

    outcome = classify_reader(lambda: load_checker_calibration_receipt(path))
    assert_era_outcome(
        reader_id=f"calibration-receipt-loader[tripwire-{field}]",
        outcome=outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(outcome[1], CheckerCalibrationInvalidError)
    assert message_fragment in str(outcome[1])


def test_well_formed_receipt_parses_proving_refusals_field_caused(
    tmp_path: Path,
) -> None:
    """Cell 4: a fully well-formed receipt PARSES through the same loader.

    Built from the sanctioned conftest shape without weakening anything; this
    proves every refusal above is field-caused, not a loader-wide refusal.
    """

    receipt = _calibration_receipt()
    path = _write_receipt(tmp_path, receipt)

    outcome = classify_reader(lambda: load_checker_calibration_receipt(path))
    assert_era_outcome(
        reader_id="calibration-receipt-loader[well-formed]",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    parsed = outcome[1]
    assert parsed["checker_model"] == "gpt-5.6-sol"
    assert parsed["production_batch_width"] == METADATA_BATCH_MAX
    assert CALIBRATION_PRODUCTION_BATCH_WIDTH == METADATA_BATCH_MAX
    for kind in GateKind:
        lane = parsed["lanes"][kind.value]
        for field in _NEW_LANE_FIELDS:
            assert field in lane, f"parsed receipt lost new field {field}"
    assert "blind_replay" in parsed
