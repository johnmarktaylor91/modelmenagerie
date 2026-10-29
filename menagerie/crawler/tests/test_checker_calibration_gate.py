"""BL-10 (a)+(i): sol calibration is a fail-closed activation gate, not a hope.

The all-sol checker rotation pinned ``gpt-5.6-sol`` for all three gate kinds
with ZERO lifetime sol sessions on ``metadata_batch``/``terminal_disposition``
and terra-recorded goldens on both lanes, and the launch plan carried no
checker capacity budget at all (measured: 51.15h of wall for 67 models, 28%
rendering no verdict). These tests pin the typed activation transaction:

* a CLOSED INVENTORY over :class:`~menagerie.crawler.constants.GateKind` -- a
  new gate kind cannot ship without a measured lane in the receipt, and a
  receipt cannot activate a lane whose committed goldens still record a
  different checker model;
* the capacity governor fails closed at the margin, BEFORE overspend;
* the real command lane refuses to buy ANY session for an uncalibrated lane --
  proven with a lane whose subprocess would fail the test if invoked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checker_calibration import (
    ACCEPTANCE_DISAGREEMENT_OUTCOME,
    ACCEPTANCE_UNFUNDED_OUTCOME,
    ACCEPTANCE_WITNESS_MINIMUM,
    CALIBRATION_MIN_SESSIONS,
    CONTESTED_CENSUS_ARM,
    CHECKER_CALIBRATION_RECEIPT_ENV,
    CHECKER_CALIBRATION_RECEIPT_PATH,
    CHECKER_CALIBRATION_RECEIPT_SCHEMA,
    CHECKER_CALIBRATION_RUN_ENV,
    CHECKER_CAPACITY_MARGIN,
    CHECKER_GOLDEN_CORPUS_DIR,
    CHECKER_GOLDEN_CORPUS_ENV,
    LOCATOR_ATTESTED_BY,
    TEXT_LINE_LOCATOR_BASIS,
    CheckerCalibrationEpochError,
    CheckerCalibrationInvalidError,
    CheckerCalibrationMissingError,
    CheckerCapacityMarginError,
    CheckerGoldenCorpusError,
    live_checker_prompt_identity,
    load_checker_calibration_receipt,
    require_checker_calibration,
)
from menagerie.crawler.constants import METADATA_BATCH_MAX, GateKind
from menagerie.crawler.operator_protocol import (
    CHECKER_SESSION_WALL_SECONDS,
    OPERATOR_ATTEMPT_TIMEOUT_SECONDS,
)

pytestmark = pytest.mark.smoke


def _receipt(**overrides: Any) -> dict[str, Any]:
    """Return one structurally complete calibration receipt."""

    lanes = {
        kind.value: {
            "sessions": CALIBRATION_MIN_SESSIONS[kind.value],
            "sessions_bought": CALIBRATION_MIN_SESSIONS[kind.value] + 2,
            "latency_p95_seconds": 400.0,
            "latency_p99_seconds": 550.0,
            "output_tokens_p99": 18_000,
            "refusal_rate": 0.05,
            "attempt_latency_p99_seconds": 600.0,
            "attempt_samples": CALIBRATION_MIN_SESSIONS[kind.value],
            "attempt_censoring_rate": 0.0,
        }
        for kind in GateKind
    }
    receipt: dict[str, Any] = {
        "schema_version": CHECKER_CALIBRATION_RECEIPT_SCHEMA,
        "checker_model": "gpt-5.6-sol",
        "captured_at": "2026-08-17T00:00:00Z",
        "source": "test fixture",
        # R2-3: the fixture binds the LIVE prompt epoch; the epoch-door RED
        # tests override this field explicitly.
        "checker_prompt_identity": live_checker_prompt_identity(),
        "production_batch_width": METADATA_BATCH_MAX,
        "lanes": lanes,
        "blind_replay": {
            "goldens": 9,
            "replays_per_golden": 3,
            "same_epoch_disagreements": 0,
            "rotation_replay_ran": True,
            "rotation_disagreements": 1,
            "width_invariance": {
                "widths": sorted({1, METADATA_BATCH_MAX}),
                "disagreements": 0,
                "degenerate_at_production_width": METADATA_BATCH_MAX == 1,
                "folded_into_same_epoch": METADATA_BATCH_MAX == 1,
            },
            "disagreement_diagnosis": [],
            "mechanical_claims": {
                "required": 0,
                "discharged": 0,
                "refuted": 0,
                "coverage_gaps": 0,
            },
            "locator_attestation": {
                "checks": 142,
                "failures": 0,
                "unattestable": 10,
                "basis": TEXT_LINE_LOCATOR_BASIS,
                "attested_by": LOCATOR_ATTESTED_BY,
                "keyset_sha256": "sha256:" + "c" * 64,
            },
            "acceptance_witness": {
                "metadata_accept_witnesses": 3,
                "fidelity_accept_witnesses": 3,
                "terminal_accept_witnesses": 3,
                "required_witnesses_per_accept": ACCEPTANCE_WITNESS_MINIMUM,
                "missing_confirmations": 0,
                "confirmation_disagreements": 0,
                "disagreement_outcome": ACCEPTANCE_DISAGREEMENT_OUTCOME,
                "unfunded_confirmation_outcome": ACCEPTANCE_UNFUNDED_OUTCOME,
                "contested_census_arm": CONTESTED_CENSUS_ARM,
            },
            "p99_at_production_width_seconds": 550.0,
            "report_sha256": "sha256:" + "b" * 64,
        },
        "capacity": {
            "next_rung_models": 200,
            "projected_sessions": 800,
            "session_budget": 2_000,
            "projected_wall_hours": 60.0,
            "wall_budget_hours": 120.0,
            "projected_dollars": 120.0,
            "dollar_budget": 300.0,
        },
    }
    receipt.update(overrides)
    return receipt


def _write_receipt(tmp_path: Path, receipt: dict[str, Any]) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _write_corpus(tmp_path: Path, models: dict[str, str]) -> Path:
    """Write one synthetic golden corpus dir with a per-kind checker model."""

    corpus = tmp_path / "goldens"
    corpus.mkdir(exist_ok=True)
    prefixes = {
        GateKind.METADATA_BATCH.value: "metadata_accept",
        GateKind.FIDELITY.value: "fidelity_accept",
        GateKind.TERMINAL_DISPOSITION.value: "terminal_accept",
    }
    for kind, name in prefixes.items():
        (corpus / f"{name}.json").write_text(
            json.dumps({"checker_model": models[kind]}), encoding="utf-8"
        )
    return corpus


@pytest.fixture()
def calibrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Point the gate at a valid fixture receipt and sol-recorded corpus."""

    receipt = _receipt()
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(_write_receipt(tmp_path, receipt)))
    monkeypatch.setenv(
        CHECKER_GOLDEN_CORPUS_ENV,
        str(_write_corpus(tmp_path, {kind.value: "gpt-5.6-sol" for kind in GateKind})),
    )
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    return receipt


# ---------------------------------------------------------------------------
# Fail-closed structure: missing, malformed, under-measured, over-margin.
# ---------------------------------------------------------------------------


def test_absent_receipt_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(tmp_path / "missing.json"))
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    with pytest.raises(CheckerCalibrationMissingError):
        require_checker_calibration("metadata_batch", "gpt-5.6-sol")


def test_the_committed_receipt_state_is_honest() -> None:
    """The committed artifact either validates or provably does not exist.

    Today no sol calibration has run, so the committed receipt must be ABSENT
    (the gate refuses everywhere real sessions would be bought). The moment a
    receipt lands, this arm flips to requiring that it validates -- either way
    a half-broken committed receipt can never sit unnoticed.
    """

    if CHECKER_CALIBRATION_RECEIPT_PATH.is_file():
        load_checker_calibration_receipt(CHECKER_CALIBRATION_RECEIPT_PATH)
    else:
        with pytest.raises(CheckerCalibrationMissingError):
            load_checker_calibration_receipt(CHECKER_CALIBRATION_RECEIPT_PATH)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(schema_version="bogus.v9"),
        lambda r: r.update(checker_model=""),
        lambda r: r["lanes"].pop("terminal_disposition"),
        lambda r: r["lanes"]["metadata_batch"].pop("output_tokens_p99"),
        lambda r: r["lanes"]["metadata_batch"].update(
            sessions=CALIBRATION_MIN_SESSIONS["metadata_batch"] - 1
        ),
        lambda r: r["lanes"]["fidelity"].update(refusal_rate=0.5),
        lambda r: r.pop("capacity"),
        lambda r: r["capacity"].pop("wall_budget_hours"),
        lambda r: r["lanes"]["metadata_batch"].pop("sessions_bought"),
        lambda r: r["lanes"]["metadata_batch"].update(
            sessions_bought=CALIBRATION_MIN_SESSIONS["metadata_batch"] - 1
        ),
        lambda r: r["lanes"]["fidelity"].pop("attempt_latency_p99_seconds"),
        lambda r: r["lanes"]["fidelity"].update(attempt_censoring_rate=0.5),
        lambda r: r.pop("production_batch_width"),
        lambda r: r.update(production_batch_width=3),
        lambda r: r["capacity"].pop("dollar_budget"),
        lambda r: r["capacity"].pop("session_budget"),
        lambda r: r.pop("blind_replay"),
        lambda r: r["blind_replay"].update(same_epoch_disagreements=1),
        lambda r: r["blind_replay"].update(replays_per_golden=2),
        lambda r: r["blind_replay"].update(rotation_replay_ran=False),
        lambda r: r["blind_replay"]["width_invariance"].update(disagreements=1),
        # Width 1 is REQUIRED in the invariance arm under every production
        # width (at production width 1 the comparison is degenerate but still
        # declared), so a widths list without it refuses width-generically.
        lambda r: r["blind_replay"]["width_invariance"].update(widths=[2, 3]),
        lambda r: r["blind_replay"].update(report_sha256=""),
        # v2 zero bars (rule-2 resolution, 2026-08-29). Each one refuses alone.
        lambda r: r.update(schema_version="menagerie.crawler.checker-calibration-receipt.v1"),
        lambda r: r["blind_replay"]["width_invariance"].pop("degenerate_at_production_width"),
        lambda r: r["blind_replay"].pop("disagreement_diagnosis"),
        lambda r: r["blind_replay"].update(
            disagreement_diagnosis=[
                {
                    "golden": "metadata_accept",
                    "item": "m10470",
                    "field": "integrity",
                    "poles": ["accurate", "inaccurate"],
                    "pole_class": "contradiction",
                    "mechanical_refutation": "refuted",
                    "defect_class": "fabricated-evidence-claim",
                }
            ]
        ),
        lambda r: r["blind_replay"].pop("mechanical_claims"),
        lambda r: r["blind_replay"]["mechanical_claims"].update(refuted=1),
        lambda r: r["blind_replay"]["mechanical_claims"].update(coverage_gaps=2),
        lambda r: r["blind_replay"]["mechanical_claims"].update(required=5, discharged=4),
        lambda r: r["blind_replay"].pop("locator_attestation"),
        lambda r: r["blind_replay"]["locator_attestation"].update(failures=1),
        lambda r: r["blind_replay"]["locator_attestation"].update(attested_by="checker"),
        lambda r: r["blind_replay"]["locator_attestation"].update(checks=0),
        lambda r: r["blind_replay"]["locator_attestation"].update(keyset_sha256=""),
        lambda r: r["blind_replay"].pop("acceptance_witness"),
        lambda r: r["blind_replay"]["acceptance_witness"].update(
            required_witnesses_per_accept=1
        ),
        lambda r: r["blind_replay"]["acceptance_witness"].update(missing_confirmations=1),
        lambda r: r["blind_replay"]["acceptance_witness"].update(
            confirmation_disagreements=1
        ),
        lambda r: r["blind_replay"]["acceptance_witness"].update(
            disagreement_outcome="re-roll"
        ),
    ],
    ids=[
        "unknown-schema",
        "empty-model",
        "missing-lane",
        "missing-measurement",
        "under-session-floor",
        "refusal-rate-too-high",
        "no-capacity-plan",
        "no-wall-budget",
        "no-bought-denominator",
        "bought-below-rendered",
        "no-attempt-p99",
        "attempt-quantile-censored",
        "no-production-width",
        "off-production-width",
        "no-dollar-budget",
        "no-session-budget",
        "no-blind-replay-matrix",
        "same-epoch-disagreement",
        "under-three-replays",
        "no-rotation-replay",
        "width-variance",
        "widths-missing-production",
        "unbound-replay-report",
        "retired-v1-schema",
        "undeclared-width-degeneracy",
        "no-disagreement-diagnosis",
        "mechanically-refuted-draw",
        "no-mechanical-claims",
        "mechanical-claim-refuted",
        "mechanical-coverage-gap",
        "mechanical-claims-undischarged",
        "no-locator-attestation",
        "locator-failure",
        "model-adjudicated-locators",
        "empty-locator-attestation",
        "unbound-attestation-keyset",
        "no-acceptance-witness",
        "acceptance-depth-below-two",
        "missing-accept-confirmation",
        "disagreeing-accept-confirmation",
        "non-park-disagreement-outcome",
    ],
)
def test_malformed_or_under_measured_receipt_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    receipt = _receipt()
    mutate(receipt)
    path = _write_receipt(tmp_path, receipt)
    with pytest.raises(CheckerCalibrationInvalidError):
        load_checker_calibration_receipt(path)


def test_the_600s_lesson_walls_must_cover_measured_p99(tmp_path: Path) -> None:
    """W2 timing re-derivation tripwire: a timing constant below the measured
    p99 censors sessions from outside (the 600s lesson: the 600s attempt grant
    censored EVERY sol metadata attempt on 2026-08-21). A receipt whose
    measured tails the CURRENT constants cannot contain refuses the lane --
    the constant must be re-derived in the same reviewed change as the
    receipt, never left hand-set below the evidence."""

    receipt = _receipt()
    receipt["lanes"]["metadata_batch"]["attempt_latency_p99_seconds"] = (
        float(OPERATOR_ATTEMPT_TIMEOUT_SECONDS) + 1.0
    )
    with pytest.raises(CheckerCalibrationInvalidError, match="600s lesson"):
        load_checker_calibration_receipt(_write_receipt(tmp_path, receipt))

    receipt = _receipt()
    receipt["lanes"]["terminal_disposition"]["latency_p99_seconds"] = (
        float(CHECKER_SESSION_WALL_SECONDS) + 1.0
    )
    with pytest.raises(CheckerCalibrationInvalidError, match="600s lesson"):
        load_checker_calibration_receipt(_write_receipt(tmp_path, receipt))


def test_production_batch_width_pin_is_the_metadata_cap() -> None:
    """The receipt's width pin IS the production metadata cap constant: a
    calibration measured at any other width is measuring a different lane."""

    from menagerie.crawler.checker_calibration import CALIBRATION_PRODUCTION_BATCH_WIDTH

    assert CALIBRATION_PRODUCTION_BATCH_WIDTH == METADATA_BATCH_MAX


def test_width_one_is_the_calibrated_production_width(tmp_path: Path) -> None:
    """Rule-1 pin (Opus adjudication 5.3, fired 2026-08-29): metadata batches
    run at width ONE -- the Arm A matrix measured width-5 acceptance flipping
    on 8 of 25 item-cells across same-epoch replays while width 1 flipped 0 of
    18, so width is the mechanism and the production lane is single-item. A
    receipt measured at the retired width-5 lane refuses BY WIDTH: the width
    rotation can never ride the old lane's evidence."""

    assert METADATA_BATCH_MAX == 1
    from menagerie.crawler.checker_calibration import CALIBRATION_PRODUCTION_BATCH_WIDTH

    assert CALIBRATION_PRODUCTION_BATCH_WIDTH == 1
    receipt = _receipt(production_batch_width=5)
    with pytest.raises(CheckerCalibrationInvalidError, match="measured at batch width 5"):
        load_checker_calibration_receipt(_write_receipt(tmp_path, receipt))


def test_capacity_margin_fails_closed_before_overspend(tmp_path: Path) -> None:
    """Projected usage over the margin refuses even when under the raw budget."""

    receipt = _receipt()
    receipt["capacity"]["projected_wall_hours"] = (
        receipt["capacity"]["wall_budget_hours"] * CHECKER_CAPACITY_MARGIN + 1.0
    )
    with pytest.raises(CheckerCapacityMarginError):
        load_checker_calibration_receipt(_write_receipt(tmp_path, receipt))


def test_declared_token_budget_is_enforced(tmp_path: Path) -> None:
    receipt = _receipt()
    receipt["capacity"]["projected_output_tokens"] = 90_000_000
    receipt["capacity"]["token_budget"] = 100_000_000
    with pytest.raises(CheckerCapacityMarginError):
        load_checker_calibration_receipt(_write_receipt(tmp_path, receipt))


# ---------------------------------------------------------------------------
# The closed inventory over gate kinds, corroborated by re-recorded goldens.
# ---------------------------------------------------------------------------


def test_every_gate_kind_is_in_the_closed_inventory(calibrated: dict[str, Any]) -> None:
    """Activation walks the CLOSED GateKind enum; each kind must be measured."""

    for kind in GateKind:
        assert require_checker_calibration(kind.value, "gpt-5.6-sol")


def test_receipt_for_another_model_refuses(calibrated: dict[str, Any]) -> None:
    with pytest.raises(CheckerCalibrationMissingError):
        require_checker_calibration("metadata_batch", "gpt-7-hypothetical")


# ---------------------------------------------------------------------------
# R2-3: the receipt-epoch door.
# ---------------------------------------------------------------------------


def test_receipt_without_a_structured_epoch_refuses_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed-receipt shape: no ``checker_prompt_identity`` never activates.

    This is the exact R2-3 hole — the shipping prompt rotated and NOTHING
    refused, because the receipt recorded its epoch only as free text inside
    ``source``. The receipt stays LOADABLE (census and diagnosis keep working);
    only the verdict-rendering activation door refuses, typed, arming the
    re-mint at the live epoch.
    """

    receipt = _receipt()
    del receipt["checker_prompt_identity"]
    receipt["source"] = "minted under prompt epoch sha256:" + "d7" * 32 + " (free text only)"
    path = _write_receipt(tmp_path, receipt)
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(path))
    monkeypatch.setenv(
        CHECKER_GOLDEN_CORPUS_ENV,
        str(_write_corpus(tmp_path, {kind.value: "gpt-5.6-sol" for kind in GateKind})),
    )
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    # Loadable for diagnosis...
    assert load_checker_calibration_receipt(path)["checker_model"] == "gpt-5.6-sol"
    # ...but never activating: the free-text epoch is prose, not evidence.
    with pytest.raises(CheckerCalibrationEpochError, match="no structured"):
        require_checker_calibration("metadata_batch", "gpt-5.6-sol")


def test_receipt_at_a_superseded_epoch_refuses_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt minted at a rotated-away prompt epoch measured another checker."""

    stale = _receipt(checker_prompt_identity="sha256:" + "d7" * 32)
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(_write_receipt(tmp_path, stale)))
    monkeypatch.setenv(
        CHECKER_GOLDEN_CORPUS_ENV,
        str(_write_corpus(tmp_path, {kind.value: "gpt-5.6-sol" for kind in GateKind})),
    )
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    with pytest.raises(CheckerCalibrationEpochError, match="shipping checker prompt epoch"):
        require_checker_calibration("metadata_batch", "gpt-5.6-sol")


def test_garbage_epoch_field_refuses_even_the_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present-but-shapeless epoch is malformed evidence, not a missing field."""

    path = _write_receipt(tmp_path, _receipt(checker_prompt_identity="epoch-7"))
    with pytest.raises(CheckerCalibrationInvalidError, match="checker_prompt_identity"):
        load_checker_calibration_receipt(path)


def test_live_epoch_is_the_shipping_prompt_content_hash() -> None:
    """The door's closed comparison target is the exact shipped prompt bytes."""

    from pathlib import Path as _Path

    from menagerie.crawler.constants import CHECKER_PROMPT_NAME
    from menagerie.crawler.identity import hash_bytes

    prompt_path = (
        _Path("menagerie/crawler/prompts") / f"{CHECKER_PROMPT_NAME}.txt"
    )
    assert live_checker_prompt_identity() == hash_bytes(prompt_path.read_bytes())


def test_the_committed_receipt_cannot_activate_until_reminted_at_the_live_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The armed consequence, pinned on the real committed artifact.

    The committed receipt predates the epoch binding (its epoch lives only in
    ``source`` free text), so every verdict-rendering lane refuses it until
    the ops re-mint at the live epoch lands. The pin flips automatically the
    day the re-minted receipt carries the structured field.
    """

    if not CHECKER_CALIBRATION_RECEIPT_PATH.is_file():
        pytest.skip("no committed receipt on this tree")
    committed = load_checker_calibration_receipt(CHECKER_CALIBRATION_RECEIPT_PATH)
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(CHECKER_CALIBRATION_RECEIPT_PATH))
    monkeypatch.setenv(
        CHECKER_GOLDEN_CORPUS_ENV,
        str(_write_corpus(tmp_path, {kind.value: str(committed["checker_model"]) for kind in GateKind})),
    )
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    recorded = committed.get("checker_prompt_identity")
    if recorded == live_checker_prompt_identity():
        # The re-mint landed at the live epoch: the door is satisfied.
        assert require_checker_calibration("metadata_batch", str(committed["checker_model"]))
    else:
        with pytest.raises(CheckerCalibrationEpochError):
            require_checker_calibration("metadata_batch", str(committed["checker_model"]))


def test_terra_goldens_cannot_activate_a_sol_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt alone cannot open a lane whose goldens name a retired checker.

    This is the committed-corpus state today: metadata and terminal goldens are
    terra transcripts. Until they are re-recorded under sol, those lanes refuse
    even with a perfect capacity receipt.
    """

    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(_write_receipt(tmp_path, _receipt())))
    monkeypatch.setenv(
        CHECKER_GOLDEN_CORPUS_ENV,
        str(
            _write_corpus(
                tmp_path,
                {
                    GateKind.METADATA_BATCH.value: "gpt-5.6-terra",
                    GateKind.FIDELITY.value: "gpt-5.6-sol",
                    GateKind.TERMINAL_DISPOSITION.value: "gpt-5.6-terra",
                },
            )
        ),
    )
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    assert require_checker_calibration("fidelity", "gpt-5.6-sol")
    with pytest.raises(CheckerGoldenCorpusError):
        require_checker_calibration("metadata_batch", "gpt-5.6-sol")
    with pytest.raises(CheckerGoldenCorpusError):
        require_checker_calibration("terminal_disposition", "gpt-5.6-sol")


def test_the_committed_corpus_is_recorded_under_the_pinned_sol_checker() -> None:
    """The corpus-state fact the gate derives from, pinned so a re-record is loud.

    The R6 two-pass re-record (2026-08-27) converged fidelity and terminal on
    the pinned sol checker, and the width-1 rotation's re-harvest (2026-08-29)
    converged the metadata classes too -- the retired terra cap-width pairs
    cannot re-seal at the single-item production width. Every committed entry
    now records the pinned checker; any future rotation must update this pin
    in the same reviewed change as its re-record.
    """

    models: dict[str, set[str]] = {"metadata": set(), "terminal": set(), "fidelity": set()}
    for entry in sorted(CHECKER_GOLDEN_CORPUS_DIR.glob("*.json")):
        data = json.loads(entry.read_text(encoding="utf-8"))
        for prefix in models:
            if entry.name.startswith(prefix):
                models[prefix].add(str(data.get("checker_model")))
    assert models["fidelity"] == {"gpt-5.6-sol"}
    assert models["metadata"] == {"gpt-5.6-sol"}
    assert models["terminal"] == {"gpt-5.6-sol"}


def test_calibration_run_env_is_the_only_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sanctioned capture run buys measurement sessions; nothing else does."""

    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(tmp_path / "missing.json"))
    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    assert require_checker_calibration("metadata_batch", "gpt-5.6-sol") == {}
    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "0")
    with pytest.raises(CheckerCalibrationMissingError):
        require_checker_calibration("metadata_batch", "gpt-5.6-sol")


# ---------------------------------------------------------------------------
# The real command lane refuses BEFORE buying anything.
# ---------------------------------------------------------------------------


def test_command_lane_leg_refuses_uncalibrated_before_any_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncalibrated leg refuses at gate readiness, before any spend.

    The lane's command is a poison argv: executing it would fail the test, so
    passing proves the refusal happened strictly before any session purchase.
    """

    from menagerie.crawler.checker_calibration import CHECKER_ACTIVATION_ENV
    from menagerie.crawler.driver_admission import CommandCheckerLane
    from menagerie.crawler.operator_checker import required_checker_model
    from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot

    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(tmp_path / "missing.json"))
    # Uncalibrated means NEITHER a receipt NOR an activation-matrix row (the
    # committed activation covers the Fable seat at the live epoch).
    monkeypatch.setenv(CHECKER_ACTIVATION_ENV, str(tmp_path / "missing-activation.json"))
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    snapshot = _snapshot(tmp_path, count=1)
    lane = CommandCheckerLane(("/nonexistent-checker-wrapper-must-never-run",))
    # The real command lane seals the seated primary; the campaign binding
    # must name it (as the CLI does), or the lane refuses on that first.
    driver = _driver(
        tmp_path, snapshot, checker=lane, checker_model=required_checker_model("metadata_batch")
    )
    with pytest.raises(CheckerCalibrationMissingError):
        driver.run()
