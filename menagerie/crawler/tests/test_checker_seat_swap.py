"""Seat swap (SEAT_RULING.md, 2026-09-01): Fable 5.1 primary, sol cross-model K=2.

DECIDED_PLAN item 32 / J6. The checker provider is DERIVED from the seated
model everywhere an identity is sealed; the wrapper selects one adapter per
session (Codex or the Claude CLI) behind one prompt/schema/stamp/validate
contract; the conflict ledger's seat-table pairing rule decides which K=2
pairs are lawful; terminal K=2 is a deterministic 10 percent sample; the
committed activation matrix activates the Fable seat without a receipt.

The Fable path runs through a FAKE Claude CLI (an injected ``invoke`` that
returns a real-shaped harness JSON document); the real path was proven by the
bench (27 arm-B sessions, all model-witnessed).
"""

from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from menagerie.crawler import checker_conflicts as cc
from menagerie.crawler.checker_calibration import (
    CHECKER_ACTIVATION_ENV,
    CHECKER_CALIBRATION_RECEIPT_ENV,
    CHECKER_CALIBRATION_RUN_ENV,
    checker_activation_for,
    evaluate_checker_calibration_mode,
    live_checker_prompt_identity,
    load_checker_activations,
)
from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    CheckerPauseReason,
    build_fidelity_envelope,
    build_metadata_vet_envelope,
    classify_checker_response,
)
from menagerie.crawler.checker_providers import (
    CLAUDE_CHECKER_ALLOWED_TOOLS,
    CLAUDE_CHECKER_DISALLOWED_TOOLS,
    CLAUDE_OVERLOADED_STATUS_BASIS,
    PROVIDER_OVERLOAD_EVENT_BASES,
    CheckerAttempt,
    adapter_for_model,
)
from menagerie.crawler.constants import (
    CHECKER_CLAUDE_CLI_VERSION,
    CHECKER_SEATS,
    TERMINAL_CONFIRMATION_SAMPLE_RATE,
    TIER_CAMPAIGN_CHECKER_MODEL_HISTORY,
    CheckerSeatRole,
    GateKind,
    checker_confirmation_partner,
    checker_provider_for_model,
    checker_seat,
    seated_checker_models,
    terminal_confirmation_sampled,
)
from menagerie.crawler.driver_admission import (
    CommandCheckerLane,
    DriverIntegrationError,
    _checker_provider_weather_status,
    buy_confirmation_session,
)
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.gate_reuse import lane_confirmation_identity
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.operator_checker import execute_checker_request, required_checker_model
from menagerie.crawler.operator_protocol import (
    OperatorExitCode,
    status_sidecar_path,
    telemetry_path,
)
from menagerie.crawler.partitioner import CAMPAIGN_SPECS
from menagerie.crawler.tests.test_accuracy_duplicate_accept import _fidelity_gate, _metadata_gate
from menagerie.crawler.tests.test_checker_verdict_conflicts import _td_gate
from menagerie.crawler.tests.test_operator_checker import _checker_item_pack, _request_and_result
from menagerie.crawler.toolchain_pins import (
    checker_toolchain_lane,
    expected_cli_version,
    toolchain_identity,
    verify_toolchain_attestation,
)

pytestmark = pytest.mark.smoke

FABLE = "claude-fable-5-1"
SOL = "gpt-5.6-sol"
PROMPT = "sha256:" + "p" * 64


def _identity(model: str, *, prompt: str = PROMPT, version: str = "current") -> dict[str, str]:
    return {
        "provider": checker_provider_for_model(model),
        "model": model,
        "version": version,
        "prompt_sha256": prompt,
    }


def _harness(items: list[dict[str, Any]], *, model: str = FABLE, **overrides: Any) -> str:
    """One real-shaped Claude CLI ``--output-format json`` result document."""

    document: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "terminal_reason": "completed",
        "api_error_status": None,
        "num_turns": 7,
        "total_cost_usd": 2.31,
        "usage": {
            "input_tokens": 376,
            "cache_creation_input_tokens": 103856,
            "cache_read_input_tokens": 346761,
            "output_tokens": 30193,
        },
        "modelUsage": {
            "claude-haiku-4-5-20251001": {"inputTokens": 10750, "outputTokens": 13},
            model: {"inputTokens": 376, "outputTokens": 30193},
        },
        "structured_output": {"items": items},
        "result": json.dumps({"items": items}),
    }
    document.update(overrides)
    return json.dumps(document)


def _telemetry(request_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in telemetry_path(request_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _status(request_path: Path) -> dict[str, Any]:
    return json.loads(status_sidecar_path(request_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. The seat table is the ruling, and every identity derives from it.
# ---------------------------------------------------------------------------


def test_seat_table_is_the_ruling() -> None:
    for kind in GateKind:
        assert checker_seat(kind, CheckerSeatRole.PRIMARY).model == FABLE
        assert checker_seat(kind, CheckerSeatRole.PRIMARY).provider == "anthropic"
        assert checker_seat(kind, CheckerSeatRole.CONFIRMATION).model == SOL
        assert checker_seat(kind, CheckerSeatRole.CONFIRMATION).provider == "openai"
        assert seated_checker_models(kind) == {FABLE, SOL}
        assert required_checker_model(kind) == FABLE
        assert required_checker_model(kind, "confirmation") == SOL
    # Every accept confirms on metadata and fidelity; terminal is a 10 percent sample.
    assert checker_seat("metadata_batch", "confirmation").confirmation_sample_rate == 1.0
    assert checker_seat("fidelity", "confirmation").confirmation_sample_rate == 1.0
    assert (
        checker_seat("terminal_disposition", "confirmation").confirmation_sample_rate
        == TERMINAL_CONFIRMATION_SAMPLE_RATE
        == 0.10
    )
    assert set(CHECKER_SEATS) == {(kind, role) for kind in GateKind for role in CheckerSeatRole}
    # Cross-model partner: the seated primary pairs with sol; anything else with itself.
    assert checker_confirmation_partner("fidelity", FABLE).model == SOL
    assert checker_confirmation_partner("fidelity", SOL) is None
    assert checker_confirmation_partner("fidelity", "codex") is None


def test_campaign_specs_and_history_end_at_the_seated_primary() -> None:
    """J6: every campaign's live checker_model IS the seat table's primary."""

    for spec in CAMPAIGN_SPECS:
        assert spec.checker_model == FABLE
        history = TIER_CAMPAIGN_CHECKER_MODEL_HISTORY[spec.campaign_id]
        assert history[-1] == FABLE
        # Era-honest: sol stays declared so every sol-labelled record validates.
        assert SOL in history


def test_provider_is_derived_from_the_model_never_declared() -> None:
    assert checker_provider_for_model(FABLE) == "anthropic"
    assert checker_provider_for_model(SOL) == "openai"
    # The era default: every pre-swap label (and every unseated test label).
    assert checker_provider_for_model("gpt-5.6-terra") == "openai"
    assert checker_provider_for_model("codex") == "openai"
    assert adapter_for_model(FABLE).provider == "anthropic"
    assert adapter_for_model(SOL).provider == "openai"


def test_envelope_builder_derives_the_provider_from_the_seated_model(tmp_path: Path) -> None:
    """RED on the pre-swap builder: ``checker.provider`` was the literal openai."""

    from menagerie.crawler.tests.conftest import make_gate

    item = _checker_item_pack(make_gate(["m_seat"])["items"][0])
    fable = build_metadata_vet_envelope(
        [item],
        gate_round=1,
        output_path=tmp_path / "a" / "result.json",
        checker_model=FABLE,
        checker_version="current",
        request_nonce="seat-a",
        final_tail=True,
    )
    sol = build_fidelity_envelope(
        item,
        gate_round=1,
        output_path=tmp_path / "b" / "result.json",
        checker_model=SOL,
        checker_version="current",
        request_nonce="seat-b",
    )
    assert fable["checker"]["provider"] == "anthropic"
    assert fable["checker"]["model"] == FABLE
    assert sol["checker"]["provider"] == "openai"
    assert sol["checker"]["model"] == SOL


# ---------------------------------------------------------------------------
# 2. The wrapper: both seats admitted, provider re-derived, adapters selected.
# ---------------------------------------------------------------------------


def _reseal_checker(request_path: Path, **checker_changes: Any) -> None:
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    envelope["checker"] = {**envelope["checker"], **checker_changes}
    envelope.pop("envelope_sha256")
    envelope["envelope_sha256"] = stable_hash(envelope)
    request_path.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")


def test_wrapper_refuses_a_forged_provider_before_any_session(tmp_path: Path) -> None:
    request_path, _result = _request_and_result(tmp_path, model=FABLE)
    _reseal_checker(request_path, provider="openai")
    called = False

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CheckerAttempt:
        nonlocal called
        del argv, last_message, timeout
        called = True
        return CheckerAttempt(0, "", "")

    exit_code = execute_checker_request(request_path, invoke=invoke, diagnostic_stream=StringIO())

    assert exit_code is OperatorExitCode.PERMANENT_CONTRACT_REJECTION
    assert called is False
    assert "provider must be 'anthropic'" in _status(request_path)["detail"]


def _run_fable_success(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """Run one Fable-seat session through the fake Claude CLI; return its traces."""

    request_path, result = _request_and_result(tmp_path, model=FABLE)
    envelope = json.loads(request_path.read_text(encoding="utf-8"))
    observed: list[tuple[str, ...]] = []

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CheckerAttempt:
        observed.append(tuple(argv))
        del last_message, timeout
        return CheckerAttempt(0, _harness(result["items"]), "")

    exit_code = execute_checker_request(
        request_path, invoke=invoke, sleep=lambda _s: None, diagnostic_stream=StringIO()
    )
    assert exit_code is OperatorExitCode.SUCCESS
    return request_path, envelope, result, observed[0]


def test_fable_seat_runs_the_bench_proven_claude_argv(tmp_path: Path) -> None:
    _request_path, envelope, _result, argv = _run_fable_success(tmp_path)
    assert argv[0] == "claude" and argv[1] == "-p"
    assert argv[argv.index("--model") + 1] == FABLE
    for flag in ("--bare", "--json-schema", "--no-session-persistence", "--add-dir"):
        assert flag in argv
    assert argv[argv.index("--setting-sources") + 1] == "project"
    assert argv[argv.index("--output-format") + 1] == "json"
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert list(schema["properties"]) == ["items"]
    assert set(CLAUDE_CHECKER_ALLOWED_TOOLS) <= set(argv)
    assert set(CLAUDE_CHECKER_DISALLOWED_TOOLS) <= set(argv)
    roots = argv[argv.index("--add-dir") + 1 :]
    assert set(roots) == set(envelope["allowed_read_roots"])
    # Codex-only flags never leak onto the Claude seat.
    assert "--output-schema" not in argv and "exec" not in argv


def test_fable_seat_publishes_the_stamped_result_and_the_model_witness(tmp_path: Path) -> None:
    request_path, _envelope, result, _argv = _run_fable_success(tmp_path)
    published = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert published["checker"]["provider"] == "anthropic"
    assert published["checker"]["model"] == FABLE
    assert published["items"] == result["items"]
    events = _telemetry(request_path)
    attempt = next(event for event in events if event["event"] == "codex-attempt")
    assert attempt["provider"] == "anthropic" and attempt["model"] == FABLE
    # Input tokens are the billed prompt total (fresh + cache creation + cache read).
    assert attempt["input_tokens"] == 376 + 103856 + 346761
    assert attempt["output_tokens"] == 30193
    witness = next(event for event in events if event["event"] == "model-witness")
    assert witness["witness"]["seated_model"] == FABLE
    assert witness["witness"]["models_used"] == ["claude-fable-5-1", "claude-haiku-4-5-20251001"]


def test_fable_session_without_the_model_witness_is_refused_unpublished(tmp_path: Path) -> None:
    request_path, result = _request_and_result(tmp_path, model=FABLE)
    document = json.loads(_harness(result["items"]))
    document["modelUsage"] = {"claude-opus-5": {"inputTokens": 1, "outputTokens": 999}}

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CheckerAttempt:
        del argv, last_message, timeout
        return CheckerAttempt(0, json.dumps(document), "")

    exit_code = execute_checker_request(
        request_path, invoke=invoke, sleep=lambda _s: None, diagnostic_stream=StringIO()
    )

    assert exit_code is OperatorExitCode.PERMANENT_CONTRACT_REJECTION
    assert not (tmp_path / "result.json").exists()
    status = _status(request_path)
    assert status["classification"] == "permanent-contract-rejection"
    assert "modelUsage witness" in status["detail"]
    assert status["attempts"] == 1, "no re-ask: another model's verdict is not a slip"
    assert any(event["event"] == "model-witness-refused" for event in _telemetry(request_path))


def test_fable_structured_limit_pauses_with_the_harness_reset(tmp_path: Path) -> None:
    request_path, _result = _request_and_result(tmp_path, model=FABLE)
    document = {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "terminal_reason": "blocking_limit",
        "rate_limit_info": {
            "status": "rejected",
            "resetsAt": 1788400000,
            "rateLimitType": "seven_day",
        },
        "result": "You have hit your usage limit.",
    }

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CheckerAttempt:
        del argv, last_message, timeout
        return CheckerAttempt(1, json.dumps(document), "")

    exit_code = execute_checker_request(request_path, invoke=invoke, diagnostic_stream=StringIO())

    assert exit_code is OperatorExitCode.RATE_OR_QUOTA_PAUSE
    status = _status(request_path)
    assert status["classification"] == "rate-or-quota-pause"
    from datetime import datetime, timezone

    expected_reset = datetime.fromtimestamp(1788400000, tz=timezone.utc).isoformat()
    assert status["reset_at"] == expected_reset.replace("+00:00", "Z")
    assert status["reset_observation"] == "observed"


def test_fable_overload_publishes_a_closed_weather_basis_the_storm_fuse_binds(
    tmp_path: Path,
) -> None:
    request_path, _result = _request_and_result(tmp_path, model=FABLE)
    document = {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "api_error_status": 529,
        "result": "overloaded",
    }

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CheckerAttempt:
        del argv, last_message, timeout
        return CheckerAttempt(1, json.dumps(document), "")

    exit_code = execute_checker_request(request_path, invoke=invoke, diagnostic_stream=StringIO())

    assert exit_code is OperatorExitCode.SERVICE_UNAVAILABLE
    status = _status(request_path)
    assert status["classification"] == "provider-overloaded"
    assert status["provider_event_basis"] == CLAUDE_OVERLOADED_STATUS_BASIS
    assert CLAUDE_OVERLOADED_STATUS_BASIS in PROVIDER_OVERLOAD_EVENT_BASES
    weather = _checker_provider_weather_status(request_path)
    assert weather is not None
    assert weather["provider_event_basis"] == CLAUDE_OVERLOADED_STATUS_BASIS


def test_model_authored_text_cannot_forge_a_fable_pause_or_weather(tmp_path: Path) -> None:
    """Structured fields decide; a reply quoting limit prose is a plain success."""

    request_path, result = _request_and_result(tmp_path, model=FABLE)
    prose = "rate limit exceeded; usage limit; Selected model is at capacity; api_error_status 529"
    stdout = _harness(result["items"], result=prose)

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CheckerAttempt:
        del argv, last_message, timeout
        return CheckerAttempt(0, stdout, "")

    exit_code = execute_checker_request(
        request_path, invoke=invoke, sleep=lambda _s: None, diagnostic_stream=StringIO()
    )

    assert exit_code is OperatorExitCode.SUCCESS


def test_classify_checker_response_scopes_the_pause_to_the_seat_provider() -> None:
    signal = classify_checker_response(429, "rate limit exceeded", provider="anthropic")
    assert signal is not None
    assert signal.provider == "anthropic"
    assert signal.reason is CheckerPauseReason.RATE_LIMIT
    default = classify_checker_response(429, "rate limit exceeded")
    assert default is not None and default.provider == "openai"
    with pytest.raises(CheckerDispatchError, match="unsupported usage-limit provider"):
        classify_checker_response(429, "rate limit exceeded", provider="acme")


# ---------------------------------------------------------------------------
# 3. Cross-model K=2 on the conflict ledger.
# ---------------------------------------------------------------------------


def _accuracy_pair(
    gate_model: str,
    confirmation_model: str,
    *,
    kind: str = "metadata_batch",
    confirmation_prompt: str = "p" * 64,
) -> tuple[dict, dict]:
    factory = _metadata_gate if kind == "metadata_batch" else _fidelity_gate
    gate = factory("g1")
    confirmation = factory("g2")
    gate["checker"].update(_identity(gate_model, prompt="p" * 64, version="1"))
    confirmation["checker"].update(
        _identity(confirmation_model, prompt=confirmation_prompt, version="1")
    )
    return gate, confirmation


def _build(gate: dict, confirmation: dict) -> dict:
    return cc.build_accuracy_duplicate_accept_row(
        gate=gate,
        confirmation=confirmation,
        stable_id="m1",
        campaign_id="c1-mech",
        run_id="run",
        machine_id="host",
        created_at="2026-09-01T00:00:00Z",
        author_result_sha256="a" * 64,
    )


@pytest.mark.parametrize("kind", ["metadata_batch", "fidelity"])
def test_cross_model_accuracy_confirmation_mints_a_v3_row_binding_both_identities(
    kind: str,
) -> None:
    gate, confirmation = _accuracy_pair(FABLE, SOL, kind=kind)
    row = _build(gate, confirmation)
    assert row["schema_version"] == cc.CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V3
    assert row["status"] == "confirmed"
    assert row["confirmation_pairing"] == "cross-model"
    assert (
        row["first_identity"]["model"] == FABLE and row["first_identity"]["provider"] == "anthropic"
    )
    assert row["confirmation_identity"]["model"] == SOL
    validated = cc.validate_checker_verdict_conflict_row(row)
    assert validated == row
    assert cc.accuracy_duplicate_accept_state("g1", "m1", [row]) == "confirmed"
    # A terminal reader never picks up an accuracy settlement.
    assert cc.duplicate_accept_state("g1", [row]) is None


def test_two_fable_sessions_are_never_a_k2_pair() -> None:
    gate, confirmation = _accuracy_pair(FABLE, FABLE)
    with pytest.raises(cc.CheckerVerdictConflictError, match="never a cross-model K=2 pair"):
        _build(gate, confirmation)


def test_cross_model_confirmation_must_share_the_frozen_prompt_epoch() -> None:
    gate, confirmation = _accuracy_pair(FABLE, SOL, confirmation_prompt="q" * 64)
    with pytest.raises(cc.CheckerVerdictConflictError, match="different question"):
        _build(gate, confirmation)


def test_unseated_primary_keeps_the_same_identity_v2_row() -> None:
    gate, confirmation = _accuracy_pair(SOL, SOL)
    row = _build(gate, confirmation)
    assert row["schema_version"] == cc.CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V2
    assert "first_identity" not in row
    with pytest.raises(cc.CheckerVerdictConflictError, match="different question"):
        _build(*_accuracy_pair(SOL, FABLE))


def test_a_forged_v3_row_refuses_on_read() -> None:
    gate, confirmation = _accuracy_pair(FABLE, SOL)
    row = _build(gate, confirmation)
    forged = dict(row)
    forged["first_identity"] = dict(row["confirmation_identity"])
    forged.pop("row_sha256")
    forged["row_sha256"] = cc.row_content_address(forged)
    with pytest.raises(cc.CheckerVerdictConflictError, match="confirmation_pairing must re-derive"):
        cc.validate_checker_verdict_conflict_row(forged)


def test_cross_model_terminal_confirmation_mints_v3_and_settles() -> None:
    gate = _td_gate("accepted", gate_id="gate-first", model=FABLE)
    gate["checker"]["provider"] = "anthropic"
    confirmation = _td_gate(
        "accepted", gate_id="gate-confirm", model=SOL, gate_identity="sha256:" + "9" * 64
    )
    assert cc.require_confirmation_pairing(gate, confirmation) == "cross-model"
    row = cc.build_duplicate_accept_row(
        stable_id="m4976",
        work_id="work-m4976",
        campaign_id="c1-mech",
        run_id="run",
        machine_id="host",
        created_at="2026-09-01T00:00:00Z",
        gate_kind="terminal_disposition",
        author_result_sha256="a" * 64,
        gate_id="gate-first",
        gate_identity=gate["gate_identity"],
        confirmation_gate_identity=confirmation["gate_identity"],
        confirmation_verdict="accepted",
        confirmation_identity=cc.adjudication_identity_fields(confirmation),
        confirmation_result_envelope_sha256="sha256:" + "f" * 64,
        first_identity=cc.adjudication_identity_fields(gate),
    )
    assert row["schema_version"] == cc.CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V3
    assert row["status"] == "confirmed"
    assert cc.validate_checker_verdict_conflict_row(row) == row
    assert cc.duplicate_accept_state("gate-first", [row]) == "confirmed"
    assert cc.accuracy_duplicate_accept_state("gate-first", "m4976", [row]) is None
    # Legacy call shape (no first identity) still mints the v1 row.
    legacy = cc.build_duplicate_accept_row(
        stable_id="m4976",
        work_id="work-m4976",
        campaign_id="c1-mech",
        run_id="run",
        machine_id="host",
        created_at="2026-09-01T00:00:00Z",
        gate_kind="terminal_disposition",
        author_result_sha256="a" * 64,
        gate_id="gate-first",
        gate_identity=gate["gate_identity"],
        confirmation_gate_identity=confirmation["gate_identity"],
        confirmation_verdict="accepted",
        confirmation_identity=_identity(SOL),
        confirmation_result_envelope_sha256="sha256:" + "f" * 64,
    )
    assert legacy["schema_version"] == cc.CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 4. The terminal 10 percent sample.
# ---------------------------------------------------------------------------


def test_terminal_sample_is_deterministic_and_near_ten_percent() -> None:
    sha = "sha256:" + "a" * 64
    draws = [terminal_confirmation_sampled(f"m{i}", sha, rate=0.10) for i in range(20000)]
    fraction = sum(draws) / len(draws)
    assert 0.085 < fraction < 0.115, fraction
    assert draws == [terminal_confirmation_sampled(f"m{i}", sha, rate=0.10) for i in range(20000)]
    assert terminal_confirmation_sampled("m1", sha, rate=1.0) is True
    assert terminal_confirmation_sampled("m1", sha, rate=0.0) is False
    # The content is part of the fact: a new author result re-draws.
    assert any(
        terminal_confirmation_sampled("m1", "sha256:" + f"{i:064x}", rate=0.10) for i in range(200)
    )


def test_confirmation_sampled_follows_the_seat_table() -> None:
    accuracy = _metadata_gate("g1")
    accuracy["checker"].update(_identity(FABLE))
    assert cc.confirmation_sampled(accuracy, stable_id="m1") is True
    legacy = _td_gate("accepted", model="codex")
    assert cc.confirmation_sampled(legacy, stable_id="m4976") is True
    fable_terminal = _td_gate("accepted", model=FABLE)
    fable_terminal["checker"]["provider"] = "anthropic"
    expected = terminal_confirmation_sampled(
        "m4976",
        fable_terminal["items"][0]["terminal_disposition"]["author_result_sha256"],
        rate=TERMINAL_CONFIRMATION_SAMPLE_RATE,
    )
    assert cc.confirmation_sampled(fable_terminal, stable_id="m4976") is expected
    # The resume guard never re-buys an unsampled accept.
    unsampled = next(
        gate
        for gate in (
            _td_gate("accepted", gate_id=f"g{i}", sha="sha256:" + f"{i:064x}", model=FABLE)
            for i in range(400)
        )
        if not terminal_confirmation_sampled(
            "m4976", gate["items"][0]["terminal_disposition"]["author_result_sha256"], rate=0.10
        )
    )
    unsampled["checker"]["provider"] = "anthropic"
    from menagerie.crawler.gate_reuse import CheckerAdjudicationIdentity

    identity = CheckerAdjudicationIdentity(**cc.adjudication_identity_fields(unsampled))
    assert (
        cc.unconfirmed_accepted_terminal_gate(
            [unsampled],
            [],
            stable_id="m4976",
            work_id="work-m4976",
            author_result_sha256=unsampled["items"][0]["terminal_disposition"][
                "author_result_sha256"
            ],
            checker_identity=identity,
        )
        is None
    )


# ---------------------------------------------------------------------------
# 5. The command lane declares both seats; other lanes confirm as themselves.
# ---------------------------------------------------------------------------


def test_command_lane_declares_the_fable_primary_and_the_sol_confirmation() -> None:
    lane = CommandCheckerLane(["python", "-m", "menagerie.crawler.operator_checker"])
    # A campaign binding that disagrees with the seat the lane seals refuses
    # before any session (the accepted-identity recompute would otherwise
    # refuse every award as stale).
    with pytest.raises(DriverIntegrationError, match="not the seated metadata_batch primary"):
        lane.adjudication_identity("metadata_batch", DriverConfig(checker_model=SOL))
    config = DriverConfig(checker_model=FABLE)
    for kind in GateKind:
        primary = lane.adjudication_identity(kind.value, config)
        confirmation = lane.confirmation_identity(kind.value, config)
        assert (primary.provider, primary.model) == ("anthropic", FABLE)
        assert (confirmation.provider, confirmation.model) == ("openai", SOL)
        assert primary.prompt_sha256 == confirmation.prompt_sha256
        assert lane_confirmation_identity(lane, kind.value, config) == confirmation

    class LegacyLane:
        def adjudication_identity(self, gate_kind: str, config: DriverConfig):  # noqa: ANN202
            from menagerie.crawler.gate_reuse import CheckerAdjudicationIdentity

            return CheckerAdjudicationIdentity("openai", "codex", "1", PROMPT)

        def check_terminal(self, artifact, work_root, config):  # noqa: ANN001, ANN202
            return ("terminal", artifact)

    legacy = LegacyLane()
    assert lane_confirmation_identity(legacy, "fidelity", config).model == "codex"
    assert buy_confirmation_session(
        legacy, "terminal_disposition", ("art",), Path("/w"), config
    ) == (
        "terminal",
        "art",
    )


# ---------------------------------------------------------------------------
# 6. Activation matrix and toolchain identity.
# ---------------------------------------------------------------------------


def test_activation_matrix_activates_the_fable_seat_without_a_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    monkeypatch.delenv(CHECKER_ACTIVATION_ENV, raising=False)
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, "/nonexistent/receipt.json")
    rows = load_checker_activations()
    assert {row["gate_kind"] for row in rows} == {kind.value for kind in GateKind}
    assert all(row["checker_model"] == FABLE for row in rows)
    assert all(row["checker_prompt_identity"] == live_checker_prompt_identity() for row in rows)
    assert all(row["model_witnessed"] == row["sessions_ok"] >= 6 for row in rows)
    for kind in GateKind:
        assert checker_activation_for(kind.value, FABLE) is not None
        assert checker_activation_for(kind.value, SOL) is None
    result = evaluate_checker_calibration_mode(
        renders_gate_verdicts=True,
        pinned_models={kind.value: FABLE for kind in GateKind},
    )
    assert result["ok"] is True and result["refusals"] == []
    assert all(row["receipt"] == "activated" for row in result["lanes"])


def test_activation_at_another_epoch_activates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = json.loads(Path("menagerie/crawler/assets/checker-activation.json").read_text())
    for row in stale["activations"]:
        row["checker_prompt_identity"] = "sha256:" + "0" * 64
    path = tmp_path / "activation.json"
    path.write_text(json.dumps(stale))
    monkeypatch.setenv(CHECKER_ACTIVATION_ENV, str(path))
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, "/nonexistent/receipt.json")
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    assert checker_activation_for("fidelity", FABLE) is None
    result = evaluate_checker_calibration_mode(
        renders_gate_verdicts=True, pinned_models={"fidelity": FABLE}
    )
    assert result["ok"] is False
    assert result["lanes"][0]["receipt"] == "refused"


def test_toolchain_identity_carries_the_seat_table_and_the_checker_claude_pin() -> None:
    assert verify_toolchain_attestation() == []
    attested = json.loads(Path("menagerie/crawler/toolchain_attestation.json").read_text())
    assert attested["toolchain_identity"] == toolchain_identity()
    assert attested["checker_claude_cli_version"] == CHECKER_CLAUDE_CLI_VERSION == "2.1.257"
    assert checker_toolchain_lane("anthropic") == "checker-anthropic"
    assert checker_toolchain_lane("openai") == "checker"
    assert expected_cli_version("checker-anthropic") == CHECKER_CLAUDE_CLI_VERSION
