"""T4-T3 provider-weather classifier arms: auth, 429, unmatched-nonzero, billing.

The pilot's checker wall is ~15 hours per leg on OAuth-authenticated Codex, so
provider weather -- an expired login, a 429 storm -- is a daily operating
condition, not an edge case. Before these arms landed, an expired-login capture
(rc != 0 with no ``thread.started`` marker) and a bare "rate limit exceeded"
response both fell through ``classify_codex_attempt`` to PERMANENT, which exits
64 and durably parks every touched model as ``permanent-contract-rejection``:
wrong-blame parks for an outage no model caused.

The arms under test:

* a typed AUTH classification that pauses the LANE like quota, never a
  per-model park;
* "rate limit" text joins the quota/pause family (429 weather);
* an unmatched nonzero exit defaults to bounded-retry TRANSIENT, never
  PERMANENT;
* a consecutive-parks tripwire that stops grinding a dead checker lane one
  parked model at a time;
* the author-side billing arm: "credit balance" is a BILLING failure with a
  sparse operator-recheck, not a quota window that pretends to reset in an
  hour.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pytest

from menagerie.crawler.author_dispatch import (
    AUTHOR_EXIT_BACKOFF,
    AuthorBackoffSignal,
    classify_author_response,
)
from menagerie.crawler.checker_dispatch import (
    CheckerBackoffSignal,
    CheckerPartialResultError,
    classify_checker_response,
)
from menagerie.crawler.constants import (
    CHECKER_LANE_PARK_STREAK_LIMIT,
    METADATA_BATCH_MAX,
    AuthorPauseReason,
    CheckerPauseReason,
)
from menagerie.crawler.driver import DriverConfig
from menagerie.crawler.driver_contracts import (
    AuthorBackoffError,
    CheckerOutcome,
    RetryableOperatorError,
)
from menagerie.crawler.driver_progress import _future_reset
from menagerie.crawler.operator_checker import (
    CHECKER_MAX_ATTEMPTS,
    CodexAttempt,
    FailureKind,
    classify_codex_attempt,
    execute_checker_request,
)
from menagerie.crawler.operator_protocol import OperatorExitCode
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    FakePauseScheduler,
    FidelityAuthor,
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)

# ---------------------------------------------------------------------------
# wrapper classification: classify_codex_attempt
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_expired_login_classifies_auth_not_permanent() -> None:
    """An expired-login capture is provider weather, never a contract rejection.

    The exact T4-T3 capture shape: nonzero exit, no ``thread.started`` marker,
    an auth complaint on stderr. PERMANENT here exits 64 and durably parks
    every batch member as ``permanent-contract-rejection`` -- wrong blame for
    an OAuth expiry that a ``codex login`` fixes.
    """

    attempt = CodexAttempt(
        returncode=1,
        stdout="",
        stderr="Not logged in. Run `codex login` to authenticate.",
    )
    assert classify_codex_attempt(attempt).kind is FailureKind.AUTH


@pytest.mark.smoke
def test_auth_texts_classify_auth_for_every_observed_spelling() -> None:
    """Every observed auth spelling routes to the typed AUTH arm."""

    for text in (
        "Not logged in",
        "401 Unauthorized",
        "authentication failed: token expired",
        "invalid API key provided",
        "please run `codex login` first",
    ):
        attempt = CodexAttempt(returncode=1, stdout="", stderr=text)
        assert classify_codex_attempt(attempt).kind is FailureKind.AUTH, text


@pytest.mark.smoke
def test_rate_limit_exceeded_classifies_as_pause_not_permanent() -> None:
    """Bare "rate limit exceeded" (no "quota", no 5xx digit) is a pause.

    At HEAD this fell through to PERMANENT when no started marker was present:
    a 429 storm parked models as contract rejections.
    """

    attempt = CodexAttempt(returncode=1, stdout="", stderr="Rate limit exceeded")
    assert classify_codex_attempt(attempt).kind is FailureKind.QUOTA


@pytest.mark.smoke
def test_unmatched_nonzero_defaults_to_bounded_retry_transient() -> None:
    """A nonzero exit no arm recognizes is TRANSIENT, never PERMANENT.

    PERMANENT is reserved for positively identified contract rejections. An
    unrecognized failure gets the bounded retry ladder; if it persists, the
    wrapper still exits retryable and the lane's own bounded budget decides.
    """

    attempt = CodexAttempt(
        returncode=1,
        stdout="",
        stderr="some entirely novel failure text the classifier has never seen",
    )
    assert classify_codex_attempt(attempt).kind is FailureKind.TRANSIENT


@pytest.mark.smoke
def test_explicit_permanent_markers_still_classify_permanent() -> None:
    """The TRANSIENT default must not absorb positively identified rejections."""

    attempt = CodexAttempt(
        returncode=1,
        stdout="",
        stderr="invalid_request_error: model is not supported",
    )
    assert classify_codex_attempt(attempt).kind is FailureKind.PERMANENT
    assert (
        classify_codex_attempt(CodexAttempt(returncode=0, stdout="", stderr="")).kind
        is FailureKind.SUCCESS
    )


# ---------------------------------------------------------------------------
# wrapper execution: execute_checker_request routing
# ---------------------------------------------------------------------------


def _write_minimal_request(tmp_path: Path) -> Path:
    """Publish one schema-valid metadata checker request envelope."""

    from menagerie.crawler.tests.test_operator_checker import _request_and_result

    request_path, _candidate = _request_and_result(tmp_path)
    return request_path


def _status_sidecar(request_path: Path) -> dict[str, Any]:
    """Read the wrapper's structured status sidecar."""

    from menagerie.crawler.operator_protocol import status_sidecar_path

    return json.loads(status_sidecar_path(request_path).read_text(encoding="utf-8"))


@pytest.mark.smoke
def test_wrapper_exits_pause_on_auth_failure_without_burning_attempts(
    tmp_path: Path,
) -> None:
    """An auth failure exits the quota/pause code with an ``auth-failure`` status.

    Exactly one attempt is spent: retrying into a dead login buys nothing, and
    exit 64 (the old fallthrough) would park every batch member durably.
    """

    calls: list[Sequence[str]] = []

    def invoke(argv: Sequence[str], last_message: Path, budget: float) -> CodexAttempt:
        calls.append(tuple(argv))
        return CodexAttempt(
            returncode=1,
            stdout="",
            stderr="Not logged in. Run `codex login` to authenticate.",
        )

    request_path = _write_minimal_request(tmp_path)
    exit_code = execute_checker_request(request_path, invoke=invoke, sleep=lambda _s: None)

    assert exit_code is OperatorExitCode.RATE_OR_QUOTA_PAUSE
    assert len(calls) == 1
    status = _status_sidecar(request_path)
    assert status["classification"] == "auth-failure"


@pytest.mark.smoke
def test_wrapper_retries_unmatched_nonzero_and_exits_retryable(tmp_path: Path) -> None:
    """An unrecognized nonzero exit rides the bounded ladder and exits retryable."""

    calls: list[int] = []

    def invoke(argv: Sequence[str], last_message: Path, budget: float) -> CodexAttempt:
        calls.append(1)
        return CodexAttempt(returncode=1, stdout="", stderr="novel unclassified failure")

    request_path = _write_minimal_request(tmp_path)
    exit_code = execute_checker_request(request_path, invoke=invoke, sleep=lambda _s: None)

    assert exit_code is OperatorExitCode.RETRYABLE_INFRASTRUCTURE
    assert len(calls) == CHECKER_MAX_ATTEMPTS
    assert _status_sidecar(request_path)["classification"] == "retryable-infrastructure"


# ---------------------------------------------------------------------------
# driver-side classification: classify_checker_response
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_driver_classifies_auth_text_as_lane_pause() -> None:
    """Auth text on a failed checker command is a typed lane pause signal."""

    signal = classify_checker_response(
        int(OperatorExitCode.PERMANENT_CONTRACT_REJECTION),
        "checker command failed: Not logged in. Run `codex login`.",
    )
    assert isinstance(signal, CheckerBackoffSignal)
    assert signal.reason is CheckerPauseReason.AUTH_FAILURE
    assert signal.provider == "openai"


@pytest.mark.smoke
def test_driver_auth_arm_never_fires_on_a_successful_exit() -> None:
    """rc == 0 published a validated gate; text can never turn it into a pause."""

    assert classify_checker_response(0, "transcript mentions 'not logged in' in prose") is None


@pytest.mark.smoke
def test_typed_pause_exit_with_unmatched_text_still_pauses() -> None:
    """Exit 76 is the wrapper's own typed pause declaration; text cannot demote it.

    Without this fallback an exit-76 whose tail matched no marker fell through
    to ``_raise_for_checker_exit``'s generic arm and became a batch failure --
    the exact wrong-blame park path the typed exit exists to prevent.
    """

    signal = classify_checker_response(
        int(OperatorExitCode.RATE_OR_QUOTA_PAUSE), "entirely unrecognized tail"
    )
    assert isinstance(signal, CheckerBackoffSignal)
    assert signal.reason is CheckerPauseReason.QUOTA_EXHAUSTED


@pytest.mark.smoke
def test_auth_pause_reaches_the_driver_as_a_lane_pause_not_a_park(tmp_path: Path) -> None:
    """An auth-failing checker pauses the campaign; no model parks or terminalizes."""

    class AuthWeatherChecker(ScriptedChecker):
        """Report the typed auth pause for every metadata batch."""

        def check_metadata(
            self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
        ) -> CheckerOutcome:
            del artifacts, work_root, config
            return CheckerOutcome(
                backoff=CheckerBackoffSignal(
                    CheckerPauseReason.AUTH_FAILURE,
                    None,
                    None,
                    "Not logged in. Run `codex login`.",
                )
            )

    snapshot = _snapshot(tmp_path, count=4)
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)
    result = _driver(
        tmp_path,
        snapshot,
        checker=AuthWeatherChecker(),
        pause_scheduler=scheduler,
    ).run()

    assert result.status == "paused:usage-limit"
    assert result.paused_reason == CheckerPauseReason.AUTH_FAILURE.value
    assert scheduler.calls == 1
    # No model was blamed: no gate was minted, nothing parked or terminalized.
    # (Execution precedes gating, so models legitimately sit at runs/pending.)
    assert scan_jsonl(paths.ledgers.gates) == []
    pending = scan_jsonl(paths.ledgers.models)
    assert {model["status"]["code"] for model in pending} <= {"runs"}
    assert all(model["authored_metadata_state"] == "pending" for model in pending)


# ---------------------------------------------------------------------------
# consecutive-parks tripwire
# ---------------------------------------------------------------------------


class AlwaysFailingLaneChecker(ScriptedChecker):
    """Fail every metadata batch with a lane-owned partial-result refusal."""

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        del work_root, config
        raise CheckerPartialResultError(
            f"checker result is partial or has extra items ({len(artifacts)} items)"
        )


@pytest.mark.smoke
def test_park_streak_limit_is_strictly_wider_than_one_full_batch() -> None:
    """One poisoned width-5 batch may park all five members by itself.

    The limit proves two things at once (constants.py): strictly more
    consecutive parks than one poisoned batch can produce by itself, AND the
    historical six-distinct-model evidence mass -- so at the width-1 config
    the tripwire does not collapse to reading two adjacent poisoned models
    as lane weather.
    """

    assert CHECKER_LANE_PARK_STREAK_LIMIT == max(6, METADATA_BATCH_MAX + 1)
    assert CHECKER_LANE_PARK_STREAK_LIMIT > METADATA_BATCH_MAX


@pytest.mark.heavy
def test_consecutive_parks_across_batches_trip_the_lane_loudly(tmp_path: Path) -> None:
    """Parks past one full batch width stop the run instead of grinding the leg.

    Eight models make two independent original batches (5 + 3). A checker lane
    that fails every batch parks the first five (one full batch: still a
    legitimate parked outcome), and the sixth consecutive park -- necessarily
    from the second batch -- is cross-batch evidence of a dead lane. The run
    goes LOUD and retryable; it must not silently convert the remaining leg
    into wrong-blame parks one bought session at a time.
    """

    snapshot = _snapshot(tmp_path, count=8)
    paths = _paths(tmp_path, snapshot)

    with pytest.raises(RetryableOperatorError, match="consecutive"):
        _driver(
            tmp_path, snapshot, author=FidelityAuthor(), checker=AlwaysFailingLaneChecker()
        ).run()

    # Nothing terminalized: parks are supersedable, the campaign is resumable.
    assert scan_jsonl(paths.ledgers.models) == []
    state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    assert state["status"] == "retryable:infrastructure"


@pytest.mark.heavy
def test_one_full_batch_of_parks_still_completes_with_deferrals(tmp_path: Path) -> None:
    """Parks bounded by one batch width keep the W4-A parked outcome exactly."""

    snapshot = _snapshot(tmp_path, count=5)
    paths = _paths(tmp_path, snapshot)

    result = _driver(
        tmp_path, snapshot, author=FidelityAuthor(), checker=AlwaysFailingLaneChecker()
    ).run()

    assert result.status == "complete-with-deferrals"
    assert scan_jsonl(paths.ledgers.models) == []


# ---------------------------------------------------------------------------
# author-side billing arm
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_credit_balance_classifies_as_billing_failure_not_quota() -> None:
    """A billing failure is not a usage window: it never resets on its own."""

    body = json.dumps(
        {
            "is_error": True,
            "result": "Your credit balance is too low to access the Anthropic API.",
        }
    )
    signal = classify_author_response(AUTHOR_EXIT_BACKOFF, body)
    assert isinstance(signal, AuthorBackoffSignal)
    assert signal.reason is AuthorPauseReason.BILLING_FAILURE
    assert signal.reset_at is None


@pytest.mark.smoke
def test_usage_limit_still_classifies_quota_exhausted() -> None:
    """The billing arm must not absorb the ordinary usage-window pause."""

    body = json.dumps(
        {
            "is_error": True,
            "result": "Claude usage limit reached.",
            "retry_after_seconds": 73,
        }
    )
    signal = classify_author_response(AUTHOR_EXIT_BACKOFF, body)
    assert isinstance(signal, AuthorBackoffSignal)
    assert signal.reason is AuthorPauseReason.QUOTA_EXHAUSTED
    assert signal.retry_after_seconds == 73
    assert _future_reset("2026-08-11T12:00:00Z", signal) == "2026-08-11T12:01:13Z"


@pytest.mark.smoke
def test_billing_recheck_is_sparse_never_an_hourly_guess() -> None:
    """A billing pause guesses a sparse operator recheck, not a quota hour.

    Money does not refill on a window: an hourly wake into a dead billing
    account buys one failed session per hour, forever. The recheck exists only
    to catch an operator refill.
    """

    now = "2026-08-11T12:00:00Z"
    billing = AuthorBackoffSignal(
        reason=AuthorPauseReason.BILLING_FAILURE,
        retry_after_seconds=None,
        reset_at=None,
        response_excerpt="credit balance is too low",
    )
    quota = AuthorBackoffSignal(
        reason=AuthorPauseReason.QUOTA_EXHAUSTED,
        retry_after_seconds=None,
        reset_at=None,
        response_excerpt="usage limit reached",
    )
    billing_reset = datetime.fromisoformat(_future_reset(now, billing).replace("Z", "+00:00"))
    quota_reset = datetime.fromisoformat(_future_reset(now, quota).replace("Z", "+00:00"))
    start = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    assert quota_reset == start.replace(hour=13)
    assert billing_reset - start >= (quota_reset - start) * 6


@pytest.mark.smoke
def test_billing_pause_pauses_the_campaign_with_its_own_reason(tmp_path: Path) -> None:
    """A billing failure pauses with the billing reason, never a quota disguise."""

    class BillingAuthor(FakeAuthor):
        """Raise the typed billing backoff for every model."""

        def author(self, item: Any, work_root: Path, config: DriverConfig, context: Any) -> Any:
            del work_root, config, context
            raise AuthorBackoffError(
                AuthorBackoffSignal(
                    reason=AuthorPauseReason.BILLING_FAILURE,
                    retry_after_seconds=None,
                    reset_at=None,
                    response_excerpt="Your credit balance is too low",
                    provider="anthropic",
                )
            )

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)
    result = _driver(
        tmp_path,
        snapshot,
        author=BillingAuthor(),
        pause_scheduler=scheduler,
    ).run()

    assert result.status == "paused:usage-limit"
    assert result.paused_reason == AuthorPauseReason.BILLING_FAILURE.value
    assert scheduler.calls == 1
    assert scan_jsonl(paths.ledgers.models) == []
