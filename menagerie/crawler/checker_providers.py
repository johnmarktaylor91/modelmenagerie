"""Provider-neutral checker session adapters (DECIDED_PLAN item 32).

The checker wrapper (:mod:`menagerie.crawler.operator_checker`) used to be a
Codex wrapper: ``checker.provider`` was the literal ``"openai"`` at envelope
build, refused otherwise at wrapper preflight, and the argv, the failure
classifier, the token accounting, and the final-message transport were all
Codex-shaped. The 2026-09-01 seat ruling (``SEAT_RULING.md``, tl-checkerbench)
seats ``claude-fable-5-1`` as the PRIMARY checker on every gate kind with
``gpt-5.6-sol`` as the cross-model K=2 confirmation seat, so the wrapper now
selects ONE adapter per session from the provider the envelope's seated model
DERIVES (:func:`menagerie.crawler.constants.checker_provider_for_model`).

Both adapters honor the SAME contract the wrapper enforces around them:

* the frozen prompt + ``_build_prompt`` dynamic region is the whole prompt;
* the native output schema derived from ``gate.v3`` is enforced by the
  provider (``--output-schema`` on Codex, ``--json-schema`` on the Claude CLI);
* the result is the checker-authored ``items`` fragment and nothing else --
  every identity is stamped by the wrapper from the envelope, never believed;
* the session runs in the wrapper's process group with the paid-session
  custody descriptors inherited, so custody, wall accounting, and the storm
  fuse see a Fable session exactly as they see a sol session (the org key
  being free is an ACCOUNTING note, never a code branch);
* every classification reads STRUCTURED provider output first (Codex JSONL
  error events; the Claude harness JSON's ``terminal_reason`` /
  ``rate_limit_info`` / ``api_error_status`` / ``error.type``), and free text
  only for the auth arm gated on a failed exit (T4-T3).

The Anthropic adapter additionally REQUIRES the harness ``modelUsage`` witness:
the seated model must have produced output tokens in the session, or the
result is refused typed before publication (``CheckerModelWitnessError``). The
reply text is never the model witness (fable-as-checker recipe).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.checker_dispatch import CheckerDispatchError, checker_usage_tokens
from menagerie.crawler.constants import (
    CHECKER_PROVIDER_ANTHROPIC,
    CHECKER_PROVIDER_OPENAI,
    checker_provider_for_model,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.native_output_schema import NativeOutputSchemaError, decode_native_result
from menagerie.crawler.operator_protocol import OPERATOR_REASONING_EFFORT

__all__ = [
    "AttemptClassification",
    "CheckerAttempt",
    "CheckerModelWitnessError",
    "CheckerProviderAdapter",
    "ClaudeCheckerAdapter",
    "CodexAttempt",
    "CodexCheckerAdapter",
    "FailureKind",
    "adapter_for_model",
    "adapter_for_provider",
    "build_claude_argv",
    "build_codex_argv",
    "classify_claude_attempt",
    "classify_codex_attempt",
    "parse_claude_harness",
]

CHECKER_EFFORT = OPERATOR_REASONING_EFFORT

_MODEL_METADATA_ERROR = "model metadata for"
_QUOTA_MARKERS = ("quota", "usage limit")
#: Codex reports both windows as ``rate limit`` text; ``retry after`` and the
#: bare 429 phrase are the transport variants the pilot saw.
_RATE_MARKERS = ("rate limit", "too many requests", "retry after")
#: Credential weather (T4-T3): read ONLY on a failed exit, never from a
#: successful gate's diagnostics.
_AUTH_MARKERS = (
    "not logged in",
    "codex login",
    "401 unauthorized",
    "unauthorized",
    "authentication failed",
    "token expired",
    "invalid api key",
    "please run /login",
)
_TRANSIENT_MARKERS = (
    "reconnecting...",
    "stream disconnected",
    "error sending request",
)
_PERMANENT_MARKERS = (
    "invalid_request_error",
    "model is not supported",
    "failed to read output schema file",
)
_STARTED_MARKERS = ('"type":"thread.started"', '"type":"turn.started"')
_RESET_PATTERN = re.compile(r"try again at\s+([^\r\n]+?)\.", re.IGNORECASE)
_SERVER_ERROR_PATTERN = re.compile(r"(?:^|[^0-9])5[0-9]{2}(?:[^0-9]|$)")
_MODEL_AT_CAPACITY_MESSAGE = "Selected model is at capacity. Please try a different model."
_MODEL_AT_CAPACITY_BASIS = "codex-event:error:model-at-capacity"

#: Closed provider-overload event bases the driver's storm fuse binds
#: (``driver_admission._checker_provider_weather_status``). Every member is a
#: MACHINE-observed structured event, never model-authored text.
CLAUDE_OVERLOADED_STATUS_BASIS = "claude-result:api_error_status:529"
CLAUDE_OVERLOADED_ERROR_BASIS = "claude-result:error:overloaded_error"
PROVIDER_OVERLOAD_EVENT_BASES = frozenset(
    {_MODEL_AT_CAPACITY_BASIS, CLAUDE_OVERLOADED_STATUS_BASIS, CLAUDE_OVERLOADED_ERROR_BASIS}
)

#: The Claude CLI read-only tool policy the bench ran under (bench.py, arms A-C):
#: read verbs plus hashing/inspection shell builtins; every writer, web, and
#: subagent tool denied. The wrapper publishes result.json; the model never
#: writes.
CLAUDE_CHECKER_ALLOWED_TOOLS: tuple[str, ...] = (
    "Read",
    "Grep",
    "Glob",
    "Bash(shasum:*)",
    "Bash(sha256sum:*)",
    "Bash(cat:*)",
    "Bash(head:*)",
    "Bash(tail:*)",
    "Bash(sed:*)",
    "Bash(wc:*)",
    "Bash(ls:*)",
    "Bash(nl:*)",
    "Bash(grep:*)",
    "Bash(find:*)",
    "Bash(awk:*)",
    "Bash(cut:*)",
)
CLAUDE_CHECKER_DISALLOWED_TOOLS: tuple[str, ...] = (
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "Task",
    "Agent",
    "TodoWrite",
)
#: Harness subtype the Claude CLI emits when the model could not satisfy the
#: ``--json-schema`` constraint within the harness's own retries: a positively
#: identified contract rejection (the schema IS the contract), never weather.
_CLAUDE_STRUCTURED_OUTPUT_EXHAUSTED = "error_max_structured_output_retries"
_OVERLOADED_HTTP_STATUS = 529
_OVERLOADED_ERROR_TYPE = "overloaded_error"


class FailureKind(str, Enum):
    """Closed wrapper classification of one checker attempt."""

    SUCCESS = "success"
    TRANSIENT = "retryable-infrastructure"
    PERMANENT = "permanent-contract-rejection"
    QUOTA = "rate-or-quota-pause"
    UNAVAILABLE = "service-unavailable"
    PROVIDER_OVERLOADED = "provider-overloaded"
    # Credential weather: the lane pauses exactly like quota, but the status names
    # re-authentication as the remedy instead of a reset window.
    AUTH = "auth-failure"


@dataclass(frozen=True)
class CheckerAttempt:
    """Captured result of one process-group-contained checker CLI invocation."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    unavailable: bool = False


#: Historical name; the wrapper's injectable ``invoke`` boundary and every
#: existing fault test spell it this way.
CodexAttempt = CheckerAttempt


@dataclass(frozen=True)
class AttemptClassification:
    """Classification and optional reset facts for one checker attempt."""

    kind: FailureKind
    reset_text: str | None = None
    provider_event_basis: str | None = None


class CheckerModelWitnessError(CheckerDispatchError):
    """The session's machine witness does not name the seated model.

    Raised by the Anthropic adapter when the harness ``modelUsage`` block does
    not show the seated model producing output tokens. A verdict without its
    model witness is refused before publication -- the flip-back condition (c)
    of the seat ruling is exactly "a session adjudicated by another model".
    """


# ---------------------------------------------------------------------------
# Codex (openai) adapter
# ---------------------------------------------------------------------------


def classify_codex_attempt(attempt: CheckerAttempt) -> AttemptClassification:
    """Classify one Codex result by merged message content, not exit code alone.

    Complexity note (cc-delta): this ladder is MOVED verbatim from
    ``operator_checker`` (item 32 adapter split), not new logic. Its
    cyclomatic weight is the essential, ORDERED precedence of the closed
    failure vocabulary (overload event > model-metadata > quota/rate > auth on
    a failed exit > transient > 5xx > permanent > unavailable > success >
    unstarted-nonzero); collapsing arms into a table would hide that order,
    which is exactly what T4-T3 debugged.
    """

    failure_text = _codex_failure_text(attempt.stdout, attempt.stderr)
    lowered = failure_text.lower()
    reset_match = _RESET_PATTERN.search(failure_text)
    reset_text = reset_match.group(1).strip(" \\\"'") if reset_match is not None else None
    provider_event_basis = _provider_overload_event_basis(attempt.stdout)
    if provider_event_basis is not None:
        return AttemptClassification(
            FailureKind.PROVIDER_OVERLOADED,
            provider_event_basis=provider_event_basis,
        )
    if _MODEL_METADATA_ERROR in lowered and "not found" in lowered:
        return AttemptClassification(FailureKind.PERMANENT)
    if any(marker in lowered for marker in _QUOTA_MARKERS) or any(
        marker in lowered for marker in _RATE_MARKERS
    ):
        return AttemptClassification(FailureKind.QUOTA, reset_text)
    if attempt.returncode != 0 and any(marker in lowered for marker in _AUTH_MARKERS):
        return AttemptClassification(FailureKind.AUTH)
    if attempt.timed_out or any(marker in lowered for marker in _TRANSIENT_MARKERS):
        return AttemptClassification(FailureKind.TRANSIENT)
    if _SERVER_ERROR_PATTERN.search(lowered):
        return AttemptClassification(FailureKind.TRANSIENT)
    if any(marker in lowered for marker in _PERMANENT_MARKERS):
        return AttemptClassification(FailureKind.PERMANENT)
    if attempt.unavailable:
        return AttemptClassification(FailureKind.UNAVAILABLE)
    if attempt.returncode == 0:
        return AttemptClassification(FailureKind.SUCCESS)
    compact_stdout = attempt.stdout.lower().replace(" ", "")
    if not any(marker in compact_stdout for marker in _STARTED_MARKERS):
        # T4-T3: PERMANENT here durably parked every batch member on any failure
        # the marker tables had never seen -- an expired login was the live case.
        # An unrecognized nonzero exit earns the bounded retry ladder instead;
        # PERMANENT stays reserved for positively identified contract rejections.
        return AttemptClassification(FailureKind.TRANSIENT)
    return AttemptClassification(FailureKind.UNAVAILABLE)


def _provider_overload_event_basis(stdout: str) -> str | None:
    """Return the exact Codex machine-event basis for provider overload, if any."""

    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, Mapping) or event.get("type") != "error":
            continue
        if event.get("message") == _MODEL_AT_CAPACITY_MESSAGE:
            return _MODEL_AT_CAPACITY_BASIS
    return None


def _codex_failure_text(stdout: str, stderr: str) -> str:
    """Extract only wrapper-visible Codex errors from a JSONL event stream."""

    messages = [stderr] if stderr else []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            messages.append(line)
            continue
        if not isinstance(event, Mapping):
            continue
        event_type = event.get("type")
        if event_type == "error" and isinstance(event.get("message"), str):
            messages.append(str(event["message"]))
        elif event_type == "turn.failed":
            error = event.get("error")
            if isinstance(error, Mapping) and isinstance(error.get("message"), str):
                messages.append(str(error["message"]))
    return "\n".join(messages)


def build_codex_argv(
    *,
    workdir: Path,
    model: str,
    schema_path: Path,
    last_message_path: Path,
    prompt: str,
) -> tuple[str, ...]:
    """Build the empirically settled headless Codex argv."""

    # W32: honor the checker CLI pin when declared. Absence stays the
    # historical bare name for analysis contexts; the launch preflight's
    # cli-versions arm refuses pinless PRODUCTION launches, and the paid
    # admission sites verify the declared pin per leg before any reserve.
    from menagerie.crawler.toolchain_pins import CHECKER_CLI_PIN_ENV, resolve_cli_pin

    pinned = resolve_cli_pin(CHECKER_CLI_PIN_ENV)
    return (
        *(pinned if pinned is not None else ("codex",)),
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "-C",
        str(workdir),
        "-m",
        model,
        "-c",
        f"model_reasoning_effort={CHECKER_EFFORT}",
        "--output-schema",
        str(schema_path),
        "-o",
        str(last_message_path),
        "--json",
        prompt,
    )


# ---------------------------------------------------------------------------
# Claude CLI (anthropic) adapter
# ---------------------------------------------------------------------------


def build_claude_argv(
    *,
    model: str,
    schema_path: Path,
    prompt: str,
    read_roots: Sequence[str],
) -> tuple[str, ...]:
    """Build the bench-proven headless Claude CLI argv for one checker session.

    ``--bare --setting-sources project`` keeps the session free of user-level
    hooks and memory; ``--json-schema`` enforces the same gate-derived native
    output schema Codex gets through ``--output-schema``; the tool policy is
    read-only; ``--add-dir`` grants exactly the envelope's declared read roots.
    The pin discipline mirrors Codex: a declared ``MENAGERIE_CHECKER_CLAUDE_BIN``
    is honored, absence stays the bare name for analysis contexts.
    """

    from menagerie.crawler.toolchain_pins import CHECKER_CLAUDE_CLI_PIN_ENV, resolve_cli_pin

    pinned = resolve_cli_pin(CHECKER_CLAUDE_CLI_PIN_ENV)
    schema_text = schema_path.read_text(encoding="utf-8")
    return (
        *(pinned if pinned is not None else ("claude",)),
        "-p",
        prompt,
        "--model",
        model,
        "--bare",
        "--setting-sources",
        "project",
        "--output-format",
        "json",
        "--json-schema",
        schema_text,
        "--no-session-persistence",
        "--allowedTools",
        *CLAUDE_CHECKER_ALLOWED_TOOLS,
        "--disallowedTools",
        *CLAUDE_CHECKER_DISALLOWED_TOOLS,
        "--add-dir",
        *read_roots,
    )


def parse_claude_harness(stdout: str) -> Optional[JsonObject]:
    """Parse the Claude CLI ``--output-format json`` result document, if any.

    The whole stdout is the document on a clean run; a trailing JSON line is
    accepted when diagnostics precede it. Non-object payloads are ``None``.
    """

    text = stdout.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def _claude_overload_basis(harness: Optional[Mapping[str, Any]]) -> str | None:
    """Return the closed overload basis the harness JSON structurally declares."""

    if not harness:
        return None
    status = harness.get("api_error_status")
    if not isinstance(status, bool) and status == _OVERLOADED_HTTP_STATUS:
        return CLAUDE_OVERLOADED_STATUS_BASIS
    error = harness.get("error")
    if isinstance(error, Mapping) and error.get("type") == _OVERLOADED_ERROR_TYPE:
        return CLAUDE_OVERLOADED_ERROR_BASIS
    return None


def classify_claude_attempt(attempt: CheckerAttempt) -> AttemptClassification:
    """Classify one Claude CLI checker result from its structured harness JSON.

    Authority order: machine overload event, structured usage-limit signal
    (the author lane's ``structured_limit_signal`` -- ``terminal_reason``,
    ``rate_limit_info.status``, ``api_error_status`` 429, ``error.type``),
    exhausted structured-output retries (contract), success, then the
    exit-gated auth markers and the bounded-retry default.
    """

    from menagerie.crawler.author_executor import (
        structured_limit_signal,
        structured_limit_reset_at,
    )

    harness = parse_claude_harness(attempt.stdout)
    overload = _claude_overload_basis(harness)
    if overload is not None:
        return AttemptClassification(FailureKind.PROVIDER_OVERLOADED, provider_event_basis=overload)
    limit = structured_limit_signal(harness)
    if limit is not None:
        return AttemptClassification(FailureKind.QUOTA, structured_limit_reset_at(harness))
    if harness is not None and harness.get("subtype") == _CLAUDE_STRUCTURED_OUTPUT_EXHAUSTED:
        return AttemptClassification(FailureKind.PERMANENT)
    if attempt.unavailable:
        return AttemptClassification(FailureKind.UNAVAILABLE)
    if attempt.timed_out:
        return AttemptClassification(FailureKind.TRANSIENT)
    if attempt.returncode == 0 and harness is not None and harness.get("is_error") is not True:
        return AttemptClassification(FailureKind.SUCCESS)
    lowered = f"{attempt.stderr}\n{_claude_result_text(harness)}".lower()
    if attempt.returncode != 0 and any(marker in lowered for marker in _AUTH_MARKERS):
        return AttemptClassification(FailureKind.AUTH)
    if any(marker in lowered for marker in _PERMANENT_MARKERS):
        return AttemptClassification(FailureKind.PERMANENT)
    # Unrecognized failure (nonzero exit, or exit 0 without a harness document):
    # the bounded retry ladder, exactly as the Codex adapter treats the unseen.
    return AttemptClassification(FailureKind.TRANSIENT)


def _claude_result_text(harness: Optional[Mapping[str, Any]]) -> str:
    """Return the harness's own error text (diagnostic labelling only)."""

    if not harness:
        return ""
    result = harness.get("result")
    return result if isinstance(result, str) else ""


def claude_usage_tokens(stdout: str) -> tuple[Optional[int], Optional[int]]:
    """Return ``(input_tokens, output_tokens)`` from the harness ``usage`` block.

    Input counts every prompt token the provider billed (fresh, cache-creation,
    and cache-read), mirroring the Codex ``turn.completed`` usage total.
    """

    harness = parse_claude_harness(stdout)
    usage = harness.get("usage") if harness else None
    if not isinstance(usage, Mapping):
        return (None, None)

    def _count(key: str) -> Optional[int]:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    parts = [_count(key) for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")]
    input_total = None if all(part is None for part in parts) else sum(part or 0 for part in parts)
    return (input_total, _count("output_tokens"))


def require_claude_model_witness(harness: Optional[Mapping[str, Any]], model: str) -> dict[str, Any]:
    """Return the harness ``modelUsage`` witness, refusing when the seat is absent.

    The seated model must appear with ``outputTokens > 0``. Every other model
    the harness used (the CLI's own side models) is DISCLOSED in the returned
    witness for telemetry; nothing here is read from the model's reply text.
    """

    usage = harness.get("modelUsage") if harness else None
    if not isinstance(usage, Mapping):
        raise CheckerModelWitnessError(
            f"checker session carries no modelUsage witness for seated model {model!r}"
        )
    seated = usage.get(model)
    output_tokens = seated.get("outputTokens") if isinstance(seated, Mapping) else None
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens <= 0:
        raise CheckerModelWitnessError(
            f"checker session modelUsage witness does not show seated model {model!r} "
            f"producing output (models used: {sorted(str(key) for key in usage)})"
        )
    return {
        "seated_model": model,
        "seated_output_tokens": output_tokens,
        "models_used": sorted(str(key) for key in usage),
    }


# ---------------------------------------------------------------------------
# Adapter surface
# ---------------------------------------------------------------------------


class CheckerProviderAdapter:
    """One provider's session mechanics behind the wrapper's fixed contract."""

    provider: str
    attempt_event: str

    def build_argv(
        self,
        *,
        workdir: Path,
        model: str,
        schema_path: Path,
        last_message_path: Path,
        prompt: str,
        read_roots: Sequence[str],
    ) -> tuple[str, ...]:
        raise NotImplementedError

    def classify(self, attempt: CheckerAttempt) -> AttemptClassification:
        raise NotImplementedError

    def usage_tokens(self, stdout: str) -> tuple[Optional[int], Optional[int]]:
        raise NotImplementedError

    def load_result(self, attempt: CheckerAttempt, last_message_path: Path, *, model: str) -> tuple[JsonObject, JsonObject]:
        """Return ``(decoded checker-authored fragment, machine witness)``."""

        raise NotImplementedError


class CodexCheckerAdapter(CheckerProviderAdapter):
    """The historical Codex path: ``-o`` final message, JSONL event stream."""

    provider = CHECKER_PROVIDER_OPENAI
    attempt_event = "codex-attempt"

    def build_argv(self, *, workdir, model, schema_path, last_message_path, prompt, read_roots):  # type: ignore[no-untyped-def]
        del read_roots  # Codex reads under its own read-only sandbox.
        return build_codex_argv(
            workdir=workdir,
            model=model,
            schema_path=schema_path,
            last_message_path=last_message_path,
            prompt=prompt,
        )

    def classify(self, attempt: CheckerAttempt) -> AttemptClassification:
        return classify_codex_attempt(attempt)

    def usage_tokens(self, stdout: str) -> tuple[Optional[int], Optional[int]]:
        return checker_usage_tokens(stdout)

    def load_result(self, attempt, last_message_path, *, model):  # type: ignore[no-untyped-def]
        del attempt
        path = last_message_path
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise CheckerDispatchError("Codex did not publish a non-empty final answer")
        try:
            decoded = decode_native_result(json.loads(path.read_text(encoding="utf-8")))
        except NativeOutputSchemaError as exc:
            raise CheckerDispatchError(str(exc)) from exc
        # The Codex CLI pins the model by argv; the pinned CLI release is the
        # witness the toolchain lane verifies per leg.
        return decoded, {"seated_model": model, "witness": "codex-cli-pin"}


class ClaudeCheckerAdapter(CheckerProviderAdapter):
    """The Anthropic seat: Claude CLI ``--json-schema`` structured output."""

    provider = CHECKER_PROVIDER_ANTHROPIC
    attempt_event = "codex-attempt"

    def build_argv(self, *, workdir, model, schema_path, last_message_path, prompt, read_roots):  # type: ignore[no-untyped-def]
        del workdir, last_message_path  # the result travels on stdout
        return build_claude_argv(
            model=model, schema_path=schema_path, prompt=prompt, read_roots=read_roots
        )

    def classify(self, attempt: CheckerAttempt) -> AttemptClassification:
        return classify_claude_attempt(attempt)

    def usage_tokens(self, stdout: str) -> tuple[Optional[int], Optional[int]]:
        return claude_usage_tokens(stdout)

    def load_result(self, attempt, last_message_path, *, model):  # type: ignore[no-untyped-def]
        del last_message_path
        harness = parse_claude_harness(attempt.stdout)
        if harness is None:
            raise CheckerDispatchError("Claude CLI did not publish a harness result document")
        # The witness is checked BEFORE the fragment is decoded: a verdict from
        # an unwitnessed model is refused whatever its shape.
        witness = require_claude_model_witness(harness, model)
        structured = harness.get("structured_output")
        if structured is None and isinstance(harness.get("result"), str):
            try:
                structured = json.loads(harness["result"])
            except json.JSONDecodeError:
                structured = None
        try:
            decoded = decode_native_result(structured)
        except NativeOutputSchemaError as exc:
            raise CheckerDispatchError(str(exc)) from exc
        return decoded, witness


_ADAPTERS: Mapping[str, CheckerProviderAdapter] = {
    CHECKER_PROVIDER_OPENAI: CodexCheckerAdapter(),
    CHECKER_PROVIDER_ANTHROPIC: ClaudeCheckerAdapter(),
}


def adapter_for_provider(provider: str) -> CheckerProviderAdapter:
    """Return the adapter for one closed checker provider, fail-closed."""

    try:
        return _ADAPTERS[provider]
    except KeyError:
        raise CheckerDispatchError(f"no checker adapter for provider {provider!r}") from None


def adapter_for_model(model: str) -> CheckerProviderAdapter:
    """Return the adapter for one seated checker model (provider derived)."""

    return adapter_for_provider(checker_provider_for_model(model))
