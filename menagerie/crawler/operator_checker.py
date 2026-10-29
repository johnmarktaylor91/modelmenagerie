"""Headless Codex wrapper for immutable menagerie checker envelopes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence, TextIO

from menagerie.crawler.checker_dispatch import (
    CheckerBindingMismatchError,
    CheckerDispatchError,
    CheckerItemContractError,
    PROMPT_PATH,
    apply_machine_owned_gate_fields,
    machine_owned_gate_fields,
    metadata_fidelity_normalizations,
    validate_checker_result_mapping,
)
from menagerie.crawler.checker_providers import (
    AttemptClassification,
    CheckerAttempt,
    CheckerModelWitnessError,
    CheckerProviderAdapter,
    CodexAttempt,
    FailureKind,
    adapter_for_provider,
    build_claude_argv,
    build_codex_argv,
    classify_claude_attempt,
    classify_codex_attempt,
)
from menagerie.crawler.constants import (
    CheckerSeatRole,
    GateKind,
    checker_provider_for_model,
    checker_seat,
    seated_checker_models,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import JsonObject
from menagerie.crawler.native_output_schema import (
    AUTHORED_PROPERTY,
    NativeOutputSchemaError,
    decode_native_result,
    native_output_schema,
)
from menagerie.crawler.operator_protocol import (
    OPERATOR_ATTEMPT_TIMEOUT_SECONDS,
    OPERATOR_MAX_ATTEMPTS,
    OPERATOR_PROTOCOL_VERSION,
    OPERATOR_REASONING_EFFORT,
    OperatorExitCode,
    OperatorProtocolError,
    append_telemetry,
    emit_combined_tail,
    load_absolute_request,
    publish_json_atomic,
    write_status,
)
from menagerie.crawler.paid_session_custody import inherited_custody_fds
from menagerie.crawler.worker_supervisor import (
    # The single hardened teardown, reused rather than re-derived: it proves the
    # group is still ours before signalling, which a raw
    # ``os.killpg(process.pid, ...)`` -- a PID used as a PGID -- never did.
    _kill_process_group as kill_process_group,
    capture_process_group,
)

CHECKER_TIMEOUT_SECONDS = float(OPERATOR_ATTEMPT_TIMEOUT_SECONDS)
CHECKER_MAX_ATTEMPTS = OPERATOR_MAX_ATTEMPTS
CHECKER_EFFORT = OPERATOR_REASONING_EFFORT
WRAPPER_VERSION = "menagerie-checker-operator 2.0.0"

#: The seated PRIMARY per gate kind (constants.CHECKER_SEATS). These names are
#: kept for their many readers; the seat table is the authority and the swap
#: (sol -> claude-fable-5-1, 2026-09-01) lands there, not here.
METADATA_CHECKER_MODEL = checker_seat(GateKind.METADATA_BATCH, CheckerSeatRole.PRIMARY).model
FIDELITY_CHECKER_MODEL = checker_seat(GateKind.FIDELITY, CheckerSeatRole.PRIMARY).model
TERMINAL_CHECKER_MODEL = checker_seat(GateKind.TERMINAL_DISPOSITION, CheckerSeatRole.PRIMARY).model

InvokeCodex = Callable[[Sequence[str], Path, float], "CodexAttempt"]
#: Provider-neutral spelling of the same injectable subprocess boundary.
InvokeCli = InvokeCodex
Sleep = Callable[[float], None]
Now = Callable[[], datetime]

__all__ = [
    "AttemptClassification",
    "CheckerAttempt",
    "CodexAttempt",
    "FailureKind",
    "FIDELITY_CHECKER_MODEL",
    "METADATA_CHECKER_MODEL",
    "TERMINAL_CHECKER_MODEL",
    "build_claude_argv",
    "build_codex_argv",
    "classify_claude_attempt",
    "classify_codex_attempt",
    "execute_checker_request",
    "main",
    "required_checker_model",
    "seated_checker_models",
]


def required_checker_model(
    gate_kind: GateKind | str, role: CheckerSeatRole | str = CheckerSeatRole.PRIMARY
) -> str:
    """Return the seated checker model for one gate kind and seat role.

    Parameters
    ----------
    gate_kind:
        Metadata, fidelity, or terminal-disposition gate.
    role:
        ``primary`` (the adjudicating seat, ``claude-fable-5-1``) or
        ``confirmation`` (the cross-model K=2 seat, ``gpt-5.6-sol``).

    Returns
    -------
    str
        Exact seated model identifier.
    """

    return checker_seat(gate_kind, role).model


def execute_checker_request(
    request_path: Path,
    *,
    invoke: InvokeCodex | None = None,
    sleep: Sleep = time.sleep,
    now: Now | None = None,
    diagnostic_stream: TextIO = sys.stderr,
) -> OperatorExitCode:
    """Execute one absolute checker request under the shared operator protocol.

    Parameters
    ----------
    request_path:
        Sole absolute request path supplied by the driver.
    invoke:
        Injectable subprocess boundary used by fault tests.
    sleep:
        Injectable retry backoff.
    now:
        Injectable UTC clock for quota fallback.
    diagnostic_stream:
        Merged bounded diagnostic destination.

    Returns
    -------
    OperatorExitCode
        Closed wrapper outcome.
    """

    runner = invoke or _invoke_cli
    clock = now or (lambda: datetime.now(timezone.utc))
    wrapper_started = _as_utc(clock())

    def elapsed() -> float:
        """Return non-negative wrapper wall seconds since the request was accepted."""

        return max((_as_utc(clock()) - wrapper_started).total_seconds(), 0.0)

    try:
        envelope = load_absolute_request(request_path)
        request_sha256, output_path, model, deadline = _validate_request(envelope, request_path)
        # Built inside the typed-rejection guard: the prompt-pin verification in
        # ``_build_prompt`` (BL-10 e) must exit as a clean classified refusal,
        # never an unhandled traceback.
        prompt = _build_prompt(envelope, request_path)
    except (OperatorProtocolError, CheckerDispatchError, OSError, ValueError) as exc:
        detail = str(exc)
        write_status(
            request_path,
            exit_code=OperatorExitCode.PERMANENT_CONTRACT_REJECTION,
            classification=FailureKind.PERMANENT.value,
            request_sha256=None,
            attempts=0,
            detail=detail,
        )
        append_telemetry(
            request_path,
            {
                "event": "request-rejected",
                "classification": FailureKind.PERMANENT.value,
                "wall_seconds": round(elapsed(), 3),
            },
        )
        diagnostic_stream.write(f"{detail}\n")
        return OperatorExitCode.PERMANENT_CONTRACT_REJECTION

    gate_kind = str(envelope.get("gate_kind"))
    raw_items = envelope.get("items")
    item_count = len(raw_items) if isinstance(raw_items, list) else None
    # Item 32: ONE adapter per session, selected from the provider the seated
    # model derives (the same derivation `_validate_request` just enforced
    # against the envelope), so the argv, classifier, token accounting, and
    # result transport are the seat's own while everything around them --
    # prompt, schema, stamping, validation, custody, exits -- stays shared.
    provider = checker_provider_for_model(model)
    adapter: CheckerProviderAdapter = adapter_for_provider(provider)
    read_roots = [str(root) for root in envelope.get("allowed_read_roots", [])]
    with tempfile.TemporaryDirectory(prefix=".checker-", dir=output_path.parent) as temporary:
        temp_root = Path(temporary)
        schema_path = temp_root / "gate-output-schema.json"
        last_message_path = temp_root / "last-message.json"
        publish_json_atomic(schema_path, _native_output_schema(GateKind(gate_kind)))
        last_detail = ""
        # Corrective feedback for the NEXT attempt after a contract refusal. The
        # frozen prompt bytes stay pinned; the feedback travels in the same
        # dynamic prompt region that already carries WORK_ENVELOPE_PATH. Rung 8
        # showed why silence loses: attempt 1 was refused for a stale
        # ``fidelity_identity`` echo, attempt 2 re-ran the identical prompt with
        # nothing naming the flub, and the batch died on its next slip.
        contract_feedback: str | None = None
        # A structural contract refusal (schema shape, decision logic -- anything
        # that is not the typed binding-mismatch flub) earns a bounded re-ask
        # with the exact error fed back, mirroring the author-side echo retry.
        # The ledger is per-DISTINCT-defect, not one shared boolean: the pilot's
        # refusal telemetry shows the fed-back error is almost always repaired
        # (102 of 107 consecutive refusal pairs name DIFFERENT defects; the
        # dominant wrong-excerpt slip recurred identically only 8 times in 116)
        # while a shared single re-ask terminalized convergent batches on their
        # second, unrelated slip. A repeat of the SAME named error is still
        # systematic divergence and exits permanent immediately -- and the
        # attempt cap bounds the whole ladder exactly as before.
        refused_contract_details: set[str] = set()
        for attempt_number in range(1, CHECKER_MAX_ATTEMPTS + 1):
            remaining = (deadline - clock().astimezone(timezone.utc)).total_seconds()
            if remaining <= 0:
                return _finish(
                    request_path,
                    OperatorExitCode.RETRYABLE_INFRASTRUCTURE,
                    FailureKind.TRANSIENT,
                    request_sha256,
                    attempt_number - 1,
                    "checker operator deadline expired before the next attempt",
                    wall_seconds=elapsed(),
                )
            # The bound this attempt is actually held to. It is the granted cap
            # until the published deadline is nearer, and it is the value that
            # CENSORS a timed-out attempt -- a duration equal to it means the
            # attempt hit the wall, a smaller one means it stopped short. Recorded
            # next to the duration so the pair is interpretable on its own.
            attempt_budget = min(CHECKER_TIMEOUT_SECONDS, remaining)
            argv = adapter.build_argv(
                workdir=request_path.parent,
                model=model,
                schema_path=schema_path,
                last_message_path=last_message_path,
                prompt=(
                    prompt
                    if contract_feedback is None
                    else _with_contract_feedback(prompt, contract_feedback)
                ),
                read_roots=read_roots,
            )
            started_instant = _as_utc(clock())
            attempt = runner(argv, last_message_path, attempt_budget)
            finished_instant = _as_utc(clock())
            started_at = _utc_timestamp(started_instant)
            finished_at = _utc_timestamp(finished_instant)
            attempt_seconds = max((finished_instant - started_instant).total_seconds(), 0.0)
            classification = adapter.classify(attempt)
            input_tokens, output_tokens = adapter.usage_tokens(attempt.stdout)
            last_detail = emit_combined_tail(
                attempt.stdout,
                attempt.stderr,
                stream=diagnostic_stream,
            )
            append_telemetry(
                request_path,
                {
                    "event": adapter.attempt_event,
                    "attempt": attempt_number,
                    # Seat swap: the adjudicating seat is named on every
                    # attempt row, so a latency or census read can group by
                    # (provider, model) without opening the request.
                    "provider": provider,
                    "model": model,
                    "returncode": attempt.returncode,
                    "classification": classification.kind.value,
                    "timed_out": attempt.timed_out,
                    # Attempt duration is recorded for COMPLETED and TIMED-OUT
                    # attempts alike: a timed-out attempt is a right-censored
                    # observation, not an absent one, and dropping it biases every
                    # later estimate of the real distribution downward. Without
                    # these fields the only recoverable timing was a lane total
                    # differenced from file mtimes, which is why the attempt cap
                    # was never measured at all.
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_seconds": round(attempt_seconds, 3),
                    "attempt_budget_seconds": round(float(attempt_budget), 3),
                    "attempt_timeout_seconds": CHECKER_TIMEOUT_SECONDS,
                    # The covariates that make the sample a DISTRIBUTION rather than
                    # a pile: one flat cap covers a one-model fidelity call and a
                    # `METADATA_BATCH_MIN`..`METADATA_BATCH_MAX` metadata batch, so
                    # durations pooled across gate kinds and batch sizes cannot say
                    # what the cap should be for either.
                    "gate_kind": gate_kind,
                    "item_count": item_count,
                    "batch_size": item_count,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "detail": last_detail,
                },
            )
            if classification.kind is FailureKind.SUCCESS:
                try:
                    # The verdict is the checker's; the identities and wall
                    # timings are the machine's. Stamping first means a complete
                    # verdict is never thrown away over a scaffold field the
                    # model could not observe, and it means no model-supplied
                    # identity is ever believed.
                    decoded, model_witness = adapter.load_result(
                        attempt, last_message_path, model=model
                    )
                    # Named BEFORE the stamp overwrites it: a divergent supplied
                    # fidelity block on a metadata gate is normalized (it is a
                    # gate-kind constant, not a judgment), but the divergence is
                    # evidence of checker confusion and must survive as
                    # telemetry rather than vanish into the machine fill.
                    normalized_fidelity_ids = metadata_fidelity_normalizations(decoded, envelope)
                    result = apply_machine_owned_gate_fields(
                        decoded,
                        envelope,
                        started_at=started_at,
                        finished_at=finished_at,
                    )
                    validate_checker_result_mapping(result, envelope)
                except CheckerModelWitnessError as exc:
                    # The harness witness does not show the seated model
                    # adjudicating. No re-ask: another model's verdict is not
                    # a slip to repair, it is the flip-back condition (c) of
                    # the seat ruling, and the result is never published.
                    last_detail = str(exc)
                    append_telemetry(
                        request_path,
                        {
                            "event": "model-witness-refused",
                            "attempt": attempt_number,
                            "provider": provider,
                            "model": model,
                            "detail": last_detail,
                        },
                    )
                    return _finish(
                        request_path,
                        OperatorExitCode.PERMANENT_CONTRACT_REJECTION,
                        FailureKind.PERMANENT,
                        request_sha256,
                        attempt_number,
                        last_detail,
                        wall_seconds=elapsed(),
                    )
                except CheckerBindingMismatchError as exc:
                    # An echoed identity binding that differs from the envelope
                    # is the one contract refusal with a stochastic cause: the
                    # model mis-transcribed a machine-known constant it was told
                    # to copy verbatim (observed live: one flubbed
                    # ``vet_identity`` out of six identity echoes in a run,
                    # every sibling exact). The mismatched result is refused
                    # unconditionally -- it is never published and the
                    # comparison is never loosened -- but refusing the whole
                    # batch PERMANENTLY on the first flub turned one bad
                    # transcription into a terminal ``protocol-violation`` for
                    # every batch member on a run-once system. A fresh bounded
                    # attempt re-reads the same frozen envelope WITH the exact
                    # refusal fed back, so the re-ask can repair a flub while a
                    # systematic divergence still exhausts the attempts and
                    # exits as the same permanent contract rejection as before.
                    last_detail = str(exc)
                    contract_feedback = last_detail
                    append_telemetry(
                        request_path,
                        {
                            "event": "binding-mismatch-refused",
                            "attempt": attempt_number,
                            "retrying": attempt_number < CHECKER_MAX_ATTEMPTS,
                            "detail": last_detail,
                        },
                    )
                    if attempt_number < CHECKER_MAX_ATTEMPTS:
                        continue
                    return _finish(
                        request_path,
                        OperatorExitCode.PERMANENT_CONTRACT_REJECTION,
                        FailureKind.PERMANENT,
                        request_sha256,
                        attempt_number,
                        str(exc),
                        wall_seconds=elapsed(),
                    )
                except CheckerItemContractError as exc:
                    # BL-10 (c): the refusal names exactly one item. An
                    # AUTHOR-side defect (the write-contract replay refusing an
                    # item over its own frozen author artifact) is deterministic
                    # for this envelope -- the checker re-answers from the same
                    # frozen bytes and is refused identically, which is exactly
                    # what 156 of the pilot's 233 refusal events did -- so no
                    # re-ask is spent on it. A CHECKER-side per-item defect
                    # (complaint-disclosure enumeration) keeps the ONE bounded
                    # re-ask with the exact error fed back, exactly like every
                    # other structural refusal. Either way the exit is the
                    # ITEM-scoped typed code with the offending stable IDs on
                    # the status sidecar, so the driver can route the named
                    # models and re-batch the innocent siblings instead of
                    # destroying the whole batch through the halving ladder.
                    last_detail = str(exc)
                    retrying = (
                        not exc.author_side
                        and last_detail not in refused_contract_details
                        and attempt_number < CHECKER_MAX_ATTEMPTS
                    )
                    append_telemetry(
                        request_path,
                        {
                            "event": "item-contract-refused",
                            "attempt": attempt_number,
                            "author_side": exc.author_side,
                            "stable_ids": list(exc.stable_ids),
                            "retrying": retrying,
                            "detail": last_detail,
                        },
                    )
                    if retrying:
                        refused_contract_details.add(last_detail)
                        contract_feedback = last_detail
                        continue
                    return _finish(
                        request_path,
                        OperatorExitCode.ITEM_CONTRACT_REJECTION,
                        FailureKind.PERMANENT,
                        request_sha256,
                        attempt_number,
                        str(exc),
                        item_contract={
                            "stable_ids": list(exc.stable_ids),
                            "author_side": exc.author_side,
                        },
                        wall_seconds=elapsed(),
                    )
                except (
                    CheckerDispatchError,
                    OSError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ) as exc:
                    # A structural contract refusal: the checker returned a
                    # complete answer whose SHAPE the machine validator refused
                    # (never its merits -- a validated adverse verdict is a
                    # SUCCESS publication and no code path re-asks over one).
                    # Rung 8 killed both members of a two-model batch here in
                    # one second, on a slip the checker was never told about.
                    # A bounded re-ask with the exact error fed back mirrors
                    # the author-side echo retry: the refused result is never
                    # published, the validator is never loosened, and a second
                    # refusal with the SAME error named is systematic
                    # divergence and exits permanent exactly as before. A
                    # refusal naming a DIFFERENT defect is a fresh slip, not
                    # divergence, and earns its own fed-back re-ask within the
                    # unchanged attempt cap.
                    last_detail = str(exc)
                    retrying = (
                        last_detail not in refused_contract_details
                        and attempt_number < CHECKER_MAX_ATTEMPTS
                    )
                    append_telemetry(
                        request_path,
                        {
                            "event": "contract-rejection-refused",
                            "attempt": attempt_number,
                            "retrying": retrying,
                            "detail": last_detail,
                        },
                    )
                    if retrying:
                        refused_contract_details.add(last_detail)
                        contract_feedback = last_detail
                        continue
                    return _finish(
                        request_path,
                        OperatorExitCode.PERMANENT_CONTRACT_REJECTION,
                        FailureKind.PERMANENT,
                        request_sha256,
                        attempt_number,
                        str(exc),
                        wall_seconds=elapsed(),
                    )
                if normalized_fidelity_ids:
                    append_telemetry(
                        request_path,
                        {
                            "event": "fidelity-normalized",
                            "attempt": attempt_number,
                            "stable_ids": list(normalized_fidelity_ids),
                        },
                    )
                append_telemetry(
                    request_path,
                    {
                        "event": "model-witness",
                        "attempt": attempt_number,
                        "provider": provider,
                        "model": model,
                        "witness": model_witness,
                    },
                )
                publish_json_atomic(output_path, result)
                return _finish(
                    request_path,
                    OperatorExitCode.SUCCESS,
                    FailureKind.SUCCESS,
                    request_sha256,
                    attempt_number,
                    "checker result validated and published atomically",
                    wall_seconds=elapsed(),
                )
            if classification.kind is FailureKind.QUOTA:
                reset_at, observation = _quota_reset(classification.reset_text, clock())
                return _finish(
                    request_path,
                    OperatorExitCode.RATE_OR_QUOTA_PAUSE,
                    FailureKind.QUOTA,
                    request_sha256,
                    attempt_number,
                    last_detail,
                    reset_at=reset_at,
                    reset_observation=observation,
                    wall_seconds=elapsed(),
                )
            if classification.kind is FailureKind.AUTH:
                # A dead credential fails every retry identically, so exactly one
                # attempt is spent. The exit is the typed pause code -- the lane
                # pauses like quota -- and the status classification names
                # re-authentication so the operator knows the remedy is `codex
                # login`, not waiting for a window.
                return _finish(
                    request_path,
                    OperatorExitCode.RATE_OR_QUOTA_PAUSE,
                    FailureKind.AUTH,
                    request_sha256,
                    attempt_number,
                    last_detail,
                    wall_seconds=elapsed(),
                )
            if classification.kind is FailureKind.PERMANENT:
                return _finish(
                    request_path,
                    OperatorExitCode.PERMANENT_CONTRACT_REJECTION,
                    FailureKind.PERMANENT,
                    request_sha256,
                    attempt_number,
                    last_detail,
                    wall_seconds=elapsed(),
                )
            if classification.kind is FailureKind.UNAVAILABLE:
                return _finish(
                    request_path,
                    OperatorExitCode.SERVICE_UNAVAILABLE,
                    FailureKind.UNAVAILABLE,
                    request_sha256,
                    attempt_number,
                    last_detail,
                    wall_seconds=elapsed(),
                )
            if classification.kind is FailureKind.PROVIDER_OVERLOADED:
                return _finish(
                    request_path,
                    OperatorExitCode.SERVICE_UNAVAILABLE,
                    FailureKind.PROVIDER_OVERLOADED,
                    request_sha256,
                    attempt_number,
                    last_detail,
                    provider_event_basis=classification.provider_event_basis,
                    wall_seconds=elapsed(),
                )
            if attempt_number < CHECKER_MAX_ATTEMPTS:
                sleep(float(2 ** (attempt_number - 1)))
        return _finish(
            request_path,
            OperatorExitCode.RETRYABLE_INFRASTRUCTURE,
            FailureKind.TRANSIENT,
            request_sha256,
            CHECKER_MAX_ATTEMPTS,
            last_detail,
            wall_seconds=elapsed(),
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checker wrapper CLI with exactly one absolute request argument.

    Parameters
    ----------
    argv:
        Optional argument vector excluding the program name.

    Returns
    -------
    int
        Closed operator exit code.
    """

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments == ("--version",):
        print(WRAPPER_VERSION)
        return int(OperatorExitCode.SUCCESS)
    if len(arguments) != 1:
        sys.stderr.write(
            "usage: python -m menagerie.crawler.operator_checker /absolute/request.json\n"
        )
        return int(OperatorExitCode.PERMANENT_CONTRACT_REJECTION)
    return int(execute_checker_request(Path(arguments[0])))


def _validate_request(
    envelope: Mapping[str, Any], request_path: Path
) -> tuple[str, Path, str, datetime]:
    """Validate checker-specific request bindings and output confinement.

    Parameters
    ----------
    envelope:
        Parsed checker envelope.
    request_path:
        Exact absolute request path.

    Returns
    -------
    tuple[str, pathlib.Path, str, datetime.datetime]
        Request digest, exact output path, locked model, and UTC deadline.

    Raises
    ------
    CheckerDispatchError
        If any identity, tier, or path binding is invalid.
    """

    if envelope.get("envelope_version") != "menagerie.crawler.checker-envelope.v3":
        raise CheckerDispatchError("unsupported checker envelope version")
    if envelope.get("operator_protocol_version") != OPERATOR_PROTOCOL_VERSION:
        raise CheckerDispatchError("unsupported common operator protocol version")
    expected_hash = stable_hash(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )
    if envelope.get("envelope_sha256") != expected_hash:
        raise CheckerDispatchError("checker envelope hash mismatch")
    try:
        kind = GateKind(str(envelope["gate_kind"]))
    except (KeyError, ValueError) as exc:
        raise CheckerDispatchError("checker envelope has an invalid gate_kind") from exc
    checker = envelope.get("checker")
    if not isinstance(checker, Mapping):
        raise CheckerDispatchError("checker envelope lacks its exact checker identity")
    model = checker.get("model")
    seated = seated_checker_models(kind)
    if not isinstance(model, str) or model not in seated:
        raise CheckerDispatchError(
            f"{kind.value} requires a seated checker model (one of {sorted(seated)}), got {model!r}"
        )
    # The provider is DERIVED from the seated model, never trusted from the
    # envelope: a Fable envelope stamped `openai` (or a sol envelope stamped
    # `anthropic`) would adjudicate under an identity its provider does not own.
    expected_provider = checker_provider_for_model(model)
    if checker.get("provider") != expected_provider:
        raise CheckerDispatchError(
            f"checker envelope provider must be {expected_provider!r} for seated model "
            f"{model!r}, got {checker.get('provider')!r}"
        )
    if envelope.get("operator_model") != model:
        raise CheckerDispatchError("operator model does not match checker identity")
    if envelope.get("work_generation_identity") != envelope.get("request_nonce"):
        raise CheckerDispatchError("operator work generation does not bind the request nonce")
    effort_grant = envelope.get("effort_grant")
    expected_effort = {
        "max_attempts": CHECKER_MAX_ATTEMPTS,
        "attempt_timeout_seconds": int(CHECKER_TIMEOUT_SECONDS),
        "reasoning_effort": CHECKER_EFFORT,
    }
    if effort_grant != expected_effort:
        raise CheckerDispatchError("checker effort grant does not match the locked cap")
    deadline_value = envelope.get("deadline_at")
    if not isinstance(deadline_value, str):
        raise CheckerDispatchError("checker envelope lacks an exact deadline")
    try:
        deadline = datetime.fromisoformat(deadline_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CheckerDispatchError("checker deadline is not an ISO timestamp") from exc
    if deadline.tzinfo is None:
        raise CheckerDispatchError("checker deadline must be timezone-aware")
    output_value = envelope.get("required_output_path")
    root_value = envelope.get("allowed_output_root")
    if not isinstance(output_value, str) or not isinstance(root_value, str):
        raise CheckerDispatchError("checker envelope lacks exact output paths")
    output_path = Path(output_value)
    output_root = Path(root_value)
    if not output_path.is_absolute() or not output_root.is_absolute():
        raise CheckerDispatchError("checker output paths must be absolute")
    if envelope.get("allowed_write_root") != str(output_root) or envelope.get(
        "operator_required_output_path"
    ) != str(output_path):
        raise CheckerDispatchError("common operator write paths do not match checker paths")
    read_roots = envelope.get("allowed_read_roots")
    if not isinstance(read_roots, list) or any(
        not isinstance(value, str) or not Path(value).is_absolute() for value in read_roots
    ):
        raise CheckerDispatchError("checker allowed read roots are incomplete or non-absolute")
    if str(request_path.parent.resolve()) not in read_roots:
        raise CheckerDispatchError("checker allowed read roots omit the request directory")
    # The declaration must describe the reads that actually happen. The frozen
    # prompt is inlined into the argv by ``_build_prompt``, so the checker never
    # opens the repository ``prompts`` directory; declaring it invited exactly
    # the repository-wandering this lane was observed doing, and made the
    # declaration unauditable against behavior. Refusing it keeps a revert
    # visible instead of silent.
    if str(PROMPT_PATH.parent.resolve()) in read_roots:
        raise CheckerDispatchError(
            "checker allowed read roots declare the frozen prompt directory, which the wrapper "
            "inlines and the checker never reads"
        )
    # A declaration naming only the request directory is degenerate: the checker
    # cannot re-derive a single excerpt without the frozen bytes under
    # ``source-cas``, so an envelope that declares no source root is describing a
    # checker that cannot do its job.
    if len(read_roots) < 2:
        raise CheckerDispatchError(
            "checker allowed read roots declare no frozen source root for excerpt re-derivation"
        )
    if (
        output_path.resolve().parent != output_root.resolve()
        or output_path.name != "result.json"
        or output_root.resolve() != request_path.parent.resolve()
    ):
        raise CheckerDispatchError("checker output must be request-sibling result.json")
    if output_root.is_symlink():
        raise CheckerDispatchError("checker output root cannot be a symlink")
    return expected_hash, output_path.resolve(), model, deadline.astimezone(timezone.utc)


def _build_prompt(envelope: Mapping[str, Any], request_path: Path) -> str:
    """Bind the frozen checker instructions to one complete request object.

    Parameters
    ----------
    envelope:
        Validated checker envelope.
    request_path:
        Exact request path used for audit identity.

    Returns
    -------
    str
        One-shot prompt argument.
    """

    # BL-10 (e): the wrapper reads the prompt FILE, but the envelope's pinned
    # ``prompt.sha256`` is the authority -- a working-tree edit between envelope
    # seal and wrapper start would otherwise adjudicate under prompt bytes the
    # gate row does not name (the CH-F3 split brain, at the session itself).
    # Bytes that disagree with the pin refuse typed; the driver's ordinary
    # contract-rejection route carries the refusal.
    frozen_bytes = PROMPT_PATH.read_bytes()
    declared = envelope.get("prompt")
    pinned = declared.get("sha256") if isinstance(declared, Mapping) else None
    if isinstance(pinned, str) and pinned:
        from menagerie.crawler.identity import hash_bytes

        live = hash_bytes(frozen_bytes)
        if live != pinned:
            raise CheckerDispatchError(
                f"checker prompt rotated between envelope seal and dispatch: the live "
                f"prompt file hashes to {live} but the envelope pinned {pinned}; prompt "
                "rotation is a leg-boundary act"
            )
    frozen = frozen_bytes.decode("utf-8")
    machine_owned_names = sorted(machine_owned_gate_fields(envelope)) + [
        "checker.provider",
        "checker.model",
        "checker.version",
        "checker.prompt_sha256",
        "checker.started_at",
        "checker.finished_at",
    ]
    # On a metadata gate the per-item fidelity block is a gate-kind constant the
    # wrapper stamps; it is absent from the metadata output schema by design, so
    # the enumeration must name it or the prompt's ownership table understates
    # what the machine owns for this envelope.
    if envelope.get("gate_kind") == GateKind.METADATA_BATCH.value:
        machine_owned_names.append("items[*].fidelity")
    machine_owned = ", ".join(machine_owned_names)
    return (
        f"{frozen}\n\n"
        f"{_digest_authority_clause(envelope)}"
        f"WORK_ENVELOPE_PATH={request_path}\n"
        "The outer read-only wrapper publishes result.json. Do not write files. "
        "Read the exact envelope at WORK_ENVELOPE_PATH, judge every item, and return your "
        f"verdicts in the output schema's sole `{AUTHORED_PROPERTY}` array.\n"
        "That schema is DERIVED FROM menagerie.crawler.gate.v3 itself and is enforced "
        "natively, so its key vocabulary IS the gate's: you cannot add a key to any block "
        "and you cannot omit one. Do not reason about the gate's shape -- emit your "
        "judgements and let the schema hold the shape.\n"
        "These gate fields are MACHINE-OWNED and are stamped by the wrapper from the "
        f"envelope: {machine_owned}. They are absent from your output schema BY DESIGN, so "
        "omitting them is not merely correct, it is the only thing you can do. Do NOT "
        "attempt to smuggle a value for one into a field that is yours. Your authority is "
        "the verdict, the per-field checks, the findings, the unsupported claims, and the "
        "required repairs."
    )


def _digest_authority_clause(envelope: Mapping[str, Any]) -> str:
    """Name the machine digest authority in the dynamic region of an attested envelope.

    F51 checker-lifecycle D8: when every item carries the coordinator-frozen
    ``digest_attestation`` block, digest verification is machine-owned and the
    frozen prompt's standing clause-1 duty ("re-verify every digest") is
    DISCHARGED by citing the rows. This rides in the same dynamic region as
    ``WORK_ENVELOPE_PATH`` and the machine-owned field enumeration -- envelope
    facts the wrapper binds per request -- so the frozen prompt bytes (the
    ``checker_prompt`` identity) do not move; a legacy or unattested envelope
    gets an empty clause and keeps historical semantics untouched. The
    result-side twin is ``checker_dispatch._validate_item_digest_authority``.
    """

    from menagerie.crawler.checker_dispatch import (
        DIGEST_ATTESTATION_AUTHORITY,
        DIGEST_ATTESTATION_UNRESOLVABLE,
        DIGEST_ATTESTATION_VALID,
    )

    items = envelope.get("items")
    items = items if isinstance(items, list) else []
    if not items or not all(
        isinstance(item, Mapping) and isinstance(item.get("digest_attestation"), Mapping)
        for item in items
    ):
        return ""
    return (
        "DIGEST VERIFICATION IS MACHINE-OWNED ON THIS ENVELOPE. Every item carries the "
        f"coordinator-frozen `digest_attestation` block (authority {DIGEST_ATTESTATION_AUTHORITY}): "
        "the machine already recomputed every non-null `verified_hashes` key against the exact "
        "frozen bytes it binds (the `basis` on each row names how) and recorded each row's "
        f"status. A mismatch never reaches you (it refuses the envelope before any session), so "
        f"every row you see is `{DIGEST_ATTESTATION_VALID}` or `{DIGEST_ATTESTATION_UNRESOLVABLE}`. "
        "Those rows ARE the digest verification AUTHORITY clause 1 asks for: do not recompute, "
        "re-hash, or dispute any digest by shell or by eye. When every row is "
        f"`{DIGEST_ATTESTATION_VALID}`, `integrity.hash_mismatches` is exactly [] and no "
        "integrity verdict may rest on a digest doubt; an entry naming a valid row disputes "
        f"machine arithmetic and your entire result is refused typed. An "
        f"`{DIGEST_ATTESTATION_UNRESOLVABLE}` row (the machine could not locate the staged bytes "
        "that key binds) is insufficiency exactly like an unattestable locator row: you may cite "
        'it as {"digest_key": <that row\'s digest_key>, "detail": <optional prose>}, it degrades '
        "only the claims resting on those bytes, and it is never a positive contradiction.\n"
    )


def _native_output_schema(gate_kind: GateKind) -> JsonObject:
    """Return the native structured-output schema derived from ``gate.v3`` itself.

    This used to be a one-field ``result_json`` STRING transport, which made the
    provider's structured-output constraint enforce nothing about the gate: the
    real content was an opaque string, so the only channel carrying the gate's
    shape was prose in the frozen prompt. The checker consequently invented
    schema-shaped keys for whichever closed block the prose had not yet
    enumerated -- ``terminal_disposition.arm``, then ``rung_check.required``,
    then ``integrity.findings`` -- and each fix moved the failure to the next
    nested block rather than closing the class.

    Parameters
    ----------
    gate_kind:
        Exact gate kind of the validated request envelope. The schema is derived
        per kind, so a metadata checker cannot represent a fidelity verdict at
        all (rung 8: one such verdict killed both models of a batch).

    Returns
    -------
    dict[str, Any]
        Strict-subset schema pinning the exact ``gate.v3`` item vocabulary for
        this gate kind. The decoded gate still passes the complete repository
        schema and semantic validator before publication; this constraint is
        additional, never a replacement.
    """

    return native_output_schema(gate_kind)


def _with_contract_feedback(prompt: str, error: str) -> str:
    """Append one refused attempt's exact contract error to the dynamic prompt.

    The frozen prompt bytes stay pinned -- this rides in the same dynamic region
    that already carries ``WORK_ENVELOPE_PATH`` -- and the feedback names the
    machine validator's exact refusal so the re-ask repairs the one defect
    instead of re-rolling the dice on an identical prompt.

    Parameters
    ----------
    prompt:
        Complete one-shot prompt for this envelope.
    error:
        Exact refusal text from the machine validator, bounded so a pathological
        error cannot displace the instructions it annotates.

    Returns
    -------
    str
        Prompt for the corrective attempt.
    """

    return (
        f"{prompt}\n\n"
        "YOUR PREVIOUS ATTEMPT WAS REFUSED BY THE MACHINE VALIDATOR. Exact error:\n"
        f"{error[:2_000]}\n"
        "That refusal is mechanical -- output shape or identity binding -- and is NOT a "
        "judgment on your verdicts; do not change any judgment because of it. Re-read the "
        "envelope at WORK_ENVELOPE_PATH, correct exactly the named defect, and return your "
        "COMPLETE verdicts for every envelope item again. Envelope-bound identity fields must "
        "be copied VERBATIM from THIS envelope's items, never from any earlier round's files."
    )


def _invoke_cli(
    argv: Sequence[str], last_message_path: Path, timeout_seconds: float
) -> CodexAttempt:
    """Run one checker CLI in a fresh process group with an external wall-clock timeout.

    Provider-neutral: the Codex and Claude adapters hand over an argv and get
    back captured streams; the process-group containment, the custody
    descriptor inheritance, and the timeout teardown are identical for both
    seats, which is exactly what makes a Fable session charge custody and wall
    like a sol session.

    Parameters
    ----------
    argv:
        Exact non-shell checker CLI argv.
    last_message_path:
        Expected native final-message output, used only for interface symmetry.
    timeout_seconds:
        External process-group wall limit.

    Returns
    -------
    CodexAttempt
        Captured streams and typed timeout/unavailability facts.
    """

    del last_message_path
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.environ.get("MENAGERIE_PARENT_DEADLINE_GUARD") != "1",
            pass_fds=inherited_custody_fds(),
        )
    except OSError as exc:
        return CodexAttempt(127, "", str(exc), unavailable=True)
    # Bound the group identity while the spawn's PID is still provably this
    # child's, so the timeout teardown below signals a group it can prove it owns
    # rather than whatever group happens to hold that ID by then.
    group = capture_process_group(process)
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return CodexAttempt(process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        del exc
        if not kill_process_group(group).benign:
            # The group is not provably ours, so it is left alone. The root child
            # is still unambiguously ours while unreaped, and killing it is what
            # keeps the drain below from blocking forever on a live process.
            process.kill()
        stdout, stderr = process.communicate()
        return CodexAttempt(
            process.returncode if process.returncode is not None else -signal.SIGKILL,
            stdout,
            stderr,
            timed_out=True,
        )


#: Historical name of the injectable runner (fault tests spell it this way).
_invoke_codex = _invoke_cli


def _load_last_message(path: Path) -> JsonObject:
    """Load the natively constrained final answer as the checker's authored gate.

    The final message is now the gate fragment itself rather than a string
    carrying one, so the provider's constraint applies to the object the wrapper
    actually consumes.

    Parameters
    ----------
    path:
        Codex ``-o`` destination.

    Returns
    -------
    dict[str, Any]
        Candidate gate carrying exactly the checker-authored items.

    Raises
    ------
    CheckerDispatchError
        If the output is absent, symlinked, empty, or does not match the native
        transport.
    """

    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise CheckerDispatchError("Codex did not publish a non-empty final answer")
    try:
        return decode_native_result(json.loads(path.read_text(encoding="utf-8")))
    except NativeOutputSchemaError as exc:
        raise CheckerDispatchError(str(exc)) from exc


def _as_utc(moment: datetime) -> datetime:
    """Return one timezone-aware UTC instant for a wrapper-observed clock reading.

    Parameters
    ----------
    moment:
        Instant observed through the injectable clock, aware or naive.

    Returns
    -------
    datetime.datetime
        The same instant in UTC, safe to subtract from another reading.
    """

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _utc_timestamp(moment: datetime) -> str:
    """Return one schema-valid RFC 3339 UTC timestamp for a wrapper-observed instant.

    Parameters
    ----------
    moment:
        Instant observed through the injectable clock.

    Returns
    -------
    str
        Timestamp ending in ``Z``.
    """

    return _as_utc(moment).isoformat().replace("+00:00", "Z")


def _quota_reset(reset_text: str | None, now: datetime) -> tuple[str, str]:
    """Normalize an extracted Codex reset string or use the one-hour fallback.

    Parameters
    ----------
    reset_text:
        Free-text value captured after ``try again at``.
    now:
        Current aware instant.

    Returns
    -------
    tuple[str, str]
        UTC reset timestamp and ``observed``/``guessed`` provenance.
    """

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if reset_text is not None:
        parsed = _parse_reset_text(reset_text, now)
        if parsed is not None:
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), "observed"
    fallback = now.astimezone(timezone.utc) + timedelta(hours=1)
    return fallback.isoformat().replace("+00:00", "Z"), "guessed"


def _parse_reset_text(value: str, now: datetime) -> datetime | None:
    """Parse known Codex free-text reset formats without guessing missing fields.

    Parameters
    ----------
    value:
        Extracted reset text.
    now:
        Aware instant supplying the machine-local timezone.

    Returns
    -------
    datetime.datetime | None
        Parsed aware timestamp, or ``None`` for the explicit fallback path.
    """

    normalized = value.strip().replace('\\"', "").strip("\"'")
    iso_candidate = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso_candidate)
    except ValueError:
        parsed = None
    if parsed is not None:
        return (
            parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=now.astimezone().tzinfo)
        )
    for format_string in (
        "%b %d, %Y %I:%M %p",
        "%B %d, %Y %I:%M %p",
        "%b %d, %Y, %I:%M %p",
        "%B %d, %Y, %I:%M %p",
        "%m/%d/%Y, %I:%M %p",
    ):
        try:
            parsed = datetime.strptime(normalized, format_string)
        except ValueError:
            continue
        return parsed.replace(tzinfo=now.astimezone().tzinfo)
    return None


def _finish(
    request_path: Path,
    exit_code: OperatorExitCode,
    classification: FailureKind,
    request_sha256: str,
    attempts: int,
    detail: str,
    *,
    reset_at: str | None = None,
    reset_observation: str | None = None,
    provider_event_basis: str | None = None,
    item_contract: Mapping[str, Any] | None = None,
    wall_seconds: float | None = None,
) -> OperatorExitCode:
    """Publish final status and telemetry for one wrapper outcome.

    Parameters
    ----------
    request_path:
        Exact request path.
    exit_code:
        Closed process outcome.
    classification:
        Structured classification authority.
    request_sha256:
        Validated request digest.
    attempts:
        External attempts consumed.
    detail:
        Bounded diagnostic.
    reset_at, reset_observation:
        Optional quota wake facts.
    provider_event_basis:
        Closed machine-derived provider-event basis for weather classifications.
    item_contract:
        BL-10 (c) item-scoped refusal facts for ``ITEM_CONTRACT_REJECTION``.
    wall_seconds:
        Total wrapper wall time. Telemetry only; the lane total previously had to
        be differenced from request and status file mtimes.

    Returns
    -------
    OperatorExitCode
        The supplied exit for direct CLI return.
    """

    write_status(
        request_path,
        exit_code=exit_code,
        classification=classification.value,
        request_sha256=request_sha256,
        attempts=attempts,
        detail=detail,
        reset_at=reset_at,
        reset_observation=reset_observation,
        provider_event_basis=provider_event_basis,
        item_contract=item_contract,
    )
    append_telemetry(
        request_path,
        {
            "event": "operator-finished",
            "exit_code": int(exit_code),
            "classification": classification.value,
            "attempts": attempts,
            "reset_at": reset_at,
            "provider_event_basis": provider_event_basis,
            "wall_seconds": None if wall_seconds is None else round(float(wall_seconds), 3),
        },
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
