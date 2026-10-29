"""Worker request, supervision, receipt projection, and diagnostic boundaries."""

from __future__ import annotations
import ast
import errno
import json
import os
import platform
import re
import secrets
import threading
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence
from menagerie.crawler.authority import (
    AuthorityDerivationError,
    EnvironmentVerificationToken,
    GenerationExhaustedError,
    per_mode_generation_span,
    ExecutableClosureV3,
    ExecutionReadManifestV3,
    RuntimeLookupDirectory,
    RuntimeMember,
    ShutdownInterruptionFact,
    WorkerLease,
    collect_executable_closure_v3,
    compile_execution_read_manifest_v3_from_closure,
    load_current_attempt_proof,
)
from menagerie.crawler.author_attempts import record_checker_findings
from menagerie.crawler.author_spend import chargeable_author_sessions
from menagerie.crawler.award_seam import award_seam_refusal_event
from menagerie.crawler.cache_registry import author_result_cache_path
from menagerie.crawler.author_dispatch import (
    AuthorEffortExhaustionClaim,
    ProposedAuthorResult,
    serialize_author_result_cache,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3,
    BACKEND_PARITY_DEFERRAL_STATUS_CODE,
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    hash_bytes,
    stable_hash,
    stamp_row_digest,
    utc_now,
)
from menagerie.crawler.host_capacity import (
    CAPACITY_DEFERRAL_DISPOSITION,
    CAPACITY_DEFERRAL_SCHEMA_VERSION,
    CapacityVerdict,
    append_capacity_deferral_row,
    assess_model_capacity,
    capacity_deferral_path,
)
from menagerie.crawler.execution_lock import global_execution_flock_path
from menagerie.crawler.gates import revalidate_recipe_revision_pin
from menagerie.crawler.text_corpus import (
    license_dispositions_from_artifact_events,
    promote_author_root,
)
from menagerie.crawler.metadata import (
    input_signature_matches_contract,
)
from menagerie.crawler.recipe import RecipeError
from menagerie.crawler.modes import classify_observed_mode_receipts
from menagerie.crawler.models import JsonObject
from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.recordio import (
    AttemptSlotResolution,
    JsonlLedger,
    LedgerConflictError,
    deterministic_attempt_id,
    resolve_attempt_slot_state,
    scan_jsonl,
)
from menagerie.crawler.reducer import (
    CanonicalReducer,
    GrantPendingSupersessionError,
    ReductionError,
    cold_forward_policy,
    expected_standard_asset,
    output_signature_error,
)
from menagerie.crawler.worker_supervisor import (
    VerifiedWorkerResult,
    WorkerUnreapableError,
    clear_worker_lease,
    current_boot_id,
    open_worker_lease,
    process_start_token,
    supervise_worker,
    verify_supervised_worker_result,
    worker_result_outer_for_diagnostics,
)
from menagerie.crawler.worker_supervisor import (
    SupervisorObservation,
    SupervisedResult,
)
from menagerie.crawler.driver_contracts import (
    ActivatedHandoffArtifact,
    AuthorArtifact,
    bound_campaign_id,
    AuthorBackoffError,
    AuthorEffortCapExceeded,
    AuthorRepairTerminal,
    DriverIntegrationError,
    DriverPaused,
    DriverShutdown,
    EnvironmentBinding,
    PublicationAuthorizationRefused,
    RetryableOperatorError,
    WorkItem,
)
from menagerie.crawler.driver_progress import (
    _is_sandbox_unavailable,
    _read_json,
    _write_driver_state,
    _write_json_atomic,
)
from menagerie.crawler.driver_models import (
    _assemble_run_model,
    _attempt_policy_satisfied,
    _detected_mode_expansion,
    _driver_failure_attempt,
    _matching_attempts,
    _matching_model_attempts,
    _recorded_matching_attempts,
    _without_ledger_fields,
)


@dataclass(frozen=True)
class _DriverReceiptDependencies:
    """Late-bound facade collaborators injected after lower modules load."""

    execution_identity: Callable[[AuthorArtifact, EnvironmentBinding, str], str]
    runner_identity: Callable[[object], str]
    checker_prompt_hash: Callable[[], str]
    injected_forward_closure_identity: Callable[[], str]


_DRIVER_RECEIPT_DEPENDENCIES: Optional[_DriverReceiptDependencies] = None


def _configure_driver_receipt_dependencies(
    *,
    execution_identity: Callable[[AuthorArtifact, EnvironmentBinding, str], str],
    runner_identity: Callable[[object], str],
    checker_prompt_hash: Callable[[], str],
    injected_forward_closure_identity: Callable[[], str],
) -> None:
    """Inject facade-owned late-bound collaborators without importing the facade."""

    global _DRIVER_RECEIPT_DEPENDENCIES
    _DRIVER_RECEIPT_DEPENDENCIES = _DriverReceiptDependencies(
        execution_identity=execution_identity,
        runner_identity=runner_identity,
        checker_prompt_hash=checker_prompt_hash,
        injected_forward_closure_identity=injected_forward_closure_identity,
    )


def _driver_receipt_dependencies() -> _DriverReceiptDependencies:
    """Return the collaborators injected by the import-compatible facade."""

    if _DRIVER_RECEIPT_DEPENDENCIES is None:
        raise DriverIntegrationError("driver receipt dependencies are not configured")
    return _DRIVER_RECEIPT_DEPENDENCIES


_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V3 "

_EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS = frozenset(
    {
        "message",
        "mode_error",
        "observed_response",
        "receipt_error",
        "response_excerpt",
        "stderr_tail",
        "stdout_tail",
        "traceback",
    }
)

_DIAGNOSTIC_REDACTION_MARKER = "externally-controlled-text-v1"

_FORBIDDEN_CACHE_ROOT_NAMES = frozenset(
    {
        ".cache",
        ".keras",
        ".paddle",
        ".torch",
        "huggingface",
        "huggingface-hub",
        "torch-hub",
        "transformers",
    }
)


def _forward_timeout_seconds(proposal: Mapping[str, Any], default_seconds: float) -> float:
    """Return a bounded producer tightening of the parent-owned forward guard.

    Parameters
    ----------
    proposal:
        Current author proposal.
    default_seconds:
        Parent-selected lane timeout and upper bound for the guard.

    Returns
    -------
    float
        Effective parent-enforced timeout. A declaration may tighten the
        guard, but cannot cross the parent-owned production floor or extend
        the parent-selected deadline.

    Raises
    ------
    DriverIntegrationError
        If a declared override is not a positive bounded integer.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    declared = (
        implementation.get("declared_timeout_seconds")
        if isinstance(implementation, Mapping)
        else None
    )
    parent_deadline = float(default_seconds)
    parent_floor = min(parent_deadline, float(DEFAULT_FORWARD_TIMEOUT_SECONDS))
    if declared is None:
        return parent_deadline
    if isinstance(declared, bool) or not isinstance(declared, int) or not 1 <= declared <= 1800:
        raise DriverIntegrationError(
            "implementation.declared_timeout_seconds must be an integer in [1, 1800]"
        )
    return max(parent_floor, min(parent_deadline, float(declared)))


#: Absolute worker memory ceiling: the historical 12 GiB cap, now a CLAMP over
#: the host-derived figure rather than an unconditional absolute (BL-12(d)).
WORKER_RSS_ABSOLUTE_CEILING_BYTES = 12 * 1024**3

#: Reserved headroom for the driver, the author lane, and the OS: at least
#: 4 GiB, and never less than a quarter of physical memory.
_WORKER_RSS_HEADROOM_FLOOR_BYTES = 4 * 1024**3

#: Smallest enforceable derived cap. A host that cannot grant this refuses at
#: lane construction instead of discovering it as anonymous signal-9s.
_WORKER_RSS_MINIMUM_CAP_BYTES = 1024**3


def derive_worker_rss_limit_bytes(
    physical_memory_bytes: Optional[int] = None,
) -> tuple[int, str]:
    """Derive the worker memory cap from the host figure with explicit headroom.

    The pre-fix cap was a hardcoded 12 GiB absolute no host figure informed
    (BL-12(d), opus W-8): 75% of the pilot host's physical RAM, and on any
    smaller fleet host a cap that could never bind, so every oversized model
    arrived as an anonymous ``runner/signal`` 9 instead of ``resource/rss-cap``.

    Parameters
    ----------
    physical_memory_bytes:
        Optional host figure override for tests. Defaults to the live host.

    Returns
    -------
    tuple[int, str]
        ``(cap_bytes, basis)`` where basis names the binding rule:
        ``"absolute-ceiling"`` or ``"host-derived"``.

    Raises
    ------
    DriverIntegrationError
        If the host cannot grant the minimum enforceable cap.
    """

    physical = (
        _physical_memory_bytes() if physical_memory_bytes is None else physical_memory_bytes
    )
    if physical <= 0:
        # No host figure available: keep the historical absolute, declared.
        return (WORKER_RSS_ABSOLUTE_CEILING_BYTES, "absolute-ceiling")
    headroom = max(_WORKER_RSS_HEADROOM_FLOOR_BYTES, physical // 4)
    derived = physical - headroom
    if derived < _WORKER_RSS_MINIMUM_CAP_BYTES:
        raise DriverIntegrationError(
            "host physical memory "
            f"{physical} bytes cannot grant the minimum enforceable worker cap "
            f"({_WORKER_RSS_MINIMUM_CAP_BYTES} bytes after {headroom} bytes headroom); "
            "refusing at admission instead of minting anonymous signal-9 terminals"
        )
    if derived >= WORKER_RSS_ABSOLUTE_CEILING_BYTES:
        return (WORKER_RSS_ABSOLUTE_CEILING_BYTES, "absolute-ceiling")
    return (derived, "host-derived")


class ModeRepairUnsatisfiable(RuntimeError):
    """The worker-demanded mode revision can never validate on this contract.

    DX-9/RE-9 (B3-XQ-5, adopted via section 6 C.12): the worker demands
    'train' while the proposal validator refuses every train declaration over
    this input contract (the batch-1 rank>=3 class). Spending author
    generations on the demanded revision is a guaranteed 2-revision burn --
    0/14 such repairs succeeded in 200a -- so the contradiction refuses typed
    at the seam, before any spend.
    """


class ModeRepairBudgetExhausted(RuntimeError):
    """The bounded detected-modes repair cap ran out without an accepted revision.

    DX-9/RE-9: budget exhaustion is an AUTHOR/INPUT fact about the repair
    lane, never a runner verdict. The cause chain carries the last rejected
    revision's error so an infrastructure-burned budget still classifies as
    weather (DX-1 composition).
    """


class WorkerSiblingBlocked(RuntimeError):
    """A sibling slot was deliberately not spawned after a deterministic kill.

    DX-7/RE-7: once one slot of a work generation observes a timeout or RSS
    watermark kill, every remaining slot of that generation would buy the
    identical kill for another full wall (the m3767 x14 class, 20.5% of 200a
    worker wall). The typed evidence row this exception seeds is derived from
    the member's own resource behavior and is never a decisive verdict.
    """


class SupervisedForwardLane:
    """Production forward lane backed by the isolated Slice-B worker supervisor."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_FORWARD_TIMEOUT_SECONDS,
        rss_limit_bytes: Optional[int] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        """Configure parent-enforced resource caps and the read-only source root.

        ``rss_limit_bytes=None`` (the production construction) derives the cap
        from the live host figure minus explicit driver/author headroom,
        clamped to the absolute ceiling; an explicit integer keeps the caller's
        cap with basis ``"configured"``.
        """

        self.timeout_seconds = timeout_seconds
        if rss_limit_bytes is None:
            self.rss_limit_bytes, self.rss_limit_basis = derive_worker_rss_limit_bytes()
        else:
            self.rss_limit_bytes = rss_limit_bytes
            self.rss_limit_basis = "configured"
        self.cwd = cwd
        self.execution_flock_path = global_execution_flock_path().resolve()

    def forward(
        self,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        cold_runs: int,
        work_root: Path,
        *,
        worker_lock_path: Optional[Path] = None,
        worker_lease_path: Optional[Path] = None,
        run_id: str = "direct-forward",
        shutdown_event: Optional[threading.Event] = None,
        lifecycle_event: Optional[Callable[[str, str, WorkerLease], None]] = None,
        attempt_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
        attempt_resolver: Optional[Callable[[int, str], Optional[Mapping[str, Any]]]] = None,
        closure: Optional[ExecutableClosureV3] = None,
        blocked_attempt_factory: Optional[
            Callable[[int, str, Mapping[str, Any]], JsonObject]
        ] = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Run each cold confirmation and fan its receipt into immutable mode attempts.

        Parameters
        ----------
        artifact, environment, cold_runs, work_root:
            Exact staged proposal, bound runtime, requested confirmations, and work root.
        worker_lock_path, worker_lease_path, run_id, shutdown_event, lifecycle_event,
        attempt_sink, attempt_resolver:
            Existing driver-owned worker lifecycle, persistence, and resume boundaries.
        closure:
            Optional closure already collected once for this artifact in the scheduling pass.
        blocked_attempt_factory:
            DX-7/RE-7 sibling suppression: called with ``(cold_index, mode,
            kill)`` to mint the typed ``blocked-by-sibling`` attempt for a slot
            deliberately not spawned after a prior slot of this same generation
            observed a deterministic resource kill (timeout or the RSS
            watermark). ``None`` keeps the historical spawn-every-slot
            behavior -- a two-mode member then buys a second full timeout.

        Returns
        -------
        collections.abc.Sequence[collections.abc.Mapping[str, Any]]
            Authenticated immutable mode attempts.
        """

        if cold_runs < 1:
            raise ValueError("cold_runs must be positive")
        proposal = artifact.proposal
        stable_id = str(proposal["stable_id"])
        authority = environment.environment_authority
        cache = environment.environment_authority_cache
        if authority is None or cache is None:
            raise DriverIntegrationError(
                "live supervised execution requires a lifecycle-owned environment authority"
            )
        if closure is None:
            with cache.currentness_pass(authority) as verification_token:
                closure = _collect_worker_executable_closure(
                    artifact,
                    environment,
                    verification_token=verification_token,
                )
        execution_identity = _driver_receipt_dependencies().execution_identity(
            artifact, environment, closure.identity
        )
        rung = proposal.get("proposed_facts", {}).get("source_resolution", {}).get("rung")
        reducer_policy = cold_forward_policy(stable_id, rung)
        required_cold_runs = reducer_policy.required_cold_forwards
        effective_timeout = _forward_timeout_seconds(proposal, self.timeout_seconds)
        attempts: list[JsonObject] = []
        observed_receipts: dict[int, dict[str, Mapping[str, object]]] = defaultdict(dict)
        lock_path = worker_lock_path or work_root / "locks" / "worker.lock"
        lease_path = worker_lease_path or work_root / "locks" / "worker-lease.json"
        shutdown = shutdown_event or threading.Event()
        # These modes are gate-authoritative. A worker-discovered expansion is a
        # contract failure below and must be re-proposed/re-gated before it can run.
        modes = tuple(
            str(value) for value in proposal["proposed_facts"]["modes"]["meaningful_modes"]
        )
        deterministic_kill: Optional[JsonObject] = None
        for cold_index in range(required_cold_runs):
            for mode in modes:
                if shutdown.is_set():
                    raise DriverShutdown(
                        ShutdownInterruptionFact(
                            invocation_id=run_id,
                            admission_boundary="pre-slot-resolution",
                            stable_id=stable_id,
                            work_id=str(proposal["work_id"]),
                            execution_identity=execution_identity,
                            request_identity=None,
                            lease_id=None,
                            child_pid=None,
                            child_start_token=None,
                            child_pgid=None,
                            signal=None,
                            parent_observation=None,
                            partial_receipt=None,
                        )
                    )
                slot_retry_generation = 0
                resolution = (
                    attempt_resolver(cold_index, mode) if attempt_resolver is not None else None
                )
                resolved_attempt: Optional[Mapping[str, Any]] = None
                if isinstance(resolution, AttemptSlotResolution):
                    resolved_attempt = resolution.record
                    slot_retry_generation = resolution.retry_generation
                elif resolution is not None:
                    resolved_attempt = resolution
                if resolved_attempt is not None:
                    resolved_error = resolved_attempt.get("error")
                    if (
                        isinstance(resolved_error, Mapping)
                        and resolved_error.get("stage") == "resource"
                        and resolved_error.get("reason_code") in {"timeout", "rss-cap"}
                    ):
                        # A resumed generation replays its recorded kill; the
                        # remaining unspawned siblings stay suppressed exactly
                        # as they would have been live (DX-7/RE-7).
                        deterministic_kill = {
                            "kind": str(resolved_error.get("reason_code")),
                            "cold_index": cold_index,
                            "mode": mode,
                            "resolved_attempt_id": resolved_attempt.get("attempt_id"),
                        }
                    attempts.append(dict(resolved_attempt))
                    continue
                if deterministic_kill is not None and blocked_attempt_factory is not None:
                    # DX-7/RE-7: a prior slot of this same generation already
                    # observed a deterministic resource kill; spawning this
                    # sibling buys the identical kill for another full wall
                    # (the m3767 waste class). The slot records its typed
                    # blocked-by-sibling evidence instead.
                    blocked = blocked_attempt_factory(
                        cold_index, mode, dict(deterministic_kill)
                    )
                    if attempt_sink is not None:
                        attempt_sink(blocked)
                    attempts.append(dict(blocked))
                    continue
                root = work_root / stable_id / "forward" / f"cold-{cold_index + 1}" / mode
                request_path = root / "request.json"
                receipt_path = root / "result" / "receipt.json"
                with cache.spawn_verification(authority) as spawn_verification:
                    manifest = _compile_worker_read_manifest(
                        artifact,
                        environment,
                        execution_identity,
                        closure=closure,
                        verification_token=spawn_verification,
                    )
                    request = _worker_request(
                        artifact,
                        root,
                        receipt_path,
                        execution_identity,
                        manifest,
                        cold_index,
                        mode,
                    )
                    _write_json_atomic(request_path, request)
                    request_identity = hash_bytes(request_path.read_bytes())
                    driver_token = process_start_token(os.getpid())
                    if driver_token is None:
                        raise DriverIntegrationError("cannot establish driver process identity")
                    opened = datetime.now(timezone.utc)
                    lease = WorkerLease(
                        lease_id=stable_hash(
                            {
                                "run_id": run_id,
                                "stable_id": stable_id,
                                "work_id": proposal["work_id"],
                                "execution_identity": execution_identity,
                                "cold_index": cold_index,
                                "mode": mode,
                                "request_identity": request_identity,
                            }
                        ),
                        nonce=str(request["request_nonce"]),
                        run_id=run_id,
                        stable_id=stable_id,
                        work_id=str(proposal["work_id"]),
                        request_identity=request_identity,
                        execution_identity=execution_identity,
                        boot_id=current_boot_id(),
                        driver_pid=os.getpid(),
                        driver_start_token=driver_token,
                        child_pid=None,
                        child_start_token=None,
                        child_pgid=None,
                        receipt_path=receipt_path,
                        opened_at=opened.isoformat().replace("+00:00", "Z"),
                        deadline_at=(opened + timedelta(seconds=effective_timeout))
                        .isoformat()
                        .replace("+00:00", "Z"),
                    )

                    def on_opened(value: WorkerLease) -> None:
                        """Append the lock-ordered opened lifecycle event."""

                        if lifecycle_event is not None:
                            lifecycle_event(
                                OperationalEventKind.WORKER_LEASE_OPENED.value,
                                OperationalEventStatus.WORKER_LEASE_OPEN.value,
                                value,
                            )

                    handle = open_worker_lease(
                        lock_path,
                        lease_path,
                        lease,
                        global_lock_path=self.execution_flock_path,
                        on_lock_acquired=on_opened,
                        shutdown_event=shutdown,
                    )

                    def on_started(value: WorkerLease) -> None:
                        """Append the exact child-start lifecycle event."""

                        if lifecycle_event is not None:
                            lifecycle_event(
                                OperationalEventKind.WORKER_LEASE_STARTED.value,
                                OperationalEventStatus.WORKER_LEASE_ACTIVE.value,
                                value,
                            )

                    try:
                        result = supervise_worker(
                            request_path,
                            receipt_path,
                            root / "supervisor",
                            timeout_seconds=effective_timeout,
                            rss_limit_bytes=self.rss_limit_bytes,
                            cwd=self.cwd,
                            execution_read_manifest=manifest,
                            worker_lease_handle=handle,
                            shutdown_event=shutdown,
                            on_lease_started=on_started,
                            verification_token=spawn_verification,
                        )
                    except BaseException:
                        if lifecycle_event is not None:
                            lifecycle_event(
                                OperationalEventKind.WORKER_LEASE_CLOSED.value,
                                OperationalEventStatus.WORKER_LEASE_CLOSED.value,
                                handle.lease,
                            )
                        clear_worker_lease(handle)
                        raise
                if result.observation.shutdown_requested:
                    if lifecycle_event is not None:
                        lifecycle_event(
                            OperationalEventKind.WORKER_LEASE_CLOSED.value,
                            OperationalEventStatus.WORKER_LEASE_CLOSED.value,
                            handle.lease,
                        )
                    interrupted_lease = handle.lease
                    clear_worker_lease(handle)
                    partial_receipt = worker_result_outer_for_diagnostics(result)
                    raise DriverShutdown(
                        ShutdownInterruptionFact(
                            invocation_id=run_id,
                            admission_boundary="worker-supervision",
                            stable_id=stable_id,
                            work_id=str(proposal["work_id"]),
                            execution_identity=execution_identity,
                            request_identity=request_identity,
                            lease_id=interrupted_lease.lease_id,
                            child_pid=interrupted_lease.child_pid,
                            child_start_token=interrupted_lease.child_start_token,
                            child_pgid=interrupted_lease.child_pgid,
                            signal=result.observation.signal_number,
                            parent_observation=result.observation.to_dict(),
                            partial_receipt=(
                                dict(partial_receipt)
                                if isinstance(partial_receipt, Mapping)
                                else None
                            ),
                        )
                    )
                if result.observation.timed_out or result.observation.rss_exceeded:
                    deterministic_kill = {
                        "kind": "timeout" if result.observation.timed_out else "rss-cap",
                        "cold_index": cold_index,
                        "mode": mode,
                        "wall_seconds": result.observation.wall_seconds,
                        "peak_rss_bytes": result.observation.peak_rss_bytes,
                        "rss_kill_threshold_bytes": getattr(
                            result.observation, "rss_kill_threshold_bytes", 0
                        ),
                    }
                generated = _attempts_from_supervised(
                    artifact,
                    result,
                    environment,
                    execution_identity,
                    cold_index,
                    effective_timeout,
                    self.rss_limit_bytes,
                    requested_mode=mode,
                    execution_read_manifest_identity=manifest.manifest_id,
                    diagnostics_root=_diagnostics_root_for_work_root(work_root),
                    rss_limit_basis=self.rss_limit_basis,
                    slot_retry_generation=slot_retry_generation,
                )
                for attempt in generated:
                    if attempt_sink is not None:
                        attempt_sink(attempt)
                    attempts.append(dict(attempt))
                if lifecycle_event is not None:
                    lifecycle_event(
                        OperationalEventKind.WORKER_LEASE_CLOSED.value,
                        OperationalEventStatus.WORKER_LEASE_CLOSED.value,
                        handle.lease,
                    )
                clear_worker_lease(handle)
                verified, _verification_error = _verified_worker_result(
                    result,
                    proposal,
                    execution_identity,
                    requested_mode=mode,
                )
                raw_per_mode = verified.diagnostic.get("per_mode") if verified is not None else None
                raw_mode_receipt = (
                    raw_per_mode.get(mode) if isinstance(raw_per_mode, Mapping) else None
                )
                if isinstance(raw_mode_receipt, Mapping):
                    observed_receipts[cold_index][mode] = raw_mode_receipt

        observation_failures: list[JsonObject] = []
        for mode in modes:
            signatures = [
                observed_receipts[index][mode].get("output_signature")
                for index in range(required_cold_runs)
                if mode in observed_receipts[index]
            ]
            if len(signatures) == required_cold_runs and any(
                signature != signatures[0] for signature in signatures[1:]
            ):
                observation_failures.append(
                    {
                        "kind": "cold-forward-nondeterminism",
                        "mode": mode,
                        "required_cold_forwards": required_cold_runs,
                    }
                )
        declared_divergence = str(
            proposal.get("proposed_facts", {}).get("modes", {}).get("train_eval_divergence", "none")
        )
        if set(modes) == {"train", "eval"}:
            for cold_index in range(required_cold_runs):
                per_mode = observed_receipts[cold_index]
                if not {"train", "eval"}.issubset(per_mode):
                    continue
                divergence = classify_observed_mode_receipts(per_mode["train"], per_mode["eval"])
                signatures_differ = per_mode["train"].get("output_signature") != per_mode[
                    "eval"
                ].get("output_signature")
                contradicted = (
                    (declared_divergence == "structural" and not signatures_differ)
                    or (declared_divergence != "structural" and signatures_differ)
                    or (divergence is not None and divergence.classification != declared_divergence)
                )
                if contradicted:
                    observation_failures.append(
                        {
                            "kind": "train-eval-divergence-mismatch",
                            "cold_index": cold_index,
                            "declared": declared_divergence,
                            "observed": (
                                divergence.classification
                                if divergence is not None
                                else "signature-compatible"
                            ),
                        }
                    )
        elif declared_divergence != "none":
            observation_failures.append(
                {
                    "kind": "single-mode-divergence-mismatch",
                    "declared": declared_divergence,
                }
            )
        if observation_failures:
            failure = _attempt_error_fields(
                "forward",
                "confirmation-mismatch",
                None,
                "mechanical forward observations contradict the accepted run contract",
                native_crash=False,
                details={"observations": observation_failures},
            )
            failure["root_cause_fingerprint"] = stable_hash(failure)
            for index, attempt in enumerate(attempts):
                if attempt.get("result") != "succeeded":
                    continue
                attempt["result"] = "failed"
                attempt["error"] = deepcopy(failure)
                attempts[index] = _redact_attempt_diagnostics(
                    attempt,
                    None,
                    _diagnostics_root_for_work_root(work_root),
                )
        return tuple(attempts)


def _collect_worker_executable_closure(
    artifact: AuthorArtifact,
    environment: EnvironmentBinding,
    *,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> ExecutableClosureV3:
    """Collect the sole executable closure before execution identity derivation.

    Parameters
    ----------
    artifact:
        Privately staged proposed result.
    environment:
        Exact materialized runtime generation.
    verification_token:
        Optional cache-created proof shared by the enclosing pass or spawn.
    Returns
    -------
    ExecutableClosureV3
        Closed exact code/crawler partitions and one sealed environment unit.
    """

    if environment.environment_authority is None:
        raise DriverIntegrationError(
            "live supervised execution requires a sealed EnvironmentAuthorityV1 binding"
        )

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    raw_manifest = implementation.get("code_manifest", [])
    if not isinstance(raw_manifest, list):
        raise DriverIntegrationError("implementation code manifest is malformed")
    members: list[RuntimeMember] = []
    for row in raw_manifest:
        if not isinstance(row, Mapping):
            raise DriverIntegrationError("implementation code manifest row is malformed")
        path = (artifact.model_dir / str(row.get("path", ""))).resolve()
        if path.suffix in {".py", ".pyi", ".pyx"}:
            kind = "python-source"
        elif path.suffix in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}:
            kind = "native-source"
        elif path.suffix in {".a", ".dylib", ".pyd", ".so"}:
            kind = "native-library"
        elif path.suffix == ".pyc":
            kind = "python-bytecode"
        else:
            raise DriverIntegrationError(f"execution code member has forbidden suffix: {path}")
        members.append(
            RuntimeMember(
                path=path,
                sha256=str(row.get("sha256")),
                kind=kind,
                provenance="accepted-model-code-manifest",
            )
        )
    selected = (
        expected_standard_asset(facts["external_metadata"]["modality"])
        if implementation.get("recipe_type") == "declarative-library"
        else None
    )
    asset = (
        (Path(selected["path"]), selected["sha256"], selected["asset_id"])
        if selected is not None
        else None
    )
    code_identity = stable_hash(raw_manifest)
    worker_members = tuple(
        RuntimeMember(
            path=path,
            sha256=hash_bytes(path.read_bytes()),
            kind=_runtime_member_kind(path, environment.python_executable.resolve()),
            provenance="crawler-worker-import-closure",
        )
        for path in _crawler_worker_runtime_paths()
    )
    lookup_candidates = {
        Path(__file__).resolve().parents[2],
        environment.prefix.resolve(),
        *(path.resolve() for path in environment.prefix.glob("lib/python*")),
        *(path.resolve() for path in environment.prefix.glob("lib/python*/site-packages")),
    }
    lookup_directories = tuple(
        RuntimeLookupDirectory(path=path, provenance="import-lookup-scaffold")
        for path in sorted(lookup_candidates, key=str)
        if path.is_dir() and not path.is_symlink()
    )
    return collect_executable_closure_v3(
        code_manifest_identity=code_identity,
        environment_authority=environment.environment_authority,
        code_members=tuple(members),
        worker_members=worker_members,
        standard_input_asset=asset,
        lookup_directories=lookup_directories,
        verification_token=verification_token,
    )


def _compile_worker_read_manifest(
    artifact: AuthorArtifact,
    environment: EnvironmentBinding,
    execution_identity: str,
    *,
    closure: Optional[ExecutableClosureV3] = None,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> ExecutionReadManifestV3:
    """Bind a verified pre-identity closure into the final worker manifest.

    Parameters
    ----------
    artifact, environment:
        Exact executable artifact and materialized environment generation.
    execution_identity:
        Reducer-compatible execution identity derived from the closure.
    closure:
        Optional already-collected closure reused across meaningful modes.
    verification_token:
        Optional cache-created proof shared across every compilation consumer.

    Returns
    -------
    ExecutionReadManifestV3
        Final exact model/crawler/asset plus sealed-environment capability.
    """

    proposal = artifact.proposal
    collected = closure or _collect_worker_executable_closure(
        artifact,
        environment,
        verification_token=verification_token,
    )
    return compile_execution_read_manifest_v3_from_closure(
        collected,
        stable_id=str(proposal["stable_id"]),
        work_id=str(proposal["work_id"]),
        execution_identity=execution_identity,
        verification_token=verification_token,
    )


def _runtime_member_kind(path: Path, interpreter: Path) -> str:
    """Classify one exact runtime file for execution-manifest v2.

    Parameters
    ----------
    path:
        Canonical regular runtime member.
    interpreter:
        Canonical selected environment interpreter.

    Returns
    -------
    str
        Closed v2 runtime-member kind.
    """

    if path == interpreter:
        return "interpreter"
    suffix = path.suffix.lower()
    if suffix in {".py", ".pyi", ".pyx"}:
        return "python-source"
    if suffix == ".pyc":
        return "python-bytecode"
    if suffix in {".pyd", ".so"}:
        return "native-extension"
    if suffix == ".dylib" or ".so." in path.name.lower():
        return "native-library"
    if (
        path.name
        in {
            "INSTALLER",
            "METADATA",
            "RECORD",
            "WHEEL",
            "entry_points.txt",
            "pyvenv.cfg",
        }
        or path.parent.name == "conda-meta"
    ):
        return "import-metadata"
    return "package-data"


def _crawler_worker_runtime_paths() -> tuple[Path, ...]:
    """Collect the exact recursive crawler-local worker import closure.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Canonical Python files seeded by the worker/supervisor/policy entry points.
    """

    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[1]
    pending = [
        package_root / "worker.py",
        package_root / "worker_supervisor.py",
        package_root / "policy.py",
    ]
    members: set[Path] = {
        repository_root / "menagerie" / "__init__.py",
        package_root / "__init__.py",
    }
    while pending:
        path = pending.pop().resolve()
        if path in members or not path.is_file():
            continue
        members.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise DriverIntegrationError(f"worker runtime source cannot be parsed: {path}") from exc
        module_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                module_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                module_names.append(node.module)
        for module_name in module_names:
            if not module_name.startswith("menagerie.crawler"):
                continue
            parts = module_name.split(".")[2:]
            candidate = package_root.joinpath(*parts).with_suffix(".py")
            if candidate.is_file():
                pending.append(candidate)
    return tuple(sorted(members, key=str))


def _worker_request(
    artifact: AuthorArtifact,
    scratch_root: Path,
    receipt_path: Path,
    execution_identity: str,
    execution_manifest: ExecutionReadManifestV3,
    cold_index: int,
    mode: str,
) -> JsonObject:
    """Build one explicit-mode v3 request bound to an out-of-band manifest."""

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    input_contract = deepcopy(dict(facts["input_contract"]))
    if "code_path" in input_contract:
        raise DriverIntegrationError("v3 execution forbids input_contract.code_path presence")
    builder_symbol = input_contract.get("builder_symbol")
    if not isinstance(builder_symbol, str) or not re.fullmatch(
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", builder_symbol
    ):
        raise DriverIntegrationError("worker input_contract.builder_symbol is malformed")
    non_tensor_values = input_contract.get("non_tensor_values")
    if not isinstance(non_tensor_values, list):
        raise DriverIntegrationError("worker input_contract.non_tensor_values is malformed")
    for leaf in non_tensor_values:
        value = leaf.get("value") if isinstance(leaf, Mapping) else None
        if isinstance(value, str):
            possible_path = Path(value)
            if possible_path.is_absolute() or ".." in possible_path.parts:
                raise DriverIntegrationError(
                    "worker refuses path-like non-tensor values outside the model root"
                )
            value_type = str(leaf.get("type", "")).casefold().replace("_", "-")
            if value_type in {"file", "file-path", "filepath", "path", "pathlib.path"}:
                resolved_value = (artifact.model_dir / possible_path).resolve()
                if not resolved_value.is_relative_to(artifact.model_dir.resolve()) or not (
                    resolved_value.is_file()
                ):
                    raise DriverIntegrationError(
                        "worker path-valued non-tensor input is not a model-local regular file"
                    )
    if implementation["recipe_type"] == "declarative-library":
        # Admission-time pin revalidation: a recipe_revision that does not
        # reproduce from THIS proposal's own components is refused before any
        # worker spawns, and the typed RecipeError routes to the bounded
        # author-repair lane instead of surfacing as a masked worker exception.
        revalidate_recipe_revision_pin(proposal)
        recipe: JsonObject = {
            "kind": "declarative-library",
            "recipe": implementation["library_recipe"],
        }
    else:
        code_path = Path(str(implementation["code_path"]))
        if code_path.is_absolute():
            raise DriverIntegrationError(
                "worker refuses a legacy absolute adapter path; re-propose and re-gate"
            )
        code_path = (artifact.model_dir / code_path).resolve()
        if not code_path.is_relative_to(artifact.model_dir.resolve()):
            raise DriverIntegrationError("worker adapter path escapes private custody")
        manifest_rows: list[JsonObject] = []
        raw_manifest = implementation.get("code_manifest")
        if not isinstance(raw_manifest, list) or not raw_manifest:
            raise DriverIntegrationError("worker adapter code manifest is missing")
        for member in raw_manifest:
            if not isinstance(member, Mapping) or not isinstance(member.get("path"), str):
                raise DriverIntegrationError("worker adapter code manifest path is malformed")
            member_path = Path(str(member["path"]))
            if member_path.is_absolute():
                raise DriverIntegrationError("worker adapter code manifest path must be relative")
            resolved_member = (artifact.model_dir / member_path).resolve()
            if (
                not resolved_member.is_relative_to(artifact.model_dir.resolve())
                or not resolved_member.is_file()
            ):
                raise DriverIntegrationError(
                    "worker adapter code manifest path is not a private regular file"
                )
            manifest_rows.append(
                {
                    "path": str(resolved_member),
                    "identity_path": member["path"],
                    "sha256": member["sha256"],
                }
            )
        try:
            adapter_bytes = code_path.read_bytes()
        except OSError as exc:
            raise DriverIntegrationError(
                f"worker adapter bytes are unreadable at admission: {exc}"
            ) from exc
        # Admission-time pin revalidation (rung-11 census, defect class 4): the
        # facts-flavor pin must reproduce from THIS proposal's own components,
        # the staged bytes must reproduce the accepted digest, and the derived
        # ADAPTER-flavor revision rides the request so the loader enforces a
        # coherent pin -- the accepted-facts hash is a structurally different
        # flavor that mismatched on every typed adapter and pushed the worker
        # into the permissive retry that masked the real failure.
        loader_recipe_revision = revalidate_recipe_revision_pin(
            proposal,
            adapter_bytes=adapter_bytes,
            adapter_path_name=code_path.name,
        )
        recipe = {
            "kind": "typed-adapter",
            "path": str(code_path),
            "adapter_sha256": implementation["code_sha256"],
            "recipe_revision": loader_recipe_revision,
            "code_manifest": manifest_rows,
            "code_manifest_sha256": stable_hash(implementation["code_manifest"]),
        }
    input_seed = int(input_contract.get("seed", 0))
    try:
        selected_asset = (
            expected_standard_asset(facts["external_metadata"]["modality"])
            if implementation["recipe_type"] == "declarative-library"
            else None
        )
    except ValueError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    standard_input_asset = (
        {"sha256": selected_asset["sha256"], "asset_id": selected_asset["asset_id"]}
        if selected_asset is not None
        else None
    )
    return {
        "protocol_version": "menagerie.crawler.worker-request.v3",
        "stable_id": proposal["stable_id"],
        "work_id": proposal["work_id"],
        "request_nonce": secrets.token_hex(32),
        "execution_read_manifest_identity": execution_manifest.manifest_id,
        "code_manifest_identity": execution_manifest.code_manifest_identity,
        "input_identity": stable_hash(
            {
                "input_contract": input_contract,
                "standard_input_asset": standard_input_asset,
                "seed": input_seed,
            }
        ),
        "recipe": recipe,
        "modality": facts["external_metadata"]["modality"],
        "input_spec": None,
        "input_contract": input_contract,
        "scratch_root": str(scratch_root),
        "receipt_path": str(receipt_path),
        "seed": input_seed,
        "input_seed": input_seed,
        "standard_input_asset": standard_input_asset,
        "device": implementation["device_policy"],
        "framework": implementation["run_framework"],
        "mode": mode,
        "meaningful_modes": list(facts["modes"]["meaningful_modes"]),
        "source_identity": proposal["source_identity"],
        "recipe_revision": proposal["recipe_revision"],
        "recipe_identity_payload": {
            "implementation": {
                key: value for key, value in implementation.items() if key != "recipe_revision"
            },
            "input_contract": deepcopy(input_contract),
            "modes": {
                "meaningful_modes": list(facts["modes"]["meaningful_modes"]),
            },
        },
        "execution_identity": execution_identity,
    }


def _expected_adapter_sha256(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the accepted adapter digest, or null for declarative recipes.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Exact accepted code digest for typed adapters.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    if not isinstance(implementation, Mapping):
        return None
    if implementation.get("recipe_type") == "declarative-library":
        return None
    value = implementation.get("code_sha256")
    return str(value) if isinstance(value, str) else None


def _expected_code_manifest_sha256(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the identity-bound recursive code-manifest digest.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Aggregate manifest digest, including the canonical empty manifest for
        declarative recipes, or ``None`` for malformed/unbound adapters.
    """

    implementation = proposal.get("proposed_facts", {}).get("implementation", {})
    manifest = implementation.get("code_manifest") if isinstance(implementation, Mapping) else None
    if isinstance(manifest, list):
        return stable_hash(manifest)
    if isinstance(implementation, Mapping) and implementation.get("recipe_type") == (
        "declarative-library"
    ):
        return stable_hash([])
    return None


def _expected_input_asset_sha256(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the digest of the selected request-bound standard input asset.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Selected standard-asset digest, or ``None`` for typed dummy calls and
        random fallback.
    """

    facts = proposal.get("proposed_facts", {})
    implementation = facts.get("implementation", {}) if isinstance(facts, Mapping) else {}
    if not isinstance(implementation, Mapping) or implementation.get("recipe_type") != (
        "declarative-library"
    ):
        return None
    external = facts.get("external_metadata", {}) if isinstance(facts, Mapping) else {}
    modality = external.get("modality") if isinstance(external, Mapping) else None
    selected = expected_standard_asset(modality)
    return selected["sha256"] if selected is not None else None


def _expected_input_asset_id(proposal: Mapping[str, Any]) -> Optional[str]:
    """Return the content-addressed selected standard input identifier.

    Parameters
    ----------
    proposal:
        Current author proposal bound into a worker request.

    Returns
    -------
    str | None
        Expected worker ``input_asset`` value.
    """

    facts = proposal.get("proposed_facts", {})
    implementation = facts.get("implementation", {}) if isinstance(facts, Mapping) else {}
    if not isinstance(implementation, Mapping) or implementation.get("recipe_type") != (
        "declarative-library"
    ):
        return None
    external = facts.get("external_metadata", {}) if isinstance(facts, Mapping) else {}
    modality = external.get("modality") if isinstance(external, Mapping) else None
    selected = expected_standard_asset(modality)
    return selected["asset_id"] if selected is not None else None


def _verified_worker_result(
    result: SupervisedResult,
    proposal: Mapping[str, Any],
    execution_identity: str,
    *,
    requested_mode: Optional[str],
) -> tuple[Optional[VerifiedWorkerResult], Optional[str]]:
    """Project a live worker-result.v3 against current driver associations.

    Parameters
    ----------
    result:
        Parent-observed live worker result.
    proposal:
        Current accepted author proposal.
    execution_identity:
        Parent-computed execution identity.
    requested_mode:
        Explicit mode assigned to the subprocess, or ``None`` for all modes.

    Returns
    -------
    tuple[VerifiedWorkerResult | None, str | None]
        Typed projection or its closed protocol error.
    """

    return verify_supervised_worker_result(
        result,
        expected_stable_id=str(proposal.get("stable_id")),
        expected_work_id=str(proposal.get("work_id")),
        expected_source_identity=str(proposal.get("source_identity")),
        expected_recipe_revision=str(proposal.get("recipe_revision")),
        expected_execution_identity=execution_identity,
        expected_code_manifest_identity=_expected_code_manifest_sha256(proposal),
        requested_mode=requested_mode,
    )


def _attempts_from_supervised(
    artifact: AuthorArtifact,
    result: SupervisedResult,
    environment: EnvironmentBinding,
    execution_identity: str,
    cold_index: int,
    timeout_seconds: float,
    rss_limit_bytes: int,
    *,
    requested_mode: Optional[str] = None,
    execution_read_manifest_identity: Optional[str] = None,
    diagnostics_root: Optional[Path] = None,
    rss_limit_basis: str = "configured",
    slot_retry_generation: int = 0,
) -> tuple[JsonObject, ...]:
    """Convert one parent observation and honest receipt into per-mode attempts.

    Parameters
    ----------
    artifact, result, environment, execution_identity, cold_index:
        Bound worker request, parent result, environment, and run identity facts.
    timeout_seconds, rss_limit_bytes:
        Parent-enforced resource limits.
    requested_mode:
        Single mode isolated in this subprocess, when applicable.
    diagnostics_root:
        Gitignored local root for exact external-text diagnostic sidecars. Production
        callers always provide ``.crawl-local/diagnostics``; tests that only exercise
        receipt classification may omit it when all controlled fields are empty.
    slot_retry_generation:
        DX-10/RE-11 typed retry generation past consumed non-model failure
        rows; folded into the deterministic attempt identity and carried on
        ``retries.slot_retry_generation`` so the identity stays recomputable.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Canonical attempts containing only redacted references to model-controlled text.
    """

    proposal = artifact.proposal
    facts = proposal["proposed_facts"]
    verified, verification_error = _verified_worker_result(
        result,
        proposal,
        execution_identity,
        requested_mode=requested_mode,
    )
    receipt = deepcopy(verified.diagnostic) if verified is not None else {}
    policy_value = receipt.get("policy_observation", {})
    policy = dict(policy_value) if isinstance(policy_value, Mapping) else {}
    policy["cache_read_attempted"] = bool(policy.get("cache_read_attempted")) or (
        _parent_cache_read_attempted(policy)
    )
    receipt["policy_observation"] = policy
    envelope_error = verification_error or _receipt_envelope_error(
        result, proposal, execution_identity, requested_mode=requested_mode
    )
    if policy.get("cache_read_attempted"):
        envelope_error = "failed:policy-cache-read"
    effective_result = (
        result
        if envelope_error is None
        else SupervisedResult(result.observation, None, envelope_error, None)
    )
    per_mode = receipt.get("per_mode", {})
    receipt_modes = receipt.get("meaningful_modes", [])
    detected_modes = tuple(
        str(value)
        for value in receipt.get("detected_meaningful_modes", [])
        if isinstance(value, str)
    )
    proposal_mode_set = {str(value) for value in facts["modes"]["meaningful_modes"]}
    missing_proposal_modes = tuple(
        mode for mode in ("train", "eval") if mode in set(detected_modes) - proposal_mode_set
    )
    modes = (
        (requested_mode,)
        if requested_mode is not None
        else tuple(
            dict.fromkeys(
                [
                    *(str(value) for value in facts["modes"]["meaningful_modes"]),
                    *(str(value) for value in receipt_modes if isinstance(receipt_modes, list)),
                ]
            )
        )
    )
    attempts: list[JsonObject] = []
    declared_modes = tuple(str(value) for value in facts["modes"]["meaningful_modes"])
    for mode in modes:
        mode_index = declared_modes.index(mode) if mode in declared_modes else len(declared_modes)
        mode_receipt = per_mode.get(mode, {}) if isinstance(per_mode, Mapping) else {}
        succeeded = bool(
            envelope_error is None
            and verified is not None
            and result.observation.exit_code == 0
            and result.observation.signal_number is None
            and verified.raw_award_receipt is not None
            and verified.raw_observation is not None
            and verified.parent_attestation is not None
            and mode_receipt.get("constructor_started")
            and mode_receipt.get("constructor_completed")
            and mode_receipt.get("input_completed")
            and mode_receipt.get("forward_started")
            and mode_receipt.get("forward_completed")
            and input_signature_matches_contract(
                mode_receipt.get("input_signature"), facts["input_contract"]
            )
        )
        attempt_id = deterministic_attempt_id(
            work_id=str(proposal["work_id"]),
            execution_identity=execution_identity,
            cold_index=cold_index,
            mode=mode,
            retry_generation=slot_retry_generation,
        )
        observation = result.observation
        error: Optional[JsonObject] = None
        attempt_stage = "forward"
        attempt_mode: Optional[str] = mode
        if not succeeded:
            if missing_proposal_modes:
                failure = _attempt_error_fields(
                    "input",
                    "contract-invalid",
                    None,
                    "worker detected meaningful modes absent from the gated proposal",
                    native_crash=False,
                    details={
                        "route": "recipe-and-gate-revision-required",
                        "proposal_meaningful_modes": list(declared_modes),
                        "detected_meaningful_modes": list(detected_modes),
                        "missing_proposal_modes": list(missing_proposal_modes),
                    },
                )
            else:
                failure = _supervised_failure(
                    effective_result,
                    receipt,
                    mode_receipt,
                    policy,
                    worker_result_present=verified is not None,
                )
            attempt_stage = failure["stage"]
            attempt_mode = mode if attempt_stage == "forward" else None
            error = {
                **failure,
                "root_cause_fingerprint": stable_hash(failure),
            }
        worker_receipt = {
            "present": verified is not None,
            "receipt_sha256": (verified.result_sha256 if verified is not None else None),
            "observed_recipe_revision": receipt.get("observed_recipe_revision"),
            "observed_adapter_sha256": receipt.get("observed_adapter_sha256"),
            "observed_code_manifest_sha256": receipt.get("observed_code_manifest_sha256"),
            "observed_input_asset_sha256": receipt.get("observed_input_asset_sha256"),
            "constructor_started": bool(mode_receipt.get("constructor_started")),
            "constructor_completed": bool(mode_receipt.get("constructor_completed")),
            "input_completed": bool(mode_receipt.get("input_completed")),
            "forward_started": bool(mode_receipt.get("forward_started")),
            "forward_completed": bool(mode_receipt.get("forward_completed")),
            "mode": mode,
            "input_signature": mode_receipt.get("input_signature"),
            "output_signature": mode_receipt.get("output_signature"),
            "output_value_sha256": mode_receipt.get("output_value_sha256"),
            "input_kind": mode_receipt.get("input_kind"),
            "input_asset": mode_receipt.get("input_asset"),
            "input_note": str(mode_receipt.get("input_note") or "worker receipt unavailable"),
            "parameter_count_total": mode_receipt.get("parameter_count_total"),
            "parameter_count_trainable": mode_receipt.get("parameter_count_trainable"),
            "native_framework": mode_receipt.get("native_framework"),
            "delegated_method": mode_receipt.get("delegated_method"),
            "constructor_seconds": mode_receipt.get("constructor_seconds"),
            "forward_seconds": mode_receipt.get("forward_seconds"),
        }
        if succeeded:
            if verified is None or verified.raw_observation is None:
                raise DriverIntegrationError("successful worker result lacks a raw award receipt")
            worker_receipt = deepcopy(verified.raw_observation)
        parent_attestation = (
            deepcopy(verified.parent_attestation)
            if verified is not None
            else deepcopy(result.parent_attestation)
        )
        attempt: JsonObject = {
            "schema_version": ATTEMPT_SCHEMA_VERSION_V3,
            "attempt_id": attempt_id,
            "work_id": proposal["work_id"],
            "stable_id": proposal["stable_id"],
            "attempt_no": cold_index * len(declared_modes) + mode_index + 1,
            "parent_attempt_id": None,
            "actor": "worker",
            "stage": attempt_stage,
            "mode": attempt_mode,
            "started_at": (
                parent_attestation.get("started_at")
                if isinstance(parent_attestation, Mapping)
                else proposal["created_at"]
            ),
            "finished_at": (
                parent_attestation.get("finished_at")
                if isinstance(parent_attestation, Mapping)
                else utc_now()
            ),
            "result": "succeeded" if succeeded else "failed",
            "attempted_rungs": [facts["source_resolution"]["rung"]],
            "retries": {
                "stage_attempt": cold_index + 1,
                # Generation zero omits the key so historical rows stay
                # byte-identical (DX-10: no gratuitous mixed-era delta).
                **(
                    {"slot_retry_generation": slot_retry_generation}
                    if slot_retry_generation > 0
                    else {}
                ),
                "root_cause_repeat": 0,
                "author_round": 1,
                "gate_round": 1,
            },
            "identities": {
                "source": proposal["source_identity"],
                "evidence": proposal["evidence_identity"],
                "recipe": proposal["recipe_revision"],
                "environment": environment.env_generation,
                "execution": execution_identity,
                "runner": _driver_receipt_dependencies().runner_identity(
                    facts["external_metadata"]["modality"]
                ),
                "author_prompt": proposal["author"]["prompt_sha256"],
                "checker_prompt": _driver_receipt_dependencies().checker_prompt_hash(),
            },
            "environment": {
                "family": environment.family,
                "target": environment.target,
                "env_id": str(environment.prefix),
                "lock_sha256": environment.lock_sha256,
                "resolved_export_sha256": environment.resolved_export_sha256,
                "python": environment.python_version,
                "packages_manifest_sha256": environment.packages_manifest_sha256,
                "compiler_identity": environment.compiler_identity,
                "sdk_identity": environment.sdk_identity,
                "authority_epoch": environment.authority_epoch,
                "base_environment_generation": environment.base_environment_generation,
                "environment_content_sha256": environment.environment_content_sha256,
                "environment_authority_id": environment.environment_authority_id,
                "selected_interpreter_relative_path": (
                    environment.selected_interpreter_relative_path
                ),
                "selected_interpreter_digest": environment.selected_interpreter_digest,
                "external_escape_records": [
                    {
                        "path": str(record.path),
                        "sha256": record.sha256,
                        "kind": record.kind,
                    }
                    for record in environment.external_escape_records
                ],
            },
            "host": {
                "machine_id": platform.node() or "unknown-machine",
                "os": platform.system().lower(),
                "os_build": platform.version(),
                "architecture": platform.machine(),
                "cpu": platform.processor() or "unknown-cpu",
                "ram_bytes": _physical_memory_bytes(),
                "accelerator": None,
                "accelerator_runtime": None,
            },
            "invocation": {
                "argv": list(observation.argv),
                "cwd": observation.cwd,
                "safe_env": {"MENAGERIE_EXECUTION_OFFLINE": "1"},
                "seed": int(facts["input_contract"].get("seed", 0)),
                "device": facts["implementation"]["device_policy"],
                "mode": attempt_mode,
                "network_policy": "offline",
                "timeout_seconds": timeout_seconds,
                "rss_limit_bytes": rss_limit_bytes,
                "rss_limit_basis": rss_limit_basis,
                "scratch_limit_bytes": rss_limit_bytes,
            },
            "worker_receipt": worker_receipt,
            "supervisor_observation": {
                "exit_code": observation.exit_code,
                "signal": observation.signal_number,
                "wall_seconds": observation.wall_seconds,
                "cpu_seconds": observation.cpu_seconds,
                "peak_rss_bytes": observation.peak_rss_bytes,
                "stdout_sha256": observation.stdout_sha256,
                "stdout_bytes": observation.stdout_bytes,
                "stdout_tail": observation.stdout_tail,
                "stdout_completion_line": (
                    _attested_completion_line(observation.stdout_tail) if succeeded else None
                ),
                "stderr_sha256": observation.stderr_sha256,
                "stderr_bytes": observation.stderr_bytes,
                "stderr_tail": observation.stderr_tail,
                "full_log_local_path": observation.stderr_path,
                "full_log_retention": "campaign",
                "group_teardown_outcome": observation.group_teardown_outcome,
                "group_survivor_pids": list(observation.group_survivor_pids),
                "peak_rss_basis": observation.peak_rss_basis,
                "rss_kill_threshold_bytes": getattr(
                    observation, "rss_kill_threshold_bytes", 0
                ),
                "rss_overshoot_bytes": getattr(observation, "rss_overshoot_bytes", 0),
            },
            "policy_observation": {
                "network_attempted": bool(policy.get("network_attempted")),
                "socket_targets": list(policy.get("socket_targets", [])),
                "checkpoint_or_weight_read_attempted": bool(
                    policy.get("checkpoint_or_weight_read_attempted")
                ),
                "checkpoint_paths": list(policy.get("checkpoint_paths", [])),
                "write_outside_scratch_attempted": bool(
                    policy.get("write_outside_scratch_attempted")
                ),
                "write_paths": list(policy.get("write_paths", [])),
                "credentials_present": bool(policy.get("credentials_present")),
                "torchlens_import_attempted": bool(policy.get("torchlens_import_attempted")),
                "cache_read_attempted": bool(policy.get("cache_read_attempted")),
            },
            "error": error,
            "defer_evidence": None,
            "capability_observation": None,
            "execution_read_manifest_identity": (
                execution_read_manifest_identity
                or stable_hash("direct-supervised-execution-manifest")
            ),
            "raw_award_receipt": (
                deepcopy(verified.raw_award_receipt) if succeeded and verified is not None else None
            ),
            "raw_award_receipt_sha256": (
                verified.raw_award_receipt_sha256 if succeeded and verified is not None else None
            ),
            "parent_attestation": parent_attestation,
            "unattested_partial": (None if succeeded else deepcopy(result.unattested_partial)),
        }
        attempts.append(_redact_attempt_diagnostics(attempt, observation, diagnostics_root))
    return tuple(attempts)


def _attested_completion_line(stdout_tail: str) -> Optional[str]:
    """Return the final TorchLens-owned completion marker from a bounded stdout tail.

    Parameters
    ----------
    stdout_tail:
        Live parent-observed stdout tail.

    Returns
    -------
    str | None
        Final completion line, or ``None`` when no marker is present.
    """

    lines = stdout_tail.splitlines()
    if not lines or not lines[-1].startswith(_WORKER_COMPLETION_PREFIX):
        return None
    return lines[-1]


def _diagnostic_relative_path(diagnostics_root: Path, attempt_id: str) -> str:
    """Return a checkpoint-safe repository-relative diagnostic sidecar locator.

    Parameters
    ----------
    diagnostics_root:
        Local diagnostics root.
    attempt_id:
        Stable attempt identifier used as the sidecar filename.

    Returns
    -------
    str
        Relative locator rooted at ``.crawl-local``.

    Raises
    ------
    DriverIntegrationError
        If diagnostics are not rooted below the gitignored runtime directory.
    """

    resolved = diagnostics_root.resolve()
    if ".crawl-local" not in resolved.parts:
        raise DriverIntegrationError("diagnostic sidecars must live below .crawl-local")
    index = max(index for index, part in enumerate(resolved.parts) if part == ".crawl-local")
    relative_root = Path(*resolved.parts[index:])
    return (relative_root / f"{attempt_id}.json").as_posix()


def _diagnostics_root_for_work_root(work_root: Path) -> Path:
    """Return a campaign-local C-07 sidecar root below ``.crawl-local``.

    Parameters
    ----------
    work_root:
        Driver work-envelope root below its runtime directory.

    Returns
    -------
    pathlib.Path
        Production's sibling diagnostics directory, or an isolated nested
        ``.crawl-local`` directory for an explicitly relocated dry-run runtime.
    """

    runtime_root = work_root.parent
    if ".crawl-local" in runtime_root.resolve().parts:
        return runtime_root / "diagnostics"
    return runtime_root / ".crawl-local" / "diagnostics"


def _existing_sidecar_holds_attempt(sidecar_path: Path, attempt_id: str) -> bool:
    """Return whether a complete sidecar for this exact attempt already exists.

    Parameters
    ----------
    sidecar_path:
        Attempt-addressed local diagnostics path.
    attempt_id:
        Immutable attempt identity the sidecar must claim.

    Returns
    -------
    bool
        ``True`` only for a readable, parseable sidecar claiming ``attempt_id``.
    """

    try:
        existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(existing, Mapping) and existing.get("attempt_id") == attempt_id


def _redact_attempt_diagnostics(
    attempt: JsonObject,
    observation: Optional[SupervisorObservation],
    diagnostics_root: Optional[Path],
) -> JsonObject:
    """Persist exact local diagnostics and redact their canonical attempt projections.

    Parameters
    ----------
    attempt:
        Newly assembled attempt before canonical persistence.
    observation:
        Live :class:`SupervisorObservation` retaining exact bounded stream tails and paths,
        or ``None`` for a driver-originated failure with no child process.
    diagnostics_root:
        Gitignored local sidecar root. It may be omitted only when every controlled value
        is empty, as in receipt-contract unit tests.

    Returns
    -------
    dict[str, Any]
        Attempt whose externally controlled values are explicit redaction references.

    Raises
    ------
    DriverIntegrationError
        If nonempty diagnostics would otherwise be lost.
    """

    attempt_id = str(attempt["attempt_id"])
    controlled: dict[str, Any] = {}
    has_nonempty_controlled = False

    def collect(value: Any, location: str = "$") -> None:
        """Collect every external-text field before replacing it."""

        nonlocal has_nonempty_controlled
        if isinstance(value, Mapping):
            for key, nested in value.items():
                nested_location = f"{location}.{key}"
                if key in _EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS:
                    if _is_diagnostic_redaction_reference(nested):
                        continue
                    controlled[nested_location] = deepcopy(nested)
                    if nested is not None and nested != "":
                        has_nonempty_controlled = True
                collect(nested, nested_location)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                collect(nested, f"{location}[{index}]")

    collect(attempt)
    if not has_nonempty_controlled:
        return attempt
    if diagnostics_root is None:
        if has_nonempty_controlled:
            raise DriverIntegrationError(
                "externally controlled attempt text requires a local diagnostic sidecar"
            )
        return attempt

    local_path = _diagnostic_relative_path(diagnostics_root, attempt_id)
    sidecar_path = diagnostics_root / f"{attempt_id}.json"
    supervisor_value = attempt.get("supervisor_observation", {})
    supervisor = supervisor_value if isinstance(supervisor_value, Mapping) else {}
    sidecar: JsonObject = {
        "schema_version": "menagerie.crawler.local-diagnostics.v1",
        "attempt_id": attempt_id,
        "stdout": {
            "stream_sha256": (
                observation.stdout_sha256
                if observation is not None
                else supervisor.get("stdout_sha256")
            ),
            "stream_bytes": (
                observation.stdout_bytes
                if observation is not None
                else supervisor.get("stdout_bytes", 0)
            ),
            "tail": observation.stdout_tail
            if observation is not None
            else supervisor.get("stdout_tail", ""),
            "full_log_path": (
                observation.stdout_path
                if observation is not None
                else supervisor.get("full_log_local_path")
            ),
        },
        "stderr": {
            "stream_sha256": (
                observation.stderr_sha256
                if observation is not None
                else supervisor.get("stderr_sha256")
            ),
            "stream_bytes": (
                observation.stderr_bytes
                if observation is not None
                else supervisor.get("stderr_bytes", 0)
            ),
            "tail": observation.stderr_tail
            if observation is not None
            else supervisor.get("stderr_tail", ""),
            "full_log_path": (
                observation.stderr_path
                if observation is not None
                else supervisor.get("full_log_local_path")
            ),
        },
        "externally_controlled_fields": controlled,
    }
    # First write wins for one immutable attempt identity: the canonical ledger
    # row references THESE bytes by content hash, and a resumed leg re-deriving
    # the same failure (same identity, fresh clock, possibly different
    # traceback line numbers after an upgrade) reuses the RECORDED row -- so
    # overwriting the sidecar would silently orphan the recorded references
    # while the replay's row is never persisted. A missing or unreadable
    # sidecar is still (re)written: losing diagnostics is worse than a
    # refreshed transcript.
    if not _existing_sidecar_holds_attempt(sidecar_path, attempt_id):
        _write_json_atomic(sidecar_path, sidecar)
        sidecar_path.chmod(0o600)

    def redact(value: Any, location: str = "$") -> Any:
        """Replace controlled values with hash-bound local references."""

        if isinstance(value, Mapping):
            redacted: dict[str, Any] = {}
            for key, nested in value.items():
                nested_location = f"{location}.{key}"
                if (
                    key in _EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS
                    and nested is not None
                    and nested != ""
                    and not _is_diagnostic_redaction_reference(nested)
                ):
                    reference: dict[str, Any] = {
                        "redaction": _DIAGNOSTIC_REDACTION_MARKER,
                        "content_sha256": hash_bytes(canonical_json_bytes(nested)),
                        "local_path": local_path,
                        "diagnostic_key": nested_location,
                    }
                    if key == "stdout_tail":
                        reference["stream_sha256"] = (
                            observation.stdout_sha256
                            if observation is not None
                            else supervisor.get("stdout_sha256")
                        )
                    elif key == "stderr_tail":
                        reference["stream_sha256"] = (
                            observation.stderr_sha256
                            if observation is not None
                            else supervisor.get("stderr_sha256")
                        )
                    redacted[key] = reference
                else:
                    redacted[key] = redact(nested, nested_location)
            return redacted
        if isinstance(value, list):
            return [redact(nested, f"{location}[{index}]") for index, nested in enumerate(value)]
        return value

    redacted_attempt = redact(attempt)
    if not isinstance(redacted_attempt, dict):
        raise AssertionError("attempt redaction must preserve the top-level object")
    redacted_supervisor = redacted_attempt.get("supervisor_observation")
    if isinstance(redacted_supervisor, dict):
        redacted_supervisor["full_log_local_path"] = local_path
    return redacted_attempt


def _is_diagnostic_redaction_reference(value: Any) -> bool:
    """Return whether a value is an already-redacted C-07 sidecar reference.

    Parameters
    ----------
    value:
        Candidate diagnostic field value.

    Returns
    -------
    bool
        Whether the value carries the closed redaction marker and required locator fields.
    """

    return bool(
        isinstance(value, Mapping)
        and value.get("redaction") == _DIAGNOSTIC_REDACTION_MARKER
        and all(
            isinstance(value.get(field), str) and value.get(field)
            for field in ("content_sha256", "local_path", "diagnostic_key")
        )
    )


def _parent_cache_read_attempted(policy: Mapping[str, Any]) -> bool:
    """Detect forbidden cache roots in parent-owned successful-read telemetry.

    Parameters
    ----------
    policy:
        Receipt policy merged with parent-owned syscall path observations.

    Returns
    -------
    bool
        True when a recorded read path falls below a closed cache root.
    """

    paths = policy.get("checkpoint_paths", [])
    if not isinstance(paths, list):
        return False
    for value in paths:
        if not isinstance(value, str):
            continue
        parts = {part.lower() for part in Path(value).parts}
        if parts & _FORBIDDEN_CACHE_ROOT_NAMES:
            return True
        normalized = value.replace("\\", "/").lower()
        if "/.crawl-local/caches/" in normalized or "/caches/" in normalized:
            return True
    return False


def _receipt_envelope_error(
    result: SupervisedResult,
    proposal: Mapping[str, Any],
    execution_identity: str,
    *,
    requested_mode: Optional[str] = None,
) -> Optional[str]:
    """Return a protocol error unless the requested child envelope is current.

    Parameters
    ----------
    result:
        Parent-owned supervisor observation and child receipt.
    proposal:
        Current accepted author proposal.
    execution_identity:
        Parent-computed execution identity.
    requested_mode:
        Explicit single mode assigned to this subprocess, or ``None`` for a
        legacy all-modes request.
    """

    verified, verification_error = _verified_worker_result(
        result,
        proposal,
        execution_identity,
        requested_mode=requested_mode,
    )
    if verified is None:
        return verification_error or "invalid-receipt:worker-result-v3"
    receipt = verified.diagnostic
    successful_exit = result.observation.exit_code == 0 and result.observation.signal_number is None
    expected_adapter = _expected_adapter_sha256(proposal)
    expected_manifest = _expected_code_manifest_sha256(proposal)
    observed_asset_pair = (
        receipt.get("observed_input_asset_sha256"),
        next(
            (
                value.get("input_asset")
                for value in receipt.get("per_mode", {}).values()
                if isinstance(value, Mapping)
            ),
            None,
        ),
    )
    expected_asset_pair = (
        _expected_input_asset_sha256(proposal),
        _expected_input_asset_id(proposal),
    )
    if (
        receipt.get("stable_id") != proposal.get("stable_id")
        or receipt.get("source_identity") != proposal.get("source_identity")
        or receipt.get("recipe_revision") != proposal.get("recipe_revision")
        or receipt.get("execution_identity") != execution_identity
        or (
            receipt.get("observed_recipe_revision") is not None
            and receipt.get("observed_recipe_revision") != proposal.get("recipe_revision")
        )
        or (
            receipt.get("observed_adapter_sha256") is not None
            and receipt.get("observed_adapter_sha256") != expected_adapter
        )
        or (
            receipt.get("observed_code_manifest_sha256") is not None
            and receipt.get("observed_code_manifest_sha256") != expected_manifest
        )
        or observed_asset_pair
        not in {
            (None, None),
            expected_asset_pair,
            (expected_asset_pair[0], None) if not successful_exit else expected_asset_pair,
        }
    ):
        return "invalid-receipt:identity"
    modes = receipt.get("meaningful_modes")
    detected = receipt.get("detected_meaningful_modes")
    declared = receipt.get("declared_meaningful_modes")
    per_mode = receipt.get("per_mode")
    if (
        not isinstance(modes, list)
        or not isinstance(detected, list)
        or not isinstance(declared, list)
        or any(not isinstance(value, str) for value in (*modes, *detected, *declared))
        or len(modes) != len(set(modes))
        or len(detected) != len(set(detected))
        or len(declared) != len(set(declared))
    ):
        return "invalid-receipt:mode-envelope"
    proposal_modes = set(
        str(value)
        for value in proposal.get("proposed_facts", {}).get("modes", {}).get("meaningful_modes", [])
    )
    mode_set = set(modes)
    detected_set = set(detected)
    declared_set = set(declared)
    valid_modes = {"train", "eval"}
    if (
        not mode_set <= valid_modes
        or not detected_set <= valid_modes
        or declared_set != proposal_modes
        or mode_set != proposal_modes | detected_set
    ):
        return "invalid-receipt:mode-envelope"
    if detected_set - proposal_modes:
        return "invalid-receipt:meaningful-mode-contract"
    if not isinstance(per_mode, Mapping):
        return "invalid-receipt:mode-envelope"
    receipt_mode = receipt.get("mode")
    validated_modes: tuple[str, ...]
    if not successful_exit:
        if requested_mode is not None and (
            requested_mode not in proposal_modes
            or receipt_mode != requested_mode
            or not set(per_mode) <= {requested_mode}
        ):
            return "invalid-receipt:mode-envelope"
        if requested_mode is None and receipt_mode is not None:
            return "invalid-receipt:mode-envelope"
        validated_modes = tuple(str(mode) for mode in per_mode)
    elif requested_mode is not None:
        if verified.parent_attestation is None:
            return "missing-parent-success-attestation"
        if (
            requested_mode not in proposal_modes
            or requested_mode not in mode_set
            or receipt_mode != requested_mode
            or set(per_mode) != {requested_mode}
        ):
            return "invalid-receipt:mode-envelope"
        validated_modes = (requested_mode,)
    else:
        if verified.parent_attestation is None:
            return "missing-parent-success-attestation"
        if receipt_mode is not None or set(per_mode) != mode_set:
            return "invalid-receipt:mode-envelope"
        validated_modes = tuple(str(mode) for mode in modes)
    required_mode = {
        "mode",
        "constructor_started",
        "constructor_completed",
        "input_completed",
        "input_signature",
        "forward_started",
        "forward_completed",
        "output_signature",
        "error",
    }
    for mode in validated_modes:
        value = per_mode.get(mode)
        if not isinstance(value, Mapping) or not required_mode <= set(value):
            return "invalid-receipt:incomplete-mode"
        if not successful_exit:
            if value.get("mode") != mode:
                return "invalid-receipt:mode-envelope"
            continue
        if (
            value.get("mode") != mode
            or value.get("error") is not None
            or not value.get("constructor_started")
            or not value.get("constructor_completed")
            or not value.get("input_completed")
            or not value.get("forward_started")
            or not value.get("forward_completed")
            or not input_signature_matches_contract(
                value.get("input_signature"),
                proposal.get("proposed_facts", {}).get("input_contract", {}),
            )
            or (
                receipt.get("observed_input_asset_sha256"),
                value.get("input_asset"),
            )
            not in {(None, None), expected_asset_pair}
        ):
            return "invalid-receipt:incomplete-mode"
        if output_signature_error(value.get("output_signature")) is not None:
            return "invalid-receipt:output-signature"
    policy = receipt.get("policy_observation")
    required_policy = {
        "network_attempted",
        "socket_targets",
        "checkpoint_or_weight_read_attempted",
        "checkpoint_paths",
        "write_outside_scratch_attempted",
        "write_paths",
        "credentials_present",
        "torchlens_import_attempted",
        "cache_read_attempted",
    }
    if not isinstance(policy, Mapping) or not required_policy <= set(policy):
        return "invalid-receipt:policy-envelope"
    if successful_exit and verified.raw_award_receipt is None:
        return "missing-authenticated-raw-award-receipt"
    if successful_exit and receipt.get("error") is not None:
        return "invalid-receipt:success-with-error"
    if not successful_exit and not (
        isinstance(receipt.get("error"), Mapping)
        or any(
            isinstance(value, Mapping) and isinstance(value.get("error"), Mapping)
            for value in per_mode.values()
        )
        or any(policy.get(field) for field in required_policy if field.endswith("attempted"))
    ):
        return "invalid-receipt:failure-without-error"
    return None


#: Closed OS-resource errno set whose pre-receipt worker aborts are HOST facts
#: (DX-13/RE-14): out-of-space, read-only filesystem, device gone, I/O fault.
#: Deliberately narrow -- every other errno keeps model ownership, because a
#: model's own code raising OSError (bad path, EPERM probes) is model behavior.
_HOST_RESOURCE_ABORT_ERRNOS = frozenset({errno.ENOSPC, errno.EROFS, errno.ENODEV, errno.EIO})


#: The worker's typed backend-parity refusal, matched on its recorded
#: fully-qualified type (DX-14/RE-15).
_BACKEND_PROTOCOL_GAP_TYPE = "menagerie.crawler.backend.BackendProtocolUnimplementedError"


def _backend_protocol_gap_error(
    receipt: Mapping[str, Any], mode_receipt: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    """Return the worker's backend-protocol-gap error when one was recorded.

    Parameters
    ----------
    receipt, mode_receipt:
        Verified non-awarding diagnostic facts.

    Returns
    -------
    Mapping[str, Any] | None
        The typed gap error (mode error preferred, then the receipt's global
        error), or ``None`` when neither names the gap type.
    """

    for candidate in (mode_receipt.get("error"), receipt.get("error")):
        if (
            isinstance(candidate, Mapping)
            and candidate.get("exception_type") == _BACKEND_PROTOCOL_GAP_TYPE
        ):
            return candidate
    return None


#: The worker's typed determinism-context refusal (DX-70/UU-24).
_EXECUTION_CONTEXT_VIOLATION_TYPE = "menagerie.crawler.worker.ExecutionContextViolationError"


def _execution_context_violation_error(
    receipt: Mapping[str, Any], mode_receipt: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    """Return the worker's execution-context violation error when recorded.

    Parameters
    ----------
    receipt, mode_receipt:
        Verified non-awarding diagnostic facts.

    Returns
    -------
    Mapping[str, Any] | None
        The typed violation error, or ``None``.
    """

    for candidate in (mode_receipt.get("error"), receipt.get("error")):
        if (
            isinstance(candidate, Mapping)
            and candidate.get("exception_type") == _EXECUTION_CONTEXT_VIOLATION_TYPE
        ):
            return candidate
    return None


def _host_resource_abort_errno(error: object) -> Optional[int]:
    """Return the typed errno when a pre-receipt abort names host resource weather.

    Parameters
    ----------
    error:
        The receipt's global error mapping, if any.

    Returns
    -------
    int | None
        The abort's errno when the error is the worker's
        ``worker-aborted-before-receipt`` diagnostic AND its recorded errno is
        in the closed OS-resource set; ``None`` otherwise. Keys on the typed
        ``errno`` integer the worker records -- never on message text.
    """

    if not isinstance(error, Mapping):
        return None
    if error.get("reason_code") != "worker-aborted-before-receipt":
        return None
    recorded = error.get("errno")
    if isinstance(recorded, int) and recorded in _HOST_RESOURCE_ABORT_ERRNOS:
        return recorded
    return None


def _supervised_failure(
    result: SupervisedResult,
    receipt: Mapping[str, Any],
    mode_receipt: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    worker_result_present: bool,
) -> JsonObject:
    """Classify a failed worker observation into its actual closed stage and reason.

    Parameters
    ----------
    result:
        Parent-observed result, possibly replaced with a projection error.
    receipt, mode_receipt, policy:
        Verified non-awarding diagnostic facts.
    worker_result_present:
        Whether the central v3 projection accepted an outer worker result.

    Returns
    -------
    dict[str, Any]
        Closed attempt failure fields.
    """

    policy_reasons = (
        ("network_attempted", "network-attempt"),
        ("checkpoint_or_weight_read_attempted", "checkpoint-read"),
        ("cache_read_attempted", "checkpoint-read"),
        ("write_outside_scratch_attempted", "write-outside-scratch"),
        ("credentials_present", "credentials-exposed"),
        ("torchlens_import_attempted", "torchlens-import"),
        # Host telemetry loss poisons the award (no award on unverifiable
        # telemetry) but is a HOST fact with its own reason, never the model
        # policy violation it used to masquerade as (BL-12(e), W-F3 lane b).
        ("telemetry_failure", "telemetry-integrity"),
    )
    for field, reason in policy_reasons:
        if policy.get(field):
            details: dict[str, Any] = {"policy_field": field}
            message = f"worker policy violation: {reason}"
            if reason == "telemetry-integrity":
                details["owner"] = "host"
                details["telemetry_failure"] = policy.get("telemetry_failure")
                message = "host sandbox telemetry failed integrity; award unverifiable"
            return _attempt_error_fields(
                "policy",
                reason,
                receipt.get("error"),
                message,
                native_crash=False,
                details=details,
            )
    observation = result.observation
    if observation.timed_out:
        return _attempt_error_fields(
            "resource",
            "timeout",
            None,
            "worker exceeded the parent wall timeout",
            native_crash=False,
            details={"wall_seconds": observation.wall_seconds},
        )
    if observation.rss_exceeded:
        return _attempt_error_fields(
            "resource",
            "rss-cap",
            None,
            "worker exceeded the parent RSS kill threshold",
            native_crash=False,
            details={
                "peak_rss_bytes": observation.peak_rss_bytes,
                "rss_kill_threshold_bytes": getattr(
                    observation, "rss_kill_threshold_bytes", 0
                ),
                "rss_overshoot_bytes": getattr(observation, "rss_overshoot_bytes", 0),
            },
        )
    if observation.signal_number is not None:
        return _attempt_error_fields(
            "runner",
            "signal",
            None,
            f"worker terminated by signal {observation.signal_number}",
            native_crash=True,
            details={"signal": observation.signal_number},
        )
    if observation.exit_code == 126 and not receipt.get("constructor_started"):
        # DX-2/RE-2: exit 126 is the exec layer refusing to run the worker at
        # all (unrunnable interpreter, permission fault) with no constructor
        # evidence -- a HOST bootstrap fact about the routed environment. A
        # model exiting 126 AFTER its constructor ran keeps the ordinary
        # receipt-evidence classification below (anti-forgery: faking a
        # bootstrap death buys a park, never an award).
        return _attempt_error_fields(
            "runner",
            "worker-bootstrap-failed",
            receipt.get("error"),
            "worker process could not be bootstrapped (exit 126 before any model code)",
            native_crash=False,
            details={
                "exit_code": observation.exit_code,
                "owner": "host",
                "constructor_started": False,
            },
        )
    if not worker_result_present:
        if result.receipt_error in {"failed:policy", "failed:sandbox-unavailable"}:
            return _attempt_error_fields(
                "policy",
                "sandbox-unavailable-v1",
                None,
                "required operating-system sandbox is unavailable",
                native_crash=False,
                details={"receipt_error": result.receipt_error},
            )
        reason = (
            "missing-receipt" if result.receipt_error == "missing-receipt" else "protocol-violation"
        )
        constructor_started = bool(receipt.get("constructor_started"))
        details: dict[str, Any] = {
            "receipt_error": result.receipt_error,
            "constructor_started": constructor_started,
        }
        if not constructor_started:
            # DX-2/RE-2: nothing proves model code ever ran, so a missing or
            # protocol-violating receipt is the harness's own delivery fault
            # until the receipt says otherwise. Anti-forgery holds: a
            # receipt-suppressing model still cannot mint success, and the
            # "receipt suppressed AFTER the constructor started" residual is
            # owned by B3-XQ-1 (constructor_started=True keeps model ownership).
            details["owner"] = "machinery"
        return _attempt_error_fields(
            "runner",
            reason,
            None,
            str(result.receipt_error or "worker receipt unavailable"),
            native_crash=False,
            details=details,
        )
    if result.receipt_error == "missing-or-mismatched-v3-attestation":
        # Worker result present but the raw-award/attestation association is
        # unclean. The pre-fix fall-through published this as an evidence-free
        # ``forward/mode-run`` -- a model verdict contradicting the parent's own
        # ``unattested_partial`` (BL-12(f), opus W-7: 17 of 45 pilot forward
        # failures). Machinery owns this failure and its record says so.
        partial = (
            result.unattested_partial if isinstance(result.unattested_partial, Mapping) else {}
        )
        return _attempt_error_fields(
            "runner",
            "attestation-mismatch",
            receipt.get("error"),
            "worker result present but its v3 attestation association is unclean",
            native_crash=False,
            details={
                "receipt_error": result.receipt_error,
                "unattested_partial_reason": partial.get("reason_code"),
                "owner": "machinery",
            },
        )
    parity_gap = _backend_protocol_gap_error(receipt, mode_receipt)
    if parity_gap is not None:
        # The worker's own typed refusal: a backend protocol (seeding,
        # inference context, input conversion, dtype normalization) is not
        # implemented for this framework. A MACHINERY capability gap, never a
        # model verdict (DX-14/RE-15) -- the owner stamp routes it to the park
        # lane. INTERIM classification: the honest terminal is
        # deferred:needs-backend-parity, whose machine-derived proof arm does
        # not exist yet (reported to the plan; the head stays pending, bounded
        # by the recidivism cap). A model fabricating this exception type from
        # inside forward() gains only a bounded quarantine hold -- it can never
        # mint success -- and the eventual deferral arm re-derives its proof
        # from the capability registry, never from this producer-written type.
        return _attempt_error_fields(
            "runner",
            "internal-error",
            parity_gap,
            "worker backend protocol is unimplemented for this framework "
            "(machinery capability gap)",
            native_crash=False,
            details={
                "receipt_error": parity_gap,
                "owner": "machinery",
                "backend_protocol_unimplemented": True,
                "intended_disposition": BACKEND_PARITY_DEFERRAL_STATUS_CODE,
            },
        )
    context_violation = _execution_context_violation_error(receipt, mode_receipt)
    if context_violation is not None:
        # The worker refused because its live execution context (thread pin)
        # contradicts the request: a HOST/machinery fact about the spawn
        # environment -- digests must never mint under a drifted context, and
        # the model is never charged for it (DX-70/UU-24).
        return _attempt_error_fields(
            "runner",
            "internal-error",
            context_violation,
            "worker execution context contradicts its pinned determinism locks",
            native_crash=False,
            details={
                "receipt_error": context_violation,
                "owner": "host",
                "execution_context_violation": True,
            },
        )
    global_error = receipt.get("error")
    host_errno = _host_resource_abort_errno(global_error)
    if host_errno is not None:
        # The worker aborted before observing anything, and its typed errno
        # names OS resource weather (ENOSPC/EROFS/ENODEV/EIO): a HOST fact,
        # never a model verdict (DX-13/RE-14). The owner stamp routes it to
        # the park lane through NEVER_MODEL_TERMINAL_OWNERS; model exceptions
        # (no errno) keep their historical ownership below.
        return _attempt_error_fields(
            "runner",
            "internal-error",
            global_error,
            f"worker aborted before its receipt on a host resource fault (errno {host_errno})",
            native_crash=False,
            details={
                "receipt_error": global_error,
                "owner": "host",
                "errno": host_errno,
            },
        )
    if not receipt.get("constructor_started"):
        runtime_frame = _crawler_runtime_import_failure(global_error)
        if runtime_frame is not None:
            # DX-2/RE-2: the import died inside the crawler's OWN worker
            # runtime frame before any model code was reachable (the
            # StrEnum/py3.10 bootstrap class) -- a HOST fact about the routed
            # environment, never a verdict about the recipe it was about to
            # load.
            return _attempt_error_fields(
                "runner",
                "worker-runtime-unavailable",
                global_error,
                "worker crawler runtime failed before any model code was reachable",
                native_crash=False,
                details={
                    "receipt_error": global_error,
                    "owner": "host",
                    "crawler_frame": runtime_frame,
                    "constructor_started": False,
                },
            )
        return _attempt_error_fields(
            "import",
            "import-exception",
            global_error,
            "worker failed while loading the recipe",
            native_crash=False,
            details={"receipt_error": global_error},
        )
    if not receipt.get("constructor_completed"):
        return _attempt_error_fields(
            "constructor",
            "exception",
            global_error,
            "model constructor failed",
            native_crash=False,
            details={"receipt_error": global_error},
        )
    if not receipt.get("input_completed"):
        return _attempt_error_fields(
            "input",
            "generation-exception",
            global_error,
            "dummy input generation failed",
            native_crash=False,
            details={"receipt_error": global_error},
        )
    # The per-mode error is the primary evidence, but a failure raised while
    # BUILDING the mode receipt (m538: the input-signature hasher TypeErrored
    # on a DGLGraph leaf) escapes _mode_receipt entirely and lands in the
    # receipt's GLOBAL error with per_mode empty. Publishing mode_error=None
    # while the real traceback sat in the work-dir receipt dropped the only
    # diagnostic (step-100 B2); the global error is propagated verbatim as the
    # fallback so the published diagnostic always carries the worker's own
    # exception evidence when any was recorded.
    mode_error = mode_receipt.get("error") or receipt.get("error")
    if isinstance(mode_error, Mapping):
        return _attempt_error_fields(
            "forward",
            "mode-run",
            mode_error,
            "meaningful mode forward failed",
            native_crash=False,
            details={"mode_error": mode_error},
        )
    # The last branch of a fall-through chain must never be an error bucket
    # that LIES: a residue with no worker error is not a proven forward
    # failure. It publishes as an explicit machinery unknown naming exactly
    # the observation fields this classifier inspected (BL-12(f), C-17).
    return _attempt_error_fields(
        "runner",
        "unclassified-supervised-failure",
        None,
        "supervised failure matched no classification arm and carried no worker error",
        native_crash=False,
        details={
            "inspected": [
                "policy_observation",
                "timed_out",
                "rss_exceeded",
                "signal",
                "worker_result_present",
                "receipt_error",
                "constructor_started",
                "constructor_completed",
                "input_completed",
                "per_mode.error",
                "error",
            ],
            "receipt_error": result.receipt_error,
            "owner": "machinery",
        },
    )


#: BL-12(e) terminal-owner vocabulary: every failure reason the WORKER AREA can
#: mint, bound to the actor that owns the fault. This is the census authority
#: for ``test_terminal_owner_census``: a reason absent here is a recurrence of
#: C-16 (owner-less terminal), and a ``host``/``machinery``-owned reason must
#: never settle a model's head -- ``_forward_and_reduce`` refuses to
#: terminalize it and parks the work behind a campaign-health event instead.
WORKER_FAILURE_REASON_OWNERS: Mapping[str, str] = {
    # In-child policy tripwires: the model's own code performed the act.
    "network-attempt": "model",
    "checkpoint-read": "model",
    "write-outside-scratch": "model",
    "credentials-exposed": "model",
    "torchlens-import": "model",
    # Host log-delivery integrity: the host, never the model.
    "telemetry-integrity": "host",
    # Parent-enforced resource facts about the model's own execution.
    "timeout": "model",
    "rss-cap": "model",
    # DX-7/RE-7: derived from the member's own sibling-slot resource kill --
    # the member's execution behavior, never a decisive verdict on its own.
    "blocked-by-sibling": "model",
    "oom": "model",
    "signal": "model",
    "native-crash": "model",
    # Worker-stage facts derived from the receipt's stage flags.
    "import-exception": "model",
    "exception": "model",
    "generation-exception": "author",
    "contract-invalid": "author",
    "mode-run": "model",
    "confirmation-mismatch": "model",
    "incomplete-receipt": "model",
    "invalid-output-signature": "model",
    # Machinery/protocol facts: the harness, never the model.
    "attestation-mismatch": "machinery",
    "unclassified-supervised-failure": "machinery",
    "missing-receipt": "model",
    "protocol-violation": "model",
    "ledger-corruption": "machinery",
    "internal-error": "machinery",
    # Host machinery facts: sandbox/telemetry/reap infrastructure.
    "sandbox-unavailable-v1": "host",
    "worker-unreapable": "host",
    # DX-2/RE-2: pre-model-code worker deaths. The exec layer refused to start
    # the worker (exit 126), or the crawler-frame runtime failed to import
    # before any model code was reachable -- HOST facts about the routed
    # environment, never model verdicts.
    "worker-bootstrap-failed": "host",
    "worker-runtime-unavailable": "host",
}

#: Owners whose failures may NEVER become a permanent model terminal (C-16):
#: infrastructure weather and harness defects are parks, not verdicts about
#: the model. ``model`` and ``author`` failures keep their terminal routes.
NEVER_MODEL_TERMINAL_OWNERS = frozenset({"host", "machinery"})


def _failure_owner(error: Mapping[str, Any]) -> Optional[str]:
    """Return the owning actor for one classified attempt failure.

    Parameters
    ----------
    error:
        Structured attempt error carrying ``reason_code`` and ``details``.

    Returns
    -------
    str | None
        Explicit per-attempt owner when the classifier stamped one, else the
        closed per-reason owner, else ``None`` (an owner-less reason -- the
        census test refuses those at review time).
    """

    details = error.get("details")
    if isinstance(details, Mapping):
        stamped = details.get("owner")
        if isinstance(stamped, str):
            return stamped
    reason = error.get("reason_code")
    return WORKER_FAILURE_REASON_OWNERS.get(str(reason)) if reason is not None else None


#: Exception types whose crawler-frame origin marks the worker RUNTIME as the
#: failing party (DX-2/RE-2). Closed on purpose: only import machinery can die
#: before any model code is reachable while still writing a receipt.
_RUNTIME_IMPORT_ERROR_TYPES = frozenset({"ImportError", "ModuleNotFoundError"})

#: Path fragment identifying the crawler's own worker runtime in a traceback.
_CRAWLER_FRAME_MARKER = "menagerie/crawler/"


def _crawler_runtime_import_failure(error: Any) -> Optional[str]:
    """Return the crawler-frame path when a pre-constructor import died in OUR code.

    DX-2/RE-2 discriminator: an ``ImportError`` whose DEEPEST traceback frame
    sits inside ``menagerie/crawler`` died in the worker's own runtime (the
    StrEnum/py3.10 bootstrap class), before any model code was reachable. An
    import failure whose deepest frame is the recipe or its libraries stays the
    model's own ``import-exception``.

    Parameters
    ----------
    error:
        The worker receipt's structured global error, if any.

    Returns
    -------
    str | None
        The offending crawler-frame path, or ``None`` when this is not a
        crawler-runtime import failure.
    """

    if not isinstance(error, Mapping):
        return None
    if str(error.get("type")) not in _RUNTIME_IMPORT_ERROR_TYPES:
        return None
    traceback_text = error.get("traceback")
    if not isinstance(traceback_text, str):
        return None
    deepest: Optional[str] = None
    for line in traceback_text.splitlines():
        stripped = line.strip()
        if stripped.startswith('File "'):
            deepest = stripped.split('"')[1] if stripped.count('"') >= 2 else None
    if deepest is not None and _CRAWLER_FRAME_MARKER in deepest.replace("\\", "/"):
        return deepest
    return None


def _stamp_failure_owner(attempt: Mapping[str, Any], owner: str) -> None:
    """Stamp the explicit per-attempt failure owner onto a classified attempt.

    Parameters
    ----------
    attempt:
        Freshly built failure attempt (mutated in place before persistence).
    owner:
        Closed owner actor (``host``/``machinery``/``model``/``author``).
    """

    error_value = attempt.get("error")
    if isinstance(error_value, dict):
        details_value = error_value.setdefault("details", {})
        if isinstance(details_value, dict):
            details_value["owner"] = owner


def _non_model_terminal_error(
    status_code: str,
    reason_code: Optional[str],
    attempts: Sequence[Mapping[str, Any]],
    detail: Optional[str],
) -> Optional[JsonObject]:
    """Return the classified failure when a failed terminal is host/machinery-owned.

    DX-1/RE-1: the ownership rule is enforced at the MINT SINK, not only at the
    classification arms -- ``_terminalize`` consults this before appending any
    ``failed:*`` revision, so a call site that never consulted
    :func:`_failure_owner` (the m12446 shape: a machinery internal-error escaping
    a repair lane) cannot settle a model's head. The decisive evidence is the
    most recent failed attempt carrying the terminal's own reason code; when the
    caller persisted no such attempt the terminal's own classification is judged
    by the closed per-reason owner map alone.

    Parameters
    ----------
    status_code, reason_code:
        Candidate terminal disposition.
    attempts:
        Decisive attempts the caller intends to bind to the terminal.
    detail:
        Caller-supplied diagnostic detail for the synthesized fallback error.

    Returns
    -------
    dict[str, Any] | None
        The classified error whose owner forbids a model terminal, or ``None``
        when the disposition is legitimately mintable.
    """

    if not status_code.startswith("failed:") or reason_code is None:
        return None
    candidate: Optional[Mapping[str, Any]] = None
    for attempt in reversed(tuple(attempts)):
        if not isinstance(attempt, Mapping) or attempt.get("result") != "failed":
            continue
        error = attempt.get("error")
        if isinstance(error, Mapping) and error.get("reason_code") == reason_code:
            candidate = error
            break
    if candidate is None:
        candidate = {
            "stage": status_code.partition(":")[2],
            "reason_code": reason_code,
            "message": detail or "",
            "details": {},
        }
    if _failure_owner(candidate) not in NEVER_MODEL_TERMINAL_OWNERS:
        return None
    return dict(candidate)


#: Formatted-traceback sentences separating the blocks of a chained exception.
_TRACEBACK_CHAIN_MARKERS = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)

#: Bound on the chain suffix appended to a diagnostic message.
_EXCEPTION_CHAIN_MESSAGE_LIMIT = 2000


def _exception_chain_message(error: Mapping[str, Any]) -> Optional[str]:
    """Compose the FULL exception chain into one diagnostic message.

    A worker receipt records ``message = str(exc)`` -- the FINAL exception
    only. When that exception was raised while HANDLING another one, the
    primary refusal exists solely inside the formatted traceback, and the
    diagnostic ``$.error.message`` lies by omission: rung-11's R2 terminals all
    read ``No module named 'pit'`` while the primary
    ``RecipeError: typed adapter recipe revision mismatch`` sat unread in the
    traceback text. This recovers every chained exception headline from the
    formatted traceback, oldest first, and appends the chain to the message.
    It is fail-open: any anomaly returns ``None`` and the plain message stands.

    Parameters
    ----------
    error:
        Worker or driver error mapping carrying ``message`` and ``traceback``.

    Returns
    -------
    str | None
        The message with its full exception chain, or ``None`` when the
        traceback shows no chain or cannot be parsed.
    """

    traceback_text = error.get("traceback")
    message = error.get("message")
    if not isinstance(traceback_text, str) or not isinstance(message, str):
        return None
    if not any(marker in traceback_text for marker in _TRACEBACK_CHAIN_MARKERS):
        return None
    skipped = set(_TRACEBACK_CHAIN_MARKERS) | {"Traceback (most recent call last):"}
    headlines = [
        line
        for line in traceback_text.splitlines()
        if line and not line[0].isspace() and line not in skipped
    ]
    if len(headlines) < 2:
        return None
    chain = " -> ".join(headlines)[:_EXCEPTION_CHAIN_MESSAGE_LIMIT]
    return f"{message} [full exception chain, oldest first: {chain}]"


def _attempt_error_fields(
    stage: str,
    reason_code: str,
    worker_error: Any,
    fallback_message: str,
    *,
    native_crash: bool,
    details: Mapping[str, Any],
) -> JsonObject:
    """Build complete attempt error fields from optional worker Python evidence."""

    error = worker_error if isinstance(worker_error, Mapping) else {}
    traceback_text = error.get("traceback")
    message = _exception_chain_message(error) or str(error.get("message") or fallback_message)
    return {
        "stage": stage,
        "reason_code": reason_code,
        "exception_type": error.get("exception_type"),
        "message": message,
        "traceback": traceback_text,
        "no_traceback_reason": None if traceback_text else "no Python traceback was available",
        "native_crash": native_crash,
        "details": dict(details),
    }


#: Exact worker/driver spelling of a typed recipe-contract refusal.
_RECIPE_ERROR_EXCEPTION_TYPE = "menagerie.crawler.recipe.RecipeError"


def _resolve_redacted_diagnostic(
    attempt: Mapping[str, Any],
    reference: Mapping[str, Any],
    diagnostics_root: Optional[Path],
) -> Optional[str]:
    """Resolve one hash-bound diagnostic redaction back to its exact local text.

    Fail-closed on every integrity question: a missing sidecar, a foreign
    key, or bytes that no longer reproduce the recorded content hash all
    return ``None`` rather than feeding unverified externally controlled text
    to a repair brief.

    Parameters
    ----------
    attempt:
        Persisted attempt carrying the redacted field.
    reference:
        Closed redaction reference (``_is_diagnostic_redaction_reference``).
    diagnostics_root:
        Gitignored local sidecar root the attempt was persisted against.

    Returns
    -------
    str | None
        The exact recorded text, or ``None`` when it cannot be verified.
    """

    if diagnostics_root is None:
        return None
    sidecar_path = diagnostics_root / f"{attempt.get('attempt_id')}.json"
    try:
        sidecar = _read_json(sidecar_path)
    except Exception:  # noqa: BLE001 -- unreadable local sidecar fails closed
        return None
    fields = sidecar.get("externally_controlled_fields")
    if not isinstance(fields, Mapping):
        return None
    value = fields.get(str(reference.get("diagnostic_key")))
    if not isinstance(value, str) or not value:
        return None
    if hash_bytes(canonical_json_bytes(value)) != reference.get("content_sha256"):
        return None
    return value


def _deterministic_recipe_refusal(
    attempt: Mapping[str, Any], diagnostics_root: Optional[Path]
) -> Optional[str]:
    """Return the refusal text when a failed attempt is a typed recipe refusal.

    A :class:`~menagerie.crawler.recipe.RecipeError` raised at the import stage
    is a deterministic recipe-CONTRACT refusal -- the loader rejected the
    declared recipe before any library import ran, so retrying the same bytes
    can never pass and terminalizing ``failed:import`` burns a repairable
    proposal (rung-11 census, defect class 3: 11 timm models). The honest
    route is one bounded author-repair generation carrying the exact refusal.

    Parameters
    ----------
    attempt:
        Persisted failed attempt, whose ``error.message`` may be a hash-bound
        diagnostic redaction reference.
    diagnostics_root:
        Local sidecar root used to resolve a redacted message.

    Returns
    -------
    str | None
        The exact refusal message, or ``None`` for every non-refusal failure.
    """

    error = attempt.get("error")
    if not isinstance(error, Mapping):
        return None
    if error.get("stage") != "import":
        return None
    if error.get("exception_type") != _RECIPE_ERROR_EXCEPTION_TYPE:
        return None
    message: Any = error.get("message")
    if _is_diagnostic_redaction_reference(message):
        message = _resolve_redacted_diagnostic(attempt, message, diagnostics_root)
    if not isinstance(message, str) or not message:
        return None
    return message


#: Closed vocabulary of recipe-refusal charge dispositions.
_RECIPE_REPAIR_CHARGE_DISPOSITIONS = frozenset(
    {"revision-rejected", "revision-accepted-for-execution"}
)


def _recipe_repair_scope_tag(
    *, work_id: str, campaign_root_work_id: str, root_cause_fingerprint: str
) -> str:
    """Return the filename discriminator for one work/refusal repair scope.

    Parameters
    ----------
    work_id, campaign_root_work_id, root_cause_fingerprint:
        Exact identity triple a repair budget is granted against.

    Returns
    -------
    str
        The FULL scope digest hex (Sol wave-8 R2 finding 5: the earlier
        48-bit tag could collide, letting a colliding scope's same-generation
        marker path REPLACE an earlier row through the atomic writer and
        suppress fresh feedback through the request-file idempotency check).
        Distinct scopes now get distinct marker paths outright.
    """

    return stable_hash(
        {
            "work_id": work_id,
            "campaign_root_work_id": campaign_root_work_id,
            "root_cause_fingerprint": root_cause_fingerprint,
        }
    ).removeprefix("sha256:")


def _valid_recipe_repair_charge(
    row: object,
    *,
    filename_generation: int,
    stable_id: str,
    work_id: str,
    campaign_root_work_id: str,
    root_cause_fingerprint: str,
) -> Optional[int]:
    """Validate one charge row against the closed schema and its filename.

    Sol wave-8 R2 finding 5's probe row -- ``generation: true`` (a boolean),
    foreign stable ID/gate/disposition, filename suffix ``-999`` -- passed
    the triple-only check and consumed budget. Every field of the closed
    schema is validated here, the boolean/int conflation is refused
    explicitly, and the filename's generation suffix must reproduce the
    content's generation.

    Parameters
    ----------
    row:
        Parsed candidate charge row.
    filename_generation:
        Generation parsed from the marker's filename suffix.
    stable_id, work_id, campaign_root_work_id, root_cause_fingerprint:
        Exact identities the row must reproduce.

    Returns
    -------
    int | None
        The validated generation, or ``None`` for residue that must not count.
    """

    if not isinstance(row, Mapping):
        return None
    generation = row.get("generation")
    if type(generation) is not int or generation < 1:
        # ``isinstance(True, int)`` holds in Python; an exact type check is
        # the only spelling that refuses the boolean forgery.
        return None
    if generation != filename_generation:
        return None
    if row.get("stable_id") != stable_id:
        return None
    if row.get("gate_kind") != "recipe_refusal":
        return None
    if (
        row.get("work_id") != work_id
        or row.get("campaign_root_work_id") != campaign_root_work_id
        or row.get("root_cause_fingerprint") != root_cause_fingerprint
    ):
        return None
    disposition = row.get("disposition")
    if disposition not in _RECIPE_REPAIR_CHARGE_DISPOSITIONS:
        return None
    error_type = row.get("error_type")
    if disposition == "revision-rejected":
        if not isinstance(error_type, str) or not error_type:
            return None
    elif error_type is not None:
        return None
    return generation


def _spent_recipe_repair_generations(
    repair_root: Path,
    *,
    scope_tag: str,
    stable_id: str,
    work_id: str,
    campaign_root_work_id: str,
    root_cause_fingerprint: str,
) -> int:
    """Count the contiguous spent repair budget for ONE work/refusal scope.

    Sol wave-8 finding 5 and its R2 follow-up: a bare per-model glob let a
    NEW work generation, a requeue, or a different refusal class inherit an
    earlier scope's spent budget, and a malformed row (boolean generation,
    foreign stable ID, unbound filename suffix) still counted. A row now
    counts ONLY when its filename carries this scope's FULL digest, its
    generation suffix reproduces its content, and every field of the closed
    charge schema validates against the current context; spent budget is the
    contiguous generation prefix ``1..N``. Resuming the SAME work generation
    therefore keeps its charges; a newly minted work generation (which also
    escapes a striker-A park row) starts with its full budget, and malformed,
    foreign, gapped, or legacy residue is inert.

    Parameters
    ----------
    repair_root:
        The model's durable ``repair/`` directory.
    scope_tag:
        Full scope digest from :func:`_recipe_repair_scope_tag`.
    stable_id, work_id, campaign_root_work_id, root_cause_fingerprint:
        Exact identities every counted row must reproduce.

    Returns
    -------
    int
        Number of already-consumed repair generations in this scope.
    """

    prefix = f"recipe-refusal-charge-{scope_tag}-"
    generations: set[int] = set()
    for path in repair_root.glob(f"{prefix}*.json"):
        suffix = path.name.removeprefix(prefix).removesuffix(".json")
        if not suffix.isdigit():
            continue
        try:
            row = _read_json(path)
        except Exception:  # noqa: BLE001 -- malformed residue never counts as spent budget
            continue
        generation = _valid_recipe_repair_charge(
            row,
            filename_generation=int(suffix),
            stable_id=stable_id,
            work_id=work_id,
            campaign_root_work_id=campaign_root_work_id,
            root_cause_fingerprint=root_cause_fingerprint,
        )
        if generation is not None:
            generations.add(generation)
    spent = 0
    while spent + 1 in generations:
        spent += 1
    return spent


def _physical_memory_bytes() -> int:
    """Return host physical memory when POSIX page counters are available."""

    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        return 0


def _worker_observed_rss_deferral_row(
    item: WorkItem,
    artifact: AuthorArtifact,
    attempts: Sequence[Mapping[str, Any]],
    *,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
) -> Optional[JsonObject]:
    """Build an observed-memory deferral keyed on the observation itself (DX-6).

    RE-6: the valve used to require static pre-admission sizing to have been
    ``NOT_DERIVABLE``, so a model whose declared config DID derive -- and was
    admitted -- still minted a permanent ``failed:resource`` when the live
    worker blew the cap (m12627/m12704/m12733). The parent's own RSS
    observation is stronger evidence than any estimate: an all-RSS-cap failed
    generation now defers whatever the static verdict said, and that static
    verdict is RECORDED on the row (the refusal predicate) instead of gating
    it. Mixed history is scoped to the ACTIVE work generation, so a stale
    non-RSS failure from an earlier generation cannot block the valve.

    Parameters
    ----------
    item, artifact:
        Scheduled model and its executable authored recipe.
    attempts:
        Durable attempt history (any generations; scoped here).
    campaign_id, run_id, machine_id, created_at:
        Canonical campaign provenance for the deferral row.

    Returns
    -------
    dict[str, Any] | None
        A validated-ledger-shaped deferral row only when the active
        generation's failed attempts are all RSS-cap observations carrying
        complete peak/limit evidence.
    """

    failed = [
        attempt
        for attempt in attempts
        if attempt.get("result") == "failed"
        and str(attempt.get("work_id")) == item.active_work_id
    ]
    rss_failed = [
        attempt
        for attempt in failed
        if isinstance(attempt.get("error"), Mapping)
        and attempt["error"].get("stage") == "resource"
        and attempt["error"].get("reason_code") == "rss-cap"
    ]
    # DX-7/RE-7: sibling slots suppressed after the kill are derived from the
    # same observation and never block the valve.
    sibling_blocked = [
        attempt
        for attempt in failed
        if isinstance(attempt.get("error"), Mapping)
        and attempt["error"].get("reason_code") == "blocked-by-sibling"
    ]
    if not rss_failed or len(rss_failed) + len(sibling_blocked) != len(failed):
        return None
    implementation = artifact.proposal.get("proposed_facts", {}).get("implementation", {})
    assessment = assess_model_capacity(implementation)

    observations: list[JsonObject] = []
    for attempt in rss_failed:
        supervisor = attempt.get("supervisor_observation")
        invocation = attempt.get("invocation")
        host = attempt.get("host")
        peak = supervisor.get("peak_rss_bytes") if isinstance(supervisor, Mapping) else None
        rss_limit = invocation.get("rss_limit_bytes") if isinstance(invocation, Mapping) else None
        if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
            return None
        if not isinstance(rss_limit, int) or isinstance(rss_limit, bool) or rss_limit <= 0:
            return None
        observations.append(
            {
                "attempt_id": attempt.get("attempt_id"),
                "mode": invocation.get("mode") if isinstance(invocation, Mapping) else None,
                "peak_rss_bytes": peak,
                "enforced_rss_limit_bytes": rss_limit,
                # DX-8/RE-8: the watermark facts travel with the observation
                # when the supervising parent recorded them (0 on older rows).
                "rss_kill_threshold_bytes": (
                    supervisor.get("rss_kill_threshold_bytes", 0)
                    if isinstance(supervisor, Mapping)
                    else 0
                ),
                "rss_overshoot_bytes": (
                    supervisor.get("rss_overshoot_bytes", 0)
                    if isinstance(supervisor, Mapping)
                    else 0
                ),
                "host_ram_bytes": host.get("ram_bytes") if isinstance(host, Mapping) else None,
                "host_architecture": (
                    host.get("architecture") if isinstance(host, Mapping) else None
                ),
                "host_machine_id": host.get("machine_id") if isinstance(host, Mapping) else None,
            }
        )
    max_peak = max(int(observation["peak_rss_bytes"]) for observation in observations)
    max_limit = max(int(observation["enforced_rss_limit_bytes"]) for observation in observations)
    host_ram = max(
        (
            int(observation["host_ram_bytes"])
            for observation in observations
            if isinstance(observation.get("host_ram_bytes"), int)
        ),
        default=0,
    )
    architecture = next(
        (
            str(observation["host_architecture"])
            for observation in observations
            if isinstance(observation.get("host_architecture"), str)
        ),
        "unknown",
    )
    capacity: JsonObject = {
        "verdict": CapacityVerdict.DEFER.value,
        "threshold": {
            "host_physical_memory_bytes": host_ram,
            "host_memory_source": "worker-attempt-host-receipt",
            "worker_rss_limit_bytes": max_limit,
        },
        "estimate": {
            "origin": "worker-observed-rss",
            "max_peak_rss_bytes": max_peak,
            # sol U-S9: the observed peak is a LOWER BOUND on what the model
            # needs (the kill truncated the trajectory), never an estimate of
            # what would have sufficed.
            "observed_peak_basis": "lower-bound",
            "attempts": observations,
        },
        # DX-6/RE-6: the static pre-admission predicate is recorded as the
        # refusal's context, not consulted as its gate -- the observation is
        # the evidence.
        "static_assessment": {
            "verdict": assessment.verdict.value,
            "explanation": assessment.explanation,
        },
        "explanation": (
            f"pre-admission sizing said {assessment.verdict.value!r}; worker supervision "
            f"then observed RSS reach {max_peak} bytes against the enforced "
            f"{max_limit}-byte cap"
        ),
    }
    row: JsonObject = {
        "schema_version": CAPACITY_DEFERRAL_SCHEMA_VERSION,
        "stable_id": item.stable_id,
        "work_id": item.active_work_id,
        "name": item.intake.name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": CAPACITY_DEFERRAL_DISPOSITION,
        "capacity": capacity,
        "recheck_hint": (
            f"re-run on {architecture} hardware whose worker RSS allowance exceeds the "
            f"observed {max_peak}-byte peak plus execution headroom; the peak is a lower "
            "bound on the requirement, never an estimate"
        ),
    }
    return stamp_row_digest(row)


class ReceiptDriverMixin:
    """Driver-side worker lifecycle, receipt, and run-award orchestration."""

    if TYPE_CHECKING:
        _reduced: int

        def __getattr__(self, name: str) -> Any:
            """Describe collaborators supplied by the concrete driver facade."""

            raise AttributeError(name)

    def _defer_worker_observed_rss(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        attempts: Sequence[Mapping[str, Any]],
        operational: JsonlLedger,
    ) -> bool:
        """Persist a recoverable observed-RSS deferral when evidence permits it.

        Parameters
        ----------
        item, artifact, attempts:
            Active model generation, authored recipe, and complete attempt history.
        operational:
            Locked operational ledger receiving the matching campaign-health event.

        Returns
        -------
        bool
            Whether an observed-capacity deferral was durably recorded.
        """

        created_at = self.dependencies.clock()
        row = _worker_observed_rss_deferral_row(
            item,
            artifact,
            attempts,
            campaign_id=bound_campaign_id(self.config),
            run_id=self.config.run_id,
            machine_id=self.config.machine_id,
            created_at=created_at,
        )
        if row is None:
            return False
        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = capacity_deferral_path(records_root)
        persisted = append_capacity_deferral_row(ledger_path, row)
        # Partition accounting requires the valve to have fired THIS run: without the
        # stash, a worker-observed-rss deferral resurrects the PartitionError the
        # deferral arm exists to prevent (W3-P composition requirement). Imported at
        # call time: driver_admission imports this module at top level.
        from menagerie.crawler.driver_admission import _stash_machine_deferral_row

        _stash_machine_deferral_row(self, item.stable_id, persisted)
        estimate = persisted["capacity"]["estimate"]
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "host-capacity-deferred",
                "origin": "worker-observed-rss",
                "ledger": str(ledger_path.relative_to(records_root)),
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "max_peak_rss_bytes": estimate["max_peak_rss_bytes"],
                "observations": estimate["attempts"],
                "recheck_hint": persisted["recheck_hint"],
            },
        }
        event = mint_operational_event(event)
        operational.append(event)
        return True

    def _park_non_model_fault(
        self,
        item: WorkItem,
        artifact: Optional[AuthorArtifact],
        error: Mapping[str, Any],
        operational: JsonlLedger,
        *,
        sink_guard: bool = False,
    ) -> None:
        """Park one host- or machinery-owned failure without a model terminal.

        BL-12(e) / catalog C-16: a fault whose cause is the host or the harness
        machinery is infrastructure weather, never a verdict about the model.
        The attempt row already carries the typed evidence; this appends the
        durable campaign-health event naming the fault and leaves the member
        PENDING, so a later leg re-drives it once the weather clears. Nothing
        here upgrades an outcome: the member holds no award and no terminal.

        BL-8 (a): the park additionally mints the durable scheduler-quarantine
        row, so THIS leg settles the member (no gate spend on a run that never
        executed, no award dead end, no no-progress tripwire) and the terminal
        partition accounts it as machine-deferred. The row auto-releases: the
        next leg re-drives the member fresh, exactly as before.

        Parameters
        ----------
        item, artifact:
            Active model generation and its authored recipe (absent for
            pre-artifact admission faults).
        error:
            Structured classified failure carrying stage/reason/owner.
        operational:
            Locked operational ledger receiving the campaign-health event.
        sink_guard:
            Whether the ``_terminalize`` mint-sink backstop (DX-1/RE-1) caught
            this, rather than a classification arm. A ``true`` here means an
            UNGUARDED call site tried to settle a model head on a host/machinery
            fault -- the census reads it as a code defect to chase, never as
            normal weather.
        """

        details = error.get("details") if isinstance(error.get("details"), Mapping) else {}
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.RUNNER_FAILED.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "non-model-fault-parked",
                "owner": _failure_owner(error),
                "stage": error.get("stage"),
                "reason_code": error.get("reason_code"),
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "message": str(error.get("message"))[:1500],
                "detail_owner_basis": (
                    "attempt-details"
                    if isinstance(details, Mapping) and details.get("owner")
                    else "reason-owner-map"
                ),
                "sink_guard": sink_guard,
            },
        }
        operational.append(mint_operational_event(event))
        # The park settles the member for THIS run: it joins the run-scoped
        # withhold set (the same set the checker-lane valve uses), so no later
        # lane this run can land a head beside the live quarantine row -- the
        # LC-1 pinned "leg contradicted itself" close refusal stays
        # unreachable, and the bounded retry is fresh-leg only, exactly as the
        # quarantine arm documents (DX-1/RE-1).
        self._checker_lane_parked_ids.add(item.stable_id)
        self._record_scheduler_quarantine(
            item,
            site="env-turn",
            reason_code="non-model-fault",
            fault=(
                f"{_failure_owner(error)}-owned {error.get('stage')} failure "
                f"({error.get('reason_code')}); infrastructure weather, no verdict "
                "about the model -- member withheld from this leg, boundedly "
                "retried next legs (W12)"
            ),
            operational=operational,
        )

    def _append_worker_lifecycle_event(
        self,
        operational: JsonlLedger,
        *,
        event_kind: str,
        status: str,
        lease_id: str,
        stable_id: str,
        details: Mapping[str, Any],
    ) -> None:
        """Append one idempotent worker lifecycle event."""

        operational.append(
            mint_operational_event(
                {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": self.dependencies.clock(),
                    "event_kind": event_kind,
                    "status": status,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 0},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": {"lease_id": lease_id, "stable_id": stable_id, **dict(details)},
                },
                identity_payload={"lease_id": lease_id, "details": dict(details)},
            )
        )

    def _promote_award_corpus(
        self,
        item: WorkItem,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
    ) -> None:
        """Promote the awarded model's retained documents into the text corpus.

        Idempotent and digest-checked (`text_corpus.promote_author_root`), so a
        replayed award converges on the identical corpus. Refusals -- evidence
        bytes whose recomputed digest no longer matches custody -- are disclosed
        durably on the operational ledger; the refused work dir additionally
        stays refused by the reap-clearance guard until an operator resolves it.
        """

        author_root = self.paths.work_root / item.stable_id / "author"
        if not author_root.is_dir():
            return
        try:
            report = promote_author_root(
                author_root,
                self.paths.ledgers.records_root,
                license_dispositions=license_dispositions_from_artifact_events(
                    reducer.artifact_ledger.events, item.stable_id
                ),
            )
        except LedgerConflictError as exc:
            # Item 9 / author-uplift R-7: the corpus index refused an
            # immutable-identity replay (pilot m5717, 2026-08-22). The award
            # is already committed and this promotion is post-commit, so the
            # fault is the crawler's own machinery, never a verdict about the
            # member: it used to escape to the scheduler's blast-radius arm and
            # park the AWARDED member ``member-fault`` beside its live head. It
            # is disclosed durably here instead (the tripwire fired: nothing
            # was written to the corpus), and the idempotent replay re-attempts
            # the promotion on re-drive once an operator resolves the conflict.
            conflict: JsonObject = {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "created_at": self.dependencies.clock(),
                "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                "status": OperationalEventStatus.RUNNER_FAILED.value,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 0},
                "current_environment": None,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": {
                    "disposition": "text-corpus-promotion-conflict",
                    "owner": "machinery",
                    "stable_id": item.stable_id,
                    "work_id": item.active_work_id,
                    "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc)[:1500],
                },
            }
            operational.append(mint_operational_event(conflict))
            return
        if not report.refusals:
            return
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 0},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "text-corpus-promotion-refusals",
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "refusals": [dict(row) for row in report.refusals],
            },
        }
        event = mint_operational_event(event)
        operational.append(event)

    def _ensure_pending_run_anchors(
        self,
        work: Sequence[WorkItem],
        artifacts: dict[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> None:
        """Require private custody before checker backoff.

        Parameters
        ----------
        work, artifacts:
            Current scheduled items and mutable normalized artifact map.
        reducer, operational, state:
            Locked canonical stores used to terminalize one failed staging transaction.

        Notes
        -----
        Pending mechanical work remains private. Publication is authorized only after the
        accepted checker decision is part of the reducer authority projection.
        """

        del reducer, state
        for item in work:
            artifact = artifacts.get(item.stable_id)
            if artifact is not None and artifact.staged is None:
                # BL-8 (a): one member's custody defect is that member's fault
                # alone. The refusal is unchanged -- this member executes
                # nothing without verified private custody -- but it parks
                # typed and durable instead of killing the leg for every
                # sibling. No verdict is minted; the next leg re-stages it.
                self._record_scheduler_quarantine(
                    item,
                    site="run-anchors",
                    reason_code="member-fault",
                    fault=(
                        "pending mechanical work must execute from verified "
                        "private custody; the staged transaction is absent"
                    ),
                    operational=operational,
                )
                artifacts.pop(item.stable_id, None)

    def _forward_and_reduce(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        environment: EnvironmentBinding,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        award_run: bool,
        closure: Optional[ExecutableClosureV3] = None,
        verification_token: Optional[EnvironmentVerificationToken] = None,
    ) -> Optional[str]:
        """Append honest worker attempts and return a checker pause from run repair.

        Parameters
        ----------
        item, artifact, environment, reducer, operational, state:
            Exact scheduled work, authority, ledgers, and durable driver state.
        award_run:
            Whether successful attempts may advance to run award.
        closure:
            Optional closure already collected once for this artifact in the pass.
        verification_token:
            Optional cache-created currentness-pass proof used for closure collection.

        Returns
        -------
        str | None
            Checker pause reason, or ``None`` after ordinary reduction.
        """

        if isinstance(self.dependencies.forward, SupervisedForwardLane):
            collected_closure = closure or _collect_worker_executable_closure(
                artifact,
                environment,
                verification_token=verification_token,
            )
            closure_identity = collected_closure.identity
        else:
            collected_closure = None
            closure_identity = _driver_receipt_dependencies().injected_forward_closure_identity()
        execution_identity = _driver_receipt_dependencies().execution_identity(
            artifact, environment, closure_identity
        )
        self.dependencies.boundary_hook("pre-forward", item.stable_id)
        self._check_shutdown(
            "forward-admission",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        attempts = _matching_attempts(
            self.paths.ledgers.attempts,
            artifact.proposal,
            environment,
            execution_identity,
        )
        rung = artifact.proposal.get("proposed_facts", {}).get("source_resolution", {}).get("rung")
        cold_runs = cold_forward_policy(item.stable_id, rung).required_cold_forwards
        if not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs):
            generated: tuple[Mapping[str, Any], ...]
            cache_identity = stable_hash(
                {
                    "execution_identity": execution_identity,
                    "work_id": artifact.proposal.get("work_id"),
                }
            )
            cache = (
                self.paths.work_root
                / item.stable_id
                / f"driver-forward-attempts-{cache_identity[7:23]}.json"
            )
            persisted_by_lane = isinstance(self.dependencies.forward, SupervisedForwardLane)
            attempts_persisted_by_lane = False

            def persist_worker_attempt(attempt: Mapping[str, Any]) -> None:
                """Persist one honest attempt before its worker lease closes."""

                candidate = _without_ledger_fields(attempt)
                reducer.append_attempt(
                    _redact_attempt_diagnostics(
                        candidate,
                        None,
                        _diagnostics_root_for_work_root(self.paths.work_root),
                    )
                )
                self.dependencies.boundary_hook("after-attempt", item.stable_id)

            def persist_worker_lifecycle(event_kind: str, status: str, lease: WorkerLease) -> None:
                """Persist a lock-ordered worker lifecycle transition."""

                self._append_worker_lifecycle_event(
                    operational,
                    event_kind=event_kind,
                    status=status,
                    lease_id=lease.lease_id,
                    stable_id=lease.stable_id,
                    details={
                        "work_id": lease.work_id,
                        "execution_identity": lease.execution_identity,
                        "request_identity": lease.request_identity,
                        "child_pid": lease.child_pid,
                    },
                )
                if event_kind == OperationalEventKind.WORKER_LEASE_STARTED.value:
                    self.dependencies.boundary_hook(event_kind, lease.stable_id)

            def blocked_worker_attempt(
                cold_index: int, mode: str, kill: Mapping[str, Any]
            ) -> JsonObject:
                """Mint the typed blocked-by-sibling slot evidence (DX-7/RE-7)."""

                blocked_exc = WorkerSiblingBlocked(
                    f"slot cold-{cold_index + 1}/{mode} not spawned: sibling "
                    f"{kill.get('kind')} kill at "
                    f"cold-{int(kill.get('cold_index', 0)) + 1}/{kill.get('mode')}"
                )
                blocked_attempt = _driver_failure_attempt(
                    item,
                    artifact,
                    "resource",
                    "blocked-by-sibling",
                    blocked_exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=environment.family,
                    created_at=self.dependencies.clock(),
                )
                blocked_error = blocked_attempt.get("error")
                if isinstance(blocked_error, dict):
                    blocked_details = blocked_error.setdefault("details", {})
                    if isinstance(blocked_details, dict):
                        blocked_details["blocked_by_sibling"] = dict(kill)
                return blocked_attempt

            def resolve_worker_attempt(cold_index: int, mode: str) -> AttemptSlotResolution:
                """Authenticate one canonical deterministic slot before any capability opens."""

                resolution = resolve_attempt_slot_state(
                    scan_jsonl(self.paths.ledgers.attempts),
                    work_id=str(artifact.proposal["work_id"]),
                    execution_identity=execution_identity,
                    cold_index=cold_index,
                    mode=mode,
                    # DX-10/RE-11: a host/machinery-owned failure row is weather,
                    # never slot satisfaction -- the slot re-runs at the next
                    # typed retry generation.
                    non_model_failure=lambda error: _failure_owner(error)
                    in NEVER_MODEL_TERMINAL_OWNERS,
                )
                resolved = resolution.record
                if resolved is None:
                    return resolution
                try:
                    authority = load_current_attempt_proof(resolved)
                except AuthorityDerivationError as exc:
                    raise DriverIntegrationError(str(exc)) from exc
                if resolved.get("result") == "succeeded" and authority is None:
                    raise DriverIntegrationError(
                        "canonical execution slot lacks authenticated success authority"
                    )
                return resolution

            cached: JsonObject | None = None
            if cache.is_file() and not persisted_by_lane:
                try:
                    candidate = _read_json(cache)
                except Exception:  # noqa: BLE001 -- disposable replay cache is regenerable
                    cache.unlink(missing_ok=True)
                else:
                    if (
                        candidate.get("work_id") == artifact.proposal.get("work_id")
                        and candidate.get("execution_identity") == execution_identity
                    ):
                        cached = candidate
                    else:
                        cache.unlink(missing_ok=True)
            if cached is not None:
                generated = tuple(
                    dict(value)
                    for value in cached.get("attempts", [])
                    if isinstance(value, Mapping)
                )
            else:
                try:
                    if isinstance(self.dependencies.forward, SupervisedForwardLane):
                        generated = tuple(
                            self.dependencies.forward.forward(
                                artifact,
                                environment,
                                cold_runs,
                                self.paths.work_root,
                                worker_lock_path=self.paths.worker_lock,
                                worker_lease_path=self.paths.worker_lease,
                                run_id=self.config.run_id,
                                shutdown_event=self._shutdown_event,
                                lifecycle_event=persist_worker_lifecycle,
                                attempt_sink=persist_worker_attempt,
                                attempt_resolver=resolve_worker_attempt,
                                closure=collected_closure,
                                blocked_attempt_factory=blocked_worker_attempt,
                            )
                        )
                        attempts_persisted_by_lane = True
                    else:
                        generated = tuple(
                            self.dependencies.forward.forward(
                                artifact,
                                environment,
                                cold_runs,
                                self.paths.work_root,
                            )
                        )
                except Exception as exc:  # noqa: BLE001 -- supervisor failure is model-local
                    failure_owner: Optional[str] = None
                    if _is_sandbox_unavailable(exc):
                        # The HOST's sandbox machinery failed, not the model
                        # (BL-12(e) lane a): a 30-second logd brownout must
                        # never permanently burn whichever model was executing.
                        stage, reason = ("policy", "sandbox-unavailable-v1")
                        failure_owner = "host"
                    elif isinstance(exc, WorkerUnreapableError):
                        # A D-state worker root is a kernel/host fact
                        # (BL-12(b)): bounded halt with evidence, never a
                        # model verdict.
                        stage, reason = ("runner", "worker-unreapable")
                        failure_owner = "host"
                    elif isinstance(exc, RecipeError):
                        # Admission-time recipe refusal (stale pin, drifted
                        # staged bytes): a recipe-contract fact about THIS
                        # proposal, classified exactly like the worker's own
                        # pre-import refusal so the bounded author-repair route
                        # below can adjudicate it.
                        stage, reason = ("import", "import-exception")
                    else:
                        # The same infrastructure-vs-model triage the repair
                        # sibling arm has always had (BL-12(e) lane c): thirty
                        # minutes of disk pressure must not burn every model
                        # scheduled in the window.
                        stage, reason = ("runner", "internal-error")
                        failure_owner = (
                            "host" if self._is_infrastructure_error(exc) else "machinery"
                        )
                    failure_attempt = _driver_failure_attempt(
                        item,
                        artifact,
                        stage,
                        reason,
                        exc,
                        self.config,
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                        environment=environment.family,
                        created_at=self.dependencies.clock(),
                    )
                    if failure_owner is not None:
                        error_value = failure_attempt.get("error")
                        if isinstance(error_value, dict):
                            details_value = error_value.setdefault("details", {})
                            if isinstance(details_value, dict):
                                details_value["owner"] = failure_owner
                    generated = (failure_attempt,)
                _write_json_atomic(
                    cache,
                    {
                        "work_id": artifact.proposal["work_id"],
                        "execution_identity": execution_identity,
                        "attempts": list(generated),
                    },
                )
            if not attempts_persisted_by_lane:
                for attempt in generated:
                    persist_worker_attempt(attempt)
            attempts = _matching_attempts(
                self.paths.ledgers.attempts,
                artifact.proposal,
                environment,
                execution_identity,
            )
            if not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs):
                all_attempts = _matching_model_attempts(
                    self.paths.ledgers.attempts, artifact.proposal
                )
                expansion = _detected_mode_expansion(all_attempts, artifact.proposal)
                if expansion is not None:
                    if not any(
                        attempt.get("error", {}).get("details", {}).get("route")
                        == "recipe-and-gate-revision-required"
                        for attempt in all_attempts
                        if isinstance(attempt.get("error"), Mapping)
                    ):
                        expansion_attempt = _driver_failure_attempt(
                            item,
                            artifact,
                            "input",
                            "contract-invalid",
                            DriverIntegrationError(
                                "worker detected meaningful modes absent from the gated proposal"
                            ),
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        )
                        expansion_attempt["attempt_id"] = stable_hash(
                            {
                                "work_id": artifact.proposal["work_id"],
                                "route": "recipe-and-gate-revision-required",
                                "detected_meaningful_modes": expansion["detected_meaningful_modes"],
                            }
                        )
                        expansion_attempt["error"]["details"] = deepcopy(expansion)
                        expansion_attempt["error"]["root_cause_fingerprint"] = stable_hash(
                            expansion
                        )
                        persisted_expansion = reducer.append_attempt(expansion_attempt).record
                        all_attempts = (*all_attempts, persisted_expansion)
                    try:
                        repaired = self._repair_author_for_detected_modes(
                            item,
                            artifact,
                            expansion,
                            reducer,
                            operational=operational,
                        )
                    except AuthorBackoffError as backoff:
                        # Provider usage exhaustion during a mode repair is a
                        # campaign pause, not a failed run. `_forward_and_reduce`
                        # already returns a pause reason from run repair, so route
                        # it through the same channel instead of the blanket arm.
                        return self._pause_for_usage(backoff.signal, operational, 1)
                    except AuthorRepairTerminal as terminal:
                        # A typed terminal recommendation from the mode-repair
                        # generation is an ANSWER, not a repair failure. Caught
                        # ahead of the blanket arm below, which would record it
                        # as `failed:runner / protocol-violation` and throw the
                        # verdict away; routed to the same terminal-disposition
                        # gate the first-call arm uses, exactly like the three
                        # `_ensure_gates` repair sites.
                        return self._route_repair_terminal_author_result(
                            item,
                            terminal,
                            reducer,
                            operational,
                            state,
                            # The worker already ran under this work generation, so
                            # the reducer will derive per-mode and environment proof
                            # from these ledger attempts; the terminal record must
                            # carry them or it contradicts that proof and refuses.
                            prior_attempts=tuple(all_attempts),
                        )
                    except (AuthorEffortCapExceeded, AuthorEffortExhaustionClaim) as exc:
                        stage, reason = self._repair_failure_stage_and_reason(exc)
                        repair_failure = reducer.append_attempt(
                            _driver_failure_attempt(
                                item,
                                artifact,
                                stage,
                                reason,
                                exc,
                                self.config,
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
                                environment=environment.family,
                                created_at=self.dependencies.clock(),
                            )
                        ).record
                        if getattr(exc, "dimension", None) == "provider-storm":
                            # W1: a storm-classified exhaustion is withheld on
                            # the deferral valve, never terminalized.
                            self._mint_quota_storm_deferral(
                                item,
                                evidence={
                                    "witness": "author-sessions-exhaustion-classifier",
                                    "exception_type": (
                                        f"{type(exc).__module__}.{type(exc).__qualname__}"
                                    ),
                                },
                            )
                            return None
                        from menagerie.crawler.driver_admission import (
                            _model_lane_failure_event,
                        )

                        # PD-13: the exhaustion terminal's cause is evented BEFORE
                        # the terminal mints, so it survives a bookkeeping refusal.
                        operational.append(
                            _model_lane_failure_event(
                                stable_id=item.stable_id,
                                work_id=item.active_work_id,
                                status_code=f"failed:{stage}",
                                reason_code=reason,
                                exc=exc,
                                run_id=self.config.run_id,
                                machine_id=self.config.machine_id,
                                created_at=self.dependencies.clock(),
                            )
                        )
                        self._terminalize(
                            item,
                            artifact,
                            f"failed:{stage}",
                            reason,
                            str(exc),
                            (*all_attempts, repair_failure),
                            reducer,
                            operational,
                            state,
                            human_review=False,
                        )
                        return None
                    except ModeRepairUnsatisfiable as exc:
                        # DX-9/RE-9: the demanded revision can NEVER validate on
                        # this contract -- a typed input-contract refusal naming
                        # the contradiction, with zero author spend.
                        unsatisfiable_attempt = _driver_failure_attempt(
                            item,
                            artifact,
                            "input",
                            "contract-invalid",
                            exc,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(
                                self.paths.work_root
                            ),
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        )
                        unsat_error = unsatisfiable_attempt.get("error")
                        if isinstance(unsat_error, dict):
                            unsat_details = unsat_error.setdefault("details", {})
                            if isinstance(unsat_details, dict):
                                unsat_details["route"] = "mode-contract-unsatisfiable"
                        persisted_unsat = reducer.append_attempt(unsatisfiable_attempt).record
                        self._terminalize(
                            item,
                            artifact,
                            "failed:input",
                            "contract-invalid",
                            str(exc),
                            (*all_attempts, persisted_unsat),
                            reducer,
                            operational,
                            state,
                            human_review=False,
                        )
                        return None
                    except ModeRepairBudgetExhausted as exc:
                        # DX-9/RE-9: budget exhaustion publishes as the AUTHOR
                        # stage, never runner. An infrastructure-burned budget
                        # (cause chain) stays weather and parks (DX-1).
                        if self._is_infrastructure_error(exc):
                            weather_attempt = _driver_failure_attempt(
                                item,
                                artifact,
                                "runner",
                                "internal-error",
                                exc,
                                self.config,
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
                                environment=environment.family,
                                created_at=self.dependencies.clock(),
                            )
                            _stamp_failure_owner(weather_attempt, "host")
                            persisted_weather = reducer.append_attempt(weather_attempt).record
                            weather_error = persisted_weather.get("error")
                            if isinstance(weather_error, Mapping):
                                self._park_non_model_fault(
                                    item, artifact, weather_error, operational
                                )
                            return None
                        exhausted_attempt = reducer.append_attempt(
                            _driver_failure_attempt(
                                item,
                                artifact,
                                "author",
                                "repair-exhausted",
                                exc,
                                self.config,
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
                                environment=environment.family,
                                created_at=self.dependencies.clock(),
                            )
                        ).record
                        self._terminalize(
                            item,
                            artifact,
                            "failed:author",
                            "repair-exhausted",
                            str(exc),
                            (*all_attempts, exhausted_attempt),
                            reducer,
                            operational,
                            state,
                            human_review=False,
                        )
                        return None
                    except Exception as exc:  # noqa: BLE001 -- bounded repair is model-local
                        reason = (
                            "protocol-violation"
                            if isinstance(exc, DriverIntegrationError)
                            and not self._is_infrastructure_error(exc)
                            else "internal-error"
                        )
                        failure_attempt = _driver_failure_attempt(
                            item,
                            artifact,
                            "runner",
                            reason,
                            exc,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(
                                self.paths.work_root
                            ),
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        )
                        if reason == "internal-error":
                            _stamp_failure_owner(
                                failure_attempt,
                                "host" if self._is_infrastructure_error(exc) else "machinery",
                            )
                        repair_failure = reducer.append_attempt(failure_attempt).record
                        repair_error = repair_failure.get("error")
                        if (
                            isinstance(repair_error, Mapping)
                            and _failure_owner(repair_error) in NEVER_MODEL_TERMINAL_OWNERS
                        ):
                            # DX-1/RE-1 (m12446 shape): a host/machinery fault inside
                            # the bounded mode-repair lane is infrastructure weather,
                            # never this model's verdict -- park loudly and leave the
                            # member pending for the next leg.
                            self._park_non_model_fault(item, artifact, repair_error, operational)
                            return None
                        self._terminalize(
                            item,
                            artifact,
                            "failed:runner",
                            reason,
                            str(exc),
                            (*all_attempts, repair_failure),
                            reducer,
                            operational,
                            state,
                        )
                        return None
                    return self._regate_and_forward_repaired(
                        item,
                        repaired,
                        environment,
                        reducer,
                        operational,
                        state,
                        award_run=award_run,
                        verification_token=verification_token,
                    )
                failed_attempts = [
                    attempt for attempt in all_attempts if attempt["result"] == "failed"
                ]
                # DX-6/RE-6: an RSS-cap row is capacity EVIDENCE, not a verdict
                # candidate -- when the generation carries any other failure,
                # that failure is the decisive one; an all-RSS-cap generation
                # is the deferral valve's case below, never a terminal's.
                # DX-7/RE-7: a blocked-by-sibling row is DERIVED evidence and
                # can never be decisive either.
                failure = next(
                    (
                        attempt
                        for attempt in reversed(failed_attempts)
                        if not (
                            isinstance(attempt.get("error"), Mapping)
                            and (
                                (
                                    attempt["error"].get("stage") == "resource"
                                    and attempt["error"].get("reason_code") == "rss-cap"
                                )
                                or attempt["error"].get("reason_code")
                                == "blocked-by-sibling"
                            )
                        )
                    ),
                    failed_attempts[-1] if failed_attempts else None,
                )
                if failure is None:
                    integration_error = DriverIntegrationError(
                        f"worker attempts do not satisfy modes/cold policy for {item.stable_id}"
                    )
                    failure = reducer.append_attempt(
                        _driver_failure_attempt(
                            item,
                            artifact,
                            "runner",
                            "protocol-violation",
                            integration_error,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        )
                    ).record
                    all_attempts = (*all_attempts, failure)
                error = failure["error"]
                if not isinstance(error, Mapping):
                    raise DriverIntegrationError("failed attempt lost its structured error")
                refusal = _deterministic_recipe_refusal(
                    failure, _diagnostics_root_for_work_root(self.paths.work_root)
                )
                if refusal is not None:
                    try:
                        repaired_for_refusal = self._repair_author_for_recipe_refusal(
                            item,
                            artifact,
                            refusal,
                            reducer,
                            operational=operational,
                        )
                    except AuthorBackoffError as backoff:
                        # Provider usage exhaustion during a recipe repair is a
                        # campaign pause, not a failed run -- same channel as
                        # the detected-modes repair arm above.
                        return self._pause_for_usage(backoff.signal, operational, 1)
                    except AuthorRepairTerminal as terminal:
                        # A typed terminal recommendation from the repair
                        # generation is an ANSWER, adjudicated through the same
                        # terminal-disposition gate as every other repair site.
                        return self._route_repair_terminal_author_result(
                            item,
                            terminal,
                            reducer,
                            operational,
                            state,
                            prior_attempts=tuple(all_attempts),
                        )
                    except (AuthorEffortCapExceeded, AuthorEffortExhaustionClaim) as exc:
                        stage, reason = self._repair_failure_stage_and_reason(exc)
                        repair_failure = reducer.append_attempt(
                            _driver_failure_attempt(
                                item,
                                artifact,
                                stage,
                                reason,
                                exc,
                                self.config,
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
                                environment=environment.family,
                                created_at=self.dependencies.clock(),
                            )
                        ).record
                        if getattr(exc, "dimension", None) == "provider-storm":
                            # W1: a storm-classified exhaustion is withheld on
                            # the deferral valve, never terminalized.
                            self._mint_quota_storm_deferral(
                                item,
                                evidence={
                                    "witness": "author-sessions-exhaustion-classifier",
                                    "exception_type": (
                                        f"{type(exc).__module__}.{type(exc).__qualname__}"
                                    ),
                                },
                            )
                            return None
                        from menagerie.crawler.driver_admission import (
                            _model_lane_failure_event,
                        )

                        # PD-13: the exhaustion terminal's cause is evented BEFORE
                        # the terminal mints, so it survives a bookkeeping refusal.
                        operational.append(
                            _model_lane_failure_event(
                                stable_id=item.stable_id,
                                work_id=item.active_work_id,
                                status_code=f"failed:{stage}",
                                reason_code=reason,
                                exc=exc,
                                run_id=self.config.run_id,
                                machine_id=self.config.machine_id,
                                created_at=self.dependencies.clock(),
                            )
                        )
                        self._terminalize(
                            item,
                            artifact,
                            f"failed:{stage}",
                            reason,
                            str(exc),
                            (*all_attempts, repair_failure),
                            reducer,
                            operational,
                            state,
                            human_review=False,
                        )
                        return None
                    except Exception as exc:  # noqa: BLE001 -- bounded repair is model-local
                        reason = (
                            "protocol-violation"
                            if isinstance(exc, DriverIntegrationError)
                            and not self._is_infrastructure_error(exc)
                            else "internal-error"
                        )
                        failure_attempt = _driver_failure_attempt(
                            item,
                            artifact,
                            "runner",
                            reason,
                            exc,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(
                                self.paths.work_root
                            ),
                            environment=environment.family,
                            created_at=self.dependencies.clock(),
                        )
                        if reason == "internal-error":
                            _stamp_failure_owner(
                                failure_attempt,
                                "host" if self._is_infrastructure_error(exc) else "machinery",
                            )
                        repair_failure = reducer.append_attempt(failure_attempt).record
                        repair_error = repair_failure.get("error")
                        if (
                            isinstance(repair_error, Mapping)
                            and _failure_owner(repair_error) in NEVER_MODEL_TERMINAL_OWNERS
                        ):
                            # DX-1/RE-1 (m12446 shape): a host/machinery fault inside
                            # the bounded recipe-repair lane is infrastructure
                            # weather, never this model's verdict -- park loudly and
                            # leave the member pending for the next leg.
                            self._park_non_model_fault(item, artifact, repair_error, operational)
                            return None
                        self._terminalize(
                            item,
                            artifact,
                            "failed:runner",
                            reason,
                            str(exc),
                            (*all_attempts, repair_failure),
                            reducer,
                            operational,
                            state,
                        )
                        return None
                    if repaired_for_refusal is not None:
                        return self._regate_and_forward_repaired(
                            item,
                            repaired_for_refusal,
                            environment,
                            reducer,
                            operational,
                            state,
                            award_run=award_run,
                            verification_token=verification_token,
                        )
                    # Repair budget exhausted: fall through to the HONEST
                    # import terminal carrying the original typed refusal --
                    # never a failed:runner protocol lie.
                if self._defer_worker_observed_rss(
                    item,
                    artifact,
                    all_attempts,
                    operational,
                ):
                    return None
                if error.get("stage") == "resource" and error.get("reason_code") in {
                    "rss-cap",
                    "blocked-by-sibling",
                }:
                    # DX-6/RE-6: reaching here means the generation's failures
                    # were ALL RSS-cap kills (or their derived blocked-sibling
                    # rows -- the decisive picker prefers any other failure)
                    # and the deferral valve still could not build its row --
                    # the observation evidence (peak/limit) is incomplete. That
                    # is a telemetry/machinery gap about the supervision, never
                    # a model verdict: no ``failed:resource`` head may mint
                    # without a recorded capacity verdict.
                    parked_error = dict(error)
                    parked_details = dict(parked_error.get("details") or {})
                    parked_details["owner"] = "machinery"
                    parked_details["capacity_valve"] = "observation-evidence-incomplete"
                    parked_error["details"] = parked_details
                    self._park_non_model_fault(item, artifact, parked_error, operational)
                    return None
                stage = str(error["stage"])
                # BL-12(e) / C-16: a fact about the HOST or the harness may
                # never settle a model's head. Host- and machinery-owned
                # failures park loudly (durable campaign-health event, member
                # left pending for the next leg) instead of minting a
                # permanent, false model terminal.
                if _failure_owner(error) in NEVER_MODEL_TERMINAL_OWNERS:
                    self._park_non_model_fault(item, artifact, error, operational)
                    return None
                self._terminalize(
                    item,
                    artifact,
                    f"failed:{stage}",
                    str(error["reason_code"]),
                    None,
                    all_attempts,
                    reducer,
                    operational,
                    state,
                    human_review=stage == "import",
                )
                return None
            self.dependencies.boundary_hook("after-forward", item.stable_id)
        if not award_run:
            return None
        return self._award_worker_result(
            item,
            artifact,
            attempts,
            reducer,
            operational,
            state,
            environment_family=environment.family,
            execution_identity=execution_identity,
        )

    def _award_recorded_run(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> tuple[bool, Optional[str]]:
        """Award ``runs`` from already-recorded attempts without an env re-entry.

        The pipelined scheduler's award step (D1 fork F2 -> full-matrix triage):
        the triage turn recorded the complete ``required_cold_runs x
        meaningful_modes`` attempt matrix, so once the metadata (and required
        fidelity) gates accept, the award is a pure writer act over those
        recorded attempts. The proof bar is unchanged -- the same
        ``_assemble_run_model`` / publication / ``append_model`` tail the wave
        driver runs, fed by attempts whose identities are read back from the
        ledger instead of recomputed against a live prefix.

        Parameters
        ----------
        item, artifact, reducer, operational, state:
            Exact scheduled work, authority, ledgers, and durable driver state.

        Returns
        -------
        tuple[bool, str | None]
            ``(awarded, pause_reason)``. ``awarded`` is ``False`` when the
            recorded attempts do not satisfy the execution policy for the
            CURRENT proposal identity (a gate-round repair changed it) -- the
            scheduler must re-execute through an ordinary env turn instead.
        """

        attempts, execution_identity, environment_family = _recorded_matching_attempts(
            self.paths.ledgers.attempts,
            artifact.proposal,
        )
        rung = artifact.proposal.get("proposed_facts", {}).get("source_resolution", {}).get("rung")
        cold_runs = cold_forward_policy(item.stable_id, rung).required_cold_forwards
        if (
            execution_identity is None
            or environment_family is None
            or not _attempt_policy_satisfied(attempts, artifact.proposal, cold_runs)
        ):
            return (False, None)
        pause = self._award_worker_result(
            item,
            artifact,
            attempts,
            reducer,
            operational,
            state,
            environment_family=environment_family,
            execution_identity=execution_identity,
        )
        return (True, pause)

    def _retire_generation_exhausted_award(
        self,
        item: WorkItem,
        failure: GenerationExhaustedError,
        operational: JsonlLedger,
    ) -> None:
        """Retire one generation-exhausted award loudly, pre-spend (AT-5).

        The member's decisive attempts span environment generations with no
        single-generation satisfying set: no coherent terminal proof exists.
        The runs themselves are not a defect, so no failed terminal is minted
        and no chain is spent -- the member discloses TW-GENSPAN (DEGRADED)
        and parks behind the typed quarantine valve, auto-releasing next leg
        (re-execution under the current generation heals the split).
        """

        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.DEGRADED.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "terminal-proof-generation-exhausted",
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "error": str(failure)[:1_500],
                "remedy": (
                    "no publication and no failed terminal: the member parks "
                    "typed and re-executes under the current environment "
                    "generation"
                ),
            },
        }
        event = mint_operational_event(event)
        operational.append(event)
        self._record_scheduler_quarantine(
            item,
            site="award",
            reason_code="terminal-proof-generation-exhausted",
            fault=failure,
            operational=operational,
        )

    def _award_worker_result(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        attempts: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        environment_family: str,
        execution_identity: str,
    ) -> Optional[str]:
        """Run the award tail: assembly, publication authorization, model append.

        Shared verbatim by the wave driver's in-turn award (`_forward_and_reduce`
        with ``award_run=True``) and the scheduler's recorded-attempt award
        (`_award_recorded_run`); the only environment fact it consumes is the
        family label stamped onto synthetic failure attempts.

        Parameters
        ----------
        item, artifact, attempts, reducer, operational, state:
            Exact scheduled work, satisfying attempts, authority, and state.
        environment_family:
            Environment family label for synthetic failure attempts.
        execution_identity:
            Execution identity binding the satisfying attempts.

        Returns
        -------
        str | None
            Checker pause reason, or ``None`` after ordinary reduction.
        """

        self.dependencies.boundary_hook("post-attempt-pre-award", item.stable_id)
        self._check_shutdown(
            "post-attempt-pre-award",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        gates = scan_jsonl(self.paths.ledgers.gates)
        representative_model = (
            reducer.current_records.get(item.family_representative_id)
            if item.is_family_variant
            else None
        )
        try:
            model = _assemble_run_model(
                item,
                artifact,
                attempts,
                gates,
                self.config,
                representative_model=representative_model,
                author_sessions_used=chargeable_author_sessions(
                    (
                        *reducer.artifact_ledger.events,
                        *operational.records,
                    ),
                    item.stable_id,
                    item.active_work_id,
                    paid_attempts=self._paid_author_attempts(
                        item.stable_id, item.active_work_id
                    ),
                ),
            )
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
            # Genuinely campaign-level: a pause, a retryable transport failure, or a
            # provider backoff must never be recorded as this model's defect.
            # `RetryableOperatorError` subclasses `DriverIntegrationError`, so it would
            # otherwise be caught by the arm below.
            raise
        except GenerationExhaustedError as exc:
            # AT-5: no single-generation satisfying set exists. The runs are
            # not a defect -- the environment rotated between modes -- so this
            # is never a failed terminal: retire loudly BEFORE any
            # publication and park behind the typed quarantine valve.
            self._retire_generation_exhausted_award(item, exc, operational)
            return None
        except Exception as exc:  # noqa: BLE001 -- run assembly is model-local
            # Only family VARIANTS used to get a model-local handler here; a
            # representative re-raised unconditionally and ended the campaign. But
            # `_assemble_run_model` refuses for conditions that are not variant-specific
            # -- a missing fidelity gate, worker receipts that disagree with the
            # proposal-declared meaningful-mode set, accepted attempts that fail the
            # clean execution policy. Every one of those is a fact about THIS model,
            # witnessed by its own attempt, exactly like the sibling arms above. The
            # refusals themselves are unchanged and remain as strict as they were.
            failure = reducer.append_attempt(
                _driver_failure_attempt(
                    item,
                    artifact,
                    "runner",
                    "protocol-violation",
                    exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=environment_family,
                    created_at=self.dependencies.clock(),
                )
            ).record
            self._terminalize(
                item,
                artifact,
                "failed:runner",
                "protocol-violation",
                str(exc),
                (*attempts, failure),
                reducer,
                operational,
                state,
            )
            return None
        current_model = reducer.current_records.get(item.stable_id)
        model["parent_revision"] = (
            current_model["record_revision"] if current_model is not None else None
        )
        if current_model is not None:
            model["status"]["supersedes_revision"] = current_model["record_revision"]

        self.dependencies.boundary_hook("pre-publication", item.stable_id)
        self._check_shutdown(
            "pre-publication-admission",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        self.dependencies.boundary_hook("pre-award-commit", item.stable_id)
        self._check_shutdown(
            "pre-award-commit",
            item=item,
            work_id=str(artifact.proposal["work_id"]),
            execution_identity=execution_identity,
        )
        self.dependencies.boundary_hook("award-commit-entered", item.stable_id)
        # Graceful-shutdown atomic award section: publication authorization and
        # materialization must remain check-free through the canonical model append.
        # POST-COMMIT publication failures are DELIBERATELY outside the model-local
        # arm below. A failed public write is not this model's defect: it leaves a
        # committed authorization on the artifact ledger and owns its own resume
        # protocol, which republishes from that capability and still awards the model
        # `runs`. Terminalizing here would burn a good model permanently because one
        # mirror write failed, and would destroy the resume path by planting a
        # terminal the resume cannot supersede. A PRE-COMMIT authorization refusal is
        # the opposite case (rung 11: `authorization dependency vector is stale`):
        # nothing durable happened, the reducer's exactness tripwire spoke about THIS
        # model's authority state, and letting it propagate aborted the whole
        # campaign. It terminalizes model-scoped below, witnessed by its own attempt,
        # with the refusal preserved verbatim; the tripwire itself is unchanged.
        # A run award blocked ONLY by a missing grant on a review-held terminal
        # must not publish: the pending revision parks below with its artifact
        # still private, the operator sees exactly what a grant would publish,
        # and grant consumption stays the only publish path.
        blocked_previous = reducer.grant_blocked_supersession(model)
        # AT-4 pre-flight (B1-XQ28, PD-14 bankable half): run the reducer's
        # ENTIRE append-validation chain as a dry run BEFORE the artifact
        # chain is spent. Every append-refusal class -- a stale authorization
        # dependency vector (rung 11), a generation-spanning proof (m12646) --
        # now refuses PRE-publication, leaving the chain at private custody
        # instead of a spent 4-event chain under a failed terminal. The dry
        # run validates against the CURRENT (staged) authority state, so a
        # refusal that only manifests against final-event authority can still
        # reach the post-publication append arm below; the pre-flight shrinks
        # the spent class, the arm below stays the backstop.
        try:
            reducer.preflight_model(reducer.prepare_model(model))
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
            raise
        except GrantPendingSupersessionError:
            # Coherent with the blocked_previous pre-check: publication is
            # withheld below and the REAL append parks the grant durably.
            pass
        except GenerationExhaustedError as exc:
            self._retire_generation_exhausted_award(item, exc, operational)
            return None
        except Exception as exc:  # noqa: BLE001 -- pre-spend refusals are model-local
            refusal_event = mint_operational_event(
                award_seam_refusal_event(
                    publication_lane="award-worker",
                    stable_id=item.stable_id,
                    work_id=str(artifact.proposal["work_id"]),
                    failure=exc,
                    run_id=self.config.run_id,
                    machine_id=self.config.machine_id,
                    created_at=self.dependencies.clock(),
                    extra_details={"seam": "append-preflight", "chain_spent": False},
                )
            )
            operational.append(refusal_event)
            failure = reducer.append_attempt(
                _driver_failure_attempt(
                    item,
                    artifact,
                    "runner",
                    "protocol-violation",
                    exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=environment_family,
                    created_at=self.dependencies.clock(),
                )
            ).record
            self._terminalize(
                item,
                artifact,
                "failed:runner",
                "protocol-violation",
                str(exc),
                (*attempts, failure),
                reducer,
                operational,
                state,
            )
            return None
        spanned_generations = per_mode_generation_span(
            attempts,
            stable_id=item.stable_id,
            work_id=str(artifact.proposal["work_id"]),
        )
        if len(spanned_generations) > 1:
            # TW-GENSPAN: the independent newest-per-mode picks span
            # generations; the coherent selection (AT-5) chose one and the
            # pre-flight above proved the proof admits, but the split itself
            # is a campaign fact worth a DEGRADED disclosure.
            span_event: JsonObject = {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "created_at": self.dependencies.clock(),
                "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                "status": OperationalEventStatus.DEGRADED.value,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 1},
                "current_environment": None,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": {
                    "disposition": "terminal-proof-generation-split",
                    "stable_id": item.stable_id,
                    "work_id": str(artifact.proposal["work_id"]),
                    "spanned_generations": list(spanned_generations),
                    "remedy": (
                        "generation-coherent selection chose one generation's "
                        "satisfying set; the newer split evidence is disclosed, "
                        "never silently mixed into the proof"
                    ),
                },
            }
            span_event = mint_operational_event(span_event)
            operational.append(span_event)
        chain_spent = False
        if (
            model.get("authored_metadata_state") == "accepted"
            and not isinstance(artifact, ActivatedHandoffArtifact)
            and blocked_previous is None
        ):
            try:
                self._authorize_and_publish_artifact(artifact, model, gates, reducer)
                chain_spent = True
            except PublicationAuthorizationRefused as refused:
                seam_event = mint_operational_event(
                    award_seam_refusal_event(
                        publication_lane="award-worker",
                        stable_id=item.stable_id,
                        work_id=str(artifact.proposal["work_id"]),
                        failure=refused,
                        run_id=self.config.run_id,
                        machine_id=self.config.machine_id,
                        created_at=self.dependencies.clock(),
                        extra_details={"seam": "publication", "chain_spent": False},
                    )
                )
                operational.append(seam_event)
                failure = reducer.append_attempt(
                    _driver_failure_attempt(
                        item,
                        artifact,
                        "runner",
                        "protocol-violation",
                        refused,
                        self.config,
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                        environment=environment_family,
                        created_at=self.dependencies.clock(),
                    )
                ).record
                self._terminalize(
                    item,
                    artifact,
                    "failed:runner",
                    "protocol-violation",
                    str(refused),
                    (*attempts, failure),
                    reducer,
                    operational,
                    state,
                    human_review=True,
                )
                return None
        prepared = reducer.prepare_model(model)
        # Captured BEFORE the append so an award-over-award supersession can be
        # disclosed (AW-1): 63 silent runs -> runs remints at the step-100
        # close, zero events -- nothing compared the new award to the one it
        # replaced. Raw head, never the projection: a stale award is still an
        # award.
        preceding_raw_head = reducer.raw_head_view.get(item.stable_id)
        try:
            result = reducer.append_model(prepared)
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
            raise
        except GrantPendingSupersessionError as refused:
            if blocked_previous is None:
                # The pre-check and the reducer disagreed; nothing was withheld
                # coherently, so the refusal stays loud.
                raise
            # The one parkable refusal: every other award validation passed, so
            # the completed run waits durably for an operator grant instead of
            # burning the model as failed:runner and failing the close.
            from menagerie.crawler.driver_admission import (
                _park_grant_pending_supersession,
            )

            _park_grant_pending_supersession(self, item, prepared, refused, operational)
            return None
        except Exception as exc:  # noqa: BLE001 -- the canonical append is model-local
            if isinstance(exc, ReductionError) and chain_spent:
                # AT-4: the reducer's typed refusal manifested only AFTER the
                # chain was spent (the pre-flight passed against staged
                # authority). Minting a failed terminal here would pair a
                # published final event with a failure head -- m12646's
                # published_with_failure_head incoherence -- so the member
                # parks typed instead, with the seam disclosure carrying the
                # spent-chain fact. Infrastructure faults (OSError et al.)
                # deliberately fall through to the terminal path below and
                # its unrecordable-ladder containment.
                seam_event = mint_operational_event(
                    award_seam_refusal_event(
                        publication_lane="award-worker",
                        stable_id=item.stable_id,
                        work_id=str(artifact.proposal["work_id"]),
                        failure=exc,
                        run_id=self.config.run_id,
                        machine_id=self.config.machine_id,
                        created_at=self.dependencies.clock(),
                        extra_details={"seam": "append", "chain_spent": True},
                    )
                )
                operational.append(seam_event)
                self._record_scheduler_quarantine(
                    item,
                    site="award",
                    reason_code="award-blocked:spent-chain",
                    fault=exc,
                    operational=operational,
                )
                return None
            # The canonical award append had no handler between here and the CLI
            # catch-all either, so a reducer refusal for ONE model -- exactly what the
            # source-link invariant is for -- ended the whole campaign. The refusal is
            # unchanged and this model is still NOT awarded: it terminalizes as a runner
            # protocol violation witnessed by its own attempt. Only the blast radius
            # shrinks.
            failure = reducer.append_attempt(
                _driver_failure_attempt(
                    item,
                    artifact,
                    "runner",
                    "protocol-violation",
                    exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=environment_family,
                    created_at=self.dependencies.clock(),
                )
            ).record
            self._terminalize(
                item,
                artifact,
                "failed:runner",
                "protocol-violation",
                str(exc),
                (*attempts, failure),
                reducer,
                operational,
                state,
            )
            return None
        if blocked_previous is not None:
            raise DriverIntegrationError(
                "grant-pending pre-check withheld publication but the reducer "
                f"accepted the supersession for {item.stable_id}"
            )
        if result.appended:
            self._reduced += 1
            if (
                preceding_raw_head is not None
                and preceding_raw_head.get("status", {}).get("kind") == "runs"
                and result.record.get("status", {}).get("kind") == "runs"
            ):
                # Award-over-award disclosure (AW-1, adjudicated): the remint
                # stays admissible -- admission control belongs to the work-
                # selection policy -- but it is never silent again. Degraded,
                # never healthy: nothing compared the new award to the old one.
                remint_event: JsonObject = {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": self.dependencies.clock(),
                    "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                    "status": OperationalEventStatus.DEGRADED.value,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 1},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": {
                        "disposition": "award-reminted",
                        "stable_id": item.stable_id,
                        "work_id": item.active_work_id,
                        "superseded_revision": str(preceding_raw_head["record_revision"]),
                        "reminted_revision": str(result.record["record_revision"]),
                    },
                }
                remint_event = mint_operational_event(remint_event)
                operational.append(remint_event)
        # Reduced (even if deduplicated): this model is no longer strandable.
        with self._author_lane_state_lock:
            self._reduced_model_ids.add(item.stable_id)
        # Fetch-once retention: promote every document this model's crawl holds
        # into the durable text corpus the moment the award is committed. Runs
        # on replayed awards too -- promotion is idempotent and a resumed leg
        # must converge on the same corpus. Failures here are post-commit, like
        # publication: they never terminalize the awarded model, they propagate
        # loud, and the idempotent replay finishes the promotion on re-drive.
        self._promote_award_corpus(item, reducer, operational)
        self.dependencies.boundary_hook("post-award-commit", item.stable_id)
        self._check_shutdown("post-award-commit")
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_records
        snapshot = self._policy_snapshot()
        self._handle_progress(operational, current_records, snapshot, state=state)
        if self._maybe_pause_for_review(operational, current_records, snapshot, state):
            raise DriverPaused("review checkpoint reached")
        state["last_terminal_count"] = len(current_records)
        state["status"] = "running"
        _write_driver_state(self.paths.driver_state, state)
        return None

    def _regate_and_forward_repaired(
        self,
        item: WorkItem,
        repaired: AuthorArtifact,
        environment: EnvironmentBinding,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        award_run: bool,
        verification_token: Optional[EnvironmentVerificationToken],
    ) -> Optional[str]:
        """Gate one repaired proposal generation and forward it when accepted.

        The shared tail of every post-worker repair arm (detected modes,
        recipe refusals): the repaired proposal must pass a fresh single-item
        gate round before it may forward again, and a parked or non-``runs``
        outcome returns without executing.

        Parameters
        ----------
        item, repaired, environment, reducer, operational, state:
            Scheduled work, repaired generation, bound runtime, and ledgers.
        award_run:
            Whether successful attempts may advance to run award.
        verification_token:
            Optional currentness-pass proof reused for closure collection.

        Returns
        -------
        str | None
            Checker pause reason, or ``None`` after ordinary reduction.
        """

        repaired_artifacts = {item.stable_id: repaired}
        pause = self._ensure_gates((item,), repaired_artifacts, reducer, operational, state)
        if item.stable_id in self._checker_lane_parked_ids:
            # The checker lane failed even the single-item gate for the
            # repaired proposal, so the model is parked on the durable
            # checker-lane deferral valve: it must not forward without
            # an accepted gate, and it requeues on the next run.
            return pause
        current = reducer.current_records.get(item.stable_id)
        if current is not None and current.get("status", {}).get("kind") != "runs":
            return pause
        if pause is not None:
            return pause
        repaired = repaired_artifacts[item.stable_id]
        self._family_artifacts[item.stable_id] = repaired
        repaired_closure = (
            _collect_worker_executable_closure(
                repaired,
                environment,
                verification_token=verification_token,
            )
            if isinstance(self.dependencies.forward, SupervisedForwardLane)
            else None
        )
        return self._forward_and_reduce(
            item,
            repaired,
            environment,
            reducer,
            operational,
            state,
            award_run=award_run,
            closure=repaired_closure,
            verification_token=verification_token,
        )

    def _repair_author_for_recipe_refusal(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        refusal: str,
        reducer: CanonicalReducer,
        *,
        operational: JsonlLedger,
    ) -> Optional[AuthorArtifact]:
        """Request bounded proposal revisions for a deterministic recipe refusal.

        The refusal is a recipe-CONTRACT fact: the loader rejected the declared
        recipe before any import ran, so re-running the same bytes cannot pass
        and terminalizing ``failed:import`` burns a repairable proposal. Each
        bounded generation carries the EXACT refusal text to the repairing
        author through the one feedback channel
        (:func:`~menagerie.crawler.author_attempts.record_checker_findings`),
        exactly like a checker repair.

        Unlike the detected-modes loop, a driver-ACCEPTED revision also
        consumes the cap: only the worker can prove the refusal is fixed, so an
        uncharged accepted-but-still-refused revision would recurse through
        this repair forever. A dead session that never answered still charges
        nothing (the rung-11 m10949 lesson), because the charge marker is
        written only after a revision COMPLETES.

        Parameters
        ----------
        item, artifact:
            Current model and the proposal generation the loader refused.
        refusal:
            Exact typed ``RecipeError`` text from the failed attempt.
        reducer:
            Active canonical writer carrying exact author and custody authority.
        operational:
            Canonical operational ledger for pre-dispatch bought-session truth.

        Returns
        -------
        AuthorArtifact | None
            A new identity-validated proposal generation, or ``None`` when the
            bounded budget is exhausted -- the caller then terminalizes on the
            HONEST import refusal, never a runner protocol lie.

        Raises
        ------
        AuthorRepairTerminal
            If a repair generation answers with a typed terminal
            recommendation; that is an ANSWER for the caller to adjudicate.
        """

        # Imported at call time: driver_admission imports this module at top level.
        from menagerie.crawler.driver_admission import (
            _normalize_artifact_modes,
            _routed_environment_packages,
            _validate_artifact_identities,
        )

        repair_root = self.paths.work_root / item.stable_id / "repair"
        refusal_fingerprint = stable_hash({"kind": "recipe-refusal", "refusal": refusal})
        work_id = str(artifact.proposal.get("work_id"))
        campaign_root_work_id = str(artifact.campaign_root_work_id)
        # The budget is granted per (work generation, campaign lineage, refusal
        # class), never per bare stable ID: durable markers carry the triple in
        # both filename tag and content, so a new work generation or a
        # different refusal class can never inherit an earlier scope's spent
        # budget (Sol wave-8 finding 5), while resuming the SAME work keeps it.
        scope_tag = _recipe_repair_scope_tag(
            work_id=work_id,
            campaign_root_work_id=campaign_root_work_id,
            root_cause_fingerprint=refusal_fingerprint,
        )
        finding = (
            "EXECUTION-LANE RECIPE REFUSAL (deterministic: re-running the same recipe "
            "cannot pass). The recipe loader refused this proposal BEFORE any model "
            "import ran. Revise the implementation facts (library_recipe, or the typed "
            "adapter and its identities) so this exact refusal cannot recur, and rerun "
            f"the identity calculator on the FINAL facts. Refusal, verbatim: {refusal}"
        )
        spent = _spent_recipe_repair_generations(
            repair_root,
            scope_tag=scope_tag,
            stable_id=item.stable_id,
            work_id=work_id,
            campaign_root_work_id=campaign_root_work_id,
            root_cause_fingerprint=refusal_fingerprint,
        )
        for generation in range(spent + 1, self.config.run_repair_max + 1):
            request = {
                "stable_id": item.stable_id,
                "generation": generation,
                "gate_kind": "recipe_refusal",
                "route": "recipe-revision-required",
                "work_id": work_id,
                "campaign_root_work_id": campaign_root_work_id,
                "proposal_sha256": artifact.proposal["proposal_sha256"],
                "refusal": refusal,
                "root_cause_fingerprint": refusal_fingerprint,
            }
            repair_path = repair_root / f"recipe-refusal-generation-{scope_tag}-{generation}.json"
            if not repair_path.is_file():
                _write_json_atomic(repair_path, request)
                record_checker_findings(
                    self.paths.work_root / item.stable_id / "author",
                    gate_kind="recipe_refusal",
                    generation=generation,
                    required_repairs=[finding],
                    root_cause_fingerprint=refusal_fingerprint,
                )
            charge = {
                "stable_id": item.stable_id,
                "generation": generation,
                "gate_kind": "recipe_refusal",
                "work_id": work_id,
                "campaign_root_work_id": campaign_root_work_id,
                "root_cause_fingerprint": refusal_fingerprint,
                "disposition": "revision-rejected",
                "error_type": None,
            }
            try:
                repaired = self._retry_infrastructure_call(
                    lambda: self.dependencies.author.author(
                        item,
                        self.paths.work_root,
                        self.config,
                        reducer.context,
                    ),
                    admission=("author", item),
                    operational=operational,
                )
                repaired = self._stage_author_result(
                    item, repaired, reducer, operational_events=operational.records
                )
                if not isinstance(repaired.author_result, ProposedAuthorResult):
                    # A typed DEFER/SKIP/BLOCKED answer is the author's ANSWER,
                    # not a failed revision; it must not consume the cap.
                    raise AuthorRepairTerminal(repaired, "recipe_refusal")
                repaired = _normalize_artifact_modes(
                    repaired,
                    self.config,
                    environment_packages=_routed_environment_packages(
                        self.registry, item.route.intent
                    ),
                )
                if repaired.proposal.get("stable_id") != item.stable_id:
                    raise DriverIntegrationError(
                        "recipe-refusal repair proposal stable_id does not match intake"
                    )
                if repaired.proposal.get("work_id") != artifact.proposal.get("work_id"):
                    # The work identity is machine-issued and constant across
                    # repair generations, exactly like the detected-modes lane.
                    raise DriverIntegrationError(
                        "recipe-refusal repair proposal drifted off its "
                        "machine-issued work identity"
                    )
                if repaired.campaign_root_work_id != artifact.campaign_root_work_id:
                    raise DriverIntegrationError("recipe-refusal repair changed campaign lineage")
                _validate_artifact_identities(repaired, self.config, item=item)
            except (RetryableOperatorError, AuthorRepairTerminal, AuthorBackoffError):
                # Infrastructure faults, terminal answers, and provider backoff
                # are not failed revisions and never consume the cap.
                raise
            except Exception as exc:  # noqa: BLE001 -- a completed, rejected revision consumes the cap
                charge["error_type"] = type(exc).__name__
                _write_json_atomic(
                    repair_root / f"recipe-refusal-charge-{scope_tag}-{generation}.json",
                    charge,
                )
                continue
            charge["disposition"] = "revision-accepted-for-execution"
            _write_json_atomic(
                repair_root / f"recipe-refusal-charge-{scope_tag}-{generation}.json",
                charge,
            )
            cache = author_result_cache_path(self.paths.work_root, item.stable_id)
            _write_json_atomic(
                cache,
                serialize_author_result_cache(
                    repaired.author_result,
                    source_manifest=repaired.source_manifest,
                    model_dir=repaired.model_dir,
                    repair_generation=generation,
                    staged_transaction_id=(
                        str(repaired.staged.transaction_id)
                        if repaired.staged is not None
                        else None
                    ),
                    lifecycle=(
                        "staged-pending" if repaired.staged is not None else "unstaged"
                    ),
                ),
            )
            return repaired
        return None
