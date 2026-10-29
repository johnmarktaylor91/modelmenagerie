"""Admission, environment materialization, and execution identity for the crawler."""

from __future__ import annotations

import ast
import errno
import json
import logging
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    cast,
)

from menagerie.crawler import author_queue
from menagerie.crawler.artifact_transactions import (
    ArtifactBindingError,
    ArtifactCheckpointError,
    ArtifactCheckpointIncrement,
    ArtifactCheckpointProjection,
    ArtifactEventKind,
    ArtifactEventLedger,
    ArtifactRehydrationError,
    ArtifactTransactionProjection,
    ArtifactTransitionError,
    rehydrate_artifact_transaction,
    resolve_final_artifact_transaction,
    SourceManifestBindingError,
    staged_artifact_for_result,
    validate_artifact_checkpoint,
)
from menagerie.crawler.cache_registry import (
    author_result_cache_path,
    startup_cache_registration_check,
)
from menagerie.crawler.checkpoint_cache import ValidationMemo
from menagerie.crawler.author_spend import (
    author_session_allowance,
    author_sessions_exhaustion_dimension,
    enforce_author_session_allowance,
)
from menagerie.crawler.author_dispatch import (
    AUTHOR_EXIT_PERMANENT,
    AUTHOR_EXIT_RETRYABLE,
    AUTHOR_EXIT_UNAVAILABLE,
    executor_typed_outage_outcome,
    AuthorBackoffSignal,
    AuthorEffortGrant,
    AuthorEffortExhaustionClaim,
    AuthorEngineFaultError,
    AuthorExecutorContractError,
    AuthorInfrastructureFaultError,
    AuthorResultMalformedError,
    AuthorSourceDeterministicallyUnfetchableError,
    BlockedRecommendation,
    ProposedAuthorResult,
    build_author_envelope,
    build_recorded_result_envelope,
    classify_author_response,
    host_capacity_block,
    parse_author_reset_at,
    plausible_author_reset_at,
    routed_environment_block,
    serialize_author_result_cache,
    validate_author_result,
    validate_author_result_cache,
    validate_author_result_mapping,
)
from menagerie.crawler.authority import (
    ArtifactTransactionId,
    AuthorityDerivationError,
    AuthorityContext,
    DependencyState,
    EnvironmentAuthorityCache,
    EnvironmentAuthorityV1,
    EnvironmentExternalTarget,
    EnvironmentVerificationToken,
    ShutdownInterruptionFact,
    authority_epoch_identity,
    build_authority_context,
    derive_execution_identity,
    derive_runner_identity,
)
from menagerie.crawler.checker_calibration import require_checker_calibration
from menagerie.crawler.checker_dispatch import (
    CHECKER_WEATHER_BACKOFF_CEILING_SECONDS,
    CHECKER_WEATHER_BACKOFF_SECONDS,
    CheckerDispatchError,
    CheckerItemContractError,
    CheckerMechanicalClaimError,
    CheckerPromptRotatedError,
    CheckerWallBudgetExceeded,
    build_fidelity_envelope,
    build_metadata_vet_envelope,
    build_terminal_disposition_envelope,
    disclosed_metadata_complaint_union,
    gate_round_ceiling_state,
    leg_checker_prompt_sha256,
)
from menagerie.crawler.checker_lane import (
    CHECKER_LANE_DEFERRAL_DISPOSITION,
    append_checker_lane_deferral_row,
    build_checker_lane_deferral_row,
    checker_lane_deferral_path,
    validate_checker_lane_deferral_row,
)

# W4-B seam (crawler/w4b, batch economics): the typed partial-result refusal and
# the shared halving-ladder mechanism live in checker_dispatch. The failure route
# below consumes both; until that branch merges, byte-equivalent stand-ins keep
# this branch importable and are shadowed dead the moment the real names land.
try:
    from menagerie.crawler.checker_dispatch import (  # type: ignore[attr-defined]
        CheckerPartialResultError,
    )
except ImportError:  # pragma: no cover -- pre-integration fallback only

    class CheckerPartialResultError(CheckerDispatchError):  # type: ignore[no-redef]
        """Stand-in for W4-B's typed partial/extra-items checker refusal."""


try:
    from menagerie.crawler.checker_dispatch import (  # type: ignore[attr-defined]
        downshift_metadata_items,
    )
except ImportError:  # pragma: no cover -- pre-integration fallback only

    def downshift_metadata_items(  # type: ignore[no-redef]
        items: Sequence[_DownshiftItemT],
    ) -> tuple[tuple[_DownshiftItemT, ...], ...]:
        """Stand-in for W4-B's halving ladder: stable-order halves, floor one."""

        batch = tuple(items)
        if len(batch) <= 1:
            return (batch,) if batch else ()
        midpoint = (len(batch) + 1) // 2
        return (batch[:midpoint], batch[midpoint:])


from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    canonical_operational_ledger_path,
    canonical_records_root,
    canonical_requeue_grants_path,
    requeue_void_identity,
)
from menagerie.crawler.gate_reuse import (
    GATE_REUSED_FROM_FIELD,
    CheckerAdjudicationIdentity,
    lane_adjudication_identity,
    lane_confirmation_identity,
    plan_metadata_gate_reuse,
    reusable_fidelity_gate,
    reusable_metadata_gate,
    reusable_terminal_gate,
)
from menagerie.crawler.constants import (
    AUTHOR_PAID_SESSIONS_PER_ATTEMPT,
    author_leg_runaway_budget,
    resolve_author_wall_seconds,
    AUTHOR_QUEUE_POLL_SECONDS,
    AUTHOR_QUEUE_STALL_SECONDS,
    CHECKER_LANE_PARK_RECIDIVISM_LIMIT,
    CHECKER_LANE_PARK_STREAK_LIMIT,
    CHECKER_SESSION_EXHAUSTION_REASON_CODE,
    CHECKER_LEG_SESSION_BUDGET,
    CHECKER_LEG_WALL_BUDGET_SECONDS,
    CHECKER_MODEL_WALL_BUDGET_SECONDS,
    CHECKER_PROMPT_NAME,
    CUSTODY_REAPER_GRACE_SECONDS,
    DEFERRED_TERMINAL_GRANT_STAGE,
    METADATA_GATE_ROUND_CEILING,
    FAILURE_REASON_CODES,
    FORGE_RATE_LIMIT_OUTAGE_THRESHOLD,
    GRANT_CAPABILITIES,
    HOST_CAPABILITY_DEFERRAL_STATUS_CODES,
    HOST_CAPABILITY_ROUTING_CAPABILITY,
    METADATA_BATCH_MIN,
    REQUEUE_GRANT_SCHEMA_LADDER,
    REQUEUE_GRANT_SCHEMA_VERSION,
    REQUEUE_GRANT_VOID_SCHEMA_VERSION,
    RESEARCH_TOOL_OUTAGE_THRESHOLD,
    STDIO_TAIL_MAX_CHARS,
    AuthorPauseReason,
    InvocationOrigin,
    AUTHOR_RESULT_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION_V3,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    AUTHOR_WALL_EXTERNAL_KILL_FACTOR,
    AUTHOR_WALL_SECONDS_ENV,
    CAMPAIGN_TEARDOWN_RECORD_ENV,
    CAMPAIGN_TEARDOWN_RECORD_VERSION,
    GateKind,
    OperationalEventKind,
    OperationalEventStatus,
    author_lane_wall_bound,
    fidelity_gate_failure_reason,
    CheckerSeatRole,
    checker_provider_for_model,
    checker_seat,
)
from menagerie.crawler.environment_coverage import (
    COVERAGE_DEFERRAL_DISPOSITION,
    CoverageAssessment,
    _inventory_provides,
    append_coverage_deferral_row,
    assess_environment_coverage,
    build_coverage_deferral_row,
    coverage_deferral_path,
    distribution_vocabulary_rejection,
    validate_coverage_deferral_row,
)
from menagerie.crawler.environment_inventory import (
    ROUTED_INVENTORY_SOURCE_ID,
    routed_inventory_source_row,
)
from menagerie.crawler.environment_rebuild import (
    REBUILD_DEFERRAL_DISPOSITION,
    REBUILD_DEFERRAL_SCHEMA_VERSION,
    append_rebuild_deferral_row,
    build_rebuild_deferral_row,
    current_rebuild_rows,
    rebuild_deferral_path,
    validate_rebuild_deferral_row,
)
from menagerie.crawler.disk_floor import (
    DISK_FLOOR_DEFERRAL_DISPOSITION,
    append_disk_floor_deferral_row,
    build_disk_floor_deferral_row,
    current_disk_floor_rows,
    disk_floor_deferral_path,
    validate_disk_floor_deferral_row,
)
from menagerie.crawler.grant_parks import (
    AWARD_PARK_DISPOSITION,
    AWARD_PARK_EVENT_DISPOSITION,
    GRANT_PARK_EVENT_DISPOSITION,
    GRANT_PENDING_DISPOSITION,
    SELECTION_EXCLUSION_DISPOSITION,
    append_grant_park_row,
    build_grant_park_row,
    current_grant_park_rows,
    grant_park_path,
    grant_park_released,
    validate_grant_park_ledger_row,
)
from menagerie.crawler.park_release import current_park_tree_identity
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_DISPOSITION,
    append_refusal_park_row,
    build_refusal_park_row,
    current_refusal_park_rows,
    refusal_park_path,
    refusal_park_released,
    validate_refusal_park_ledger_row,
)
from menagerie.crawler.scheduler_parks import (
    SCHEDULER_QUARANTINE_DISPOSITION,
    SCHEDULER_QUARANTINE_PARK_EVENT_DISPOSITION,
    SCHEDULER_QUARANTINE_REUSED_DISPOSITION,
    append_scheduler_quarantine_row,
    build_scheduler_quarantine_row,
    current_scheduler_quarantine_rows,
    load_scheduler_quarantine_ledger,
    member_quarantine_reason,
    scheduler_quarantine_path,
    scheduler_quarantine_released,
    validate_scheduler_quarantine_ledger_row,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
    EnvironmentRegistry,
    IntentProbes,
    overlay_package_rows,
)
from menagerie.crawler.host_capacity import (
    CAPACITY_DEFERRAL_DISPOSITION,
    CapacityAssessment,
    CapacityDeferralError,
    append_capacity_deferral_row,
    assess_model_capacity,
    build_capacity_deferral_row,
    build_host_capability_claim,
    capacity_deferral_path,
    host_capability_still_refuses,
    host_capability_terminal_refusal,
    host_capacity,
    validate_capacity_deferral_row,
)
from menagerie.crawler.env_lifecycle import (
    ArtifactReceipt,
    DiskFloorError,
    EnvironmentBackendContractError,
    EnvironmentBackendUnavailableError,
    EnvironmentExactnessError,
    EnvironmentProbeError,
    PerIntentGenerationEffort,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
    disk_free_bytes,
    environment_turn_effort_path,
    installed_package_inventory_bytes,
    materialized_environment_generation,
    parse_exact_lock,
    parse_probe_receipt_bytes,
    parse_resolved_export,
    recover_environment_turn_effort_ledger,
    validate_probe_receipts,
)
from menagerie.crawler.evidence import trusted_intake_identity_mismatches
from menagerie.crawler.effort import StageCap
from menagerie.crawler.fetcher import (
    CasObjectCorruptError,
    DigestOrigin,
    FetchHashMismatchError,
    FetchRetrievalError,
    FetchTarget,
    UnpinnedTargetError,
    UpstreamContentDriftError,
)
from menagerie.crawler.family_templates import (
    FamilyTemplateError,
    instantiate_size_variant,
    mechanical_variant_parameter_input_line,
    specialize_size_variant_recipe,
    validate_size_variant,
)
from menagerie.crawler.gates import (
    GateRoutingError,
    emit_gate_records,
    route_fidelity_gate,
    route_metadata_gate,
)
from menagerie.crawler.grant_identity import (
    GrantIdentityError,
    expected_grant_id,
    requeue_adjudication_key,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    hash_bytes,
    stable_hash,
    stamp_row_digest,
    utc_now,
)
from menagerie.crawler.infra_release import (
    MACHINE_CLASS_RELEASE_AUTHORITY_PREFIX,
    RELEASE_AUTHORITY_AWARD_NAMING,
    RELEASE_AUTHORITY_HUMAN_REVIEW,
    RELEASE_AUTHORITY_INFRA_WAIVER,
    derive_infra_terminal_release,
    derive_machine_class_release,
)
from menagerie.crawler.intake import (
    IntakeError,
    IntakeItem,
    IntakeSnapshot,
    load_intake_snapshot,
    trusted_identity_fields,
)
from menagerie.crawler.metadata import (
    MetadataValidationError,
    canonical_meaningful_modes,
    recompute_accepted_identities,
)
from menagerie.crawler.models import SUPPLEMENTARY_SOURCES_KEY, JsonObject
from menagerie.crawler.mirrors import MirrorClass, MirrorStore
from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.operator_protocol import (
    CHECKER_SESSION_WALL_SECONDS,
    CHECKER_WALL_EXTERNAL_KILL_FACTOR,
    checker_session_kill_seconds,
    OPERATOR_DEADLINE_SECONDS,
    OPERATOR_INTER_ATTEMPT_BACKOFF_SECONDS,
    OPERATOR_PROTOCOL_VERSION,
    OPERATOR_STATUS_VERSION,
    OperatorExitCode,
    load_absolute_request,
    status_sidecar_path,
)
from menagerie.crawler.paid_session_custody import (
    ALIVE,
    CUSTODY_FD_ENV,
    FREE,
    PaidCustodyLease,
    custodian_provably_dead,
    custody_recovery_decision,
    open_subtree_lock,
    process_group_liveness,
    subtree_liveness,
)
from menagerie.crawler.paid_session_accounting import (
    ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES,
    AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME,
    BudgetPolicy,
    PaidSessionBudgetExceeded,
    PaidSessionError,
)
from menagerie.crawler.paid_session_lifecycle import (
    CommandProviderReconciler,
    PaidSessionAuthority,
    ReconcilerNegativeControlError,
    Reservation,
    authorized_paid_dispatch,
    current_paid_reservation,
    verify_reconciler_negative_control,
)
from menagerie.crawler.proposal import (
    ProposalValidationError,
    model_code_manifest,
    parse_device_policy,
)
from menagerie.crawler.recipe import RecipeError, bind_library_artifact_digest
from menagerie.crawler.storm_fuse import (
    CrossModelProviderStorm,
    CrossModelStormFuse,
)
from menagerie.crawler.recordio import (
    JsonlLedger,
    LedgerConflictError,
    LedgerCorruptionError,
    scan_jsonl,
)
from menagerie.crawler.reducer import (
    CanonicalReducer,
    ReductionError,
    _select_current,
    expected_standard_asset,
)
from menagerie.crawler.routing import (
    IntentRoute,
    phase_routes,
    route_model,
)
from menagerie.crawler.status import machine_deferral_rejection
from menagerie.crawler.work_selection import (
    _award_naming_grant,
    derive_generation_discharge,
)
from menagerie.crawler.source_broker import (
    BROKER_MANIFEST_ROLES,
    BROKER_PACK_VERSION,
    SourceBrokerError,
    broker_evidence_dirs,
    broker_source_pack,
    default_transport,
    promote_broker_evidence,
    write_broker_outputs,
)
from menagerie.crawler.wakeup import WakeupManager, reduce_wake_episodes
from menagerie.crawler.worker_supervisor import (
    # The hardened teardown: it proves the group is still ours (unreaped child plus an
    # unchanged process-start token) before it signals, so a recycled PID can never be
    # killed. Reused rather than re-derived; there must be exactly one such routine.
    _kill_process_group as kill_process_group,
    capture_process_group,
    current_boot_id,
    process_start_token,
    reconcile_worker_lease,
    signal_verified_process_group,
)

from menagerie.crawler.driver_contracts import (
    ActivatedHandoffArtifact,
    AuthorCampaignTeardown,
    AuthorArtifact,
    bound_campaign_id,
    AuthorBackoffError,
    AuthorEffortCapExceeded,
    AuthorQueueStalled,
    AuthorRepairTerminal,
    AuthorSessionRetryExhausted,
    AuthorUsagePause,
    CheckerOutcome,
    DriverConfig,
    DriverIntegrationError,
    DriverPaths,
    DriverPaused,
    DriverResult,
    DriverShutdown,
    EnvironmentBinding,
    MEMBER_CONTAINMENT_PASSTHROUGH,
    AuthorOutageError,
    ForgeRateLimitedError,
    RecordedIdentityBasis,
    ResearchToolsUnavailableError,
    RetryableOperatorError,
    StaleGateBindingError,
    VariantRecipeUnsupported,
    WorkItem,
    _campaign_id_for_item,
    _intake_discovery_urls,
)

from menagerie.crawler.checker_conflicts import (
    CheckerVerdictConflictError,
    accuracy_acceptance_witnesses,
    accuracy_duplicate_accept_state,
    accuracy_item_accepted,
    append_checker_verdict_conflict_row,
    build_accuracy_duplicate_accept_row,
    checker_verdict_conflict_path,
    load_checker_verdict_conflict_rows,
)
from menagerie.crawler.driver_models import (
    _artifact_lineage,
    _checker_item,
    _chargeable_metadata_repair_count,
    _driver_failure_attempt,
    _fidelity_gate_history,
    _fidelity_item_accepted,
    _fidelity_required,
    _find_gate,
    _gate_item_fingerprint,
    _matching_model_attempts,
    _metadata_batches,
    _metadata_gate_accepted,
    _metadata_gate_history,
    _metadata_repair_count,
    _normalize_gate_generation,
    _prepare_ledger_record,
    _require_gate,
    _require_gate_bindings,
    _require_legacy_audit_fidelity,
    _terminal_checker_item,
    _terminal_fidelity_gate,
    _terminal_metadata_gate,
    _usable_family_representative,
    _without_ledger_fields,
)

from menagerie.crawler.driver_progress import (
    EnvironmentFailureRoute,
    _environment_failure,
    _framework_from_intake,
    _read_json,
    _write_json_atomic,
)
from menagerie.crawler.discovery import (
    DiscoveryError,
    FoundDiscovery,
    AccessBlockedDiscovery,
    HigherTierDiscovery,
    NegativeDiscovery,
    RetryableToolFailureDiscovery,
    freeze_discovery_evidence,
    materialize_discovery_artifact,
    validate_source_discovery,
)

from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    SupervisedForwardLane,
    _collect_worker_executable_closure,
    _diagnostics_root_for_work_root,
    _failure_owner,
    _stamp_failure_owner,
)

LOGGER = logging.getLogger("menagerie.crawler.driver")

_DownshiftItemT = TypeVar("_DownshiftItemT")

_T = TypeVar("_T")


@dataclass(frozen=True)
class _DriverAdmissionDependencies:
    """Late-bound facade collaborators injected after lower modules load."""

    award_closure_identity: Callable[[], str]
    fetch_targets: Callable[[list[FetchTarget], Path], dict[str, object]]


_DRIVER_ADMISSION_DEPENDENCIES: Optional[_DriverAdmissionDependencies] = None


def _configure_driver_admission_dependencies(
    *,
    award_closure_identity: Callable[[], str],
    fetch_targets: Callable[[list[FetchTarget], Path], dict[str, object]],
) -> None:
    """Inject facade-owned late-bound collaborators without importing the facade."""

    global _DRIVER_ADMISSION_DEPENDENCIES
    _DRIVER_ADMISSION_DEPENDENCIES = _DriverAdmissionDependencies(
        award_closure_identity=award_closure_identity,
        fetch_targets=fetch_targets,
    )


def _driver_admission_dependencies() -> _DriverAdmissionDependencies:
    """Return the collaborators injected by the import-compatible facade."""

    if _DRIVER_ADMISSION_DEPENDENCIES is None:
        raise DriverIntegrationError("driver admission dependencies are not configured")
    return _DRIVER_ADMISSION_DEPENDENCIES


def _current_award_closure_identity() -> str:
    """Return the award closure identity exposed by the compatibility facade.

    Returns
    -------
    str
        Current award closure identity, including compatibility monkeypatches.
    """

    return _driver_admission_dependencies().award_closure_identity()


def _current_fetch_targets(
    targets: list[FetchTarget],
    cas_root: Path,
) -> dict[str, object]:
    """Fetch author sources through the compatibility facade.

    Parameters
    ----------
    targets:
        Exact pinned source targets.
    cas_root:
        Controlled source content-addressed store.

    Returns
    -------
    dict[str, object]
        Frozen aggregate source manifest.
    """

    return _driver_admission_dependencies().fetch_targets(targets, cas_root)


class AuthorEnvironmentBindingError(DriverIntegrationError):
    """A published author recipe contradicts its routed environment inventory."""


class _AuthorCacheSuperseded(DriverIntegrationError):
    """A resume's author-result cache anchors bytes the ledger already superseded.

    Carries the EXACT staged ``author_result_id`` set the discard retires, so the
    ``author-cache-discarded`` operational event can attest the lineage break and
    the author-session valve can stop charging those superseded-generation
    authorings against the lineage that re-authors under current authority (they
    remain spend-census facts). Only this exception -- never a corrupt-cache or
    ambiguous-authority discard -- retires anything.
    """

    def __init__(self, message: str, *, superseded_author_result_ids: frozenset[str]):
        super().__init__(message)
        self.superseded_author_result_ids = superseded_author_result_ids


def _author_lane_failure(exc: Exception) -> tuple[str, str]:
    """Map one model-local author-lane exception to its closed stage and reason.

    The default arm is deliberately last. Everything ahead of it names a cause an
    operator can act on; collapsing a controlled-fetch rejection into
    ``identity-unresolved`` sent operators hunting for a missing implementation
    when the real cause was the author's pinned-target declaration or the
    retrieval itself.

    Parameters
    ----------
    exc:
        Exception raised while resolving, fetching, or authoring this model.

    Returns
    -------
    tuple[str, str]
        Closed ``(stage, reason_code)`` pair from ``FAILURE_REASON_CODES``.
    """

    # PLAN.md LP-13.2: cap exhaustion is `failed:<actual-stage>` with a reason from
    # that stage's vocabulary, distinct from an unresolved identity. The stage travels
    # on the exception because only the raise site knows whether the source had
    # already been resolved and frozen when the budget ran out.
    if isinstance(exc, AuthorEffortCapExceeded):
        return exc.stage, exc.reason_code
    # A session that ran out of budget and tried to file it as a prerequisite block is
    # refused at the parse boundary. It is still an exhausted session, so it lands on the
    # same doctrine terminal rather than falling through to `identity-unresolved`, which
    # would trade one false statement about this model for another. The stage travels on
    # the claim for the same reason it travels on the cap: the arm the author named is the
    # stage that was actually in flight when the budget ran out.
    if isinstance(exc, AuthorEffortExhaustionClaim):
        return exc.stage, exc.reason_code
    if isinstance(exc, CasObjectCorruptError):
        # Our own store failed to match its own content address. No retry can fix
        # a damaged object, so this is terminal by construction, and it is
        # spelled out here so it can never again be indistinguishable from the
        # upstream-drift case below except by parsing a message string.
        return "fetch", "hash-mismatch"
    if isinstance(exc, UpstreamContentDriftError):
        # `fetch_targets` resolves drift into a recorded per-source disposition,
        # so reaching this arm means a caller invoked the singular `fetch_target`
        # directly and declined to resolve it. Kept distinct and explicit rather
        # than folded into the refusal below, which means something else.
        return "fetch", "hash-mismatch"
    if isinstance(exc, FetchHashMismatchError):
        return "fetch", "hash-mismatch"
    if isinstance(exc, FetchRetrievalError):
        return "fetch", "unreachable"
    if isinstance(exc, UnpinnedTargetError):
        # The author named a target the controlled fetch contract cannot accept.
        # That is a declaration defect, not an unresolvable identity.
        return "source", "source-target-invalid"
    if isinstance(exc, AuthorSourceDeterministicallyUnfetchableError):
        # AL-3: two consecutive attempts produced byte-identical deterministic
        # fetch refusals (bad-ref, oversized, redirect-refused). Same closed
        # pair as `UnpinnedTargetError` and for the same reason: the declared
        # targets are ones the controlled fetch contract deterministically
        # cannot accept. Never the `session-crashed` catch-all -- both
        # sessions ran to completion.
        return "source", "source-target-invalid"
    if isinstance(exc, LedgerConflictError):
        # Item 9 / author-uplift R-7: the crawler's OWN corpus-index ledger
        # refused an immutable-identity replay inside the accept-time
        # ``text_corpus.promote_author_root`` (m5520 / m7749 / m279 recorded
        # ``failed:author / session-crashed`` -- a death that did not occur,
        # charged to the model). Machinery, same routing as the engine faults
        # below; the ``internal-error`` owner is ``machinery``, so the arm
        # parks the member (``_park_non_model_fault``) instead of minting.
        return "runner", "internal-error"
    if isinstance(exc, SourceManifestBindingError):
        # The author session ran to a clean typed verdict and the ENGINE's own
        # driver-built manifest then failed its self-consistency checks. Calling
        # that `session-crashed` states a fact that did not occur, and lands an
        # engine fault in a vocabulary already carrying the author-transport
        # catch-all -- two unrelated causes in one field the reducer counts.
        # Deliberately narrow: the sibling `ArtifactBindingError` arms reject
        # author-SUPPLIED content and stay author-owned below.
        return "runner", "internal-error"
    # `AuthorDispatchError` has 33 raise sites with three different owners, and the
    # blanket arm below recorded ALL of them as `session-crashed` -- a session that
    # published a complete result and was then refused on its CONTENT read as a session
    # that died. These arms split by OWNERSHIP of the input each raise site refused.
    # They dispatch on subtypes chosen AT the raise sites, never on message text, so
    # nothing here has to be kept in sync with any wording, and a site whose ownership
    # is genuinely unknowable still raises the bare base and still lands on the
    # catch-all below -- which is the honest answer for it.
    if isinstance(exc, AuthorResultMalformedError):
        # The session ran to completion and published bytes that fail a contract it is
        # bound to. `malformed-result` is the reason the author vocabulary already
        # carries for exactly this class.
        return "author", "malformed-result"
    if isinstance(exc, (AuthorEngineFaultError, AuthorInfrastructureFaultError)):
        # The engine's own envelope/grant/intake/cache, or the host underneath them,
        # refused. Identical routing to the `SourceManifestBindingError` arm above and
        # for the identical reason: an engine fault counted in the author-stage census
        # makes the campaign's own failure counts wrong. The two types stay distinct so
        # the operational event's `error_summary` still says which one it was -- the
        # `runner` reason vocabulary has no retryable member, and inventing one would be
        # a vocabulary change rather than a taxonomy fix.
        return "runner", "internal-error"
    if isinstance(exc, ArtifactBindingError):
        # Staging refused a result the session ran to completion and PUBLISHED --
        # `session-crashed` asserts a death that did not occur. Rung 7, m9617: a
        # replay-valid PROPOSED result whose declared source set contradicted the
        # frozen manifest was recorded as a session crash, permanently on a run-once
        # campaign. The reachable raise sites in `validate_artifact_transaction`'s
        # staging path all refuse the PUBLISHED CONTENT against machine authority,
        # which is exactly the `malformed-result` class; the exact refusal type,
        # message, and raise site still travel on the model-lane failure event, so
        # nothing is lost by the honest label. (A finer split -- author-owned vs
        # engine-owned binding subtypes -- belongs in `artifact_transactions.py`.)
        return "author", "malformed-result"
    if isinstance(exc, AuthorEnvironmentBindingError):
        # The author published a complete recipe, but its distribution or version
        # contradicts the machine-owned routed inventory. This is published-content
        # refusal, not a runner protocol failure: no worker process was launched.
        return "author", "malformed-result"
    return "author", "session-crashed"


def _raise_site(exc: BaseException) -> Optional[str]:
    """Return the deepest crawler-owned ``file:line in function`` frame of ``exc``.

    The terminal record redacts ``status.detail`` and files the Python traceback into a
    gitignored local diagnostics sidecar, so on any host that did not keep that sidecar
    the reason code is the only surviving statement about the cause. A message alone is
    not always enough to find the check: several refusals share an f-string template
    across shared helpers. The frame is machine-derived from the exception's own
    traceback, never from externally-controlled text.

    Parameters
    ----------
    exc:
        Exception whose raise site is wanted.

    Returns
    -------
    str | None
        ``"<module>.py:<lineno> in <function>"`` for the deepest frame inside the
        crawler package, or ``None`` when the exception carries no usable traceback.
    """

    package_root = str(Path(__file__).parent)
    site: Optional[str] = None
    frame = exc.__traceback__
    while frame is not None:
        filename = frame.tb_frame.f_code.co_filename
        if filename.startswith(package_root):
            site = f"{Path(filename).name}:{frame.tb_lineno} in {frame.tb_frame.f_code.co_name}"
        frame = frame.tb_next
    return site


def _model_lane_failure_event(
    *,
    stable_id: str,
    work_id: str,
    status_code: str,
    reason_code: str,
    exc: BaseException,
    run_id: str,
    machine_id: str,
    created_at: str,
) -> JsonObject:
    """Build the operational event that NAMES the cause of one model-local terminal.

    A ``failed:`` terminal deliberately nulls ``status.detail`` -- the detail can carry
    externally-controlled text -- and points ``status.traceback`` at a gitignored local
    diagnostics sidecar. That is safe but it is not legible: the terminal on its own
    says only ``failed:author`` / ``<reason_code>``, and a whole partition of models can
    terminalize for one cause with nothing in any durable operational record naming it.

    This event closes that gap the same way the terminal-unrecordable ladder did: the
    exception TYPE and MESSAGE are summarised once and surfaced at the event's TOP
    LEVEL, not nested where they need excavating, alongside the machine-derived raise
    site. All three are engine-derived: the type is a class name, the message on the
    author-lane types is a fixed literal from the raise site, and the frame comes from
    the traceback object.

    Parameters
    ----------
    stable_id, work_id:
        Terminal model and its active work generation.
    status_code, reason_code:
        Closed disposition the lane recorded for this model.
    exc:
        Exception that produced the terminal.
    run_id, machine_id, created_at:
        Campaign run identity and event timestamp.

    Returns
    -------
    JsonObject
        Schema-valid ``model-lane-failed`` operational event.
    """

    error_summary = f"{type(exc).__name__}: {exc}"
    raise_site = _raise_site(exc)
    event: JsonObject = {
        "error_summary": error_summary,
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        # The registry pins this kind per-occurrence, so created_at remains an
        # ID axis: the same failure observed on a later resume is a new row,
        # not the conflicting replay that crashed the step-100 leg (2026-08-13).
        "created_at": created_at,
        "event_kind": OperationalEventKind.MODEL_LANE_FAILED.value,
        "status": OperationalEventStatus.MODEL_LANE_FAILED.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": run_id,
        "machine_id": machine_id,
        "details": {
            "stable_id": stable_id,
            "work_id": work_id,
            "status_code": status_code,
            "reason_code": reason_code,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "raise_site": raise_site,
        },
    }
    return mint_operational_event(event)


# Reviewed runtime roots. ``_runner_identity`` discovers their transitive local call
# graph and hashes semantic AST nodes, not whole modules or operational schemas.
_RUNNER_COMMON_EXECUTION_CLOSURE = {
    "worker.py": ("main",),
    "worker_supervisor.py": ("run_isolated_subprocess",),
}
_RUNNER_IDENTITY_CACHE: dict[str, str] = {}
# Backward-compatible inspection alias for the reviewed runtime roots.
_RUNNER_EXECUTION_CLOSURE = _RUNNER_COMMON_EXECUTION_CLOSURE
_INJECTED_FORWARD_CLOSURE_IDENTITY = stable_hash("injected-forward-lane-executable-closure")
_AWARD_CLOSURE_SYMBOLS = {
    "driver_admission.py": (
        "AdmissionEnvironmentMixin._run_environment_work",
        "_source_symbol_bytes",
        "_award_closure_from_bytes",
        "_award_closure_identity",
        "_runner_identity",
        "_execution_identity",
        "_current_run_is_fresh",
        "_validate_artifact_identities",
        "_validate_trusted_intake_identity",
        "_environment_binding",
        "_installed_package_manifest_bytes",
        "_observed_interpreter_facts",
    ),
    "driver_receipts.py": (
        "ReceiptDriverMixin._forward_and_reduce",
        "ReceiptDriverMixin._ensure_pending_run_anchors",
        "SupervisedForwardLane.forward",
        "_attempts_from_supervised",
        "_collect_worker_executable_closure",
        "_compile_worker_read_manifest",
        "_expected_adapter_sha256",
        "_expected_code_manifest_sha256",
        "_expected_input_asset_id",
        "_expected_input_asset_sha256",
        "_receipt_envelope_error",
        "_supervised_failure",
        "_worker_request",
    ),
    "driver_models.py": (
        "_assemble_run_model",
        "_attempt_policy_satisfied",
        "_fidelity_required",
        "_find_gate",
        "_gate_item_matches_proposal",
        "_matching_attempts",
    ),
    "env_lifecycle.py": (
        "SequentialEnvironmentLifecycle.run",
        "installed_package_inventory_bytes",
        "materialized_environment_generation",
        "parse_exact_lock",
        "parse_probe_receipt_bytes",
        "parse_resolved_export",
        "validate_probe_receipts",
    ),
    "family_templates.py": (
        "instantiate_size_variant",
        "mechanical_variant_parameter_input_line",
        "validate_size_variant",
        "_template_identity_payload",
        "_validate_inherited_metadata",
        "_validate_variant_line",
        "_validate_representative",
    ),
    "proposal.py": ("validate_author_proposal",),
    "gates.py": (
        "MetadataRouteDecision",
        "FidelityRouteDecision",
        "route_metadata_gate",
        "route_fidelity_gate",
        "_validate_gate",
        "_items",
    ),
    "identity.py": (
        "canonical_json_bytes",
        "hash_bytes",
        "stable_hash",
        "normalize_url",
        "compute_source_identity",
        "compute_evidence_identity",
        "compute_recipe_revision",
        "compute_fidelity_identity",
        "compute_vet_identity",
        "compute_execution_identity",
    ),
    "metadata.py": (
        "_required_external_fields",
        "AcceptedIdentities",
        "canonical_meaningful_modes",
        "authored_fact_leaves",
        "_evidence_references",
        "recompute_accepted_identities",
        "validate_external_metadata",
        "validate_authored_facts_for_write",
        "input_signature_matches_contract",
        "_validate_gate_header",
        "_mapping",
    ),
    "reducer.py": (
        "expected_standard_asset",
        "output_signature_error",
        "_select_current",
        "_records_root",
        "_revision_work_ids",
        "_validate_persisted_requeue_lineage",
        "_model_facts",
        "_checker_prompt_hash",
        "CanonicalReducer.__init__",
        "CanonicalReducer.append_attempt",
        "CanonicalReducer.append_gate",
        "CanonicalReducer.append_model",
        "CanonicalReducer._admit_model",
        "CanonicalReducer._validate_status",
        "CanonicalReducer._validate_source",
        "CanonicalReducer._gate_item",
        "CanonicalReducer._validate_gates",
        "CanonicalReducer._is_fidelity_repair_failure",
        "CanonicalReducer._fidelity_gate_has_no_consumer",
        "CanonicalReducer._derive_execution_failure_waiver",
        "_execution_evidence_beyond_stage",
        "CanonicalReducer._validate_family_template",
        "CanonicalReducer._validate_execution",
        "project_dependency_current",
    ),
    "recordio.py": (
        "_fsync_directory",
        "_logical_payload",
        "_identity_key",
        "_verify_hash",
        "scan_jsonl",
        "recover_torn_tail",
        "JsonlLedger.__init__",
        "JsonlLedger.append",
        "JsonlLedger._existing_by_identity",
        "JsonlLedger._register_record",
    ),
    "schema.py": (
        "load_schema",
        "get_validator",
        "validate_payload",
    ),
    "state.py": ("_select_current",),
}

_AWARD_CLOSURE_SCHEMAS = (
    "schemas/attempt-v3.schema.json",
    "schemas/author-proposal-v3.schema.json",
    "schemas/author-result-v4.schema.json",
    "schemas/gate-v3.schema.json",
    "schemas/model-v3.schema.json",
)


def _quarantine_environment_payload(
    environment: Optional[EnvironmentBinding],
) -> Optional[JsonObject]:
    """Serialize the exact observed environment used by a completed work set.

    Parameters
    ----------
    environment:
        Parent-observed environment generation, if creation reached use.

    Returns
    -------
    dict[str, Any] | None
        Canonical environment facts sufficient for the ordinary freshness predicate.
    """

    if environment is None:
        return None
    return {
        "prefix": str(environment.prefix),
        "python_executable": str(environment.python_executable),
        "family": environment.family,
        "target": environment.target,
        "env_generation": environment.env_generation,
        "lock_sha256": environment.lock_sha256,
        "resolved_export_sha256": environment.resolved_export_sha256,
        "packages_manifest_sha256": environment.packages_manifest_sha256,
        "python_version": environment.python_version,
        "compiler_identity": environment.compiler_identity,
        "sdk_identity": environment.sdk_identity,
        "authority_epoch": environment.authority_epoch,
        "base_environment_generation": environment.base_environment_generation,
        "environment_content_sha256": environment.environment_content_sha256,
        "environment_authority_id": environment.environment_authority_id,
        "selected_interpreter_relative_path": (environment.selected_interpreter_relative_path),
        "selected_interpreter_digest": environment.selected_interpreter_digest,
        "external_escape_records": [
            {"path": str(record.path), "sha256": record.sha256, "kind": record.kind}
            for record in environment.external_escape_records
        ],
    }


def restored_environment_generations(
    base_generations: Mapping[str, str],
    attempts_path: Path,
    operational_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Derive durable per-family environment generations from ledger evidence.

    The dependency-currency projection treats a record whose environment
    family is absent from the context as STALE, so any consumer that projects
    without live environments -- the resuming driver before its first env
    activation, and offline projection-only audits -- must first restore the
    generations the ledgers actually attest: every attempt row's recorded
    environment identity (newest wins per family) plus the quarantine events'
    parent-observed bindings.

    Parameters
    ----------
    base_generations:
        Starting generations (usually the authority context's current map).
    attempts_path:
        Canonical attempts ledger path.
    operational_records:
        Parsed operational ledger records.

    Returns
    -------
    dict[str, str]
        Family/intent name to environment generation.
    """

    generations = dict(base_generations)
    if attempts_path.exists():
        for attempt in scan_jsonl(attempts_path):
            environment = attempt.get("environment")
            generation = attempt.get("identities", {}).get("environment")
            family = environment.get("family") if isinstance(environment, Mapping) else None
            if isinstance(family, str) and family and isinstance(generation, str):
                generations[family] = generation
    for event in operational_records:
        details = event.get("details", {})
        if event.get("event_kind") != OperationalEventKind.CAMPAIGN_HEALTH.value or not isinstance(
            details, Mapping
        ):
            continue
        disposition = details.get("disposition")
        intent = details.get("intent")
        if disposition == "environment-integrity-quarantined":
            generation = details.get("env_generation")
            if isinstance(intent, str) and intent and isinstance(generation, str) and generation:
                generations[intent] = generation
            continue
        if disposition != "environment-cleanup-quarantined":
            continue
        environment_binding = _environment_from_quarantine(details)
        if environment_binding is not None and isinstance(intent, str) and intent:
            generations[intent] = environment_binding.env_generation
    return generations


def attested_generation_families(
    attempts_path: Path,
    operational_records: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Derive EVERY ledger-attested generation -> family binding, not one per family.

    :func:`restored_environment_generations` keeps the newest generation per
    family because currency projection wants the current map. Terminal
    assembly wants the inverse and the full history: a resumed leg whose
    family REBUILT mid-run must still resolve an OLDER generation's family
    from its proof attempts (the m8830/m8838/m8887 recording circuit-breaker,
    2026-08-13). Same evidence sources, zero producer input.

    Parameters
    ----------
    attempts_path:
        Canonical attempts ledger path.
    operational_records:
        Parsed operational ledger records.

    Returns
    -------
    dict[str, str]
        Environment generation identity to family/intent name.
    """

    bindings: dict[str, str] = {}
    if attempts_path.exists():
        for attempt in scan_jsonl(attempts_path):
            environment = attempt.get("environment")
            generation = attempt.get("identities", {}).get("environment")
            family = environment.get("family") if isinstance(environment, Mapping) else None
            if isinstance(family, str) and family and isinstance(generation, str) and generation:
                bindings[generation] = family
    for event in operational_records:
        details = event.get("details", {})
        if not isinstance(details, Mapping):
            continue
        intent = details.get("intent")
        generation = details.get("env_generation")
        if isinstance(intent, str) and intent and isinstance(generation, str) and generation:
            bindings[generation] = intent
        environment_binding = _environment_from_quarantine(details)
        if environment_binding is not None and isinstance(intent, str) and intent:
            bindings[environment_binding.env_generation] = intent
    return bindings


def _environment_from_quarantine(details: Mapping[str, Any]) -> Optional[EnvironmentBinding]:
    """Rehydrate a quarantine's exact parent-observed environment binding.

    Parameters
    ----------
    details:
        Canonical environment-cleanup event details.

    Returns
    -------
    EnvironmentBinding | None
        Exact binding, or ``None`` for a legacy/incomplete quarantine event.
    """

    value = details.get("environment")
    required = (
        "prefix",
        "python_executable",
        "family",
        "target",
        "env_generation",
        "lock_sha256",
        "resolved_export_sha256",
        "packages_manifest_sha256",
        "python_version",
        "compiler_identity",
        "sdk_identity",
    )
    if not isinstance(value, Mapping) or any(
        not isinstance(value.get(key), str) for key in required
    ):
        return None
    return EnvironmentBinding(
        prefix=Path(str(value["prefix"])),
        python_executable=Path(str(value["python_executable"])),
        family=str(value["family"]),
        target=str(value["target"]),
        env_generation=str(value["env_generation"]),
        lock_sha256=str(value["lock_sha256"]),
        resolved_export_sha256=str(value["resolved_export_sha256"]),
        packages_manifest_sha256=str(value["packages_manifest_sha256"]),
        python_version=str(value["python_version"]),
        compiler_identity=str(value["compiler_identity"]),
        sdk_identity=str(value["sdk_identity"]),
        authority_epoch=(
            str(value["authority_epoch"]) if isinstance(value.get("authority_epoch"), str) else None
        ),
        base_environment_generation=(
            str(value["base_environment_generation"])
            if isinstance(value.get("base_environment_generation"), str)
            else None
        ),
        environment_content_sha256=(
            str(value["environment_content_sha256"])
            if isinstance(value.get("environment_content_sha256"), str)
            else None
        ),
        environment_authority_id=(
            str(value["environment_authority_id"])
            if isinstance(value.get("environment_authority_id"), str)
            else None
        ),
        selected_interpreter_relative_path=(
            str(value["selected_interpreter_relative_path"])
            if isinstance(value.get("selected_interpreter_relative_path"), str)
            else None
        ),
        selected_interpreter_digest=(
            str(value["selected_interpreter_digest"])
            if isinstance(value.get("selected_interpreter_digest"), str)
            else None
        ),
        external_escape_records=tuple(
            EnvironmentExternalTarget(
                path=Path(str(record["path"])),
                sha256=str(record["sha256"]),
                kind=str(record["kind"]),
            )
            for record in value.get("external_escape_records", ())
            if isinstance(record, Mapping)
            and all(isinstance(record.get(key), str) for key in ("path", "sha256", "kind"))
        ),
    )


def _quarantine_work_identity(item: WorkItem, artifact: AuthorArtifact) -> JsonObject:
    """Bind every non-environment input used by run freshness and quarantine reuse.

    History-vs-live doctrine (rung 11): quarantine snapshots RECORD this full
    identity unchanged, but the resuming admission decision no longer requires
    every component to match -- the code/work-generation components (see
    :data:`_RECORD_ANCHORED_WORK_IDENTITY_KEYS` for the record-anchored rest)
    may move across a hotfix, with the drift disclosed in a typed operational
    event instead of voiding completed work.

    Parameters
    ----------
    item, artifact:
        Exact scheduled intake/work generation and normalized proposal.

    Returns
    -------
    dict[str, Any]
        Closed work identity used only with an exact observed environment binding.
    """

    proposal = artifact.proposal
    facts = proposal.get("proposed_facts", {})
    external = facts.get("external_metadata") if isinstance(facts, Mapping) else None
    modality = external.get("modality") if isinstance(external, Mapping) else None
    return {
        "stable_id": item.stable_id,
        "intake_item_sha256": stable_hash(item.intake.to_dict()),
        "work_id": proposal.get("work_id"),
        "proposal_sha256": proposal.get("proposal_sha256"),
        "source_identity": proposal.get("source_identity"),
        "evidence_identity": proposal.get("evidence_identity"),
        "recipe_revision": proposal.get("recipe_revision"),
        "vet_identity": proposal.get("vet_identity"),
        "fidelity_identity": proposal.get("fidelity_identity"),
        "author_prompt": proposal.get("author", {}).get("prompt_sha256"),
        "checker_prompt": leg_checker_prompt_sha256(),
        "runner_identity": _runner_identity(modality),
        "award_closure": _current_award_closure_identity(),
        "template_source_revision": artifact.template_source_revision,
        "campaign_root_work_id": _artifact_lineage(artifact),
    }


#: Campaign-health disposition for one intent's quarantine parking decision.
ENVIRONMENT_REBUILD_DEFERRED_DISPOSITION = "environment-rebuild-deferred"

#: Campaign-health disposition for a pre-use environment failure parked through
#: the typed routing registry (DD-1): the intent-level event carries the typed
#: failure and its routing decision, and each affected work identity's park row
#: lands in the environment-rebuild machine-deferral ledger.
ENVIRONMENT_SETUP_FAILURE_PARKED_DISPOSITION = "environment-setup-failure-parked"

#: Campaign-health disposition for one work identity parked behind this host's
#: unmet environment disk floor (a machine condition, never a model terminal).
HOST_DISK_FLOOR_DEFERRED_DISPOSITION = "host-disk-floor-deferred"

#: Campaign-health disposition for completed work accepted across a code hotfix:
#: every RECORD identity (proposal, sources, receipts, environment generation)
#: matches the quarantine snapshot, and only CODE/work-generation identities moved.
QUARANTINE_RECORD_ANCHORED_ACCEPTANCE_DISPOSITION = (
    "environment-quarantine-record-anchored-acceptance"
)

#: Campaign-health disposition for a byte-verified finalized artifact whose
#: recorded author result no longer validates against the LIVE author-result
#: schema. The solve cache is reuse-convenience, not history: the entry is
#: invalidated loudly and the model routes to fresh authorship (rung-12 field
#: failure: one such entry crashed the whole 100-model campaign at rehydration).
ARTIFACT_AUTHORITY_SCHEMA_STALE_DISPOSITION = "artifact-authority-schema-stale"

#: Campaign-health disposition for a rehydrated finalized artifact that fails
#: identity revalidation at its own RECORDED basis (or whose recorded basis is
#: unrecoverable). Sibling of the schema-stale disposition: the entry is
#: invalidated loudly and the model routes to fresh authorship, never a driver
#: crash (rung-12 leg 2: the live-basis identity recompute aborted the campaign
#: at the first pre-rotation artifact).
ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION = "artifact-authority-identity-stale"

#: Campaign-health disposition for a finalized transaction the checkpoint
#: quarantined because it contradicts or fails ITS OWN record: a dependency
#: vector contradicting the final event (rung-12 leg 4: checker_gate_identity),
#: rewritten or unreadable reconstruction bytes, or an interrupted event chain.
#: Same consumption as its siblings: loud invalidation, fresh authorship,
#: never a driver crash.
ARTIFACT_AUTHORITY_RECORD_STALE_DISPOSITION = "artifact-authority-record-stale"

#: Campaign-health disposition for an operator's explicit quarantine clearance
#: receipt. An out-of-band backend/tool repair can leave the setup artifacts
#: byte-identical, so identity comparison alone can never observe it (rung-11
#: review finding 6): the operator instead appends one campaign-health event
#: with this disposition whose ``details.quarantine_event_id`` names the exact
#: cleared quarantine event. A clearance releases BOTH the live admission valve
#: (the quarantine no longer governs its intent) and every persisted rebuild
#: deferral row parked on that quarantine. A LATER quarantine event is a new
#: fact and governs again until it is itself cleared.
ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION = "environment-quarantine-cleared"


def _quarantine_is_cleared(
    events: Sequence[Mapping[str, Any]],
    quarantine_event_id: str,
) -> bool:
    """Return whether an explicit clearance receipt names this quarantine event.

    Parameters
    ----------
    events:
        Canonical operational events in ledger order.
    quarantine_event_id:
        Exact ``environment-cleanup-quarantined`` event to test.

    Returns
    -------
    bool
        Whether a clearance event names the quarantine. Ledger order needs no
        inspection: event IDs are content-addressed, so a clearance can only
        name a quarantine that already exists, and a LATER quarantine carries
        a new event ID that no earlier clearance can name.
    """

    return any(
        event.get("event_kind") == OperationalEventKind.CAMPAIGN_HEALTH.value
        and event.get("details", {}).get("disposition")
        == ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION
        and event.get("details", {}).get("quarantine_event_id") == quarantine_event_id
        for event in events
    )


#: BL-9c: clearance receipts get their own dedicated append-only sidecar so an
#: operator can record one WHILE A LEG RUNS. The canonical operational ledger
#: is single-writer property of the running driver (the clearance tool used to
#: take the driver lock and simply refused mid-leg -- so recovering a parked
#: intent always cost a full leg restart). The sidecar has exactly one writer
#: (the clearance tool) and the driver only ever READS it, so no writer
#: discipline is violated. Receipts already persisted in the canonical
#: operational ledger keep releasing exactly as before; readers merge both.
ENVIRONMENT_CLEARANCE_RELATIVE_PATH = "operational/environment-clearances.jsonl"


def environment_clearance_path(records_root: Path) -> Path:
    """Return the operator-writable clearance-receipt sidecar ledger path."""

    return Path(records_root) / ENVIRONMENT_CLEARANCE_RELATIVE_PATH


def environment_clearance_events(records_root: Path) -> tuple[JsonObject, ...]:
    """Read sidecar clearance receipts, erring toward the park HOLDING.

    A clearance receipt is release-only evidence, so the failure direction is
    closed: an absent sidecar, an undecodable line, or a torn tail simply
    contributes no receipt (the park keeps holding until a readable receipt
    exists). Nothing here can ever CREATE a release that was not durably and
    readably recorded.
    """

    path = environment_clearance_path(records_root)
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return ()
    events: list[JsonObject] = []
    for line in content.split(b"\n"):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            events.append(value)
    return tuple(events)


def append_environment_clearance_event(records_root: Path, event: Mapping[str, Any]) -> None:
    """Durably append one clearance receipt to the operator sidecar ledger.

    One line per receipt through a single ``write`` on an ``O_APPEND``
    descriptor, fsynced before return: concurrent tool invocations cannot
    interleave within a line, and a reader racing the append sees either no
    receipt yet or the complete line -- both safe under the release-only
    reader above.
    """

    path = environment_clearance_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, payload)
        durable_fsync(descriptor)
    finally:
        os.close(descriptor)


#: The RECORD-anchored subset of :func:`_quarantine_work_identity`. These keys
#: derive from durable record content alone -- the intake row, the frozen
#: proposal, and the family-representative record revision -- so they survive a
#: code hotfix between the quarantine snapshot and a resume. The remaining keys
#: (``checker_prompt``, ``runner_identity``, ``award_closure``, ``work_id``,
#: ``campaign_root_work_id``) bind live code or work-generation lineage and are
#: exactly what mass-invalidated all eleven rung-11 run awards.
_RECORD_ANCHORED_WORK_IDENTITY_KEYS: tuple[str, ...] = (
    "stable_id",
    "intake_item_sha256",
    "proposal_sha256",
    "source_identity",
    "evidence_identity",
    "recipe_revision",
    "vet_identity",
    "fidelity_identity",
    "author_prompt",
    "template_source_revision",
)


def _quarantine_code_identity_drift(
    recorded_identity: Mapping[str, Any],
    live_identity: Mapping[str, Any],
) -> Optional[tuple[str, ...]]:
    """Name the moved non-record identity components, or refuse on record drift.

    History-vs-live doctrine: the quarantine snapshot keeps its recorded basis
    untouched; only the ADMISSION decision consults this comparison.

    AUTHORITY BOUNDARY (rung-11 review finding 7): this classifier decides
    nothing about POLICY safety. The single policy-closure authority is the
    dependency-current replay (``authority.validate_currency`` under an
    ``AuthorityContext`` whose identities are recomputed from live bytes at
    run start): it independently re-adjudicates every policy axis the record
    binds -- checker prompt, runner policy, award/reducer policy, author
    prompt, schemas, terminal rules, publication policy, intake, and
    environment generation -- and a record failing ANY of them never reaches
    this comparison (see :func:`_record_anchored_run_is_current`, which
    requires that proof as an explicit argument). Given a record that PASSED
    that replay, the non-record components named here are exactly the
    identities the replay does not treat as award-invalidating (execution-time
    code closures and work-generation lineage), and this function's job is to
    NAME them for the typed disclosure event -- while still refusing, as a
    redundant tripwire, when any record-anchored component moved.

    Parameters
    ----------
    recorded_identity:
        Work identity the quarantine event snapshot recorded for completed work.
    live_identity:
        The same identity recomputed by the resuming leg.

    Returns
    -------
    tuple[str, ...] | None
        ``None`` when ANY record-anchored component differs (the completed work
        is genuinely stale), otherwise the sorted names of every moved
        code/work-generation component (empty when the identities are equal).
    """

    for key in _RECORD_ANCHORED_WORK_IDENTITY_KEYS:
        if recorded_identity.get(key) != live_identity.get(key):
            return None
    return tuple(
        sorted(
            key
            for key in set(recorded_identity) | set(live_identity)
            if key not in _RECORD_ANCHORED_WORK_IDENTITY_KEYS
            and recorded_identity.get(key) != live_identity.get(key)
        )
    )


def _record_anchored_run_is_current(
    current: Mapping[str, Any],
    exact: Mapping[str, Any],
    quarantined_environment: EnvironmentBinding,
    *,
    dependency_current: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Return whether a run award holds on record identities plus replay proof.

    The record checks are RECORD-anchored: the canonical revision the
    quarantine snapshot witnessed (which pins every receipt hash in the
    record), the runs status, the executed environment generation, and the
    accepted-attempt receipts. The REQUIRED ``dependency_current`` argument is
    the replayed dependency-vector proof: the record must appear, at this
    exact revision, in the dependency-current projection computed under the
    live ``AuthorityContext`` (with the environment axis pinned to the
    quarantined generation). That projection independently re-adjudicates
    every policy axis the record binds, so acceptance can never bless a record
    whose checker, runner, award, author, schema, terminal, or publication
    policy contradicts the live tree -- the proof is a structural input here,
    not an ambient assumption of the call site (rung-11 review finding 7).

    Parameters
    ----------
    current:
        Current canonical record for the model.
    exact:
        The quarantine snapshot's completed-work entry for the model.
    quarantined_environment:
        Environment binding the quarantine event recorded.
    dependency_current:
        Freshly projected dependency-current records under the live authority
        context, keyed by stable ID.

    Returns
    -------
    bool
        Whether the completed run still binds its recorded basis exactly.
    """

    if current.get("status", {}).get("kind") != "runs":
        return False
    if exact.get("record_revision") != current.get("record_revision"):
        return False
    replayed = dependency_current.get(str(current.get("stable_id")))
    if replayed is None or replayed.get("record_revision") != current.get("record_revision"):
        return False
    execution = current.get("execution", {})
    accepted = execution.get("accepted_attempt_ids")
    return bool(
        execution.get("current")
        and execution.get("env_generation") == quarantined_environment.env_generation
        and isinstance(accepted, list)
        and accepted
    )


def _intent_setup_artifact_state(intent: EnvironmentIntent) -> JsonObject:
    """Return one intent's typed per-member setup-artifact state vector.

    Parameters
    ----------
    intent:
        Loaded environment intent whose lock family names the setup artifacts.

    Returns
    -------
    dict[str, Any]
        Each of the four setup-artifact members (lock, resolved export, export
        hash, probe receipt) mapped to ``{"state": "present", "sha256": ...}``
        or ``{"state": "absent"}``. Distinct partial families always produce
        distinct vectors; absence is a typed member state, never a collapsed
        sentinel (rung-11 review finding 6).
    """

    receipt_path = intent.lock.lock_path.with_name(f"{intent.lock.target}.probes.json")
    paths = [
        intent.lock.lock_path,
        intent.lock.export_path,
        intent.lock.export_hash_path,
        receipt_path,
    ]
    # A declared pip overlay adds its pinned manifest as a FIFTH committed
    # member, so an overlay rotation changes the identity exactly like a conda
    # rotation does. An intent WITHOUT a declared overlay keeps the historical
    # four-member vector bit-for-bit, so every existing quarantine/park row's
    # identity comparison is untouched.
    overlay = getattr(intent, "pip_overlay", None)
    if overlay is not None:
        paths.append(intent.lock.lock_path.with_name(f"{intent.lock.target}.pip-overlay.txt"))
    state: JsonObject = {}
    for path in paths:
        if path.is_file():
            state[path.name] = {"state": "present", "sha256": hash_bytes(path.read_bytes())}
        else:
            state[path.name] = {"state": "absent"}
    return state


def _intent_setup_artifact_identity(intent: EnvironmentIntent) -> str:
    """Return one intent's typed committed setup-artifact identity.

    Never ``None``: a COMPLETE artifact family hashes its member digests
    exactly as recorded by pre-v2 quarantine events (so pilot-era events with
    all four setup files keep matching), while an INCOMPLETE family hashes its
    full per-member state vector under a distinct domain key -- so two
    different partial states, or a mutation to a surviving member, always
    produce different identities instead of collapsing to a shared ``None``
    (rung-11 review finding 6).

    Parameters
    ----------
    intent:
        Loaded environment intent whose lock family names the setup artifacts.

    Returns
    -------
    str
        Typed setup-artifact identity.
    """

    state = _intent_setup_artifact_state(intent)
    if all(member.get("state") == "present" for member in state.values()):
        return stable_hash({name: member["sha256"] for name, member in state.items()})
    return stable_hash({"partial_setup_artifact_state": state})


class _AuthorLaneBase:
    """Shared author-lane workflow around one injected operator transport.

    Both production lanes build the identical two envelopes, run the identical
    controlled fetch, and run the identical validators. They differ only in how
    one request round trip reaches an operator: a subprocess wrapper
    (:class:`CommandAuthorLane`) or a file-queue RPC serviced by the managing
    Claude session (:class:`QueueAuthorLane`).
    """

    effort_grant: AuthorEffortGrant
    registry: Optional[EnvironmentRegistry] = None

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        """Build and execute one frozen author envelope."""

        root = work_root / item.stable_id / "author"
        model_dir = root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        result_path = root / "result.json"
        source_result = self._fetch_author_sources(item, root, config)
        if isinstance(
            source_result, (NegativeDiscovery, HigherTierDiscovery, AccessBlockedDiscovery)
        ):
            return materialize_discovery_artifact(
                source_result,
                item=item,
                context=context,
                root=root,
            )
        source_manifest = source_result
        # Past this line the model's sources are resolved, fetched, and hash-frozen, so a
        # later cap exhaustion can never honestly be a source failure. This is the one
        # boundary that knows it, so it is where the stage is re-attributed for both lanes.
        try:
            return self._author_from_frozen_sources(
                item, config, context, root, model_dir, result_path, source_manifest
            )
        except AuthorEffortCapExceeded as exc:
            raise AuthorEffortCapExceeded(
                *exc.args,
                stage="author",
                dimension=exc.dimension,
            ) from exc

    def _author_from_frozen_sources(
        self,
        item: WorkItem,
        config: DriverConfig,
        context: AuthorityContext,
        root: Path,
        model_dir: Path,
        result_path: Path,
        source_manifest: JsonObject,
    ) -> AuthorArtifact:
        """Run and validate one author session against an already-frozen manifest.

        Parameters
        ----------
        item, config, context:
            Work item, frozen campaign configuration, and authority context.
        root, model_dir, result_path:
            Model-local author root, staged-code directory, and required result path.
        source_manifest:
            Exact controlled-fetch manifest this session must quote from.

        Returns
        -------
        AuthorArtifact
            Validated author result bound to the frozen manifest.
        """

        from menagerie.crawler.author_attempts import prior_attempts_summary
        from menagerie.crawler.author_dispatch import write_envelope_atomic

        work_id = item.active_work_id
        envelope = build_author_envelope(
            context=context,
            work_id=work_id,
            stable_id=item.stable_id,
            campaign_id=_campaign_id_for_item(item),
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            untrusted_hints=item.intake.to_dict(),
            source_manifest=source_manifest,
            allowed_model_dir=model_dir,
            output_path=result_path,
            # The ONE retry feedback channel: requeues and checker repairs
            # alike surface every prior attempt's outcome, failure reason, and
            # verbatim checker findings to the next author session.
            prior_attempts=prior_attempts_summary(root),
            routed_environment_intent=item.route.intent,
            routed_environment_packages=_routed_environment_packages(
                getattr(self, "registry", None), item.route.intent
            ),
        )
        envelope_path = write_envelope_atomic(envelope, root / "request.json")
        launched_author_model = self._dispatch(
            kind="author",
            item=item,
            config=config,
            work_id=work_id,
            request_path=envelope_path,
            output_path=result_path,
        )
        # The session may have spent its ONE supplementary source round. Those
        # sources are ours -- our broker fetched and hashed them -- so they join
        # private custody BEFORE the result is validated against them. Doing it
        # after would refuse every excerpt citing a source we ourselves supplied.
        grounding_manifest = self._extend_manifest_with_supplement(item, root, source_manifest)
        result = validate_author_result(
            result_path,
            envelope,
            cas_root=root / "source-cas",
            supplementary_sources=grounding_manifest.get("supplementary_sources", ()),
        )
        return AuthorArtifact(
            result,
            grounding_manifest,
            model_dir,
            launched_author_model=(str(launched_author_model) if launched_author_model else None),
        )

    def _fetch_author_sources(
        self,
        item: WorkItem,
        root: Path,
        config: Optional[DriverConfig] = None,
    ) -> JsonObject | NegativeDiscovery | HigherTierDiscovery | AccessBlockedDiscovery:
        """Validate discovery, broker FOUND locators, and freeze machine source facts.

        Parameters
        ----------
        item:
            Exact scheduled work item.
        root:
            Per-model author custody root.
        config:
            Optional active driver configuration.

        Returns
        -------
        dict[str, Any] | NegativeDiscovery | HigherTierDiscovery | AccessBlockedDiscovery
            Frozen source manifest, or a checked-materialization discovery arm.
        """

        request_path = root / "source-request.json"
        output_path = root / "source-targets.json"
        work_id = item.active_work_id
        # Item 13 (author R-3, section 4): stage 1 carries the SAME routed
        # environment stage 2 receives, so the author pins library files at the
        # installed version rather than a guessed tag (removing the forced
        # version re-fetch that drove 432 of 573 supplement requests), plus a
        # coarse host-capacity hint so it can size the model to the host up
        # front (6 of 52 R6b members died `needs-more-memory` on tiny configs).
        intent = getattr(getattr(item, "route", None), "intent", None)
        body: JsonObject = {
            "envelope_version": "menagerie.crawler.author-source-request.v1",
            "work_id": work_id,
            "stable_id": item.stable_id,
            "untrusted_hints": item.intake.to_dict(),
            "required_output_path": str(output_path.resolve()),
            "max_sources": self.effort_grant.fetch_targets,
            "discovery_schema_version": "menagerie.crawler.source-discovery.v1",
            "routed_environment": routed_environment_block(
                intent,
                _routed_environment_packages(getattr(self, "registry", None), intent),
            ),
            "host_capacity": host_capacity_block(),
        }
        request = {**body, "envelope_sha256": stable_hash(body)}
        from menagerie.crawler.author_dispatch import write_envelope_atomic

        written = write_envelope_atomic(request, request_path)
        self._dispatch(
            kind="source-request",
            item=item,
            config=config,
            work_id=work_id,
            request_path=written,
            output_path=output_path,
        )
        value = _read_json(output_path)
        discovery: FoundDiscovery | NegativeDiscovery | HigherTierDiscovery | AccessBlockedDiscovery
        if value.get("schema_version") == "menagerie.crawler.source-discovery.v1":
            try:
                parsed = validate_source_discovery(
                    value,
                    stable_id=item.stable_id,
                    work_id=work_id,
                )
            except DiscoveryError as exc:
                raise DriverIntegrationError(
                    f"author source discovery violates its registered contract: {exc}"
                ) from exc
            if isinstance(parsed, RetryableToolFailureDiscovery):
                raise RetryableOperatorError(
                    f"author research tool failed ({parsed.tool_spelling}): {parsed.error}"
                )
            if isinstance(parsed, (NegativeDiscovery, HigherTierDiscovery, AccessBlockedDiscovery)):
                return parsed
            discovery = parsed
            if len(discovery.descriptors) > self.effort_grant.fetch_targets:
                raise AuthorEffortCapExceeded(
                    f"author source discovery for {item.stable_id} named "
                    f"{len(discovery.descriptors)} targets, exceeding the "
                    f"{self.effort_grant.fetch_targets} controlled-fetch grant"
                )
            try:
                pack = broker_source_pack(
                    [descriptor.to_mapping() for descriptor in discovery.descriptors],
                    broker_dir=root / "broker",
                    transport=default_transport(),
                )
            except SourceBrokerError as exc:
                raise DriverIntegrationError(
                    f"author source descriptor was rejected by the broker: {exc}"
                ) from exc
            write_broker_outputs(pack, root / "broker")
            if not pack.implementation_rows():
                raise DriverIntegrationError(
                    "source broker found no machine-classified implementation object"
                )
            value = {
                **pack.to_dict(),
                "discovery": discovery.raw_result,
                "discovery_sha256": stable_hash(discovery.raw_result),
            }
        elif value.get("pack_version") == BROKER_PACK_VERSION:
            raw_discovery = value.get("discovery")
            if not isinstance(raw_discovery, Mapping):
                raise DriverIntegrationError(
                    "source broker pack must retain its registered discovery envelope"
                )
            if value.get("discovery_sha256") != stable_hash(raw_discovery):
                raise DriverIntegrationError(
                    "source broker pack discovery digest does not match its envelope"
                )
            try:
                parsed = validate_source_discovery(
                    raw_discovery,
                    stable_id=item.stable_id,
                    work_id=work_id,
                )
            except DiscoveryError as exc:
                raise DriverIntegrationError(
                    f"source broker pack discovery binding is invalid: {exc}"
                ) from exc
            if not isinstance(parsed, FoundDiscovery):
                raise DriverIntegrationError("source broker pack must originate from FOUND")
            discovery = parsed
        else:
            raise DriverIntegrationError(
                "author source response is neither the registered discovery envelope "
                "nor a machine broker pack"
            )
        raw_targets = value.get("sources")
        if not isinstance(raw_targets, list) or not raw_targets:
            raise DriverIntegrationError(
                "source broker pack must carry at least one fetched source"
            )
        # LP-13.2 controlled-fetch ceiling. The lane is the only engine boundary
        # that observes the pinned target count, so it enforces it here rather
        # than trusting the operator's own accounting.
        if len(raw_targets) > self.effort_grant.fetch_targets:
            raise AuthorEffortCapExceeded(
                f"author source request for {item.stable_id} named {len(raw_targets)} targets, "
                f"exceeding the {self.effort_grant.fetch_targets} controlled-fetch grant"
            )
        if not any(
            isinstance(raw, Mapping) and raw.get("broker_role") == "implementation"
            for raw in raw_targets
        ):
            raise DriverIntegrationError(
                "source broker pack has no machine-classified implementation row"
            )
        merged_rows, promotions = self._freeze_broker_pack_rows(raw_targets, root)
        # Provenance is recorded BESIDE the manifest, never inside it. The
        # `sources` list and its digest carry ONE identity contract every
        # producer and consumer shares: `manifest_sha256 == stable_hash(sources)`.
        # The author echoes that identity back and `_validate_context_result`
        # re-derives it from the rows before binding private custody, so a
        # manifest whose digest disagrees with its own rows cannot be staged at
        # all -- a perfectly good typed author result dies at the binder and the
        # model is terminalized on a cause that never occurred. The ONE key that
        # may join them later is `supplementary_sources` (see
        # `_extend_manifest_with_supplement`), which sits deliberately OUTSIDE
        # `sources` precisely so it cannot move the identity already echoed.
        #
        # Per-source custody disposition: which sources were served from bytes we
        # already held, which fell back to the network, and which re-pinned
        # because the document moved under us. A drift is a real
        # evidence-integrity event, so it is recorded rather than passed over.
        write_envelope_atomic(
            {
                "provenance_version": "menagerie.crawler.source-custody-provenance.v1",
                "stable_id": item.stable_id,
                "work_id": work_id,
                "promotions": promotions,
                "upstream_drift": [
                    dict(row["upstream_drift"])
                    for row in merged_rows
                    if isinstance(row.get("upstream_drift"), Mapping)
                ],
            },
            root / "source-custody.json",
        )
        # Freeze the exact discovery bytes into the model's CAS as well.
        discovery_evidence = freeze_discovery_evidence(discovery, root)
        write_envelope_atomic(
            {
                "provenance_version": "menagerie.crawler.found-discovery-provenance.v1",
                "stable_id": item.stable_id,
                "work_id": work_id,
                "discovery_evidence": discovery_evidence,
                "discovery_evidence_sha256": stable_hash(discovery_evidence),
            },
            root / "discovery-provenance.json",
        )
        # Freeze the routed environment's inventory as a citable machine source.
        # The envelope has always DISCLOSED this inventory to the author, but a
        # disclosure is not evidence: an R1-availability judgment resting on it
        # ("no legal distribution ships this architecture, so R2 is highest
        # applicable") could only be quoted as proposal prose, which the checker
        # rightly refuses to re-attest (m1359 burned its whole bounded fidelity
        # cap on exactly that unsatisfiable repair). Frozen inside `sources`, the
        # same fact becomes ordinary hash-covered evidence: the author binds
        # literal excerpts to it and the checker re-reads the exact bytes. The
        # row is deterministic, so an unchanged environment reproduces an
        # unchanged manifest identity across repair generations; when the machine
        # holds no inventory the row is absent, exactly like the envelope's
        # `inventory_disclosed: false` state.
        inventory_row = routed_inventory_source_row(
            intent=item.route.intent,
            packages=_routed_environment_packages(
                getattr(self, "registry", None), item.route.intent
            ),
            cas_root=root / "source-cas",
            resolved_export_sha256=self._routed_resolved_export_sha256(item.route.intent),
            # Identity stability across repair generations and restarts: the
            # minted row (with its real retrieved_at) is frozen beside the
            # stage-1 pack and reused while the inventory bytes are unchanged,
            # exactly like the broker pack itself.
            row_cache_path=root / "routed-inventory-row.json",
        )
        if inventory_row is not None:
            if any(
                isinstance(row, Mapping) and row.get("source_id") == ROUTED_INVENTORY_SOURCE_ID
                for row in merged_rows
            ):
                raise DriverIntegrationError(
                    "source manifest already carries the reserved machine source "
                    f"{ROUTED_INVENTORY_SOURCE_ID!r}"
                )
            merged_rows = [*merged_rows, inventory_row]
        return {
            "sources": merged_rows,
            "manifest_sha256": stable_hash(merged_rows),
        }

    def _routed_resolved_export_sha256(self, intent_name: Optional[str]) -> Optional[str]:
        """Return the routed intent's locked resolved-export digest, when held.

        Parameters
        ----------
        intent_name:
            Routed environment intent for this work item.

        Returns
        -------
        str | None
            The lock's declared export digest for a ``locked`` target, or
            ``None`` whenever :func:`_routed_environment_packages` would also
            return an empty inventory. Both readers share the same gate so the
            frozen inventory document can never bind a digest its own package
            rows did not come from.
        """

        registry = getattr(self, "registry", None)
        if registry is None or not isinstance(intent_name, str):
            return None
        intent = registry.intents.get(intent_name)
        if intent is None or intent.lock.status != "locked":
            return None
        return intent.lock.declared_export_hash

    def _freeze_broker_pack_rows(
        self,
        raw_targets: Sequence[Any],
        root: Path,
    ) -> tuple[list[JsonObject], list[JsonObject]]:
        """Promote one broker pack's own bytes and controlled-fetch it into custody.

        Shared by the stage-1 pack and by the ONE supplementary pack the executor
        may grant mid-session, so both reach the model's CAS through the identical
        promotion, digest, and merge path. A second implementation is exactly how
        the supplementary round ended up with fetched, hash-pinned bytes that no
        consumer could read.

        Parameters
        ----------
        raw_targets:
            Broker manifest rows carrying the machine-derived identity fields.
        root:
            Per-model author custody root owning ``source-cas``.

        Returns
        -------
        tuple[list[dict[str, Any]], list[dict[str, Any]]]
            Frozen merged manifest rows, and the per-row promotion dispositions.

        Raises
        ------
        DriverIntegrationError
            If a row lacks machine-derived identity, duplicates a ``source_id``,
            or the controlled fetch returns a source the pack never named.
        """

        # Promote the bytes the broker ALREADY fetched into the controlled-fetch
        # CAS before the fetch runs. Without this the driver goes back to the
        # network to verify bytes it already possesses: `already-present` can
        # never fire, every source costs a second GET, and a page carrying an
        # embedded nonce or timestamp differs between the two reads and trips a
        # digest check that means nothing. Promotion is full-digest-verified, so
        # it publishes exactly the bytes the manifest already pins.
        promotions = promote_broker_evidence(
            [row for row in raw_targets if isinstance(row, Mapping)],
            broker_evidence_dirs(root),
            root / "source-cas",
        )
        broker_lengths = {
            str(entry["source_id"]): int(entry["bytes_len"])
            for entry in promotions
            if entry.get("disposition") == "promoted"
        }
        targets: list[FetchTarget] = []
        for raw in raw_targets:
            if not isinstance(raw, Mapping):
                raise DriverIntegrationError("source broker manifest rows must be objects")
            required = {
                "source_id",
                "url",
                "final_url",
                "revision",
                "expected_sha256",
                "media_type",
                "media_type_method",
                "broker_role",
            }
            if not required.issubset(raw):
                raise DriverIntegrationError(
                    "source broker manifest row lacks machine-derived identity fields"
                )
            if (
                any(
                    not isinstance(raw.get(name), str) or not str(raw[name]).strip()
                    for name in required - {"broker_role"}
                )
                or raw.get("broker_role") not in BROKER_MANIFEST_ROLES
            ):
                raise DriverIntegrationError(
                    "source broker manifest row has empty or invalid machine-derived facts"
                )
            targets.append(
                FetchTarget(
                    source_id=str(raw.get("source_id", "")),
                    url=str(raw.get("url", "")),
                    revision=str(raw.get("revision", "")),
                    expected_sha256=str(raw.get("expected_sha256") or ""),
                    media_type=str(raw.get("media_type", "application/octet-stream")),
                    # Every digest on this path is broker-derived: the broker
                    # hashes its OWN fetch, and `expected_sha256` is a
                    # machine-owned field a descriptor is refused for carrying.
                    # So a later disagreement is our-fetch-versus-our-fetch, not
                    # a substituted document.
                    digest_origin=DigestOrigin.CONTROLLED_FETCH,
                    expected_bytes_len=broker_lengths.get(str(raw.get("source_id", ""))),
                )
            )
        fetched = _current_fetch_targets(targets, root / "source-cas")
        fetched_rows = fetched.get("sources")
        if not isinstance(fetched_rows, list) or not fetched_rows:
            raise DriverIntegrationError("controlled fetch produced an empty source manifest")
        broker_rows = {
            str(row["source_id"]): row for row in raw_targets if isinstance(row, Mapping)
        }
        if len(broker_rows) != len(raw_targets):
            raise DriverIntegrationError("source broker manifest source_id values must be unique")
        merged_rows: list[JsonObject] = []
        for row in fetched_rows:
            if not isinstance(row, Mapping):
                raise DriverIntegrationError("controlled fetch manifest rows must be objects")
            source_id = str(row.get("source_id", ""))
            broker_row = broker_rows.get(source_id)
            if broker_row is None:
                raise DriverIntegrationError(
                    "controlled fetch returned a source absent from the broker pack"
                )
            merged_rows.append(
                {
                    **dict(row),
                    **{
                        key: deepcopy(value)
                        for key, value in broker_row.items()
                        if key
                        not in {
                            "source_id",
                            "url",
                            "revision",
                            "expected_sha256",
                            "media_type",
                        }
                    },
                }
            )
        return merged_rows, promotions

    def _extend_manifest_with_supplement(
        self,
        item: WorkItem,
        root: Path,
        source_manifest: JsonObject,
    ) -> JsonObject:
        """Fold the ONE granted supplementary broker pack into the model's custody.

        The executor may grant exactly one supplementary source round mid-session:
        the author names descriptors, OUR broker fetches and hashes them, and the
        session is resumed with the resulting manifest. Those sources are real,
        machine-fetched, digest-pinned evidence the author was legitimately handed
        -- but until this method existed nothing carried them back to the lane, so
        the frozen manifest never learned about them and every excerpt citing one
        was refused as ``references unknown source``. An honest author looked like
        a fabricator, and the pack that proved otherwise left no trace on the
        model's custody. That is a strictly worse failure than the fabrication the
        check exists to catch, so the pack is ingested here through the identical
        promotion and controlled-fetch path stage 1 uses.

        The extension is deliberately a SEPARATE key rather than more ``sources``
        rows. ``source_manifest_identity`` is the identity of the manifest the
        author was handed at dispatch and echoed back in its result; growing
        ``sources`` after the fact would move it and terminalize a good result on
        an identity the author never saw. So ``sources`` and ``manifest_sha256``
        stay exactly as frozen, and the supplementary rows ride beside them under
        their own digest.

        Parameters
        ----------
        item:
            Scheduled work item owning this author root.
        root:
            Per-model author custody root.
        source_manifest:
            Frozen stage-1 manifest the author session was dispatched against.

        Returns
        -------
        dict[str, Any]
            The frozen manifest, extended with ``supplementary_sources`` and
            ``supplementary_manifest_sha256`` when a pack was granted and
            unchanged otherwise.

        Raises
        ------
        DriverIntegrationError
            If the recorded pack is untyped, is not entirely fetched, exceeds the
            controlled-fetch grant, or would redefine a frozen ``source_id``.
        """

        from menagerie.crawler.author_attempts import latest_attempt
        from menagerie.crawler.author_dispatch import write_envelope_atomic

        attempt = latest_attempt(root)
        if attempt is None:
            return source_manifest
        supplement = attempt.record.get("supplement")
        if not isinstance(supplement, Mapping):
            return source_manifest
        manifest_path = supplement.get("manifest_path")
        if not isinstance(manifest_path, str) or not manifest_path.strip():
            return source_manifest
        path = Path(manifest_path)
        if not path.is_file():
            raise DriverIntegrationError(
                "the executor recorded a supplementary source pack whose manifest is absent"
            )
        pack = _read_json(path)
        if pack.get("pack_version") != BROKER_PACK_VERSION:
            raise DriverIntegrationError(
                "supplementary source pack is not the registered broker pack envelope"
            )
        raw_rows = pack.get("sources")
        if not isinstance(raw_rows, list) or not raw_rows:
            raise DriverIntegrationError("supplementary source pack carries no fetched source")
        if len(raw_rows) > self.effort_grant.fetch_targets:
            raise AuthorEffortCapExceeded(
                f"supplementary source pack for {item.stable_id} named {len(raw_rows)} targets, "
                f"exceeding the {self.effort_grant.fetch_targets} controlled-fetch grant"
            )
        # A supplementary row may only ADD a source. Letting one reuse a frozen
        # `source_id` would let an author repoint an already-quoted identifier at
        # different bytes after the fact, which is the one thing the frozen
        # manifest exists to prevent.
        frozen_ids = {
            str(row.get("source_id"))
            for row in source_manifest.get("sources", [])
            if isinstance(row, Mapping)
        }
        for row in raw_rows:
            if not isinstance(row, Mapping):
                raise DriverIntegrationError("supplementary source pack rows must be objects")
            if str(row.get("source_id")) in frozen_ids:
                raise DriverIntegrationError(
                    "supplementary source may not redefine a frozen manifest source_id"
                )
            if row.get("broker_outcome") != "fetched":
                raise DriverIntegrationError(
                    "supplementary manifest rows must all record a fetched broker outcome"
                )
        merged_rows, promotions = self._freeze_broker_pack_rows(raw_rows, root)
        # Record the ingest beside the manifest. The whole point of the defect
        # this closes was that a real, fetched source could vanish with nothing
        # written down, so the extension is never allowed to be silent either.
        custody_path = root / "source-custody.json"
        custody = _read_json(custody_path) if custody_path.is_file() else {}
        write_envelope_atomic(
            {
                **custody,
                "provenance_version": "menagerie.crawler.source-custody-provenance.v1",
                "stable_id": item.stable_id,
                "supplement_manifest_path": str(path),
                "supplement_promotions": promotions,
                "supplement_upstream_drift": [
                    dict(row["upstream_drift"])
                    for row in merged_rows
                    if isinstance(row.get("upstream_drift"), Mapping)
                ],
            },
            custody_path,
        )
        return {
            **source_manifest,
            SUPPLEMENTARY_SOURCES_KEY: merged_rows,
            "supplementary_manifest_sha256": stable_hash(merged_rows),
        }

    def _dispatch(
        self,
        *,
        kind: str,
        item: WorkItem,
        config: Optional[DriverConfig],
        work_id: str,
        request_path: Path,
        output_path: Path,
    ) -> Optional[str]:
        """Run one operator round trip that publishes ``output_path``."""

        raise NotImplementedError


def classify_author_exit(
    kind: str,
    stable_id: str,
    returncode: int,
    stdout: str,
    stderr: str,
    output_path: Optional[Path] = None,
) -> None:
    """Convert one operator exit code into the typed author-lane outcome.

    Risk R8: an unrecognized nonzero exit used to become a permanent model
    failure. The operator protocol exit codes are honored here so a transient
    failure is retryable, a quota response is a scheduler pause, and only a
    declared contract rejection is permanent.

    Parameters
    ----------
    kind:
        ``source-request`` or ``author`` round-trip label.
    stable_id:
        Model whose author session produced the exit.
    returncode:
        Operator process exit status.
    stdout, stderr:
        Captured operator output. Both are inspected: structured provider errors
        are observed on either stream.
    output_path:
        The envelope's required output path, when the caller can name it. Its
        state (a non-empty regular file or not) is the driver-observed
        publication fact the typed-outage refund rule reads; ``None`` (legacy
        callers) can never refund.

    Raises
    ------
    AuthorBackoffError
        On a rate/quota response, including exit ``76``.
    AuthorExecutorContractError
        On the executor's own refusal of a session's PUBLISHED discovery or
        result payload -- a model-scoped authoring failure, never a campaign
        abort.
    RetryableOperatorError
        On a declared retryable or unavailable operator failure.
    DriverIntegrationError
        On a declared permanent contract rejection or an unclassified exit.
    """

    if returncode == 0:
        return
    combined = f"{stderr}\n{stdout}".strip()
    label = "author command failed" if kind == "author" else "author source request failed"
    tail = combined[-STDIO_TAIL_MAX_CHARS:]
    session_notice = _structured_author_session_failure_notice(combined)
    signal = classify_author_response(returncode, stdout)
    if signal is None:
        signal = classify_author_response(returncode, stderr)
    if signal is not None:
        raise AuthorBackoffError(signal)
    if returncode in (AUTHOR_EXIT_RETRYABLE, AUTHOR_EXIT_UNAVAILABLE):
        expected_tool_failure = (
            "author executor stage1 failed: research-tools-unavailable (attempt "
        )
        expected_forge_limit = "author executor broker failed: forge-rate-limited (attempt "
        if returncode == AUTHOR_EXIT_RETRYABLE and any(
            line.startswith(expected_tool_failure) and line.endswith(")")
            for line in combined.splitlines()
        ):
            raise ResearchToolsUnavailableError(stable_id, tail)
        if returncode == AUTHOR_EXIT_RETRYABLE and any(
            line.startswith(expected_forge_limit) and line.endswith(")")
            for line in combined.splitlines()
        ):
            raise ForgeRateLimitedError(stable_id, tail)
        # A session that RAN TO COMPLETION and published payload bytes the
        # executor's own contract validator refused is evidence about this
        # model's authoring, not about the machine. The executor still relays
        # it through its retryable exit so a fresh session gets one bounded
        # chance, but classifying it as campaign-scoped transport (the
        # ``RetryableOperatorError`` below) let ONE schema-invalid author
        # result terminate an entire rung with every sibling's completed work
        # unrecorded (2026-07-27, m4066). Matched on the executor's exact
        # structured failure lines, like the two provider-outage arms above.
        # ``session-crashed`` / ``wall-exceeded`` / ``no-result-output`` are
        # deliberately NOT here: a session that died without publishing is
        # indistinguishable from broken infrastructure, and mis-terminalizing
        # every model on a host whose provider CLI is broken would be the
        # worse defect. Item 9 (recovery F1) made the settlement agree with
        # this comment: an executor that died TYPED before any session ran
        # (its own ``author executor error:`` / ``io error:`` line, no payload
        # at ``output_path``, no harness notice) now settles the closed
        # ``failed-executor-typed-outage`` and refunds the ATTEMPT projection;
        # the 82 pilot / 742 R6b sub-10 s deaths were charged as model
        # attempts under the bare ``failed`` this arm used to settle.
        expected_contract_failures = (
            "author executor stage1 failed: discovery-contract-invalid (attempt ",
            "author executor stage2 failed: result-not-json (attempt ",
            "author executor stage2 failed: result-contract-invalid (attempt ",
            # Pre-publication gate: a wall claim the executor's own clock refuted
            # (wall_seconds_observed < 0.5*grant, timed_out=false). The session ran
            # to completion and published an unverified self-report -- model
            # authoring evidence, never campaign transport.
            "author executor stage2 failed: unverified-wall-claim (attempt ",
        )
        if returncode == AUTHOR_EXIT_RETRYABLE and any(
            line.startswith(expected_contract_failures) and line.endswith(")")
            for line in combined.splitlines()
        ):
            raise AuthorExecutorContractError(stable_id, tail)
        exc = RetryableOperatorError(f"{label} for {stable_id} (exit {returncode}): {tail}")
        _stamp_retryable_exit_evidence(
            exc,
            session_notice,
            returncode=returncode,
            combined=combined,
            output_path=output_path,
        )
        raise exc
    if returncode == AUTHOR_EXIT_PERMANENT:
        # AL-3: the executor proved the broker refusal deterministic (identical
        # evidence digests on two consecutive attempts) and relayed it through
        # its permanent exit. Matched on the exact structured line, like the
        # provider-outage and contract arms above, and routed to its own typed
        # member-scoped terminal instead of the generic contract rejection --
        # whose catch-all would record `session-crashed`, a death that did not
        # occur.
        expected_deterministic = (
            "author executor broker failed: primary-implementation-unfetchable-deterministic "
            "(attempt "
        )
        if any(
            line.startswith(expected_deterministic) and line.endswith(")")
            for line in combined.splitlines()
        ):
            raise AuthorSourceDeterministicallyUnfetchableError(stable_id, tail)
        # Deliberately does NOT use the historical retryable prefix: a declared
        # contract rejection must not be retried.
        raise DriverIntegrationError(
            f"author {kind} rejected the contract for {stable_id} (exit {returncode}): {tail}"
        )
    # Unclassified exits keep the historical message prefix, which
    # ``_is_infrastructure_error`` already treats as one retryable transport
    # failure rather than an immediate permanent model failure.
    raise DriverIntegrationError(f"{label} for {stable_id}: {tail}")


def _stamp_retryable_exit_evidence(
    exc: RetryableOperatorError,
    session_notice: Optional[Mapping[str, Any]],
    *,
    returncode: int,
    combined: str,
    output_path: Optional[Path],
) -> None:
    """Stamp machine-derived classification onto one typed retryable exit.

    Two mutually exclusive evidence sources, both driver-observed. With a
    harness session-failure notice the diagnostic class rides along and only
    the notice's machine envelope may mint a refund outcome
    (``_notice_infrastructure_local_outcome``). Without a notice, the executor
    may have died TYPED before any session: the closed proof rule
    ``executor_typed_outage_outcome`` (exit code, the executor's own
    pre-session line, no payload at ``output_path``) mints
    ``failed-executor-typed-outage`` (item 9 / recovery F1).

    Parameters
    ----------
    exc:
        The typed retryable failure about to be raised (mutated in place).
    session_notice:
        Parsed executor-authored notice, or ``None``.
    returncode, combined, output_path:
        The observed exit status, the unsliced combined stdio, and the
        envelope's required output path (``None`` when unknown).
    """

    if session_notice is not None:
        exc.failure_class = _notice_string(session_notice, "failure_class")
        exc.failure_class_basis = _notice_failure_class_basis(session_notice)
        exc.refusal_excerpt = _notice_string(session_notice, "refusal_excerpt_quarantined")
        outcome = _notice_infrastructure_local_outcome(session_notice)
    else:
        outcome = executor_typed_outage_outcome(
            returncode,
            combined,
            published_payload=_author_payload_published(output_path),
        )
    if outcome is not None:
        exc.paid_session_local_outcome = outcome
        exc._paid_session_infrastructure_token = _PAID_SESSION_INFRASTRUCTURE_TOKEN


def _author_payload_published(output_path: Optional[Path]) -> Optional[bool]:
    """Return whether the executor published a payload at the envelope output path.

    The same predicate ``_read_json_object`` applies before it reads a result
    -- a non-empty regular file, not a symlink -- observed by the driver on
    its own filesystem. ``None`` when the caller could not name the path, so
    the refund rule charges conservatively.

    Parameters
    ----------
    output_path:
        The envelope's required output path, or ``None``.

    Returns
    -------
    bool | None
        ``True`` when a non-empty regular file is present, ``False`` when the
        path is absent, empty, or not a regular file, ``None`` when unknown.
    """

    if output_path is None:
        return None
    try:
        if output_path.is_symlink() or not output_path.is_file():
            return False
        return output_path.stat().st_size > 0
    except OSError:
        # An unreadable path is not a proven absence: charge.
        return None


def _notice_string(notice: Mapping[str, Any], key: str) -> Optional[str]:
    """Return a string field from a parsed executor notice.

    Parameters
    ----------
    notice:
        Parsed executor-authored notice.
    key:
        Field name to read.

    Returns
    -------
    str | None
        Stripped string field, or ``None`` when absent or non-string.
    """

    value = notice.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _notice_failure_class_basis(notice: Mapping[str, Any]) -> Optional[str]:
    """Return the trusted basis for one executor-authored session failure notice.

    Parameters
    ----------
    notice:
        Parsed executor-authored notice.

    Returns
    -------
    str | None
        Machine-readable weather-authority basis, or ``None`` when absent.
    """

    explicit = _notice_string(notice, "failure_class_basis")
    if explicit == "api_error_status:529":
        return explicit
    if (
        notice.get("failure_class") == "provider-overloaded"
        and notice.get("harness_terminal_reason") == "api_error"
        and notice.get("harness_is_error") is True
        and notice.get("returncode") == 1
    ):
        return "executor-record:provider-overloaded-api-error"
    if explicit is not None:
        return explicit
    return None


_PAID_SESSION_INFRASTRUCTURE_TOKEN = object()
_NOTICE_INFRASTRUCTURE_TERMINAL_OUTCOMES: Mapping[str, str] = {
    "api_error": "failed-harness-api-error",
}


def _notice_infrastructure_local_outcome(notice: Mapping[str, Any]) -> Optional[str]:
    """Derive one paid-session raw infrastructure outcome from a harness notice.

    Only machine envelope fields participate.  The notice's diagnostic class,
    basis, subtype, and quarantined text are ignored, so producer prose cannot
    mint an attempt refund.  Any missing or contradictory field returns
    ``None`` and therefore charges the attempt.

    Parameters
    ----------
    notice:
        Executor-authored session failure notice.

    Returns
    -------
    str | None
        Closed paid-session local outcome, or ``None`` conservatively.
    """

    if notice.get("harness_is_error") is not True:
        return None
    returncode = notice.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int) or returncode == 0:
        return None
    terminal = notice.get("harness_terminal_reason")
    if not isinstance(terminal, str):
        return None
    if terminal == "aborted_streaming" and returncode in {
        -signal.SIGTERM,
        128 + signal.SIGTERM,
    }:
        return "failed-harness-aborted-streaming-sigterm"
    return _NOTICE_INFRASTRUCTURE_TERMINAL_OUTCOMES.get(terminal)


def _paid_session_infrastructure_local_outcome(exc: BaseException) -> Optional[str]:
    """Return trusted machine infrastructure evidence from an exception chain.

    Parameters
    ----------
    exc:
        Failure raised around one paid guard session.

    Returns
    -------
    str | None
        Closed raw local outcome, or ``None`` when the attempt must charge.
    """

    current: Optional[BaseException] = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AuthorCampaignTeardown):
            return "campaign-teardown"
        if isinstance(current, OSError):
            return "failed-machine-spawn"
        if isinstance(current, AuthorQueueStalled):
            return "failed-author-queue-stalled"
        token = getattr(current, "_paid_session_infrastructure_token", None)
        outcome = getattr(current, "paid_session_local_outcome", None)
        if (
            token is _PAID_SESSION_INFRASTRUCTURE_TOKEN
            and isinstance(outcome, str)
            # The INFRASTRUCTURE subset only: the author-blocked-prerequisite
            # refund is stamped exclusively from the typed SUCCESS result at
            # the settle site, so an exception-carried spelling of it is
            # forged evidence and must charge (AL-6).
            and outcome in ATTEMPT_REFUND_INFRASTRUCTURE_LOCAL_OUTCOMES
        ):
            return outcome
        current = current.__cause__ or current.__context__
    return None


def _author_blocked_prerequisite_settlement_outcome(
    admission: Optional[tuple[str, Any]],
    result: Any,
) -> Optional[str]:
    """Return the AL-6 attempt-refund outcome for a typed BLOCKED author answer.

    The decision keys on the schema-validated recommendation KIND -- the typed
    ``BlockedRecommendation`` class ``parse_author_result`` produced from the
    closed ``arm`` discriminator -- never on any message text, and only for the
    author lane's own settle. An honest checked prerequisite verdict is the
    author doing its job; charging the model's attempt allowance for it is how
    the pilot's 20 reclassified members each burned an attempt on the truth.

    Parameters
    ----------
    admission:
        The retry ladder's ``(lane, item)`` admission pair, if any.
    result:
        The completed operation's return value.

    Returns
    -------
    str | None
        :data:`~menagerie.crawler.paid_session_accounting.AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME`
        exactly when an author-lane session returned a typed BLOCKED terminal
        recommendation; ``None`` charges conservatively everywhere else.
    """

    if admission is None or admission[0] != "author":
        return None
    # The typed BlockedRecommendation on the staged artifact is the whole
    # trust anchor; the wrapper's concrete type carries no extra evidence.
    if isinstance(getattr(result, "author_result", None), BlockedRecommendation):
        return AUTHOR_BLOCKED_PREREQUISITE_LOCAL_OUTCOME
    return None


def _author_session_failure_from_chain(exc: BaseException) -> Optional[dict[str, Any]]:
    """Return the executor session classification riding an exception chain.

    ``classify_author_exit`` stamps the structured notice's ``failure_class``,
    basis, and bounded refusal excerpt onto the typed retryable failure; the
    bounded ladder then wraps that failure in
    :class:`AuthorSessionRetryExhausted` ``from`` it. This walks the chain so
    the exhaustion boundary can persist the classification on the canonical
    attempt row (T4-T4) instead of losing it with the intermediate exception.

    Parameters
    ----------
    exc:
        Final exception after the bounded retry budget spent.

    Returns
    -------
    dict[str, Any] | None
        ``session_failure`` mapping for the attempt row builder, or ``None``
        when no link in the chain carries a classification.
    """

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        failure_class = getattr(current, "failure_class", None)
        if isinstance(failure_class, str) and failure_class:
            basis = getattr(current, "failure_class_basis", None)
            excerpt = getattr(current, "refusal_excerpt", None)
            return {
                "failure_class": failure_class,
                "failure_class_basis": basis if isinstance(basis, str) else None,
                "refusal_excerpt": excerpt if isinstance(excerpt, str) else None,
            }
        current = current.__cause__ or current.__context__
    return None


def _chain_carries_source_unavailability(exc: BaseException) -> bool:
    """Return whether an exhaustion chain carries the typed unfetchable class.

    W5 S arm: a primary-implementation-unavailable death is evidence about
    the model's SOURCE, not about the author lane's health, so the drought
    detector must never count it toward a lane-wide pattern. The class is
    typed (:class:`AuthorSourceDeterministicallyUnfetchableError`, the closed
    ``primary-implementation-unfetchable`` pair's carrier); prose matching is
    deliberately not accepted.
    """

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AuthorSourceDeterministicallyUnfetchableError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _structured_author_session_failure_notice(text: str) -> Optional[Mapping[str, Any]]:
    """Return the executor-authored author-session-failure notice in full text.

    The notice is line-oriented JSON printed by ``author_executor`` itself, not
    arbitrary model/checker transcript text. Callers pass the unsliced combined
    process output so a verbose quarantined error body cannot push the opening
    brace out of the stdio tail.

    Parameters
    ----------
    text:
        Full combined process output.

    Returns
    -------
    Mapping[str, Any] | None
        Parsed notice with the exact machine shape, or ``None``.
    """

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, Mapping)
            and payload.get("type") == "result"
            and payload.get("subtype") == "author-session-failure"
        ):
            return payload
    return None


def _checker_failure_evidence(request_path: Path, stdout: str, stderr: str) -> str:
    """Return the wrapper's own structured reason for one nonzero checker exit.

    The wrapper already classified its own failure and wrote the exact typed
    reason to its status sidecar. The driver used to ignore that and substitute
    ``stderr + stdout`` instead, which is not the reason: the wrapper re-emits
    the raw Codex ``--json`` EVENT STREAM on stderr, so the tail is whatever the
    model's last tool call happened to print. On 2026-07-30 that put verbatim
    repository source -- including this package's own test fixtures -- into a
    durable campaign record, while the sidecar sitting beside it said exactly
    ``'campaign_root_work_id' is a required property``.

    The stream tail is therefore a fallback, used only when the wrapper
    published no usable sidecar, and it is labelled as such so a reader is never
    left guessing whether the text is a diagnosis or a transcript.

    Parameters
    ----------
    request_path:
        Exact checker request path whose sidecar is read.
    stdout, stderr:
        Captured wrapper streams, used only for the labelled fallback.

    Returns
    -------
    str
        Bounded failure evidence.
    """

    try:
        sidecar = json.loads(status_sidecar_path(request_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        sidecar = None
    if isinstance(sidecar, Mapping):
        detail = sidecar.get("detail")
        if isinstance(detail, str) and detail.strip():
            classification = sidecar.get("classification")
            label = f" [{classification}]" if isinstance(classification, str) else ""
            # HEAD-bounded, unlike the stream tail below. A wrapper detail is a
            # DIAGNOSIS and its first clause is the diagnosis ("gate.v3
            # validation failed at items[0] ... is a required property");
            # keeping the last 1500 characters of a long schema error throws
            # exactly that away and leaves a list of property names.
            reason = detail.strip()
            if len(reason) > STDIO_TAIL_MAX_CHARS:
                reason = reason[:STDIO_TAIL_MAX_CHARS] + "...<truncated>"
            return f"{reason}{label}"
    tail = f"{stderr}\n{stdout}".strip()[-STDIO_TAIL_MAX_CHARS:]
    return f"the checker wrapper published no structured reason; raw stream tail: {tail}"


_CHECKER_PROVIDER_OVERLOAD_CLASS = "provider-overloaded"
_CHECKER_PROVIDER_OVERLOAD_BASIS = "codex-event:error:model-at-capacity"
#: Seat swap: the closed set of MACHINE-observed overload bases either checker
#: adapter may publish (checker_providers.PROVIDER_OVERLOAD_EVENT_BASES); the
#: storm fuse binds any member, never free text.
from menagerie.crawler.checker_providers import PROVIDER_OVERLOAD_EVENT_BASES as _CHECKER_PROVIDER_OVERLOAD_BASES  # noqa: E402,E501


def _request_checker_provider(request_path: Path) -> Optional[str]:
    """Return the provider of the seat one frozen checker request was sealed for."""

    try:
        request = load_absolute_request(request_path)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    checker = request.get("checker")
    model = checker.get("model") if isinstance(checker, Mapping) else None
    return checker_provider_for_model(str(model)) if isinstance(model, str) else None


def _checker_provider_weather_status(request_path: Path) -> Optional[Mapping[str, Any]]:
    """Return a fully bound checker provider-weather status sidecar.

    Parameters
    ----------
    request_path:
        Exact frozen checker request whose sibling status is inspected.

    Returns
    -------
    Mapping[str, Any] | None
        Structured status only when every machine-owned weather binding agrees.
    """

    try:
        request = load_absolute_request(request_path)
        status = json.loads(status_sidecar_path(request_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(status, Mapping):
        return None
    expected_request_sha256 = request.get("envelope_sha256")
    if not isinstance(expected_request_sha256, str):
        return None
    if (
        status.get("protocol_version") != OPERATOR_PROTOCOL_VERSION
        or status.get("status_version") != OPERATOR_STATUS_VERSION
        or status.get("exit_code") != int(OperatorExitCode.SERVICE_UNAVAILABLE)
        or status.get("classification") != _CHECKER_PROVIDER_OVERLOAD_CLASS
        or status.get("provider_event_basis") not in _CHECKER_PROVIDER_OVERLOAD_BASES
        or status.get("request_sha256") != expected_request_sha256
    ):
        return None
    return status


def _raise_for_checker_exit(
    returncode: int, stdout: str, stderr: str, *, request_path: Path
) -> None:
    """Convert one checker-wrapper exit code into its typed lane outcome.

    A checker VERDICT -- accepted, rejected, or cannot-verify -- only ever
    travels on exit ``0`` as a schema-valid gate result, and is a per-model
    outcome the terminal machinery already routes. A nonzero exit therefore
    always means "no verdict exists", but it does NOT always mean the
    infrastructure is broken, and it must never mean "abandon the campaign".
    The checker lane previously raised one undifferentiated
    ``DriverIntegrationError`` for every nonzero exit, so a declared-retryable
    or service-unavailable wrapper exit -- routine transport weather over a
    month-long run -- was indistinguishable from a genuine execution failure.
    This mirrors the author lane's R8 classification exactly.

    Parameters
    ----------
    returncode:
        Checker wrapper exit status.
    stdout, stderr:
        Captured wrapper output; structured provider errors appear on either.
    request_path:
        Exact checker request path, used to read the wrapper's structured
        status sidecar in preference to the raw stream tail.

    Raises
    ------
    RetryableOperatorError
        On a declared retryable or service-unavailable operator exit.
    DriverIntegrationError
        On a declared contract rejection (the wrapper ran but produced no valid
        gate) or an unclassified exit (a genuine execution failure: missing
        binary, crash, or absent output).
    """

    if returncode == 0:
        return
    tail = _checker_failure_evidence(request_path, stdout, stderr)
    if returncode in (
        int(OperatorExitCode.RETRYABLE_INFRASTRUCTURE),
        int(OperatorExitCode.SERVICE_UNAVAILABLE),
    ):
        exc = RetryableOperatorError(f"checker command failed (exit {returncode}): {tail}")
        weather = _checker_provider_weather_status(request_path)
        if weather is not None:
            exc.failure_class = "checker-provider-overloaded"
            exc.failure_class_basis = str(weather["provider_event_basis"])
            exc.request_sha256 = str(weather["request_sha256"])
            exc.checker_provider = _request_checker_provider(request_path)
        raise exc
    if returncode == int(OperatorExitCode.ITEM_CONTRACT_REJECTION):
        # BL-10 (c): the wrapper named the exact items whose contract failed.
        # Reconstruct the typed item-scoped error from the status sidecar so
        # the batch lane can route the named models and re-batch the innocent
        # siblings. An unreadable or incomplete sidecar falls through to the
        # generic contract rejection below (fail-closed into the whole-batch
        # downshift route, never into a silent pass).
        item_contract = _checker_item_contract_status(request_path)
        if item_contract is not None:
            stable_ids, author_side = item_contract
            raise CheckerItemContractError(
                f"checker item contract refused for {', '.join(stable_ids)} "
                f"(exit {returncode}): {tail}",
                stable_ids=stable_ids,
                author_side=author_side,
            )
        raise DriverIntegrationError(
            f"checker item contract rejection carried no readable item attribution "
            f"(exit {returncode}): {tail}"
        )
    if returncode == int(OperatorExitCode.PERMANENT_CONTRACT_REJECTION):
        # Names the CHECKER as the violating party. The wrapper is the party
        # that DETECTED the violation, and saying "the checker wrapper rejected
        # the gate contract" reads as though the wrapper were at fault; every
        # exit-64 sidecar observed so far records a checker-authored gate that
        # failed its own schema. The wrapper is still named, as the detector.
        raise DriverIntegrationError(
            f"checker violated the gate contract and its result was refused by the operator "
            f"wrapper (exit {returncode}): {tail}"
        )
    # Unclassified exits keep the historical message prefix, which
    # ``_is_infrastructure_error`` already treats as one retryable transport
    # failure before it becomes permanent.
    raise DriverIntegrationError(f"checker command failed: {tail}")


def _checker_item_contract_status(request_path: Path) -> Optional[tuple[tuple[str, ...], bool]]:
    """Read the wrapper's item-scoped contract attribution from its sidecar.

    Returns ``(stable_ids, author_side)`` or ``None`` when the sidecar is
    absent, unreadable, or does not carry a well-formed ``item_contract``.
    """

    from menagerie.crawler.operator_protocol import status_sidecar_path

    try:
        status = json.loads(status_sidecar_path(request_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(status, Mapping):
        return None
    contract = status.get("item_contract")
    if not isinstance(contract, Mapping):
        return None
    stable_ids = contract.get("stable_ids")
    author_side = contract.get("author_side")
    if (
        not isinstance(stable_ids, list)
        or not stable_ids
        or not all(isinstance(value, str) and value for value in stable_ids)
        or not isinstance(author_side, bool)
    ):
        return None
    return tuple(stable_ids), author_side


#: How long the driver waits for a paid subtree to disappear from BOTH kernel
#: oracles after its guard exits, and how long it then waits for a reap.  Both
#: are bounded: an unbounded wait on a descendant is how a fail-closed refusal
#: became a silent total stall (round-3 O3-B2, invariant I2 DEADLINE).
PAID_SUBTREE_EXIT_GRACE_SECONDS = 5.0
PAID_SUBTREE_REAP_GRACE_SECONDS = 5.0


def _close_paid_custody(
    authority: PaidSessionAuthority,
    reservation: Reservation,
    group: Any,
    lock_identity: tuple[int, int],
) -> None:
    """Release custody ONLY on a machine-observed exit of the whole paid subtree.

    ``verified-exit`` is granted when both independent kernel oracles -- the
    inherited ``flock`` (which the credential-holding grandchild also holds) and
    the paid process group -- report the subtree gone.  Nothing the wrapper says
    about itself participates: a token a producer writes is evidence about the
    writer, never about its descendants, and a positive custody outcome derived
    from it is the producer-supplied field the waiver doctrine forbids.

    Nothing is signalled here.  The guard has already been reaped, so its start
    token no longer exists and this routine cannot VERIFY the group it would be
    targeting; the timeout arm above kills the group while the leader is still
    alive and verifiable, and the restart reaper does the same for a lease whose
    identity still matches.  A subtree that cannot be proven gone therefore
    leaves custody UNRELEASED and the lease DISCLOSED for the reaper and the
    operator drain -- fail-closed, with recovery, and bounded: the two kernel
    probes are cheap, the deadline is fixed, and there is no wait on a
    descendant that a live orphan could extend.
    """

    lock_deadline = time.monotonic() + PAID_SUBTREE_EXIT_GRACE_SECONDS
    lock_state = subtree_liveness(reservation.lock_path, expected_identity=lock_identity)
    while lock_state == ALIVE and time.monotonic() < lock_deadline:
        time.sleep(0.05)
        lock_state = subtree_liveness(reservation.lock_path, expected_identity=lock_identity)
    # The membership enumeration is the expensive probe, so it runs only after
    # the cheap one settles, and at a coarse interval within its own deadline.
    group_deadline = time.monotonic() + PAID_SUBTREE_REAP_GRACE_SECONDS
    group_state = process_group_liveness(group.pgid)
    while group_state == ALIVE and time.monotonic() < group_deadline:
        time.sleep(0.25)
        group_state = process_group_liveness(group.pgid)
    # The flock is authoritative and must be FREE. The group probe only ever
    # ADDS a refusal (the descendant that dropped the descriptor but stayed in
    # the group); a host that cannot enumerate membership falls back to the
    # flock rather than refusing every clean session.
    if lock_state == FREE and group_state != ALIVE:
        authority.release_custody(reservation, outcome="verified-exit")
        return
    raise DriverIntegrationError(
        "paid subtree could not be proven exited after its guard returned "
        f"(custody flock {lock_state}, process group {group_state}); custody stays "
        "unreleased and the lease stays disclosed for the restart reaper"
    )


class _PaidDispatchSessions:
    """One dispatch operation's paid-session discipline: one reservation per guard.

    A :class:`Reservation` is ONE paid provider session -- one provider
    idempotency key, one custody bind, one settlement.  An author-lane operation
    legitimately runs MORE than one guard session (``_fetch_author_sources``
    then ``_author_from_frozen_sources``), and on 2026-08-20 the first real paid
    step-200 leg reused its attempt's single reservation for the second guard:
    the rebind row (``session_custody_bound`` after ``session_custody_released``,
    same spend id) was durably appended and every later money read of the ledger
    refused with ``duplicate custody`` -- three consecutive terminal-recording
    failures and a halted leg.

    This object makes reservation reuse impossible by construction:

    * the FIRST guard session consumes the attempt's already-minted reservation
      (the attempt group's HEAD);
    * every LATER guard session first settles the finished one (its own measured
      wall span, provider-attested as always) and only then admits a fresh
      reservation through the same writer-owned admission valve the attempt
      used, JOINED to the head's attempt group -- so each guard session carries
      its own spend id, idempotency key, custody lease, and budget charge,
      while the durable attempt count stays one;
    * a lane that declares NO per-session admission (the checker lane dispatches
      exactly one guard per reservation) refuses a second guard session typed,
      BEFORE any child is spawned or any provider byte is exposed.

    The ladder settles whatever reservation is still unsettled when the
    operation returns or raises, exactly once.  Wall spans: each settlement's
    machine floor covers the time from the previous boundary's timer reset to
    the moment the boundary is reached, so inter-guard work (broker time before
    the second guard) is charged to the FIRST reservation's span, and the time
    spent settling and admitting at the boundary itself is excluded from both
    spans.  The sum of spans is therefore a floor on (not an exact partition
    of) the attempt's wall; no second is ever double-charged, and the
    provider-attested wall can still raise any span it meters.
    """

    def __init__(
        self,
        reservation: Reservation,
        *,
        admit: Optional[Callable[[Optional[str]], Reservation]],
        settle: Optional[Callable[[Reservation, float, bool, str], None]],
        span_started: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the attempt's minted reservation and its admission/settlement seams.

        Parameters
        ----------
        reservation:
            The attempt's already-admitted reservation, unconsumed.  Its spend
            id is the attempt group's durable HEAD identity.
        admit:
            Writer-owned admission for a FRESH reservation, called with this
            attempt's group head so the fresh buy JOINS the attempt instead of
            opening a new one; ``None`` for a lane that allows exactly one
            guard session per reservation.
        settle:
            Writer-owned settlement callable ``(reservation, elapsed_seconds,
            completed)``.  Required when ``admit`` is provided; may be ``None``
            only alongside ``admit=None`` (consume-once enforcement without a
            mid-operation boundary to settle).
        span_started:
            ``monotonic()`` origin of the active reservation's wall span.
        monotonic:
            Injectable monotonic clock measuring every span this object owns.
            ``span_started`` must come from the SAME clock.
        """

        if admit is not None and settle is None:
            raise DriverIntegrationError(
                "per-session paid admission requires a settlement seam: a fresh "
                "reservation may only be admitted after the finished session settles"
            )
        self._unsettled: Optional[Reservation] = reservation
        self._attempt_head = reservation.spend_id
        self._admit = admit
        self._settle = settle
        self._span_started = span_started
        self._monotonic = monotonic
        self._guard_consumed = False
        self._lock = threading.Lock()

    def begin_guard_session(self) -> Reservation:
        """Return this guard session's own reservation, minting one if needed.

        Raises
        ------
        DriverIntegrationError
            When a second guard session starts in a lane that declared no
            per-session admission.  Raised before any spawn, so no provider
            byte is exposed and no custody row is minted.
        """

        with self._lock:
            if self._unsettled is not None and not self._guard_consumed:
                self._guard_consumed = True
                return self._unsettled
            if self._admit is None or self._settle is None:
                raise DriverIntegrationError(
                    "a second paid guard session started inside one paid dispatch, "
                    "but this lane admits exactly one custody bind per reservation; "
                    "refusing before any provider byte is exposed"
                )
            if self._unsettled is not None:
                elapsed = max(0.0, self._monotonic() - self._span_started)
                finished, self._unsettled = self._unsettled, None
                self._span_started = self._monotonic()
                # The finished guard exited machine-verified (its dispatch
                # returned), so its session settles completed over its own
                # measured span BEFORE the next reservation can exist: the
                # ledger's verified chain head moves, and the next admission
                # mints a different spend id and provider idempotency key.
                self._settle(finished, elapsed, True, "completed")
            # The fresh buy JOINS this attempt's durable group, so the attempt
            # allowance counts ONE attempt however many guard sessions it runs.
            fresh = self._admit(self._attempt_head)
            self._unsettled = fresh
            self._span_started = self._monotonic()
            self._guard_consumed = True
            return fresh

    def settle_outcome(
        self,
        *,
        completed: bool,
        minimum_wall: float = 0.0,
        observed_wall: Optional[float] = None,
        local_outcome: Optional[str] = None,
    ) -> None:
        """Settle the still-unsettled reservation exactly once, if any remains.

        Parameters
        ----------
        completed:
            Whether the operation returned; disclosure-only, as always.
        minimum_wall:
            Prepaid wall floor a pool worker already measured around this
            reservation's real session (a replay costs ~0s on this thread).
        observed_wall:
            The lane's own authoritative session wall (checker outcomes carry
            one), overriding the span measurement when present.
        local_outcome:
            Exact driver-derived local outcome.  ``None`` preserves the generic
            completed/failed disclosure and therefore charges conservatively.
        """

        with self._lock:
            if self._unsettled is None:
                return
            elapsed = max(0.0, self._monotonic() - self._span_started)
            if observed_wall is not None:
                elapsed = max(0.0, float(observed_wall))
            if self._settle is None:
                raise DriverIntegrationError(
                    "a paid dispatch scope without a settlement seam cannot settle; "
                    "the caller owns this reservation's settlement"
                )
            unsettled, self._unsettled = self._unsettled, None
            outcome = local_outcome or ("completed" if completed else "failed")
            self._settle(unsettled, max(minimum_wall, elapsed), completed, outcome)


_PAID_DISPATCH_SESSIONS: ContextVar[Optional[_PaidDispatchSessions]] = ContextVar(
    "paid_dispatch_sessions", default=None
)


@contextmanager
def _paid_dispatch_sessions(sessions: _PaidDispatchSessions) -> Iterator[None]:
    """Bind one operation's per-guard paid-session discipline."""

    token = _PAID_DISPATCH_SESSIONS.set(sessions)
    try:
        yield
    finally:
        _PAID_DISPATCH_SESSIONS.reset(token)


#: Writer-context binding of the live author wave's bounded admission drain
#: (:meth:`_AuthorWavePool.service_admissions`), stamped with the installing
#: thread's identity. Installed by ``_ensure_authors`` in the CANONICAL
#: WRITER's own context for exactly the wave's lifetime, so the writer's long
#: non-claim stretches -- an inline operator command's outcome poll, an
#: infrastructure-retry backoff -- drain queued worker buys instead of starving
#: them. The reserve site can never move off the writer through this variable:
#: pool worker threads normally read the ``None`` default (fresh thread
#: contexts), and even under a context-propagating runtime the helper no-ops on
#: any thread other than the one that installed the binding; the pool's own
#: thread tripwire refuses a direct off-writer drain a third time over.
_AUTHOR_ADMISSION_RELAY_SERVICE: ContextVar[
    Optional[tuple[int, Callable[[], None]]]
] = ContextVar("author_admission_relay_service", default=None)


@contextmanager
def _author_admission_relay_service(service: Callable[[], None]) -> Iterator[None]:
    """Bind one wave's writer-owned relay drain for the writer's own waits."""

    token = _AUTHOR_ADMISSION_RELAY_SERVICE.set((threading.get_ident(), service))
    try:
        yield
    finally:
        _AUTHOR_ADMISSION_RELAY_SERVICE.reset(token)


def _service_author_admission_relay() -> None:
    """Drain queued author admissions when this thread owns the wave relay.

    A no-op everywhere except the canonical writer inside a live author wave:
    the service callable commits each queued buy through the same
    ``_admit_author_spend`` valve (shutdown re-check, custody reaper, durable
    allowance, fold-first reserve) the claim-wait servicing already uses, and
    a sealed window still refuses every request typed at service time (W1).
    """

    entry = _AUTHOR_ADMISSION_RELAY_SERVICE.get()
    if entry is None:
        return
    ident, service = entry
    if threading.get_ident() != ident:
        return
    service()


class _CampaignTeardownController:
    """Thread-safe parent authority for one author wave's mandatory teardown."""

    def __init__(self) -> None:
        """Open a controller with no requested halt."""

        self._event = threading.Event()
        self._lock = threading.Lock()
        self._trigger: Optional[str] = None
        self._reason: Optional[str] = None

    def request(self, *, trigger: str, reason: str) -> None:
        """Publish the wave halt before any worker delivers a teardown signal.

        Parameters
        ----------
        trigger:
            Stable driver boundary that initiated the halt.
        reason:
            Bounded machine explanation retained in the teardown record.
        """

        with self._lock:
            if self._trigger is None:
                self._trigger = trigger[:128]
                self._reason = reason[:512]
                self._event.set()

    def requested(self) -> Optional[tuple[str, str]]:
        """Return the immutable halt trigger and reason once requested.

        Returns
        -------
        tuple[str, str] | None
            Halt facts, or ``None`` while this wave remains live.
        """

        if not self._event.is_set():
            return None
        with self._lock:
            if self._trigger is None or self._reason is None:
                return None
            return self._trigger, self._reason


_CAMPAIGN_TEARDOWN_CONTROLLER: ContextVar[Optional[_CampaignTeardownController]] = ContextVar(
    "campaign_teardown_controller", default=None
)


@contextmanager
def _campaign_teardown_controller(
    controller: _CampaignTeardownController,
) -> Iterator[None]:
    """Bind a wave teardown controller to one author worker thread."""

    token = _CAMPAIGN_TEARDOWN_CONTROLLER.set(controller)
    try:
        yield
    finally:
        _CAMPAIGN_TEARDOWN_CONTROLLER.reset(token)


def _write_campaign_teardown_record(path: Path, value: Mapping[str, Any]) -> None:
    """Durably publish one parent-owned teardown intent before signalling.

    Parameters
    ----------
    path:
        Per-guard record path supplied in the child's environment.
    value:
        Complete machine-authored record.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_operator_command(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run one operator command in its own process group under a hard wall bound.

    An operator wrapper is an interactive agent session behind a subprocess, and an
    unbounded one blocks the single-threaded driver forever: in a month-long unattended
    campaign a single hung session is a silent total stall with no recovery. The bound is
    external because the wrapper is exactly the thing that may have stopped making
    progress, so its own deadline cannot be trusted to fire.

    The child leads its own session, so the timeout tears down the **whole group**. A
    wrapper typically spawns the real agent as a grandchild; killing only the direct child
    would leave that grandchild holding its file handles, its lease, and its provider
    session.

    Parameters
    ----------
    argv:
        Exact non-shell operator argv.
    timeout_seconds:
        Wall-clock ceiling for the round trip.
    env:
        Complete environment for the child, or ``None`` to inherit the driver's.
        Per-invocation values belong here rather than in the driver's own
        ``os.environ``: a library function that mutates the process environment
        leaks into every later caller in that process, so a supervised driver
        handling one campaign would leave that campaign's values visible to the
        next one.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Captured exit status and streams, for the caller's own exit classification.

    Raises
    ------
    subprocess.TimeoutExpired
        When the command outlives the bound. The group is force-terminated and reaped
        before this is raised, and whatever the wrapper managed to emit is attached.
    """

    # One guard session = one reservation. The dispatch scope settles the
    # previous guard's reservation and admits a fresh one when the SAME
    # operation runs a second guard (the author lane's source-request then
    # author sessions); rebinding one reservation's custody is what bricked
    # the 2026-08-20 step-200 spend ledger. Direct callers without a scope
    # (unit rigs) keep the single-session contextvar contract.
    sessions = _PAID_DISPATCH_SESSIONS.get()
    reservation = (
        sessions.begin_guard_session() if sessions is not None else current_paid_reservation()
    )
    authority = PaidSessionAuthority(
        reservation.ledger_path,
        runtime_root=reservation.lock_path.parent.parent,
        clock=utc_now,
    )
    subtree_lock = open_subtree_lock(reservation.lock_path)
    lock_identity = (subtree_lock.device, subtree_lock.inode)
    release_read_fd, release_write_fd = os.pipe()
    os.set_inheritable(release_read_fd, True)
    child_environment = os.environ.copy() if env is None else dict(env)
    child_environment[CUSTODY_FD_ENV] = str(subtree_lock.fd)
    child_environment["MENAGERIE_PAID_RELEASE_FD"] = str(release_read_fd)
    child_environment["MENAGERIE_PROVIDER_IDEMPOTENCY_KEY"] = (
        reservation.provider_idempotency_key
    )
    teardown_controller = _CAMPAIGN_TEARDOWN_CONTROLLER.get()
    request_parent = Path(argv[-1]).expanduser().resolve().parent
    teardown_path = request_parent / (
        f"campaign-teardown-{reservation.spend_id}-{uuid.uuid4().hex}.json"
    )
    child_environment[CAMPAIGN_TEARDOWN_RECORD_ENV] = str(teardown_path)
    # The guard starts behind a driver-owned release pipe.  The durable custody
    # binding lands after spawn but before the guard can expose a provider byte.
    guarded_argv = [
        sys.executable,
        "-m",
        "menagerie.crawler.author_executor",
        "--parent-deadline-guard",
        str(os.getpid()),
        repr(float(timeout_seconds)),
        *argv,
    ]
    try:
        process = subprocess.Popen(
            guarded_argv,
            env=child_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            pass_fds=(subtree_lock.fd, release_read_fd),
        )
    except BaseException:
        os.close(release_read_fd)
        os.close(release_write_fd)
        subtree_lock.close_parent_copy()
        raise
    group = capture_process_group(process)
    campaign_teardown_record: Optional[JsonObject] = None
    leader_token = process_start_token(process.pid)
    if leader_token is None or not group.isolated:
        kill_process_group(group)
        os.close(release_read_fd)
        os.close(release_write_fd)
        subtree_lock.close_parent_copy()
        raise DriverIntegrationError("paid guard did not establish a verifiable process group")
    try:
        authority.bind_custody(
            reservation,
            leader_pid=process.pid,
            leader_pgid=group.pgid,
            leader_start_token=leader_token,
            boot_id=current_boot_id(),
            host_id=reservation.host_id,
            lock_device=subtree_lock.device,
            lock_inode=subtree_lock.inode,
        )
        subtree_lock.close_parent_copy()
        if subtree_liveness(reservation.lock_path, expected_identity=lock_identity) != ALIVE:
            kill_process_group(group)
            raise DriverIntegrationError(
                "paid guard did not inherit the custody flock; provider exposure refused"
            )
        os.write(release_write_fd, b"1")
        os.close(release_write_fd)
        release_write_fd = -1
        os.close(release_read_fd)
        release_read_fd = -1
        # The guard owns the exact deadline and needs a small teardown window
        # after it fires to kill the separate operator group and remove its
        # credential directory. Killing the guard at the same instant as its
        # deadline would recreate the orphan boundary it exists to close.
        communicate_deadline = time.monotonic() + timeout_seconds + 10.0
        while True:
            # The canonical writer dispatches inline operator commands (a
            # terminal-disposition checker session, the non-pool-certain author
            # fallback) with the author wave still open; each outcome poll
            # drains the wave's queued worker buys so a multi-minute inline
            # session no longer starves every sibling's session start. Worker
            # threads read the ``None`` default here and skip.
            _service_author_admission_relay()
            requested = (
                teardown_controller.requested()
                if teardown_controller is not None
                else None
            )
            if requested is not None:
                trigger, reason = requested
                campaign_teardown_record = {
                    "version": CAMPAIGN_TEARDOWN_RECORD_VERSION,
                    "driver_shutdown_state": "halt-teardown-requested",
                    "target_pid": group.pid,
                    "target_pgid": group.pgid,
                    "signal": signal.SIGTERM,
                    "reason": reason,
                    "teardown_trigger": trigger,
                    "recorded_at": utc_now(),
                    "outcome": "signal-pending",
                }
                _write_campaign_teardown_record(teardown_path, campaign_teardown_record)
                delivered = kill_process_group(group, signal.SIGTERM)
                campaign_teardown_record.update(
                    {
                        "outcome": delivered.outcome,
                        "detail": delivered.detail,
                    }
                )
                _write_campaign_teardown_record(teardown_path, campaign_teardown_record)
                try:
                    stdout, stderr = process.communicate(timeout=30.0)
                except subprocess.TimeoutExpired:
                    forced = kill_process_group(group, signal.SIGKILL)
                    campaign_teardown_record["escalation"] = {
                        "target_pid": group.pid,
                        "target_pgid": group.pgid,
                        "signal": signal.SIGKILL,
                        "reason": "campaign teardown grace expired",
                        "teardown_trigger": trigger,
                        "outcome": forced.outcome,
                        "detail": forced.detail,
                    }
                    _write_campaign_teardown_record(teardown_path, campaign_teardown_record)
                    try:
                        stdout, stderr = process.communicate(timeout=5.0)
                    except subprocess.TimeoutExpired as exc:
                        stdout = str(exc.output or "")
                        stderr = str(exc.stderr or "")
                break
            remaining = communicate_deadline - time.monotonic()
            if remaining <= 0.0:
                raise subprocess.TimeoutExpired(list(argv), timeout_seconds)
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
            break
    except subprocess.TimeoutExpired:
        timeout_record: JsonObject = {
            "version": CAMPAIGN_TEARDOWN_RECORD_VERSION,
            "driver_shutdown_state": "driver-timeout-escalation",
            "target_pid": group.pid,
            "target_pgid": group.pgid,
            "signal": signal.SIGTERM,
            "reason": "driver stall bound expired around the paid author guard",
            "teardown_trigger": "driver-stall-timeout",
            "recorded_at": utc_now(),
            "outcome": "signal-pending",
        }
        _write_campaign_teardown_record(teardown_path, timeout_record)
        delivered = kill_process_group(group, signal.SIGTERM)
        timeout_record.update({"outcome": delivered.outcome, "detail": delivered.detail})
        _write_campaign_teardown_record(teardown_path, timeout_record)
        try:
            stdout, stderr = process.communicate(timeout=30.0)
        except subprocess.TimeoutExpired as exc:
            forced = kill_process_group(group, signal.SIGKILL)
            timeout_record["escalation"] = {
                "target_pid": group.pid,
                "target_pgid": group.pgid,
                "signal": signal.SIGKILL,
                "reason": "driver stall teardown grace expired",
                "teardown_trigger": "driver-stall-timeout",
                "outcome": forced.outcome,
                "detail": forced.detail,
            }
            _write_campaign_teardown_record(teardown_path, timeout_record)
            stdout = str(exc.output or "")
            stderr = str(exc.stderr or "")
        raise subprocess.TimeoutExpired(
            list(argv), timeout_seconds, output=stdout, stderr=stderr
        ) from None
    finally:
        if release_read_fd >= 0:
            os.close(release_read_fd)
        if release_write_fd >= 0:
            os.close(release_write_fd)
        subtree_lock.close_parent_copy()
    if campaign_teardown_record is not None:
        _close_paid_custody(authority, reservation, group, lock_identity)
        raise AuthorCampaignTeardown(
            trigger=str(campaign_teardown_record["teardown_trigger"]),
            teardown_record=campaign_teardown_record,
            stderr_tail=(stderr or "")[-STDIO_TAIL_MAX_CHARS:],
        )
    if process.returncode in {70, -signal.SIGKILL, -signal.SIGTERM}:
        raise subprocess.TimeoutExpired(
            list(argv), timeout_seconds, output=stdout, stderr=stderr
        )
    _close_paid_custody(authority, reservation, group, lock_identity)
    return subprocess.CompletedProcess(
        list(argv),
        process.returncode if process.returncode is not None else 0,
        stdout,
        stderr,
    )


class CommandAuthorLane(_AuthorLaneBase):
    """Author lane that writes the frozen envelope and invokes an injected command."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        effort_grant: Optional[AuthorEffortGrant] = None,
        require_receipt: bool = False,
        stall_bound_seconds: Optional[float] = None,
    ) -> None:
        """Store a non-shell Claude Code command prefix and its effort grant.

        Parameters
        ----------
        command:
            Wrapper argv prefix; the absolute request path is appended.
        effort_grant:
            Per-session effort budget.
        require_receipt:
            When true, every successful round trip must print an executor
            publication receipt whose digest matches the published bytes.
            Set for the headless author executor; fixture wrappers that
            predate the receipt protocol leave it off, and any receipt they
            *do* print is still verified.
        stall_bound_seconds:
            Outer stall bound, mirroring :class:`QueueAuthorLane`'s injectable
            deadline. Production leaves this ``None`` and takes the bound
            derived from the grant. It is validated against the executor's own
            kill so an explicit value can never reintroduce a bound that
            truncates the campaign's budget from outside.

        Raises
        ------
        ValueError
            If the command is empty, or an explicit stall bound would fire
            before the executor's own per-session kill.
        """

        if not command:
            raise ValueError("author command cannot be empty")
        self.command = tuple(command)
        self.effort_grant = effort_grant or AuthorEffortGrant()
        self.require_receipt = require_receipt
        if stall_bound_seconds is not None:
            executor_kill = self.effort_grant.wall_seconds * AUTHOR_WALL_EXTERNAL_KILL_FACTOR
            if float(stall_bound_seconds) <= executor_kill:
                raise ValueError(
                    f"author stall bound {float(stall_bound_seconds):g}s would fire at or "
                    f"before the executor's own {executor_kill:g}s kill for a "
                    f"{self.effort_grant.wall_seconds:g}s grant, truncating the session "
                    "budget from outside"
                )
        self.stall_bound_seconds = stall_bound_seconds

    def _child_environment(self) -> dict[str, str]:
        """Return the wrapper's complete environment, carrying this lane's grant.

        The executor is a subprocess that reads its wall grant from the
        environment, so the grant travels in the CHILD's environment and nowhere
        else. Writing it into the driver's own ``os.environ`` instead would make
        one campaign's budget visible to everything that runs later in the same
        process -- including a supervised driver's next campaign, whose correct
        configuration the startup guard would then refuse.

        Returns
        -------
        dict[str, str]
            The driver's environment overlaid with the resolved grant.
        """

        return {
            **os.environ,
            AUTHOR_WALL_SECONDS_ENV: repr(float(self.effort_grant.wall_seconds)),
        }

    def _dispatch(
        self,
        *,
        kind: str,
        item: WorkItem,
        config: Optional[DriverConfig],
        work_id: str,
        request_path: Path,
        output_path: Path,
    ) -> Optional[str]:
        """Invoke the wrapper with one absolute request path and classify its exit.

        Returns
        -------
        Optional[str]
            The executor's ``author_model`` launch attestation from the verified
            publication receipt, or ``None`` when no attestation was printed.
        """

        # ``output_path`` stays live: the executor's publication receipt is
        # verified against the bytes at that path after a clean exit.
        del config, work_id
        argv = [*self.command, str(request_path)]
        # A STALL GUARD, derived from the published grant -- not a second budget.
        # The grant is enforced INSIDE the wrapper, per session, where the stage
        # that overran is known and a typed recoverable outcome can be recorded;
        # this bound only stops a wedged wrapper blocking the driver forever. It
        # therefore has to clear every inner limit, including a stage 2 that
        # legitimately runs a cold rerun and a supplementary round. Bounding at
        # the bare grant instead silently truncated exactly the long tail the
        # per-campaign grant exists to cover, and mislabelled it as a timeout.
        bound = (
            author_lane_wall_bound(self.effort_grant.wall_seconds)
            if self.stall_bound_seconds is None
            else float(self.stall_bound_seconds)
        )
        try:
            completed = _run_operator_command(
                argv, timeout_seconds=bound, env=self._child_environment()
            )
        except subprocess.TimeoutExpired as exc:
            # Typed and retryable, like every other transport failure this lane classifies.
            # A hung session is stalled infrastructure; burning the model permanently for
            # it would turn one transient hang into a lost model.
            raise RetryableOperatorError(
                f"author {kind} for {item.stable_id} outlived the {bound:g}s stall bound "
                f"around its {self.effort_grant.wall_seconds:g}s wall grant and its process "
                f"group was terminated: "
                f"{str(exc.stderr or exc.output or '')[-STDIO_TAIL_MAX_CHARS:]}"
            ) from exc
        classify_author_exit(
            kind,
            item.stable_id,
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
            output_path=output_path,
        )
        receipt = _verify_executor_receipt(
            completed.stdout or "",
            output_path,
            stable_id=item.stable_id,
            required=self.require_receipt,
        )
        return _receipt_author_model(receipt)


def _verify_executor_receipt(
    stdout: str,
    output_path: Path,
    *,
    stable_id: str,
    required: bool,
) -> Optional[Mapping[str, Any]]:
    """Verify an executor publication receipt against the bytes now on disk.

    The supersession discipline (SEAM_REDESIGN 3.4): completion is bound to
    the attempt nonce it started with, and the lane re-verifies the receipt
    digest against the bytes it is about to read, so a late writer racing the
    published result — the TOCTOU half of the nonce-laundering class — is a
    typed integration failure, never a silently consumed result.

    Parameters
    ----------
    stdout:
        Wrapper stdout; the receipt is its last JSON object line.
    output_path:
        The envelope's required output path.
    stable_id:
        Model identity, for diagnostics.
    required:
        Whether a missing receipt is itself a failure.

    Returns
    -------
    Optional[Mapping[str, Any]]
        The verified receipt, or ``None`` when a fixture wrapper printed none
        and none was required. The lane reads the executor's ``author_model``
        launch attestation off it.

    Raises
    ------
    DriverIntegrationError
        On a missing-but-required, unbound, or digest-mismatched receipt.
    """

    from menagerie.crawler.author_executor import RECEIPT_VERSION

    receipt: Optional[Mapping[str, Any]] = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping) and parsed.get("receipt_version") == RECEIPT_VERSION:
            receipt = parsed
            break
    if receipt is None:
        if required:
            raise DriverIntegrationError(
                f"author executor printed no publication receipt for {stable_id}"
            )
        return None
    nonce = str(receipt.get("attempt_nonce", ""))
    digest = str(receipt.get("result_sha256", ""))
    if not nonce or not digest:
        raise DriverIntegrationError(
            f"author executor receipt for {stable_id} is not attempt-bound"
        )
    try:
        actual = hash_bytes(Path(output_path).read_bytes())
    except OSError as exc:
        raise DriverIntegrationError(
            f"author executor receipt for {stable_id} names an unreadable result: {exc}"
        ) from exc
    if actual != digest:
        raise DriverIntegrationError(
            f"author result bytes for {stable_id} do not match the executor receipt "
            f"digest (receipt {digest}, on disk {actual}); refusing a possibly "
            "laundered or raced result"
        )
    return receipt


def _receipt_author_model(receipt: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return a receipt's launch attestation, or ``None`` when it carries none.

    ``None`` covers three honestly distinct cases the record layer treats
    identically -- no receipt (fixture wrapper), a pre-AM1 receipt without the
    field, and an explicitly unpinned launch -- because all three mean the same
    thing downstream: no machine attestation exists, so provenance keeps the
    stamped claim.
    """

    if receipt is None:
        return None
    value = receipt.get("author_model")
    return value if isinstance(value, str) and value else None


_AUTHOR_JOB_VERSION = author_queue.AUTHOR_JOB_VERSION
author_queue_directories = author_queue.author_queue_directories


class QueueAuthorLane(_AuthorLaneBase):
    """Author lane that bridges the engine to an in-session Claude subagent pool.

    The engine's author boundary is a subprocess contract: run a command with one
    absolute request path and read the result from the envelope's exact
    ``required_output_path``. Re-entering the ``claude`` CLI per model to satisfy
    that contract costs a full interactive context load per invocation, so the
    production pool is instead a set of subagents dispatched inside one live
    managing session.

    This lane is the bridge. It builds the same two envelopes as
    :class:`CommandAuthorLane`, publishes a job descriptor into a file queue, and
    blocks until the operator publishes the result at the exact required path.
    Every validator, staging step, and identity derivation downstream is
    unchanged, so the authority side of the contract is fully conserved.

    Queue protocol (all paths absolute, all writes atomic)::

        <queue_root>/pending/<job_id>.json     lane -> pool   job descriptor
        <queue_root>/claimed/<job_id>.json     pool -> lane   lease, for liveness
        <queue_root>/receipts/<job_id>.json    pool -> lane   completion receipt
        <queue_root>/signals/<job_id>.backoff.json   pool -> lane   typed pause
        <queue_root>/signals/<job_id>.failure.json   pool -> lane   typed failure

    The result itself is published to ``required_output_path``; the receipt is the
    commit point, so the lane never reads a half-written result. Every pool file
    must echo the job's ``attempt_nonce``: a late file from a superseded attempt
    is ignored, never mistaken for this attempt's answer.
    """

    def __init__(
        self,
        queue_root: Path,
        *,
        effort_grant: Optional[AuthorEffortGrant] = None,
        stall_timeout_seconds: float = float(AUTHOR_QUEUE_STALL_SECONDS),
        poll_interval_seconds: float = AUTHOR_QUEUE_POLL_SECONDS,
        clock: Optional[Callable[[], str]] = None,
        monotonic: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        nonce_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        """Bind the queue root, the effort grant, and the stall deadline.

        Parameters
        ----------
        queue_root:
            Campaign-local author-queue root serviced by the managing session.
        effort_grant:
            Per-session effort budget published to the pool and audited on return.
        stall_timeout_seconds:
            Outer deadline. Exceeding it is retryable infrastructure (risk R6),
            never a model failure.
        poll_interval_seconds:
            Result polling cadence.
        clock, monotonic, sleep, nonce_factory:
            Injectable time and identity sources for deterministic tests.
        """

        if stall_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("author queue timeouts must be positive")
        self.queue_root = Path(queue_root)
        self.effort_grant = effort_grant or AuthorEffortGrant()
        self.stall_timeout_seconds = float(stall_timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self._clock = clock or utc_now
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._nonce_factory = nonce_factory or (lambda: uuid.uuid4().hex)
        self.directories = author_queue.author_queue_directories(self.queue_root)

    def _dispatch(
        self,
        *,
        kind: str,
        item: WorkItem,
        config: Optional[DriverConfig],
        work_id: str,
        request_path: Path,
        output_path: Path,
    ) -> Optional[str]:
        """Enqueue one job and block until the operator publishes its result.

        Returns
        -------
        Optional[str]
            The pool receipt's ``author_model``. On this lane it is the pool's
            declared model, which :meth:`AuthorPool.subagent_model` refuses to
            dispatch against a contradicting frozen tier -- an enforced claim,
            not a spawn observation.
        """

        self.directories = author_queue.ensure_author_queue(self.queue_root)
        protocol_grant = author_queue.EffortGrant.from_mapping(self.effort_grant.to_dict())
        descriptor = author_queue.build_job_descriptor(
            queue_root=self.queue_root,
            kind=kind,
            stable_id=item.stable_id,
            work_id=work_id,
            request_path=request_path,
            required_output_path=output_path,
            effort_grant=protocol_grant,
            stall_timeout_seconds=self.stall_timeout_seconds,
            attempt_nonce=self._nonce_factory(),
            author_model=None if config is None else config.author_model,
            # Two distinct identities, never one. The repair scope is this item's bounded
            # authority lineage (``campaign-<stable_id>``); the tier campaign is the run's
            # frozen partitioner campaign, which is what selects the pool's author model
            # and standards prompt.
            repair_campaign_id=_campaign_id_for_item(item),
            tier_campaign_id=None if config is None else config.campaign_id,
            enqueued_at=self._clock(),
        )
        job = author_queue.QueueJob.from_mapping(descriptor)
        paths = author_queue.job_paths(self.queue_root, job.job_id)
        author_queue.write_json_atomic(paths["pending"], descriptor)
        try:
            receipt = self._await_result(job, paths)
        finally:
            self._discard_job_files(paths)
        return _receipt_author_model(receipt)

    def _await_result(
        self,
        job: author_queue.QueueJob,
        paths: Mapping[str, Path],
    ) -> Mapping[str, Any]:
        """Block until a nonce-matched result, pause, failure, or stall.

        Parameters
        ----------
        job:
            Published job descriptor.
        paths:
            Queue paths owned by the job.

        Returns
        -------
        Mapping[str, Any]
            The pool's audited completion receipt.

        Raises
        ------
        AuthorBackoffError
            When the pool reports a provider rate/quota pause.
        RetryableOperatorError
            When the pool reports a transient failure.
        AuthorQueueStalled
            When neither a result nor a signal arrives before the deadline.
        AuthorEffortCapExceeded
            When declared consumption exceeds the published grant.
        DriverIntegrationError
            When the pool violates the result-publication contract.
        """

        deadline = self._monotonic() + self.stall_timeout_seconds
        while True:
            backoff = self._matched_signal(paths["backoff"], job)
            if backoff is not None:
                raise AuthorBackoffError(_author_backoff_from_signal(backoff))
            failure = self._matched_signal(paths["failure"], job)
            if failure is not None:
                self._raise_operator_failure(job, failure)
            receipt = self._matched_signal(paths["receipt"], job)
            if receipt is not None:
                if not job.required_output_path.is_file():
                    raise DriverIntegrationError(
                        f"author queue job {job.job_id} reported completion without a "
                        f"result at {job.required_output_path}"
                    )
                self._audit_consumption(job, receipt)
                return receipt
            if self._monotonic() >= deadline:
                claimed = paths["claim"].is_file()
                self._record_queue_stall(job, claimed=claimed)
                raise AuthorQueueStalled(
                    f"author queue job {job.job_id} for {job.stable_id} produced no "
                    f"result within {self.stall_timeout_seconds:g}s "
                    f"({'claimed but unfinished' if claimed else 'never claimed'})"
                )
            self._sleep(self.poll_interval_seconds)

    def _record_queue_stall(self, job: author_queue.QueueJob, *, claimed: bool) -> None:
        """Persist one nonce-bound watchdog fact before retry cleanup.

        Parameters
        ----------
        job:
            Exact queue descriptor that crossed its deadline.
        claimed:
            Whether the managing session had claimed the job.
        """

        marker = {
            "format": "menagerie.crawler.author-queue-stall.v1",
            "job_id": job.job_id,
            "attempt_nonce": job.attempt_nonce,
            "stable_id": job.stable_id,
            "enqueued_at": job.enqueued_at,
            "detected_at": self._clock(),
            "stall_timeout_seconds": self.stall_timeout_seconds,
            "claimed": claimed,
        }
        path = self.queue_root / "watchdog" / f"{job.job_id}-{job.attempt_nonce}.stall.json"
        _write_json_atomic(path, marker)

    def _matched_signal(
        self,
        path: Path,
        job: author_queue.QueueJob,
    ) -> Optional[JsonObject]:
        """Read one pool-published file if it belongs to this attempt.

        Parameters
        ----------
        path:
            Candidate pool file.
        job:
            Published job descriptor carrying the attempt nonce.

        Returns
        -------
        dict[str, Any] | None
            Parsed payload, or ``None`` when absent, unreadable, or superseded.
        """

        value = author_queue.read_json(path)
        if value is None:
            return None
        if author_queue.matches_attempt(value, job):
            return value
        if value.get("job_id") != job.job_id:
            return None
        if value.get("attempt_nonce") != job.attempt_nonce:
            # A late file from a superseded attempt is never this attempt's answer.
            path.unlink(missing_ok=True)
        return None

    def _raise_operator_failure(
        self,
        job: author_queue.QueueJob,
        failure: Mapping[str, Any],
    ) -> None:
        """Raise the typed error declared by an operator failure signal.

        Parameters
        ----------
        job, failure:
            Published job descriptor and the pool's failure payload.

        Raises
        ------
        RetryableOperatorError
            When the pool declares the failure transient.
        DriverIntegrationError
            When the pool declares the failure permanent.
        """

        detail = str(failure.get("detail", ""))[:STDIO_TAIL_MAX_CHARS]
        reason = str(failure.get("reason", "unspecified"))
        retryable = failure.get("retryable")
        if not isinstance(retryable, bool):
            raise DriverIntegrationError(
                f"author queue failure for {job.stable_id} omits a boolean "
                f"retryable classification: {reason}"
            )
        message = f"author queue job {job.job_id} failed ({reason}): {detail}"
        if retryable:
            if reason == "research-tools-unavailable":
                raise ResearchToolsUnavailableError(job.stable_id, detail)
            if reason == "forge-rate-limited":
                raise ForgeRateLimitedError(job.stable_id, detail)
            raise RetryableOperatorError(message)
        raise DriverIntegrationError(message)

    def _audit_consumption(
        self,
        job: author_queue.QueueJob,
        receipt: Mapping[str, Any],
    ) -> None:
        """Reject a completion whose declared effort exceeds the published grant.

        The pool owns the tool-call and wall-clock timers because it is the only
        boundary that observes them (PLAN_RECONCILED section 3.4). The lane makes
        that enforcement checkable by requiring the pool to declare what it spent
        and refusing a receipt that spent more than it was granted.

        Parameters
        ----------
        job, receipt:
            Published job descriptor and the pool's completion receipt.

        Raises
        ------
        DriverIntegrationError
            When the receipt omits declared consumption.
        AuthorEffortCapExceeded
            When any declared dimension exceeds its grant.
        """

        consumption = receipt.get("consumption")
        if not isinstance(consumption, Mapping):
            raise DriverIntegrationError(
                f"author queue receipt for {job.stable_id} omits declared effort consumption"
            )
        grant = job.effort_grant
        limits = (
            ("tool_calls", float(grant.tool_calls)),
            ("fetch_targets", float(grant.fetch_targets)),
            ("wall_seconds", float(grant.wall_seconds)),
        )
        for metric, limit in limits:
            raw = consumption.get(metric)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw < 0:
                raise DriverIntegrationError(
                    f"author queue receipt for {job.stable_id} declares an invalid "
                    f"{metric} consumption"
                )
            if float(raw) > limit:
                raise AuthorEffortCapExceeded(
                    f"author session for {job.stable_id} consumed {metric} "
                    f"{float(raw):g}, exceeding its {limit:g} grant",
                    dimension=metric.replace("_", "-"),
                )

    def _discard_job_files(self, paths: Mapping[str, Path]) -> None:
        """Remove this attempt's queue files so no stale job is re-served.

        Parameters
        ----------
        paths:
            Queue paths owned by the job. The published result is never removed;
            only the queue's own control files are.
        """

        for name in ("pending", "claim", "receipt", "backoff", "failure"):
            try:
                paths[name].unlink(missing_ok=True)
            except OSError:  # noqa: PERF203 -- queue cleanup is best effort
                logging.getLogger(__name__).debug("author queue cleanup failed: %s", paths[name])


def _author_backoff_from_signal(payload: Mapping[str, Any]) -> AuthorBackoffSignal:
    """Parse one pool-published backoff sidecar into the typed pause signal.

    Parameters
    ----------
    payload:
        Pool backoff payload carrying provider, reset evidence, and an excerpt.

    Returns
    -------
    AuthorBackoffSignal
        Typed pause signal with a reset time. An unparseable or absent reset
        falls back to the scheduler's one-hour re-check rather than blocking.

    Raises
    ------
    DriverIntegrationError
        When the provider is outside the closed usage-limit vocabulary.
    """

    provider = str(payload.get("provider", "anthropic"))
    if provider not in author_queue.USAGE_LIMIT_PROVIDERS:
        raise DriverIntegrationError(
            f"author backoff names an unsupported usage-limit provider: {provider!r}"
        )
    excerpt = str(payload.get("response_excerpt", ""))[:1_500]
    raw_reason = str(payload.get("reason", AuthorPauseReason.QUOTA_EXHAUSTED.value))
    try:
        reason = AuthorPauseReason(raw_reason)
    except ValueError as exc:
        raise DriverIntegrationError(
            f"author backoff names an unsupported pause reason: {raw_reason!r}"
        ) from exc
    raw_reset = payload.get("reset_at")
    reset_at = (
        plausible_author_reset_at(raw_reset)
        if isinstance(raw_reset, str) and raw_reset.strip()
        else None
    )
    if reset_at is None:
        reset_at = parse_author_reset_at(excerpt)
    raw_retry = payload.get("retry_after_seconds")
    retry_after = (
        int(raw_retry)
        if isinstance(raw_retry, (int, float)) and not isinstance(raw_retry, bool)
        else None
    )
    return AuthorBackoffSignal(
        reason=reason,
        retry_after_seconds=retry_after,
        reset_at=reset_at,
        response_excerpt=excerpt,
        provider=provider,
    )


# The checker wrapper polices itself to the ``deadline_at`` this lane stamps into every
# envelope: it refuses to start an attempt past it and clamps each attempt to the time
# remaining. Above that ceiling it may still sleep its inter-attempt backoff and needs room
# to start an interpreter, validate the request, and publish atomically. The lane bound is
# therefore the published deadline plus exactly that declared slack, so the lane is the
# backstop that fires only once the wrapper has genuinely failed to police itself -- never
# the primary limit.
CHECKER_LANE_WALL_GRACE_SECONDS = 60.0
# The wrapper's own inter-attempt backoff sleeps (2**0 + ... + 2**(n-2)) are the one part of
# its budget that is not clamped by the deadline, so they are added on top of it. The sum is
# single-sourced with the protocol's own budget derivation rather than restated here, so the
# lane bound cannot drift away from the grant it is supposed to sit above.
CHECKER_LANE_BACKOFF_SLACK_SECONDS = float(OPERATOR_INTER_ATTEMPT_BACKOFF_SECONDS)


def _checker_wall_bound(
    envelope: Mapping[str, Any],
    *,
    grace_seconds: float = CHECKER_LANE_WALL_GRACE_SECONDS,
    now: Optional[datetime] = None,
) -> float:
    """Derive the lane's external kill bound from the named per-session wall.

    The checker analogue of the author lane's ``grant x kill factor`` (leg-12
    incident, 2026-08-21): the session's own self-policing budget is the
    NAMED ``CHECKER_SESSION_WALL_SECONDS`` -- never more than the remaining
    time to the deadline the envelope already publishes -- and the external
    kill fires at that budget times ``CHECKER_WALL_EXTERNAL_KILL_FACTOR``, so
    the wrapper watching its own clock always gets to land a typed outcome
    first and this bound catches only the sessions that stopped policing
    themselves.

    Parameters
    ----------
    envelope:
        Frozen checker envelope, whose ``deadline_at`` is the exact ceiling the wrapper
        was told to hold itself to.
    grace_seconds:
        Allowance above the wrapper's own ceiling for interpreter start, request
        validation, and atomic publication.
    now:
        Optional timezone-aware instant, for deterministic tests.

    Returns
    -------
    float
        Wall-clock ceiling, strictly greater than the wrapper's internal budget.
    """

    remaining = float(CHECKER_SESSION_WALL_SECONDS)
    declared = envelope.get("deadline_at")
    if isinstance(declared, str) and declared.strip():
        try:
            deadline: Optional[datetime] = datetime.fromisoformat(declared.replace("Z", "+00:00"))
        except ValueError:
            # A malformed deadline is a wrapper-contract problem the wrapper itself
            # rejects; the lane still needs a bound, so it falls back to the named
            # per-session wall.
            deadline = None
        if deadline is not None and deadline.tzinfo is not None:
            instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
            remaining = min(
                (deadline.astimezone(timezone.utc) - instant).total_seconds(),
                float(CHECKER_SESSION_WALL_SECONDS),
            )
    return (
        max(remaining, 0.0) * CHECKER_WALL_EXTERNAL_KILL_FACTOR
        + CHECKER_LANE_BACKOFF_SLACK_SECONDS
        + grace_seconds
    )


def _frozen_checker_weather_request(
    request_path: Path, candidate: Mapping[str, Any]
) -> Optional[JsonObject]:
    """Reuse exact checker request bytes during one structured weather storm.

    Parameters
    ----------
    request_path:
        Previously published checker request path.
    candidate:
        Freshly derived request for the current checker call.

    Returns
    -------
    dict[str, Any] | None
        Existing frozen request when its non-clock identity equals ``candidate``
        and its machine status proves provider weather, otherwise ``None``.
    """

    if _checker_provider_weather_status(request_path) is None:
        return None
    try:
        frozen = load_absolute_request(request_path)
    except (OSError, ValueError):
        return None

    def replay_identity(envelope: Mapping[str, Any]) -> str:
        """Hash request semantics while excluding only the reminted clock fields."""

        return stable_hash(
            {
                key: value
                for key, value in envelope.items()
                if key not in {"deadline_at", "envelope_sha256"}
            }
        )

    if replay_identity(frozen) != replay_identity(candidate):
        return None
    return dict(frozen)


class CheckerLegBudgetExceeded(RetryableOperatorError):
    """The whole leg's checker wall/session budget breaker fired (BL-10 g).

    Loud and campaign-scoped like the park-streak breaker: the campaign pauses
    resumably, and a resume restores this leg's durable spend. Deliberately
    exempted from the infrastructure retry ladder -- retrying cannot change the
    accounting the breaker derives from.
    """


def buy_confirmation_session(
    lane: Any,
    gate_kind: str,
    artifacts: Sequence[AuthorArtifact],
    work_root: Path,
    config: DriverConfig,
) -> CheckerOutcome:
    """Buy one blind K=2 confirmation session from a checker lane.

    A lane that seats a confirmation partner (``check_confirmation``) is asked
    for it; every other lane (test lanes, the pre-seat protocol) confirms
    through its ordinary per-kind method under its own identity -- the
    historical same-identity K=2. Which recorded pair is LAWFUL is decided by
    the conflict ledger's seat-table pairing rule, never by this call path.
    """

    confirm = getattr(lane, "check_confirmation", None)
    if callable(confirm):
        return confirm(gate_kind, artifacts, work_root, config)
    if gate_kind == GateKind.METADATA_BATCH.value:
        return lane.check_metadata(tuple(artifacts), work_root, config)
    if gate_kind == GateKind.FIDELITY.value:
        return lane.check_fidelity(artifacts[0], work_root, config)
    if gate_kind == GateKind.TERMINAL_DISPOSITION.value:
        return lane.check_terminal(artifacts[0], work_root, config)
    raise DriverIntegrationError(f"unknown checker gate kind {gate_kind!r}")


class CommandCheckerLane:
    """Checker lane that uses frozen envelopes and an argv-only executor."""

    def __init__(
        self, command: Sequence[str], *, wall_grace_seconds: float = CHECKER_LANE_WALL_GRACE_SECONDS
    ) -> None:
        """Store a non-shell Codex command prefix and its wall-bound grace."""

        if not command:
            raise ValueError("checker command cannot be empty")
        if wall_grace_seconds < 0:
            raise ValueError("checker wall grace cannot be negative")
        self.command = tuple(command)
        self.wall_grace_seconds = float(wall_grace_seconds)

    def adjudication_identity(
        self, gate_kind: str, config: DriverConfig
    ) -> CheckerAdjudicationIdentity:
        """Declare the exact checker identity a session for this kind buys.

        Reuse eligibility (``gate_reuse``) is gated on this declaration: the
        LANE owns per-gate-kind model selection, so only the lane can say
        which recorded checker identity a fresh session would reproduce. The
        values here are byte-for-byte the ones ``_build_envelope`` seals into
        every live request this lane dispatches.
        """

        return self._seat_identity(gate_kind, config, CheckerSeatRole.PRIMARY)

    def confirmation_identity(
        self, gate_kind: str, config: DriverConfig
    ) -> CheckerAdjudicationIdentity:
        """Declare the cross-model K=2 CONFIRMATION identity for this kind.

        Seat swap (2026-09-01): the confirmation seat is the OTHER provider
        (``gpt-5.6-sol`` under ``openai`` against the Fable primary), so the
        two blind sessions of a K=2 pair never share a model. The conflict
        ledger's pairing rule re-derives this from the seat table on every
        row; this declaration is what the lane BUYS, the rule is what the
        ledger ACCEPTS.
        """

        return self._seat_identity(gate_kind, config, CheckerSeatRole.CONFIRMATION)

    @staticmethod
    def _seat_identity(
        gate_kind: str, config: DriverConfig, role: CheckerSeatRole
    ) -> CheckerAdjudicationIdentity:
        """Return the exact identity one seat's session is sealed under."""

        seat = checker_seat(gate_kind, role)
        if role is CheckerSeatRole.PRIMARY and config.checker_model != seat.model:
            # The campaign's declared checker binding (the CampaignSpec
            # projection the CLI resolves) and the seated primary this lane
            # SEALS on every session must be one model: the accepted-identity
            # recompute binds config.checker_model, so a divergence makes every
            # award refuse at the seam as "stale identities" -- refuse it
            # HERE, before any session is bought, with both values named.
            raise DriverIntegrationError(
                f"campaign checker_model {config.checker_model!r} is not the seated "
                f"{gate_kind} primary {seat.model!r} (constants.CHECKER_SEATS); the "
                "CampaignSpec projection and the seat table move together"
            )
        return CheckerAdjudicationIdentity(
            # Derived from the seated model, byte-identical to the envelope
            # builder's derivation (`checker_provider_for_model`).
            provider=seat.provider,
            model=seat.model,
            version=config.checker_version,
            # BL-10 (e): the LEG PIN, not a live working-tree read -- the same
            # identity every envelope this lane seals carries, by construction.
            prompt_sha256=leg_checker_prompt_sha256(),
        )

    def check_confirmation(
        self,
        gate_kind: str,
        artifacts: Sequence[AuthorArtifact],
        work_root: Path,
        config: DriverConfig,
    ) -> CheckerOutcome:
        """Buy one blind K=2 confirmation session on the CONFIRMATION seat.

        Same envelope builders, same wrapper, same custody and wall accounting
        as a primary session; only the seated model (and therefore the
        provider and the adapter the wrapper selects) differs. Metadata
        confirmations are single-item batches over the accepted member.
        """

        role = CheckerSeatRole.CONFIRMATION
        if gate_kind == GateKind.METADATA_BATCH.value:
            return self.check_metadata(artifacts, work_root, config, role=role)
        if gate_kind == GateKind.FIDELITY.value:
            return self.check_fidelity(artifacts[0], work_root, config, role=role)
        if gate_kind == GateKind.TERMINAL_DISPOSITION.value:
            return self.check_terminal(artifacts[0], work_root, config, role=role)
        raise DriverIntegrationError(f"unknown checker gate kind {gate_kind!r}")

    def check_metadata(
        self,
        artifacts: Sequence[AuthorArtifact],
        work_root: Path,
        config: DriverConfig,
        *,
        role: CheckerSeatRole = CheckerSeatRole.PRIMARY,
    ) -> CheckerOutcome:
        """Execute one strict metadata checker batch on one seat."""

        items = [_checker_item(artifact) for artifact in artifacts]
        batch_id = stable_hash([item["work_id"] for item in items])[7:23]
        # A confirmation session gets its own request root: the primary's
        # frozen request stays byte-stable for weather replay, and the K=2
        # pair must be two distinct envelopes (P5 fable-F16).
        suffix = "" if role is CheckerSeatRole.PRIMARY else f"-{role.value}"
        root = work_root / "checker" / f"metadata-{batch_id}{suffix}"
        return self._run(
            build_metadata_vet_envelope(
                items,
                gate_round=1,
                output_path=root / "result.json",
                checker_model=checker_seat(GateKind.METADATA_BATCH, role).model,
                checker_version=config.checker_version,
                request_nonce=batch_id,
                final_tail=len(items) < METADATA_BATCH_MIN,
            ),
            root,
        )

    def check_fidelity(
        self,
        artifact: AuthorArtifact,
        work_root: Path,
        config: DriverConfig,
        *,
        role: CheckerSeatRole = CheckerSeatRole.PRIMARY,
    ) -> CheckerOutcome:
        """Execute one strict per-model fidelity checker request on one seat."""

        stable_id = str(artifact.proposal["stable_id"])
        suffix = "" if role is CheckerSeatRole.PRIMARY else f"-{role.value}"
        root = work_root / stable_id / f"checker-fidelity{suffix}"
        return self._run(
            build_fidelity_envelope(
                _checker_item(artifact),
                gate_round=1,
                output_path=root / "result.json",
                checker_model=checker_seat(GateKind.FIDELITY, role).model,
                checker_version=config.checker_version,
                request_nonce=f"fidelity-{stable_id}",
            ),
            root,
        )

    def check_terminal(
        self,
        artifact: AuthorArtifact,
        work_root: Path,
        config: DriverConfig,
        *,
        role: CheckerSeatRole = CheckerSeatRole.PRIMARY,
    ) -> CheckerOutcome:
        """Execute one strict typed terminal-disposition request on one seat."""

        result = artifact.author_result
        if isinstance(result, ProposedAuthorResult):
            raise DriverIntegrationError("proposed result cannot enter the terminal checker")
        stable_id = result.binding.stable_id
        suffix = "" if role is CheckerSeatRole.PRIMARY else f"-{role.value}"
        root = work_root / stable_id / f"checker-terminal{suffix}"
        item = _terminal_checker_item(artifact)
        return self._run(
            build_terminal_disposition_envelope(
                item,
                gate_round=1,
                output_path=root / "result.json",
                checker_model=checker_seat(GateKind.TERMINAL_DISPOSITION, role).model,
                checker_version=config.checker_version,
                request_nonce=f"terminal-{stable_id}",
            ),
            root,
        )

    def _run(self, envelope: JsonObject, root: Path) -> CheckerOutcome:
        """Write, execute, classify, and validate one checker envelope."""

        from menagerie.crawler.author_dispatch import write_envelope_atomic
        from menagerie.crawler.checker_dispatch import (
            classify_checker_response,
            validate_checker_result,
        )

        root.mkdir(parents=True, exist_ok=True)
        request_path = root / "request.json"
        frozen = _frozen_checker_weather_request(request_path, envelope)
        if frozen is None:
            request_path = write_envelope_atomic(envelope, request_path)
        else:
            envelope = frozen
        session_id = f"checker-session-{uuid.uuid4().hex}"
        session_started = time.monotonic()
        try:
            completed = _run_operator_command(
                [*self.command, str(request_path)],
                timeout_seconds=_checker_wall_bound(
                    envelope, grace_seconds=self.wall_grace_seconds
                ),
            )
        except subprocess.TimeoutExpired as exc:
            # Typed retryable, exactly like the author lane. The wrapper bounds its own
            # Codex attempts, but that does not protect this lane: a wrapper that hangs
            # before or outside that call would otherwise block the single-threaded
            # driver forever. A hang is stalled infrastructure, so
            # ``_is_infrastructure_error`` routes it to the bounded transport retry
            # rather than permanently burning the model. No checker VERDICT is affected:
            # a genuine rejection still arrives as a typed exit or a validated gate
            # result below, and is still a rejection.
            error = RetryableOperatorError(
                f"checker command exceeded its {exc.timeout:g}s wall bound and its process "
                f"group was terminated: "
                f"{str(exc.stderr or exc.output or '')[-STDIO_TAIL_MAX_CHARS:]}"
            )
            # The kill is this driver's own machine act at the named wall x kill
            # factor, so the settlement carries the closed refundable outcome:
            # sessions and machine wall stay charged, only the ATTEMPT projection
            # refunds (leg-12 incident, 2026-08-21).
            error.paid_session_local_outcome = "failed-checker-session-wall"
            error._paid_session_infrastructure_token = _PAID_SESSION_INFRASTRUCTURE_TOKEN
            raise error from exc
        elapsed_seconds = time.monotonic() - session_started
        checker_block = envelope.get("checker")
        seated_model = str(checker_block.get("model")) if isinstance(checker_block, Mapping) else ""
        signal = classify_checker_response(
            completed.returncode,
            completed.stderr + "\n" + completed.stdout,
            # The pause episode is scoped to the provider of the seat that
            # ran: a Fable session's limit pauses `anthropic`, a sol
            # confirmation's pauses `openai` (DX-67 coalesces both lanes).
            provider=checker_provider_for_model(seated_model),
        )
        if signal is not None:
            return CheckerOutcome(
                backoff=signal,
                session_id=session_id,
                wall_seconds=elapsed_seconds,
            )
        _raise_for_checker_exit(
            completed.returncode,
            completed.stdout,
            completed.stderr,
            request_path=request_path,
        )
        result = validate_checker_result(root / "result.json", envelope)
        return CheckerOutcome(
            gate=result,
            session_id=session_id,
            wall_seconds=elapsed_seconds,
        )


class CommandEnvironmentBackend:
    """Argv-only adapter for an exact-lock environment tooling wrapper."""

    def __init__(self, command: Sequence[str], *, tmp_root: Path | None = None) -> None:
        """Store the wrapper command and its explicit temporary write root."""

        if not command:
            raise ValueError("environment command cannot be empty")
        self.command = tuple(command)
        self._tmp_root = tmp_root.resolve() if tmp_root is not None else None

    def solve(self, environment_file: Path, target: str) -> SolveResult:
        """Request a solve and verify every lock digest from materialized bytes."""

        payload = self._json_action("solve", str(environment_file), target)
        lock_path = Path(str(payload.get("lock_path", "")))
        export_path = Path(str(payload.get("resolved_export_path", "")))
        if not lock_path.is_file() or not export_path.is_file():
            raise DriverIntegrationError(
                "environment solve wrapper must return lock_path and resolved_export_path"
            )
        lock_bytes = lock_path.read_bytes()
        export_bytes = export_path.read_bytes()
        try:
            lock_receipts = parse_exact_lock(lock_bytes)
            parse_resolved_export(export_bytes)
        except EnvironmentExactnessError as exc:
            raise DriverIntegrationError(str(exc)) from exc
        raw_artifacts = payload.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise DriverIntegrationError(
                "environment solve wrapper must return materialized artifact receipts"
            )
        receipts: list[ArtifactReceipt] = []
        for value in raw_artifacts:
            if not isinstance(value, Mapping):
                raise DriverIntegrationError("environment artifact receipt must be an object")
            url = value.get("url")
            path_value = value.get("path")
            declared = value.get("sha256")
            if not all(isinstance(item, str) and item for item in (url, path_value, declared)):
                raise DriverIntegrationError(
                    "environment artifact receipt requires url, path, and sha256"
                )
            artifact_path = Path(str(path_value))
            if not artifact_path.is_file():
                raise DriverIntegrationError(
                    f"environment artifact is not materialized: {artifact_path}"
                )
            observed = hash_bytes(artifact_path.read_bytes())
            if observed != declared:
                raise DriverIntegrationError(
                    f"environment artifact digest mismatch: {artifact_path}"
                )
            receipts.append(ArtifactReceipt(str(url), observed))
        if tuple(receipts) != lock_receipts:
            raise DriverIntegrationError(
                "materialized artifact receipts do not exactly match the solved lock"
            )
        return SolveResult(
            lock_bytes=lock_bytes,
            resolved_export_bytes=export_bytes,
            elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
            artifact_bytes=int(payload.get("artifact_bytes", 0)),
            artifact_receipts=tuple(receipts),
        )

    def create(self, lock_file: Path, prefix: Path) -> bytes:
        """Create one prefix and derive its inventory from installed metadata."""

        self._checked_action("create", str(lock_file), str(prefix))
        try:
            return installed_package_inventory_bytes(prefix)
        except EnvironmentExactnessError as exc:
            raise DriverIntegrationError(str(exc)) from exc

    def probe(self, prefix: Path, probes: IntentProbes) -> Sequence[ProbeResult]:
        """Run declared canaries and return typed per-probe observations."""

        payload = self._json_action(
            "probe",
            str(prefix),
            json.dumps(
                {
                    "imports": list(probes.imports),
                    "export_checks": [vars(check) for check in probes.export_checks],
                    "source_build": [vars(build) for build in probes.source_build],
                },
                sort_keys=True,
            ),
        )
        values = payload.get("results")
        if not isinstance(values, list):
            raise DriverIntegrationError("environment probe wrapper returned no results")
        results: list[ProbeResult] = []
        for value in values:
            if (
                not isinstance(value, Mapping)
                or not isinstance(value.get("name"), str)
                or not isinstance(value.get("passed"), bool)
                or not isinstance(value.get("detail"), str)
            ):
                raise DriverIntegrationError("environment probe receipt is malformed")
            results.append(
                ProbeResult(
                    name=str(value["name"]),
                    passed=bool(value["passed"]),
                    detail=str(value["detail"]),
                )
            )
        try:
            return validate_probe_receipts(probes, results)
        except EnvironmentProbeError as exc:
            raise DriverIntegrationError(str(exc)) from exc

    def overlay(self, prefix: Path, manifest_path: Path) -> Mapping[str, Any]:
        """Install one committed pip-overlay manifest through the wrapper.

        There is deliberately NO legacy degradation here, unlike
        :meth:`owned_paths`: an intent that declares an overlay cannot honestly
        run without it, so a wrapper that predates the ``overlay`` action fails
        CLOSED with its own typed error.
        """

        payload = self._json_action("overlay", str(prefix), str(manifest_path))
        packages = payload.get("packages")
        if not isinstance(packages, list) or not all(
            isinstance(row, Mapping)
            and isinstance(row.get("name"), str)
            and isinstance(row.get("version"), str)
            and isinstance(row.get("sha256"), str)
            for row in packages
        ):
            raise DriverIntegrationError(
                "environment overlay wrapper returned a malformed package receipt"
            )
        if not isinstance(payload.get("manifest_sha256"), str):
            raise DriverIntegrationError("environment overlay wrapper returned no manifest digest")
        return payload

    def remove(self, prefix: Path) -> None:
        """Remove only the named environment and its dedicated state."""

        self._checked_action("remove", str(prefix))

    #: Positive identification of a wrapper that PREDATES the ``owned-paths``
    #: action: argparse rejects the unknown subcommand with exactly this
    #: phrase on stderr, which :meth:`_checked_action` folds into its error.
    #: Only this identified legacy protocol may degrade to the prefix-only
    #: inventory; every other enumeration failure fails CLOSED (Sol round-2
    #: review finding 3: a generic failure cannot prove that no auxiliary
    #: generation-owned roots exist).
    _LEGACY_OWNED_PATHS_SIGNATURE = "invalid choice: 'owned-paths'"

    def owned_paths(self, prefix: Path) -> tuple[Path, ...]:
        """Enumerate every generation-owned root for the lifecycle residue check.

        The wrapper's receipt names the prefix, its durable registration, and
        the registered generation-local package cache, so a falsely successful
        removal that leaves dedicated state behind is caught as residue rather
        than misread as ambient disk consumption (rung-11 review finding 4).

        Degradation to the prefix-only inventory happens ONLY for a positively
        identified legacy wrapper (the argparse unknown-action signature for
        ``owned-paths``), and loudly. A CURRENT wrapper that crashes, exits
        nonzero, or returns malformed output fails CLOSED with a typed error:
        an unexplained enumeration failure cannot prove that no auxiliary
        roots exist, so the lifecycle must not proceed to a residue check it
        cannot complete (Sol round-2 review finding 3).

        Parameters
        ----------
        prefix:
            Materialized environment prefix.

        Returns
        -------
        tuple[pathlib.Path, ...]
            Generation-owned filesystem roots, always including the prefix.

        Raises
        ------
        DriverIntegrationError
            If the enumeration fails for any reason other than the positively
            identified legacy wrapper protocol.
        """

        try:
            payload = self._json_action("owned-paths", str(prefix))
        except DriverIntegrationError as exc:
            if self._LEGACY_OWNED_PATHS_SIGNATURE in str(exc):
                LOGGER.warning(
                    "environment wrapper predates the owned-paths action for %s; "
                    "residue checking degrades to the prefix alone: %s",
                    prefix,
                    exc,
                )
                return (prefix,)
            raise DriverIntegrationError(
                f"environment owned-paths enumeration failed for {prefix}; a "
                "generic wrapper failure cannot prove that no generation-owned "
                f"auxiliary roots exist, so the teardown residue check refuses "
                f"to proceed: {exc}"
            ) from exc
        values = payload.get("paths")
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise DriverIntegrationError(
                "environment owned-paths wrapper returned a malformed inventory"
            )
        paths = tuple(Path(value) for value in values)
        return paths if prefix in paths else (prefix, *paths)

    def _json_action(self, action: str, *arguments: str) -> JsonObject:
        """Run one wrapper action and parse its stdout JSON object."""

        completed = self._checked_action(action, *arguments)
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise DriverIntegrationError(
                f"environment {action} returned invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DriverIntegrationError(f"environment {action} must return a JSON object")
        return value

    #: Wrapper-side ENOSPC surfaces as exit 64 with the errno detail in the
    #: typed stderr payload. It is a DISK condition, not a permanent contract
    #: rejection: re-raising it as ``OSError(ENOSPC)`` lets the lifecycle's
    #: floor conversion park it behind the disk-floor valve.
    _ENOSPC_SIGNATURES = ("No space left on device", "[Errno 28]", "ENOSPC")

    #: BL-9e: parent-owned deadline per wrapper action, in seconds. The
    #: wrapper bounds each of ITS conda subprocesses, but the wrapper process
    #: itself was unbounded -- a wedge on its own timeout-free registration
    #: flock or an unmounted-store rmtree hung the untimed ``subprocess.run``
    #: below forever, pinning the driver writer lock (and with it every
    #: operator drain tool) and the leg's completion partition. Each deadline
    #: deliberately exceeds the wrapper's own inner timeout for the action
    #: (solve/create 30m, overlay 20m, probe 5m, remove unbounded inside), so
    #: the parent only fires when the wrapper itself is stuck, never as a
    #: second, tighter clock racing a healthy action.
    _ACTION_DEADLINE_SECONDS: Mapping[str, float] = {
        "solve": 45 * 60.0,
        "create": 45 * 60.0,
        "overlay": 30 * 60.0,
        "probe": 10 * 60.0,
        "remove": 20 * 60.0,
        "owned-paths": 5 * 60.0,
    }

    #: Fallback deadline for any wrapper action the table does not name.
    _DEFAULT_ACTION_DEADLINE_SECONDS = 15 * 60.0

    #: BL-8 (e): bound on the post-kill reap after a deadline fires. A wrapper
    #: in uninterruptible D-state ignores SIGKILL; without this bound the
    #: expiry path itself blocked forever on ``process.wait()`` and the
    #: deadline bought nothing (the worker lane solved this identically:
    #: bounded wait -> group kill -> bounded wait -> typed host fault).
    _POST_KILL_REAP_SECONDS = 30.0

    def _run_wrapper_with_parent_deadline(
        self,
        argv: Sequence[str],
        *,
        action: str,
        deadline: float,
        environment: Optional[Mapping[str, str]],
    ) -> subprocess.CompletedProcess[str]:
        """Run one wrapper action under the parent-owned deadline, group-killed.

        BL-8 (e), layered on BL-9e's deadline table: plain ``subprocess.run``
        (i) killed only the wrapper root, so its conda/pip grandchildren kept
        mutating the shared store after the park, and (ii) waited UNBOUNDED
        for the corpse, so a D-state wrapper still pinned the writer thread,
        every ledger flock, and the completion partition forever. The wrapper
        runs in its own session; expiry kills the whole process group and the
        reap itself is bounded -- an unreapable wrapper raises the same typed
        host condition instead of hanging the expiry path.

        Raises
        ------
        EnvironmentBackendUnavailableError
            When the deadline fires (killed or unreapable wrapper).
        """

        process = subprocess.Popen(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(environment) if environment is not None else None,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=deadline)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
            try:
                process.communicate(timeout=self._POST_KILL_REAP_SECONDS)
            except subprocess.TimeoutExpired:
                raise EnvironmentBackendUnavailableError(
                    f"environment {action} exceeded its parent-owned {deadline:g}s "
                    "deadline and its process group could not be reaped within "
                    f"{self._POST_KILL_REAP_SECONDS:g}s (uninterruptible host state); "
                    "treating as retryable infrastructure"
                ) from exc
            raise EnvironmentBackendUnavailableError(
                f"environment {action} exceeded its parent-owned {deadline:g}s "
                f"deadline and its process group was killed; treating as retryable "
                f"infrastructure: {exc}"
            ) from exc
        return subprocess.CompletedProcess(list(argv), process.returncode, stdout, stderr)

    def _checked_action(self, action: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run one non-shell wrapper action, preserving the operator exit typing.

        The operator protocol's sysexits split MUST survive this boundary
        (T3-T2 facet a: folding every nonzero exit into one integration error
        collapsed the typed transient/permanent distinction, and the old
        catch-all then terminalized transient outages):

        * exit 75 (retryable infrastructure) ->
          :class:`EnvironmentBackendUnavailableError` -- a host condition the
          driver parks;
        * exit 64 (permanent rejection) with an ENOSPC detail ->
          ``OSError(ENOSPC)`` for the lifecycle's disk-floor conversion;
        * exit 64 otherwise -> :class:`EnvironmentBackendContractError` -- a
          setup fact the driver parks until the setup artifacts change;
        * any other nonzero exit -> :class:`DriverIntegrationError` -- a loud
          harness refusal, exactly as before.
        """

        environment = None
        if self._tmp_root is not None:
            self._tmp_root.mkdir(parents=True, exist_ok=True)
            environment = {**os.environ, "TMPDIR": str(self._tmp_root)}
        deadline = self._ACTION_DEADLINE_SECONDS.get(
            action, self._DEFAULT_ACTION_DEADLINE_SECONDS
        )
        # BL-9e: the parent-owned deadline. A HOST condition RIGHT NOW (the
        # wrapper wedged past every inner timeout) routes exactly like exit
        # 75 -- the driver parks host-condition (which never re-claims)
        # instead of terminalizing models or holding the writer lock and
        # completion partition forever. BL-8 (e) hardened the expiry: group
        # kill plus a bounded reap, so the deadline holds even for a wedged
        # process tree or a D-state wrapper.
        completed = self._run_wrapper_with_parent_deadline(
            [*self.command, action, *arguments],
            action=action,
            deadline=deadline,
            environment=environment,
        )
        if completed.returncode == 0:
            return completed
        detail = completed.stderr[-1500:]
        if completed.returncode == OperatorExitCode.RETRYABLE_INFRASTRUCTURE:
            raise EnvironmentBackendUnavailableError(
                f"environment {action} reported retryable infrastructure (exit 75): {detail}"
            )
        if completed.returncode == OperatorExitCode.PERMANENT_CONTRACT_REJECTION:
            if any(signature in detail for signature in self._ENOSPC_SIGNATURES):
                raise OSError(
                    errno.ENOSPC,
                    f"environment {action} failed with ENOSPC in the wrapper: {detail}",
                )
            raise EnvironmentBackendContractError(
                f"environment {action} was permanently rejected (exit 64): {detail}"
            )
        raise DriverIntegrationError(f"environment {action} failed: {detail}")

    def write_destinations(self) -> tuple[Path, ...]:
        """Enumerate the operator-owned write roots for the disk-floor probes.

        Mirrors the wrapper's own state-root resolution (explicit
        ``MENAGERIE_ENVIRONMENT_STATE_ROOT``, else the repository-local
        ``.crawl-local/environment-operator`` default): the artifact CAS,
        package caches, and wheelhouses all live below that root, which is
        exactly the filesystem the T3-T3 floor never watched. The temporary
        directory is probed by the lifecycle itself.
        """

        configured = os.environ.get("MENAGERIE_ENVIRONMENT_STATE_ROOT")
        if configured and Path(configured).is_absolute():
            return (Path(configured),)
        return (Path.cwd() / ".crawl-local" / "environment-operator",)


def build_command_environment_lane(
    command: Sequence[str],
    runtime_root: Path,
    *,
    tmp_root: Path | None = None,
    records_root: Path | None = None,
) -> SequentialEnvironmentLifecycle:
    """Build the production lifecycle with one explicit disk-space authority.

    BL-9b: with ``records_root`` provided, environment-turn effort admission
    is mirrored into the durable ``operational/environment-turn-effort.jsonl``
    ledger and rederived by replay on resume, so a supervisor/wake restart can
    no longer reset the in-memory attempt count and re-permit full-prefix
    env-build churn.
    """

    if records_root is not None:
        # D-60: heal a crash-torn effort-ledger tail at lane build -- the one
        # window it can exist (the ledger's sole writer is this DriverLock-
        # excluded driver) and BEFORE the run's long-lived operational-ledger
        # handle would contend with the disclosure append.
        recover_environment_turn_effort_ledger(records_root)
    return SequentialEnvironmentLifecycle(
        CommandEnvironmentBackend(command, tmp_root=tmp_root),
        PerIntentGenerationEffort(
            StageCap(
                attempts=2,
                seconds=30 * 60,
                bytes=100 * 1024**3,
            ),
            ledger_path=(
                None if records_root is None else environment_turn_effort_path(records_root)
            ),
        ),
        env_root=runtime_root / "envs",
        tmp_root=tmp_root,
    )


@dataclass(frozen=True)
class _AuthorDispatchOutcome:
    """Exactly one completed author-lane round trip, success or typed failure.

    The dispatch phase never interprets an outcome; it only carries it back to the
    single canonical writer, which re-raises the original exception inside the
    unchanged routing block so every typed author outcome routes exactly as it does
    under a serial lane.
    """

    artifact: Optional[AuthorArtifact] = None
    error: Optional[BaseException] = None

    def unwrap(self) -> AuthorArtifact:
        """Return the artifact, or re-raise the lane's own exception in place.

        Returns
        -------
        AuthorArtifact
            Exact artifact the lane produced.

        Raises
        ------
        BaseException
            The exception the lane raised, unchanged in type and payload.
        """

        if self.error is not None:
            raise self.error
        if self.artifact is None:
            raise DriverIntegrationError("author dispatch outcome carries neither result nor error")
        return self.artifact


class _AuthorAdmissionRequest:
    """One worker request for a writer-committed author-session buy."""

    def __init__(
        self,
        item: WorkItem,
        attempt_group: Optional[str] = None,
        *,
        enqueued_at: Optional[float] = None,
    ) -> None:
        """Bind the work item, attempt group, and one-shot completion signal."""

        self.item = item
        self.attempt_group = attempt_group
        self.completed = threading.Event()
        self.reservation: Optional[Reservation] = None
        self.error: Optional[BaseException] = None
        #: Monotonic enqueue instant, read only for the starvation stderr note.
        #: The owning pool stamps it from ITS clock so waits are one-clock math.
        self.enqueued_at = time.monotonic() if enqueued_at is None else enqueued_at


class _AuthorWavePool:
    """Bounded concurrent author-session dispatcher for exactly one wave.

    The author lane is the campaign's throughput ceiling: it is roughly three
    quarters of all projected work and every session is a multi-minute agent round
    trip. Running one session at a time caps a four-campaign fleet at four
    concurrent sessions, which is below what the reconciled schedule needs before
    anything else goes wrong.

    Only the *sessions* overlap. This object owns the "dispatch and wait" phase and
    nothing else: it calls the injected author lane, captures whatever that call
    produced, and hands it back untouched. It never appends to a ledger, never
    publishes an artifact, and never routes a terminal, so the single canonical
    writer is entirely unaffected -- the driver still commits every model itself,
    one at a time, in work order.

    Two properties make that safe:

    * **Exact eligibility, never speculation.** A model is only dispatched when the
      serial path provably reaches a fresh author session for it. Nothing is
      started that the serial lane would have resolved from canonical authority, a
      family representative, or the disposable on-disk cache.
    * **Started sessions are never cancelled; unstarted admissions are.** A
      session that has started has already spent provider budget, so an aborting
      wave drains its started siblings instead of orphaning them, and the driver
      preserves their validated results for the next run. BL-8 (d): admission is
      WINDOWED -- at most ``degree`` running plus ``degree`` of lookahead is ever
      handed to the executor; the rest of the cohort is held back here, so an
      early usage pause executes no more sessions than the declared window and
      cancels everything held or queued instead of buying it.
    """

    def __init__(
        self,
        degree: int,
        dispatch: Callable[
            [WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact
        ],
        admit: Callable[[WorkItem, Optional[str]], Reservation],
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the parallelism bound and the per-model dispatch callable.

        Parameters
        ----------
        degree:
            Maximum author sessions in flight at once. ``1`` disables the pool
            entirely so the driver keeps its historical fully inline lane.
        dispatch:
            Per-model author-lane invocation. Its callback requests each exact
            paid-session admission from the canonical writer before dispatch,
            naming the attempt group to join (``None`` heads a new attempt).
        admit:
            Writer-owned durable allowance check and money-row append.
        monotonic:
            Injectable monotonic clock for the pool's wait/pacing measurements.
        """

        self._dispatch = dispatch
        self._admit = admit
        self._monotonic = monotonic
        self._executor = (
            ThreadPoolExecutor(max_workers=degree, thread_name_prefix="author-wave")
            if degree > 1
            else None
        )
        self._futures: dict[str, "Future[_AuthorDispatchOutcome]"] = {}
        self._submitted: list[str] = []
        self._claimed: set[str] = set()
        #: BL-8 (d): the declared admission window -- ``degree`` running plus one
        #: wave (``degree``) of lookahead. ``submit()`` past it holds the item in
        #: ``_held`` instead of queueing it in the executor, which is what makes
        #: post-pause spend bounded by construction rather than by racing cancel.
        self._window = 2 * max(1, degree)
        self._admission_lock = threading.RLock()
        self._open_admissions = 0
        self._held: list[WorkItem] = []
        self._admission_closed = False
        self._cancelled_ids: set[str] = set()
        self._admission_requests: deque[_AuthorAdmissionRequest] = deque()
        self._teardown_controller = _CampaignTeardownController()
        #: The pool is constructed by the canonical writer; only that thread may
        #: ever commit a reservation, so the public drain pins its identity.
        self._writer_ident = threading.get_ident()
        self._relay_wait_noted_at: Optional[float] = None

    @property
    def enabled(self) -> bool:
        """Return whether this wave dispatches author sessions concurrently."""

        return self._executor is not None

    def submit(self, item: WorkItem) -> None:
        """Admit one author session, or hold it once the window is full."""

        if self._executor is None:
            return
        with self._admission_lock:
            if item.stable_id in self._futures or self._admission_closed:
                return
            if any(held.stable_id == item.stable_id for held in self._held):
                return
            if self._open_admissions >= self._window:
                self._held.append(item)
                return
            self._admit_locked(item)

    def _admit_locked(self, item: WorkItem) -> None:
        """Hand one held or fresh item to the executor (admission lock held)."""

        assert self._executor is not None
        self._open_admissions += 1
        future = self._executor.submit(self._run, item)
        self._futures[item.stable_id] = future
        self._submitted.append(item.stable_id)
        future.add_done_callback(self._feed_from_window)

    def _feed_from_window(self, _future: "Future[_AuthorDispatchOutcome]") -> None:
        """Release the next held item into the freed window slot."""

        with self._admission_lock:
            self._open_admissions -= 1
            if self._admission_closed:
                return
            while self._held and self._open_admissions < self._window:
                self._admit_locked(self._held.pop(0))

    def claim(self, item: WorkItem) -> Optional[_AuthorDispatchOutcome]:
        """Return one model's completed outcome, or ``None`` when never dispatched.

        Blocks until this model's own session finishes. Siblings keep running, so
        the commit order is always work order regardless of completion order. A
        model still held behind the admission window is admitted immediately: the
        writer needs it now, and it blocks anyway.
        """

        if self._executor is None:
            return None
        with self._admission_lock:
            if item.stable_id not in self._futures and not self._admission_closed:
                for index, held in enumerate(self._held):
                    if held.stable_id == item.stable_id:
                        self._admit_locked(self._held.pop(index))
                        break
            future = self._futures.get(item.stable_id)
        if future is None:
            return None
        self._claimed.add(item.stable_id)
        return self._wait_for_future(future)

    def _request_admission(
        self, item: WorkItem, attempt_group: Optional[str] = None
    ) -> Reservation:
        """Block one worker until the writer durably commits its exact buy."""

        request = _AuthorAdmissionRequest(item, attempt_group, enqueued_at=self._monotonic())
        with self._admission_lock:
            # W1 (PSLRETRY round 5): a sealed window refuses the buy before it
            # is ever queued -- the relay must never hand the writer a paid
            # admission the wave has already decided to stop spending on.
            if self._admission_closed:
                raise self._sealed_admission_refusal()
            self._admission_requests.append(request)
        request.completed.wait()
        if request.error is not None:
            raise request.error
        if request.reservation is None:
            raise DriverIntegrationError("author admission completed without a reservation")
        return request.reservation

    def _sealed_admission_refusal(self) -> AuthorUsagePause:
        """Return the typed refusal for a paid admission after the window sealed.

        BL-8 (d) extended to the worker-to-writer relay (W1): ``cancel_pending()``
        and ``close()`` are decisions to STOP SPENDING, so a sealed window
        refuses every fresh guard-session buy with the lane's typed pause
        outcome -- never an untyped hang and never a durable money row.
        """

        return AuthorUsagePause(
            "author wave admissions are sealed; a recorded pause or shutdown "
            "already stopped paid session admissions for this wave"
        )

    def _refuse_queued_admissions_locked(self) -> None:
        """Refuse every queued, unserviced admission request (lock held)."""

        while self._admission_requests:
            request = self._admission_requests.popleft()
            request.error = self._sealed_admission_refusal()
            request.completed.set()

    def service_admissions(self) -> None:
        """Drain the queued admissions at one bounded writer-owned point.

        This is the same commit path the claim-wait servicing uses; it exists so
        the writer's OTHER long stretches (inline operator-command outcome
        polls, retry backoffs, per-item commit boundaries) stop starving queued
        session starts. It drains only what is queued right now and never
        blocks for more. The reserve site must stay on the single canonical
        writer: the pool is built on that thread, and any other caller is a
        driver bug refused before it can commit a money row off-writer.
        """

        if threading.get_ident() != self._writer_ident:
            raise DriverIntegrationError(
                "author admission relay must be serviced on the canonical writer thread"
            )
        self._service_admission_requests()

    def _note_relay_wait(self, stable_id: str, waited_seconds: float) -> None:
        """Log one rate-limited line when a session start starved on the relay.

        The pilot's starvation diagnoses were all ledger archaeology; like the
        env-turn deferral note, one stderr line per minute makes a starving
        relay directly observable in the leg log without touching any ledger.

        Best-effort by contract: this runs between a request's ``popleft()``
        and the ``try/finally`` that guarantees its completion signal, so a
        failed stderr write (ENOSPC on a redirected leg log, a broken pipe, a
        closed stream) must never escape -- it would strand the popped worker
        on an unbounded wait, the exact untyped hang W1 forbids.
        """

        if waited_seconds < 30.0:
            return
        now = self._monotonic()
        if self._relay_wait_noted_at is not None and now - self._relay_wait_noted_at < 60.0:
            return
        self._relay_wait_noted_at = now
        try:
            print(
                f"author-admission relay: buy for {stable_id} waited "
                f"{waited_seconds:.0f}s for writer service",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ValueError):
            # A diagnostics line may be lost; a session start may not be.
            pass

    def _service_admission_requests(self) -> None:
        """Commit every queued worker admission on the canonical writer thread."""

        while True:
            with self._admission_lock:
                if not self._admission_requests:
                    return
                request = self._admission_requests.popleft()
                closed = self._admission_closed
            # The completion guarantee begins the instant the request leaves
            # the queue: NOTHING between popleft() and completed.set() may run
            # outside this try/finally. Sol review finding 1 (2026-08-21): the
            # starvation diagnostic ran before the guard with a whitelist of
            # anticipated stderr exception types, so any other raise stranded
            # a dequeued worker forever -- observability must never be
            # load-bearing.
            try:
                try:
                    self._note_relay_wait(
                        request.item.stable_id, self._monotonic() - request.enqueued_at
                    )
                except BaseException:  # noqa: BLE001 -- diagnostics are best-effort
                    pass
                if closed:
                    # W1: a request that raced into the queue around the seal is
                    # refused typed at service time, never committed.
                    request.error = self._sealed_admission_refusal()
                    continue
                request.reservation = self._admit(request.item, request.attempt_group)
            except BaseException as exc:  # noqa: BLE001 -- returned to the exact worker
                request.error = exc
            finally:
                request.completed.set()

    def _wait_for_future(
        self, future: "Future[_AuthorDispatchOutcome]"
    ) -> _AuthorDispatchOutcome:
        """Wait for one worker while servicing its durable admission barrier."""

        while True:
            self._service_admission_requests()
            try:
                return future.result(timeout=0.01)
            except FutureTimeoutError:
                continue

    def cancel_pending(self) -> tuple[str, ...]:
        """Cancel every session that has not started; return the cancelled ids.

        BL-8 (d): a usage pause is a decision to STOP SPENDING, so the wave must
        not convert it into continued spend. Held items never reached the
        executor and are simply dropped; queued futures are cancelled (a future
        that already started keeps running and is drained normally). Admission
        closes permanently, so late window feeds admit nothing more. Cancelled
        members hold no cache and no record -- the next leg authors them fresh.
        """

        with self._admission_lock:
            self._admission_closed = True
            # W1: workers already blocked on the relay must not hang on a
            # window that will never service them, nor buy after the seal.
            self._refuse_queued_admissions_locked()
            cancelled = [item.stable_id for item in self._held]
            self._held.clear()
            for stable_id in self._submitted:
                if stable_id in self._claimed or stable_id in self._cancelled_ids:
                    continue
                if self._futures[stable_id].cancel():
                    self._cancelled_ids.add(stable_id)
                    cancelled.append(stable_id)
            return tuple(cancelled)

    def halt(self, *, trigger: str, reason: str) -> tuple[str, ...]:
        """Seal admissions and request teardown of every in-flight guard.

        Parameters
        ----------
        trigger:
            Stable campaign halt boundary.
        reason:
            Machine-owned bounded explanation written before each signal.

        Returns
        -------
        tuple[str, ...]
            Unstarted model ids cancelled by the halt.
        """

        cancelled = self.cancel_pending()
        self._teardown_controller.request(trigger=trigger, reason=reason)
        return cancelled

    def drain(self) -> tuple[tuple[str, _AuthorDispatchOutcome], ...]:
        """Wait for every unclaimed started session; return outcomes in submit order.

        Returns
        -------
        tuple[tuple[str, _AuthorDispatchOutcome], ...]
            Unclaimed ``(stable_id, outcome)`` pairs, deterministically ordered by
            submission (work) order rather than completion order. Cancelled
            admissions never ran, so they carry no outcome and are excluded.
        """

        drained: list[tuple[str, _AuthorDispatchOutcome]] = []
        for stable_id in self._submitted:
            if stable_id in self._claimed or stable_id in self._cancelled_ids:
                continue
            future = self._futures[stable_id]
            self._claimed.add(stable_id)
            try:
                drained.append((stable_id, self._wait_for_future(future)))
            except BaseException as exc:  # noqa: BLE001 -- a drain must never mask the abort
                drained.append((stable_id, _AuthorDispatchOutcome(error=exc)))
        return tuple(drained)

    def close(self) -> None:
        """Release the worker threads after every session has finished.

        Admission closes first: a completion callback firing during (or after)
        shutdown must find the window sealed, never feed a held item into the
        shut-down executor.
        """

        if self._executor is not None:
            with self._admission_lock:
                self._admission_closed = True
                self._refuse_queued_admissions_locked()
            self._executor.shutdown(wait=True)

    def _run(self, item: WorkItem) -> _AuthorDispatchOutcome:
        """Run one author session on a worker thread and capture its outcome.

        Every exception is captured rather than propagated so a failing, pausing,
        or timing-out sibling can never cancel the wave from inside a worker. The
        driver re-raises it on the writer thread, in work order, inside the exact
        routing block that owned it before.
        """

        try:
            # Zero-arg calls head a NEW attempt (the pre-attempt-group
            # contract); mid-attempt guard sessions pass their group. Kept as a
            # visible lambda call for the admission-boundary census.
            request_admission = cast(
                "Callable[[Optional[str]], Reservation]",
                lambda attempt_group=None: self._request_admission(item, attempt_group),
            )
            with _campaign_teardown_controller(self._teardown_controller):
                return _AuthorDispatchOutcome(artifact=self._dispatch(item, request_admission))
        except BaseException as exc:  # noqa: BLE001 -- routed verbatim by the writer
            return _AuthorDispatchOutcome(error=exc)


@dataclass(frozen=True)
class _CheckerDispatchOutcome:
    """Exactly one completed checker-lane round trip, success or captured failure.

    The pool never interprets an outcome; it carries it back to the single
    canonical writer, which replays it inside the unchanged retry/routing block
    so every typed checker outcome routes exactly as it does under a serial lane.
    """

    outcome: Optional[CheckerOutcome] = None
    error: Optional[BaseException] = None
    reservation: Optional[Reservation] = None
    elapsed_seconds: float = 0.0

    def unwrap(self) -> CheckerOutcome:
        """Return the outcome, or re-raise the lane's own exception in place.

        Returns
        -------
        CheckerOutcome
            Exact outcome the lane produced.

        Raises
        ------
        BaseException
            The exception the lane raised, unchanged in type and payload.
        """

        if self.error is not None:
            raise self.error
        if self.outcome is None:
            raise DriverIntegrationError(
                "checker dispatch outcome carries neither result nor error"
            )
        return self.outcome


@dataclass(frozen=True)
class _CheckerSpendScope:
    """Durable work generation and checker identity for one paid session."""

    gate_kind: str
    work_generations: tuple[tuple[str, str], ...]
    checker_identity: Optional[CheckerAdjudicationIdentity]


class _CheckerAdmissionRequest:
    """One worker request for a writer-committed checker-session buy."""

    def __init__(self, scope: _CheckerSpendScope) -> None:
        """Bind the durable scope and create the one-shot completion signal."""

        self.scope = scope
        self.completed = threading.Event()
        self.reservation: Optional[Reservation] = None
        self.error: Optional[BaseException] = None


class _CheckerSessionPool:
    """Bounded concurrent checker-session dispatcher for one gate round.

    The checker-lane sibling of :class:`_AuthorWavePool` (composition of the
    same contract, not a shared base class): the lane was ~15 hours of wall per
    pilot leg dispatched strictly one session at a time. Only the *sessions*
    overlap. This object owns "spawn, wait, capture" and nothing else -- it
    never touches a ledger, never routes a gate, and never parks a model, so
    the single canonical writer still commits every gate itself, in ladder
    order.

    The author pool's two safety properties carry over exactly:

    * **Exact eligibility, never speculation.** A batch is submitted only when
      the writer has already ruled out every sessionless resolution the serial
      lane would try first -- a persisted acceptable gate and a reusable
      recorded adjudication (``gate_reuse``). Work keys within one round are
      disjoint item sets, so a sibling's commit can never make a submitted
      session redundant.
    * **Started sessions are never cancelled; unstarted admissions are.** A
      session that has started has already spent provider budget, so an
      aborting round drains its started siblings instead of orphaning them. A
      drained checker result is not committed (its gate was never normalized
      or routed); its envelope and result bytes remain on disk under the batch
      work directory for the next run. BL-8 (d): admission is WINDOWED exactly
      like the author pool's -- at most ``degree`` running plus ``degree`` of
      lookahead is ever handed to the executor -- so an early usage pause
      executes no more sessions than the declared window and cancels
      everything held or queued instead of buying it.

    Keys are caller-chosen strings (batch member IDs, ``fidelity:<stable_id>``)
    rather than work items, because the checker lane's dispatch unit is a batch.
    ``claim`` pops, so a key legitimately re-asked later in the same round (a
    floor-one re-ask) dispatches a fresh session instead of replaying a consumed
    one.
    """

    def __init__(
        self,
        degree: int,
        admit: Callable[[_CheckerSpendScope], Reservation],
        settle_discarded: Optional[Callable[[Reservation, float, bool], None]] = None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind the parallelism bound.

        Parameters
        ----------
        degree:
            Maximum checker sessions in flight at once. ``1`` disables the pool
            entirely so the driver keeps its historical fully serial lane.
        admit:
            Writer-owned durable buy append, completed before dispatch.
        settle_discarded:
            Writer-owned settlement for a started sibling drained during an
            aborting round. ``None`` is reserved for isolated pool unit tests.
        monotonic:
            Injectable monotonic clock measuring every paid session span this
            pool owns.
        """

        self._admit = admit
        self._settle_discarded = settle_discarded
        self._monotonic = monotonic
        self._executor = (
            ThreadPoolExecutor(max_workers=degree, thread_name_prefix="checker-wave")
            if degree > 1
            else None
        )
        self._futures: dict[str, "Future[_CheckerDispatchOutcome]"] = {}
        #: BL-8 (d): declared admission window (running + one wave of lookahead).
        self._window = 2 * max(1, degree)
        self._admission_lock = threading.RLock()
        self._open_admissions = 0
        self._held: list[
            tuple[str, Callable[[], CheckerOutcome], _CheckerSpendScope]
        ] = []
        self._admission_closed = False
        self._admission_requests: deque[_CheckerAdmissionRequest] = deque()

    @property
    def enabled(self) -> bool:
        """Return whether this round dispatches checker sessions concurrently."""

        return self._executor is not None

    def submit(
        self,
        key: str,
        dispatch: Callable[[], CheckerOutcome],
        scope: _CheckerSpendScope,
    ) -> None:
        """Admit one checker session, or hold it once the window is full."""

        if self._executor is None:
            return
        with self._admission_lock:
            if key in self._futures or self._admission_closed:
                return
            if any(held_key == key for held_key, _dispatch, _scope in self._held):
                return
            if self._open_admissions >= self._window:
                self._held.append((key, dispatch, scope))
                return
            self._admit_locked(key, dispatch, scope)

    def _admit_locked(
        self,
        key: str,
        dispatch: Callable[[], CheckerOutcome],
        scope: _CheckerSpendScope,
    ) -> None:
        """Hand one held or fresh session to the executor (admission lock held)."""

        assert self._executor is not None
        self._open_admissions += 1
        future = self._executor.submit(self._run, dispatch, scope)
        self._futures[key] = future
        future.add_done_callback(self._feed_from_window)

    def _feed_from_window(self, _future: "Future[_CheckerDispatchOutcome]") -> None:
        """Release the next held session into the freed window slot."""

        with self._admission_lock:
            self._open_admissions -= 1
            if self._admission_closed:
                return
            while self._held and self._open_admissions < self._window:
                key, dispatch, scope = self._held.pop(0)
                self._admit_locked(key, dispatch, scope)

    def claim(self, key: str) -> Optional[_CheckerDispatchOutcome]:
        """Return one key's completed outcome, or ``None`` when never dispatched.

        Blocks until this key's own session finishes; siblings keep running, so
        the commit order is always ladder order regardless of completion order.
        The future is popped so a later re-ask under the same key runs live. A
        key still held behind the admission window is admitted immediately: the
        writer needs it now, and it blocks anyway.
        """

        if self._executor is None:
            return None
        with self._admission_lock:
            if key not in self._futures and not self._admission_closed:
                for index, (held_key, dispatch, scope) in enumerate(self._held):
                    if held_key == key:
                        del self._held[index]
                        self._admit_locked(key, dispatch, scope)
                        break
            future = self._futures.pop(key, None)
        return None if future is None else self._wait_for_future(future)

    def _request_admission(self, scope: _CheckerSpendScope) -> Reservation:
        """Block one worker until the writer durably commits its exact buy."""

        request = _CheckerAdmissionRequest(scope)
        with self._admission_lock:
            self._admission_requests.append(request)
        request.completed.wait()
        if request.error is not None:
            raise request.error
        if request.reservation is None:
            raise DriverIntegrationError("checker admission completed without a reservation")
        return request.reservation

    def _service_admission_requests(self) -> None:
        """Commit every queued checker admission on the canonical writer thread."""

        while True:
            with self._admission_lock:
                if not self._admission_requests:
                    return
                request = self._admission_requests.popleft()
            try:
                request.reservation = self._admit(request.scope)
            except BaseException as exc:  # noqa: BLE001 -- returned to the exact worker
                request.error = exc
            finally:
                request.completed.set()

    def _wait_for_future(
        self, future: "Future[_CheckerDispatchOutcome]"
    ) -> _CheckerDispatchOutcome:
        """Wait for one worker while servicing its durable admission barrier."""

        while True:
            self._service_admission_requests()
            try:
                return future.result(timeout=0.01)
            except FutureTimeoutError:
                continue

    def cancel_pending(self) -> tuple[str, ...]:
        """Cancel every session that has not started; return the cancelled keys.

        BL-8 (d): a usage pause is a decision to STOP SPENDING. Held sessions
        never reached the executor and are dropped; queued futures are
        cancelled (a started future keeps running and is drained normally).
        Admission closes permanently for this round.
        """

        with self._admission_lock:
            self._admission_closed = True
            cancelled = [key for key, _dispatch, _scope in self._held]
            self._held.clear()
            for key in list(self._futures):
                if self._futures[key].cancel():
                    del self._futures[key]
                    cancelled.append(key)
            return tuple(cancelled)

    def drain(self) -> None:
        """Wait for every unclaimed started session before the round unwinds.

        A drained outcome is deliberately discarded rather than committed: the
        writer's abort decision already owns the round, and committing a gate
        from inside an unwind would reorder the canonical sequence relative to
        the serial lane. The session's own artifacts stay on disk.
        """

        with self._admission_lock:
            self._admission_closed = True
            self._held.clear()
        for future in self._futures.values():
            if future.cancelled():
                continue
            try:
                outcome = self._wait_for_future(future)
            except BaseException:  # noqa: BLE001 -- a drain must never mask the abort
                continue
            if outcome.reservation is not None and self._settle_discarded is not None:
                self._settle_discarded(
                    outcome.reservation,
                    outcome.elapsed_seconds,
                    outcome.error is None,
                )
        self._futures.clear()

    def close(self) -> None:
        """Release the worker threads after every session has finished.

        Admission closes first: a completion callback firing during (or after)
        shutdown must find the window sealed, never feed a held item into the
        shut-down executor.
        """

        if self._executor is not None:
            with self._admission_lock:
                self._admission_closed = True
            self._executor.shutdown(wait=True)

    def _run(
        self,
        dispatch: Callable[[], CheckerOutcome],
        scope: _CheckerSpendScope,
    ) -> _CheckerDispatchOutcome:
        """Run one checker session on a worker thread and capture its outcome.

        Every exception is captured rather than propagated so a failing or
        pausing sibling can never abort the round from inside a worker. The
        writer replays it, in ladder order, inside the exact retry and routing
        blocks that owned it before.
        """

        reservation: Optional[Reservation] = None
        attempt_started: Optional[float] = None
        try:
            reservation = self._request_admission(scope)
            attempt_started = self._monotonic()
            # Consume-once discipline: a checker dispatch runs exactly one guard
            # session per reservation, and the settlement belongs to the writer
            # (ladder replay or drain), so the scope declares no per-session
            # admission -- a second guard here refuses typed before any spawn
            # instead of rebinding this reservation's custody.
            with (
                authorized_paid_dispatch(reservation),
                _paid_dispatch_sessions(
                    _PaidDispatchSessions(
                        reservation,
                        admit=None,
                        settle=None,
                        span_started=attempt_started,
                        monotonic=self._monotonic,
                    )
                ),
            ):
                outcome = dispatch()
            elapsed = (
                float(outcome.wall_seconds)
                if outcome.wall_seconds is not None
                else self._monotonic() - attempt_started
            )
            return _CheckerDispatchOutcome(
                outcome=outcome,
                reservation=reservation,
                elapsed_seconds=elapsed,
            )
        except BaseException as exc:  # noqa: BLE001 -- replayed verbatim by the writer
            return _CheckerDispatchOutcome(
                error=exc,
                reservation=reservation,
                elapsed_seconds=(
                    self._monotonic() - attempt_started
                    if reservation is not None and attempt_started is not None
                    else 0.0
                ),
            )


def _metadata_batch_pool_key(batch: Sequence[AuthorArtifact]) -> str:
    """Name one metadata batch's pool slot by its exact member IDs, in order."""

    return "metadata:" + "+".join(str(artifact.proposal["stable_id"]) for artifact in batch)


def _fidelity_pool_key(stable_id: str) -> str:
    """Name one model's fidelity pool slot."""

    return f"fidelity:{stable_id}"


def _terminal_pool_key(stable_id: str) -> str:
    """Name one backlog head's terminal-review pool slot."""

    return f"terminal:{stable_id}"


class _ReplayedFirstAttempt:
    """The retry operation that spends one pool capsule as attempt one.

    The first invocation replays the pool-captured outcome -- returning the
    result or re-raising the captured exception -- and every later invocation
    dispatches live on the writer thread. Threaded through
    ``_retry_infrastructure_call`` unchanged, this keeps the bounded retry
    ladder, the provider-weather parks, and the shutdown checks byte-identical
    to the serial lane: the pool's session simply *is* the first attempt.

    It is an OBJECT rather than a closure for exactly one reason: the capsule's
    prepaid reservation and the wall its worker thread MEASURED around the real
    session travel together, on the operation that replays them. Before PSLRE4
    F1 only the reservation crossed the writer seam, so the failure branch
    settled a writer re-measurement of an instantaneous replay -- a ~0s machine
    floor for a session that really burned its wall, under-counting the
    wall-based checker QUOTA and the epoch RUNAWAY_BREAKER. Carrying both on one
    value is what makes dropping the measurement impossible rather than merely
    discouraged.
    """

    def __init__(
        self,
        capsule: Optional[_CheckerDispatchOutcome],
        live: Callable[[], CheckerOutcome],
    ) -> None:
        """Bind one optional capsule and the writer-thread live dispatch.

        Parameters
        ----------
        capsule:
            Captured pool outcome, or ``None`` when the key was never dispatched
            (pool disabled, or a re-ask): the operation is then fully live.
        live:
            Writer-thread dispatch for retries.
        """

        self._remaining = capsule
        self._live = live
        #: The pool worker's own measurement of the prepaid session's wall.
        self.prepaid_wall_seconds = (
            0.0 if capsule is None else max(0.0, float(capsule.elapsed_seconds))
        )
        #: The reservation that session already durably bought, if any.
        self.prepaid_reservation = None if capsule is None else capsule.reservation

    def __call__(self) -> CheckerOutcome:
        """Replay the captured first attempt once, then dispatch live."""

        if self._remaining is not None:
            captured, self._remaining = self._remaining, None
            return captured.unwrap()
        return self._live()


def _replay_then_live(
    capsule: Optional[_CheckerDispatchOutcome],
    live: Callable[[], CheckerOutcome],
) -> _ReplayedFirstAttempt:
    """Build the retry operation that spends a pool capsule as attempt one."""

    return _ReplayedFirstAttempt(capsule, live)


def _prepaid_checker_session(
    operation: object, declared: Optional[Reservation]
) -> tuple[Optional[Reservation], float]:
    """Return the prepaid reservation and its measured wall for one retry ladder.

    The capsule's replay operation is the authority for BOTH, so a caller cannot
    thread a prepaid reservation across the writer seam while dropping the
    seconds the pool already spent on it. A caller that also names the
    reservation explicitly must name the SAME one.

    The seam fails CLOSED (PSLRE6 acct F3). It used to answer a non-capsule operation
    with ``(declared, 0.0)``: a silent zero wall floor, i.e. exactly the pre-fix
    behaviour, reachable by threading a prepaid reservation with any other operation.
    No caller does that today -- both live call sites pass ``_replay_then_live`` -- so
    the hole was regression-resistance rather than a live path, and a guard that
    answers a caller it cannot measure with a zero floor is a guard that stops
    guarding the moment such a caller appears. It now refuses typed instead.

    Parameters
    ----------
    operation:
        The retry operation ``_retry_infrastructure_call`` was handed.
    declared:
        Reservation the caller passed as ``prepaid_checker_reservation``.

    Returns
    -------
    tuple[Optional[Reservation], float]
        The reservation to spend as attempt one, and its measured wall floor.

    Raises
    ------
    DriverIntegrationError
        If a prepaid reservation is threaded without the capsule that measured it,
        or the capsule carries a different reservation than the caller declared.
    """

    if not isinstance(operation, _ReplayedFirstAttempt):
        if declared is not None:
            raise DriverIntegrationError(
                "a prepaid checker reservation was threaded without the replay capsule "
                "that measured it: the capsule is the authority for the reservation AND "
                "its already-spent wall, and a caller that cannot supply the measured "
                "wall may not spend the reservation as attempt one"
            )
        return None, 0.0
    carried = operation.prepaid_reservation
    if declared is not None and carried is not None and declared is not carried:
        raise DriverIntegrationError(
            "a replayed checker capsule was threaded with a different prepaid "
            "reservation than the session it measured"
        )
    return (carried if carried is not None else declared), operation.prepaid_wall_seconds


#: Each sustained infrastructure outage the author wave can promote to a pause.
#: One mechanism, one streak rule, one place to reason about the economics; the
#: rows differ only in identity and remedy. Keyed by the typed error the lane
#: raises, which is the non-string-matched membership test.
_OUTAGE_PROMOTIONS: tuple[
    tuple[type[AuthorOutageError], str, AuthorPauseReason, int, str, str], ...
] = (
    (
        ResearchToolsUnavailableError,
        "research-tools",
        AuthorPauseReason.RESEARCH_TOOLS_UNAVAILABLE,
        RESEARCH_TOOL_OUTAGE_THRESHOLD,
        "the research provider was unreachable",
        "Restore the configured research MCP/provider, then resume",
    ),
    (
        ForgeRateLimitedError,
        "forge",
        AuthorPauseReason.FORGE_RATE_LIMITED,
        FORGE_RATE_LIMIT_OUTAGE_THRESHOLD,
        "the source forge rate limited the broker",
        (
            "Wait for the forge budget to reset, or supply a credential "
            "(GH_TOKEN/GITHUB_TOKEN, or `gh auth login`) to raise the ceiling from "
            "60 to 5,000 requests per hour, then resume"
        ),
    ),
)


def _author_outage_signal(
    primary: BaseException,
    drained: Sequence[tuple[str, _AuthorDispatchOutcome]],
) -> Optional[AuthorBackoffSignal]:
    """Promote a consecutive cross-model infrastructure-failure streak to a pause.

    Both promotable outages are global conditions wearing a per-model disguise: a
    dead research provider and an exhausted forge budget each fail every model
    that follows, so retrying the rest of the campaign into them buys nothing and
    spends the wave. One or two failures stay retryable because they can be
    transient; a consecutive streak across distinct models is cross-model evidence
    that the infrastructure, not the model, is the problem.

    Parameters
    ----------
    primary:
        First promotable failure encountered by the canonical writer.
    drained:
        Later uncommitted author outcomes in deterministic work order.

    Returns
    -------
    AuthorBackoffSignal | None
        Actionable scheduled-recheck signal once the distinct-model threshold is
        reached, otherwise ``None`` so the isolated failure remains retryable.
    """

    for (
        error_type,
        provider,
        reason,
        threshold,
        symptom,
        remedy,
    ) in _OUTAGE_PROMOTIONS:
        if not isinstance(primary, error_type):
            continue
        signal = _outage_streak_signal(
            primary=primary,
            drained=drained,
            error_type=error_type,
            provider=provider,
            reason=reason,
            threshold=threshold,
            symptom=symptom,
            remedy=remedy,
        )
        if signal is not None:
            return signal
    return None


def _outage_streak_signal(
    *,
    primary: BaseException,
    drained: Sequence[tuple[str, _AuthorDispatchOutcome]],
    error_type: type[AuthorOutageError],
    provider: str,
    reason: AuthorPauseReason,
    threshold: int,
    symptom: str,
    remedy: str,
) -> Optional[AuthorBackoffSignal]:
    """Count one outage kind's consecutive distinct-model streak."""

    ordered: list[tuple[str, Optional[BaseException]]] = [
        (getattr(primary, "stable_id", ""), primary),
        *((stable_id, outcome.error) for stable_id, outcome in drained),
    ]
    streak: list[AuthorOutageError] = []
    for stable_id, error in ordered:
        if not isinstance(error, error_type):
            streak.clear()
            continue
        if error.stable_id != stable_id or any(
            previous.stable_id == stable_id for previous in streak
        ):
            streak.clear()
        streak.append(error)
        if len(streak) < threshold:
            continue
        affected = ", ".join(failure.stable_id for failure in streak)
        details = "; ".join(failure.detail for failure in streak)
        return AuthorBackoffSignal(
            reason=reason,
            retry_after_seconds=None,
            reset_at=None,
            response_excerpt=(
                f"Campaign paused because {symptom} for "
                f"{len(streak)} consecutive models ({affected}). {remedy}; "
                "all affected models remain retryable. "
                f"Observed failures: {details}"
            )[:1_500],
            provider=provider,
        )
    return None


#: Waits, in seconds, before each bounded infrastructure retry. One initial attempt
#: plus one retry per entry. Rung 7's abort (m7362) was two sub-second session deaths
#: 0.85s apart: a zero-backoff single retry cannot survive any host condition lasting
#: longer than about a second, and the observed cause -- a transient CLI credential
#: blip -- resolves on the order of seconds. The schedule stays short enough that a
#: genuinely dead host still halts within about half a minute per lane call.
_INFRASTRUCTURE_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (5.0, 30.0)

#: Machine-classified infrastructure deaths retry at the SESSION level until a
#: paid-session fuse refuses the next buy.  The last entry repeats, while
#: deterministic per-model jitter prevents a killed wave from synchronizing its
#: redrive.  There is intentionally no attempt-count stop: QUOTA/RUNAWAY still
#: charge every provider session and remain the hard bound.
_INFRASTRUCTURE_SESSION_BACKOFF_SECONDS: tuple[float, ...] = (
    5.0,
    30.0,
    60.0,
    120.0,
    240.0,
)
_INFRASTRUCTURE_SESSION_BACKOFF_CEILING_SECONDS = 300.0

#: Weather-scale retry waits for repeated provider-overloaded author crashes. The
#: first wait stays short in case the wrapper caught a single transient edge; later
#: waits park the lane through weather-scale provider storms without spending the
#: ordinary two-entry infrastructure budget on each re-entry.
_PROVIDER_OVERLOAD_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (
    5.0,
    300.0,
    900.0,
    900.0,
    900.0,
)

#: Hard upper bound for provider-overload weather waits.
_PROVIDER_OVERLOAD_BACKOFF_CEILING_SECONDS: float = 900.0

#: Campaign-volume free-space floor for the lane-health host check. Deliberately far
#: below the 30 GiB env-solve floor: this check asks only "can the host still write
#: records and session artifacts", not "can it solve environments".
_AUTHOR_LANE_DISK_FLOOR_BYTES: int = 5 * 1024**3

#: Retained publication witnesses. Only existence and the most recent entry feed the
#: boundary, so the list stays bounded on a 28k-model run.
_AUTHOR_LANE_WITNESS_CAP: int = 50

#: FLOOR of the width-normalized drought threshold (W5): the lane never reads
#: as lane-wide below this many DISTINCT unpublished deaths, whatever the
#: configured width. The full threshold is ``max(FLOOR, ceil(width / 2))`` --
#: at the serial width the historical two-distinct-deaths rule survives; at
#: width 12-18 two independent deaths 5-10 s apart are one member's weather
#: each (the leg-27 escalation this replaces).
_AUTHOR_NO_PUBLICATION_STREAK_LIMIT: int = 2

#: Drought observation window (W5 O arm). The detector is a windowed RATE:
#: zero publications AND the width-normalized distinct-death threshold inside
#: this window. Mirrors the cross-model storm fuse's window; measured on the
#: injectable monotonic clock so virtual-time rehearsals can shape it.
_AUTHOR_LANE_DROUGHT_WINDOW_SECONDS: float = 30.0 * 60.0


@dataclass
class _RetryLadderState:
    """One retry ladder's counters, liftable across scheduler rounds (ACAP).

    ``_retry_infrastructure_call`` historically kept these counters as local
    variables, which forced every model's whole author retry ladder -- every
    attempt and every backoff wait -- to resolve inside ONE pooled
    ``_ensure_authors`` call, in one pool slot. At release-wave scale that was
    the round-clogging pathology: a handful of pathological multi-attempt
    ladders held their slots for hours while ~65 ready models queued behind a
    1-2-wide wave. With the counters in a continuation object, a capped author
    ladder can stop after ``per_round_cap`` attempts, defer typed, and resume
    in a later round exactly where it stopped: same cumulative attempt counts,
    same exhaustion boundaries, same waits -- only WHEN the attempts run moves.

    In-memory, derived-and-disposable: every durable fact (money rows, the
    staged-history allowance, the QUOTA/RUNAWAY fuses) already lives in the
    ledgers, and each run attempt buys its admission exactly when it runs, so
    a deferred continuation can never double-charge. ``work_id`` binds the
    continuation to one work generation; a rotated generation restarts fresh.
    ``per_round_cap=None`` (wave driver, checker lane, direct calls)
    reproduces the historical unbounded-in-call ladder unchanged.
    """

    work_id: str = ""
    per_round_cap: Optional[int] = None
    contract_retries: int = 0
    infrastructure_retries: int = 0
    infrastructure_session_retries: int = 0
    provider_overload_retries: int = 0
    failed_attempts: int = 0
    observed_backoffs: list[float] = field(default_factory=list)
    #: Attempts headed (admissions bought) by the CURRENT pooled call; reset at
    #: every ``_retry_infrastructure_call`` entry.
    attempts_this_round: int = 0
    #: Monotonic deadline for a retry wait parked by a deferral, so the resumed
    #: round waits out only the remainder of the attempt spacing.
    pending_backoff_deadline: Optional[float] = None


class _AuthorRoundAttemptsDeferred(Exception):
    """One author ladder spent its per-round attempt budget without resolving.

    Control flow, not a failure: the commit lane returns the model UNRESOLVED
    (no ledger append, no artifact publication, no terminal), the pipelined
    scheduler keeps it author-ready, and a later round's slice resumes the
    same continuation. Never escapes ``_commit_author_wave``.
    """

    def __init__(
        self, stable_id: str, *, attempts_this_round: int, failed_attempts: int
    ) -> None:
        """Bind the deferring model and its ladder position."""

        super().__init__(
            f"author ladder for {stable_id} deferred after {attempts_this_round} "
            f"attempt(s) this round ({failed_attempts} failed attempt(s) so far); "
            "remaining attempts board a later scheduler round"
        )
        self.stable_id = stable_id
        self.attempts_this_round = attempts_this_round
        self.failed_attempts = failed_attempts


HOST_CAPABILITY_ROUTING_GRANT_CAPABILITY = HOST_CAPABILITY_ROUTING_CAPABILITY
"""The one grant capability that may re-open a host-capability deferral head.

A host-capability head says "not on THIS machine". The two honest ways out are a host
that actually has the memory (re-derived, no operator needed) and an operator declaring
"I am routing this member to a capable host" -- which has to be TYPED. Grant rows carry
``{grant_id, granted_by, new_work_generation, reason, stable_id, stage, attempts}`` and
nothing else, so before this an ordinary requeue was indistinguishable from a capability
routing decision: a generic grant would have re-opened the head, re-bought the author
session, and hit the same wall the head recorded. A new work ID is not evidence that the
host gained RAM.
"""


_HANDOFF_AUTHORITY_STATUS_CODES: frozenset[str] = frozenset(
    {"deferred:needs-cuda", "deferred:needs-x86"}
)
"""Deferrals whose re-run CONSUMES a published executable handoff bundle.

Only the platform pair. The host-capability deferral re-runs the recipe itself on a
bigger machine and never published a handoff, so requiring one of it would refuse the
cohort instead of routing it.
"""


def _only_status_selection(only_status: str) -> set[str]:
    """Expand one ``--only-status`` selector into the exact status codes it selects.

    ONE expansion, consulted by the work filter and the handoff preflight. They were two
    literal copies, and both listed only the platform codes -- so ``deferred:*`` silently
    skipped the host-capability cohort even after the head existed.

    Parameters
    ----------
    only_status:
        Closed selector from the campaign config or driver config.

    Returns
    -------
    set[str]
        Selected status codes.
    """

    if only_status == "deferred:*":
        return {
            "deferred:needs-cuda",
            "deferred:needs-x86",
            *HOST_CAPABILITY_DEFERRAL_STATUS_CODES,
        }
    return {only_status}


def _artifact_has_exact_primary_source(artifact: Any) -> bool:
    """Return whether an artifact publishes an exact public primary source link.

    Mirrors the reducer's own ``exact_primary`` predicate
    (``CanonicalReducer._validate_source_resolution``) so the mint asks the SAME
    question the append will ask, one step earlier and without the refusal. A
    ``deferred:*`` head still owes a real primary link; the carried five never had one
    (``legacy-source-unresolved``), which is precisely why WS-2 back-fills nothing.

    Parameters
    ----------
    artifact:
        Staged author artifact, or ``None``.

    Returns
    -------
    bool
        Whether an exact ``http(s)`` primary source is published.
    """

    if artifact is None:
        return False
    try:
        proposal = artifact.proposal
    except DriverIntegrationError:
        return False
    facts = proposal.get("proposed_facts") if isinstance(proposal, Mapping) else None
    resolution = facts.get("source_resolution") if isinstance(facts, Mapping) else None
    if not isinstance(resolution, Mapping):
        return False
    sources = resolution.get("sources")
    primary = resolution.get("primary_source_id")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        return False
    return any(
        isinstance(source, Mapping)
        and source.get("source_id") == primary
        and str(source.get("url", "")).startswith(("http://", "https://"))
        for source in sources
    )


def _mint_host_capability_head(
    driver: Any,
    item: WorkItem,
    row: JsonObject,
    operational: JsonlLedger,
    *,
    artifact: Optional[Any],
    reducer: Optional[Any],
    state: Optional[JsonObject],
) -> bool:
    """Mint the host-capability model head for one durable static refusal.

    The WS-2 fix, at its single chokepoint. Before this, a host-capacity refusal wrote a
    row and a HEALTHY event and NO head, so ``scheduler.settled()`` counted the run-scoped
    stash as done inside a leg while the operator-facing tally counted heads and reported
    the same members unsettled forever. Eight finishing legs chased that gap.

    Four preconditions, all required, all checked here rather than discovered as an append
    refusal downstream (a refused terminal parks the model and advances the recording
    breaker -- the mint must never spend either):

    * the reducer and driver state are in hand (a canonical head needs canonical
      authority);
    * the row is the static declared-config kind, so a machine-derived proof exists
      (``worker-observed-rss`` never mints);
    * the artifact publishes an exact primary source link, which every ``deferred:*``
      head owes;
    * the projected claim recomputes exactly.

    Parameters
    ----------
    driver:
        The driving object.
    item:
        Exact scheduled work generation.
    row:
        The durable, validated deferral row just persisted.
    operational:
        Locked operational ledger.
    artifact, reducer, state:
        Staged artifact, canonical authority, and mutable driver state.

    Returns
    -------
    bool
        Whether a head was minted.
    """

    if reducer is None or state is None:
        return False
    refusal = host_capability_terminal_refusal(
        row, source_resolved=_artifact_has_exact_primary_source(artifact)
    )
    if refusal is not None:
        return False
    claim = build_host_capability_claim(row)
    try:
        driver._append_terminal_revision(  # noqa: SLF001 -- the driver's own append
            item,
            artifact,
            CAPACITY_DEFERRAL_DISPOSITION,
            None,
            None,
            (),
            reducer,
            operational,
            state,
            # Machine-scoped and re-openable by a capability fact or a typed capability
            # grant, so it needs no human queue -- and it must not consume one: 14% of a
            # rung already lands in human review.
            human_review=False,
            # No gate adjudicated this and none could: the proof IS the measurement.
            terminal_gate_obtained=False,
            host_capability_claim=claim,
        )
    except (DriverPaused, RetryableOperatorError, AuthorBackoffError, CheckerPromptRotatedError):
        # Genuinely campaign-level; the durable row is already written.
        raise
    except Exception as exc:  # noqa: BLE001 -- see the disclosure below
        # The append is deliberately NOT routed through `_terminalize`. That
        # chokepoint's containment ladder parks the model and advances the
        # unrecordable-terminal streak, so an unforeseen refusal of an ADDITIVE
        # accounting improvement could trip the campaign's recording breaker -- three
        # of them would halt a leg over a head the campaign does not even need to
        # proceed (fable's F3(b) hazard, one lane over). Before WS-2 this refusal could
        # not exist at all, so the honest fallback is the pre-WS-2 state: the durable
        # row stands as the accounting, and the failure is DISCLOSED rather than
        # swallowed. The member simply reads `deferred_here` from its row instead of
        # its head.
        _disclose_host_capability_head_refusal(driver, item, row, operational, exc)
        return False
    return True


def _disclose_host_capability_head_refusal(
    driver: Any,
    item: WorkItem,
    row: JsonObject,
    operational: JsonlLedger,
    error: BaseException,
) -> None:
    """Record that a host-capability head could not be minted, and why.

    A silent skip would make the settlement census quietly wrong in the one direction
    WS-2 exists to fix, so the refusal is a first-class disclosure.

    Parameters
    ----------
    driver:
        The driving object.
    item:
        Withheld work generation.
    row:
        The durable deferral row that stands as the accounting instead.
    operational:
        Locked operational ledger.
    error:
        The refusal.
    """

    event: JsonObject = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": driver.dependencies.clock(),
        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
        "status": OperationalEventStatus.HEALTHY.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": driver.config.run_id,
        "machine_id": driver.config.machine_id,
        "details": {
            "disposition": "host-capability-head-refused",
            "stable_id": item.stable_id,
            "work_id": item.active_work_id,
            "row_sha256": row.get("row_sha256"),
            "error_class": type(error).__name__,
            "error": str(error)[:400],
        },
    }
    event = mint_operational_event(event)
    operational.append(event)


def host_capability_head_reopens(
    record: Optional[Mapping[str, Any]],
    *,
    host: Optional[Any] = None,
    grants: Sequence[Mapping[str, Any]] = (),
) -> bool:
    """Return whether a host-capability head is schedulable work again.

    The mirror-image bug WS-2 must not create: a machine-scoped terminal with no way
    back is as wrong as a deferral with no head. Exactly two routes re-open it, and
    both are checked here so every selection path answers identically:

    * the recorded capability fact no longer holds on the host we are actually on --
      re-derived, no operator involved. This is the remedy the row itself prints;
    * an operator grant that NAMES the host-capability routing capability. A generic
      requeue does not qualify: a new work ID is not evidence that the host gained RAM.

    Anything else -- any other status code, or a claim that still refuses here -- stays
    settled. A head whose claim cannot be recomputed re-opens rather than freezing the
    member, matching the valve's documented bias toward attempting.

    WS-1 COORDINATION: this is deliberately ONE pure function so the work-selection
    predicate can consume it verbatim when `work_selection.py` lands; the two call sites
    below are the same two selection seams WS-1 replaces.

    Parameters
    ----------
    record:
        The member's raw current record, or ``None``.
    host:
        Host to judge against. Read from the machine when omitted.
    grants:
        Live grant rows for the member.

    Returns
    -------
    bool
        Whether the head re-opens.
    """

    if record is None:
        return False
    if str(record.get("status", {}).get("code")) not in HOST_CAPABILITY_DEFERRAL_STATUS_CODES:
        return False
    if any(grant_reopens_host_capability_head(grant) for grant in grants):
        return True
    claim = record.get("host_capability")
    if not isinstance(claim, Mapping):
        # A head with no recomputable claim cannot prove it still refuses.
        return True
    return not host_capability_still_refuses(claim, host=host)


def recorded_head_blocks_work(driver: Any, reducer: Any, item: WorkItem) -> bool:
    """Return whether a member's existing record excludes it from this leg's lanes.

    The three author/checker/fidelity lane gates each spelled this condition inline
    (``has_current and not requeue_active and only_status is None``). WS-2 adds a fourth
    clause -- a re-opened host-capability head -- and one shared spelling is the only way
    a member re-opened at SELECTION cannot be silently dropped by a later lane: run 2 on
    a bigger host re-authored the model and then landed no revision at all, because the
    lanes still read the old head as settled.

    Parameters
    ----------
    driver:
        The driving object.
    reducer:
        Canonical reducer.
    item:
        Scheduled work generation.

    Returns
    -------
    bool
        Whether the recorded head blocks further work this leg.
    """

    if not reducer.has_current(item.stable_id):
        return False
    if item.requeue_active or driver.config.only_status is not None:
        return False
    return not host_capability_head_reopens(reducer.current_view.get(item.stable_id))


def grant_reopens_host_capability_head(grant: Mapping[str, Any]) -> bool:
    """Return whether one live grant authorizes re-opening a host-capability head.

    Parameters
    ----------
    grant:
        Requeue-grant row.

    Returns
    -------
    bool
        Whether the grant NAMES the host-capability routing capability. A generic
        requeue never does, so it never re-opens the head.
    """

    return grant.get("capability") == HOST_CAPABILITY_ROUTING_GRANT_CAPABILITY


def _stash_machine_deferral_row(driver: Any, stable_id: str, row: JsonObject) -> None:
    """Retain one persisted deferral valve row for the completion partition.

    Every valve-fire site calls this with the row the ledger actually holds
    (the append returns the validated persisted row, including the original on
    an idempotent resume re-derivation). The completion partition accounts a
    model as machine-deferred ONLY from this run-scoped stash -- never from a
    blind ledger read -- so a stale row from an earlier run cannot mask a model
    the current run admitted and then silently lost.

    Parameters
    ----------
    driver:
        The driving object. Duck-typed test drivers gain the stash lazily.
    stable_id:
        Withheld model.
    row:
        Validated persisted deferral ledger row.
    """

    rows = getattr(driver, "_machine_deferral_rows", None)
    if rows is None:
        rows = {}
        driver._machine_deferral_rows = rows
    rows[stable_id] = row

    generation = getattr(driver, "_leg_generation", None)
    config = getattr(driver, "config", None)
    paths = getattr(driver, "paths", None)
    if generation is None or config is None or paths is None:
        return

    from menagerie.crawler.leg_generation import record_leg_withholding

    record_leg_withholding(
        paths.ledgers.models.parent.parent,
        campaign_id=bound_campaign_id(config),
        generation_id=generation.generation_id,
        run_id=config.run_id,
        stable_id=stable_id,
        withholding_row_sha256=str(row["row_sha256"]),
        created_at=str(row["created_at"]),
    )


def _park_grant_pending_supersession(
    driver: Any,
    item: Any,
    pending_model: Mapping[str, Any],
    blocked: Any,
    operational: JsonlLedger,
) -> JsonObject:
    """Park one completed supersession blocked only by a missing review grant.

    Appends one durable grant-pending park row plus its campaign-health event
    and stashes the row for the completion partition. The model is deliberately
    NOT terminalized and nothing publishes: the review-held terminal stays the
    model's governing record, the embedded pending revision is what an operator
    grant would let the next leg publish, and the run closes
    ``terminal-partition-complete-with-review-holds`` instead of dying at the
    final partition assertion (rung 12 legs 3 and 6).

    Parameters
    ----------
    driver:
        The driving object (duck-typed exactly as the sibling valve helpers).
    item:
        Completed scheduled work generation whose award is blocked.
    pending_model:
        Fully prepared candidate revision the reducer refused solely for want
        of a grant.
    blocked:
        The reducer's typed :class:`GrantPendingSupersessionError`.
    operational:
        Locked operational ledger for the health event.

    Returns
    -------
    dict[str, Any]
        Validated persisted park row.
    """

    created_at = driver.dependencies.clock()
    records_root = driver.paths.ledgers.models.parent.parent
    ledger_path = grant_park_path(records_root)
    persisted = append_grant_park_row(
        ledger_path,
        build_grant_park_row(
            stable_id=item.stable_id,
            work_id=item.active_work_id,
            name=item.intake.name,
            campaign_id=bound_campaign_id(driver.config),
            run_id=driver.config.run_id,
            machine_id=driver.config.machine_id,
            created_at=created_at,
            blocked_parent_revision=str(blocked.blocked_parent_revision),
            # A null blocked stage (the T2-05 review-held deferral class)
            # records the SENTINEL an operator grant must actually name, never
            # the dead literal "none": the row's own recheck hint prints
            # `requeue <id> --stage <blocked_stage>`, and the tool refuses every
            # stage outside FAILURE_REASON_CODES plus this sentinel, so "none"
            # sent the operator to a command that could only fail. The sentinel
            # is what the grant builder accepts and the consumption predicate
            # binds. (Found independently by C3 labs a and b.)
            blocked_stage=(
                str(blocked.blocked_stage)
                if blocked.blocked_stage
                else DEFERRED_TERMINAL_GRANT_STAGE
            ),
            pending_model=pending_model,
        ),
    )
    _stash_machine_deferral_row(driver, item.stable_id, persisted)
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
        "run_id": driver.config.run_id,
        "machine_id": driver.config.machine_id,
        "details": {
            "disposition": GRANT_PARK_EVENT_DISPOSITION,
            "ledger": str(ledger_path.relative_to(records_root)),
            "stable_id": item.stable_id,
            "work_id": item.active_work_id,
            "blocked_parent_revision": str(blocked.blocked_parent_revision),
            "blocked_stage": persisted["pending"]["blocked_stage"],
            "pending_status_code": persisted["pending"]["pending_status_code"],
            "row_sha256": persisted["row_sha256"],
            "recheck_hint": persisted["recheck_hint"],
        },
    }
    # The typed per-occurrence identity covers the complete payload and
    # created_at: a later observation appends, while an exact byte replay no-ops.
    event = mint_operational_event(event)
    operational.append(event)
    return persisted


def _park_award_supersession(
    driver: Any,
    item: Any,
    pending_model: Mapping[str, Any],
    refused: Any,
    operational: Any,
) -> JsonObject:
    """Park one machine re-adjudication the award-custody guard refused (AW-1).

    The award analogue of :func:`_park_grant_pending_supersession`, replacing
    the old route through ``_report_unrecordable_terminal`` (which landed the
    refusal in the terminal-refusal ledger with ``blocked_parent_revision:
    null`` -- ten such rows at the step-100 close -- because the binding read
    the dependency-currency projection). The row binds the reducer-returned
    RAW blocked award head, embeds the refused candidate, and the disclosure
    event is ``degraded``, never ``healthy``: an award supersession attempt
    is operator-attention state by definition.

    Parameters
    ----------
    driver:
        The driving object (duck-typed exactly as the sibling valve helpers).
    item:
        Scheduled work generation whose downgrade is blocked.
    pending_model:
        Fully prepared candidate revision the reducer refused solely under
        the award-custody guard.
    refused:
        The reducer's typed :class:`AwardSupersessionError`, carrying the raw
        blocked award revision.
    operational:
        Locked operational ledger for the degraded disclosure event.

    Returns
    -------
    dict[str, Any]
        Validated persisted park row.
    """

    created_at = driver.dependencies.clock()
    records_root = canonical_records_root(driver.paths.ledgers.models)
    ledger_path = grant_park_path(records_root)
    persisted = append_grant_park_row(
        ledger_path,
        build_grant_park_row(
            stable_id=item.stable_id,
            work_id=item.active_work_id,
            name=item.intake.name,
            campaign_id=bound_campaign_id(driver.config),
            run_id=driver.config.run_id,
            machine_id=driver.config.machine_id,
            created_at=created_at,
            blocked_parent_revision=str(refused.blocked_parent_revision),
            blocked_stage=DEFERRED_TERMINAL_GRANT_STAGE,
            pending_model=pending_model,
            disposition=AWARD_PARK_DISPOSITION,
        ),
    )
    _stash_machine_deferral_row(driver, item.stable_id, persisted)
    event: JsonObject = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": created_at,
        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
        "status": OperationalEventStatus.DEGRADED.value,
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": driver.config.run_id,
        "machine_id": driver.config.machine_id,
        "details": {
            "disposition": AWARD_PARK_EVENT_DISPOSITION,
            "ledger": str(ledger_path.relative_to(records_root)),
            "stable_id": item.stable_id,
            "work_id": item.active_work_id,
            "blocked_parent_revision": str(refused.blocked_parent_revision),
            "pending_status_code": persisted["pending"]["pending_status_code"],
            "pending_model_sha256": persisted["pending"]["pending_model_sha256"],
            "row_sha256": persisted["row_sha256"],
            "recheck_hint": persisted["recheck_hint"],
        },
    }
    event = mint_operational_event(event)
    operational.append(event)
    return persisted


def _park_terminal_refusal(
    driver: Any,
    item: Any,
    *,
    status_code: str,
    reason_code: Optional[str],
    record_error: BaseException,
    fallback_error: Optional[BaseException],
    blocked_parent_revision: Optional[str],
    candidate_sha256: Optional[str],
) -> JsonObject:
    """Park one model whose terminal the recording path refused to record.

    Appends one durable typed refusal-park row and stashes it for the
    completion partition. The model is deliberately NOT terminalized (the
    reducer just refused exactly that) and nothing publishes: the row claims
    the model as machine-withheld so the run closes under the distinct
    ``...-with-refusal-parks`` status instead of dying at the final partition
    assertion, and an operator releases it by fixing the root cause and
    re-driving (a landed revision consumes the row) or by an explicit void.

    The caller's ``terminal-unrecordable`` operational event remains the loud
    disclosure and now names the park it minted; this helper never emits its
    own event, so one refusal stays one fact with one report.

    Parameters
    ----------
    driver:
        The driving object (duck-typed exactly as the sibling valve helpers).
    item:
        Scheduled work generation whose terminal could not be recorded.
    status_code, reason_code:
        Disposition the driver intended to record.
    record_error:
        Failure raised by the intended terminal append.
    fallback_error:
        Failure raised by the minimal fallback append, when one was attempted.
    blocked_parent_revision:
        The model's current record revision at refusal time, or ``None``.
    candidate_sha256:
        Content digest of the last prepared candidate the reducer refused,
        when preparation itself succeeded.

    Returns
    -------
    dict[str, Any]
        Validated persisted park row.
    """

    records_root = driver.paths.ledgers.models.parent.parent
    persisted = append_refusal_park_row(
        refusal_park_path(records_root),
        build_refusal_park_row(
            stable_id=item.stable_id,
            work_id=item.active_work_id,
            name=item.intake.name,
            campaign_id=bound_campaign_id(driver.config),
            run_id=driver.config.run_id,
            machine_id=driver.config.machine_id,
            created_at=driver.dependencies.clock(),
            refusal_class=type(record_error).__name__,
            record_error_type=type(record_error).__name__,
            record_error=str(record_error) or type(record_error).__name__,
            fallback_error_type=(
                type(fallback_error).__name__ if fallback_error is not None else None
            ),
            fallback_error=(
                (str(fallback_error) or type(fallback_error).__name__)
                if fallback_error is not None
                else None
            ),
            intended_status_code=status_code,
            intended_reason_code=reason_code,
            blocked_parent_revision=blocked_parent_revision,
            candidate_sha256=candidate_sha256,
            # D-48: record the parking code tree. It joins the park FACT
            # identity, so this exact refusal re-derived after a tree change
            # appends a fresh governing row and closes the machine release
            # lane again until the NEXT tree change.
            tree_sha=current_park_tree_identity(),
        ),
    )
    _stash_machine_deferral_row(driver, item.stable_id, persisted)
    return persisted


def _is_checker_session_exhaustion(failure: Exception) -> bool:
    """Return whether one checker-lane failure is the closed QUOTA-exhaustion class.

    Machine-classified, never message-text matching: ``CheckerWallBudgetExceeded``
    is minted at exactly one site -- the checker admission's
    ``PaidSessionBudgetExceeded`` QUOTA remap -- and a bare QUOTA refusal that
    escapes un-remapped classifies identically. Generation-scoped QUOTA
    exhaustion is permanent for the generation, so the park path terminalizes
    it (reviewable, grant-re-drivable) instead of parking run-scoped, where the
    model would requeue and re-park on every future leg forever (the pilot's
    five no-exit parks, 2026-08-21).

    Parameters
    ----------
    failure:
        Lane failure caught by the metadata-gate failure route.

    Returns
    -------
    bool
        Whether the failure is generation-scoped checker budget exhaustion.
    """

    if isinstance(failure, CheckerWallBudgetExceeded):
        return True
    return isinstance(failure, PaidSessionBudgetExceeded) and failure.instrument == "QUOTA"


def _checker_lane_failure_class(failure: Exception) -> str:
    """Name what the checker lane did instead of rendering a verdict.

    Parameters
    ----------
    failure:
        Lane failure caught by the metadata-gate failure route.

    Returns
    -------
    str
        One of :data:`menagerie.crawler.checker_lane.CHECKER_LANE_FAILURE_CLASSES`.
    """

    if isinstance(failure, CheckerItemContractError):
        # BL-10 (c): truthful attribution. An author-side write-contract defect
        # is a fact about the model's own author artifact -- the next census
        # must look in the author lane, not the checker lane. A checker-side
        # per-item defect stays an invalid gate result.
        return "author-evidence-defect" if failure.author_side else "invalid-gate-result"
    if isinstance(failure, CheckerWallBudgetExceeded):
        return "checker-wall-exhausted"
    if isinstance(failure, CheckerPartialResultError):
        return "partial-result"
    message = str(failure)
    # The operator wrapper's own exit-64 classification travels in the typed
    # ``DriverIntegrationError`` raised by ``_raise_for_checker_exit``; both its
    # bracketed sidecar label and its fixed message prefix identify the class.
    if "permanent-contract-rejection" in message or message.startswith(
        "checker violated the gate contract"
    ):
        return "permanent-contract-rejection"
    if isinstance(failure, (GateRoutingError, CheckerDispatchError)):
        return "invalid-gate-result"
    return "lane-error"


def _file_stat_signature(paths: Sequence[Path]) -> tuple[Any, ...]:
    """Return an append-detecting stat signature over ordered ledger files.

    Append-only ledgers grow monotonically, so ``(size, mtime_ns)`` per file
    changes on every append (including appends from OTHER processes, which a
    generation counter held in this process cannot see). A missing file is a
    distinct ``None`` arm, so creation and deletion both change the signature.

    Parameters
    ----------
    paths:
        Ordered candidate ledger paths.

    Returns
    -------
    tuple[Any, ...]
        One ``(path, size, mtime_ns)`` or ``(path, None)`` fact per input.
    """

    facts: list[tuple[Any, ...]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            facts.append((str(path), None))
            continue
        facts.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(facts)


def _stat_signature_cached(driver: Any, cache_key: str, paths: Sequence[Path], compute: Any) -> Any:
    """Serve one driver-scoped derived read while its source files are unchanged.

    The valve-certainty reads below re-read and re-validate whole ledgers from
    disk PER WORK ITEM (T4-T5: the models-shard and valve-ledger full scans
    were a measured per-model hot-path cost). Caching on the stat signature
    keeps every consumer exactly as fresh as the files themselves: any append,
    truncation, creation, or removal changes the signature and recomputes.
    Cached values are shared; consumers must never mutate them.

    Parameters
    ----------
    driver:
        The driving object (duck-typed for test drivers); the cache lives on
        it so one run's caches die with it.
    cache_key:
        Stable name for this derived read.
    paths:
        Every file the derived read consumes.
    compute:
        Zero-argument callable producing the derived value.

    Returns
    -------
    Any
        The cached or freshly computed derived value.
    """

    signature = _file_stat_signature(paths)
    caches = getattr(driver, "_stat_signature_caches", None)
    if not isinstance(caches, dict):
        caches = {}
        try:
            driver._stat_signature_caches = caches
        except AttributeError:
            return compute()
    cached = caches.get(cache_key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    value = compute()
    caches[cache_key] = (signature, value)
    return value


def _persisted_deferral_rows(driver: Any) -> dict[str, tuple[JsonObject, ...]]:
    """Read every durable valve row one campaign holds, keyed by stable ID.

    The read is deliberately LENIENT per row (an unreadable or invalid line is
    skipped, never fatal) because both consumers already refuse bad rows where
    it matters: a row can only CLAIM a model on resume after the same
    fail-closed :func:`~menagerie.crawler.status.machine_deferral_rejection`
    validation the completion partition applies plus a freshness re-check, and
    completion reads these rows for supersession REPORTING only, never for
    claims. A model whose only accounting was a malformed row therefore still
    ends in the partition's loud ``PartitionError``.

    Parameters
    ----------
    driver:
        The driving object (duck-typed for test drivers, exactly as
        :func:`_stash_machine_deferral_row`).

    Returns
    -------
    dict[str, tuple[dict[str, Any], ...]]
        Validated ledger rows per stable ID, capacity rows before coverage
        rows before checker-lane rows before rebuild rows before disk-floor
        rows before grant-pending park rows before terminal-refusal park rows.
        The capacity, coverage, and checker-lane ledgers surface in file order
        (each writes at most one row per work generation); the rebuild,
        disk-floor, grant-park, and refusal-park ledgers surface only their
        GOVERNING row per work generation (rebuild: latest v2, else latest
        legacy v1; disk floor: latest host fact; grant/refusal park: latest
        unvoided row), so a superseded, voided, or legacy-duplicated park fact
        can never out-claim its successor.

    Notes
    -----
    The read covers EVERY member of the closed
    :data:`~menagerie.crawler.status.MACHINE_DEFERRAL_DISPOSITIONS` set. The
    checker-lane ledger was write-only to this reader until C3-B, even though
    its disposition is accounted, ``crawl status --verify-partition`` loads it,
    and its module contract says the partition dereferences its rows like any
    admission-valve row. That omission cost two things: a superseded
    checker-lane row could never reach the close's ``superseded_rows`` (the
    rung-10 disclosure gap), and ``tools/merge_host_shards`` -- which derives
    shard OWNERSHIP from this exact read -- saw a checker-lane-parked model as
    owned by no shard and refused the merge. Reading it changes no spend
    behavior: the checker-lane fact has no
    :func:`_deferral_row_fact_still_holds` arm, so it never reclaims a model.
    """

    from menagerie.crawler.quota_deferral import (
        quota_deferral_path,
        validate_quota_deferral_row,
    )

    records_root = driver.paths.ledgers.models.parent.parent
    valve_paths = (
        capacity_deferral_path(records_root),
        coverage_deferral_path(records_root),
        checker_lane_deferral_path(records_root),
        rebuild_deferral_path(records_root),
        disk_floor_deferral_path(records_root),
        grant_park_path(records_root),
        refusal_park_path(records_root),
        scheduler_quarantine_path(records_root),
        quota_deferral_path(records_root),
    )

    def lenient_rows(path: Path, validator: Any) -> list[JsonObject]:
        """Read one ledger, skipping unreadable or invalid lines."""

        if not path.is_file():
            return []
        try:
            payload = path.read_text(encoding="utf-8")
        except OSError:
            return []
        found: list[JsonObject] = []
        for line in payload.splitlines():
            if not line.strip():
                continue
            try:
                found.append(validator(json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue
        return found

    def compute() -> dict[str, tuple[JsonObject, ...]]:
        """Read and validate every valve ledger once per file change."""

        rows: dict[str, list[JsonObject]] = {}
        for path, validator in (
            (valve_paths[0], validate_capacity_deferral_row),
            (valve_paths[1], validate_coverage_deferral_row),
            (valve_paths[2], validate_checker_lane_deferral_row),
            # W1 quota-deferral rows surface in file order, exactly like the
            # checker-lane valve: no _deferral_row_fact_still_holds arm, so a
            # row never reclaims a model -- the recidivist hold is enforced at
            # the paid-admission chokepoint, not here.
            (valve_paths[8], validate_quota_deferral_row),
        ):
            for row in lenient_rows(path, validator):
                rows.setdefault(str(row["stable_id"]), []).append(row)
        for row in current_scheduler_quarantine_rows(
            lenient_rows(valve_paths[7], validate_scheduler_quarantine_ledger_row)
        ):
            rows.setdefault(str(row["stable_id"]), []).append(row)
        for row in current_rebuild_rows(
            lenient_rows(valve_paths[3], validate_rebuild_deferral_row)
        ):
            rows.setdefault(str(row["stable_id"]), []).append(row)
        for row in current_disk_floor_rows(
            lenient_rows(valve_paths[4], validate_disk_floor_deferral_row)
        ):
            rows.setdefault(str(row["stable_id"]), []).append(row)
        for row in current_grant_park_rows(
            lenient_rows(valve_paths[5], validate_grant_park_ledger_row)
        ):
            rows.setdefault(str(row["stable_id"]), []).append(row)
        for row in current_refusal_park_rows(
            lenient_rows(valve_paths[6], validate_refusal_park_ledger_row)
        ):
            rows.setdefault(str(row["stable_id"]), []).append(row)
        return {stable_id: tuple(found) for stable_id, found in rows.items()}

    return _stat_signature_cached(driver, "persisted-deferral-rows", valve_paths, compute)


def _still_valid_deferral_row(driver: Any, item: WorkItem) -> Optional[JsonObject]:
    """Return the durable valve row that still withholds this exact work, if any.

    Resume idempotency (rung 10): leg 2 re-bought five full author sessions
    (~50 minutes of LLM spend) to re-derive byte-equivalent capacity deferrals,
    because the durable valve rows were never consulted before author dispatch.
    A row short-circuits re-authoring ONLY when all of these hold:

    * it names this stable ID and this exact active work generation -- a
      requeue or refresh mints a new work ID, so operator-forced re-authoring
      always escapes the short-circuit;
    * it survives the same fail-closed digest/disposition/remedy validation the
      completion partition applies to every claim;
    * the fact it records is still true HERE and NOW
      (:func:`_deferral_row_fact_still_holds`) -- a bigger host or a re-solved
      environment invalidates the row and the model is authored normally.

    Supersession semantics are preserved by caller ordering: canonical and
    rehydrated authority are consulted first, so a model holding a terminal or
    a staged final result never reaches this check (the record wins, exactly as
    in the completion partition's W3-P arm).

    Parameters
    ----------
    driver:
        The driving object (duck-typed for test drivers).
    item:
        Exact scheduled work generation.

    Returns
    -------
    dict[str, Any] | None
        The first still-valid persisted row, or ``None`` when the model must be
        authored.
    """

    for row in _persisted_deferral_rows(driver).get(item.stable_id, ()):
        if str(row.get("work_id")) != item.active_work_id:
            continue
        if machine_deferral_rejection(item.stable_id, row) is not None:
            continue
        if _deferral_row_fact_still_holds(driver, item, row):
            return row
    return None


def _deferral_row_fact_still_holds(driver: Any, item: WorkItem, row: JsonObject) -> bool:
    """Re-check one valve row's recorded fact against the current world.

    Every check errs toward ``False`` -- attempting the model -- because the
    valves themselves are documented fail-open toward attempting: a wrongly
    re-claimed deferral would freeze a model past the exact change (bigger
    host, re-solved environment) its remedy asked for, while a wrongly released
    row merely re-buys one author session before the live valve re-derives the
    same deferral from current facts.

    Parameters
    ----------
    driver:
        The driving object (duck-typed for test drivers).
    item:
        Exact scheduled work generation.
    row:
        Digest-validated persisted deferral row.

    Returns
    -------
    bool
        Whether the recorded refusal is still demonstrably true here.
    """

    disposition = row.get("disposition")
    if disposition == CAPACITY_DEFERRAL_DISPOSITION:
        try:
            host = host_capacity()
        except CapacityDeferralError:
            return False
        estimate = row.get("capacity", {}).get("estimate") or {}
        bound = estimate.get("parameter_count_lower_bound")
        if isinstance(bound, int) and not isinstance(bound, bool) and bound > 0:
            return bound > host.admissible_parameter_ceiling
        # W3-R worker-observed-RSS rows carry a measured peak instead of a
        # static bound. DX-7/RE-7: the binding constraint is the WORKER RSS
        # ALLOWANCE this host would actually enforce, not physical RAM -- on
        # the very host that killed the worker, RAM always exceeds the peak
        # (the cap sat below RAM by construction), so the old RAM comparison
        # released the row immediately and the member re-spawned into the
        # identical kill (the m3767 x14 waste class, 20.5% of 200a worker
        # wall). The refusal still holds while the CURRENT effective kill
        # threshold does not clear the observed peak; a bigger host or a
        # raised allowance releases it exactly as the remedy asks.
        peak = estimate.get("max_peak_rss_bytes")
        if isinstance(peak, int) and not isinstance(peak, bool) and peak > 0:
            from menagerie.crawler.driver_receipts import (  # noqa: PLC0415 -- import cycle guard
                derive_worker_rss_limit_bytes,
            )
            from menagerie.crawler.worker_supervisor import (  # noqa: PLC0415
                effective_rss_kill_threshold_bytes,
            )

            allowance, _basis = derive_worker_rss_limit_bytes()
            return effective_rss_kill_threshold_bytes(allowance) <= peak
        return False
    if disposition == COVERAGE_DEFERRAL_DISPOSITION:
        coverage = row.get("coverage") or {}
        library = coverage.get("library") or {}
        distribution = library.get("distribution")
        if not isinstance(distribution, str) or not distribution.strip():
            return False
        if distribution_vocabulary_rejection(distribution) is not None:
            # A package-index token is not a distribution; the row's remedy
            # ("declare 'pypi' in an intent") can never be actioned, so it can
            # never re-claim a model (rung 10 held five such rows).
            return False
        if coverage.get("routed_intent") != item.route.intent:
            return False
        packages = _routed_environment_packages(
            getattr(driver, "registry", None), item.route.intent
        )
        if not packages:
            # An unlocked or unreadable inventory is "unknown", never "still
            # provably lacking"; unknown admits, exactly as it does in the live
            # assessment.
            return False
        return _inventory_provides(packages, distribution) is None
    if disposition == REBUILD_DEFERRAL_DISPOSITION:
        quarantine = row.get("quarantine") or {}
        if quarantine.get("condition_scope") == "intent-setup-cohort":
            # BL-9d: a cohort row bounds the blast radius of one environment
            # effort exhaustion. Exactly ONE representative member holds the
            # durable ``intent-setup`` park (the human-visible fact carrier
            # released by rotation or an explicit clearance); every other
            # member of the cohort was withheld for THAT LEG'S accounting only
            # and re-admits automatically, exactly like a host-condition row.
            # If the setup is still broken, the live valve re-derives the
            # withholding from current facts without any per-member human
            # action -- one operator action on the representative's park frees
            # the whole rung, instead of 46/60 members sitting behind a
            # human-only release.
            return False
        if quarantine.get("condition_scope") == "host-condition":
            # A transient host condition (busy lifecycle, store unmounted,
            # residue awaiting cleanup, wrapper outage) is never provably
            # still true later: the row can never re-claim the model, and the
            # live environment-failure valve re-derives the park from current
            # facts if the condition persists. Erring toward attempting is
            # cheap here -- these models were already authored when the
            # environment refused them.
            return False
        if quarantine.get("intent") != item.route.intent:
            return False
        registry = getattr(driver, "registry", None)
        intents = getattr(registry, "intents", None) if registry is not None else None
        intent = intents.get(item.route.intent) if isinstance(intents, Mapping) else None
        if intent is None or quarantine.get("target") != intent.lock.target:
            return False
        recorded_identity = quarantine.get("artifact_identity")
        if not isinstance(recorded_identity, str) or not recorded_identity:
            return False
        # An explicit operator clearance receipt releases the row even when the
        # setup artifacts are byte-identical: an out-of-band backend repair is
        # invisible to identity comparison by construction, so the receipt is
        # the only honest release signal for that case.
        quarantine_event_id = quarantine.get("event_id")
        operational_path = canonical_operational_ledger_path(driver.paths.ledgers.models)
        # BL-9c: receipts land in the canonical operational ledger (historical)
        # OR the operator-writable clearance sidecar (recordable mid-leg); the
        # release test merges both.
        clearance_records_root = driver.paths.ledgers.models.parent.parent
        clearance_sidecar = environment_clearance_path(clearance_records_root)
        if isinstance(quarantine_event_id, str) and _quarantine_is_cleared(
            _stat_signature_cached(
                driver,
                "quarantine-clearance-events",
                (operational_path, clearance_sidecar),
                lambda: (
                    *scan_jsonl(operational_path),
                    *environment_clearance_events(clearance_records_root),
                ),
            ),
            quarantine_event_id,
        ):
            return False
        # This mirrors the live quarantine valve's matching rule EXACTLY: the
        # ``environment-cleanup-quarantined`` event governs an intent while the
        # TYPED committed setup-artifact identity is unchanged (absence of a
        # member is itself a typed, distinguishable state -- distinct partial
        # families never collapse together). An identical identity means the
        # env lane would re-derive this same park, so claiming here only saves
        # the spend; a recovery/rotation re-solves the intent, changes the
        # identity, and releases the row.
        return _intent_setup_artifact_identity(intent) == recorded_identity
    if disposition == DISK_FLOOR_DEFERRAL_DISPOSITION:
        # The refusal is a fact about the RECORDING host. Any other machine
        # attempts normally -- that is exactly the re-drive path the row exists
        # to keep open -- and on the same machine the row holds only while a
        # live measurement still falls below the recorded floor.
        machine_id = getattr(getattr(driver, "config", None), "machine_id", None)
        if not isinstance(machine_id, str) or row.get("machine_id") != machine_id:
            return False
        disk = row.get("disk") or {}
        floor = disk.get("floor_bytes")
        env_root = disk.get("env_root")
        if not isinstance(floor, int) or isinstance(floor, bool) or floor <= 0:
            return False
        if not isinstance(env_root, str) or not env_root:
            return False
        destination_role = disk.get("destination_role")
        if destination_role == "tmp":
            # The row froze the RECORDED tmp path. Tmp is configuration
            # (TMPDIR), so the fact holds only while that configuration is
            # still in effect: a moved tmp destination means the recorded
            # starvation describes a path no longer in use -- release toward
            # attempting, and the live per-destination floor check re-derives
            # the park against the NEW path if it is genuinely starved.
            import tempfile

            environment_lane = getattr(getattr(driver, "dependencies", None), "environments", None)
            active_tmp_root = getattr(environment_lane, "tmp_root", None)
            current_tmp_root = (
                Path(active_tmp_root)
                if isinstance(active_tmp_root, Path)
                else Path(tempfile.gettempdir())
            )
            if str(current_tmp_root.resolve()) != str(Path(env_root).resolve()):
                return False
        root = Path(env_root)
        if not root.is_dir():
            # A vanished root is "unknown", never "still provably starved";
            # unknown attempts, exactly as in the live floor check.
            return False
        try:
            free = disk_free_bytes(root)
        except OSError:
            return False
        return free < floor
    if disposition == GRANT_PENDING_DISPOSITION:
        # The fact is "this exact review-held revision still governs the
        # ledger head". A granted supersession (or any newer revision) moved
        # the head and releases the row; an unreadable ledger is "unknown",
        # never "still provably blocked", so the model is attempted normally
        # and the live tripwire re-derives the park from current facts.
        pending = row.get("pending") or {}
        blocked_revision = pending.get("blocked_parent_revision")
        if not isinstance(blocked_revision, str) or not blocked_revision:
            return False
        try:
            models_path = driver.paths.ledgers.models
            head = _stat_signature_cached(
                driver,
                "grant-park-models-head",
                (models_path,),
                lambda: _select_current(scan_jsonl(models_path)),
            ).get(str(row.get("stable_id")))
        except (OSError, ValueError):
            return False
        if head is None:
            return False
        return head.get("record_revision") == blocked_revision and bool(
            head.get("status", {}).get("human_review", {}).get("required")
        )
    if disposition == CHECKER_LANE_DEFERRAL_DISPOSITION:
        # BL-10 (f): the causal release predicate. A v2 row records the checker
        # ADJUDICATION IDENTITY the lane failed under; the recorded fact
        # ("this lane, under this identity, could not deliver a verdict on this
        # model") provably still holds while that identity is unchanged AND the
        # model has already spent its bounded recidivism budget -- re-attempting
        # then re-buys the full halving ladder every leg forever (the pilot's
        # six parks were all perpetual re-buyers). The row RELEASES the moment
        # the lane's identity rotates (model, version, or prompt -- the causal
        # change the remedy asks for), on a v1 row (no identity recorded), or
        # while the recidivism budget still has a fresh-leg retry left.
        lane = row.get("checker_lane") or {}
        recorded_identity = lane.get("checker_identity")
        if not isinstance(recorded_identity, Mapping):
            return False
        checker = getattr(getattr(driver, "dependencies", None), "checker", None)
        config = getattr(driver, "config", None)
        if checker is None or config is None:
            return False
        current = lane_adjudication_identity(checker, str(lane.get("gate_kind")), config)
        if current is None:
            return False
        if (
            current.provider != recorded_identity.get("provider")
            or current.model != recorded_identity.get("model")
            or current.version != recorded_identity.get("version")
            or current.prompt_sha256 != recorded_identity.get("prompt_sha256")
        ):
            return False
        operational_path = canonical_operational_ledger_path(driver.paths.ledgers.models)

        def _park_counts() -> dict[tuple[str, str], int]:
            counts: dict[tuple[str, str], int] = {}
            for event in scan_jsonl(operational_path):
                details = event.get("details") if isinstance(event, Mapping) else None
                if (
                    not isinstance(details, Mapping)
                    or details.get("disposition") != "checker-lane-parked"
                ):
                    continue
                key = (str(details.get("stable_id")), str(details.get("work_id")))
                counts[key] = counts.get(key, 0) + 1
            return counts

        try:
            counts = _stat_signature_cached(
                driver, "checker-lane-park-counts", (operational_path,), _park_counts
            )
        except (OSError, ValueError):
            # An unreadable ledger is "unknown", never "provably exhausted";
            # unknown attempts, as every sibling arm does.
            return False
        parks = counts.get((str(row.get("stable_id")), str(row.get("work_id"))), 0)
        return parks >= CHECKER_LANE_PARK_RECIDIVISM_LIMIT
    if disposition == SCHEDULER_QUARANTINE_DISPOSITION:
        # N2/W12: one exact work generation earns bounded fresh-leg retries
        # counted since the last BUDGET-RESTORING release (remedied void or
        # eligible machine drain -- PD-11's composition). Once charged parks
        # reach the cap, the current v2 row holds until the dedicated
        # scheduler-quarantine tool appends a restoring release. Legacy v1
        # rows preserve requeue-every-run behavior.
        #
        # DX-1/RE-1 composition detail: the retry is FRESH-LEG only, exactly as
        # this arm has always documented. In-run exclusion is owned by the
        # run-scoped withhold set (`_park_non_model_fault` joins
        # `_checker_lane_parked_ids`, which the scheduler consults), so a park
        # can never land an in-window head beside its own live quarantine row
        # -- the LC-1 pinned "leg contradicted itself" refusal stays
        # unreachable while the PD-1 ruling that owns that cell is pending.
        operational_path = canonical_operational_ledger_path(driver.paths.ledgers.models)
        records_root = driver.paths.ledgers.models.parent.parent
        quarantine_path = scheduler_quarantine_path(records_root)
        try:
            events = _stat_signature_cached(
                driver,
                "scheduler-quarantine-events",
                (operational_path,),
                lambda: scan_jsonl(operational_path),
            )
            ledger_rows = _stat_signature_cached(
                driver,
                "scheduler-quarantine-ledger",
                (quarantine_path,),
                lambda: load_scheduler_quarantine_ledger([quarantine_path]),
            )
        except (OSError, ValueError):
            return False
        return not scheduler_quarantine_released(row, events, ledger_rows=ledger_rows)
    # Terminal-refusal parks deliberately have NO still-holds arm: their fact
    # ("the recording path refused this candidate") can only be re-verified by
    # re-running, so the model is attempted normally and the respective ladder
    # re-parks idempotently if the refusal still holds.
    return False


class AdmissionEnvironmentMixin:
    """Admission, environment, and execution workflow methods for the driver."""

    _authority_context: Optional[AuthorityContext]
    _artifact_checkpoint_cache: Optional[
        tuple[ArtifactEventLedger, int, tuple[Path, ...], ArtifactCheckpointIncrement]
    ]
    _checkpoint_memo_instance: Optional[ValidationMemo]
    _ledger_hot_path_samples: list[tuple[int, float]]
    # Author-lane health state (initialized by the concrete driver facade).
    _author_lane_publications: list[dict[str, Any]]
    _author_no_publication_deaths: list[dict[str, Any]]
    _models_with_staged_work: set[str]
    _reduced_model_ids: set[str]

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            """Describe collaborators supplied by the concrete driver facade."""

            raise AttributeError(name)

    def _restore_quarantined_environment_context(
        self,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
    ) -> None:
        """Restore exact durable environment generations before currency projection.

        Parameters
        ----------
        reducer, operational:
            Locked reducer and canonical lifecycle-event ledger.
        """

        generations = restored_environment_generations(
            reducer.context.environment_generations,
            self.paths.ledgers.attempts,
            operational.records,
        )
        # Terminal assembly must also resolve OLDER generations of families
        # that rebuilt mid-leg: register the full attested generation->family
        # history alongside the one-slot currency map (m8830 circuit-breaker).
        reducer.register_attested_generation_families(
            attested_generation_families(self.paths.ledgers.attempts, operational.records)
        )
        if generations == dict(reducer.context.environment_generations):
            return
        refreshed = replace(reducer.context, environment_generations=generations)
        reducer.update_context(refreshed)
        self._authority_context = refreshed

    def _reconcile_lifecycle_before_admission(
        self,
        operational: JsonlLedger,
        reducer: CanonicalReducer,
        snapshot: IntakeSnapshot,
    ) -> Optional[DriverResult]:
        """Reconcile durable worker and recurring wake state before scheduling."""

        recovery = reconcile_worker_lease(self.paths.worker_lock, self.paths.worker_lease)
        if recovery.state in {"active", "failed-closed"}:
            raise DriverIntegrationError(
                f"worker lease recovery blocks admission: {recovery.state}: {recovery.detail}"
            )
        if recovery.lease is not None:
            event_kind = (
                OperationalEventKind.WORKER_LEASE_REAPED.value
                if recovery.reaped
                else OperationalEventKind.WORKER_LEASE_CLOSED.value
            )
            status = (
                OperationalEventStatus.WORKER_LEASE_REAPED.value
                if recovery.reaped
                else OperationalEventStatus.WORKER_LEASE_CLOSED.value
            )
            self._append_worker_lifecycle_event(
                operational,
                event_kind=event_kind,
                status=status,
                lease_id=recovery.lease.lease_id,
                stable_id=recovery.lease.stable_id,
                details={"recovery_state": recovery.state, "detail": recovery.detail},
            )
            if recovery.reaped:
                intake = next(
                    (
                        candidate
                        for candidate in snapshot.items
                        if candidate.stable_id == recovery.lease.stable_id
                    ),
                    None,
                )
                if intake is None:
                    raise DriverIntegrationError("reaped worker lease is outside active intake")
                item = WorkItem(
                    intake,
                    route_model(intake.to_model_requirements(_framework_from_intake(intake))),
                )
                attempt = _driver_failure_attempt(
                    item,
                    None,
                    "runner",
                    "internal-error",
                    DriverIntegrationError(recovery.detail),
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                attempt["work_id"] = recovery.lease.work_id
                reducer.append_attempt(attempt)
            self.paths.worker_lease.unlink(missing_ok=True)

        context = self._context(0, None)
        now = self.dependencies.clock()
        projection = reduce_wake_episodes(operational.records)
        episode_state = (
            projection.episodes.get(self.config.wake_episode_id)
            if self.config.wake_episode_id is not None
            else None
        )
        if self.config.wake_episode_id is not None and episode_state is None:
            raise DriverIntegrationError("scheduled callback names an unknown wake episode")
        callback_argv = (
            episode_state.episode.callback_argv[:-2]
            if episode_state is not None
            else self._wakeup_callback_argv()
        )
        manager = WakeupManager(
            self.paths.wakeup_root,
            operational,
            callback_argv,
            backend=episode_state.backend if episode_state is not None else None,
            installer=self.dependencies.wakeup_installer,
            verifier=self.dependencies.wakeup_verifier,
            deactivator=self.dependencies.wakeup_deactivator,
        )
        manager.ingest_fire_intents(context=context, created_at=now)
        if self.config.invocation_origin is InvocationOrigin.MANUAL_RESUME:
            for active in manager.projection.active_episodes:
                manager.resolve_episode(
                    active.episode.episode_id,
                    resolution="manual-resume",
                    context=context,
                    created_at=now,
                )
        if self.config.invocation_origin is InvocationOrigin.WAKE_CALLBACK:
            assert self.config.wake_episode_id is not None
            callback = manager.handle_fire(
                self.config.wake_episode_id,
                fired_at=now,
                context=context,
            )
            if not callback.should_resume:
                manager.reconcile(context=context, created_at=now)
                return DriverResult("wake-noop", len(reducer.current_view), 0, None)
            # D-62: a fired wake honors the durable operator HOLD sentinel --
            # the resume parks typed (wake-noop, disclosed) instead of running
            # a leg through a standing hold. Ambiguity fails closed.
            from menagerie.crawler.pause_authority import evaluate_sprint_pause

            hold_decision = evaluate_sprint_pause(
                canonical_records_root(self.paths.ledgers.models),
                now=now,
            )
            if hold_decision.held:
                hold_event: JsonObject = {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": now,
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
                        "disposition": "sprint-pause-hold",
                        "admission_boundary": "wake-resume",
                        "decision": hold_decision.decision,
                        "detail": hold_decision.detail,
                    },
                }
                operational.append(mint_operational_event(hold_event))
                manager.reconcile(context=context, created_at=now)
                return DriverResult("wake-noop", len(reducer.current_view), 0, None)
        reconciliation = manager.reconcile(context=context, created_at=now)
        if reconciliation.failures:
            raise DriverIntegrationError(
                f"wakeup projection reconciliation failed: {reconciliation.failures}"
            )
        return None

    def _refresh_authority_context(self, snapshot: IntakeSnapshot) -> AuthorityContext:
        """Return the live authority context that binds dependency-refresh work.

        ``_run_locked`` builds and stores the run's context before any work is
        ordered; direct ``_ordered_work`` callers (tests, tools) fall back to an
        identical construction from the same snapshot and config so the epoch
        binding is never silently absent.

        Parameters
        ----------
        snapshot:
            Active validated intake snapshot.

        Returns
        -------
        AuthorityContext
            The run's active trust roots.
        """

        context = getattr(self, "_authority_context", None)
        if context is not None:
            return context
        return build_authority_context(
            active_intake_snapshot_id=snapshot.snapshot_id,
            active_intake_snapshot_sha256=snapshot.snapshot_sha256,
            intake_rows=(item.to_dict() for item in snapshot.items),
            author_model=self.config.author_model,
            author_version=self.config.author_version,
            checker_model=self.config.checker_model,
            checker_version=self.config.checker_version,
        )

    def _refuse_lockless_admission(
        self, routes: "Sequence[tuple[IntakeItem, IntentRoute]]"
    ) -> None:
        """Refuse admission of any member routed to a lockless non-exempt intent (EN-2).

        Parameters
        ----------
        routes:
            Final per-member routes, including requeue re-targets.

        Raises
        ------
        DriverIntegrationError
            If ``config.allow_lockless_intents`` carries a stale entry (an
            intent that is actually locked in the live registry) or any member
            routes to an intent whose lock family is absent or lockless and
            not exempt.
        """

        intents = self.registry.intents
        for name in self.config.allow_lockless_intents:
            intent = intents.get(name)
            if intent is None:
                raise DriverIntegrationError(
                    f"allow-lockless names unknown environment intent {name!r}"
                )
            if intent.lock.status == "locked":
                raise DriverIntegrationError(
                    f"allow-lockless names intent {name!r}, which is LOCKED in the "
                    "live registry; the exemption is stale -- exemptions are "
                    "re-derived against the live registry, never trusted from a "
                    "recorded field"
                )
        allowed = frozenset(self.config.allow_lockless_intents)
        violations: list[str] = []
        for item, route in routes:
            intent = intents.get(route.intent)
            lock_status = getattr(getattr(intent, "lock", None), "status", None)
            if lock_status != "locked" and route.intent not in allowed:
                violations.append(
                    f"{item.stable_id} ({item.name!r}, zoo {item.zoo!r}) routes to "
                    f"{route.intent!r} whose lock family is "
                    f"{'absent' if intent is None else str(lock_status)}"
                )
        if violations:
            raise DriverIntegrationError(
                "lockless-route refusal (EN-2), PRE-AUTHOR: no member may admit "
                "onto an intent without a committed lock family; pass "
                "allow-lockless for a deliberately lockless intent or fix the "
                "routing/composition: " + "; ".join(violations)
            )

    def _ordered_work(
        self,
        snapshot: IntakeSnapshot,
        current: Mapping[str, JsonObject],
        requeues: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        refresh_ids: Optional[frozenset[str]] = None,
    ) -> tuple[WorkItem, ...]:
        """Route incomplete intake rows and enforce global phase order.

        ``refresh_ids`` is the WS-1 seam: a dependency-refresh work generation
        is minted ONLY for members the selection plan admitted through the
        named re-drive lane. The historical trigger -- projection absence over
        a recorded head -- was the award-eater's engine (vector 8b): every
        authority rotation silently re-generationed every settled member.
        ``None`` preserves the historical trigger for direct callers (tests,
        tools) that construct work without a selection plan.
        """

        routes: list[tuple[IntakeItem, IntentRoute]] = []
        discovery_urls = _intake_discovery_urls(snapshot)
        for item in snapshot.items:
            framework = _framework_from_intake(item)
            route = route_model(item.to_model_requirements(framework))
            target_intent = (requeues or {}).get(item.stable_id, {}).get("target_intent")
            if target_intent is not None:
                if not isinstance(target_intent, str) or target_intent not in self.registry.intents:
                    raise DriverIntegrationError(
                        f"requeue for {item.stable_id} names an unknown environment intent"
                    )
                target = self.registry.intents[target_intent]
                if target.phase is not route.phase:
                    raise DriverIntegrationError(
                        f"requeue for {item.stable_id} cannot cross environment phases"
                    )
                route = IntentRoute(item.stable_id, target.name, target.phase)
            routes.append((item, route))
        # EN-2: no member admits onto an intent without a committed lock
        # family unless the operator exempted that intent explicitly. This is
        # the PRE-AUTHOR door -- the pilot's m4714 paid a full author session
        # and then died at a lockless env turn. Lockless-ness is re-derived
        # from the LIVE registry here (never a recorded field), and a stale
        # exemption naming a locked intent refuses.
        self._refuse_lockless_admission(routes)
        ordered_routes = phase_routes(route for _item, route in routes)
        by_id = {item.stable_id: item for item, _route in routes}
        bindings = requeues or {}
        latest_history: dict[str, Mapping[str, Any]] = {}
        for record in reversed(scan_jsonl(self.paths.ledgers.models)):
            stable_id = str(record.get("stable_id", ""))
            if stable_id and stable_id not in latest_history:
                latest_history[stable_id] = record

        authority_epoch = authority_epoch_identity(self._refresh_authority_context(snapshot))

        def refresh_work_id(stable_id: str) -> Optional[str]:
            """Issue a fresh generation when dependency projection rejects prior history.

            The generation binds the live authority epoch (T1-06): every prior
            preimage field is rotation-invariant, so a relaunch that rotated the
            author/checker/policy authorities used to resume the SAME campaign
            root and let old-world gate history cap or fingerprint-terminalize
            the new-world proposal before any current gate saw it.
            """

            if refresh_ids is not None:
                if stable_id not in refresh_ids or stable_id not in latest_history:
                    return None
            elif stable_id in current or stable_id not in latest_history:
                return None
            return stable_hash(
                {
                    "kind": "dependency-refresh",
                    "stable_id": stable_id,
                    "prior_revision": latest_history[stable_id].get("record_revision"),
                    "intake_snapshot_sha256": snapshot.snapshot_sha256,
                    "authority_epoch": authority_epoch,
                }
            )

        def _active_binding(stable_id: str) -> Mapping[str, Any]:
            """Return the grant binding ONLY while its consumption is active.

            A spent-history binding (``active=False``) carries the LAST
            consumed generation's ``work_id`` for diagnostics; reading it here
            used to pin ``WorkItem.active_work_id`` to that obsolete identity,
            so an invalidated recorded authority was re-bought and re-discarded
            under the same work_id every leg (docket T2-13, m7305 bought the
            same stale generation six times). Inactive bindings still surface
            their ``grant_ids`` as recorded budget history below.
            """

            binding = bindings.get(stable_id, {})
            return binding if binding.get("active") else {}

        def _granted_attempts(stable_id: str) -> Optional[int]:
            """Return the active granted generation's validated attempts extension."""

            attempts = _active_binding(stable_id).get("attempts")
            if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 1:
                return attempts
            return None

        work = tuple(
            WorkItem(
                intake=by_id[route.stable_id],
                route=route,
                explicit_grants=tuple(bindings.get(route.stable_id, {}).get("grant_ids", ())),
                requeue_work_id=_active_binding(route.stable_id).get("work_id"),
                requeue_active=bool(bindings.get(route.stable_id, {}).get("active")),
                discovery_source_url=discovery_urls.get(route.stable_id),
                refresh_work_id=(
                    None
                    if _active_binding(route.stable_id).get("work_id") is not None
                    else refresh_work_id(route.stable_id)
                ),
                granted_attempts=_granted_attempts(route.stable_id),
            )
            for route in ordered_routes
        )
        if self.config.phase is not None:
            if self.config.phase == "native-tail" and any(
                item.route.phase.value == "pytorch" and item.stable_id not in current
                for item in work
            ):
                raise DriverIntegrationError(
                    "native-tail cannot start while PyTorch workflow rows remain"
                )
            work = tuple(item for item in work if item.route.phase.value == self.config.phase)
        if self.config.only_status is not None:
            selected = _only_status_selection(self.config.only_status)
            work = tuple(
                item
                for item in work
                if current.get(item.stable_id, {}).get("status", {}).get("code") in selected
            )
        return work

    def _requeue_grant_state(
        self,
        operational: JsonlLedger,
        intake_ids: frozenset[str],
        *,
        prior_member_ids: frozenset[str] = frozenset(),
    ) -> dict[str, JsonObject]:
        """Derive per-member grant facts WITHOUT consuming anything (WS-1).

        This is the pure half of grant reconciliation: it registers runtime
        grants on the canonical ledger, validates every grant, void, and
        historical consumption fail-closed, and returns the facts the
        work-selection predicate needs -- the live grant (if any), whether it
        can bind this head, the active consumed-but-unsettled generation (if
        any), and that generation's discharge state (clause 6). Consumption
        itself is a CONSEQUENCE of the selection plan
        (:meth:`_apply_selection_grants`), never a precondition: the step-100
        finishing legs consumed grants pre-selection and then excluded the
        members, leaving undischargeable in-flight identities.

        Grant facts bind the model's CURRENT TERMINAL REVISION -- the
        persisted supersession-lineage head of the append-only models ledger
        -- never the dependency-current projection: a grant exists precisely
        to re-adjudicate a reviewed terminal that may no longer replay under
        the current tree.

        Parameters
        ----------
        operational:
            Locked append-only operational ledger (read for consumptions).
        intake_ids:
            Exact trusted intake membership.
        prior_member_ids:
            Registered prior same-campaign snapshot members: their grants are
            DORMANT on a subset rung (validated, retained, never consumed).

        Returns
        -------
        dict[str, dict[str, Any]]
            Stable-ID keyed grant facts:
            ``record`` (raw terminal head), ``recorded_grants``,
            ``consumed`` (validated consumption events), ``active_event``,
            ``active_discharge`` (state, evidence), ``live_grant``,
            ``bindable``, ``hold_reason``, ``dormant``,
            ``release_authority`` (closed class behind a bindable grant:
            human-review, award-naming, or infra-waiver), and
            ``infra_release`` (the replay-derived
            :class:`~menagerie.crawler.infra_release.InfraTerminalRelease`
            when the infra-terminal waiver released the bind, else ``None``).
        """

        canonical_grants_path = canonical_requeue_grants_path(self.paths.ledgers.models)
        runtime_ledger = _validated_requeue_grants(
            self.paths.requeue_grants, intake_ids, prior_member_ids=prior_member_ids
        )
        if runtime_ledger.voids:
            raise DriverIntegrationError(
                "requeue voids must be appended to the canonical grant ledger"
            )
        for grant in runtime_ledger.grants:
            target_intent = grant.get("target_intent")
            if target_intent is not None and target_intent not in self.registry.intents:
                raise DriverIntegrationError(
                    f"requeue grant names an unknown environment intent: {target_intent}"
                )
            append_canonical_requeue_grant(canonical_grants_path, grant)
        canonical_ledger = _validated_requeue_grants(
            canonical_grants_path, intake_ids, prior_member_ids=prior_member_ids
        )
        grants = canonical_ledger.grants
        if any(
            grant.get("target_intent") is not None
            and grant.get("target_intent") not in self.registry.intents
            for grant in grants
        ):
            raise DriverIntegrationError("canonical requeue grant names an unknown intent")
        by_id = {str(grant["grant_id"]): grant for grant in grants}
        voided_ids = {str(void["void_of"]) for void in canonical_ledger.voids}
        canonical_events_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        canonical_events = scan_jsonl(canonical_events_path)
        consumed_events = [
            event
            for event in (*canonical_events, *operational.records)
            if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
        ]
        model_revisions = scan_jsonl(self.paths.ledgers.models)
        try:
            terminal_records = _select_current(model_revisions)
        except (LedgerCorruptionError, ReductionError) as exc:
            raise DriverIntegrationError(
                f"model ledger supersession lineage is invalid: {exc}"
            ) from exc

        # Infra-terminal waiver evidence (four-part doctrine part 3): the
        # attempts and gates ledgers are read lazily -- only a live grant on an
        # UNREVIEWED terminal consults them -- and re-scanned every leg, so
        # eligibility is always recomputed from ledger-bound records, never
        # from any persisted flag.
        infra_evidence: Optional[tuple[list[JsonObject], list[JsonObject]]] = None

        def _infra_release_evidence() -> tuple[list[JsonObject], list[JsonObject]]:
            nonlocal infra_evidence
            if infra_evidence is None:
                infra_evidence = (
                    scan_jsonl(self.paths.ledgers.attempts),
                    scan_jsonl(self.paths.ledgers.gates),
                )
            return infra_evidence
        consumed_by_id: dict[str, JsonObject] = {}
        # W31 single consumption: one adjudicated decision spends at most once
        # against one superseded head. Two same-key grants (pre-flock race
        # residue) each consumed against the SAME parent revision are one
        # decision spent twice; a same-key consumption against a LATER head is
        # the sanctioned T1-13b re-adjudication and stays legal. Quantified
        # over grants that HAVE the key (v2 era rule).
        adjudicated_spends: dict[tuple[str, str], str] = {}
        for event in consumed_events:
            details = event.get("details", {})
            grant_id = str(details.get("grant_id", ""))
            bound_grant = by_id.get(grant_id)
            if bound_grant is None:
                raise DriverIntegrationError(
                    f"requeue consumption references an unknown grant: {grant_id}"
                )
            if details.get("stable_id") != bound_grant.get("stable_id"):
                raise DriverIntegrationError("requeue consumption stable_id mismatch")
            if any(
                details.get(field) != bound_grant.get(field)
                for field in (
                    "stage",
                    "reason",
                    "attempts",
                    "target_intent",
                    # AW-1's award-naming seam is checked here or nowhere: the
                    # generation arm reads the admitting authority out of this
                    # receipt, a producer-written copy, so a receipt naming an
                    # award its durable grant never named would be award-work
                    # authority nobody granted (round-2 break 5, doctrine part 3).
                    "blocked_award_revision",
                )
            ):
                raise DriverIntegrationError("requeue consumption grant facts mismatch")
            generation = details.get("new_work_generation")
            source_revision = details.get("source_record_revision")
            if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
                raise DriverIntegrationError("requeue consumption generation is invalid")
            expected_work_id = stable_hash(
                {
                    "stable_id": bound_grant["stable_id"],
                    "grant_id": grant_id,
                    "parent_revision": source_revision,
                    "generation": generation,
                }
            )
            if details.get("new_work_id") != expected_work_id:
                raise DriverIntegrationError("requeue consumption new-work identity mismatch")
            stable_revisions = [
                revision
                for revision in model_revisions
                if revision.get("stable_id") == bound_grant.get("stable_id")
            ]
            introducing = next(
                (
                    revision
                    for revision in stable_revisions
                    if grant_id in revision.get("budget", {}).get("explicit_grants", [])
                    and (
                        revision.get("parent_revision") is None
                        or not any(
                            parent.get("record_revision") == revision.get("parent_revision")
                            and grant_id in parent.get("budget", {}).get("explicit_grants", [])
                            for parent in stable_revisions
                        )
                    )
                ),
                None,
            )
            expected_source = (
                introducing.get("parent_revision")
                if introducing is not None
                else terminal_records.get(str(bound_grant["stable_id"]), {}).get("record_revision")
            )
            if source_revision != expected_source:
                raise DriverIntegrationError(
                    "requeue consumption does not bind the exact superseded parent revision"
                )
            prior = consumed_by_id.get(grant_id)
            if prior is not None and prior.get("details") != event.get("details"):
                raise DriverIntegrationError(f"conflicting requeue consumption for {grant_id}")
            adjudication_key = bound_grant.get("adjudication_key")
            if isinstance(adjudication_key, str):
                spend = (adjudication_key, str(source_revision))
                first_spender = adjudicated_spends.setdefault(spend, grant_id)
                if first_spender != grant_id:
                    raise DriverIntegrationError(
                        "second consumption recorded for one adjudicated decision: "
                        f"grants {first_spender} and {grant_id} both spent "
                        f"adjudication {adjudication_key} against {source_revision}"
                    )
            consumed_by_id[grant_id] = event

        discharge_events = [
            event
            for event in (*canonical_events, *operational.records)
            if event.get("event_kind") == OperationalEventKind.REQUEUE_GENERATION_DISCHARGED.value
        ]
        # Unbound duck-typed callers reach this through the compat spelling,
        # so the sibling helper is invoked through the mixin class explicitly.
        live_park_work_ids = AdmissionEnvironmentMixin._live_park_generation_work_ids(
            self, terminal_records
        )

        state: dict[str, JsonObject] = {}
        grouped: dict[str, list[JsonObject]] = defaultdict(list)
        for grant in grants:
            grouped[str(grant["stable_id"])].append(grant)
        for stable_id, model_grants in grouped.items():
            record = terminal_records.get(stable_id)
            recorded_grants = (
                list(record.get("budget", {}).get("explicit_grants", [])) if record else []
            )
            unknown_recorded = set(recorded_grants) - set(by_id)
            if unknown_recorded:
                raise DriverIntegrationError(
                    f"canonical model references unknown requeue grants: {sorted(unknown_recorded)}"
                )
            consumed = [
                consumed_by_id[str(grant["grant_id"])]
                for grant in model_grants
                if str(grant["grant_id"]) in consumed_by_id
            ]
            active = [
                event
                for event in consumed
                if event.get("details", {}).get("grant_id") not in recorded_grants
            ]
            if len(active) > 1:
                raise DriverIntegrationError(
                    f"multiple active requeue generations exist for {stable_id}"
                )
            dormant = stable_id not in intake_ids
            # DORMANT (T4-T1 membership half, grant lane): a REGISTERED PRIOR
            # same-campaign member's grants and voids are read-only HISTORY on
            # this subset rung -- retained, never consumed, never invalidating,
            # never bound to work. The integrity raises above still guard the
            # history itself.
            active_event = active[0] if active else None
            active_discharge: Optional[tuple[Optional[str], Optional[str]]] = None
            if active_event is not None:
                discharge, evidence = derive_generation_discharge(
                    active_event.get("details", {}),
                    model_revisions=model_revisions,
                    live_park_work_ids=live_park_work_ids,
                    discharge_events=discharge_events,
                    all_consumptions=consumed_events,
                    voided_grant_ids=frozenset(voided_ids),
                )
                active_discharge = (
                    discharge.value if discharge is not None else None,
                    evidence,
                )
            unconsumed = [
                grant
                for grant in model_grants
                if str(grant["grant_id"]) not in consumed_by_id
                and str(grant["grant_id"]) not in voided_ids
            ]
            live_grant = unconsumed[0] if unconsumed and active_event is None else None
            hold_reason: Optional[str] = None
            bindable = False
            release_authority: Optional[str] = None
            infra_release = None
            machine_class = None
            if live_grant is not None:
                if record is not None and _award_naming_grant(live_grant, record):
                    # AW-1's award-naming lane, THE policy's own disjunct
                    # (work_selection_predicate clause 3, called -- never
                    # re-implemented): a grant naming the exact blocked award
                    # revision IS the binding authority, and a runs head is
                    # not a review-held terminal, so the reviewed-terminal
                    # rule below can never admit it.
                    bindable = True
                    release_authority = RELEASE_AUTHORITY_AWARD_NAMING
                else:
                    reviewed = record is not None and bool(
                        record.get("status", {}).get("human_review", {}).get("required")
                    )
                    if record is not None and not reviewed:
                        # KERNEL_DECISION 2026-08-21 (four-part waiver): an
                        # unreviewed terminal binds ONLY when it is provably a
                        # claim-less machine-infrastructure death -- the review
                        # requirement has no consumer. Recomputed here every
                        # leg from ledger-bound attempt records; one non-infra
                        # witnessing attempt and the hold stands.
                        attempt_rows, gate_rows = _infra_release_evidence()
                        infra_release = derive_infra_terminal_release(
                            record, attempt_rows, gate_rows
                        )
                        if infra_release is None:
                            # W14 signed class widenings: the same fail-closed
                            # identity binding, per-class closed proof rules.
                            # Evidence is loaded from its parent-written homes;
                            # a class whose evidence has no emitter yet simply
                            # never derives (era rows hold for humans).
                            from menagerie.crawler.quota_deferral import (
                                load_quota_deferral_rows,
                            )

                            machine_class = derive_machine_class_release(
                                record,
                                attempt_rows,
                                gate_rows,
                                # Unbound duck-typed callers reach this through
                                # the compat spelling, so the sibling helper is
                                # invoked through the mixin class explicitly.
                                valve_rows=load_quota_deferral_rows(
                                    AdmissionEnvironmentMixin._quota_deferral_ledger_path(
                                        self
                                    )
                                ),
                                invalidation_events=(
                                    *canonical_events,
                                    *operational.records,
                                ),
                            )
                    if record is None or (
                        not reviewed and infra_release is None and machine_class is None
                    ):
                        hold_reason = "unreviewed-terminal"
                    elif not _grant_stage_binds_terminal(live_grant, record):
                        hold_reason = "stage-mismatch"
                    else:
                        bindable = True
                        if reviewed:
                            release_authority = RELEASE_AUTHORITY_HUMAN_REVIEW
                        elif infra_release is not None:
                            release_authority = RELEASE_AUTHORITY_INFRA_WAIVER
                        else:
                            release_authority = (
                                MACHINE_CLASS_RELEASE_AUTHORITY_PREFIX
                                + machine_class.release_class
                            )
            state[stable_id] = {
                "record": record,
                "recorded_grants": recorded_grants,
                "consumed": consumed,
                "active_event": active_event,
                "active_discharge": active_discharge,
                "live_grant": live_grant,
                "bindable": bindable,
                "hold_reason": hold_reason,
                "dormant": dormant,
                "release_authority": release_authority if bindable else None,
                "infra_release": (
                    infra_release
                    if bindable and release_authority == RELEASE_AUTHORITY_INFRA_WAIVER
                    else None
                ),
                "machine_class_release": (
                    machine_class
                    if bindable
                    and release_authority is not None
                    and release_authority.startswith(MACHINE_CLASS_RELEASE_AUTHORITY_PREFIX)
                    else None
                ),
            }
        return state

    def _live_park_generation_work_ids(
        self, terminal_records: Mapping[str, JsonObject]
    ) -> frozenset[str]:
        """Return work IDs held by LIVE operator-owned park rows.

        Feeds clause 6 (a park row IS a discharge) and the selection facts:
        grant-pending and terminal-refusal parks are operator property until
        released by a landed revision or an explicit void. Checker-lane rows
        are deliberately absent -- their documented contract is leg-scoped
        withholding with requeue on the next run.
        """

        released = {
            GRANT_PENDING_DISPOSITION: grant_park_released,
            # R2-1: a live exclusion park is operator property exactly like a
            # grant-pending park; its (synthetic) work id discharges nothing
            # real, but the uniform rule keeps the two ledger families one.
            SELECTION_EXCLUSION_DISPOSITION: grant_park_released,
            REFUSAL_PARK_DISPOSITION: refusal_park_released,
        }
        work_ids: set[str] = set()
        for stable_id, rows in _persisted_deferral_rows(self).items():
            for row in rows:
                is_released = released.get(str(row.get("disposition")))
                if is_released is None:
                    continue
                if machine_deferral_rejection(stable_id, row) is not None:
                    continue
                if is_released(row, terminal_records.get(stable_id)):
                    continue
                work_id = row.get("work_id")
                if isinstance(work_id, str) and work_id:
                    work_ids.add(work_id)
        return frozenset(work_ids)

    def _apply_selection_grants(
        self,
        grant_state: Mapping[str, JsonObject],
        admitted_granted: frozenset[str],
        operational: JsonlLedger,
        *,
        emit_holds: bool = True,
    ) -> dict[str, JsonObject]:
        """Consume grants for plan-admitted members and return work bindings.

        This is the side-effecting half of grant reconciliation, run strictly
        AFTER the selection plan (WS-1 clause 3): grant consumption and the
        work-generation mint happen atomically here, only for members the
        plan admitted through the granted lane. Live-but-unbindable grants
        are disclosed durably (the T1-13 model-local hold), and spent-history
        bindings are always returned for budget threading.

        Parameters
        ----------
        grant_state:
            :meth:`_requeue_grant_state` output.
        admitted_granted:
            Stable IDs the selection plan admitted as ``granted-redrive``.
        operational:
            Locked append-only operational ledger.
        emit_holds:
            Whether to disclose model-local holds for live-but-unbindable
            grants. Scope-excluded modes (``--only-unsettled``) never touch
            grants and emit nothing.

        Returns
        -------
        dict[str, dict[str, Any]]
            Stable-ID keyed grant history, work identity, and active-work
            marker -- the exact ``_ordered_work`` bindings shape.
        """

        result: dict[str, JsonObject] = {}
        for stable_id, facts in grant_state.items():
            if facts["dormant"]:
                continue
            record = facts["record"]
            recorded_grants = list(facts["recorded_grants"])
            consumed = list(facts["consumed"])
            active_event = facts["active_event"]
            live_grant = facts["live_grant"]
            if active_event is not None:
                discharge_state = (facts.get("active_discharge") or (None, None))[0]
                if discharge_state is None:
                    details = active_event["details"]
                    result[stable_id] = {
                        "grant_ids": [*recorded_grants, str(details["grant_id"])],
                        "work_id": str(details["new_work_id"]),
                        "active": True,
                        "target_intent": details.get("target_intent"),
                        "attempts": details.get("attempts"),
                    }
                    continue
                # A discharged generation is settled work-identity history
                # (clause 6): it never replays, and its budget history stays
                # visible exactly like a recorded spent grant.
                details = active_event["details"]
                result[stable_id] = {
                    "grant_ids": [*recorded_grants, str(details["grant_id"])],
                    "work_id": str(details["new_work_id"]),
                    "active": False,
                }
                continue

            if live_grant is not None and stable_id in admitted_granted:
                if not facts["bindable"]:
                    raise DriverIntegrationError(
                        f"selection admitted {stable_id} on an unbindable grant"
                    )
                generation = len(consumed) + 1
                new_work_id = stable_hash(
                    {
                        "stable_id": stable_id,
                        "grant_id": live_grant["grant_id"],
                        "parent_revision": record["record_revision"],
                        "generation": generation,
                    }
                )
                event_details: JsonObject = {
                    "grant_id": live_grant["grant_id"],
                    "stable_id": stable_id,
                    "stage": live_grant["stage"],
                    "reason": live_grant["reason"],
                    "attempts": live_grant["attempts"],
                    "source_record_revision": record["record_revision"],
                    "new_work_generation": generation,
                    "new_work_id": new_work_id,
                }
                if live_grant.get("target_intent") is not None:
                    event_details["target_intent"] = live_grant["target_intent"]
                if live_grant.get("blocked_award_revision") is not None:
                    # AW-1: the consumption receipt must carry the named award
                    # revision -- the reducer's grant-fact conformance compares
                    # it against the durable grant row.
                    event_details["blocked_award_revision"] = live_grant["blocked_award_revision"]
                release_authority = facts.get("release_authority")
                if release_authority is not None:
                    # Four-part waiver doctrine part 4: DISCLOSURE only. The
                    # census recomputes the class from ledgers on every read;
                    # nothing ever reads this field back as authority.
                    event_details["release_authority"] = release_authority
                infra_release = facts.get("infra_release")
                if infra_release is not None:
                    event_details["infra_release_witness_bases"] = list(
                        infra_release.witness_bases
                    )
                machine_class_release = facts.get("machine_class_release")
                if machine_class_release is not None:
                    # Same disclosure discipline as the base waiver: written
                    # for the census, never read back as authority.
                    event_details["machine_class_witness_bases"] = list(
                        machine_class_release.witness_bases
                    )
                event = {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": self.dependencies.clock(),
                    "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
                    "status": OperationalEventStatus.REQUEUE_GRANT_CONSUMED.value,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 1},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": event_details,
                }
                event = mint_operational_event(
                    event,
                    identity_payload={"grant_id": str(live_grant["grant_id"])},
                )
                operational.append(event)
                self.dependencies.boundary_hook("after-requeue-consume", stable_id)
                result[stable_id] = {
                    "grant_ids": [*recorded_grants, str(live_grant["grant_id"])],
                    "work_id": new_work_id,
                    "active": True,
                    "target_intent": live_grant.get("target_intent"),
                    "attempts": live_grant.get("attempts"),
                }
                continue

            if live_grant is not None and emit_holds and facts["hold_reason"] is not None:
                # T1-13 (blast scope): the refusal is model-local -- the grant
                # simply cannot bind THIS head -- so it discloses durably and
                # holds this model while the leg proceeds. The grant stays
                # outstanding: the operator voids it (or supersedes the head)
                # and the next leg re-evaluates.
                hold_event: JsonObject = {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": self.dependencies.clock(),
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
                        "disposition": "requeue-grant-review-hold",
                        "stable_id": stable_id,
                        "grant_id": str(live_grant["grant_id"]),
                        "hold_reason": facts["hold_reason"],
                        "grant_stage": live_grant.get("stage"),
                        "terminal_stage": (
                            record.get("status", {}).get("stage") if record is not None else None
                        ),
                        "terminal_kind": (
                            record.get("status", {}).get("kind") if record is not None else None
                        ),
                    },
                }
                hold_event = mint_operational_event(hold_event)
                operational.append(hold_event)

            if recorded_grants:
                latest = consumed[-1]["details"] if consumed else None
                result[stable_id] = {
                    "grant_ids": recorded_grants,
                    "work_id": str(latest["new_work_id"]) if latest is not None else None,
                    "active": False,
                }
        return result

    def _consume_requeue_grants(
        self,
        operational: JsonlLedger,
        intake_ids: frozenset[str],
        *,
        prior_member_ids: frozenset[str] = frozenset(),
    ) -> dict[str, JsonObject]:
        """Validate grants and durably bind at most one active generation per model.

        Compatibility spelling over the WS-1 split: derive the pure grant
        state, then consume every bindable live grant, exactly the historical
        behavior. The DRIVER no longer calls this pre-selection -- it builds a
        selection plan first and consumes only for plan-admitted members --
        but direct callers (tests, tools) keep the one-shot semantics.

        Parameters
        ----------
        operational, intake_ids, prior_member_ids:
            Exactly as :meth:`_requeue_grant_state`.

        Returns
        -------
        dict[str, dict[str, Any]]
            Stable-ID keyed grant history, work identity, and active-work marker.
        """

        # Called unbound with duck-typed stub drivers by tests and tools, so
        # the split halves are invoked through the mixin class explicitly.
        grant_state = AdmissionEnvironmentMixin._requeue_grant_state(
            self, operational, intake_ids, prior_member_ids=prior_member_ids
        )
        admitted = frozenset(
            stable_id
            for stable_id, facts in grant_state.items()
            if not facts["dormant"] and facts["live_grant"] is not None and facts["bindable"]
        )
        return AdmissionEnvironmentMixin._apply_selection_grants(
            self, grant_state, admitted, operational
        )

    def _author_attempt_ladders_for(
        self,
        work: Sequence[WorkItem],
        attempts_per_round: Optional[int],
    ) -> Optional[dict[str, _RetryLadderState]]:
        """Resolve each model's cross-round retry-ladder continuation (ACAP).

        In-memory and disposable: the durable allowance and paid-session fuses
        in the ledgers remain the only money truth; a continuation remembers
        only WHERE a capped ladder stopped so a later round resumes the same
        cumulative counters instead of restarting the ladder. A rotated work
        generation always starts fresh.

        Parameters
        ----------
        work:
            Scheduled work items for this pooled author call.
        attempts_per_round:
            Per-call attempt cap, or ``None`` for the uncapped historical lane.

        Returns
        -------
        dict[str, _RetryLadderState] | None
            Stable-id keyed continuations, or ``None`` when uncapped.
        """

        if attempts_per_round is None:
            return None
        ladders_by_id = getattr(self, "_author_attempt_ladders", None)
        if ladders_by_id is None:
            ladders_by_id = {}
            self._author_attempt_ladders: dict[str, _RetryLadderState] = ladders_by_id
        resolved: dict[str, _RetryLadderState] = {}
        for item in work:
            ladder = ladders_by_id.get(item.stable_id)
            if ladder is None or ladder.work_id != item.active_work_id:
                ladder = _RetryLadderState(
                    work_id=item.active_work_id, per_round_cap=attempts_per_round
                )
                ladders_by_id[item.stable_id] = ladder
            else:
                ladder.per_round_cap = attempts_per_round
            resolved[item.stable_id] = ladder
        return resolved

    def _author_attempt_ladders_pop(self, stable_id: str) -> None:
        """Drop one resolved model's cross-round ladder continuation, if any.

        Parameters
        ----------
        stable_id:
            Model whose author lane just resolved (artifact or terminal).
        """

        ladders_by_id = getattr(self, "_author_attempt_ladders", None)
        if ladders_by_id is not None:
            ladders_by_id.pop(stable_id, None)

    def _ensure_authors(
        self,
        work: Sequence[WorkItem],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        attempts_per_round: Optional[int] = None,
    ) -> dict[str, AuthorArtifact]:
        """Create or reload one privately staged typed author result per model.

        Author *sessions* run concurrently, up to ``config.author_concurrency``.
        Everything else -- staging, ledger appends, artifact publication, terminal
        routing, the returned ``artifacts`` map -- stays on this single writer, in
        work order, exactly as it was when the lane was serial. Completion order can
        therefore never influence what is recorded.

        With ``attempts_per_round`` set (ACAP; the pipelined scheduler's
        per-round cap), a model whose retry ladder neither resolves nor
        terminalizes within that many attempts is returned UNRESOLVED -- no
        ledger append, no artifact, no terminal -- and named in
        ``self._author_wave_deferred_ids``; its cross-round continuation in
        ``self._author_attempt_ladders`` resumes the same ladder, exactly
        where it stopped, in a later call. ``None`` keeps the historical
        resolve-inside-one-call behavior for the wave driver and direct
        callers.
        """

        by_stable_id = {item.stable_id: item for item in work}
        self._author_wave_deferred_ids: set[str] = set()
        # Resolved on the writer BEFORE any dispatch: pool workers only ever
        # READ this mapping, each ladder is mutated by exactly one worker, and
        # the writer reads it back only after that worker's future completed.
        attempt_ladders = self._author_attempt_ladders_for(work, attempts_per_round)
        pool = _AuthorWavePool(
            max(1, int(self.config.author_concurrency)),
            lambda item, request_admission: self._retry_infrastructure_call(
                lambda: self.dependencies.author.author(
                    item,
                    self.paths.work_root,
                    self.config,
                    reducer.context,
                ),
                admission=("author", item),
                operational=operational,
                author_admission=request_admission,
                author_ladder=(
                    None if attempt_ladders is None else attempt_ladders.get(item.stable_id)
                ),
            ),
            lambda item, attempt_group: self._admit_author_spend(
                item, operational, attempt_group=attempt_group
            ),
            monotonic=self.dependencies.monotonic,
        )
        self._author_spend_artifact_events = reducer.artifact_ledger.events
        self._author_spend_operational = operational
        outage_failure: Optional[BaseException] = None
        drained: tuple[tuple[str, _AuthorDispatchOutcome], ...] = ()
        try:
            # The writer's own context carries the wave's bounded relay drain,
            # so every writer-side wait inside this scope (inline operator
            # commands, retry backoffs) services queued worker buys. Worker
            # threads never see the binding; the reserve site stays writer-only.
            with _author_admission_relay_service(pool.service_admissions):
                if pool.enabled:
                    for item in work:
                        # A requested shutdown must not fan out new sessions. Workers
                        # already re-check this at their own admission boundary, so this
                        # only avoids paying for work the driver is about to abandon.
                        if self._shutdown_event.is_set():
                            break
                        if self._author_session_is_certain(item, reducer):
                            pool.submit(item)
                return self._commit_author_wave(
                    work, pool, reducer, operational, state, attempt_ladders=attempt_ladders
                )
        except AuthorUsagePause as exc:
            # BL-8 (d): a usage pause is a decision to STOP SPENDING. Everything
            # already started is drained and preserved below, but the held and
            # queued admissions past the declared running+lookahead window are
            # cancelled instead of bought -- the historical drain-all here
            # converted an early pause into a whole cohort of continued spend.
            pool.halt(
                trigger="usage-limit-halt",
                reason=f"author usage pause: {exc.reason}",
            )
            raise
        except (ResearchToolsUnavailableError, ForgeRateLimitedError) as exc:
            # The outage arms deliberately keep the full drain: the drained
            # outcomes are the consecutive-failure EVIDENCE the outage
            # promotion below consumes, and a downed tool fails sessions
            # quickly rather than buying provider budget.
            outage_failure = exc
        except BaseException as exc:
            trigger = (
                "retry-exhaustion-halt"
                if isinstance(exc, AuthorSessionRetryExhausted)
                else (
                    "breaker-halt"
                    if isinstance(exc, (AuthorEffortCapExceeded, CheckerLegBudgetExceeded))
                    else "campaign-halt"
                )
            )
            pool.halt(
                trigger=trigger,
                reason=f"{type(exc).__name__}: {str(exc)[:384]}",
            )
            raise
        finally:
            # A started session has already spent provider budget, so an aborting
            # wave never orphans one: drain every sibling, then keep whatever it
            # produced in the disposable per-model cache so the next run reloads it
            # instead of re-authoring. This writes no ledger record and publishes
            # nothing, so an abort's canonical sequence stays identical to serial.
            drained = pool.drain()
            for stable_id, outcome in drained:
                preserved = by_stable_id.get(stable_id)
                if preserved is not None and outcome.error is None and outcome.artifact is not None:
                    self._preserve_uncommitted_author_result(preserved, outcome.artifact)
            pool.close()
        if outage_failure is None:
            raise AssertionError("author outage unwind lost its typed cause")
        outage = _author_outage_signal(outage_failure, drained)
        if outage is not None:
            raise AuthorUsagePause(
                self._pause_for_usage(outage, operational, len(work))
            ) from outage_failure
        raise outage_failure

    def _author_session_is_certain(self, item: WorkItem, reducer: CanonicalReducer) -> bool:
        """Return whether the serial path provably reaches a fresh author session.

        This is an exactness gate, not a heuristic: it is ``True`` only when every
        earlier resolution the serial lane tries -- finalized canonical authority, a
        usable family representative, and the disposable on-disk result cache -- is
        already ruled out on inputs that no sibling in this wave can change. A
        false negative only costs concurrency for one model; a false positive would
        burn a session the serial lane would never have run, so every check errs
        toward ``False``.

        Parameters
        ----------
        item:
            Exact scheduled work generation.
        reducer:
            Locked reducer exposing canonical authority for the wave.

        Returns
        -------
        bool
            Whether this model may be dispatched ahead of its commit turn.
        """

        if reducer.artifact_ledger.event_count and reducer.artifact_ledger.has_final_event(
            item.stable_id, item.active_work_id
        ):
            return False
        try:
            operational = self._author_spend_operational
            enforce_author_session_allowance(
                (*reducer.artifact_ledger.events, *operational.records),
                item,
                operational_events=operational.records,
            )
        except AuthorEffortCapExceeded:
            # The serial commit lane raises the typed allowance exhaustion for
            # this model (T2-07/T1-13); a speculative session here would burn
            # exactly the spend the derived cap exists to stop re-buying.
            return False
        if item.is_family_variant and reducer.has_current(item.family_representative_id):
            # Either the representative seeds this variant with no session at all, or
            # the serial lane raises on unusable representative authority. Both are
            # decided by the writer, never by a speculative session.
            return False
        if _still_valid_deferral_row(self, item) is not None:
            # The commit lane reclaims the durable valve row without a session;
            # dispatching one speculatively would re-buy exactly the author
            # spend the row exists to save (the rung-10 resume tax).
            return False
        cache = author_result_cache_path(self.paths.work_root, item.stable_id)
        if item.requeue_work_id is not None:
            # A granted re-drive runs a NEW work generation; an on-disk cache
            # is stale by construction (the serial lane discards it --
            # `author-cache-discarded`) except the narrow case where an
            # earlier leg of THIS generation preserved a drained result. The
            # cache records no generation binding, so the two are not cheaply
            # distinguishable here. Erring certain costs at most one bounded
            # speculative session in the narrow case; erring uncertain routed
            # EVERY re-drive to the serial commit lane and collapsed the
            # release wave to 4-6 live sessions against 68 queued models
            # (step-200 leg 21/22, 2026-08-21).
            return True
        return not cache.is_file()

    def _withhold_for_environment_coverage(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        operational: JsonlLedger,
    ) -> bool:
        """Withhold a model whose routed intent cannot install its library, and say so.

        Routing is fixed at intake from the roster zoo; the distribution is
        declared later by the author. When they disagree the model reaches an
        environment that simply does not have its library, and the R1 digest
        refusal that follows is CORRECT -- it stays exactly as it is, and a
        withheld model still never runs, still gets no digest, and still asserts
        no rung.

        What changes is the record. Terminalizing ``failed:runner`` /
        ``protocol-violation`` asserted that our pipeline broke on the model,
        which is false: the author is right, the environment is honestly missing
        the package, and another intent carries it on purpose. So the model is
        withheld -- not terminalized -- and recorded in the durable
        environment-coverage deferral ledger naming the intent that would cover
        it, exactly as ``_admit_within_host_capacity`` records a model this host
        is too small for.

        The disposable author-result cache is written first, so a resume
        re-derives the same deferral from the cached result instead of spending a
        second agentic session on a model it already knows it cannot route.

        Parameters
        ----------
        item:
            Exact scheduled work generation.
        artifact:
            Validated, staged author artifact carrying the declarative recipe.
        operational:
            Locked operational ledger for the health event.

        Returns
        -------
        bool
            Whether the model was withheld from this wave.
        """

        try:
            proposal = artifact.proposal
        except DriverIntegrationError:
            # A terminal author result has no executable proposal and declares no
            # recipe; there is no routed distribution to assess.
            return False
        facts = proposal.get("proposed_facts")
        implementation = facts.get("implementation") if isinstance(facts, Mapping) else None
        if not isinstance(implementation, Mapping):
            return False
        registry = getattr(self, "registry", None)
        assessment = assess_environment_coverage(
            implementation,
            routed_intent=item.route.intent,
            routed_packages=_routed_environment_packages(registry, item.route.intent),
            registry=registry,
        )
        if not assessment.deferred:
            return False
        cache = author_result_cache_path(self.paths.work_root, item.stable_id)
        try:
            _write_json_atomic(
                cache,
                serialize_author_result_cache(
                    artifact.author_result,
                    source_manifest=artifact.source_manifest,
                    model_dir=artifact.model_dir,
                    staged_transaction_id=(
                        str(artifact.staged.transaction_id)
                        if artifact.staged is not None
                        else None
                    ),
                    lifecycle=(
                        "staged-pending" if artifact.staged is not None else "unstaged"
                    ),
                ),
            )
        except Exception:  # noqa: BLE001 -- a disposable cache must never block the record
            cache.unlink(missing_ok=True)
        self._record_environment_coverage_deferral(item, assessment, operational)
        return True

    def _record_environment_coverage_deferral(
        self,
        item: WorkItem,
        assessment: CoverageAssessment,
        operational: JsonlLedger,
    ) -> None:
        """Append one durable coverage deferral row plus its campaign-health event."""

        created_at = self.dependencies.clock()
        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = coverage_deferral_path(records_root)
        # BL-9d: bind the deferral to the exact routed inventory it was
        # adjudicated against, so a rotation's deferral blast radius is
        # auditable per environment generation.
        registry = getattr(self, "registry", None)
        intents = getattr(registry, "intents", None) if registry is not None else None
        routed = intents.get(item.route.intent) if isinstance(intents, Mapping) else None
        routed_export_sha256 = None
        if routed is not None and routed.lock.status == "locked" and routed.lock.export_bytes:
            routed_export_sha256 = hash_bytes(routed.lock.export_bytes)
        persisted = append_coverage_deferral_row(
            ledger_path,
            build_coverage_deferral_row(
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                name=item.intake.name,
                campaign_id=bound_campaign_id(self.config),
                run_id=self.config.run_id,
                machine_id=self.config.machine_id,
                created_at=created_at,
                assessment=assessment,
                routed_export_sha256=routed_export_sha256,
            ),
        )
        _stash_machine_deferral_row(self, item.stable_id, persisted)
        library = assessment.library
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": item.route.intent,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "environment-coverage-deferred",
                "ledger": str(ledger_path.relative_to(records_root)),
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "routed_intent": item.route.intent,
                "distribution": library.distribution if library is not None else None,
                "covering_intents": [entry.intent for entry in assessment.covering],
            },
        }
        # The typed per-occurrence identity covers the complete payload and
        # created_at: a later observation appends, while an exact byte replay no-ops.
        event = mint_operational_event(event)
        operational.append(event)

    def _record_environment_rebuild_deferral(
        self,
        item: WorkItem,
        *,
        intent: EnvironmentIntent,
        quarantine_event: Mapping[str, Any],
        quarantined_environment: Optional[EnvironmentBinding],
        artifact_identity: str,
        artifact_state: JsonObject,
        operational: JsonlLedger,
    ) -> None:
        """Park one work identity behind a quarantined environment generation.

        Appends one durable rebuild deferral row plus its campaign-health
        event. The model is deliberately NOT terminalized: no build ran, the
        quarantine names an infrastructure failure during cleanup, and the
        remedy (recover or rotate the environment, or record an explicit
        clearance receipt, then re-drive) is a campaign-level action the row
        records verbatim.

        Parameters
        ----------
        item:
            Exact scheduled work generation the quarantine refused.
        intent:
            Quarantined environment intent.
        quarantine_event:
            The matched ``environment-cleanup-quarantined`` campaign-health event.
        quarantined_environment:
            Environment binding the quarantine event recorded, when complete.
        artifact_identity:
            Typed committed setup-artifact identity the quarantine matched on.
        artifact_state:
            Typed per-member setup-artifact state vector behind the identity.
        operational:
            Locked operational ledger for the health event.
        """

        created_at = self.dependencies.clock()
        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = rebuild_deferral_path(records_root)
        details = quarantine_event.get("details", {})
        recorded_generation = details.get("env_generation")
        env_generation = (
            quarantined_environment.env_generation
            if quarantined_environment is not None
            else (recorded_generation if isinstance(recorded_generation, str) else None)
        )
        failure_type = details.get("failure_type")
        persisted = append_rebuild_deferral_row(
            ledger_path,
            build_rebuild_deferral_row(
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                name=item.intake.name,
                campaign_id=bound_campaign_id(self.config),
                run_id=self.config.run_id,
                machine_id=self.config.machine_id,
                created_at=created_at,
                quarantine_event_id=str(quarantine_event.get("event_id")),
                intent=intent.name,
                target=intent.lock.target,
                artifact_identity=artifact_identity,
                artifact_state=artifact_state,
                env_generation=env_generation,
                failure_type=failure_type if isinstance(failure_type, str) else None,
            ),
        )
        _stash_machine_deferral_row(self, item.stable_id, persisted)
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": intent.name,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": ENVIRONMENT_REBUILD_DEFERRED_DISPOSITION,
                "ledger": str(ledger_path.relative_to(records_root)),
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "intent": intent.name,
                "target": intent.lock.target,
                "quarantine_event_id": quarantine_event.get("event_id"),
                "env_generation": env_generation,
                "recheck_hint": persisted["recheck_hint"],
            },
        }
        # The typed per-occurrence identity covers the complete payload and
        # created_at: a later observation appends, while an exact byte replay no-ops.
        event = mint_operational_event(event)
        operational.append(event)

    def _record_disk_floor_deferral(
        self,
        item: WorkItem,
        *,
        intent: EnvironmentIntent,
        failure: DiskFloorError,
        operational: JsonlLedger,
    ) -> None:
        """Park one work identity behind this host's unmet environment disk floor.

        Appends one durable disk-floor deferral row plus its campaign-health
        event. The model is deliberately NOT terminalized: no build ran, the
        refusal names a host measurement, and the remedy (free disk on this
        host, or re-drive on any host with disk) is a machine-level action the
        row records verbatim.

        Parameters
        ----------
        item:
            Exact scheduled work generation the floor refused.
        intent:
            Routed environment intent the floor refused to serve.
        failure:
            Typed floor refusal carrying the floor, measurement, and root.
        operational:
            Locked operational ledger for the health event.
        """

        created_at = self.dependencies.clock()
        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = disk_floor_deferral_path(records_root)
        persisted = append_disk_floor_deferral_row(
            ledger_path,
            build_disk_floor_deferral_row(
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                name=item.intake.name,
                campaign_id=bound_campaign_id(self.config),
                run_id=self.config.run_id,
                machine_id=self.config.machine_id,
                created_at=created_at,
                intent=intent.name,
                target=intent.lock.target,
                env_root=failure.env_root,
                floor_bytes=failure.floor_bytes,
                free_bytes=failure.free_bytes,
                condition=getattr(failure, "condition", "floor"),
                destination_role=getattr(failure, "destination_role", None),
            ),
        )
        _stash_machine_deferral_row(self, item.stable_id, persisted)
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": intent.name,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": HOST_DISK_FLOOR_DEFERRED_DISPOSITION,
                "ledger": str(ledger_path.relative_to(records_root)),
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "intent": intent.name,
                "target": intent.lock.target,
                "env_root": failure.env_root,
                "floor_bytes": failure.floor_bytes,
                "free_bytes": failure.free_bytes,
                "condition": getattr(failure, "condition", "floor"),
                "recheck_hint": persisted["recheck_hint"],
            },
        }
        # The typed per-occurrence identity covers the complete payload and
        # created_at: a later observation appends, while an exact byte replay no-ops.
        event = mint_operational_event(event)
        operational.append(event)

    def _latest_effort_exhaustion_park_event(
        self, intent: EnvironmentIntent, artifact_identity: str
    ) -> Optional[tuple[JsonObject, bool]]:
        """Return the latest representative-seating effort-exhaustion park.

        BL-9c/d: the LATEST ``environment-setup-failure-parked`` event that
        (i) names this exact intent, target, and setup-artifact identity,
        (ii) was routed ``environment-effort-exhausted``, and (iii) seated a
        representative, together with whether a clearance receipt (canonical
        operational ledger or the operator sidecar) names it.

        Returns
        -------
        tuple[dict[str, Any], bool] | None
            ``(event, cleared)``, or ``None`` when no such park exists.
        """

        records_root = self.paths.ledgers.models.parent.parent
        canonical_path = canonical_operational_ledger_path(self.paths.ledgers.models)
        events = scan_jsonl(canonical_path)
        clearance_view = (*events, *environment_clearance_events(records_root))
        for event in reversed(events):
            details = event.get("details", {})
            if not isinstance(details, Mapping):
                continue
            if (
                event.get("event_kind") == OperationalEventKind.CAMPAIGN_HEALTH.value
                and details.get("disposition") == ENVIRONMENT_SETUP_FAILURE_PARKED_DISPOSITION
                and details.get("intent") == intent.name
                and details.get("target") == intent.lock.target
                and details.get("artifact_identity") == artifact_identity
                and details.get("failure_class") == "environment-effort-exhausted"
                and details.get("representative_stable_id")
            ):
                event_id = event.get("event_id")
                cleared = isinstance(event_id, str) and _quarantine_is_cleared(
                    clearance_view, event_id
                )
                return dict(event), cleared
        return None

    def _live_effort_exhaustion_park_event(
        self, intent: EnvironmentIntent, artifact_identity: str
    ) -> Optional[JsonObject]:
        """Return the standing UNCLEARED effort-exhaustion park, if one governs.

        A cleared park no longer governs: a new exhaustion is a new fact and
        may seat a new representative (BL-9d).
        """

        latest = self._latest_effort_exhaustion_park_event(intent, artifact_identity)
        if latest is None:
            return None
        event, cleared = latest
        return None if cleared else event

    def _park_environment_setup_failure(
        self,
        pending: Sequence[WorkItem],
        *,
        intent: EnvironmentIntent,
        failure: Exception,
        route: EnvironmentFailureRoute,
        observed_generation: Optional[str],
        operational: JsonlLedger,
    ) -> str:
        """Park one intent's pending work behind a routed pre-use env failure.

        Appends ONE intent-level campaign-health event naming the typed
        failure and its routing decision, then one durable machine-deferral
        row per pending work identity in the environment-rebuild ledger. No
        model is terminalized: no build served any of these work identities,
        so a ``failed:environment`` terminal would assert a world-fact the
        run never observed -- exactly the fan-out that wiped 11 released
        awards in 124 seconds on 2026-08-08 (T2-08) and left 19/63 pilot
        heads ``failed:environment`` (T3-T2).

        Parameters
        ----------
        pending:
            Every scheduled work generation the failed environment refused.
        intent:
            Routed environment intent that failed pre-use.
        failure:
            The typed lifecycle failure.
        route:
            Registry row that adjudicated the failure (setup or host scope).
        observed_generation:
            Environment generation, when one was observed before the failure.
        operational:
            Locked operational ledger for the health events.

        Returns
        -------
        str
            The intent-level parking event's content-addressed id (BL-9c: the
            caller watches it for a mid-leg clearance receipt).
        """

        created_at = self.dependencies.clock()
        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = rebuild_deferral_path(records_root)
        artifact_identity = _intent_setup_artifact_identity(intent)
        artifact_state = _intent_setup_artifact_state(intent)
        failure_type = f"{type(failure).__module__}.{type(failure).__qualname__}"
        scope = "host-condition" if route.kind == "park-host-condition" else "intent-setup"
        # BL-9d: environment effort exhaustion bounds its blast radius. The
        # pilot mass-parked 46/60 of a rung behind a human-only release from
        # ONE exhausted intent budget; now exactly one deterministic
        # representative (first pending in work order) carries the durable
        # ``intent-setup`` park, and every other member gets a self-releasing
        # ``intent-setup-cohort`` row: withheld honestly for THIS leg's
        # completion partition, re-admitted automatically afterwards. When an
        # UNCLEARED effort-exhaustion park already names a representative for
        # this exact intent+target+identity, no additional representative is
        # seated -- the cohort hangs off the standing fact instead of
        # accreting one newly human-parked member per leg.
        representative: Optional[WorkItem] = None
        prior_park_event_id: Optional[str] = None
        effort_exhausted = (
            scope == "intent-setup" and route.failure_class == "environment-effort-exhausted"
        )
        if effort_exhausted and pending:
            prior = self._live_effort_exhaustion_park_event(intent, artifact_identity)
            if prior is None:
                representative = pending[0]
            else:
                prior_park_event_id = str(prior.get("event_id"))
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.RUNNER_FAILED.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": len(pending)},
            "current_environment": intent.name,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": ENVIRONMENT_SETUP_FAILURE_PARKED_DISPOSITION,
                "ledger": str(ledger_path.relative_to(records_root)),
                "intent": intent.name,
                "target": intent.lock.target,
                "artifact_identity": artifact_identity,
                "env_generation": observed_generation,
                "failure_type": failure_type,
                "failure_class": route.failure_class,
                "condition_scope": scope,
                "detail": str(failure)[:1500],
            },
        }
        if effort_exhausted:
            details = event["details"]
            assert isinstance(details, dict)
            details["representative_stable_id"] = (
                representative.stable_id if representative is not None else None
            )
            details["cohort_size"] = len(pending) - (1 if representative is not None else 0)
            if prior_park_event_id is not None:
                details["prior_park_event_id"] = prior_park_event_id
        # The typed per-occurrence identity covers the complete payload and
        # created_at: a later observation appends, while an exact byte replay no-ops.
        event = mint_operational_event(event)
        operational.append(event)
        event_id = str(event["event_id"])
        for item in pending:
            member_scope = scope
            if effort_exhausted and item is not representative:
                member_scope = "intent-setup-cohort"
            self._record_environment_failure_park(
                item,
                intent=intent,
                park_event_id=event_id,
                artifact_identity=artifact_identity,
                artifact_state=artifact_state,
                env_generation=observed_generation,
                failure_type=failure_type,
                failure_class=route.failure_class,
                condition_scope=member_scope,
                operational=operational,
            )
        return event_id

    def _record_environment_failure_park(
        self,
        item: WorkItem,
        *,
        intent: EnvironmentIntent,
        park_event_id: str,
        artifact_identity: str,
        artifact_state: JsonObject,
        env_generation: Optional[str],
        failure_type: str,
        failure_class: str,
        condition_scope: str,
        operational: JsonlLedger,
    ) -> None:
        """Append one work identity's durable pre-use env-failure park row.

        The row lives in the environment-rebuild machine-deferral ledger (the
        established intent-scoped park the completion partition already
        accounts), with two additive fields: ``failure_class`` (the registry's
        adjudication) and ``condition_scope``. Scope governs the admission
        fact-recheck: an ``intent-setup`` row holds while the setup-artifact
        identity is unchanged (rotation or an explicit clearance releases
        it), and a ``host-condition`` row NEVER re-claims -- a transient host
        condition is not provable later, so the model is simply attempted and
        the live valve re-derives the park if the condition persists.

        Parameters
        ----------
        item:
            Exact scheduled work generation the failed environment refused.
        intent:
            Routed environment intent that failed pre-use.
        park_event_id:
            Intent-level ``environment-setup-failure-parked`` event this row
            hangs from (an explicit clearance receipt naming it releases the
            row exactly like a cleanup quarantine's).
        artifact_identity, artifact_state:
            Typed committed setup-artifact identity and per-member state.
        env_generation:
            Observed environment generation, when one exists.
        failure_type, failure_class, condition_scope:
            Typed failure identity and the registry's routing adjudication.
        operational:
            Locked operational ledger for the per-item health event.
        """

        created_at = self.dependencies.clock()
        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = rebuild_deferral_path(records_root)
        if condition_scope == "host-condition":
            remedy = (
                "The condition names HOST/infrastructure state, not the model and "
                "not the committed family: recover the host (or re-drive this work "
                "item on any healthy host) and the model is attempted normally on "
                "the next run -- this park never re-claims the model by itself."
            )
        elif condition_scope == "intent-setup-cohort":
            # BL-9d cohort row: leg-scoped withholding, never a human hold.
            remedy = (
                "This member was withheld for THIS leg only because its intent's "
                "environment effort budget is exhausted; the durable park lives on "
                "the cohort's single representative (named in the parking event). "
                "Recover or rotate the environment, or clear the representative's "
                "park, and every cohort member re-admits automatically -- this row "
                "never re-claims the model by itself."
            )
        else:
            remedy = (
                "The condition names the intent's committed SETUP: recover or "
                "rotate the environment (a re-solve changes the intent's "
                "setup-artifact identity), or record an explicit "
                "environment-quarantine-cleared event naming the parking event "
                "after an out-of-band repair, then re-drive this work item."
            )
        recheck_hint = (
            f"environment setup for intent {intent.name!r} (target "
            f"{intent.lock.target!r}) failed PRE-USE with {failure_type} "
            f"(routed {failure_class!r}, campaign-health event {park_event_id!r}); "
            "no build result was recorded for this model and nothing about the "
            f"model was assessed. {remedy}"
        )
        row: JsonObject = {
            "schema_version": REBUILD_DEFERRAL_SCHEMA_VERSION,
            "stable_id": item.stable_id,
            "work_id": item.active_work_id,
            "name": item.intake.name,
            "campaign_id": bound_campaign_id(self.config),
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "created_at": created_at,
            "disposition": REBUILD_DEFERRAL_DISPOSITION,
            "quarantine": {
                "event_id": park_event_id,
                "intent": intent.name,
                "target": intent.lock.target,
                "artifact_identity": artifact_identity,
                "artifact_state": {name: dict(state) for name, state in artifact_state.items()},
                "env_generation": env_generation,
                "failure_type": failure_type,
                "failure_class": failure_class,
                "condition_scope": condition_scope,
            },
            "recheck_hint": recheck_hint,
        }
        row = stamp_row_digest(row)
        persisted = append_rebuild_deferral_row(ledger_path, row)
        _stash_machine_deferral_row(self, item.stable_id, persisted)
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": intent.name,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": ENVIRONMENT_SETUP_FAILURE_PARKED_DISPOSITION,
                "ledger": str(ledger_path.relative_to(records_root)),
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "intent": intent.name,
                "target": intent.lock.target,
                "park_event_id": park_event_id,
                "failure_type": failure_type,
                "failure_class": failure_class,
                "condition_scope": condition_scope,
                "recheck_hint": persisted["recheck_hint"],
            },
        }
        # The typed per-occurrence identity covers the complete payload and
        # created_at: a later observation appends, while an exact byte replay no-ops.
        event = mint_operational_event(event)
        operational.append(event)

    def _withhold_for_host_capacity(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        operational: JsonlLedger,
        reducer: Optional[Any] = None,
        state: Optional[JsonObject] = None,
    ) -> bool:
        """Withhold a model this host provably cannot instantiate, pre-normalization.

        The environment-lane capacity gate (``_admit_within_host_capacity``)
        already refuses oversized models -- but only for artifacts that SURVIVE
        author-lane normalization. The 2026-08-04 pilot rung proved that order
        wrong: full-config Mixtral (m5915, a ~31B-parameter lower bound against
        a 24 GiB host) terminalized ``failed:runner`` / ``protocol-violation``
        on an unrelated normalization refusal before any size check ran, and
        ``capacity-deferrals`` reported zero. The size is knowable from the
        authored recipe alone, so the assessment runs here too -- right after
        environment coverage and BEFORE ``_normalize_artifact_modes`` -- and an
        oversized model defers recoverably instead of terminalizing on whatever
        normalization happens to refuse first.

        The environment-lane gate stays as the backstop; the durable ledger is
        idempotent per work generation, so both firing can never double-record.

        Since WS-2 the withholding also MINTS a model head when the refusal is the
        static declared-config kind and the artifact carries an exact primary source:
        the durable row alone made the member invisible to every head-counting
        consumer, so a leg's ``settled()`` and the operator's tally disagreed about the
        same five members for three days.

        Parameters
        ----------
        item:
            Exact scheduled work generation.
        artifact:
            Validated, staged author artifact carrying the declarative recipe.
        operational:
            Locked operational ledger for the health event.
        reducer, state:
            Canonical authority and mutable driver state, required to MINT the
            host-capability head. Both omitted means row-and-stash only, which is what
            a bare unit-level call gets: a canonical head is never invented without the
            reducer that validates it.

        Returns
        -------
        bool
            Whether the model was withheld from this wave.
        """

        try:
            proposal = artifact.proposal
        except DriverIntegrationError:
            # A terminal author result has no executable proposal and declares no
            # recipe; there is nothing to size.
            return False
        facts = proposal.get("proposed_facts")
        implementation = facts.get("implementation") if isinstance(facts, Mapping) else None
        if not isinstance(implementation, Mapping):
            return False
        assessment = assess_model_capacity(implementation)
        if not assessment.deferred:
            return False
        cache = author_result_cache_path(self.paths.work_root, item.stable_id)
        try:
            _write_json_atomic(
                cache,
                serialize_author_result_cache(
                    artifact.author_result,
                    source_manifest=artifact.source_manifest,
                    model_dir=artifact.model_dir,
                    staged_transaction_id=(
                        str(artifact.staged.transaction_id)
                        if artifact.staged is not None
                        else None
                    ),
                    lifecycle=(
                        "staged-pending" if artifact.staged is not None else "unstaged"
                    ),
                ),
            )
        except Exception:  # noqa: BLE001 -- a disposable cache must never block the record
            cache.unlink(missing_ok=True)
        self._record_host_capacity_deferral(
            item, assessment, operational, artifact=artifact, reducer=reducer, state=state
        )
        return True

    def _record_host_capacity_deferral(
        self,
        item: WorkItem,
        assessment: CapacityAssessment,
        operational: JsonlLedger,
        *,
        artifact: Optional[AuthorArtifact] = None,
        reducer: Optional[Any] = None,
        state: Optional[JsonObject] = None,
    ) -> None:
        """Append one durable host-capacity deferral row plus its campaign-health event.

        Then mint the model head, when all three of the mint's own preconditions hold
        (:func:`~menagerie.crawler.driver_admission._mint_host_capability_head`). The row
        is written FIRST and unconditionally: it is the evidence the head is derived
        from, and a refused mint must still leave the campaign's honest statement behind.
        """

        created_at = self.dependencies.clock()
        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = capacity_deferral_path(records_root)
        persisted = append_capacity_deferral_row(
            ledger_path,
            build_capacity_deferral_row(
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                name=item.intake.name,
                campaign_id=bound_campaign_id(self.config),
                run_id=self.config.run_id,
                machine_id=self.config.machine_id,
                created_at=created_at,
                assessment=assessment,
            ),
        )
        _stash_machine_deferral_row(self, item.stable_id, persisted)
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
                "ledger": str(ledger_path.relative_to(records_root)),
                "host_physical_memory_bytes": assessment.host.physical_memory_bytes,
                "admissible_parameter_ceiling": assessment.host.admissible_parameter_ceiling,
                "deferred": [
                    {
                        "stable_id": item.stable_id,
                        "work_id": item.active_work_id,
                        "parameter_count_lower_bound": (
                            assessment.estimate.parameter_count_lower_bound
                            if assessment.estimate is not None
                            else None
                        ),
                    }
                ],
            },
        }
        # The typed per-occurrence identity covers the complete payload and
        # created_at: a later observation appends, while an exact byte replay no-ops.
        event = mint_operational_event(event)
        operational.append(event)
        _mint_host_capability_head(
            self,
            item,
            persisted,
            operational,
            artifact=artifact,
            reducer=reducer,
            state=state,
        )

    def _short_circuit_still_deferred(self, item: WorkItem, operational: JsonlLedger) -> bool:
        """Reclaim a still-valid durable valve row instead of re-buying the author.

        The reclaimed row enters the same run-scoped stash the live valve fires
        feed, so the completion partition accounts the model machine-deferred
        through the identical path -- never through a blind ledger read.

        Parameters
        ----------
        item:
            Exact scheduled work generation.
        operational:
            Locked operational ledger for the health event.

        Returns
        -------
        bool
            Whether the model was withheld from this wave without a session.
        """

        row = _still_valid_deferral_row(self, item)
        if row is None:
            return False
        _stash_machine_deferral_row(self, item.stable_id, row)
        created_at = self.dependencies.clock()
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": (
                item.route.intent
                if row.get("disposition") == COVERAGE_DEFERRAL_DISPOSITION
                else None
            ),
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "machine-deferral-row-reused",
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "row_disposition": row.get("disposition"),
                "row_sha256": row.get("row_sha256"),
                "recheck_hint": row.get("recheck_hint"),
            },
        }
        # The typed per-occurrence identity covers the complete payload and
        # created_at: a later observation appends, while an exact byte replay no-ops.
        event = mint_operational_event(event)
        operational.append(event)
        return True

    def _preserve_uncommitted_author_result(self, item: WorkItem, artifact: AuthorArtifact) -> None:
        """Cache one dispatched-but-uncommitted author result for the next run.

        Called only when a wave unwinds (usage pause, retryable operator failure,
        review checkpoint, shutdown) with siblings already finished. It writes the
        same disposable cache the committed path writes, after the same validation
        gauntlet, and nothing else: no ledger append, no staging, no artifact map
        entry. A result that cannot pass validation is simply dropped, so the model
        is re-authored and terminalized properly rather than reloaded from a cache
        that skipped a check.
        """

        try:
            if not isinstance(artifact.author_result, ProposedAuthorResult):
                return
            candidate = _normalize_artifact_modes(
                artifact,
                self.config,
                environment_packages=_routed_environment_packages(
                    getattr(self, "registry", None), item.route.intent
                ),
            )
            if candidate.proposal.get("stable_id") != item.stable_id:
                return
            if candidate.author_result.binding.work_id != item.active_work_id:
                return
            _validate_artifact_identities(candidate, self.config, item=item)
            _write_json_atomic(
                author_result_cache_path(self.paths.work_root, item.stable_id),
                serialize_author_result_cache(
                    candidate.author_result,
                    source_manifest=candidate.source_manifest,
                    model_dir=candidate.model_dir,
                    staged_transaction_id=(
                        str(candidate.staged.transaction_id)
                        if candidate.staged is not None
                        else None
                    ),
                    lifecycle=(
                        "staged-pending" if candidate.staged is not None else "unstaged"
                    ),
                ),
            )
        except Exception:  # noqa: BLE001 -- a best-effort cache must never mask the abort
            return

    def _commit_author_wave(
        self,
        work: Sequence[WorkItem],
        pool: _AuthorWavePool,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        attempt_ladders: Optional[Mapping[str, _RetryLadderState]] = None,
    ) -> dict[str, AuthorArtifact]:
        """Commit one wave's author results on the single canonical writer.

        Walks ``work`` in its original order and performs every canonical effect --
        rehydration, family-variant instantiation, cache reload, staging, validation,
        terminal routing, and the usage-pause unwind -- exactly as the historical
        serial lane did. The only difference is where a fresh author result comes
        from: a session this wave already dispatched, or an inline call.

        ``attempt_ladders`` carries the ACAP cross-round continuations: a model
        whose capped ladder raises :class:`_AuthorRoundAttemptsDeferred` is
        returned unresolved (nothing appended, staged, or routed) and recorded
        in ``self._author_wave_deferred_ids`` for the scheduler, still in work
        order on this writer, so completion order can never influence records.
        """

        # W9 cache-layer registry: refuse before the first admission if any
        # registered on-disk cache lacks a named invalidation authority. An
        # authority-less durable cache steering admissions is exactly the R16
        # class; the refusal is loud and pre-spend.
        startup_cache_registration_check()
        artifacts: dict[str, AuthorArtifact] = {}
        for item in work:
            self._check_shutdown("author-admission", item=item)
            # Bounded drain point: the previous item's commit work (staging,
            # terminal routing, cache validation) ran with the relay unserviced;
            # queued session starts board here instead of waiting for the next
            # claim.
            pool.service_admissions()
            canonical_artifact = self._rehydrate_final_authority(item, reducer, operational)
            if canonical_artifact is not None:
                if isinstance(canonical_artifact, (ActivatedHandoffArtifact,)) or isinstance(
                    canonical_artifact.author_result, ProposedAuthorResult
                ):
                    try:
                        _validate_artifact_identities(
                            canonical_artifact,
                            self.config,
                            item=item,
                            recorded_basis=canonical_artifact.recorded_identity_basis,
                        )
                    except DriverIntegrationError as exc:
                        # A rehydrated finalized artifact failing identity
                        # validation at its own recorded basis is a stale or
                        # internally inconsistent CACHE entry, never a driver
                        # crash (rung-12 leg 2: the live-basis variant of this
                        # check aborted the campaign at the first pre-rotation
                        # artifact). Invalidate loudly and fall through to the
                        # disposable-cache/fresh-authorship path below.
                        self._report_identity_stale_artifact_authority(
                            item, canonical_artifact, exc, operational
                        )
                        canonical_artifact = None
                    else:
                        artifacts[item.stable_id] = canonical_artifact
                        self._family_artifacts[item.stable_id] = canonical_artifact
                else:
                    pause = self._route_terminal_author_result(
                        item,
                        canonical_artifact,
                        reducer,
                        operational,
                        state,
                    )
                    if pause is not None:
                        raise DriverPaused(pause)
                if canonical_artifact is not None:
                    self.dependencies.boundary_hook("after-author", item.stable_id)
                    continue
            if item.stable_id in getattr(self, "_invalidation_nonconvergent_ids", ()):
                # W9 N-repeat fuse: the third same-key identity-stale
                # retirement (this leg's included -- the authority above just
                # derived the ordinal from the durable ledger) proves the
                # invalidation is not converging for this member (R16's paid
                # loop). Buying another author session or reloading the
                # scratch cache would re-enter it; the member parks behind
                # the durable partition-accounted quarantine valve, typed
                # ``invalidation-nonconvergent``, auto-releasing next leg
                # (the fuse re-derives from the ledger and re-parks until
                # the underlying rot is fixed).
                self._record_scheduler_quarantine(
                    item,
                    site="admission",
                    reason_code="invalidation-nonconvergent",
                    fault=(
                        "third same-key identity-stale cache retirement; "
                        "invalidation is not converging for this member"
                    ),
                    operational=operational,
                )
                continue
            # Resume idempotency (rung 10): a still-valid durable valve row for
            # this exact work generation already IS the model's disposition;
            # re-buying an author session to re-derive the identical row taxed
            # leg 2 five full sessions (~50 minutes). Canonical authority was
            # consulted first, so a terminal or staged result has already won
            # by the time this fires -- the record wins, W3-P semantics.
            if self._short_circuit_still_deferred(item, operational):
                continue
            if item.is_family_variant:
                representative_model = reducer.current_view.get(item.family_representative_id)
                representative_artifact = self._family_artifacts.get(item.family_representative_id)
                if (
                    _usable_family_representative(
                        representative_model, item.family_representative_id
                    )
                    and representative_artifact is not None
                ):
                    assert representative_model is not None
                    variant = _instantiate_variant_artifact(
                        item,
                        representative_artifact,
                        representative_model,
                        self.config,
                        reducer.context,
                    )
                    variant = self._stage_author_result(
                        item, variant, reducer, operational_events=operational.records
                    )
                    _validate_artifact_identities(variant, self.config, item=item)
                    artifacts[item.stable_id] = variant
                    self._family_artifacts[item.stable_id] = variant
                    self.dependencies.boundary_hook("after-author", item.stable_id)
                    continue
                if representative_model is not None:
                    raise DriverIntegrationError(
                        "trusted family variant has no usable representative authority"
                    )
            cache = author_result_cache_path(self.paths.work_root, item.stable_id)
            if cache.is_file():
                try:
                    cached_value = _read_json(cache)
                    cached_manifest = cached_value.get("source_manifest")
                    cached_model_dir = cached_value.get("model_dir")
                    if not isinstance(cached_manifest, Mapping) or not isinstance(
                        cached_model_dir, str
                    ):
                        raise DriverIntegrationError("author-result cache lacks staging inputs")
                    cached_raw_result = cached_value.get("result")
                    if not isinstance(cached_raw_result, Mapping):
                        raise DriverIntegrationError("author-result cache lacks its raw result")
                    result_id = cached_raw_result.get("result_id")
                    cache_binding = cached_value.get("generation_binding")
                    staged_reference = (
                        cache_binding.get("staged_transaction_id")
                        if isinstance(cache_binding, Mapping)
                        else None
                    )
                    if isinstance(staged_reference, str) and staged_reference:
                        # AT-10: a cache may reference only a staging
                        # transaction the ledger itself holds. A reference the
                        # ledger does not know is foreign evidence, and a
                        # chain that reached its FINAL event is owned by
                        # canonical authority -- which was consulted FIRST
                        # and declined -- so serving either would bypass the
                        # ledger's disposition. A chain that advanced but
                        # TORE (no final event: the fold quarantines it stale)
                        # keeps the cached result servable -- the paid author
                        # work is same-generation durable evidence, and
                        # re-serving it is the AT-4 conservation law (no paid
                        # author call for a publication-only retry). m12646's
                        # class stays closed by the binding checks, not this
                        # shape check: a re-drive's new work generation is a
                        # foreign work_id the loader refuses outright.
                        reference_chain = reducer.artifact_ledger.transaction_events(
                            ArtifactTransactionId(staged_reference)
                        )
                        if (
                            not reference_chain
                            or reference_chain[0].get("event_kind")
                            != ArtifactEventKind.STAGED_PRIVATE.value
                        ):
                            raise DriverIntegrationError(
                                "author-result cache references a non-pending "
                                f"staging transaction ({len(reference_chain)} "
                                "events); the canonical ledger owns this "
                                "result's disposition (AT-10)"
                            )
                        if reference_chain[-1].get("event_kind") in {
                            ArtifactEventKind.PUBLISHED.value,
                            ArtifactEventKind.PRIVATE_COMMITTED.value,
                        }:
                            raise DriverIntegrationError(
                                "author-result cache references a FINALIZED "
                                "artifact transaction; canonical authority "
                                "owns this result's disposition (AT-10)"
                            )
                    anchored_events = tuple(
                        event
                        for event in reducer.artifact_ledger.events
                        if event.get("stable_id") == item.stable_id
                        and event.get("author_result_id") == result_id
                    )
                    anchored_work_ids = {str(event["work_id"]) for event in anchored_events}
                    if item.requeue_work_id is None and len(anchored_work_ids) > 1:
                        raise DriverIntegrationError(
                            "author-result cache has ambiguous artifact-ledger work authority"
                        )
                    expected_work_id = (
                        item.active_work_id
                        if item.requeue_work_id is not None or item.refresh_work_id is not None
                        else next(iter(anchored_work_ids))
                        if anchored_work_ids
                        else item.active_work_id
                    )
                    # Supersession tripwire (rung 11, 2026-08-07, m10949): a leg
                    # that dies between a repair re-author's staging and its cache
                    # rewrite leaves the disposable cache anchored to an OLDER
                    # staged authoring than the ledger's latest for the same work.
                    # The reducer's `prepare_model` derives artifact authority from
                    # the work-scoped LATEST event, so accepting the stale cache
                    # guarantees `append_artifact_authorization` refuses at
                    # terminalization (`authorization dependency vector is stale`).
                    # Mirror that exact selection here: when the cached result was
                    # staged but is no longer the work's latest artifact event, the
                    # cache is superseded -- discard it and re-author under the
                    # current authority instead of running a whole worker phase
                    # against bytes the reducer will refuse to finalize.
                    if anchored_events:
                        work_scoped_events = tuple(
                            event
                            for event in reducer.artifact_ledger.events
                            if event.get("stable_id") == item.stable_id
                            and str(event.get("work_id")) == expected_work_id
                        )
                        if (
                            work_scoped_events
                            and work_scoped_events[-1].get("author_result_id") != result_id
                        ):
                            raise _AuthorCacheSuperseded(
                                "author-result cache is superseded by a later staged "
                                "authoring for the same work generation",
                                superseded_author_result_ids=frozenset(
                                    str(event.get("author_result_id"))
                                    for event in work_scoped_events
                                    if event.get("event_kind")
                                    == ArtifactEventKind.STAGED_PRIVATE.value
                                ),
                            )
                    anchored_campaign_ids = {
                        str(gate_item["campaign_root_work_id"])
                        for gate in reducer.gate_records
                        for gate_item in gate.get("items", [])
                        if isinstance(gate_item, Mapping)
                        and gate_item.get("stable_id") == item.stable_id
                        and gate_item.get("work_id") == expected_work_id
                    }
                    if len(anchored_campaign_ids) > 1:
                        raise DriverIntegrationError(
                            "author-result cache has ambiguous checker campaign authority"
                        )
                    expected_campaign_id = (
                        next(iter(anchored_campaign_ids))
                        if anchored_campaign_ids
                        else _campaign_id_for_item(item)
                    )
                    cache_context = reducer.context
                    if anchored_events:
                        anchored_snapshot_bindings = {
                            (
                                str(event["intake_snapshot_id"]),
                                str(event["intake_snapshot_sha256"]),
                            )
                            for event in anchored_events
                        }
                        if len(anchored_snapshot_bindings) != 1:
                            raise DriverIntegrationError(
                                "author-result cache has ambiguous intake snapshot authority"
                            )
                        snapshot_id, snapshot_sha256 = next(iter(anchored_snapshot_bindings))
                        if (
                            cached_raw_result.get("intake_snapshot_id") != snapshot_id
                            or cached_raw_result.get("intake_snapshot_sha256") != snapshot_sha256
                        ):
                            raise DriverIntegrationError(
                                "author-result cache contradicts its staged intake anchor"
                            )
                        cache_context = replace(
                            reducer.context,
                            active_intake_snapshot_id=snapshot_id,
                            active_intake_snapshot_sha256=snapshot_sha256,
                        )
                    cached_envelope = build_author_envelope(
                        context=cache_context,
                        work_id=expected_work_id,
                        stable_id=item.stable_id,
                        campaign_id=expected_campaign_id,
                        created_at=self.dependencies.clock(),
                        untrusted_hints=item.intake.to_dict(),
                        source_manifest=cached_manifest,
                        allowed_model_dir=cached_model_dir,
                        output_path=cache.parent / "author" / "result.json",
                        routed_environment_intent=item.route.intent,
                        routed_environment_packages=_routed_environment_packages(
                            getattr(self, "registry", None), item.route.intent
                        ),
                    )
                    cached_result = validate_author_result_cache(
                        cached_value,
                        cached_envelope,
                        cas_root=cache.parent / "author" / "source-cas",
                    )
                    cached_artifact_v3 = AuthorArtifact(
                        cached_result,
                        dict(cached_manifest),
                        Path(cached_model_dir),
                    )
                    if isinstance(cached_result, ProposedAuthorResult):
                        _validate_artifact_identities(cached_artifact_v3, self.config, item=item)
                    anchored_staged = staged_artifact_for_result(
                        reducer.artifact_ledger,
                        stable_id=item.stable_id,
                        work_id=expected_work_id,
                        author_result_id=str(result_id),
                    )
                    cached_artifact_v3 = (
                        replace(cached_artifact_v3, staged=anchored_staged)
                        if anchored_staged is not None
                        else self._stage_author_result(
                            item,
                            cached_artifact_v3,
                            reducer,
                            operational_events=operational.records,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- disposable cache is regenerable
                    created_at = self.dependencies.clock()
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
                            "disposition": "author-cache-discarded",
                            "stable_id": item.stable_id,
                            "work_id": item.active_work_id,
                            "cache": str(cache),
                            "error": f"{type(exc).__name__}: {exc}"[:STDIO_TAIL_MAX_CHARS],
                            # Exact superseded ids remain diagnostic lineage facts.
                            # They never subtract durable spend for this generation.
                            **(
                                {
                                    "superseded_author_result_ids": sorted(
                                        exc.superseded_author_result_ids
                                    )
                                }
                                if isinstance(exc, _AuthorCacheSuperseded)
                                else {}
                            ),
                        },
                    }
                    event = mint_operational_event(event)
                    operational.append(event)
                    # Retire-by-rename (A20/IC-Q1 ruling): a discarded cache
                    # is EVIDENCE of the discard cause, never unlinked. The
                    # reaper protects the renamed file.
                    self._retire_author_scratch_cache(
                        item.stable_id, str(event["event_id"])
                    )
                else:
                    if isinstance(cached_result, ProposedAuthorResult):
                        # Re-derived on the cached path too, so a resume repeats the
                        # withholding instead of letting a model the campaign already
                        # deferred slip into the environment lane.
                        if self._withhold_for_environment_coverage(
                            item, cached_artifact_v3, operational
                        ):
                            continue
                        if self._withhold_for_host_capacity(
                            item, cached_artifact_v3, operational, reducer, state
                        ):
                            continue
                        artifacts[item.stable_id] = cached_artifact_v3
                    else:
                        pause = self._route_terminal_author_result(
                            item,
                            cached_artifact_v3,
                            reducer,
                            operational,
                            state,
                        )
                        if pause is not None:
                            raise DriverPaused(pause)
                    continue
            try:
                # T2-07/T1-13: the allowance is derived from append-only staged
                # history at every buy, so a rehydrated leg (whose in-memory
                # counters restarted at zero) can never re-buy spend its
                # predecessors already staged. Raises inside this try so the
                # model-local blanket arm records `failed:author` /
                # `effort-exhausted:author-sessions` for THIS model only.
                # Cache-discard rows never subtract money truth: a resume over
                # stale bytes retains every buy for this work generation.
                enforce_author_session_allowance(
                    (*reducer.artifact_ledger.events, *operational.records),
                    item,
                    operational_events=operational.records,
                )
                dispatched = pool.claim(item)
                artifact = (
                    dispatched.unwrap()
                    if dispatched is not None
                    else self._retry_infrastructure_call(
                        lambda: self.dependencies.author.author(
                            item,
                            self.paths.work_root,
                            self.config,
                            reducer.context,
                        ),
                        admission=("author", item),
                        operational=operational,
                        author_ladder=(
                            None
                            if attempt_ladders is None
                            else attempt_ladders.get(item.stable_id)
                        ),
                    )
                )
                # The ladder resolved: drop its cross-round continuation.
                self._author_attempt_ladders_pop(item.stable_id)
                artifact = self._stage_author_result(
                    item, artifact, reducer, operational_events=operational.records
                )
                # Staging reads and mirrors every artifact byte; drain the buys
                # that queued behind it before the terminal/validation tail.
                pool.service_admissions()
            except AuthorBackoffError as backoff:
                # Provider usage exhaustion is a campaign pause with a reset time,
                # never this model's failure. Caught ahead of the blanket arm below
                # so it can never be recorded as `failed:source`.
                raise AuthorUsagePause(
                    self._pause_for_usage(backoff.signal, operational, len(work))
                ) from backoff
            except AuthorSessionRetryExhausted as exhausted:
                # The bounded ladder spent its budget on no-publication session
                # deaths. The mechanical lane-health boundary decides the scope:
                # a contemporaneous publication witness plus passing host checks
                # terminalizes THIS model; anything else stays the campaign-scoped
                # loud halt the re-raise below preserves.
                self._author_attempt_ladders_pop(item.stable_id)
                if self._scope_author_retry_exhaustion(
                    item, None, exhausted, reducer, operational, state
                ):
                    continue
                raise
            except _AuthorRoundAttemptsDeferred:
                # ACAP: the in-round attempt budget is spent without an artifact
                # or a terminal. The model returns UNRESOLVED -- nothing appended,
                # nothing staged, nothing routed -- and the pipelined scheduler
                # keeps it author-ready, so its remaining attempts board a later
                # round's slice while other models use this round's slots.
                self._author_wave_deferred_ids.add(item.stable_id)
                continue
            except RetryableOperatorError:
                raise
            except Exception as exc:  # noqa: BLE001 -- author failure belongs to this model
                self._author_attempt_ladders_pop(item.stable_id)
                stage, reason_code = _author_lane_failure(exc)
                # Emitted BEFORE the terminal is appended, so the cause survives even if
                # the terminal bookkeeping below refuses. Without it the recorded facts
                # are `failed:<stage>` plus a reason code and a null detail: a whole
                # partition can terminalize for one cause with nothing durable naming it.
                operational.append(
                    _model_lane_failure_event(
                        stable_id=item.stable_id,
                        work_id=item.active_work_id,
                        status_code=f"failed:{stage}",
                        reason_code=reason_code,
                        exc=exc,
                        run_id=self.config.run_id,
                        machine_id=self.config.machine_id,
                        created_at=self.dependencies.clock(),
                    )
                )
                attempt = _driver_failure_attempt(
                    item,
                    None,
                    stage,
                    reason_code,
                    exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                if isinstance(exc, LedgerConflictError):
                    _stamp_failure_owner(attempt, "machinery")
                persisted = reducer.append_attempt(attempt).record
                persisted_error = persisted.get("error")
                if (
                    isinstance(persisted_error, Mapping)
                    and _failure_owner(persisted_error) in NEVER_MODEL_TERMINAL_OWNERS
                ):
                    # DX-1/RE-1 at the arm (item 9): a host/machinery-owned
                    # author-lane fault is infrastructure weather, never this
                    # model's verdict -- park loudly, member re-drives next leg.
                    # The mint sink would refuse the terminal anyway; parking
                    # here keeps its ``sink_guard`` a pure defect signal.
                    self._park_non_model_fault(item, None, persisted_error, operational)
                    continue
                self._terminalize(
                    item,
                    None,
                    f"failed:{stage}",
                    reason_code,
                    str(exc),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                )
                continue
            if not isinstance(artifact.author_result, ProposedAuthorResult):
                # The terminal arm was the ONE author outcome with no model-local
                # handler: every sibling path below records a `protocol-violation`
                # attempt and moves to the next model, while a stale gate binding
                # or a refused checker contract on a single terminal recommendation
                # unwound the whole campaign. A per-model outcome belongs to its
                # model. Transport failures and campaign pauses still propagate --
                # they are genuinely campaign-wide.
                try:
                    pause = self._route_terminal_author_result(
                        item, artifact, reducer, operational, state
                    )
                except (DriverPaused, RetryableOperatorError, AuthorBackoffError, CheckerPromptRotatedError):
                    raise
                except Exception as exc:  # noqa: BLE001 -- terminal gating is model-local
                    # The STAGE stays `runner`, and that is not a compromise: a lane
                    # that never obtained a gate cannot assert a gate verdict, and
                    # `failed:accuracy-gate` is refused outright by terminal-proof
                    # derivation ("lacks exact rejected gate evidence"). The reason
                    # still has to separate two different truths that were collapsing
                    # onto one label: a refused checker CONTRACT is recoverable by
                    # fixing the checker and re-running exactly these models, while a
                    # transport/internal fault is a different fact. The fidelity lane
                    # already splits on exactly this; this arm now matches it.
                    reason = (
                        "internal-error"
                        if self._is_infrastructure_error(exc)
                        else "protocol-violation"
                    )
                    attempt = _driver_failure_attempt(
                        item,
                        artifact,
                        "runner",
                        reason,
                        exc,
                        self.config,
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                        environment=None,
                        created_at=self.dependencies.clock(),
                    )
                    if reason == "internal-error":
                        _stamp_failure_owner(attempt, "host")
                    persisted = reducer.append_attempt(attempt).record
                    persisted_error = persisted.get("error")
                    if (
                        isinstance(persisted_error, Mapping)
                        and _failure_owner(persisted_error) in NEVER_MODEL_TERMINAL_OWNERS
                    ):
                        # DX-1/RE-1 (m12446 shape): a host/machinery fault while
                        # gating a first-call terminal recommendation is
                        # infrastructure weather, never this model's verdict --
                        # park loudly, member re-drives next leg.
                        self._park_non_model_fault(item, artifact, persisted_error, operational)
                        continue
                    # The artifact travels as EVIDENCE, never as authority: the
                    # broker already froze and hashed these sources, so recording
                    # `sources: []` and "source resolution did not complete" for a
                    # model whose manifest is full is a falsehood in a durable
                    # record. `terminal_gate_obtained=False` keeps those exact
                    # facts while withholding finalization and every checked rung.
                    self._terminalize(
                        item,
                        artifact,
                        "failed:runner",
                        reason,
                        str(exc),
                        (persisted,),
                        reducer,
                        operational,
                        state,
                        terminal_gate_obtained=False,
                    )
                    continue
                if pause is not None:
                    raise DriverPaused(pause)
                self.dependencies.boundary_hook("after-author", item.stable_id)
                continue
            # Checked BEFORE normalization, because the R1 digest binding inside it
            # is exactly what refuses an unservable routing -- and that refusal,
            # reaching the blanket arm below, is what used to blame the model for a
            # routing decision made at intake.
            if self._withhold_for_environment_coverage(item, artifact, operational):
                continue
            # Sized BEFORE normalization for the same reason coverage is: the R1
            # digest binding inside normalization can refuse for reasons of its
            # own (a version drift, a missing package), and letting it fire first
            # terminalizes a model this host was never going to attempt anyway.
            if self._withhold_for_host_capacity(item, artifact, operational, reducer, state):
                continue
            try:
                artifact = _normalize_artifact_modes(
                    artifact,
                    self.config,
                    environment_packages=_routed_environment_packages(
                        getattr(self, "registry", None), item.route.intent
                    ),
                )
                if artifact.proposal.get("stable_id") != item.stable_id:
                    raise DriverIntegrationError("author proposal stable_id does not match intake")
                expected_work_id = item.active_work_id
                if artifact.author_result.binding.work_id != expected_work_id:
                    raise DriverIntegrationError(
                        "author result does not bind the active work generation"
                    )
            except Exception as exc:  # noqa: BLE001 -- post-author validation is model-local
                stage, reason = (
                    _author_lane_failure(exc)
                    if isinstance(exc, AuthorEnvironmentBindingError)
                    else ("runner", "protocol-violation")
                )
                attempt = _driver_failure_attempt(
                    item,
                    artifact,
                    stage,
                    reason,
                    exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                persisted = reducer.append_attempt(attempt).record
                self._terminalize(
                    item,
                    artifact,
                    f"failed:{stage}",
                    reason,
                    str(exc),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                )
                continue
            try:
                _validate_artifact_identities(artifact, self.config, item=item)
            except DriverIntegrationError as exc:
                # Same legibility gap as the author-lane arm above, and demonstrably the
                # NEXT wall for the same models: `_validate_artifact_identities`
                # recomputes five author-gated identities and its message carries the
                # per-field claimed/computed mismatch dict -- the entire diagnostic --
                # into a `detail` the terminal then nulls. The reason code is correct
                # here, so only the cause needs surfacing.
                operational.append(
                    _model_lane_failure_event(
                        stable_id=item.stable_id,
                        work_id=item.active_work_id,
                        status_code="failed:evidence",
                        reason_code="coverage-incomplete",
                        exc=exc,
                        run_id=self.config.run_id,
                        machine_id=self.config.machine_id,
                        created_at=self.dependencies.clock(),
                    )
                )
                attempt = _driver_failure_attempt(
                    item,
                    artifact,
                    "evidence",
                    "coverage-incomplete",
                    exc,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=None,
                    created_at=self.dependencies.clock(),
                )
                persisted = reducer.append_attempt(attempt).record
                self._terminalize(
                    item,
                    artifact,
                    "failed:evidence",
                    "coverage-incomplete",
                    str(exc),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                )
                continue
            _write_json_atomic(
                cache,
                serialize_author_result_cache(
                    artifact.author_result,
                    source_manifest=artifact.source_manifest,
                    model_dir=artifact.model_dir,
                    staged_transaction_id=(
                        str(artifact.staged.transaction_id)
                        if artifact.staged is not None
                        else None
                    ),
                    lifecycle=(
                        "staged-pending" if artifact.staged is not None else "unstaged"
                    ),
                ),
            )
            artifacts[item.stable_id] = artifact
            self.dependencies.boundary_hook("after-author", item.stable_id)
        self._family_artifacts.update(artifacts)
        return artifacts

    def _rehydrate_final_authority(
        self,
        item: WorkItem,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
    ) -> Optional[AuthorArtifact]:
        """Return canonical finalized author authority before disposable fallbacks.

        Parameters
        ----------
        item:
            Exact scheduled work generation.
        reducer:
            Locked reducer exposing active authority and artifact history.
        operational:
            Canonical operational ledger for loud stale-cache disclosures.

        Returns
        -------
        AuthorArtifact | None
            Revalidated transaction-backed result, or ``None`` when the exact work
            generation has no finalized artifact authority -- including when its
            recorded authority is a schema-stale reuse-cache entry, which is
            invalidated with a typed operational event so the model routes to
            fresh authorship under the live author-result schema.

        Raises
        ------
        DriverIntegrationError
            If canonical authority is present but incomplete, ambiguous, or corrupt.
        """

        started = time.perf_counter()
        terminal_count = len(reducer.current_view)
        if reducer.artifact_ledger.event_count == 0:
            self._record_ledger_hot_path_metric(
                terminal_count=terminal_count,
                elapsed_seconds=time.perf_counter() - started,
                artifact_event_count=0,
                checkpoint_cache_mode="not-consulted",
            )
            return None
        if not reducer.artifact_ledger.has_final_event(item.stable_id, item.active_work_id):
            # Historical transactions remain immutable audit evidence, but they do
            # not become reconstruction authority for a new work generation.
            self._record_ledger_hot_path_metric(
                terminal_count=terminal_count,
                elapsed_seconds=time.perf_counter() - started,
                artifact_event_count=reducer.artifact_ledger.event_count,
                checkpoint_cache_mode="not-consulted",
            )
            return None
        canonical_root = _canonical_crawler_root(self.paths)
        repository_root = _canonical_repo_root(canonical_root)
        mirrors = MirrorStore(
            self.paths.runtime_root / "mirrors" / "public",
            self.paths.runtime_root / "mirrors" / "private",
            self.paths.runtime_root / "mirrors" / "local",
        )
        artifact_paths = tuple(sorted(self.paths.ledgers.artifacts.parent.glob("*.jsonl"))) or (
            self.paths.ledgers.artifacts,
        )
        try:
            projection, cache_mode = self._validated_artifact_projection(
                reducer,
                artifact_paths=artifact_paths,
                mirrors=mirrors,
                canonical_root=canonical_root,
                repository_root=repository_root,
            )
        except (ArtifactCheckpointError, ValueError) as exc:
            # GLOBAL ledger/store integrity only (per-transaction defects are
            # quarantined inside the projection): a cross-transaction
            # collision, mirror mismatch, or orphan is corruption of the
            # shared store, and silently re-authoring the whole campaign on
            # top of it would be worse than stopping.
            raise DriverIntegrationError(
                f"canonical artifact authority cannot be rehydrated: {exc}"
            ) from exc
        recorded_transaction: Optional[ArtifactTransactionId] = None
        current = reducer.current_view.get(item.stable_id)
        if current is not None and item.requeue_work_id is None and item.refresh_work_id is None:
            authority = current.get("artifact_authority")
            transaction_value = (
                authority.get("transaction_id") if isinstance(authority, Mapping) else None
            )
            if isinstance(transaction_value, str) and transaction_value:
                recorded_transaction = ArtifactTransactionId(transaction_value)
        try:
            transaction = resolve_final_artifact_transaction(
                projection,
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                transaction_id=recorded_transaction,
            )
        except ArtifactRehydrationError as exc:
            # A recorded transaction absent from final authority, or ambiguous
            # unrecorded authority, is this ONE model's defective cache state:
            # invalidate loudly and author fresh, never abort the campaign.
            self._report_identity_stale_artifact_authority(
                item,
                None,
                exc,
                operational,
                transaction_id=(
                    str(recorded_transaction) if recorded_transaction is not None else None
                ),
                recorded_basis_available=False,
            )
            self._record_ledger_hot_path_metric(
                terminal_count=terminal_count,
                elapsed_seconds=time.perf_counter() - started,
                artifact_event_count=reducer.artifact_ledger.event_count,
                checkpoint_cache_mode=cache_mode,
            )
            return None
        if transaction is None:
            self._report_stale_artifact_authority(item, projection, operational)
            self._record_ledger_hot_path_metric(
                terminal_count=terminal_count,
                elapsed_seconds=time.perf_counter() - started,
                artifact_event_count=reducer.artifact_ledger.event_count,
                checkpoint_cache_mode=cache_mode,
            )
            return None
        self._final_artifact_transactions[
            (transaction.stable_id, transaction.work_id, transaction.transaction_id)
        ] = transaction
        inputs = transaction.reconstruction_inputs
        if self.config.only_status is not None and inputs.handoff_execution is None:
            # An operator-invoked handoff resume (only_status) exists to
            # consume this exact authority; without it there is nothing to
            # resume, so this operational precondition stays a hard refusal.
            raise DriverIntegrationError("handoff-authority-unavailable")
        try:
            rehydrated = rehydrate_artifact_transaction(
                transaction,
                mirrors=mirrors,
                staging_root=self.paths.work_root / "rehydrated-artifacts",
            )
            # The fresh author lane creates the model dir unconditionally
            # before any claims land (a terminal recommendation has zero code
            # claims and still owns an empty one), and the checker envelope
            # declares it as a read root that must dereference pre-purchase.
            # Rehydration stages only claim-addressed files, so the driver
            # restores the same canonical author layout it guarantees fresh.
            rehydrated.model_dir.mkdir(parents=True, exist_ok=True)
            raw_result = inputs.author_result
            campaign_id = raw_result.get("campaign_id")
            if not isinstance(campaign_id, str) or campaign_id != _campaign_id_for_item(item):
                raise DriverIntegrationError(
                    "canonical author result campaign differs from active scheduled work"
                )
            # The recorded result revalidates at its OWN recorded identity
            # basis: the prompt/dispatcher/author/intake echoes below were
            # already cross-checked against the transaction's recorded
            # dependency vector and final event by the checkpoint projection
            # above, so stamping them into the envelope reuses history's
            # commitments. Building the DISPATCH envelope here instead
            # re-imposed the live context on finalized transactions -- the
            # rung-11 history-vs-live family: one author-prompt re-pin or
            # dispatcher edit and every finalized artifact refused to
            # rehydrate, aborting the campaign.
            envelope = build_recorded_result_envelope(
                context=reducer.context,
                work_id=item.active_work_id,
                stable_id=item.stable_id,
                campaign_id=campaign_id,
                created_at=self.dependencies.clock(),
                untrusted_hints=item.intake.to_dict(),
                source_manifest=inputs.source_manifest,
                allowed_model_dir=rehydrated.model_dir,
                output_path=rehydrated.root / "author" / "result.json",
                recorded_bindings=raw_result,
                routed_environment_intent=item.route.intent,
                routed_environment_packages=_routed_environment_packages(
                    getattr(self, "registry", None), item.route.intent
                ),
            )
            # Relocation-safe revalidation (Wave-2 batch assembly, W9 x warm
            # redrive): dereference source evidence at the DIGEST-ADDRESSED
            # twin CAS the rehydrator itself staged and verified, never the
            # manifest row's recorded absolute cas_path -- on a copied or
            # relocated campaign that absolute path points inside the deleted
            # origin tree, and before W9 the resulting refusal was silently
            # carried by the work-dir scratch cache; W9's composed retirement
            # then renamed that still-valid cache, so every warm re-drive
            # re-bought author sessions (the pilot 15h-wall class,
            # test_warm_redrive_economics). Genuinely missing or mismatched
            # bytes still refuse fail-closed through the same rehash.
            result = validate_author_result_mapping(
                raw_result, envelope, cas_root=rehydrated.root / "source-cas"
            )
            staged = staged_artifact_for_result(
                reducer.artifact_ledger,
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                author_result_id=result.binding.result_id,
            )
            if staged is None:
                raise DriverIntegrationError(
                    "canonical final transaction lacks its exact staged-private predecessor"
                )
            artifact_type = (
                ActivatedHandoffArtifact if self.config.only_status is not None else AuthorArtifact
            )
            # The transaction's own record-time prompt pins travel with the
            # rehydrated artifact: identity revalidation runs at THIS recorded
            # basis, never at the live prompt files (rung-12 leg-2 crash: the
            # checker-prompt re-pin made the live vet recompute mismatch every
            # pre-rotation artifact and abort the campaign at the first one).
            recorded_basis = (
                RecordedIdentityBasis(
                    author_prompt_sha256=transaction.recorded_author_prompt_identity,
                    checker_prompt_sha256=transaction.recorded_checker_prompt_identity,
                )
                if transaction.recorded_author_prompt_identity is not None
                and transaction.recorded_checker_prompt_identity is not None
                else None
            )
            if recorded_basis is not None:
                rotated = _identity_basis_rotation(recorded_basis)
                if rotated:
                    self._report_identity_stale_artifact_authority(
                        item,
                        None,
                        DriverIntegrationError(
                            "recorded identity basis rotated: " + "; ".join(rotated)
                        ),
                        operational,
                        transaction_id=str(transaction.transaction_id),
                        recorded_basis_available=True,
                    )
                    self._record_ledger_hot_path_metric(
                        terminal_count=terminal_count,
                        elapsed_seconds=time.perf_counter() - started,
                        artifact_event_count=reducer.artifact_ledger.event_count,
                        checkpoint_cache_mode=cache_mode,
                    )
                    return None
            artifact = artifact_type(
                author_result=result,
                source_manifest=dict(inputs.source_manifest),
                model_dir=rehydrated.model_dir,
                staged=staged,
                canonical_code_root=rehydrated.model_dir,
                recorded_identity_basis=recorded_basis,
                **(
                    {
                        "handoff_sha256": str(
                            transaction.reconstruction_inputs.handoff_execution["handoff_sha256"]
                        )
                    }
                    if artifact_type is ActivatedHandoffArtifact
                    and transaction.reconstruction_inputs.handoff_execution is not None
                    else {}
                ),
            )
            self._record_ledger_hot_path_metric(
                terminal_count=terminal_count,
                elapsed_seconds=time.perf_counter() - started,
                artifact_event_count=reducer.artifact_ledger.event_count,
                checkpoint_cache_mode=cache_mode,
            )
            return artifact
        except (
            ArtifactCheckpointError,
            ArtifactRehydrationError,
            DriverIntegrationError,
            ValueError,
        ) as exc:
            # Consumption-side member of the rung-12 crash family: any refusal
            # while materializing or revalidating THIS ONE recorded
            # transaction (custody bytes, claim lineage, campaign binding,
            # recorded-envelope validation, missing staged predecessor) is a
            # defective cache entry for this model alone. Invalidate loudly
            # and author fresh; never abort the campaign.
            self._report_identity_stale_artifact_authority(
                item,
                None,
                exc,
                operational,
                transaction_id=str(transaction.transaction_id),
                recorded_basis_available=(
                    transaction.recorded_author_prompt_identity is not None
                    and transaction.recorded_checker_prompt_identity is not None
                ),
            )
            self._record_ledger_hot_path_metric(
                terminal_count=terminal_count,
                elapsed_seconds=time.perf_counter() - started,
                artifact_event_count=reducer.artifact_ledger.event_count,
                checkpoint_cache_mode=cache_mode,
            )
            return None

    def _report_stale_artifact_authority(
        self,
        item: WorkItem,
        projection: ArtifactCheckpointProjection,
        operational: JsonlLedger,
    ) -> None:
        """Loudly disclose one model's quarantined artifact authority, once per run.

        A stale entry is a transaction the checkpoint quarantined out of
        reusable authority: either its recorded author result no longer
        validates against the LIVE author-result schema (schema-stale -- the
        live schema governs reuse eligibility), or the transaction contradicts
        or fails ITS OWN record (record-stale: a vector-vs-record
        contradiction, rewritten or unreadable reconstruction bytes, an
        interrupted chain). Either way the entry is consumed as a cache miss
        (the caller returns ``None`` and the model routes to fresh
        authorship); this disclosure makes the cache death loud instead of
        silent, naming the stale entry, its flavor, and the violation.

        Parameters
        ----------
        item:
            Scheduled work item whose canonical authority resolved stale.
        projection:
            Validated artifact projection carrying the stale-entry ledger.
        operational:
            Canonical operational ledger receiving the typed event.
        """

        reported: set[tuple[str, str, str]] = getattr(
            self, "_reported_stale_artifact_authorities", set()
        )
        self._reported_stale_artifact_authorities = reported
        for (stable_id, work_id, transaction_id), stale in sorted(
            projection.stale_transactions.items(), key=lambda pair: tuple(map(str, pair[0]))
        ):
            if stable_id != item.stable_id:
                continue
            guard_key = (stable_id, work_id, str(transaction_id))
            if guard_key in reported:
                continue
            reported.add(guard_key)
            schema_markers = (
                "invalid typed author result",
                "invalid author proposal",
                "invalid handoff proposal",
            )
            disposition = (
                ARTIFACT_AUTHORITY_SCHEMA_STALE_DISPOSITION
                if any(marker in stale.reason for marker in schema_markers)
                else ARTIFACT_AUTHORITY_RECORD_STALE_DISPOSITION
            )
            event: JsonObject = {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "created_at": self.dependencies.clock(),
                "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                "status": OperationalEventStatus.HEALTHY.value,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 1},
                "current_environment": item.route.intent,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": {
                    "disposition": disposition,
                    "stable_id": stable_id,
                    "work_id": work_id,
                    "transaction_id": str(transaction_id),
                    "result_schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
                    "violation_summary": stale.reason[:600],
                    "remedy": (
                        "stale canonical-artifact cache entry invalidated; the "
                        "model is routed to fresh authorship under the live "
                        "author-result schema"
                    ),
                },
            }
            event = mint_operational_event(event)
            operational.append(event)

    def _report_identity_stale_artifact_authority(
        self,
        item: WorkItem,
        artifact: Optional[AuthorArtifact],
        failure: Exception,
        operational: JsonlLedger,
        *,
        transaction_id: Optional[str] = None,
        recorded_basis_available: Optional[bool] = None,
    ) -> str:
        """Retire one identity-stale canonical-cache entry, composed (W9).

        The ONE invalidation authority every identity-stale call site routes
        through, in S-U3 crash order: (1) the durable RETIREMENT ROW first --
        this typed operational event, keyed on the exact
        (stable_id, work_id, result_id, transaction_id, cache_identity,
        reason) axes and re-derived from the ledger on every evaluation, never
        from a persisted flag; (2) the atomic evidence-preserving
        ``.stale-<event_id>`` RENAME of the work-dir author-result scratch
        cache (retire-by-rename ruling A20/IC-Q1 -- never unlink; the reaper
        protects renamed evidence); (3) eviction of every in-memory layer
        holding the dead authority (``_family_artifacts``,
        ``_final_artifact_transactions``; the scheduler wave map evicts at its
        own call site). A crash between any two steps converges on re-drive:
        the row is authoritative and steps 2-3 are idempotent derivatives.

        N-REPEAT FUSE: the same retirement key recurring across legs means the
        invalidation is NOT converging -- the paid loop R16 names. The repeat
        ordinal is derived by replaying this ledger's own prior retirement
        rows; the third same-key retirement marks the member
        ``invalidation-nonconvergent`` and the author admission parks it
        (deferred, disclosed) instead of buying another session.

        Status is DEGRADED, never HEALTHY: an invalidation is a degradation
        (DX-36's honest-status rule for the award seam applies here too).

        Parameters
        ----------
        item:
            Scheduled work item whose canonical authority was invalidated.
        artifact:
            The rehydrated artifact, when one was built before invalidation.
        failure:
            Exact refusal or rotation description.
        operational:
            Canonical operational ledger receiving the typed event.
        transaction_id, recorded_basis_available:
            Overrides for the pre-artifact rotation arm.

        Returns
        -------
        str
            ``"retired"``, or ``"invalidation-nonconvergent"`` when this is
            the third-or-later same-key retirement.
        """

        if transaction_id is None and artifact is not None and artifact.staged is not None:
            transaction_id = str(artifact.staged.transaction_id)
        if recorded_basis_available is None:
            recorded_basis_available = (
                artifact is not None and artifact.recorded_identity_basis is not None
            )
        result_id: Optional[str] = None
        if artifact is not None:
            result_id = str(artifact.author_result.binding.result_id)
        prior_rows = 0
        for prior in operational.records:
            details = prior.get("details")
            if not isinstance(details, Mapping):
                continue
            if details.get("disposition") != ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION:
                continue
            if (
                details.get("stable_id") == item.stable_id
                and details.get("work_id") == item.active_work_id
                and details.get("transaction_id") == transaction_id
            ):
                prior_rows += 1
        repeat_ordinal = prior_rows + 1
        nonconvergent = repeat_ordinal >= 3
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.DEGRADED.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": item.route.intent,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": ARTIFACT_AUTHORITY_IDENTITY_STALE_DISPOSITION,
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "result_id": result_id,
                "transaction_id": transaction_id,
                # The invalidated layer is the canonical solve cache; its
                # work-dir scratch derivative is retired by rename below.
                "cache_identity": "C1",
                "recorded_basis_available": recorded_basis_available,
                "repeat_ordinal": repeat_ordinal,
                **({"park": "invalidation-nonconvergent"} if nonconvergent else {}),
                "violation_summary": str(failure)[:600],
                "remedy": (
                    "invalidation is not converging for this member; the "
                    "member is parked this leg instead of re-buying the "
                    "same authorship loop"
                    if nonconvergent
                    else "stale canonical-artifact cache entry invalidated; the "
                    "model is routed to fresh authorship under the live "
                    "author-result contract"
                ),
            },
        }
        event = mint_operational_event(event)
        operational.append(event)
        self._retire_author_scratch_cache(item.stable_id, str(event["event_id"]))
        self._family_artifacts.pop(item.stable_id, None)
        for key in [
            key for key in self._final_artifact_transactions if key[0] == item.stable_id
        ]:
            del self._final_artifact_transactions[key]
        if nonconvergent:
            parked = getattr(self, "_invalidation_nonconvergent_ids", None)
            if parked is None:
                parked = set()
                self._invalidation_nonconvergent_ids = parked
            parked.add(item.stable_id)
            return "invalidation-nonconvergent"
        return "retired"

    def _retire_author_scratch_cache(self, stable_id: str, event_id: str) -> Optional[str]:
        """Atomically retire one work-dir scratch cache by typed rename.

        Evidence-preserving per the A20/IC-Q1 ruling: the cache renames to
        ``driver-author-artifact.json.stale-<event_id>`` in the same
        directory (atomic on POSIX), never unlinks. Idempotent: an absent
        cache is a no-op, and a rename failure leaves the file for the next
        retirement to converge on (the durable row is the authority; this is
        its derivative).

        Returns
        -------
        str | None
            The retired filename, or ``None`` when nothing was renamed.
        """

        cache = author_result_cache_path(self.paths.work_root, stable_id)
        if not cache.is_file():
            return None
        target = cache.with_name(f"{cache.name}.stale-{event_id.replace(':', '-')}")
        try:
            os.replace(cache, target)
        except OSError:
            return None
        return target.name

    def _checkpoint_validation_memo(self) -> ValidationMemo:
        """Return this run's persistent validation memo, loading it once.

        Returns
        -------
        ValidationMemo
            Positive-outcome memo below the runtime derived store. Load-time
            degrades (corrupt, torn, unreadable store) surface as
            ``checkpoint-cache-event`` instrumentation rows and fall back to
            an empty memo: live derivation, never an error surface change.
        """

        memo = getattr(self, "_checkpoint_memo_instance", None)
        if memo is None:
            memo = ValidationMemo.open(self.paths.runtime_root / "derived" / "checkpoint-cache")
            self._checkpoint_memo_instance = memo
            for diagnostic in memo.diagnostics:
                self._record_checkpoint_cache_event(dict(diagnostic))
        return memo

    def _flush_checkpoint_memo(self, memo: ValidationMemo) -> None:
        """Persist newly memoized outcomes; a write failure costs, never fails.

        Parameters
        ----------
        memo:
            The run's validation memo.
        """

        before = len(memo.diagnostics)
        memo.flush()
        for diagnostic in memo.diagnostics[before:]:
            self._record_checkpoint_cache_event(dict(diagnostic))

    def _validated_artifact_projection(
        self,
        reducer: CanonicalReducer,
        *,
        artifact_paths: Sequence[Path],
        mirrors: MirrorStore,
        canonical_root: Path,
        repository_root: Path,
    ) -> tuple[ArtifactCheckpointProjection, str]:
        """Validate artifact authority through the final-authority checkpoint cache.

        Default-on, three modes, all sound under the reducer's exclusive
        writer lock on the artifact shard:

        - ``"reserve-hit"``: the exact same locked
          :class:`ArtifactEventLedger` object at an unchanged append
          generation pins the exact validated bytes, so the retained
          projection is served as-is.
        - ``"incremental"``: the same locked single-shard ledger advanced by
          appends; only the appended suffix folds
          (:meth:`ArtifactCheckpointIncrement.advance`), with every touched
          transaction re-validated fully live. Any advance refusal is
          treated as a cache death -- disclosed as a
          ``checkpoint-cache-event`` and answered with full live validation
          -- and ``MENAGERIE_CRAWLER_CHECKPOINT_CACHE_VERIFY=1`` shadows
          every advance with a full live derivation, failing loud toward the
          live result on disagreement.
        - ``"full"``: complete live validation (first consultation, foreign
          ledger, multi-shard set, fallback), threading the persistent
          validation memo whose hits are pure-function jsonschema skips over
          live-reverified content.

        ``MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION=0`` disables every layer and
        restores uncached full validation on each call.

        Parameters
        ----------
        reducer:
            Locked canonical reducer and artifact append generation.
        artifact_paths, mirrors, canonical_root, repository_root:
            Unchanged inputs to full checkpoint validation.

        Returns
        -------
        tuple[ArtifactCheckpointProjection, str]
            Validated projection and the serving mode (``"full"``,
            ``"reserve-hit"``, or ``"incremental"``).
        """

        if os.environ.get("MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION") == "0":
            projection = validate_artifact_checkpoint(
                artifact_paths,
                context=reducer.context,
                mirrors=mirrors,
                canonical_root=canonical_root,
                repository_root=repository_root,
            )
            return projection, "full"
        memo = self._checkpoint_validation_memo()
        binder = getattr(reducer.artifact_ledger, "bind_validation_memo", None)
        if callable(binder):
            binder(memo)
        generation = reducer.artifact_ledger.event_count
        paths = tuple(artifact_paths)
        cached = self._artifact_checkpoint_cache
        if cached is not None and cached[0] is reducer.artifact_ledger and cached[2] == paths:
            cached_generation = cached[1]
            increment = cached[3]
            if cached_generation == generation:
                return increment.projection, "reserve-hit"
            ledger_path = getattr(reducer.artifact_ledger, "path", None)
            if (
                generation > cached_generation
                and len(paths) == 1
                and ledger_path is not None
                and Path(ledger_path) == paths[0]
            ):
                try:
                    suffix = reducer.artifact_ledger.events_since(cached_generation)
                    projection = increment.advance(suffix)
                except (ArtifactCheckpointError, ArtifactTransitionError, ValueError) as exc:
                    # Fail toward live: an advance refusal may be a cache
                    # defect rather than genuine corruption, so the answer is
                    # a loud disclosure plus full live validation below. If
                    # the ledger is truly corrupt, the full validation raises
                    # exactly like today.
                    self._artifact_checkpoint_cache = None
                    self._record_checkpoint_cache_event(
                        {
                            "reason": "incremental-advance-failed",
                            "detail": str(exc)[:300],
                        }
                    )
                else:
                    # Amortized persistence: the memo is a derived artifact,
                    # so deferring durability costs at most one future live
                    # revalidation, while a per-advance fsync was measurable
                    # hot-path latency.
                    if memo.pending_count >= 256:
                        self._flush_checkpoint_memo(memo)
                    if os.environ.get("MENAGERIE_CRAWLER_CHECKPOINT_CACHE_VERIFY") == "1":
                        live = validate_artifact_checkpoint(
                            paths,
                            context=reducer.context,
                            mirrors=mirrors,
                            canonical_root=canonical_root,
                            repository_root=repository_root,
                            memo=memo,
                        )
                        if live != projection:
                            self._artifact_checkpoint_cache = None
                            self._record_checkpoint_cache_event(
                                {"reason": "incremental-divergence"}
                            )
                            return live, "full"
                    self._artifact_checkpoint_cache = (
                        reducer.artifact_ledger,
                        generation,
                        paths,
                        increment,
                    )
                    return projection, "incremental"
        increment = ArtifactCheckpointIncrement.start(
            paths,
            context=reducer.context,
            mirrors=mirrors,
            canonical_root=canonical_root,
            repository_root=repository_root,
            memo=memo,
        )
        self._flush_checkpoint_memo(memo)
        self._artifact_checkpoint_cache = (
            reducer.artifact_ledger,
            generation,
            paths,
            increment,
        )
        return increment.projection, "full"

    def _record_ledger_hot_path_metric(
        self,
        *,
        terminal_count: int,
        elapsed_seconds: float,
        artifact_event_count: int,
        checkpoint_cache_mode: str,
    ) -> None:
        """Append local slope telemetry without changing canonical record semantics.

        Parameters
        ----------
        terminal_count, elapsed_seconds:
            Per-model authority time paired with current terminal count.
        artifact_event_count:
            Append generation that invalidates the reserve cache.
        checkpoint_cache_mode:
            How authority was served: ``"full"``, ``"reserve-hit"``,
            ``"incremental"``, or ``"not-consulted"`` for early exits that
            never reach checkpoint validation.
        """

        self._ledger_hot_path_samples.append((terminal_count, elapsed_seconds))
        slope = _linear_slope(self._ledger_hot_path_samples)
        payload = {
            "metric": "final-authority-per-model-seconds-vs-terminal-count",
            "terminal_count": terminal_count,
            "elapsed_seconds": elapsed_seconds,
            "artifact_event_count": artifact_event_count,
            "checkpoint_cache_enabled": (
                os.environ.get("MENAGERIE_CRAWLER_ARTIFACT_MEMOIZATION") != "0"
            ),
            "checkpoint_cache_hit": checkpoint_cache_mode in {"reserve-hit", "incremental"},
            "checkpoint_cache_mode": checkpoint_cache_mode,
            "slope_seconds_per_terminal": slope,
            "positive_slope_detected": slope is not None and slope > 0.0,
        }
        self._append_instrumentation_row(payload)

    def _record_checkpoint_cache_event(self, fields: Mapping[str, Any]) -> None:
        """Disclose one checkpoint-cache degrade or divergence locally.

        Parameters
        ----------
        fields:
            Typed event fields; ``reason`` names the degrade class.
        """

        self._append_instrumentation_row({"metric": "checkpoint-cache-event", **fields})

    def _append_instrumentation_row(self, payload: Mapping[str, Any]) -> None:
        """Durably append one local instrumentation row.

        Parameters
        ----------
        payload:
            Complete JSON row.
        """

        path = self.paths.runtime_root / "instrumentation" / "ledger-hot-path.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(descriptor, canonical_json_bytes(payload) + b"\n")
            durable_fsync(descriptor)
        finally:
            os.close(descriptor)

    @property
    def _checker_lane_parked_ids(self) -> set[str]:
        """Stable IDs this run parked on the checker-lane deferral valve.

        Lazily attached exactly like the machine-deferral row stash, so the
        mixin never depends on a specific ``__init__`` running first. Parked
        models are withheld from every later lane this run (fidelity gating,
        environment, worker) and requeue on the next run.
        """

        parked = getattr(self, "_checker_lane_parked_id_set", None)
        if parked is None:
            parked = set()
            self._checker_lane_parked_id_set = parked
        return parked

    @property
    def _checker_lane_park_streak(self) -> list[str]:
        """Models parked consecutively, with no successfully routed gate between.

        The T4-T3 tripwire's evidence: each park is a distinct model (a model
        parks at most once per run), so a streak past one full batch width is
        necessarily cross-batch -- the LANE is not answering anyone, and every
        further park burns a failed session ladder for a verdict that will not
        come. Lazily attached like the parked-ID set.
        """

        streak = getattr(self, "_checker_lane_park_streak_list", None)
        if streak is None:
            streak = []
            self._checker_lane_park_streak_list = streak
        return streak

    def _note_checker_gate_routed(self) -> None:
        """Reset the consecutive-parks streak: the lane just delivered a gate."""

        self._checker_lane_park_streak.clear()

    def _downshift_or_park_checker_batch(
        self,
        batch: Sequence[AuthorArtifact],
        failure: Exception,
        *,
        ladder: deque[tuple[AuthorArtifact, ...]],
        failed_sizes: dict[str, list[int]],
        items_by_id: Mapping[str, WorkItem],
        pending_ids: set[str],
        operational: JsonlLedger,
        failure_site: str,
        artifacts: Mapping[str, AuthorArtifact],
        reducer: CanonicalReducer,
        state: JsonObject,
    ) -> None:
        """Route one checker-lane batch failure: downshift, park, or terminalize.

        A TRANSIENT checker-lane failure carries NO verdict about any member
        model, so no member may terminalize on it (rung 10: one over-budget
        20-item batch annihilated 20 models, including runs-proven m4334). Above
        the floor the batch splits along the shared halving ladder and the halves
        are re-asked; at the floor the single model parks on the durable
        checker-lane deferral valve -- accounted by the terminal partition,
        requeued on the next run.

        The ONE exception is the closed machine-classified QUOTA/budget-exhaustion
        class (:func:`_is_checker_session_exhaustion`): generation-scoped
        exhaustion is permanent for the generation, so the run-scoped park's
        requeue promise is false -- the model re-enters and re-parks on every
        future leg with no reviewable head a grant could bind (the pilot's five
        no-exit parks, 2026-08-21). At the floor that class terminalizes
        ``failed:accuracy-gate / checker-session-exhaustion`` with mandatory
        human review instead of parking.

        Parameters
        ----------
        batch:
            Exact sub-batch whose checker round failed.
        failure:
            Lane failure that produced no usable gate for this sub-batch.
        ladder:
            Live sub-batch work queue; halves are pushed back onto it.
        failed_sizes:
            Per-model trail of failed sub-batch sizes, for the parked row.
        items_by_id:
            Scheduled work items keyed by stable ID.
        pending_ids:
            Live pending set; parked models are discharged from it.
        operational:
            Locked operational ledger for the health events.
        failure_site:
            ``checker-dispatch`` or ``gate-routing``, for the record.
        artifacts:
            Live author artifacts keyed by stable ID, for the exhaustion terminal.
        reducer:
            Canonical authority, for the exhaustion terminal.
        state:
            Mutable driver state, for the exhaustion terminal.
        """

        batch_ids = tuple(str(artifact.proposal["stable_id"]) for artifact in batch)
        for stable_id in batch_ids:
            failed_sizes.setdefault(stable_id, []).append(len(batch))
        if len(batch) == 1 and failure_site == "gate-routing":
            # W4-C seam: a checker-authored contract violation at gate ROUTING has
            # spent no re-ask at all -- the operator wrapper's bounded re-asks only
            # cover its own dispatch-side validation. The checker is stochastic
            # (rung 3: one flubbed identity echo among clean siblings), so a
            # floor-one routing refusal earns exactly ONE fresh single-item
            # re-ask before the model parks; the routing error is durable on the
            # operational ledger beside the re-ask event.
            if failed_sizes[batch_ids[0]].count(1) < 2:
                ladder.append(tuple(batch))
                created_at = self.dependencies.clock()
                reask: JsonObject = {
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
                        "disposition": "checker-gate-routing-reask",
                        "stable_id": batch_ids[0],
                        "error": str(failure)[:STDIO_TAIL_MAX_CHARS],
                    },
                }
                reask = mint_operational_event(reask)
                operational.append(reask)
                return
        if len(batch) > 1:
            halves = downshift_metadata_items(tuple(batch))
            ladder.extendleft(tuple(reversed(halves)))
            created_at = self.dependencies.clock()
            event: JsonObject = {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "created_at": created_at,
                "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                "status": OperationalEventStatus.HEALTHY.value,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": len(batch)},
                "current_environment": None,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": {
                    "disposition": "checker-batch-downshift",
                    "failure_site": failure_site,
                    "from_size": len(batch),
                    "to_sizes": [len(half) for half in halves],
                    "stable_ids": list(batch_ids),
                    "error": str(failure)[:STDIO_TAIL_MAX_CHARS],
                },
            }
            event = mint_operational_event(event)
            operational.append(event)
            return
        stable_id = batch_ids[0]
        if _is_checker_session_exhaustion(failure):
            self._terminalize_checker_session_exhaustion(
                items_by_id[stable_id],
                artifacts[stable_id],
                failure,
                reducer=reducer,
                operational=operational,
                state=state,
            )
            pending_ids.discard(stable_id)
            return
        self._record_checker_lane_deferral(
            items_by_id[stable_id],
            failure,
            attempted_batch_sizes=tuple(failed_sizes.get(stable_id, [1])),
            operational=operational,
            failure_site=failure_site,
        )
        pending_ids.discard(stable_id)

    def _terminalize_checker_session_exhaustion(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        failure: Exception,
        *,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> None:
        """Terminalize one model whose checker session/wall budget is exhausted.

        The honest terminal for the closed QUOTA-exhaustion park cause
        (:func:`_is_checker_session_exhaustion`): a witnessing driver attempt at
        the exact closed reason, then a ``failed:accuracy-gate /
        checker-session-exhaustion`` terminal under mandatory human review --
        exactly as reviewable and grant-re-drivable as every other terminal.
        Like ``_terminalize_accuracy_gate``, the assembled record carries the
        work generation's full persisted attempt history: under execute-first
        triage those attempts ARE the reducer's proof context, and a candidate
        without them contradicts the authority the reducer re-derives.
        """

        operational.append(
            _model_lane_failure_event(
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                status_code="failed:accuracy-gate",
                reason_code=CHECKER_SESSION_EXHAUSTION_REASON_CODE,
                exc=failure,
                run_id=self.config.run_id,
                machine_id=self.config.machine_id,
                created_at=self.dependencies.clock(),
            )
        )
        attempt = _driver_failure_attempt(
            item,
            artifact,
            "accuracy-gate",
            CHECKER_SESSION_EXHAUSTION_REASON_CODE,
            failure,
            self.config,
            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
            environment=None,
            created_at=self.dependencies.clock(),
        )
        persisted_attempt = reducer.append_attempt(attempt).record
        persisted_attempts = _matching_model_attempts(
            self.paths.ledgers.attempts, artifact.proposal
        )
        self._terminalize(
            item,
            artifact,
            "failed:accuracy-gate",
            CHECKER_SESSION_EXHAUSTION_REASON_CODE,
            str(failure),
            persisted_attempts,
            reducer,
            operational,
            state,
            human_review=True,
            root_cause_fingerprint=str(
                persisted_attempt.get("error", {}).get("root_cause_fingerprint")
            ),
        )

    def _scope_item_contract_failure(
        self,
        batch: Sequence[AuthorArtifact],
        failure: CheckerItemContractError,
        *,
        ladder: deque[tuple[AuthorArtifact, ...]],
        failed_sizes: dict[str, list[int]],
        items_by_id: Mapping[str, WorkItem],
        pending_ids: set[str],
        operational: JsonlLedger,
        artifacts: Mapping[str, AuthorArtifact],
        reducer: CanonicalReducer,
        state: JsonObject,
    ) -> None:
        """Scope one item-attributed contract failure to exactly the named models.

        BL-10 (c) / audit B-3. The typed refusal names the items whose contract
        failed; everything else in the batch is INNOCENT and re-batches together
        (one fresh session, no halving-tree tax, no mis-attributed park). Each
        named model rides one bounded singleton re-adjudication -- a fresh
        checker round can legitimately answer differently (the rung-9 checker
        emitted complete ``checked_source_ids`` on its single-item rounds after
        flubbing them batch-wide) -- and a named SINGLETON that fails again
        parks immediately with truthful attribution
        (``author-evidence-defect`` for the write-contract class).

        Named IDs outside the batch, or an empty intersection, fall back to the
        whole-batch downshift route: a mis-attributed refusal must never
        silently pass anyone.
        """

        batch_ids = tuple(str(artifact.proposal["stable_id"]) for artifact in batch)
        named = set(failure.stable_ids) & set(batch_ids)
        if not named or set(failure.stable_ids) - set(batch_ids):
            self._downshift_or_park_checker_batch(
                batch,
                failure,
                ladder=ladder,
                failed_sizes=failed_sizes,
                items_by_id=items_by_id,
                pending_ids=pending_ids,
                operational=operational,
                failure_site="checker-dispatch",
                artifacts=artifacts,
                reducer=reducer,
                state=state,
            )
            return
        siblings = tuple(
            artifact
            for artifact in batch
            if str(artifact.proposal["stable_id"]) not in named
        )
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": len(batch)},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "checker-item-contract-scoped",
                "stable_ids": sorted(named),
                "sibling_count": len(siblings),
                "author_side": failure.author_side,
                "error": str(failure)[:STDIO_TAIL_MAX_CHARS],
            },
        }
        operational.append(mint_operational_event(event))
        if siblings:
            ladder.appendleft(siblings)
        for artifact in batch:
            stable_id = str(artifact.proposal["stable_id"])
            if stable_id not in named:
                continue
            failed_sizes.setdefault(stable_id, []).append(len(batch))
            if len(batch) == 1:
                self._record_checker_lane_deferral(
                    items_by_id[stable_id],
                    failure,
                    attempted_batch_sizes=tuple(failed_sizes.get(stable_id, [1])),
                    operational=operational,
                    failure_site="checker-dispatch",
                )
                pending_ids.discard(stable_id)
            else:
                ladder.append((artifact,))

    def _record_checker_lane_deferral(
        self,
        item: WorkItem,
        failure: Exception,
        *,
        attempted_batch_sizes: tuple[int, ...],
        operational: JsonlLedger,
        failure_site: str,
        gate_kind: str = "metadata_batch",
        failure_class: Optional[str] = None,
        count_in_streak: bool = True,
    ) -> None:
        """Append one durable checker-lane deferral row plus its campaign-health event.

        ``failure_class`` overrides the exception-derived class for parks minted
        by a routing DECISION rather than a lane exception (the disclosure-debt
        ceiling). ``count_in_streak=False`` exempts verdict-shaped per-model
        parks (disclosure debt, wall budget) from the lane-weather streak
        breaker: they are facts about one model, not evidence the lane has
        stopped answering anyone.
        """

        created_at = self.dependencies.clock()
        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = checker_lane_deferral_path(records_root)
        # BL-10 (f): record the adjudication identity the lane failed under, so
        # the park's release predicate is causal (holds while the identity is
        # unchanged and the recidivism budget is spent; releases on rotation).
        identity = lane_adjudication_identity(self.dependencies.checker, gate_kind, self.config)
        identity_fields = (
            {
                "provider": identity.provider,
                "model": identity.model,
                "version": identity.version,
                "prompt_sha256": identity.prompt_sha256,
            }
            if identity is not None
            else None
        )
        persisted = append_checker_lane_deferral_row(
            ledger_path,
            build_checker_lane_deferral_row(
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                name=item.intake.name,
                campaign_id=bound_campaign_id(self.config),
                run_id=self.config.run_id,
                machine_id=self.config.machine_id,
                created_at=created_at,
                gate_kind=gate_kind,
                failure_class=failure_class or _checker_lane_failure_class(failure),
                failure_evidence=str(failure),
                attempted_batch_sizes=attempted_batch_sizes,
                checker_identity=identity_fields,
            ),
        )
        _stash_machine_deferral_row(self, item.stable_id, persisted)
        self._checker_lane_parked_ids.add(item.stable_id)
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
                "disposition": "checker-lane-parked",
                "ledger": str(ledger_path.relative_to(records_root)),
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "failure_site": failure_site,
                "gate_kind": gate_kind,
                "failure_class": persisted["checker_lane"]["failure_class"],
                "attempted_batch_sizes": list(attempted_batch_sizes),
                "error": str(failure)[:STDIO_TAIL_MAX_CHARS],
            },
        }
        event = mint_operational_event(event)
        operational.append(event)
        if not count_in_streak:
            return
        streak = self._checker_lane_park_streak
        streak.append(item.stable_id)
        if len(streak) >= CHECKER_LANE_PARK_STREAK_LIMIT:
            # The park above is already durable and stays supersedable; going
            # LOUD here stops the leg from being converted into wrong-blame
            # parks one bought session ladder at a time. Retryable keeps the
            # campaign resumable once the lane weather is fixed.
            raise RetryableOperatorError(
                f"checker lane parked {len(streak)} consecutive models "
                f"({', '.join(streak)}) with no successfully routed gate in "
                "between -- cross-batch evidence that the checker lane is not "
                "answering anyone. Fix the lane (credentials, provider health, "
                "wrapper) and resume; every parked model requeues on the next run"
            )

    def _record_scheduler_quarantine(
        self,
        item: WorkItem,
        *,
        site: str,
        reason_code: str,
        fault: object,
        operational: JsonlLedger,
    ) -> JsonObject:
        """Quarantine ONE member behind a durable typed park; the leg continues.

        BL-8 (a)/(b): the blast radius of a single member's fault is that
        member. The row is durable and partition-accounted
        (``deferred:scheduler-fault-quarantined``), the member is stashed as
        machine-deferred so this leg treats it as settled, and the fault is
        disclosed on the operational ledger. No model record is minted -- the
        fault may be the scheduler's own routing, so no verdict exists.

        Parameters
        ----------
        item:
            Exact scheduled work generation being isolated.
        site:
            Closed :data:`~menagerie.crawler.scheduler_parks.SCHEDULER_QUARANTINE_SITES`
            member naming where the decision was made.
        reason_code:
            Closed :data:`~menagerie.crawler.scheduler_parks.SCHEDULER_QUARANTINE_REASONS`
            member naming why.
        fault:
            The exception (or diagnosis string) that drove the decision.
        operational:
            Locked operational ledger receiving the campaign-health event.

        Returns
        -------
        dict[str, Any]
            The persisted quarantine row.
        """

        created_at = self.dependencies.clock()
        records_root = self.paths.ledgers.models.parent.parent
        exception_type = (
            f"{type(fault).__module__}.{type(fault).__qualname__}"
            if isinstance(fault, BaseException)
            else None
        )
        built = build_scheduler_quarantine_row(
            stable_id=item.stable_id,
            work_id=item.active_work_id,
            name=item.intake.name,
            campaign_id=bound_campaign_id(self.config),
            run_id=self.config.run_id,
            machine_id=self.config.machine_id,
            created_at=created_at,
            site=site,
            reason_code=reason_code,
            fault_evidence=str(fault),
            exception_type=exception_type,
        )
        persisted = append_scheduler_quarantine_row(
            scheduler_quarantine_path(records_root), built
        )
        _stash_machine_deferral_row(self, item.stable_id, persisted)
        # W12: the row append is idempotent per work generation, so a re-park
        # observation against an incumbent live row is a REUSED disclosure
        # that charges NOTHING against the retry budget. Emitting the charging
        # disposition unconditionally was the 28-events-vs-22-rows drift: the
        # budget counter counted observations, not rows.
        reused = str(persisted.get("row_sha256")) != str(built.get("row_sha256"))
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
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
                "disposition": (
                    SCHEDULER_QUARANTINE_REUSED_DISPOSITION
                    if reused
                    else SCHEDULER_QUARANTINE_PARK_EVENT_DISPOSITION
                ),
                "site": site,
                "reason_code": reason_code,
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "exception_type": exception_type,
                "message": str(fault)[:STDIO_TAIL_MAX_CHARS],
                **({"reused_row_sha256": str(persisted.get("row_sha256"))} if reused else {}),
            },
        }
        operational.append(mint_operational_event(event))
        return persisted

    def _record_checker_reuse(
        self,
        operational: JsonlLedger,
        *,
        gate_kind: str,
        stable_ids: Sequence[str],
        reused_from: Mapping[str, Any],
    ) -> None:
        """Append one campaign-health event for a sessionless verdict reuse.

        Live buys are committed by :meth:`_admit_checker_spend` before dispatch.
        This separate event retains the prior gate binding so throughput reports
        can quantify sessions saved without mistaking reuse for paid work.
        """

        created_at = self.dependencies.clock()
        details: JsonObject = {
            "disposition": "checker-verdict-reused",
            "gate_kind": gate_kind,
            "stable_ids": list(stable_ids),
            "batch_size": len(stable_ids),
        }
        details["reused_from"] = dict(reused_from)
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": len(stable_ids)},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": details,
        }
        event = mint_operational_event(event)
        operational.append(event)

    def _paid_session_authority(self) -> PaidSessionAuthority:
        """Return this driver's sole checker/author money authority."""

        authority = getattr(self, "_paid_session_authority_instance", None)
        if authority is None:
            authority = PaidSessionAuthority(
                self.paths.paid_session_ledger,
                runtime_root=self.paths.runtime_root,
                clock=self.dependencies.clock,
            )
            authority.initialize_legacy_prefix(self.paths.operational_ledger)
            self._paid_session_authority_instance = authority
        return authority

    def _paid_generation_id(self) -> str:
        """Return the already-open durable runaway epoch or refuse admission."""

        generation = getattr(self, "_leg_generation", None)
        generation_id = getattr(generation, "generation_id", None)
        if not isinstance(generation_id, str) or not generation_id:
            raise DriverIntegrationError(
                "paid-session reserve requires an open durable leg generation"
            )
        return generation_id

    def _paid_author_attempts(self, stable_id: str, work_id: str) -> int:
        """Return one model's cached O(1) chargeable author ATTEMPT projection.

        Attempt groups, not raw guard sessions: this is what canonical
        ``budget.author_sessions_used`` discloses, so it shares its unit with
        ``author_sessions_max`` (the attempt allowance).  Raw guard sessions
        remain readable from the paid ledger's own QUOTA projection.
        """

        return (
            self._paid_session_authority()
            .state()
            .quota_attempt_groups.get(
                (bound_campaign_id(self.config), stable_id, work_id, "author"), 0
            )
        )

    def _reap_expired_dead_custody(self) -> tuple[str, ...]:
        """Force-settle bound custody whose custodian is machine-provably dead.

        The custody deadline reaper (legs 8 + 12, 2026-08-21): a paid subtree
        that died WITHOUT settling used to hold its custody binding until the
        next leg start, and mid-leg every consumer of the lease waited on it
        forever.  This pass runs at the writer's own admission boundaries and
        at leg start, and is deliberately narrower than the restart ladder:

        * only BOUND leases participate -- never-bound reservations belong to
          startup reconciliation, whose provider-key ladder already settles
          them;
        * only leases past their durable reservation ``deadline_at`` plus the
          named ``CUSTODY_REAPER_GRACE_SECONDS`` are even PROBED, so the
          common case is one cheap timestamp pass over the outstanding set --
          measured on the campaign clock (``dependencies.clock``), the same
          authority that minted ``deadline_at``, never the host wall clock;
        * a LIVE custodian is never reaped on time alone: the flock (probed
          first, authoritative and cheap) must be ``FREE`` AND the process
          group must be enumerably EMPTY -- two independent kernel signals
          through :func:`custodian_provably_dead`.  ``UNANSWERABLE`` on either
          skips, fail-closed;
        * every settlement input is replay-derived with zero producer input:
          the provider answers for the durable idempotency key in both
          directions, the machine wall floor is the durable reservation age,
          and the closed ``failed-custody-reaped`` disposition is re-weighed
          by the fold on every replay -- sessions and wall stay charged, only
          the ATTEMPT projection refunds;
        * a typed refusal (adapter outage, a settle raced by the owning
          dispatch thread) skips the lease, which stays DISCLOSED for the next
          boundary or leg start -- the reaper delays nothing and wedges
          nothing.

        Returns
        -------
        tuple[str, ...]
            Spend ids this pass force-settled.
        """

        authority = self._paid_session_authority()
        leases = authority.outstanding_custody_leases()
        if not leases:
            return ()
        reconciler = getattr(self.dependencies, "provider_reconciler", None)
        # The deadline gate must read the SAME clock authority that minted
        # ``deadline_at`` (the campaign clock, ``dependencies.clock``). Reading
        # the host wall clock here voided the gate under any injected clock:
        # a frozen dry-run clock made every lease "expired" at birth, so the
        # reaper raced the owning dispatch thread's inter-guard custody window
        # and force-settled a LIVE attempt's reservation -- the owner's own
        # settle then raised "already settled", classified session-crashed,
        # and terminalized the model failed:author with no retry (the
        # session-death rehearsal's Wave-3-boundary red, 2026-08-25). In
        # production the campaign clock IS the wall clock, so this changes
        # nothing there.
        try:
            now = datetime.fromisoformat(
                str(self.dependencies.clock()).replace("Z", "+00:00")
            )
        except ValueError:
            return ()
        if now.tzinfo is None:
            return ()
        reaped: list[str] = []
        for lease in leases:
            if not lease.is_bound:
                continue
            try:
                deadline = datetime.fromisoformat(lease.deadline_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if deadline.tzinfo is None:
                continue
            if now < deadline + timedelta(seconds=CUSTODY_REAPER_GRACE_SECONDS):
                continue
            # Flock first: a live session holds it, so the common live case
            # never pays the process-group enumeration.
            lock_state = subtree_liveness(
                lease.lock_path, expected_identity=lease.lock_identity
            )
            if lock_state != FREE:
                continue
            group_state = process_group_liveness(lease.leader_pgid)
            if not custodian_provably_dead(lock_state, group_state):
                continue
            if reconciler is None:
                continue
            # Money first: the settle queries the provider and refuses typed
            # without writing anything, so an adapter outage leaves the WHOLE
            # lease (binding and reservation) disclosed instead of stranding a
            # released-but-unsettled row for the generic chargeable ladder.
            # Settling while still bound is the lane's normal mid-leg shape (a
            # wall-killed session settles before its custody is released), and
            # the release lands second exactly as leg start orders it.
            try:
                authority.settle_reaped(
                    lease.spend_id,
                    query=reconciler,
                    local_outcome="failed-custody-reaped",
                )
            except (PaidSessionError, ValueError):
                # Adapter outage, or the owning dispatch thread settled first:
                # either way the lease stays disclosed and nothing is wedged.
                continue
            try:
                authority.release_reconciled_custody(lease, outcome="verified-exit")
            except ValueError:
                # Raced by another release; the binding stays disclosed for the
                # leg-start ladder, and the typed settlement above stands.
                continue
            reaped.append(lease.spend_id)
        return tuple(reaped)

    def _reconcile_paid_sessions_before_admission(self) -> None:
        """Reap disclosed paid subtrees and reconcile every provider-key ambiguity."""

        # The custody deadline reaper runs FIRST so a bound-dead expired lease
        # settles with its closed machine infrastructure classification (the
        # attempt refunds) instead of falling through to the generic
        # provider-key ladder's chargeable ``unknown`` disclosure.
        self._reap_expired_dead_custody()
        authority = self._paid_session_authority()
        for lease in authority.outstanding_custody_leases():
            self._reconcile_paid_custody_lease(authority, lease)
        reconciler = self.dependencies.provider_reconciler
        # D-1/DX-75/B2-X-V12: the STANDING negative control. Runs at EVERY
        # admission-boundary reconcile -- outstanding ambiguity or not -- for
        # the production command-adapter wiring: a well-formed canary key no
        # reservation names must be REFUSED; any answer proves a constant-
        # accept-class adapter is wired and refuses the leg typed. 200a's two
        # adapters were observationally identical on every ledger row; this
        # probe is the only falsifiable test of which one is wired. In-process
        # rig reconcilers are not command wiring and are not probed.
        if isinstance(reconciler, CommandProviderReconciler):
            try:
                verify_reconciler_negative_control(reconciler)
            except ReconcilerNegativeControlError as exc:
                raise DriverIntegrationError(
                    f"paid-session reconciler failed its negative control: {exc}"
                ) from exc
            except PaidSessionError as exc:
                raise DriverIntegrationError(
                    f"paid-session provider reconciliation is unavailable: {exc}"
                ) from exc
        outstanding = authority.outstanding_spend_ids()
        if not outstanding:
            return
        if reconciler is None:
            raise DriverIntegrationError(
                "unsettled paid-session reservations require provider idempotency-key "
                "reconciliation before admission: " + ", ".join(outstanding)
            )
        try:
            authority.reconcile(reconciler)
        except PaidSessionError as exc:
            # Typed, and never a DURABLE block: reconciliation writes nothing on
            # failure and the whole ladder re-runs at the next leg start, so a
            # broken or transiently unavailable adapter delays admission instead
            # of wedging the campaign behind a manual drain.
            raise DriverIntegrationError(
                f"paid-session provider reconciliation is unavailable: {exc}"
            ) from exc
        remaining = authority.outstanding_spend_ids()
        if remaining:
            raise DriverIntegrationError(
                "provider reconciliation left paid-session ambiguity: "
                + ", ".join(remaining)
            )

    def _reconcile_paid_custody_lease(
        self, authority: PaidSessionAuthority, lease: PaidCustodyLease
    ) -> None:
        """Reap, release, or fail-closed-disclose one outstanding paid lease.

        Both kernel oracles are consulted, and an UNANSWERABLE ``flock`` (its
        pathname unlinked, or now naming a different inode than the binding
        pinned) is never read as death: one ``rm`` under the runtime root must
        not be able to retire a live credential-bearing subtree.
        """

        lock_state = subtree_liveness(lease.lock_path, expected_identity=lease.lock_identity)
        group_state = process_group_liveness(lease.leader_pgid)
        identity_matches = False
        if (
            lock_state == ALIVE
            and lease.leader_pid is not None
            and lease.leader_pgid is not None
            and lease.leader_start_token is not None
        ):
            try:
                observed_pgid: Optional[int] = os.getpgid(lease.leader_pid)
            except OSError:
                observed_pgid = None
            identity_matches = (
                observed_pgid == lease.leader_pgid == lease.leader_pid
                and process_start_token(lease.leader_pid) == lease.leader_start_token
            )
        deadline = datetime.fromisoformat(lease.deadline_at.replace("Z", "+00:00"))
        decision = custody_recovery_decision(
            lease,
            current_boot_id=current_boot_id(),
            liveness=lock_state,
            group_liveness=group_state,
            identity_matches=identity_matches,
            past_deadline=datetime.now(timezone.utc) >= deadline,
        )
        if decision.action == "reap-verified-group":
            assert lease.leader_pid is not None
            assert lease.leader_pgid is not None
            assert lease.leader_start_token is not None
            signal_verified_process_group(
                lease.leader_pid, lease.leader_pgid, lease.leader_start_token
            )
            release_deadline = time.monotonic() + PAID_SUBTREE_REAP_GRACE_SECONDS
            while time.monotonic() < release_deadline:
                if (
                    subtree_liveness(lease.lock_path, expected_identity=lease.lock_identity)
                    == FREE
                    and process_group_liveness(lease.leader_pgid) == FREE
                ):
                    authority.release_reconciled_custody(lease, outcome="verified-reap")
                    return
                time.sleep(0.05)
            raise DriverIntegrationError(
                f"paid lease {lease.lease_id} remained alive after verified reap"
            )
        if decision.action == "stale-boot":
            # Money first (item 9 / recovery F1.2): a bound custodian from
            # another boot is proven dead, so its reservation settles with the
            # closed ``failed-custody-reaped`` classification before the
            # binding is released, exactly as the mid-leg reaper orders it.
            self._settle_provably_dead_lease(authority, lease)
            authority.release_reconciled_custody(lease, outcome="stale-boot")
            return
        if decision.action == "already-dead":
            # A clean session whose wrapper never said anything about itself:
            # both oracles are free, so the machine has the evidence and no
            # operator drain is needed.  An unbound lease has no binding row a
            # release could reference; its money is settled by the ladder below.
            if lease.is_bound:
                self._settle_provably_dead_lease(authority, lease)
                authority.release_reconciled_custody(lease, outcome="verified-exit")
            return
        drain_hint = (
            "; after resolving the orphan, run python -m "
            "menagerie.crawler.tools.paid_session_custody --drain "
            f"{lease.lease_id} --confirm-resolved {lease.lease_id} --reason ..."
            if decision.action == "unreapable-refuse"
            else ""
        )
        raise DriverIntegrationError(
            f"paid lease {lease.lease_id} blocks admission: {decision.detail}{drain_hint}"
        )

    def _settle_provably_dead_lease(
        self, authority: PaidSessionAuthority, lease: PaidCustodyLease
    ) -> None:
        """Settle one bound, still-unsettled lease whose custodian is proven dead.

        Item 9 (recovery F1.2). Before this, the outcome of a killed leg
        depended on the OPERATOR's restart timing: a bound-dead lease still
        inside ``deadline_at + CUSTODY_REAPER_GRACE_SECONDS`` fell through the
        mid-leg reaper's deadline gate to the generic provider-key ladder and
        settled ``local_outcome="unknown"`` at attempt weight 1 (pilot: 11
        rows), while waiting ~35 minutes refunded it. The deadline was only
        ever the reaper's PROBE SCHEDULE; the safety property is the closed
        custody ladder's proof of death (``already-dead``: flock FREE and
        group EMPTY; ``stale-boot``: another boot). Settlement inputs stay
        replay-derived with zero producer input -- the provider answers for
        the idempotency key, the machine wall floor is the durable reservation
        age, and the closed classification is re-weighed by the fold -- so
        sessions and wall stay charged and only the ATTEMPT projection refunds.

        A never-bound lease has no custodian and stays the generic ladder's
        job; an already-settled lease (settled, then killed before its
        release) has nothing left to settle. A provider-adapter refusal
        propagates typed exactly as the generic ladder's does.
        """

        if not lease.is_bound or lease.spend_id not in authority.outstanding_spend_ids():
            return
        reconciler = self.dependencies.provider_reconciler
        if reconciler is None:
            # The generic ladder below raises its own typed refusal for the
            # missing adapter; settling nothing here keeps that one message.
            return
        try:
            authority.settle_reaped(
                lease.spend_id,
                query=reconciler,
                local_outcome="failed-custody-reaped",
            )
        except PaidSessionError as exc:
            raise DriverIntegrationError(
                f"paid-session provider reconciliation is unavailable: {exc}"
            ) from exc

    def _refuse_paid_admission_under_sprint_pause(
        self, operational: JsonlLedger, admission_boundary: str
    ) -> None:
        """Refuse one paid admission while the HOLD sentinel stands (D-62).

        The durable sprint-pause sentinel is re-read at EVERY paid admission
        (author and checker alike), so a hold the orchestrator drops mid-leg
        stops the next buy rather than the next leg. The refusal is the
        graceful-shutdown shape -- a typed, resumable interruption with no
        model-failure authority -- preceded by a durable disclosure row naming
        the hold, so the interruption's cause is never ambiguous in records.
        Ambiguity in the sentinel itself fails closed.
        """

        from menagerie.crawler.pause_authority import evaluate_sprint_pause

        decision = evaluate_sprint_pause(
            canonical_records_root(self.paths.ledgers.models),
            now=self.dependencies.clock(),
        )
        if not decision.held:
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
                "disposition": "sprint-pause-hold",
                "admission_boundary": admission_boundary,
                "decision": decision.decision,
                "detail": decision.detail,
            },
        }
        operational.append(mint_operational_event(event))
        raise DriverShutdown(
            ShutdownInterruptionFact(
                invocation_id=self.config.run_id,
                admission_boundary=admission_boundary,
                stable_id=None,
                work_id=None,
                execution_identity=None,
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

    def _refuse_paid_admission_under_leg_purpose(self, admission_boundary: str) -> None:
        """Refuse every paid buy typed under a capability-forfeiting purpose.

        Shape 4 (Sol contract point 4): a ``licensing_close`` leg forfeits ALL
        paid spend. The forfeit is structurally enforced upstream (the WS-1
        plan seals empty and the backlog sweep is skipped), so this chokepoint
        is the belt: ANY path that still reaches a reserve under the purpose
        -- present or future -- refuses here, before a money row exists, and
        the close's ledger-derived re-check
        (:func:`menagerie.crawler.leg_purpose._paid_spend_refusal`, pinned
        against a REAL reservation row) catches a bypass of even this.
        """

        from menagerie.crawler.leg_purpose import (
            LEG_PURPOSE_LICENSING_CLOSE,
            LegPurposeError,
        )

        declared = {
            getattr(self.config, "leg_purpose", None),
            getattr(getattr(self, "_leg_generation", None), "leg_purpose", None),
        }
        if LEG_PURPOSE_LICENSING_CLOSE in declared:
            raise LegPurposeError(
                f"{admission_boundary} refuses under leg purpose "
                f"{LEG_PURPOSE_LICENSING_CLOSE!r}: the purpose forfeits all "
                "paid admission"
            )

    def _admit_checker_spend(
        self,
        operational: JsonlLedger,
        scope: _CheckerSpendScope,
    ) -> Reservation:
        """Authorize and durably reserve one checker session before dispatch."""

        # Shape 4: a capability-forfeiting purpose refuses every buy typed.
        self._refuse_paid_admission_under_leg_purpose("checker-admission")
        # D-62: the HOLD sentinel is consulted before every checker buy (this
        # lane historically had no admission-time authority re-consult at all).
        self._refuse_paid_admission_under_sprint_pause(operational, "checker-admission")
        # W32: the declared checker CLI pin is verified (cached per leg)
        # BEFORE any money row is minted -- the pin of the SEAT that is about
        # to run (codex for the openai seat, the Claude CLI for the Fable
        # primary); a scope with no declared identity keeps the codex lane.
        from menagerie.crawler.toolchain_pins import checker_toolchain_lane

        seat_identity = scope.checker_identity
        self._require_verified_toolchain(
            checker_toolchain_lane(seat_identity.provider) if seat_identity is not None else "checker"
        )
        # The custody deadline reaper runs at every writer-owned admission
        # boundary, so a mid-leg bound-dead lease settles at the very next buy
        # instead of waiting for the next leg start (leg 12, 2026-08-21).
        self._reap_expired_dead_custody()
        identity = scope.checker_identity
        identity_fields: Optional[JsonObject] = (
            {
                "provider": identity.provider,
                "model": identity.model,
                "version": identity.version,
                "prompt_sha256": identity.prompt_sha256,
            }
            if identity is not None
            else None
        )
        # The durable custody deadline must sit at or above the lane's own
        # external kill, so a lease past ``deadline_at`` plus the reaper grace
        # has provably outlived every layer of the kill ladder.
        reservation_wall = (
            checker_session_kill_seconds()
            + CHECKER_LANE_BACKOFF_SLACK_SECONDS
            + CHECKER_LANE_WALL_GRACE_SECONDS
        )
        try:
            return self._paid_session_authority().reserve(
                scope={
                    "campaign_id": bound_campaign_id(self.config),
                    "generation_id": self._paid_generation_id(),
                    "lane": "checker",
                    "dispatch_kind": scope.gate_kind,
                    "work_generations": scope.work_generations,
                    "gate_kind": scope.gate_kind,
                    "checker_identity": identity_fields,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                },
                policy=BudgetPolicy(
                    quota_sessions=2**31 - 1,
                    quota_wall_seconds=CHECKER_MODEL_WALL_BUDGET_SECONDS,
                    runaway_sessions=CHECKER_LEG_SESSION_BUDGET,
                    runaway_wall_seconds=CHECKER_LEG_WALL_BUDGET_SECONDS,
                    reservation_wall_seconds=reservation_wall,
                ),
            )
        except PaidSessionBudgetExceeded as exc:
            if exc.instrument == "RUNAWAY_BREAKER":
                raise CheckerLegBudgetExceeded(str(exc)) from exc
            raise CheckerWallBudgetExceeded(
                str(exc),
                stable_ids=tuple(
                    stable_id for stable_id, _work_id in exc.work_generations
                ),
            ) from exc

    def _admit_author_spend(
        self,
        item: WorkItem,
        operational: JsonlLedger,
        attempt_group: Optional[str] = None,
    ) -> Reservation:
        """Authorize and durably reserve one author guard session before dispatch.

        ``attempt_group=None`` heads a NEW attempt and is bounded by the
        durable attempt allowance below; a spend id joins that attempt's group
        as a later guard session (source-request, then author) and buys no new
        attempt.  The group identity is stamped on the reservation row, so the
        attempt count is replay-derived from ledger rows -- never a mutable
        counter, and never priced in guard-session units (PSLRETRY round-2 F1:
        a session-unit quota of ``2*A`` let one work generation buy up to
        ``2*A`` source-only failed attempts).
        """

        # W1 (PSLRETRY round 5): re-consult the shutdown authority at the
        # writer's own reserve site. The retry ladder checks shutdown only at
        # attempt start, so a mid-attempt guard admission (the join bought by
        # ``_PaidDispatchSessions.begin_guard_session``) could land a fresh
        # paid reservation after shutdown was requested. Checking here -- head
        # and join alike -- closes the race between a worker's earlier check
        # and the actual fold-first append; the finished guard's settlement
        # stands and the typed refusal mints nothing.
        self._check_shutdown("author-admission", item=item)
        # Shape 4: a capability-forfeiting purpose refuses every buy typed.
        self._refuse_paid_admission_under_leg_purpose("author-admission")
        # D-62: the HOLD sentinel is consulted beside the shutdown authority
        # at the same writer-owned reserve site.
        self._refuse_paid_admission_under_sprint_pause(operational, "author-admission")
        # W32: the declared author CLI pin is verified (cached per leg)
        # BEFORE any money row is minted — a drifted or missing pinned binary
        # refuses the buy typed instead of burning a paid session on it.
        self._require_verified_toolchain("author")
        # W26/TE-23: the correlated cross-model storm fuse refuses the buy
        # BEFORE any money row is minted. K distinct models settling paid
        # author sessions with the SAME closed machine classification inside
        # one window is org-wide lane weather: the halt is typed, retryable,
        # campaign-scoped, mints zero terminals (the author loop re-raises
        # RetryableOperatorError ahead of its model-scoped blanket), and
        # self-expires as the window slides.
        storm = self._cross_model_storm_fuse().tripped(time.monotonic())
        if storm is not None:
            # W1/W26 composition: the halt strands THIS member with no durable
            # per-member row of its own. Mint its generation-scoped deferral
            # valve row BEFORE re-raising, so the typed campaign halt leaves
            # exactly the stranded members deferral-accounted -- zero terminals.
            self._mint_quota_storm_deferral(
                item, evidence={"witness": "cross-model-storm-fuse"}
            )
            raise CrossModelProviderStorm(
                "cross-model provider storm: paid author admission refused -- "
                + storm.describe()
            )
        # W1/PD-3: a generation past its storm-deferral bound holds for a human;
        # the buy refuses BEFORE any money row, and the provider-storm dimension
        # routes the refusal to the valve arm (which replays the park row), so
        # the hold stays a withholding, never a terminal.
        from menagerie.crawler.quota_deferral import (
            load_quota_deferral_rows,
            storm_recidivist_hold,
        )

        if storm_recidivist_hold(
            load_quota_deferral_rows(self._quota_deferral_ledger_path()),
            item.stable_id,
            item.active_work_id,
        ):
            raise AuthorEffortCapExceeded(
                f"storm-recidivist hold: {item.stable_id} generation "
                f"{item.active_work_id} has spent its PD-3 machine-deferral bound",
                stage="author",
                dimension="provider-storm",
            )
        # The custody deadline reaper runs at every writer-owned admission
        # boundary, so a mid-leg bound-dead lease settles at the very next buy
        # instead of waiting for the next leg start (leg 8, 2026-08-21).
        self._reap_expired_dead_custody()
        # Quiescent-cutover compatibility: historical artifact/operational rows
        # remain a fail-closed floor.  Every post-cutover buy is authorized by
        # the shared paid-session ledger below.
        enforce_author_session_allowance(
            (
                *getattr(self, "_author_spend_artifact_events", ()),
                *operational.records,
            ),
            item,
            operational_events=operational.records,
        )
        authority = self._paid_session_authority()
        campaign_id = bound_campaign_id(self.config)
        allowance = author_session_allowance(item.granted_attempts)
        runaway_sessions, runaway_wall = self._author_leg_runaway_budget()
        if attempt_group is None:
            # The durable ATTEMPT backstop, source-only failures included: a
            # new attempt group is admitted only while distinct chargeable
            # groups for this exact work generation stay under the allowance.
            consumed = authority.state().quota_attempt_groups.get(
                (campaign_id, item.stable_id, item.active_work_id, "author"), 0
            )
            if consumed >= allowance:
                raise AuthorEffortCapExceeded(
                    f"work generation {item.active_work_id} for {item.stable_id} "
                    f"already bought {consumed} author attempts of its "
                    f"{allowance}-attempt allowance",
                    stage="author",
                    dimension="author-sessions",
                )
        try:
            return authority.reserve(
                scope={
                    "campaign_id": campaign_id,
                    "generation_id": self._paid_generation_id(),
                    "lane": "author",
                    "dispatch_kind": "author_source",
                    "attempt_group": attempt_group,
                    "work_generations": ((item.stable_id, item.active_work_id),),
                    "gate_kind": None,
                    "checker_identity": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                },
                policy=BudgetPolicy(
                    # One attempt runs UP TO two paid guard sessions (source-request
                    # plus author), each its own reservation. The ATTEMPT bound is
                    # the attempt-group allowance above (plus the artifact-ledger
                    # floor); this paid QUOTA ceiling stays a true raw-session
                    # bound at the per-attempt factor.
                    quota_sessions=AUTHOR_PAID_SESSIONS_PER_ATTEMPT
                    * author_session_allowance(item.granted_attempts),
                    quota_wall_seconds=10**15,
                    # The leg fuse is DERIVED from the leg's own theoretical
                    # maximum spend (2026-08-20: the checker-lane constants
                    # tripped a healthy-shaped author leg at ~2.5 clock-hours
                    # of concurrency-12 authoring).
                    runaway_sessions=runaway_sessions,
                    runaway_wall_seconds=runaway_wall,
                    reservation_wall_seconds=float(OPERATOR_DEADLINE_SECONDS),
                ),
            )
        except PaidSessionBudgetExceeded as exc:
            if exc.instrument == "RUNAWAY_BREAKER":
                raise CheckerLegBudgetExceeded(str(exc)) from exc
            # W1: the raw-session QUOTA trip runs the SAME closed settlement
            # classifier the durable backstop runs, so a generation whose
            # allowance was burned by machine-classified infrastructure raises
            # the restorable provider-storm dimension and the storm valve --
            # never a permanent exhaustion terminal (R7: 28 members in one
            # storm leg, 21 used<max contradiction heads).
            raise AuthorEffortCapExceeded(
                str(exc),
                stage="author",
                dimension=author_sessions_exhaustion_dimension(
                    (
                        *getattr(self, "_author_spend_artifact_events", ()),
                        *operational.records,
                    ),
                    item.stable_id,
                    item.active_work_id,
                ),
            ) from exc

    def _require_verified_toolchain(self, lane: str) -> None:
        """W32: cached per-leg CLI pin verification ahead of the paid reserve.

        An undeclared pin is a disclosed no-op here (the launch preflight's
        cli-versions arm owns refusing pinless PRODUCTION launches); a
        declared pin that is non-absolute, missing, or off-version raises the
        typed :class:`~menagerie.crawler.toolchain_pins.ToolchainPinError`
        before any money row is minted.
        """

        from menagerie.crawler.toolchain_pins import require_verified_toolchain

        # The cache key only scopes the per-leg memo; minimal admission test
        # rigs exercise this mixin without a full driver config, so the key
        # degrades to process scope rather than assuming the attributes.
        config = getattr(self, "config", None)
        require_verified_toolchain(
            lane,
            cache_key=(
                bound_campaign_id(config) if config is not None else None,
                self._paid_generation_id() if hasattr(self, "config") else None,
            ),
        )

    def _cross_model_storm_fuse(self) -> CrossModelStormFuse:
        """Return this driver's windowed cross-model storm fuse (W26/TE-23).

        W32 rider (F-O23): the fuse parameters come from the campaign's
        DERIVED per-tier fuse table, so a model-tier change re-derives the
        lane-weather window with the tier's wall economics.
        """

        fuse = getattr(self, "_cross_model_storm_fuse_state", None)
        if fuse is None:
            from menagerie.crawler.storm_fuse import derive_fuse_table

            config = getattr(self, "config", None)
            table = derive_fuse_table(
                bound_campaign_id(config) if config is not None else None
            )
            fuse = CrossModelStormFuse(
                distinct_models=table.distinct_models,
                window_seconds=table.window_seconds,
            )
            self._cross_model_storm_fuse_state = fuse
        return fuse

    def _quota_deferral_ledger_path(self) -> Path:
        """Return the QUOTA-storm deferral valve ledger under this records root."""

        from menagerie.crawler.quota_deferral import quota_deferral_path

        return quota_deferral_path(self.paths.ledgers.models.parent.parent)

    def _mint_quota_storm_deferral(
        self, item: WorkItem, evidence: "Mapping[str, Any]"
    ) -> JsonObject:
        """Mint (or replay) one storm deferral valve row and stash it (W1).

        The valve chokepoint's driver half: the durable row is the member's
        accounting for this run's completion partition -- withheld, never
        terminal. A pair past its PD-3 bound comes back as the recidivist park
        row; both dispositions stash identically and both are members of the
        partition's closed machine-deferral vocabulary.
        """

        from menagerie.crawler.quota_deferral import mint_quota_deferral

        row = mint_quota_deferral(
            self._quota_deferral_ledger_path(),
            stable_id=item.stable_id,
            work_id=item.active_work_id,
            campaign_id=bound_campaign_id(self.config),
            run_id=self.config.run_id,
            machine_id=self.config.machine_id,
            created_at=self.dependencies.clock(),
            evidence=dict(evidence),
        )
        _stash_machine_deferral_row(self, item.stable_id, row)
        return row

    def _terminalize_effort_exhaustion(
        self,
        item: WorkItem,
        artifact: Optional[AuthorArtifact],
        exc: Exception,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> None:
        """Record one effort-exhaustion outcome: storm valve XOR evented terminal.

        The ONE home for both W1 invariants, called by every effort-exhaustion
        arm so no future arm can drift:

        * A ``provider-storm``-classified exhaustion (the closed settlement
          classifier's verdict, carried on the exception's ``dimension``) mints
          the transient deferral valve row -- withheld, PD-3 bounded, ZERO
          terminals -- with its failing attempt still appended so the money and
          effort record stays whole.
        * Every OTHER exhaustion mints its terminal WITH the ``model-lane-failed``
          operational event emitted first (PD-13: five 200a sites terminalized
          event-lessly, so 8 exhaustion heads had no durable cause row). The
          event precedes the terminal so the cause survives a bookkeeping
          refusal.
        """

        stage, reason = self._repair_failure_stage_and_reason(exc)
        attempt = _driver_failure_attempt(
            item,
            artifact,
            stage,
            reason,
            exc,
            self.config,
            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
            environment=None,
            created_at=self.dependencies.clock(),
        )
        persisted_attempt = reducer.append_attempt(attempt).record
        if getattr(exc, "dimension", None) == "provider-storm":
            self._mint_quota_storm_deferral(
                item,
                evidence={
                    "witness": "author-sessions-exhaustion-classifier",
                    "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                },
            )
            return
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
            (persisted_attempt,),
            reducer,
            operational,
            state,
            human_review=False,
        )

    def _author_leg_runaway_budget(self) -> "tuple[int, float]":
        """Return the cached author-leg runaway fuse for this leg's intake.

        Derived once per driver from the frozen intake snapshot's member count
        and the campaign's wall grant (:func:`author_leg_runaway_budget`); the
        checker-lane constants stay the floor for tiny legs.
        """

        cached = getattr(self, "_author_leg_runaway_cache", None)
        if cached is not None:
            return cached
        try:
            snapshot = load_intake_snapshot(self.paths.intake_root)
            member_count = len(snapshot.items)
        except Exception:  # noqa: BLE001 -- fuse sizing must never block admission
            # No loadable frozen intake (minimal test rigs; a production driver
            # cannot run without one). The CONSERVATIVE direction for a fuse is
            # the small floor: it trips earlier, never later.
            member_count = 0
        wall_grant = resolve_author_wall_seconds(
            bound_campaign_id(self.config),
            getattr(self.config, "author_wall_seconds", None),
        )
        budget = author_leg_runaway_budget(member_count, wall_grant)
        self._author_leg_runaway_cache = budget
        return budget

    def _settle_paid_session(
        self,
        reservation: Reservation,
        *,
        elapsed_seconds: float,
        completed: bool,
        local_outcome: Optional[str] = None,
        storm_model: Optional[str] = None,
    ) -> None:
        """Settle one outcome from provider attestation over a machine wall floor.

        The SESSION count is provider-attested in both directions: a
        provider-consummated session charges one even when this host failed
        locally, and a provider-declined one charges zero.  An exact
        machine-derived ``local_outcome`` may refund only the separate ATTEMPT
        projection; no local outcome changes the provider session or wall charge.

        The WALL charge keeps this driver's own ``time.monotonic`` measurement as
        a FLOOR (``elapsed_seconds``), because a runaway is a LOCAL phenomenon
        and must not be maskable by a remote answer: a provider that meters
        nothing, under-reports, or declines outright cannot make the seconds this
        host observably spent disappear from the cumulative instrument.

        ``storm_model`` (author-lane call sites only) additionally feeds the
        settled machine classification to the cross-model storm fuse
        (W26/TE-23) AFTER the durable settle: observation is read-only over an
        already-written fact and never gates or alters the settlement itself.
        """

        reconciler = self.dependencies.provider_reconciler
        if reconciler is None:
            raise DriverIntegrationError(
                "paid-session settlement requires provider idempotency-key reconciliation"
            )
        resolved_outcome = local_outcome or ("completed" if completed else "failed")
        self._paid_session_authority().settle(
            reservation,
            provider_query=reconciler,
            custody_outcome="exited" if completed else "exited-error",
            machine_wall_seconds=max(0.0, float(elapsed_seconds)),
            local_outcome=resolved_outcome,
        )
        if storm_model is not None:
            self._cross_model_storm_fuse().observe(
                stable_id=storm_model,
                lane="author",
                local_outcome=resolved_outcome,
                at=time.monotonic(),
            )

    def _ensure_gates(
        self,
        work: Sequence[WorkItem],
        artifacts: dict[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> Optional[str]:
        """Run metadata batches and required per-model fidelity gates durably.

        Checker *sessions* run concurrently, up to ``config.checker_concurrency``,
        through one round-scoped :class:`_CheckerSessionPool`. Everything else --
        envelope identity, gate normalization, routing, every ledger append, and
        every park or terminal decision -- stays on this single writer, in ladder
        order, exactly as it was when the lane was serial.
        """

        self._require_checker_leg_readiness(operational)
        pool = _CheckerSessionPool(
            max(1, int(self.config.checker_concurrency)),
            lambda scope: self._admit_checker_spend(operational, scope),
            lambda reservation, elapsed, completed: self._settle_paid_session(
                reservation,
                elapsed_seconds=elapsed,
                completed=completed,
            ),
            monotonic=self.dependencies.monotonic,
        )
        try:
            return self._run_gate_rounds(work, artifacts, reducer, operational, state, pool)
        finally:
            # A started checker session has already spent provider budget, so an
            # aborting round (a pause, a loud storm, the parks tripwire) drains
            # its siblings instead of orphaning them. Drained results are not
            # committed; their bytes stay under the batch work directories.
            pool.drain()
            pool.close()

    def _checker_spend_scope(
        self,
        gate_kind: str,
        artifacts: Sequence[AuthorArtifact],
        *,
        role: CheckerSeatRole = CheckerSeatRole.PRIMARY,
    ) -> _CheckerSpendScope:
        """Bind one paid checker session to durable work and checker identity.

        The money row binds the identity of the seat that RUNS: a K=2
        confirmation on the sol seat is charged under sol's identity, a primary
        under Fable's, so custody and spend accounting see both seats exactly.
        """

        generations = tuple(
            (
                artifact.author_result.binding.stable_id,
                artifact.author_result.binding.work_id,
            )
            for artifact in artifacts
        )
        declare = (
            lane_adjudication_identity
            if role is CheckerSeatRole.PRIMARY
            else lane_confirmation_identity
        )
        return _CheckerSpendScope(
            gate_kind=gate_kind,
            work_generations=generations,
            checker_identity=declare(self.dependencies.checker, gate_kind, self.config),
        )

    def _require_checker_leg_readiness(self, operational: JsonlLedger) -> None:
        """Run the once-per-leg checker readiness gates before any gate spend.

        Two BL-10 gates, both fail-closed and both BEFORE any session or author
        repair is bought:

        * **(e) prompt pin.** The leg's checker prompt identity is pinned (or
          re-verified against the pin) and attested durably on the operational
          ledger (``checker-prompt-pinned``), so every later seam -- envelope
          build, adjudication identity, catch-up currency -- consults the PIN
          and a mid-leg working-tree edit refuses typed at the next dispatch.
        * **(a)+(i) calibration.** When the leg's checker lane buys REAL
          provider sessions (the command lane), every gate kind must hold a
          measured calibration receipt for its pinned model, corroborated by
          re-recorded goldens, whose capacity plan fits the declared budgets
          with margin. No receipt, no sessions -- the sole checker cannot meet
          its first full rung uncalibrated.
        """

        if getattr(self, "_checker_leg_readiness_done", False):
            return
        pinned = leg_checker_prompt_sha256()
        if isinstance(self.dependencies.checker, CommandCheckerLane):
            from menagerie.crawler.operator_checker import required_checker_model

            for kind in GateKind:
                require_checker_calibration(kind.value, required_checker_model(kind))
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
                "disposition": "checker-prompt-pinned",
                "prompt_name": CHECKER_PROMPT_NAME,
                "checker_prompt_sha256": pinned,
            },
        }
        operational.append(mint_operational_event(event))
        self._checker_leg_readiness_done = True

    def _prefetch_metadata_batches(
        self,
        pool: _CheckerSessionPool,
        ladder: Sequence[tuple[AuthorArtifact, ...]],
        persisted: Sequence[Mapping[str, Any]],
        reuse_identity: Any,
    ) -> None:
        """Submit every ladder batch the serial path provably buys a session for.

        Exact eligibility, never speculation: a batch is submitted only when no
        reusable recorded adjudication exists for it (the same
        ``reusable_metadata_gate`` predicate the commit site consults). Within
        one round the ladder's item sets are disjoint, so a sibling's committed
        gate can never make a submitted session redundant.
        """

        if not pool.enabled:
            return
        for batch in ladder:
            if self._shutdown_event.is_set():
                return
            if reusable_metadata_gate(batch, persisted, checker=reuse_identity) is not None:
                continue
            # The pool submit IS this session's spend, so it sits behind the
            # same signalable admission ceremony as the serial dispatch
            # (pre-checker hook, then the shutdown door) -- otherwise a
            # SIGTERM delivered at the pre-checker boundary still buys and
            # spends the prefetched session (the checker-admission cell of
            # test_boundary_shutdown_composition). The writer's own admission
            # then classifies the interruption.
            self.dependencies.boundary_hook(
                "pre-checker", str(batch[0].proposal["stable_id"])
            )
            if self._shutdown_event.is_set():
                return
            # A visible CALL expression, deliberately: the anti-substitution
            # census pins this sensitive checker edge by AST call node under
            # THIS method, and a ``partial`` reference would blind it while a
            # nested ``def`` would re-own it (2026-08-20 review).
            dispatch_metadata = cast(
                "Callable[[], CheckerOutcome]",
                lambda batch=batch: self.dependencies.checker.check_metadata(
                    batch, self.paths.work_root, self.config
                ),
            )
            pool.submit(
                _metadata_batch_pool_key(batch),
                dispatch_metadata,
                self._checker_spend_scope("metadata_batch", batch),
            )

    def _prefetch_fidelity_sessions(
        self,
        pool: _CheckerSessionPool,
        work: Sequence[WorkItem],
        artifacts: Mapping[str, AuthorArtifact],
        persisted: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
    ) -> None:
        """Submit the first fidelity session for every model that certainly buys one.

        The predicate mirrors the fidelity loop's own path to its first dispatch
        and errs toward ``False``: a model with any rejected fidelity history
        repairs first (its artifact changes), an accepted or terminal history
        never dispatches, and a reusable recorded adjudication resolves without a
        session. Nothing the writer commits for a sibling can flip any of these
        inputs for this model, so every submitted session is consumed.
        """

        if not pool.enabled:
            return
        for item in work:
            if self._shutdown_event.is_set():
                return
            if recorded_head_blocks_work(self, reducer, item):
                continue
            if item.stable_id in self._checker_lane_parked_ids:
                continue
            artifact = artifacts[item.stable_id]
            if isinstance(artifact, ActivatedHandoffArtifact):
                continue
            if not _fidelity_required(artifact.proposal):
                continue
            history = _fidelity_gate_history(persisted, item.stable_id, proposal=artifact.proposal)
            if history and _fidelity_item_accepted(history[-1][1]):
                continue
            if (
                _terminal_fidelity_gate(
                    persisted, item.stable_id, _artifact_lineage(artifact), max_repairs=2
                )
                is not None
            ):
                continue
            if any(
                not _fidelity_item_accepted(gate_item)
                for _gate, gate_item in _fidelity_gate_history(
                    persisted,
                    item.stable_id,
                    campaign_root_work_id=_artifact_lineage(artifact),
                )
            ):
                continue
            if (
                reusable_fidelity_gate(
                    artifact,
                    persisted,
                    checker=lane_adjudication_identity(
                        self.dependencies.checker, "fidelity", self.config
                    ),
                )
                is not None
            ):
                continue
            # Same signalable admission ceremony as the serial dispatch --
            # the submit is the spend (see _prefetch_metadata_batches).
            self.dependencies.boundary_hook("pre-checker", item.stable_id)
            if self._shutdown_event.is_set():
                return
            # A visible CALL expression, deliberately: the anti-substitution
            # census pins this sensitive checker edge by AST call node under
            # THIS method, and a ``partial`` reference would blind it while a
            # nested ``def`` would re-own it (2026-08-20 review).
            dispatch_fidelity = cast(
                "Callable[[], CheckerOutcome]",
                lambda artifact=artifact: self.dependencies.checker.check_fidelity(
                    artifact, self.paths.work_root, self.config
                ),
            )
            pool.submit(
                _fidelity_pool_key(item.stable_id),
                dispatch_fidelity,
                self._checker_spend_scope("fidelity", (artifact,)),
            )

    def _prefetch_terminal_review_sessions(
        self,
        pool: _CheckerSessionPool,
        entries: Sequence[tuple[str, AuthorArtifact]],
        reducer: CanonicalReducer,
    ) -> None:
        """Submit the terminal-review session for every backlog head that buys one.

        DX-54: the terminal-review sweep was the last serial multi-minute
        checker lane in the leg (114 heads x ~146 s = 4.61 h in 200a). Exact
        eligibility, never speculation: a head is submitted only when no
        reusable recorded terminal adjudication exists for its rehydrated
        artifact -- the same ``reusable_terminal_gate`` predicate the commit
        site consults. Backlog heads are distinct stable IDs, so a sibling's
        committed gate can never make a submitted session redundant.
        """

        if not pool.enabled:
            return
        checker_identity = lane_adjudication_identity(
            self.dependencies.checker, "terminal_disposition", self.config
        )
        for stable_id, artifact in entries:
            if self._shutdown_event.is_set():
                return
            if (
                reusable_terminal_gate(
                    artifact,
                    reducer.gate_records,
                    checker=checker_identity,
                )
                is not None
            ):
                continue
            # Same signalable admission ceremony as the serial dispatch --
            # the submit is the spend (see _prefetch_metadata_batches).
            self.dependencies.boundary_hook("pre-checker", stable_id)
            if self._shutdown_event.is_set():
                return
            # A visible CALL expression, deliberately: the anti-substitution
            # census pins this sensitive checker edge by AST call node under
            # THIS method, and a ``partial`` reference would blind it while a
            # nested ``def`` would re-own it (2026-08-20 review).
            dispatch_terminal = cast(
                "Callable[[], CheckerOutcome]",
                lambda artifact=artifact: self.dependencies.checker.check_terminal(
                    artifact, self.paths.work_root, self.config
                ),
            )
            pool.submit(
                _terminal_pool_key(stable_id),
                dispatch_terminal,
                self._checker_spend_scope("terminal_disposition", (artifact,)),
            )

    def _require_confirmed_accuracy_acceptance(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        gate: JsonObject,
        gate_kind: str,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
    ) -> tuple[bool, Optional[str]]:
        """H-5: the metadata/fidelity duplicate-accept write door (rule-2, 2026-08-29).

        The calibration receipt's ``acceptance_witness`` block binds production
        acceptance depth K=2 UNCONDITIONALLY for every gate kind; the terminal
        lane has enforced it since AG-4 while metadata and fidelity acceptances
        stood on one session. This door makes the receipt true at rung time:
        an ACCEPTED accuracy gate item stands only over a settled ``confirmed``
        duplicate-accept row, or a distinct persisted same-question accept
        witness -- otherwise ONE more blind LIVE session over the same question
        is bought (never a pool capsule, never a reused verdict) and settled on
        the conflict ledger. Disagreement parks ``checker-verdict-conflict``
        (never a re-roll); an unfundable confirmation parks
        ``checker-accept-unconfirmed`` (the budget fuse: park unconfirmed,
        never publish unconfirmed).

        Capacity, disclosed (Opus §8.2 / the minted wall math): confirmation
        cost scales with the ACCEPT RATE, not the item count -- rejections are
        never duplicated. Measured metadata accept rate 0.077; the mint's
        capacity basis already prices K=2 confirmations at an assumed 0.25
        accept rate (3.2x measured, break-even 0.867), projecting 41.7h
        against the 56.0h margin ceiling -- 14.3h margin, no ceiling raise.
        The fuse is the existing reservation governor: the per-model checker
        wall budget refuses an unfundable confirmation PRE-DISPATCH
        (:class:`CheckerWallBudgetExceeded` -> the typed unfunded park), and
        the leg-scoped breaker stays a loud resumable campaign pause.

        Returns
        -------
        tuple[bool, str | None]
            ``(acceptance_stands, pause_reason)``. ``(False, None)`` means the
            item parked or stays unconfirmed (the door re-derives from the
            ledger on every leg, so a crash window resumes the CONFIRMATION,
            never a duplicate first session).
        """

        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = checker_verdict_conflict_path(records_root)
        gate_id = str(gate.get("gate_id"))
        settled = accuracy_duplicate_accept_state(
            gate_id, item.stable_id, load_checker_verdict_conflict_rows((ledger_path,))
        )
        if settled == "confirmed":
            return True, None
        if settled == "conflict":
            self._record_checker_lane_deferral(
                item,
                DriverIntegrationError(
                    f"duplicate-accept conflict already settled for {gate_kind} gate "
                    f"{gate_id!r}; the acceptance stays parked for review"
                ),
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind=gate_kind,
                failure_class="checker-verdict-conflict",
                count_in_streak=False,
            )
            return False, None
        try:
            if accuracy_acceptance_witnesses(gate, item.stable_id, reducer.gate_records):
                # Two blind consistent accepts already stand on the gates
                # ledger; a third session would be pure spend.
                return True, None
        except CheckerVerdictConflictError:
            pass
        try:
            # Seat swap: the blind confirmation is bought on the CONFIRMATION
            # seat (cross-model against the primary); a lane without a
            # confirmation seat confirms under its own identity, and the
            # conflict ledger's pairing rule decides which pairs are lawful.
            outcome = self._retry_infrastructure_call(
                lambda: buy_confirmation_session(
                    self.dependencies.checker,
                    gate_kind,
                    (artifact,),
                    self.paths.work_root,
                    self.config,
                ),
                admission=("checker", item),
                operational=operational,
                checker_scope=self._checker_spend_scope(
                    gate_kind, (artifact,), role=CheckerSeatRole.CONFIRMATION
                ),
            )
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError, CheckerPromptRotatedError):
            # CheckerLegBudgetExceeded is a RetryableOperatorError by design:
            # the leg-scoped breaker pauses the campaign resumably; the
            # acceptance stays unconfirmed and this door resumes it.
            raise
        except CheckerWallBudgetExceeded as exc:
            # The budget fuse: the confirmation cannot be FUNDED for this
            # model, so the item parks unconfirmed rather than publishing
            # unconfirmed -- the receipt's bound unfunded_confirmation_outcome.
            self._record_checker_lane_deferral(
                item,
                exc,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept-fuse",
                gate_kind=gate_kind,
                failure_class="checker-accept-unconfirmed",
                count_in_streak=False,
            )
            return False, None
        except CheckerMechanicalClaimError as exc:
            # W-8 x H-5: a confirmation session asserting machine-refuted
            # claims is checker instability, parked -- not left unconfirmed
            # for a silent resume re-buy of the same defect.
            self._record_checker_lane_deferral(
                item,
                exc,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="mechanical-refutation",
                gate_kind=gate_kind,
                failure_class="checker-verdict-conflict",
                count_in_streak=False,
            )
            return False, None
        except Exception as exc:  # noqa: BLE001 -- confirmation failure renders no settlement
            self._record_checker_lane_deferral(
                item,
                exc,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind=gate_kind,
            )
            return False, None
        if outcome.backoff is not None:
            return False, self._pause_for_usage(outcome.backoff, operational, 1)
        confirmation = outcome.gate
        if confirmation is None:
            self._record_checker_lane_deferral(
                item,
                DriverIntegrationError(
                    f"{gate_kind} duplicate-accept confirmation produced no durable gate"
                ),
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind=gate_kind,
            )
            return False, None
        try:
            # The confirmation is NEVER appended to the gates ledger (a second
            # accepted item row would violate exact-membership at publication);
            # its full verdict projection settles durably on the conflict
            # ledger, bound to the accepted row's gate_id + stable_id.
            row = build_accuracy_duplicate_accept_row(
                gate=gate,
                confirmation=confirmation,
                stable_id=item.stable_id,
                campaign_id=bound_campaign_id(self.config),
                run_id=self.config.run_id,
                machine_id=self.config.machine_id,
                created_at=self.dependencies.clock(),
                author_result_sha256=str(artifact.author_result.binding.result_sha256),
            )
        except CheckerVerdictConflictError as exc:
            # Divergent question, unreadable projection, or a half-identified
            # confirmation: the acceptance cannot settle, fail-closed.
            self._record_checker_lane_deferral(
                item,
                exc,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind=gate_kind,
            )
            return False, None
        persisted_row = append_checker_verdict_conflict_row(ledger_path, row)
        if persisted_row["status"] != "confirmed":
            self._record_checker_lane_deferral(
                item,
                DriverIntegrationError(
                    f"duplicate-accept disagreement: the blind confirmation session's "
                    f"full verdict projection diverged from the accepted {gate_kind} "
                    f"gate {gate_id!r} (measured bless-direction bad-pass floor 6.7%, "
                    "sol U-S1); the acceptance parks for human review"
                ),
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind=gate_kind,
                failure_class="checker-verdict-conflict",
                count_in_streak=False,
            )
            return False, None
        return True, None

    def _run_gate_rounds(
        self,
        work: Sequence[WorkItem],
        artifacts: dict[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        pool: _CheckerSessionPool,
    ) -> Optional[str]:
        """Run metadata batches and required per-model fidelity gates durably."""

        for item in work:
            if isinstance(artifacts[item.stable_id], ActivatedHandoffArtifact):
                continue
            artifacts[item.stable_id] = _require_legacy_audit_fidelity(
                item, artifacts[item.stable_id], self.config
            )
        persisted = list(reducer.gate_records)
        items_by_id = {item.stable_id: item for item in work}
        pending_ids = {
            item.stable_id
            for item in work
            if not isinstance(artifacts[item.stable_id], ActivatedHandoffArtifact)
            and not _metadata_gate_accepted(
                persisted, item.stable_id, artifacts[item.stable_id].proposal
            )
        }
        while pending_ids:
            for stable_id in tuple(sorted(pending_ids)):
                terminal_gate = _terminal_metadata_gate(
                    persisted,
                    stable_id,
                    _artifact_lineage(artifacts[stable_id]),
                    max_repairs=2,
                )
                if terminal_gate is None:
                    continue
                self._terminalize_accuracy_gate(
                    items_by_id[stable_id],
                    artifacts[stable_id],
                    terminal_gate,
                    reducer,
                    operational,
                    state,
                )
                pending_ids.remove(stable_id)
            if not pending_ids:
                break

            # BL-10 (b): the ABSOLUTE per-lineage round ceiling. The chargeable
            # cap above bounds author incorrigibility only; a checker that keeps
            # disclosing fresh complaints is never chargeable and would hold this
            # loop open forever (pilot m8650: 10 rounds, 9 rejections, 1 charged).
            # Past the ceiling the lineage is checker disclosure debt: it PARKS
            # on the checker-lane valve for human review -- withheld, accounted,
            # requeue-eligible -- and never mints a model-blaming ``failed:*``.
            # Checked BEFORE batching, so a lineage already at the ceiling never
            # buys another session.
            for stable_id in tuple(sorted(pending_ids)):
                ceiling_history = _metadata_gate_history(
                    persisted,
                    stable_id,
                    campaign_root_work_id=_artifact_lineage(artifacts[stable_id]),
                )
                if len(ceiling_history) < METADATA_GATE_ROUND_CEILING:
                    continue
                self._record_checker_lane_deferral(
                    items_by_id[stable_id],
                    DriverIntegrationError(
                        f"metadata gate round ceiling: {len(ceiling_history)} persisted "
                        f"rounds without acceptance (ceiling {METADATA_GATE_ROUND_CEILING}); "
                        "checker disclosure debt routed to human review"
                    ),
                    attempted_batch_sizes=tuple(
                        max(int(gate.get("batch_size", 1) or 1), 1)
                        for gate, _item in ceiling_history
                    ),
                    operational=operational,
                    failure_site="disclosure-debt-ceiling",
                    gate_kind="metadata_batch",
                    failure_class="checker-disclosure-debt",
                    count_in_streak=False,
                )
                pending_ids.discard(stable_id)
            if not pending_ids:
                break

            repair_counts = {
                stable_id: _metadata_repair_count(
                    persisted, stable_id, _artifact_lineage(artifacts[stable_id])
                )
                for stable_id in pending_ids
            }
            for stable_id, count in repair_counts.items():
                if count == 0 or not _metadata_gate_history(
                    persisted, stable_id, artifacts[stable_id].proposal
                ):
                    continue
                try:
                    artifacts[stable_id] = _require_legacy_audit_fidelity(
                        items_by_id[stable_id],
                        self._retry_infrastructure_call(
                            lambda: self._repair_author(
                                items_by_id[stable_id],
                                artifacts[stable_id],
                                persisted,
                                count,
                                reducer,
                                gate_kind="metadata_batch",
                                operational_events=operational.records,
                            ),
                            admission=("author", items_by_id[stable_id]),
                            operational=operational,
                        ),
                        self.config,
                    )
                except AuthorBackoffError as backoff:
                    # A provider pause during a bounded repair is a campaign pause with a
                    # reset time, not a failed repair. `_ensure_gates` already returns a
                    # usage-pause reason, so route it the same way as a checker backoff.
                    pool.cancel_pending()  # BL-8 (d): a pause never buys queued sessions
                    return self._pause_for_usage(backoff.signal, operational, len(work))
                except AuthorSessionRetryExhausted as exhausted:
                    # Rung 7's exact crash site: a repair-round session died without
                    # publishing until the bounded ladder spent. The lane-health
                    # boundary decides model-vs-campaign scope; the retained prior
                    # artifact keeps the frozen source facts on a model-scoped
                    # terminal instead of erasing work the broker already proved.
                    if self._scope_author_retry_exhaustion(
                        items_by_id[stable_id],
                        artifacts[stable_id],
                        exhausted,
                        reducer,
                        operational,
                        state,
                    ):
                        pending_ids.discard(stable_id)
                        continue
                    raise
                except (AuthorEffortCapExceeded, AuthorEffortExhaustionClaim) as exc:
                    # W1/PD-13: the shared arm events every exhaustion terminal
                    # and routes storm-classified exhaustion to the deferral
                    # valve (withheld, zero terminals).
                    self._terminalize_effort_exhaustion(
                        items_by_id[stable_id],
                        artifacts[stable_id],
                        exc,
                        reducer,
                        operational,
                        state,
                    )
                    pending_ids.discard(stable_id)
                    continue
                except RetryableOperatorError:
                    raise
                except AuthorRepairTerminal as terminal:
                    # A typed terminal recommendation is an ANSWER, not a repair
                    # failure. Caught ahead of the blanket arm below, which would
                    # record it as `failed:runner / protocol-violation` and throw
                    # the verdict away.
                    pause = self._route_repair_terminal_author_result(
                        items_by_id[stable_id], terminal, reducer, operational, state
                    )
                    pending_ids.discard(stable_id)
                    if pause is not None:
                        return pause
                    continue
                except Exception as exc:  # noqa: BLE001 -- repair failure is model-local
                    stage, reason = self._repair_failure_stage_and_reason(exc)
                    attempt = _driver_failure_attempt(
                        items_by_id[stable_id],
                        artifacts[stable_id],
                        stage,
                        reason,
                        exc,
                        self.config,
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                        environment=None,
                        created_at=self.dependencies.clock(),
                    )
                    if reason == "internal-error":
                        _stamp_failure_owner(
                            attempt,
                            "host" if self._is_infrastructure_error(exc) else "machinery",
                        )
                    persisted_attempt = reducer.append_attempt(attempt).record
                    persisted_error = persisted_attempt.get("error")
                    if (
                        isinstance(persisted_error, Mapping)
                        and _failure_owner(persisted_error) in NEVER_MODEL_TERMINAL_OWNERS
                    ):
                        # DX-1/RE-1 (m12446 shape): a host/machinery fault in the
                        # gate-round repair lane is infrastructure weather, never
                        # this model's verdict -- park loudly instead.
                        self._park_non_model_fault(
                            items_by_id[stable_id],
                            artifacts[stable_id],
                            persisted_error,
                            operational,
                        )
                        pending_ids.discard(stable_id)
                        continue
                    self._terminalize(
                        items_by_id[stable_id],
                        artifacts[stable_id],
                        f"failed:{stage}",
                        reason,
                        str(exc),
                        (persisted_attempt,),
                        reducer,
                        operational,
                        state,
                    )
                    pending_ids.discard(stable_id)

            if not pending_ids:
                break

            pending_artifacts = [artifacts[stable_id] for stable_id in sorted(pending_ids)]
            requeued: set[str] = set()
            # Durable progress is "did this round resolve or requeue anything", not
            # "did it requeue anything". Scoping a stale binding to its own item can
            # retire one model and leave its siblings pending, which is progress.
            pending_before_round = set(pending_ids)
            # W4-A failure route: a checker-lane BATCH failure is never a fact about
            # any member model (rung 10 terminalized 41 innocent models, one of them
            # runs-proven, from three such failures). A failed batch rides a halving
            # ladder (shared mechanism: ``downshift_metadata_items``, floor one); a
            # floor item whose gate still cannot be delivered PARKS on the durable
            # checker-lane deferral valve -- accounted by the terminal partition,
            # requeued on the next run -- instead of minting a fabricated terminal.
            #
            # Checker economics: members whose exact adjudication is already on the
            # ledger (full identity match, see ``gate_reuse``) are extracted into
            # their own ladder entries BEFORE live batching, so replayed verdicts
            # never ride -- or pad -- a bought session, and live batches pack only
            # work that needs one.
            reuse_identity = lane_adjudication_identity(
                self.dependencies.checker, "metadata_batch", self.config
            )
            reuse_groups, live_artifacts = plan_metadata_gate_reuse(
                pending_artifacts, persisted, checker=reuse_identity
            )
            ladder: deque[tuple[AuthorArtifact, ...]] = deque(
                (
                    *reuse_groups,
                    *(tuple(batch) for batch in _metadata_batches(live_artifacts)),
                )
            )
            self._prefetch_metadata_batches(pool, ladder, persisted, reuse_identity)
            ladder_failed_sizes: dict[str, list[int]] = {}
            while ladder:
                batch = ladder.popleft()
                batch_ids = tuple(str(artifact.proposal["stable_id"]) for artifact in batch)
                reused_gate = reusable_metadata_gate(batch, persisted, checker=reuse_identity)
                if reused_gate is not None:
                    outcome = CheckerOutcome(gate=reused_gate)
                    self._record_checker_reuse(
                        operational,
                        gate_kind="metadata_batch",
                        stable_ids=batch_ids,
                        reused_from=reused_gate[GATE_REUSED_FROM_FIELD],
                    )
                else:
                    try:
                        # The closure is invoked before the next ladder entry rebinds
                        # ``batch``, exactly like the historical per-batch lambda.
                        # A pool-prefetched session is spent as attempt one; the
                        # bounded retry ladder and its weather parks then run
                        # unchanged on this writer thread.
                        capsule = pool.claim(_metadata_batch_pool_key(batch))
                        checker_scope = self._checker_spend_scope("metadata_batch", batch)
                        outcome = self._retry_infrastructure_call(
                            _replay_then_live(
                                capsule,
                                lambda: self.dependencies.checker.check_metadata(
                                    batch, self.paths.work_root, self.config
                                ),
                            ),
                            admission=("checker", items_by_id[batch_ids[0]]),
                            operational=operational,
                            weather_queued_models=len(batch_ids),
                            checker_scope=checker_scope,
                            prepaid_checker_reservation=(
                                capsule.reservation if capsule is not None else None
                            ),
                        )
                    except (DriverPaused, RetryableOperatorError, AuthorBackoffError, CheckerPromptRotatedError):
                        # Genuinely campaign-scoped checker weather -- outage, transport,
                        # timeout, provider backoff -- stays LOUD; the ladder never
                        # absorbs it into a model-shaped outcome.
                        raise
                    except CheckerMechanicalClaimError as exc:
                        # W-8: a machine-refuted checker claim (a fabricated
                        # locator, hash, citation, or search-attestation
                        # allegation) is checker instability, operator
                        # property. The named items park as
                        # checker-verdict-conflict -- never a member verdict
                        # in either direction, never a silent re-roll, never a
                        # model-blaming terminal (the one measured instance
                        # mass-rejected a clean pack) -- and the innocent
                        # siblings re-batch as their own ladder entries
                        # without a downshift tax.
                        named = set(exc.stable_ids) & set(batch_ids) or set(batch_ids)
                        for parked_id in sorted(named):
                            self._record_checker_lane_deferral(
                                items_by_id[parked_id],
                                exc,
                                attempted_batch_sizes=(len(batch_ids),),
                                operational=operational,
                                failure_site="mechanical-refutation",
                                gate_kind="metadata_batch",
                                failure_class="checker-verdict-conflict",
                                count_in_streak=False,
                            )
                            pending_ids.discard(parked_id)
                        for sibling in batch:
                            if str(sibling.proposal["stable_id"]) not in named:
                                ladder.append((sibling,))
                        self._prefetch_metadata_batches(
                            pool, ladder, persisted, reuse_identity
                        )
                        continue
                    except CheckerItemContractError as exc:
                        # BL-10 (c): the wrapper named the exact items whose
                        # contract failed. Scope the failure to those models --
                        # the innocent siblings re-batch together without a
                        # downshift tax, and each named model earns one bounded
                        # singleton re-adjudication before parking truthfully.
                        self._scope_item_contract_failure(
                            batch,
                            exc,
                            ladder=ladder,
                            failed_sizes=ladder_failed_sizes,
                            items_by_id=items_by_id,
                            pending_ids=pending_ids,
                            operational=operational,
                            artifacts=artifacts,
                            reducer=reducer,
                            state=state,
                        )
                        self._prefetch_metadata_batches(pool, ladder, persisted, reuse_identity)
                        continue
                    except Exception as exc:  # noqa: BLE001 -- lane failure, never a model fact
                        # ``CheckerPartialResultError`` (W4-B's typed partial/extra-items
                        # refusal) and every other lane-owned dispatch failure land here.
                        # A typed partial result is not an infrastructure error, so it
                        # reaches this arm on its FIRST occurrence and the downshift is
                        # spent before any same-size contract re-ask would be.
                        self._downshift_or_park_checker_batch(
                            batch,
                            exc,
                            ladder=ladder,
                            failed_sizes=ladder_failed_sizes,
                            items_by_id=items_by_id,
                            pending_ids=pending_ids,
                            operational=operational,
                            failure_site="checker-dispatch",
                            artifacts=artifacts,
                            reducer=reducer,
                            state=state,
                        )
                        # The ladder may have gained halves; they are live by
                        # construction, so overlap them like any other entry.
                        self._prefetch_metadata_batches(pool, ladder, persisted, reuse_identity)
                        continue
                if outcome.backoff is not None:
                    pool.cancel_pending()  # BL-8 (d): a pause never buys queued sessions
                    return self._pause_for_usage(outcome.backoff, operational, len(work))
                try:
                    raw_gate = _require_gate(outcome)
                    gate = _normalize_gate_generation(raw_gate, persisted, batch_ids)
                    _require_gate_bindings(gate, batch, "metadata_batch")
                    route_ready = _prepare_ledger_record(gate, len(persisted) + 1)
                    # The route charges CHARGEABLE prior rejections against the
                    # cap and judges the current adverse items against what the
                    # prior round actually disclosed (complaint-disclosure
                    # rule). The raw `_metadata_repair_count` keeps numbering
                    # repair generations elsewhere; it no longer bounds anyone.
                    counts = {
                        stable_id: _chargeable_metadata_repair_count(
                            persisted, stable_id, _artifact_lineage(artifacts[stable_id])
                        )
                        for stable_id in batch_ids
                    }
                    # BL-10 (b): the disclosure baseline is the UNION over every
                    # prior round, never the latest round alone -- one-round
                    # comparison made chargeability walkable by disjoint
                    # complaint alternation (audit B-1).
                    disclosed = {
                        stable_id: disclosed_metadata_complaint_union(
                            _metadata_gate_history(
                                persisted,
                                stable_id,
                                campaign_root_work_id=_artifact_lineage(artifacts[stable_id]),
                            )
                        )
                        for stable_id in batch_ids
                    }
                    decisions = route_metadata_gate(
                        route_ready,
                        counts,
                        max_repairs=2,
                        prior_complaint_fields=disclosed,
                    )
                    # `emit_gate_records` raises `GateRoutingError`, which had no handler
                    # anywhere between here and the CLI catch-all: one unroutable gate
                    # ended the campaign with zero records written. It belongs to the
                    # batch that produced it, like every other gate contract failure.
                    gate_records = emit_gate_records(route_ready)
                except StaleGateBindingError as exc:
                    # A stale binding is a fact about the named items only. Terminalize
                    # exactly those; the siblings stay pending and are re-batched by the
                    # enclosing loop, so one bad member no longer discards its batch.
                    for stable_id in exc.stale_ids:
                        item = items_by_id[stable_id]
                        attempt = _driver_failure_attempt(
                            item,
                            artifacts[stable_id],
                            "runner",
                            "protocol-violation",
                            exc,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=None,
                            created_at=self.dependencies.clock(),
                        )
                        persisted_attempt = reducer.append_attempt(attempt).record
                        self._terminalize(
                            item,
                            artifacts[stable_id],
                            "failed:runner",
                            "protocol-violation",
                            str(exc),
                            (persisted_attempt,),
                            reducer,
                            operational,
                            state,
                            human_review=False,
                        )
                        pending_ids.discard(stable_id)
                    continue
                except (DriverPaused, RetryableOperatorError, AuthorBackoffError, CheckerPromptRotatedError):
                    # Genuinely campaign-level: a pause, a retryable transport failure,
                    # or a provider backoff must never be recorded as a model defect.
                    raise
                except (AuthorEffortCapExceeded, AuthorEffortExhaustionClaim) as exc:
                    for stable_id in batch_ids:
                        # W1/PD-13: shared exhaustion arm (event XOR storm valve).
                        self._terminalize_effort_exhaustion(
                            items_by_id[stable_id],
                            artifacts[stable_id],
                            exc,
                            reducer,
                            operational,
                            state,
                        )
                        pending_ids.discard(stable_id)
                    continue
                except Exception as exc:  # noqa: BLE001 -- invalid checker output is lane-owned
                    # A delivered gate the routing validators refuse (missing items,
                    # unroutable records, contract-invalid payloads) is exactly as
                    # verdict-free for the members as a dispatch failure: a fresh
                    # smaller-batch response routinely routes cleanly, and a floor
                    # item that still cannot route parks rather than terminalizes.
                    # The per-ITEM conditions keep their own arms above: a stale
                    # binding and a typed effort answer are facts about named models.
                    self._downshift_or_park_checker_batch(
                        batch,
                        exc,
                        ladder=ladder,
                        failed_sizes=ladder_failed_sizes,
                        items_by_id=items_by_id,
                        pending_ids=pending_ids,
                        operational=operational,
                        failure_site="gate-routing",
                        artifacts=artifacts,
                        reducer=reducer,
                        state=state,
                    )
                    self._prefetch_metadata_batches(pool, ladder, persisted, reuse_identity)
                    continue
                for record in gate_records:
                    result = reducer.append_gate(_without_ledger_fields(record))
                    if result.appended:
                        persisted.append(result.record)
                    elif not any(
                        existing.get("gate_id") == result.record.get("gate_id")
                        for existing in persisted
                    ):
                        persisted.append(result.record)
                self._note_checker_gate_routed()
                for decision in decisions:
                    stable_id = decision.stable_id
                    if decision.canonical_write_allowed:
                        pending_ids.discard(stable_id)
                    elif decision.human_review_required:
                        latest = _find_gate(
                            persisted,
                            stable_id,
                            "metadata_batch",
                            artifacts[stable_id].proposal,
                        )
                        if latest is None:
                            raise DriverIntegrationError(
                                f"persisted metadata gate missing for {stable_id}"
                            )
                        self._terminalize_accuracy_gate(
                            items_by_id[stable_id],
                            artifacts[stable_id],
                            latest,
                            reducer,
                            operational,
                            state,
                        )
                        pending_ids.discard(stable_id)
                    else:
                        requeued.add(stable_id)
                    self.dependencies.boundary_hook("after-gate", stable_id)
            if pending_ids and not requeued and pending_ids == pending_before_round:
                # BL-8 (b): the no-progress tripwire quarantines the stuck
                # MEMBERS, it does not kill the leg. A round that neither
                # resolved nor requeued anything is provably wedged for
                # exactly these members; each gets a durable typed park
                # (partition-accounted, boundedly retried per W12: released
                # while retry budget remains, then operator-drained) and is
                # withheld from the fidelity lane below, so the rest of the
                # leg keeps its progress.
                for stable_id in sorted(pending_ids):
                    stuck = items_by_id[stable_id]
                    self._record_scheduler_quarantine(
                        stuck,
                        site="gate-round",
                        reason_code="no-durable-routing-progress",
                        fault=(
                            "metadata gate made no durable routing progress for "
                            "this member across a full round; member isolated so "
                            "the leg continues"
                        ),
                        operational=operational,
                    )
                    self._checker_lane_parked_ids.add(stable_id)
                pending_ids.clear()

        # H-5: the metadata duplicate-accept write door. Runs AFTER the gate
        # rounds and BEFORE the fidelity lane and any publication, over the
        # same publication-eligible gate `_authorize_terminal_artifact`
        # consumes -- so both the fresh accept minted this leg and an accepted
        # gate resumed from a crashed leg pass the SAME door, re-derived from
        # the ledger (the crash window between the accept and its confirmation
        # settlement resumes the CONFIRMATION, never a duplicate first
        # session).
        for item in work:
            artifact = artifacts[item.stable_id]
            if isinstance(artifact, ActivatedHandoffArtifact):
                continue
            if item.stable_id in self._checker_lane_parked_ids:
                continue
            if recorded_head_blocks_work(self, reducer, item):
                continue
            accepted_gate = _find_gate(
                persisted, item.stable_id, "metadata_batch", artifact.proposal
            )
            if accepted_gate is None:
                continue
            accepted_item = next(
                (
                    value
                    for value in accepted_gate.get("items", [])
                    if value.get("stable_id") == item.stable_id
                ),
                None,
            )
            if accepted_item is None or not accuracy_item_accepted(
                "metadata_batch", accepted_item
            ):
                continue
            _stands, confirm_pause = self._require_confirmed_accuracy_acceptance(
                item, artifact, accepted_gate, "metadata_batch", reducer, operational
            )
            if confirm_pause is not None:
                pool.cancel_pending()  # BL-8 (d): a pause never buys queued sessions
                return confirm_pause

        # Every artifact is final for the fidelity lane's FIRST dispatch (only a
        # model's own repair path mutates it, and repair-needing models are not
        # prefetched), so the certain first sessions can overlap now.
        self._prefetch_fidelity_sessions(pool, work, artifacts, persisted, reducer)
        for item in work:
            if recorded_head_blocks_work(self, reducer, item):
                continue
            if item.stable_id in self._checker_lane_parked_ids:
                # A parked model holds no accepted metadata gate, so the fidelity
                # lane has no adjudicated proposal to certify; it is withheld from
                # this run entirely and requeued when the checker lane recovers.
                continue
            artifact = artifacts[item.stable_id]
            if isinstance(artifact, ActivatedHandoffArtifact):
                continue
            if not _fidelity_required(artifact.proposal):
                continue
            while True:
                current_history = _fidelity_gate_history(
                    persisted,
                    item.stable_id,
                    proposal=artifact.proposal,
                )
                if current_history and _fidelity_item_accepted(current_history[-1][1]):
                    # H-5: the fidelity duplicate-accept write door -- the
                    # exact publication-eligible gate, same door as metadata.
                    fidelity_gate = _find_gate(
                        persisted, item.stable_id, "fidelity", artifact.proposal
                    )
                    if fidelity_gate is not None:
                        fidelity_item = next(
                            (
                                value
                                for value in fidelity_gate.get("items", [])
                                if value.get("stable_id") == item.stable_id
                            ),
                            None,
                        )
                        if fidelity_item is not None and accuracy_item_accepted(
                            "fidelity", fidelity_item
                        ):
                            _stands, confirm_pause = (
                                self._require_confirmed_accuracy_acceptance(
                                    item,
                                    artifact,
                                    fidelity_gate,
                                    "fidelity",
                                    reducer,
                                    operational,
                                )
                            )
                            if confirm_pause is not None:
                                pool.cancel_pending()  # BL-8 (d)
                                return confirm_pause
                    break
                # AG-6: the ABSOLUTE per-lineage round ceiling (the fidelity
                # twin of the metadata block above). The chargeable cap bounds
                # author incorrigibility only -- a checker disclosing fresh
                # complaints every round is never chargeable and would hold
                # this loop open until author exhaustion (200a m1359/m5445
                # ran 7 rounds). Checked BEFORE the cap terminal and BEFORE a
                # session is bought; past the ceiling the lineage parks as
                # checker disclosure debt, never a model-blaming terminal.
                ceiling_history, ceiling = gate_round_ceiling_state(
                    persisted,
                    gate_kind="fidelity",
                    stable_id=item.stable_id,
                    campaign_root_work_id=_artifact_lineage(artifact),
                )
                if len(ceiling_history) >= ceiling:
                    self._record_checker_lane_deferral(
                        item,
                        DriverIntegrationError(
                            f"fidelity gate round ceiling: {len(ceiling_history)} persisted "
                            f"rounds without acceptance (ceiling {ceiling}); "
                            "checker disclosure debt routed to human review"
                        ),
                        attempted_batch_sizes=tuple(
                            max(int(gate.get("batch_size", 1) or 1), 1)
                            for gate, _item in ceiling_history
                        ),
                        operational=operational,
                        failure_site="disclosure-debt-ceiling",
                        gate_kind="fidelity",
                        failure_class="checker-disclosure-debt",
                        count_in_streak=False,
                    )
                    break
                terminal_gate = _terminal_fidelity_gate(
                    persisted,
                    item.stable_id,
                    _artifact_lineage(artifact),
                    max_repairs=2,
                )
                if terminal_gate is not None:
                    terminal_item = next(
                        value
                        for value in terminal_gate["items"]
                        if value["stable_id"] == item.stable_id
                    )
                    verdict = str(terminal_item.get("fidelity", {}).get("verdict"))
                    # The shared kernel keeps this candidate coherent with the
                    # terminal proof the reducer re-derives: a content-accepted
                    # item rejected by a rung-inaccurate check is an identity
                    # dispute here too, never a cap claim the proof contradicts.
                    reason = fidelity_gate_failure_reason(terminal_item)
                    self._terminalize(
                        item,
                        artifact,
                        "failed:fidelity",
                        reason,
                        f"fidelity gate blocked after bounded repair: {verdict}",
                        (),
                        reducer,
                        operational,
                        state,
                        human_review=True,
                        root_cause_fingerprint=_gate_item_fingerprint(terminal_item),
                    )
                    break

                rejected_count = sum(
                    not _fidelity_item_accepted(gate_item)
                    for _gate, gate_item in _fidelity_gate_history(
                        persisted,
                        item.stable_id,
                        campaign_root_work_id=_artifact_lineage(artifact),
                    )
                )
                if rejected_count:
                    try:
                        artifact = _require_legacy_audit_fidelity(
                            item,
                            self._retry_infrastructure_call(
                                lambda: self._repair_author(
                                    item,
                                    artifact,
                                    persisted,
                                    rejected_count,
                                    reducer,
                                    gate_kind="fidelity",
                                    operational_events=operational.records,
                                ),
                                admission=("author", item),
                                operational=operational,
                            ),
                            self.config,
                        )
                    except AuthorBackoffError as backoff:
                        # A provider pause during a bounded repair is a campaign pause with a
                        # reset time, not a failed repair. `_ensure_gates` already returns a
                        # usage-pause reason, so route it the same way as a checker backoff.
                        pool.cancel_pending()  # BL-8 (d): a pause never buys queued sessions
                        return self._pause_for_usage(backoff.signal, operational, len(work))
                    except AuthorSessionRetryExhausted as exhausted:
                        # No-publication session deaths through the fidelity-repair
                        # lane meet the same mechanical boundary as the metadata arm.
                        if self._scope_author_retry_exhaustion(
                            item, artifact, exhausted, reducer, operational, state
                        ):
                            break
                        raise
                    except RetryableOperatorError:
                        raise
                    except AuthorRepairTerminal as terminal:
                        # See the metadata arm above: a typed terminal recommendation
                        # is the author's answer to the fidelity findings, not a
                        # failed repair, and settles through the terminal gate.
                        pause = self._route_repair_terminal_author_result(
                            item, terminal, reducer, operational, state
                        )
                        if pause is not None:
                            return pause
                        break
                    except Exception as exc:  # noqa: BLE001 -- repair failure is model-local
                        stage, reason = self._repair_failure_stage_and_reason(exc)
                        attempt = _driver_failure_attempt(
                            item,
                            artifact,
                            stage,
                            reason,
                            exc,
                            self.config,
                            diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                            environment=None,
                            created_at=self.dependencies.clock(),
                        )
                        persisted_attempt = reducer.append_attempt(attempt).record
                        self._terminalize(
                            item,
                            artifact,
                            f"failed:{stage}",
                            reason,
                            str(exc),
                            (persisted_attempt,),
                            reducer,
                            operational,
                            state,
                        )
                        break
                    artifacts[item.stable_id] = artifact
                    metadata_blocked = False
                    while True:
                        # BL-10 (b): the absolute round ceiling holds in the
                        # repair-loop re-check exactly as in the batch lane --
                        # checked BEFORE a session is bought.
                        ceiling_history = _metadata_gate_history(
                            persisted,
                            item.stable_id,
                            campaign_root_work_id=_artifact_lineage(artifact),
                        )
                        if len(ceiling_history) >= METADATA_GATE_ROUND_CEILING:
                            self._record_checker_lane_deferral(
                                item,
                                DriverIntegrationError(
                                    f"metadata gate round ceiling: {len(ceiling_history)} "
                                    "persisted rounds without acceptance (ceiling "
                                    f"{METADATA_GATE_ROUND_CEILING}); checker disclosure "
                                    "debt routed to human review"
                                ),
                                attempted_batch_sizes=tuple(
                                    max(int(gate.get("batch_size", 1) or 1), 1)
                                    for gate, _gate_item in ceiling_history
                                ),
                                operational=operational,
                                failure_site="disclosure-debt-ceiling",
                                gate_kind="metadata_batch",
                                failure_class="checker-disclosure-debt",
                                count_in_streak=False,
                            )
                            metadata_blocked = True
                            break
                        reused_metadata = reusable_metadata_gate(
                            (artifact,),
                            persisted,
                            checker=lane_adjudication_identity(
                                self.dependencies.checker, "metadata_batch", self.config
                            ),
                        )
                        if reused_metadata is not None:
                            metadata_outcome = CheckerOutcome(gate=reused_metadata)
                            self._record_checker_reuse(
                                operational,
                                gate_kind="metadata_batch",
                                stable_ids=(item.stable_id,),
                                reused_from=reused_metadata[GATE_REUSED_FROM_FIELD],
                            )
                        else:
                            try:
                                metadata_outcome = self._retry_infrastructure_call(
                                    lambda: self.dependencies.checker.check_metadata(
                                        (artifact,), self.paths.work_root, self.config
                                    ),
                                    admission=("checker", item),
                                    operational=operational,
                                    checker_scope=self._checker_spend_scope(
                                        "metadata_batch", (artifact,)
                                    ),
                                )
                            except (RetryableOperatorError, CheckerPromptRotatedError):
                                raise
                            except Exception as exc:  # noqa: BLE001 -- lane failure, never a model fact
                                # BL-10 CH-F4: the single-model metadata re-check
                                # inside the fidelity-repair loop used to
                                # terminalize ``failed:accuracy-gate /
                                # checker-contract-invalid`` on a checker LANE
                                # failure -- the rung-10 wrong-blame class,
                                # surviving in exactly this path while the batch
                                # lane was rebuilt so a lane failure can never
                                # blame a model. A lane failure PARKS; only a
                                # validated adverse VERDICT terminalizes.
                                self._record_checker_lane_deferral(
                                    item,
                                    exc,
                                    attempted_batch_sizes=(1,),
                                    operational=operational,
                                    failure_site="fidelity-repair-recheck",
                                    gate_kind="metadata_batch",
                                )
                                metadata_blocked = True
                                break
                        if metadata_outcome.backoff is not None:
                            pool.cancel_pending()  # BL-8 (d)
                            return self._pause_for_usage(
                                metadata_outcome.backoff, operational, len(work)
                            )
                        try:
                            metadata_gate = _normalize_gate_generation(
                                _require_gate(metadata_outcome), persisted, (item.stable_id,)
                            )
                            _require_gate_bindings(metadata_gate, (artifact,), "metadata_batch")
                            metadata_ready = _prepare_ledger_record(
                                metadata_gate, len(persisted) + 1
                            )
                            metadata_decision = route_metadata_gate(
                                metadata_ready,
                                {
                                    item.stable_id: _chargeable_metadata_repair_count(
                                        persisted,
                                        item.stable_id,
                                        _artifact_lineage(artifact),
                                    )
                                },
                                max_repairs=2,
                                prior_complaint_fields={
                                    item.stable_id: disclosed_metadata_complaint_union(
                                        _metadata_gate_history(
                                            persisted,
                                            item.stable_id,
                                            campaign_root_work_id=_artifact_lineage(artifact),
                                        )
                                    )
                                },
                            )[0]
                            # Same unhandled `GateRoutingError` as the batch lane above:
                            # validate the record inside the model-local handler.
                            metadata_gate_records = emit_gate_records(metadata_ready)
                        except (DriverPaused, RetryableOperatorError, AuthorBackoffError, CheckerPromptRotatedError):
                            raise
                        except (AuthorEffortCapExceeded, AuthorEffortExhaustionClaim) as exc:
                            # W1/PD-13: shared exhaustion arm (event XOR storm valve).
                            self._terminalize_effort_exhaustion(
                                item,
                                artifact,
                                exc,
                                reducer,
                                operational,
                                state,
                            )
                            metadata_blocked = True
                            break
                        except Exception as exc:  # noqa: BLE001 -- invalid checker contract is lane-owned
                            # BL-10 CH-F4: same wrong-blame arm as the dispatch
                            # side above -- a gate the routing validators refuse
                            # is exactly as verdict-free as a dispatch failure.
                            # Park, never terminalize.
                            self._record_checker_lane_deferral(
                                item,
                                exc,
                                attempted_batch_sizes=(1,),
                                operational=operational,
                                failure_site="fidelity-repair-recheck-routing",
                                gate_kind="metadata_batch",
                            )
                            metadata_blocked = True
                            break
                        for record in metadata_gate_records:
                            appended = reducer.append_gate(_without_ledger_fields(record))
                            if appended.appended:
                                persisted.append(appended.record)
                        self._note_checker_gate_routed()
                        if metadata_decision.canonical_write_allowed:
                            # H-5: the re-checked acceptance passes the SAME
                            # duplicate-accept write door as the batch lane's
                            # (this is the second of the two metadata dispatch
                            # sites; an acceptance minted here without the
                            # bound confirmation would bypass K=2).
                            recheck_gate = _find_gate(
                                persisted, item.stable_id, "metadata_batch", artifact.proposal
                            )
                            if recheck_gate is not None:
                                _stands, confirm_pause = (
                                    self._require_confirmed_accuracy_acceptance(
                                        item,
                                        artifact,
                                        recheck_gate,
                                        "metadata_batch",
                                        reducer,
                                        operational,
                                    )
                                )
                                if confirm_pause is not None:
                                    pool.cancel_pending()  # BL-8 (d)
                                    return confirm_pause
                                if not _stands:
                                    metadata_blocked = True
                            break
                        if metadata_decision.human_review_required:
                            self._terminalize_accuracy_gate(
                                item,
                                artifact,
                                metadata_ready,
                                reducer,
                                operational,
                                state,
                            )
                            metadata_blocked = True
                            break
                        metadata_repair_count = _metadata_repair_count(
                            persisted,
                            item.stable_id,
                            _artifact_lineage(artifact),
                        )
                        try:
                            artifact = self._retry_infrastructure_call(
                                lambda: self._repair_author(
                                    item,
                                    artifact,
                                    persisted,
                                    metadata_repair_count,
                                    reducer,
                                    gate_kind="metadata_batch",
                                    operational_events=operational.records,
                                ),
                                admission=("author", item),
                                operational=operational,
                            )
                        except AuthorBackoffError as backoff:
                            # A provider pause during a bounded repair is a campaign pause with a
                            # reset time, not a failed repair. `_ensure_gates` already returns a
                            # usage-pause reason, so route it the same way as a checker backoff.
                            pool.cancel_pending()  # BL-8 (d): a pause never buys queued sessions
                            return self._pause_for_usage(backoff.signal, operational, len(work))
                        except AuthorSessionRetryExhausted as exhausted:
                            # Same mechanical boundary as the batch repair lane: a
                            # model-scoped verdict terminalizes this model with its
                            # retained artifact; anything else re-raises loud.
                            if self._scope_author_retry_exhaustion(
                                item, artifact, exhausted, reducer, operational, state
                            ):
                                metadata_blocked = True
                                break
                            raise
                        except RetryableOperatorError:
                            raise
                        except AuthorRepairTerminal as terminal:
                            # See the metadata arm above: the author's typed terminal
                            # answer settles through the terminal-disposition gate
                            # instead of being recorded as a repair protocol breach.
                            pause = self._route_repair_terminal_author_result(
                                item, terminal, reducer, operational, state
                            )
                            if pause is not None:
                                return pause
                            metadata_blocked = True
                            break
                        except Exception as exc:  # noqa: BLE001 -- repair failure is model-local
                            stage, reason = self._repair_failure_stage_and_reason(exc)
                            attempt = _driver_failure_attempt(
                                item,
                                artifact,
                                stage,
                                reason,
                                exc,
                                self.config,
                                diagnostics_root=_diagnostics_root_for_work_root(
                                    self.paths.work_root
                                ),
                                environment=None,
                                created_at=self.dependencies.clock(),
                            )
                            persisted_attempt = reducer.append_attempt(attempt).record
                            self._terminalize(
                                item,
                                artifact,
                                f"failed:{stage}",
                                reason,
                                str(exc),
                                (persisted_attempt,),
                                reducer,
                                operational,
                                state,
                            )
                            metadata_blocked = True
                            break
                        artifacts[item.stable_id] = artifact
                    if metadata_blocked:
                        break

                reused_fidelity = reusable_fidelity_gate(
                    artifact,
                    persisted,
                    checker=lane_adjudication_identity(
                        self.dependencies.checker, "fidelity", self.config
                    ),
                )
                if reused_fidelity is not None:
                    outcome = CheckerOutcome(gate=reused_fidelity)
                    self._record_checker_reuse(
                        operational,
                        gate_kind="fidelity",
                        stable_ids=(item.stable_id,),
                        reused_from=reused_fidelity[GATE_REUSED_FROM_FIELD],
                    )
                else:
                    try:
                        # The pool holds at most this model's FIRST fidelity
                        # session; ``claim`` pops, so post-repair re-checks in
                        # later passes of this loop dispatch live.
                        capsule = pool.claim(_fidelity_pool_key(item.stable_id))
                        checker_scope = self._checker_spend_scope("fidelity", (artifact,))
                        outcome = self._retry_infrastructure_call(
                            _replay_then_live(
                                capsule,
                                lambda: self.dependencies.checker.check_fidelity(
                                    artifact, self.paths.work_root, self.config
                                ),
                            ),
                            admission=("checker", item),
                            operational=operational,
                            checker_scope=checker_scope,
                            prepaid_checker_reservation=(
                                capsule.reservation if capsule is not None else None
                            ),
                        )
                    except (DriverPaused, RetryableOperatorError, AuthorBackoffError, CheckerPromptRotatedError):
                        raise
                    except CheckerMechanicalClaimError as exc:
                        # W-8: a machine-refuted fidelity checker claim parks
                        # the model as checker-verdict-conflict -- never a
                        # member verdict, never a silent re-roll.
                        self._record_checker_lane_deferral(
                            item,
                            exc,
                            attempted_batch_sizes=(1,),
                            operational=operational,
                            failure_site="mechanical-refutation",
                            gate_kind="fidelity",
                            failure_class="checker-verdict-conflict",
                            count_in_streak=False,
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 -- checker failure belongs to this model
                        self._record_checker_lane_deferral(
                            item,
                            exc,
                            attempted_batch_sizes=(1,),
                            operational=operational,
                            failure_site="checker-dispatch",
                            gate_kind="fidelity",
                        )
                        break
                if outcome.backoff is not None:
                    pool.cancel_pending()  # BL-8 (d): a pause never buys queued sessions
                    return self._pause_for_usage(outcome.backoff, operational, len(work))
                try:
                    gate = _normalize_gate_generation(
                        _require_gate(outcome), persisted, (item.stable_id,)
                    )
                    _require_gate_bindings(gate, (artifact,), "fidelity")
                    route_ready = _prepare_ledger_record(gate, len(persisted) + 1)
                    route_fidelity_gate(route_ready, artifact.proposal)
                    # `GateRoutingError` from routing had no handler above this frame
                    # either; keep the whole gate contract inside the model-local arm.
                    emit_gate_records(route_ready)
                except (DriverPaused, RetryableOperatorError, AuthorBackoffError, CheckerPromptRotatedError):
                    raise
                except StaleGateBindingError as exc:
                    attempt = _driver_failure_attempt(
                        item,
                        artifact,
                        "runner",
                        "protocol-violation",
                        exc,
                        self.config,
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                        environment=None,
                        created_at=self.dependencies.clock(),
                    )
                    persisted_attempt = reducer.append_attempt(attempt).record
                    self._terminalize(
                        item,
                        artifact,
                        "failed:runner",
                        "protocol-violation",
                        str(exc),
                        (persisted_attempt,),
                        reducer,
                        operational,
                        state,
                        human_review=True,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 -- invalid checker contract is model-local
                    self._record_checker_lane_deferral(
                        item,
                        exc,
                        attempted_batch_sizes=(1,),
                        operational=operational,
                        failure_site="gate-routing",
                        gate_kind="fidelity",
                    )
                    break
                persisted_gate = reducer.append_gate(_without_ledger_fields(route_ready)).record
                persisted.append(persisted_gate)
                self._note_checker_gate_routed()
                self.dependencies.boundary_hook("after-gate", item.stable_id)
        return None

    def _retry_infrastructure_call(
        self,
        operation: Callable[[], _T],
        *,
        admission: Optional[tuple[str, WorkItem]] = None,
        operational: Optional[JsonlLedger] = None,
        weather_queued_models: int = 1,
        author_admission: Optional[Callable[[Optional[str]], Reservation]] = None,
        checker_scope: Optional[_CheckerSpendScope] = None,
        prepaid_checker_reservation: Optional[Reservation] = None,
        author_ladder: Optional[_RetryLadderState] = None,
    ) -> _T:
        """Retry one external author or checker infrastructure failure, with backoff.

        Rung 7 (m7362) measured the old ladder: exactly one retry, dispatched 0.85s
        after a 0.6-second rc=1 session death, then a campaign abort. A zero-backoff
        single retry cannot survive any host condition lasting longer than about a
        second -- the observed cause was a transient CLI credential blip ("Not logged
        in") that resolved on its own. Unclassified failures retain the bounded
        ``_INFRASTRUCTURE_RETRY_BACKOFF_SECONDS`` ladder. Exact machine-derived
        stream aborts and spawn failures instead retry at
        the SESSION level with deterministic jitter until paid-session QUOTA or
        RUNAWAY refuses the next buy; their settled attempt group weighs zero,
        but every provider session and machine wall second remains charged. All
        waits ride the shutdown event (or the injected test sleeper), so SIGTERM
        interrupts them immediately.

        On bounded-ladder exhaustion, an AUTHOR-lane no-publication failure is
        re-raised as :class:`AuthorSessionRetryExhausted` -- still a
        ``RetryableOperatorError``, so every handler that does not consult the
        lane-health boundary keeps the campaign-scoped LOUD behavior -- while
        specialized transports (provider outages with their own streak promotion,
        queue stalls, backoff signals) and published-content refusals propagate
        unwrapped exactly as before.

        Parameters
        ----------
        operation:
            Zero-argument external lane invocation.
        admission:
            Optional lane and work item for the exact signalable admission event.
        operational:
            Canonical operational ledger used for checker weather heartbeats.
        weather_queued_models:
            Number of live-pending models held by this checker invocation.
        author_admission:
            Optional wave-worker handshake that blocks until the canonical
            writer durably records this exact author buy.  Called with the
            attempt group to join (``None`` heads a new attempt).
        checker_scope:
            Durable work/checker binding for each live checker attempt.
        prepaid_checker_reservation:
            Reservation already minted by a checker-pool writer handshake. When
            ``operation`` is a :class:`_ReplayedFirstAttempt` the reservation AND
            the wall the pool worker measured around it are taken from the capsule
            itself, so the two cannot be threaded apart: a replay costs ~0s on this
            thread, and a writer re-measurement of it would delete the real session
            wall from the wall-based checker QUOTA and the epoch RUNAWAY_BREAKER --
            under-counting exactly the runaway the breaker exists for (PSLRE4 F1).
        author_ladder:
            Optional cross-round author retry-ladder continuation (ACAP). When
            present, its ``per_round_cap`` bounds how many attempts THIS call
            may head; a ladder that neither resolves nor terminalizes within
            the budget raises the typed :class:`_AuthorRoundAttemptsDeferred`
            so the commit lane returns the model unresolved and a later
            scheduler round resumes the same counters. ``None`` (wave driver,
            checker lane, direct callers) keeps the historical
            unbounded-in-call ladder.

        Returns
        -------
        Any
            Successful lane result.

        Raises
        ------
        Exception
            The first contract error, or the final infrastructure error once the
            bounded budget is spent.
        """

        backoffs = _INFRASTRUCTURE_RETRY_BACKOFF_SECONDS
        # ACAP: the ladder counters live in one continuation object so a capped
        # author ladder can defer between scheduler rounds and resume exactly
        # where it stopped. Uncapped lanes get a fresh call-local continuation,
        # which reproduces the historical local-variable ladder unchanged.
        ladder = author_ladder if author_ladder is not None else _RetryLadderState()
        ladder.attempts_this_round = 0
        # The reservation and the wall the pool worker measured around it are ONE
        # value carried by the capsule's own replay operation: whichever attempt
        # spends the prepaid reservation also inherits its measured wall as the
        # settlement floor, and both are consumed together.
        prepaid_reservation, prepaid_wall = _prepaid_checker_session(
            operation, prepaid_checker_reservation
        )
        while True:
            if ladder.per_round_cap is not None and (
                ladder.attempts_this_round >= ladder.per_round_cap
            ):
                # ACAP: the in-round attempt budget is spent without an artifact
                # or a terminal. Raised BEFORE any admission is bought, so the
                # unrun attempt mints no money row and the resumed round heads
                # it exactly as the unbounded ladder would have -- accounting
                # is unchanged, only WHEN the attempt runs moves.
                if admission is None or admission[0] != "author":
                    raise DriverIntegrationError(
                        "per-round attempt caps are an author-lane continuation"
                    )
                raise _AuthorRoundAttemptsDeferred(
                    admission[1].stable_id,
                    attempts_this_round=ladder.attempts_this_round,
                    failed_attempts=ladder.failed_attempts,
                )
            if ladder.pending_backoff_deadline is not None:
                # A deferral parked the previous retry's wait; the resumed round
                # owes only the remainder of the attempt spacing.
                remaining = ladder.pending_backoff_deadline - self.dependencies.monotonic()
                ladder.pending_backoff_deadline = None
                if remaining > 0:
                    self._wait_before_infrastructure_retry(remaining)
            if admission is not None:
                lane, item = admission
                self.dependencies.boundary_hook(f"pre-{lane}", item.stable_id)
                self._check_shutdown(f"{lane}-admission", item=item)
            self._check_shutdown("external-call-admission")
            reservation: Optional[Reservation] = None
            # Measured wall this attempt's reservation already carries, before
            # this thread measures anything of its own.
            prepaid_attempt_wall = 0.0
            admit_next: Optional[Callable[[Optional[str]], Reservation]] = None
            if admission is not None and admission[0] == "author":
                if operational is None:
                    raise DriverIntegrationError(
                        "author session admission requires the operational spend ledger"
                    )
                # The author lane runs UP TO two guard sessions per attempt
                # (source-request, then author). Each is its own reservation, so
                # the same admission valve that bought attempt-session one is
                # handed to the dispatch scope for session two: settle the
                # finished session, then admit the next JOINED to this attempt's
                # group -- never rebind, and never a second attempt charge.
                if author_admission is None:
                    author_item = admission[1]
                    admit_next = lambda attempt_group: self._admit_author_spend(  # noqa: E731
                        author_item, operational, attempt_group=attempt_group
                    )
                else:
                    admit_next = author_admission
                # Each ladder attempt heads its OWN durable attempt group.
                reservation = admit_next(None)
                ladder.attempts_this_round += 1
            if (
                admission is not None
                and admission[0] == "checker"
                and (operational is not None or checker_scope is not None)
            ):
                if operational is None or checker_scope is None:
                    raise DriverIntegrationError(
                        "checker session admission requires a durable scope and spend ledger"
                    )
                if prepaid_reservation is not None:
                    reservation, prepaid_reservation = prepaid_reservation, None
                    prepaid_attempt_wall, prepaid_wall = prepaid_wall, 0.0
                else:
                    reservation = self._admit_checker_spend(operational, checker_scope)
            attempt_started = self.dependencies.monotonic()
            sessions: Optional[_PaidDispatchSessions] = None
            if reservation is not None:
                # The checker lane dispatches exactly one guard per reservation,
                # so it declares no per-session admission (admit=None): a second
                # guard session there refuses typed before any spawn.
                sessions = _PaidDispatchSessions(
                    reservation,
                    admit=admit_next,
                    settle=lambda held, elapsed, done, outcome: self._settle_paid_session(
                        held,
                        elapsed_seconds=elapsed,
                        completed=done,
                        local_outcome=outcome,
                        # Author-lane settlements feed the cross-model storm
                        # fuse (W26); checker sessions ride their own bounded
                        # weather park ladder and are not counted.
                        storm_model=(
                            admission[1].stable_id
                            if admission is not None and admission[0] == "author"
                            else None
                        ),
                    ),
                    span_started=attempt_started,
                    monotonic=self.dependencies.monotonic,
                )
            try:
                try:
                    if sessions is None:
                        result = operation()
                    else:
                        # `sessions` is only ever built from a non-None
                        # `reservation` (above), so this holds by construction.
                        assert reservation is not None
                        with (
                            authorized_paid_dispatch(reservation),
                            _paid_dispatch_sessions(sessions),
                        ):
                            result = operation()
                except BaseException as exc:
                    if sessions is not None:
                        # A replayed pool capsule costs ~0s on THIS thread, so the
                        # wall the worker measured is the floor -- never a writer
                        # re-measurement of the replay. Guard sessions the scope
                        # already settled at their boundary stay settled; only
                        # the still-unsettled reservation settles here.
                        sessions.settle_outcome(
                            completed=False,
                            minimum_wall=prepaid_attempt_wall,
                            local_outcome=(
                                _paid_session_infrastructure_local_outcome(exc) or "failed"
                            ),
                        )
                    raise
                else:
                    if sessions is not None:
                        observed_wall = (
                            float(result.wall_seconds)
                            if isinstance(result, CheckerOutcome)
                            and result.wall_seconds is not None
                            else None
                        )
                        sessions.settle_outcome(
                            completed=True,
                            minimum_wall=prepaid_attempt_wall,
                            observed_wall=observed_wall,
                            # AL-6: a typed BLOCKED terminal recommendation
                            # refunds the ATTEMPT projection only (the whole
                            # group closes on this zero-weight settlement);
                            # sessions and wall stay charged as always.
                            local_outcome=_author_blocked_prerequisite_settlement_outcome(
                                admission, result
                            ),
                        )
            except AuthorExecutorContractError:
                # ``AuthorExecutorContractError`` keeps its own calibrated budget:
                # ONE fresh bounded session -- the author is stochastic, and one
                # fresh session routinely gets further than the last -- with NO
                # backoff, because a contract-invalid payload is not a
                # time-correlated condition and a full author session is
                # expensive. It is deliberately NOT ``_is_infrastructure_error``:
                # that predicate also feeds the reason pickers, and a
                # contract-invalid author payload recorded as ``internal-error``
                # would indict the engine for the author's bytes. When this
                # budget is spent it propagates as the model-local failure it is,
                # never as a campaign abort.
                if ladder.contract_retries >= 1:
                    raise
                ladder.contract_retries += 1
                continue
            except Exception as exc:  # noqa: BLE001 -- typed below before retry
                if isinstance(exc, (CheckerLegBudgetExceeded, CrossModelProviderStorm)):
                    # BL-10 (g) / W26: a self-imposed breaker is not weather --
                    # retrying inside this ladder cannot change the accounting
                    # (or drain the storm window) it derives from, and burning
                    # the bounded retries here would exhaust into a model-scoped
                    # terminal. Loud and resumable immediately, campaign scope.
                    raise
                if not self._is_infrastructure_error(exc):
                    raise
                ladder.failed_attempts += 1
                if self._has_provider_overloaded_failure_class(exc):
                    weather_schedule = (
                        CHECKER_WEATHER_BACKOFF_SECONDS
                        if self._has_checker_provider_weather_class(exc)
                        else _PROVIDER_OVERLOAD_RETRY_BACKOFF_SECONDS
                    )
                    if ladder.provider_overload_retries < len(weather_schedule):
                        backoff = self._provider_overload_retry_backoff(
                            ladder.provider_overload_retries,
                            schedule=weather_schedule,
                        )
                        if (
                            admission is not None
                            and admission[0] == "checker"
                            and operational is not None
                        ):
                            self._record_checker_weather_park(
                                operational,
                                admission[1],
                                exc,
                                interval_index=ladder.provider_overload_retries,
                                backoff_seconds=backoff,
                                queued_models=weather_queued_models,
                            )
                        ladder.provider_overload_retries += 1
                        self._wait_or_park_infrastructure_backoff(ladder, backoff)
                        continue
                    if self._has_checker_provider_weather_class(
                        exc
                    ) and isinstance(exc, RetryableOperatorError):
                        exhausted = RetryableOperatorError(
                            "checker provider-overloaded storm exhausted its bounded park "
                            f"after {ladder.failed_attempts} invocations and waits "
                            f"{ladder.observed_backoffs}; the campaign remains resumable"
                        )
                        exhausted.failure_class = exc.failure_class
                        exhausted.failure_class_basis = exc.failure_class_basis
                        exhausted.request_sha256 = exc.request_sha256
                        raise exhausted from exc
                    ladder.infrastructure_retries = len(backoffs)
                else:
                    ladder.provider_overload_retries = 0
                machine_infrastructure = _paid_session_infrastructure_local_outcome(exc)
                if (
                    machine_infrastructure is not None
                    and admission is not None
                    and admission[0] == "author"
                    and not self._has_provider_overloaded_failure_class(exc)
                    and machine_infrastructure != "failed-author-queue-stalled"
                ):
                    # Retry a machine-proven death as another SESSION.  The
                    # just-settled group now weighs zero attempts, while its raw
                    # session/wall charge remains visible to the next admission's
                    # QUOTA and RUNAWAY checks.  Those fuses, not the model effort
                    # allowance, are the terminal bound on an infrastructure storm.
                    backoff = self._infrastructure_session_retry_backoff(
                        retry_index=ladder.infrastructure_session_retries,
                        jitter_key=admission[1].stable_id,
                    )
                    ladder.infrastructure_session_retries += 1
                    self._wait_or_park_infrastructure_backoff(ladder, backoff)
                    continue
                if ladder.infrastructure_retries < len(backoffs):
                    backoff = self._infrastructure_retry_backoff(
                        retry_index=ladder.infrastructure_retries
                    )
                    ladder.infrastructure_retries += 1
                    self._wait_or_park_infrastructure_backoff(ladder, backoff)
                    continue
                if (
                    admission is not None
                    and admission[0] == "author"
                    and self._author_no_publication_class(exc)
                ):
                    raise AuthorSessionRetryExhausted(
                        admission[1].stable_id,
                        attempts=ladder.failed_attempts,
                        backoff_seconds=tuple(ladder.observed_backoffs),
                        detail=str(exc)[-STDIO_TAIL_MAX_CHARS:],
                    ) from exc
                raise
            else:
                if admission is not None and admission[0] == "author":
                    # Any successful author-lane round trip is a same-lane,
                    # same-run publication witness for the boundary below.
                    self._note_author_lane_publication(admission[1].stable_id)
                return result

    def _wait_or_park_infrastructure_backoff(
        self, ladder: _RetryLadderState, backoff: float
    ) -> None:
        """Wait one retry backoff inline, or park it on a deferring ladder.

        A capped ladder whose per-round attempt budget is already spent is
        about to defer at the loop head; sleeping first would idle the pool
        slot for exactly the wait the cap exists to reclaim (ACAP). The
        attempt spacing is preserved, never skipped: the deadline rides the
        continuation and the resumed round waits out only the remainder
        before buying its next attempt.

        Parameters
        ----------
        ladder:
            Retry-ladder continuation owning the counters and the parked wait.
        backoff:
            Wait duration selected by the retry arm, in seconds.
        """

        ladder.observed_backoffs.append(backoff)
        if ladder.per_round_cap is not None and (
            ladder.attempts_this_round >= ladder.per_round_cap
        ):
            ladder.pending_backoff_deadline = self.dependencies.monotonic() + backoff
            return
        self._wait_before_infrastructure_retry(backoff)

    @staticmethod
    def _infrastructure_retry_backoff(
        *,
        retry_index: int,
    ) -> float:
        """Return the bounded wait before the next infrastructure retry.

        Parameters
        ----------
        retry_index:
            Zero-based retry wait index.

        Returns
        -------
        float
            Backoff duration in seconds.
        """

        return _INFRASTRUCTURE_RETRY_BACKOFF_SECONDS[retry_index]

    @staticmethod
    def _infrastructure_session_retry_backoff(
        *,
        retry_index: int,
        jitter_key: str,
    ) -> float:
        """Return one deterministic jittered session-infrastructure wait.

        Parameters
        ----------
        retry_index:
            Zero-based infrastructure session retry index.
        jitter_key:
            Stable model identity used to de-correlate concurrent waves.

        Returns
        -------
        float
            Jittered backoff capped by the closed session ceiling.
        """

        base = _INFRASTRUCTURE_SESSION_BACKOFF_SECONDS[
            min(retry_index, len(_INFRASTRUCTURE_SESSION_BACKOFF_SECONDS) - 1)
        ]
        digest = stable_hash(
            {
                "kind": "author-infrastructure-session-backoff",
                "stable_id": jitter_key,
                "retry_index": retry_index,
            }
        ).removeprefix("sha256:")
        fraction = int(digest[:8], 16) / 0xFFFFFFFF
        jittered = base * (0.8 + 0.4 * fraction)
        return round(min(jittered, _INFRASTRUCTURE_SESSION_BACKOFF_CEILING_SECONDS), 3)

    @staticmethod
    def _provider_overload_retry_backoff(
        retry_index: int,
        *,
        schedule: tuple[float, ...] = _PROVIDER_OVERLOAD_RETRY_BACKOFF_SECONDS,
    ) -> float:
        """Return the bounded wait before a provider-overload storm retry.

        Parameters
        ----------
        retry_index:
            Zero-based provider-overload wait index.
        schedule:
            Per-storm weather schedule for the active operator lane.

        Returns
        -------
        float
            Weather backoff duration in seconds, capped by the public ceiling.
        """

        weather_index = min(
            retry_index,
            len(schedule) - 1,
        )
        return min(
            schedule[weather_index],
            min(
                _PROVIDER_OVERLOAD_BACKOFF_CEILING_SECONDS,
                CHECKER_WEATHER_BACKOFF_CEILING_SECONDS,
            ),
        )

    def _wait_before_infrastructure_retry(self, seconds: float) -> None:
        """Wait out one backoff entry without ever outliving a shutdown request.

        Parameters
        ----------
        seconds:
            Backoff duration from ``_INFRASTRUCTURE_RETRY_BACKOFF_SECONDS``.
        """

        if seconds <= 0:
            return
        sleeper = self.dependencies.sleeper
        if sleeper is not None:
            sleeper(seconds)
            # Injected sleepers consume no real time; one drain afterwards keeps
            # fake-clock rigs from parking queued buys behind a zero-length wait.
            _service_author_admission_relay()
            return
        entry = _AUTHOR_ADMISSION_RELAY_SERVICE.get()
        if entry is None or entry[0] != threading.get_ident():
            # ``Event.wait`` returns early the moment the signal handler sets the
            # event; the admission check on the next loop iteration then raises
            # the typed shutdown, so a SIGTERM never waits out a 30-second backoff.
            self._shutdown_event.wait(seconds)
            return
        # Writer-side backoff inside a live author wave: drain queued worker
        # buys between short slices so a 5-900s inline-retry sleep no longer
        # starves every sibling's session start. Shutdown responsiveness is
        # unchanged -- the event still ends the wait within one slice.
        service = entry[1]
        deadline = time.monotonic() + seconds
        while True:
            service()
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._shutdown_event.wait(min(0.1, remaining)):
                return

    @staticmethod
    def _has_provider_overloaded_failure_class(exc: Exception) -> bool:
        """Return whether an exception chain carries the structured 529 class.

        Parameters
        ----------
        exc:
            External lane exception.

        Returns
        -------
        bool
            ``True`` only when an executor-authored typed notice carried
            ``failure_class="provider-overloaded"`` with a trusted executor basis.
        """

        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, RetryableOperatorError) and (
                (
                    current.failure_class == "provider-overloaded"
                    and current.failure_class_basis
                    in {
                        "api_error_status:529",
                        "executor-record:provider-overloaded-api-error",
                    }
                )
                or AdmissionEnvironmentMixin._has_checker_provider_weather_class(current)
            ):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _has_checker_provider_weather_class(exc: Exception) -> bool:
        """Return whether an exception has the exact checker weather class and basis."""

        return (
            isinstance(exc, RetryableOperatorError)
            and exc.failure_class == "checker-provider-overloaded"
            and exc.failure_class_basis in _CHECKER_PROVIDER_OVERLOAD_BASES
            and isinstance(exc.request_sha256, str)
            and exc.request_sha256.startswith("sha256:")
        )

    def _record_checker_weather_park(
        self,
        operational: JsonlLedger,
        item: WorkItem,
        exc: Exception,
        *,
        interval_index: int,
        backoff_seconds: float,
        queued_models: int,
    ) -> None:
        """Append one observable heartbeat for a bounded checker weather interval."""

        created_at = self.dependencies.clock()
        request_sha256 = getattr(exc, "request_sha256", None)
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.RETRYABLE_INFRASTRUCTURE.value,
            # The provider of the SEAT whose request parked (stamped on the
            # typed error by `_raise_for_checker_exit`); the pre-swap era
            # value when the error predates the stamp.
            "provider": getattr(exc, "checker_provider", None) or "openai",
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": queued_models},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "kind": "checker-provider-weather-park",
                "storm_id": request_sha256,
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "interval_index": interval_index,
                "backoff_seconds": backoff_seconds,
                "failure_class": "checker-provider-overloaded",
                "failure_class_basis": _CHECKER_PROVIDER_OVERLOAD_BASIS,
            },
        }
        event = mint_operational_event(event)
        operational.append(event)

    @staticmethod
    def _repair_failure_stage_and_reason(exc: Exception) -> tuple[str, str]:
        """Classify a model-local author repair failure for terminal recording.

        Parameters
        ----------
        exc:
            Exception raised by a bounded author repair round.

        Returns
        -------
        tuple[str, str]
            Closed ``(stage, reason_code)`` pair.
        """

        if isinstance(
            exc,
            (
                AuthorEffortCapExceeded,
                AuthorEffortExhaustionClaim,
                AuthorResultMalformedError,
                AuthorSourceDeterministicallyUnfetchableError,
                ArtifactBindingError,
                AuthorEnvironmentBindingError,
            ),
        ):
            return _author_lane_failure(exc)
        if isinstance(
            exc, DriverIntegrationError
        ) and not AdmissionEnvironmentMixin._is_infrastructure_error(exc):
            return "runner", "protocol-violation"
        return "runner", "internal-error"

    def _author_no_publication_class(self, exc: Exception) -> bool:
        """Return whether one exhausted author failure is a NO-PUBLICATION death.

        The lane-health boundary applies to exactly the class BATON trap 3 leaves
        ambiguous: sessions that died without publishing (``session-crashed``,
        ``wall-exceeded``, ``no-result-output``, unclassified exits, spawn
        failures). Specialized transports carry their own semantics and are
        excluded: provider outages (``AuthorOutageError``) feed the author-wave
        streak promotion, queue stalls are lane-wide by construction, backoff
        signals are campaign pauses, and published-content refusals are already
        model-scoped under their own types.

        Parameters
        ----------
        exc:
            Final exception after the bounded retry budget spent.

        Returns
        -------
        bool
            Whether the lane-health boundary may adjudicate this failure.
        """

        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(
                current,
                (
                    AuthorOutageError,
                    AuthorQueueStalled,
                    AuthorSessionRetryExhausted,
                    AuthorBackoffError,
                    AuthorExecutorContractError,
                ),
            ):
                return False
            current = current.__cause__ or current.__context__
        return self._is_infrastructure_error(exc)

    def _note_author_lane_publication(self, stable_id: str) -> None:
        """Record one successful author-lane round trip as a lane-health witness.

        Any author session that ran to completion and published -- a proposal, a
        typed terminal, a repair generation -- is contemporaneous evidence that the
        host's author lane works, and it clears the consecutive no-publication
        death streak. Wave-pool workers call this from their own threads, so the
        two structures share one lock.

        Parameters
        ----------
        stable_id:
            Model whose author session published.
        """

        with self._author_lane_state_lock:
            self._author_lane_publications.append(
                {
                    "stable_id": stable_id,
                    "at": self.dependencies.clock(),
                    "at_monotonic": self.dependencies.monotonic(),
                }
            )
            del self._author_lane_publications[:-_AUTHOR_LANE_WITNESS_CAP]
            self._author_no_publication_deaths.clear()

    def _author_lane_health_checks(self) -> JsonObject:
        """Run the cheap host checks the lane-health boundary requires.

        Three mechanical facts, none of which is a liveness heuristic: the campaign
        volume holds more than the floor (a full disk kills every later session the
        same way), the canonical ledger directory is writable (a read-only ledger
        makes every terminal unrecordable), and the driver still holds its
        single-writer lock. Absence of a host FAILURE signal is never treated as
        evidence of host HEALTH -- these checks can only veto model-scoping, never
        substitute for the publication witness.

        Returns
        -------
        dict[str, Any]
            Structured check results with an overall ``passed`` verdict.
        """

        try:
            disk_free = shutil.disk_usage(self.paths.work_root).free
        except OSError:
            disk_free = -1
        disk_ok = disk_free >= _AUTHOR_LANE_DISK_FLOOR_BYTES
        ledger_dir = self.paths.ledgers.models.parent
        ledger_writable = os.access(ledger_dir, os.W_OK)
        lock = getattr(self, "_driver_lock", None)
        lock_held = bool(lock is not None and lock.held())
        return {
            "disk_free_bytes": disk_free,
            "disk_floor_bytes": _AUTHOR_LANE_DISK_FLOOR_BYTES,
            "disk_ok": disk_ok,
            "ledger_dir_writable": ledger_writable,
            "driver_lock_held": lock_held,
            "passed": disk_ok and ledger_writable and lock_held,
        }

    def _author_lane_drought(self) -> JsonObject:
        """Evaluate the W5 lane-health detector: a windowed publication drought.

        Windowed RATE, normalized by the configured width, never adjacency:
        the lane is in drought exactly when the observation window holds ZERO
        publications AND at least ``max(2, ceil(width / 2))`` DISTINCT
        unpublished deaths (source-unavailability deaths excluded -- content
        evidence is not lane evidence, the S arm). Two independent deaths
        5-10 s apart at width 12-18 are one member's weather each; the same
        two deaths at the serial width are still the historical lane pattern.

        Returns
        -------
        dict[str, Any]
            ``drought`` verdict plus the census it was derived from:
            ``width``, ``threshold``, ``window_seconds``,
            ``recent_publications``, ``recent_distinct_deaths``,
            ``machine_class_only``.
        """

        now = self.dependencies.monotonic()
        floor = now - _AUTHOR_LANE_DROUGHT_WINDOW_SECONDS
        with self._author_lane_state_lock:
            publications = list(self._author_lane_publications)
            deaths = list(self._author_no_publication_deaths)
        recent_publications = [
            entry
            for entry in publications
            if float(entry.get("at_monotonic", float("-inf"))) >= floor
        ]
        recent_deaths: dict[str, JsonObject] = {}
        for entry in deaths:
            if float(entry.get("at_monotonic", float("-inf"))) < floor:
                continue
            if entry.get("source_excluded"):
                continue
            recent_deaths[str(entry["stable_id"])] = entry
        width = max(1, int(self.config.author_concurrency))
        threshold = max(
            _AUTHOR_NO_PUBLICATION_STREAK_LIMIT, math.ceil(width / 2)
        )
        drought = not recent_publications and len(recent_deaths) >= threshold
        machine_class_only = all(
            entry.get("failure_class") != "content-refusal"
            for entry in recent_deaths.values()
        )
        return {
            "drought": drought,
            "width": width,
            "threshold": threshold,
            "window_seconds": _AUTHOR_LANE_DROUGHT_WINDOW_SECONDS,
            "recent_publications": len(recent_publications),
            "recent_distinct_deaths": sorted(recent_deaths),
            "machine_class_only": machine_class_only,
        }

    def _scope_author_retry_exhaustion(
        self,
        item: WorkItem,
        artifact: Optional[AuthorArtifact],
        exc: AuthorSessionRetryExhausted,
        reducer: Optional[CanonicalReducer],
        operational: JsonlLedger,
        state: JsonObject,
    ) -> bool:
        """Adjudicate one exhausted no-publication author death (W5 compose).

        The merged tri-lab containment law, replacing the rung-7 three-clause
        boundary whose "second consecutive DISTINCT model" clause killed
        leg 27 (seq 5821-5831: two members died 10 s apart at width 12-18
        with 41 publication witnesses, and the leg died with them):

        - F arm: host-checks-pass => MEMBER scope always. The containment
          terminal is a deferred-family park (``author-wave`` scheduler
          quarantine, bounded by W12's retry budget), never a machine
          ``failed:author`` without adjudication (OC-Q6).
        - O arm: the lane-health detector is :meth:`_author_lane_drought`
          (windowed rate, width-normalized), never death adjacency.
        - S arm: typed primary-implementation-unavailable deaths are excluded
          from lane evidence before the detector ever sees them.
        - Storm routing: a drought whose deaths are all machine-class routes
          to the DURABLE usage-pause lane (wake episode armed); a drought
          with content evidence, or a failed host check, stays campaign-
          scoped and LOUD.

        The decision is recorded on the operational ledger on every arm, so
        the durable record always shows which side fired and why.

        Parameters
        ----------
        item, artifact:
            Model whose sessions died, and its retained prior artifact when
            one exists.
        exc:
            Typed retry-exhaustion failure from the bounded ladder.
        reducer, operational, state:
            Locked canonical writers and scheduler state (the reducer is
            unused by the containment arms and may be ``None`` in rigs; it
            stays in the signature so every catch site keeps one calling
            convention).

        Returns
        -------
        bool
            ``True`` when the member was contained (author-wave park);
            ``False`` when the caller must re-raise campaign-scoped.

        Raises
        ------
        AuthorUsagePause
            When the drought carries only machine-class evidence: the pause
            is recorded durably with a scheduled wake before raising.
        """

        session_failure = _author_session_failure_from_chain(exc)
        failure_class = (
            str(session_failure.get("failure_class"))
            if session_failure is not None
            else "session-crashed"
        )
        source_excluded = _chain_carries_source_unavailability(exc)
        with self._author_lane_state_lock:
            self._author_no_publication_deaths[:] = [
                entry
                for entry in self._author_no_publication_deaths
                if entry.get("stable_id") != item.stable_id
            ]
            self._author_no_publication_deaths.append(
                {
                    "stable_id": item.stable_id,
                    "at": self.dependencies.clock(),
                    "at_monotonic": self.dependencies.monotonic(),
                    "failure_class": failure_class,
                    "source_excluded": source_excluded,
                }
            )
            del self._author_no_publication_deaths[:-_AUTHOR_LANE_WITNESS_CAP]
            witnesses = list(self._author_lane_publications)
        host_checks = self._author_lane_health_checks()
        drought = self._author_lane_drought()
        if not host_checks["passed"]:
            decision = "campaign-scoped"
        elif drought["drought"] and drought["machine_class_only"]:
            decision = "usage-pause-storm"
        elif drought["drought"]:
            decision = "campaign-scoped"
        else:
            decision = "member-contained"
        created_at = self.dependencies.clock()
        witness_summary = {
            "publications_this_run": len(witnesses),
            "last_publication": witnesses[-1] if witnesses else None,
            "drought": drought,
            "host_checks": host_checks,
            "failure_class": failure_class,
            "source_excluded": source_excluded,
            "retry_attempts": exc.attempts,
            "retry_backoff_seconds": list(exc.backoff_seconds),
        }
        operational.append(
            mint_operational_event(
                {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": created_at,
                    "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                    "status": (
                        OperationalEventStatus.HEALTHY.value
                        if decision == "member-contained"
                        else OperationalEventStatus.RETRYABLE_INFRASTRUCTURE.value
                    ),
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 1},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": {
                        "kind": "author-no-publication-boundary",
                        "stable_id": item.stable_id,
                        "work_id": item.active_work_id,
                        "decision": decision,
                        "message": str(exc),
                        **witness_summary,
                    },
                }
            )
        )
        if decision == "usage-pause-storm":
            signal = AuthorBackoffSignal(
                reason=AuthorPauseReason.RATE_LIMIT,
                retry_after_seconds=None,
                reset_at=None,
                response_excerpt=(
                    "author-lane publication drought (W5 storm shape): "
                    f"{len(drought['recent_distinct_deaths'])} distinct machine-class "
                    f"no-publication deaths and zero publications inside "
                    f"{int(drought['window_seconds'])}s at width {drought['width']}"
                ),
                provider="anthropic",
            )
            raise AuthorUsagePause(
                self._pause_for_usage(signal, operational, 1)
            ) from exc
        if decision != "member-contained":
            return False
        # T4-T4/T4-B5: the classified attempt row is EVIDENCE, not a verdict --
        # it persists the failure class, basis, bounded refusal excerpt, and
        # the exact session count the bounded ladder consumed, whatever the
        # scope decision. Rigs that adjudicate without a reducer skip it.
        if reducer is not None:
            reason_code = (
                "content-refusal" if failure_class == "content-refusal" else "session-crashed"
            )
            attempt = _driver_failure_attempt(
                item,
                artifact,
                "author",
                reason_code,
                exc,
                self.config,
                diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                environment=None,
                created_at=self.dependencies.clock(),
                stage_attempt=exc.attempts,
                session_failure=session_failure,
            )
            reducer.append_attempt(attempt)
        # OC-Q6: the containment terminal is a deferred-family park -- the
        # fault may be machinery's, so no model verdict exists. W12 bounds
        # the park's retries and PM-5's fuse catches an identical-cause loop.
        self._record_scheduler_quarantine(
            item,
            site="author-wave",
            reason_code="member-fault",
            fault=(
                f"{exc} [lane-health: {json.dumps(witness_summary, sort_keys=True)}]"
            ),
            operational=operational,
        )
        return True

    @staticmethod
    def _is_infrastructure_error(exc: Exception) -> bool:
        """Return whether an exception represents spawn or transport infrastructure.

        Parameters
        ----------
        exc:
            External lane exception.

        Returns
        -------
        bool
            Whether retrying the external process or transport is appropriate.
        """

        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, (OSError, subprocess.SubprocessError)):
                return True
            # Typed transient operator failures (R8): a declared-retryable operator
            # exit or a stalled author queue must be retried, never recorded as a
            # permanent model failure.
            if isinstance(current, RetryableOperatorError):
                return True
            if isinstance(current, DriverIntegrationError):
                message = str(current).lower()
                if message.startswith(
                    (
                        "author command failed",
                        "author source request failed",
                        "checker command failed",
                    )
                ):
                    return True
            current = current.__cause__ or current.__context__
        return False

    def _note_env_turn_deferred(
        self,
        intent_name: str,
        *,
        ready: int,
        threshold: int,
        waited_seconds: float,
    ) -> None:
        """Log one rate-limited line when the env-turn trigger defers an intent.

        Leg 23 (2026-08-21): four staged artifacts starved for four hours with
        zero worker leases and NOTHING recorded anywhere about why -- every
        starvation diagnosis this campaign was archaeology over ledgers that
        deliberately do not capture scheduler-trigger decisions. One stderr
        line per intent per ten minutes makes the trigger's arithmetic
        directly observable in the leg log without touching any ledger.
        """

        stamps = getattr(self, "_env_turn_deferral_stamps", None)
        if stamps is None:
            stamps = {}
            self._env_turn_deferral_stamps = stamps
        now = self.dependencies.monotonic()
        last = stamps.get(intent_name)
        if last is not None and now - last < 600.0:
            return
        stamps[intent_name] = now
        print(
            f"env-turn deferred: intent={intent_name} ready={ready} "
            f"threshold={threshold} waited={waited_seconds:.0f}s",
            file=sys.stderr,
            flush=True,
        )

    def _run_environment_work(
        self,
        work: Sequence[WorkItem],
        artifacts: dict[str, AuthorArtifact],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        award_run: bool,
        currentness_validation_only: bool = False,
        _setup_park_retry: bool = False,
    ) -> Optional[str]:
        """Run grouped environments and return a checker pause from mode repair.

        Parameters
        ----------
        work, artifacts, reducer, operational, state:
            Scheduled work, executable authority, and locked canonical state.
        award_run:
            Whether successful attempts may become canonical runs.
        currentness_validation_only:
            Validate complete runs through the live cache without executing or republishing them.
        _setup_park_retry:
            Internal recursion guard (BL-9c): the one bounded in-leg retry of
            intents whose setup park was cleared mid-leg never retries again.

        Returns
        -------
        str | None
            Checker pause reason, or ``None`` after all environments finish.
        """

        by_intent: dict[str, list[WorkItem]] = defaultdict(list)
        for item in work:
            by_intent[item.route.intent].append(item)
        # BL-9c: intents parked behind an intent-setup fact THIS pass, with
        # the parking event id and the exact parked members -- watched after
        # the pass for a mid-leg operator clearance so the lane restores
        # without a leg restart.
        parked_setup_retry: dict[str, tuple[str, list[WorkItem]]] = {}
        for intent_name in self._ordered_intents(by_intent):
            self._check_shutdown("environment-admission")
            intent = self.registry.intents[intent_name]
            use_entered = False
            use_completed = False
            observed_generation: Optional[str] = None
            observed_environment: Optional[EnvironmentBinding] = None
            completed_work: list[JsonObject] = []
            repair_pause: Optional[str] = None

            def cleanup_artifact_identity() -> str:
                """Return the current typed committed setup-artifact identity."""

                return _intent_setup_artifact_identity(intent)

            artifact_identity = cleanup_artifact_identity()
            artifact_state = _intent_setup_artifact_state(intent)
            canonical_events_path = canonical_operational_ledger_path(self.paths.ledgers.models)
            operational_events = scan_jsonl(canonical_events_path)
            quarantine = next(
                (
                    event
                    for event in reversed(operational_events)
                    if event.get("event_kind") == OperationalEventKind.CAMPAIGN_HEALTH.value
                    and event.get("details", {}).get("disposition")
                    == "environment-cleanup-quarantined"
                    and event.get("details", {}).get("intent") == intent.name
                    and event.get("details", {}).get("target") == intent.lock.target
                    and event.get("details", {}).get("artifact_identity") == artifact_identity
                ),
                None,
            )
            if quarantine is not None and _quarantine_is_cleared(
                operational_events, str(quarantine.get("event_id"))
            ):
                # An explicit operator clearance receipt supersedes the
                # quarantine: the intent builds normally again, and any
                # persisted park rows on this quarantine release through the
                # same receipt.
                quarantine = None
            if quarantine is not None:
                details = quarantine.get("details", {})
                quarantined_environment = _environment_from_quarantine(details)
                if quarantined_environment is not None:
                    generations = dict(reducer.context.environment_generations)
                    generations[intent.name] = quarantined_environment.env_generation
                    refreshed_context = replace(
                        reducer.context, environment_generations=generations
                    )
                    reducer.update_context(refreshed_context)
                    self._authority_context = refreshed_context
                completed = {
                    str(entry.get("stable_id")): entry
                    for entry in details.get("completed_work", [])
                    if isinstance(entry, Mapping)
                }
                gates = reducer.gate_records
                # ONE dependency-current projection, bound once, is both the
                # membership authority for every item below AND the replayed
                # policy-closure proof handed to the record-anchored fallback
                # (rung-11 review finding 7: the proof is a structural input,
                # never an ambient assumption).
                dependency_current = reducer.current_view
                unsatisfied: list[WorkItem] = []
                accepted_on_record_identity: list[JsonObject] = []
                for item in by_intent[intent_name]:
                    current = dependency_current.get(item.stable_id)
                    exact = completed.get(item.stable_id)
                    if not (
                        award_run
                        and current is not None
                        and quarantined_environment is not None
                        and exact is not None
                        and exact.get("record_revision") == current.get("record_revision")
                    ):
                        unsatisfied.append(item)
                        continue
                    live_identity = _quarantine_work_identity(item, artifacts[item.stable_id])
                    recorded_identity = exact.get("work_identity")
                    if recorded_identity == live_identity and _current_run_is_fresh(
                        current,
                        artifacts[item.stable_id],
                        quarantined_environment,
                        gates,
                        representative_model=(
                            dependency_current.get(item.family_representative_id)
                            if item.is_family_variant
                            else None
                        ),
                    ):
                        continue
                    # Record-anchored freshness (rung 11): the snapshot's full
                    # work identity folds in live code and work-generation
                    # lineage, so ANY hotfix between snapshot and resume voids
                    # every completed run it witnessed. When only those
                    # code-side components moved, every RECORD identity still
                    # matches -- proposal, sources, receipts, and the exact
                    # quarantined environment generation -- AND the record
                    # carries the replayed dependency-vector proof (it is
                    # dependency-current under the live authority context,
                    # which independently re-adjudicates every policy axis the
                    # record binds), the completed work is accepted, with the
                    # drift named in a typed operational event rather than
                    # silently.
                    moved = (
                        _quarantine_code_identity_drift(recorded_identity, live_identity)
                        if isinstance(recorded_identity, Mapping)
                        else None
                    )
                    if (
                        moved  # nonempty: pure record-level drift stays parked
                        and _record_anchored_run_is_current(
                            current,
                            exact,
                            quarantined_environment,
                            dependency_current=dependency_current,
                        )
                    ):
                        accepted_on_record_identity.append(
                            {
                                "stable_id": item.stable_id,
                                "record_revision": current.get("record_revision"),
                                "moved_identity_components": list(moved),
                            }
                        )
                        continue
                    unsatisfied.append(item)
                if accepted_on_record_identity:
                    acceptance_event: JsonObject = {
                        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                        "created_at": self.dependencies.clock(),
                        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                        "status": OperationalEventStatus.HEALTHY.value,
                        "provider": None,
                        "observed_response": None,
                        "reset_at": None,
                        "queued_work_counts": {"models": len(accepted_on_record_identity)},
                        "current_environment": intent.name,
                        "run_id": self.config.run_id,
                        "machine_id": self.config.machine_id,
                        "details": {
                            "disposition": (QUARANTINE_RECORD_ANCHORED_ACCEPTANCE_DISPOSITION),
                            "intent": intent.name,
                            "target": intent.lock.target,
                            "quarantine_event_id": quarantine.get("event_id"),
                            "env_generation": (
                                quarantined_environment.env_generation
                                if quarantined_environment is not None
                                else None
                            ),
                            "accepted": accepted_on_record_identity,
                        },
                    }
                    # This per-occurrence disclosure includes created_at; later
                    # observations append and only exact byte replays no-op.
                    acceptance_event = mint_operational_event(acceptance_event)
                    operational.append(acceptance_event)
                # Park, never terminalize (rung 11): the quarantine names an
                # infrastructure failure during CLEANUP -- no build ran for any
                # of these work identities, so a `failed:environment /
                # build-failed` terminal would assert a world-fact the run
                # never observed (and rung 11's fan-out destroyed eleven
                # verified run awards exactly this way). Each unsatisfied item
                # is recorded in the durable environment-rebuild deferral
                # ledger and claimed by the completion partition as
                # machine-deferred; recovery is an environment
                # recovery/rotation plus a re-drive, both named in the row.
                for item in unsatisfied:
                    self._record_environment_rebuild_deferral(
                        item,
                        intent=intent,
                        quarantine_event=quarantine,
                        quarantined_environment=quarantined_environment,
                        artifact_identity=artifact_identity,
                        artifact_state=artifact_state,
                        operational=operational,
                    )
                continue

            # BL-9c: an operator clearance of an effort-exhaustion park must
            # also restore the intent's durable turn budget BEFORE admission,
            # or the very next consume re-raises off the exhausted count and
            # re-parks the freshly cleared intent behind a new human-only
            # release. The reset is replay-idempotent per cleared park event,
            # so a genuine post-clearance failure is never erased by a later
            # leg re-observing the same receipt.
            latest_effort_park = self._latest_effort_exhaustion_park_event(
                intent, artifact_identity
            )
            if latest_effort_park is not None and latest_effort_park[1]:
                restore_budget = getattr(
                    self.dependencies.environments, "reset_effort_after_clearance", None
                )
                if restore_budget is not None:
                    restore_budget(intent, str(latest_effort_park[0].get("event_id")))

            def use(
                prefix: Path,
                probe_results: tuple[ProbeResult, ...],
                *,
                items: Sequence[WorkItem] = by_intent[intent_name],
            ) -> None:
                """Process one intent's models while its sole environment exists."""

                nonlocal observed_environment, observed_generation, repair_pause, use_entered
                nonlocal use_completed
                self.dependencies.boundary_hook("pre-environment-use", intent_name)
                self._check_shutdown("environment-use-admission")
                use_entered = True
                environment_lane = self.dependencies.environments
                live_supervised_environment = isinstance(
                    self.dependencies.forward, SupervisedForwardLane
                ) and isinstance(environment_lane, SequentialEnvironmentLifecycle)
                if live_supervised_environment:
                    assert isinstance(environment_lane, SequentialEnvironmentLifecycle)
                    authority_cache = environment_lane.active_authority_cache(prefix)
                    prior_authority = authority_cache.authority
                    try:
                        environment = bind_materialized_environment(
                            intent,
                            prefix,
                            probe_results,
                            authority_cache=authority_cache,
                            defer_currentness_validation=True,
                        )
                    except AuthorityDerivationError:
                        if prior_authority is not None:
                            observed_environment = _stale_environment_binding(
                                intent,
                                prefix,
                                probe_results,
                                authority=prior_authority,
                            )
                            observed_generation = observed_environment.env_generation
                        raise
                else:
                    environment = _environment_binding(
                        intent,
                        prefix,
                        probe_results,
                        strict=False,
                    )
                observed_environment = environment
                observed_generation = environment.env_generation
                refreshed_generations = dict(reducer.context.environment_generations)
                refreshed_generations[intent.name] = environment.env_generation
                refreshed_context = replace(
                    reducer.context,
                    environment_generations=refreshed_generations,
                )
                reducer.update_context(refreshed_context)
                self._authority_context = refreshed_context
                gates = reducer.gate_records
                authority = environment.environment_authority
                cache = environment.environment_authority_cache
                if live_supervised_environment and (authority is None or cache is None):
                    raise DriverIntegrationError(
                        "live currentness pass lacks lifecycle-owned environment authority"
                    )
                pass_context = (
                    cache.currentness_pass(authority)
                    if cache is not None and authority is not None
                    else nullcontext(None)
                )
                with pass_context as verification_token:
                    for item in items:
                        self.dependencies.boundary_hook("pre-model", item.stable_id)
                        self._check_shutdown("model-admission", item=item)
                        # BL-8 (a): the per-member blast-radius boundary. The
                        # env turn had NO per-member containment -- any raise
                        # on one member's facts (a malformed manifest, a
                        # member-scoped integration refusal) unwound the whole
                        # turn and killed the leg for every sibling sharing
                        # the environment. Campaign-level control flow and
                        # malformed-PROGRAM errors still pass through.
                        try:
                            current = reducer.current_view.get(item.stable_id)
                            closure = (
                                _collect_worker_executable_closure(
                                    artifacts[item.stable_id],
                                    environment,
                                    verification_token=verification_token,
                                )
                                if live_supervised_environment
                                else None
                            )
                            fresh = bool(
                                award_run
                                and current is not None
                                and _current_run_is_fresh(
                                    current,
                                    artifacts[item.stable_id],
                                    environment,
                                    gates,
                                    representative_model=(
                                        reducer.current_view.get(item.family_representative_id)
                                        if item.is_family_variant
                                        else None
                                    ),
                                    closure_identity=(
                                        closure.identity if closure is not None else None
                                    ),
                                    verification_token=verification_token,
                                )
                            )
                            if fresh or currentness_validation_only:
                                if current is None:
                                    continue
                                completed_work.append(
                                    {
                                        "stable_id": item.stable_id,
                                        "record_revision": current["record_revision"],
                                        "work_identity": _quarantine_work_identity(
                                            item, artifacts[item.stable_id]
                                        ),
                                    }
                                )
                                continue
                            repair_pause = self._forward_and_reduce(
                                item,
                                artifacts[item.stable_id],
                                environment,
                                reducer,
                                operational,
                                state,
                                award_run=award_run,
                                closure=closure,
                                verification_token=verification_token,
                            )
                            artifacts[item.stable_id] = self._family_artifacts.get(
                                item.stable_id, artifacts[item.stable_id]
                            )
                            current = reducer.current_view.get(item.stable_id)
                            if current is not None:
                                completed_work.append(
                                    {
                                        "stable_id": item.stable_id,
                                        "record_revision": current["record_revision"],
                                        "work_identity": _quarantine_work_identity(
                                            item, artifacts[item.stable_id]
                                        ),
                                    }
                                )
                        except MEMBER_CONTAINMENT_PASSTHROUGH:
                            raise
                        except Exception as exc:  # noqa: BLE001 -- BL-8 (a)
                            # Item 9: the closure derivation's own limit on
                            # vendored packages (RelativeImportUnbindableError,
                            # R6b m243) is machinery -> ``non-model-fault``.
                            self._record_scheduler_quarantine(
                                item,
                                site="env-turn",
                                reason_code=member_quarantine_reason(exc),
                                fault=exc,
                                operational=operational,
                            )
                            continue
                        if repair_pause is not None:
                            break
                use_completed = True

            environment_failure: Exception | None = None
            for environment_attempt in range(2):
                use_entered = False
                use_completed = False
                try:
                    self.dependencies.boundary_hook("pre-environment-create", intent_name)
                    self._check_shutdown("environment-create-admission")
                    self.dependencies.environments.run(intent, use=use)
                except DriverPaused:
                    raise
                except Exception as exc:  # noqa: BLE001 -- lifecycle phase decides ownership
                    if use_entered and isinstance(exc, AuthorityDerivationError):
                        observed_generation = (
                            observed_generation
                            or reducer.context.environment_generations.get(intent.name)
                        )
                        event = {
                            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                            "created_at": self.dependencies.clock(),
                            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                            "status": OperationalEventStatus.RUNNER_FAILED.value,
                            "provider": None,
                            "observed_response": None,
                            "reset_at": None,
                            "queued_work_counts": {"models": len(by_intent[intent_name])},
                            "current_environment": intent.name,
                            "run_id": self.config.run_id,
                            "machine_id": self.config.machine_id,
                            "details": {
                                "disposition": "environment-integrity-quarantined",
                                "intent": intent.name,
                                "target": intent.lock.target,
                                "artifact_identity": cleanup_artifact_identity(),
                                "env_generation": observed_generation,
                                "failure_type": (
                                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                                ),
                            },
                        }
                        # The typed identity covers the complete payload including
                        # created_at. A recurrence in a later environment cycle or
                        # resume appends a distinct occurrence; only a byte-identical
                        # replay no-ops instead of tripping the ledger's
                        # conflicting-replay tripwire and killing the resume
                        # (2026-08-03 incident: two same-identity quarantines with
                        # differing queued_work_counts crashed `resume`).
                        event = mint_operational_event(event)
                        operational.append(event)
                        environment_failure = exc
                        break
                    if use_completed:
                        cleanup_identity = cleanup_artifact_identity()
                        event = {
                            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                            "created_at": self.dependencies.clock(),
                            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                            "status": OperationalEventStatus.RUNNER_FAILED.value,
                            "provider": None,
                            "observed_response": None,
                            "reset_at": None,
                            "queued_work_counts": {"models": 0},
                            "current_environment": intent.name,
                            "run_id": self.config.run_id,
                            "machine_id": self.config.machine_id,
                            "details": {
                                "disposition": "environment-cleanup-quarantined",
                                "intent": intent.name,
                                "target": intent.lock.target,
                                "artifact_identity": cleanup_identity,
                                "artifact_state": _intent_setup_artifact_state(intent),
                                "env_generation": observed_generation,
                                "environment": _quarantine_environment_payload(
                                    observed_environment
                                ),
                                "completed_work": completed_work,
                                "completed_work_identity": stable_hash(completed_work),
                                "failure_type": (
                                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                                ),
                            },
                        }
                        # Same per-occurrence rule as the integrity quarantine:
                        # later observations append and exact byte replays no-op.
                        event = mint_operational_event(event)
                        operational.append(event)
                        environment_failure = None
                        break
                    if use_entered:
                        raise
                    environment_failure = exc
                    if environment_attempt == 0:
                        continue
                else:
                    environment_failure = None
                break
            if environment_failure is None:
                if repair_pause is not None:
                    return repair_pause
                continue
            pending = list(by_intent[intent_name])
            # The routing registry is CLOSED and deliberate (DD-1): an
            # exception with no row raises the loud typed refusal right here,
            # so an unadjudicated failure can never fan out into intent-wide
            # ``failed:environment / build-failed`` model terminals again
            # (T5-2 mutation-verified; T2-08's live award wipe was this arm).
            route = _environment_failure(environment_failure)
            if isinstance(environment_failure, DiskFloorError):
                # The env lane's disk floor refused (pre-write measurement or
                # observed ENOSPC): a MACHINE condition, never a fact about
                # any model (rung-12 census class A: six models terminalized
                # ``failed:resource/disk-floor`` from host disk state, one
                # burying its own honest solve-failed negative and two spending
                # review-held supersession grants on an infrastructure
                # artifact). Each affected work identity is parked in the
                # durable disk-floor deferral ledger and claimed by the
                # completion partition as machine-deferred; recovery is freeing
                # disk on this host or re-driving on any host with disk, both
                # named in the row. A model already holding a canonical
                # terminal keeps it -- the record wins over the park.
                for item in pending:
                    self._record_disk_floor_deferral(
                        item,
                        intent=intent,
                        failure=environment_failure,
                        operational=operational,
                    )
                continue
            if route.kind in ("park-intent-setup", "park-host-condition"):
                # Park, never terminalize (DD-1): a pre-use environment
                # failure is a fact about the SETUP or the HOST -- no model
                # ran, so a model terminal would assert a world-fact the run
                # never observed. Each pending work identity is recorded in
                # the durable environment-failure machine-deferral ledger and
                # claimed by the completion partition as machine-deferred.
                park_event_id = self._park_environment_setup_failure(
                    pending,
                    intent=intent,
                    failure=environment_failure,
                    route=route,
                    observed_generation=observed_generation,
                    operational=operational,
                )
                if route.kind == "park-intent-setup":
                    # BL-9c: watch this park for a mid-leg clearance receipt.
                    parked_setup_retry[intent_name] = (park_event_id, list(pending))
                continue
            if route.kind == "propagate":
                # A harness/integration defect is campaign-level: re-raise
                # loudly with the original type. It will recur identically for
                # every model after this one; silencing it into terminals or
                # parks would hide the defect behind per-model bookkeeping.
                raise environment_failure
            # route.kind == "terminal": the one deliberate typed model
            # disposition left in this arm (the supervisor's fail-closed
            # sandbox signal), routed with its explicit stage and reason.
            current_before_failure = reducer.current_view
            stage, reason = route.stage, route.reason_code
            assert stage is not None and reason is not None
            for item in pending:
                attempt = _driver_failure_attempt(
                    item,
                    artifacts[item.stable_id],
                    stage,
                    reason,
                    environment_failure,
                    self.config,
                    diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                    environment=observed_environment or intent.name,
                    created_at=self.dependencies.clock(),
                )
                persisted = reducer.append_attempt(attempt).record
                self._terminalize(
                    item,
                    artifacts[item.stable_id],
                    f"failed:{stage}",
                    reason,
                    str(environment_failure),
                    (persisted,),
                    reducer,
                    operational,
                    state,
                    superseded_model=current_before_failure.get(item.stable_id),
                )
        # BL-9c: ONE bounded in-leg recovery pass. An intent-setup park is now
        # clearable while the leg runs (receipts land in the operator sidecar
        # the running driver only reads); if a clearance arrived for a park
        # this pass created, restore the intent's turn budget, release the
        # run-scoped valve stash for its members, and re-run exactly those
        # members through the environment lane -- the lane recovers WITHOUT a
        # leg restart. The recursion is depth-one by the guard: a park minted
        # during the retry stands until the next leg.
        if parked_setup_retry and not _setup_park_retry:
            records_root = self.paths.ledgers.models.parent.parent
            canonical_path = canonical_operational_ledger_path(self.paths.ledgers.models)
            clearance_view = (
                *scan_jsonl(canonical_path),
                *environment_clearance_events(records_root),
            )
            retry_work: list[WorkItem] = []
            for intent_name, (park_event_id, parked_items) in parked_setup_retry.items():
                if not _quarantine_is_cleared(clearance_view, park_event_id):
                    continue
                restore_budget = getattr(
                    self.dependencies.environments, "reset_effort_after_clearance", None
                )
                if restore_budget is not None:
                    restore_budget(self.registry.intents[intent_name], park_event_id)
                stash = getattr(self, "_machine_deferral_rows", None)
                for parked in parked_items:
                    if stash is not None:
                        # The member is being re-attempted NOW: the valve no
                        # longer claims it for this run's completion partition
                        # (its durable row is already released by the receipt).
                        stash.pop(parked.stable_id, None)
                    retry_work.append(parked)
            if retry_work:
                return self._run_environment_work(
                    retry_work,
                    artifacts,
                    reducer,
                    operational,
                    state,
                    award_run=award_run,
                    currentness_validation_only=currentness_validation_only,
                    _setup_park_retry=True,
                )
        return None


@dataclass(frozen=True)
class RequeueGrantLedger:
    """Validated content of one append-only requeue grant ledger.

    Attributes
    ----------
    grants:
        Unique validated grants in append order.
    voids:
        Unique validated operator void events in append order. A void never
        deletes its grant; it only marks the grant as no longer consumable.
    """

    grants: tuple[JsonObject, ...]
    voids: tuple[JsonObject, ...]


def _grant_stage_binds_terminal(grant: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    """Return whether an outstanding grant's stage binds the terminal head.

    An ordinary grant binds by exact stage equality. The typed
    :data:`~menagerie.crawler.constants.DEFERRED_TERMINAL_GRANT_STAGE` sentinel
    binds EXACTLY the heads that have no failure stage to name -- those whose
    ``status.stage`` is null (docket T2-05) -- and nothing else, so no ordinary
    failure stage can launder through the sentinel and the sentinel can never
    bind a staged head.

    A null stage is not a "deferred" property: ``driver_models`` derives
    ``stage = code.split(":", 1)[1] if kind == "failed" else None``, so
    ``stage is None`` holds for the ``deferred``, ``skipped``, AND ``runs`` kinds
    alike, and only for those. C3-B narrowed this to ``kind == "deferred"`` at
    first and found a second unreachable class: the ``_terminalize`` containment
    ladder (minimal rung and ``_contain_terminal_finalization_refusal``) re-appends
    the ORIGINAL status code with ``human_review=True``, which is exactly how the
    live pilot's m4256 got a review-held ``deferred:needs-cuda`` head -- and the
    same two lines mint a review-held ``skipped:*`` head from a
    ``SkipRecommendation``. Keyed on the kind, no operator grant could ever bind
    that head: ordinary stages mismatch null, and the sentinel refused a
    non-deferred kind. Keying on the null stage itself is the total rule and
    leaves both narrowness guarantees intact.

    Parameters
    ----------
    grant:
        Validated outstanding grant row.
    record:
        Persisted supersession-lineage terminal head.

    Returns
    -------
    bool
        Whether consumption may bind this grant to this head.
    """

    status = record.get("status", {})
    if grant.get("stage") == DEFERRED_TERMINAL_GRANT_STAGE:
        return status.get("stage") is None
    return grant.get("stage") == status.get("stage")


def _validated_requeue_grants(
    path: Path,
    intake_ids: frozenset[str],
    *,
    prior_member_ids: frozenset[str] = frozenset(),
) -> RequeueGrantLedger:
    """Read and integrity-check the append-only human requeue grant ledger.

    Parameters
    ----------
    path:
        Runtime grant ledger written by either supported crawler requeue command.
    intake_ids:
        Exact trusted intake membership.
    prior_member_ids:
        Stable IDs from REGISTERED PRIOR same-campaign intake snapshots (the
        driver-read lineage union of
        :func:`menagerie.crawler.rung_staging.registered_prior_intake_members`).
        A grant for a prior member is DORMANT on a subset rung: it validates and
        is retained, but the caller never consumes it this leg. A stable ID in
        NEITHER set is still the hard integrity failure.

    Returns
    -------
    RequeueGrantLedger
        Unique validated grants and operator void events in append order.

    Raises
    ------
    DriverIntegrationError
        If any grant or void is malformed, forged, conflicting, or outside both
        the intake and the registered prior-snapshot lineage.
    """

    rows = scan_jsonl(path, validate=False)
    validated: list[JsonObject] = []
    by_id: dict[str, JsonObject] = {}
    voids: list[JsonObject] = []
    voids_by_id: dict[str, JsonObject] = {}
    common = {"grant_id", "stable_id", "stage", "reason", "attempts"}
    optional = {
        # AW-1's award-naming seam and WS-2's typed capability are canonical
        # builder fields folded into the grant identity; the admission field
        # contract must admit what the one sanctioned mint emits, or the
        # award-custody release lever is unusable end-to-end. W31's schema-v2
        # fields (adjudication_key, effective_session_ceiling) are optional at
        # the CONTRACT level because the era ladder is monotone -- v1/legacy
        # rows never carry them -- and REQUIRED per-row for the v2 rung below.
        "adjudication_key",
        "blocked_award_revision",
        "capability",
        "created_at",
        "effective_session_ceiling",
        "granted_by",
        "new_work_generation",
        "schema_version",
        "target_intent",
    }
    void_fields = {"void_id", "void_of", "reason", "voided_by", "created_at"}
    # T1-13 (voidability): membership refusals are COLLECTED, not raised in
    # place. The old single-pass raise fired on an out-of-intake grant row
    # before any void row was scanned (and a void scanned first referenced a
    # not-yet-seen grant), so a foreign grant was unvoidable in either ledger
    # order and bricked every future admission. Every OTHER integrity failure
    # still raises exactly where it always did; at the end of the pass an
    # out-of-membership grant raises the byte-identical error unless an
    # operator void neutralized it.
    membership_violations: dict[str, JsonObject] = {}
    for row in rows:
        if "void_of" in row:
            row_schema = row.get("schema_version")
            if row_schema is not None and row_schema != REQUEUE_GRANT_VOID_SCHEMA_VERSION:
                raise DriverIntegrationError(
                    f"requeue void carries a foreign schema identity: {row_schema}"
                )
            if set(row) - {"schema_version"} != void_fields:
                raise DriverIntegrationError("requeue void has an invalid field contract")
            if any(
                not isinstance(row.get(field), str) or not str(row.get(field)).strip()
                for field in void_fields
            ):
                raise DriverIntegrationError("requeue void values are invalid")
            void_id = str(row["void_id"])
            if void_id != requeue_void_identity(
                void_of=str(row["void_of"]),
                reason=str(row["reason"]),
                voided_by=str(row["voided_by"]),
            ):
                raise DriverIntegrationError(f"requeue void hash mismatch: {void_id}")
            if str(row["void_of"]) not in by_id:
                raise DriverIntegrationError(
                    f"requeue void references an unknown grant: {row['void_of']}"
                )
            prior_void = voids_by_id.get(void_id)
            if prior_void is not None:
                logical_prior = {
                    key: value for key, value in prior_void.items() if key != "created_at"
                }
                logical_current = {key: value for key, value in row.items() if key != "created_at"}
                if logical_prior != logical_current:
                    raise DriverIntegrationError(f"conflicting duplicate requeue void: {void_id}")
                continue
            normalized_void = dict(row)
            voids_by_id[void_id] = normalized_void
            voids.append(normalized_void)
            continue
        if set(row) - common - optional or not common <= set(row):
            raise DriverIntegrationError("requeue grant has an invalid field contract")
        row_schema = row.get("schema_version")
        if row_schema is not None and row_schema not in REQUEUE_GRANT_SCHEMA_LADDER:
            # Monotone ladder: every rung <= current reads; an unknown or
            # FUTURE version refuses typed rather than defaulting (W31).
            raise DriverIntegrationError(
                f"requeue grant carries a foreign schema identity: {row_schema}"
            )
        grant_id = row.get("grant_id")
        stable_id = row.get("stable_id")
        stage = row.get("stage")
        reason = row.get("reason")
        attempts = row.get("attempts")
        target_intent = row.get("target_intent")
        blocked_award_revision = row.get("blocked_award_revision")
        grant_capability = row.get("capability")
        if (
            not isinstance(grant_id, str)
            or not isinstance(stable_id, str)
            or not isinstance(stage, str)
            or (stage not in FAILURE_REASON_CODES and stage != DEFERRED_TERMINAL_GRANT_STAGE)
            or not isinstance(reason, str)
            or not reason.strip()
            or not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or attempts < 1
            or (
                target_intent is not None
                and (not isinstance(target_intent, str) or not target_intent)
            )
            or (
                blocked_award_revision is not None
                and (not isinstance(blocked_award_revision, str) or not blocked_award_revision)
            )
            or (grant_capability is not None and grant_capability not in GRANT_CAPABILITIES)
        ):
            raise DriverIntegrationError("requeue grant values are invalid")
        out_of_membership = stable_id not in intake_ids and stable_id not in prior_member_ids
        if "new_work_generation" in row:
            generation = row.get("new_work_generation")
            granted_by = row.get("granted_by")
            if (
                not isinstance(generation, int)
                or generation < 1
                or not isinstance(granted_by, str)
                or not granted_by
            ):
                raise DriverIntegrationError("requeue tool grant generation is invalid")
        elif not isinstance(row.get("created_at"), str):
            raise DriverIntegrationError("crawler requeue grant has no creation time")
        try:
            expected = expected_grant_id(row)
        except GrantIdentityError as exc:
            raise DriverIntegrationError(str(exc)) from exc
        if grant_id != expected:
            raise DriverIntegrationError(f"requeue grant hash mismatch: {grant_id}")
        if row_schema == REQUEUE_GRANT_SCHEMA_VERSION:
            # W31 v2 rung, checked AFTER the grant-id preimage so tampered
            # operator facts keep their historical hash-mismatch refusal:
            # created_at, a RECOMPUTED adjudication key (nothing a producer
            # writes may steer the decision identity), and the MC-Q4 ceiling
            # disclosure. The ceiling is shape-checked, never recomputed
            # against the live factor: the row discloses the ceiling as of
            # ISSUANCE, and a later factor change must not brick history.
            created_stamp = row.get("created_at")
            if not isinstance(created_stamp, str) or not created_stamp.strip():
                raise DriverIntegrationError("requeue grant v2 requires created_at")
            expected_key = requeue_adjudication_key(
                stable_id=stable_id,
                stage=stage,
                reason=reason,
                target_intent=target_intent,
                blocked_award_revision=blocked_award_revision,
                capability=grant_capability,
            )
            if row.get("adjudication_key") != expected_key:
                raise DriverIntegrationError(
                    f"requeue grant adjudication key mismatch: {grant_id}"
                )
            ceiling = row.get("effective_session_ceiling")
            if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 1:
                raise DriverIntegrationError(
                    "requeue grant v2 requires a positive effective_session_ceiling"
                )
        elif "adjudication_key" in row or "effective_session_ceiling" in row:
            # A pre-v2 row cannot smuggle v2 authority fields: the v1 schema
            # closed its vocabulary, and an unstamped row that carries a key
            # nothing recomputes would be an unauthenticated identity claim.
            raise DriverIntegrationError(
                "requeue grant carries schema-v2 fields without the v2 identity"
            )
        normalized = dict(row)
        prior = by_id.get(grant_id)
        if prior is not None:
            logical_prior = {key: value for key, value in prior.items() if key != "created_at"}
            logical_current = {
                key: value for key, value in normalized.items() if key != "created_at"
            }
            if logical_prior != logical_current:
                raise DriverIntegrationError(f"conflicting duplicate requeue grant: {grant_id}")
            continue
        by_id[grant_id] = normalized
        if out_of_membership:
            membership_violations[grant_id] = normalized
        validated.append(normalized)
    voided_of = {str(void["void_of"]) for void in voids}
    if set(membership_violations) - voided_of:
        # Byte-identical with the historical composite refusal so operators and
        # tests keep one greppable message for the membership tripwire. A
        # VOIDED out-of-membership grant is retained as inert history instead:
        # it stays replayable for any recorded consumption, is filtered from
        # `unconsumed` by its void, and is dormant-skipped by the grouping loop.
        raise DriverIntegrationError("requeue grant values are invalid")
    return RequeueGrantLedger(grants=tuple(validated), voids=tuple(voids))


def _environment_binding(
    intent: EnvironmentIntent,
    prefix: Path,
    probe_results: Sequence[ProbeResult],
    *,
    strict: bool,
    authority_cache: Optional[EnvironmentAuthorityCache] = None,
    defer_currentness_validation: bool = False,
) -> EnvironmentBinding:
    """Bind exact lifecycle, probe, package, and interpreter observations.

    Parameters
    ----------
    intent, prefix, probe_results:
        Declared environment, materialized prefix, and observed probe results.
    strict:
        Whether all production provenance and selected-interpreter facts are mandatory.
    authority_cache:
        Lifecycle-owned collector required for strict binding. ``None`` never constructs a cache.
    defer_currentness_validation:
        Whether a matching active authority will be validated by the immediately enclosing
        cache-owned scheduling/currentness pass instead of by this binding call.

    Returns
    -------
    EnvironmentBinding
        Exact observed environment binding.
    """

    fallback_artifact_hash = hash_bytes(b"test-environment-artifact")
    fallback_url = "https://example.test/test-environment.conda"
    fallback_lock = f"{fallback_url}#{fallback_artifact_hash.removeprefix('sha256:')}\n".encode()
    fallback_export = (
        canonical_json_bytes(
            {
                "packages": [
                    {
                        "name": "test-environment",
                        "version": "1",
                        "build": "test_0",
                        "url": fallback_url,
                        "sha256": fallback_artifact_hash,
                    }
                ]
            }
        )
        + b"\n"
    )
    lock_bytes = _required_artifact_bytes(
        intent.lock.lock_path, fallback_lock, strict=strict, label="lock"
    )
    export_bytes = _required_artifact_bytes(
        intent.lock.export_path, fallback_export, strict=strict, label="resolved export"
    )
    package_bytes = _installed_package_manifest_bytes(prefix, strict=strict)
    if not strict and package_bytes == b"test-packages":
        package_bytes = fallback_export
    if strict:
        try:
            parse_exact_lock(lock_bytes)
            declared_packages = parse_resolved_export(export_bytes)
        except EnvironmentExactnessError as exc:
            raise DriverIntegrationError(str(exc)) from exc
        if declared_packages != package_bytes:
            raise DriverIntegrationError(
                "created-prefix packages do not match the declared resolved export"
            )
        export_hash_path = intent.lock.export_hash_path
        try:
            declared_export_hash = export_hash_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise DriverIntegrationError(
                f"environment resolved-export digest is missing: {export_hash_path}"
            ) from exc
        if declared_export_hash != hash_bytes(export_bytes):
            raise DriverIntegrationError("environment resolved-export digest is stale")
    interpreter = prefix / "bin" / "python"
    if strict and not interpreter.is_file():
        raise DriverIntegrationError(f"environment interpreter is missing: {interpreter}")
    if not interpreter.is_file():
        interpreter = Path(sys.executable)
    python_version, compiler_identity, sdk_identity, _interpreter_facts = (
        _observed_interpreter_facts(interpreter)
    )
    lock_sha256 = hash_bytes(lock_bytes)
    export_sha256 = hash_bytes(export_bytes)
    packages_sha256 = hash_bytes(package_bytes)
    try:
        observed_probes = validate_probe_receipts(intent.probes, probe_results)
        if strict:
            receipt_path = intent.lock.lock_path.with_name(f"{intent.lock.target}.probes.json")
            durable_probes = parse_probe_receipt_bytes(
                intent.probes,
                _required_artifact_bytes(receipt_path, b"", strict=True, label="probe receipt"),
            )
            if durable_probes != observed_probes:
                raise DriverIntegrationError(
                    "lifecycle probe receipts differ from the committed receipt artifact"
                )
        base_generation = materialized_environment_generation(
            intent,
            lock_bytes=lock_bytes,
            export_bytes=export_bytes,
            package_bytes=package_bytes,
            python_version=python_version,
            compiler_identity=compiler_identity,
            sdk_identity=sdk_identity,
            probe_results=observed_probes,
        )
    except (EnvironmentExactnessError, EnvironmentProbeError) as exc:
        raise DriverIntegrationError(str(exc)) from exc
    authority: Optional[EnvironmentAuthorityV1] = None
    generation = base_generation
    if strict:
        if authority_cache is None:
            raise DriverIntegrationError(
                "strict environment binding requires the active lifecycle authority cache"
            )
        authority = authority_cache.bind(
            prefix=prefix,
            selected_interpreter=interpreter,
            base_environment_generation=base_generation,
            validate_active=not defer_currentness_validation,
        )
        generation = authority.environment_generation
        interpreter = authority.selected_interpreter
    return EnvironmentBinding(
        prefix=prefix.resolve(),
        python_executable=interpreter.absolute(),
        family=intent.name,
        target=intent.lock.target,
        env_generation=generation,
        lock_sha256=lock_sha256,
        resolved_export_sha256=export_sha256,
        packages_manifest_sha256=packages_sha256,
        python_version=python_version,
        compiler_identity=compiler_identity,
        sdk_identity=sdk_identity,
        authority_epoch=(authority.authority_version if authority is not None else None),
        base_environment_generation=(
            authority.base_environment_generation if authority is not None else None
        ),
        environment_content_sha256=(
            authority.content_manifest_sha256 if authority is not None else None
        ),
        environment_authority_id=(authority.authority_id if authority is not None else None),
        selected_interpreter_relative_path=(
            authority.selected_interpreter_relative_path if authority is not None else None
        ),
        selected_interpreter_digest=(
            authority.selected_interpreter_digest if authority is not None else None
        ),
        external_escape_records=(authority.external_targets if authority is not None else ()),
        environment_authority=authority,
        environment_authority_cache=authority_cache,
    )


def bind_materialized_environment(
    intent: EnvironmentIntent,
    prefix: Path,
    probe_results: Sequence[ProbeResult],
    *,
    authority_cache: EnvironmentAuthorityCache,
    defer_currentness_validation: bool = False,
) -> EnvironmentBinding:
    """Strictly bind lifecycle provenance and a complete prefix content seal.

    Parameters
    ----------
    intent, prefix, probe_results:
        Exact committed lifecycle contract, materialized prefix, and durable probes.
    authority_cache:
        Sole cache owned by the active lifecycle prefix.
    defer_currentness_validation:
        Whether the driver will immediately validate the returned authority with one
        currentness-pass token shared by every scheduled model.

    Returns
    -------
    EnvironmentBinding
        Sole live binding accepted by supervised model execution.
    """

    return _environment_binding(
        intent,
        prefix,
        probe_results,
        strict=True,
        authority_cache=authority_cache,
        defer_currentness_validation=defer_currentness_validation,
    )


def _stale_environment_binding(
    intent: EnvironmentIntent,
    prefix: Path,
    probe_results: Sequence[ProbeResult],
    *,
    authority: EnvironmentAuthorityV1,
) -> EnvironmentBinding:
    """Retain invalidated authority facts solely for honest quarantine proof.

    Parameters
    ----------
    intent, prefix, probe_results:
        Lifecycle contract and observations that preceded the failed cache validation.
    authority:
        Exact previously active authority invalidated by the changed prefix.

    Returns
    -------
    EnvironmentBinding
        Non-active binding carrying the stale generation into failure attempts.
    """

    base = _environment_binding(intent, prefix, probe_results, strict=False)
    if base.env_generation != authority.base_environment_generation:
        raise DriverIntegrationError(
            "invalidated authority base generation contradicts current lifecycle facts"
        )
    return replace(
        base,
        prefix=authority.prefix,
        python_executable=authority.selected_interpreter,
        env_generation=authority.environment_generation,
        authority_epoch=authority.authority_version,
        base_environment_generation=authority.base_environment_generation,
        environment_content_sha256=authority.content_manifest_sha256,
        environment_authority_id=authority.authority_id,
        selected_interpreter_relative_path=authority.selected_interpreter_relative_path,
        selected_interpreter_digest=authority.selected_interpreter_digest,
        external_escape_records=authority.external_targets,
    )


def _observed_interpreter_facts(interpreter: Path) -> tuple[str, str, str, bytes]:
    """Execute an environment interpreter and return its exact platform facts.

    Parameters
    ----------
    interpreter:
        Exact active-prefix Python executable.

    Returns
    -------
    tuple[str, str, str, bytes]
        Python, compiler, SDK strings and the exact stdout bytes hashed by the generation.

    Raises
    ------
    DriverIntegrationError
        If the interpreter cannot report a complete canonical fact object.
    """

    program = (
        "import json, platform, sys, sysconfig; "
        "print(json.dumps({"
        "'python_version': sys.version, "
        "'compiler_identity': platform.python_compiler(), "
        "'sdk_identity': json.dumps({"
        "'platform': sysconfig.get_platform(), "
        "'platform_detail': platform.platform(), "
        "'sdkroot': sysconfig.get_config_var('SDKROOT'), "
        "'deployment_target': sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET'), "
        "'cc': sysconfig.get_config_var('CC'), "
        "'cxx': sysconfig.get_config_var('CXX')}, "
        "sort_keys=True, separators=(',', ':'))}, "
        "sort_keys=True, separators=(',', ':')))"
    )
    # BL-8 (e): this exec of the just-built prefix's own python was the one
    # remaining UNBOUNDED wait on the env-turn path (BL-9e bounded every
    # wrapper action, but not this driver-side probe) -- a hung interpreter
    # (dangling store symlink, unmounted volume, blocking sitecustomize)
    # pinned the writer thread, every ledger flock, and the completion
    # partition forever. It reuses BL-9e's parent-owned deadline table
    # (the ``probe`` row) rather than growing a second clock.
    deadline = CommandEnvironmentBackend._ACTION_DEADLINE_SECONDS["probe"]  # noqa: SLF001
    try:
        completed = subprocess.run(
            [str(interpreter.absolute()), "-c", program],
            check=False,
            capture_output=True,
            timeout=deadline,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise EnvironmentBackendUnavailableError(
            f"environment interpreter facts for {interpreter} exceeded the "
            f"parent-owned {deadline:g}s probe deadline and were killed; "
            "treating as retryable infrastructure"
        ) from exc
    except OSError as exc:
        raise DriverIntegrationError(
            f"environment interpreter facts failed for {interpreter}: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise DriverIntegrationError(
            "environment interpreter facts failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-1500:]
        )
    try:
        value = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriverIntegrationError("environment interpreter facts are invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise DriverIntegrationError("environment interpreter facts must be an object")
    facts = tuple(
        value.get(field) for field in ("python_version", "compiler_identity", "sdk_identity")
    )
    if not all(isinstance(fact, str) and fact for fact in facts):
        raise DriverIntegrationError("environment interpreter facts are incomplete")
    return str(facts[0]), str(facts[1]), str(facts[2]), completed.stdout


def _required_artifact_bytes(path: Path, fallback: bytes, *, strict: bool, label: str) -> bytes:
    """Read nonempty lifecycle bytes, allowing explicit test-lane fallbacks only."""

    if path.is_file():
        value = path.read_bytes()
        if value:
            return value
    if strict:
        raise DriverIntegrationError(f"environment {label} artifact is missing or empty: {path}")
    return fallback


def _installed_package_manifest_bytes(prefix: Path, *, strict: bool) -> bytes:
    """Return deterministic bytes derived only from actual installed package metadata."""

    try:
        return installed_package_inventory_bytes(prefix)
    except EnvironmentExactnessError as exc:
        if strict:
            raise DriverIntegrationError(str(exc)) from exc
    return b"test-packages"


def _source_symbol_bytes(
    path: Path,
    qualified_name: str,
    *,
    source: Optional[str] = None,
    tree: Optional[ast.Module] = None,
) -> bytes:
    """Return normalized semantic AST bytes for one source-level binding.

    Parameters
    ----------
    path:
        Python source file.
    qualified_name:
        Top-level function name or ``Class.method`` path.
    source, tree:
        Optional once-read/once-parsed module state for compositional batches.

    Returns
    -------
    bytes
        Stable AST bytes with docstrings and source locations omitted.

    Raises
    ------
    DriverIntegrationError
        If the requested award binding cannot be located exactly.
    """

    source_text = path.read_text(encoding="utf-8") if source is None else source
    module_tree = ast.parse(source_text, filename=str(path)) if tree is None else tree
    parts = qualified_name.split(".")
    body: Sequence[ast.stmt] = module_tree.body
    found: Optional[ast.stmt] = None
    for part in parts:
        found = next(
            (
                node
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if found is None:
            if len(parts) == 1:
                break
            raise DriverIntegrationError(
                f"award-closure source symbol is missing: {path.name}:{qualified_name}"
            )
        body = found.body
    if found is None:
        if len(parts) != 1:
            raise DriverIntegrationError(
                f"award-closure source binding is missing: {path.name}:{qualified_name}"
            )
        for node in module_tree.body:
            targets: Sequence[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = (node.target,)
            else:
                continue
            if any(
                isinstance(target, ast.Name) and target.id == qualified_name for target in targets
            ):
                found = node
                break
    if found is None:
        raise DriverIntegrationError(
            f"award-closure source binding is missing: {path.name}:{qualified_name}"
        )
    semantic = deepcopy(found)
    for descendant in ast.walk(semantic):
        descendant_body = getattr(descendant, "body", None)
        if (
            isinstance(descendant_body, list)
            and descendant_body
            and isinstance(descendant_body[0], ast.Expr)
            and isinstance(descendant_body[0].value, ast.Constant)
            and isinstance(descendant_body[0].value.value, str)
        ):
            del descendant_body[0]
    return ast.dump(semantic, annotate_fields=True, include_attributes=False).encode("utf-8")


def _award_closure_components(
    source_items: tuple[tuple[str, bytes], ...],
    closure_symbols: Mapping[str, tuple[str, ...]],
) -> dict[str, str]:
    """Walk the semantic award closure and hash every reachable definition.

    Parameters
    ----------
    source_items:
        Module-relative names and exact source bytes.
    closure_symbols:
        Seed symbols per module-relative name.

    Returns
    -------
    dict[str, str]
        Component hash per ``relative:symbol`` definition.

    Raises
    ------
    DriverIntegrationError
        If a reachable symbol has no definition in any tracked module.
    """

    root = Path(__file__).parent
    components: dict[str, str] = {}
    source_by_relative = {
        relative: source_bytes.decode("utf-8") for relative, source_bytes in source_items
    }
    module_trees: dict[str, ast.Module] = {}
    module_definitions: dict[str, dict[str, ast.stmt]] = {}
    module_imports: dict[str, dict[str, tuple[str, str]]] = {}
    for relative, source in source_by_relative.items():
        path = root / relative
        tree = ast.parse(source, filename=str(path))
        definitions: dict[str, ast.stmt] = {}
        imports: dict[str, tuple[str, str]] = {}
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[node.name] = node
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            definitions[f"{node.name}.{child.name}"] = child
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if isinstance(target, ast.Name):
                        definitions[target.id] = node
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                prefix = "menagerie.crawler."
                if node.module.startswith(prefix):
                    imported_relative = f"{node.module.removeprefix(prefix).replace('.', '/')}.py"
                    if imported_relative in source_by_relative:
                        for imported_name in node.names:
                            imports[imported_name.asname or imported_name.name] = (
                                imported_relative,
                                imported_name.name,
                            )
            elif isinstance(node, ast.Import):
                prefix = "menagerie.crawler."
                for imported_name in node.names:
                    if not imported_name.name.startswith(prefix):
                        continue
                    imported_relative = (
                        f"{imported_name.name.removeprefix(prefix).replace('.', '/')}.py"
                    )
                    if imported_relative in source_by_relative:
                        imports[imported_name.asname or imported_name.name.rsplit(".", 1)[-1]] = (
                            imported_relative,
                            "",
                        )
        module_trees[relative] = tree
        module_definitions[relative] = definitions
        module_imports[relative] = imports

    pending = [
        (relative, symbol)
        for relative, symbols in closure_symbols.items()
        for symbol in symbols
    ]
    visited: set[str] = set()
    while pending:
        relative, symbol = pending.pop()
        component = f"{relative}:{symbol}"
        if component in components or component in visited:
            continue
        visited.add(component)
        relative_definitions = module_definitions.get(relative)
        if relative_definitions is None or symbol not in relative_definitions:
            # A same-package re-export is a binding, not a definition: follow
            # the ImportFrom edge to the defining module so the REAL body is
            # what gets hashed (DX-3 moved the completion-line protocol behind
            # an authority.py re-export). Anything unresolvable stays a typed
            # fail-closed refusal, and the visited set bounds alias cycles.
            reexport = (
                module_imports.get(relative, {}).get(symbol) if "." not in symbol else None
            )
            if reexport is not None and reexport[1]:
                pending.append(reexport)
                continue
            raise DriverIntegrationError(f"award-closure source symbol is missing: {component}")
        definition = relative_definitions[symbol]
        components[component] = hash_bytes(
            _source_symbol_bytes(
                root / relative,
                symbol,
                source=source_by_relative[relative],
                tree=module_trees[relative],
            )
        )
        class_name = symbol.split(".", 1)[0] if "." in symbol else None
        for descendant in ast.walk(definition):
            if isinstance(descendant, ast.Name) and isinstance(descendant.ctx, ast.Load):
                if descendant.id in relative_definitions:
                    pending.append((relative, descendant.id))
                    continue
                imported = module_imports[relative].get(descendant.id)
                if imported is not None and imported[1]:
                    pending.append(imported)
            elif (
                class_name is not None
                and isinstance(descendant, ast.Attribute)
                and isinstance(descendant.value, ast.Name)
                and descendant.value.id in {"self", "cls"}
            ):
                method = f"{class_name}.{descendant.attr}"
                if method in relative_definitions:
                    pending.append((relative, method))
            elif (
                isinstance(descendant, ast.Attribute)
                and isinstance(descendant.value, ast.Name)
                and descendant.value.id in module_imports[relative]
            ):
                imported_relative, imported_symbol = module_imports[relative][descendant.value.id]
                if not imported_symbol:
                    pending.append((imported_relative, descendant.attr))
    return components


@lru_cache(maxsize=8)
def _award_closure_from_bytes(
    source_items: tuple[tuple[str, bytes], ...],
    schema_items: tuple[tuple[str, bytes], ...],
) -> str:
    """Hash the transitive semantic award closure from one byte snapshot.

    Parameters
    ----------
    source_items:
        Module-relative names and exact source bytes.
    schema_items:
        Schema-relative names and exact bytes.

    Returns
    -------
    str
        Compositional award-closure identity.
    """

    components = _award_closure_components(source_items, _AWARD_CLOSURE_SYMBOLS)
    for relative, schema_bytes in schema_items:
        components[relative] = hash_bytes(schema_bytes)
    return stable_hash(components)


def _award_closure_identity() -> str:
    """Hash only the parent/reducer code and schemas that decide run awards.

    Returns
    -------
    str
        Compositional award-closure identity kept separate from child runtime.
    """

    root = Path(__file__).parent
    source_items = tuple(
        (path.relative_to(root).as_posix(), path.read_bytes()) for path in sorted(root.glob("*.py"))
    )
    schema_items = tuple(
        (relative, (root / relative).read_bytes()) for relative in _AWARD_CLOSURE_SCHEMAS
    )
    return _award_closure_from_bytes(source_items, schema_items)


def _runner_identity(modality: object = None, *, platform_name: Optional[str] = None) -> str:
    """Hash transitive runtime behavior plus the exact selected input asset.

    Parameters
    ----------
    modality:
        Accepted modality string or sequence used to select the only bundled
        asset that can participate in this execution.
    platform_name:
        Execution-host platform. Historical replay passes the recorded host OS;
        live execution defaults to the reviewing process platform.

    Returns
    -------
    str
        Compositional execution-closure identity.
    """

    selected_platform = platform_name or sys.platform
    root = Path(__file__).parent
    source_texts = {
        path.name: path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py"))
    }
    selected_asset = expected_standard_asset(modality)
    cache_key = stable_hash(
        {
            "platform": selected_platform,
            "sources": {
                relative: hash_bytes(source.encode("utf-8"))
                for relative, source in source_texts.items()
            },
            "selected_asset": (
                {
                    "asset_id": selected_asset["asset_id"],
                    "sha256": selected_asset["sha256"],
                }
                if selected_asset is not None
                else None
            ),
        }
    )
    cached = _RUNNER_IDENTITY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    module_trees: dict[str, ast.Module] = {}
    module_definitions: dict[str, dict[str, ast.stmt]] = {}
    module_imports: dict[str, dict[str, tuple[str, str]]] = {}
    pending = [
        (relative, symbol)
        for relative, symbols in _RUNNER_COMMON_EXECUTION_CLOSURE.items()
        for symbol in symbols
    ]
    components: dict[str, str] = {}
    visited: set[str] = set()
    while pending:
        relative, symbol = pending.pop()
        component = f"{relative}:{symbol}"
        if component in components or component in visited:
            continue
        visited.add(component)
        path = root / relative
        if relative not in module_trees:
            source = source_texts.get(relative)
            if source is None:
                source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            definitions: dict[str, ast.stmt] = {}
            imports: dict[str, tuple[str, str]] = {}
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions[node.name] = node
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                    for target in targets:
                        if isinstance(target, ast.Name):
                            definitions[target.id] = node
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    prefix = "menagerie.crawler."
                    if node.module.startswith(prefix):
                        imported_relative = (
                            f"{node.module.removeprefix(prefix).replace('.', '/')}.py"
                        )
                        if (root / imported_relative).is_file():
                            for imported_name in node.names:
                                imports[imported_name.asname or imported_name.name] = (
                                    imported_relative,
                                    imported_name.name,
                                )
            module_trees[relative] = tree
            module_definitions[relative] = definitions
            module_imports[relative] = imports
        definition = module_definitions[relative].get(symbol)
        if definition is None:
            # Same-package re-exports resolve to their defining module (see
            # _award_closure_components); unresolvable symbols stay typed
            # refusals and the visited set bounds alias cycles.
            reexport = (
                module_imports[relative].get(symbol) if "." not in symbol else None
            )
            if reexport is not None and reexport[1]:
                pending.append(reexport)
                continue
            raise DriverIntegrationError(
                f"runner-closure source symbol is missing: {relative}:{symbol}"
            )
        semantic = deepcopy(definition)
        for descendant in ast.walk(semantic):
            body = getattr(descendant, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                del body[0]
        components[component] = hash_bytes(
            ast.dump(semantic, annotate_fields=True, include_attributes=False).encode("utf-8")
        )
        loaded_names = {
            node.id
            for node in ast.walk(definition)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        for name in loaded_names:
            if name in module_definitions[relative]:
                pending.append((relative, name))
                continue
            resolved_import = module_imports[relative].get(name)
            if resolved_import is None:
                continue
            imported_relative, imported_symbol = resolved_import
            lowered = imported_symbol.lower()
            if selected_platform.startswith("linux") and (
                "macos" in lowered or "sandbox_exec" in lowered
            ):
                continue
            if selected_platform == "darwin" and ("linux" in lowered or "bubblewrap" in lowered):
                continue
            pending.append((imported_relative, imported_symbol))
    if selected_asset is not None:
        components["selected_standard_asset"] = stable_hash(
            {
                "asset_id": selected_asset["asset_id"],
                "sha256": selected_asset["sha256"],
            }
        )
    identity = derive_runner_identity(
        components,
        platform_name=selected_platform,
        selected_asset_identity=(components.get("selected_standard_asset") or "not-applicable"),
    )
    if len(_RUNNER_IDENTITY_CACHE) >= 16:
        _RUNNER_IDENTITY_CACHE.clear()
    _RUNNER_IDENTITY_CACHE[cache_key] = identity
    return identity


def _checker_prompt_hash() -> str:
    """Hash the exact current frozen checker prompt bytes."""

    path = Path(__file__).with_name("prompts") / f"{CHECKER_PROMPT_NAME}.txt"
    try:
        return hash_bytes(path.read_bytes())
    except OSError as exc:
        raise DriverIntegrationError(f"checker prompt bytes are unavailable: {exc}") from exc


def _validate_trusted_intake_identity(facts: Mapping[str, Any], item: WorkItem) -> None:
    """Reject proposed facts whose identity contradicts trusted intake.

    ``$.identity.variant``, ``$.identity.variant_scope``, and
    ``$.identity.family_representative_id`` are ``trusted-intake`` leaves. No
    author lane owns them, so an author-proposed value that differs from the one
    derived from the trusted roster is a contract violation, not a fact.

    Parameters
    ----------
    facts:
        Proposed or accepted canonical fact block.
    item:
        Scheduled work item carrying its trusted intake row.

    Raises
    ------
    DriverIntegrationError
        If the identity block is absent or any trusted leaf differs.
    """

    identity = facts.get("identity")
    if not isinstance(identity, Mapping):
        raise DriverIntegrationError("author proposal has no identity object")
    try:
        expected = trusted_identity_fields(item.intake)
    except IntakeError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    mismatches = trusted_intake_identity_mismatches(
        identity,
        expected,
        intake_name=item.intake.name,
        intake_zoo=item.intake.zoo,
    )
    if mismatches:
        raise DriverIntegrationError(f"identity contradicts trusted intake: {mismatches}")


def _identity_basis_rotation(recorded_basis: RecordedIdentityBasis) -> tuple[str, ...]:
    """Name the prompt-world axes that moved since one artifact's recorded basis.

    Reuse eligibility under a MOVED prompt world mirrors the reducer's own
    model-write rule (``_validate_ordinary_metadata`` requires gate-vet and
    model-vet to equal the vet recomputed under the LIVE checker prompt): a
    proposal's embedded, self-hashed vet identity can never reduce under a
    rotated checker world, and an author block binding a retired author prompt
    cannot stage as fresh work. This is a mechanical pre-check of the
    reducer's stated requirement -- NOT a second freshness authority; record
    currency remains the dependency-currency projection's sole decision. A
    nonempty result means the cached artifact is consumed as a cache miss and
    the model re-authors under the live world.

    Parameters
    ----------
    recorded_basis:
        The prompt world one finalized artifact was accepted under.

    Returns
    -------
    tuple[str, ...]
        Human-readable descriptions of every moved axis; empty when the
        recorded basis still matches the live world.
    """

    rotated: list[str] = []
    if recorded_basis.checker_prompt_sha256 != DependencyState.NOT_APPLICABLE.value:
        # A terminal-claim artifact finalizes BEFORE any checker gate exists, so
        # its vector pins checker_prompt_identity to the exact ``not-applicable``
        # sentinel: there is no recorded checker world to rotate, and no embedded
        # vet identity bound to one. ``validate_currency`` applies the same
        # closed rule (the sentinel is never a stale checker_prompt_identity);
        # comparing the sentinel byte-wise against the live prompt hash read
        # every claim-bearing terminal head as permanently rotated, so the
        # backlog sweep miscensused it ``no_consumer`` and its owed review
        # could never be bought.
        live_checker_prompt = leg_checker_prompt_sha256()
        if recorded_basis.checker_prompt_sha256 != live_checker_prompt:
            rotated.append(
                f"checker_prompt {recorded_basis.checker_prompt_sha256} -> {live_checker_prompt}"
            )
    live_author_prompt = _live_author_prompt_hash()
    if recorded_basis.author_prompt_sha256 != live_author_prompt:
        rotated.append(
            f"author_prompt {recorded_basis.author_prompt_sha256} -> {live_author_prompt}"
        )
    return tuple(rotated)


def _live_author_prompt_hash() -> str:
    """Hash the exact current frozen author prompt bytes.

    Returns
    -------
    str
        Canonical digest of the live author prompt file.

    Raises
    ------
    DriverIntegrationError
        If the prompt bytes are unavailable.
    """

    prompt_path = Path(__file__).with_name("prompts") / "claude_crawler_author_v2.txt"
    try:
        return hash_bytes(prompt_path.read_bytes())
    except OSError as exc:
        raise DriverIntegrationError(f"author prompt bytes are unavailable: {exc}") from exc


def _validate_artifact_identities(
    artifact: AuthorArtifact,
    config: DriverConfig,
    *,
    item: WorkItem,
    recorded_basis: Optional[RecordedIdentityBasis] = None,
) -> None:
    """Reject an author artifact whose claimed identities do not match accepted facts.

    ``recorded_basis`` selects the identity world the derivation checks run in.
    ``None`` (every FRESH path: authoring, repairs, the disposable cache, the
    uncommitted-result preserve) validates against the LIVE frozen prompts --
    a fresh result must bind the current world, and a mismatch is a hard
    per-model failure. A rehydrated finalized artifact instead passes its
    transaction's own record-time pins, so history proves internal consistency
    at the basis it was accepted under (history-vs-live doctrine); whether that
    history is REUSABLE stays the reducer dependency-currency projection's sole
    decision. Rung-12 leg 2: validating a rehydrated artifact at the LIVE
    checker prompt made every pre-rotation vet identity mismatch and crashed
    the campaign at the first one.
    """

    proposal = artifact.proposal
    if recorded_basis is not None:
        expected_author_prompt = recorded_basis.author_prompt_sha256
        prompt_error = "author proposal does not bind its recorded author prompt identity"
    else:
        expected_author_prompt = _live_author_prompt_hash()
        prompt_error = "author proposal does not bind the current frozen prompt"
    author = proposal.get("author")
    if not isinstance(author, Mapping) or author.get("prompt_sha256") != expected_author_prompt:
        raise DriverIntegrationError(prompt_error)
    facts = proposal.get("proposed_facts")
    if not isinstance(facts, Mapping):
        raise DriverIntegrationError("author proposal has no proposed_facts object")
    _validate_trusted_intake_identity(facts, item)
    implementation = facts.get("implementation")
    if isinstance(implementation, Mapping):
        code_value = implementation.get("code_path")
        if isinstance(code_value, str) and Path(code_value).is_absolute():
            raise DriverIntegrationError(
                "absolute code path requires a fresh repository-relative proposal"
            )
        patches = implementation.get("patches", [])
        if not isinstance(patches, list) or any(
            not isinstance(patch, Mapping)
            or not isinstance(patch.get("path"), str)
            or Path(str(patch["path"])).is_absolute()
            for patch in patches
        ):
            raise DriverIntegrationError(
                "absolute/invalid patch path requires a fresh repository-relative proposal"
            )
        _verify_model_code_manifest(implementation, artifact.model_dir, proposal)
    try:
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=(
                recorded_basis.checker_prompt_sha256
                if recorded_basis is not None
                else leg_checker_prompt_sha256()
            ),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
            schema_version=MODEL_SCHEMA_VERSION_V3,
        )
    except MetadataValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    claimed = {
        "source_identity": identities.source,
        "evidence_identity": identities.evidence,
        "recipe_revision": identities.recipe,
        "vet_identity": identities.vet,
        "fidelity_identity": identities.fidelity,
    }
    mismatches = {
        field: {"claimed": proposal.get(field), "computed": value}
        for field, value in claimed.items()
        if proposal.get(field) != value
    }
    if mismatches:
        raise DriverIntegrationError(f"author proposal identity mismatch: {mismatches}")
    implementation = facts.get("implementation")
    evidence = facts.get("evidence")
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("recipe_revision") != identities.recipe
        or not isinstance(evidence, Mapping)
        or evidence.get("evidence_identity") != identities.evidence
    ):
        raise DriverIntegrationError("embedded recipe/evidence identities are stale")
    # ``verified_hashes.source_manifest`` must equal the proposal's own
    # ``source_manifest_identity``: the machine hands the author that identity as
    # a request binding (``_PROPOSAL_BINDING_KEYS``), the terminal lane derives
    # the pair equal (``driver_models._terminal_checker_item``), and
    # ``artifact_transactions`` already refuses a mismatch -- but only at
    # publication-authorization time, long after the checker has gated the
    # proposal on it. Checking it here is the same refusal, moved to the first
    # point the machine can make it, so a disagreeing proposal never reaches a
    # checker as though it were bound.
    #
    # NOTE: ``verified_hashes.evidence`` and ``.source_to_code_map`` are NOT
    # checked here, and deliberately. Both are author's-word-only today: no
    # production code produces a proposal's ``verified_hashes``, and the author
    # prompt never states the rule, so enforcing an equality the authoring
    # contract does not specify would refuse honestly authored work. Binding them
    # needs a spec decision (stamp them machine-side, or state the derivation in
    # the prompt), not a silent guard.
    verified_hashes = proposal.get("verified_hashes")
    if not isinstance(verified_hashes, Mapping):
        raise DriverIntegrationError("author proposal has no verified_hashes object")
    if verified_hashes.get("source_manifest") != proposal.get("source_manifest_identity"):
        raise DriverIntegrationError(
            "verified_hashes.source_manifest does not bind the proposal source manifest identity"
        )
    expected_proposal_hash = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    if proposal.get("proposal_sha256") != expected_proposal_hash:
        raise DriverIntegrationError("proposal_sha256 does not bind the complete proposal")


def _verify_model_code_manifest(
    implementation: Mapping[str, Any], model_dir: Path, proposal: Mapping[str, Any]
) -> None:
    """Verify the accepted recursive code manifest against current model bytes.

    Parameters
    ----------
    implementation:
        Accepted implementation facts.
    model_dir:
        Current staged or canonical model root.
    proposal:
        Complete proposal carrying the gate-facing verified hashes.

    Raises
    ------
    DriverIntegrationError
        If any member, path, digest, or closure edge is stale.
    """

    code_value = implementation.get("code_path")
    if not isinstance(code_value, str):
        if implementation.get("code_manifest") is not None:
            raise DriverIntegrationError("declarative implementation carries model-code members")
        return
    manifest = implementation.get("code_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise DriverIntegrationError("typed implementation lacks a closed model-code manifest")
    code_path = Path(code_value)
    if code_path.is_absolute():
        raise DriverIntegrationError("model-code manifest refuses an absolute entry point")
    try:
        observed = [dict(row) for row in model_code_manifest(model_dir / code_path, model_dir)]
    except ProposalValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    if observed != manifest:
        raise DriverIntegrationError("accepted model-code import closure changed")
    verified = proposal.get("verified_hashes")
    if (
        not isinstance(verified, Mapping)
        or verified.get("code") != implementation.get("code_sha256")
        or verified.get("code_manifest") != stable_hash(observed)
    ):
        raise DriverIntegrationError("verified hashes do not bind the model-code entry and closure")


def _routed_environment_packages(
    registry: Optional[EnvironmentRegistry], intent_name: Optional[str]
) -> tuple[Mapping[str, Any], ...]:
    """Return the routed intent's exact resolved-export package rows.

    The rows are a static, pre-gate artifact of the very lock that materializes
    the environment (``env_lifecycle._require_lock_inventory_match`` keeps the
    lock and the created prefix one-to-one), so resolving a distribution digest
    from them is equivalent to resolving it from the live prefix while remaining
    available before the environment stage runs.

    Parameters
    ----------
    registry:
        Loaded environment registry, when one is available.
    intent_name:
        Routed environment intent for this work item.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Exact package rows, or an empty tuple when the intent is unknown or its
        target is not locked. An empty inventory degrades to an honest null
        digest; it never aborts admission.
    """

    if registry is None or not isinstance(intent_name, str):
        return ()
    intent = registry.intents.get(intent_name)
    if intent is None or intent.lock.export_bytes is None:
        return ()
    if intent.lock.status != "locked":
        # Lock artifacts are the environment lane's own solve residue, rewritten
        # on every run. A residue whose resolved export no longer satisfies the
        # intent's CURRENT declaration (`superseded`) proves the spec moved after
        # the solve; reading it as this run's inventory would refuse every model
        # needing the newly declared package against an environment the next
        # solve will not produce. Unknown is honest; a stale certainty is not.
        return ()
    try:
        value = json.loads(parse_resolved_export(intent.lock.export_bytes))
    except (EnvironmentExactnessError, UnicodeDecodeError, json.JSONDecodeError):
        return ()
    packages = value.get("packages") if isinstance(value, Mapping) else None
    if not isinstance(packages, list):
        return ()
    rows = tuple(row for row in packages if isinstance(row, Mapping))
    # A pinned pip overlay is part of the SAME exact runtime the lock realizes
    # (installed offline from hash-verified wheels between create and probe),
    # so its `name`/`version`/`sha256` rows join the routed inventory: coverage
    # can serve overlay-provided distributions, the author disclosure names
    # them, and R1 digest resolution derives the wheel digest. An unpinned,
    # invalid, or superseded overlay contributes nothing (unknown is honest).
    return rows + overlay_package_rows(intent)


def _normalize_artifact_modes(
    artifact: AuthorArtifact,
    config: DriverConfig,
    *,
    environment_packages: Sequence[Mapping[str, Any]] = (),
) -> AuthorArtifact:
    """Canonicalize modes, model code, and the derived R1 artifact digest.

    Parameters
    ----------
    artifact:
        Validated author artifact whose mutable proposal has not entered a gate.
    config:
        Exact checker identity participating in vet and fidelity identities.
    environment_packages:
        Exact package rows for the routed environment. The R1 artifact digest
        identifies the installed distribution, which the author stage cannot
        derive, so the driver resolves it here -- before the gate, while the
        recipe bytes and their dependent identities can still rebind together.

    Returns
    -------
    AuthorArtifact
        Copy with both mode declarations ordered identically, the derived
        artifact digest bound, and all dependent identities rebound to those
        canonical bytes.
    """

    proposal = deepcopy(artifact.proposal)
    facts = proposal.get("proposed_facts")
    if not isinstance(facts, dict):
        raise DriverIntegrationError("author proposal has no mutable proposed_facts object")
    modes = facts.get("modes")
    external = facts.get("external_metadata")
    external_modes = external.get("modes") if isinstance(external, dict) else None
    if not isinstance(modes, dict) or not isinstance(external_modes, dict):
        raise DriverIntegrationError("proposal meaningful-mode declarations are incomplete")
    try:
        canonical = canonical_meaningful_modes(
            modes.get("meaningful_modes"), field="modes.meaningful_modes"
        )
        external_canonical = canonical_meaningful_modes(
            external_modes.get("meaningful_modes"),
            field="external_metadata.modes.meaningful_modes",
        )
    except MetadataValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    if canonical != external_canonical:
        raise DriverIntegrationError("proposal meaningful-mode declarations disagree")
    changed = bool(
        modes.get("meaningful_modes") != canonical
        or external_modes.get("meaningful_modes") != canonical
    )
    modes["meaningful_modes"] = canonical
    external_modes["meaningful_modes"] = canonical
    code_changed = _bind_model_code_manifest(proposal, artifact.model_dir)
    implementation_block = facts.get("implementation")
    if isinstance(implementation_block, dict):
        try:
            recipe_changed = bind_library_artifact_digest(
                implementation_block, environment_packages
            )
        except RecipeError as exc:
            raise AuthorEnvironmentBindingError(str(exc)) from exc
    else:
        recipe_changed = False
    if not changed and not code_changed and not recipe_changed:
        return artifact
    try:
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=leg_checker_prompt_sha256(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
            schema_version=MODEL_SCHEMA_VERSION_V3,
        )
        implementation = facts.get("implementation")
        evidence = facts.get("evidence")
        if not isinstance(implementation, dict) or not isinstance(evidence, dict):
            raise DriverIntegrationError("proposal implementation/evidence facts are incomplete")
        implementation["recipe_revision"] = identities.recipe
        evidence["evidence_identity"] = identities.evidence
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=leg_checker_prompt_sha256(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
            schema_version=MODEL_SCHEMA_VERSION_V3,
        )
    except MetadataValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    proposal.update(
        {
            "source_identity": identities.source,
            "evidence_identity": identities.evidence,
            "recipe_revision": identities.recipe,
            "vet_identity": identities.vet,
            "fidelity_identity": identities.fidelity,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    author_result = artifact.author_result
    if not isinstance(author_result, ProposedAuthorResult):
        raise DriverIntegrationError("legacy fidelity policy requires a proposed author result")
    raw_result = deepcopy(author_result.binding.raw_result)
    raw_result["payload"] = {"arm": "PROPOSED", "proposal": proposal}
    raw_result["result_sha256"] = stable_hash(
        {key: value for key, value in raw_result.items() if key != "result_sha256"}
    )
    rebound = ProposedAuthorResult(
        replace(
            author_result.binding,
            result_sha256=str(raw_result["result_sha256"]),
            raw_result=raw_result,
        ),
        proposal,
        author_result.validation_report,
    )
    return replace(artifact, author_result=rebound)


def _bind_model_code_manifest(proposal: JsonObject, model_dir: Path) -> bool:
    """Bind every recursively imported model-local module into proposal identities.

    Parameters
    ----------
    proposal:
        Mutable author proposal before any checker gate exists.
    model_dir:
        Model-local root containing the accepted adapter or port.

    Returns
    -------
    bool
        Whether the proposal changed.

    Raises
    ------
    DriverIntegrationError
        If the declared entry point or its recursive closure is invalid.
    """

    facts = proposal.get("proposed_facts")
    implementation = facts.get("implementation") if isinstance(facts, Mapping) else None
    if not isinstance(implementation, dict):
        raise DriverIntegrationError("proposal implementation is incomplete")
    code_value = implementation.get("code_path")
    if not isinstance(code_value, str):
        verified_hashes = proposal.get("verified_hashes")
        verified_has_manifest = isinstance(verified_hashes, dict) and (
            "code_manifest" in verified_hashes
        )
        if "code_manifest" not in implementation and not verified_has_manifest:
            return False
        implementation.pop("code_manifest")
        if isinstance(verified_hashes, dict):
            verified_hashes.pop("code_manifest", None)
        return True
    code_path = Path(code_value)
    if code_path.is_absolute():
        raise DriverIntegrationError("model-code manifest refuses an absolute entry point")
    try:
        manifest = [dict(row) for row in model_code_manifest(model_dir / code_path, model_dir)]
    except ProposalValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    if not manifest:
        raise DriverIntegrationError("typed model code has an empty import closure")
    main_digest = next(
        (row["sha256"] for row in manifest if row["path"] == code_path.as_posix()), None
    )
    if main_digest != implementation.get("code_sha256"):
        raise DriverIntegrationError("model-code entry digest disagrees with the proposal")
    manifest_digest = stable_hash(manifest)
    verified_hashes = proposal.get("verified_hashes")
    if not isinstance(verified_hashes, dict):
        raise DriverIntegrationError("proposal verified_hashes is incomplete")
    changed = bool(
        implementation.get("code_manifest") != manifest
        or verified_hashes.get("code") != main_digest
        or verified_hashes.get("code_manifest") != manifest_digest
    )
    implementation["code_manifest"] = manifest
    verified_hashes["code"] = main_digest
    verified_hashes["code_manifest"] = manifest_digest
    return changed


def _execution_identity(
    artifact: AuthorArtifact,
    environment: EnvironmentBinding,
    *,
    closure_identity: Optional[str] = None,
    host_os: Optional[str] = None,
    machine_class: Optional[str] = None,
) -> str:
    """Compute execution identity from runtime dependencies and execution-host facts.

    Parameters
    ----------
    artifact, environment:
        Exact executable artifact and committed environment generation.
    closure_identity:
        Optional already-collected executable closure identity.
    host_os, machine_class:
        OS and architecture of the host that executed the attempt. Live calls
        default to the current host; historical replay supplies recorded facts.

    Returns
    -------
    str
        Exact execution identity.
    """

    proposal = artifact.proposal
    effective_closure_identity = (
        closure_identity or _collect_worker_executable_closure(artifact, environment).identity
    )
    facts = proposal["proposed_facts"]
    implementation = facts["implementation"]
    external = facts.get("external_metadata")
    modality = external.get("modality") if isinstance(external, Mapping) else None
    runtime_dependencies_identity = stable_hash(
        {
            "source_identity": proposal.get("source_identity"),
            "evidence_identity": proposal.get("evidence_identity"),
            "recipe_revision": proposal.get("recipe_revision"),
            "implementation": implementation,
            "source_resolution": facts.get("source_resolution"),
            "evidence": facts.get("evidence"),
            "input_contract": facts.get("input_contract"),
            "modes": facts.get("modes"),
            "verified_hashes": proposal.get("verified_hashes"),
            "author_prompt": proposal.get("author", {}).get("prompt_sha256"),
            "checker_prompt": leg_checker_prompt_sha256(),
            "vet_identity": proposal.get("vet_identity"),
            "fidelity_identity": proposal.get("fidelity_identity"),
            "executable_closure_identity": effective_closure_identity,
        }
    )
    return derive_execution_identity(
        stable_id=str(proposal["stable_id"]),
        recipe_revision=str(proposal["recipe_revision"]),
        environment_generation=environment.env_generation,
        runner_identity=_runner_identity(modality, platform_name=host_os),
        target=environment.target,
        machine_class=machine_class or platform.machine(),
        input_seed=int(facts.get("input_contract", {}).get("seed", 0)),
        framework=str(implementation["run_framework"]),
        recipe_type=str(implementation["recipe_type"]),
        award_closure_identity=_current_award_closure_identity(),
        runtime_dependencies_identity=runtime_dependencies_identity,
        # Defense in depth for the DEVICE-POLICY-PROSE class: the proposal gate
        # refuses prose at acceptance, and the driver's own consumption refuses
        # it again typed, naming the field -- str() forwarded the authored
        # sentence verbatim into the worker's torch device (step-100 rank 3).
        device=parse_device_policy(implementation["device_policy"]),
    )


def _current_run_is_fresh(
    model: Mapping[str, Any],
    artifact: AuthorArtifact,
    environment: EnvironmentBinding,
    gates: Sequence[Mapping[str, Any]],
    *,
    representative_model: Optional[Mapping[str, Any]] = None,
    closure_identity: Optional[str] = None,
    verification_token: Optional[EnvironmentVerificationToken] = None,
) -> bool:
    """Return whether a current run still binds all independently current inputs.

    Parameters
    ----------
    model, artifact, environment, gates:
        Current canonical run and exact live dependencies.
    representative_model:
        Current family representative for a templated size variant.
    closure_identity:
        Optional executable closure already collected once for this artifact in the pass.
    verification_token:
        Optional cache-created currentness-pass proof shared by every model.

    Returns
    -------
    bool
        Whether no canonical rewrite or execution is required.
    """

    authority = environment.environment_authority
    cache = environment.environment_authority_cache
    if (authority is None) != (cache is None):
        return False
    if authority is not None and cache is not None:
        try:
            cache.verify(authority, verification_token=verification_token)
        except AuthorityDerivationError:
            return False
    if model.get("status", {}).get("kind") != "runs":
        return False
    proposal = artifact.proposal
    prompt_path = Path(__file__).with_name("prompts") / "claude_crawler_author_v2.txt"
    try:
        live_author_prompt = hash_bytes(prompt_path.read_bytes())
    except OSError as exc:
        raise DriverIntegrationError(f"author prompt bytes are unavailable: {exc}") from exc
    if proposal.get("author", {}).get("prompt_sha256") != live_author_prompt:
        return False
    facts = proposal.get("proposed_facts", {})
    effective_closure_identity = (
        closure_identity
        if closure_identity is not None
        else (
            None
            if environment.environment_authority is not None
            else _INJECTED_FORWARD_CLOSURE_IDENTITY
        )
    )
    if model.get("authored_metadata_state") == "pending":
        untrusted = model.get("untrusted_attempt")
        retained_proposal = untrusted.get("proposal") if isinstance(untrusted, Mapping) else None
        pending_metadata_gate = _find_gate(
            gates, str(proposal["stable_id"]), "metadata_batch", proposal
        )
        execution = model.get("execution", {})
        return bool(
            pending_metadata_gate is None
            and retained_proposal == proposal
            and facts.get("source_resolution", {}).get("rung") in {"R1_LIBRARY", "R2_VENDOR"}
            and not _fidelity_required(proposal)
            and execution.get("current")
            and execution.get("env_generation") == environment.env_generation
            and execution.get("execution_identity")
            == _execution_identity(
                artifact,
                environment,
                closure_identity=effective_closure_identity,
            )
            and model.get("provenance", {}).get("author_prompt_sha256") == live_author_prompt
        )
    if artifact.template_source_revision is not None:
        if (
            not _usable_family_representative(
                representative_model,
                str(model.get("identity", {}).get("family_representative_id")),
            )
            or representative_model is None
            or representative_model.get("record_revision") != artifact.template_source_revision
        ):
            return False
        try:
            validate_size_variant(
                representative_model,
                model,
                str(model.get("identity", {}).get("family_representative_id")),
                parameter_count_total=model.get("observed", {}).get("parameter_count_total"),
                input_contract=model.get("input_contract", {}),
            )
        except FamilyTemplateError:
            return False
        for field in ("identity", "implementation", "input_contract"):
            if model.get(field) != facts.get(field):
                return False
        accuracy = model.get("accuracy_gate", {})
        representative_accuracy = representative_model.get("accuracy_gate", {})
        metadata_gate = next(
            (
                gate
                for gate in gates
                if gate.get("gate_id") == representative_accuracy.get("gate_id")
            ),
            None,
        )
        if (
            metadata_gate is None
            or accuracy.get("gate_id") != representative_accuracy.get("gate_id")
            or accuracy.get("vet_identity") != representative_accuracy.get("vet_identity")
            or metadata_gate.get("checker", {}).get("prompt_sha256")
            != leg_checker_prompt_sha256()
        ):
            return False
        if _fidelity_required(proposal):
            fidelity_gate = _find_gate(gates, str(proposal["stable_id"]), "fidelity", proposal)
            fidelity = model.get("fidelity", {})
            if (
                fidelity_gate is None
                or fidelity.get("gate_id") != fidelity_gate.get("gate_id")
                or fidelity.get("fidelity_identity") != proposal.get("fidelity_identity")
            ):
                return False
        execution = model.get("execution", {})
        return bool(
            execution.get("current")
            and execution.get("env_generation") == environment.env_generation
            and execution.get("execution_identity")
            == _execution_identity(
                artifact,
                environment,
                closure_identity=effective_closure_identity,
            )
            and model.get("provenance", {}).get("author_prompt_sha256") == live_author_prompt
        )
    for field in (
        "identity",
        "taxonomy",
        "external_metadata",
        "website",
        "people_and_origin",
        "dates",
        "citation",
        "licenses",
        "source_resolution",
        "evidence",
        "implementation",
        "input_contract",
    ):
        if model.get(field) != facts.get(field):
            return False
    metadata_gate = _find_gate(gates, str(proposal["stable_id"]), "metadata_batch", proposal)
    if metadata_gate is None:
        return False
    accuracy = model.get("accuracy_gate", {})
    if accuracy.get("gate_id") != metadata_gate.get("gate_id") or accuracy.get(
        "vet_identity"
    ) != proposal.get("vet_identity"):
        return False
    if _fidelity_required(proposal):
        fidelity_gate = _find_gate(gates, str(proposal["stable_id"]), "fidelity", proposal)
        fidelity = model.get("fidelity", {})
        if (
            fidelity_gate is None
            or fidelity.get("gate_id") != fidelity_gate.get("gate_id")
            or fidelity.get("fidelity_identity") != proposal.get("fidelity_identity")
        ):
            return False
    execution = model.get("execution", {})
    return bool(
        execution.get("current")
        and execution.get("env_generation") == environment.env_generation
        and execution.get("execution_identity")
        == _execution_identity(
            artifact,
            environment,
            closure_identity=effective_closure_identity,
        )
        and model.get("provenance", {}).get("author_prompt_sha256") == live_author_prompt
    )


def _canonical_crawler_root(paths: DriverPaths) -> Path:
    """Return the canonical ``menagerie/crawler`` root for driver ledgers.

    Parameters
    ----------
    paths:
        Bound driver paths.

    Returns
    -------
    pathlib.Path
        Canonical crawler root containing the records directory.
    """

    candidate = paths.ledgers.models.resolve().parent
    while candidate.name != "records" and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.name != "records":
        raise DriverIntegrationError("canonical model ledger is not below a records root")
    return candidate.parent


def _artifact_with_final_authority_objects(
    artifact: AuthorArtifact,
    transaction: ArtifactTransactionProjection,
) -> AuthorArtifact:
    """Return ``artifact`` with the finalized artifact object inventory attached.

    Parameters
    ----------
    artifact:
        Cached or rehydrated author result carrying its staged-private custody root.
    transaction:
        Verified final transaction projection whose object inventory includes
        retained private custody and any published public objects.

    Returns
    -------
    AuthorArtifact
        Equivalent artifact whose staged transaction exposes the complete final
        object inventory for retained-authority comparison.

    Raises
    ------
    DriverIntegrationError
        If the staged-private custody no longer matches the finalized transaction.
    """

    if artifact.staged is None:
        raise DriverIntegrationError("retained-artifact-authority-mismatch")
    private_objects = tuple(
        obj for obj in transaction.objects if obj.mirror_class == MirrorClass.PRIVATE.value
    )
    if tuple(sorted(private_objects, key=lambda obj: str(obj.object_id))) != tuple(
        sorted(artifact.staged.objects, key=lambda obj: str(obj.object_id))
    ):
        raise DriverIntegrationError("retained-artifact-authority-mismatch")
    return replace(
        artifact,
        staged=replace(artifact.staged, objects=transaction.objects),
    )


def _assert_persisted_handoff_authority_available(
    paths: DriverPaths,
    only_status: Optional[str],
) -> None:
    """Reject selected legacy deferrals before reducer projection can hide them.

    Parameters
    ----------
    paths:
        Canonical model and artifact ledger paths for the active campaign.
    only_status:
        Closed deferred selector, or ``None`` outside re-validation mode.

    Raises
    ------
    DriverIntegrationError
        If an exact selected PLATFORM-deferred model names a finalized transaction whose
        persisted executable handoff proposal fields are absent.

    Notes
    -----
    Scoped to the platform handoff codes on purpose. A ``needs-cuda`` re-run consumes a
    published handoff bundle, so a selected model without one is a real integrity hole.
    A host-capability re-run consumes nothing of the kind -- it re-executes the model
    from its own recipe on a machine with more memory, and by construction it never
    published anything (the valve fires before the environment lane). Demanding a
    finalized transaction of it would refuse ``--only-status
    deferred:needs-more-memory`` for exactly the cohort that flag now exists to route:
    the unroutable-head bug, one layer down.
    """

    if only_status is None:
        return
    selected_statuses = _only_status_selection(only_status) & _HANDOFF_AUTHORITY_STATUS_CODES
    if not selected_statuses:
        return
    latest: dict[str, Mapping[str, Any]] = {}
    for persisted_model in scan_jsonl(paths.ledgers.models):
        latest[str(persisted_model["stable_id"])] = persisted_model
    artifact_paths = tuple(sorted(paths.ledgers.artifacts.parent.glob("*.jsonl"))) or (
        paths.ledgers.artifacts,
    )
    final_events = tuple(
        event
        for artifact_path in artifact_paths
        for event in scan_jsonl(artifact_path)
        if event.get("event_kind")
        in {ArtifactEventKind.PUBLISHED.value, ArtifactEventKind.PRIVATE_COMMITTED.value}
    )
    for model in latest.values():
        status = model.get("status")
        if not isinstance(status, Mapping) or status.get("code") not in selected_statuses:
            continue
        authority = model.get("artifact_authority")
        transaction_id = authority.get("transaction_id") if isinstance(authority, Mapping) else None
        if not isinstance(transaction_id, str) or not transaction_id:
            raise DriverIntegrationError("handoff-authority-unavailable")
        matching = tuple(
            event
            for event in final_events
            if event.get("stable_id") == model.get("stable_id")
            and event.get("transaction_id") == transaction_id
        )
        if len(matching) != 1:
            raise DriverIntegrationError("handoff-authority-unavailable")
        if (
            matching[0].get("handoff_proposal_id") is None
            or matching[0].get("handoff_sha256") is None
        ):
            raise DriverIntegrationError("handoff-authority-unavailable")


def _canonical_repo_root(canonical_root: Path) -> Path:
    """Return the worktree root for canonical or isolated test layouts.

    Parameters
    ----------
    canonical_root:
        Crawler canonical root.

    Returns
    -------
    pathlib.Path
        Worktree root used for repository-relative manifests.
    """

    if canonical_root.name == "crawler" and canonical_root.parent.name == "menagerie":
        return canonical_root.parents[1]
    return canonical_root


def _instantiate_variant_artifact(
    item: WorkItem,
    representative_artifact: AuthorArtifact,
    representative_model: Mapping[str, Any],
    config: DriverConfig,
    context: AuthorityContext,
) -> AuthorArtifact:
    """Build a recipe-bearing variant artifact without an author or metadata-vet session.

    The accepted representative contributes its exact source, evidence, implementation,
    input contract, and family metadata. The variant receives its own work identity and
    later its own execution receipts. The provisional zero-count line is never written
    canonically; assembly replaces it from the constructed variant's worker receipt.

    Parameters
    ----------
    item:
        Explicitly designated non-representative family member.
    representative_artifact:
        Accepted representative recipe/source artifact.
    representative_model:
        Exact current accepted canonical representative revision.
    config:
        Current checker identity used to bind derived proposal identities.
    context:
        Active trusted intake and author-result binding roots.

    Returns
    -------
    AuthorArtifact
        Deterministic variant execution artifact bound to the representative revision.
    """

    if not item.is_family_variant:
        raise DriverIntegrationError("family template artifact requires a size variant")
    revision = representative_model.get("record_revision")
    if not isinstance(revision, str) or not revision:
        raise DriverIntegrationError("family representative has no accepted revision")
    proposal = deepcopy(representative_artifact.proposal)
    facts = proposal.get("proposed_facts")
    if not isinstance(facts, dict):
        raise DriverIntegrationError("representative proposal facts are incomplete")
    for fact_key in (
        "taxonomy",
        "external_metadata",
        "people_and_origin",
        "dates",
        "citation",
        "licenses",
        "source_resolution",
        "evidence",
    ):
        facts[fact_key] = deepcopy(representative_model.get(fact_key))
    identity = deepcopy(dict(representative_model["identity"]))
    identity.update(
        {
            "canonical_name": item.intake.name,
            **trusted_identity_fields(item.intake),
            "duplicate_of": None,
            "alias_of": None,
        }
    )
    facts["identity"] = identity
    _specialize_variant_recipe(facts, item, representative_model)
    try:
        provisional_line = mechanical_variant_parameter_input_line(0, facts["input_contract"])
        facts["website"] = instantiate_size_variant(
            representative_model,
            representative_model_id=item.family_representative_id,
            variant_parameter_input_line=provisional_line,
        )
    except FamilyTemplateError as exc:
        raise VariantRecipeUnsupported(str(exc)) from exc
    work_id = item.active_work_id
    proposal.update(
        {
            "proposal_id": stable_hash(
                {
                    "template_source_revision": revision,
                    "stable_id": item.stable_id,
                    "work_id": work_id,
                }
            ),
            "work_id": work_id,
            "stable_id": item.stable_id,
            "campaign_id": _campaign_id_for_item(item),
            "intake_snapshot_id": context.active_intake_snapshot_id,
            "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            "intake_item_sha256": stable_hash(context.intake_by_stable_id[item.stable_id]),
            "source_manifest_identity": representative_artifact.author_result.binding.source_manifest_identity,
            "dispatcher_identity": context.author_dispatcher_identity,
        }
    )
    verified_hashes = proposal.get("verified_hashes")
    if not isinstance(verified_hashes, dict):
        raise DriverIntegrationError("representative verified hashes are incomplete")
    verified_hashes["family_template"] = facts["website"]["template_hash"]
    try:
        identities = recompute_accepted_identities(
            facts,
            checker_prompt_hash=leg_checker_prompt_sha256(),
            checker_model=config.checker_model,
            checker_version=config.checker_version,
            schema_version=MODEL_SCHEMA_VERSION_V3,
        )
    except MetadataValidationError as exc:
        raise DriverIntegrationError(str(exc)) from exc
    implementation = facts.get("implementation")
    evidence = facts.get("evidence")
    if not isinstance(implementation, dict) or not isinstance(evidence, dict):
        raise DriverIntegrationError("representative recipe/evidence facts are incomplete")
    implementation["recipe_revision"] = identities.recipe
    evidence["evidence_identity"] = identities.evidence
    identities = recompute_accepted_identities(
        facts,
        checker_prompt_hash=leg_checker_prompt_sha256(),
        checker_model=config.checker_model,
        checker_version=config.checker_version,
        schema_version=MODEL_SCHEMA_VERSION_V3,
    )
    proposal.update(
        {
            "source_identity": identities.source,
            "evidence_identity": identities.evidence,
            "recipe_revision": identities.recipe,
            "fidelity_identity": identities.fidelity,
            "vet_identity": identities.vet,
        }
    )
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    representative_result = representative_artifact.author_result
    if not isinstance(representative_result, ProposedAuthorResult):
        raise DriverIntegrationError("family representative lacks a proposed author result")
    raw_result = deepcopy(representative_result.binding.raw_result)
    raw_result.update(
        {
            "result_id": stable_hash(
                {
                    "kind": "mechanical-family-variant",
                    "stable_id": item.stable_id,
                    "work_id": work_id,
                    "template_source_revision": revision,
                }
            ),
            "result_sha256": "sha256:" + "0" * 64,
            "stable_id": item.stable_id,
            "work_id": work_id,
            "campaign_id": f"campaign-{item.stable_id}",
            "dispatcher_identity": context.author_dispatcher_identity,
            "intake_snapshot_id": context.active_intake_snapshot_id,
            "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            "intake_item_sha256": proposal["intake_item_sha256"],
            "payload": {"arm": "PROPOSED", "proposal": proposal},
        }
    )
    raw_result["result_sha256"] = stable_hash(
        {key: value for key, value in raw_result.items() if key != "result_sha256"}
    )
    binding = replace(
        representative_result.binding,
        result_id=str(raw_result["result_id"]),
        result_sha256=str(raw_result["result_sha256"]),
        stable_id=item.stable_id,
        work_id=work_id,
        campaign_id=_campaign_id_for_item(item),
        dispatcher_identity=context.author_dispatcher_identity,
        intake_snapshot_id=context.active_intake_snapshot_id,
        intake_snapshot_sha256=context.active_intake_snapshot_sha256,
        intake_item_sha256=str(proposal["intake_item_sha256"]),
        raw_result=raw_result,
    )
    author_result = ProposedAuthorResult(
        binding,
        proposal,
        replace(representative_result.validation_report, stable_id=item.stable_id),
    )
    return AuthorArtifact(
        author_result=author_result,
        source_manifest=deepcopy(representative_artifact.source_manifest),
        model_dir=representative_artifact.model_dir,
        canonical_code_root=representative_artifact.canonical_code_root,
        template_source_revision=revision,
    )


def _specialize_variant_recipe(
    facts: JsonObject, item: WorkItem, representative_model: Mapping[str, Any]
) -> None:
    """Mechanically select a sibling constructor without accepting authored prose.

    Closed declarative family recipes can expose a conventional variant-selector
    keyword, or use the intake variant as a direct constructor symbol. Any adapter
    or ambiguous recipe falls back to the ordinary per-variant author/gate path.

    Parameters
    ----------
    facts:
        Mutable representative proposal facts copied for the variant.
    item:
        Explicit intake size variant providing the selector token.
    representative_model:
        Exact accepted representative revision supplying the only recipe/input base.

    Raises
    ------
    VariantRecipeUnsupported
        If the recipe has no single closed mechanical specialization.
    """

    try:
        implementation, input_contract, _derivation = specialize_size_variant_recipe(
            representative_model,
            representative_model_id=item.family_representative_id,
            variant_token=item.intake.variant,
        )
    except FamilyTemplateError as exc:
        raise VariantRecipeUnsupported(str(exc)) from exc
    facts["implementation"] = implementation
    facts["input_contract"] = input_contract


def _linear_slope(samples: Sequence[tuple[int, float]]) -> Optional[float]:
    """Return the least-squares timing slope against terminal count.

    Parameters
    ----------
    samples:
        ``(terminal_count, per_model_seconds)`` observations.

    Returns
    -------
    float | None
        Seconds added per terminal, or ``None`` until counts vary.
    """

    if len(samples) < 2:
        return None
    count = float(len(samples))
    sum_x = sum(float(sample[0]) for sample in samples)
    sum_y = sum(sample[1] for sample in samples)
    sum_xx = sum(float(sample[0]) ** 2 for sample in samples)
    sum_xy = sum(float(sample[0]) * sample[1] for sample in samples)
    denominator = count * sum_xx - sum_x**2
    if denominator == 0.0:
        return None
    return (count * sum_xy - sum_x * sum_y) / denominator
