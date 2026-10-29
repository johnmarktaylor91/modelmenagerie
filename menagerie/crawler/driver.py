"""Single-writer, resumable menagerie crawler scheduler.

The orchestration boundaries in this module are intentionally injectable.  Production
lanes may invoke Claude Code, Codex, conda, and the isolated worker; tests provide
deterministic fakes and never need those external systems.
"""

# ruff: noqa: F401

from __future__ import annotations

import ast
from dataclasses import dataclass as dataclass
import fcntl as fcntl
import json
import logging
import os
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import traceback
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol as Protocol, Sequence, TypeVar

from menagerie.crawler.artifact_transactions import (
    ArtifactBindingError,
    ArtifactCheckpointError,
    ArtifactCheckpointIncrement,
    ArtifactCheckpointProjection,
    ArtifactEventKind,
    ArtifactEventLedger,
    ArtifactInput,
    ArtifactRehydrationError,
    ArtifactTransactionProjection,
    ReconstructionInputs,
    StagedArtifact as StagedArtifact,
    authored_code_logical_path,
    publish_authorized_artifact,
    rehydrate_artifact_transaction,
    resolve_final_artifact_transaction,
    stage_private_artifact,
    staged_artifact_for_result,
    validate_artifact_checkpoint,
)
from menagerie.crawler.author_attempts import record_checker_findings
from menagerie.crawler.author_spend import (
    chargeable_author_sessions,
    enforce_author_session_allowance,
)
from menagerie.crawler.author_dispatch import (
    AuthorResult as AuthorResult,
    BlockedRecommendation,
    DeferRecommendation,
    ProposedAuthorResult,
    SkipRecommendation,
    build_author_envelope,
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
    ExecutableClosureV3,
    ExecutionReadManifestV3,
    RuntimeLookupDirectory,
    RuntimeMember,
    ShutdownInterruptionFact,
    WorkerLease,
    build_authority_context,
    collect_executable_closure_v3,
    compile_execution_read_manifest_v3_from_closure,
    derive_attempt_projection,
    derive_execution_identity,
    derive_runner_identity,
    derive_terminal_observation,
    load_current_attempt_proof,
    validate_currency,
)
from menagerie.crawler.checker_calibration import calibration_environment_snapshot
from menagerie.crawler.checker_conflicts import (
    CheckerVerdictConflictError,
    accuracy_acceptance_witnesses,
    accuracy_duplicate_accept_state,
    adjudication_identity_fields,
    append_checker_verdict_conflict_row,
    build_duplicate_accept_row,
    build_identity_flip_row,
    build_verdict_conflict_row,
    checker_verdict_conflict_path,
    confirmation_sampled,
    duplicate_accept_state,
    load_checker_verdict_conflict_rows,
    require_confirmation_pairing,
    terminal_acceptance_witnesses,
    terminal_gate_verdict,
    terminal_verdict_disagreements,
    unconfirmed_accepted_terminal_gate,
)
from menagerie.crawler.checker_dispatch import (
    CheckerBackoffSignal,
    CheckerMechanicalClaimError,
    build_fidelity_envelope,
    build_metadata_vet_envelope,
    build_terminal_disposition_envelope,
    gate_round_ceiling_state,
)
from menagerie.crawler.gate_reuse import (
    lane_adjudication_identity,
    reusable_terminal_gate,
)
from menagerie.crawler.award_seam import (
    disclosed_award_seam_classification as _disclosed_award_seam_classification,
)
from menagerie.crawler.cache_registry import author_result_cache_path
from menagerie.crawler.census_ack import (
    award_holds_ack_digest,
    is_ack_digest,
    stale_excluded_ack_digest,
)
from menagerie.crawler.checkpoint_cache import ValidationMemo
from menagerie.crawler.checkpoint import (
    FunnelSnapshot,
    append_canonical_requeue_grant,
    canonical_operational_ledger_path,
    canonical_records_root,
    canonical_requeue_grants_path,
    record_checkpoint_review,
    record_review_signoff,
)
from menagerie.crawler.notification_policy import classify_notification_event
from menagerie.crawler.constants import (
    ACCESS_BLOCKED_REASON_CODE,
    ACCESS_BLOCKED_STATUS_CODE,
    AUTHOR_SESSION_WALL_SECONDS,
    BLOCKED_DISPOSITION_REVIEW_STATUS_CODE,
    BLOCKED_PREREQUISITE_REASON_CODE,
    BLOCKED_PREREQUISITE_STAGE_REASON_CODES,
    ENVIRONMENT_ADJACENT_STATUS_CODE,
    ATTEMPT_SCHEMA_VERSION_V3,
    CHECKER_PROMPT_NAME,
    HOST_CAPABILITY_DEFERRAL_STATUS_CODES,
    DEFAULT_FORWARD_TIMEOUT_SECONDS,
    DEFAULT_NOTIFY_COMMAND as DEFAULT_NOTIFY_COMMAND,
    DEFAULT_PROGRESS_MILESTONES as DEFAULT_PROGRESS_MILESTONES,
    DEFAULT_REVIEW_CHECKPOINT_AT as DEFAULT_REVIEW_CHECKPOINT_AT,
    DEFAULT_NOTIFY_TIMEOUT_SECONDS,
    FAILURE_REASON_CODES,
    InvocationOrigin,
    MODEL_SCHEMA_VERSION_V3,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
    PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE,
    STDIO_TAIL_MAX_CHARS,
    metadata_gate_failure_reason,
    terminal_disposition_failure_reason,
    CheckerSeatRole,
)
from menagerie.crawler.environment_coverage import (
    PrerequisiteAggregateDisposition,
    aggregate_prerequisite_tokens,
)
from menagerie.crawler.envs import (
    EnvironmentIntent,
    EnvironmentRegistry,
    IntentProbes,
    load_environment_registry,
)
from menagerie.crawler.env_lifecycle import (
    ArtifactReceipt,
    DiskRecoveryError,
    EnvironmentExactnessError,
    EnvironmentProbeError,
    EnvironmentSolveError,
    ProbeResult,
    SequentialEnvironmentLifecycle,
    SolveResult,
    installed_package_inventory_bytes,
    materialized_environment_generation,
    parse_exact_lock,
    parse_probe_receipt_bytes,
    parse_resolved_export,
    validate_probe_receipts,
)
from menagerie.crawler.effort import EffortTracker, StageCap
from menagerie.crawler.fetcher import FetchTarget, fetch_targets
from menagerie.crawler.family_templates import (
    FamilyTemplateError,
    build_size_variant_derivation,
    family_representative_is_usable,
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
    validate_terminal_disposition_gate,
)
from menagerie.crawler.grant_parks import (
    AWARD_PARK_DISPOSITION,
    GRANT_PENDING_DISPOSITION,
    SELECTION_EXCLUSION_DISPOSITION,
    grant_park_path,
    grant_park_released,
    load_grant_park_rows,
)
from menagerie.crawler.close_object import (
    append_close_object,
    assert_close_discloses_the_census,
    build_close_object,
    close_object_path,
    continuing_member_disclosures,
)
from menagerie.crawler.park_release import (
    current_park_tree_identity,
    derive_park_machine_readmission,
    park_release_path_census,
    void_released_parks,
)
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_DISPOSITION,
    refusal_park_released,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    hash_bytes,
    payload_hash,
    stable_hash,
    utc_now,
)
from menagerie.crawler.intake import (
    IntakeItem,
    IntakeSnapshot,
    legacy_requires_fidelity_audit,
    load_intake_snapshot,
)
from menagerie.crawler.licenses import (
    LicenseDecision,
    LicenseEvidence,
    LicenseEvidenceStatus,
    RedistributionClass,
    recompute_license_decision,
)
from menagerie.crawler.metadata import (
    MetadataValidationError,
    canonical_meaningful_modes,
    input_signature_matches_contract,
    recompute_accepted_identities,
    validate_authored_facts_for_write,
)
from menagerie.crawler.modes import classify_observed_mode_receipts
from menagerie.crawler.models import (
    JsonObject,
    LedgerPaths as LedgerPaths,
    PartitionReport,
    manifest_source_rows,
)
from menagerie.crawler.operational_identity import mint_operational_event
from menagerie.crawler.host_capacity import (
    CapacityAssessment,
    append_capacity_deferral_row,
    assess_model_capacity,
    build_capacity_deferral_row,
    capacity_deferral_path,
    host_capability_preflight,
    host_capacity,
)
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorClass, MirrorStore
from menagerie.crawler.proposal import ProposalValidationError, model_code_manifest
from menagerie.crawler.promotion import (
    PROMOTION_SOURCE_CAMPAIGNS,
    append_promotion_row,
    build_promotion_row,
    promotion_extension_path,
)
from menagerie.crawler.recordio import (
    JsonlLedger,
    SingleWriterError,
    resolve_attempt_slot,
    scan_jsonl,
)
from menagerie.crawler.reducer import (
    AwardSupersessionError,
    AWARD_SUPERSESSION_SCAR_PREFIX,
    CanonicalReducer,
    GrantPendingSupersessionError,
    ReductionError,
    cold_forward_policy,
    default_ledger_paths as default_ledger_paths,
    expected_standard_asset,
    introduced_award_naming_grant_id,
    output_signature_error,
)
from menagerie.crawler.rung_staging import (
    registered_prior_intake_members,
    registered_prior_intake_snapshots,
)
from menagerie.crawler.routing import (
    IntentRoute,
    ModelRequirements,
    phase_routes,
    route_model,
)
from menagerie.crawler.state import _select_current, rebuild_state
from menagerie.crawler.terminal_review import (
    ORPHAN_ADOPTION_PER_LEG,
    ORPHAN_CITATION_ADOPTION_DISPOSITION,
    ORPHAN_CITATION_ADOPTION_MEMBER_DISPOSITION,
    TERMINAL_REVIEW_ADJUDICATED_DISPOSITION,
    TERMINAL_REVIEW_BACKLOG_PER_LEG,
    TERMINAL_REVIEW_CENSUS_CLASSES,
    TERMINAL_REVIEW_CENSUS_DISPOSITION,
    TERMINAL_REVIEW_CENSUS_ID_CAP,
    TERMINAL_UNADJUDICATED_DECISION,
    current_identity_terminal_gate,
    orphan_adoption_drift,
    orphan_hold_adoption_refusal,
    terminal_review_hold_candidate,
    unreviewed_failed_terminal_heads,
)
from menagerie.crawler.text_corpus import promote_author_root
from menagerie.crawler.status import (
    MACHINE_DEFERRAL_DISPOSITIONS,
    REPLAY_REFUSED_STALE_AXIS,
    assert_partition,
    assert_status_completeness,
    awards_stale_on_live_basis,
    completeness_report,
    funnel_counts,
    grant_release_census_from_ledgers,
    machine_deferral_rejection,
    scheduler_drain_census_from_ledgers,
    stale_census,
)
from menagerie.crawler.scheduler_parks import (
    load_scheduler_quarantine_ledger,
    scheduler_quarantine_path,
    unsatisfied_drain_tree_pins,
)
from menagerie.crawler.leg_exit_vocabulary import (
    LegExitClass,
    StopClass,
    cause_fingerprint,
    classify_exception,
    classify_result_status,
    exception_type_chain,
    repeat_cause_fuse_tripped,
)
from menagerie.crawler.wakeup import OperationalContext, WakeupManager, reduce_wake_episodes
from menagerie.crawler.leg_purpose import (
    LEG_PURPOSE_LICENSING_CLOSE,
    LegPurposeError,
    licensing_close_forfeit_refusals,
)
from menagerie.crawler.work_selection import (
    MemberFacts,
    SelectionDecision,
    SelectionMode,
    SelectionPlan,
    build_selection_plan,
    selection_continuation_identity,
    selection_mode_of,
)
from menagerie.crawler.worker_supervisor import (
    VerifiedWorkerResult,
    clear_worker_lease,
    current_boot_id,
    open_worker_lease,
    process_start_token,
    reconcile_worker_lease,
    shutdown_signal_handlers,
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
    AuthorBlockedPrerequisite,
    AuthorLane as AuthorLane,
    AuthorRepairTerminal,
    AuthorSessionRetryExhausted,
    AuthorUsagePause,
    BoundaryHook as BoundaryHook,
    CheckerLane as CheckerLane,
    CheckerOutcome,
    Clock as Clock,
    DriverConfig,
    DriverDependencies,
    AuthorBackoffError,
    DriverError as DriverError,
    DriverIntegrationError,
    LegStartRefused,
    HostCapabilityPreflightError,
    DriverLock,
    DriverLockError,
    DriverPaths,
    DriverPaused,
    DriverResult,
    DriverShutdown,
    PublicationAuthorizationRefused,
    EnvironmentBinding,
    EnvironmentLane as EnvironmentLane,
    ForwardLane as ForwardLane,
    Notifier as Notifier,
    RetryableOperatorError,
    UsageBackoffSignal,
    UsagePauseScheduler as UsagePauseScheduler,
    VariantRecipeUnsupported,
    WorkItem,
    _campaign_id_for_item,
    _intake_discovery_urls,
    default_driver_paths as default_driver_paths,
)

from menagerie.crawler.driver_models import (
    _artifact_lineage,
    _assemble_run_model,
    _assemble_terminal_model,
    _attempt_has_current_raw_authority,
    _attempt_policy_satisfied,
    _checker_item,
    _configure_driver_model_dependencies,
    _detected_mode_expansion,
    _driver_failure_attempt,
    _fidelity_gate_history,
    _fidelity_item_accepted,
    _fidelity_required,
    _find_gate,
    _gate_item_fingerprint,
    _gate_item_matches_proposal,
    _matching_attempts,
    _matching_model_attempts,
    _metadata_batches,
    _metadata_gate_accepted,
    _metadata_gate_history,
    _metadata_repair_count,
    _normalize_gate_generation,
    _placeholder_facts,
    _prepare_ledger_record,
    _redact_terminal_detail,
    _require_gate,
    _require_gate_bindings,
    _require_legacy_audit_fidelity,
    _terminal_checker_item,
    _terminal_fidelity_gate,
    _terminal_metadata_gate,
    _usable_family_representative,
    _without_ledger_fields,
)

from menagerie.crawler.code_identity import observed_repo_tree, runner_code_identity
from menagerie.crawler.power_assertion import driver_power_assertion
from menagerie.crawler.driver_progress import (
    CommandNotifier,
    _ascii_line,
    _completion_workflows,
    _environment_failure,
    _framework_from_intake,
    _framework_phase,
    _funnel_snapshot,
    _future_reset,
    _normalize_wake_reset,
    _is_sandbox_unavailable,
    _load_driver_state,
    _progress_summary,
    _read_json,
    _resolve_notify_command,
    _review_report,
    _review_summary,
    _write_driver_state,
    _write_json_atomic,
)

from menagerie.crawler.driver_receipts import (
    ModeRepairBudgetExhausted,
    ModeRepairUnsatisfiable,
    NEVER_MODEL_TERMINAL_OWNERS,
    ReceiptDriverMixin,
    SupervisedForwardLane,
    _DIAGNOSTIC_REDACTION_MARKER,
    _EXTERNALLY_CONTROLLED_ATTEMPT_FIELDS,
    _FORBIDDEN_CACHE_ROOT_NAMES,
    _WORKER_COMPLETION_PREFIX,
    _attempt_error_fields,
    _attempts_from_supervised,
    _attested_completion_line,
    _collect_worker_executable_closure,
    _compile_worker_read_manifest,
    _configure_driver_receipt_dependencies,
    _crawler_worker_runtime_paths,
    _diagnostic_relative_path,
    _diagnostics_root_for_work_root,
    _expected_adapter_sha256,
    _expected_code_manifest_sha256,
    _expected_input_asset_id,
    _expected_input_asset_sha256,
    _failure_owner,
    _forward_timeout_seconds,
    _is_diagnostic_redaction_reference,
    _non_model_terminal_error,
    _parent_cache_read_attempted,
    _physical_memory_bytes,
    _receipt_envelope_error,
    _redact_attempt_diagnostics,
    _runtime_member_kind,
    _stamp_failure_owner,
    _supervised_failure,
    _verified_worker_result,
    _worker_request,
)

from menagerie.crawler.driver_admission import (
    AdmissionEnvironmentMixin,
    CommandAuthorLane as CommandAuthorLane,
    CommandCheckerLane as CommandCheckerLane,
    CommandEnvironmentBackend as CommandEnvironmentBackend,
    QueueAuthorLane as QueueAuthorLane,
    _AWARD_CLOSURE_SCHEMAS,
    _AWARD_CLOSURE_SYMBOLS,
    _CheckerSessionPool,
    _INJECTED_FORWARD_CLOSURE_IDENTITY,
    _RUNNER_COMMON_EXECUTION_CLOSURE,
    _RUNNER_EXECUTION_CLOSURE,
    _RUNNER_IDENTITY_CACHE,
    _artifact_with_final_authority_objects,
    _assert_persisted_handoff_authority_available,
    _award_closure_from_bytes,
    _award_closure_identity,
    _bind_model_code_manifest,
    _canonical_crawler_root,
    _canonical_repo_root,
    _checker_prompt_hash,
    _configure_driver_admission_dependencies,
    _current_run_is_fresh,
    _environment_binding,
    _environment_from_quarantine,
    _execution_identity,
    _installed_package_manifest_bytes,
    _instantiate_variant_artifact,
    _normalize_artifact_modes,
    _routed_environment_packages,
    _observed_interpreter_facts,
    _park_grant_pending_supersession,
    _park_terminal_refusal,
    _persisted_deferral_rows,
    _still_valid_deferral_row,
    _quarantine_environment_payload,
    _quarantine_work_identity,
    _replay_then_live,
    _required_artifact_bytes,
    _runner_identity,
    _source_symbol_bytes,
    _specialize_variant_recipe,
    _stale_environment_binding,
    _mint_host_capability_head,
    _stash_machine_deferral_row,
    _terminal_pool_key,
    grant_reopens_host_capability_head,
    host_capability_head_reopens,
    _validate_artifact_identities,
    _validated_requeue_grants,
    _verify_model_code_manifest,
    bind_materialized_environment as bind_materialized_environment,
    build_command_environment_lane as build_command_environment_lane,
    buy_confirmation_session,
)

_configure_driver_model_dependencies(
    checker_prompt_hash=lambda: _checker_prompt_hash(),
    runner_identity=lambda modality: _runner_identity(modality),
    expected_input_asset_sha256=lambda proposal: _expected_input_asset_sha256(proposal),
    expected_input_asset_id=lambda proposal: _expected_input_asset_id(proposal),
    expected_adapter_sha256=lambda proposal: _expected_adapter_sha256(proposal),
    expected_code_manifest_sha256=lambda proposal: _expected_code_manifest_sha256(proposal),
    physical_memory_bytes=lambda: _physical_memory_bytes(),
    redact_attempt_diagnostics=lambda attempt, observation, diagnostics_root: (
        _redact_attempt_diagnostics(attempt, observation, diagnostics_root)
    ),
    framework_from_intake=lambda item: _framework_from_intake(item),
)
_configure_driver_receipt_dependencies(
    execution_identity=lambda artifact, environment, closure_identity: _execution_identity(
        artifact, environment, closure_identity=closure_identity
    ),
    runner_identity=lambda modality: _runner_identity(modality),
    checker_prompt_hash=lambda: _checker_prompt_hash(),
    injected_forward_closure_identity=lambda: _INJECTED_FORWARD_CLOSURE_IDENTITY,
)
_configure_driver_admission_dependencies(
    award_closure_identity=lambda: _award_closure_identity(),
    fetch_targets=lambda targets, cas_root: fetch_targets(targets, cas_root),
)

LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T")


_SHUTDOWN_ADMISSION_REGISTRY: Mapping[str, str] = {
    "author": "guard:author-admission",
    "checker": "guard:checker-admission",
    "environment-create": "guard:environment-create-admission",
    "environment-use": "guard:environment-use-admission",
    "model": "guard:model-admission",
    "lease": "guard:forward-admission|pre-slot-resolution",
    "spawn": "guard:forward-admission|pre-slot-resolution",
    "run-model-assembly": "guard:post-attempt-pre-award",
    "publication-admission": "guard:pre-publication-admission",
    "publication": "atomic:award-commit",
    "terminal-publication": "atomic:award-commit",
    "model-append": "atomic:award-commit",
    "post-award-observation": "guard:post-award-commit",
}
_SHUTDOWN_COMPOSITION_HOOK_REGISTRY: Mapping[str, str] = {
    "author": "pre-author",
    "checker": "pre-checker",
    "environment-create": "pre-environment-create",
    "environment-use": "pre-environment-use",
    "model": "pre-model",
    "lease|spawn": "pre-forward",
    "child-durable": "worker-lease-started",
    "attempt-durable": "after-forward",
    "run-model-assembly": "post-attempt-pre-award",
    "publication-admission": "pre-publication",
    "award-entry": "pre-award-commit",
    "award-atomic": "award-commit-entered",
    "post-award-observation": "post-award-commit",
    "wave-transition": "after-reduce",
}


#: A malformed PROGRAM is never a fact about a model's data. These recur identically for
#: every subsequent model, so absorbing them into the per-model record ladder converts a
#: fail-fast bug into a campaign-wide silent degradation: at 28,482 models a single
#: `NameError` in the terminal path yields 28,482 `terminal-unrecordable` events and zero
#: records, while the run LOOKS like it is working through the roster. That is strictly
#: worse than the abort the ladder exists to prevent, which at least stopped at model 1
#: with the real traceback. Deliberately EXCLUDED: `KeyError`, `IndexError`, `TypeError`
#: and `ValueError`, which record traversal raises on genuinely malformed record data --
#: those ARE facts about the model and belong in the ladder.
_PROGRAMMING_ERRORS: tuple[type[BaseException], ...] = (
    NameError,  # also covers UnboundLocalError
    AttributeError,
    ImportError,
    RecursionError,
    MemoryError,
    SystemError,
)

#: Distinct models in a row whose terminal could not be recorded before the recording path
#: is declared systemically broken. Matches the project's three-strikes anti-flail rule.
TERMINAL_UNRECORDABLE_STREAK_LIMIT = 3

#: D-63 (PM-1): the CLOSED set of leg-config axes that must not vary across
#: the legs of one certified rung. ``rung_config_digest`` is ``stable_hash``
#: over exactly these fields, and rung certification refuses a rung whose
#: legs disagree on it. Deliberately EXCLUDED (they legitimately vary per
#: leg): invocation_origin, wake_episode_id, the redrive/gate-catchup lanes,
#: only_unsettled/only_status, the acknowledgment flags, and the width
#: provenance sources (an explicit override that lands on the SAME width is
#: homogeneous; a different width is caught by the width axis itself).
RUNG_INVARIANT_CONFIG_FIELDS: tuple[str, ...] = (
    "scheduler",
    "author_concurrency",
    "author_wall_seconds",
    "checker_concurrency",
    "author_model",
    "author_version",
    "claude_cli_version",
    "checker_model",
    "checker_version",
    "campaign_id",
    "target",
    "phase",
    "review_checkpoint_at",
    "run_repair_max",
    "checker_calibration",
    "repo_tree_sha",
)


class TerminalRecordingUnavailable(DriverError):
    """The terminal recording path is systemically broken; the campaign must stop.

    This is the type-AGNOSTIC backstop. It bounds the damage from any systemic recording
    failure without depending on classifying exceptions correctly up front, which is why
    it matters more than the programming-error taxonomy above.

    Derived from `DriverError`, NOT `BaseException`, and the choice was probed rather than
    assumed. A `BaseException` would guarantee no per-model handler could absorb it -- but
    `run()` records its typed campaign-health failure event under `except Exception`, so
    `BaseException` would silently SKIP that event and cost exactly the diagnosability
    this backstop exists to provide. Probed both ways: the stop propagates identically,
    so the cheaper, better-instrumented type wins. It is raised from
    `_report_unrecordable_terminal`, which `_terminalize` calls OUTSIDE its own try, so
    the ladder can never re-absorb its own alarm.
    """


#: Distinct models in a row whose AWARD PARK append itself failed before the
#: park path is declared broken (AW-1). Separate from the recording streak:
#: an award refusal is the guard working, so it never advances the generic
#: breaker -- but the park write failing repeatedly is its own systemic fault.
AWARD_PARK_FAILURE_STREAK_LIMIT = 3


class AwardParkUnavailable(DriverError):
    """The award-supersession park path is systemically broken (AW-1).

    A refused award downgrade must land durably in the grant-pending ledger;
    when the park append itself keeps failing, the refusals would otherwise
    degrade into silent partition holes. This typed breaker owns that edge --
    it is never folded into :class:`TerminalRecordingUnavailable`, whose
    streak an award-custody refusal deliberately does not advance.
    """


class AwardCustodyError(DriverError):
    """A leg-appended ``runs -> non-runs`` transition lacks its award action.

    The close-time custody assertion (AW-1) reconciles every raw-head
    transition this leg appended: scalar award counts cannot see one eating
    balanced by one remint, so each downgrade must match one exact award
    action (an award-naming grant plus the typed scar receipt) and each
    ``runs -> runs`` remint is censused for disclosure.
    """


def _reconcile_award_custody(
    model_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_row_count: int,
    records_root: Path,
) -> JsonObject:
    """Reconcile every award transition appended after the leg's baseline.

    Parameters
    ----------
    model_rows:
        Complete canonical model rows in ledger order.
    baseline_row_count:
        Number of model rows that existed when the leg started; earlier rows
        are history other legs already own.
    records_root:
        Canonical records root holding the requeue-grants ledger.

    Returns
    -------
    dict[str, Any]
        Census: ``award_downgrades`` (each reconciled to its authority) and
        ``award_remints`` (disclosure of every award-over-award supersession).

    Raises
    ------
    AwardCustodyError
        If any leg-appended ``runs -> non-runs`` transition lacks an
        award-naming grant with its typed scar receipt.
    """

    rows = tuple(model_rows)
    by_revision = {str(row.get("record_revision")): row for row in rows}
    grants_path = records_root / "operational" / "requeue-grants.jsonl"
    grant_rows = scan_jsonl(grants_path, validate=False) if grants_path.exists() else ()
    downgrades: list[JsonObject] = []
    remints: list[JsonObject] = []
    violations: list[str] = []
    for row in rows[baseline_row_count:]:
        parent_revision = row.get("parent_revision")
        parent = by_revision.get(str(parent_revision)) if parent_revision else None
        if parent is None or parent.get("status", {}).get("kind") != "runs":
            continue
        stable_id = str(row.get("stable_id"))
        if row.get("status", {}).get("kind") == "runs":
            remints.append(
                {
                    "stable_id": stable_id,
                    "superseded_revision": str(parent_revision),
                    "reminted_revision": str(row.get("record_revision")),
                }
            )
            continue
        scar = f"{AWARD_SUPERSESSION_SCAR_PREFIX}{parent_revision}"
        parent_grants = set(parent.get("budget", {}).get("explicit_grants", []))
        row_grants = set(row.get("budget", {}).get("explicit_grants", []))
        named = introduced_award_naming_grant_id(
            grant_rows,
            row_grants - parent_grants,
            str(parent_revision),
            stable_id,
        )
        if named is not None and scar in tuple(row.get("scar_history", ())):
            downgrades.append(
                {
                    "stable_id": stable_id,
                    "superseded_revision": str(parent_revision),
                    "authority": "award-naming-grant",
                }
            )
        else:
            violations.append(
                f"{stable_id}: runs head {parent_revision} superseded by "
                f"{row.get('record_revision')} without an award-naming grant "
                "and its scar receipt"
            )
    if violations:
        raise AwardCustodyError(
            "award custody violated by this leg's appends: " + "; ".join(violations)
        )
    return {"award_downgrades": downgrades, "award_remints": remints}


def _award_naming_grant_id(
    ledgers: LedgerPaths,
    candidate: Mapping[str, Any],
    blocked_head: Mapping[str, Any],
) -> Optional[str]:
    """Return the introduced grant id naming the blocked award head, if any.

    The decision reads the durable requeue-grants ledger, never candidate
    fields (nothing a producer writes may authorize a downgrade); it mirrors
    the reducer's release rule exactly, so the stamp and the append can never
    disagree.

    Parameters
    ----------
    ledgers:
        Canonical ledger paths (locates the operational grants ledger).
    candidate:
        Proposed non-award terminal revision.
    blocked_head:
        The governing ``runs`` raw head the candidate would supersede.

    Returns
    -------
    str | None
        The award-naming grant id the candidate introduces, or ``None``.
    """

    head_grants = set(blocked_head.get("budget", {}).get("explicit_grants", []))
    candidate_grants = set(candidate.get("budget", {}).get("explicit_grants", []))
    introduced = candidate_grants - head_grants
    if not introduced:
        return None
    grants_path = canonical_requeue_grants_path(ledgers.models)
    grant_rows = scan_jsonl(grants_path, validate=False) if grants_path.exists() else ()
    blocked_revision = str(blocked_head.get("record_revision"))
    return introduced_award_naming_grant_id(
        grant_rows,
        introduced,
        blocked_revision,
        str(candidate.get("stable_id")),
    )


def _deferral_disclosure(
    stable_id: str,
    stashed: Optional[Mapping[str, Any]],
    persisted: Sequence[Mapping[str, Any]],
    record: Optional[Mapping[str, Any]],
) -> Optional[JsonObject]:
    """Describe one accounted deferral for the driver-state close block.

    Three sources, in descending directness: the row this run's valve stashed, the
    first VALIDATED non-park durable row a previous run wrote (BD-1: the same
    ``machine_deferral_rejection`` gauntlet every census applies, over every valve
    disposition -- a stale-deferred member's re-drive row discloses here too), and
    -- for a host-capability head whose valve did not fire this leg at all -- the
    head's own machine-derived claim. The last case is the one WS-2 creates: a
    resumed leg sees the member already settled, so no valve fires and no stash
    entry exists, and looking the row up blindly raised ``KeyError`` inside the
    close (i.e. crashed the campaign's own bookkeeping over an accounting
    improvement). A member none of the three sources can describe returns ``None``
    and is skipped rather than invented, so ``count`` and the disclosed list are
    kept consistent by the caller's filter.

    Parameters
    ----------
    stable_id:
        Accounted member.
    stashed:
        Row stashed by a valve fire this run, if any.
    persisted:
        Durable valve rows for the member, if any.
    record:
        The member's RAW ledger head, if any.

    Returns
    -------
    dict[str, Any] | None
        Disposition and remedy for the close block, or ``None``.
    """

    row = stashed
    if row is None:
        row = next(
            (
                candidate
                for candidate in persisted
                if machine_deferral_rejection(stable_id, candidate) is None
                and str(candidate.get("disposition"))
                not in {
                    GRANT_PENDING_DISPOSITION,
                    AWARD_PARK_DISPOSITION,
                    SELECTION_EXCLUSION_DISPOSITION,
                    REFUSAL_PARK_DISPOSITION,
                }
            ),
            None,
        )
    if row is not None:
        return {
            "stable_id": stable_id,
            "disposition": row["disposition"],
            "recheck_hint": row["recheck_hint"],
        }
    if (
        str((record or {}).get("status", {}).get("code"))
        not in HOST_CAPABILITY_DEFERRAL_STATUS_CODES
    ):
        return None
    claim = (record or {}).get("host_capability")
    required = claim.get("required_physical_memory_bytes") if isinstance(claim, Mapping) else None
    return {
        "stable_id": stable_id,
        "disposition": str((record or {}).get("status", {}).get("code")),
        "recheck_hint": (
            "re-run this campaign on a host with at least "
            f"{int(required) / 2**30:.1f} GiB of physical memory"
            if isinstance(required, int)
            else "recorded deferral; its durable valve row is not readable here"
        ),
    }


def _completion_close_status(
    *,
    complete: bool,
    phase: Optional[str],
    has_deferrals: bool,
    has_review_holds: bool,
    has_refusal_parks: bool,
) -> str:
    """Derive the closed-vocabulary completion status for one partition close.

    ``complete`` (with an optional ``-with-deferrals`` suffix) may surface ONLY
    when no live park of either kind exists: a grant-pending supersession and a
    terminal-refusal park are both operator decisions still owed, so either
    forces the distinct ``terminal-partition-complete`` family with its own
    suffix. Suffixes compose in the fixed order deferrals, review holds,
    refusal parks; a phase-scoped close carries the ``phase-`` prefix and
    trails ``:{phase}``.

    Parameters
    ----------
    complete:
        Whether the completeness report read complete.
    phase:
        Configured phase scope, or ``None`` for a whole-campaign close.
    has_deferrals:
        Whether machine-deferral valve rows (excluding parks) claimed models.
    has_review_holds:
        Whether live grant-pending supersession parks exist.
    has_refusal_parks:
        Whether live terminal-refusal parks exist.

    Returns
    -------
    str
        Exact completion status string.
    """

    deferral_suffix = "-with-deferrals" if has_deferrals else ""
    if complete and not has_review_holds and not has_refusal_parks:
        core = f"complete{deferral_suffix}"
    else:
        review_suffix = "-with-review-holds" if has_review_holds else ""
        refusal_suffix = "-with-refusal-parks" if has_refusal_parks else ""
        core = f"terminal-partition-complete{deferral_suffix}{review_suffix}{refusal_suffix}"
    return core if phase is None else f"phase-{core}:{phase}"


def _scoped_ledger_heads(models_path: Path, in_scope_ids: Sequence[str]) -> dict[str, JsonObject]:
    """Return the RAW ledger head of every in-scope model.

    The raw head is the settlement authority BD-1 restores to the boundary: a
    recorded revision exists or it does not, and no currency projection may
    turn one into a coverage hole. Read through the ONE source census
    (:func:`menagerie.crawler.leg_generation.current_model_heads`) -- every
    attested models layout, unioned, ambiguity refusing -- so the boundary can
    never disagree with the staging door about what a head is (rt3 carried
    finding C: these raw heads used to come from one shard alone).

    Parameters
    ----------
    models_path:
        Canonical model shard.
    in_scope_ids:
        Members this close accounts.

    Returns
    -------
    dict[str, dict[str, Any]]
        In-scope stable ID to raw ledger head.
    """

    from menagerie.crawler.leg_generation import current_model_heads

    scoped = frozenset(str(stable_id) for stable_id in in_scope_ids)
    heads = current_model_heads(canonical_records_root(models_path), require_ledger=False)
    return {stable_id: head for stable_id, head in heads.items() if stable_id in scoped}


def _scoped_stale_reasons(
    context: AuthorityContext,
    ledger_heads: Mapping[str, Mapping[str, Any]],
    current_ids: frozenset[str],
    projection_stale_reasons: Mapping[str, str],
) -> dict[str, str]:
    """Explain, per head, why the live projection could not vouch for it.

    A head the projection excluded is stale for a REASON, and the reason is
    what makes the refusal actionable. The PROJECTION'S OWN reason is the
    label authority (D-17): it is the verdict of the recorded-basis,
    registry-aware replay that actually refused the head, and it is the same
    map ``status --verify-partition`` renders, so the close and the boundary
    gate name one axis for one refusal. Re-attributing here through the
    registry-BLIND ``validate_currency`` named the first mismatching axis in
    its fixed tuple order instead: a registered-refinement-covered member
    whose real refusal was a non-identity replay class closed as
    ``stale:author_dispatcher_identity`` (remedy "register the rotation",
    unactionable -- the rotation was already registered) while the boundary
    verify printed ``stale:unclassified`` for the SAME members. Three labels,
    one mechanism; it mis-aimed three successive dispatch briefs (the D-17
    stale-label artifact, second mechanism).

    ``validate_currency`` remains only the FALLBACK for a head the projection
    never adjudicated (present in the unioned raw-head census but absent from
    the projection's domain), where no projection reason exists to quote; a
    head it calls current while still unvouched is attributed to
    :data:`~menagerie.crawler.status.REPLAY_REFUSED_STALE_AXIS` rather than to
    an invented axis.

    Parameters
    ----------
    context:
        Live authority the close projected under.
    ledger_heads:
        In-scope raw ledger heads.
    current_ids:
        Members the projection vouched for.
    projection_stale_reasons:
        The projection's own per-member stale reasons
        (:attr:`~menagerie.crawler.reducer.CanonicalReducer.dependency_stale_reasons`).

    Returns
    -------
    dict[str, str]
        Stale stable ID to its reason.
    """

    reasons: dict[str, str] = {}
    for stable_id, head in ledger_heads.items():
        if stable_id in current_ids:
            continue
        projected = projection_stale_reasons.get(stable_id)
        if projected is not None:
            reasons[stable_id] = projected
            continue
        reasons[stable_id] = (
            validate_currency(context, head)
            or f"{REPLAY_REFUSED_STALE_AXIS}: the "
            "dependency-currency replay refused this head while every directly "
            "replayable identity axis matched"
        )
    return reasons


def _live_grant_disclosures(ledgers: LedgerPaths, closed_at: str) -> list[JsonObject]:
    """W42/OS-10: every LIVE requeue grant at close, with its hold_reason.

    Liveness is the W31 aging census's replay derivation (ledger voids +
    consumption receipts); the hold_reason is the newest recorded
    ``requeue-grant-review-hold`` disclosure for the grant, or ``None`` when
    no hold was ever observed -- disclosed, never guessed. Disclosure only.
    """

    from menagerie.crawler.status import (  # noqa: PLC0415 -- lazy: close-path only
        grant_aging_census_from_ledgers,
    )

    try:
        now = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        aging = grant_aging_census_from_ledgers(ledgers, now=now)
    except (ValueError, OSError):
        return []
    operational_path = canonical_operational_ledger_path(ledgers.models)
    hold_reasons: dict[str, Optional[str]] = {}
    if operational_path.exists():
        for event in scan_jsonl(operational_path):
            details = dict(event.get("details") or {})
            if details.get("disposition") == "requeue-grant-review-hold":
                hold_reasons[str(details.get("grant_id"))] = details.get("hold_reason")
    grants_path = canonical_requeue_grants_path(ledgers.models)
    members: dict[str, str] = {}
    if grants_path.is_file():
        for row in scan_jsonl(grants_path, validate=False):
            if "void_of" not in row and row.get("grant_id") is not None:
                members[str(row["grant_id"])] = str(row.get("stable_id"))
    return [
        {
            "grant_id": grant_id,
            "stable_id": members.get(grant_id),
            "age_class": age_class,
            "hold_reason": hold_reasons.get(grant_id),
        }
        for age_class, grant_ids in sorted(aging.items())
        for grant_id in grant_ids
    ]


def _close_partition_arms(
    partition: PartitionReport,
    *,
    stale_park_rows: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> JsonObject:
    """Project one partition report into the close object's arms.

    DX-22 accounting totality: every intake member lands in EXACTLY ONE
    accounting arm -- ``buckets`` (current recorded head), ``machine_deferred``
    (recordless member a claimed row accounts), or ``stale_deferred``
    (stale-headed member a durable valve OR park row accounts the withheld
    re-drive for, BD-1). ``grant_pending``/``refusal_parked`` are DISCLOSURE
    arms: a live park coexists with its member's accounting by design and
    counts nothing.

    Parameters
    ----------
    partition:
        Report the close's own tripwire produced.
    stale_park_rows:
        Disposition-keyed live park rows (stable_id -> row) the close read
        durably. Stale-headed members among them are accounted here under
        their park disposition -- the park is the durable row that accounts
        the member (``externally_accounted_stale_ids``' close-census face).

    Returns
    -------
    dict[str, Any]
        Coverage arms, per-code buckets, and valve/park accounting.
    """

    stale_deferred: dict[str, set[str]] = {
        disposition: set(stable_ids)
        for disposition, stable_ids in partition.stale_deferred.items()
    }
    stale = partition.stale_ids
    already_deferred = partition.stale_deferred_ids
    for fallback_disposition, rows in (stale_park_rows or {}).items():
        for stable_id, row in rows.items():
            if stable_id in stale and stable_id not in already_deferred:
                disposition = str(dict(row).get("disposition") or fallback_disposition)
                stale_deferred.setdefault(disposition, set()).add(str(stable_id))
    return {
        "intake": len(partition.intake_ids),
        "valid": partition.valid,
        "missing": sorted(partition.missing_ids),
        "extra": sorted(partition.extra_ids),
        "duplicates": sorted(partition.duplicate_ids),
        "invalid": sorted(code for code in partition.buckets if code.startswith("invalid:")),
        "buckets": {
            code: sorted(stable_ids) for code, stable_ids in sorted(partition.buckets.items())
        },
        "machine_deferred": {
            disposition: sorted(stable_ids)
            for disposition, stable_ids in sorted(partition.machine_deferred.items())
        },
        "stale_deferred": {
            disposition: sorted(stable_ids)
            for disposition, stable_ids in sorted(stale_deferred.items())
        },
        "superseded_deferral_rows": sorted(partition.superseded_deferral_ids),
        "grant_pending": sorted(partition.grant_pending_ids),
        "refusal_parked": sorted(partition.refusal_parked_ids),
    }


def _close_currency_census(
    ledger_heads: Mapping[str, Mapping[str, Any]],
    scoped_current: Mapping[str, Mapping[str, Any]],
    stale_reasons: Mapping[str, str],
) -> JsonObject:
    """Project the dependency-currency divergence into the close object.

    The census the step-100 close never printed (T6-opus-13): the ledger said
    67 heads were decided while the live authority could vouch for 25, and
    nothing named the gap -- so an operator reading ``status`` saw a number 63%
    below the ledger with no signal that the cause was identity rotation
    rather than lost work.

    Parameters
    ----------
    ledger_heads:
        In-scope raw ledger heads.
    scoped_current:
        In-scope revisions the projection vouched for.
    stale_reasons:
        Per-head stale reasons.

    Returns
    -------
    dict[str, Any]
        Head/current counts, the per-axis stale bucket, verbatim reasons, and
        the stale-award line.
    """

    census = stale_census(stale_reasons)
    return {
        "ledger_heads": len(ledger_heads),
        "dependency_current": len(scoped_current),
        "stale": len(stale_reasons),
        "stale_by_axis": {axis: sorted(ids) for axis, ids in census.items()},
        "stale_reasons": {
            stable_id: stale_reasons[stable_id] for stable_id in sorted(stale_reasons)
        },
        "awards_stale_on_live_basis": sorted(
            awards_stale_on_live_basis(ledger_heads, stale_reasons)
        ),
    }


def _close_authority_identities(context: AuthorityContext) -> dict[str, str]:
    """Return the rotation-sensitive identities in force at one close.

    These are what make a later reader able to say WHY a re-derivation of the
    same books disagrees: every one of them is a content digest of shipped
    policy, and the step-100 saga was one rotation of exactly this set
    (51cdae77) re-litigating an entire campaign.

    Parameters
    ----------
    context:
        Live authority the close projected under.

    Returns
    -------
    dict[str, str]
        Identity name to value.
    """

    return {
        "author_prompt_identity": context.author_prompt_identity,
        "author_schema_identity": context.author_schema_identity,
        "author_dispatcher_identity": context.author_dispatcher_identity,
        "author_model_identity": context.author_model_identity,
        "checker_prompt_identity": context.checker_prompt_identity,
        "checker_schema_identity": context.checker_schema_identity,
        "checker_model_identity": context.checker_model_identity,
        "reducer_policy_identity": context.reducer_policy_identity,
        "runner_policy_identity": context.runner_policy_identity,
        "terminal_policy_identity": context.terminal_policy_identity,
        "publication_policy_identity": context.publication_policy_identity,
        "active_intake_snapshot_sha256": context.active_intake_snapshot_sha256,
    }


def _live_park_rows(
    persisted_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    scoped_ids: frozenset[str],
    scoped_current: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, JsonObject]]:
    """Return every in-scope model whose durable PARK row is still a live decision.

    Both park kinds record an operator decision still OWED, so neither may be
    invisible at close. The completion arm derives claims from the run-scoped
    stash on purpose (a stale row must never mask a model the run admitted and
    then lost), but the stash keeps exactly ONE row per model: on a resume that
    is whichever row :func:`~menagerie.crawler.driver_admission._still_valid_deferral_row`
    returns FIRST, and both park ledgers sit LAST in its fixed capacity ->
    coverage -> rebuild -> disk-floor -> grant-park -> refusal-park order. A
    model carrying an ordinary valve row AND a live park for the same work
    generation therefore closed ``complete-with-deferrals`` with the park
    nowhere in the status or the driver state -- the ``complete*``-behind-a-live-
    park state W3 and V2 both forbid.

    Park LIVENESS is a fact about the durable ledgers and the RAW LEDGER HEAD,
    so it is read from them. This is purely additive to the partition's
    accounting and can only ever move a close AWAY from ``complete``: it never
    claims a model, never satisfies coverage, and never relaxes a suffix.

    BD-1: the head, not the currency projection, is the release authority. A
    park whose model later recorded a revision is consumed by that revision;
    asking the projection instead meant a stale head could never release its
    park, so 16 of 17 terminal-refusal rows reported LIVE at the step-100 close
    for models terminalized hours earlier (T1-opus-04, T2-opus-07).

    Parameters
    ----------
    persisted_rows:
        Governing durable valve rows per stable ID (see
        :func:`~menagerie.crawler.driver_admission._persisted_deferral_rows`).
    scoped_ids:
        In-scope intake membership for this close.
    scoped_current:
        In-scope RAW ledger heads, keyed by stable ID.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Live park rows keyed by park disposition then stable ID. Both park
        dispositions are always present, possibly empty.
    """

    released = {
        GRANT_PENDING_DISPOSITION: grant_park_released,
        AWARD_PARK_DISPOSITION: grant_park_released,
        # R2-1: an exclusion park's liveness is the same head-bound fact — it
        # releases only when the held revision lawfully moved.
        SELECTION_EXCLUSION_DISPOSITION: grant_park_released,
        REFUSAL_PARK_DISPOSITION: refusal_park_released,
    }
    live: dict[str, dict[str, JsonObject]] = {disposition: {} for disposition in released}
    for stable_id, rows in persisted_rows.items():
        if stable_id not in scoped_ids:
            continue
        for row in rows:
            disposition = str(row.get("disposition"))
            is_released = released.get(disposition)
            if is_released is None:
                continue
            # Fail-closed exactly like the partition arm: a row failing the
            # digest/disposition/remedy contract accounts for nothing, here as
            # there. It cannot force a close either.
            if machine_deferral_rejection(stable_id, row) is not None:
                continue
            if is_released(row, scoped_current.get(stable_id)):
                continue
            live[disposition][stable_id] = dict(row)
    return live


@dataclass(frozen=True)
class DurableReadinessAccounting:
    """The durable half of the close's partition accounting, as ONE derivation.

    D-17 (LC-3): the close, the driver, and ``crawl status --verify-partition``
    must derive readiness identically, and until this object existed they did
    not -- the CLI omitted the scheduler-quarantine ledger from its claims view
    and passed no ``stale_accounted_ids`` at all, so a rung the close would
    complete reported missing members and a stale head a durable park accounts
    refused at the boundary while the real close passed. Both consumers now
    read THIS derivation; neither re-implements it.
    """

    persisted_rows: Mapping[str, tuple[JsonObject, ...]]
    live_parks: Mapping[str, Mapping[str, JsonObject]]
    durable_park_accounted: frozenset[str]
    durable_valve_accounted: frozenset[str]

    @property
    def stale_accounted_ids(self) -> frozenset[str]:
        """Members whose staleness a durable park or valve row accounts."""

        return self.durable_park_accounted | self.durable_valve_accounted


def durable_readiness_accounting(
    driver: Any,
    *,
    scoped_ids: frozenset[str],
    ledger_heads: Mapping[str, Mapping[str, Any]],
    claimed_ids: frozenset[str] = frozenset(),
) -> DurableReadinessAccounting:
    """Derive the durable park/valve accounting the close asserts under.

    Park liveness comes from the DURABLE ledgers, not only from a run-scoped
    claim stash: the stash holds one row per model, and an ordinary valve row
    winning the resume short-circuit used to hide a live park from the close
    entirely (C3-B). Additive and fail-closed -- this can only ever move the
    close away from ``complete``.

    C3-B completed for VALVE rows: the close assertion runs PRE-admission, so
    no valve fired this run and the run-scoped stash is empty -- a stale head
    whose durable capacity/coverage/rebuild/disk row is a validating claim was
    refused as unaccounted, and every relaunch died there before the valves
    could re-fire (leg-30: seven prior-era awards deferred needs-host-disk /
    needs-environment-coverage). Pre-admission there is no admitted-then-lost
    model to mask; park dispositions stay with the liveness-filtered park
    accounting.

    Parameters
    ----------
    driver:
        The driving object (duck-typed: ``paths.ledgers.models`` locates the
        records root; offline consumers pass a namespace).
    scoped_ids:
        In-scope intake membership for this close or boundary check.
    ledger_heads:
        In-scope RAW ledger heads (the park-release authority, BD-1).
    claimed_ids:
        Members a run-scoped claim stash already accounts; their valve rows
        are the stash's business, never double-counted here.

    Returns
    -------
    DurableReadinessAccounting
        Durable rows, live parks, and the exact ``stale_accounted_ids`` the
        close passes to :func:`~menagerie.crawler.status.assert_partition`.
    """

    persisted_rows = _persisted_deferral_rows(driver)
    live_parks = _live_park_rows(persisted_rows, scoped_ids, ledger_heads)
    durable_park_accounted = frozenset(
        stable_id for rows in live_parks.values() for stable_id in rows
    )
    park_dispositions = frozenset(live_parks)
    durable_valve_accounted = frozenset(
        stable_id
        for stable_id, rows in persisted_rows.items()
        if stable_id in scoped_ids
        and stable_id not in claimed_ids
        and any(
            isinstance(row, Mapping)
            and str(row.get("disposition", "")) in MACHINE_DEFERRAL_DISPOSITIONS
            and str(row.get("disposition", "")) not in park_dispositions
            and machine_deferral_rejection(stable_id, row) is None
            for row in rows
        )
    )
    return DurableReadinessAccounting(
        persisted_rows=persisted_rows,
        live_parks=live_parks,
        durable_park_accounted=durable_park_accounted,
        durable_valve_accounted=durable_valve_accounted,
    )


def _park_released_for_selection(
    row: Mapping[str, Any], live_grant: Mapping[str, Any] | None
) -> bool:
    """Return whether a LIVE grant is the operator decision one park row awaits.

    COMPOSE ruling (AW x WS-1): both grant-park dispositions record an operator
    decision still OWED, and their own remedy text names the release lever --
    a grant the NEXT leg consumes (``requeue ... --stage <blocked_stage>``, an
    award park additionally ``--blocked-award-revision <revision>``). Once
    that exact grant exists LIVE, the pending decision has been made:
    selection must let the grant lane admit and consume it (the close-time
    supersession still validates fully, and close-time park LIVENESS stays
    ledger+head derived -- only the published superseding head releases the
    durable row). Terminal-refusal parks have no grant lever and stay
    absolute until an operator void (WS-1 adjudication 2).

    Parameters
    ----------
    row:
        One live durable park row.
    live_grant:
        The member's live (unvoided, unconsumed) requeue grant, if any.

    Returns
    -------
    bool
        True when the grant releases this row for SELECTION only.
    """

    if live_grant is None:
        return False
    disposition = str(row.get("disposition"))
    pending = row.get("pending")
    if not isinstance(pending, Mapping):
        return False
    if disposition == AWARD_PARK_DISPOSITION:
        # AW-1's award-naming seam: only a grant naming the EXACT blocked
        # award revision reopens a runs head; stage alone releases nothing.
        named = live_grant.get("blocked_award_revision")
        return isinstance(named, str) and named == str(pending.get("blocked_parent_revision"))
    if disposition == GRANT_PENDING_DISPOSITION:
        return str(live_grant.get("stage")) == str(pending.get("blocked_stage"))
    # R2-1: a selection-exclusion park is released by NO grant of any spelling
    # (its stage sentinel is outside the grant vocabulary; the fall-through
    # False is the contract, stated explicitly so nobody "fixes" it).
    return False


def _pending_metadata_readmission(record: Mapping[str, Any]) -> bool:
    """Return whether one current record re-enters work for its catch-up gate.

    The ONLY documented producer of a pending-metadata record that re-admission
    may re-buy is the R1/R2 catch-up path: a ``runs`` record awaiting its
    metadata_batch gate (PLAN.md's pending path). The execution-failure fidelity
    waiver (M1) creates a second pending-metadata producer -- machine-dead
    failure terminals -- and re-admitting those would re-enter every import-dead
    model into the author/gate lane on every leg (D1-F5). A failure terminal
    stays closed until an operator requeue grant reopens it
    (``_consume_requeue_grants``), under the wave driver AND any scheduler that
    shares this predicate; a rollback from the scheduler to the wave driver is
    only safe because both consult this one spelling.

    Parameters
    ----------
    record:
        Current canonical model revision for a scheduled intake item.

    Returns
    -------
    bool
        Whether the pending-metadata catch-up path re-admits this record.
    """

    return (
        record.get("authored_metadata_state") == "pending"
        and record.get("status", {}).get("kind") == "runs"
    )


class _TerminalArtifactEscape(Exception):
    """Internal carrier routing a terminal artifact failure past the record ladder.

    `_terminalize` contains failures of the RECORD -- construction and canonical append.
    Artifact FINALIZATION is a different kind of failure: campaign-level causes (pauses,
    retryable transport, provider backoff, programming errors) re-raise with their real
    type, while model-scoped refusals are contained by
    `_contain_terminal_finalization_refusal` -- the same record lands with the artifact
    kept and finalization withheld. The old blanket re-raise assumed a model-local
    handler in every calling lane; rung 11 (2026-08-07) proved the
    `_forward_and_reduce` failure arms have none, so one model's stale dependency
    vector aborted the campaign. This type never escapes `_terminalize`.

    Parameters
    ----------
    cause:
        Artifact failure to carry out of the terminal append.
    """

    def __init__(self, cause: BaseException) -> None:
        """Store the carried artifact failure."""

        super().__init__(str(cause))
        self.cause = cause


#: AL-6 prerequisite classifiers over the TYPED ``prerequisite_ids`` of a checked
#: BLOCKED recommendation -- closed structural token grammars, never free text or
#: reason strings, so no author prose can steer the routing split. Both grammars
#: are transcribed from the pilot's 20 reclassified members (rung 200a census).
#: NOTE (W14 seam): the parent lane's ``menagerie/crawler/release_vocabulary.py``
#: will own the ``runtime-dependency:<pep503>`` token grammar; these move there
#: verbatim when that module lands.
#:
#: The CUDA set names the CUDA PLATFORM CAPABILITY itself (pilot spellings:
#: ``cuda-platform`` m4388, ``cuda-device`` m8215, ``cuda-runtime`` m9073, plus
#: the toolkit/compiler spellings the same grammar admits). It deliberately does
#: NOT match package ids that merely contain "cuda" (``runtime-dependency-
#: knn-cuda`` names an installable package, not the capability).
_CUDA_CAPABILITY_PREREQUISITE_IDS: frozenset[str] = frozenset(
    {
        "cuda",
        "cuda-platform",
        "cuda-device",
        "cuda-runtime",
        "cuda-toolkit",
        "needs-cuda",
        "nvcc",
    }
)


def _is_runtime_dependency_prerequisite_id(prerequisite_id: str) -> bool:
    """Return whether one typed prerequisite ID names an installable runtime dep.

    The pilot cohort's exact spellings: the bare ``runtime-dependency`` class
    marker, the ``runtime-dependency-<name>`` family, and the
    ``<name>-distribution`` / ``<name>-package`` distribution spellings.

    Parameters
    ----------
    prerequisite_id:
        One schema-validated blocked-arm prerequisite ID.

    Returns
    -------
    bool
        Whether the ID matches the closed runtime-dependency grammar.
    """

    # The bare class marker ``runtime-dependency`` is deliberately EXCLUDED
    # (EN-6 x AL-6 batch-assembly reconciliation): it names no distribution,
    # so it is CONTENT-FREE under the EN-6 grammar and can never steer a
    # disposition -- routing it into the grantable environment-adjacent class
    # would let the cheapest possible author claim (m4954's shape) launder
    # into a machine-releasable arm. W14's own seam translation
    # (``prerequisite_token_from_author_id``) already fails the bare marker
    # closed toward the human lane; this predicate agrees. A set carrying the
    # bare marker BESIDE content-bearing ids (m4746) still classifies through
    # the content-bearing members.
    return (
        prerequisite_id.startswith("runtime-dependency-")
        or prerequisite_id.endswith("-distribution")
        or prerequisite_id.endswith("-package")
    )


def _blocked_prerequisite_class(prerequisite_ids: tuple[str, ...]) -> Optional[str]:
    """Classify one checked BLOCKED recommendation's typed prerequisite set.

    CUDA outranks runtime-dependency on purpose: a member naming both (m8215:
    four runtime deps plus ``cuda-device``) is blocked on the CAPABILITY --
    without a CUDA host the runtime deps are unbuildable -- so the capability
    deferral is the binding, recoverable fact.

    Parameters
    ----------
    prerequisite_ids:
        Typed prerequisite IDs from the checked recommendation.

    Returns
    -------
    str | None
        ``"cuda"``, ``"runtime-dependency"``, or ``None`` for every other
        prerequisite class (which keeps the plain ``failed:<stage>`` arm).
    """

    ids = {str(value) for value in prerequisite_ids}
    if ids & _CUDA_CAPABILITY_PREREQUISITE_IDS:
        return "cuda"
    if any(_is_runtime_dependency_prerequisite_id(value) for value in ids):
        return "runtime-dependency"
    return None


def _handoff_inherited_acceptance_authority(
    artifact: AuthorArtifact,
    final_transactions: Mapping[
        tuple[str, str, ArtifactTransactionId], ArtifactTransactionProjection
    ],
) -> bool:
    """Return whether one artifact PROVABLY inherits prior-campaign acceptance.

    The H-5 K=2 duplicate-accept door exemption (opus F4 / fable F11): the old
    guard keyed on ``isinstance(artifact, ActivatedHandoffArtifact)``, but that
    class is assigned purely from ``config.only_status`` -- configuration, not
    provenance -- and ``handoff_sha256`` defaults to ``""``, so a
    handoff-free artifact (a SAME-campaign ``--only-status`` re-run, e.g. the
    host-capability memory cohort) rode the exemption and published without
    ever passing the K=2 door. This predicate keys on what the real
    cross-campaign handoff path actually stamps, all replay-derived from the
    ledger-backed final-transaction projection with zero config input:

    * the artifact's exact staged transaction resolves in the validated
      final-transaction projection;
    * that transaction's reconstruction inputs carry a ``handoff_execution``
      whose ``handoff_sha256`` is a non-empty string equal to the artifact's
      own stamp (the constructor stamp alone is never trusted); and
    * the recorded author result is the ``DEFER_RECOMMENDATION`` arm -- the
      one arm minted ONLY as a terminal deferral of its ORIGIN campaign and
      consumed ONLY by the routed re-run on a capable host
      (``_HANDOFF_AUTHORITY_STATUS_CODES``), so the consuming campaign is by
      construction distinct from the minting one. A same-campaign re-run
      rehydrates its OWN ``PROPOSED`` record and therefore fails this arm:
      its acceptance authority is this campaign's and the K=2 door applies.

    Anything failing any conjunct gets the full K=2 door (fail-closed);
    recorded history is untouched -- this is a refusal-point tightening on
    LIVE publication only.

    Parameters
    ----------
    artifact:
        Publication-eligible artifact at the K=2 door.
    final_transactions:
        The driver's validated ``(stable_id, work_id, transaction_id)`` ->
        final-transaction projection index.
    """

    if not isinstance(artifact, ActivatedHandoffArtifact):
        return False
    if not isinstance(artifact.handoff_sha256, str) or not artifact.handoff_sha256:
        return False
    if artifact.staged is None:
        return False
    binding = artifact.author_result.binding
    transaction = final_transactions.get(
        (binding.stable_id, binding.work_id, artifact.staged.transaction_id)
    )
    if transaction is None:
        return False
    handoff = transaction.reconstruction_inputs.handoff_execution
    if not isinstance(handoff, Mapping):
        return False
    if handoff.get("handoff_sha256") != artifact.handoff_sha256:
        return False
    recorded = transaction.reconstruction_inputs.author_result
    return (
        isinstance(recorded, Mapping) and recorded.get("kind") == "DEFER_RECOMMENDATION"
    )


class CrawlerDriver(AdmissionEnvironmentMixin, ReceiptDriverMixin):
    """Lock-guarded single-writer scheduler integrating slices A through E."""

    def __init__(
        self,
        paths: DriverPaths,
        config: DriverConfig,
        dependencies: DriverDependencies,
        *,
        registry: Optional[EnvironmentRegistry] = None,
    ) -> None:
        """Bind deterministic paths, policy, injected effects, and intent registry."""

        self.paths = paths
        self.config = config
        self.dependencies = dependencies
        self.registry = registry or load_environment_registry(target=config.target)
        if isinstance(self.dependencies.author, (CommandAuthorLane, QueueAuthorLane)):
            # The author envelope is built inside the lane, while the exact routed
            # inventory is owned by this driver. Bind the already-loaded registry
            # once before concurrent author sessions begin so every request sees
            # the same capability sheet later used by admission and normalization.
            self.dependencies.author.registry = self.registry
        self._reduced = 0
        # Per-model ``ExcType: message`` for the failure that made a terminal
        # unrecordable. Rung 3 converts an upstream defect from a MISLABELLED
        # record into a MISSING one -- the better trade -- but the loss then
        # presents as `PartitionError: missing=[...]`, which reads like a coverage
        # hole rather than the one-line defect it usually is. Held here so the
        # completion assertion can name the original exception instead of leaving
        # it only inside the ``terminal-unrecordable`` operational report.
        self._unrecordable_diagnostics: dict[str, str] = {}
        # Streak of DISTINCT models whose terminal could not be recorded, cleared by any
        # successful terminal append. Distinct, not raw consecutive: one stubborn model
        # reached twice by later lanes is ONE fact about ONE model, already covered by the
        # partition tripwire, and must not be able to trip a systemic alarm by itself.
        # A broken recording path shows up as DIFFERENT models failing to record.
        self._unrecordable_streak: list[str] = []
        # Streak of DISTINCT models whose award-supersession PARK append failed
        # (AW-1's typed breaker); cleared by any successful award park. Never
        # folded into the recording streak above -- an award refusal is the
        # guard working.
        self._award_park_failure_streak: list[str] = []
        # Model-ledger row count at leg start: the close-time award-custody
        # reconciliation adjudicates only the transitions THIS leg appended.
        self._models_row_baseline: int = 0
        self._leg_generation: Optional[Any] = None
        # W6: the live leg's lifecycle span (set by _record_leg_started,
        # consumed exactly once by the run() wrapper's leg_ended emission).
        self._leg_lifecycle: Optional[dict[str, Any]] = None
        self._leg_config_emitted: bool = False
        self._selection_plan_admitted: frozenset[str] = frozenset()
        # Content digest of the LAST candidate prepared for each model's terminal
        # append, kept only until that append succeeds. When both recording rungs
        # refuse, the digest identifies the exact rejected candidate inside the
        # durable terminal-refusal park row.
        self._refused_candidate_digests: dict[str, str] = {}
        self._family_artifacts: dict[str, AuthorArtifact] = {}
        self._final_artifact_transactions: dict[
            tuple[str, str, ArtifactTransactionId], ArtifactTransactionProjection
        ] = {}
        self._intake_snapshot: Optional[IntakeSnapshot] = None
        self._authority_context: Optional[AuthorityContext] = None
        self._shutdown_event = threading.Event()
        # Author-lane health state for the no-publication boundary (rung 7, m7362).
        # Publications witness that THIS RUN's author lane works; the death streak
        # counts consecutive DISTINCT models dying without publication since the
        # last witness. Wave-pool worker threads record publications concurrently,
        # so both structures share one lock.
        self._author_lane_state_lock = threading.Lock()
        self._author_lane_publications: list[dict[str, Any]] = []
        # W5: each death is a structured witness (stable_id, wall + monotonic
        # instants, failure class, source-exclusion) so the drought detector
        # can window and classify instead of counting adjacency.
        self._author_no_publication_deaths: list[dict[str, Any]] = []
        # Strand accounting: models whose published author work this run staged,
        # minus models whose record reduced. On a campaign abort the difference is
        # every model holding published-but-unreduced work, and each one gets a
        # durable stranded record instead of silently vanishing (rung 7 stranded
        # five models, including its only checker-accurate proposal, with nothing
        # durable naming them).
        self._models_with_staged_work: set[str] = set()
        self._reduced_model_ids: set[str] = set()
        # Validated deferral valve ledger rows for models THIS run withheld,
        # keyed by stable ID. The two admission valves (environment coverage,
        # host capacity) refuse to mint canonical terminals by design, so these
        # rows are the completion partition's only truthful accounting for a
        # withheld model. Stashed at valve-fire time -- never re-read blind from
        # the ledger at completion -- so a stale row from an earlier run can
        # never mask a model the CURRENT run admitted and then silently lost
        # (rung 9b: 9 deliberate deferrals guaranteed a PartitionError because
        # the partition had no arm for the class the valves themselves create).
        self._machine_deferral_rows: dict[str, JsonObject] = {}
        self._driver_lock: Optional[DriverLock] = None
        self._artifact_checkpoint_cache: Optional[
            tuple[ArtifactEventLedger, int, tuple[Path, ...], ArtifactCheckpointIncrement]
        ] = None
        self._checkpoint_memo_instance: Optional[ValidationMemo] = None
        self._ledger_hot_path_samples: list[tuple[int, float]] = []

    def run(self, *, after_review: bool = False) -> DriverResult:
        """Acquire authority and resume the first unsatisfied durable work identity."""

        # Sleep immunity is driver-held, never borrowed from ambient host state
        # (200a rode a stray caffeinate nothing owned); the guard state is
        # published in the lock owner record for the armed-guard census.
        with driver_power_assertion(watched_pid=os.getpid()) as power:
            owner = {
                "pid": os.getpid(),
                "process_started_at": self.dependencies.clock(),
                "boot_id": current_boot_id(),
                # PID-reuse-proof identity for the crawl_doctor liveness probe:
                # a held lock whose recorded start_token no longer matches the
                # running pid is a stranger, never the driver (D-35).
                "start_token": process_start_token(os.getpid()),
                "run_id": self.config.run_id,
                "target": self.config.target,
                "command": list(sys.argv),
                "power_assertion": power.to_dict(),
                # DX-45: WHICH code adjudicates this leg, recorded durably.
                "runner_identity": runner_code_identity(),
            }
            try:
                with DriverLock(self.paths.lock_path, owner) as lock:
                    # Held for the author-lane host checks: "locks held" is one of the
                    # three mechanical facts the no-publication boundary verifies.
                    self._driver_lock = lock
                    try:
                        with shutdown_signal_handlers(self._shutdown_event):
                            # W6: EVERY _run_locked exit path pairs the leg's
                            # durable leg_started with exactly one typed
                            # leg_ended -- returns through the else arm,
                            # raises (BaseException included: teardown and
                            # shutdown facts are deliberately outside
                            # Exception) through the except arm. A run that
                            # never emitted leg_started (wake-noop, leg-start
                            # refusal) has nothing to pair and both arms
                            # no-op.
                            try:
                                result = self._run_locked(after_review=after_review)
                            except Exception as exc:
                                from menagerie.crawler.crawl_preflight import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
                                    InvocationRefused,
                                )

                                if not isinstance(exc, InvocationRefused):
                                    # RC-1: a typed pre-spend invocation refusal
                                    # stranded nothing and opened no span; it is
                                    # recorded as its own operational event, never
                                    # as a campaign-health failure.
                                    self._record_driver_failure(exc)
                                self._record_leg_ended_for_exception(exc)
                                raise
                            except BaseException as exc:
                                self._record_leg_ended_for_exception(exc)
                                raise
                            else:
                                self._record_leg_ended_for_result(result)
                                return result
                    finally:
                        self._driver_lock = None
            except SingleWriterError as exc:
                raise DriverLockError(str(exc)) from exc

    def _record_driver_failure(self, exc: Exception) -> None:
        """Append a typed campaign-health failure, then strand records, then state.

        A campaign abort must never silently discard in-flight work. Rung 7
        stranded five models -- all holding staged published results and recorded
        gates, one of them the rung's only checker-accurate proposal -- with
        nothing durable naming them: they were simply MISSING from the terminal
        table. Every model whose published work this run staged but whose record
        never reduced now gets a durable ``model-stranded-by-campaign-abort``
        operational event, and the driver state file carries the full stranded
        list plus the retry-exhaustion shape, so resume tooling and a human
        reading the wreckage both see exactly what survives and what to recover.

        Parameters
        ----------
        exc:
            Failure about to propagate out of the locked run.
        """

        created_at = self.dependencies.clock()
        retryable = isinstance(exc, RetryableOperatorError)
        retry_exhausted = isinstance(exc, AuthorSessionRetryExhausted)
        exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
        with self._author_lane_state_lock:
            unreduced = self._models_with_staged_work - self._reduced_model_ids
        # A machine-deferred model is NOT stranded: its deferral valve row is the
        # durable record naming its remedy, so labelling it
        # ``model-stranded-by-campaign-abort`` with ``staged-artifact-and-gates``
        # would be false twice over (rung 9b recorded all nine deferrals that way,
        # and none of them held a gate).
        machine_deferred = sorted(unreduced & set(self._machine_deferral_rows))
        stranded = sorted(unreduced - set(self._machine_deferral_rows))
        event = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": (
                OperationalEventStatus.RETRYABLE_INFRASTRUCTURE.value
                if retryable
                else OperationalEventStatus.RUNNER_FAILED.value
            ),
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": len(stranded)},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "exception_type": exception_type,
                "message": str(exc),
                "retryable_infrastructure": retryable,
                "retry_exhausted": retry_exhausted,
                "stranded_models": stranded,
                "machine_deferred_models": machine_deferred,
            },
        }
        event = mint_operational_event(event)
        with JsonlLedger(
            self.paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION
        ) as operational:
            operational.append(event)
            for stable_id in stranded:
                operational.append(
                    mint_operational_event(
                        {
                            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                            "created_at": created_at,
                            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                            "status": (
                                OperationalEventStatus.RETRYABLE_INFRASTRUCTURE.value
                                if retryable
                                else OperationalEventStatus.RUNNER_FAILED.value
                            ),
                            "provider": None,
                            "observed_response": None,
                            "reset_at": None,
                            "queued_work_counts": {"models": 1},
                            "current_environment": None,
                            "run_id": self.config.run_id,
                            "machine_id": self.config.machine_id,
                            "details": {
                                "kind": "model-stranded-by-campaign-abort",
                                "stable_id": stable_id,
                                "driver_failure_event_id": event["event_id"],
                                "exception_type": exception_type,
                                # The staged artifact cache and recorded gates survive
                                # on disk; a resume rehydrates them without re-running
                                # the author. This event exists so nothing has to
                                # infer that from a shortfall in the terminal table.
                                "recoverable_state": "staged-artifact-and-gates",
                            },
                        }
                    )
                )
        _write_driver_state(
            self.paths.driver_state,
            {
                "status": (
                    OperationalEventStatus.RETRYABLE_INFRASTRUCTURE.value
                    if retryable
                    else "failed:runner"
                ),
                "exception_type": exception_type,
                "message": str(exc),
                "retry_exhausted": retry_exhausted,
                "stranded_models": stranded,
                "machine_deferred_models": machine_deferred,
            },
        )

    def _check_shutdown(
        self,
        admission_boundary: str,
        *,
        item: Optional[WorkItem] = None,
        work_id: Optional[str] = None,
        execution_identity: Optional[str] = None,
    ) -> None:
        """Stop before an external or admission boundary once shutdown is requested.

        Parameters
        ----------
        admission_boundary:
            Stable inventory name for the boundary being guarded.
        item:
            Optional scheduled model association.
        work_id, execution_identity:
            Optional exact work and execution identities already derived by the caller.

        Raises
        ------
        DriverShutdown
            If the async-safe signal event has been set.
        """

        if not self._shutdown_event.is_set():
            return
        raise DriverShutdown(
            ShutdownInterruptionFact(
                invocation_id=self.config.run_id,
                admission_boundary=admission_boundary,
                stable_id=item.stable_id if item is not None else None,
                work_id=work_id or (item.active_work_id if item is not None else None),
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

    def _record_shutdown_interruption(
        self,
        operational: JsonlLedger,
        fact: ShutdownInterruptionFact,
        reducer: CanonicalReducer,
        snapshot: IntakeSnapshot,
    ) -> DriverResult:
        """Persist one idempotent operational interruption and resumable state.

        Parameters
        ----------
        operational, reducer, snapshot:
            Locked operational writer, reducer, and active intake used for the final rebuild.
        fact:
            Exact parent-owned shutdown observation.

        Returns
        -------
        DriverResult
            Resumable shutdown disposition with no model-failure authority.
        """

        details: JsonObject = {
            "invocation_id": fact.invocation_id,
            "admission_boundary": fact.admission_boundary,
            "stable_id": fact.stable_id,
            "work_id": fact.work_id,
            "execution_identity": fact.execution_identity,
            "request_identity": fact.request_identity,
            "lease_id": fact.lease_id,
            "child_pid": fact.child_pid,
            "child_start_token": fact.child_start_token,
            "child_pgid": fact.child_pgid,
            "signal": fact.signal,
            "parent_observation": (
                dict(fact.parent_observation) if fact.parent_observation is not None else None
            ),
            "partial_receipt": (
                dict(fact.partial_receipt) if fact.partial_receipt is not None else None
            ),
        }
        operational.append(
            mint_operational_event(
                {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": self.dependencies.clock(),
                    "event_kind": OperationalEventKind.WORKER_SHUTDOWN_INTERRUPTED.value,
                    "status": OperationalEventStatus.SHUTDOWN_INTERRUPTED.value,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 0},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": details,
                },
                identity_payload=details,
            )
        )
        rebuild_state(
            self.paths.state_database,
            snapshot.root / "items.jsonl",
            self.paths.ledgers,
            context=reducer.context,
        )
        _write_driver_state(self.paths.driver_state, {"status": "interrupted:shutdown"})
        return DriverResult(
            "interrupted:shutdown",
            len(reducer.current_view),
            self._reduced,
            "shutdown",
            shutdown_interruption=fact,
        )

    # ------------------------------------------------------------------
    # Overhaul RC-1: the ``crawl`` invocation path (RULING D.1).
    # ------------------------------------------------------------------

    def _crawl_locked(self, *, after_review: bool) -> DriverResult:
        """Run one continuous-crawl invocation under the held driver lock.

        Behavior 3: the in-process preflight runs first, before any spend, and a
        refusal is a typed ``invocation-refused`` event that counts toward no
        fuse. Behavior 2: the persisted stratified order is loaded (composed on
        first use) and ``--limit`` selects its prefix. Behavior 4: the existing
        ``leg_open`` row is the invocation record (no purpose, no close
        obligation); the pipelined scheduler runs; the exit is typed
        (``work_exhausted``, a pause, a fuse raise, a shutdown). Behavior 5: the
        paced start caps author width under a new configuration identity.

        No licensing-debt guard, no carried-authority preflight, no close
        preview, no award-holds census, no stale-excluded refusal, no repeat-
        cause fuse, no close, no partition proof.
        """

        # Complexity: essential. This is the invocation's lifecycle state
        # machine (preflight -> lifecycle reconcile -> order/limit -> open ->
        # paced start -> review gates -> selection -> scheduler -> typed exit),
        # mirroring `_run_locked`'s open range; every arm is a typed exit or a
        # ledger write in a fixed order, and splitting it would scatter the
        # shutdown/pause arms that must wrap each stage.
        from menagerie.crawler.checker_dispatch import begin_checker_prompt_leg
        from menagerie.crawler.crawl_order import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            CrawlOrderError,
            crawl_scope,
            ensure_crawl_order,
        )
        from menagerie.crawler.crawl_preflight import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            InvocationRefused,
            run_invocation_preflight,
        )
        from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            open_leg_generation,
            sealed_selection_plan,
        )
        from menagerie.crawler.scheduler import run_pipelined_leg

        begin_checker_prompt_leg()
        snapshot = load_intake_snapshot(self.paths.intake_root)
        canonical_snapshot_id = f"intake-{snapshot.snapshot_sha256.removeprefix('sha256:')[:20]}"
        if snapshot.snapshot_id != canonical_snapshot_id:
            raise DriverIntegrationError(
                "intake manifest snapshot_id conflicts with its verified snapshot_sha256"
            )
        self._intake_snapshot = snapshot
        campaign_id = bound_campaign_id(self.config)
        records_root = canonical_records_root(self.paths.ledgers.models)
        intake_ids = tuple(item.stable_id for item in snapshot.items)
        # B.4 currency carve-out under the campaign ROOT: every registered
        # same-campaign rung snapshot is prior lineage, so heads recorded
        # under the rungs stay current (frozen registry read, never a door).
        lineage_snapshots, lineage_members = self._campaign_lineage(
            records_root, campaign_id, snapshot
        )
        context = build_authority_context(
            active_intake_snapshot_id=snapshot.snapshot_id,
            active_intake_snapshot_sha256=snapshot.snapshot_sha256,
            intake_rows=(item.to_dict() for item in snapshot.items),
            author_model=self.config.author_model,
            author_version=self.config.author_version,
            checker_model=self.config.checker_model,
            checker_version=self.config.checker_version,
            prior_intake_snapshots=lineage_snapshots,
            prior_intake_member_ids=lineage_members,
        )
        self._authority_context = context
        _assert_persisted_handoff_authority_available(self.paths, self.config.only_status)
        self.paths.runtime_root.mkdir(parents=True, exist_ok=True)
        with (
            JsonlLedger(
                self.paths.operational_ledger,
                OPERATIONAL_EVENT_SCHEMA_VERSION,
                lock_timeout=10.0,
            ) as operational,
            CanonicalReducer(self.paths.ledgers, context) as reducer,
        ):
            self._crawl_operational = operational
            # Behavior 3: the preflight, before any lifecycle write or spend.
            report = run_invocation_preflight(
                records_root=records_root,
                repo_root=self._crawl_repo_root(),
                runtime_root=self.paths.runtime_root,
                campaign_id=campaign_id,
                target=self.config.target,
                real_lanes=self._crawl_buys_real_sessions(),
                envs_root=self._crawl_envs_root(),
                state_root=self._crawl_optional_root(
                    os.environ.get("MENAGERIE_ENVIRONMENT_STATE_ROOT")
                ),
                tmp_root=self._crawl_optional_root(os.environ.get("MENAGERIE_TMP_ROOT")),
                now=self.dependencies.clock(),
                lock_path=self.paths.lock_path,
                disk_free=self._crawl_disk_free,
                allow_lockless=self._crawl_allow_lockless(),
            )
            if report.refusal is not None:
                self._record_invocation_refused(
                    operational, report.refusal.name, str(report.refusal.refusal), report.to_dict()
                )
                report.raise_for_refusal()
            self._crawl_preflight_census = report.to_dict()
            try:
                self._check_shutdown("lifecycle-reconciliation")
                self._restore_quarantined_environment_context(reducer, operational)
                lifecycle_result = self._reconcile_lifecycle_before_admission(
                    operational, reducer, snapshot
                )
                if lifecycle_result is not None:
                    return lifecycle_result
                rebuild_state(
                    self.paths.state_database,
                    snapshot.root / "items.jsonl",
                    self.paths.ledgers,
                    context=context,
                )
            except DriverShutdown as shutdown:
                return self._record_shutdown_interruption(
                    operational, shutdown.fact, reducer, snapshot
                )
            state = _load_driver_state(self.paths.driver_state)
            self._models_row_baseline = (
                len(tuple(scan_jsonl(self.paths.ledgers.models)))
                if self.paths.ledgers.models.exists()
                else 0
            )
            if not self.paths.ledgers.models.exists():
                self.paths.ledgers.models.parent.mkdir(parents=True, exist_ok=True)
                self.paths.ledgers.models.touch()
            self._reconcile_orphaned_legs(operational)
            # Behavior 2: the order (composed once, never re-derived) and the limit.
            try:
                order, composed = ensure_crawl_order(
                    records_root,
                    campaign_id=campaign_id,
                    seed=self.config.crawl_order_seed,
                    created_at=self.dependencies.clock(),
                    expected_member_ids=frozenset(intake_ids),
                    expected_snapshot_sha256=snapshot.snapshot_sha256,
                )
            except CrawlOrderError as exc:
                self._record_invocation_refused(operational, "crawl-order", str(exc), report.to_dict())
                raise InvocationRefused("crawl-order", str(exc)) from exc
            self._record_crawl_limit(operational, campaign_id, order, composed, report.to_dict())
            scope = crawl_scope(order, self.config.crawl_limit)
            # Behavior 4: the leg_open row is the invocation record (no purpose).
            invocation_generation = open_leg_generation(
                records_root,
                campaign_id=campaign_id,
                intake_snapshot_id=snapshot.snapshot_id,
                intake_snapshot_sha256=snapshot.snapshot_sha256,
                phase=self.config.phase,
                run_id=self.config.run_id,
                opened_at=self.dependencies.clock(),
            )
            self._leg_generation = invocation_generation
            self._reconcile_paid_sessions_before_admission()
            self._record_leg_started()
            self._record_leg_config(operational)
            # Behavior 5: paced start under a new configuration identity.
            self._open_paced_start(records_root, campaign_id, context)
            self._record_scheduler_transition(operational)
            self._retry_notification_outbox(operational)
            if self._review_is_pending(operational, snapshot):
                if not after_review:
                    _write_driver_state(
                        self.paths.driver_state,
                        {**state, "status": "paused:review-checkpoint"},
                    )
                    return DriverResult(
                        "paused:review-checkpoint", len(reducer.current_view), 0, "review-checkpoint"
                    )
                self._record_review_signoff(operational, reducer, snapshot, "crawl --after-review")
                state["status"] = "running"
            elif after_review:
                raise DriverIntegrationError("--after-review requires a pending review checkpoint")
            self._handle_progress(operational, reducer.current_view, snapshot, state=state)
            if self._maybe_pause_for_review(operational, reducer.current_view, snapshot, state):
                return DriverResult(
                    "paused:review-checkpoint", len(reducer.current_view), 0, "review-checkpoint"
                )
            review_pause = self._review_terminal_backlog(snapshot, reducer, operational)
            if review_pause is not None:
                return DriverResult(
                    "paused:usage-limit", len(reducer.current_view), 0, review_pause
                )
            self._enforce_wake_continuation_identity()
            # WS-1 over the order prefix, in ORDER order: the predicate decides
            # work inside the scope; nothing outside the scope is selected.
            items_by_id = {item.stable_id: item for item in snapshot.items}
            scoped_snapshot = replace(
                snapshot,
                items=tuple(items_by_id[stable_id] for stable_id in scope if stable_id in items_by_id),
            )
            _plan, work = self._select_work(
                scoped_snapshot,
                reducer,
                operational,
                frozenset(intake_ids),
                prior_member_ids=frozenset(context.prior_intake_member_ids),
            )
            generation = self._leg_generation
            if generation is None:  # pragma: no cover - opened above
                raise DriverIntegrationError("cannot replay a plan without an open generation")
            self._selection_plan_admitted = sealed_selection_plan(
                records_root, generation.generation_id
            )[0]
            try:
                pipelined_pause = run_pipelined_leg(self, work, reducer, operational, state)
                if pipelined_pause is not None:
                    return DriverResult(
                        "paused:usage-limit", len(reducer.current_records), 0, pipelined_pause
                    )
            except DriverPaused:
                return DriverResult(
                    "paused:review-checkpoint",
                    len(reducer.current_view),
                    self._reduced,
                    "review-checkpoint",
                )
            except DriverShutdown as shutdown:
                return self._record_shutdown_interruption(
                    operational, shutdown.fact, reducer, snapshot
                )
            try:
                self._check_shutdown("completion-admission")
            except DriverShutdown as shutdown:
                return self._record_shutdown_interruption(
                    operational, shutdown.fact, reducer, snapshot
                )
            rebuild_state(
                self.paths.state_database,
                snapshot.root / "items.jsonl",
                self.paths.ledgers,
                context=reducer.context,
            )
            current = reducer.current_view
            state.update(
                {
                    "status": "work_exhausted",
                    "last_terminal_count": len(current),
                    "crawl": {
                        "campaign_id": campaign_id,
                        "limit": self.config.crawl_limit,
                        "scope": len(scope),
                        "admitted": len(work),
                        "order_sha256": order.order_sha256,
                        "run_id": self.config.run_id,
                        "preflight": getattr(self, "_crawl_preflight_census", None),
                    },
                }
            )
            _write_driver_state(self.paths.driver_state, state)
            return DriverResult("work_exhausted", len(current), self._reduced, None)

    def _crawl_repo_root(self) -> Path:
        """Return the running tree root the invocation is attributed to."""

        return (
            self.paths.runtime_root.parent
            if self.paths.runtime_root.name == ".crawl-local"
            else Path.cwd()
        )

    def _crawl_buys_real_sessions(self) -> bool:
        """Return whether the bound lanes buy real provider sessions."""

        return isinstance(
            self.dependencies.author, (CommandAuthorLane, QueueAuthorLane)
        ) or isinstance(self.dependencies.checker, CommandCheckerLane)

    @staticmethod
    def _crawl_disk_free(path: Path) -> int:
        """Measure free bytes at ``path`` (nearest existing ancestor); a test seam."""

        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        return int(shutil.disk_usage(probe).free)

    def _crawl_allow_lockless(self) -> tuple[str, ...]:
        """Return the lockless-intent exemptions: the config list plus the env list.

        The SAME vocabulary the EN-2 admission door reads
        (``config.allow_lockless_intents``), widened by the launch
        environment's ``MENAGERIE_ALLOW_LOCKLESS`` (comma-separated), which is
        the retired preflight's ``--allow-lockless`` spelled as an env fact.
        """

        raw = os.environ.get("MENAGERIE_ALLOW_LOCKLESS", "")
        from_env = tuple(part.strip() for part in raw.split(",") if part.strip())
        return tuple(sorted({*self.config.allow_lockless_intents, *from_env}))

    @staticmethod
    def _crawl_envs_root() -> Path:
        """Return the environment registry root the preflight arms read."""

        from menagerie.crawler.envs import DEFAULT_ENVS_ROOT

        return Path(DEFAULT_ENVS_ROOT)

    @staticmethod
    def _crawl_optional_root(raw_value: Optional[str]) -> Optional[Path]:
        """Return one optional absolute root from a launch-environment value."""

        raw = (raw_value or "").strip()
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        return candidate.resolve() if candidate.is_absolute() else None

    @staticmethod
    def _campaign_lineage(
        records_root: Path, campaign_id: str, snapshot: IntakeSnapshot
    ) -> tuple[dict[str, str], frozenset[str]]:
        """Return every registered same-campaign rung as prior lineage.

        Frozen-history read of the rung registry: each registration whose
        durable snapshot bytes still reproduce the registered identity joins
        the currency carve-out (B.4); unreadable or drifted rungs grant no
        carve-out (fail-closed), exactly like the rung-era lineage reader.
        """

        from menagerie.crawler.rung_staging import (  # noqa: PLC0415 -- frozen registry read
            RungStagingError,
            load_rung_registrations,
            rung_registry_path,
        )

        if not rung_registry_path(records_root).is_file():
            return {}, frozenset()
        try:
            registrations = load_rung_registrations(records_root)
        except RungStagingError:
            return {}, frozenset()
        snapshots: dict[str, str] = {}
        members: set[str] = set()
        for row in registrations:
            if row.campaign_id != campaign_id or row.snapshot_id == snapshot.snapshot_id:
                continue
            try:
                rung_snapshot = load_intake_snapshot(records_root / "intake" / row.snapshot_id)
            except (OSError, ValueError):
                continue
            if rung_snapshot.snapshot_sha256 != row.snapshot_sha256:
                continue
            snapshots[row.snapshot_id] = row.snapshot_sha256
            members.update(item.stable_id for item in rung_snapshot.items)
        return snapshots, frozenset(members)

    def _crawl_event_base(self) -> JsonObject:
        """Return the shared operational-event envelope for crawl facts."""

        return {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 0},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
        }

    def _record_invocation_refused(
        self,
        operational: JsonlLedger,
        arm: str,
        detail: str,
        census: Mapping[str, Any],
    ) -> None:
        """Append the typed ``invocation-refused`` event (behavior 3; no fuse)."""

        operational.append(
            mint_operational_event(
                {
                    **self._crawl_event_base(),
                    "event_kind": OperationalEventKind.INVOCATION_REFUSED.value,
                    "status": OperationalEventStatus.INVOCATION_REFUSED.value,
                    "details": {
                        "campaign_id": self.config.campaign_id,
                        "arm": arm,
                        "detail": detail,
                        "preflight": dict(census),
                        "invocation_origin": self.config.invocation_origin.value,
                    },
                }
            )
        )

    def _record_crawl_limit(
        self,
        operational: JsonlLedger,
        campaign_id: str,
        order: Any,
        composed: bool,
        census: Mapping[str, Any],
    ) -> None:
        """Record a set or raised limit; refuse a narrowed one typed (behavior 2)."""

        from menagerie.crawler.crawl_preflight import InvocationRefused

        previous: Optional[int] = None
        previous_seen = False
        for row in scan_jsonl(self.paths.operational_ledger):
            if row.get("event_kind") != OperationalEventKind.CRAWL_LIMIT.value:
                continue
            details = row.get("details")
            if not isinstance(details, Mapping) or details.get("campaign_id") != campaign_id:
                continue
            previous_seen = True
            raw = details.get("limit")
            previous = int(raw) if isinstance(raw, int) else None
        limit = self.config.crawl_limit
        if previous_seen and limit is not None and (previous is None or limit < previous):
            detail = (
                f"crawl limit {limit} narrows the recorded limit "
                f"{'unbounded' if previous is None else previous}; widening only raises N"
            )
            self._record_invocation_refused(operational, "crawl-limit", detail, census)
            raise InvocationRefused("crawl-limit", detail)
        if previous_seen and limit == previous:
            return
        operational.append(
            mint_operational_event(
                {
                    **self._crawl_event_base(),
                    "event_kind": OperationalEventKind.CRAWL_LIMIT.value,
                    "status": OperationalEventStatus.HEALTHY.value,
                    "details": {
                        "campaign_id": campaign_id,
                        "limit": limit,
                        "previous_limit": previous if previous_seen else None,
                        "first": not previous_seen,
                        "order_sha256": order.order_sha256,
                        "order_composed_now": bool(composed),
                        "order_member_count": len(order.members),
                    },
                }
            )
        )

    def _open_paced_start(
        self, records_root: Path, campaign_id: str, context: AuthorityContext
    ) -> None:
        """Cap the author width while a new configuration identity is paced."""

        from menagerie.crawler.configuration_canary import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            ConfigurationIdentity,
            evaluate_paced_start,
            ledger_row_count,
        )

        tree = observed_repo_tree(self._crawl_repo_root())
        identity = ConfigurationIdentity(
            tree_sha=tree.get("tree_sha"),
            author_model=self.config.author_model,
            checker_model=self.config.checker_model,
            author_prompt_identity=str(context.author_prompt_identity),
            checker_prompt_identity=str(context.checker_prompt_identity),
        )
        paced = evaluate_paced_start(
            records_root,
            campaign_id=campaign_id,
            identity=identity,
            current_counts={
                "models_rows": ledger_row_count(self.paths.ledgers.models),
                "gates_rows": ledger_row_count(self.paths.ledgers.gates),
                "spend_rows": ledger_row_count(self.paths.paid_session_ledger),
            },
            run_id=self.config.run_id,
            now=self.dependencies.clock(),
        )
        self._paced_start = paced
        self._paced_start_requested_width = int(self.config.author_concurrency)
        if paced is not None and self.config.author_concurrency > paced.cap:
            self.config = replace(self.config, author_concurrency=paced.cap)

    def _paced_start_tick(self) -> None:
        """Lift the paced-start cap once 20 new terminals exist (scheduler hook)."""

        paced = getattr(self, "_paced_start", None)
        if paced is None:
            return
        from menagerie.crawler.configuration_canary import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            build_canary_census,
            ledger_row_count,
            record_configuration_canary,
        )

        new_terminals = ledger_row_count(self.paths.ledgers.models) - int(
            paced.baselines.get("models_rows", 0)
        )
        if new_terminals < paced.required_terminals:
            return
        fuse_state = "not-armed"
        describe = getattr(self, "_cross_model_storm_fuse", None)
        if callable(describe):
            try:
                fuse_state = str(describe().describe())
            except Exception:  # noqa: BLE001 -- census disclosure, never a fault
                fuse_state = "unobservable"
        census = build_canary_census(
            models_path=self.paths.ledgers.models,
            gates_path=self.paths.ledgers.gates,
            spend_path=self.paths.paid_session_ledger,
            baselines=paced.baselines,
            storm_fuse_state=fuse_state,
        )
        records_root = canonical_records_root(self.paths.ledgers.models)
        campaign_id = bound_campaign_id(self.config)
        row = record_configuration_canary(
            records_root,
            campaign_id=campaign_id,
            paced=paced,
            census=census,
            run_id=self.config.run_id,
            now=self.dependencies.clock(),
        )
        requested = int(getattr(self, "_paced_start_requested_width", self.config.author_concurrency))
        operational = getattr(self, "_crawl_operational", None)
        if operational is not None:
            operational.append(
                mint_operational_event(
                    {
                        **self._crawl_event_base(),
                        "event_kind": OperationalEventKind.CONFIGURATION_CANARY.value,
                        "status": OperationalEventStatus.HEALTHY.value,
                        "details": {
                            "campaign_id": campaign_id,
                            "identity": paced.identity.to_dict(),
                            "identity_sha256": paced.identity.digest,
                            "census": census,
                            "canary_row_sha256": str(row["row_sha256"]),
                            "author_width_capped": paced.cap,
                            "author_width_lifted_to": requested,
                        },
                    }
                )
            )
        self._paced_start = None
        self.config = replace(self.config, author_concurrency=requested)

    def _run_locked(self, *, after_review: bool) -> DriverResult:
        """Run while holding both the process lock and canonical reducer locks."""

        if self.config.invocation_shape == "crawl":
            # Overhaul RC-1 (RULING D.1): the continuous-crawl invocation.
            return self._crawl_locked(after_review=after_review)
        # BL-8 (f): the checker prompt pin is LEG-scoped by contract ("rotation
        # is a leg-boundary act"); opening the scope here makes an in-process
        # relaunch behave exactly like the fresh process the contract assumes,
        # while the mid-leg rotation tripwire stays armed for this whole leg.
        from menagerie.crawler.checker_dispatch import begin_checker_prompt_leg

        begin_checker_prompt_leg()
        snapshot = load_intake_snapshot(self.paths.intake_root)
        canonical_snapshot_id = f"intake-{snapshot.snapshot_sha256.removeprefix('sha256:')[:20]}"
        if snapshot.snapshot_id != canonical_snapshot_id:
            raise DriverIntegrationError(
                "intake manifest snapshot_id conflicts with its verified snapshot_sha256"
            )
        path_claim = self.paths.intake_root.name
        if re.fullmatch(r"intake-[0-9a-f]{20}", path_claim) and path_claim != snapshot.snapshot_id:
            raise DriverIntegrationError(
                "intake path basename claims a conflicting canonical snapshot identity"
            )
        self._intake_snapshot = snapshot
        intake_ids = tuple(item.stable_id for item in snapshot.items)
        context = build_authority_context(
            active_intake_snapshot_id=snapshot.snapshot_id,
            active_intake_snapshot_sha256=snapshot.snapshot_sha256,
            intake_rows=(item.to_dict() for item in snapshot.items),
            author_model=self.config.author_model,
            author_version=self.config.author_version,
            checker_model=self.config.checker_model,
            checker_version=self.config.checker_version,
            prior_intake_snapshots=registered_prior_intake_snapshots(
                self.paths.intake_root, snapshot.snapshot_id
            ),
            prior_intake_member_ids=registered_prior_intake_members(
                self.paths.intake_root, snapshot.snapshot_id
            ),
        )
        self._authority_context = context
        _assert_persisted_handoff_authority_available(self.paths, self.config.only_status)
        self.paths.runtime_root.mkdir(parents=True, exist_ok=True)
        with (
            JsonlLedger(
                self.paths.operational_ledger,
                OPERATIONAL_EVENT_SCHEMA_VERSION,
                # Bounded wait (T5-5): a closing sibling writer or a finishing
                # operator tool is waited out instead of hard-erroring; the
                # DriverLock already excludes a concurrent driver.
                lock_timeout=10.0,
            ) as operational,
            CanonicalReducer(
                self.paths.ledgers,
                context,
            ) as reducer,
        ):
            try:
                self._check_shutdown("lifecycle-reconciliation")
                self._restore_quarantined_environment_context(reducer, operational)
                lifecycle_result = self._reconcile_lifecycle_before_admission(
                    operational, reducer, snapshot
                )
                if lifecycle_result is not None:
                    return lifecycle_result
                rebuild_state(
                    self.paths.state_database,
                    snapshot.root / "items.jsonl",
                    self.paths.ledgers,
                    context=context,
                )
            except DriverShutdown as shutdown:
                return self._record_shutdown_interruption(
                    operational, shutdown.fact, reducer, snapshot
                )
            state = _load_driver_state(self.paths.driver_state)
            # The close-time award-custody reconciliation adjudicates exactly
            # the transitions THIS leg appends.
            self._models_row_baseline = (
                len(tuple(scan_jsonl(self.paths.ledgers.models)))
                if self.paths.ledgers.models.exists()
                else 0
            )
            # RD-2: open (or continue) the durable leg generation BEFORE any
            # admission decision. Its baselines over the full source census
            # are the evidence-window floor everything this leg's close cites
            # is verified against; a killed-and-resumed leg continues the same
            # open generation, so the floor survives interruption.
            from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
                open_leg_generation,
            )

            if not self.paths.ledgers.models.exists():
                # Materialize the ACTIVE models layout before baselining: an
                # EMPTY ledger file is the explicit statement that no awards
                # exist, and a layout may not change presence during a leg.
                self.paths.ledgers.models.parent.mkdir(parents=True, exist_ok=True)
                self.paths.ledgers.models.touch()
            # W6: the boot reconciler closes every orphaned leg span (a
            # leg_started no leg_ended references -- a kill -9, a power cut)
            # as orphaned_uncontained BEFORE a new leg opens, so the
            # OB-M12 census stays exact. We hold the driver lock, so any
            # unreferenced span provably belongs to a dead process.
            self._reconcile_orphaned_legs(operational)
            # SD-2: leg open refuses on outstanding licensing debt / stale
            # rotated axes, BEFORE the generation opens and before any
            # admission or spend -- R9's rung-END refusal converted to a
            # leg-START refusal. DX-24's probe alerts; this arm refuses. The
            # explicit-ack arm (a durable acknowledgment naming the exact
            # census digest) lets the operator continue deliberately.
            from menagerie.crawler.leg_licensing_guard import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
                LegLicensingDebtError,
                assert_leg_open_licensed,
            )

            repo_root = Path.cwd()
            try:
                assert_leg_open_licensed(
                    canonical_records_root(self.paths.ledgers.models),
                    campaign_id=bound_campaign_id(self.config),
                    repo_root=repo_root if (repo_root / ".git").exists() else None,
                )
            except LegLicensingDebtError as debt:
                raise LegStartRefused(
                    StopClass.LICENSING_DEBT_UNACKED.value, str(debt)
                ) from debt
            # D-27 (LC-16): carried-authority preflight -- every authority the
            # predecessor close carried (live parks, scheduler markers, exact
            # origin rows) must STILL be live before this leg spends anything.
            # The close-side refusal already exists; this moves detection from
            # leg END to leg START. One well-meant park void bricks the next
            # close (12 carries at 200a); dispose carries by RELEASE, never
            # VOID (B2-X-O10 -- the FREEZE this preflight retires).
            from menagerie.crawler.partition_proof import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
                carried_authority_preflight,
            )

            carry_findings = carried_authority_preflight(
                canonical_records_root(self.paths.ledgers.models),
                campaign_id=bound_campaign_id(self.config),
                # A voided park over a member THIS leg can select and
                # re-drive is the sanctioned operator drain, never a brick.
                selectable_member_ids=(item.stable_id for item in snapshot.items),
            )
            if carry_findings:
                raise LegStartRefused(
                    StopClass.PARTITION_CONTRADICTION.value,
                    "carried-authority preflight refuses leg start: "
                    + "; ".join(carry_findings),
                )
            # Shape 4 (Sol contract point 1): the declared purpose is
            # content-addressed into the durable leg_open row BEFORE the
            # first ledger mutation and is immutable for the generation --
            # open_leg_generation refuses a resume whose declared purpose
            # differs from the open row's (either direction), so a
            # mis-declared production resume can never reclassify an epoch.
            self._leg_generation = open_leg_generation(
                canonical_records_root(self.paths.ledgers.models),
                campaign_id=bound_campaign_id(self.config),
                intake_snapshot_id=snapshot.snapshot_id,
                intake_snapshot_sha256=snapshot.snapshot_sha256,
                phase=self.config.phase,
                run_id=self.config.run_id,
                opened_at=self.dependencies.clock(),
                leg_purpose=self.config.leg_purpose,
            )
            # D-62: the durable operator HOLD sentinel is read at leg start,
            # generation-bound, before ANY admission or spend -- and ambiguity
            # fails closed (an unreadable sentinel holds like an active one).
            from menagerie.crawler.pause_authority import evaluate_sprint_pause

            hold = evaluate_sprint_pause(
                canonical_records_root(self.paths.ledgers.models),
                now=self.dependencies.clock(),
            )
            if hold.held:
                raise LegStartRefused(
                    StopClass.HOLD_SENTINEL_PRESENT.value,
                    "sprint-pause hold refuses leg start for generation "
                    f"{self._leg_generation.generation_id} ({hold.decision}: "
                    f"{hold.detail or 'operator hold'}). The durable sentinel at "
                    "<records>/control/sprint-pause.json governs launch, wake, "
                    "and paid admission (orchestrator-owned, D-62); clear it or "
                    "let it expire with default_action=resume",
                )
            # PM-5 (W6): the repeat-cause fuse. K identical-cause zero-delta
            # leg deaths in a row refuse the next leg without an explicit
            # operator acknowledgment carrying a rationale (a durable
            # operator row) -- the machine never funds a 4th identical death.
            self._check_repeat_cause_fuse(operational)
            # W12: a machine drain binds the NEXT leg to the tree carrying its
            # declared fix. Admission refuses while any drain's fixed-in sha
            # is not an ancestor of the running HEAD -- run the leg only after
            # the fix is actually merged (fail-closed outside a git repo).
            drain_ledger_rows = load_scheduler_quarantine_ledger(
                [scheduler_quarantine_path(canonical_records_root(self.paths.ledgers.models))]
            )
            unsatisfied_pins = unsatisfied_drain_tree_pins(
                drain_ledger_rows, self.paths.runtime_root.parent
            )
            if unsatisfied_pins:
                missing = ", ".join(
                    sorted({str(pin["fixed_in_tree_sha"]) for pin in unsatisfied_pins})
                )
                raise LegStartRefused(
                    StopClass.CERTIFICATE_STALE_VS_TREE.value,
                    "scheduler-quarantine drain tree pin refuses leg start: the "
                    f"running tree lacks the declared fixed-in sha(s) {missing}. "
                    "A machine drain (W12) released its member against a named "
                    "fix; merge that fix (or append a corrected drain via "
                    "tools/scheduler_quarantines.py) before the next leg",
                )
            if self.config.environment_readiness_receipt_path is not None:
                # W15 (B3-XO-8): a production leg -- direct launch or fired
                # wake alike -- refuses LOUDLY at leg open when a required
                # secret is absent, naming the exact variable. The historical
                # behavior was a phantom campaign-wide pause: every author
                # session died as infra and read as a storm.
                from menagerie.crawler.env_census import missing_wake_secrets

                missing_secrets = missing_wake_secrets()
                if missing_secrets:
                    raise LegStartRefused(
                        StopClass.SECRETS_ABSENT_AT_WAKE.value,
                        "required secrets absent at leg open: "
                        f"{', '.join(missing_secrets)}. Export them in the "
                        "launch environment (they are never persisted) -- an "
                        "absent secret must refuse here, never surface as a "
                        "phantom campaign pause (W15/B3-XO-8)",
                    )
            self._reconcile_paid_sessions_before_admission()
            # W6: the leg's lifecycle span opens durably once every leg-start
            # refusal has passed; every _run_locked exit from here on pairs
            # it with exactly one typed leg_ended (the run() wrapper's job).
            self._record_leg_started()
            # B-09: attest the effective config BEFORE the first admission
            # decision, so a leg that refuses at the award-holds census or
            # pauses for review has still recorded what it was launched as.
            self._record_leg_config(operational)
            # D-20 (LC-7): the leg-start read-only close preview. Report-only
            # by contract -- it rides the DX-24 probe path (the exact
            # production boundary predicate, no reimplementation), triggers no
            # recovery writes, never replaces the final replay, and never
            # kills the leg: a preview failure is a disclosed degradation.
            self._emit_close_preview(operational)
            self._disclose_award_holds(operational, reducer)
            self._record_scheduler_transition(operational)
            self._retry_notification_outbox(operational)
            if self._review_is_pending(operational, snapshot):
                if not after_review:
                    _write_driver_state(
                        self.paths.driver_state,
                        {**state, "status": "paused:review-checkpoint"},
                    )
                    return DriverResult(
                        "paused:review-checkpoint",
                        len(reducer.current_view),
                        0,
                        "review-checkpoint",
                    )
                self._record_review_signoff(operational, reducer, snapshot, "resume --after-review")
                state["status"] = "running"
            elif after_review:
                raise DriverIntegrationError("--after-review requires a pending review checkpoint")

            self._handle_progress(operational, reducer.current_view, snapshot, state=state)
            if self._maybe_pause_for_review(operational, reducer.current_view, snapshot, state):
                return DriverResult(
                    "paused:review-checkpoint", len(reducer.current_view), 0, "review-checkpoint"
                )

            # Orphan-citation adoption: re-cite lawful review-held heads whose
            # in-window citation names a run that never joined the open
            # generation, BEFORE the backlog sweep and the WS-1 selection, so
            # the admission partition proof derived later this same leg sees
            # the adopting run's own citation (anything unadoptable keeps
            # refusing there exactly as before).
            self._adopt_orphan_citations(reducer, operational)
            if self._leg_generation.leg_purpose == LEG_PURPOSE_LICENSING_CLOSE:
                # Shape 4 forfeit (Sol contract points 3 + 4): a
                # licensing_close leg's only lawful ledger acts are the
                # adoption pass above and the close below. The terminal-review
                # backlog sweep is a pre-plan PAID path (it buys checker
                # sessions), so it is skipped wholesale -- its heads stay
                # eligible for the next ordinary leg, exactly as if the bound
                # had been zero. The WS-1 plan seals EMPTY before any admitted
                # work could exist; the paid-admission chokepoints refuse
                # typed under the purpose as the belt to this suspender, and
                # the close re-verifies the whole forfeit from ledger bytes.
                self._enforce_wake_continuation_identity()
                from menagerie.crawler.leg_generation import record_leg_plan

                record_leg_plan(
                    canonical_records_root(self.paths.ledgers.models),
                    campaign_id=bound_campaign_id(self.config),
                    generation_id=self._leg_generation.generation_id,
                    run_id=self.config.run_id,
                    admitted_ids=(),
                    plan_payload={
                        "leg_purpose": LEG_PURPOSE_LICENSING_CLOSE,
                        "selection": "forfeited",
                        "continuation_identity": selection_continuation_identity(
                            self.config
                        ),
                    },
                    created_at=self.dependencies.clock(),
                )
                work: tuple[WorkItem, ...] = ()
            else:
                # Terminal-review backlog: adjudicate review-owed failed terminals
                # BEFORE the WS-1 selection, so a requeue grant held
                # `unreviewed-terminal` binds in this same leg once its review
                # lands (grant state is computed exactly once, at selection time).
                review_pause = self._review_terminal_backlog(snapshot, reducer, operational)
                if review_pause is not None:
                    return DriverResult(
                        "paused:usage-limit",
                        len(reducer.current_view),
                        0,
                        review_pause,
                    )
                # WS-1: ONE selection, evaluated BEFORE any side effect, consumed
                # identically by both schedulers and the wake-resume path. Grant
                # consumption is a consequence of the persisted plan, never a
                # precondition; the dependency-currency projection is census and
                # routing data, never existence authority.
                self._enforce_wake_continuation_identity()
                _plan, work = self._select_work(
                    snapshot,
                    reducer,
                    operational,
                    frozenset(intake_ids),
                    prior_member_ids=frozenset(context.prior_intake_member_ids),
                )
            # The close's carried arm may never absorb a member ANY run in this
            # continued generation admitted to work (R9 anti-laundering).  A
            # kill/resume appends another plan row after earlier admissions may
            # already have become terminals, so the resumed in-memory plan is
            # not authoritative by itself.  Replay the durable union that the
            # partition proof will independently derive at close.
            from menagerie.crawler.leg_generation import sealed_selection_plan

            generation = self._leg_generation
            if generation is None:  # pragma: no cover - _select_work enforces this first
                raise DriverIntegrationError("cannot replay a plan without an open generation")
            self._selection_plan_admitted = sealed_selection_plan(
                canonical_records_root(self.paths.ledgers.models),
                generation.generation_id,
            )[0]
            if (
                generation.leg_purpose == LEG_PURPOSE_LICENSING_CLOSE
                and self._selection_plan_admitted
            ):
                # Sol contract point 3: the replayed union must be empty on
                # EVERY invocation, not only at close -- a resumed
                # licensing_close generation that somehow carries an earlier
                # non-empty plan row refuses before any work could run.
                raise LegPurposeError(
                    "licensing_close generation "
                    f"{generation.generation_id} replays a NON-EMPTY sealed "
                    f"selection union ({len(self._selection_plan_admitted)} "
                    "member(s)): the purpose forfeits all admission"
                )
            try:
                if self.config.scheduler == "pipelined":
                    from menagerie.crawler.scheduler import run_pipelined_leg

                    pipelined_pause = run_pipelined_leg(self, work, reducer, operational, state)
                    if pipelined_pause is not None:
                        return DriverResult(
                            "paused:usage-limit",
                            len(reducer.current_records),
                            0,
                            pipelined_pause,
                        )
                    phase_iteration: tuple = ()
                else:
                    phase_iteration = tuple(self.registry.phase_order)
                for phase in phase_iteration:
                    self._check_shutdown("phase-admission")
                    # The selection plan IS the wave admission (WS-1 clause 7):
                    # `work` already holds exactly the plan-admitted members.
                    # The award-currentness re-validation lane is DELETED --
                    # it re-authored dependency-stale runs awards each leg
                    # (the finish-saga livelock); what re-validation awards
                    # ever need under authority rotation is AW-2's
                    # recorded-basis matrix, reached only through the named
                    # re-validation modes.
                    phase_work = tuple(item for item in work if item.route.phase is phase)
                    if not phase_work:
                        continue
                    representative_work = tuple(
                        item for item in phase_work if not item.is_family_variant
                    )
                    variant_work = tuple(item for item in phase_work if item.is_family_variant)
                    for scheduled_work in (representative_work, variant_work):
                        self._check_shutdown("wave-admission")
                        if not scheduled_work:
                            continue
                        pause = self._process_scheduled_work(
                            scheduled_work,
                            reducer,
                            operational,
                            state,
                        )
                        if pause is not None:
                            return DriverResult(
                                "paused:usage-limit",
                                len(reducer.current_view),
                                0,
                                pause,
                            )
            except DriverPaused:
                return DriverResult(
                    "paused:review-checkpoint",
                    len(reducer.current_view),
                    self._reduced,
                    "review-checkpoint",
                )
            except DriverShutdown as shutdown:
                return self._record_shutdown_interruption(
                    operational, shutdown.fact, reducer, snapshot
                )
            try:
                self._check_shutdown("completion-admission")
            except DriverShutdown as shutdown:
                return self._record_shutdown_interruption(
                    operational, shutdown.fact, reducer, snapshot
                )
            rebuild_state(
                self.paths.state_database,
                snapshot.root / "items.jsonl",
                self.paths.ledgers,
                context=reducer.context,
            )
            current = reducer.current_view
            in_scope_ids = tuple(item.stable_id for item in snapshot.items)
            if self.config.phase is not None:
                in_scope_ids = tuple(
                    item.stable_id
                    for item in snapshot.items
                    if _framework_phase(item) == self.config.phase
                )
            scoped_current = {
                stable_id: current[stable_id] for stable_id in in_scope_ids if stable_id in current
            }
            # BD-1: the close reads the RAW ledger heads beside the projection.
            # Coverage is a fact about recorded revisions; dependency currency
            # is a separate fact about whether live authority can vouch for
            # them. The close asserts BOTH -- a member holding a head the
            # projection dropped is covered-and-STALE, which still refuses,
            # with the axis named instead of a phantom coverage hole.
            ledger_heads = _scoped_ledger_heads(self.paths.ledgers.models, in_scope_ids)
            stale_reasons = _scoped_stale_reasons(
                reducer.context,
                ledger_heads,
                frozenset(scoped_current),
                reducer.dependency_stale_reasons,
            )
            # The two admission valves deliberately terminalize nothing; their
            # durable ledger rows -- stashed at valve-fire time this run -- are
            # the partition's accounting for withheld models. Every in-scope
            # model must be terminal XOR machine-deferral-claimed; a model with
            # neither still raises PartitionError (rung 9b's honest tripwire).
            scoped_ids = frozenset(in_scope_ids)
            scoped_deferrals = {
                stable_id: row
                for stable_id, row in self._machine_deferral_rows.items()
                if stable_id in scoped_ids
            }
            # Park liveness comes from the DURABLE ledgers, not only from the
            # run-scoped stash the partition accounts claims through (C3-B):
            # ONE derivation, shared verbatim with `crawl status
            # --verify-partition` so the boundary gate and the close can never
            # disagree (D-17). Derived BEFORE the assertion so a stale head
            # whose LIVE PARK is durable-only (a park written this run never
            # enters the claim stash) is disclosed rather than refused: the
            # park IS the accounting, and it keeps the close out of `complete`
            # through its own suffix.
            accounting = durable_readiness_accounting(
                self,
                scoped_ids=scoped_ids,
                ledger_heads=ledger_heads,
                claimed_ids=frozenset(scoped_deferrals),
            )
            persisted_rows = accounting.persisted_rows
            live_parks = accounting.live_parks
            partition = assert_partition(
                in_scope_ids,
                scoped_current,
                unrecordable_diagnostics=self._unrecordable_diagnostics,
                machine_deferrals=scoped_deferrals,
                ledger_heads=ledger_heads,
                stale_reasons=stale_reasons,
                stale_accounted_ids=accounting.stale_accounted_ids,
            )
            assert_status_completeness(scoped_current)
            # AW-1 close-time custody assertion: reconcile every raw-head
            # transition this leg appended -- each runs -> non-runs downgrade
            # must match one exact award action (naming grant + scar receipt),
            # and every runs -> runs remint is censused for disclosure. Scalar
            # award counts cannot see one eating balanced by one remint.
            custody_census = _reconcile_award_custody(
                tuple(scan_jsonl(self.paths.ledgers.models))
                if self.paths.ledgers.models.exists()
                else (),
                baseline_row_count=self._models_row_baseline,
                records_root=canonical_records_root(self.paths.ledgers.models),
            )
            self._append_award_custody_census(operational, custody_census)
            workflows = _completion_workflows(scoped_current)
            report = completeness_report(
                in_scope_ids,
                scoped_current,
                workflow_states=workflows,
                machine_deferrals=scoped_deferrals,
                ledger_heads=ledger_heads,
                stale_reasons=stale_reasons,
                stale_accounted_ids=accounting.stale_accounted_ids,
            )
            grant_pending_rows = {
                **{
                    stable_id: scoped_deferrals[stable_id]
                    for stable_id in partition.grant_pending_ids
                    if stable_id in scoped_deferrals
                },
                **live_parks[GRANT_PENDING_DISPOSITION],
                # Award-supersession parks are review holds (AW-1): a live one
                # forces the ``-with-review-holds`` close exactly like a
                # grant-pending supersession.
                **live_parks[AWARD_PARK_DISPOSITION],
                # R2-1: selection-exclusion parks hold review-held terminals
                # too — a live one forces the same review-holds close family.
                **live_parks[SELECTION_EXCLUSION_DISPOSITION],
            }
            refusal_parked_rows = {
                **{
                    stable_id: scoped_deferrals[stable_id]
                    for stable_id in partition.refusal_parked_ids
                    if stable_id in scoped_deferrals
                },
                **live_parks[REFUSAL_PARK_DISPOSITION],
            }
            grant_pending_ids = frozenset(grant_pending_rows)
            refusal_parked_ids = frozenset(refusal_parked_rows)
            # Deferral disclosure covers BOTH accountings: a valve row claiming
            # a member with no record at all, and (BD-1) a valve row that is the
            # live accounting for a STALE-headed member's withheld re-drive. The
            # second used to be a coverage CLAIM, which is how 8 step-100 awards
            # read as `deferred:*`; coverage now comes from the head, and the row
            # keeps only its truthful half -- the deferral suffix on the close.
            deferred_accounted = sorted(
                {
                    stable_id
                    for disposition, stable_ids in (
                        *partition.machine_deferred.items(),
                        *partition.stale_deferred.items(),
                    )
                    if disposition
                    not in {
                        GRANT_PENDING_DISPOSITION,
                        AWARD_PARK_DISPOSITION,
                        SELECTION_EXCLUSION_DISPOSITION,
                        REFUSAL_PARK_DISPOSITION,
                    }
                    for stable_id in stable_ids
                }
                # A host-capability head is STILL a deferral (WS-2). Once it is minted
                # its own valve row reports `superseded` -- the record wins -- so
                # counting only valve claims would have quietly turned a
                # `complete-with-deferrals` close into a bare `complete` the day the
                # heads arrived. The suffix answers "did this close leave members this
                # host cannot run?", and the answer does not change because the
                # accounting got better. Keyed on the RAW ledger head (BD-1's
                # coverage authority), so a dependency-stale head keeps its
                # deferral suffix too.
                | {
                    stable_id
                    for stable_id, record in ledger_heads.items()
                    if str(record.get("status", {}).get("code"))
                    in HOST_CAPABILITY_DEFERRAL_STATUS_CODES
                }
            )
            # A live park of either kind is an operator decision still owed,
            # so the close is DISTINCT and never reads "complete": parked
            # supersessions publish only after the next leg consumes grants,
            # and refusal-parked models record only after their root cause is
            # fixed and a real revision lands.
            completion_status = _completion_close_status(
                complete=report.complete,
                phase=self.config.phase,
                has_deferrals=bool(deferred_accounted),
                has_review_holds=bool(grant_pending_ids),
                has_refusal_parks=bool(refusal_parked_ids),
            )
            self._resolve_active_wake_episodes(
                operational,
                resolution="campaign-completed",
            )
            state.update({"status": completion_status, "last_terminal_count": len(current)})
            # A completed partition has just PROVED every previously stranded
            # model reconciled (terminal XOR machine-deferral-claimed, or
            # assert_partition raised), so the failure projection must not
            # outlive its resolution. Rung 10's final state said
            # `complete-with-deferrals` while still carrying the leg-1 outage
            # message, `retry_exhausted`, and 41 phantom `stranded_models`. The
            # fields move into a typed `prior_failure` history block rather
            # than being silently dropped: the wreckage a resume recovered from
            # stays legible without ever reading as a live strand again.
            resolved_failure = {
                key: state.pop(key)
                for key in (
                    "exception_type",
                    "message",
                    "retry_exhausted",
                    "stranded_models",
                    "machine_deferred_models",
                )
                if key in state
            }
            if resolved_failure:
                state["prior_failure"] = {
                    **resolved_failure,
                    "reconciled_by_run_id": self.config.run_id,
                    "reconciled_at": self.dependencies.clock(),
                }
            # Inherited valve rows -- durable ledger facts from earlier runs --
            # never CLAIM a model here (claims stay run-scoped through the
            # stash, so a stale row cannot mask lost current work), but a row
            # whose model now holds a terminal is a real supersession and the
            # completion summary must disclose it. Rung 10 reported
            # `superseded_rows: []` while the file-based partition held eight,
            # because the leg-2 in-memory stash carried only its own fires.
            # Release/supersession keys on the RAW HEAD (BD-1), so a model whose
            # head went stale still discloses its consumed rows instead of
            # silently keeping them alive.
            inherited_superseded = frozenset(
                stable_id
                for stable_id, rows in persisted_rows.items()
                if stable_id in ledger_heads
                and any(
                    machine_deferral_rejection(stable_id, row) is None
                    # A park coexisting with the exact still-current revision it
                    # recorded is a LIVE pending decision, not a superseded row;
                    # it discloses as superseded only once a newer revision
                    # actually landed (for grant parks, the granted supersession;
                    # for refusal parks, the re-driven terminal).
                    and (
                        row.get("disposition")
                        not in {GRANT_PENDING_DISPOSITION, AWARD_PARK_DISPOSITION}
                        or grant_park_released(row, ledger_heads[stable_id])
                    )
                    and (
                        row.get("disposition") != REFUSAL_PARK_DISPOSITION
                        or refusal_park_released(row, ledger_heads[stable_id])
                    )
                    for row in rows
                )
            )
            # Void-on-record (T2-opus-07): a park row whose release condition
            # the ledger head has ALREADY satisfied asserts a review-held claim
            # that is no longer true. Head-keyed release makes such a row inert
            # for every derived census, and this durable disclosure keeps the
            # LEDGER itself honest so no later reader has to re-derive
            # inertness. It is disposition-invariant by construction: it voids
            # ONLY rows the shared release predicate already reports consumed,
            # so it can never release a live decision, never claim coverage,
            # and never move a close toward `complete`.
            void_released_parks(
                self.paths.ledgers.models.parent.parent,
                persisted_rows,
                ledger_heads,
                created_at=self.dependencies.clock(),
                scoped_ids=scoped_ids,
            )
            superseded_rows = sorted(partition.superseded_deferral_ids | inherited_superseded)
            if deferred_accounted or superseded_rows:
                state["machine_deferrals"] = {
                    "count": len(deferred_accounted),
                    "by_disposition": {
                        disposition: sorted(stable_ids)
                        for disposition, stable_ids in partition.machine_deferred.items()
                        if disposition
                        not in {
                            GRANT_PENDING_DISPOSITION,
                            AWARD_PARK_DISPOSITION,
                            REFUSAL_PARK_DISPOSITION,
                        }
                    },
                    # Segregated, never merged into the claim census: these
                    # members hold a RECORD, and their row defers the re-drive.
                    "stale_deferred_by_disposition": {
                        disposition: sorted(stable_ids)
                        for disposition, stable_ids in partition.stale_deferred.items()
                    },
                    # Resolved from the stash, falling back to the durable row,
                    # for the same reason the review-holds block is built from
                    # its disclosure rows: an id the stash has no row for used to
                    # raise the count and then vanish from the block a human
                    # reads, so `count` and this list could disagree.
                    "deferred": [
                        disclosure
                        for disclosure in (
                            _deferral_disclosure(
                                stable_id,
                                scoped_deferrals.get(stable_id),
                                persisted_rows.get(stable_id, ()),
                                ledger_heads.get(stable_id),
                            )
                            for stable_id in deferred_accounted
                        )
                        if disclosure is not None
                    ],
                    "superseded_rows": superseded_rows,
                }
            # The review-holds block mirrors the live parks exactly: a clean
            # close (every park consumed or none ever fired) clears it, so a
            # stale block can never outlive the operator decision it awaited.
            state.pop("review_holds", None)
            if grant_pending_ids:
                # Built from the disclosure rows, never from a stash lookup: a
                # park id the stash has no row for used to raise the suffix and
                # then silently vanish from the block a human reads.
                held_rows = [
                    grant_pending_rows[stable_id] for stable_id in sorted(grant_pending_ids)
                ]
                state["review_holds"] = {
                    "count": len(held_rows),
                    "pending": [
                        {
                            "stable_id": row["stable_id"],
                            "work_id": row["work_id"],
                            "blocked_parent_revision": row["pending"]["blocked_parent_revision"],
                            "stage": row["pending"]["blocked_stage"],
                            "pending_status_code": row["pending"]["pending_status_code"],
                            "row_sha256": row["row_sha256"],
                            "recheck_hint": row["recheck_hint"],
                        }
                        for row in held_rows
                    ],
                }
            # The refusal-parks block mirrors the live parks exactly, like the
            # review-holds block: a clean close (every park released or none
            # ever fired) clears it, so a stale block can never outlive the
            # operator decision it awaited.
            state.pop("refusal_parks", None)
            if refusal_parked_ids:
                parked_rows = [
                    refusal_parked_rows[stable_id] for stable_id in sorted(refusal_parked_ids)
                ]
                state["refusal_parks"] = {
                    "count": len(parked_rows),
                    "pending": [
                        {
                            "stable_id": row["stable_id"],
                            "work_id": row["work_id"],
                            "refusal_class": row["refusal"]["refusal_class"],
                            "record_error": row["refusal"]["record_error"],
                            "intended_status_code": row["refusal"]["intended_status_code"],
                            "blocked_parent_revision": row["refusal"]["blocked_parent_revision"],
                            "row_sha256": row["row_sha256"],
                            "recheck_hint": row["recheck_hint"],
                        }
                        for row in parked_rows
                    ],
                }
            # The immutable close attestation (T1-sol-02): ONE durable,
            # content-addressed statement of what closed, over which ledger
            # bytes, with which partition arms and which currency census. The
            # next rung's staging consumes it and re-verifies those bytes;
            # interruption JSON never can. Appended AFTER the park voids so the
            # ledger identities it attests are the ones a later reader
            # re-observes, and the state file then carries its digest.
            # The sealed PartitionProofV2 (RD-1 + RD-2): derived from the
            # pinned bytes alone -- window from the leg-open baselines, every
            # member cited exactly -- AFTER the park voids, so the bytes it
            # seals are the ones the census attests and a later gate
            # re-observes. The disclosure arms below ride along and can never
            # improve the decision.
            from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
                census_identities,
            )
            from menagerie.crawler.partition_proof import (  # noqa: PLC0415
                CARRIED,
                adjudicated_here,
                attributed_here,
                carried,
                continuing_here,
                derive_partition_proof,
                proved_here,
                withheld_here,
                write_partition_proof,
            )

            records_root = canonical_records_root(self.paths.ledgers.models)
            generation = self._leg_generation
            if generation is None:  # pragma: no cover -- run() always opens one
                raise DriverIntegrationError("cannot close a leg whose generation was never opened")
            if generation.leg_purpose == LEG_PURPOSE_LICENSING_CLOSE:
                # Sol contract point 6: the whole forfeit re-verified from
                # ledger bytes AT CLOSE, after every write this leg could have
                # landed (adoption re-appends, release voids) -- a non-empty
                # admitted union, a paid-session reservation bound to this
                # generation, a positive disposition change, or ANY in-window
                # mutation outside the adoption/release-void families is a
                # typed close FAILURE, on first runs and resumes alike.
                forfeit_refusals = licensing_close_forfeit_refusals(
                    records_root,
                    generation_id=generation.generation_id,
                    baselines=generation.baselines,
                    paid_rows=self._paid_session_authority().rows(),
                )
                if forfeit_refusals:
                    raise LegPurposeError(
                        "licensing_close forfeit violated at close: "
                        + "; ".join(forfeit_refusals)
                    )
            proof = derive_partition_proof(
                records_root,
                campaign_id=bound_campaign_id(self.config),
                intake_snapshot_id=snapshot.snapshot_id,
                intake_snapshot_sha256=snapshot.snapshot_sha256,
                phase=self.config.phase,
                close_kind="leg",
                generation_id=generation.generation_id,
                baselines=generation.baselines,
                member_ids=in_scope_ids,
                selected_ids=sorted(self._selection_plan_admitted),
            )
            write_partition_proof(records_root, proof)
            # D-49: operator-row time authentication. Every row appended in
            # THIS leg's window to a withholding/park ledger (park rows and
            # the operator voids that share those ledgers) must not postdate
            # the close that counts it -- 200a shipped two future-dated voids
            # and nothing refused. History below the floor is corrected
            # forward-only (appended notes), never re-refused here.
            from menagerie.crawler.operator_provenance import (  # noqa: PLC0415
                OperatorProvenanceError,
                operator_act_census,
                post_dated_window_rows,
                void_only_with_live_grant_findings,
            )

            closed_at = self.dependencies.clock()
            time_findings = post_dated_window_rows(
                records_root, baselines=generation.baselines, as_of=closed_at
            )
            if time_findings:
                raise OperatorProvenanceError(
                    "close refuses counted post-dated operator rows: "
                    + "; ".join(time_findings)
                )
            # W42/OS-10: a member serviced by an in-window VOID while its
            # requeue grant stays LIVE is a receipt-less intervention beside
            # an open authorization -- typed close FAILURE, never silent.
            void_grant_findings = void_only_with_live_grant_findings(
                records_root, baselines=generation.baselines, as_of=closed_at
            )
            if void_grant_findings:
                raise OperatorProvenanceError(
                    "close refuses operator-void-only members with live grants: "
                    + "; ".join(void_grant_findings)
                )
            # DX-1/RE-1 standing close-proof clause: no terminal head with a
            # host- or machinery-owned reason may be counted by this close. The
            # mint sink already refuses these, so a finding here is a mint-sink
            # BYPASS -- a program defect caught in the leg that minted it, never
            # a fact about the model. Scoped to THIS leg's window so pre-DX-1
            # history cannot brick a close it did not write.
            from menagerie.crawler.terminal_ownership import (  # noqa: PLC0415 -- lazy: close-path only
                NonModelTerminalHeadError,
                non_model_owned_failed_heads_in_window,
                pre_model_code_runner_head_findings,
                unverified_resource_head_findings,
            )

            ownership_findings = (
                *non_model_owned_failed_heads_in_window(
                    records_root, baselines=generation.baselines
                ),
                # DX-2/RE-2 clause: no failed:runner head whose worker attempts
                # all died before any model code ran.
                *pre_model_code_runner_head_findings(
                    records_root, baselines=generation.baselines
                ),
                # DX-6/RE-6 clause: no failed:resource head on an RSS
                # observation -- those defer through the capacity valve.
                *unverified_resource_head_findings(
                    records_root, baselines=generation.baselines
                ),
            )
            if ownership_findings:
                raise NonModelTerminalHeadError(
                    "close refuses non-model-owned terminal heads minted in this leg: "
                    + "; ".join(
                        f"{finding['stable_id']} {finding['status_code']}/"
                        f"{finding['reason_code']} ({finding['owner']}-owned, "
                        f"{finding['source']}#{finding['row_index']})"
                        for finding in ownership_findings
                    )
                )
            dispositions = [
                str(entry.get("disposition")) for entry in dict(proof["members"]).values()
            ]
            from menagerie.crawler.close_object import predecessor_close_sha256
            from menagerie.crawler.rung_staging import (
                introduced_member_ids,
                load_rung_registrations,
            )

            registration = next(
                (
                    row
                    for row in load_rung_registrations(records_root)
                    if row.snapshot_id == snapshot.snapshot_id
                    and row.campaign_id == bound_campaign_id(self.config)
                ),
                None,
            )
            introduced = (
                introduced_member_ids(records_root, registration)
                if registration is not None
                else frozenset(in_scope_ids)
            )
            close_row = build_close_object(
                    # A close attestation is a durable row, so it names a REAL
                    # campaign or refuses loudly. `str(config.campaign_id)` is
                    # the extinct spelling that froze `campaign_id="None"` into
                    # 62 pilot valve-row digests (T2-12).
                    campaign_id=bound_campaign_id(self.config),
                    intake_snapshot_id=snapshot.snapshot_id,
                    intake_snapshot_sha256=snapshot.snapshot_sha256,
                    phase=self.config.phase,
                    close_kind="leg",
                    generation_id=generation.generation_id,
                    leg_open_row_sha256=generation.row_sha256,
                    sources=census_identities(records_root),
                    partition_proof_sha256=proof["partition_proof_sha256"],
                    predecessor_close_sha256=predecessor_close_sha256(
                        records_root, bound_campaign_id(self.config)
                    ),
                    run_id=self.config.run_id,
                    machine_id=self.config.machine_id,
                    closed_at=closed_at,
                    completion_status=completion_status,
                    partition_arms={
                        **_close_partition_arms(
                            partition,
                            stale_park_rows={
                                GRANT_PENDING_DISPOSITION: grant_pending_rows,
                                REFUSAL_PARK_DISPOSITION: refusal_parked_rows,
                            },
                        ),
                        # Leg-evidence disclosure: how many members THIS leg
                        # adjudicated versus inherited or withheld, so an
                        # honest no-op leg is visible at a glance.
                        "adjudicated_here": sorted(adjudicated_here(proof)),
                        "carried_count": dispositions.count(CARRIED),
                        "withheld_here": sorted(withheld_here(proof)),
                        "attributed_here": sorted(attributed_here(proof)),
                        "carried": sorted(carried(proof)),
                        "proved_here": sorted(proved_here(proof)),
                        # RULING W7 clause 6: first-class continuation
                        # disclosure -- members whose only in-window evidence
                        # is a routing marker; never proving, never licensing.
                        "continuing": sorted(continuing_here(proof)),
                        "introduced": sorted(introduced),
                    },
                    currency_census={
                        **_close_currency_census(
                            ledger_heads, scoped_current, stale_reasons
                        ),
                        "award_runs_to_non_runs": list(
                            dict(proof["award_transition_census"])["runs_to_non_runs"]
                        ),
                        "award_runs_to_runs": list(
                            dict(proof["award_transition_census"])["runs_to_runs"]
                        ),
                        "award_transition_instrumentation": dict(
                            proof["award_transition_census"]
                        )["instrumentation"],
                        "award_transition_windows": dict(
                            proof["award_transition_census"]
                        )["coverage_windows"],
                        # RULING W7 clause 6: each continuing member's cited
                        # row, kind, reason code, charged-park count, and HELD
                        # state (OPEN-5). Disclosure only, through the ONE
                        # shared derivation.
                        "continuing_members": [
                            dict(entry)
                            for entry in continuing_member_disclosures(
                                records_root, proof
                            )
                        ],
                        # D-19 (waiver doctrine part 4): every consumed requeue
                        # grant's RECOMPUTED release class rides in the close as
                        # the registered infra-terminal waiver's standing
                        # production disclosure -- infra-waiver-released is
                        # segregated from review-released, so a waived release
                        # can never be counted as reviewed. Disclosure only; it
                        # may never improve the decision.
                        "grant_release_census": {
                            name: list(members)
                            for name, members in grant_release_census_from_ledgers(
                                self.paths.ledgers
                            ).items()
                        },
                        # D-48 close-time release-path assertion (joins
                        # W42/OS-10): every live park/hold row the close
                        # accounts names its COMPUTED release path from the
                        # closed vocabulary. Disclosure only -- but a row whose
                        # path is uncomputable raises ParkReleasePathError and
                        # REFUSES the close: a park with no reachable exit may
                        # never be silently accounted.
                        "park_release_paths": park_release_path_census(
                            {
                                disposition: dict(rows)
                                for disposition, rows in live_parks.items()
                            },
                            tree_identity=current_park_tree_identity(),
                        ),
                        # W42: the operator-act census -- every in-window
                        # park/valve append and void, with derived identity
                        # spelling counts (D-53 seam). Disclosure only.
                        "operator_act_census": operator_act_census(
                            records_root, baselines=generation.baselines
                        ),
                        # W42/OS-10: every LIVE requeue grant at close with
                        # its hold_reason. Disclosure only; the void-only-
                        # beside-live-grant arm above is the refusing half.
                        "live_grants": _live_grant_disclosures(
                            self.paths.ledgers, closed_at
                        ),
                        # W12 (waiver doctrine part 4): the scheduler-quarantine
                        # release census; held-awaiting-operator-drain is the
                        # operator-owed quarantine debt disclosed at close.
                        "scheduler_drain_census": {
                            name: list(members)
                            for name, members in scheduler_drain_census_from_ledgers(
                                self.paths.ledgers
                            ).items()
                        },
                    },
                    authority_identities=_close_authority_identities(reducer.context),
                    # The close repeats the generation's declared purpose
                    # verbatim from its leg-open row (never from config), so
                    # the leg's own replay is checkable (Sol contract point 1).
                    leg_purpose=generation.leg_purpose,
            )
            # DX-22: refusals == () is asserted at close BUILD. A census that
            # cannot pass its own successor-staging disclosure gates is a
            # producer bug this leg must surface NOW, not a refusal the next
            # rung trips over after spending its open. Disposition-negative:
            # the assertion can only refuse a lying close, never bless one.
            assert_close_discloses_the_census(
                close_row,
                records_root=records_root,
                lineage_member_ids=in_scope_ids,
                introduced_member_ids=introduced,
            )
            close = append_close_object(close_object_path(records_root), close_row)
            # The disposable cursor discloses WHAT closed and under WHICH
            # authority, never the attestation's content address: that digest
            # binds the observed ledger bytes, so a killed-and-resumed campaign
            # legitimately mints a different one from an uninterrupted campaign
            # over the same models (its artifact/operational history carries the
            # disclosed kill evidence). The durability oracle compares this file
            # byte-for-byte, and a non-convergent field here would assert a
            # divergence that does not exist. The address lives in its one home,
            # the append-only close ledger, where `find_close_object` resolves
            # the governing attestation for an intake.
            state["close_object"] = {
                "completion_status": close["completion_status"],
                "verifier_identity": close["verifier_identity"],
            }
            _write_driver_state(self.paths.driver_state, state)
            return DriverResult(completion_status, len(current), self._reduced, None)

    def _resolve_active_wake_episodes(
        self,
        operational: JsonlLedger,
        *,
        resolution: str,
    ) -> None:
        """Durably resolve every active campaign wake episode before deactivation.

        Parameters
        ----------
        operational:
            Locked append-only operational ledger.
        resolution:
            Closed WakeupManager resolution label.
        """

        manager = WakeupManager(
            self.paths.wakeup_root,
            operational,
            self._wakeup_callback_argv(),
            installer=self.dependencies.wakeup_installer,
            verifier=self.dependencies.wakeup_verifier,
            deactivator=self.dependencies.wakeup_deactivator,
        )
        context = self._context(0, None)
        created_at = self.dependencies.clock()
        for active in manager.projection.active_episodes:
            manager.resolve_episode(
                active.episode.episode_id,
                resolution=resolution,
                context=context,
                created_at=created_at,
            )

    def _enforce_wake_continuation_identity(self) -> None:
        """Refuse a wake-callback resume whose selection world changed (WS-1).

        The step-100 wake argv carried no scheduler and no scope flags, so a
        fired wake would have resumed UNSCOPED under the default scheduler
        with the award-currentness lane live -- a fully composed award-eating
        path (S2-opus). The callback now persists the launching config's
        continuation identity; a resumed leg recomputes its own and compares
        byte-for-byte.
        """

        expected = self.config.wake_continuation_identity
        if expected is None:
            return
        live = selection_continuation_identity(self.config)
        if live != expected:
            raise DriverIntegrationError(
                "wake continuation identity mismatch: the resumed selection "
                f"config ({live}) is not the launching config ({expected}); "
                "a wake callback must replay the COMPLETE launching config"
            )

    def _selection_decision_of(self, stable_id: str) -> Optional[SelectionDecision]:
        """Return this leg's persisted selection decision for one member."""

        plan = getattr(self, "_selection_plan", None)
        if plan is None:
            return None
        return plan.decision_of(stable_id)

    def _select_work(
        self,
        snapshot: IntakeSnapshot,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        intake_ids: frozenset[str],
        *,
        prior_member_ids: frozenset[str] = frozenset(),
    ) -> tuple[SelectionPlan, tuple[WorkItem, ...]]:
        """Build, disclose, enforce, and apply the ONE selection plan (WS-1).

        Order is the contract: pure grant state, pure member facts, the pure
        predicate over every intake member, THEN disclosure (census +
        projection-staleness events), THEN the ordinary-mode stale-census
        refusal, and only then the side effects -- park/deferral stashes for
        excluded members, grant consumption for plan-admitted members
        (atomic with the generation mint), and the final ordered work.

        Returns
        -------
        tuple[SelectionPlan, tuple[WorkItem, ...]]
            The persisted plan and exactly the plan-admitted work items.
        """

        if self.config.redrive_manifest_path is not None:
            # D-54: re-validate the digested manifest against the ACTIVE
            # intake at leg start -- the launch-time load proves structure,
            # only the leg knows the snapshot. A stale or drifted manifest
            # refuses before any selection side effect.
            from menagerie.crawler.redrive_manifest import (
                RedriveManifestError,
                load_redrive_manifest,
            )

            try:
                manifest = load_redrive_manifest(self.config.redrive_manifest_path)
                manifest.require_snapshot(str(snapshot.snapshot_sha256))
                if manifest.targets != tuple(self.config.redrive_targets):
                    raise RedriveManifestError(
                        "re-drive manifest targets drifted from the launching "
                        "config's admitted target list"
                    )
            except RedriveManifestError as exc:
                raise DriverIntegrationError(str(exc)) from exc
        grant_state = self._requeue_grant_state(
            operational, intake_ids, prior_member_ids=prior_member_ids
        )
        raw_heads = reducer.raw_head_view
        projection = reducer.current_view
        stale_reasons = reducer.dependency_stale_reasons
        recheck_authority: dict[str, str] = {}
        if self.config.recheck:
            # RC-2: the RECHECK targets are DERIVED here from the raw heads,
            # the measured-rotations ledger, and the live authority context
            # (never typed on a command line); the census's rotation ids ride
            # into the plan as each admitted member's work authority.
            from menagerie.crawler.recheck_census import (  # noqa: PLC0415 -- lazy, as above
                recheck_authorities,
                recheck_census,
            )
            from menagerie.crawler.rotations import load_rotations  # noqa: PLC0415

            census = recheck_census(
                raw_heads,
                load_rotations(canonical_records_root(self.paths.ledgers.models)),
                reducer.context,
                scope=intake_ids,
            )
            recheck_targets = frozenset(
                str(row["stable_id"]) for row in census["needs_recheck"]
            )
            recheck_authority = recheck_authorities(census)
        else:
            recheck_targets = frozenset()
        mode, mode_targets = selection_mode_of(self.config, recheck_targets=recheck_targets)
        only_status_codes = frozenset(
            {"deferred:needs-cuda", "deferred:needs-x86"}
            if self.config.only_status == "deferred:*"
            else {self.config.only_status}
            if self.config.only_status is not None
            else ()
        )
        facts_bindings: dict[str, JsonObject] = {}
        for stable_id, facts in grant_state.items():
            active_event = facts.get("active_event")
            if facts.get("dormant") or active_event is None:
                continue
            discharge_state = (facts.get("active_discharge") or (None, None))[0]
            if discharge_state is not None:
                continue
            details = active_event["details"]
            facts_bindings[stable_id] = {
                "grant_ids": [*facts["recorded_grants"], str(details["grant_id"])],
                "work_id": str(details["new_work_id"]),
                "active": True,
                "target_intent": details.get("target_intent"),
                "attempts": details.get("attempts"),
            }
        base_work = self._ordered_work(snapshot, raw_heads, facts_bindings, refresh_ids=frozenset())
        # WS-2 leg-start preflight: state the needs-vs-actual arithmetic over
        # the FULL ordered target set (pre-exclusion -- an excluded member is
        # exactly what the disclosure is about) and refuse a leg whose every
        # member is host-capability deferred before any author purchase.
        self._disclose_host_capability_preflight(base_work, operational)
        persisted_rows = _persisted_deferral_rows(self)
        live_parks = _live_park_rows(
            persisted_rows, frozenset(item.stable_id for item in base_work), raw_heads
        )
        parked_rows: dict[str, JsonObject] = {}
        member_park_rows: dict[str, list[JsonObject]] = {}
        for rows_by_id in live_parks.values():
            for stable_id, row in rows_by_id.items():
                if _park_released_for_selection(
                    row, grant_state.get(stable_id, {}).get("live_grant")
                ):
                    continue
                parked_rows.setdefault(stable_id, row)
                member_park_rows.setdefault(stable_id, []).append(row)
        # D-48: the machine release lane's per-member readmission fact,
        # replay-derived from the governing durable rows plus the running
        # tree identity. A leg-scoped checker-lane park is not a durable
        # machinery row and never readmits within its own leg.
        park_tree_identity = current_park_tree_identity()
        park_readmissions = {
            stable_id: fact
            for stable_id, rows in member_park_rows.items()
            if stable_id not in self._checker_lane_parked_ids
            and (
                fact := derive_park_machine_readmission(
                    rows, tree_identity=park_tree_identity
                )
            )
            is not None
        }

        deferral_rows: dict[str, JsonObject] = {}
        members: list[MemberFacts] = []
        for item in base_work:
            stable_id = item.stable_id
            facts = grant_state.get(stable_id, {})
            active_event = facts.get("active_event")
            discharge_state = (facts.get("active_discharge") or (None, None))[0]
            deferral_row = _still_valid_deferral_row(self, item)
            if deferral_row is not None:
                deferral_rows[stable_id] = deferral_row
            raw_head = raw_heads.get(stable_id)
            live_grant = facts.get("live_grant")
            # WS-2: exactly two routes re-open a host-capability head, judged
            # by the ONE pure function so every selection path answers
            # identically. The host-fact route (no operator) becomes the
            # `capability_reopened` fact; the typed capability GRANT route
            # binds through the ordinary grant arm so its consumption stays
            # atomic with the plan, and it outranks a still-valid valve row --
            # it is the explicit authorization for re-running WITHOUT relying
            # on the host fact.
            head_is_host_capability = (
                raw_head is not None
                and str(raw_head.get("status", {}).get("code"))
                in HOST_CAPABILITY_DEFERRAL_STATUS_CODES
            )
            capability_grant = (
                head_is_host_capability
                and live_grant is not None
                and grant_reopens_host_capability_head(live_grant)
            )
            capability_reopened = host_capability_head_reopens(raw_head)
            active_details = (
                active_event["details"]
                if active_event is not None and discharge_state is None
                else None
            )
            # A still-valid valve row minted FOR the member's active work
            # generation is NEWER than any raw head: it claims the re-drive
            # itself (the BL-8 quarantine shape), so record-wins does not
            # apply -- letting clause 6 re-admit that generation buys work
            # the machine claim will refuse again, every leg, and leaves the
            # close nothing in-window to cite (a selected member may not be
            # closed as carried).
            deferral_claims_active_generation = (
                deferral_row is not None
                and active_details is not None
                and str(deferral_row.get("work_id")) == str(active_details.get("new_work_id"))
            )
            members.append(
                MemberFacts(
                    stable_id=stable_id,
                    raw_head=raw_head,
                    dependency_current=stable_id in projection,
                    stale_reason=stale_reasons.get(stable_id),
                    parked=(stable_id in parked_rows or stable_id in self._checker_lane_parked_ids),
                    park_machine_readmission=park_readmissions.get(stable_id),
                    # Record-wins (WS-2 B4): a member holding ANY raw head is
                    # settled by its RECORD -- clause 1's one settlement
                    # authority -- and the row-reclaim lane survives only for
                    # legacy row-only members. Stashing a head-holder's valve
                    # row would re-emit the row-reuse disclosure the head
                    # already supersedes. The one exception is a row claiming
                    # the member's own ACTIVE generation, which post-dates the
                    # head.
                    deferral_live=(
                        deferral_row is not None
                        and (raw_head is None or deferral_claims_active_generation)
                    ),
                    live_grant=live_grant,
                    grant_bindable=bool(facts.get("bindable")) or capability_grant,
                    grant_release_authority=facts.get("release_authority"),
                    active_generation=(
                        active_event["details"]
                        if active_event is not None and discharge_state is None
                        else None
                    ),
                    active_generation_discharged=(
                        active_event is not None and discharge_state is not None
                    ),
                    capability_reopened=capability_reopened,
                )
            )
        plan = build_selection_plan(
            members,
            mode=mode,
            mode_targets=mode_targets,
            only_status_codes=only_status_codes,
            recheck_authorities=recheck_authority,
        )
        self._selection_plan = plan
        generation = self._leg_generation
        if generation is None:  # pragma: no cover - run opens before selection
            raise DriverIntegrationError("cannot seal a plan without an open leg generation")
        from menagerie.crawler.leg_generation import record_leg_plan

        record_leg_plan(
            canonical_records_root(self.paths.ledgers.models),
            campaign_id=bound_campaign_id(self.config),
            generation_id=generation.generation_id,
            run_id=self.config.run_id,
            admitted_ids=sorted(plan.admitted_ids),
            plan_payload={
                **plan.to_json(),
                "continuation_identity": selection_continuation_identity(self.config),
            },
            created_at=self.dependencies.clock(),
        )
        self._disclose_selection(plan, operational)
        self._stash_selection_exclusions(plan, base_work, parked_rows, deferral_rows, operational)
        grant_scoped = mode in {
            SelectionMode.ORDINARY,
            SelectionMode.REDRIVE,
            SelectionMode.GATE_CATCHUP,
        }
        admitted_granted = (
            frozenset(
                member.stable_id
                for member in plan.members
                if member.decision is SelectionDecision.GRANTED_REDRIVE
            )
            if grant_scoped
            else frozenset()
        )
        bindings = self._apply_selection_grants(
            grant_state, admitted_granted, operational, emit_holds=grant_scoped
        )
        refresh_ids = frozenset(
            member.stable_id
            for member in plan.members
            # D-48: a park-machine-retry admission is exactly the "one FRESH
            # generation" the lane promises -- it re-generations like a named
            # re-drive, so the recording ladder adjudicates fresh work instead
            # of replaying the refused generation's history.
            if member.decision
            in {SelectionDecision.NAMED_REDRIVE, SelectionDecision.PARK_MACHINE_RETRY}
        )
        work = self._ordered_work(snapshot, raw_heads, bindings, refresh_ids=refresh_ids)
        admitted_ids = plan.admitted_ids
        work = tuple(item for item in work if item.stable_id in admitted_ids)
        self._persist_selection_plan(plan)
        return plan, work

    def _redrive_manifest_sha256(self) -> Optional[str]:
        """Return the configured re-drive manifest's self-digest, if any."""

        if self.config.redrive_manifest_path is None:
            return None
        from menagerie.crawler.redrive_manifest import (
            RedriveManifestError,
            load_redrive_manifest,
        )

        try:
            return load_redrive_manifest(self.config.redrive_manifest_path).manifest_sha256
        except RedriveManifestError:
            # Currency was already enforced fail-closed at leg start; the
            # disclosure never masks that refusal with a second one.
            return None

    def _disclose_selection(self, plan: SelectionPlan, operational: JsonlLedger) -> None:
        """Emit the leg-start selection census and staleness report; enforce ack.

        The projection-staleness event is the single cheapest diagnostic
        nobody ran during the finishing saga: {stale_count, reasons_histogram,
        sample_ids}. An ordinary leg with a nonzero stale census REFUSES to
        start without the explicit operator acknowledgment -- silently
        excluding settled-but-stale members is how a scope bug hides.
        """

        census = plan.census()
        created_at = self.dependencies.clock()
        base: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": len(plan.admitted_ids)},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
        }
        census_event: JsonObject = {
            **base,
            "event_kind": OperationalEventKind.WORK_SELECTION_CENSUS.value,
            "details": census,
        }
        census_event = mint_operational_event(census_event)
        operational.append(census_event)
        reasons_histogram: dict[str, int] = {}
        for reason in plan.stale_reasons.values():
            reasons_histogram[reason] = reasons_histogram.get(reason, 0) + 1
        staleness_event: JsonObject = {
            **base,
            "event_kind": OperationalEventKind.PROJECTION_STALENESS.value,
            "details": {
                "stale_count": len(plan.stale_reasons),
                "reasons_histogram": reasons_histogram,
                "sample_ids": sorted(plan.stale_reasons)[:10],
            },
        }
        staleness_event = mint_operational_event(staleness_event)
        operational.append(staleness_event)
        if plan.mode is SelectionMode.REDRIVE:
            # D-54 admitted-delta preview: which NAMED members selection
            # actually admitted, disclosed durably BEFORE any work generation
            # mints (this method runs before grant application and
            # _ordered_work). A named target that is NOT admitted is a fact
            # the operator composed the list against -- parks are absolute,
            # awards refuse, live deferrals withhold -- and silence here is
            # how a mis-composed wave hides.
            named = sorted(self.config.redrive_targets)
            admitted = sorted(set(named) & set(plan.admitted_ids))
            not_admitted = sorted(set(named) - set(plan.admitted_ids))
            delta_event: JsonObject = {
                **base,
                "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
                "details": {
                    "disposition": "redrive-admitted-delta",
                    "named": len(named),
                    "admitted": len(admitted),
                    "not_admitted": len(not_admitted),
                    "not_admitted_sample": not_admitted[:32],
                    "redrive_reason": self.config.redrive_reason,
                    "redrive_manifest_sha256": self._redrive_manifest_sha256(),
                },
            }
            operational.append(mint_operational_event(delta_event))
        if (
            plan.mode is SelectionMode.ORDINARY
            and census["stale_excluded"] > 0
            and self.config.invocation_shape != "crawl"
        ):
            # RC-1 (behavior 4): a crawl invocation runs NO stale-excluded
            # refusal -- stale heads stay excluded and censused above; RC-2
            # turns them into the derived needs_recheck census.
            # D-47 canonical binding: the digest covers member ids plus CLOSED
            # reason classes, never the free-form reason strings (those embed
            # live-derived identities and flapped the digest between launches
            # while the actual membership stood still).
            census_digest = stale_excluded_ack_digest(plan.stale_reasons)
            if not self.config.ack_stale_excluded:
                raise DriverIntegrationError(
                    f"{census['stale_excluded']} settled heads are dependency-stale "
                    f"(sample: {sorted(plan.stale_reasons)[:5]}); they are EXCLUDED "
                    "from selection, never silently re-driven. Review the "
                    "projection-staleness event and relaunch with "
                    "--ack-stale-excluded and the binding --stale-excluded-digest "
                    f"{census_digest}, or route them through a named "
                    "re-validation mode"
                )
            if not is_ack_digest(self.config.stale_excluded_digest):
                # D-47: a legacy free-form acknowledgment string is refused on
                # its SHAPE, with its own message -- it never reads as a mere
                # digest mismatch against a census the operator never saw.
                raise DriverIntegrationError(
                    "launch refused: --stale-excluded-digest must be the exact "
                    "census digest (sha256:<64 hex>); a legacy free-form "
                    "acknowledgment refuses (D-47). Supplied "
                    f"{self.config.stale_excluded_digest!r}; the standing "
                    f"census digest is {census_digest}"
                )
            if self.config.stale_excluded_digest != census_digest:
                # D-47: the blanket acknowledgment is retired — the ack must
                # bind exactly the standing stale-excluded set.
                raise DriverIntegrationError(
                    "launch refused: the stale-excluded acknowledgment must "
                    f"bind the exact standing census (D-47). Computed census "
                    f"digest {census_digest}; supplied "
                    f"{self.config.stale_excluded_digest!r}. Relaunch with "
                    f"--stale-excluded-digest {census_digest}"
                )

    def _stash_selection_exclusions(
        self,
        plan: SelectionPlan,
        base_work: Sequence[WorkItem],
        parked_rows: Mapping[str, JsonObject],
        deferral_rows: Mapping[str, JsonObject],
        operational: JsonlLedger,
    ) -> None:
        """Claim selection-excluded parked/deferred members for the partition.

        At the audit base these members were SCHEDULED and re-claimed (or
        re-run) through the author lane -- the park treadmill. Selection now
        excludes them before any spend, so the completion partition's
        run-scoped stash is fed here: the same rows, the same validation, the
        same ``machine-deferral-row-reused`` disclosure for valve rows,
        without buying anything.
        """

        items_by_id = {item.stable_id: item for item in base_work}
        for member in plan.members:
            if member.decision is SelectionDecision.DEFERRED:
                item = items_by_id.get(member.stable_id)
                if item is not None:
                    self._short_circuit_still_deferred(item, operational)
            elif member.decision is SelectionDecision.PARKED:
                # A member withheld for BOTH reasons (a live park AND a
                # still-holding valve refusal) keeps the documented stash
                # order: the valve claim stashes (capacity/coverage/... sit
                # BEFORE the park ledgers), and the park stays disclosed from
                # the durable ledgers by the completion arm's own
                # ``_live_park_rows`` read -- never from the stash.
                item = items_by_id.get(member.stable_id)
                if member.stable_id in deferral_rows and item is not None:
                    self._short_circuit_still_deferred(item, operational)
                    continue
                row = parked_rows.get(member.stable_id)
                if row is not None:
                    _stash_machine_deferral_row(self, member.stable_id, row)

    def _persist_selection_plan(self, plan: SelectionPlan) -> None:
        """Persist the leg's selection plan under the disposable runtime root."""

        plan_path = self.paths.runtime_root / "selection-plan.json"
        payload = {
            **plan.to_json(),
            "continuation_identity": selection_continuation_identity(self.config),
            "run_id": self.config.run_id,
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = plan_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(plan_path)

    def _wakeup_callback_argv(self) -> tuple[str, ...]:
        """Return the base recurring callback command without an episode ID."""

        repo_root = (
            self.paths.runtime_root.parent
            if self.paths.runtime_root.name == ".crawl-local"
            else Path.cwd()
        )
        # The COMPLETE launching config (WS-1 clause 7 / DELTA F5-D8): the
        # step-100 callback carried only resume/intake/target/run-id, so a
        # fired wake would have resumed UNSCOPED under the default scheduler.
        # W15: the contract is rendered from the ONE continuation vocabulary
        # (menagerie.crawler.continuation_manifest) so BOTH branches below
        # carry the identical, complete contract -- the hand-enumerated
        # per-branch flag lists are the class that dropped the readiness
        # receipt from the supervisor branch (adjudicated D1). Flags are
        # replayed verbatim and sealed by the continuation identity, which
        # the resumed leg re-derives and compares.
        from menagerie.crawler.continuation_manifest import render_continuation_flags

        if self.config.invocation_shape == "crawl":
            return self._crawl_wakeup_callback_argv(repo_root)
        selection_flags = list(render_continuation_flags(self.config))
        direct_callback = [
            sys.executable,
            "-m",
            "menagerie.crawler",
            "--repo-root",
            str(repo_root),
            "run",
            "--resume",
            "--intake",
            str(self.paths.intake_root),
            "--target",
            self.config.target,
            "--run-id",
            self.config.run_id,
            *selection_flags,
        ]
        # The readiness receipt now rides selection_flags (the continuation
        # vocabulary), so BOTH branches carry it; production legs cannot pass
        # the CLI readiness gate without it, and a callback missing it strands
        # every wake at the door.
        if self.config.campaign_config_path is not None:
            direct_callback.extend(("--campaign-config", str(self.config.campaign_config_path)))
        supervisor = repo_root / "menagerie" / "crawler" / "tools" / "crawler_supervisor.sh"
        if (
            self.config.campaign_config_path is None
            or self.config.campaign_id is None
            or not supervisor.is_file()
        ):
            return tuple(direct_callback)
        callback = [
            str(supervisor),
            "--python",
            sys.executable,
            "run",
            "--repo-root",
            str(repo_root),
            "--campaign-config",
            str(self.config.campaign_config_path),
        ]
        callback.extend(("--campaign-id", self.config.campaign_id))
        if self.config.author_queue_root is not None:
            callback.extend(("--author-queue", str(self.config.author_queue_root)))
        callback.extend(("--target", self.config.target, *selection_flags))
        return tuple(callback)

    def _crawl_wakeup_callback_argv(self, repo_root: Path) -> tuple[str, ...]:
        """Return the crawl-shaped wake callback (RC-1 behavior 4/6).

        A paused crawl wakes back INTO ``crawl`` with the same records root,
        campaign, limit and width -- never into a leg-shaped resume, which
        would carry the door machinery and a close obligation.
        """

        records_root = canonical_records_root(self.paths.ledgers.models)
        campaign_id = bound_campaign_id(self.config)
        crawl_flags: list[str] = [
            "--records-root",
            str(records_root),
            "--campaign",
            campaign_id,
            "--target",
            self.config.target,
            "--author-width",
            str(getattr(self, "_paced_start_requested_width", self.config.author_concurrency)),
            "--checker-concurrency",
            str(self.config.checker_concurrency),
        ]
        if self.config.crawl_limit is not None:
            crawl_flags.extend(("--limit", str(self.config.crawl_limit)))
        if self.config.environment_readiness_receipt_path is not None:
            crawl_flags.extend(
                ("--environment-readiness-receipt", str(self.config.environment_readiness_receipt_path))
            )
        supervisor = repo_root / "menagerie" / "crawler" / "tools" / "crawler_supervisor.sh"
        if not supervisor.is_file():
            return (
                sys.executable,
                "-m",
                "menagerie.crawler",
                "--repo-root",
                str(repo_root),
                "crawl",
                *crawl_flags,
            )
        return (
            str(supervisor),
            "--python",
            sys.executable,
            "crawl",
            "--repo-root",
            str(repo_root),
            "--campaign-id",
            campaign_id,
            *crawl_flags,
        )

    def _process_scheduled_work(
        self,
        work: Sequence[WorkItem],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        currentness_validation_only: bool = False,
    ) -> Optional[str]:
        """Author/template, gate, and execute one representative-ordered work wave.

        Parameters
        ----------
        work:
            Representatives or their later-scheduled family variants.
        reducer, operational, state:
            Current locked canonical and driver state.
        currentness_validation_only:
            Validate already-complete runs against the live environment without republishing them.

        Returns
        -------
        str | None
            Usage-pause reason, or ``None`` after the wave finishes.
        """

        self._check_shutdown("scheduled-work-admission")
        try:
            artifacts = self._ensure_authors(work, reducer, operational, state)
        except AuthorUsagePause as usage_pause:
            # The pause is already recorded and its reset wakeup scheduled. Return
            # it as an ordinary usage pause so the run loop reports
            # `paused:usage-limit` instead of failing the in-flight model.
            return usage_pause.reason
        eligible_work = tuple(item for item in work if item.stable_id in artifacts)
        # Before the checker and before the environment lane: a model this host
        # cannot instantiate should cost nothing further on this run or on any
        # resume. The author session is already spent and cannot be recovered --
        # the parameter count first exists in the authored recipe.
        eligible_work = self._admit_within_host_capacity(
            eligible_work, artifacts, operational, reducer=reducer, state=state
        )
        self._ensure_pending_run_anchors(eligible_work, artifacts, reducer, operational, state)
        eligible_work = tuple(item for item in eligible_work if item.stable_id in artifacts)
        pause = self._ensure_gates(eligible_work, artifacts, reducer, operational, state)
        eligible_work = tuple(
            item
            for item in eligible_work
            # A model the gate lane PARKED holds no accepted metadata gate and no
            # terminal: it is withheld from the environment and worker lanes
            # entirely (its durable checker-lane deferral row is the partition's
            # accounting) and requeues on the next run.
            if item.stable_id not in self._checker_lane_parked_ids
            and (
                # Mid-wave settlement reads RAW heads (WS-1 clause 1): a
                # terminal that landed during the gate round closes the item;
                # a runs head continues (pending catch-up/award), and an
                # active granted generation, a status-mode member, or a
                # NAMED re-drive member proceeds by its selection decision
                # (the plan admitted it OVER its settled head -- the old
                # terminal is exactly what the re-drive supersedes).
                item.stable_id not in reducer.raw_head_view
                or reducer.raw_head_view[item.stable_id]["status"]["kind"] == "runs"
                or item.requeue_active
                or self._selection_decision_of(item.stable_id)
                in {
                    SelectionDecision.STATUS_MODE,
                    SelectionDecision.NAMED_REDRIVE,
                    # WS-2: re-opened on a now-capable host (or by a typed
                    # capability grant) -- the environment and worker lanes
                    # are exactly where the member belongs; its settled head
                    # is what the re-run supersedes.
                    SelectionDecision.CAPABILITY_REOPENED,
                }
            )
        )
        if pause is not None:
            mechanical_work = tuple(
                item
                for item in eligible_work
                if not _fidelity_required(artifacts[item.stable_id].proposal)
            )
            self._run_environment_work(
                mechanical_work,
                artifacts,
                reducer,
                operational,
                state,
                award_run=True,
                currentness_validation_only=currentness_validation_only,
            )
            return pause
        return self._run_environment_work(
            eligible_work,
            artifacts,
            reducer,
            operational,
            state,
            award_run=True,
            currentness_validation_only=currentness_validation_only,
        )

    def _admit_within_host_capacity(
        self,
        work: Sequence[WorkItem],
        artifacts: Mapping[str, AuthorArtifact],
        operational: JsonlLedger,
        *,
        reducer: Optional[CanonicalReducer] = None,
        state: Optional[JsonObject] = None,
    ) -> tuple[WorkItem, ...]:
        """Withhold models this host provably cannot instantiate, and record why.

        This runs AFTER the proposal is authored and gated and BEFORE the
        environment lane, because that is the earliest point at which the model's
        size is knowable: an intake row carries only a name, and a parameter
        count cannot be obtained by building the model, which is the very thing
        being avoided. So the check cannot save the agentic author session -- it
        saves the environment solve, the worker run, and the OOM kill.

        A withheld model is recorded in the durable host-capacity deferral ledger AND
        (WS-2, static declared-config refusals only) given a
        ``deferred:needs-more-memory`` model head proved by the measurement itself. That
        is the truthful statement: not attempted, not failed, recoverable on bigger
        hardware -- and now visible to every head-counting consumer, which the row alone
        was not. Re-running the identical campaign on a larger host lifts the ceiling
        automatically -- the threshold is read from the machine -- so the model is
        admitted there with no code change, and the head re-opens with it.

        Parameters
        ----------
        work:
            Gated work items eligible for the environment lane.
        artifacts:
            Executable authority keyed by stable ID.
        operational:
            Locked operational ledger for the health event.

        Returns
        -------
        tuple[WorkItem, ...]
            The subset of ``work`` this host will attempt, in original order.
        """

        if not work:
            return tuple(work)
        host = host_capacity()
        admitted: list[WorkItem] = []
        deferred: list[tuple[WorkItem, CapacityAssessment]] = []
        for item in work:
            artifact = artifacts[item.stable_id]
            try:
                recipe = artifact.proposal.get("proposed_facts", {}).get("implementation", {})
            except DriverIntegrationError:
                # A terminal author result has no executable proposal and never
                # reaches the environment lane on its own account. Nothing to size.
                admitted.append(item)
                continue
            assessment = assess_model_capacity(recipe, host=host)
            if assessment.deferred:
                deferred.append((item, assessment))
            else:
                admitted.append(item)
        if not deferred:
            return tuple(admitted)
        created_at = self.dependencies.clock()
        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = capacity_deferral_path(records_root)
        for item, assessment in deferred:
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
            # WS-2: the environment-lane backstop mints the head on the same terms as
            # the pre-normalization valve. Whichever gate fires first, the member ends
            # with the same accounting -- a head plus its durable row -- so the
            # settlement census can never depend on which one won.
            _mint_host_capability_head(
                self,
                item,
                persisted,
                operational,
                artifact=artifacts.get(item.stable_id),
                reducer=reducer,
                state=state,
            )
        event = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": len(deferred)},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "host-capacity-deferred",
                "ledger": str(ledger_path.relative_to(records_root)),
                "host_physical_memory_bytes": host.physical_memory_bytes,
                "admissible_parameter_ceiling": host.admissible_parameter_ceiling,
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
                    for item, assessment in deferred
                ],
            },
        }
        # The typed per-occurrence identity covers the complete payload and
        # created_at: a later observation appends, while an exact byte replay no-ops.
        event = mint_operational_event(event)
        operational.append(event)
        return tuple(admitted)

    def _stage_author_result(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        reducer: CanonicalReducer,
        *,
        operational_events: Sequence[Mapping[str, Any]] = (),
    ) -> AuthorArtifact:
        """Place every typed author outcome and byte in private custody."""

        if artifact.author_result.binding.stable_id != item.stable_id:
            raise DriverIntegrationError("author result stable_id does not match scheduled intake")
        # The m4851 overshoot (6 staged of a 3-session allowance): buy-site
        # valves cannot see stagings the bounded repair loops mint after the
        # buy, so the allowance is enforced HERE too -- at the ONE chokepoint
        # every staged-private event flows through, BEFORE the byte lands. The
        # same owner function does the arithmetic; a replayed already-staged
        # result id is idempotent, and TWH-RED supersession-retired stagings
        # stay unchargeable through the operational rows.
        enforce_author_session_allowance(
            (*reducer.artifact_ledger.events, *operational_events),
            item,
            operational_events=operational_events,
            staging_author_result_id=str(artifact.author_result.binding.result_id),
        )
        sources = artifact.source_manifest.get("sources")
        if not isinstance(sources, list) or not sources:
            raise DriverIntegrationError("author result has no frozen source bytes")
        # Supplementary rows are staged alongside the frozen ones: their bytes
        # are ours, they are digest-pinned identically, and an excerpt is allowed
        # to cite them, so custody that held only the frozen half would leave a
        # cited source with no private byte for anything downstream to re-read.
        inputs: list[ArtifactInput] = []
        source_rows = manifest_source_rows(artifact.source_manifest)
        for index, row in enumerate(source_rows):
            cas_path = row.get("cas_path")
            digest = row.get("content_sha256")
            if not isinstance(cas_path, str) or not isinstance(digest, str):
                raise DriverIntegrationError("source row lacks exact CAS path or digest")
            path = Path(cas_path)
            if not path.is_file():
                raise DriverIntegrationError(f"private source byte is unavailable: {path}")
            source_id = str(row.get("source_id") or f"source-{index + 1}")
            origin = ArtifactOrigin(
                url=str(row.get("url") or "local:controlled-fetch"),
                revision=str(row.get("revision") or digest),
            )
            inputs.append(
                ArtifactInput(
                    content=path.read_bytes(),
                    content_sha256=digest,
                    logical_role="source",
                    logical_path=(
                        f"menagerie/crawler/source_cas/{digest.removeprefix('sha256:')}.source"
                    ),
                    source_id=source_id,
                    origin=origin,
                    fetch_recipe=json.dumps(row.get("fetch_recipe", {}), sort_keys=True),
                    evidence_ids=(),
                    media_type=str(row.get("media_type") or "application/octet-stream"),
                )
            )
        execution_proposal = (
            artifact.author_result.proposal
            if isinstance(artifact.author_result, ProposedAuthorResult)
            else artifact.author_result.handoff_execution.proposal
            if isinstance(artifact.author_result, DeferRecommendation)
            and artifact.author_result.handoff_execution is not None
            else None
        )
        if execution_proposal is not None:
            implementation = execution_proposal.get("proposed_facts", {}).get("implementation", {})
            primary = inputs[0]
            for role, field in (("code", "code_manifest"), ("patch", "patches")):
                rows = implementation.get(field, []) if isinstance(implementation, Mapping) else []
                if not isinstance(rows, list):
                    raise DriverIntegrationError(f"implementation.{field} must be a list")
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise DriverIntegrationError(f"implementation.{field} row is malformed")
                    relative = Path(str(row.get("path", "")))
                    code_path = (artifact.model_dir / relative).resolve()
                    try:
                        code_path.relative_to(artifact.model_dir.resolve())
                    except ValueError as exc:
                        raise DriverIntegrationError(
                            "authored code escapes its model root"
                        ) from exc
                    if not code_path.is_file():
                        raise DriverIntegrationError(f"authored byte is unavailable: {relative}")
                    inputs.append(
                        ArtifactInput(
                            content=code_path.read_bytes(),
                            content_sha256=str(row.get("sha256")),
                            logical_role=role,
                            # Generation-scoped by the ONE mint site: a re-driven
                            # model's new work generation publishes alongside its
                            # predecessor instead of colliding with an
                            # already-published immutable path.
                            logical_path=authored_code_logical_path(
                                logical_role=role,
                                stable_id=item.stable_id,
                                work_id=artifact.author_result.binding.work_id,
                                relative=relative.as_posix(),
                            ),
                            source_id=primary.source_id,
                            origin=primary.origin,
                            fetch_recipe=primary.fetch_recipe,
                            evidence_ids=(),
                            media_type="text/x-python"
                            if code_path.suffix == ".py"
                            else "text/plain",
                        )
                    )
        mirrors = MirrorStore(
            self.paths.runtime_root / "mirrors" / "public",
            self.paths.runtime_root / "mirrors" / "private",
            self.paths.runtime_root / "mirrors" / "local",
        )
        staged = stage_private_artifact(
            tuple(inputs),
            context=reducer.context,
            stable_id=item.stable_id,
            work_id=artifact.author_result.binding.work_id,
            author_result=artifact.author_result.binding.raw_result,
            proposal=(
                artifact.author_result.proposal
                if isinstance(artifact.author_result, ProposedAuthorResult)
                else None
            ),
            source_manifest=artifact.source_manifest,
            mirrors=mirrors,
            ledger=reducer.artifact_ledger,
            created_at=self.dependencies.clock(),
        )
        # Strand accounting: this model now holds staged published work. If the
        # campaign aborts before its record reduces, `_record_driver_failure`
        # writes it a durable stranded record instead of letting it vanish.
        with self._author_lane_state_lock:
            self._models_with_staged_work.add(item.stable_id)
        # BL14 fetch-once chokepoint: private staging is the first point at
        # which every accepted author result's bytes are durable and complete.
        # Promotion therefore happens here, independently of every later
        # terminal/award judgment. The award hook remains an idempotent replay.
        promote_author_root(
            self.paths.work_root / item.stable_id / "author",
            self.paths.ledgers.records_root,
            stable_id_override=item.stable_id,
            work_id_override=artifact.author_result.binding.work_id,
        )
        return replace(artifact, staged=staged)

    def _license_decisions(self, artifact: AuthorArtifact) -> dict[Any, LicenseDecision]:
        """Recompute one conservative decision for every staged custody claim.

        Parameters
        ----------
        artifact:
            Typed result with mandatory private custody.

        Returns
        -------
        dict[Any, LicenseDecision]
            Exact custody-claim-keyed redistribution decisions.
        """

        if artifact.staged is None:
            raise DriverIntegrationError("license decisions require private custody")
        findings: tuple[LicenseEvidence, ...] = ()
        if isinstance(artifact.author_result, ProposedAuthorResult):
            facts = artifact.proposal.get("proposed_facts", {})
            licenses = facts.get("licenses", {}) if isinstance(facts, Mapping) else {}
            code = licenses.get("code", {}) if isinstance(licenses, Mapping) else {}
            evidence = facts.get("evidence", {}) if isinstance(facts, Mapping) else {}
            excerpts = evidence.get("excerpts", []) if isinstance(evidence, Mapping) else []
            evidence_ids = set(code.get("evidence_ids", [])) if isinstance(code, Mapping) else set()
            built: list[LicenseEvidence] = []
            for excerpt in excerpts if isinstance(excerpts, list) else []:
                if (
                    not isinstance(excerpt, Mapping)
                    or excerpt.get("evidence_id") not in evidence_ids
                ):
                    continue
                try:
                    status = LicenseEvidenceStatus(str(code.get("status")))
                    built.append(
                        LicenseEvidence(
                            evidence_id=str(excerpt["evidence_id"]),
                            source_id=str(excerpt["source_id"]),
                            locator=str(code.get("locator") or excerpt.get("locator")),
                            excerpt=str(excerpt.get("text") or ""),
                            status=status,
                            spdx=(
                                str(code["spdx"])
                                if status is LicenseEvidenceStatus.DECLARED
                                else None
                            ),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            findings = tuple(built)
        return {
            claim.claim_id: recompute_license_decision(
                next(
                    obj.content_sha256
                    for obj in artifact.staged.objects
                    if obj.object_id == claim.object_id
                ),
                findings,
            )
            for claim in artifact.staged.custody_claims
        }

    def _authorize_and_publish_artifact(
        self,
        artifact: AuthorArtifact,
        model: Mapping[str, Any],
        gates: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
    ) -> None:
        """Commit reducer authority before any public or private finalization.

        Parameters
        ----------
        artifact:
            Proposed result with private custody.
        model:
            Structurally complete run candidate.
        gates:
            Durable v3 checker history.
        reducer:
            Sole authority and artifact-ledger writer.
        """

        if artifact.staged is None:
            raise DriverIntegrationError("artifact publication requires private custody")
        gate_stable_id = str(model["stable_id"])
        gate = _find_gate(gates, gate_stable_id, "metadata_batch", artifact.proposal)
        if gate is None:
            raise DriverIntegrationError("artifact publication requires an accepted metadata gate")
        gate_item = next(item for item in gate["items"] if item.get("stable_id") == gate_stable_id)
        mirrors = MirrorStore(
            self.paths.runtime_root / "mirrors" / "public",
            self.paths.runtime_root / "mirrors" / "private",
            self.paths.runtime_root / "mirrors" / "local",
        )
        try:
            authorization = reducer.authorize_publication(
                model,
                artifact.staged,
                gate_item,
                self._license_decisions(artifact),
                mirrors,
            )
        except (ArtifactBindingError, ReductionError) as exc:
            # The reducer refused BEFORE committing anything: no authorization
            # event, no public byte. That is this model's authority state, so it
            # carries a model-scoped type; post-commit publication failures below
            # keep their own types and their own resume protocol.
            raise PublicationAuthorizationRefused(exc) from exc
        canonical_root = _canonical_crawler_root(self.paths)
        publish_authorized_artifact(
            artifact.staged,
            authorization,
            reconstruction_inputs=ReconstructionInputs(
                author_result=artifact.author_result.binding.raw_result,
                proposal=artifact.proposal,
                handoff_execution=None,
                source_manifest=artifact.source_manifest,
                accepted_gate_item=gate_item,
            ),
            context=reducer.context,
            mirrors=mirrors,
            ledger=reducer.artifact_ledger,
            canonical_root=canonical_root,
            repository_root=_canonical_repo_root(canonical_root),
            created_at=self.dependencies.clock(),
        )

    def _authorize_terminal_artifact(
        self,
        artifact: AuthorArtifact,
        model: Mapping[str, Any],
        gates: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
    ) -> None:
        """Finalize an accepted terminal result while retaining private custody.

        Parameters
        ----------
        artifact, model, gates, reducer:
            Exact staged result, terminal candidate, gate history, and sole authority.
        """

        if artifact.staged is None:
            raise DriverIntegrationError("terminal finalization requires private custody")
        if isinstance(artifact.author_result, ProposedAuthorResult):
            metadata_gate = _find_gate(
                gates,
                str(model["stable_id"]),
                "metadata_batch",
                artifact.proposal,
            )
            metadata_item = (
                next(
                    item
                    for item in metadata_gate["items"]
                    if item.get("stable_id") == model.get("stable_id")
                )
                if metadata_gate is not None
                else None
            )
            if (
                metadata_gate is not None
                and metadata_item is not None
                and metadata_item.get("verdict") == "accurate"
                and metadata_item.get("integrity", {}).get("verdict") == "accurate"
                and metadata_item.get("rung_check", {}).get("verdict") == "accurate"
            ):
                # H-5 belt (redundant-by-design over the gate-lane door):
                # publication consumes only a K=2-witnessed metadata
                # acceptance -- a settled confirmed duplicate-accept row XOR a
                # distinct persisted same-question accept witness. A refusal
                # here rides the _TerminalArtifactEscape containment: the
                # record keeps its facts, only the finalization is withheld.
                # Handoff resumes inherit their origin campaign's acceptance
                # authority and are exempt (their gate lane never re-runs
                # here), like every other prior-campaign adjudication -- but
                # the exemption keys on PROVENANCE, never the artifact class
                # alone (opus F4 / fable F11: the class is assigned from
                # ``config.only_status``, so a same-campaign handoff-free
                # re-run was constructible and exempt). Only an artifact whose
                # ledger-backed final transaction carries a genuine
                # cross-campaign handoff (recorded DEFER arm + matching
                # non-empty handoff_sha256) skips the door.
                if not _handoff_inherited_acceptance_authority(
                    artifact, self._final_artifact_transactions
                ):
                    settled = accuracy_duplicate_accept_state(
                        str(metadata_gate.get("gate_id")),
                        str(model["stable_id"]),
                        load_checker_verdict_conflict_rows(
                            (
                                checker_verdict_conflict_path(
                                    self.paths.ledgers.models.parent.parent
                                ),
                            )
                        ),
                    )
                    if settled != "confirmed" and not accuracy_acceptance_witnesses(
                        metadata_gate, str(model["stable_id"]), reducer.gate_records
                    ):
                        raise DriverIntegrationError(
                            "metadata acceptance for "
                            f"{model.get('stable_id')!r} (gate "
                            f"{metadata_gate.get('gate_id')!r}) has no settled K=2 "
                            "blind confirmation and no persisted accept witness; "
                            "production acceptance depth is bound to K=2 by the "
                            "calibration receipt and publication never consumes an "
                            "unconfirmed accept (H-5)"
                        )
                self._authorize_and_publish_artifact(artifact, model, gates, reducer)
            return
        gate = next(
            (
                value
                for value in reversed(gates)
                if value.get("gate_kind") == "terminal_disposition"
                and any(
                    item.get("stable_id") == model.get("stable_id")
                    for item in value.get("items", [])
                )
            ),
            None,
        )
        if gate is None:
            raise DriverIntegrationError("terminal finalization requires its exact gate")
        gate_item = next(
            item for item in gate["items"] if item.get("stable_id") == model.get("stable_id")
        )
        terminal = gate_item.get("terminal_disposition", {})
        if not isinstance(terminal, Mapping) or terminal.get("verdict") != "accepted":
            return
        mirrors = MirrorStore(
            self.paths.runtime_root / "mirrors" / "public",
            self.paths.runtime_root / "mirrors" / "private",
            self.paths.runtime_root / "mirrors" / "local",
        )
        terminal_decisions = {
            claim_id: replace(
                decision,
                redistribution_class=RedistributionClass.RESTRICTED_PRIVATE,
                rationale=(
                    f"terminal private custody; public redistribution withheld: "
                    f"{decision.rationale}"
                ),
            )
            for claim_id, decision in self._license_decisions(artifact).items()
        }
        try:
            authorization = reducer.authorize_publication(
                model,
                artifact.staged,
                gate_item,
                terminal_decisions,
                mirrors,
                terminal=True,
            )
        except (ArtifactBindingError, ReductionError) as exc:
            # Same pre-commit boundary as the publication arm: the reducer's
            # typed refusal is this model's authority state, not the campaign's.
            raise PublicationAuthorizationRefused(exc) from exc
        canonical_root = _canonical_crawler_root(self.paths)
        publish_authorized_artifact(
            artifact.staged,
            authorization,
            reconstruction_inputs=ReconstructionInputs(
                author_result=artifact.author_result.binding.raw_result,
                proposal=None,
                handoff_execution=(
                    artifact.author_result.binding.raw_result.get("payload", {}).get(
                        "handoff_execution"
                    )
                    if isinstance(artifact.author_result, DeferRecommendation)
                    else None
                ),
                source_manifest=artifact.source_manifest,
                accepted_gate_item=gate_item,
            ),
            context=reducer.context,
            mirrors=mirrors,
            ledger=reducer.artifact_ledger,
            canonical_root=canonical_root,
            repository_root=_canonical_repo_root(canonical_root),
            created_at=self.dependencies.clock(),
        )

    def _blocked_terminal(self, result: BlockedRecommendation) -> tuple[str, Optional[str]]:
        """Map one checked ``BLOCKED`` verdict onto its truthful closed terminal.

        This is the single crossing between the author's reason vocabulary and
        the record's, and it is total: every arm returns a pair the reducer can
        accept. An unmapped pair used to surface as a ``ReductionError`` raised
        several layers below the decision that produced it, which -- for one odd
        reason string on one model -- takes the whole campaign down.

        Parameters
        ----------
        result:
            Checked ``BLOCKED`` recommendation from the terminal disposition.

        Returns
        -------
        tuple[str, str | None]
            Closed terminal status code and its record-vocabulary reason.
        """

        if result.reason_code == ACCESS_BLOCKED_REASON_CODE:
            # Access is a named capability, exactly like CUDA or a stronger tier, so
            # this is `deferred:` and not `failed:fetch/access-denied`. Routing a
            # WORLD-FACT under "our pipeline broke" would invite retry triage to sweep
            # models where nothing is retryable -- one conflation traded for another.
            # Unlike the tier deferral it is NOT campaign-conditional: no authoring
            # tier can read a paper it is not allowed to open.
            return ACCESS_BLOCKED_STATUS_CODE, None
        if result.reason_code == "needs-higher-tier":
            # Escalation is an AUTHORING-tier fact regardless of the stage the
            # author named, so it routes on the campaign, not on the stage. With
            # a higher tier available it is a promotion deferral and keeps one
            # lineage; at the top tier there is nowhere to escalate to, and that
            # is an honest author-stage failure rather than a lost verdict.
            if self.config.campaign_id in PROMOTION_SOURCE_CAMPAIGNS:
                return "deferred:needs-opus-tier", None
            return "failed:author", "needs-higher-tier"
        if result.stage == "environment":
            # EN-6: a prerequisite set consisting ENTIRELY of the closed
            # CUDA-platform class (triton/flash-attn/spconv/knn-cuda/
            # cuda-runtime) is a capability fact, not a pipeline failure --
            # ``failed:environment`` would invite retry triage on a host that
            # can never satisfy it. The decision keys on the PREREQUISITES,
            # never on the reason spelling, and is replay-derivable from the
            # verbatim tokens plus the committed platform class. A content-free
            # token (m4954's bare ``runtime-dependency``) refuses aggregation
            # and keeps the honest blocked arm below.
            aggregation = aggregate_prerequisite_tokens(result.prerequisite_ids)
            if aggregation.disposition is PrerequisiteAggregateDisposition.PLATFORM_DEFERRAL:
                return PLATFORM_PREREQUISITE_DEFERRAL_STATUS_CODE, None
        stage = result.stage
        if result.reason_code in FAILURE_REASON_CODES.get(stage, frozenset()):
            return f"failed:{stage}", result.reason_code
        if stage in FAILURE_REASON_CODES:
            # The author's reason is open checker-adjudicated vocabulary the
            # record cannot spell, but the FACT is exact: a checked prerequisite
            # blocked the author's own named stage. Recording it as
            # `failed:author`/`malformed-result` asserted an authoring defect
            # for a verdict the checker adjudicated ACCURATE end to end (pilot
            # m1777: BLOCKED at `environment`, missing-runtime-dependency
            # minkowskiengine). The stage keeps the author's naming; the
            # author's exact reason string and prerequisite IDs survive on the
            # attempt's typed cause and the terminal detail.
            #
            # AL-6 routing split (JMT-signed 2026-08-24), classified from the
            # TYPED prerequisite IDs only -- never the reason text:
            prerequisite_class = _blocked_prerequisite_class(result.prerequisite_ids)
            if prerequisite_class == "cuda":
                # The CUDA capability is the binding constraint, and
                # `deferred:*` already means exactly "terminal for this
                # campaign, recoverable given a named capability". The head
                # settles through the SAME accepted-BLOCKED capability proof
                # rule as the opus-tier/source-access pair (a BLOCKED carries
                # no handoff, so the frozen platform-handoff rule cannot and
                # must not prove it).
                return "deferred:needs-cuda", None
            suffixed = BLOCKED_PREREQUISITE_STAGE_REASON_CODES.get(
                stage, BLOCKED_PREREQUISITE_REASON_CODE
            )
            if prerequisite_class == "runtime-dependency":
                # An installable-runtime-dep prerequisite states a fact about
                # the environment ECOSYSTEM, not a defect at the stage the
                # author was standing in; the blocking stage survives in the
                # reason suffix, keeping the pair injective.
                return ENVIRONMENT_ADJACENT_STATUS_CODE, suffixed
            # Every other prerequisite class keeps the author's own stage; new
            # mints spell the stage-suffixed member (the bare member is
            # read-side legacy only -- see BLOCKED_PREREQUISITE_REASON_CODE).
            return f"failed:{stage}", suffixed
        # A stage outside the record vocabulary entirely: the result is well
        # formed and the session completed; only its naming is unrecordable.
        # That is precisely `malformed-result`, and it must not borrow
        # `session-crashed`, which asserts the session died.
        return "failed:author", "malformed-result"

    def _obtain_terminal_disposition_gate(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        pool: Optional[_CheckerSessionPool] = None,
        conflict_parking: bool = True,
    ) -> tuple[Optional[JsonObject], Optional[str]]:
        """Buy (or replay) one terminal-disposition adjudication and publish its gate.

        The ONE terminal-lane front half, shared verbatim by the fresh
        first-call/repair arms (:meth:`_route_terminal_author_result`) and the
        backlog review sweep (:meth:`_review_terminal_backlog`): the reuse
        probe, spend admission and once-per-leg calibration readiness through
        ``_retry_infrastructure_call``, the lane's deferral parks, and durable
        gate publication. Splitting the halves is what keeps the sweep the
        SAME lane rather than a second path -- the verdict consumers differ
        (the fresh arm mints a terminal, the sweep revises review marking),
        the adjudication never does.

        Parameters
        ----------
        item, artifact:
            Work identity and the typed terminal recommendation under review.
        reducer, operational:
            Locked canonical writers.
        pool:
            Optional sweep-scoped checker session pool (DX-54). A prefetched
            session under this item's terminal pool key is spent as attempt
            one through the SAME retry ladder; ``None`` keeps the historical
            fully live dispatch (the fresh first-call/repair arms).
        conflict_parking:
            Whether a same-identity verdict conflict PARKS the model (AG-4).
            ``True`` on the fresh arms: a spontaneous contradiction is checker
            instability, operator property. The review-backlog sweep passes
            ``False``: it IS the sanctioned re-adjudication of settled claims,
            and its disagreements route through the review-held supersession
            (which also releases held operator grants) -- the conflict/flip
            DISCLOSURE rows are appended either way.

        Returns
        -------
        tuple[dict | None, str | None]
            ``(gate_record, None)`` when the lane produced a durable verdict;
            ``(None, pause_reason)`` on a provider usage pause; ``(None,
            None)`` when the lane failed and parked the model
            (``_record_checker_lane_deferral`` already recorded the brake).
        """

        # AG-6: the ABSOLUTE per-lineage round ceiling, checked BEFORE the
        # reuse probe and any spend admission. Terminal rejection currently
        # terminalizes without repair rounds, so ceiling-length histories
        # accumulate across legs and generations (200a m4257: 12 all-rejected
        # rounds); past the ceiling the lineage is checker disclosure debt --
        # parked for human review, never re-bought and never re-replayed.
        ceiling_history, ceiling = gate_round_ceiling_state(
            reducer.gate_records,
            gate_kind="terminal_disposition",
            stable_id=item.stable_id,
            campaign_root_work_id=_artifact_lineage(artifact),
        )
        if len(ceiling_history) >= ceiling:
            self._record_checker_lane_deferral(
                item,
                DriverIntegrationError(
                    f"terminal-disposition gate round ceiling: {len(ceiling_history)} "
                    f"persisted rounds without acceptance (ceiling {ceiling}); "
                    "checker disclosure debt routed to human review"
                ),
                attempted_batch_sizes=tuple(
                    max(int(gate.get("batch_size", 1) or 1), 1)
                    for gate, _item in ceiling_history
                ),
                operational=operational,
                failure_site="disclosure-debt-ceiling",
                gate_kind="terminal_disposition",
                failure_class="checker-disclosure-debt",
                count_in_streak=False,
            )
            return None, None
        checker_identity = lane_adjudication_identity(
            self.dependencies.checker, "terminal_disposition", self.config
        )
        # AG-4 resume guard: the crash window between an accept's gate append
        # and its blind-confirmation settlement must resume the CONFIRMATION,
        # never buy a duplicate first session -- a second accepted row for the
        # same item bytes would violate the exact-membership rule at
        # publication. The probe binds identity, work generation, and content;
        # anything it cannot bind falls open to the historical live buy.
        binding = getattr(getattr(artifact, "author_result", None), "binding", None)
        binding_work_id = str(getattr(binding, "work_id", "") or "")
        binding_sha = str(getattr(binding, "result_sha256", "") or "")
        if checker_identity is not None and binding_work_id and binding_sha:
            conflict_ledger = checker_verdict_conflict_path(
                self.paths.ledgers.models.parent.parent
            )
            pending = unconfirmed_accepted_terminal_gate(
                reducer.gate_records,
                load_checker_verdict_conflict_rows((conflict_ledger,)),
                stable_id=item.stable_id,
                work_id=binding_work_id,
                author_result_sha256=binding_sha,
                checker_identity=checker_identity,
            )
            if pending is not None:
                return self._confirm_terminal_acceptance(
                    item, artifact, pending, operational
                )
        reused_terminal = reusable_terminal_gate(
            artifact,
            reducer.gate_records,
            checker=checker_identity,
        )
        if reused_terminal is not None:
            # The replay IS the prior row's payload (its re-append is a
            # byte-idempotent no-op), so the reuse binding names that row.
            outcome = CheckerOutcome(gate=reused_terminal)
            self._record_checker_reuse(
                operational,
                gate_kind="terminal_disposition",
                stable_ids=(item.stable_id,),
                reused_from={
                    "gate_id": str(reused_terminal["gate_id"]),
                    "gate_identity": str(reused_terminal["gate_identity"]),
                    "result_envelope_sha256": str(reused_terminal["result_envelope_sha256"]),
                },
            )
        else:
            try:
                # A pool-prefetched session is spent as attempt one; the
                # bounded retry ladder and its weather parks then run
                # unchanged on this writer thread (DX-54, the metadata-batch
                # replay shape verbatim).
                capsule = (
                    pool.claim(_terminal_pool_key(item.stable_id))
                    if pool is not None
                    else None
                )
                outcome = self._retry_infrastructure_call(
                    _replay_then_live(
                        capsule,
                        lambda: self.dependencies.checker.check_terminal(
                            artifact, self.paths.work_root, self.config
                        ),
                    ),
                    admission=("checker", item),
                    operational=operational,
                    checker_scope=self._checker_spend_scope(
                        "terminal_disposition", (artifact,)
                    ),
                    prepaid_checker_reservation=(
                        capsule.reservation if capsule is not None else None
                    ),
                )
            except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
                raise
            except CheckerMechanicalClaimError as exc:
                # W-8: a machine-refuted terminal checker claim (the m10759
                # terminal se2 shape) parks as checker-verdict-conflict --
                # never a member verdict, never a silent re-roll.
                self._record_checker_lane_deferral(
                    item,
                    exc,
                    attempted_batch_sizes=(1,),
                    operational=operational,
                    failure_site="mechanical-refutation",
                    gate_kind="terminal_disposition",
                    failure_class="checker-verdict-conflict",
                    count_in_streak=False,
                )
                return None, None
            except Exception as exc:  # noqa: BLE001 -- lane failure rendered no model verdict
                self._record_checker_lane_deferral(
                    item,
                    exc,
                    attempted_batch_sizes=(1,),
                    operational=operational,
                    failure_site="checker-dispatch",
                    gate_kind="terminal_disposition",
                )
                return None, None
        if outcome.backoff is not None:
            return None, self._pause_for_usage(outcome.backoff, operational, 1)
        if outcome.gate is None:
            failure = DriverIntegrationError("terminal checker produced no durable gate")
            self._record_checker_lane_deferral(
                item,
                failure,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="checker-dispatch",
                gate_kind="terminal_disposition",
            )
            return None, None
        try:
            gate = reducer.append_gate(outcome.gate).record
        except Exception as exc:  # noqa: BLE001 -- invalid lane output rendered no verdict
            self._record_checker_lane_deferral(
                item,
                exc,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="gate-publication",
                gate_kind="terminal_disposition",
            )
            return None, None
        return self._settle_terminal_verdict_policy(
            item, artifact, gate, reducer, operational, conflict_parking=conflict_parking
        )

    def _settle_terminal_verdict_policy(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        gate: JsonObject,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        *,
        conflict_parking: bool = True,
    ) -> tuple[Optional[JsonObject], Optional[str]]:
        """AG-4: conflict authority + duplicate-accept over one fresh TD verdict.

        Runs on every durable terminal-disposition verdict the shared front
        half publishes, in order:

        1. **Conflict authority.** The fresh verdict is compared against every
           persisted verdict for the same ``(stable_id, author_result_sha256)``.
           Same-identity disagreement (the m4976 same-prompt flip) appends a
           durable ``verdict-conflict`` row and PARKS the model as
           ``checker-verdict-conflict`` -- instability is operator property.
           Cross-identity disagreement is a lawful rotation effect: recorded
           as an ``identity-attributed-flip`` disclosure row, never parked.
        2. **Duplicate-accept.** An ACCEPT requires one more blind consistent
           accept (two sessions, same envelope, no shared context); rejections
           are never duplicated. The confirmation settles on the conflict
           ledger, so this method is idempotent across resumes.
        """

        records_root = self.paths.ledgers.models.parent.parent
        ledger_path = checker_verdict_conflict_path(records_root)
        try:
            verdict = terminal_gate_verdict(gate)
            split = terminal_verdict_disagreements(gate, reducer.gate_records)
        except CheckerVerdictConflictError as exc:
            self._record_checker_lane_deferral(
                item,
                exc,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="verdict-conflict-policy",
                gate_kind="terminal_disposition",
            )
            return None, None
        common = {
            "stable_id": item.stable_id,
            "work_id": item.active_work_id,
            "campaign_id": bound_campaign_id(self.config),
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "gate_kind": "terminal_disposition",
            "author_result_sha256": str(
                gate["items"][0]["terminal_disposition"].get("author_result_sha256")
            ),
        }
        for prior in split["flips"]:
            try:
                append_checker_verdict_conflict_row(
                    ledger_path,
                    build_identity_flip_row(
                        created_at=self.dependencies.clock(),
                        new_gate_id=str(gate.get("gate_id")),
                        prior_gate_id=str(prior.get("gate_id")),
                        new_verdict=verdict,
                        prior_verdict=terminal_gate_verdict(prior),
                        new_identity=adjudication_identity_fields(gate),
                        prior_identity=adjudication_identity_fields(prior),
                        **common,
                    ),
                )
            except CheckerVerdictConflictError:
                # A flip disclosure row that cannot bind (a half-identified
                # historical checker block) is skipped: the flip itself stays
                # derivable from the gates ledger (gate_acceptance_timeline),
                # and a lawful rotation effect never blocks a fresh verdict.
                continue
        if split["conflicts"]:
            for prior in split["conflicts"]:
                try:
                    append_checker_verdict_conflict_row(
                        ledger_path,
                        build_verdict_conflict_row(
                            created_at=self.dependencies.clock(),
                            new_gate_id=str(gate.get("gate_id")),
                            prior_gate_id=str(prior.get("gate_id")),
                            new_verdict=verdict,
                            prior_verdict=terminal_gate_verdict(prior),
                            new_identity=adjudication_identity_fields(gate),
                            prior_identity=adjudication_identity_fields(prior),
                            **common,
                        ),
                    )
                except CheckerVerdictConflictError:
                    # The PARK below is the authority; an unbuildable
                    # disclosure row (malformed prior identity) never lets
                    # the conflict itself pass unparked.
                    continue
            if not conflict_parking:
                # The review-backlog sweep IS the sanctioned re-adjudication:
                # its disagreement routes through the review-held supersession
                # (which also releases held operator grants), never a park.
                # The conflict rows above stay the durable instability census.
                return gate, None
            first = split["conflicts"][0]
            self._record_checker_lane_deferral(
                item,
                DriverIntegrationError(
                    "checker verdict conflict: fresh terminal-disposition verdict "
                    f"{verdict!r} contradicts persisted verdict "
                    f"{terminal_gate_verdict(first)!r} (gate {first.get('gate_id')!r}) "
                    "for byte-identical content under the SAME adjudication "
                    "identity (the m4976 same-prompt flip shape); instability is "
                    "routed to human review, never re-rolled"
                ),
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="verdict-conflict",
                gate_kind="terminal_disposition",
                failure_class="checker-verdict-conflict",
                count_in_streak=False,
            )
            return None, None
        if verdict != "accepted":
            # Rejections are never duplicated (B1-XQ5's spend shape doubles
            # only the bless direction, where the measured bad-pass floor is
            # 6.7% -- sol U-S1).
            return gate, None
        settled = duplicate_accept_state(
            str(gate.get("gate_id")),
            load_checker_verdict_conflict_rows((ledger_path,)),
        )
        if settled == "confirmed":
            return gate, None
        if settled is None and terminal_acceptance_witnesses(gate, reducer.gate_records):
            # Two blind consistent accepts already stand on the gates ledger
            # (this row plus a distinct same-identity, same-envelope accept
            # session -- e.g. a review-sweep re-gate). The policy is satisfied
            # by ledger inspection; a third session would be pure spend.
            return gate, None
        if settled == "conflict":
            self._record_checker_lane_deferral(
                item,
                DriverIntegrationError(
                    "duplicate-accept conflict already settled for gate "
                    f"{gate.get('gate_id')!r}; the acceptance stays parked for review"
                ),
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind="terminal_disposition",
                failure_class="checker-verdict-conflict",
                count_in_streak=False,
            )
            return None, None
        if not confirmation_sampled(gate, stable_id=item.stable_id):
            # RULING checker row: terminal K=2 is a 10 percent DETERMINISTIC
            # sample of Fable-primary accepts (constants.CHECKER_SEATS). The
            # predicate is a pure function of (member, author result content,
            # seated rate), so a resume re-derives the same answer and the
            # census recomputes who was sampled from the gates ledger alone;
            # nothing persisted steers it. Lanes without a seated partner
            # (test lanes, legacy sol-primary) confirm every accept as before.
            return gate, None
        return self._confirm_terminal_acceptance(item, artifact, gate, operational)

    def _confirm_terminal_acceptance(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        gate: JsonObject,
        operational: JsonlLedger,
    ) -> tuple[Optional[JsonObject], Optional[str]]:
        """Buy the blind duplicate-accept confirmation for one accepted verdict.

        A second LIVE session over the byte-identical envelope -- never a pool
        replay (the call deliberately takes no capsule) and never a reused
        verdict. Agreement settles ``confirmed`` and the acceptance stands;
        disagreement settles ``conflict`` and parks. The confirmation is NEVER
        appended to the gates ledger: a duplicate accepted item row would
        violate the exact-membership anti-substitution rule at publication, so
        its verdict, identity, and result-envelope digest settle durably on
        the conflict ledger, bound to the confirmed row's ``gate_id``.
        """

        ledger_path = checker_verdict_conflict_path(
            self.paths.ledgers.models.parent.parent
        )
        try:
            # Seat swap: bought on the CONFIRMATION seat (cross-model against
            # the Fable primary); the ledger's pairing rule is the authority.
            outcome = self._retry_infrastructure_call(
                lambda: buy_confirmation_session(
                    self.dependencies.checker,
                    "terminal_disposition",
                    (artifact,),
                    self.paths.work_root,
                    self.config,
                ),
                admission=("checker", item),
                operational=operational,
                checker_scope=self._checker_spend_scope(
                    "terminal_disposition", (artifact,), role=CheckerSeatRole.CONFIRMATION
                ),
            )
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
            raise
        except CheckerMechanicalClaimError as exc:
            # W-8: a confirmation session asserting machine-refuted claims is
            # checker instability; park rather than leave the acceptance
            # unconfirmed for a silent resume re-buy of the same defect.
            self._record_checker_lane_deferral(
                item,
                exc,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="mechanical-refutation",
                gate_kind="terminal_disposition",
                failure_class="checker-verdict-conflict",
                count_in_streak=False,
            )
            return None, None
        except Exception as exc:  # noqa: BLE001 -- confirmation failure renders no settlement
            self._record_checker_lane_deferral(
                item,
                exc,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind="terminal_disposition",
            )
            return None, None
        if outcome.backoff is not None:
            # The acceptance stays unconfirmed; the resume guard at the lane
            # front half resumes exactly this confirmation on the next entry.
            return None, self._pause_for_usage(outcome.backoff, operational, 1)
        confirmation = outcome.gate
        if confirmation is None:
            self._record_checker_lane_deferral(
                item,
                DriverIntegrationError(
                    "duplicate-accept confirmation produced no durable gate"
                ),
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind="terminal_disposition",
            )
            return None, None
        try:
            confirmation_verdict = terminal_gate_verdict(confirmation)
            # Same question (item + content), lawful seat pairing (same identity
            # for an unseated primary, the seated cross-model partner at the
            # same prompt epoch for a Fable primary); refuses typed otherwise.
            require_confirmation_pairing(gate, confirmation)
            row = build_duplicate_accept_row(
                first_identity=adjudication_identity_fields(gate),
                stable_id=item.stable_id,
                work_id=item.active_work_id,
                campaign_id=bound_campaign_id(self.config),
                run_id=self.config.run_id,
                machine_id=self.config.machine_id,
                created_at=self.dependencies.clock(),
                gate_kind="terminal_disposition",
                author_result_sha256=str(
                    gate["items"][0]["terminal_disposition"].get("author_result_sha256")
                ),
                gate_id=str(gate.get("gate_id")),
                gate_identity=str(gate.get("gate_identity")),
                confirmation_gate_identity=str(confirmation.get("gate_identity")),
                confirmation_verdict=confirmation_verdict,
                confirmation_identity=adjudication_identity_fields(confirmation),
                confirmation_result_envelope_sha256=str(
                    confirmation.get("result_envelope_sha256")
                ),
            )
        except CheckerVerdictConflictError as exc:
            # Divergent envelope, unreadable verdict, or a half-identified
            # confirmation: the acceptance cannot settle, fail-closed.
            self._record_checker_lane_deferral(
                item,
                exc,
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind="terminal_disposition",
            )
            return None, None
        persisted = append_checker_verdict_conflict_row(ledger_path, row)
        if persisted["status"] != "confirmed":
            self._record_checker_lane_deferral(
                item,
                DriverIntegrationError(
                    "duplicate-accept disagreement: the blind confirmation session "
                    f"rendered {confirmation_verdict!r} against an accept for gate "
                    f"{gate.get('gate_id')!r} (measured bless-direction bad-pass "
                    "floor 6.7%, sol U-S1); the acceptance parks for human review"
                ),
                attempted_batch_sizes=(1,),
                operational=operational,
                failure_site="duplicate-accept",
                gate_kind="terminal_disposition",
                failure_class="checker-verdict-conflict",
                count_in_streak=False,
            )
            return None, None
        return gate, None

    def _route_terminal_author_result(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        prior_attempts: tuple[Mapping[str, Any], ...] = (),
    ) -> Optional[str]:
        """Gate and reduce one advisory terminal author-result arm.

        Parameters
        ----------
        item, artifact:
            Scheduled intake row and privately staged typed recommendation.
        reducer, operational, state:
            Locked canonical writers and scheduler state.
        prior_attempts:
            Already-persisted ledger attempts for this model's active work
            generation. A first-call terminal has none. A terminal reached
            AFTER the worker ran -- the mode-repair generation answering
            terminally -- must carry them: the reducer independently derives
            the record's per-mode and environment proof from the attempts
            LEDGER, so a record assembled without them claims a not-applicable
            environment that contradicts the proof and refuses
            (``model environment generation contradicts its attempt proof``).
            Every row here is a ``reducer.append_attempt(...).record``, so
            nothing unvalidated enters the record through this argument.

        Returns
        -------
        str | None
            Usage-limit pause reason, if the checker is unavailable.
        """

        gate, pause = self._obtain_terminal_disposition_gate(item, artifact, reducer, operational)
        if pause is not None:
            return pause
        if gate is None:
            return None
        gate_item = next(
            value for value in gate["items"] if value.get("stable_id") == item.stable_id
        )
        pack = _terminal_checker_item(artifact)
        decision = validate_terminal_disposition_gate(
            gate,
            artifact.author_result,
            source_manifest=artifact.source_manifest,
            evidence_pack=pack["evidence_pack"],
            license_identity=str(pack["license_identity"]),
        )
        if isinstance(artifact.author_result, DeferRecommendation):
            status_code = f"deferred:needs-{artifact.author_result.platform}"
            reason_code = None
        elif isinstance(artifact.author_result, SkipRecommendation):
            status_code = artifact.author_result.status_code
            reason_code = None
        elif isinstance(artifact.author_result, BlockedRecommendation):
            status_code, reason_code = self._blocked_terminal(artifact.author_result)
        else:
            raise DriverIntegrationError("unknown terminal author-result arm")
        if not decision.accepted:
            # W13: a refused (or unverifiable) advisory terminal claim is a
            # DISPOSITION-REVIEW hold, not an accuracy-gate failure -- the
            # accuracy gate never rendered a verdict here, and minting
            # `failed:accuracy-gate` published exactly the R24 mislabel (27
            # such 200a heads with accuracy_gate.verdict=None). The authority
            # kernel derives the reason independently from the same gate
            # through the same total mapping, so the driver's candidate can
            # never contradict the proof; reduction refuses a fresh
            # accuracy-gate spelling of this shape outright.
            status_code = BLOCKED_DISPOSITION_REVIEW_STATUS_CODE
            reason_code = terminal_disposition_failure_reason(
                gate_item.get("terminal_disposition", {}).get("verdict")
            )
        elif status_code == "deferred:needs-opus-tier":
            if self._intake_snapshot is None:
                raise DriverIntegrationError("promotion routing has no active intake snapshot")
            if not isinstance(artifact.author_result, BlockedRecommendation):
                raise DriverIntegrationError("Opus-tier deferral lost its BLOCKED result")
            promotion = build_promotion_row(
                source_campaign_id=bound_campaign_id(self.config),
                snapshot=self._intake_snapshot,
                item=item.intake,
                result=artifact.author_result,
                source_manifest=artifact.source_manifest,
                created_at=artifact.author_result.binding.created_at,
            )
            append_promotion_row(
                promotion_extension_path(self.paths.ledgers.models.parent.parent),
                promotion,
            )
        # A `failed:<stage>` terminal is only reducible when a failed attempt at
        # that SAME stage witnesses it. Gate-derived stages carry their proof on
        # the gate; a checked BLOCKED recommendation carries it nowhere, so the
        # author's own typed verdict is recorded as the attempt it is. Without
        # this the arm cannot terminalize at all, and the verdict degrades into
        # whichever generic cause the surrounding handler happens to name.
        attempts: tuple[Mapping[str, Any], ...] = tuple(prior_attempts)
        blocked_summary: Optional[str] = None
        if (
            decision.accepted
            and isinstance(artifact.author_result, BlockedRecommendation)
            and status_code.startswith("failed:")
        ):
            # The author's own reason string and prerequisite IDs are open
            # vocabulary the closed `reason_code` cannot spell, so they ride
            # here: on the attempt's typed cause, and as the terminal detail
            # whenever the gate published no findings of its own (pilot m1777
            # terminalized with detail null, leaving the census only an opaque
            # fallback reason).
            blocked_summary = (
                "author returned a checked BLOCKED recommendation at stage "
                f"{artifact.author_result.stage}: {artifact.author_result.reason_code} "
                f"(prerequisites: {', '.join(artifact.author_result.prerequisite_ids)})"
            )
            # The attempt's stage is the TERMINAL's stage, which is what the
            # reducer looks for, and is not always the stage the author named:
            # an escalation with nowhere to escalate to lands on `author` even
            # when the author named `source` as its blocker.
            attempts = (
                *attempts,
                reducer.append_attempt(
                    _driver_failure_attempt(
                        item,
                        artifact,
                        status_code.removeprefix("failed:"),
                        str(reason_code),
                        AuthorBlockedPrerequisite(blocked_summary),
                        self.config,
                        diagnostics_root=_diagnostics_root_for_work_root(self.paths.work_root),
                        environment=None,
                        created_at=self.dependencies.clock(),
                    )
                ).record,
            )
            # W14 seam record (the AL-6/EN-6 typed prerequisite naming): minted
            # ONLY when EVERY typed prerequisite id translates into the closed
            # record grammar -- a partially translatable set would understate
            # the prerequisites and let the post-rotation class release on
            # incomplete witnesses, so it stays on the human lane instead.
            # The record binds the adjudicating gate and author result by
            # content address; it never gates the terminal itself.
            from menagerie.crawler.release_vocabulary import (
                append_blocked_prerequisite_record,
                blocked_prerequisite_record_path,
                build_blocked_prerequisite_record,
                prerequisite_token_from_author_id,
            )

            tokens = tuple(
                prerequisite_token_from_author_id(str(value))
                for value in artifact.author_result.prerequisite_ids
            )
            gate_result_sha = str(
                gate_item.get("terminal_disposition", {}).get("author_result_sha256")
            )
            if tokens and all(tokens) and gate_result_sha:
                append_blocked_prerequisite_record(
                    blocked_prerequisite_record_path(self.paths.ledgers.models.parent.parent),
                    build_blocked_prerequisite_record(
                        stable_id=item.stable_id,
                        work_id=item.active_work_id,
                        stage=artifact.author_result.stage,
                        prerequisite_tokens=tuple(sorted(set(tokens))),
                        gate_id=str(gate["gate_id"]),
                        author_result_sha256=gate_result_sha,
                        campaign_id=bound_campaign_id(self.config),
                        run_id=self.config.run_id,
                        machine_id=self.config.machine_id,
                        created_at=self.dependencies.clock(),
                    ),
                )
        self._terminalize(
            item,
            artifact,
            status_code,
            reason_code,
            "; ".join(decision.findings) or blocked_summary,
            attempts,
            reducer,
            operational,
            state,
            human_review=not decision.accepted,
        )
        return None

    def _route_repair_terminal_author_result(
        self,
        item: WorkItem,
        terminal: AuthorRepairTerminal,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        prior_attempts: tuple[Mapping[str, Any], ...] = (),
    ) -> Optional[str]:
        """Adjudicate a repair generation's terminal recommendation as a terminal.

        The repair arm and the first-call arm now settle through ONE adjudicator.
        The first-call arm (``_route_terminal_author_result``, wrapped by its
        caller) already had both halves: the terminal-disposition gate, and a
        model-local fallback for a gate that itself refuses. The repair arm had
        neither, so an identical recommendation got an entirely different record
        depending only on WHICH author call produced it -- visible in one run, in
        pilot rung 3: ``m3671`` returned ``BLOCKED`` on its first call and
        terminalized ``failed:accuracy-gate / terminal-disposition-unverifiable``
        through the gate, while ``m5888`` returned an equally well-formed
        ``BLOCKED`` one generation later and terminalized ``failed:runner /
        protocol-violation`` with the verdict discarded.

        The fallback mirrors the first-call arm exactly, including
        ``terminal_gate_obtained=False``: a lane that never obtained a gate cannot
        assert a gate verdict, but the broker's frozen source facts are true
        whatever the gate did and stay in the record.

        Parameters
        ----------
        item:
            Model whose bounded repair generation answered terminally.
        terminal:
            Control-flow signal carrying the staged terminal artifact.
        reducer, operational, state:
            Locked canonical writers and scheduler state.
        prior_attempts:
            Already-persisted ledger attempts for this work generation, when the
            repair that answered terminally was requested AFTER the worker ran
            (the ``run_modes`` site). See ``_route_terminal_author_result``.

        Returns
        -------
        str | None
            Usage-limit pause reason, if the terminal checker is unavailable.
        """

        artifact = terminal.artifact
        try:
            return self._route_terminal_author_result(
                item,
                artifact,
                reducer,
                operational,
                state,
                prior_attempts=prior_attempts,
            )
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
            raise
        except Exception as exc:  # noqa: BLE001 -- terminal gating is model-local
            reason = (
                "internal-error" if self._is_infrastructure_error(exc) else "protocol-violation"
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
                # DX-1/RE-1 (m12446 shape): a host/machinery fault while gating a
                # repair-generation terminal recommendation is infrastructure
                # weather, never this model's verdict -- park loudly, member
                # re-drives next leg.
                self._park_non_model_fault(item, artifact, persisted_error, operational)
                return None
            self._terminalize(
                item,
                artifact,
                "failed:runner",
                reason,
                str(exc),
                (*prior_attempts, persisted),
                reducer,
                operational,
                state,
                terminal_gate_obtained=False,
            )
            return None

    def _adopt_orphan_citations(
        self,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
    ) -> None:
        """Re-cite lawful review-held heads whose in-window citation is orphaned.

        The partition proof refuses any in-window head whose citation names a
        run that never joined the open generation ("position is not
        provenance") -- the tripwire is correct and stays untouched.  But a
        review-held supersession minted OUTSIDE a joined run (the 2026-08-27
        out-of-band appends; historically also any hold candidate inheriting
        its superseded head's retired citation) leaves the campaign
        unresumable: the sweep never re-visits already-held heads, admission
        refuses before any close, and no operator tool may rewrite a ledger
        row.  This pass is the head-continuity remedy, run by the leg itself
        under its OWN joined run.

        The adoption law (all replay-derived, zero producer-trusted fields;
        pure halves in :mod:`menagerie.crawler.terminal_review`):

        * eligible is EXACTLY an in-window head whose citation run did not
          join the open generation, whose row is a review-held failure
          terminal byte-equal to the containment MARKING the sweep applies
          over its parent (the marking shape, never the sweep's adjudicated
          verdict -- adoption consults no disposition gate, and a hold is the
          negative disposition the gate could only have relaxed), and whose
          parent sits strictly below the window floor (sealed pre-generation
          history -- a forged in-window parent refuses, so adoption can never
          launder out-of-band content);
        * the dependency-vector snapshot/item pins and evidence bindings are
          adjudicated byte-exact on BOTH comparisons; only the enumerated
          rotatable identity axes may move, so adoption can never re-stamp a
          member's recorded snapshot, un-stale its item pin, or rebind the
          close's stale-excluded cohort (Opus C1);
        * a member whose head is named by a LIVE (unvoided, unreleased)
          grant-pending or award-supersession park refuses: the adoption
          re-append would advance the head revision and silently consume the
          park's pending operator decision (Opus C2);
        * the re-append is the SAME containment builder over the orphan head,
          differing only in parentage and the adopting run's citation, and a
          post-``prepare_model`` drift check refuses anything beyond that
          (disposition-invariant: a review-held failure terminal stays a
          review-held failure terminal);
        * the pass is shutdown-guarded per member and bounded per leg
          (``ORPHAN_ADOPTION_PER_LEG``; members beyond the bound are
          disclosed and retried next leg -- Opus C3);
        * every orphan -- adopted, refused, or beyond the bound -- is
          disclosed on the leg's durable summary census event, and every
          ADOPTION is additionally disclosed on a per-member census event
          appended BEFORE the re-append it discloses, so a leg dying
          mid-pass can never leave an adopted row undisclosed (Opus C5); and
        * anything unadoptable is left in place for the partition proof to
          refuse exactly as before (fail-closed).

        Parameters
        ----------
        reducer, operational:
            Locked canonical writers.
        """

        from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            MODELS_SOURCE_NAMES,
            close_ledger_sources,
            generation_run_ids,
            model_head_citations,
        )

        # The ONE citation reader, shared with the partition proof so the
        # adoption predicate and the refusal it remedies can never disagree.
        from menagerie.crawler.partition_proof import (  # noqa: PLC0415,PLC2701 -- the proof's own reader
            _model_row_run_id,
        )

        generation = self._leg_generation
        if generation is None:
            raise DriverIntegrationError(
                "cannot adopt orphan citations without an open leg generation"
            )
        self._check_shutdown("orphan-citation-adoption")
        records_root = canonical_records_root(self.paths.ledgers.models)
        generation_runs = generation_run_ids(records_root, generation.generation_id)
        if self.config.run_id not in generation_runs:
            raise DriverIntegrationError(
                f"run {self.config.run_id!r} has not joined generation "
                f"{generation.generation_id}: adopting under it would mint a "
                "new orphan citation"
            )
        floors = {
            name: int(dict(generation.baselines.get(name) or {}).get("rows") or 0)
            for name in MODELS_SOURCE_NAMES
        }
        heads = model_head_citations(records_root, require_ledger=False)
        orphans: list[tuple[str, Mapping[str, Any], Optional[str]]] = []
        for stable_id in sorted(heads):
            head_entry = heads[stable_id]
            source = str(head_entry["source"])
            if source not in floors or int(head_entry["row_index"]) < floors[source]:
                # Pre-window history belongs to the carry path, whatever its
                # citation; adoption owns only the open window.
                continue
            cited_run = _model_row_run_id(head_entry["record"])
            if cited_run in generation_runs:
                continue
            orphans.append((stable_id, head_entry, cited_run))
        if not orphans:
            return

        # One read per source ledger holding orphans: the parent rows (and
        # their non-empty-row indices) come from the same bytes the proof
        # cites, with the head's own citation re-verified as an alignment
        # tripwire before any parent lookup is trusted.
        sources = close_ledger_sources(records_root)
        rows_by_source: dict[str, list[JsonObject]] = {
            source: scan_jsonl(sources[source])
            for source in sorted({str(entry["source"]) for _, entry, _ in orphans})
        }

        # One durable-ledger read for the whole pass: any LIVE (unvoided,
        # unreleased) park row naming a member's head refuses its adoption --
        # the re-append would advance the head revision, and
        # ``grant_park_released`` reads any revision advance as "the granted
        # supersession landed", silently consuming the park's pending
        # operator decision (Opus C2). Both park dispositions live in this
        # one ledger, and an unknown future disposition blocks too
        # (fail-closed: the pending decision always wins). Since P5-SOL-05
        # (round 4) the append itself goes through
        # ``CanonicalReducer.append_orphan_adoption``, which re-derives the
        # ENTIRE closed adoption rule -- joined run, orphaned citation,
        # unique below-floor parent, hold-marking equality, THIS park
        # liveness, and post-prepare drift -- from ledgers at the write door,
        # so every direct ``append_model`` caller now refuses the carve
        # outright. This driver-side pass is therefore the redundant FIRST
        # fence BY DESIGN -- kept, like the era arm's 4633 containment, so
        # the member refuses typed and census-disclosed here instead of
        # surfacing as a raw reducer error.
        park_rows_by_member: dict[str, list[JsonObject]] = {}
        for park_row in load_grant_park_rows([grant_park_path(records_root)]):
            park_rows_by_member.setdefault(str(park_row.get("stable_id")), []).append(
                park_row
            )

        adopted: list[JsonObject] = []
        unadoptable: list[JsonObject] = []
        beyond_bound: list[JsonObject] = []
        for stable_id, entry, cited_run in orphans:
            # The adjacent backlog sweep guards the shutdown boundary per
            # head and bounds its per-leg burst; the adoption pass matches
            # that discipline (Opus C3). Beyond-bound members stay orphaned
            # and disclosed; the next leg adopts the next tranche.
            self._check_shutdown("orphan-citation-adoption")
            source = str(entry["source"])
            head = dict(entry["record"])
            citation: JsonObject = {
                "stable_id": stable_id,
                "source": source,
                "row_index": int(entry["row_index"]),
                "cited_run_id": cited_run,
            }
            if len(adopted) >= ORPHAN_ADOPTION_PER_LEG:
                beyond_bound.append(citation)
                continue

            def refuse(reason: str, *, citation: JsonObject = citation) -> None:
                unadoptable.append({**citation, "reason": reason})

            rows = rows_by_source[source]
            index = int(entry["row_index"])
            if not (0 <= index < len(rows)) or rows[index].get(
                "record_revision"
            ) != head.get("record_revision"):
                refuse(
                    "head citation does not align with the source ledger read: "
                    "refusing every lookup derived from it"
                )
                continue
            parent_matches = [
                (row_index, row)
                for row_index, row in enumerate(rows)
                if row.get("record_revision") == head.get("parent_revision")
            ]
            if len(parent_matches) != 1:
                refuse(
                    f"parent revision resolves to {len(parent_matches)} rows "
                    "in the cited source; adoption requires exactly one"
                )
                continue
            parent_index, parent = parent_matches[0]
            refusal = orphan_hold_adoption_refusal(
                head,
                parent,
                parent_row_index=parent_index,
                window_floor=floors[source],
            )
            if refusal is not None:
                refuse(refusal)
                continue
            live_parks = [
                park_row
                for park_row in park_rows_by_member.get(stable_id, ())
                if not grant_park_released(park_row, head)
            ]
            if live_parks:
                dispositions = ", ".join(
                    sorted({str(row.get("disposition")) for row in live_parks})
                )
                refuse(
                    f"head is named by {len(live_parks)} live park row(s) "
                    f"({dispositions}): the adoption re-append would advance "
                    "the head revision and silently consume the park's "
                    "pending operator decision; drain or void the park first"
                )
                continue
            try:
                # Inside the typed-refusal ladder deliberately (Opus C4): a
                # malformed head the closed rule cannot see (e.g. a
                # provenance block that is not a mapping) raises the
                # builder's ValueError, which is a fact about ONE member --
                # it refuses typed and disclosed, never crashing the leg.
                candidate = terminal_review_hold_candidate(
                    head,
                    requested_at=str(head["status"]["human_review"]["requested_at"]),
                    producer_run_id=self.config.run_id,
                    machine_id=self.config.machine_id,
                )
                prepared = reducer.prepare_model(candidate)
                drift = orphan_adoption_drift(prepared, head)
                if drift is not None:
                    refuse(drift)
                    continue
                # Crash-durable disclosure (Opus C5): the per-member census
                # event lands BEFORE the append it discloses, so a leg dying
                # between the append and the summary census can never leave
                # an adopted row undisclosed. A member event with no adopted
                # revision is coherent: the leg died before its append, the
                # member is still orphaned, and the next leg retries.
                operational.append(
                    mint_operational_event(
                        {
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
                                "disposition": ORPHAN_CITATION_ADOPTION_MEMBER_DISPOSITION,
                                "generation_id": generation.generation_id,
                                **citation,
                                "orphan_revision": head.get("record_revision"),
                            },
                        }
                    )
                )
                # The write door re-derives the whole closed rule again from
                # ledgers (P5-SOL-05); its typed OrphanAdoptionRefusal (or any
                # other reducer refusal) is one member's fact, caught below
                # and disclosed as the same per-member census refusal shape.
                result = reducer.append_orphan_adoption(prepared)
            except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
                raise
            except _PROGRAMMING_ERRORS:
                raise
            except Exception as exc:  # noqa: BLE001 -- one head's recording refusal
                refuse(f"{type(exc).__name__}: {exc}"[:500])
                continue
            adopted.append(
                {**citation, "adopted_revision": result.record.get("record_revision")}
            )

        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": len(orphans)},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": ORPHAN_CITATION_ADOPTION_DISPOSITION,
                "generation_id": generation.generation_id,
                "eligible": len(orphans),
                "bound": ORPHAN_ADOPTION_PER_LEG,
                "adopted": {
                    "count": len(adopted),
                    "members": adopted[:TERMINAL_REVIEW_CENSUS_ID_CAP],
                },
                "unadoptable": {
                    "count": len(unadoptable),
                    "members": unadoptable[:TERMINAL_REVIEW_CENSUS_ID_CAP],
                },
                "beyond_bound": {
                    "count": len(beyond_bound),
                    "members": beyond_bound[:TERMINAL_REVIEW_CENSUS_ID_CAP],
                },
            },
        }
        operational.append(mint_operational_event(event))

    def _review_terminal_backlog(
        self,
        snapshot: IntakeSnapshot,
        reducer: CanonicalReducer,
        operational: JsonlLedger,
    ) -> Optional[str]:
        """Adjudicate the unreviewed failed-terminal backlog, bounded per leg.

        The requeue-grant lane admits a re-drive only over a REVIEW-HELD
        terminal, the marking is produced only by the terminal-disposition
        lane, and that lane's only fresh caller runs when an author session IN
        THE CURRENT LEG returns a typed recommendation -- so a settled failed
        head's owed review can otherwise never be scheduled (the
        grants-wait-on-review-waits-on-scheduling cycle in
        :mod:`menagerie.crawler.terminal_review`). This sweep runs BEFORE the
        WS-1 selection so a hold released here binds in the same leg.

        Per eligible head, in stable order: dormant and lane-parked members
        are disclosed untouched; a current-identity terminal gate already
        covering the head's recorded work generation needs no new session;
        otherwise the claim is rehydrated from FINAL ledger-bound artifact
        authority only (adjudicating unattested work-dir bytes is forbidden)
        and adjudicated through the exact fresh-terminal front half --
        :meth:`_obtain_terminal_disposition_gate`, including spend admission,
        the once-per-leg calibration readiness gate, and the lane's own
        deferral parks and streak breaker. An accepted claim is CONFIRMED (the
        gate row is the durable adjudication; the settled record is already
        true). A refused claim re-appends the head verbatim with the
        containment ladder's review-held marking
        (:func:`~menagerie.crawler.terminal_review.terminal_review_hold_candidate`);
        the ORIGINAL status code survives so the outstanding grant's stage
        still binds. Live sessions are bounded by
        ``TERMINAL_REVIEW_BACKLOG_PER_LEG``; every head gets exactly one
        closed census class on the leg's durable disclosure event -- no
        silent outcomes.

        Parameters
        ----------
        snapshot:
            Verified active intake snapshot.
        reducer, operational:
            Locked canonical writers.

        Returns
        -------
        str | None
            Usage-limit pause reason when the checker provider paused the
            leg; remaining heads are disclosed unprocessed and stay eligible.
        """

        heads = unreviewed_failed_terminal_heads(reducer.raw_head_view)
        if not heads:
            return None
        census: dict[str, list[str]] = {name: [] for name in TERMINAL_REVIEW_CENSUS_CLASSES}
        refusals: dict[str, str] = {}
        items_by_id = {entry.stable_id: entry for entry in snapshot.items}
        checker_identity = lane_adjudication_identity(
            self.dependencies.checker, "terminal_disposition", self.config
        )
        # Driver-owned durable record of every CONFIRMED sweep review. The
        # ``already_adjudicated`` skip requires one of these rows AND a
        # still-current gate identity: a review-free settled head always
        # carries its fresh-mint terminal gate at some identity, and when the
        # lane identity is stable that mint gate alone would silently starve
        # the independent re-review this sweep exists to buy (the held grant
        # would then wait forever -- the exact deadlock again).
        swept: set[tuple[str, str]] = {
            (str(details["stable_id"]), str(details["work_id"]))
            for event in scan_jsonl(self.paths.operational_ledger)
            if isinstance(details := event.get("details"), Mapping)
            and details.get("disposition") == TERMINAL_REVIEW_ADJUDICATED_DISPOSITION
            and details.get("stable_id") is not None
            and details.get("work_id") is not None
        } if self.paths.operational_ledger.exists() else set()
        # -- Pass 1 (writer, stable order): classify every head WITHOUT
        # dispatching. Classification is dispatch-independent -- dormant,
        # parked, and already_adjudicated consult only pre-sweep state, and
        # backlog heads are distinct stable IDs -- so it can run ahead of the
        # sessions it plans; that is what lets the multi-minute check_terminal
        # sessions overlap (DX-54: 114 heads x ~146 s ran strictly serially in
        # 200a). Every census contribution carries its head index so a
        # mid-commit pause reproduces the serial census exactly: classes land
        # only for heads BEFORE the pausing head, everything from it onward is
        # disclosed unprocessed.
        contributions: list[tuple[int, str, str]] = []
        eligible: list[tuple[int, str, Optional[str], WorkItem, AuthorArtifact]] = []
        for head_index, head in enumerate(heads):
            stable_id = str(head.get("stable_id"))
            self._check_shutdown("terminal-review-backlog")
            intake = items_by_id.get(stable_id)
            if intake is None:
                contributions.append((head_index, "dormant", stable_id))
                continue
            if stable_id in self._checker_lane_parked_ids:
                contributions.append((head_index, "parked", stable_id))
                continue
            work_id = reducer.terminal_work_generation(stable_id)
            if (
                work_id is not None
                and (stable_id, work_id) in swept
                and current_identity_terminal_gate(
                    reducer.gate_records,
                    stable_id=stable_id,
                    work_id=work_id,
                    checker_identity=checker_identity,
                )
                is not None
            ):
                contributions.append((head_index, "already_adjudicated", stable_id))
                continue
            item = WorkItem(
                intake,
                route_model(intake.to_model_requirements(_framework_from_intake(intake))),
                requeue_work_id=(
                    work_id
                    if work_id is not None and work_id != f"work-{stable_id}"
                    else None
                ),
            )
            try:
                artifact = (
                    self._rehydrate_final_authority(item, reducer, operational)
                    if work_id is not None
                    else None
                )
            except DriverIntegrationError:
                # BL-8 (a) blast radius: one head whose recorded authority
                # cannot be rehydrated (missing, superseded, or an AMBIGUOUS
                # mirror object -- leg 16, 2026-08-21: ambiguous intrinsic
                # inventory killed the whole leg at the sweep) is a fact about
                # THAT head. Disclose it in the census and keep sweeping;
                # never abort the campaign for it.
                contributions.append((head_index, "claim_unrehydratable", stable_id))
                continue
            if artifact is None or isinstance(artifact.author_result, ProposedAuthorResult):
                decision_text = (head.get("source_resolution") or {}).get("decision")
                if decision_text == TERMINAL_UNADJUDICATED_DECISION:
                    contributions.append((head_index, "claim_unrehydratable", stable_id))
                else:
                    contributions.append((head_index, "no_consumer", stable_id))
                continue
            if len(eligible) >= TERMINAL_REVIEW_BACKLOG_PER_LEG:
                contributions.append((head_index, "beyond_bound", stable_id))
                continue
            # The exact fresh-lane readiness gate: receiptless production legs
            # refuse HERE, before any session is bought (or even planned); the
            # calibration-run flag bypasses inside require_checker_calibration
            # itself.
            self._require_checker_leg_readiness(operational)
            eligible.append((head_index, stable_id, work_id, item, artifact))

        # -- Pass 2 (commit, stable order): buy or replay each planned
        # adjudication through the unchanged front half. Only the SESSIONS
        # overlap, in a sweep-scoped pool; every ledger append, validation,
        # and review-held supersession still happens here, on this writer, in
        # stable-ID order -- so a hold released here binds in the same leg
        # exactly as under the serial sweep (the FT-(c) position invariant:
        # this method returns only after every commit has landed, and WS-1
        # selection re-reads the durable models ledger afterwards).
        dispatched = 0
        pause: Optional[str] = None
        pause_head_index = len(heads)
        pool = (
            _CheckerSessionPool(
                max(1, int(self.config.checker_concurrency)),
                lambda scope: self._admit_checker_spend(operational, scope),
                lambda reservation, elapsed, completed: self._settle_paid_session(
                    reservation,
                    elapsed_seconds=elapsed,
                    completed=completed,
                ),
            )
            if eligible
            else None
        )
        try:
            if pool is not None:
                self._prefetch_terminal_review_sessions(
                    pool,
                    tuple(
                        (entry_id, entry_artifact)
                        for _index, entry_id, _work, _item, entry_artifact in eligible
                    ),
                    reducer,
                )
            for head_index, stable_id, work_id, item, artifact in eligible:
                self._check_shutdown("terminal-review-backlog")
                head = heads[head_index]
                dispatched += 1
                gate, pause = self._obtain_terminal_disposition_gate(
                    item,
                    artifact,
                    reducer,
                    operational,
                    pool=pool,
                    # AG-4: the sweep IS the sanctioned re-adjudication; its
                    # disagreements route through the review-held supersession
                    # below, never a conflict park (disclosure rows still land).
                    conflict_parking=False,
                )
                if pause is not None:
                    # A usage pause is a decision to STOP SPENDING: unstarted
                    # sibling sessions are cancelled here; started ones are
                    # drained uncommitted in the finally (their reservations
                    # settle, their bytes stay on disk for the next leg).
                    pause_head_index = head_index
                    if pool is not None:
                        pool.cancel_pending()
                    break
                if gate is None:
                    contributions.append((head_index, "parked", stable_id))
                    continue
                pack = _terminal_checker_item(artifact)
                try:
                    decision = validate_terminal_disposition_gate(
                        gate,
                        artifact.author_result,
                        source_manifest=artifact.source_manifest,
                        evidence_pack=pack["evidence_pack"],
                        license_identity=str(pack["license_identity"]),
                    )
                except GateRoutingError as exc:
                    # W5 member-fault containment (calibration leg 6,
                    # 2026-08-27, m3423): a recorded claim whose FINAL
                    # evidence pack refuses gate validation -- e.g. a
                    # ``supports`` claim outside the arm's closed predicate
                    # vocabulary -- re-raises identically on every replay of
                    # THAT head; it is machinery about one member's recorded
                    # artifact, never a model verdict and never the leg's.
                    # Park on the lane's own typed brake: the durable deferral
                    # row discloses the defect, the head stays untouched and
                    # re-eligible next run, and ``count_in_streak=False``
                    # keeps one record's defect from reading as lane weather.
                    self._record_checker_lane_deferral(
                        item,
                        exc,
                        attempted_batch_sizes=(1,),
                        operational=operational,
                        failure_site="terminal-review-validate",
                        gate_kind="terminal_disposition",
                        count_in_streak=False,
                    )
                    contributions.append((head_index, "parked", stable_id))
                    continue
                if decision.accepted:
                    adjudicated: JsonObject = {
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
                            "disposition": TERMINAL_REVIEW_ADJUDICATED_DISPOSITION,
                            "stable_id": stable_id,
                            "work_id": work_id,
                            "gate_id": str(gate.get("gate_id")),
                        },
                    }
                    operational.append(mint_operational_event(adjudicated))
                    swept.add((stable_id, str(work_id)))
                    contributions.append((head_index, "confirmed", stable_id))
                    continue
                candidate = terminal_review_hold_candidate(
                    head,
                    requested_at=self.dependencies.clock(),
                    # Position is provenance: the re-append cites the run that
                    # appends it, never the superseded head's stale citation
                    # (an inherited retired run id orphans the row in the open
                    # generation's window and the partition proof refuses the
                    # whole leg -- ordinary leg 2, 2026-08-27).
                    producer_run_id=self.config.run_id,
                    machine_id=self.config.machine_id,
                )
                try:
                    reducer.append_model(reducer.prepare_model(candidate))
                except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
                    raise
                except _PROGRAMMING_ERRORS:
                    raise
                except Exception as exc:  # noqa: BLE001 -- one head's recording refusal
                    contributions.append((head_index, "revision_refused", stable_id))
                    refusals[stable_id] = f"{type(exc).__name__}: {exc}"[:500]
                    continue
                contributions.append((head_index, "review_held", stable_id))
        finally:
            if pool is not None:
                pool.drain()
                pool.close()

        for _head_index, class_name, class_stable_id in sorted(contributions):
            if _head_index < pause_head_index:
                census[class_name].append(class_stable_id)
        unprocessed = len(heads) - pause_head_index if pause is not None else 0
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.HEALTHY.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": len(heads)},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": TERMINAL_REVIEW_CENSUS_DISPOSITION,
                "eligible": len(heads),
                "dispatched": dispatched,
                "bound": TERMINAL_REVIEW_BACKLOG_PER_LEG,
                "unprocessed_after_pause": unprocessed,
                "pause_reason": pause,
                "classes": {
                    name: {
                        "count": len(ids),
                        "stable_ids": ids[:TERMINAL_REVIEW_CENSUS_ID_CAP],
                    }
                    for name, ids in census.items()
                },
                "revision_refusals": refusals,
            },
        }
        operational.append(mint_operational_event(event))
        return pause

    def _repair_author(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        gates: Sequence[Mapping[str, Any]],
        generation: int,
        reducer: CanonicalReducer,
        *,
        gate_kind: str,
        operational_events: Sequence[Mapping[str, Any]] = (),
    ) -> AuthorArtifact:
        """Persist checker findings and request one bounded author repair generation.

        Parameters
        ----------
        item:
            Current routed model.
        artifact:
            Most recent authored generation.
        gates:
            Durable checker history.
        generation:
            One-based bounded repair generation.
        reducer:
            Active canonical writer carrying the exact authority context.
        gate_kind:
            ``metadata_batch`` or ``fidelity`` source finding.

        Returns
        -------
        AuthorArtifact
            Re-authored, normalized, identity-validated generation.
        """

        # T2-07/T1-13: repairs buy sessions for the SAME work generation, so
        # the derived allowance gates them exactly like fresh dispatches. The
        # callers already catch AuthorEffortCapExceeded and terminalize this
        # model as `failed:author` with the cap disclosure. Operational rows
        # carry any supersession-discard attestation so a post-discard lineage
        # counts only its own stagings. Every caller reaches here through
        # `_retry_infrastructure_call(admission=("author", item))`, whose
        # `_admit_author_spend` has ALREADY enforced this valve and recorded
        # the buy for THIS session -- so this re-check tolerates exactly that
        # one admitted in-flight charge; re-counting it refused the allowance's
        # own final repair session as `failed:author` before the metadata gate
        # could adjudicate its review-held terminal.
        enforce_author_session_allowance(
            (*reducer.artifact_ledger.events, *operational_events),
            item,
            operational_events=operational_events,
            admitted_in_flight=True,
        )
        latest = _find_gate(gates, item.stable_id, gate_kind, artifact.proposal)
        if latest is None:
            raise DriverIntegrationError(f"{gate_kind} repair gate missing for {item.stable_id}")
        gate_item = next(value for value in latest["items"] if value["stable_id"] == item.stable_id)
        repair_path = (
            self.paths.work_root
            / item.stable_id
            / "repair"
            / f"{gate_kind}-generation-{generation}.json"
        )
        request = {
            "stable_id": item.stable_id,
            "generation": generation,
            "gate_kind": gate_kind,
            "gate_id": latest["gate_id"],
            "root_cause_fingerprint": _gate_item_fingerprint(gate_item),
            "required_repairs": list(gate_item.get("required_repairs", [])),
        }
        if not repair_path.is_file():
            _write_json_atomic(repair_path, request)
        # The ONE feedback channel (SEAM_REDESIGN 3.4): the checker's verbatim
        # findings attach to the rejected generation's attempt record, and the
        # repairing author's envelope + brief carry them under prior_attempts.
        # Before this, findings went to a file no author ever read.
        record_checker_findings(
            self.paths.work_root / item.stable_id / "author",
            gate_kind=gate_kind,
            generation=generation,
            required_repairs=list(gate_item.get("required_repairs", [])),
            root_cause_fingerprint=str(_gate_item_fingerprint(gate_item)),
        )
        repaired = self.dependencies.author.author(
            item, self.paths.work_root, self.config, reducer.context
        )
        repaired = self._stage_author_result(
            item, repaired, reducer, operational_events=operational_events
        )
        if not isinstance(repaired.author_result, ProposedAuthorResult):
            # NOT a contract breach, and the message this replaces said otherwise
            # twice over: it blamed the CHECKER for an answer the AUTHOR gave, and
            # it was a `DriverIntegrationError`, which the caller reads as
            # `protocol-violation`. The three typed terminal arms are legitimate
            # author outcomes on every generation, not just the first, and a repair
            # generation -- re-reading the sources under the checker's findings --
            # is exactly when an author learns the model cannot be authored at all.
            # Hand it to the caller as control flow so it reaches the SAME
            # terminal-disposition gate a first-call terminal reaches.
            raise AuthorRepairTerminal(repaired, gate_kind)
        repaired = _normalize_artifact_modes(
            repaired,
            self.config,
            environment_packages=_routed_environment_packages(self.registry, item.route.intent),
        )
        if repaired.proposal.get("stable_id") != item.stable_id:
            raise DriverIntegrationError("repaired author proposal stable_id does not match intake")
        if repaired.campaign_root_work_id != artifact.campaign_root_work_id:
            raise DriverIntegrationError("repaired author result changed campaign lineage")
        _validate_artifact_identities(repaired, self.config, item=item)
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
        del artifact
        return repaired

    def _repair_author_for_detected_modes(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        expansion: Mapping[str, Any],
        reducer: CanonicalReducer,
        *,
        operational: JsonlLedger,
    ) -> AuthorArtifact:
        """Request bounded proposal revisions until every detected mode is declared.

        Parameters
        ----------
        item, artifact:
            Current model and proposal generation that produced the observation.
        expansion:
            Typed worker observation naming the complete detected mode set.
        reducer:
            Active canonical writer carrying exact author and custody authority.
        operational:
            Canonical operational ledger for pre-dispatch bought-session truth.

        Returns
        -------
        AuthorArtifact
            A new identity-validated proposal GENERATION covering the full
            detected set. Its new identity is the proposal identity
            (``proposal_sha256``/``proposal_id``); the work identity is
            machine-issued at scheduling and stays exactly
            ``item.active_work_id``, like every ``_repair_author`` checker-repair
            generation.

        Raises
        ------
        DriverIntegrationError
            If no revision satisfies the observation before the configured cap.
            The cap charges only revisions the author actually COMPLETED and the
            driver rejected, witnessed by durable per-generation charge markers.
        AuthorRepairTerminal
            If a repair generation answers with a typed terminal recommendation.
            That is an ANSWER, not a failed revision: re-authoring after it would
            burn the remaining cap asking a question the author already declined,
            so it propagates immediately for the caller to adjudicate through the
            same terminal-disposition gate the first-call arm uses.
        """

        detected = canonical_meaningful_modes(
            expansion.get("detected_meaningful_modes"),
            field="detected_meaningful_modes",
        )
        if "train" in tuple(
            str(value) for value in expansion.get("missing_proposal_modes", [])
        ):
            # DX-9/RE-9 unsatisfiability guard: a demanded train declaration
            # over a batch-1 rank>=3 probe contract would be refused by the
            # SAME validator on every revision -- the m-batch-1 class burned
            # its whole cap on this exact contradiction (0/14 successes).
            # Refuse typed BEFORE any author spend.
            from menagerie.crawler.proposal import (  # noqa: PLC0415 -- lazy: repair-path only
                train_probe_batch_conflict,
            )

            conflict = train_probe_batch_conflict(
                artifact.proposal.get("proposed_facts", {}).get("input_contract")
            )
            if conflict is not None:
                raise ModeRepairUnsatisfiable(
                    "worker-detected mode expansion contradicts the gated input "
                    f"contract: {conflict}"
                )
        repair_root = self.paths.work_root / item.stable_id / "repair"
        # The cap charges COMPLETED-and-rejected revisions, witnessed by their
        # durable charge markers -- never request files. A request is written
        # BEFORE the author dispatch, so counting requests charged every round a
        # dead run merely STARTED: rung 11 left m10949 with two request files and
        # zero author answers standing, which made ``range(3, 3)`` empty and any
        # resume die ``cap exhausted`` without running a single revision. A crash
        # between the author's answer and its charge marker re-runs that round
        # uncharged: under-charging on a rare crash costs one bounded session;
        # systematic over-charging killed the model.
        charged = tuple(sorted(repair_root.glob("run-modes-charge-*.json")))
        last_error: Optional[BaseException] = None
        for generation in range(len(charged) + 1, self.config.run_repair_max + 1):
            request = {
                "stable_id": item.stable_id,
                "generation": generation,
                "gate_kind": "run_modes",
                "route": "recipe-and-gate-revision-required",
                "campaign_root_work_id": _artifact_lineage(artifact),
                "proposal_sha256": artifact.proposal["proposal_sha256"],
                "proposal_meaningful_modes": list(expansion.get("proposal_meaningful_modes", [])),
                "detected_meaningful_modes": list(detected),
                "missing_proposal_modes": list(expansion.get("missing_proposal_modes", [])),
            }
            repair_path = repair_root / f"run-modes-generation-{generation}.json"
            if not repair_path.is_file():
                _write_json_atomic(repair_path, request)
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
                    # The same defect `_repair_author` just had, with a worse
                    # cost profile: this used to be a `DriverIntegrationError`,
                    # which the `except Exception` arm below consumed into
                    # `last_error` -- so a typed DEFER/SKIP/BLOCKED answer
                    # re-ran the AGENTIC author for every remaining generation
                    # of the cap before dying `cap exhausted` as
                    # `failed:runner / protocol-violation`, discarding the
                    # verdict. The three terminal arms are legitimate author
                    # outcomes on a mode-repair generation too; hand the staged
                    # artifact to the caller as control flow so it reaches the
                    # SAME terminal-disposition gate a first-call terminal
                    # reaches.
                    raise AuthorRepairTerminal(repaired, "run_modes")
                repaired = _normalize_artifact_modes(
                    repaired,
                    self.config,
                    environment_packages=_routed_environment_packages(
                        self.registry, item.route.intent
                    ),
                )
                if repaired.proposal.get("stable_id") != item.stable_id:
                    raise DriverIntegrationError(
                        "mode-repair proposal stable_id does not match intake"
                    )
                repaired_modes = canonical_meaningful_modes(
                    repaired.proposal.get("proposed_facts", {})
                    .get("modes", {})
                    .get("meaningful_modes"),
                    field="modes.meaningful_modes",
                )
                if not set(detected).issubset(repaired_modes):
                    raise DriverIntegrationError(
                        "mode-repair proposal does not cover every worker-detected mode"
                    )
                if repaired.proposal.get("work_id") != artifact.proposal.get("work_id"):
                    # Work identity is MACHINE-issued: the author lane stamps
                    # ``item.active_work_id`` into the envelope
                    # (``_author_from_frozen_sources``) and the result validator
                    # pins the proposal to it (``_validate_proposal_binding``), so
                    # a repair generation legitimately RETURNS the same work
                    # identity and issues its new identity as a new PROPOSAL
                    # generation, exactly like the ``_repair_author``
                    # checker-repair lane. The previous check here demanded the
                    # opposite -- a NEW work identity -- which no real author can
                    # ever satisfy: rung 11's m10949 answered the repair with a
                    # correct dual-mode revision twice and both were refused on
                    # this rule alone, burning the whole cap into
                    # ``failed:runner`` (the fixture that blessed the rule forged
                    # its own envelope, which the production lane cannot do). A
                    # repaired proposal on a DIFFERENT work identity can only be
                    # a lane handing back another generation's artifact, so the
                    # tripwire now refuses in the direction that is actually
                    # reachable.
                    raise DriverIntegrationError(
                        "mode-repair proposal drifted off its machine-issued work identity"
                    )
                if repaired.campaign_root_work_id != artifact.campaign_root_work_id:
                    raise DriverIntegrationError("mode repair changed campaign lineage")
                _validate_artifact_identities(repaired, self.config, item=item)
            except (RetryableOperatorError, AuthorRepairTerminal, AuthorBackoffError):
                # None of these is a failed revision, so none may consume the
                # cap: a retryable operator fault is infrastructure; a terminal
                # recommendation is the author's ANSWER; and a provider backoff
                # is a campaign pause -- the caller's `except AuthorBackoffError`
                # arm was unreachable while the blanket arm below swallowed it
                # into `last_error` and re-dispatched against an exhausted
                # provider.
                raise
            except Exception as exc:  # noqa: BLE001 -- a completed, rejected revision consumes the cap
                last_error = exc
                # The durable charge: this generation ran an author to an answer
                # and the driver rejected it. Only the exception TYPE is
                # recorded -- exception text can carry externally controlled
                # bytes, and this sidecar outlives the run.
                _write_json_atomic(
                    repair_root / f"run-modes-charge-{generation}.json",
                    {
                        "stable_id": item.stable_id,
                        "generation": generation,
                        "gate_kind": "run_modes",
                        "disposition": "revision-rejected",
                        "error_type": type(exc).__name__,
                    },
                )
                continue
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
        detail = f": {last_error}" if last_error is not None else ""
        # DX-9/RE-9: budget exhaustion is an author-lane fact, never a runner
        # verdict. The cause chain is the OWNER evidence (DX-1/RE-1): the
        # catch arm's infrastructure triage walks ``__cause__``, so a cap
        # burned by host weather (m12446 class) still parks as weather, while
        # a cap burned by genuinely rejected revisions publishes
        # ``failed:author / repair-exhausted``.
        raise ModeRepairBudgetExhausted(
            f"detected-mode repair cap exhausted after {self.config.run_repair_max} revisions{detail}"
        ) from last_error

    def _terminalize_accuracy_gate(
        self,
        item: WorkItem,
        artifact: AuthorArtifact,
        gate: Mapping[str, Any],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
    ) -> None:
        """Append the bounded human-review terminal for one rejected metadata item."""

        gate_item = next(value for value in gate["items"] if value["stable_id"] == item.stable_id)
        verdict = str(gate_item["verdict"])
        # The shared kernel keeps this candidate coherent with the terminal proof
        # the reducer re-derives: an integrity-disputed item is an identity
        # mismatch here too, never a cap claim the proof contradicts.
        reason = metadata_gate_failure_reason(gate_item)
        # Under the wave order no attempt exists yet at gate time, so this tuple
        # is empty and the assembly is byte-identical to the historical record.
        # Under execute-first triage the work generation's persisted attempts
        # ARE the reducer's proof; the assembled terminal must carry them or
        # the reducer's proof-consistency tripwire rightly refuses the record.
        persisted_attempts = _matching_model_attempts(
            self.paths.ledgers.attempts, artifact.proposal
        )
        self._terminalize(
            item,
            artifact,
            "failed:accuracy-gate",
            reason,
            "; ".join(str(value) for value in gate_item.get("required_repairs", []))
            or f"metadata gate {verdict}",
            persisted_attempts,
            reducer,
            operational,
            state,
            human_review=True,
            root_cause_fingerprint=_gate_item_fingerprint(gate_item),
        )

    def _terminalize_publication_escape(self, escape: "_TerminalArtifactEscape") -> None:
        """Re-raise a terminal artifact failure with its original type and traceback.

        Defensive only: the minimal retry rung passes ``artifact=None``, so its
        append can never reach finalization and never raises the escape. The
        primary rung routes escapes through
        :meth:`_contain_terminal_finalization_refusal` instead -- rung 11
        (2026-08-07) proved the blanket re-raise wrong: the ``_forward_and_reduce``
        failure arms call ``_terminalize`` from INSIDE exception handlers with no
        model-local handler above them, so one model's
        ``authorization dependency vector is stale`` unwound the whole campaign.

        Parameters
        ----------
        escape:
            Internal carrier holding the artifact failure.

        Raises
        ------
        BaseException
            Always; the carried exception, so callers and the CLI see the real type.
        """

        raise escape.cause

    def _contain_terminal_finalization_refusal(
        self,
        escape: "_TerminalArtifactEscape",
        item: WorkItem,
        artifact: Optional[AuthorArtifact],
        status_code: str,
        reason_code: Optional[str],
        detail: Optional[str],
        attempts: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        root_cause_fingerprint: Optional[str],
        superseded_model: Optional[Mapping[str, Any]],
        terminal_gate_obtained: bool,
    ) -> Optional[BaseException]:
        """Land one model's terminal without finalization when finalization refuses.

        Rung 11 (2026-08-07): m10949's resume reloaded an author-result cache
        superseded by two later same-work repair stagings; at terminalization the
        reducer's dependency vector pinned the ledger-latest staged transaction
        while the in-memory artifact carried the cached one, and the exactness
        tripwire refused (``authorization dependency vector is stale``). The
        tripwire is CORRECT -- superseded bytes must never finalize -- but the
        refusal is a fact about ONE model's artifact authority, and the blanket
        re-raise turned it into a campaign abort stranding three siblings.

        Containment, in order: campaign-level causes (pauses, retryable
        transport, provider backoff, shutdown, programming errors) re-raise
        untouched. Everything else -- ``ArtifactBindingError``,
        ``ReductionError``, ``PublicationAuthorizationRefused``, mirror faults --
        is reported LOUD on the operational ledger, then the SAME terminal is
        re-appended with ``finalize_artifact=False``: every frozen fact and the
        artifact's evidence stay in the record, finalization is withheld (the
        artifact ledger keeps whatever state it truthfully reached), and the
        record is flagged for human review so the model can be re-driven under
        the current dependency vector. No rung turns a failure into a pass, and
        the check itself is never weakened -- the refused finalization simply
        does not happen.

        Parameters
        ----------
        escape:
            Carrier holding the finalization failure.
        item, artifact, status_code, reason_code, detail, attempts:
            The exact terminal being recorded.
        reducer, operational, state:
            Canonical authority, operational ledger, and mutable driver state.
        root_cause_fingerprint, superseded_model, terminal_gate_obtained:
            Threaded through unchanged so the retry names the same authority.

        Returns
        -------
        Optional[BaseException]
            ``None`` when the unfinalized terminal landed; otherwise the retry's
            failure, for the caller's existing recording ladder.

        Raises
        ------
        BaseException
            The carried cause, when it is genuinely campaign-level.
        """

        cause = escape.cause
        if isinstance(
            cause, (DriverPaused, RetryableOperatorError, AuthorBackoffError, DriverShutdown)
        ) or isinstance(cause, _PROGRAMMING_ERRORS):
            raise cause
        self._report_terminal_finalization_refused(item, status_code, cause, operational)
        refusal_note = f"artifact finalization refused: {type(cause).__name__}: {cause}"
        try:
            self._append_terminal_revision(
                item,
                artifact,
                status_code,
                reason_code,
                f"{detail}; {refusal_note}" if detail else refusal_note,
                attempts,
                reducer,
                operational,
                state,
                human_review=True,
                root_cause_fingerprint=root_cause_fingerprint,
                superseded_model=superseded_model,
                terminal_gate_obtained=terminal_gate_obtained,
                finalize_artifact=False,
            )
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
            raise
        except _PROGRAMMING_ERRORS:
            raise
        except Exception as retry_error:  # noqa: BLE001 -- ladder continues below
            return retry_error
        return None

    def _report_terminal_finalization_refused(
        self,
        item: WorkItem,
        status_code: str,
        cause: BaseException,
        operational: JsonlLedger,
    ) -> None:
        """Durably report that one terminal's artifact finalization was refused.

        The refusal's TYPE and MESSAGE are the diagnosis (rung 11's was
        ``ArtifactBindingError: authorization dependency vector is stale``), so
        they are recorded before the containment retry: even if that retry also
        refuses, the cause survives in the operational ledger instead of being
        inferred from a stranded-record census.

        Parameters
        ----------
        item:
            Model whose artifact finalization refused.
        status_code:
            Terminal disposition being recorded.
        cause:
            The reducer's or mirror's exact refusal.
        operational:
            Locked append-only operational ledger.
        """

        created_at = self.dependencies.clock()
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
                "disposition": "terminal-finalization-refused",
                # AT-6/AT-Q9: the shared award-seam classification and the
                # lane that spent (or refused to spend) the chain -- m12646's
                # 12:42Z publication ran on THIS lane and the attribution had
                # to be settled forensically.
                "publication_lane": "terminal-finalization",
                "award_seam_classification": _disclosed_award_seam_classification(cause),
                "stable_id": item.stable_id,
                "work_id": item.active_work_id,
                "intended_status_code": status_code,
                "error_type": type(cause).__name__,
                "error": str(cause)[:STDIO_TAIL_MAX_CHARS],
            },
        }
        # The typed per-occurrence identity covers the complete payload and
        # created_at: a later observation appends, while an exact byte replay no-ops.
        event = mint_operational_event(event)
        operational.append(event)

    def _current_record_revision(self, item: WorkItem, reducer: CanonicalReducer) -> Optional[str]:
        """Return the current canonical record revision for one model, if any.

        Parameters
        ----------
        item:
            Work whose canonical record revision is read.
        reducer:
            Active canonical writer holding the current records.

        Returns
        -------
        Optional[str]
            Current record revision, or ``None`` when no record exists yet.
        """

        record = reducer.current_view.get(item.stable_id)
        if not isinstance(record, Mapping):
            return None
        revision = record.get("record_revision")
        return str(revision) if revision is not None else None

    def _report_unrecordable_terminal(
        self,
        item: WorkItem,
        status_code: str,
        reason_code: Optional[str],
        record_error: BaseException,
        fallback_error: Optional[BaseException],
        operational: JsonlLedger,
        *,
        recorded: bool,
        parked: Optional[JsonObject] = None,
        park_error: Optional[BaseException] = None,
    ) -> None:
        """Report on the operational ledger that recording a terminal itself failed.

        A model whose terminal cannot be written is not the same fact as a model that
        was never reached, and neither is a terminal that landed before its progress
        bookkeeping refused. Both are recorded here so the loss is visible rather than
        inferred from a short record count.

        Parameters
        ----------
        item:
            Model whose terminal could not be recorded as intended.
        status_code, reason_code:
            Disposition the driver intended to record.
        record_error:
            Failure raised by the intended terminal append.
        fallback_error:
            Failure raised by the minimal fallback append, when one was attempted.
        operational:
            Locked append-only operational ledger.
        recorded:
            Whether a canonical revision for this model nonetheless landed.
        parked:
            Persisted terminal-refusal park row now claiming the model, when
            one was minted; the event names it so the disclosure and the claim
            are one navigable fact.
        park_error:
            Failure raised by the park append itself, when parking was
            attempted and refused; the model then stays a partition hole and
            this event carries the reason.
        """

        # The occurrence number is part of the event's immutable identity, so it
        # must be derived from the LEDGER, not from a process-local counter: a
        # resumed driver whose counter restarted at 1 re-minted the identity of a
        # PRIOR leg's event for the same repeating failure, and the fresh
        # ``created_at`` made the replay a conflicting append -- the rung-11
        # leg-4 crash chain started exactly here. Counting the model's durable
        # terminal-unrecordable history makes every new observation mint a
        # genuinely NEW identity while identical replays stay byte-idempotent.
        occurrence = 1 + sum(
            1
            for event in operational.iter_records()
            if event.get("event_kind") == OperationalEventKind.TERMINAL_UNRECORDABLE.value
            and event.get("details", {}).get("stable_id") == item.stable_id
        )
        # The exception TYPE and MESSAGE are the whole diagnostic value here, and
        # burying them in a nested report is how a one-line `NameError` came to
        # present as an unexplained partition hole. Summarised once, then surfaced
        # at the event's top level AND carried to the completion assertion.
        error_summary = f"{type(record_error).__name__}: {record_error}"
        if fallback_error is not None:
            error_summary += f" (fallback {type(fallback_error).__name__}: {fallback_error})"
        self._unrecordable_diagnostics[item.stable_id] = error_summary
        operational.append(
            mint_operational_event(
                {
                    "error_summary": error_summary,
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": self.dependencies.clock(),
                    "event_kind": OperationalEventKind.TERMINAL_UNRECORDABLE.value,
                    "status": OperationalEventStatus.RUNNER_FAILED.value,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 1},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": {
                        "stable_id": item.stable_id,
                        "work_id": item.active_work_id,
                        "intended_status_code": status_code,
                        "intended_reason_code": reason_code,
                        "record_error_type": type(record_error).__name__,
                        "record_error": str(record_error),
                        "fallback_error_type": (
                            type(fallback_error).__name__ if fallback_error is not None else None
                        ),
                        "fallback_error": (
                            str(fallback_error) if fallback_error is not None else None
                        ),
                        "canonical_revision_recorded": recorded,
                        "occurrence": occurrence,
                        "unrecordable_streak": len(self._unrecordable_streak),
                        "refusal_parked": parked is not None,
                        "park_row_sha256": (
                            str(parked["row_sha256"]) if parked is not None else None
                        ),
                        "park_error": str(park_error) if park_error is not None else None,
                    },
                }
            )
        )
        if recorded:
            # A record landed; only the post-append bookkeeping refused. That is not a
            # failure of the recording path, so it must not advance the streak.
            return
        if isinstance(record_error, AwardSupersessionError):
            # A POLICY refusal, not a broken path: the award-supersession guard
            # refused a downgrade exactly as designed. Three protected awards
            # in a row tripped the breaker on the step-100 finishing leg
            # (m634/m7362/m9430, 2026-08-15) -- the recording path was working
            # PERFECTLY. Only unexplained recording failures advance this
            # streak; a park APPEND failure is owned by the separate typed
            # award-park breaker (AW-1), never conflated with "recording path
            # broken".
            return
        if item.stable_id not in self._unrecordable_streak:
            self._unrecordable_streak.append(item.stable_id)
        if len(self._unrecordable_streak) >= TERMINAL_UNRECORDABLE_STREAK_LIMIT:
            # The report above is durable BEFORE this raise, so the evidence for the
            # decision survives the stop that the decision causes.
            raise TerminalRecordingUnavailable(
                f"{len(self._unrecordable_streak)} consecutive models could not have a "
                "terminal recorded, so the recording path is broken rather than the "
                f"models: {', '.join(self._unrecordable_streak)}. Last failure "
                f"{error_summary}"
            )

    def _disclose_award_holds(self, operational: JsonlLedger, reducer: CanonicalReducer) -> None:
        """Disclose live award holds at leg start and refuse an unacked launch.

        AW-1/AW-2 launch guard: the eight finishing legs chased stale awards
        the projection reported as nonexistent, and the park treadmill grew to
        17 rows with no leg ever disclosing them. Every leg now censuses, from
        RAW heads plus the projection's stale reasons, (a) runs awards that are
        projection-stale (each one a would-re-drive under a scheduler that
        consumes projection absence as nonexistence) and (b) live undrained
        award-supersession parks. Anything live appends one degraded census
        event; the launch then REFUSES unless the operator acknowledged the
        holds (``acknowledge_award_holds``), so a fix-wave rotation can never
        silently funnel settled awards back into the re-drive treadmill.

        Parameters
        ----------
        operational:
            Locked operational ledger.
        reducer:
            Open canonical reducer.
        """

        raw_heads = reducer.raw_head_view
        stale_reasons = reducer.dependency_projection.stale_reasons
        stale_awards = sorted(
            stable_id
            for stable_id, record in raw_heads.items()
            if record.get("status", {}).get("kind") == "runs" and stable_id in stale_reasons
        )
        records_root = canonical_records_root(self.paths.ledgers.models)
        award_parks = [
            row
            for row in load_grant_park_rows([grant_park_path(records_root)])
            if row.get("disposition") == AWARD_PARK_DISPOSITION
            # Release keys on RAW ledger heads: a projection-stale award is
            # still an award and still releases nothing until a real revision
            # supersedes the blocked head.
            and not grant_park_released(row, raw_heads.get(str(row.get("stable_id"))))
        ]
        if not stale_awards and not award_parks:
            return
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.DEGRADED.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": len(stale_awards) + len(award_parks)},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "award-holds-census",
                "stale_award_count": len(stale_awards),
                "stale_awards": stale_awards[:20],
                "stale_reason_sample": {
                    stable_id: str(stale_reasons[stable_id])[:200] for stable_id in stale_awards[:5]
                },
                "live_award_park_count": len(award_parks),
                "live_award_park_models": sorted(
                    {str(row.get("stable_id")) for row in award_parks}
                )[:20],
                "acknowledged": bool(self.config.acknowledge_award_holds),
                # D-47: the digest this census demands; the acknowledgment
                # must bind exactly this value.
                "census_digest": award_holds_ack_digest(
                    stale_awards,
                    (str(row.get("row_sha256")) for row in award_parks),
                ),
                "acknowledged_digest": self.config.award_holds_digest,
            },
        }
        event = mint_operational_event(event)
        operational.append(event)
        census_digest = str(event["details"]["census_digest"])
        if self.config.acknowledge_award_holds:
            if self.config.award_holds_digest == census_digest:
                return
            # D-47: the blanket acknowledgment is retired — an ack that does
            # not bind THIS census (or binds an older one) refuses typed.
            raise DriverIntegrationError(
                "launch refused: the award-holds acknowledgment must bind the "
                f"exact standing census (D-47). Computed census digest "
                f"{census_digest}; supplied "
                f"{self.config.award_holds_digest!r}. Read the "
                "award-holds-census event, then relaunch with "
                f"--award-holds-digest {census_digest}"
            )
        raise DriverIntegrationError(
            f"launch refused: {len(stale_awards)} projection-stale runs award(s) "
            f"and {len(award_parks)} live award-supersession park(s) are "
            "outstanding. Register the authority rotations (refinement or "
            "soundness), drain the parks (requeue --release-park), or launch "
            "with acknowledge_award_holds AND the binding --award-holds-digest "
            f"{census_digest} after reading the award-holds-census event."
        )

    def _append_award_custody_census(
        self, operational: JsonlLedger, census: Mapping[str, Any]
    ) -> None:
        """Append the close-time award-custody census when anything moved.

        Parameters
        ----------
        operational:
            Locked operational ledger.
        census:
            Result of :func:`_reconcile_award_custody`.
        """

        if not census["award_downgrades"] and not census["award_remints"]:
            return
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": OperationalEventStatus.DEGRADED.value,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {
                "models": len(census["award_downgrades"]) + len(census["award_remints"])
            },
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "award-custody-census",
                "award_downgrades": census["award_downgrades"],
                "award_remints": census["award_remints"],
                "award_downgrade_count": len(census["award_downgrades"]),
                "award_remint_count": len(census["award_remints"]),
            },
        }
        event = mint_operational_event(event)
        operational.append(event)

    def _handle_award_supersession_refusal(
        self,
        item: WorkItem,
        prepared: Mapping[str, Any],
        refused: AwardSupersessionError,
        status_code: str,
        reason_code: Optional[str],
        operational: JsonlLedger,
    ) -> None:
        """Park one award-custody refusal durably, with its own typed breaker.

        A successful park is the model's durable disposition for this leg
        (``review-held:award-supersession-pending-grant``) and resets the
        award-park failure streak. A park APPEND failure is disclosed on the
        operational ledger and advances :data:`AWARD_PARK_FAILURE_STREAK_LIMIT`
        -- the typed :class:`AwardParkUnavailable` breaker -- rather than the
        generic recording streak: the refusal is the guard WORKING, and the
        model then stays a loud partition hole instead of silently degrading.

        Parameters
        ----------
        item, prepared, refused:
            Scheduled work, the reducer-refused candidate, and the typed
            refusal carrying the raw blocked award head.
        status_code, reason_code:
            Disposition the driver intended to record (for the disclosure).
        operational:
            Locked operational ledger.
        """

        from menagerie.crawler.driver_admission import _park_award_supersession

        try:
            _park_award_supersession(self, item, prepared, refused, operational)
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
            raise
        except _PROGRAMMING_ERRORS:
            raise
        except Exception as park_error:  # noqa: BLE001 -- parking may itself fail
            if item.stable_id not in self._award_park_failure_streak:
                self._award_park_failure_streak.append(item.stable_id)
            self._report_unrecordable_terminal(
                item,
                status_code,
                reason_code,
                refused,
                None,
                operational,
                recorded=False,
                parked=None,
                park_error=park_error,
            )
            if len(self._award_park_failure_streak) >= AWARD_PARK_FAILURE_STREAK_LIMIT:
                raise AwardParkUnavailable(
                    f"{len(self._award_park_failure_streak)} consecutive award-"
                    "supersession refusals could not be parked, so the award "
                    "park path is broken rather than the models: "
                    f"{', '.join(self._award_park_failure_streak)}. Last "
                    f"failure {type(park_error).__name__}: {park_error}"
                ) from park_error
            return
        self._award_park_failure_streak.clear()

    def _disclose_host_capability_preflight(
        self,
        work: Sequence[WorkItem],
        operational: JsonlLedger,
    ) -> JsonObject:
        """Print and ledger "N members cannot run on this host", once, before spending.

        The cheapest diagnostic nobody ran in three days. Step-100's five carried
        members were unschedulable on the Mac crawl host since 2026-08-05/06 and said so in every
        one of 123 reuse events; eight finishing legs (~23-28h wall, 357 forward
        re-executions) chased them anyway. This states the needs-vs-actual arithmetic at
        leg start, and REFUSES to launch when nothing in the target set can run here --
        a leg with no runnable member is not a leg, and its only possible outcome is the
        no-progress halt step-100 actually got.

        Parameters
        ----------
        work:
            This leg's ordered target set.
        operational:
            Locked operational ledger receiving the disclosure.

        Returns
        -------
        dict[str, Any]
            The census, for the caller's diagnostics.

        Raises
        ------
        HostCapabilityPreflightError
            If every member of a non-empty target set is host-capability deferred.
        """

        census = host_capability_preflight(
            tuple(item.stable_id for item in work),
            deferral_rows=_persisted_deferral_rows(self),
        )
        if not census["host_deferred_count"]:
            return census
        created_at = self.dependencies.clock()
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": created_at,
            "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
            "status": (
                OperationalEventStatus.HOST_CAPABILITY_LEG_REFUSED.value
                if census["fully_host_deferred"]
                else OperationalEventStatus.HEALTHY.value
            ),
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": int(census["host_deferred_count"])},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "disposition": "host-capability-preflight",
                "target_count": census["target_count"],
                "host_deferred_count": census["host_deferred_count"],
                "host_physical_memory_bytes": census["host_physical_memory_bytes"],
                "host_memory_source": census["host_memory_source"],
                "fully_host_deferred": census["fully_host_deferred"],
                "members": census["members"],
            },
        }
        # Complete-payload identity, like every other campaign-health mint (ID-1:
        # through the typed constructor -- CAMPAIGN_HEALTH is per-occurrence, so
        # the full event including created_at is the preimage): an identical
        # preflight on a resume no-ops instead of tripping the ledger's
        # conflicting-replay tripwire.
        event = mint_operational_event(event)
        operational.append(event)
        summary = ", ".join(
            f"{member['stable_id']} needs "
            f"{int(member['required_physical_memory_bytes']) / 2**30:.1f} GiB"
            for member in census["members"][:10]
        )
        actual = census["host_physical_memory_bytes"]
        actual_gib = "unknown" if actual is None else f"{int(actual) / 2**30:.1f} GiB"
        message = (
            f"{census['host_deferred_count']} of {census['target_count']} members cannot "
            f"run on this host (have {actual_gib}): {summary}"
        )
        print(f"[host-capability] {message}", flush=True)
        if census["fully_host_deferred"]:
            raise HostCapabilityPreflightError(
                "every member of this leg is host-capability deferred -- route them to a "
                f"capable host instead of launching here: {message}"
            )
        return census

    def _terminalize(
        self,
        item: WorkItem,
        artifact: Optional[AuthorArtifact],
        status_code: str,
        reason_code: Optional[str],
        detail: Optional[str],
        attempts: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        human_review: bool = False,
        root_cause_fingerprint: Optional[str] = None,
        superseded_model: Optional[Mapping[str, Any]] = None,
        terminal_gate_obtained: bool = True,
        host_capability_claim: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record one terminal, surviving a failure in the bookkeeping that records it.

        ``_terminalize`` protects the original per-model failure but not the bookkeeping
        that writes it down: it appends attempts, authorizes artifacts and calls
        ``reducer.append_model``, any of which may raise ``ReductionError``,
        ``DriverIntegrationError``, ``ArtifactBindingError`` or ``MirrorError``. None of
        those had a handler between here and the CLI catch-all, so the fix for a
        per-model failure could re-abort inside the handler meant to contain it -- and
        every scoping fix routes MORE traffic through this path.

        The ladder, in order, is deliberate. An artifact FINALIZATION failure is
        contained first: the same record is re-appended with the artifact kept and only
        finalization withheld (``finalize_artifact=False``), reported loud on the
        operational ledger and flagged for human review -- one model's authority
        disagreement (rung 11: ``authorization dependency vector is stale``) is a fact
        about that model, never the campaign. A terminal that landed before its progress
        bookkeeping refused is NOT re-appended: superseding a correct record with a
        worse one would lose the very fact just recorded. Otherwise one minimal retry is
        made with the same disposition but no artifact or derived evidence, since those
        are the parts that can refuse; the already-persisted attempt records are KEPT,
        because the reducer derives terminal authority from the ledger either way and an
        attempt-free record contradicts that authority (its placeholder environment
        generation refuses against the proof's real one) whenever an
        environment-carrying attempt exists. The retry is flagged for human review and
        cannot publish anything. If even that refuses, the model is reported on the
        operational ledger, because an unrecordable model is itself a fact and must not
        become a silent hole in the catalog. No rung can turn a failure into a pass:
        every one of them writes a ``failed:`` disposition or nothing at all.

        Pauses, retryable transport failures and provider backoff still propagate from
        every rung -- they are genuinely campaign-level.

        Parameters
        ----------
        item, artifact, status_code, reason_code, detail, attempts:
            Terminal work, retained artifact, disposition, diagnostics, and decisive
            attempts.
        reducer, operational, state:
            Canonical authority, operational ledger, and mutable driver state.
        human_review:
            Whether the terminal requires explicit human review before requeue.
        root_cause_fingerprint:
            Optional checker-derived failure identity.
        superseded_model:
            Exact current predecessor captured before a decisive attempt made it stale.
        host_capability_claim:
            Machine-derived capability measurement for a host-capability deferral. The
            minimal containment rung deliberately does NOT carry it: that rung exists to
            record a FAILURE when the full append refused, and a deferral it could not
            prove must stay a durable valve row instead.
        """

        guard_error = _non_model_terminal_error(status_code, reason_code, attempts, detail)
        if guard_error is not None:
            # DX-1/RE-1 fail-closed mint sink: a ``failed:*`` disposition whose
            # classified owner is the HOST or the harness MACHINERY may never
            # settle a model's head, whatever call site asked for it. Reaching
            # this guard means an UNGUARDED site classified the fault correctly
            # and still tried to terminalize (the m12446 shape) -- the member
            # parks loudly instead, and the census flags the site to fix.
            self._park_non_model_fault(
                item, artifact, guard_error, operational, sink_guard=True
            )
            return

        before_revision = self._current_record_revision(item, reducer)
        try:
            self._append_terminal_revision(
                item,
                artifact,
                status_code,
                reason_code,
                detail,
                attempts,
                reducer,
                operational,
                state,
                human_review=human_review,
                root_cause_fingerprint=root_cause_fingerprint,
                superseded_model=superseded_model,
                terminal_gate_obtained=terminal_gate_obtained,
                host_capability_claim=host_capability_claim,
            )
            self._unrecordable_streak.clear()
            return
        except _TerminalArtifactEscape as escape:
            contained = self._contain_terminal_finalization_refusal(
                escape,
                item,
                artifact,
                status_code,
                reason_code,
                detail,
                attempts,
                reducer,
                operational,
                state,
                root_cause_fingerprint=root_cause_fingerprint,
                superseded_model=superseded_model,
                terminal_gate_obtained=terminal_gate_obtained,
            )
            if contained is None:
                self._unrecordable_streak.clear()
                return
            record_error: BaseException = contained
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
            raise
        except _PROGRAMMING_ERRORS:
            # A broken program is not a fact about this model, and will recur identically
            # for every model after it. Fail loudly and immediately with the real type.
            raise
        except Exception as exc:  # noqa: BLE001 -- recording a failure may itself fail
            record_error = exc

        if self._current_record_revision(item, reducer) != before_revision:
            # The terminal itself landed; only the bookkeeping after the canonical
            # append refused. Re-appending would supersede a correct record.
            self._report_unrecordable_terminal(
                item, status_code, reason_code, record_error, None, operational, recorded=True
            )
            return

        fallback_detail = f"terminal bookkeeping failed: {record_error}"
        if detail:
            fallback_detail = f"{detail}; {fallback_detail}"
        try:
            self._append_terminal_revision(
                item,
                None,
                status_code,
                reason_code,
                fallback_detail,
                # The SAME already-persisted attempt records, artifact dropped.
                # Passing `()` here made the minimal rung UNRECORDABLE for every
                # model with a persisted environment-carrying attempt: the
                # reducer derives the terminal proof from the LEDGER regardless
                # of what this record names, so the placeholder
                # ``execution.env_generation`` an attempt-free assembly stamps
                # contradicts the proof's real generation and the one rung that
                # exists to survive bookkeeping failures refused
                # deterministically (rung 9: three consecutive models, campaign
                # halt). These records are reducer-blessed machine facts --
                # every caller passes attempts returned by
                # ``reducer.append_attempt`` -- so re-naming them cannot refuse
                # for the artifact-derived reasons this rung drops.
                attempts,
                reducer,
                operational,
                state,
                human_review=True,
                root_cause_fingerprint=None,
                superseded_model=None,
                # The minimal rung drops the artifact, so no finalization can be
                # attempted and no frozen source facts are read. Threaded anyway so
                # the two rungs can never disagree about what authority was held --
                # a silently-defaulted `True` here is exactly the shape that made
                # the merged tree raise `NameError` on every terminal append.
                terminal_gate_obtained=terminal_gate_obtained,
            )
            self._unrecordable_streak.clear()
            return
        except _TerminalArtifactEscape as escape:
            self._terminalize_publication_escape(escape)
        except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
            raise
        except _PROGRAMMING_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 -- the minimal terminal may also refuse
            fallback_error: BaseException = exc

        after_revision = self._current_record_revision(item, reducer)
        recorded = after_revision != before_revision
        parked: Optional[JsonObject] = None
        park_error: Optional[BaseException] = None
        if not recorded:
            # Both rungs refused: park the MODEL so the completion partition
            # accounts it machine-withheld instead of dying with a hole for a
            # fact the run already disclosed. The refusal itself is untouched
            # -- no record is minted, nothing publishes, and the systemic
            # streak backstop below still stops a broken recording path. The
            # park is durable BEFORE the report so it survives the streak's
            # own raise; a park that itself refuses is contained (the model
            # then stays a loud PartitionError hole, exactly as before).
            try:
                parked = _park_terminal_refusal(
                    self,
                    item,
                    status_code=status_code,
                    reason_code=reason_code,
                    record_error=record_error,
                    fallback_error=fallback_error,
                    blocked_parent_revision=after_revision,
                    candidate_sha256=self._refused_candidate_digests.get(item.stable_id),
                )
            except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
                raise
            except _PROGRAMMING_ERRORS:
                raise
            except Exception as exc:  # noqa: BLE001 -- parking a refusal may itself fail
                park_error = exc
        self._report_unrecordable_terminal(
            item,
            status_code,
            reason_code,
            record_error,
            fallback_error,
            operational,
            recorded=recorded,
            parked=parked,
            park_error=park_error,
        )

    def _append_terminal_revision(
        self,
        item: WorkItem,
        artifact: Optional[AuthorArtifact],
        status_code: str,
        reason_code: Optional[str],
        detail: Optional[str],
        attempts: Sequence[Mapping[str, Any]],
        reducer: CanonicalReducer,
        operational: JsonlLedger,
        state: JsonObject,
        *,
        human_review: bool = False,
        root_cause_fingerprint: Optional[str] = None,
        superseded_model: Optional[Mapping[str, Any]] = None,
        terminal_gate_obtained: bool = True,
        finalize_artifact: bool = True,
        host_capability_claim: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append one driver-owned non-run terminal revision through the reducer.

        Parameters
        ----------
        item, artifact, status_code, reason_code, detail, attempts:
            Terminal work, retained artifact, disposition, diagnostics, and decisive attempts.
        reducer, operational, state:
            Canonical authority, operational ledger, and mutable driver state.
        human_review:
            Whether the terminal requires explicit human review before requeue.
        root_cause_fingerprint:
            Optional checker-derived failure identity.
        superseded_model:
            Exact current predecessor captured before a decisive attempt made it stale.
        terminal_gate_obtained:
            Whether the terminal-disposition gate that adjudicates ``artifact`` was
            actually obtained. A retained artifact carries two independent things: the
            EXACT source facts the broker already froze, and the AUTHORITY to finalize
            and publish. Only the second requires an accepted gate. Passing ``False``
            keeps the frozen source facts in the record -- they are machine-derived and
            true whatever the gate did -- while withholding finalization entirely, so
            the record can be honest about what was fetched without ever claiming an
            adjudication that never happened.
        finalize_artifact:
            Containment retry's explicit finalization withholding. Unlike
            ``terminal_gate_obtained=False`` it asserts NOTHING about adjudication --
            the record keeps every fact, including an obtained gate -- it only skips
            the ``_authorize_terminal_artifact`` call that already refused, leaving
            the artifact ledger's truthful state (usually staged-only) as the record's
            artifact authority.
        host_capability_claim:
            Machine-derived capability measurement for a host-capability deferral. It
            carries only the inputs the reducer recomputes the refusal from, so it can
            never assert a head into existence -- a forged field fails the recomputation
            and the terminal refuses.
        """

        terminal_attempts = tuple(attempts)
        gates = reducer.gate_records
        created_at = self.dependencies.clock()
        terminal_diagnostic_reference = (
            _redact_terminal_detail(
                detail,
                item.stable_id,
                status_code,
                created_at,
                _diagnostics_root_for_work_root(self.paths.work_root),
            )
            if status_code.startswith("failed:")
            else None
        )
        model = _assemble_terminal_model(
            item,
            artifact,
            status_code,
            reason_code,
            detail,
            terminal_attempts,
            gates,
            self.config,
            created_at,
            human_review=human_review,
            root_cause_fingerprint=root_cause_fingerprint,
            terminal_diagnostic_reference=terminal_diagnostic_reference,
            terminal_gate_obtained=terminal_gate_obtained,
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
            # A terminal this leg raised WITHOUT running an attempt of its own --
            # the pre-dispatch author-session cap, whose whole point is that
            # earlier legs already spent the allowance -- carries only its
            # synthetic failure attempt, so the environment fields would degrade
            # to `not-applicable` while the reducer's terminal proof still cites
            # those earlier legs' forward attempts and their real environment.
            # Naming the generation's authenticated environment here keeps the
            # candidate self-consistent and leaves `_derive_environment` armed.
            durable_environment_attempt=(
                None
                if any(
                    isinstance((attempt.get("identities") or {}).get("environment"), str)
                    for attempt in terminal_attempts
                )
                else reducer.work_generation_environment_attempt(
                    item.stable_id, item.active_work_id
                )
            ),
            host_capability_claim=host_capability_claim,
        )
        if item.is_family_variant:
            representative_model = reducer.current_records.get(item.family_representative_id)
            if representative_model is None:
                raise DriverIntegrationError(
                    "terminal family variant lost its trusted representative"
                )
            model["family_variant_derivation"] = build_size_variant_derivation(
                representative_model,
                representative_model_id=item.family_representative_id,
                variant_token=item.intake.variant,
            )
            model["accuracy_gate"] = deepcopy(representative_model["accuracy_gate"])
        current_model = (
            superseded_model
            if superseded_model is not None
            else reducer.current_view.get(item.stable_id)
        )
        if current_model is not None and current_model.get("stable_id") != item.stable_id:
            raise DriverIntegrationError("terminal predecessor stable_id does not match work")
        model["parent_revision"] = (
            current_model["record_revision"] if current_model is not None else None
        )
        if current_model is not None:
            model["status"]["supersedes_revision"] = current_model["record_revision"]
        # T2-08 guard, driver half. The AUTOMATIC stamp stays RETIRED
        # (2026-08-14): resume-4's re-adjudication used that arm to replace
        # three VERIFIED awards with failures (m4334/m5273/m7546), and an
        # automatic adjudication may never destroy a verified award. What
        # remains is the DELIBERATE arm (AW-1): when the candidate consumes an
        # operator grant naming the exact blocked award revision, the driver
        # stamps the typed scar as the RECEIPT of that decision -- the scar is
        # minted only at grant consumption, never as a substitute for one.
        # Every other downgrade proceeds unstamped, the reducer's tripwire
        # refuses it, and the refusal parks in the award lane below for
        # OPERATOR adjudication.
        award_supersession_scar: Optional[str] = None
        superseded_award_revision: Optional[str] = None
        award_naming_grant_id: Optional[str] = None
        if current_model is not None and current_model.get("status", {}).get("kind") == "runs":
            award_naming_grant_id = _award_naming_grant_id(self.paths.ledgers, model, current_model)
            if award_naming_grant_id is not None:
                superseded_award_revision = str(current_model["record_revision"])
                award_supersession_scar = (
                    f"{AWARD_SUPERSESSION_SCAR_PREFIX}{superseded_award_revision}"
                )
                scars = list(model.get("scar_history", ()))
                if award_supersession_scar not in scars:
                    scars.append(award_supersession_scar)
                    model["scar_history"] = scars
        prior_artifact_authority = (
            current_model.get("artifact_authority") if current_model is not None else None
        )
        retain_prior_artifact_authority = bool(
            artifact is not None
            and isinstance(prior_artifact_authority, Mapping)
            and prior_artifact_authority.get("state") in {"private-committed", "published"}
        )
        if retain_prior_artifact_authority:
            assert artifact is not None
            assert isinstance(prior_artifact_authority, Mapping)
            transaction_value = prior_artifact_authority.get("transaction_id")
            if not isinstance(transaction_value, str) or not transaction_value:
                raise DriverIntegrationError("retained-artifact-authority-mismatch")
            binding = artifact.author_result.binding
            transaction_key = (
                binding.stable_id,
                binding.work_id,
                ArtifactTransactionId(transaction_value),
            )
            transaction = self._final_artifact_transactions.get(transaction_key)
            if transaction is None:
                try:
                    self._rehydrate_final_authority(item, reducer, operational)
                except DriverIntegrationError as exc:
                    raise DriverIntegrationError("retained-artifact-authority-mismatch") from exc
                transaction = self._final_artifact_transactions.get(transaction_key)
            if transaction is None:
                raise DriverIntegrationError("retained-artifact-authority-mismatch")
            retained_artifact = _artifact_with_final_authority_objects(artifact, transaction)
            self._assert_retained_artifact_authority_matches(
                retained_artifact,
                prior_artifact_authority,
                transaction,
            )
            model["artifact_authority"] = deepcopy(dict(prior_artifact_authority))

        self.dependencies.boundary_hook("pre-publication", item.stable_id)
        self._check_shutdown(
            "pre-publication-admission",
            item=item,
            work_id=(
                artifact.author_result.binding.work_id
                if artifact is not None
                else item.active_work_id
            ),
        )
        self.dependencies.boundary_hook("pre-award-commit", item.stable_id)
        self._check_shutdown(
            "pre-award-commit",
            item=item,
            work_id=(
                artifact.author_result.binding.work_id
                if artifact is not None
                else item.active_work_id
            ),
        )
        self.dependencies.boundary_hook("award-commit-entered", item.stable_id)
        # A supersession blocked ONLY by a missing grant on a review-held
        # terminal must not finalize artifact authority: the pending revision
        # parks below with its artifact evidence intact, and grant consumption
        # stays the only path that ever finalizes or publishes it. The award
        # pre-check composes identically (AW-1): a downgrade the custody guard
        # will refuse must not finalize anything either -- the refused
        # candidate parks with its evidence intact and publishes nothing.
        blocked_previous = reducer.grant_blocked_supersession(model)
        blocked_award_pending = reducer.award_blocked_supersession(model)
        # Terminal materialization and its model append share the same graceful-
        # shutdown atomic section as a successful run award.
        if (
            artifact is not None
            and not retain_prior_artifact_authority
            and terminal_gate_obtained
            and finalize_artifact
            and blocked_previous is None
            and blocked_award_pending is None
        ):
            # Finalization is gate-derived authority. Without the accepted
            # disposition gate there is nothing to authorize, so the artifact
            # stays evidence-only and `_authorize_terminal_artifact`'s
            # "requires its exact gate" tripwire is respected rather than
            # reached and worked around.
            #
            # Materialization is also NOT record bookkeeping, so a failure HERE must
            # escape the `_terminalize` ladder rather than be absorbed by it: keeping
            # the original `skipped:`/`deferred:` status while DROPPING the artifact
            # makes a record the reducer rightly refuses. The escape hands the failure
            # to `_contain_terminal_finalization_refusal`, which re-appends the SAME
            # record with the artifact KEPT and only `finalize_artifact=False` -- so
            # the facts stay, the refused finalization simply does not happen, and one
            # model's authority disagreement can no longer unwind the campaign
            # (rung 11, 2026-08-07).
            #
            # The guards compose: the gate condition decides WHETHER finalization is
            # authorized at all, `finalize_artifact` is the containment retry's
            # explicit withholding, and the escape decides who owns a finalization
            # that fails.
            try:
                self._authorize_terminal_artifact(artifact, model, gates, reducer)
            except (DriverPaused, RetryableOperatorError, AuthorBackoffError):
                raise
            except Exception as exc:  # noqa: BLE001 -- routed to the caller's handler
                raise _TerminalArtifactEscape(exc) from exc
        prepared = reducer.prepare_model(model)
        self._refused_candidate_digests[item.stable_id] = stable_hash(prepared)
        try:
            result = reducer.append_model(prepared)
        except AwardSupersessionError as refused:
            if blocked_award_pending is None:
                # The pre-check and the reducer disagreed; nothing was
                # withheld coherently, so the refusal stays loud.
                raise
            # AW-1: a machine re-adjudication yielding a worse disposition
            # PARKS -- it never appends over the award, nothing publishes,
            # and nothing is ever auto-granted. The park binds the reducer-
            # returned RAW blocked head; its append failure advances the
            # typed award-park breaker, never the generic recording streak.
            self._handle_award_supersession_refusal(
                item, prepared, refused, status_code, reason_code, operational
            )
            return
        except GrantPendingSupersessionError as refused:
            if blocked_previous is None:
                # The pre-check and the reducer disagreed; nothing was
                # withheld coherently, so the refusal stays loud.
                raise
            # The one parkable refusal: everything else about the candidate
            # validated, so the completed supersession waits durably for an
            # operator grant instead of failing the run at partition close.
            _park_grant_pending_supersession(self, item, prepared, refused, operational)
            return
        if blocked_previous is not None:
            raise DriverIntegrationError(
                "grant-pending pre-check withheld finalization but the reducer "
                f"accepted the supersession for {item.stable_id}"
            )
        if blocked_award_pending is not None:
            raise DriverIntegrationError(
                "award pre-check withheld finalization but the reducer "
                f"accepted the supersession for {item.stable_id}"
            )
        self._refused_candidate_digests.pop(item.stable_id, None)
        if result.appended and award_supersession_scar is not None:
            # The event half of the deliberate award-supersession contract:
            # the operator decision and its scar receipt are visible on the
            # operational ledger the moment they land, never only in the
            # record body -- and NEVER ``healthy`` (AW-1: the three destroyed
            # awards' events all read healthy, 1 of 315 identical lines).
            scar_event: JsonObject = {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "created_at": created_at,
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
                    "disposition": "award-superseded-by-grant",
                    "stable_id": item.stable_id,
                    "superseded_revision": superseded_award_revision,
                    "superseding_revision": result.record["record_revision"],
                    "status_code": status_code,
                    "reason_code": reason_code,
                    "scar": award_supersession_scar,
                    "grant_id": award_naming_grant_id,
                },
            }
            # Complete-payload identity: an idempotent resume replaying the
            # identical supersession no-ops instead of conflicting.
            scar_event = mint_operational_event(scar_event)
            operational.append(scar_event)
        if result.appended:
            self._reduced += 1
        # Reduced (even if deduplicated): this model is no longer strandable.
        with self._author_lane_state_lock:
            self._reduced_model_ids.add(item.stable_id)
        self.dependencies.boundary_hook("post-award-commit", item.stable_id)
        self._check_shutdown("post-award-commit")
        self.dependencies.boundary_hook("after-reduce", item.stable_id)
        current_records = reducer.current_view
        snapshot = self._policy_snapshot()
        self._handle_progress(operational, current_records, snapshot, state=state)
        if self._maybe_pause_for_review(operational, current_records, snapshot, state):
            raise DriverPaused("review checkpoint reached")
        state.update({"last_terminal_count": len(current_records), "status": "running"})
        _write_driver_state(self.paths.driver_state, state)

    def _assert_retained_artifact_authority_matches(
        self,
        artifact: AuthorArtifact,
        authority: Mapping[str, Any],
        transaction: ArtifactTransactionProjection,
    ) -> None:
        """Require retained final authority to identify the exact new artifact.

        Parameters
        ----------
        artifact:
            New terminal artifact that would inherit prior finalized authority.
        authority:
            Prior model revision's finalized artifact-authority projection.
        transaction:
            Single final transaction projection already resolved and validated by
            :meth:`_rehydrate_final_authority`.

        Raises
        ------
        DriverIntegrationError
            If any final event, reconstruction input, claim, work, proposal, or
            private-custody identity differs from the new terminal artifact.
        """

        if artifact.staged is None:
            raise DriverIntegrationError("retained-artifact-authority-mismatch")
        binding = artifact.author_result.binding
        inputs = transaction.reconstruction_inputs
        raw_result = artifact.author_result.binding.raw_result
        expected_proposal = (
            artifact.proposal if isinstance(artifact.author_result, ProposedAuthorResult) else None
        )
        expected_handoff = (
            raw_result.get("payload", {}).get("handoff_execution")
            if isinstance(artifact.author_result, DeferRecommendation)
            else None
        )
        claim_ids = sorted(str(claim.claim_id) for claim in transaction.claims)
        if (
            transaction.stable_id != binding.stable_id
            or transaction.work_id != binding.work_id
            or transaction.transaction_id != artifact.staged.transaction_id
            or transaction.final_event_id != authority.get("committed_event_id")
            or transaction.final_event_kind != authority.get("state")
            or transaction.authorization_id != authority.get("authorization_id")
            or transaction.reconstruction_sha256 != authority.get("reconstruction_sha256")
            or claim_ids != authority.get("claim_ids")
            or transaction.objects != artifact.staged.objects
            or inputs.author_result != raw_result
            or inputs.proposal != expected_proposal
            or inputs.handoff_execution != expected_handoff
            or inputs.source_manifest != artifact.source_manifest
        ):
            raise DriverIntegrationError("retained-artifact-authority-mismatch")

    def _pause_for_usage(
        self, signal: UsageBackoffSignal, operational: JsonlLedger, queued: int
    ) -> str:
        """Record a visible provider pause and schedule one idempotent reset wakeup.

        The provider is derived from the signal: the checker lane pauses on
        ``openai`` and the author lane on ``anthropic``. Both members are already
        accepted by the wakeup layer's closed provider vocabulary.
        """

        reset_observation = "observed" if signal.reset_at is not None else "guessed"
        reset_at = _normalize_wake_reset(
            signal.reset_at or _future_reset(self.dependencies.clock(), signal)
        )
        provider = signal.provider
        context = self._context(queued, None)
        created_at = self.dependencies.clock()
        scheduler = self.dependencies.usage_pause_scheduler
        if scheduler is not None:
            scheduler.schedule(
                signal,
                operational,
                context,
                created_at,
                reset_at,
                reset_observation,
            )
        else:
            manager = WakeupManager(
                self.paths.wakeup_root,
                operational,
                self._wakeup_callback_argv(),
                installer=self.dependencies.wakeup_installer,
                verifier=self.dependencies.wakeup_verifier,
                deactivator=self.dependencies.wakeup_deactivator,
            )
            scheduled = manager.record_pause_and_schedule(
                provider=provider,
                observed_response=signal.response_excerpt,
                reset_at=reset_at,
                reset_observation=reset_observation,
                context=context,
                created_at=created_at,
            )
            if not scheduled.verified:
                raise DriverIntegrationError("recurring wake projection was not verified")
        driver_state: JsonObject = {
            "status": "paused:usage-limit",
            "provider": provider,
            "reset_at": reset_at,
        }
        if provider in ("research-tools", "forge"):
            # Neither is a usage limit, so `paused:usage-limit` alone would not say
            # what actually happened or what to do about it. The reason and the
            # observed detail carry that.
            driver_state.update(
                {
                    "reason": signal.reason.value,
                    "detail": signal.response_excerpt,
                }
            )
        _write_driver_state(self.paths.driver_state, driver_state)
        return signal.reason.value

    def _handle_progress(
        self,
        operational: JsonlLedger,
        current: Mapping[str, JsonObject],
        intake_snapshot: IntakeSnapshot,
        *,
        state: JsonObject,
    ) -> None:
        """Derive every unrecorded crossed milestone from canonical facts alone."""

        completed = len(current)
        intake_snapshot_id = intake_snapshot.snapshot_id
        intake_snapshot_sha256 = intake_snapshot.snapshot_sha256
        canonical_events = operational.iter_records()
        existing = {
            str(event.get("details", {}).get("policy_key"))
            for event in canonical_events
            if event.get("event_kind") == OperationalEventKind.PROGRESS_NOTIFICATION.value
            and isinstance(event.get("details", {}).get("policy_key"), str)
        }
        snapshot = _funnel_snapshot(current)
        for milestone in sorted(self.config.progress_milestones):
            policy_key = stable_hash(
                {
                    "policy": "crawler-progress-milestone-v1",
                    "intake_snapshot_id": intake_snapshot_id,
                    "intake_snapshot_sha256": intake_snapshot_sha256,
                    "milestone": milestone,
                }
            )
            if policy_key in existing or milestone > completed:
                continue
            created_at = self.dependencies.clock()
            event = {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "created_at": created_at,
                "event_kind": OperationalEventKind.PROGRESS_NOTIFICATION.value,
                "status": OperationalEventStatus.PROGRESS_RECORDED.value,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 0},
                "current_environment": None,
                "run_id": self.config.run_id,
                "machine_id": self.config.machine_id,
                "details": {
                    "identity_only": True,
                    "policy_key": policy_key,
                    "intake_snapshot_id": intake_snapshot_id,
                    "intake_snapshot_sha256": intake_snapshot_sha256,
                },
                "models_completed": completed,
                "milestone": milestone,
                "funnel_snapshot": snapshot.to_dict(),
            }
            event = mint_operational_event(
                event,
                identity_payload={"policy_key": policy_key},
            )
            event_id = str(event["event_id"])
            operational.append(event)
            self.dependencies.boundary_hook("after-notification-identity", event_id)
            self._deliver_notification(
                operational,
                event_id,
                _progress_summary(completed, milestone, snapshot),
                notification_event="progress-milestone",
            )
        state["last_terminal_count"] = completed

    def _retry_notification_outbox(self, operational: JsonlLedger) -> None:
        """Retry every durable milestone/checkpoint identity lacking delivery completion.

        Parameters
        ----------
        operational:
            Locked append-only operational ledger containing identities and attempts.
        """

        canonical_events = operational.iter_records()
        durable_events_by_id = {str(event.get("event_id")): event for event in canonical_events}
        durable_events = tuple(durable_events_by_id.values())
        delivered = {
            str(event.get("details", {}).get("notification_event_id"))
            for event in durable_events
            if event.get("event_kind") == OperationalEventKind.NOTIFICATION_DELIVERY.value
            and event.get("status") == OperationalEventStatus.NOTIFICATION_DELIVERED.value
        }
        for event in durable_events:
            event_id = str(event.get("event_id"))
            if event_id in delivered:
                continue
            kind = event.get("event_kind")
            snapshot_value = event.get("funnel_snapshot", {})
            if not isinstance(snapshot_value, Mapping):
                continue
            snapshot = FunnelSnapshot(
                runs=int(snapshot_value.get("runs", 0)),
                deferred=int(snapshot_value.get("deferred", 0)),
                skipped=int(snapshot_value.get("skipped", 0)),
                failed=int(snapshot_value.get("failed", 0)),
            )
            if kind == OperationalEventKind.PROGRESS_NOTIFICATION.value:
                summary = _progress_summary(
                    int(event["models_completed"]), int(event["milestone"]), snapshot
                )
                notification_event = "progress-milestone"
            elif kind == OperationalEventKind.CHECKPOINT_REVIEW.value:
                summary = _review_summary(
                    int(event["models_completed"]), snapshot, Path(str(event["report_path"]))
                )
                notification_event = "review-checkpoint"
            else:
                continue
            self._deliver_notification(
                operational, event_id, summary, notification_event=notification_event
            )

    def _deliver_notification(
        self,
        operational: JsonlLedger,
        notification_event_id: str,
        summary: str,
        *,
        notification_event: Optional[str] = None,
    ) -> bool:
        """Record best-effort delivery separately from its durable notification identity.

        ``notification_event`` names the alert-event identity for the class
        policy; ``None`` classifies as unclassified and always attempts
        delivery (fail-open).
        """

        canonical_events = operational.iter_records()
        combined_by_id = {str(event.get("event_id")): event for event in canonical_events}
        combined = tuple(combined_by_id.values())
        completed = [
            event
            for event in combined
            if event.get("event_kind") == OperationalEventKind.NOTIFICATION_DELIVERY.value
            and event.get("details", {}).get("notification_event_id") == notification_event_id
            and event.get("status") == OperationalEventStatus.NOTIFICATION_DELIVERED.value
        ]
        if completed:
            return True
        prior_attempts = sum(
            event.get("event_kind") == OperationalEventKind.NOTIFICATION_DELIVERY.value
            and event.get("details", {}).get("notification_event_id") == notification_event_id
            for event in combined
        )
        idempotency_key = stable_hash(
            {
                "notification_event_id": notification_event_id,
            }
        )
        notification_class = (
            None if notification_event is None else classify_notification_event(notification_event)
        )
        try:
            delivered = bool(
                self.dependencies.notifier.notify(
                    summary,
                    idempotency_key=idempotency_key,
                    notification_class=(
                        None if notification_class is None else notification_class.value
                    ),
                )
            )
            error: Optional[str] = None
        except Exception as exc:  # noqa: BLE001 -- injected notifiers are also best-effort
            delivered = False
            error = stable_hash(
                {
                    "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "message": str(exc),
                }
            )
            LOGGER.warning("crawler notifier failed (%s): %s", type(exc).__qualname__, summary)
        delivery_event = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "created_at": self.dependencies.clock(),
            "event_kind": OperationalEventKind.NOTIFICATION_DELIVERY.value,
            "status": (
                OperationalEventStatus.NOTIFICATION_DELIVERED.value
                if delivered
                else OperationalEventStatus.NOTIFICATION_FAILED.value
            ),
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 0},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                "notification_event_id": notification_event_id,
                "idempotency_key": idempotency_key,
                "attempt": prior_attempts + 1,
                "completion": delivered,
                "delivered": delivered,
                "error": error,
            },
        }
        delivery_event = mint_operational_event(
            delivery_event,
            identity_payload={
                "idempotency_key": idempotency_key,
                "attempt": prior_attempts + 1,
            },
        )
        operational.append(delivery_event)
        return delivered

    def _effective_scheduler_mode(self, operational: JsonlLedger) -> str:
        """Return the scheduler mode the campaign's last leg ran under.

        Derived exclusively from the durable operational ledger -- never from
        ``driver-state.json``, which the rollback procedure discards as a
        disposable cursor. Campaigns predating the transition events (or that
        never switched) read as ``waves``, the historical default.

        Parameters
        ----------
        operational:
            Locked append-only operational ledger.

        Returns
        -------
        str
            ``"waves"`` or ``"pipelined"``.
        """

        mode = "waves"
        for event in operational.iter_records():
            if event.get("event_kind") in (
                OperationalEventKind.SCHEDULER_CUTOVER.value,
                OperationalEventKind.SCHEDULER_ROLLBACK.value,
            ):
                mode = str(event.get("details", {}).get("to_scheduler", mode))
        return mode

    def _leg_config_details(self) -> JsonObject:
        """Return this leg's effective run configuration as durable details.

        The recorded set is exactly what decides the SHAPE of the leg: which
        scheduler ran it, how wide each lane was, which author/checker models
        adjudicated it, which selection scope it ran under, and which stable IDs
        the named lanes admitted. ``config_digest`` content-addresses that set,
        so "did these two legs run the same config?" is one field comparison
        rather than a field-by-field diff of the whole record.

        Returns
        -------
        dict[str, object]
            Effective configuration plus its content address.
        """

        config = self.config
        effective: JsonObject = {
            "scheduler": config.scheduler,
            "author_concurrency": config.author_concurrency,
            # The leg's first-author-wall grant (M10/OB-Q6): the monitor's
            # NN-6 stall suppression is bounded by this value, and reading it
            # from the attested leg-config instead of an operator-retyped flag
            # closes the silent mis-timing seam TRI-B named.
            "author_wall_seconds": float(AUTHOR_SESSION_WALL_SECONDS),
            "checker_concurrency": config.checker_concurrency,
            "author_model": config.author_model,
            "author_version": config.author_version,
            "claude_cli_version": config.claude_cli_version,
            "checker_model": config.checker_model,
            "checker_version": config.checker_version,
            "only_unsettled": config.only_unsettled,
            "recheck": config.recheck,
            "only_status": config.only_status,
            "leg_purpose": config.leg_purpose,
            "redrive_targets": list(config.redrive_targets),
            "redrive_reason": config.redrive_reason,
            "redrive_manifest_sha256": self._redrive_manifest_sha256(),
            "gate_catchup_targets": list(config.gate_catchup_targets),
            "ack_stale_excluded": config.ack_stale_excluded,
            "acknowledge_award_holds": config.acknowledge_award_holds,
            "award_holds_digest": config.award_holds_digest,
            "stale_excluded_digest": config.stale_excluded_digest,
            "campaign_id": config.campaign_id,
            "target": config.target,
            "phase": config.phase,
            "review_checkpoint_at": config.review_checkpoint_at,
            "run_repair_max": config.run_repair_max,
            "invocation_origin": config.invocation_origin.value,
            "wake_episode_id": config.wake_episode_id,
            # W3/TE-3: calibration mode and the complete verdict-steering
            # env-read inventory are part of the leg's effective config -- a
            # bypass-adjudicated leg must be distinguishable in the ledgers.
            "checker_calibration": calibration_environment_snapshot(),
            # D-63 (PM-1): which channel decided each lane width. The 200a
            # ledger claimed an override "in effect" that nothing consumed;
            # provenance beside the value makes that lie impossible to repeat.
            "author_concurrency_source": config.author_concurrency_source,
            "checker_concurrency_source": config.checker_concurrency_source,
        }
        # D-63: the running tree joins the attestation (disclosed, never
        # refusing -- the certificate owns enforcement).
        tree = observed_repo_tree(self.paths.runtime_root.parent)
        effective["repo_tree_sha"] = tree["tree_sha"]
        effective["repo_tree_dirty"] = tree["dirty"]
        # D-63: one digest over exactly the RUNG-INVARIANT axes. The full
        # config_digest legitimately varies per leg (wake episode, invocation
        # origin, named lanes), so rung certification compares THIS field:
        # legs of one certified rung must agree on every axis named here.
        effective["rung_config_digest"] = stable_hash(
            {field: effective[field] for field in RUNG_INVARIANT_CONFIG_FIELDS}
        )
        return {**effective, "config_digest": stable_hash(effective)}

    def _reconcile_orphaned_legs(self, operational: JsonlLedger) -> None:
        """Close every orphaned leg span as ``orphaned_uncontained`` (W6).

        A ``leg_started`` no ``leg_ended`` references is a leg that died with
        no exit write at all -- a kill -9, a power cut, an OOM. We hold the
        driver lock, so the span's process is provably gone; the reconciler
        synthesizes its typed close (the ONLY lawful producer of
        ``orphaned_uncontained``) before a new leg opens, and mirrors the
        OB-M12 leg-outcome event when the orphan had attested a leg-config.
        """

        from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            load_leg_events,
            orphaned_leg_started_rows,
            record_leg_ended,
        )

        records_root = canonical_records_root(self.paths.ledgers.models)
        try:
            events = load_leg_events(records_root)
        except Exception:  # noqa: BLE001 -- an absent/empty ledger has no orphans
            return
        orphans = orphaned_leg_started_rows(events)
        if not orphans:
            return
        config_run_ids = {
            str(event.get("run_id"))
            for event in operational.records
            if event.get("event_kind") == OperationalEventKind.LEG_CONFIG.value
        }
        for orphan in orphans:
            fingerprint = cause_fingerprint(
                LegExitClass.ORPHANED_UNCONTAINED.value,
                evidence="no-exit-write",
            )
            ended = record_leg_ended(
                records_root,
                campaign_id=str(orphan["campaign_id"]),
                generation_id=str(orphan["generation_id"]),
                run_id=str(orphan["run_id"]),
                created_at=self.dependencies.clock(),
                leg_started_sha256=str(orphan["row_sha256"]),
                exit_class=LegExitClass.ORPHANED_UNCONTAINED.value,
                cause_fingerprint=fingerprint,
                terminal_delta=None,
                invocation_origin=str(orphan.get("invocation_origin", "unknown")),
                detail=(
                    f"boot reconciler: leg_started pid={orphan.get('pid')} "
                    "wrote no exit"
                ),
                synthesized=True,
            )
            if str(orphan["run_id"]) in config_run_ids:
                self._append_leg_outcome_event(
                    operational,
                    generation_id=str(orphan["generation_id"]),
                    exit_class=LegExitClass.ORPHANED_UNCONTAINED.value,
                    fingerprint=fingerprint,
                    terminal_delta=None,
                    invocation_origin=str(orphan.get("invocation_origin", "unknown")),
                    synthesized=True,
                    leg_ended_sha256=str(ended["row_sha256"]),
                )

    def _check_repeat_cause_fuse(self, operational: JsonlLedger) -> None:
        """Refuse the leg when PM-5's repeat-cause fuse holds unacknowledged."""

        from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            load_leg_events,
        )

        records_root = canonical_records_root(self.paths.ledgers.models)
        try:
            events = load_leg_events(records_root)
        except Exception:  # noqa: BLE001 -- no ledger, no history, no fuse
            return
        fingerprint = repeat_cause_fuse_tripped(events)
        if fingerprint is None:
            return
        rationale = self.config.acknowledge_repeat_cause
        if not rationale:
            raise LegStartRefused(
                StopClass.REPEATED_CAUSE_LEG_DEATH.value,
                "repeat-cause fuse refuses leg start: the last "
                "legs died of the identical cause with zero terminals minted "
                f"(cause_fingerprint {fingerprint}). Re-running the same death "
                "is flail, not progress -- pass --acknowledge-repeat-cause "
                "'<rationale>' to fund one acknowledged retry (a durable "
                "operator row), or fix the cause (PM-5)",
            )
        operational.append(
            mint_operational_event(
                {
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
                        "disposition": "repeat-cause-fuse-acknowledged",
                        "cause_fingerprint": fingerprint,
                        "rationale": str(rationale),
                        "invocation_origin": self.config.invocation_origin.value,
                    },
                }
            )
        )

    def _record_leg_started(self) -> None:
        """Open this leg's durable lifecycle span (W6)."""

        from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            record_leg_started,
        )

        generation = self._leg_generation
        if generation is None:
            raise DriverIntegrationError("leg_started requires an open generation")
        row = record_leg_started(
            canonical_records_root(self.paths.ledgers.models),
            campaign_id=bound_campaign_id(self.config),
            generation_id=generation.generation_id,
            run_id=self.config.run_id,
            created_at=self.dependencies.clock(),
            invocation_origin=self.config.invocation_origin.value,
            invocation_ordinal=int(self.config.invocation_ordinal or 0),
            pid=os.getpid(),
        )
        self._leg_lifecycle = {
            "campaign_id": bound_campaign_id(self.config),
            "generation_id": generation.generation_id,
            "run_id": self.config.run_id,
            "leg_started_sha256": str(row["row_sha256"]),
            "invocation_origin": self.config.invocation_origin.value,
        }

    def _leg_terminal_delta(self) -> Optional[int]:
        """Return terminals minted this leg, ``None`` when unobservable."""

        try:
            current = (
                len(tuple(scan_jsonl(self.paths.ledgers.models)))
                if self.paths.ledgers.models.exists()
                else 0
            )
        except Exception:  # noqa: BLE001 -- disclosure, never a second fault
            return None
        return max(0, current - int(self._models_row_baseline))

    def _record_leg_ended_for_result(self, result: DriverResult) -> None:
        """Pair the live span with its typed outcome on a returning exit."""

        exit_class = classify_result_status(result.status)
        fingerprint = cause_fingerprint(exit_class, evidence=str(result.status))
        self._record_leg_ended(exit_class, fingerprint, detail=str(result.status))

    def _record_leg_ended_for_exception(self, exc: BaseException) -> None:
        """Pair the live span with its typed outcome on a raising exit."""

        exit_class, evidence = classify_exception(exc)
        chain = exception_type_chain(exc)
        fingerprint = cause_fingerprint(
            exit_class, exception_types=chain, evidence=evidence
        )
        self._record_leg_ended(
            exit_class, fingerprint, detail=f"{type(exc).__qualname__}: {exc}"
        )

    def _record_leg_ended(
        self, exit_class: str, fingerprint: str, *, detail: str
    ) -> None:
        """Emit the leg_ended row (and OB-M12 event) for the live span, once.

        Runs OUTSIDE ``_run_locked``'s ledger contexts (exactly like
        ``_record_driver_failure``), so it opens its own operational handle
        for the leg-outcome mirror. A failure to write the outcome must never
        mask the leg's own exit: it is reported on stderr and swallowed.
        """

        lifecycle = self._leg_lifecycle
        self._leg_lifecycle = None
        if lifecycle is None:
            return
        from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: keeps driver import cost flat
            record_leg_ended,
        )

        delta = self._leg_terminal_delta()
        try:
            ended = record_leg_ended(
                canonical_records_root(self.paths.ledgers.models),
                campaign_id=lifecycle["campaign_id"],
                generation_id=lifecycle["generation_id"],
                run_id=lifecycle["run_id"],
                created_at=self.dependencies.clock(),
                leg_started_sha256=lifecycle["leg_started_sha256"],
                exit_class=exit_class,
                cause_fingerprint=fingerprint,
                terminal_delta=delta,
                invocation_origin=lifecycle["invocation_origin"],
                detail=detail,
            )
        except Exception as write_failure:  # noqa: BLE001 -- never mask the exit
            print(
                f"leg_ended write failed (outcome lost to the ledger): {write_failure}",
                file=sys.stderr,
            )
            return
        if not self._leg_config_emitted:
            return
        self._leg_config_emitted = False
        try:
            with JsonlLedger(
                self.paths.operational_ledger,
                OPERATIONAL_EVENT_SCHEMA_VERSION,
                lock_timeout=10.0,
            ) as operational:
                self._append_leg_outcome_event(
                    operational,
                    generation_id=lifecycle["generation_id"],
                    exit_class=exit_class,
                    fingerprint=fingerprint,
                    terminal_delta=delta,
                    invocation_origin=lifecycle["invocation_origin"],
                    synthesized=False,
                    leg_ended_sha256=str(ended["row_sha256"]),
                )
                # DX-26 (RATIFIED, RULING W7 clause 15 under the OPEN-6
                # default): a close's existence asserts the leg ended
                # honestly, so EVERY other leg exit leaves the immutable
                # nonterminal leg-partial.v1 checkpoint -- the D-20 preview
                # payload at exit. It never satisfies the close grammar,
                # never joins the predecessor chain, never settles a member,
                # and licenses nothing; it exists so an interrupted leg's
                # readiness state is durable evidence instead of a hand tally.
                if self.config.invocation_shape != "crawl":
                    # RC-1: a crawl invocation has no close, so it leaves no
                    # close-shaped checkpoint; its typed leg_ended row IS the
                    # durable exit evidence.
                    self._emit_close_preview(
                        operational,
                        at="leg-exit",
                        generation_id=lifecycle["generation_id"],
                        extra_details={
                            "checkpoint": "leg-partial.v1",
                            "exit_class": exit_class,
                            "leg_ended_sha256": str(ended["row_sha256"]),
                        },
                    )
        except Exception as event_failure:  # noqa: BLE001 -- never mask the exit
            print(
                f"leg-outcome event append failed: {event_failure}", file=sys.stderr
            )

    def _append_leg_outcome_event(
        self,
        operational: JsonlLedger,
        *,
        generation_id: str,
        exit_class: str,
        fingerprint: str,
        terminal_delta: Optional[int],
        invocation_origin: str,
        synthesized: bool,
        leg_ended_sha256: str,
    ) -> None:
        """Append the OB-M12 leg-outcome mirror of one leg_ended row."""

        operational.append(
            mint_operational_event(
                {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": self.dependencies.clock(),
                    "event_kind": OperationalEventKind.LEG_OUTCOME.value,
                    "status": OperationalEventStatus.HEALTHY.value,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 0},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": {
                        "generation_id": generation_id,
                        "exit_class": exit_class,
                        "cause_fingerprint": fingerprint,
                        "terminal_delta": terminal_delta,
                        "invocation_origin": invocation_origin,
                        "synthesized": synthesized,
                        "leg_ended_sha256": leg_ended_sha256,
                    },
                }
            )
        )

    def _emit_close_preview(
        self,
        operational: JsonlLedger,
        *,
        at: str = "leg-start",
        generation_id: Optional[str] = None,
        extra_details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Append one read-only close preview event (D-20/LC-7, DX-26).

        Derives, through the DX-24 probe path (the exact production boundary
        predicate -- one implementation, zero drift), whether the close WOULD
        refuse today and which members it would refuse on, and discloses the
        verdict as ONE per-phase operational event. 8-11h of 200a was lost to
        refusals whose evidence rows had been on disk for hours; this makes
        them legible at leg START (``at="leg-start"``) and again at every leg
        EXIT, where the same payload is DX-26's immutable nonterminal
        ``leg-partial.v1`` checkpoint (RULING W7 clause 6 / clause 15: a
        close's existence asserts the leg ended honestly; every other exit
        leaves this checkpoint instead -- which never satisfies the close
        grammar, never settles a member, and licenses nothing).

        Report-only by contract: the probe path takes no lock and triggers no
        recovery writes; a preview that cannot run appends a DEGRADED
        ``preview-error`` event and the leg continues -- the preview may never
        kill a leg, upgrade a verdict, or replace the final replay.
        """

        records_root = canonical_records_root(self.paths.ledgers.models)
        created_at = self.dependencies.clock()
        details: JsonObject
        status = OperationalEventStatus.HEALTHY.value
        try:
            from menagerie.crawler.tools.shadow_close_probe import (  # noqa: PLC0415 -- lazy: breaks the driver<->cli import cycle
                run_probe,
            )

            would_refuse, refusal, payload = run_probe(
                intake=self.paths.intake_root,
                records_root=records_root,
                # The campaign manifest resolves repository-relative
                # (menagerie/crawler/records/campaigns.json), so the probe
                # needs the CANONICAL repository root -- the same derivation
                # publication uses. ``intake_root.parent`` doubled the path
                # under the production nested intake
                # (<repo>/menagerie/crawler/records/intake/<snapshot>) and
                # blinded every per-leg preview with a manifest-resolution
                # error (P5-R3-SOL-02).
                repo_root=_canonical_repo_root(_canonical_crawler_root(self.paths)),
            )
            details = {
                "verdict": "would-refuse" if would_refuse else "would-close",
                "refusal": refusal,
                "partition_valid": payload.get("partition_valid"),
                "missing": payload.get("missing"),
                "stale": payload.get("stale"),
                "review_holds": payload.get("review_holds"),
                "refusal_parks": payload.get("refusal_parks"),
            }
            if would_refuse:
                status = OperationalEventStatus.DEGRADED.value
        except Exception as exc:  # noqa: BLE001 -- report-only: a broken preview is a fact, never a leg killer
            details = {
                "verdict": "preview-error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            status = OperationalEventStatus.DEGRADED.value
        event: JsonObject = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "event_kind": OperationalEventKind.CLOSE_PREVIEW.value,
            "created_at": created_at,
            "status": status,
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {},
            "current_environment": None,
            "run_id": self.config.run_id,
            "machine_id": self.config.machine_id,
            "details": {
                **details,
                "at": at,
                **dict(extra_details or {}),
                "generation_id": generation_id
                if generation_id is not None
                else (
                    self._leg_generation.generation_id
                    if self._leg_generation is not None
                    else None
                ),
            },
        }
        operational.append(mint_operational_event(event))

    def _record_leg_config(self, operational: JsonlLedger) -> None:
        """Append the ONE leg-start effective-config attestation (B-09).

        Every leg records the config it ran under, exactly once, before any
        admission decision. At the audit base this was recoverable only from an
        operator's shell history: which scheduler a leg used, how wide its lanes
        were, which checker model adjudicated its gates, and whether it ran
        ``--only-unsettled`` or a named lane were all invisible in the ledgers,
        so a behavioral difference between two legs could not be attributed to
        the config that caused it.

        Per-occurrence identity: two legs launched with identical flags are two
        separate attestations (their occurrence axis is ``created_at``), so a
        relaunch never dedups into the first leg's row.

        Boundary: the call site sits AFTER lifecycle reconciliation, so a
        ``wake-noop`` invocation -- a wake callback whose episode says not to
        resume, which selects no work and admits nothing -- appends no config
        event. It is not a leg; its own ``wake-noop-already-running`` event is
        the record of what happened.
        """

        self._leg_config_emitted = True
        operational.append(
            mint_operational_event(
                {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": self.dependencies.clock(),
                    "event_kind": OperationalEventKind.LEG_CONFIG.value,
                    "status": OperationalEventStatus.HEALTHY.value,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 0},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": self._leg_config_details(),
                }
            )
        )

    def _record_scheduler_transition(self, operational: JsonlLedger) -> None:
        """Append the cutover/rollback provenance event when the mode changed.

        D1 migration ceremony (Sol cutover list, M4 territory): one
        ``scheduler-cutover`` (waves -> pipelined) or ``scheduler-rollback``
        (pipelined -> waves) operational event per transition, carrying the
        canonical ledger tail counts/digests attested at the moment of the
        switch. Recorded AFTER lifecycle reconciliation proved quiescence (the
        call site follows ``_reconcile_lifecycle_before_admission``) and never
        for ``--only-status`` re-validation passes, which adjudicate nothing.
        The records themselves are scheduler-agnostic; this event is pure
        provenance so an auditor can bracket which legs ran under which mode.
        """

        if self.config.only_status is not None:
            return
        current_mode = self.config.scheduler
        previous_mode = self._effective_scheduler_mode(operational)
        if current_mode == previous_mode:
            return
        kind = (
            OperationalEventKind.SCHEDULER_CUTOVER
            if current_mode == "pipelined"
            else OperationalEventKind.SCHEDULER_ROLLBACK
        )
        occurrence = 1 + sum(
            1
            for event in operational.iter_records()
            if event.get("event_kind")
            in (
                OperationalEventKind.SCHEDULER_CUTOVER.value,
                OperationalEventKind.SCHEDULER_ROLLBACK.value,
            )
        )
        tails: dict[str, Any] = {}
        for name, path in (
            ("models", self.paths.ledgers.models),
            ("attempts", self.paths.ledgers.attempts),
            ("gates", self.paths.ledgers.gates),
        ):
            records = scan_jsonl(path) if path.exists() else []
            last: Optional[str] = None
            if records:
                # Attempt/gate rows carry their payload digest; model rows use
                # a record_revision identity instead, so recompute the same
                # canonical payload hash on read for a uniform attestation.
                recorded = records[-1].get("payload_sha256")
                last = str(recorded) if recorded else payload_hash(records[-1])
            tails[name] = {"count": len(records), "last_payload_sha256": last}
        details = {
            "from_scheduler": previous_mode,
            "to_scheduler": current_mode,
            "occurrence": occurrence,
            "ledger_tails": tails,
        }
        operational.append(
            mint_operational_event(
                {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "created_at": self.dependencies.clock(),
                    "event_kind": kind.value,
                    "status": OperationalEventStatus.SCHEDULER_MODE_CHANGED.value,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 0},
                    "current_environment": None,
                    "run_id": self.config.run_id,
                    "machine_id": self.config.machine_id,
                    "details": details,
                }
            )
        )

    def _maybe_pause_for_review(
        self,
        operational: JsonlLedger,
        current: Mapping[str, JsonObject],
        intake_snapshot: IntakeSnapshot,
        state: JsonObject,
    ) -> bool:
        """Create the one-shot blocking review report/event at its configured count."""

        threshold = self.config.review_checkpoint_at
        if not threshold or len(current) < threshold:
            return False
        intake_snapshot_id = intake_snapshot.snapshot_id
        intake_snapshot_sha256 = intake_snapshot.snapshot_sha256
        policy_key = stable_hash(
            {
                "policy": "crawler-review-checkpoint-v1",
                "intake_snapshot_id": intake_snapshot_id,
                "intake_snapshot_sha256": intake_snapshot_sha256,
                "review_threshold": threshold,
            }
        )
        policy_events = [
            event
            for event in operational.iter_records()
            if event.get("details", {}).get("policy_key") == policy_key
        ]
        kinds = [event.get("event_kind") for event in policy_events]
        if OperationalEventKind.REVIEW_SIGNOFF.value in kinds:
            return False
        if OperationalEventKind.CHECKPOINT_REVIEW.value in kinds:
            state["status"] = "paused:review-checkpoint"
            _write_driver_state(self.paths.driver_state, state)
            return True
        report_path = self.paths.reports_root / f"checkpoint-review-{threshold}.json"
        report = _review_report(current, threshold)
        _write_json_atomic(report_path, report)
        snapshot = _funnel_snapshot(current)
        review = record_checkpoint_review(
            operational,
            models_completed=len(current),
            funnel_snapshot=snapshot,
            report_path=str(report_path),
            context=self._context(0, None),
            created_at=self.dependencies.clock(),
            policy_identity={
                "policy_key": policy_key,
                "intake_snapshot_id": intake_snapshot_id,
                "intake_snapshot_sha256": intake_snapshot_sha256,
                "review_threshold": threshold,
            },
        )
        event_id = str(review.record["event_id"])
        self.dependencies.boundary_hook("after-notification-identity", event_id)
        self._deliver_notification(
            operational,
            event_id,
            _review_summary(len(current), snapshot, report_path),
            notification_event="review-checkpoint",
        )
        state.update(
            {
                "status": "paused:review-checkpoint",
                "last_terminal_count": len(current),
                "review_report": str(report_path),
            }
        )
        _write_driver_state(self.paths.driver_state, state)
        return True

    def _review_is_pending(self, operational: JsonlLedger, intake_snapshot: IntakeSnapshot) -> bool:
        """Return whether this verified snapshot's review policy lacks a signoff."""

        combined_by_id = {
            str(event.get("event_id")): event for event in operational.iter_records()
        }
        combined = tuple(combined_by_id.values())
        snapshot_policy_keys = {
            stable_hash(
                {
                    "policy": "crawler-review-checkpoint-v1",
                    "intake_snapshot_id": intake_snapshot.snapshot_id,
                    "intake_snapshot_sha256": intake_snapshot.snapshot_sha256,
                    "review_threshold": self.config.review_checkpoint_at,
                }
            )
        }
        review_keys = {
            str(event.get("details", {}).get("policy_key"))
            for event in combined
            if event.get("event_kind") == OperationalEventKind.CHECKPOINT_REVIEW.value
            and isinstance(event.get("details"), Mapping)
            and isinstance(event.get("details", {}).get("policy_key"), str)
            and event.get("details", {}).get("policy_key") in snapshot_policy_keys
        }
        signoff_keys = {
            str(event.get("details", {}).get("policy_key"))
            for event in combined
            if event.get("event_kind") == OperationalEventKind.REVIEW_SIGNOFF.value
            and isinstance(event.get("details"), Mapping)
            and isinstance(event.get("details", {}).get("policy_key"), str)
            and event.get("details", {}).get("policy_key") in snapshot_policy_keys
        }
        return bool(review_keys - signoff_keys)

    def _record_review_signoff(
        self,
        operational: JsonlLedger,
        reducer: CanonicalReducer,
        intake_snapshot: IntakeSnapshot,
        note: str,
    ) -> None:
        """Append the one-shot human review approval consumed by resume."""

        expected_policy_key = stable_hash(
            {
                "policy": "crawler-review-checkpoint-v1",
                "intake_snapshot_id": intake_snapshot.snapshot_id,
                "intake_snapshot_sha256": intake_snapshot.snapshot_sha256,
                "review_threshold": self.config.review_checkpoint_at,
            }
        )
        review_event: Optional[Mapping[str, Any]] = None
        for event in operational.iter_records():
            if (
                event.get("event_kind") == OperationalEventKind.CHECKPOINT_REVIEW.value
                and event.get("details", {}).get("policy_key") == expected_policy_key
            ):
                review_event = event
        if review_event is None:
            raise DriverIntegrationError("review signoff has no exact checkpoint authority")
        signoff = record_review_signoff(
            operational,
            approved_by_note=note,
            resume_after=int(review_event["models_completed"]),
            context=self._context(0, None),
            created_at=self.dependencies.clock(),
        )
        if signoff.record.get("details", {}).get("policy_key") != expected_policy_key:
            raise DriverIntegrationError("review signoff did not bind the verified intake snapshot")

    def _policy_snapshot(self) -> IntakeSnapshot:
        """Return the verified snapshot loaded for the active locked run.

        Returns
        -------
        IntakeSnapshot
            Exact snapshot object whose bytes were verified by ``_run_locked``.

        Raises
        ------
        DriverIntegrationError
            If a policy is evaluated outside an active locked run.
        """

        if self._intake_snapshot is None:
            raise DriverIntegrationError("snapshot policy evaluated before intake verification")
        return self._intake_snapshot

    def _ordered_intents(self, grouped: Mapping[str, Sequence[WorkItem]]) -> tuple[str, ...]:
        """Return intent names in registry phase order and deterministic name order."""

        phase_index = {phase: index for index, phase in enumerate(self.registry.phase_order)}
        return tuple(
            sorted(
                grouped,
                key=lambda name: (phase_index[self.registry.intents[name].phase], name),
            )
        )

    def _context(self, queued: int, environment: Optional[str]) -> OperationalContext:
        """Build common operational-event context for this invocation."""

        return OperationalContext(
            self.config.run_id,
            self.config.machine_id,
            {"models": queued},
            environment,
        )


del AdmissionEnvironmentMixin
del ReceiptDriverMixin
