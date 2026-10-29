"""Canonical reducer and sole model/attempt/gate ledger writer."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field, replace
from pathlib import Path
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, TypeAlias, Union

from menagerie.crawler.artifact_transactions import (
    ArtifactAuthorityRecord,
    ArtifactEventKind,
    ArtifactEventLedger,
    ReconstructionAnchorRecord,
    StagedArtifact,
    append_artifact_authorization,
    derive_artifact_claims,
    derive_publication_authorization_id,
)
from menagerie.crawler.authority import (
    AttemptAuthority,
    AuthorityContext,
    AuthorityDerivationError,
    DependencyCurrencyProjection,
    DependencyState,
    DependencyVector,
    DependencyValue,
    FamilyAuthority,
    PublicationAuthorization,
    TerminalProof,
    authenticate_accepted_attempts,
    derive_attempt_projection,
    dependency_vector_projection,
    derive_dependency_vector,
    derive_family_authority,
    derive_mode_summary,
    derive_per_mode_attempt_ids,
    derive_per_mode_run,
    derive_terminal_observation,
    derive_terminal_proof,
    family_authority_projection,
    layered_environment_generation,
    mode_summary_projection,
    load_current_attempt_proof,
    load_current_gate_proof,
    resolve_exact_gate_item_membership,
    validate_currency,
    _attempt_order,
    _require_derived_component_identity,
    _terminal_gate,
    _validate_terminal_gate_identities,
    _validate_terminal_references,
)

from menagerie.crawler.checker_calibration import (
    CHECKER_CALIBRATION_RUN_ENV,
    SHADOW_REHEARSAL_RUN_ENV,
    CheckerCalibrationWithheldError,
    calibration_run_active,
    shadow_rehearsal_run_active,
)
from menagerie.crawler.constants import (
    AUTHOR_DISPATCHER_COMPONENT,
    AUTHOR_RESULT_SCHEMA_COMPONENT,
    BLOCKED_DISPOSITION_REVIEW_REASON_CODES,
    CAPABILITY_DEFERRAL_STATUS_CODES,
    DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS,
    ATTEMPT_SCHEMA_VERSION_V3,
    CHECKER_PROMPT_NAME,
    EFFORT_EXHAUSTION_REASON_CODES,
    FAILURE_REASON_CODES,
    GATE_SCHEMA_VERSION_V3,
    LINK_EVIDENCED_DISCOVERY_STATUS_CODES,
    MODEL_SCHEMA_VERSION_V3,
    NO_RUNG_SELECTED,
    SKIPPED_STATUS_CODES,
    TERMINAL_STATUS_CODES,
    SourceRung,
)
from menagerie.crawler.env_lifecycle import (
    EnvironmentExactnessError,
    EnvironmentProbeError,
    materialized_environment_generation,
    parse_probe_receipt_bytes,
    parse_resolved_export,
)
from menagerie.crawler.envs import EnvironmentSpecError, load_environment_registry
from menagerie.crawler.family_templates import (
    FamilyTemplateError,
    family_representative_is_usable,
    family_variant_currency_error,
    validate_size_variant,
    validate_size_variant_derivation,
)
from menagerie.crawler.grant_identity import GrantIdentityError, expected_grant_id
from menagerie.crawler.grant_parks import (
    grant_park_path,
    grant_park_released,
    load_grant_park_rows,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    hash_bytes,
    is_sha256,
    is_shadow_rehearsal_record,
    payload_hash,
    stable_hash,
    stamp_shadow_rehearsal,
)
from menagerie.crawler.intake import legacy_requires_fidelity_audit
from menagerie.crawler.metadata import (
    AcceptedIdentities,
    MetadataValidationError,
    input_signature_matches_contract,
    recompute_accepted_identities,
    validate_authored_facts_for_write,
)
from menagerie.crawler.models import (
    AppendResult,
    JsonObject,
    LedgerPaths,
    manifest_source_rows,
)
from menagerie.crawler.licenses import LicenseDecision, RedistributionClass
from menagerie.crawler.mirrors import MirrorStore
from menagerie.crawler.recordio import (
    IncrementalJsonlScan,
    JsonlLedger,
    LedgerConflictError,
    driver_failure_attempt_identity,
    scan_jsonl,
)
from menagerie.crawler.state import _select_current as _select_current_by_parent
from menagerie.crawler.standard_inputs import ASSET_ROOT


class ReductionError(ValueError):
    """Raised when a proposed canonical fact violates reducer invariants."""


#: Typed scar marking a DELIBERATE grant-free supersession of a ``runs``
#: revision by a non-award terminal. The scar is permanent, greppable record
#: content (``scar_history``), so the deviation is visible forever; the
#: revision it superseded is named after the colon. Nothing in the driver
#: mints it automatically -- it exists so an adjudicated repair can be
#: recorded honestly instead of being laundered through a silent append.
AWARD_SUPERSESSION_SCAR_PREFIX = "award-superseded-without-grant:"

#: Typed scar marking an OPERATOR-restored award revision (AW-3). The suffix is
#: the ``award-restored-by-operator`` campaign-health event id that authorized
#: the restoration, so the decision and the record reference each other; the
#: derived census segregates restored awards from earned ones on this mark.
AWARD_RESTORATION_SCAR_PREFIX = "award-restored-by-operator:"

#: Operational-event details disposition of the restoration decision event.
AWARD_RESTORATION_EVENT_DISPOSITION = "award-restored-by-operator"


class AwardSupersessionError(ReductionError):
    """Raised when a non-award terminal would silently supersede a ``runs`` head.

    T2-08 (live, 2026-08-08): one ``EnvironmentExactnessError`` superseded 14
    ``runs`` revisions -- 11 released, published awards -- in 124 seconds,
    grant-free, scar-free, with ZERO operational events; the requeue grants
    were then voided. Award supersession was free while review-held FAILURE
    supersession required a grant: exactly backwards. AW-1 (AUDIT100, 2026-08-16)
    narrowed the release further after resume-4 destroyed three VERIFIED awards
    through the old escapes: a candidate-carried scar released the guard (the
    producer waiving its own tripwire) and ANY strictly-larger grant set
    released it (retry permission read as downgrade permission). A ``runs``
    revision is operator property; at live admission ONLY a durable requeue
    grant whose row names the exact blocked award revision
    (``blocked_award_revision``) releases the downgrade. Historical scarred
    supersessions remain replay-acceptable (the m5273 chain), and the scar is
    stamped at grant consumption as the RECEIPT of the decision, never as a
    substitute for one.
    """

    def __init__(self, *, stable_id: str, blocked_parent_revision: str) -> None:
        """Record the exact runs revision the candidate may not silently wipe.

        Parameters
        ----------
        stable_id:
            Model whose award supersession is refused.
        blocked_parent_revision:
            ``record_revision`` of the governing ``runs`` head.
        """

        super().__init__(
            "a runs revision cannot be superseded by a non-award terminal without "
            "a new explicit grant or the typed award-supersession scar"
        )
        self.stable_id = stable_id
        self.blocked_parent_revision = blocked_parent_revision


def introduced_award_naming_grant_id(
    grant_rows: Iterable[Mapping[str, Any]],
    introduced: Iterable[str],
    award_revision: str,
    stable_id: str,
) -> Optional[str]:
    """Return the one live introduced grant naming an exact member award.

    This is the single award-custody grant authority shared by reducer
    admission, driver-side scar stamping, and close-time reconciliation. A
    voided grant, foreign member, wrong revision, absent row, or grant already
    present on the award head releases nothing.

    Parameters
    ----------
    grant_rows:
        Durable grant and void rows from the canonical requeue-grant ledger.
    introduced:
        Grant IDs introduced by the candidate over the governing award head.
    award_revision:
        Exact ``runs`` revision being superseded.
    stable_id:
        Exact member whose award is in custody.

    Returns
    -------
    str | None
        Deterministically selected releasing grant ID, or ``None``.
    """

    rows = tuple(grant_rows)
    by_id = {
        str(row.get("grant_id")): row
        for row in rows
        if "void_of" not in row and isinstance(row.get("grant_id"), str)
    }
    voided = {str(row.get("void_of")) for row in rows if "void_of" in row}
    for grant_id in sorted(str(value) for value in introduced):
        if grant_id in voided:
            continue
        row = by_id.get(grant_id)
        if (
            row is not None
            and row.get("blocked_award_revision") == award_revision
            and str(row.get("stable_id")) == stable_id
        ):
            return grant_id
    return None


class GrantPendingSupersessionError(ReductionError):
    """Raised when an otherwise-valid supersession lacks only a review grant.

    The refusal itself is the long-standing review-hold tripwire: a review-held
    terminal is operator property, and no new revision may supersede it without
    a NEW explicit requeue grant. This subclass exists so the driver can tell
    the ONE parkable refusal apart from every other reducer refusal: it is
    raised only after the candidate passed every other append validation, so a
    caller catching it knows the append is blocked SOLELY by the missing grant.
    Any other defect still raises the plain :class:`ReductionError` and stays a
    loud failure.
    """

    def __init__(
        self,
        *,
        stable_id: str,
        blocked_parent_revision: str,
        blocked_stage: Optional[str],
    ) -> None:
        """Record the exact review-held revision that blocks the candidate.

        Parameters
        ----------
        stable_id:
            Model whose supersession is blocked.
        blocked_parent_revision:
            ``record_revision`` of the governing review-held terminal.
        blocked_stage:
            Recorded ``status.stage`` of that terminal; an operator grant must
            name this stage to be consumable.
        """

        super().__init__("human-review terminal supersession requires a new explicit grant")
        self.stable_id = stable_id
        self.blocked_parent_revision = blocked_parent_revision
        self.blocked_stage = blocked_stage


class OrphanAdoptionRefusal(ReductionError):
    """Raised when :meth:`CanonicalReducer.append_orphan_adoption` refuses one member.

    Every arm of the closed adoption rule (P5-SOL-05) that fails at the write
    door raises this type with the exact typed reason, so the driver's
    adoption pass can disclose it as the SAME per-member census refusal shape
    its own redundant first fence produces -- one head's fact, never a leg
    crash.
    """

    def __init__(self, *, stable_id: str, reason: str) -> None:
        """Record the refused member and the closed-rule arm that refused it.

        Parameters
        ----------
        stable_id:
            Model whose adoption re-append is refused.
        reason:
            Exact typed refusal reason (closed-rule arm output).
        """

        super().__init__(reason)
        self.stable_id = stable_id
        self.reason = reason


@dataclass(frozen=True)
class ColdForwardPolicy:
    """Reducer-owned cold-forward admission policy.

    Parameters
    ----------
    confirmation_policy:
        Canonical policy label persisted in a run record.
    required_cold_forwards:
        Exact number of independent forwards required for every meaningful mode.
    """

    confirmation_policy: str
    required_cold_forwards: int


def cold_forward_policy(stable_id: str, rung: object) -> ColdForwardPolicy:
    """Derive the authoritative deterministic cold-forward policy for one run.

    Parameters
    ----------
    stable_id:
        Durable model identity used for reproducible canary membership.
    rung:
        Canonical source rung.

    Returns
    -------
    ColdForwardPolicy
        Two cold forwards for R3/R4 and for the deterministic two-percent R1/R2
        canary; one cold forward for the remaining R1/R2 population.
    """

    if str(rung) in {"R3_PORT", "R4_REIMPLEMENT"}:
        return ColdForwardPolicy("two-cold-r3-r4", 2)
    canary_digest = hashlib.sha256(f"{stable_id}mechanical-canary-v1".encode("utf-8")).digest()
    if int.from_bytes(canary_digest, "big") % 100 < 2:
        return ColdForwardPolicy("mechanical-canary", 2)
    return ColdForwardPolicy("single-mechanical", 1)


#: Closed execution-stage set the fidelity no-consumer waiver may admit
#: (KERNEL_DECISION condition 3). The pre-existing ``runner`` allowlist arm keeps
#: its own driver-owned / ``environment is None`` constraint; ``resource`` and
#: ``policy`` are deliberately excluded. ANY widening of this set requires its own
#: deliberated commit passing the four-part waiver doctrine (repo CLAUDE.md).
FIDELITY_WAIVER_EXECUTION_STAGES: frozenset[str] = frozenset(
    {"environment", "import", "constructor", "input", "forward"}
)

#: W8's CLOSED INVENTORY: every failure stage the pre-execution no-consumer arm
#: admits, each mapped to the proof it is pre-execution. The first five are
#: pre-execution UNCONDITIONALLY (the pipeline cannot have executed anything at
#: them). ``accuracy-gate`` and ``runner`` are CONDITIONAL members: under
#: execute-first triage a forward can complete before either adjudicates, so
#: their membership is sound only because the arm independently sweeps every
#: referenced AND same-work ledger attempt for execution evidence (a worker
#: receipt or a started forward disqualifies regardless of stage). Widening
#: this inventory is a waiver-instance change and needs its own four-part
#: reviewed commit.
PRE_FIDELITY_FAILURE_STAGES: Mapping[str, str] = {
    "intake": "intake precedes source resolution; no execution lane exists yet",
    "source": "source resolution precedes authoring and every execution lane",
    "fetch": "artifact fetch precedes authoring and every execution lane",
    "author": "the author session produces the recipe an execution would run",
    "evidence": "evidence assembly adjudicates authored text, never a run",
    "accuracy-gate": (
        "CONDITIONAL: metadata adjudication itself runs nothing, but "
        "execute-first triage may order it after a completed forward -- the "
        "execution-evidence sweep is the guard (opus E7 / AG-10)"
    ),
    "runner": (
        "CONDITIONAL: the driver-owned runner arm dies dispatching the worker; "
        "the execution-evidence sweep plus the historical environment-is-None "
        "allowlist constraint are the guard"
    ),
}

#: Complete runner-lifecycle stage set that can disqualify the narrow fidelity
#: waiver. ``resource``, ``policy``, and ``runner`` remain ineligible as waiver
#: claims, but an attempt reaching one of them proves that an earlier eligible
#: stage is no longer the decisive machine death.
_FIDELITY_WAIVER_ORDERING_STAGES: frozenset[str] = FIDELITY_WAIVER_EXECUTION_STAGES | {
    "resource",
    "policy",
    "runner",
}

_FIDELITY_WAIVER_POLICY_FLAGS: tuple[str, ...] = (
    "network_attempted",
    "checkpoint_or_weight_read_attempted",
    "write_outside_scratch_attempted",
    "credentials_present",
    "torchlens_import_attempted",
    "cache_read_attempted",
)
_FIDELITY_WAIVER_POLICY_SEQUENCES: tuple[str, ...] = (
    "socket_targets",
    "checkpoint_paths",
    "write_paths",
)

#: Worker-receipt milestones in execution order. Together with the per-stage
#: allowances below they define "receipt evidence beyond the claimed stage":
#: a death at stage S may honestly carry only the milestones its own stage and
#: earlier stages produce.
_RECEIPT_MILESTONE_FLAGS: tuple[str, ...] = (
    "present",
    "constructor_started",
    "constructor_completed",
    "input_completed",
    "forward_started",
    "forward_completed",
)

#: Maximal receipt milestones an attempt may carry for a death claimed at each
#: waiver stage. ``present`` is import-stage evidence (the worker ran); a death
#: at ``environment`` therefore admits no worker receipt at all. Within-stage
#: completion flags are admitted (a ``constructor`` death whose constructor
#: returned an invalid model object honestly carries ``constructor_completed``).
#:
#: ``forward_completed`` is the ONE within-stage completion milestone that is
#: NOT admitted, and the asymmetry is the no-consumer contract itself: a
#: constructor or input death produced no execution outcome, so the fidelity
#: gate has nothing to adjudicate, but a COMPLETED forward produced exactly the
#: execution outcome the gate exists to compare against a reference. The
#: post-completion forward failures are live reason codes -- the driver copies
#: the worker's ``forward_completed`` onto failed attempts verbatim, and
#: ``invalid-output-signature`` / ``confirmation-mismatch`` / ``incomplete-receipt``
#: only occur AFTER a forward returns -- so admitting it waived the gate on
#: models that had in fact run (C1 rehearsal, 2026-08-12). Such a terminal must
#: obtain a real fidelity gate; it is never no-consumer.
_STAGE_RECEIPT_ALLOWANCES: dict[str, frozenset[str]] = {
    "environment": frozenset(),
    "import": frozenset({"present"}),
    "constructor": frozenset({"present", "constructor_started", "constructor_completed"}),
    "input": frozenset(
        {"present", "constructor_started", "constructor_completed", "input_completed"}
    ),
    "forward": frozenset(set(_RECEIPT_MILESTONE_FLAGS) - {"forward_completed"}),
}


def _execution_evidence_beyond_stage(attempt: Mapping[str, Any], stage: str) -> bool:
    """Return whether one attempt carries execution evidence beyond a claimed stage.

    Parameters
    ----------
    attempt:
        Current authenticated attempt record.
    stage:
        Claimed death stage from :data:`FIDELITY_WAIVER_EXECUTION_STAGES`.

    Returns
    -------
    bool
        True when the attempt's worker receipt or identity evidence proves the
        work progressed past the claimed stage.
    """

    allowed = _STAGE_RECEIPT_ALLOWANCES[stage]
    receipt = attempt.get("worker_receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    if (stage == "environment" and receipt.get("present")) or (
        stage != "environment" and receipt.get("present") is not True
    ):
        return True
    required_milestones: Mapping[str, tuple[str, ...]] = {
        "environment": (),
        "import": ("present",),
        "constructor": ("present", "constructor_started"),
        "input": ("present", "constructor_started", "constructor_completed"),
        "forward": (
            "present",
            "constructor_started",
            "constructor_completed",
            "input_completed",
            "forward_started",
        ),
    }
    if any(not receipt.get(flag) for flag in required_milestones[stage]):
        return True
    if any(receipt.get(flag) for flag in _RECEIPT_MILESTONE_FLAGS if flag not in allowed):
        return True
    policy = attempt.get("policy_observation")
    policy = policy if isinstance(policy, Mapping) else {}
    if any(bool(policy.get(field)) for field in _FIDELITY_WAIVER_POLICY_FLAGS):
        return True
    if any(bool(policy.get(field)) for field in _FIDELITY_WAIVER_POLICY_SEQUENCES):
        return True
    parent = attempt.get("parent_attestation")
    parent = parent if isinstance(parent, Mapping) else {}
    if (
        bool(parent.get("timed_out"))
        or bool(parent.get("rss_exceeded"))
        or parent.get("signal") is not None
    ):
        return True
    if attempt.get("unattested_partial") is not None:
        return True
    if (
        attempt.get("defer_evidence") is not None
        or attempt.get("capability_observation") is not None
    ):
        return True
    if stage != "forward" and any(
        receipt.get(field) is not None for field in ("output_signature", "output_value_sha256")
    ):
        return True
    if stage != "forward" and any(
        receipt.get(field) is not None
        for field in (
            "input_signature",
            "input_kind",
            "input_asset",
            "parameter_count_total",
            "parameter_count_trainable",
            "native_framework",
            "delegated_method",
        )
    ):
        return True
    if stage == "environment":
        identities = attempt.get("identities")
        identities = identities if isinstance(identities, Mapping) else {}
        if (
            attempt.get("environment") is not None
            or identities.get("environment") is not None
            or identities.get("execution") is not None
        ):
            return True
    error = attempt.get("error")
    if isinstance(error, Mapping) and error.get("native_crash") is True:
        return True
    return False


def _derive_blocked_capability_deferral_proof(
    *,
    status_code: str,
    stable_id: str,
    work_id: str,
    attempts: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    source_manifest: Sequence[Mapping[str, Any]],
    evidence_excerpts: Sequence[Mapping[str, Any]],
    source_manifest_identity: Optional[str],
    evidence_identity: Optional[str],
    license_identity: Optional[str],
    meaningful_modes: Iterable[str],
    proof_rule_identity: str,
) -> TerminalProof:
    """Derive the additive capability-deferral proof for an accepted BLOCKED verdict.

    The frozen platform-deferral rule remains unchanged and continues to prove only
    CUDA/x86 handoffs. This separate reducer rule reuses its exact terminal-gate
    identity and reference validators for an accepted ``BLOCKED`` result, but
    deliberately carries no platform claim or execution attempt.

    It serves both capability deferrals, because their PROOF is identical: an accepted
    BLOCKED terminal disposition over a frozen manifest. They differ only in which
    capability the terminal names -- a stronger authoring tier, or the access to read
    material we located and were refused -- and that difference is carried by
    ``status_code``, not by a second copy of this rule.

    Parameters
    ----------
    status_code:
        Exact capability-deferral terminal being proved.
    stable_id, work_id:
        Exact promoted model and work generation.
    attempts, gates:
        Current canonical authority records.
    source_manifest, evidence_excerpts:
        Frozen source and evidence rows retained by the blocked result.
    source_manifest_identity, evidence_identity, license_identity:
        Exact terminal-disposition identity axes.
    meaningful_modes:
        Declared modes retained by the common proof projection.
    proof_rule_identity:
        Frozen terminal-policy identity.

    Returns
    -------
    TerminalProof
        Reducer-derived proof of the accepted higher-tier deferral.

    Raises
    ------
    AuthorityDerivationError
        If the gate or any referenced identity differs.
    """

    gate, item, disposition = _terminal_gate(
        gates,
        stable_id=stable_id,
        work_id=work_id,
        predicate="blocked-prerequisite",
    )
    if disposition.get("kind") != "BLOCKED" or disposition.get("verdict") != "accepted":
        raise AuthorityDerivationError(
            f"{status_code} requires an accepted BLOCKED terminal disposition"
        )
    _validate_terminal_gate_identities(
        disposition,
        source_manifest_identity=source_manifest_identity,
        evidence_identity=evidence_identity,
        license_identity=license_identity,
    )
    source_ids, evidence_ids = _validate_terminal_references(
        disposition,
        predicate="blocked-prerequisite",
        source_manifest=source_manifest,
        evidence_excerpts=evidence_excerpts,
    )
    per_mode = derive_per_mode_attempt_ids(
        attempts,
        stable_id=stable_id,
        work_id=work_id,
        meaningful_modes=meaningful_modes,
    )
    terminal_observation_sha256 = stable_hash(
        derive_terminal_observation(attempts, stable_id=stable_id, work_id=work_id)
    )
    gate_id = str(gate["gate_id"])
    proof_payload = {
        "proof_rule_identity": proof_rule_identity,
        "stable_id": stable_id,
        "work_id": work_id,
        "status_code": status_code,
        "decisive_attempt_ids": [],
        "gate_id": gate_id,
        "source_ids": list(source_ids),
        "evidence_ids": list(evidence_ids),
        "failure_stage": DependencyState.NOT_APPLICABLE,
        "reason_code": DependencyState.NOT_APPLICABLE,
        "root_cause_fingerprint": DependencyState.NOT_APPLICABLE,
        "platform_claim": DependencyState.NOT_APPLICABLE,
        "per_mode_attempt_ids": [list(value) for value in per_mode],
        "terminal_observation_sha256": terminal_observation_sha256,
        "gate_proof_identity": stable_hash({"gate_id": gate_id, "item": item}),
        "resolved_reference_identity": stable_hash(
            {
                "sources": [
                    source
                    for source in source_manifest
                    if str(source.get("source_id")) in source_ids
                ],
                "evidence": [
                    excerpt
                    for excerpt in evidence_excerpts
                    if str(excerpt.get("evidence_id")) in evidence_ids
                ],
            }
        ),
    }
    return TerminalProof(
        proof_id=stable_hash(proof_payload),
        proof_rule_identity=proof_rule_identity,
        stable_id=stable_id,
        work_id=work_id,
        status_code=status_code,
        decisive_attempt_ids=(),
        gate_id=gate_id,
        source_ids=source_ids,
        evidence_ids=evidence_ids,
        failure_stage=DependencyState.NOT_APPLICABLE,
        reason_code=DependencyState.NOT_APPLICABLE,
        root_cause_fingerprint=DependencyState.NOT_APPLICABLE,
        platform_claim=DependencyState.NOT_APPLICABLE,
        per_mode_attempt_ids=per_mode,
        terminal_observation_sha256=terminal_observation_sha256,
    )


class _ReplayLedger:
    """Minimal in-memory ledger used to replay reducer admission without writes."""

    def __init__(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Retain immutable records in an append-compatible replay ledger.

        Replay never mutates canonical input rows, so sharing their mappings
        avoids a full-ledger deepcopy for every projected model.
        """

        self.records: list[Mapping[str, Any]] = list(records)

    def iter_records(self) -> Iterable[Mapping[str, Any]]:
        """Iterate retained records without copies (read-only consumers)."""

        return iter(self.records)

    def append(self, record: Mapping[str, Any]) -> AppendResult:
        """Record one already hash-validated replay candidate in memory."""

        identity = next(
            (
                (field, record.get(field))
                for field in ("attempt_id", "gate_id", "record_revision")
                if record.get(field) is not None
            ),
            None,
        )
        if identity is not None:
            field, value = identity
            existing = next(
                (candidate for candidate in self.records if candidate.get(field) == value),
                None,
            )
            if existing is not None:
                return AppendResult(deepcopy(dict(existing)), appended=False)
        copied = deepcopy(dict(record))
        self.records.append(copied)
        return AppendResult(copied, appended=True)


class _ReplayArtifactLedger:
    """Read-only artifact-event facade used during semantic projection replay."""

    def __init__(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Retain immutable artifact events without acquiring another writer lock."""

        self.events = tuple(records)

    def iter_events(self) -> Iterator[Mapping[str, Any]]:
        """Iterate retained events, mirroring the writer's no-copy accessor."""

        return iter(self.events)


_FACT_ROOTS = (
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
    "modes",
    "fidelity",
)

_MODALITY_STANDARD_ASSETS = {
    "audio": "audio.csv",
    "computer-vision": "image.ppm",
    "image": "image.ppm",
    "language": "text.txt",
    "nlp": "text.txt",
    "recsys": "tabular.csv",
    "speech": "audio.csv",
    "tabular": "tabular.csv",
    "text": "text.txt",
    "video": "image.ppm",
    "vision": "image.ppm",
}
_WORKER_COMPLETION_PREFIX = "MENAGERIE_WORKER_COMPLETION_V1 "
_DIAGNOSTIC_REDACTION_MARKER = "externally-controlled-text-v1"
_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SAFE_EXCERPT_DISPOSITIONS = frozenset(
    {"public-compatible", "public-domain", "short-excerpt-committed"}
)
_EXTERNALLY_CONTROLLED_RECORD_FIELDS = frozenset(
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


def _is_safe_diagnostic_redaction(value: Any) -> bool:
    """Return whether a C-07 field contains no raw externally controlled text.

    Parameters
    ----------
    value:
        Candidate externally controlled field value.

    Returns
    -------
    bool
        ``True`` for empty values or the checkpoint-approved sidecar reference shape.
    """

    if value is None or value == "":
        return True
    if not isinstance(value, Mapping):
        return False
    if value.get("license_disposition") in _SAFE_EXCERPT_DISPOSITIONS:
        return bool(
            _HASH_PATTERN.fullmatch(str(value.get("text_sha256", "")))
            and isinstance(value.get("locator"), str)
        )
    required = {"redaction", "content_sha256", "local_path", "diagnostic_key"}
    allowed = required | {"stream_sha256"}
    local_path = str(value.get("local_path", ""))
    pure_local = PurePosixPath(local_path)
    stream_sha256 = value.get("stream_sha256")
    return bool(
        set(value) <= allowed
        and required <= set(value)
        and value.get("redaction") == _DIAGNOSTIC_REDACTION_MARKER
        and _HASH_PATTERN.fullmatch(str(value.get("content_sha256", "")))
        and pure_local.parts[:2] == (".crawl-local", "diagnostics")
        and ".." not in pure_local.parts
        and pure_local.suffix == ".json"
        and re.fullmatch(r"\$[A-Za-z0-9_.\[\]-]+", str(value.get("diagnostic_key", "")))
        and (stream_sha256 is None or _HASH_PATTERN.fullmatch(str(stream_sha256)) is not None)
    )


def _validate_c07_diagnostic_fields(record: Mapping[str, Any], *, model: bool) -> None:
    """Reject raw C-07 diagnostic values before a canonical append.

    Parameters
    ----------
    record:
        Proposed attempt or model record.
    model:
        Whether model-only failed-detail and human-review fields are in scope.

    Raises
    ------
    ReductionError
        If any protected field contains raw text or a forged redaction reference.
    """

    findings: list[str] = []

    def visit(value: Any, location: str = "$") -> None:
        """Collect unsafe protected values recursively."""

        if isinstance(value, Mapping):
            for key, nested in value.items():
                nested_location = f"{location}.{key}"
                failed_status_detail = bool(
                    model
                    and key == "detail"
                    and location.endswith(".status")
                    and value.get("kind") == "failed"
                )
                human_review_reason = bool(
                    model and key == "reason" and location.endswith(".status.human_review")
                )
                if (
                    key in _EXTERNALLY_CONTROLLED_RECORD_FIELDS
                    or failed_status_detail
                    or human_review_reason
                ) and not _is_safe_diagnostic_redaction(nested):
                    findings.append(nested_location)
                visit(nested, nested_location)
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                visit(nested, f"{location}[{index}]")

    visit(record)
    if findings:
        raise ReductionError(
            f"canonical record contains unredacted externally controlled text: fields={findings}"
        )


def expected_standard_asset(modality: object) -> Optional[dict[str, str]]:
    """Return the exact standard asset selected by input materialization.

    Parameters
    ----------
    modality:
        Accepted modality string or sequence.

    Returns
    -------
    dict[str, str] | None
        Asset path, byte digest, and content-addressed identifier, or ``None``
        when input materialization must use deterministic random fallback.
    """

    values = (modality,) if isinstance(modality, str) else modality
    if not isinstance(values, (list, tuple)):
        return None
    normalized = tuple(str(value).strip().lower() for value in values)
    selected_name: Optional[str] = None
    for candidates in (
        {"vision", "image", "computer-vision"},
        {"language", "text", "nlp"},
        {"audio", "speech"},
        {"video"},
        {"tabular", "recsys"},
    ):
        selected = next((value for value in normalized if value in candidates), None)
        if selected is not None:
            selected_name = _MODALITY_STANDARD_ASSETS[selected]
            break
    if selected_name is None:
        return None
    path = ASSET_ROOT / selected_name
    try:
        digest = hash_bytes(path.read_bytes())
    except OSError as exc:
        raise ReductionError(f"selected standard input asset is unavailable: {path}") from exc
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "asset_id": f"standard:{selected_name}:{digest}",
    }


def output_signature_error(signature: object) -> Optional[str]:
    """Return why an output signature is not a complete pytree contract.

    Parameters
    ----------
    signature:
        Candidate ``output_signature`` mapping.

    Returns
    -------
    str | None
        Stable validation error, or ``None`` for a complete signature.
    """

    if not isinstance(signature, Mapping):
        return "signature is not an object"
    tree = signature.get("tree")
    leaves = signature.get("leaves")
    if tree is None or not isinstance(leaves, list) or not leaves:
        return "signature tree and leaves must be non-empty"
    referenced: list[int] = []

    def visit(node: object) -> bool:
        """Validate one tree node and collect its leaf indices."""

        if not isinstance(node, Mapping) or not node:
            return False
        if set(node) == {"leaf"} and isinstance(node.get("leaf"), int):
            index = node.get("leaf")
            if isinstance(index, bool):
                return False
            assert isinstance(index, int)
            referenced.append(index)
            return 0 <= index < len(leaves)
        if set(node) in ({"tuple"}, {"list"}) and isinstance(next(iter(node.values())), list):
            children = next(iter(node.values()))
            assert isinstance(children, list)
            return all(visit(child) for child in children)
        return all(isinstance(key, str) and visit(child) for key, child in node.items())

    if not visit(tree) or sorted(referenced) != list(range(len(leaves))):
        return "signature tree does not reference each leaf exactly once"
    for leaf in leaves:
        if not isinstance(leaf, Mapping):
            return "signature leaf is not an object"
        path = leaf.get("path")
        kind = leaf.get("kind")
        python_type = leaf.get("python_type")
        if (
            not isinstance(path, str)
            or not path
            or kind not in {"tensor", "python"}
            or not isinstance(python_type, str)
            or not python_type
        ):
            return "signature leaf lacks path, kind, or python type"
        if kind == "tensor":
            shape = leaf.get("shape")
            dtype = leaf.get("dtype")
            if (
                not isinstance(shape, list)
                or any(
                    not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
                    for dimension in shape
                )
                or not isinstance(dtype, str)
                or not dtype
            ):
                return "tensor signature leaf lacks a concrete shape or dtype"
    return None


def _select_current(records: Iterable[Mapping[str, Any]]) -> dict[str, JsonObject]:
    """Validate public supersession lineage and select current revisions.

    Parameters
    ----------
    records:
        Canonical model revisions in append order.

    Returns
    -------
    dict[str, dict[str, Any]]
        Current revisions selected after exact public-lineage validation.
    """

    materialized = tuple(records)
    for record in materialized:
        parent = record.get("parent_revision")
        supersedes = record.get("status", {}).get("supersedes_revision")
        if parent is None and supersedes is not None:
            raise ReductionError("a first model revision cannot supersede another revision")
        if parent is not None and supersedes != parent:
            raise ReductionError(
                "persisted status.supersedes_revision does not match parent_revision"
            )
    return _select_current_by_parent(materialized)


def _records_root(ledgers: LedgerPaths) -> Path:
    """Return the canonical records root for sibling operational evidence.

    Parameters
    ----------
    ledgers:
        Canonical model/attempt/gate ledger paths.

    Returns
    -------
    pathlib.Path
        Records root containing the durable operational directory.
    """

    parent = ledgers.models.resolve().parent
    return parent.parent if parent.name == "models" else parent


#: IC-5 declared mutable inputs of the dependency-currency projection. The
#: projection token carries EXACTLY one value per name, in this order; adding
#: a new mutable projection input without extending both is the token-
#: incompleteness class (T1-15, then the family-map/registry recurrence).
_PROJECTION_TOKEN_MUTABLE_INPUTS: tuple[str, ...] = (
    "authority_bytes_digest",
    "models_append_generation",
    "attempts_append_generation",
    "gates_append_generation",
    "artifact_events_append_generation",
    "historical_generation_families_epoch",
)

_ProjectionToken: TypeAlias = tuple[str, int, int, int, int, int]


def _revision_work_ids(
    revision: Mapping[str, Any],
    attempts: Iterable[Mapping[str, Any]],
    gates: Iterable[Mapping[str, Any]],
    artifact_events: Iterable[Mapping[str, Any]] = (),
) -> frozenset[str]:
    """Collect durable work identities referenced by one model revision.

    Parameters
    ----------
    revision:
        Candidate or persisted model revision.
    attempts, gates:
        Canonical execution and checker evidence.
    artifact_events:
        Canonical artifact-authority events. The revision's own
        ``artifact_authority.transaction_id`` names append-only staged or
        finalized custody whose ledger rows carry the exact work generation
        the artifact was produced under -- for a checker-rejected terminal
        recommendation the staged transaction is the ONLY durable evidence
        binding the new generation (no attempt, no accepted gate, no retained
        proposal), so omitting it made a granted re-adjudication's
        supersession structurally unable to bind its granted work identity
        (rung-11 leg 6: m4256/m4257).

    Returns
    -------
    frozenset[str]
        Work identities bound by the revision's referenced evidence.
    """

    stable_id = revision.get("stable_id")
    attempt_ids = set(revision.get("status", {}).get("attempt_ids", [])) | set(
        revision.get("execution", {}).get("accepted_attempt_ids", [])
    )
    work_ids = {
        str(attempt.get("work_id"))
        for attempt in attempts
        if attempt.get("attempt_id") in attempt_ids and attempt.get("work_id") is not None
    }
    gate_ids = {
        revision.get("accuracy_gate", {}).get("gate_id"),
        revision.get("fidelity", {}).get("gate_id"),
    } - {None}
    for gate in gates:
        if gate.get("gate_id") not in gate_ids:
            continue
        for item in gate.get("items", []):
            if item.get("stable_id") == stable_id:
                for field in ("work_id", "campaign_root_work_id"):
                    if item.get(field) is not None:
                        work_ids.add(str(item[field]))
    untrusted = revision.get("untrusted_attempt")
    if isinstance(untrusted, Mapping):
        proposal = untrusted.get("proposal")
        if isinstance(proposal, Mapping) and proposal.get("work_id") is not None:
            work_ids.add(str(proposal["work_id"]))
    authority = revision.get("artifact_authority")
    transaction_id = authority.get("transaction_id") if isinstance(authority, Mapping) else None
    if isinstance(transaction_id, str) and transaction_id.startswith("sha256:"):
        for event in artifact_events:
            if (
                event.get("transaction_id") == transaction_id
                and event.get("stable_id") == stable_id
                and isinstance(event.get("work_id"), str)
            ):
                work_ids.add(str(event["work_id"]))
    return frozenset(work_ids)


#: (resolved path, validate) -> suffix-incremental reader.
_OPERATIONAL_EVIDENCE_SCAN_CACHE: dict[tuple[str, bool], IncrementalJsonlScan] = {}


def _retain_reducer_operational_evidence(record: Mapping[str, Any]) -> bool:
    """Return whether one operational row can steer reducer authority.

    Parameters
    ----------
    record:
        Validated operational event.

    Returns
    -------
    bool
        Whether reducer lineage, restoration, revocation, or environment
        attestation can consult the row.
    """

    details = record.get("details")
    if record.get("event_kind") == "requeue-grant-consumed":
        return True
    if not isinstance(details, Mapping):
        return False
    return (
        details.get("disposition")
        in {AWARD_RESTORATION_EVENT_DISPOSITION, "award-revocation"}
        or (
            isinstance(details.get("intent"), str)
            and (
                isinstance(details.get("env_generation"), str)
                or isinstance(details.get("environment"), Mapping)
            )
        )
    )


def _reset_operational_evidence_scan_cache() -> None:
    """Drop every cached operational-evidence scan (test hook)."""

    _OPERATIONAL_EVIDENCE_SCAN_CACHE.clear()


def _scan_operational_evidence_cached(
    path: Path, *, validate: bool = True
) -> tuple[JsonObject, ...]:
    """Scan one operational evidence shard, cached on its stat signature.

    The requeue-lineage validator historically re-read and re-validated the
    whole grants and events shards from disk on EVERY grant-bearing model
    append (T1-12). The scan result is exactly as fresh as the file: append,
    truncation, creation, and removal -- including by OTHER processes -- all
    change the ``(size, mtime_ns)`` signature and force a live re-scan (the
    same contract as PC1's valve-row index). Rows are shared read-only
    snapshots; consumers must not mutate them.

    Parameters
    ----------
    path:
        Operational ledger path. A missing file is an empty shard.
    validate:
        Whether recognized payloads schema-validate, exactly like
        :func:`scan_jsonl`.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Scanned rows in ledger order.
    """

    key = (str(path.resolve()), validate)
    reader = _OPERATIONAL_EVIDENCE_SCAN_CACHE.get(key)
    if reader is None:
        reader = IncrementalJsonlScan(
            path,
            validate=validate,
            retain=(
                _retain_reducer_operational_evidence
                if path.name == "events.jsonl"
                else None
            ),
        )
        _OPERATIONAL_EVIDENCE_SCAN_CACHE[key] = reader
    return reader.refresh()


def _validate_persisted_requeue_lineage(
    records: Iterable[Mapping[str, Any]],
    ledgers: LedgerPaths,
    attempts: Iterable[Mapping[str, Any]],
    gates: Iterable[Mapping[str, Any]],
    artifact_events: Iterable[Mapping[str, Any]] = (),
) -> None:
    """Validate every explicit grant introduction against canonical durable proof.

    Parameters
    ----------
    records:
        Persisted revisions plus an optional append candidate.
    ledgers:
        Canonical ledger paths used to locate operational evidence.
    attempts, gates:
        Canonical evidence carrying new-work identities.
    artifact_events:
        Canonical artifact-authority events; the revision's own referenced
        transaction is durable evidence of its work generation exactly like
        attempts and gates (see :func:`_revision_work_ids`).

    Raises
    ------
    ReductionError
        If a grant, parent binding, generation, or new-work identity is inconsistent.
    """

    materialized = tuple(records)
    if not any(record.get("budget", {}).get("explicit_grants", []) for record in materialized):
        return
    # Materialize every evidence iterable ONCE: _revision_work_ids re-reads
    # attempts and gates for EACH grant-bearing revision, so a one-shot
    # iterator would be exhausted after the first revision and every later
    # grant would falsely refuse as unbound.
    attempts = tuple(attempts)
    gates = tuple(gates)
    artifact_events = tuple(artifact_events)
    operational_root = _records_root(ledgers) / "operational"
    grant_rows = _scan_operational_evidence_cached(
        operational_root / "requeue-grants.jsonl", validate=False
    )
    event_rows = _scan_operational_evidence_cached(operational_root / "events.jsonl")
    grants = {str(grant.get("grant_id")): grant for grant in grant_rows}
    consumptions = {
        str(event.get("details", {}).get("grant_id")): event
        for event in event_rows
        if event.get("event_kind") == "requeue-grant-consumed"
    }
    prior_grants: dict[str, set[str]] = {}
    for revision in materialized:
        stable_id = str(revision.get("stable_id"))
        inherited = prior_grants.get(stable_id, set())
        current_grants = set(revision.get("budget", {}).get("explicit_grants", []))
        if not inherited <= current_grants:
            raise ReductionError("explicit requeue grants cannot be removed from a lineage")
        introduced = current_grants - inherited
        if len(introduced) > 1:
            raise ReductionError("one model revision cannot consume multiple requeue grants")
        for grant_id in introduced:
            grant = grants.get(grant_id)
            event = consumptions.get(grant_id)
            if grant is None or event is None:
                missing = " and ".join(
                    name
                    for name, value in (
                        ("its grant row", grant),
                        ("its consumption event", event),
                    )
                    if value is None
                )
                raise ReductionError(
                    "explicit requeue grant lacks canonical durable proof: "
                    f"grant {grant_id} for {stable_id} is missing {missing} "
                    "under records/operational/"
                )
            try:
                canonical_grant_id = expected_grant_id(grant)
            except GrantIdentityError as exc:
                raise ReductionError(str(exc)) from exc
            if canonical_grant_id != grant_id:
                raise ReductionError("explicit requeue grant identity is invalid")
            details = event.get("details", {})
            if (
                grant.get("stable_id") != stable_id
                or details.get("stable_id") != stable_id
                or any(
                    details.get(field) != grant.get(field)
                    for field in (
                        "stage",
                        "reason",
                        "attempts",
                        "target_intent",
                        "blocked_award_revision",
                    )
                )
            ):
                raise ReductionError("explicit requeue grant facts conflict with consumption")
            parent_revision = revision.get("parent_revision")
            if parent_revision is None or details.get("source_record_revision") != parent_revision:
                raise ReductionError("explicit requeue grant is bound to the wrong parent revision")
            generation = details.get("new_work_generation")
            if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
                raise ReductionError("explicit requeue grant generation is invalid")
            expected_work_id = stable_hash(
                {
                    "stable_id": stable_id,
                    "grant_id": grant_id,
                    "parent_revision": parent_revision,
                    "generation": generation,
                }
            )
            if details.get("new_work_id") != expected_work_id:
                raise ReductionError("explicit requeue grant new-work identity is invalid")
            if expected_work_id not in _revision_work_ids(
                revision, attempts, gates, artifact_events
            ):
                raise ReductionError(
                    "superseding revision does not bind the granted new-work identity"
                )
        prior_grants[stable_id] = current_grants


def _model_facts(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract the complete accepted fact tree from a canonical model revision."""

    return {field: model.get(field) for field in _FACT_ROOTS}


def _authored_facts_for_vet(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recover authored facts before a fidelity gate projected its verdict fields."""

    facts = deepcopy(dict(_model_facts(model)))
    fidelity = facts.get("fidelity")
    if isinstance(fidelity, dict) and fidelity.get("gate_id") is not None:
        fidelity.update(
            {
                "current": False,
                "deviations": [],
                "gate_id": None,
                "permanent_scar": False,
                "verdict": None,
            }
        )
    return facts


def _checker_prompt_hash() -> str:
    """Hash the exact current checker prompt bytes used for freshness."""

    path = Path(__file__).with_name("prompts") / f"{CHECKER_PROMPT_NAME}.txt"
    try:
        return hash_bytes(path.read_bytes())
    except OSError as exc:
        raise ReductionError(f"checker prompt bytes are unavailable: {exc}") from exc


def _gate_item_fingerprint(item: Mapping[str, Any]) -> str:
    """Return the canonical checker root-cause fingerprint for one item."""

    return stable_hash(
        {
            "verdict": item.get("verdict"),
            "integrity": item.get("integrity"),
            "field_checks": item.get("field_checks"),
            "rung_check": item.get("rung_check"),
            "fidelity": item.get("fidelity"),
            "unsupported_claims": item.get("unsupported_claims"),
            "required_repairs": item.get("required_repairs"),
        }
    )


def _record_attempt_ids(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every attempt reference carried by a terminal or run record."""

    referenced: list[str] = []
    blocks = (
        record.get("status", {}).get("attempt_ids", []),
        record.get("execution", {}).get("accepted_attempt_ids", []),
        record.get("observed", {}).get("measurement_attempt_ids", []),
        [
            value.get("attempt_id")
            for value in record.get("modes", {}).get("per_mode_run", {}).values()
            if isinstance(value, Mapping)
        ],
    )
    for values in blocks:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            if value is not None and str(value) not in referenced:
                referenced.append(str(value))
    return tuple(referenced)


@dataclass
class _GateValidationPipeline:
    """Run reducer gate checks in their frozen diagnostic order.

    Parameters
    ----------
    reducer:
        Locked canonical reducer supplying current gate and model authority.
    model:
        Proposed model revision.
    """

    reducer: "CanonicalReducer"
    model: Mapping[str, Any]
    stable_id: str = dataclass_field(init=False, default="")
    accuracy: Mapping[str, Any] = dataclass_field(init=False, default_factory=dict)
    rung: object = dataclass_field(init=False, default=None)
    status_kind: object = dataclass_field(init=False, default=None)
    metadata_state: object = dataclass_field(init=False, default=None)
    family_variant: bool = dataclass_field(init=False, default=False)
    representative_id: str = dataclass_field(init=False, default="")
    representative: Optional[Mapping[str, Any]] = dataclass_field(init=False, default=None)
    identities: Optional[AcceptedIdentities] = dataclass_field(init=False, default=None)
    gate_stable_id: str = dataclass_field(init=False, default="")

    def run(self) -> None:
        """Execute rung, authored-metadata, and fidelity checks in order."""

        self._initialize()
        rung_gate_current = self._derive_rung_gate_current()
        self._validate_runs_rung(rung_gate_current)
        self._validate_accepted_metadata()
        self._validate_fidelity()

    def _initialize(self) -> None:
        """Derive shared gate-validation inputs in original evaluation order."""

        self.stable_id = str(self.model["stable_id"])
        self.accuracy = self.model.get("accuracy_gate", {})
        self.rung = self.model.get("source_resolution", {}).get("rung")
        self.status_kind = self.model.get("status", {}).get("kind")
        self.metadata_state = self.model.get("authored_metadata_state")
        self.family_variant = self.stable_id in self.reducer.intake_variant_bindings
        self.representative_id = str(
            self.model.get("identity", {}).get("family_representative_id", "")
        )
        self.representative = (
            self.reducer._validation_current_records().get(self.representative_id)
            if self.family_variant
            else None
        )
        self.gate_stable_id = self.representative_id if self.family_variant else self.stable_id

    def _semantic_replay(self) -> bool:
        """Return whether this pipeline validates persisted history."""

        return bool(getattr(self.reducer, "_semantic_replay", False))

    def _prompt_basis(self, gate: Mapping[str, Any]) -> str:
        """Return the checker-prompt basis for identity checks.

        A LIVE append binds the currently shipped prompt bytes (the
        anti-templating door). SEMANTIC REPLAY re-validates a persisted
        record at the prompt basis its SEALED gate recorded -- that value
        was proven against the then-live bytes at the gate's own append and
        is committed by the gate's ledger row hash, never a producer claim.
        Re-imposing today's bytes on sealed history refused every held
        runs/accepted head whenever the checker prompt legitimately re-pinned
        (the rung-11/rung-12 history-vs-live family at the model layer;
        m7305 paid six re-authors for prompt churn). Currency of the standing
        record against the live contract stays ``validate_currency``'s
        decision, exactly like the append_gate replay carve-out.
        """

        if self._semantic_replay():
            return str(gate.get("checker", {}).get("prompt_sha256"))
        return _checker_prompt_hash()

    def _derive_rung_gate_current(self) -> bool:
        """Return whether the referenced anti-slop/rung gate is current."""

        rung_gate_current = False
        rung_found = self.reducer._gate_item(self.accuracy.get("gate_id"), self.gate_stable_id)
        if rung_found is not None:
            rung_gate, rung_item = rung_found
            rung_check = rung_item.get("rung_check")
            verified_hashes = rung_item.get("verified_hashes", {})
            # Replay binds the gate's recorded prompt to the MODEL's own
            # recorded basis (identity-tight, record-to-record -- the same
            # shape the accepted-metadata prompt-binding check uses); a live
            # append keeps demanding the currently shipped bytes.
            expected_prompt = (
                self.accuracy.get("prompt_sha256")
                if self._semantic_replay()
                else _checker_prompt_hash()
            )
            rung_gate_current = bool(
                rung_gate.get("gate_kind") == "metadata_batch"
                and rung_item.get("vet_identity") == self.accuracy.get("vet_identity")
                and isinstance(rung_check, Mapping)
                and rung_check.get("selected_rung") == self.rung
                and rung_check.get("verdict") == "accurate"
                and verified_hashes.get("code")
                == self.model.get("implementation", {}).get("code_sha256")
                and rung_gate.get("checker", {}).get("prompt_sha256") == expected_prompt
            )
        return rung_gate_current

    def _validate_runs_rung(self, rung_gate_current: bool) -> None:
        """Require current rung authority before a runs disposition."""

        if self.status_kind == "runs" and not rung_gate_current:
            if self.metadata_state == "pending":
                self.reducer._validate_pending_run_rung(self.model)
            else:
                raise ReductionError(
                    "runs requires a current identity-tight anti-slop/rung check gate"
                )

    def _validate_accepted_metadata(self) -> None:
        """Validate accepted authored metadata and recompute its identities."""

        if self.metadata_state != "accepted":
            return
        found = self.reducer._gate_item(self.accuracy.get("gate_id"), self.gate_stable_id)
        if found is None:
            raise ReductionError("accepted authored metadata is missing its gate")
        gate, item = found
        if gate["gate_kind"] != "metadata_batch":
            raise ReductionError("authored metadata must reference a metadata_batch gate")
        rung_check = item.get("rung_check")
        checks = (
            ("vet-identity", item["vet_identity"] == self.accuracy.get("vet_identity")),
            ("item-verdict", item["verdict"] == "accurate"),
            ("integrity-verdict", item["integrity"]["verdict"] == "accurate"),
            (
                "rung-check",
                isinstance(rung_check, Mapping)
                and rung_check.get("selected_rung") == self.rung
                and rung_check.get("verdict") == "accurate",
            ),
            ("model-verdict", self.accuracy.get("verdict") == "accurate"),
            ("not-current", bool(self.accuracy.get("current"))),
            (
                "prompt-binding",
                gate.get("checker", {}).get("prompt_sha256") == self.accuracy.get("prompt_sha256"),
            ),
        )
        failed_checks = [name for name, passed in checks if not passed]
        if failed_checks:
            raise ReductionError(
                "authored metadata gate is missing, stale, inaccurate, or has a blocked "
                "rung check: " + ", ".join(failed_checks)
            )
        facts = _authored_facts_for_vet(self.model)
        if self.family_variant:
            self._validate_variant_metadata(gate, facts)
        else:
            self._validate_ordinary_metadata(gate, item, facts)

    def _validate_variant_metadata(
        self,
        gate: Mapping[str, Any],
        facts: Mapping[str, Any],
    ) -> None:
        """Validate representative inheritance and variant identity freshness."""

        representative_accuracy = (
            self.representative.get("accuracy_gate", {})
            if isinstance(self.representative, Mapping)
            else {}
        )
        if (
            not isinstance(self.representative, Mapping)
            or self.representative.get("authored_metadata_state") != "accepted"
            or representative_accuracy.get("current") is not True
            or self.accuracy != representative_accuracy
        ):
            raise ReductionError(
                "family variant does not inherit its current representative accuracy gate"
            )
        try:
            self.identities = recompute_accepted_identities(
                facts,
                checker_prompt_hash=self._prompt_basis(gate),
                checker_model=str(gate.get("checker", {}).get("model")),
                checker_version=str(gate.get("checker", {}).get("version")),
                schema_version=MODEL_SCHEMA_VERSION_V3,
            )
        except MetadataValidationError as exc:
            raise ReductionError(str(exc)) from exc
        if (
            self.model.get("evidence", {}).get("evidence_identity") != self.identities.evidence
            or self.model.get("implementation", {}).get("recipe_revision") != self.identities.recipe
        ):
            raise ReductionError("family variant source/evidence/recipe identities are stale")

    def _validate_ordinary_metadata(
        self,
        gate: Mapping[str, Any],
        item: Mapping[str, Any],
        facts: Mapping[str, Any],
    ) -> None:
        """Validate ordinary authored facts and exact identity freshness."""

        try:
            validate_authored_facts_for_write(facts, item)
            self.identities = recompute_accepted_identities(
                facts,
                checker_prompt_hash=self._prompt_basis(gate),
                checker_model=str(gate.get("checker", {}).get("model")),
                checker_version=str(gate.get("checker", {}).get("version")),
                schema_version=MODEL_SCHEMA_VERSION_V3,
            )
        except MetadataValidationError as exc:
            raise ReductionError(str(exc)) from exc
        if (
            self.model.get("evidence", {}).get("evidence_identity") != self.identities.evidence
            or self.model.get("implementation", {}).get("recipe_revision") != self.identities.recipe
            or item.get("vet_identity") != self.identities.vet
            or self.accuracy.get("vet_identity") != self.identities.vet
        ):
            stale = [
                name
                for name, actual, expected in (
                    (
                        "evidence",
                        self.model.get("evidence", {}).get("evidence_identity"),
                        self.identities.evidence,
                    ),
                    (
                        "recipe",
                        self.model.get("implementation", {}).get("recipe_revision"),
                        self.identities.recipe,
                    ),
                    ("gate-vet", item.get("vet_identity"), self.identities.vet),
                    ("model-vet", self.accuracy.get("vet_identity"), self.identities.vet),
                )
                if actual != expected
            ]
            raise ReductionError(
                "accepted source/evidence/recipe/vet identities are stale: " + ", ".join(stale)
            )

    def _validate_fidelity(self) -> None:
        """Validate required fidelity authority after authored metadata checks."""

        fidelity = self.model.get("fidelity", {})
        legacy_flags = self.model.get("intake", {}).get("preserved_legacy_flags", [])
        required = (
            bool(fidelity.get("required"))
            or self.rung in {"R3_PORT", "R4_REIMPLEMENT"}
            or legacy_requires_fidelity_audit(legacy_flags)
        )
        if not required:
            return
        found = self.reducer._gate_item(fidelity.get("gate_id"), self.stable_id)
        if found is None:
            status = self.model.get("status", {})
            if self.reducer._fidelity_gate_has_no_consumer(self.model):
                return
            if (
                status.get("code") == "failed:fidelity"
                and status.get("reason_code") == "identity-mismatch"
                and not fidelity.get("current")
            ):
                return
            # AG-10 (opus E7 / W8): an adjudication-stage terminal minted AFTER
            # a completed forward HAS a fidelity consumer -- the executed code
            # must be audited -- so the refusal names the ROUTE (obtain a real
            # fidelity gate) instead of the generic fall-through that read as
            # an unmintable dead end and parked the member (the leg-death
            # class). The refusal itself never weakens: the record stays
            # refused until the gate exists.
            if self.reducer._work_execution_evidence(self.model):
                raise ReductionError(
                    "fidelity-consumer-exists: required fidelity is missing its gate for "
                    f"{self.stable_id} (status={status.get('code')!r}) while same-work "
                    "execution evidence exists (a worker receipt or started forward); "
                    "this terminal routes to a REAL fidelity gate -- run the fidelity "
                    "checker over the executed work before terminalizing"
                )
            # Actionability: name the model, the gate it looked for, and the two
            # legitimate ways out. An operator seeing this in a 20k crawl log
            # otherwise cannot tell WHICH model stalled or what to do next.
            raise ReductionError(
                "required fidelity is missing its gate for "
                f"{self.stable_id}: no fidelity gate item was found for "
                f"gate_id={fidelity.get('gate_id')!r} and no no-consumer arm applies "
                f"(status={status.get('code')!r}); run the fidelity checker for this "
                "model, or -- if it died before producing any execution outcome -- "
                "record the decisive execution attempt that proves the death"
            )
        gate, item = found
        if self.identities is None:
            try:
                self.identities = recompute_accepted_identities(
                    _model_facts(self.model),
                    checker_prompt_hash=self._prompt_basis(gate),
                    checker_model=str(gate.get("checker", {}).get("model")),
                    checker_version=str(gate.get("checker", {}).get("version")),
                    schema_version=MODEL_SCHEMA_VERSION_V3,
                )
            except MetadataValidationError as exc:
                raise ReductionError(str(exc)) from exc
        if gate["gate_kind"] != "fidelity":
            raise ReductionError("fidelity must reference a per-model fidelity gate")
        rung_check = item.get("rung_check")
        rejected_terminal = self.model.get("status", {}).get(
            "code"
        ) == "failed:fidelity" or self.reducer._is_fidelity_repair_failure(self.model)
        allowed_verdicts = (
            {"major-drift", "slop", "cannot-verify"}
            if rejected_terminal
            else {"match", "minor-drift"}
        )
        rung_accepted = isinstance(rung_check, Mapping) and (
            rejected_terminal
            or (
                rung_check.get("verdict") == "accurate"
                and rung_check.get("selected_rung") == self.rung
            )
        )
        checks = (
            (
                "fidelity-identity",
                item.get("fidelity_identity") == fidelity.get("fidelity_identity")
                and item.get("fidelity_identity") == self.identities.fidelity,
            ),
            ("verdict-mismatch", item["fidelity"]["verdict"] == fidelity.get("verdict")),
            (
                f"verdict-not-in-{sorted(allowed_verdicts)}",
                fidelity.get("verdict") in allowed_verdicts,
            ),
            ("rung-check", rung_accepted),
            ("not-current", bool(fidelity.get("current"))),
        )
        failed_checks = [name for name, passed in checks if not passed]
        if failed_checks:
            raise ReductionError(
                "required fidelity gate is stale, unacceptable, or has a blocked rung "
                "check: " + ", ".join(failed_checks)
            )


@dataclass
class _ModelAuthorityPipeline:
    """Derive one model's authority through ordered, independently sourced steps."""

    reducer: CanonicalReducer
    model: Mapping[str, Any]
    artifact_inputs: JsonObject = dataclass_field(init=False)
    stable_id: str = dataclass_field(init=False)
    work_id: str = dataclass_field(init=False)
    meaningful_modes: tuple[str, ...] = dataclass_field(init=False)
    event: Any = dataclass_field(init=False)
    document: Any = dataclass_field(init=False)
    source_manifest: Sequence[Mapping[str, Any]] = dataclass_field(init=False)
    evidence_excerpts: Sequence[Mapping[str, Any]] = dataclass_field(init=False)
    source_resolution: Optional[Mapping[str, Any]] = dataclass_field(init=False)
    source_manifest_identity: Optional[str] = dataclass_field(init=False)
    evidence_identity: Any = dataclass_field(init=False)
    license_identity: Any = dataclass_field(init=False)
    current_attempts: tuple[Mapping[str, Any], ...] = dataclass_field(init=False)
    terminal_proof: TerminalProof = dataclass_field(init=False)
    family_authority: FamilyAuthority = dataclass_field(init=False)
    checker_gate: DependencyValue = dataclass_field(init=False)
    environment_id: Optional[str] = dataclass_field(init=False)

    def run(self) -> tuple[TerminalProof, FamilyAuthority, JsonObject, JsonObject]:
        """Run the model-authority derivation in canonical validation order.

        Returns
        -------
        tuple[TerminalProof, FamilyAuthority, dict[str, Any], dict[str, Any]]
            Exact proof objects plus the derived vector and artifact inputs.
        """

        self._initialize()
        self._derive_persisted_evidence()
        self._apply_evidence_fallbacks()
        self._derive_terminal_and_family_proofs()
        self._validate_mode_projections()
        self._derive_checker_gate()
        self._derive_environment()
        vector_projection = self._derive_dependency_projection()
        return (
            self.terminal_proof,
            self.family_authority,
            vector_projection,
            self.artifact_inputs,
        )

    def _initialize(self) -> None:
        """Initialize model identities and empty evidence projections."""

        self.artifact_inputs = self.reducer._artifact_authority_inputs(self.model)
        self.stable_id = str(self.model.get("stable_id"))
        self.work_id = self.reducer._model_work_id(self.model, self.artifact_inputs)
        self.meaningful_modes = tuple(
            str(value) for value in self.model.get("modes", {}).get("meaningful_modes", ())
        )
        self.event = self.artifact_inputs.get("event")
        self.document = self.artifact_inputs.get("document")
        self.source_manifest = ()
        self.evidence_excerpts = ()
        self.source_resolution = None
        self.source_manifest_identity = None
        self.evidence_identity = None
        self.license_identity = None

    def _derive_persisted_evidence(self) -> None:
        """Derive evidence only from the independently persisted artifact inputs."""

        if isinstance(self.event, Mapping):
            self.source_manifest_identity = str(self.event.get("source_manifest_identity"))
        if not isinstance(self.document, Mapping):
            return
        manifest = self.document.get("source_manifest")
        # Coverage, not identity: this tuple is the SOURCE FACT index that
        # `_validate_terminal_references` resolves a terminal arm's cited
        # `source_ids` against, and a SKIP/DEFER arm may legitimately cite a row
        # from the ONE granted supplementary broker pack. Reading `sources` alone
        # refused those as "terminal gate names a missing source fact" -- the
        # same defect as the author-side "references unknown source", one lane
        # further downstream. `source_manifest_identity` above is still the
        # frozen digest and does not move.
        if isinstance(manifest, Mapping) and isinstance(manifest.get("sources"), list):
            self.source_manifest = tuple(manifest_source_rows(manifest))
        author_result = self.document.get("author_result")
        payload = author_result.get("payload") if isinstance(author_result, Mapping) else None
        if isinstance(payload, Mapping):
            self.evidence_identity = payload.get("evidence_identity")
            self.license_identity = payload.get("license_identity")
        proposal = self.document.get("proposal")
        facts = proposal.get("proposed_facts") if isinstance(proposal, Mapping) else None
        if not isinstance(facts, Mapping):
            return
        source_resolution_value = facts.get("source_resolution")
        evidence = facts.get("evidence")
        if isinstance(source_resolution_value, Mapping):
            self.source_resolution = source_resolution_value
        excerpts = evidence.get("excerpts") if isinstance(evidence, Mapping) else None
        if isinstance(excerpts, list):
            self.evidence_excerpts = tuple(
                value for value in excerpts if isinstance(value, Mapping)
            )
        if self.evidence_identity is None and isinstance(evidence, Mapping):
            self.evidence_identity = evidence.get("evidence_identity")
        if self.license_identity is None:
            licenses = facts.get("licenses")
            if isinstance(licenses, Mapping):
                self.license_identity = stable_hash(licenses)

    def _apply_evidence_fallbacks(self) -> None:
        """Apply legacy model and gate evidence fallbacks in canonical order."""

        if self.source_resolution is None:
            candidate_resolution = self.model.get("source_resolution")
            if isinstance(candidate_resolution, Mapping):
                self.source_resolution = candidate_resolution
        if not self.source_manifest and isinstance(self.source_resolution, Mapping):
            candidate_sources = self.source_resolution.get("sources")
            if isinstance(candidate_sources, list):
                self.source_manifest = tuple(
                    value for value in candidate_sources if isinstance(value, Mapping)
                )
        if not self.evidence_excerpts:
            evidence = self.model.get("evidence")
            excerpts = evidence.get("excerpts") if isinstance(evidence, Mapping) else None
            if isinstance(excerpts, list):
                self.evidence_excerpts = tuple(
                    value for value in excerpts if isinstance(value, Mapping)
                )
            if self.evidence_identity is None and isinstance(evidence, Mapping):
                self.evidence_identity = evidence.get("evidence_identity")
        if self.license_identity is not None:
            return
        for gate in self.reducer._gates.iter_records():
            for item in gate.get("items", []):
                if item.get("stable_id") != self.stable_id or item.get("work_id") != self.work_id:
                    continue
                terminal = item.get("terminal_disposition")
                if isinstance(terminal, Mapping):
                    self.license_identity = terminal.get("license_identity")
                    self.evidence_identity = terminal.get(
                        "evidence_identity", self.evidence_identity
                    )

    def _derive_terminal_and_family_proofs(self) -> None:
        """Derive terminal and family proof objects from active authority."""

        try:
            all_current_attempts = self.reducer._current_attempt_records()
            current_gates = self.reducer._current_gate_records()
            status_code = str(self.model.get("status", {}).get("code"))
            if status_code == "runs":
                accepted_ids = {
                    str(value)
                    for value in self.model.get("execution", {}).get("accepted_attempt_ids", ())
                }
                self.current_attempts = tuple(
                    attempt
                    for attempt in all_current_attempts
                    if str(attempt.get("attempt_id")) in accepted_ids
                )
            else:
                self.current_attempts = all_current_attempts
            resolved_evidence_identity = (
                str(self.evidence_identity) if self.evidence_identity is not None else None
            )
            resolved_license_identity = (
                str(self.license_identity) if self.license_identity is not None else None
            )
            if status_code in CAPABILITY_DEFERRAL_STATUS_CODES:
                self.terminal_proof = _derive_blocked_capability_deferral_proof(
                    status_code=status_code,
                    stable_id=self.stable_id,
                    work_id=self.work_id,
                    attempts=self.current_attempts,
                    gates=current_gates,
                    source_manifest=self.source_manifest,
                    evidence_excerpts=self.evidence_excerpts,
                    source_manifest_identity=self.source_manifest_identity,
                    evidence_identity=resolved_evidence_identity,
                    license_identity=resolved_license_identity,
                    meaningful_modes=self.meaningful_modes,
                    proof_rule_identity=self.reducer.context.terminal_policy_identity,
                )
            else:
                try:
                    self.terminal_proof = derive_terminal_proof(
                        self.stable_id,
                        self.work_id,
                        status_code,
                        # The machine-derived capability claim, when the revision carries
                        # one. It is INPUT to the host-capability rule, which recomputes
                        # the refusal; every other terminal rule ignores it.
                        host_capability_claim=(
                            self.model.get("host_capability")
                            if isinstance(self.model.get("host_capability"), Mapping)
                            else None
                        ),
                        attempts=self.current_attempts,
                        gates=current_gates,
                        source_manifest=self.source_manifest,
                        evidence_excerpts=self.evidence_excerpts,
                        source_resolution=self.source_resolution,
                        source_manifest_identity=self.source_manifest_identity,
                        evidence_identity=resolved_evidence_identity,
                        license_identity=resolved_license_identity,
                        meaningful_modes=self.meaningful_modes,
                        proof_rule_identity=self.reducer.context.terminal_policy_identity,
                    )
                except AuthorityDerivationError as platform_refusal:
                    # AL-6, fixed precedence (the checker-session-exhaustion
                    # pattern): the frozen platform-handoff rule outranks, and
                    # ONLY `deferred:needs-cuda` has a second proof -- a CUDA-
                    # class checked BLOCKED verdict carries no handoff, so its
                    # head settles through the same accepted-BLOCKED capability
                    # rule as the opus-tier/source-access pair. The fallback
                    # demands its own accepted gate, so it can never upgrade an
                    # outcome; when it also refuses, the PLATFORM refusal is
                    # re-raised as the real diagnosis.
                    if status_code != "deferred:needs-cuda":
                        raise
                    try:
                        self.terminal_proof = _derive_blocked_capability_deferral_proof(
                            status_code=status_code,
                            stable_id=self.stable_id,
                            work_id=self.work_id,
                            attempts=self.current_attempts,
                            gates=current_gates,
                            source_manifest=self.source_manifest,
                            evidence_excerpts=self.evidence_excerpts,
                            source_manifest_identity=self.source_manifest_identity,
                            evidence_identity=resolved_evidence_identity,
                            license_identity=resolved_license_identity,
                            meaningful_modes=self.meaningful_modes,
                            proof_rule_identity=(
                                self.reducer.context.terminal_policy_identity
                            ),
                        )
                    except AuthorityDerivationError:
                        raise platform_refusal from None
            representative: Optional[Mapping[str, Any]] = None
            binding = self.reducer.context.family_bindings.get(self.stable_id)
            if isinstance(binding, Mapping):
                representative_id = binding.get(
                    "representative_stable_id", binding.get("family_representative_id")
                )
                if isinstance(representative_id, str) and representative_id != self.stable_id:
                    representative = self.reducer._validation_current_records().get(
                        representative_id
                    )
            self.family_authority = derive_family_authority(
                self.reducer.context,
                self.stable_id,
                representative_record=representative,
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc

    def _validate_mode_projections(self) -> None:
        """Validate per-mode, summary, and terminal-observation projections."""

        relevant = tuple(
            attempt
            for attempt in self.current_attempts
            if attempt.get("stable_id") == self.stable_id and attempt.get("work_id") == self.work_id
        )
        expected_per_mode = derive_per_mode_run(
            relevant,
            stable_id=self.stable_id,
            work_id=self.work_id,
            meaningful_modes=self.meaningful_modes,
        )
        if self.model.get("modes", {}).get("per_mode_run") != expected_per_mode:
            raise ReductionError("model per_mode_run contradicts reducer-derived attempt authority")
        expected_attempt_ids = {value["attempt_id"] for value in expected_per_mode.values()}
        by_mode = {
            str(attempt.get("mode")): attempt
            for attempt in relevant
            if attempt.get("attempt_id") in expected_attempt_ids
            and attempt.get("result") == "succeeded"
        }
        try:
            mode_projection = mode_summary_projection(
                derive_mode_summary(by_mode.get("train"), by_mode.get("eval"))
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        for field, expected in mode_projection.items():
            if self.model.get("modes", {}).get(field) != expected:
                raise ReductionError(f"modes.{field} contradicts reducer-derived mode authority")
        if self.model.get("status", {}).get("kind") != "runs":
            expected_observed = derive_terminal_observation(
                self.current_attempts,
                stable_id=self.stable_id,
                work_id=self.work_id,
            )
            if self.model.get("observed") != expected_observed:
                raise ReductionError("terminal observed facts contradict reducer-derived authority")

    def _derive_checker_gate(self) -> None:
        """Derive the dependency identity for the checker gate."""

        self.checker_gate = self.terminal_proof.gate_id
        if self.checker_gate != DependencyState.NOT_APPLICABLE:
            return
        accuracy_gate = self.model.get("accuracy_gate", {}).get("gate_id")
        self.checker_gate = (
            accuracy_gate
            if isinstance(accuracy_gate, str) and accuracy_gate
            else DependencyState.PENDING_UNTRUSTED
            if self.model.get("authored_metadata_state") == "pending"
            else DependencyState.NOT_APPLICABLE
        )

    def _derive_environment(self) -> None:
        """Validate and resolve the environment named by the terminal proof."""

        self.environment_id = None
        self.proof_environment_generation: Optional[str] = None
        env_generation = self.model.get("execution", {}).get("env_generation")
        proof_attempt_ids = set(self.terminal_proof.decisive_attempt_ids) | {
            attempt_id for _mode, attempt_id in self.terminal_proof.per_mode_attempt_ids
        }
        proof_environment_generations = {
            attempt.get("identities", {}).get("environment")
            for attempt in self.current_attempts
            if attempt.get("attempt_id") in proof_attempt_ids
            and isinstance(attempt.get("identities", {}).get("environment"), str)
        }
        if len(proof_environment_generations) > 1:
            raise ReductionError("terminal proof spans multiple environment generations")
        proof_environment_generation = next(iter(proof_environment_generations), None)
        self.proof_environment_generation = proof_environment_generation
        if proof_environment_generation is not None:
            if env_generation != proof_environment_generation:
                raise ReductionError("model environment generation contradicts its attempt proof")
            self.environment_id = next(
                (
                    name
                    for name, generation in self.reducer.context.environment_generations.items()
                    if generation == proof_environment_generation
                ),
                None,
            )
            if self.environment_id is None:
                # The context map holds ONE generation per family (newest
                # wins), so a proof from an OLDER generation of a family that
                # rebuilt mid-leg can never resolve there. Ledger-attested
                # bindings registered by the driver cover exactly that case;
                # a generation attested nowhere still refuses.
                self.environment_id = self.reducer.historical_generation_families.get(
                    proof_environment_generation
                )
            if self.environment_id is None:
                # Both maps are point-in-time snapshots (construction-time
                # derivation, admission-time registration). A family that
                # rebuilt AFTER those snapshots attests its new generation in
                # attempt rows and lifecycle events this reducer itself
                # appended, so re-derive once from the live ledgers before
                # refusing -- resolution must not depend on registration
                # timing (AW-2; the leg-28 m1163/m1221/m2062 circuit-break).
                self.environment_id = self.reducer.refresh_attested_generation_families().get(
                    proof_environment_generation
                )
            if self.environment_id is None:
                raise ReductionError("model environment generation is absent from active context")
        elif self.model.get("status", {}).get("kind") == "runs":
            raise ReductionError("runs proof has no environment generation")

    def _derive_dependency_projection(self) -> JsonObject:
        """Derive the final dependency-vector projection."""

        accepted_ids = (
            self.model.get("execution", {}).get("accepted_attempt_ids", [])
            if self.model.get("status", {}).get("kind") == "runs"
            else self.model.get("status", {}).get("attempt_ids", [])
        )
        proposal_identity: DependencyValue = DependencyState.NOT_APPLICABLE
        author_result_identity: DependencyValue = DependencyState.NOT_APPLICABLE
        if isinstance(self.event, Mapping):
            event_proposal = self.event.get("proposal_id")
            event_result = self.event.get("author_result_id")
            if isinstance(event_proposal, str) and event_proposal:
                proposal_identity = event_proposal
            if isinstance(event_result, str) and event_result:
                author_result_identity = event_result
        candidate_recipe = self.model.get("implementation", {}).get("recipe_revision")
        recipe_revision: DependencyValue = (
            candidate_recipe
            if isinstance(candidate_recipe, str)
            else DependencyState.NOT_APPLICABLE
        )
        proof_environment_generation = self.proof_environment_generation
        if bool(getattr(self.reducer, "_semantic_replay", False)):
            # Era-independent replay for the environment axis: recorded
            # vectors are reducer-derived projections whose stamping rule has
            # legitimately evolved (era slots before 5acaf6ac, the proof's
            # generation after). Replay reproduces the RECORDED value exactly
            # when ledger evidence attests it for the SAME resolved family --
            # never a producer claim, and a generation attested nowhere still
            # derives fresh and refuses on mismatch (leg-29 close: 7 proof-
            # valued heads under slot-wins, 16 slot-valued heads under
            # proof-wins; both eras must replay).
            recorded_vector = self.model.get("dependency_vector")
            if isinstance(recorded_vector, Mapping):
                recorded_generation = recorded_vector.get("environment_generation")
                if (
                    isinstance(recorded_generation, str)
                    and recorded_generation
                    and recorded_generation != DependencyState.NOT_APPLICABLE.value
                    and self.reducer.historical_generation_families.get(recorded_generation)
                    == self.environment_id
                ):
                    proof_environment_generation = recorded_generation
        try:
            vector = derive_dependency_vector(
                self.reducer.context,
                stable_id=self.stable_id,
                terminal_proof=self.terminal_proof,
                source_manifest_identity=(
                    self.source_manifest_identity or DependencyState.NOT_APPLICABLE
                ),
                proposal_identity=proposal_identity,
                author_result_identity=author_result_identity,
                checker_gate_identity=self.checker_gate,
                recipe_revision=recipe_revision,
                environment_id=self.environment_id,
                proof_environment_generation=proof_environment_generation,
                accepted_attempt_ids=accepted_ids,
                artifact_transaction_id=self.artifact_inputs["transaction_id"],
                artifact_claim_ids=self.artifact_inputs["claim_ids"],
                family_authority=self.family_authority,
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        projection = dependency_vector_projection(vector)
        # T4-T1 intake-axis carve-out, replay half: a persisted revision recorded
        # under a PRIOR REGISTERED snapshot of the SAME campaign reproduces its
        # recorded snapshot axes during semantic replay instead of claiming the
        # active pair. The recorded pair is honored ONLY when both halves match
        # one operator-durable registered lineage entry (never a producer claim),
        # and only during replay -- a live append always derives the active
        # snapshot, so new work can never backdate its provenance.
        if bool(getattr(self.reducer, "_semantic_replay", False)):
            recorded = self.model.get("dependency_vector")
            recorded = recorded if isinstance(recorded, Mapping) else {}
            recorded_id = recorded.get("intake_snapshot_id")
            recorded_sha256 = recorded.get("intake_snapshot_sha256")
            if (
                isinstance(recorded_id, str)
                and isinstance(recorded_sha256, str)
                and recorded_id != self.reducer.context.active_intake_snapshot_id
                and self.reducer.context.prior_intake_snapshots.get(recorded_id) == recorded_sha256
            ):
                projection["intake_snapshot_id"] = recorded_id
                projection["intake_snapshot_sha256"] = recorded_sha256
        return projection


@dataclass
class _ExecutionValidationPipeline:
    """Validate run execution authority through ordered receipt checks."""

    reducer: CanonicalReducer
    model: Mapping[str, Any]
    execution: Mapping[str, Any] = dataclass_field(init=False)
    modes: Mapping[str, Any] = dataclass_field(init=False)
    meaningful_order: tuple[str, ...] = dataclass_field(init=False)
    meaningful: set[str] = dataclass_field(init=False)
    per_mode: Mapping[str, Any] = dataclass_field(init=False)
    attempts_by_id: Mapping[str, Mapping[str, Any]] = dataclass_field(init=False)
    accepted: set[Any] = dataclass_field(init=False)
    stable_id: Any = dataclass_field(init=False)
    cold_policy: ColdForwardPolicy = dataclass_field(init=False)
    signatures: dict[str, list[Any]] = dataclass_field(init=False)
    counts: dict[str, int] = dataclass_field(init=False)
    cold_indexes: dict[str, set[int]] = dataclass_field(init=False)
    parent_witnesses: set[str] = dataclass_field(init=False)
    invocations: set[bytes] = dataclass_field(init=False)
    accepted_work_ids: set[str] = dataclass_field(init=False)
    implementation: Mapping[str, Any] = dataclass_field(init=False)
    evidence: Mapping[str, Any] = dataclass_field(init=False)
    pending_proposal: Any = dataclass_field(init=False)
    expected_manifest_digest: Any = dataclass_field(init=False)
    expected_asset_digest: Any = dataclass_field(init=False)
    expected_asset_id: Any = dataclass_field(init=False)
    accepted_identities: AcceptedIdentities = dataclass_field(init=False)
    input_contract: Mapping[str, Any] = dataclass_field(init=False)
    accepted_in_order: list[Any] = dataclass_field(init=False)
    authenticated_by_id: dict[str, AttemptAuthority] = dataclass_field(init=False)

    def run(self) -> None:
        """Run execution validation in canonical diagnostic order."""

        if self.model.get("status", {}).get("kind") != "runs":
            return
        self._initialize()
        self._authenticate_attempts()
        self._validate_accepted_attempts()
        self._validate_work_binding()
        self._validate_per_mode_receipts()
        self._validate_observed_facts()
        self._validate_cold_forward_policy()

    def _initialize(self) -> None:
        """Initialize execution inputs and reducer-derived identities."""

        self.execution = self.model.get("execution", {})
        if not self.execution.get("current"):
            raise ReductionError("runs requires a current execution identity")
        if self.execution.get("network_attempted") or self.execution.get("checkpoint_accessed"):
            raise ReductionError("policy-contaminated execution cannot earn runs")
        self.modes = self.model.get("modes", {})
        self.meaningful_order = tuple(str(mode) for mode in self.modes.get("meaningful_modes", []))
        self.meaningful = set(self.meaningful_order)
        self.per_mode = self.modes.get("per_mode_run", {})
        if self.meaningful != set(self.per_mode):
            raise ReductionError("per_mode_run must cover exactly every meaningful mode")
        self.attempts_by_id = self.reducer._attempt_index()
        self.accepted = set(self.execution.get("accepted_attempt_ids", []))
        self.stable_id = self.model["stable_id"]
        rung = self.model.get("source_resolution", {}).get("rung")
        self.cold_policy = cold_forward_policy(str(self.stable_id), rung)
        self.signatures = {mode: [] for mode in self.meaningful}
        self.counts = {mode: 0 for mode in self.meaningful}
        self.cold_indexes = {mode: set() for mode in self.meaningful}
        self.parent_witnesses = set()
        self.invocations = set()
        self.accepted_work_ids = set()
        self.implementation = self.model.get("implementation", {})
        self.evidence = self.model.get("evidence", {})
        untrusted = self.model.get("untrusted_attempt")
        self.pending_proposal = (
            untrusted.get("proposal") if isinstance(untrusted, Mapping) else None
        )
        pending_facts = (
            self.pending_proposal.get("proposed_facts")
            if isinstance(self.pending_proposal, Mapping)
            and self.model.get("authored_metadata_state") == "pending"
            else None
        )
        code_manifest = self.implementation.get("code_manifest")
        self.expected_manifest_digest = (
            stable_hash(code_manifest)
            if isinstance(code_manifest, list) and code_manifest
            else None
        )
        external_metadata = (
            pending_facts.get("external_metadata")
            if isinstance(pending_facts, Mapping)
            else self.model.get("external_metadata", {})
        )
        modality = (
            external_metadata.get("modality") if isinstance(external_metadata, Mapping) else None
        )
        expected_asset = (
            expected_standard_asset(modality)
            if self.implementation.get("recipe_type") == "declarative-library"
            else None
        )
        self.expected_asset_digest = (
            expected_asset["sha256"] if expected_asset is not None else None
        )
        self.expected_asset_id = expected_asset["asset_id"] if expected_asset is not None else None
        # Replay recomputes at the MODEL's own recorded prompt basis (the
        # accepted-metadata door already proved it equals the sealed gate's);
        # a live append keeps binding the currently shipped bytes. Same
        # carve-out shape as _GateValidationPipeline._prompt_basis.
        self.prompt_basis = (
            str(self.model.get("accuracy_gate", {}).get("prompt_sha256"))
            if bool(getattr(self.reducer, "_semantic_replay", False))
            else _checker_prompt_hash()
        )
        try:
            self.accepted_identities = recompute_accepted_identities(
                pending_facts if isinstance(pending_facts, Mapping) else _model_facts(self.model),
                checker_prompt_hash=self.prompt_basis,
                checker_model=str(self.model.get("accuracy_gate", {}).get("checker_model")),
                checker_version=str(self.model.get("accuracy_gate", {}).get("checker_version")),
                schema_version=MODEL_SCHEMA_VERSION_V3,
            )
        except MetadataValidationError as exc:
            raise ReductionError(str(exc)) from exc
        self.input_contract = self.model.get("input_contract", {})

    def _authenticate_attempts(self) -> None:
        """Authenticate the unique ordered accepted-attempt list."""

        self.accepted_in_order = self.execution.get("accepted_attempt_ids", [])
        if not isinstance(self.accepted_in_order, list) or len(self.accepted_in_order) != len(
            self.accepted
        ):
            raise ReductionError("accepted execution attempt IDs must be a unique ordered list")
        try:
            authenticated_attempts = authenticate_accepted_attempts(
                self.accepted_in_order,
                tuple(self.reducer._attempts.iter_records()),
                stable_id=str(self.stable_id),
                execution_identity=str(self.execution.get("execution_identity")),
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        self.authenticated_by_id = {
            authority.attempt_id: authority for authority in authenticated_attempts
        }

    def _validate_accepted_attempts(self) -> None:
        """Validate every authenticated attempt and accumulate cold-run facts."""

        for attempt_id in self.accepted:
            self._validate_accepted_attempt(attempt_id)

    def _validate_accepted_attempt(self, attempt_id: Any) -> None:
        """Validate one accepted attempt in canonical check order.

        Parameters
        ----------
        attempt_id:
            Accepted attempt identity selected from the execution record.
        """

        attempt = self.attempts_by_id.get(attempt_id)
        if attempt is None:
            raise ReductionError("accepted execution attempt is missing")
        authority = self.authenticated_by_id[attempt_id]
        self.accepted_work_ids.add(authority.work_id)
        identities = attempt.get("identities", {})
        receipt = attempt.get("worker_receipt", {})
        policy = attempt.get("policy_observation", {})
        self._validate_clean_attempt_policy(policy)

        expected_observed_manifest = self.expected_manifest_digest
        raw_receipt = attempt.get("raw_award_receipt")
        if expected_observed_manifest is None and isinstance(raw_receipt, Mapping):
            expected_observed_manifest = raw_receipt.get("code_manifest_identity")
        self._validate_attempt_identities(
            identities,
            receipt,
            expected_observed_manifest=expected_observed_manifest,
        )
        self._validate_and_record_attempt_completion(attempt, receipt)

    def _validate_clean_attempt_policy(self, policy: Any) -> None:
        """Require one accepted attempt's clean policy observation."""

        if not isinstance(policy, Mapping) or any(
            policy.get(key)
            for key in (
                "network_attempted",
                "checkpoint_or_weight_read_attempted",
                "cache_read_attempted",
                "write_outside_scratch_attempted",
                "credentials_present",
                "torchlens_import_attempted",
            )
        ):
            raise ReductionError("accepted attempt lacks a clean successful worker receipt")

    def _validate_attempt_identities(
        self,
        identities: Any,
        receipt: Any,
        *,
        expected_observed_manifest: Any,
    ) -> None:
        """Require one accepted attempt's current model and input identities.

        Parameters
        ----------
        identities:
            Attempt identity projection.
        receipt:
            Worker-owned receipt projection.
        expected_observed_manifest:
            Reducer-derived code-manifest identity.
        """

        observed_asset_pair = (
            receipt.get("observed_input_asset_sha256"),
            receipt.get("input_asset"),
        )
        checks = (
            ("identities.source", identities.get("source") == self.accepted_identities.source),
            (
                "identities.recipe",
                identities.get("recipe") == self.accepted_identities.recipe
                and identities.get("recipe") == self.implementation.get("recipe_revision"),
            ),
            (
                "identities.evidence",
                identities.get("evidence") == self.accepted_identities.evidence
                and identities.get("evidence") == self.evidence.get("evidence_identity"),
            ),
            (
                "identities.environment",
                identities.get("environment") == self.execution.get("env_generation"),
            ),
            (
                "identities.checker_prompt",
                identities.get("checker_prompt") == self.prompt_basis,
            ),
            (
                "worker_receipt.observed_recipe_revision",
                receipt.get("observed_recipe_revision") == self.accepted_identities.recipe,
            ),
            (
                "worker_receipt.observed_adapter_sha256",
                receipt.get("observed_adapter_sha256") == self.implementation.get("code_sha256"),
            ),
            (
                "worker_receipt.observed_code_manifest_sha256",
                receipt.get("observed_code_manifest_sha256") == expected_observed_manifest,
            ),
            (
                "worker_receipt.observed_input_asset",
                observed_asset_pair
                in {(None, None), (self.expected_asset_digest, self.expected_asset_id)},
            ),
        )
        stale = [name for name, current in checks if not current]
        if stale:
            raise ReductionError(
                "accepted attempt identities are stale for the current model: " + ", ".join(stale)
            )

    def _validate_and_record_attempt_completion(
        self,
        attempt: Mapping[str, Any],
        receipt: Any,
    ) -> None:
        """Validate and accumulate one accepted attempt's completion receipt.

        Parameters
        ----------
        attempt:
            Authenticated current-v3 attempt.
        receipt:
            Worker-owned receipt projection.
        """

        mode = str(attempt.get("mode"))
        observation = attempt.get("supervisor_observation", {})
        signature = receipt.get("output_signature")
        retries = attempt.get("retries", {})
        stage_attempt = retries.get("stage_attempt")
        cold_index = stage_attempt - 1 if isinstance(stage_attempt, int) else -1
        mode_index = self.meaningful_order.index(mode) if mode in self.meaningful_order else -1
        # DX-10/RE-11: the identity binds the CARRIED slot retry generation --
        # a forged generation changes the recomputed hash and refuses here, so
        # nothing producer-written steers the binding.
        carried_generation = retries.get("slot_retry_generation")
        retry_generation = (
            carried_generation
            if isinstance(carried_generation, int) and carried_generation > 0
            else 0
        )
        identity_payload: dict[str, Any] = {
            "work_id": attempt.get("work_id"),
            "execution_identity": self.execution.get("execution_identity"),
            "cold_index": cold_index,
            "mode": mode,
        }
        if retry_generation > 0:
            identity_payload["retry_generation"] = retry_generation
        expected_attempt_id = stable_hash(identity_payload)
        completion_line = observation.get("stdout_completion_line")
        invocation = attempt.get("invocation", {})
        argv = invocation.get("argv") if isinstance(invocation, Mapping) else None
        checks = (
            ("mode-not-meaningful", mode in self.meaningful),
            ("result", attempt.get("result") == "succeeded"),
            ("exit_code", observation.get("exit_code") == 0),
            ("signal", observation.get("signal") is None),
            ("constructor_started", bool(receipt.get("constructor_started"))),
            ("constructor_completed", bool(receipt.get("constructor_completed"))),
            ("input_completed", bool(receipt.get("input_completed"))),
            ("forward_started", bool(receipt.get("forward_started"))),
            ("forward_completed", bool(receipt.get("forward_completed"))),
            (
                "input_signature",
                input_signature_matches_contract(
                    receipt.get("input_signature"), self.input_contract
                ),
            ),
            ("output_signature", output_signature_error(signature) is None),
            (
                "cold-forward-binding",
                self.cold_policy.required_cold_forwards <= 1
                or (
                    cold_index >= 0
                    and attempt.get("attempt_no")
                    == cold_index * len(self.meaningful_order) + mode_index + 1
                    and attempt.get("attempt_id") == expected_attempt_id
                    and isinstance(completion_line, str)
                    and isinstance(argv, list)
                    and all(isinstance(value, str) for value in argv)
                ),
            ),
        )
        incomplete = [name for name, complete in checks if not complete]
        if incomplete:
            raise ReductionError(
                "accepted attempt lacks a complete zero-exit receipt: " + ", ".join(incomplete)
            )
        self._record_attempt_completion(
            mode=mode,
            signature=signature,
            cold_index=cold_index,
            completion_line=completion_line,
            argv=argv,
        )

    def _record_attempt_completion(
        self,
        *,
        mode: str,
        signature: Any,
        cold_index: int,
        completion_line: Any,
        argv: Any,
    ) -> None:
        """Accumulate one valid completion and enforce unique cold witnesses.

        Parameters
        ----------
        mode:
            Meaningful execution mode.
        signature:
            Validated output signature.
        cold_index:
            Reducer-derived cold-forward index.
        completion_line:
            Parent-process witness from the supervisor.
        argv:
            Exact isolated invocation arguments.
        """

        self.counts[mode] += 1
        self.signatures[mode].append(signature)
        if self.cold_policy.required_cold_forwards > 1:
            self.cold_indexes[mode].add(cold_index)
            if completion_line in self.parent_witnesses:
                raise ReductionError("accepted cold forwards reuse a parent process witness")
            self.parent_witnesses.add(completion_line)
            invocation_bytes = canonical_json_bytes(argv)
            if invocation_bytes in self.invocations:
                raise ReductionError("accepted cold forwards reuse a scratch/request invocation")
            self.invocations.add(invocation_bytes)
        else:
            self.cold_indexes[mode].add(0)

    def _validate_work_binding(self) -> None:
        """Validate accepted attempts against proposal or anti-slop authority."""

        if len(self.accepted_work_ids) != 1:
            raise ReductionError("accepted attempts span multiple proposal work identities")
        if isinstance(self.pending_proposal, Mapping):
            if str(self.pending_proposal.get("work_id")) not in self.accepted_work_ids:
                raise ReductionError("pending run proposal is stale for the accepted work identity")
            return
        family_variant = str(self.stable_id) in self.reducer.intake_variant_bindings
        if family_variant:
            representative_id = str(
                self.model.get("identity", {}).get("family_representative_id", "")
            )
            representative = self.reducer._validation_current_records().get(representative_id)
            if not isinstance(representative, Mapping) or (
                self.model.get("accuracy_gate") != representative.get("accuracy_gate")
            ):
                raise ReductionError(
                    "family variant anti-slop authority is not its current representative"
                )
            return
        rung_gate = self.reducer._gate_item(
            self.model.get("accuracy_gate", {}).get("gate_id"), str(self.stable_id)
        )
        if rung_gate is None or str(rung_gate[1].get("work_id")) not in self.accepted_work_ids:
            raise ReductionError("anti-slop/rung gate is stale for the accepted work identity")

    def _validate_per_mode_receipts(self) -> None:
        """Validate each meaningful mode's selected worker receipt."""

        for mode in self.meaningful:
            reference = self.per_mode[mode]
            attempt_id = reference.get("attempt_id")
            attempt = self.attempts_by_id.get(attempt_id)
            if attempt_id not in self.accepted or attempt is None:
                raise ReductionError(f"mode {mode} references an unaccepted/missing attempt")
            receipt = attempt["worker_receipt"]
            policy = attempt["policy_observation"]
            if (
                attempt["stable_id"] != self.stable_id
                or attempt["stage"] != "forward"
                or attempt["mode"] != mode
                or attempt["result"] != "succeeded"
                or reference.get("status") != "succeeded"
                or attempt["identities"]["execution"] != self.execution.get("execution_identity")
                or not receipt["present"]
                or not receipt["constructor_started"]
                or not receipt["constructor_completed"]
                or not receipt["input_completed"]
                or not receipt["forward_started"]
                or not receipt["forward_completed"]
                or not input_signature_matches_contract(
                    receipt.get("input_signature"), self.input_contract
                )
                or receipt["mode"] != mode
                or any(
                    policy[key]
                    for key in (
                        "network_attempted",
                        "checkpoint_or_weight_read_attempted",
                        "cache_read_attempted",
                        "write_outside_scratch_attempted",
                        "credentials_present",
                        "torchlens_import_attempted",
                    )
                )
            ):
                raise ReductionError(f"mode {mode} lacks a clean successful worker receipt")

    def _validate_observed_facts(self) -> None:
        """Validate model observations against the designated worker receipt."""

        meaningful_order = self.model.get("modes", {}).get("meaningful_modes", [])
        if not isinstance(meaningful_order, list) or not meaningful_order:
            raise ReductionError("observed facts require an ordered meaningful-mode set")
        measurement_mode = str(meaningful_order[0])
        measurement_reference = self.per_mode.get(measurement_mode, {})
        measurement_attempt = self.attempts_by_id.get(measurement_reference.get("attempt_id"))
        if measurement_attempt is None:
            raise ReductionError("observed facts lack their designated measurement attempt")
        measurement_receipt = measurement_attempt.get("worker_receipt", {})
        measurement_supervisor = measurement_attempt.get("supervisor_observation", {})
        observed = self.model.get("observed", {})
        receipt_observations = {
            "parameter_count_total": measurement_receipt.get("parameter_count_total"),
            "parameter_count_trainable": measurement_receipt.get("parameter_count_trainable"),
            "native_framework": measurement_receipt.get("native_framework"),
            "delegated_method": measurement_receipt.get("delegated_method"),
            "output_signature": measurement_receipt.get("output_signature"),
            "input_kind": measurement_receipt.get("input_kind"),
            "input_asset": measurement_receipt.get("input_asset"),
            "input_note": measurement_receipt.get("input_note"),
            "constructor_seconds": measurement_receipt.get("constructor_seconds"),
            "forward_seconds": measurement_receipt.get("forward_seconds"),
            "peak_rss_bytes": measurement_supervisor.get("peak_rss_bytes"),
            "measurement_attempt_ids": self.accepted_in_order,
        }
        if not isinstance(observed, Mapping) or any(
            observed.get(field) != value for field, value in receipt_observations.items()
        ):
            raise ReductionError("observed runtime facts contradict accepted worker receipts")
        snippet = "driver-owned isolated forward"
        if observed.get("snippet") != snippet or observed.get("snippet_sha256") != stable_hash(
            snippet
        ):
            raise ReductionError("observed snippet is not the driver-owned mechanical recipe")

    def _validate_cold_forward_policy(self) -> None:
        """Validate confirmation policy, cold indexes, and deterministic outputs."""

        confirmation_policy = self.execution.get("confirmation_policy")
        if confirmation_policy != self.cold_policy.confirmation_policy:
            raise ReductionError("execution confirmation policy contradicts reducer policy")
        expected_indexes = set(range(self.cold_policy.required_cold_forwards))
        if any(
            self.counts[mode] != self.cold_policy.required_cold_forwards
            or self.cold_indexes[mode] != expected_indexes
            for mode in self.meaningful
        ):
            raise ReductionError("accepted attempts do not satisfy reducer cold-forward policy")
        if any(
            any(
                canonical_json_bytes(signature) != canonical_json_bytes(mode_signatures[0])
                for signature in mode_signatures[1:]
            )
            for mode_signatures in self.signatures.values()
        ):
            raise ReductionError("non-deterministic cold-forward output signature")


class CanonicalReducer:
    """Exclusive canonical writer enforcing parentage, gates, runs, and statuses.

    Parameters
    ----------
    ledgers:
        Paths to the three canonical append-only ledgers.
    context:
        Mandatory active intake, contract, environment, and policy authority.
    """

    def __init__(
        self,
        ledgers: LedgerPaths,
        context: AuthorityContext,
    ) -> None:
        """Acquire all canonical writer locks and load current facts.

        Parameters
        ----------
        ledgers:
            Paths to canonical ledgers.
        context:
            Mandatory active authority shared by admission and every projection.
        """

        self.ledger_paths = ledgers
        self.context = context
        #: Ledger-attested generation -> family bindings beyond the context's
        #: one-slot-per-family map. A resumed leg whose family REBUILT mid-run
        #: holds terminals whose proofs name an OLDER generation of the same
        #: family; the context map (newest wins) can never resolve them, and
        #: three such models tripped the recording circuit breaker on
        #: 2026-08-13 (m8830/m8838/m8887). The driver registers bindings
        #: recomputed from attempt rows and quarantine events -- replay-derived
        #: evidence, zero producer input -- via
        #: :meth:`register_attested_generation_families`. Resolution still
        #: refuses any generation attested NOWHERE.
        self.historical_generation_families: dict[str, str] = {}
        self.intake_ids = frozenset(context.intake_by_stable_id)
        self.intake_variant_bindings = {
            stable_id: (
                str(
                    binding.get("representative_stable_id", binding.get("family_representative_id"))
                ),
                str(binding.get("variant_token", binding.get("variant", ""))),
            )
            for stable_id, binding in context.family_bindings.items()
            if isinstance(binding, Mapping)
            and binding.get("binding_state") != "ordinary"
            and binding.get("representative_stable_id", binding.get("family_representative_id"))
            not in {None, stable_id}
        }
        unknown_bindings = set(context.family_bindings) - self.intake_ids
        if unknown_bindings:
            raise ReductionError(
                f"family variant bindings exist outside intake: {sorted(unknown_bindings)}"
            )
        opened: list[JsonlLedger] = []
        try:
            self._models = JsonlLedger(ledgers.models, MODEL_SCHEMA_VERSION_V3)
            opened.append(self._models)
            self._attempts = JsonlLedger(ledgers.attempts, ATTEMPT_SCHEMA_VERSION_V3)
            opened.append(self._attempts)
            self._gates = JsonlLedger(ledgers.gates, GATE_SCHEMA_VERSION_V3)
            opened.append(self._gates)
            self._artifacts = ArtifactEventLedger(ledgers.artifacts)
        except Exception:
            artifact_ledger = getattr(self, "_artifacts", None)
            if artifact_ledger is not None:
                artifact_ledger.close()
            for ledger in reversed(opened):
                ledger.close()
            raise
        try:
            for attempt in self._attempts.iter_records():
                _validate_c07_diagnostic_fields(attempt, model=False)
            for model in self._models.iter_records():
                _validate_c07_diagnostic_fields(model, model=True)
        except ReductionError:
            self.close()
            raise
        # _select_current shallow-copies each row, so `_current` values share
        # nested structures with the live ledger records: read-only by the
        # same doctrine as iter_records()/current_view (mutating consumers
        # must deepcopy, as append_model and _maybe_review_hold already do).
        self._current = _select_current(self._models.iter_records())
        # AW-2 (m9819/m9911 class): the attested generation -> family bindings
        # are SELF-DERIVED from this reducer's own ledgers -- the same
        # replay-derived evidence the driver registers -- so environment
        # resolution never depends on which consumer remembered to call
        # ``register_attested_generation_families``. The finish-8 publication
        # seam refused two settled models with "model environment generation
        # is absent from active context" while their generations were attested
        # by their own proof attempts; boundary tools carried a second and
        # third compensation mechanism for the same gap. Driver registration
        # stays additive on top.
        self.historical_generation_families.update(
            _attested_generation_bindings(
                self._attempts.iter_records(),
                _scan_operational_evidence_cached(
                    _records_root(ledgers) / "operational" / "events.jsonl"
                ),
            )
        )
        # IC-5: every mutation of the attested family map bumps this epoch so
        # the projection token counts it (mid-leg registration previously kept
        # serving a projection computed over the pre-mutation map).
        self._generation_families_epoch = 0
        self._projection_cache: Optional[
            tuple[_ProjectionToken, DependencyCurrencyProjection]
        ] = None
        self._attempt_index_cache: Optional[Mapping[str, Mapping[str, Any]]] = None
        self._gate_index_cache: Optional[Mapping[str, Mapping[str, Any]]] = None
        _validate_persisted_requeue_lineage(
            self._models.iter_records(),
            self.ledger_paths,
            self._attempts.iter_records(),
            self._gates.iter_records(),
            self._artifacts.iter_events(),
        )
        # T4-T1 membership half: revisions whose stable id belongs to a
        # REGISTERED PRIOR same-campaign snapshot (the driver-read lineage K2
        # plumbed for currency) are known read-only HISTORY -- a SUBSET rung
        # must open over its superset history. The carve-out is
        # construction-time only: every append-side check below keeps binding
        # to ``intake_ids`` (the CURRENT intake), so prior members still
        # refuse new writes, and a stable id in NEITHER set hard-stops here
        # exactly as before (the foreign-records tripwire).
        unknown = (
            set(self._current) - self.intake_ids - frozenset(self.context.prior_intake_member_ids)
        )
        if unknown:
            self.close()
            raise ReductionError(f"model revisions exist outside intake: {sorted(unknown)}")

    def __enter__(self) -> "CanonicalReducer":
        """Return this exclusive reducer.

        Returns
        -------
        CanonicalReducer
            This reducer.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release every canonical writer lock.

        Parameters
        ----------
        exc_type, exc_value, traceback:
            Context-manager exception state.
        """

        self.close()

    def close(self) -> None:
        """Release every canonical writer lock idempotently."""

        for ledger_name in ("_artifacts", "_gates", "_attempts", "_models"):
            ledger = getattr(self, ledger_name, None)
            if ledger is not None:
                ledger.close()

    def _projection_token(self) -> "_ProjectionToken":
        """Return the live projection-cache token.

        The token counts EVERY mutable input of the projection, per
        :data:`_PROJECTION_TOKEN_MUTABLE_INPUTS` -- the process-stable
        authority-bytes digest, the append generation of every ledger the
        projection reads (T1-15: the code-bytes-only token survived
        gate/attempt appends), and the attested generation-family epoch (IC-5:
        a mid-leg family registration kept serving a projection computed over
        the pre-mutation world). The rotations ledger is deliberately NOT an
        input: since RC-2 the projection never reads a rotation class -- every
        settled head replays at its recorded policy basis unconditionally, and
        the class feeds the recheck census, which is derived per read.

        Returns
        -------
        tuple
            One value per declared mutable input, in declaration order.
        """

        return (
            _dependency_bytes_token(self.ledger_paths),
            self._models.record_count,
            self._attempts.record_count,
            self._gates.record_count,
            self._artifacts.event_count,
            self._generation_families_epoch,
        )

    def _current_projection(self) -> DependencyCurrencyProjection:
        """Return the dependency-current projection for the live ledger state.

        Model appends install their exact replayed patch. Evidence-only growth
        replays only stable IDs named by the appended suffix, including their
        bound family variants. Authority movement or an unpatched model
        generation still falls back to a full derivation.

        Returns
        -------
        DependencyCurrencyProjection
            Projection whose contents callers must never mutate.
        """

        token = self._projection_token()
        cached = self._projection_cache
        if cached is not None and cached[0] == token:
            return cached[1]
        if cached is not None:
            prior_token, prior_projection = cached
            # The patch fast-path is sound ONLY for pure evidence appends:
            # a families-epoch bump changes currency semantics beyond the
            # appended rows' stable IDs, so it forces the full derivation
            # below (IC-5).
            evidence_only_growth = (
                prior_token[0] == token[0]
                and prior_token[1] == token[1]
                and prior_token[5] == token[5]
                and all(old <= new for old, new in zip(prior_token[2:5], token[2:5]))
            )
            if evidence_only_growth:
                affected = self._affected_evidence_stable_ids(prior_token)
                projection = self._patched_projection_after_evidence_growth(
                    prior_projection,
                    affected,
                )
                self._projection_cache = (token, projection)
                return projection
        projection = project_dependency_current(
            self.ledger_paths,
            context=self.context,
            model_records=tuple(self._models.iter_records()),
            attempt_records=tuple(self._attempts.iter_records()),
            gate_records=tuple(self._gates.iter_records()),
            artifact_records=tuple(self._artifacts.iter_events()),
        )
        self._projection_cache = (token, projection)
        return projection

    def _affected_evidence_stable_ids(
        self,
        prior_token: "_ProjectionToken",
    ) -> frozenset[str]:
        """Return stable IDs named by the exact newly appended evidence suffix.

        Parameters
        ----------
        prior_token:
            Projection token whose evidence generations were already consumed.

        Returns
        -------
        frozenset[str]
            Directly named models plus every variant sharing a representative.
        """

        rows: tuple[Mapping[str, Any], ...] = (
            *self._attempts.records_since(prior_token[2]),
            *self._gates.records_since(prior_token[3]),
            *self._artifacts.events_since(prior_token[4]),
        )
        affected: set[str] = set()
        pending: list[Any] = list(rows)
        while pending:
            value = pending.pop()
            if isinstance(value, Mapping):
                stable_id = value.get("stable_id")
                if isinstance(stable_id, str) and stable_id in self.intake_ids:
                    affected.add(stable_id)
                pending.extend(value.values())
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                pending.extend(value)
        representatives = {
            representative
            for stable_id in affected
            for representative in (
                self.intake_variant_bindings.get(stable_id, (stable_id, ""))[0],
            )
        }
        for variant_id, (representative_id, _variant_token) in self.intake_variant_bindings.items():
            if representative_id in representatives:
                affected.add(variant_id)
                affected.add(representative_id)
        return frozenset(affected)

    def _patched_projection_after_evidence_growth(
        self,
        prior_projection: DependencyCurrencyProjection,
        affected: frozenset[str],
    ) -> DependencyCurrencyProjection:
        """Replay only current heads touched by an evidence suffix.

        Parameters
        ----------
        prior_projection:
            Projection computed before the evidence suffix was appended.
        affected:
            Stable IDs named by that suffix, expanded across family bindings.

        Returns
        -------
        DependencyCurrencyProjection
            Exact post-suffix currency with unrelated members retained.
        """

        projected = dict(prior_projection.current_records)
        stale = dict(prior_projection.stale_reasons)
        replay_records = [
            self._current[stable_id]
            for stable_id in affected
            if stable_id in self._current
        ]
        for stable_id in affected:
            projected.pop(stable_id, None)
            stale.pop(stable_id, None)
        for record in sorted(replay_records, key=lambda value: int(value.get("record_seq", 0))):
            stable_id = str(record.get("stable_id"))
            try:
                self._replay_appended_currency(record, projected)
            except (KeyError, ReductionError, TypeError, ValueError) as exc:
                stale[stable_id] = str(exc)
            else:
                projected[stable_id] = deepcopy(dict(record))
        return DependencyCurrencyProjection(projected, stale)

    @property
    def current_records(self) -> Mapping[str, JsonObject]:
        """Return defensive copies of materialized current revisions.

        Returns
        -------
        Mapping[str, dict[str, Any]]
            Stable-ID keyed current model revisions.
        """

        return deepcopy(self._current_projection().current_records)

    @property
    def current_view(self) -> Mapping[str, JsonObject]:
        """Return a read-only view of materialized current revisions.

        The view shares the cached projection without the per-access deepcopy
        behind :attr:`current_records` (measured 0.29 ms/model/access on the
        pilot corpus, at >=6 accesses per model per leg). The mapping itself
        is immutable; nested records are shared and callers must never mutate
        them. Each returned view is a stable snapshot: appends replace the
        cached projection rather than mutating it in place.

        Returns
        -------
        Mapping[str, dict[str, Any]]
            Read-only stable-ID keyed current model revisions.
        """

        records = self._current_projection().current_records
        if isinstance(records, dict):
            return MappingProxyType(records)
        return records

    @property
    def dependency_projection(self) -> "DependencyCurrencyProjection":
        """Return the cached full dependency-currency projection.

        Exposes stale reasons alongside current records so boundary consumers
        can DISCLOSE staleness (the award-holds census) instead of consuming
        absence as nonexistence.

        Returns
        -------
        DependencyCurrencyProjection
            Current records plus fail-closed stale reasons.
        """

        return self._current_projection()

    def has_current(self, stable_id: str) -> bool:
        """Return whether one stable ID holds a dependency-current revision.

        Parameters
        ----------
        stable_id:
            Candidate stable model identifier.

        Returns
        -------
        bool
            ``True`` when the live projection holds a current revision.
        """

        return str(stable_id) in self._current_projection().current_records

    @property
    def dependency_stale_reasons(self) -> Mapping[str, str]:
        """Return the projection's per-member stale reasons (routing/census data).

        The projection's stale bucket used to be computed and thrown away
        (``materialize_current`` returned only ``current_records``), so every
        consumer saw "stale" as "absent" with the reason discarded -- the
        vector-8b mechanism. Selection now reads existence from
        :attr:`raw_head_view` and consumes THIS map as disclosure (the
        leg-start projection-staleness census and the operator acknowledgment)
        and as the close's stale-label authority
        (:func:`~menagerie.crawler.driver._scoped_stale_reasons` quotes it, so
        the close and ``status --verify-partition`` render one label per
        refusal -- the D-17 contract).

        Returns
        -------
        Mapping[str, str]
            Stable-ID keyed replay-failure reasons for stale current heads.
        """

        return MappingProxyType(dict(self._current_projection().stale_reasons))

    @property
    def raw_head_view(self) -> Mapping[str, JsonObject]:
        """Return the RAW model-ledger heads: THE settlement authority (WS-1).

        This is the construction-time :func:`_select_current` map (public
        supersession lineage validated), maintained on every append -- never
        the dependency-currency projection. "A stale award is still an
        award": the reducer's own award guard consults exactly this map
        (:meth:`award_blocked_supersession`), and the work-selection policy
        (:mod:`menagerie.crawler.work_selection`) reads settlement from it
        and nothing else. Read-only by the same doctrine as
        :attr:`current_view`: nested records are shared and must never be
        mutated.

        Returns
        -------
        Mapping[str, dict[str, Any]]
            Read-only stable-ID keyed raw head revisions.
        """

        return MappingProxyType(self._current)

    @property
    def artifact_ledger(self) -> ArtifactEventLedger:
        """Return the reducer-owned artifact writer held in canonical lock order."""

        return self._artifacts

    @property
    def gate_records(self) -> tuple[JsonObject, ...]:
        """Return defensive copies from the already-open gate ledger.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Persisted gate records in ledger order.
        """

        return tuple(self._gates.records)

    def work_generation_environment_attempt(
        self, stable_id: str, work_id: str
    ) -> Optional[Mapping[str, Any]]:
        """Return the latest authenticated attempt naming this generation's environment.

        One work generation's environment is a durable fact its attempt history
        already carries, and ``_derive_environment`` refuses any candidate whose
        ``execution.env_generation`` contradicts the attempts its terminal proof
        cites. A driver leg that terminalizes a generation WITHOUT running an
        attempt of its own (the pre-dispatch author-session cap: the allowance
        was already spent by earlier legs, so no session and no forward run
        happens on this leg) has no local attempt carrying an environment, and
        stamping the ``not-applicable`` sentinel contradicts the prior legs'
        forward attempts the proof still cites. Callers use this to name the
        SAME authenticated authority the validator will check against, so the
        tripwire stays armed and the candidate stops contradicting itself.

        Parameters
        ----------
        stable_id, work_id:
            Exact model and work-generation identities.

        Returns
        -------
        Optional[Mapping[str, Any]]
            Latest current-v3 attempt for the pair whose
            ``identities.environment`` is a string, or ``None`` when this
            generation's authenticated history names no environment at all.
        """

        return next(
            (
                attempt
                for attempt in reversed(self._current_attempt_records())
                if attempt.get("stable_id") == str(stable_id)
                and attempt.get("work_id") == str(work_id)
                and isinstance((attempt.get("identities") or {}).get("environment"), str)
            ),
            None,
        )

    def terminal_work_generation(self, stable_id: str) -> Optional[str]:
        """Return the work generation a settled head's own terminal proof cites.

        Replay-derived with zero producer input: the raw ledger head names its
        decisive attempts (``status.attempt_ids``), every one of which is a
        reducer-blessed attempt row carrying its exact ``work_id``. The newest
        cited attempt that names a generation decides. The terminal-review
        backlog sweep uses this to key its gate-coverage and rehydration
        lookups to the SAME generation the recorded terminal adjudicated,
        never a guessed default -- a grant-minted generation's terminal must
        not be answered with the organic generation's gate.

        Parameters
        ----------
        stable_id:
            Durable model identifier.

        Returns
        -------
        str | None
            The cited generation's ``work_id``, or ``None`` when the head is
            absent or its authenticated history names no generation at all.
        """

        head = self._current.get(str(stable_id))
        if head is None:
            return None
        attempt_ids = head.get("status", {}).get("attempt_ids") or []
        index = self._attempt_index()
        for attempt_id in reversed([str(value) for value in attempt_ids]):
            attempt = index.get(attempt_id)
            if attempt is None:
                continue
            work_id = attempt.get("work_id")
            if isinstance(work_id, str) and work_id:
                return work_id
        return None

    def _attempt_index(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the shared immutable attempt-ID index for reducer validation."""

        cached = getattr(self, "_attempt_index_cache", None)
        if cached is None:
            cached = {
                str(record.get("attempt_id")): record for record in self._attempts.iter_records()
            }
            self._attempt_index_cache = cached
        return cached

    def register_attested_generation_families(self, bindings: Mapping[str, str]) -> None:
        """Register ledger-attested generation -> family bindings for terminal assembly.

        Parameters
        ----------
        bindings:
            Generation identity to environment family/intent name, recomputed
            by the driver exclusively from attempt rows and quarantine events.
            Merged additively; the context's current map always wins first at
            resolution time, and unattested generations still refuse.
        """

        for generation, family in bindings.items():
            if isinstance(generation, str) and generation and isinstance(family, str) and family:
                if self.historical_generation_families.get(generation) != family:
                    # IC-5: only a REAL map movement invalidates the cached
                    # projection; identical re-registration stays warm.
                    self._generation_families_epoch = (
                        getattr(self, "_generation_families_epoch", 0) + 1
                    )
                self.historical_generation_families[generation] = family

    def refresh_attested_generation_families(self) -> Mapping[str, str]:
        """Re-derive ledger-attested generation -> family bindings from live ledgers.

        Construction-time derivation and admission-time driver registration are
        both point-in-time snapshots; a family that rebuilds mid-leg attests its
        new generation in attempt rows and lifecycle events appended AFTER them.
        Terminal assembly calls this once on its resolution miss path so those
        models still record (leg-28 ``TerminalRecordingUnavailable``: three
        consecutive gate-round terminals refused with
        "model environment generation is absent from active context" while
        their proof attempts attested the binding). Same closed evidence
        sources as construction, replay-derived with zero producer input,
        additive exactly like :meth:`register_attested_generation_families`.

        Returns
        -------
        Mapping[str, str]
            The refreshed generation -> family map (shared, read-only use).
        """

        self.register_attested_generation_families(
            _attested_generation_bindings(
                self._attempts.iter_records(),
                _scan_operational_evidence_cached(
                    _records_root(self.ledger_paths) / "operational" / "events.jsonl"
                ),
            )
        )
        return self.historical_generation_families

    def update_context(self, context: AuthorityContext) -> None:
        """Replace only runtime-current environment axes without changing intake roots.

        Parameters
        ----------
        context:
            Refreshed authority after exact environment materialization.

        Raises
        ------
        ReductionError
            If a caller attempts to change the active intake or contract roots
            during one locked reducer lifetime.
        """

        stable_axes = (
            "active_intake_snapshot_id",
            "active_intake_snapshot_sha256",
            "intake_by_stable_id",
            "family_bindings",
            "author_prompt_identity",
            "author_model_identity",
            "author_schema_identity",
            "author_dispatcher_identity",
            "author_model_fields",
            "checker_prompt_identity",
            "checker_model_identity",
            "checker_schema_identity",
            "checker_model_fields",
            "reducer_policy_identity",
            "runner_policy_identity",
            "terminal_policy_identity",
            "publication_policy_identity",
            "prior_intake_snapshots",
            "prior_intake_member_ids",
        )
        changed_axes = [
            axis for axis in stable_axes if getattr(context, axis) != getattr(self.context, axis)
        ]
        if changed_axes:
            raise ReductionError(
                "locked reducer context changed a non-environment authority axis: "
                f"{', '.join(changed_axes)}; close and reopen the reducer under the new context"
            )
        self.context = context
        self._projection_cache = None

    def _gate_index(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the shared immutable gate-ID index for reducer validation."""

        cached = getattr(self, "_gate_index_cache", None)
        if cached is None:
            cached = {str(record.get("gate_id")): record for record in self._gates.iter_records()}
            self._gate_index_cache = cached
        return cached

    def _validation_current_records(self) -> Mapping[str, Mapping[str, Any]]:
        """Return the dependency-current map selected for this admission pass."""

        if bool(getattr(self, "_semantic_replay", False)):
            projected = getattr(self, "_validation_current", None)
            if isinstance(projected, Mapping):
                return projected
            return self._current
        return self._current_projection().current_records

    def _current_attempt_records(self) -> tuple[Mapping[str, Any], ...]:
        """Return only attempts carrying authenticated current-v3 proof authority.

        Returns
        -------
        tuple[Mapping[str, Any], ...]
            Current authenticated attempts in immutable ledger order. Legacy rows
            remain readable audit history but cannot participate in model authority.

        Raises
        ------
        ReductionError
            If a purported current-v3 row fails its proof replay.
        """

        current: list[Mapping[str, Any]] = []
        for attempt in self._attempts.iter_records():
            if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION_V3:
                continue
            try:
                load_current_attempt_proof(attempt)
            except AuthorityDerivationError as exc:
                raise ReductionError(str(exc)) from exc
            current.append(attempt)
        return tuple(current)

    def _current_gate_records(self) -> tuple[Mapping[str, Any], ...]:
        """Return only gates carrying authenticated current-v3 proof authority.

        Returns
        -------
        tuple[Mapping[str, Any], ...]
            Current authenticated gates in immutable ledger order.

        Raises
        ------
        ReductionError
            If a purported current-v3 gate fails its proof replay.
        """

        current: list[Mapping[str, Any]] = []
        for gate in self._gates.iter_records():
            if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3:
                continue
            # W44 SHADOW-EXCLUSION: a shadow-rehearsal-marked gate (R6b) is a
            # real, attested adjudication of a REHEARSAL world -- it never
            # accumulates into current proof authority. Keyed on the record's
            # own marker; disclosure is the segregated rehearsal census
            # (W3's half), never this accumulator.
            if is_shadow_rehearsal_record(gate):
                continue
            try:
                load_current_gate_proof(gate)
            except AuthorityDerivationError as exc:
                raise ReductionError(str(exc)) from exc
            current.append(gate)
        return tuple(current)

    def append_attempt(self, attempt: Mapping[str, Any]) -> AppendResult:
        """Append one immutable attempt after parent/work validation.

        Parameters
        ----------
        attempt:
            Attempt payload; ledger sequence and payload hash may be omitted.

        Returns
        -------
        AppendResult
            Idempotent append result.
        """

        _validate_c07_diagnostic_fields(attempt, model=False)
        if attempt.get("schema_version") != ATTEMPT_SCHEMA_VERSION_V3:
            raise ReductionError("attempt appends require the current v3 authority contract")
        stable_id = attempt.get("stable_id")
        if stable_id is not None and stable_id not in self.intake_ids:
            raise ReductionError(f"attempt stable_id is outside intake: {stable_id}")
        parent_id = attempt.get("parent_attempt_id")
        by_id = self._attempt_index()
        if parent_id is not None:
            parent = by_id.get(parent_id)
            if parent is None:
                raise ReductionError(f"missing parent attempt {parent_id}")
            if parent["work_id"] != attempt.get("work_id"):
                raise ReductionError("parent attempt belongs to a different work identity")
            if int(attempt.get("attempt_no", 0)) <= int(parent["attempt_no"]):
                raise ReductionError("attempt_no must increase after its parent")
        error = attempt.get("error")
        if attempt.get("result") == "failed":
            if not isinstance(error, Mapping):
                raise ReductionError("failed attempt requires a structured error")
            stage = error.get("stage")
            if stage != attempt.get("stage"):
                raise ReductionError("attempt and error stages must match")
            if error.get("reason_code") not in FAILURE_REASON_CODES.get(str(stage), frozenset()):
                raise ReductionError("attempt error reason is not allowed for its stage")
        elif error is not None:
            raise ReductionError("non-failed attempt cannot carry an error")
        environment = attempt.get("environment")
        identities = attempt.get("identities", {})
        receipt = attempt.get("worker_receipt", {})
        observation = attempt.get("supervisor_observation", {})
        if environment is None:
            if identities.get("environment") is not None or identities.get("execution") is not None:
                raise ReductionError(
                    "an unobserved environment cannot carry environment/execution identities"
                )
            if receipt.get("present"):
                raise ReductionError("a present worker receipt requires observed environment facts")
            if (
                observation.get("stdout_sha256") is not None
                or observation.get("stderr_sha256") is not None
            ):
                raise ReductionError("unobserved process streams cannot carry artifact digests")
        elif not all(
            environment.get(field)
            for field in (
                "lock_sha256",
                "resolved_export_sha256",
                "packages_manifest_sha256",
            )
        ):
            raise ReductionError("observed environment facts require exact artifact digests")
        elif observation.get("stdout_sha256") is None or observation.get("stderr_sha256") is None:
            raise ReductionError("an observed subprocess requires exact stream-byte digests")
        if environment is not None:
            self._validate_environment_generation(attempt)
        if attempt.get("result") == "succeeded" and (
            receipt.get("observed_recipe_revision") != identities.get("recipe")
            or "observed_adapter_sha256" not in receipt
        ):
            raise ReductionError("successful worker receipt lacks current observed recipe bindings")
        if attempt.get("result") == "succeeded":
            raw_receipt = attempt.get("raw_award_receipt")
            parent_attestation = attempt.get("parent_attestation")
            if not isinstance(raw_receipt, Mapping) or not isinstance(parent_attestation, Mapping):
                raise ReductionError("successful attempt lacks retained v3 raw proof")
            try:
                derive_attempt_projection(
                    raw_receipt,
                    parent_attestation,
                    candidate_attempt=attempt,
                )
            except AuthorityDerivationError as exc:
                raise ReductionError(str(exc)) from exc
        elif any(
            attempt.get(field) is not None
            for field in ("raw_award_receipt", "raw_award_receipt_sha256")
        ):
            raise ReductionError("non-success attempt cannot retain award-eligible raw proof")
        replayed = self._replayed_driver_failure_attempt(attempt)
        if replayed is not None:
            return AppendResult(replayed, appended=False)
        result = self._attempts.append(attempt)
        if result.appended:
            self._attempt_index_cache = None
        return result

    def _replayed_driver_failure_attempt(self, attempt: Mapping[str, Any]) -> Optional[JsonObject]:
        """Return the recorded row a driver-failure replay must reuse, if any.

        A driver-observed lane failure has a TIME-FREE immutable identity
        (:func:`~menagerie.crawler.recordio.driver_failure_attempt_identity`
        over work generation, stage, reason, and root-cause fingerprint), but
        its record carries wall-clock observation fields. A resumed leg that
        re-observes the same failure therefore re-derives the same identity
        with fresh clocks, and appending that replay raised
        ``LedgerConflictError`` and killed the campaign (rung-11 leg-4:
        m4256's repeating supersession refusal re-derived attempt
        ``f4626f26`` on every resume, diverging from the frozen row only at
        ``started_at``/``finished_at`` and the attestation recomputed over
        them). The recorded row is append-only history: a replay that reaches
        an existing identity REUSES the row exactly as recorded, provided the
        recorded row still derives its own identity from its own recorded
        basis. A recorded row that contradicts its own basis, a non-driver
        actor, or any basis disagreement falls through to the ledger's
        conflict tripwire untouched -- and a genuinely NEW failure (different
        inputs) hashes to a different identity and appends normally.

        Parameters
        ----------
        attempt:
            Candidate attempt that passed the structural validations above.

        Returns
        -------
        dict[str, Any] | None
            Defensive copy of the recorded row to reuse, or ``None``.
        """

        attempt_id = attempt.get("attempt_id")
        if not isinstance(attempt_id, str):
            return None
        recorded = self._attempt_index().get(attempt_id)
        if recorded is None:
            return None

        def _failure_basis(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
            error = row.get("error")
            if not isinstance(error, Mapping):
                return None
            return {
                "work_id": row.get("work_id"),
                "stage": row.get("stage"),
                "reason_code": error.get("reason_code"),
                "root_cause_fingerprint": error.get("root_cause_fingerprint"),
            }

        proposed_basis = _failure_basis(attempt)
        recorded_basis = _failure_basis(recorded)
        if (
            attempt.get("actor") != "driver"
            or recorded.get("actor") != "driver"
            or attempt.get("result") != "failed"
            or recorded.get("result") != "failed"
            or recorded.get("schema_version") != ATTEMPT_SCHEMA_VERSION_V3
            or attempt.get("stable_id") != recorded.get("stable_id")
            or proposed_basis is None
            or recorded_basis is None
            or proposed_basis != recorded_basis
            or not all(isinstance(value, str) and value for value in recorded_basis.values())
            or driver_failure_attempt_identity(
                work_id=str(recorded_basis["work_id"]),
                stage=str(recorded_basis["stage"]),
                reason_code=str(recorded_basis["reason_code"]),
                root_cause_fingerprint=str(recorded_basis["root_cause_fingerprint"]),
            )
            != attempt_id
        ):
            return None
        return deepcopy(dict(recorded))

    def _validate_environment_generation(self, attempt: Mapping[str, Any]) -> None:
        """Recompute production environment identity from committed exact artifacts.

        Parameters
        ----------
        attempt:
            Candidate attempt carrying observed environment and generation facts.

        Raises
        ------
        ReductionError
            If a canonical campaign attempt is not backed by exact committed bytes.
        """

        records_root = self.ledger_paths.models.parent.parent
        canonical_root = records_root.parent
        production_layout = (
            records_root.name == "records"
            and canonical_root.name == "crawler"
            and canonical_root.parent.name == "menagerie"
        )
        if not production_layout:
            return
        environment = attempt.get("environment")
        identities = attempt.get("identities")
        if not isinstance(environment, Mapping) or not isinstance(identities, Mapping):
            raise ReductionError("environment generation facts are malformed")
        family = environment.get("family")
        target = environment.get("target")
        if not isinstance(family, str) or not isinstance(target, str):
            raise ReductionError("environment generation lacks family/target")
        if bool(getattr(self, "_semantic_replay", False)):
            # Replay of a RECORDED head anchors to ledger attestation, not to
            # today's committed artifacts: the committed lock/export/probe
            # bytes legitimately move between the record and its replay
            # (rebuilds, round updates, decommissions), and recomputing the
            # base from live disk re-adjudicates history against the wrong
            # era (leg-29: six verified awards read
            # "does not chain to the committed base generation" and were one
            # relaunch away from the AW-2 re-drive incident). The attested
            # generation identity COMMITS to the exact recorded bytes, and
            # the bindings are parent-attested (driver lifecycle events and
            # canonical attempt rows), never producer input. A generation
            # attested nowhere still falls through to the live-disk anchor
            # and its fail-closed refusals.
            declared = attempt.get("identities", {}).get("environment")
            if (
                isinstance(declared, str)
                and declared
                and self.historical_generation_families.get(declared) == family
            ):
                return
        try:
            registry = load_environment_registry(canonical_root / "envs", target=target)
            intent = registry.intents[family]
            lock_bytes = intent.lock.lock_path.read_bytes()
            export_bytes = intent.lock.export_path.read_bytes()
            export_hash = intent.lock.export_hash_path.read_text(encoding="utf-8").strip()
            receipt_path = intent.lock.lock_path.with_name(f"{target}.probes.json")
            probe_results = parse_probe_receipt_bytes(intent.probes, receipt_path.read_bytes())
            package_bytes = parse_resolved_export(export_bytes)
            generation = materialized_environment_generation(
                intent,
                lock_bytes=lock_bytes,
                export_bytes=export_bytes,
                package_bytes=package_bytes,
                python_version=str(environment.get("python")),
                compiler_identity=str(environment.get("compiler_identity")),
                sdk_identity=str(environment.get("sdk_identity")),
                probe_results=probe_results,
            )
        except (
            KeyError,
            OSError,
            UnicodeError,
            EnvironmentSpecError,
            EnvironmentExactnessError,
            EnvironmentProbeError,
        ) as exc:
            raise ReductionError(
                "observed environment lacks exact committed generation artifacts"
            ) from exc
        # ``generation`` is the machine-derived v1 base recomputed from committed
        # lock/export/package/probe bytes. A strict authority binding layers the
        # parent-witnessed prefix content seal over that base (generation v2), so
        # an authority-bound attempt legitimately carries the LAYERED identity.
        # The layer is only accepted when its declared base equals the base this
        # reducer just recomputed: the anchor stays machine-derived, and a record
        # whose base drifts from the committed artifacts still refuses here.
        declared_base = environment.get("base_environment_generation")
        declared_content = environment.get("environment_content_sha256")
        if declared_base is None and declared_content is None:
            expected_generation = generation
        elif (
            isinstance(declared_base, str)
            and isinstance(declared_content, str)
            and declared_base == generation
            and is_sha256(declared_content)
        ):
            expected_generation = layered_environment_generation(generation, declared_content)
        else:
            raise ReductionError(
                "observed environment authority layer does not chain to the committed "
                f"base generation: base_expected={generation}, "
                f"base_observed={declared_base}, "
                "content_seal_well_formed="
                f"{isinstance(declared_content, str) and is_sha256(declared_content)}"
            )
        if (
            environment.get("lock_sha256") != hash_bytes(lock_bytes)
            or environment.get("resolved_export_sha256") != hash_bytes(export_bytes)
            or environment.get("packages_manifest_sha256") != hash_bytes(package_bytes)
            or export_hash != hash_bytes(export_bytes)
            or identities.get("environment") != expected_generation
        ):
            raise ReductionError(
                "observed environment generation is stale or self-attested: "
                f"expected={expected_generation}, observed={identities.get('environment')}, "
                f"lock={environment.get('lock_sha256') == hash_bytes(lock_bytes)}, "
                f"export={environment.get('resolved_export_sha256') == hash_bytes(export_bytes)}, "
                f"packages={environment.get('packages_manifest_sha256') == hash_bytes(package_bytes)}"
            )

    def append_gate(self, gate: Mapping[str, Any]) -> AppendResult:
        """Append one immutable checker gate after intake/integrity checks.

        Parameters
        ----------
        gate:
            Gate payload; ledger sequence and payload hash may be omitted.

        Returns
        -------
        AppendResult
            Idempotent append result.
        """

        if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3:
            raise ReductionError("gate appends require the current v3 authority contract")
        items = gate.get("items", [])
        stable_ids = [item.get("stable_id") for item in items]
        if gate.get("batch_size") != len(items):
            raise ReductionError("gate batch_size must equal the number of items")
        if len(stable_ids) != len(set(stable_ids)):
            raise ReductionError("gate items must have unique stable IDs")
        # WRITE-time live-world checks: a gate being sealed into the ledger NOW
        # must bind the currently shipped prompt and component bytes -- this is
        # the anti-templating door (a checker output claiming identities the
        # machine cannot re-derive refuses here). A SEMANTIC REPLAY re-admits a
        # persisted gate as append-only history at its own recorded basis: its
        # identities are committed by the gate's ``result_envelope_sha256``
        # self-hash and the ledger row hash, and re-imposing the live bytes on
        # it refused every sealed gate whenever prompts or the dispatcher
        # legitimately evolved between legs (the rung-11 history-vs-live
        # family). Currency of the standing MODEL record against the live
        # contract stays ``validate_currency``'s decision.
        if not bool(getattr(self, "_semantic_replay", False)):
            _refuse_unmarked_gate_in_shadow_mode(gate)
            if gate.get("checker", {}).get("prompt_sha256") != _checker_prompt_hash():
                raise ReductionError("checker gate does not bind the current prompt bytes")
            for field, component in (
                ("author_result_schema_identity", AUTHOR_RESULT_SCHEMA_COMPONENT),
                ("dispatcher_identity", AUTHOR_DISPATCHER_COMPONENT),
            ):
                try:
                    _require_derived_component_identity(gate.get(field), f"gate.{field}", component)
                except AuthorityDerivationError as exc:
                    raise ReductionError(str(exc)) from exc
        for item in items:
            # Intake membership is a WRITE-time live-world check like the
            # prompt/component bindings above: the sealing door validated
            # every item against ITS OWN era's intake. Re-imposing today's
            # scope on sealed history refuses evidence whenever scope
            # legitimately evolved -- the leg-30 close read seven verified
            # awards stale because their step-100-era batches also reviewed
            # models of sibling tier campaigns ("gate item stable_id is
            # outside intake"). Replay re-admits the row at its recorded
            # basis; the foreign items are adjudicated by THEIR campaigns'
            # replays, and this campaign's award binds only its own item.
            if not bool(getattr(self, "_semantic_replay", False)) and item.get(
                "stable_id"
            ) not in self.intake_ids:
                raise ReductionError(
                    f"gate item stable_id is outside intake: {item.get('stable_id')}"
                )
            if not isinstance(item.get("campaign_root_work_id"), str) or not item.get(
                "campaign_root_work_id"
            ):
                raise ReductionError("gate item has no campaign/root-work lineage identity")
            verdict = item.get("verdict")
            integrity = item.get("integrity", {}).get("verdict")
            field_verdicts = [check.get("verdict") for check in item.get("field_checks", [])]
            if verdict == "accurate" and (
                integrity != "accurate" or any(value != "accurate" for value in field_verdicts)
            ):
                raise ReductionError(
                    "an accurate gate item requires accurate integrity/field checks"
                )
        result = self._gates.append(gate)
        if result.appended:
            self._gate_index_cache = None
        return result

    def _artifact_authority_inputs(self, model: Mapping[str, Any]) -> JsonObject:
        """Resolve and validate one model's exact append-only artifact authority.

        Parameters
        ----------
        model:
            Candidate v3 model revision.

        Returns
        -------
        dict[str, Any]
            Latest event plus immutable reconstruction inputs when applicable.

        Raises
        ------
        ReductionError
            If the model names no exact transaction state or immutable bytes.
        """

        authority = model.get("artifact_authority")
        if not isinstance(authority, Mapping):
            raise ReductionError("model lacks its mandatory artifact authority block")
        if authority.get("state") == DependencyState.NOT_APPLICABLE.value:
            expected = ArtifactAuthorityRecord.not_applicable().to_payload()
            if dict(authority) != expected:
                raise ReductionError("not-applicable artifact authority is not closed")
            return {
                "event": None,
                "document": None,
                "transaction_id": DependencyState.NOT_APPLICABLE,
                "claim_ids": (),
            }
        transaction_id = authority.get("transaction_id")
        events = [
            event
            for event in self._artifacts.iter_events()
            if event.get("transaction_id") == transaction_id
        ]
        if not events:
            raise ReductionError("model artifact transaction is absent from the append-only ledger")
        latest = events[-1]
        claims = tuple(sorted(str(value.get("claim_id")) for value in latest.get("claims", [])))
        reconstruction = latest.get("reconstruction")
        reconstruction_sha256: object = DependencyState.NOT_APPLICABLE.value
        document: Optional[JsonObject] = None
        if isinstance(reconstruction, Mapping):
            anchor = ReconstructionAnchorRecord.from_payload(reconstruction)
            reconstruction_sha256 = anchor.sha256
            if not isinstance(reconstruction.get("path"), str):
                raise ReductionError("artifact reconstruction lacks its immutable path")
            path_value = anchor.path
            canonical_root = _production_canonical_root(self.ledger_paths)
            if canonical_root is not None:
                repository_root = canonical_root.parents[1]
                path = (repository_root / path_value).resolve()
                try:
                    raw = path.read_bytes()
                    parsed = json.loads(raw)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ReductionError("artifact reconstruction bytes are unavailable") from exc
                if hash_bytes(raw) != reconstruction_sha256 or not isinstance(parsed, dict):
                    raise ReductionError("artifact reconstruction digest changed")
                document = parsed
        expected_authority = ArtifactAuthorityRecord.from_event(latest).to_payload()
        if dict(authority) != expected_authority:
            raise ReductionError("model artifact authority contradicts the latest ledger event")
        return {
            "event": latest,
            "document": document,
            "transaction_id": str(transaction_id),
            "claim_ids": claims,
        }

    def _model_work_id(self, model: Mapping[str, Any], artifact_inputs: Mapping[str, Any]) -> str:
        """Resolve one exact work generation from all model authority references.

        Parameters
        ----------
        model, artifact_inputs:
            Candidate model and its resolved artifact transaction.

        Returns
        -------
        str
            Unique work identity shared by attempts, gates, proposal, and custody.
        """

        stable_id = str(model.get("stable_id"))
        work_ids: set[str] = set()
        referenced_attempts = set(_record_attempt_ids(model))
        for attempt in self._attempts.iter_records():
            if (
                attempt.get("attempt_id") in referenced_attempts
                and attempt.get("stable_id") == stable_id
                and isinstance(attempt.get("work_id"), str)
            ):
                work_ids.add(str(attempt["work_id"]))
        event = artifact_inputs.get("event")
        if isinstance(event, Mapping) and isinstance(event.get("work_id"), str):
            work_ids.add(str(event["work_id"]))
        for block_name in ("accuracy_gate", "fidelity"):
            gate_id = model.get(block_name, {}).get("gate_id")
            found = self._gate_item(gate_id, stable_id)
            if found is not None and isinstance(found[1].get("work_id"), str):
                work_ids.add(str(found[1]["work_id"]))
        retained = model.get("untrusted_attempt")
        proposal = retained.get("proposal") if isinstance(retained, Mapping) else None
        if isinstance(proposal, Mapping) and isinstance(proposal.get("work_id"), str):
            work_ids.add(str(proposal["work_id"]))
        if len(work_ids) != 1:
            raise ReductionError(
                f"model authority for {stable_id} does not resolve one work identity: "
                f"{sorted(work_ids)} (referenced attempts, gates, artifact events, and "
                "the retained proposal must all name the same work generation)"
            )
        return next(iter(work_ids))

    def _derive_model_authority(
        self, model: Mapping[str, Any]
    ) -> tuple[TerminalProof, FamilyAuthority, JsonObject, JsonObject]:
        """Derive terminal, family, dependency, mode, and artifact projections.

        Parameters
        ----------
        model:
            Candidate current v3 model revision.

        Returns
        -------
        tuple[TerminalProof, FamilyAuthority, dict[str, Any], dict[str, Any]]
            Exact proof objects plus the derived vector and artifact inputs.
        """

        return _ModelAuthorityPipeline(self, model).run()

    def _validate_derived_model_authority(self, model: Mapping[str, Any]) -> None:
        """Require exact reducer-derived proof, family, and dependency projections.

        Parameters
        ----------
        model:
            Candidate v3 model revision.

        Raises
        ------
        ReductionError
            If any copied driver projection differs from replayed authority.
        """

        terminal_proof, family_authority, vector, _artifact_inputs = self._derive_model_authority(
            model
        )
        if model.get("family_authority") != family_authority_projection(family_authority):
            raise ReductionError("model family authority contradicts trusted intake")
        recorded_vector = model.get("dependency_vector")
        if recorded_vector != vector:
            # Name the model and the exact components that differ: the bare
            # "contradicts reducer derivation" refusal forced an operator to
            # re-derive the vector by hand to learn which axis moved.
            if isinstance(recorded_vector, Mapping):
                differing = sorted(
                    key
                    for key in set(recorded_vector) | set(vector)
                    if recorded_vector.get(key) != vector.get(key)
                )
                detail = ", ".join(
                    f"{key}: recorded={recorded_vector.get(key)!r} derived={vector.get(key)!r}"
                    for key in differing
                )
            else:
                detail = f"recorded vector is {type(recorded_vector).__name__}, not a mapping"
            raise ReductionError(
                "model dependency vector contradicts reducer derivation for "
                f"{model.get('stable_id')}: {detail}"
            )
        status = model.get("status")
        if not isinstance(status, Mapping):
            raise ReductionError("model status is malformed")
        if status.get("code") != terminal_proof.status_code:
            raise ReductionError("model status code contradicts its terminal proof")
        if terminal_proof.status_code.startswith("failed:") and (
            status.get("stage") != terminal_proof.failure_stage
            or status.get("reason_code") != terminal_proof.reason_code
            or status.get("root_cause_fingerprint") != terminal_proof.root_cause_fingerprint
        ):
            raise ReductionError("failed status fields contradict its decisive proof")
        currency_context = self.context
        vector_generation = vector.get("environment_generation")
        if (
            isinstance(vector_generation, str)
            and vector_generation
            and vector_generation != DependencyState.NOT_APPLICABLE.value
            and vector_generation not in set(self.context.environment_generations.values())
            and vector_generation in self.historical_generation_families
        ):
            # The derived vector (recorded==derived enforced above) carries a
            # LEDGER-ATTESTED generation whose family holds no live slot --
            # rebuilt or torn down mid-leg before this terminal recorded.
            # Currency-validate under the same recorded-basis extension the
            # award carve-out uses: the reserved key never shadows a real
            # family, and a generation attested nowhere still reads stale
            # (leg-28 m1163/m1221/m2062 circuit-break).
            currency_context = replace(
                self.context,
                environment_generations={
                    **dict(self.context.environment_generations),
                    _AWARD_RECORDED_BASIS_GENERATION_KEY: vector_generation,
                },
            )
        stale_reason = validate_currency(
            currency_context,
            model,
            terminal_proof=terminal_proof,
            family_authority=family_authority,
        )
        if stale_reason is not None:
            raise ReductionError(stale_reason)

    def prepare_model(self, model: Mapping[str, Any]) -> JsonObject:
        """Project one driver candidate exclusively from reducer-owned authority.

        Parameters
        ----------
        model:
            Structurally assembled candidate without copied authority fields.

        Returns
        -------
        dict[str, Any]
            Candidate with exact artifact, mode, family, and dependency projections.
        """

        candidate = deepcopy(dict(model))
        stable_id = str(candidate.get("stable_id"))
        events = [
            event for event in self._artifacts.iter_events() if event.get("stable_id") == stable_id
        ]
        referenced_attempt_ids = {
            str(value) for value in candidate.get("status", {}).get("attempt_ids", [])
        }
        referenced_work_ids = {
            str(attempt.get("work_id"))
            for attempt in self._attempts.iter_records()
            if attempt.get("attempt_id") in referenced_attempt_ids
            and isinstance(attempt.get("work_id"), str)
        }
        untrusted = candidate.get("untrusted_attempt")
        proposal = untrusted.get("proposal") if isinstance(untrusted, Mapping) else None
        if isinstance(proposal, Mapping) and isinstance(proposal.get("work_id"), str):
            referenced_work_ids.add(str(proposal["work_id"]))
        if len(referenced_work_ids) == 1:
            # The candidate's referenced evidence resolves ONE work generation,
            # so its artifact authority comes from THAT generation's events
            # only. Falling back to the model's other generations when the
            # resolved generation has no events re-imposed a FOREIGN
            # generation's custody on the candidate (rung-11 leg 6: a re-drive
            # generation's ``failed:author`` terminal inherited the root
            # generation's published transaction), and the derivation below
            # then refused the record for spanning two work identities. A
            # generation that staged nothing honestly has no artifact
            # authority.
            exact_work_id = next(iter(referenced_work_ids))
            events = [event for event in events if event.get("work_id") == exact_work_id]
        if not events:
            candidate["artifact_authority"] = ArtifactAuthorityRecord.not_applicable().to_payload()
        else:
            latest = events[-1]
            candidate["artifact_authority"] = ArtifactAuthorityRecord.from_event(
                latest
            ).to_payload()
        artifact_inputs = self._artifact_authority_inputs(candidate)
        work_id = self._model_work_id(candidate, artifact_inputs)
        meaningful_modes = tuple(
            str(value) for value in candidate.get("modes", {}).get("meaningful_modes", ())
        )
        relevant = tuple(
            attempt
            for attempt in self._attempts.iter_records()
            if attempt.get("stable_id") == stable_id and attempt.get("work_id") == work_id
        )
        per_mode = derive_per_mode_run(
            relevant,
            stable_id=stable_id,
            work_id=work_id,
            meaningful_modes=meaningful_modes,
        )
        modes = candidate.get("modes")
        if not isinstance(modes, dict):
            raise ReductionError("model modes must be mutable during authority projection")
        modes["per_mode_run"] = per_mode
        by_mode = {
            str(attempt.get("mode")): attempt
            for attempt in relevant
            if attempt.get("attempt_id") in {value["attempt_id"] for value in per_mode.values()}
            and attempt.get("result") == "succeeded"
        }
        try:
            modes.update(
                mode_summary_projection(
                    derive_mode_summary(by_mode.get("train"), by_mode.get("eval"))
                )
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        if candidate.get("status", {}).get("kind") != "runs":
            candidate["observed"] = derive_terminal_observation(
                relevant,
                stable_id=stable_id,
                work_id=work_id,
            )
        terminal_proof, family_authority, vector, _ = self._derive_model_authority(candidate)
        candidate["family_authority"] = family_authority_projection(family_authority)
        candidate["dependency_vector"] = vector
        status = candidate.get("status")
        if not isinstance(status, dict):
            raise ReductionError("model status must be mutable during authority projection")
        status["code"] = terminal_proof.status_code
        status["kind"] = terminal_proof.status_code.split(":", 1)[0]
        if terminal_proof.status_code.startswith("failed:"):
            status["stage"] = terminal_proof.failure_stage
            status["reason_code"] = terminal_proof.reason_code
            status["root_cause_fingerprint"] = terminal_proof.root_cause_fingerprint
        return candidate

    def selected_artifact_transaction(self, stable_id: str, work_id: str) -> Optional[str]:
        """Return the ONE authoritative staged transaction for a work generation.

        The latest ``staged-private`` root in ledger order is the work
        generation's author authority; every earlier staged root it superseded
        is history. Replay-derived from ledger bytes with zero producer input.

        Parameters
        ----------
        stable_id, work_id:
            Exact work-generation identity.

        Returns
        -------
        str | None
            Transaction id of the latest staged root, or ``None`` when the
            generation has no staged root at all.
        """

        selected: Optional[str] = None
        for event in self._artifacts.iter_events():
            if (
                event.get("event_kind") == ArtifactEventKind.STAGED_PRIVATE.value
                and str(event.get("stable_id")) == str(stable_id)
                and str(event.get("work_id")) == str(work_id)
            ):
                selected = str(event.get("transaction_id"))
        return selected

    def authorize_publication(
        self,
        model: Mapping[str, Any],
        staged: StagedArtifact,
        accepted_gate_item: Mapping[str, Any],
        decisions: Mapping[Any, LicenseDecision],
        mirrors: MirrorStore,
        *,
        terminal: bool = False,
    ) -> PublicationAuthorization:
        """Derive and append the sole artifact publication capability.

        Parameters
        ----------
        model:
            Structurally complete candidate whose proof/vector can be replayed.
        staged:
            Exact private-custody transaction.
        accepted_gate_item:
            Independently accepted v3 gate item.
        decisions:
            Exact claim-keyed license decisions.
        mirrors:
            Separated physical mirror store.
        terminal:
            Whether the gate authorizes a terminal private/public commitment.

        Returns
        -------
        PublicationAuthorization
            Frozen capability already committed to the artifact ledger.
        """

        # One reducer-selected transaction pointer (AW-2, the m10949 class):
        # two competing staged authorings for one work generation meant two
        # competing selections of author authority at the publication seam,
        # and the loser terminalized the model. The LATEST staged-private root
        # for this work generation is the single authority; authorizing any
        # earlier one refuses typed here, before any event is appended.
        selected_transaction = self.selected_artifact_transaction(
            str(staged.event["stable_id"]), str(staged.event["work_id"])
        )
        if selected_transaction is not None and str(staged.transaction_id) != (
            selected_transaction
        ):
            raise ReductionError(
                "publication transaction is not the reducer-selected authority "
                f"for {staged.event['stable_id']}/{staged.event['work_id']}: "
                f"staged={staged.transaction_id} selected={selected_transaction}"
            )
        prepared = self.prepare_model(model)
        raw_vector = dict(prepared["dependency_vector"])
        raw_vector["accepted_attempt_ids"] = tuple(raw_vector["accepted_attempt_ids"])
        raw_vector["artifact_claim_ids"] = ()
        provisional = DependencyVector(**raw_vector)
        gate_item_sha256 = stable_hash(accepted_gate_item)
        try:
            owning_gate, ledger_gate_item = resolve_exact_gate_item_membership(
                tuple(self._gates.iter_records()),
                accepted_gate_item=accepted_gate_item,
                accepted_gate_item_sha256=gate_item_sha256,
            )
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        gate_id = str(owning_gate["gate_id"])
        # Write-side coherence tripwire (rung-12 leg 4, m1359): the checkpoint
        # read path requires the reconstruction vector's checker_gate_identity
        # to equal the final event's gate_id, so minting an authorization whose
        # accepted gate differs from the vector's decisive gate poisons the
        # ledger -- the next projection consult quarantines the transaction.
        # A fidelity-failed terminal authorized under its accurate metadata
        # gate produced exactly that pair. Refuse HERE, per model and loudly,
        # before any event is appended.
        #
        # KNOWN CARVE-OUT (family variants, pre-existing): a variant's vector
        # inherits the REPRESENTATIVE's accuracy gate while its publication is
        # accepted under the gate owning the variant's own item, so every
        # published variant has always carried this pair. Refusing it here
        # would break the variant lane outright; the read side now consumes
        # such entries as loud quarantined cache misses instead of crashing.
        # Making the variant write coherent (vector or authorization gate
        # choice) is a family-lane design decision for the census, not a
        # tripwire's.
        identity = model.get("identity") if isinstance(model, Mapping) else None
        representative = (
            identity.get("family_representative_id") if isinstance(identity, Mapping) else None
        )
        is_family_variant = isinstance(representative, str) and representative not in (
            "",
            str(model.get("stable_id")),
        )
        # SENTINEL CARVE-OUT (blocked/terminal lane, pre-existing): a decisive
        # proof with no checker gate derives the typed NOT_APPLICABLE /
        # PENDING_UNTRUSTED sentinel on this axis, while the terminal artifact
        # is authorized under the disposition gate that accepted the arm --
        # every historical blocked-terminal transaction carries exactly this
        # pair. A sentinel asserts "no gate in the proof", not a contradictory
        # gate, so only a CONCRETE vector gate differing from the accepted
        # gate refuses (the m1359 shape: fidelity gate vs metadata gate).
        vector_gate = provisional.checker_gate_identity
        vector_gate_value = (
            vector_gate.value if isinstance(vector_gate, DependencyState) else vector_gate
        )
        vector_gate_is_concrete = vector_gate_value not in {
            DependencyState.NOT_APPLICABLE.value,
            DependencyState.PENDING_UNTRUSTED.value,
        }
        if vector_gate_value != gate_id and vector_gate_is_concrete and not is_family_variant:
            raise ReductionError(
                "publication authorization gate contradicts the model's dependency "
                f"vector at checker_gate_identity: vector={vector_gate_value!r} "
                f"accepted_gate={gate_id!r}"
            )
        authorization_id = derive_publication_authorization_id(
            staged,
            accepted_gate_id=gate_id,
            accepted_gate_item_sha256=gate_item_sha256,
            dependency_vector=provisional,
            decisions=decisions,
            publication_policy_identity=self.context.publication_policy_identity,
        )
        claims = derive_artifact_claims(
            staged,
            accepted_gate_id=gate_id,
            authorization_id=authorization_id,
            decisions=decisions,
            mirrors=mirrors,
        )
        final_vector = replace(
            provisional,
            artifact_claim_ids=tuple(claim.claim_id for claim in claims),
        )
        public_object_ids = tuple(
            sorted(
                {
                    claim.object_id
                    for claim in claims
                    if claim.license_disposition == RedistributionClass.PUBLIC_OK.value
                },
                key=str,
            )
        )
        private_object_ids = tuple(
            sorted(
                {
                    claim.object_id
                    for claim in claims
                    if claim.license_disposition != RedistributionClass.PUBLIC_OK.value
                },
                key=str,
            )
        )
        authorization = PublicationAuthorization(
            authorization_id=authorization_id,
            stable_id=str(staged.event["stable_id"]),
            work_id=str(staged.event["work_id"]),
            transaction_id=staged.transaction_id,
            accepted_gate_id=gate_id,
            accepted_gate_item_sha256=gate_item_sha256,
            dependency_vector=final_vector,
            claim_ids=tuple(claim.claim_id for claim in claims),
            public_object_ids=public_object_ids,
            private_object_ids=private_object_ids,
            publication_policy_identity=self.context.publication_policy_identity,
        )
        append_artifact_authorization(
            staged,
            authorization,
            claims,
            accepted_gate_item=ledger_gate_item,
            event_kind=(
                ArtifactEventKind.TERMINAL_AUTHORIZED
                if terminal
                else ArtifactEventKind.PUBLICATION_AUTHORIZED
            ),
            context=self.context,
            mirrors=mirrors,
            ledger=self._artifacts,
        )
        self._projection_cache = None
        return authorization

    def award_blocked_supersession(self, candidate: Mapping[str, Any]) -> Optional[JsonObject]:
        """Return the ``runs`` head a candidate cannot supersede without authority.

        The exact predicate of the T2-08 award tripwire in
        :meth:`append_model`, exposed so the driver can plan its park (and the
        deliberate scar receipt when a naming grant is consumed) BEFORE
        attempting the append, exactly as :meth:`grant_blocked_supersession`
        exposes the review-hold predicate. The ledger HEAD is consulted --
        never the dependency-current projection -- because a stale award is
        still an award: the 124-second wipe superseded 11 released,
        projection-stale ``runs`` revisions.

        AW-1 release rule: at live admission, ONLY an introduced durable grant
        whose row names the exact blocked award revision
        (``blocked_award_revision``) releases the downgrade. A candidate-
        carried scar is producer input and releases nothing live -- resume-4
        used exactly that arm to destroy three verified awards -- but stays
        honored on append-only REPLAY, where the scarred row is authenticated
        ledger history (the m5273 chain must keep projecting).

        Parameters
        ----------
        candidate:
            Proposed model revision.

        Returns
        -------
        dict[str, Any] | None
            Defensive copy of the governing ``runs`` ledger-head revision when
            the candidate is a non-award terminal carrying no authority to
            supersede it, else ``None``.
        """

        previous = self._current.get(str(candidate.get("stable_id")))
        if previous is None:
            return None
        if previous.get("status", {}).get("kind") != "runs":
            return None
        if candidate.get("status", {}).get("kind") == "runs":
            return None
        previous_revision = str(previous["record_revision"])
        previous_grants = set(previous.get("budget", {}).get("explicit_grants", []))
        candidate_grants = set(candidate.get("budget", {}).get("explicit_grants", []))
        introduced = candidate_grants - previous_grants
        if introduced and self._introduced_grant_names_award(
            introduced, previous_revision, str(candidate.get("stable_id"))
        ):
            return None
        if bool(getattr(self, "_semantic_replay", False)):
            scar = f"{AWARD_SUPERSESSION_SCAR_PREFIX}{previous_revision}"
            if scar in tuple(candidate.get("scar_history", ())):
                return None
        return deepcopy(previous)

    def _introduced_grant_names_award(
        self, introduced: Iterable[str], award_revision: str, stable_id: str
    ) -> bool:
        """Return whether a LIVE introduced grant row names the exact blocked award.

        The decision reads the durable requeue-grants ledger, never candidate
        fields: nothing a producer writes may release the guard. A grant id
        with no row, a row without ``blocked_award_revision``, a row naming a
        different revision, a row issued for a DIFFERENT member, or a row the
        operator has VOIDED, releases nothing. The void read is the 2026-08-16
        red-team fix (break 6/opus #7): selection already treats a voided
        grant as dead (``driver_admission`` skips it), and the revocation must
        hold at the ONE guard that decides award custody too -- otherwise an
        in-flight leg's terminal append supersedes the award on an authority
        the operator withdrew mid-leg.

        Parameters
        ----------
        introduced:
            Grant ids the candidate introduces over the governing head.
        award_revision:
            ``record_revision`` of the governing ``runs`` head.
        stable_id:
            Member the candidate supersedes; the releasing grant must have
            been issued for this exact member.
        """

        grant_rows = _scan_operational_evidence_cached(
            _records_root(self.ledger_paths) / "operational" / "requeue-grants.jsonl",
            validate=False,
        )
        return (
            introduced_award_naming_grant_id(
                grant_rows, introduced, award_revision, stable_id
            )
            is not None
        )

    def grant_blocked_supersession(self, candidate: Mapping[str, Any]) -> Optional[JsonObject]:
        """Return the review-held revision a candidate cannot supersede grantless.

        This is the exact predicate of the review-hold tripwire in
        :meth:`append_model`, exposed so the driver can withhold artifact
        publication BEFORE attempting a grant-blocked append: a pending
        supersession must never publish anything -- grant consumption stays the
        only publish path, and the operator gets to see what a grant would
        publish while it is still private.

        Parameters
        ----------
        candidate:
            Proposed model revision.

        Returns
        -------
        dict[str, Any] | None
            Defensive copy of the governing review-held ledger-head revision
            when the candidate carries no NEW explicit grant over it, else
            ``None``.
        """

        previous = self._current.get(str(candidate.get("stable_id")))
        if previous is None:
            return None
        if not previous.get("status", {}).get("human_review", {}).get("required"):
            return None
        previous_grants = set(previous.get("budget", {}).get("explicit_grants", []))
        candidate_grants = set(candidate.get("budget", {}).get("explicit_grants", []))
        if candidate_grants > previous_grants:
            return None
        if bool(getattr(self, "_semantic_replay", False)) and self._replay_restoration_event(
            candidate
        ):
            # AW-3 replay discharge: an operator award restoration IS the owed
            # review decision, and its authority is recomputed here from the
            # durable decision event -- never from the scar alone (anything a
            # producer writes can be forged; the event binds the exact
            # superseded head this candidate names as its parent).
            return None
        return deepcopy(previous)

    def _live_grant_parks(
        self, stable_id: str, head: Mapping[str, Any]
    ) -> list[JsonObject]:
        """Return LIVE (unvoided, unreleased) park rows naming one raw head.

        Re-derived from the durable park ledger itself with zero caller input
        (anything a producer supplies can be forged), against the RAW ledger
        head ``grant_park_released`` requires (BD-1). Both park dispositions
        live in the one ledger; no disposition filter, so an unknown future
        disposition blocks too (fail-closed: the pending decision always
        wins).

        Parameters
        ----------
        stable_id:
            Model whose head the park rows would name.
        head:
            The model's current raw ledger head.

        Returns
        -------
        list[dict[str, Any]]
            Live park rows naming the head; empty when none do.
        """

        return [
            row
            for row in load_grant_park_rows(
                [grant_park_path(_records_root(self.ledger_paths))]
            )
            if str(row.get("stable_id")) == stable_id
            and not grant_park_released(row, head)
        ]

    def _replay_restoration_event(self, candidate: Mapping[str, Any]) -> bool:
        """Return whether a durable restoration event authorizes this candidate.

        Replay-derived with zero producer input beyond the scar's event-id
        pointer: the operational ledger must hold the
        ``award-restored-by-operator`` event whose id the restoration scar
        names, for this exact stable id, with ``superseding_revision`` equal to
        the candidate's parent -- so a fabricated scar over a different chain
        releases nothing.

        Parameters
        ----------
        candidate:
            Replayed model revision carrying a restoration scar.
        """

        stable_id = str(candidate.get("stable_id"))
        parent_revision = candidate.get("parent_revision")
        pointers = [
            str(scar)[len(AWARD_RESTORATION_SCAR_PREFIX) :]
            for scar in tuple(candidate.get("scar_history", ()))
            if str(scar).startswith(AWARD_RESTORATION_SCAR_PREFIX)
        ]
        if not pointers:
            return False
        events = _scan_operational_evidence_cached(
            _records_root(self.ledger_paths) / "operational" / "events.jsonl"
        )
        for event in events:
            details = event.get("details", {})
            if (
                str(event.get("event_id")) in pointers
                and isinstance(details, Mapping)
                and details.get("disposition") == AWARD_RESTORATION_EVENT_DISPOSITION
                and details.get("stable_id") == stable_id
                and details.get("superseding_revision") == parent_revision
            ):
                return True
        return False

    def append_award_restoration(
        self,
        *,
        stable_id: str,
        restored_award_revision: str,
        superseding_revision: str,
        restoration_event_id: str,
        dry_run: bool = False,
    ) -> JsonObject:
        """Restore one operator-adjudicated eaten award, append-only (AW-3).

        Every acceptance condition is proven from ledger bytes alone -- no
        producer-supplied field steers the decision:

        1. the restoration target is a real ``runs`` ANCESTOR of the current
           head, reachable through the recorded parent chain;
        2. the superseding revision is the CURRENT head and carries the typed
           ``award-superseded-without-grant:<target>`` scar (a scarless
           downgrade is indistinguishable from a legitimate one and never
           qualifies -- the condition is deliberately not widenable);
        3. the adverse child is EVIDENCE-DISJOINT from the target: it cites
           none of the target's artifact transaction, author result, recipe,
           accepted attempts, or gate items (different work ids alone are
           cheap to mint and can describe the same defective artifact), and
           no authenticated soundness revocation covers the target;
        4. the candidate equals the target exactly except parentage and the
           typed restoration scar; the target's grant set must equal the
           head's (a grant-consuming eater is a different adjudication shape
           and refuses into operator review rather than restoring);
        5. the dependency vector replays at the RECORDED basis (through the
           registered-rotation carve-out, so an unregistered or soundness
           rotation blocks the restoration too) and every record-anchored
           evidence cross-check re-passes live.

        The one deliberate discharge: a review-held superseding head normally
        demands a grant to be superseded, and this operator restoration IS the
        owed decision -- the replay accepts exactly and only the refusal that
        says "blocked SOLELY by the missing grant" after everything else
        validated.

        Parameters
        ----------
        stable_id:
            Model whose eaten award is restored.
        restored_award_revision:
            Exact ``record_revision`` of the runs ancestor to restore.
        superseding_revision:
            Exact current-head revision that superseded the award.
        restoration_event_id:
            Durable ``award-restored-by-operator`` event id (already appended
            by the operator tool; verified here except under ``dry_run``).
        dry_run:
            Prove every acceptance condition and report, appending nothing.

        Returns
        -------
        dict[str, Any]
            Restoration report: target, superseding head, the appended (or
            previewed) revision, and ``appended``.

        Raises
        ------
        ReductionError
            If any acceptance condition fails.
        """

        # W3 calibration withhold: restoration is the second runs-award mint
        # path; it refuses before any ledger scan while calibration mode --
        # or the W3 shadow-rehearsal mode -- is active (semantic replay of
        # recorded restorations stays exempt).
        if not bool(getattr(self, "_semantic_replay", False)):
            if calibration_run_active():
                raise CheckerCalibrationWithheldError(
                    f"calibration mode withholds awards: refusing to restore a runs award "
                    f"head for {stable_id!r} while {CHECKER_CALIBRATION_RUN_ENV}=1"
                )
            if shadow_rehearsal_run_active():
                raise CheckerCalibrationWithheldError(
                    f"shadow-rehearsal mode withholds awards: refusing to restore a runs "
                    f"award head for {stable_id!r} while {SHADOW_REHEARSAL_RUN_ENV} is set"
                )
        rows = tuple(self._models.iter_records())
        by_revision = {str(row.get("record_revision")): row for row in rows}
        target = by_revision.get(restored_award_revision)
        if target is None or str(target.get("stable_id")) != stable_id:
            raise ReductionError(
                f"restoration target is not a persisted revision of {stable_id}: "
                f"{restored_award_revision}"
            )
        if target.get("status", {}).get("kind") != "runs":
            # NAMED CONTAINMENT (recon C1): besides its own purpose, this
            # runs-only gate keeps every non-award status shape -- including
            # the W13 disposition-mislabel class -- out of the one replay-mode
            # path that appends without the live door. The restoration replay
            # below additionally declares source_record=None, so even a
            # future non-runs restoration sibling would carry no era
            # authority. Pinned by
            # test_award_restoration_runs_gate_contains_the_era_arm.
            raise ReductionError("restoration target is not a runs award")
        head = self._current.get(stable_id)
        restoration_scar = f"{AWARD_RESTORATION_SCAR_PREFIX}{restoration_event_id}"
        if (
            head is not None
            and head.get("status", {}).get("kind") == "runs"
            and str(head.get("parent_revision")) == superseding_revision
            and restoration_scar in tuple(head.get("scar_history", ()))
        ):
            # Idempotent replay: the identical restoration already governs.
            return {
                "stable_id": stable_id,
                "restored_award_revision": restored_award_revision,
                "superseding_revision": superseding_revision,
                "restoration_event_id": restoration_event_id,
                "restored_status_code": head.get("status", {}).get("code"),
                "dry_run": dry_run,
                "appended": False,
                "restored_head_revision": str(head.get("record_revision")),
            }
        if head is None or str(head.get("record_revision")) != superseding_revision:
            raise ReductionError(
                "superseding revision is not the current head for "
                f"{stable_id}: head={None if head is None else head.get('record_revision')}"
            )
        # (1) ancestor walk over recorded parentage.
        cursor: Optional[Mapping[str, Any]] = head
        ancestors: set[str] = set()
        while cursor is not None:
            parent = cursor.get("parent_revision")
            if not isinstance(parent, str) or parent in ancestors:
                break
            ancestors.add(parent)
            cursor = by_revision.get(parent)
        if restored_award_revision not in ancestors:
            raise ReductionError("restoration target is not an ancestor of the current head")
        # (2) the exact naming scar on the current head.
        scar = f"{AWARD_SUPERSESSION_SCAR_PREFIX}{restored_award_revision}"
        if scar not in tuple(head.get("scar_history", ())):
            raise ReductionError(
                "superseding head carries no award-supersession scar naming the "
                "target; a scarless downgrade never qualifies for restoration"
            )
        # (3) evidence disjointness + no soundness revocation. Sentinel axis
        # values assert ABSENCE, never shared evidence, so they are skipped.
        target_vector = target.get("dependency_vector", {})
        head_vector = head.get("dependency_vector", {})
        sentinels = {
            DependencyState.NOT_APPLICABLE.value,
            DependencyState.PENDING_UNTRUSTED.value,
        }
        for axis in ("artifact_transaction_id", "author_result_identity"):
            target_value = target_vector.get(axis)
            if (
                isinstance(target_value, str)
                and target_value not in sentinels
                and target_value == head_vector.get(axis)
            ):
                raise ReductionError(
                    f"adverse child cites the target's {axis}; its verdict is "
                    "evidence about the awarded artifact, not a later generation"
                )
        target_recipe = target.get("implementation", {}).get("recipe_revision")
        if (
            isinstance(target_recipe, str)
            and target_recipe
            and target_recipe == head.get("implementation", {}).get("recipe_revision")
        ):
            raise ReductionError(
                "adverse child cites the target's recipe; its verdict is evidence "
                "about the awarded artifact, not a later generation"
            )
        target_attempts = set(target.get("execution", {}).get("accepted_attempt_ids", []))
        head_attempts = set(head.get("status", {}).get("attempt_ids", [])) | set(
            head.get("execution", {}).get("accepted_attempt_ids", [])
        )
        if target_attempts & head_attempts:
            raise ReductionError(
                "adverse child cites the target's accepted attempts; its verdict "
                "is evidence about the awarded artifact, not a later generation"
            )
        target_gates = {
            target.get("accuracy_gate", {}).get("gate_id"),
            target.get("fidelity", {}).get("gate_id"),
        } - {None}
        head_gates = {
            head.get("accuracy_gate", {}).get("gate_id"),
            head.get("fidelity", {}).get("gate_id"),
        } - {None}
        if target_gates & head_gates:
            raise ReductionError(
                "adverse child cites the target's gate items; its verdict is "
                "evidence about the awarded artifact, not a later generation"
            )
        operational_rows = _scan_operational_evidence_cached(
            _records_root(self.ledger_paths) / "operational" / "events.jsonl"
        )
        for event in operational_rows:
            details = event.get("details", {})
            if (
                isinstance(details, Mapping)
                and details.get("disposition") == "award-revocation"
                and details.get("award_revision") == restored_award_revision
            ):
                raise ReductionError(
                    "an authenticated award revocation covers the restoration "
                    f"target: {event.get('event_id')}"
                )
        # (4) exact-content candidate; grant sets must agree.
        if set(target.get("budget", {}).get("explicit_grants", [])) != set(
            head.get("budget", {}).get("explicit_grants", [])
        ):
            raise ReductionError(
                "superseding head consumed grants the target lacks; a "
                "grant-consuming eater is adjudicated by the operator, never "
                "auto-restored"
            )
        candidate = {
            key: deepcopy(value)
            for key, value in target.items()
            if key not in {"record_seq", "record_revision", "ledger_seq", "payload_sha256"}
        }
        candidate["parent_revision"] = superseding_revision
        status = candidate.get("status")
        if isinstance(status, dict):
            status["supersedes_revision"] = superseding_revision
        # record_seq is LEDGER-local, not chain-local: the appender derives the
        # next local sequence when the field is omitted, and stamping
        # head.record_seq + 1 here collides with its tripwire the moment any
        # other model appended after the supersession (the live m5273 shape).
        candidate["scar_history"] = [
            *tuple(target.get("scar_history", ())),
            f"{AWARD_RESTORATION_SCAR_PREFIX}{restoration_event_id}",
        ]
        if not dry_run:
            restoration_event = next(
                (
                    event
                    for event in operational_rows
                    if event.get("event_id") == restoration_event_id
                    and event.get("details", {}).get("disposition")
                    == AWARD_RESTORATION_EVENT_DISPOSITION
                    and event.get("details", {}).get("stable_id") == stable_id
                    and event.get("details", {}).get("restored_award_revision")
                    == restored_award_revision
                    # P5-SOL-03/07: the decision event must bind the EXACT
                    # current head it authorizes superseding — the same
                    # equality the semantic-replay rule enforces
                    # (_replay_restoration_event). Without it, an event minted
                    # against a different (or fabricated) head discharged a
                    # restoration over THIS head.
                    and event.get("details", {}).get("superseding_revision")
                    == superseding_revision
                ),
                None,
            )
            if restoration_event is None:
                raise ReductionError(
                    "restoration decision event is absent from the operational "
                    f"ledger (or binds a different superseding head): "
                    f"{restoration_event_id} for {stable_id} over "
                    f"{superseding_revision}"
                )
        # (5) recorded-basis replay with live record-anchored cross-checks.
        attempts = tuple(self._attempts.iter_records())
        gates = tuple(self._gates.iter_records())
        artifacts = tuple(self._artifacts.iter_events())
        attested = _attested_generation_bindings(attempts, operational_rows)
        replay_context = _award_recorded_basis_context(
            self.context,
            candidate,
            attested_bindings=attested,
        )
        replay = _replay_reducer(
            self.ledger_paths,
            context=replay_context,
            prior_model_records=tuple(
                row for row in rows if str(row.get("stable_id")) == stable_id
            ),
            attempt_records=attempts,
            gate_records=gates,
            artifact_records=artifacts,
            raw_prior_current=self._current,
            dependency_current=self._current_projection().current_records,
            attempt_index=self._attempt_index(),
            gate_index=self._gate_index(),
            # Declared NON-residency (recon C1): this replay validates a NOVEL
            # candidate the restoration is about to mint, not a persisted row,
            # so it must carry no era authority. Together with the runs-only
            # target gate above ("restoration target is not a runs award"),
            # this keeps the W13 era arm doubly unreachable from the one
            # replay-mode path that appends without going through the live
            # ``append_model`` door.
            source_record=None,
        )
        _referenced_evidence_repasses(replay, candidate, attempts, gates)
        replay_candidate = {
            key: value
            for key, value in candidate.items()
            if key not in {"record_seq", "record_revision"}
        }
        try:
            replay.append_model(replay_candidate)
        except GrantPendingSupersessionError:
            # Blocked SOLELY by the missing grant after every other validation
            # passed: this operator restoration is exactly the owed decision.
            # P5-SOL-03: the discharge is granted ONLY when the replay's own
            # closed rule independently accepts the durable decision event for
            # THIS candidate (exact stable id, event id, and superseding head
            # as the candidate's parent). The replay raising this error while
            # its rule refuses means the event does not authorize the append
            # it is being offered for — re-raise, never swallow. A dry run is
            # exempt for the one honest reason that its decision event is not
            # yet appended (the operator tool appends event-then-revision);
            # the live path always proves the discharge.
            if not dry_run and not replay._replay_restoration_event(replay_candidate):
                raise
        report: JsonObject = {
            "stable_id": stable_id,
            "restored_award_revision": restored_award_revision,
            "superseding_revision": superseding_revision,
            "restoration_event_id": restoration_event_id,
            "restored_status_code": candidate.get("status", {}).get("code"),
            "dry_run": dry_run,
            "appended": False,
        }
        if dry_run:
            return report
        result = self._models.append(candidate)
        self._current[stable_id] = deepcopy(result.record)
        self._projection_cache = None
        report["appended"] = bool(result.appended)
        report["restored_head_revision"] = str(result.record["record_revision"])
        return report

    def append_model(self, model: Mapping[str, Any]) -> AppendResult:
        """Validate and append a full canonical model revision.

        Parameters
        ----------
        model:
            Complete model revision. The reducer computes its canonical revision hash.

        Returns
        -------
        AppendResult
            Idempotent append result.

        Raises
        ------
        ReductionError
            If parentage, taxonomy, gate, attempt, mode, or partition rules fail.
        """

        return self._append_model_impl(model, preflight=False)

    def preflight_model(self, model: Mapping[str, Any]) -> AppendResult:
        """Run the ENTIRE model-append validation chain as a pre-spend dry run.

        AT-4 (B1-XQ28): every refusal class raises exactly as
        :meth:`append_model` would, but nothing persists and no reducer state
        advances. The award worker runs this BEFORE spending the artifact
        chain, so an append-refusal class (a stale dependency vector, a
        generation-spanning proof) refuses pre-publication instead of leaving
        a spent 4-event chain under a failed terminal (m12646). Deliberately
        a SEPARATE entry point: :meth:`append_model`'s signature, call sites,
        and crash/kill injection semantics stay byte-identical.

        Returns
        -------
        AppendResult
            The validated candidate echoed with ``appended=False``.
        """

        return self._append_model_impl(model, preflight=True)

    def append_orphan_adoption(self, prepared: Mapping[str, Any]) -> AppendResult:
        """Append one orphan-citation adoption re-append through the closed rule.

        P5-SOL-05: this is the ONLY live path that discharges the moved-citation
        carve inside the review-hold grant tripwire. The entire closed adoption
        rule is re-derived HERE, at the write door, exclusively from ledgers
        with zero producer-steered input -- the caller supplies nothing but the
        prepared candidate, and every eligibility fact is recomputed from the
        leg-generation ledger, the models source ledgers, and the grant-park
        ledger (the same readers the driver's redundant first fence uses):

        1. the APPENDING run (the candidate's ``provenance.producer_run_id``)
           durably joined exactly one OPEN leg generation;
        2. the member's current head citation is actually ORPHANED (its cited
           run never joined that generation) and the head sits inside the open
           window;
        3. the head's parent revision resolves to exactly ONE row in the cited
           source ledger, and the head's own citation aligns with the same
           ledger read;
        4. the full closed predicate
           :func:`~menagerie.crawler.terminal_review.orphan_hold_adoption_refusal`
           passes (review-held containment marking over sealed below-floor
           history, byte-verbatim modulo the enumerated rotatable axes);
        5. no LIVE grant park names the head (:meth:`_live_grant_parks`); and
        6. the prepared candidate shows zero post-``prepare_model`` drift
           beyond re-citation
           (:func:`~menagerie.crawler.terminal_review.orphan_adoption_drift`).

        Only then does the append run, internally through the ordinary
        :meth:`append_model` validation chain with the module-private
        admission flag set. Direct ``append_model`` callers -- exactly the
        2026-08-27 out-of-band incident actor class -- can never reach the
        carve, and replay semantics are untouched (replay is write-free and
        never routes through this method).

        Parameters
        ----------
        prepared:
            ``prepare_model`` output for the adoption re-append candidate.

        Returns
        -------
        AppendResult
            Idempotent append result.

        Raises
        ------
        OrphanAdoptionRefusal
            If any arm of the closed adoption rule refuses; one member's
            typed fact, disclosed by the driver census, never a leg crash.
        ReductionError
            If the ordinary append validation chain refuses the candidate.
        """

        # CC D-26 justified (essential closed proof rule): the adoption
        # eligibility ladder is ONE ordered, exhaustively enumerated rule
        # (four-part waiver doctrine, part 2 -- closed sets, exact identity
        # bindings). Every arm must stay visible in a single reviewed block at
        # the write door; extracting arms into helpers would fragment the
        # closed rule the doctrine requires to be reviewable as one, exactly
        # like _admit_model's CC F-42 admission ladder.
        from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: avoids an import cycle at load
            MODELS_SOURCE_NAMES,
            _closed_generation_ids,
            close_ledger_sources,
            generation_run_ids,
            load_leg_open_rows,
            model_head_citations,
        )
        from menagerie.crawler.partition_proof import (  # noqa: PLC0415,PLC2701 -- the proof's own citation reader
            _model_row_run_id,
        )
        from menagerie.crawler.terminal_review import (  # noqa: PLC0415 -- leaf module; avoids an import cycle at load
            orphan_adoption_drift,
            orphan_hold_adoption_refusal,
        )

        candidate = dict(prepared)
        stable_id = str(candidate.get("stable_id"))

        def refuse(reason: str) -> OrphanAdoptionRefusal:
            return OrphanAdoptionRefusal(stable_id=stable_id, reason=reason)

        provenance = candidate.get("provenance")
        adopting_run = (
            provenance.get("producer_run_id") if isinstance(provenance, Mapping) else None
        )
        if not isinstance(adopting_run, str) or not adopting_run:
            raise refuse("adoption candidate carries no producer_run_id citation")
        records_root = _records_root(self.ledger_paths)
        closed_generations = _closed_generation_ids(records_root)
        joined_open = [
            row
            for row in load_leg_open_rows(records_root)
            if str(row["generation_id"]) not in closed_generations
            and adopting_run in generation_run_ids(records_root, str(row["generation_id"]))
        ]
        if len(joined_open) != 1:
            raise refuse(
                f"appending run {adopting_run!r} durably joined {len(joined_open)} open "
                "leg generations; adoption requires exactly one"
            )
        generation = joined_open[0]
        generation_runs = generation_run_ids(records_root, str(generation["generation_id"]))
        floors = {
            name: int(dict(dict(generation["baselines"]).get(name) or {}).get("rows") or 0)
            for name in MODELS_SOURCE_NAMES
        }
        entry = model_head_citations(records_root, require_ledger=False).get(stable_id)
        if entry is None:
            raise refuse("no models-ledger head exists for the member")
        head = dict(entry["record"])
        source = str(entry["source"])
        index = int(entry["row_index"])
        if source not in floors or index < floors[source]:
            raise refuse(
                "head citation is pre-window history; adoption owns only the open window"
            )
        cited_run = _model_row_run_id(entry["record"])
        if cited_run in generation_runs:
            raise refuse(
                f"head citation names run {cited_run!r}, which joined the open "
                "generation; nothing is orphaned"
            )
        rows = scan_jsonl(close_ledger_sources(records_root)[source])
        if not (0 <= index < len(rows)) or rows[index].get("record_revision") != head.get(
            "record_revision"
        ):
            raise refuse(
                "head citation does not align with the source ledger read: "
                "refusing every lookup derived from it"
            )
        parent_matches = [
            (row_index, row)
            for row_index, row in enumerate(rows)
            if row.get("record_revision") == head.get("parent_revision")
        ]
        if len(parent_matches) != 1:
            raise refuse(
                f"parent revision resolves to {len(parent_matches)} rows in the "
                "cited source; adoption requires exactly one"
            )
        parent_index, parent = parent_matches[0]
        refusal = orphan_hold_adoption_refusal(
            head,
            parent,
            parent_row_index=parent_index,
            window_floor=floors[source],
        )
        if refusal is not None:
            raise refuse(refusal)
        raw_head = self._current.get(stable_id)
        if raw_head is None or raw_head.get("record_revision") != head.get("record_revision"):
            raise refuse(
                "reducer head disagrees with the models-ledger citation read: "
                "refusing every lookup derived from it"
            )
        live_parks = self._live_grant_parks(stable_id, raw_head)
        if live_parks:
            dispositions = ", ".join(
                sorted({str(row.get("disposition")) for row in live_parks})
            )
            raise refuse(
                f"head is named by {len(live_parks)} live park row(s) "
                f"({dispositions}): the adoption re-append would advance the "
                "head revision and silently consume the park's pending "
                "operator decision; drain or void the park first"
            )
        drift = orphan_adoption_drift(candidate, head)
        if drift is not None:
            raise refuse(drift)
        # The module-private admission flag is the internal token only this
        # method sets; :meth:`append_model` runs the WHOLE ordinary validation
        # chain (parentage, verbatim carve, park liveness) under it.
        self._orphan_adoption_admission = True
        try:
            return self.append_model(candidate)
        finally:
            self._orphan_adoption_admission = False

    def _append_model_impl(
        self, model: Mapping[str, Any], *, preflight: bool
    ) -> AppendResult:
        candidate = deepcopy(dict(model))
        if candidate.get("schema_version") != MODEL_SCHEMA_VERSION_V3:
            raise ReductionError("model appends require the current v3 authority contract")
        _validate_c07_diagnostic_fields(candidate, model=True)
        stable_id = candidate.get("stable_id")
        if stable_id not in self.intake_ids:
            raise ReductionError(f"model stable_id is outside intake: {stable_id}")
        semantic_replay = bool(getattr(self, "_semantic_replay", False))
        # W3 write-side producer (model door): stamp BEFORE the idempotent
        # duplicate comparison in _admit_model, so an in-mode re-drive of an
        # already-marked revision still compares equal instead of failing
        # parentage. Live only -- semantic replay re-admits history exactly
        # as recorded and never stamps. Preflight stamps for exact parity
        # with the append it predicts (nothing persists on that path).
        if not semantic_replay:
            candidate = stamp_shadow_rehearsal(candidate)
        projection_before: Optional[DependencyCurrencyProjection] = None
        if not semantic_replay:
            projection_before = self._current_projection()
            self._validation_current = MappingProxyType(dict(projection_before.current_records))
        try:
            return self._admit_model(
                candidate, stable_id, semantic_replay, projection_before, preflight=preflight
            )
        finally:
            # The pre-append projection pin is scoped to THIS admission pass.
            # EVERY exit -- a refusal anywhere in the validation chain, the
            # parkable grant block, the idempotent duplicate early return --
            # must release it: a dangling pin serves a stale dependency view to
            # prepare_model's family lookups until the next append happens to
            # reassign it. Replay reducers keep their construction-time pin.
            if not semantic_replay:
                self.__dict__.pop("_validation_current", None)

    def _admit_model(
        self,
        candidate: JsonObject,
        stable_id: Any,
        semantic_replay: bool,
        projection_before: Optional[DependencyCurrencyProjection],
        *,
        preflight: bool = False,
    ) -> AppendResult:
        """Run one model admission pass under the caller-scoped projection pin.

        Parameters
        ----------
        candidate:
            Deep-copied, header-validated candidate revision.
        stable_id:
            Candidate stable identity already checked against the active intake.
        semantic_replay:
            Whether this reducer replays append-only history.
        projection_before:
            Projection cached immediately before this pass (live mode only).

        Returns
        -------
        AppendResult
            Idempotent append result.
        """

        for existing in self._models.iter_records():
            if existing["stable_id"] != stable_id:
                continue
            logical_existing = {
                key: value
                for key, value in existing.items()
                if key not in {"record_seq", "record_revision"}
            }
            logical_candidate = {
                key: value
                for key, value in candidate.items()
                if key not in {"record_seq", "record_revision"}
            }
            if logical_existing == logical_candidate:
                return AppendResult(deepcopy(existing), appended=False)
        # W3 calibration withhold: a LIVE award issuance (or release-eligible
        # marking) refuses typed while the sanctioned calibration capture run
        # is active. It runs after the idempotent-duplicate return (re-append
        # of issued bytes is re-admission, not issuance) and before any deeper
        # validation, so a calibration leg can never bank an adjudication.
        if not semantic_replay:
            _withhold_award_issuance_in_calibration_mode(candidate)
        previous = self._current.get(str(stable_id))
        expected_parent = previous["record_revision"] if previous is not None else None
        if (
            previous is not None
            and candidate.get("parent_revision") is None
            and not semantic_replay
            and not self.has_current(str(stable_id))
        ):
            candidate["parent_revision"] = expected_parent
            status = candidate.get("status")
            if isinstance(status, dict):
                status["supersedes_revision"] = expected_parent
        if candidate.get("parent_revision") != expected_parent:
            raise ReductionError(
                f"bad parentage for {stable_id}: expected {expected_parent!r}, "
                f"received {candidate.get('parent_revision')!r}"
            )
        supersedes_revision = candidate.get("status", {}).get("supersedes_revision")
        if previous is None and supersedes_revision is not None:
            raise ReductionError("a first model revision cannot supersede another revision")
        if previous is not None and supersedes_revision != expected_parent:
            raise ReductionError("status.supersedes_revision must exactly match parent_revision")
        if previous is None and candidate.get("record_seq") not in {None, 1}:
            raise ReductionError("a first model revision must start at record_seq 1 or omit it")
        if previous is not None and candidate.get("record_seq") is not None:
            if int(candidate["record_seq"]) <= int(previous["record_seq"]):
                raise ReductionError("record_seq must increase over the current revision")
        if not semantic_replay:
            _validate_persisted_requeue_lineage(
                (*self._models.iter_records(), candidate),
                self.ledger_paths,
                self._attempts.iter_records(),
                self._gates.iter_records(),
                self._artifacts.iter_events(),
            )
        self._validate_status(candidate)
        self._validate_source(candidate)
        self._validate_derived_model_authority(candidate)
        self._validate_family_template(candidate)
        self._validate_gates(candidate)
        self._validate_execution(candidate)
        self._validate_completeness(candidate)
        # The T2-08 award guard: a `runs` revision is NEVER superseded by a
        # non-award terminal without a NEW explicit grant or the typed
        # award-supersession scar. One grant-free EnvironmentExactnessError
        # wiped 11 released awards in 124 seconds with zero events; this
        # tripwire makes that append refuse loudly. It runs before the
        # review-hold tripwire so the parkable grant signal below keeps its
        # "blocked SOLELY by the missing grant" contract.
        if self.award_blocked_supersession(candidate) is not None:
            assert previous is not None
            raise AwardSupersessionError(
                stable_id=str(stable_id),
                blocked_parent_revision=str(expected_parent),
            )
        # The review-hold grant tripwire runs LAST deliberately: raising its
        # typed refusal only after every other validation passed is what makes
        # the refusal parkable -- a caller catching it knows the append is
        # blocked SOLELY by the missing grant, so the grant-pending park can
        # never mask a defective candidate (any other defect already raised the
        # plain ReductionError above).
        if self.grant_blocked_supersession(candidate) is not None:
            assert previous is not None
            # Orphan-citation adoption carve-out (2026-08-27): a candidate that
            # is the held head BYTE-VERBATIM modulo ledger parentage, the
            # appending run's citation, and the enumerated rotatable
            # identity axes is the SAME operator property -- the hold block,
            # status code, stage, and every recorded fact survive unchanged,
            # so no grant is consumed and nothing an operator owns is
            # rewritten.  The citation must actually MOVE (Opus C2a): a
            # citation-UNCHANGED verbatim re-append is a no-op supersession
            # that would silently consume a live grant-pending park
            # (`grant_park_released` reads any revision advance as "the
            # granted supersession landed"), so it keeps refusing on this
            # tripwire.  Anything differing beyond the closed re-citation
            # domain keeps refusing here exactly as before.  The verbatim
            # shape below is a NECESSARY condition only: a LIVE append
            # additionally requires the append to arrive through
            # :meth:`append_orphan_adoption`, which re-derives the full
            # closed adoption rule from ledgers first (P5-SOL-05); semantic
            # replay keeps admitting lawfully adopted persisted rows on this
            # shape alone (era honesty).
            from menagerie.crawler.terminal_review import (  # noqa: PLC0415 -- leaf module; avoids an import cycle at load
                orphan_adoption_drift,
            )

            candidate_provenance = candidate.get("provenance")
            previous_provenance = previous.get("provenance")
            candidate_run = (
                candidate_provenance.get("producer_run_id")
                if isinstance(candidate_provenance, Mapping)
                else None
            )
            previous_run = (
                previous_provenance.get("producer_run_id")
                if isinstance(previous_provenance, Mapping)
                else None
            )
            recites_verbatim = (
                isinstance(candidate_run, str)
                and candidate_run != previous_run
                and orphan_adoption_drift(candidate, previous) is None
                and candidate.get("status", {}).get("supersedes_revision")
                == previous.get("record_revision")
            )
            if not recites_verbatim:
                raise GrantPendingSupersessionError(
                    stable_id=str(stable_id),
                    blocked_parent_revision=str(previous["record_revision"]),
                    blocked_stage=previous.get("status", {}).get("stage"),
                )
            # LIVE appends never discharge the carve directly (P5-SOL-05,
            # round 4): a moved citation is not evidence that the closed
            # adoption rule ran -- Sol's probe appended a drift-clean
            # moved-citation candidate whose PARENT sat inside the open
            # window, exactly the shape ``orphan_hold_adoption_refusal``
            # exists to refuse, and the old grant-free carve admitted it.
            # The only live path through this clause is
            # :meth:`append_orphan_adoption`, which re-derives the ENTIRE
            # closed adoption rule (joined appending run, actually-orphaned
            # citation, unique below-floor parent, hold-marking equality,
            # live-park liveness, post-prepare drift) exclusively from
            # ledgers at the write door and then sets the module-private
            # admission flag around the internal re-append. Any OTHER direct
            # ``append_model`` caller -- exactly the 2026-08-27 out-of-band
            # incident actor class -- keeps meeting this typed refusal.
            # Replay mode keeps the historical carve unchanged (era honesty):
            # a replay reducer is write-free -- nothing it admits advances
            # the durable head, so no park can be consumed through it -- and
            # its mid-history ``previous`` is not the raw ledger head that
            # ``grant_park_released`` requires (BD-1), so a lawfully adopted
            # persisted row keeps replaying at its recorded basis.
            # CC F-42 justified (essential state machine): the admission
            # ladder's clause ORDER is the parkability contract -- this
            # tripwire must run exactly here, after the verbatim carve and
            # before persist; splitting the ladder would sever that order.
            if not semantic_replay:
                if not bool(getattr(self, "_orphan_adoption_admission", False)):
                    raise GrantPendingSupersessionError(
                        stable_id=str(stable_id),
                        blocked_parent_revision=str(previous["record_revision"]),
                        blocked_stage=previous.get("status", {}).get("stage"),
                    )
                # Live-park non-interference AT THE APPEND DOOR (Sol C2,
                # round 3), retained inside the adoption path: any LIVE
                # (unvoided, unreleased) park row naming this head records a
                # pending operator decision, and this re-append would advance
                # the head revision -- ``grant_park_released`` reads any
                # revision advance as "the granted supersession landed" --
                # silently consuming that decision without a grant.
                # ``_live_grant_parks`` re-derives liveness from the park
                # ledger itself with zero caller input; the driver-side check
                # stays as the first fence, redundant BY DESIGN like the era
                # arm's 4633 containment.
                if self._live_grant_parks(str(stable_id), previous):
                    raise GrantPendingSupersessionError(
                        stable_id=str(stable_id),
                        blocked_parent_revision=str(previous["record_revision"]),
                        blocked_stage=previous.get("status", {}).get("stage"),
                    )
        if preflight:
            # AT-4 pre-spend dry run: every validation above ran and passed;
            # nothing is persisted and no reducer state advances.
            return AppendResult(record=candidate, appended=False)
        try:
            result = self._models.append(candidate)
        except LedgerConflictError as exc:
            raise ReductionError(str(exc)) from exc
        self._current[str(stable_id)] = deepcopy(result.record)
        if not semantic_replay:
            if projection_before is None:
                self._projection_cache = None
            else:
                self._projection_cache = (
                    self._projection_token(),
                    self._patched_projection_after_append(projection_before, result.record),
                )
        else:
            self._projection_cache = None
        return result

    def _patched_projection_after_append(
        self,
        prior_projection: DependencyCurrencyProjection,
        record: Mapping[str, Any],
    ) -> DependencyCurrencyProjection:
        """Advance the cached projection over exactly one appended revision.

        The appended row is the ledger tail, so a fresh
        :func:`project_dependency_current` replays every other current record
        with UNCHANGED inputs and replays this record LAST -- the incremental
        step is therefore exact if it does what the fresh final iteration
        does: pop the family variants bound to this stable ID (in fresh
        revision order they now replay before their representative and fail
        its currency lookup), then REPLAY the appended record's currency
        through the same replay reducer, evidence repasses, and family check,
        never asserting it current without proof (T1-15: the previous patch
        stamped the new revision dependency-current with no replay).

        Parameters
        ----------
        prior_projection:
            Projection cached immediately before the append.
        record:
            The persisted just-appended model revision.

        Returns
        -------
        DependencyCurrencyProjection
            Projection equal in current/stale membership to a fresh
            projection over the post-append ledgers.
        """

        sid = str(record.get("stable_id"))
        projected = dict(prior_projection.current_records)
        stale = dict(prior_projection.stale_reasons)
        for variant_id, (
            representative_id,
            _variant_token,
        ) in self.intake_variant_bindings.items():
            if representative_id != sid or variant_id == sid:
                continue
            # Only a variant that HOLDS a current revision changes disposition;
            # a variant with no revision (or one already stale) has no entry to
            # move -- a fresh projection only carries dispositions for stable
            # IDs present in raw current.
            if variant_id not in projected:
                continue
            projected.pop(variant_id, None)
            stale[variant_id] = "variant lacks its exact current representative record"
        projected.pop(sid, None)
        stale.pop(sid, None)
        try:
            self._replay_appended_currency(record, projected)
        except (KeyError, ReductionError, TypeError, ValueError) as exc:
            stale[sid] = str(exc)
        else:
            projected[sid] = deepcopy(dict(record))
        return DependencyCurrencyProjection(projected, stale)

    def _replay_appended_currency(
        self,
        record: Mapping[str, Any],
        dependency_current: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Replay one appended tail revision's currency against live evidence.

        Mirrors the final loop iteration of
        :func:`project_dependency_current` for the record at the ledger tail:
        an in-memory replay reducer positioned immediately before the
        candidate, full referenced-evidence repasses, the replayed admission
        itself, and the family-variant currency check.

        Parameters
        ----------
        record:
            The persisted just-appended model revision (the models tail row).
        dependency_current:
            Currency of every OTHER record exactly as a fresh projection
            holds it when the tail record replays.

        Raises
        ------
        ReductionError
            If the appended revision fails the same fresh-projection replay
            that would mark it stale (KeyError/TypeError/ValueError from the
            replay surface propagate identically).
        """

        sid = str(record.get("stable_id"))
        revision = record.get("record_revision")
        record_seq = int(record.get("record_seq", 0))
        prior_rows = tuple(
            row
            for row in self._models.records_for_stable_id(sid)
            if row.get("record_revision") != revision
            and int(row.get("record_seq", 0)) < record_seq
        )
        dependency_at_record = {
            stable_id: value
            for stable_id, value in dependency_current.items()
            if int(value.get("record_seq", 0)) < record_seq
        }
        raw_prior_current = {
            stable_id: deepcopy(dict(value))
            for stable_id, value in self._current.items()
            if stable_id != sid and int(value.get("record_seq", 0)) < record_seq
        }
        if prior_rows:
            raw_prior_current[sid] = deepcopy(dict(prior_rows[-1]))
        attempts_by_id: dict[str, Mapping[str, Any]] = {
            str(value.get("attempt_id")): value
            for value in self._attempts.records_for_stable_id(sid)
        }
        pending_attempt_ids = list(_record_attempt_ids(record))
        while pending_attempt_ids:
            attempt_id = str(pending_attempt_ids.pop())
            if attempt_id in attempts_by_id:
                attempt = attempts_by_id[attempt_id]
            else:
                matches = self._attempts.records_for_field("attempt_id", attempt_id)
                if not matches:
                    continue
                attempt = matches[-1]
                attempts_by_id[attempt_id] = attempt
            defer_evidence = attempt.get("defer_evidence")
            if isinstance(defer_evidence, Mapping):
                pending_attempt_ids.extend(
                    str(value)
                    for value in defer_evidence.get("probe_attempt_ids", ())
                    if str(value) not in attempts_by_id
                )
        attempts = tuple(
            sorted(attempts_by_id.values(), key=lambda value: int(value.get("ledger_seq", 0)))
        )
        gates_by_id: dict[str, Mapping[str, Any]] = {
            str(value.get("gate_id")): value
            for value in self._gates.records_for_stable_id(sid)
        }
        for gate_id in (
            record.get("accuracy_gate", {}).get("gate_id"),
            record.get("fidelity", {}).get("gate_id"),
        ):
            if gate_id is None or str(gate_id) in gates_by_id:
                continue
            matches = self._gates.records_for_field("gate_id", gate_id)
            if matches:
                gates_by_id[str(gate_id)] = matches[-1]
        gates = tuple(
            sorted(gates_by_id.values(), key=lambda value: int(value.get("ledger_seq", 0)))
        )
        artifacts = self._artifacts.events_for_stable_id(sid)
        if _record_requires_human_review(record):
            replay_context = _review_held_recorded_basis_context(self.context, record)
        else:
            # EVERY settled head replays at its recorded basis, exactly as the
            # full derivation does (the leg-29 extension in
            # :func:`project_dependency_current`) -- this fast path and the
            # full projection adjudicate the SAME ledger state and must be
            # observationally identical. While this arm pinned only ``runs``
            # awards, a mid-leg evidence append (a terminal-review sweep gate
            # row) re-replayed the swept settled terminals under LIVE
            # identities: each flipped transiently stale with the
            # derivation-contradiction reason, and the D-47 stale-excluded
            # census digest flapped between consecutive launches (the
            # 2026-08-27 calibration 26/31 flap). Attested generation bindings
            # read the FULL attempts ledger for the same parity reason: the
            # full projection attests a recorded generation from ANY member's
            # evidence, never just the replayed member's own rows.
            records_root = _records_root(self.ledger_paths)
            replay_context = _award_recorded_basis_context(
                self.context,
                record,
                attested_bindings=_attested_generation_bindings(
                    tuple(self._attempts.iter_records()),
                    _scan_operational_evidence_cached(
                        records_root / "operational" / "events.jsonl"
                    ),
                ),
            )
        replay = _replay_reducer(
            self.ledger_paths,
            context=replay_context,
            prior_model_records=prior_rows,
            attempt_records=attempts,
            gate_records=gates,
            artifact_records=artifacts,
            raw_prior_current=raw_prior_current,
            dependency_current=dependency_at_record,
            attempt_index=attempts_by_id,
            gate_index=gates_by_id,
            source_record=record,
        )
        _referenced_evidence_repasses(replay, record, attempts, gates)
        replay_candidate = {
            key: value
            for key, value in record.items()
            if key not in {"record_seq", "record_revision"}
        }
        replay.append_model(replay_candidate)
        family_error = family_variant_currency_error(record, dependency_at_record)
        if family_error is not None:
            raise ReductionError(family_error)

    def _validate_status(self, model: Mapping[str, Any]) -> None:
        """Validate closed terminal code/kind/stage/reason relationships.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        status = model.get("status", {})
        code = status.get("code")
        kind = status.get("kind")
        if code not in TERMINAL_STATUS_CODES:
            raise ReductionError(f"unknown terminal status code: {code!r}")
        expected_kind = str(code).split(":", 1)[0]
        if kind != expected_kind:
            raise ReductionError(f"status kind {kind!r} does not match code {code!r}")
        stage = status.get("stage")
        reason = status.get("reason_code")
        if kind == "failed":
            expected_stage = str(code).split(":", 1)[1]
            if stage != expected_stage:
                raise ReductionError("failure stage must match the failed status code")
            if reason not in FAILURE_REASON_CODES[expected_stage]:
                raise ReductionError(
                    f"reason {reason!r} is not allowed for failure stage {expected_stage!r}"
                )
            if status.get("traceback") is None and status.get("no_traceback_reason") is None:
                raise ReductionError("failure requires a traceback or explicit no_traceback_reason")
            if code == "failed:accuracy-gate" and not status.get("human_review", {}).get(
                "required"
            ):
                raise ReductionError("failed:accuracy-gate requires human review")
            if code == "failed:accuracy-gate" and reason in BLOCKED_DISPOSITION_REVIEW_REASON_CODES:
                # W13: a disposition rejection is not an accuracy-gate failure --
                # the accuracy gate never rendered a verdict for it (R24: the
                # 200a heads with accuracy_gate.verdict=None). A FRESH claim of
                # this shape must mint blocked:disposition-review. Era honesty
                # (PD-13, forward-only) admits exactly two NON-fresh passages:
                #
                # (1) a live re-append whose current head already carries this
                #     exact code+reason (the review sweep and the containment
                #     ladder re-append a historical head under its ORIGINAL
                #     code) -- so a resident row of the shape is historical OR a
                #     continuity descendant of one;
                # (2) ERA REPLAY of frozen realized history, decided by
                #     ``_era_replay_of_frozen_history``: semantic replay AND the
                #     threaded persisted source row's RECOMPUTED bytes digest is
                #     a member of the frozen, reviewed
                #     ``DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS`` cohort AND
                #     ``accuracy_gate.verdict is None`` (a rendered verdict
                #     beside a disposition reason is a CONTRADICTION and refuses
                #     as one, never as an era artifact). Replay mode alone is
                #     caller identity, not era authority: a replay reducer also
                #     validates novel candidates (award restoration), so the
                #     residency fact is threaded explicitly from the ledger read
                #     and the digest binds the exact historical bytes.
                #
                # Ledger-tamper scope (in-repo statement, recon C3): an actor
                # with direct filesystem write access to the parent-owned models
                # ledger can append a self-hashed, honest-sequence row of any
                # currently-legal shape -- that power precedes and exceeds this
                # arm. What this arm's frozen digest set removes is the era
                # DIFFERENTIAL: such an actor can no longer mint a NEW row that
                # vouches under historical law, because membership is a
                # recomputed content hash over exact realized bytes
                # (record_seq included), never a shape, an id, or anything the
                # row says about itself. A byte-exact copy of a frozen member
                # only re-lands at its own sequence position (the append-only
                # conflict rules refuse it elsewhere). The remaining boundary is
                # coherent operator-level reauthoring of the whole prefix --
                # already the documented out-of-scope threat everywhere else in
                # this module. Widening the cohort is a reviewed change to the
                # frozen set in constants.py, per the four-part doctrine's
                # exact-identity-bindings clause.
                #
                # The one replay-mode path that appends without the live door,
                # ``append_award_restoration``, is contained twice: it declares
                # source_record=None (no era authority), and its runs-only
                # target gate ("restoration target is not a runs award") keeps
                # candidates of this shape out of that path entirely (pinned by
                # test_award_restoration_runs_gate_contains_the_era_arm).
                #
                # Refusing the shape on replay of its FIRST mint (where no prior
                # head exists to match) re-litigated already-attested history
                # under later law -- the projection could not vouch the
                # pre-split heads, every close refused them as unaccounted
                # stale, and the requeue grants that re-drive them were
                # consumable only by the selection those closes gate. The class
                # stays disclosed on every census via
                # ``accuracy_gate_disposition_mislabel``, never silently normal.
                head_status = self._current.get(str(model.get("stable_id")), {}).get("status", {})
                if not (
                    head_status.get("code") == code and head_status.get("reason_code") == reason
                ) and not _era_replay_of_frozen_history(self, model):
                    raise ReductionError(
                        "failed:accuracy-gate requires a RENDERED accuracy adjudication; a "
                        f"terminal-disposition review outcome ({reason!r}) mints "
                        "blocked:disposition-review, never an accuracy-gate failure (W13)"
                    )
        elif kind == "blocked":
            # W13: the disposition-review hold. Stage-less by construction (the
            # deferred-terminal grant sentinel binds it), reason from the closed
            # pair, and ALWAYS a mandatory human-review hold -- that is what the
            # kind means.
            if stage is not None:
                raise ReductionError("blocked statuses cannot carry a failure stage")
            if reason not in BLOCKED_DISPOSITION_REVIEW_REASON_CODES:
                raise ReductionError(
                    f"reason {reason!r} is not allowed for blocked:disposition-review"
                )
            if not status.get("human_review", {}).get("required"):
                raise ReductionError("blocked:disposition-review requires human review")
        elif stage is not None or reason is not None:
            raise ReductionError("non-failed statuses cannot carry failure stage/reason")

    def _validate_source(self, model: Mapping[str, Any]) -> None:
        """Enforce the mandatory public source-link invariant.

        The guarantee this protects is about *catalogued* models: every completed record we
        publish must carry a real, exact, public primary source link, and must not misreport
        that it has one. The sole non-failure exception is a typed R5 discovery disposition
        whose primary source is the content-addressed discovery result itself. Such a record
        is either an epistemic skip or an unconsumed Opus-tier deferral, never a completion.

        A **failure** record is a different object. It exists precisely to record that the
        model could not be completed, and the failure itself is frequently what prevented a
        source from ever being resolved -- an author lane that died before source resolution,
        or a pinned artifact that could not be fetched. Demanding a resolved primary source
        from such a record demands information the failure is reporting the absence of.

        This distinction previously keyed on the single code ``failed:source`` rather than on
        the failure *kind*, which made the other eight terminal failure codes unsatisfiable
        whenever they occurred before source resolution. ``_placeholder_facts`` in
        ``driver_models`` builds exactly that shape (``sources=[]`` with
        ``mandatory_link_status="failed"``), so the engine was constructing records its own
        reducer categorically rejected, and a single unfetchable source aborted the whole
        campaign instead of recording one failed model.

        Failure records are still not permitted to *lie*: ``mandatory_link_status`` must agree
        with the source evidence actually present. That consistency check now applies to every
        failure kind, where it previously covered only ``failed:source``.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        resolution = model.get("source_resolution", {})
        sources = resolution.get("sources", [])
        primary = resolution.get("primary_source_id")
        exact_primary = bool(sources) and any(
            source.get("source_id") == primary
            and str(source.get("url", "")).startswith(("http://", "https://"))
            for source in sources
        )
        status_code = model.get("status", {}).get("code")
        typed_discovery = (
            status_code in SKIPPED_STATUS_CODES | CAPABILITY_DEFERRAL_STATUS_CODES
            # ``R5_SKIP`` is the CHECKED conclusion that no faithful source path
            # exists. For a capability deferral that conclusion is false -- both codes
            # are proved by an accepted BLOCKED verdict, which asserts only that a
            # prerequisite stopped us before the ladder ended, and both name a model we
            # LOCATED and could not author here. So the whole BLOCKED-proved class is
            # additionally admitted with the honest sentinel.
            #
            # The carve-out follows the ARM, not one code. Keying it on the access code
            # alone left ``deferred:needs-opus-tier`` -- the sibling half of the very set
            # ``CAPABILITY_DEFERRAL_STATUS_CODES`` exists to say shares one rule -- with
            # only one acceptable shape: the false ``R5_SKIP``. A validation rule that
            # is satisfiable only by an untrue record is not a tripwire, and the driver
            # dutifully wrote the lie to satisfy it.
            #
            # This relaxes NOTHING for a skip. Every ``skipped:*`` code is an epistemic
            # claim and still has to carry its checked R5 here, exactly as before, and
            # the bounded-search evidence below is still required of all of them.
            and (
                resolution.get("rung") == SourceRung.SKIP.value
                or (
                    status_code in CAPABILITY_DEFERRAL_STATUS_CODES
                    and resolution.get("rung") == NO_RUNG_SELECTED
                )
            )
            and len(sources) == 1
            and isinstance(sources[0], Mapping)
            and sources[0].get("source_id") == primary
            and sources[0].get("kind") == "discovery-evidence"
            and str(sources[0].get("url", "")).startswith("urn:menagerie:source-discovery:")
            and sources[0].get("revision_kind") == "source-discovery-sha256"
            and sources[0].get("revision") == sources[0].get("content_sha256")
            and sources[0].get("mirror_digest") == sources[0].get("content_sha256")
        )
        if typed_discovery:
            search_report = resolution.get("search_report")
            required_search_fields: tuple[str, ...] = (
                "queries",
                "places_checked",
                "languages_checked",
            )
            if status_code in LINK_EVIDENCED_DISCOVERY_STATUS_CODES:
                required_search_fields += ("links_checked",)
            if not isinstance(search_report, Mapping) or not all(
                isinstance(search_report.get(field), list)
                and bool(search_report.get(field))
                and all(
                    isinstance(value, str) and bool(value) for value in search_report.get(field, [])
                )
                for field in required_search_fields
            ):
                raise ReductionError(
                    "typed discovery disposition lacks bounded query/place/language evidence"
                    + (
                        " with the candidate locators its absence claim rests on"
                        if status_code in LINK_EVIDENCED_DISCOVERY_STATUS_CODES
                        else ""
                    )
                )
            if not isinstance(search_report.get("conclusion"), str) or not search_report.get(
                "conclusion"
            ):
                raise ReductionError("typed discovery disposition lacks a bounded conclusion")
        # ``kind`` is validated against the code prefix by ``_validate_status``, so it is the
        # reliable discriminator across every terminal failure code. W13: a
        # ``blocked`` disposition-review hold records that the member's own
        # terminal CLAIM was refused -- frequently a SKIP whose whole content is
        # that no usable source exists -- so demanding a resolved primary source
        # would demand information whose absence is being adjudicated, exactly
        # the failure-record argument above.
        failed = model.get("status", {}).get("kind") in {"failed", "blocked"}
        if not failed and not exact_primary and not typed_discovery:
            raise ReductionError("missing mandatory exact public primary source link")
        if failed and resolution.get("mandatory_link_status") != (
            "ok" if exact_primary else "failed"
        ):
            raise ReductionError("failure mandatory-link status contradicts source evidence")
        if not failed and resolution.get("mandatory_link_status") != (
            "failed" if typed_discovery else "ok"
        ):
            raise ReductionError(
                "terminal mandatory-link status contradicts its exact source evidence"
            )

    def _gate_item(
        self, gate_id: Optional[str], stable_id: str
    ) -> Optional[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        """Find a gate and its unique item for a stable ID.

        Parameters
        ----------
        gate_id:
            Referenced gate ID.
        stable_id:
            Model stable ID.

        Returns
        -------
        tuple[Mapping[str, Any], Mapping[str, Any]] | None
            Gate envelope and matching item, if present.
        """

        if gate_id is None:
            return None
        gate = self._gate_index().get(str(gate_id))
        if gate is None:
            return None
        if gate.get("schema_version") != GATE_SCHEMA_VERSION_V3:
            return None
        try:
            load_current_gate_proof(gate)
        except AuthorityDerivationError as exc:
            raise ReductionError(str(exc)) from exc
        matches = [item for item in gate["items"] if item["stable_id"] == stable_id]
        if len(matches) != 1:
            return None
        return gate, matches[0]

    def _validate_gates(self, model: Mapping[str, Any]) -> None:
        """Enforce block-at-write metadata and required fidelity gates.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        _GateValidationPipeline(self, model).run()

    def _validate_pending_run_rung(self, model: Mapping[str, Any]) -> None:
        """Validate a driver-owned R1/R2 run while authored metadata is pending.

        Parameters
        ----------
        model:
            Proposed pending-metadata run revision.
        """

        rung = model.get("source_resolution", {}).get("rung")
        if rung not in {"R1_LIBRARY", "R2_VENDOR"}:
            raise ReductionError("pending metadata runs are restricted to mechanical R1/R2")
        accuracy = model.get("accuracy_gate", {})
        # A pending run has NO sealed gate: unlike the FT-A recorded-basis
        # carve-outs above (each anchored to a gate row the ledger attests),
        # its prompt_sha256 records only the world it was ADMITTED under --
        # the checker never adjudicated it. Its currency across a checker
        # rotation therefore stays bound to the LIVE prompt bytes even under
        # semantic replay: a rotation stales the pending award, the launch
        # holds it at the award-custody door, and the operator's named
        # gate-catchup re-vets it under the live checker (AW x WS-1 COMPOSE,
        # pinned end-to-end by test_checker_digest_rotation_admission).
        prompt_basis = _checker_prompt_hash()
        if (
            accuracy.get("gate_id") is not None
            or accuracy.get("vet_identity") is not None
            or accuracy.get("verdict") is not None
            or accuracy.get("current")
            or not isinstance(accuracy.get("prompt_sha256"), str)
            or accuracy.get("prompt_sha256") != prompt_basis
        ):
            raise ReductionError("pending metadata run carries false accuracy-gate authority")
        for field in (
            "taxonomy",
            "external_metadata",
            "website",
            "people_and_origin",
            "dates",
            "citation",
            "licenses",
        ):
            if model.get(field) is not None:
                raise ReductionError("pending metadata run exposes ungated source-read fields")
        untrusted = model.get("untrusted_attempt")
        proposal = untrusted.get("proposal") if isinstance(untrusted, Mapping) else None
        if not isinstance(untrusted, Mapping) or not isinstance(proposal, Mapping):
            raise ReductionError("pending metadata run lacks its exact untrusted proposal")
        proposal_sha256 = stable_hash(
            {key: value for key, value in proposal.items() if key != "proposal_sha256"}
        )
        if (
            untrusted.get("proposal_sha256") != proposal_sha256
            or proposal.get("proposal_sha256") != proposal_sha256
            or proposal.get("stable_id") != model.get("stable_id")
        ):
            raise ReductionError("pending metadata run proposal identity is stale")
        proposed_facts = proposal.get("proposed_facts")
        if not isinstance(proposed_facts, Mapping):
            raise ReductionError("pending metadata run proposal facts are incomplete")
        for field in (
            "identity",
            "source_resolution",
            "evidence",
            "implementation",
            "input_contract",
            "fidelity",
        ):
            if model.get(field) != proposed_facts.get(field):
                raise ReductionError(f"pending metadata run changed mechanical fact root {field}")
        proposed_modes = proposed_facts.get("modes")
        model_modes = model.get("modes")
        if not isinstance(proposed_modes, Mapping) or not isinstance(model_modes, Mapping):
            raise ReductionError("pending metadata run modes are incomplete")
        if model_modes.get("meaningful_modes") != proposed_modes.get("meaningful_modes"):
            raise ReductionError("pending metadata run changed authored meaningful modes")
        try:
            identities = recompute_accepted_identities(
                proposed_facts,
                checker_prompt_hash=prompt_basis,
                checker_model=str(accuracy.get("checker_model")),
                checker_version=str(accuracy.get("checker_version")),
                schema_version=MODEL_SCHEMA_VERSION_V3,
            )
        except MetadataValidationError as exc:
            raise ReductionError(str(exc)) from exc
        if (
            proposal.get("source_identity") != identities.source
            or proposal.get("evidence_identity") != identities.evidence
            or proposal.get("recipe_revision") != identities.recipe
            or model.get("evidence", {}).get("evidence_identity") != identities.evidence
            or model.get("implementation", {}).get("recipe_revision") != identities.recipe
        ):
            raise ReductionError("pending metadata mechanical identities are stale")

    def _is_fidelity_repair_failure(self, model: Mapping[str, Any]) -> bool:
        """Return whether a runner terminal is an evidenced fidelity-repair failure.

        Parameters
        ----------
        model:
            Proposed terminal revision.

        Returns
        -------
        bool
            True only when a driver-owned runner attempt failed after a current
            rejected fidelity gate.
        """

        if model.get("status", {}).get("code") != "failed:runner":
            return False
        fidelity = model.get("fidelity", {})
        if not fidelity.get("current") or fidelity.get("verdict") not in {
            "major-drift",
            "slop",
            "cannot-verify",
        }:
            return False
        attempts_by_id = self._attempt_index()
        return any(
            (attempt := attempts_by_id.get(str(attempt_id))) is not None
            and attempt.get("actor") == "driver"
            and attempt.get("stage") == "runner"
            and attempt.get("result") == "failed"
            and attempt.get("environment") is None
            for attempt_id in model.get("status", {}).get("attempt_ids", [])
        )

    def _work_execution_evidence(self, model: Mapping[str, Any]) -> bool:
        """Return whether any same-work ledger attempt carries execution evidence.

        The PD-2 evidence rule, shared by the no-consumer arm and the AG-10
        routing refusal: execution evidence is a worker receipt
        (``worker_receipt.present``) or a started forward
        (``worker_receipt.forward_started``) on a current, ledger-bound attempt
        in a work generation the record's own referenced attempts name --
        never a dispatch intent (an environment binding or a pre-declared
        identity), and never anything a producer wrote on the record itself.

        Parameters
        ----------
        model:
            Proposed model revision.

        Returns
        -------
        bool
            Whether replay-derived execution evidence exists for this work.
        """

        stable_id = str(model.get("stable_id"))
        attempts_by_id = self._attempt_index()
        work_ids = {
            str(attempt.get("work_id"))
            for attempt_id in model.get("status", {}).get("attempt_ids", [])
            if (attempt := attempts_by_id.get(attempt_id)) is not None
        }
        if not work_ids:
            return False
        for attempt in self._current_attempt_records():
            if (
                attempt.get("stable_id") != stable_id
                or str(attempt.get("work_id")) not in work_ids
            ):
                continue
            receipt = attempt.get("worker_receipt", {})
            if receipt.get("present") or receipt.get("forward_started"):
                return True
        return False

    def _fidelity_gate_has_no_consumer(self, model: Mapping[str, Any]) -> bool:
        """Return whether the required fidelity gate has nothing it can adjudicate.

        This is the no-consumer contract: the fidelity gate verifies faithful
        execution of code that runs, so its absence is permitted exactly when the
        record proves there is no execution outcome for it to adjudicate. Two
        closed arms prove that:

        * the pre-execution arm -- the terminal stopped at one of the closed
          :data:`PRE_FIDELITY_FAILURE_STAGES` and NO referenced or same-work
          ledger attempt carries execution evidence (a worker receipt or a
          started forward). Since the JMT-signed PD-2/W8 widening the
          disqualifier is that EVIDENCE, never dispatch intent: a worker that
          was dispatched (environment bound, identities pre-declared) but
          provably died before executing still has nothing the gate can
          adjudicate; and
        * the execution-failure waiver arm -- an R3/R4 failure terminal whose
          death is replay-proven at exactly one of
          :data:`FIDELITY_WAIVER_EXECUTION_STAGES`
          (:meth:`_derive_execution_failure_waiver`).

        Both arms satisfy the four-part waiver doctrine (repo CLAUDE.md, beside
        the LOCKED validation-integrity principle): disposition-invariant,
        independently proven by a closed proof rule, replay-derived from
        authenticated evidence with zero producer input, and disclosed through
        the derived census (:func:`menagerie.crawler.status.funnel_counts`).

        Parameters
        ----------
        model:
            Proposed terminal revision with an R3/R4-required fidelity block.

        Returns
        -------
        bool
            Whether durable evidence proves the fidelity gate has no consumer.
        """

        status = model.get("status", {})
        fidelity = model.get("fidelity", {})
        execution = model.get("execution", {})
        if (
            status.get("kind") == "runs"
            or status.get("code") == "failed:fidelity"
            or fidelity.get("current")
            or fidelity.get("gate_id") is not None
            or fidelity.get("verdict") is not None
            or execution.get("current")
            or execution.get("accepted_attempt_ids")
        ):
            return False
        if (
            status.get("kind") == "failed"
            and status.get("stage") not in PRE_FIDELITY_FAILURE_STAGES
        ):
            if status.get("stage") in FIDELITY_WAIVER_EXECUTION_STAGES:
                return self._derive_execution_failure_waiver(model)
            return False
        # PD-2/W8 (the JMT-signed widening of registered waiver instance #1;
        # STATE.md "JMT APPROVALS RECORDED 2026-08-24"): the disqualifier is
        # EXECUTION EVIDENCE, never dispatch intent. The closed conjunction,
        # per referenced attempt AND per same-work ledger attempt: no worker
        # receipt (``receipt.present`` false) AND ``forward_started`` false --
        # composed with the model-level "no ``execution.current`` AND no
        # ``accepted_attempt_ids``" refusals above. A recorded environment
        # binding or a pre-declared ``identities.environment`` /
        # ``identities.execution`` says only that a worker was DISPATCHED;
        # a worker that provably died before executing left no receipt, and
        # disqualifying on the dispatch intent alone is what parked the five
        # bootstrap deaths (m2783, m2426, m3416, m1163, m1221 -- R18).
        # ``identities.execution`` therefore counts only when
        # receipt-corroborated, which the ``receipt.present`` arm already is.
        #
        # FOUR-PART PROOF (each part tested in
        # ``tests/test_fidelity_no_consumer_widening.py``):
        # 1. Disposition-invariant -- the arm only lets an already NEGATIVE
        #    failure terminal record; it upgrades nothing, publishes nothing,
        #    and unlocks nothing (the record stays release-ineligible and is
        #    never counted fidelity-audited).
        # 2. Closed proof rule -- the enumerated conjunction above, with the
        #    near-miss matrix pinned (receipt present, forward started,
        #    accepted attempts, execution.current, missing referenced attempt:
        #    every one-flag flip refuses).
        # 3. Replay-derived -- recomputed here on every append/replay from the
        #    ledger-bound attempt records; the same-work sweep below reads the
        #    CURRENT attempt ledger, so omitting an attempt from
        #    ``status.attempt_ids`` cannot hide its receipt; no persisted bit,
        #    zero producer input steers it.
        # 4. Disclosed -- the segregated ``fidelity_no_consumer_pre_execution``
        #    census arm (:func:`menagerie.crawler.status.funnel_counts`),
        #    never counted beside audited outcomes.
        attempts_by_id = self._attempt_index()
        referenced: list[Mapping[str, Any]] = []
        for attempt_id in status.get("attempt_ids", []):
            attempt = attempts_by_id.get(attempt_id)
            if attempt is None:
                return False
            referenced.append(attempt)
        # The same-work sweep keys on the REFERENCED attempts' own ledger-bound
        # work identity (never a producer-writable field on the record): every
        # current attempt in any referenced work generation is inspected, so
        # omitting an attempt from ``status.attempt_ids`` cannot hide its
        # receipt. A terminal referencing no attempts has no identifiable work
        # to sweep (the historical shape) and is still guarded by the
        # model-level ``execution.current``/``accepted_attempt_ids`` refusals.
        stable_id = str(model.get("stable_id"))
        work_ids = {str(attempt.get("work_id")) for attempt in referenced}
        same_work = tuple(
            attempt
            for attempt in self._current_attempt_records()
            if attempt.get("stable_id") == stable_id and str(attempt.get("work_id")) in work_ids
        )
        for attempt in (*referenced, *same_work):
            receipt = attempt.get("worker_receipt", {})
            if receipt.get("present") or receipt.get("forward_started"):
                return False
        return True

    def _derive_execution_failure_waiver(self, model: Mapping[str, Any]) -> bool:
        """Replay-derive the execution-failure fidelity waiver for one terminal.

        KERNEL_DECISION conditions 1-7, recomputed on every replay/rebuild with
        no persisted waiver bit and zero producer input:

        1. evidence comes exclusively from current-v3, ledger-bound,
           parent-attested attempts (:meth:`_current_attempt_records`) at the
           exact ``stable_id``/``work_id``; ``status.attempt_ids`` is consulted
           only as a fail-closed disqualifier (a referenced attempt missing from
           the ledger OR living at another stable ID / work generation refuses
           -- laundered foreign provenance is worse than a dangling reference),
           never as evidence;
        3. the claimed stage is in the closed set
           :data:`FIDELITY_WAIVER_EXECUTION_STAGES`;
        4. the decisive attempt is the maximal-order execution attempt in the
           current work, its stage matches ``status.stage`` and its error stage,
           and neither it nor ANY other non-succeeded same-work attempt --
           whatever its own stage; a resource or policy death that struck after
           the forward completed honestly carries the full milestone receipt --
           carries receipt or identity evidence beyond the claimed stage
           (succeeded partial-progress attempts are adjudicated by the
           no-success rule -- the deliberation's majority fork keeps
           partial-success-then-required-failure waiver-eligible);
        5. the decisive reason is closed for its stage and is not an effort
           exhaustion (an exhausted model is unfinished, not machine-dead);
        6. authenticated same-generation successes must not satisfy the required
           cold-forward policy, counted over MACHINE-WITNESSED modes (the modes
           the same-work attempts themselves name), never the producer's
           ``meaningful_modes`` claim alone: a claimed mode no attempt ever
           witnessed cannot hold the waiver open, so forging the mode list
           cannot hide completed forwards from the refusal. A later success in
           the required slot supersedes the failure (enforced by decisive
           maximality);
        7. the decisive attempt's already-derived environment binding must equal
           ``execution.env_generation`` (asserted, not re-derived), and its
           recipe identity must equal the CURRENT ``implementation.recipe_revision``
           so a pre-repair death can never waive a repaired recipe.

        Parameters
        ----------
        model:
            Proposed failure terminal at an execution stage.

        Returns
        -------
        bool
            Whether the waiver is proven. False always falls back to the
            unweakened "required fidelity is missing its gate" refusal.
        """

        status = model.get("status", {})
        fidelity = model.get("fidelity", {})
        execution = model.get("execution", {})
        stage = str(status.get("stage"))
        stable_id = str(model.get("stable_id"))
        if (
            status.get("kind") != "failed"
            or status.get("code") != f"failed:{stage}"
            or model.get("source_resolution", {}).get("rung") not in {"R3_PORT", "R4_REIMPLEMENT"}
            or fidelity.get("required") is not True
            or fidelity.get("current")
            or fidelity.get("gate_id") is not None
            or fidelity.get("verdict") is not None
            or execution.get("current")
            or execution.get("accepted_attempt_ids")
        ):
            return False
        work_id = self._model_work_id(model, self._artifact_authority_inputs(model))
        current = self._current_attempt_records()
        relevant = tuple(
            attempt
            for attempt in current
            if attempt.get("stable_id") == stable_id and attempt.get("work_id") == work_id
        )
        relevant_ids = {str(attempt.get("attempt_id")) for attempt in relevant}
        for referenced in status.get("attempt_ids", []) or ():
            if str(referenced) not in relevant_ids:
                return False
        execution_attempts = [
            attempt
            for attempt in relevant
            if attempt.get("stage") in _FIDELITY_WAIVER_ORDERING_STAGES
        ]
        if not execution_attempts:
            return False
        decisive = max(execution_attempts, key=_attempt_order)
        ledger_decisive = max(
            execution_attempts,
            key=lambda attempt: (
                attempt.get("ledger_seq") if isinstance(attempt.get("ledger_seq"), int) else -1,
                str(attempt.get("attempt_id", "")),
            ),
        )
        # ``attempt_no`` is allocated before concurrent workers finish, while
        # ``ledger_seq`` is reducer-owned completion order. A waiver is safe
        # only when both independent orders name the same final attempt. Any
        # disagreement is an ambiguous interleaving, never evidence that an
        # earlier failure remained decisive.
        if decisive.get("attempt_id") != ledger_decisive.get("attempt_id"):
            return False
        error = decisive.get("error")
        if (
            decisive.get("result") != "failed"
            or decisive.get("stage") != stage
            or decisive.get("actor") != ("driver" if stage == "environment" else "worker")
            or not isinstance(error, Mapping)
            or error.get("stage") != stage
        ):
            return False
        reason = error.get("reason_code")
        if (
            not isinstance(reason, str)
            or reason not in FAILURE_REASON_CODES.get(stage, frozenset())
            or reason in EFFORT_EXHAUSTION_REASON_CODES
            or reason != status.get("reason_code")
        ):
            return False
        identities = decisive.get("identities")
        identities = identities if isinstance(identities, Mapping) else {}
        # Condition 7: assert -- never re-derive -- the single environment
        # generation binding the authority pipeline resolves over the proof's
        # evidence (the decisive attempt plus same-generation successes). A
        # split binding refuses, and a proof with no observed environment must
        # carry the exact NOT_APPLICABLE sentinel the driver assembles.
        generation_witnesses = (
            decisive,
            *(attempt for attempt in relevant if attempt.get("result") == "succeeded"),
        )
        generations = {
            witness.get("identities", {}).get("environment")
            for witness in generation_witnesses
            if isinstance(witness.get("identities", {}).get("environment"), str)
        }
        if len(generations) > 1:
            return False
        derived_generation = next(
            iter(generations),
            stable_hash({"environment": DependencyState.NOT_APPLICABLE.value}),
        )
        if derived_generation != execution.get("env_generation"):
            return False
        environment_ids = {
            witness.get("environment", {}).get("env_id")
            for witness in generation_witnesses
            if isinstance(witness.get("environment"), Mapping)
            and isinstance(witness.get("environment", {}).get("env_id"), str)
        }
        if environment_ids and environment_ids != {execution.get("environment_id")}:
            return False
        recipe_revision = model.get("implementation", {}).get("recipe_revision")
        recipe_witnesses = {
            witness.get("identities", {}).get("recipe") for witness in generation_witnesses
        }
        if recipe_witnesses != {recipe_revision} or identities.get("recipe") != recipe_revision:
            return False
        selected_rung = model.get("source_resolution", {}).get("rung")
        if any(
            selected_rung not in witness.get("attempted_rungs", ())
            for witness in generation_witnesses
        ):
            return False
        if stage != "environment" and (
            not isinstance(decisive.get("environment"), Mapping)
            or not isinstance(identities.get("environment"), str)
            or not isinstance(identities.get("execution"), str)
        ):
            return False
        if _execution_evidence_beyond_stage(decisive, stage):
            return False
        for attempt in relevant:
            if attempt is decisive or attempt.get("result") == "succeeded":
                continue
            if attempt.get(
                "stage"
            ) in _FIDELITY_WAIVER_ORDERING_STAGES and _execution_evidence_beyond_stage(
                attempt, stage
            ):
                return False
        meaningful = tuple(
            dict.fromkeys(str(mode) for mode in model.get("modes", {}).get("meaningful_modes", ()))
        )
        if not meaningful:
            return False
        policy = cold_forward_policy(stable_id, model.get("source_resolution", {}).get("rung"))
        # Condition 6, machine-witnessed: count successes by the mode the
        # attempts THEMSELVES name. A claimed meaningful mode holds the waiver
        # open only while the attempt evidence witnesses it as short of the
        # policy; a mode present only in the producer's claim proves nothing.
        success_counts: dict[str, int] = {}
        witnessed_modes: set[str] = set()
        for attempt in relevant:
            mode = attempt.get("mode")
            if not isinstance(mode, str) or not mode:
                continue
            witnessed_modes.add(mode)
            if attempt.get("result") == "succeeded" and attempt.get("stage") == "forward":
                success_counts[mode] = success_counts.get(mode, 0) + 1
        policy_satisfied = any(
            count >= policy.required_cold_forwards for count in success_counts.values()
        )
        witnessed_short_mode = any(
            mode in witnessed_modes and success_counts.get(mode, 0) < policy.required_cold_forwards
            for mode in meaningful
        )
        if policy_satisfied and not witnessed_short_mode:
            return False
        return True

    def fidelity_waived_terminals(self) -> tuple[str, ...]:
        """Return current failure terminals whose required fidelity is waived.

        The derived-only disclosure marker (KERNEL_DECISION condition 8): the
        waiver is recomputed here from replayed authority, never persisted, and
        the matching census counters live in
        :func:`menagerie.crawler.status.funnel_counts`. Waived R3/R4
        implementations are never counted or published as fidelity-audited, and
        release eligibility keeps excluding them.

        Returns
        -------
        tuple[str, ...]
            Sorted stable IDs of waived current failure terminals.
        """

        waived = []
        for stable_id, record in self.current_records.items():
            status = record.get("status", {})
            required = (
                bool(record.get("fidelity", {}).get("required"))
                or record.get("source_resolution", {}).get("rung") in {"R3_PORT", "R4_REIMPLEMENT"}
                or legacy_requires_fidelity_audit(
                    record.get("intake", {}).get("preserved_legacy_flags", [])
                )
            )
            if (
                required
                and status.get("kind") == "failed"
                and status.get("stage") in FIDELITY_WAIVER_EXECUTION_STAGES
                and self._derive_execution_failure_waiver(record)
            ):
                waived.append(str(stable_id))
        return tuple(sorted(waived))

    def _validate_family_template(self, model: Mapping[str, Any]) -> None:
        """Validate the trusted-intake family branch before record-shaped hints."""

        stable_id = str(model.get("stable_id", ""))
        website = model.get("website")
        binding = self.intake_variant_bindings.get(stable_id)
        if binding is None:
            if isinstance(website, Mapping) and website.get("kind") == "size-variant-template":
                raise ReductionError("ordinary intake item cannot claim family variant authority")
            return
        if not isinstance(website, Mapping) or website.get("kind") != "size-variant-template":
            raise ReductionError("trusted family variant cannot omit its structural template")
        trusted_representative_id, trusted_variant_token = binding
        representative_id = model.get("identity", {}).get("family_representative_id")
        if (
            representative_id != trusted_representative_id
            or website.get("template_source_model_id") != trusted_representative_id
        ):
            raise ReductionError("family variant contradicts its trusted intake family binding")
        representative = self._validation_current_records().get(str(representative_id))
        if not isinstance(representative, Mapping) or not family_representative_is_usable(
            representative, str(representative_id)
        ):
            raise ReductionError("family variant representative is not a usable accepted record")
        try:
            terminal_variant = model.get("status", {}).get("kind") != "runs"
            validate_size_variant(
                representative,
                model,
                str(representative_id),
                parameter_count_total=(
                    None
                    if terminal_variant
                    else model.get("observed", {}).get("parameter_count_total")
                ),
                input_contract=(None if terminal_variant else model.get("input_contract", {})),
            )
            validate_size_variant_derivation(
                representative,
                model,
                str(representative_id),
                trusted_variant_token=trusted_variant_token,
            )
        except FamilyTemplateError as exc:
            raise ReductionError(f"family template validation failed: {exc}") from exc

    def _validate_completeness(self, model: Mapping[str, Any]) -> None:
        """Derive every completeness bit and reject producer contradictions.

        Parameters
        ----------
        model:
            Candidate already validated for source, gates, family, and execution.

        Raises
        ------
        ReductionError
            If any supplied completeness value differs from reducer-derived facts.
        """

        completeness = model.get("completeness")
        if not isinstance(completeness, Mapping):
            raise ReductionError("model completeness block is missing")
        status = model.get("status", {})
        status_kind = status.get("kind")
        metadata_accepted = model.get("authored_metadata_state") == "accepted"
        source_fields = (
            "taxonomy",
            "external_metadata",
            "website",
            "people_and_origin",
            "dates",
            "citation",
            "licenses",
        )
        source_read_complete = metadata_accepted and all(
            model.get(field) is not None for field in source_fields
        )
        coverage = model.get("evidence", {}).get("coverage", {})
        evidence_complete = bool(
            metadata_accepted
            and coverage.get("all_agent_fields_have_support") is True
            and coverage.get("family_grounding_complete") is True
            and coverage.get("missing_support") == []
        )
        accuracy = model.get("accuracy_gate", {})
        accuracy_current = bool(
            metadata_accepted
            and accuracy.get("current") is True
            and accuracy.get("verdict") == "accurate"
            and accuracy.get("gate_id")
            and accuracy.get("vet_identity")
        )
        fidelity = model.get("fidelity", {})
        fidelity_current = bool(not fidelity.get("required") or fidelity.get("current") is True)
        execution_current = bool(
            status_kind == "runs" and model.get("execution", {}).get("current") is True
        )
        family_valid = True
        human_review_pending = status.get("human_review", {}).get("required") is True
        release_eligible = bool(
            status_kind == "runs"
            and source_read_complete
            and evidence_complete
            and accuracy_current
            and fidelity_current
            and execution_current
            and family_valid
            and not human_review_pending
        )
        status_code = str(status.get("code"))
        if status_kind == "runs":
            expected_issues = [] if metadata_accepted else ["authored-metadata-pending"]
        else:
            expected_issues = [status_code]
        resolution = model.get("source_resolution", {})
        sources = resolution.get("sources", []) if isinstance(resolution, Mapping) else []
        primary = resolution.get("primary_source_id") if isinstance(resolution, Mapping) else None
        mandatory_source_present = bool(sources) and any(
            isinstance(source, Mapping)
            and source.get("source_id") == primary
            and str(source.get("url", "")).startswith(("http://", "https://"))
            for source in sources
        )
        expected: JsonObject = {
            "schema_valid": True,
            "mandatory_source_present": mandatory_source_present,
            "source_read_fields_complete": source_read_complete,
            "evidence_coverage_complete": evidence_complete,
            "accuracy_gate_current": accuracy_current,
            "required_fidelity_current": fidelity_current,
            "execution_current": execution_current,
            "family_template_valid": family_valid,
            "release_eligible": release_eligible,
            "issues": expected_issues,
        }
        for field, expected_value in expected.items():
            if completeness.get(field) != expected_value:
                raise ReductionError(
                    f"completeness.{field} contradicts reducer-derived value {expected_value!r}"
                )

    def _validate_execution(self, model: Mapping[str, Any]) -> None:
        """Enforce attempt/receipt and meaningful-mode rules for run awards.

        Parameters
        ----------
        model:
            Proposed model revision.
        """

        _ExecutionValidationPipeline(self, model).run()


def _production_canonical_root(ledgers: LedgerPaths) -> Optional[Path]:
    """Return the canonical crawler root for the production records layout.

    Parameters
    ----------
    ledgers:
        Candidate canonical ledger paths.

    Returns
    -------
    pathlib.Path | None
        ``menagerie/crawler`` root, or ``None`` for isolated test layouts.
    """

    records_root = ledgers.models.resolve().parent.parent
    canonical_root = records_root.parent
    if (
        records_root.name == "records"
        and canonical_root.name == "crawler"
        and canonical_root.parent.name == "menagerie"
    ):
        return canonical_root
    return None


#: Process-lifetime memo for :func:`_dependency_bytes_token`, keyed by the
#: resolved canonical root (or ``None`` for isolated test layouts). Wave
#: discipline freezes code, schema, prompt, and environment-manifest bytes for
#: a process's lifetime, so hashing them once per root is sound; LIVE
#: invalidation is carried by the per-ledger append generations in
#: :meth:`CanonicalReducer._projection_token`, never by these bytes. The
#: unmemoized scan was measured at 5.4 MB / 97 files = 20.5 ms per
#: ``current_records`` access (T4-T5).
_DEPENDENCY_BYTES_TOKEN_MEMO: dict[Optional[str], str] = {}


def _reset_dependency_bytes_token_memo() -> None:
    """Clear the process-lifetime authority-bytes memo (test hook)."""

    _DEPENDENCY_BYTES_TOKEN_MEMO.clear()


def _dependency_bytes_token(ledgers: LedgerPaths) -> str:
    """Return a cheap live-cache token for shared projection authorities.

    Parameters
    ----------
    ledgers:
        Canonical ledger and records-root paths.

    Returns
    -------
    str
        Stable token over prompts, authority code, schemas, and environment
        manifests, memoized for the process lifetime per canonical root.

        Per-model reconstruction/source bytes are validated while projecting that
        model and by every one-shot checkpoint/rebuild projection. They are not
        rescanned globally on each driver hot-path lookup.
    """

    canonical_root = _production_canonical_root(ledgers)
    memo_key = str(canonical_root) if canonical_root is not None else None
    memoized = _DEPENDENCY_BYTES_TOKEN_MEMO.get(memo_key)
    if memoized is not None:
        return memoized
    package_root = Path(__file__).parent
    authority_paths = [
        *package_root.glob("*.py"),
        *(package_root / "schemas").glob("*.json"),
        *(package_root / "prompts").glob("*.txt"),
    ]
    if canonical_root is not None:
        envs_root = canonical_root / "envs"
        if envs_root.exists():
            authority_paths.extend(path for path in envs_root.rglob("*") if path.is_file())
    facts: list[tuple[str, str]] = []
    for path in sorted(set(authority_paths)):
        try:
            digest = hash_bytes(path.read_bytes())
        except OSError:
            continue
        facts.append((str(path), digest))
    token = stable_hash(facts)
    _DEPENDENCY_BYTES_TOKEN_MEMO[memo_key] = token
    return token


def intake_variant_bindings_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[str, str]]:
    """Derive trusted non-representative family bindings from intake rows.

    Parameters
    ----------
    rows:
        Trusted immutable intake items.

    Returns
    -------
    dict[str, tuple[str, str]]
        Variant stable ID to exact representative ID and variant token.
    """

    bindings: dict[str, tuple[str, str]] = {}
    for row in rows:
        stable_id = row.get("stable_id")
        representative_id = row.get("family_representative_id") or stable_id
        if (
            isinstance(stable_id, str)
            and isinstance(representative_id, str)
            and representative_id != stable_id
            and row.get("variant_scope", "family") == "family"
        ):
            bindings[stable_id] = (representative_id, str(row.get("variant", "")))
    return bindings


def _era_replay_of_frozen_history(reducer: Any, model: Mapping[str, Any]) -> bool:
    """Decide the W13 era passage for one replayed candidate, producer-blind.

    True only when ALL of the following hold, each independently fail-closed:

    * the validating reducer is a semantic-replay reducer (module-private
      ``_replay_reducer`` construction over hash-validated ledger bytes);
    * the caller threaded the exact PERSISTED source row this replay
      re-attests (``source_record``); a novel candidate replaying under
      caller-identity replay mode (award restoration) threads ``None`` and is
      refused here -- ledger residency is the authority, not replay mode;
    * the candidate under validation IS that row (logical equality minus the
      append-assigned ``record_seq``/``record_revision``), so no caller can
      launder a different candidate through a resident row's era;
    * the row's ``accuracy_gate.verdict`` is ``None`` -- a rendered verdict
      beside a disposition reason is a contradiction, refused as one;
    * the digest RECOMPUTED from the source row's bytes is a member of the
      frozen realized-history cohort
      :data:`DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS`. The recompute makes
      membership bind bytes: the row's own ``record_revision`` field is never
      consulted, so a forged label names nothing.

    Parameters
    ----------
    reducer:
        The reducer running ``_validate_status`` (live or replay).
    model:
        Candidate revision under validation.

    Returns
    -------
    bool
        Whether the candidate is a frozen-cohort historical row replaying
        under the law of its own era.
    """

    if not bool(getattr(reducer, "_semantic_replay", False)):
        return False
    source = getattr(reducer, "_replay_source_record", None)
    if source is None:
        return False
    appended_assigned = {"record_seq", "record_revision"}
    candidate_logical = {
        key: value for key, value in model.items() if key not in appended_assigned
    }
    source_logical = {
        key: value for key, value in source.items() if key not in appended_assigned
    }
    if candidate_logical != source_logical:
        return False
    gate = model.get("accuracy_gate")
    verdict = gate.get("verdict") if isinstance(gate, Mapping) else gate
    if verdict is not None:
        return False
    return payload_hash(source) in DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS


def _replay_reducer(
    ledgers: LedgerPaths,
    *,
    context: AuthorityContext,
    prior_model_records: Sequence[Mapping[str, Any]],
    attempt_records: Sequence[Mapping[str, Any]],
    gate_records: Sequence[Mapping[str, Any]],
    artifact_records: Sequence[Mapping[str, Any]],
    raw_prior_current: Mapping[str, Mapping[str, Any]],
    dependency_current: Mapping[str, Mapping[str, Any]],
    attempt_index: Mapping[str, Mapping[str, Any]],
    gate_index: Mapping[str, Mapping[str, Any]],
    source_record: Optional[Mapping[str, Any]] = None,
) -> CanonicalReducer:
    """Build a write-free reducer positioned immediately before one candidate.

    Parameters
    ----------
    ledgers, context:
        Canonical paths and mandatory active authority.
    prior_model_records, attempt_records, gate_records, artifact_records:
        Hash-validated append-only history.
    raw_prior_current, dependency_current:
        Raw parent-lineage state and already dependency-current representatives.
    attempt_index, gate_index:
        Projection-wide immutable evidence indexes shared across every replay.
    source_record:
        The exact PERSISTED ledger row this replay re-attests, threaded from
        the caller that read it out of the hash-validated models ledger, or
        ``None`` when the candidate is NOT ledger-resident (the award
        restoration path replays a novel candidate it is about to mint). The
        era passage in ``_validate_status`` derives its residency fact from
        this row alone -- an absent row grants nothing (fail-closed), so a
        replay-mode reducer over a non-resident candidate carries replay
        CALLER identity but never replay ERA authority.

    Returns
    -------
    CanonicalReducer
        In-memory reducer whose ordinary ``append_model`` performs the replay.
    """

    replay: Any = CanonicalReducer.__new__(CanonicalReducer)
    replay.ledger_paths = ledgers
    replay._semantic_replay = True
    replay._replay_source_record = source_record
    replay.context = context
    # Replay resolves environment families with the same ledger-attested
    # bindings the live driver registers: every attempt row in evidence
    # attests its generation -> family pair, and driver lifecycle events
    # attest theirs (base generations included), so a terminal recorded
    # through the historical fallback -- and the replay-side environment
    # anchor in _validate_environment_generation -- re-validates identically
    # on replay. Same closed evidence rule as reducer construction.
    replay.historical_generation_families = _attested_generation_bindings(
        attempt_records,
        _scan_operational_evidence_cached(
            _records_root(ledgers) / "operational" / "events.jsonl"
        ),
    )
    replay.intake_ids = frozenset(context.intake_by_stable_id)
    replay.intake_variant_bindings = {
        stable_id: (
            str(binding.get("representative_stable_id", binding.get("family_representative_id"))),
            str(binding.get("variant_token", binding.get("variant", ""))),
        )
        for stable_id, binding in context.family_bindings.items()
        if isinstance(binding, Mapping)
        and binding.get("binding_state") != "ordinary"
        and binding.get("representative_stable_id", binding.get("family_representative_id"))
        not in {None, stable_id}
    }
    replay._models = _ReplayLedger(prior_model_records)
    replay._attempts = _ReplayLedger(attempt_records)
    replay._gates = _ReplayLedger(gate_records)
    replay._artifacts = _ReplayArtifactLedger(artifact_records)
    replay._current = {stable_id: record for stable_id, record in raw_prior_current.items()}
    replay._validation_current = dependency_current
    replay._attempt_index_cache = attempt_index
    replay._gate_index_cache = gate_index
    replay._projection_cache = None
    replay._generation_families_epoch = 0
    return replay


def _referenced_evidence_repasses(
    replay: CanonicalReducer,
    record: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> None:
    """Replay append admission for every attempt and gate used by one record.

    Parameters
    ----------
    replay:
        Side-effect-free reducer carrying canonical authority inputs.
    record:
        Candidate current model revision.
    attempts, gates:
        Complete canonical evidence histories.

    Raises
    ------
    ReductionError
        If referenced evidence no longer passes its original admission predicate.
    """

    attempt_ids = set(_record_attempt_ids(record))
    attempts_by_id = {str(value.get("attempt_id")): value for value in attempts}
    pending = list(attempt_ids)
    while pending:
        referenced_id = pending.pop()
        referenced = attempts_by_id.get(str(referenced_id))
        if referenced is None:
            continue
        defer_evidence = referenced.get("defer_evidence")
        probe_ids = (
            defer_evidence.get("probe_attempt_ids", [])
            if isinstance(defer_evidence, Mapping)
            else []
        )
        for probe_id in probe_ids:
            normalized = str(probe_id)
            if normalized not in attempt_ids:
                attempt_ids.add(normalized)
                pending.append(normalized)
    for attempt_id in sorted(str(value) for value in attempt_ids):
        attempt = attempts_by_id.get(attempt_id)
        if attempt is None:
            raise ReductionError(f"referenced attempt is missing: {attempt_id}")
        replay.append_attempt(attempt)

    gate_ids = {
        record.get("accuracy_gate", {}).get("gate_id"),
        record.get("fidelity", {}).get("gate_id"),
    } - {None}
    gates_by_id = {str(value.get("gate_id")): value for value in gates}
    for gate_id in sorted(str(value) for value in gate_ids):
        gate = gates_by_id.get(gate_id)
        if gate is None:
            raise ReductionError(f"referenced gate is missing: {gate_id}")
        replay.append_gate(gate)


#: Dependency-vector axes that are content hashes of shipped policy bytes (or
#: policy closures), paired with the ``AuthorityContext`` field each derives
#: from. They legitimately move whenever the shipped code, prompts, or schemas
#: evolve between legs. For an ordinary record that movement is the CURRENCY
#: SIGNAL: the projection marks the record stale and the scheduler re-drives
#: the model under current rules (striker-4 ruling). For a record HELD FOR
#: HUMAN REVIEW that re-drive is fenced by design -- ``append_model`` refuses
#: any supersession without a new explicit grant -- so flagging it stale
#: schedules work whose outcome can never be recorded: rung-11 leg-4 re-drove
#: 21 held models, and every one ended in ``human-review terminal supersession
#: requires a new explicit grant``, an unrecordable-terminal cascade, and
#: finally the ledger-conflict abort. A held record therefore REPLAYS AT THE
#: RULE BASIS IT RECORDED (these axes pinned from its own vector); every
#: record-anchored axis and proof cross-check still runs at full strength, and
#: the explicit requeue grant remains the one path to re-adjudication.
_REVIEW_HELD_RECORDED_AXES: tuple[tuple[str, str], ...] = (
    ("author_result_schema_identity", "author_schema_identity"),
    ("author_dispatcher_identity", "author_dispatcher_identity"),
    ("author_prompt_identity", "author_prompt_identity"),
    ("checker_prompt_identity", "checker_prompt_identity"),
    ("terminal_rule_identity", "terminal_policy_identity"),
    ("publication_policy_identity", "publication_policy_identity"),
    ("runner_identity", "runner_policy_identity"),
    ("award_closure_identity", "reducer_policy_identity"),
)

#: Public spelling of the recorded-basis axes for the recheck census and the
#: rotation tooling (RC-2): the same eight (dependency-vector axis,
#: AuthorityContext field) pairs, one literal.
RECORDED_BASIS_AXES = _REVIEW_HELD_RECORDED_AXES


def _record_requires_human_review(record: Mapping[str, Any]) -> bool:
    """Return whether one model revision is held for explicit human review."""

    status = record.get("status")
    if not isinstance(status, Mapping):
        return False
    human_review = status.get("human_review")
    return isinstance(human_review, Mapping) and bool(human_review.get("required"))


def _review_held_recorded_basis_context(
    context: AuthorityContext, record: Mapping[str, Any]
) -> AuthorityContext:
    """Return the replay context for one review-held record's own rule basis.

    Only the shipped-policy identity axes and the recorded environment
    generation are pinned (see :data:`_REVIEW_HELD_RECORDED_AXES`); the intake
    axes, family authority, and every record-anchored evidence cross-check
    stay live so a record contradicting its own ledger evidence still reads
    stale and fail-closed. Pin authenticity is the ledger's own row hash:
    every scanned record validates its content digest, so a rewritten vector
    cannot pass as recorded history.

    Parameters
    ----------
    context, record:
        Live authority context and the held current model revision.

    Returns
    -------
    AuthorityContext
        Per-record replay context; the live context whenever the record
        carries no usable recorded vector.
    """

    vector = record.get("dependency_vector")
    if not isinstance(vector, Mapping):
        return context
    overrides: dict[str, Any] = {}
    for axis, field in _REVIEW_HELD_RECORDED_AXES:
        value = vector.get(axis)
        if isinstance(value, str) and value and value != DependencyState.NOT_APPLICABLE.value:
            overrides[field] = value
    generation = vector.get("environment_generation")
    if (
        isinstance(generation, str)
        and generation
        and generation != DependencyState.NOT_APPLICABLE.value
        and generation not in set(context.environment_generations.values())
    ):
        overrides["environment_generations"] = {
            **dict(context.environment_generations),
            "review-held-recorded-basis": generation,
        }
    if not overrides:
        return context
    return replace(context, **overrides)


#: Environment-generations key under which a runs head's recorded, ledger-
#: attested generation is injected for its own recorded-basis replay.
_AWARD_RECORDED_BASIS_GENERATION_KEY = "award-recorded-basis"


def _record_is_runs_award(record: Mapping[str, Any]) -> bool:
    """Return whether one model revision is a ``runs`` award head."""

    status = record.get("status")
    return isinstance(status, Mapping) and status.get("kind") == "runs"


def _refuse_unmarked_gate_in_shadow_mode(gate: Mapping[str, Any]) -> None:
    """W3 write-side producer (gate door): refuse an unmarked live gate.

    A gate's shadow marker must be sealed INSIDE ``result_envelope_sha256``
    at the dispatch scaffold (``apply_machine_owned_gate_fields``) -- stamping
    at the append door, after the seal, would mint a row whose self-hash
    every later read refuses. So the door REFUSES an unmarked gate while the
    shadow-rehearsal mode is active instead of silently stamping.

    Parameters
    ----------
    gate:
        Proposed live gate row (semantic replay is exempt at the call site).

    Raises
    ------
    ReductionError
        The shadow-rehearsal mode is active and the gate carries no marker.
    """

    if shadow_rehearsal_run_active() and not is_shadow_rehearsal_record(gate):
        raise ReductionError(
            "shadow-rehearsal mode refuses an unmarked gate row: the "
            "shadow_rehearsal marker must be sealed into the gate's "
            "result envelope at the dispatch scaffold, not appended after"
        )


def _withhold_award_issuance_in_calibration_mode(candidate: Mapping[str, Any]) -> None:
    """W3: calibration mode structurally withholds award issuance, fail-closed.

    The sanctioned calibration capture run
    (``MENAGERIE_CHECKER_CALIBRATION_RUN=1``) exists to buy measurement
    sessions and render verdicts, never to bank them: a LIVE append that would
    issue a ``runs`` award head or mark a record release-eligible refuses
    typed while the flag is set. The W3 shadow-rehearsal mode
    (``MENAGERIE_SHADOW_REHEARSAL_RUN``) withholds identically -- marker or
    no marker on the candidate. Semantic replay is exempt at the call sites
    -- the withhold governs issuance, not re-admission of issued history.

    Parameters
    ----------
    candidate:
        Proposed model revision.

    Raises
    ------
    CheckerCalibrationWithheldError
        The candidate is a runs award or claims release eligibility while
        calibration mode is active.
    """

    completeness = candidate.get("completeness")
    marked_release_eligible = (
        isinstance(completeness, Mapping) and completeness.get("release_eligible") is True
    )
    # W44 award-seam rider (W3's shadow-marker half): a candidate CARRYING the
    # structural shadow_rehearsal marker refuses award issuance REGARDLESS of
    # any mode flag -- the marker is on the record, the env flag is
    # disclosure-only. A rehearsal verdict is attested measurement, never a
    # bankable award.
    if is_shadow_rehearsal_record(candidate) and (
        _record_is_runs_award(candidate) or marked_release_eligible
    ):
        act = (
            "issue a runs award head"
            if _record_is_runs_award(candidate)
            else "mark a record release-eligible"
        )
        raise CheckerCalibrationWithheldError(
            f"shadow-rehearsal record withholds awards: refusing to {act} for "
            f"{candidate.get('stable_id')!r}; the record carries the structural "
            "shadow_rehearsal marker and its verdicts are measurements, never "
            "bankable adjudications"
        )
    if not calibration_run_active() and not shadow_rehearsal_run_active():
        return
    if _record_is_runs_award(candidate) or marked_release_eligible:
        act = (
            "issue a runs award head"
            if _record_is_runs_award(candidate)
            else "mark a record release-eligible"
        )
        if calibration_run_active():
            raise CheckerCalibrationWithheldError(
                f"calibration mode withholds awards: refusing to {act} for "
                f"{candidate.get('stable_id')!r} while {CHECKER_CALIBRATION_RUN_ENV}=1; "
                "calibration verdicts are measurements, never bankable adjudications"
            )
        # W3 third mode: the shadow-rehearsal MODE withholds exactly like
        # calibration -- even for a candidate not yet carrying the marker
        # (the marker-based rider above stays independent). Rehearsal
        # verdicts are attested measurements, never bankable adjudications.
        raise CheckerCalibrationWithheldError(
            f"shadow-rehearsal mode withholds awards: refusing to {act} for "
            f"{candidate.get('stable_id')!r} while {SHADOW_REHEARSAL_RUN_ENV} is set; "
            "rehearsal verdicts are measurements, never bankable adjudications"
        )


def _attested_generation_bindings(
    attempts: Iterable[Mapping[str, Any]],
    operational_rows: Iterable[Mapping[str, Any]],
) -> dict[str, str]:
    """Derive every ledger-attested environment generation -> family binding.

    Replay-derived from authenticated evidence with zero producer input: every
    attempt row attests its own generation/family pair, and environment
    lifecycle events (rebuilds, quarantines) attest theirs in their details.
    This is the projection-side inverse of the driver's
    ``attested_generation_families`` registration (d165c9d9), computed from
    the same sources so the recorded-basis carve-out is identically live in
    the driver, the status boundary, and every staging tool -- the
    authority-construction drift that made a driver-closed rung report
    missing members cannot recur for award currency.

    Parameters
    ----------
    attempts:
        Canonical attempt rows.
    operational_rows:
        Parsed operational ledger records.

    Returns
    -------
    dict[str, str]
        Environment generation identity to family/intent name.
    """

    bindings: dict[str, str] = {}
    for attempt in attempts:
        environment = attempt.get("environment")
        generation = attempt.get("identities", {}).get("environment")
        family = environment.get("family") if isinstance(environment, Mapping) else None
        if isinstance(family, str) and family and isinstance(generation, str) and generation:
            bindings[generation] = family
    for event in operational_rows:
        details = event.get("details", {})
        if not isinstance(details, Mapping):
            continue
        intent = details.get("intent")
        generation = details.get("env_generation")
        if isinstance(intent, str) and intent and isinstance(generation, str) and generation:
            bindings[generation] = intent
        binding = details.get("environment")
        if isinstance(binding, Mapping):
            bound_generation = binding.get("env_generation")
            if (
                isinstance(intent, str)
                and intent
                and isinstance(bound_generation, str)
                and bound_generation
            ):
                bindings[bound_generation] = intent
    return bindings


def _award_recorded_basis_context(
    context: AuthorityContext,
    record: Mapping[str, Any],
    *,
    attested_bindings: Mapping[str, str],
) -> AuthorityContext:
    """Return the replay context for one settled head's own recorded basis.

    A record is a claim about a specific executed artifact under recorded
    closures, proven by replay-derived evidence; rotating live code or prompts
    changes what NEW work would do, never what happened (AUDIT100 SYNAUD_A
    section 2.2). The eight shipped-policy axes
    (:data:`RECORDED_BASIS_AXES`) therefore replay at their RECORDED values
    for every settled head, unconditionally (RC-2). Whether a move on one of
    those axes impugns the recorded acceptance is a separate fact -- the
    measured rotation class in ``records/control/rotations.jsonl`` -- and it
    feeds exactly one derived view, the recheck census, never this replay:
    an award under a soundness rotation stays an award, projects current,
    and is release-excluded and re-checked through the census. The recorded
    environment generation resolves against the LEDGER-ATTESTED generation
    set -- historical is not stale; attested nowhere refuses (a changed
    input). Intake axes, family authority, and every record-anchored evidence
    cross-check stay live, so a record contradicting its own ledger still
    fails closed. Pin authenticity is the ledger's own row hash.

    Parameters
    ----------
    context, record:
        Live authority context and the current model revision.
    attested_bindings:
        Ledger-attested environment generation -> family bindings.

    Returns
    -------
    AuthorityContext
        Per-record replay context; the live context whenever the record
        carries no usable recorded vector.

    Raises
    ------
    ReductionError
        For a recorded generation no ledger evidence attests.
    """

    vector = record.get("dependency_vector")
    if not isinstance(vector, Mapping):
        return context
    overrides: dict[str, Any] = {}
    for axis, field in RECORDED_BASIS_AXES:
        recorded = vector.get(axis)
        if (
            not isinstance(recorded, str)
            or not recorded
            or recorded == DependencyState.NOT_APPLICABLE.value
        ):
            continue
        if recorded != getattr(context, field):
            overrides[field] = recorded
    generation = vector.get("environment_generation")
    if (
        isinstance(generation, str)
        and generation
        and generation != DependencyState.NOT_APPLICABLE.value
        and generation not in set(context.environment_generations.values())
    ):
        if generation not in attested_bindings:
            raise ReductionError(
                "award environment generation is attested nowhere in ledger "
                f"evidence: {generation!r}"
            )
        overrides["environment_generations"] = {
            **dict(context.environment_generations),
            _AWARD_RECORDED_BASIS_GENERATION_KEY: generation,
        }
    if not overrides:
        return context
    return replace(context, **overrides)


def _derived_disposition(replay: CanonicalReducer, candidate: Mapping[str, Any]) -> JsonObject:
    """Return the RUNNING code's terminal disposition for one replayed head.

    RC-2 predicate replay probe support: the reducer re-derives the terminal
    proof from recorded evidence under the head's recorded policy basis; the
    probe compares these fields with the recorded status field by field, so a
    predicate edit that turns a prior non-acceptance into ``runs``, or one
    failure disposition into another, is a reported flip with both values --
    never inferred from whether the replay happened to refuse.
    """

    try:
        terminal_proof, _family, _vector, _artifacts = replay._derive_model_authority(candidate)
    except (KeyError, ReductionError, TypeError, ValueError) as exc:
        return {"error": str(exc)}
    code = str(terminal_proof.status_code)

    def _plain(value: Any) -> Any:
        return value.value if isinstance(value, DependencyState) else value

    return {
        "disposition": {
            "kind": code.split(":", 1)[0],
            "code": code,
            "stage": _plain(terminal_proof.failure_stage),
            "reason_code": _plain(terminal_proof.reason_code),
            "root_cause_fingerprint": _plain(terminal_proof.root_cause_fingerprint),
        }
    }


def project_dependency_current(
    ledgers: LedgerPaths,
    *,
    context: AuthorityContext,
    model_records: Optional[Sequence[Mapping[str, Any]]] = None,
    attempt_records: Optional[Sequence[Mapping[str, Any]]] = None,
    gate_records: Optional[Sequence[Mapping[str, Any]]] = None,
    artifact_records: Optional[Sequence[Mapping[str, Any]]] = None,
    disposition_sink: Optional[MutableMapping[str, JsonObject]] = None,
) -> DependencyCurrencyProjection:
    """Project the one authoritative current view from current dependency bytes.

    Parameters
    ----------
    ledgers:
        Canonical ledger paths and records-root identity.
    context:
        Mandatory active authority shared with every reducer consumer.
    model_records, attempt_records, gate_records, artifact_records:
        Optional already scanned rows used by the live reducer without another disk read.
    disposition_sink:
        RC-2 predicate replay probe: when given, every head's RUNNING-code
        derivation is recorded here as ``{"disposition": {...}}`` (status
        kind and code plus the failed-status proof fields) or
        ``{"error": str}`` when the derivation itself refuses, BEFORE the
        replay compares it to the recorded head. The probe compares the two
        field by field; the projection's own verdict is unchanged.

    Returns
    -------
    DependencyCurrencyProjection
        Replayed dependency-current records and fail-closed stale reasons.
    """

    models = tuple(model_records) if model_records is not None else scan_jsonl(ledgers.models)
    attempts = (
        tuple(attempt_records) if attempt_records is not None else scan_jsonl(ledgers.attempts)
    )
    gates = tuple(gate_records) if gate_records is not None else scan_jsonl(ledgers.gates)
    artifacts = (
        tuple(artifact_records)
        if artifact_records is not None
        else tuple(scan_jsonl(ledgers.artifacts))
    )
    _validate_persisted_requeue_lineage(models, ledgers, attempts, gates, artifacts)
    raw_current = _select_current(models)
    by_revision = {str(record.get("record_revision")): index for index, record in enumerate(models)}
    attempts_by_id = {str(record.get("attempt_id")): record for record in attempts}
    gates_by_id = {str(record.get("gate_id")): record for record in gates}
    # Recorded-basis input, loaded ONCE per projection and only when a head
    # exists: the ledger-attested generation set (historical is not stale),
    # replay-derived from the records root with zero caller plumbing, so the
    # carve-out is identically live in the driver, the status boundary, and
    # every staging tool. Since RC-2 NO rotation ledger is read here: every
    # settled head replays at its recorded policy basis unconditionally, and a
    # rotation's CLASS feeds the recheck census (recheck_census.py), never
    # currency.
    records_root = _records_root(ledgers)
    if raw_current:
        award_attested = _attested_generation_bindings(
            attempts,
            _scan_operational_evidence_cached(records_root / "operational" / "events.jsonl"),
        )
    else:
        award_attested = {}
    current: dict[str, JsonObject] = {}
    stale: dict[str, str] = {}
    ordered_current = sorted(
        raw_current.items(), key=lambda value: by_revision[str(value[1].get("record_revision"))]
    )
    prior_current: dict[str, JsonObject] = {}
    prior_models_by_stable_id: dict[str, list[Mapping[str, Any]]] = {}
    cursor = 0
    for stable_id, record in ordered_current:
        try:
            index = by_revision[str(record.get("record_revision"))]
            while cursor < index:
                prior = models[cursor]
                prior_current[str(prior.get("stable_id"))] = deepcopy(dict(prior))
                prior_models_by_stable_id.setdefault(str(prior.get("stable_id")), []).append(prior)
                cursor += 1
            replay = _replay_reducer(
                ledgers,
                context=(
                    _review_held_recorded_basis_context(context, record)
                    if _record_requires_human_review(record)
                    # EVERY settled head replays at its recorded basis: a
                    # terminal (failed/skipped/deferred alike) is a claim
                    # adjudicated under recorded closures, and rotating live
                    # code changes what NEW work would do, never what
                    # happened (AW-2, extended at the leg-29 close: the first
                    # clean close after identity-component fixes read 139
                    # recorded terminals as stale:terminal_rule_identity /
                    # stale:author_dispatcher_identity / stale:environment_
                    # generation). RC-2: the pin is UNCONDITIONAL -- a rotation
                    # class never holds or stales a head here; soundness feeds
                    # the recheck census instead. A generation attested
                    # nowhere still refuses (a changed input), landing the
                    # head in `stale`, never silently current.
                    else _award_recorded_basis_context(
                        context,
                        record,
                        attested_bindings=award_attested,
                    )
                ),
                prior_model_records=prior_models_by_stable_id.get(stable_id, ()),
                attempt_records=attempts,
                gate_records=gates,
                artifact_records=artifacts,
                raw_prior_current=prior_current,
                dependency_current=current,
                attempt_index=attempts_by_id,
                gate_index=gates_by_id,
                source_record=record,
            )
            _referenced_evidence_repasses(replay, record, attempts, gates)
            replay_candidate = {
                key: value
                for key, value in record.items()
                if key not in {"record_seq", "record_revision"}
            }
            if disposition_sink is not None:
                disposition_sink[stable_id] = _derived_disposition(replay, replay_candidate)
            replay.append_model(replay_candidate)
            family_error = family_variant_currency_error(record, current)
            if family_error is not None:
                raise ReductionError(family_error)
            current[stable_id] = deepcopy(dict(record))
        except (KeyError, ReductionError, TypeError, ValueError) as exc:
            stale[stable_id] = str(exc)
    return DependencyCurrencyProjection(current, stale)


def materialize_current(
    ledgers: LedgerPaths,
    *,
    context: AuthorityContext,
) -> Mapping[str, JsonObject]:
    """Read and deterministically materialize current model revisions.

    Parameters
    ----------
    ledgers:
        Canonical ledger paths.

    Returns
    -------
    Mapping[str, dict[str, Any]]
        Highest valid dependency-current revision per stable ID. Stale family
        variants remain in append-only history but are excluded from this view.
    """

    return project_dependency_current(
        ledgers,
        context=context,
    ).current_records


def default_ledger_paths(root: Union[str, Path]) -> LedgerPaths:
    """Return conventional Slice-A ledger paths below a root directory.

    Parameters
    ----------
    root:
        Canonical records root.

    Returns
    -------
    LedgerPaths
        Models, attempts, and gates JSONL paths.
    """

    base = Path(root)
    return LedgerPaths(
        models=base / "models" / "current-shard.jsonl",
        attempts=base / "attempts" / "local.jsonl",
        gates=base / "gates" / "current-shard.jsonl",
    )
