"""Private-first artifact custody, authorization, and publication transactions.

This module is intentionally independent of the driver and reducer hubs.  It
implements the frozen ``artifact-event.v1`` state machine and accepts the
frozen authority values emitted by the reducer.  Public bytes are unreachable
until an exact :class:`PublicationAuthorization` has first been committed to
the append-only artifact ledger.
"""

from __future__ import annotations

import os
from bisect import insort
from copy import deepcopy
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Optional, Sequence, TypeAlias

if TYPE_CHECKING:
    from menagerie.crawler.checkpoint_cache import ValidationMemo

from menagerie.crawler.authority import (
    ArtifactClaim,
    ArtifactClaimId,
    ArtifactTransactionId,
    AuthorityContext,
    DependencyState,
    DependencyValue,
    DependencyVector,
    MirrorObject,
    MirrorObjectId,
    PublicationAuthorization,
    PublicationAuthorizationId,
)
from menagerie.crawler.constants import (
    ARTIFACT_EVENT_SCHEMA_VERSION,
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_RESULT_SCHEMA_VERSION,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    hash_bytes,
    is_shadow_rehearsal_record,
    payload_hash,
    stable_hash,
    stamp_shadow_rehearsal,
    utc_now,
)
from menagerie.crawler.licenses import (
    LicenseDecision,
    RedistributionClass,
    pre_public_claim_sweep,
)
from menagerie.crawler.mirrors import (
    ArtifactOrigin,
    MirrorClass,
    MirrorStore,
    RetentionClass,
    is_addressable_object_key,
)
from menagerie.crawler.models import (
    SUPPLEMENTARY_SOURCES_KEY,
    AppendResult,
    JsonObject,
    manifest_source_rows,
)
from menagerie.crawler.recordio import (
    JsonlLedger,
    durable_fsync,
    record_validation_memo_key,
    scan_jsonl,
)
from menagerie.crawler.schema import PayloadValidationError, validate_payload

ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION_V1 = "menagerie.crawler.artifact-reconstruction.v1"
ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION = "menagerie.crawler.artifact-reconstruction.v2"


class ArtifactTransactionError(RuntimeError):
    """Base class for artifact transaction failures."""


class ArtifactBindingError(ArtifactTransactionError):
    """Raised when supplied bytes or authority do not match frozen inputs.

    The default reading is AUTHOR-owned: almost every binding refusal is the
    engine rejecting something the author supplied.
    """


class SourceManifestBindingError(ArtifactBindingError):
    """Raised when the ENGINE's own frozen source manifest is self-inconsistent.

    Split out because ownership decides how the model is terminalized. The
    source manifest is built by the driver, not the author -- the author only
    echoes its identity, and that echo is already verified against the envelope
    before staging is reached. So a manifest that fails its own shape or
    identity checks here is an engine fault, and reporting it against the
    author (as a crashed session) sends operators hunting for a research
    failure that never happened.
    """


class ArtifactTransitionError(ArtifactTransactionError):
    """Raised when an artifact event violates the predecessor state machine."""


class ArtifactChainSpentError(ArtifactTransitionError):
    """Raised for a NEW transition against a chain already carrying its final event.

    The ONE recoverable transition refusal. Every other transition error is
    corruption-shaped (conflicting replay, forked predecessor, mutated
    identity) and must stay loud: a recovery path may route THIS class to the
    solve-cache-miss / fresh-transaction arm, and may never widen its catch to
    the base class. m12646 (2026-08-23) looped for four hours because the
    spent-chain retry surfaced as the same generic ``too many transitions``
    text as ledger corruption. Byte-exact replay of an existing event is NOT
    this refusal -- it stays idempotent recognition upstream of this check.
    """

    def __init__(
        self,
        *,
        transaction_id: str,
        stable_id: object,
        work_id: object,
        final_event_kind: str,
        attempted_event_kind: object,
    ) -> None:
        super().__init__(
            f"artifact transaction chain is spent: {transaction_id} already "
            f"carries final event {final_event_kind}; refusing new "
            f"{attempted_event_kind} transition for stable_id={stable_id} "
            f"work_id={work_id}"
        )
        self.transaction_id = transaction_id
        self.stable_id = stable_id
        self.work_id = work_id
        self.final_event_kind = final_event_kind
        self.attempted_event_kind = attempted_event_kind


class ArtifactPublicationError(ArtifactTransactionError):
    """Raised before an unauthorized or contradictory materialization."""


class AuthorResultSchemaDriftError(ArtifactBindingError):
    """Raised when an author result fails the LIVE result-schema validation.

    Split from the plain binding refusal because the READ path needs the
    distinction: a recorded transaction whose bytes still match every anchor
    but whose author result no longer validates against the CURRENT
    author-result schema is a stale reuse-cache entry (the schema tightened
    after the artifact was written), not corruption. On the WRITE path this
    remains an ordinary hard :class:`ArtifactBindingError`.
    """


class ArtifactCheckpointError(ArtifactTransactionError):
    """Raised when ledger, reconstruction, or physical inventory diverges."""


class ArtifactRehydrationError(ArtifactCheckpointError):
    """Raised when final authority cannot be materialized from private custody."""


class ArtifactSchemaStaleError(ArtifactCheckpointError):
    """Raised for one finalized transaction whose recorded author result fails
    the LIVE author-result schema.

    Every byte anchor of the transaction still verifies; only live-schema
    reuse eligibility failed. :func:`validate_artifact_checkpoint` converts
    this into a per-transaction stale entry instead of failing the whole
    projection: the rung-12 field failure was ONE historical transaction with
    a since-tightened spelling aborting the entire 100-model campaign at
    rehydration.
    """


class ArtifactEventKind(str, Enum):
    """Closed artifact-event transitions from the frozen schema."""

    STAGED_PRIVATE = "staged-private"
    TERMINAL_AUTHORIZED = "terminal-authorized"
    PUBLICATION_AUTHORIZED = "publication-authorized"
    RECONSTRUCTION_COMMITTED = "reconstruction-committed"
    PUBLISHED = "published"
    PRIVATE_COMMITTED = "private-committed"


@dataclass(frozen=True)
class ArtifactInput:
    """One exact source, code, or patch byte presented for private custody.

    Parameters
    ----------
    content, content_sha256:
        Exact bytes and their independently declared digest.
    logical_role, logical_path:
        Model-specific role and safe repository-relative logical location.
    source_id, origin, fetch_recipe:
        Exact frozen source lineage.
    evidence_ids:
        Candidate evidence identities retained as untrusted staging metadata.
    media_type:
        Stable non-empty media type.
    """

    content: bytes
    content_sha256: str
    logical_role: str
    logical_path: str
    source_id: str
    origin: ArtifactOrigin
    fetch_recipe: str
    evidence_ids: tuple[str, ...] = ()
    media_type: str = "application/octet-stream"


@dataclass(frozen=True)
class ArtifactObjectRecord:
    """Canonical typed representation of one persisted mirror-object row."""

    value: MirrorObject

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ArtifactObjectRecord":
        """Parse and identity-check one persisted mirror-object row.

        Parameters
        ----------
        payload:
            Persisted artifact-event object mapping.

        Returns
        -------
        ArtifactObjectRecord
            Canonical typed object record.

        Raises
        ------
        ArtifactBindingError
            If the claimed object ID is not intrinsic to the row.
        """

        obj = MirrorObject(
            object_id=MirrorObjectId(str(payload["object_id"])),
            mirror_class=str(payload["mirror_class"]),
            content_sha256=str(payload["content_sha256"]),
            byte_count=int(payload["byte_count"]),
            media_type=str(payload["media_type"]),
            object_key=str(payload["object_key"]),
        )
        expected = mirror_object_id(
            obj.mirror_class,
            obj.content_sha256,
            obj.byte_count,
            obj.media_type,
            obj.object_key,
        )
        if obj.object_id != expected:
            raise ArtifactBindingError(f"artifact object ID changed: {obj.object_id}")
        return cls(obj)

    def to_payload(self) -> JsonObject:
        """Return the canonical intrinsic mirror-object payload.

        Returns
        -------
        dict[str, Any]
            Artifact-event object fields excluding the event-specific custody key.
        """

        return {
            "object_id": str(self.value.object_id),
            "mirror_class": self.value.mirror_class,
            "content_sha256": self.value.content_sha256,
            "byte_count": self.value.byte_count,
            "media_type": self.value.media_type,
            "object_key": self.value.object_key,
        }


@dataclass(frozen=True)
class ArtifactClaimRecord:
    """Canonical typed representation of one persisted artifact-claim row."""

    value: ArtifactClaim

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ArtifactClaimRecord":
        """Parse and identity-check one persisted artifact-claim row.

        Parameters
        ----------
        payload:
            Persisted artifact-event claim mapping.

        Returns
        -------
        ArtifactClaimRecord
            Canonical typed claim record.

        Raises
        ------
        ArtifactBindingError
            If the claimed ID is not derived from the complete claim.
        """

        claim = ArtifactClaim(
            claim_id=ArtifactClaimId(str(payload["claim_id"])),
            object_id=MirrorObjectId(str(payload["object_id"])),
            stable_id=str(payload["stable_id"]),
            work_id=str(payload["work_id"]),
            proposal_id=str(payload["proposal_id"]),
            gate_id=str(payload["gate_id"]),
            authorization_id=str(payload["authorization_id"]),
            logical_role=str(payload["logical_role"]),
            logical_path=str(payload["logical_path"]),
            source_id=str(payload["source_id"]),
            origin=str(payload["origin"]),
            revision=str(payload["revision"]),
            fetch_recipe_sha256=str(payload["fetch_recipe_sha256"]),
            evidence_ids=tuple(str(value) for value in payload["evidence_ids"]),
            license_disposition=str(payload["license_disposition"]),
        )
        expected = artifact_claim_id(
            object_id=claim.object_id,
            stable_id=claim.stable_id,
            work_id=claim.work_id,
            proposal_id=claim.proposal_id,
            gate_id=claim.gate_id,
            authorization_id=claim.authorization_id,
            logical_role=claim.logical_role,
            logical_path=claim.logical_path,
            source_id=claim.source_id,
            origin=claim.origin,
            revision=claim.revision,
            fetch_recipe_sha256=claim.fetch_recipe_sha256,
            evidence_ids=claim.evidence_ids,
            license_disposition=claim.license_disposition,
        )
        if claim.claim_id != expected:
            raise ArtifactBindingError(f"artifact claim ID changed: {claim.claim_id}")
        return cls(claim)

    def to_payload(self) -> JsonObject:
        """Return the canonical artifact-claim payload.

        Returns
        -------
        dict[str, Any]
            Schema-compatible artifact-event claim row.
        """

        return {
            "claim_id": str(self.value.claim_id),
            "object_id": str(self.value.object_id),
            "stable_id": self.value.stable_id,
            "work_id": self.value.work_id,
            "proposal_id": _dependency_value(self.value.proposal_id),
            "gate_id": _dependency_value(self.value.gate_id),
            "authorization_id": _dependency_value(self.value.authorization_id),
            "logical_role": self.value.logical_role,
            "logical_path": self.value.logical_path,
            "source_id": self.value.source_id,
            "origin": self.value.origin,
            "revision": self.value.revision,
            "fetch_recipe_sha256": self.value.fetch_recipe_sha256,
            "evidence_ids": list(self.value.evidence_ids),
            "license_disposition": self.value.license_disposition,
        }


@dataclass(frozen=True)
class ReconstructionAnchorRecord:
    """Canonical typed representation of one persisted reconstruction anchor."""

    path: str
    sha256: str
    claim_ids: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReconstructionAnchorRecord":
        """Parse one schema-validated reconstruction anchor.

        Parameters
        ----------
        payload:
            Persisted artifact-event reconstruction mapping.

        Returns
        -------
        ReconstructionAnchorRecord
            Normalized immutable anchor fields.
        """

        return cls(
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            claim_ids=tuple(str(value) for value in payload["claim_ids"]),
        )

    def to_payload(self) -> JsonObject:
        """Return the canonical reconstruction-anchor payload.

        Returns
        -------
        dict[str, Any]
            Schema-compatible artifact-event reconstruction fields.
        """

        return {
            "path": self.path,
            "sha256": self.sha256,
            "claim_ids": list(self.claim_ids),
        }


@dataclass(frozen=True)
class ArtifactAuthorityRecord:
    """Canonical typed model reference to one latest artifact-event state."""

    state: str
    transaction_id: str
    committed_event_id: str
    authorization_id: str
    reconstruction_sha256: str
    claim_ids: tuple[str, ...]

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> "ArtifactAuthorityRecord":
        """Derive the canonical model authority reference from one latest event.

        Parameters
        ----------
        event:
            Schema-validated latest artifact event.

        Returns
        -------
        ArtifactAuthorityRecord
            Normalized model checkpoint reference.
        """

        reconstruction = event.get("reconstruction")
        return cls(
            state=str(event["event_kind"]),
            transaction_id=str(event["transaction_id"]),
            committed_event_id=str(event["artifact_event_id"]),
            authorization_id=str(
                event.get("authorization_id") or DependencyState.PENDING_UNTRUSTED.value
            ),
            reconstruction_sha256=(
                ReconstructionAnchorRecord.from_payload(reconstruction).sha256
                if isinstance(reconstruction, Mapping)
                else DependencyState.NOT_APPLICABLE.value
            ),
            claim_ids=tuple(sorted(str(value["claim_id"]) for value in event.get("claims", []))),
        )

    @classmethod
    def not_applicable(cls) -> "ArtifactAuthorityRecord":
        """Return the closed not-applicable model authority reference.

        Returns
        -------
        ArtifactAuthorityRecord
            Canonical typed not-applicable checkpoint reference.
        """

        value = DependencyState.NOT_APPLICABLE.value
        return cls(
            state=value,
            transaction_id=value,
            committed_event_id=value,
            authorization_id=value,
            reconstruction_sha256=value,
            claim_ids=(),
        )

    def to_payload(self) -> JsonObject:
        """Return the canonical model artifact-authority payload.

        Returns
        -------
        dict[str, Any]
            Schema-compatible model checkpoint reference.
        """

        return {
            "state": self.state,
            "transaction_id": self.transaction_id,
            "committed_event_id": self.committed_event_id,
            "authorization_id": self.authorization_id,
            "reconstruction_sha256": self.reconstruction_sha256,
            "claim_ids": list(self.claim_ids),
        }


@dataclass(frozen=True)
class ArtifactCheckpointReference:
    """Canonical typed transaction and claim reference consumed by checkpoint policy."""

    transaction_id: str
    claim_ids: tuple[str, ...]

    @classmethod
    def from_authority_payload(cls, payload: Mapping[str, Any]) -> "ArtifactCheckpointReference":
        """Parse the checkpoint-owned subset of a model authority payload.

        Parameters
        ----------
        payload:
            Model artifact-authority mapping whose local shape guard already passed.

        Returns
        -------
        ArtifactCheckpointReference
            Normalized transaction and claim identities.
        """

        return cls(
            transaction_id=str(payload["transaction_id"]),
            claim_ids=tuple(str(value) for value in payload["claim_ids"]),
        )


@dataclass(frozen=True)
class StagedArtifact:
    """Private-custody transaction produced before any checker authorization."""

    transaction_id: ArtifactTransactionId
    staged_event_id: str
    event: JsonObject
    objects: tuple[MirrorObject, ...]
    custody_claims: tuple[ArtifactClaim, ...]


@dataclass(frozen=True)
class ReconstructionInputs:
    """Exact immutable facts embedded in a reconstruction document."""

    author_result: Mapping[str, Any]
    proposal: Optional[Mapping[str, Any]]
    handoff_execution: Optional[Mapping[str, Any]]
    source_manifest: Mapping[str, Any]
    accepted_gate_item: Mapping[str, Any]


ArtifactProjectionKey: TypeAlias = tuple[str, str, ArtifactTransactionId]


@dataclass(frozen=True)
class ArtifactTransactionProjection:
    """One uniquely verified final artifact transaction in checkpoint authority.

    ``recorded_author_prompt_identity`` and ``recorded_checker_prompt_identity``
    are the transaction's own record-time pins, lifted from its validated
    dependency vector (both are REQUIRED pins -- see
    ``_RECORD_TIME_PINNED_AXES``). Rehydration consumers validate recorded
    history at THIS basis, never at the live prompt files: re-imposing live
    identities on finalized transactions is the rung-11/rung-12 history-vs-live
    family (one checker-prompt re-pin crashed the whole rung-12 campaign at the
    first rehydrated artifact's identity recompute).
    """

    stable_id: str
    work_id: str
    transaction_id: ArtifactTransactionId
    final_event_id: str
    final_event_kind: str
    authorization_id: str
    accepted_gate_id: str
    reconstruction_path: Path
    reconstruction_sha256: str
    reconstruction_inputs: ReconstructionInputs
    objects: tuple[MirrorObject, ...]
    claims: tuple[ArtifactClaim, ...]
    recorded_author_prompt_identity: Optional[str] = None
    recorded_checker_prompt_identity: Optional[str] = None


@dataclass(frozen=True)
class StaleArtifactTransaction:
    """One finalized transaction quarantined out of reusable checkpoint authority.

    A stale entry is a transaction the checkpoint could parse from the
    append-only ledger but must not serve for reuse, for one of two reasons:
    the recorded author result no longer validates against the CURRENT
    author-result schema (live-schema reuse ineligibility -- the solve cache
    is convenience, not history), or the transaction contradicts or fails ITS
    OWN record (vector-vs-record contradiction, unreadable or rewritten
    reconstruction bytes, an interrupted event chain). Either way the entry
    reads as a cache miss for the affected model -- disclosed loudly and
    routed to fresh authorship -- while every OTHER model's authority stays
    alive: the rung-12 crash family was single defective transactions
    aborting whole 100-model campaigns at rehydration.

    ``claims`` retains the transaction's accepted claims (from its own final
    event) so record-anchored consumers (release-checkpoint claim coverage)
    can keep cross-checking recorded history; interrupted chains carry none.
    """

    stable_id: str
    work_id: str
    transaction_id: ArtifactTransactionId
    claims: tuple[ArtifactClaim, ...]
    reason: str


@dataclass(frozen=True)
class ArtifactCheckpointProjection:
    """Read-only normalized object/claim authority shared by later consumers.

    ``transactions`` is indexed by exact ``(stable_id, work_id, transaction_id)``.
    ``objects`` contains one intrinsic row per physical object identity, while
    ``claims`` retains every independent model-specific claim even when digests are
    shared. The future rehydrator and canonical checkpoint must consume this same
    projection instead of independently folding artifact events.
    ``stale_transactions`` carries byte-verified finalized transactions whose
    recorded author results no longer validate against the LIVE author-result
    schema: ineligible for reuse (a cache miss routing the model to fresh
    authorship), still available to record-anchored history consumers.
    """

    transactions: Mapping[ArtifactProjectionKey, ArtifactTransactionProjection]
    objects: tuple[MirrorObject, ...]
    claims: tuple[ArtifactClaim, ...]
    stale_transactions: Mapping[ArtifactProjectionKey, StaleArtifactTransaction] = dataclass_field(
        default_factory=dict
    )
    #: IC-3 typed store-litter disclosure: non-addressable-shaped unattested
    #: ``(mirror_class, object_key)`` files tolerated by inventory
    #: reconciliation. Disclosure only -- litter is never served and never
    #: counts as custody; populated by full validation.
    store_litter: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Defensively freeze the transaction indexes after construction."""

        object.__setattr__(self, "transactions", MappingProxyType(dict(self.transactions)))
        object.__setattr__(
            self, "stale_transactions", MappingProxyType(dict(self.stale_transactions))
        )


@dataclass(frozen=True)
class RehydratedArtifactTransaction:
    """Disposable transaction-addressed materialization of final artifact authority."""

    transaction: ArtifactTransactionProjection
    root: Path
    model_dir: Path
    claim_paths: Mapping[ArtifactClaimId, Path]

    def __post_init__(self) -> None:
        """Defensively freeze the claim-to-staged-path index."""

        object.__setattr__(self, "claim_paths", MappingProxyType(dict(self.claim_paths)))


@dataclass(frozen=True)
class PublishedArtifact:
    """Completed immutable artifact transaction."""

    transaction_id: ArtifactTransactionId
    final_event_id: str
    reconstruction_path: Path
    reconstruction_sha256: str
    event: JsonObject


def _dependency_json(vector: DependencyVector) -> JsonObject:
    """Return one canonical JSON-compatible dependency vector.

    Parameters
    ----------
    vector:
        Frozen reducer-derived dependency vector.

    Returns
    -------
    dict[str, Any]
        Canonical vector payload.
    """

    payload = asdict(vector)
    payload["artifact_claim_ids"] = [str(value) for value in vector.artifact_claim_ids]
    payload["accepted_attempt_ids"] = list(vector.accepted_attempt_ids)
    for key, value in tuple(payload.items()):
        if isinstance(value, DependencyState):
            payload[key] = value.value
    return payload


def _dependency_value(value: DependencyValue) -> str:
    """Normalize a dependency identity or typed state to its string value.

    Parameters
    ----------
    value:
        Frozen dependency value.

    Returns
    -------
    str
        Exact serialized value.
    """

    return value.value if isinstance(value, DependencyState) else value


def mirror_object_id(
    mirror_class: str,
    content_sha256: str,
    byte_count: int,
    media_type: str,
    object_key: str,
) -> MirrorObjectId:
    """Derive the frozen intrinsic physical-object identity.

    Parameters
    ----------
    mirror_class, content_sha256, byte_count, media_type, object_key:
        Exact intrinsic object tuple frozen in ``AuthorityContext`` contracts.

    Returns
    -------
    MirrorObjectId
        Deterministic content-addressed object identity.
    """

    return MirrorObjectId(
        stable_hash(
            {
                "mirror_class": mirror_class,
                "content_sha256": content_sha256,
                "byte_count": byte_count,
                "media_type": media_type,
                "object_key": object_key,
            }
        )
    )


def make_mirror_object(
    *,
    mirror_class: MirrorClass,
    content_sha256: str,
    byte_count: int,
    media_type: str,
    object_key: str,
) -> MirrorObject:
    """Construct one intrinsic mirror object with its recomputed identity.

    Parameters
    ----------
    mirror_class, content_sha256, byte_count, media_type, object_key:
        Exact physical object facts.

    Returns
    -------
    MirrorObject
        Frozen intrinsic object.
    """

    object_id = mirror_object_id(
        mirror_class.value, content_sha256, byte_count, media_type, object_key
    )
    return MirrorObject(
        object_id=object_id,
        mirror_class=mirror_class.value,
        content_sha256=content_sha256,
        byte_count=byte_count,
        media_type=media_type,
        object_key=object_key,
    )


def artifact_claim_id(
    *,
    object_id: MirrorObjectId,
    stable_id: str,
    work_id: str,
    proposal_id: DependencyValue,
    gate_id: DependencyValue,
    authorization_id: DependencyValue,
    logical_role: str,
    logical_path: str,
    source_id: str,
    origin: str,
    revision: str,
    fetch_recipe_sha256: str,
    evidence_ids: Sequence[str],
    license_disposition: str,
) -> ArtifactClaimId:
    """Derive one model-specific claim identity over the complete claim.

    Parameters
    ----------
    object_id, stable_id, work_id, proposal_id, gate_id, authorization_id,
    logical_role, logical_path, source_id, origin, revision,
    fetch_recipe_sha256, evidence_ids, license_disposition:
        Complete frozen claim facts.

    Returns
    -------
    ArtifactClaimId
        Deterministic claim identity independent of object deduplication.
    """

    return ArtifactClaimId(
        stable_hash(
            {
                "object_id": str(object_id),
                "stable_id": stable_id,
                "work_id": work_id,
                "proposal_id": _dependency_value(proposal_id),
                "gate_id": _dependency_value(gate_id),
                "authorization_id": _dependency_value(authorization_id),
                "logical_role": logical_role,
                "logical_path": logical_path,
                "source_id": source_id,
                "origin": origin,
                "revision": revision,
                "fetch_recipe_sha256": fetch_recipe_sha256,
                "evidence_ids": sorted(evidence_ids),
                "license_disposition": license_disposition,
            }
        )
    )


def _make_claim(
    *,
    object_id: MirrorObjectId,
    stable_id: str,
    work_id: str,
    proposal_id: DependencyValue,
    gate_id: DependencyValue,
    authorization_id: DependencyValue,
    logical_role: str,
    logical_path: str,
    source_id: str,
    origin: str,
    revision: str,
    fetch_recipe_sha256: str,
    evidence_ids: Sequence[str],
    license_disposition: str,
) -> ArtifactClaim:
    """Construct one fully identity-bound artifact claim.

    Parameters
    ----------
    object_id, stable_id, work_id, proposal_id, gate_id, authorization_id,
    logical_role, logical_path, source_id, origin, revision,
    fetch_recipe_sha256, evidence_ids, license_disposition:
        Complete claim facts.

    Returns
    -------
    ArtifactClaim
        Frozen normalized claim.
    """

    normalized_evidence = tuple(sorted(evidence_ids))
    claim_id = artifact_claim_id(
        object_id=object_id,
        stable_id=stable_id,
        work_id=work_id,
        proposal_id=proposal_id,
        gate_id=gate_id,
        authorization_id=authorization_id,
        logical_role=logical_role,
        logical_path=logical_path,
        source_id=source_id,
        origin=origin,
        revision=revision,
        fetch_recipe_sha256=fetch_recipe_sha256,
        evidence_ids=normalized_evidence,
        license_disposition=license_disposition,
    )
    return ArtifactClaim(
        claim_id=claim_id,
        object_id=object_id,
        stable_id=stable_id,
        work_id=work_id,
        proposal_id=proposal_id,
        gate_id=gate_id,
        authorization_id=authorization_id,
        logical_role=logical_role,
        logical_path=logical_path,
        source_id=source_id,
        origin=origin,
        revision=revision,
        fetch_recipe_sha256=fetch_recipe_sha256,
        evidence_ids=normalized_evidence,
        license_disposition=license_disposition,
    )


def _object_json(obj: MirrorObject, private_custody_key: str) -> JsonObject:
    """Serialize one frozen object for ``artifact-event.v1``.

    Parameters
    ----------
    obj:
        Intrinsic object.
    private_custody_key:
        Exact private object used as the custody source.

    Returns
    -------
    dict[str, Any]
        Schema-compatible object row.
    """

    payload = ArtifactObjectRecord(obj).to_payload()
    payload["private_custody_key"] = private_custody_key
    return payload


def _claim_json(claim: ArtifactClaim) -> JsonObject:
    """Serialize one frozen claim for ``artifact-event.v1``.

    Parameters
    ----------
    claim:
        Normalized claim.

    Returns
    -------
    dict[str, Any]
        Schema-compatible claim row.
    """

    return ArtifactClaimRecord(claim).to_payload()


def _parse_object(payload: Mapping[str, Any]) -> MirrorObject:
    """Parse and identity-check one artifact-event object.

    Parameters
    ----------
    payload:
        Persisted object row.

    Returns
    -------
    MirrorObject
        Verified intrinsic object.

    Raises
    ------
    ArtifactBindingError
        If the claimed object ID is not intrinsic to the row.
    """

    return ArtifactObjectRecord.from_payload(payload).value


def _parse_claim(payload: Mapping[str, Any]) -> ArtifactClaim:
    """Parse and identity-check one artifact-event claim.

    Parameters
    ----------
    payload:
        Persisted claim row.

    Returns
    -------
    ArtifactClaim
        Verified normalized claim.

    Raises
    ------
    ArtifactBindingError
        If the claimed ID is not derived from the complete claim.
    """

    return ArtifactClaimRecord.from_payload(payload).value


def _safe_relative_path(value: str) -> Path:
    """Normalize one repository-relative logical path.

    Parameters
    ----------
    value:
        Candidate POSIX path.

    Returns
    -------
    pathlib.Path
        Safe relative path.

    Raises
    ------
    ArtifactBindingError
        If the value is empty, absolute, or traversing.
    """

    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ArtifactBindingError(f"unsafe artifact logical path: {value!r}")
    return Path(*pure.parts)


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    """Hash a self-identifying mapping without its digest field.

    Parameters
    ----------
    payload:
        Mapping containing a self digest.
    field:
        Digest field to omit.

    Returns
    -------
    str
        Canonical SHA-256 identity.
    """

    return stable_hash({key: value for key, value in payload.items() if key != field})


@dataclass(frozen=True)
class _AuthorResultBindingBasis:
    """Exact authority basis one typed author result validates against.

    Two constructors, two different truths, one validator. At WRITE time
    (:func:`stage_private_artifact`, :func:`_validate_reconstruction_inputs`)
    the basis is the LIVE active authority context: a result being committed
    now must bind the prompt, dispatcher, and intake that are active now. At
    READ time (:func:`_validate_reconstruction_document`) the basis is the
    transaction's OWN RECORDED commitments: prompt/dispatcher/intake
    identities are content hashes of shipped bytes that legitimately move
    between legs, and re-imposing the live values on append-only history
    permanently poisoned every finalized transaction after an upgrade (the
    rung-11 history-vs-live family -- one author-prompt re-pin away from a
    campaign abort). ``author_model_identity`` is ``None`` at read time
    because no recorded axis pins it independently; the recorded value is
    already committed inside the self-hashed result whose ``result_id`` the
    final event names, so presence is required and equality is meaningless.
    """

    intake_snapshot_id: str
    intake_snapshot_sha256: str
    intake_item_sha256: str
    author_prompt_identity: str
    author_dispatcher_identity: str
    author_model_identity: Optional[str]


def _live_context_basis(context: AuthorityContext, stable_id: str) -> _AuthorResultBindingBasis:
    """Build the write-time binding basis from the LIVE authority context.

    Parameters
    ----------
    context, stable_id:
        Active trust roots and the model being committed.

    Returns
    -------
    _AuthorResultBindingBasis
        Live basis for a result being staged or finalized now.

    Raises
    ------
    ArtifactBindingError
        If the model is absent from the active intake.
    """

    intake_item = context.intake_by_stable_id.get(stable_id)
    if intake_item is None:
        raise ArtifactBindingError(f"stable ID is absent from active intake: {stable_id}")
    return _AuthorResultBindingBasis(
        intake_snapshot_id=context.active_intake_snapshot_id,
        intake_snapshot_sha256=context.active_intake_snapshot_sha256,
        intake_item_sha256=stable_hash(intake_item),
        author_prompt_identity=context.author_prompt_identity,
        author_dispatcher_identity=context.author_dispatcher_identity,
        author_model_identity=context.author_model_identity,
    )


def _recorded_transaction_basis(
    document: Mapping[str, Any], final_event: Mapping[str, Any]
) -> _AuthorResultBindingBasis:
    """Build the read-time binding basis from the transaction's OWN record.

    Every value is drawn from anchors the ledger already commits: the final
    event's recorded intake snapshot identity, the reconstruction document's
    recorded intake item (internally re-hashed), and the recorded dependency
    vector's prompt/dispatcher pins -- re-verified here against the final
    event's ``dependency_vector_sha256`` so this basis never depends on call
    order for its authenticity. A record contradicting its own commitments
    still refuses; a record whose rule identities were legitimately superseded
    validates at the identities it was recorded under.

    Parameters
    ----------
    document, final_event:
        Byte-anchored reconstruction document and its final ledger event.

    Returns
    -------
    _AuthorResultBindingBasis
        Recorded basis for validating append-only history.

    Raises
    ------
    ArtifactCheckpointError
        If the recorded vector, intake item, or pins contradict the record.
    """

    vector = document.get("dependency_vector")
    if (
        not isinstance(vector, Mapping)
        or stable_hash(vector) != final_event["dependency_vector_sha256"]
    ):
        raise ArtifactCheckpointError(
            "reconstruction dependency vector does not match its ledger commitment"
        )
    intake_item = document.get("intake_item")
    intake_item_sha256 = document.get("intake_item_sha256")
    if (
        not isinstance(intake_item, Mapping)
        or not isinstance(intake_item_sha256, str)
        or stable_hash(intake_item) != intake_item_sha256
        or vector.get("intake_item_sha256") != intake_item_sha256
    ):
        raise ArtifactCheckpointError(
            "reconstruction intake item contradicts its recorded identity"
        )
    pins: dict[str, str] = {}
    for axis in ("author_prompt_identity", "author_dispatcher_identity"):
        observed = vector.get(axis)
        if not isinstance(observed, str) or not observed:
            raise ArtifactCheckpointError(
                f"reconstruction dependency vector lacks its recorded pin at {axis}"
            )
        pins[axis] = observed
    return _AuthorResultBindingBasis(
        intake_snapshot_id=str(final_event["intake_snapshot_id"]),
        intake_snapshot_sha256=str(final_event["intake_snapshot_sha256"]),
        intake_item_sha256=intake_item_sha256,
        author_prompt_identity=pins["author_prompt_identity"],
        author_dispatcher_identity=pins["author_dispatcher_identity"],
        author_model_identity=None,
    )


def _validate_context_result(
    basis: _AuthorResultBindingBasis,
    stable_id: str,
    work_id: str,
    author_result: Mapping[str, Any],
    proposal: Optional[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
) -> tuple[
    str,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[Mapping[str, Any]],
    str,
    tuple[str, ...],
]:
    """Validate typed author, proposal, source, and intake bindings.

    Parameters
    ----------
    basis, stable_id, work_id, author_result, proposal, source_manifest:
        Binding basis (live at write time, recorded at read time -- see
        :class:`_AuthorResultBindingBasis`) and exact frozen transaction
        inputs.

    Returns
    -------
    tuple[str, str | None, str | None, str | None, Mapping | None, str, tuple[str, ...]]
        Result ID, ordinary proposal ID, handoff proposal/hash/proposal, source-manifest
        identity, and exact source set.

    Raises
    ------
    ArtifactBindingError
        If any identity is stale, partial, or inconsistent.
    """

    try:
        validate_payload(author_result, AUTHOR_RESULT_SCHEMA_VERSION)
    except PayloadValidationError as exc:
        raise AuthorResultSchemaDriftError(f"invalid typed author result: {exc}") from exc
    if author_result.get("stable_id") != stable_id or author_result.get("work_id") != work_id:
        raise ArtifactBindingError("author result stable/work binding changed")
    if author_result.get("result_sha256") != _self_hash(author_result, "result_sha256"):
        raise ArtifactBindingError("author result self hash changed")
    author_identity = author_result.get("author_identity")
    author_identity_bound = (
        author_identity == basis.author_model_identity
        if basis.author_model_identity is not None
        else isinstance(author_identity, str) and bool(author_identity)
    )
    if (
        author_result.get("intake_snapshot_id") != basis.intake_snapshot_id
        or author_result.get("intake_snapshot_sha256") != basis.intake_snapshot_sha256
        or author_result.get("intake_item_sha256") != basis.intake_item_sha256
        or author_result.get("prompt_identity") != basis.author_prompt_identity
        or not author_identity_bound
        or author_result.get("dispatcher_identity") != basis.author_dispatcher_identity
    ):
        raise ArtifactBindingError("author result is stale against active authority context")
    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or any(not isinstance(row, Mapping) for row in sources):
        raise SourceManifestBindingError("source manifest requires object rows")
    source_identity = stable_hash(sources)
    # Two checks, deliberately not one: the manifest disagreeing with its own
    # rows is the ENGINE's fault, while the author's echo disagreeing is the
    # AUTHOR's. Collapsing them made an engine defect indistinguishable from a
    # fabricated echo, and every such engine defect was then recorded against
    # the author as a crashed session.
    if source_manifest.get("manifest_sha256") != source_identity:
        raise SourceManifestBindingError("source manifest identity changed")
    if author_result.get("source_manifest_identity") != source_identity:
        raise ArtifactBindingError("author result quotes a different source manifest identity")
    # CITABLE, not frozen: `source_identity` above is the identity question and
    # stays on `sources` alone, but "which sources may this proposal name?" is a
    # coverage question, and the answer includes the ONE supplementary broker
    # pack the executor may grant mid-session. The author is told those rows are
    # citable exactly like frozen ones, and it is OUR fetch and OUR digest -- so
    # echoing one back is compliance, not fabrication. Reading `sources` alone
    # here refused a correct proposal for quoting a document we handed it, which
    # is the same defect `manifest_source_rows` was introduced to close for byte
    # custody (see `models.SUPPLEMENTARY_SOURCES_KEY`).
    citable_rows = manifest_source_rows(source_manifest)
    source_ids = tuple(sorted(str(row.get("source_id")) for row in citable_rows))
    if any(value in {"", "None"} for value in source_ids) or len(set(source_ids)) != len(
        source_ids
    ):
        raise SourceManifestBindingError("source manifest IDs must be unique and non-empty")

    kind = author_result.get("kind")
    payload = author_result.get("payload")
    embedded = payload.get("proposal") if isinstance(payload, Mapping) else None
    proposal_id: Optional[str] = None
    handoff_proposal_id: Optional[str] = None
    handoff_sha256: Optional[str] = None
    handoff_proposal: Optional[Mapping[str, Any]] = None
    if kind == "PROPOSED":
        if not isinstance(proposal, Mapping) or not isinstance(embedded, Mapping):
            raise ArtifactBindingError("PROPOSED author result requires its exact proposal")
        if dict(proposal) != dict(embedded):
            raise ArtifactBindingError("supplied proposal differs from typed author result")
        try:
            validate_payload(proposal, AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
        except PayloadValidationError as exc:
            raise ArtifactBindingError(f"invalid author proposal: {exc}") from exc
        if (
            proposal.get("proposal_sha256") != _self_hash(proposal, "proposal_sha256")
            or proposal.get("stable_id") != stable_id
            or proposal.get("work_id") != work_id
            or proposal.get("intake_snapshot_id") != basis.intake_snapshot_id
            or proposal.get("intake_snapshot_sha256") != basis.intake_snapshot_sha256
            or proposal.get("intake_item_sha256") != basis.intake_item_sha256
            or proposal.get("source_manifest_identity") != source_identity
            or proposal.get("dispatcher_identity") != basis.author_dispatcher_identity
            or not isinstance(proposal.get("author"), Mapping)
            or proposal.get("author", {}).get("prompt_sha256") != basis.author_prompt_identity
        ):
            raise ArtifactBindingError("proposal is stale against transaction authority")
        proposed_facts = proposal.get("proposed_facts")
        resolution = (
            proposed_facts.get("source_resolution") if isinstance(proposed_facts, Mapping) else None
        )
        proposed_sources = resolution.get("sources") if isinstance(resolution, Mapping) else None
        if not isinstance(proposed_sources, list):
            raise ArtifactBindingError("proposal lacks its exact source set")
        proposed_by_id = {
            str(row.get("source_id")): row for row in proposed_sources if isinstance(row, Mapping)
        }
        if set(proposed_by_id) != set(source_ids):
            raise ArtifactBindingError("proposal and source manifest source sets differ")
        # Every citable row is field-checked, supplementary ones included: the
        # widened set above may only ADD an obligation to echo our exact bytes,
        # never remove one.
        for fetched in citable_rows:
            authored = proposed_by_id[str(fetched["source_id"])]
            expected_size = fetched.get("fetched_bytes_len", fetched.get("byte_count"))
            retrieved_at_matches = "retrieved_at" not in fetched or (
                authored.get("retrieved_at") == fetched.get("retrieved_at")
            )
            if (
                authored.get("url") != fetched.get("url")
                or authored.get("revision") != fetched.get("revision")
                or authored.get("content_sha256") != fetched.get("content_sha256")
                or authored.get("byte_count") != expected_size
                or authored.get("media_type") != fetched.get("media_type")
                or not retrieved_at_matches
            ):
                raise ArtifactBindingError(
                    "proposal source fields differ from controlled-fetch manifest"
                )
        proposal_id = str(proposal["proposal_id"])
    elif kind == "DEFER_RECOMMENDATION":
        if proposal is not None or embedded is not None or not isinstance(payload, Mapping):
            raise ArtifactBindingError("terminal deferral cannot masquerade as PROPOSED")
        handoff = payload.get("handoff_execution")
        candidate = handoff.get("proposal") if isinstance(handoff, Mapping) else None
        if not isinstance(handoff, Mapping) or not isinstance(candidate, Mapping):
            raise ArtifactBindingError("terminal deferral lacks executable handoff authority")
        try:
            validate_payload(candidate, AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)
        except PayloadValidationError as exc:
            raise ArtifactBindingError(f"invalid handoff proposal: {exc}") from exc
        implementation = candidate.get("proposed_facts", {}).get("implementation", {})
        code_manifest = (
            implementation.get("code_manifest") or [] if isinstance(implementation, Mapping) else []
        )
        handoff_basis = {
            "proposal": candidate,
            "proposal_sha256": handoff.get("proposal_sha256"),
            "code_manifest_identity": handoff.get("code_manifest_identity"),
            "source_manifest_identity": handoff.get("source_manifest_identity"),
        }
        if (
            candidate.get("proposal_sha256") != handoff.get("proposal_sha256")
            or candidate.get("stable_id") != stable_id
            or candidate.get("work_id") != work_id
            or candidate.get("source_manifest_identity") != source_identity
            or candidate.get("intake_snapshot_id") != basis.intake_snapshot_id
            or candidate.get("intake_snapshot_sha256") != basis.intake_snapshot_sha256
            or candidate.get("intake_item_sha256") != basis.intake_item_sha256
            or candidate.get("dispatcher_identity") != basis.author_dispatcher_identity
            or not isinstance(code_manifest, list)
            or stable_hash(code_manifest) != handoff.get("code_manifest_identity")
            or handoff.get("source_manifest_identity") != source_identity
            or handoff.get("handoff_sha256") != stable_hash(handoff_basis)
        ):
            raise ArtifactBindingError("terminal executable handoff binding changed")
        handoff_proposal = candidate
        handoff_proposal_id = str(candidate["proposal_id"])
        handoff_sha256 = str(handoff["handoff_sha256"])
    elif proposal is not None or embedded is not None:
        raise ArtifactBindingError("non-proposed author result cannot carry a proposal")
    return (
        str(author_result["result_id"]),
        proposal_id,
        handoff_proposal_id,
        handoff_sha256,
        handoff_proposal,
        source_identity,
        source_ids,
    )


def _validate_artifact_inputs(
    artifacts: Sequence[ArtifactInput],
    source_manifest: Mapping[str, Any],
    proposal: Optional[Mapping[str, Any]],
) -> None:
    """Verify every staged byte and its exact source row.

    Parameters
    ----------
    artifacts:
        Complete private-stage byte set.
    source_manifest:
        Frozen controlled-fetch manifest.
    proposal:
        Exact typed proposal when code or patch bytes are present.

    Raises
    ------
    ArtifactBindingError
        If bytes, paths, origins, roles, or source coverage diverge.
    """

    if not artifacts:
        raise ArtifactBindingError("private staging requires at least one artifact byte")
    raw_sources = source_manifest.get("sources")
    if not isinstance(raw_sources, list):
        raise SourceManifestBindingError("source manifest requires source rows")
    # BYTE custody covers the supplementary rows too. This is the coverage
    # question ("is every staged byte a source we fetched, and is every source we
    # fetched staged?"), not the identity question -- `bind_author_artifact`
    # still derives `source_identity` from `sources` alone, so widening here
    # cannot move a bound identity.
    sources = {str(row.get("source_id")): row for row in manifest_source_rows(source_manifest)}
    source_artifact_ids: set[str] = set()
    seen_paths: dict[str, str] = {}
    for artifact in artifacts:
        _safe_relative_path(artifact.logical_path)
        if artifact.logical_role not in {"source", "code", "patch"}:
            raise ArtifactBindingError(f"unsupported artifact role: {artifact.logical_role}")
        if not artifact.media_type.strip() or not artifact.fetch_recipe.strip():
            raise ArtifactBindingError("artifact media type and fetch recipe must be non-empty")
        actual = hash_bytes(artifact.content)
        if actual != artifact.content_sha256:
            raise ArtifactBindingError(
                f"artifact digest changed at {artifact.logical_path}: {actual}"
            )
        previous = seen_paths.setdefault(artifact.logical_path, actual)
        if previous != actual:
            raise ArtifactBindingError(
                f"logical artifact path has conflicting bytes: {artifact.logical_path}"
            )
        source = sources.get(artifact.source_id)
        if source is None:
            raise ArtifactBindingError(
                f"artifact references a source outside the frozen set: {artifact.source_id}"
            )
        if (
            source.get("url") != artifact.origin.url
            or source.get("revision") != artifact.origin.revision
        ):
            raise ArtifactBindingError("artifact origin differs from frozen source row")
        if artifact.logical_role == "source":
            source_artifact_ids.add(artifact.source_id)
            expected_size = source.get("fetched_bytes_len", source.get("byte_count"))
            if (
                source.get("content_sha256") != actual
                or expected_size not in {None, len(artifact.content)}
                or source.get("media_type") not in {None, artifact.media_type}
            ):
                raise ArtifactBindingError("staged source bytes differ from frozen source row")
    byte_backed_ids = {
        source_id
        for source_id, row in sources.items()
        if isinstance(row.get("content_sha256"), str)
    }
    if source_artifact_ids != byte_backed_ids:
        raise ArtifactBindingError(
            "private staging requires exact source-byte coverage: "
            f"expected={sorted(byte_backed_ids)} observed={sorted(source_artifact_ids)}"
        )
    _validate_declared_code_inputs(artifacts, proposal)


def _validate_declared_code_inputs(
    artifacts: Sequence[ArtifactInput], proposal: Optional[Mapping[str, Any]]
) -> None:
    """Require exact proposal coverage for every staged code and patch byte.

    Parameters
    ----------
    artifacts:
        Complete private-stage byte set.
    proposal:
        Exact typed proposal, or ``None`` for terminal recommendation lanes.

    Raises
    ------
    ArtifactBindingError
        If code/patch bytes lack a proposal or differ from its closed manifests.
    """

    code_inputs = tuple(
        artifact for artifact in artifacts if artifact.logical_role in {"code", "patch"}
    )
    if proposal is None:
        if code_inputs:
            raise ArtifactBindingError("code or patch bytes require an exact typed proposal")
        return
    proposed_facts = proposal.get("proposed_facts")
    implementation = (
        proposed_facts.get("implementation") if isinstance(proposed_facts, Mapping) else None
    )
    if not isinstance(implementation, Mapping):
        raise ArtifactBindingError("proposal implementation is malformed")

    def declared_rows(field: str) -> tuple[tuple[str, str], ...]:
        """Return normalized declared path/digest rows for one manifest field."""

        raw = implementation.get(field)
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise ArtifactBindingError(f"proposal {field} must be an array")
        rows: list[tuple[str, str]] = []
        for value in raw:
            if (
                not isinstance(value, Mapping)
                or not isinstance(value.get("path"), str)
                or not isinstance(value.get("sha256"), str)
            ):
                raise ArtifactBindingError(f"proposal {field} row is malformed")
            rows.append((str(value["path"]), str(value["sha256"])))
        return tuple(sorted(rows))

    for role, field in (("code", "code_manifest"), ("patch", "patches")):
        declared = declared_rows(field)
        observed = tuple(
            sorted(
                (artifact.logical_path, artifact.content_sha256)
                for artifact in artifacts
                if artifact.logical_role == role
            )
        )
        unmatched = list(observed)
        for declared_path, declared_digest in declared:
            declared_parts = PurePosixPath(declared_path).parts
            matches = [
                row
                for row in unmatched
                if PurePosixPath(row[0]).parts[-len(declared_parts) :] == declared_parts
                and row[1] == declared_digest
            ]
            if len(matches) != 1:
                raise ArtifactBindingError(
                    f"staged {role} bytes differ from proposal {field}: {declared_path}"
                )
            unmatched.remove(matches[0])
        if unmatched:
            raise ArtifactBindingError(f"staged {role} bytes exceed proposal {field}")


def _transaction_id(
    *,
    stable_id: str,
    work_id: str,
    result_id: str,
    proposal_id: Optional[str],
    handoff_proposal_id: Optional[str],
    handoff_sha256: Optional[str],
    source_manifest_identity: str,
    source_ids: Sequence[str],
    objects: Sequence[MirrorObject],
    claims: Sequence[ArtifactClaim],
    context: AuthorityContext,
) -> ArtifactTransactionId:
    """Derive one immutable private-stage transaction identity.

    Parameters
    ----------
    stable_id, work_id, result_id, proposal_id, handoff_proposal_id, handoff_sha256,
    source_manifest_identity,
    source_ids, objects, claims, context:
        Complete private-stage authority basis.

    Returns
    -------
    ArtifactTransactionId
        Deterministic transaction identity.
    """

    return ArtifactTransactionId(
        stable_hash(
            {
                "stable_id": stable_id,
                "work_id": work_id,
                "author_result_id": result_id,
                "proposal_id": proposal_id,
                "handoff_proposal_id": handoff_proposal_id,
                "handoff_sha256": handoff_sha256,
                "source_manifest_identity": source_manifest_identity,
                "source_ids": sorted(source_ids),
                "object_ids": sorted(str(value.object_id) for value in objects),
                "custody_claim_ids": sorted(str(value.claim_id) for value in claims),
                "intake_snapshot_id": context.active_intake_snapshot_id,
                "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            }
        )
    )


def _event_id(payload: Mapping[str, Any]) -> str:
    """Derive one deterministic artifact-event identity.

    Parameters
    ----------
    payload:
        Event without append-assigned fields.

    Returns
    -------
    str
        Deterministic transition identity independent of wall-clock metadata.
    """

    omitted = {"artifact_event_id", "created_at", "ledger_seq", "payload_sha256"}
    return stable_hash({key: value for key, value in payload.items() if key not in omitted})


def _semantic_event(payload: Mapping[str, Any]) -> JsonObject:
    """Return event facts relevant to idempotent transition comparison.

    Parameters
    ----------
    payload:
        Proposed or persisted event.

    Returns
    -------
    dict[str, Any]
        Event without append/time metadata.
    """

    omitted = {"created_at", "ledger_seq", "payload_sha256"}
    return {key: deepcopy(value) for key, value in payload.items() if key not in omitted}


def _event_payload(
    *,
    transaction_id: ArtifactTransactionId,
    predecessor_event_id: Optional[str],
    event_kind: ArtifactEventKind,
    created_at: str,
    stable_id: str,
    work_id: str,
    context: AuthorityContext,
    author_result_id: str,
    proposal_id: Optional[str],
    handoff_proposal_id: Optional[str],
    handoff_sha256: Optional[str],
    source_manifest_identity: str,
    source_ids: Sequence[str],
    objects: Sequence[tuple[MirrorObject, str]],
    claims: Sequence[ArtifactClaim],
    gate_id: Optional[str] = None,
    gate_item_sha256: Optional[str] = None,
    dependency_vector_sha256: Optional[str] = None,
    authorization_id: Optional[str] = None,
    reconstruction: Optional[Mapping[str, Any]] = None,
    publication_inventory: Optional[Mapping[str, Any]] = None,
) -> JsonObject:
    """Build one complete logical artifact event.

    Parameters
    ----------
    transaction_id, predecessor_event_id, event_kind, created_at, stable_id,
    work_id, context, author_result_id, proposal_id, handoff_proposal_id, handoff_sha256,
    source_manifest_identity, source_ids, objects, claims, gate_id,
    gate_item_sha256, dependency_vector_sha256, authorization_id,
    reconstruction, publication_inventory:
        Complete transition facts.

    Returns
    -------
    dict[str, Any]
        Schema-ready logical event.
    """

    payload: JsonObject = {
        "schema_version": ARTIFACT_EVENT_SCHEMA_VERSION,
        "transaction_id": str(transaction_id),
        "predecessor_event_id": predecessor_event_id,
        "event_kind": event_kind.value,
        "created_at": created_at,
        "stable_id": stable_id,
        "work_id": work_id,
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "author_result_id": author_result_id,
        "proposal_id": proposal_id,
        "handoff_proposal_id": handoff_proposal_id,
        "handoff_sha256": handoff_sha256,
        "source_manifest_identity": source_manifest_identity,
        "source_ids": sorted(source_ids),
        "objects": [
            _object_json(obj, custody_key)
            for obj, custody_key in sorted(objects, key=lambda value: str(value[0].object_id))
        ],
        "claims": [
            _claim_json(claim) for claim in sorted(claims, key=lambda value: str(value.claim_id))
        ],
        "gate_id": gate_id,
        "gate_item_sha256": gate_item_sha256,
        "dependency_vector_sha256": dependency_vector_sha256,
        "authorization_id": authorization_id,
        "reconstruction": dict(reconstruction) if reconstruction is not None else None,
        "publication_inventory": (
            dict(publication_inventory) if publication_inventory is not None else None
        ),
    }
    # W3 write-side producer (artifact-event mint): stamped BEFORE the
    # intrinsic event identity below, so the marker is inside
    # ``artifact_event_id`` (chain validation recomputes that identity from
    # content on every read) and cannot be stripped without invalidating it.
    stamp_shadow_rehearsal(payload)
    payload["artifact_event_id"] = _event_id(payload)
    return payload


class ArtifactEventLedger:
    """Locked append-only writer and validator for artifact transactions."""

    def __init__(self, path: Path, *, recover_tail: bool = True) -> None:
        """Open and validate one artifact-event shard.

        Parameters
        ----------
        path:
            Canonical ``records/artifacts/<shard>.jsonl`` path.
        recover_tail:
            Whether ordinary evidenced torn-tail recovery is enabled.
        """

        self._ledger = JsonlLedger(path, ARTIFACT_EVENT_SCHEMA_VERSION, recover_tail=recover_tail)
        # The full chain validation (fork/reorder/identity tripwires) runs on
        # every open; its per-event jsonschema pass is skipped because the
        # JsonlLedger open one line above just proved every record
        # schema-valid (live validated scan, or a prior validated open whose
        # prefix digest still holds).
        validate_artifact_event_chains(tuple(self._ledger.iter_records()), validate_schema=False)
        self._final_work_ids = {
            (str(event["stable_id"]), str(event["work_id"]))
            for event in self._ledger.iter_records()
            if event.get("event_kind")
            in {
                ArtifactEventKind.PUBLISHED.value,
                ArtifactEventKind.PRIVATE_COMMITTED.value,
            }
        }
        self._staged_roots: dict[tuple[str, str, str], list[JsonObject]] = {}
        for event in self._ledger.iter_records():
            self._index_staged_root(event)
        self._validation_memo: Optional["ValidationMemo"] = None

    def bind_validation_memo(self, memo: Optional["ValidationMemo"]) -> None:
        """Bind the persistent validation memo for append-time revalidation.

        The post-append whole-ledger chain validation keeps running in full;
        the memo only removes redundant jsonschema passes over records whose
        content identity re-verifies live. Binding ``None`` restores fully
        uncached behavior.

        Parameters
        ----------
        memo:
            Positive-outcome validation memo, or ``None``.
        """

        self._validation_memo = memo

    def _index_staged_root(self, event: Mapping[str, Any]) -> None:
        """Index one staged-private root for O(1) result-binding recovery.

        Parameters
        ----------
        event:
            Persisted artifact event.
        """

        if event.get("event_kind") != ArtifactEventKind.STAGED_PRIVATE.value:
            return
        key = (
            str(event.get("stable_id")),
            str(event.get("work_id")),
            str(event.get("author_result_id")),
        )
        self._staged_roots.setdefault(key, []).append(dict(event))

    def staged_events(
        self, *, stable_id: str, work_id: str, author_result_id: str
    ) -> tuple[JsonObject, ...]:
        """Return every staged-private root for one exact result binding.

        Parameters
        ----------
        stable_id, work_id, author_result_id:
            Exact immutable result association.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Defensive copies of matching staged events, in ledger order.
        """

        matches = self._staged_roots.get((stable_id, work_id, author_result_id), [])
        return tuple(deepcopy(event) for event in matches)

    def __enter__(self) -> ArtifactEventLedger:
        """Return this locked ledger.

        Returns
        -------
        ArtifactEventLedger
            This ledger.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the underlying single-writer ledger.

        Parameters
        ----------
        exc_type, exc_value, traceback:
            Context-manager exception state.
        """

        self.close()

    @property
    def path(self) -> Path:
        """Return the canonical ledger path.

        Returns
        -------
        pathlib.Path
            Artifact ledger path.
        """

        return self._ledger.path

    @property
    def events(self) -> tuple[JsonObject, ...]:
        """Return defensive event copies in ledger order.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Persisted artifact events.
        """

        return tuple(self._ledger.records)

    @property
    def event_count(self) -> int:
        """Return the append generation used by validation caches.

        Returns
        -------
        int
            Number of persisted artifact events.
        """

        return self._ledger.record_count

    def iter_events(self) -> Iterator[JsonObject]:
        """Iterate persisted events WITHOUT defensive copies.

        For read-only internal consumers on hot paths: the whole-ledger
        deepcopy behind :attr:`events` is O(history) per access and was a
        measured per-append cost in the reducer (T1-12). Callers must never
        mutate the yielded events.

        Returns
        -------
        Iterator[dict[str, Any]]
            Live events in ledger order.
        """

        return self._ledger.iter_records()

    def has_final_event(self, stable_id: str, work_id: str) -> bool:
        """Return whether exact finalized authority exists in this shard.

        Parameters
        ----------
        stable_id, work_id:
            Exact model and work-generation identities.

        Returns
        -------
        bool
            Whether a published or private-committed event exists.
        """

        return (stable_id, work_id) in self._final_work_ids

    def close(self) -> None:
        """Release the underlying writer lock idempotently."""

        self._ledger.close()

    def transaction_events(self, transaction_id: ArtifactTransactionId) -> tuple[JsonObject, ...]:
        """Return one transaction's predecessor chain in ledger order.

        Parameters
        ----------
        transaction_id:
            Exact transaction identity.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Matching events.
        """

        return self._ledger.records_for_field("transaction_id", str(transaction_id))

    def append_event(self, payload: Mapping[str, Any]) -> AppendResult:
        """Append one idempotent state-machine transition.

        Parameters
        ----------
        payload:
            Complete logical ``artifact-event.v1`` payload.

        Returns
        -------
        AppendResult
            Persisted transition and append status.

        Raises
        ------
        ArtifactTransitionError
            If the transition forks or skips the predecessor chain.
        """

        # W3 write-side producer (artifact-event door): the marker must be
        # sealed inside ``artifact_event_id`` at the mint (_event_payload) --
        # stamping here, after the id, would mint a durable row whose
        # intrinsic identity every later chain validation refuses. So the
        # door REFUSES an unmarked event while the mode is active instead of
        # silently stamping. Runtime import mirrors identity's own seam note.
        from menagerie.crawler.checker_calibration import shadow_rehearsal_run_active

        if shadow_rehearsal_run_active() and not is_shadow_rehearsal_record(payload):
            raise ArtifactTransitionError(
                "shadow-rehearsal mode refuses an unmarked artifact event: the "
                "shadow_rehearsal marker must be minted into the event before its "
                "artifact_event_id is derived, not appended after"
            )
        event_id = payload.get("artifact_event_id")
        for existing in self._ledger.records_for_field("artifact_event_id", event_id):
            if _semantic_event(existing) != _semantic_event(payload):
                raise ArtifactTransitionError(f"conflicting artifact event replay: {event_id}")
            return AppendResult(deepcopy(existing), appended=False)
        transaction_id = str(payload.get("transaction_id"))
        existing_chain = self._ledger.records_for_field("transaction_id", transaction_id)
        # AT-2 spent/closed-chain split: a NEW transition onto a chain that
        # already carries its final event is the ONE recoverable refusal --
        # typed BEFORE the generic chain-shape validation so recovery can
        # route it without consuming corruption classes, and BEFORE any
        # append so the retry never persists a fifth event.
        final_kinds = {
            ArtifactEventKind.PUBLISHED.value,
            ArtifactEventKind.PRIVATE_COMMITTED.value,
        }
        final_event = next(
            (event for event in existing_chain if str(event.get("event_kind")) in final_kinds),
            None,
        )
        if final_event is not None:
            raise ArtifactChainSpentError(
                transaction_id=transaction_id,
                stable_id=final_event.get("stable_id"),
                work_id=final_event.get("work_id"),
                final_event_kind=str(final_event.get("event_kind")),
                attempted_event_kind=payload.get("event_kind"),
            )
        proposed = (*existing_chain, dict(payload))
        validate_artifact_event_chains(proposed, validate_schema=False)
        result = self._ledger.append(payload)
        validate_artifact_event_chains(
            self._ledger.records_for_field("transaction_id", transaction_id),
            memo=self._validation_memo,
        )
        if result.appended and result.record.get("event_kind") in {
            ArtifactEventKind.PUBLISHED.value,
            ArtifactEventKind.PRIVATE_COMMITTED.value,
        }:
            self._final_work_ids.add(
                (str(result.record["stable_id"]), str(result.record["work_id"]))
            )
        if result.appended:
            self._index_staged_root(result.record)
        return result

    def events_since(self, index: int) -> tuple[JsonObject, ...]:
        """Return defensive copies of the exact appended suffix.

        Parameters
        ----------
        index:
            Append generation already consumed by the caller.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Events appended after ``index``, in ledger order.
        """

        return self._ledger.records_since(index)

    def events_for_stable_id(self, stable_id: str) -> tuple[JsonObject, ...]:
        """Return only artifact events indexed to one stable model.

        Parameters
        ----------
        stable_id:
            Exact model identity.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Matching artifact events in ledger order.
        """

        return self._ledger.records_for_stable_id(stable_id)


def validate_artifact_event_chains(
    events: Sequence[Mapping[str, Any]],
    *,
    validate_schema: bool = True,
    memo: Optional["ValidationMemo"] = None,
) -> None:
    """Validate every predecessor-linked artifact transaction.

    Parameters
    ----------
    events:
        Artifact events in canonical ledger order.
    validate_schema:
        Whether to validate complete persisted schema fields.  Logical events
        awaiting ledger sequence/hash assignment use ``False``.
    memo:
        Optional positive-outcome validation memo. A memoized event skips
        only the jsonschema pass; its intrinsic ``artifact_event_id``
        identity is still recomputed and compared live below, and its
        ``payload_sha256`` was re-verified on ledger read.

    Raises
    ------
    ArtifactTransitionError
        If an event forks, reorders, repeats, or mutates transaction identity.
    """

    by_transaction: dict[str, list[Mapping[str, Any]]] = {}
    seen_event_ids: set[str] = set()
    for event in events:
        if validate_schema:
            # The memo key is trusted only after re-proving the digest against
            # THIS event's bytes: hits and records both require the live
            # content-identity check, so the memo can never launder foreign
            # content under a borrowed payload_sha256.
            memo_key: Optional[str] = None
            if memo is not None:
                candidate = record_validation_memo_key(event)
                if candidate is not None and event.get("payload_sha256") == payload_hash(event):
                    memo_key = candidate
            try:
                if not (
                    memo_key is not None
                    and memo is not None
                    and memo.seen("record", memo_key)
                ):
                    validate_payload(event, ARTIFACT_EVENT_SCHEMA_VERSION)
                    if memo_key is not None and memo is not None:
                        memo.record("record", memo_key)
            except PayloadValidationError as exc:
                raise ArtifactTransitionError(f"invalid artifact event: {exc}") from exc
        event_id = str(event.get("artifact_event_id"))
        if event_id in seen_event_ids or event_id != _event_id(event):
            raise ArtifactTransitionError(f"duplicate or invalid artifact event ID: {event_id}")
        seen_event_ids.add(event_id)
        for value in event.get("objects", []):
            if not isinstance(value, Mapping):
                raise ArtifactTransitionError("artifact object row is malformed")
            _parse_object(value)
        for value in event.get("claims", []):
            if not isinstance(value, Mapping):
                raise ArtifactTransitionError("artifact claim row is malformed")
            _parse_claim(value)
        by_transaction.setdefault(str(event.get("transaction_id")), []).append(event)

    allowed = (
        ArtifactEventKind.STAGED_PRIVATE,
        ArtifactEventKind.TERMINAL_AUTHORIZED,
        ArtifactEventKind.PUBLICATION_AUTHORIZED,
        ArtifactEventKind.RECONSTRUCTION_COMMITTED,
        ArtifactEventKind.PUBLISHED,
        ArtifactEventKind.PRIVATE_COMMITTED,
    )
    for transaction_id, chain in by_transaction.items():
        kinds = [ArtifactEventKind(str(event.get("event_kind"))) for event in chain]
        if not kinds or kinds[0] is not ArtifactEventKind.STAGED_PRIVATE:
            raise ArtifactTransitionError(
                f"transaction lacks staged-private root: {transaction_id}"
            )
        if len(kinds) > 4:
            raise ArtifactTransitionError(f"transaction has too many transitions: {transaction_id}")
        if len(kinds) >= 2 and kinds[1] not in {
            ArtifactEventKind.TERMINAL_AUTHORIZED,
            ArtifactEventKind.PUBLICATION_AUTHORIZED,
        }:
            raise ArtifactTransitionError(f"transaction authorization transition invalid: {kinds}")
        if len(kinds) >= 3 and kinds[2] is not ArtifactEventKind.RECONSTRUCTION_COMMITTED:
            raise ArtifactTransitionError(f"transaction reconstruction transition invalid: {kinds}")
        if len(kinds) == 4 and kinds[3] not in {
            ArtifactEventKind.PUBLISHED,
            ArtifactEventKind.PRIVATE_COMMITTED,
        }:
            raise ArtifactTransitionError(f"transaction final transition invalid: {kinds}")
        if any(kind not in allowed for kind in kinds):
            raise ArtifactTransitionError(f"unknown artifact transition in {transaction_id}")
        baseline = chain[0]
        previous: Optional[str] = None
        for event in chain:
            if event.get("predecessor_event_id") != previous:
                raise ArtifactTransitionError(
                    f"artifact predecessor chain forked in {transaction_id}"
                )
            if any(
                event.get(field) != baseline.get(field)
                for field in (
                    "transaction_id",
                    "stable_id",
                    "work_id",
                    "intake_snapshot_id",
                    "intake_snapshot_sha256",
                    "author_result_id",
                    "proposal_id",
                    "handoff_proposal_id",
                    "handoff_sha256",
                    "source_manifest_identity",
                    "source_ids",
                )
            ):
                raise ArtifactTransitionError(
                    f"artifact transaction identity mutated in {transaction_id}"
                )
            previous = str(event["artifact_event_id"])
        if len(chain) >= 2:
            staged_objects = {str(value["object_id"]): value for value in chain[0]["objects"]}
            authorized_objects = {str(value["object_id"]): value for value in chain[1]["objects"]}
            if any(
                authorized_objects.get(object_id) != value
                for object_id, value in staged_objects.items()
            ):
                raise ArtifactTransitionError(
                    f"authorization dropped or mutated private custody in {transaction_id}"
                )
        if len(chain) >= 3:
            authorization = chain[1]
            for event in chain[2:]:
                if any(
                    event.get(field) != authorization.get(field)
                    for field in (
                        "objects",
                        "claims",
                        "gate_id",
                        "gate_item_sha256",
                        "dependency_vector_sha256",
                        "authorization_id",
                    )
                ):
                    raise ArtifactTransitionError(
                        f"authorized artifact snapshot mutated in {transaction_id}"
                    )
        if len(chain) == 4 and chain[2].get("reconstruction") != chain[3].get("reconstruction"):
            raise ArtifactTransitionError(
                f"reconstruction anchor mutated at finalization in {transaction_id}"
            )


def _staged_from_event(event: Mapping[str, Any]) -> StagedArtifact:
    """Rehydrate one verified staged-private event.

    Parameters
    ----------
    event:
        Persisted staged-private event.

    Returns
    -------
    StagedArtifact
        Typed private-custody handle.
    """

    objects = tuple(_parse_object(value) for value in event["objects"])
    claims = tuple(_parse_claim(value) for value in event["claims"])
    return StagedArtifact(
        transaction_id=ArtifactTransactionId(str(event["transaction_id"])),
        staged_event_id=str(event["artifact_event_id"]),
        event=deepcopy(dict(event)),
        objects=objects,
        custody_claims=claims,
    )


def staged_artifact_for_result(
    ledger: ArtifactEventLedger,
    *,
    stable_id: str,
    work_id: str,
    author_result_id: str,
) -> Optional[StagedArtifact]:
    """Recover exact private custody for an append-only anchored author result.

    Parameters
    ----------
    ledger:
        Validated canonical artifact-event ledger.
    stable_id, work_id, author_result_id:
        Exact immutable result association to recover.

    Returns
    -------
    StagedArtifact | None
        Typed custody handle for the unique staged event, if present.

    Raises
    ------
    ArtifactTransitionError
        If the ledger contains multiple staged roots for the same result binding.
    """

    matches = ledger.staged_events(
        stable_id=stable_id, work_id=work_id, author_result_id=author_result_id
    )
    if len(matches) > 1:
        raise ArtifactTransitionError("author result has multiple private-custody roots")
    return _staged_from_event(matches[0]) if matches else None


def stage_private_artifact(
    artifacts: Sequence[ArtifactInput],
    *,
    context: AuthorityContext,
    stable_id: str,
    work_id: str,
    author_result: Mapping[str, Any],
    proposal: Optional[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    mirrors: MirrorStore,
    ledger: ArtifactEventLedger,
    created_at: Optional[str] = None,
) -> StagedArtifact:
    """Verify and stage every artifact privately before any authorization.

    The function has no public-store or repository-root parameter.  Its only
    byte writer is the private content-addressed mirror.

    Parameters
    ----------
    artifacts, context, stable_id, work_id, author_result, proposal,
    source_manifest, mirrors, ledger, created_at:
        Complete byte set, frozen authority inputs, private store, artifact
        ledger, and optional deterministic event time.

    Returns
    -------
    StagedArtifact
        Verified private-custody transaction.
    """

    (
        result_id,
        proposal_id,
        handoff_proposal_id,
        handoff_sha256,
        handoff_proposal,
        source_identity,
        source_ids,
    ) = _validate_context_result(
        _live_context_basis(context, stable_id),
        stable_id,
        work_id,
        author_result,
        proposal,
        source_manifest,
    )
    execution_proposal = proposal if proposal is not None else handoff_proposal
    _validate_artifact_inputs(artifacts, source_manifest, execution_proposal)
    objects_by_id: dict[MirrorObjectId, MirrorObject] = {}
    custody_keys: dict[MirrorObjectId, str] = {}
    claims: list[ArtifactClaim] = []
    proposal_value: DependencyValue = (
        proposal_id or handoff_proposal_id or DependencyState.NOT_APPLICABLE
    )
    for artifact in sorted(
        artifacts,
        key=lambda value: (
            value.logical_path,
            value.logical_role,
            value.source_id,
            value.content_sha256,
        ),
    ):
        manifest = mirrors.put(
            artifact.content,
            mirror_class=MirrorClass.PRIVATE,
            retention_class=RetentionClass.CAMPAIGN_PRIVATE,
            origin=artifact.origin,
            media_type=artifact.media_type,
        )
        obj = make_mirror_object(
            mirror_class=MirrorClass.PRIVATE,
            content_sha256=manifest.content_sha256,
            byte_count=manifest.byte_count,
            media_type=manifest.media_type,
            object_key=manifest.object_key,
        )
        previous = objects_by_id.setdefault(obj.object_id, obj)
        if previous != obj:
            raise ArtifactBindingError(f"intrinsic object collision: {obj.object_id}")
        custody_keys[obj.object_id] = obj.object_key
        claims.append(
            _make_claim(
                object_id=obj.object_id,
                stable_id=stable_id,
                work_id=work_id,
                proposal_id=proposal_value,
                gate_id=DependencyState.PENDING_UNTRUSTED,
                authorization_id=DependencyState.PENDING_UNTRUSTED,
                logical_role=artifact.logical_role,
                logical_path=artifact.logical_path,
                source_id=artifact.source_id,
                origin=artifact.origin.url,
                revision=artifact.origin.revision,
                fetch_recipe_sha256=stable_hash(artifact.fetch_recipe),
                evidence_ids=artifact.evidence_ids,
                license_disposition=RedistributionClass.UNKNOWN.value,
            )
        )
    objects = tuple(sorted(objects_by_id.values(), key=lambda value: str(value.object_id)))
    normalized_claims = tuple(sorted(claims, key=lambda value: str(value.claim_id)))
    transaction_id = _transaction_id(
        stable_id=stable_id,
        work_id=work_id,
        result_id=result_id,
        proposal_id=proposal_id,
        handoff_proposal_id=handoff_proposal_id,
        handoff_sha256=handoff_sha256,
        source_manifest_identity=source_identity,
        source_ids=source_ids,
        objects=objects,
        claims=normalized_claims,
        context=context,
    )
    event = _event_payload(
        transaction_id=transaction_id,
        predecessor_event_id=None,
        event_kind=ArtifactEventKind.STAGED_PRIVATE,
        created_at=created_at or utc_now(),
        stable_id=stable_id,
        work_id=work_id,
        context=context,
        author_result_id=result_id,
        proposal_id=proposal_id,
        handoff_proposal_id=handoff_proposal_id,
        handoff_sha256=handoff_sha256,
        source_manifest_identity=source_identity,
        source_ids=source_ids,
        objects=tuple((obj, custody_keys[obj.object_id]) for obj in objects),
        claims=normalized_claims,
    )
    result = ledger.append_event(event)
    return _staged_from_event(result.record)


def derive_publication_authorization_id(
    staged: StagedArtifact,
    *,
    accepted_gate_id: str,
    accepted_gate_item_sha256: str,
    dependency_vector: DependencyVector,
    decisions: Mapping[ArtifactClaimId, LicenseDecision],
    publication_policy_identity: str,
) -> PublicationAuthorizationId:
    """Derive the authorization identity a reducer must place in its capability.

    Parameters
    ----------
    staged, accepted_gate_id, accepted_gate_item_sha256, dependency_vector,
    decisions, publication_policy_identity:
        Complete accepted authority and per-custody-claim decisions.

    Returns
    -------
    PublicationAuthorizationId
        Deterministic authorization identity.
    """

    dependency_basis = _dependency_json(dependency_vector)
    # Claim IDs include the authorization ID, while the final dependency vector
    # must include those claims.  Excluding only that closed axis breaks the
    # intentional construction cycle; append_artifact_authorization separately
    # requires exact vector/authorization claim equality.
    dependency_basis["artifact_claim_ids"] = []
    return PublicationAuthorizationId(
        stable_hash(
            {
                "transaction_id": str(staged.transaction_id),
                "accepted_gate_id": accepted_gate_id,
                "accepted_gate_item_sha256": accepted_gate_item_sha256,
                "dependency_vector": dependency_basis,
                "decisions": {
                    str(claim_id): decision.to_dict()
                    for claim_id, decision in sorted(
                        decisions.items(), key=lambda value: str(value[0])
                    )
                },
                "publication_policy_identity": publication_policy_identity,
            }
        )
    )


def _public_object_for(private_object: MirrorObject, mirrors: MirrorStore) -> MirrorObject:
    """Derive a prospective public object without writing its bytes.

    Parameters
    ----------
    private_object:
        Verified private custody object.
    mirrors:
        Separated mirror addressing authority.

    Returns
    -------
    MirrorObject
        Intrinsic public object expected after authorization.
    """

    address = mirrors.address(private_object.content_sha256, MirrorClass.PUBLIC)
    key = address.relative_to(mirrors.root(MirrorClass.PUBLIC)).as_posix()
    return make_mirror_object(
        mirror_class=MirrorClass.PUBLIC,
        content_sha256=private_object.content_sha256,
        byte_count=private_object.byte_count,
        media_type=private_object.media_type,
        object_key=key,
    )


def derive_artifact_claims(
    staged: StagedArtifact,
    *,
    accepted_gate_id: str,
    authorization_id: PublicationAuthorizationId,
    decisions: Mapping[ArtifactClaimId, LicenseDecision],
    mirrors: MirrorStore,
) -> tuple[ArtifactClaim, ...]:
    """Derive accepted claims without constructing reducer authority.

    Parameters
    ----------
    staged, accepted_gate_id, authorization_id, decisions, mirrors:
        Private custody, accepted gate, reducer-selected authorization identity,
        exact gated decisions, and deterministic mirror addressing.

    Returns
    -------
    tuple[ArtifactClaim, ...]
        Complete accepted claim set.  Public-compatible claims reference a
        prospective public object; all other bytes retain the private object.

    Raises
    ------
    ArtifactBindingError
        If decisions are incomplete, hash-mismatched, or not byte dispositions.
    """

    expected_ids = {claim.claim_id for claim in staged.custody_claims}
    if set(decisions) != expected_ids:
        raise ArtifactBindingError("license decisions must cover every staged claim exactly")
    private_by_id = {obj.object_id: obj for obj in staged.objects}
    claims: list[ArtifactClaim] = []
    for custody in staged.custody_claims:
        decision = decisions[custody.claim_id]
        private_object = private_by_id[custody.object_id]
        if decision.content_sha256 != private_object.content_sha256:
            raise ArtifactBindingError("license decision digest differs from staged bytes")
        if decision.redistribution_class is RedistributionClass.NOT_APPLICABLE:
            raise ArtifactBindingError("stored artifact bytes cannot be not-applicable")
        selected_object = (
            _public_object_for(private_object, mirrors)
            if decision.redistribution_class is RedistributionClass.PUBLIC_OK
            else private_object
        )
        claims.append(
            _make_claim(
                object_id=selected_object.object_id,
                stable_id=custody.stable_id,
                work_id=custody.work_id,
                proposal_id=custody.proposal_id,
                gate_id=accepted_gate_id,
                authorization_id=str(authorization_id),
                logical_role=custody.logical_role,
                logical_path=custody.logical_path,
                source_id=custody.source_id,
                origin=custody.origin,
                revision=custody.revision,
                fetch_recipe_sha256=custody.fetch_recipe_sha256,
                evidence_ids=decision.evidence_ids,
                license_disposition=decision.redistribution_class.value,
            )
        )
    return tuple(sorted(claims, key=lambda value: str(value.claim_id)))


def _accepted_gate_matches(item: Mapping[str, Any], event_kind: ArtifactEventKind) -> bool:
    """Return whether a gate item accepts the requested authorization lane.

    Parameters
    ----------
    item:
        Exact checker gate item.
    event_kind:
        Publication or terminal authorization transition.

    Returns
    -------
    bool
        True only for the closed accepted verdict shape.
    """

    if event_kind is ArtifactEventKind.PUBLICATION_AUTHORIZED:
        integrity = item.get("integrity")
        rung = item.get("rung_check")
        return bool(
            item.get("verdict") == "accurate"
            and isinstance(integrity, Mapping)
            and integrity.get("verdict") == "accurate"
            and isinstance(rung, Mapping)
            and rung.get("verdict") == "accurate"
        )
    terminal = item.get("terminal_disposition")
    return bool(isinstance(terminal, Mapping) and terminal.get("verdict") == "accepted")


def _authorization_objects(
    staged: StagedArtifact, claims: Sequence[ArtifactClaim], mirrors: MirrorStore
) -> tuple[tuple[MirrorObject, str], ...]:
    """Derive the complete private plus prospective public object inventory.

    Parameters
    ----------
    staged, claims, mirrors:
        Private transaction, accepted claims, and mirror addressing.

    Returns
    -------
    tuple[tuple[MirrorObject, str], ...]
        Objects paired with their private custody keys.
    """

    objects: dict[MirrorObjectId, tuple[MirrorObject, str]] = {}
    for private in staged.objects:
        objects[private.object_id] = (private, private.object_key)
    for claim in claims:
        if claim.license_disposition == RedistributionClass.PUBLIC_OK.value:
            public_candidates = [
                _public_object_for(value, mirrors)
                for value in staged.objects
                if _public_object_for(value, mirrors).object_id == claim.object_id
            ]
            if len(public_candidates) != 1:
                raise ArtifactBindingError("public claim does not resolve to staged private bytes")
            public = public_candidates[0]
            private = next(
                value for value in staged.objects if value.content_sha256 == public.content_sha256
            )
            objects[public.object_id] = (public, private.object_key)
        elif claim.object_id not in objects:
            raise ArtifactBindingError("private claim does not resolve to staged custody")
    return tuple(sorted(objects.values(), key=lambda value: str(value[0].object_id)))


def append_artifact_authorization(
    staged: StagedArtifact,
    authorization: PublicationAuthorization,
    claims: Sequence[ArtifactClaim],
    *,
    accepted_gate_item: Mapping[str, Any],
    event_kind: ArtifactEventKind,
    context: AuthorityContext,
    mirrors: MirrorStore,
    ledger: ArtifactEventLedger,
    created_at: Optional[str] = None,
) -> JsonObject:
    """Commit reducer-created authorization before any public byte write.

    Parameters
    ----------
    staged, authorization, claims, accepted_gate_item, event_kind, context,
    mirrors, ledger, created_at:
        Private transaction, frozen reducer capability, exact accepted claims
        and gate item, authority context, stores, ledger, and optional time.

    Returns
    -------
    dict[str, Any]
        Persisted authorization event.

    Raises
    ------
    ArtifactBindingError
        If the capability, claims, gate, or dependency vector is not exact.
    """

    if event_kind not in {
        ArtifactEventKind.TERMINAL_AUTHORIZED,
        ArtifactEventKind.PUBLICATION_AUTHORIZED,
    }:
        raise ArtifactTransitionError("authorization requires a closed authorization event kind")
    if (
        type(authorization) is not PublicationAuthorization
        or authorization.transaction_id != staged.transaction_id
        or authorization.stable_id != staged.event["stable_id"]
        or authorization.work_id != staged.event["work_id"]
        or authorization.accepted_gate_item_sha256 != stable_hash(accepted_gate_item)
        or authorization.publication_policy_identity != context.publication_policy_identity
        or not _accepted_gate_matches(accepted_gate_item, event_kind)
        or accepted_gate_item.get("stable_id") != authorization.stable_id
        or accepted_gate_item.get("work_id") != authorization.work_id
    ):
        raise ArtifactBindingError("publication authorization is not exact accepted authority")
    if event_kind is ArtifactEventKind.PUBLICATION_AUTHORIZED:
        verified_hashes = accepted_gate_item.get("verified_hashes")
        if (
            not isinstance(verified_hashes, Mapping)
            or verified_hashes.get("source_manifest") != staged.event["source_manifest_identity"]
        ):
            raise ArtifactBindingError("publication gate source binding differs from staging")
    else:
        terminal = accepted_gate_item.get("terminal_disposition")
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("author_result_id") != staged.event["author_result_id"]
            or terminal.get("handoff_proposal_id") != staged.event.get("handoff_proposal_id")
            or terminal.get("handoff_sha256") != staged.event.get("handoff_sha256")
            or terminal.get("source_manifest_identity") != staged.event["source_manifest_identity"]
            or sorted(terminal.get("source_ids", [])) != staged.event["source_ids"]
        ):
            raise ArtifactBindingError("terminal gate lineage differs from staging")
    vector = authorization.dependency_vector
    if (
        _dependency_value(vector.intake_snapshot_id) != context.active_intake_snapshot_id
        or _dependency_value(vector.intake_snapshot_sha256) != context.active_intake_snapshot_sha256
        or _dependency_value(vector.source_manifest_identity)
        != staged.event["source_manifest_identity"]
        or _dependency_value(vector.author_result_identity) != staged.event["author_result_id"]
        or _dependency_value(vector.artifact_transaction_id) != str(staged.transaction_id)
        or _dependency_value(vector.publication_policy_identity)
        != context.publication_policy_identity
    ):
        raise ArtifactBindingError("authorization dependency vector is stale")
    normalized_claims = tuple(sorted(claims, key=lambda value: str(value.claim_id)))
    if tuple(authorization.claim_ids) != tuple(claim.claim_id for claim in normalized_claims):
        raise ArtifactBindingError("authorization claim set differs from accepted claims")
    if tuple(vector.artifact_claim_ids) != tuple(authorization.claim_ids):
        raise ArtifactBindingError("dependency vector does not bind the authorization claims")
    objects = _authorization_objects(staged, normalized_claims, mirrors)
    objects_by_id = {obj.object_id: obj for obj, _key in objects}
    for claim in normalized_claims:
        if claim.object_id not in objects_by_id:
            raise ArtifactBindingError("accepted claim references an unknown object")
        if (
            claim.stable_id != authorization.stable_id
            or claim.work_id != authorization.work_id
            or _dependency_value(claim.gate_id) != authorization.accepted_gate_id
            or _dependency_value(claim.authorization_id) != str(authorization.authorization_id)
        ):
            raise ArtifactBindingError("accepted claim lineage differs from authorization")
    public_ids = tuple(
        sorted(
            {
                claim.object_id
                for claim in normalized_claims
                if claim.license_disposition == RedistributionClass.PUBLIC_OK.value
            },
            key=str,
        )
    )
    private_ids = tuple(
        sorted(
            {
                claim.object_id
                for claim in normalized_claims
                if claim.license_disposition
                in {
                    RedistributionClass.RESTRICTED_PRIVATE.value,
                    RedistributionClass.UNKNOWN.value,
                }
            },
            key=str,
        )
    )
    if (
        tuple(authorization.public_object_ids) != public_ids
        or tuple(authorization.private_object_ids) != private_ids
        or (event_kind is ArtifactEventKind.TERMINAL_AUTHORIZED and authorization.public_object_ids)
    ):
        raise ArtifactBindingError("authorization object partition differs from claim licenses")
    event = _event_payload(
        transaction_id=staged.transaction_id,
        predecessor_event_id=staged.staged_event_id,
        event_kind=event_kind,
        created_at=created_at or utc_now(),
        stable_id=authorization.stable_id,
        work_id=authorization.work_id,
        context=context,
        author_result_id=str(staged.event["author_result_id"]),
        proposal_id=staged.event.get("proposal_id"),
        handoff_proposal_id=staged.event.get("handoff_proposal_id"),
        handoff_sha256=staged.event.get("handoff_sha256"),
        source_manifest_identity=str(staged.event["source_manifest_identity"]),
        source_ids=tuple(str(value) for value in staged.event["source_ids"]),
        objects=objects,
        claims=normalized_claims,
        gate_id=authorization.accepted_gate_id,
        gate_item_sha256=authorization.accepted_gate_item_sha256,
        dependency_vector_sha256=stable_hash(_dependency_json(vector)),
        authorization_id=str(authorization.authorization_id),
    )
    return ledger.append_event(event).record


def _validate_reconstruction_inputs(
    staged: StagedArtifact,
    authorization: PublicationAuthorization,
    authorization_event: Mapping[str, Any],
    inputs: ReconstructionInputs,
    context: AuthorityContext,
) -> None:
    """Reverify every authored and accepted input before materialization.

    Parameters
    ----------
    staged, authorization, authorization_event, inputs, context:
        Private transaction, reducer authority, exact reconstruction facts, and
        active trust roots.

    Raises
    ------
    ArtifactBindingError
        If any input differs from the staged or authorized transaction.
    """

    (
        result_id,
        proposal_id,
        handoff_proposal_id,
        handoff_sha256,
        _handoff_proposal,
        source_identity,
        source_ids,
    ) = _validate_context_result(
        _live_context_basis(context, authorization.stable_id),
        authorization.stable_id,
        authorization.work_id,
        inputs.author_result,
        inputs.proposal,
        inputs.source_manifest,
    )
    if (
        result_id != staged.event["author_result_id"]
        or proposal_id != staged.event.get("proposal_id")
        or handoff_proposal_id != staged.event.get("handoff_proposal_id")
        or handoff_sha256 != staged.event.get("handoff_sha256")
        or (dict(inputs.handoff_execution) if inputs.handoff_execution is not None else None)
        != (
            dict(inputs.author_result.get("payload", {}).get("handoff_execution", {}))
            if inputs.author_result.get("kind") == "DEFER_RECOMMENDATION"
            else None
        )
        or source_identity != staged.event["source_manifest_identity"]
        or list(source_ids) != staged.event["source_ids"]
        or stable_hash(inputs.accepted_gate_item) != authorization.accepted_gate_item_sha256
    ):
        raise ArtifactBindingError("reconstruction inputs differ from staged authority")
    if (
        authorization_event.get("event_kind") == ArtifactEventKind.PUBLICATION_AUTHORIZED.value
        and inputs.author_result.get("kind") != "PROPOSED"
    ):
        raise ArtifactBindingError("public publication requires a PROPOSED author result")


def _atomic_write_immutable(path: Path, content: bytes) -> None:
    """Create one immutable file atomically or verify an identical replay.

    Parameters
    ----------
    path:
        Final immutable path.
    content:
        Exact file bytes.

    Raises
    ------
    ArtifactPublicationError
        If an existing file has different bytes.
    """

    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ArtifactPublicationError(f"immutable artifact path changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            durable_fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ArtifactPublicationError(f"immutable artifact path raced: {path}")
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            durable_fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _authorization_event(
    ledger: ArtifactEventLedger, authorization: PublicationAuthorization
) -> JsonObject:
    """Resolve the exact committed authorization event.

    Parameters
    ----------
    ledger, authorization:
        Artifact ledger and reducer capability.

    Returns
    -------
    dict[str, Any]
        Unique matching authorization transition.

    Raises
    ------
    ArtifactPublicationError
        If the capability lacks an exact prior ledger fact.
    """

    matches = [
        event
        for event in ledger.transaction_events(authorization.transaction_id)
        if event.get("authorization_id") == str(authorization.authorization_id)
        and event.get("event_kind")
        in {
            ArtifactEventKind.TERMINAL_AUTHORIZED.value,
            ArtifactEventKind.PUBLICATION_AUTHORIZED.value,
        }
    ]
    if len(matches) != 1:
        raise ArtifactPublicationError(
            "public/private commitment requires one exact prior authorization event"
        )
    return matches[0]


def _reconstruction_path(
    canonical_root: Path, stable_id: str, transaction_id: ArtifactTransactionId
) -> Path:
    """Return the immutable transaction-addressed reconstruction path.

    Parameters
    ----------
    canonical_root, stable_id, transaction_id:
        Canonical crawler root and transaction identity.

    Returns
    -------
    pathlib.Path
        ``reconstruction/<prefix>/<stable_id>/<transaction_id>.json``.
    """

    prefix = stable_id.removeprefix("m_")[:2] or "__"
    return canonical_root / "reconstruction" / prefix / stable_id / f"{transaction_id}.json"


#: Identity tag for the authored-publication generation component. Tagged so the
#: component can never collide with another digest namespace minted from the same
#: ``work_id`` string.
_AUTHORED_GENERATION_TAG = "menagerie.crawler.authored-publication-generation.v1"

#: Closed authored-role to publication-root map. Widening it is a new publication
#: root and needs its own reviewed change (checkpoint's allowlist roots too).
_AUTHORED_PUBLICATION_ROOTS = {"code": "adapters", "patch": "patches"}


def authored_generation_component(work_id: str) -> str:
    """Return the path-safe work-generation component of an authored path.

    Parameters
    ----------
    work_id:
        Exact scheduled authority generation for the model.

    Returns
    -------
    str
        ``gen-<24 hex>`` derived from the work generation.

    Raises
    ------
    ArtifactBindingError
        If the work generation is absent or blank.
    """

    value = work_id.strip()
    if not value:
        raise ArtifactBindingError("authored publication requires a work generation")
    # Derived, never the raw ``work_id``: a first-generation id is the path-safe
    # ``work-<stable_id>``, but a requeue/refresh generation is a
    # ``sha256:<hex>`` digest, and the namespace carries no path-safety
    # constraint at all. Hashing makes every generation a legal single path
    # component by construction; the ledger claim carries both the ``work_id``
    # and this ``logical_path``, so the mapping stays recoverable from records.
    return "gen-" + stable_hash({"tag": _AUTHORED_GENERATION_TAG, "work_id": value})[7:31]


def authored_code_logical_path(
    *,
    logical_role: str,
    stable_id: str,
    work_id: str,
    relative: str,
) -> str:
    """Return the immutable publication path for one authored byte.

    This is the ONE mint site for authored (adapter/patch) publication paths.
    The work generation is part of the path because the publication is
    per-generation immutable: a RE-DRIVEN model re-authors its closure under a
    new ``work_id`` and may legitimately vendor DIFFERENT bytes, which the
    generation-unscoped path reported as ``immutable artifact path changed``
    (m8635 re-vendored ``recbole/utils/__init__.py`` and crashed the step-100
    leg, 2026-08-14). Scoping by generation keeps the changed-bytes tripwire
    fully armed WITHIN a generation -- the same ``work_id`` always mints the
    same path, so re-publishing different bytes for the same generation still
    refuses -- while a new generation publishes alongside its predecessor
    instead of overwriting an already-checkpointed public path. Old- and
    new-style paths coexist: every historical claim replays against the
    ``logical_path`` recorded in its own ledger event, never a recomputed one.

    Parameters
    ----------
    logical_role:
        Authored role; ``code`` publishes under ``adapters``, ``patch`` under
        ``patches``.
    stable_id:
        Model whose authored closure is published.
    work_id:
        Exact work generation that authored these bytes.
    relative:
        Declared model-root-relative POSIX path of the authored file.

    Returns
    -------
    str
        ``menagerie/crawler/<root>/<prefix>/<stable_id>/<generation>/<relative>``.

    Raises
    ------
    ArtifactBindingError
        If the role is not an authored role, or the resulting path is unsafe.
    """

    root = _AUTHORED_PUBLICATION_ROOTS.get(logical_role)
    if root is None:
        raise ArtifactBindingError(f"unsupported authored artifact role: {logical_role}")
    prefix = stable_id.removeprefix("m_")[:2] or "__"
    tail = PurePosixPath(relative).as_posix()
    # The declared relative path stays the exact SUFFIX: staging's proposal bind
    # and rehydration both match claims by declared-parts suffix, so inserting
    # the generation ahead of it cannot move a bound identity.
    value = (
        f"menagerie/crawler/{root}/{prefix}/{stable_id}/"
        f"{authored_generation_component(work_id)}/{tail}"
    )
    _safe_relative_path(value)
    return value


def _materialize_public_claims(
    claims: Sequence[ArtifactClaim],
    objects: Mapping[MirrorObjectId, MirrorObject],
    staged: StagedArtifact,
    *,
    mirrors: MirrorStore,
    repository_root: Path,
) -> tuple[str, ...]:
    """Write only public-compatible accepted claims to public destinations.

    Parameters
    ----------
    claims, objects, staged, mirrors, repository_root:
        Accepted claims, complete object map, private custody, separated stores,
        and public repository root.

    Returns
    -------
    tuple[str, ...]
        Exact repository-relative public materialization paths.
    """

    private_by_digest = {obj.content_sha256: obj for obj in staged.objects}
    materialized: dict[str, str] = {}
    for claim in claims:
        if claim.license_disposition != RedistributionClass.PUBLIC_OK.value:
            continue
        public_object = objects[claim.object_id]
        private_object = private_by_digest.get(public_object.content_sha256)
        if private_object is None:
            raise ArtifactPublicationError("public object lacks private-first custody")
        content = mirrors.address(private_object.content_sha256, MirrorClass.PRIVATE).read_bytes()
        if hash_bytes(content) != private_object.content_sha256:
            raise ArtifactPublicationError("private custody bytes changed before publication")
        manifest = mirrors.put(
            content,
            mirror_class=MirrorClass.PUBLIC,
            retention_class=RetentionClass.DURABLE_PUBLIC,
            origin=ArtifactOrigin(claim.origin, claim.revision),
            media_type=public_object.media_type,
        )
        observed = make_mirror_object(
            mirror_class=MirrorClass.PUBLIC,
            content_sha256=manifest.content_sha256,
            byte_count=manifest.byte_count,
            media_type=manifest.media_type,
            object_key=manifest.object_key,
        )
        if observed != public_object:
            raise ArtifactPublicationError("public object differs from authorization")
        relative = _safe_relative_path(claim.logical_path)
        previous = materialized.setdefault(relative.as_posix(), public_object.content_sha256)
        if previous != public_object.content_sha256:
            raise ArtifactPublicationError("public logical path has conflicting authorized bytes")
        _atomic_write_immutable(repository_root / relative, content)
    return tuple(sorted(materialized))


# Structural authority registry consumed by the Round-17 AST inventory. Initial
# public placement has one low-level writer and one authorization-bearing caller.
_PUBLIC_WRITER_REGISTRY = {
    "_materialize_public_claims": "publish_authorized_artifact:committed-authorization",
}


def _public_authorizes_path(
    events: Sequence[Mapping[str, Any]], logical_path: str, content_sha256: str
) -> bool:
    """Return whether any committed accepted public claim authorizes a path.

    Parameters
    ----------
    events:
        Complete artifact event history visible to the writer.
    logical_path, content_sha256:
        Candidate repository path and exact bytes.

    Returns
    -------
    bool
        True only for an accepted public-compatible claim over the same digest.
    """

    object_digests: dict[str, str] = {}
    for event in events:
        for value in event.get("objects", []):
            if isinstance(value, Mapping):
                object_digests[str(value.get("object_id"))] = str(value.get("content_sha256"))
    return any(
        isinstance(value, Mapping)
        and value.get("logical_path") == logical_path
        and value.get("license_disposition") == RedistributionClass.PUBLIC_OK.value
        and value.get("gate_id") not in {None, DependencyState.PENDING_UNTRUSTED.value}
        and value.get("authorization_id") not in {None, DependencyState.PENDING_UNTRUSTED.value}
        and object_digests.get(str(value.get("object_id"))) == content_sha256
        for event in events
        for value in event.get("claims", [])
    )


def _assert_private_claim_absence(
    claims: Sequence[ArtifactClaim],
    objects: Mapping[MirrorObjectId, MirrorObject],
    events: Sequence[Mapping[str, Any]],
    repository_root: Path,
) -> None:
    """Reject repository bytes attributable only to private claims.

    Parameters
    ----------
    claims, objects, events, repository_root:
        Accepted claims, object map, complete ledger history, and repository root.

    Raises
    ------
    ArtifactPublicationError
        If restricted/manifest-only bytes appear without an independent accepted
        public claim for the exact path and digest.
    """

    for claim in claims:
        if claim.license_disposition == RedistributionClass.PUBLIC_OK.value:
            continue
        path = repository_root / _safe_relative_path(claim.logical_path)
        if not path.exists():
            continue
        obj = objects[claim.object_id]
        if not _public_authorizes_path(events, claim.logical_path, obj.content_sha256):
            raise ArtifactPublicationError(
                f"private-only artifact appears in repository: {claim.logical_path}"
            )


def _reconstruction_document(
    *,
    staged: StagedArtifact,
    authorization: PublicationAuthorization,
    authorization_event: Mapping[str, Any],
    inputs: ReconstructionInputs,
    context: AuthorityContext,
    claims: Sequence[ArtifactClaim],
    objects: Sequence[MirrorObject],
    inventory: Mapping[str, Any],
) -> JsonObject:
    """Build deterministic immutable reconstruction bytes.

    Parameters
    ----------
    staged, authorization, authorization_event, inputs, context, claims,
    objects, inventory:
        Complete independently anchored reconstruction facts.

    Returns
    -------
    dict[str, Any]
        Canonical artifact reconstruction document.
    """

    intake_item = context.intake_by_stable_id[authorization.stable_id]
    family_binding = context.family_bindings.get(authorization.stable_id)
    return {
        "schema_version": ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION,
        "transaction_id": str(staged.transaction_id),
        "authorization_event_id": authorization_event["artifact_event_id"],
        "authorization_id": str(authorization.authorization_id),
        "stable_id": authorization.stable_id,
        "work_id": authorization.work_id,
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "intake_item": deepcopy(dict(intake_item)),
        "intake_item_sha256": stable_hash(intake_item),
        "family_binding": deepcopy(dict(family_binding)) if family_binding is not None else None,
        "author_result": deepcopy(dict(inputs.author_result)),
        "proposal": deepcopy(dict(inputs.proposal)) if inputs.proposal is not None else None,
        "handoff_execution": (
            deepcopy(dict(inputs.handoff_execution))
            if inputs.handoff_execution is not None
            else None
        ),
        "handoff_proposal_id": staged.event.get("handoff_proposal_id"),
        "handoff_sha256": staged.event.get("handoff_sha256"),
        "source_manifest": deepcopy(dict(inputs.source_manifest)),
        "source_ids": list(staged.event["source_ids"]),
        "accepted_gate_id": authorization.accepted_gate_id,
        "accepted_gate_item": deepcopy(dict(inputs.accepted_gate_item)),
        "accepted_gate_item_sha256": authorization.accepted_gate_item_sha256,
        "dependency_vector": _dependency_json(authorization.dependency_vector),
        "dependency_vector_sha256": stable_hash(_dependency_json(authorization.dependency_vector)),
        "objects": [
            {
                "object_id": str(obj.object_id),
                "mirror_class": obj.mirror_class,
                "content_sha256": obj.content_sha256,
                "byte_count": obj.byte_count,
                "media_type": obj.media_type,
                "object_key": obj.object_key,
            }
            for obj in sorted(objects, key=lambda value: str(value.object_id))
        ],
        "claims": [
            _claim_json(claim) for claim in sorted(claims, key=lambda value: str(value.claim_id))
        ],
        "publication_inventory": deepcopy(dict(inventory)),
    }


def resume_torn_publication(
    *,
    ledger: "ArtifactEventLedger",
    transaction_id: ArtifactTransactionId,
    dependency_vector: "DependencyVector",
    accepted_gate_item: Mapping[str, Any],
    decisions: Mapping[ArtifactClaimId, LicenseDecision],
    author_result: Mapping[str, Any],
    proposal: Optional[Mapping[str, Any]],
    handoff_execution: Optional[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    context: AuthorityContext,
    mirrors: MirrorStore,
    canonical_root: Path,
    repository_root: Path,
    created_at: Optional[str] = None,
) -> PublishedArtifact:
    """Complete one torn transaction from its COMMITTED capability (AT-7).

    A crash between the authorization event and publication leaves a
    recoverable 2-event chain (m8635 / leg-39). This resume replays
    :func:`publish_authorized_artifact` for the SAME transaction with ZERO
    author dispatch, trusting NOTHING the caller supplies on its own word:

    - the capability is rebuilt from the committed authorization event, and
      every caller-supplied input re-verifies against it -- the gate item
      against ``gate_item_sha256``, the dependency vector against
      ``dependency_vector_sha256``, and the (vector, gate, decisions, policy)
      set jointly against the recorded ``authorization_id`` via
      :func:`derive_publication_authorization_id`;
    - the reconstruction inputs then pass the exact
      :func:`_validate_reconstruction_inputs` bindings against the staged
      event, as any first publication would.

    Raises
    ------
    ArtifactTransitionError
        If the transaction is not a torn authorized 2-event chain.
    ArtifactBindingError
        If any supplied input differs from the committed authority.
    """

    chain = ledger.transaction_events(transaction_id)
    if len(chain) != 2 or chain[1].get("event_kind") not in {
        ArtifactEventKind.PUBLICATION_AUTHORIZED.value,
        ArtifactEventKind.TERMINAL_AUTHORIZED.value,
    }:
        raise ArtifactTransitionError(
            f"transaction {transaction_id} is not a torn authorized chain "
            f"({len(chain)} events); resume replays torn chains only"
        )
    staged = _staged_from_event(chain[0])
    authorization_event = chain[1]
    if stable_hash(accepted_gate_item) != authorization_event["gate_item_sha256"]:
        raise ArtifactBindingError(
            "resume gate item differs from the committed authorization event"
        )
    if (
        stable_hash(_dependency_json(dependency_vector))
        != authorization_event["dependency_vector_sha256"]
    ):
        raise ArtifactBindingError(
            "resume dependency vector differs from the committed authorization event"
        )
    recomputed_authorization_id = derive_publication_authorization_id(
        staged,
        accepted_gate_id=str(authorization_event["gate_id"]),
        accepted_gate_item_sha256=str(authorization_event["gate_item_sha256"]),
        dependency_vector=dependency_vector,
        decisions=decisions,
        publication_policy_identity=context.publication_policy_identity,
    )
    if str(recomputed_authorization_id) != str(authorization_event["authorization_id"]):
        raise ArtifactBindingError(
            "resume decisions/policy do not re-derive the committed authorization id"
        )
    claims = tuple(_parse_claim(value) for value in authorization_event["claims"])
    public_ids = tuple(
        sorted(
            {
                claim.object_id
                for claim in claims
                if claim.license_disposition == RedistributionClass.PUBLIC_OK.value
            },
            key=str,
        )
    )
    private_ids = tuple(
        sorted(
            {
                claim.object_id
                for claim in claims
                if claim.license_disposition
                in {
                    RedistributionClass.RESTRICTED_PRIVATE.value,
                    RedistributionClass.UNKNOWN.value,
                }
            },
            key=str,
        )
    )
    authorization = PublicationAuthorization(
        authorization_id=PublicationAuthorizationId(
            str(authorization_event["authorization_id"])
        ),
        stable_id=str(authorization_event["stable_id"]),
        work_id=str(authorization_event["work_id"]),
        transaction_id=staged.transaction_id,
        accepted_gate_id=str(authorization_event["gate_id"]),
        accepted_gate_item_sha256=str(authorization_event["gate_item_sha256"]),
        dependency_vector=dependency_vector,
        claim_ids=tuple(claim.claim_id for claim in claims),
        public_object_ids=public_ids,
        private_object_ids=private_ids,
        publication_policy_identity=context.publication_policy_identity,
    )
    return publish_authorized_artifact(
        staged,
        authorization,
        reconstruction_inputs=ReconstructionInputs(
            author_result=author_result,
            proposal=proposal,
            handoff_execution=handoff_execution,
            source_manifest=source_manifest,
            accepted_gate_item=accepted_gate_item,
        ),
        context=context,
        mirrors=mirrors,
        ledger=ledger,
        canonical_root=canonical_root,
        repository_root=repository_root,
        created_at=created_at,
    )


def torn_chain_census(ledger: "ArtifactEventLedger") -> tuple[JsonObject, ...]:
    """Classify every incomplete transaction's resume readiness, read-only (TW-TORN).

    One row per non-final chain: ``resumable`` (staged + committed
    authorization -- :func:`resume_torn_publication`'s exact input shape),
    ``pending-custody`` (staged only -- ordinary durable custody, not torn),
    or ``interrupted-other`` (any other incomplete shape). The census steers
    no disposition; the artifact/record incoherence censuses (AT-8/IC-12)
    consume it at close time.
    """

    chains: dict[str, list[JsonObject]] = {}
    for event in ledger.iter_events():
        chains.setdefault(str(event["transaction_id"]), []).append(event)
    rows: list[JsonObject] = []
    for transaction_id, chain in sorted(chains.items()):
        if chain[-1]["event_kind"] in {
            ArtifactEventKind.PUBLISHED.value,
            ArtifactEventKind.PRIVATE_COMMITTED.value,
        }:
            continue
        if len(chain) == 2 and chain[1]["event_kind"] in {
            ArtifactEventKind.PUBLICATION_AUTHORIZED.value,
            ArtifactEventKind.TERMINAL_AUTHORIZED.value,
        }:
            classification = "resumable"
        elif len(chain) == 1 and chain[0]["event_kind"] == (
            ArtifactEventKind.STAGED_PRIVATE.value
        ):
            classification = "pending-custody"
        else:
            classification = "interrupted-other"
        rows.append(
            {
                "transaction_id": transaction_id,
                "stable_id": str(chain[0].get("stable_id")),
                "work_id": str(chain[0].get("work_id")),
                "event_kinds": [str(event["event_kind"]) for event in chain],
                "classification": classification,
            }
        )
    return tuple(rows)


def tw_spent_ledger_join(
    artifact_events: Sequence[Mapping[str, Any]],
    model_records: Sequence[Mapping[str, Any]],
) -> tuple[JsonObject, ...]:
    """TW-SPENT: join every FINAL artifact event to the revisions naming it.

    One row per final event (``published`` / ``private-committed``), joined on
    ``artifact_authority.committed_event_id``:

    - ``paired`` -- exactly one model revision names it (the lawful shape);
    - ``orphaned-final`` -- ZERO revisions name it: a chain was SPENT with no
      canonical award or terminal binding it (the m12646 class: publication
      succeeded, the append refused, and under the pre-AT-4 order not even a
      failed terminal captured the event);
    - ``multiply-named`` -- more than one revision names it (supersession
      lineage over one final event; disclosed for the close to adjudicate).

    Read-only and pure: the fail-closed CLOSE wiring consumes it (W42's close
    half), refusing at minimum on any ``orphaned-final`` row.
    """

    final_kinds = {
        ArtifactEventKind.PUBLISHED.value,
        ArtifactEventKind.PRIVATE_COMMITTED.value,
    }
    naming: dict[str, list[str]] = {}
    for record in model_records:
        authority = record.get("artifact_authority")
        if not isinstance(authority, Mapping):
            continue
        event_id = authority.get("committed_event_id")
        if isinstance(event_id, str) and event_id:
            naming.setdefault(event_id, []).append(str(record.get("record_revision")))
    rows: list[JsonObject] = []
    for event in artifact_events:
        if event.get("event_kind") not in final_kinds:
            continue
        event_id = str(event["artifact_event_id"])
        revisions = naming.get(event_id, [])
        rows.append(
            {
                "artifact_event_id": event_id,
                "transaction_id": str(event["transaction_id"]),
                "stable_id": str(event.get("stable_id")),
                "work_id": str(event.get("work_id")),
                "event_kind": str(event["event_kind"]),
                "naming_revisions": sorted(revisions),
                "classification": (
                    "paired"
                    if len(revisions) == 1
                    else "orphaned-final"
                    if not revisions
                    else "multiply-named"
                ),
            }
        )
    return tuple(rows)


def publish_authorized_artifact(
    staged: StagedArtifact,
    authorization: PublicationAuthorization,
    *,
    reconstruction_inputs: ReconstructionInputs,
    context: AuthorityContext,
    mirrors: MirrorStore,
    ledger: ArtifactEventLedger,
    canonical_root: Path,
    repository_root: Path,
    created_at: Optional[str] = None,
) -> PublishedArtifact:
    """Publish or privately commit one previously authorized transaction.

    No caller-supplied license claim can reach this function without first
    matching an exact authorization event already committed in ``ledger``.

    Parameters
    ----------
    staged, authorization, reconstruction_inputs, context, mirrors, ledger,
    canonical_root, repository_root, created_at:
        Private custody, frozen reducer capability, exact immutable facts,
        active trust roots, storage/ledger roots, and optional event time.

    Returns
    -------
    PublishedArtifact
        Completed transaction and immutable reconstruction anchor.
    """

    authorization_event = _authorization_event(ledger, authorization)
    _validate_reconstruction_inputs(
        staged,
        authorization,
        authorization_event,
        reconstruction_inputs,
        context,
    )
    claims = tuple(_parse_claim(value) for value in authorization_event["claims"])
    objects_with_keys = tuple(
        (_parse_object(value), str(value["private_custody_key"]))
        for value in authorization_event["objects"]
    )
    objects = tuple(value[0] for value in objects_with_keys)
    by_id = {obj.object_id: obj for obj in objects}
    public_paths = _materialize_public_claims(
        claims,
        by_id,
        staged,
        mirrors=mirrors,
        repository_root=repository_root,
    )
    _assert_private_claim_absence(claims, by_id, ledger.events, repository_root)
    public_object_ids = tuple(
        sorted(
            {
                str(claim.object_id)
                for claim in claims
                if claim.license_disposition == RedistributionClass.PUBLIC_OK.value
            }
        )
    )
    private_object_ids = tuple(
        sorted(
            {
                str(claim.object_id)
                for claim in claims
                if claim.license_disposition
                in {
                    RedistributionClass.RESTRICTED_PRIVATE.value,
                    RedistributionClass.UNKNOWN.value,
                }
            }
        )
    )
    publishes = bool(public_object_ids)
    if authorization_event["event_kind"] == ArtifactEventKind.TERMINAL_AUTHORIZED.value:
        publishes = False
    inventory: JsonObject = {
        "lane": "public" if publishes else "private",
        "object_ids": list(public_object_ids if publishes else private_object_ids),
        "materialization_paths": (
            list(public_paths)
            if publishes
            else sorted(by_id[MirrorObjectId(value)].object_key for value in private_object_ids)
        ),
    }
    document = _reconstruction_document(
        staged=staged,
        authorization=authorization,
        authorization_event=authorization_event,
        inputs=reconstruction_inputs,
        context=context,
        claims=claims,
        objects=objects,
        inventory=inventory,
    )
    document_bytes = canonical_json_bytes(document) + b"\n"
    reconstruction_path = _reconstruction_path(
        canonical_root.resolve(), authorization.stable_id, staged.transaction_id
    )
    _atomic_write_immutable(reconstruction_path, document_bytes)
    reconstruction_sha256 = hash_bytes(document_bytes)
    reconstruction_relative = reconstruction_path.resolve().relative_to(repository_root.resolve())
    reconstruction_anchor = ReconstructionAnchorRecord(
        path=reconstruction_relative.as_posix(),
        sha256=reconstruction_sha256,
        claim_ids=tuple(str(claim.claim_id) for claim in claims),
    ).to_payload()
    reconstruction_event = _event_payload(
        transaction_id=staged.transaction_id,
        predecessor_event_id=str(authorization_event["artifact_event_id"]),
        event_kind=ArtifactEventKind.RECONSTRUCTION_COMMITTED,
        created_at=created_at or utc_now(),
        stable_id=authorization.stable_id,
        work_id=authorization.work_id,
        context=context,
        author_result_id=str(staged.event["author_result_id"]),
        proposal_id=staged.event.get("proposal_id"),
        handoff_proposal_id=staged.event.get("handoff_proposal_id"),
        handoff_sha256=staged.event.get("handoff_sha256"),
        source_manifest_identity=str(staged.event["source_manifest_identity"]),
        source_ids=tuple(str(value) for value in staged.event["source_ids"]),
        objects=objects_with_keys,
        claims=claims,
        gate_id=authorization.accepted_gate_id,
        gate_item_sha256=authorization.accepted_gate_item_sha256,
        dependency_vector_sha256=stable_hash(_dependency_json(authorization.dependency_vector)),
        authorization_id=str(authorization.authorization_id),
        reconstruction=reconstruction_anchor,
    )
    persisted_reconstruction = ledger.append_event(reconstruction_event).record
    final_kind = ArtifactEventKind.PUBLISHED if publishes else ArtifactEventKind.PRIVATE_COMMITTED
    final_event = _event_payload(
        transaction_id=staged.transaction_id,
        predecessor_event_id=str(persisted_reconstruction["artifact_event_id"]),
        event_kind=final_kind,
        created_at=created_at or utc_now(),
        stable_id=authorization.stable_id,
        work_id=authorization.work_id,
        context=context,
        author_result_id=str(staged.event["author_result_id"]),
        proposal_id=staged.event.get("proposal_id"),
        handoff_proposal_id=staged.event.get("handoff_proposal_id"),
        handoff_sha256=staged.event.get("handoff_sha256"),
        source_manifest_identity=str(staged.event["source_manifest_identity"]),
        source_ids=tuple(str(value) for value in staged.event["source_ids"]),
        objects=objects_with_keys,
        claims=claims,
        gate_id=authorization.accepted_gate_id,
        gate_item_sha256=authorization.accepted_gate_item_sha256,
        dependency_vector_sha256=stable_hash(_dependency_json(authorization.dependency_vector)),
        authorization_id=str(authorization.authorization_id),
        reconstruction=reconstruction_anchor,
        publication_inventory=inventory,
    )
    persisted_final = ledger.append_event(final_event).record
    return PublishedArtifact(
        transaction_id=staged.transaction_id,
        final_event_id=str(persisted_final["artifact_event_id"]),
        reconstruction_path=reconstruction_path,
        reconstruction_sha256=reconstruction_sha256,
        event=persisted_final,
    )


def _load_events(
    paths: Iterable[Path], *, memo: Optional["ValidationMemo"] = None
) -> tuple[JsonObject, ...]:
    """Load and validate artifact shards in deterministic order.

    Parameters
    ----------
    paths:
        Artifact ledger shard paths.
    memo:
        Optional positive-outcome validation memo threaded through record
        and chain validation. Content digests always re-verify live.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Complete event history.
    """

    events: list[JsonObject] = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        events.extend(scan_jsonl(path, validation_memo=memo))
    validate_artifact_event_chains(events, memo=memo)
    return tuple(events)


def _validate_reconstruction_document(
    document: Mapping[str, Any],
    final_event: Mapping[str, Any],
    authorization_event: Mapping[str, Any],
) -> None:
    """Validate reconstruction bytes against the transaction's own record.

    A finalized transaction is append-only history: its intake snapshot,
    intake item, family binding, and author-contract identities are validated
    against the anchors the ledger recorded for THIS transaction (the final
    event's fields, the anchor-committed document bytes, and the recorded
    dependency vector), never against the ACTIVE authority context. The live
    context deliberately does not reach this function: re-imposing live
    identity axes on recorded history is the rung-11 history-vs-live family
    (one intake refresh or author-prompt re-pin bricked every future
    rehydration of a pre-upgrade final). Currency of the standing MODEL
    record against the live context remains ``authority.validate_currency``'s
    decision, which re-drives that one model and appends new history.

    Parameters
    ----------
    document, final_event, authorization_event:
        Immutable document and its terminal transaction events.

    Raises
    ------
    ArtifactCheckpointError
        If any embedded fact contradicts the append-only record.
    """

    stable_id = str(final_event["stable_id"])
    family_binding = document.get("family_binding")
    if (
        document.get("schema_version") != ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION
        or document.get("transaction_id") != final_event["transaction_id"]
        or document.get("authorization_event_id") != authorization_event["artifact_event_id"]
        or document.get("authorization_id") != final_event["authorization_id"]
        or document.get("stable_id") != stable_id
        or document.get("work_id") != final_event["work_id"]
        or document.get("intake_snapshot_id") != final_event["intake_snapshot_id"]
        or document.get("intake_snapshot_sha256") != final_event["intake_snapshot_sha256"]
        or not isinstance(document.get("intake_item"), Mapping)
        or document.get("intake_item_sha256") != stable_hash(document["intake_item"])
        or not (family_binding is None or isinstance(family_binding, Mapping))
    ):
        raise ArtifactCheckpointError("reconstruction authority/intake/family binding changed")
    result = document.get("author_result")
    proposal = document.get("proposal")
    source_manifest = document.get("source_manifest")
    gate_item = document.get("accepted_gate_item")
    if not all(isinstance(value, Mapping) for value in (result, source_manifest, gate_item)):
        raise ArtifactCheckpointError("reconstruction lacks exact author/source/gate facts")
    assert isinstance(result, Mapping)
    assert isinstance(source_manifest, Mapping)
    assert isinstance(gate_item, Mapping)
    try:
        (
            result_id,
            proposal_id,
            handoff_proposal_id,
            handoff_sha256,
            _handoff_proposal,
            source_identity,
            source_ids,
        ) = _validate_context_result(
            _recorded_transaction_basis(document, final_event),
            stable_id,
            str(final_event["work_id"]),
            result,
            proposal if isinstance(proposal, Mapping) else None,
            source_manifest,
        )
    except AuthorResultSchemaDriftError as exc:
        # Live-schema ineligibility of a byte-verified recorded result is a
        # stale reuse-cache entry, not reconstruction corruption: the caller
        # quarantines this ONE transaction instead of failing the projection.
        raise ArtifactSchemaStaleError(str(exc)) from exc
    except ArtifactBindingError as exc:
        raise ArtifactCheckpointError(str(exc)) from exc
    if (
        result_id != final_event["author_result_id"]
        or proposal_id != final_event.get("proposal_id")
        or handoff_proposal_id != final_event.get("handoff_proposal_id")
        or handoff_sha256 != final_event.get("handoff_sha256")
        or document.get("handoff_proposal_id") != final_event.get("handoff_proposal_id")
        or document.get("handoff_sha256") != final_event.get("handoff_sha256")
        or document.get("handoff_execution")
        != (
            result.get("payload", {}).get("handoff_execution")
            if result.get("kind") == "DEFER_RECOMMENDATION"
            else None
        )
        or source_identity != final_event["source_manifest_identity"]
        or list(source_ids) != final_event["source_ids"]
        or document.get("source_ids") != final_event["source_ids"]
        or document.get("accepted_gate_id") != final_event["gate_id"]
        or document.get("accepted_gate_item_sha256") != final_event["gate_item_sha256"]
        or stable_hash(gate_item) != final_event["gate_item_sha256"]
        or document.get("dependency_vector_sha256") != final_event["dependency_vector_sha256"]
        or stable_hash(document.get("dependency_vector")) != final_event["dependency_vector_sha256"]
        or document.get("publication_inventory") != final_event["publication_inventory"]
    ):
        raise ArtifactCheckpointError("reconstruction proof facts differ from artifact ledger")
    document_objects = document.get("objects")
    document_claims = document.get("claims")
    expected_objects = [
        {key: value for key, value in row.items() if key != "private_custody_key"}
        for row in final_event["objects"]
    ]
    if document_objects != expected_objects or document_claims != final_event["claims"]:
        raise ArtifactCheckpointError("reconstruction object/claim inventory changed")


# Rule/contract identities the reconstruction vector pins AT RECORD TIME. Each
# is a content hash of shipped policy bytes (prompts, schemas, authority.py,
# artifact_transactions.py), so they legitimately move whenever those files
# evolve between legs. A finalized transaction is append-only history recorded
# under the identities that were active when it was adjudicated: it is
# validated at ITS OWN recorded identity, never re-judged against the live
# context (rung-11 leg-3 abort: one batch-13/14 edit to authority.py moved
# terminal_policy_identity and the whole-ledger checkpoint scan refused every
# pre-upgrade final transaction as "stale", permanently -- the ledger cannot be
# rewritten). Currency against the live context is the reducer's model-record
# question: `authority.validate_currency` flags the standing record and the
# scheduler re-drives that one model under current rules, appending new
# history. Tampering with a recorded pin still refuses -- the vector bytes are
# committed by the final event's ``dependency_vector_sha256`` and by the
# reconstruction anchor hash validated before this check runs.
_RECORD_TIME_PINNED_AXES: tuple[str, ...] = (
    "author_result_schema_identity",
    "author_dispatcher_identity",
    "author_prompt_identity",
    "checker_prompt_identity",
    "terminal_rule_identity",
    "publication_policy_identity",
)


def _recorded_vector_pin(document: Mapping[str, Any], axis: str) -> Optional[str]:
    """Return one record-time pin from a reconstruction document's vector.

    Parameters
    ----------
    document:
        Anchor-committed reconstruction document.
    axis:
        Exact dependency-vector axis name.

    Returns
    -------
    str | None
        The recorded pin, or ``None`` when the vector or pin is absent
        (``_validate_projection_dependencies`` separately refuses a valid
        transaction missing a required pin; this accessor stays defensive).
    """

    vector = document.get("dependency_vector")
    if not isinstance(vector, Mapping):
        return None
    value = vector.get(axis)
    return value if isinstance(value, str) and value else None


def _validate_projection_dependencies(
    document: Mapping[str, Any],
    final_event: Mapping[str, Any],
) -> None:
    """Validate the recorded dependency vector against its own transaction.

    Every axis of the closed projection set is validated: axes with an
    independent anchor inside the transaction (its author result, gate,
    claims, sources, and the final event's recorded intake identity) are
    cross-checked so a vector that contradicts the record it claims to bind
    refuses; the remaining rule/contract identities are record-time pins
    (see ``_RECORD_TIME_PINNED_AXES``) required present under the closed
    vocabulary and guaranteed authentic by the ledger's hash commitments.
    No axis is compared against the ACTIVE authority context here: standing
    history is rehydrated under the rule identity it was recorded under, and
    re-adjudication under current rules is the reducer's model-scoped
    currency decision, not a checkpoint integrity question.

    Parameters
    ----------
    document, final_event:
        Verified reconstruction and its final ledger event.

    Raises
    ------
    ArtifactCheckpointError
        If the vector contradicts its own recorded transaction, or a recorded
        pin is missing or malformed.
    """

    vector = document.get("dependency_vector")
    result = document.get("author_result")
    proposal = document.get("proposal")
    claims = document.get("claims")
    if (
        not isinstance(vector, Mapping)
        or not isinstance(result, Mapping)
        or not isinstance(claims, list)
    ):
        raise ArtifactCheckpointError("reconstruction dependency projection is malformed")
    proposal_identity = (
        str(proposal.get("proposal_id"))
        if isinstance(proposal, Mapping)
        else DependencyState.NOT_APPLICABLE.value
    )
    recorded = {
        "intake_snapshot_id": final_event["intake_snapshot_id"],
        "intake_snapshot_sha256": final_event["intake_snapshot_sha256"],
        "intake_item_sha256": document.get("intake_item_sha256"),
        "source_manifest_identity": final_event["source_manifest_identity"],
        "proposal_identity": proposal_identity,
        "author_result_identity": result.get("result_id"),
        "checker_gate_identity": final_event["gate_id"],
        "artifact_transaction_id": final_event["transaction_id"],
        "artifact_claim_ids": sorted(str(row["claim_id"]) for row in claims),
    }
    for field, recorded_value in recorded.items():
        observed = vector.get(field)
        if field == "artifact_claim_ids" and isinstance(observed, list):
            observed = sorted(str(value) for value in observed)
        if field == "checker_gate_identity" and observed in {
            DependencyState.NOT_APPLICABLE.value,
            DependencyState.PENDING_UNTRUSTED.value,
        }:
            # A typed sentinel on this axis states the decisive proof carried
            # NO checker gate (the blocked/terminal lane's historical shape);
            # the final event's gate is the DISPOSITION gate that accepted the
            # arm. An absence statement is not a contradictory gate, so only a
            # concrete mismatch refuses (the rung-12 leg-4 m1359 shape). The
            # write side applies the same rule at `authorize_publication`.
            continue
        if observed != recorded_value:
            raise ArtifactCheckpointError(
                f"reconstruction dependency vector contradicts its record at {field}"
            )
    for field in _RECORD_TIME_PINNED_AXES:
        observed = vector.get(field)
        if not isinstance(observed, str) or not observed:
            raise ArtifactCheckpointError(
                f"reconstruction dependency vector lacks its recorded pin at {field}"
            )


def _register_pending_custody(
    head: Mapping[str, Any],
    object_rows: dict[MirrorObjectId, MirrorObject],
    custody_claims: dict[ArtifactClaimId, ArtifactClaim],
) -> None:
    """Register one staged event's private custody into the global registries.

    Parameters
    ----------
    head:
        Staged-private root event.
    object_rows, custody_claims:
        Whole-store intrinsic object and pending custody registries.

    Raises
    ------
    ArtifactCheckpointError
        On non-private pending custody or a cross-transaction collision.
    """

    for row in head["objects"]:
        obj = _parse_object(row)
        if obj.mirror_class != MirrorClass.PRIVATE.value:
            raise ArtifactCheckpointError(f"pending custody object is not private: {obj.object_id}")
        previous_object = object_rows.setdefault(obj.object_id, obj)
        if previous_object != obj:
            raise ArtifactCheckpointError(f"intrinsic object collision: {obj.object_id}")
    for row in head["claims"]:
        claim = _parse_claim(row)
        previous_claim = custody_claims.setdefault(claim.claim_id, claim)
        if previous_claim != claim:
            raise ArtifactCheckpointError(f"custody claim collision: {claim.claim_id}")


def _transaction_document_memo_key(
    anchor_sha256: str,
    final_event: Mapping[str, Any],
    authorization_event: Mapping[str, Any],
) -> Optional[str]:
    """Return the memo key for one transaction's pure document validation.

    ``_validate_reconstruction_document`` and
    ``_validate_projection_dependencies`` are pure in (document bytes, final
    event, authorization event) plus the schema environment the memo binds
    separately. The anchor sha256 pins the document bytes and is re-verified
    live against the committed file on every fold; the two event digests are
    re-proven here against the events' own bytes, so a key never exists for
    content that was not just re-verified.

    Parameters
    ----------
    anchor_sha256, final_event, authorization_event:
        Live-verified document digest and the transaction's terminal events.

    Returns
    -------
    str | None
        Content-identity key, or ``None`` when either event lacks a
        live-verifiable digest and the block must validate live.
    """

    final_digest = final_event.get("payload_sha256")
    authorization_digest = authorization_event.get("payload_sha256")
    if (
        not isinstance(final_digest, str)
        or not isinstance(authorization_digest, str)
        or final_digest != payload_hash(final_event)
        or authorization_digest != payload_hash(authorization_event)
    ):
        return None
    return stable_hash(
        {
            "reconstruction_sha256": anchor_sha256,
            "final_event_sha256": final_digest,
            "authorization_event_sha256": authorization_digest,
        }
    )


def _fold_transaction_chain(
    transaction_id: str,
    chain: Sequence[JsonObject],
    *,
    object_rows: dict[MirrorObjectId, MirrorObject],
    custody_claims: dict[ArtifactClaimId, ArtifactClaim],
    accepted_claims: dict[ArtifactClaimId, ArtifactClaim],
    transactions: dict[ArtifactProjectionKey, ArtifactTransactionProjection],
    stale_transactions: dict[ArtifactProjectionKey, StaleArtifactTransaction],
    canonical_root: Path,
    repository_root: Path,
    memo: Optional["ValidationMemo"] = None,
) -> None:
    """Fold one complete transaction chain into the checkpoint registries.

    This is the exact per-transaction body of full checkpoint validation,
    shared verbatim with the incremental fold so both paths cannot diverge:
    pending custody registers durably, an interrupted or contradicted
    transaction quarantines into ``stale_transactions``, and only a fully
    verified 4-event chain projects reusable authority.

    Parameters
    ----------
    transaction_id, chain:
        One transaction's complete events in ledger order.
    object_rows, custody_claims, accepted_claims, transactions,
    stale_transactions:
        Whole-store registries mutated by the fold.
    canonical_root, repository_root:
        Canonical crawler root and public repository root.
    memo:
        Optional positive-outcome validation memo for the pure document
        block. Byte anchors always re-verify live.

    Raises
    ------
    ArtifactCheckpointError
        Only for GLOBAL ledger/store integrity, exactly like full validation.
    """

    if len(chain) == 1 and chain[0]["event_kind"] == ArtifactEventKind.STAGED_PRIVATE.value:
        _register_pending_custody(chain[0], object_rows, custody_claims)
        return
    if len(chain) != 4 or chain[-1]["event_kind"] not in {
        ArtifactEventKind.PUBLISHED.value,
        ArtifactEventKind.PRIVATE_COMMITTED.value,
    }:
        # An interrupted finalization (a crash between authorization and
        # commitment) is one model's defective cache entry, never a
        # projection failure: its private custody stays registered exactly
        # like durable pending custody, and the transaction quarantines so
        # the model re-drives fresh (rung-12 crash family).
        head = chain[0]
        if head["event_kind"] != ArtifactEventKind.STAGED_PRIVATE.value:
            raise ArtifactCheckpointError(
                f"artifact transaction chain does not begin with staged custody: {transaction_id}"
            )
        _register_pending_custody(head, object_rows, custody_claims)
        key = (
            str(head["stable_id"]),
            str(head["work_id"]),
            ArtifactTransactionId(transaction_id),
        )
        resumable = len(chain) == 2 and chain[-1]["event_kind"] in {
            ArtifactEventKind.PUBLICATION_AUTHORIZED.value,
            ArtifactEventKind.TERMINAL_AUTHORIZED.value,
        }
        stale_transactions[key] = StaleArtifactTransaction(
            stable_id=key[0],
            work_id=key[1],
            transaction_id=key[2],
            claims=(),
            reason=(
                # AT-7: a staged + committed-authorization chain is the
                # RESUMABLE torn shape -- resume_torn_publication completes
                # the same transaction with zero author dispatch.
                f"torn-chain-resumable: {transaction_id} "
                f"(2 events, final {chain[-1]['event_kind']!r})"
                if resumable
                else f"incomplete artifact transaction: {transaction_id} "
                f"({len(chain)} events, final {chain[-1]['event_kind']!r})"
            ),
        )
        return
    final_event = chain[-1]
    authorization_event = chain[1]
    key = (
        str(final_event["stable_id"]),
        str(final_event["work_id"]),
        ArtifactTransactionId(transaction_id),
    )
    # GLOBAL ledger/store integrity first, quarantined or not: intrinsic
    # object and claim registration feeds the whole-store inventory below,
    # and a cross-transaction collision is corruption of the shared store,
    # never one model's stale cache entry.
    for row in final_event["objects"]:
        obj = _parse_object(row)
        previous_object = object_rows.setdefault(obj.object_id, obj)
        if previous_object != obj:
            raise ArtifactCheckpointError(f"intrinsic object collision: {obj.object_id}")
    for row in chain[0]["claims"]:
        claim = _parse_claim(row)
        previous_custody_claim = custody_claims.setdefault(claim.claim_id, claim)
        if previous_custody_claim != claim:
            raise ArtifactCheckpointError(f"custody claim collision: {claim.claim_id}")
    for row in final_event["claims"]:
        claim = _parse_claim(row)
        previous_accepted_claim = accepted_claims.setdefault(claim.claim_id, claim)
        if previous_accepted_claim != claim:
            raise ArtifactCheckpointError(f"accepted claim collision: {claim.claim_id}")
    final_claims = tuple(_parse_claim(row) for row in final_event["claims"])
    # W44 SHADOW-EXCLUSION: a chain any of whose events carries the structural
    # shadow_rehearsal marker (an R6b rehearsal record) never projects
    # reusable authority. It quarantines as a DISCLOSED stale entry -- keyed
    # on the record's own marker, never an environment or mode flag -- so the
    # model reads as a cache miss in normal operation while its physical
    # custody and claims stay registered above (rehearsal bytes are attested
    # store contents, never foreign litter).
    if any(is_shadow_rehearsal_record(event) for event in chain):
        stale_transactions[key] = StaleArtifactTransaction(
            stable_id=key[0],
            work_id=key[1],
            transaction_id=key[2],
            claims=final_claims,
            reason=f"shadow-rehearsal-marked artifact transaction: {transaction_id}",
        )
        return
    # PER-TRANSACTION validation, guarded: any refusal here -- a rewritten
    # or unreadable reconstruction, a vector contradicting its own record
    # (rung-12 leg 4: checker_gate_identity), a schema-stale author result
    # (rung-12 leg 1), an inventory shape mismatch -- quarantines exactly
    # this transaction as a stale cache entry. The model re-drives fresh
    # and loudly; the projection stays alive for every other model.
    stale_reason: Optional[str] = None
    document: Optional[Mapping[str, Any]] = None
    reconstruction_path: Optional[Path] = None
    anchor: Optional[ReconstructionAnchorRecord] = None
    try:
        reconstruction = final_event.get("reconstruction")
        if not isinstance(reconstruction, Mapping):
            raise ArtifactCheckpointError("final artifact event lacks reconstruction anchor")
        anchor = ReconstructionAnchorRecord.from_payload(reconstruction)
        relative = _safe_relative_path(anchor.path)
        reconstruction_path = repository_root.resolve() / relative
        expected_path = _reconstruction_path(
            canonical_root.resolve(),
            str(final_event["stable_id"]),
            ArtifactTransactionId(transaction_id),
        )
        if reconstruction_path.resolve() != expected_path.resolve():
            raise ArtifactCheckpointError("reconstruction path is not transaction-addressed")
        try:
            reconstruction_bytes = reconstruction_path.read_bytes()
        except OSError as exc:
            raise ArtifactCheckpointError("committed reconstruction bytes are missing") from exc
        if hash_bytes(reconstruction_bytes) != anchor.sha256:
            raise ArtifactCheckpointError("committed reconstruction bytes changed")
        try:
            import json

            loaded_document = json.loads(reconstruction_bytes)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ArtifactCheckpointError("committed reconstruction JSON is invalid") from exc
        if not isinstance(loaded_document, Mapping):
            raise ArtifactCheckpointError("committed reconstruction must be an object")
        document = loaded_document
        document_memo_key = (
            _transaction_document_memo_key(anchor.sha256, final_event, authorization_event)
            if memo is not None
            else None
        )
        if not (
            document_memo_key is not None
            and memo is not None
            and memo.seen("transaction-document", document_memo_key)
        ):
            try:
                _validate_reconstruction_document(document, final_event, authorization_event)
            except ArtifactSchemaStaleError as exc:
                # Every byte anchor verified; only live-schema reuse
                # eligibility failed (the solve cache is convenience, not
                # history). Continue the remaining record-anchored checks so a
                # schema-stale entry that ALSO contradicts its record reports
                # the contradiction.
                stale_reason = str(exc)
            _validate_projection_dependencies(document, final_event)
            if stale_reason is None and document_memo_key is not None and memo is not None:
                memo.record("transaction-document", document_memo_key)
        claim_ids = sorted(str(row["claim_id"]) for row in final_event["claims"])
        if list(anchor.claim_ids) != claim_ids:
            raise ArtifactCheckpointError("reconstruction claim set differs from final event")
        publishes = final_event["event_kind"] == ArtifactEventKind.PUBLISHED.value
        if publishes and authorization_event["event_kind"] != (
            ArtifactEventKind.PUBLICATION_AUTHORIZED.value
        ):
            raise ArtifactCheckpointError(
                f"published transaction lacks publication authorization: {transaction_id}"
            )
        selected_claims = tuple(
            claim
            for claim in final_claims
            if (
                claim.license_disposition == RedistributionClass.PUBLIC_OK.value
                if publishes
                else claim.license_disposition
                in {
                    RedistributionClass.RESTRICTED_PRIVATE.value,
                    RedistributionClass.UNKNOWN.value,
                }
            )
        )
        final_objects = {
            obj.object_id: obj for obj in (_parse_object(row) for row in final_event["objects"])
        }
        expected_inventory = {
            "lane": "public" if publishes else "private",
            "object_ids": sorted({str(claim.object_id) for claim in selected_claims}),
            "materialization_paths": (
                sorted({claim.logical_path for claim in selected_claims})
                if publishes
                else sorted(
                    {final_objects[claim.object_id].object_key for claim in selected_claims}
                )
            ),
        }
        if final_event.get("publication_inventory") != expected_inventory:
            raise ArtifactCheckpointError(
                f"final publication inventory differs from claims: {transaction_id}"
            )
        if publishes and not selected_claims:
            raise ArtifactCheckpointError(
                f"published transaction has no public-compatible claim: {transaction_id}"
            )
    except ArtifactCheckpointError as exc:
        stale_reason = str(exc)
    if stale_reason is not None:
        stale_transactions[key] = StaleArtifactTransaction(
            stable_id=key[0],
            work_id=key[1],
            transaction_id=key[2],
            claims=final_claims,
            reason=stale_reason,
        )
        return
    assert document is not None and reconstruction_path is not None and anchor is not None
    projection = ArtifactTransactionProjection(
        stable_id=key[0],
        work_id=key[1],
        transaction_id=key[2],
        final_event_id=str(final_event["artifact_event_id"]),
        final_event_kind=str(final_event["event_kind"]),
        authorization_id=str(final_event["authorization_id"]),
        accepted_gate_id=str(final_event["gate_id"]),
        reconstruction_path=reconstruction_path,
        reconstruction_sha256=anchor.sha256,
        reconstruction_inputs=ReconstructionInputs(
            author_result=deepcopy(dict(document["author_result"])),
            proposal=(
                deepcopy(dict(document["proposal"]))
                if isinstance(document.get("proposal"), Mapping)
                else None
            ),
            handoff_execution=(
                deepcopy(dict(document["handoff_execution"]))
                if isinstance(document.get("handoff_execution"), Mapping)
                else None
            ),
            source_manifest=deepcopy(dict(document["source_manifest"])),
            accepted_gate_item=deepcopy(dict(document["accepted_gate_item"])),
        ),
        objects=tuple(sorted(final_objects.values(), key=lambda value: str(value.object_id))),
        claims=tuple(sorted(final_claims, key=lambda value: str(value.claim_id))),
        recorded_author_prompt_identity=_recorded_vector_pin(document, "author_prompt_identity"),
        recorded_checker_prompt_identity=_recorded_vector_pin(document, "checker_prompt_identity"),
    )
    previous_projection = transactions.setdefault(key, projection)
    if previous_projection != projection:
        raise ArtifactCheckpointError(f"final transaction projection collision: {key!r}")


@dataclass
class _InsertionLoggedDict(dict):
    """Registry recording newly inserted values for O(delta) capture.

    The incremental fold must learn which objects and claims each advance
    added without re-walking the whole registry (the re-walk was itself a
    measured linear term in the r5 residual slope).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Bind an empty insertion log beside the mapping."""

        super().__init__(*args, **kwargs)
        self.insertion_log: list[Any] = []

    def setdefault(self, key: Any, default: Any = None) -> Any:
        """Register ``default`` under ``key`` and log a first insertion.

        Parameters
        ----------
        key, default:
            Exactly :meth:`dict.setdefault` semantics.

        Returns
        -------
        Any
            The stored value.
        """

        if key not in self:
            self.insertion_log.append(default)
        return super().setdefault(key, default)


_PUBLIC_AUTHORITY_PENDING = {None, DependencyState.PENDING_UNTRUSTED.value}


def _record_public_path_authorities(
    event: Mapping[str, Any],
    object_rows: Mapping[MirrorObjectId, MirrorObject],
    public_path_digests: dict[str, set[str]],
) -> None:
    """Fold one event's accepted public claims into the path-authority index.

    Maintains, incrementally, exactly the predicate of
    :func:`_public_authorizes_path`: an accepted public-compatible claim over
    an exact object digest authorizes that repository path.

    Parameters
    ----------
    event, object_rows, public_path_digests:
        One artifact event, the intrinsic object registry, and the index.
    """

    for value in event.get("claims", []):
        if (
            isinstance(value, Mapping)
            and value.get("license_disposition") == RedistributionClass.PUBLIC_OK.value
            and value.get("gate_id") not in _PUBLIC_AUTHORITY_PENDING
            and value.get("authorization_id") not in _PUBLIC_AUTHORITY_PENDING
        ):
            obj = object_rows.get(MirrorObjectId(str(value.get("object_id"))))
            if obj is not None:
                public_path_digests.setdefault(str(value.get("logical_path")), set()).add(
                    obj.content_sha256
                )


@dataclass
class _CheckpointFoldState:
    """Mutable whole-store registries retained across incremental folds."""

    events: list[JsonObject]
    seen_event_ids: set[str]
    chains: dict[str, list[JsonObject]]
    object_rows: _InsertionLoggedDict
    custody_claims: _InsertionLoggedDict
    accepted_claims: _InsertionLoggedDict
    transactions: dict[ArtifactProjectionKey, ArtifactTransactionProjection]
    stale_transactions: dict[ArtifactProjectionKey, StaleArtifactTransaction]
    expected_physical: dict[tuple[MirrorClass, str], MirrorObject]
    public_by_path: dict[str, MirrorObject]
    sorted_objects: list[MirrorObject]
    sorted_accepted: list[ArtifactClaim]
    public_path_digests: dict[str, set[str]]
    store_litter: tuple[tuple[str, str], ...] = ()

    def build_projection(self) -> ArtifactCheckpointProjection:
        """Return the projection of the current registries.

        The projection shares the live registry containers (exactly like the
        one-shot full validation, whose locals die at return): it is current
        until the next :meth:`ArtifactCheckpointIncrement.advance`, and
        consumers read it immediately. Copying or re-sorting here was a
        measured linear term in the r5 residual slope.

        Returns
        -------
        ArtifactCheckpointProjection
            Normalized final transactions, intrinsic objects, and accepted
            claims.
        """

        return ArtifactCheckpointProjection(
            transactions=self.transactions,
            objects=tuple(self.sorted_objects),
            claims=tuple(self.sorted_accepted),
            stale_transactions=self.stale_transactions,
            store_litter=self.store_litter,
        )


def _intrinsic_physical_inventory(obj: MirrorObject) -> tuple[str, str, int, str]:
    """Return the physical identity the mirror STORE itself proves for one object.

    The ambiguity tripwire guards the content-addressed store: two ledger rows
    naming one physical key must describe the same BYTES. ``media_type`` is a
    DECLARED classification carried by each transaction's claim -- two eras
    legitimately classified the same 11,357-byte file text/plain (m5445,
    2026-08-07) and text/markdown (m13746, 2026-08-20), and ``object_id``
    hashes that declaration -- so neither is intrinsic to the bytes. Divergent
    intrinsic facts (digest, size, key, class) still refuse fatally.

    Parameters
    ----------
    obj:
        Ledger-minted mirror object row.

    Returns
    -------
    tuple[str, str, int, str]
        Mirror class, content digest, byte count, and object key.
    """

    return (obj.mirror_class, obj.content_sha256, obj.byte_count, obj.object_key)


def validate_artifact_checkpoint(
    artifact_ledger_paths: Iterable[Path],
    *,
    context: AuthorityContext,
    mirrors: MirrorStore,
    canonical_root: Path,
    repository_root: Path,
    memo: Optional["ValidationMemo"] = None,
) -> ArtifactCheckpointProjection:
    """Validate and project artifact authority, reconstruction, and mirror inventory.

    Parameters
    ----------
    artifact_ledger_paths, context, mirrors, canonical_root, repository_root:
        Complete artifact ledger shards, active trust roots, physical
        mirrors, canonical crawler root, and public repository root.
        ``context`` is retained for signature stability only: every finalized
        transaction validates against its OWN recorded anchors (final event,
        anchor-committed document, recorded dependency vector), never against
        the live context -- see :func:`_validate_reconstruction_document`.
    memo:
        Optional positive-outcome validation memo (the persistent checkpoint
        cache). A memoized record or document skips only its pure jsonschema
        pass; every byte anchor and content digest still re-verifies live,
        and an absent, corrupt, or foreign-environment memo degrades to
        exactly the uncached behavior.

    Returns
    -------
    ArtifactCheckpointProjection
        Immutable normalized final transactions, intrinsic objects, and accepted
        claims shared by reconstruction and checkpoint consumers.

    Raises
    ------
    ArtifactCheckpointError
        Only for GLOBAL ledger/store integrity: a cross-transaction object or
        claim collision, a chain not rooted in staged custody, a physical
        mirror mismatch or orphan, or a public materialization conflict. Every
        PER-TRANSACTION refusal -- an incomplete chain, a rewritten, missing,
        or unreadable reconstruction, a vector contradicting its own record, a
        schema-stale author result, an inventory shape mismatch -- quarantines
        that one transaction into ``stale_transactions`` instead: single
        defective cache entries aborted whole campaigns at rehydration
        (rung-12 legs 1 and 4). A sole ``staged-private`` event is durable
        pending custody and is checkpoint-valid.
    """

    _state, projection = _full_checkpoint_fold(
        artifact_ledger_paths,
        mirrors=mirrors,
        canonical_root=canonical_root,
        repository_root=repository_root,
        memo=memo,
    )
    return projection


def _ledger_attested_physical_keys(
    chains: Iterable[Sequence[Mapping[str, Any]]],
) -> set[tuple[MirrorClass, str]]:
    """Return every public/private ``(mirror_class, object_key)`` any event names.

    The 967b32a0 attestation harvest, shared by inventory reconciliation and
    :func:`cas_inventory_audit` so the two can never diverge on what counts
    as ledger-attested.

    Parameters
    ----------
    chains:
        Transaction event chains.

    Returns
    -------
    set[tuple[MirrorClass, str]]
        Attested physical keys.
    """

    attested: set[tuple[MirrorClass, str]] = set()
    for chain in chains:
        for event in chain:
            stack: list[Any] = [event]
            while stack:
                node = stack.pop()
                if isinstance(node, Mapping):
                    mirror_class = node.get("mirror_class")
                    object_key = node.get("object_key")
                    if (
                        mirror_class in {MirrorClass.PUBLIC.value, MirrorClass.PRIVATE.value}
                        and isinstance(object_key, str)
                        and object_key
                    ):
                        attested.add((MirrorClass(mirror_class), object_key))
                    stack.extend(node.values())
                elif isinstance(node, (list, tuple)):
                    stack.extend(node)
    return attested


def cas_inventory_audit(
    artifact_ledger_paths: Iterable[Path],
    *,
    mirrors: MirrorStore,
    preflight: bool = False,
) -> tuple[JsonObject, ...]:
    """Classify every attested or observed physical CAS key, read-only (IC-2).

    Three dispositions over the union of ledger-attested and on-disk keys:

    - ``attested-present``: named by a ledger event and on disk.
    - ``attested-absent``: named by a ledger event, absent from disk -- the
      m8635 shape (a torn publication left 2 of 23 attested public objects
      unwritten); ONE un-staling action away from an admission brick, so the
      preflight arm refuses on any such key.
    - ``present-unattested``: on disk, named by nothing -- split into
      ``store-litter`` (non-addressable-shaped, IC-3) and ``foreign``
      (addressable-shaped, corruption-indistinguishable).

    Parameters
    ----------
    artifact_ledger_paths:
        Complete artifact ledger shards.
    mirrors:
        Physical mirror store.
    preflight:
        Refuse (typed) on any ``attested-absent`` or ``foreign`` row instead
        of returning the report.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One disposition row per key, sorted.

    Raises
    ------
    ArtifactCheckpointError
        In preflight mode, when any attested-absent or foreign key exists.
    """

    events = _load_events(artifact_ledger_paths, memo=None)
    by_transaction: dict[str, list[JsonObject]] = {}
    for event in events:
        by_transaction.setdefault(str(event["transaction_id"]), []).append(event)
    attested = _ledger_attested_physical_keys(by_transaction.values())
    observed = set(
        pair
        for pair in mirrors.iter_objects()
        if pair[0] in {MirrorClass.PUBLIC, MirrorClass.PRIVATE}
    )
    rows: list[JsonObject] = []
    for mirror_class, object_key in sorted(attested | observed, key=str):
        key = (mirror_class, object_key)
        if key in attested and key in observed:
            disposition = "attested-present"
        elif key in attested:
            disposition = "attested-absent"
        elif is_addressable_object_key(object_key):
            disposition = "present-unattested-foreign"
        else:
            disposition = "present-unattested-litter"
        rows.append(
            {
                "mirror_class": mirror_class.value,
                "object_key": object_key,
                "disposition": disposition,
            }
        )
    if preflight:
        refusing = [
            f"{row['mirror_class']}/{row['object_key']} ({row['disposition']})"
            for row in rows
            if row["disposition"] in {"attested-absent", "present-unattested-foreign"}
        ]
        if refusing:
            raise ArtifactCheckpointError(
                "cas inventory preflight refuses: "
                + "; ".join(refusing[:8])
                + (f" (+{len(refusing) - 8} more)" if len(refusing) > 8 else "")
            )
    return tuple(rows)


def _full_checkpoint_fold(
    artifact_ledger_paths: Iterable[Path],
    *,
    mirrors: MirrorStore,
    canonical_root: Path,
    repository_root: Path,
    memo: Optional["ValidationMemo"] = None,
) -> tuple[_CheckpointFoldState, ArtifactCheckpointProjection]:
    """Run one complete checkpoint validation, retaining the fold registries.

    Parameters
    ----------
    artifact_ledger_paths, mirrors, canonical_root, repository_root, memo:
        Exactly the inputs of :func:`validate_artifact_checkpoint`.

    Returns
    -------
    tuple[_CheckpointFoldState, ArtifactCheckpointProjection]
        Whole-store registries and the validated projection.
    """

    events = _load_events(artifact_ledger_paths, memo=memo)
    by_transaction: dict[str, list[JsonObject]] = {}
    for event in events:
        by_transaction.setdefault(str(event["transaction_id"]), []).append(event)
    object_rows = _InsertionLoggedDict()
    custody_claims = _InsertionLoggedDict()
    accepted_claims = _InsertionLoggedDict()
    transactions: dict[ArtifactProjectionKey, ArtifactTransactionProjection] = {}
    stale_transactions: dict[ArtifactProjectionKey, StaleArtifactTransaction] = {}
    for transaction_id, chain in by_transaction.items():
        _fold_transaction_chain(
            transaction_id,
            chain,
            object_rows=object_rows,
            custody_claims=custody_claims,
            accepted_claims=accepted_claims,
            transactions=transactions,
            stale_transactions=stale_transactions,
            canonical_root=canonical_root,
            repository_root=repository_root,
            memo=memo,
        )
    all_claims = {**custody_claims, **accepted_claims}
    for claim in all_claims.values():
        if claim.object_id not in object_rows:
            raise ArtifactCheckpointError(f"claim references absent object: {claim.claim_id}")

    expected_physical: dict[tuple[MirrorClass, str], MirrorObject] = {}
    for obj in object_rows.values():
        if obj.mirror_class not in {MirrorClass.PUBLIC.value, MirrorClass.PRIVATE.value}:
            continue
        physical_key = (MirrorClass(obj.mirror_class), obj.object_key)
        previous_object = expected_physical.setdefault(physical_key, obj)
        if _intrinsic_physical_inventory(previous_object) != _intrinsic_physical_inventory(obj):
            raise ArtifactCheckpointError(
                f"physical mirror object has ambiguous intrinsic inventory: {physical_key}"
            )
    observed_physical = set(
        pair
        for pair in mirrors.iter_objects()
        if pair[0] in {MirrorClass.PUBLIC, MirrorClass.PRIVATE}
    )
    # A quarantined-stale transaction's physical objects stay on disk as
    # immutable audit evidence while its object rows never register into
    # reusable authority ("cache miss for the affected model, every OTHER
    # model's authority stays alive"). Reading those ledger-attested bytes
    # as foreign re-created the rung-12 family one layer up: two Aug-14
    # stale-era PUBLIC files aborted leg-39's whole admission rehydration
    # as "orphans". A file is foreign only when NO ledger event names its
    # key; attested-but-stale files are tolerated for inventory and remain
    # unserved (claims resolve only through registered current objects).
    ledger_attested_keys = _ledger_attested_physical_keys(by_transaction.values())
    unattested = observed_physical - ledger_attested_keys - set(expected_physical)
    # IC-3 litter partition: a non-addressable-shaped unattested name (a
    # Finder .DS_Store, a SIGKILL-orphaned put temporary) is typed store
    # litter -- disclosed on the projection, never served, never a campaign
    # brick. An unattested file at a VALID content address stays a fatal
    # refusal: it is indistinguishable from store corruption, and narrowing
    # that boundary needs its own four-part review.
    store_litter = tuple(
        sorted(
            (
                (mirror_class.value, object_key)
                for mirror_class, object_key in unattested
                if not is_addressable_object_key(object_key)
            ),
        )
    )
    litter_keys = {
        (MirrorClass(mirror_class), object_key) for mirror_class, object_key in store_litter
    }
    # IC-2 missing-of-STALE scoping (four-part shape, stated in full):
    # a registered physical absent from disk is FATAL only when a CURRENT
    # (fully verified 4-event) transaction registers it; a physical
    # registered ONLY by non-current chains (durable pending custody or an
    # already-quarantined interrupted chain) is that transaction's defect,
    # quarantined typed, never a campaign brick.
    # (1) Disposition-invariant: the scoped arm only DEMOTES -- the owning
    #     transaction becomes/stays a stale cache miss; nothing is upgraded,
    #     published, or unlocked, and missing-of-CURRENT keeps refusing.
    # (2) Closed proof rule: object registration has exactly TWO sites
    #     (_fold_transaction_chain: pending-custody registration for <4-event
    #     chains; final-event registration for complete chains), so
    #     current-vs-stale ownership of every key is an exhaustive partition
    #     over `transactions` -- no heuristic, no tolerance.
    # (3) Replay-derived: recomputed on every fold from ledger chains plus
    #     live disk state; no persisted flag or producer field steers it.
    # (4) Disclosed: the demoted transaction lands in `stale_transactions`
    #     with a typed custody-missing reason (a derived census surface,
    #     never counted as healthy); the cas_inventory_audit tool reports
    #     the same keys as attested-absent.
    missing = set(expected_physical) - observed_physical
    current_keys = {
        (MirrorClass(obj.mirror_class), obj.object_key)
        for transaction in transactions.values()
        for obj in transaction.objects
        if obj.mirror_class in {MirrorClass.PUBLIC.value, MirrorClass.PRIVATE.value}
    }
    fatal_missing = {key for key in missing if key in current_keys}
    scoped_missing = missing - fatal_missing
    if (unattested - litter_keys) or fatal_missing:
        orphan = sorted(unattested - litter_keys, key=str)
        raise ArtifactCheckpointError(
            f"mirror inventory mismatch: missing={sorted(fatal_missing, key=str)} "
            f"orphan={orphan}"
        )
    if scoped_missing:
        for key in scoped_missing:
            expected_physical.pop(key, None)
        for transaction_id, chain in by_transaction.items():
            head = chain[0]
            tx_key = (
                str(head["stable_id"]),
                str(head["work_id"]),
                ArtifactTransactionId(transaction_id),
            )
            if tx_key in transactions:
                continue
            chain_keys = _ledger_attested_physical_keys((chain,))
            absent = sorted(
                object_key for _cls, object_key in (chain_keys & scoped_missing)
            )
            if not absent:
                continue
            prior = stale_transactions.get(tx_key)
            stale_transactions[tx_key] = StaleArtifactTransaction(
                stable_id=tx_key[0],
                work_id=tx_key[1],
                transaction_id=tx_key[2],
                claims=prior.claims if prior is not None else (),
                reason=(
                    f"artifact custody missing from mirror store: {transaction_id} "
                    f"lacks {absent}"
                ),
            )
    for (mirror_class, object_key), obj in expected_physical.items():
        path = mirrors.root(mirror_class) / object_key
        content = path.read_bytes()
        if hash_bytes(content) != obj.content_sha256 or len(content) != obj.byte_count:
            raise ArtifactCheckpointError(f"physical mirror object changed: {object_key}")
    try:
        pre_public_claim_sweep(
            tuple(object_rows.values()), tuple(accepted_claims.values()), mirrors
        )
    except RuntimeError as exc:
        raise ArtifactCheckpointError(str(exc)) from exc

    public_by_path: dict[str, MirrorObject] = {}
    for claim in accepted_claims.values():
        if claim.license_disposition != RedistributionClass.PUBLIC_OK.value:
            continue
        obj = object_rows[claim.object_id]
        previous_object = public_by_path.setdefault(claim.logical_path, obj)
        if previous_object != obj:
            raise ArtifactCheckpointError(
                f"public logical path has conflicting objects: {claim.logical_path}"
            )
    for logical_path, obj in public_by_path.items():
        path = repository_root / _safe_relative_path(logical_path)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise ArtifactCheckpointError(
                f"public materialization is missing: {logical_path}"
            ) from exc
        if hash_bytes(content) != obj.content_sha256:
            raise ArtifactCheckpointError(f"public materialization changed: {logical_path}")
    for claim in accepted_claims.values():
        if claim.license_disposition == RedistributionClass.PUBLIC_OK.value:
            continue
        path = repository_root / _safe_relative_path(claim.logical_path)
        obj = object_rows[claim.object_id]
        if path.exists() and not _public_authorizes_path(
            events, claim.logical_path, obj.content_sha256
        ):
            raise ArtifactCheckpointError(
                f"private-only materialization appears in repository: {claim.logical_path}"
            )
    public_path_digests: dict[str, set[str]] = {}
    for event in events:
        _record_public_path_authorities(event, object_rows, public_path_digests)
    state = _CheckpointFoldState(
        events=list(events),
        seen_event_ids={str(event["artifact_event_id"]) for event in events},
        chains=by_transaction,
        object_rows=object_rows,
        custody_claims=custody_claims,
        accepted_claims=accepted_claims,
        transactions=transactions,
        stale_transactions=stale_transactions,
        expected_physical=expected_physical,
        public_by_path=public_by_path,
        sorted_objects=sorted(object_rows.values(), key=lambda value: str(value.object_id)),
        sorted_accepted=sorted(accepted_claims.values(), key=lambda value: str(value.claim_id)),
        public_path_digests=public_path_digests,
        store_litter=store_litter,
    )
    return state, state.build_projection()


class ArtifactCheckpointIncrement:
    """Validated checkpoint projection that folds appended suffixes in O(delta).

    :meth:`start` is exactly one full checkpoint validation (same code path,
    same refusals) that additionally retains the whole-store fold registries.
    :meth:`advance` then folds ONLY newly appended events: every touched
    transaction re-validates completely live (chain shape, schema, document,
    reconstruction bytes, its mirror objects, its materializations) through
    the same ``_fold_transaction_chain`` used by full validation, and the
    delta's global-registry consequences (event replay, object/claim
    collisions, claim-object linkage, physical byte verification, public
    authority, materialization conflicts) are checked against the carried
    registries.

    Soundness boundary (documented, deliberate): an advance re-verifies the
    bytes belonging to the delta; bytes belonging to untouched transactions
    were verified by this instance's own full validation at :meth:`start` and
    are not re-read per advance. Out-of-band mutation between advances is
    caught by the next full validation (every cold start, every run start,
    the one-shot ``checkpoint`` command). Today's per-close full re-read is
    exactly the O(history)-per-model cost this class exists to remove; the
    holder must own the shard's exclusive writer lock for the lifetime of
    the instance.
    """

    def __init__(
        self,
        state: _CheckpointFoldState,
        projection: ArtifactCheckpointProjection,
        *,
        mirrors: MirrorStore,
        canonical_root: Path,
        repository_root: Path,
        memo: Optional["ValidationMemo"] = None,
    ) -> None:
        """Bind retained fold state; use :meth:`start` instead of calling this.

        Parameters
        ----------
        state, projection:
            Registries and projection from one full checkpoint validation.
        mirrors, canonical_root, repository_root, memo:
            Unchanged fold inputs.
        """

        self._state = state
        self._projection = projection
        self._mirrors = mirrors
        self._canonical_root = canonical_root
        self._repository_root = repository_root
        self._memo = memo

    @classmethod
    def start(
        cls,
        artifact_ledger_paths: Iterable[Path],
        *,
        context: AuthorityContext,
        mirrors: MirrorStore,
        canonical_root: Path,
        repository_root: Path,
        memo: Optional["ValidationMemo"] = None,
    ) -> "ArtifactCheckpointIncrement":
        """Run one full checkpoint validation and retain its fold state.

        Parameters
        ----------
        artifact_ledger_paths, context, mirrors, canonical_root,
        repository_root, memo:
            Exactly the inputs of :func:`validate_artifact_checkpoint`.

        Returns
        -------
        ArtifactCheckpointIncrement
            Fully validated incremental fold.

        Raises
        ------
        ArtifactCheckpointError
            Exactly like :func:`validate_artifact_checkpoint`.
        """

        del context  # Signature stability with validate_artifact_checkpoint.
        state, projection = _full_checkpoint_fold(
            artifact_ledger_paths,
            mirrors=mirrors,
            canonical_root=canonical_root,
            repository_root=repository_root,
            memo=memo,
        )
        return cls(
            state,
            projection,
            mirrors=mirrors,
            canonical_root=canonical_root,
            repository_root=repository_root,
            memo=memo,
        )

    @property
    def projection(self) -> ArtifactCheckpointProjection:
        """Return the current validated projection.

        Returns
        -------
        ArtifactCheckpointProjection
            Projection equal to full validation of the folded history.
        """

        return self._projection

    @property
    def generation(self) -> int:
        """Return the number of folded events.

        Returns
        -------
        int
            Folded event count across every consumed shard and suffix.
        """

        return len(self._state.events)

    def advance(self, new_events: Sequence[JsonObject]) -> ArtifactCheckpointProjection:
        """Fold an exact appended suffix and return the updated projection.

        Parameters
        ----------
        new_events:
            Events appended after this instance's folded generation, in
            ledger order.

        Returns
        -------
        ArtifactCheckpointProjection
            Projection equal to full validation of the extended history.

        Raises
        ------
        ArtifactTransitionError
            If the suffix replays a folded event or breaks chain shape.
        ArtifactCheckpointError
            For GLOBAL ledger/store integrity, exactly like full validation.
            After any raise this instance is defunct: the holder must fall
            back to a fresh full validation.
        """

        events = tuple(new_events)
        if not events:
            return self._projection
        state = self._state
        touched: dict[str, list[JsonObject]] = {}
        for event in events:
            event_id = str(event.get("artifact_event_id"))
            if event_id in state.seen_event_ids:
                raise ArtifactTransitionError(f"duplicate or invalid artifact event ID: {event_id}")
            touched.setdefault(str(event.get("transaction_id")), []).append(event)
        object_mark = len(state.object_rows.insertion_log)
        custody_mark = len(state.custody_claims.insertion_log)
        accepted_mark = len(state.accepted_claims.insertion_log)
        for transaction_id, delta in touched.items():
            chain = state.chains.setdefault(transaction_id, [])
            chain.extend(delta)
            # Complete per-chain revalidation: schema (memoized for already
            # validated events), intrinsic event identity, chain shape, and
            # identity-mutation checks -- identical to full validation.
            validate_artifact_event_chains(chain, memo=self._memo)
            head = chain[0]
            key = (
                str(head["stable_id"]),
                str(head["work_id"]),
                ArtifactTransactionId(transaction_id),
            )
            state.transactions.pop(key, None)
            state.stale_transactions.pop(key, None)
            _fold_transaction_chain(
                transaction_id,
                chain,
                object_rows=state.object_rows,
                custody_claims=state.custody_claims,
                accepted_claims=state.accepted_claims,
                transactions=state.transactions,
                stale_transactions=state.stale_transactions,
                canonical_root=self._canonical_root,
                repository_root=self._repository_root,
                memo=self._memo,
            )
        state.events.extend(events)
        for event in events:
            state.seen_event_ids.add(str(event["artifact_event_id"]))
            _record_public_path_authorities(event, state.object_rows, state.public_path_digests)
        new_objects = state.object_rows.insertion_log[object_mark:]
        new_accepted = state.accepted_claims.insertion_log[accepted_mark:]
        new_claims = state.custody_claims.insertion_log[custody_mark:] + new_accepted
        self._verify_delta_globals(new_objects, new_claims, new_accepted)
        for obj in new_objects:
            insort(state.sorted_objects, obj, key=lambda value: str(value.object_id))
        for claim in new_accepted:
            insort(state.sorted_accepted, claim, key=lambda value: str(value.claim_id))
        self._projection = state.build_projection()
        return self._projection

    def _verify_delta_globals(
        self,
        new_objects: Sequence[MirrorObject],
        new_claims: Sequence[ArtifactClaim],
        new_accepted: Sequence[ArtifactClaim],
    ) -> None:
        """Run the delta-scoped whole-store checks of full validation.

        Parameters
        ----------
        new_objects, new_claims, new_accepted:
            Registry entries added by this advance.

        Raises
        ------
        ArtifactCheckpointError
            On any linkage, inventory, byte, or authority violation.
        """

        state = self._state
        pending_markers = {"", DependencyState.PENDING_UNTRUSTED.value, "not-applicable"}
        for claim in new_claims:
            if claim.object_id not in state.object_rows:
                raise ArtifactCheckpointError(f"claim references absent object: {claim.claim_id}")
        for obj in new_objects:
            if obj.mirror_class not in {MirrorClass.PUBLIC.value, MirrorClass.PRIVATE.value}:
                continue
            physical_key = (MirrorClass(obj.mirror_class), obj.object_key)
            previous_object = state.expected_physical.setdefault(physical_key, obj)
            if _intrinsic_physical_inventory(previous_object) != _intrinsic_physical_inventory(
                obj
            ):
                raise ArtifactCheckpointError(
                    f"physical mirror object has ambiguous intrinsic inventory: {physical_key}"
                )
            path = self._mirrors.root(physical_key[0]) / obj.object_key
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise ArtifactCheckpointError(
                    f"mirror inventory mismatch: missing={[physical_key]} orphan=[]"
                ) from exc
            if hash_bytes(content) != obj.content_sha256 or len(content) != obj.byte_count:
                raise ArtifactCheckpointError(f"physical mirror object changed: {obj.object_key}")
        # Delta-scoped public claim sweep: class consistency for every new
        # claim, accepted public authority plus canonical addressing for
        # every new public object.
        for claim in new_accepted:
            claimed_object = state.object_rows[claim.object_id]
            expected_class = (
                MirrorClass.PUBLIC.value
                if claim.license_disposition == RedistributionClass.PUBLIC_OK.value
                else MirrorClass.PRIVATE.value
            )
            if claimed_object.mirror_class != expected_class:
                raise ArtifactCheckpointError(
                    f"{claim.claim_id}: {claim.license_disposition} claim references "
                    f"{claimed_object.mirror_class} object"
                )
        for obj in new_objects:
            if obj.mirror_class != MirrorClass.PUBLIC.value:
                continue
            accepted = tuple(
                claim
                for claim in state.accepted_claims.values()
                if claim.object_id == obj.object_id
                and claim.license_disposition == RedistributionClass.PUBLIC_OK.value
                and str(claim.gate_id) not in pending_markers
                and str(claim.authorization_id) not in pending_markers
            )
            if not accepted:
                raise ArtifactCheckpointError(
                    f"{obj.object_id}: public object lacks an accepted public claim"
                )
            try:
                path = self._mirrors.address(obj.content_sha256, MirrorClass.PUBLIC)
                expected_key = path.relative_to(self._mirrors.root(MirrorClass.PUBLIC)).as_posix()
            except (OSError, RuntimeError) as exc:
                raise ArtifactCheckpointError(
                    f"{obj.object_id}: public object is not retrievable: {exc}"
                ) from exc
            if obj.object_key != expected_key:
                raise ArtifactCheckpointError(
                    f"{obj.object_id}: public object key is not canonical"
                )
        for claim in new_accepted:
            obj = state.object_rows[claim.object_id]
            if claim.license_disposition == RedistributionClass.PUBLIC_OK.value:
                previous_object = state.public_by_path.setdefault(claim.logical_path, obj)
                if previous_object != obj:
                    raise ArtifactCheckpointError(
                        f"public logical path has conflicting objects: {claim.logical_path}"
                    )
                path = self._repository_root / _safe_relative_path(claim.logical_path)
                try:
                    content = path.read_bytes()
                except OSError as exc:
                    raise ArtifactCheckpointError(
                        f"public materialization is missing: {claim.logical_path}"
                    ) from exc
                if hash_bytes(content) != obj.content_sha256:
                    raise ArtifactCheckpointError(
                        f"public materialization changed: {claim.logical_path}"
                    )
            else:
                path = self._repository_root / _safe_relative_path(claim.logical_path)
                if path.exists() and obj.content_sha256 not in state.public_path_digests.get(
                    claim.logical_path, ()
                ):
                    raise ArtifactCheckpointError(
                        f"private-only materialization appears in repository: {claim.logical_path}"
                    )


def resolve_final_artifact_transaction(
    projection: ArtifactCheckpointProjection,
    *,
    stable_id: str,
    work_id: str,
    transaction_id: Optional[ArtifactTransactionId] = None,
) -> Optional[ArtifactTransactionProjection]:
    """Resolve exact final authority without latest-transaction selection.

    Parameters
    ----------
    projection:
        Fully verified normalized artifact authority.
    stable_id, work_id:
        Exact active model and work generation.
    transaction_id:
        Preferred transaction recorded by current model authority. When present,
        absence is an error rather than permission to choose another transaction --
        except a schema-stale entry, which reads as a cache miss.

    Returns
    -------
    ArtifactTransactionProjection | None
        Sole exact final transaction, or ``None`` when no final authority exists
        or the recorded/matching authority is a schema-stale reuse-cache entry
        (the caller then routes the model to fresh authorship).

    Raises
    ------
    ArtifactRehydrationError
        If recorded authority is missing/wrong or unrecorded final authority is
        ambiguous for the active work generation.
    """

    if transaction_id is not None:
        key = (stable_id, work_id, transaction_id)
        exact = projection.transactions.get(key)
        if exact is None:
            if key in projection.stale_transactions:
                # The recorded authority exists and byte-verifies, but its
                # author result no longer validates against the live result
                # schema: a stale reuse-cache entry, consumed as a miss.
                return None
            raise ArtifactRehydrationError(
                "recorded artifact transaction is absent from exact final authority"
            )
        return exact
    matches = tuple(
        transaction
        for (candidate_stable, candidate_work, _candidate_id), transaction in (
            projection.transactions.items()
        )
        if candidate_stable == stable_id and candidate_work == work_id
    )
    if len(matches) > 1:
        raise ArtifactRehydrationError(
            "active work has multiple final artifact transactions and no recorded selection"
        )
    return matches[0] if matches else None


def unrehydratable_authority_census(
    checkpoint: ArtifactCheckpointProjection,
) -> tuple[JsonObject, ...]:
    """Classify every finalized transaction's rehydration readiness, read-only.

    Dry-runs the PRODUCTION claim-coverage predicate (:func:`_rehydration_targets`)
    over each projected final transaction and reports one row per transaction:
    ``rehydratable``, or ``unrehydratable`` with the exact typed refusal. The
    census steers no disposition and can only disclose -- it is the derived,
    replay-computed input the close-time artifact/record incoherence censuses
    (AT-8/IC-12) consume, and the post-merge probe that proved the 7 poisoned
    200a transactions flipped. Quarantined stale transactions are ALREADY
    disclosed cache misses and are not double-counted here.

    Parameters
    ----------
    checkpoint:
        Validated artifact checkpoint projection.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One disposition row per finalized transaction, in key order.
    """

    rows: list[JsonObject] = []
    for key in sorted(checkpoint.transactions, key=lambda item: (item[0], item[1], str(item[2]))):
        transaction = checkpoint.transactions[key]
        row: JsonObject = {
            "stable_id": transaction.stable_id,
            "work_id": transaction.work_id,
            "transaction_id": str(transaction.transaction_id),
            "final_event_kind": transaction.final_event_kind,
        }
        try:
            _rehydration_targets(transaction)
        except ArtifactRehydrationError as exc:
            row["disposition"] = "unrehydratable"
            row["reason"] = str(exc)
        else:
            row["disposition"] = "rehydratable"
        rows.append(row)
    return tuple(rows)


def _rehydration_targets(
    transaction: ArtifactTransactionProjection,
) -> Mapping[ArtifactClaimId, Path]:
    """Validate exact claim coverage and derive disposable relative targets.

    Parameters
    ----------
    transaction:
        Verified final transaction selected from the normalized projection.

    Returns
    -------
    Mapping[ArtifactClaimId, pathlib.Path]
        Claim-keyed paths relative to the transaction staging directory.

    Raises
    ------
    ArtifactRehydrationError
        If claims, source lineage, logical paths, or executable manifests differ.
    """

    objects = {obj.object_id: obj for obj in transaction.objects}
    if len(objects) != len(transaction.objects):
        raise ArtifactRehydrationError("rehydration object inventory has duplicate identities")
    inputs = transaction.reconstruction_inputs
    for field, rows in (
        ("sources", inputs.source_manifest.get("sources")),
        (SUPPLEMENTARY_SOURCES_KEY, inputs.source_manifest.get(SUPPLEMENTARY_SOURCES_KEY, [])),
    ):
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise ArtifactRehydrationError(f"rehydration source manifest is malformed: {field}")
    # CITABLE, not frozen: the write side staged byte custody over
    # `manifest_source_rows(...)` (the frozen rows plus the one supplementary
    # broker pack) and refused staging without exact coverage of every
    # byte-backed citable row, so the read side must resolve claims against
    # the SAME rows. Reading `sources` alone made every finalized transaction
    # carrying a supplementary source structurally unrehydratable (7 of
    # 200a's 218 finals) -- the same frozen-vs-citable defect
    # `manifest_source_rows` was introduced to close at staging.
    sources_value = manifest_source_rows(inputs.source_manifest)
    sources = {str(row["source_id"]): row for row in sources_value}
    if len(sources) != len(sources_value):
        raise ArtifactRehydrationError("rehydration source IDs are not unique")
    by_role: dict[str, list[ArtifactClaim]] = {"source": [], "code": [], "patch": []}
    seen_logical_paths: dict[str, str] = {}
    for claim in transaction.claims:
        if (
            claim.stable_id != transaction.stable_id
            or claim.work_id != transaction.work_id
            or _dependency_value(claim.gate_id) != transaction.accepted_gate_id
            or _dependency_value(claim.authorization_id) != transaction.authorization_id
        ):
            raise ArtifactRehydrationError("rehydration claim lineage differs from transaction")
        if claim.logical_role not in by_role:
            raise ArtifactRehydrationError(
                f"rehydration claim has unsupported role: {claim.logical_role}"
            )
        try:
            _safe_relative_path(claim.logical_path)
        except ArtifactBindingError as exc:
            raise ArtifactRehydrationError(str(exc)) from exc
        obj = objects.get(claim.object_id)
        source = sources.get(claim.source_id)
        if obj is None or source is None:
            raise ArtifactRehydrationError("rehydration claim lacks its exact object or source")
        # The write side's own duplicate rule, mirrored exactly: staging
        # ACCEPTS one logical path staged twice with identical bytes (two
        # citable rows fetching the same content -- the m1115 carried-award
        # shape) and refuses only conflicting bytes; `_atomic_write_immutable`
        # verifies the identical replay at rehydration. A blanket duplicate
        # refusal here made lawfully staged transactions unrehydratable.
        previous_digest = seen_logical_paths.setdefault(claim.logical_path, obj.content_sha256)
        if previous_digest != obj.content_sha256:
            raise ArtifactRehydrationError(
                f"rehydration logical path has conflicting bytes: {claim.logical_path}"
            )
        if claim.origin != source.get("url") or claim.revision != source.get("revision"):
            raise ArtifactRehydrationError("rehydration claim origin differs from source manifest")
        by_role[claim.logical_role].append(claim)
    source_claim_ids = {claim.source_id for claim in by_role["source"]}
    # The write side's own coverage rule, mirrored exactly: staging required a
    # source-role byte for every BYTE-BACKED citable row, no more and no less.
    byte_backed_ids = {
        source_id
        for source_id, row in sources.items()
        if isinstance(row.get("content_sha256"), str)
    }
    if source_claim_ids != byte_backed_ids or len(by_role["source"]) != len(byte_backed_ids):
        raise ArtifactRehydrationError("rehydration source claims differ from exact source set")
    targets: dict[ArtifactClaimId, Path] = {}
    for claim in by_role["source"]:
        obj = objects[claim.object_id]
        source = sources[claim.source_id]
        expected_size = source.get("fetched_bytes_len", source.get("byte_count"))
        if (
            obj.content_sha256 != source.get("content_sha256")
            or expected_size not in {None, obj.byte_count}
            or source.get("media_type") not in {None, obj.media_type}
        ):
            raise ArtifactRehydrationError(
                f"rehydration source object differs from manifest: {claim.source_id}"
            )
        targets[claim.claim_id] = Path("sources") / (
            f"{obj.content_sha256.removeprefix('sha256:')}.source"
        )

    proposal = inputs.proposal
    if proposal is None and isinstance(inputs.handoff_execution, Mapping):
        candidate = inputs.handoff_execution.get("proposal")
        proposal = dict(candidate) if isinstance(candidate, Mapping) else None
    implementation: Optional[Mapping[str, Any]] = None
    if proposal is not None:
        facts = proposal.get("proposed_facts")
        candidate = facts.get("implementation") if isinstance(facts, Mapping) else None
        if not isinstance(candidate, Mapping):
            raise ArtifactRehydrationError("rehydration proposal implementation is malformed")
        implementation = candidate
    for role, field in (("code", "code_manifest"), ("patch", "patches")):
        raw_rows = implementation.get(field, []) if implementation is not None else []
        if not isinstance(raw_rows, list):
            raise ArtifactRehydrationError(f"rehydration proposal {field} is malformed")
        unmatched = list(by_role[role])
        for row in raw_rows:
            if (
                not isinstance(row, Mapping)
                or not isinstance(row.get("path"), str)
                or not isinstance(row.get("sha256"), str)
            ):
                raise ArtifactRehydrationError(f"rehydration proposal {field} row is malformed")
            try:
                relative = _safe_relative_path(str(row["path"]))
            except ArtifactBindingError as exc:
                raise ArtifactRehydrationError(str(exc)) from exc
            declared_parts = PurePosixPath(str(row["path"])).parts
            matches = [
                claim
                for claim in unmatched
                if PurePosixPath(claim.logical_path).parts[-len(declared_parts) :] == declared_parts
                and objects[claim.object_id].content_sha256 == row["sha256"]
            ]
            if len(matches) != 1:
                raise ArtifactRehydrationError(
                    f"rehydration executable claim differs from {field}: {row['path']}"
                )
            claim = matches[0]
            unmatched.remove(claim)
            target = Path("model") / relative
            if target in targets.values():
                raise ArtifactRehydrationError(
                    f"rehydration executable target conflicts: {relative.as_posix()}"
                )
            targets[claim.claim_id] = target
        if unmatched:
            raise ArtifactRehydrationError(
                f"rehydration has unexpected executable claims for {field}"
            )
    if set(targets) != {claim.claim_id for claim in transaction.claims}:
        raise ArtifactRehydrationError("rehydration omitted one or more final claims")
    return MappingProxyType(targets)


def rehydrate_artifact_transaction(
    transaction: ArtifactTransactionProjection,
    *,
    mirrors: MirrorStore,
    staging_root: Path,
) -> RehydratedArtifactTransaction:
    """Materialize exact final code, patch, and source claims from private custody.

    Parameters
    ----------
    transaction:
        Exact final transaction selected from a verified projection.
    mirrors:
        Canonical public/private/local mirror roots. Reads are forced through the
        retained private custody copy even for public-compatible claims.
    staging_root:
        Disposable root beneath which a transaction-addressed directory is created.

    Returns
    -------
    RehydratedArtifactTransaction
        Verified staged handle containing raw reconstruction inputs and claim paths.

    Raises
    ------
    ArtifactRehydrationError
        If custody bytes, paths, coverage, aliases, or existing staged bytes differ.
    """

    targets = _rehydration_targets(transaction)
    for component in (transaction.stable_id, transaction.work_id, str(transaction.transaction_id)):
        pure = PurePosixPath(component)
        if not component or pure.is_absolute() or len(pure.parts) != 1 or component in {".", ".."}:
            raise ArtifactRehydrationError(
                f"rehydration transaction address has unsafe component: {component!r}"
            )
    root = (
        staging_root / transaction.stable_id / transaction.work_id / str(transaction.transaction_id)
    )
    cursor = staging_root
    for component in (
        transaction.stable_id,
        transaction.work_id,
        str(transaction.transaction_id),
    ):
        if cursor.is_symlink():
            raise ArtifactRehydrationError(
                f"rehydration staging ancestor cannot be a symlink: {cursor}"
            )
        cursor /= component
    if root.is_symlink():
        raise ArtifactRehydrationError("rehydration root cannot be a symlink")
    objects = {obj.object_id: obj for obj in transaction.objects}
    # Source claims are staged twice: once at their claim-addressed path and
    # once at the canonical author-layout CAS path. The fresh terminal lane's
    # consumers dereference frozen source bytes at ``<author_root>/source-cas``
    # (the checker envelope's declared read root and the terminal evidence
    # re-derivation both), so a rehydrated transaction that omits that tree
    # cannot buy the review its claim is owed -- every declared excerpt reads
    # as unresolved and a rejected verdict refuses at gate routing.
    cas_twins: dict[ArtifactClaimId, Path] = {}
    for claim in transaction.claims:
        if targets[claim.claim_id].parts[0] != "sources":
            continue
        hex_digest = objects[claim.object_id].content_sha256.removeprefix("sha256:")
        cas_twins[claim.claim_id] = Path("source-cas") / "sha256" / hex_digest[:2] / hex_digest
    expected_relative = set(targets.values()) | set(cas_twins.values())
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ArtifactRehydrationError(f"rehydration staging alias is forbidden: {path}")
        observed = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
        if observed - expected_relative:
            raise ArtifactRehydrationError(
                f"rehydration staging has extra files: {sorted(map(str, observed - expected_relative))}"
            )
    claim_paths: dict[ArtifactClaimId, Path] = {}
    for claim in transaction.claims:
        obj = objects[claim.object_id]
        try:
            content = mirrors.fetch_object(obj, custody_class=MirrorClass.PRIVATE)
        except RuntimeError as exc:
            raise ArtifactRehydrationError(
                f"private custody is unavailable for claim {claim.claim_id}: {exc}"
            ) from exc
        twin = cas_twins.get(claim.claim_id)
        destinations = (root / targets[claim.claim_id],) + (
            (root / twin,) if twin is not None else ()
        )
        for destination in destinations:
            try:
                if destination.is_symlink():
                    raise ArtifactRehydrationError(
                        f"rehydration destination cannot be a symlink: {destination}"
                    )
                _atomic_write_immutable(destination, content)
            except ArtifactPublicationError as exc:
                raise ArtifactRehydrationError(str(exc)) from exc
            if destination.is_symlink() or destination.stat().st_nlink != 1:
                raise ArtifactRehydrationError(
                    f"rehydration destination has an alias: {destination}"
                )
            if hash_bytes(destination.read_bytes()) != obj.content_sha256:
                raise ArtifactRehydrationError(
                    f"rehydration destination changed after write: {destination}"
                )
        claim_paths[claim.claim_id] = root / targets[claim.claim_id]
    return RehydratedArtifactTransaction(
        transaction=transaction,
        root=root,
        model_dir=root / "model",
        claim_paths=claim_paths,
    )


def artifact_reconstruction_paths(
    artifact_ledger_paths: Iterable[Path], repository_root: Path
) -> tuple[Path, ...]:
    """Return immutable reconstruction paths named by append-only events.

    Parameters
    ----------
    artifact_ledger_paths:
        Complete artifact ledger shards.
    repository_root:
        Public repository root used to resolve safe relative paths.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Sorted unique reconstruction files.
    """

    events = _load_events(artifact_ledger_paths)
    paths = {
        repository_root
        / _safe_relative_path(ReconstructionAnchorRecord.from_payload(reconstruction).path)
        for event in events
        if isinstance((reconstruction := event.get("reconstruction")), Mapping)
    }
    return tuple(sorted(paths, key=lambda value: value.as_posix()))
