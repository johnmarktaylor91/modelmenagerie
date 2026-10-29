"""Round-14 Root-B artifact transaction and checkpoint regressions."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from menagerie.crawler.artifact_transactions import (
    ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION,
    ArtifactBindingError,
    ArtifactCheckpointError,
    ArtifactEventKind,
    ArtifactEventLedger,
    ArtifactInput,
    ArtifactPublicationError,
    ArtifactRehydrationError,
    ReconstructionInputs,
    _atomic_write_immutable,
    _validate_projection_dependencies,
    _validate_reconstruction_document,
    append_artifact_authorization,
    authored_code_logical_path,
    derive_artifact_claims,
    derive_publication_authorization_id,
    publish_authorized_artifact,
    rehydrate_artifact_transaction,
    resolve_final_artifact_transaction,
    stage_private_artifact,
    validate_artifact_checkpoint,
)
from menagerie.crawler.authority import (
    ArtifactClaimId,
    AuthorityContext,
    DependencyState,
    DependencyValue,
    DependencyVector,
    PublicationAuthorization,
    PublicationAuthorizationId,
)
from menagerie.crawler.checkpoint import CheckpointValidationError, _derive_candidate_paths
from menagerie.crawler.constants import (
    AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
    AUTHOR_RESULT_SCHEMA_VERSION,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.licenses import LicenseDecision, RedistributionClass
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorClass, MirrorStore
from menagerie.crawler.tests.conftest import make_author_proposal

#: Sub-second load-bearing coverage promoted into the mandatory smoke tier (T5-7).
pytestmark = pytest.mark.smoke

HASH = "sha256:" + "a" * 64
NOW = "2026-07-16T12:00:00Z"


def _context(*stable_ids: str) -> AuthorityContext:
    """Build exact active authority for focused artifact tests.

    Parameters
    ----------
    stable_ids:
        Trusted intake identities.

    Returns
    -------
    AuthorityContext
        Frozen test authority.
    """

    intake = {
        stable_id: {"stable_id": stable_id, "natural_key": f"fixture:{stable_id}"}
        for stable_id in stable_ids
    }
    return AuthorityContext(
        active_intake_snapshot_id="intake-test",
        active_intake_snapshot_sha256=HASH,
        intake_by_stable_id=intake,
        family_bindings={},
        author_prompt_identity=HASH,
        author_model_identity=HASH,
        author_schema_identity=HASH,
        author_dispatcher_identity=HASH,
        author_model_fields={
            "provider": "anthropic",
            "model": "claude-fixture",
            "version": "current",
            "prompt_sha256": HASH,
        },
        checker_prompt_identity=HASH,
        checker_model_identity=HASH,
        checker_schema_identity=HASH,
        checker_model_fields={
            "provider": "openai",
            "model": "gpt-fixture",
            "version": "current",
            "prompt_sha256": HASH,
        },
        environment_generations={},
        reducer_policy_identity=HASH,
        runner_policy_identity=HASH,
        terminal_policy_identity=HASH,
        publication_policy_identity=HASH,
    )


def _source_manifest(content: bytes) -> dict[str, Any]:
    """Build one exact controlled-fetch source manifest.

    Parameters
    ----------
    content:
        Frozen upstream bytes.

    Returns
    -------
    dict[str, Any]
        Hash-bound source manifest.
    """

    source = {
        "source_id": "src-upstream",
        "url": "https://example.test/source.tar",
        "revision": "v1",
        "content_sha256": hash_bytes(content),
        "fetched_bytes_len": len(content),
        "media_type": "application/x-tar",
    }
    return {"sources": [source], "manifest_sha256": stable_hash([source])}


def _blocked_result(
    stable_id: str, context: AuthorityContext, source_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Build one valid typed blocked recommendation.

    Parameters
    ----------
    stable_id, context, source_manifest:
        Exact model, active trust roots, and source manifest.

    Returns
    -------
    dict[str, Any]
        Valid ``author-result.v4`` mapping.
    """

    payload = {
        "arm": "BLOCKED",
        "stage": "source",
        "reason_code": "missing-mandatory-link",
        "prerequisite_ids": ["src-upstream"],
        "evidence_ids": ["ev-source"],
        "evidence_identity": HASH,
        "license_identity": HASH,
        "recommendation_sha256": HASH,
    }
    result = {
        "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
        "result_id": f"result-{stable_id}",
        "result_sha256": HASH,
        "kind": "BLOCKED",
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "campaign_id": "campaign-test",
        "created_at": NOW,
        "author_identity": context.author_model_identity,
        "prompt_identity": context.author_prompt_identity,
        "dispatcher_identity": context.author_dispatcher_identity,
        "source_manifest_identity": source_manifest["manifest_sha256"],
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "intake_item_sha256": stable_hash(context.intake_by_stable_id[stable_id]),
        "payload": payload,
    }
    result["result_sha256"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return result


def _proposed_result(
    stable_id: str, context: AuthorityContext, content: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a valid proposed result and matching controlled source manifest.

    Parameters
    ----------
    stable_id, context, content:
        Exact model, active trust roots, and upstream bytes.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        Typed author result, embedded proposal, and source manifest.
    """

    proposal = make_author_proposal(stable_id)
    digest = hash_bytes(content)
    source = proposal["proposed_facts"]["source_resolution"]["sources"][0]
    source.update(
        {
            "source_id": "src-upstream",
            "url": "https://example.test/source.tar",
            "revision": "v1",
            "content_sha256": digest,
            "byte_count": len(content),
            "media_type": "application/x-tar",
        }
    )
    fetched_source = {
        "source_id": "src-upstream",
        "url": "https://example.test/source.tar",
        "revision": "v1",
        "content_sha256": digest,
        "fetched_bytes_len": len(content),
        "media_type": "application/x-tar",
    }
    source_manifest = {
        "sources": [fetched_source],
        "manifest_sha256": stable_hash([fetched_source]),
    }
    proposal.update(
        {
            "schema_version": AUTHOR_PROPOSAL_SCHEMA_VERSION_V3,
            "campaign_id": "campaign-test",
            "intake_snapshot_id": context.active_intake_snapshot_id,
            "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
            "intake_item_sha256": stable_hash(context.intake_by_stable_id[stable_id]),
            "source_manifest_identity": source_manifest["manifest_sha256"],
            "dispatcher_identity": context.author_dispatcher_identity,
        }
    )
    proposal["author"]["prompt_sha256"] = context.author_prompt_identity
    proposal["verified_hashes"]["source_manifest"] = source_manifest["manifest_sha256"]
    proposal["proposal_sha256"] = stable_hash(
        {key: value for key, value in proposal.items() if key != "proposal_sha256"}
    )
    result = {
        "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
        "result_id": f"result-{stable_id}",
        "result_sha256": HASH,
        "kind": "PROPOSED",
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "campaign_id": "campaign-test",
        "created_at": NOW,
        "author_identity": context.author_model_identity,
        "prompt_identity": context.author_prompt_identity,
        "dispatcher_identity": context.author_dispatcher_identity,
        "source_manifest_identity": source_manifest["manifest_sha256"],
        "intake_snapshot_id": context.active_intake_snapshot_id,
        "intake_snapshot_sha256": context.active_intake_snapshot_sha256,
        "intake_item_sha256": stable_hash(context.intake_by_stable_id[stable_id]),
        "payload": {"arm": "PROPOSED", "proposal": proposal},
    }
    result["result_sha256"] = stable_hash(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return result, proposal, source_manifest


def _mirrors(tmp_path: Path) -> MirrorStore:
    """Create separated physical mirrors.

    Parameters
    ----------
    tmp_path:
        Test root.

    Returns
    -------
    MirrorStore
        Separated public/private/local roots.
    """

    return MirrorStore(
        tmp_path / "mirror-public",
        tmp_path / "mirror-private",
        tmp_path / "mirror-local",
    )


def _vector(
    context: AuthorityContext,
    stable_id: str,
    transaction_id: str,
    result_id: str,
    source_manifest_identity: str,
    claim_ids: tuple[ArtifactClaimId, ...] = (),
    proposal_identity: DependencyValue = DependencyState.NOT_APPLICABLE,
    checker_gate_identity: str = "gate-terminal",
) -> DependencyVector:
    """Build a transaction-bound dependency vector.

    Parameters
    ----------
    context, stable_id, transaction_id, result_id, source_manifest_identity,
    claim_ids, proposal_identity, checker_gate_identity:
        Exact authority axes and optional accepted claim set.

    Returns
    -------
    DependencyVector
        Frozen dependency vector.
    """

    return DependencyVector(
        intake_snapshot_id=context.active_intake_snapshot_id,
        intake_snapshot_sha256=context.active_intake_snapshot_sha256,
        intake_item_sha256=stable_hash(context.intake_by_stable_id[stable_id]),
        author_result_schema_identity=context.author_schema_identity,
        author_dispatcher_identity=context.author_dispatcher_identity,
        author_prompt_identity=context.author_prompt_identity,
        checker_prompt_identity=context.checker_prompt_identity,
        terminal_rule_identity=context.terminal_policy_identity,
        status_proof_identity=HASH,
        source_manifest_identity=source_manifest_identity,
        proposal_identity=proposal_identity,
        author_result_identity=result_id,
        checker_gate_identity=checker_gate_identity,
        recipe_revision=DependencyState.NOT_APPLICABLE,
        runner_identity=DependencyState.NOT_APPLICABLE,
        award_closure_identity=DependencyState.NOT_APPLICABLE,
        environment_generation=DependencyState.NOT_APPLICABLE,
        accepted_attempt_ids=(),
        artifact_transaction_id=transaction_id,
        artifact_claim_ids=claim_ids,
        representative_revision=DependencyState.NOT_APPLICABLE,
        publication_policy_identity=context.publication_policy_identity,
    )


def _gate_item(stable_id: str, result: dict[str, Any], source_identity: str) -> dict[str, Any]:
    """Build one accepted terminal gate item.

    Parameters
    ----------
    stable_id, result, source_identity:
        Exact model, author result, and source-manifest identity.

    Returns
    -------
    dict[str, Any]
        Accepted terminal-disposition item.
    """

    return {
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "terminal_disposition": {
            "verdict": "accepted",
            "author_result_id": result["result_id"],
            "author_result_sha256": result["result_sha256"],
            "handoff_proposal_id": None,
            "handoff_sha256": None,
            "source_manifest_identity": source_identity,
            "source_ids": ["src-upstream"],
        },
    }


def _commit_private(
    stable_id: str,
    *,
    content: bytes,
    context: AuthorityContext,
    mirrors: MirrorStore,
    ledger: ArtifactEventLedger,
    canonical_root: Path,
    repository_root: Path,
    evidence_id: str,
) -> tuple[Any, Any]:
    """Stage, authorize, and privately commit one test transaction.

    Parameters
    ----------
    stable_id, content, context, mirrors, ledger, canonical_root,
    repository_root, evidence_id:
        Complete deterministic transaction inputs.

    Returns
    -------
    tuple[StagedArtifact, PublishedArtifact]
        Private stage and completed commitment.
    """

    source_manifest = _source_manifest(content)
    result = _blocked_result(stable_id, context, source_manifest)
    digest = hash_bytes(content)
    staged = stage_private_artifact(
        (
            ArtifactInput(
                content=content,
                content_sha256=digest,
                logical_role="source",
                logical_path=f"menagerie/crawler/source_cas/{digest[7:]}.source",
                source_id="src-upstream",
                origin=ArtifactOrigin("https://example.test/source.tar", "v1"),
                fetch_recipe="GET exact source.tar@v1",
                evidence_ids=(evidence_id,),
                media_type="application/x-tar",
            ),
        ),
        context=context,
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        author_result=result,
        proposal=None,
        source_manifest=source_manifest,
        mirrors=mirrors,
        ledger=ledger,
        created_at=NOW,
    )
    gate_item = _gate_item(stable_id, result, source_manifest["manifest_sha256"])
    provisional_vector = _vector(
        context,
        stable_id,
        str(staged.transaction_id),
        result["result_id"],
        source_manifest["manifest_sha256"],
    )
    decision = LicenseDecision(
        content_sha256=digest,
        redistribution_class=RedistributionClass.RESTRICTED_PRIVATE,
        evidence_ids=(evidence_id,),
        rationale="restricted fixture",
    )
    decisions = {staged.custody_claims[0].claim_id: decision}
    authorization_id = derive_publication_authorization_id(
        staged,
        accepted_gate_id="gate-terminal",
        accepted_gate_item_sha256=stable_hash(gate_item),
        dependency_vector=provisional_vector,
        decisions=decisions,
        publication_policy_identity=context.publication_policy_identity,
    )
    claims = derive_artifact_claims(
        staged,
        accepted_gate_id="gate-terminal",
        authorization_id=authorization_id,
        decisions=decisions,
        mirrors=mirrors,
    )
    vector = replace(
        provisional_vector, artifact_claim_ids=tuple(claim.claim_id for claim in claims)
    )
    assert authorization_id == derive_publication_authorization_id(
        staged,
        accepted_gate_id="gate-terminal",
        accepted_gate_item_sha256=stable_hash(gate_item),
        dependency_vector=vector,
        decisions=decisions,
        publication_policy_identity=context.publication_policy_identity,
    )
    authorization = PublicationAuthorization(
        authorization_id=authorization_id,
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        transaction_id=staged.transaction_id,
        accepted_gate_id="gate-terminal",
        accepted_gate_item_sha256=stable_hash(gate_item),
        dependency_vector=vector,
        claim_ids=tuple(claim.claim_id for claim in claims),
        public_object_ids=(),
        private_object_ids=tuple(claim.object_id for claim in claims),
        publication_policy_identity=context.publication_policy_identity,
    )
    append_artifact_authorization(
        staged,
        authorization,
        claims,
        accepted_gate_item=gate_item,
        event_kind=ArtifactEventKind.TERMINAL_AUTHORIZED,
        context=context,
        mirrors=mirrors,
        ledger=ledger,
        created_at=NOW,
    )
    published = publish_authorized_artifact(
        staged,
        authorization,
        reconstruction_inputs=ReconstructionInputs(
            author_result=result,
            proposal=None,
            handoff_execution=None,
            source_manifest=source_manifest,
            accepted_gate_item=gate_item,
        ),
        context=context,
        mirrors=mirrors,
        ledger=ledger,
        canonical_root=canonical_root,
        repository_root=repository_root,
        created_at=NOW,
    )
    return staged, published


def test_private_first_transaction_commits_immutable_reconstruction(tmp_path: Path) -> None:
    """All bytes remain private while reconstruction is ledger anchored."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        staged, published = _commit_private(
            "m_one",
            content=b"shared upstream bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
        assert [event["event_kind"] for event in ledger.events] == [
            "staged-private",
            "terminal-authorized",
            "reconstruction-committed",
            "private-committed",
        ]
        assert published.reconstruction_path.is_file()
        assert not (repo / staged.custody_claims[0].logical_path).exists()
        validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )


def test_publication_occurs_only_after_committed_public_authorization(tmp_path: Path) -> None:
    """Accepted proposed bytes publish from retained private custody."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_public")
    mirrors = _mirrors(tmp_path)
    content = b"public-compatible source"
    result, proposal, source_manifest = _proposed_result("m_public", context, content)
    digest = hash_bytes(content)
    logical_path = f"menagerie/crawler/source_cas/{digest[7:]}.source"
    with ArtifactEventLedger(ledger_path) as ledger:
        staged = stage_private_artifact(
            (
                ArtifactInput(
                    content=content,
                    content_sha256=digest,
                    logical_role="source",
                    logical_path=logical_path,
                    source_id="src-upstream",
                    origin=ArtifactOrigin("https://example.test/source.tar", "v1"),
                    fetch_recipe="GET exact source.tar@v1",
                    evidence_ids=("ev-license",),
                    media_type="application/x-tar",
                ),
            ),
            context=context,
            stable_id="m_public",
            work_id="work-m_public",
            author_result=result,
            proposal=proposal,
            source_manifest=source_manifest,
            mirrors=mirrors,
            ledger=ledger,
            created_at=NOW,
        )
        assert tuple(mirrors.iter_objects(MirrorClass.PUBLIC)) == ()
        gate_item = {
            "stable_id": "m_public",
            "work_id": "work-m_public",
            "verdict": "accurate",
            "integrity": {"verdict": "accurate"},
            "rung_check": {"verdict": "accurate"},
            "verified_hashes": {"source_manifest": source_manifest["manifest_sha256"]},
        }
        provisional_vector = _vector(
            context,
            "m_public",
            str(staged.transaction_id),
            result["result_id"],
            source_manifest["manifest_sha256"],
            proposal_identity=proposal["proposal_id"],
            checker_gate_identity="gate-public",
        )
        decision = LicenseDecision(
            content_sha256=digest,
            redistribution_class=RedistributionClass.PUBLIC_OK,
            evidence_ids=("ev-license",),
            rationale="public fixture",
        )
        decisions = {staged.custody_claims[0].claim_id: decision}
        authorization_id = derive_publication_authorization_id(
            staged,
            accepted_gate_id="gate-public",
            accepted_gate_item_sha256=stable_hash(gate_item),
            dependency_vector=provisional_vector,
            decisions=decisions,
            publication_policy_identity=context.publication_policy_identity,
        )
        claims = derive_artifact_claims(
            staged,
            accepted_gate_id="gate-public",
            authorization_id=authorization_id,
            decisions=decisions,
            mirrors=mirrors,
        )
        vector = replace(
            provisional_vector,
            artifact_claim_ids=tuple(claim.claim_id for claim in claims),
        )
        authorization = PublicationAuthorization(
            authorization_id=authorization_id,
            stable_id="m_public",
            work_id="work-m_public",
            transaction_id=staged.transaction_id,
            accepted_gate_id="gate-public",
            accepted_gate_item_sha256=stable_hash(gate_item),
            dependency_vector=vector,
            claim_ids=tuple(claim.claim_id for claim in claims),
            public_object_ids=tuple(claim.object_id for claim in claims),
            private_object_ids=(),
            publication_policy_identity=context.publication_policy_identity,
        )
        append_artifact_authorization(
            staged,
            authorization,
            claims,
            accepted_gate_item=gate_item,
            event_kind=ArtifactEventKind.PUBLICATION_AUTHORIZED,
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            created_at=NOW,
        )
        assert tuple(mirrors.iter_objects(MirrorClass.PUBLIC)) == ()
        publish_authorized_artifact(
            staged,
            authorization,
            reconstruction_inputs=ReconstructionInputs(
                author_result=result,
                proposal=proposal,
                handoff_execution=None,
                source_manifest=source_manifest,
                accepted_gate_item=gate_item,
            ),
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            created_at=NOW,
        )
        assert (repo / logical_path).read_bytes() == content
        assert len(tuple(mirrors.iter_objects(MirrorClass.PRIVATE))) == 1
        assert len(tuple(mirrors.iter_objects(MirrorClass.PUBLIC))) == 1
        validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )


def test_publication_capability_without_ledger_authorization_writes_nothing(
    tmp_path: Path,
) -> None:
    """A directly constructed capability cannot cross the public API boundary."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    content = b"private first"
    source_manifest = _source_manifest(content)
    result = _blocked_result("m_one", context, source_manifest)
    with ArtifactEventLedger(ledger_path) as ledger:
        staged = stage_private_artifact(
            (
                ArtifactInput(
                    content=content,
                    content_sha256=hash_bytes(content),
                    logical_role="source",
                    logical_path="menagerie/crawler/source_cas/object.source",
                    source_id="src-upstream",
                    origin=ArtifactOrigin("https://example.test/source.tar", "v1"),
                    fetch_recipe="GET exact source.tar@v1",
                    media_type="application/x-tar",
                ),
            ),
            context=context,
            stable_id="m_one",
            work_id="work-m_one",
            author_result=result,
            proposal=None,
            source_manifest=source_manifest,
            mirrors=mirrors,
            ledger=ledger,
            created_at=NOW,
        )
        vector = _vector(
            context,
            "m_one",
            str(staged.transaction_id),
            result["result_id"],
            source_manifest["manifest_sha256"],
        )
        forged = PublicationAuthorization(
            authorization_id=PublicationAuthorizationId("forged"),
            stable_id="m_one",
            work_id="work-m_one",
            transaction_id=staged.transaction_id,
            accepted_gate_id="gate-forged",
            accepted_gate_item_sha256=HASH,
            dependency_vector=vector,
            claim_ids=(),
            public_object_ids=(),
            private_object_ids=(),
            publication_policy_identity=context.publication_policy_identity,
        )
        with pytest.raises(ArtifactPublicationError, match="prior authorization"):
            publish_authorized_artifact(
                staged,
                forged,
                reconstruction_inputs=ReconstructionInputs(
                    author_result=result,
                    proposal=None,
                    handoff_execution=None,
                    source_manifest=source_manifest,
                    accepted_gate_item={},
                ),
                context=context,
                mirrors=mirrors,
                ledger=ledger,
                canonical_root=canonical,
                repository_root=repo,
            )
    assert tuple(mirrors.iter_objects(MirrorClass.PUBLIC)) == ()


def test_two_models_share_one_object_but_retain_independent_claims(tmp_path: Path) -> None:
    """Shared bytes deduplicate physically without source-claim collision."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one", "m_two")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        first, _published_first = _commit_private(
            "m_one",
            content=b"same source object",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
        second, _published_second = _commit_private(
            "m_two",
            content=b"same source object",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-two",
        )
        assert first.objects[0].object_id == second.objects[0].object_id
        assert first.custody_claims[0].claim_id != second.custody_claims[0].claim_id
        assert len(tuple(mirrors.iter_objects(MirrorClass.PRIVATE))) == 1
        projection = validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )
        assert len(projection.objects) == 1
        assert len(projection.claims) == 2
        assert len(projection.transactions) == 2
        assert {claim.stable_id for claim in projection.claims} == {"m_one", "m_two"}


def test_final_transaction_rehydrates_from_private_mirror_only(tmp_path: Path) -> None:
    """A verified final transaction restores exact bytes into disposable staging."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        staged, _published = _commit_private(
            "m_one",
            content=b"rehydrated source bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    transaction = resolve_final_artifact_transaction(
        projection,
        stable_id="m_one",
        work_id="work-m_one",
        transaction_id=staged.transaction_id,
    )
    assert transaction is not None

    rehydrated = rehydrate_artifact_transaction(
        transaction,
        mirrors=mirrors,
        staging_root=tmp_path / "disposable",
    )

    assert rehydrated.transaction == transaction
    assert tuple(rehydrated.claim_paths.values())[0].read_bytes() == b"rehydrated source bytes"
    assert rehydrated.root.is_relative_to(tmp_path / "disposable")


def test_rehydrator_rejects_corrupt_private_custody(tmp_path: Path) -> None:
    """Canonical reconstruction never falls through after private bytes mutate."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        staged, _published = _commit_private(
            "m_one",
            content=b"custody bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    transaction = resolve_final_artifact_transaction(
        projection,
        stable_id="m_one",
        work_id="work-m_one",
        transaction_id=staged.transaction_id,
    )
    assert transaction is not None
    private_path = mirrors.address(transaction.objects[0].content_sha256, MirrorClass.PRIVATE)
    private_path.write_bytes(b"corrupt")

    with pytest.raises(ArtifactRehydrationError, match="private custody is unavailable"):
        rehydrate_artifact_transaction(
            transaction,
            mirrors=mirrors,
            staging_root=tmp_path / "disposable",
        )


def test_unrecorded_final_transaction_selection_rejects_ambiguity(tmp_path: Path) -> None:
    """Two final candidates for one active work never gain latest-row authority."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _commit_private(
            "m_one",
            content=b"first authority",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
        _commit_private(
            "m_one",
            content=b"second authority",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-two",
        )
    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )

    with pytest.raises(ArtifactRehydrationError, match="multiple final"):
        resolve_final_artifact_transaction(
            projection,
            stable_id="m_one",
            work_id="work-m_one",
        )


def test_checkpoint_rejects_reconstruction_rewrite_and_mirror_orphan(tmp_path: Path) -> None:
    """Independent ledger anchors catch coherent-file rewrites and unknown objects.

    Refusal blast radius follows the rung-12 ruling: a rewritten
    reconstruction is ONE transaction's defect, so it quarantines that
    transaction (no reuse, loud stale entry) while the projection survives for
    every other model; a physical mirror orphan is corruption of the SHARED
    store and still fails the whole projection.
    """

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _staged, published = _commit_private(
            "m_one",
            content=b"checkpoint object",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    original = published.reconstruction_path.read_bytes()
    published.reconstruction_path.write_bytes(original + b" ")
    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    assert not [key for key in projection.transactions if key[0] == "m_one"]
    rewritten = [stale for key, stale in projection.stale_transactions.items() if key[0] == "m_one"]
    assert len(rewritten) == 1
    assert "reconstruction bytes changed" in rewritten[0].reason
    published.reconstruction_path.write_bytes(original)
    # An unattested file at a VALID content address is indistinguishable from
    # shared-store corruption and stays a whole-projection refusal. (A
    # non-addressable-shaped name is IC-3 typed store litter instead -- see
    # test_mirror_store_litter.py.)
    orphan = mirrors.root(MirrorClass.PRIVATE) / "sha256" / "dd" / ("dd" + "d" * 62)
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"orphan")
    with pytest.raises(ArtifactCheckpointError, match="orphan"):
        validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )


def test_candidate_derivation_refuses_unanchored_reconstruction_document(
    tmp_path: Path,
) -> None:
    """A current-schema reconstruction no artifact event names can never be signed.

    ``_validate_artifact_reconstruction_append_only`` asserts only that every
    ledger-named reconstruction is present in the checkpoint candidate set. The
    converse -- that every reconstruction document IN the candidate set is
    ledger-named -- has to hold here, or the ``reconstruction`` allowlist root
    sweeps an orphaned or fabricated document into the signed checkpoint with
    nothing binding it to canonical authority.
    """

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_resnet50", "m_vgg16")
    mirrors = _mirrors(tmp_path)
    published = {}
    with ArtifactEventLedger(ledger_path) as ledger:
        for stable_id, content, evidence_id in (
            ("m_resnet50", b"resnet50 object", "ev-resnet50"),
            ("m_vgg16", b"vgg16 object", "ev-vgg16"),
        ):
            _staged, published[stable_id] = _commit_private(
                stable_id,
                content=content,
                context=context,
                mirrors=mirrors,
                ledger=ledger,
                canonical_root=canonical,
                repository_root=repo,
                evidence_id=evidence_id,
            )

    # Non-vacuity: two distinct genuinely ledger-named documents are ADMITTED, so
    # the refusal below discriminates on the anchor rather than refusing wholesale.
    anchored = _derive_candidate_paths(repo, canonical, (ledger_path,))
    expected_anchored = {
        published[stable_id].reconstruction_path.relative_to(repo)
        for stable_id in ("m_resnet50", "m_vgg16")
    }
    assert len(expected_anchored) == 2
    assert expected_anchored <= set(anchored)

    # A well-formed CURRENT-schema document that no artifact event names. Copying
    # real published bytes keeps every other field internally coherent, so only the
    # missing ledger anchor can be what refuses it.
    forged = dict(
        json.loads(published["m_resnet50"].reconstruction_path.read_text(encoding="utf-8"))
    )
    assert forged["schema_version"] == ARTIFACT_RECONSTRUCTION_SCHEMA_VERSION
    forged["stable_id"] = "m_fabricated"
    forged["transaction_id"] = "txn-fabricated"
    unanchored = canonical / "reconstruction" / "zz" / "m_fabricated" / "txn-fabricated.json"
    unanchored.parent.mkdir(parents=True, exist_ok=True)
    unanchored.write_text(json.dumps(forged, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(CheckpointValidationError) as refusal:
        _derive_candidate_paths(repo, canonical, (ledger_path,))
    assert str(refusal.value) == (
        "reconstruction document is not named by the artifact ledger: "
        "menagerie/crawler/reconstruction/zz/m_fabricated/txn-fabricated.json"
    )

    # An EMPTY ledger names nothing, so a real document is unanchored too. The
    # per-shard artifact validation runs only when artifact events exist, which is
    # exactly the window this sweep has to cover on its own.
    unanchored.unlink()
    with pytest.raises(CheckpointValidationError) as unledgered:
        _derive_candidate_paths(repo, canonical, ())
    assert str(unledgered.value).startswith(
        "reconstruction document is not named by the artifact ledger: "
    )


def test_checkpoint_accepts_history_recorded_under_superseded_rule_identities(
    tmp_path: Path,
) -> None:
    """Standing final transactions rehydrate at the rule identity they recorded.

    Rung-11 leg-3 abort regression: ``terminal_policy_identity`` (and its
    siblings) are content hashes of shipped policy bytes, so they legitimately
    move between legs. The whole-ledger checkpoint scan used to demand every
    RECORDED vector equal the LIVE context on those axes, which turned one
    ``authority.py`` edit into a permanent campaign-killer -- the append-only
    ledger retained pre-upgrade finals that could never validate again. A
    recorded terminal adjudication is append-only history: it validates against
    its own recorded commitments, while re-adjudication under current rules is
    the reducer's model-scoped currency decision (``validate_currency`` flags
    the standing model record -- pinned in ``test_authority.py``).
    """

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _commit_private(
            "m_one",
            content=b"historical authority",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    upgraded = replace(
        context,
        terminal_policy_identity="sha256:" + "1" * 64,
        checker_prompt_identity="sha256:" + "2" * 64,
        publication_policy_identity="sha256:" + "3" * 64,
        author_schema_identity="sha256:" + "4" * 64,
    )

    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=upgraded,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )

    transaction = resolve_final_artifact_transaction(
        projection,
        stable_id="m_one",
        work_id="work-m_one",
    )
    assert transaction is not None
    assert transaction.stable_id == "m_one"


def test_checkpoint_refuses_vector_contradicting_its_own_record(tmp_path: Path) -> None:
    """Every record-anchored axis and every recorded pin still refuses corruption.

    The historical-acceptance change above never weakens the tripwire: a vector
    that contradicts the transaction's OWN recorded author result, gate, claims,
    sources, intake identity, or transaction id refuses, and a missing recorded
    rule pin refuses as malformed. (Byte-level tampering refuses earlier, at the
    ledger's reconstruction anchor -- pinned by
    ``test_checkpoint_rejects_reconstruction_rewrite_and_mirror_orphan``.)
    """

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _staged, published = _commit_private(
            "m_one",
            content=b"corruption fixture",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    document = json.loads(published.reconstruction_path.read_text(encoding="utf-8"))
    final_event = published.event

    # Non-vacuity: the genuine document validates against its own record.
    _validate_projection_dependencies(document, final_event)

    record_anchored_faults = {
        "author_result_identity": "result-forged",
        "checker_gate_identity": "gate-forged",
        "artifact_transaction_id": "txn-forged",
        "artifact_claim_ids": ["claim-forged"],
        "source_manifest_identity": "sha256:" + "f" * 64,
        "intake_snapshot_id": "intake-forged",
        "intake_item_sha256": "sha256:" + "f" * 64,
    }
    for axis, forged_value in record_anchored_faults.items():
        corrupted = dict(document)
        corrupted["dependency_vector"] = dict(document["dependency_vector"])
        corrupted["dependency_vector"][axis] = forged_value
        with pytest.raises(ArtifactCheckpointError, match=f"contradicts its record at {axis}"):
            _validate_projection_dependencies(corrupted, final_event)

    for pin in (
        "author_result_schema_identity",
        "author_dispatcher_identity",
        "author_prompt_identity",
        "checker_prompt_identity",
        "terminal_rule_identity",
        "publication_policy_identity",
    ):
        stripped = dict(document)
        stripped["dependency_vector"] = dict(document["dependency_vector"])
        del stripped["dependency_vector"][pin]
        with pytest.raises(ArtifactCheckpointError, match=f"lacks its recorded pin at {pin}"):
            _validate_projection_dependencies(stripped, final_event)
        blanked = dict(document)
        blanked["dependency_vector"] = dict(document["dependency_vector"])
        blanked["dependency_vector"][pin] = ""
        with pytest.raises(ArtifactCheckpointError, match=f"lacks its recorded pin at {pin}"):
            _validate_projection_dependencies(blanked, final_event)


def test_checkpoint_refuses_rewritten_vector_through_ledger_anchor(tmp_path: Path) -> None:
    """A coherent on-disk rewrite of a recorded rule pin refuses at the byte anchor.

    The record-time pins carry no independent in-record anchor, so their
    authenticity IS the ledger's dual hash commitment: the reconstruction
    anchor sha256 and the final event's ``dependency_vector_sha256``. Rewriting
    the committed document with a different ``terminal_rule_identity`` -- the
    exact "record whose content does not match its recorded vector" fault --
    must refuse even though the JSON stays well-formed and self-consistent.
    Refusal is per-transaction under the rung-12 ruling: the rewritten entry
    quarantines (no reuse, loud) and the projection survives.
    """

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _staged, published = _commit_private(
            "m_one",
            content=b"anchored fixture",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    document = json.loads(published.reconstruction_path.read_text(encoding="utf-8"))
    document["dependency_vector"] = dict(document["dependency_vector"])
    document["dependency_vector"]["terminal_rule_identity"] = "sha256:" + "d" * 64
    published.reconstruction_path.write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )

    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    assert not [key for key in projection.transactions if key[0] == "m_one"]
    quarantined = [
        stale for key, stale in projection.stale_transactions.items() if key[0] == "m_one"
    ]
    assert len(quarantined) == 1
    assert "reconstruction bytes changed" in quarantined[0].reason


def test_checkpoint_accepts_history_after_author_contract_and_intake_movement(
    tmp_path: Path,
) -> None:
    """Finalized transactions rehydrate after prompt/dispatcher/intake movement.

    Rung-11 leg-4 family regression, second and third documented sites: the
    reconstruction document validated its intake snapshot, intake item, and
    family binding against the LIVE context, and ``_validate_context_result``
    re-imposed the LIVE author prompt/dispatcher/model identities on the
    recorded author result at READ time -- one author-prompt re-pin (a routine
    event) refused every finalized transaction forever. Both now validate at
    the transaction's OWN recorded anchors: the final event's intake identity,
    the anchor-committed document, and the recorded dependency-vector pins.
    """

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _commit_private(
            "m_one",
            content=b"prompt re-pin fixture",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    upgraded = replace(
        context,
        author_prompt_identity="sha256:" + "5" * 64,
        author_dispatcher_identity="sha256:" + "6" * 64,
        author_model_identity="sha256:" + "7" * 64,
        active_intake_snapshot_id="intake-refreshed",
        active_intake_snapshot_sha256="sha256:" + "8" * 64,
        intake_by_stable_id={
            "m_one": {"stable_id": "m_one", "natural_key": "fixture:m_one", "edited": True}
        },
        family_bindings={
            "m_one": {
                "binding_state": "variant",
                "representative_stable_id": "m_rep",
                "variant_token": "small",
                "derivation_rule_identity": HASH,
            }
        },
    )

    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=upgraded,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )

    transaction = resolve_final_artifact_transaction(
        projection,
        stable_id="m_one",
        work_id="work-m_one",
    )
    assert transaction is not None
    assert transaction.reconstruction_inputs.author_result["prompt_identity"] == (
        context.author_prompt_identity
    )


def test_reconstruction_document_contradicting_its_record_refuses(
    tmp_path: Path,
) -> None:
    """Record-anchored intake/result cross-checks keep full strength.

    The recorded-basis conversion never weakens the tripwire: a document
    whose intake item no longer hashes to its recorded identity, whose intake
    snapshot disagrees with the final event, whose dependency vector breaks
    the ledger commitment, or whose author result contradicts the recorded
    prompt pin still refuses.
    """

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _staged, published = _commit_private(
            "m_one",
            content=b"recorded-basis corruption fixture",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    document = json.loads(published.reconstruction_path.read_text(encoding="utf-8"))
    final_event = dict(published.event)
    authorization_event = {"artifact_event_id": document["authorization_event_id"]}

    # Non-vacuity: the genuine document validates against its own record.
    _validate_reconstruction_document(document, final_event, authorization_event)

    mutated_item = dict(document)
    mutated_item["intake_item"] = {"stable_id": "m_one", "natural_key": "forged"}
    with pytest.raises(ArtifactCheckpointError, match="authority/intake/family binding changed"):
        _validate_reconstruction_document(mutated_item, final_event, authorization_event)

    moved_snapshot = dict(document)
    moved_snapshot["intake_snapshot_id"] = "intake-forged"
    with pytest.raises(ArtifactCheckpointError, match="authority/intake/family binding changed"):
        _validate_reconstruction_document(moved_snapshot, final_event, authorization_event)

    broken_vector = dict(document)
    broken_vector["dependency_vector"] = dict(document["dependency_vector"])
    broken_vector["dependency_vector"]["author_prompt_identity"] = "sha256:" + "f" * 64
    with pytest.raises(ArtifactCheckpointError, match="does not match its ledger commitment"):
        _validate_reconstruction_document(broken_vector, final_event, authorization_event)

    stripped_pin = dict(document)
    stripped_pin["dependency_vector"] = dict(document["dependency_vector"])
    del stripped_pin["dependency_vector"]["author_prompt_identity"]
    stripped_event = dict(final_event)
    stripped_event["dependency_vector_sha256"] = stable_hash(stripped_pin["dependency_vector"])
    with pytest.raises(
        ArtifactCheckpointError, match="lacks its recorded pin at author_prompt_identity"
    ):
        _validate_reconstruction_document(stripped_pin, stripped_event, authorization_event)


def test_write_time_staging_still_binds_the_live_context(tmp_path: Path) -> None:
    """The recorded-basis conversion leaves WRITE-time live checks untouched.

    Staging a result authored under a superseded prompt against the upgraded
    live context must refuse: a result being committed NOW binds the world as
    it is now. Only READ-time revalidation of already-finalized history moved
    to the recorded basis.
    """

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    context = _context("m_one")
    source_manifest = _source_manifest(b"write-time fixture")
    result = _blocked_result("m_one", context, source_manifest)
    upgraded = replace(context, author_prompt_identity="sha256:" + "5" * 64)
    digest = hash_bytes(b"write-time fixture")
    with ArtifactEventLedger(canonical / "records" / "artifacts" / "shard.jsonl") as ledger:
        with pytest.raises(ArtifactBindingError, match="stale against active authority context"):
            stage_private_artifact(
                (
                    ArtifactInput(
                        content=b"write-time fixture",
                        content_sha256=digest,
                        logical_role="source",
                        logical_path=f"menagerie/crawler/source_cas/{digest[7:]}.source",
                        source_id="src-upstream",
                        origin=ArtifactOrigin("https://example.test/source.tar", "v1"),
                        fetch_recipe="GET exact source.tar@v1",
                        evidence_ids=("ev-one",),
                        media_type="application/x-tar",
                    ),
                ),
                context=upgraded,
                stable_id="m_one",
                work_id="work-m_one",
                author_result=result,
                proposal=None,
                source_manifest=source_manifest,
                mirrors=_mirrors(tmp_path),
                ledger=ledger,
                created_at=NOW,
            )


def test_authored_publication_paths_are_generation_scoped() -> None:
    """The fourth step-100 crash shape (2026-08-14): m8635's re-drive re-vendored
    its adapter closure with DIFFERENT bytes, and the generation-unscoped
    publication path reported ``immutable artifact path changed``. A new work
    generation is a new publication -- a new path alongside its predecessor --
    and one generation still mints exactly one path."""

    relative = "recbole/utils/__init__.py"
    first = authored_code_logical_path(
        logical_role="code",
        stable_id="m8635",
        work_id="work-m8635",
        relative=relative,
    )
    redrive_work_id = "sha256:" + "0f3c1d" + "a" * 58
    second = authored_code_logical_path(
        logical_role="code",
        stable_id="m8635",
        work_id=redrive_work_id,
        relative=relative,
    )
    assert first != second
    # Replay of the SAME generation mints the SAME path: that is where the
    # changed-bytes tripwire lives, and it stays armed.
    assert first == authored_code_logical_path(
        logical_role="code",
        stable_id="m8635",
        work_id="work-m8635",
        relative=relative,
    )
    for value in (first, second):
        assert value.startswith("menagerie/crawler/adapters/m8/m8635/")
        # The declared model-relative path stays the exact suffix, so staging's
        # proposal bind and rehydration keep matching claims by declared parts.
        assert value.endswith(f"/{relative}")
        assert PurePosixPath(value).parts[5].startswith("gen-")
        # A digest-shaped work generation never leaks its separator into a path.
        assert ":" not in value
    patch = authored_code_logical_path(
        logical_role="patch",
        stable_id="m8635",
        work_id=redrive_work_id,
        relative="fix.diff",
    )
    generation = PurePosixPath(second).parts[5]
    assert patch == f"menagerie/crawler/patches/m8/m8635/{generation}/fix.diff"
    with pytest.raises(ArtifactBindingError):
        authored_code_logical_path(
            logical_role="source",
            stable_id="m8635",
            work_id="work-m8635",
            relative=relative,
        )
    with pytest.raises(ArtifactBindingError):
        authored_code_logical_path(
            logical_role="code",
            stable_id="m8635",
            work_id="   ",
            relative=relative,
        )


def test_immutable_publication_refuses_changed_bytes_within_one_generation(
    tmp_path: Path,
) -> None:
    """Generation scoping must not weaken the tripwire: republishing DIFFERENT
    bytes at one generation's own path is still a hard publication failure, and
    an identical replay is still idempotent."""

    logical = authored_code_logical_path(
        logical_role="code",
        stable_id="m8635",
        work_id="work-m8635",
        relative="recbole/utils/__init__.py",
    )
    path = tmp_path / logical
    _atomic_write_immutable(path, b"vendored bytes\n")
    _atomic_write_immutable(path, b"vendored bytes\n")
    assert path.read_bytes() == b"vendored bytes\n"
    with pytest.raises(ArtifactPublicationError, match="immutable artifact path changed"):
        _atomic_write_immutable(path, b"re-vendored bytes\n")
    # The re-driven generation publishes ALONGSIDE it, never over it.
    redrive = tmp_path / authored_code_logical_path(
        logical_role="code",
        stable_id="m8635",
        work_id="sha256:" + "0f3c1d" + "a" * 58,
        relative="recbole/utils/__init__.py",
    )
    _atomic_write_immutable(redrive, b"re-vendored bytes\n")
    assert path.read_bytes() == b"vendored bytes\n"
    assert redrive.read_bytes() == b"re-vendored bytes\n"


def test_authored_publication_path_has_one_mint_site() -> None:
    """The driver must not rebuild the publication path inline: a second mint
    site is how the generation axis got dropped in the first place."""

    source = (Path(__file__).resolve().parents[1] / "driver.py").read_text(encoding="utf-8")
    assert "authored_code_logical_path(" in source
    assert "crawler/adapters" not in source
    assert "crawler/patches" not in source


def test_declared_media_type_divergence_is_not_intrinsic_ambiguity() -> None:
    """The leg-38 admission crash: two eras minted the SAME bytes under one
    physical key with different DECLARED media types (m5445 text/plain
    2026-08-07 vs m13746 text/markdown 2026-08-20), and full-row equality read
    the declaration divergence as store corruption, leg-fatally. Intrinsic
    physical identity is what the content-addressed store proves: class,
    digest, size, key. Divergent BYTES must still refuse."""

    from menagerie.crawler.artifact_transactions import _intrinsic_physical_inventory
    from menagerie.crawler.authority import MirrorObject

    plain = MirrorObject(
        object_id="sha256:" + "1" * 64,
        mirror_class="private",
        content_sha256="sha256:" + "c" * 64,
        byte_count=11357,
        media_type="text/plain",
        object_key="sha256/cc/" + "c" * 64,
    )
    markdown = MirrorObject(
        object_id="sha256:" + "2" * 64,
        mirror_class="private",
        content_sha256="sha256:" + "c" * 64,
        byte_count=11357,
        media_type="text/markdown",
        object_key="sha256/cc/" + "c" * 64,
    )
    assert plain != markdown
    assert _intrinsic_physical_inventory(plain) == _intrinsic_physical_inventory(markdown)
    truncated = MirrorObject(
        object_id="sha256:" + "3" * 64,
        mirror_class="private",
        content_sha256="sha256:" + "d" * 64,
        byte_count=11356,
        media_type="text/plain",
        object_key="sha256/cc/" + "c" * 64,
    )
    assert _intrinsic_physical_inventory(plain) != _intrinsic_physical_inventory(truncated)


# The leg-39 stale-physical tolerance regression moved to
# test_stale_physical_inventory.py: the original test here constructed three
# local sets and performed the subtraction inline (deleting the production
# harvest would not have failed it -- adjudicated vacuous, trilab synthesis
# D2). The replacement exercises validate_artifact_checkpoint over a real
# temp ledger/CAS with tolerated/foreign/missing arms and a mutation check.
