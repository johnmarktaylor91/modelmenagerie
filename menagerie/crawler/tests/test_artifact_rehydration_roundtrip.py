"""AT-1: rehydration reads citable rows (TW-ROUNDTRIP).

The write side stages byte custody over ``manifest_source_rows(...)`` -- the
frozen rows PLUS the one supplementary broker pack -- and refuses staging
without exact byte coverage of every byte-backed citable row. The read side
(``_rehydration_targets``) resolved claims against the FROZEN rows alone, so
every finalized transaction carrying a supplementary source was structurally
unrehydratable ("rehydration claim lacks its exact object or source"): 7 of
200a's 218 finalized transactions, including 4 of the census's 10 runs.

TW-ROUNDTRIP pins the property the write side already guarantees: staged
claims == rehydratable claims, for 0, 1, and N supplementary rows, through
the PRODUCTION stage -> authorize -> commit -> checkpoint -> resolve ->
rehydrate path. The write-time claims-subset invariant (staging a byte whose
source_id is outside the citable set refuses) is pinned beside it so the two
sides can never diverge silently again.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.artifact_transactions import (
    ArtifactBindingError,
    ArtifactRehydrationError,
    ArtifactEventKind,
    ArtifactEventLedger,
    ArtifactInput,
    ReconstructionInputs,
    append_artifact_authorization,
    derive_artifact_claims,
    derive_publication_authorization_id,
    publish_authorized_artifact,
    rehydrate_artifact_transaction,
    resolve_final_artifact_transaction,
    stage_private_artifact,
    unrehydratable_authority_census,
    validate_artifact_checkpoint,
)
from menagerie.crawler.authority import PublicationAuthorization
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.licenses import LicenseDecision, RedistributionClass
from menagerie.crawler.mirrors import ArtifactOrigin
from menagerie.crawler.models import SUPPLEMENTARY_SOURCES_KEY
from menagerie.crawler.tests.test_artifact_transactions import (
    NOW,
    _blocked_result,
    _context,
    _gate_item,
    _mirrors,
    _vector,
)

pytestmark = pytest.mark.smoke


def _source_row(source_id: str, content: bytes) -> dict[str, Any]:
    """Build one digest-pinned fetched source row."""

    return {
        "source_id": source_id,
        "url": f"https://example.test/{source_id}.tar",
        "revision": "v1",
        "content_sha256": hash_bytes(content),
        "fetched_bytes_len": len(content),
        "media_type": "application/x-tar",
    }


def _artifact_input(source_id: str, content: bytes) -> ArtifactInput:
    """Build the staged byte custody matching one fetched source row."""

    digest = hash_bytes(content)
    return ArtifactInput(
        content=content,
        content_sha256=digest,
        logical_role="source",
        logical_path=f"menagerie/crawler/source_cas/{digest[7:]}.source",
        source_id=source_id,
        origin=ArtifactOrigin(f"https://example.test/{source_id}.tar", "v1"),
        fetch_recipe=f"GET exact {source_id}.tar@v1",
        evidence_ids=(f"ev-{source_id}",),
        media_type="application/x-tar",
    )


def _commit_with_supplementary(
    tmp_path: Path, stable_id: str, supplementary_contents: dict[str, bytes]
) -> tuple[Any, Any, Any, Path, Path, dict[str, bytes]]:
    """Stage, authorize, and privately commit one transaction whose manifest
    carries the frozen row plus the given supplementary rows, exactly like a
    lane that consumed its one granted supplementary broker round."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context(stable_id)
    mirrors = _mirrors(tmp_path)
    frozen_content = b"frozen upstream bytes for " + stable_id.encode()
    contents = {"src-upstream": frozen_content, **supplementary_contents}
    frozen_row = _source_row("src-upstream", frozen_content)
    source_manifest: dict[str, Any] = {
        "sources": [frozen_row],
        "manifest_sha256": stable_hash([frozen_row]),
    }
    if supplementary_contents:
        source_manifest[SUPPLEMENTARY_SOURCES_KEY] = [
            _source_row(source_id, content)
            for source_id, content in supplementary_contents.items()
        ]
    result = _blocked_result(stable_id, context, source_manifest)
    with ArtifactEventLedger(ledger_path) as ledger:
        staged = stage_private_artifact(
            tuple(
                _artifact_input(source_id, content) for source_id, content in contents.items()
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
        gate_item["terminal_disposition"]["source_ids"] = sorted(contents)
        provisional_vector = _vector(
            context,
            stable_id,
            str(staged.transaction_id),
            result["result_id"],
            source_manifest["manifest_sha256"],
        )
        objects_by_id = {obj.object_id: obj for obj in staged.objects}
        decisions = {
            claim.claim_id: LicenseDecision(
                content_sha256=objects_by_id[claim.object_id].content_sha256,
                redistribution_class=RedistributionClass.RESTRICTED_PRIVATE,
                evidence_ids=claim.evidence_ids,
                rationale="restricted fixture",
            )
            for claim in staged.custody_claims
        }
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
            private_object_ids=tuple(sorted({claim.object_id for claim in claims}, key=str)),
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
        publish_authorized_artifact(
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
            canonical_root=canonical,
            repository_root=repo,
            created_at=NOW,
        )
    return staged, context, mirrors, ledger_path, canonical.parent.parent, contents


@pytest.mark.parametrize("n_supplementary", [0, 1, 3])
def test_roundtrip_staged_claims_all_rehydrate(tmp_path: Path, n_supplementary: int) -> None:
    """TW-ROUNDTRIP: every claim the write side staged -- frozen and
    supplementary alike -- rehydrates to its exact bytes on the read side."""

    stable_id = f"m_supp{n_supplementary}"
    supplementary = {
        f"src-supp-{index}": f"supplementary bytes {index}".encode()
        for index in range(n_supplementary)
    }
    staged, context, mirrors, ledger_path, repo, contents = _commit_with_supplementary(
        tmp_path, stable_id, supplementary
    )
    canonical = repo / "menagerie" / "crawler"
    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    transaction = resolve_final_artifact_transaction(
        projection,
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        transaction_id=staged.transaction_id,
    )
    assert transaction is not None
    assert len(transaction.claims) == 1 + n_supplementary

    rehydrated = rehydrate_artifact_transaction(
        transaction,
        mirrors=mirrors,
        staging_root=tmp_path / "disposable",
    )

    rehydrated_bytes = sorted(
        path.read_bytes() for path in rehydrated.claim_paths.values()
    )
    assert rehydrated_bytes == sorted(contents.values())


def test_write_time_staging_refuses_claim_outside_citable_set(tmp_path: Path) -> None:
    """The write-time claims-subset invariant: a staged byte citing a source_id
    outside the citable rows (frozen + supplementary) refuses at staging, so
    no persisted claim can ever exceed what rehydration may resolve."""

    stable_id = "m_foreign"
    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context(stable_id)
    mirrors = _mirrors(tmp_path)
    frozen_content = b"frozen bytes"
    frozen_row = _source_row("src-upstream", frozen_content)
    source_manifest = {
        "sources": [frozen_row],
        "manifest_sha256": stable_hash([frozen_row]),
    }
    result = _blocked_result(stable_id, context, source_manifest)
    with ArtifactEventLedger(ledger_path) as ledger:
        with pytest.raises(ArtifactBindingError, match="outside the frozen set"):
            stage_private_artifact(
                (
                    _artifact_input("src-upstream", frozen_content),
                    _artifact_input("src-foreign", b"uncovenanted bytes"),
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
        assert tuple(ledger.events) == ()


def test_duplicate_logical_path_same_bytes_rehydrates(tmp_path: Path) -> None:
    """The m1115 carried-award shape: two citable rows fetching IDENTICAL
    bytes stage one logical path twice -- lawful at the write side, so the
    read side must rehydrate it, not refuse it as a duplicate."""

    stable_id = "m_twin"
    staged, context, mirrors, ledger_path, repo, contents = _commit_with_supplementary(
        tmp_path, stable_id, {"src-supp-twin": b"frozen upstream bytes for " + stable_id.encode()}
    )
    canonical = repo / "menagerie" / "crawler"
    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    transaction = resolve_final_artifact_transaction(
        projection,
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        transaction_id=staged.transaction_id,
    )
    assert transaction is not None
    assert len(transaction.claims) == 2
    assert len({claim.logical_path for claim in transaction.claims}) == 1

    rehydrated = rehydrate_artifact_transaction(
        transaction,
        mirrors=mirrors,
        staging_root=tmp_path / "disposable",
    )
    for path in rehydrated.claim_paths.values():
        assert path.read_bytes() == contents["src-upstream"]


def test_duplicate_logical_path_conflicting_bytes_still_refuses(tmp_path: Path) -> None:
    """The armed direction: one logical path claimed with DIFFERENT bytes is
    corruption and keeps refusing typed -- the narrowing tolerates only the
    identical-replay shape the write side already accepts."""

    stable_id = "m_conflict"
    staged, context, mirrors, ledger_path, repo, _ = _commit_with_supplementary(
        tmp_path, stable_id, {"src-supp-0": b"different supplementary bytes"}
    )
    canonical = repo / "menagerie" / "crawler"
    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    key = (stable_id, f"work-{stable_id}", staged.transaction_id)
    transaction = projection.transactions[key]
    first_path = transaction.claims[0].logical_path
    forged_claims = (
        transaction.claims[0],
        replace(transaction.claims[1], logical_path=first_path),
    )
    forged = replace(transaction, claims=forged_claims)
    with pytest.raises(ArtifactRehydrationError, match="conflicting bytes"):
        rehydrate_artifact_transaction(
            forged,
            mirrors=mirrors,
            staging_root=tmp_path / "disposable",
        )


def test_unrehydratable_census_reports_both_arms(tmp_path: Path) -> None:
    """The census dry-runs the PRODUCTION rehydration predicate per finalized
    transaction: healthy authority reads ``rehydratable``; a transaction whose
    claims cite a source its manifest cannot resolve reads ``unrehydratable``
    with the exact refusal, and the census never mutates or upgrades."""

    stable_id = "m_census"
    staged, context, mirrors, ledger_path, repo, _ = _commit_with_supplementary(
        tmp_path, stable_id, {"src-supp-0": b"supplementary census bytes"}
    )
    canonical = repo / "menagerie" / "crawler"
    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    healthy = unrehydratable_authority_census(projection)
    assert [row["disposition"] for row in healthy] == ["rehydratable"]
    assert healthy[0]["stable_id"] == stable_id
    assert healthy[0]["transaction_id"] == str(staged.transaction_id)

    # The 200a poisoned shape: claims cite a supplementary source the read
    # side cannot resolve. Reproduced by narrowing the manifest the census
    # sees to its frozen rows alone.
    key = (stable_id, f"work-{stable_id}", staged.transaction_id)
    transaction = projection.transactions[key]
    narrowed_manifest = {
        field: value
        for field, value in transaction.reconstruction_inputs.source_manifest.items()
        if field != SUPPLEMENTARY_SOURCES_KEY
    }
    poisoned = replace(
        transaction,
        reconstruction_inputs=replace(
            transaction.reconstruction_inputs, source_manifest=narrowed_manifest
        ),
    )
    poisoned_projection = replace(projection, transactions={key: poisoned})
    rows = unrehydratable_authority_census(poisoned_projection)
    assert [row["disposition"] for row in rows] == ["unrehydratable"]
    assert "lacks its exact object or source" in rows[0]["reason"]
