"""AT-7: a torn authorized chain completes the SAME transaction, zero authors.

The m8635 / leg-39 shape: the process died between the committed
publication-authorization event and the physical publication, leaving a
2-event chain whose capability is durable. Re-authoring such a member re-buys
a full agentic session for bytes the ledger already committed to publish.
:func:`resume_torn_publication` replays ``publish_authorized_artifact`` from
the committed capability -- rebuilt entirely from the authorization event,
with every caller-supplied input re-verified against it (gate item sha,
dependency-vector sha, and the joint (vector, gate, decisions, policy)
re-derivation of the recorded authorization id). TW-TORN
(:func:`torn_chain_census`) classifies every incomplete chain read-only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.crawler.artifact_transactions import (
    ArtifactBindingError,
    ArtifactEventKind,
    ArtifactEventLedger,
    ArtifactTransitionError,
    derive_artifact_claims,
    derive_publication_authorization_id,
    resume_torn_publication,
    stage_private_artifact,
    torn_chain_census,
)
from menagerie.crawler.artifact_transactions import ArtifactInput
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.licenses import LicenseDecision, RedistributionClass
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorClass
from menagerie.crawler.tests.test_artifact_transactions import (
    NOW,
    _context,
    _mirrors,
    _proposed_result,
    _vector,
)
from menagerie.crawler.artifact_transactions import (
    PublicationAuthorization,
    append_artifact_authorization,
)

pytestmark = pytest.mark.smoke


def _torn_chain(tmp_path: Path, stable_id: str) -> dict:
    """Drive one m8635-shaped chain to a committed authorization, then stop."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context(stable_id)
    mirrors = _mirrors(tmp_path)
    content = b"m8635-shaped public source"
    result, proposal, source_manifest = _proposed_result(stable_id, context, content)
    digest = hash_bytes(content)
    ledger = ArtifactEventLedger(ledger_path)
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
                evidence_ids=("ev-license",),
                media_type="application/x-tar",
            ),
        ),
        context=context,
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        author_result=result,
        proposal=proposal,
        source_manifest=source_manifest,
        mirrors=mirrors,
        ledger=ledger,
        created_at=NOW,
    )
    gate_item = {
        "stable_id": stable_id,
        "work_id": f"work-{stable_id}",
        "verdict": "accurate",
        "integrity": {"verdict": "accurate"},
        "rung_check": {"verdict": "accurate"},
        "verified_hashes": {"source_manifest": source_manifest["manifest_sha256"]},
    }
    provisional_vector = _vector(
        context,
        stable_id,
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
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
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
    # The crash: the process dies here; publication never ran.
    return {
        "ledger": ledger,
        "mirrors": mirrors,
        "canonical": canonical,
        "repo": repo,
        "context": context,
        "staged": staged,
        "gate_item": gate_item,
        "vector": vector,
        "decisions": decisions,
        "result": result,
        "proposal": proposal,
        "source_manifest": source_manifest,
    }


def _resume(fixture: dict, **overrides) -> object:
    arguments = dict(
        ledger=fixture["ledger"],
        transaction_id=fixture["staged"].transaction_id,
        dependency_vector=fixture["vector"],
        accepted_gate_item=fixture["gate_item"],
        decisions=fixture["decisions"],
        author_result=fixture["result"],
        proposal=fixture["proposal"],
        handoff_execution=None,
        source_manifest=fixture["source_manifest"],
        context=fixture["context"],
        mirrors=fixture["mirrors"],
        canonical_root=fixture["canonical"],
        repository_root=fixture["repo"],
        created_at=NOW,
    )
    arguments.update(overrides)
    return resume_torn_publication(**arguments)


def test_torn_chain_resumes_the_same_transaction_with_zero_author_dispatch(tmp_path):
    fixture = _torn_chain(tmp_path, "m_torn")
    ledger = fixture["ledger"]
    with ledger:
        census = torn_chain_census(ledger)
        assert [row["classification"] for row in census] == ["resumable"]

        _resume(fixture)

        events = list(ledger.events)
        # SAME transaction completed -- no fresh authoring, no new chain.
        assert {str(event["transaction_id"]) for event in events} == {
            str(fixture["staged"].transaction_id)
        }
        assert [event["event_kind"] for event in events] == [
            "staged-private",
            "publication-authorized",
            "reconstruction-committed",
            "published",
        ]
        assert torn_chain_census(ledger) == ()
    # The physical public byte landed.
    public_root = fixture["mirrors"].root(MirrorClass.PUBLIC)
    assert any(path.is_file() for path in public_root.rglob("*"))


def test_resume_refuses_inputs_that_differ_from_the_committed_capability(tmp_path):
    fixture = _torn_chain(tmp_path, "m_torn_refuse")
    with fixture["ledger"]:
        forged_gate = dict(fixture["gate_item"])
        forged_gate["verdict"] = "inaccurate"
        with pytest.raises(ArtifactBindingError, match="gate item"):
            _resume(fixture, accepted_gate_item=forged_gate)

        forged_decisions = {
            claim_id: replace(decision, rationale="forged rationale")
            for claim_id, decision in fixture["decisions"].items()
        }
        with pytest.raises(ArtifactBindingError, match="authorization id"):
            _resume(fixture, decisions=forged_decisions)

        # Nothing above appended anything: the chain is still torn.
        assert [row["classification"] for row in torn_chain_census(fixture["ledger"])] == [
            "resumable"
        ]


def test_resume_refuses_chains_that_are_not_torn(tmp_path):
    fixture = _torn_chain(tmp_path, "m_torn_full")
    with fixture["ledger"]:
        _resume(fixture)
        with pytest.raises(ArtifactTransitionError, match="not a torn"):
            _resume(fixture)


def test_tw_spent_join_classifies_orphaned_and_paired_finals(tmp_path):
    """TW-SPENT: a final event with no naming revision is an orphaned spend;
    one naming revision pairs; two naming revisions disclose multiply-named."""

    from menagerie.crawler.artifact_transactions import tw_spent_ledger_join

    fixture = _torn_chain(tmp_path, "m_spent")
    with fixture["ledger"] as ledger:
        _resume(fixture)
        events = list(ledger.events)
    final = [event for event in events if event["event_kind"] == "published"]
    assert len(final) == 1
    event_id = str(final[0]["artifact_event_id"])

    orphaned = tw_spent_ledger_join(events, [])
    assert [row["classification"] for row in orphaned] == ["orphaned-final"]

    naming = {
        "record_revision": "sha256:" + "1" * 64,
        "artifact_authority": {"committed_event_id": event_id},
    }
    paired = tw_spent_ledger_join(events, [naming])
    assert [row["classification"] for row in paired] == ["paired"]
    assert paired[0]["naming_revisions"] == ["sha256:" + "1" * 64]

    second = {
        "record_revision": "sha256:" + "2" * 64,
        "artifact_authority": {"committed_event_id": event_id},
    }
    doubled = tw_spent_ledger_join(events, [naming, second])
    assert [row["classification"] for row in doubled] == ["multiply-named"]


def test_census_classifies_pending_custody_separately(tmp_path):
    """A staged-only chain is durable pending custody, never a torn chain."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_pending")
    mirrors = _mirrors(tmp_path)
    content = b"pending custody bytes"
    result, proposal, source_manifest = _proposed_result("m_pending", context, content)
    with ArtifactEventLedger(ledger_path) as ledger:
        stage_private_artifact(
            (
                ArtifactInput(
                    content=content,
                    content_sha256=hash_bytes(content),
                    logical_role="source",
                    logical_path=(
                        f"menagerie/crawler/source_cas/{hash_bytes(content)[7:]}.source"
                    ),
                    source_id="src-upstream",
                    origin=ArtifactOrigin("https://example.test/source.tar", "v1"),
                    fetch_recipe="GET exact source.tar@v1",
                    evidence_ids=("ev-license",),
                    media_type="application/x-tar",
                ),
            ),
            context=context,
            stable_id="m_pending",
            work_id="work-m_pending",
            author_result=result,
            proposal=proposal,
            source_manifest=source_manifest,
            mirrors=mirrors,
            ledger=ledger,
            created_at=NOW,
        )
        assert [row["classification"] for row in torn_chain_census(ledger)] == [
            "pending-custody"
        ]
