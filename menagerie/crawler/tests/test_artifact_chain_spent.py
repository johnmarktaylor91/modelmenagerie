"""AT-2: the typed spent/closed-chain transition split.

The m12646 pileup (2026-08-23): the 12:42Z terminal-finalization publication
spent transaction sha256:3d711879... (staged-private -> publication-authorized
-> reconstruction-committed -> published), the paired model append refused 45
seconds later, and every subsequent award retry died as a GENERIC
``ArtifactTransitionError: transaction has too many transitions`` -- a refusal
indistinguishable from ledger corruption, so the catch sites consumed it as a
benign cache miss and the member looped for four hours.

The split this module pins: a NEW transition against a chain that already
carries its final event raises :class:`ArtifactChainSpentError` -- the ONLY
transition refusal a recovery path may treat as recoverable -- while every
corruption-shaped refusal (conflicting replay, forked predecessor, identity
mutation) stays the loud base :class:`ArtifactTransitionError` and is NEVER
the spent subclass. The retry must refuse BEFORE appending: the chain stays
exactly four events.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.artifact_transactions import (
    ArtifactChainSpentError,
    ArtifactEventKind,
    ArtifactEventLedger,
    ArtifactInput,
    ArtifactTransitionError,
    ReconstructionInputs,
    _event_id,
    append_artifact_authorization,
    derive_artifact_claims,
    derive_publication_authorization_id,
    publish_authorized_artifact,
    stage_private_artifact,
)
from menagerie.crawler.authority import PublicationAuthorization
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.licenses import LicenseDecision, RedistributionClass
from menagerie.crawler.mirrors import ArtifactOrigin, MirrorStore
from menagerie.crawler.tests.test_artifact_transactions import (
    NOW,
    _context,
    _mirrors,
    _proposed_result,
    _vector,
)

pytestmark = pytest.mark.smoke


def _publish_public_chain(
    tmp_path: Path, stable_id: str, *, stop_after_authorization: bool = False
) -> tuple[ArtifactEventLedger, MirrorStore, Path, Path]:
    """Drive one m12646-shaped chain to ``published`` and return the ledger.

    The exact production shape: staged-private -> publication-authorized ->
    reconstruction-committed -> published, four events, one transaction.
    With ``stop_after_authorization`` the chain is left torn at two events --
    the crashed-mid-publication (m8635/leg-39) shape.
    """

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context(stable_id)
    mirrors = _mirrors(tmp_path)
    content = b"m12646-shaped public source"
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
    if stop_after_authorization:
        return ledger, mirrors, canonical, repo
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
    return ledger, mirrors, canonical, repo


def _retry_transition(final_event: dict[str, Any], event_kind: str) -> dict[str, Any]:
    """Shape one retry transition chained onto an already-final event."""

    payload = {
        key: value
        for key, value in final_event.items()
        if key not in {"artifact_event_id", "created_at", "ledger_seq", "payload_sha256"}
    }
    payload["event_kind"] = event_kind
    payload["predecessor_event_id"] = final_event["artifact_event_id"]
    payload["artifact_event_id"] = _event_id(payload)
    return payload


def test_m12646_spent_chain_retry_is_typed_and_appends_nothing(tmp_path: Path) -> None:
    """A retry against a published 4-event chain raises the SPENT split, not
    the generic transition error, and persists no fifth event."""

    ledger, _, _, _ = _publish_public_chain(tmp_path, "m12646x")
    with ledger:
        chain = list(ledger.events)
        assert [event["event_kind"] for event in chain] == [
            "staged-private",
            "publication-authorized",
            "reconstruction-committed",
            "published",
        ]
        retry = _retry_transition(chain[-1], "published")
        with pytest.raises(ArtifactChainSpentError) as excinfo:
            ledger.append_event(retry)
        error = excinfo.value
        assert error.transaction_id == str(chain[-1]["transaction_id"])
        assert error.final_event_kind == "published"
        assert error.attempted_event_kind == "published"
        assert error.stable_id == "m12646x"
        # The retry never attempts event 5: the chain is untouched.
        assert len(tuple(ledger.events)) == 4


def test_spent_split_is_recoverable_subclass_of_transition_error() -> None:
    """Recovery routing keys on the subclass: it IS a transition error, so
    unrotated catch sites keep working until the AT-4/5/6 batch narrows them."""

    assert issubclass(ArtifactChainSpentError, ArtifactTransitionError)


def test_corruption_shaped_refusals_are_never_the_spent_split(tmp_path: Path) -> None:
    """Conflicting replay of an existing event id stays the LOUD base error --
    the split must not consume ledger-corruption classes as recoverable."""

    ledger, _, _, _ = _publish_public_chain(tmp_path, "m_loud")
    with ledger:
        chain = list(ledger.events)
        forged = dict(chain[1])
        forged["gate_id"] = "gate-forged"
        with pytest.raises(ArtifactTransitionError) as excinfo:
            ledger.append_event(forged)
        assert not isinstance(excinfo.value, ArtifactChainSpentError)
        assert "conflicting artifact event replay" in str(excinfo.value)


def test_exact_final_event_replay_stays_idempotent(tmp_path: Path) -> None:
    """Byte-exact replay of the final event is idempotent recognition, never a
    spent refusal: crash-recovery replays must keep completing for free."""

    ledger, _, _, _ = _publish_public_chain(tmp_path, "m_replay")
    with ledger:
        chain = list(ledger.events)
        result = ledger.append_event(dict(chain[-1]))
        assert result.appended is False
        assert len(tuple(ledger.events)) == 4
