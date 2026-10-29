"""IC-11: intrinsic-tuple membership census + cross-transaction collision arms.

17c48de4 keyed mirror-store ambiguity on the intrinsic physical four-tuple
(mirror_class, content_sha256, byte_count, object_key), tolerating divergence
in DECLARED classification (media_type -- the m5445/m13746 era divergence;
object_id hashes the declaration so it diverges with it). The residual all
three labs named: the tuple's membership was not census-pinned, so a future
declared axis added to MirrorObject would silently re-create the class.

Pins here: (1) the tuple membership census -- every MirrorObject field is
either intrinsic or explicitly declared-excluded, so a NEW field fails CI
until this census names it in a reviewed change; (2) cross-transaction
collision arms driven through the PRODUCTION checkpoint fold: declared-only
divergence on one physical key folds, intrinsic divergence refuses.

retention_class disposition (documented per the plan): retention_class is not
a MirrorObject field at all -- it lives on the ArtifactManifest/claim layer as
a DECLARED per-transaction retention policy, so it is outside intrinsic
physical identity by construction; were it ever added to MirrorObject, the
membership census below forces the reviewed intrinsic-vs-declared decision.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import pytest

from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointError,
    ArtifactEventLedger,
    _event_id,
    _intrinsic_physical_inventory,
    artifact_claim_id,
    mirror_object_id,
    validate_artifact_checkpoint,
)
from menagerie.crawler.authority import (
    MirrorObject,
    MirrorObjectId,
)
from menagerie.crawler.tests.test_artifact_transactions import (
    _commit_private,
    _context,
    _mirrors,
)

pytestmark = pytest.mark.smoke

INTRINSIC_FIELDS = ("mirror_class", "content_sha256", "byte_count", "object_key")
DECLARED_FIELDS = ("object_id", "media_type")


def test_intrinsic_tuple_membership_census() -> None:
    """Every MirrorObject field is either intrinsic or declared-excluded; a
    new field fails here until a reviewed change names its disposition."""

    assert {field.name for field in fields(MirrorObject)} == set(
        INTRINSIC_FIELDS + DECLARED_FIELDS
    ), (
        "MirrorObject grew or lost a field: decide intrinsic vs declared in "
        "the same reviewed change (a silently-declared axis re-creates the "
        "m5445/m13746 ambiguity class; a silently-intrinsic one re-creates "
        "leg 38's cross-era refusals)"
    )
    obj = MirrorObject(
        object_id=MirrorObjectId("obj-census"),
        mirror_class="private",
        content_sha256="sha256:" + "a" * 64,
        byte_count=7,
        media_type="text/plain",
        object_key="sha256/aa/" + "a" * 64,
    )
    assert _intrinsic_physical_inventory(obj) == (
        "private",
        "sha256:" + "a" * 64,
        7,
        "sha256/aa/" + "a" * 64,
    )


def _healthy_store(tmp_path: Path):
    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one", "m_forged")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _commit_private(
            "m_one",
            content=b"collision census bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
        staged_event = deepcopy(next(iter(ledger.events)))
    return ledger_path, context, mirrors, canonical, repo, staged_event


def _forged_staged_event(staged_event: dict, **object_overrides) -> dict:
    """Craft a second transaction's staged event naming the SAME physical key
    with the given object-row overrides, identity-consistent at every layer."""

    forged = deepcopy(staged_event)
    for field in ("ledger_seq", "payload_sha256"):
        forged.pop(field, None)
    forged["stable_id"] = "m_forged"
    forged["work_id"] = "work-m_forged"
    forged["transaction_id"] = "sha256:" + "f" * 64
    row = dict(forged["objects"][0])
    row.update(object_overrides)
    row["object_id"] = str(
        mirror_object_id(
            row["mirror_class"],
            row["content_sha256"],
            row["byte_count"],
            row["media_type"],
            row["object_key"],
        )
    )
    forged["objects"] = [row]
    claim = dict(forged["claims"][0])
    claim["object_id"] = row["object_id"]
    claim["stable_id"] = "m_forged"
    claim["work_id"] = "work-m_forged"
    claim["claim_id"] = str(
        artifact_claim_id(
            object_id=MirrorObjectId(row["object_id"]),
            stable_id=claim["stable_id"],
            work_id=claim["work_id"],
            proposal_id=claim["proposal_id"],
            gate_id=claim["gate_id"],
            authorization_id=claim["authorization_id"],
            logical_role=claim["logical_role"],
            logical_path=claim["logical_path"],
            source_id=claim["source_id"],
            origin=claim["origin"],
            revision=claim["revision"],
            fetch_recipe_sha256=claim["fetch_recipe_sha256"],
            evidence_ids=tuple(claim["evidence_ids"]),
            license_disposition=claim["license_disposition"],
        )
    )
    forged["claims"] = [claim]
    forged["predecessor_event_id"] = None
    forged["artifact_event_id"] = _event_id(forged)
    return forged


def test_cross_transaction_declared_only_divergence_folds(tmp_path: Path) -> None:
    """Two transactions naming one physical key with divergent DECLARED
    classification (media_type; object_id follows the declaration) fold
    without ambiguity -- the m5445/m13746 era shape."""

    ledger_path, context, mirrors, canonical, repo, staged_event = _healthy_store(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        ledger.append_event(_forged_staged_event(staged_event, media_type="text/markdown"))
    projection = validate_artifact_checkpoint(
        (ledger_path,),
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    assert len(projection.transactions) == 1


def test_cross_transaction_intrinsic_divergence_refuses(tmp_path: Path) -> None:
    """Two ledger rows naming one physical key with divergent INTRINSIC facts
    (byte_count) are shared-store corruption and refuse fatally."""

    ledger_path, context, mirrors, canonical, repo, staged_event = _healthy_store(tmp_path)
    honest_count = int(staged_event["objects"][0]["byte_count"])
    with ArtifactEventLedger(ledger_path) as ledger:
        ledger.append_event(_forged_staged_event(staged_event, byte_count=honest_count + 1))
    with pytest.raises(ArtifactCheckpointError, match="ambiguous intrinsic inventory"):
        validate_artifact_checkpoint(
            (ledger_path,),
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )
