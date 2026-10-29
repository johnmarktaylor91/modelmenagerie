"""IC-2: cas_inventory_audit + attested-absent preflight + missing-of-STALE scoping.

The m8635 live shape: a torn publication left attested public objects
partially written; the fold tolerated the present ones (leg-39 fix) but an
ABSENT attested key was invisible until an un-staling action turned it into a
campaign-wide admission brick. The auditor names all three dispositions
read-only; the preflight arm refuses on attested-absent; and the fold's
missing refusal is scoped by the CLOSED current/stale registration partition:
missing-of-CURRENT stays a campaign brick, missing-of-STALE-only demotes that
one transaction to a typed custody-missing quarantine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointError,
    cas_inventory_audit,
    validate_artifact_checkpoint,
)
from menagerie.crawler.mirrors import MirrorClass
from menagerie.crawler.tests.test_stale_physical_inventory import _stale_public_store

pytestmark = pytest.mark.smoke


def _audit_index(rows):
    return {(row["mirror_class"], row["object_key"]): row["disposition"] for row in rows}


def test_m8635_shape_attested_absent_is_named_and_preflight_refuses(tmp_path: Path) -> None:
    """Deleting one attested (stale-shielded) public physical is reported as
    attested-absent, and the preflight arm refuses naming the key."""

    shards, context, mirrors, canonical, repo = _stale_public_store(tmp_path)
    del context, canonical, repo
    healthy_rows = cas_inventory_audit(shards, mirrors=mirrors)
    assert {row["disposition"] for row in healthy_rows} == {"attested-present"}

    torn_public = [
        mirrors.root(MirrorClass.PUBLIC) / key
        for mirror_class, key in mirrors.iter_objects(MirrorClass.PUBLIC)
    ]
    assert torn_public
    absent_key = torn_public[0].relative_to(mirrors.root(MirrorClass.PUBLIC)).as_posix()
    torn_public[0].unlink()

    rows = cas_inventory_audit(shards, mirrors=mirrors)
    index = _audit_index(rows)
    assert index[("public", absent_key)] == "attested-absent"
    with pytest.raises(ArtifactCheckpointError, match="cas inventory preflight refuses"):
        cas_inventory_audit(shards, mirrors=mirrors, preflight=True)


def test_present_unattested_partition_matches_ic3(tmp_path: Path) -> None:
    """On-disk files no event names split litter vs foreign exactly like the
    fold's IC-3 partition; litter alone never trips the preflight."""

    shards, context, mirrors, canonical, repo = _stale_public_store(tmp_path)
    del context, canonical, repo
    public_root = mirrors.root(MirrorClass.PUBLIC)
    (public_root / ".DS_Store").write_bytes(b"finder litter")
    rows = cas_inventory_audit(shards, mirrors=mirrors)
    index = _audit_index(rows)
    assert index[("public", ".DS_Store")] == "present-unattested-litter"
    cas_inventory_audit(shards, mirrors=mirrors, preflight=True)

    foreign_key = "sha256/bb/" + "b" * 64
    foreign = public_root / foreign_key
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"uncovenanted")
    rows = cas_inventory_audit(shards, mirrors=mirrors)
    assert _audit_index(rows)[("public", foreign_key)] == "present-unattested-foreign"
    with pytest.raises(ArtifactCheckpointError, match="preflight refuses"):
        cas_inventory_audit(shards, mirrors=mirrors, preflight=True)


def test_missing_of_stale_only_custody_demotes_not_bricks(tmp_path: Path) -> None:
    """A physical registered ONLY by a non-current chain that is absent from
    disk quarantines THAT transaction typed; the campaign folds on."""

    shards, context, mirrors, canonical, repo = _stale_public_store(tmp_path)
    # The torn chain's staged PRIVATE custody: registered pending custody,
    # not part of any CURRENT transaction. Its bytes differ from the healthy
    # transaction's, so the key is exclusively stale-owned.
    healthy_projection = validate_artifact_checkpoint(
        shards, context=context, mirrors=mirrors, canonical_root=canonical, repository_root=repo
    )
    healthy_private = {
        (MirrorClass(obj.mirror_class), obj.object_key)
        for tx in healthy_projection.transactions.values()
        for obj in tx.objects
    }
    stale_only = [
        mirrors.root(MirrorClass.PRIVATE) / key
        for mirror_class, key in mirrors.iter_objects(MirrorClass.PRIVATE)
        if (MirrorClass.PRIVATE, key) not in healthy_private
    ]
    assert stale_only, "the torn chain must hold private custody of its own"
    stale_only[0].unlink()

    projection = validate_artifact_checkpoint(
        shards, context=context, mirrors=mirrors, canonical_root=canonical, repository_root=repo
    )
    assert len(projection.transactions) == 1
    reasons = [entry.reason for entry in projection.stale_transactions.values()]
    assert any("artifact custody missing from mirror store" in reason for reason in reasons)


def test_missing_of_current_custody_still_bricks(tmp_path: Path) -> None:
    """The armed direction: a CURRENT transaction's registered physical
    missing from disk stays a campaign-wide refusal naming the key."""

    shards, context, mirrors, canonical, repo = _stale_public_store(tmp_path)
    projection = validate_artifact_checkpoint(
        shards, context=context, mirrors=mirrors, canonical_root=canonical, repository_root=repo
    )
    current_keys = [
        obj.object_key
        for tx in projection.transactions.values()
        for obj in tx.objects
        if obj.mirror_class == MirrorClass.PRIVATE.value
    ]
    assert current_keys
    (mirrors.root(MirrorClass.PRIVATE) / current_keys[0]).unlink()
    with pytest.raises(ArtifactCheckpointError, match="mirror inventory mismatch"):
        validate_artifact_checkpoint(
            shards, context=context, mirrors=mirrors, canonical_root=canonical, repository_root=repo
        )
