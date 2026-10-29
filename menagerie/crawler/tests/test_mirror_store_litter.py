"""IC-3: mirror-store litter must not brick every admission.

`MirrorStore.put` wrote its temporary BESIDE the destination and cleaned it in
a ``finally`` -- which never runs on SIGKILL -- and `iter_objects` rglob-
enumerates dot-names, so the store could litter ITSELF via any crash and then
refuse to admit anything campaign-wide until a human deleted a file the system
created. A single Finder-dropped ``.DS_Store`` had the same blast radius.

The funded shape (PLAN_v6 IC-3): put stages through ``<root>/.staging/``
(excluded from enumeration), and the inventory reconciliation classifies
non-addressable-shaped unattested files as typed ``store-litter`` disclosure
instead of a campaign brick. The boundary stays armed: an ADDRESSABLE-shaped
unattested file is indistinguishable from store corruption and still refuses
fatally (any further narrowing needs four-part review).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointError,
    ArtifactEventLedger,
    validate_artifact_checkpoint,
)
from menagerie.crawler.mirrors import (
    ArtifactOrigin,
    MirrorClass,
    RetentionClass,
    is_addressable_object_key,
)
from menagerie.crawler.tests.test_artifact_transactions import (
    _commit_private,
    _context,
    _mirrors,
)

pytestmark = pytest.mark.smoke


def _healthy_store(tmp_path: Path):
    """One healthy committed transaction over a real ledger and CAS."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_one")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _commit_private(
            "m_one",
            content=b"healthy bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-one",
        )
    return (ledger_path,), context, mirrors, canonical, repo


def _validate(shards, context, mirrors, canonical, repo):
    return validate_artifact_checkpoint(
        shards,
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )


def test_ds_store_litter_is_typed_disclosure_not_a_brick(tmp_path: Path) -> None:
    """One .DS_Store in an enumerated root must not abort every admission."""

    shards, context, mirrors, canonical, repo = _healthy_store(tmp_path)
    public_root = mirrors.root(MirrorClass.PUBLIC)
    public_root.mkdir(parents=True, exist_ok=True)
    (public_root / ".DS_Store").write_bytes(b"finder litter")
    projection = _validate(shards, context, mirrors, canonical, repo)
    assert (MirrorClass.PUBLIC.value, ".DS_Store") in projection.store_litter


def test_sigkill_orphaned_legacy_temp_is_litter_not_a_brick(tmp_path: Path) -> None:
    """A dotted put-temporary orphaned by SIGKILL (the pre-IC-3 beside-the-
    destination spelling) reads as litter, never as a foreign orphan."""

    shards, context, mirrors, canonical, repo = _healthy_store(tmp_path)
    private_root = mirrors.root(MirrorClass.PRIVATE)
    digest = "c" * 64
    parent = private_root / "sha256" / "cc"
    parent.mkdir(parents=True, exist_ok=True)
    orphaned = parent / f".{digest}.{os.getpid()}.tmp"
    orphaned.write_bytes(b"torn write")
    projection = _validate(shards, context, mirrors, canonical, repo)
    assert any(row[1].endswith(".tmp") for row in projection.store_litter)


def test_staging_dir_is_outside_the_enumerated_namespace(tmp_path: Path) -> None:
    """put() stages through <root>/.staging/; a SIGKILL-orphaned staging temp
    is invisible to enumeration and to the inventory reconciliation."""

    shards, context, mirrors, canonical, repo = _healthy_store(tmp_path)
    manifest = mirrors.put(
        b"fresh object",
        mirror_class=MirrorClass.LOCAL,
        retention_class=RetentionClass.LOCAL_EPHEMERAL,
        origin=ArtifactOrigin("https://example.test/fresh.tar", "v1"),
        media_type="application/x-tar",
    )
    assert not manifest.object_key.startswith(".staging/")
    staging = mirrors.root(MirrorClass.PRIVATE) / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "sha256-torn.12345.tmp").write_bytes(b"crashed mid-put")
    observed = set(mirrors.iter_objects())
    assert all(not key.startswith(".staging/") for _cls, key in observed)
    projection = _validate(shards, context, mirrors, canonical, repo)
    assert projection.store_litter == ()


def test_addressable_shaped_unattested_file_still_refuses(tmp_path: Path) -> None:
    """The armed boundary: an unattested file at a VALID content address is
    indistinguishable from corruption and stays a campaign-wide refusal."""

    shards, context, mirrors, canonical, repo = _healthy_store(tmp_path)
    key = "sha256/bb/" + "b" * 64
    destination = mirrors.root(MirrorClass.PUBLIC) / key
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"uncovenanted addressable bytes")
    with pytest.raises(ArtifactCheckpointError, match="mirror inventory mismatch"):
        _validate(shards, context, mirrors, canonical, repo)


def test_addressable_key_predicate_is_exact() -> None:
    """The litter/foreign partition keys on the exact canonical address shape,
    prefix consistency included."""

    good = "sha256/ab/ab" + "c" * 62
    assert is_addressable_object_key(good)
    assert not is_addressable_object_key(".DS_Store")
    assert not is_addressable_object_key("sha256/ab/" + "z" * 64)
    assert not is_addressable_object_key("sha256/xx/" + "a" * 64)
    # prefix must equal the digest's first two characters
    assert not is_addressable_object_key("sha256/ab/cd" + "c" * 62)
    assert not is_addressable_object_key("sha256/ab/" + "a" * 63)
    assert not is_addressable_object_key("other/ab/" + "a" * 64)
