"""AT-11: real production-harvest arms for the leg-39 stale-physical tolerance.

967b32a0 shipped with a vacuous regression test (it constructed three local
sets and performed the subtraction inline -- deleting the production harvest
would not have failed it). These arms exercise the PRODUCTION inventory
reconciliation (`validate_artifact_checkpoint` -> `_full_checkpoint_fold`)
over a real temp ledger and CAS:

* tolerated arm -- the leg-39/m8635 shape: a quarantined-stale transaction's
  ledger-attested PUBLIC physical (written by a crashed partial publication)
  is inventory-tolerated, never a foreign orphan. This arm IS the mutation
  check: delete the fold's ledger-attested-keys harvest and it fails.
* foreign arm (negative control): a file NO ledger event names still refuses
  campaign-wide, naming the orphan -- proving the tolerance did not widen.
* missing arm: an expected-but-missing registered physical still refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointError,
    ArtifactEventLedger,
    validate_artifact_checkpoint,
)
from menagerie.crawler.mirrors import MirrorClass
from menagerie.crawler.tests.test_artifact_transactions import (
    _commit_private,
    _context,
    _mirrors,
)
from menagerie.crawler.tests.test_artifact_chain_spent import _publish_public_chain

pytestmark = pytest.mark.smoke


def _stale_public_store(tmp_path: Path):
    """Build one healthy committed transaction plus one quarantined-stale
    2-event chain whose authorized PUBLIC physical is on disk (the crashed
    partial publication that armed leg-39)."""

    repo = tmp_path / "repo"
    canonical = repo / "menagerie" / "crawler"
    ledger_path = canonical / "records" / "artifacts" / "shard.jsonl"
    context = _context("m_healthy", "m_stale")
    mirrors = _mirrors(tmp_path)
    with ArtifactEventLedger(ledger_path) as ledger:
        _commit_private(
            "m_healthy",
            content=b"healthy committed bytes",
            context=context,
            mirrors=mirrors,
            ledger=ledger,
            canonical_root=canonical,
            repository_root=repo,
            evidence_id="ev-healthy",
        )
    # A second shard holds the interrupted chain: staged + publication
    # authorized, then the process died before reconstruction/publication.
    stale_shard = canonical / "records" / "artifacts" / "stale-shard.jsonl"
    stale_ledger, stale_mirrors, _, _ = _publish_public_chain(
        tmp_path / "stale", "m_stale", stop_after_authorization=True
    )
    with stale_ledger:
        events = list(stale_ledger.events)
    assert [event["event_kind"] for event in events] == [
        "staged-private",
        "publication-authorized",
    ]
    with ArtifactEventLedger(stale_shard) as shard:
        for event in events:
            logical = {
                key: value
                for key, value in event.items()
                if key not in {"ledger_seq", "payload_sha256"}
            }
            shard.append_event(logical)
    authorization_event = events[1]
    public_rows = [
        row
        for row in authorization_event["objects"]
        if row.get("mirror_class") == MirrorClass.PUBLIC.value
    ]
    assert public_rows, "authorized chain must attest a public object"
    # Replay the crashed host's physical state into this store: the staged
    # PRIVATE custody bytes exist, and the crashed partial publication also
    # landed the attested PUBLIC bytes -- while the chain never finalized.
    # This is the leg-39 shape the vacuous test never constructed.
    for mirror_class, key in stale_mirrors.iter_objects():
        content = (stale_mirrors.root(mirror_class) / key).read_bytes()
        destination = mirrors.root(mirror_class) / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    for row in public_rows:
        destination = mirrors.root(MirrorClass.PUBLIC) / str(row["object_key"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"m12646-shaped public source")
    return (ledger_path, stale_shard), context, mirrors, canonical, repo


def test_ledger_attested_stale_physicals_are_tolerated(tmp_path: Path) -> None:
    """The production reconciliation tolerates a stale transaction's attested
    public physical. Deleting the fold's ledger-attested-keys harvest turns
    this file into a foreign orphan and fails this test (mutation check)."""

    shards, context, mirrors, canonical, repo = _stale_public_store(tmp_path)
    projection = validate_artifact_checkpoint(
        shards,
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    stale_keys = {key[2] for key in projection.stale_transactions}
    assert len(projection.stale_transactions) == 1
    assert len(projection.transactions) == 1
    assert all(tx.stable_id == "m_healthy" for tx in projection.transactions.values())
    assert stale_keys, "the interrupted chain must quarantine, not project authority"


def test_unattested_foreign_file_still_refuses(tmp_path: Path) -> None:
    """Negative control: a file NO ledger event names is still a campaign
    brick naming the orphan -- the tolerance is scoped to attested keys."""

    shards, context, mirrors, canonical, repo = _stale_public_store(tmp_path)
    foreign_key = "sha256/bb/" + "b" * 64
    foreign = mirrors.root(MirrorClass.PUBLIC) / foreign_key
    foreign.parent.mkdir(parents=True, exist_ok=True)
    foreign.write_bytes(b"uncovenanted bytes")
    with pytest.raises(ArtifactCheckpointError, match="mirror inventory mismatch") as excinfo:
        validate_artifact_checkpoint(
            shards,
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )
    assert foreign_key in str(excinfo.value)


def test_expected_but_missing_registered_physical_still_refuses(tmp_path: Path) -> None:
    """The other direction stays armed: deleting a CURRENT transaction's
    registered physical refuses with the missing key named (a stale-only
    custody gap demotes instead -- see test_cas_inventory_audit)."""

    shards, context, mirrors, canonical, repo = _stale_public_store(tmp_path)
    projection = validate_artifact_checkpoint(
        shards,
        context=context,
        mirrors=mirrors,
        canonical_root=canonical,
        repository_root=repo,
    )
    current_private = [
        obj.object_key
        for tx in projection.transactions.values()
        for obj in tx.objects
        if obj.mirror_class == MirrorClass.PRIVATE.value
    ]
    assert current_private, "healthy committed transaction must hold private custody"
    (mirrors.root(MirrorClass.PRIVATE) / current_private[0]).unlink()
    with pytest.raises(ArtifactCheckpointError, match="missing"):
        validate_artifact_checkpoint(
            shards,
            context=context,
            mirrors=mirrors,
            canonical_root=canonical,
            repository_root=repo,
        )
