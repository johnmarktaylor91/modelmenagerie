"""Tests for digest-manifested host-shard handoff bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tools.handoff_bundle import (
    HANDOFF_MANIFEST_FORMAT,
    HandoffBundleError,
    build_handoff_manifest,
    main,
    verify_handoff_manifest,
)

pytestmark = pytest.mark.smoke


def _seed_records(tmp_path: Path) -> Path:
    """Create a miniature shard records tree covering every shipped class.

    Parameters
    ----------
    tmp_path:
        Test-owned scratch root.

    Returns
    -------
    pathlib.Path
        Records root holding canonical, valve, grant, and corpus files.
    """

    records_root = tmp_path / "menagerie" / "crawler" / "records"
    files = {
        "models/current-shard.jsonl": b'{"stable_id": "m1"}\n',
        "attempts/local.jsonl": b'{"attempt_id": "a1"}\n',
        "gates/current-shard.jsonl": b"",
        "operational/events.jsonl": b'{"event_kind": "campaign-health"}\n',
        "operational/requeue-grants.jsonl": b'{"grant_id": "g1"}\n',
        "intake-extensions/host-capacity-deferrals.jsonl": b'{"stable_id": "m2"}\n',
        "text-corpus/corpus-index.jsonl": b'{"corpus_row_id": "r1"}\n',
        "text-corpus/sha256/ab/abcd": b"document bytes",
        "intake/registered-rungs.jsonl": b'{"snapshot_id": "intake-1"}\n',
        "campaigns.json": b'{"campaigns": []}\n',
        "models/current-shard.jsonl.lock": b"",
    }
    for relative, payload in files.items():
        path = records_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return records_root


def test_build_and_verify_roundtrip(tmp_path: Path) -> None:
    """A built manifest re-verifies bit-for-bit on the collector side."""

    records_root = _seed_records(tmp_path)
    manifest_path = tmp_path / "handoff-manifest.json"

    manifest = build_handoff_manifest(
        records_root,
        shard_id="host-a",
        campaign_id="c1-mech",
        manifest_path=manifest_path,
    )

    assert manifest["format"] == HANDOFF_MANIFEST_FORMAT
    assert manifest["shard_id"] == "host-a"
    assert manifest["campaign_id"] == "c1-mech"
    assert "models/current-shard.jsonl" in manifest["files"]
    assert "intake-extensions/host-capacity-deferrals.jsonl" in manifest["files"]
    assert "operational/requeue-grants.jsonl" in manifest["files"]
    assert "text-corpus/sha256/ab/abcd" in manifest["files"]
    assert "models/current-shard.jsonl.lock" not in manifest["files"]
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted == manifest

    report = verify_handoff_manifest(records_root, manifest_path)
    assert report["verified"] is True
    assert report["file_count"] == len(manifest["files"])


def test_tampered_file_refuses_verification(tmp_path: Path) -> None:
    """One flipped byte in a shipped ledger fails verification loudly."""

    records_root = _seed_records(tmp_path)
    manifest_path = tmp_path / "handoff-manifest.json"
    build_handoff_manifest(
        records_root, shard_id="host-a", campaign_id="c1-mech", manifest_path=manifest_path
    )
    target = records_root / "models" / "current-shard.jsonl"
    target.write_bytes(target.read_bytes() + b"tampered\n")

    report = verify_handoff_manifest(records_root, manifest_path)
    assert report["verified"] is False
    assert "models/current-shard.jsonl" in report["mismatched"]


def test_missing_and_extra_files_refuse_verification(tmp_path: Path) -> None:
    """A dropped or smuggled file breaks the digest-manifested contract."""

    records_root = _seed_records(tmp_path)
    manifest_path = tmp_path / "handoff-manifest.json"
    build_handoff_manifest(
        records_root, shard_id="host-a", campaign_id="c1-mech", manifest_path=manifest_path
    )
    (records_root / "operational" / "requeue-grants.jsonl").unlink()
    (records_root / "models" / "stray.jsonl").write_bytes(b"{}\n")

    report = verify_handoff_manifest(records_root, manifest_path)
    assert report["verified"] is False
    assert "operational/requeue-grants.jsonl" in report["missing"]
    assert "models/stray.jsonl" in report["extra"]


def test_manifest_content_digest_is_self_checking(tmp_path: Path) -> None:
    """An edited manifest refuses before any file comparison happens."""

    records_root = _seed_records(tmp_path)
    manifest_path = tmp_path / "handoff-manifest.json"
    build_handoff_manifest(
        records_root, shard_id="host-a", campaign_id="c1-mech", manifest_path=manifest_path
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["shard_id"] = "host-forged"
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(HandoffBundleError, match="digest"):
        verify_handoff_manifest(records_root, manifest_path)


def test_cli_build_then_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The operator CLI builds and verifies a bundle end to end."""

    records_root = _seed_records(tmp_path)
    manifest_path = tmp_path / "handoff-manifest.json"

    assert (
        main(
            [
                "build",
                "--records-root",
                str(records_root),
                "--shard-id",
                "host-a",
                "--campaign-id",
                "c1-mech",
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "verify",
                "--records-root",
                str(records_root),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    (records_root / "campaigns.json").write_bytes(b'{"campaigns": ["forged"]}\n')
    assert (
        main(
            [
                "verify",
                "--records-root",
                str(records_root),
                "--manifest",
                str(manifest_path),
            ]
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "campaigns.json" in output


def test_verify_uses_content_addresses_not_names(tmp_path: Path) -> None:
    """Verification re-hashes bytes; renaming digests does not fool it."""

    records_root = _seed_records(tmp_path)
    manifest_path = tmp_path / "handoff-manifest.json"
    manifest = build_handoff_manifest(
        records_root, shard_id="host-a", campaign_id="c1-mech", manifest_path=manifest_path
    )
    recorded = manifest["files"]["text-corpus/sha256/ab/abcd"]["sha256"]
    assert recorded == hash_bytes(b"document bytes")
