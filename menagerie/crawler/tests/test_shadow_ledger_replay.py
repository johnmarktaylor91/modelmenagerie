"""AB-15 shadow-ledger replay tool: snapshot epoch proof, replay, owner census.

The tool's whole value is fail-closed properties, so every arm here is a
refusal or a tamper: a replay must never run against a live (un-snapshotted)
root, a drifted shadow must refuse instead of reporting, and the owner census
must disclose every owner-less failure reason rather than absorbing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot
from menagerie.crawler.tools.shadow_ledger_replay import (
    MANIFEST_NAME,
    attempts_owners,
    replay,
    snapshot,
)


def _fake_campaign(tmp_path: Path) -> tuple[Path, Path]:
    """Run one real fake leg and return its records root and intake root."""

    intake_snapshot = _snapshot(tmp_path, count=2)
    driver = _driver(tmp_path, intake_snapshot)
    result = driver.run()
    assert result.status == "complete"
    records_root = driver.paths.ledgers.models.parent.parent
    return records_root, intake_snapshot.root


@pytest.mark.smoke
def test_snapshot_replay_rederives_the_campaign_green(tmp_path: Path) -> None:
    """A snapshotted fake campaign replays green through the real machinery."""

    records_root, intake_root = _fake_campaign(tmp_path)
    shadow_root = tmp_path / "shadow"
    manifest = snapshot(records_root, shadow_root)
    assert manifest["files"], "the snapshot copied no campaign files"
    report = replay(shadow_root, intake_root, author_model=None, checker_model=None)
    assert report["green"] is True
    assert report["chain_refusals"] == []
    assert report["projected_records"] == 2
    assert report["ledger_heads"] == 2
    assert report["stale_heads"] == 0
    assert report["intake_members"] == 2
    assert report["spend"]["present"] is True
    assert report["spend"]["reservations"] == report["spend"]["settlements"]
    assert report["spend"]["unsettled_reservations"] == []


@pytest.mark.smoke
def test_replay_refuses_a_live_unsnapshotted_root(tmp_path: Path) -> None:
    """The live records root is never a replay substrate."""

    records_root, intake_root = _fake_campaign(tmp_path)
    with pytest.raises(ValueError, match="no shadow-snapshot-manifest"):
        replay(records_root, intake_root, author_model=None, checker_model=None)


@pytest.mark.smoke
def test_replay_refuses_a_tampered_shadow(tmp_path: Path) -> None:
    """One flipped byte anywhere in the shadow refuses the whole replay."""

    records_root, intake_root = _fake_campaign(tmp_path)
    shadow_root = tmp_path / "shadow"
    snapshot(records_root, shadow_root)
    models = shadow_root / "models" / "current-shard.jsonl"
    payload = models.read_bytes()
    models.write_bytes(payload.replace(b'"stable_id"', b'"stable_1d"', 1))
    with pytest.raises(ValueError, match="drifted"):
        replay(shadow_root, intake_root, author_model=None, checker_model=None)


@pytest.mark.smoke
def test_snapshot_refuses_a_source_that_changed_mid_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source mutating during the copy fails the one-epoch proof."""

    records_root, _intake_root = _fake_campaign(tmp_path)
    import menagerie.crawler.tools.shadow_ledger_replay as tool

    real_hash_tree = tool._hash_tree
    passes: list[int] = []

    def mutating_hash_tree(root: Path) -> dict[str, str]:
        digests = real_hash_tree(root)
        if root == records_root:
            passes.append(len(passes))
            if len(passes) == 2:
                target = records_root / "models" / "current-shard.jsonl"
                with target.open("ab") as handle:
                    handle.write(b"\n")
                return real_hash_tree(root)
        return digests

    monkeypatch.setattr(tool, "_hash_tree", mutating_hash_tree)
    with pytest.raises(ValueError, match="more than one census epoch"):
        tool.snapshot(records_root, tmp_path / "shadow")


@pytest.mark.smoke
def test_attempts_owner_census_is_closed_and_discloses_ownerless(
    tmp_path: Path,
) -> None:
    """Every recorded failure classifies into the closed owner vocabulary.

    The fake campaign's attempts all succeed (failures 0, census green); a
    forged failed attempt with an unknown reason, planted BEFORE the snapshot,
    must surface in ``ownerless`` and turn the census red -- disclosure, never
    absorption.
    """

    records_root, _intake_root = _fake_campaign(tmp_path)
    attempts_path = records_root / "attempts" / "local.jsonl"
    baseline_rows = len(tuple(scan_jsonl(attempts_path)))
    clean_shadow = tmp_path / "shadow-clean"
    snapshot(records_root, clean_shadow)
    clean = attempts_owners(clean_shadow)
    assert clean["green"] is True
    assert clean["attempts"] == baseline_rows
    assert clean["failed_attempts"] == sum(clean["owners"].values())

    forged = {
        "stable_id": "m_forged",
        "error": {"reason_code": "reason-nobody-owns", "details": {}},
    }
    with attempts_path.open("ab") as handle:
        handle.write(json.dumps(forged, sort_keys=True).encode("utf-8") + b"\n")
    forged_shadow = tmp_path / "shadow-forged"
    snapshot(records_root, forged_shadow)
    census = attempts_owners(forged_shadow)
    assert census["green"] is False
    assert census["ownerless"] == [
        {"stable_id": "m_forged", "reason_code": "reason-nobody-owns"}
    ]


@pytest.mark.smoke
def test_snapshot_manifest_names_every_campaign_file(tmp_path: Path) -> None:
    """The manifest is the shadow's complete tamper boundary."""

    records_root, _intake_root = _fake_campaign(tmp_path)
    shadow_root = tmp_path / "shadow"
    manifest = snapshot(records_root, shadow_root)
    manifest_files = set(manifest["files"])
    on_disk = {
        str(path.relative_to(shadow_root))
        for path in shadow_root.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    assert on_disk == manifest_files
