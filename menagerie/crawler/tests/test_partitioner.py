"""Tests for the four tier-aligned crawler campaigns."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.cli import _snapshot_driver_config
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.partitioner import (
    CAMPAIGN_SPECS,
    REVIEW_COHORT_SIZE,
    CampaignPartitionError,
    assert_campaign_partition,
    campaign_binding_for_snapshot,
    load_campaign_bindings,
    partition_roster,
    refresh_campaign_manifest,
)
from menagerie.crawler.tools.stage_rung import main as stage_rung_main

#: Sub-second load-bearing coverage promoted into the mandatory smoke tier (T5-7).
pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).parents[3]
_ROSTER_PATH = _REPO_ROOT / "menagerie" / "data" / "crawl_roster.jsonl"
_RECORDS_ROOT = _REPO_ROOT / "menagerie" / "crawler" / "records"
_EXPECTED_COUNTS = {
    "c1-mech": 6_968,
    "c2-disco": 14_798,
    "c3-classics": 5_669,
    "c4-native": 1_047,
}
#: Keys the roster provenance repair adds; the frozen partitions predate them.
_REPAIR_KEYS = frozenset(
    {"schema", "stable_id", "found_by", "hint_urls", "hint_framework", "hint_paper", "state"}
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a test JSONL fixture as object rows.

    Parameters
    ----------
    path:
        JSONL path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed rows.
    """

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _original_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Return a repaired roster row reduced to its pre-repair fields.

    Parameters
    ----------
    row:
        Roster row, repaired or not.

    Returns
    -------
    dict[str, Any]
        The row without the repair's keys; an empty ``aliases`` list is dropped too,
        because the repair writes it on rows that had none.
    """

    return {
        key: value
        for key, value in row.items()
        if key not in _REPAIR_KEYS and not (key == "aliases" and value == [])
    }


def _natural_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Return natural keys for roster or intake rows.

    Parameters
    ----------
    rows:
        Roster-shaped rows.

    Returns
    -------
    set[tuple[str, str, str]]
        Name, zoo, and variant keys.
    """

    return {(str(row["name"]), str(row["zoo"]), str(row.get("variant", ""))) for row in rows}


def test_real_roster_campaigns_are_disjoint_and_complete() -> None:
    """Every real roster row lands in exactly one family-co-located campaign.

    The partitions are frozen from the pre-repair roster, so rows compare on their
    original fields only.
    """

    roster = [_original_fields(row) for row in _read_jsonl(_ROSTER_PATH)]
    emitted = {
        spec.campaign_id: _read_jsonl(_RECORDS_ROOT / "partitions" / f"{spec.campaign_id}.jsonl")
        for spec in CAMPAIGN_SPECS
    }

    assert len(roster) == 28_482
    assert {campaign_id: len(rows) for campaign_id, rows in emitted.items()} == _EXPECTED_COUNTS
    assert_campaign_partition(roster, emitted)
    expected = partition_roster(roster)
    assert all(emitted[campaign_id] == list(rows) for campaign_id, rows in expected.items())

    key_sets = [_natural_keys(rows) for rows in emitted.values()]
    assert sum(len(keys) for keys in key_sets) == len(set().union(*key_sets)) == len(roster)
    assert set().union(*key_sets) == _natural_keys(roster)

    family_owners: dict[str, set[str]] = defaultdict(set)
    for campaign_id, rows in emitted.items():
        for row in rows:
            family = row.get("family")
            if isinstance(family, str) and family:
                family_owners[family].add(campaign_id)
    assert all(len(owners) == 1 for owners in family_owners.values())


def test_c1_review_prefix_is_zoo_proportional() -> None:
    """The C1 canary prefix follows the full zoo mix within Hamilton rounding."""

    c1_rows = _read_jsonl(_RECORDS_ROOT / "partitions" / "c1-mech.jsonl")
    all_counts = Counter(str(row["zoo"]) for row in c1_rows)
    prefix_counts = Counter(str(row["zoo"]) for row in c1_rows[:REVIEW_COHORT_SIZE])

    assert len(c1_rows[:REVIEW_COHORT_SIZE]) == REVIEW_COHORT_SIZE
    for zoo, count in all_counts.items():
        exact_share = REVIEW_COHORT_SIZE * count / len(c1_rows)
        assert abs(prefix_counts[zoo] - exact_share) < 1


def test_step_100_ernie_rows_reconcile_with_bound_c1_intake() -> None:
    """T2-14: every ERNIE canary row resolves to its durable intake member."""

    c1_rows = _read_jsonl(_RECORDS_ROOT / "partitions" / "c1-mech.jsonl")
    ernie_rows = [row for row in c1_rows[:100] if str(row["name"]).startswith("ERNIE")]
    binding = next(
        row
        for row in load_campaign_bindings(_RECORDS_ROOT / "campaigns.json")
        if row.spec.campaign_id == "c1-mech"
    )
    snapshot = load_intake_snapshot(_RECORDS_ROOT / binding.intake_path)
    stable_id_by_key = {item.natural_key: item.stable_id for item in snapshot.items}
    resolved = {
        stable_id_by_key[(str(row["name"]), str(row["zoo"]), str(row.get("variant", "")))]
        for row in ernie_rows
    }

    assert {str(row["name"]) for row in ernie_rows} == {
        "ERNIE 3.0",
        "ERNIE-Doc",
        "ERNIE-Gram",
        "ERNIE-ViL",
        "ERNIE-ViLG",
    }
    assert len(resolved) == len(ernie_rows) == 5


def test_campaign_manifest_binds_models_partitions_and_intakes() -> None:
    """The manifest binds each campaign to exact models and verified snapshots."""

    manifest_path = _RECORDS_ROOT / "campaigns.json"
    bindings = load_campaign_bindings(manifest_path)
    expected_models = {
        # Seat swap (SEAT_RULING.md 2026-09-01): claude-fable-5-1 is the primary
        # checker seat on every campaign; gpt-5.6-sol stays the K=2 confirmation
        # seat and the DriverConfig default (same-identity test lanes).
        "c1-mech": ("claude-fable-5-1", "claude-fable-5-1"),
        "c2-disco": ("claude-sonnet", "claude-fable-5-1"),
        "c3-classics": ("claude-opus-5", "claude-fable-5-1"),
        "c4-native": ("claude-sonnet", "claude-fable-5-1"),
    }

    assert DriverConfig().author_model == "claude-fable-5-1"
    assert DriverConfig().checker_model == "gpt-5.6-sol"
    assert {
        binding.spec.campaign_id: (
            binding.spec.author_model,
            binding.spec.checker_model,
        )
        for binding in bindings
    } == expected_models

    for binding in bindings:
        partition_path = _RECORDS_ROOT / binding.partition_path
        partition_rows = _read_jsonl(partition_path)
        snapshot = load_intake_snapshot(_RECORDS_ROOT / binding.intake_path)
        intake_rows = [item.to_dict() for item in snapshot.items]

        assert binding.row_count == _EXPECTED_COUNTS[binding.spec.campaign_id]
        assert hash_bytes(partition_path.read_bytes()) == binding.partition_sha256
        assert len(snapshot.items) == binding.row_count
        assert _natural_keys(partition_rows) == _natural_keys(intake_rows)
        assert campaign_binding_for_snapshot(manifest_path, snapshot) == binding
        config = _snapshot_driver_config(_REPO_ROOT, snapshot)
        assert config.author_model == binding.spec.author_model
        assert config.checker_model == binding.spec.checker_model


def test_refresh_campaign_manifest_reprojects_specs_and_preserves_bindings(
    tmp_path: Path,
) -> None:
    """Refresh rotates non-identity projections and PRESERVES recorded identities.

    OPS-10 asked for a way to re-project a stale ``campaigns.json``. Since BL-3
    that is a convenience rather than a cure: a declared identity rotation loads
    fine, so the default refresh must not overwrite ``author_model`` /
    ``checker_model``. Overwriting them in place is exactly how the pre-BL-3
    refresh destroyed history -- after the "cure" the manifest asserted Sol had
    adjudicated the step-100 rung, which Terra did. An explicit rotation is
    available and records what it superseded.
    """

    current = json.loads((_RECORDS_ROOT / "campaigns.json").read_text(encoding="utf-8"))
    stale = json.loads(json.dumps(current))
    for row in stale["campaigns"]:
        if row["campaign_id"] == "c1-mech":
            # A DECLARED prior binding on both identity axes (c1-mech ran under
            # claude-sonnet before 2026-08-11 and under gpt-5.6-terra before
            # 2026-08-14), plus a rotated non-identity projection.
            row["author_model"] = "claude-sonnet"
            row["checker_model"] = "gpt-5.6-terra"
            row["phase"] = "pytorch-legacy-label"
            row["operator_note"] = "non-spec content the refresh must not touch"
    records_root = tmp_path / "records"
    records_root.mkdir()
    manifest_path = records_root / "campaigns.json"
    manifest_path.write_text(json.dumps(stale, sort_keys=True) + "\n", encoding="utf-8")

    # The whole point of BL-3: a declared rotation is READABLE, and the drift is
    # disclosed rather than fatal. Strict mode is the only reading that refuses.
    c1 = next(
        binding
        for binding in load_campaign_bindings(manifest_path)
        if binding.spec.campaign_id == "c1-mech"
    )
    assert c1.projection_drift == ("author_model", "checker_model", "phase")
    assert dict(c1.recorded_projection)["checker_model"] == "gpt-5.6-terra"
    assert c1.spec.checker_model == "claude-fable-5-1"
    with pytest.raises(CampaignPartitionError, match="projections have drifted"):
        load_campaign_bindings(manifest_path, require_current_projection=True)

    refreshed_bindings = refresh_campaign_manifest(manifest_path)
    assert refreshed_bindings == load_campaign_bindings(manifest_path)

    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert refreshed["roster"] == current["roster"]
    assert refreshed["proof"] == current["proof"]
    spec_fields = {
        "workload_class",
        "phase",
        "author_model",
        "checker_model",
        "review_checkpoint_at",
        "review_cohort",
    }
    for stale_row, refreshed_row in zip(stale["campaigns"], refreshed["campaigns"]):
        spec = next(
            spec for spec in CAMPAIGN_SPECS if spec.campaign_id == stale_row["campaign_id"]
        )
        # Non-identity projections are re-projected; identities are preserved.
        assert refreshed_row["phase"] == spec.phase
        assert refreshed_row["author_model"] == stale_row["author_model"]
        assert refreshed_row["checker_model"] == stale_row["checker_model"]
        for field in set(stale_row) - spec_fields:
            assert refreshed_row[field] == stale_row[field]
    assert refreshed["campaigns"][0]["operator_note"] == (
        "non-spec content the refresh must not touch"
    )

    # The explicit rotation moves them and keeps what it superseded.
    rotated_bindings = refresh_campaign_manifest(
        manifest_path, rotate_identity_projections=True
    )
    assert all(binding.projection_drift == () for binding in rotated_bindings)
    rotated_c1 = next(
        row
        for row in json.loads(manifest_path.read_text(encoding="utf-8"))["campaigns"]
        if row["campaign_id"] == "c1-mech"
    )
    assert rotated_c1["author_model"] == "claude-fable-5-1"
    assert rotated_c1["checker_model"] == "claude-fable-5-1"
    # The committed manifest already carries its own truthful supersession
    # history (the 2026-09-01 fable -> fable-5.1 rotation); the injected stale
    # bindings append AFTER it, never overwrite it.
    prior_supersessions = next(
        row.get("superseded_projection", [])
        for row in current["campaigns"]
        if row["campaign_id"] == "c1-mech"
    )
    assert rotated_c1["superseded_projection"] == [
        *prior_supersessions,
        {"field": "author_model", "value": "claude-sonnet"},
        {"field": "checker_model", "value": "gpt-5.6-terra"},
    ]

    # A structurally broken manifest refuses instead of being "refreshed" into
    # validity: the remediation rotates spec projections, it never repairs.
    broken = records_root / "broken.json"
    broken.write_text(json.dumps({"format": "wrong"}), encoding="utf-8")
    with pytest.raises(CampaignPartitionError):
        refresh_campaign_manifest(broken)

    # The operator spelling: stage_rung refresh-manifest re-emits in place.
    stale_again = records_root / "cli" / "campaigns.json"
    stale_again.parent.mkdir()
    stale_again.write_text(json.dumps(stale, sort_keys=True) + "\n", encoding="utf-8")
    assert stage_rung_main(["refresh-manifest", "--records-root", str(stale_again.parent)]) == 0
    assert load_campaign_bindings(stale_again)
