"""Blocking-item resolution schema: misuse RED tests + completeness census.

These fixtures are the SAME misuse matrix run at every consumer of the
blocking-item-resolution fact: no arbitrary unmerged blocker, no unit missing
a specified dependent exclusion or stated negative disposition, and no
producer-supplied status or producer-created unit can use the unsigned-arm
receipt. Mixed signed/unsigned composed-item fixtures prove per-arm
disclosure, never wholly-merged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.tools.blocking_resolution import (
    BLOCKING_TIERS,
    CENSUS_SCHEMA_VERSION,
    ENVELOPE_SCHEMA_VERSION,
    FIRST_GENERATION_SURFACES,
    ResolutionSchemaError,
    derive_units,
    evaluate_census,
    load_envelope,
    write_census,
)


def _envelope(
    items: list[dict[str, Any]],
    arm_table: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble one canonical resolution envelope."""

    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "items": items,
        "arm_table": arm_table,
        "evidence": evidence,
    }


def _merged(item_id: str, *, included_arms: list[str] | None = None) -> dict[str, Any]:
    """Build one merge-receipt evidence row."""

    return {
        "kind": "merged",
        "item_id": item_id,
        "sha": "a" * 40,
        "test_node": f"tests::{item_id}",
        "included_arms": included_arms or [],
    }


def _c8(item_id: str, arm_id: str, *, exclusions: list[str] | None = None) -> dict[str, Any]:
    """Build one c8_unsigned_negative receipt row."""

    return {
        "kind": "c8_unsigned_negative",
        "item_id": item_id,
        "arm_id": arm_id,
        "deadline_event": "seal-2026-08-30",
        "audit_register_row": f"BLOCKED-ON-JMT-SIGNATURE:{item_id}/{arm_id}",
        "dependent_exclusions_present": exclusions or [],
    }


_W14_ARMS = [
    {
        "item_id": "W14",
        "arm_id": arm,
        "signed_at_seal": signed,
        "stated_negative_disposition": f"class {arm} stays closed; releases refuse",
        "dependent_exclusions": [f"exclude:{arm}"],
    }
    for arm, signed in (
        ("quota-deferral", True),
        ("corpus-conflict", True),
        ("blocked-prerequisite-post-rotation", False),
        ("engine-invalidated-artifact-cache", False),
        ("terminal-vocabulary", False),
    )
]
_W14_ITEM = {"item_id": "W14", "tier": "200B-BLOCKING"}


@pytest.mark.smoke
def test_census_green_with_ordinary_merges() -> None:
    """Every ordinary unit merged with sha+node makes the census pass."""

    census = evaluate_census(
        _envelope(
            [
                {"item_id": "A", "tier": "200B-BLOCKING"},
                {"item_id": "B", "tier": "200B-BLOCKING-EFFICIENCY"},
            ],
            [],
            [_merged("A"), _merged("B")],
        )
    )
    assert census["schema_version"] == CENSUS_SCHEMA_VERSION
    assert census["status"] == "passed"
    assert census["operationally_resolved"] == []


@pytest.mark.smoke
def test_arbitrary_unmerged_blocker_fails_and_cannot_use_c8() -> None:
    """An unresolved unit fails the census; a no-arm item can never take the
    unsigned-arm receipt."""

    census = evaluate_census(_envelope([{"item_id": "A", "tier": "200B-BLOCKING"}], [], []))
    assert census["status"] == "failed"
    assert census["unresolved"] == ["A/base"]
    census = evaluate_census(
        _envelope(
            [{"item_id": "A", "tier": "200B-BLOCKING"}],
            [],
            [_c8("A", "")],
        )
    )
    assert census["status"] == "failed"
    assert any("not a single signature arm" in reason for reason in census["refusals"])


@pytest.mark.smoke
def test_tier_vocabulary_is_closed() -> None:
    """A blocking-tier spelling outside the closed set refuses outright."""

    assert BLOCKING_TIERS == {"200B-BLOCKING", "200B-BLOCKING-EFFICIENCY"}
    with pytest.raises(ResolutionSchemaError, match="tier-vocabulary-refused"):
        derive_units([{"item_id": "A", "tier": "200B-BLOCKING-URGENT"}], [])


@pytest.mark.smoke
def test_first_generation_enumeration_is_exactly_four() -> None:
    """Membership closure: the first-generation surface set is exactly four.

    A later change self-declaring first-generation status fails this census
    and goes to the wave-default STOP-and-file path.
    """

    assert FIRST_GENERATION_SURFACES == {
        "certificate-kind-vocabulary",
        "per-kind-required-input-manifest",
        "blocking-tiers-census-membership",
        "blocking-item-resolution-vocabulary",
    }


@pytest.mark.smoke
def test_legacy_resolution_artifact_refuses_typed(tmp_path: Path) -> None:
    """FIRST-GENERATION proof: pre-schema resolution-shaped artifacts refuse
    typed AS envelope inputs rather than parse."""

    legacy = tmp_path / "old-merge-bookkeeping.json"
    legacy.write_text(json.dumps({"merged": ["W1", "W2"], "dead": []}), encoding="utf-8")
    with pytest.raises(ResolutionSchemaError, match="resolution-envelope-legacy"):
        load_envelope(legacy)
    not_json = tmp_path / "notes.json"
    not_json.write_text("resolution: all done\n", encoding="utf-8")
    with pytest.raises(ResolutionSchemaError, match="resolution-envelope-invalid"):
        load_envelope(not_json)


@pytest.mark.smoke
def test_dead_with_evidence_is_efficiency_tier_only() -> None:
    """The DEAD-WITH-EVIDENCE exit exists only for the efficiency tier and
    requires its disclosed census entry."""

    dead = {"kind": "dead_with_evidence", "item_id": "A", "census_entry": "row-7"}
    census = evaluate_census(_envelope([{"item_id": "A", "tier": "200B-BLOCKING"}], [], [dead]))
    assert census["status"] == "failed"
    assert any("efficiency-tier only" in reason for reason in census["refusals"])
    census = evaluate_census(
        _envelope([{"item_id": "A", "tier": "200B-BLOCKING-EFFICIENCY"}], [], [dead])
    )
    assert census["status"] == "passed"
    assert census["operationally_resolved"][0]["resolution"] == "dead_with_evidence"
    undisclosed = {"kind": "dead_with_evidence", "item_id": "A"}
    census = evaluate_census(
        _envelope([{"item_id": "A", "tier": "200B-BLOCKING-EFFICIENCY"}], [], [undisclosed])
    )
    assert census["status"] == "failed"


@pytest.mark.smoke
def test_c8_non_vacuity_and_dependent_exclusions() -> None:
    """A c8 receipt refuses without a stated disposition or a required
    dependent exclusion (the non-vacuity rule)."""

    arms = [
        {
            "item_id": "I",
            "arm_id": "narrowing",
            "signed_at_seal": False,
            "stated_negative_disposition": "",
            "dependent_exclusions": [],
        }
    ]
    items = [{"item_id": "I", "tier": "200B-BLOCKING"}]
    census = evaluate_census(_envelope(items, arms, [_merged("I"), _c8("I", "narrowing")]))
    assert census["status"] == "failed"
    assert any("non-vacuity" in reason for reason in census["refusals"])
    arms[0]["stated_negative_disposition"] = "narrowing does not ship; wide path stays"
    arms[0]["dependent_exclusions"] = ["exclude:redrive"]
    census = evaluate_census(_envelope(items, arms, [_merged("I"), _c8("I", "narrowing")]))
    assert census["status"] == "failed"
    assert any("dependent exclusions" in reason for reason in census["refusals"])
    census = evaluate_census(
        _envelope(
            items,
            arms,
            [_merged("I"), _c8("I", "narrowing", exclusions=["exclude:redrive"])],
        )
    )
    assert census["status"] == "passed"
    assert [row["unit"] for row in census["operationally_resolved"]] == ["narrowing"]


@pytest.mark.smoke
def test_producer_created_units_and_statuses_refuse() -> None:
    """Evidence naming underivable units, or carrying a foreign kind, refuses."""

    census = evaluate_census(
        _envelope(
            [{"item_id": "A", "tier": "200B-BLOCKING"}],
            [],
            [_merged("A"), _merged("GHOST")],
        )
    )
    assert census["status"] == "failed"
    assert any("producer-created" in reason for reason in census["refusals"])
    census = evaluate_census(
        _envelope(
            [{"item_id": "A", "tier": "200B-BLOCKING"}],
            [],
            [{"kind": "resolved", "item_id": "A", "status": "done"}],
        )
    )
    assert census["status"] == "failed"
    assert any("closed" in reason for reason in census["refusals"])


@pytest.mark.smoke
def test_w14_mixed_signed_unsigned_fixtures() -> None:
    """The composed-item matrix: per-arm disclosure, never wholly-merged.

    (i) base merged binding the two signed classes + valid receipts for the
    three unsigned arms = resolved with disclosed rows; (ii) one receipt
    missing = refused; (iii) a merge receipt claiming an unsigned arm =
    refused; (iv) an arm both included and receipted = duplicate refusal.
    """

    signed = ["quota-deferral", "corpus-conflict"]
    unsigned = [
        "blocked-prerequisite-post-rotation",
        "engine-invalidated-artifact-cache",
        "terminal-vocabulary",
    ]
    receipts = [_c8("W14", arm, exclusions=[f"exclude:{arm}"]) for arm in unsigned]

    # (i) fully lawful mixed resolution.
    census = evaluate_census(
        _envelope(
            [_W14_ITEM],
            _W14_ARMS,
            [_merged("W14", included_arms=signed), *receipts],
        )
    )
    assert census["status"] == "passed"
    assert census["units_total"] == 6  # base + five arms
    disclosed = {row["unit"] for row in census["operationally_resolved"]}
    assert disclosed == set(unsigned)
    merged_units = {row["unit"] for row in census["units"] if row["resolution"] == "merged"}
    assert merged_units == {"base"}  # never wholly merged

    # (ii) one unsigned-arm receipt missing: the census refuses completeness.
    census = evaluate_census(
        _envelope(
            [_W14_ITEM],
            _W14_ARMS,
            [_merged("W14", included_arms=signed), *receipts[:-1]],
        )
    )
    assert census["status"] == "failed"
    assert "W14/terminal-vocabulary" in census["unresolved"]

    # (iii) a merge receipt claiming an arm whose signature was absent.
    census = evaluate_census(
        _envelope(
            [_W14_ITEM],
            _W14_ARMS,
            [_merged("W14", included_arms=signed + [unsigned[0]]), *receipts],
        )
    )
    assert census["status"] == "failed"
    assert any("absent at the seal" in reason for reason in census["refusals"])

    # (iv) an arm both in the included set and carrying a receipt duplicates.
    arms = [dict(row) for row in _W14_ARMS]
    for row in arms:
        if row["arm_id"] == unsigned[0]:
            row["signed_at_seal"] = True
    census = evaluate_census(
        _envelope(
            [_W14_ITEM],
            arms,
            [_merged("W14", included_arms=signed + [unsigned[0]]), *receipts],
        )
    )
    assert census["status"] == "failed"
    assert any("duplicate" in reason for reason in census["refusals"])


@pytest.mark.smoke
def test_entire_substance_arm_item_may_use_c8() -> None:
    """An item whose ENTIRE substance is one signature arm resolves by receipt."""

    items = [{"item_id": "W8", "tier": "200B-BLOCKING", "entire_substance_is_arm": True}]
    arms = [
        {
            "item_id": "W8",
            "arm_id": "fidelity-no-consumer-widening",
            "signed_at_seal": False,
            "stated_negative_disposition": "the five parks stay parked; no re-drive",
            "dependent_exclusions": ["exclude:park-redrives"],
            "produces_unit": True,
        }
    ]
    receipt = _c8("W8", "", exclusions=["exclude:park-redrives"])
    census = evaluate_census(_envelope(items, arms, [receipt]))
    assert census["status"] == "passed"
    assert census["operationally_resolved"][0]["item_id"] == "W8"


@pytest.mark.smoke
def test_census_artifact_is_mint_consumable(tmp_path: Path) -> None:
    """write_census emits the schema the certificate manifest declares."""

    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text(
        json.dumps(_envelope([{"item_id": "A", "tier": "200B-BLOCKING"}], [], [_merged("A")])),
        encoding="utf-8",
    )
    census = write_census(envelope_path, tmp_path / "census.json")
    stored = json.loads((tmp_path / "census.json").read_text(encoding="utf-8"))
    assert stored == census
    assert stored["schema_version"] == "menagerie.crawler.blocking-tiers-census.v1"
    assert stored["status"] == "passed"
