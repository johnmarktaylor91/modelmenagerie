"""Membership census + durable acknowledgments: the staging-door authorities.

LADDER row "campaign-scoped membership census + durable acknowledgments"
named this module for months while it did not exist (a doc-phantom), and
``rung_stratification._reconcile_membership`` / ``append_membership_
acknowledgment`` had ZERO test references — on the exact staging-door path
that composes the next rung's intake. These are the real tests: foreign ids
refuse, sibling record-holders stay informational, acknowledgments are
disposition-invariant disclosures, and the durable ledger is idempotent,
content-addressed, and campaign-scoped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.rung_stratification import (
    MembershipReconciliationError,
    StratificationError,
    _reconcile_membership,
    append_membership_acknowledgment,
    load_membership_acknowledgments,
    membership_acknowledgment_path,
)

_UNIVERSE = frozenset({"m1", "m2", "m3"})
_SIBLING = frozenset({"s1"})


def _census(**overrides: object):
    """Run the census with green defaults; overrides shape each scenario."""

    from typing import Any

    arguments: dict[str, Any] = {
        "universe": _UNIVERSE,
        "sibling_universe": _SIBLING,
        "heads": {"m1": {}},
        "attempt_subjects": frozenset({"m1"}),
        "valves": {},
        "prior_members": frozenset({"m1", "m2", "m3"}),
        "acknowledged": frozenset(),
        "durable_acknowledged": frozenset(),
    }
    arguments.update(overrides)
    return _reconcile_membership(**arguments)


@pytest.mark.smoke
def test_clean_census_passes_and_accounts_nothing() -> None:
    """A universe-consistent records set reconciles with empty disclosures."""

    report = _census()
    assert report == {
        "foreign_to_partition": [],
        "records_outside_this_campaign": [],
        "outside_registered_lineage": [],
        "acknowledged": [],
    }


@pytest.mark.smoke
def test_foreign_record_holder_refuses_without_acknowledgment() -> None:
    """An id in NO campaign's universe refuses the composition typed."""

    with pytest.raises(MembershipReconciliationError) as excinfo:
        _census(heads={"m1": {}, "ghost": {}})
    assert "ghost" in str(excinfo.value)
    # Acknowledged, the same id composes — but stays DISCLOSED as foreign
    # (acknowledgment is disposition-invariant: it discloses, never selects).
    report = _census(heads={"m1": {}, "ghost": {}}, acknowledged=frozenset({"ghost"}))
    assert report["foreign_to_partition"] == ["ghost"]
    assert report["acknowledged"] == ["ghost"]


@pytest.mark.smoke
def test_sibling_record_holder_is_informational_never_refusing() -> None:
    """A sibling campaign's record-holder is accounted, not foreign (C-27)."""

    report = _census(heads={"m1": {}, "s1": {}})
    assert report["records_outside_this_campaign"] == ["s1"]
    assert report["foreign_to_partition"] == []


@pytest.mark.smoke
def test_unregistered_lineage_refuses_without_acknowledgment() -> None:
    """An in-universe record holder outside registered lineage refuses."""

    with pytest.raises(MembershipReconciliationError) as excinfo:
        _census(prior_members=frozenset({"m2", "m3"}))
    assert "m1" in str(excinfo.value)
    report = _census(prior_members=frozenset({"m2", "m3"}), acknowledged=frozenset({"m1"}))
    assert report["outside_registered_lineage"] == ["m1"]


@pytest.mark.smoke
def test_unmatched_explicit_acknowledgment_refuses() -> None:
    """A mistyped per-invocation acknowledgment must surface, not pass."""

    with pytest.raises(MembershipReconciliationError) as excinfo:
        _census(acknowledged=frozenset({"typo-id"}))
    assert "typo-id" in str(excinfo.value)


@pytest.mark.smoke
def test_spent_durable_acknowledgment_never_bricks_a_later_composition() -> None:
    """A durable row whose id is no longer censused is spent history."""

    report = _census(durable_acknowledged=frozenset({"long-gone"}))
    assert report["acknowledged"] == []
    # And a durable row that DOES match a censused id discloses it.
    report = _census(heads={"m1": {}, "ghost": {}}, durable_acknowledged=frozenset({"ghost"}))
    assert report["acknowledged"] == ["ghost"]


@pytest.mark.smoke
def test_durable_acknowledgment_append_is_idempotent(tmp_path: Path) -> None:
    """One row per (campaign, stable_id); re-appends return the first row."""

    first = append_membership_acknowledgment(
        tmp_path,
        campaign_id="camp-a",
        stable_id="ghost",
        reason="orphan of the leg-29 sweep",
        acknowledged_by="tests",
        created_at="2026-08-24T00:00:00Z",
    )
    second = append_membership_acknowledgment(
        tmp_path,
        campaign_id="camp-a",
        stable_id="ghost",
        reason="a DIFFERENT reason must not mint a second row",
        acknowledged_by="tests",
        created_at="2026-08-24T01:00:00Z",
    )
    assert second == first
    ledger = membership_acknowledgment_path(tmp_path)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    rows = load_membership_acknowledgments(tmp_path, "camp-a")
    assert [row["stable_id"] for row in rows] == ["ghost"]


@pytest.mark.smoke
def test_durable_ledger_is_campaign_scoped_and_tamper_refusing(tmp_path: Path) -> None:
    """Rows never apply across campaigns; hand-edited rows refuse typed."""

    append_membership_acknowledgment(
        tmp_path,
        campaign_id="camp-a",
        stable_id="ghost",
        reason="r",
        acknowledged_by="tests",
        created_at="2026-08-24T00:00:00Z",
    )
    assert load_membership_acknowledgments(tmp_path, "camp-b") == ()
    ledger = membership_acknowledgment_path(tmp_path)
    ledger.write_text(
        ledger.read_text(encoding="utf-8").replace('"r"', '"edited-in-place"'),
        encoding="utf-8",
    )
    with pytest.raises(StratificationError, match="content address"):
        load_membership_acknowledgments(tmp_path, "camp-a")
