"""T4-T1: terminals under a PRIOR registered snapshot of the SAME campaign stay current.

``validate_currency`` pins every record to the ACTIVE snapshot by exact equality
on both snapshot-identity axes, so a rung boundary (a new content-addressed
subset snapshot on the same campaign) used to project EVERY prior terminal
dependency-stale and re-drive it -- quadratic ladder cost, and the exact reason
SG1's staging tool refuses to stage a new rung without this carve-out
(``CurrencyCarveOutMissingError`` names ``PRIOR_SNAPSHOT_TERMINAL_CURRENCY_CARVE_OUT``).

The carve-out is exactly two axes wide and lineage-gated: the recorded
``(intake_snapshot_id, intake_snapshot_sha256)`` pair must both match one
REGISTERED prior snapshot of the active snapshot's own campaign. Everything
else -- the member's own row content, every authority identity, unregistered
or cross-campaign snapshots -- keeps exact-pin semantics unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import menagerie.crawler.authority as authority_module
from menagerie.crawler.authority import validate_currency
from menagerie.crawler.rung_staging import registered_prior_intake_snapshots
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import CanonicalReducer
from menagerie.crawler.rung_staging import (
    CURRENCY_CARVE_OUT_ATTRIBUTE,
    currency_carve_out_present,
)
from menagerie.crawler.tests.conftest import (
    make_authority_context,
    make_failed_attempt,
    make_model,
)

pytestmark = pytest.mark.smoke

_OLD_ID = "intake-" + "a" * 20
_OLD_SHA = "sha256:" + "a" * 64
_NEW_ID = "intake-" + "b" * 20
_NEW_SHA = "sha256:" + "b" * 64


def _ledger_paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths."""

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _terminal_under_old_snapshot(tmp_path: Path) -> dict:
    """Append one terminal under the OLD snapshot's context and return it."""

    paths = _ledger_paths(tmp_path)
    context = make_authority_context(["m_example"], snapshot_id=_OLD_ID, snapshot_sha256=_OLD_SHA)
    attempt = make_failed_attempt()
    model = make_model(accepted=False, status_code="failed:source")
    model["status"]["attempt_ids"] = [attempt["attempt_id"]]
    model["execution"]["accepted_attempt_ids"] = []
    with CanonicalReducer(paths, context) as reducer:
        reducer.append_attempt(attempt)
        record = reducer.append_model(reducer.prepare_model(model)).record
    return dict(record)


def test_capability_attribute_is_declared_for_the_staging_tool() -> None:
    """SG1's gate reads exactly this attribute; K2 flips it with the carve-out."""

    assert getattr(authority_module, CURRENCY_CARVE_OUT_ATTRIBUTE) is True
    assert currency_carve_out_present() is True


def test_terminal_under_prior_registered_snapshot_stays_current(
    tmp_path: Path,
) -> None:
    """The intake axes carve out; every other axis keeps exact-pin semantics.

    RED before the carve-out: the same record read
    ``dependency-vector: stale intake_snapshot_id`` and left ``current_records``.
    """

    record = _terminal_under_old_snapshot(tmp_path)
    new_context = replace(
        make_authority_context(["m_example"], snapshot_id=_NEW_ID, snapshot_sha256=_NEW_SHA),
        prior_intake_snapshots={_OLD_ID: _OLD_SHA},
    )
    assert validate_currency(new_context, record) is None


def test_unregistered_or_wrong_lineage_snapshot_stays_stale(tmp_path: Path) -> None:
    """No registered lineage, wrong digest, or a foreign snapshot never carves out."""

    record = _terminal_under_old_snapshot(tmp_path)
    base = make_authority_context(["m_example"], snapshot_id=_NEW_ID, snapshot_sha256=_NEW_SHA)
    # Empty lineage: exact-pin semantics unchanged.
    assert validate_currency(base, record) == "dependency-vector: stale intake_snapshot_id"
    # Registered id with a different digest: both axes must match.
    tampered = replace(base, prior_intake_snapshots={_OLD_ID: "sha256:" + "f" * 64})
    assert validate_currency(tampered, record) == "dependency-vector: stale intake_snapshot_id"
    # A different campaign's registered snapshot proves nothing here.
    foreign = replace(base, prior_intake_snapshots={"intake-" + "c" * 20: "sha256:" + "c" * 64})
    assert validate_currency(foreign, record) == "dependency-vector: stale intake_snapshot_id"


def test_member_row_content_still_pins_exactly_under_the_carve_out(
    tmp_path: Path,
) -> None:
    """A changed intake row stays stale even when the snapshot lineage matches."""

    record = _terminal_under_old_snapshot(tmp_path)
    changed_row_context = replace(
        make_authority_context(["m_example"], snapshot_id=_NEW_ID, snapshot_sha256=_NEW_SHA),
        prior_intake_snapshots={_OLD_ID: _OLD_SHA},
        intake_by_stable_id={"m_example": {"stable_id": "m_example", "name": "renamed"}},
    )
    assert (
        validate_currency(changed_row_context, record)
        == "dependency-vector: stale intake_item_sha256"
    )


def test_active_snapshot_never_carves_out_against_itself(tmp_path: Path) -> None:
    """A record already on the active snapshot uses the exact-pin path only."""

    record = _terminal_under_old_snapshot(tmp_path)
    context = replace(
        make_authority_context(["m_example"], snapshot_id=_OLD_ID, snapshot_sha256=_OLD_SHA),
        # A hostile lineage entry for the ACTIVE id with a wrong digest must not
        # shadow the exact pin.
        prior_intake_snapshots={_OLD_ID: "sha256:" + "f" * 64},
    )
    assert validate_currency(context, record) is None


def test_registered_lineage_is_campaign_scoped_and_order_aware(tmp_path: Path) -> None:
    """The helper returns the campaign root plus strictly earlier same-campaign rungs."""

    from menagerie.crawler.tests.test_rung_staging import (
        _close_staged_rung,
        _root_binding,
        _root_snapshot,
        _stage,
        campaign_repo,  # noqa: F401 -- fixture re-export
    )

    repo_root = tmp_path / "repo"
    from menagerie.crawler.partitioner import emit_campaign_partitions
    from menagerie.crawler.identity import canonical_json_bytes
    from menagerie.crawler.tests.test_rung_staging import _ROSTER_ROWS

    records_root = repo_root / "menagerie" / "crawler" / "records"
    roster_path = tmp_path / "roster.jsonl"
    roster_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in _ROSTER_ROWS))
    emit_campaign_partitions(roster_path, records_root)
    # AW-2 fails CLOSED on an absent models ledger; the empty shard states
    # explicitly that no awards exist when the rungs stage.
    from menagerie.crawler.tests.test_rung_staging import _write_models_shard

    _write_models_shard(records_root, [])

    root_binding = _root_binding(repo_root, "c3-classics")
    root = _root_snapshot(repo_root, "c3-classics")
    ids = tuple(item.stable_id for item in root.items)
    rung_a = _stage(repo_root, "c3-classics", stable_ids=ids[:1], label="rung-a")
    # BD-1: rung-b derives from rung-a, so rung-a must hold an immutable close.
    _close_staged_rung(repo_root, rung_a, "c3-classics")
    rung_b = _stage(
        repo_root,
        "c3-classics",
        stable_ids=ids[:2],
        derived_from=rung_a.registration.snapshot_id,
        label="rung-b",
    )

    intake_root_b = records_root / "intake" / rung_b.registration.snapshot_id
    prior_for_b = registered_prior_intake_snapshots(intake_root_b, rung_b.registration.snapshot_id)
    assert prior_for_b == {
        root_binding.intake_snapshot_id: root_binding.intake_snapshot_sha256,
        rung_a.registration.snapshot_id: rung_a.registration.snapshot_sha256,
    }

    intake_root_a = records_root / "intake" / rung_a.registration.snapshot_id
    prior_for_a = registered_prior_intake_snapshots(intake_root_a, rung_a.registration.snapshot_id)
    # Rung B is LATER, so it never enters A's lineage; the root does.
    assert prior_for_a == {root_binding.intake_snapshot_id: root_binding.intake_snapshot_sha256}

    # The root snapshot itself is not rung-registered: exact-pin world, no lineage.
    root_intake = records_root / root_binding.intake_path
    assert registered_prior_intake_snapshots(root_intake, root_binding.intake_snapshot_id) == {}


def test_terminal_proof_replays_identically_under_the_next_rung(tmp_path: Path) -> None:
    """End to end: the S2 shape flips -- prior terminals keep current_records.

    A superset rung snapshot used to leave every prior terminal
    dependency-stale, re-admitted as dependency-refresh work. Under the
    carve-out the projection keeps the record current.
    """

    from menagerie.crawler.reducer import project_dependency_current

    _terminal_under_old_snapshot(tmp_path)
    new_context = replace(
        make_authority_context(["m_example"], snapshot_id=_NEW_ID, snapshot_sha256=_NEW_SHA),
        prior_intake_snapshots={_OLD_ID: _OLD_SHA},
    )
    projection = project_dependency_current(_ledger_paths(tmp_path), context=new_context)
    assert "m_example" in projection.current_records
    assert "m_example" not in projection.stale_reasons

    # Without the registered lineage, the S2 defect shape stands: the same
    # terminal projects dependency-stale on the new snapshot.
    bare_context = make_authority_context(
        ["m_example"], snapshot_id=_NEW_ID, snapshot_sha256=_NEW_SHA
    )
    bare = project_dependency_current(_ledger_paths(tmp_path), context=bare_context)
    assert "m_example" not in bare.current_records
    assert "m_example" in bare.stale_reasons
