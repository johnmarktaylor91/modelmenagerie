"""BD-1: a park a landed revision consumed is voided in the LEDGER, not just derived.

T2-opus-07: all five ``review-held:terminal-recording-refused`` rows from the
step-100 recording outage were still unvoided at close although every one of
their models had recorded a terminal hours later (m10517 09:12:51Z -> head
12:54:06Z, and four siblings). No void row and no ``park-cleared`` event existed,
so the park ledger asserted five review-held claims that were not true, and the
inventory an operator triages at each boundary grew monotonically (11 -> 17 rows
during the finishing legs).

Head-keyed release already makes such a row inert for every derived census. This
disclosure keeps the ledger itself honest. The pins below hold both directions:
a consumed row IS voided, a LIVE decision is NEVER voided, and the disclosure is
idempotent and never blocks a genuinely new refusal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.driver import DriverPaths
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.park_release import PARK_RELEASE_VOID_ACTOR, void_released_parks
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_VOID_SCHEMA_VERSION,
    append_refusal_park_row,
    build_refusal_park_row,
    load_refusal_park_rows,
    refusal_park_path,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.state import _select_current
from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot


def _park(stable_id: str, *, blocked_parent_revision: Optional[str] = None) -> Any:
    """Build one terminal-refusal park row for a fixture model."""

    return build_refusal_park_row(
        stable_id=stable_id,
        work_id="w-earlier",
        name="Example",
        campaign_id="c1-mech",
        run_id="run-earlier",
        machine_id="host-1",
        created_at="2026-08-13T09:12:51Z",
        refusal_class="ReductionError",
        record_error="model environment generation is absent from active context",
        record_error_type="ReductionError",
        fallback_error_type=None,
        fallback_error=None,
        intended_status_code="failed:author",
        intended_reason_code=None,
        blocked_parent_revision=blocked_parent_revision,
        candidate_sha256=None,
    )


@pytest.mark.smoke
def test_a_consumed_park_is_voided_under_the_machine_actor(tmp_path: Path) -> None:
    """The row a landed revision consumed leaves the live inventory durably."""

    records_root = tmp_path / "records"
    row = _park("m1")
    append_refusal_park_row(refusal_park_path(records_root), row)
    assert len(load_refusal_park_rows([refusal_park_path(records_root)])) == 1

    voided = void_released_parks(
        records_root,
        {"m1": [row]},
        {"m1": {"stable_id": "m1", "record_revision": "sha256:landed"}},
        created_at="2026-08-16T00:00:00Z",
    )
    assert len(voided) == 1
    assert voided[0]["schema_version"] == REFUSAL_PARK_VOID_SCHEMA_VERSION
    assert voided[0]["void_of"] == row["row_sha256"]
    # A disclosure, never an operator decision: the actor says which it is.
    assert voided[0]["voided_by"] == PARK_RELEASE_VOID_ACTOR
    assert "sha256:landed" in voided[0]["reason"]
    assert load_refusal_park_rows([refusal_park_path(records_root)]) == ()


@pytest.mark.smoke
def test_a_live_decision_is_never_voided(tmp_path: Path) -> None:
    """A park coexisting with the exact revision it recorded stays LIVE."""

    records_root = tmp_path / "records"
    row = _park("m1", blocked_parent_revision="sha256:blocked")
    append_refusal_park_row(refusal_park_path(records_root), row)

    assert (
        void_released_parks(
            records_root,
            {"m1": [row]},
            # The head is still the very revision the refusal recorded: the
            # decision is owed, and nothing here may release it.
            {"m1": {"stable_id": "m1", "record_revision": "sha256:blocked"}},
            created_at="2026-08-16T00:00:00Z",
        )
        == ()
    )
    # A model with NO head at all is likewise untouched.
    assert (
        void_released_parks(records_root, {"m1": [row]}, {}, created_at="2026-08-16T00:00:00Z")
        == ()
    )
    assert len(load_refusal_park_rows([refusal_park_path(records_root)])) == 1


@pytest.mark.smoke
def test_the_disclosure_is_idempotent_and_scope_bounded(tmp_path: Path) -> None:
    """Re-closing appends nothing new, and out-of-scope models are untouched."""

    records_root = tmp_path / "records"
    first = _park("m1")
    second = _park("m2")
    append_refusal_park_row(refusal_park_path(records_root), first)
    append_refusal_park_row(refusal_park_path(records_root), second)
    heads = {
        "m1": {"stable_id": "m1", "record_revision": "sha256:landed-1"},
        "m2": {"stable_id": "m2", "record_revision": "sha256:landed-2"},
    }
    persisted = {"m1": [first], "m2": [second]}

    once = void_released_parks(
        records_root, persisted, heads, created_at="2026-08-16T00:00:00Z", scoped_ids=("m1",)
    )
    again = void_released_parks(
        records_root, persisted, heads, created_at="2026-08-16T01:00:00Z", scoped_ids=("m1",)
    )
    assert [row["void_of"] for row in once] == [first["row_sha256"]]
    assert [row["void_of"] for row in again] == [first["row_sha256"]]
    assert again[0]["created_at"] == "2026-08-16T00:00:00Z"
    ledger_rows = scan_jsonl(refusal_park_path(records_root), validate=False)
    assert sum(1 for row in ledger_rows if row.get("void_of")) == 1
    # m2 was out of scope for this close, so its park is still live accounting.
    assert [
        row["stable_id"] for row in load_refusal_park_rows([refusal_park_path(records_root)])
    ] == ["m2"]


@pytest.mark.smoke
def test_a_driver_close_discloses_the_parks_its_records_consumed(tmp_path: Path) -> None:
    """End to end, in the step-100 shape: park written, model recorded, close voids."""

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    records_root = tmp_path / "records"
    paths = DriverPaths(
        tmp_path / "runtime",
        snapshot.root,
        LedgerPaths(
            records_root / "models" / "current-shard.jsonl",
            records_root / "attempts" / "local.jsonl",
            records_root / "gates" / "current-shard.jsonl",
        ),
    )
    assert _driver(tmp_path, snapshot, paths_override=paths).run().status == "complete"
    parked = sorted(item.stable_id for item in snapshot.items)[0]
    append_refusal_park_row(refusal_park_path(records_root), _park(parked))
    assert len(load_refusal_park_rows([refusal_park_path(records_root)])) == 1

    # The next close observes a head that consumed the refusal.
    result = _driver(tmp_path, snapshot, paths_override=paths).run()
    assert result.status == "complete"
    assert load_refusal_park_rows([refusal_park_path(records_root)]) == ()
    voids = [
        row
        for row in scan_jsonl(refusal_park_path(records_root), validate=False)
        if row.get("void_of")
    ]
    assert len(voids) == 1
    assert voids[0]["voided_by"] == PARK_RELEASE_VOID_ACTOR
    heads = _select_current(scan_jsonl(paths.ledgers.models))
    assert heads[parked]["record_revision"] in voids[0]["reason"]
