"""D-49 (WS7 close half): operator rows are authenticated in time at close.

200a shipped 2 operator voids dated in the FUTURE and 0/208 grants carried
any timestamp -- the audit trail was fabricable and nothing refused. These
tests pin the shared authentication mechanism and the close-side sweep:

* one shared :func:`assert_operator_timestamp` (strict UTC parse, non-future)
  for every write site and the close;
* a leg close REFUSES any row appended in ITS OWN window to a
  withholding/park ledger (park rows and the operator voids that share those
  ledgers) whose ``created_at`` postdates the close instant;
* era honesty (PD-13, forward-only corrections): rows below the window floor
  are prior-rung history -- the two known 200a future-dated voids are
  corrected by APPENDED notes, never re-refused by later legs -- and rows
  without any ``created_at`` are tolerated in every era (grants gain
  ``created_at`` only in their schema-v2 wave).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from menagerie.crawler import driver as driver_module
from menagerie.crawler.operator_provenance import (
    OperatorProvenanceError,
    assert_operator_timestamp,
    operator_identity_census,
    parse_operator_timestamp,
    post_dated_window_rows,
)
from menagerie.crawler.scheduler_parks import (
    append_scheduler_quarantine_row,
    append_scheduler_quarantine_void,
    build_scheduler_quarantine_row,
    build_scheduler_quarantine_void,
    scheduler_quarantine_path,
)
from menagerie.crawler.tests.rt3_support import (
    build_repo,
    campaign_member_ids,
    members_of,
    records_root_of,
    seed_heads,
    stage,
)

pytestmark = pytest.mark.smoke

CAMPAIGN = "c3-classics"

#: The fixture close instant record_fixture_close stamps by default.
_CLOSED_AT = "2026-08-11T00:00:00Z"


def _v2_support():
    from menagerie.crawler.tests import close_object_support as support

    return support


def _open_leg(repo: Path, staged):
    support = _v2_support()
    return support.open_fixture_leg(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
    )


def _append_quarantine(records_root: Path, stable_id: str, *, created_at: str):
    row = build_scheduler_quarantine_row(
        stable_id=stable_id,
        work_id=f"w-{stable_id}",
        name=stable_id,
        campaign_id=CAMPAIGN,
        run_id="pytest-close",
        machine_id="pytest-host",
        created_at=created_at,
        site="scheduler-round",
        reason_code="no-durable-routing-progress",
        fault_evidence="fixture scheduler fault",
    )
    return append_scheduler_quarantine_row(scheduler_quarantine_path(records_root), row)


def _void(records_root: Path, row, *, created_at: str) -> None:
    append_scheduler_quarantine_void(
        scheduler_quarantine_path(records_root),
        build_scheduler_quarantine_void(
            void_of=str(row["row_sha256"]),
            reason="fixture drain",
            voided_by="pytest-operator",
            created_at=created_at,
        ),
    )


def _close(repo: Path, staged):
    support = _v2_support()
    return support.record_fixture_close(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_ids=members_of(repo, staged.registration.snapshot_id),
        completion_status="complete-with-deferrals",
        closed_at=_CLOSED_AT,
    )


def test_the_shared_timestamp_authority_refuses_future_and_garbage() -> None:
    """One parse + one non-future rule, shared by write sites and the close."""

    parsed = assert_operator_timestamp(
        "2026-08-01T00:00:00Z", as_of=_CLOSED_AT, source="fixture"
    )
    assert parsed.tzinfo is not None
    # Equality is legitimate under one coarse clock.
    assert_operator_timestamp(_CLOSED_AT, as_of=_CLOSED_AT, source="fixture")
    with pytest.raises(OperatorProvenanceError) as post_dated:
        assert_operator_timestamp("2027-01-01T00:00:00Z", as_of=_CLOSED_AT, source="void")
    assert "post-dated" in str(post_dated.value)
    for garbage in (None, "", "not-a-time", 12345):
        with pytest.raises(OperatorProvenanceError):
            parse_operator_timestamp(garbage, source="void")


def test_a_post_dated_in_window_void_refuses_the_close(tmp_path: Path) -> None:
    """The 200a shape, refused at the source: a future-dated operator void
    appended during the leg refuses the close with the row named."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    row = _append_quarantine(rr, member, created_at="2026-08-01T00:00:10Z")
    _void(rr, row, created_at="2027-01-01T00:00:00Z")  # fabricated future date

    with pytest.raises(OperatorProvenanceError) as excinfo:
        _close(repo, staged)
    assert "post-dated" in str(excinfo.value)
    assert "valve.scheduler-quarantine" in str(excinfo.value)


def test_a_post_dated_in_window_park_row_refuses_the_close(tmp_path: Path) -> None:
    """Not only voids: any counted withholding row fabricating time refuses."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _append_quarantine(rr, ids[1], created_at="2027-06-01T00:00:00Z")

    with pytest.raises(OperatorProvenanceError) as excinfo:
        _close(repo, staged)
    assert "post-dated" in str(excinfo.value)


def test_history_below_the_floor_is_never_re_refused(tmp_path: Path) -> None:
    """Era honesty: a prior-rung future-dated row (the two known 200a voids)
    is corrected forward-only and must not brick every later close."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    # The debris lands BEFORE the leg opens: below this leg's window floor.
    row = _append_quarantine(rr, member, created_at="2026-08-01T00:00:10Z")
    _void(rr, row, created_at="2027-01-01T00:00:00Z")

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)

    close = _close(repo, staged)
    assert member in close["partition"]["adjudicated_here"]


def test_rows_without_created_at_are_tolerated_in_every_era(tmp_path: Path) -> None:
    """0/208 grants carry timestamps today; absence never refuses (v2 adds it)."""

    findings = post_dated_window_rows(
        records_root_of(build_repo(tmp_path)),
        baselines={},
        as_of=_CLOSED_AT,
    )
    assert findings == ()


def test_the_driver_close_path_runs_the_sweep() -> None:
    """Source binding: the production close cannot silently drop the sweep."""

    source = inspect.getsource(driver_module.CrawlerDriver._run_locked)
    assert "post_dated_window_rows(" in source
    # The fixture close raises exactly like the driver (parity contract).
    support_source = inspect.getsource(_v2_support().record_fixture_close)
    assert "post_dated_window_rows(" in support_source


def test_identity_census_discloses_spelling_drift_verbatim() -> None:
    """D-53 seam: per-identity counts are DERIVED and never normalized."""

    census = operator_identity_census(
        [
            {"voided_by": "jmt"},
            {"voided_by": "JMT"},
            {"voided_by": "jmt"},
            {"registered_by": "pytest-force-register"},
            {"unrelated": 1},
            "not-a-row",
        ]
    )
    assert census == {
        "registered_by": {"pytest-force-register": 1},
        "voided_by": {"JMT": 1, "jmt": 2},
    }
