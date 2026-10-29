"""The operator quarantine-clearance receipt tool.

The designed release signal for an out-of-band repair is one campaign-health
event naming the exact quarantine event. These tests bind the tool to the REAL
consumer -- :func:`menagerie.crawler.driver_admission._quarantine_is_cleared` --
so a receipt the tool writes is provably one the admission valve honors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.checkpoint import append_canonical_operational_event
from menagerie.crawler.driver_admission import (
    _quarantine_is_cleared,
    environment_clearance_events,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tools.clear_environment_quarantine import main

QUARANTINE_EVENT_ID = "environment-setup-failure-ced76d4e775cee9ae659bda8"


def _seed_repo(tmp_path: Path) -> Path:
    """Create a minimal campaign tree with one quarantine and one plain event."""

    repo = tmp_path / "campaign"
    operational = repo / "menagerie" / "crawler" / "records" / "operational"
    operational.mkdir(parents=True)
    (repo / ".crawl-local" / "locks").mkdir(parents=True)
    quarantine = {
        "schema_version": "menagerie.crawler.operational-event.v1",
        "created_at": "2026-08-12T07:52:50.098266Z",
        "event_kind": "campaign-health",
        "status": "failed:runner",
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 6},
        "current_environment": "core",
        "run_id": "crawler-run",
        "machine_id": "worker-a",
        "details": {
            "disposition": "environment-setup-failure-parked",
            "intent": "core",
            "target": "osx-arm64",
            "failure_class": "environment-authority",
        },
        "event_id": QUARANTINE_EVENT_ID,
    }
    healthy = {
        "schema_version": "menagerie.crawler.operational-event.v1",
        "created_at": "2026-08-12T07:53:00.000000Z",
        "event_kind": "campaign-health",
        "status": "healthy",
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {},
        "current_environment": None,
        "run_id": "crawler-run",
        "machine_id": "worker-a",
        "details": {"disposition": "host-capacity-deferred"},
        "event_id": "host-capacity-observed-0abc0abc0abc0abc0abc0abc",
    }
    second_quarantine = dict(
        quarantine,
        created_at="2026-08-07T12:29:00.000000Z",
        details={"disposition": "environment-cleanup-quarantined", "intent": "core"},
        event_id="environment-cleanup-467a0c5a16bc6aa949d60c68",
    )
    ledger = operational / "events.jsonl"
    append_canonical_operational_event(ledger, quarantine)
    append_canonical_operational_event(ledger, healthy)
    append_canonical_operational_event(ledger, second_quarantine)
    return repo


@pytest.mark.smoke
def test_clearance_receipt_is_honored_by_the_admission_predicate(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path)
    rc = main(
        [
            "environment-setup-failure-ced76d4e",
            "--evidence",
            "seal fix fe994240; leg-4 same-identity build+seal+run+teardown",
            "--repo-root",
            str(repo),
        ]
    )
    assert rc == 0
    records_root = repo / "menagerie" / "crawler" / "records"
    canonical_events = scan_jsonl(records_root / "operational" / "events.jsonl", validate=False)
    # BL-9c: the receipt lands in the operator SIDECAR (recordable while a
    # leg runs), never in the canonical ledger; the admission predicate
    # consumes the merged view.
    receipts = environment_clearance_events(records_root)
    assert not _quarantine_is_cleared(canonical_events, QUARANTINE_EVENT_ID)
    assert _quarantine_is_cleared((*canonical_events, *receipts), QUARANTINE_EVENT_ID)
    receipt = receipts[-1]
    assert receipt["details"]["quarantine_event_id"] == QUARANTINE_EVENT_ID
    assert receipt["details"]["intent"] == "core"
    assert receipt["event_id"].startswith("campaign-health-")


@pytest.mark.smoke
def test_double_clearance_is_refused(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path)
    assert main([QUARANTINE_EVENT_ID, "--evidence", "repair", "--repo-root", str(repo)]) == 0
    assert main([QUARANTINE_EVENT_ID, "--evidence", "repair", "--repo-root", str(repo)]) == 1


@pytest.mark.smoke
def test_non_quarantine_event_is_refused(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path)
    rc = main(["host-capacity-observed-0abc", "--evidence", "repair", "--repo-root", str(repo)])
    assert rc == 1
    events = scan_jsonl(
        repo / "menagerie" / "crawler" / "records" / "operational" / "events.jsonl",
        validate=False,
    )
    assert not any(
        (event.get("details") or {}).get("disposition") == "environment-quarantine-cleared"
        for event in events
    )
    assert environment_clearance_events(repo / "menagerie" / "crawler" / "records") == ()


@pytest.mark.smoke
def test_ambiguous_prefix_is_refused(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path)
    assert main(["environment", "--evidence", "repair", "--repo-root", str(repo)]) == 1


@pytest.mark.smoke
def test_dry_run_appends_nothing(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path)
    rc = main([QUARANTINE_EVENT_ID, "--evidence", "repair", "--repo-root", str(repo), "--dry-run"])
    assert rc == 0
    events = scan_jsonl(
        repo / "menagerie" / "crawler" / "records" / "operational" / "events.jsonl",
        validate=False,
    )
    assert len(events) == 3
    assert environment_clearance_events(repo / "menagerie" / "crawler" / "records") == ()
    assert not _quarantine_is_cleared(events, QUARANTINE_EVENT_ID)
