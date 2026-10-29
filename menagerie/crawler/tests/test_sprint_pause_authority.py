"""Durable sprint-pause authority (D-62): the crawler-side HOLD check.

The orchestrator owns the sentinel; the crawler owns the check (OC-Q3). These
tests pin the three admission sites and the fail-closed core: leg start
refuses typed and generation-bound; a paid admission under a hold interrupts
in the graceful-shutdown shape (no model-failure authority) after a durable
disclosure row; and EVERY ambiguity -- unreadable bytes, format drift, field
drift, invalid timestamps, an unknown default_action -- holds exactly like a
valid active sentinel. The one state that never holds is clean absence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.driver_contracts import DriverIntegrationError, DriverShutdown
from menagerie.crawler.pause_authority import (
    SPRINT_PAUSE_FORMAT,
    evaluate_sprint_pause,
    sprint_pause_path,
)
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

NOW = "2026-08-24T12:00:00Z"


def _write_sentinel(records_root: Path, **overrides: object) -> Path:
    """Write one sentinel with valid defaults, override-per-test."""

    payload: dict[str, object] = {
        "format": SPRINT_PAUSE_FORMAT,
        "reason": "operator: quota window closed",
        "paused_by": "orchestrator",
        "created_at": "2026-08-24T11:00:00Z",
        "expires_at": None,
        "default_action": "stay-paused",
    }
    payload.update(overrides)
    path = sprint_pause_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.smoke
def test_ambiguity_fails_closed_and_absence_alone_clears(tmp_path: Path) -> None:
    """Every malformed sentinel HOLDS; only clean absence (or typed expiry) clears."""

    # Clean absence: the only free pass.
    clear = evaluate_sprint_pause(tmp_path, now=NOW)
    assert (clear.held, clear.decision) == (False, "absent")

    # Valid, expiry-less: held until the operator clears it.
    _write_sentinel(tmp_path)
    active = evaluate_sprint_pause(tmp_path, now=NOW)
    assert (active.held, active.decision) == (True, "active")
    assert active.detail == "operator: quota window closed"

    # Typed expiry honored through its closed default_action, both ways.
    _write_sentinel(tmp_path, expires_at="2026-08-24T11:30:00Z", default_action="resume")
    assert evaluate_sprint_pause(tmp_path, now=NOW).held is False
    _write_sentinel(tmp_path, expires_at="2026-08-24T11:30:00Z", default_action="stay-paused")
    assert evaluate_sprint_pause(tmp_path, now=NOW).held is True
    # A future expiry holds regardless of default_action.
    _write_sentinel(tmp_path, expires_at="2026-08-24T13:00:00Z", default_action="resume")
    assert evaluate_sprint_pause(tmp_path, now=NOW).held is True

    # The ambiguity matrix: every one of these HOLDS.
    path = sprint_pause_path(tmp_path)
    ambiguous_payloads = (
        "not json at all",
        json.dumps(["a", "list"]),
        json.dumps({"format": "menagerie.crawler.sprint-pause.v0"}),
    )
    for raw in ambiguous_payloads:
        path.write_text(raw, encoding="utf-8")
        decision = evaluate_sprint_pause(tmp_path, now=NOW)
        assert (decision.held, decision.decision) == (True, "ambiguous"), raw
    for overrides in (
        {"reason": "   "},
        {"created_at": "2026-08-24 11:00:00"},
        {"expires_at": "not-a-timestamp"},
        {"default_action": "shrug"},
        {"extra_field": True},
    ):
        _write_sentinel(tmp_path, **overrides)
        decision = evaluate_sprint_pause(tmp_path, now=NOW)
        assert (decision.held, decision.decision) == (True, "ambiguous"), overrides
    # Field SUBSET drifts too (delete one required key).
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["extra_field"], payload["paused_by"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert evaluate_sprint_pause(tmp_path, now=NOW).decision == "ambiguous"


@pytest.mark.smoke
def test_leg_start_refuses_generation_bound_under_a_hold(tmp_path: Path) -> None:
    """A launch under a standing hold refuses typed, citing the generation."""

    snapshot = _snapshot(tmp_path, count=1)
    _write_sentinel(tmp_path / "records")
    with pytest.raises(DriverIntegrationError, match="sprint-pause hold") as excinfo:
        _driver(tmp_path, snapshot).run()
    assert "generation" in str(excinfo.value)

    # The expired-resume sentinel releases the same launch, typed.
    _write_sentinel(
        tmp_path / "records",
        expires_at="2020-01-01T00:00:00Z",
        default_action="resume",
    )
    assert _driver(tmp_path, snapshot).run().status == "complete"


@pytest.mark.smoke
def test_paid_admission_interrupts_typed_after_a_durable_disclosure(
    tmp_path: Path,
) -> None:
    """A hold arriving mid-leg stops the NEXT buy in the shutdown shape."""

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    _write_sentinel(tmp_path / "records")
    ledger_path = tmp_path / "operational-probe.jsonl"
    with JsonlLedger(ledger_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        with pytest.raises(DriverShutdown) as excinfo:
            driver._refuse_paid_admission_under_sprint_pause(  # noqa: SLF001
                ledger, "author-admission"
            )
    assert excinfo.value.fact.admission_boundary == "author-admission"
    rows = [
        event
        for event in scan_jsonl(ledger_path)
        if event.get("details", {}).get("disposition") == "sprint-pause-hold"
    ]
    assert len(rows) == 1
    assert rows[0]["details"]["decision"] == "active"
    del paths