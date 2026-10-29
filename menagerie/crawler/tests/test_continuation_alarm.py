"""LC-11 (D-23): the generation-span + continuation-budget alarms, read-side.

RULING W7 clause 14: the LC-11 alarm ships regardless of the deferred
segments design. Today one generation's window spans the whole rung, so a
generation open past the leg cadence, a restart loop joining one generation
again and again, or a member whose continuation budget is exhausted (OPEN-5:
HELD, operator-owed) must be legible in-flight -- not at rung end. Pure
derivations; an alarm can never improve, settle, or license anything.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from menagerie.crawler.continuation_alarm import (
    continuation_budget_alarm,
    generation_span_alarm,
)
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.recordio import operational_events_path
from menagerie.crawler.scheduler_parks import (
    SCHEDULER_QUARANTINE_PARK_EVENT_DISPOSITION,
    append_scheduler_quarantine_row,
    build_scheduler_quarantine_row,
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
from menagerie.crawler.tests.test_close_v2_contract import (
    CAMPAIGN,
    _open_leg,
    _v2_support,
)

NOW = datetime(2026, 8, 25, 12, 0, 0, tzinfo=timezone.utc)


def _staged_open_leg(tmp_path: Path):
    repo = build_repo(tmp_path)
    records_root = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    generation = _open_leg(repo, staged)
    return repo, records_root, ids, staged, generation


@pytest.mark.smoke
def test_generation_span_alarms_past_the_threshold_and_quiets_on_close(
    tmp_path: Path,
) -> None:
    """An open generation ages into the alarm; its close retires it."""

    repo, records_root, ids, staged, generation = _staged_open_leg(tmp_path)
    report = generation_span_alarm(records_root, now=NOW)
    opened = datetime.fromisoformat(
        str(report["open_generations"][0]["opened_at"]).replace("Z", "+00:00")
    )

    fresh = generation_span_alarm(records_root, now=opened + timedelta(hours=1))
    assert fresh["status"] == "quiet"
    assert fresh["open_generations"][0]["generation_id"] == generation.generation_id

    aged = generation_span_alarm(records_root, now=opened + timedelta(hours=100))
    assert aged["status"] == "alarm"
    assert aged["alarms"][0]["generation_id"] == generation.generation_id
    assert any("span threshold" in reason for reason in aged["alarms"][0]["reasons"])

    seed_heads(records_root, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _v2_support().record_fixture_close(
        records_root,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_ids=members_of(repo, staged.registration.snapshot_id),
    )
    closed = generation_span_alarm(records_root, now=opened + timedelta(hours=100))
    assert closed["status"] == "quiet"
    assert closed["open_generations"] == []


@pytest.mark.smoke
def test_a_restart_loop_joining_one_generation_alarms_on_runs(tmp_path: Path) -> None:
    """Seven joined invocations on one open generation read as a loop."""

    _repo, records_root, _ids, staged, _generation = _staged_open_leg(tmp_path)
    for index in range(7):
        _open_leg(_repo, staged, run_id=f"pytest-resume-{index}")
    report = generation_span_alarm(
        records_root,
        now=datetime.fromisoformat(
            str(
                generation_span_alarm(records_root, now=NOW)["open_generations"][0][
                    "opened_at"
                ]
            ).replace("Z", "+00:00")
        )
        + timedelta(hours=1),
    )
    assert report["status"] == "alarm"
    assert any("run threshold" in reason for reason in report["alarms"][0]["reasons"])


@pytest.mark.smoke
def test_continuation_budget_alarm_flags_held_members_only(tmp_path: Path) -> None:
    """OPEN-5's arm: charged >= limit is HELD debt; below the limit is quiet.

    Same budget derivation the close's continuation disclosure uses -- one
    implementation; the alarm just makes the debt legible in-flight.
    """

    records_root = tmp_path / "records"
    quarantine = scheduler_quarantine_path(records_root)
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    row = build_scheduler_quarantine_row(
        stable_id="m_looping",
        work_id="w-loop",
        name="m_looping",
        campaign_id="c1-mech",
        run_id="pytest-run",
        machine_id="pytest-host",
        created_at="2026-08-25T00:00:00Z",
        site="scheduler-round",
        reason_code="no-durable-routing-progress",
        fault_evidence="fixture scheduler fault",
    )
    append_scheduler_quarantine_row(quarantine, row)
    events_path = operational_events_path(records_root)
    events_path.parent.mkdir(parents=True, exist_ok=True)

    def charge(created_at: str) -> None:
        event = {
            "created_at": created_at,
            "details": {
                "disposition": SCHEDULER_QUARANTINE_PARK_EVENT_DISPOSITION,
                "stable_id": "m_looping",
                "work_id": "w-loop",
            },
        }
        with events_path.open("ab") as handle:
            handle.write(canonical_json_bytes(event) + b"\n")

    charge("2026-08-25T00:00:01Z")
    below = continuation_budget_alarm(records_root)
    assert below["status"] == "quiet"
    assert below["budgets"][0]["charged_parks"] == 1
    assert below["budgets"][0]["held"] is False

    charge("2026-08-25T00:00:02Z")
    held = continuation_budget_alarm(records_root)
    assert held["status"] == "alarm"
    assert held["alarms"][0]["stable_id"] == "m_looping"
    assert held["alarms"][0]["held"] is True
    assert "buys nothing" in held["alarms"][0]["reasons"][0]


@pytest.mark.smoke
def test_the_alarm_tool_exits_two_on_breach_and_zero_quiet(tmp_path: Path, capsys) -> None:
    """The monitor-facing surface keys on exit codes, like the DX-24 probe."""

    from menagerie.crawler.tools.continuation_alarm import main as alarm_main

    records_root = tmp_path / "records"
    records_root.mkdir(parents=True)
    assert alarm_main(["--records-root", str(records_root)]) == 0
    aged_root = tmp_path / "aged"
    aged_root.mkdir(parents=True)
    _repo, records_root2, _ids, _staged, _generation = _staged_open_leg(aged_root)
    capsys.readouterr()
    assert (
        alarm_main(["--records-root", str(records_root2), "--max-span-hours", "0"]) == 2
    )
    err = capsys.readouterr().err
    assert "LC-11 alarm" in err
