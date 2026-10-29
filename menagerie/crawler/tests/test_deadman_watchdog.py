"""D-58: dead-man pair semantics -- expiry-bound holds, beat, boot survival.

The 5h46m 200a outage ran inside an expiry-less operator hold. The sentinel
matrix pins the closed shape: live holds suppress, expired holds execute
their closed default_action, and anything expiry-less or malformed pages
immediately. The beat check is the off-host half: a dead crawl host is
detected by hardware it cannot take down.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from menagerie.crawler.tools.deadman_watchdog import (
    CHECK_VERDICTS,
    HOLD_DEFAULT_ACTIONS,
    check_beat,
    evaluate_check,
    read_hold_sentinel,
    render_launchd_plist,
    render_remote_cron,
    write_beat,
)
from menagerie.crawler.tools.launch_monitor import write_heartbeat

pytestmark = pytest.mark.smoke

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)


def _hold(path: Path, **payload) -> Path:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


class TestHoldSentinel:
    def test_absent_is_none(self, tmp_path: Path) -> None:
        assert read_hold_sentinel(tmp_path / "hold.json", now=NOW).state == "none"

    def test_live_hold_is_active(self, tmp_path: Path) -> None:
        hold = _hold(
            tmp_path / "hold.json",
            reason="storm",
            expires_at=(NOW + timedelta(hours=1)).isoformat(),
            default_action="page",
        )
        assert read_hold_sentinel(hold, now=NOW).state == "active"

    def test_expired_hold_executes_its_default(self, tmp_path: Path) -> None:
        base = dict(expires_at=(NOW - timedelta(minutes=1)).isoformat())
        page = _hold(tmp_path / "p.json", default_action="page", **base)
        assert read_hold_sentinel(page, now=NOW).state == "expired-page"
        release = _hold(tmp_path / "r.json", default_action="release", **base)
        assert read_hold_sentinel(release, now=NOW).state == "expired-release"

    def test_expiry_less_hold_is_invalid(self, tmp_path: Path) -> None:
        hold = _hold(tmp_path / "hold.json", reason="storm", default_action="page")
        disposition = read_hold_sentinel(hold, now=NOW)
        assert disposition.state == "invalid"
        assert "expiry-less" in disposition.detail

    def test_off_vocabulary_action_is_invalid(self, tmp_path: Path) -> None:
        hold = _hold(
            tmp_path / "hold.json",
            expires_at=(NOW + timedelta(hours=1)).isoformat(),
            default_action="snooze",
        )
        assert read_hold_sentinel(hold, now=NOW).state == "invalid"

    def test_action_vocabulary_is_closed(self) -> None:
        assert HOLD_DEFAULT_ACTIONS == {"page", "release"}
        assert set(CHECK_VERDICTS) == {"healthy", "holding", "page", "invalid-hold"}


def _campaign(tmp_path: Path) -> tuple[Path, Path, Path]:
    records = tmp_path / "records"
    runtime = tmp_path / ".crawl-local"
    state_file = tmp_path / "monitor" / "state.json"
    for rel, rows in {
        "intake/registered-rungs.jsonl": [{"label": "s", "snapshot_id": "snap"}],
        "intake/snap/items.jsonl": [{"stable_id": "m1"}],
        "models/current-shard.jsonl": [
            {
                "stable_id": "m1",
                "record_seq": 1,
                "status": {"code": "runs", "kind": "runs"},
                "created_at": NOW.isoformat(),
            }
        ],
    }.items():
        path = records / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return records, runtime, state_file


class TestOnHostCheck:
    def test_fresh_heartbeat_and_complete_campaign_is_healthy(
        self, tmp_path: Path
    ) -> None:
        records, runtime, state_file = _campaign(tmp_path)
        write_heartbeat(
            state_file,
            tick_ok=True,
            reason=None,
            consecutive_failures=0,
            tick_counter=1,
            tick_duration_seconds=0.1,
            generation=1,
            poll_seconds=60,
            now=NOW,
        )
        verdict, evidence = evaluate_check(
            records_root=records,
            runtime_root=runtime,
            state_file=state_file,
            hold_path=runtime / "deadman-hold.json",
            heartbeat_max_age_seconds=900,
            now=NOW,
        )
        assert verdict == "healthy", evidence

    def test_stale_heartbeat_pages(self, tmp_path: Path) -> None:
        records, runtime, state_file = _campaign(tmp_path)
        write_heartbeat(
            state_file,
            tick_ok=True,
            reason=None,
            consecutive_failures=0,
            tick_counter=1,
            tick_duration_seconds=0.1,
            generation=1,
            poll_seconds=60,
            now=NOW - timedelta(hours=2),
        )
        verdict, _evidence = evaluate_check(
            records_root=records,
            runtime_root=runtime,
            state_file=state_file,
            hold_path=runtime / "deadman-hold.json",
            heartbeat_max_age_seconds=900,
            now=NOW,
        )
        assert verdict == "page"

    def test_live_hold_suppresses_and_expired_hold_pages(self, tmp_path: Path) -> None:
        records, runtime, state_file = _campaign(tmp_path)
        runtime.mkdir(parents=True, exist_ok=True)
        hold = runtime / "deadman-hold.json"
        _hold(
            hold,
            expires_at=(NOW + timedelta(hours=1)).isoformat(),
            default_action="page",
        )
        verdict, _ = evaluate_check(
            records_root=records,
            runtime_root=runtime,
            state_file=state_file,
            hold_path=hold,
            heartbeat_max_age_seconds=900,
            now=NOW,
        )
        assert verdict == "holding"
        _hold(
            hold,
            expires_at=(NOW - timedelta(minutes=5)).isoformat(),
            default_action="page",
        )
        verdict, _ = evaluate_check(
            records_root=records,
            runtime_root=runtime,
            state_file=state_file,
            hold_path=hold,
            heartbeat_max_age_seconds=900,
            now=NOW,
        )
        assert verdict == "page"

    def test_invalid_hold_pages_immediately(self, tmp_path: Path) -> None:
        records, runtime, state_file = _campaign(tmp_path)
        runtime.mkdir(parents=True, exist_ok=True)
        hold = _hold(runtime / "deadman-hold.json", default_action="page")
        verdict, _ = evaluate_check(
            records_root=records,
            runtime_root=runtime,
            state_file=state_file,
            hold_path=hold,
            heartbeat_max_age_seconds=900,
            now=NOW,
        )
        assert verdict == "invalid-hold"

    def test_incomplete_campaign_with_no_driver_pages(self, tmp_path: Path) -> None:
        records, runtime, state_file = _campaign(tmp_path)
        items = records / "intake" / "snap" / "items.jsonl"
        items.write_text(
            json.dumps({"stable_id": "m1"}) + "\n" + json.dumps({"stable_id": "m2"}) + "\n",
            encoding="utf-8",
        )
        write_heartbeat(
            state_file,
            tick_ok=True,
            reason=None,
            consecutive_failures=0,
            tick_counter=1,
            tick_duration_seconds=0.1,
            generation=1,
            poll_seconds=60,
            now=NOW,
        )
        verdict, evidence = evaluate_check(
            records_root=records,
            runtime_root=runtime,
            state_file=state_file,
            hold_path=runtime / "deadman-hold.json",
            heartbeat_max_age_seconds=900,
            now=NOW,
        )
        assert verdict == "page", evidence


class TestBeat:
    def test_beat_round_trip_and_staleness(self, tmp_path: Path) -> None:
        beat = tmp_path / "beat.json"
        assert check_beat(beat, max_age_seconds=900, now=NOW)[0] == "missing"
        write_beat(beat, verdict="healthy", now=NOW)
        assert check_beat(beat, max_age_seconds=900, now=NOW)[0] == "fresh"
        later = NOW + timedelta(seconds=901)
        assert check_beat(beat, max_age_seconds=900, now=later)[0] == "stale"
        beat.write_text("garbage", encoding="utf-8")
        assert check_beat(beat, max_age_seconds=900, now=NOW)[0] == "unreadable"


class TestEmitters:
    def test_launchd_plist_boots_and_polls(self, tmp_path: Path) -> None:
        import plistlib

        args = argparse.Namespace(
            records_root=tmp_path / "records",
            runtime_root=tmp_path / ".crawl-local",
            beat=tmp_path / "beat.json",
            interval=300.0,
        )
        payload = plistlib.loads(render_launchd_plist(args).encode("utf-8"))
        assert payload["Label"] == "com.menagerie.deadman-watchdog"
        assert payload["RunAtLoad"] is True
        assert payload["StartInterval"] == 300
        assert "--check" in payload["ProgramArguments"]
        assert "--send" in payload["ProgramArguments"]

    def test_remote_cron_line_names_the_beat(self, tmp_path: Path) -> None:
        args = argparse.Namespace(beat=tmp_path / "beat.json", interval=300.0)
        line = render_remote_cron(args)
        assert "--check-beat" in line
        assert str(tmp_path / "beat.json") in line
        assert "--max-age-seconds 900" in line
