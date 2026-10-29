"""D-33 + UU-7: the launch monitor survives anything and proves its liveness.

The 200a monitor's tick was wrapped in ``except (OSError, ValueError)`` only:
any unexpected ledger row shape killed the loop, launchd did not own it, and
the absence of texts read as a quiet healthy campaign (LP NN-5, open since
before the 200a launch). These tests pin the survival contract: the loop body
outlives ANY tick exception, writes its heartbeat every iteration (success or
failure), escalates a typed monitor-degraded page after three consecutive
failures while continuing to tick, refuses to run as a second instance, seeds
its restart baselines instead of fabricating a block-tail alarm (U-O6), and
treats an unreadable-but-present ledger as typed unavailable evidence rather
than a healthy zero.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from menagerie.crawler.tools import launch_monitor
from menagerie.crawler.tools.launch_monitor import (
    DEGRADED_ESCALATION_TICKS,
    MonitorEvidenceUnavailable,
    MonitorState,
    Snapshot,
    Thresholds,
    acquire_monitor_lock,
    build_snapshot,
    check_heartbeat,
    evaluate_alerts,
    heartbeat_path,
    read_jsonl_tolerant,
    render_monitor_launchd_plist,
    render_watchdog_launchd_plist,
    run_loop,
    write_heartbeat,
)

pytestmark = pytest.mark.smoke


def _write_ledger(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _records_root(tmp_path: Path) -> Path:
    root = tmp_path / "records"
    _write_ledger(
        root / "intake" / "registered-rungs.jsonl",
        [{"label": "step-200", "snapshot_id": "snap-1"}],
    )
    _write_ledger(
        root / "intake" / "snap-1" / "items.jsonl",
        [{"stable_id": "m1"}, {"stable_id": "m2"}],
    )
    _write_ledger(
        root / "models" / "current-shard.jsonl",
        [
            {
                "stable_id": "m1",
                "record_seq": 1,
                "status": {"code": "runs", "kind": "runs"},
                "created_at": "2026-08-24T00:00:00+00:00",
            }
        ],
    )
    return root


def _args(tmp_path: Path, records_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        records_root=records_root,
        rung_label=None,
        rolling_window=20,
        error_window=20,
        cas_root=None,
        outbox=None,
        state_file=tmp_path / "state" / "state-test.json",
        poll=60.0,
        interval=3600.0,
        max_age_seconds=None,
    )


class TestMalformedRowSurvival:
    def test_snapshot_survives_a_poisoned_row_in_every_ledger(
        self, tmp_path: Path
    ) -> None:
        root = _records_root(tmp_path)
        poison = {
            "stable_id": 17,
            "record_seq": "not-an-int",
            "status": "not-a-mapping",
            "created_at": 12,
        }
        for relative in (
            "models/current-shard.jsonl",
            "attempts/local.jsonl",
            "gates/current-shard.jsonl",
            "operational/events.jsonl",
            *launch_monitor.PARK_LEDGER_RELATIVE_PATHS,
        ):
            path = root / relative
            existing, _ = (
                read_jsonl_tolerant(path) if path.exists() else ([], 0)
            )
            _write_ledger(path, existing + [poison])
        snapshot = build_snapshot(root, check_disk=False)
        assert snapshot.roster_total == 2
        assert snapshot.done == 1


class TestLoopSurvival:
    def test_loop_outlives_unknown_exceptions_and_heartbeats(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        root = _records_root(tmp_path)
        args = _args(tmp_path, root)
        state = MonitorState(generation=4)
        calls = {"n": 0}

        def _boom(*a, **k):
            calls["n"] += 1
            raise TypeError("unexpected ledger shape")

        monkeypatch.setattr(launch_monitor, "_tick", _boom)
        sent: list[str] = []
        monkeypatch.setattr(
            launch_monitor, "_send", lambda message, script: sent.append(message)
        )
        rc = run_loop(
            args, Thresholds(), state, None, max_ticks=5, sleep_fn=lambda _s: None
        )
        assert rc == 0
        assert calls["n"] == 5, "the loop must keep ticking through failures"
        payload = json.loads(
            heartbeat_path(args.state_file).read_text(encoding="utf-8")
        )
        assert payload["tick_counter"] == 5
        assert payload["tick_ok"] is False
        assert payload["reason"].startswith("DEGRADED: TypeError")
        assert payload["consecutive_failures"] == 5
        assert payload["generation"] == 4
        degraded = [message for message in sent if "MONITOR DEGRADED" in message]
        assert len(degraded) == 1, "escalates once at the threshold, keeps going"
        assert str(DEGRADED_ESCALATION_TICKS) in degraded[0]

    def test_heartbeat_advances_on_healthy_ticks(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        root = _records_root(tmp_path)
        args = _args(tmp_path, root)
        state = MonitorState()
        rc = run_loop(
            args, Thresholds(), state, None, max_ticks=2, sleep_fn=lambda _s: None
        )
        assert rc == 0
        payload = json.loads(
            heartbeat_path(args.state_file).read_text(encoding="utf-8")
        )
        assert payload["tick_ok"] is True
        assert payload["tick_counter"] == 2

    def test_evidence_unavailable_is_typed_not_a_healthy_zero(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        root = _records_root(tmp_path)
        models = root / "models" / "current-shard.jsonl"
        original = Path.read_bytes

        def _deny(self):
            if self == models:
                raise PermissionError(13, "denied")
            return original(self)

        monkeypatch.setattr(Path, "read_bytes", _deny)
        with pytest.raises(MonitorEvidenceUnavailable) as excinfo:
            build_snapshot(root, check_disk=False)
        assert "monitor_evidence_unavailable" in str(excinfo.value)

    def test_missing_ledger_still_reads_as_empty(self, tmp_path: Path) -> None:
        assert read_jsonl_tolerant(tmp_path / "absent.jsonl") == ([], 0)


class TestRestartSeed:
    def test_fresh_state_baselines_instead_of_alarming(self, tmp_path: Path) -> None:
        # U-O6: 8 standing parks + fresh state must NOT read as +8 growth.
        snapshot_kwargs = dict(
            generated_at="t",
            roster_label="step-200",
            roster_source="registry",
            roster_total=20,
            done=2,
            runs=2,
            failed=0,
            skipped=0,
            deferred=0,
            in_flight=10,
            in_flight_started=1,
            block_reasons={"deferred:needs-host-disk": 8},
            corrupt_lines=0,
            overall_rate_per_hour=None,
            rolling_rate_per_hour=None,
            eta_seconds=None,
            recent_error_rate=None,
            recent_error_window=0,
            quota_paused=False,
            disk_free_gb={},
            last_activity_age_minutes=1.0,
        )
        snapshot = Snapshot(parked=8, **snapshot_kwargs)
        state = MonitorState()
        assert not state.baselines_seeded
        # Seeding is what _tick does before evaluating; emulate its rule here.
        state.baselines_seeded = True
        state.last_blocked_count = snapshot.blocked
        kinds = {kind for kind, _ in evaluate_alerts(snapshot, state, Thresholds())}
        assert "block-tail" not in kinds
        grown = Snapshot(parked=11, **snapshot_kwargs)
        kinds = {kind for kind, _ in evaluate_alerts(grown, state, Thresholds())}
        assert "block-tail" in kinds

    def test_state_round_trips_seed_and_generation(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"
        state = MonitorState(baselines_seeded=True, generation=7)
        state.save(path)
        loaded = MonitorState.load(path)
        assert loaded.baselines_seeded is True
        assert loaded.generation == 7


class TestSingleInstanceLock:
    def test_second_instance_is_refused(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        first = acquire_monitor_lock(state_path)
        assert first is not None
        assert acquire_monitor_lock(state_path) is None
        first.close()
        again = acquire_monitor_lock(state_path)
        assert again is not None
        again.close()


class TestWatcherOfWatcher:
    def test_heartbeat_verdicts(self, tmp_path: Path) -> None:
        state_path = tmp_path / "state.json"
        assert check_heartbeat(state_path, max_age_seconds=180)[0] == "missing"
        now = datetime.now(timezone.utc)
        write_heartbeat(
            state_path,
            tick_ok=True,
            reason=None,
            consecutive_failures=0,
            tick_counter=1,
            tick_duration_seconds=0.1,
            generation=1,
            poll_seconds=60,
            now=now,
        )
        verdict, _ = check_heartbeat(state_path, max_age_seconds=180, now=now)
        assert verdict == "fresh"
        verdict, detail = check_heartbeat(
            state_path,
            max_age_seconds=180,
            now=now + timedelta(seconds=181),
        )
        assert verdict == "stale"
        assert "181s" in detail
        heartbeat_path(state_path).write_text("not json", encoding="utf-8")
        assert check_heartbeat(state_path, max_age_seconds=180)[0] == "unreadable"

    def test_failed_tick_is_disclosed_in_a_fresh_verdict(
        self, tmp_path: Path
    ) -> None:
        state_path = tmp_path / "state.json"
        now = datetime.now(timezone.utc)
        write_heartbeat(
            state_path,
            tick_ok=False,
            reason="DEGRADED: KeyError: 'status'",
            consecutive_failures=2,
            tick_counter=9,
            tick_duration_seconds=0.1,
            generation=1,
            poll_seconds=60,
            now=now,
        )
        verdict, detail = check_heartbeat(state_path, max_age_seconds=180, now=now)
        assert verdict == "fresh"
        assert "last tick FAILED" in detail


class TestLaunchdLifecycle:
    def test_monitor_plist_keeps_alive_and_watchdog_polls(
        self, tmp_path: Path
    ) -> None:
        import plistlib

        args = _args(tmp_path, tmp_path / "records")
        monitor = plistlib.loads(render_monitor_launchd_plist(args).encode("utf-8"))
        assert monitor["Label"] == "com.menagerie.launch-monitor"
        assert monitor["KeepAlive"] == {"SuccessfulExit": False}
        assert monitor["RunAtLoad"] is True
        assert "--loop" in monitor["ProgramArguments"]
        watchdog = plistlib.loads(render_watchdog_launchd_plist(args).encode("utf-8"))
        assert watchdog["Label"] == "com.menagerie.launch-monitor-watchdog"
        assert watchdog["StartInterval"] == 180
        assert "--check-heartbeat" in watchdog["ProgramArguments"]
        assert "--send" in watchdog["ProgramArguments"]
