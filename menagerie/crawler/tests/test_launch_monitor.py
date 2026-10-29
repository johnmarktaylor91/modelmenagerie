"""Unit tests for the read-only launch monitor over a fabricated records root."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

from menagerie.crawler.tools import launch_monitor as lm

_BASE = datetime(2026, 8, 19, 12, 0, 0, tzinfo=timezone.utc)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _model_row(
    stable_id: str, seq: int, code: str, kind: str, created_at: datetime
) -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "record_seq": seq,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "status": {"code": code, "kind": kind, "stage": None, "reason": None},
    }


@pytest.fixture()
def records_root(tmp_path: Path) -> Path:
    """Fabricate a small step-200-shaped records root: done, in-flight, blocked."""

    root = tmp_path / "records"
    members = [f"m{i}" for i in range(1, 9)]
    _write_jsonl(
        root / "intake" / "registered-rungs.jsonl",
        [
            {
                "label": "step-200",
                "snapshot_id": "intake-test",
                "member_count": len(members),
                "created_at": _BASE.isoformat(),
            }
        ],
    )
    _write_jsonl(
        root / "intake" / "intake-test" / "items.jsonl",
        [{"stable_id": member, "name": member} for member in members],
    )
    minute = timedelta(minutes=1)
    _write_jsonl(
        root / "models" / "current-shard.jsonl",
        [
            # m1 has a superseded failed revision; the head (seq 2) is runs.
            _model_row("m1", 1, "failed:forward", "failed", _BASE - 60 * minute),
            _model_row("m4", 1, "runs", "runs", _BASE),
            _model_row("m2", 1, "failed:forward", "failed", _BASE + 10 * minute),
            _model_row("m3", 1, "deferred:needs-cuda", "deferred", _BASE + 20 * minute),
            _model_row("m1", 2, "runs", "runs", _BASE + 30 * minute),
        ],
    )
    _write_jsonl(
        root / "attempts" / "local.jsonl",
        [
            {
                "stable_id": "m5",
                "stage": "forward",
                "started_at": _BASE.isoformat(),
                "finished_at": None,
            }
        ],
    )
    _write_jsonl(
        root / "intake-extensions" / "host-capacity-deferrals.jsonl",
        [
            {
                "stable_id": "m6",
                "disposition": "deferred:host-capacity-wait",
                "recheck_hint": "re-run when a lane frees up",
                "created_at": _BASE.isoformat(),
            }
        ],
    )
    _write_jsonl(root / "operational" / "events.jsonl", [])
    return root


def _snapshot(root: Path, **kwargs: Any) -> lm.Snapshot:
    return lm.build_snapshot(root, now=_BASE + timedelta(hours=1), **kwargs)


@pytest.mark.smoke
def test_snapshot_counts_partition_the_roster(records_root: Path) -> None:
    snapshot = _snapshot(records_root)
    assert snapshot.roster_label == "step-200"
    assert snapshot.roster_source == "registry"
    assert snapshot.roster_total == 8
    assert snapshot.done == 4
    assert snapshot.runs == 2
    assert snapshot.failed == 1
    assert snapshot.skipped == 0
    assert snapshot.deferred == 1
    assert snapshot.parked == 1
    assert snapshot.blocked == 2
    # m5 (started), m7, m8 hold neither a head nor a park row.
    assert snapshot.in_flight == 3
    assert snapshot.in_flight_started == 1
    assert not snapshot.complete
    assert snapshot.corrupt_lines == 0


@pytest.mark.smoke
def test_head_selection_and_block_reasons(records_root: Path) -> None:
    snapshot = _snapshot(records_root)
    # m1's superseded failed revision must not leak into the counts (head = max seq).
    assert snapshot.block_reasons == {
        "deferred:host-capacity-wait": 1,
        "deferred:needs-cuda": 1,
        "failed:forward": 1,
    }


@pytest.mark.smoke
def test_eta_math_from_net_first_terminal_timestamps(records_root: Path) -> None:
    snapshot = _snapshot(records_root)
    # M5 net semantics: the rate derives from each member's FIRST terminal,
    # never head re-mints. m1's first terminal is its superseded failed
    # revision at -60m; its later runs head at +30m is CHURN, not progress.
    # First terminals: -60m, 0m, +10m, +20m -> 3 completions over 80m = 2.25/h.
    assert snapshot.net_terminals == 4
    assert snapshot.churn_terminal_rows == 1
    assert snapshot.overall_rate_per_hour == pytest.approx(2.25)
    assert snapshot.rolling_rate_per_hour == pytest.approx(2.25)
    # 3 in-flight at 2.25/h -> 4800 seconds.
    assert snapshot.eta_seconds == pytest.approx(4800.0)
    assert snapshot.eta_basis == "net-terminal-rate"
    # Kinds in terminal-time order: runs, failed, deferred, runs -> 1/4 failed.
    assert snapshot.recent_error_rate == pytest.approx(0.25)
    assert snapshot.recent_error_window == 4
    assert snapshot.first_terminal_at is not None
    assert snapshot.last_terminal_at is not None


@pytest.mark.smoke
def test_torn_tail_is_tolerated_and_interior_corruption_disclosed(
    records_root: Path,
) -> None:
    models = records_root / "models" / "current-shard.jsonl"
    with models.open("ab") as handle:
        handle.write(b'{"stable_id": "m9", "record_')  # live mid-append torn tail
    snapshot = _snapshot(records_root)
    assert snapshot.done == 4
    assert snapshot.corrupt_lines == 0

    content = models.read_bytes()
    models.write_bytes(b'not-json\n' + content)
    snapshot = _snapshot(records_root)
    assert snapshot.done == 4
    assert snapshot.corrupt_lines == 1


@pytest.mark.smoke
def test_quota_pause_and_resume_events(records_root: Path) -> None:
    events = records_root / "operational" / "events.jsonl"
    _write_jsonl(events, [{"event_kind": "usage-pause", "created_at": _BASE.isoformat()}])
    assert _snapshot(records_root).quota_paused is True
    _write_jsonl(
        events,
        [
            {"event_kind": "usage-pause", "created_at": _BASE.isoformat()},
            {"event_kind": "usage-resume", "created_at": _BASE.isoformat()},
        ],
    )
    assert _snapshot(records_root).quota_paused is False


@pytest.mark.smoke
def test_roster_falls_back_to_models_ledger_without_registry(tmp_path: Path) -> None:
    root = tmp_path / "records"
    _write_jsonl(
        root / "models" / "current-shard.jsonl",
        [_model_row("m1", 1, "runs", "runs", _BASE)],
    )
    snapshot = lm.build_snapshot(root, now=_BASE)
    assert snapshot.roster_source == "models-ledger"
    assert snapshot.roster_total == 1
    assert snapshot.done == 1
    assert snapshot.complete


def _make_snapshot(**overrides: Any) -> lm.Snapshot:
    base: dict[str, Any] = {
        "generated_at": _BASE.isoformat(),
        "roster_label": "step-200",
        "roster_source": "registry",
        "roster_total": 200,
        "done": 50,
        "runs": 40,
        "failed": 5,
        "skipped": 2,
        "deferred": 3,
        "parked": 0,
        "in_flight": 150,
        "in_flight_started": 10,
        "block_reasons": {},
        "corrupt_lines": 0,
        "overall_rate_per_hour": 4.0,
        "rolling_rate_per_hour": 4.0,
        "eta_seconds": 3600.0,
        "recent_error_rate": 0.1,
        "recent_error_window": 20,
        "quota_paused": False,
        "disk_free_gb": {"/records": 100.0},
        "last_activity_age_minutes": 5.0,
    }
    base.update(overrides)
    return lm.Snapshot(**base)


def _alert_kinds(
    snapshot: lm.Snapshot, state: Optional[lm.MonitorState] = None
) -> list[str]:
    state = state or lm.MonitorState()
    return [kind for kind, _message in lm.evaluate_alerts(snapshot, state, lm.Thresholds())]


@pytest.mark.smoke
def test_stall_alert_and_quota_suppression() -> None:
    assert "stall" in _alert_kinds(_make_snapshot(last_activity_age_minutes=45.0))
    assert "stall" not in _alert_kinds(
        _make_snapshot(last_activity_age_minutes=45.0, quota_paused=True)
    )
    assert "stall" not in _alert_kinds(_make_snapshot(last_activity_age_minutes=5.0))


@pytest.mark.smoke
def test_disk_error_rate_and_block_tail_alerts() -> None:
    assert "disk" in _alert_kinds(_make_snapshot(disk_free_gb={"/records": 5.0}))
    assert "error-rate" in _alert_kinds(_make_snapshot(recent_error_rate=0.6))
    assert "error-rate" not in _alert_kinds(
        _make_snapshot(recent_error_rate=0.6, recent_error_window=2)
    )
    assert "block-tail" in _alert_kinds(_make_snapshot(deferred=6, parked=2))


@pytest.mark.smoke
def test_completion_alert_fires_once() -> None:
    done_snapshot = _make_snapshot(in_flight=0, done=195, parked=5)
    assert "completion" in _alert_kinds(done_snapshot)
    reported = lm.MonitorState(completion_sent=True)
    assert "completion" not in _alert_kinds(done_snapshot, reported)


@pytest.mark.smoke
def test_quota_alert_fires_once_until_resume() -> None:
    paused = _make_snapshot(quota_paused=True)
    assert "quota" in _alert_kinds(paused)
    assert "quota" not in _alert_kinds(paused, lm.MonitorState(quota_pause_reported=True))


@pytest.mark.smoke
def test_new_block_reasons_diff_against_state() -> None:
    snapshot = _make_snapshot(block_reasons={"deferred:needs-cuda": 2, "failed:forward": 1})
    state = lm.MonitorState(known_block_reasons=["failed:forward"])
    assert lm.new_block_reasons(snapshot, state) == ["deferred:needs-cuda"]


@pytest.mark.smoke
def test_state_roundtrip_is_durable(tmp_path: Path) -> None:
    path = tmp_path / "state" / "monitor.json"
    state = lm.MonitorState(
        last_hourly_epoch=123.0,
        first_eta_sent=True,
        completion_sent=False,
        last_blocked_count=4,
        known_block_reasons=["deferred:needs-cuda"],
        last_alert_epochs={"stall": 99.0},
    )
    state.save(path)
    loaded = lm.MonitorState.load(path)
    assert loaded == state
    # A missing or foreign-schema file loads as a fresh state, never raises.
    assert lm.MonitorState.load(tmp_path / "absent.json") == lm.MonitorState()
    (tmp_path / "foreign.json").write_text('{"schema_version": "other"}', encoding="utf-8")
    assert lm.MonitorState.load(tmp_path / "foreign.json") == lm.MonitorState()


@pytest.mark.smoke
def test_render_is_terse_ascii(records_root: Path) -> None:
    rendered = lm.render_snapshot(_snapshot(records_root))
    assert all(ord(char) < 128 for char in rendered)
    assert "4/8 done" in rendered
    assert "3 in-flight" in rendered
    assert "block reasons:" in rendered


# --- C-4: a terminal-looking head with an off-vocabulary code is never silently done ---


@pytest.mark.smoke
def test_off_vocabulary_code_is_not_counted_as_done(records_root: Path) -> None:
    """A head whose CODE is outside the closed terminal set must not read as done.

    The kind field alone used to license the count, so a record carrying a
    recognized ``kind`` with an unknown ``code`` -- a new code this monitor has
    not learned, or a corrupted status -- was reported as a completed model.
    """

    _write_jsonl(
        records_root / "models" / "current-shard.jsonl",
        [
            _model_row("m1", 1, "runs", "runs", _BASE),
            _model_row("m2", 1, "runs", "runs", _BASE + timedelta(minutes=10)),
            # Terminal KIND, off-vocabulary CODE: the C-4 silent pass-through.
            _model_row(
                "m3", 1, "runs:partially-verified", "runs", _BASE + timedelta(minutes=20)
            ),
        ],
    )
    snapshot = _snapshot(records_root)

    assert snapshot.done == 2
    assert snapshot.unrecognized == 1
    assert snapshot.unrecognized_codes == {"runs:partially-verified": 1}
    # It is disclosed as its own bucket, NOT laundered into the corrupt count.
    assert snapshot.corrupt_lines == 0


@pytest.mark.smoke
def test_recognized_code_with_off_vocabulary_kind_is_unrecognized(records_root: Path) -> None:
    """Both coordinates must be in their closed vocabularies to count as done."""

    _write_jsonl(
        records_root / "models" / "current-shard.jsonl",
        [_model_row("m1", 1, "runs", "thriving", _BASE)],
    )
    snapshot = _snapshot(records_root)

    assert snapshot.done == 0
    assert snapshot.unrecognized == 1


@pytest.mark.smoke
def test_unrecognized_head_blocks_completion_and_alerts(records_root: Path) -> None:
    """A crawl with unclassifiable heads is never announced COMPLETE."""

    _write_jsonl(
        records_root / "intake" / "intake-test" / "items.jsonl",
        [{"stable_id": "m1", "name": "m1"}],
    )
    _write_jsonl(
        records_root / "models" / "current-shard.jsonl",
        [_model_row("m1", 1, "runs:invented", "runs", _BASE)],
    )
    snapshot = _snapshot(records_root)

    assert snapshot.in_flight == 0
    assert not snapshot.complete
    assert "unrecognized" in _alert_kinds(snapshot)
    assert "completion" not in _alert_kinds(snapshot)


@pytest.mark.smoke
def test_unrecognized_is_rendered(records_root: Path) -> None:
    """The bucket is disclosed in the reported line block, not only in the object."""

    _write_jsonl(
        records_root / "models" / "current-shard.jsonl",
        [_model_row("m1", 1, "runs:invented", "runs", _BASE)],
    )
    rendered = lm.render_snapshot(_snapshot(records_root))

    assert "unrecognized" in rendered
    assert "runs:invented" in rendered


@pytest.mark.smoke
def test_partition_still_accounts_for_every_roster_member(records_root: Path) -> None:
    """done + parked + in_flight + unrecognized covers the whole roster."""

    _write_jsonl(
        records_root / "models" / "current-shard.jsonl",
        [
            _model_row("m1", 1, "runs", "runs", _BASE),
            _model_row("m2", 1, "runs:invented", "runs", _BASE),
        ],
    )
    snapshot = _snapshot(records_root)

    assert (
        snapshot.done + snapshot.parked + snapshot.in_flight + snapshot.unrecognized
        == snapshot.roster_total
    )


# --- NN-6: no first-hour STALL alarm before the first author session can finish ---


def _leg_config_event(created_at: datetime) -> dict[str, Any]:
    return {
        "event_kind": "leg-config",
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "details": {"scheduler": "waves", "author_concurrency": 2},
    }


@pytest.mark.smoke
def test_leg_start_is_read_from_the_leg_config_event(records_root: Path) -> None:
    """Leg age derives from the durable per-leg config attestation."""

    _write_jsonl(
        records_root / "operational" / "events.jsonl",
        [
            _leg_config_event(_BASE - timedelta(hours=5)),
            _leg_config_event(_BASE + timedelta(minutes=30)),
        ],
    )
    snapshot = _snapshot(records_root)

    # Newest leg-config wins: this leg started 30 minutes into the hour.
    assert snapshot.leg_started_at is not None
    assert snapshot.leg_age_minutes == pytest.approx(30.0)


@pytest.mark.smoke
def test_no_stall_alarm_in_the_first_author_wall_before_any_model_starts() -> None:
    """The NN-6 false alarm: a long first author session writes no ledger row.

    A 30-minute stall threshold under a 60-minute author wall grant reported a
    STALL every single launch, because the first author session legitimately
    runs longer than the threshold before anything lands.
    """

    thresholds = lm.Thresholds(stall_minutes=30.0, author_wall_seconds=3600.0)
    quiet_launch = _make_snapshot(
        done=0,
        runs=0,
        failed=0,
        skipped=0,
        deferred=0,
        in_flight=200,
        in_flight_started=0,
        last_activity_age_minutes=45.0,
        leg_age_minutes=45.0,
    )
    kinds = [
        kind
        for kind, _message in lm.evaluate_alerts(quiet_launch, lm.MonitorState(), thresholds)
    ]

    assert "stall" not in kinds


@pytest.mark.smoke
def test_stall_alarm_returns_once_past_the_first_author_wall() -> None:
    """Suppression is bounded by the grant; it is not an open-ended amnesty."""

    thresholds = lm.Thresholds(stall_minutes=30.0, author_wall_seconds=3600.0)
    stalled = _make_snapshot(
        done=0,
        in_flight=200,
        in_flight_started=0,
        last_activity_age_minutes=75.0,
        leg_age_minutes=75.0,
    )
    kinds = [
        kind for kind, _message in lm.evaluate_alerts(stalled, lm.MonitorState(), thresholds)
    ]

    assert "stall" in kinds


@pytest.mark.smoke
def test_stall_alarm_fires_inside_the_wall_once_a_model_has_started() -> None:
    """Once work has started, silence is a real stall even in the first hour."""

    thresholds = lm.Thresholds(stall_minutes=30.0, author_wall_seconds=3600.0)
    started = _make_snapshot(
        done=0,
        in_flight=200,
        in_flight_started=3,
        last_activity_age_minutes=45.0,
        leg_age_minutes=45.0,
    )
    kinds = [
        kind for kind, _message in lm.evaluate_alerts(started, lm.MonitorState(), thresholds)
    ]

    assert "stall" in kinds


@pytest.mark.smoke
def test_unknown_leg_age_does_not_suppress_a_stall() -> None:
    """No leg-config attestation means no suppression licence -- fail loud."""

    thresholds = lm.Thresholds(stall_minutes=30.0, author_wall_seconds=3600.0)
    unknown = _make_snapshot(
        done=0,
        in_flight=200,
        in_flight_started=0,
        last_activity_age_minutes=45.0,
        leg_age_minutes=None,
    )
    kinds = [
        kind for kind, _message in lm.evaluate_alerts(unknown, lm.MonitorState(), thresholds)
    ]

    assert "stall" in kinds
