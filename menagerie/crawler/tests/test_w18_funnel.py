"""W18 core: rung epochs, net-vs-churn, the flow funnel, and its alarms.

M5: rates and the ETA derive from NET first-terminals inside THIS rung's
epoch -- 200a's `done` sat at 139 for 14 hours while the reported rate swung
on head re-mints, and the first-ETA fired on carried terminals. M7: per-class
flows over a trailing window with a starvation alarm on input>0/flow=0 --
the gate-wedge ran ~10 hours because author attempts kept every ledger warm
while exec/gates starved. M9: leg-cycling (>=3 leg-configs / 2h / 0 net
terminals -- the legs 31-36 loop). M10: the leg-config-attested author wall
bounds the NN-6 grace, not an operator-retyped flag. Width comes from
spend-ledger leases, never a process listing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from menagerie.crawler.tools import launch_monitor as lm

pytestmark = pytest.mark.smoke

T0 = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
MINUTE = timedelta(minutes=1)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _model_row(stable_id: str, seq: int, code: str, kind: str, at: datetime) -> dict:
    return {
        "stable_id": stable_id,
        "record_seq": seq,
        "created_at": at.isoformat(),
        "status": {"code": code, "kind": kind},
    }


def _leg_config(at: datetime, **details: Any) -> dict:
    return {
        "event_kind": "leg-config",
        "created_at": at.isoformat(),
        "details": {"author_concurrency": 12, **details},
    }


def _spend(events: Sequence[tuple[str, str, datetime, dict]]) -> list[dict]:
    rows = []
    for event, spend_id, at, extra in events:
        rows.append(
            {"event": event, "spend_id": spend_id, "created_at": at.isoformat(), **extra}
        )
    return rows


def _root(
    tmp_path: Path,
    *,
    members: int = 6,
    model_rows: Sequence[dict] = (),
    operational: Sequence[dict] = (),
    attempts: Sequence[dict] = (),
    gates: Sequence[dict] = (),
    spend: Sequence[dict] = (),
    registered_at: datetime = T0,
) -> Path:
    root = tmp_path / "records"
    ids = [f"m{i}" for i in range(1, members + 1)]
    _write_jsonl(
        root / "intake" / "registered-rungs.jsonl",
        [
            {
                "label": "step-200",
                "snapshot_id": "snap",
                "created_at": registered_at.isoformat(),
            }
        ],
    )
    _write_jsonl(
        root / "intake" / "snap" / "items.jsonl", [{"stable_id": i} for i in ids]
    )
    _write_jsonl(root / "models" / "current-shard.jsonl", list(model_rows))
    _write_jsonl(root / "attempts" / "local.jsonl", list(attempts))
    _write_jsonl(root / "gates" / "current-shard.jsonl", list(gates))
    _write_jsonl(root / "operational" / "events.jsonl", list(operational))
    _write_jsonl(root / "operational" / "spend.jsonl", list(spend))
    return root


class TestNetVsChurn:
    def test_two_rung_replay_rate_ignores_carried_terminals(
        self, tmp_path: Path
    ) -> None:
        # Rung 1 minted m1/m2 BEFORE this rung's epoch; this rung minted m3/m4.
        rows = [
            _model_row("m1", 1, "runs", "runs", T0 - 120 * MINUTE),
            _model_row("m2", 1, "runs", "runs", T0 - 110 * MINUTE),
            _model_row("m3", 1, "runs", "runs", T0 + 10 * MINUTE),
            _model_row("m4", 1, "runs", "runs", T0 + 40 * MINUTE),
        ]
        root = _root(
            tmp_path,
            model_rows=rows,
            operational=[_leg_config(T0)],
        )
        snapshot = lm.build_snapshot(
            root, now=T0 + 60 * MINUTE, check_disk=False
        )
        assert snapshot.rung_epoch_started_at == T0.isoformat()
        assert snapshot.net_terminals == 2
        assert snapshot.carried_terminals == 2
        # 1 completion over 30 minutes of net instants = 2/h; carried spacing
        # (10 minutes apart, 2 hours earlier) must not touch the rate.
        assert snapshot.overall_rate_per_hour == pytest.approx(2.0)

    def test_zero_net_window_is_honest(self, tmp_path: Path) -> None:
        rows = [
            _model_row("m1", 1, "runs", "runs", T0 - 120 * MINUTE),
            _model_row("m1", 2, "runs", "runs", T0 + 10 * MINUTE),  # churn re-mint
            _model_row("m1", 3, "runs", "runs", T0 + 20 * MINUTE),  # churn re-mint
        ]
        root = _root(tmp_path, model_rows=rows, operational=[_leg_config(T0)])
        snapshot = lm.build_snapshot(root, now=T0 + 60 * MINUTE, check_disk=False)
        assert snapshot.net_terminals == 0
        assert snapshot.churn_terminal_rows == 2
        assert snapshot.net_rate_per_hour is None
        assert snapshot.eta_seconds is None
        assert snapshot.eta_basis == "insufficient-evidence"

    def test_regressed_member_is_counted(self, tmp_path: Path) -> None:
        rows = [
            _model_row("m1", 1, "runs", "runs", T0 + 5 * MINUTE),
            # Head regressed to a non-terminal redrive state.
            {
                "stable_id": "m1",
                "record_seq": 2,
                "created_at": (T0 + 30 * MINUTE).isoformat(),
                "status": {"code": "pending:redrive", "kind": "pending"},
            },
        ]
        root = _root(tmp_path, model_rows=rows, operational=[_leg_config(T0)])
        snapshot = lm.build_snapshot(root, now=T0 + 60 * MINUTE, check_disk=False)
        assert snapshot.regressed_members == 1

    def test_first_eta_cannot_fire_on_carried(self, tmp_path: Path, monkeypatch) -> None:
        carried = [
            _model_row("m1", 1, "runs", "runs", T0 - 120 * MINUTE),
            _model_row("m2", 1, "runs", "runs", T0 - 110 * MINUTE),
        ]
        root = _root(tmp_path, model_rows=carried, operational=[_leg_config(T0)])
        sent: list[str] = []
        monkeypatch.setattr(lm, "_send", lambda message, script: sent.append(message))
        monkeypatch.setattr(lm, "_utc_now", lambda: T0 + 30 * MINUTE)
        args = argparse.Namespace(
            records_root=root,
            rung_label=None,
            rolling_window=20,
            error_window=20,
            cas_root=None,
            outbox=None,
            state_file=tmp_path / "state.json",
            interval=3600.0,
        )
        state = lm.MonitorState()
        thresholds = lm.Thresholds(first_eta_min=2)
        lm._tick(args, thresholds, state, None)
        assert state.first_eta_sent is False, "2 carried terminals must not fire it"
        fresh = carried + [
            _model_row("m3", 1, "runs", "runs", T0 + 5 * MINUTE),
            _model_row("m4", 1, "runs", "runs", T0 + 10 * MINUTE),
        ]
        _write_jsonl(root / "models" / "current-shard.jsonl", fresh)
        lm._tick(args, thresholds, state, None)
        assert state.first_eta_sent is True


class TestFlowFunnelAndStarvation:
    def test_exec_starvation_is_named_while_authors_flow(self, tmp_path: Path) -> None:
        # Three author sessions completed in-window; ZERO exec attempts ever.
        spend = _spend(
            [
                ("session_reserved", f"s{i}", T0 + i * MINUTE, {"lane": "author"})
                for i in range(3)
            ]
            + [
                (
                    "session_settled",
                    f"s{i}",
                    T0 + (20 + i) * MINUTE,
                    {"local_outcome": "completed", "machine": {"wall_seconds": 60.0}},
                )
                for i in range(3)
            ]
        )
        root = _root(tmp_path, spend=spend, operational=[_leg_config(T0)])
        snapshot = lm.build_snapshot(root, now=T0 + 40 * MINUTE, check_disk=False)
        assert snapshot.flows["author-sessions"] == 3
        assert snapshot.flows["exec-attempts"] == 0
        assert snapshot.stage_inputs["exec"] == 3
        assert "exec" in snapshot.starved_stages
        kinds = {
            kind
            for kind, _ in lm.evaluate_alerts(
                snapshot, lm.MonitorState(baselines_seeded=True), lm.Thresholds()
            )
        }
        assert "stage-starvation:exec" in kinds

    def test_gate_flow_is_counted_by_kind(self, tmp_path: Path) -> None:
        gates = [
            {
                "gate_kind": "metadata_batch",
                "items": ["m1"],
                "checker": {"finished_at": (T0 + 10 * MINUTE).isoformat()},
            },
            {
                "gate_kind": "terminal_disposition",
                "items": ["m1"],
                "checker": {"finished_at": (T0 + 12 * MINUTE).isoformat()},
            },
        ]
        root = _root(tmp_path, gates=gates, operational=[_leg_config(T0)])
        snapshot = lm.build_snapshot(root, now=T0 + 30 * MINUTE, check_disk=False)
        assert snapshot.flows["gate-adjudications"] == 2
        assert snapshot.flows["gate:metadata_batch"] == 1
        assert snapshot.flows["gate:terminal_disposition"] == 1

    def test_quota_pause_suppresses_starvation(self, tmp_path: Path) -> None:
        spend = _spend(
            [
                ("session_reserved", "s0", T0, {"lane": "author"}),
                (
                    "session_settled",
                    "s0",
                    T0 + 20 * MINUTE,
                    {"local_outcome": "completed", "machine": {"wall_seconds": 60.0}},
                ),
            ]
        )
        operational = [
            _leg_config(T0),
            {"event_kind": "usage-pause", "created_at": (T0 + 21 * MINUTE).isoformat()},
        ]
        root = _root(tmp_path, spend=spend, operational=operational)
        snapshot = lm.build_snapshot(root, now=T0 + 40 * MINUTE, check_disk=False)
        assert snapshot.quota_paused is True
        assert snapshot.starved_stages == []


class TestLegCycling:
    def test_three_leg_configs_two_hours_zero_net_alarms(self, tmp_path: Path) -> None:
        operational = [
            _leg_config(T0),
            _leg_config(T0 + 30 * MINUTE),
            _leg_config(T0 + 60 * MINUTE),
        ]
        root = _root(tmp_path, operational=operational)
        snapshot = lm.build_snapshot(root, now=T0 + 70 * MINUTE, check_disk=False)
        assert snapshot.leg_configs_in_cycle_window == 3
        assert snapshot.leg_cycling is True
        kinds = {
            kind
            for kind, _ in lm.evaluate_alerts(
                snapshot, lm.MonitorState(baselines_seeded=True), lm.Thresholds()
            )
        }
        assert "leg-cycling" in kinds

    def test_net_terminal_inside_the_window_disarms(self, tmp_path: Path) -> None:
        operational = [
            _leg_config(T0),
            _leg_config(T0 + 30 * MINUTE),
            _leg_config(T0 + 60 * MINUTE),
        ]
        rows = [_model_row("m1", 1, "runs", "runs", T0 + 45 * MINUTE)]
        root = _root(tmp_path, operational=operational, model_rows=rows)
        snapshot = lm.build_snapshot(root, now=T0 + 70 * MINUTE, check_disk=False)
        assert snapshot.leg_cycling is False


class TestWidthFromSpendLeases:
    def test_effective_width_is_open_author_leases(self, tmp_path: Path) -> None:
        spend = _spend(
            [
                ("session_reserved", "a1", T0, {"lane": "author"}),
                ("session_reserved", "a2", T0, {"lane": "author"}),
                ("session_reserved", "c1", T0, {"lane": "checker"}),
                (
                    "session_custody_released",
                    "a1",
                    T0 + 10 * MINUTE,
                    {"outcome": "verified-exit"},
                ),
            ]
        )
        root = _root(tmp_path, spend=spend, operational=[_leg_config(T0)])
        snapshot = lm.build_snapshot(root, now=T0 + 20 * MINUTE, check_disk=False)
        assert snapshot.effective_width == 1
        assert snapshot.configured_width == 12

    def test_uninstrumented_spend_reads_width_as_unknown_never_zero(
        self, tmp_path: Path
    ) -> None:
        # 200a's spend ledger began mid-campaign; its absence must read n/a.
        root = _root(tmp_path, operational=[_leg_config(T0)])
        snapshot = lm.build_snapshot(root, now=T0, check_disk=False)
        assert snapshot.spend_instrumented is False
        assert snapshot.effective_width is None
        assert snapshot.wall_hours_per_net_terminal is None
        kinds = {
            kind
            for kind, _ in lm.evaluate_alerts(
                snapshot, lm.MonitorState(baselines_seeded=True), lm.Thresholds()
            )
        }
        assert "width-collapse" not in kinds
        assert "stage-starvation:author" not in kinds

    def test_width_collapse_pages(self, tmp_path: Path) -> None:
        # Configured 12 wide, spend instrumented with zero OPEN leases,
        # unauthored backlog, past the author wall: the watchdog PAGES.
        spend = _spend(
            [
                ("session_reserved", "a1", T0 - 3 * timedelta(hours=1), {"lane": "author"}),
                (
                    "session_custody_released",
                    "a1",
                    T0 - 2 * timedelta(hours=1),
                    {"outcome": "verified-exit"},
                ),
            ]
        )
        root = _root(
            tmp_path,
            spend=spend,
            operational=[_leg_config(T0 - 3 * timedelta(hours=1))],
        )
        snapshot = lm.build_snapshot(root, now=T0, check_disk=False)
        assert snapshot.effective_width == 0
        assert snapshot.stage_inputs["author"] > 0
        kinds = {
            kind
            for kind, _ in lm.evaluate_alerts(
                snapshot, lm.MonitorState(baselines_seeded=True), lm.Thresholds()
            )
        }
        assert "width-collapse" in kinds


class TestRecordedAuthorWall:
    def test_recorded_wall_bounds_the_grace(self, tmp_path: Path) -> None:
        operational = [_leg_config(T0, author_wall_seconds=600.0)]
        root = _root(tmp_path, operational=operational)
        snapshot = lm.build_snapshot(root, now=T0 + 5 * MINUTE, check_disk=False)
        assert snapshot.author_wall_seconds_recorded == 600.0
        # Inside the recorded 10-minute grant.
        assert lm.within_first_author_wall(snapshot, lm.Thresholds()) is True
        late = lm.build_snapshot(root, now=T0 + 11 * MINUTE, check_disk=False)
        # Past the recorded grant even though the flag default is far larger.
        assert lm.within_first_author_wall(late, lm.Thresholds()) is False


class TestAuthorityProjection:
    def test_lag_and_root_are_always_stated(self, tmp_path: Path) -> None:
        rows = [_model_row("m1", 1, "runs", "runs", T0)]
        root = _root(tmp_path, model_rows=rows, operational=[_leg_config(T0)])
        snapshot = lm.build_snapshot(root, now=T0 + 90 * MINUTE, check_disk=False)
        assert snapshot.authority_root
        assert snapshot.projection_newest_row_at == T0.isoformat()
        assert snapshot.projection_lag_minutes == pytest.approx(90.0)
        rendered = lm.render_snapshot(snapshot)
        assert "authority" in rendered
        assert "90m ago" in rendered

    def test_no_rows_renders_red_never_zero(self, tmp_path: Path) -> None:
        root = _root(tmp_path)
        snapshot = lm.build_snapshot(root, now=T0, check_disk=False)
        assert snapshot.projection_lag_minutes is None
        assert "NO ROWS (RED)" in lm.render_snapshot(snapshot)

    def test_spend_cost_denominator_is_net(self, tmp_path: Path) -> None:
        spend = _spend(
            [
                ("session_reserved", "s0", T0, {"lane": "author"}),
                (
                    "session_settled",
                    "s0",
                    T0 + 10 * MINUTE,
                    {"local_outcome": "completed", "machine": {"wall_seconds": 7200.0}},
                ),
            ]
        )
        rows = [_model_row("m1", 1, "runs", "runs", T0 + 15 * MINUTE)]
        root = _root(tmp_path, spend=spend, model_rows=rows, operational=[_leg_config(T0)])
        snapshot = lm.build_snapshot(root, now=T0 + 20 * MINUTE, check_disk=False)
        assert snapshot.spend_wall_hours == pytest.approx(2.0)
        assert snapshot.wall_hours_per_net_terminal == pytest.approx(2.0)
        zero_net = _root(
            tmp_path / "b", spend=spend, operational=[_leg_config(T0)]
        )
        snapshot = lm.build_snapshot(zero_net, now=T0 + 20 * MINUTE, check_disk=False)
        assert snapshot.wall_hours_per_net_terminal is None


class TestLegOutcomeCensus:
    def test_not_instrumented_until_the_driver_emits(self, tmp_path: Path) -> None:
        root = _root(tmp_path, operational=[_leg_config(T0)])
        snapshot = lm.build_snapshot(root, now=T0, check_disk=False)
        assert snapshot.leg_outcome_census == "not-instrumented"
        kinds = {
            kind
            for kind, _ in lm.evaluate_alerts(
                snapshot, lm.MonitorState(baselines_seeded=True), lm.Thresholds()
            )
        }
        assert "leg-census" not in kinds

    def test_consistent_with_running_leg(self, tmp_path: Path) -> None:
        operational = [
            _leg_config(T0),
            {"event_kind": "leg-outcome", "created_at": (T0 + MINUTE).isoformat()},
            _leg_config(T0 + 2 * MINUTE),
        ]
        root = _root(tmp_path, operational=operational)
        snapshot = lm.build_snapshot(root, now=T0 + 3 * MINUTE, check_disk=False)
        assert snapshot.leg_outcome_census == "consistent"

    def test_inconsistent_census_alarms(self, tmp_path: Path) -> None:
        operational = [
            _leg_config(T0),
            _leg_config(T0 + MINUTE),
            _leg_config(T0 + 2 * MINUTE),
            {"event_kind": "leg-outcome", "created_at": (T0 + MINUTE).isoformat()},
        ]
        root = _root(tmp_path, operational=operational)
        snapshot = lm.build_snapshot(root, now=T0 + 3 * MINUTE, check_disk=False)
        assert snapshot.leg_outcome_census == "inconsistent"
        kinds = {
            kind
            for kind, _ in lm.evaluate_alerts(
                snapshot, lm.MonitorState(baselines_seeded=True), lm.Thresholds()
            )
        }
        assert "leg-census" in kinds


class TestOperatorOwedAging:
    def test_park_age_renders_and_pages_past_threshold(self, tmp_path: Path) -> None:
        root = _root(tmp_path, operational=[_leg_config(T0)])
        park = {
            "stable_id": "m1",
            "disposition": "review-held:terminal-recording-refused",
            "created_at": (T0 - 31 * timedelta(hours=1)).isoformat(),
        }
        path = root / "intake-extensions" / "terminal-refusal-parks.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(park) + chr(10), encoding="utf-8")
        snapshot = lm.build_snapshot(root, now=T0, check_disk=False)
        assert snapshot.parked == 1
        assert snapshot.operator_owed_oldest_hours == pytest.approx(31.0)
        assert "operator-owed: 1 park(s) (oldest 31h)" in lm.render_snapshot(snapshot)
        kinds = {
            kind
            for kind, _ in lm.evaluate_alerts(
                snapshot, lm.MonitorState(baselines_seeded=True), lm.Thresholds()
            )
        }
        assert "operator-owed" in kinds

    def test_young_park_does_not_page(self, tmp_path: Path) -> None:
        root = _root(tmp_path, operational=[_leg_config(T0)])
        park = {
            "stable_id": "m1",
            "disposition": "review-held:terminal-recording-refused",
            "created_at": (T0 - 2 * timedelta(hours=1)).isoformat(),
        }
        path = root / "intake-extensions" / "terminal-refusal-parks.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(park) + chr(10), encoding="utf-8")
        snapshot = lm.build_snapshot(root, now=T0, check_disk=False)
        kinds = {
            kind
            for kind, _ in lm.evaluate_alerts(
                snapshot, lm.MonitorState(baselines_seeded=True), lm.Thresholds()
            )
        }
        assert "operator-owed" not in kinds
