"""OB-M11b: shadow ETA scorecard -- prediction rows, scoring, 200a backtest.

Shadow-first: nothing in the scorecard sends. These tests pin the closed
reachability vocabulary (a starved pipeline is ``blocked``, never a finite
extrapolation), the Little's-law bottleneck choice, the scorer's error table,
and -- B3-XV-23 -- that the 200a replay retro-flags the width saga and the
OMP loop in its rows BEFORE anyone is allowed to trust a delivered ETA. The
measured 200a error is recorded as the method's baseline; the p50<=25%
delivery floor is a JMT-confirmed target (B2-X-O15), deliberately NOT
asserted here.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from menagerie.crawler.tools.eta_scorecard import (
    REACHABILITY_STATES,
    append_row,
    predict,
    read_rows,
    score,
)
from menagerie.crawler.tools.launch_monitor import Snapshot

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "replay_200a"


def _snapshot(**overrides) -> Snapshot:
    base = dict(
        generated_at="2026-08-24T00:00:00+00:00",
        roster_label="step-200",
        roster_source="registry",
        roster_total=10,
        done=2,
        runs=2,
        failed=0,
        skipped=0,
        deferred=0,
        parked=0,
        in_flight=8,
        in_flight_started=2,
        block_reasons={},
        corrupt_lines=0,
        overall_rate_per_hour=2.0,
        rolling_rate_per_hour=2.0,
        eta_seconds=None,
        recent_error_rate=None,
        recent_error_window=0,
        quota_paused=False,
        disk_free_gb={},
        last_activity_age_minutes=1.0,
        net_terminals=2,
        net_rate_per_hour=2.0,
        flows={
            "author-sessions": 4,
            "exec-attempts": 3,
            "gate-adjudications": 2,
            "net-terminals": 2,
        },
        stage_inputs={"author": 2, "exec": 1, "gate": 0, "terminal": 0},
        flow_window_minutes=60.0,
    )
    base.update(overrides)
    return Snapshot(**base)


pytestmark = pytest.mark.smoke


class TestPredict:
    def test_flowing_picks_the_slowest_drain(self) -> None:
        row = predict(_snapshot())
        assert row["reachability"] == "flowing"
        # terminal-rate drain: 8 remaining / 2 per hour = 4h; every stage
        # drains faster, so the terminal rate is the bottleneck.
        assert row["bottleneck"] == "terminal-rate"
        assert row["eta_p50_seconds"] == pytest.approx(4 * 3600.0)
        assert row["eta_p90_seconds"] == pytest.approx(8 * 3600.0)

    def test_starved_stage_is_blocked_with_no_finite_eta(self) -> None:
        row = predict(_snapshot(starved_stages=["exec"]))
        assert row["reachability"] == "blocked"
        assert row["bottleneck"] == "exec"
        assert row["eta_p50_seconds"] is None

    def test_complete_and_insufficient_states(self) -> None:
        assert predict(_snapshot(in_flight=0))["reachability"] == "complete"
        row = predict(_snapshot(net_rate_per_hour=None))
        assert row["reachability"] == "insufficient-evidence"
        assert row["eta_p50_seconds"] is None

    def test_vocabulary_is_closed(self) -> None:
        for snapshot in (
            _snapshot(),
            _snapshot(in_flight=0),
            _snapshot(starved_stages=["gate"]),
            _snapshot(net_rate_per_hour=None),
        ):
            assert predict(snapshot)["reachability"] in REACHABILITY_STATES


class TestScore:
    def test_error_table_shape(self, tmp_path: Path) -> None:
        ledger = tmp_path / "eta.jsonl"
        t0 = datetime(2026, 8, 24, 0, 0, tzinfo=timezone.utc)
        actual = t0 + timedelta(hours=10)
        for hour, net in ((0, 0), (2, 3), (5, 6), (8, 9)):
            ts = t0 + timedelta(hours=hour)
            remaining = 10 - net
            append_row(
                ledger,
                {
                    "ts": ts.isoformat(),
                    "remaining": remaining,
                    "net_terminals": net,
                    "reachability": "flowing",
                    "eta_p50_seconds": (10 - hour) * 3600.0,  # perfect predictions
                },
            )
        table = score(read_rows(ledger), actual)
        assert table["n_scored"] == 4
        assert table["p50_abs_rel_error"] == pytest.approx(0.0)
        assert table["p50_abs_rel_error_after_25pct"] == pytest.approx(0.0)
        assert table["n_after_25pct"] == 3

    def test_empty_ledger_never_reads_as_a_pass(self) -> None:
        table = score([], datetime(2026, 8, 24, tzinfo=timezone.utc))
        assert table["n_scored"] == 0
        assert table["p50_abs_rel_error"] is None


@pytest.mark.usefixtures()
def _noop() -> None:  # keeps pytest from treating the module tail as tests
    pass


def test_200a_backtest_records_baseline_and_retro_flags(tmp_path: Path) -> None:
    """B3-XV-23: the scorecard must retro-flag the 200a incidents on replay.

    The OMP loop must show as leg_cycling/blocked rows in its window, and the
    width saga must show effective width sagging under the configured width.
    The measured error table is PRINTED as the method baseline -- no accuracy
    floor is asserted (delivery stays shadow until floors + JMT sign-off).
    """

    if not FIXTURE_ROOT.exists():
        pytest.fail("replay_200a fixture missing -- D-37 owns it")
    from menagerie.crawler.tools.detection_replay import ReplayHarness

    harness = ReplayHarness(FIXTURE_ROOT, tmp_path / "work")
    ledger = tmp_path / "eta-scorecard.jsonl"
    start = datetime(2026, 8, 20, 3, 30, tzinfo=timezone.utc)
    end = datetime(2026, 8, 23, 17, 0, tzinfo=timezone.utc)
    instant = start
    while instant <= end:
        snapshot = harness.snapshot_at(instant)
        append_row(ledger, predict(snapshot))
        instant += timedelta(minutes=30)
    rows = read_rows(ledger)

    def _within(row, window_start, window_end):
        ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
        return window_start <= ts <= window_end

    omp_start = datetime(2026, 8, 23, 9, 0, tzinfo=timezone.utc)
    omp_end = datetime(2026, 8, 23, 16, 0, tzinfo=timezone.utc)
    omp_rows = [row for row in rows if _within(row, omp_start, omp_end)]
    assert any(
        row.get("leg_cycling") or row.get("reachability") == "blocked"
        for row in omp_rows
    ), "the OMP leg-cycling loop is not retro-flagged"

    saga_start = datetime(2026, 8, 21, 8, 0, tzinfo=timezone.utc)
    saga_end = datetime(2026, 8, 22, 0, 0, tzinfo=timezone.utc)
    saga_rows = [row for row in rows if _within(row, saga_start, saga_end)]
    assert any(
        isinstance(row.get("effective_width"), int)
        and isinstance(row.get("configured_width"), int)
        and row["effective_width"] < row["configured_width"]
        for row in saga_rows
    ), "the width saga is not retro-flagged"

    actual = datetime(2026, 8, 23, 16, 50, 25, tzinfo=timezone.utc)
    table = score(rows, actual)
    assert table["n_scored"] > 0
    print("200a ETA baseline:", json.dumps(table, indent=1, sort_keys=True))
