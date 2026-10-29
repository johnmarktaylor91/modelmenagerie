"""D-35: the one-command probe, lock-holder-only liveness, and the pgrep ban.

200a's watchers keyed liveness on ``pgrep`` patterns and completion on a
string match of a rendered line; both false-read in production (a watcher
matched its own shell as the driver; a kill aimed at a phantom match). The
probe's liveness authority is the driver.lock kernel flock holder ONLY, with
pid + boot_id + start_token cross-checks, a closed verdict vocabulary mirrored
by exit codes, staleness-as-death for the driver-owned status marker, and an
arm-time self-test against simulated dead/held locks. The repo-grep ban makes
a new pgrep watcher a failing test, not a memory rule.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from menagerie.crawler.tools.crawl_doctor import (
    VERDICT_EXIT_CODES,
    VERDICTS,
    driver_lock_path,
    driver_status_marker_path,
    evaluate,
    read_driver_liveness,
    read_status_marker,
    self_test,
)
from menagerie.crawler.worker_supervisor import current_boot_id, process_start_token

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_ledger(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _campaign(tmp_path: Path, *, terminal: bool = False) -> tuple[Path, Path]:
    records = tmp_path / "records"
    runtime = tmp_path / ".crawl-local"
    _write_ledger(
        records / "intake" / "registered-rungs.jsonl",
        [{"label": "step-200", "snapshot_id": "snap-1"}],
    )
    _write_ledger(records / "intake" / "snap-1" / "items.jsonl", [{"stable_id": "m1"}])
    rows = []
    if terminal:
        rows.append(
            {
                "stable_id": "m1",
                "record_seq": 1,
                "status": {"code": "runs", "kind": "runs"},
                "created_at": "2026-08-24T00:00:00+00:00",
            }
        )
    _write_ledger(records / "models" / "current-shard.jsonl", rows)
    return records, runtime


def _hold_lock(runtime_root: Path, owner: dict):
    path = driver_lock_path(runtime_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w", encoding="utf-8")
    handle.write(json.dumps(owner) + "\n")
    handle.flush()
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _self_owner() -> dict:
    return {
        "pid": os.getpid(),
        "boot_id": current_boot_id(),
        "start_token": process_start_token(os.getpid()),
    }


class TestLiveness:
    def test_missing_lock_is_dead(self, tmp_path: Path) -> None:
        liveness = read_driver_liveness(driver_lock_path(tmp_path))
        assert liveness.held is False

    def test_unheld_lock_is_dead(self, tmp_path: Path) -> None:
        path = driver_lock_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"pid": 4}) + "\n", encoding="utf-8")
        liveness = read_driver_liveness(path)
        assert liveness.held is False
        assert "UNHELD" in liveness.detail

    def test_held_lock_with_matching_identity_verifies(self, tmp_path: Path) -> None:
        handle = _hold_lock(tmp_path, _self_owner())
        try:
            liveness = read_driver_liveness(driver_lock_path(tmp_path))
        finally:
            handle.close()
        assert liveness.held is True
        assert liveness.identity_verified is True

    def test_pid_reuse_is_named_not_trusted(self, tmp_path: Path) -> None:
        owner = _self_owner()
        owner["start_token"] = "boot-0-started-1"
        handle = _hold_lock(tmp_path, owner)
        try:
            liveness = read_driver_liveness(driver_lock_path(tmp_path))
        finally:
            handle.close()
        assert liveness.held is True
        assert liveness.identity_verified is False
        assert "start_token mismatch" in liveness.detail


class TestVerdicts:
    def test_vocabulary_is_closed_and_exit_codes_mirror(self) -> None:
        assert set(VERDICT_EXIT_CODES) == set(VERDICTS)
        assert sorted(VERDICT_EXIT_CODES.values()) == [0, 1, 2, 3, 4]

    def test_dead_without_a_driver(self, tmp_path: Path) -> None:
        records, runtime = _campaign(tmp_path)
        report = evaluate(records, runtime)
        assert report.verdict == "DEAD"
        assert report.exit_code == 3

    def test_running_healthy_under_a_held_lock(self, tmp_path: Path) -> None:
        records, runtime = _campaign(tmp_path, terminal=True)
        handle = _hold_lock(runtime, _self_owner())
        try:
            report = evaluate(records, runtime, marker_max_age_seconds=900)
        finally:
            handle.close()
        assert report.verdict in {"RUNNING-HEALTHY", "RUNNING-SUSPECT"}
        # Absent marker is not suspect; transport disposition may be in this
        # environment. The WEDGED/DEAD verdicts must NOT fire here.
        assert report.verdict_line.startswith("RUNNING-")

    def test_wedged_on_stalled_ledgers(self, tmp_path: Path) -> None:
        records, runtime = _campaign(tmp_path)
        old = datetime.now(timezone.utc).timestamp() - 3 * 3600
        for ledger in records.rglob("*.jsonl"):
            os.utime(ledger, (old, old))
        handle = _hold_lock(runtime, _self_owner())
        try:
            report = evaluate(records, runtime)
        finally:
            handle.close()
        assert report.verdict == "WEDGED"
        assert report.qualifier.startswith("no-ledger-write-")
        assert report.exit_code == 2

    def test_holding_on_quota_pause(self, tmp_path: Path) -> None:
        records, runtime = _campaign(tmp_path)
        _write_ledger(
            records / "operational" / "events.jsonl",
            [{"event_kind": "usage-pause", "created_at": "2026-08-24T00:00:00+00:00"}],
        )
        handle = _hold_lock(runtime, _self_owner())
        try:
            report = evaluate(records, runtime)
        finally:
            handle.close()
        assert report.verdict == "HOLDING"
        assert report.qualifier == "quota-pause"
        assert report.exit_code == 4

    def test_identity_mismatch_is_suspect(self, tmp_path: Path) -> None:
        records, runtime = _campaign(tmp_path, terminal=True)
        owner = _self_owner()
        owner["start_token"] = "boot-0-started-1"
        handle = _hold_lock(runtime, owner)
        try:
            report = evaluate(records, runtime)
        finally:
            handle.close()
        assert report.verdict == "RUNNING-SUSPECT"
        assert "lock-holder-identity-mismatch" in report.qualifier


class TestStatusMarker:
    def test_staleness_as_death(self, tmp_path: Path) -> None:
        runtime = tmp_path
        assert read_status_marker(runtime, max_age_seconds=900)[0] == "absent"
        path = driver_status_marker_path(runtime)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        path.write_text(
            json.dumps({"written_at": now.isoformat(), "leg_ordinal": 3, "arm": "author"})
            + "\n",
            encoding="utf-8",
        )
        assert read_status_marker(runtime, max_age_seconds=900, now=now)[0] == "fresh"
        later = now + timedelta(seconds=901)
        assert read_status_marker(runtime, max_age_seconds=900, now=later)[0] == "stale"
        path.write_text("not json", encoding="utf-8")
        assert read_status_marker(runtime, max_age_seconds=900)[0] == "stale"

    def test_writer_round_trips_through_the_reader(self, tmp_path: Path) -> None:
        from menagerie.crawler.driver_progress import write_driver_status_marker

        write_driver_status_marker(tmp_path, leg_ordinal=7, arm="gate-batch")
        state, payload = read_status_marker(tmp_path, max_age_seconds=900)
        assert state == "fresh"
        assert payload is not None
        assert payload["leg_ordinal"] == 7
        assert payload["arm"] == "gate-batch"

    def test_stale_marker_is_suspect_under_a_live_lock(self, tmp_path: Path) -> None:
        records, runtime = _campaign(tmp_path, terminal=True)
        path = driver_status_marker_path(runtime)
        path.parent.mkdir(parents=True, exist_ok=True)
        stale = datetime.now(timezone.utc) - timedelta(hours=3)
        path.write_text(
            json.dumps({"written_at": stale.isoformat(), "leg_ordinal": 1, "arm": "x"})
            + "\n",
            encoding="utf-8",
        )
        handle = _hold_lock(runtime, _self_owner())
        try:
            report = evaluate(records, runtime)
        finally:
            handle.close()
        assert report.verdict == "RUNNING-SUSPECT"
        assert "driver-status-marker-stale" in report.qualifier


class TestSelfTest:
    def test_self_test_passes(self, capsys) -> None:
        assert self_test() == 0
        assert "self-test OK" in capsys.readouterr().out


class TestPgrepBan:
    def test_no_pgrep_anywhere_in_production_or_tooling(self) -> None:
        """Process identity comes from lock holders, never name patterns.

        Scope: every tracked .py and .sh under menagerie/, tools/, scripts/.
        Watchers consume crawl_doctor exit codes or the monitor's --json
        snapshot; a pgrep here is a new instance of the class that nearly
        killed healthy legs twice in 200a.
        """

        offenders: list[str] = []
        for root in ("menagerie", "tools", "scripts"):
            base = REPO_ROOT / root
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if path.suffix not in {".py", ".sh"} or not path.is_file():
                    continue
                if "pgrep" in path.read_text(encoding="utf-8", errors="ignore"):
                    offenders.append(str(path.relative_to(REPO_ROOT)))
        allowed = {
            "menagerie/crawler/tests/test_crawl_doctor.py",
            # The two other anti-pgrep authorities: WS11's hygiene census and
            # the lock-based liveness tool document the banned pattern in prose.
            "menagerie/crawler/tests/test_repo_hygiene_census.py",
            "menagerie/crawler/tools/whoami_driver.py",
        }
        assert set(offenders) <= allowed, (
            f"pgrep-based process identity is banned (D-35): {offenders}"
        )


class TestWatcherPredicate:
    def test_json_snapshot_carries_the_machine_completion_predicate(
        self, tmp_path: Path
    ) -> None:
        from menagerie.crawler.tools import launch_monitor

        records, _ = _campaign(tmp_path, terminal=True)
        rc = launch_monitor.main(
            ["--records-root", str(records), "--once", "--json"]
        )
        assert rc == 0

    def test_snapshot_complete_derives_from_the_closed_terminal_set(
        self, tmp_path: Path
    ) -> None:
        from menagerie.crawler.tools.launch_monitor import build_snapshot

        records, _ = _campaign(tmp_path, terminal=True)
        snapshot = build_snapshot(records, check_disk=False)
        assert snapshot.complete is True
        assert snapshot.roster_source == "registry"
        off_vocab = {
            "stable_id": "m1",
            "record_seq": 2,
            "status": {"code": "runs-mostly", "kind": "runs"},
            "created_at": "2026-08-24T01:00:00+00:00",
        }
        with open(records / "models" / "current-shard.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(off_vocab) + "\n")
        snapshot = build_snapshot(records, check_disk=False)
        assert snapshot.complete is False
        assert snapshot.unrecognized == 1
