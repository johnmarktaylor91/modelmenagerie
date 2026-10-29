"""D-37: detection-regression + convergence replay over the 200a golden corpus.

200a's monitor was gated the way every instrument before it was: fixture-shaped
unit tests of counting math, never a production-shaped stream -- so C-4 and
NN-6 were caught on launch night, and the activity-vs-progress blindness was
never caught at all. The 200a records root is the asset those gates lacked: a
full production-shaped corpus containing every incident. This module makes it
a COMMITTED benchmark:

* :func:`project_records` reduces a records root to exactly the fields the
  monitor consumes (a ~1 MB committed fixture instead of ~50 MB of fat rows).
  The projection's fidelity is not assumed: the B2-X-V25 arm diffs REAL
  ``build_snapshot`` output over time-sliced full-fidelity ledgers against the
  projection at the same instants.
* :class:`ReplayHarness` replays a projected root through the real
  ``build_snapshot`` + ``evaluate_alerts`` at simulated instants (append-only
  time slicing; sim-time cooldowns), recording every alert with its instant.
* CI asserts, against the committed incident manifest: each known incident is
  NAMED within its bounded sim-latency; the monitor's numbers equal the close
  row's partition numbers at the close instant; and the alert stream is
  two-sided -- at most the message budget overall AND at least one message
  inside every incident window (a monitor silenced into its budget is as wrong
  as one that spams past it).

A monitor change that regresses detection latency on 200a fails here BEFORE it
ships -- the method change the whole observability area hangs on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.status import MACHINE_DEFERRAL_VALVES
from menagerie.crawler.tools.launch_monitor import (
    MonitorState,
    Snapshot,
    Thresholds,
    build_snapshot,
    deliverable_alerts,
)

REPLAY_FIXTURE_SCHEMA_VERSION = "menagerie.crawler.detection-replay-fixture.v1"


def _parse(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _project_model(row: Mapping[str, Any]) -> dict[str, Any]:
    status = row.get("status")
    return {
        "stable_id": row.get("stable_id"),
        "record_seq": row.get("record_seq"),
        "created_at": row.get("created_at"),
        "status": {
            "code": status.get("code") if isinstance(status, Mapping) else None,
            "kind": status.get("kind") if isinstance(status, Mapping) else None,
        },
    }


def _project_attempt(row: Mapping[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    return {
        "stable_id": row.get("stable_id"),
        "stage": row.get("stage"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "result": {
            "status": result.get("status") if isinstance(result, Mapping) else None
        },
    }


def _project_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    checker = row.get("checker")
    items = row.get("items")
    return {
        "gate_kind": row.get("gate_kind"),
        "items": [item for item in items if isinstance(item, str)]
        if isinstance(items, list)
        else [],
        "checker": {
            "finished_at": checker.get("finished_at")
            if isinstance(checker, Mapping)
            else None
        },
    }


def _project_operational(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    # EVERY operational row keeps its kind + instant: the projection-lag field
    # derives from all of them, and the fidelity arm caught the divergence
    # when unconsumed kinds were dropped. Details survive only for leg-config
    # (the monitor reads author_concurrency / author_wall_seconds there).
    kind = row.get("event_kind")
    projected: dict[str, Any] = {"event_kind": kind, "created_at": row.get("created_at")}
    if kind == "leg-config":
        details = row.get("details")
        if isinstance(details, Mapping):
            projected["details"] = {
                key: details[key]
                for key in ("author_concurrency", "author_wall_seconds")
                if key in details
            }
    return projected


def _project_spend(row: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    event = row.get("event")
    if event == "session_reserved":
        return {
            "event": event,
            "spend_id": row.get("spend_id"),
            "lane": row.get("lane"),
            "created_at": row.get("created_at"),
        }
    if event == "session_custody_released":
        return {
            "event": event,
            "spend_id": row.get("spend_id"),
            "created_at": row.get("created_at"),
        }
    if event == "session_settled":
        machine = row.get("machine")
        return {
            "event": event,
            "spend_id": row.get("spend_id"),
            "created_at": row.get("created_at"),
            "local_outcome": row.get("local_outcome"),
            "machine": {
                "wall_seconds": machine.get("wall_seconds")
                if isinstance(machine, Mapping)
                else None
            },
        }
    return None


def _project_park(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stable_id": row.get("stable_id"),
        "disposition": row.get("disposition"),
        "created_at": row.get("created_at"),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_bytes().split(b"\n"):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def project_records(src_root: Path, dest_root: Path) -> dict[str, str]:
    """Project one records root onto the monitor-consumed fields.

    Returns the per-file sha256 digests for the fixture manifest.
    """

    outputs: dict[str, list[dict[str, Any]]] = {}
    outputs["models/current-shard.jsonl"] = [
        _project_model(row)
        for row in _read_rows(src_root / "models" / "current-shard.jsonl")
    ]
    outputs["attempts/local.jsonl"] = [
        _project_attempt(row)
        for row in _read_rows(src_root / "attempts" / "local.jsonl")
    ]
    outputs["gates/current-shard.jsonl"] = [
        _project_gate(row)
        for row in _read_rows(src_root / "gates" / "current-shard.jsonl")
    ]
    outputs["operational/events.jsonl"] = [
        projected
        for row in _read_rows(src_root / "operational" / "events.jsonl")
        if (projected := _project_operational(row)) is not None
    ]
    outputs["operational/spend.jsonl"] = [
        projected
        for row in _read_rows(src_root / "operational" / "spend.jsonl")
        if (projected := _project_spend(row)) is not None
    ]
    for valve in MACHINE_DEFERRAL_VALVES:
        outputs[valve.relative_path] = [
            _project_park(row) for row in _read_rows(src_root / valve.relative_path)
        ]
    registry_rows = _read_rows(src_root / "intake" / "registered-rungs.jsonl")
    outputs["intake/registered-rungs.jsonl"] = [
        {
            "label": row.get("label"),
            "snapshot_id": row.get("snapshot_id"),
            "created_at": row.get("created_at"),
        }
        for row in registry_rows
    ]
    if registry_rows:
        snapshot_id = str(registry_rows[-1].get("snapshot_id", ""))
        outputs[f"intake/{snapshot_id}/items.jsonl"] = [
            {"stable_id": row.get("stable_id")}
            for row in _read_rows(src_root / "intake" / snapshot_id / "items.jsonl")
        ]
    digests: dict[str, str] = {}
    for relative, rows in outputs.items():
        path = dest_root / relative
        _write_rows(path, rows)
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


def _row_instant(relative: str, row: Mapping[str, Any]) -> Optional[datetime]:
    """Return the authoritative replay instant for one projected row."""

    if relative.startswith("gates/"):
        checker = row.get("checker")
        return _parse(checker.get("finished_at")) if isinstance(checker, Mapping) else None
    if relative.startswith("attempts/"):
        return _parse(row.get("finished_at")) or _parse(row.get("started_at"))
    return _parse(row.get("created_at"))


@dataclass
class ReplayAlert:
    """One alert the replayed monitor would have delivered."""

    at: datetime
    kind: str
    message: str


@dataclass
class ReplayHarness:
    """Replay a projected records root tick by tick under simulated time.

    Time slicing is append-only: rows enter the working root exactly when
    their authoritative instant passes, and the REAL ``build_snapshot`` +
    ``evaluate_alerts`` run at every tick. Cooldowns run on SIM time with the
    production default, and once-only state (completion, first-ETA, the
    restart-seed baselines) uses the production :class:`MonitorState`, so the
    replayed message stream is the stream a live monitor would have sent.
    """

    fixture_root: Path
    work_root: Path
    thresholds: Thresholds = field(default_factory=Thresholds)
    rolling_window: int = 20

    def __post_init__(self) -> None:
        self._pending: dict[str, list[tuple[Optional[datetime], dict[str, Any]]]] = {}
        self._written: dict[str, int] = {}
        timeless: dict[str, list[dict[str, Any]]] = {}
        for path in sorted(self.fixture_root.rglob("*.jsonl")):
            relative = path.relative_to(self.fixture_root).as_posix()
            rows = [(_row_instant(relative, row), row) for row in _read_rows(path)]
            if relative.startswith("intake/") and not relative.endswith(
                "registered-rungs.jsonl"
            ):
                timeless[relative] = [row for _instant, row in rows]
                continue
            rows.sort(key=lambda pair: pair[0] or datetime.min.replace(tzinfo=timezone.utc))
            self._pending[relative] = rows
            self._written[relative] = 0
            _write_rows(self.work_root / relative, [])
        for relative, timeless_rows in timeless.items():
            _write_rows(self.work_root / relative, timeless_rows)

    def _advance(self, instant: datetime) -> None:
        for relative, rows in self._pending.items():
            index = self._written[relative]
            fresh = []
            while index < len(rows):
                row_instant, row = rows[index]
                if row_instant is not None and row_instant > instant:
                    break
                fresh.append(row)
                index += 1
            if fresh:
                with open(self.work_root / relative, "a", encoding="utf-8") as handle:
                    for row in fresh:
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                self._written[relative] = index

    def replay(
        self,
        start: datetime,
        end: datetime,
        *,
        step: timedelta = timedelta(minutes=10),
        state: Optional[MonitorState] = None,
    ) -> tuple[list[ReplayAlert], Optional[Snapshot], MonitorState]:
        """Tick from ``start`` to ``end``; return alerts, final snapshot, state."""

        state = state or MonitorState()
        alerts: list[ReplayAlert] = []
        snapshot: Optional[Snapshot] = None
        instant = start
        while instant <= end:
            snapshot = self.snapshot_at(instant)
            if not state.baselines_seeded:
                state.baselines_seeded = True
                state.last_blocked_count = snapshot.blocked
                state.known_block_reasons = sorted(snapshot.block_reasons)
            for kind, message in deliverable_alerts(
                snapshot, state, self.thresholds, instant.timestamp()
            ):
                alerts.append(ReplayAlert(at=instant, kind=kind, message=message))
                if kind == "completion":
                    state.completion_sent = True
                if kind == "quota":
                    state.quota_pause_reported = True
                if kind == "block-tail":
                    state.last_blocked_count = snapshot.blocked
            instant += step
        return alerts, snapshot, state

    def snapshot_at(self, instant: datetime) -> Snapshot:
        """Advance the working root to ``instant`` and build the real snapshot."""

        self._advance(instant)
        return build_snapshot(
            self.work_root,
            now=instant,
            check_disk=False,
            rolling_window=self.rolling_window,
        )


def first_alert_in_window(
    alerts: Sequence[ReplayAlert],
    *,
    kind_prefix: str,
    window_start: datetime,
    window_end: datetime,
) -> Optional[ReplayAlert]:
    """Return the first matching alert inside one incident window."""

    for alert in alerts:
        if window_start <= alert.at <= window_end and alert.kind.startswith(kind_prefix):
            return alert
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m menagerie.crawler.tools.detection_replay",
        description="Project a records root into the committed replay fixture.",
    )
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args(argv)
    digests = project_records(args.src, args.dest)
    print(json.dumps(digests, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
