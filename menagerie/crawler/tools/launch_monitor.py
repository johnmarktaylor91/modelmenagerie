"""Read-only live monitor for a running step-200 crawl, reporting on a fixed cadence.

The monitor NEVER writes under the records root. It re-reads the durable ledgers on
every tick (crash-safe, idempotent) and persists only its own "last reported" markers
in a separate state file, so a restart neither double-sends nor loses the hourly beat.

Signal mapping (traced, not guessed)
------------------------------------
- Ledger layout: ``<records>/models/current-shard.jsonl``, ``<records>/attempts/local.jsonl``,
  ``<records>/gates/current-shard.jsonl`` (``reducer.default_ledger_paths``,
  ``menagerie/crawler/reducer.py:6272``); operational events at
  ``<records>/operational/events.jsonl`` (``recordio.operational_events_path``,
  ``menagerie/crawler/recordio.py:768``).
- Model DONE: a model-ledger head (max ``record_seq`` per ``stable_id`` — the reducer's
  replay order, ``menagerie/crawler/reducer.py:2830``) whose ``status.code`` is in the
  closed ``TERMINAL_STATUS_CODES`` vocabulary (``menagerie/crawler/constants.py:1050``)
  AND whose ``status.kind`` is in the closed ``StatusKind`` vocabulary. Every canonical
  model record carries ``status.code``/``status.kind`` (model schema; consumed identically
  by ``status.funnel_counts``, ``menagerie/crawler/status.py:881``). The terminal timestamp
  is the head's reducer-derived ``created_at``.
- Model UNRECOGNIZED: a head whose code or kind falls outside those closed vocabularies.
  BOTH coordinates are required because either one alone can be satisfied by a record the
  monitor cannot classify: a terminal-looking ``kind`` with an off-vocabulary ``code`` used
  to count silently as done (LP C-4), which is how a code this monitor has not learned, or
  a corrupted status, was reported to the operator as a completed model. Such a head is
  counted and disclosed in its own bucket — never as done, never laundered into the corrupt
  line count (the record decoded fine; it is its VOCABULARY that is unknown), and never as
  in-flight, since it does hold a head. It also blocks a COMPLETE announcement: declaring a
  crawl finished while some heads are unclassifiable is the same silent pass-through one
  level up.
- Model BLOCKED: either (a) a terminal head with ``status.kind == "deferred"`` (review-held
  ``deferred:*`` codes, incl. access-blocked/backend-parity/host-capability), or (b) a live
  machine-deferral valve row claiming a model that has NO head — every
  ``intake-extensions/*.jsonl`` park ledger in the single registry beside the
  terminal-partition authority (``status.MACHINE_DEFERRAL_VALVES``). Valve rows
  carry ``stable_id`` + ``disposition`` + ``created_at``. This monitor's park reading is the
  partition's fail-safe approximation: a park row counts only while the model has no head
  (revision-aware grant/refusal-park release semantics belong to ``status.partition_report``;
  a released park lands a head and drops out here automatically).
- Model IN-FLIGHT: a roster member with no head and no live park row. Members with
  attempt rows (``stable_id`` on attempts, attempt schema) are additionally counted as
  "started" in-flight.
- Roster: the registered rung — ``<records>/intake/registered-rungs.jsonl``
  (``rung_staging.rung_registry_path``, ``menagerie/crawler/rung_staging.py:594``) binds a
  ``label`` (e.g. ``step-200``) to a staged intake snapshot; members come from
  ``<records>/intake/<snapshot_id>/items.jsonl`` ``stable_id`` fields. Without a registry
  the roster degrades (disclosed) to all stable IDs seen in the models ledger.
- Throughput / ETA: terminal-head ``created_at`` timestamps, overall and over a rolling
  window; ETA = remaining / rolling rate. The first-rungs recalibrated ETA fires once
  ``done >= --first-eta-min`` (default 10).
- Quota: ``usage-pause`` / ``usage-resume`` operational events
  (``OperationalEventKind``, ``menagerie/crawler/constants.py:1705``): paused when the
  latest of the pair is a pause.
- Leg start: the newest ``leg-config`` operational event's ``created_at``
  (``OperationalEventKind.LEG_CONFIG``, appended exactly once per leg BEFORE any admission
  decision, ``menagerie/crawler/driver.py:5642``). A wake-noop invocation appends none and
  is not a leg.
- Stall: newest mtime across the writable ledgers (models/attempts/gates/operational +
  intake-extensions) older than ``--stall-minutes``; suppressed while quota-paused, and
  suppressed during the leg's FIRST author wall grant while zero models have started
  (NN-6). The first author session legitimately runs longer than the stall threshold
  before anything lands, so a 30-minute threshold under a 60-minute grant alarmed on
  every launch. The suppression is bounded by the grant (``--author-wall-seconds``,
  default ``AUTHOR_SESSION_WALL_SECONDS``) and requires a leg-config attestation: with no
  known leg age, or once any model has started, a stall alarms normally.
- Disk: ``shutil.disk_usage`` free space on the records root and any ``--cas-root``.

Ledger reads are deliberately tolerant: a live crawler may hold a torn (mid-append) final
line, so the reader skips an undecodable FINAL line silently and counts any other
undecodable line as corrupt — disclosed in the snapshot, never fatal. This is an advisory
monitor; canonical validation stays with ``recordio.scan_jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.constants import (
    AUTHOR_SESSION_WALL_SECONDS,
    TERMINAL_STATUS_CODES,
    OperationalEventKind,
    StatusKind,
)
from menagerie.crawler.identity import durable_fsync, fsync_directory
from menagerie.crawler.notification_outbox import (
    pending_critical_sends,
    read_outbox_rows,
)
from menagerie.crawler.operator_notify import evaluate_notification_transport
from menagerie.crawler.status import MACHINE_DEFERRAL_VALVES
from menagerie.local_config import local_value

#: Every machine-deferral park ledger, relative to the records root, DERIVED
#: from the single registry beside the partition authority
#: (:data:`menagerie.crawler.status.MACHINE_DEFERRAL_VALVES`). This used to be
#: seven hand-maintained string literals that omitted the environment-rebuild
#: valve, so a member claimed by a live rebuild row (m10502 in the 200a
#: records) was invisible here: never
#: counted parked, always counted in-flight, and the completion detector could
#: never read the campaign complete (OB-M1).
PARK_LEDGER_RELATIVE_PATHS: tuple[str, ...] = tuple(
    valve.relative_path for valve in MACHINE_DEFERRAL_VALVES
)

DEFAULT_SEND_SCRIPT = "~/.claude/scripts/send-to-jmt.sh"
STATE_SCHEMA_VERSION = "menagerie.crawler.launch-monitor-state.v1"
HEARTBEAT_SCHEMA_VERSION = "menagerie.crawler.launch-monitor-heartbeat.v1"

#: Consecutive failed ticks before the loop escalates a monitor-degraded page.
DEGRADED_ESCALATION_TICKS = 3

#: Closed heartbeat verdicts for the watcher-of-watcher.
HEARTBEAT_VERDICTS = frozenset({"fresh", "stale", "missing", "unreadable"})


class MonitorEvidenceUnavailable(RuntimeError):
    """A ledger EXISTS but cannot be read: typed, never a healthy empty read.

    D-33's fail-closed rule: a missing file is a legitimately empty ledger, but
    an ``OSError`` on a present one (permissions, I/O, unmounted volume) is
    UNAVAILABLE EVIDENCE. Reading it as zero rows would let the detectors
    report progress -- or completion -- over a ledger they never saw.
    """

    def __init__(self, path: Path, error: OSError) -> None:
        super().__init__(f"monitor_evidence_unavailable: {path}: {error}")
        self.path = path
        self.error = error


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: object) -> Optional[datetime]:
    """Parse one RFC 3339 timestamp, returning None for absent or invalid data."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def read_jsonl_tolerant(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read one append-only JSONL ledger without ever raising on live-write races.

    Returns the decoded object rows plus a corrupt-line count. An undecodable FINAL
    line is a torn tail mid-append and is skipped without counting; any other
    undecodable or non-object line counts as corrupt. A missing file reads as
    empty; any other ``OSError`` on a PRESENT path raises the typed
    :class:`MonitorEvidenceUnavailable` -- unreadable evidence must never read
    as a healthy zero (D-33 fail-closed rule).
    """

    try:
        data = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return [], 0
    except OSError as exc:
        raise MonitorEvidenceUnavailable(path, exc) from exc
    rows: list[dict[str, Any]] = []
    corrupt = 0
    lines = data.split(b"\n")
    last_content_index = max(
        (index for index, line in enumerate(lines) if line.strip()), default=-1
    )
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            if index != last_content_index:
                corrupt += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            corrupt += 1
    return rows, corrupt


@dataclass(frozen=True)
class RungRoster:
    """The staged rung the crawl is running: label, snapshot, and member IDs."""

    label: str
    snapshot_id: str
    member_ids: frozenset[str]
    source: str  # "registry" | "models-ledger"
    registered_at: Optional[str] = None


def load_rung_roster(records_root: Path, label: Optional[str] = None) -> Optional[RungRoster]:
    """Load the roster from the rung registry, preferring the newest matching row."""

    registry, _ = read_jsonl_tolerant(records_root / "intake" / "registered-rungs.jsonl")
    if label is not None:
        registry = [row for row in registry if row.get("label") == label]
    if not registry:
        return None
    row = registry[-1]
    snapshot_id = str(row.get("snapshot_id", ""))
    items, _ = read_jsonl_tolerant(records_root / "intake" / snapshot_id / "items.jsonl")
    members = frozenset(
        str(item["stable_id"]) for item in items if isinstance(item.get("stable_id"), str)
    )
    if not members:
        return None
    return RungRoster(
        label=str(row.get("label", "")),
        snapshot_id=snapshot_id,
        member_ids=members,
        source="registry",
        registered_at=(
            str(row["created_at"]) if isinstance(row.get("created_at"), str) else None
        ),
    )


def model_heads(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Project the models ledger onto its current head per stable ID (max record_seq)."""

    heads: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        stable_id = row.get("stable_id")
        if not isinstance(stable_id, str) or not stable_id:
            continue
        seq = row.get("record_seq")
        seq_value = seq if isinstance(seq, int) and not isinstance(seq, bool) else -1
        prior = heads.get(stable_id)
        prior_raw = prior.get("record_seq") if prior is not None else None
        prior_seq = (
            prior_raw if isinstance(prior_raw, int) and not isinstance(prior_raw, bool) else -1
        )
        if prior is None or seq_value >= prior_seq:
            heads[stable_id] = row
    return heads


def _status_field(record: Mapping[str, Any], key: str) -> str:
    status = record.get("status")
    if isinstance(status, Mapping):
        return str(status.get(key, ""))
    return ""


@dataclass(frozen=True)
class Snapshot:
    """One deterministic, re-derivable view of crawl progress."""

    generated_at: str
    roster_label: str
    roster_source: str
    roster_total: int
    done: int
    runs: int
    failed: int
    skipped: int
    deferred: int
    parked: int
    in_flight: int
    in_flight_started: int
    block_reasons: dict[str, int]
    corrupt_lines: int
    overall_rate_per_hour: Optional[float]
    rolling_rate_per_hour: Optional[float]
    eta_seconds: Optional[float]
    recent_error_rate: Optional[float]
    recent_error_window: int
    quota_paused: bool
    disk_free_gb: dict[str, float]
    last_activity_age_minutes: Optional[float]
    first_terminal_at: Optional[str] = None
    last_terminal_at: Optional[str] = None
    unrecognized: int = 0
    unrecognized_codes: dict[str, int] = field(default_factory=dict)
    leg_started_at: Optional[str] = None
    leg_age_minutes: Optional[float] = None
    #: Campaign notify-transport disposition (closed vocabulary: absent |
    #: discarding | live), from this process's environment resolution. A
    #: discarding or absent transport renders "notifications: DISCARDED" and
    #: alarms until the operator re-arms a real transport (OB-M3).
    notify_transport_disposition: str = "live"
    #: Critical-class sends recorded in the notification outbox without a
    #: transport-echoed settlement (D-38). ``None`` = no outbox observed
    #: (disclosed as n/a, never rendered as a healthy zero).
    notify_pending_critical: Optional[int] = None
    notify_oldest_pending_at: Optional[str] = None
    #: W18 rung-scoped epoch + net-vs-churn accounting (M5). ``done`` counts
    #: heads; ``net_terminals`` counts members whose FIRST terminal landed in
    #: THIS rung's epoch. 200a pinned ``done`` at 139 for 14 hours while the
    #: rate swung on churn (90/210 members re-minted revisions) -- rates and
    #: ETA derive from net first-terminals ONLY.
    rung_epoch_started_at: Optional[str] = None
    net_terminals: int = 0
    carried_terminals: int = 0
    churn_terminal_rows: int = 0
    regressed_members: int = 0
    net_rate_per_hour: Optional[float] = None
    #: Closed ETA basis vocabulary: an ETA is honest about WHY it exists or
    #: does not; a flat terminal count during calibration-withhold says so
    #: instead of extrapolating zero.
    eta_basis: str = "insufficient-evidence"
    #: W18 per-class flow funnel (M7): trailing-window arrivals per stage,
    #: derived from row timestamps only (replayable). Gate flows are BY KIND.
    flow_window_minutes: float = 60.0
    flows: dict[str, int] = field(default_factory=dict)
    stage_inputs: dict[str, int] = field(default_factory=dict)
    starved_stages: list[str] = field(default_factory=list)
    #: Leg-cycling alarm inputs (M9): >=3 leg-configs / 2h / 0 net terminals.
    leg_configs_in_cycle_window: int = 0
    leg_cycling: bool = False
    #: Whether the spend ledger has ANY rows: width and author-session flows
    #: are UNKNOWN (n/a, never zero) on a root whose custody machinery has not
    #: written yet -- 200a's spend ledger began mid-campaign, and reading its
    #: absence as width 0 false-paged the whole early era on replay.
    spend_instrumented: bool = False
    #: Effective width from spend-ledger leases, never a process listing (W18).
    effective_width: Optional[int] = None
    configured_width: Optional[int] = None
    #: The leg-config-attested author wall grant (M10); the operator flag
    #: stays as override, divergence disclosed by the render.
    author_wall_seconds_recorded: Optional[float] = None
    #: Cost accounting with the DECLARED denominator (OB-Q3):
    #: net-new-canonical-terminal-this-rung.
    spend_sessions_settled: Optional[int] = None
    spend_wall_hours: Optional[float] = None
    wall_hours_per_net_terminal: Optional[float] = None
    #: Authority root + projection epoch + lag on every status: a stale
    #: projection renders RED with its age, never as healthy zeros.
    authority_root: str = ""
    projection_newest_row_at: Optional[str] = None
    projection_lag_minutes: Optional[float] = None
    #: OB-M12 consumer half: count(leg-config) == count(leg-outcome) + running.
    #: ``not-instrumented`` until the driver emits leg-outcome events (WS3).
    leg_outcome_census: str = "not-instrumented"
    leg_configs_total: int = 0
    leg_outcomes_total: int = 0
    #: OB-M17: owed operator decisions age VISIBLY -- a park is a human's
    #: queue item, not a resting count.
    operator_owed_oldest_hours: Optional[float] = None

    @property
    def blocked(self) -> int:
        """Blocked = review-held deferred terminals + live machine parks."""

        return self.deferred + self.parked

    @property
    def complete(self) -> bool:
        """Complete only when every roster member is CLASSIFIED and terminal.

        An unrecognized head is why this is not just ``in_flight == 0``: such a
        member holds a head, so it is not in-flight, and announcing COMPLETE
        over it would report an unclassifiable status as a finished model.
        """

        return self.roster_total > 0 and self.in_flight == 0 and self.unrecognized == 0


def _rolling_rate(
    terminal_times: Sequence[datetime], window: int
) -> tuple[Optional[float], Optional[float]]:
    """Return (overall, rolling) terminals-per-hour from sorted terminal timestamps."""

    if len(terminal_times) < 2:
        return None, None
    total_span = (terminal_times[-1] - terminal_times[0]).total_seconds()
    overall = (len(terminal_times) - 1) / (total_span / 3600.0) if total_span > 0 else None
    tail = terminal_times[-min(len(terminal_times), max(window, 2)) :]
    tail_span = (tail[-1] - tail[0]).total_seconds()
    rolling = (len(tail) - 1) / (tail_span / 3600.0) if tail_span > 0 else None
    return overall, rolling


def _ledger_activity_age_minutes(records_root: Path, now: datetime) -> Optional[float]:
    """Age in minutes of the newest write across the crawl's durable ledgers."""

    candidates = [
        records_root / "models" / "current-shard.jsonl",
        records_root / "attempts" / "local.jsonl",
        records_root / "gates" / "current-shard.jsonl",
        records_root / "operational" / "events.jsonl",
        *(records_root / relative for relative in PARK_LEDGER_RELATIVE_PATHS),
    ]
    newest: Optional[float] = None
    for candidate in candidates:
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    if newest is None:
        return None
    return max(0.0, (now.timestamp() - newest) / 60.0)


def _leg_started_at(operational_rows: Sequence[Mapping[str, Any]]) -> Optional[datetime]:
    """Return the newest ``leg-config`` attestation's timestamp, if any.

    One event per leg, appended before any admission decision, so this is the
    leg's own start instant rather than an inference from file mtimes. The
    newest wins: a resumed campaign's later leg is the one now running.
    """

    newest: Optional[datetime] = None
    for row in operational_rows:
        if row.get("event_kind") != OperationalEventKind.LEG_CONFIG.value:
            continue
        created = _parse_timestamp(row.get("created_at"))
        if created is not None and (newest is None or created > newest):
            newest = created
    return newest


def _quota_paused(operational_rows: Sequence[Mapping[str, Any]]) -> bool:
    """Paused when the newest usage-pause/usage-resume event is a pause."""

    paused = False
    for row in operational_rows:
        kind = row.get("event_kind")
        if kind == OperationalEventKind.USAGE_PAUSE.value:
            paused = True
        elif kind == OperationalEventKind.USAGE_RESUME.value:
            paused = False
    return paused


def _rung_epoch(
    operational_rows: Sequence[Mapping[str, Any]],
    registry_created_at: Optional[datetime],
) -> Optional[datetime]:
    """Return the rung epoch: the first leg-config at/after rung registration.

    Falls back to the first leg-config overall when the registry row carries
    no timestamp; a root with no leg-config yet has no epoch (disclosed n/a).
    """

    instants = sorted(
        created
        for row in operational_rows
        if row.get("event_kind") == OperationalEventKind.LEG_CONFIG.value
        if (created := _parse_timestamp(row.get("created_at"))) is not None
    )
    if not instants:
        return None
    if registry_created_at is not None:
        for instant in instants:
            if instant >= registry_created_at:
                return instant
    return instants[0]


def _first_terminals(
    model_rows: Sequence[Mapping[str, Any]], member_ids: frozenset[str]
) -> tuple[dict[str, datetime], int]:
    """Return each member's FIRST terminal instant + total terminal row count."""

    first: dict[str, datetime] = {}
    terminal_rows = 0
    for row in model_rows:
        stable_id = row.get("stable_id")
        if not isinstance(stable_id, str) or stable_id not in member_ids:
            continue
        code = _status_field(row, "code")
        kind = _status_field(row, "kind")
        if code not in TERMINAL_STATUS_CODES or kind not in {
            status_kind.value for status_kind in StatusKind
        }:
            continue
        terminal_rows += 1
        created = _parse_timestamp(row.get("created_at"))
        if created is None:
            continue
        prior = first.get(stable_id)
        if prior is None or created < prior:
            first[stable_id] = created
    return first, terminal_rows


def _spend_lane_events(
    spend_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], list[tuple[datetime, str]], set[str], set[str], int, float]:
    """Project the spend ledger onto lanes, settlements, and open leases.

    Returns ``(lane_by_spend_id, settled_instants_by_lane, reserved_ids,
    closed_ids, settled_count, wall_seconds_total)``. Width derives from
    reserved-minus-released leases -- the spend ledger, never a process
    listing.
    """

    lane_by_id: dict[str, str] = {}
    settled: list[tuple[datetime, str]] = []
    reserved: set[str] = set()
    closed: set[str] = set()
    settled_count = 0
    wall_seconds = 0.0
    for row in spend_rows:
        event = row.get("event")
        spend_id = row.get("spend_id")
        if not isinstance(spend_id, str):
            continue
        if event == "session_reserved":
            lane = str(row.get("lane", ""))
            lane_by_id[spend_id] = lane
            reserved.add(spend_id)
        elif event in ("session_custody_released", "session_settled"):
            closed.add(spend_id)
            if event == "session_settled":
                settled_count += 1
                machine = row.get("machine")
                if isinstance(machine, Mapping):
                    seconds = machine.get("wall_seconds")
                    if isinstance(seconds, (int, float)) and not isinstance(
                        seconds, bool
                    ):
                        wall_seconds += float(seconds)
                created = _parse_timestamp(row.get("created_at"))
                if created is not None and row.get("local_outcome") == "completed":
                    settled.append((created, lane_by_id.get(spend_id, "")))
    return lane_by_id, settled, reserved, closed, settled_count, wall_seconds


def build_snapshot(
    records_root: Path,
    *,
    rung_label: Optional[str] = None,
    rolling_window: int = 20,
    error_window: int = 20,
    cas_roots: Sequence[Path] = (),
    now: Optional[datetime] = None,
    check_disk: bool = True,
    outbox_path: Optional[Path] = None,
    flow_window_minutes: float = 60.0,
) -> Snapshot:
    """Compute the current progress snapshot purely from the durable ledgers."""

    now = now or _utc_now()
    corrupt_total = 0

    notify_disposition = str(evaluate_notification_transport()["disposition"])
    pending_critical: Optional[int] = None
    oldest_pending_at: Optional[str] = None
    if outbox_path is not None:
        try:
            pending = pending_critical_sends(read_outbox_rows(outbox_path))
        except (OSError, ValueError):
            # An unreadable outbox is disclosed as unmeasured, never as zero.
            pending = None
        if pending is not None:
            pending_critical = len(pending)
            if pending:
                oldest_pending_at = pending[0].created_at

    model_rows, corrupt = read_jsonl_tolerant(records_root / "models" / "current-shard.jsonl")
    corrupt_total += corrupt
    heads = model_heads(model_rows)

    roster = load_rung_roster(records_root, rung_label)
    if roster is None:
        roster = RungRoster(
            label=rung_label or "<all-models-ledger-ids>",
            snapshot_id="",
            member_ids=frozenset(heads),
            source="models-ledger",
        )
    member_ids = roster.member_ids

    kind_counts = {kind.value: 0 for kind in StatusKind}
    block_reasons: dict[str, int] = {}
    unrecognized_codes: dict[str, int] = {}
    terminal_times: list[datetime] = []
    terminal_kinds_by_time: list[tuple[datetime, str]] = []
    done = 0
    for stable_id in member_ids:
        head = heads.get(stable_id)
        if head is None:
            continue
        code = _status_field(head, "code")
        kind = _status_field(head, "kind")
        if code not in TERMINAL_STATUS_CODES or kind not in kind_counts:
            # C-4: BOTH coordinates must be inside their closed vocabularies.
            # The old test was `code not in CODES and kind not in KINDS`, so a
            # head with a terminal-looking KIND and an off-vocabulary CODE fell
            # through and counted as done -- exactly the silent pass-through a
            # monitor exists to prevent. It is counted and named instead, and it
            # is not corruption: the line decoded, its vocabulary is unknown.
            label = code or "<absent>"
            unrecognized_codes[label] = unrecognized_codes.get(label, 0) + 1
            continue
        done += 1
        kind_counts[kind] += 1
        if kind in (StatusKind.DEFERRED.value, StatusKind.FAILED.value):
            block_reasons[code] = block_reasons.get(code, 0) + 1
        created = _parse_timestamp(head.get("created_at"))
        if created is not None:
            terminal_times.append(created)
            terminal_kinds_by_time.append((created, kind))

    parked_ids: dict[str, str] = {}
    park_created: dict[str, datetime] = {}
    for relative in PARK_LEDGER_RELATIVE_PATHS:
        rows, corrupt = read_jsonl_tolerant(records_root / relative)
        corrupt_total += corrupt
        for row in rows:
            park_id = row.get("stable_id")
            if not isinstance(park_id, str) or park_id not in member_ids:
                continue
            if park_id in heads:
                continue  # terminal head owns the story; park release is revision-aware
            parked_ids[park_id] = str(row.get("disposition", "parked"))
            created = _parse_timestamp(row.get("created_at"))
            if created is not None:
                prior = park_created.get(park_id)
                if prior is None or created < prior:
                    park_created[park_id] = created
    for disposition in parked_ids.values():
        block_reasons[disposition] = block_reasons.get(disposition, 0) + 1

    attempt_rows, corrupt = read_jsonl_tolerant(records_root / "attempts" / "local.jsonl")
    corrupt_total += corrupt
    attempted_ids = {
        row["stable_id"]
        for row in attempt_rows
        if isinstance(row.get("stable_id"), str)
    }

    gate_rows, corrupt = read_jsonl_tolerant(records_root / "gates" / "current-shard.jsonl")
    corrupt_total += corrupt
    spend_rows, corrupt = read_jsonl_tolerant(
        records_root / "operational" / "spend.jsonl"
    )
    corrupt_total += corrupt
    operational_rows, corrupt = read_jsonl_tolerant(
        records_root / "operational" / "events.jsonl"
    )
    corrupt_total += corrupt
    leg_started = _leg_started_at(operational_rows)

    in_flight_ids = member_ids - set(heads) - set(parked_ids)
    in_flight_started = len(in_flight_ids & attempted_ids)
    remaining = len(in_flight_ids)

    # --- W18 M5: rung-scoped epoch + net-vs-churn; rates NEVER count carried
    # or re-minted terminals (the 200a churn-as-progress class).
    epoch = _rung_epoch(operational_rows, _parse_timestamp(roster.registered_at))
    first_terminal, terminal_rows_total = _first_terminals(model_rows, member_ids)
    net_instants = sorted(
        instant
        for instant in first_terminal.values()
        if epoch is None or instant >= epoch
    )
    net_terminals = len(net_instants)
    carried_terminals = len(first_terminal) - net_terminals
    churn_terminal_rows = max(0, terminal_rows_total - len(first_terminal))
    regressed_members = sum(
        1
        for stable_id in first_terminal
        if stable_id in heads
        and (
            _status_field(heads[stable_id], "code") not in TERMINAL_STATUS_CODES
            or _status_field(heads[stable_id], "kind")
            not in {status_kind.value for status_kind in StatusKind}
        )
    )

    terminal_times.sort()
    terminal_kinds_by_time.sort(key=lambda pair: pair[0])
    overall_rate, rolling_rate = _rolling_rate(net_instants, rolling_window)
    eta_seconds: Optional[float] = None
    if remaining == 0:
        eta_seconds = 0.0
        eta_basis = "complete"
    elif rolling_rate is not None and rolling_rate > 0:
        eta_seconds = remaining / rolling_rate * 3600.0
        eta_basis = "net-terminal-rate"
    else:
        eta_basis = "insufficient-evidence"

    recent = terminal_kinds_by_time[-max(error_window, 1) :]
    recent_error_rate: Optional[float] = None
    if recent:
        failures = sum(1 for _at, kind in recent if kind == StatusKind.FAILED.value)
        recent_error_rate = failures / len(recent)

    # --- W18 M7: per-class flow funnel over the trailing window, from row
    # timestamps only (replayable under an injected `now`).
    window_start = now - timedelta(minutes=flow_window_minutes)
    _lane_by_id, settled_events, reserved_ids, closed_ids, settled_count, wall_seconds = (
        _spend_lane_events(spend_rows)
    )
    author_flow = sum(
        1
        for instant, lane in settled_events
        if lane == "author" and instant >= window_start
    )
    exec_instants: list[datetime] = []
    exec_members: set[str] = set()
    exec_succeeded_members: set[str] = set()
    for row in attempt_rows:
        stable_id = row.get("stable_id")
        if row.get("stage") != "forward" or not isinstance(stable_id, str):
            continue
        exec_members.add(stable_id)
        result = row.get("result")
        if isinstance(result, Mapping) and result.get("status") == "succeeded":
            exec_succeeded_members.add(stable_id)
        instant = _parse_timestamp(row.get("finished_at")) or _parse_timestamp(
            row.get("started_at")
        )
        if instant is not None:
            exec_instants.append(instant)
    exec_flow = sum(1 for instant in exec_instants if instant >= window_start)
    gated_members: set[str] = set()
    gate_flow_by_kind: dict[str, int] = {}
    gate_flow = 0
    for row in gate_rows:
        checker = row.get("checker")
        instant = (
            _parse_timestamp(checker.get("finished_at"))
            if isinstance(checker, Mapping)
            else None
        )
        items = row.get("items")
        if isinstance(items, list):
            gated_members.update(item for item in items if isinstance(item, str))
        if instant is not None and instant >= window_start:
            kind = str(row.get("gate_kind", "<unknown>"))
            gate_flow_by_kind[kind] = gate_flow_by_kind.get(kind, 0) + 1
            gate_flow += 1
    terminal_flow = sum(1 for instant in net_instants if instant >= window_start)

    flows: dict[str, int] = {
        "author-sessions": author_flow,
        "exec-attempts": exec_flow,
        "gate-adjudications": gate_flow,
        "net-terminals": terminal_flow,
    }
    for kind, count in sorted(gate_flow_by_kind.items()):
        flows[f"gate:{kind}"] = count

    author_completions_total = sum(1 for _t, lane in settled_events if lane == "author")
    stage_inputs: dict[str, int] = {
        "author": len(in_flight_ids - attempted_ids),
        # Aggregate PROXY: completed author sessions not yet reflected in any
        # exec attempt (the gate-wedge population is invisible member-by-member
        # in the records; the backlog difference is not). Bounded by the
        # in-flight population: author RETRY sessions would otherwise inflate
        # the backlog long after the members they retried went terminal.
        "exec": min(
            max(0, author_completions_total - len(exec_members)), remaining
        ),
        "gate": len(exec_succeeded_members - gated_members - set(heads)),
        "terminal": len((gated_members & in_flight_ids) - set(heads)),
    }
    flow_by_stage = {
        "author": author_flow,
        "exec": exec_flow,
        "gate": gate_flow,
        "terminal": terminal_flow,
    }
    quota_paused = _quota_paused(operational_rows)
    spend_instrumented = bool(spend_rows)
    starved_stages = [
        stage
        for stage in ("author", "exec", "gate", "terminal")
        if stage_inputs[stage] > 0
        and flow_by_stage[stage] == 0
        and remaining > 0
        and not quota_paused
        # Author flow comes from the spend ledger; with no spend rows at all
        # the flow is UNKNOWN, not zero, and unknown never alarms.
        and not (stage == "author" and not spend_instrumented)
    ]

    # --- W18 M9: leg-cycling (>=3 leg-configs / 2h / 0 net terminals).
    cycle_start = now - timedelta(hours=2)
    leg_configs_in_window = sum(
        1
        for row in operational_rows
        if row.get("event_kind") == OperationalEventKind.LEG_CONFIG.value
        and (created := _parse_timestamp(row.get("created_at"))) is not None
        and created >= cycle_start
    )
    leg_cycling = leg_configs_in_window >= 3 and not any(
        instant >= cycle_start for instant in net_instants
    )

    # --- OB-M12 consumer half: leg-config vs leg-outcome census.
    leg_configs_total = sum(
        1
        for row in operational_rows
        if row.get("event_kind") == OperationalEventKind.LEG_CONFIG.value
    )
    leg_outcomes_total = sum(
        1 for row in operational_rows if row.get("event_kind") == "leg-outcome"
    )
    if leg_outcomes_total == 0:
        leg_outcome_census = "not-instrumented"
    elif leg_configs_total - leg_outcomes_total in (0, 1):
        leg_outcome_census = "consistent"
    else:
        leg_outcome_census = "inconsistent"

    # --- OB-M17: oldest owed operator decision, computed here so no human
    # ever eyeballs timestamp arithmetic.
    operator_owed_oldest_hours = (
        max(
            0.0,
            (now - min(park_created.values())).total_seconds() / 3600.0,
        )
        if park_created
        else None
    )

    # --- W18 width from spend leases + M10 recorded author wall.
    effective_width = (
        len(
            {
                spend_id
                for spend_id in reserved_ids
                if _lane_by_id.get(spend_id) == "author"
            }
            - closed_ids
        )
        if spend_instrumented
        else None
    )
    configured_width: Optional[int] = None
    author_wall_recorded: Optional[float] = None
    newest_leg_config: Optional[Mapping[str, Any]] = None
    newest_leg_instant: Optional[datetime] = None
    for row in operational_rows:
        if row.get("event_kind") != OperationalEventKind.LEG_CONFIG.value:
            continue
        created = _parse_timestamp(row.get("created_at"))
        if created is not None and (
            newest_leg_instant is None or created > newest_leg_instant
        ):
            newest_leg_instant = created
            details = row.get("details")
            newest_leg_config = details if isinstance(details, Mapping) else None
    if newest_leg_config is not None:
        width = newest_leg_config.get("author_concurrency")
        if isinstance(width, int) and not isinstance(width, bool):
            configured_width = width
        wall = newest_leg_config.get("author_wall_seconds")
        if isinstance(wall, (int, float)) and not isinstance(wall, bool):
            author_wall_recorded = float(wall)

    # --- Authority root + projection epoch + lag on EVERY status.
    projection_instants: list[datetime] = list(net_instants)
    projection_instants.extend(exec_instants)
    projection_instants.extend(instant for instant, _lane in settled_events)
    for row in operational_rows:
        instant = _parse_timestamp(row.get("created_at"))
        if instant is not None:
            projection_instants.append(instant)
    newest_projection = max(projection_instants, default=None)

    disk_free_gb: dict[str, float] = {}
    if check_disk:
        for root_path in (records_root, *cas_roots):
            try:
                usage = shutil.disk_usage(root_path)
            except OSError:
                continue
            disk_free_gb[str(root_path)] = usage.free / 1e9

    return Snapshot(
        generated_at=now.isoformat(),
        roster_label=roster.label,
        roster_source=roster.source,
        roster_total=len(member_ids),
        done=done,
        runs=kind_counts[StatusKind.RUNS.value],
        failed=kind_counts[StatusKind.FAILED.value],
        skipped=kind_counts[StatusKind.SKIPPED.value],
        deferred=kind_counts[StatusKind.DEFERRED.value],
        parked=len(parked_ids),
        in_flight=remaining,
        in_flight_started=in_flight_started,
        block_reasons=dict(sorted(block_reasons.items())),
        corrupt_lines=corrupt_total,
        overall_rate_per_hour=overall_rate,
        rolling_rate_per_hour=rolling_rate,
        eta_seconds=eta_seconds,
        recent_error_rate=recent_error_rate,
        recent_error_window=len(recent),
        quota_paused=quota_paused,
        disk_free_gb=disk_free_gb,
        last_activity_age_minutes=_ledger_activity_age_minutes(records_root, now),
        first_terminal_at=terminal_times[0].isoformat() if terminal_times else None,
        last_terminal_at=terminal_times[-1].isoformat() if terminal_times else None,
        unrecognized=sum(unrecognized_codes.values()),
        unrecognized_codes=dict(sorted(unrecognized_codes.items())),
        leg_started_at=leg_started.isoformat() if leg_started is not None else None,
        leg_age_minutes=(
            None
            if leg_started is None
            else max(0.0, (now - leg_started).total_seconds() / 60.0)
        ),
        notify_transport_disposition=notify_disposition,
        notify_pending_critical=pending_critical,
        notify_oldest_pending_at=oldest_pending_at,
        spend_instrumented=spend_instrumented,
        rung_epoch_started_at=epoch.isoformat() if epoch is not None else None,
        net_terminals=net_terminals,
        carried_terminals=carried_terminals,
        churn_terminal_rows=churn_terminal_rows,
        regressed_members=regressed_members,
        net_rate_per_hour=rolling_rate,
        eta_basis=eta_basis,
        flow_window_minutes=flow_window_minutes,
        flows=flows,
        stage_inputs=stage_inputs,
        starved_stages=starved_stages,
        leg_configs_in_cycle_window=leg_configs_in_window,
        leg_cycling=leg_cycling,
        effective_width=effective_width,
        configured_width=configured_width,
        author_wall_seconds_recorded=author_wall_recorded,
        spend_sessions_settled=settled_count if spend_instrumented else None,
        spend_wall_hours=(wall_seconds / 3600.0) if spend_instrumented else None,
        wall_hours_per_net_terminal=(
            (wall_seconds / 3600.0) / net_terminals
            if net_terminals and spend_instrumented
            else None
        ),
        authority_root=str(records_root.resolve()),
        projection_newest_row_at=(
            newest_projection.isoformat() if newest_projection is not None else None
        ),
        projection_lag_minutes=(
            None
            if newest_projection is None
            else max(0.0, (now - newest_projection).total_seconds() / 60.0)
        ),
        leg_outcome_census=leg_outcome_census,
        leg_configs_total=leg_configs_total,
        leg_outcomes_total=leg_outcomes_total,
        operator_owed_oldest_hours=operator_owed_oldest_hours,
    )


def _format_eta(eta_seconds: Optional[float]) -> str:
    if eta_seconds is None:
        return "eta n/a"
    if eta_seconds <= 0:
        return "eta done"
    hours = int(eta_seconds // 3600)
    minutes = int((eta_seconds % 3600) // 60)
    return f"eta ~{hours}h{minutes:02d}m"


def render_snapshot(snapshot: Snapshot) -> str:
    """Render one terse ASCII progress line block."""

    rate = (
        f"{snapshot.rolling_rate_per_hour:.1f}/h"
        if snapshot.rolling_rate_per_hour is not None
        else "n/a"
    )
    lines = [
        (
            f"crawl {snapshot.roster_label}: {snapshot.done}/{snapshot.roster_total} done"
            f" ({snapshot.runs} runs, {snapshot.failed} failed, {snapshot.skipped} skipped,"
            f" {snapshot.deferred} deferred), {snapshot.in_flight} in-flight"
            f" ({snapshot.in_flight_started} started), {snapshot.parked} parked"
        ),
        (
            f"net this rung: {snapshot.net_terminals} terminals"
            f" ({snapshot.carried_terminals} carried,"
            f" {snapshot.churn_terminal_rows} churn rows,"
            f" {snapshot.regressed_members} regressed)"
        ),
        f"rate {rate} (net), {_format_eta(snapshot.eta_seconds)} (basis: {snapshot.eta_basis})",
    ]
    if snapshot.flows:
        window = int(snapshot.flow_window_minutes)
        lines.append(
            f"flows last {window}m: author {snapshot.flows.get('author-sessions', 0)}"
            f" -> exec {snapshot.flows.get('exec-attempts', 0)}"
            f" -> gates {snapshot.flows.get('gate-adjudications', 0)}"
            f" -> net terminals {snapshot.flows.get('net-terminals', 0)};"
            f" width {snapshot.effective_width if snapshot.effective_width is not None else 'n/a'}"
            f"/{snapshot.configured_width if snapshot.configured_width is not None else 'n/a'}"
        )
    if snapshot.starved_stages:
        inputs = ", ".join(
            f"{stage} (input {snapshot.stage_inputs.get(stage, 0)})"
            for stage in snapshot.starved_stages
        )
        lines.append(f"STARVED stages (input>0, flow=0): {inputs}")
    if snapshot.leg_cycling:
        lines.append(
            f"LEG-CYCLING: {snapshot.leg_configs_in_cycle_window} leg-configs in 2h"
            " with zero net terminals"
        )
    if snapshot.wall_hours_per_net_terminal is not None and snapshot.spend_wall_hours is not None:
        lines.append(
            f"spend: {snapshot.spend_sessions_settled} sessions,"
            f" {snapshot.spend_wall_hours:.1f} wall-h,"
            f" {snapshot.wall_hours_per_net_terminal:.2f} h per net terminal"
            " (denominator: net-new-canonical-terminal-this-rung)"
        )
    if snapshot.parked:
        oldest = (
            f"oldest {snapshot.operator_owed_oldest_hours:.0f}h"
            if snapshot.operator_owed_oldest_hours is not None
            else "oldest age unknown"
        )
        lines.append(f"operator-owed: {snapshot.parked} park(s) ({oldest})")
    if snapshot.leg_outcome_census == "inconsistent":
        lines.append(
            f"LEG CENSUS INCONSISTENT: {snapshot.leg_configs_total} leg-configs vs"
            f" {snapshot.leg_outcomes_total} leg-outcomes (+running)"
        )
    if snapshot.authority_root:
        lag = (
            f"{snapshot.projection_lag_minutes:.0f}m ago"
            if snapshot.projection_lag_minutes is not None
            else "NO ROWS (RED)"
        )
        lines.append(
            f"authority {snapshot.authority_root}; newest row {lag}"
        )
    if snapshot.block_reasons:
        reasons = ", ".join(
            f"{reason} x{count}" for reason, count in snapshot.block_reasons.items()
        )
        lines.append(f"block reasons: {reasons}")
    if snapshot.unrecognized:
        codes = ", ".join(
            f"{code} x{count}" for code, count in snapshot.unrecognized_codes.items()
        )
        lines.append(
            f"UNRECOGNIZED heads (not counted as done): {snapshot.unrecognized} -- {codes}"
        )
    if snapshot.notify_transport_disposition == "discarding":
        lines.append(
            "notifications: DISCARDED (transport is a no-op; re-arm a real"
            " transport with the operator's class-routing sign-off)"
        )
    elif snapshot.notify_transport_disposition == "absent":
        lines.append("notifications: NO TRANSPORT INSTALLED")
    if snapshot.notify_pending_critical:
        oldest = snapshot.notify_oldest_pending_at or "unknown"
        lines.append(
            f"notifications: {snapshot.notify_pending_critical} critical"
            f" send(s) pending without transport echo (oldest {oldest})"
        )
    for root_path, free in sorted(snapshot.disk_free_gb.items()):
        lines.append(f"disk {root_path}: {free:.1f} GB free")
    if snapshot.quota_paused:
        lines.append("QUOTA PAUSED (usage-pause without resume)")
    if snapshot.last_activity_age_minutes is not None:
        lines.append(f"last ledger write {snapshot.last_activity_age_minutes:.0f}m ago")
    if snapshot.corrupt_lines:
        lines.append(f"corrupt ledger lines: {snapshot.corrupt_lines}")
    return "\n".join(lines)


@dataclass
class MonitorState:
    """Durable "last reported" markers; everything else re-derives from the ledgers."""

    last_hourly_epoch: float = 0.0
    first_eta_sent: bool = False
    completion_sent: bool = False
    quota_pause_reported: bool = False
    last_blocked_count: int = 0
    known_block_reasons: list[str] = field(default_factory=list)
    last_alert_epochs: dict[str, float] = field(default_factory=dict)
    #: U-O6 restart-seed: a FRESH state must first observe the standing blocked
    #: population before block-tail growth means anything; otherwise the first
    #: tick after a restart alarms on every long-standing park.
    baselines_seeded: bool = False
    #: Last instant each alert condition was OBSERVED true (epoch seconds);
    #: the transitions-not-levels delivery layer derives onset/cleared from it.
    condition_last_seen: dict[str, float] = field(default_factory=dict)
    #: Monitor process generation: bumped once per --loop start, carried in the
    #: heartbeat so a forked or superseded monitor instance is distinguishable.
    generation: int = 0

    @classmethod
    def load(cls, path: Path) -> "MonitorState":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict) or data.get("schema_version") != STATE_SCHEMA_VERSION:
            return cls()
        return cls(
            last_hourly_epoch=float(data.get("last_hourly_epoch", 0.0)),
            first_eta_sent=bool(data.get("first_eta_sent", False)),
            completion_sent=bool(data.get("completion_sent", False)),
            quota_pause_reported=bool(data.get("quota_pause_reported", False)),
            last_blocked_count=int(data.get("last_blocked_count", 0)),
            known_block_reasons=[str(r) for r in data.get("known_block_reasons", [])],
            last_alert_epochs={
                str(k): float(v) for k, v in dict(data.get("last_alert_epochs", {})).items()
            },
            baselines_seeded=bool(data.get("baselines_seeded", False)),
            condition_last_seen={
                str(k): float(v)
                for k, v in dict(data.get("condition_last_seen", {})).items()
            },
            generation=int(data.get("generation", 0)),
        )

    def save(self, path: Path) -> None:
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "last_hourly_epoch": self.last_hourly_epoch,
            "first_eta_sent": self.first_eta_sent,
            "completion_sent": self.completion_sent,
            "quota_pause_reported": self.quota_pause_reported,
            "last_blocked_count": self.last_blocked_count,
            "known_block_reasons": sorted(self.known_block_reasons),
            "last_alert_epochs": self.last_alert_epochs,
            "baselines_seeded": self.baselines_seeded,
            "condition_last_seen": self.condition_last_seen,
            "generation": self.generation,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        with open(temp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temp, path)
        fsync_directory(path.parent)


def heartbeat_path(state_path: Path) -> Path:
    """Return the loop heartbeat path beside one monitor state file."""

    return state_path.with_name(f"heartbeat-{state_path.name}")


def write_heartbeat(
    state_path: Path,
    *,
    tick_ok: bool,
    reason: Optional[str],
    consecutive_failures: int,
    tick_counter: int,
    tick_duration_seconds: float,
    generation: int,
    poll_seconds: float,
    now: Optional[datetime] = None,
) -> None:
    """Write the loop-body heartbeat atomically; failures are never fatal.

    The LOOP body writes this every iteration -- including iterations whose
    tick raised -- so a dead OR wedged monitor shows as a stale heartbeat to
    the watcher-of-watcher, and a failing-but-alive monitor shows its typed
    reason instead of silence.
    """

    path = heartbeat_path(state_path)
    payload = {
        "schema_version": HEARTBEAT_SCHEMA_VERSION,
        "written_at": (now or _utc_now()).isoformat(),
        "pid": os.getpid(),
        "tick_ok": tick_ok,
        "reason": reason,
        "consecutive_failures": consecutive_failures,
        "tick_counter": tick_counter,
        "tick_duration_seconds": round(tick_duration_seconds, 3),
        "generation": generation,
        "poll_seconds": poll_seconds,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    except OSError as exc:  # pragma: no cover -- disk-full etc.; keep monitoring
        print(f"[launch-monitor] heartbeat write failed: {exc}", file=sys.stderr)


def check_heartbeat(
    state_path: Path, *, max_age_seconds: float, now: Optional[datetime] = None
) -> tuple[str, str]:
    """Watcher-of-watcher verdict over one heartbeat: closed vocabulary.

    Returns ``(verdict, detail)`` with verdict in :data:`HEARTBEAT_VERDICTS`.
    A missing or unreadable heartbeat is its own verdict, never "fresh": the
    instrument's absence is exactly what this check exists to catch.
    """

    now = now or _utc_now()
    path = heartbeat_path(state_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        written = _parse_timestamp(payload.get("written_at"))
    except FileNotFoundError:
        return "missing", f"no heartbeat at {path}"
    except (OSError, ValueError):
        return "unreadable", f"unreadable heartbeat at {path}"
    if written is None:
        return "unreadable", f"heartbeat at {path} carries no written_at"
    age = (now - written).total_seconds()
    if age > max_age_seconds:
        return "stale", f"heartbeat is {age:.0f}s old (max {max_age_seconds:.0f}s)"
    detail = f"heartbeat is {age:.0f}s old"
    if payload.get("tick_ok") is False:
        detail += f"; last tick FAILED ({payload.get('reason')})"
    return "fresh", detail


def acquire_monitor_lock(state_path: Path):
    """Take the single-monitor exclusive lock, or return None when held.

    Two monitors over one state file double-send every alert and race the
    state saves; the loop refuses to start as a second instance.
    """

    import fcntl

    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


@dataclass(frozen=True)
class Thresholds:
    stall_minutes: float = 30.0
    disk_floor_gb: float = 20.0
    error_rate: float = 0.5
    error_min_terminals: int = 5
    block_tail_growth: int = 3
    first_eta_min: int = 10
    alert_cooldown_seconds: float = 1800.0
    #: The leg's first author wall grant. A launch is legitimately silent for
    #: this long before the first author session lands anything, so it bounds
    #: the NN-6 stall suppression -- and only bounds it: past the grant, or once
    #: any model has started, a stall alarms normally.
    author_wall_seconds: float = float(AUTHOR_SESSION_WALL_SECONDS)
    #: Transitions-not-levels: a STANDING condition re-pages on this interval,
    #: not on every cooldown expiry -- a failure-majority funnel would
    #: otherwise page error-rate twice an hour for days (the ignored-alarm
    #: failure mode is worse than no alarm).
    standing_realert_seconds: float = 4 * 3600.0
    #: OB-M17: how long an owed operator decision may age before it pages.
    operator_owed_hours: float = 12.0
    #: Hysteresis before a condition counts as CLEARED: a flapping flow does
    #: not re-page on every flap, and a genuinely new incident after a real
    #: recovery pages fresh (FF-7).
    condition_clear_seconds: float = 3600.0


def within_first_author_wall(snapshot: Snapshot, thresholds: Thresholds) -> bool:
    """Return whether a silent leg is still inside its first author wall grant.

    All three conditions are required, and each closes a way the suppression
    could hide a real stall: zero models started (nothing has proven the lane
    works yet), a KNOWN leg age (no leg-config attestation grants no licence),
    and that age under the grant (the window is bounded, not open-ended).

    The grant value prefers the LEG-CONFIG-ATTESTED ``author_wall_seconds``
    (M10): a monitor launched with a different flag than the crawl silently
    mis-times this grace, so the recorded value wins and the flag stays as
    the fallback/override for roots whose legs predate the field.
    """

    grant = (
        snapshot.author_wall_seconds_recorded
        if snapshot.author_wall_seconds_recorded is not None
        else thresholds.author_wall_seconds
    )
    return (
        snapshot.done == 0
        and snapshot.in_flight_started == 0
        and snapshot.leg_age_minutes is not None
        and snapshot.leg_age_minutes * 60.0 < grant
    )


def evaluate_alerts(
    snapshot: Snapshot, state: MonitorState, thresholds: Thresholds
) -> list[tuple[str, str]]:
    """Return (kind, message) problem/completion alerts implied by one snapshot.

    Pure over its inputs; cooldown and once-only suppression live in the caller's
    state handling so this stays unit-testable.
    """

    alerts: list[tuple[str, str]] = []
    if snapshot.complete and not state.completion_sent:
        alerts.append(
            (
                "completion",
                f"crawl {snapshot.roster_label} COMPLETE: {snapshot.done}/"
                f"{snapshot.roster_total} terminal, {snapshot.parked} parked"
                f" ({snapshot.runs} runs, {snapshot.failed} failed,"
                f" {snapshot.skipped} skipped, {snapshot.deferred} deferred)",
            )
        )
    if snapshot.quota_paused and not state.quota_pause_reported:
        alerts.append(("quota", "crawl QUOTA-PAUSED: usage-pause event without a resume"))
    if snapshot.unrecognized:
        codes = ", ".join(
            f"{code} x{count}" for code, count in snapshot.unrecognized_codes.items()
        )
        alerts.append(
            (
                "unrecognized",
                f"crawl UNRECOGNIZED HEADS: {snapshot.unrecognized} model head(s) carry a"
                f" status outside the closed terminal vocabulary and are NOT counted as"
                f" done -- {codes}",
            )
        )
    if (
        not snapshot.complete
        and not snapshot.quota_paused
        and snapshot.in_flight > 0
        and snapshot.last_activity_age_minutes is not None
        and snapshot.last_activity_age_minutes >= thresholds.stall_minutes
        # NN-6: the first author session outlives the stall threshold, so a
        # launch that has not started a model yet is quiet, not stalled.
        and not within_first_author_wall(snapshot, thresholds)
    ):
        alerts.append(
            (
                "stall",
                f"crawl STALL: no ledger write for"
                f" {snapshot.last_activity_age_minutes:.0f}m"
                f" ({snapshot.in_flight} in-flight)",
            )
        )
    for root_path, free in sorted(snapshot.disk_free_gb.items()):
        if free < thresholds.disk_floor_gb:
            alerts.append(
                (
                    "disk",
                    f"crawl DISK PRESSURE: {root_path} has {free:.1f} GB free"
                    f" (floor {thresholds.disk_floor_gb:.0f} GB)",
                )
            )
    if (
        snapshot.recent_error_rate is not None
        and snapshot.recent_error_window >= thresholds.error_min_terminals
        and snapshot.recent_error_rate >= thresholds.error_rate
    ):
        alerts.append(
            (
                "error-rate",
                f"crawl ERROR SPIKE: {snapshot.recent_error_rate:.0%} failed of last"
                f" {snapshot.recent_error_window} terminals",
            )
        )
    for stage in snapshot.starved_stages:
        # Per-stage alert identity => per-instance cooldown: a second,
        # different starvation inside one cooldown window still pages.
        alerts.append(
            (
                f"stage-starvation:{stage}",
                f"crawl STAGE STARVED: {stage} has input"
                f" {snapshot.stage_inputs.get(stage, 0)} and ZERO flow in the"
                f" last {int(snapshot.flow_window_minutes)}m"
                " (activity without progress -- the 200a wedge class)",
            )
        )
    if snapshot.leg_cycling:
        alerts.append(
            (
                "leg-cycling",
                f"crawl LEG-CYCLING: {snapshot.leg_configs_in_cycle_window}"
                " leg-configs within 2h and zero net terminals -- identical"
                " leg deaths are burning wall-clock (legs 31-36 class)",
            )
        )
    if (
        snapshot.effective_width == 0
        and snapshot.configured_width
        and not snapshot.complete
        and not snapshot.quota_paused
        and snapshot.stage_inputs.get("author", 0) > 0
        and not within_first_author_wall(snapshot, thresholds)
    ):
        alerts.append(
            (
                "width-collapse",
                f"crawl WIDTH COLLAPSE: 0 open author leases against a"
                f" configured width of {snapshot.configured_width}"
                " (from spend-ledger leases)",
            )
        )
    if (
        snapshot.operator_owed_oldest_hours is not None
        and snapshot.operator_owed_oldest_hours >= thresholds.operator_owed_hours
    ):
        alerts.append(
            (
                "operator-owed",
                f"crawl DECISIONS OWED: {snapshot.parked} park(s) await an"
                f" operator, oldest {snapshot.operator_owed_oldest_hours:.0f}h"
                " -- no park may reach close without a decisions-owed text"
                " at least one leg earlier",
            )
        )
    if snapshot.leg_outcome_census == "inconsistent":
        alerts.append(
            (
                "leg-census",
                f"crawl LEG CENSUS INCONSISTENT: {snapshot.leg_configs_total}"
                f" leg-configs vs {snapshot.leg_outcomes_total} leg-outcomes"
                " -- a leg ended without recording how (OB-M12)",
            )
        )
    if snapshot.notify_transport_disposition != "live":
        alerts.append(
            (
                "notify-transport",
                "crawl NOTIFICATIONS "
                + (
                    "DISCARDED: the notify transport is a no-op"
                    if snapshot.notify_transport_disposition == "discarding"
                    else "UNDELIVERABLE: no notify transport is installed"
                )
                + " -- critical pages are being lost until an operator re-arms"
                " a real transport",
            )
        )
    if snapshot.notify_pending_critical:
        alerts.append(
            (
                "notify-pending",
                f"crawl NOTIFICATIONS PENDING: {snapshot.notify_pending_critical}"
                " critical send(s) lack a transport-echoed receipt"
                f" (oldest {snapshot.notify_oldest_pending_at or 'unknown'})",
            )
        )
    growth = snapshot.blocked - state.last_blocked_count
    if growth >= thresholds.block_tail_growth:
        alerts.append(
            (
                "block-tail",
                f"crawl BLOCK TAIL GROWING: +{growth} blocked since last report"
                f" (now {snapshot.blocked}: {snapshot.deferred} deferred,"
                f" {snapshot.parked} parked)",
            )
        )
    return alerts


def deliverable_alerts(
    snapshot: Snapshot,
    state: MonitorState,
    thresholds: Thresholds,
    now_epoch: float,
) -> list[tuple[str, str]]:
    """Transitions-not-levels delivery over :func:`evaluate_alerts`.

    ``evaluate_alerts`` names every condition CURRENTLY true; this layer
    decides which page: a condition pages at ONSET (absent for at least the
    clear hysteresis), then re-pages only after ``standing_realert_seconds``
    while it persists. Clearing resets the cycle, so a second, different
    incident after a real recovery pages fresh instead of being swallowed by
    a cooldown (FF-7) -- and a standing condition never trains the reader to
    ignore the channel by paging on every cooldown expiry.

    ONE implementation serves the live tick and the D-37 replay harness, so
    the replayed message stream is the stream a live monitor would send.
    """

    fired: list[tuple[str, str]] = []
    for kind, message in evaluate_alerts(snapshot, state, thresholds):
        last_seen = state.condition_last_seen.get(kind)
        last_fired = state.last_alert_epochs.get(kind, 0.0)
        onset = (
            last_seen is None
            or now_epoch - last_seen > thresholds.condition_clear_seconds
        )
        state.condition_last_seen[kind] = now_epoch
        if onset:
            if now_epoch - last_fired >= thresholds.alert_cooldown_seconds:
                fired.append((kind, message))
                state.last_alert_epochs[kind] = now_epoch
        elif now_epoch - last_fired >= thresholds.standing_realert_seconds:
            fired.append((kind, f"[still] {message}"))
            state.last_alert_epochs[kind] = now_epoch
    return fired


def new_block_reasons(snapshot: Snapshot, state: MonitorState) -> list[str]:
    """Block reasons present now that were absent at the last report."""

    return sorted(set(snapshot.block_reasons) - set(state.known_block_reasons))


def _send(message: str, script: Optional[str]) -> None:
    """Deliver one message via send-to-jmt.sh; delivery failure is never fatal."""

    print(message, flush=True)
    if script is None:
        return
    script_path = Path(script).expanduser()
    if not script_path.exists():
        print(f"[launch-monitor] send script missing: {script_path}", file=sys.stderr)
        return
    try:
        subprocess.run([str(script_path), message], check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[launch-monitor] send failed: {exc}", file=sys.stderr)


def _monitor_state_dir() -> str:
    """The launch monitor's state folder (``menagerie.local_config`` key LAUNCH_MONITOR_STATE_DIR)."""
    return local_value("LAUNCH_MONITOR_STATE_DIR", "~/.local/state/menagerie/launch-monitor")


def default_state_path(records_root: Path) -> Path:
    digest = hashlib.sha256(str(records_root.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(_monitor_state_dir()).expanduser() / f"state-{digest}.json"


def _tick(
    args: argparse.Namespace,
    thresholds: Thresholds,
    state: MonitorState,
    send_script: Optional[str],
) -> Snapshot:
    """One idempotent monitoring pass: snapshot, alerts, first-ETA, hourly beat."""

    now_epoch = time.time()
    snapshot = build_snapshot(
        args.records_root,
        rung_label=args.rung_label,
        rolling_window=args.rolling_window,
        error_window=args.error_window,
        cas_roots=tuple(args.cas_root or ()),
        outbox_path=args.outbox,
    )

    if not state.baselines_seeded:
        # U-O6 restart-seed: the first observation BASELINES the standing
        # blocked population and known reasons; a fresh state otherwise reads
        # every long-standing park as brand-new growth and alarms on tick one.
        state.baselines_seeded = True
        state.last_blocked_count = snapshot.blocked
        state.known_block_reasons = sorted(snapshot.block_reasons)

    for kind, message in deliverable_alerts(snapshot, state, thresholds, now_epoch):
        _send(message, send_script)
        if kind == "completion":
            state.completion_sent = True
        if kind == "quota":
            state.quota_pause_reported = True
        if kind == "block-tail":
            state.last_blocked_count = snapshot.blocked
    if not snapshot.quota_paused:
        state.quota_pause_reported = False

    if not state.first_eta_sent and snapshot.net_terminals >= thresholds.first_eta_min:
        # M5: the first-ETA gate counts NET terminals this rung; carried
        # terminals from a prior rung can never fire it on tick one.
        _send(
            "first-rungs recalibrated ETA -- " + render_snapshot(snapshot),
            send_script,
        )
        state.first_eta_sent = True

    if now_epoch - state.last_hourly_epoch >= args.interval:
        fresh_reasons = new_block_reasons(snapshot, state)
        message = render_snapshot(snapshot)
        if fresh_reasons:
            message += "\nNEW block reasons: " + ", ".join(fresh_reasons)
        _send(message, send_script)
        state.last_hourly_epoch = now_epoch
        state.known_block_reasons = sorted(
            set(state.known_block_reasons) | set(snapshot.block_reasons)
        )
        state.last_blocked_count = snapshot.blocked

    state.save(args.state_file)
    return snapshot


def _plist(payload: Mapping[str, Any]) -> str:
    import plistlib

    return plistlib.dumps(dict(payload), fmt=plistlib.FMT_XML).decode("utf-8")


def render_monitor_launchd_plist(args: argparse.Namespace) -> str:
    """Render the KeepAlive launchd agent for the standing monitor loop.

    launchd owns the lifecycle (D-33): the 200a monitor was a shell background
    job, killed at the storm hold and restarted by hand legs later. KeepAlive
    restarts it on ANY exit except the deliberate completion exit 0.
    """

    argv = [
        sys.executable,
        "-m",
        "menagerie.crawler.tools.launch_monitor",
        "--records-root",
        str(Path(args.records_root).resolve()),
        "--loop",
        "--poll",
        str(args.poll),
        "--interval",
        str(args.interval),
        "--state-file",
        str(Path(args.state_file).resolve()),
    ]
    if args.outbox is not None:
        argv.extend(("--outbox", str(Path(args.outbox).resolve())))
    return _plist(
        {
            "Label": "com.menagerie.launch-monitor",
            "ProgramArguments": argv,
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "WorkingDirectory": str(Path.cwd()),
        }
    )


def render_watchdog_launchd_plist(args: argparse.Namespace) -> str:
    """Render the watcher-of-watcher agent: heartbeat freshness on an interval.

    Runs ``--check-heartbeat --send`` every 3x poll, so a dead OR wedged
    monitor pages within one interval instead of reading as a quiet campaign.
    """

    max_age = args.max_age_seconds or 3.0 * args.poll
    argv = [
        sys.executable,
        "-m",
        "menagerie.crawler.tools.launch_monitor",
        "--records-root",
        str(Path(args.records_root).resolve()),
        "--check-heartbeat",
        "--max-age-seconds",
        str(max_age),
        "--send",
        "--state-file",
        str(Path(args.state_file).resolve()),
    ]
    return _plist(
        {
            "Label": "com.menagerie.launch-monitor-watchdog",
            "ProgramArguments": argv,
            "RunAtLoad": True,
            "StartInterval": int(max_age),
            "WorkingDirectory": str(Path.cwd()),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--rung-label", default=None, help="e.g. step-200; default newest rung")
    parser.add_argument("--cas-root", action="append", type=Path, default=None)
    parser.add_argument(
        "--outbox",
        type=Path,
        default=None,
        help="notification-outbox ledger to watch for unechoed critical sends",
    )
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument(
        "--json",
        action="store_true",
        help="with --once: print the machine-readable snapshot for watchers",
    )
    parser.add_argument("--loop", action="store_true", help="run the standing reporter")
    parser.add_argument(
        "--check-heartbeat",
        action="store_true",
        help=(
            "watcher-of-watcher: verdict on the loop heartbeat's freshness"
            " (exit 0 fresh, 1 stale, 2 missing/unreadable)"
        ),
    )
    parser.add_argument(
        "--max-age-seconds",
        type=float,
        default=None,
        help="heartbeat freshness bound for --check-heartbeat (default 3x --poll)",
    )
    parser.add_argument(
        "--emit-launchd-plists",
        action="store_true",
        help="print the monitor KeepAlive plist and the heartbeat-watchdog plist",
    )
    parser.add_argument("--interval", type=float, default=3600.0, help="hourly beat seconds")
    parser.add_argument("--poll", type=float, default=60.0, help="loop poll seconds")
    parser.add_argument("--send", action="store_true", help="deliver via send script in --once")
    parser.add_argument("--send-script", default=DEFAULT_SEND_SCRIPT)
    parser.add_argument("--no-send", action="store_true", help="print-only in --loop")
    parser.add_argument("--state-file", type=Path, default=None)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--disk-floor-gb", type=float, default=20.0)
    parser.add_argument("--error-rate", type=float, default=0.5)
    parser.add_argument("--error-window", type=int, default=20)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--first-eta-min", type=int, default=10)
    parser.add_argument(
        "--author-wall-seconds",
        type=float,
        default=float(AUTHOR_SESSION_WALL_SECONDS),
        help=(
            "the leg's first author wall grant; bounds the launch-window stall "
            "suppression. Pass the same value the crawl was launched with."
        ),
    )
    parser.add_argument("--block-tail-growth", type=int, default=3)
    parser.add_argument("--alert-cooldown", type=float, default=1800.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    modes = [args.once, args.loop, args.check_heartbeat, args.emit_launchd_plists]
    if sum(1 for mode in modes if mode) != 1:
        print(
            "launch-monitor: pass exactly one of --once / --loop /"
            " --check-heartbeat / --emit-launchd-plists",
            file=sys.stderr,
        )
        return 2
    if args.state_file is None:
        args.state_file = default_state_path(args.records_root)
    if args.emit_launchd_plists:
        print(render_monitor_launchd_plist(args))
        print(render_watchdog_launchd_plist(args))
        return 0
    if args.check_heartbeat:
        max_age = args.max_age_seconds or 3.0 * args.poll
        verdict, detail = check_heartbeat(args.state_file, max_age_seconds=max_age)
        print(f"launch-monitor heartbeat: {verdict} -- {detail}")
        if verdict == "fresh":
            return 0
        if args.send:
            _send(
                f"crawl MONITOR {verdict.upper()}: {detail} -- the launch"
                " monitor is not proving liveness; restart it",
                args.send_script,
            )
        return 1 if verdict == "stale" else 2
    thresholds = Thresholds(
        stall_minutes=args.stall_minutes,
        disk_floor_gb=args.disk_floor_gb,
        error_rate=args.error_rate,
        block_tail_growth=args.block_tail_growth,
        first_eta_min=args.first_eta_min,
        alert_cooldown_seconds=args.alert_cooldown,
        author_wall_seconds=args.author_wall_seconds,
    )

    if args.once:
        try:
            snapshot = build_snapshot(
                args.records_root,
                rung_label=args.rung_label,
                rolling_window=args.rolling_window,
                error_window=args.error_window,
                cas_roots=tuple(args.cas_root or ()),
                outbox_path=args.outbox,
            )
        except MonitorEvidenceUnavailable as exc:
            # Fail-closed: unreadable evidence is a typed refusal, never a
            # rendered snapshot full of healthy-looking zeros.
            print(f"launch-monitor: {exc}", file=sys.stderr)
            return 3
        if args.json:
            # Machine surface for watchers: the completion predicate is
            # `complete and roster_source == "registry"`, never a string
            # match on the rendered lines (FF-6: a wording change silently
            # killed the 200a completion watcher's only trigger).
            from dataclasses import asdict

            print(json.dumps(asdict(snapshot), sort_keys=True))
            return 0
        message = render_snapshot(snapshot)
        _send(message, args.send_script if args.send else None)
        return 0

    send_script = None if args.no_send else args.send_script
    lock_handle = acquire_monitor_lock(args.state_file)
    if lock_handle is None:
        print(
            "launch-monitor: another monitor instance holds the state lock;"
            " refusing to double-send",
            file=sys.stderr,
        )
        return 2
    try:
        state = MonitorState.load(args.state_file)
        state.generation += 1
        return run_loop(args, thresholds, state, send_script)
    finally:
        lock_handle.close()


def run_loop(
    args: argparse.Namespace,
    thresholds: Thresholds,
    state: MonitorState,
    send_script: Optional[str],
    *,
    max_ticks: Optional[int] = None,
    sleep_fn=time.sleep,
) -> int:
    """The standing reporter loop: survives ANY tick failure, always heartbeats.

    D-33/UU-7: only ``KeyboardInterrupt``/``SystemExit`` escape. A failed tick
    is typed (``monitor_evidence_unavailable`` for an unreadable ledger,
    ``DEGRADED:<exception>`` for anything unknown), counted, and escalated to
    a monitor-degraded page after :data:`DEGRADED_ESCALATION_TICKS`
    consecutive failures -- while the loop keeps ticking. The heartbeat is
    written by THIS body every iteration, success or failure, so a dead or
    wedged monitor is a stale heartbeat to the watcher-of-watcher rather than
    plausible silence (the 200a monitor died on the first unexpected row shape
    and stayed dead for legs).
    """

    consecutive_failures = 0
    tick_counter = 0
    while True:
        tick_counter += 1
        started = time.monotonic()
        reason: Optional[str] = None
        tick_snapshot: Optional[Snapshot] = None
        try:
            tick_snapshot = _tick(args, thresholds, state, send_script)
            consecutive_failures = 0
        except MonitorEvidenceUnavailable as exc:
            consecutive_failures += 1
            reason = str(exc)
            print(f"[launch-monitor] {exc}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 -- the monitor must outlive its inputs
            consecutive_failures += 1
            reason = f"DEGRADED: {type(exc).__name__}: {exc}"
            print(f"[launch-monitor] tick failed: {reason}", file=sys.stderr)
        if consecutive_failures == DEGRADED_ESCALATION_TICKS:
            now_epoch = time.time()
            last = state.last_alert_epochs.get("monitor-degraded", 0.0)
            if now_epoch - last >= thresholds.alert_cooldown_seconds:
                _send(
                    f"crawl MONITOR DEGRADED: {consecutive_failures} consecutive"
                    f" failed ticks ({reason}); the monitor is still running"
                    " but its numbers are unavailable",
                    send_script,
                )
                state.last_alert_epochs["monitor-degraded"] = now_epoch
                state.save(args.state_file)
        write_heartbeat(
            args.state_file,
            tick_ok=reason is None,
            reason=reason,
            consecutive_failures=consecutive_failures,
            tick_counter=tick_counter,
            tick_duration_seconds=time.monotonic() - started,
            generation=state.generation,
            poll_seconds=args.poll,
        )
        if tick_snapshot is not None and tick_snapshot.complete and state.completion_sent:
            return 0
        if max_ticks is not None and tick_counter >= max_ticks:
            return 0
        sleep_fn(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
