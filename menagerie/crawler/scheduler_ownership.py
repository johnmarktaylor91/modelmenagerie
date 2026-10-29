"""D-59 (OC-12): host scheduler ownership registry + decommission verb.

The 200a incident class: a legitimately installed recurring wake entry (the
b1e1e9ee */15min resume cron) outlived its owning campaign ledger. Install
and deactivation were keyed to a LIVE ledger's episodes, so once the records
root moved, nothing on the host said who owned the entry and nothing could
lawfully remove it — it ran ``crawler run --resume`` silently every 15
minutes for days.

Three pieces close the class:

- **Registry**: a machine-scoped append-only JSONL (outside any records root,
  so it survives the owner it exists to outlive). Every scheduler install
  appends a ``registered`` row in the same transaction (a failed append
  refuses the install — an unregisterable entry would recreate the
  blindspot); every deactivation appends a ``decommissioned`` row.
- **Census**: the reverse sweep the wakeup manager never had — enumerate the
  HOST's torchlens scheduler entries (crontab wake blocks, launchd agent-root
  labels), join them against the registry and each row's owning operational
  ledger, and classify: ``registered-live`` (owner still holds the episode
  active), ``registered-stale`` (owner settled it but the entry survives),
  ``orphaned-owner-lost`` (owning ledger gone/unreadable), ``unregistered``
  (pre-registry era — exactly b1e1e9ee), ``decommissioned-residue`` (removed
  once, present again).
- **Decommission verb**: removes an eligible entry with a byte-exact
  provenance snapshot in the appended row, asserts the host is residue-free
  before the row is written, and REFUSES ``registered-live`` entries — those
  belong to the owning WakeupManager's deactivation path, never to cleanup.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    fsync_directory,
    stable_hash,
    stamp_row_digest,
)
from menagerie.crawler.models import JsonObject

SCHEDULER_OWNERSHIP_SCHEMA_VERSION = "menagerie.crawler.scheduler-ownership.v1"

#: Machine-scoped registry root override (tests, odd hosts).
SCHEDULER_OWNERSHIP_ROOT_ENV = "MENAGERIE_SCHEDULER_OWNERSHIP_ROOT"

#: The exact host footprints the wakeup machinery leaves (wakeup._cron_marker
#: and wakeup._launchd_label): the census sweep keys on these, so it finds
#: every era's entries, registered or not.
CRON_WAKE_MARKER_PREFIX = "# menagerie-wake-episode "
LAUNCHD_WAKE_LABEL_PREFIX = "org.menagerie.crawler-wake."

_ROW_KINDS = frozenset({"registered", "decommissioned"})
_ELIGIBLE_CLASSIFICATIONS = frozenset(
    {"unregistered", "registered-stale", "orphaned-owner-lost", "decommissioned-residue"}
)


class SchedulerOwnershipError(RuntimeError):
    """Raised on registry, census, or decommission refusals."""


def default_registry_path() -> Path:
    """Return the machine-scoped registry path (env-overridable root)."""

    override = os.environ.get(SCHEDULER_OWNERSHIP_ROOT_ENV)
    root = Path(override) if override else Path.home() / ".local" / "state" / "torchlens-crawler"
    return root / "scheduler-ownership.jsonl"


@dataclass(frozen=True)
class HostSchedulerEntry:
    """One torchlens-owned scheduler entry observed on the host."""

    backend: str
    entry_id: str
    content: str
    source: str


@dataclass(frozen=True)
class SchedulerOwnershipRegistry:
    """Append-only machine-scoped ownership ledger."""

    path: Path

    @classmethod
    def from_env(cls) -> "SchedulerOwnershipRegistry":
        return cls(default_registry_path())

    def rows(self) -> list[JsonObject]:
        """Load every validated registry row, fail-closed on corruption."""

        if not self.path.exists():
            return []
        rows: list[JsonObject] = []
        for index, line in enumerate(self.path.read_bytes().splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchedulerOwnershipError(
                    f"scheduler-ownership registry corrupt at line {index + 1}: {exc}"
                ) from exc
            if (
                not isinstance(row, dict)
                or row.get("schema_version") != SCHEDULER_OWNERSHIP_SCHEMA_VERSION
                or row.get("kind") not in _ROW_KINDS
                or not row.get("entry_id")
            ):
                raise SchedulerOwnershipError(
                    f"scheduler-ownership registry row {index + 1} is malformed"
                )
            rows.append(row)
        return rows

    def _append(self, row: JsonObject) -> JsonObject:
        # The shared stamping helper owns the row's content address (Track A
        # routing census): no writer spells its own exclusion set.
        payload = stamp_row_digest(dict(row))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            durable_fsync(handle.fileno())
        fsync_directory(self.path.parent)
        return payload

    def register_installation(
        self,
        *,
        entry_id: str,
        backend: str,
        label: str,
        callback_argv: Sequence[str],
        run_id: str,
        machine_id: str,
        owning_ledger: Path,
        created_at: str,
    ) -> JsonObject:
        """Durably record one scheduler installation's owner."""

        return self._append(
            {
                "schema_version": SCHEDULER_OWNERSHIP_SCHEMA_VERSION,
                "kind": "registered",
                "entry_id": entry_id,
                "backend": backend,
                "label": label,
                "callback_argv": list(callback_argv),
                "run_id": run_id,
                "machine_id": machine_id,
                "owning_ledger": str(owning_ledger),
                "created_at": created_at,
            }
        )

    def record_decommission(
        self,
        *,
        entry_id: str,
        backend: str,
        verb: str,
        reason: str,
        removed_content: str,
        run_id: str,
        machine_id: str,
        created_at: str,
    ) -> JsonObject:
        """Durably record one entry's removal with its provenance snapshot."""

        return self._append(
            {
                "schema_version": SCHEDULER_OWNERSHIP_SCHEMA_VERSION,
                "kind": "decommissioned",
                "entry_id": entry_id,
                "backend": backend,
                "verb": verb,
                "reason": reason,
                "removed_content": removed_content,
                "removed_content_sha256": stable_hash(removed_content),
                "run_id": run_id,
                "machine_id": machine_id,
                "created_at": created_at,
            }
        )


_CRON_BLOCK_RE = re.compile(
    re.escape(CRON_WAKE_MARKER_PREFIX) + r"(?P<entry>\S+) begin\n"
)


def parse_cron_wake_blocks(crontab_text: str) -> list[tuple[str, str]]:
    """Return every ``(episode_id, block_text)`` wake block in a crontab."""

    blocks: list[tuple[str, str]] = []
    lines = crontab_text.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        line = lines[index].rstrip("\n")
        if line.startswith(CRON_WAKE_MARKER_PREFIX) and line.endswith(" begin"):
            entry_id = line[len(CRON_WAKE_MARKER_PREFIX) : -len(" begin")]
            end_marker = f"{CRON_WAKE_MARKER_PREFIX}{entry_id} end"
            block = [lines[index]]
            index += 1
            terminated = False
            while index < len(lines):
                block.append(lines[index])
                if lines[index].rstrip("\n") == end_marker:
                    terminated = True
                    index += 1
                    break
                index += 1
            if not terminated:
                raise SchedulerOwnershipError(
                    f"unterminated cron wake block for {entry_id}"
                )
            blocks.append((entry_id, "".join(block)))
            continue
        index += 1
    return blocks


def remove_cron_wake_block(crontab_text: str, entry_id: str) -> str:
    """Remove exactly one marked wake block, preserving everything else."""

    begin = f"{CRON_WAKE_MARKER_PREFIX}{entry_id} begin"
    end = f"{CRON_WAKE_MARKER_PREFIX}{entry_id} end"
    output: list[str] = []
    skipping = False
    for line in crontab_text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped == begin:
            skipping = True
            continue
        if skipping and stripped == end:
            skipping = False
            continue
        if not skipping:
            output.append(line)
    if skipping:
        raise SchedulerOwnershipError(f"unterminated cron wake block for {entry_id}")
    return "".join(output)


def census_host_entries(
    *,
    crontab_text: str,
    agent_root: Path,
) -> list[HostSchedulerEntry]:
    """Enumerate every torchlens scheduler entry visible on the host."""

    entries = [
        HostSchedulerEntry(
            backend="cron", entry_id=entry_id, content=block, source="crontab"
        )
        for entry_id, block in parse_cron_wake_blocks(crontab_text)
    ]
    if agent_root.is_dir():
        for plist in sorted(agent_root.glob(f"{LAUNCHD_WAKE_LABEL_PREFIX}*.plist")):
            label = plist.name[: -len(".plist")]
            entries.append(
                HostSchedulerEntry(
                    backend="launchd",
                    entry_id=label[len(LAUNCHD_WAKE_LABEL_PREFIX) :],
                    content=plist.read_text(encoding="utf-8", errors="replace"),
                    source=str(plist),
                )
            )
    return entries


def _owner_episode_active(owning_ledger: Path, entry_id: str) -> Optional[bool]:
    """Return the owner's verdict on the episode, or None when the owner is lost.

    Reads the operational ledger leniently (the census must never take the
    single-writer lock a live driver holds) and reduces the wake-episode
    projection through the wakeup reducer.
    """

    if not owning_ledger.is_file():
        return None
    try:
        records = [
            json.loads(line)
            for line in owning_ledger.read_bytes().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    from menagerie.crawler.wakeup import reduce_wake_episodes

    try:
        projection = reduce_wake_episodes(records)
    except Exception:
        return None
    state = projection.episodes.get(entry_id)
    if state is None:
        return None
    return state.active


def classify_entry(
    registry_rows: Sequence[JsonObject], entry: HostSchedulerEntry
) -> str:
    """Classify one host entry against the registry + its owning ledger."""

    mine = [row for row in registry_rows if row.get("entry_id") == entry.entry_id]
    if not mine:
        return "unregistered"
    latest = mine[-1]
    if latest["kind"] == "decommissioned":
        return "decommissioned-residue"
    active = _owner_episode_active(Path(str(latest.get("owning_ledger", ""))), entry.entry_id)
    if active is None:
        return "orphaned-owner-lost"
    return "registered-live" if active else "registered-stale"


def evaluate_ownership_census(
    registry: SchedulerOwnershipRegistry, entries: Sequence[HostSchedulerEntry]
) -> JsonObject:
    """Join host entries against the registry; disclose every anomaly."""

    rows = registry.rows()
    entry_reports: list[JsonObject] = []
    for entry in entries:
        classification = classify_entry(rows, entry)
        entry_reports.append(
            {
                "entry_id": entry.entry_id,
                "backend": entry.backend,
                "source": entry.source,
                "classification": classification,
                "decommission_eligible": classification in _ELIGIBLE_CLASSIFICATIONS,
            }
        )
    observed_ids = {entry.entry_id for entry in entries}
    missing_host_entries = sorted(
        {
            str(row["entry_id"])
            for row in rows
            if row["kind"] == "registered"
            and str(row["entry_id"]) not in observed_ids
            and not any(
                later.get("entry_id") == row["entry_id"] and later["kind"] == "decommissioned"
                for later in rows
            )
        }
    )
    attention = [report for report in entry_reports if report["classification"] != "registered-live"]
    return {
        "schema_version": SCHEDULER_OWNERSHIP_SCHEMA_VERSION,
        "status": "attention" if attention else "clean",
        "entries": entry_reports,
        "missing_host_entries": missing_host_entries,
    }


def decommission_entry(
    registry: SchedulerOwnershipRegistry,
    entry: HostSchedulerEntry,
    *,
    reason: str,
    context: Any,
    created_at: str,
    crontab_reader: Optional[Callable[[], str]] = None,
    crontab_writer: Optional[Callable[[str], None]] = None,
    launchd_bootout: Optional[Callable[[str], None]] = None,
) -> JsonObject:
    """Remove one eligible host entry with provenance, residue-checked.

    Raises
    ------
    SchedulerOwnershipError
        If the entry is live under a reachable owner, the reason is empty,
        or residue survives the removal (no row is appended in that case —
        a decommission row may never stand over a present entry).
    """

    if not reason.strip():
        raise SchedulerOwnershipError("a decommission requires a non-empty operator reason")
    classification = classify_entry(registry.rows(), entry)
    if classification not in _ELIGIBLE_CLASSIFICATIONS:
        raise SchedulerOwnershipError(
            f"entry {entry.entry_id!r} is {classification}: a live owner's entry is "
            "deactivated by its own WakeupManager, never by cleanup"
        )
    if entry.backend == "cron":
        if crontab_reader is None or crontab_writer is None:
            raise SchedulerOwnershipError("cron decommission requires crontab reader+writer")
        current = crontab_reader()
        crontab_writer(remove_cron_wake_block(current, entry.entry_id))
        if any(entry_id == entry.entry_id for entry_id, _ in parse_cron_wake_blocks(crontab_reader())):
            raise SchedulerOwnershipError(
                f"residue: cron wake block {entry.entry_id!r} survived its removal"
            )
    elif entry.backend == "launchd":
        label = f"{LAUNCHD_WAKE_LABEL_PREFIX}{entry.entry_id}"
        if launchd_bootout is not None:
            launchd_bootout(label)
        plist = Path(entry.source)
        plist.unlink(missing_ok=True)
        if plist.exists():
            raise SchedulerOwnershipError(
                f"residue: launchd projection {plist} survived its removal"
            )
    else:
        raise SchedulerOwnershipError(f"unknown scheduler backend {entry.backend!r}")
    return registry.record_decommission(
        entry_id=entry.entry_id,
        backend=entry.backend,
        verb="decommission-tool",
        reason=reason,
        removed_content=entry.content,
        run_id=str(getattr(context, "run_id", "operator")),
        machine_id=str(getattr(context, "machine_id", "unknown")),
        created_at=created_at,
    )
