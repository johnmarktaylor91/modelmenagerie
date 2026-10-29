"""Paced start and the ``configuration_canary`` census (overhaul RC-1, D.1 behavior 5).

The R7 canary GATE is retired. Its replacement is a paced start: whenever the
configuration identity tuple (tree sha, author model, checker model, author
prompt, checker prompt) differs from the newest ``configuration_canary`` row
for the campaign, the author width is capped at :data:`PACED_AUTHOR_WIDTH`
until :data:`PACED_TERMINALS` new terminals exist; the census row is then
emitted (publication rate, accept rate, storm-fuse state, mean session wall)
and the cap lifts. The fuses remain the automatic stop.

Durable state lives in ``records/control/configuration-canary.jsonl``:

* ``paced_start`` -- opened when a new identity is first seen, carrying the
  ledger row-count baselines the census window is measured from (so a killed
  and restarted invocation continues the SAME count instead of re-baselining);
* ``configuration_canary`` -- the census, bound to the same identity digest.

Both rows are content-stamped (``row_sha256``). Nothing here grants a
disposition to any record; the census is disclosure and the cap is pacing.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    stable_hash,
    stamp_row_digest,
)
from menagerie.crawler.identity import fsync_directory
from menagerie.crawler.models import JsonObject

__all__ = [
    "CONFIGURATION_CANARY_SCHEMA_VERSION",
    "PACED_AUTHOR_WIDTH",
    "PACED_TERMINALS",
    "ConfigurationIdentity",
    "PacedStart",
    "build_canary_census",
    "canary_ledger_path",
    "evaluate_paced_start",
    "ledger_row_count",
    "record_configuration_canary",
]

CONFIGURATION_CANARY_SCHEMA_VERSION = "menagerie.crawler.configuration-canary.v1"

#: Author width while a new configuration is paced (RULING D.1 behavior 5).
PACED_AUTHOR_WIDTH = 4

#: New terminals required before the census row is emitted and the cap lifts.
PACED_TERMINALS = 20

_PACED_START_EVENT = "paced_start"
_CANARY_EVENT = "configuration_canary"


@dataclass(frozen=True)
class ConfigurationIdentity:
    """The identity tuple a paced start keys on."""

    tree_sha: Optional[str]
    author_model: str
    checker_model: str
    author_prompt_identity: str
    checker_prompt_identity: str

    def to_dict(self) -> dict[str, Any]:
        """Return the tuple as a JSON object."""

        return {
            "tree_sha": self.tree_sha,
            "author_model": self.author_model,
            "checker_model": self.checker_model,
            "author_prompt_identity": self.author_prompt_identity,
            "checker_prompt_identity": self.checker_prompt_identity,
        }

    @property
    def digest(self) -> str:
        """Return the content address of the tuple."""

        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class PacedStart:
    """One open paced start: the cap and the census window baselines."""

    identity: ConfigurationIdentity
    baselines: Mapping[str, int]
    opened_row_sha256: str
    cap: int = PACED_AUTHOR_WIDTH
    required_terminals: int = PACED_TERMINALS


def canary_ledger_path(records_root: Path) -> Path:
    """Return the campaign's paced-start / canary ledger path."""

    return records_root / "control" / "configuration-canary.jsonl"


def ledger_row_count(path: Path) -> int:
    """Count non-blank rows of one append-only JSONL ledger (0 when absent)."""

    if not path.is_file():
        return 0
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _load_rows(path: Path) -> list[JsonObject]:
    """Load every ledger row, refusing malformed lines typed."""

    if not path.is_file():
        return []
    rows: list[JsonObject] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _append(path: Path, body: Mapping[str, Any]) -> JsonObject:
    """Stamp and append one row under an exclusive data-inode flock, fsynced.

    D-60 writer-lock discipline: the flock spans the append so a torn tail is
    prevented under the same lock; appends are additionally serialized by the
    driver lock (the only writer), and reads are plain row loads.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    row = stamp_row_digest(body)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(row) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return row


def evaluate_paced_start(
    records_root: Path,
    *,
    campaign_id: str,
    identity: ConfigurationIdentity,
    current_counts: Mapping[str, int],
    run_id: str,
    now: str,
) -> Optional[PacedStart]:
    """Decide whether this invocation runs paced, opening the window if needed.

    Returns ``None`` when the newest row for the campaign is a
    ``configuration_canary`` for this exact identity (the configuration has
    already been canaried). Otherwise returns the open :class:`PacedStart`:
    the existing ``paced_start`` row for this identity when one is the newest
    row (a continued window), else a freshly appended one baselined at
    ``current_counts``.
    """

    path = canary_ledger_path(records_root)
    rows = [row for row in _load_rows(path) if row.get("campaign_id") == campaign_id]
    newest = rows[-1] if rows else None
    if newest is not None and newest.get("identity_sha256") == identity.digest:
        if newest.get("event") == _CANARY_EVENT:
            return None
        if newest.get("event") == _PACED_START_EVENT:
            baselines = newest.get("baselines")
            if isinstance(baselines, Mapping):
                return PacedStart(
                    identity=identity,
                    baselines={str(k): int(v) for k, v in baselines.items()},
                    opened_row_sha256=str(newest.get("row_sha256")),
                )
    opened = _append(
        path,
        {
            "schema_version": CONFIGURATION_CANARY_SCHEMA_VERSION,
            "event": _PACED_START_EVENT,
            "campaign_id": campaign_id,
            "identity": identity.to_dict(),
            "identity_sha256": identity.digest,
            "baselines": {str(k): int(v) for k, v in current_counts.items()},
            "cap_author_width": PACED_AUTHOR_WIDTH,
            "required_terminals": PACED_TERMINALS,
            "run_id": run_id,
            "created_at": now,
        },
    )
    return PacedStart(
        identity=identity,
        baselines={str(k): int(v) for k, v in current_counts.items()},
        opened_row_sha256=str(opened["row_sha256"]),
    )


def _rows_after(path: Path, skip: int) -> list[JsonObject]:
    """Return the JSON rows of one ledger after its first ``skip`` rows."""

    if not path.is_file():
        return []
    rows: list[JsonObject] = []
    index = 0
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            index += 1
            if index <= skip:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _publication(rows: Sequence[JsonObject]) -> tuple[int, int, Optional[float]]:
    """Return ``(terminals, runs, publication rate)`` over new model rows."""

    runs = sum(
        1
        for row in rows
        if isinstance(row.get("status"), Mapping) and row["status"].get("kind") == "runs"
    )
    return len(rows), runs, (runs / len(rows)) if rows else None


def _acceptance(rows: Sequence[JsonObject]) -> tuple[int, int, Optional[float]]:
    """Return ``(items, accurate, accept rate)`` over new metadata-batch gate rows."""

    items = 0
    accurate = 0
    for row in rows:
        if row.get("gate_kind") != "metadata_batch":
            continue
        for item in row.get("items") or ():
            if isinstance(item, Mapping):
                items += 1
                accurate += 1 if item.get("verdict") == "accurate" else 0
    return items, accurate, (accurate / items) if items else None


def _session_wall(rows: Sequence[JsonObject]) -> tuple[int, Optional[float]]:
    """Return ``(settled sessions, mean machine wall seconds)`` over new spend rows."""

    walls: list[float] = []
    for row in rows:
        if row.get("event") != "session_settled":
            continue
        machine = row.get("machine")
        wall = machine.get("wall_seconds") if isinstance(machine, Mapping) else None
        if isinstance(wall, (int, float)):
            walls.append(float(wall))
    return len(walls), (sum(walls) / len(walls)) if walls else None


def build_canary_census(
    *,
    models_path: Path,
    gates_path: Path,
    spend_path: Path,
    baselines: Mapping[str, int],
    storm_fuse_state: str,
) -> JsonObject:
    """Compute the census over the rows appended since the paced-start baselines.

    * ``publication_rate``: new model rows whose status kind is ``runs`` over
      all new model rows (every model row is a terminal record);
    * ``accept_rate``: ``metadata_batch`` gate items whose verdict is
      ``accurate`` over all new metadata-batch items;
    * ``storm_fuse``: the driver's own description of its cross-model fuse;
    * ``mean_session_wall_seconds``: mean machine wall of the new settled
      paid sessions.
    """

    terminals, runs, publication_rate = _publication(
        _rows_after(models_path, int(baselines.get("models_rows", 0)))
    )
    items, accurate, accept_rate = _acceptance(
        _rows_after(gates_path, int(baselines.get("gates_rows", 0)))
    )
    sessions, mean_wall = _session_wall(_rows_after(spend_path, int(baselines.get("spend_rows", 0))))
    return {
        "new_terminals": terminals,
        "new_runs": runs,
        "publication_rate": publication_rate,
        "metadata_items": items,
        "metadata_accurate": accurate,
        "accept_rate": accept_rate,
        "storm_fuse": storm_fuse_state,
        "settled_sessions": sessions,
        "mean_session_wall_seconds": mean_wall,
    }


def record_configuration_canary(
    records_root: Path,
    *,
    campaign_id: str,
    paced: PacedStart,
    census: Mapping[str, Any],
    run_id: str,
    now: str,
) -> JsonObject:
    """Append the census row that closes one paced start."""

    return _append(
        canary_ledger_path(records_root),
        {
            "schema_version": CONFIGURATION_CANARY_SCHEMA_VERSION,
            "event": _CANARY_EVENT,
            "campaign_id": campaign_id,
            "identity": paced.identity.to_dict(),
            "identity_sha256": paced.identity.digest,
            "paced_start_row_sha256": paced.opened_row_sha256,
            "baselines": dict(paced.baselines),
            "census": dict(census),
            "run_id": run_id,
            "created_at": now,
        },
    )
