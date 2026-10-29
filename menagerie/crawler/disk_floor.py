"""Durable parking rows for work refused by the host's environment disk floor.

The env lane refuses to START a build when the host holds less free disk than
the configured floor (30 GiB by default). Terminalizing that refusal as
``failed:resource`` / ``disk-floor`` was rung 12's class-A defect: a full disk
is a MACHINE condition, not a fact about any model -- no solve, build, or probe
ran, six models were terminalized from host disk state, one honest
``failed:environment/solve-failed`` negative was buried, and two review-held
supersession grants were consumed publishing infrastructure artifacts. The
refusal itself is real -- the lane genuinely must not build under the floor --
but it is recoverable by freeing disk on this host or re-driving the work on
any host with disk, so the model is PARKED in this durable machine-deferral
ledger instead, exactly as the host-capacity, environment-coverage, and
environment-rebuild valves park what they cannot attempt.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional, Sequence

from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    row_content_address,
    stamp_row_digest,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import read_with_tail_recovery

DISK_FLOOR_DEFERRAL_SCHEMA_VERSION = "menagerie.crawler.disk-floor-deferral.v1"

DISK_FLOOR_DEFERRAL_DISPOSITION = "deferred:needs-host-disk"
"""Greppable disposition for work parked behind an unmet host disk floor.

It follows the established ``deferred:needs-<capability>`` shape
(``deferred:needs-more-memory``, ``deferred:needs-environment-coverage``): a
real, located, correctly authored model that this campaign cannot execute
RIGHT NOW, naming the exact remedy that recovers it (free disk above the env
floor on the recorded host, or re-drive the work item on any host with disk).

It is deliberately NOT ``failed:*``, and above all not
``failed:resource/disk-floor``: nothing about the model was assessed, so a
resource terminal would assert a world-fact the run never observed. It is
also deliberately not minted as a canonical model terminal, for exactly the
reason the sibling valve dispositions are not: a canonical ``deferred:*``
record must carry a checker-adjudicated terminal gate over frozen sources,
and this disposition is machine-derived from a host ``statvfs`` measurement
with no such adjudication. The model instead stays UNCOMPLETED for this
campaign and is recorded here, which is the truthful statement: not
attempted, not failed, recoverable wherever disk exists.
"""

DISK_FLOOR_DEFERRAL_RELATIVE_PATH = Path("intake-extensions") / "disk-floor-deferrals.jsonl"

#: Closed refusal-condition vocabulary (additive schema, 2026-08-11). ``floor``
#: is a pre-write measurement below the configured minimum; ``enospc`` is a
#: write that ALREADY failed with ``ENOSPC`` -- there the failed write itself is
#: the refusal evidence, so the recorded measurement may legitimately sit at or
#: above the floor (quota exhaustion, or another writer freeing space between
#: the failure and the probe). Rows persisted before this field default to
#: ``floor`` at read time.
DISK_FLOOR_CONDITIONS = frozenset({"floor", "enospc"})


class DiskFloorDeferralError(ValueError):
    """Raised when a disk-floor deferral row is malformed."""


def disk_floor_deferral_path(records_root: Path) -> Path:
    """Return the canonical disk-floor deferral ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only disk-floor deferral JSONL path.
    """

    return records_root / DISK_FLOOR_DEFERRAL_RELATIVE_PATH


def build_disk_floor_deferral_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    intent: str,
    target: str,
    env_root: str,
    floor_bytes: int,
    free_bytes: int,
    condition: str = "floor",
    destination_role: Optional[str] = None,
) -> JsonObject:
    """Assemble one complete, self-describing disk-floor deferral row.

    The row names the refusing host, the routed intent/target, the refused
    floor, and the measured free bytes, so a later reviewer or scheduler can
    act on it without the code that made the call.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the active scheduled work generation.
    campaign_id, run_id, machine_id:
        Campaign and host provenance; the machine is the refusal's subject.
    created_at:
        UTC timestamp of the decision.
    intent, target:
        Routed environment intent and solve target the floor refused to serve.
    env_root:
        Environment root whose filesystem was measured.
    floor_bytes:
        Configured minimum free bytes the env lane requires.
    free_bytes:
        Measured free bytes behind the refusal.
    condition:
        Refusal shape from :data:`DISK_FLOOR_CONDITIONS`. Only a ``floor``
        row must measure below the floor; an ``enospc`` row's evidence is the
        failed write itself.
    destination_role:
        Optional starved write-destination role (``env-root``,
        ``backend-state``, ``tmp``) when the refusing measurement names one.

    Returns
    -------
    dict[str, Any]
        Content-addressed deferral row.

    Raises
    ------
    DiskFloorDeferralError
        If an identity field is empty, a measurement is mistyped, or the
        measurement does not actually record a refusal.
    """

    if not campaign_id or campaign_id == str(None):
        # T2-12: the pilot froze campaign_id="None" (a stringified absence)
        # into 62 row digests. The absence must refuse at BUILD time; read-side
        # validators stay tolerant so those historical rows remain readable.
        raise ValueError("campaign_id must name a real campaign, never a stringified absence")
    if not _positive_int(floor_bytes):
        raise DiskFloorDeferralError("disk-floor deferral row must record a positive byte floor")
    if not isinstance(free_bytes, int) or isinstance(free_bytes, bool) or free_bytes < 0:
        raise DiskFloorDeferralError(
            "disk-floor deferral row has a non-measurement free_bytes value"
        )
    if condition not in DISK_FLOOR_CONDITIONS:
        raise DiskFloorDeferralError(
            f"disk-floor deferral row declares unknown condition {condition!r}"
        )
    if condition == "floor" and free_bytes >= floor_bytes:
        raise DiskFloorDeferralError(
            "only a refusing disk measurement may be recorded as a deferral"
        )
    if condition == "enospc":
        refusal_clause = (
            f"an environment write on host {machine_id!r} failed with ENOSPC "
            f"(least free write destination {env_root!r} measured {free_bytes} "
            f"bytes against the {floor_bytes}-byte floor)"
        )
    else:
        refusal_clause = (
            f"host {machine_id!r} held {free_bytes} free bytes at {env_root!r}, below "
            f"the {floor_bytes}-byte environment disk floor"
        )
    recheck_hint = (
        f"{refusal_clause}, so environment setup for intent "
        f"{intent!r} (target {target!r}) was refused before any build result; nothing "
        "about the model was assessed. Free disk above the floor on that host, or "
        "re-drive this work item on any host with disk -- the refusal names the "
        "machine, never the model."
    )
    disk: JsonObject = {
        "intent": intent,
        "target": target,
        "env_root": env_root,
        "floor_bytes": floor_bytes,
        "free_bytes": free_bytes,
        "condition": condition,
    }
    if destination_role is not None:
        disk["destination_role"] = destination_role
    row: JsonObject = {
        "schema_version": DISK_FLOOR_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": DISK_FLOOR_DEFERRAL_DISPOSITION,
        "disk": disk,
        "recheck_hint": recheck_hint,
    }
    row = stamp_row_digest(row)
    return validate_disk_floor_deferral_row(row)


def _positive_int(value: Any) -> bool:
    """Return whether one candidate is a positive non-bool integer."""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_disk_floor_deferral_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one disk-floor deferral row and return its canonical copy.

    Parameters
    ----------
    row:
        Candidate deferral row.

    Returns
    -------
    dict[str, Any]
        Validated row.

    Raises
    ------
    DiskFloorDeferralError
        If a required field is missing, mistyped, or the digest disagrees.
    """

    required = (
        "schema_version",
        "stable_id",
        "work_id",
        "name",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
        "disposition",
        "disk",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise DiskFloorDeferralError(
            f"disk-floor deferral row is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != DISK_FLOOR_DEFERRAL_SCHEMA_VERSION:
        raise DiskFloorDeferralError(
            f"disk-floor deferral row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] != DISK_FLOOR_DEFERRAL_DISPOSITION:
        raise DiskFloorDeferralError(
            f"disk-floor deferral row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise DiskFloorDeferralError(f"disk-floor deferral row has an empty {field}")
    disk = row["disk"]
    if not isinstance(disk, Mapping):
        raise DiskFloorDeferralError("disk-floor deferral row must record its refusal measurement")
    for field in ("intent", "target", "env_root"):
        if not isinstance(disk.get(field), str) or not disk.get(field):
            raise DiskFloorDeferralError(f"disk-floor deferral row must name the refused {field}")
    if not _positive_int(disk.get("floor_bytes")):
        raise DiskFloorDeferralError("disk-floor deferral row must record a positive byte floor")
    free = disk.get("free_bytes")
    if not isinstance(free, int) or isinstance(free, bool) or free < 0:
        raise DiskFloorDeferralError(
            "disk-floor deferral row has a non-measurement free_bytes value"
        )
    # Rows persisted before the additive condition field are floor refusals.
    condition = disk.get("condition", "floor")
    if condition not in DISK_FLOOR_CONDITIONS:
        raise DiskFloorDeferralError(
            f"disk-floor deferral row declares unknown condition {condition!r}"
        )
    if condition == "floor" and free >= disk["floor_bytes"]:
        raise DiskFloorDeferralError("disk-floor deferral row does not record a refusal")
    role = disk.get("destination_role")
    if role is not None and (not isinstance(role, str) or not role):
        raise DiskFloorDeferralError("disk-floor deferral row has an empty destination_role")
    if not isinstance(row["recheck_hint"], str) or not row["recheck_hint"].strip():
        raise DiskFloorDeferralError(
            "disk-floor deferral row must name what would recover the model later"
        )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise DiskFloorDeferralError(
            f"disk-floor deferral row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise DiskFloorDeferralError("disk-floor deferral row digest does not match its payload")
    validated["row_sha256"] = digest
    return validated


def _disk_floor_fact_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the host-refusal FACT identity one park row records.

    Parameters
    ----------
    row:
        Validated disk-floor deferral row.

    Returns
    -------
    tuple[str, str, str]
        Stable ID, work generation, and the refusing machine.
    """

    return (str(row.get("stable_id")), str(row.get("work_id")), str(row.get("machine_id")))


def append_disk_floor_deferral_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one disk-floor deferral durably and idempotently.

    Parameters
    ----------
    path:
        Destination disk-floor deferral JSONL.
    row:
        Complete typed deferral row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    DiskFloorDeferralError
        If the row is malformed.

    Notes
    -----
    Idempotency is keyed on the host-refusal FACT identity -- stable ID, work
    generation, AND the refusing machine -- never on the work generation
    alone. A resume on the same starved host re-derives the SAME park
    (``created_at``, ``run_id``, and the fresh ``free_bytes`` measurement
    necessarily differ) and is a no-op returning the first persisted row. A
    refusal observed on a DIFFERENT host is a NEW fact and appends a NEW row
    that supersedes the earlier one at read time (latest row per work
    generation wins), exactly as the rebuild valve keys on its quarantine
    fact. A host whose disk recovered writes nothing here at all, so a stale
    row can only be superseded by the model completing normally.
    """

    validated = validate_disk_floor_deferral_row(row)
    key = _disk_floor_fact_key(validated)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_disk_floor_lines(_ledger_bytes(path, handle), path)
        matching = [item for item in existing if _disk_floor_fact_key(item) == key]
        if matching:
            return matching[0]
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def _ledger_bytes(path: Path, handle: Optional[BinaryIO] = None) -> bytes:
    """Read the whole ledger, recovering only a torn final line with evidence.

    A complete malformed line stays this valve's typed hard refusal: the
    recovery's prefix check is this module's own strict decoder.
    """

    return read_with_tail_recovery(
        path,
        prefix_check=lambda retained: _decode_disk_floor_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def _decode_disk_floor_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one deferral ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiskFloorDeferralError(
                f"disk-floor deferral ledger {path} has an unreadable row at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise DiskFloorDeferralError(
                f"disk-floor deferral ledger {path} has a non-object row at line {index}"
            )
        rows.append(validate_disk_floor_deferral_row(decoded))
    return tuple(rows)


def current_disk_floor_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[JsonObject, ...]:
    """Reduce validated rows to the ONE governing row per work generation.

    Ordering/supersession rule for the durable append-only ledger: within one
    (stable ID, work ID) work generation, the LATEST row governs -- a refusal
    recorded later (typically on another host) supersedes the earlier park.

    Parameters
    ----------
    rows:
        Already-validated rows in ledger order.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Governing row per (stable ID, work ID), ordered by that key.
    """

    latest: dict[tuple[str, str], JsonObject] = {}
    for row in rows:
        latest[(str(row["stable_id"]), str(row["work_id"]))] = dict(row)
    return tuple(latest[key] for key in sorted(latest))


def load_disk_floor_deferral_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load the CURRENT disk-floor deferral per work generation.

    The ledger is append-only and idempotent per host-refusal fact, so
    several rows may exist for one work generation (one per refusing host).
    The governing row per work generation follows
    :func:`current_disk_floor_rows`.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Governing row per (stable ID, work ID), ordered by that key.
    """

    rows: list[JsonObject] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.extend(_decode_disk_floor_lines(_ledger_bytes(path), path))
    return current_disk_floor_rows(rows)
