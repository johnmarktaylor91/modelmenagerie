"""Durable parking rows for work refused by a quarantined environment generation.

A quarantined environment generation (an ``environment-cleanup-quarantined``
campaign-health event) refuses execution for every work identity it cannot
prove completed-and-fresh. Terminalizing that refusal as ``failed:environment``
/ ``build-failed`` was rung 11's headline defect: no build ever ran, the root
cause was pure infrastructure (a cleanup-time ``DiskRecoveryError``), and the
21-record fan-out superseded eleven verified run awards. The refusal itself is
real -- the driver genuinely must not rebuild inside a quarantined generation --
but it is a CAMPAIGN fact, recoverable by recovering or rotating the
environment, so the model is PARKED in this durable machine-deferral ledger
instead, exactly as the host-capacity and environment-coverage valves park what
they cannot attempt.
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

REBUILD_DEFERRAL_SCHEMA_VERSION = "menagerie.crawler.environment-rebuild-deferral.v2"

#: Read-only legacy schema. The ledger is durable and append-only, so rows a
#: v1 implementation persisted MUST stay readable forever (Sol round-2 review
#: finding 2: v2-only validation made a digest-valid v1 row abort every load).
#: v1 rows are HISTORY: reads accept them, appends refuse them, and wherever a
#: v1 and a v2 row coexist for one work generation the v2 row governs.
REBUILD_DEFERRAL_SCHEMA_VERSION_V1 = "menagerie.crawler.environment-rebuild-deferral.v1"

#: Closed per-member setup-artifact states carried by ``quarantine.artifact_state``.
#: ``present`` members carry their exact byte digest; ``absent`` members carry no
#: digest. Distinct partial families therefore always produce distinct vectors --
#: shared absence is a typed, distinguishable fact, never a ``None == None`` match
#: (rung-11 review finding 6).
ARTIFACT_MEMBER_STATES = frozenset({"present", "absent"})

REBUILD_DEFERRAL_DISPOSITION = "deferred:environment-rebuild-required"
"""Greppable disposition for work parked behind a quarantined environment.

It follows the established ``deferred:<remedy>`` shape: a real, located,
correctly authored model that this campaign cannot execute RIGHT NOW, naming
the exact remedy that recovers it (recover or rotate the quarantined
environment generation, then re-drive the work item).

It is deliberately NOT ``failed:*``, and above all not ``build-failed``: no
build was attempted, so a ``build-failed`` terminal would assert a world-fact
the run never observed. The quarantine event names an infrastructure failure
during CLEANUP of a previous environment cycle -- nothing about the model is
known to be broken, and the same environment typically built and ran fine
earlier in the same campaign.

It is also deliberately not minted as a canonical model terminal, for exactly
the reason the host-capacity and environment-coverage dispositions are not: a
canonical ``deferred:*`` record must carry a checker-adjudicated terminal gate
over frozen sources, and this disposition is machine-derived from an
operational quarantine event with no such adjudication. The model instead stays
UNCOMPLETED for this campaign and is recorded here, which is the truthful
statement: not attempted, not failed, recoverable after an environment
recovery/rotation.
"""

REBUILD_DEFERRAL_RELATIVE_PATH = Path("intake-extensions") / "environment-rebuild-deferrals.jsonl"


class RebuildDeferralError(ValueError):
    """Raised when an environment-rebuild deferral row is malformed."""


def rebuild_deferral_path(records_root: Path) -> Path:
    """Return the canonical environment-rebuild deferral ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only environment-rebuild deferral JSONL path.
    """

    return records_root / REBUILD_DEFERRAL_RELATIVE_PATH


def build_rebuild_deferral_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    quarantine_event_id: str,
    intent: str,
    target: str,
    artifact_identity: str,
    artifact_state: Mapping[str, Mapping[str, str]],
    env_generation: Optional[str],
    failure_type: Optional[str],
) -> JsonObject:
    """Assemble one complete, self-describing environment-rebuild deferral row.

    The row names the quarantine event, the quarantined intent/target, the
    typed per-member setup-artifact state it matched on, and the quarantined
    environment generation, so a later reviewer can act on it without the code
    that made the call.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the active scheduled work generation.
    campaign_id, run_id, machine_id:
        Campaign and host provenance for the deferral.
    created_at:
        UTC timestamp of the decision.
    quarantine_event_id:
        The ``environment-cleanup-quarantined`` campaign-health event that
        refused this work.
    intent, target:
        Quarantined environment intent and solve target.
    artifact_identity:
        Typed setup-artifact identity the quarantine matched on. Never null:
        an incomplete artifact family hashes its full per-member state vector,
        so distinct partial states never collapse together.
    artifact_state:
        Per-member state vector: each setup-artifact member maps to
        ``{"state": "present", "sha256": ...}`` or ``{"state": "absent"}``.
    env_generation:
        Quarantined environment generation, when the event recorded one.
    failure_type:
        Fully qualified exception type the quarantine event recorded.

    Returns
    -------
    dict[str, Any]
        Content-addressed deferral row.

    Raises
    ------
    RebuildDeferralError
        If a required identity field is empty or mistyped.
    """

    if not campaign_id or campaign_id == str(None):
        # T2-12: the pilot froze campaign_id="None" (a stringified absence)
        # into 62 row digests. The absence must refuse at BUILD time; read-side
        # validators stay tolerant so those historical rows remain readable.
        raise ValueError("campaign_id must name a real campaign, never a stringified absence")
    generation_clause = (
        f"environment generation {env_generation}"
        if env_generation
        else "the environment generation"
    )
    recheck_hint = (
        f"{generation_clause} for intent {intent!r} (target {target!r}) is quarantined "
        f"by campaign-health event {quarantine_event_id!r} after an incomplete-cleanup "
        f"refusal ({failure_type or 'unrecorded failure type'}); no build was attempted "
        "for this model. Recover or rotate the environment (a re-solve changes the "
        "intent's setup-artifact identity), or record an explicit "
        "environment-quarantine-cleared event naming this quarantine after an "
        "out-of-band repair, then re-drive this work item."
    )
    row: JsonObject = {
        "schema_version": REBUILD_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": REBUILD_DEFERRAL_DISPOSITION,
        "quarantine": {
            "event_id": quarantine_event_id,
            "intent": intent,
            "target": target,
            "artifact_identity": artifact_identity,
            "artifact_state": {name: dict(state) for name, state in artifact_state.items()},
            "env_generation": env_generation,
            "failure_type": failure_type,
        },
        "recheck_hint": recheck_hint,
    }
    row = stamp_row_digest(row)
    return validate_rebuild_deferral_row(row)


def validate_rebuild_deferral_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one v2 OR read-only legacy v1 row and return its canonical copy.

    The ledger is durable and append-only, so both persisted generations stay
    readable: v2 rows validate against the full current schema (typed
    artifact identity plus the per-member state vector), while v1 rows
    validate against exactly the shape the v1 implementation wrote (nullable
    artifact identity, no state vector). Appends remain v2-only
    (:func:`append_rebuild_deferral_row`), and wherever both generations
    coexist for one work generation the v2 row governs
    (:func:`current_rebuild_rows`).

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
    RebuildDeferralError
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
        "quarantine",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise RebuildDeferralError(
            f"environment-rebuild deferral row is missing {', '.join(sorted(missing))}"
        )
    schema_version = row["schema_version"]
    if schema_version not in (
        REBUILD_DEFERRAL_SCHEMA_VERSION,
        REBUILD_DEFERRAL_SCHEMA_VERSION_V1,
    ):
        raise RebuildDeferralError(
            f"environment-rebuild deferral row declares unknown schema {schema_version!r}"
        )
    if row["disposition"] != REBUILD_DEFERRAL_DISPOSITION:
        raise RebuildDeferralError(
            f"environment-rebuild deferral row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise RebuildDeferralError(f"environment-rebuild deferral row has an empty {field}")
    quarantine = row["quarantine"]
    if not isinstance(quarantine, Mapping):
        raise RebuildDeferralError(
            "environment-rebuild deferral row must record its quarantine provenance"
        )
    for field in ("event_id", "intent", "target"):
        if not isinstance(quarantine.get(field), str) or not quarantine.get(field):
            raise RebuildDeferralError(
                f"environment-rebuild deferral row must name the quarantine {field}"
            )
    if schema_version == REBUILD_DEFERRAL_SCHEMA_VERSION:
        _validate_v2_quarantine_block(quarantine)
    else:
        # Legacy v1 shape, read-only: artifact identity was nullable (shared
        # absence collapsed to null) and no state vector existed. The rows
        # stay loadable exactly as written; the v6-finding semantics apply
        # only to v2 rows.
        for field in ("artifact_identity", "env_generation", "failure_type"):
            value = quarantine.get(field)
            if value is not None and (not isinstance(value, str) or not value):
                raise RebuildDeferralError(
                    f"environment-rebuild deferral row has a non-string quarantine {field}"
                )
    if not isinstance(row["recheck_hint"], str) or not row["recheck_hint"].strip():
        raise RebuildDeferralError(
            "environment-rebuild deferral row must name what would recover the model later"
        )
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise RebuildDeferralError(
            f"rebuild deferral row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise RebuildDeferralError(
            "environment-rebuild deferral row digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def _validate_v2_quarantine_block(quarantine: Mapping[str, Any]) -> None:
    """Validate the v2 typed identity and per-member state vector.

    Parameters
    ----------
    quarantine:
        The ``quarantine`` block of a v2 candidate row.

    Raises
    ------
    RebuildDeferralError
        If the identity or state vector is missing or mistyped.
    """

    identity = quarantine.get("artifact_identity")
    if not isinstance(identity, str) or not identity:
        raise RebuildDeferralError(
            "environment-rebuild deferral row must name the quarantine artifact_identity"
        )
    for field in ("env_generation", "failure_type"):
        value = quarantine.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise RebuildDeferralError(
                f"environment-rebuild deferral row has a non-string quarantine {field}"
            )
    artifact_state = quarantine.get("artifact_state")
    if not isinstance(artifact_state, Mapping) or not artifact_state:
        raise RebuildDeferralError(
            "environment-rebuild deferral row must record the typed per-member "
            "setup-artifact state vector"
        )
    for member, state in artifact_state.items():
        if not isinstance(member, str) or not member or not isinstance(state, Mapping):
            raise RebuildDeferralError(
                "environment-rebuild deferral row has a malformed artifact_state member"
            )
        kind = state.get("state")
        if kind not in ARTIFACT_MEMBER_STATES:
            raise RebuildDeferralError(
                f"environment-rebuild deferral row artifact member {member!r} declares "
                f"unknown state {kind!r}"
            )
        digest_value = state.get("sha256")
        if kind == "present" and (not isinstance(digest_value, str) or not digest_value):
            raise RebuildDeferralError(
                f"environment-rebuild deferral row artifact member {member!r} is present "
                "without its byte digest"
            )
        if kind == "absent" and set(state) != {"state"}:
            raise RebuildDeferralError(
                f"environment-rebuild deferral row artifact member {member!r} is absent "
                "but carries extra fields"
            )


def _rebuild_fact_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the quarantine FACT identity one park row records.

    Parameters
    ----------
    row:
        Validated rebuild deferral row.

    Returns
    -------
    tuple[str, str, str]
        Stable ID, work generation, and the quarantine event that refused it.
    """

    quarantine = row.get("quarantine", {})
    return (
        str(row.get("stable_id")),
        str(row.get("work_id")),
        str(quarantine.get("event_id") if isinstance(quarantine, Mapping) else None),
    )


def append_rebuild_deferral_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one environment-rebuild deferral durably and idempotently.

    Parameters
    ----------
    path:
        Destination environment-rebuild deferral JSONL.
    row:
        Complete typed deferral row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    RebuildDeferralError
        If the row is malformed.

    Notes
    -----
    Idempotency is keyed on the complete quarantine FACT identity -- stable
    ID, work generation, AND the quarantine event that refused the work --
    never on the work generation alone. A resume that re-derives the SAME park
    (same quarantine event; ``created_at``/``run_id`` necessarily differ) is a
    no-op returning the first persisted row. A SECOND quarantine for the same
    work generation -- rotate to new setup artifacts, resume, and the rebuilt
    environment quarantines again -- is a NEW fact and appends a NEW row that
    supersedes the released one at read time (latest row per work generation
    wins; rung-11 review finding: keying on the work generation alone
    resurrected the released first row forever and the second quarantine was
    silently lost).
    """

    validated = validate_rebuild_deferral_row(row)
    if validated["schema_version"] != REBUILD_DEFERRAL_SCHEMA_VERSION:
        raise RebuildDeferralError(
            "environment-rebuild deferral appends are v2-only; legacy v1 rows are read-only history"
        )
    key = _rebuild_fact_key(validated)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_rebuild_lines(_ledger_bytes(path, handle), path)
        # Only a v2 row satisfies the fact key: a fresh derivation over a
        # ledger holding the SAME fact as a legacy v1 row persists the v2
        # upgrade, which then governs at read time (v1 rows are history).
        matching = [
            item
            for item in existing
            if _rebuild_fact_key(item) == key
            and item["schema_version"] == REBUILD_DEFERRAL_SCHEMA_VERSION
        ]
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
        prefix_check=lambda retained: _decode_rebuild_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def _decode_rebuild_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one deferral ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RebuildDeferralError(
                f"environment-rebuild deferral ledger {path} has an unreadable row at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise RebuildDeferralError(
                f"environment-rebuild deferral ledger {path} has a non-object row at line {index}"
            )
        rows.append(validate_rebuild_deferral_row(decoded))
    return tuple(rows)


def current_rebuild_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[JsonObject, ...]:
    """Reduce validated rows to the ONE governing row per work generation.

    Ordering/supersession rule for the durable append-only ledger: within one
    (stable ID, work ID) work generation, the LATEST v2 row governs; the
    LATEST v1 row governs only while no v2 row exists for that generation
    (legacy history readable, current schema authoritative -- Sol round-2
    review finding 2).

    Parameters
    ----------
    rows:
        Already-validated rows in ledger order.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Governing row per (stable ID, work ID), ordered by that key.
    """

    latest_v2: dict[tuple[str, str], JsonObject] = {}
    latest_v1: dict[tuple[str, str], JsonObject] = {}
    for row in rows:
        key = (str(row["stable_id"]), str(row["work_id"]))
        if row["schema_version"] == REBUILD_DEFERRAL_SCHEMA_VERSION:
            latest_v2[key] = dict(row)
        else:
            latest_v1[key] = dict(row)
    governed = {**latest_v1, **latest_v2}
    return tuple(governed[key] for key in sorted(governed))


def load_rebuild_deferral_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load the CURRENT environment-rebuild deferral per work generation.

    The ledger is append-only and idempotent per quarantine fact, so several
    rows may exist for one work generation (one per distinct quarantine
    event, across both persisted schema generations). The governing row per
    work generation follows :func:`current_rebuild_rows`: latest v2, else
    latest v1.

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
        rows.extend(_decode_rebuild_lines(_ledger_bytes(path), path))
    return current_rebuild_rows(rows)
