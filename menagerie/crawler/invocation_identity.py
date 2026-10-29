"""Invocation-identity ledger and leg attribution (D-22/DX-23, B3-XQ-12).

Every historical driver start shared ``run_id="crawler-run"``, so no ledger
read could prove which process invocation wrote a row or how many invocations
one "leg" actually spanned. This module rotates the run_id axis onto real
per-invocation identities and makes the invocation itself a durable fact:

* one append-only row per process invocation
  (``records_root/operational/invocations.jsonl``), carrying the machine and
  host identity the records-root guard's foreign-live-leg arm will consume;
* a TYPED succession row for an autonomous pause/resume, linking its
  predecessor invocation under a closed reason vocabulary;
* the B3-XQ-12 one-leg certification: "one leg" is one logical generation AND
  one process invocation -- a typed succession chain preserves it, an untyped
  restart voids it;
* close-side citation derivation (a close names the exact runs it spans and
  their invocation rows) and void-append attribution per invocation.

Era rule: historical rows written under the legacy constant stay readable
everywhere; only NEW identities are refused it, with
:class:`LegacyRunIdentityError`.
"""

from __future__ import annotations

import fcntl
import itertools
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

from menagerie.crawler.constants import InvocationOrigin
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    row_content_address,
    stable_hash,
    stamp_row_digest,
    utc_now,
)
from menagerie.crawler.models import JsonObject

#: Closed record identity for one durable process-invocation row.
INVOCATION_IDENTITY_SCHEMA_VERSION = "menagerie.crawler.invocation-identity.v1"

#: The retired shared run identity every pre-ledger driver start reused. Rows
#: already written under it stay readable (era rule); a NEW identity is never
#: allowed to be it.
LEGACY_RUN_ID = "crawler-run"

#: Era label for rows attributable to no invocation row (legacy run identities
#: and anything written before this ledger existed).
PRE_LEDGER_ERA = "pre-invocation-ledger"

#: Era label for rows attributed through a durable invocation row.
INVOCATION_LEDGER_ERA = "invocation-ledger"

#: Closed event vocabulary: one row kind per process invocation. A typed
#: pause/resume IS an invocation -- its row carries the succession link.
_INVOCATION_EVENTS = frozenset({"invocation_start", "invocation_succession"})

#: Closed succession reason vocabulary (B3-XQ-12): only a typed autonomous
#: pause/resume preserves one-leg certification. Widening this set is a new
#: waiver-class decision, never a convenience edit.
SUCCESSION_REASONS = frozenset({"autonomous-pause-resume"})

#: Closed one-leg certification reason vocabulary.
CERTIFICATION_REASONS = frozenset(
    {
        "single-invocation",
        "typed-succession-chain",
        "no-joined-runs",
        "unattributed-run-identity",
        "untyped-restart",
        "broken-succession-chain",
    }
)

#: Ledger location below one canonical records root.
INVOCATIONS_RELATIVE_PATH = ("operational", "invocations.jsonl")

_DIGEST_PREFIX_LENGTH = len("sha256:")
_IDENTITY_HEX_LENGTH = 16

#: Process-scoped mint sequence: two default mints in the same microsecond on
#: the same pid still yield distinct identities.
_MINT_SEQUENCE = itertools.count(1)


class InvocationIdentityError(Exception):
    """A malformed, corrupt, or unlinkable invocation-identity record."""


class LegacyRunIdentityError(ValueError):
    """A NEW identity tried to reuse the retired shared ``crawler-run`` constant."""


def invocation_ledger_path(records_root: Path) -> Path:
    """Return the append-only invocation ledger below one records root."""

    return Path(records_root).joinpath(*INVOCATIONS_RELATIVE_PATH)


def _identity_digest(payload: Mapping[str, Any]) -> str:
    """Return the short hex identity component of one canonical payload."""

    digest = stable_hash(dict(payload))
    return digest[_DIGEST_PREFIX_LENGTH : _DIGEST_PREFIX_LENGTH + _IDENTITY_HEX_LENGTH]


def mint_run_id(
    *,
    machine_id: str,
    hostname: str,
    pid: int,
    started_at: str,
    sequence: Optional[int] = None,
) -> str:
    """Mint one per-invocation run identity from time, host, and pid.

    Parameters
    ----------
    machine_id, hostname, pid, started_at:
        Identity of the starting process invocation.
    sequence:
        Optional process-scoped disambiguator for same-microsecond mints.

    Returns
    -------
    str
        Deterministic ``run-<16 hex>`` digest; never the legacy constant.
    """

    return "run-" + _identity_digest(
        {
            "schema_version": INVOCATION_IDENTITY_SCHEMA_VERSION,
            "axis": "run_id",
            "machine_id": machine_id,
            "hostname": hostname,
            "pid": pid,
            "started_at": started_at,
            "sequence": sequence,
        }
    )


def mint_default_run_id() -> str:
    """Mint a run identity for the CURRENT process invocation.

    Returns
    -------
    str
        Fresh ``run-<16 hex>`` identity, unique per call within this process.
    """

    node = platform.node() or "unknown-machine"
    return mint_run_id(
        machine_id=node,
        hostname=node,
        pid=os.getpid(),
        started_at=utc_now(),
        sequence=next(_MINT_SEQUENCE),
    )


def _mint_invocation_id(
    *, run_id: str, machine_id: str, hostname: str, pid: int, started_at: str, sequence: int
) -> str:
    """Mint the ledger-row identity of one process invocation."""

    return "inv-" + _identity_digest(
        {
            "schema_version": INVOCATION_IDENTITY_SCHEMA_VERSION,
            "axis": "invocation_id",
            "run_id": run_id,
            "machine_id": machine_id,
            "hostname": hostname,
            "pid": pid,
            "started_at": started_at,
            "sequence": sequence,
        }
    )


def validate_invocation_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one invocation row and return its canonical copy.

    Raises
    ------
    InvocationIdentityError
        If a field is missing/mistyped, an enum value leaves its closed
        vocabulary, or the content address disagrees with the payload.
    """

    if not isinstance(row, Mapping):
        raise InvocationIdentityError("invocation row is not an object")
    if row.get("schema_version") != INVOCATION_IDENTITY_SCHEMA_VERSION:
        raise InvocationIdentityError(
            f"invocation row declares unknown schema {row.get('schema_version')!r}"
        )
    event = row.get("event")
    if event not in _INVOCATION_EVENTS:
        raise InvocationIdentityError(
            f"invocation event {event!r} is outside the closed vocabulary"
        )
    for field_name in ("invocation_id", "run_id", "machine_id", "hostname", "started_at"):
        if not isinstance(row.get(field_name), str) or not row[field_name].strip():
            raise InvocationIdentityError(f"invocation row has an empty {field_name}")
    if row["run_id"] == LEGACY_RUN_ID:
        raise InvocationIdentityError(
            "invocation row carries the retired shared run identity; the ledger "
            "is new-era only (historical rows live outside it)"
        )
    pid = row.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
        raise InvocationIdentityError("invocation row pid must be a positive integer")
    if row.get("invocation_origin") not in {origin.value for origin in InvocationOrigin}:
        raise InvocationIdentityError(
            f"invocation origin {row.get('invocation_origin')!r} is outside the closed vocabulary"
        )
    wake_episode_id = row.get("wake_episode_id")
    if wake_episode_id is not None and (
        not isinstance(wake_episode_id, str) or not wake_episode_id.strip()
    ):
        raise InvocationIdentityError("invocation row wake_episode_id must be a string or null")
    succession_fields = ("predecessor_invocation_id", "reason")
    if event == "invocation_succession":
        for field_name in succession_fields:
            if not isinstance(row.get(field_name), str) or not row[field_name].strip():
                raise InvocationIdentityError(f"succession row has an empty {field_name}")
        if row["reason"] not in SUCCESSION_REASONS:
            raise InvocationIdentityError(
                f"succession reason {row['reason']!r} is outside the closed vocabulary"
            )
    else:
        for field_name in succession_fields:
            if field_name in row:
                raise InvocationIdentityError(
                    f"invocation_start row may not carry {field_name}: an untyped "
                    "restart must be visible as one"
                )
    validated = dict(json.loads(canonical_json_bytes(dict(row)).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise InvocationIdentityError(
            f"invocation row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256", None)
    if not isinstance(digest, str) or row_content_address(validated) != digest:
        raise InvocationIdentityError("invocation row digest does not match its payload")
    validated["row_sha256"] = digest
    return validated


def _read_invocation_rows(path: Path) -> tuple[JsonObject, ...]:
    """Read one invocation ledger, tolerating only a torn final line.

    A crash mid-append leaves at most one non-newline-terminated tail, which
    is dropped (that invocation durably recorded nothing). A malformed
    INTERIOR line is hard corruption and refuses loudly: silently skipping it
    could hide a process invocation from the one-leg certification.
    """

    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return ()
    lines = content.split(b"\n")
    if lines and lines[-1] != b"":
        lines = lines[:-1]
    rows: list[JsonObject] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvocationIdentityError(
                f"invocation ledger has a corrupt interior row at {path}:{index + 1}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise InvocationIdentityError(
                f"invocation ledger row {index + 1} at {path} is not an object"
            )
        rows.append(validate_invocation_row(value))
    return tuple(rows)


def load_invocation_rows(records_root: Path) -> tuple[JsonObject, ...]:
    """Load every validated invocation row in ledger order."""

    return _read_invocation_rows(invocation_ledger_path(records_root))


def _append_invocation_row(records_root: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one validated invocation row under the ledger's exclusive lock."""

    validated = validate_invocation_row(row)
    path = invocation_ledger_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _read_invocation_rows(path)
        for candidate in existing:
            if candidate["invocation_id"] == validated["invocation_id"]:
                if candidate["row_sha256"] == validated["row_sha256"]:
                    return candidate
                raise InvocationIdentityError(
                    f"invocation {validated['invocation_id']} is already recorded "
                    "with a different payload"
                )
        # A torn tail is a crashed append that never happened; heal it under
        # the exclusive lock so this append starts on a line boundary.
        content = path.read_bytes()
        if content and not content.endswith(b"\n"):
            os.ftruncate(handle.fileno(), content.rfind(b"\n") + 1)
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def _normalized_origin(invocation_origin: Union[InvocationOrigin, str]) -> str:
    """Return one closed-vocabulary origin value."""

    try:
        return InvocationOrigin(invocation_origin).value
    except ValueError as exc:
        raise InvocationIdentityError(
            f"invocation origin {invocation_origin!r} is outside the closed vocabulary"
        ) from exc


def record_invocation_start(
    records_root: Path,
    *,
    machine_id: str,
    invocation_origin: Union[InvocationOrigin, str] = InvocationOrigin.ORDINARY_RUN,
    hostname: Optional[str] = None,
    pid: Optional[int] = None,
    started_at: Optional[str] = None,
    wake_episode_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> JsonObject:
    """Durably record one process invocation, minting its run identity.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    machine_id:
        Stable machine identity of the invoking host.
    invocation_origin:
        Closed :class:`InvocationOrigin` value for this start.
    hostname, pid, started_at:
        Host identity; defaulted from the current process when omitted.
    wake_episode_id:
        Optional wake episode this invocation serves.
    run_id:
        Explicit run identity (e.g. a wake callback replaying its launching
        config). Omitted, a fresh identity is minted. The legacy constant is
        refused: era rule.

    Returns
    -------
    dict[str, Any]
        The validated, content-addressed invocation row.
    """

    if run_id == LEGACY_RUN_ID:
        raise LegacyRunIdentityError(
            "run_id 'crawler-run' is the retired shared constant: a NEW invocation "
            "must carry its own minted identity (historical rows keep it; era rule)"
        )
    resolved_hostname = hostname or platform.node() or "unknown-host"
    resolved_pid = pid if pid is not None else os.getpid()
    resolved_started_at = started_at or utc_now()
    sequence = next(_MINT_SEQUENCE)
    resolved_run_id = run_id or mint_run_id(
        machine_id=machine_id,
        hostname=resolved_hostname,
        pid=resolved_pid,
        started_at=resolved_started_at,
        sequence=sequence,
    )
    body: JsonObject = {
        "schema_version": INVOCATION_IDENTITY_SCHEMA_VERSION,
        "event": "invocation_start",
        "invocation_id": _mint_invocation_id(
            run_id=resolved_run_id,
            machine_id=machine_id,
            hostname=resolved_hostname,
            pid=resolved_pid,
            started_at=resolved_started_at,
            sequence=sequence,
        ),
        "run_id": resolved_run_id,
        "machine_id": machine_id,
        "hostname": resolved_hostname,
        "pid": resolved_pid,
        "started_at": resolved_started_at,
        "invocation_origin": _normalized_origin(invocation_origin),
    }
    if wake_episode_id is not None:
        body["wake_episode_id"] = wake_episode_id
    return _append_invocation_row(records_root, stamp_row_digest(body))


def record_invocation_succession(
    records_root: Path,
    *,
    predecessor_invocation_id: str,
    machine_id: str,
    reason: str = "autonomous-pause-resume",
    invocation_origin: Union[InvocationOrigin, str] = InvocationOrigin.WAKE_CALLBACK,
    hostname: Optional[str] = None,
    pid: Optional[int] = None,
    started_at: Optional[str] = None,
    wake_episode_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> JsonObject:
    """Durably record one TYPED pause/resume invocation (B3-XQ-12).

    The row is itself an invocation row -- one process start -- carrying the
    closed-vocabulary reason and the exact predecessor it succeeds. Omitting
    ``run_id`` continues the predecessor's run identity; an explicit fresh
    identity is also a preserved succession. The legacy constant is refused.

    Raises
    ------
    InvocationIdentityError
        If the reason leaves the closed vocabulary or the predecessor has no
        durable invocation row.
    """

    if reason not in SUCCESSION_REASONS:
        raise InvocationIdentityError(
            f"succession reason {reason!r} is outside the closed vocabulary "
            f"{sorted(SUCCESSION_REASONS)}"
        )
    if run_id == LEGACY_RUN_ID:
        raise LegacyRunIdentityError(
            "run_id 'crawler-run' is the retired shared constant: a NEW invocation "
            "must carry its own minted identity (historical rows keep it; era rule)"
        )
    predecessors = {
        row["invocation_id"]: row for row in load_invocation_rows(records_root)
    }
    predecessor = predecessors.get(predecessor_invocation_id)
    if predecessor is None:
        raise InvocationIdentityError(
            f"succession names predecessor {predecessor_invocation_id!r} with no "
            "durable invocation row: an unlinkable resume is an untyped restart"
        )
    resolved_hostname = hostname or platform.node() or "unknown-host"
    resolved_pid = pid if pid is not None else os.getpid()
    resolved_started_at = started_at or utc_now()
    sequence = next(_MINT_SEQUENCE)
    resolved_run_id = run_id or str(predecessor["run_id"])
    body: JsonObject = {
        "schema_version": INVOCATION_IDENTITY_SCHEMA_VERSION,
        "event": "invocation_succession",
        "invocation_id": _mint_invocation_id(
            run_id=resolved_run_id,
            machine_id=machine_id,
            hostname=resolved_hostname,
            pid=resolved_pid,
            started_at=resolved_started_at,
            sequence=sequence,
        ),
        "run_id": resolved_run_id,
        "machine_id": machine_id,
        "hostname": resolved_hostname,
        "pid": resolved_pid,
        "started_at": resolved_started_at,
        "invocation_origin": _normalized_origin(invocation_origin),
        "predecessor_invocation_id": predecessor_invocation_id,
        "reason": reason,
    }
    if wake_episode_id is not None:
        body["wake_episode_id"] = wake_episode_id
    return _append_invocation_row(records_root, stamp_row_digest(body))


@dataclass(frozen=True)
class SingleInvocationCertification:
    """B3-XQ-12 one-leg verdict for one leg generation.

    ``valid`` is True only when the generation's durably joined runs are all
    attributed to invocation rows forming ONE typed chain: a single root
    ``invocation_start`` and, for every later invocation, an
    ``invocation_succession`` linking the chain's previous member. ``reason``
    stays inside :data:`CERTIFICATION_REASONS`.
    """

    valid: bool
    reason: str
    generation_id: str
    invocation_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    unattributed_run_ids: tuple[str, ...]


def _generation_invocation_rows(
    records_root: Path, generation_id: str
) -> tuple[tuple[str, ...], tuple[JsonObject, ...], tuple[str, ...]]:
    """Join one generation's durable runs to their invocation rows."""

    from menagerie.crawler.leg_generation import (  # noqa: PLC0415 -- lazy: avoids a heavy leaf import
        generation_run_ids,
    )

    run_ids = tuple(sorted(generation_run_ids(records_root, generation_id)))
    rows = tuple(
        row for row in load_invocation_rows(records_root) if row["run_id"] in set(run_ids)
    )
    attributed = {str(row["run_id"]) for row in rows}
    unattributed = tuple(sorted(set(run_ids) - attributed))
    return run_ids, rows, unattributed


def leg_valid_single_invocation(
    records_root: Path, generation_id: str
) -> SingleInvocationCertification:
    """Certify B3-XQ-12 "one leg" for one leg generation.

    One logical generation AND one process invocation: a typed autonomous
    pause/resume chain preserves the certification; an untyped process restart
    (a second ``invocation_start`` joining the generation, or a succession
    that does not link the chain) voids it. Runs with no invocation row --
    including every legacy ``crawler-run`` era run -- are unattributable and
    never certify (era-honest: old legs are uncertifiable, not upgraded).
    """

    run_ids, rows, unattributed = _generation_invocation_rows(records_root, generation_id)
    invocation_ids = tuple(str(row["invocation_id"]) for row in rows)

    def verdict(valid: bool, reason: str) -> SingleInvocationCertification:
        assert reason in CERTIFICATION_REASONS
        return SingleInvocationCertification(
            valid=valid,
            reason=reason,
            generation_id=generation_id,
            invocation_ids=invocation_ids,
            run_ids=run_ids,
            unattributed_run_ids=unattributed,
        )

    if not run_ids:
        return verdict(False, "no-joined-runs")
    if unattributed:
        return verdict(False, "unattributed-run-identity")
    if rows[0]["event"] != "invocation_start":
        return verdict(False, "broken-succession-chain")
    for previous, current in zip(rows, rows[1:]):
        if current["event"] == "invocation_start":
            return verdict(False, "untyped-restart")
        if current["predecessor_invocation_id"] != previous["invocation_id"]:
            return verdict(False, "broken-succession-chain")
    if len(rows) == 1:
        return verdict(True, "single-invocation")
    return verdict(True, "typed-succession-chain")


def close_leg_citations(records_root: Path, generation_id: str) -> tuple[JsonObject, ...]:
    """Derive the exact leg citations one close must name.

    One citation per durably joined run, sorted by run identity, each naming
    every invocation row carrying that run in ledger order. A run with no
    invocation row is the legacy era and cites :data:`PRE_LEDGER_ERA`
    honestly instead of inventing an attribution.
    """

    run_ids, rows, _unattributed = _generation_invocation_rows(records_root, generation_id)
    citations: list[JsonObject] = []
    for run_id in run_ids:
        invocation_ids = [
            str(row["invocation_id"]) for row in rows if row["run_id"] == run_id
        ]
        citations.append(
            {
                "run_id": run_id,
                "invocation_ids": invocation_ids,
                "era": INVOCATION_LEDGER_ERA if invocation_ids else PRE_LEDGER_ERA,
            }
        )
    return tuple(citations)


def void_interventions_by_invocation(
    void_rows: Iterable[Mapping[str, Any]],
    invocation_rows: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Count requeue-void appends as interventions, grouped by invocation.

    A void row (the ``void_of`` family) is an operator intervention; the
    ledger becomes machine-checkable when each one is attributed to the
    process invocation that appended it. Attribution keys on the void row's
    ``run_id`` mapped through the invocation ledger (the LAST invocation
    carrying that run identity, in ledger order, wins -- a typed resume
    inherits its predecessor's run). Rows without a resolvable run identity
    -- the entire historical void population -- count under
    :data:`PRE_LEDGER_ERA`, never silently.
    """

    invocation_by_run: dict[str, str] = {}
    for row in invocation_rows:
        invocation_by_run[str(row["run_id"])] = str(row["invocation_id"])
    counts: dict[str, int] = {}
    for row in void_rows:
        if not isinstance(row, Mapping) or "void_of" not in row:
            continue
        run_id = row.get("run_id")
        key = (
            invocation_by_run.get(run_id, PRE_LEDGER_ERA)
            if isinstance(run_id, str)
            else PRE_LEDGER_ERA
        )
        counts[key] = counts.get(key, 0) + 1
    return counts
