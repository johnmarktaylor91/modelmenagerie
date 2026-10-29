"""Durable parking rows for terminals the recording path refused to record.

Only one reducer refusal used to park (the grant-pending supersession class);
every other refusal of a prepared terminal -- with the minimal fallback refusing
identically -- left the model with NO record and NO claim. The tripwire at the
completion partition is correct (a silent hole is the harm), but the blast
radius was not: the leg paid its full compute cost and then died at close with
``PartitionError``, permanently, every leg (pilot T1-03: 19 terminal-unrecordable
events across 8 models; the m9617 env-generation contradiction and the
m3669/m10759 cap-not-reached shapes). The adjudicated fix parks the MODEL here
instead: a durable, typed refusal row -- stable id, work generation, refusal
class, and an evidence digest over the exact errors and refused candidate --
claims the model as machine-withheld so the close completes honestly under a
distinct, never-"complete" status. NOTHING is softened: the reducer refusal
stays a refusal, no record is minted, nothing publishes, and release is a real
recorded revision (a later leg succeeding after the root cause is fixed) or an
explicit operator void -- never machine time.

Unlike the capability valves, a park here is evidence the RECORDING PATH is
broken for this candidate, so a persisted row never short-circuits a resume
attempt once the member is admitted: the ladder re-parks idempotently if the
refusal still holds (fail-open toward attempting, exactly like the sibling
valves' doctrine). ADMISSION itself is bounded by D-48's machine release lane
(:mod:`menagerie.crawler.park_release`): a live row machine-readmits its member
for ONE fresh work generation per change of the running code tree -- the
``tree_sha`` each row records, which joins the park FACT identity so a re-park
under a new tree appends a fresh governing row -- and under an unchanged tree
the member stays machine-withheld (selection clause 5) until an operator void.
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
    stable_hash,
    stamp_row_digest,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import read_with_tail_recovery

REFUSAL_PARK_SCHEMA_VERSION = "menagerie.crawler.terminal-refusal-park.v1"

REFUSAL_PARK_VOID_SCHEMA_VERSION = "menagerie.crawler.terminal-refusal-park-void.v1"

REFUSAL_PARK_DISPOSITION = "review-held:terminal-recording-refused"
"""Greppable disposition for a model whose terminal could not be recorded.

The model is not waiting on machine capability: its work RAN and produced a
terminal disposition the recording path then refused to write (intended append
refused AND the minimal fallback refused). What is missing is a root-cause fix
plus an operator decision, so the disposition names the review hold. It is
deliberately NOT ``failed:*`` (no canonical fact was recorded at all) and
deliberately not a canonical model terminal (minting one is exactly what the
reducer refused); this row only accounts the withheld model.
"""

REFUSAL_PARK_EVENT_DISPOSITION = "terminal-refusal-parked"
"""Operational-event details disposition attached to the park's disclosure."""

REFUSAL_PARK_RELATIVE_PATH = Path("intake-extensions") / "terminal-refusal-parks.jsonl"

#: Evidence fields whose digest is the row's typed refusal identity.
_EVIDENCE_FIELDS = (
    "refusal_class",
    "record_error_type",
    "record_error",
    "fallback_error_type",
    "fallback_error",
    "intended_status_code",
    "intended_reason_code",
    "blocked_parent_revision",
    "candidate_sha256",
)


class RefusalParkError(ValueError):
    """Raised when a terminal-refusal park row or void row is malformed."""


def refusal_park_path(records_root: Path) -> Path:
    """Return the canonical terminal-refusal park ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only terminal-refusal park JSONL path.
    """

    return records_root / REFUSAL_PARK_RELATIVE_PATH


def build_refusal_park_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    refusal_class: str,
    record_error_type: str,
    record_error: str,
    fallback_error_type: Optional[str],
    fallback_error: Optional[str],
    intended_status_code: str,
    intended_reason_code: Optional[str],
    blocked_parent_revision: Optional[str],
    candidate_sha256: Optional[str],
    tree_sha: Optional[str] = None,
) -> JsonObject:
    """Assemble one complete, self-describing terminal-refusal park row.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the scheduled work generation whose terminal
        could not be recorded.
    campaign_id, run_id, machine_id:
        Campaign and host provenance for the refusing run.
    created_at:
        UTC timestamp of the refusal.
    refusal_class:
        Typed refusal class -- the exception type of the refusing append.
    record_error_type, record_error:
        Exception type and message raised by the intended terminal append.
    fallback_error_type, fallback_error:
        Exception type and message raised by the minimal fallback append, when
        one was attempted.
    intended_status_code, intended_reason_code:
        Disposition the driver tried to record.
    blocked_parent_revision:
        ``record_revision`` of the model's current revision at refusal time, or
        ``None`` for a first terminal. Release keys on it: ANY newer landed
        revision (any record at all, for a parent-less park) consumes the row.
    candidate_sha256:
        Content digest of the exact prepared candidate the reducer refused,
        when preparation itself succeeded.
    tree_sha:
        Identity of the running code tree the refusal happened under
        (:func:`~menagerie.crawler.park_release.current_park_tree_identity`),
        or ``None`` when underivable. It joins the park FACT identity: the
        same refusal re-derived under a NEW tree is a genuinely new fact
        (one machine readmission was spent proving it), so it appends a new
        governing row and closes D-48's one-generation-per-tree lane until
        the next tree change. An absent tree fails CLOSED for readmission.

    Returns
    -------
    dict[str, Any]
        Content-addressed park row.

    Raises
    ------
    RefusalParkError
        If an identity or evidence field is empty or mistyped.
    """

    if not campaign_id or campaign_id == str(None):
        # T2-12: the pilot froze campaign_id="None" (a stringified absence)
        # into 62 row digests. The absence must refuse at BUILD time; read-side
        # validators stay tolerant so those historical rows remain readable.
        raise ValueError("campaign_id must name a real campaign, never a stringified absence")
    refusal: JsonObject = {
        "refusal_class": refusal_class,
        "record_error_type": record_error_type,
        "record_error": record_error,
        "fallback_error_type": fallback_error_type,
        "fallback_error": fallback_error,
        "intended_status_code": intended_status_code,
        "intended_reason_code": intended_reason_code,
        "blocked_parent_revision": blocked_parent_revision,
        "candidate_sha256": candidate_sha256,
    }
    refusal["evidence_sha256"] = _evidence_digest(refusal)
    # D-48: the hint names the REAL levers. The old wording told operators a
    # requeue would land a releasing revision, but selection clause 5 fires
    # before the grant arm and no grant binds this disposition -- the promised
    # exit was mechanically unreachable (the no-head-park fixture proves it).
    recheck_hint = (
        f"model {stable_id!r}'s work generation {work_id!r} produced a "
        f"{intended_status_code!r} terminal the recording path refused to write "
        f"({refusal_class}: {record_error}). No record was minted and nothing was "
        "published. Two real exits: (1) fix the recording path or the candidate "
        "derivation in the code tree -- under a CHANGED tree the machine release "
        "lane readmits this member for one fresh generation and a landed revision "
        "consumes this row (under the same tree it stays machine-withheld; requeue "
        "grants cannot bind it); or (2) an operator voids this park row explicitly."
    )
    row: JsonObject = {
        "schema_version": REFUSAL_PARK_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": REFUSAL_PARK_DISPOSITION,
        "tree_sha": tree_sha,
        "refusal": refusal,
        "recheck_hint": recheck_hint,
    }
    row = stamp_row_digest(row)
    return validate_refusal_park_row(row)


def _evidence_digest(refusal: Mapping[str, Any]) -> str:
    """Return the content digest over the closed refusal-evidence fields."""

    return stable_hash({field: refusal.get(field) for field in _EVIDENCE_FIELDS})


def validate_refusal_park_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one terminal-refusal park row and return its canonical copy.

    Parameters
    ----------
    row:
        Candidate park row.

    Returns
    -------
    dict[str, Any]
        Validated row.

    Raises
    ------
    RefusalParkError
        If a required field is missing, mistyped, or either content digest
        (evidence or row) disagrees with its payload.
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
        "refusal",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise RefusalParkError(f"terminal-refusal park row is missing {', '.join(sorted(missing))}")
    if row["schema_version"] != REFUSAL_PARK_SCHEMA_VERSION:
        raise RefusalParkError(
            f"terminal-refusal park row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] != REFUSAL_PARK_DISPOSITION:
        raise RefusalParkError(
            f"terminal-refusal park row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise RefusalParkError(f"terminal-refusal park row has an empty {field}")
    refusal = row["refusal"]
    if not isinstance(refusal, Mapping):
        raise RefusalParkError("terminal-refusal park row must record its refusal evidence")
    for field in ("refusal_class", "record_error_type", "record_error", "intended_status_code"):
        if not isinstance(refusal.get(field), str) or not refusal.get(field):
            raise RefusalParkError(f"terminal-refusal park row must name {field}")
    for field in (
        "fallback_error_type",
        "fallback_error",
        "intended_reason_code",
        "blocked_parent_revision",
        "candidate_sha256",
    ):
        value = refusal.get(field)
        if value is not None and (not isinstance(value, str) or not value):
            raise RefusalParkError(f"terminal-refusal park row has a non-null empty {field}")
    # D-48 era tolerance: pre-lane rows lack ``tree_sha`` entirely and stay
    # readable; a new row records its parking tree or an explicit ``None``
    # (underivable, which fails closed for machine readmission).
    tree_sha = row.get("tree_sha")
    if tree_sha is not None and (not isinstance(tree_sha, str) or not tree_sha):
        raise RefusalParkError("terminal-refusal park row has a non-null empty tree_sha")
    if refusal.get("evidence_sha256") != _evidence_digest(refusal):
        raise RefusalParkError(
            "terminal-refusal park row evidence digest does not match its payload"
        )
    if not isinstance(row["recheck_hint"], str) or not row["recheck_hint"].strip():
        raise RefusalParkError("terminal-refusal park row must name what releases it later")
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise RefusalParkError(
            f"refusal park row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise RefusalParkError("terminal-refusal park row digest does not match its payload")
    validated["row_sha256"] = digest
    return validated


def build_refusal_park_void(
    *,
    void_of: str,
    reason: str,
    voided_by: str,
    created_at: str,
) -> JsonObject:
    """Assemble one operator void event for an existing park row.

    Parameters
    ----------
    void_of:
        ``row_sha256`` of the park row the operator releases.
    reason:
        Operator-recorded justification; voids are decisions, never mechanics.
    voided_by:
        Deciding operator identity.
    created_at:
        UTC timestamp of the decision.

    Returns
    -------
    dict[str, Any]
        Content-addressed void row.

    Raises
    ------
    RefusalParkError
        If a field is empty.
    """

    row: JsonObject = {
        "schema_version": REFUSAL_PARK_VOID_SCHEMA_VERSION,
        "void_of": void_of,
        "reason": reason,
        "voided_by": voided_by,
        "created_at": created_at,
    }
    row = stamp_row_digest(row)
    return validate_refusal_park_void(row)


def validate_refusal_park_void(row: Mapping[str, Any]) -> JsonObject:
    """Validate one park void row and return its canonical copy.

    Parameters
    ----------
    row:
        Candidate void row.

    Returns
    -------
    dict[str, Any]
        Validated row.

    Raises
    ------
    RefusalParkError
        If a required field is missing or empty, or the digest disagrees.
    """

    required = ("schema_version", "void_of", "reason", "voided_by", "created_at", "row_sha256")
    missing = [field for field in required if field not in row]
    if missing:
        raise RefusalParkError(
            f"terminal-refusal park void is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != REFUSAL_PARK_VOID_SCHEMA_VERSION:
        raise RefusalParkError(
            f"terminal-refusal park void declares unknown schema {row['schema_version']!r}"
        )
    for field in ("void_of", "reason", "voided_by", "created_at"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise RefusalParkError(f"terminal-refusal park void has an empty {field}")
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise RefusalParkError(
            f"refusal park row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise RefusalParkError("terminal-refusal park void digest does not match its payload")
    validated["row_sha256"] = digest
    return validated


def validate_refusal_park_ledger_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one ledger line as either a park row or a void row.

    Parameters
    ----------
    row:
        Decoded JSONL object.

    Returns
    -------
    dict[str, Any]
        Validated park or void row.

    Raises
    ------
    RefusalParkError
        If the row matches neither schema.
    """

    if not isinstance(row, Mapping):
        raise RefusalParkError("terminal-refusal park ledger row is not an object")
    if row.get("schema_version") == REFUSAL_PARK_VOID_SCHEMA_VERSION:
        return validate_refusal_park_void(row)
    return validate_refusal_park_row(row)


def _park_fact_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    """Return the refusal FACT identity one park row records.

    Deliberately EXCLUDES the candidate digest and the fallback fields: a
    resume re-derives the same deterministic refusal over freshly persisted
    attempts, so the candidate's incidental identities (attempt ids,
    timestamps) drift while the refusal fact does not. A refusal whose CLASS,
    MESSAGE, or blocked parent changes is a genuinely new fact and appends a
    new governing row.

    D-48: the parking TREE is part of the fact. The same refusal re-derived
    under a changed code tree consumed the machine release lane's one fresh
    generation for that tree, so it must append a fresh governing row (whose
    recorded tree closes the lane again) rather than no-op against the old
    row -- otherwise the stale recorded tree would re-open the lane every leg.

    Parameters
    ----------
    row:
        Validated park row.

    Returns
    -------
    tuple[str, str, str, str, str, str]
        Stable ID, work generation, blocked parent revision, refusal class,
        the intended append's error message, and the parking tree identity.
    """

    refusal = row.get("refusal", {})
    return (
        str(row.get("stable_id")),
        str(row.get("work_id")),
        str(refusal.get("blocked_parent_revision")),
        str(refusal.get("refusal_class")),
        str(refusal.get("record_error")),
        str(row.get("tree_sha")),
    )


def _split_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonObject], set[str]]:
    """Split validated ledger rows into park rows and voided digests."""

    parks: list[JsonObject] = []
    voided: set[str] = set()
    for row in rows:
        if row.get("schema_version") == REFUSAL_PARK_VOID_SCHEMA_VERSION:
            voided.add(str(row["void_of"]))
        else:
            parks.append(dict(row))
    return parks, voided


def append_refusal_park_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one terminal-refusal park durably and idempotently.

    Parameters
    ----------
    path:
        Destination terminal-refusal park JSONL.
    row:
        Complete typed park row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    RefusalParkError
        If the row is malformed.

    Notes
    -----
    Idempotency is keyed on the refusal FACT identity (see
    :func:`_park_fact_key`) among UNVOIDED rows only. A resume re-deriving the
    same deterministic refusal is a no-op returning the first persisted row; a
    park an operator VOIDED never swallows a fresh fire.
    """

    validated = validate_refusal_park_row(row)
    key = _park_fact_key(validated)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_refusal_park_lines(_ledger_bytes(path, handle), path)
        parks, voided = _split_rows(existing)
        generation = [
            item
            for item in parks
            if (str(item["stable_id"]), str(item["work_id"])) == (key[0], key[1])
        ]
        # Idempotency consults every LIVE row of the generation (RULING W7
        # clause 9: per-row liveness -- an earlier unvoided fact stays live
        # beside a later one, so replaying ANY still-live fact is a no-op,
        # newest match returned). A fact whose every prior row was voided
        # appends fresh: an operator void never swallows a fresh fire.
        for item in reversed(generation):
            if str(item["row_sha256"]) not in voided and _park_fact_key(item) == key:
                return item
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def append_refusal_park_void(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one operator park void durably and idempotently.

    Parameters
    ----------
    path:
        Terminal-refusal park JSONL holding the row being voided.
    row:
        Complete typed void row.

    Returns
    -------
    dict[str, Any]
        Validated persisted void row.

    Raises
    ------
    RefusalParkError
        If the void is malformed or references no persisted park row.
    """

    validated = validate_refusal_park_void(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_refusal_park_lines(_ledger_bytes(path, handle), path)
        parks, _voided = _split_rows(existing)
        target = str(validated["void_of"])
        if not any(str(item["row_sha256"]) == target for item in parks):
            raise RefusalParkError(
                f"terminal-refusal park void references an unknown row: {target}"
            )
        prior = [
            item
            for item in existing
            if item.get("schema_version") == REFUSAL_PARK_VOID_SCHEMA_VERSION
            and str(item["void_of"]) == target
        ]
        if prior:
            if str(prior[0]["row_sha256"]) == str(validated["row_sha256"]):
                # Byte-identical re-void: idempotent.
                return dict(prior[0])
            # RULING W7 clause 9 (closing O C-6's silent no-op): a release
            # must name a CURRENTLY-LIVE row; the prior release stands.
            raise RefusalParkError(
                f"terminal-refusal park row {target} was already voided; the "
                "prior release stands (a release must name a live row)"
            )
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
        prefix_check=lambda retained: _decode_refusal_park_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def _decode_refusal_park_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one park ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RefusalParkError(
                f"terminal-refusal park ledger {path} has an unreadable row at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise RefusalParkError(
                f"terminal-refusal park ledger {path} has a non-object row at line {index}"
            )
        rows.append(validate_refusal_park_ledger_row(decoded))
    return tuple(rows)


def current_refusal_park_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[JsonObject, ...]:
    """Return EVERY unvoided park row -- per-row liveness (RULING W7 clause 9).

    Withholding liveness is derived per row, never per (stable_id, work_id)-
    latest. The old latest-per-key supersession ("a newer refusal permanently
    supersedes the older fact") let one lawful void of the LATER of two
    unvoided parks silently erase the EARLIER unvoided, still-verdict-bearing
    row -- the O A3 defect, verified by execution. Every unvoided park is
    live; releasing a generation means releasing every live row it holds,
    each by name.

    Parameters
    ----------
    rows:
        Already-validated park and void rows in ledger order.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Live rows ordered by ``(stable_id, work_id, ledger position)``.
    """

    parks, voided = _split_rows(rows)
    live = [
        (str(row["stable_id"]), str(row["work_id"]), position, dict(row))
        for position, row in enumerate(parks)
        if str(row["row_sha256"]) not in voided
    ]
    return tuple(row for _sid, _wid, _position, row in sorted(live, key=lambda item: item[:3]))


def existing_refusal_park_release(path: Path, row_sha256: str) -> Optional[JsonObject]:
    """Return the release that already voided one park row, if any.

    The D-48 machine release lane re-derives releases from released-ness on
    every close, so a re-close consults the standing release instead of
    appending a second one -- the append door itself REFUSES a different
    release of an already-released row (RULING W7 clause 9, closing O C-6).
    """

    if not path.is_file():
        return None
    for item in _decode_refusal_park_lines(_ledger_bytes(path), path):
        if (
            item.get("schema_version") == REFUSAL_PARK_VOID_SCHEMA_VERSION
            and str(item.get("void_of")) == row_sha256
        ):
            return dict(item)
    return None


def load_refusal_park_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load the CURRENT unvoided terminal-refusal park per work generation.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Every live (unvoided) park row, per-row liveness (RULING W7 clause 9).
    """

    rows: list[JsonObject] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.extend(_decode_refusal_park_lines(_ledger_bytes(path), path))
    return current_refusal_park_rows(rows)


def refusal_park_released(
    row: Mapping[str, Any], current_record: Optional[Mapping[str, Any]]
) -> bool:
    """Return whether a landed revision releases (consumes) one park row.

    A park row is released exactly when a REAL canonical revision landed past
    the refusal it recorded: any record at all for a parent-less park (the
    refused terminal was the model's first), or any revision other than the
    recorded blocked parent for a refused supersession. An absent record
    releases nothing -- the refusal is still the model's only accounting.

    The record every CALLER must supply is the model's RAW LEDGER HEAD, not a
    dependency-currency projection of it (BD-1): release is a fact about what
    landed, and a head the live authority cannot vouch for still landed. See
    :func:`~menagerie.crawler.grant_parks.grant_park_released` for the
    step-100 evidence.

    Parameters
    ----------
    row:
        Validated park row.
    current_record:
        The model's raw ledger head, or ``None`` when it holds no revision.

    Returns
    -------
    bool
        Whether the park row is consumed by the record.
    """

    if not isinstance(current_record, Mapping):
        return False
    blocked = row.get("refusal", {}).get("blocked_parent_revision")
    if blocked is None:
        return True
    revision = current_record.get("record_revision")
    return isinstance(revision, str) and revision != blocked
