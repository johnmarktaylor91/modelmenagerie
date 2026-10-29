"""Durable parking rows for supersessions blocked only by a missing review grant.

A model whose current terminal is review-held is operator property: the reducer
refuses to let ANY new revision supersede it without a new explicit requeue
grant, and that tripwire is correct. What was NOT correct was the blast radius:
when a re-driven model finished fresh work, the refusal at the completion
partition failed the whole run (rung 12 legs 3 and 6, 42 hours of compute ending
in ``PartitionError`` twice) and bred the manual pre-granting dance of
2026-08-10. The adjudicated fix parks the blocked append here instead: a
durable, typed grant-pending row that keeps the review-held terminal governing,
closes the partition as ``terminal-partition-complete-with-review-holds``, and
lets the NEXT leg consume an operator grant and publish through the existing
requeue lane. The row embeds the exact pending revision, so the operator sees
WHAT a grant would publish before granting -- and nothing is ever auto-granted:
release is a human grant (or an explicit operator void), never machine time.
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

GRANT_PARK_SCHEMA_VERSION = "menagerie.crawler.grant-pending-supersession.v1"

GRANT_PARK_VOID_SCHEMA_VERSION = "menagerie.crawler.grant-pending-supersession-void.v1"

GRANT_PENDING_DISPOSITION = "review-held:supersession-pending-grant"
"""Greppable disposition for a supersession parked behind a missing grant.

Unlike the ``deferred:needs-<capability>`` valve dispositions, the parked model
is not waiting on machine capability at all: its work is DONE and its pending
revision is embedded in the row. What is missing is an operator decision --
the review-held terminal may only be superseded through a new explicit requeue
grant -- so the disposition names the review hold itself. It is deliberately
NOT ``failed:*`` (nothing about the model failed) and deliberately not a
canonical model terminal (the review-held terminal is the model's truthful
current record; this row only accounts the wait).
"""

GRANT_PARK_EVENT_DISPOSITION = "grant-pending-supersession-parked"
"""Operational-event details disposition emitted at every park fire."""

AWARD_PARK_DISPOSITION = "review-held:award-supersession-pending-grant"

#: P5 round-2 R2-1: the SELECTION EXCLUSION park. A review hold alone is a
#: grant RELEASE authority (``status.human_review.required`` is exactly what
#: lets a stage-matching requeue grant bind), so "excluded by review holds" was
#: never true — appending the PD-7 holds actually UN-barred four members. An
#: exclusion park is the durable operator instrument that makes the exclusion
#: real: its live predicate reads the hold (the exact review-held revision must
#: still govern), selection clause 5 makes a live park absolute in every mode,
#: and NO grant of any spelling releases it — only an explicit operator void,
#: or the held head lawfully moving through an operator lane.
SELECTION_EXCLUSION_DISPOSITION = "review-held:selection-exclusion"

#: The blocked_stage sentinel an exclusion park records. It is deliberately
#: OUTSIDE the closed grant-stage vocabulary the requeue tool accepts, so no
#: lawful grant can ever stage-match it.
SELECTION_EXCLUSION_STAGE = "selection-exclusion"
"""Disposition of a machine re-adjudication parked by the award-custody guard.

AW-1 (AUDIT100 section 2.1): a ``runs`` head is operator property, so a
machine re-adjudication yielding any worse disposition PARKS here -- it never
appends over the award and nothing is ever auto-granted. The row binds the
reducer-returned RAW blocked award head (the ten close-state refusal parks
all carried ``blocked_parent_revision: null`` because the old route read the
dependency-currency projection) and embeds the refused candidate so the
operator adjudicates from the row alone. Release is an award-naming requeue
grant (``requeue <id> --blocked-award-revision <revision>``) consumed by a
later leg, or an explicit operator void.
"""

AWARD_PARK_EVENT_DISPOSITION = "award-supersession-parked"
"""Operational-event details disposition emitted at every award-park fire."""

GRANT_PARK_RELATIVE_PATH = Path("intake-extensions") / "grant-pending-supersessions.jsonl"


class GrantParkError(ValueError):
    """Raised when a grant-pending park row or void row is malformed."""


def grant_park_path(records_root: Path) -> Path:
    """Return the canonical grant-pending park ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only grant-pending supersession JSONL path.
    """

    return records_root / GRANT_PARK_RELATIVE_PATH


def build_grant_park_row(
    *,
    stable_id: str,
    work_id: str,
    name: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
    blocked_parent_revision: str,
    blocked_stage: str,
    pending_model: Mapping[str, Any],
    disposition: str = GRANT_PENDING_DISPOSITION,
) -> JsonObject:
    """Assemble one complete, self-describing grant-pending park row.

    The row names the review-held revision that governs, the stage an operator
    grant must match, and embeds the exact pending revision the grant would
    publish, so the operator can adjudicate without re-running anything.

    Parameters
    ----------
    stable_id, work_id, name:
        Durable model identity and the completed scheduled work generation.
    campaign_id, run_id, machine_id:
        Campaign and host provenance for the run that derived the pending append.
    created_at:
        UTC timestamp of the refusal.
    blocked_parent_revision:
        Exact review-held ``record_revision`` the pending append may not
        supersede without a grant.
    blocked_stage:
        Recorded ``status.stage`` of the review-held terminal; a requeue grant
        must name this stage to be consumable.
    pending_model:
        Full prepared candidate revision the reducer refused solely for want of
        a grant.

    Returns
    -------
    dict[str, Any]
        Content-addressed park row.

    Raises
    ------
    GrantParkError
        If an identity field is empty or the pending revision is inconsistent
        with the blocked parent it claims.
    """

    if not campaign_id or campaign_id == str(None):
        # T2-12: the pilot froze campaign_id="None" (a stringified absence)
        # into 62 row digests. The absence must refuse at BUILD time; read-side
        # validators stay tolerant so those historical rows remain readable.
        raise ValueError("campaign_id must name a real campaign, never a stringified absence")
    if disposition not in {GRANT_PENDING_DISPOSITION, AWARD_PARK_DISPOSITION}:
        raise GrantParkError(f"grant-pending park row declares unknown disposition {disposition!r}")
    pending_status = pending_model.get("status") if isinstance(pending_model, Mapping) else None
    status_code = pending_status.get("code") if isinstance(pending_status, Mapping) else None
    if disposition == AWARD_PARK_DISPOSITION:
        recheck_hint = (
            f"model {stable_id!r}'s current head {blocked_parent_revision!r} is a runs "
            "AWARD and is operator property: the completed work generation "
            f"{work_id!r} produced the embedded worse re-adjudication, which no run "
            "may publish over the award. An operator releases the park by issuing an "
            f"award-naming grant (requeue {stable_id!r} --stage {blocked_stage!r} "
            f"--blocked-award-revision {blocked_parent_revision!r}) that the next leg "
            "consumes -- stamping the typed scar as the receipt -- or by voiding this "
            "park row; nothing is ever granted automatically."
        )
    else:
        recheck_hint = (
            f"model {stable_id!r}'s current terminal {blocked_parent_revision!r} is "
            f"review-held (stage {blocked_stage!r}) and is operator property: the completed "
            f"work generation {work_id!r} produced the embedded superseding revision, which "
            "this run may not publish without a new explicit requeue grant. An operator "
            f"releases the park by issuing a matching grant (requeue {stable_id!r} "
            f"--stage {blocked_stage!r}) that the next leg consumes and publishes, or by "
            "voiding this park row; nothing is ever granted automatically."
        )
    row: JsonObject = {
        "schema_version": GRANT_PARK_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": name,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": disposition,
        "pending": {
            "blocked_parent_revision": blocked_parent_revision,
            "blocked_stage": blocked_stage,
            "pending_status_code": status_code,
            "pending_reason_code": (
                pending_status.get("reason_code") if isinstance(pending_status, Mapping) else None
            ),
            "pending_model": json.loads(canonical_json_bytes(dict(pending_model)).decode("utf-8"))
            if isinstance(pending_model, Mapping)
            else pending_model,
            "pending_model_sha256": stable_hash(dict(pending_model))
            if isinstance(pending_model, Mapping)
            else None,
        },
        "recheck_hint": recheck_hint,
    }
    row = stamp_row_digest(row)
    return validate_grant_park_row(row)


def build_selection_exclusion_park_row(
    *,
    stable_id: str,
    name: str,
    campaign_id: str,
    machine_id: str,
    created_at: str,
    held_revision: str,
    held_status_code: str,
    decision: str,
    reason: str,
) -> JsonObject:
    """Assemble one selection-exclusion park row for a review-held terminal.

    The row binds the EXACT review-held revision that governs the member. Its
    live predicate (the grant-pending release rule) holds while that revision
    is still the raw head; selection clause 5 then excludes the member in
    every mode, and no requeue grant can release it — ``blocked_stage`` is the
    :data:`SELECTION_EXCLUSION_STAGE` sentinel outside the grant-stage
    vocabulary. Release is an explicit operator void, or the head lawfully
    moving through an operator lane (which consumes the row exactly like a
    granted supersession consumes a grant-pending park).

    Parameters
    ----------
    stable_id, name, campaign_id, machine_id, created_at:
        Member identity and operator provenance.
    held_revision:
        Exact review-held ``record_revision`` currently governing the member,
        derived from the LIVE models ledger by the caller — never typed by
        hand.
    held_status_code:
        The held head's recorded status code, for the status surfacing.
    decision:
        The frozen decision this exclusion executes (e.g.
        ``PD-7-slop-family-routes``).
    reason:
        Non-empty operator rationale.

    Returns
    -------
    dict[str, Any]
        Content-addressed park row.
    """

    if not decision.strip():
        raise GrantParkError("a selection-exclusion park must name the decision it executes")
    if not reason.strip():
        raise GrantParkError("a selection-exclusion park requires a non-empty reason")
    recheck_hint = (
        f"model {stable_id!r} is EXCLUDED from selection by frozen decision "
        f"{decision!r} while its review-held terminal {held_revision!r} governs. "
        "No requeue grant releases this park (its stage sentinel is outside the "
        "grant vocabulary); an operator releases it only by voiding this row "
        f"(python -m menagerie.crawler.tools.grant_parks --void <row_sha256> "
        "--reason '<why>') after the decision itself is revisited."
    )
    row: JsonObject = {
        "schema_version": GRANT_PARK_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": f"selection-exclusion:{decision}:{stable_id}",
        "name": name,
        "campaign_id": campaign_id,
        "run_id": f"operator:{decision}",
        "machine_id": machine_id,
        "created_at": created_at,
        "disposition": SELECTION_EXCLUSION_DISPOSITION,
        "pending": {
            "blocked_parent_revision": held_revision,
            "blocked_stage": SELECTION_EXCLUSION_STAGE,
            "pending_status_code": held_status_code,
            "exclusion_decision": decision,
            "exclusion_reason": reason,
        },
        "recheck_hint": recheck_hint,
    }
    row = stamp_row_digest(row)
    return validate_grant_park_row(row)


def validate_grant_park_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one grant-pending park row and return its canonical copy.

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
    GrantParkError
        If a required field is missing, mistyped, inconsistent with the pending
        revision it embeds, or the digest disagrees.
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
        "pending",
        "recheck_hint",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise GrantParkError(f"grant-pending park row is missing {', '.join(sorted(missing))}")
    if row["schema_version"] != GRANT_PARK_SCHEMA_VERSION:
        raise GrantParkError(
            f"grant-pending park row declares unknown schema {row['schema_version']!r}"
        )
    if row["disposition"] not in {
        GRANT_PENDING_DISPOSITION,
        AWARD_PARK_DISPOSITION,
        SELECTION_EXCLUSION_DISPOSITION,
    }:
        raise GrantParkError(
            f"grant-pending park row declares unknown disposition {row['disposition']!r}"
        )
    for field in ("stable_id", "work_id", "name", "campaign_id", "run_id", "machine_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise GrantParkError(f"grant-pending park row has an empty {field}")
    pending = row["pending"]
    if not isinstance(pending, Mapping):
        raise GrantParkError("grant-pending park row must record its pending append")
    for field in ("blocked_parent_revision", "blocked_stage"):
        if not isinstance(pending.get(field), str) or not pending.get(field):
            raise GrantParkError(f"grant-pending park row must name {field}")
    if row["disposition"] == SELECTION_EXCLUSION_DISPOSITION:
        # An exclusion embeds NO pending append: it parks the member as-is.
        # Embedding one would smuggle a publishable candidate into a row whose
        # whole purpose is that nothing may publish.
        if pending.get("pending_model") is not None:
            raise GrantParkError(
                "a selection-exclusion park row may not embed a pending model revision"
            )
        if pending.get("blocked_stage") != SELECTION_EXCLUSION_STAGE:
            raise GrantParkError(
                "a selection-exclusion park row must record the "
                f"{SELECTION_EXCLUSION_STAGE!r} stage sentinel (no grant may ever "
                "stage-match it)"
            )
        for field in ("pending_status_code", "exclusion_decision", "exclusion_reason"):
            if not isinstance(pending.get(field), str) or not pending.get(field):
                raise GrantParkError(f"a selection-exclusion park row must name {field}")
    else:
        pending_model = pending.get("pending_model")
        if not isinstance(pending_model, Mapping):
            raise GrantParkError("grant-pending park row must embed the pending model revision")
        if str(pending_model.get("stable_id")) != str(row["stable_id"]):
            raise GrantParkError(
                "grant-pending park row embeds a pending revision for another stable_id"
            )
        pending_status = pending_model.get("status")
        if not isinstance(pending_status, Mapping) or not isinstance(
            pending_status.get("code"), str
        ):
            raise GrantParkError(
                "grant-pending park row's pending revision names no terminal status"
            )
        embedded_parent = pending_model.get("parent_revision")
        if embedded_parent is not None and embedded_parent != pending["blocked_parent_revision"]:
            raise GrantParkError(
                "grant-pending park row's pending revision binds a different parent"
            )
        if pending.get("pending_status_code") != pending_status.get("code"):
            raise GrantParkError(
                "grant-pending park row's status summary disagrees with its pending revision"
            )
    if not isinstance(row["recheck_hint"], str) or not row["recheck_hint"].strip():
        raise GrantParkError("grant-pending park row must name what releases it later")
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise GrantParkError(
            f"grant park row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise GrantParkError("grant-pending park row digest does not match its payload")
    validated["row_sha256"] = digest
    return validated


def build_grant_park_void(
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
    GrantParkError
        If a field is empty.
    """

    row: JsonObject = {
        "schema_version": GRANT_PARK_VOID_SCHEMA_VERSION,
        "void_of": void_of,
        "reason": reason,
        "voided_by": voided_by,
        "created_at": created_at,
    }
    row = stamp_row_digest(row)
    return validate_grant_park_void(row)


def validate_grant_park_void(row: Mapping[str, Any]) -> JsonObject:
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
    GrantParkError
        If a required field is missing or empty, or the digest disagrees.
    """

    required = ("schema_version", "void_of", "reason", "voided_by", "created_at", "row_sha256")
    missing = [field for field in required if field not in row]
    if missing:
        raise GrantParkError(f"grant-pending park void is missing {', '.join(sorted(missing))}")
    if row["schema_version"] != GRANT_PARK_VOID_SCHEMA_VERSION:
        raise GrantParkError(
            f"grant-pending park void declares unknown schema {row['schema_version']!r}"
        )
    for field in ("void_of", "reason", "voided_by", "created_at"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise GrantParkError(f"grant-pending park void has an empty {field}")
    validated = dict(json.loads(canonical_json_bytes(row).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise GrantParkError(
            f"grant park row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise GrantParkError("grant-pending park void digest does not match its payload")
    validated["row_sha256"] = digest
    return validated


def validate_grant_park_ledger_row(row: Mapping[str, Any]) -> JsonObject:
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
    GrantParkError
        If the row matches neither schema.
    """

    if not isinstance(row, Mapping):
        raise GrantParkError("grant-pending park ledger row is not an object")
    if row.get("schema_version") == GRANT_PARK_VOID_SCHEMA_VERSION:
        return validate_grant_park_void(row)
    return validate_grant_park_row(row)


def _park_fact_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    """Return the blocked-supersession FACT identity one park row records.

    Parameters
    ----------
    row:
        Validated park row.

    Returns
    -------
    tuple[str, str, str, str, str]
        Stable ID, work generation, review-held parent revision, blocked
        stage, and the exact pending-model digest.
    """

    return (
        str(row.get("stable_id")),
        str(row.get("work_id")),
        str(row.get("pending", {}).get("blocked_parent_revision")),
        str(row.get("pending", {}).get("blocked_stage")),
        str(row.get("pending", {}).get("pending_model_sha256")),
    )


def _split_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonObject], set[str]]:
    """Split validated ledger rows into park rows and voided digests."""

    parks: list[JsonObject] = []
    voided: set[str] = set()
    for row in rows:
        if row.get("schema_version") == GRANT_PARK_VOID_SCHEMA_VERSION:
            voided.add(str(row["void_of"]))
        else:
            parks.append(dict(row))
    return parks, voided


def append_grant_park_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one grant-pending park durably and idempotently.

    Parameters
    ----------
    path:
        Destination grant-pending park JSONL.
    row:
        Complete typed park row.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    GrantParkError
        If the row is malformed.

    Notes
    -----
    Idempotency is keyed on the blocked-supersession FACT identity -- stable
    ID, work generation, AND the review-held parent revision -- among UNVOIDED
    rows only. A resume re-derives the SAME blocked append (``created_at``,
    ``run_id``, and pending-model timestamps necessarily differ) and is a no-op
    returning the first persisted row. A park an operator VOIDED never swallows
    a fresh fire: re-deriving the same blocked fact after a void appends a new
    row, exactly as the operator's decision demanded a fresh derivation.
    """

    validated = validate_grant_park_row(row)
    key = _park_fact_key(validated)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_grant_park_lines(_ledger_bytes(path, handle), path)
        parks, voided = _split_rows(existing)
        generation = [
            item
            for item in parks
            if (str(item["stable_id"]), str(item["work_id"])) == (key[0], key[1])
        ]
        # Idempotency consults every LIVE row of the generation (RULING W7
        # clause 9: per-row liveness), newest matching fact returned; a fact
        # whose every prior row was voided appends fresh.
        for item in reversed(generation):
            if str(item["row_sha256"]) not in voided and _park_fact_key(item) == key:
                return item
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def append_grant_park_void(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one operator park void durably and idempotently.

    Parameters
    ----------
    path:
        Grant-pending park JSONL holding the row being voided.
    row:
        Complete typed void row.

    Returns
    -------
    dict[str, Any]
        Validated persisted void row.

    Raises
    ------
    GrantParkError
        If the void is malformed or references no persisted park row.
    """

    validated = validate_grant_park_void(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_grant_park_lines(_ledger_bytes(path, handle), path)
        parks, _voided = _split_rows(existing)
        target = str(validated["void_of"])
        if not any(str(item["row_sha256"]) == target for item in parks):
            raise GrantParkError(f"grant-pending park void references an unknown row: {target}")
        prior = [
            item
            for item in existing
            if item.get("schema_version") == GRANT_PARK_VOID_SCHEMA_VERSION
            and str(item["void_of"]) == target
        ]
        if prior:
            if str(prior[0]["row_sha256"]) == str(validated["row_sha256"]):
                # Byte-identical re-void: idempotent.
                return dict(prior[0])
            # RULING W7 clause 9 (closing O C-6's silent no-op): a release
            # must name a CURRENTLY-LIVE row. A second, different void of an
            # already-released park would let its caller believe a fresh
            # operator decision landed when the prior release stands.
            raise GrantParkError(
                f"grant-pending park row {target} was already voided; the "
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
        prefix_check=lambda retained: _decode_grant_park_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def _decode_grant_park_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode and validate every row in one park ledger's bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GrantParkError(
                f"grant-pending park ledger {path} has an unreadable row at line {index}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise GrantParkError(
                f"grant-pending park ledger {path} has a non-object row at line {index}"
            )
        rows.append(validate_grant_park_ledger_row(decoded))
    return tuple(rows)


def current_grant_park_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[JsonObject, ...]:
    """Return EVERY unvoided park row -- per-row liveness (RULING W7 clause 9).

    Withholding liveness is derived per row, never per (stable_id, work_id)-
    latest: under the old latest-per-key map, lawfully voiding the LATER of
    two unvoided parks for one work generation silently erased the EARLIER
    unvoided row too (the O A3 defect, verified by execution). Every unvoided
    park is live; a void removes exactly the row it names.

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


def existing_grant_park_release(path: Path, row_sha256: str) -> Optional[JsonObject]:
    """Return the release that already voided one park row, if any.

    The D-48 machine release lane re-derives releases from released-ness on
    every close, so a re-close consults the standing release instead of
    appending a second one -- the append door itself REFUSES a different
    release of an already-released row (RULING W7 clause 9, closing O C-6).
    """

    if not path.is_file():
        return None
    for item in _decode_grant_park_lines(_ledger_bytes(path), path):
        if (
            item.get("schema_version") == GRANT_PARK_VOID_SCHEMA_VERSION
            and str(item.get("void_of")) == row_sha256
        ):
            return dict(item)
    return None


def load_grant_park_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load the CURRENT unvoided grant-pending park per work generation.

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
        rows.extend(_decode_grant_park_lines(_ledger_bytes(path), path))
    return current_grant_park_rows(rows)


def grant_park_released(
    row: Mapping[str, Any], current_record: Optional[Mapping[str, Any]]
) -> bool:
    """Return whether a landed revision releases (consumes) one park row.

    A park row is released exactly when the model's revision moved past the
    review-held parent the row recorded: the granted supersession landed. The
    same review-held revision still governing means the park is still LIVE --
    a pending supersession, never a stale row -- and an absent revision
    releases nothing.

    The record every CALLER must supply is the model's RAW LEDGER HEAD, not a
    dependency-currency projection of it (BD-1). Release is a fact about what
    landed; a head the live authority cannot vouch for is still a landed
    revision. Callers that asked the projection made a stale head unable to
    ever release its park, and the step-100 close reported 16 of 17
    terminal-refusal rows LIVE for models that had terminalized hours earlier.

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
    revision = current_record.get("record_revision")
    blocked = row.get("pending", {}).get("blocked_parent_revision")
    return isinstance(revision, str) and revision != blocked
