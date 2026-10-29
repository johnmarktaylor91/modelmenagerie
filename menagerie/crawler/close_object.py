"""The immutable leg-close attestation, and the staging gate that consumes it.

T1-sol-02, CONFIRMED by all three S2 labs: at the operator halt the newest
driver JSON was ``{"status": "interrupted:shutdown", "models_reduced": 12}`` --
an INTERRUPTION record, not a close -- while the same books answered
``status --verify-partition --require-valid-partition`` with exit 1. There was
no machine-readable statement of whether step-100 had closed, so every
downstream decision (audit dispatch, restoration, step-200 staging) was made
from hand-reconciled spreadsheets of raw heads.

A close object fixes that by being the ONE durable, content-addressed answer:

- **intake identity** -- which snapshot closed, by id and digest;
- **ledger tail identities** -- the exact bytes the close observed, per
  canonical ledger (row count, tail record digest, whole-ledger digest);
- **partition arms** -- coverage, extras, duplicates, per-code buckets,
  deferral/park accounting;
- **dependency-current census** -- the projection's own verdict, including the
  per-axis stale bucket and ``awards_stale_on_live_basis``;
- **verifier digest** -- the authority identities that adjudicated it, so a
  reader can tell whether the world has rotated since.

Staging the NEXT rung consumes it (:func:`require_closed_lineage`) and
re-verifies the recorded ledger digests against the ledgers as they are NOW.
Interruption JSON can never satisfy the gate, and neither can a close object
whose ledgers have moved since: the attestation describes a state, and staging
from a different state must re-attest it.

The object is EVIDENCE, never authority: nothing here can bless a partition.
``partition_valid`` is recomputed by the producer's own tripwire
(:func:`~menagerie.crawler.status.assert_partition`) and merely recorded, and a
close object that records an invalid or stale partition refuses staging exactly
as a missing one does.

Because the arms are a PRODUCER STATEMENT, they are never the decision. The
staging gate (:func:`require_closed_lineage`) is handed the lineage's real
identity (campaign, registered snapshot digest), its real membership, and the
members the models ledger actually recorded a head for, and it recomputes the
licensing question against those facts (2026-08-16 red-team round 2):

- a close that self-describes as another campaign's, or over a digest that is
  not the registered rung's, licenses nothing -- matching an id STRING is not
  matching a rung;
- the accounting arms must be exactly the lineage's members, each in exactly
  one arm -- cardinality agreement admitted any two strings; and
- a leg that recorded no head for a member closed nothing for that member, so
  the attested census cannot substitute for the ledger.

The attested arms remain the DISCLOSURE; the decision is replay-derived.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.close_reads import (
    DECODED_ROW_WEIGHT_FACTOR,
    PROJECTION_BASE_WEIGHT,
    LedgerRow,
    content_identity,
    derive,
    read_epoch,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    hash_bytes,
    row_content_address,
    stable_hash,
    stamp_row_digest,
)
from menagerie.crawler.leg_generation import CLOSE_SOURCE_NAMES, census_identities
from menagerie.crawler.models import JsonObject, LedgerPaths

#: The pre-generation close schema. Rows stay readable HISTORY; they license
#: nothing (2026-08-16 tri-lab contract): a v1 close predates the leg window,
#: the source census, and the sealed partition proof, so the gate refuses it
#: with a remedy pointing at ``tools/backfill_close``.
CLOSE_OBJECT_SCHEMA_VERSION_V1 = "menagerie.crawler.leg-close.v1"

CLOSE_OBJECT_SCHEMA_VERSION = "menagerie.crawler.leg-close.v2"

#: The two close kinds. ``leg`` is a live driver leg proven against its own
#: generation window; ``backfill`` is the migration mint for pre-machinery
#: legs, proven by the era-honest predicates. The kind is inside the content
#: address and the gate never applies one kind's predicates to the other.
CLOSE_KINDS = ("leg", "backfill")

#: Versioned preimage tag for the verifier digest, so a later change to WHAT
#: the digest covers can never be mistaken for a change in the world it covers.
CLOSE_VERIFIER_VERSION = "menagerie-leg-close-verifier-v1"

CLOSE_OBJECT_RELATIVE_PATH = ("intake", "leg-closes.jsonl")

#: Canonical ledger names the close object binds, in fixed order.
CLOSE_LEDGER_NAMES = ("models", "attempts", "gates", "artifacts")

#: The EXACT closed grammar of close statuses (D-24, RULING W7 clause 6 /
#: N11). Every spelling is a ``_completion_close_status`` output: the two
#: ``complete`` forms, plus ``terminal-partition-complete`` with its optional
#: suffixes composed in the FIXED order deferrals -> review-holds ->
#: refusal-parks. Nothing else is a close -- no prefix match, no default arm.
#: If ANY future amendment extends a status spelling, this enumeration must
#: change in the SAME reviewed change (the ruling's stated prerequisite).
CLOSE_STATUS_SPELLINGS = frozenset(
    {
        "complete",
        "complete-with-deferrals",
        "terminal-partition-complete",
        "terminal-partition-complete-with-deferrals",
        "terminal-partition-complete-with-review-holds",
        "terminal-partition-complete-with-refusal-parks",
        "terminal-partition-complete-with-deferrals-with-review-holds",
        "terminal-partition-complete-with-deferrals-with-refusal-parks",
        "terminal-partition-complete-with-review-holds-with-refusal-parks",
        "terminal-partition-complete-with-deferrals-with-review-holds-with-refusal-parks",
    }
)

#: Partition arms every close object must CARRY (2026-08-16 red-team break 5:
#: the arms only had to BE mappings, so ``partition_arms={}`` had no arm to
#: trip and a vacuous attestation licensed a rung). Sequence-valued arms first,
#: then the accounting mappings; ``intake`` is the coverage count and ``valid``
#: the producer tripwire's recorded verdict.
_PARTITION_SEQUENCE_ARMS = ("missing", "extra", "duplicates", "invalid")
#: DX-22: the park arms are DISCLOSURE, never accounting. A live park
#: legitimately coexists with its member's coverage -- the member's record
#: sits in ``buckets`` (or its recordless claim in ``machine_deferred``) while
#: the park stays live awaiting an operator decision -- so counting these arms
#: into the accounting multiset double-counted every parked member (the
#: 212-of-210 overlap that made the truthful 200a close unstageable, R2).
#: They stay REQUIRED arms of the close, and every id they name must be a
#: lineage member; they simply account for nothing.
_PARTITION_DISCLOSURE_SEQUENCE_ARMS = ("grant_pending", "refusal_parked")
_PARTITION_MAPPING_ARMS = ("buckets", "machine_deferred")
_CURRENCY_MAPPING_ARMS = ("stale_by_axis",)
_CURRENCY_SEQUENCE_ARMS = ("award_runs_to_non_runs", "award_runs_to_runs")


class CloseObjectError(ValueError):
    """Raised when a close object is malformed, or absent where required."""


def close_object_path(records_root: Path) -> Path:
    """Return the append-only leg-close ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        ``records/intake/leg-closes.jsonl``.
    """

    return records_root.joinpath(*CLOSE_OBJECT_RELATIVE_PATH)


def is_close_status(status: str) -> bool:
    """Return whether one completion status is a CLOSE rather than a stop.

    D-24 (RULING W7 / N11): membership in the EXACT closed grammar, never a
    prefix match. The old ``startswith`` arm accepted any unenumerated
    extension -- ``complete-with-continuations-typo``, suffixes composed in
    the wrong order -- as a licensing close; only
    :data:`CLOSE_STATUS_SPELLINGS` cores are closes, bare or in the exact
    ``phase-<core>:<phase>`` form with a non-empty phase.

    Parameters
    ----------
    status:
        Driver completion status.

    Returns
    -------
    bool
        True for an enumerated close spelling, phase-scoped or not.
    """

    text = str(status)
    if text in CLOSE_STATUS_SPELLINGS:
        return True
    if not text.startswith("phase-"):
        return False
    core, separator, phase = text[len("phase-") :].partition(":")
    return bool(separator) and bool(phase) and core in CLOSE_STATUS_SPELLINGS


def ledger_identity(path: Path) -> JsonObject:
    """Return the exact observed identity of one canonical ledger.

    Parameters
    ----------
    path:
        Canonical ledger JSONL path.

    Returns
    -------
    dict[str, Any]
        ``rows``, ``ledger_sha256`` over the whole file, and ``tail_sha256``
        over the last line. A ledger that does not exist is an honest empty
        identity rather than an omission, so an absent ledger cannot silently
        read as "unchanged".
    """

    if not path.is_file():
        return {"rows": 0, "ledger_sha256": hash_bytes(b""), "tail_sha256": None}
    # Streamed, never materialized (round-2 T4): the digest still covers every
    # byte AS THEY ARE, with no torn-tail recovery, because this v1 disclosure
    # field has always hashed the raw file.
    identity = content_identity(path, projection="close.ledger-identity")
    return {
        "rows": identity.rows,
        "ledger_sha256": identity.ledger_sha256,
        "tail_sha256": identity.tail_sha256,
    }


def ledger_identities(ledgers: LedgerPaths) -> JsonObject:
    """Return every canonical ledger's observed identity.

    Parameters
    ----------
    ledgers:
        Canonical ledger paths.

    Returns
    -------
    dict[str, Any]
        Ledger name to :func:`ledger_identity`.
    """

    return {name: ledger_identity(getattr(ledgers, name)) for name in CLOSE_LEDGER_NAMES}


def verifier_identity(authority_identities: Mapping[str, str]) -> str:
    """Digest the authority identities that adjudicated one close.

    Parameters
    ----------
    authority_identities:
        Rotation-sensitive identity names to values.

    Returns
    -------
    str
        Versioned verifier digest.
    """

    return stable_hash(
        {
            "verifier_version": CLOSE_VERIFIER_VERSION,
            "authority_identities": dict(sorted(authority_identities.items())),
        }
    )


def build_close_object(
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str],
    close_kind: str,
    generation_id: str,
    leg_open_row_sha256: Optional[str],
    sources: Mapping[str, Any],
    partition_proof_sha256: str,
    predecessor_close_sha256: Optional[str],
    run_id: str,
    machine_id: str,
    closed_at: str,
    completion_status: str,
    partition_arms: Mapping[str, Any],
    currency_census: Mapping[str, Any],
    authority_identities: Mapping[str, str],
    operator_attributions: Optional[Mapping[str, str]] = None,
    leg_purpose: Optional[str] = None,
) -> JsonObject:
    """Assemble one content-addressed leg-close attestation (v2).

    The BINDING is the generation-scoped sealed proof: ``(campaign_id,
    intake_snapshot_sha256, generation_id, partition_proof_sha256)`` plus the
    leg-open row and the full source census. The partition arms and currency
    census ride along as DISCLOSURE and may never improve the decision.

    Parameters
    ----------
    campaign_id, intake_snapshot_id, intake_snapshot_sha256, phase:
        Exactly what closed.
    close_kind:
        ``leg`` or ``backfill``.
    generation_id:
        The leg generation this close seals.
    leg_open_row_sha256:
        Content address of the durable ``leg_open`` row (required for kind
        ``leg``; ``None`` for a backfill, whose era had no open).
    sources:
        The FULL source census identities observed at close
        (:func:`menagerie.crawler.leg_generation.census_identities`).
    partition_proof_sha256:
        Content address of the sealed, resident ``PartitionProofV2``.
    run_id, machine_id, closed_at:
        Producer provenance.
    completion_status:
        Driver completion status; must belong to the close family.
    partition_arms:
        Coverage arms the producer's own tripwire computed (disclosure).
    currency_census:
        Dependency-currency census including the per-axis stale bucket.
    authority_identities:
        Rotation-sensitive authority identities in force at close.
    operator_attributions:
        Backfill-only operator authorities, disclosed and content-addressed
        with the close so the gate can recompute the proof they steered.
    leg_purpose:
        The closing generation's declared purpose
        (:mod:`menagerie.crawler.leg_purpose`), repeated verbatim from its
        leg-open row so the leg's own replay is checkable, or ``None`` for an
        ordinary leg (versioned-monotone: rows minted before the vocabulary
        carry no field).

    Returns
    -------
    dict[str, Any]
        Content-addressed close object.

    Raises
    ------
    CloseObjectError
        If provenance is empty, the status is not a close, or the binding
        fields are malformed.
    """

    if not is_close_status(completion_status):
        raise CloseObjectError(
            f"{completion_status!r} is not a close: an interruption, pause, or "
            "failure state never attests a boundary, and no close object may "
            "exist for one"
        )
    body: JsonObject = {
        "schema_version": CLOSE_OBJECT_SCHEMA_VERSION,
        "close_kind": close_kind,
        "campaign_id": campaign_id,
        "intake_snapshot_id": intake_snapshot_id,
        "intake_snapshot_sha256": intake_snapshot_sha256,
        "phase": phase,
        "generation_id": generation_id,
        "leg_open_row_sha256": leg_open_row_sha256,
        "run_id": run_id,
        "machine_id": machine_id,
        "closed_at": closed_at,
        "completion_status": completion_status,
        "sources": {name: dict(identity) for name, identity in sources.items()},
        "partition_proof_sha256": partition_proof_sha256,
        "predecessor_close_sha256": predecessor_close_sha256,
        "partition": dict(partition_arms),
        "currency": dict(currency_census),
        "verifier_identity": verifier_identity(authority_identities),
        "authority_identities": dict(sorted(authority_identities.items())),
    }
    if operator_attributions:
        body["operator_attributions"] = dict(sorted(operator_attributions.items()))
    if leg_purpose is not None:
        body["leg_purpose"] = leg_purpose
    return validate_close_object(stamp_row_digest(body, field="close_sha256"))


def validate_close_object(row: Mapping[str, Any]) -> JsonObject:
    """Validate one close object and return its canonical copy.

    Parameters
    ----------
    row:
        Candidate close object.

    Returns
    -------
    dict[str, Any]
        Validated row.

    Raises
    ------
    CloseObjectError
        If a required field is missing/empty, the status is not a close, or the
        content address disagrees with the payload.
    """

    if not isinstance(row, Mapping):
        raise CloseObjectError("close object is not an object")
    schema = row.get("schema_version")
    if schema not in (CLOSE_OBJECT_SCHEMA_VERSION, CLOSE_OBJECT_SCHEMA_VERSION_V1):
        raise CloseObjectError(f"close object declares unknown schema {schema!r}")
    required: tuple[str, ...] = (
        "campaign_id",
        "intake_snapshot_id",
        "intake_snapshot_sha256",
        "run_id",
        "machine_id",
        "closed_at",
        "completion_status",
        "verifier_identity",
        "close_sha256",
    )
    if schema == CLOSE_OBJECT_SCHEMA_VERSION:
        required = (
            *required,
            "close_kind",
            "generation_id",
            "partition_proof_sha256",
            "predecessor_close_sha256",
        )
    missing = [field for field in required if field not in row]
    if missing:
        raise CloseObjectError(f"close object is missing {', '.join(sorted(missing))}")
    for field in required:
        if field == "predecessor_close_sha256":
            continue
        if not isinstance(row[field], str) or not row[field].strip():
            raise CloseObjectError(f"close object has an empty {field}")
    if "phase" not in row or not (row["phase"] is None or isinstance(row["phase"], str)):
        raise CloseObjectError("close object phase must be a string or null")
    for field in ("partition", "currency", "authority_identities"):
        if not isinstance(row.get(field), Mapping):
            raise CloseObjectError(f"close object {field} must be an object")
    if schema == CLOSE_OBJECT_SCHEMA_VERSION_V1:
        if not isinstance(row.get("ledgers"), Mapping):
            raise CloseObjectError("close object ledgers must be an object")
        for name in CLOSE_LEDGER_NAMES:
            identity = row["ledgers"].get(name)
            if not isinstance(identity, Mapping) or not isinstance(
                identity.get("ledger_sha256"), str
            ):
                raise CloseObjectError(f"close object names no {name} ledger identity")
        # v1 rows minted after machinery-era closes exist chain through the
        # durable same-campaign tip (A-1); a link-free historical v1 row stays
        # valid, but a present predecessor must be a well-typed one.
        predecessor = row.get("predecessor_close_sha256")
        if not (predecessor is None or isinstance(predecessor, str)):
            raise CloseObjectError(
                "close object predecessor_close_sha256 must be a string or null"
            )
    else:
        if row["close_kind"] not in CLOSE_KINDS:
            raise CloseObjectError(
                f"close object declares unknown close kind {row['close_kind']!r}"
            )
        open_digest = row.get("leg_open_row_sha256")
        if row["close_kind"] == "leg":
            if not isinstance(open_digest, str) or not open_digest.strip():
                raise CloseObjectError("a leg close must bind its durable leg-open row")
        elif not (open_digest is None or isinstance(open_digest, str)):
            raise CloseObjectError("close object leg_open_row_sha256 must be a string or null")
        sources = row.get("sources")
        if not isinstance(sources, Mapping) or set(sources) != set(CLOSE_SOURCE_NAMES):
            raise CloseObjectError(
                "close object sources must attest the closed source universe "
                "exactly, every slot present-or-absent"
            )
        for name, identity in sources.items():
            if not isinstance(identity, Mapping) or not isinstance(
                identity.get("ledger_sha256"), str
            ):
                raise CloseObjectError(f"close object names no {name} source identity")
            if not isinstance(identity.get("chain_head"), str):
                raise CloseObjectError(
                    f"close object names no Track A chain head for {name}"
                )
        attributions = row.get("operator_attributions")
        if attributions is not None and not isinstance(attributions, Mapping):
            raise CloseObjectError("close object operator_attributions must be an object")
        predecessor = row.get("predecessor_close_sha256")
        if predecessor is not None and not isinstance(predecessor, str):
            raise CloseObjectError("close object predecessor_close_sha256 must be a string or null")
        # Versioned-monotone declared purpose (shape 4): absent on ordinary
        # legs and every pre-vocabulary row; a present value must belong to
        # the closed vocabulary and may only ride a leg-kind close (the
        # backfill mint predates the purpose era by construction).
        if "leg_purpose" in row:
            from menagerie.crawler.leg_purpose import LEG_PURPOSES

            if row["leg_purpose"] not in LEG_PURPOSES:
                raise CloseObjectError(
                    f"close object declares unknown leg purpose {row['leg_purpose']!r}"
                )
            if row["close_kind"] != "leg":
                raise CloseObjectError(
                    "close object leg_purpose may only ride a leg-kind close"
                )
    # The arms must be PRESENT, not merely mapping-typed: an attestation that
    # attests nothing is not a close (red-team break 5). Absence refuses on the
    # same principle ledger_identity applies -- an absent census can never
    # silently read as "nothing to report".
    partition = row["partition"]
    if not isinstance(partition.get("valid"), bool):
        raise CloseObjectError("close object partition records no valid verdict")
    if not isinstance(partition.get("intake"), int):
        raise CloseObjectError("close object partition records no intake coverage count")
    for arm in (*_PARTITION_SEQUENCE_ARMS, *_PARTITION_DISCLOSURE_SEQUENCE_ARMS):
        if not isinstance(partition.get(arm), Sequence) or isinstance(
            partition.get(arm), (str, bytes)
        ):
            raise CloseObjectError(f"close object partition carries no {arm} arm")
    for arm in _PARTITION_MAPPING_ARMS:
        if not isinstance(partition.get(arm), Mapping):
            raise CloseObjectError(f"close object partition carries no {arm} accounting")
    for arm in _CURRENCY_MAPPING_ARMS:
        if not isinstance(row["currency"].get(arm), Mapping):
            raise CloseObjectError(f"close object currency carries no {arm} census")
    if schema == CLOSE_OBJECT_SCHEMA_VERSION:
        for arm in _CURRENCY_SEQUENCE_ARMS:
            value = row["currency"].get(arm)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise CloseObjectError(f"close object currency carries no {arm} census")
    if not is_close_status(str(row["completion_status"])):
        raise CloseObjectError(
            f"close object records {row['completion_status']!r}, which is not a close"
        )
    validated = dict(json.loads(canonical_json_bytes(dict(row)).decode("utf-8")))
    foreign = foreign_content_address_fields(validated, own="close_sha256")
    if foreign:
        raise CloseObjectError(
            f"close object carries a foreign content-address field "
            f"{', '.join(foreign)}: a close owns exactly one content address"
        )
    chain = validated.pop("chain", None)
    digest = validated.pop("close_sha256")
    if row_content_address(validated) != digest:
        raise CloseObjectError("close object digest does not match its payload")
    if verifier_identity(validated["authority_identities"]) != validated["verifier_identity"]:
        raise CloseObjectError("close object verifier digest contradicts its own identities")
    validated["close_sha256"] = digest
    if chain is not None:
        validated["chain"] = chain
    return validated


def _ledger_bytes(path: Path, handle: Optional[BinaryIO] = None) -> bytes:
    """Read the whole close ledger, recovering only a torn final line first.

    The close ledger is the licensing authority, so a crash mid-append used to
    brick both closing and staging permanently: the strict whole-file decode
    below refused the torn fragment on every read and every append, forever.
    Recovery is recordio's evidence-then-truncate with THIS module's own strict
    decoder as the prefix check, so a COMPLETE malformed line stays hard,
    typed corruption preserved byte-for-byte. A caller inside
    :func:`append_close_object` already holds the ledger's ``flock`` and passes
    its handle through, so recovery never contends with its own writer.
    """

    from menagerie.crawler.recordio import read_with_tail_recovery

    position = None if handle is None else handle.tell()
    payload = read_with_tail_recovery(
        path,
        prefix_check=lambda retained: _decode_close_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )
    if handle is not None and position is not None:
        handle.seek(position)
    return payload


def _decode_close_lines(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Validate every line of one close ledger.

    Parameters
    ----------
    payload, path:
        Ledger bytes and their path, for diagnostics.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Validated close objects in ledger order.

    Raises
    ------
    CloseObjectError
        If any line is not a valid close object.
    """

    rows: list[JsonObject] = []
    for index, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        rows.append(_decode_close_row(path, index, line))
    from menagerie.crawler.ledger_chain import LedgerChainError, verify_rows

    try:
        # D-27 require_links hardening (O B-9): a close ledger whose FIRST row
        # is linked has no historical link-free prefix, so every row must
        # carry its link -- replay-derived and era-honest (a genuinely
        # unlinked historical prefix keeps verifying exactly as before).
        verify_rows(
            rows,
            scope="meta.closes",
            require_links=bool(rows and rows[0].get("chain") is not None),
        )
    except LedgerChainError as exc:
        raise CloseObjectError(str(exc)) from exc
    return tuple(rows)


def _decode_close_row(path: Path, line_number: int, line: bytes) -> JsonObject:
    """Decode and validate one close-ledger row fail-closed.

    The single-row half of :func:`_decode_close_lines`, so the whole-bytes
    decoder and the incremental derivation cannot drift in what they accept or
    how they name a refusal.
    """

    try:
        decoded = json.loads(line)
    except ValueError as exc:
        raise CloseObjectError(
            f"close ledger {path} has a malformed row at line {line_number}: {exc}"
        ) from exc
    return validate_close_object(decoded)


@dataclass(frozen=True)
class _CloseLedger:
    """One held close-ledger derivation: validated rows plus the chain extent.

    Parameters
    ----------
    rows:
        Validated close objects in ledger order.
    head_sha256:
        Verified ``meta.closes`` chain head after those rows.
    linked:
        Whether a persisted chain link has been seen (the link-free historical
        prefix closes for good at the first one).
    """

    rows: tuple[JsonObject, ...]
    head_sha256: str
    linked: bool


def close_ledger_rows(
    path: Path, *, locked_handle: Optional[BinaryIO] = None
) -> tuple[JsonObject, ...]:
    """Return one close ledger's validated rows, derived incrementally.

    Round-2 T4 (scale): the close ledger is consulted a dozen times by ONE
    licensing decision -- the lookup, the predecessor spine, the carry
    authorities, the era predicates -- and each consultation used to re-decode,
    re-validate (a canonical re-serialization and two digests per row), and
    re-chain-verify every row ever written. Rows are now validated once per
    observed ledger state and the chain is extended over the appended suffix,
    so the cost of the close ledger is O(rows appended), not
    O(#closes) x O(#consultations). Every refusal is byte-identical to
    :func:`_decode_close_lines`.

    Parameters
    ----------
    path:
        Close ledger JSONL path. A missing file is an empty ledger.
    locked_handle:
        Open handle on ``path`` whose exclusive ``flock`` the caller holds.

    Raises
    ------
    CloseObjectError
        If any complete row is malformed, invalid, or breaks the chain.
    """

    from menagerie.crawler.ledger_chain import LedgerChainError, _genesis, _verified_link

    if locked_handle is None and not path.is_file():
        return ()

    def extend(value: _CloseLedger, rows: Iterable[LedgerRow]) -> _CloseLedger:
        # Validate every new row BEFORE extending the chain, exactly as the
        # whole-bytes decoder validates all rows before verifying the chain.
        decoded = tuple(
            _decode_close_row(path, number, line) for number, _offset, line in rows
        )
        if not decoded:
            return value
        previous = value.head_sha256
        linked = value.linked
        index = len(value.rows)
        try:
            for row in decoded:
                previous, linked = _verified_link(
                    row,
                    scope="meta.closes",
                    index=index,
                    previous=previous,
                    linked=linked,
                )
                index += 1
        except LedgerChainError as exc:
            raise CloseObjectError(str(exc)) from exc
        return _CloseLedger(
            rows=(*value.rows, *decoded), head_sha256=previous, linked=linked
        )

    def recover(target: Path) -> None:
        _ledger_bytes(target, locked_handle)

    ledger: _CloseLedger = derive(
        path,
        projection="close.rows",
        seed=lambda: _CloseLedger(
            rows=(), head_sha256=_genesis("meta.closes"), linked=False
        ),
        extend=extend,
        recover=recover,
        locked_handle=locked_handle,
        # Every validated close row, retained. O(#closes) by construction: the
        # close ledger never carries campaign history.
        weigh=lambda _value, byte_count: (
            PROJECTION_BASE_WEIGHT + DECODED_ROW_WEIGHT_FACTOR * byte_count
        ),
    )
    return ledger.rows


def append_close_object(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one close object durably and idempotently.

    Idempotency is on the content address: re-closing the identical observed
    state is a no-op, while a close over CHANGED ledgers is a new attestation
    and appends beside the old one (the ledger is append-only history, and the
    newest attestation for an intake governs).

    Parameters
    ----------
    path:
        Destination close JSONL.
    row:
        Complete close object.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    CloseObjectError
        If the object is malformed.
    """

    from menagerie.crawler.ledger_chain import link_row

    validated = validate_close_object(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = close_ledger_rows(path, locked_handle=handle)
        for candidate in existing:
            if candidate["close_sha256"] == validated["close_sha256"]:
                return dict(candidate)
        if validated.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION:
            # The proof-era door (RULING W7 clause 8, attack D): resident
            # history re-derives under its own sealed era forever, but a NEW
            # attestation may bind only the CURRENT proof era -- the era is
            # law, never a producer choice. A close binding an absent proof
            # refuses on the same fail-closed principle.
            from menagerie.crawler.partition_proof import (  # noqa: PLC0415 -- lazy: breaks the proof<->close import cycle
                PARTITION_PROOF_SCHEMA_VERSION,
                PartitionProofError,
                load_partition_proof,
            )

            records_root = path.parent.parent
            try:
                bound_proof = load_partition_proof(
                    records_root, str(validated["partition_proof_sha256"])
                )
            except PartitionProofError as exc:
                raise CloseObjectError(
                    f"a new close must bind a resident sealed proof: {exc}"
                ) from exc
            if bound_proof.get("schema_version") != PARTITION_PROOF_SCHEMA_VERSION:
                raise CloseObjectError(
                    "a new close may bind only the current proof era "
                    f"({PARTITION_PROOF_SCHEMA_VERSION}); this close binds "
                    f"{bound_proof.get('schema_version')!r} -- sealed history "
                    "keeps its own era at replay, but the era of a NEW "
                    "attestation is never producer-choosable (RULING W7 "
                    "clause 8)"
                )
        previous = next(
            (
                candidate
                for candidate in reversed(existing)
                if candidate.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION
                and candidate.get("campaign_id") == validated.get("campaign_id")
            ),
            None,
        )
        expected_predecessor = None if previous is None else previous["close_sha256"]
        if validated.get("predecessor_close_sha256") != expected_predecessor:
            raise CloseObjectError(
                f"close predecessor {validated.get('predecessor_close_sha256')!r} "
                f"does not equal the durable same-campaign tip {expected_predecessor!r}"
            )
        validated = validate_close_object(
            link_row(validated, scope="meta.closes", existing_rows=existing)
        )
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def load_close_objects(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load every persisted close object in ledger order.

    Parameters
    ----------
    paths:
        Candidate ledger paths. Missing files are empty ledgers.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Validated close objects.
    """

    rows: list[JsonObject] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.extend(close_ledger_rows(path))
    return tuple(rows)


def predecessor_close_sha256(records_root: Path, campaign_id: str) -> Optional[str]:
    """Return the durable same-campaign v2 close tip, or ``None`` at genesis."""

    matches = [
        row
        for row in load_close_objects([close_object_path(records_root)])
        if row.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION
        and str(row.get("campaign_id")) == campaign_id
    ]
    return None if not matches else str(matches[-1]["close_sha256"])


def find_close_object(
    rows: Iterable[Mapping[str, Any]],
    *,
    intake_snapshot_id: str,
    phase: Optional[str] = None,
) -> Optional[JsonObject]:
    """Return the newest close object for one intake snapshot.

    Parameters
    ----------
    rows:
        Validated close objects in ledger order.
    intake_snapshot_id:
        Snapshot whose close is sought.
    phase:
        Phase scope to match, or ``None`` for a whole-campaign close.

    Returns
    -------
    dict[str, Any] | None
        Newest matching close object, or ``None``.
    """

    matches = [
        dict(row)
        for row in rows
        if str(row.get("intake_snapshot_id")) == intake_snapshot_id and row.get("phase") == phase
    ]
    if not matches:
        return None
    leg_matches = [row for row in matches if row.get("close_kind") == "leg"]
    return leg_matches[-1] if leg_matches else matches[-1]


def close_object_disagreements(close: Mapping[str, Any], records_root: Path) -> tuple[str, ...]:
    """Return every way one close disagrees with the source census as it is.

    RD-1's re-observation arm: the census is re-derived through the SAME
    resolver the close attested through, slot for slot, presence included. A
    slot that appeared, disappeared, grew, or was rewritten since the close is
    a refusal -- which is what kills the shadow-layout break: a
    ``models.jsonl`` materialized after the close is a presence delta on an
    attested slot, never an unhashed free input.

    Parameters
    ----------
    close:
        Validated v2 close object.
    records_root:
        Campaign canonical records directory to re-observe.

    Returns
    -------
    tuple[str, ...]
        Human-readable disagreements; empty when the attested bytes still hold.
    """

    observed = census_identities(records_root)
    findings: list[str] = []
    attested = dict(close.get("sources") or {})
    for name in CLOSE_SOURCE_NAMES:
        recorded = dict(attested.get(name) or {})
        now = observed[name]
        if bool(recorded.get("present")) != bool(now["present"]):
            axis = (
                "present now but absent from the attestation"
                if now["present"]
                else "attested present but absent now"
            )
            findings.append(
                f"{name} is {axis}: a source the door reads must be one the "
                "close attests -- re-attest the close after reconciling layouts"
            )
            continue
        if recorded.get("ledger_sha256") != now["ledger_sha256"]:
            findings.append(
                f"{name} moved since the close: attested "
                f"{recorded.get('rows')} rows / {recorded.get('ledger_sha256')}, "
                f"observed {now['rows']} rows / {now['ledger_sha256']}"
            )
    return tuple(findings)


def close_object_refusals(close: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every reason one close object cannot license a next rung.

    The arms are recomputed from the RECORDED census, never from a producer
    boolean read at face value: a close whose partition was invalid, or whose
    projection could not vouch for a recorded head, has attested exactly that,
    and staging on top of it would inherit the unresolved state.

    Parameters
    ----------
    close:
        Validated close object.

    Returns
    -------
    tuple[str, ...]
        Human-readable refusals; empty for a clean close.
    """

    partition = dict(close.get("partition") or {})
    currency = dict(close.get("currency") or {})
    findings: list[str] = []
    for arm in ("missing", "extra", "duplicates", "invalid"):
        values = partition.get(arm) or []
        if values:
            findings.append(f"the close attests {arm}={sorted(str(v) for v in values)}")
    stale = dict(currency.get("stale_by_axis") or {})
    if stale:
        findings.append(
            "the close attests members whose recorded head the live projection "
            f"could not vouch for: { {axis: sorted(ids) for axis, ids in stale.items()} }"
        )
    # A close licenses a rung only if it attests a RESOLVED partition
    # (red-team break 5): the recorded verdict must be valid, the coverage
    # non-empty, and every intake member recomputably accounted by one of the
    # recorded arms -- a count the census cannot reproduce is a member nothing
    # accounts.
    if partition.get("valid") is not True:
        findings.append(
            f"the close records an unresolved partition verdict (valid={partition.get('valid')!r})"
        )
    findings.extend(close_accounting_refusals(close))
    return tuple(findings)


def close_accounting_refusals(close: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the accounting-INTEGRITY refusals alone (the DX-22 class).

    Exactly the multiset-discipline findings: empty coverage, and an
    accounted-member count the intake cannot reproduce. Distinct from the
    full licensing gate because a lying census is MACHINERY debt in any leg,
    while a stale attestation or an unresolved verdict is a lawful, disclosed
    mid-rung state -- SD-2's leg-start arm consumes this narrow class so a
    rebuild-deferral leg is never refused for doing its job.

    Parameters
    ----------
    close:
        Validated close object.

    Returns
    -------
    tuple[str, ...]
        Accounting-integrity refusals; empty when the census adds up.
    """

    partition = dict(close.get("partition") or {})
    findings: list[str] = []
    intake = partition.get("intake")
    if not isinstance(intake, int) or intake < 1:
        findings.append(
            f"the close attests empty coverage (intake={intake!r}): a close over "
            "no members licenses nothing"
        )
    else:
        accounted = accounted_members(close)
        if len(accounted) != intake:
            findings.append(
                f"the close accounts {len(accounted)} member(s) of an intake of "
                f"{intake}: every member must land in exactly one recorded "
                "accounting arm"
            )
    return tuple(findings)


def accounted_members(close: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every member id the close's accounting arms name, WITH repeats.

    Repeats are preserved on purpose: "exactly one recorded accounting arm" is
    a multiset claim, and a set union silently absorbs a member double-listed
    across two arms while a ghost fills the count (2026-08-16 red-team round 2,
    break 2b).

    The accounting arms are ``buckets`` and ``machine_deferred`` ONLY. The
    park arms (``grant_pending``/``refusal_parked``) are disclosure (DX-22): a
    live park coexists with its member's coverage by design, so counting them
    here double-counted every parked member -- the 212-of-210 overlap that
    refused the truthful 200a close at successor staging (R2).

    Parameters
    ----------
    close:
        Validated close object.

    Returns
    -------
    tuple[str, ...]
        Accounted member ids in arm order, duplicates included.
    """

    partition = dict(close.get("partition") or {})
    accounted: list[str] = []
    for bucket in dict(partition.get("buckets") or {}).values():
        accounted.extend(str(member) for member in bucket or ())
    for claimed in dict(partition.get("machine_deferred") or {}).values():
        accounted.extend(str(member) for member in claimed or ())
    # DX-22: the third accounting arm -- STALE-headed members a durable row
    # (valve or park) accounts for a withheld re-drive (BD-1). Optional for
    # era honesty: closes sealed before the arm existed simply have no such
    # members accounted, exactly as their bytes always said.
    for stale_claimed in dict(partition.get("stale_deferred") or {}).values():
        accounted.extend(str(member) for member in stale_claimed or ())
    return tuple(accounted)


def close_disclosure_refusals(
    close: Mapping[str, Any],
    *,
    records_root: Path,
    lineage_member_ids: Iterable[str],
    introduced_member_ids: Iterable[str],
) -> tuple[str, ...]:
    """Refuse a close whose DISCLOSURE arms lie about the membership.

    The sealed proof is the licensing authority; the arms are the census a
    human reads, and a divergence between them is a producer bug that must
    not license anything (round-2 break 2b's multiset discipline, retained):
    every lineage member lands in exactly one accounting arm, and no arm
    names a ghost. Disposition-negative -- this can only refuse.

    Parameters
    ----------
    close:
        Validated close object.
    lineage_member_ids:
        Stable ids the lineage snapshot's own bytes contain.

    Returns
    -------
    tuple[str, ...]
        Human-readable refusals; empty when the disclosure matches.
    """

    members = frozenset(str(member) for member in lineage_member_ids)
    findings: list[str] = []
    partition = dict(close.get("partition") or {})
    accounted = accounted_members(close)
    duplicated = sorted({member for member in accounted if accounted.count(member) > 1})
    if duplicated:
        findings.append(
            f"the close accounts {duplicated} in more than one arm: every member "
            "must land in exactly one recorded accounting arm"
        )
    unaccounted = sorted(members - set(accounted))
    if unaccounted:
        findings.append(
            f"the close accounts nothing for {len(unaccounted)} lineage member(s): "
            f"{unaccounted[:5]}"
        )
    foreign = sorted(set(accounted) - members)
    if foreign:
        findings.append(
            f"the close accounts {len(foreign)} id(s) the lineage snapshot does "
            f"not contain: {foreign[:5]}"
        )
    # DX-22: the park arms left the accounting multiset, but a census a human
    # reads is still ghost-checked -- a disclosure arm naming an id the lineage
    # snapshot does not contain is the same lie without the arithmetic.
    for arm in _PARTITION_DISCLOSURE_SEQUENCE_ARMS:
        ghosts = sorted(
            {str(member) for member in partition.get(arm) or ()} - members
        )
        if ghosts:
            findings.append(
                f"the close discloses {arm} ghost id(s) the lineage snapshot "
                f"does not contain: {ghosts[:5]}"
            )
    intake = partition.get("intake")
    if isinstance(intake, int) and intake != len(members):
        findings.append(
            f"the close attests an intake of {intake} over a lineage snapshot of "
            f"{len(members)} member(s)"
        )
    from menagerie.crawler.partition_proof import (
        PARTITION_PROOF_SCHEMA_VERSION_V3,
        adjudicated_here,
        attributed_here,
        carried,
        continuing_here,
        load_partition_proof,
        proved_here,
        withheld_here,
    )

    try:
        proof = load_partition_proof(records_root, str(close["partition_proof_sha256"]))
    except (KeyError, ValueError) as exc:
        findings.append(str(exc))
        return tuple(findings)
    recomputed = {
        "adjudicated_here": sorted(adjudicated_here(proof)),
        "withheld_here": sorted(withheld_here(proof)),
        "attributed_here": sorted(attributed_here(proof)),
        "carried": sorted(carried(proof)),
        "proved_here": sorted(proved_here(proof)),
        "introduced": sorted(str(value) for value in introduced_member_ids),
    }
    if proof.get("schema_version") != PARTITION_PROOF_SCHEMA_VERSION_V3:
        # RULING W7 clause 6: the continuation disclosure is first-class and
        # gate-recomputed. v3 proofs predate the disposition, so their closes
        # carry no such arm -- era-honest, never retro-required.
        recomputed["continuing"] = sorted(continuing_here(proof))
    for arm, expected in recomputed.items():
        disclosed = partition.get(arm)
        if disclosed != expected:
            findings.append(
                f"the close discloses {arm}={disclosed!r}, but gate recomputation "
                f"over the sealed proof yields {expected!r}"
            )
    award_census = dict(proof.get("award_transition_census") or {})
    for arm, proof_arm in (
        ("award_runs_to_non_runs", "runs_to_non_runs"),
        ("award_runs_to_runs", "runs_to_runs"),
    ):
        census_expected = award_census.get(proof_arm)
        disclosed = dict(close.get("currency") or {}).get(arm)
        if disclosed != census_expected:
            findings.append(
                f"the close discloses {arm}={disclosed!r}, but gate recomputation "
                f"over the sealed proof yields {census_expected!r}"
            )
    for arm, proof_arm in (
        ("award_transition_instrumentation", "instrumentation"),
        ("award_transition_windows", "coverage_windows"),
    ):
        census_expected = award_census.get(proof_arm)
        disclosed = dict(close.get("currency") or {}).get(arm)
        if disclosed != census_expected:
            findings.append(
                f"the close discloses {arm}={disclosed!r}, but gate recomputation "
                f"over the sealed proof yields {census_expected!r}"
            )
    return tuple(findings)


def assert_close_discloses_the_census(
    close: Mapping[str, Any],
    *,
    records_root: Path,
    lineage_member_ids: Iterable[str],
    introduced_member_ids: Iterable[str],
) -> None:
    """Refuse to seal a close whose disclosure census lies (DX-22).

    The producer runs this at close BUILD, before the append: a lying census
    is a producer bug, and leaving it in the ledger means the next rung's
    staging refuses hours later at the end of a spent leg instead of the
    close refusing now. Disposition-negative -- this can only refuse; it
    never upgrades, publishes, or licenses anything.

    Parameters
    ----------
    close:
        The close row about to be appended.
    records_root:
        Campaign canonical records directory holding the sealed proof.
    lineage_member_ids:
        Stable ids the lineage snapshot's own bytes contain.
    introduced_member_ids:
        Stable ids this rung introduced.

    Raises
    ------
    CloseObjectError
        If any disclosure or accounting finding fires.
    """

    refusals = close_disclosure_refusals(
        close,
        records_root=records_root,
        lineage_member_ids=lineage_member_ids,
        introduced_member_ids=introduced_member_ids,
    )
    if refusals:
        raise CloseObjectError(
            "close build refuses to seal a lying census: " + "; ".join(refusals)
        )


def _charged_park_events(records_root: Path) -> tuple[JsonObject, ...]:
    """Derive the budget-charging park events, incrementally held (D-21).

    The continuation disclosure needs only the events that CHARGE the
    scheduler-quarantine retry budget (the exact
    ``_park_event_key`` shape); folding them through the close-read substrate
    keeps the growing operational ledger off the whole-file path -- the same
    discipline every other close-path ledger read follows.
    """

    from menagerie.crawler.recordio import operational_events_path
    from menagerie.crawler.scheduler_parks import _park_event_key  # noqa: PLC2701 -- the ONE charged-event key

    path = operational_events_path(records_root)
    if not path.is_file():
        return ()

    def extend(
        value: tuple[JsonObject, ...], rows: Iterable[LedgerRow]
    ) -> tuple[JsonObject, ...]:
        kept = list(value)
        for _number, _offset, line in rows:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, Mapping):
                continue
            if _park_event_key(event) is None:
                continue
            kept.append(
                {
                    "created_at": event.get("created_at"),
                    "details": dict(event.get("details") or {}),
                }
            )
        return tuple(kept)

    charged: tuple[JsonObject, ...] = derive(
        path,
        projection="close.charged-park-events",
        seed=tuple,
        extend=extend,
        # Charged park events only: O(#parks), tiny beside the ledger.
        weigh=lambda value, _byte_count: (
            PROJECTION_BASE_WEIGHT + DECODED_ROW_WEIGHT_FACTOR * 256 * len(value)
        ),
    )
    return charged


def continuing_member_disclosures(
    records_root: Path, proof: Mapping[str, Any]
) -> tuple[JsonObject, ...]:
    """Derive the close-currency ``continuing_members`` block from one proof.

    RULING W7 clause 6: each CONTINUING member's cited row, kind, reason code,
    charged-park count, and HELD state (OPEN-5 default: an exhausted member
    holds as operator-owed debt while the close completes). ONE derivation for
    the driver and every fixture -- a second spelling anywhere is the
    K2/split-brain class reborn. This is DISCLOSURE riding the close currency:
    the sealed proof carries the classification; nothing here can improve a
    verdict.
    """

    from menagerie.crawler.partition_proof import CONTINUING
    from menagerie.crawler.scheduler_parks import (
        load_scheduler_quarantine_ledger,
        scheduler_quarantine_path,
        scheduler_quarantine_released,
    )
    from menagerie.crawler.scheduler_parks import (
        budget_restoring_boundaries as _boundaries,
    )
    from menagerie.crawler.scheduler_parks import _park_event_key  # noqa: PLC2701 -- the ONE charged-event key

    members = {
        str(member): dict(entry)
        for member, entry in dict(proof.get("members") or {}).items()
        if dict(entry).get("disposition") == CONTINUING
    }
    if not members:
        return ()
    ledger = load_scheduler_quarantine_ledger([scheduler_quarantine_path(records_root)])
    rows_by_digest = {str(row.get("row_sha256")): row for row in ledger}
    events = _charged_park_events(records_root)
    boundaries = _boundaries(ledger, events)
    disclosures: list[JsonObject] = []
    for stable_id in sorted(members):
        evidence = dict(members[stable_id].get("evidence") or {})
        disclosure: JsonObject = {
            "stable_id": stable_id,
            "source": evidence.get("source"),
            "row_index": evidence.get("row_index"),
            "row_sha256": evidence.get("row_sha256"),
            "kind": evidence.get("kind"),
            "reason_code": evidence.get("reason_code"),
            "charged_parks": 0,
            "held": False,
        }
        cited = rows_by_digest.get(str(evidence.get("row_sha256")))
        if cited is not None:
            boundary = boundaries.get(
                (str(cited.get("stable_id")), str(cited.get("work_id"))), ""
            )
            charged = 0
            for event in events:
                key = _park_event_key(event)
                if (
                    key is not None
                    and key[0] == str(cited.get("stable_id"))
                    and key[1] == str(cited.get("work_id"))
                    and key[2] > boundary
                ):
                    charged += 1
            disclosure["charged_parks"] = charged
            disclosure["held"] = not scheduler_quarantine_released(
                cited, events, ledger_rows=ledger
            )
        disclosures.append(disclosure)
    return tuple(disclosures)


def close_object_identity_refusals(
    close: Mapping[str, Any],
    *,
    campaign_id: str,
    intake_snapshot_sha256: str,
) -> tuple[str, ...]:
    """Return every way one close attests something other than THIS rung.

    ``find_close_object`` matches an intake id STRING and a phase, and one
    shared ``records/intake/leg-closes.jsonl`` serves all four frozen
    campaigns -- ``campaign_merge`` and ``merge_host_shards`` are the paths
    that put another campaign's rows into one root. A close is bound to the
    registration it licenses by campaign AND by the digest the registry
    recorded for that snapshot, so an id collision is not a licence (2026-08-16
    red-team round 2, break 3).

    Parameters
    ----------
    close:
        Validated close object.
    campaign_id:
        Campaign of the registration being licensed.
    intake_snapshot_sha256:
        Registered content digest of the lineage snapshot.

    Returns
    -------
    tuple[str, ...]
        Human-readable refusals; empty when the close attests this rung.
    """

    findings: list[str] = []
    attested_campaign = str(close.get("campaign_id"))
    if attested_campaign != campaign_id:
        findings.append(
            f"the close attests campaign {attested_campaign!r}, not {campaign_id!r}: "
            "one shared close ledger serves every campaign, so a close licenses "
            "only the campaign it names"
        )
    attested_digest = str(close.get("intake_snapshot_sha256"))
    if attested_digest != intake_snapshot_sha256:
        findings.append(
            f"the close attests intake digest {attested_digest}, not the "
            f"registered {intake_snapshot_sha256}: the id string is not the rung"
        )
    return tuple(findings)


def require_closed_lineage(
    records_root: Path,
    *,
    intake_snapshot_id: str,
    campaign_id: str,
    intake_snapshot_sha256: str,
    lineage_member_ids: Iterable[str],
    introduced_member_ids: Iterable[str],
    label: str,
    phase: Optional[str] = None,
) -> JsonObject:
    """Require a v2 close whose sealed proof licenses the lineage boundary.

    Gate order (each an independent refusal family):

    1. a close exists for the lineage snapshot (id + phase lookup);
    2. it is v2 -- a v1 close is readable history that licenses nothing, and
       the refusal names the backfill remedy;
    3. it attests THIS campaign and THIS registered digest (id strings are
       not rungs);
    4. its fast-fail disclosures are clean (unresolved verdict, coverage
       holes, stale census -- cheap, retained, no longer where safety lives);
    5. the source census re-observes byte-identically, every slot,
       presence included (RD-1);
    6. the sealed partition proof recomputes bitwise from the pinned bytes:
       leg-open row bound, window floors prefix-verified and anchored against
       the campaign's previous close, every member's citation re-derived
       (RD-2). The identity, membership, and evidence all come from durable
       state -- the close supplies the attestation, never the facts it is
       checked against.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    intake_snapshot_id:
        Prior rung snapshot the new rung derives from.
    campaign_id:
        Campaign that owns the registration being licensed.
    intake_snapshot_sha256:
        Registered content digest of that lineage snapshot.
    lineage_member_ids:
        Stable ids the lineage snapshot's durable bytes contain.
    label:
        Diagnostic name of the rung being staged.
    phase:
        Phase scope of the close being demanded.

    Returns
    -------
    dict[str, Any]
        The satisfying close object.

    Raises
    ------
    CloseObjectError
        If no licensing close exists, or any gate family above refuses.
    """

    # THE licensing verdict, and therefore ONE close-read epoch (round-3 B1):
    # every door that licenses a boundary -- ``stage_rung``,
    # ``register_existing_snapshot``, and any future one -- reaches the gates
    # below through here, so this is where the decision's read window belongs.
    # Nested inside a caller's epoch it joins it and costs nothing.
    with read_epoch():
        return _require_closed_lineage(
            records_root,
            intake_snapshot_id=intake_snapshot_id,
            campaign_id=campaign_id,
            intake_snapshot_sha256=intake_snapshot_sha256,
            lineage_member_ids=lineage_member_ids,
            introduced_member_ids=introduced_member_ids,
            label=label,
            phase=phase,
        )


def _require_closed_lineage(
    records_root: Path,
    *,
    intake_snapshot_id: str,
    campaign_id: str,
    intake_snapshot_sha256: str,
    lineage_member_ids: Iterable[str],
    introduced_member_ids: Iterable[str],
    label: str,
    phase: Optional[str] = None,
) -> JsonObject:
    """Run every licensing gate inside the caller's open close-read epoch."""

    close = find_close_object(
        load_close_objects([close_object_path(records_root)]),
        intake_snapshot_id=intake_snapshot_id,
        phase=phase,
    )
    if close is None:
        raise CloseObjectError(
            f"cannot stage {label!r}: the lineage snapshot {intake_snapshot_id} has "
            "no immutable close object. A rung boundary is crossed only on a "
            "durable, content-addressed close attestation -- interruption JSON, a "
            "raw-head tally, and a hand-reconciled spreadsheet are not closes "
            f"(T1-sol-02). Close the prior leg and re-run staging: {close_object_path(records_root)}"
        )
    if close.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION_V1:
        raise CloseObjectError(
            f"cannot stage {label!r}: the newest close for {intake_snapshot_id} is "
            "a leg-close.v1 attestation, which licenses nothing -- it predates "
            "the leg window, the source census, and the sealed partition proof. "
            "It stays readable history; mint an honest v2 close from the "
            "recorded evidence with tools/backfill_close"
        )
    from menagerie.crawler.partition_proof import (  # noqa: PLC0415 -- lazy: breaks the proof<->close import cycle
        licence_refusals,
        proof_refusals,
    )

    members = tuple(str(member) for member in lineage_member_ids)
    introduced = tuple(str(member) for member in introduced_member_ids)
    from menagerie.crawler.ledger_chain import ledger_chain_refusals

    refusals = (
        *ledger_chain_refusals(records_root),
        *close_object_identity_refusals(
            close,
            campaign_id=campaign_id,
            intake_snapshot_sha256=intake_snapshot_sha256,
        ),
        *close_object_refusals(close),
        *close_disclosure_refusals(
            close,
            records_root=records_root,
            lineage_member_ids=members,
            introduced_member_ids=introduced,
        ),
        *close_object_disagreements(close, records_root),
    )
    if not refusals:
        # The proof recomputation re-reads every pinned source; running it
        # after the census re-observation means it recomputes over exactly
        # the bytes the close attested.
        refusals = tuple(proof_refusals(records_root, close, lineage_member_ids=members))
    if not refusals:
        refusals = tuple(
            licence_refusals(
                records_root,
                close,
                lineage_member_ids=members,
                introduced_member_ids=introduced,
            )
        )
    if refusals:
        raise CloseObjectError(
            f"cannot stage {label!r}: the close object "
            f"{close['close_sha256']} for {intake_snapshot_id} does not license a "
            "next rung -- " + "; ".join(refusals)
        )
    return close
