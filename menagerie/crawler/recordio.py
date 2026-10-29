"""Crash-safe, single-writer append-only JSONL ledger I/O."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    from menagerie.crawler.checkpoint_cache import ValidationMemo

from menagerie.crawler.constants import (
    ARTIFACT_EVENT_SCHEMA_VERSION,
    ATTEMPT_SCHEMA_VERSION,
    ATTEMPT_SCHEMA_VERSION_V3,
    GATE_SCHEMA_VERSION,
    GATE_SCHEMA_VERSION_V3,
    MODEL_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION_V3,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    TEXT_CORPUS_INDEX_SCHEMA_VERSION,
)
from menagerie.crawler.identity import (
    DARWIN_F_FULLFSYNC as DARWIN_F_FULLFSYNC,  # re-export: historical import site
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    hash_bytes,
    payload_hash,
    stable_hash,
    stamp_payload_digest,
    stamp_shadow_rehearsal,
    utc_now,
)
from menagerie.crawler.identity import (
    durable_fsync as durable_fsync,  # re-export: historical import site
)
from menagerie.crawler.models import AppendResult, JsonObject, TailRecoveryEvidence
from menagerie.crawler.schema import (
    SCHEMA_FILES,
    PayloadValidationError,
    RequiredFieldProjection,
    required_field_projection_spec,
    validate_payload,
)

_SCHEMA_READ_VERSIONS: dict[str, frozenset[str]] = {
    MODEL_SCHEMA_VERSION: frozenset({MODEL_SCHEMA_VERSION}),
    MODEL_SCHEMA_VERSION_V3: frozenset({MODEL_SCHEMA_VERSION, MODEL_SCHEMA_VERSION_V3}),
    ATTEMPT_SCHEMA_VERSION: frozenset({ATTEMPT_SCHEMA_VERSION}),
    ATTEMPT_SCHEMA_VERSION_V3: frozenset({ATTEMPT_SCHEMA_VERSION, ATTEMPT_SCHEMA_VERSION_V3}),
    GATE_SCHEMA_VERSION: frozenset({GATE_SCHEMA_VERSION}),
    GATE_SCHEMA_VERSION_V3: frozenset({GATE_SCHEMA_VERSION, GATE_SCHEMA_VERSION_V3}),
    ARTIFACT_EVENT_SCHEMA_VERSION: frozenset({ARTIFACT_EVENT_SCHEMA_VERSION}),
    OPERATIONAL_EVENT_SCHEMA_VERSION: frozenset({OPERATIONAL_EVENT_SCHEMA_VERSION}),
    TEXT_CORPUS_INDEX_SCHEMA_VERSION: frozenset({TEXT_CORPUS_INDEX_SCHEMA_VERSION}),
}

_MODEL_SCHEMA_VERSIONS = frozenset({MODEL_SCHEMA_VERSION, MODEL_SCHEMA_VERSION_V3})
_ATTEMPT_SCHEMA_VERSIONS = frozenset({ATTEMPT_SCHEMA_VERSION, ATTEMPT_SCHEMA_VERSION_V3})
_GATE_SCHEMA_VERSIONS = frozenset({GATE_SCHEMA_VERSION, GATE_SCHEMA_VERSION_V3})


class LedgerError(RuntimeError):
    """Base class for canonical-ledger failures."""


class SingleWriterError(LedgerError):
    """Raised when another process owns the ledger writer lock."""


class LedgerThreadAffinityError(LedgerError):
    """Raised when a thread other than the owning writer thread appends.

    ``JsonlLedger.append`` derives the next sequence from ``self._records``
    with no thread lock; it is correct only under the one-writer-thread
    convention (`driver_contracts` single-writer discipline). The scheduler
    makes that convention load-bearing, so the ledger enforces it as a typed
    refusal instead of a silent interleaving race (T5-5).
    """


class LedgerCorruptionError(LedgerError):
    """Raised for a malformed complete line or invalid persisted record."""


class TornTailError(LedgerError):
    """Raised when a ledger ends in a non-newline-terminated tail."""


class LedgerConflictError(LedgerError):
    """Raised when one immutable identity maps to different payload bytes."""


class AttemptSlotResolutionError(LedgerError):
    """Raised when canonical history cannot satisfy one deterministic v3 slot."""


class LedgerDivergenceError(LedgerError):
    """Raised when the on-disk ledger no longer matches the writer's state.

    An append writes at the byte extent the writer proved at open and has
    tracked since. If the file's size disagrees -- residue of this writer's own
    failed write that could not be restored, or a foreign raw append while the
    lock was held -- writing blindly would FUSE the new line into the stray
    bytes, minting a complete malformed line that recovery is forbidden to
    repair. The append refuses instead; a fresh open re-validates the ledger
    and recovers a torn tail through the ordinary evidence path.
    """


def _lock_identity_holds(handle: BinaryIO, lock_path: Path) -> bool:
    """Return whether a held flock still covers the file AT ``lock_path``.

    ``flock`` binds to an inode, not a name: a lock file deleted or replaced
    while held leaves the holder locking an orphaned inode while the path
    admits a fresh writer. The holder therefore re-proves, at acquisition and
    before every append, that its locked descriptor and the current directory
    entry are one and the same inode.

    Parameters
    ----------
    handle:
        Open handle whose descriptor holds (or is about to hold) the lock.
    lock_path:
        Lock-file path the descriptor was opened from.

    Returns
    -------
    bool
        True only when the path resolves to the exact locked inode.
    """

    try:
        current = os.stat(lock_path)
        held = os.fstat(handle.fileno())
    except OSError:
        return False
    return held.st_nlink > 0 and (current.st_dev, current.st_ino) == (held.st_dev, held.st_ino)


def _writer_lock_path(path: Path) -> Path:
    """Return the canonical writer lock-file path beside one ledger.

    Parameters
    ----------
    path:
        Ledger path.

    Returns
    -------
    pathlib.Path
        ``<ledger>.lock``.
    """

    return path.with_suffix(f"{path.suffix}.lock")


def _fsync_directory(path: Path) -> None:
    """Fsync a directory after creating or replacing one of its entries.

    Parameters
    ----------
    path:
        Directory to synchronize.
    """

    descriptor = os.open(path, os.O_RDONLY)
    try:
        durable_fsync(descriptor)
    finally:
        os.close(descriptor)


def _logical_payload(record: Mapping[str, Any]) -> JsonObject:
    """Remove append-assigned fields for idempotency comparison.

    The returned mapping shares the record's nested values: it exists only for
    equality comparison against another logical payload, never for mutation
    (the historical per-value deepcopy made every replay/conflict check
    O(record size) in copies for a read-only comparison).

    Parameters
    ----------
    record:
        Persisted or proposed ledger record.

    Returns
    -------
    dict[str, Any]
        Logical immutable payload view.
    """

    omitted = {"payload_sha256", "ledger_seq", "record_revision", "record_seq"}
    if record.get("schema_version") == OPERATIONAL_EVENT_SCHEMA_VERSION:
        omitted.add("created_at")
    return {key: value for key, value in record.items() if key not in omitted}


def deterministic_attempt_id(
    *,
    work_id: str,
    execution_identity: str,
    cold_index: int,
    mode: str,
    retry_generation: int = 0,
) -> str:
    """Derive the immutable identity for one execution slot.

    Parameters
    ----------
    work_id, execution_identity:
        Exact work generation and executable dependency closure.
    cold_index:
        Zero-based cold-run index.
    mode:
        Exact isolated execution mode.
    retry_generation:
        DX-10/RE-11 typed retry generation: how many host/machinery-owned
        failure rows this slot has already CONSUMED (each is weather about the
        infrastructure, never slot satisfaction). Generation zero keeps the
        historical identity spelling byte-for-byte, so every recorded row and
        carried head stays addressable.

    Returns
    -------
    str
        Canonical attempt identity used by the ledger and worker request.

    Raises
    ------
    ValueError
        If any slot component is empty or an index is negative.
    """

    if not work_id.strip() or not execution_identity.strip() or not mode.strip():
        raise ValueError("attempt slot identities and mode must be non-empty")
    if cold_index < 0:
        raise ValueError("attempt slot cold_index must be non-negative")
    if retry_generation < 0:
        raise ValueError("attempt slot retry_generation must be non-negative")
    if retry_generation > 0:
        return stable_hash(
            {
                "work_id": work_id,
                "execution_identity": execution_identity,
                "cold_index": cold_index,
                "mode": mode,
                "retry_generation": retry_generation,
            }
        )
    return stable_hash(
        {
            "work_id": work_id,
            "execution_identity": execution_identity,
            "cold_index": cold_index,
            "mode": mode,
        }
    )


def driver_failure_attempt_identity(
    *, work_id: str, stage: str, reason_code: str, root_cause_fingerprint: str
) -> str:
    """Derive the immutable identity of one driver-observed lane failure.

    This is the ONE spelling of the failure-attempt identity basis. The driver
    derives it when it records a model-local lane failure, and the reducer
    re-derives it from a RECORDED row's own fields to recognize a replay: the
    identity is time-free, so a resumed leg that re-observes the same failure
    (same work generation, stage, reason, and root-cause fingerprint) reaches
    the same identity with nothing new to say -- only its wall-clock fields
    differ -- and must reuse the recorded row rather than conflict with it.

    Parameters
    ----------
    work_id, stage, reason_code:
        Exact work generation and failure classification.
    root_cause_fingerprint:
        Stable hash of the failure signature (stable_id, stage, reason,
        exception type, and message).

    Returns
    -------
    str
        Canonical attempt identity for the failure slot.
    """

    return stable_hash(
        {
            "work_id": work_id,
            "stage": stage,
            "reason_code": reason_code,
            "root_cause_fingerprint": root_cause_fingerprint,
        }
    )


@dataclass(frozen=True)
class AttemptSlotResolution:
    """One deterministic slot's resolved state across its retry generations.

    Parameters
    ----------
    record:
        The satisfying row (a success, or a MODEL/AUTHOR-owned failure -- a
        verdict candidate), or ``None`` when the slot's current generation has
        no row yet and must run.
    retry_generation:
        The generation the satisfying row occupies, or the generation a fresh
        run must mint at (DX-10/RE-11).
    consumed_non_model_rows:
        Attempt identities of the host/machinery-owned failure rows this slot
        walked past: visible immutable weather, never slot satisfaction.
    """

    record: Optional[JsonObject]
    retry_generation: int
    consumed_non_model_rows: tuple[str, ...] = ()


def resolve_attempt_slot_state(
    records: Iterable[Mapping[str, Any]],
    *,
    work_id: str,
    execution_identity: str,
    cold_index: int,
    mode: str,
    non_model_failure: Optional[Callable[[Mapping[str, Any]], bool]] = None,
) -> AttemptSlotResolution:
    """Resolve one deterministic slot, walking past non-model failure rows.

    DX-10/RE-11 (the owner-blind slot fix): a host/machinery-owned failure row
    is immutable evidence about the INFRASTRUCTURE, never about the model --
    treating it as slot satisfaction froze the false failure into every resume
    (B3-XQ-2's narrow rule; the full identity migration stays deferred to
    fan-out). Each consumed non-model row advances the typed retry generation,
    so the retry mints a DISTINCT deterministic identity instead of colliding
    with the recorded weather.

    Parameters
    ----------
    records:
        Canonical attempt rows, including immutable legacy history.
    work_id, execution_identity, cold_index, mode:
        Exact slot tuple used to derive the attempt identities.
    non_model_failure:
        Predicate over a failed row's structured error returning whether its
        classified owner forbids model verdicts. ``None`` keeps the historical
        single-generation semantics (every recorded row satisfies).

    Returns
    -------
    AttemptSlotResolution
        The satisfying row (if any), its generation, and the consumed weather.

    Raises
    ------
    AttemptSlotResolutionError
        If same-ID history is duplicate, legacy, malformed, or contradicts the
        exact slot tuple (including a carried retry generation that disagrees
        with the identity that binds it).
    """

    rows = list(records)
    consumed: list[str] = []
    generation = 0
    while True:
        attempt_id = deterministic_attempt_id(
            work_id=work_id,
            execution_identity=execution_identity,
            cold_index=cold_index,
            mode=mode,
            retry_generation=generation,
        )
        record = _validated_slot_row(
            rows,
            attempt_id,
            work_id=work_id,
            execution_identity=execution_identity,
            cold_index=cold_index,
            mode=mode,
            retry_generation=generation,
        )
        if record is None:
            return AttemptSlotResolution(None, generation, tuple(consumed))
        error = record.get("error")
        if (
            non_model_failure is not None
            and record.get("result") == "failed"
            and isinstance(error, Mapping)
            and non_model_failure(error)
        ):
            consumed.append(attempt_id)
            generation += 1
            continue
        return AttemptSlotResolution(record, generation, tuple(consumed))


def resolve_attempt_slot(
    records: Iterable[Mapping[str, Any]],
    *,
    work_id: str,
    execution_identity: str,
    cold_index: int,
    mode: str,
) -> Optional[JsonObject]:
    """Resolve one deterministic slot from canonical attempt history.

    The function intentionally performs no scratch lookup and never chooses a
    highest or latest record. A matching v2, duplicate, or slot-contradictory row
    is visible immutable history but cannot silently satisfy or rerun the slot.
    Raw-receipt and parent-proof authentication remains the reducer's authority
    boundary; the Phase-2 integrator supplies only authenticated current-v3 rows.

    Parameters
    ----------
    records:
        Canonical attempt rows, including immutable legacy history.
    work_id, execution_identity, cold_index, mode:
        Exact slot tuple used to derive the attempt identity.

    Returns
    -------
    dict[str, Any] | None
        Defensive copy of the sole exact v3 row, or ``None`` when absent.

    Raises
    ------
    AttemptSlotResolutionError
        If same-ID history is duplicate, legacy, malformed, or contradicts the
        exact slot tuple.
    """

    return resolve_attempt_slot_state(
        records,
        work_id=work_id,
        execution_identity=execution_identity,
        cold_index=cold_index,
        mode=mode,
    ).record


def _validated_slot_row(
    rows: list[Mapping[str, Any]],
    attempt_id: str,
    *,
    work_id: str,
    execution_identity: str,
    cold_index: int,
    mode: str,
    retry_generation: int,
) -> Optional[JsonObject]:
    """Return the sole validated v3 row for one exact slot identity, if any.

    Parameters
    ----------
    rows:
        Canonical attempt rows.
    attempt_id:
        The exact generation-bearing slot identity.
    work_id, execution_identity, cold_index, mode, retry_generation:
        The slot tuple the row must corroborate.

    Returns
    -------
    dict[str, Any] | None
        Defensive copy of the sole exact v3 row, or ``None`` when absent.

    Raises
    ------
    AttemptSlotResolutionError
        If same-ID history is duplicate, legacy, malformed, or contradictory.
    """

    matches = [record for record in rows if record.get("attempt_id") == attempt_id]
    if not matches:
        return None
    if len(matches) != 1:
        raise AttemptSlotResolutionError(
            f"deterministic attempt slot has duplicate canonical rows: {attempt_id}"
        )
    record = matches[0]
    if record.get("schema_version") != ATTEMPT_SCHEMA_VERSION_V3:
        raise AttemptSlotResolutionError(
            f"deterministic attempt slot is not current v3: {attempt_id}"
        )
    try:
        validate_payload(record, ATTEMPT_SCHEMA_VERSION_V3)
    except PayloadValidationError as exc:
        raise AttemptSlotResolutionError(
            f"deterministic attempt slot has an invalid v3 row {attempt_id}: {exc}"
        ) from exc
    identities = record.get("identities")
    retries = record.get("retries")
    invocation = record.get("invocation")
    raw_receipt = record.get("raw_award_receipt")
    observed_modes = {
        value
        for value in (
            record.get("mode"),
            invocation.get("mode") if isinstance(invocation, Mapping) else None,
            raw_receipt.get("requested_mode") if isinstance(raw_receipt, Mapping) else None,
        )
        if value is not None
    }
    if (
        record.get("work_id") != work_id
        or not isinstance(identities, Mapping)
        or identities.get("execution") != execution_identity
        or not isinstance(retries, Mapping)
        or retries.get("stage_attempt") != cold_index + 1
        or int(retries.get("slot_retry_generation") or 0) != retry_generation
        or observed_modes - {mode}
    ):
        raise AttemptSlotResolutionError(
            f"deterministic attempt row contradicts its exact slot: {attempt_id}"
        )
    return deepcopy(dict(record))


def _stamped_record(record: Mapping[str, Any], *, field: str) -> JsonObject:
    """Return one fact record stamped by the SHARED payload-digest helper.

    Both fact-ledger digest fields route through
    :func:`menagerie.crawler.identity.stamp_payload_digest`, so this writer spells no
    preimage of its own and cannot drift from the chain's content address (PSLRE6 acct
    F2).  The helper's coupling refusal is re-raised in this module's typed
    vocabulary: a record body carrying a top-level content-address name would have a
    payload digest that is NOT its chain address, so it is refused where it is minted
    rather than by every chain read for the rest of the ledger's life (PSLRE6 D6-3).

    Parameters
    ----------
    record:
        Complete record body without its own payload digest.
    field:
        ``record_revision`` or ``payload_sha256``.

    Returns
    -------
    JsonObject
        The record carrying its stamped payload digest.

    Raises
    ------
    PayloadValidationError
        If the body carries a top-level content-address name.
    """

    try:
        return stamp_payload_digest(record, field=field)
    except ValueError as exc:
        raise PayloadValidationError(str(exc)) from exc


def _identity_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the immutable replay/conflict key for a ledger record.

    Parameters
    ----------
    record:
        Ledger record.

    Returns
    -------
    tuple[Any, ...]
        Type-tagged identity key.
    """

    version = record.get("schema_version")
    if version == MODEL_SCHEMA_VERSION_V3:
        identity_fields = required_field_projection_spec(
            RequiredFieldProjection.MODEL_LEDGER_IDENTITY
        ).field_order
        stable_id_field, revision_field = identity_fields
        revision = record.get(revision_field)
        if revision is not None:
            return (version, record.get(stable_id_field), revision)
        return (version, record.get(stable_id_field), record.get("record_seq"))
    if version == MODEL_SCHEMA_VERSION:
        revision = record.get("record_revision")
        if revision is not None:
            return (version, record.get("stable_id"), revision)
        return (version, record.get("stable_id"), record.get("record_seq"))
    if version == ATTEMPT_SCHEMA_VERSION_V3:
        identity_fields = required_field_projection_spec(
            RequiredFieldProjection.ATTEMPT_LEDGER_IDENTITY
        ).field_order
        return (version, *(record.get(field) for field in identity_fields))
    if version == ATTEMPT_SCHEMA_VERSION:
        return (version, record.get("attempt_id"))
    if version == GATE_SCHEMA_VERSION_V3:
        identity_fields = required_field_projection_spec(
            RequiredFieldProjection.GATE_LEDGER_IDENTITY
        ).field_order
        return (version, *(record.get(field) for field in identity_fields))
    if version == GATE_SCHEMA_VERSION:
        return (version, record.get("gate_id"))
    if version == ARTIFACT_EVENT_SCHEMA_VERSION:
        return (version, record.get("artifact_event_id"))
    if version == TEXT_CORPUS_INDEX_SCHEMA_VERSION:
        return (version, record.get("corpus_row_id"))
    return (version, record.get("event_id") or record.get("proposal_id"))


def _verify_hash(record: Mapping[str, Any]) -> None:
    """Verify a persisted record's payload/revision digest.

    Parameters
    ----------
    record:
        Parsed persisted record.

    Raises
    ------
    LedgerCorruptionError
        If a present digest does not match canonical payload bytes.
    """

    digest_field = (
        "record_revision" if record.get("schema_version") in _MODEL_SCHEMA_VERSIONS else None
    )
    if digest_field is None and "payload_sha256" in record:
        digest_field = "payload_sha256"
    if digest_field is not None and record[digest_field] != payload_hash(record):
        raise LedgerCorruptionError(f"invalid {digest_field} for {_identity_key(record)!r}")


def record_validation_memo_key(record: Mapping[str, Any]) -> Optional[str]:
    """Return the content-identity memo key for one persisted record.

    The key exists only for records carrying a self-referential payload
    digest, because only that digest is re-verified live against the record
    bytes on every read (:func:`_verify_hash`): a memoized schema validation
    may be skipped exactly when the bytes it validated are re-proven present.

    Parameters
    ----------
    record:
        Parsed persisted record.

    Returns
    -------
    str | None
        ``stable_hash([schema_version, digest])``, or ``None`` when the
        record carries no live-verified digest and must validate live.
    """

    digest_field = (
        "record_revision" if record.get("schema_version") in _MODEL_SCHEMA_VERSIONS else None
    )
    if digest_field is None and "payload_sha256" in record:
        digest_field = "payload_sha256"
    if digest_field is None:
        return None
    digest = record.get(digest_field)
    if not isinstance(digest, str) or not digest:
        return None
    return stable_hash([record.get("schema_version"), digest])


def _decode_validated_line(
    line: bytes,
    *,
    ledger_path: Path,
    line_number: int,
    validate: bool = True,
    validation_memo: Optional["ValidationMemo"] = None,
) -> JsonObject:
    """Decode, schema-validate, and digest-verify one complete ledger line.

    Parameters
    ----------
    line:
        One complete (newline-terminated or bare) ledger line.
    ledger_path, line_number:
        Error-message context.
    validate, validation_memo:
        Exactly the :func:`scan_jsonl` validation controls.

    Returns
    -------
    dict[str, Any]
        Parsed, verified record.

    Raises
    ------
    LedgerCorruptionError
        If the complete line is malformed, invalid, or hash-inconsistent.
    """

    try:
        decoded = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerCorruptionError(
            f"malformed complete JSON at {ledger_path}:{line_number}: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise LedgerCorruptionError(f"non-object JSON record at {ledger_path}:{line_number}")
    try:
        needs_validation = validate and decoded.get("schema_version") in SCHEMA_FILES
        memo_key = (
            record_validation_memo_key(decoded)
            if needs_validation and validation_memo is not None
            else None
        )
        if needs_validation and not (
            memo_key is not None
            and validation_memo is not None
            and validation_memo.seen("record", memo_key)
        ):
            validate_payload(decoded)
        _verify_hash(decoded)
        if memo_key is not None and validation_memo is not None:
            validation_memo.record("record", memo_key)
    except (PayloadValidationError, LedgerCorruptionError) as exc:
        raise LedgerCorruptionError(
            f"invalid record at {ledger_path}:{line_number}: {exc}"
        ) from exc
    return decoded


def scan_jsonl(
    path: Union[str, Path],
    *,
    validate: bool = True,
    validation_memo: Optional["ValidationMemo"] = None,
) -> list[JsonObject]:
    """Scan a complete JSONL ledger, stopping on corruption or a torn tail.

    Parameters
    ----------
    path:
        Ledger path. A missing file is an empty ledger.
    validate:
        Whether to schema-validate recognized crawler payloads.
    validation_memo:
        Optional positive-outcome memo. A memoized record skips only the
        jsonschema pass; payload digests are always re-verified live, so the
        memo can never excuse content-identity verification.

    Returns
    -------
    list[dict[str, Any]]
        Parsed records in byte order.

    Raises
    ------
    TornTailError
        If the final line lacks a newline.
    LedgerCorruptionError
        If any complete line is malformed, invalid, or hash-inconsistent.
    """

    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    records: list[JsonObject] = []
    with ledger_path.open("rb") as handle:
        line_number = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.endswith(b"\n"):
                raise TornTailError(
                    f"torn final line at byte {handle.tell() - len(line)} in {ledger_path}"
                )
            if not line.strip():
                # A whitespace-only TERMINATED line carries no record, so it is not
                # corruption: the partitioner's `empty-deferred.jsonl` and an empty
                # snapshot `sources/deferred.jsonl` are both literally one newline
                # byte, and the checkpoint validator scans every `.jsonl` under
                # `records/`. Nothing is weakened -- a torn tail is a line WITHOUT a
                # newline (refused above) and a truncated record is a partial JSON
                # document (refused by the decoder) -- and this makes the strict
                # reader agree with every lenient reader in the tree, which have
                # always skipped blank lines.
                continue
            records.append(
                _decode_validated_line(
                    line,
                    ledger_path=ledger_path,
                    line_number=line_number,
                    validate=validate,
                    validation_memo=validation_memo,
                )
            )
    return records


class IncrementalJsonlScan:
    """Generation-bound JSONL reader that validates appended suffixes only.

    The cache is process-local and advisory. Shrink, replacement, same-size
    mutation, or an invalid append shape discards it and runs the ordinary
    full scan, so the optimization cannot turn a non-append mutation into an
    accepted record. Canonical writers are append-only and inode-locked; the
    common growth path therefore decodes exactly the bytes after the last
    validated offset.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        validate: bool = True,
        retain: Optional[Callable[[Mapping[str, Any]], bool]] = None,
    ) -> None:
        """Bind one ledger path without reading it.

        Parameters
        ----------
        path:
            Append-only JSONL path.
        validate:
            Whether recognized crawler rows receive schema validation.
        retain:
            Optional predicate selecting the small evidence subset consumers
            need to retain after every row has been validated.
        """

        self.path = Path(path)
        self.validate = validate
        self.retain = retain
        self._rows: list[JsonObject] = []
        self._byte_count = 0
        self._line_count = 0
        self._signature: Optional[tuple[int, int, int, int]] = None

    def _stat_signature(self) -> Optional[tuple[int, int, int, int]]:
        """Return the live inode/extent/mtime signature, or ``None`` if absent."""

        try:
            stat = self.path.stat()
        except OSError:
            return None
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _full_refresh(
        self, signature: Optional[tuple[int, int, int, int]]
    ) -> tuple[JsonObject, ...]:
        """Replace cached state through the canonical full scanner.

        Parameters
        ----------
        signature:
            Stat signature observed immediately before the scan.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Read-only row snapshot.
        """

        rows: list[JsonObject] = []
        line_count = 0
        try:
            handle = self.path.open("rb")
        except FileNotFoundError:
            handle = None
        if handle is not None:
            with handle:
                for line_number, line in enumerate(handle, start=1):
                    line_count = line_number
                    if not line.endswith(b"\n"):
                        raise TornTailError(
                            f"torn final line at byte {handle.tell() - len(line)} in {self.path}"
                        )
                    if not line.strip():
                        continue
                    decoded = _decode_validated_line(
                        line,
                        ledger_path=self.path,
                        line_number=line_number,
                        validate=self.validate,
                    )
                    if self.retain is None or self.retain(decoded):
                        rows.append(decoded)
        self._rows = rows
        self._byte_count = 0 if signature is None else signature[2]
        self._line_count = line_count
        self._signature = self._stat_signature()
        return tuple(self._rows)

    def refresh(self) -> tuple[JsonObject, ...]:
        """Return rows current to the file, decoding only an honest suffix.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Shared read-only rows in ledger order.
        """

        signature = self._stat_signature()
        if signature == self._signature:
            return tuple(self._rows)
        if signature is None:
            self._rows = []
            self._byte_count = 0
            self._line_count = 0
            self._signature = None
            return ()
        if self._signature is None:
            return self._full_refresh(signature)
        old_device, old_inode, old_size, _old_mtime = self._signature
        device, inode, size, _mtime = signature
        if (
            device != old_device
            or inode != old_inode
            or size <= old_size
            or old_size != self._byte_count
        ):
            return self._full_refresh(signature)
        appended: list[JsonObject] = []
        with self.path.open("rb") as handle:
            handle.seek(old_size)
            line_number = self._line_count
            while True:
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.endswith(b"\n"):
                    raise TornTailError(
                        f"torn final line at byte {handle.tell() - len(line)} in {self.path}"
                    )
                if not line.strip():
                    continue
                decoded = _decode_validated_line(
                    line,
                    ledger_path=self.path,
                    line_number=line_number,
                    validate=self.validate,
                )
                if self.retain is None or self.retain(decoded):
                    appended.append(decoded)
        self._rows.extend(appended)
        self._byte_count = size
        self._line_count = line_number
        self._signature = self._stat_signature()
        return tuple(self._rows)


def operational_events_path(records_root: Union[str, Path]) -> Path:
    """Return the canonical append-only operational event ledger.

    The records-root-shaped spelling, matching every valve ledger helper, so a
    read-only reporting tool can locate the ledger without importing the driver
    stack. :func:`menagerie.crawler.checkpoint.canonical_operational_ledger_path`
    is the model-ledger-shaped wrapper over this one authority.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        ``records/operational/events.jsonl``.
    """

    return Path(records_root) / "operational" / "events.jsonl"


#: Frozen schema identity for the advisory per-ledger open-state sidecar.
LEDGER_OPEN_STATE_SCHEMA_VERSION = "ledger-open-state.v1"


def open_state_path(path: Union[str, Path]) -> Path:
    """Return the advisory open-state sidecar path beside one ledger.

    The final suffix is deliberately ``.open-state`` (not ``.json``): the
    checkpoint allowlist sweeps ``records/`` by final suffix, and a derived
    advisory cache must never become a public checkpoint candidate.

    Parameters
    ----------
    path:
        Ledger path.

    Returns
    -------
    pathlib.Path
        Sidecar path (``<ledger>.open-state``).
    """

    ledger_path = Path(path)
    return ledger_path.with_suffix(f"{ledger_path.suffix}.open-state")


def _open_state_environment() -> str:
    """Return the schema-validation environment identity binding a sidecar.

    A recorded prefix digest attests jsonschema conformance only for the
    schema bytes and jsonschema version that validated it, so the sidecar
    reuses W5's checkpoint-cache environment identity: any schema-file or
    library change voids every recorded skip and the next open re-validates
    live.

    Returns
    -------
    str
        Environment identity digest.
    """

    from menagerie.crawler.checkpoint_cache import checkpoint_cache_environment_identity

    return checkpoint_cache_environment_identity()


def _load_open_state(path: Path, schema_version: str) -> Optional[JsonObject]:
    """Load one plausible open-state sidecar, degrading to ``None`` loudly-free.

    The sidecar is an advisory cache: ANY anomaly -- missing, unreadable,
    malformed, wrong schema, wrong ledger schema, wrong environment,
    implausible counters -- returns ``None`` and the open falls back to the
    full validated scan. Nothing in the sidecar can upgrade an outcome; it
    can only skip re-validating bytes whose digest the open itself re-proves.

    Parameters
    ----------
    path:
        Ledger path.
    schema_version:
        The opening writer's exact schema version.

    Returns
    -------
    dict[str, Any] | None
        Plausible sidecar state, or ``None``.
    """

    try:
        raw = open_state_path(path).read_bytes()
    except OSError:
        return None
    try:
        state = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != LEDGER_OPEN_STATE_SCHEMA_VERSION
        or state.get("ledger_schema_version") != schema_version
        or not isinstance(state.get("byte_count"), int)
        or isinstance(state.get("byte_count"), bool)
        or state["byte_count"] <= 0
        or not isinstance(state.get("line_count"), int)
        or isinstance(state.get("line_count"), bool)
        or state["line_count"] < 0
        or not isinstance(state.get("prefix_sha256"), str)
        or state.get("environment") != _open_state_environment()
    ):
        return None
    return state


def _store_open_state(
    path: Path, schema_version: str, *, byte_count: int, line_count: int, prefix_sha256: str
) -> None:
    """Persist the advisory open-state sidecar; a write failure costs only.

    Parameters
    ----------
    path:
        Ledger path.
    schema_version:
        Exact writer schema version.
    byte_count, line_count, prefix_sha256:
        Fully validated ledger extent and its whole-prefix digest.
    """

    if byte_count <= 0:
        return
    state_path = open_state_path(path)
    payload = (
        json.dumps(
            {
                "schema_version": LEDGER_OPEN_STATE_SCHEMA_VERSION,
                "ledger_schema_version": schema_version,
                "environment": _open_state_environment(),
                "byte_count": byte_count,
                "line_count": line_count,
                "prefix_sha256": prefix_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    temporary = state_path.with_name(f".{state_path.name}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, state_path)
    except OSError:
        temporary.unlink(missing_ok=True)


def _tail_is_torn(path: Path) -> bool:
    """Return whether a ledger ends without a newline, by seeking not reading.

    Parameters
    ----------
    path:
        Ledger path.

    Returns
    -------
    bool
        True only for a non-empty ledger whose final byte is not a newline.
    """

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                return False
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except FileNotFoundError:
        return False


def _last_newline_end(handle: BinaryIO, size: int) -> int:
    """Return the offset just past the FINAL newline, scanning backward.

    Parameters
    ----------
    handle:
        Readable binary handle on the ledger.
    size:
        Total file size in bytes.

    Returns
    -------
    int
        Offset of the first torn-tail byte (0 when no newline exists).
    """

    chunk_size = 1 << 16
    position = size
    while position > 0:
        start = max(0, position - chunk_size)
        handle.seek(start)
        chunk = handle.read(position - start)
        newline = chunk.rfind(b"\n")
        if newline != -1:
            return start + newline + 1
        position = start
    return 0


def _stream_validated_prefix(ledger_path: Path, good_end: int) -> None:
    """Run the canonical full validation over exactly ``good_end`` bytes.

    This is the streaming equivalent of materializing the retained prefix and
    running :func:`scan_jsonl` over it: identical per-line decode, schema
    validation, and digest verification, without the whole-prefix buffer or
    the on-disk ``.recovery-scan`` copy the historical recovery paid per kill.

    Parameters
    ----------
    ledger_path:
        Ledger whose retained prefix is being proven.
    good_end:
        Exact retained byte extent; every byte before it ends in a newline.

    Raises
    ------
    LedgerCorruptionError
        If any complete retained line is malformed, invalid, or
        hash-inconsistent.
    """

    remaining = good_end
    line_number = 0
    with ledger_path.open("rb") as handle:
        while remaining > 0:
            line = handle.readline()
            if not line or len(line) > remaining or not line.endswith(b"\n"):
                raise LedgerCorruptionError(
                    f"retained prefix changed under recovery at {ledger_path}"
                )
            line_number += 1
            remaining -= len(line)
            _decode_validated_line(line, ledger_path=ledger_path, line_number=line_number)


def _validate_recovery_prefix(ledger_path: Path, good_end: int) -> None:
    """Validate a recovery prefix from its attested extent plus live suffix.

    The advisory state never becomes authority by itself: its recorded prefix
    digest is re-derived from the ledger bytes before prior schema work is
    reused. Once that digest agrees, only complete rows after the attested
    extent require decoding and validation.

    Parameters
    ----------
    ledger_path:
        Ledger whose retained prefix is being proven.
    good_end:
        Exact retained byte extent.
    """

    try:
        candidate = json.loads(open_state_path(ledger_path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        candidate = None
    schema_version = (
        candidate.get("ledger_schema_version") if isinstance(candidate, Mapping) else None
    )
    state = (
        _load_open_state(ledger_path, schema_version)
        if isinstance(schema_version, str)
        else None
    )
    if state is None or state["byte_count"] > good_end:
        _stream_validated_prefix(ledger_path, good_end)
        return
    hasher = hashlib.sha256()
    remaining = int(state["byte_count"])
    line_number = int(state["line_count"])
    with ledger_path.open("rb") as handle:
        while remaining > 0:
            chunk = handle.read(min(remaining, 1 << 20))
            if not chunk:
                _stream_validated_prefix(ledger_path, good_end)
                return
            hasher.update(chunk)
            remaining -= len(chunk)
        if hasher.hexdigest() != state["prefix_sha256"]:
            _stream_validated_prefix(ledger_path, good_end)
            return
        remaining = good_end - int(state["byte_count"])
        while remaining > 0:
            line = handle.readline()
            if not line or len(line) > remaining or not line.endswith(b"\n"):
                raise LedgerCorruptionError(
                    f"retained prefix changed under recovery at {ledger_path}"
                )
            line_number += 1
            remaining -= len(line)
            _decode_validated_line(
                line,
                ledger_path=ledger_path,
                line_number=line_number,
            )


def _heal_torn_evidence_tail(recovery_path: Path) -> None:
    """Truncate ONLY a torn (non-newline-terminated) evidence-ledger tail.

    A torn evidence tail is the signature of a crash mid-evidence-write: that
    recovery never completed, its ledger was never truncated, and the current
    recovery re-derives the same evidence. Appending after the torn fragment
    would weld it into a malformed COMPLETE line and permanently wedge every
    consumer that scans the evidence ledger (the checkpoint truncation
    validator scans it on every ledger shrink). Complete evidence rows are
    never touched.

    Parameters
    ----------
    recovery_path:
        Evidence ledger to heal before appending.
    """

    if not recovery_path.exists() or not _tail_is_torn(recovery_path):
        return
    with recovery_path.open("r+b") as handle:
        size = handle.seek(0, os.SEEK_END)
        good_end = _last_newline_end(handle, size)
        handle.truncate(good_end)
        handle.flush()
        durable_fsync(handle.fileno())


def recover_torn_tail(
    path: Union[str, Path],
    *,
    evidence_path: Union[str, Path, None] = None,
    prefix_check: Optional[Callable[[Path], object]] = None,
    writer_lock_held: bool = False,
    data_lock_held: bool = False,
) -> Optional[TailRecoveryEvidence]:
    """Evidence and truncate only a non-newline-terminated final line.

    Complete malformed lines are never repaired. Recovery evidence is fsynced before
    the ledger is truncated.

    Truncation is a WRITE: when the ledger has a writer lock file
    (``<ledger>.lock`` -- every ``JsonlLedger`` path) and the caller does not
    already hold it, recovery takes that lock non-blocking for the duration
    and refuses with :class:`SingleWriterError` while a live writer owns it,
    so recovery can never truncate an in-flight acknowledged append. Ledgers
    without a lock file (the hand-rolled valve ledgers, which flock the
    ledger inode itself through :func:`read_with_tail_recovery`) are
    unaffected.

    Parameters
    ----------
    path:
        Ledger to inspect.
    evidence_path:
        Optional append-only recovery ledger. Defaults beside the source ledger.
    prefix_check:
        Strict decoder that must accept the retained content before anything
        is truncated; it receives a path holding candidate bytes and raises
        the ledger's own typed error on a malformed complete line. The
        default is the canonical validation of :func:`scan_jsonl`, streamed
        in place without materializing a prefix copy.
    writer_lock_held:
        Whether the caller already owns ``<ledger>.lock`` (the
        ``JsonlLedger`` open path). Never pass True without holding it.
    data_lock_held:
        Whether the caller already holds an exclusive flock on the ledger's
        DATA inode (the :func:`read_with_tail_recovery` paths). A live
        ``JsonlLedger`` writer holds that flock for its whole lifetime, so
        when neither lock is held recovery additionally takes it non-blocking
        -- a deleted side-lock FILE must not let recovery truncate a live
        writer's in-flight append. Never pass True without holding it.

    Returns
    -------
    TailRecoveryEvidence | None
        Evidence for a removed tail, or None for an already complete ledger.

    Raises
    ------
    LedgerCorruptionError
        If a preceding complete line is malformed (or the ``prefix_check``'s
        own typed error when one is supplied).
    SingleWriterError
        If a live writer owns the ledger's writer lock.
    """

    ledger_path = Path(path)
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        return None
    if not _tail_is_torn(ledger_path):
        if prefix_check is not None:
            prefix_check(ledger_path)
        else:
            scan_jsonl(ledger_path)
        return None
    guard: Optional[BinaryIO] = None
    lock_path = _writer_lock_path(ledger_path)
    if not writer_lock_held and lock_path.exists():
        guard = lock_path.open("a+b")
        try:
            fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            guard.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise SingleWriterError(
                    f"tail recovery refused: a live writer owns {ledger_path}"
                ) from exc
            raise
        if not _lock_identity_holds(guard, lock_path):
            guard.close()
            raise SingleWriterError(
                f"tail recovery refused: writer lock is contended at {ledger_path}"
            )
    data_guard: Optional[BinaryIO] = None
    if not writer_lock_held and not data_lock_held:
        # Second exclusion domain: a live JsonlLedger writer flocks the DATA
        # inode itself, and (unlike the side lock) that flock cannot be swept
        # away by deleting a file. Taking it non-blocking here means a torn
        # tail is only ever truncated when no writer is alive.
        data_guard = ledger_path.open("r+b")
        try:
            fcntl.flock(data_guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            data_guard.close()
            if guard is not None:
                fcntl.flock(guard.fileno(), fcntl.LOCK_UN)
                guard.close()
                guard = None
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise SingleWriterError(
                    f"tail recovery refused: a live writer owns the data inode of {ledger_path}"
                ) from exc
            raise
    try:
        # A SIGKILL mid-prefix-check can strand one .recovery-scan copy; it is
        # always disposable (a partial duplicate of retained bytes), and this
        # is the ONE writer-locked place that can clean it.
        ledger_path.with_name(f".{ledger_path.name}.recovery-scan").unlink(missing_ok=True)
        with ledger_path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            good_end = _last_newline_end(handle, size)
            handle.seek(good_end)
            tail = handle.read()
        if good_end:
            if prefix_check is not None:
                temporary = ledger_path.with_name(f".{ledger_path.name}.recovery-scan")
                try:
                    with temporary.open("wb") as destination, ledger_path.open("rb") as source:
                        remaining = good_end
                        while remaining > 0:
                            chunk = source.read(min(remaining, 1 << 20))
                            if not chunk:
                                raise LedgerCorruptionError(
                                    f"retained prefix changed under recovery at {ledger_path}"
                                )
                            destination.write(chunk)
                            remaining -= len(chunk)
                    prefix_check(temporary)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                _validate_recovery_prefix(ledger_path, good_end)
        recovery_path = (
            Path(evidence_path)
            if evidence_path is not None
            else ledger_path.with_suffix(f"{ledger_path.suffix}.recovery.jsonl")
        )
        recovery_path.parent.mkdir(parents=True, exist_ok=True)
        tail_digest = hash_bytes(tail)
        recovered_at = utc_now()
        evidence_record = {
            "ledger_path": str(ledger_path),
            "byte_offset": good_end,
            "byte_count": len(tail),
            "tail_sha256": tail_digest,
            "recovered_at": recovered_at,
        }
        _heal_torn_evidence_tail(recovery_path)
        with recovery_path.open("ab") as handle:
            handle.write(canonical_json_bytes(evidence_record) + b"\n")
            handle.flush()
            durable_fsync(handle.fileno())
        _fsync_directory(recovery_path.parent)
        with ledger_path.open("r+b") as handle:
            handle.truncate(good_end)
            handle.flush()
            durable_fsync(handle.fileno())
    finally:
        if data_guard is not None:
            fcntl.flock(data_guard.fileno(), fcntl.LOCK_UN)
            data_guard.close()
        if guard is not None:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)
            guard.close()
    evidence = TailRecoveryEvidence(
        ledger_path=ledger_path,
        byte_offset=good_end,
        byte_count=len(tail),
        tail_sha256=tail_digest,
        recovered_at=recovered_at,
        evidence_path=recovery_path,
    )
    return evidence


def read_with_tail_recovery(
    path: Union[str, Path],
    *,
    prefix_check: Callable[[Path], object],
    locked_handle: Optional[BinaryIO] = None,
) -> bytes:
    """Read one whole JSONL ledger, recovering only a torn final line first.

    This is the ONE read shape for the hand-rolled valve ledgers (and any
    other whole-file strict decoder): a non-newline-terminated final line --
    the signature of a SIGKILL, power loss, or ENOSPC mid-append -- is
    evidenced and truncated through :func:`recover_torn_tail`, then the
    complete bytes are returned for the caller's own strict decode. Complete
    malformed lines are never touched: the ``prefix_check`` refuses them with
    the caller's typed error before any truncation, so hard corruption is
    preserved byte-for-byte.

    Parameters
    ----------
    path:
        Ledger to read. A missing file is an empty ledger.
    prefix_check:
        The caller's strict decoder, applied to retained bytes before any
        truncation (see :func:`recover_torn_tail`).
    locked_handle:
        Open handle on ``path`` whose exclusive ``flock`` the caller already
        holds (the valve append pattern). When absent, the read self-locks --
        only if the tail is torn -- so recovery can never race a live
        writer's in-flight append.

    Returns
    -------
    bytes
        Complete, newline-terminated (or empty) ledger content.
    """

    ledger_path = Path(path)
    if locked_handle is not None:
        locked_handle.seek(0)
        payload = locked_handle.read()
        if not payload or payload.endswith(b"\n"):
            return payload
        recover_torn_tail(ledger_path, prefix_check=prefix_check, data_lock_held=True)
        locked_handle.seek(0)
        return locked_handle.read()
    try:
        payload = ledger_path.read_bytes()
    except FileNotFoundError:
        return b""
    if not payload or payload.endswith(b"\n"):
        return payload
    with ledger_path.open("r+b") as handle:
        try:
            # Non-blocking: a live JsonlLedger writer holds the DATA-inode
            # flock for its whole lifetime (hours on a campaign leg), so a
            # blocking wait here would hang an operator tool indefinitely. A
            # live writer also means the "torn" tail is an in-flight append,
            # not a death -- the typed refusal is the correct answer and the
            # caller retries after the writer closes. Valve-shape ledgers
            # (no persistent writer) still acquire instantly.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise SingleWriterError(
                    f"tail recovery refused: {ledger_path} has a live writer "
                    "holding its data inode; retry after the writer closes"
                ) from exc
            raise
        payload = handle.read()
        if payload and not payload.endswith(b"\n"):
            recover_torn_tail(ledger_path, prefix_check=prefix_check, data_lock_held=True)
            handle.seek(0)
            payload = handle.read()
        return payload


class _DiskBackedRecords(Sequence[JsonObject]):
    """Positional record sequence with at most one decoded payload resident."""

    def __init__(self, path: Path, spans: Iterable[tuple[int, int, int]]) -> None:
        """Bind validated record byte spans.

        Parameters
        ----------
        path:
            Canonical ledger path.
        spans:
            ``(offset, length, line_number)`` triples in ledger order.
        """

        self._path = path
        self._spans = list(spans)
        self._cached_index: Optional[int] = None
        self._cached_record: Optional[JsonObject] = None

    def __len__(self) -> int:
        """Return the number of indexed records."""

        return len(self._spans)

    def __getitem__(self, index: Union[int, slice]) -> Any:
        """Decode one indexed row or a requested slice.

        Parameters
        ----------
        index:
            Integer position or slice.

        Returns
        -------
        dict[str, Any] | tuple[dict[str, Any], ...]
            Decoded record or records.
        """

        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        normalized = index if index >= 0 else len(self) + index
        if normalized < 0 or normalized >= len(self):
            raise IndexError(index)
        if self._cached_index == normalized and self._cached_record is not None:
            return self._cached_record
        offset, length, line_number = self._spans[normalized]
        with self._path.open("rb") as handle:
            handle.seek(offset)
            line = handle.read(length)
        if len(line) != length or not line.endswith(b"\n"):
            raise LedgerCorruptionError(f"indexed ledger row changed at {self._path}:{line_number}")
        record = _decode_validated_line(
            line,
            ledger_path=self._path,
            line_number=line_number,
            validate=False,
        )
        self._cached_index = normalized
        self._cached_record = record
        return record

    def __iter__(self) -> Iterator[JsonObject]:
        """Stream decoded records in ledger order without retaining history."""

        if not self._spans:
            return
        with self._path.open("rb") as handle:
            for index, (offset, length, line_number) in enumerate(self._spans):
                handle.seek(offset)
                line = handle.read(length)
                if len(line) != length or not line.endswith(b"\n"):
                    raise LedgerCorruptionError(
                        f"indexed ledger row changed at {self._path}:{line_number}"
                    )
                record = _decode_validated_line(
                    line,
                    ledger_path=self._path,
                    line_number=line_number,
                    validate=False,
                )
                self._cached_index = index
                self._cached_record = record
                yield record

    def append_span(
        self, *, offset: int, length: int, line_number: int, record: JsonObject
    ) -> None:
        """Index one newly durable row and retain only that tail payload.

        Parameters
        ----------
        offset, length, line_number:
            Exact persisted byte location.
        record:
            Just-validated appended record.
        """

        self._spans.append((offset, length, line_number))
        self._cached_index = len(self._spans) - 1
        self._cached_record = record

    @property
    def resident_payload_count(self) -> int:
        """Return the number of decoded payloads retained by the sequence."""

        return int(self._cached_record is not None)


class JsonlLedger:
    """Single-writer append-only canonical ledger.

    Parameters
    ----------
    path:
        JSONL ledger path.
    schema_version:
        Current schema accepted for append. Its declared compatible immutable
        history versions are accepted on read.
    recover_tail:
        Whether startup may evidence and truncate a torn final line.
    """

    #: How often a waiting writer re-polls the flock during the bounded wait.
    _LOCK_POLL_SECONDS = 0.05

    def __init__(
        self,
        path: Union[str, Path],
        schema_version: str,
        *,
        recover_tail: bool = True,
        lock_timeout: float = 0.0,
    ) -> None:
        """Initialize and exclusively lock a canonical ledger.

        Parameters
        ----------
        path:
            JSONL ledger path.
        schema_version:
            Exact schema accepted by this ledger.
        recover_tail:
            Whether to recover a torn final line before scanning.
        lock_timeout:
            Bounded wait for the writer flock. The default ``0`` is the
            historical immediate refusal. A positive bound waits out transient
            contention (a closing sibling writer, an operator tool finishing)
            instead of hard-erroring; contention that outlives the bound stays
            a typed ``SingleWriterError`` (T5-5).
        """

        self.path = Path(path)
        self.schema_version = schema_version
        try:
            self.read_schema_versions = _SCHEMA_READ_VERSIONS[schema_version]
        except KeyError as exc:
            raise ValueError(
                f"schema is not a canonical ledger version: {schema_version!r}"
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = _writer_lock_path(self.path)
        deadline = time.monotonic() + max(0.0, lock_timeout)
        while True:
            handle = self._lock_path.open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                handle.close()
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if time.monotonic() >= deadline:
                    raise SingleWriterError(f"ledger already has a writer: {self.path}") from exc
                time.sleep(self._LOCK_POLL_SECONDS)
                continue
            # flock binds the inode, not the name: if the lock file was
            # unlinked or replaced between open and flock, this lock covers
            # an orphaned inode and excludes nobody -- reacquire from the
            # current directory entry.
            if _lock_identity_holds(handle, self._lock_path):
                self._lock_handle = handle
                break
            handle.close()
        # Second, independent lock domain: the DATA inode itself (C4 c4c
        # design, adjudged the better half of the dual-writer defense). The
        # side lock is a name beside the ledger and can be deleted or swapped;
        # this flock rides the canonical bytes, so an intruder is refused AT
        # OPEN ("data inode already has a writer") instead of the innocent
        # owner discovering the theft at its next append. A fresh ledger has
        # no data file yet; the first append acquires it then.
        self._data_lock_handle: Optional[BinaryIO] = None
        if self.path.exists():
            try:
                self._acquire_data_lock()
            except BaseException:
                fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                self._lock_handle.close()
                raise
        self._closed = False
        self._open_complete = False
        try:
            # Seek-based tail probe first, so a CLEAN ledger runs exactly ONE
            # validated scan (the historical open ran recover_torn_tail's full
            # whole-file check and then a second complete scan_jsonl). A torn
            # tail keeps the strict recovery contract: validate the retained
            # prefix BEFORE truncating anything.
            self.recovery_evidence = (
                recover_torn_tail(self.path, writer_lock_held=True)
                if recover_tail and _tail_is_torn(self.path)
                else None
            )
            self._records, self._byte_count, self._line_count, self._hasher = self._scan_for_open()
            for record in self._records:
                if record.get("schema_version") not in self.read_schema_versions:
                    raise LedgerCorruptionError(
                        f"schema {record.get('schema_version')!r} is not readable by "
                        f"current writer {self.schema_version!r} in {self.path}"
                    )
            self._identity_index: dict[tuple[Any, ...], int] = {}
            self._stable_id_index: dict[Any, list[int]] = {}
            self._field_indices: dict[str, dict[Any, list[int]]] = {
                field: {}
                for field in (
                    "artifact_event_id",
                    "attempt_id",
                    "gate_id",
                    "transaction_id",
                )
            }
            sequence_field = (
                "record_seq" if self.schema_version in _MODEL_SCHEMA_VERSIONS else "ledger_seq"
            )
            self._sequence_field = sequence_field
            self._max_sequence = 0
            for index, record in enumerate(self._records):
                self._register_record(index, record)
        except Exception:
            self.close()
            raise
        # Appends are pinned to the constructing thread: `append` mutates
        # `self._records` and derives sequences with no thread lock, so the
        # single-writer-thread convention is enforced as a typed refusal
        # instead of trusted (T5-5).
        self._writer_thread_id = threading.get_ident()
        # Only a FULLY successful open may attest validated bytes: the flag
        # keeps the failure path in close() from publishing a sidecar for an
        # open that was refused after its scan (e.g. unreadable schema).
        self._open_complete = True
        _store_open_state(
            self.path,
            self.schema_version,
            byte_count=self._byte_count,
            line_count=self._line_count,
            prefix_sha256=self._hasher.hexdigest(),
        )

    def _scan_for_open(
        self,
    ) -> tuple[_DiskBackedRecords, int, int, "hashlib._Hash"]:
        """Run the ONE startup scan, streaming validation over new rows only.

        Where a plausible open-state sidecar exists and the recorded prefix
        digest still holds against the file's own bytes, the prefix -- bytes
        this writer's own prior open fully schema-validated and
        digest-verified -- is parsed without re-validation, and only rows
        beyond it validate live. ANY mismatch (changed bytes, short file,
        unparseable prefix, wrong row count) falls back to the full validated
        scan, so corruption is always caught by exactly the checks the
        historical open ran.

        Returns
        -------
        tuple[_DiskBackedRecords, int, int, hashlib._Hash]
            Positional record index, validated byte extent, line count, and
            the running whole-ledger digest positioned at that extent.
        """

        spans: list[tuple[int, int, int]] = []
        hasher = hashlib.sha256()
        byte_count = 0
        line_count = 0
        try:
            handle = self.path.open("rb")
        except FileNotFoundError:
            return _DiskBackedRecords(self.path, spans), byte_count, line_count, hasher
        with handle:
            state = _load_open_state(self.path, self.schema_version)
            if state is not None:
                # Stream the recorded prefix line by line (no whole-prefix
                # buffer): hash and parse as we go, but COMMIT the parsed rows
                # only after the digest over exactly byte_count bytes matches.
                accepted = False
                remaining = state["byte_count"]
                clean = True
                while remaining > 0:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line or len(line) > remaining or not line.endswith(b"\n"):
                        clean = False
                        break
                    hasher.update(line)
                    remaining -= len(line)
                    if not line.strip():
                        # Same rule as `scan_jsonl`: a whitespace-only terminated
                        # line is not a record, so it is neither counted nor
                        # decoded. Byte extent and digest still cover it.
                        continue
                    try:
                        decoded = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        clean = False
                        break
                    if not isinstance(decoded, dict):
                        clean = False
                        break
                    # Per-record content digests are re-verified LIVE even under
                    # a held prefix digest (the W5 ValidationMemo doctrine: a
                    # memo skips only the jsonschema pass, never content
                    # identity). Nothing binds this sidecar to a prior open, so
                    # without this a sidecar REGENERATED over corrupted bytes
                    # smuggles the corruption past the open as canonical
                    # history -- and the promise two paragraphs up, that
                    # corruption is caught by exactly the checks the historical
                    # open ran, would be false.
                    try:
                        _verify_hash(decoded)
                    except LedgerCorruptionError:
                        clean = False
                        break
                    spans.append((offset, len(line), len(spans) + 1))
                if (
                    clean
                    and remaining == 0
                    and hasher.hexdigest() == state["prefix_sha256"]
                    and len(spans) == state["line_count"]
                ):
                    accepted = True
                    byte_count = state["byte_count"]
                    line_count = state["line_count"]
                if not accepted:
                    handle.seek(0)
                    hasher = hashlib.sha256()
                    spans = []
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    raise TornTailError(
                        f"torn final line at byte {handle.tell() - len(line)} in {self.path}"
                    )
                hasher.update(line)
                byte_count += len(line)
                if not line.strip():
                    # Keep the ledger open in agreement with `scan_jsonl`: a blank
                    # line is not a record here either, so the two readers can never
                    # disagree about whether the same bytes are corrupt. It IS part
                    # of the validated byte extent and prefix digest -- exactly as
                    # the sidecar fast path above counts it -- so the append-time
                    # divergence guard and a later warm open agree with this scan.
                    continue
                line_count += 1
                _decode_validated_line(line, ledger_path=self.path, line_number=line_count)
                spans.append(
                    (offset, len(line), line_count)
                )
        return _DiskBackedRecords(self.path, spans), byte_count, line_count, hasher

    def _register_record(self, index: int, record: Mapping[str, Any]) -> None:
        """Index one persisted record for O(1) append resolution.

        Parameters
        ----------
        index:
            Position of the record in ledger order.
        record:
            Persisted record.

        Raises
        ------
        LedgerCorruptionError
            If the record's sequence does not strictly increase. One writer
            can never produce a duplicate or regressed sequence (appends are
            always ``max + 1``), so the state is proof of a dual-writer
            interleave or external tampering and must never read back clean.
        """

        self._identity_index.setdefault(_identity_key(record), index)
        stable_ids = {record.get("stable_id")} - {None}
        items = record.get("items")
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            stable_ids.update(
                item.get("stable_id")
                for item in items
                if isinstance(item, Mapping) and item.get("stable_id") is not None
            )
        for stable_id in stable_ids:
            self._stable_id_index.setdefault(stable_id, []).append(index)
        for field, values in self._field_indices.items():
            value = record.get(field)
            if value is not None:
                values.setdefault(value, []).append(index)
        value = record.get(self._sequence_field)
        if isinstance(value, int) and not isinstance(value, bool):
            if value <= self._max_sequence:
                raise LedgerCorruptionError(
                    f"non-increasing {self._sequence_field} {value} after "
                    f"{self._max_sequence} in {self.path}: impossible for a "
                    "single writer (dual-writer interleave or tampering)"
                )
            self._max_sequence = value

    def __enter__(self) -> "JsonlLedger":
        """Return this locked writer.

        Returns
        -------
        JsonlLedger
            This writer.
        """

        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release the exclusive writer lock.

        Parameters
        ----------
        exc_type, exc_value, traceback:
            Context-manager exception state.
        """

        self.close()

    def __iter__(self) -> Iterator[JsonObject]:
        """Iterate over defensive copies of persisted records.

        Returns
        -------
        Iterator[dict[str, Any]]
            Records in ledger order.
        """

        return (deepcopy(record) for record in self._records)

    def close(self) -> None:
        """Persist the advisory open-state, then release the lock idempotently.

        A failed open (``_open_complete`` never set) attests nothing, and a
        writer whose lock file was deleted or replaced no longer owns the
        sidecar and must not clobber the current owner's attestation.
        """

        if getattr(self, "_closed", False):
            return
        if (
            getattr(self, "_open_complete", False)
            and getattr(self, "_hasher", None) is not None
            and _lock_identity_holds(self._lock_handle, self._lock_path)
        ):
            _store_open_state(
                self.path,
                self.schema_version,
                byte_count=self._byte_count,
                line_count=self._line_count,
                prefix_sha256=self._hasher.hexdigest(),
            )
        data_lock_handle = getattr(self, "_data_lock_handle", None)
        if data_lock_handle is not None:
            fcntl.flock(data_lock_handle.fileno(), fcntl.LOCK_UN)
            data_lock_handle.close()
            self._data_lock_handle = None
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()
        self._closed = True

    def _acquire_data_lock(self) -> BinaryIO:
        """Return an exclusive lock on the current canonical ledger data inode.

        Returns
        -------
        BinaryIO
            Persistent handle whose inode lock survives side-lock-file deletion
            or replacement -- the second, independent writer-exclusion domain.

        Raises
        ------
        SingleWriterError
            If another writer already owns the data inode, or the canonical
            path was removed or replaced while this writer was live.
        """

        held = getattr(self, "_data_lock_handle", None)
        if held is not None:
            try:
                path_stat = self.path.stat()
            except FileNotFoundError as exc:
                raise SingleWriterError(
                    f"live ledger path was removed while writer held it: {self.path}"
                ) from exc
            handle_stat = os.fstat(held.fileno())
            if (path_stat.st_dev, path_stat.st_ino) != (
                handle_stat.st_dev,
                handle_stat.st_ino,
            ):
                raise SingleWriterError(f"live ledger path inode was replaced: {self.path}")
            return held
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise SingleWriterError(
                    f"ledger data inode already has a writer: {self.path}"
                ) from exc
            raise
        try:
            # Establish the directory entry's durability BEFORE any record can
            # commit. A directory-sync failure after the data barrier creates
            # an ambiguous "present but reported failed" append. Paying this
            # metadata barrier once per writer open removes that window while
            # keeping per-row acknowledgement on the data barrier alone.
            _fsync_directory(self.path.parent)
        except BaseException:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            raise
        self._data_lock_handle = handle
        return handle

    @property
    def records(self) -> Sequence[JsonObject]:
        """Return defensive copies of all records.

        Returns
        -------
        Sequence[dict[str, Any]]
            Persisted records in ledger order.
        """

        return tuple(deepcopy(record) for record in self._records)

    @property
    def resident_record_payload_count(self) -> int:
        """Return decoded history payloads retained by the writer.

        Returns
        -------
        int
            Zero or one; canonical history is represented by byte spans.
        """

        return self._records.resident_payload_count

    @property
    def record_count(self) -> int:
        """Return the append generation without copying ledger payloads.

        Returns
        -------
        int
            Number of persisted records.
        """

        return len(self._records)

    def iter_records(self) -> Iterator[JsonObject]:
        """Iterate persisted records WITHOUT defensive copies.

        For read-only internal consumers on hot paths: the whole-ledger
        deepcopy behind :attr:`records` is O(history) per access and was a
        measured superlinear contributor (risk_r5). Callers must never
        mutate the yielded records.

        Returns
        -------
        Iterator[dict[str, Any]]
            Live records in ledger order.
        """

        return iter(self._records)

    def records_since(self, index: int) -> tuple[JsonObject, ...]:
        """Return defensive copies of the exact appended suffix.

        Parameters
        ----------
        index:
            Append generation already consumed by the caller.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Records appended after ``index``, in ledger order. Copying only
            the suffix keeps incremental consumers O(delta) instead of
            re-copying the whole history per read.
        """

        return tuple(deepcopy(record) for record in self._records[index:])

    def records_for_stable_id(self, stable_id: str) -> tuple[JsonObject, ...]:
        """Return only rows indexed to one stable model identity.

        Parameters
        ----------
        stable_id:
            Exact model identity.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Matching rows in ledger order.
        """

        return tuple(
            deepcopy(self._records[index])
            for index in self._stable_id_index.get(stable_id, ())
        )

    def records_for_field(self, field: str, value: Any) -> tuple[JsonObject, ...]:
        """Return rows selected through one maintained immutable-field index.

        Parameters
        ----------
        field, value:
            Indexed field name and exact value.

        Returns
        -------
        tuple[dict[str, Any], ...]
            Matching rows in ledger order.

        Raises
        ------
        ValueError
            If ``field`` is not a declared ledger index.
        """

        try:
            indices = self._field_indices[field].get(value, ())
        except KeyError as exc:
            raise ValueError(f"field is not indexed: {field!r}") from exc
        return tuple(deepcopy(self._records[index]) for index in indices)

    def append(self, payload: Mapping[str, Any]) -> AppendResult:
        """Validate, idempotently append, flush, and fsync one complete line.

        Parameters
        ----------
        payload:
            Complete logical payload. Sequence and payload-hash fields may be omitted.

        Returns
        -------
        AppendResult
            Persisted record and whether a new line was written.

        Raises
        ------
        LedgerConflictError
            If an immutable identity already has a different logical payload.
        PayloadValidationError
            If the completed payload violates its schema.
        """

        if self._closed:
            raise LedgerError("cannot append through a closed ledger")
        if threading.get_ident() != self._writer_thread_id:
            raise LedgerThreadAffinityError(
                f"append from a foreign thread refused for {self.path}: the ledger "
                "is single-writer-thread; route every append through the owning "
                "writer thread"
            )
        if not _lock_identity_holds(self._lock_handle, self._lock_path) and (
            getattr(self, "_data_lock_handle", None) is None
        ):
            # flock binds an inode: a deleted or replaced lock file admits a
            # second live writer at the same path, and two writers interleave
            # sequences silently. Before the FIRST append of a fresh ledger the
            # side lock is the only exclusion domain, so its theft still makes
            # the original holder step back loudly. Once the data-inode lock is
            # held (below), a swept or swapped side lock excludes nobody the
            # data inode does not already refuse at open -- the innocent owner
            # keeps working and the INTRUDER is the one turned away (C4 c4c
            # ruling: sacrificing the owner was the worse half of both worlds).
            raise SingleWriterError(
                f"writer lock for {self.path} was deleted or replaced while "
                "held: refusing to append as a possible second live writer"
            )
        # The data-inode domain: lazily acquired on the first append of a
        # fresh ledger, verified (same dev/ino as the path) on every later
        # one.
        self._acquire_data_lock()
        record = deepcopy(dict(payload))
        # W3 write-side producer: THE durable JSONL write door. Stamping runs
        # before the idempotency comparison and before the sequence/digest
        # stamp below, so under an active shadow-rehearsal mode the marker is
        # inside ``payload_sha256``/``record_revision`` (unstrippable without
        # changing the identity) and an in-mode re-drive of an already-marked
        # row still compares equal. Semantic replay never reaches this door
        # (it runs on the in-memory replay ledgers); rehydration only reads.
        record = stamp_shadow_rehearsal(record)
        if record.get("schema_version") != self.schema_version:
            raise PayloadValidationError(
                f"expected schema_version {self.schema_version!r}, "
                f"received {record.get('schema_version')!r}"
            )
        proposed_logical = _logical_payload(record)
        proposed_key = _identity_key(record)
        if all(value is not None for value in proposed_key[1:]):
            existing = self._existing_by_identity(proposed_key)
            if existing is not None:
                if _logical_payload(existing) == proposed_logical:
                    return AppendResult(deepcopy(existing), appended=False)
                raise LedgerConflictError(
                    f"conflicting replay for immutable identity {proposed_key!r}"
                )
        if self.schema_version in _MODEL_SCHEMA_VERSIONS:
            for index in self._stable_id_index.get(record.get("stable_id"), ()):
                if _logical_payload(self._records[index]) == proposed_logical:
                    return AppendResult(deepcopy(self._records[index]), appended=False)
            expected_sequence = self._max_sequence + 1
            if record.get("record_seq", expected_sequence) != expected_sequence:
                raise LedgerConflictError(
                    f"record_seq must be next local sequence {expected_sequence}"
                )
            record.setdefault("record_seq", expected_sequence)
            record = _stamped_record(record, field="record_revision")
        else:
            expected_sequence = self._max_sequence + 1
            if record.get("ledger_seq", expected_sequence) != expected_sequence:
                raise LedgerConflictError(
                    f"ledger_seq must be next local sequence {expected_sequence}"
                )
            record.setdefault("ledger_seq", expected_sequence)
            record = _stamped_record(record, field="payload_sha256")
        key = _identity_key(record)
        logical = _logical_payload(record)
        existing = self._existing_by_identity(key)
        if existing is not None:
            if _logical_payload(existing) == logical:
                return AppendResult(deepcopy(existing), appended=False)
            raise LedgerConflictError(f"conflicting replay for immutable identity {key!r}")
        validate_payload(record, self.schema_version)
        line = canonical_json_bytes(record) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Appending blindly after a size divergence -- residue of this writer's
        # own failed write, or a foreign raw append under our lock -- would fuse
        # the new line into stray bytes, producing one complete MALFORMED line
        # that recovery is forbidden to repair. Refuse typed instead.
        try:
            disk_size = self.path.stat().st_size
        except FileNotFoundError:
            disk_size = 0
        if disk_size != self._byte_count:
            raise LedgerDivergenceError(
                f"ledger bytes diverged from writer state for {self.path}: disk "
                f"{disk_size}, writer {self._byte_count}; reopen the ledger to "
                "re-validate and recover"
            )
        with self.path.open("ab") as handle:
            try:
                written = handle.write(line)
                if written != len(line):
                    raise OSError(f"short ledger append: wrote {written} of {len(line)} bytes")
                handle.flush()
                durable_fsync(handle.fileno())
            except Exception:
                # A partial line may have reached the file. Restore the exact
                # validated extent so the writer and later appends stay sound;
                # nothing durable is lost, because this record was never
                # acknowledged. Truncation RELEASES space, so it succeeds even
                # under the ENOSPC that most plausibly caused the failure; if it
                # fails anyway, the divergence check above keeps every later
                # append refusing typed rather than fusing lines.
                try:
                    handle.truncate(self._byte_count)
                    handle.flush()
                    durable_fsync(handle.fileno())
                except OSError:
                    pass
                raise
        self._records.append_span(
            offset=self._byte_count,
            length=len(line),
            line_number=self._line_count + 1,
            record=record,
        )
        self._register_record(len(self._records) - 1, record)
        self._hasher.update(line)
        self._byte_count += len(line)
        self._line_count += 1
        return AppendResult(deepcopy(record), appended=True)

    def _existing_by_identity(self, key: tuple[Any, ...]) -> Optional[JsonObject]:
        """Return the first persisted record holding one immutable identity.

        Parameters
        ----------
        key:
            Type-tagged identity key.

        Returns
        -------
        dict[str, Any] | None
            Earliest matching record, exactly as the historical linear scan
            selected it, or ``None``.
        """

        index = self._identity_index.get(key)
        return self._records[index] if index is not None else None
