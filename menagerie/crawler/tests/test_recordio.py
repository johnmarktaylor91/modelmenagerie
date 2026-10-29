"""Append-only JSONL durability, recovery, and idempotency tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION,
    ATTEMPT_SCHEMA_VERSION_V3,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)
from menagerie.crawler.recordio import (
    AttemptSlotResolutionError,
    JsonlLedger,
    LedgerCorruptionError,
    LedgerConflictError,
    LedgerDivergenceError,
    SingleWriterError,
    deterministic_attempt_id,
    recover_torn_tail,
    resolve_attempt_slot,
    resolve_attempt_slot_state,
    scan_jsonl,
)
from menagerie.crawler.schema import PayloadValidationError
from menagerie.crawler.tests.conftest import make_attempt, make_operational_event

#: Sub-second load-bearing coverage promoted into the mandatory smoke tier (T5-7).
pytestmark = pytest.mark.smoke


def test_append_is_persisted_and_idempotent(tmp_path: Path) -> None:
    """A replay returns the original fsynced fact without another line.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    payload = make_attempt()
    payload.pop("ledger_seq")
    payload.pop("payload_sha256")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        first = ledger.append(payload)
        replay = ledger.append(payload)
        assert first.appended
        assert not replay.appended
        assert first.record == replay.record
    assert len(scan_jsonl(path)) == 1


def test_operational_event_semantic_replay_ignores_only_created_at(tmp_path: Path) -> None:
    """A same-fact operational replay keeps its first observation clock.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "operational.jsonl"
    original = make_operational_event()
    original.pop("ledger_seq")
    original.pop("payload_sha256")
    replay = deepcopy(original)
    replay["created_at"] = "2026-07-14T12:05:00Z"
    with JsonlLedger(path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        first = ledger.append(original)
        repeated = ledger.append(replay)
        changed = deepcopy(replay)
        changed["status"] = "resumed:usage-limit"
        with pytest.raises(LedgerConflictError):
            ledger.append(changed)
    assert first.appended
    assert not repeated.appended
    assert repeated.record["created_at"] == original["created_at"]
    assert len(scan_jsonl(path)) == 1


def test_wiped_ledger_parent_refuses_live_append_and_recovers_on_reopen(
    tmp_path: Path,
) -> None:
    """A wiped records dir refuses the live writer typed; a reopen recovers.

    The historical behavior silently recreated the parent and kept appending,
    masking external destruction of already acknowledged state -- and the wipe
    also took the lock file, so exclusion was gone and a second process could
    bootstrap the same ledger concurrently (rehearsed live as the C4
    double-writer). Refusal costs one reopen -- which re-materializes the
    parent, re-acquires a legitimate lock, and loses nothing, because a refused
    append was never acknowledged.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "records" / "attempts.jsonl"
    payload = make_attempt()
    payload.pop("ledger_seq")
    payload.pop("payload_sha256")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        path.with_suffix(f"{path.suffix}.lock").unlink()
        path.parent.rmdir()
        with pytest.raises(SingleWriterError, match="deleted or replaced"):
            ledger.append(payload)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as reopened:
        assert reopened.append(payload).appended
    assert path.parent.is_dir()
    assert len(scan_jsonl(path)) == 1


def test_single_writer_lock_is_exclusive(tmp_path: Path) -> None:
    """Two live canonical writers cannot own one ledger.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3):
        with pytest.raises(SingleWriterError):
            JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)


def test_torn_tail_is_evidenced_before_truncation(tmp_path: Path) -> None:
    """Recovery retains offset/hash evidence and preserves valid facts.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(make_attempt())
    with path.open("ab") as handle:
        handle.write(b'{"schema_version":"menagerie.crawler.attempt.v2"')
    evidence = recover_torn_tail(path)
    assert evidence is not None
    assert evidence.byte_count > 0
    assert evidence.evidence_path.exists()
    assert len(scan_jsonl(path)) == 1


def test_complete_malformed_line_is_never_recovered(tmp_path: Path) -> None:
    """A newline-terminated malformed record is hard corruption.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    path.write_bytes(b"not-json\n")
    with pytest.raises(LedgerCorruptionError):
        recover_torn_tail(path)


def test_v3_writer_reads_v2_history_but_appends_only_v3(tmp_path: Path) -> None:
    """Mixed immutable attempt history is readable without enabling legacy appends.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "mixed-attempts.jsonl"
    legacy = make_attempt()
    legacy["schema_version"] = ATTEMPT_SCHEMA_VERSION
    for field in (
        "capability_observation",
        "execution_read_manifest_identity",
        "raw_award_receipt",
        "raw_award_receipt_sha256",
        "parent_attestation",
        "unattested_partial",
    ):
        legacy.pop(field)
    for field in (
        "authority_epoch",
        "base_environment_generation",
        "environment_content_sha256",
        "environment_authority_id",
        "selected_interpreter_relative_path",
        "selected_interpreter_digest",
        "external_escape_records",
    ):
        legacy["environment"].pop(field)
    legacy["worker_receipt"].pop("output_value_sha256")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION) as ledger:
        ledger.append(legacy)

    current = make_attempt()
    current["schema_version"] = ATTEMPT_SCHEMA_VERSION_V3
    current["attempt_id"] = "attempt-v3"
    current["execution_read_manifest_identity"] = "sha256:" + "e" * 64
    current["raw_award_receipt"] = {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": "nonce-v3",
        "request_sha256": "sha256:" + "1" * 64,
        "stable_id": current["stable_id"],
        "work_id": current["work_id"],
        "execution_identity": current["identities"]["execution"],
        "recipe_revision": current["identities"]["recipe"],
        "code_manifest_identity": "sha256:" + "2" * 64,
        "input_identity": "sha256:" + "3" * 64,
        "requested_mode": current["mode"],
        "observation": current["worker_receipt"],
    }
    current["raw_award_receipt_sha256"] = "sha256:" + "4" * 64
    current["parent_attestation"] = {
        "attestation_version": "menagerie.crawler.parent-attestation.v2",
        "request_nonce": "nonce-v3",
        "request_sha256": "sha256:" + "1" * 64,
        "completion_line_sha256": "sha256:" + "5" * 64,
        "named_raw_award_receipt_sha256": "sha256:" + "4" * 64,
        "exit_code": 0,
        "signal": None,
        "timed_out": False,
        "rss_exceeded": False,
        "peak_rss_bytes": 1,
        "stdout_sha256": "sha256:" + "6" * 64,
        "stderr_sha256": "sha256:" + "7" * 64,
        "started_at": current["started_at"],
        "finished_at": current["finished_at"],
        "attestation_sha256": "sha256:" + "8" * 64,
    }
    current["unattested_partial"] = None
    current.pop("ledger_seq")
    current.pop("payload_sha256")

    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert [row["schema_version"] for row in ledger.records] == [ATTEMPT_SCHEMA_VERSION]
        ledger.append(current)
        with pytest.raises(PayloadValidationError):
            ledger.append(legacy)
    assert [row["schema_version"] for row in scan_jsonl(path)] == [
        ATTEMPT_SCHEMA_VERSION,
        ATTEMPT_SCHEMA_VERSION_V3,
    ]


def test_resolve_attempt_slot_reuses_one_exact_current_row() -> None:
    """The canonical ledger row satisfies its deterministic slot without replay."""

    work_id = "work-m_example"
    execution_identity = "sha256:" + "e" * 64
    attempt_id = deterministic_attempt_id(
        work_id=work_id,
        execution_identity=execution_identity,
        cold_index=0,
        mode="eval",
    )
    attempt = make_attempt(attempt_id=attempt_id, execution_identity=execution_identity)

    resolved = resolve_attempt_slot(
        (attempt,),
        work_id=work_id,
        execution_identity=execution_identity,
        cold_index=0,
        mode="eval",
    )

    assert resolved == attempt
    assert resolved is not attempt


@pytest.mark.parametrize("mutation", ["legacy", "mode", "duplicate"])
def test_resolve_attempt_slot_rejects_noncanonical_same_id(mutation: str) -> None:
    """Legacy, contradictory, and duplicate same-ID rows fail closed."""

    work_id = "work-m_example"
    execution_identity = "sha256:" + "e" * 64
    attempt_id = deterministic_attempt_id(
        work_id=work_id,
        execution_identity=execution_identity,
        cold_index=0,
        mode="eval",
    )
    attempt = make_attempt(attempt_id=attempt_id, execution_identity=execution_identity)
    records = [attempt]
    if mutation == "legacy":
        attempt["schema_version"] = ATTEMPT_SCHEMA_VERSION
    elif mutation == "mode":
        attempt["mode"] = "train"
    else:
        records.append(dict(attempt))

    with pytest.raises(AttemptSlotResolutionError):
        resolve_attempt_slot(
            records,
            work_id=work_id,
            execution_identity=execution_identity,
            cold_index=0,
            mode="eval",
        )


def test_resolve_attempt_slot_retains_actionable_schema_error() -> None:
    """Attempt-slot rejection preserves the specific malformed constraint."""

    work_id = "work-m_example"
    execution_identity = "sha256:" + "e" * 64
    attempt_id = deterministic_attempt_id(
        work_id=work_id,
        execution_identity=execution_identity,
        cold_index=0,
        mode="eval",
    )
    attempt = make_attempt(attempt_id=attempt_id, execution_identity=execution_identity)
    attempt["undeclared_observation"] = "No contract home."

    with pytest.raises(AttemptSlotResolutionError) as caught:
        resolve_attempt_slot(
            (attempt,),
            work_id=work_id,
            execution_identity=execution_identity,
            cold_index=0,
            mode="eval",
        )

    diagnostic = str(caught.value)
    assert "undeclared_observation" in diagnostic
    assert "additionalProperties" in diagnostic


def test_attempt_timestamp_change_remains_an_immutable_conflict(tmp_path: Path) -> None:
    """M-02 retains timestamp fields in logical replay comparisons."""

    path = tmp_path / "attempts.jsonl"
    attempt = make_attempt()
    attempt.pop("ledger_seq")
    attempt.pop("payload_sha256")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(attempt)
        attempt["finished_at"] = "2026-07-16T12:00:01Z"
        with pytest.raises(LedgerConflictError):
            ledger.append(attempt)


def _append_one_attempt(path: Path) -> None:
    """Persist one canonical attempt row through the production writer.

    Parameters
    ----------
    path:
        Ledger path receiving the row.
    """

    attempt = make_attempt()
    attempt.pop("ledger_seq")
    attempt.pop("payload_sha256")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(attempt)


def test_a_tampered_on_disk_payload_digest_refuses_loudly_on_read(tmp_path: Path) -> None:
    """One flipped byte of a persisted ``payload_sha256`` refuses the whole scan.

    A6: the digest is a live tripwire re-verified on every read, never a
    decoration -- an on-disk edit of the digest itself must surface as loud
    ledger corruption naming the digest field, not as a readable row.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    _append_one_attempt(path)
    persisted = path.read_text(encoding="utf-8")
    digest = json.loads(persisted)["payload_sha256"].removeprefix("sha256:")
    flipped_character = "0" if digest[0] != "0" else "f"
    tampered = persisted.replace(digest, flipped_character + digest[1:], 1)
    assert tampered != persisted
    path.write_text(tampered, encoding="utf-8")

    with pytest.raises(LedgerCorruptionError, match="payload_sha256"):
        scan_jsonl(path)


def test_a_tampered_on_disk_payload_byte_refuses_loudly_on_read(tmp_path: Path) -> None:
    """One flipped content byte under an intact digest is the same corruption.

    Inverse direction of the digest-flip: the persisted content no longer
    matches its digest, so the scan refuses identically instead of returning
    silently altered facts.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    path = tmp_path / "attempts.jsonl"
    _append_one_attempt(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    # A schema-valid mutation, so ONLY the digest tripwire can catch it.
    original = persisted["finished_at"]
    persisted["finished_at"] = original.replace("Z", "").rsplit(":", 1)[0] + ":59Z"
    assert persisted["finished_at"] != original
    path.write_text(
        json.dumps(persisted, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(LedgerCorruptionError, match="payload_sha256"):
        scan_jsonl(path)


def _fresh_attempt(attempt_id: str) -> dict:
    """Return one append-ready attempt payload.

    Parameters
    ----------
    attempt_id:
        Immutable attempt identity.
    """

    attempt = make_attempt(attempt_id=attempt_id)
    attempt.pop("ledger_seq")
    attempt.pop("payload_sha256")
    return attempt


def test_failed_append_write_restores_tail_and_retry_stays_sound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial write (ENOSPC mid-line) restores the exact validated extent.

    Rehearsed live pre-fix: the half-written line stayed on disk, the RETRY
    append fused into it and reported success, and the shard was permanently
    corrupt (a complete malformed line recovery must never repair) while the
    writer believed both rows landed -- silent loss until the next open.
    """

    path = tmp_path / "attempts.jsonl"
    ledger = JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)
    ledger.append(_fresh_attempt("attempt-0"))
    size_before = path.stat().st_size

    real_open = Path.open

    class _HalfWrite:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def write(self, data: bytes) -> None:
            self._handle.write(data[: len(data) // 2])
            self._handle.flush()
            raise OSError(28, "No space left on device")

        def __getattr__(self, name: str) -> object:
            return getattr(self._handle, name)

        def __enter__(self) -> "_HalfWrite":
            return self

        def __exit__(self, *exc: object) -> None:
            self._handle.__exit__(*exc)

    armed = {"on": True}

    def half_open(self: Path, mode: str = "r", *args: object, **kwargs: object) -> object:
        handle = real_open(self, mode, *args, **kwargs)
        if armed["on"] and self == path and mode == "ab":
            armed["on"] = False
            return _HalfWrite(handle)
        return handle

    monkeypatch.setattr(Path, "open", half_open)
    with pytest.raises(OSError, match="No space left"):
        ledger.append(_fresh_attempt("attempt-1"))
    monkeypatch.setattr(Path, "open", real_open)

    assert path.stat().st_size == size_before, "failed write left torn bytes behind"
    retry = ledger.append(_fresh_attempt("attempt-1"))
    assert retry.appended
    assert retry.record["ledger_seq"] == 2
    ledger.close()
    assert [row["attempt_id"] for row in scan_jsonl(path)] == ["attempt-0", "attempt-1"]


def test_append_refuses_typed_on_diverged_ledger_bytes(tmp_path: Path) -> None:
    """Stray bytes under a held writer refuse the append; nothing ever fuses."""

    path = tmp_path / "attempts.jsonl"
    ledger = JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)
    ledger.append(_fresh_attempt("attempt-0"))
    with path.open("ab") as handle:
        handle.write(b'{"foreign": ')
    with pytest.raises(LedgerDivergenceError, match="diverged"):
        ledger.append(_fresh_attempt("attempt-1"))
    ledger.close()
    # the refused append wrote nothing: the stray tail is intact for the next
    # open's ordinary evidence-and-truncate recovery
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as reopened:
        assert reopened.recovery_evidence is not None
        assert reopened.record_count == 1


def test_lock_file_deletion_never_admits_a_double_writer(tmp_path: Path) -> None:
    """A deleted side-lock file admits nobody and sacrifices nobody.

    Rehearsed live pre-fix: deleting the ``.lock`` admitted a second-process
    co-writer that minted a duplicate sequence -- double-writer interleaving.
    Under the dual-domain contract the DATA-inode flock still excludes every
    second open, so the elder writer keeps appending and a fresh open at the
    same path is the one refused. Before the FIRST append of a fresh ledger
    (no data inode yet) side-lock theft still refuses the owner typed.
    """

    path = tmp_path / "attempts.jsonl"
    ledger = JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)
    ledger.append(_fresh_attempt("attempt-0"))
    path.with_suffix(".jsonl.lock").unlink()
    with pytest.raises(SingleWriterError):
        JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)
    ledger.append(_fresh_attempt("attempt-1"))
    ledger.close()
    assert len(scan_jsonl(path)) == 2

    fresh = tmp_path / "fresh.jsonl"
    empty = JsonlLedger(fresh, ATTEMPT_SCHEMA_VERSION_V3)
    fresh.with_suffix(".jsonl.lock").unlink()
    with pytest.raises(SingleWriterError, match="deleted or replaced"):
        empty.append(_fresh_attempt("attempt-0"))
    empty.close()


def test_torn_evidence_ledger_heals_before_new_evidence(tmp_path: Path) -> None:
    """A crash mid-evidence-write never leaves the evidence ledger unscannable.

    Rehearsed live pre-fix: recovery appended its new evidence directly after
    a torn evidence line, fusing them into one complete malformed line -- and
    the checkpoint's append-only proof scans the evidence ledger strictly, so
    a later legitimate recovery read as unevidenced.
    """

    path = tmp_path / "attempts.jsonl"
    _append_one_attempt(path)
    good_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(b'{"torn": ')
    evidence_path = path.with_suffix(".jsonl.recovery.jsonl")
    evidence_path.write_bytes(b'{"ledger_path": "half-written evid')
    evidence = recover_torn_tail(path)
    assert evidence is not None
    assert path.stat().st_size == good_size
    rows = scan_jsonl(evidence_path, validate=False)
    assert len(rows) == 1
    assert rows[0]["byte_offset"] == good_size


# ---------------------------------------------------------------------------
# DX-10 / RE-11: the owner-blind slot fix (typed retry generations)
# ---------------------------------------------------------------------------


def _machinery_failed_attempt(attempt_id: str, execution_identity: str) -> dict[str, Any]:
    """Build a schema-valid machinery-owned failed row for one exact slot."""

    attempt = make_attempt(attempt_id=attempt_id, execution_identity=execution_identity)
    attempt["result"] = "failed"
    attempt["error"] = {
        "stage": "runner",
        "reason_code": "internal-error",
        "exception_type": "builtins.RuntimeError",
        "message": "synthetic harness defect",
        "traceback": None,
        "no_traceback_reason": "synthetic fixture",
        "native_crash": False,
        "root_cause_fingerprint": "sha256:" + "d" * 64,
        "details": {"owner": "machinery"},
    }
    return attempt


def _non_model(error: dict[str, Any]) -> bool:
    """The DX-10 predicate the driver supplies (owner in the NEVER set)."""

    from menagerie.crawler.driver_receipts import (
        NEVER_MODEL_TERMINAL_OWNERS,
        _failure_owner,
    )

    return _failure_owner(error) in NEVER_MODEL_TERMINAL_OWNERS


@pytest.mark.smoke
def test_slot_generation_zero_identity_is_byte_stable() -> None:
    """Generation zero keeps the historical identity spelling exactly."""

    kwargs = dict(
        work_id="work-m_example",
        execution_identity="sha256:" + "e" * 64,
        cold_index=0,
        mode="eval",
    )
    assert deterministic_attempt_id(**kwargs) == deterministic_attempt_id(
        **kwargs, retry_generation=0
    )
    assert deterministic_attempt_id(**kwargs, retry_generation=1) != deterministic_attempt_id(
        **kwargs
    )
    with pytest.raises(ValueError):
        deterministic_attempt_id(**kwargs, retry_generation=-1)


@pytest.mark.smoke
def test_non_model_failure_row_never_satisfies_the_slot() -> None:
    """A machinery-owned failure is consumed weather; the slot re-runs typed."""

    work_id = "work-m_example"
    execution_identity = "sha256:" + "e" * 64
    slot = dict(
        work_id=work_id, execution_identity=execution_identity, cold_index=0, mode="eval"
    )
    weather = _machinery_failed_attempt(
        deterministic_attempt_id(**slot), execution_identity
    )

    state = resolve_attempt_slot_state((weather,), **slot, non_model_failure=_non_model)
    assert state.record is None
    assert state.retry_generation == 1
    assert state.consumed_non_model_rows == (weather["attempt_id"],)

    # The retry generation's success row then satisfies the slot.
    retry_id = deterministic_attempt_id(**slot, retry_generation=1)
    success = make_attempt(attempt_id=retry_id, execution_identity=execution_identity)
    success["retries"] = {**success["retries"], "slot_retry_generation": 1}
    state = resolve_attempt_slot_state(
        (weather, success), **slot, non_model_failure=_non_model
    )
    assert state.record is not None
    assert state.record["attempt_id"] == retry_id
    assert state.retry_generation == 1
    assert state.consumed_non_model_rows == (weather["attempt_id"],)


@pytest.mark.smoke
def test_model_owned_failure_row_still_satisfies_the_slot() -> None:
    """RED direction two: a model-owned failure remains a verdict candidate."""

    work_id = "work-m_example"
    execution_identity = "sha256:" + "e" * 64
    slot = dict(
        work_id=work_id, execution_identity=execution_identity, cold_index=0, mode="eval"
    )
    failed = _machinery_failed_attempt(deterministic_attempt_id(**slot), execution_identity)
    failed["error"]["details"] = {}
    failed["error"]["stage"] = "forward"
    failed["error"]["reason_code"] = "mode-run"

    state = resolve_attempt_slot_state((failed,), **slot, non_model_failure=_non_model)
    assert state.record is not None
    assert state.retry_generation == 0
    assert state.consumed_non_model_rows == ()
    # The legacy resolver keeps its historical single-generation semantics.
    assert resolve_attempt_slot((failed,), **slot) is not None


@pytest.mark.smoke
def test_retry_generation_row_must_carry_its_bound_generation() -> None:
    """A generation-1 identity whose row omits the carried counter refuses."""

    work_id = "work-m_example"
    execution_identity = "sha256:" + "e" * 64
    slot = dict(
        work_id=work_id, execution_identity=execution_identity, cold_index=0, mode="eval"
    )
    weather = _machinery_failed_attempt(
        deterministic_attempt_id(**slot), execution_identity
    )
    forged = make_attempt(
        attempt_id=deterministic_attempt_id(**slot, retry_generation=1),
        execution_identity=execution_identity,
    )
    with pytest.raises(AttemptSlotResolutionError, match="contradicts its exact slot"):
        resolve_attempt_slot_state(
            (weather, forged), **slot, non_model_failure=_non_model
        )
