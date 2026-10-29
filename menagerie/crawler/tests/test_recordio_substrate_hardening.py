"""Substrate hardening pins: writer-lock integrity, recovery/evidence safety.

C4-B rehearsal findings, each pinned by a test that fails without its fix:

1. Deleting or replacing the writer lock file while a ``JsonlLedger`` is live
   silently admitted a SECOND live writer; the interleaved ledger then read
   back clean with duplicate sequence values. The lock now verifies its inode
   identity at acquisition and before every append, and the open scan refuses
   any non-increasing sequence (a state one writer can never produce).
2. ``recover_torn_tail`` appended evidence after a torn evidence tail (a
   crash mid-evidence-write), producing a malformed COMPLETE line that
   permanently wedged the checkpoint truncation validator for that shard.
3. ``read_with_tail_recovery``'s self-locking path locked only the ledger
   inode while ``JsonlLedger`` writers lock ``<ledger>.lock``, so recovery
   could truncate a live writer's in-flight append (silent loss of an
   acknowledged record). Recovery now also takes the writer lock domain.
4. A refused ``JsonlLedger.__init__`` (unreadable schema version) published
   an open-state sidecar through ``close()``; a failed open publishes nothing.
5. Ledger runtime lock files (``*.jsonl.lock``) were swept into PUBLIC
   checkpoint candidates by the ``.lock`` allowlist suffix.
6. Torn-tail recovery read the whole ledger and wrote a whole-prefix copy per
   recovery; the default path now streams (O(tail) memory, no temp copy).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from menagerie.crawler.checkpoint import (
    NonAllowlistedPath,
    _validate_allowlisted_path,
)
from menagerie.crawler.constants import (
    ATTEMPT_SCHEMA_VERSION_V3,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)
from menagerie.crawler.recordio import (
    JsonlLedger,
    LedgerCorruptionError,
    SingleWriterError,
    open_state_path,
    read_with_tail_recovery,
    recover_torn_tail,
    scan_jsonl,
)
from menagerie.crawler.tests.conftest import make_attempt


def _payload(index: int) -> dict[str, Any]:
    payload = make_attempt(attempt_id=f"attempt-{index}")
    payload.pop("ledger_seq", None)
    payload.pop("payload_sha256", None)
    return payload


def _write_ledger(path: Path, count: int) -> None:
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        for index in range(count):
            ledger.append(_payload(index))


@pytest.mark.smoke
def test_lock_deleted_while_held_keeps_owner_and_refuses_intruders(
    tmp_path: Path,
) -> None:
    """A stolen side lock never admits a second writer NOR kills the owner.

    The adjudicated dual-domain contract (C4 c4c ruling): the side lock is a
    deletable name, but the data-inode flock rides the canonical bytes. With
    it held, sweeping the side lock away excludes nobody -- a second open is
    refused at the DATA inode -- so the innocent owner keeps appending instead
    of being sacrificed for an attack the substrate already repelled. Only
    before the first append of a FRESH ledger (no data inode yet) does side-
    lock theft still refuse the owner's append.
    """

    path = tmp_path / "attempts.jsonl"
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_payload(0))
        path.with_suffix(f"{path.suffix}.lock").unlink()
        with pytest.raises(SingleWriterError):
            JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)
        ledger.append(_payload(1))
    assert len(scan_jsonl(path)) == 2

    fresh = tmp_path / "fresh.jsonl"
    with JsonlLedger(fresh, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        # No data inode exists yet, so the side lock is the only domain and
        # its theft still makes the owner step back loudly.
        fresh.with_suffix(f"{fresh.suffix}.lock").unlink()
        with pytest.raises(SingleWriterError):
            ledger.append(_payload(0))
    assert not fresh.exists()


@pytest.mark.smoke
def test_lock_replaced_while_held_keeps_owner_excluding_intruders(
    tmp_path: Path,
) -> None:
    """A replaced side-lock inode neither kills the owner nor admits a rival.

    Dual-domain contract (C4 c4c ruling): the data-inode flock still excludes
    every second writer, so the owner keeps appending and a new open is
    refused at the data inode.
    """

    path = tmp_path / "attempts.jsonl"
    lock = path.with_suffix(f"{path.suffix}.lock")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_payload(0))
        lock.unlink()
        lock.write_bytes(b"")
        with pytest.raises(SingleWriterError):
            JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3)
        ledger.append(_payload(1))
    assert len(scan_jsonl(path)) == 2


def test_second_process_refused_after_lock_deletion_and_first_append(
    tmp_path: Path,
) -> None:
    """Exactly one writer remains after side-lock theft: the ORIGINAL one.

    The second process acquires the RECREATED side-lock file (it cannot see
    the deleted inode) but is refused at the DATA inode the live owner still
    flocks -- so the intruder is excluded at its open and the innocent
    owner's next append keeps working (C4 c4c ruling: refusing the owner for
    an attack the substrate already repelled was the worse half of both
    outcomes).
    """

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 1)
    lock = path.with_suffix(f"{path.suffix}.lock")
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as first:
        lock.unlink()
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    f"""
                    import sys
                    sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
                    from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3
                    from menagerie.crawler.recordio import JsonlLedger, SingleWriterError
                    try:
                        JsonlLedger({str(path)!r}, ATTEMPT_SCHEMA_VERSION_V3)
                    except SingleWriterError:
                        print("REFUSED")
                    else:
                        print("ACQUIRED")
                    """
                ),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "REFUSED" in child.stdout, child.stdout + child.stderr
        first.append(_payload(9))
    assert len(scan_jsonl(path)) == 2


@pytest.mark.smoke
def test_duplicate_sequence_ledger_refuses_on_open(tmp_path: Path) -> None:
    """A dual-writer interleave (duplicate sequence) can never read clean."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 2)
    base = path.read_bytes()
    fork = tmp_path / "fork.jsonl"
    fork.write_bytes(base)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_payload(100))
    with JsonlLedger(fork, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_payload(200))
    interleaved = path.read_bytes() + fork.read_bytes()[len(base) :]
    victim = tmp_path / "victim.jsonl"
    victim.write_bytes(interleaved)
    with pytest.raises(LedgerCorruptionError):
        JsonlLedger(victim, ATTEMPT_SCHEMA_VERSION_V3)


@pytest.mark.smoke
def test_regressed_sequence_ledger_refuses_on_open(tmp_path: Path) -> None:
    """A sequence that goes backwards is equally impossible for one writer."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 3)
    lines = path.read_bytes().splitlines(keepends=True)
    victim = tmp_path / "victim.jsonl"
    victim.write_bytes(b"".join([lines[0], lines[2], lines[1]]))
    with pytest.raises(LedgerCorruptionError):
        JsonlLedger(victim, ATTEMPT_SCHEMA_VERSION_V3)


@pytest.mark.smoke
def test_duplicate_sequence_refuses_on_warm_reopen_too(tmp_path: Path) -> None:
    """The digest-held streaming path enforces the same sequence tripwire."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 2)
    base = path.read_bytes()
    fork = tmp_path / "fork.jsonl"
    fork.write_bytes(base)
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_payload(100))
    with JsonlLedger(fork, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_payload(200))
    victim = tmp_path / "victim.jsonl"
    victim.write_bytes(path.read_bytes() + fork.read_bytes()[len(base) :])
    # Plant a sidecar attesting the whole interleaved file, as a crashed pair
    # of dual writers could have left behind.
    import hashlib

    honest = json.loads(open_state_path(path).read_bytes())
    honest["byte_count"] = len(victim.read_bytes())
    honest["line_count"] = 4
    honest["prefix_sha256"] = hashlib.sha256(victim.read_bytes()).hexdigest()
    open_state_path(victim).write_bytes(
        json.dumps(honest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    with pytest.raises(LedgerCorruptionError):
        JsonlLedger(victim, ATTEMPT_SCHEMA_VERSION_V3)


@pytest.mark.smoke
def test_failed_open_publishes_no_sidecar(tmp_path: Path) -> None:
    """An open refused by the schema-version check attests nothing."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 2)
    open_state_path(path).unlink()
    with pytest.raises(LedgerCorruptionError):
        JsonlLedger(path, OPERATIONAL_EVENT_SCHEMA_VERSION)
    assert not open_state_path(path).exists()


@pytest.mark.smoke
def test_failed_open_preserves_prior_writer_sidecar(tmp_path: Path) -> None:
    """A refused foreign-writer open cannot clobber the owner's attestation."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 2)
    before = open_state_path(path).read_bytes()
    with pytest.raises(LedgerCorruptionError):
        JsonlLedger(path, OPERATIONAL_EVENT_SCHEMA_VERSION)
    assert open_state_path(path).read_bytes() == before


@pytest.mark.smoke
def test_unlocked_recovery_refuses_under_live_writer(tmp_path: Path) -> None:
    """Self-locking tail recovery cannot truncate a live writer's append."""

    path = tmp_path / "attempts.jsonl"
    with JsonlLedger(path, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(_payload(0))
        # A reader observes the ledger exactly mid-append: bytes written,
        # fsync (and acknowledgement) still pending.
        with path.open("ab") as handle:
            handle.write(b'{"schema_version":"menagerie.crawler.attempt.v3","x":')
        before = path.read_bytes()
        with pytest.raises(SingleWriterError):
            read_with_tail_recovery(path, prefix_check=scan_jsonl)
        assert path.read_bytes() == before
        with pytest.raises(SingleWriterError):
            recover_torn_tail(path)
        assert path.read_bytes() == before


@pytest.mark.smoke
def test_unlocked_recovery_still_works_without_a_writer(tmp_path: Path) -> None:
    """Valve-shape ledgers (no writer lock file) recover exactly as before."""

    path = tmp_path / "valve.jsonl"
    row = {"kind": "park", "row": 1}
    path.write_bytes(json.dumps(row).encode() + b"\n" + b'{"torn":')
    payload = read_with_tail_recovery(path, prefix_check=lambda p: json.loads(p.read_bytes()))
    assert payload.endswith(b"\n")
    assert json.loads(payload) == row
    assert path.with_suffix(f"{path.suffix}.recovery.jsonl").exists()


@pytest.mark.smoke
def test_recovery_after_writer_closes_succeeds(tmp_path: Path) -> None:
    """The writer-lock guard yields as soon as the writer is gone."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 1)
    with path.open("ab") as handle:
        handle.write(b'{"torn":')
    evidence = recover_torn_tail(path)
    assert evidence is not None
    assert len(scan_jsonl(path)) == 1


@pytest.mark.smoke
def test_torn_evidence_tail_never_wedges_the_evidence_ledger(tmp_path: Path) -> None:
    """A crash mid-evidence-write must not leave a permanently unscannable
    evidence ledger after the next recovery (checkpoint truncation validation
    scans it on every shrink)."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 2)
    clean = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"first-tear":')
    evidence_path = path.with_suffix(f"{path.suffix}.recovery.jsonl")
    first = recover_torn_tail(path)
    assert first is not None
    complete_evidence = evidence_path.read_bytes()
    # Crash mid-evidence-write: a torn evidence tail, ledger still torn.
    evidence_path.write_bytes(complete_evidence[: len(complete_evidence) // 2])
    path.write_bytes(clean + b'{"second-tear":')
    second = recover_torn_tail(path)
    assert second is not None
    rows = scan_jsonl(evidence_path, validate=False)
    assert any(
        row.get("byte_offset") == second.byte_offset
        and row.get("tail_sha256") == second.tail_sha256
        for row in rows
    )
    assert len(scan_jsonl(path)) == 2


@pytest.mark.smoke
def test_complete_evidence_rows_survive_later_recoveries(tmp_path: Path) -> None:
    """Torn-evidence truncation removes ONLY the unacknowledged torn tail."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 2)
    clean = path.read_bytes()
    evidence_path = path.with_suffix(f"{path.suffix}.recovery.jsonl")
    with path.open("ab") as handle:
        handle.write(b'{"first-tear":')
    recover_torn_tail(path)
    first_rows = scan_jsonl(evidence_path, validate=False)
    path.write_bytes(clean + b'{"second-tear":')
    recover_torn_tail(path)
    rows = scan_jsonl(evidence_path, validate=False)
    assert rows[: len(first_rows)] == first_rows
    assert len(rows) == len(first_rows) + 1


@pytest.mark.smoke
def test_ledger_lock_files_are_not_checkpoint_candidates(tmp_path: Path) -> None:
    """Runtime writer locks under records/ never reach the public sweep."""

    rel = PurePosixPath("menagerie/crawler/records/models/current-shard.jsonl.lock")
    target = tmp_path / Path(rel.as_posix())
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    with pytest.raises(NonAllowlistedPath):
        _validate_allowlisted_path(tmp_path, Path(rel.as_posix()), require_exists=True)


@pytest.mark.smoke
def test_environment_lock_files_stay_checkpoint_candidates(tmp_path: Path) -> None:
    """The conda lock family (single .lock suffix) remains allowlisted."""

    rel = PurePosixPath("menagerie/crawler/envs/core/locks/osx-arm64.lock")
    target = tmp_path / Path(rel.as_posix())
    target.parent.mkdir(parents=True)
    target.write_bytes(b"platform: osx-arm64\n")
    _validate_allowlisted_path(tmp_path, Path(rel.as_posix()), require_exists=True)


@pytest.mark.smoke
def test_recovery_evidence_ledgers_stay_checkpoint_candidates(tmp_path: Path) -> None:
    """Recovery evidence is disclosure and must keep shipping."""

    rel = PurePosixPath("menagerie/crawler/records/models/current-shard.jsonl.recovery.jsonl")
    target = tmp_path / Path(rel.as_posix())
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")
    _validate_allowlisted_path(tmp_path, Path(rel.as_posix()), require_exists=True)


@pytest.mark.smoke
def test_torn_recovery_streams_without_whole_file_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default torn-tail recovery is O(tail), not O(ledger): no whole-file
    ``read_bytes`` and no whole-prefix temp copy."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 3)
    with path.open("ab") as handle:
        handle.write(b'{"torn":')
    real_read_bytes = Path.read_bytes
    whole_reads = [0]

    def counting_read_bytes(self: Path) -> bytes:
        if self == path:
            whole_reads[0] += 1
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)
    created: list[Path] = []
    real_open = Path.open

    def watching_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any):
        if "w" in mode and self.name.endswith(".recovery-scan"):
            created.append(self)
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", watching_open)
    evidence = recover_torn_tail(path)
    assert evidence is not None
    assert whole_reads[0] == 0
    assert created == []
    assert len(scan_jsonl(path)) == 3


@pytest.mark.smoke
def test_custom_prefix_check_still_validates_before_truncation(tmp_path: Path) -> None:
    """A caller-supplied strict decoder still runs BEFORE any truncation and
    its typed refusal still protects a malformed complete line."""

    path = tmp_path / "valve.jsonl"
    path.write_bytes(b"not-json-complete-line\n" + b'{"torn":')

    class ValveError(RuntimeError):
        pass

    def strict(candidate: Path) -> None:
        for line in candidate.read_bytes().splitlines():
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValveError(str(exc)) from exc

    before = path.read_bytes()
    with pytest.raises(ValveError):
        recover_torn_tail(path, prefix_check=strict)
    assert path.read_bytes() == before


@pytest.mark.smoke
def test_degenerate_ledgers_open_or_refuse_loudly(tmp_path: Path) -> None:
    """L7 cells: newline-only opens EMPTY and stays appendable; torn-first-line
    recovers to empty with evidence.

    Blank TERMINATED lines are not records and not corruption -- the
    partitioner's ``empty-deferred.jsonl`` and an empty snapshot
    ``sources/deferred.jsonl`` are literally one newline byte, and every
    reader in the tree (``scan_jsonl``, the checkpoint validator) agrees.
    They ARE validated prefix bytes, so the append-time divergence guard
    must accept the file extent and the first append must land after them.
    """

    newline_only = tmp_path / "newlines.jsonl"
    newline_only.write_bytes(b"\n\n\n")
    with JsonlLedger(newline_only, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert ledger.record_count == 0
        ledger.append(_payload(0))
    assert len(scan_jsonl(newline_only)) == 1
    assert newline_only.read_bytes().startswith(b"\n\n\n")

    torn_first = tmp_path / "torn-first.jsonl"
    torn_first.write_bytes(b'{"schema_version":')
    with JsonlLedger(torn_first, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        assert ledger.record_count == 0
        assert ledger.recovery_evidence is not None
        assert ledger.recovery_evidence.byte_offset == 0
    assert torn_first.read_bytes() == b""


@pytest.mark.smoke
def test_mixed_version_sidecar_binds_to_writer_version(tmp_path: Path) -> None:
    """A sidecar written by one writer version never streams for another."""

    path = tmp_path / "attempts.jsonl"
    _write_ledger(path, 2)
    state = json.loads(open_state_path(path).read_bytes())
    assert state["ledger_schema_version"] == ATTEMPT_SCHEMA_VERSION_V3
    # Rewriting the binding to another version must void the stream skip
    # (full scan; rows unreadable by that writer refuse loudly).
    state["ledger_schema_version"] = OPERATIONAL_EVENT_SCHEMA_VERSION
    open_state_path(path).write_bytes(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    with pytest.raises(LedgerCorruptionError):
        JsonlLedger(path, OPERATIONAL_EVENT_SCHEMA_VERSION)
