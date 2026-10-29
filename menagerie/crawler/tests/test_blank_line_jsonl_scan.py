"""A whitespace-only JSONL line carries no record and must not read as corruption.

Found rehearsing the step-100 launch against a copy of the pilot records: NINE `.jsonl`
files in the live campaign are exactly one newline byte -- the partitioner's
`records/partitions/empty-deferred.jsonl` and every intake snapshot's empty
`sources/deferred.jsonl`. `scan_jsonl` classifies a blank line as a malformed COMPLETE
line, so `_validate_canonical_jsonl_append_only` -- which selects EVERY `.jsonl` under
`records/` and calls `scan_jsonl` on it -- raises `canonical JSONL has incomplete or
invalid appended lines` on a campaign whose records are perfectly well-formed.

Skipping whitespace-only lines weakens nothing:

* a torn tail is still a line WITHOUT a trailing newline, caught before decoding;
* a truncated record is still a partial JSON document, caught by the decoder; and
* every lenient reader in the tree (the six valve loaders, the rung registry decoder)
  already skips blank lines, so this makes the strict reader agree with them instead of
  disagreeing.

What stays fail-closed is pinned below: garbage that is not whitespace, a torn tail
after a blank line, and a hash-inconsistent record all still raise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.recordio import (
    LedgerCorruptionError,
    TornTailError,
    scan_jsonl,
)

pytestmark = pytest.mark.smoke


def _row(index: int) -> bytes:
    """Return one canonical operational-event line with a valid payload digest."""

    payload = {
        "schema_version": "menagerie.crawler.operational-event.v1",
        "event_id": f"probe-{index}",
        "created_at": "2026-08-12T00:00:00Z",
        "event_kind": "campaign-health",
        "ledger_seq": index,
        "status": "healthy",
        "details": {"disposition": "probe"},
    }
    payload["payload_sha256"] = stable_hash(payload)
    return canonical_json_bytes(payload) + b"\n"


def test_a_single_newline_file_is_an_empty_ledger(tmp_path: Path) -> None:
    """The exact shape of `empty-deferred.jsonl` and an empty `deferred.jsonl`."""

    path = tmp_path / "empty-deferred.jsonl"
    path.write_bytes(b"\n")
    assert scan_jsonl(path, validate=False) == []


def test_a_zero_byte_file_is_still_an_empty_ledger(tmp_path: Path) -> None:
    """The zero-byte spelling keeps reading as empty (unchanged behavior)."""

    path = tmp_path / "zero.jsonl"
    path.write_bytes(b"")
    assert scan_jsonl(path, validate=False) == []


def test_blank_lines_between_records_are_skipped_not_counted(tmp_path: Path) -> None:
    """Records survive; the blank lines contribute nothing."""

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(_row(1) + b"\n" + _row(2) + b"   \n" + b"\t\n" + _row(3))
    records = scan_jsonl(path, validate=False)
    assert [record["ledger_seq"] for record in records] == [1, 2, 3]


def test_a_non_whitespace_malformed_line_still_raises(tmp_path: Path) -> None:
    """Garbage is still corruption -- the tripwire is unchanged."""

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(_row(1) + b"not json\n")
    with pytest.raises(LedgerCorruptionError, match="malformed complete JSON"):
        scan_jsonl(path, validate=False)


def test_a_torn_tail_after_a_blank_line_still_raises(tmp_path: Path) -> None:
    """A newline-less final line is still a torn tail, blank lines notwithstanding."""

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(_row(1) + b"\n" + _row(2).rstrip(b"\n"))
    with pytest.raises(TornTailError):
        scan_jsonl(path, validate=False)


def test_a_blank_final_line_is_not_a_torn_tail(tmp_path: Path) -> None:
    """A trailing blank line is a terminated line, so it is not a torn tail."""

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(_row(1) + b"\n")
    assert [record["ledger_seq"] for record in scan_jsonl(path, validate=False)] == [1]


def test_a_hash_inconsistent_record_still_raises(tmp_path: Path) -> None:
    """Content-identity verification is untouched by the blank-line rule."""

    path = tmp_path / "ledger.jsonl"
    payload = json.loads(_row(1))
    payload["details"] = {"disposition": "tampered"}
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    with pytest.raises(LedgerCorruptionError):
        scan_jsonl(path)


def test_every_whitespace_only_line_shape_reads_as_empty(tmp_path: Path) -> None:
    """CRLF, spaces, and tabs are all whitespace-only, not records."""

    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b"\r\n" + b"    \n" + b"\t\t\n" + b" \r\n")
    assert scan_jsonl(path, validate=False) == []
