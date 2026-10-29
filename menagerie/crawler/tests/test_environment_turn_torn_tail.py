"""D-60 tripwires: env-turn effort ledger torn-tail evidence + writer lock.

The exposure this suite pins closed: ``environment-turn-effort.jsonl`` was the
ONE append-only ledger with neither a writer lock nor an evidenced torn-tail
recovery path. A crash mid-append left a torn final line that the reader
silently dropped -- un-counting one env-turn attempt per crash -- and the
lockless writer welded the next row onto the torn fragment, turning one lost
attempt into permanent interior corruption. The contract under test:

* a torn (non-newline-terminated) final line is recovered through
  ``recordio.recover_torn_tail`` with a durable evidence row AND a
  campaign-health operational event -- never a silent skip;
* the append site recovers before writing (no welding) and holds the same
  exclusive data-inode flock the sibling valve ledgers hold;
* a COMPLETE malformed interior line stays hard corruption: typed refusal,
  no truncation, no evidence -- the tripwire is never weakened;
* driver wiring heals the ledger at lane build, before the driver's
  long-lived operational ledger handle exists (the only window a torn tail
  can exist: the ledger's sole writer is the DriverLock-excluded driver).
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from menagerie.crawler.env_lifecycle import (
    ENVIRONMENT_TURN_EFFORT_RECORD_VERSION,
    EnvironmentLifecycleError,
    _append_environment_turn_row,
    environment_turn_effort_path,
    replay_environment_turn_attempts,
)
from menagerie.crawler.recordio import (
    SingleWriterError,
    operational_events_path,
    scan_jsonl,
)

pytestmark = pytest.mark.smoke

_KEY = ("py311-default", "osx-arm64", "gen-1")

#: A crash-mid-append signature: a partial JSON object with no newline.
_TORN_FRAGMENT = b'{"generation": "gen-1", "intent": "py311-default", "outco'


def _row_payload(outcome: str) -> dict[str, object]:
    return {
        "record_version": ENVIRONMENT_TURN_EFFORT_RECORD_VERSION,
        "intent": _KEY[0],
        "target": _KEY[1],
        "generation": _KEY[2],
        "outcome": outcome,
    }


def _row_line(outcome: str) -> bytes:
    return (
        json.dumps(_row_payload(outcome), ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _seed(records_root: Path, *, complete: int, torn: bool) -> Path:
    ledger = environment_turn_effort_path(records_root)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_row_line("attempt") for _ in range(complete))
    if torn:
        payload += _TORN_FRAGMENT
    ledger.write_bytes(payload)
    return ledger


def _recovery_path(ledger: Path) -> Path:
    return ledger.with_suffix(f"{ledger.suffix}.recovery.jsonl")


def _recovery_rows(ledger: Path) -> list[dict[str, object]]:
    path = _recovery_path(ledger)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _tail_recovery_events(records_root: Path) -> list[dict[str, object]]:
    from menagerie.crawler.env_lifecycle import (
        ENVIRONMENT_TURN_TAIL_RECOVERY_DISPOSITION,
    )

    events_path = operational_events_path(records_root)
    if not events_path.is_file():
        return []
    return [
        event
        for event in scan_jsonl(events_path)
        if isinstance(event.get("details"), dict)
        and event["details"].get("disposition")
        == ENVIRONMENT_TURN_TAIL_RECOVERY_DISPOSITION
    ]


def test_torn_tail_read_is_evidenced_and_truncated(tmp_path: Path) -> None:
    """A torn final line is truncated WITH a durable evidence row, never silently."""

    ledger = _seed(tmp_path, complete=2, torn=True)
    expected_offset = len(_row_line("attempt")) * 2

    attempts = replay_environment_turn_attempts(ledger, _KEY)

    assert attempts == 2
    content = ledger.read_bytes()
    assert content.endswith(b"\n"), "torn tail must be truncated, not silently retained"
    assert len(content) == expected_offset
    evidence = _recovery_rows(ledger)
    assert len(evidence) == 1, "tail recovery must leave a durable evidence row"
    assert evidence[0]["byte_offset"] == expected_offset
    assert evidence[0]["byte_count"] == len(_TORN_FRAGMENT)
    assert str(evidence[0]["tail_sha256"]).startswith("sha256:")


def test_torn_tail_recovery_appends_operational_event(tmp_path: Path) -> None:
    """Recovery discloses one campaign-health event binding the exact torn bytes."""

    ledger = _seed(tmp_path, complete=2, torn=True)
    replay_environment_turn_attempts(ledger, _KEY)

    events = _tail_recovery_events(tmp_path)
    assert len(events) == 1, "tail recovery must append an operational event"
    event = events[0]
    assert event["event_kind"] == "campaign-health"
    assert event["status"] == "degraded"
    details = event["details"]
    assert details["ledger_path"] == str(ledger)
    evidence = _recovery_rows(ledger)
    assert details["tail_sha256"] == evidence[0]["tail_sha256"]
    assert details["byte_offset"] == evidence[0]["byte_offset"]
    assert details["evidence_path"] == str(_recovery_path(ledger))

    # A later clean read never re-discloses: the recovery is one occurrence.
    replay_environment_turn_attempts(ledger, _KEY)
    assert len(_tail_recovery_events(tmp_path)) == 1


def test_append_recovers_torn_tail_instead_of_welding(tmp_path: Path) -> None:
    """The writer never welds a new row onto a torn fragment."""

    ledger = _seed(tmp_path, complete=1, torn=True)

    _append_environment_turn_row(ledger, _row_payload("attempt"))

    # Pre-fix, the welded line is interior corruption and this read raises.
    assert replay_environment_turn_attempts(ledger, _KEY) == 2
    assert len(_recovery_rows(ledger)) == 1
    assert len(_tail_recovery_events(tmp_path)) == 1


def test_interior_corruption_still_refuses_untouched(tmp_path: Path) -> None:
    """A COMPLETE malformed line stays hard corruption: no truncation, no evidence."""

    ledger = environment_turn_effort_path(tmp_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    corrupt = _row_line("attempt") + b"not json at all\n" + _row_line("attempt")
    ledger.write_bytes(corrupt)

    with pytest.raises(EnvironmentLifecycleError):
        replay_environment_turn_attempts(ledger, _KEY)
    assert ledger.read_bytes() == corrupt
    assert _recovery_rows(ledger) == []

    # Interior corruption BEHIND a torn tail refuses before any truncation:
    # recovery may only ever retain bytes the strict decoder accepts.
    torn_after_corrupt = corrupt + _TORN_FRAGMENT
    ledger.write_bytes(torn_after_corrupt)
    with pytest.raises(EnvironmentLifecycleError):
        replay_environment_turn_attempts(ledger, _KEY)
    assert ledger.read_bytes() == torn_after_corrupt
    assert _recovery_rows(ledger) == []


def test_live_writer_refuses_tail_truncation(tmp_path: Path) -> None:
    """A held data-inode flock means an in-flight append: typed refusal, no truncate."""

    ledger = _seed(tmp_path, complete=1, torn=True)
    with ledger.open("r+b") as writer:
        fcntl.flock(writer.fileno(), fcntl.LOCK_EX)
        with pytest.raises(SingleWriterError):
            replay_environment_turn_attempts(ledger, _KEY)
        assert not ledger.read_bytes().endswith(b"\n")
    assert _recovery_rows(ledger) == []


def test_lane_wiring_heals_ledger_at_build(tmp_path: Path) -> None:
    """build_command_environment_lane recovers the ledger before any driver read."""

    from menagerie.crawler.driver_admission import build_command_environment_lane

    records_root = tmp_path / "records"
    ledger = _seed(records_root, complete=2, torn=True)

    build_command_environment_lane(
        ("/bin/false",),
        tmp_path / "runtime",
        records_root=records_root,
    )

    assert ledger.read_bytes().endswith(b"\n")
    assert len(_recovery_rows(ledger)) == 1
    assert len(_tail_recovery_events(records_root)) == 1
