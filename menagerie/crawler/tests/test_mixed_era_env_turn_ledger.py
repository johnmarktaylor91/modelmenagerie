"""AB-30 mixed-era matrix: env-turn-ledger-recovery seam (D-60, WS8).

Declared change under adjudication: ``environment-turn-effort.jsonl`` gains
flock-locked appends plus EVIDENCED torn-tail recovery -- a recovery writes a
durable evidence row to ``environment-turn-effort.jsonl.recovery.jsonl`` AND a
campaign-health operational event with disposition
``environment-turn-effort-tail-recovered``. The mixed-era obligations pinned
here:

* a PRE-ERA clean shard (old-shape rows, no recovery evidence sibling) reads
  through the real replay reader unchanged and byte-identical -- the new
  machinery may never rewrite, truncate, or re-disclose an intact old ledger;
* a torn (crash-mid-append) tail recovers ERA-SCOPED: the reader returns the
  intact rows while the removed bytes land in the durable evidence ledger and
  the campaign-health disclosure -- never a silent drop;
* INTERIOR corruption stays a hard typed refusal with zero truncation and
  zero evidence -- the tripwire is never weakened by the recovery path;
* the recovery evidence row itself is durable and readable through its own
  real readers (``recordio.scan_jsonl`` and the checkpoint validator's
  evidence adjudicator), so the era transition is auditable after the fact.

This is the matrix module discharging the ``env-turn-ledger-recovery`` seam
(``matrix_disposition: owed-by-batch-lane``, ``owed_by: D-60``) in the
committed Wave-2 declaration registry. The seam has no frozen fixture: the
prior-era artifact class is "an existing env-turn ledger without recovery
evidence rows", minted here byte-for-byte in the pre-era writer's shape
(canonical sorted JSON, one row per newline-terminated line, no lock
sidecar, no recovery sibling).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.checkpoint import _tail_recovery_is_evidenced
from menagerie.crawler.env_lifecycle import (
    ENVIRONMENT_TURN_EFFORT_RECORD_VERSION,
    ENVIRONMENT_TURN_TAIL_RECOVERY_DISPOSITION,
    EnvironmentLifecycleError,
    environment_turn_effort_path,
    recover_environment_turn_effort_ledger,
    replay_environment_turn_attempts,
)
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.recordio import operational_events_path, scan_jsonl
from menagerie.crawler.tests.mixed_era import assert_era_outcome, classify_reader

pytestmark = pytest.mark.smoke

#: Exact (intent, target, generation) replay scope used by every cell.
_KEY = ("py311-default", "osx-arm64", "gen-w2")

#: Crash-mid-append signature: a partial JSON object with no trailing newline.
_TORN_FRAGMENT = b'{"generation": "gen-w2", "intent": "py311-default", "outc'


def _row_payload(outcome: str) -> dict[str, object]:
    """One turn-effort outcome row in the (unchanged) v1 record shape."""

    return {
        "record_version": ENVIRONMENT_TURN_EFFORT_RECORD_VERSION,
        "intent": _KEY[0],
        "target": _KEY[1],
        "generation": _KEY[2],
        "outcome": outcome,
    }


def _row_line(outcome: str) -> bytes:
    """Serialize one row exactly as the pre-era (and current) writer does."""

    return (
        json.dumps(_row_payload(outcome), ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _mint_pre_era_shard(records_root: Path, outcomes: tuple[str, ...], *, torn: bool) -> Path:
    """Mint a prior-era ledger shard: plain rows, no recovery sibling, no lock."""

    ledger = environment_turn_effort_path(records_root)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_row_line(outcome) for outcome in outcomes)
    if torn:
        payload += _TORN_FRAGMENT
    ledger.write_bytes(payload)
    return ledger


def _evidence_path(ledger: Path) -> Path:
    return ledger.with_suffix(f"{ledger.suffix}.recovery.jsonl")


def _tail_recovery_events(records_root: Path) -> list[dict[str, object]]:
    """Read the campaign-health disclosures through the real strict reader."""

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


def test_pre_era_clean_shard_reads_unchanged_and_byte_identical(tmp_path: Path) -> None:
    """Matrix cell 1: an intact prior-era shard parses with zero mutation.

    The D-60 machinery only ever activates on a torn (non-newline-terminated)
    tail. A clean old shard must produce the same replay verdict as before the
    batch, keep its exact bytes, and mint neither evidence nor disclosure.
    """

    ledger = _mint_pre_era_shard(
        tmp_path, ("attempt", "attempt", "success", "attempt"), torn=False
    )
    pre_era_bytes = ledger.read_bytes()

    outcome = classify_reader(lambda: replay_environment_turn_attempts(ledger, _KEY))

    assert_era_outcome(
        reader_id="env-turn-replay-clean-prior-era-shard",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    # Era-scoped semantics preserved: attempts since the last success.
    assert outcome[1] == 1
    assert ledger.read_bytes() == pre_era_bytes, (
        "a clean prior-era shard must read byte-identical: the recovery "
        "machinery may never rewrite an intact old ledger"
    )
    assert not _evidence_path(ledger).exists(), (
        "no torn tail happened: an evidence row here would fabricate a recovery"
    )
    assert _tail_recovery_events(tmp_path) == []


def test_torn_tail_recovers_era_scoped_with_durable_evidence(tmp_path: Path) -> None:
    """Matrix cell 2: the reader heals a torn tail WITH evidence, never silently.

    The replay reader returns exactly the intact rows; the removed bytes are
    durably evidenced in the recovery ledger and disclosed as one
    campaign-health event carrying the closed D-60 disposition.
    """

    ledger = _mint_pre_era_shard(tmp_path, ("attempt", "attempt"), torn=True)
    intact_prefix = _row_line("attempt") * 2

    outcome = classify_reader(lambda: replay_environment_turn_attempts(ledger, _KEY))

    assert_era_outcome(
        reader_id="env-turn-replay-torn-tail-shard",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    assert outcome[1] == 2, "both intact attempt rows must survive the recovery"
    assert ledger.read_bytes() == intact_prefix, (
        "recovery must truncate exactly the torn fragment, nothing more"
    )

    evidence_rows = [
        json.loads(line)
        for line in _evidence_path(ledger).read_bytes().splitlines()
        if line.strip()
    ]
    assert len(evidence_rows) == 1, "tail recovery must leave one durable evidence row"
    evidence = evidence_rows[0]
    assert evidence["byte_offset"] == len(intact_prefix)
    assert evidence["byte_count"] == len(_TORN_FRAGMENT)
    assert evidence["tail_sha256"] == hash_bytes(_TORN_FRAGMENT)

    events = _tail_recovery_events(tmp_path)
    assert len(events) == 1, "recovery must disclose one campaign-health event"
    details = events[0]["details"]
    assert isinstance(details, dict)
    assert details["disposition"] == ENVIRONMENT_TURN_TAIL_RECOVERY_DISPOSITION
    assert details["ledger_path"] == str(ledger)
    assert details["tail_sha256"] == evidence["tail_sha256"]
    assert details["byte_offset"] == evidence["byte_offset"]
    assert details["evidence_path"] == str(_evidence_path(ledger))

    # One occurrence, one disclosure: a later clean read never re-discloses.
    assert replay_environment_turn_attempts(ledger, _KEY) == 2
    assert len(_tail_recovery_events(tmp_path)) == 1


def test_startup_recovery_api_is_evidenced_and_idempotent(tmp_path: Path) -> None:
    """Matrix cell 2 (real recovery API): startup healing returns the evidence.

    ``recover_environment_turn_effort_ledger`` is the lane-build entry point;
    its returned ``TailRecoveryEvidence`` must bind the exact removed bytes,
    and a second invocation over the healed ledger must be a no-op.
    """

    ledger = _mint_pre_era_shard(tmp_path, ("attempt",), torn=True)
    intact_prefix = _row_line("attempt")

    evidence = recover_environment_turn_effort_ledger(tmp_path)

    assert evidence is not None, "a torn tail must yield recovery evidence"
    assert evidence.ledger_path == ledger
    assert evidence.byte_offset == len(intact_prefix)
    assert evidence.byte_count == len(_TORN_FRAGMENT)
    assert evidence.tail_sha256 == hash_bytes(_TORN_FRAGMENT)
    assert evidence.evidence_path == _evidence_path(ledger)
    assert ledger.read_bytes() == intact_prefix
    assert len(_tail_recovery_events(tmp_path)) == 1

    # Idempotent over the healed ledger: no second truncation, no re-disclosure.
    assert recover_environment_turn_effort_ledger(tmp_path) is None
    assert ledger.read_bytes() == intact_prefix
    assert len(_tail_recovery_events(tmp_path)) == 1


def test_interior_corruption_refuses_typed_never_recovers(tmp_path: Path) -> None:
    """Matrix cell 3: a complete malformed interior line stays hard corruption.

    Typed refusal, zero truncation, zero evidence -- recovery is scoped to the
    torn TAIL only, and interior corruption behind a torn tail must refuse
    BEFORE any truncation (recovery may only retain bytes the strict decoder
    accepts).
    """

    ledger = environment_turn_effort_path(tmp_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    corrupt = _row_line("attempt") + b"not json at all\n" + _row_line("attempt")
    ledger.write_bytes(corrupt)

    outcome = classify_reader(lambda: replay_environment_turn_attempts(ledger, _KEY))

    assert_era_outcome(
        reader_id="env-turn-replay-interior-corrupt-shard",
        outcome=outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(outcome[1], EnvironmentLifecycleError)
    assert ledger.read_bytes() == corrupt, "interior corruption must never be truncated"
    assert not _evidence_path(ledger).exists()
    assert _tail_recovery_events(tmp_path) == []

    # Interior corruption BEHIND a torn tail: still refused, still untouched.
    torn_after_corrupt = corrupt + _TORN_FRAGMENT
    ledger.write_bytes(torn_after_corrupt)
    outcome = classify_reader(lambda: replay_environment_turn_attempts(ledger, _KEY))
    assert_era_outcome(
        reader_id="env-turn-replay-interior-corrupt-behind-torn-tail",
        outcome=outcome,
        prebatch_expected="refused-typed",
    )
    assert isinstance(outcome[1], EnvironmentLifecycleError)
    assert ledger.read_bytes() == torn_after_corrupt
    assert not _evidence_path(ledger).exists()
    assert _tail_recovery_events(tmp_path) == []


def test_recovery_evidence_row_reads_through_its_own_readers(tmp_path: Path) -> None:
    """Matrix cell 4: the evidence row is durable and readable, not write-only.

    Two real consumers adjudicate it: ``recordio.scan_jsonl`` (the strict
    ledger reader; the evidence row must be a complete, well-formed line) and
    the checkpoint validator's ``_tail_recovery_is_evidenced`` (the
    append-only-history authority that accepts a prefix truncation ONLY when
    the recovery ledger binds the exact removed bytes).
    """

    ledger = _mint_pre_era_shard(tmp_path, ("attempt", "success"), torn=True)
    committed = ledger.read_bytes()  # the pre-recovery (torn) byte history

    assert replay_environment_turn_attempts(ledger, _KEY) == 0
    working = ledger.read_bytes()  # the post-recovery truncated prefix

    outcome = classify_reader(lambda: scan_jsonl(_evidence_path(ledger)))
    assert_era_outcome(
        reader_id="recordio-scan-jsonl-over-recovery-evidence",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    rows = outcome[1]
    assert len(rows) == 1
    assert rows[0]["ledger_path"] == str(ledger)
    assert rows[0]["byte_offset"] == len(working)
    assert rows[0]["byte_count"] == len(_TORN_FRAGMENT)
    assert rows[0]["tail_sha256"] == hash_bytes(_TORN_FRAGMENT)

    # The checkpoint validator accepts THIS truncation as evidenced...
    assert _tail_recovery_is_evidenced(ledger, committed, working) is True
    # ...and only this one: a different torn tail is NOT excused by the row.
    forged = working + _TORN_FRAGMENT + b"X"
    assert _tail_recovery_is_evidenced(ledger, forged, working) is False
