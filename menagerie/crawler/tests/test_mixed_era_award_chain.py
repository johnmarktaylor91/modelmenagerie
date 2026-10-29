"""AB-30 matrix: artifact-transaction chain seam (AT-4/5/6/7 award-tail batch).

The Wave-2 batch reorders the award tail (authorization before spend), adds
the torn-chain resume class, and stamps the ``publication_lane`` disclosure
field -- rotating ``award_closure_identity``. Every prior-era finalized
4-event chain (the whole 200a corpus) must keep loading era-scoped: the
reorder shapes NEW chains only. The event-kind vocabulary and the per-event
content hash are closed authorities that must refuse foreign or tampered
rows identically in every era.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.artifact_transactions import ArtifactEventKind, _load_events
from menagerie.crawler.tests.mixed_era import (
    assert_era_outcome,
    classify_reader,
    read_fixture_bytes,
)

pytestmark = pytest.mark.smoke

#: The frozen pre-batch chain's exact event sequence.
FROZEN_KINDS = (
    "staged-private",
    "terminal-authorized",
    "reconstruction-committed",
    "private-committed",
)


def _materialize_shard(tmp_path: Path) -> Path:
    shard = tmp_path / "shard.jsonl"
    shard.write_bytes(read_fixture_bytes("award_chain/shard.jsonl"))
    return shard


def test_prior_era_chain_loads_with_its_exact_event_sequence(tmp_path: Path) -> None:
    """The frozen finalized chain loads and keeps its recorded event order."""

    shard = _materialize_shard(tmp_path)
    outcome = classify_reader(
        lambda: tuple(str(event["event_kind"]) for event in _load_events((shard,), memo=None))
    )
    assert_era_outcome(
        reader_id="artifact-event-ledger-load",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    if outcome[0] == "parsed":
        assert outcome[1] == FROZEN_KINDS


def test_frozen_event_kinds_remain_members_of_the_live_closed_enum() -> None:
    """Removing a prior-era event kind would orphan every 200a chain.

    The batch may EXTEND ``ArtifactEventKind``; it may never drop a member
    the frozen corpus uses (a downgrade-ladder violation the first-generation
    rule forbids).
    """

    live = {member.value for member in ArtifactEventKind}
    missing = set(FROZEN_KINDS) - live
    assert not missing, f"live event-kind enum dropped prior-era members: {sorted(missing)}"


def test_unknown_event_kind_refuses_typed(tmp_path: Path) -> None:
    """A row from outside the closed event-kind enum never parses silently."""

    lines = read_fixture_bytes("award_chain/shard.jsonl").decode().splitlines()
    row = json.loads(lines[0])
    row["event_kind"] = "event-from-the-future"
    shard = tmp_path / "mutated.jsonl"
    shard.write_text("\n".join([json.dumps(row, sort_keys=True), *lines[1:]]) + "\n")

    outcome = classify_reader(lambda: _load_events((shard,), memo=None))
    assert outcome[0] == "refused-typed"


def test_tampered_event_payload_refuses_typed(tmp_path: Path) -> None:
    """The per-event content hash refuses byte tampering in every era."""

    lines = read_fixture_bytes("award_chain/shard.jsonl").decode().splitlines()
    row = json.loads(lines[0])
    row["payload_sha256"] = "sha256:" + "0" * 64
    shard = tmp_path / "mutated.jsonl"
    shard.write_text("\n".join([json.dumps(row, sort_keys=True), *lines[1:]]) + "\n")

    outcome = classify_reader(lambda: _load_events((shard,), memo=None))
    assert outcome[0] == "refused-typed"
    assert "payload_sha256" in str(outcome[1])
