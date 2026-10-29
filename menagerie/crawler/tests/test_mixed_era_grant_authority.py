"""AB-30 matrix: requeue-grant authority seam (W31 grant/authority schema v2).

The Wave-2 batch ladders the grant/authority schema (``created_at`` on every
authority row, unique adjudication key, idempotent double-void refusal).
Prior-era grant rows carry NO ``created_at`` and must stay readable and
consumable in every era (WS7's D-49 rule: absent ``created_at`` tolerated
forever; grants gain it in W31). The consumption-side content-hash binding
must refuse tampered rows identically in every era.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    canonical_requeue_grants_path,
)
from menagerie.crawler.tests.mixed_era import (
    assert_era_outcome,
    classify_reader,
    read_fixture_json,
)
from menagerie.crawler.tests.test_infra_terminal_grant_release import (
    STABLE_ID,
    _consume,
    _paths,
    _terminal_record,
    _worker_crash_attempt,
    _write_attempts,
    _write_models,
)

pytestmark = pytest.mark.smoke


def _grant_ledger(tmp_path: Path) -> tuple[Any, Path]:
    """Return harness paths and the canonical grant ledger under them."""

    paths = _paths(tmp_path)
    _write_attempts(paths, [_worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-1")])
    _write_models(paths, [_terminal_record(STABLE_ID, attempt_ids=("attempt-crash-1",))])
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    return paths, canonical


def test_prior_era_grant_row_appends_and_consumes(tmp_path: Path) -> None:
    """A frozen pre-batch grant row (no created_at) stays consumable."""

    grant = read_fixture_json("requeue_grant.json")
    assert "created_at" not in grant, "the frozen row must model the pre-created_at era"
    paths, canonical = _grant_ledger(tmp_path)

    def _consume_frozen() -> list[tuple[Any, Any]]:
        append_canonical_requeue_grant(canonical, grant)
        _, events = _consume(paths)
        return [
            (event.get("event_kind"), event.get("details", {}).get("release_authority"))
            for event in events
        ]

    outcome = classify_reader(_consume_frozen)
    assert_era_outcome(
        reader_id="grant-append-replay",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    if outcome[0] == "parsed":
        assert ("requeue-grant-consumed", "infra-waiver") in outcome[1]


def test_tampered_grant_row_refuses_typed_at_consumption(tmp_path: Path) -> None:
    """grant_id binds content: a tampered row refuses typed in every era."""

    grant = dict(read_fixture_json("requeue_grant.json"))
    grant["reason"] = "tampered after mint"
    paths, canonical = _grant_ledger(tmp_path)
    canonical.write_text(json.dumps(grant, sort_keys=True) + "\n")

    outcome = classify_reader(lambda: _consume(paths))
    assert outcome[0] == "refused-typed"
    assert "hash mismatch" in str(outcome[1])


def test_mixed_era_created_at_duplicate_reads_as_one_logical_grant(tmp_path: Path) -> None:
    """A W31-era duplicate differing only by created_at is ONE logical grant.

    Models the mixed-era ledger the batch creates: a pre-batch writer
    appended the row without ``created_at``; a schema-v2 writer re-appends
    the same logical grant with it. The consumption reader's duplicate
    comparison already excludes ``created_at``, so the pair must consume as
    exactly one grant -- never a conflicting-duplicate refusal, never two
    consumptions.
    """

    grant = read_fixture_json("requeue_grant.json")
    later_era = dict(grant)
    later_era["created_at"] = "2026-08-24T00:00:00Z"
    paths, canonical = _grant_ledger(tmp_path)
    with canonical.open("w") as handle:
        handle.write(json.dumps(grant, sort_keys=True) + "\n")
        handle.write(json.dumps(later_era, sort_keys=True) + "\n")

    def _consume_mixed() -> list[Any]:
        _, events = _consume(paths)
        return [
            event
            for event in events
            if event.get("event_kind") == "requeue-grant-consumed"
        ]

    outcome = classify_reader(_consume_mixed)
    assert_era_outcome(
        reader_id="grant-consumption-created-at-era-scope",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    if outcome[0] == "parsed":
        assert len(outcome[1]) == 1, "the mixed-era pair must consume as ONE grant"


def test_append_side_conflict_on_created_at_divergence_stays_typed(tmp_path: Path) -> None:
    """The canonical APPEND authority refuses a created_at-divergent duplicate.

    Pre-batch asymmetry pin: append-side dedup compares full logical content
    (so the divergent duplicate refuses typed), while consumption tolerates
    the pair. W31's ladder must resolve this seam deliberately -- whichever
    side it moves, this cell may flip only under the declared W31 change,
    and it must never become a silent double-append.
    """

    grant = read_fixture_json("requeue_grant.json")
    paths, canonical = _grant_ledger(tmp_path)
    append_canonical_requeue_grant(canonical, grant)
    later_era = dict(grant)
    later_era["created_at"] = "2026-08-24T00:00:00Z"

    outcome = classify_reader(lambda: append_canonical_requeue_grant(canonical, later_era))
    assert_era_outcome(
        reader_id="grant-append-created-at-divergence",
        outcome=outcome,
        prebatch_expected="refused-typed",
        expected_flip="W31",
    )
