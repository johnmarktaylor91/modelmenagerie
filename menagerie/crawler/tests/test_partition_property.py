"""Partition close as a state-space PROPERTY, not a set of hand-picked cells.

T5-6: ``partition_report`` is a pure no-I/O function with a >500-cell input
space, and the suite held ~30 hand-picked cells -- defects 2/3/5 were each an
unvisited cell. These tests sweep the space exhaustively for a two-model
universe and by seeded random for a three-model universe, asserting the ONE
invariant everything else derives from:

    every intake id lands in EXACTLY one accounting bucket (a terminal status
    bucket XOR a machine-deferral claim) XOR the report is invalid and
    ``assert_partition`` raises a ``PartitionError`` naming the hole --
    and the verdict is stable under record permutation.

The oracle below recomputes coverage independently from raw sets and revision
comparisons; it never calls the code under test to decide what to expect.
Also pinned here: the reducer's record_seq-regression refusal (dead code as far
as the suite knew) and the close-status composition matrix (the combined
suffixes and every ``phase-`` close were never asserted).
"""

from __future__ import annotations

import itertools
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler.driver import _completion_close_status
from menagerie.crawler.grant_parks import build_grant_park_row
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.refusal_parks import build_refusal_park_row
from menagerie.crawler.status import (
    PartitionError,
    assert_partition,
    partition_report,
)
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_failed_attempt,
    make_model,
)
from menagerie.crawler.tests.test_reducer import CanonicalReducer

pytestmark = pytest.mark.smoke

REVISION = "sha256:" + "a" * 64
OTHER_REVISION = "sha256:" + "b" * 64

RECORD_STATES = (
    "absent",
    "runs",
    "failed",
    "skipped",
    "invalid-status",
    "duplicate",
)

ROW_STATES = (
    None,
    "grant-park",
    "refusal-park-parentless",
    "refusal-park-parented",
    "forged",
)


def _records_for(stable_id: str, state: str) -> list[dict]:
    """Build the current-record list one record state contributes."""

    def record(code: str, revision: str = REVISION) -> dict:
        return {
            "stable_id": stable_id,
            "record_revision": revision,
            "status": {"code": code},
        }

    if state == "absent":
        return []
    if state == "runs":
        return [record("runs")]
    if state == "failed":
        return [record("failed:runner")]
    if state == "skipped":
        return [record("skipped:no-description")]
    if state == "invalid-status":
        return [record("bogus:not-terminal")]
    if state == "duplicate":
        return [record("runs"), record("failed:runner", OTHER_REVISION)]
    raise AssertionError(state)


def _row_for(stable_id: str, state: Optional[str]) -> Optional[dict]:
    """Build the machine-deferral candidate row one row state contributes."""

    if state is None:
        return None
    if state == "grant-park":
        return build_grant_park_row(
            stable_id=stable_id,
            work_id=f"work-{stable_id}",
            name=stable_id,
            campaign_id="campaign-1",
            run_id="run-1",
            machine_id="machine-1",
            created_at="2026-08-11T00:00:00Z",
            blocked_parent_revision=REVISION,
            blocked_stage="accuracy-gate",
            pending_model={
                "stable_id": stable_id,
                "parent_revision": REVISION,
                "status": {"code": "runs", "reason_code": None, "stage": None},
            },
        )
    if state in {"refusal-park-parentless", "refusal-park-parented"}:
        return build_refusal_park_row(
            stable_id=stable_id,
            work_id=f"work-{stable_id}",
            name=stable_id,
            campaign_id="campaign-1",
            run_id="run-1",
            machine_id="machine-1",
            created_at="2026-08-11T00:00:00Z",
            refusal_class="ReductionError",
            record_error_type="ReductionError",
            record_error="synthetic refusal",
            fallback_error_type="ReductionError",
            fallback_error="synthetic refusal",
            intended_status_code="failed:runner",
            intended_reason_code="protocol-violation",
            blocked_parent_revision=(REVISION if state == "refusal-park-parented" else None),
            candidate_sha256=OTHER_REVISION,
        )
    if state == "forged":
        forged = dict(
            _row_for(stable_id, "refusal-park-parentless")  # type: ignore[arg-type]
        )
        forged["run_id"] = "forged"  # breaks the content digest
        return forged
    raise AssertionError(state)


def _row_claims_or_coexists(state: Optional[str], records: list[dict]) -> tuple[bool, bool]:
    """Independent oracle: (claims-when-absent, live-when-present).

    Derived straight from the row-state semantics and revision comparisons,
    never from the accounting code under test. ``forged`` rows account for
    nothing. A park row claims an ABSENT model; with a PRESENT record it stays
    live exactly while the revision it recorded still governs (grant parks:
    recorded parent still current; parented refusal parks: same; parentless
    refusal parks are released by ANY record).
    """

    if state is None or state == "forged":
        return False, False
    if not records:
        return True, False
    if state == "grant-park" or state == "refusal-park-parented":
        # Live only while EVERY record still shows the recorded revision: any
        # newer revision anywhere releases the park (order-stable by design).
        return True, all(record["record_revision"] == REVISION for record in records)
    if state == "refusal-park-parentless":
        return True, False
    raise AssertionError(state)


def _check_invariant(
    intake: tuple[str, ...],
    per_model: Mapping[str, tuple[str, Optional[str]]],
    extra_present: bool,
) -> None:
    """Assert the total-disjoint-cover invariant for one universe cell."""

    records: list[dict] = []
    deferrals: dict[str, Mapping[str, Any]] = {}
    for stable_id, (record_state, row_state) in per_model.items():
        records.extend(_records_for(stable_id, record_state))
        row = _row_for(stable_id, row_state)
        if row is not None:
            deferrals[stable_id] = row
    if extra_present:
        records.append(
            {
                "stable_id": "mx_extra",
                "record_revision": REVISION,
                "status": {"code": "runs"},
            }
        )

    report = partition_report(intake, records, machine_deferrals=deferrals)

    # Independent oracle over raw sets and revisions.
    counts: dict[str, int] = {}
    for record in records:
        counts[str(record["stable_id"])] = counts.get(str(record["stable_id"]), 0) + 1
    present = set(counts)
    expected_duplicates = {stable_id for stable_id, count in counts.items() if count > 1}
    expected_extra = present - set(intake)
    expected_claimed = set()
    expected_live_parks = set()
    for stable_id, (record_state, row_state) in per_model.items():
        model_records = _records_for(stable_id, record_state)
        claims, live_when_present = _row_claims_or_coexists(row_state, model_records)
        if not model_records and claims:
            expected_claimed.add(stable_id)
            expected_live_parks.add(stable_id)
        elif model_records and live_when_present:
            expected_live_parks.add(stable_id)
    expected_missing = set(intake) - present - expected_claimed
    expected_valid = not (expected_missing or expected_extra or expected_duplicates)

    assert report.missing_ids == frozenset(expected_missing)
    assert report.extra_ids == frozenset(expected_extra)
    assert report.duplicate_ids == frozenset(expected_duplicates)
    assert report.valid == expected_valid

    # Live parks surface exactly once, split by kind, and never overlap the
    # superseded disclosures.
    live_parks = set(report.grant_pending_ids) | set(report.refusal_parked_ids)
    assert live_parks == expected_live_parks
    assert not set(report.grant_pending_ids) & set(report.refusal_parked_ids)
    assert not live_parks & set(report.superseded_deferral_ids)

    # THE invariant: every intake id in exactly one accounting bucket, XOR the
    # partition is invalid and assert_partition raises naming every hole.
    claimed_ids = {
        stable_id for stable_ids in report.machine_deferred.values() for stable_id in stable_ids
    }
    for stable_id in intake:
        buckets_holding = [code for code, members in report.buckets.items() if stable_id in members]
        in_terminal = bool(buckets_holding)
        in_claim = stable_id in claimed_ids
        in_missing = stable_id in report.missing_ids
        assert (in_terminal, in_claim, in_missing).count(True) == 1, (
            stable_id,
            buckets_holding,
            in_claim,
            in_missing,
        )
        if in_terminal and per_model[stable_id][0] != "duplicate":
            assert len(buckets_holding) == 1

    has_invalid_bucket = any(code.startswith("invalid:") for code in report.buckets)
    if expected_valid and not has_invalid_bucket:
        assert_partition(intake, records, machine_deferrals=deferrals)
    else:
        with pytest.raises(PartitionError) as raised:
            assert_partition(intake, records, machine_deferrals=deferrals)
        for stable_id in expected_missing:
            assert stable_id in str(raised.value)

    # Permutation stability: accounting never depends on record order.
    shuffled = list(records)
    random.Random(1359).shuffle(shuffled)
    assert partition_report(intake, shuffled, machine_deferrals=deferrals) == report


def test_partition_is_a_total_disjoint_cover_over_every_model_state() -> None:
    """Exhaustive two-model sweep over record x deferral-row state products."""

    intake = ("m0", "m1")
    states = tuple(itertools.product(RECORD_STATES, ROW_STATES))
    for m0_state, m1_state in itertools.product(states, states):
        _check_invariant(intake, {"m0": m0_state, "m1": m1_state}, extra_present=False)


def test_partition_invariant_holds_on_seeded_random_three_model_universes() -> None:
    """Seeded random three-model sample, extras included."""

    intake = ("m0", "m1", "m2")
    rng = random.Random(9617)
    states = tuple(itertools.product(RECORD_STATES, ROW_STATES))
    for _ in range(250):
        per_model = {stable_id: rng.choice(states) for stable_id in intake}
        _check_invariant(intake, per_model, extra_present=rng.random() < 0.5)


def test_reducer_refuses_a_record_seq_regression(tmp_path: Path) -> None:
    """A superseding candidate whose record_seq does not increase is refused."""

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    failed_attempt = make_failed_attempt()
    first = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
        persisted = reducer.append_model(first).record
        candidate = deepcopy(persisted)
        candidate.pop("record_revision", None)
        candidate["parent_revision"] = persisted["record_revision"]
        candidate["status"]["supersedes_revision"] = persisted["record_revision"]
        candidate["notes"] = "regressed sequence"
        candidate["record_seq"] = persisted["record_seq"]
        with pytest.raises(Exception, match="record_seq must increase"):
            reducer.append_model(candidate)


def test_close_status_composition_matrix() -> None:
    """Every close-status cell: complete family, suffix composition, phase prefix.

    ``complete`` may surface ONLY with no live park of either kind; live parks
    always force the terminal-partition family with their suffix; suffix order
    is deferrals, then review holds, then refusal parks; a phase close carries
    the ``phase-`` prefix and trails ``:{phase}``.
    """

    for complete, phase, deferred, review, refusal in itertools.product(
        (True, False), (None, "pytorch"), *(((False, True),) * 3)
    ):
        status = _completion_close_status(
            complete=complete,
            phase=phase,
            has_deferrals=deferred,
            has_review_holds=review,
            has_refusal_parks=refusal,
        )
        deferral_suffix = "-with-deferrals" if deferred else ""
        review_suffix = "-with-review-holds" if review else ""
        refusal_suffix = "-with-refusal-parks" if refusal else ""
        if complete and not review and not refusal:
            expected_core = f"complete{deferral_suffix}"
        else:
            expected_core = (
                f"terminal-partition-complete{deferral_suffix}{review_suffix}{refusal_suffix}"
            )
        expected = expected_core if phase is None else f"phase-{expected_core}:{phase}"
        assert status == expected, (complete, phase, deferred, review, refusal)
        # A live operator decision (either park kind) never reads "complete".
        if review or refusal:
            assert not status.startswith("complete")
            assert not status.startswith("phase-complete")
