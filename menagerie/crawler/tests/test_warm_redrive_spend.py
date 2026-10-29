"""A warm re-drive over copied records must buy essentially no LLM sessions.

The pilot's operating cost is dominated by two purchased lanes: author sessions
(one agentic session per work generation, capped by ``AUTHOR_SESSIONS_MAX``) and
checker sessions (~15 wall-hours per leg). Every leg after the first is a WARM
leg -- it re-drives models whose authoring already staged and whose gates already
adjudicated -- and the whole resume design (the author-result cache, gate verdict
reuse, the durable valve rows, and the derived author-session allowance of docket
T2-07) exists so a warm leg re-mints its records without re-buying that spend.

That claim was only ever proved per-mechanism. This module proves it END TO END on
the composed driver, and proves it over a records tree COPIED to a fresh root, so
no part of the result can depend on the original leg's absolute paths or on
leftover in-memory state.

The tripwire is structural, not a count comparison: both warm lanes RAISE on any
dispatch, so a single re-bought session fails loudly at its exact call site.
"""

from __future__ import annotations

import shutil
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_spend import author_spend_census, chargeable_author_sessions
from menagerie.crawler.constants import METADATA_BATCH_MAX
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.paid_session_lifecycle import load_paid_session_rows
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_author_wave_concurrency import SiblingFailureAuthor
from menagerie.crawler.tests.test_gate_verdict_reuse import (
    ReuseAwareChecker,
    _clear_model_history,
    _spend_events,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeAuthor,
    _driver,
    _paths,
    _snapshot,
)

MODEL_COUNT = 3


class NoSessionAuthor(FakeAuthor):
    """Refuse to sell any author session: every dispatch is a warm-leg failure."""

    def author(self, item: Any, work_root: Path, config: DriverConfig, context: Any) -> Any:
        raise AssertionError(
            f"author session bought for {item.stable_id} where a warm leg must buy none"
        )


class NoSessionChecker(ReuseAwareChecker):
    """Refuse to sell any checker session: every dispatch is a warm-leg failure."""

    def check_metadata(self, artifacts: Any, work_root: Path, config: DriverConfig) -> Any:
        raise AssertionError("metadata session bought where a warm leg must buy none")

    def check_fidelity(self, artifact: Any, work_root: Path, config: DriverConfig) -> Any:
        raise AssertionError("fidelity session bought where a warm leg must buy none")

    def check_terminal(self, artifact: Any, work_root: Path, config: DriverConfig) -> Any:
        raise AssertionError("terminal session bought where a warm leg must buy none")


def _statuses(paths: Any) -> dict[str, str]:
    """Return the current status code of every model revision on the ledger."""

    return {
        str(record["stable_id"]): str(record["status"]["code"])
        for record in scan_jsonl(paths.ledgers.models)
    }


@pytest.mark.smoke
def test_warm_redrive_over_copied_records_buys_no_author_or_checker_session(
    tmp_path: Path,
) -> None:
    """A warm leg re-mints every record while both purchased lanes stay untouched.

    Leg 1 drives a healthy campaign to ``complete``, buying exactly one author
    session per model and one batched metadata gate. The records tree is COPIED to
    a fresh root, the model ledger and driver state are cleared (the re-drive leg's
    starting condition -- every model re-enters work), and leg 2 runs with lanes
    that raise on dispatch.

    Leg 2 must still land all three models ``runs``, and the replay-derived author
    spend census must be BYTE-identical across the two legs: the warm leg staged
    no new authoring, so it bought nothing.
    """

    snapshot = _snapshot(tmp_path, count=MODEL_COUNT)
    cold_author = FakeAuthor()
    cold_checker = ReuseAwareChecker()

    first = _driver(tmp_path, snapshot, author=cold_author, checker=cold_checker).run()

    assert first.status == "complete"
    cold_paths = _paths(tmp_path, snapshot)
    cold_statuses = _statuses(cold_paths)
    assert set(cold_statuses.values()) == {"runs"}
    assert len(cold_statuses) == MODEL_COUNT
    # The cold leg genuinely bought the spend a warm leg must not repeat.
    assert sum(cold_author.calls.values()) == MODEL_COUNT
    cold_census = author_spend_census(
        (
            *scan_jsonl(cold_paths.ledgers.artifacts),
            *scan_jsonl(cold_paths.operational_ledger),
        )
    )
    assert cold_census["sessions_staged"] == MODEL_COUNT
    assert cold_census["work_generations"] == MODEL_COUNT
    assert cold_census["generations_over_allowance"] == 0
    cold_bought, _cold_reused = _spend_events(cold_paths)
    # ceil(N / width) primary adjudications plus one H-5 blind duplicate-accept
    # confirmation per accepted model. The warm leg below buys NEITHER: the
    # copied conflict ledger carries every settled confirmation, so the door
    # is satisfied by replay exactly like the gates themselves.
    assert [event["details"]["gate_kind"] for event in cold_bought] == [
        "metadata_batch"
    ] * (-(-MODEL_COUNT // METADATA_BATCH_MAX) + MODEL_COUNT)

    # Copy the whole tree to a FRESH root: a warm leg's freeness must not depend
    # on the cold leg's absolute paths or on any surviving in-process state.
    warm_root = tmp_path.parent / f"{tmp_path.name}-warm"
    shutil.copytree(tmp_path, warm_root, symlinks=True)
    warm_snapshot = _snapshot(warm_root, count=MODEL_COUNT)
    warm_paths = _paths(warm_root, warm_snapshot)
    _clear_model_history(warm_paths)
    assert not warm_paths.ledgers.models.exists(), "the warm leg must re-mint every record"

    warm_author = NoSessionAuthor()
    warm_checker = NoSessionChecker()

    second = _driver(warm_root, warm_snapshot, author=warm_author, checker=warm_checker).run()

    assert second.status == "complete"
    assert _statuses(warm_paths) == cold_statuses, (
        "the warm leg must reproduce the cold leg's dispositions exactly"
    )
    assert warm_author.calls == {}, "the warm leg bought an author session"
    # The census is a pure replay of the artifact ledger, so an identical census
    # across the two legs is the durable proof that nothing was re-staged.
    assert author_spend_census(
        (
            *scan_jsonl(warm_paths.ledgers.artifacts),
            *scan_jsonl(warm_paths.operational_ledger),
        )
    ) == cold_census
    warm_bought, _warm_reused = _spend_events(warm_paths)
    assert warm_bought == cold_bought, (
        "the warm leg appended a checker-session-bought event of its own"
    )


@pytest.mark.smoke
def test_derived_author_spend_is_exact_under_burst_and_mid_batch_weather(
    tmp_path: Path,
) -> None:
    """Fan-out plus a session death mid-batch must not blur the derived spend.

    Six models at the shipped ``author_concurrency`` of 4, with one model's session
    dying while its siblings are held in flight by a barrier -- provider weather
    arriving mid-batch rather than at a wave boundary.

    The census replays artifact plus operational ledgers. Every survivor stages
    one result, while the model whose session died stages nothing but still owns
    one provider-billed ``author-session-bought`` event. Each record's budget
    must equal that bought-session truth.
    """

    snapshot = _snapshot(tmp_path, count=6)
    failing_id = snapshot.items[2].stable_id
    gate = threading.Barrier(3, timeout=30.0)

    def hold(stable_id: str) -> None:
        """Keep siblings in flight so the death lands mid-batch, not between waves."""

        if stable_id != failing_id:
            try:
                gate.wait()
            except threading.BrokenBarrierError:
                return

    author = SiblingFailureAuthor(
        failing_id, RuntimeError("synthetic mid-batch author weather"), hold=hold
    )

    result = _driver(tmp_path, snapshot, author=author, author_concurrency=4).run()

    assert result.status == "terminal-partition-complete"
    paths = _paths(tmp_path, snapshot)
    artifact_events = scan_jsonl(paths.ledgers.artifacts)
    operational_events = scan_jsonl(paths.operational_ledger)
    paid_session_events = load_paid_session_rows(paths.paid_session_ledger)
    events = (*artifact_events, *operational_events, *paid_session_events)
    staged = Counter(
        (str(event.get("stable_id")), str(event.get("work_id")))
        for event in artifact_events
        if event.get("event_kind") == "staged-private"
    )
    survivors = [item.stable_id for item in snapshot.items if item.stable_id != failing_id]
    assert survivors, "the fixture must contain siblings for the weather to spare"
    # The dead session staged nothing, so it owns no work generation at all.
    assert not [key for key in staged if key[0] == failing_id]
    assert set(staged.values()) == {1}, "no generation may stage twice in one leg"

    census = author_spend_census(events)
    assert census["work_generations"] == len(snapshot.items)
    assert census["sessions_staged"] == len(survivors)
    assert census["sessions_bought"] == len(survivors) + 1
    assert census["sessions_chargeable"] == len(survivors) + 1
    assert census["generations_over_allowance"] == 0
    assert census["over_allowance"] == {}

    for record in scan_jsonl(paths.ledgers.models):
        stable_id = str(record["stable_id"])
        true_count = chargeable_author_sessions(events, stable_id, f"work-{stable_id}")
        assert record["budget"]["author_sessions_used"] == true_count, stable_id
    failed = {str(record["stable_id"]): record for record in scan_jsonl(paths.ledgers.models)}[
        failing_id
    ]
    assert failed["status"]["kind"] == "failed"
    assert failed["budget"]["author_sessions_used"] == 1
