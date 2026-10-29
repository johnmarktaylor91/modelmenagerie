"""Warm re-drive economics: a re-drive over unchanged work buys ~zero sessions.

The pilot's 15h checker wall was bought upstream of the checker: re-drive legs
re-authored and re-froze sources for models whose facts never changed, moving
every identity and defeating any honest cache. W4 (verdict reuse) and the
artifact-authority rehydration path (W5) exist to make the WARM leg -- same
bytes, same identities -- nearly free. These tests prove that end to end at
the driver level, on a physically COPIED campaign with the original deleted,
with lanes that expose every attempted buy so the exact residual spend remains
visible after checker-verdict reuse retirement.

They also pin spend TRUTH: the derived author/checker spend censuses must
equal the fake lanes' own ground-truth call counters, so a census over any
historical shard can be believed without re-deriving anything.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_spend import author_spend_census
from menagerie.crawler.paid_session_lifecycle import load_paid_session_rows
from menagerie.crawler.authority import AuthorityContext
from menagerie.crawler.driver import AuthorArtifact, DriverConfig
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_gate_verdict_reuse import (
    NoSessionChecker,
    ReuseAwareChecker,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    AuthorScript,
    CheckerScript,
    FakeForward,
    RepairingIdentityAuthor,
    ScriptedAuthor,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.heavy

LANE_DEGREE = 4  # the planned step-100 lane configuration: author 4 / checker 4


class NoSessionAuthor(ScriptedAuthor):
    """Refuse to sell any author session: every dispatch is a test failure."""

    def author(
        self,
        item: WorkItem,
        work_root: Path,
        config: DriverConfig,
        context: AuthorityContext,
    ) -> AuthorArtifact:
        raise AssertionError(
            f"author session bought on a warm leg for {item.stable_id}: "
            "rehydration must supply the artifact"
        )


def _author_sessions_bought(paths: Any) -> int:
    """Return replay-derived author spend across artifact and operational shards."""

    census = author_spend_census(
        (
            *scan_jsonl(paths.ledgers.artifacts, validate=False),
            *scan_jsonl(paths.operational_ledger, validate=False),
            *load_paid_session_rows(paths.paid_session_ledger),
        )
    )
    return int(census["sessions_chargeable"])


def _checker_sessions_bought(paths: Any) -> list[dict[str, Any]]:
    """Return the checker-session-bought spend events in append order."""

    return [
        row
        for row in load_paid_session_rows(paths.paid_session_ledger)
        if row.get("event") == "session_reserved" and row.get("lane") == "checker"
    ]


def test_rerun_of_a_settled_campaign_buys_zero_sessions(tmp_path: Path) -> None:
    """Re-running a completed campaign in place is free.

    The operator double-invocation shape (a resumed watcher, a crontab retry):
    every model is settled, so the leg must close by re-validation alone. The
    lanes raise on any dispatch, so one bought session is a loud failure.
    """

    snapshot = _snapshot(tmp_path, count=6)
    first = _driver(tmp_path, snapshot, checker=ReuseAwareChecker()).run()
    assert first.status == "complete"
    paths = _paths(tmp_path, snapshot)
    baseline_author_spend = _author_sessions_bought(paths)
    baseline_checker_spend = len(_checker_sessions_bought(paths))

    rerun = _driver(
        tmp_path,
        snapshot,
        author=NoSessionAuthor(),
        checker=NoSessionChecker(),
        author_concurrency=LANE_DEGREE,
        checker_concurrency=LANE_DEGREE,
    ).run()

    assert rerun.status == "complete"
    assert _author_sessions_bought(paths) == baseline_author_spend
    assert len(_checker_sessions_bought(paths)) == baseline_checker_spend


def test_warm_redrive_on_a_copied_campaign_buys_zero_sessions(tmp_path: Path) -> None:
    """The load-bearing demonstration: a warm re-drive leg is nearly free.

    Leg 1 completes a mixed campaign (runs plus a checker-adjudicated terminal
    skip). The WHOLE campaign is then copied byte-for-byte, the original is
    DELETED (so nothing can silently read the old absolute paths), the copy's
    model ledger and driver state are cleared -- the exact re-drive shape the
    pilot used -- and the warm leg runs with lanes that refuse to sell:

    * author sessions come from artifact-authority rehydration (zero bought);
    * finalized terminal bytes are re-adjudicated live under retired reuse;
    * accepted metadata/fidelity gates are consumed by the gate-history skip
      (zero bought).

    The warm leg must converge to the same terminal statuses. Execution
    (FakeForward) is not an LLM session and is deliberately free to re-run.
    """

    origin_root = tmp_path / "origin"
    origin_root.mkdir()
    origin_snapshot = _snapshot(origin_root, count=4)
    skip_id = origin_snapshot.items[0].stable_id
    author = ScriptedAuthor(
        AuthorScript(
            terminal_outcomes=(("status", "skipped:not-a-real-NN"),),
            terminal_id=skip_id,
        )
    )
    first = _driver(origin_root, origin_snapshot, author=author, checker=ReuseAwareChecker()).run()
    assert first.status == "complete"
    origin_paths = _paths(origin_root, origin_snapshot)
    origin_statuses = {
        record["stable_id"]: record["status"]["code"]
        for record in scan_jsonl(origin_paths.ledgers.models)
    }
    assert origin_statuses[skip_id] == "skipped:not-a-real-NN"
    leg_one_author_spend = _author_sessions_bought(origin_paths)
    assert leg_one_author_spend >= 4
    leg_one_gate_rows = len(scan_jsonl(origin_paths.ledgers.gates))

    warm_root = tmp_path / "warm"
    shutil.copytree(origin_root, warm_root)
    shutil.rmtree(origin_root)

    warm_snapshot = _snapshot(warm_root, count=4)
    warm_paths = _paths(warm_root, warm_snapshot)
    assert warm_snapshot.snapshot_id == origin_snapshot.snapshot_id, (
        "the copied campaign must reload the identical intake snapshot"
    )
    checker_spend_before_warm = len(_checker_sessions_bought(warm_paths))
    # The re-drive shape: model history and driver state reset, every other
    # ledger (gates, attempts, artifacts) and the runtime work tree preserved.
    warm_paths.ledgers.models.unlink()
    if warm_paths.driver_state.exists():
        warm_paths.driver_state.unlink()

    warm = _driver(
        warm_root,
        warm_snapshot,
        author=NoSessionAuthor(),
        checker=ReuseAwareChecker(),
        author_concurrency=LANE_DEGREE,
        checker_concurrency=LANE_DEGREE,
        forward=FakeForward(),
    ).run()

    assert warm.status == "complete"
    warm_statuses = {
        record["stable_id"]: record["status"]["code"]
        for record in scan_jsonl(warm_paths.ledgers.models)
    }
    assert warm_statuses == origin_statuses
    # Spend truth on the copied shard: the warm leg staged no new author
    # session. Retired semantic reuse requires one fresh terminal adjudication,
    # and the pre-dispatch census must retain that real paid call even though
    # its deterministic gate bytes deduplicate against the prior row.
    assert _author_sessions_bought(warm_paths) == leg_one_author_spend
    assert len(_checker_sessions_bought(warm_paths)) == checker_spend_before_warm + 1, (
        "the warm leg's paid terminal re-adjudication was lost from the census"
    )
    # Accepted proposal gates are consumed upstream by the gate-history skip,
    # while deterministic terminal gate bytes append idempotently.
    assert len(scan_jsonl(warm_paths.ledgers.gates)) == leg_one_gate_rows


def test_spend_census_equals_the_fake_lane_ground_truth(tmp_path: Path) -> None:
    """The derived censuses agree with the lanes' own call counters exactly.

    Synthetic history with KNOWN true counts: a burst 4/4 campaign with one
    repair lineage (extra author generations and gate rounds). The artifact
    shard census must count exactly the author sessions the lane sold, and the
    operational spend events must count exactly the checker sessions bought --
    retroactively, from the records alone.
    """

    snapshot = _snapshot(tmp_path, count=7)
    repaired_id = snapshot.items[0].stable_id
    checker = ReuseAwareChecker(
        script=CheckerScript(metadata_verdict="inaccurate", selected_id=repaired_id)
    )
    # Repairs mint a distinct identity every round, so every sold author
    # session is a distinct staged result -- the census must count them all.
    author = RepairingIdentityAuthor()

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        checker=checker,
        author_concurrency=LANE_DEGREE,
        checker_concurrency=LANE_DEGREE,
    ).run()
    # The rejected model exhausts its bounded repairs; the rest run.
    assert result.status == "terminal-partition-complete"
    assert author.calls[repaired_id] > 1, "the fixture must exercise repair spend"

    paths = _paths(tmp_path, snapshot)
    author_sessions_sold = sum(author.calls.values())
    census = author_spend_census(
        (
            *scan_jsonl(paths.ledgers.artifacts, validate=False),
            *scan_jsonl(paths.operational_ledger, validate=False),
            *load_paid_session_rows(paths.paid_session_ledger),
        )
    )
    assert census["sessions_staged"] == author_sessions_sold
    # Every repair round minted its own work generation for the repaired model.
    assert census["work_generations"] == (len(snapshot.items) - 1 + author.calls[repaired_id])
    assert census["generations_over_allowance"] == 0

    bought = _checker_sessions_bought(paths)
    by_kind: dict[str, int] = {}
    for event in bought:
        kind = str(event["gate_kind"])
        by_kind[kind] = by_kind.get(kind, 0) + 1
    assert by_kind.get("metadata_batch", 0) == checker.metadata_calls
    assert by_kind.get("fidelity", 0) == checker.fidelity_calls
    assert by_kind.get("terminal_disposition", 0) == checker.terminal_calls
