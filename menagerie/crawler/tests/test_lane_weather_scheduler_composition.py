"""Provider weather and burst session lanes under the pipelined scheduler.

The step-100 leg is planned on ``--scheduler`` with the author lane at 4 and
the checker lane at 4. Every existing weather suite (quota, auth, billing,
content refusal) exercises the WAVE driver, and every pipelined suite runs in
fair weather; nothing pinned the composition the real leg will actually face.
These tests walk the exact launch shapes: mid-leg checker quota at the lane
bound, an author usage pause with pipelined siblings in flight, a content
refusal exhausting inside a burst wave, both pools at their configured bound
in one leg, and a TW-H review-held grant coexisting with a pipelined close.

Weather semantics must be scheduler-invariant: model-local failures stay
model-local, campaign pauses buy nothing further, a resumed leg converges
without re-buying settled work, and the 4/4 lane configuration changes no
canonical byte relative to the serial reference.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    canonical_requeue_grants_path,
)
from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.paid_session_lifecycle import load_paid_session_rows
from menagerie.crawler.tests.test_author_wave_concurrency import (
    ObservedAuthor,
    PausingAuthor,
    _canonical_projection,
)
from menagerie.crawler.tests.test_checker_wave_concurrency import ObservedChecker
from menagerie.crawler.tests.test_gate_verdict_reuse import ReuseAwareChecker
from menagerie.crawler.tests.test_refusal_schema import ContentRefusalAuthor
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeChecker,
    FakePauseScheduler,
    InaccurateChecker,
    _driver,
    _paths,
    _snapshot,
)

LANE_DEGREE = 4  # the planned step-100 lane configuration: author 4 / checker 4


@pytest.mark.heavy
def test_pipelined_checker_quota_pause_at_the_lane_bound_pauses_once_and_resumes(
    tmp_path: Path,
) -> None:
    """A mid-leg checker quota pause under the scheduler is one clean pause.

    Execute-first means every model has already executed when the metadata
    lane hits quota; the pause must unwind the pipelined round loop without
    minting a gate, a park, or a terminal for any innocent model, and the
    resumed leg must finish the campaign from the recorded triage attempts.
    """

    snapshot = _snapshot(tmp_path, count=10)
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)
    checker = ObservedChecker(quota=True)

    result = _driver(
        tmp_path,
        snapshot,
        checker=checker,
        checker_concurrency=LANE_DEGREE,
        pause_scheduler=scheduler,
        scheduler="pipelined",
    ).run()

    assert result.status == "paused:usage-limit"
    assert scheduler.calls == 1
    # No gate was appended and no model was blamed for the provider weather.
    assert scan_jsonl(paths.ledgers.gates) == []
    for record in scan_jsonl(paths.ledgers.models):
        assert not str(record["status"]["code"]).startswith("failed:")

    resumed = _driver(
        tmp_path,
        snapshot,
        checker=FakeChecker(),
        checker_concurrency=LANE_DEGREE,
        scheduler="pipelined",
    ).run()
    assert resumed.status == "complete"
    settled = {record["stable_id"] for record in scan_jsonl(paths.ledgers.models)}
    assert settled == {item.stable_id for item in snapshot.items}


@pytest.mark.heavy
def test_pipelined_author_usage_pause_preserves_pipelined_siblings(
    tmp_path: Path,
) -> None:
    """An author quota pause mid-round keeps every drained sibling session.

    The wave-pool drain-never-cancel contract must survive the scheduler: the
    pause unwinds ``run_pipelined_leg`` through the pooled ``_ensure_authors``
    call, finished siblings stay in the disposable cache, and the resumed
    pipelined leg reloads them instead of re-buying their sessions.
    """

    snapshot = _snapshot(tmp_path, count=8)
    pausing_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)
    author = PausingAuthor(pausing_id, hold=lambda _stable_id: None)

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=LANE_DEGREE,
        pause_scheduler=scheduler,
        scheduler="pipelined",
    ).run()

    assert result.status == "paused:usage-limit"
    assert scheduler.calls == 1
    drained = [stable_id for stable_id in author.completed if stable_id != pausing_id]
    assert drained, "the fixture must keep siblings in flight at pause time"
    for stable_id in drained:
        cache = paths.work_root / stable_id / "driver-author-artifact.json"
        assert cache.is_file(), f"paused pipelined round discarded a sibling: {stable_id}"

    resumed_author = ObservedAuthor()
    resumed = _driver(
        tmp_path,
        snapshot,
        author=resumed_author,
        author_concurrency=LANE_DEGREE,
        scheduler="pipelined",
    ).run()
    assert resumed.status == "complete"
    for stable_id in drained:
        assert stable_id not in resumed_author.calls, (
            "the warm resume re-bought a drained sibling's author session"
        )


@pytest.mark.heavy
def test_pipelined_content_refusal_exhaustion_stays_model_local_in_a_burst_wave(
    tmp_path: Path,
) -> None:
    """A refusal exhausting inside a 4-wide wave contains only its member.

    The TK1 refusal shape was pinned at ``author_concurrency=1``; the launch
    lane is 4. Under W5 the victim is CONTAINED as an author-wave
    deferred-family park (no machine verdict -- OC-Q6) while every sibling in
    the same pooled wave completes, and the refusal must never escalate to a
    campaign pause or a lane-wide stop.
    """

    snapshot = _snapshot(tmp_path, count=8)
    victim = snapshot.items[-1].stable_id
    author = ContentRefusalAuthor({victim: "stage2"})

    result = _driver(
        tmp_path,
        snapshot,
        author=author,
        author_concurrency=LANE_DEGREE,
        scheduler="pipelined",
        sleeper=lambda _seconds: None,
    ).run()

    assert result.status == "complete-with-deferrals"
    models = {
        record["stable_id"]: record
        for record in scan_jsonl(_paths(tmp_path, snapshot).ledgers.models)
    }
    assert victim not in models
    from menagerie.crawler.scheduler_parks import (
        load_scheduler_quarantine_rows,
        scheduler_quarantine_path,
    )

    records_root = _paths(tmp_path, snapshot).ledgers.models.parent.parent
    (park,) = load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)])
    assert park["stable_id"] == victim
    assert park["scheduler_quarantine"]["site"] == "author-wave"
    for item in snapshot.items:
        if item.stable_id == victim:
            continue
        assert models[item.stable_id]["status"]["code"] == "runs", (
            f"sibling {item.stable_id} was dragged down by a model-local refusal"
        )


@pytest.mark.heavy
def test_pipelined_burst_four_by_four_closes_the_serial_partition(
    tmp_path: Path,
) -> None:
    """The planned 4/4 lane configuration changes no canonical byte.

    The author pool must genuinely reach the configured bound inside one
    pipelined round (a rendezvous a narrower lane cannot satisfy), the checker
    lane must respect its bound, and the projection must equal the serial
    pipelined reference record-for-record.
    """

    burst_root = tmp_path / "burst"
    serial_root = tmp_path / "serial"
    burst_root.mkdir()
    serial_root.mkdir()

    burst_snapshot = _snapshot(burst_root, count=8)
    barrier = threading.Barrier(LANE_DEGREE, timeout=30.0)
    author = ObservedAuthor(hold=lambda _stable_id: barrier.wait())
    checker = ObservedChecker()
    burst = _driver(
        burst_root,
        burst_snapshot,
        author=author,
        checker=checker,
        author_concurrency=LANE_DEGREE,
        checker_concurrency=LANE_DEGREE,
        scheduler="pipelined",
    ).run()
    assert burst.status == "complete"
    assert author.peak_concurrency == LANE_DEGREE
    assert checker.peak_concurrency <= LANE_DEGREE

    serial_snapshot = _snapshot(serial_root, count=8)
    serial = _driver(
        serial_root,
        serial_snapshot,
        author_concurrency=1,
        checker_concurrency=1,
        scheduler="pipelined",
    ).run()
    assert serial.status == "complete"

    assert _canonical_projection(
        _paths(burst_root, burst_snapshot), burst_root
    ) == _canonical_projection(_paths(serial_root, serial_snapshot), serial_root)


@pytest.mark.heavy
def test_pipelined_leg_holds_a_stage_mismatched_grant_model_locally(
    tmp_path: Path,
) -> None:
    """A TW-H review-held grant coexists with a pipelined close.

    Leg 1 terminalizes one model through a rejecting checker; an operator then
    issues a grant whose stage contradicts the reviewed terminal. The next
    pipelined leg must disclose the model-local review hold durably, bind no
    work for the held model, leave the grant outstanding, and still drive its
    sibling to completion -- the T2-01 leg-killer shape, dead under both
    schedulers.
    """

    snapshot = _snapshot(tmp_path, count=2)
    held_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)

    first = _driver(
        tmp_path,
        snapshot,
        checker=InaccurateChecker(),
        scheduler="pipelined",
    ).run()
    assert first.status == "terminal-partition-complete"

    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    grant = build_canonical_requeue_grant(
        canonical,
        stable_id=held_id,
        stage="import",  # contradicts the recorded accuracy-gate terminal
        reason="operator granted against the wrong stage",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(canonical, grant)

    second = _driver(tmp_path, snapshot, scheduler="pipelined").run()
    assert second.status == "terminal-partition-complete"

    holds = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("event_kind") == OperationalEventKind.CAMPAIGN_HEALTH.value
        and event.get("details", {}).get("disposition") == "requeue-grant-review-hold"
    ]
    assert [hold["details"]["stable_id"] for hold in holds] == [held_id]
    consumed = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    assert consumed == [], "a review-held grant must stay outstanding"
    revisions = [
        record["status"]["code"]
        for record in scan_jsonl(paths.ledgers.models)
        if record["stable_id"] == held_id
    ]
    assert revisions == ["failed:accuracy-gate"], (
        "the held model minted a new generation despite the unbindable grant"
    )


@pytest.mark.heavy
def test_pipelined_weather_pause_buys_no_session_twice_across_the_resume(
    tmp_path: Path,
) -> None:
    """Every paid session survives quota accounting across a resume.

    A quota response rendered no verdict, so its work must be re-adjudicated,
    but that paid attempt remains a durable buy. Changing invocations can
    neither erase it nor misdescribe the replacement session as the first.
    """

    snapshot = _snapshot(tmp_path, count=10)
    paths = _paths(tmp_path, snapshot)
    scheduler = FakePauseScheduler(paths.wakeup_root)

    paused_checker = ObservedChecker(quota=True)
    paused = _driver(
        tmp_path,
        snapshot,
        checker=paused_checker,
        checker_concurrency=LANE_DEGREE,
        pause_scheduler=scheduler,
        scheduler="pipelined",
    ).run()
    assert paused.status == "paused:usage-limit"

    resumed_checker = ReuseAwareChecker()
    resumed = _driver(
        tmp_path,
        snapshot,
        checker=resumed_checker,
        checker_concurrency=LANE_DEGREE,
        scheduler="pipelined",
    ).run()
    assert resumed.status == "complete"

    bought = [
        row
        for row in load_paid_session_rows(paths.paid_session_ledger)
        if row.get("event") == "session_reserved"
        and row.get("lane") == "checker"
        and row.get("gate_kind") == "metadata_batch"
    ]
    assert len(bought) == paused_checker.metadata_calls + resumed_checker.metadata_calls
    covered = [
        str(member["stable_id"])
        for row in bought
        for member in row["axis"]["work_generations"]
    ]
    assert set(covered) == {item.stable_id for item in snapshot.items}
    assert len(covered) > len(set(covered)), (
        "the quota-paid batch must remain charged when its work is re-adjudicated"
    )
