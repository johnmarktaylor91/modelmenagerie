"""W17 sim-rig core self-tests + the TE-6 author-utilization floor (AB-12/AB-13).

The rig's numbers are only as trustworthy as its instruments, so the AB-13
self-tests come first: the committed 200a distribution artifact must carry
exactly the measured campaign (the plan's median 243s / p90 1663s / max 2666s
are pinned to the decimal), replay assignment must be deterministic, and the
virtual clock must PROVE it can measure overlap on a planted known-concurrent
load -- an early rig bug advanced time before the writer's real-time slices
could admit queued siblings, measuring mean concurrency ~0.95 for every
scheduler; the planted-overlap self-test exists so that artifact class can
never return silently.

TE-6 (the pool's future acceptance instrument, RED-first by design): today's
tree measures mean author concurrency ~2-3 at width 12 against the >=8 floor
-- the residual round/wave-tail barrier all three labs measured on 200a. The
floor test is strict-xfail: it documents the RED, and the day it passes
(the async author pool, or any scheduler change that closes the barrier) the
strict flag forces this file to be revisited and the floor promoted to a
hard gate.
"""

from __future__ import annotations

from pathlib import Path
from threading import Thread

import pytest

from menagerie.crawler.tests.sim_rig_support import (
    DISTRIBUTION_SCHEMA_VERSION,
    MeasuredReplaySession,
    SessionLedger,
    SessionSpan,
    VirtualClock,
    build_rig_driver,
    load_measured_wall_distribution,
    measured_wall_for,
    production_shaped_asserts,
    run_rig,
)

#: TE-6 / M-6: the async-pool acceptance floor for mean author concurrency.
AUTHOR_UTILIZATION_FLOOR = 8.0


@pytest.mark.smoke
def test_distribution_artifact_is_the_measured_200a_campaign() -> None:
    """The committed replay artifact carries exactly the measured 200a walls.

    The plan's quoted quantiles (median 243s / p90 1663s / max 2666s over
    1547 settled sessions) are pinned here to the decimal: a regenerated or
    hand-edited artifact that no longer matches the measured campaign fails
    loudly instead of silently reshaping every rehearsal built on it.
    """

    artifact = load_measured_wall_distribution()
    assert artifact["schema_version"] == DISTRIBUTION_SCHEMA_VERSION
    assert artifact["settled_sessions"] == 1547
    assert artifact["quantiles"]["median"] == pytest.approx(242.69523912481964)
    assert artifact["quantiles"]["p90"] == pytest.approx(1663.4481881246902)
    assert artifact["quantiles"]["max"] == pytest.approx(2666.0621066656895)
    lanes = artifact["lanes"]
    assert set(lanes) == {"author", "checker"}
    assert lanes["author"]["count"] == 1418
    assert lanes["checker"]["count"] == 129
    for lane in lanes.values():
        walls = lane["wall_seconds"]
        assert len(walls) == lane["count"]
        assert walls == sorted(walls)
        assert all(wall > 0 for wall in walls)


@pytest.mark.smoke
def test_measured_wall_assignment_is_deterministic_and_spans_the_distribution() -> None:
    """Replay durations are identity-hashed: reproducible, spread, seedless."""

    first = measured_wall_for("m_alpha", "author")
    assert measured_wall_for("m_alpha", "author") == first
    walls = {measured_wall_for(f"m_{index}", "author") for index in range(64)}
    assert len(walls) > 32, "hash assignment collapsed onto a few durations"
    assert measured_wall_for("m_alpha", "checker") != first


@pytest.mark.smoke
def test_virtual_clock_measures_a_planted_concurrent_load() -> None:
    """Two overlapping replay sessions measure overlap, not serialization.

    The rig-vacuousness tripwire (AB-13): a virtual clock that advances before
    concurrent siblings can start measures mean concurrency ~1 for EVERY
    scheduler, and every floor built on it is prose. Plant two sessions that
    genuinely overlap on real threads and refuse any serial measurement.
    """

    clock = VirtualClock()
    ledger = SessionLedger()

    def session(stable_id: str) -> None:
        MeasuredReplaySession(clock, ledger, "author").replay_session(stable_id)

    threads = [
        Thread(target=session, args=(stable_id,))
        for stable_id in ("m_planted_one", "m_planted_two")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180.0)
        assert not thread.is_alive(), "planted session wedged the virtual clock"
    walls = {span.stable_id: span.wall_seconds for span in ledger.spans}
    assert walls["m_planted_one"] == pytest.approx(
        measured_wall_for("m_planted_one", "author")
    )
    assert walls["m_planted_two"] == pytest.approx(
        measured_wall_for("m_planted_two", "author")
    )
    measured = ledger.mean_concurrency("author")
    assert measured > 1.5, (
        f"two fully-overlapping planted sessions measured {measured:.2f}: the "
        "clock is advancing before siblings can run and every rig number is vacuous"
    )
    assert clock() == pytest.approx(max(walls.values()))


@pytest.mark.slow
@pytest.mark.rig
def test_rig_runs_a_real_pipelined_leg_in_virtual_time(tmp_path: Path) -> None:
    """The rig drives the REAL driver end to end and simulates hours in seconds.

    Every layer is production: real ``run_pipelined_leg``, real admission and
    wave pools, real ledger appends. Only the sessions are measured replays.
    The overlap floor asserted here (>1.2 at width 4) is the rig's own
    non-vacuousness bound, far below any scheduler target: today's tree
    measures ~2, and a rig artifact measures ~1.
    """

    result = run_rig(tmp_path, members=12, author_concurrency=4)
    assert result.status == "complete"
    assert result.author_sessions == 12
    assert result.virtual_elapsed > 3600.0, "12 measured members must simulate hours"
    assert result.first_gate_at is not None
    assert result.author_mean_concurrency > 1.2


@pytest.mark.slow
@pytest.mark.rig
def test_rung_scale_quick_rehearsal_holds_production_shaped_invariants(
    tmp_path: Path,
) -> None:
    """The scale-time-shaped-blindness instrument (W17 rig as a tree node).

    Bounded quick mode of the W17 virtual-time rung-scale rehearsal: the REAL
    driver (real ``run_pipelined_leg``, real admission, real wave pools, real
    ledger appends) at the production author width of 24 over the committed
    measured 200a duration distribution, 48-member intake -- a few minutes of
    real wall simulating over an hour of virtual scheduling. Asserts exactly
    the production-shaped invariants the blindness class names: completion,
    no-starvation, no-double-author.

    Calibration is the Wave-4 full-scale rehearsal artifact
    ``r4-w17-scheduler-rehearsal.json`` (200 members at width 24: all four
    production-shaped asserts green, virtual elapsed 21969.7s, mean author
    concurrency 6.53; the same rig against the pre-a2f6e938 scheduler caught
    the round-barrier defect). This bounded leg measures ~6.7 mean author
    concurrency at the same width; the floors below sit far beneath both
    measurements so the node is time-shaped without being throughput-flaky
    (throughput has its own strict-xfail TE-6 floor below).
    """

    driver, clock, ledger = build_rig_driver(
        tmp_path, members=48, author_concurrency=24
    )
    result = driver.run()
    guard = production_shaped_asserts(
        status=str(result.status), members=48, ledger=ledger
    )
    assert guard == {
        "driver_completed": True,
        "no_starvation_every_member_authored": True,
        "no_member_authored_twice": True,
    }, f"production-shaped invariants violated: {guard}"
    assert clock() > 3600.0, "48 measured members must simulate over an hour"
    assert ledger.lane_spans("checker"), "no gate session ran in the leg"
    assert ledger.mean_concurrency("author") > 2.0, (
        "width-24 authoring measured near-serial: the leg is not "
        "production-shaped (rig vacuousness or a scheduler regression)"
    )


@pytest.mark.smoke
def test_production_shaped_guard_turns_red_on_each_planted_violation() -> None:
    """Mutation proof for the scale-time-shaped-blindness instrument.

    The blindness manifest's law: an instrument counts only if disabling the
    guarded behavior turns its node red. Plant each violation the guard
    exists to catch -- a starved member, a double-authored member, a dead
    driver -- and require the guard to report exactly that violation. A guard
    edit that stops flagging any of them fails here before it can silently
    disarm the launch-gating node above. (The deep falsification ran in
    Wave-4: the same rig against the pre-a2f6e938 scheduler measured the
    round-barrier defect -- ``rig_catches_old_bug`` true in
    ``r4-w17-scheduler-rehearsal.json``.)
    """

    def ledger_of(*stable_ids: str) -> SessionLedger:
        ledger = SessionLedger()
        for index, stable_id in enumerate(stable_ids):
            started = float(index)
            ledger.record(
                SessionSpan(
                    stable_id=stable_id,
                    lane="author",
                    started_at=started,
                    ended_at=started + 1.0,
                )
            )
        return ledger

    healthy = production_shaped_asserts(
        status="complete", members=3, ledger=ledger_of("m_1", "m_2", "m_3")
    )
    assert healthy == {
        "driver_completed": True,
        "no_starvation_every_member_authored": True,
        "no_member_authored_twice": True,
    }

    starved = production_shaped_asserts(
        status="complete", members=3, ledger=ledger_of("m_1", "m_2")
    )
    assert starved["no_starvation_every_member_authored"] is False

    doubled = production_shaped_asserts(
        status="complete", members=3, ledger=ledger_of("m_1", "m_1", "m_2", "m_3")
    )
    assert doubled["no_member_authored_twice"] is False

    dead = production_shaped_asserts(
        status="failed", members=3, ledger=ledger_of("m_1", "m_2", "m_3")
    )
    assert dead["driver_completed"] is False

    empty = production_shaped_asserts(status="complete", members=3, ledger=ledger_of())
    assert empty["no_starvation_every_member_authored"] is False, (
        "an author lane that recorded nothing must read as starvation, "
        "never as vacuous green"
    )


@pytest.mark.slow
@pytest.mark.rig
@pytest.mark.xfail(
    strict=True,
    reason=(
        "TE-6 (RED-first by design): the residual round/wave-tail author "
        "barrier holds measured mean concurrency ~2-3 at width 12 on today's "
        "tree; this floor is the async author pool's acceptance instrument "
        "(M-12) and must stay strict-xfail until a scheduler change closes "
        "the barrier -- at which point the strict flag forces promotion to a "
        "hard gate in the same change"
    ),
)
def test_author_utilization_holds_the_te6_floor(tmp_path: Path) -> None:
    """TE-6: mean author concurrency >= 8 at width 12 over a measured load."""

    result = run_rig(tmp_path, members=40, author_concurrency=12)
    assert result.status == "complete"
    assert result.author_mean_concurrency >= AUTHOR_UTILIZATION_FLOOR
