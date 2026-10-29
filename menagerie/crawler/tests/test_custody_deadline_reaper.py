"""Custody deadline reaper + per-session checker wall: the twice-bitten wedge class.

Two production incidents, same class, one leg apart:

* **Leg 8 (2026-08-21):** an ``author_source`` session died in a provider storm
  WITHOUT settling; custody ``paid-spend-b3f3a...`` stayed bound forever, the
  wave admission relay waited forever, and graceful SIGTERM hung -- ``kill -9``
  was required.  Startup reconciliation handles never-bound rows, but a lease
  that goes bound-and-dead MID-LEG was reconciled by nothing until the next leg
  start, and even then settled ``local_outcome="unknown"`` -- chargeable.
  (Since item 9 / recovery F1.2, 2026-09-01, leg-start reconciliation settles
  a bound custodian the custody ladder proves dead as ``failed-custody-reaped``
  regardless of deadline; the residual ``unknown`` is the never-bound case.)
* **Leg 12 (02:59Z):** the campaign's FIRST sol ``metadata_batch`` session ran
  4.5 hours with no output then exited; custody ``paid-spend-b0cca1e48`` stayed
  bound and the writer waited hours.  Author sessions have a wall-grant plus a
  10% external kill; the checker lane's bound was derived from the PROVISIONAL
  operator deadline, had no named wall, no kill factor, and a wall-killed
  session settled ``"failed"`` -- chargeable, with no typed machine evidence.

These tests are the permanent regression suite for both halves:

* a custody binding whose custodian is machine-provably dead (flock FREE and
  process group EMPTY -- two independent kernel signals) past its durable
  reservation deadline plus a named grace FORCE-SETTLES with the closed machine
  infrastructure classification, at admission boundaries and at leg start;
* a LIVE custodian is NEVER reaped on time alone, and an UNANSWERABLE oracle is
  never read as death;
* the checker lane has a NAMED per-session wall that contains the operator
  wrapper's own typed attempt ladder, an external kill at wall x factor
  mirroring the author lane, and a wall-killed session carries a typed
  refundable infrastructure outcome the existing retry/refund machinery takes.
"""

from __future__ import annotations

import inspect
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from menagerie.crawler import driver_admission as driver_admission_module
from menagerie.crawler.constants import (
    AUTHOR_WALL_EXTERNAL_KILL_FACTOR,
    CUSTODY_REAPER_GRACE_SECONDS,
)
from menagerie.crawler.driver_contracts import (
    DriverIntegrationError,
    RetryableOperatorError,
    WorkItem,
)
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.operator_protocol import (
    CHECKER_SESSION_WALL_SECONDS,
    CHECKER_WALL_EXTERNAL_KILL_FACTOR,
    OPERATOR_DEADLINE_SECONDS,
    checker_session_kill_seconds,
)
from menagerie.crawler.paid_session_accounting import (
    ATTEMPT_REFUND_LOCAL_OUTCOMES,
    LOCAL_OUTCOMES,
)
from menagerie.crawler.paid_session_custody import (
    ALIVE,
    FREE,
    UNANSWERABLE,
    custodian_provably_dead,
    open_subtree_lock,
    subtree_liveness,
)
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    ProviderReconciliation,
    Reservation,
    authorized_paid_dispatch,
    load_paid_session_rows,
)
from menagerie.crawler.worker_supervisor import current_boot_id, process_start_token

_CAMPAIGN = "c1-mech"
_STABLE_ID = "m_reaper"
_WORK_ID = "work-m_reaper"

#: A clock far enough in the past that every minted reservation deadline plus
#: the reaper grace has long expired by the time the test observes it.
_PAST_CLOCK = "2026-08-19T00:00:00Z"


def _authority(tmp_path: Path, *, clock: str = _PAST_CLOCK) -> PaidSessionAuthority:
    """Return one temporary paid-session authority with a deterministic clock."""

    return PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: clock,
    )


def _checker_scope(attempt_group: str | None = None) -> dict[str, Any]:
    """Return one checker reservation scope carrying both accounting axes."""

    return {
        "campaign_id": _CAMPAIGN,
        "generation_id": "generation-open",
        "lane": "checker",
        "dispatch_kind": "metadata_batch",
        "attempt_group": attempt_group,
        "work_generations": ((_STABLE_ID, _WORK_ID),),
        "gate_kind": "metadata_batch",
        "checker_identity": None,
        "run_id": "run-one",
        "machine_id": "host-one",
    }


def _policy(*, reservation_wall_seconds: float = 100.0) -> BudgetPolicy:
    """Return a small deterministic budget policy."""

    return BudgetPolicy(
        quota_sessions=100,
        quota_wall_seconds=100_000.0,
        runaway_sessions=100,
        runaway_wall_seconds=100_000.0,
        reservation_wall_seconds=reservation_wall_seconds,
    )


def _constant_accept(key: str) -> ProviderReconciliation:
    """Constant-accept provider adapter: consummated, wall unmetered."""

    return ProviderReconciliation(True, None, f"sha256:receipt-{key[-12:]}")


def _rows_for(authority: PaidSessionAuthority, spend_id: str) -> dict[str, dict[str, Any]]:
    """Return this spend id's rows keyed by event kind."""

    return {
        str(row["event"]): row
        for row in load_paid_session_rows(authority.path)
        if row.get("spend_id") == spend_id
    }


def _bind_real_child(
    authority: PaidSessionAuthority, reservation: Reservation
) -> subprocess.Popen[bytes]:
    """Spawn one real flock-inheriting group leader and durably bind it."""

    handle = open_subtree_lock(reservation.lock_path)
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(handle.fd,),
        )
        token = process_start_token(child.pid)
        assert token is not None
        authority.bind_custody(
            reservation,
            leader_pid=child.pid,
            leader_pgid=child.pid,
            leader_start_token=token,
            boot_id=current_boot_id(),
            host_id="host-one",
            lock_device=handle.device,
            lock_inode=handle.inode,
        )
    finally:
        handle.close_parent_copy()
    return child


def _kill_and_reap(child: subprocess.Popen[bytes]) -> None:
    """Kill one bound child and wait until both kernel oracles read dead."""

    child.send_signal(signal.SIGKILL)
    child.wait(timeout=30.0)


def _reserve_bound_dead_lease(
    authority: PaidSessionAuthority,
) -> Reservation:
    """Mint one bound reservation whose custodian is machine-provably dead.

    The authoritative flock oracle is exercised for REAL: the child inherited
    the one live descriptor, so its death frees the lock.  The process-group
    enumeration shells out to ``ps`` and is unavailable on sandboxed hosts, so
    reaping tests force its answer through the module seam
    (``monkeypatch.setattr(driver_admission_module, "process_group_liveness",
    ...)``) -- the same idiom the custody topology matrix uses -- while the
    fail-closed UNANSWERABLE discipline keeps its own dedicated test.
    """

    reservation = authority.reserve(scope=_checker_scope(), policy=_policy())
    child = _bind_real_child(authority, reservation)
    _kill_and_reap(child)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if subtree_liveness(reservation.lock_path) == FREE:
            return reservation
        time.sleep(0.05)
    raise AssertionError("killed custody child never freed the inherited flock")


def _force_group_answer(monkeypatch: pytest.MonkeyPatch, answer: str) -> None:
    """Force the driver's process-group oracle to one closed-set answer."""

    monkeypatch.setattr(
        driver_admission_module, "process_group_liveness", lambda _pgid: answer
    )


class _ReaperRig(driver_admission_module.AdmissionEnvironmentMixin):
    """Production admission seams over one real money authority.

    ``_admit_checker_spend``, ``_reap_expired_dead_custody``, and
    ``_reconcile_paid_sessions_before_admission`` are inherited VERBATIM; only
    driver configuration reads are stubbed to fixed identities.
    """

    def __init__(self, authority: PaidSessionAuthority) -> None:
        """Bind the real valves' configuration reads to fixed identities."""

        self._authority = authority
        self.waited: list[float] = []
        self.config = SimpleNamespace(
            campaign_id=_CAMPAIGN, run_id="run-one", machine_id="host-one"
        )
        self._leg_generation = SimpleNamespace(generation_id="generation-open")
        self.dependencies = SimpleNamespace(
            boundary_hook=lambda *_args: None,
            provider_reconciler=_constant_accept,
            monotonic=time.monotonic,
            clock=lambda: "2026-08-24T12:00:00Z",
        )
        # D-62: real paid admission reads the sprint-pause sentinel under the
        # records root; this rig runs with a cleanly ABSENT sentinel (the one
        # state that never holds).
        self.paths = SimpleNamespace(
            ledgers=SimpleNamespace(
                models=Path("/nonexistent-rig-records/models/current-shard.jsonl")
            )
        )

    def _check_shutdown(self, *_args: object, **_kwargs: object) -> None:
        """No shutdown is pending in this rig."""

    def _paid_session_authority(self) -> PaidSessionAuthority:
        """Return the one real money authority."""

        return self._authority

    def _wait_before_infrastructure_retry(self, seconds: float) -> None:
        """Record the ladder's wait instead of sleeping."""

        self.waited.append(seconds)


def _checker_spend_scope() -> Any:
    """Return one production checker spend scope for the admission valve."""

    return driver_admission_module._CheckerSpendScope(
        gate_kind="metadata_batch",
        work_generations=((_STABLE_ID, _WORK_ID),),
        checker_identity=None,
    )


# ---------------------------------------------------------------------------
# B. The custody deadline reaper
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_bound_dead_custody_force_settles_typed_at_admission_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound-dead lease past deadline+grace settles typed at the next buy.

    Leg 12's shape: a checker custody stayed bound after its session died, and
    nothing before the next leg start could settle it.  The reaper runs at the
    writer's own admission valve, so the very next session buy force-settles
    the dead custody with the closed machine infrastructure classification and
    releases the binding -- unblocking every consumer that waits on the lease.
    """

    authority = _authority(tmp_path)
    dead = _reserve_bound_dead_lease(authority)
    _force_group_answer(monkeypatch, FREE)
    rig = _ReaperRig(authority)

    fresh = rig._admit_checker_spend(
        cast(JsonlLedger, SimpleNamespace(records=())), _checker_spend_scope()
    )

    assert fresh.spend_id != dead.spend_id
    rows = _rows_for(authority, dead.spend_id)
    assert "session_custody_released" in rows, (
        "the admission boundary must release a machine-provably-dead custody"
    )
    assert rows["session_custody_released"]["outcome"] == "verified-exit"
    assert "session_settled" in rows, (
        "the admission boundary must force-settle a machine-provably-dead custody"
    )
    settlement = rows["session_settled"]
    assert settlement["local_outcome"] == "failed-custody-reaped"
    assert settlement["machine"]["source"] == "writer-reservation-age"
    # The provider-accepted session and its machine wall stay charged; only the
    # ATTEMPT projection refunds (infrastructure deaths refund attempts, never
    # paid-session fuses).
    state = authority.state()
    assert authority.quota_sessions(_CAMPAIGN, _STABLE_ID, _WORK_ID, "checker") >= 1
    assert (
        state.quota_attempt_groups.get((_CAMPAIGN, _STABLE_ID, _WORK_ID, "checker"), 0)
        == 1
    ), "only the fresh admission's attempt group may count; the reaped one refunds"
    outstanding = {lease.spend_id for lease in authority.outstanding_custody_leases()}
    assert dead.spend_id not in outstanding


@pytest.mark.smoke
def test_live_custodian_is_never_reaped_on_time_alone(tmp_path: Path) -> None:
    """A live long session past its deadline is disclosed, never reaped.

    Dead-process evidence is REQUIRED: the flock is the authoritative oracle,
    and a subtree that still holds it is alive whatever the clock says.
    """

    authority = _authority(tmp_path)
    reservation = authority.reserve(scope=_checker_scope(), policy=_policy())
    child = _bind_real_child(authority, reservation)
    try:
        assert subtree_liveness(reservation.lock_path) == ALIVE
        rig = _ReaperRig(authority)
        reaped = rig._reap_expired_dead_custody()
        assert reaped == ()
        rows = _rows_for(authority, reservation.spend_id)
        assert "session_custody_released" not in rows
        assert "session_settled" not in rows
        outstanding = {
            lease.spend_id for lease in authority.outstanding_custody_leases()
        }
        assert reservation.spend_id in outstanding
    finally:
        _kill_and_reap(child)


@pytest.mark.smoke
def test_unanswerable_oracles_never_read_as_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UNANSWERABLE on either kernel oracle refuses the reap, fail-closed.

    Two independent signals are required (flock FREE and process group EMPTY);
    a host that cannot enumerate the group, or a replaced lock inode, leaves
    the lease disclosed instead of settled.
    """

    authority = _authority(tmp_path)
    dead = _reserve_bound_dead_lease(authority)
    rig = _ReaperRig(authority)

    _force_group_answer(monkeypatch, UNANSWERABLE)
    assert rig._reap_expired_dead_custody() == ()
    rows = _rows_for(authority, dead.spend_id)
    assert "session_settled" not in rows

    _force_group_answer(monkeypatch, FREE)
    monkeypatch.setattr(
        driver_admission_module, "subtree_liveness", lambda *_a, **_k: UNANSWERABLE
    )
    assert rig._reap_expired_dead_custody() == ()
    rows = _rows_for(authority, dead.spend_id)
    assert "session_settled" not in rows


@pytest.mark.smoke
def test_bound_dead_custody_inside_grace_is_not_reaped(tmp_path: Path) -> None:
    """Dead custodian before deadline+grace stays with the live settle path.

    The dispatch thread that owns the session settles within seconds of the
    kill escalation; the reaper only takes over once the named grace past the
    durable reservation deadline proves that thread is wedged or gone.
    """

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    authority = _authority(tmp_path, clock=now)
    reservation = authority.reserve(
        scope=_checker_scope(),
        policy=_policy(reservation_wall_seconds=3600.0),
    )
    child = _bind_real_child(authority, reservation)
    _kill_and_reap(child)
    rig = _ReaperRig(authority)
    assert rig._reap_expired_dead_custody() == ()
    rows = _rows_for(authority, reservation.spend_id)
    assert "session_settled" not in rows


@pytest.mark.smoke
def test_reaper_deadline_gate_reads_the_campaign_clock_never_the_host_wall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lease that is fresh on the campaign clock is never reaped, whatever
    the host wall clock says.

    ``deadline_at`` is minted by the campaign clock authority, so the gate
    must read the SAME clock. Reading the host wall clock voided the gate
    under any injected clock: the acceptance dry-run's frozen clock made
    every reservation "expired" at birth, so the reaper raced the owning
    dispatch thread's inter-guard custody window (source guard settled,
    author guard not yet begun: flock momentarily free, guard child's group
    empty) and force-settled a LIVE attempt's reservation. The owner's own
    settle then raised "already settled", classified session-crashed, and
    terminalized the model ``failed:author`` with no retry -- the
    session-death rehearsal's Wave-3-boundary red (2026-08-25). This pin
    freezes BOTH the authority and the rig on one campaign instant weeks in
    the host's past and shapes the lease exactly like the incident's
    (machine-provably dead: flock genuinely freed by the dead child, group
    answer forced FREE): on the campaign clock the deadline has not passed,
    so the reaper must leave the lease disclosed and untouched.
    """

    frozen = "2026-07-14T12:00:00Z"
    authority = _authority(tmp_path, clock=frozen)
    dead = _reserve_bound_dead_lease(authority)
    _force_group_answer(monkeypatch, FREE)
    rig = _ReaperRig(authority)
    rig.dependencies.clock = lambda: frozen
    assert rig._reap_expired_dead_custody() == ()
    rows = _rows_for(authority, dead.spend_id)
    assert "session_settled" not in rows, (
        "a lease that is fresh on the campaign clock must never be reaped on "
        "host wall-clock age"
    )
    outstanding = {lease.spend_id for lease in authority.outstanding_custody_leases()}
    assert dead.spend_id in outstanding, "the live lease must stay disclosed"


@pytest.mark.smoke
def test_leg_start_reconciliation_classifies_reaped_custody_as_machine_infra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leg-start reconciliation settles a bound-dead expired lease typed.

    Leg 8's shape: the driver was kill -9'd with custody bound.  Before the
    reaper, the next leg start released the binding but settled the money as
    ``local_outcome="unknown"`` -- chargeable.  The reaper runs FIRST inside
    the same reconciliation, so the settlement carries the closed machine
    infrastructure classification and the attempt refunds.
    """

    authority = _authority(tmp_path)
    dead = _reserve_bound_dead_lease(authority)
    _force_group_answer(monkeypatch, FREE)
    rig = _ReaperRig(authority)

    rig._reconcile_paid_sessions_before_admission()

    rows = _rows_for(authority, dead.spend_id)
    assert rows["session_custody_released"]["outcome"] == "verified-exit"
    assert rows["session_settled"]["local_outcome"] == "failed-custody-reaped"
    assert authority.outstanding_spend_ids() == ()
    assert (
        authority.state().quota_attempt_groups.get(
            (_CAMPAIGN, _STABLE_ID, _WORK_ID, "checker"), 0
        )
        == 0
    )


@pytest.mark.smoke
def test_reconciliation_settles_a_dead_custodian_inside_its_deadline_as_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 08-25 kill drill, restarted one second later (item 9, recovery F1.2).

    The pilot's ``paid-spend-b3f3a499`` shape: a bound author session died
    with the leg (kill -9), and the next leg start reconciled it. Before this
    fix the outcome depended on the OPERATOR's restart timing -- a lease still
    inside ``deadline_at + CUSTODY_REAPER_GRACE_SECONDS`` fell through the
    reaper's deadline gate to the generic provider-key ladder and settled
    ``local_outcome="unknown"`` at attempt weight 1, while waiting 35 minutes
    refunded it. Faster recovery was penalized. A custodian the two kernel
    oracles prove dead (flock FREE and group EMPTY) is dead whatever the
    clock says: reconciliation now settles it ``failed-custody-reaped``
    regardless of deadline, money first, then releases the binding. The
    deadline + grace stays as the MID-LEG probe schedule only.
    """

    # Reservation minted on the same clock the rig reads: the deadline
    # (clock + 100 s wall) has NOT expired, so the reaper's deadline gate skips.
    fresh = "2026-08-24T12:00:00Z"
    authority = _authority(tmp_path, clock=fresh)
    dead = _reserve_bound_dead_lease(authority)
    _force_group_answer(monkeypatch, FREE)
    rig = _ReaperRig(authority)
    rig.dependencies.clock = lambda: fresh
    assert rig._reap_expired_dead_custody() == (), "the mid-leg reaper must respect its deadline gate"

    rig._reconcile_paid_sessions_before_admission()

    rows = _rows_for(authority, dead.spend_id)
    assert rows["session_settled"]["local_outcome"] == "failed-custody-reaped", (
        "a provably dead custodian settled chargeable 'unknown' because the "
        "operator restarted inside the deadline"
    )
    assert rows["session_settled"]["machine"]["source"] == "writer-reservation-age"
    assert rows["session_custody_released"]["outcome"] == "verified-exit"
    assert authority.outstanding_spend_ids() == ()
    assert (
        authority.state().quota_attempt_groups.get(
            (_CAMPAIGN, _STABLE_ID, _WORK_ID, "checker"), 0
        )
        == 0
    )
    # Money invariants untouched: the provider-consummated session stays charged.
    assert authority.state().quota_by_lane[(_CAMPAIGN, _STABLE_ID, _WORK_ID, "checker")].sessions == 1


@pytest.mark.smoke
def test_reconciliation_settles_a_stale_boot_custodian_as_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bound lease from another boot cannot hold a process: reaped, not unknown.

    The whole-host reboot shape (the SSD crash, 2026-08-23). ``stale-boot`` is
    the custody ladder's own proof of death ("old boot cannot retain a local
    paid process or descriptor"), yet the money settled ``unknown`` and charged
    the attempt.
    """

    fresh = "2026-08-24T12:00:00Z"
    authority = _authority(tmp_path, clock=fresh)
    reservation = authority.reserve(scope=_checker_scope(), policy=_policy())
    authority.bind_custody(
        reservation,
        leader_pid=424242,
        leader_pgid=424242,
        leader_start_token="darwin-start-time:1787264490.0",
        boot_id=current_boot_id(),
        host_id="host-one",
        lock_device=16777234,
        lock_inode=688883454,
    )
    # The machine rebooted: the reservation's boot id (stamped at reserve time)
    # no longer matches the boot the reconciling driver observes.
    monkeypatch.setattr(
        driver_admission_module, "current_boot_id", lambda: "sha256:boot-after-the-ssd-crash"
    )
    _force_group_answer(monkeypatch, FREE)
    rig = _ReaperRig(authority)
    rig.dependencies.clock = lambda: fresh

    rig._reconcile_paid_sessions_before_admission()

    rows = _rows_for(authority, reservation.spend_id)
    assert rows["session_custody_released"]["outcome"] == "stale-boot"
    assert rows["session_settled"]["local_outcome"] == "failed-custody-reaped"
    assert authority.outstanding_spend_ids() == ()
    assert (
        authority.state().quota_attempt_groups.get(
            (_CAMPAIGN, _STABLE_ID, _WORK_ID, "checker"), 0
        )
        == 0
    )


@pytest.mark.smoke
def test_reconciliation_never_reaps_a_never_bound_or_live_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The widening is exactly the two proven-dead actions; nothing else moves.

    A never-bound reservation has no custodian to be dead (the startup ladder
    settles it ``unknown`` from the provider answer as before), and a LIVE
    custodian inside its deadline still refuses admission typed -- no
    settlement is ever written for it.
    """

    fresh = "2026-08-24T12:00:00Z"
    authority = _authority(tmp_path, clock=fresh)
    never_bound = authority.reserve(scope=_checker_scope(), policy=_policy())
    rig = _ReaperRig(authority)
    rig.dependencies.clock = lambda: fresh
    rig._reconcile_paid_sessions_before_admission()
    rows = _rows_for(authority, never_bound.spend_id)
    assert rows["session_settled"]["local_outcome"] == "unknown"
    assert "session_custody_released" not in rows

    # The live lease must be inside its deadline on the HOST clock too (the
    # custody ladder's ``past_deadline`` reads it), so its clock is real now.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    live_authority = _authority(tmp_path / "live", clock=now)
    live = live_authority.reserve(scope=_checker_scope(), policy=_policy())
    child = _bind_real_child(live_authority, live)
    try:
        _force_group_answer(monkeypatch, ALIVE)
        live_rig = _ReaperRig(live_authority)
        live_rig.dependencies.clock = lambda: now
        with pytest.raises(DriverIntegrationError, match="blocks admission"):
            live_rig._reconcile_paid_sessions_before_admission()
        assert "session_settled" not in _rows_for(live_authority, live.spend_id)
    finally:
        _kill_and_reap(child)


@pytest.mark.smoke
def test_custodian_provably_dead_is_two_signal_and_closed(tmp_path: Path) -> None:
    """The pure predicate demands BOTH kernel signals and a closed vocabulary."""

    assert custodian_provably_dead(FREE, FREE) is True
    for liveness, group in (
        (ALIVE, FREE),
        (FREE, ALIVE),
        (ALIVE, ALIVE),
        (UNANSWERABLE, FREE),
        (FREE, UNANSWERABLE),
        (UNANSWERABLE, UNANSWERABLE),
    ):
        assert custodian_provably_dead(liveness, group) is False
    with pytest.raises(ValueError):
        custodian_provably_dead("dead", FREE)
    with pytest.raises(ValueError):
        custodian_provably_dead(FREE, "gone")


@pytest.mark.smoke
def test_reaper_runs_at_every_admission_boundary_and_leg_start() -> None:
    """The reaper call sites are pinned textually in the production valves."""

    mixin = driver_admission_module.AdmissionEnvironmentMixin
    for method in (
        mixin._admit_checker_spend,
        mixin._admit_author_spend,
        mixin._reconcile_paid_sessions_before_admission,
    ):
        assert "_reap_expired_dead_custody(" in inspect.getsource(method), (
            f"{method.__name__} must run the custody deadline reaper"
        )


@pytest.mark.smoke
def test_reaper_skips_never_bound_leases(tmp_path: Path) -> None:
    """Never-bound reservations belong to startup reconciliation, not the reaper."""

    authority = _authority(tmp_path)
    reservation = authority.reserve(scope=_checker_scope(), policy=_policy())
    rig = _ReaperRig(authority)
    assert rig._reap_expired_dead_custody() == ()
    rows = _rows_for(authority, reservation.spend_id)
    assert "session_settled" not in rows
    assert "session_custody_released" not in rows


@pytest.mark.smoke
def test_reaper_adapter_outage_stays_disclosed_not_wedged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider-adapter outage skips the reap; the lease stays disclosed.

    The reaper writes the custody release only after the provider query
    answers, so an adapter outage leaves the WHOLE lease -- binding and money
    -- for the next boundary or leg start instead of stranding a released but
    unsettleable reservation.
    """

    authority = _authority(tmp_path)
    dead = _reserve_bound_dead_lease(authority)
    _force_group_answer(monkeypatch, FREE)
    rig = _ReaperRig(authority)
    rig.dependencies = SimpleNamespace(
        boundary_hook=lambda *_args: None,
        provider_reconciler=None,
        monotonic=time.monotonic,
        clock=lambda: "2026-08-24T12:00:00Z",
    )
    assert rig._reap_expired_dead_custody() == ()
    rows = _rows_for(authority, dead.spend_id)
    assert "session_settled" not in rows
    assert "session_custody_released" not in rows
    outstanding = {lease.spend_id for lease in authority.outstanding_custody_leases()}
    assert dead.spend_id in outstanding


# ---------------------------------------------------------------------------
# A. The per-session checker wall
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_checker_session_wall_ladder_is_strictly_ordered() -> None:
    """wall >= wrapper ladder < external kill <= durable reservation wall.

    The named wall must contain the operator wrapper's own granted attempt
    ladder (``OPERATOR_DEADLINE_SECONDS``), or the external kill would truncate
    a typed in-ladder outcome from outside -- the exact silent-truncation
    defect the author lane's constructor guard exists to refuse.
    """

    assert CHECKER_SESSION_WALL_SECONDS >= float(OPERATOR_DEADLINE_SECONDS)
    assert CHECKER_WALL_EXTERNAL_KILL_FACTOR == AUTHOR_WALL_EXTERNAL_KILL_FACTOR
    kill = checker_session_kill_seconds()
    assert kill == CHECKER_SESSION_WALL_SECONDS * CHECKER_WALL_EXTERNAL_KILL_FACTOR
    assert kill > CHECKER_SESSION_WALL_SECONDS
    # The reaper's grace sits on top of the durable reservation deadline, which
    # itself covers the kill bound, so a reaped lease has provably outlived
    # every layer of the lane's own kill ladder.
    assert CUSTODY_REAPER_GRACE_SECONDS > 0.0


@pytest.mark.smoke
def test_checker_reservation_wall_covers_the_external_kill(tmp_path: Path) -> None:
    """The durable custody deadline sits at or above the external kill bound.

    The reaper derives its deadline from the reservation row, so a lease that
    outlives ``deadline_at`` plus the grace has provably outlived every layer
    of the lane's own kill ladder.
    """

    authority = _authority(tmp_path)
    rig = _ReaperRig(authority)
    reservation = rig._admit_checker_spend(
        cast(JsonlLedger, SimpleNamespace(records=())), _checker_spend_scope()
    )
    row = _rows_for(authority, reservation.spend_id)["session_reserved"]
    assert float(row["reserved"]["wall_seconds"]) >= checker_session_kill_seconds()


@pytest.mark.smoke
def test_checker_wall_bound_derives_from_the_named_wall() -> None:
    """The lane's external bound derives from the named wall and kill factor."""

    bound = driver_admission_module._checker_wall_bound({}, grace_seconds=0.0)
    slack = driver_admission_module.CHECKER_LANE_BACKOFF_SLACK_SECONDS
    assert bound == pytest.approx(checker_session_kill_seconds() + slack)
    near_deadline = (
        datetime.now(timezone.utc) + timedelta(seconds=10.0)
    ).isoformat()
    clamped = driver_admission_module._checker_wall_bound(
        {"deadline_at": near_deadline}, grace_seconds=0.0
    )
    assert clamped <= 10.0 * CHECKER_WALL_EXTERNAL_KILL_FACTOR + slack + 1.0


@pytest.mark.smoke
def test_checker_wall_refund_vocabulary_is_closed_and_admitted() -> None:
    """Both new machine outcomes are closed-set members, auto-joined to LOCAL."""

    assert "failed-checker-session-wall" in ATTEMPT_REFUND_LOCAL_OUTCOMES
    assert "failed-custody-reaped" in ATTEMPT_REFUND_LOCAL_OUTCOMES
    assert "failed-checker-session-wall" in LOCAL_OUTCOMES
    assert "failed-custody-reaped" in LOCAL_OUTCOMES


@pytest.mark.heavy
def test_checker_wall_kills_a_hung_session_typed(tmp_path: Path) -> None:
    """A hung checker session dies at the wall with the typed refundable outcome.

    Leg 12's fix: the kill is external (the wrapper is exactly the thing that
    may have stopped making progress), the raised error is retryable transport
    the existing ladder takes, and it carries the closed machine outcome so the
    settlement refunds the attempt.
    """

    authority = _authority(tmp_path, clock="2026-08-20T03:45:26Z")
    reservation = authority.reserve(scope=_checker_scope(), policy=_policy())
    lane = driver_admission_module.CommandCheckerLane(
        (sys.executable, "-c", "import time; time.sleep(300)"),
        wall_grace_seconds=0.5,
    )
    expired = (
        datetime.now(timezone.utc) - timedelta(seconds=1.0)
    ).isoformat().replace("+00:00", "Z")
    envelope = {"deadline_at": expired, "gate_kind": "metadata_batch"}

    started = time.monotonic()
    with pytest.raises(RetryableOperatorError) as raised:
        with authorized_paid_dispatch(reservation):
            lane._run(envelope, tmp_path / "checker-root")
    elapsed = time.monotonic() - started
    assert elapsed < 60.0, "the external kill must fire at the wall, not at exit"

    error = raised.value
    assert getattr(error, "paid_session_local_outcome", None) == (
        "failed-checker-session-wall"
    )
    outcome = driver_admission_module._paid_session_infrastructure_local_outcome(error)
    assert outcome == "failed-checker-session-wall"

    authority.settle(
        reservation,
        provider_query=_constant_accept,
        custody_outcome="exited-error",
        machine_wall_seconds=elapsed,
        local_outcome=outcome,
    )
    settlement = _rows_for(authority, reservation.spend_id)["session_settled"]
    assert settlement["local_outcome"] == "failed-checker-session-wall"
    # Session and wall stay charged; only the attempt projection refunds.
    assert authority.quota_sessions(_CAMPAIGN, _STABLE_ID, _WORK_ID, "checker") == 1
    assert (
        authority.state().quota_attempt_groups.get(
            (_CAMPAIGN, _STABLE_ID, _WORK_ID, "checker"), 0
        )
        == 0
    )


@pytest.mark.smoke
def test_checker_wall_kill_retries_the_batch_and_refunds_the_attempt(
    tmp_path: Path,
) -> None:
    """The existing transport ladder retries a wall-killed batch, refunded.

    The stamped error mirrors ``CommandCheckerLane._run``'s wall-kill arm
    exactly (attribute names pinned by the heavy subprocess test above): the
    first dispatch dies at the wall, its settlement refunds the attempt, and
    the ladder's bounded transport retry re-buys and completes the batch.
    """

    authority = _authority(tmp_path)
    rig = _ReaperRig(authority)
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        if len(calls) == 1:
            error = RetryableOperatorError(
                "checker command exceeded its 1.0s wall bound and its process "
                "group was terminated: "
            )
            error.paid_session_local_outcome = "failed-checker-session-wall"
            error._paid_session_infrastructure_token = (
                driver_admission_module._PAID_SESSION_INFRASTRUCTURE_TOKEN
            )
            raise error
        return "gate-result"

    result = rig._retry_infrastructure_call(
        operation,
        admission=("checker", cast(WorkItem, SimpleNamespace(stable_id=_STABLE_ID))),
        operational=cast(JsonlLedger, SimpleNamespace(records=())),
        checker_scope=_checker_spend_scope(),
    )
    assert result == "gate-result"
    assert len(calls) == 2
    assert rig.waited, "the bounded transport ladder must back off before the retry"

    settlements = [
        row
        for row in load_paid_session_rows(authority.path)
        if row["event"] == "session_settled"
    ]
    assert [row["local_outcome"] for row in settlements] == [
        "failed-checker-session-wall",
        "completed",
    ]
    # One attempt group refunded (the wall kill), one charged (the completion).
    assert (
        authority.state().quota_attempt_groups.get(
            (_CAMPAIGN, _STABLE_ID, _WORK_ID, "checker"), 0
        )
        == 1
    )
