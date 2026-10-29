"""One reservation is ONE paid guard session: the 2026-08-20 rebind incident.

The first real paid step-200 leg (m3353) ran its author attempt's TWO guard
sessions -- ``source-request`` then ``author`` -- under the attempt's single
reservation.  The second guard's ``bind_custody`` appended a durable
``session_custody_bound`` row for a spend id whose custody was already bound
and released; the accounting fold then refused EVERY later read of the honest
ledger with ``duplicate custody``, three consecutive models failed at terminal
recording, and the driver's tripwire halted the leg.

Pilot evidence (``operational/spend.jsonl``, chain seq 2-5):

* ``session_reserved``          paid-spend-05f1dbd1...   03:45:26.420Z
* ``session_custody_bound``     same spend, pid 83080    03:45:26.457Z
* ``session_custody_released``  verified-exit            03:47:52.461Z
* ``session_custody_bound``     same spend, pid 83232    03:47:52.604Z

No settlement and no backoff between release and rebind: this was not the
bounded retry ladder (which settles each attempt and re-reserves) but the
second guard session of ONE attempt, 0.14s after the first exited.

These tests are the incident's permanent regression suite:

* the lifecycle writer refuses a rebind BEFORE any byte lands;
* the durability substrate folds BEFORE the durable write, so a fold-refused
  row can never poison an honest ledger again (any event kind, any writer bug);
* the accounting fold's own duplicate-custody refusal is pinned UNWEAKENED;
* the production dispatch seam buys one reservation per guard session,
  settling the finished session before the next reservation can exist;
* the retry ladder's per-attempt settle-then-re-reserve shape (which was
  already correct and was the incident's first suspect) is pinned too.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, cast

import pytest

from menagerie.crawler import driver as driver_module
from menagerie.crawler import driver_admission as driver_admission_module
from menagerie.crawler.author_spend import (
    author_session_allowance,
    chargeable_author_sessions,
)
from menagerie.crawler.constants import (
    AUTHOR_PAID_SESSIONS_PER_ATTEMPT,
    AUTHOR_SESSIONS_MAX,
)
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    AuthorCampaignTeardown,
    AuthorEffortCapExceeded,
    AuthorUsagePause,
    DriverIntegrationError,
    DriverShutdown,
    RetryableOperatorError,
    WorkItem,
)
from menagerie.crawler.paid_session_accounting import (
    ATTEMPT_REFUND_LOCAL_OUTCOMES,
    LANE_ATTEMPT_SESSION_BOUNDS,
    LANES,
    Weight,
    index_rows,
)
from menagerie.crawler.paid_session_durability import AnchoredLedger
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    ProviderReconciliation,
    Reservation,
    authorized_paid_dispatch,
    load_paid_session_rows,
)
from menagerie.crawler.recordio import JsonlLedger

pytestmark = pytest.mark.smoke

_EXIT_ZERO_WRAPPER = (sys.executable, "-c", "import sys; sys.exit(0)")


def _authority(tmp_path: Path) -> PaidSessionAuthority:
    """Return one temporary paid-session authority with a deterministic clock."""

    return PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-20T03:45:26Z",
    )


def _author_scope(
    stable_id: str, work_id: str, attempt_group: str | None = None
) -> dict[str, Any]:
    """Return one author reservation scope carrying both accounting axes."""

    return {
        "campaign_id": "c1-mech",
        "generation_id": "generation-open",
        "lane": "author",
        "dispatch_kind": "author_source",
        "attempt_group": attempt_group,
        "work_generations": ((stable_id, work_id),),
        "gate_kind": None,
        "checker_identity": None,
        "run_id": "run-one",
        "machine_id": "host-one",
    }


def _policy() -> BudgetPolicy:
    """Return an author-shaped policy sized for a multi-session attempt."""

    return BudgetPolicy(
        quota_sessions=100,
        quota_wall_seconds=100_000.0,
        runaway_sessions=100,
        runaway_wall_seconds=100_000.0,
        reservation_wall_seconds=100.0,
    )


def _constant_accept(key: str) -> ProviderReconciliation:
    """The step-200 constant-accept adapter: consummated, wall unmetered."""

    return ProviderReconciliation(True, None, f"sha256:receipt-{key[-12:]}")


def _events(authority: PaidSessionAuthority) -> list[tuple[str, str]]:
    """Return the ledger's (event, spend_id) sequence for order assertions."""

    return [
        (str(row["event"]), str(row.get("spend_id", "")))
        for row in load_paid_session_rows(authority.path)
    ]


def _work_item(**kwargs: Any) -> WorkItem:
    """Duck-typed WorkItem stand-in: only the attributes the ladder reads need to exist."""

    return cast(WorkItem, SimpleNamespace(**kwargs))


def _ledger(**kwargs: Any) -> JsonlLedger:
    """Duck-typed JsonlLedger stand-in for tests that never read past ``records``."""

    return cast(JsonlLedger, SimpleNamespace(**kwargs))


class _AuthorLadderStub(driver_admission_module.AdmissionEnvironmentMixin):
    """The exact production seams one author-lane retry ladder call touches.

    ``_retry_infrastructure_call``, ``_settle_paid_session``, and the paid
    dispatch scope are inherited VERBATIM; only admission plumbing that reads
    driver configuration is replaced with a real reservation against a real
    money authority.
    """

    def __init__(self, authority: PaidSessionAuthority) -> None:
        """Bind one real money authority behind the real settlement method."""

        self._authority = authority
        self.waited: list[float] = []
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

    def _admit_author_spend(
        self, item: Any, _operational: Any, attempt_group: str | None = None
    ) -> Reservation:
        """Reserve one author guard session exactly as production admission does."""

        return self._authority.reserve(
            scope=_author_scope(item.stable_id, item.active_work_id, attempt_group),
            policy=_policy(),
        )

    def _wait_before_infrastructure_retry(self, seconds: float) -> None:
        """Record the ladder's wait instead of sleeping."""

        self.waited.append(seconds)

    def _author_no_publication_class(self, _exc: BaseException) -> bool:
        """Keep exhaustion wrapping out of this rig."""

        return False

    def _note_author_lane_publication(self, _stable_id: str) -> None:
        """No lane-health bookkeeping in this rig."""


def _author_item() -> WorkItem:
    """Return the minimal work-item shape the ladder's author arm reads."""

    return _work_item(
        stable_id="m_two_guards", active_work_id="work-1", granted_attempts=None
    )


def _storm_retryable_error() -> RetryableOperatorError:
    """Return the pilot's exact executor-notice failure as the driver types it."""

    notice = {
        "type": "result",
        "subtype": "author-session-failure",
        "failure_class": "unclassified",
        "failure_class_basis": "none",
        "harness_is_error": True,
        "harness_subtype": "error_during_execution",
        "harness_terminal_reason": "aborted_streaming",
        "returncode": 143,
        "session_error_text_quarantined": None,
        "refusal_excerpt_quarantined": None,
    }
    with pytest.raises(RetryableOperatorError) as raised:
        driver_admission_module.classify_author_exit(
            "author", "m_two_guards", 75, json.dumps(notice), ""
        )
    return raised.value


class _ProductionAdmissionStub(_AuthorLadderStub):
    """Ladder rig whose admission valve is the REAL ``_admit_author_spend``.

    Only driver configuration is stubbed (campaign, run, machine, open leg
    generation); the allowance arithmetic, the attempt-group valve, the scope
    stamping, and the budget policy are production code verbatim.
    """

    _admit_author_spend = (
        driver_admission_module.AdmissionEnvironmentMixin._admit_author_spend  # type: ignore[assignment]
    )

    def __init__(self, authority: PaidSessionAuthority) -> None:
        """Bind the real valve's configuration reads to fixed identities."""

        super().__init__(authority)
        self.config = SimpleNamespace(
            campaign_id="c1-mech", run_id="run-one", machine_id="host-one"
        )
        self._leg_generation = SimpleNamespace(generation_id="generation-open")


def test_rebinding_custody_after_release_refuses_before_any_byte(
    tmp_path: Path,
) -> None:
    """The pilot's exact rebind now refuses typed and mints NO row.

    Before the fix the duplicate ``session_custody_bound`` row was durably
    appended first and only THEN refused by the fold, so the refusal poisoned
    every later read: settle, reserve, and terminal recording all raised
    ``duplicate custody`` and the leg halted.  The writer now refuses inside
    ``bind_custody`` with the fold's own vocabulary, before any byte lands,
    and the ledger stays fully readable and settleable.
    """

    authority = _authority(tmp_path)
    reservation = authority.reserve(
        scope=_author_scope("m3353", "work-3353"), policy=_policy()
    )
    authority.bind_custody(
        reservation,
        leader_pid=83080,
        leader_pgid=83080,
        leader_start_token="start-83080",
        boot_id="boot-one",
        host_id="host-one",
        lock_device=1,
        lock_inode=2,
    )
    authority.release_custody(reservation, outcome="verified-exit")
    with pytest.raises(ValueError, match="has duplicate custody"):
        authority.bind_custody(
            reservation,
            leader_pid=83232,
            leader_pgid=83232,
            leader_start_token="start-83232",
            boot_id="boot-one",
            host_id="host-one",
            lock_device=1,
            lock_inode=2,
        )
    # The refusal minted nothing: exactly one bound row is durable.
    rows = load_paid_session_rows(authority.path)
    assert [row["event"] for row in rows] == [
        "session_reserved",
        "session_custody_bound",
        "session_custody_released",
    ], "the refused rebind reached the durable ledger"
    # The ledger is NOT bricked: money reads and the settlement still work.
    assert authority.outstanding_spend_ids() == (reservation.spend_id,)
    authority.settle(
        reservation,
        provider_query=_constant_accept,
        custody_outcome="exited-error",
        machine_wall_seconds=146.0,
        local_outcome="failed",
    )
    assert authority.outstanding_spend_ids() == ()


def test_binding_custody_after_settlement_refuses_before_any_byte(
    tmp_path: Path,
) -> None:
    """A bind after the reservation settled refuses with the fold's vocabulary."""

    authority = _authority(tmp_path)
    reservation = authority.reserve(
        scope=_author_scope("m_late", "work-late"), policy=_policy()
    )
    authority.settle(
        reservation,
        provider_query=_constant_accept,
        custody_outcome="exited",
        machine_wall_seconds=1.0,
        local_outcome="completed",
    )
    with pytest.raises(ValueError, match="follows settlement"):
        authority.bind_custody(
            reservation,
            leader_pid=1,
            leader_pgid=1,
            leader_start_token="start-1",
            boot_id="boot-one",
            host_id="host-one",
            lock_device=1,
            lock_inode=2,
        )
    assert [row["event"] for row in load_paid_session_rows(authority.path)] == [
        "session_reserved",
        "session_settled",
    ]


def test_the_fold_still_refuses_a_durable_duplicate_custody_row(
    tmp_path: Path,
) -> None:
    """The accounting fold's duplicate-custody refusal is UNWEAKENED.

    The writer-side refusal above is defense in depth, not a replacement: a
    duplicate bound row already on disk (the pilot's exact residue shape --
    reserved, bound, released, bound) must still refuse on every replay, with
    zero producer input.  This is the tripwire working, not a defect.
    """

    authority = _authority(tmp_path)
    reservation = authority.reserve(
        scope=_author_scope("m_replay", "work-replay"), policy=_policy()
    )
    authority.bind_custody(
        reservation,
        leader_pid=83080,
        leader_pgid=83080,
        leader_start_token="start-83080",
        boot_id="boot-one",
        host_id="host-one",
        lock_device=1,
        lock_inode=2,
    )
    authority.release_custody(reservation, outcome="verified-exit")
    rows = load_paid_session_rows(authority.path)
    duplicate = dict(rows[1])
    assert duplicate["event"] == "session_custody_bound"
    with pytest.raises(ValueError, match="has duplicate custody"):
        index_rows([*rows, duplicate])


def test_a_fold_refused_row_never_becomes_durable(tmp_path: Path) -> None:
    """The durability substrate folds BEFORE the durable write.

    The incident's poison mechanics: ``AnchoredLedger.append`` used to fold the
    row only after the anchor intent, the write, and the promote -- so a row the
    fold refuses landed durably and every later read of the honest ledger
    refused.  The fold now runs before any durable byte, inside the same
    exclusive lock, closing the whole class for every event kind and every
    future writer bug.
    """

    def fold(value: list[dict[str, Any]], rows: Any) -> list[dict[str, Any]]:
        """Refuse any row carrying ``poison``."""

        for row in rows:
            if row.get("poison"):
                raise ValueError("the fold refuses this row")
        return [*value, *map(dict, rows)]

    ledger = AnchoredLedger(
        tmp_path / "ledger.jsonl",
        chain_scope="menagerie.crawler.tests.fold-first",
        subject="fold-first-test",
        seed=list,
        fold=fold,
        projection="menagerie.crawler.tests.fold-first",
    )
    ledger.append(lambda _snapshot: {"event": "fine", "sequence": 1})
    with pytest.raises(ValueError, match="the fold refuses this row"):
        ledger.append(lambda _snapshot: {"event": "bad", "poison": True})
    snapshot = ledger.read_locked()
    assert [row["event"] for row in snapshot.rows] == ["fine"], (
        "a fold-refused row reached the durable ledger"
    )
    anchor = ledger.read_anchor()
    assert anchor is not None and anchor.intent is None
    assert anchor.committed.rows == 1
    # The ledger is not poisoned: the next honest append still lands.
    ledger.append(lambda _snapshot: {"event": "fine", "sequence": 2})
    assert len(ledger.read_locked().rows) == 2


def test_the_author_attempt_buys_one_reservation_per_guard_session(
    tmp_path: Path,
) -> None:
    """The production seam: two guard sessions, two reservations, in order.

    This is tonight's flow made green: one ladder attempt whose operation runs
    the real ``_run_operator_command`` twice (source-request then author).  The
    finished first session SETTLES before the second reservation exists, so the
    ledger's verified chain head moves and the second admission mints a
    different spend id and provider idempotency key -- reservation reuse is
    impossible by construction.  Before the fix this exact sequence appended
    the duplicate bound row and bricked every later money read.
    """

    authority = _authority(tmp_path)
    stub = _AuthorLadderStub(authority)

    def operation() -> str:
        """One author attempt: source-request guard, then author guard."""

        for _ in range(2):
            driver_admission_module._run_operator_command(  # noqa: SLF001
                _EXIT_ZERO_WRAPPER, timeout_seconds=30.0, env=None
            )
        return "authored"

    result = stub._retry_infrastructure_call(  # noqa: SLF001
        operation,
        admission=("author", _author_item()),
        operational=_ledger(records=()),
    )
    assert result == "authored"
    events = _events(authority)
    spend_ids = [spend for event, spend in events if event == "session_reserved"]
    assert len(spend_ids) == 2 and len(set(spend_ids)) == 2, (
        "each guard session must buy its own reservation"
    )
    first, second = spend_ids
    assert events == [
        ("session_reserved", first),
        ("session_custody_bound", first),
        ("session_custody_released", first),
        ("session_settled", first),
        ("session_reserved", second),
        ("session_custody_bound", second),
        ("session_custody_released", second),
        ("session_settled", second),
    ], (
        "the finished guard session must settle BEFORE the next reservation "
        f"exists; observed {events}"
    )
    rows = load_paid_session_rows(authority.path)
    keys = {
        str(row["provider_idempotency_key"])
        for row in rows
        if row["event"] == "session_reserved"
    }
    assert len(keys) == 2, "two guard sessions shared one provider idempotency key"
    assert authority.outstanding_spend_ids() == ()
    assert authority.outstanding_custody_leases() == ()
    state = authority.state()
    charged = state.quota_by_lane.get(
        ("c1-mech", "m_two_guards", "work-1", "author")
    )
    assert charged is not None and charged.sessions == 2, (
        "the QUOTA instrument must count both consummated provider sessions"
    )
    # Round-2 F1: the two guard sessions share ONE durable attempt group -- the
    # second reservation joins the first's group, so the attempt unit stays 1.
    second_row = next(
        row
        for row in rows
        if row["event"] == "session_reserved" and row["spend_id"] == second
    )
    assert second_row["attempt_group"] == first, (
        "the second guard session must durably join the attempt's group head"
    )
    assert state.attempt_groups == {first: (first, second)}
    assert state.quota_attempt_groups.get(
        ("c1-mech", "m_two_guards", "work-1", "author")
    ) == 1, "one attempt, however many guard sessions, is ONE attempt group"


def test_the_retry_ladder_settles_each_attempt_before_reserving_the_next(
    tmp_path: Path,
) -> None:
    """A failed attempt settles, then the retry mints a FRESH reservation.

    This was the incident brief's first suspect.  The ladder already had this
    shape (per-attempt admission and settlement, so the pilot's rebind came
    from the two-guard attempt, not from a retry) -- pinned here permanently so
    a future ladder change cannot regress into reservation reuse either.
    """

    authority = _authority(tmp_path)
    stub = _AuthorLadderStub(authority)
    calls = {"count": 0}

    def operation() -> str:
        """Fail the first session the way a transient wrapper failure fails."""

        calls["count"] += 1
        if calls["count"] == 1:
            raise RetryableOperatorError("author command failed: transient blip")
        return "authored"

    result = stub._retry_infrastructure_call(  # noqa: SLF001
        operation,
        admission=("author", _author_item()),
        operational=_ledger(records=()),
    )
    assert result == "authored"
    assert stub.waited, "the ladder must wait between attempts"
    events = _events(authority)
    spend_ids = [spend for event, spend in events if event == "session_reserved"]
    assert len(spend_ids) == 2 and len(set(spend_ids)) == 2
    first, second = spend_ids
    assert events == [
        ("session_reserved", first),
        ("session_settled", first),
        ("session_reserved", second),
        ("session_settled", second),
    ], f"the failed attempt must settle before the retry reserves; got {events}"
    rows = load_paid_session_rows(authority.path)
    failed = next(row for row in rows if row["spend_id"] == first and row["event"] == "session_settled")
    assert failed["custody_outcome"] == "exited-error"
    assert failed["local_outcome"] == "failed"
    assert authority.outstanding_spend_ids() == ()


@pytest.mark.smoke
def test_storm_sessions_charge_fuses_but_not_attempt_allowance(tmp_path: Path) -> None:
    """Exact rc-143 storm rows retry as sessions and refund only attempt weight."""

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    calls = {"count": 0}

    def operation() -> str:
        """Replay three pilot deaths, then publish on the fourth session."""

        calls["count"] += 1
        if calls["count"] <= 3:
            raise _storm_retryable_error()
        return "authored"

    result = stub._retry_infrastructure_call(  # noqa: SLF001
        operation,
        admission=("author", _author_item()),
        operational=_ledger(records=()),
    )
    assert result == "authored"
    assert calls["count"] == 4, "machine-classified deaths must outlive the old 3-attempt ladder"
    assert len(stub.waited) == 3
    assert stub.waited != [5.0, 30.0, 30.0], "session retries require bounded jitter"

    state = authority.state()
    coordinate = ("c1-mech", "m_two_guards", "work-1", "author")
    # The monetary/runaway fuses still see every provider-consummated session.
    assert state.quota_by_lane[coordinate].sessions == 4
    assert state.runaway_by_lane[("c1-mech", "generation-open", "author")].sessions == 4
    # Only the successful content-bearing attempt consumes effort allowance.
    assert state.quota_attempt_groups[coordinate] == 1
    settlements = [
        row for row in load_paid_session_rows(authority.path) if row["event"] == "session_settled"
    ]
    assert [row["local_outcome"] for row in settlements[:3]] == [
        "failed-harness-aborted-streaming-sigterm"
    ] * 3
    assert settlements[-1]["local_outcome"] == "completed"
    assert "failed-harness-aborted-streaming-sigterm" in ATTEMPT_REFUND_LOCAL_OUTCOMES


@pytest.mark.smoke
def test_campaign_halt_teardown_refunds_and_reselects_on_resume(tmp_path: Path) -> None:
    """An in-flight halt kill neither retries nor consumes the model attempt."""

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    item = _author_item()
    calls = {"count": 0}
    teardown_record = {
        "target_pid": 5101,
        "target_pgid": 5101,
        "signal": 15,
        "reason": "campaign halt while author wave was in flight",
        "teardown_trigger": "breaker-halt",
        "outcome": "signalled",
    }

    def halted_operation() -> str:
        """Replay the sibling session interrupted by the campaign itself."""

        calls["count"] += 1
        raise AuthorCampaignTeardown(
            trigger="breaker-halt",
            teardown_record=teardown_record,
            stderr_tail="stderr-before-breaker",
        )

    with pytest.raises(AuthorCampaignTeardown):
        stub._retry_infrastructure_call(  # noqa: SLF001
            halted_operation,
            admission=("author", item),
            operational=_ledger(records=()),
        )

    coordinate = ("c1-mech", item.stable_id, item.active_work_id, "author")
    state = authority.state()
    assert calls["count"] == 1, "campaign teardown must bypass every retry ladder"
    assert stub.waited == []
    assert state.quota_attempt_groups.get(coordinate) == 0
    settlements = [
        row for row in load_paid_session_rows(authority.path) if row["event"] == "session_settled"
    ]
    assert settlements[-1]["local_outcome"] == "campaign-teardown"

    # Resume can immediately select/buy the same model; no terminal or attempt
    # allowance was consumed by the prior leg's own teardown.
    assert (
        stub._retry_infrastructure_call(  # noqa: SLF001
            lambda: "authored-on-resume",
            admission=("author", item),
            operational=_ledger(records=()),
        )
        == "authored-on-resume"
    )
    assert authority.state().quota_attempt_groups[coordinate] == 1
    assert "campaign-teardown" in ATTEMPT_REFUND_LOCAL_OUTCOMES


@pytest.mark.smoke
def test_inflight_wave_halt_tears_down_without_author_failures(tmp_path: Path) -> None:
    """A real two-session halt settles both as resumable campaign teardown."""

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    operational = _ledger(records=())
    items = [
        _work_item(
            stable_id=f"m_halt_{index}",
            active_work_id=f"work-halt-{index}",
            granted_attempts=None,
        )
        for index in range(2)
    ]
    request_paths = {
        item.stable_id: tmp_path / item.stable_id / "request.json" for item in items
    }
    for path in request_paths.values():
        path.parent.mkdir(parents=True)
        path.write_text("{}\n", encoding="utf-8")

    def dispatch(item: Any, request_admission: Any) -> str:
        """Run one real paid guard until the wave controller halts it."""

        command = (
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            str(request_paths[item.stable_id]),
        )
        return stub._retry_infrastructure_call(  # noqa: SLF001
            lambda: driver_admission_module._run_operator_command(  # noqa: SLF001
                command,
                timeout_seconds=90.0,
                env=None,
            ).stdout,
            admission=("author", item),
            operational=operational,
            author_admission=request_admission,
        )

    pool = driver_admission_module._AuthorWavePool(  # noqa: SLF001
        2,
        cast(
            Callable[[WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact],
            dispatch,
        ),
        lambda item, attempt_group: stub._admit_author_spend(  # noqa: SLF001
            item,
            operational,
            attempt_group=attempt_group,
        ),
    )
    for item in items:
        pool.submit(item)
    deadline = time.monotonic() + 30.0
    while True:
        pool._service_admission_requests()  # noqa: SLF001
        rows = load_paid_session_rows(authority.path) if authority.path.exists() else ()
        if len([row for row in rows if row["event"] == "session_custody_bound"]) == 2:
            break
        assert time.monotonic() < deadline, "both paid guards must enter custody before halt"
        time.sleep(0.01)

    assert pool.halt(
        trigger="retry-exhaustion-halt",
        reason="m12412 retry exhaustion halted the author wave",
    ) == ()
    drained = dict(pool.drain())
    pool.close()

    assert set(drained) == {item.stable_id for item in items}
    assert all(isinstance(outcome.error, AuthorCampaignTeardown) for outcome in drained.values())
    state = authority.state()
    for item in items:
        coordinate = ("c1-mech", item.stable_id, item.active_work_id, "author")
        assert state.quota_attempt_groups.get(coordinate) == 0
    settlements = [
        row for row in load_paid_session_rows(authority.path) if row["event"] == "session_settled"
    ]
    assert [row["local_outcome"] for row in settlements] == [
        "campaign-teardown",
        "campaign-teardown",
    ]
    records = list(tmp_path.glob("m_halt_*/campaign-teardown-*.json"))
    assert len(records) == 2
    for path in records:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["teardown_trigger"] == "retry-exhaustion-halt"
        assert record["signal"] == signal.SIGTERM
        assert record["target_pid"] == record["target_pgid"]


@pytest.mark.smoke
def test_storm_session_retries_stop_at_the_paid_quota_fuse(tmp_path: Path) -> None:
    """Zero attempt weight never makes an infrastructure storm unbounded."""

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)

    def operation() -> str:
        """Die with exact machine infrastructure evidence on every session."""

        raise _storm_retryable_error()

    with pytest.raises(AuthorEffortCapExceeded, match="QUOTA"):
        stub._retry_infrastructure_call(  # noqa: SLF001
            operation,
            admission=("author", _author_item()),
            operational=_ledger(records=()),
        )
    state = authority.state()
    coordinate = ("c1-mech", "m_two_guards", "work-1", "author")
    assert state.quota_by_lane[coordinate].sessions == 6
    assert state.quota_attempt_groups[coordinate] == 0
    assert len(stub.waited) == 6


@pytest.mark.smoke
def test_producer_influenceable_claims_cannot_refund_an_attempt_group(tmp_path: Path) -> None:
    """Forged prose and author-influenceable timeout stay chargeable."""

    authority = _authority(tmp_path)
    stub = _AuthorLadderStub(authority)
    calls = {"count": 0}

    def operation() -> str:
        """Fail once with producer-controlled prose, then succeed."""

        calls["count"] += 1
        if calls["count"] == 1:
            raise RetryableOperatorError(
                "author says harness_is_error=true terminal_reason=aborted_streaming rc=143"
            )
        return "authored"

    assert (
        stub._retry_infrastructure_call(  # noqa: SLF001
            operation,
            admission=("author", _author_item()),
            operational=_ledger(records=()),
        )
        == "authored"
    )
    state = authority.state()
    coordinate = ("c1-mech", "m_two_guards", "work-1", "author")
    assert state.quota_attempt_groups[coordinate] == 2
    settlements = [
        row for row in load_paid_session_rows(authority.path) if row["event"] == "session_settled"
    ]
    assert settlements[0]["local_outcome"] == "failed"

    timeout_authority = _authority(tmp_path / "timeout")
    timeout_stub = _AuthorLadderStub(timeout_authority)
    timeout_calls = {"count": 0}

    def timeout_operation() -> str:
        """Raise one producer-influenceable timeout, then succeed."""

        timeout_calls["count"] += 1
        if timeout_calls["count"] == 1:
            try:
                raise subprocess.TimeoutExpired("claude", 30.0)
            except subprocess.TimeoutExpired as exc:
                raise RetryableOperatorError("author session timed out") from exc
        return "authored"

    assert (
        timeout_stub._retry_infrastructure_call(  # noqa: SLF001
            timeout_operation,
            admission=("author", _author_item()),
            operational=_ledger(records=()),
        )
        == "authored"
    )
    timeout_state = timeout_authority.state()
    assert timeout_state.quota_attempt_groups[coordinate] == 2


def _storm_529_error() -> RetryableOperatorError:
    """Return m4294's exact supplement-stage 529 death as the driver types it.

    The 2026-08-20 Anthropic 529 storm: the source-request guard completed,
    the supplement continuation died ``api_error``, and the executor's notice
    carried the full machine envelope (``harness_is_error``, terminal reason,
    returncode) alongside the text-derived diagnostic class.
    """

    notice = {
        "type": "result",
        "subtype": "author-session-failure",
        "failure_class": "provider-overloaded",
        "failure_class_basis": "diagnostic-text:overload-marker",
        "harness_is_error": True,
        "harness_subtype": "success",
        "harness_terminal_reason": "api_error",
        "returncode": 1,
        "session_error_text_quarantined": "API Error: Repeated 529 Overloaded errors.",
        "refusal_excerpt_quarantined": None,
    }
    with pytest.raises(RetryableOperatorError) as raised:
        driver_admission_module.classify_author_exit(
            "author", "m_two_guards", 75, json.dumps(notice), ""
        )
    return raised.value


def test_a_terminal_infra_death_refunds_past_a_completed_guard(tmp_path: Path) -> None:
    """A two-guard attempt whose SECOND session dies infra refunds its group.

    The 2026-08-20 refund gap: m4294/m4295 burned their allowance on attempts
    whose source-request guard COMPLETED and whose author/supplement session
    then died with machine infrastructure evidence.  The group refund keyed on
    every member weighing zero, so the completed source session blocked the
    refund the single-session storm arm already granted.  Members settle
    strictly in sequence (a join is admitted only after every earlier member
    settled), so a zero-weight settlement IS the attempt's terminal
    disposition and refunds the whole group.
    """

    key = ("c1-mech", "m_two_guards", "work-1", "author")
    for infra_outcome in (
        "failed-harness-api-error",
        "failed-harness-aborted-streaming-sigterm",
    ):
        authority = _authority(tmp_path / infra_outcome)
        head = authority.reserve(
            scope=_author_scope("m_two_guards", "work-1"), policy=_policy()
        )
        authority.settle(
            head,
            provider_query=_constant_accept,
            custody_outcome="exited",
            machine_wall_seconds=30.0,
            local_outcome="completed",
        )
        join = authority.reserve(
            scope=_author_scope("m_two_guards", "work-1", attempt_group=head.spend_id),
            policy=_policy(),
        )
        assert authority.state().quota_attempt_groups[key] == 1
        authority.settle(
            join,
            provider_query=_constant_accept,
            custody_outcome="exited-error",
            machine_wall_seconds=30.0,
            local_outcome=infra_outcome,
        )
        state = authority.state()
        assert state.quota_attempt_groups.get(key) == 0, (
            f"a terminal {infra_outcome} death must refund the attempt group "
            "even past a completed source-request guard session"
        )
        # The money/fuse instruments never refund: both provider-consummated
        # sessions and their walls stay charged to QUOTA and RUNAWAY.
        assert state.quota_by_lane[key].sessions == 2
        assert state.quota_by_lane[key].wall_seconds == 60.0
        assert (
            state.runaway_by_lane[("c1-mech", "generation-open", "author")].sessions == 2
        )


def test_a_healthy_session_contract_violation_still_consumes_the_attempt(
    tmp_path: Path,
) -> None:
    """A completed-session malformed result keeps its full attempt charge.

    The precision half of the refund rule: the storm window's
    m4388/m4413/m4713 terminals settled every session ``completed`` with NO
    machine infrastructure evidence -- genuine author contract violations.  A
    generic ``failed`` settlement (the driver's conservative disclosure for
    every death without the machine-derived token) must never refund.
    """

    key = ("c1-mech", "m_two_guards", "work-1", "author")
    for terminal_outcome in ("failed", "completed", "unknown"):
        authority = _authority(tmp_path / terminal_outcome)
        head = authority.reserve(
            scope=_author_scope("m_two_guards", "work-1"), policy=_policy()
        )
        authority.settle(
            head,
            provider_query=_constant_accept,
            custody_outcome="exited",
            machine_wall_seconds=30.0,
            local_outcome="completed",
        )
        join = authority.reserve(
            scope=_author_scope("m_two_guards", "work-1", attempt_group=head.spend_id),
            policy=_policy(),
        )
        authority.settle(
            join,
            provider_query=_constant_accept,
            custody_outcome="exited-error" if terminal_outcome != "completed" else "exited",
            machine_wall_seconds=30.0,
            local_outcome=terminal_outcome,
        )
        assert authority.state().quota_attempt_groups[key] == 1, (
            f"a {terminal_outcome!r} terminal settlement carries no machine "
            "infrastructure evidence and must keep consuming the attempt"
        )


def test_supplement_stage_storm_death_refunds_the_attempt_group(tmp_path: Path) -> None:
    """The pilot's exact supplement-stage 529 death refunds through the ladder.

    Production seams end to end: the source-request guard completes (settling
    ``completed`` and joining the fresh reservation to the attempt group), the
    continuation dies with m4294's exact executor notice, and the ladder's
    weather arm retries.  Only the successful attempt may consume allowance;
    every provider-consummated session still charges the QUOTA/RUNAWAY fuses.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    calls = {"count": 0}

    def operation() -> str:
        """Complete the source guard, then die 529 once at the continuation."""

        calls["count"] += 1
        for _ in range(2):
            driver_admission_module._run_operator_command(  # noqa: SLF001
                _EXIT_ZERO_WRAPPER, timeout_seconds=30.0, env=None
            )
        if calls["count"] == 1:
            raise _storm_529_error()
        return "authored"

    result = stub._retry_infrastructure_call(  # noqa: SLF001
        operation,
        admission=("author", _author_item()),
        operational=_ledger(records=()),
    )
    assert result == "authored"
    assert calls["count"] == 2
    settlements = [
        row
        for row in load_paid_session_rows(authority.path)
        if row["event"] == "session_settled"
    ]
    assert [row["local_outcome"] for row in settlements] == [
        "completed",
        "failed-harness-api-error",
        "completed",
        "completed",
    ], "the supplement-stage death must settle with its machine-derived outcome"
    state = authority.state()
    coordinate = ("c1-mech", "m_two_guards", "work-1", "author")
    assert state.quota_by_lane[coordinate].sessions == 4
    assert state.runaway_by_lane[("c1-mech", "generation-open", "author")].sessions == 4
    assert state.quota_attempt_groups[coordinate] == 1, (
        "the storm-killed attempt must refund; only the successful attempt consumes"
    )


def test_a_second_guard_without_per_session_admission_refuses_before_spawn(
    tmp_path: Path,
) -> None:
    """A consume-once lane (the checker shape) refuses a second guard typed.

    The checker pool worker declares no per-session admission because a checker
    dispatch runs exactly one guard per reservation.  If that invariant ever
    breaks, the refusal fires BEFORE any child spawns: no provider byte, no
    custody row, no poisoned ledger.
    """

    authority = _authority(tmp_path)
    reservation = authority.reserve(
        scope=_author_scope("m_once", "work-once"), policy=_policy()
    )
    scope = driver_admission_module._PaidDispatchSessions(  # noqa: SLF001
        reservation, admit=None, settle=None, span_started=time.monotonic()
    )
    with (
        authorized_paid_dispatch(reservation),
        driver_admission_module._paid_dispatch_sessions(scope),  # noqa: SLF001
    ):
        driver_admission_module._run_operator_command(  # noqa: SLF001
            _EXIT_ZERO_WRAPPER, timeout_seconds=30.0, env=None
        )
        with pytest.raises(DriverIntegrationError, match="second paid guard session"):
            driver_admission_module._run_operator_command(  # noqa: SLF001
                _EXIT_ZERO_WRAPPER, timeout_seconds=30.0, env=None
            )
    assert [row["event"] for row in load_paid_session_rows(authority.path)] == [
        "session_reserved",
        "session_custody_bound",
        "session_custody_released",
    ], "the refused second guard must mint nothing"
    # The reservation is still settleable: nothing was poisoned.
    authority.settle(
        reservation,
        provider_query=_constant_accept,
        custody_outcome="exited",
        machine_wall_seconds=1.0,
        local_outcome="completed",
    )
    assert authority.outstanding_spend_ids() == ()


def test_mid_operation_admission_failures_still_settle_every_reservation(
    tmp_path: Path,
) -> None:
    """If the second admission refuses, the first session is already settled.

    The allowance ladder still bounds total buys: a mid-operation admission
    refusal (budget exhausted, adapter outage) propagates out of the attempt,
    and no reservation is left unsettled or double-settled behind it.
    """

    authority = _authority(tmp_path)
    stub = _AuthorLadderStub(authority)
    minted = {"count": 0}
    real_admit = stub._admit_author_spend

    def admit_once(
        item: Any, operational: Any, attempt_group: str | None = None
    ) -> Reservation:
        """Admit the first session, refuse the second like a spent budget."""

        minted["count"] += 1
        if minted["count"] > 1:
            raise DriverIntegrationError("author session budget exhausted in-flight")
        return real_admit(item, operational, attempt_group)

    stub._admit_author_spend = admit_once  # type: ignore[method-assign, assignment]

    def operation() -> str:
        """Run two guard sessions; the second admission refuses."""

        for _ in range(2):
            driver_admission_module._run_operator_command(  # noqa: SLF001
                _EXIT_ZERO_WRAPPER, timeout_seconds=30.0, env=None
            )
        return "authored"

    with pytest.raises(DriverIntegrationError, match="budget exhausted in-flight"):
        stub._retry_infrastructure_call(  # noqa: SLF001
            operation,
            admission=("author", _author_item()),
            operational=_ledger(records=()),
        )
    events = _events(authority)
    settled = [spend for event, spend in events if event == "session_settled"]
    reserved = [spend for event, spend in events if event == "session_reserved"]
    assert reserved and settled == reserved, (
        f"every minted reservation must settle exactly once; got {events}"
    )
    assert authority.outstanding_spend_ids() == ()


def test_the_attempt_allowance_prices_attempts_not_guard_sessions(
    tmp_path: Path,
) -> None:
    """Sol F1: source-only failed attempts exhaust at A, never at 2*A.

    Under session-unit pricing (round 1's ``quota_sessions = 2*A``), a
    source-only failure consumed ONE guard session, so one work generation
    could buy up to ``2*A`` such billable attempts.  The durable attempt-group
    valve now refuses the (A+1)th NEW attempt while the raw session QUOTA
    still has headroom -- proving the refusal is the ATTEMPT unit, not the
    session ceiling.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    item = _author_item()
    operational = _ledger(records=())
    allowance = author_session_allowance(item.granted_attempts)
    assert allowance == AUTHOR_SESSIONS_MAX
    for _ in range(allowance):
        # A source-only failure: the attempt's ONE guard session (the
        # source-request) runs and fails; the attempt settles and ends.
        reservation = stub._admit_author_spend(item, operational)  # noqa: SLF001
        stub._settle_paid_session(  # noqa: SLF001
            reservation, elapsed_seconds=1.0, completed=False
        )
    state = authority.state()
    key = ("c1-mech", item.stable_id, item.active_work_id, "author")
    assert state.quota_attempt_groups.get(key) == allowance
    consumed_sessions = cast(Weight, state.quota_by_lane.get(key)).sessions
    assert consumed_sessions == allowance < 2 * allowance, (
        "the raw session QUOTA must still have headroom, so the refusal below "
        "can only come from the attempt-group valve"
    )
    with pytest.raises(AuthorEffortCapExceeded, match="attempt"):
        stub._admit_author_spend(item, operational)  # noqa: SLF001
    # The refusal minted no row: the ledger still holds exactly A reservations.
    reserved = [event for event, _spend in _events(authority) if event == "session_reserved"]
    assert len(reserved) == allowance


def test_canonical_budget_used_and_max_share_the_attempt_unit(
    tmp_path: Path,
) -> None:
    """Sol N2: ``author_sessions_used`` and ``author_sessions_max`` agree in units.

    One COMPLETE attempt buys two guard sessions but is ONE attempt: the
    canonical used count (``chargeable_author_sessions`` fed the attempt-group
    projection, exactly as model assembly feeds it) reads 1 against a max of
    ``A`` -- never the raw session count that made a normal trajectory read
    ``used=6, max=3``.  Raw guard sessions remain separately disclosed by the
    paid ledger's own QUOTA projection.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    item = _author_item()
    operational = _ledger(records=())
    # One complete attempt: source-request guard, settle, author guard joins
    # the same attempt group (the production dispatch scope's exact calls).
    head = stub._admit_author_spend(item, operational)  # noqa: SLF001
    stub._settle_paid_session(head, elapsed_seconds=1.0, completed=True)  # noqa: SLF001
    author = stub._admit_author_spend(  # noqa: SLF001
        item, operational, attempt_group=head.spend_id
    )
    stub._settle_paid_session(author, elapsed_seconds=1.0, completed=True)  # noqa: SLF001
    state = authority.state()
    key = ("c1-mech", item.stable_id, item.active_work_id, "author")
    raw_sessions = cast(Weight, state.quota_by_lane.get(key)).sessions
    paid_attempts = state.quota_attempt_groups.get(key)
    assert raw_sessions == 2 and paid_attempts == 1, (
        "two consummated guard sessions must weigh exactly one attempt"
    )
    # The exact model-assembly derivation: used shares max's unit again.
    used = chargeable_author_sessions(
        (), item.stable_id, item.active_work_id, paid_attempts=paid_attempts
    )
    maximum = author_session_allowance(item.granted_attempts)
    assert used == 1 <= maximum, f"used={used} must be attempts, not sessions"
    # A merged event stream (throughput census) reads the same unit from the
    # rows themselves, through the durable attempt_group identity.
    assert (
        chargeable_author_sessions(
            load_paid_session_rows(authority.path),
            item.stable_id,
            item.active_work_id,
        )
        == 1
    )


def test_a_fully_declined_attempt_group_refunds_the_attempt(tmp_path: Path) -> None:
    """A provider-declined attempt weighs zero, exactly as before the split.

    Session-unit pricing refunded a declined session's quota; the attempt
    valve keeps that behavior at the group level: a group whose every member
    the provider declined stops counting against the allowance, and a group
    with ANY consummated member keeps counting.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    item = _author_item()
    operational = _ledger(records=())
    key = ("c1-mech", item.stable_id, item.active_work_id, "author")

    def declined(query_key: str) -> ProviderReconciliation:
        """A truthful provider decline: never consummated, never billed."""

        return ProviderReconciliation(False, None, f"sha256:declined-{query_key[-8:]}")

    reservation = stub._admit_author_spend(item, operational)  # noqa: SLF001
    assert authority.state().quota_attempt_groups.get(key) == 1
    authority.settle(
        reservation,
        provider_query=declined,
        custody_outcome="exited-error",
        machine_wall_seconds=1.0,
        local_outcome="failed",
    )
    assert authority.state().quota_attempt_groups.get(key) == 0, (
        "a fully declined attempt group must refund its attempt"
    )
    # The refunded generation may head a fresh attempt; consummating it counts.
    replacement = stub._admit_author_spend(item, operational)  # noqa: SLF001
    stub._settle_paid_session(  # noqa: SLF001
        replacement, elapsed_seconds=1.0, completed=False
    )
    assert authority.state().quota_attempt_groups.get(key) == 1


def test_a_third_guard_session_of_one_attempt_refuses_at_the_fold(
    tmp_path: Path,
) -> None:
    """Sol R2 F1: the join rule is no longer open-ended -- author bound is 2.

    Once every member settled, another reservation could previously join the
    same head indefinitely (five accepted members folded as ONE attempt), so
    unlimited paid sessions laundered into one disclosed attempt while every
    join bypassed the new-attempt valve.  The fold now refuses the join that
    would exceed the head's lane bound, before any byte lands.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    item = _author_item()
    operational = _ledger(records=())
    head = stub._admit_author_spend(item, operational)  # noqa: SLF001
    stub._settle_paid_session(head, elapsed_seconds=1.0, completed=True)  # noqa: SLF001
    second = stub._admit_author_spend(  # noqa: SLF001
        item, operational, attempt_group=head.spend_id
    )
    stub._settle_paid_session(second, elapsed_seconds=1.0, completed=True)  # noqa: SLF001
    with pytest.raises(ValueError, match="bounds one attempt at 2 paid sessions"):
        stub._admit_author_spend(  # noqa: SLF001
            item, operational, attempt_group=head.spend_id
        )
    reserved = [event for event, _spend in _events(authority) if event == "session_reserved"]
    assert len(reserved) == 2, "the refused third session must mint no row"
    key = ("c1-mech", item.stable_id, item.active_work_id, "author")
    assert authority.state().quota_attempt_groups.get(key) == 1


def test_a_refunded_attempt_group_cannot_be_revived_by_a_join(tmp_path: Path) -> None:
    """Sol R2 F1 residual: the per-group refund is FINAL.

    A group every provider answer declined weighs zero attempts; a later join
    previously revived it (+1 chargeable group) while bypassing the
    new-attempt valve, so in a concurrent wave the projection could exceed the
    allowance ``A``.  The fold now refuses the revival join outright; a fresh
    session must HEAD a new attempt through the admission valve.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    item = _author_item()
    operational = _ledger(records=())
    key = ("c1-mech", item.stable_id, item.active_work_id, "author")

    def declined(query_key: str) -> ProviderReconciliation:
        """A truthful provider decline: never consummated, never billed."""

        return ProviderReconciliation(False, None, f"sha256:declined-{query_key[-8:]}")

    head = stub._admit_author_spend(item, operational)  # noqa: SLF001
    authority.settle(
        head,
        provider_query=declined,
        custody_outcome="exited-error",
        machine_wall_seconds=1.0,
        local_outcome="failed",
    )
    assert authority.state().quota_attempt_groups.get(key) == 0
    with pytest.raises(ValueError, match="refunded attempt group"):
        stub._admit_author_spend(  # noqa: SLF001
            item, operational, attempt_group=head.spend_id
        )
    assert authority.state().quota_attempt_groups.get(key) == 0, (
        "a refused revival must not move the chargeable projection"
    )
    # A fresh HEAD still admits -- through the valve, as a new attempt.
    replacement = stub._admit_author_spend(item, operational)  # noqa: SLF001
    stub._settle_paid_session(  # noqa: SLF001
        replacement, elapsed_seconds=1.0, completed=True
    )
    assert authority.state().quota_attempt_groups.get(key) == 1


def test_checker_attempt_groups_are_single_session(tmp_path: Path) -> None:
    """The checker lane's declared bound is 1: any join is unrepresentable."""

    authority = _authority(tmp_path)
    scope = {
        "campaign_id": "c1-mech",
        "generation_id": "generation-open",
        "lane": "checker",
        "dispatch_kind": "checker",
        "attempt_group": None,
        "work_generations": (("m_check", "work-check"),),
        "gate_kind": "structural",
        "checker_identity": "checker-one",
        "run_id": "run-one",
        "machine_id": "host-one",
    }
    head = authority.reserve(scope=scope, policy=_policy())
    authority.settle(
        head,
        provider_query=_constant_accept,
        custody_outcome="exited",
        machine_wall_seconds=1.0,
        local_outcome="completed",
    )
    with pytest.raises(ValueError, match="bounds one attempt at 1 paid session"):
        authority.reserve(
            scope={**scope, "attempt_group": head.spend_id}, policy=_policy()
        )


def test_lane_attempt_session_bounds_mirror_the_production_constants() -> None:
    """The fold's closed bound vocabulary and the QUOTA factor agree.

    ``AUTHOR_PAID_SESSIONS_PER_ATTEMPT`` scales the production QUOTA ceiling;
    ``LANE_ATTEMPT_SESSION_BOUNDS`` is the fold's own money truth.  They must
    stay equal, and every closed lane must declare a bound (a lane without one
    refuses joins entirely).
    """

    assert LANE_ATTEMPT_SESSION_BOUNDS == {
        "author": AUTHOR_PAID_SESSIONS_PER_ATTEMPT,
        "checker": 1,
    }
    assert set(LANE_ATTEMPT_SESSION_BOUNDS) == LANES


class _ShutdownGatedAdmissionStub(_ProductionAdmissionStub):
    """Production admission valve behind the REAL shutdown authority check.

    ``_check_shutdown`` is the production method verbatim, reading a real
    async-safe event, so these tests prove the writer-side reserve site itself
    consults the shutdown authority -- not a rig's re-implementation of it.
    """

    _check_shutdown = driver_module.CrawlerDriver._check_shutdown  # type: ignore[assignment]

    def __init__(self, authority: PaidSessionAuthority) -> None:
        """Bind the real shutdown event the production check reads."""

        super().__init__(authority)
        self._shutdown_event = threading.Event()


def test_a_shutdown_between_guard_sessions_refuses_the_second_admission(
    tmp_path: Path,
) -> None:
    """Sol R3 W1, shutdown arm: a mid-attempt admission consults shutdown.

    The retry ladder checks shutdown only at attempt start, so a SIGTERM
    landing between the two guard sessions used to let
    ``begin_guard_session`` buy the second reservation anyway and return the
    operation normally.  The writer-side reserve site now re-checks: the
    finished guard's settlement stands, the second buy mints NOTHING, and the
    typed shutdown outcome surfaces through the ladder exactly like a
    start-of-attempt refusal -- never a retry, never a session crash.
    """

    authority = _authority(tmp_path)
    stub = _ShutdownGatedAdmissionStub(authority)

    def operation() -> str:
        """Run guard one, then take the shutdown signal, then try guard two."""

        driver_admission_module._run_operator_command(  # noqa: SLF001
            _EXIT_ZERO_WRAPPER, timeout_seconds=30.0, env=None
        )
        stub._shutdown_event.set()  # noqa: SLF001 -- the signal lands mid-attempt
        driver_admission_module._run_operator_command(  # noqa: SLF001
            _EXIT_ZERO_WRAPPER, timeout_seconds=30.0, env=None
        )
        return "authored"

    with pytest.raises(DriverShutdown) as caught:
        stub._retry_infrastructure_call(  # noqa: SLF001
            operation,
            admission=("author", _author_item()),
            operational=_ledger(records=()),
        )
    assert caught.value.fact.admission_boundary == "author-admission"
    events = _events(authority)
    reserved = [spend for event, spend in events if event == "session_reserved"]
    assert len(reserved) == 1, (
        f"the post-shutdown admission must mint no second reservation; got {events}"
    )
    first = reserved[0]
    assert events == [
        ("session_reserved", first),
        ("session_custody_bound", first),
        ("session_custody_released", first),
        ("session_settled", first),
    ], f"guard one's settlement must stand and nothing else may land; got {events}"
    assert authority.outstanding_spend_ids() == ()
    assert stub.waited == [], "a shutdown refusal must never burn retry backoff"


def test_a_sealed_wave_window_refuses_the_joined_guard_admission(
    tmp_path: Path,
) -> None:
    """Sol R3 W1, sealed-window arm: ``cancel_pending()`` seals the relay too.

    Two running attempts buy and settle their heads, the writer seals the
    window (a usage pause is a decision to STOP SPENDING), and both workers
    then request their joined guard-two reservation.  The sealed relay must
    refuse both typed: no second reservation row, no provider dispatch, and
    drain/close terminates with every settled head accounted.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    operational = _ledger(records=())
    items = [
        _work_item(
            stable_id=f"m{index}", active_work_id=f"work-{index}", granted_attempts=None
        )
        for index in range(2)
    ]
    heads_settled = {item.stable_id: threading.Event() for item in items}
    sealed = threading.Event()

    def dispatch(item: Any, request_admission: Any) -> str:
        """One attempt: buy and settle the head, then join after the seal."""

        head = request_admission(None)
        stub._settle_paid_session(  # noqa: SLF001
            head, elapsed_seconds=1.0, completed=True
        )
        heads_settled[item.stable_id].set()
        assert sealed.wait(timeout=30.0), "the test must seal the window"
        request_admission(head.spend_id)
        raise AssertionError("a sealed window admitted a joined guard session")

    pool = driver_admission_module._AuthorWavePool(  # noqa: SLF001
        2,
        cast(
            Callable[[WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact],
            dispatch,
        ),
        lambda item, attempt_group: stub._admit_author_spend(  # noqa: SLF001
            item, operational, attempt_group=attempt_group
        ),
    )
    for item in items:
        pool.submit(item)
    deadline = time.monotonic() + 30.0
    while not all(event.is_set() for event in heads_settled.values()):
        assert time.monotonic() < deadline, "head sessions never settled"
        pool._service_admission_requests()  # noqa: SLF001
        time.sleep(0.005)
    assert pool.cancel_pending() == ()
    sealed.set()
    drained = dict(pool.drain())
    with pytest.raises(AuthorUsagePause):
        pool._request_admission(items[0], None)  # noqa: SLF001
    pool.close()
    assert set(drained) == {"m0", "m1"}
    for stable_id, outcome in drained.items():
        assert isinstance(outcome.error, AuthorUsagePause), (
            f"{stable_id} must surface the typed sealed-window refusal, "
            f"got {outcome.error!r}"
        )
    rows = load_paid_session_rows(authority.path)
    reserved = [row for row in rows if row["event"] == "session_reserved"]
    assert len(reserved) == 2, "only the two settled heads may hold reservations"
    assert all(row["attempt_group"] is None for row in reserved), (
        "no joined guard reservation may land after the seal"
    )
    assert len([row for row in rows if row["event"] == "session_settled"]) == 2
    assert authority.outstanding_spend_ids() == ()


def test_a_sealed_window_refuses_already_queued_admission_requests(
    tmp_path: Path,
) -> None:
    """Sol R3 W1: sealing refuses queued-but-unserviced requests typed.

    ``cancel_pending()`` must not strand a worker already blocked on the
    relay (an untyped hang) nor let a later service pass commit its buy (a
    durable money row).  Both the seal itself and a service pass that races
    it refuse typed, and the money ledger never comes into existence.
    """

    authority = _authority(tmp_path)
    stub = _ProductionAdmissionStub(authority)
    operational = _ledger(records=())
    item = _author_item()
    pool = driver_admission_module._AuthorWavePool(  # noqa: SLF001
        2,
        cast(
            Callable[[WorkItem, Callable[[Optional[str]], Reservation]], AuthorArtifact],
            lambda _item, _request_admission: "never dispatched",
        ),
        lambda work, attempt_group: stub._admit_author_spend(  # noqa: SLF001
            work, operational, attempt_group=attempt_group
        ),
    )
    queued = driver_admission_module._AuthorAdmissionRequest(item, None)  # noqa: SLF001
    pool._admission_requests.append(queued)  # noqa: SLF001
    pool.cancel_pending()
    assert queued.completed.is_set(), "a sealed window must not hang the queued worker"
    assert isinstance(queued.error, AuthorUsagePause)
    # A request that races past the seal into the queue refuses at service time.
    raced = driver_admission_module._AuthorAdmissionRequest(item, "head-spend")  # noqa: SLF001
    pool._admission_requests.append(raced)  # noqa: SLF001
    pool._service_admission_requests()  # noqa: SLF001
    assert raced.completed.is_set()
    assert isinstance(raced.error, AuthorUsagePause)
    # A fresh request after the seal refuses before it is ever queued.
    with pytest.raises(AuthorUsagePause):
        pool._request_admission(item, None)  # noqa: SLF001
    pool.close()
    assert not authority.path.exists(), "a refused admission must mint no money row"


def test_author_leg_runaway_fuse_derives_from_the_leg_shape() -> None:
    """The 2026-08-20 incident fuse: a healthy step-200 leg must fit inside it.

    The author lane borrowed the checker leg's runaway constants, and a
    concurrency-12 author leg accumulates 12 wall-hours per clock-hour -- the
    30h fuse tripped at ~2.5 clock-hours (29.84h accumulated) on a leg whose
    honest maximum is far larger. The fuse now derives from the leg's own
    theoretical maximum spend; the checker constants stay the floor so a tiny
    leg keeps a meaningful fuse.
    """

    from menagerie.crawler.constants import (
        CHECKER_LEG_SESSION_BUDGET,
        CHECKER_LEG_WALL_BUDGET_SECONDS,
        author_leg_runaway_budget,
    )

    incident_wall_seconds = 29.84 * 3600
    sessions, wall = author_leg_runaway_budget(210, 1800.0)
    assert wall > incident_wall_seconds
    assert sessions >= 210 * 4 * 2 or sessions == CHECKER_LEG_SESSION_BUDGET
    floor_sessions, floor_wall = author_leg_runaway_budget(2, 1800.0)
    assert floor_sessions == CHECKER_LEG_SESSION_BUDGET
    assert floor_wall == float(CHECKER_LEG_WALL_BUDGET_SECONDS)
    # Monotone in leg width: a wider leg never gets a smaller fuse.
    wider_sessions, wider_wall = author_leg_runaway_budget(1000, 1800.0)
    assert wider_sessions >= sessions and wider_wall >= wall
