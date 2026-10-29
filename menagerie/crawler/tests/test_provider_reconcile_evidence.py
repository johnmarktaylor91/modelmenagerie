"""Unit tests for the receipt-backed provider reconciliation adapter.

The adapter under test replaces a constant-accept stub, so the tests are written
around the ONE direction that costs money to get wrong: ``accepted: false`` must
be PROVEN by the absence of every positive marker in a closed vocabulary, and
every ambiguity must refuse instead of answering.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    ProviderReconciliation,
    Reservation,
    load_paid_session_rows,
)
from menagerie.crawler.tools import provider_reconcile_evidence as pre

CHECKER_LEG_WALL_BUDGET_SECONDS = 100_000.0


def _authority(tmp_path: Path) -> PaidSessionAuthority:
    """Return one temporary paid-session authority over a real records root."""

    return PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-20T12:00:00Z",
    )


def _scope(stable_id: str, work_id: str) -> dict[str, Any]:
    """Return one author reservation scope carrying both accounting axes."""

    return {
        "campaign_id": "c1-mech",
        "generation_id": "generation-open",
        "lane": "author",
        "dispatch_kind": "author_session",
        "work_generations": ((stable_id, work_id),),
        "gate_kind": None,
        "checker_identity": None,
        "run_id": "run-one",
        "machine_id": "host-one",
    }


def _policy() -> BudgetPolicy:
    """Return a policy wide enough that no test refuses on budget."""

    return BudgetPolicy(
        quota_sessions=100,
        quota_wall_seconds=100_000.0,
        runaway_sessions=100,
        runaway_wall_seconds=CHECKER_LEG_WALL_BUDGET_SECONDS,
        reservation_wall_seconds=100.0,
    )


def _reserve(authority: PaidSessionAuthority, stable_id: str = "m_one") -> Reservation:
    """Durably reserve one paid session and return its authorization value."""

    return authority.reserve(scope=_scope(stable_id, f"work-{stable_id}"), policy=_policy())


def _bind(authority: PaidSessionAuthority, reservation: Reservation) -> None:
    """Durably bind one spawned paid subtree's custody identity."""

    authority.bind_custody(
        reservation,
        leader_pid=4242,
        leader_pgid=4242,
        leader_start_token="start-token",
        boot_id="boot-one",
        host_id="host-one",
        lock_device=1,
        lock_inode=2,
    )


@pytest.mark.smoke
def test_reserve_then_never_bound_answers_not_accepted(tmp_path: Path) -> None:
    """The crash window with no custody row is the one PROVABLE decline.

    No custody binding means the driver never wrote the release byte, so the
    guard exited 70 without ever running the operator command and no provider
    byte could exist.  That is the whole reason a decline is representable at
    all: the constant-accept stub charged this window as consummated.
    """

    authority = _authority(tmp_path)
    reservation = _reserve(authority)
    rows = load_paid_session_rows(authority.path)

    answer = pre.reconcile_from_rows(rows, reservation.provider_idempotency_key)

    assert answer.accepted is False
    assert answer.idempotency_key == reservation.provider_idempotency_key
    assert answer.observed_wall_seconds is None
    assert answer.evidence == ()


@pytest.mark.smoke
def test_custody_bound_answers_accepted(tmp_path: Path) -> None:
    """A durable custody binding is sufficient positive evidence: spawn happened."""

    authority = _authority(tmp_path)
    reservation = _reserve(authority)
    _bind(authority, reservation)
    rows = load_paid_session_rows(authority.path)

    answer = pre.reconcile_from_rows(rows, reservation.provider_idempotency_key)

    assert answer.accepted is True
    assert "session_custody_bound" in answer.evidence


@pytest.mark.smoke
def test_released_custody_answers_accepted(tmp_path: Path) -> None:
    """A custody release is an independent positive marker, not only a bind."""

    authority = _authority(tmp_path)
    reservation = _reserve(authority)
    _bind(authority, reservation)
    authority.release_custody(reservation, outcome="verified-exit")
    rows = load_paid_session_rows(authority.path)

    answer = pre.reconcile_from_rows(rows, reservation.provider_idempotency_key)

    assert answer.accepted is True
    assert "session_custody_released" in answer.evidence


@pytest.mark.smoke
def test_prior_accepted_settlement_answers_accepted(tmp_path: Path) -> None:
    """An already-settled consummated session echoes the provider's own verdict."""

    authority = _authority(tmp_path)
    reservation = _reserve(authority)
    authority.settle(
        reservation,
        provider_query=lambda _key: ProviderReconciliation(True, 12.5, "sha256:receipt"),
        custody_outcome="verified-exit",
        machine_wall_seconds=10.0,
        local_outcome="completed",
    )
    rows = load_paid_session_rows(authority.path)

    answer = pre.reconcile_from_rows(rows, reservation.provider_idempotency_key)

    assert answer.accepted is True
    assert "session_settled" in answer.evidence


@pytest.mark.smoke
def test_unknown_key_refuses_instead_of_declining(tmp_path: Path) -> None:
    """A key with no reservation is unanswerable and must never read as free.

    Answering ``false`` about a key this ledger has no rows for would let a
    misdirected records root zero a real charge -- the money-negative direction
    the adapter exists to make unreachable.
    """

    authority = _authority(tmp_path)
    _reserve(authority)
    rows = load_paid_session_rows(authority.path)

    with pytest.raises(pre.EvidenceRefusal) as caught:
        pre.reconcile_from_rows(rows, "provider-key-this-ledger-never-minted")

    assert "unknown" in str(caught.value)


@pytest.mark.smoke
def test_duplicate_key_refuses(tmp_path: Path) -> None:
    """Two reservations claiming one provider key is tampering, not an answer."""

    authority = _authority(tmp_path)
    reservation = _reserve(authority)
    rows = list(load_paid_session_rows(authority.path))
    duplicate = dict(next(row for row in rows if row["event"] == "session_reserved"))
    duplicate["spend_id"] = "spend-forged"
    rows.append(duplicate)

    with pytest.raises(pre.EvidenceRefusal) as caught:
        pre.reconcile_from_rows(tuple(rows), reservation.provider_idempotency_key)

    assert "ambiguous" in str(caught.value)


@pytest.mark.smoke
def test_positive_evidence_vocabulary_is_closed() -> None:
    """The proof rule is an enumerated row-event set, never a heuristic."""

    assert pre.PROVIDER_REACHED_EVENTS == frozenset(
        {"session_custody_bound", "session_custody_released"}
    )


@pytest.mark.smoke
def test_answer_json_matches_the_adapter_field_contract(tmp_path: Path) -> None:
    """stdout carries EXACTLY the contracted fields, identity stamp included."""

    from menagerie.crawler.paid_session_lifecycle import PROVIDER_STATUS_FIELDS

    authority = _authority(tmp_path)
    reservation = _reserve(authority)
    rows = load_paid_session_rows(authority.path)

    payload = pre.reconcile_from_rows(rows, reservation.provider_idempotency_key).to_json()

    assert set(payload) == PROVIDER_STATUS_FIELDS
    assert payload["adapter_identity"] == pre.ADAPTER_IDENTITY


@pytest.mark.smoke
def test_cli_takes_the_key_as_its_final_argument(tmp_path: Path) -> None:
    """The shipped spelling is flags-then-key, JSON on stdout, exit zero."""

    authority = _authority(tmp_path)
    reservation = _reserve(authority)
    _bind(authority, reservation)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "menagerie.crawler.tools.provider_reconcile_evidence",
            "--records-root",
            str(tmp_path / "records"),
            reservation.provider_idempotency_key,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "idempotency_key": reservation.provider_idempotency_key,
        "accepted": True,
        "observed_wall_seconds": None,
        "adapter_identity": pre.ADAPTER_IDENTITY,
    }


@pytest.mark.smoke
def test_cli_answer_satisfies_the_driver_side_reconciler(tmp_path: Path) -> None:
    """The production `CommandProviderReconciler` parses this adapter's output."""

    from menagerie.crawler.paid_session_lifecycle import CommandProviderReconciler

    authority = _authority(tmp_path)
    reservation = _reserve(authority)
    reconciler = CommandProviderReconciler(
        (
            sys.executable,
            "-m",
            "menagerie.crawler.tools.provider_reconcile_evidence",
            "--records-root",
            str(tmp_path / "records"),
        )
    )

    receipt = reconciler(reservation.provider_idempotency_key)

    assert isinstance(receipt, ProviderReconciliation)
    assert receipt.accepted is False
    assert receipt.observed_wall_seconds is None


@pytest.mark.smoke
def test_release_gate_still_orders_binding_before_provider_exposure() -> None:
    """Pin the ordering the adapter's PROVABLE decline rests on.

    ``accepted: false`` is only a machine proof while the paid chokepoint keeps
    spawning its guard behind a release pipe whose byte is written AFTER the
    durable custody binding.  Moving the release before the bind, or dropping
    the pipe, silently converts every proven decline into a guess -- so the
    premise is asserted here rather than assumed by a docstring.
    """

    import inspect

    from menagerie.crawler import driver_admission

    source = inspect.getsource(driver_admission._run_operator_command)
    bind_at = source.index("authority.bind_custody(")
    release_at = source.index("os.write(release_write_fd")
    assert bind_at < release_at, (
        "the custody binding must land BEFORE the release byte, or a reservation "
        "with no custody row could still have reached the provider"
    )
    assert "pass_fds=(subtree_lock.fd, release_read_fd)" in source


@pytest.mark.smoke
def test_unreleased_guard_never_runs_the_operator_command() -> None:
    """A guard that reads EOF must refuse, not proceed to the provider.

    This is the other half of the premise: if EOF read as "released", a driver
    death before the binding would leave a spawned guard free to buy a session
    with no custody row behind it, and the decline would be wrong.
    """

    import inspect

    from menagerie.crawler import author_executor

    source = inspect.getsource(author_executor.run_parent_deadline_guard)
    gate_at = source.index('if released != b"1":')
    spawn_at = source.index("subprocess.Popen(")
    assert gate_at < spawn_at, "the release gate must precede the operator spawn"
    assert 'released = b""' in source, "an unreadable release fd must fail closed"
    assert "return 70" in source[gate_at:spawn_at]


@pytest.mark.smoke
def test_cli_refuses_nonzero_on_an_unknown_key(tmp_path: Path) -> None:
    """An unanswerable key exits nonzero so leg start retries instead of zeroing."""

    authority = _authority(tmp_path)
    _reserve(authority)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "menagerie.crawler.tools.provider_reconcile_evidence",
            "--records-root",
            str(tmp_path / "records"),
            "provider-key-never-minted",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert completed.stdout.strip() == ""
