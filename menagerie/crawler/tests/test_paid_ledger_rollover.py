"""Rollover of a fold-refused paid ledger: the pilot residue's continuation.

The 2026-08-20 step-200 leg left ``reserved -> bound -> released -> bound``
on its spend ledger; the accounting fold refuses that residue on every read,
so the records root bricks loudly BEFORE any admission -- correct, permanent,
and pinned here.  ``tools/rollover_paid_ledger`` is the reviewed same-campaign
continuation: preserve the poisoned bytes immutably, mint a new authority
whose genesis embeds the poisoned rows (lineage + replay-derived eligibility),
and carry the conservative constant-accept totals forward as opening charges.

Every clause of the four-part waiver doctrine is asserted:

* disposition-invariant -- the new authority starts MORE charged, never less;
* closed proof rule -- ONLY a fold-refused ledger is eligible; a healthy one
  refuses at the tool AND at the fold, on every replay;
* replay-derived -- eligibility and charges recompute from the embedded
  digest-bound rows, never from a tool-written field;
* disclosed -- ``AccountingState.rollover`` and one ``paid-ledger-rollover``
  operational event carry the census.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler import driver_admission as driver_admission_module
from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.driver_contracts import DriverIntegrationError
from menagerie.crawler.identity import stable_hash, stamp_row_digest
from menagerie.crawler.paid_session_accounting import (
    PAID_SESSION_CHAIN_SCOPE,
    PAID_SESSION_RESERVATION_SCHEMA,
    PAID_SESSION_ROLLOVER_SCHEMA,
    ROLLOVER_MAX_CARRIED_ROWS,
    ROLLOVER_MAX_DEPTH,
    Weight,
    index_rows,
    validate_accounting_rollover,
)
from menagerie.crawler.paid_session_durability import AnchoredLedger
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    ProviderReconciliation,
)
from menagerie.crawler.recordio import (
    JsonlLedger,
    SingleWriterError,
    operational_events_path,
)
from menagerie.crawler.tools.rollover_paid_ledger import (
    main as rollover_main,
    paid_ledger_path,
    rollover_paid_ledger,
)

pytestmark = pytest.mark.smoke

_RESERVED_WALL = 100.0


def _constant_accept(key: str) -> ProviderReconciliation:
    """The deployed constant-accept adapter."""

    return ProviderReconciliation(True, None, f"sha256:receipt-{key[-12:]}")


def _authority(records_root: Path) -> PaidSessionAuthority:
    """Return one authority over the canonical spend ledger location."""

    return PaidSessionAuthority(
        paid_ledger_path(records_root),
        runtime_root=records_root.parent / ".crawl-local",
        clock=lambda: "2026-08-20T03:45:26Z",
    )


def _author_scope(
    stable_id: str, work_id: str, generation_id: str = "generation-pilot"
) -> dict[str, Any]:
    """Return one author reservation scope on both accounting axes."""

    return {
        "campaign_id": "c1-mech",
        "generation_id": generation_id,
        "lane": "author",
        "dispatch_kind": "author_source",
        "attempt_group": None,
        "work_generations": ((stable_id, work_id),),
        "gate_kind": None,
        "checker_identity": None,
        "run_id": "run-pilot",
        "machine_id": "host-one",
    }


def _policy() -> BudgetPolicy:
    """Return the test budget policy."""

    return BudgetPolicy(
        quota_sessions=100,
        quota_wall_seconds=100_000.0,
        runaway_sessions=100,
        runaway_wall_seconds=100_000.0,
        reservation_wall_seconds=_RESERVED_WALL,
    )


def _poison_pilot_residue(records_root: Path) -> Path:
    """Materialize the pilot's exact 5-row residue shape on disk.

    ``genesis, reserved, bound, released, bound-duplicate`` -- the duplicate
    row appended through a fold-permissive writer over the SAME chain scope
    and anchor protocol, exactly as the pre-fix writer bug landed it (the
    production append is fold-first now, so only a substrate with no
    accounting fold can reproduce the residue).
    """

    ledger_path = paid_ledger_path(records_root)
    authority = _authority(records_root)
    legacy = records_root / "operational" / "events.jsonl"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    authority.initialize_legacy_prefix(legacy)
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
    permissive = AnchoredLedger(
        ledger_path,
        chain_scope=PAID_SESSION_CHAIN_SCOPE,
        subject="test-poison-writer",
        seed=lambda: None,
        fold=lambda value, _rows: value,
        projection="menagerie.crawler.tests.poison-writer",
    )
    bound_row = next(
        row for row in permissive.read_locked().rows if row["event"] == "session_custody_bound"
    )
    duplicate = {
        key: value
        for key, value in bound_row.items()
        if key not in {"chain", "row_sha256"}
    }
    duplicate.update(
        leader_pid=83232,
        leader_pgid=83232,
        leader_start_token="start-83232",
        created_at="2026-08-20T03:47:52Z",
    )
    permissive.append(lambda _snapshot: stamp_row_digest(duplicate))
    return ledger_path


class _ReconcilerStub(driver_admission_module.AdmissionEnvironmentMixin):
    """The production admission-time reconciliation seam over a real root."""

    def __init__(self, records_root: Path) -> None:
        """Bind the constant-accept adapter to the root's authority."""

        self._authority = _authority(records_root)
        # The production dependency contract requires ``clock``; the custody
        # deadline reaper reads it as the deadline gate's authority.
        self.dependencies = SimpleNamespace(
            provider_reconciler=_constant_accept,
            clock=lambda: "2026-08-24T12:00:00Z",
        )

    def _paid_session_authority(self) -> PaidSessionAuthority:
        """Return the root's money authority."""

        return self._authority


def test_the_pilot_residue_bricks_admission_reconciliation_typed(
    tmp_path: Path,
) -> None:
    """RED pin: the exact 5-row residue refuses BEFORE any adapter query.

    ``_reconcile_paid_sessions_before_admission`` makes its first money read
    via ``outstanding_custody_leases``; the fold refuses on the duplicate row,
    so the next paid leg fails loudly at admission and nothing can settle,
    reserve, or reconcile the residue.  This refusal is the tripwire working
    and must never be waived -- rollover is the ONLY sanctioned continuation.
    """

    records_root = tmp_path / "records"
    _poison_pilot_residue(records_root)
    stub = _ReconcilerStub(records_root)
    with pytest.raises(ValueError, match="has duplicate custody"):
        stub._reconcile_paid_sessions_before_admission()  # noqa: SLF001
    with pytest.raises(ValueError, match="has duplicate custody"):
        stub._authority.reserve(  # noqa: SLF001
            scope=_author_scope("m_next", "work-next"), policy=_policy()
        )


def test_rollover_preserves_bytes_and_mints_a_reconcilable_authority(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The full protocol: preserve, mint, carry charges, disclose, relaunch."""

    records_root = tmp_path / "records"
    ledger_path = _poison_pilot_residue(records_root)
    original_bytes = ledger_path.read_bytes()
    # Dry-run first: eligibility only, nothing moves.
    report = rollover_paid_ledger(
        records_root, operator="jmt", reason="pilot residue", apply=False
    )
    assert report["applied"] is False
    assert "has duplicate custody" in report["fold_refusal"]
    assert ledger_path.read_bytes() == original_bytes
    # Apply through the CLI entry point.
    assert (
        rollover_main(
            [
                "--records-root",
                str(records_root),
                "--operator",
                "jmt",
                "--reason",
                "pilot residue rollover",
                "--apply",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert "PAID_LEDGER_ROLLOVER_COMPLETE" in stdout
    # 1. The poisoned sibling is byte-identical and the anchor moved with it.
    siblings = sorted(ledger_path.parent.glob("spend.jsonl.poisoned-*"))
    poisoned = [path for path in siblings if not path.name.endswith(".anchor.json")]
    assert len(poisoned) == 1
    assert poisoned[0].read_bytes() == original_bytes
    assert poisoned[0].with_name(f"{poisoned[0].name}.anchor.json").is_file()
    # 2. The new authority folds, reconciles, and admits.
    authority = _authority(records_root)
    state = authority.state()
    assert state.rollover is not None
    assert authority.outstanding_custody_leases() == ()
    assert authority.outstanding_spend_ids() == ()
    assert authority.reconcile(_constant_accept) == ()
    # The relaunch opens its OWN runaway generation, as a real leg start does.
    fresh = authority.reserve(
        scope=_author_scope("m3353", "work-3353", "generation-relaunch"),
        policy=_policy(),
    )
    assert fresh.spend_id not in state.rollover["rollover_opening_charges"]
    # The folded state is immutable; re-read it so the fresh reservation shows.
    state = authority.state()
    assert state.rollover is not None
    # 3. Opening charges equal the conservative constant-accept reading:
    #    every carried reservation charged 1 session at its FULL reserved wall.
    carried = state.rollover["rollover_opening_charges"]
    assert len(carried) == state.rollover["carried_reservations"] == 1
    (charge,) = carried.values()
    assert charge == {"wall_seconds": _RESERVED_WALL, "sessions": 1}
    quota = state.quota_by_lane.get(("c1-mech", "m3353", "work-3353", "author"))
    # Opening charge plus the fresh reservation's conservative charge.
    assert quota == Weight(2 * _RESERVED_WALL, 2)
    runaway = state.runaway_by_lane.get(
        ("c1-mech", "generation-pilot", "author"), Weight()
    )
    assert runaway == Weight(_RESERVED_WALL, 1), (
        "the carried charge stays on ITS OWN recorded generation axis"
    )
    # The carried attempt keeps counting against the author allowance.
    assert state.rollover["carried_attempt_groups"] == 1
    assert (
        state.quota_attempt_groups.get(("c1-mech", "m3353", "work-3353", "author")) == 2
    ), "one carried attempt plus the fresh head"
    # 4. The operational ledger discloses the rollover.
    events_path = records_root / "operational" / "events.jsonl"
    events = [
        json.loads(line) for line in events_path.read_text().splitlines() if line.strip()
    ]
    disclosures = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "paid-ledger-rollover"
    ]
    assert len(disclosures) == 1
    details = disclosures[0]["details"]
    assert details["poisoned_sibling"] == poisoned[0].name
    assert details["operator"] == "jmt"
    assert "has duplicate custody" in details["fold_refusal"]
    # 5. Admission-time reconciliation now passes on the same records root.
    _ReconcilerStub(records_root)._reconcile_paid_sessions_before_admission()  # noqa: SLF001


def test_rollover_of_a_healthy_ledger_refuses_typed(tmp_path: Path) -> None:
    """A ledger the fold accepts is the accounting authority; rollover refuses."""

    records_root = tmp_path / "records"
    authority = _authority(records_root)
    reservation = authority.reserve(
        scope=_author_scope("m_ok", "work-ok"), policy=_policy()
    )
    authority.settle(
        reservation,
        provider_query=_constant_accept,
        custody_outcome="exited",
        machine_wall_seconds=1.0,
        local_outcome="completed",
    )
    ledger_path = paid_ledger_path(records_root)
    before = ledger_path.read_bytes()
    with pytest.raises(SystemExit, match="folds cleanly"):
        rollover_paid_ledger(
            records_root, operator="jmt", reason="should refuse", apply=True
        )
    assert ledger_path.read_bytes() == before
    assert not list(ledger_path.parent.glob("spend.jsonl.poisoned-*"))


def test_the_fold_itself_refuses_healthy_carried_rows(tmp_path: Path) -> None:
    """Replay-derived eligibility: the FOLD re-refuses a healthy rollover.

    The strongest healthy-laundering attempt available: the predecessor is
    FULLY authenticated (its bytes and anchor physically preserved as
    ``.poisoned-*`` siblings beside the successor, whole-file digest and
    anchor pin correct), yet the fold still refuses, because the carried rows
    fold cleanly and a healthy ledger remains the accounting authority.  The
    round-2 version of this test supplied ``sha256:fake`` and never reached
    the eligibility clause; this one proves the clause itself.
    """

    donor_root = tmp_path / "donor"
    donor = _authority(donor_root)
    reservation = donor.reserve(scope=_author_scope("m_ok", "work-ok"), policy=_policy())
    donor.settle(
        reservation,
        provider_query=_constant_accept,
        custody_outcome="exited",
        machine_wall_seconds=1.0,
        local_outcome="completed",
    )
    healthy_rows = donor.rows()
    assert index_rows(healthy_rows).row_count == len(healthy_rows)
    donor_path = paid_ledger_path(donor_root)
    successor = _authority(tmp_path / "successor")
    successor_path = paid_ledger_path(tmp_path / "successor")
    successor_path.parent.mkdir(parents=True, exist_ok=True)
    sibling = successor_path.with_name("spend.jsonl.poisoned-healthy")
    sibling.write_bytes(donor_path.read_bytes())
    sibling.with_name(f"{sibling.name}.anchor.json").write_bytes(
        donor_path.with_name("spend.jsonl.anchor.json").read_bytes()
    )
    with pytest.raises(ValueError, match="fold-refused predecessor"):
        successor.initialize_rollover(
            carried_rows=healthy_rows,
            poisoned_ledger_name=sibling.name,
            poisoned_ledger_sha256=stamp_digest_of(donor_path),
            poisoned_ledger_bytes=donor_path.stat().st_size,
        )
    # The fold-first append refuses before any byte lands: the path may have
    # been pre-created by the append's open, but zero rows are durable.
    assert not successor_path.exists() or successor_path.stat().st_size == 0, (
        "a fold-refused rollover row must never become durable"
    )
    assert successor.rows() == ()


def stamp_digest_of(path: Path) -> str:
    """Return the whole-file content digest the rollover row must name."""

    return stable_hash({"bytes_hex": path.read_bytes().hex()})


def test_a_rollover_must_be_the_successor_ledger_s_first_row(tmp_path: Path) -> None:
    """A second rollover (or one onto a used ledger) refuses typed."""

    records_root = tmp_path / "records"
    _poison_pilot_residue(records_root)
    rollover_paid_ledger(records_root, operator="jmt", reason="first", apply=True)
    authority = _authority(records_root)
    carried = authority.state().rollover
    assert carried is not None
    with pytest.raises(ValueError, match="empty successor ledger"):
        authority.initialize_rollover(
            carried_rows=authority.rows(),
            poisoned_ledger_name="again",
            poisoned_ledger_sha256="sha256:again",
            poisoned_ledger_bytes=1,
        )


def test_a_second_guard_scope_error_is_not_a_rollover_case(tmp_path: Path) -> None:
    """Contrast pin: the consume-once refusal never poisons, so no rollover.

    The refused second guard mints nothing (round 1), the ledger stays
    healthy, and the rollover tool refuses it -- the protocol is ONLY for a
    ledger whose durable rows the fold already refuses.
    """

    records_root = tmp_path / "records"
    authority = _authority(records_root)
    reservation = authority.reserve(
        scope=_author_scope("m_once", "work-once"), policy=_policy()
    )
    scope = driver_admission_module._PaidDispatchSessions(  # noqa: SLF001
        reservation, admit=None, settle=None, span_started=0.0
    )
    scope.begin_guard_session()
    with pytest.raises(DriverIntegrationError, match="second paid guard session"):
        scope.begin_guard_session()
    with pytest.raises(SystemExit, match="folds cleanly"):
        rollover_paid_ledger(
            records_root, operator="jmt", reason="not poisoned", apply=True
        )


def _permissive_writer(ledger_path: Path) -> AnchoredLedger:
    """A chain-and-anchor-correct writer with NO accounting fold (forgery rig)."""

    return AnchoredLedger(
        ledger_path,
        chain_scope=PAID_SESSION_CHAIN_SCOPE,
        subject="test-forge-writer",
        seed=lambda: None,
        fold=lambda value, _rows: value,
        projection="menagerie.crawler.tests.forge-writer",
    )


def _genesis_body(marker: str) -> dict[str, Any]:
    """One vocabulary-exact, digest-stamped accounting_genesis body."""

    return stamp_row_digest(
        {
            "schema_version": "menagerie.crawler.paid-session-genesis.v1",
            "event": "accounting_genesis",
            "legacy_operational_path": f"/forged/{marker}",
            "legacy_operational_sha256": "sha256:forged",
            "legacy_operational_bytes": 0,
            "legacy_settlement_schemas": [],
            "created_at": "2026-08-20T03:45:26Z",
        }
    )


def _reservation_body(
    spend_id: str, *, attempt_group: str | None
) -> dict[str, Any]:
    """One vocabulary-exact v2 reservation body passing validate_reservation."""

    return stamp_row_digest(
        {
            "schema_version": PAID_SESSION_RESERVATION_SCHEMA,
            "event": "session_reserved",
            "spend_id": spend_id,
            "provider_idempotency_key": f"sha256:key-{spend_id}",
            "lane": "author",
            "dispatch_kind": "author_source",
            "attempt_group": attempt_group,
            "axis": {
                "campaign_id": "c1-mech",
                "generation_id": "generation-pilot",
                "work_generations": [{"stable_id": "m3353", "work_id": "work-3353"}],
            },
            "reserved": {"wall_seconds": _RESERVED_WALL, "sessions": 1},
            "reservation_evidence": "writer-fsync",
            "gate_kind": None,
            "checker_identity": None,
            "run_id": "run-pilot",
            "machine_id": "host-one",
            "created_at": "2026-08-20T03:45:26Z",
            "custody": {
                "lease_id": f"paid-lease-{spend_id}",
                "boot_id": "boot-one",
                "host_id": "host-one",
                "lock_path": f"/locks/{spend_id}.lock",
                "driver_pid": 1,
                "driver_start_token": "start-1",
                "leader_pid": None,
                "leader_start_token": None,
                "leader_pgid": None,
                "deadline_at": "2026-08-20T03:47:06Z",
            },
        }
    )


def test_a_synthetic_fold_refused_chain_cannot_mint_a_rollover(
    tmp_path: Path,
) -> None:
    """Sol R2 R1 RED probe: fabricated evidence mints no successor authority.

    Sol's exact attack: a chain-valid inner ledger of two ``accounting_genesis``
    rows (the second makes the fold raise, so round 2's 'authentic chain the
    fold refuses' rule accepted it) minted a zero-charge successor.  The
    rollover's identity is now ANCHOR-DERIVED: the fold requires the named
    ``.poisoned-*`` sibling AND its preserved anchor on disk beside the
    successor ledger, on every replay -- a synthetic chain has neither, and the
    row lands nowhere even with a byte-correct whole-file digest.
    """

    forge_path = tmp_path / "forge" / "spend.jsonl"
    forge = _permissive_writer(forge_path)
    forge.append(lambda _snapshot: _genesis_body("one"))
    forge.append(lambda _snapshot: _genesis_body("two"))
    forged_rows = forge.read_locked().rows
    with pytest.raises(ValueError, match="must be the first row"):
        index_rows(forged_rows)  # chain-valid, fold-refused: Sol's residue class
    successor_root = tmp_path / "successor"
    successor = _authority(successor_root)
    forged_bytes = forge_path.read_bytes()
    with pytest.raises(ValueError, match="lineage does not authenticate"):
        successor.initialize_rollover(
            carried_rows=forged_rows,
            poisoned_ledger_name="spend.jsonl.poisoned-forged",
            poisoned_ledger_sha256=stable_hash({"bytes_hex": forged_bytes.hex()}),
            poisoned_ledger_bytes=len(forged_bytes),
        )
    successor_path = paid_ledger_path(successor_root)
    assert not successor_path.exists() or successor_path.stat().st_size == 0
    assert successor.rows() == ()


def test_a_false_predecessor_identity_refuses_at_the_fold(tmp_path: Path) -> None:
    """Sol R2 R1 RED probe: a false digest/name can no longer land.

    Round 2 accepted ``poisoned_ledger_sha256='sha256:deliberately-wrong'``
    and exposed it as authoritative disclosure.  The whole-file digest now
    re-derives from the embedded rows (the ledger's own canonical line
    encoding), and an out-of-namespace predecessor name refuses even when the
    digest is byte-correct.
    """

    records_root = tmp_path / "records"
    ledger_path = _poison_pilot_residue(records_root)
    poisoned_rows = _permissive_writer(ledger_path).read_locked().rows
    real_bytes = ledger_path.read_bytes()
    successor = _authority(tmp_path / "successor")
    with pytest.raises(ValueError, match="whole-file digest does not re-derive"):
        successor.initialize_rollover(
            carried_rows=poisoned_rows,
            poisoned_ledger_name="spend.jsonl.poisoned-x",
            poisoned_ledger_sha256="sha256:deliberately-wrong",
            poisoned_ledger_bytes=1,
        )
    with pytest.raises(ValueError, match="outside the preserved predecessor namespace"):
        successor.initialize_rollover(
            carried_rows=poisoned_rows,
            poisoned_ledger_name="not-the-preserved-file",
            poisoned_ledger_sha256=stable_hash({"bytes_hex": real_bytes.hex()}),
            poisoned_ledger_bytes=len(real_bytes),
        )
    assert successor.rows() == ()


def test_invalid_carried_attempt_topology_charges_conservatively(
    tmp_path: Path,
) -> None:
    """Sol R2 R1 RED probe: 2 sessions can never open as 1 attempt.

    Sol's inner ledger held two reservations both joining the same ABSENT
    head; round 2 grouped them by their raw ``attempt_group`` field and opened
    ``{'sessions': 2, 'attempt_groups': 1}``, so invalid producer fields
    changed an author-budget verdict.  The opening projection now re-runs the
    SAME closed join rule as live folding: a refused join charges as its OWN
    attempt head, so the discrepancy is unrepresentable.
    """

    records_root = tmp_path / "records"
    ledger_path = paid_ledger_path(records_root)
    writer = _permissive_writer(ledger_path)
    writer.append(lambda _snapshot: _reservation_body("paid-spend-aaa", attempt_group="paid-spend-absent"))
    writer.append(lambda _snapshot: _reservation_body("paid-spend-bbb", attempt_group="paid-spend-absent"))
    report = rollover_paid_ledger(
        records_root, operator="jmt", reason="invalid topology residue", apply=True
    )
    assert report["applied"] is True
    assert "joins absent attempt group" in report["fold_refusal"]
    state = _authority(records_root).state()
    assert state.rollover is not None
    charges = state.rollover["rollover_opening_charges"]
    sessions = sum(charge["sessions"] for charge in charges.values())
    assert sessions == 2
    assert state.rollover["carried_attempt_groups"] == 2, (
        "an invalid join topology must charge one attempt PER session, never fewer"
    )
    assert (
        state.quota_attempt_groups.get(("c1-mech", "m3353", "work-3353", "author")) == 2
    )


def test_nested_rollover_depth_and_row_bounds_refuse() -> None:
    """Sol R2 F2b: the nested rollover walk is hard-bounded, iteratively.

    Nested verification and charge enumeration previously recursed with no
    depth or work bound, so producer-shaped nesting could drive
    ``RecursionError`` -- which is not the eligibility ``ValueError`` the fold
    catches.  The extent scan refuses past the closed constants before any
    recursion happens, and the scan itself is an explicit stack.
    """

    def rollover_shape(carried: list[dict[str, Any]]) -> dict[str, Any]:
        """A shape-valid rollover row over the given carried rows."""

        return {
            "schema_version": PAID_SESSION_ROLLOVER_SCHEMA,
            "event": "accounting_rollover",
            "poisoned_ledger_name": "spend.jsonl.poisoned-x",
            "poisoned_ledger_sha256": "sha256:x",
            "poisoned_ledger_bytes": 1,
            "poisoned_chain_head": "sha256:head",
            "poisoned_row_count": len(carried),
            "poisoned_anchor": {
                "ledger_identity": "sha256:identity",
                "rows": len(carried),
                "bytes": 1,
                "chain_head": "sha256:head",
            },
            "carried_rows": carried,
            "created_at": "2026-08-20T04:00:00Z",
            "row_sha256": "sha256:not-checked-by-shape-validation",
        }

    # Depth ROLLOVER_MAX_DEPTH is admissible; one deeper refuses.
    level: dict[str, Any] = {"event": "noise"}
    for _ in range(ROLLOVER_MAX_DEPTH - 1):
        level = {"event": "accounting_rollover", "carried_rows": [level]}
    validate_accounting_rollover(rollover_shape([level]))
    level = {"event": "accounting_rollover", "carried_rows": [level]}
    with pytest.raises(ValueError, match="nests deeper than"):
        validate_accounting_rollover(rollover_shape([level]))
    # The total embedded-row walk is bounded too.
    flooded = [{"event": "noise"}] * (ROLLOVER_MAX_CARRIED_ROWS + 1)
    with pytest.raises(ValueError, match="walks more than"):
        validate_accounting_rollover(rollover_shape(flooded))


def test_mint_failure_restores_the_predecessor_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sol R2 note: an injected mint failure restores ledger AND anchor."""

    records_root = tmp_path / "records"
    ledger_path = _poison_pilot_residue(records_root)
    anchor_path = ledger_path.with_name("spend.jsonl.anchor.json")
    ledger_bytes = ledger_path.read_bytes()
    anchor_bytes = anchor_path.read_bytes()

    def explode(self: PaidSessionAuthority, **_kwargs: Any) -> dict[str, Any]:
        """The injected mint failure."""

        raise RuntimeError("injected mint failure")

    monkeypatch.setattr(PaidSessionAuthority, "initialize_rollover", explode)
    with pytest.raises(RuntimeError, match="injected mint failure"):
        rollover_paid_ledger(
            records_root, operator="jmt", reason="mint failure drill", apply=True
        )
    assert ledger_path.read_bytes() == ledger_bytes
    assert anchor_path.read_bytes() == anchor_bytes
    assert not [
        path
        for path in ledger_path.parent.glob("spend.jsonl.poisoned-*")
        if not path.name.endswith(".anchor.json")
    ]


def test_a_live_operational_writer_blocks_apply(tmp_path: Path) -> None:
    """Sol R2 note: the operational writer lock is the whole-operation guard."""

    records_root = tmp_path / "records"
    ledger_path = _poison_pilot_residue(records_root)
    before = ledger_path.read_bytes()
    live_writer = JsonlLedger(
        operational_events_path(records_root), OPERATIONAL_EVENT_SCHEMA_VERSION
    )
    try:
        with pytest.raises(SingleWriterError):
            rollover_paid_ledger(
                records_root, operator="jmt", reason="live driver drill", apply=True
            )
    finally:
        live_writer.close()
    assert ledger_path.read_bytes() == before
    assert not list(ledger_path.parent.glob("spend.jsonl.poisoned-*"))
