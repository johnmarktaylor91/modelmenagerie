"""BL-10 (g): the checker lane has wall/session breakers; cost control is a tripwire.

Audit CH-F8 / sol B03 / opus B-5: the author lane had wall constants, the
checker lane had NONE -- one 5-item batch could legally spend multiple unbounded
hours across the halving ladder, and the campaign had no ceiling at all (51.15h
of checker wall bought 67 models on the pilot, 28% rendering no verdict). Both
breakers fire BEFORE a session is bought:

* per-MODEL wall exhaustion is a lane failure for the batch: the ladder splits
  the over-budget member away; at the metadata floor the QUOTA-exhaustion class
  TERMINALIZES ``failed:accuracy-gate / checker-session-exhaustion`` under
  mandatory human review (generation-scoped exhaustion is permanent, so the
  historical run-scoped ``checker-wall-exhausted`` park re-parked forever --
  the qpark fix, 2026-08-21), while non-metadata park paths still record
  ``checker-wall-exhausted`` valve rows;
* per-LEG wall/session exhaustion is the loud resumable campaign stop
  (``CheckerLegBudgetExceeded``), exempt from the infrastructure retry ladder
  because retrying cannot change the accounting it derives from.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence, cast

import pytest

from menagerie.crawler import author_executor as author_executor_module
from menagerie.crawler import driver_admission as driver_admission_module
from menagerie.crawler import paid_session_accounting as paid_accounting_module
from menagerie.crawler import paid_session_custody as paid_custody_module
from menagerie.crawler import paid_session_durability as paid_durability_module
from menagerie.crawler import paid_session_lifecycle as paid_lifecycle
from menagerie.crawler.cli import build_parser
from menagerie.crawler.close_reads import reset_incremental_reads
from menagerie.crawler import identity as identity_module
from menagerie.crawler import ledger_chain as ledger_chain_module
from menagerie.crawler.close_reads import read_statistics, reset_read_statistics
from menagerie.crawler.identity import (
    IDENTITY_STATS,
    canonical_json_bytes,
    reset_identity_stats,
    row_content_address,
    stable_hash,
    stamp_row_digest,
)
from menagerie.crawler.checker_dispatch import CheckerWallBudgetExceeded
from menagerie.crawler.constants import (
    CHECKER_LEG_WALL_BUDGET_SECONDS,
    METADATA_BATCH_MAX,
)
from menagerie.crawler.driver_admission import (
    CheckerLegBudgetExceeded,
    CommandCheckerLane,
    _checker_lane_failure_class,
    _run_operator_command,
)
from menagerie.crawler.driver_contracts import (
    CheckerOutcome,
    DriverConfig,
    DriverIntegrationError,
    RetryableOperatorError,
)
from menagerie.crawler.ledger_chain import (
    LEDGER_CHAIN_STATS,
    LedgerChainError,
    link_row,
    reset_ledger_chain_stats,
    verify_records_integrity,
    verify_rows,
)
from menagerie.crawler.paid_session_accounting import (
    ACCOUNTING_STATS,
    LEGACY_SETTLEMENT_SCHEMAS,
    LOCAL_OUTCOMES,
    MACHINE_WALL_SOURCES,
    PAID_SESSION_ROW_VOCABULARY,
    PAID_SESSION_SETTLEMENT_SCHEMA,
    PAID_SESSION_SETTLEMENT_SCHEMA_V1,
    PAID_SESSION_SETTLEMENT_SCHEMA_V2,
    PAID_SESSION_SETTLEMENT_SCHEMA_V3,
    charged_wall_seconds,
    index_rows,
    reset_accounting_stats,
    validate_settlement,
)
from menagerie.crawler.paid_session_durability import (
    DURABILITY_STATS,
    reset_durability_stats,
)
from menagerie.crawler.paid_session_lifecycle import (
    ACCOUNTING_AXIS_PROVENANCE,
    INSTRUMENTS,
    MONEY_EVENTS,
    PROVIDER_STATUS_FIELDS,
    REQUIRED_AXIS_FIELDS,
    BudgetPolicy,
    CommandProviderReconciler,
    PaidSessionAuthority,
    PaidSessionBudgetExceeded,
    PaidSessionError,
    ProviderReconciliation,
    Reservation,
    authorized_paid_dispatch,
    campaign_spend,
    load_paid_session_rows,
    quota_spend,
    runaway_spend,
)
from menagerie.crawler.paid_session_custody import (
    ALIVE,
    CUSTODY_ACTIONS,
    CUSTODY_FD_ENV,
    CUSTODY_RESIDUAL,
    CUSTODY_TOPOLOGIES,
    FREE,
    UNANSWERABLE,
    PaidCustodyLease,
    anonymous_credential,
    inherited_custody_fds,
    custody_recovery_decision,
    open_subtree_lock,
    process_group_liveness,
    subtree_is_alive,
    subtree_liveness,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_checker_lane_failure_scoping import _valve_rows
from menagerie.crawler.tests.test_slice_f_driver import (
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)
from menagerie.crawler.tools.paid_session_custody import main as paid_custody_main

pytestmark = pytest.mark.smoke


def _lane() -> CommandCheckerLane:
    return CommandCheckerLane(("/nonexistent-wrapper",))


def _metadata_envelope(*stable_ids: str) -> dict[str, Any]:
    """Return one legacy-shaped metadata envelope for routing tests."""

    return {
        "gate_kind": "metadata_batch",
        "items": [{"stable_id": stable_id} for stable_id in stable_ids],
    }


# ---------------------------------------------------------------------------
# Pre-dispatch enforcement, per model and per leg.
# ---------------------------------------------------------------------------


def _authority(tmp_path: Path) -> PaidSessionAuthority:
    """Return one temporary shared paid-session authority."""

    return PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-18T12:00:00Z",
    )


def _provider_query(
    *, accepted: bool, wall_seconds: float | None, digest: str
) -> Callable[[str], ProviderReconciliation]:
    """Return one deterministic provider reconciliation test query."""

    def query(_provider_idempotency_key: str) -> ProviderReconciliation:
        """Return the configured provider truth for any reservation key."""

        return ProviderReconciliation(accepted, wall_seconds, digest)

    return query


def _scope(*members: tuple[str, str], generation_id: str = "generation-open") -> dict[str, Any]:
    """Return one checker reservation scope carrying both accounting axes."""

    return {
        "campaign_id": "c1-mech",
        "generation_id": generation_id,
        "lane": "checker",
        "dispatch_kind": "metadata_batch",
        "work_generations": members,
        "gate_kind": "metadata_batch",
        "checker_identity": None,
        "run_id": "run-one",
        "machine_id": "host-one",
    }


def _policy(*, quota_wall: float, runaway_sessions: int = 100) -> BudgetPolicy:
    """Return a checker policy sized for one reservation-based test."""

    return BudgetPolicy(
        quota_sessions=100,
        quota_wall_seconds=quota_wall,
        runaway_sessions=runaway_sessions,
        runaway_wall_seconds=CHECKER_LEG_WALL_BUDGET_SECONDS,
        reservation_wall_seconds=100.0,
    )


def _function_names(function: Callable[..., Any]) -> set[str]:
    """Return every loaded identifier in one production function's AST."""

    tree = ast.parse(inspect.getsource(function))
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def test_over_budget_model_refuses_before_dispatch_naming_only_itself(tmp_path: Path) -> None:
    """A batch refusal identifies the exact exhausted work coordinate."""

    authority = _authority(tmp_path)
    authority.reserve(
        scope=_scope(("m_hot", "work-hot")),
        policy=_policy(quota_wall=100.0),
    )
    rows_before_refusal = load_paid_session_rows(authority.path)
    with pytest.raises(PaidSessionBudgetExceeded) as caught:
        authority.reserve(
            scope=_scope(("m_hot", "work-hot"), ("m_cold", "work-cold")),
            policy=_policy(quota_wall=100.0),
        )
    assert load_paid_session_rows(authority.path) == rows_before_refusal
    assert caught.value.work_generations == (("m_hot", "work-hot"),)
    authority.reserve(
        scope=_scope(("m_cold", "work-cold")),
        policy=_policy(quota_wall=100.0),
    )
    concurrent = _authority(tmp_path / "concurrent")
    cap_one = BudgetPolicy(1, 100.0, 1, 100.0, 100.0)

    def attempt() -> str:
        """Return the result of one serialized concurrent reservation."""

        try:
            concurrent.reserve(scope=_scope(("m_one", "work-one")), policy=cap_one)
        except PaidSessionBudgetExceeded:
            return "refused"
        return "admitted"

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(lambda _index: attempt(), range(8)))
    assert outcomes.count("admitted") == 1
    assert outcomes.count("refused") == 7
    with pytest.raises((TypeError, ValueError)):
        Reservation(spend_id="fabricated")  # type: ignore[call-arg]


def test_terminal_dispatches_are_exempt_from_the_per_model_arm() -> None:
    """Keep quota authority out of the checker lane, including terminal calls."""

    lane = _lane()
    assert not hasattr(lane, "_enforce_checker_budgets")
    assert not hasattr(lane, "_model_wall_spent")
    assert INSTRUMENTS == {
        "QUOTA": frozenset({"campaign_id", "work_generations"}),
        "RUNAWAY_BREAKER": frozenset({"campaign_id", "generation_id"}),
    }
    assert REQUIRED_AXIS_FIELDS == frozenset(
        {"campaign_id", "generation_id", "work_generations"}
    )
    assert set(ACCOUNTING_AXIS_PROVENANCE) == REQUIRED_AXIS_FIELDS
    assert MONEY_EVENTS == frozenset({"session_reserved", "session_settled"})
    forbidden = {
        "run_id",
        "config",
        "checker_identity",
        "environ",
        "getenv",
        "self",
    }
    for projection in (quota_spend, runaway_spend, campaign_spend):
        assert not (_function_names(projection) & forbidden)
    source = inspect.getsource(CommandCheckerLane)
    for removed in (
        "restore_durable_spend",
        "_leg_wall_spent",
        "_leg_sessions",
        "_model_wall_spent",
        "_work_generation_wall_spent",
        "_account_checker_wall",
    ):
        assert removed not in source


def test_leg_wall_budget_breaker_is_loud_and_typed(tmp_path: Path) -> None:
    """Refuse a reservation that would cross the row-derived runaway wall."""

    authority = _authority(tmp_path)
    policy = BudgetPolicy(
        quota_sessions=100,
        quota_wall_seconds=100_000.0,
        runaway_sessions=100,
        runaway_wall_seconds=100.0,
        reservation_wall_seconds=100.0,
    )
    authority.reserve(scope=_scope(("m_any", "work-any")), policy=policy)
    with pytest.raises(PaidSessionBudgetExceeded, match="RUNAWAY_BREAKER"):
        authority.reserve(scope=_scope(("m_other", "work-other")), policy=policy)
    authority.reserve(
        scope=_scope(("m_other", "work-other"), generation_id="generation-fresh"),
        policy=policy,
    )


def test_leg_session_budget_breaker_is_loud_and_typed(tmp_path: Path) -> None:
    """A new invocation ID cannot reset durable work/checker spend at the cap."""

    authority = _authority(tmp_path)
    policy = _policy(quota_wall=100_000.0, runaway_sessions=1)
    authority.reserve(scope=_scope(("m_any", "work-any")), policy=policy)
    rows_before_refusal = load_paid_session_rows(authority.path)
    with pytest.raises(PaidSessionBudgetExceeded, match="RUNAWAY_BREAKER"):
        authority.reserve(
            scope={
                **_scope(("m_other", "work-other")),
                "run_id": "rotated-run-id",
                "checker_identity": {
                    "provider": "rotated",
                    "model": "rotated",
                    "version": "rotated",
                    "prompt_sha256": "sha256:rotated",
                },
            },
            policy=policy,
        )
    assert load_paid_session_rows(authority.path) == rows_before_refusal
    rows = load_paid_session_rows(authority.path)
    assert runaway_spend(rows, "c1-mech", "generation-open").sessions == 1
    reservation_row = next(row for row in rows if row["event"] == "session_reserved")

    def query(provider_idempotency_key: str) -> ProviderReconciliation:
        """Return the authoritative receipt for the reserved provider key."""

        assert provider_idempotency_key == reservation_row["provider_idempotency_key"]
        return ProviderReconciliation(True, 42.0, "sha256:provider-receipt")

    assert authority.reconcile(query) == (reservation_row["spend_id"],)
    assert authority.reconcile(query) == ()
    reconciled = load_paid_session_rows(authority.path)
    assert [row["event"] for row in reconciled].count("session_settled") == 1


# ---------------------------------------------------------------------------
# The paid-session censuses, ONE NODE EACH.
#
# PSLRE4 hygiene: these twelve ran as a serial tail of the breaker test above, so
# the FIRST failure masked every later one -- including the anchor direction
# matrix, which ran last. A census that cannot report independently is a census
# whose siblings are unmeasured whenever it is red.
# ---------------------------------------------------------------------------


def test_pslc_scale_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reserve cost per durable row must not grow with campaign history."""

    _assert_pslc_scale_acceptance(tmp_path / "scale", monkeypatch)


def test_receipt_projection_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-model receipt lookup must not re-validate the paid ledger."""

    _assert_receipt_projection_cache(tmp_path / "receipt-cache", monkeypatch)


def test_provider_not_accepted_quota_finalization(tmp_path: Path) -> None:
    """A provider decline weighs zero; an accepted local failure weighs one."""

    _assert_provider_not_accepted_quota_finalization(tmp_path / "quota-finalization")


def test_wall_axis_is_provider_attested_by_evidence(tmp_path: Path) -> None:
    """The settled wall re-derives from the machine floor and provider truth."""

    _assert_wall_axis_is_provider_attested_by_evidence(tmp_path / "wall-axis")


def test_declined_wall_is_representable(tmp_path: Path) -> None:
    """A declined session that still metered wall remains settleable."""

    _assert_declined_wall_is_representable(tmp_path / "declined-wall")


def test_provider_key_is_not_position_derived(tmp_path: Path) -> None:
    """Two hosts at the same ledger extent cannot mint the same provider key."""

    _assert_provider_key_is_not_position_derived(tmp_path / "provider-key")


def test_non_writer_weight_path_closed() -> None:
    """No non-writer path can mint or move a session weight."""

    _assert_non_writer_weight_path_closed()


def test_cross_concern_separation() -> None:
    """Accounting, custody, and durability share ZERO mutable state."""

    _assert_cross_concern_separation()


def test_no_hydration_or_mutable_counter() -> None:
    """There is no spend counter, no hydration input, no subtractive operation."""

    _assert_no_hydration_or_mutable_counter()


def test_production_provider_reconciliation_option() -> None:
    """Production exposes the durable provider-query reconciliation command."""

    _assert_production_provider_reconciliation_option()


def test_leg_start_crash_reconciliation(tmp_path: Path) -> None:
    """A durable pre-settlement crash is queried terminally before the next buy."""

    _assert_leg_start_crash_reconciliation(tmp_path / "crash-reconciliation")


def test_paid_session_anchor_direction_matrix(tmp_path: Path) -> None:
    """Every crash window recovers and every tamper direction refuses."""

    _assert_paid_session_anchor(tmp_path / "anchor")


# ---------------------------------------------------------------------------
# The row's own content digest is EVIDENCE, not a self-declared label (PSLRE4
# D-1/D-2). Both holes lived in the same place: nothing re-derived
# ``row_sha256``, and the chain -- and therefore the anchor -- is a function of
# exactly that field.
# ---------------------------------------------------------------------------

#: Length-PRESERVING in-place edits of a money row, as
#: ``(arm, needle, replacement)``. Each rewrites a charge; none moves the row
#: count, the byte count, or (before the fix) any digest, so each one shipped with
#: load, integrity, and the anchor all GREEN.
_IN_PLACE_MONEY_EDITS: tuple[tuple[str, bytes, bytes], ...] = (
    # Move the victim reservation out of the audited campaign entirely.
    ("campaign-axis", b"c1-mech", b"c1-mecX"),
    # Shrink the authorized wall on the QUOTA and RUNAWAY axes: ``6e-05`` is valid
    # JSON and exactly as many bytes as ``100.0``.
    ("reserved-wall", b"100.0", b"1e-05"),
    # Re-point the victim's QUOTA work coordinate at a decoy key.
    ("work-coordinate", b"work-victim", b"work-decoyy"),
)


@pytest.mark.parametrize(("arm", "needle", "replacement"), _IN_PLACE_MONEY_EDITS)
def test_in_place_money_row_edit_refuses_typed(
    tmp_path: Path, arm: str, needle: bytes, replacement: bytes
) -> None:
    """A length-preserving in-place edit of a money row REFUSES, typed.

    This is the "non-tail corruption must refuse" arm of the durability
    invariant, and it was adopted: the edit needs no anchor write, no preimage,
    and no append, because ``row_content_digest`` fed the chain the row's own
    ``row_sha256`` and nothing ever recomputed it.

    The refusal now comes from the CHAIN rather than the accounting fold, because
    the chain derives the address from the body and therefore notices the edit
    first.  That is one authority earlier, not one authority fewer: the accounting
    re-derivation keeps its own arms
    (:func:`test_the_accounting_authority_refuses_a_copied_digest_without_the_chain`)
    so neither authority silently stops being tested.
    """

    reset_incremental_reads()
    authority = _authority(tmp_path / arm)
    first = authority.reserve(
        scope=_scope(("m_first", "work-first")), policy=_policy(quota_wall=100_000.0)
    )
    authority.settle(
        first,
        provider_query=_provider_query(
            accepted=True, wall_seconds=1.0, digest="sha256:honest-settlement"
        ),
        custody_outcome="exited",
        machine_wall_seconds=1.0,
        local_outcome="completed",
    )
    authority.reserve(
        scope=_scope(("m_victim", "work-victim")), policy=_policy(quota_wall=100_000.0)
    )
    before = campaign_spend(load_paid_session_rows(authority.path), "c1-mech")

    payload = authority.path.read_bytes()
    assert needle in payload, f"{arm}: the tamper needle must exist to be edited"
    authority.path.write_bytes(payload.replace(needle, replacement, 1))
    assert authority.path.stat().st_size == len(payload), (
        f"{arm}: the edit must preserve the byte length, or the anchor's byte pin "
        "refuses it for the wrong reason"
    )

    reset_incremental_reads()
    with pytest.raises(ValueError, match=_CHAIN_ADDRESS_REFUSAL):
        load_paid_session_rows(authority.path)
    reset_incremental_reads()
    with pytest.raises(LedgerChainError, match=_CHAIN_ADDRESS_REFUSAL):
        verify_records_integrity(authority.path.parent.parent)
    assert before.sessions == 2


def test_the_accounting_authority_refuses_a_copied_digest_without_the_chain() -> None:
    """The accounting re-derivation is an authority in its OWN right.

    The chain's derived address catches a tampered body first on every durable path,
    which would leave the accounting re-derivation untested if the only arms were
    ledger-level.  So this exercises it directly, with no chain in the loop -- the
    shape any future chain-free money reader would have (two non-authoritative ones
    already exist in ``tools/``), and the reason the fold keeps its own check.
    """

    honest = _synthetic_reservation("victim")
    copied = {
        **{key: value for key, value in honest.items() if key != "row_sha256"},
        "reserved": {"wall_seconds": 0.0, "sessions": 1},
        "row_sha256": honest["row_sha256"],
    }
    with pytest.raises(ValueError, match=_ROW_DIGEST_REFUSAL):
        index_rows([copied])

    # And the foreign content-address field a money row has no reason to bind.
    foreign = _stamped(
        {
            **{key: value for key, value in honest.items() if key != "row_sha256"},
            "close_sha256": honest["row_sha256"],
        }
    )
    with pytest.raises(ValueError, match="foreign content-address field"):
        index_rows([foreign])


def test_a_digest_only_edit_leaves_the_address_and_still_refuses(tmp_path: Path) -> None:
    """Editing ONLY ``row_sha256`` moves no address -- and is refused anyway.

    The structural inverse of the PSLRE5 repro, on a real ledger. The digest fields
    are outside the preimage (that is what makes honest heads byte-identical), so a
    reader that tried to CACHE the address in a field would be indistinguishable
    from today -- except that the fused re-certification refuses any declared digest
    that is not the derivation. Both halves are asserted: the address is UNCHANGED,
    and the ledger is UNREADABLE.
    """

    reset_incremental_reads()
    authority = _authority(tmp_path / "digest-only")
    authority.reserve(
        scope=_scope(("m_victim", "work-victim")), policy=_policy(quota_wall=100_000.0)
    )
    lines = authority.path.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[-1])
    address_before = ledger_chain_module.row_content_digest(row)

    tampered = dict(row)
    tampered["row_sha256"] = "sha256:" + "0" * 64
    assert ledger_chain_module.row_content_digest(tampered) == address_before
    assert len(canonical_json_bytes(tampered)) == len(canonical_json_bytes(row))
    authority.path.write_bytes(
        b"".join(lines[:-1]) + canonical_json_bytes(tampered) + b"\n"
    )

    reset_incremental_reads()
    with pytest.raises(ValueError, match=_CHAIN_ADDRESS_REFUSAL):
        load_paid_session_rows(authority.path)


def test_copied_row_digest_cannot_satisfy_an_open_anchor_intent(tmp_path: Path) -> None:
    """A forged row wearing the intended row's digest cannot satisfy the intent.

    The direction matrix carries this arm too; it is spelled out separately
    because the honesty claim it repairs is load-bearing in both the contract and
    the durability module docstring: satisfying an open intent must require a
    PREIMAGE, and it only does while the digest is re-derived from the body.
    """

    reset_incremental_reads()
    authority = _authority(tmp_path / "copied-digest")
    first = authority.reserve(
        scope=_scope(("m_first", "work-first")), policy=_policy(quota_wall=100_000.0)
    )
    authority.settle(
        first,
        provider_query=_provider_query(
            accepted=True, wall_seconds=1.0, digest="sha256:honest-settlement"
        ),
        custody_outcome="exited",
        machine_wall_seconds=1.0,
        local_outcome="completed",
    )
    authority.reserve(
        scope=_scope(("m_victim", "work-victim")), policy=_policy(quota_wall=100_000.0)
    )
    anchor = authority.path.with_name(f"{authority.path.name}.anchor.json")
    _copied_row_digest_into_open_intent(authority.path, anchor)

    reset_incremental_reads()
    with pytest.raises(ValueError, match=_CHAIN_ADDRESS_REFUSAL):
        load_paid_session_rows(authority.path)


def _stamped(body: Mapping[str, Any]) -> dict[str, Any]:
    """Return one synthesized row carrying the digest its writer would stamp.

    Through the SHARED stamp helper, so a fixture can never drift from the
    derivation the chain and the accounting authority both use.
    """

    return stamp_row_digest(body)


def _synthetic_reservation(name: str, *, reserved: float = 100.0) -> dict[str, Any]:
    """Return one honestly stamped reservation row for a synthesized ledger."""

    return _stamped(
        {
            "schema_version": "menagerie.crawler.paid-session-reservation.v1",
            "event": "session_reserved",
            "spend_id": f"paid-spend-{name}",
            "provider_idempotency_key": f"sha256:provider-key-{name}",
            "lane": "checker",
            "dispatch_kind": "metadata_batch",
            "axis": {
                "campaign_id": "c1-mech",
                "generation_id": "generation-open",
                "work_generations": [{"stable_id": f"m_{name}", "work_id": f"work-{name}"}],
            },
            "reserved": {"wall_seconds": reserved, "sessions": 1},
            "reservation_evidence": "writer-fsync",
            # The writer's disclosure fields are part of the reservation's CLOSED
            # top-level vocabulary, so a fixture that omits them is not a row the
            # writer could have minted.
            "gate_kind": "metadata_batch",
            "checker_identity": None,
            "run_id": f"run-{name}",
            "machine_id": "host-one",
            "created_at": "2026-08-18T12:00:00Z",
            "custody": {
                "lease_id": f"paid-lease-{name}",
                "boot_id": "boot-one",
                "host_id": "host-one",
                "lock_path": f"/tmp/{name}.lock",
                "driver_pid": 4242,
                "driver_start_token": "token-4242",
                "leader_pid": None,
                "leader_start_token": None,
                "leader_pgid": None,
                "deadline_at": "2026-08-18T12:10:00Z",
            },
        }
    )


def _synthetic_settlement(name: str, schema: str, *, wall: float = 60.0) -> dict[str, Any]:
    """Return one honestly stamped settlement row on its own original schema.

    Each legacy schema is spelled on ITS OWN historical terms -- v1's bare
    observed weight, v2's self-describing sessions field, v3's
    ``observed_wall_seconds`` provider block -- because the point of the
    historical-prefix arm is that a real legacy ledger keeps resolving.
    """

    body: dict[str, Any] = {
        "schema_version": schema,
        "event": "session_settled",
        "spend_id": f"paid-spend-{name}",
        "observed": {"wall_seconds": wall},
        "evidence": "receipt",
        "custody_outcome": "exited",
        "created_at": "2026-08-18T12:05:00Z",
    }
    if schema == PAID_SESSION_SETTLEMENT_SCHEMA_V2:
        body["observed"] = {"wall_seconds": wall, "sessions": 1}
    if schema == PAID_SESSION_SETTLEMENT_SCHEMA_V3:
        body["evidence"] = "provider-query"
        body["provider"] = {
            "accepted": True,
            "observed_wall_seconds": wall,
            "receipt_sha256": f"sha256:legacy-receipt-{name}",
        }
    if schema == PAID_SESSION_SETTLEMENT_SCHEMA:
        body["evidence"] = "provider-query"
        body["local_outcome"] = "completed"
        body["provider"] = {
            "accepted": True,
            "attested_wall_seconds": wall,
            "receipt_sha256": f"sha256:receipt-{name}",
            # v5 (D-1): the current provider block names its adapter.
            "adapter_identity": "synthetic-adapter/1",
        }
        body["machine"] = {"wall_seconds": wall, "source": "writer-monotonic"}
    return _stamped(body)


def _synthetic_genesis(*, declared: Sequence[str]) -> dict[str, Any]:
    """Return one cutover genesis row declaring an exact legacy schema set."""

    return _stamped(
        {
            "schema_version": "menagerie.crawler.paid-session-genesis.v1",
            "event": "accounting_genesis",
            "legacy_operational_path": "/records/operational/events.jsonl",
            "legacy_operational_sha256": stable_hash({"bytes_hex": ""}),
            "legacy_operational_bytes": 0,
            "legacy_settlement_schemas": sorted(declared),
            "created_at": "2026-08-18T11:00:00Z",
        }
    )


def _legacy_prefix_rows(*, declared: Sequence[str]) -> list[dict[str, Any]]:
    """Return a synthesized cutover ledger: a legacy prefix, then a current row."""

    rows: list[dict[str, Any]] = [_synthetic_genesis(declared=declared)]
    ladder = (
        PAID_SESSION_SETTLEMENT_SCHEMA_V1,
        PAID_SESSION_SETTLEMENT_SCHEMA_V2,
        PAID_SESSION_SETTLEMENT_SCHEMA_V3,
        PAID_SESSION_SETTLEMENT_SCHEMA,
    )
    for index, schema in enumerate(ladder):
        name = f"legacy{index}"
        rows.append(_synthetic_reservation(name))
        rows.append(_synthetic_settlement(name, schema))
    return rows


def test_a_declared_legacy_prefix_licenses_nothing_on_the_cutover_ledger() -> None:
    """A minted declaration grants NO legacy admission on this ledger (PSLRE5 F2).

    The declaration used to be the gate, and it is written by the same producer,
    with the same unkeyed authority, as the settlements it licensed: inserting a
    genesis row that declares v1 cost one row and zeroed every settlement behind
    it.  The gate is now the ledger's STRUCTURE -- the cutover row must be the first
    row and pins the pre-machine-floor prefix as a separate object, so every row
    HERE is post-cutover and a legacy settlement refuses whatever is declared.

    Transparency in the same breath: an honest CURRENT settlement on that very
    declared ledger still folds and charges exactly as before, so the declaration
    changes nothing for honest rows either.
    """

    declared = sorted(LEGACY_SETTLEMENT_SCHEMAS)
    for schema in declared:
        rows = [
            _synthetic_genesis(declared=declared),
            _synthetic_reservation("legacy"),
            _synthetic_settlement("legacy", schema),
        ]
        with pytest.raises(ValueError, match="post-cutover by construction"):
            index_rows(rows)

    state = index_rows(
        [
            _synthetic_genesis(declared=declared),
            _synthetic_reservation("current"),
            _synthetic_settlement("current", PAID_SESSION_SETTLEMENT_SCHEMA),
        ]
    )
    assert state.genesis_seen
    assert not hasattr(state, "admitted_legacy_schemas"), (
        "the accounting state must hold no producer-declared admission at all: a "
        "decision cannot be steered by state that does not exist"
    )
    assert state.campaign["c1-mech"].sessions == 1
    assert state.charges["paid-spend-current"].wall_seconds == pytest.approx(60.0)


@pytest.mark.parametrize("schema", sorted(LEGACY_SETTLEMENT_SCHEMAS))
def test_a_legacy_settlement_still_validates_on_its_own_historical_terms(
    schema: str,
) -> None:
    """The legacy schema branches keep their own coverage as pure validators.

    A legacy schema is no longer ADMISSIBLE for a row of this ledger, but the rules
    that describe those historical rows (v1's bare observed weight, v2's
    self-describing sessions field, v3's ``observed_wall_seconds`` provider block)
    are still the definition of what such a row means, and must not silently rot.
    """

    validate_settlement(_synthetic_settlement("historic", schema))
    broken = dict(_synthetic_settlement("historic", schema))
    broken["evidence"] = "not-in-the-closed-set"
    with pytest.raises(ValueError, match="evidence is outside the closed set"):
        validate_settlement(broken)


def test_an_undeclared_legacy_settlement_schema_has_no_consumer() -> None:
    """Absent a cutover declaration, every legacy settlement schema REFUSES.

    PSLRE4 F2: "the legacy schemas can only ever be the ledger's historical
    prefix" was an assertion nothing enforced, and a downgrade to one of them
    skipped the wall re-derivation and read the session weight off a
    self-describing label. A relaunched campaign ledger declares nothing, so on a
    fresh ledger ``LEGACY_SETTLEMENT_SCHEMAS`` has no consumer at all.
    """

    for schema in sorted(LEGACY_SETTLEMENT_SCHEMAS):
        rows = [
            _synthetic_genesis(declared=()),
            _synthetic_reservation("undeclared"),
            _synthetic_settlement("undeclared", schema),
        ]
        with pytest.raises(ValueError, match="predates the machine-floor wall rule"):
            index_rows(rows)
        # A ledger with NO cutover row at all is the relaunch case, and refuses too.
        with pytest.raises(ValueError, match="predates the machine-floor wall rule"):
            index_rows(rows[1:])


def test_a_uniform_downgrade_refuses_on_its_first_row() -> None:
    """Downgrading EVERY settlement never dodges the gate (PSLRE5 acct F2).

    The monotone ladder is monotone rather than strictly increasing, so a uniform
    rewrite stayed at rank 0 forever and the ladder had nothing to say -- the whole
    ledger's money was zeroable.  With legacy schemas structurally inadmissible the
    FIRST downgraded settlement refuses, whatever the genesis row declares, so the
    loss is not merely bounded to one session: there is none.
    """

    declared = sorted(LEGACY_SETTLEMENT_SCHEMAS)
    rows: list[dict[str, Any]] = [_synthetic_genesis(declared=declared)]
    for index in range(4):
        rows.append(_synthetic_reservation(f"uniform{index}"))
        rows.append(
            _synthetic_settlement(f"uniform{index}", PAID_SESSION_SETTLEMENT_SCHEMA_V1)
        )
    with pytest.raises(ValueError, match="post-cutover by construction"):
        index_rows(rows)


def test_the_settlement_schema_ladder_is_monotone_per_ledger() -> None:
    """A settlement may never fold below a schema this ledger already settled with.

    Both gates are asserted here, and they are independent: the legacy gate refuses a
    pre-machine-floor schema on this ledger at all (so it fires FIRST on these arms,
    one authority earlier than before), and the monotone ladder keeps holding for a
    downgrade between CURRENT schemas whenever the ladder grows one.
    """

    declared = sorted(LEGACY_SETTLEMENT_SCHEMAS)
    rows = [
        _synthetic_genesis(declared=declared),
        _synthetic_reservation("current"),
        _synthetic_settlement("current", PAID_SESSION_SETTLEMENT_SCHEMA),
        _synthetic_reservation("after"),
    ]
    for schema in declared:
        with pytest.raises(ValueError, match="post-cutover by construction"):
            index_rows([*rows, _synthetic_settlement("after", schema)])
    # Forward along the ladder stays legal.
    index_rows([*rows, _synthetic_settlement("after", PAID_SESSION_SETTLEMENT_SCHEMA)])


def test_the_monotone_ladder_still_bounds_a_future_current_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ladder's own rule keeps holding when the ladder grows a new rung.

    Every legacy rung is now refused one authority earlier (a legacy schema is never
    admissible on this ledger), which would leave the monotone rule with no arm at
    all -- and an untested rule is a rule that rots.  So the rule is exercised on a
    hypothetical NEXT current schema, which is exactly the case it exists for: a
    ledger that has settled with v6 may never fold a v5 row afterwards.
    """

    future = "menagerie.crawler.paid-session-settlement.v6"
    monkeypatch.setattr(
        paid_accounting_module,
        "SETTLEMENT_SCHEMA_LADDER",
        (*paid_accounting_module.SETTLEMENT_SCHEMA_LADDER, future),
    )
    builder = paid_accounting_module._StateBuilder(  # noqa: SLF001
        paid_accounting_module.AccountingState()
    )
    builder._admit_settlement_schema(future)  # noqa: SLF001
    with pytest.raises(ValueError, match="is below"):
        builder._admit_settlement_schema(  # noqa: SLF001
            PAID_SESSION_SETTLEMENT_SCHEMA
        )


class _PooledSettlementStub(driver_admission_module.AdmissionEnvironmentMixin):
    """The exact production collaborators the pooled retry seam touches.

    ``_retry_infrastructure_call`` and ``_settle_paid_session`` are inherited
    VERBATIM, so the machine wall floor this records is the floor production
    records.
    """

    def __init__(self, authority: PaidSessionAuthority, scope: dict[str, Any]) -> None:
        """Bind one real money authority behind the real settlement method."""

        self._authority = authority
        self._scope = scope
        self.dependencies = SimpleNamespace(
            boundary_hook=lambda *_args: None,
            # A token-billed provider: it consummated (and bills) the session but
            # meters no wall clock, so the machine floor is the ONLY wall evidence.
            provider_reconciler=lambda key: ProviderReconciliation(
                True, None, f"sha256:receipt-{key[-8:]}"
            ),
            monotonic=time.monotonic,
        )

    def _check_shutdown(self, *_args: object, **_kwargs: object) -> None:
        """No shutdown is pending in this rig."""

    def _paid_session_authority(self) -> PaidSessionAuthority:
        """Return the one real money authority."""

        return self._authority

    def _admit_checker_spend(self, _operational: object, _scope: object) -> Reservation:
        """Reserve exactly as the production admission does."""

        return self._authority.reserve(scope=self._scope, policy=self._pool_policy())

    def _pool_policy(self) -> BudgetPolicy:
        """Return the production-shaped checker policy: wall-only cumulative quota."""

        return BudgetPolicy(
            quota_sessions=2**31 - 1,
            quota_wall_seconds=100_000.0,
            runaway_sessions=10**6,
            runaway_wall_seconds=100_000.0,
            reservation_wall_seconds=100.0,
        )

    @staticmethod
    def _is_infrastructure_error(_exc: BaseException) -> bool:
        """Propagate immediately: one attempt, no retry ladder."""

        return False


def test_a_failed_pooled_checker_session_settles_its_measured_wall(tmp_path: Path) -> None:
    """PSLRE4 F1: a FAILED pool-prefetched session cannot settle a ~0 wall floor.

    The checker lane's only cumulative instrument is WALL, and the epoch
    RUNAWAY_BREAKER is wall-based too, so a deleted machine floor under-counts
    exactly the runaway the breaker exists for. The worker thread measures the
    real session wall onto the capsule; before the fix only the RESERVATION
    crossed the writer seam, and the failure branch re-measured an instantaneous
    replay on the writer thread.

    This is deliberately BEHAVIOURAL rather than text-shaped: it asserts the
    settled ``machine.wall_seconds`` of a failed pooled session is at least the
    wall the session really burned.
    """

    session_wall = 0.4
    authority = _authority(tmp_path / "pooled")
    scope = _scope(("m_pool_fail", "work-pool-fail"))
    stub = _PooledSettlementStub(authority, scope)
    checker_scope = driver_admission_module._CheckerSpendScope(
        "metadata_batch", (("m_pool_fail", "work-pool-fail"),), None
    )
    pool = driver_admission_module._CheckerSessionPool(
        2,
        lambda _scope_arg: authority.reserve(scope=scope, policy=stub._pool_policy()),
        lambda reservation, elapsed, completed: stub._settle_paid_session(
            reservation, elapsed_seconds=elapsed, completed=completed
        ),
    )

    def dispatch() -> CheckerOutcome:
        """Burn a real session wall, then fail the way a lane failure fails."""

        time.sleep(session_wall)
        raise DriverIntegrationError("checker wrapper produced an invalid result")

    key = "metadata:pool-fail"
    pool.submit(key, dispatch, checker_scope)
    capsule = pool.claim(key)
    assert capsule is not None and capsule.error is not None
    measured = float(capsule.elapsed_seconds)
    assert measured >= session_wall, "the pool must measure the real session wall"
    try:
        with pytest.raises(DriverIntegrationError):
            stub._retry_infrastructure_call(
                driver_admission_module._replay_then_live(capsule, dispatch),
                admission=(
                    "checker",
                    cast(driver_admission_module.WorkItem, SimpleNamespace(stable_id="m_pool_fail")),
                ),
                operational=cast(driver_admission_module.JsonlLedger, SimpleNamespace(records=())),
                checker_scope=checker_scope,
                prepaid_checker_reservation=capsule.reservation,
            )
    finally:
        pool.close()

    rows = authority.rows()
    settlement = [row for row in rows if row["event"] == "session_settled"][-1]
    assert settlement["machine"]["wall_seconds"] >= measured, (
        "the FAILED pooled session settled a machine wall floor of "
        f"{settlement['machine']['wall_seconds']!r} for a session the pool measured "
        f"at {measured:.3f}s: the writer re-measured its own instantaneous replay "
        "instead of inheriting the capsule's wall"
    )
    charged = quota_spend(
        rows, "c1-mech", "m_pool_fail", "work-pool-fail", "checker"
    ).wall_seconds
    assert charged >= session_wall, (
        f"the wall-based checker QUOTA charged {charged:.3f}s for a session that "
        f"really burned {session_wall}s"
    )


def test_a_replayed_capsule_cannot_be_threaded_with_a_foreign_reservation(
    tmp_path: Path,
) -> None:
    """The prepaid reservation and its measured wall are ONE value, or it refuses.

    The seam is what failed in F1, so it fails closed rather than silently
    preferring one of two disagreeing sources.
    """

    authority = _authority(tmp_path / "seam")
    scope = _scope(("m_seam", "work-seam"))
    stub = _PooledSettlementStub(authority, scope)
    mine = authority.reserve(scope=scope, policy=stub._pool_policy())
    theirs = authority.reserve(
        scope=_scope(("m_other", "work-other")), policy=stub._pool_policy()
    )
    capsule = driver_admission_module._CheckerDispatchOutcome(
        error=DriverIntegrationError("the pooled session failed"),
        reservation=mine,
        elapsed_seconds=7.0,
    )
    operation = driver_admission_module._replay_then_live(capsule, lambda: capsule.unwrap())
    assert operation.prepaid_wall_seconds == 7.0
    with pytest.raises(DriverIntegrationError, match="different prepaid reservation"):
        stub._retry_infrastructure_call(
            operation,
            admission=(
                "checker",
                cast(driver_admission_module.WorkItem, SimpleNamespace(stable_id="m_seam")),
            ),
            operational=cast(driver_admission_module.JsonlLedger, SimpleNamespace(records=())),
            checker_scope=driver_admission_module._CheckerSpendScope(
                "metadata_batch", (("m_seam", "work-seam"),), None
            ),
            prepaid_checker_reservation=theirs,
        )


def test_a_prepaid_reservation_without_its_capsule_refuses_typed(tmp_path: Path) -> None:
    """PSLRE6 acct F3: the seam guard raises instead of answering with a zero floor.

    ``_prepaid_checker_session`` IS the F1 fix -- it takes the prepaid reservation AND its
    measured wall from the capsule's own replay operation, so the two cannot be threaded
    apart. Its non-capsule branch used to return ``(declared, 0.0)``: a SILENT zero wall
    floor, i.e. exactly the pre-fix behaviour, for any caller that threaded a prepaid
    reservation with some other operation. No caller does that today -- both live sites
    pass ``_replay_then_live`` -- so the hole was regression-resistance rather than a live
    path, and a guard that answers a caller it cannot measure with a zero floor is a guard
    that stops guarding the moment such a caller appears.

    Both halves are asserted: the non-prepaid path (``declared is None``) is UNCHANGED, so
    the strictness is scoped to the seam it protects.
    """

    authority = _authority(tmp_path / "seam-closed")
    scope = _scope(("m_seam", "work-seam"))
    stub = _PooledSettlementStub(authority, scope)
    reservation = authority.reserve(scope=scope, policy=stub._pool_policy())

    with pytest.raises(DriverIntegrationError, match="without the replay capsule"):
        driver_admission_module._prepaid_checker_session(lambda: None, reservation)

    # The ordinary, non-prepaid call is untouched: no reservation, no wall floor.
    assert driver_admission_module._prepaid_checker_session(lambda: None, None) == (None, 0.0)

    # ...and a real capsule still carries BOTH values, from the one authority.
    capsule = driver_admission_module._CheckerDispatchOutcome(
        error=DriverIntegrationError("the pooled session failed"),
        reservation=reservation,
        elapsed_seconds=7.0,
    )
    operation = driver_admission_module._replay_then_live(capsule, lambda: capsule.unwrap())
    assert driver_admission_module._prepaid_checker_session(operation, None) == (
        reservation,
        7.0,
    )
    # Every live call site passes the capsule, which is WHY the refusal is free.
    source = Path(driver_admission_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    operations = [
        node.args[0] if node.args else None
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "prepaid_checker_reservation" for keyword in node.keywords)
    ]
    assert operations, "the prepaid seam has no call sites left to protect"
    assert all(
        isinstance(operation, ast.Call)
        and isinstance(operation.func, ast.Name)
        and operation.func.id == "_replay_then_live"
        for operation in operations
    ), "a prepaid call site stopped passing the capsule that measures its wall"


def test_an_unreadable_leader_start_token_is_not_proof_of_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PSLRE4 C-O1: absence of evidence must never read as proof of death.

    ``leader_is_provably_gone`` is the ONLY evidence an operator drain may rely on
    when the ``flock`` oracle is unanswerable, and an unreadable start token (a
    ``ps`` failure, a permission refusal, a host with neither probe) makes a LIVE
    orphan indistinguishable from a recycled pid. The supervisor's own group
    ownership proof already requires the observed token to be readable; this
    mirrors it.
    """

    lease = PaidCustodyLease(
        lease_id="paid-lease-unreadable",
        spend_id="paid-spend-unreadable",
        boot_id="boot-one",
        host_id="host-one",
        lock_path=Path("/nonexistent/paid-lease-unreadable.lock"),
        leader_pid=os.getpid(),
        leader_start_token="token-bound-at-spawn",
        leader_pgid=os.getpid(),
        deadline_at="2026-08-18T12:10:00Z",
        reserved_wall_seconds=100.0,
    )
    monkeypatch.setattr(
        paid_custody_module, "process_group_liveness", lambda _pgid: UNANSWERABLE
    )
    # A READABLE token that differs is real evidence the pid was recycled.
    monkeypatch.setattr(paid_custody_module, "process_start_token", lambda _pid: "other")
    assert paid_custody_module.leader_is_provably_gone(lease) is True
    # An UNREADABLE token proves nothing at all.
    monkeypatch.setattr(paid_custody_module, "process_start_token", lambda _pid: None)
    assert paid_custody_module.leader_is_provably_gone(lease) is False


def test_an_unusable_custody_descriptor_refuses_credential_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PSLRE4 C-O2: a declared-but-dead custody fd exposes no credential bytes.

    A credential-bearing subtree the driver's kernel liveness oracle cannot see is
    exactly the undisclosed orphan the custody invariant forbids, so the executor
    must refuse BEFORE the anonymous credential inode is allocated and before any
    provider process is spawned.
    """

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    os.close(write_fd)
    monkeypatch.setenv(CUSTODY_FD_ENV, str(read_fd))
    assert inherited_custody_fds() == (), "a closed descriptor is not live custody"

    spawned: list[object] = []
    monkeypatch.setattr(
        author_executor_module.subprocess, "Popen", lambda *a, **k: spawned.append((a, k))
    )
    with pytest.raises(author_executor_module.AuthorExecutorError, match="did not survive"):
        author_executor_module._custody_fds_for_paid_exposure()
    work = tmp_path / "attempt"
    work.mkdir()
    config = author_executor_module.ExecutorConfig(
        claude_command=("/nonexistent-claude",),
        author_model=None,
        campaign_id="c1-mech",
        wall_seconds_override=None,
        pause_after=None,
    )
    with pytest.raises(author_executor_module.AuthorExecutorError, match="did not survive"):
        author_executor_module.run_claude_session(
            "prompt",
            cwd=work,
            wall_seconds=1.0,
            config=config,
            write_root=work,
        )
    assert spawned == [], "a refused custody exposure must spawn no provider process"
    assert list(work.iterdir()) == [], (
        "a refused custody exposure must not have allocated the credential inode"
    )


def test_the_paid_row_vocabulary_is_closed_exactly_and_at_the_chokepoint() -> None:
    """Every paid row's top-level key set is EXACT, checked before any dispatch.

    Four conditions, all four asserted here (PSLRE5's allowlist analysis):

    1. exact set EQUALITY, never a superset test -- an extra key is the injection
       vector and a missing key is a validator dodged by omission;
    2. the UNION over every admissible ``(event, schema_version)`` pair names no
       content-address field other than each row's own ``row_sha256``, so a row
       that chooses WHICH vocabulary judges it cannot choose one that admits a
       foreign digest;
    3. enforcement at the ONE fold chokepoint before the event dispatch, so a
       future event branch cannot forget it;
    4. the ``chain`` envelope is exempt here because its own vocabulary is closed
       structurally inside the chain (see ``test_ledger_chain_content_address``).
    """

    union = {key for keys in PAID_SESSION_ROW_VOCABULARY.values() for key in keys}
    assert union & set(identity_module.ROW_CONTENT_ADDRESS_EXCLUDED) == {"row_sha256"}, (
        "a paid schema may never carry a content-address field other than its own"
    )

    honest = _synthetic_reservation("closed")
    index_rows([honest])

    extra = _stamped({**honest, "envelope_sha256": "sha256:not-a-known-field"})
    with pytest.raises(ValueError, match="vocabulary is not exact"):
        index_rows([extra])

    absent = {key: value for key, value in honest.items() if key != "machine_id"}
    with pytest.raises(ValueError, match="vocabulary is not exact"):
        index_rows([_stamped(absent)])

    unknown_pair = _stamped({**honest, "event": "session_invented"})
    with pytest.raises(ValueError, match="unknown event/schema pair"):
        index_rows([unknown_pair])

    # The chokepoint is the fold's FIRST act, before any per-event validator: the
    # vocabulary check must be visible in ``_StateBuilder.apply`` itself.
    body = inspect.getsource(paid_accounting_module._StateBuilder.apply)  # noqa: SLF001
    assert body.index("validate_row_vocabulary(row)") < body.index("event ="), (
        "the closed vocabulary must be enforced at the chokepoint BEFORE the event "
        "dispatch chooses which validator judges the row"
    )


def test_a_row_without_its_own_digest_refuses() -> None:
    """An unstamped or empty-digest row is refused before any field is read."""

    honest = _synthetic_reservation("digestless")
    for broken in ({key: value for key, value in honest.items() if key != "row_sha256"},
                   {**honest, "row_sha256": ""}):
        with pytest.raises(ValueError, match="requires its own declared content digest"):
            index_rows([broken])


def _reserve_cost(tmp_path: Path, sessions: int) -> tuple[float, ...]:
    """Return per-row validation, chain, anchor-write, decode, byte, and hash cost.

    The unit is deliberately WORK PER DURABLE ROW, not wall-clock: the regression
    this census exists to catch (round-3 O3-B4) was an anchor decoded and
    chain-verified three times per money append, growing ~O(n^1.85) in campaign
    rows -- and the previous tripwire could never fail on it because its own
    bound (``visits <= 3 * reserves * ledger_size``) was quadratic too.  A ratio
    that stays FLAT as the ledger grows is the only shape that proves the cost is
    O(1) per append rather than O(history).

    PSLRE4 D-3: the first three counters live on the consumer's own incremental
    paths, so a re-verify spelled some OTHER way bypassed all three and shipped
    O(history) work per append with this census green.  The next two are counted
    inside :mod:`menagerie.crawler.ledger_chain` itself -- in the ONE per-row
    chain rule and the shared row decoders -- so "chain-verify a row" is measured
    at a boundary no spelling can route around.

    PSLRE5 P5-D-2 then showed that ROWS are not the only shape a history walk has:
    a re-read spelled with the substrate's own reader, and a whole-history
    re-DIGEST that reads no bytes and decodes no row, both stayed invisible while
    real work grew linearly in ledger size.  So the last two terms are measured at
    two further unbypassable boundaries:

    * ``bytes_read``, from :mod:`menagerie.crawler.close_reads` -- every byte the
      read substrate consumes, normalized per durable row PER LEDGER BYTE, because
      each append re-proves its own held prefix by design and that term is
      inherently size-proportional.  A second whole-ledger read per append doubles
      it;
    * ``stable_hashes``, from :func:`menagerie.crawler.identity.stable_hash` -- THE
      canonical hashing boundary for the whole tree, so a re-digest of history is
      counted whichever module spells the loop and whether or not it touches disk.

    Disclosed residual (never assumed absent): a RAW ``path.read_bytes()`` plus
    ``json.loads`` inside the money path would touch none of these boundaries.
    That spelling is made unrepresentable structurally instead, by
    :func:`test_the_money_append_path_reads_rows_only_through_the_substrate`.
    """

    reset_accounting_stats()
    reset_durability_stats()
    reset_ledger_chain_stats()
    reset_read_statistics()
    reset_identity_stats()
    authority = _authority(tmp_path)
    policy = _policy(quota_wall=10.0**9, runaway_sessions=10**6)
    for index in range(sessions):
        reservation = authority.reserve(
            scope=_scope((f"m_{index}", f"work-{index}")), policy=policy
        )
        authority.settle(
            reservation,
            provider_query=_provider_query(
                accepted=True, wall_seconds=1.0, digest=f"sha256:scale-{index}"
            ),
            custody_outcome="exited",
            machine_wall_seconds=1.0,
            local_outcome="completed",
        )
    rows = 2 * sessions
    ledger_bytes = authority.path.stat().st_size
    return (
        ACCOUNTING_STATS["rows_validated"] / rows,
        DURABILITY_STATS["ledger_rows_verified"] / rows,
        DURABILITY_STATS["anchor_writes"] / rows,
        LEDGER_CHAIN_STATS["rows_linked"] / rows,
        LEDGER_CHAIN_STATS["rows_decoded"] / rows,
        read_statistics().bytes_read / rows / ledger_bytes,
        IDENTITY_STATS["stable_hashes"] / rows,
    )


#: THE cost envelope, as ``(name, counter index, O(1) ceiling)``.  ``chain_linked``,
#: ``rows_decoded``, ``bytes_read`` and ``stable_hashes`` are measured inside the
#: shared substrate rather than on the money path, so they hold whatever spelling of
#: a history walk a future change reaches for.
#:
#: ``rows_decoded`` was a placeholder 0.0 while the durability concern decoded its
#: own suffix inline; that decode now routes through the shared row decoder, so the
#: measurement is REAL (0.98 per durable row) and the ceiling it always carried is
#: unchanged.  No ceiling here has ever been raised: ``bytes_read`` and
#: ``stable_hashes`` are new terms, measured (0.50 bytes read per durable row per
#: ledger byte, and 11.45 canonical hashes per durable row) and bounded with the same
#: discipline: roughly 2x headroom over the measurement, and a ratio that grows as
#: the ledger quadruples fails the growth arm regardless of its ceiling.
_SCALE_COST_ENVELOPE: tuple[tuple[str, int, float], ...] = (
    ("rows_validated", 0, 3.0),
    ("chain_verified", 1, 2.0),
    ("anchor_writes", 2, 3.0),
    ("chain_linked", 3, 2.0),
    ("rows_decoded", 4, 1.0),
    ("bytes_read", 5, 1.0),
    ("stable_hashes", 6, 14.0),
)


def _scale_cost_findings(
    small: Sequence[float], large: Sequence[float]
) -> list[str]:
    """Return every cost-envelope violation between two ledger sizes.

    THE comparison both the census and its own negative control run, so the
    census can never disagree with the check that proves it is not vacuous.
    """

    findings: list[str] = []
    for name, index, ceiling in _SCALE_COST_ENVELOPE:
        cheap, dear = small[index], large[index]
        if dear > ceiling:
            findings.append(
                f"paid-session {name} per durable row is {dear:.2f}, above the O(1) "
                f"envelope of {ceiling}: something on the money path reads history"
            )
        if dear > cheap * 1.25 + 0.05:
            findings.append(
                f"paid-session {name} per durable row grew from {cheap:.2f} to "
                f"{dear:.2f} as the ledger quadrupled: the cost is now super-linear "
                "in campaign history, exactly the regression this census exists for"
            )
    return findings


def _assert_pslc_scale_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PERMANENT scale-cost envelope: reserve cost must not grow with history.

    Every projection is pure over rows that were verified and indexed ONCE, so a
    ledger that is four times longer must cost four times as much in total and
    exactly the same per append.  Any future change that re-reads, re-verifies,
    or re-indexes history inside the money path makes these ratios grow and this
    census RED -- which is the point: the previous bound was itself quadratic, so
    a 4x regression passed it.
    """

    del monkeypatch
    findings = _scale_cost_findings(
        _reserve_cost(tmp_path / "small", sessions=8),
        _reserve_cost(tmp_path / "large", sessions=32),
    )
    assert findings == [], "; ".join(findings)


def test_the_scale_census_sees_a_whole_history_reverify_in_the_money_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PSLRE4 D-3 negative control: the census is RED on the exact bypass.

    The regression the audit shipped past this census was a "belt and braces"
    whole-history re-verification inside ``AnchoredLedger.append`` -- O(history)
    real work per money append, and INVISIBLE, because the three counters the
    census read are incremented only on the instrumented incremental paths. This
    injects that regression at the same seam and asserts the census now names it,
    so the census can never pass vacuously against an uninstrumented re-read.
    """

    real_link_row = paid_durability_module.link_row

    def belt_and_braces(
        row: Any,
        *,
        scope: str,
        existing_rows: Sequence[Any],
        existing_state: Any = None,
    ) -> Any:
        """Re-verify the whole committed prefix before linking the next row."""

        verify_rows(list(existing_rows), scope=scope, require_links=True)
        return real_link_row(
            row, scope=scope, existing_rows=existing_rows, existing_state=existing_state
        )

    monkeypatch.setattr(paid_durability_module, "link_row", belt_and_braces)
    findings = _scale_cost_findings(
        _reserve_cost(tmp_path / "small", sessions=8),
        _reserve_cost(tmp_path / "large", sessions=32),
    )
    assert any("chain_linked" in finding for finding in findings), (
        "the scale census did not see an O(history) chain re-verification inside the "
        f"money append: {findings}"
    )


def test_the_scale_census_sees_a_whole_history_redigest_in_the_money_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PSLRE5 P5-D-2 negative control: a re-DIGEST of history is RED too.

    The bypass this closes read no bytes, decoded no row, folded nothing and
    imported nothing new: a "belt and braces" whole-history re-hash using the
    durability concern's OWN already-imported ``stable_hash``.  Every row-shaped
    counter stayed pinned flat while real work grew linearly in ledger size.
    Counted at the canonical hashing boundary, it is measured.
    """

    real_link_row = paid_durability_module.link_row

    def belt_and_braces(
        row: Any,
        *,
        scope: str,
        existing_rows: Sequence[Any],
        existing_state: Any = None,
    ) -> Any:
        """Re-digest the whole committed prefix before linking the next row."""

        for existing in existing_rows:
            stable_hash(existing)
        return real_link_row(
            row, scope=scope, existing_rows=existing_rows, existing_state=existing_state
        )

    monkeypatch.setattr(paid_durability_module, "link_row", belt_and_braces)
    findings = _scale_cost_findings(
        _reserve_cost(tmp_path / "small", sessions=8),
        _reserve_cost(tmp_path / "large", sessions=32),
    )
    assert any("stable_hashes" in finding for finding in findings), (
        "the scale census did not see an O(history) re-digest inside the money "
        f"append: {findings}"
    )


def test_the_scale_census_sees_a_whole_history_reread_in_the_money_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PSLRE5 P5-D-2 negative control: a substrate re-READ is RED too.

    The other invisible spelling: re-reading the whole ledger through the read
    substrate the module already imports.  Bytes are counted where they are actually
    consumed, so extra whole-history proofs per append break the envelope.

    Note what the baseline says about the honest path: re-proving a prefix the epoch
    has ALREADY proven costs zero bytes, because the substrate holds its proofs. The
    regression that costs real bytes is proving history at extents nobody proved
    before -- a "verify the prefix at several lengths" belt-and-braces -- and that is
    what this injects.
    """

    real_extend = paid_durability_module.AnchoredLedger._extend

    def belt_and_braces(self: Any, value: Any, rows: Any) -> Any:
        """Re-prove the committed prefix at three fresh extents per extension."""

        from menagerie.crawler.close_reads import prefix_digest

        extended = real_extend(self, value, rows)
        size = self.path.stat().st_size
        for offset in (1, 2, 3):
            prefix_digest(self.path, max(0, size - offset))
        return extended

    monkeypatch.setattr(
        paid_durability_module.AnchoredLedger, "_extend", belt_and_braces
    )
    findings = _scale_cost_findings(
        _reserve_cost(tmp_path / "small", sessions=8),
        _reserve_cost(tmp_path / "large", sessions=32),
    )
    assert any("bytes_read" in finding for finding in findings), (
        "the scale census did not see an O(history) byte re-read inside the money "
        f"append: {findings}"
    )


_MONEY_ROW_READ_FUNCTIONS = ("_extend", "_derive", "read", "read_locked", "append")

#: Row-reading spellings the money path may NOT use: they bypass every boundary the
#: scale census measures.  The sanctioned way to obtain ledger rows is the shared
#: decoder (``ledger_chain._decode_row``) over the read substrate (``close_reads``).
_FORBIDDEN_ROW_READ_CALLS = ("loads", "read_bytes", "read_text", "readlines")


def test_the_money_append_path_reads_rows_only_through_the_substrate() -> None:
    """The ONE spelling the scale census cannot see is unrepresentable.

    A raw ``path.read_bytes()`` plus ``json.loads`` inside the money path reads no
    substrate bytes, decodes through no shared decoder, and hashes nothing -- so no
    counter can price it (PSLRE5 P5-D-2 arm A).  Rather than leave that as a
    disclosed hole, the money path's row-reading functions may not spell it: every
    row arrives through ``ledger_chain``'s decoder over ``close_reads``, both of
    which are counted.  The anchor is exempt by name -- it is ONE O(1) object, not
    history -- and it is read in its own function.
    """

    tree = ast.parse(Path(paid_durability_module.__file__).read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in _MONEY_ROW_READ_FUNCTIONS:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            function = inner.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            if name in _FORBIDDEN_ROW_READ_CALLS:
                offenders.append(f"{node.name}:{inner.lineno} calls {name}")
    assert offenders == [], (
        "the money path reads ledger rows outside the counted substrate, so a "
        "history re-read spelled this way is invisible to the scale census: "
        + ", ".join(offenders)
    )


def _assert_receipt_projection_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receipt-time load is O(1) per model over an unchanged ledger extent."""

    del monkeypatch
    authority = _authority(tmp_path)
    authority.reserve(
        scope=_scope(("m_first", "work-first")),
        policy=_policy(quota_wall=100_000.0),
    )
    sibling_writer = _authority(tmp_path)
    sibling_writer.reserve(
        scope=_scope(("m_second", "work-second")),
        policy=_policy(quota_wall=100_000.0),
    )
    reset_accounting_stats()
    reset_durability_stats()
    for index in range(100):
        authority.quota_sessions("c1-mech", f"m_{index}", f"work-{index}", "author")
    assert ACCOUNTING_STATS["rows_validated"] <= 2, (
        "a per-model receipt lookup re-validated the paid ledger: the projection "
        "must be pure over rows already verified once"
    )
    assert DURABILITY_STATS["ledger_rows_verified"] <= 2


def _assert_provider_not_accepted_quota_finalization(tmp_path: Path) -> None:
    """Provider decline weighs zero while accepted local failure weighs one."""

    authority = _authority(tmp_path)
    policy = BudgetPolicy(3, 100_000.0, 100, 100_000.0, 100.0)
    scope = {
        **_scope(("m_victim", "work-victim")),
        "lane": "author",
        "dispatch_kind": "author_source",
    }
    for _index in range(3):
        reservation = authority.reserve(scope=scope, policy=policy)
        reservation_row = next(
            row
            for row in load_paid_session_rows(authority.path)
            if row.get("spend_id") == reservation.spend_id
        )

        def query(provider_idempotency_key: str) -> ProviderReconciliation:
            """Return authoritative evidence that this reservation was not accepted."""

            assert provider_idempotency_key == reservation_row["provider_idempotency_key"]
            return ProviderReconciliation(False, 0.0, f"sha256:not-accepted-{_index}")

        assert authority.reconcile(query) == (reservation.spend_id,)
    rows = load_paid_session_rows(authority.path)
    assert quota_spend(rows, "c1-mech", "m_victim", "work-victim", "author").sessions == 0
    authority.reserve(scope=scope, policy=policy)

    failed_scope = {
        **_scope(("m_local_failure", "work-local-failure")),
        "lane": "author",
        "dispatch_kind": "author_source",
    }
    locally_failed = authority.reserve(scope=failed_scope, policy=policy)
    authority.settle(
        locally_failed,
        provider_query=_provider_query(
            accepted=True,
            wall_seconds=100.0,
            digest="sha256:provider-accepted-local-failure",
        ),
        custody_outcome="exited-error",
        machine_wall_seconds=100.0,
        local_outcome="failed",
    )
    accepted_rows = load_paid_session_rows(authority.path)
    assert quota_spend(
        accepted_rows,
        "c1-mech",
        "m_local_failure",
        "work-local-failure",
        "author",
    ).sessions == 1


def _assert_wall_axis_is_provider_attested_by_evidence(tmp_path: Path) -> None:
    """The WALL axis is evidence, not a label, and closed in BOTH directions.

    Round 3 moved the whole settlement onto the reconcile adapter's answer, and
    the checker lane's only cumulative instrument is wall-based.  Two cheap
    adapters were then catastrophic in opposite directions: ``0.0`` bought 2000
    sessions against a 2.5h quota that charged nothing, and ``None`` -- the only
    honest answer a token-billed provider can give -- charged the FULL reservation
    for every session (22.7x).  The rule is one pure function of three persisted
    numbers, so both arms are bounded and a replay re-derives the charge.
    """

    # The rule itself, stated once, in both directions.
    assert charged_wall_seconds(
        reserved_wall_seconds=100.0, machine_wall_seconds=42.0, provider_wall_seconds=0.0
    ) == 42.0, "a provider reporting zero cannot delete the machine-side floor"
    assert charged_wall_seconds(
        reserved_wall_seconds=100.0, machine_wall_seconds=42.0, provider_wall_seconds=None
    ) == 42.0, "an unmetered provider cannot inflate the charge to the reservation"
    assert charged_wall_seconds(
        reserved_wall_seconds=100.0, machine_wall_seconds=42.0, provider_wall_seconds=90.0
    ) == 90.0, "a provider that bills more than we measured raises the charge"
    assert charged_wall_seconds(
        reserved_wall_seconds=100.0, machine_wall_seconds=1.0, provider_wall_seconds=10_000.0
    ) == 100.0, "a provider cannot charge past the authorization it was granted"
    assert charged_wall_seconds(
        reserved_wall_seconds=100.0, machine_wall_seconds=42.0, provider_wall_seconds=None
    ) == charged_wall_seconds(
        reserved_wall_seconds=100.0, machine_wall_seconds=42.0, provider_wall_seconds=0.0
    ), "a DECLINED session must charge the same machine floor as an unmetered one"

    # UNDER direction, end to end: an adapter that reports zero wall for every
    # completed session still exhausts a wall-only QUOTA.
    under = _authority(tmp_path / "under")
    wall_only = BudgetPolicy(2**31 - 1, 300.0, 10**6, 10.0**9, 100.0)
    charged = 0
    for index in range(100):
        try:
            reservation = under.reserve(
                scope=_scope(("m_wall", "work-wall")), policy=wall_only
            )
        except PaidSessionBudgetExceeded as exc:
            assert exc.instrument == "QUOTA"
            break
        under.settle(
            reservation,
            provider_query=_provider_query(
                accepted=True, wall_seconds=0.0, digest=f"sha256:zero-{index}"
            ),
            custody_outcome="exited",
            machine_wall_seconds=90.0,
            local_outcome="completed",
        )
        charged += 1
    assert 3 <= charged <= 4, (
        f"a provider reporting 0.0s let {charged} sessions through a 300s wall "
        "QUOTA: the machine-side floor was deleted"
    )

    # OVER direction: an unmetered provider charges what we measured, not the
    # whole reservation, so the same budget buys the honest number of sessions.
    over = _authority(tmp_path / "over")
    bought = 0
    for index in range(100):
        try:
            reservation = over.reserve(
                scope=_scope(("m_wall", "work-wall")), policy=wall_only
            )
        except PaidSessionBudgetExceeded:
            break
        over.settle(
            reservation,
            provider_query=_provider_query(
                accepted=True, wall_seconds=None, digest=f"sha256:unmetered-{index}"
            ),
            custody_outcome="exited",
            machine_wall_seconds=10.0,
            local_outcome="completed",
        )
        bought += 1
    # 300s of QUOTA, 10s measured per session, and a 100s reservation the buyer
    # must be able to afford up front: 20 settled sessions plus the one in
    # flight. Charging the RESERVATION instead of the measured wall would buy 3.
    assert bought == 21, (
        f"an unmetered provider bought {bought} sessions of a 300s budget at 10s "
        "measured each; the reservation ceiling was charged instead of the wall "
        "this host actually observed"
    )

    # The row is checkable evidence: the provider's RAW attestation is persisted
    # unclamped beside the charge, and the charge re-derives from both numbers.
    clamped = _authority(tmp_path / "clamped")
    reservation = clamped.reserve(
        scope=_scope(("m_clamp", "work-clamp")), policy=_policy(quota_wall=100_000.0)
    )
    clamped.settle(
        reservation,
        provider_query=_provider_query(
            accepted=True, wall_seconds=20_460.0, digest="sha256:ten-times-over"
        ),
        custody_outcome="exited",
        machine_wall_seconds=3.0,
        local_outcome="completed",
    )
    settlement = load_paid_session_rows(clamped.path)[-1]
    assert settlement["provider"]["attested_wall_seconds"] == 20_460.0, (
        "the row asserted a provider attestation the provider never made: the "
        "clamped local variable was written into the provider's own field"
    )
    assert settlement["observed"]["wall_seconds"] == 100.0
    assert settlement["machine"] == {"wall_seconds": 3.0, "source": "writer-monotonic"}
    assert settlement["local_outcome"] in LOCAL_OUTCOMES
    assert set(MACHINE_WALL_SOURCES) == {
        "writer-monotonic",
        "writer-reservation-age",
        "writer-unobserved",
    }

    # A local success contradicted by a provider decline is DISCLOSED and still
    # charges the wall this host observed: no free session, and the disagreement
    # is on the row rather than overwritten by it.
    contradiction = _authority(tmp_path / "contradiction")
    reservation = contradiction.reserve(
        scope=_scope(("m_conflict", "work-conflict")),
        policy=_policy(quota_wall=100_000.0),
    )
    contradiction.settle(
        reservation,
        provider_query=_provider_query(
            accepted=False, wall_seconds=None, digest="sha256:declined-but-completed"
        ),
        custody_outcome="exited",
        machine_wall_seconds=90.0,
        local_outcome="completed",
    )
    rows = load_paid_session_rows(contradiction.path)
    assert rows[-1]["local_outcome"] == "completed"
    assert rows[-1]["custody_outcome"] == "reconciled-provider-not-accepted"
    weight = quota_spend(rows, "c1-mech", "m_conflict", "work-conflict", "checker")
    assert weight.sessions == 0, "the session count stays provider-attested"
    assert weight.wall_seconds == 90.0, (
        "a provider decline deleted the last machine-side wall floor, so a "
        "locally completed session became free on every instrument"
    )


def _assert_declined_wall_is_representable(tmp_path: Path) -> None:
    """A truthful "declined but metered" answer settles instead of bricking.

    Refusing to REPRESENT that answer made the reservation permanently
    unsettleable and killed every later leg start with an untyped ``ValueError``
    that no caller could classify.
    """

    receipt = ProviderReconciliation(False, 12.5, "sha256:declined-but-metered")
    assert receipt.observed_wall_seconds == 12.5
    authority = _authority(tmp_path)
    reservation = authority.reserve(
        scope=_scope(("m_declined", "work-declined")),
        policy=_policy(quota_wall=100_000.0),
    )
    assert authority.reconcile(lambda _key: receipt) == (reservation.spend_id,)
    assert authority.outstanding_spend_ids() == ()
    rows = load_paid_session_rows(authority.path)
    weight = quota_spend(rows, "c1-mech", "m_declined", "work-declined", "checker")
    assert (weight.sessions, weight.wall_seconds) == (0, 12.5)

    # Every adapter failure leaves the reconcile ladder TYPED and retryable, so a
    # leg start is delayed rather than wedged.
    broken = _authority(tmp_path / "broken")
    broken.reserve(
        scope=_scope(("m_broken", "work-broken")), policy=_policy(quota_wall=100_000.0)
    )

    def unrepresentable(_key: str) -> ProviderReconciliation:
        """Answer with a value no reconciliation can be built from."""

        raise ValueError("provider adapter is misconfigured")

    for _attempt in range(3):
        with pytest.raises(PaidSessionError, match="did not answer"):
            broken.reconcile(unrepresentable)
    assert broken.reconcile(
        lambda _key: ProviderReconciliation(True, 5.0, "sha256:recovered")
    ) != (), "a transient adapter failure must not permanently block settlement"


def _assert_provider_key_is_not_position_derived(tmp_path: Path) -> None:
    """Two independent ledgers at the same extent mint different provider keys."""

    keys = set()
    for host in ("host-a", "host-b"):
        authority = _authority(tmp_path / host)
        reservation = authority.reserve(
            scope={**_scope(("m_same", "work-same")), "machine_id": host},
            policy=_policy(quota_wall=100_000.0),
        )
        keys.add(reservation.provider_idempotency_key)
    assert len(keys) == 2, (
        "the provider idempotency key is derived from ledger POSITION alone, so "
        "two hosts at the same extent claim the same provider session"
    )
    assert PROVIDER_STATUS_FIELDS == frozenset(
        {"idempotency_key", "accepted", "observed_wall_seconds", "adapter_identity"}
    )


def _assert_non_writer_weight_path_closed() -> None:
    """A session weight can be minted ONLY by the writer from provider evidence."""

    settle_source = inspect.getsource(PaidSessionAuthority._settle_provider_receipt)
    assert "receipt.accepted" in settle_source
    for forbidden in ("completed", "sessions=", "consummated"):
        assert forbidden not in settle_source, (
            f"{forbidden!r} reaches the settlement writer: a local flag can mint a "
            "session weight without provider evidence"
        )
    # No public entry point accepts a caller-supplied weight.
    for name in ("settle", "_settle_provider_receipt", "reserve"):
        signature = inspect.signature(getattr(PaidSessionAuthority, name))
        assert not (set(signature.parameters) & {"sessions", "weight", "consummated_sessions"})
    weight_source = inspect.getsource(paid_lifecycle.settled_session_weight)
    assert "provider" in weight_source and "custody_outcome" in weight_source
    # P-03: the production settle path must actually SUPPLY the machine floor and
    # the disclosure, or the rule above is a library nobody calls.
    production = Path(driver_admission_module.__file__).read_text(encoding="utf-8")
    settle_call = production[production.index("def _settle_paid_session") :]
    settle_call = settle_call[: settle_call.index("def _ensure_gates")]
    assert "machine_wall_seconds=max(0.0, float(elapsed_seconds))" in settle_call, (
        "the driver stopped passing its own measurement, so the wall axis has no "
        "machine-side floor and a remote answer owns it outright"
    )
    assert "local_outcome=" in settle_call
    assert "del elapsed_seconds" not in settle_call


def _assert_operator_custody_drain(tmp_path: Path) -> None:
    """An explicit operator drain clears one resolved unreapable refusal."""

    authority = _authority(tmp_path)
    reservation = authority.reserve(
        scope=_scope(("m_drain", "work-drain")),
        policy=_policy(quota_wall=100_000.0),
    )
    authority.bind_custody(
        reservation,
        leader_pid=123,
        leader_pgid=123,
        leader_start_token="start-token",
        boot_id="boot-one",
        host_id="host-one",
        lock_device=1,
        lock_inode=2,
    )
    lease = authority.outstanding_custody_leases()[0]
    decision = custody_recovery_decision(
        lease,
        current_boot_id=lease.boot_id,
        liveness=UNANSWERABLE,
        group_liveness=FREE,
        identity_matches=False,
        past_deadline=True,
    )
    assert decision.action == "unreapable-refuse"
    assert paid_custody_main(
        (
            "--drain",
            lease.lease_id,
            "--confirm-resolved",
            lease.lease_id,
            "--reason",
            "out-of-group orphan independently resolved",
            "--drained-by",
            "test-operator",
            "--records-root",
            str(tmp_path / "records"),
            "--runtime-root",
            str(tmp_path / ".crawl-local"),
        )
    ) == 0
    assert authority.outstanding_custody_leases() == ()
    release = load_paid_session_rows(authority.path)[-1]
    assert release["outcome"] == "operator-drain"
    assert release["operator"] == "test-operator"


def _assert_settled_custody_disclosure(tmp_path: Path) -> None:
    """Money settlement cannot hide an escaped bound paid subtree from the reaper."""

    authority = _authority(tmp_path)
    reservation = authority.reserve(
        scope=_scope(("m_orphan", "work-orphan")),
        policy=_policy(quota_wall=100_000.0),
    )
    authority.bind_custody(
        reservation,
        leader_pid=123,
        leader_pgid=123,
        leader_start_token="start-token",
        boot_id="boot-one",
        host_id="host-one",
        lock_device=1,
        lock_inode=2,
    )
    authority.settle(
        reservation,
        provider_query=_provider_query(
            accepted=True,
            wall_seconds=1.0,
            digest="sha256:settled-custody",
        ),
        custody_outcome="exited-error",
        machine_wall_seconds=1.0,
        local_outcome="failed",
    )
    assert [lease.spend_id for lease in authority.outstanding_custody_leases()] == [
        reservation.spend_id
    ]
    authority.release_custody(reservation, outcome="verified-exit")
    assert authority.outstanding_custody_leases() == ()


def _assert_missing_flock_fail_closed(tmp_path: Path) -> None:
    """An unanswerable liveness oracle is unreapable, never evidence of death."""

    lease = PaidCustodyLease(
        lease_id="lease-sanitized-wrapper",
        spend_id="spend-sanitized-wrapper",
        boot_id="boot-one",
        host_id="host-one",
        lock_path=tmp_path / "sanitized-wrapper.lock",
        leader_pid=123,
        leader_start_token="start-one",
        leader_pgid=123,
        deadline_at="2026-08-18T12:30:00Z",
        reserved_wall_seconds=100.0,
    )
    decision = custody_recovery_decision(
        lease,
        current_boot_id="boot-one",
        liveness=UNANSWERABLE,
        group_liveness=FREE,
        identity_matches=False,
        past_deadline=True,
    )
    assert decision.action == "unreapable-refuse"
    assert "Darwin residual" in decision.detail
    # The unlinked-pathname oracle: a missing or replaced lock inode is
    # UNANSWERABLE, so one ``rm`` under the runtime root cannot retire a lease.
    assert subtree_liveness(tmp_path / "never-existed.lock") == UNANSWERABLE
    present = tmp_path / "present.lock"
    handle = open_subtree_lock(present)
    handle.close_parent_copy()
    assert subtree_liveness(present) == FREE
    assert subtree_liveness(present, expected_identity=(0, 0)) == UNANSWERABLE
    assert subtree_is_alive(tmp_path / "never-existed.lock") is True


def _assert_clean_wrapper_releases_without_self_certification(tmp_path: Path) -> None:
    """A wrapper that says NOTHING about itself still ends a clean paid session.

    Round 3 required a token the wrapper wrote about itself before custody could
    be released.  That is evidence about the caller, never about its descendants,
    so it both certified an escaping orphan (when the wrapper cooperated) and
    bricked every clean session (when it did not) -- five reachable exits of the
    real production wrapper never wrote it.  ``verified-exit`` is now a MACHINE
    observation of both kernel oracles, so an arbitrary third-party wrapper
    works and no drain is needed.
    """

    tmp_path.mkdir(parents=True)
    authority = _authority(tmp_path)
    for index, wrapper in enumerate(
        (
            # A legacy wrapper that certifies nothing and exits cleanly.
            "import sys; sys.exit(0)",
            # The routine failure exits the real author executor takes.
            "import sys; sys.stderr.write('provider error\\n'); sys.exit(69)",
            # A shell-shaped wrapper whose grandchild exits before it does.
            "import subprocess, sys; "
            "subprocess.run([sys.executable, '-c', 'pass'], check=True)",
        )
    ):
        reservation = authority.reserve(
            scope=_scope((f"m_clean_{index}", f"work-clean-{index}")),
            policy=_policy(quota_wall=100_000.0),
        )
        with authorized_paid_dispatch(reservation):
            _run_operator_command(
                (sys.executable, "-c", wrapper), timeout_seconds=30.0, env=None
            )
        releases = [
            row
            for row in load_paid_session_rows(authority.path)
            if row["event"] == "session_custody_released"
            and row["spend_id"] == reservation.spend_id
        ]
        assert [row["outcome"] for row in releases] == ["verified-exit"], (
            f"wrapper {wrapper!r} left custody unreleased even though both kernel "
            "oracles reported its subtree gone: the next leg would refuse until a "
            "human ran the drain tool"
        )
    assert authority.outstanding_custody_leases() == ()


def _assert_grandchild_credential_custody(tmp_path: Path) -> None:
    """The credential-holding GRANDCHILD is covered, and the residual is disclosed.

    Rounds 1-3 all missed the process that actually does the buying: the
    grandchild.  It is covered structurally, because the provider credential and
    the custody lock are handed to a descendant in ONE ``pass_fds``, so a subtree
    that can buy is a subtree the ``flock`` oracle sees.  Here the grandchild
    keeps that descriptor, outlives its wrapper, and must NOT be certified as
    exited.
    """

    tmp_path.mkdir(parents=True)
    agent = tmp_path / "agent.py"
    wrapper = tmp_path / "wrapper.py"
    pid_path = tmp_path / "agent.pid"
    agent.write_text(
        "import os, pathlib, sys, time\n"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    # The realistic paid wrapper: it backgrounds the real agent and returns at
    # once.  The agent inherits the custody descriptor exactly as it would
    # inherit the credential descriptor, and it leaves the paid process group.
    wrapper.write_text(
        "import os, subprocess, sys\n"
        "log = open(sys.argv[2] + '.log', 'wb')\n"
        "fd = int(os.environ['MENAGERIE_PAID_SUBTREE_LOCK_FD'])\n"
        "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]], "
        "stdin=subprocess.DEVNULL, stdout=log, stderr=log, "
        "start_new_session=True, pass_fds=(fd,))\n",
        encoding="utf-8",
    )
    authority = _authority(tmp_path)
    reservation = authority.reserve(
        scope=_scope(("m_grandchild", "work-grandchild")),
        policy=_policy(quota_wall=100_000.0),
    )
    started = time.monotonic()
    with authorized_paid_dispatch(reservation), pytest.raises(
        DriverIntegrationError, match="could not be proven exited"
    ):
        _run_operator_command(
            (sys.executable, str(wrapper), str(agent), str(pid_path)),
            timeout_seconds=10.0,
            env=None,
        )
    # I2 DEADLINE: the refusal is BOUNDED. Round 3 read the exit certificate with
    # an unbounded ``os.read`` on the driver's main thread, so an orphan holding
    # the inherited pipe stalled the single-threaded driver for as long as it
    # lived -- a fail-closed refusal turned into a silent total stall.
    elapsed = time.monotonic() - started
    assert elapsed < 30.0, (
        f"the driver waited {elapsed:.1f}s on a live descendant; a paid orphan "
        "lives for a whole agent session, so this is an unbounded stall"
    )
    deadline = time.monotonic() + 5.0
    while not pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    pid = int(pid_path.read_text(encoding="utf-8"))
    try:
        os.kill(pid, 0)
        rows = load_paid_session_rows(authority.path)
        assert not any(row["event"] == "session_custody_released" for row in rows), (
            "a live credential-capable grandchild was certified as exited"
        )
        assert [lease.spend_id for lease in authority.outstanding_custody_leases()] == [
            reservation.spend_id
        ], "the orphan must stay DISCLOSED for the restart reaper"
        lease = authority.outstanding_custody_leases()[0]
        assert (
            subtree_liveness(lease.lock_path, expected_identity=lease.lock_identity)
            == ALIVE
        ), "the kernel oracle must answer 'alive' for anyone who asks, with no custodian"
    finally:
        os.kill(pid, 9)
    # And the ONE irreducible Darwin residual is named rather than papered over.
    assert "PR_SET_PDEATHSIG" in CUSTODY_RESIDUAL
    assert "financially bounded" in CUSTODY_RESIDUAL


def _assert_drain_refuses_an_unlinked_lock(tmp_path: Path) -> None:
    """Deleting a lock pathname cannot drain a live credential-bearing orphan.

    A live subtree holds its ``flock`` on the INODE, so a drain guarded only by
    an ``O_CREAT`` liveness probe could be satisfied by one ``rm`` of a nonsecret
    runtime pathname -- and the drain then erased the orphan's disclosure, which
    is the very property the honest invariant buys.
    """

    tmp_path.mkdir(parents=True)
    authority = _authority(tmp_path)
    reservation = authority.reserve(
        scope=_scope(("m_unlink", "work-unlink")),
        policy=_policy(quota_wall=100_000.0),
    )
    handle = open_subtree_lock(reservation.lock_path)
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            pass_fds=(handle.fd,),
            start_new_session=True,
        )
    finally:
        handle.close_parent_copy()
    authority.bind_custody(
        reservation,
        leader_pid=child.pid,
        leader_pgid=child.pid,
        leader_start_token=paid_lifecycle.process_start_token(child.pid) or "unknown",
        boot_id=paid_lifecycle.current_boot_id(),
        host_id="host-one",
        lock_device=handle.device,
        lock_inode=handle.inode,
    )
    drain_argv = (
        "--drain",
        reservation.lease_id,
        "--confirm-resolved",
        reservation.lease_id,
        "--reason",
        "claimed resolved",
        "--drained-by",
        "test-operator",
        "--records-root",
        str(tmp_path / "records"),
        "--runtime-root",
        str(tmp_path / ".crawl-local"),
    )
    try:
        assert paid_custody_main(drain_argv) == 1, "a live subtree was drained"
        reservation.lock_path.unlink()
        assert paid_custody_main(drain_argv) == 1, (
            "unlinking the runtime lock pathname drained a LIVE orphan and erased "
            "its disclosure: the aliveness primitive was made to lie"
        )
        assert child.poll() is None
        assert [lease.spend_id for lease in authority.outstanding_custody_leases()] == [
            reservation.spend_id
        ]
    finally:
        child.kill()
        child.wait(timeout=5.0)
    # Genuine resolution -- the bound leader is provably gone -- does drain.
    assert paid_custody_main(drain_argv) == 0
    assert authority.outstanding_custody_leases() == ()
    assert load_paid_session_rows(authority.path)[-1]["outcome"] == "operator-drain"


def _anchor_pin(lines: Sequence[bytes]) -> dict[str, Any]:
    """Return the exact extent pin one ledger prefix asserts."""

    state = verify_rows(
        [json.loads(line) for line in lines], scope="paid.sessions", require_links=True
    )
    return {
        "ledger_identity": state.heads[0] if state.heads else state.head_sha256,
        "rows": state.rows,
        "bytes": len(b"".join(lines)),
        "chain_head": state.head_sha256,
    }


def _write_anchor(anchor: Path, committed: Any, intent: Any) -> None:
    """Replace one anchor object with an exact committed/intent pair."""

    body = {
        "schema_version": "menagerie.crawler.paid-session-anchor.v3",
        "ledger_name": "spend.jsonl",
        "committed": committed,
        "intent": intent,
    }
    anchor.write_text(
        json.dumps({**body, "anchor_sha256": stable_hash(body)}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _anchor_phase_three_lost(ledger: Path, anchor: Path) -> None:
    """The row is durable; the anchor's promote-to-committed write was lost."""

    lines = ledger.read_bytes().splitlines(keepends=True)
    _write_anchor(anchor, _anchor_pin(lines[:-1]), _anchor_pin(lines))


def _anchor_torn_line(_ledger: Path, anchor: Path) -> None:
    """The anchor's own final line is half written."""

    with anchor.open("ab") as handle:
        handle.write(b'{"torn-anchor-row":')


def _anchor_phase_one_crash(ledger: Path, anchor: Path) -> None:
    """The anchor declared the exact next row and the row never landed."""

    lines = ledger.read_bytes().splitlines(keepends=True)
    intent = _anchor_pin(lines)
    ledger.write_bytes(b"".join(lines[:-1]))
    _write_anchor(anchor, _anchor_pin(lines[:-1]), intent)


def _ledger_tail_torn(ledger: Path, anchor: Path) -> None:
    """The writer died MID-append: a half-written final line over a phase-one anchor.

    Torn-tail recovery truncates the partial line, which lands the ledger exactly
    on the anchor's ``committed`` pin -- so the crash resolves instead of bricking
    the money ledger, and the extent is re-read AFTER recovery rather than before.
    """

    lines = ledger.read_bytes().splitlines(keepends=True)
    intended_line = lines[-1]
    committed = _anchor_pin(lines[:-1])
    intent = _anchor_pin(lines)
    ledger.write_bytes(b"".join(lines[:-1]) + intended_line[: len(intended_line) // 2])
    _write_anchor(anchor, committed, intent)


def _ledger_truncated(ledger: Path, _anchor: Path) -> None:
    """One durably committed row is removed."""

    lines = ledger.read_bytes().splitlines(keepends=True)
    ledger.write_bytes(b"".join(lines[:-1]))


def _anchor_digest_flipped(_ledger: Path, anchor: Path) -> None:
    """One anchor field is edited without its digest."""

    body = json.loads(anchor.read_text(encoding="utf-8"))
    body["committed"]["rows"] += 1
    anchor.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")


def _unsettled_spend_id(rows: Sequence[Any]) -> str:
    """Return the last reservation no settlement covers, so the forged row is VALID.

    The arm must be refused by the ANCHOR, not incidentally by a state-machine
    rule: a forgery the accounting layer would happily accept is the one the
    external extent pin exists to catch.
    """

    settled = {
        str(row["spend_id"]) for row in rows if row.get("event") == "session_settled"
    }
    candidates = [
        str(row["spend_id"])
        for row in rows
        if row.get("event") == "session_reserved" and str(row["spend_id"]) not in settled
    ]
    assert candidates, "the anchor forgery arms need an unsettled reservation to target"
    return candidates[-1]


def _forged_settlement_appended(ledger: Path, _anchor: Path) -> None:
    """A producer appends ONE chain-valid declined settlement, anchor untouched.

    This is round-3 O3-B3 exactly: the crash and the forgery used to be
    byte-identical states, so the permissive comparison adopted both and then
    re-pinned the anchor over the tamper, dropping the victim charge from
    ``sessions=1 wall=600`` to zero and turning ``verify_records_integrity``
    green.  The anchor now declares its intent BEFORE the row exists, so the two
    states are distinguishable by an ordering rather than a tolerance.
    """

    rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    body = {
        # The producer forges on the ledger's CURRENT schema: a downgraded
        # rung would refuse at the monotone ladder before the anchor arm
        # under test ever sees the row.
        "schema_version": "menagerie.crawler.paid-session-settlement.v5",
        "event": "session_settled",
        "spend_id": _unsettled_spend_id(rows),
        "observed": {"wall_seconds": 0.0},
        "evidence": "provider-query",
        "provider": {
            "accepted": False,
            "attested_wall_seconds": 0.0,
            "receipt_sha256": "sha256:forged-by-the-producer",
            "adapter_identity": "forged-adapter/1",
        },
        "machine": {"wall_seconds": None, "source": "writer-unobserved"},
        "custody_outcome": "reconciled-provider-not-accepted",
        "local_outcome": "unknown",
        "created_at": "2026-08-18T12:00:00Z",
    }
    row = {**body, "row_sha256": stable_hash(body)}
    linked = link_row(row, scope="paid.sessions", existing_rows=rows)
    with ledger.open("ab") as handle:
        handle.write(canonical_json_bytes(linked) + b"\n")


def _forged_settlement_into_open_intent(ledger: Path, anchor: Path) -> None:
    """Forge a row while a genuine phase-one intent is OPEN.

    The hardest arm: the writer had already declared the exact next row when it
    crashed, so a permissive "the ledger is one row ahead" comparison is at its
    most tempting.  The intent pins the intended row's own content digest, so
    satisfying it with a DIFFERENT row needs a preimage, not an append.
    """

    lines = ledger.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    intended_body = {
        "schema_version": "menagerie.crawler.paid-session-custody-release.v1",
        "event": "session_custody_released",
        "spend_id": _unsettled_spend_id(rows),
        "lease_id": "paid-lease-intended-but-never-written",
        "outcome": "verified-exit",
        "operator": None,
        "reason": None,
        "created_at": "2026-08-18T12:00:00Z",
    }
    intended = link_row(
        {**intended_body, "row_sha256": stable_hash(intended_body)},
        scope="paid.sessions",
        existing_rows=rows,
    )
    intended_line = canonical_json_bytes(intended) + b"\n"
    _write_anchor(
        anchor, _anchor_pin(lines), _anchor_pin([*lines, intended_line])
    )
    _forged_settlement_appended(ledger, anchor)


def _rebuild_ledger(ledger: Path, anchor: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Re-stamp, re-link, and re-anchor a WHOLE substituted row sequence.

    The strongest producer the row-level rules must survive: full write access to
    the money ledger AND its external anchor, so every digest, every chain link,
    and the committed pin are internally consistent with the substituted bodies.
    Whatever still refuses under this mutator refuses on the row's own evidence,
    not on a byte count the forger forgot to update.
    """

    rebuilt: list[dict[str, Any]] = []
    payload = b""
    for row in rows:
        body = {
            str(key): value
            for key, value in row.items()
            if key not in {"chain", "row_sha256"}
        }
        stamped = {**body, "row_sha256": stable_hash(body)}
        linked = link_row(stamped, scope="paid.sessions", existing_rows=rebuilt)
        rebuilt.append(linked)
        payload += canonical_json_bytes(linked) + b"\n"
    ledger.write_bytes(payload)
    _write_anchor(anchor, _anchor_pin(payload.splitlines(keepends=True)), None)


def _downgraded_settlement(row: Mapping[str, Any], schema: str) -> dict[str, Any]:
    """Return one honest v4 settlement rewritten as a zero-charge legacy row."""

    forged = {
        str(key): value
        for key, value in row.items()
        if key not in {"provider", "machine", "local_outcome", "chain", "row_sha256"}
    }
    forged["schema_version"] = schema
    forged["evidence"] = "receipt"
    if schema == PAID_SESSION_SETTLEMENT_SCHEMA_V2:
        # The v2 self-describing sessions field zeroes BOTH axes at once.
        forged["observed"] = {"wall_seconds": 0.0, "sessions": 0}
        forged["custody_outcome"] = "exited"
    else:
        forged["observed"] = {"wall_seconds": 0.0}
        forged["custody_outcome"] = "reconciled-provider-not-accepted"
    return forged


def _settlement_schema_downgrade(schema: str) -> Callable[[Path, Path], None]:
    """Build the mutator that rewrites this ledger's settlement as ``schema``.

    PSLRE4 F2: the current-schema wall re-derivation and the provider-attested
    session weight are BOTH keyed on the settlement schema string, so the exact
    charge rewrite a v4 row refuses was accepted when it also downgraded the
    schema -- and the contract's "legacy schemas can only ever be the ledger's
    historical prefix" claim was unenforced.
    """

    def mutate(ledger: Path, anchor: Path) -> None:
        """Downgrade every settlement on the ledger and re-anchor it."""

        rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]
        _rebuild_ledger(
            ledger,
            anchor,
            [
                _downgraded_settlement(row, schema)
                if row.get("event") == "session_settled"
                else row
                for row in rows
            ],
        )

    return mutate


def _copied_row_digest_into_open_intent(ledger: Path, anchor: Path) -> None:
    """Satisfy an OPEN phase-one intent by COPYING the intended row's digest.

    PSLRE4 D-2, the hardest forgery arm: the intent pins the intended row's chain
    head, the chain head is a function of the row's ``row_sha256`` field, and that
    field is chosen by whoever writes the row.  So a forger who can guess the
    intended body (a custody release is a spend id, a lease id, a closed outcome
    vocabulary, and the clock) needed NO preimage at all -- copy the digest onto a
    different body and pad it to the intended byte length.  Re-deriving the digest
    from the body is what turns that copy back into a preimage requirement.
    """

    lines = ledger.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    spend_id = _unsettled_spend_id(rows)
    intended_body = {
        "schema_version": "menagerie.crawler.paid-session-custody-release.v1",
        "event": "session_custody_released",
        "spend_id": spend_id,
        "lease_id": "paid-lease-intended-but-never-written",
        "outcome": "verified-exit",
        "operator": None,
        "reason": None,
        "created_at": "2026-08-18T12:00:00Z",
    }
    stolen_digest = stable_hash(intended_body)
    intended = link_row(
        {**intended_body, "row_sha256": stolen_digest},
        scope="paid.sessions",
        existing_rows=rows,
    )
    intended_line = canonical_json_bytes(intended) + b"\n"
    _write_anchor(anchor, _anchor_pin(lines), _anchor_pin([*lines, intended_line]))

    def forged(padding: int) -> bytes:
        """Return a DIFFERENT release row wearing the intended row's digest.

        The forger writes bytes directly, so it wears the INTENDED row's chain
        envelope verbatim -- satisfying the intent's pinned ``chain_head`` is the
        whole point of the arm, and the honest writer's own linker now refuses a
        stamp that is not the body derivation.
        """

        body = {
            "schema_version": "menagerie.crawler.paid-session-custody-release.v1",
            "event": "session_custody_released",
            "spend_id": spend_id,
            "lease_id": "paid-lease-intended-but-never-written",
            "outcome": "operator-drain",
            "operator": "x",
            "reason": "f" * max(1, padding),
            "created_at": "2026-08-18T12:00:00Z",
        }
        return (
            canonical_json_bytes(
                {
                    **body,
                    "row_sha256": stolen_digest,
                    "chain": dict(intended["chain"]),
                }
            )
            + b"\n"
        )

    padding = 1
    while len(forged(padding)) < len(intended_line):
        padding += 1
    line = forged(padding)
    assert len(line) == len(intended_line), (
        "the copied-digest arm must match the intended row's exact byte extent, "
        "otherwise the anchor's byte pin refuses it for the wrong reason"
    )
    with ledger.open("ab") as handle:
        handle.write(line)


def _pinned_close_sha256_in_place_edit(ledger: Path, _anchor: Path) -> None:
    """Rewrite a settled money row in place, pinning the chain via ``close_sha256``.

    PSLRE5 acct F1 / dur P5-D-1, the CRITICAL this substrate change closes: the
    chain read the FIRST self-declared digest field a row carried, and
    ``close_sha256`` outranked ``row_sha256``.  So the forger set ``close_sha256``
    to the row's ORIGINAL address, rewrote the body to zero both money axes,
    re-stamped an entirely HONEST ``row_sha256`` over the edited body, and padded a
    disclosure string to the original byte length.  Chain link, byte count, and the
    anchor object were all BYTE-IDENTICAL -- no anchor write and no preimage.
    """

    lines = ledger.read_bytes().splitlines(keepends=True)
    positions = [
        index
        for index, line in enumerate(lines)
        if json.loads(line).get("event") == "session_settled"
    ]
    assert positions, "the in-place hijack arm needs a settled money row to rewrite"
    position = positions[-1]
    original = json.loads(lines[position])
    stolen = str(original["row_sha256"])
    body = {
        key: value
        for key, value in original.items()
        if key not in {"chain", "row_sha256"}
    }

    def forged(padding: int) -> bytes:
        """Return the money-zeroed body wearing the honest row's address."""

        # Every edit stays inside what ``validate_settlement`` accepts on its own
        # terms: with the machine floor and the provider attestation both zero and
        # ``accepted`` false, the current-schema wall re-derivation legitimately
        # yields the forged 0.0 and the session weight legitimately yields 0.  The
        # disclosure strings no projection re-derives are the byte-length donors.
        zeroed = {
            **body,
            "observed": {"wall_seconds": 0.0},
            "machine": {"wall_seconds": 0.0, "source": "writer-monotonic"},
            "provider": {
                "accepted": False,
                "attested_wall_seconds": 0.0,
                "receipt_sha256": "x" * max(1, padding),
                "adapter_identity": "forged/1",
            },
            "custody_outcome": "x",
            "local_outcome": "failed",
            "close_sha256": stolen,
        }
        return (
            canonical_json_bytes(
                {
                    **zeroed,
                    "row_sha256": row_content_address(zeroed),
                    "chain": dict(original["chain"]),
                }
            )
            + b"\n"
        )

    padding = 1
    while len(forged(padding)) < len(lines[position]):
        padding += 1
    line = forged(padding)
    assert len(line) == len(lines[position]), (
        "the in-place hijack arm must preserve the row's byte length, or the anchor "
        "refuses it for the wrong reason"
    )
    ledger.write_bytes(
        b"".join(lines[:position]) + line + b"".join(lines[position + 1 :])
    )


def _close_sha256_substituted_into_open_intent(ledger: Path, anchor: Path) -> None:
    """Satisfy an OPEN intent with a substituted row pinned via ``close_sha256``.

    The D-2 half of the same PSLRE5 finding: the intent pins the intended row's
    ``(rows, bytes, chain_head)``, and under the ladder a forged body could hold the
    intended row's address in a foreign digest field -- so the state the protocol is
    DESIGNED TO ADOPT (a crash in window 2-3) was satisfiable without a preimage.
    """

    lines = ledger.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    intended_body = {
        "schema_version": "menagerie.crawler.paid-session-custody-release.v1",
        "event": "session_custody_released",
        "spend_id": _unsettled_spend_id(rows),
        "lease_id": "paid-lease-intended-but-never-written",
        "outcome": "verified-exit",
        "operator": None,
        "reason": None,
        "created_at": "2026-08-18T12:00:00Z",
    }
    intended = link_row(_stamped(intended_body), scope="paid.sessions", existing_rows=rows)
    intended_line = canonical_json_bytes(intended) + b"\n"
    _write_anchor(anchor, _anchor_pin(lines), _anchor_pin([*lines, intended_line]))
    stolen = row_content_address(intended_body)

    def forged(padding: int) -> bytes:
        """Return a different release row holding the intended address foreignly."""

        body = {
            "schema_version": "menagerie.crawler.paid-session-custody-release.v1",
            "event": "session_custody_released",
            "spend_id": _unsettled_spend_id(rows),
            "lease_id": "paid-lease-intended-but-never-written",
            "outcome": "operator-drain",
            "operator": "x",
            "reason": "f" * max(1, padding),
            "created_at": "2026-08-18T12:00:00Z",
            "close_sha256": stolen,
        }
        return (
            canonical_json_bytes(
                {
                    **body,
                    "row_sha256": row_content_address(body),
                    "chain": dict(intended["chain"]),
                }
            )
            + b"\n"
        )

    padding = 1
    while len(forged(padding)) < len(intended_line):
        padding += 1
    with ledger.open("ab") as handle:
        handle.write(forged(padding))


def _forged_genesis_uniform_downgrade(ledger: Path, anchor: Path) -> None:
    """Mint a cutover declaration and downgrade EVERY settlement behind it.

    PSLRE5 acct F2: the legacy-schema gate keyed on a producer-written declaration,
    and the monotone ladder is monotone rather than strictly increasing -- so
    inserting a genesis row that declares v1 and downgrading ALL settlements never
    moved below the ladder's own starting rank and zeroed the whole ledger.  Both
    halves are now structural: a legacy schema is never admissible for a row of this
    ledger, so the declaration grants nothing and the first downgraded row refuses.
    """

    rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    genesis = {
        "schema_version": "menagerie.crawler.paid-session-genesis.v1",
        "event": "accounting_genesis",
        "legacy_operational_path": "/dev/null",
        "legacy_operational_sha256": stable_hash({"bytes_hex": ""}),
        "legacy_operational_bytes": 0,
        "legacy_settlement_schemas": [PAID_SESSION_SETTLEMENT_SCHEMA_V1],
        "created_at": "2026-08-18T11:00:00Z",
    }
    _rebuild_ledger(
        ledger,
        anchor,
        [
            genesis,
            *(
                _downgraded_settlement(row, PAID_SESSION_SETTLEMENT_SCHEMA_V1)
                if row.get("event") == "session_settled"
                else row
                for row in rows
            ),
        ],
    )


def _ledger_deleted(ledger: Path, _anchor: Path) -> None:
    """The whole money ledger disappears while its anchor remains."""

    ledger.unlink()


def _anchor_deleted(_ledger: Path, anchor: Path) -> None:
    """The external extent pin disappears while the money ledger remains."""

    anchor.unlink()


def _ledger_replaced(ledger: Path, _anchor: Path) -> None:
    """An internally valid ledger of the same shape is swapped in."""

    replacement = ledger.with_name("replacement.jsonl")
    rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    rebuilt: list[dict[str, Any]] = []
    payload = b""
    for row in rows:
        clone = {key: value for key, value in row.items() if key != "chain"}
        clone["created_at"] = "2026-08-18T13:00:00Z"
        clone["row_sha256"] = stable_hash(
            {key: value for key, value in clone.items() if key != "row_sha256"}
        )
        linked = link_row(clone, scope="paid.sessions", existing_rows=rebuilt)
        rebuilt.append(linked)
        payload += canonical_json_bytes(linked) + b"\n"
    replacement.write_bytes(payload)
    ledger.write_bytes(payload)
    replacement.unlink()


#: The refusal an arm caught by the EXTERNAL extent pin must name.
_ANCHOR_REFUSAL = r"paid-session.*(anchor|extent|missing)"
#: The refusal an arm caught by the row's own re-derived content digest must name.
#: This is the ACCOUNTING authority's vocabulary; it is reachable on any path that
#: folds rows without the chain (:func:`index_rows`, and any future chain-free
#: reader), which is why it keeps its own dedicated arms below.
_ROW_DIGEST_REFUSAL = r"paid-session row digest does not re-derive"
#: The refusal an arm caught by the CHAIN's body-derived content address must name.
#: Since the address is derived rather than read, an in-place body edit is caught
#: here FIRST -- one authority earlier than the accounting re-derivation that used
#: to be the only one that noticed (PSLRE5).
_CHAIN_ADDRESS_REFUSAL = r"is not its own body content address"
#: The refusal an arm caught by the settlement-schema gates must name.
_SCHEMA_LADDER_REFUSAL = r"paid-session settlement schema .* (predates|is below)"

#: The anchor's BOTH-DIRECTIONS matrix, as
#: ``(arm, mutator, "RECOVER" | "REFUSE", refusal pattern)``. Every crash window
#: recovers and every tamper refuses, each REFUSE arm naming the exact authority
#: that catches it, so an arm can never pass on some unrelated error. An arm may
#: not be removed, and a pattern may not be broadened, without a reviewed change.
_ANCHOR_DIRECTION_MATRIX: tuple[
    tuple[str, Callable[[Path, Path], None], str, str], ...
] = (
    ("phase-three-lost", _anchor_phase_three_lost, "RECOVER", ""),
    ("torn-anchor-line", _anchor_torn_line, "RECOVER", ""),
    ("phase-one-crash", _anchor_phase_one_crash, "RECOVER", ""),
    ("ledger-tail-torn", _ledger_tail_torn, "RECOVER", ""),
    ("ledger-truncated", _ledger_truncated, "REFUSE", _ANCHOR_REFUSAL),
    ("anchor-digest-flipped", _anchor_digest_flipped, "REFUSE", _ANCHOR_REFUSAL),
    ("forged-settlement-appended", _forged_settlement_appended, "REFUSE", _ANCHOR_REFUSAL),
    (
        "forged-into-open-intent",
        _forged_settlement_into_open_intent,
        "REFUSE",
        _ANCHOR_REFUSAL,
    ),
    # PSLRE4 D-2: the same forgery with the intended row's digest COPIED, which
    # satisfied the open intent exactly until the digest was re-derived.  The chain
    # now derives its address from the body, so the copy is caught one authority
    # EARLIER than the accounting re-derivation (which keeps its own arms below).
    (
        "copied-row-digest-into-open-intent",
        _copied_row_digest_into_open_intent,
        "REFUSE",
        _CHAIN_ADDRESS_REFUSAL,
    ),
    # PSLRE5 P5-D-1: the arms that ADD a field rather than substituting bytes.  The
    # in-place arm rewrites a settled charge with the chain link, the byte count and
    # the anchor object all byte-identical; the intent arm satisfies an OPEN intent
    # with a substituted body.  Both were ACCEPTED under the digest ladder.
    (
        "close-sha256-pinned-in-place-edit",
        _pinned_close_sha256_in_place_edit,
        "REFUSE",
        _CHAIN_ADDRESS_REFUSAL,
    ),
    (
        "close-sha256-substituted-into-open-intent",
        _close_sha256_substituted_into_open_intent,
        "REFUSE",
        _CHAIN_ADDRESS_REFUSAL,
    ),
    # PSLRE4 F2: the charge rewrite a current-schema row refuses, smuggled through
    # a settlement-schema downgrade that skips the wall re-derivation entirely.
    (
        "settlement-downgraded-to-v1",
        _settlement_schema_downgrade(PAID_SESSION_SETTLEMENT_SCHEMA_V1),
        "REFUSE",
        _SCHEMA_LADDER_REFUSAL,
    ),
    (
        "settlement-downgraded-to-v2",
        _settlement_schema_downgrade(PAID_SESSION_SETTLEMENT_SCHEMA_V2),
        "REFUSE",
        _SCHEMA_LADDER_REFUSAL,
    ),
    # PSLRE5 acct F2: a MINTED cutover declaration plus a UNIFORM downgrade, which
    # the monotone ladder could not express -- rank 0 never moves below itself.
    (
        "forged-genesis-uniform-downgrade",
        _forged_genesis_uniform_downgrade,
        "REFUSE",
        _SCHEMA_LADDER_REFUSAL,
    ),
    ("ledger-deleted", _ledger_deleted, "REFUSE", _ANCHOR_REFUSAL),
    ("anchor-deleted", _anchor_deleted, "REFUSE", _ANCHOR_REFUSAL),
    ("ledger-replaced", _ledger_replaced, "REFUSE", _ANCHOR_REFUSAL),
)


#: The crawler production package root the counter census walks.
_CRAWLER_PACKAGE_ROOT = Path(paid_accounting_module.__file__).resolve().parent

#: The counters the census MUST find. Anti-substitution only: the set itself is
#: DERIVED (see :func:`derived_counter_slots`), so this is a floor that fails if a
#: counter is deleted or renamed out from under the census, never the census's input.
#: ``close_reads._STATS`` is here because PSLRE6 acct F1 found it uncovered: the
#: previous census iterated a HAND-WRITTEN four-module list, so the physical-read
#: counters in the module every ledger derivation flows through had no census slot --
#: the same coverage-hole shape as PSLRE5 acct F3, one module over.
_REQUIRED_COUNTER_SLOTS = frozenset(
    {
        ("paid_session_accounting.py", "ACCOUNTING_STATS"),
        ("paid_session_durability.py", "DURABILITY_STATS"),
        ("ledger_chain.py", "LEDGER_CHAIN_STATS"),
        ("identity.py", "IDENTITY_STATS"),
        ("close_reads.py", "_STATS"),
    }
)


def _production_crawler_modules() -> tuple[Path, Path]:
    """Yield every production crawler module path, tests and caches excluded."""

    return tuple(  # type: ignore[return-value]
        path
        for path in sorted(_CRAWLER_PACKAGE_ROOT.rglob("*.py"))
        if path.relative_to(_CRAWLER_PACKAGE_ROOT).parts[0] not in {"tests", "__pycache__"}
    )


def _subscript_mutated_names(tree: ast.AST) -> set[str]:
    """Return every name a module assigns THROUGH a subscript, at any depth."""

    names: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                names.add(target.value.id)
    return names


#: The module-level integer dicts in the crawler package that are READ-ONLY CONSTANTS
#: rather than counters, disclosed by name so the discovery below can be TOTAL over
#: every integer dict in the package instead of guessing which ones are counters.
#: Each is asserted never to be subscript-mutated, so one cannot quietly become mutable
#: state and slip out of the write-only property by being on this list.
_READ_ONLY_INTEGER_DICTS = frozenset(
    {
        ("authority.py", "_MODE_ORDER"),
        ("authority.py", "_ENVIRONMENT_TREE_WALK_REGISTRY"),
        ("checker_calibration.py", "CALIBRATION_MIN_SESSIONS"),
        # The calibration runner's per-lane measurement floors and buy caps:
        # decision VOCABULARIES of the sanctioned capture run, never counters
        # (the never-mutated assertion below still applies).
        ("tools/checker_calibration_runner.py", "MEASURE_TARGETS"),
        ("tools/checker_calibration_runner.py", "MEASURE_BUY_CAPS"),
        # The fold's closed per-lane attempt session bounds (PSLRETRY round-4
        # F1): a decision VOCABULARY like the closed lane set, not a counter.
        ("paid_session_accounting.py", "LANE_ATTEMPT_SESSION_BOUNDS"),
        # D-35/D-58 closed verdict->exit-code vocabularies: watchers branch on
        # these instead of string-matching rendered lines. Constants, never
        # counters; the never-mutated assertion below still applies.
        ("tools/crawl_doctor.py", "VERDICT_EXIT_CODES"),
        ("tools/deadman_watchdog.py", "CHECK_VERDICTS"),
    }
)


def module_level_integer_dicts(source: str) -> tuple[tuple[str, bool], ...]:
    """Return every module-level integer dict as ``(name, is subscript-mutated)``.

    THE derivation behind the census's module set (PSLRE6 acct F1). It names NOTHING:
    the shape is a top-level assignment of a dict literal whose values are all integer
    constants, i.e. every candidate for module-level mutable integer state in the
    package. Whether the module mutates it through a subscript (``X[k] += 1`` /
    ``X[k] = 0``) is reported rather than used to filter, so the caller can require that
    EVERY integer dict is either a counter under the write-only property or a disclosed
    read-only constant -- with no third category to hide in.

    The previous census's module list was hand-written AND its inner rule additionally
    required the name to be UPPER-CASE and to end in ``STATS``, so three separate
    escapes existed: a counter in an unnamed module (``close_reads._STATS``, the finding),
    a counter spelled lower-case, and a counter spelled ``LEDGER_COUNTERS``. Discovering
    by shape over the whole package closes all three at once.
    """

    tree = ast.parse(source)
    mutated = _subscript_mutated_names(tree)
    found: list[tuple[str, bool]] = []
    for node in tree.body:
        targets = (
            [node.target]
            if isinstance(node, ast.AnnAssign)
            else list(node.targets)
            if isinstance(node, ast.Assign)
            else []
        )
        value = getattr(node, "value", None)
        if not isinstance(value, ast.Dict) or not value.keys:
            continue
        if not all(
            isinstance(item, ast.Constant)
            and isinstance(item.value, int)
            and not isinstance(item.value, bool)
            for item in value.values
        ):
            continue
        found.extend(
            (target.id, target.id in mutated)
            for target in targets
            if isinstance(target, ast.Name)
        )
    return tuple(sorted(found))


def module_level_counter_dicts(source: str) -> tuple[str, ...]:
    """Return every module-level integer dict that is not a disclosed constant."""

    return tuple(name for name, _mutated in module_level_integer_dicts(source))


def derived_counter_slots() -> tuple[tuple[str, str], ...]:
    """Return every ``(module file name, counter)`` in the whole crawler package.

    Total over integer dicts, minus the disclosed read-only constants.
    """

    slots: list[tuple[str, str]] = []
    for path in _production_crawler_modules():
        module_file = str(path.relative_to(_CRAWLER_PACKAGE_ROOT))
        for name, _mutated in module_level_integer_dicts(path.read_text(encoding="utf-8")):
            if (module_file, name) in _READ_ONLY_INTEGER_DICTS:
                continue
            slots.append((module_file, name))
    return tuple(sorted(slots))


def _production_called_names() -> set[str]:
    """Return every function name CALLED anywhere in production crawler code."""

    called: set[str] = set()
    for path in _production_crawler_modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name):
                called.add(function.id)
            elif isinstance(function, ast.Attribute):
                called.add(function.attr)
    return called


def _enclosing_function_of_line(tree: ast.AST, line: int) -> str:
    """Return the innermost function definition enclosing one source line."""

    best = "<module>"
    best_start = -1
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = node.end_lineno or node.lineno
        if node.lineno <= line <= end and node.lineno > best_start:
            best = node.name
            best_start = node.lineno
    return best


def counter_read_lines(source: str, counter: str) -> list[int]:
    """Return every line where one write-only counter is READ, structurally.

    THE scanner both the census and its own negative control run.

    It is an AST pass, not a line filter (PSLRE5 P5-D-4).  The old filter skipped
    any line containing ``+=`` -- there to skip the increment itself -- so an
    ordinary accumulation, ``total += DURABILITY_STATS["anchor_writes"]``, made the
    counter steer a decision with the census fully green.  Here a read is any LOAD
    of the counter name that is not part of its own assignment target and not the
    ``for key in <counter>`` of its reset, so no textual spelling escapes: comments
    and docstrings are not AST nodes at all, and an accumulation is a load.
    """

    tree = ast.parse(source)
    written: set[int] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            for inner in ast.walk(target):
                if isinstance(inner, ast.Name) and inner.id == counter:
                    written.add(id(inner))
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name):
            if node.iter.id == counter:
                written.add(id(node.iter))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and node.id == counter
        and isinstance(node.ctx, ast.Load)
        and id(node) not in written
    )


def test_the_write_only_counter_scanner_sees_every_read_spelling() -> None:
    """The counter scanner is not vacuous, including the ``+=`` accumulation.

    Four arms: the increment and the reset loop are NOT reads; a plain read is; and
    the PSLRE5 P5-D-4 spelling -- accumulating the counter into a running total, on
    a line the old textual filter skipped because it contained ``+=`` -- is too.
    """

    innocent = (
        'STATS: dict[str, int] = {"a": 0}\n'
        "def reset():\n"
        "    for key in STATS:\n"
        "        STATS[key] = 0\n"
        "def bump():\n"
        '    STATS["a"] += 1\n'
    )
    assert counter_read_lines(innocent, "STATS") == []

    plain_read = innocent + 'def peek():\n    return STATS["a"]\n'
    assert counter_read_lines(plain_read, "STATS") == [8]

    accumulated = innocent + (
        "def decide(self):\n"
        "    total = 0\n"
        '    total += STATS["a"]\n'
        "    if total > 10:\n"
        '        raise ValueError("budget exhausted")\n'
    )
    assert counter_read_lines(accumulated, "STATS") == [9]

    aliased = innocent + "def leak():\n    return dict(STATS)\n"
    assert counter_read_lines(aliased, "STATS") == [8]


def test_every_write_only_counter_is_under_the_separation_census() -> None:
    """The census's module set is DERIVED from the package, not hand-written.

    PSLRE5 acct F3 was a coverage hole, not a broken check: the check existed and
    worked, and the newest counter simply was not in its list. Making the list total
    over the modules IT NAMED left the same hole one module over, which is what PSLRE6
    acct F1 then found (``close_reads._STATS``, in the module every ledger derivation
    flows through). So the set is no longer an input at all: it is discovered from the
    whole crawler package BY SHAPE, and this asserts that the discovery still finds
    every counter the money substrate is known to carry.
    """

    derived = set(derived_counter_slots())
    assert _REQUIRED_COUNTER_SLOTS <= derived, (
        "the counter discovery lost a known counter: "
        f"{sorted(_REQUIRED_COUNTER_SLOTS - derived)}"
    )
    # Discovery is by shape over the WHOLE package, so a NEW counter anywhere is covered
    # automatically -- and the census below runs over the derived set, never a list.
    assert derived == _REQUIRED_COUNTER_SLOTS, (
        "a module-level integer dict appeared with no write-only census entry: "
        f"{sorted(derived - _REQUIRED_COUNTER_SLOTS)}; it is already under the "
        "write-only property below, so add it to _REQUIRED_COUNTER_SLOTS to disclose it "
        "(or to _READ_ONLY_INTEGER_DICTS if it is a constant, which is then asserted)"
    )

    # The only third category is the DISCLOSED read-only constant, and disclosure buys
    # nothing: each is asserted never to be subscript-mutated, so it cannot become
    # mutable state while sitting outside the write-only property.
    for module_file, name in sorted(_READ_ONLY_INTEGER_DICTS):
        source = (_CRAWLER_PACKAGE_ROOT / module_file).read_text(encoding="utf-8")
        declared = dict(module_level_integer_dicts(source))
        assert name in declared, f"{module_file}.{name} no longer exists: re-derive the list"
        assert not declared[name], (
            f"{module_file}.{name} is disclosed as a read-only constant but the module "
            "now mutates it through a subscript: it is module-level mutable state and "
            "belongs under the write-only census, not on the constants list"
        )


def test_the_counter_discovery_sees_the_spellings_the_named_list_missed() -> None:
    """The shape-based discovery is not vacuous, on all three escapes it closes."""

    named = (
        'STATS: dict[str, int] = {"a": 0}\n'
        "def bump():\n"
        '    STATS["a"] += 1\n'
    )
    assert module_level_counter_dicts(named) == ("STATS",)

    # Escape 1: a counter not spelled *STATS (the old inner rule required the suffix).
    renamed = named.replace("STATS", "LEDGER_COUNTERS")
    assert module_level_counter_dicts(renamed) == ("LEDGER_COUNTERS",)

    # Escape 2: a private lower-case counter (the old rule required UPPER-CASE).
    private = named.replace("STATS", "_stats")
    assert module_level_counter_dicts(private) == ("_stats",)

    # Escape 3: a counter DECLARED but not yet mutated. Discovery is total over integer
    # dicts and reports mutation rather than filtering on it, so a declaration cannot
    # sit outside the census waiting for its first increment.
    declared_only = 'LEDGER_COUNTERS: dict[str, int] = {"n": 0}\n'
    assert module_level_integer_dicts(declared_only) == (("LEDGER_COUNTERS", False),)
    assert module_level_counter_dicts(declared_only) == ("LEDGER_COUNTERS",)
    assert module_level_integer_dicts(named) == (("STATS", True),)

    # A dict whose values are not integers is not integer state at all.
    strings = 'NAMES = {"a": "b"}\ndef bump():\n    NAMES["a"] = "c"\n'
    assert module_level_counter_dicts(strings) == ()
    assert module_level_counter_dicts("EMPTY: dict[str, int] = {}\n") == ()


def _module_symbols(module: Any) -> set[str]:
    """Return every imported and globally referenced name in one module's AST."""

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _assert_cross_concern_separation() -> None:
    """THE anti-regression census: three concerns, zero shared mutable state.

    Six rounds of this surface failed the same way -- accounting, custody, and
    durability were coupled, so every fix regressed a neighbour.  This asserts
    the boundaries structurally, so a future fix in one concern CANNOT reach
    into another: custody imports no budget symbol, accounting touches no
    filesystem or process primitive, and durability knows no money vocabulary.
    """

    budget_vocabulary = {
        "BudgetPolicy",
        "Weight",
        "quota_spend",
        "runaway_spend",
        "campaign_spend",
        "charge",
        "charged_wall_seconds",
        "authorize",
        "settled_session_weight",
        "reserve",
        "settle",
        "AccountingState",
        "menagerie.crawler.paid_session_accounting",
        "menagerie.crawler.paid_session_lifecycle",
        "menagerie.crawler.paid_session_durability",
    }
    custody_symbols = _module_symbols(paid_custody_module)
    leaked = custody_symbols & budget_vocabulary
    assert not leaked, (
        f"the custody concern reaches budget vocabulary {sorted(leaked)}: a custody "
        "fix can now change a charge, which is exactly the coupling that made six "
        "rounds of this surface regress each other"
    )

    accounting_symbols = _module_symbols(paid_accounting_module)
    forbidden_effects = {
        "os",
        "fcntl",
        "open",
        "subprocess",
        "Path",
        "read_bytes",
        "write_bytes",
        "flock",
        "durable_fsync",
        "getenv",
        "environ",
        "menagerie.crawler.paid_session_custody",
        "menagerie.crawler.paid_session_durability",
    }
    impure = accounting_symbols & forbidden_effects
    assert not impure, (
        f"the accounting concern reaches {sorted(impure)}: a budget must be a PURE "
        "function of durable rows, with no way for an IO or process change to move it"
    )

    durability_symbols = _module_symbols(paid_durability_module)
    money_vocabulary = {
        "BudgetPolicy",
        "Weight",
        "quota_spend",
        "runaway_spend",
        "authorize",
        "settled_session_weight",
        "PaidSessionBudgetExceeded",
        "menagerie.crawler.paid_session_accounting",
        "menagerie.crawler.paid_session_custody",
        "menagerie.crawler.paid_session_lifecycle",
    }
    coupled = durability_symbols & money_vocabulary
    assert not coupled, (
        f"the durability concern reaches money vocabulary {sorted(coupled)}: a "
        "durability change could then alter a charge"
    )

    # The only module-level mutable state in any concern is the write-only scale
    # instrumentation, and no DECISION may read it.  The module set is DERIVED from the
    # whole crawler package by shape (PSLRE6 acct F1), so a counter added anywhere is
    # under this property the moment it is written -- no list to forget to update.
    #
    # "Write-only" is precise: a counter may be loaded ONLY inside a function that no
    # production module calls, i.e. a measurement surface the campaign never consults.
    # That is what makes ``close_reads.read_statistics`` (a test-only disclosure
    # accessor) compatible with the property while an inline read in any production
    # path is not -- and it is derived from the tree, with nothing a producer declares.
    production_calls = _production_called_names()
    for module_file, counter in derived_counter_slots():
        path = _CRAWLER_PACKAGE_ROOT / module_file
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for line in counter_read_lines(source, counter):
            owner = _enclosing_function_of_line(tree, line)
            assert owner not in production_calls and owner != "<module>", (
                f"{module_file}:{line} reads {counter} inside {owner}(), which "
                "production code calls: a scale counter may never steer a decision"
            )


def _assert_no_hydration_or_mutable_counter() -> None:
    """No spend counter, no hydration input, no subtractive money operation.

    This is the single structural move that ended the round-2/3/4 oscillation:
    a budget is a pure function of durable rows evaluated at the chokepoint, so
    there is nothing to hydrate, nothing to reset, and nothing to restore.
    """

    for module in (paid_lifecycle, paid_accounting_module, paid_durability_module):
        source = Path(cast(str, module.__file__)).read_text(encoding="utf-8")
        for banned in (
            "restore_durable_spend",
            "_leg_wall_spent",
            "_leg_sessions",
            "_model_wall_spent",
            "hydrate",
        ):
            assert banned not in source, f"{banned} reappeared in {module.__name__}"
    # The instrument set is CLOSED, each instrument names exactly ONE axis, and
    # every axis field has a named machine provenance.
    assert set(INSTRUMENTS) == {"QUOTA", "RUNAWAY_BREAKER"}
    assert INSTRUMENTS["QUOTA"] != INSTRUMENTS["RUNAWAY_BREAKER"]
    assert set().union(*INSTRUMENTS.values()) == REQUIRED_AXIS_FIELDS
    assert set(ACCOUNTING_AXIS_PROVENANCE) == REQUIRED_AXIS_FIELDS
    assert MONEY_EVENTS == frozenset({"session_reserved", "session_settled"})
    # There is exactly one settlement writer and it is provider-driven.
    lifecycle_source = Path(paid_lifecycle.__file__).read_text(encoding="utf-8")
    assert lifecycle_source.count('"event": "session_settled"') == 1


def _assert_production_provider_reconciliation_option() -> None:
    """Production exposes a durable provider-query command for crash reconciliation."""

    parser = build_parser()
    run_parser_action = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None)
        and "run" in cast(Mapping[str, Any], action.choices)
    )
    run_parser = cast(Mapping[str, Any], run_parser_action.choices)["run"]
    assert any(
        "--provider-reconcile-command" in action.option_strings
        for action in run_parser._actions
    )


def _assert_leg_start_crash_reconciliation(tmp_path: Path) -> None:
    """A durable pre-settlement crash is queried terminally before the next buy."""

    tmp_path.mkdir(parents=True)
    query_script = tmp_path / "provider-query.py"
    observed_key = tmp_path / "queried-key.txt"
    expected_key_file = tmp_path / "expected-key.txt"
    # An honest adapter shape: refuses keys outside its ledger (here, the one
    # expected reservation key), so the admission boundary's standing
    # canary-nonce negative control passes instead of flagging the rig as a
    # constant-answer adapter.
    query_script.write_text(
        "import json, pathlib, sys\n"
        f"expected = pathlib.Path({str(expected_key_file)!r}).read_text(encoding='utf-8')\n"
        "if sys.argv[1] != expected:\n"
        "    print('unknown provider idempotency key', file=sys.stderr)\n"
        "    raise SystemExit(3)\n"
        f"pathlib.Path({str(observed_key)!r}).write_text(sys.argv[1])\n"
        "print(json.dumps({'idempotency_key': sys.argv[1], 'accepted': False, "
        "'observed_wall_seconds': 0.0, 'adapter_identity': 'rig-query/1'}))\n",
        encoding="utf-8",
    )
    silent_script = tmp_path / "provider-query-wrong-key.py"
    silent_script.write_text(
        "import json\n"
        "print(json.dumps({'idempotency_key': 'some-other-session', "
        "'accepted': True, 'observed_wall_seconds': 1.0, "
        "'adapter_identity': 'rig-wrong-key/1'}))\n",
        encoding="utf-8",
    )
    driver_root = tmp_path / "driver"
    driver_root.mkdir()
    snapshot = _snapshot(driver_root, count=1)
    driver = _driver(driver_root, snapshot)
    driver.dependencies = replace(
        driver.dependencies,
        provider_reconciler=CommandProviderReconciler((sys.executable, str(query_script))),
    )
    authority = driver._paid_session_authority()
    reservation = authority.reserve(
        scope=_scope(("m_crashed", "work-crashed")),
        policy=_policy(quota_wall=100_000.0),
    )
    assert authority.outstanding_spend_ids() == (reservation.spend_id,)
    expected_key_file.write_text(reservation.provider_idempotency_key, encoding="utf-8")

    driver._reconcile_paid_sessions_before_admission()

    assert observed_key.read_text(encoding="utf-8") == reservation.provider_idempotency_key
    assert authority.outstanding_spend_ids() == ()
    authority.reserve(
        scope=_scope(("m_next", "work-next")),
        policy=_policy(quota_wall=100_000.0),
    )
    # The echoed key is what makes the answer an attestation ABOUT a reservation
    # rather than a self-describing string that attests only itself.
    wrong_key = CommandProviderReconciler((sys.executable, str(silent_script)))
    with pytest.raises(PaidSessionError, match="different idempotency key"):
        wrong_key(reservation.provider_idempotency_key)


def _assert_paid_session_anchor(tmp_path: Path) -> None:
    """The anchor recovers crash tails while still refusing ledger rollback."""

    for arm, mutate, expected, pattern in _ANCHOR_DIRECTION_MATRIX:
        reset_incremental_reads()
        authority = _authority(tmp_path / arm)
        first = authority.reserve(
            scope=_scope(("m_first", "work-first")), policy=_policy(quota_wall=100_000.0)
        )
        authority.settle(
            first,
            provider_query=_provider_query(
                # A PRODUCTION-shaped receipt digest (the real 71-character
                # spelling), so the in-place arms rewrite a row whose disclosure
                # fields donate the byte length a real forger has to work with.
                accepted=True,
                wall_seconds=1.0,
                digest="sha256:" + "ab" * 32,
            ),
            # Also production-shaped: this is the disposition the crash-reconcile
            # path records, and the in-place arms need the row to look like a real
            # settlement rather than a short test token.
            custody_outcome="reconciled-provider-accepted",
            machine_wall_seconds=1.0,
            local_outcome="completed",
        )
        authority.reserve(
            scope=_scope(("m_second", "work-second")),
            policy=_policy(quota_wall=100_000.0),
        )
        rows_before = len(load_paid_session_rows(authority.path))
        anchor = authority.path.with_name(f"{authority.path.name}.anchor.json")
        mutate(authority.path, anchor)
        reset_incremental_reads()
        if expected == "RECOVER":
            recovered = load_paid_session_rows(authority.path)
            assert len(recovered) in {rows_before - 1, rows_before}, arm
            verify_records_integrity(authority.path.parent.parent)
            continue
        with pytest.raises(ValueError, match=pattern):
            load_paid_session_rows(authority.path)
        reset_incremental_reads()
        with pytest.raises(LedgerChainError, match=pattern):
            verify_records_integrity(authority.path.parent.parent)


def test_leg_breaker_is_campaign_scoped_retryable() -> None:
    """The breaker must ride the loud re-raise arms, exactly like park-streak."""

    assert issubclass(CheckerLegBudgetExceeded, RetryableOperatorError)


def test_accounting_attributes_wall_to_every_member_and_the_leg(tmp_path: Path) -> None:
    """Attribute a batch reservation fully to each work key and once to its epoch."""

    authority = _authority(tmp_path)
    reserved = authority.reserve(
        scope=_scope(("m_a", "work-a"), ("m_b", "work-b")),
        policy=_policy(quota_wall=100_000.0),
    )
    rows = load_paid_session_rows(authority.path)
    assert quota_spend(rows, "c1-mech", "m_a", "work-a").wall_seconds == 100.0
    assert quota_spend(rows, "c1-mech", "m_b", "work-b").wall_seconds == 100.0
    assert runaway_spend(rows, "c1-mech", "generation-open").wall_seconds == 100.0
    authority.settle(
        reserved,
        provider_query=_provider_query(
            accepted=True,
            wall_seconds=37.5,
            digest="sha256:accounting-settlement",
        ),
        custody_outcome="exited",
        machine_wall_seconds=37.5,
        local_outcome="completed",
    )
    settled = load_paid_session_rows(authority.path)
    assert quota_spend(settled, "c1-mech", "m_a", "work-a").wall_seconds == 37.5
    authority.reserve(
        scope={
            **_scope(("m_a", "work-a")),
            "lane": "author",
            "dispatch_kind": "author_source",
        },
        policy=BudgetPolicy(100, 100_000.0, 100, 100_000.0, 100.0),
    )
    shared_rows = load_paid_session_rows(authority.path)
    assert quota_spend(shared_rows, "c1-mech", "m_a", "work-a").sessions == 2
    assert quota_spend(
        shared_rows, "c1-mech", "m_a", "work-a", lane="checker"
    ).sessions == 1
    assert runaway_spend(shared_rows, "c1-mech", "generation-open").sessions == 2
    assert runaway_spend(
        shared_rows, "c1-mech", "generation-open", lane="checker"
    ).sessions == 1
    with pytest.raises(ValueError, match="already settled"):
        authority.settle(
            reserved,
            provider_query=_provider_query(
                accepted=True,
                wall_seconds=1.0,
                digest="sha256:duplicate-settlement",
            ),
            custody_outcome="exited",
            machine_wall_seconds=1.0,
            local_outcome="completed",
        )
    # THE custody topology matrix. ``paid_grandchild_credential`` is the row
    # rounds 1-3 never had: the process that holds the credential and does the
    # buying is usually the grandchild, and every earlier closure reasoned only
    # about the direct child.
    assert CUSTODY_TOPOLOGIES == frozenset(
        {
            "driver_sigkill",
            "operator_different_group",
            "mid_session_reboot",
            "wrapper_hang",
            "paid_grandchild_credential",
        }
    )
    topology_facts = {
        # (boot at reconcile, flock answer, group answer, identity, action)
        "driver_sigkill": ("boot-one", UNANSWERABLE, FREE, False, "unreapable-refuse"),
        "operator_different_group": ("boot-one", ALIVE, FREE, True, "reap-verified-group"),
        "mid_session_reboot": ("boot-two", UNANSWERABLE, FREE, False, "stale-boot"),
        "wrapper_hang": ("boot-one", ALIVE, ALIVE, True, "reap-verified-group"),
        # The credential-bearing grandchild: it dropped out of the process group
        # but still holds the inherited lock, so the second oracle answers.
        "paid_grandchild_credential": ("boot-one", ALIVE, FREE, False, "unreapable-refuse"),
        # A clean subtree: BOTH oracles free, so the machine has its own evidence
        # and no wrapper needs to assert anything about itself.
        "clean_exit": ("boot-one", FREE, FREE, False, "already-dead"),
        # The fd-dropping descendant the flock alone would miss.
        "group_only_survivor": ("boot-one", FREE, ALIVE, False, "unreapable-refuse"),
    }
    for topology, (
        boot_id,
        lock_state,
        group_state,
        identity_matches,
        expected_action,
    ) in topology_facts.items():
        lease = PaidCustodyLease(
            lease_id=f"lease-{topology}",
            spend_id=f"spend-{topology}",
            boot_id="boot-one",
            host_id="host-one",
            lock_path=tmp_path / f"{topology}.lock",
            leader_pid=123,
            leader_start_token="start-one",
            leader_pgid=123,
            deadline_at="2026-08-18T12:30:00Z",
            reserved_wall_seconds=100.0,
            lock_device=1,
            lock_inode=2,
        )
        decision = custody_recovery_decision(
            lease,
            current_boot_id=boot_id,
            liveness=lock_state,
            group_liveness=group_state,
            identity_matches=identity_matches,
            past_deadline=True,
        )
        assert decision.financially_bounded and decision.disclosed, topology
        assert decision.credential_path is None, topology
        assert decision.action == expected_action, topology
    # An unverifiable live identity is NEVER certified as exited, in either
    # oracle, and never carries a positive outcome.
    for lock_state, group_state in ((ALIVE, FREE), (FREE, ALIVE), (UNANSWERABLE, FREE)):
        residual = custody_recovery_decision(
            PaidCustodyLease(
                lease_id="lease-darwin-residual",
                spend_id="spend-darwin-residual",
                boot_id="boot-one",
                host_id="host-one",
                lock_path=tmp_path / "residual.lock",
                leader_pid=123,
                leader_start_token="start-one",
                leader_pgid=123,
                deadline_at="2026-08-18T12:30:00Z",
                reserved_wall_seconds=100.0,
            ),
            current_boot_id="boot-one",
            liveness=lock_state,
            group_liveness=group_state,
            identity_matches=False,
            past_deadline=True,
        )
        assert residual.action == "unreapable-refuse"
        assert residual.financially_bounded and residual.disclosed
        assert "Darwin residual" in residual.detail
    assert process_group_liveness(None) == FREE
    # The group probe reuses the supervisor's membership enumeration, which
    # answers UNANSWERABLE on a host that cannot enumerate processes at all. It
    # must therefore never claim a group is FREE that is not, and -- because it
    # is the strictly ADDITIVE oracle -- its silence must never block on its own.
    assert process_group_liveness(os.getpgrp()) in {ALIVE, UNANSWERABLE}
    assert custody_recovery_decision(
        PaidCustodyLease(
            lease_id="lease-group-unanswerable",
            spend_id="spend-group-unanswerable",
            boot_id="boot-one",
            host_id="host-one",
            lock_path=tmp_path / "group-unanswerable.lock",
            leader_pid=123,
            leader_start_token="start-one",
            leader_pgid=123,
            deadline_at="2026-08-18T12:30:00Z",
            reserved_wall_seconds=100.0,
        ),
        current_boot_id="boot-one",
        liveness=FREE,
        group_liveness=UNANSWERABLE,
        identity_matches=False,
        past_deadline=True,
    ).action == "already-dead", (
        "a host that cannot enumerate process groups would refuse every clean "
        "paid session; the flock is the authoritative oracle and the group probe "
        "may only ever ADD a refusal"
    )
    # The restart-action vocabulary is CLOSED, and the matrix above exercises
    # every member of it: a new disposition cannot appear unreviewed, and an
    # existing one cannot silently stop being covered.
    assert CUSTODY_ACTIONS == {
        "already-dead",
        "stale-boot",
        "reap-verified-group",
        "active-refuse",
        "unreapable-refuse",
    }
    exercised = {action for _, _, _, _, action in topology_facts.values()}
    assert exercised | {"active-refuse"} == CUSTODY_ACTIONS
    inside_deadline = custody_recovery_decision(
        PaidCustodyLease(
            lease_id="lease-active",
            spend_id="spend-active",
            boot_id="boot-one",
            host_id="host-one",
            lock_path=tmp_path / "active.lock",
            leader_pid=123,
            leader_start_token="start-one",
            leader_pgid=123,
            deadline_at="2026-08-18T12:30:00Z",
            reserved_wall_seconds=100.0,
        ),
        current_boot_id="boot-one",
        liveness=ALIVE,
        group_liveness=ALIVE,
        identity_matches=True,
        past_deadline=False,
    )
    assert inside_deadline.action == "active-refuse"
    _assert_settled_custody_disclosure(tmp_path / "settled-custody")
    _assert_missing_flock_fail_closed(tmp_path / "missing-flock")
    _assert_clean_wrapper_releases_without_self_certification(tmp_path / "clean-wrapper")
    _assert_grandchild_credential_custody(tmp_path / "grandchild")
    _assert_drain_refuses_an_unlinked_lock(tmp_path / "unlinked-lock")
    _assert_operator_custody_drain(tmp_path / "operator-drain")
    sentinel = b"PSL-SENTINEL-CREDENTIAL"
    with anonymous_credential(sentinel, directory=tmp_path / "credential") as credential:
        assert os.pread(credential.fd, len(sentinel), 0) == sentinel
        assert all(
            sentinel not in path.read_bytes()
            for path in (tmp_path / "credential").iterdir()
        )
    lock_path = tmp_path / "subtree.lock"
    handle = open_subtree_lock(lock_path)
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            pass_fds=(handle.fd,),
        )
    finally:
        handle.close_parent_copy()
    try:
        assert subtree_is_alive(lock_path)
    finally:
        child.terminate()
        child.wait(timeout=5.0)
    assert not subtree_is_alive(lock_path)


def test_wall_exhaustion_parks_with_its_own_failure_class() -> None:
    error = CheckerWallBudgetExceeded("spent", stable_ids=("m_hot",))
    assert _checker_lane_failure_class(error) == "checker-wall-exhausted"


# ---------------------------------------------------------------------------
# Driver routing: the leg breaker never retries, never blames a model.
# ---------------------------------------------------------------------------


class ExhaustedLegChecker(ScriptedChecker):
    """Raise the leg breaker on every dispatch, counting invocations."""

    def __init__(self) -> None:
        super().__init__()
        self.invocations = 0

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        del artifacts, work_root, config
        self.invocations += 1
        raise CheckerLegBudgetExceeded("checker leg budget exhausted: synthetic")


def test_leg_breaker_stops_the_run_loud_without_retries_or_blame(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path, count=2)
    checker = ExhaustedLegChecker()
    with pytest.raises(CheckerLegBudgetExceeded):
        _driver(tmp_path, snapshot, checker=checker).run()

    # At a multi-item width the two models share one batch; at width 1 they
    # are two batches, and a sibling already submitted when the breaker fires
    # still refuses PRE-SPEND on its own reserve (drain-never-cancel). What may
    # never happen is a RE-invocation of a refused batch.
    assert 1 <= checker.invocations <= -(-2 // METADATA_BATCH_MAX), (
        "a budget breaker is not weather; the retry ladder must not re-invoke it"
    )
    paths = _paths(tmp_path, snapshot)
    assert _valve_rows(tmp_path, snapshot) == []
    models = scan_jsonl(paths.ledgers.models) if paths.ledgers.models.exists() else []
    assert models == [], "the breaker blames nobody; the campaign pauses resumably"
