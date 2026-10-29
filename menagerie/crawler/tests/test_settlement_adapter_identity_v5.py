"""D-1 rider: settlement rows carry ``provider.adapter_identity`` (schema v5).

Wave-1 put the adapter's name/version stamp inside the digested status receipt
(``ProviderStatusFields``), so ``receipt_sha256`` already covers WHICH reconciler
answered. What was still missing is row LEGIBILITY: reading a settlement row
alone could not name the adapter. Settlement schema v5 promotes the stamp to
the row's ``provider`` block. The ladder stays monotone: v4 rows validate on
their own original terms forever, v5 is the only minted version, and the v4-era
machine-floor checks (wall re-derivation, refund semantics) keep applying to v4
rows -- versioning up must never weaken what already held.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.paid_session_accounting import (
    ATTEMPT_REFUND_LOCAL_OUTCOMES,
    LEGACY_SETTLEMENT_SCHEMAS,
    PAID_SESSION_SETTLEMENT_SCHEMA,
    SETTLEMENT_SCHEMA_LADDER,
    settled_attempt_weight,
    settled_session_weight,
    validate_settlement,
)
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    ProviderReconciliation,
    load_paid_session_rows,
)

pytestmark = pytest.mark.smoke

ADAPTER = "provider-reconcile-evidence/2"


def _authority(tmp_path: Path) -> PaidSessionAuthority:
    """Return one temporary paid-session authority with a deterministic clock."""

    return PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-24T12:00:00Z",
    )


def _scope(stable_id: str, work_id: str) -> dict[str, Any]:
    """Return one author reservation scope carrying both accounting axes."""

    return {
        "campaign_id": "c1-mech",
        "generation_id": "generation-open",
        "lane": "author",
        "dispatch_kind": "author_source",
        "attempt_group": None,
        "work_generations": ((stable_id, work_id),),
        "gate_kind": None,
        "checker_identity": None,
        "run_id": "run-one",
        "machine_id": "host-one",
    }


def _policy() -> BudgetPolicy:
    """Return a policy sized for one settled session."""

    return BudgetPolicy(
        quota_sessions=10,
        quota_wall_seconds=10_000.0,
        runaway_sessions=10,
        runaway_wall_seconds=10_000.0,
        reservation_wall_seconds=100.0,
    )


def _settlement_row(
    schema: str,
    *,
    provider: Any,
    local_outcome: str = "completed",
) -> dict[str, Any]:
    """Return one minimally valid current-family settlement row."""

    return {
        "schema_version": schema,
        "event": "session_settled",
        "spend_id": "spend-1",
        "observed": {"wall_seconds": 5.0},
        "evidence": "provider-query",
        "custody_outcome": "exited",
        "local_outcome": local_outcome,
        "provider": provider,
        "machine": {"wall_seconds": 5.0, "source": "writer-monotonic"},
    }


def _v5_provider(adapter_identity: Any = ADAPTER) -> dict[str, Any]:
    """Return one v5 provider truth block."""

    return {
        "accepted": True,
        "attested_wall_seconds": 5.0,
        "receipt_sha256": "sha256:" + "d" * 64,
        "adapter_identity": adapter_identity,
    }


def _v4_provider() -> dict[str, Any]:
    """Return one v4 provider truth block, its own original exact shape."""

    return {
        "accepted": True,
        "attested_wall_seconds": 5.0,
        "receipt_sha256": "sha256:" + "d" * 64,
    }


def _v4_schema() -> str:
    """Return the v4 settlement schema identifier from the closed ladder."""

    return SETTLEMENT_SCHEMA_LADDER[-2]


def test_ladder_grew_monotonically_to_v5() -> None:
    """v5 is current, v4 stays post-cutover-admissible, legacy stays v1-v3."""

    assert PAID_SESSION_SETTLEMENT_SCHEMA.endswith(".v5")
    assert SETTLEMENT_SCHEMA_LADDER[-1] == PAID_SESSION_SETTLEMENT_SCHEMA
    assert SETTLEMENT_SCHEMA_LADDER[-2].endswith(".v4")
    assert SETTLEMENT_SCHEMA_LADDER[-2] not in LEGACY_SETTLEMENT_SCHEMAS
    assert len(SETTLEMENT_SCHEMA_LADDER) == len(set(SETTLEMENT_SCHEMA_LADDER))


def test_writer_mints_v5_with_adapter_identity(tmp_path: Path) -> None:
    """The one settlement writer stamps the answering adapter on the row."""

    authority = _authority(tmp_path)
    reservation = authority.reserve(scope=_scope("m1", "work-1"), policy=_policy())
    authority.settle(
        reservation,
        provider_query=lambda key: ProviderReconciliation(
            True, 5.0, f"sha256:receipt-{key[-12:]}", adapter_identity=ADAPTER
        ),
        custody_outcome="exited",
        machine_wall_seconds=5.0,
        local_outcome="completed",
    )
    rows = load_paid_session_rows(authority.path)
    settled = [row for row in rows if row["event"] == "session_settled"]
    assert len(settled) == 1
    assert settled[0]["schema_version"] == PAID_SESSION_SETTLEMENT_SCHEMA
    assert settled[0]["provider"]["adapter_identity"] == ADAPTER


def test_writer_discloses_null_identity_for_in_process_reconcilers(
    tmp_path: Path,
) -> None:
    """An in-process rig reconciler discloses ``None``, never a fabricated name."""

    authority = _authority(tmp_path)
    reservation = authority.reserve(scope=_scope("m2", "work-2"), policy=_policy())
    authority.settle(
        reservation,
        provider_query=lambda key: ProviderReconciliation(
            True, None, f"sha256:receipt-{key[-12:]}"
        ),
        custody_outcome="exited",
        machine_wall_seconds=1.0,
        local_outcome="completed",
    )
    rows = load_paid_session_rows(authority.path)
    settled = [row for row in rows if row["event"] == "session_settled"]
    assert settled[0]["provider"]["adapter_identity"] is None


def test_v5_validator_enforces_adapter_identity_shape() -> None:
    """v5 requires the key; the value is a nonempty string or a disclosed null."""

    validate_settlement(
        _settlement_row(PAID_SESSION_SETTLEMENT_SCHEMA, provider=_v5_provider())
    )
    validate_settlement(
        _settlement_row(PAID_SESSION_SETTLEMENT_SCHEMA, provider=_v5_provider(None))
    )
    missing = _v5_provider()
    del missing["adapter_identity"]
    with pytest.raises(ValueError, match="provider truth"):
        validate_settlement(
            _settlement_row(PAID_SESSION_SETTLEMENT_SCHEMA, provider=missing)
        )
    with pytest.raises(ValueError, match="adapter identity"):
        validate_settlement(
            _settlement_row(PAID_SESSION_SETTLEMENT_SCHEMA, provider=_v5_provider(""))
        )
    with pytest.raises(ValueError, match="adapter identity"):
        validate_settlement(
            _settlement_row(PAID_SESSION_SETTLEMENT_SCHEMA, provider=_v5_provider(7))
        )


def test_v4_rows_stay_valid_on_their_own_exact_terms() -> None:
    """A historical v4 row validates forever; a v5 field on it refuses."""

    validate_settlement(_settlement_row(_v4_schema(), provider=_v4_provider()))
    smuggled = _settlement_row(_v4_schema(), provider=_v5_provider())
    with pytest.raises(ValueError, match="provider truth"):
        validate_settlement(smuggled)


def test_v4_rows_keep_machine_floor_semantics() -> None:
    """Versioning up never weakens v4: refund and session weights still apply."""

    refund = list(sorted(ATTEMPT_REFUND_LOCAL_OUTCOMES))[0]
    v4_row = _settlement_row(
        _v4_schema(), provider=_v4_provider(), local_outcome=refund
    )
    v5_row = _settlement_row(
        PAID_SESSION_SETTLEMENT_SCHEMA, provider=_v5_provider(), local_outcome=refund
    )
    assert settled_session_weight(v4_row) == 1
    assert settled_session_weight(v5_row) == 1
    assert settled_attempt_weight(v4_row) == 0
    assert settled_attempt_weight(v5_row) == 0


def test_unknown_future_settlement_schema_refuses() -> None:
    """A schema outside the closed ladder refuses before anything else."""

    row = _settlement_row(
        "menagerie.crawler.paid-session-settlement.v6", provider=_v5_provider()
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_settlement(row)
