"""W42 close half: operator-act census + live-grant disclosure + void-vs-grant refusal.

OS-10's close-side triple: every in-window operator act censused (with D-53's
derived identity spellings), every LIVE requeue grant disclosed with its
hold_reason, and the one refusing arm -- a member serviced by an in-window
VOID while its grant stays LIVE is a receipt-less intervention beside an open
authorization (D-50 forbids receipt-less intervention; this refuses the close
that would launder one). The sanctioned drain (void with NO live grant, or
consuming the grant itself) never trips it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    canonical_operational_ledger_path,
    canonical_requeue_grants_path,
)
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
    OperationalEventStatus,
)
from menagerie.crawler.operator_provenance import (
    operator_act_census,
    void_only_with_live_grant_findings,
)
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.reducer import default_ledger_paths
from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign
from menagerie.crawler.tests.rt3_support import (
    append_refusal_park,
    void_refusal_park,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot

AS_OF = "2026-08-25T12:00:00Z"


def _seed_ledgers(records_root: Path):
    ledgers = default_ledger_paths(records_root)
    for name in ("models", "attempts", "gates", "artifacts"):
        path = getattr(ledgers, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"")
    return ledgers


def _mint_grant(records_root: Path, stable_id: str) -> dict:
    ledgers = default_ledger_paths(records_root)
    canonical = canonical_requeue_grants_path(ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    grant = build_canonical_requeue_grant(
        canonical,
        stable_id=stable_id,
        stage="accuracy-gate",
        reason="pytest re-adjudication",
        attempts=1,
        granted_by="pytest-operator",
    )
    append_canonical_requeue_grant(canonical, grant)
    return dict(grant)


@pytest.mark.smoke
def test_operator_act_census_counts_window_acts_and_identities(tmp_path: Path) -> None:
    """Appends and voids census separately; identities disclose verbatim."""

    records_root = tmp_path / "records"
    _seed_ledgers(records_root)
    park = append_refusal_park(records_root, "m_censused")
    empty = operator_act_census(records_root, baselines={})
    assert empty["appends"] == 1 and empty["voids"] == 0
    void_refusal_park(records_root, park)
    census = operator_act_census(records_root, baselines={})
    assert census["rows"] == 2
    assert census["appends"] == 1
    assert census["voids"] == 1
    assert census["voided_row_sha256s"] == [park["row_sha256"]]
    assert census["identities"], "identity spellings disclose verbatim (D-53)"


@pytest.mark.smoke
def test_a_void_only_member_with_a_live_grant_refuses(tmp_path: Path) -> None:
    """The W42 refusing arm, member-scoped, with its sanctioned negatives."""

    records_root = tmp_path / "records"
    _seed_ledgers(records_root)
    park = append_refusal_park(records_root, "m_dodged")
    # Sanctioned drain: a void with NO live grant is the release path.
    void_refusal_park(records_root, park)
    assert void_only_with_live_grant_findings(records_root, baselines={}, as_of=AS_OF) == ()

    grant = _mint_grant(records_root, "m_dodged")
    findings = void_only_with_live_grant_findings(records_root, baselines={}, as_of=AS_OF)
    assert len(findings) == 1
    assert "m_dodged" in findings[0]
    assert grant["grant_id"] in findings[0]
    assert "GRANT" in findings[0]

    # Consuming the grant clears the arm: the authorized lane was used.
    ledgers = default_ledger_paths(records_root)
    operational = canonical_operational_ledger_path(ledgers.models)
    with JsonlLedger(operational, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        from menagerie.crawler.operational_identity import mint_operational_event

        ledger.append(
            mint_operational_event(
                {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
                    "created_at": AS_OF,
                    "status": OperationalEventStatus.HEALTHY.value,
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {},
                    "current_environment": None,
                    "run_id": "pytest",
                    "machine_id": "pytest",
                    "details": {"grant_id": grant["grant_id"]},
                },
                identity_payload={"grant_id": grant["grant_id"]},
            )
        )
    assert void_only_with_live_grant_findings(records_root, baselines={}, as_of=AS_OF) == ()


@pytest.mark.smoke
def test_the_close_discloses_the_census_and_live_grants(tmp_path: Path) -> None:
    """The production close carries both W42 disclosure blocks, additively."""

    from menagerie.crawler.close_object import close_object_path, load_close_objects

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    assert driver.run().status == "complete"
    records_root = tmp_path / "records"
    closes = load_close_objects([close_object_path(records_root)])
    assert len(closes) == 1
    currency = dict(closes[0]["currency"])
    census = currency["operator_act_census"]
    assert set(census) == {"rows", "appends", "voids", "voided_row_sha256s", "identities"}
    assert census["voids"] == 0
    assert currency["live_grants"] == []


@pytest.mark.smoke
def test_the_fixture_close_keeps_parity_with_the_driver_arm(tmp_path: Path) -> None:
    """record_fixture_close refuses the void-beside-live-grant shape too."""

    from menagerie.crawler.operator_provenance import OperatorProvenanceError
    from menagerie.crawler.tests.close_object_support import (
        open_fixture_leg,
        record_fixture_close,
    )

    records_root = tmp_path / "records"
    _seed_ledgers(records_root)
    # Open the leg FIRST so the void and the grant land inside its window
    # (record_fixture_close continues the same open generation, keeping the
    # original baselines as the window floor).
    open_fixture_leg(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-parity",
        intake_snapshot_sha256="sha256:intake-parity",
        phase=None,
        run_id="pytest-close",
    )
    park = append_refusal_park(records_root, "m_parity")
    void_refusal_park(records_root, park)
    _mint_grant(records_root, "m_parity")
    with pytest.raises(OperatorProvenanceError, match="live grants"):
        record_fixture_close(
            records_root,
            campaign_id="c1-mech",
            intake_snapshot_id="intake-parity",
            intake_snapshot_sha256="sha256:intake-parity",
            member_ids=("m_a", "m_b"),
            seed_member_heads=True,
        )
