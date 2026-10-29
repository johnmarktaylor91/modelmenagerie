"""SD-2: leg open refuses on outstanding licensing debt / stale rotated axes.

R9's refusal fired at rung END (identity axes rotated, heads stale, every
leg's spend already burned); R2's fired at successor STAGING (a close that
could not license, discovered after the rung). This arm converts both into a
LEG-START refusal through the existing authorities -- ``close_object_refusals``
(the stage_rung licensing half) and the W43 ``stale_head_census`` -- with a
durable explicit-ack arm scoped to the EXACT census digest. DX-24's read-only
probe alerts; this arm refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.leg_licensing_guard import (
    LegLicensingDebtError,
    acknowledge_licensing_debt,
    assert_leg_open_licensed,
    licensing_debt_census,
    read_licensing_debt_ack,
)
from menagerie.crawler.reducer import default_ledger_paths
from menagerie.crawler.rotation_registry import (
    append_rotation_registration,
    build_rotation_registration,
    rotation_registry_path,
)
from menagerie.crawler.tests.close_object_support import record_fixture_close


def _seed_ledger(records_root: Path) -> None:
    ledgers = default_ledger_paths(records_root)
    for name in ("models", "attempts", "gates", "artifacts"):
        path = getattr(ledgers, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"")


def _mint_debt_close(
    records_root: Path,
    *,
    intake_snapshot_id: str,
    member_ids: tuple[str, str] = ("m_one", "m_two"),
) -> None:
    """Seal a close whose accounting census LIES (the DX-22 machinery class).

    A member double-listed across two accounting arms: the census counts 3
    over an intake of 2. Deliberately NOT a stale/deferral/unresolved shape --
    those are lawful mid-rung states SD-2 must never refuse on.
    """

    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id=intake_snapshot_id,
        intake_snapshot_sha256=f"sha256:{intake_snapshot_id}",
        member_ids=member_ids,
        seed_member_heads=True,
        partition_overrides={
            "buckets": {"runs": list(member_ids), "failed": [member_ids[0]]},
        },
    )


@pytest.mark.smoke
def test_a_clean_root_opens_without_debt(tmp_path: Path) -> None:
    """No close, no registry: a fresh campaign owes nothing and opens."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    census = assert_leg_open_licensed(records_root, campaign_id="c1-mech")
    assert census["in_debt"] is False
    assert census["acknowledged"] is False


@pytest.mark.smoke
def test_leg_open_refuses_on_outstanding_licensing_debt(tmp_path: Path) -> None:
    """A campaign whose newest close cannot license refuses the NEXT leg open.

    The refusal derives through close_object_refusals -- the exact
    self-contained half of the stage_rung licensing gate; no second
    implementation -- and names the census digest an acknowledgment must cite.
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    _mint_debt_close(records_root, intake_snapshot_id="intake-debt")
    with pytest.raises(LegLicensingDebtError, match="licensing debt") as caught:
        assert_leg_open_licensed(records_root, campaign_id="c1-mech")
    census = caught.value.census
    assert census["in_debt"] is True
    assert census["licensing_findings"]
    assert census["debt_sha256"].startswith("sha256:")
    assert census["debt_sha256"] in str(caught.value)


@pytest.mark.smoke
def test_the_explicit_ack_arm_is_scoped_to_the_exact_census(tmp_path: Path) -> None:
    """An ack naming the exact digest opens the leg; ANY debt change re-arms.

    The digest is recomputed from evidence at every open, so an acknowledgment
    can never be minted in advance of debt it has not seen -- a standing
    waiver is structurally impossible.
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    _mint_debt_close(records_root, intake_snapshot_id="intake-debt")
    with pytest.raises(LegLicensingDebtError) as caught:
        assert_leg_open_licensed(records_root, campaign_id="c1-mech")
    debt_sha256 = caught.value.census["debt_sha256"]

    acknowledge_licensing_debt(
        records_root,
        debt_sha256=debt_sha256,
        reason="continuing deliberately; debt tracked in the rung docket",
        acked_at="2026-08-25T00:00:00Z",
        acked_by="pytest-operator",
    )
    assert read_licensing_debt_ack(records_root) is not None
    acknowledged = assert_leg_open_licensed(records_root, campaign_id="c1-mech")
    assert acknowledged["acknowledged"] is True

    # The debt CHANGES (a newer debt-bearing close): the old ack buys nothing.
    _mint_debt_close(
        records_root,
        intake_snapshot_id="intake-debt-2",
        member_ids=("m_three", "m_four"),
    )
    with pytest.raises(LegLicensingDebtError):
        assert_leg_open_licensed(records_root, campaign_id="c1-mech")


@pytest.mark.smoke
def test_a_blank_or_malformed_ack_reads_as_absent(tmp_path: Path) -> None:
    """Ambiguity fails closed: a corrupt ack file never opens a leg."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    _mint_debt_close(records_root, intake_snapshot_id="intake-debt")
    ack_path = records_root / "control" / "licensing-debt-ack.json"
    ack_path.parent.mkdir(parents=True, exist_ok=True)
    ack_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LegLicensingDebtError):
        assert_leg_open_licensed(records_root, campaign_id="c1-mech")
    with pytest.raises(ValueError, match="reason"):
        acknowledge_licensing_debt(
            records_root,
            debt_sha256="sha256:" + "0" * 64,
            reason="   ",
            acked_at="2026-08-25T00:00:00Z",
            acked_by="pytest-operator",
        )


@pytest.mark.smoke
def test_a_lawful_mid_rung_close_is_never_debt(tmp_path: Path) -> None:
    """Disclosed in-flight states open freely -- the zero-intervention target.

    A close attesting an unresolved verdict with a coverage hole (the shape
    every interrupted or deferral-carrying leg seals) is exactly what the
    NEXT leg exists to resolve. PD-1 adjudicated close-blocking arms BLOCKING;
    SD-2's debt is machinery lies and rotation staleness, never the funnel.
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-midrung",
        intake_snapshot_sha256="sha256:intake-midrung",
        member_ids=("m_one", "m_two"),
        seed_member_heads=True,
        partition_overrides={"missing": ["m_two"], "valid": False},
    )
    census = assert_leg_open_licensed(records_root, campaign_id="c1-mech")
    assert census["in_debt"] is False


def _register_live_award_closure(registry: Path) -> str:
    """Register the LIVE award-closure identity (the --print-live value)."""

    from menagerie.crawler.tools.register_authority_rotation import (
        live_authority_identities,
    )

    live = live_authority_identities()["award_closure_identity"]
    append_rotation_registration(
        registry,
        build_rotation_registration(
            axis="award_closure_identity",
            from_identity="sha256:" + "0" * 64,
            to_identity=live,
            rotation_class="refinement",
            rationale="pytest: live head",
            registered_by="pytest-operator",
            created_at="2026-08-25T00:00:00Z",
        ),
    )
    return live


@pytest.mark.smoke
def test_award_closure_is_adjudicated_in_the_register_tools_space(tmp_path: Path) -> None:
    """Item 11 (D-17): one authority for ``award_closure_identity``.

    The pilot's exact shape: the registry head equals the value
    ``register_authority_rotation --print-live`` prints, while the natively
    hashed source-tree closure is a DIFFERENT digest. The census used to
    compare the head against the native value and read the axis stale
    forever (five acks "on precedent"). It now derives the axis through the
    register tool's function and reads FRESH.
    """

    from menagerie.crawler.tools.rotation_diff import closure_identity_at

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    registry = rotation_registry_path(records_root)
    registry.parent.mkdir(parents=True, exist_ok=True)
    live = _register_live_award_closure(registry)
    repo_root = Path(__file__).resolve().parents[3]
    native = closure_identity_at(repo_root, None)
    assert native != live, "the two derivations agree here; the pilot shape is not reproduced"
    census = assert_leg_open_licensed(records_root, campaign_id="c1-mech", repo_root=repo_root)
    assert census["in_debt"] is False
    row = next(
        axis for axis in census["axis_census"]["axes"] if axis["axis"] == "award_closure_identity"
    )
    assert row["status"] == "fresh"
    assert row["current_identity"] == live


@pytest.mark.smoke
def test_debt_digest_binds_each_stale_axis_as_registered_head_and_live_pair(
    tmp_path: Path,
) -> None:
    """Item 11: a REAL move on an already-stale axis is a new debt.

    The digest used to bind ``stale_axes`` as a list, so once an axis was
    stale every further rotation on it hid inside the same digest and
    surfaced two doors later. The digest now binds (registered head, live
    identity) per axis.
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    registry = rotation_registry_path(records_root)
    registry.parent.mkdir(parents=True, exist_ok=True)
    _register_live_award_closure(registry)
    append_rotation_registration(
        registry,
        build_rotation_registration(
            axis="runner_identity",
            from_identity="sha256:" + "1" * 64,
            to_identity="sha256:" + "2" * 64,
            rotation_class="refinement",
            rationale="pytest rotation",
            registered_by="pytest-operator",
            created_at="2026-08-25T00:00:00Z",
        ),
    )
    first = licensing_debt_census(
        records_root,
        campaign_id="c1-mech",
        current_identities={"runner_identity": "sha256:" + "3" * 64},
    )
    second = licensing_debt_census(
        records_root,
        campaign_id="c1-mech",
        current_identities={"runner_identity": "sha256:" + "4" * 64},
    )
    assert first["stale_axes"] == second["stale_axes"] == ["runner_identity"]
    assert first["debt_sha256"] != second["debt_sha256"]
    assert first["stale_axis_heads"]["runner_identity"] == {
        "newest_registered_head": "sha256:" + "2" * 64,
        "live_identity": "sha256:" + "3" * 64,
        "status": "stale-head-drifted",
    }
    # An ack of the first debt covers exactly it, never the moved axis.
    acknowledge_licensing_debt(
        records_root,
        debt_sha256=first["debt_sha256"],
        reason="first debt read",
        acked_at="2026-09-01T00:00:00Z",
        acked_by="pytest-operator",
    )
    assert assert_leg_open_licensed(
        records_root,
        campaign_id="c1-mech",
        current_identities={"runner_identity": "sha256:" + "3" * 64},
    )["acknowledged"]
    with pytest.raises(LegLicensingDebtError, match="stale rotated axis"):
        assert_leg_open_licensed(
            records_root,
            campaign_id="c1-mech",
            current_identities={"runner_identity": "sha256:" + "4" * 64},
        )


@pytest.mark.smoke
def test_leg_open_refuses_on_a_stale_rotated_axis(tmp_path: Path) -> None:
    """R9 at leg START: a registered axis drifted from the caller's identity.

    The adjudication is the W43 census -- one implementation. An axis the
    caller cannot compute reads not-computed (disclosed, never refused); a
    computed axis that DRIFTED from its newest registered head refuses.
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    registry = rotation_registry_path(records_root)
    registry.parent.mkdir(parents=True, exist_ok=True)
    append_rotation_registration(
        registry,
        build_rotation_registration(
            axis="runner_identity",
            from_identity="sha256:" + "1" * 64,
            to_identity="sha256:" + "2" * 64,
            rotation_class="refinement",
            rationale="pytest rotation",
            registered_by="pytest-operator",
            created_at="2026-08-25T00:00:00Z",
        ),
    )
    # Item 11: the award-closure axis is always derived (live identity), so a
    # registry that has rows must carry its live head to read fresh.
    _register_live_award_closure(registry)
    fresh = assert_leg_open_licensed(
        records_root,
        campaign_id="c1-mech",
        current_identities={"runner_identity": "sha256:" + "2" * 64},
    )
    assert fresh["in_debt"] is False
    with pytest.raises(LegLicensingDebtError, match="stale rotated axis") as caught:
        assert_leg_open_licensed(
            records_root,
            campaign_id="c1-mech",
            current_identities={"runner_identity": "sha256:" + "3" * 64},
        )
    assert "runner_identity" in caught.value.census["stale_axes"]


@pytest.mark.smoke
def test_the_driver_arms_the_refusal_before_the_generation_opens() -> None:
    """Wiring pin: the SD-2 arm fires BEFORE open_leg_generation, typed.

    The refusal must be a LegStartRefused carrying the closed
    licensing_debt_unacked StopClass, raised before any generation opens or
    admission spends -- the whole point is refusing at the cheapest moment.
    """

    from menagerie.crawler.leg_exit_vocabulary import (
        STOP_CLASS_EVALUATOR_COVERAGE,
        StopClass,
    )

    source = (
        Path(__file__).resolve().parents[1] / "driver.py"
    ).read_text(encoding="utf-8")
    guard_at = source.index("assert_leg_open_licensed(")
    open_at = source.index("self._leg_generation = open_leg_generation(")
    assert guard_at < open_at, "the SD-2 arm must fire before the generation opens"
    refusal_at = source.index("StopClass.LICENSING_DEBT_UNACKED.value", guard_at)
    assert refusal_at < open_at, "the refusal must be typed at the leg-start seam"
    assert StopClass.LICENSING_DEBT_UNACKED.value == "licensing_debt_unacked"
    assert STOP_CLASS_EVALUATOR_COVERAGE[StopClass.LICENSING_DEBT_UNACKED.value] is True


@pytest.mark.smoke
def test_the_operator_tool_round_trips_status_and_ack(tmp_path: Path, capsys) -> None:
    """The ack arm has a real operator door: status exits 2 in debt, 0 clean."""

    from menagerie.crawler.tools.licensing_debt import main as tool_main

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    assert (
        tool_main(
            ["status", "--records-root", str(records_root), "--campaign-id", "c1-mech"]
        )
        == 0
    )
    _mint_debt_close(records_root, intake_snapshot_id="intake-debt")
    assert (
        tool_main(
            ["status", "--records-root", str(records_root), "--campaign-id", "c1-mech"]
        )
        == 2
    )
    census = licensing_debt_census(records_root, campaign_id="c1-mech")
    capsys.readouterr()
    assert (
        tool_main(
            [
                "ack",
                "--records-root",
                str(records_root),
                "--debt-sha256",
                census["debt_sha256"],
                "--reason",
                "pytest deliberate continuation",
            ]
        )
        == 0
    )
    assert assert_leg_open_licensed(records_root, campaign_id="c1-mech")[
        "acknowledged"
    ] is True


def test_a_pre_ledger_gating_ack_is_disclosed_never_a_bare_zero(tmp_path: Path) -> None:
    """P5 round-2 R2-5/sol-11: the census never reads zero beside a gating ack.

    The live shape: the re-ack was performed BEFORE the append-first writer
    landed, so the gating sentinel names a digest the history ledger does not
    hold and the ack_history arm reported ``count: 0`` beside a PASSING gate —
    the opposite of the disclosure it was built for. The census now discloses
    the pre-ledger gating ack, its digest, and the exact backfill remedy; the
    idempotent re-ack through the writer clears the disclosure.
    """

    from menagerie.crawler.leg_licensing_guard import (
        licensing_debt_ack_path,
        licensing_debt_census,
    )
    from menagerie.crawler.identity import canonical_json_bytes

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    _mint_debt_close(records_root, intake_snapshot_id="intake-debt")
    with pytest.raises(LegLicensingDebtError) as caught:
        assert_leg_open_licensed(records_root, campaign_id="c1-mech")
    debt_sha256 = caught.value.census["debt_sha256"]

    # Reproduce the pre-ledger world: the sentinel exists (written directly,
    # as the pre-writer ops act did), the history ledger does not.
    sentinel_path = licensing_debt_ack_path(records_root)
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "menagerie.crawler.licensing-debt-ack.v1",
                "debt_sha256": debt_sha256,
                "reason": "pre-ledger operator ack",
                "acked_at": "2026-08-30T00:00:00Z",
                "acked_by": "operator-preledger",
            }
        )
        + b"\n"
    )
    # The gate passes on the sentinel...
    passed = assert_leg_open_licensed(records_root, campaign_id="c1-mech")
    assert passed["acknowledged"] is True
    # ...and the census DISCLOSES the pre-ledger ack instead of a bare zero.
    history = passed["ack_history"]
    assert history["count"] == 0
    assert history["history_incomplete"] is True
    assert history["pre_ledger_gating_ack"]["debt_sha256"] == debt_sha256
    assert "licensing_debt ack" in history["pre_ledger_gating_ack"]["remedy"]

    # The remedy: one idempotent re-ack of the SAME digest through the
    # append-first writer backfills the canonical row and clears the arm.
    acknowledge_licensing_debt(
        records_root,
        debt_sha256=debt_sha256,
        reason="canonical-history backfill of the pre-ledger ack",
        acked_at="2026-08-31T00:00:00Z",
        acked_by="operator-test",
    )
    census = licensing_debt_census(records_root, campaign_id="c1-mech")
    assert census["ack_history"]["count"] == 1
    assert census["ack_history"]["acknowledged_debt_sha256s"] == [debt_sha256]
    assert "history_incomplete" not in census["ack_history"]
    assert "pre_ledger_gating_ack" not in census["ack_history"]
    # The disclosure never moved the debt it acknowledges: same digest, still
    # acknowledged, gate still passing.
    assert census["debt_sha256"] == debt_sha256
    assert assert_leg_open_licensed(records_root, campaign_id="c1-mech")["acknowledged"]


def test_every_ack_lands_in_the_append_only_history(tmp_path: Path) -> None:
    """P5 fable-F5/F19, sol-11: the overwritten sentinel destroyed ack history.

    RED at bd5c7837b: four production acknowledgments left exactly ONE
    surviving record (the newest sentinel); the earlier three existed only as
    prose. Every ack now also appends a digest-stamped row to the append-only
    ledger BEFORE the sentinel moves, and the reader refuses tampered rows.
    """

    from menagerie.crawler.leg_licensing_guard import (
        licensing_debt_ack_ledger_path,
        load_licensing_debt_acks,
        read_licensing_debt_ack,
    )

    records_root = tmp_path
    first = acknowledge_licensing_debt(
        records_root,
        debt_sha256="sha256:" + "1" * 64,
        reason="first ack",
        acked_at="2026-08-30T00:00:00Z",
        acked_by="operator-test",
    )
    second = acknowledge_licensing_debt(
        records_root,
        debt_sha256="sha256:" + "2" * 64,
        reason="second ack",
        acked_at="2026-08-31T00:00:00Z",
        acked_by="operator-test",
    )
    # The sentinel gates on the NEWEST ack only, exactly as before.
    sentinel = read_licensing_debt_ack(records_root)
    assert sentinel is not None and sentinel["debt_sha256"] == second["debt_sha256"]
    # The history carries BOTH, in order, digest-stamped.
    history = load_licensing_debt_acks(records_root)
    assert [row["debt_sha256"] for row in history] == [
        first["debt_sha256"],
        second["debt_sha256"],
    ]
    assert all(row["ack_sha256"].startswith("sha256:") for row in history)
    # A tampered history row refuses the whole read (audit evidence is
    # never silently dropped).
    ledger = licensing_debt_ack_ledger_path(records_root)
    tampered = ledger.read_text(encoding="utf-8").replace("first ack", "forged ack")
    ledger.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="digest stamp"):
        load_licensing_debt_acks(records_root)
