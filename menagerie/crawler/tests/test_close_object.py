"""BD-1: a rung boundary is crossed on an immutable close object, or not at all.

T1-sol-02 (CONFIRMED by all three S2 labs): when the operator halted step-100
the newest driver JSON was ``{"status": "interrupted:shutdown",
"models_reduced": 12, "terminal_models": 29}`` -- an INTERRUPTION record -- while
the same books answered the boundary gate with exit 1 and 24 "missing" members.
Nothing machine-readable said whether the rung had closed, so audit dispatch,
restoration, and step-200 staging were all planned from hand-tallied raw heads.

The close object is that missing statement: content-addressed, append-only,
naming the intake identity, the leg generation it seals, the full source
census it observed, and -- the v2 licensing authority -- the sealed
``PartitionProofV2`` it binds by digest. Staging the next rung consumes it,
re-observes the census, and recomputes the proof from the pinned bytes.

It is EVIDENCE, never authority. These pins hold the line in both directions:
a close cannot exist for a non-close status, a close whose disclosure lies
about the membership refuses, and an unclean attested census (a coverage hole,
a stale head) refuses to license exactly as a missing close does. The
window/citation doors themselves are pinned in ``test_close_v2_contract.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.close_object import (
    CLOSE_OBJECT_SCHEMA_VERSION,
    CLOSE_OBJECT_SCHEMA_VERSION_V1,
    CloseObjectError,
    build_close_object,
    close_disclosure_refusals,
    close_object_disagreements,
    close_object_path,
    close_object_refusals,
    find_close_object,
    is_close_status,
    ledger_identities,
    load_close_objects,
    require_closed_lineage,
    verifier_identity,
)
from menagerie.crawler.driver import DriverPaths
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.leg_generation import census_identities
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import default_ledger_paths
from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign
from menagerie.crawler.tests.close_object_support import (
    append_award_downgrade,
    open_fixture_leg,
    record_fixture_close,
    seed_award_head,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot


def _demand(
    records_root: Path,
    *,
    intake_snapshot_id: str,
    campaign_id: str = "c1-mech",
    intake_snapshot_sha256: str = "sha256:aaaa",
    members: tuple[str, ...] = ("m1",),
    label: str = "next-rung",
):
    """Demand a licensing close with the replay-derived fact set.

    The gate binds the campaign, the registered snapshot digest, and the
    lineage's REAL membership; everything else -- the census, the leg window,
    the per-member citations -- it re-derives from the records root itself.
    """

    return require_closed_lineage(
        records_root,
        intake_snapshot_id=intake_snapshot_id,
        campaign_id=campaign_id,
        intake_snapshot_sha256=intake_snapshot_sha256,
        lineage_member_ids=frozenset(members),
        introduced_member_ids=frozenset(members),
        label=label,
    )


def _seed_ledger(records_root: Path) -> LedgerPaths:
    """Materialize empty canonical ledgers so the fixture leg can baseline them."""

    ledgers = default_ledger_paths(records_root)
    for name in ("models", "attempts", "gates", "artifacts"):
        path = getattr(ledgers, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"")
    return ledgers


@pytest.mark.smoke
def test_only_the_close_family_is_a_close() -> None:
    """An interruption, a pause, or a failure never attests a boundary."""

    assert is_close_status("complete")
    assert is_close_status("complete-with-deferrals")
    assert is_close_status("terminal-partition-complete-with-review-holds")
    assert is_close_status("phase-complete-with-deferrals:pytorch")
    assert not is_close_status("interrupted:shutdown")
    assert not is_close_status("paused:review-checkpoint")
    assert not is_close_status("failed")


@pytest.mark.smoke
def test_an_interruption_can_never_become_a_close_object(tmp_path: Path) -> None:
    """The halt-state record the step-100 audit was handed refuses at build."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    with pytest.raises(CloseObjectError, match="not a close"):
        build_close_object(
            campaign_id="c1-mech",
            intake_snapshot_id="intake-aaaa",
            intake_snapshot_sha256="sha256:aaaa",
            phase=None,
            close_kind="leg",
            generation_id="sha256:" + "3" * 64,
            leg_open_row_sha256="sha256:" + "4" * 64,
            sources=census_identities(records_root),
            partition_proof_sha256="sha256:" + "5" * 64,
            predecessor_close_sha256=None,
            run_id="run-1",
            machine_id="host-1",
            closed_at="2026-08-14T19:00:00Z",
            completion_status="interrupted:shutdown",
            partition_arms={"missing": []},
            currency_census={
                "stale_by_axis": {},
                "award_runs_to_non_runs": [],
                "award_runs_to_runs": [],
            },
            authority_identities={"reducer_policy_identity": "sha256:aaa"},
        )


@pytest.mark.smoke
def test_identical_close_rows_are_append_idempotent_and_recloses_are_new_epochs(
    tmp_path: Path,
) -> None:
    """The content address dedupes; a RE-close is a new generation, disclosed.

    ``append_close_object`` returns the existing row for identical bytes. But
    re-closing a rung whose generation already closed is a NEW adjudication
    epoch under the v2 contract ("newest row wins" is not an authority rule;
    a re-close needs its own generation), so the fixture's second close mints
    a second attestation carrying a different generation -- and its members
    are honestly CARRIED from the first close, not re-claimed as run outcomes.
    """

    from menagerie.crawler.close_object import append_close_object

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    first = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-aaaa",
        intake_snapshot_sha256="sha256:aaaa",
        member_ids=("m1", "m2"),
        seed_member_heads=True,
    )
    assert first["schema_version"] == CLOSE_OBJECT_SCHEMA_VERSION
    assert append_close_object(close_object_path(records_root), first) == first
    assert len(load_close_objects([close_object_path(records_root)])) == 1

    reclose = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-aaaa",
        intake_snapshot_sha256="sha256:aaaa",
        member_ids=("m1", "m2"),
    )
    assert reclose["generation_id"] != first["generation_id"]
    assert reclose["partition"]["carried_count"] == 2
    assert reclose["partition"]["adjudicated_here"] == []
    from menagerie.crawler.partition_proof import load_partition_proof, proved_here

    proof = load_partition_proof(records_root, reclose["partition_proof_sha256"])
    assert proved_here(proof) == frozenset({"m1", "m2"})
    assert len(load_close_objects([close_object_path(records_root)])) == 2


@pytest.mark.smoke
def test_a_tampered_close_object_refuses_to_load(tmp_path: Path) -> None:
    """The content address is the authenticity, so a rewritten arm cannot pass."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-aaaa",
        intake_snapshot_sha256="sha256:aaaa",
        member_ids=("m1",),
        seed_member_heads=True,
        partition_overrides={"missing": ["m1"], "valid": False},
    )
    path = close_object_path(records_root)
    row = json.loads(path.read_text(encoding="utf-8").strip())
    row["partition"]["missing"] = []
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(CloseObjectError, match="digest does not match"):
        load_close_objects([path])


@pytest.mark.smoke
def test_an_attested_hole_or_stale_head_licenses_nothing(tmp_path: Path) -> None:
    """A close that recorded the problem does not become permission."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    holed = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-holed",
        intake_snapshot_sha256="sha256:holed",
        member_ids=("m1",),
        seed_member_heads=True,
        partition_overrides={"missing": ["m1"], "valid": False},
    )
    stale = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-stale",
        intake_snapshot_sha256="sha256:stale",
        member_ids=("m2",),
        seed_member_heads=True,
        currency_overrides={
            "stale": 1,
            "stale_by_axis": {"publication_policy_identity": ["m2"]},
            "awards_stale_on_live_basis": ["m2"],
        },
    )
    assert any("missing" in reason for reason in close_object_refusals(holed))
    assert any("could not vouch for" in reason for reason in close_object_refusals(stale))


@pytest.mark.smoke
def test_a_close_over_moved_sources_no_longer_describes_the_state(
    tmp_path: Path,
) -> None:
    """Staging from a state the close never observed must re-attest it."""

    records_root = tmp_path / "records"
    ledgers = _seed_ledger(records_root)
    close = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-aaaa",
        intake_snapshot_sha256="sha256:aaaa",
        member_ids=("m1",),
        seed_member_heads=True,
    )
    assert close_object_disagreements(close, records_root) == ()
    with ledgers.attempts.open("ab") as handle:
        handle.write(b'{"stable_id": "m1", "work_id": "w-late"}\n')
    disagreements = close_object_disagreements(close, records_root)
    assert len(disagreements) == 1
    assert "moved since the close" in disagreements[0]
    assert "attempts" in disagreements[0]


@pytest.mark.smoke
def test_require_closed_lineage_refuses_when_no_close_exists(tmp_path: Path) -> None:
    """The gate names the missing attestation and where it belongs."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    with pytest.raises(CloseObjectError, match="no immutable close object"):
        _demand(records_root, intake_snapshot_id="intake-never-closed", label="step-200")
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-never-closed",
        intake_snapshot_sha256="sha256:aaaa",
        member_ids=("m1",),
        seed_member_heads=True,
    )
    close = _demand(records_root, intake_snapshot_id="intake-never-closed", label="step-200")
    assert close["intake_snapshot_id"] == "intake-never-closed"
    assert (
        find_close_object(
            load_close_objects([close_object_path(records_root)]),
            intake_snapshot_id="intake-never-closed",
        )
        == close
    )


@pytest.mark.smoke
def test_the_disposable_state_discloses_the_close_without_its_address(
    tmp_path: Path,
) -> None:
    """The state file stays CONVERGENT; the content address has one home.

    ``driver-state.json`` is compared byte-for-byte against the serial no-kill
    oracle by the uncatchable-kill durability matrix. A close object's address
    binds the ledger BYTES it observed, so a killed-and-resumed campaign mints a
    different address from an uninterrupted campaign over the same models -- its
    artifact/operational history carries the disclosed kill evidence. Putting
    that address in the cursor asserts a divergence that does not exist, so the
    cursor carries only convergent facts (what closed, under which authority)
    and the address lives in the append-only close ledger.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=1)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    records_root = tmp_path / "records"
    paths = DriverPaths(
        tmp_path / "runtime",
        snapshot.root,
        LedgerPaths(
            records_root / "models" / "current-shard.jsonl",
            records_root / "attempts" / "local.jsonl",
            records_root / "gates" / "current-shard.jsonl",
        ),
    )
    assert _driver(tmp_path, snapshot, paths_override=paths).run().status == "complete"

    state = json.loads(paths.driver_state.read_text(encoding="utf-8"))
    close = load_close_objects([close_object_path(records_root)])[0]
    assert state["close_object"] == {
        "completion_status": close["completion_status"],
        "verifier_identity": close["verifier_identity"],
    }
    assert "close_sha256" not in state["close_object"]
    # The address is resolvable from the ledger, which is its authority anyway.
    assert (
        find_close_object(
            load_close_objects([close_object_path(records_root)]),
            intake_snapshot_id=snapshot.snapshot_id,
        )["close_sha256"]
        == close["close_sha256"]
    )


# ---------------------------------------------------------------------------
# Stage-gate hardening (2026-08-16 red-team break 5): a close licenses a rung
# only if it attests a RESOLVED partition. A vacuous attestation -- empty arms,
# empty coverage, an unaccounted member, or a recorded invalid verdict -- is
# structurally refusable, never silently licensing.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_a_vacuous_close_cannot_be_built_or_validated(tmp_path: Path) -> None:
    """Empty partition/currency arms refuse at build AND at load.

    Red-team break 5 (opus #4): ``validate_close_object`` required the arms to
    BE mappings but never to carry anything, so a hand-minted close with
    ``partition_arms={}, currency_census={}`` had no arm to trip and
    ``require_closed_lineage`` returned it as licensing. The persisted-bytes
    variant is minted in the v1 shape on purpose: readable legacy rows keep
    every arm check too.
    """

    from menagerie.crawler.close_object import validate_close_object

    records_root = tmp_path / "records"
    ledgers = _seed_ledger(records_root)
    with pytest.raises(CloseObjectError, match="partition"):
        build_close_object(
            campaign_id="c1-mech",
            intake_snapshot_id="intake-vacuous",
            intake_snapshot_sha256="sha256:vacuous",
            phase=None,
            close_kind="leg",
            generation_id="sha256:" + "3" * 64,
            leg_open_row_sha256="sha256:" + "4" * 64,
            sources=census_identities(records_root),
            partition_proof_sha256="sha256:" + "5" * 64,
            predecessor_close_sha256=None,
            run_id="run-1",
            machine_id="host-1",
            closed_at="2026-08-16T00:00:00Z",
            completion_status="complete",
            partition_arms={},
            currency_census={},
            authority_identities={"reducer_policy_identity": "sha256:aaa"},
        )

    # The same emptiness injected as persisted bytes refuses at validation, so
    # no reader can consume a vacuous close that bypassed the builder.
    identities = {"reducer_policy_identity": "sha256:aaa"}
    body = {
        "schema_version": CLOSE_OBJECT_SCHEMA_VERSION_V1,
        "campaign_id": "c1-mech",
        "intake_snapshot_id": "intake-vacuous",
        "intake_snapshot_sha256": "sha256:vacuous",
        "phase": None,
        "run_id": "run-1",
        "machine_id": "host-1",
        "closed_at": "2026-08-16T00:00:00Z",
        "completion_status": "complete",
        "ledgers": ledger_identities(ledgers),
        "partition": {},
        "currency": {},
        "verifier_identity": verifier_identity(identities),
        "authority_identities": identities,
    }
    row = dict(body)
    row["close_sha256"] = stable_hash(body)
    with pytest.raises(CloseObjectError, match="partition"):
        validate_close_object(row)


@pytest.mark.smoke
def test_a_close_recording_an_invalid_partition_licenses_nothing(tmp_path: Path) -> None:
    """The recorded ``valid: false`` verdict alone refuses licensing."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    unresolved = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-unresolved",
        intake_snapshot_sha256="sha256:unresolved",
        member_ids=("m1",),
        seed_member_heads=True,
        partition_overrides={"valid": False},
    )
    refusals = close_object_refusals(unresolved)
    assert refusals, "a close recording an INVALID partition licensed a next rung"
    assert any("valid" in reason for reason in refusals)


@pytest.mark.smoke
def test_a_close_with_empty_coverage_licenses_nothing(tmp_path: Path) -> None:
    """Non-empty coverage: a close over zero members attests nothing."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    empty = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-empty",
        intake_snapshot_sha256="sha256:empty",
        member_ids=(),
    )
    refusals = close_object_refusals(empty)
    assert refusals, "a close over an EMPTY partition licensed a next rung"
    with pytest.raises(CloseObjectError):
        _demand(
            records_root,
            intake_snapshot_id="intake-empty",
            intake_snapshot_sha256="sha256:empty",
            members=(),
        )


@pytest.mark.smoke
def test_a_close_with_an_unaccounted_member_licenses_nothing(tmp_path: Path) -> None:
    """Every intake member must land in exactly one recorded accounting arm."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-hole",
        intake_snapshot_sha256="sha256:hole",
        member_ids=("m1", "m2"),
        seed_member_heads=True,
        partition_overrides={"buckets": {"runs": ["m1"]}, "intake": 1},
    )
    with pytest.raises(CloseObjectError, match="account"):
        _demand(
            records_root,
            intake_snapshot_id="intake-hole",
            intake_snapshot_sha256="sha256:hole",
            members=("m1", "m2"),
        )


@pytest.mark.smoke
def test_a_driver_close_persists_its_own_attestation(
    tmp_path: Path,
) -> None:
    """The producer is the driver's close, and it binds the observed bytes."""

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    records_root = tmp_path / "records"
    paths = DriverPaths(
        tmp_path / "runtime",
        snapshot.root,
        LedgerPaths(
            records_root / "models" / "current-shard.jsonl",
            records_root / "attempts" / "local.jsonl",
            records_root / "gates" / "current-shard.jsonl",
        ),
    )
    result = _driver(tmp_path, snapshot, paths_override=paths).run()
    assert result.status == "complete"

    closes = load_close_objects([close_object_path(records_root)])
    assert len(closes) == 1
    close = closes[0]
    assert close["schema_version"] == CLOSE_OBJECT_SCHEMA_VERSION
    assert close["close_kind"] == "leg"
    assert close["intake_snapshot_id"] == snapshot.snapshot_id
    assert close["intake_snapshot_sha256"] == snapshot.snapshot_sha256
    assert close["completion_status"] == "complete"
    assert close["partition"]["valid"] is True
    assert close["partition"]["missing"] == []
    assert close["partition"]["buckets"]["runs"] == sorted(
        item.stable_id for item in snapshot.items
    )
    assert close["partition"]["adjudicated_here"] == sorted(
        item.stable_id for item in snapshot.items
    )
    assert close["partition"]["carried_count"] == 0
    assert close["currency"]["ledger_heads"] == 2
    assert close["currency"]["stale_by_axis"] == {}
    assert close["currency"]["awards_stale_on_live_basis"] == []
    # The generation machinery is durable evidence, not a producer claim: the
    # close binds a real leg-open row and a resident sealed proof.
    from menagerie.crawler.leg_generation import find_leg_open_row
    from menagerie.crawler.partition_proof import load_partition_proof

    open_row = find_leg_open_row(records_root, close["leg_open_row_sha256"])
    assert open_row is not None
    assert open_row["generation_id"] == close["generation_id"]
    proof = load_partition_proof(records_root, close["partition_proof_sha256"])
    assert set(proof["members"]) == {item.stable_id for item in snapshot.items}
    assert all(entry["disposition"] == "adjudicated" for entry in proof["members"].values())
    # The attestation describes the bytes it observed, so a re-observation of
    # the untouched census agrees, and the close licenses a next rung.
    assert close_object_disagreements(close, records_root) == ()
    assert close_object_refusals(close) == ()
    members = tuple(item.stable_id for item in snapshot.items)
    assert (
        _demand(
            records_root,
            intake_snapshot_id=snapshot.snapshot_id,
            intake_snapshot_sha256=snapshot.snapshot_sha256,
            members=members,
        )["close_sha256"]
        == close["close_sha256"]
    )

    lie_root = tmp_path / "award-census-lie"
    _seed_ledger(lie_root)
    record_fixture_close(
        lie_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-award-lie",
        intake_snapshot_sha256="sha256:award-lie",
        member_ids=("m-award",),
        seed_member_heads=True,
        currency_overrides={
            "award_runs_to_runs": [
                {
                    "stable_id": "m-award",
                    "predecessor_revision": "sha256:forged",
                    "successor_revision": "sha256:forged",
                }
            ]
        },
    )
    with pytest.raises(CloseObjectError, match="award_runs_to_runs"):
        _demand(
            lie_root,
            intake_snapshot_id="intake-award-lie",
            intake_snapshot_sha256="sha256:award-lie",
            members=("m-award",),
        )


def test_a_close_cannot_hide_a_real_award_downgrade(tmp_path: Path) -> None:
    """The census arms catch the close that HIDES a transition, not just the lie.

    Every close-object fixture head is a parentless ``failed`` row, so the
    award-transition census had nothing to find anywhere in this corpus and
    both arms were structurally ``[]`` -- the recomputation gate could only
    ever fire on a close that INVENTED a transition (the forged-arm case
    above), while the ``runs -> non-runs`` downgrade the AW-1 census exists to
    disclose was untestable. This seeds a REAL downgrade: an award below the
    generation floor, superseded in-window by a ``failed:accuracy-gate``
    successor naming it as parent.
    """

    honest_root = tmp_path / "award-downgrade"
    _seed_ledger(honest_root)
    award = seed_award_head(
        honest_root, stable_id="m-award", intake_snapshot_id="intake-downgrade"
    )
    open_fixture_leg(
        honest_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-downgrade",
        intake_snapshot_sha256="sha256:downgrade",
    )
    successor = append_award_downgrade(
        honest_root,
        stable_id="m-award",
        parent_revision=award,
        intake_snapshot_id="intake-downgrade",
    )
    honest = record_fixture_close(
        honest_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-downgrade",
        intake_snapshot_sha256="sha256:downgrade",
        member_ids=("m-award",),
    )
    # The census SEES the transition, so the honest close discloses it and the
    # gate agrees. This is the assertion that goes vacuous when the fixture
    # stops recording a runs kind or a parent revision.
    disclosed = honest["currency"]["award_runs_to_non_runs"]
    assert [row["stable_id"] for row in disclosed] == ["m-award"]
    assert [row["predecessor_revision"] for row in disclosed] == [award]
    assert [row["successor_revision"] for row in disclosed] == [successor]
    assert honest["currency"]["award_runs_to_runs"] == []
    assert close_object_disagreements(honest, honest_root) == ()
    assert (
        _demand(
            honest_root,
            intake_snapshot_id="intake-downgrade",
            intake_snapshot_sha256="sha256:downgrade",
            members=("m-award",),
        )["close_sha256"]
        == honest["close_sha256"]
    )

    # ...and a close that discloses an EMPTY arm over the same real downgrade
    # is refused, which is the direction no fixture could previously express.
    hidden_root = tmp_path / "award-downgrade-hidden"
    _seed_ledger(hidden_root)
    hidden_award = seed_award_head(
        hidden_root, stable_id="m-award", intake_snapshot_id="intake-hidden"
    )
    open_fixture_leg(
        hidden_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-hidden",
        intake_snapshot_sha256="sha256:hidden",
    )
    append_award_downgrade(
        hidden_root,
        stable_id="m-award",
        parent_revision=hidden_award,
        intake_snapshot_id="intake-hidden",
    )
    hidden = record_fixture_close(
        hidden_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-hidden",
        intake_snapshot_sha256="sha256:hidden",
        member_ids=("m-award",),
        currency_overrides={"award_runs_to_non_runs": []},
    )
    # The hidden arm is a DISCLOSURE lie, so it is the disclosure gate that
    # recomputes it (the attestation gate only re-observes the ledger bytes,
    # which the close still describes correctly).
    assert close_object_disagreements(hidden, hidden_root) == ()
    findings = close_disclosure_refusals(
        hidden,
        records_root=hidden_root,
        lineage_member_ids=frozenset({"m-award"}),
        introduced_member_ids=frozenset({"m-award"}),
    )
    assert any("award_runs_to_non_runs" in finding for finding in findings), findings
    with pytest.raises(CloseObjectError, match="award_runs_to_non_runs"):
        _demand(
            hidden_root,
            intake_snapshot_id="intake-hidden",
            intake_snapshot_sha256="sha256:hidden",
            members=("m-award",),
        )


# ---------------------------------------------------------------------------
# Close-object hardening (2026-08-16 red-team ROUND 2, breaks 2 and 3): round 1
# killed the STRUCTURALLY vacuous close; these pin the SUBSTANTIVELY vacuous
# one. Under v2 the builder itself derives the proof from the bytes, so an
# unrun leg is UNBUILDABLE here; the forged variants (a structurally valid
# close whose citations lie) are pinned in test_close_v2_contract.py.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_a_close_over_a_leg_that_recorded_nothing_licenses_nothing(
    tmp_path: Path,
) -> None:
    """A 0-head leg cannot even MINT a close (round-2 break 2a, v2 posture).

    Round 2's escape minted arms claiming two members as ``runs`` over a
    models ledger holding no head for either. The v2 builder derives the
    sealed proof from the pinned bytes, so the same lie now refuses at BUILD:
    a member with no in-window head, no live in-window withholding row, and no
    ancestor resolution is unresolvable, named. (The gate-side refusal of a
    FORGED such close is test_close_v2_contract's zero-byte-rung pin.)
    """

    from menagerie.crawler.partition_proof import PartitionProofError

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    with pytest.raises(PartitionProofError, match="m_alpha"):
        record_fixture_close(
            records_root,
            campaign_id="c1-mech",
            intake_snapshot_id="intake-neverran",
            intake_snapshot_sha256="sha256:neverran",
            member_ids=("m_alpha", "m_beta"),
        )
    # With every member recorded in-window the same close builds and licenses:
    # the derivation refuses the unresolved leg, never the resolved one.
    licensed = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-neverran",
        intake_snapshot_sha256="sha256:neverran",
        member_ids=("m_alpha", "m_beta"),
        seed_member_heads=True,
    )
    assert licensed["partition"]["adjudicated_here"] == ["m_alpha", "m_beta"]
    assert (
        _demand(
            records_root,
            intake_snapshot_id="intake-neverran",
            intake_snapshot_sha256="sha256:neverran",
            members=("m_alpha", "m_beta"),
        )["close_sha256"]
        == licensed["close_sha256"]
    )


@pytest.mark.smoke
def test_close_accounting_binds_membership_not_cardinality(tmp_path: Path) -> None:
    """Any two strings closed a two-member rung (round-2 break 2b).

    The accounting arm was ``len(accounted) != intake`` over a set union of the
    bucket ids, so it never asked whether the accounted ids were the intake's
    MEMBERS. Two ghosts closed a two-member rung, and a member double-listed in
    two arms plus one ghost passed the same way. Under v2 the licensing
    authority is the sealed proof -- but the DISCLOSURE arms must still match
    the membership exactly, or the close refuses (a lying census over a valid
    proof is a producer bug, not an operator problem).
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    members = ("m_alpha", "m_beta")
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-ghosts",
        intake_snapshot_sha256="sha256:ghosts",
        member_ids=members,
        seed_member_heads=True,
        partition_overrides={"buckets": {"runs": ["ghost-1", "ghost-2"]}, "intake": 2},
    )
    with pytest.raises(CloseObjectError, match="account"):
        _demand(
            records_root,
            intake_snapshot_id="intake-ghosts",
            intake_snapshot_sha256="sha256:ghosts",
            members=members,
        )

    # Double-listing one member across two arms while a ghost fills the count
    # is the same lie by a different arithmetic. Fresh member ids: a fixture
    # head is recorded-under exactly one snapshot, so reusing the ghosts
    # rung's members here would refuse at build for the wrong reason.
    doubled = ("m_gamma", "m_delta")
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-double",
        intake_snapshot_sha256="sha256:double",
        member_ids=doubled,
        seed_member_heads=True,
        partition_overrides={
            "buckets": {"runs": ["m_gamma"], "failed": ["m_gamma"]},
            "grant_pending": ["ghost-1"],
            "intake": 2,
        },
    )
    with pytest.raises(CloseObjectError, match="exactly one"):
        _demand(
            records_root,
            intake_snapshot_id="intake-double",
            intake_snapshot_sha256="sha256:double",
            members=doubled,
        )


@pytest.mark.smoke
def test_a_close_for_another_campaign_or_digest_licenses_nothing(
    tmp_path: Path,
) -> None:
    """The gate binds campaign AND attested digest, not the id string (break 3).

    ``find_close_object`` filters on ``intake_snapshot_id`` and ``phase`` only,
    and the gate never compared the close's ``campaign_id`` or
    ``intake_snapshot_sha256`` against the registration it was licensing. One
    shared ``records/intake/leg-closes.jsonl`` serves all four frozen campaigns
    and ``campaign_merge`` / ``merge_host_shards`` put other campaigns' rows
    into one root, so a close that self-describes as another campaign's, over a
    digest that is not the rung's, licensed the rung.
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    record_fixture_close(
        records_root,
        campaign_id="c4-native",
        intake_snapshot_id="intake-shared",
        intake_snapshot_sha256="sha256:" + "0" * 64,
        member_ids=("m_alpha",),
        seed_member_heads=True,
    )
    with pytest.raises(CloseObjectError, match="campaign"):
        _demand(
            records_root,
            intake_snapshot_id="intake-shared",
            campaign_id="c3-classics",
            intake_snapshot_sha256="sha256:" + "0" * 64,
            members=("m_alpha",),
        )
    with pytest.raises(CloseObjectError, match="digest|sha256"):
        _demand(
            records_root,
            intake_snapshot_id="intake-shared",
            campaign_id="c4-native",
            intake_snapshot_sha256="sha256:" + "1" * 64,
            members=("m_alpha",),
        )
    # Both bound correctly, the same close licenses.
    licensed = _demand(
        records_root,
        intake_snapshot_id="intake-shared",
        campaign_id="c4-native",
        intake_snapshot_sha256="sha256:" + "0" * 64,
        members=("m_alpha",),
    )
    assert licensed["campaign_id"] == "c4-native"


@pytest.mark.smoke
def test_a_fixture_leg_is_a_real_generation(tmp_path: Path) -> None:
    """The fixture close is structurally a driver close: open row, floor, proof."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    generation = open_fixture_leg(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-real",
        intake_snapshot_sha256="sha256:real",
    )
    close = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-real",
        intake_snapshot_sha256="sha256:real",
        member_ids=("m1",),
        seed_member_heads=True,
    )
    assert close["generation_id"] == generation.generation_id
    assert close["leg_open_row_sha256"] == generation.row_sha256
    assert close["sources"]["models.shard"]["present"] is True
