"""B-03: a stage must be the composition the composer attested, or refuse.

The S200-PREP hazard (``e3745a1a``): the step-200 snapshot sha was carried
forward by hand between the composition and the stage, and nothing in the
staging path could tell that the sha no longer described the members being
staged. Staging is content-addressed, so the wrong sha does not produce a
corrupt snapshot -- it produces a PERFECTLY VALID snapshot that is not the one
anybody reviewed, registered under a lineage row whose reviewers believed they
were approving something else.

Pinned here:

* the composer publishes a composition attestation (campaign, member/stratum
  pairs, and the ``(snapshot_id, snapshot_sha256)`` those members compose to)
  in its strata report, digested;
* a hand-edited attestation refuses at decode (its own digest);
* an attestation whose digest was RE-COMPUTED after the edit -- the forger who
  reads the code -- still refuses, because the identity is recomputed from the
  stage, not read from the attestation;
* a hand-edited member list refuses on the member set;
* an attestation for another campaign, or naming a stratum no composer can
  produce, refuses;
* every refusal leaves ZERO durable bytes (no snapshot, no registry row); and
* the ``register`` door walks the identical gate -- the 2026-08-16 red-team
  lesson that a gate on one entry point is a gate the sibling door bypasses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.rung_staging import (
    COMPOSITION_ATTESTATION_REPORT_KEY,
    CompositionAttestationError,
    RungCompositionAttestation,
    build_composition_attestation,
    load_composition_attestation,
    load_rung_registrations,
    planned_subset_identity,
    register_existing_snapshot,
    rung_registry_path,
    stage_rung,
)
from menagerie.crawler.tests.test_rung_stratification import (
    _CAMPAIGN,
    _TEST_SHARES,
    _records_root,
    _root_snapshot,
    _runs_award_authorities,
    _stratify,
    build_stratified_repo,
)

pytestmark = pytest.mark.smoke


@pytest.fixture()
def stratified_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the shared stratification fixture repository.

    Returns
    -------
    pathlib.Path
        Repository root holding the campaign partitions, the closed prior rung,
        and the records state every stratum detector reads.
    """

    return build_stratified_repo(tmp_path, monkeypatch)


#: A well-formed identity that is NOT the one any composition produces. The
#: digits are the S200-PREP sha prefix the ladder carried forward by hand.
_WRONG_SNAPSHOT_ID = "intake-e3745a1a" + "0" * 12
_WRONG_SNAPSHOT_SHA256 = "sha256:" + "e3745a1a" * 8


def _stage(
    repo_root: Path,
    member_ids: tuple[str, ...],
    attestation: RungCompositionAttestation | None,
    *,
    label: str = "step-200",
    rung_class: str = "successor_rung",
) -> Any:
    """Stage one rung with the fixture's provenance and the given attestation.

    The AW-2 authorities are supplied for whatever is being staged: a composed
    rung is CUMULATIVE (C-2), so it carries the lineage tip's converted models
    and the award door legitimately demands them. Passing them keeps each test
    below about the attestation gate it is actually pinning.
    """

    return stage_rung(
        _records_root(repo_root),
        campaign_id=_CAMPAIGN,
        stable_ids=member_ids,
        count=None,
        derived_from_snapshot_id=None,
        label=label,
        registered_by="pytest",
        created_at="2026-08-16T00:00:00Z",
        award_authorities=_runs_award_authorities(repo_root, member_ids),
        composition_attestation=attestation,
        # EN-2 exemption, stated explicitly: the stratified fixture universe
        # composes lockless-routing members by design (oddball-intent stratum).
        allow_lockless=("jax-flax-arm64", "tf-keras-arm64"),
        # Fixtures model the composed next-rung shape (W16/S-2 declaration).
        rung_class=rung_class,
    )


def _durable_state(repo_root: Path) -> tuple[list[str], int]:
    """Return the durable intake directory listing and registry row count."""

    records_root = _records_root(repo_root)
    intake = sorted(path.name for path in (records_root / "intake").iterdir())
    return intake, len(load_rung_registrations(records_root))


def _reattested(
    attestation: RungCompositionAttestation, **overrides: Any
) -> RungCompositionAttestation:
    """Return a RE-DIGESTED attestation with fields replaced.

    This is the forger who read the code: they edit the body AND recompute the
    ``composition_digest``, so the self-consistency check passes. Everything
    after that has to hold on its own.
    """

    fields: dict[str, Any] = {
        "campaign_id": attestation.campaign_id,
        "size": attestation.size,
        "seed": attestation.seed,
        "member_strata": dict(attestation.member_strata),
        "snapshot_id": attestation.snapshot_id,
        "snapshot_sha256": attestation.snapshot_sha256,
        "pd7_disposition": attestation.pd7_disposition,
    }
    fields.update(overrides)
    return build_composition_attestation(**fields)


def _report_with(report: Mapping[str, Any], attestation_row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a strata report carrying a replaced attestation object."""

    return {**report, COMPOSITION_ATTESTATION_REPORT_KEY: dict(attestation_row)}


# ---------------------------------------------------------------------------
# The composer attests what it composed.
# ---------------------------------------------------------------------------


def test_plan_report_attests_the_composition(stratified_repo: Path) -> None:
    """The strata report carries a digested, self-consistent attestation."""

    plan = _stratify(stratified_repo)
    attestation = plan.composition_attestation
    assert attestation.campaign_id == _CAMPAIGN
    assert attestation.member_ids == {member.stable_id for member in plan.members}
    assert dict(attestation.member_strata) == {
        member.stable_id: member.stratum for member in plan.members
    }
    # The attested identity is exactly what staging these members produces.
    assert (attestation.snapshot_id, attestation.snapshot_sha256) == planned_subset_identity(
        _records_root(stratified_repo), _CAMPAIGN, plan.member_ids()
    )
    row = plan.to_report()[COMPOSITION_ATTESTATION_REPORT_KEY]
    assert row["composition_digest"].startswith("sha256:")
    # SOL-02: the fixture composes with a PD-7 disposition, so the attestation
    # is v2 and the disposition rides INSIDE the digested body.
    assert row["schema_version"] == "menagerie.crawler.rung-composition-attestation.v2"
    assert row["pd7"] == {
        "status": "SATISFIED",
        "source_attestation_sha256": "sha256:" + "7d" * 32,
    }
    # Member ORDER is deliberately not attested: the subset writer emits root
    # manifest order, so order is not part of what gets staged.
    assert row["member_strata"] == sorted(row["member_strata"])
    assert attestation.strata_histogram() == {
        stratum: count for stratum, count in plan.selected_counts.items() if count
    }


def test_report_round_trips_through_the_loader(stratified_repo: Path, tmp_path: Path) -> None:
    """The composer's own ``--out-report`` file is what the door consumes."""

    plan = _stratify(stratified_repo)
    report_path = tmp_path / "step-200-report.json"
    report_path.write_text(json.dumps(plan.to_report(), sort_keys=True), encoding="utf-8")
    assert load_composition_attestation(report_path) == plan.composition_attestation


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"member_strata": "m_0000"}, "must be a non-empty list"),
        ({"member_strata": {"m_0000": "ladder-bulk"}}, "must be a non-empty list"),
        ({"member_strata": []}, "must be a non-empty list"),
        ({"member_strata": [["m_0000", "ladder-bulk", "extra"]]}, "must be"),
        ({"member_strata": [[1, "ladder-bulk"]]}, "must be"),
        (
            {"member_strata": [["m_0000", "ladder-bulk"], ["m_0000", "hard-tail"]]},
            "names a member twice",
        ),
        ({"size": True}, "size must be an integer"),
        ({"seed": "seven"}, "seed must be an integer"),
        ({"campaign_id": ""}, "campaign_id must be non-empty"),
        ({"snapshot_id": "not-a-snapshot"}, "snapshot_id is not a snapshot id"),
        ({"snapshot_sha256": "deadbeef"}, "snapshot_sha256 is invalid"),
    ],
)
def test_malformed_attestations_refuse_typed(
    stratified_repo: Path, tmp_path: Path, mutation: Mapping[str, Any], expected: str
) -> None:
    """The decoder refuses structurally, never with a stray TypeError.

    The decode step is the first line of defence, and a door that crashes on a
    malformed attestation instead of refusing one is a door whose refusal an
    operator cannot distinguish from a bug.
    """

    plan = _stratify(stratified_repo)
    row = {**plan.to_report()[COMPOSITION_ATTESTATION_REPORT_KEY], **mutation}
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")
    with pytest.raises(CompositionAttestationError, match=expected):
        load_composition_attestation(path)


def test_a_report_without_an_attestation_refuses(tmp_path: Path) -> None:
    """A pre-B-03 (v2) report cannot silently pass as an attestation."""

    stale = tmp_path / "v2-report.json"
    stale.write_text(
        json.dumps({"schema_version": "menagerie.crawler.stratified-rung-plan.v2"}),
        encoding="utf-8",
    )
    with pytest.raises(CompositionAttestationError, match="schema must be"):
        load_composition_attestation(stale)


# ---------------------------------------------------------------------------
# The happy path: the composer's attestation licenses its own stage.
# ---------------------------------------------------------------------------


def test_the_composers_attestation_stages(stratified_repo: Path) -> None:
    """The composer's attestation licenses its own stage, end to end.

    Since C-2 the composition is cumulative -- it carries the lineage tip -- so
    the attested member set IS a successor of the registry tip and the whole gate
    stack (widen, close, award, B-03) is satisfiable at once. That this test used
    to assert a refusal was the defect: no attested composition could ever stage.
    """

    plan = _stratify(stratified_repo)
    staged = _stage(stratified_repo, plan.member_ids(), plan.composition_attestation)
    attestation = plan.composition_attestation
    assert staged.registration_created is True
    assert staged.registration.snapshot_id == attestation.snapshot_id
    assert staged.registration.snapshot_sha256 == attestation.snapshot_sha256
    assert staged.registration.member_count == len(attestation.member_ids)


def test_absent_attestation_leaves_staging_unchanged(stratified_repo: Path) -> None:
    """The gate adjudicates a presented attestation; it invents no new refusal.

    B-03 is disposition-negative: it can only refuse a stage. A door with no
    attestation to check must behave exactly as one that got a matching
    attestation, or landing the gate would itself be the outage.
    """

    plan = _stratify(stratified_repo)
    # SOL-02 narrows the optionality to PRE-LAUNCH classes: a successor-rung
    # stage now REQUIRES the PD-7-bearing attestation, so the
    # disposition-negative pin holds for the canary-seed shape.
    staged = _stage(stratified_repo, plan.member_ids(), None, rung_class="canary_seed")
    assert staged.registration_created is True
    assert staged.registration.snapshot_id == plan.composition_attestation.snapshot_id


# ---------------------------------------------------------------------------
# The three refusals.
# ---------------------------------------------------------------------------


def test_hand_edited_attestation_refuses_at_decode(stratified_repo: Path, tmp_path: Path) -> None:
    """Editing the sha in the report breaks the attestation's own digest.

    The first line of defence, and the one the S200-PREP hazard would have hit:
    a sha pasted over the composer's output no longer matches the body it was
    signed with.
    """

    plan = _stratify(stratified_repo)
    row = dict(plan.to_report()[COMPOSITION_ATTESTATION_REPORT_KEY])
    row["snapshot_sha256"] = _WRONG_SNAPSHOT_SHA256
    report_path = tmp_path / "edited-report.json"
    report_path.write_text(
        json.dumps(_report_with(plan.to_report(), row), sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(CompositionAttestationError, match="composition_digest does not match"):
        load_composition_attestation(report_path)


def test_recomputed_digest_still_refuses_on_the_composed_identity(
    stratified_repo: Path,
) -> None:
    """A re-digested attestation refuses: identity comes from the STAGE.

    The gate is not a checksum of the attestation. A forger (or, identically to
    this gate, a genuinely STALE attestation whose records state has moved) can
    present a perfectly self-consistent attestation naming an identity these
    members do not compose to -- and the door recomputes that identity from the
    members it is actually staging, so the two disagree and staging refuses.
    """

    plan = _stratify(stratified_repo)
    forged = _reattested(
        plan.composition_attestation,
        snapshot_sha256=_WRONG_SNAPSHOT_SHA256,
        snapshot_id=_WRONG_SNAPSHOT_ID,
    )
    before = _durable_state(stratified_repo)
    with pytest.raises(CompositionAttestationError) as excinfo:
        _stage(stratified_repo, plan.member_ids(), forged)
    message = str(excinfo.value)
    assert "STALE" in message
    assert plan.composition_attestation.snapshot_id in message
    assert forged.snapshot_id in message
    assert _durable_state(stratified_repo) == before


def test_hand_edited_member_list_refuses(stratified_repo: Path) -> None:
    """Staging a member list the composition never produced refuses.

    Both directions are named in the refusal, because both happen: a member
    dropped from the ids file by hand, and a member appended to it.
    """

    plan = _stratify(stratified_repo)
    attestation = plan.composition_attestation
    members = plan.member_ids()
    before = _durable_state(stratified_repo)

    with pytest.raises(CompositionAttestationError) as dropped:
        _stage(stratified_repo, members[:-1], attestation)
    assert members[-1] in str(dropped.value)
    assert "attested-but-absent" in str(dropped.value)

    outsider = next(
        item.stable_id
        for item in _root_snapshot(stratified_repo).items
        if item.stable_id not in attestation.member_ids
    )
    with pytest.raises(CompositionAttestationError) as added:
        _stage(stratified_repo, (*members, outsider), attestation)
    assert outsider in str(added.value)
    assert "staged-but-unattested" in str(added.value)

    assert _durable_state(stratified_repo) == before
    assert (
        not rung_registry_path(_records_root(stratified_repo))
        .read_text(encoding="utf-8")
        .count(attestation.snapshot_id)
    )


def test_the_gate_reverifies_its_own_argument(stratified_repo: Path) -> None:
    """The gate does not trust HOW its attestation was constructed.

    The decode path checks the digest, but a hand-assembled dataclass never went
    through it. If the gate assumed its argument had been decoded, an attestation
    carrying a digest nothing ever verified would reach the comparisons -- so the
    gate re-derives it itself.
    """

    plan = _stratify(stratified_repo)
    unverified = RungCompositionAttestation(
        campaign_id=_CAMPAIGN,
        size=plan.size,
        seed=plan.seed,
        member_strata=plan.composition_attestation.member_strata,
        snapshot_id=plan.composition_attestation.snapshot_id,
        snapshot_sha256=plan.composition_attestation.snapshot_sha256,
        composition_digest="sha256:" + "f" * 64,
    )
    with pytest.raises(CompositionAttestationError, match="composition_digest does not match"):
        _stage(stratified_repo, plan.member_ids(), unverified)


def test_attestation_for_another_campaign_refuses(stratified_repo: Path) -> None:
    """An attestation composed for a different campaign adjudicates nothing."""

    plan = _stratify(stratified_repo)
    foreign = _reattested(plan.composition_attestation, campaign_id="c3-classics")
    with pytest.raises(CompositionAttestationError, match="composed for c3-classics"):
        _stage(stratified_repo, plan.member_ids(), foreign)


def test_unknown_stratum_refuses(stratified_repo: Path) -> None:
    """The stratum vocabulary is closed: an invented stratum is not a plan."""

    plan = _stratify(stratified_repo)
    strata = dict(plan.composition_attestation.member_strata)
    strata[plan.member_ids()[0]] = "hand-picked"
    with pytest.raises(CompositionAttestationError, match="unknown stratum 'hand-picked'"):
        build_composition_attestation(
            campaign_id=_CAMPAIGN,
            size=plan.size,
            seed=plan.seed,
            member_strata=strata,
            snapshot_id=plan.composition_attestation.snapshot_id,
            snapshot_sha256=plan.composition_attestation.snapshot_sha256,
        )


# ---------------------------------------------------------------------------
# The gate binds the evidence, never the entry point.
# ---------------------------------------------------------------------------


def test_register_door_walks_the_same_gate(stratified_repo: Path) -> None:
    """``register`` enforces the attestation against the ADOPTED identity.

    2026-08-16 red-team break 1 in this module: a gate that lives only on
    ``stage_rung`` is a gate the ``register`` door adopts its way around. Here
    the adopted snapshot is real and valid; only the attestation is wrong, and
    the sibling door refuses it exactly as ``stage_rung`` does.
    """

    plan = _stratify(stratified_repo)
    from menagerie.crawler.rung_staging import _write_subset_snapshot

    root = _root_snapshot(stratified_repo)
    staged_snapshot, _created = _write_subset_snapshot(
        root.root,
        plan.member_ids(),
        _records_root(stratified_repo) / "intake",
    )
    forged = _reattested(
        plan.composition_attestation,
        snapshot_sha256=_WRONG_SNAPSHOT_SHA256,
        snapshot_id=_WRONG_SNAPSHOT_ID,
    )
    with pytest.raises(CompositionAttestationError, match="STALE"):
        register_existing_snapshot(
            _records_root(stratified_repo),
            staged_snapshot.root,
            campaign_id=_CAMPAIGN,
            label="step-200-readopt",
            registered_by="pytest",
            created_at="2026-08-16T00:00:00Z",
            historical_only=False,
            composition_attestation=forged,
        )


# ---------------------------------------------------------------------------
# SOL-02: B-03 CONSUMES the PD-7 disposition, digest-bound.
# ---------------------------------------------------------------------------


def test_blocking_pd7_disposition_refuses_every_stage(stratified_repo: Path) -> None:
    """UNSATISFIED-BLOCKING refuses the stage regardless of rung class."""

    plan = _stratify(stratified_repo)
    blocking = _reattested(
        plan.composition_attestation,
        pd7_disposition={
            "status": "UNSATISFIED-BLOCKING",
            "source_attestation_sha256": "sha256:" + "7d" * 32,
        },
    )
    before = _durable_state(stratified_repo)
    for rung_class in ("successor_rung", "canary_seed"):
        with pytest.raises(CompositionAttestationError, match="UNSATISFIED-BLOCKING"):
            _stage(stratified_repo, plan.member_ids(), blocking, rung_class=rung_class)
    assert _durable_state(stratified_repo) == before


def test_successor_stage_demands_a_satisfied_pd7_disposition(stratified_repo: Path) -> None:
    """The R2 hole, closed: a successor stage without a bound SATISFIED pd7 refuses.

    All three missing-consumption shapes are the same refusal: no attestation
    at all, a pre-vocabulary v1 attestation (no pd7 block), and a bound
    NOT-EVALUATED status. A green side file the attestation does not bind can
    no longer stage the 200b shape.
    """

    plan = _stratify(stratified_repo)
    before = _durable_state(stratified_repo)
    with pytest.raises(CompositionAttestationError, match="pd7"):
        _stage(stratified_repo, plan.member_ids(), None, rung_class="successor_rung")
    v1 = _reattested(plan.composition_attestation, pd7_disposition=None)
    with pytest.raises(CompositionAttestationError, match="pd7"):
        _stage(stratified_repo, plan.member_ids(), v1, rung_class="successor_rung")
    not_evaluated = _reattested(
        plan.composition_attestation,
        pd7_disposition={
            "status": "NOT-EVALUATED",
            "source_attestation_sha256": "sha256:" + "7d" * 32,
        },
    )
    with pytest.raises(CompositionAttestationError, match="SATISFIED"):
        _stage(stratified_repo, plan.member_ids(), not_evaluated, rung_class="successor_rung")
    assert _durable_state(stratified_repo) == before
    # The register door walks the identical PD-7 gate (red-team break 1).
    from menagerie.crawler.rung_staging import _write_subset_snapshot

    root = _root_snapshot(stratified_repo)
    staged_snapshot, _created = _write_subset_snapshot(
        root.root,
        plan.member_ids(),
        _records_root(stratified_repo) / "intake",
    )
    with pytest.raises(CompositionAttestationError, match="pd7"):
        register_existing_snapshot(
            _records_root(stratified_repo),
            staged_snapshot.root,
            campaign_id=_CAMPAIGN,
            label="step-200-readopt",
            registered_by="pytest",
            created_at="2026-08-16T00:00:00Z",
            historical_only=False,
            composition_attestation=v1,
            rung_class="successor_rung",
        )


def test_pd7_disposition_is_digest_bound(stratified_repo: Path) -> None:
    """Editing the bound pd7 status after the composer signed breaks the digest.

    The no-side-file-trust half of SOL-02: the status travels INSIDE
    ``composition_digest``, so a post-hoc edit (SATISFIED painted over a
    blocking adjudication) refuses at decode, and the closed vocabulary
    refuses any invented status.
    """

    plan = _stratify(stratified_repo)
    row = dict(plan.composition_attestation.to_row())
    row["pd7"] = {**row["pd7"], "status": "UNSATISFIED-BLOCKING"}
    with pytest.raises(CompositionAttestationError, match="composition_digest"):
        _decode_row(row)
    junk = dict(plan.composition_attestation.to_row())
    junk["pd7"] = {**junk["pd7"], "status": "ALL-GOOD-TRUST-ME"}
    with pytest.raises(CompositionAttestationError, match="closed vocabulary"):
        _decode_row(junk)
    # A v1 attestation smuggling a pd7 block refuses: PD-7-bearing rows
    # declare the v2 schema, so the block is always inside a digest.
    v1_row = dict(_reattested(plan.composition_attestation, pd7_disposition=None).to_row())
    v1_row["pd7"] = {
        "status": "SATISFIED",
        "source_attestation_sha256": "sha256:" + "7d" * 32,
    }
    with pytest.raises(CompositionAttestationError, match="v1 composition attestation"):
        _decode_row(v1_row)


def _decode_row(row: Mapping[str, Any]) -> RungCompositionAttestation:
    """Decode one raw attestation row through the production parser."""

    from menagerie.crawler.rung_staging import _composition_attestation_from_row

    return _composition_attestation_from_row(row, source="test")


# ---------------------------------------------------------------------------
# The operator-facing surface: the refusal prints and exits, never tracebacks.
# ---------------------------------------------------------------------------


def test_cli_stages_the_report_and_refuses_a_forged_one(
    stratified_repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--composition-attestation`` takes the report; a bad one exits 2."""

    from menagerie.crawler.tools.stage_rung import main as stage_main
    from menagerie.crawler.tools.stratify_rung import main as stratify_main

    members_path = tmp_path / "members.txt"
    report_path = tmp_path / "report.json"
    # SOL-02 end to end: the CLI consumes the PD-7 adjudication artifact and
    # binds its status + file digest into the attestation the door verifies.
    pd7_path = tmp_path / "pd7-adjudication.json"
    pd7_path.write_text(
        json.dumps({"pd7_slop_family_routes": {"status": "SATISFIED"}}, sort_keys=True),
        encoding="utf-8",
    )
    assert (
        stratify_main(
            [
                "--repo-root",
                str(stratified_repo),
                "--campaign",
                _CAMPAIGN,
                "--size",
                "10",
                "--seed",
                "7",
                "--pd7-attestation",
                str(pd7_path),
                "--out-members",
                str(members_path),
                "--out-report",
                str(report_path),
                *(
                    part
                    for name, share in sorted(_TEST_SHARES.items())
                    for part in ("--share", f"{name}={share}")
                ),
            ]
        )
        == 0
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    forged_path = tmp_path / "forged-report.json"
    forged = _reattested(
        load_composition_attestation(report_path),
        snapshot_sha256=_WRONG_SNAPSHOT_SHA256,
        snapshot_id=_WRONG_SNAPSHOT_ID,
    )
    forged_path.write_text(
        json.dumps(_report_with(report, forged.to_row()), sort_keys=True), encoding="utf-8"
    )

    # The report's own disclosure IS the authority list an operator passes: the
    # composed rung carries the lineage tip, whose converted models are operator
    # property (AW-2), and the door re-derives the demand from the ledger.
    authority_flags = [
        part
        for stable_id, revision in sorted(report["award_authorities_required"].items())
        for part in ("--award-authority", f"{stable_id}={revision}")
    ]
    assert authority_flags, "the fixture tip must hold a runs award to carry"

    def stage(attestation_path: Path) -> int:
        return stage_main(
            [
                "stage",
                "--repo-root",
                str(stratified_repo),
                "--campaign",
                _CAMPAIGN,
                "--label",
                "step-200",
                "--rung-class",
                "successor_rung",
                "--stable-ids-file",
                str(members_path),
                "--composition-attestation",
                str(attestation_path),
                # EN-2 exemption for the fixture's deliberate lockless members.
                "--allow-lockless",
                "jax-flax-arm64",
                "--allow-lockless",
                "tf-keras-arm64",
                *authority_flags,
            ]
        )

    before = _durable_state(stratified_repo)
    assert stage(forged_path) == 2
    assert "stage-rung refused" in capsys.readouterr().err
    assert _durable_state(stratified_repo) == before

    # The composer's own report stages -- the documented two-command sequence
    # composes end to end through the CLI (C-2), award authorities included.
    assert stage(report_path) == 0
    staged = json.loads(capsys.readouterr().out)
    assert staged["snapshot_id"] == report[COMPOSITION_ATTESTATION_REPORT_KEY]["snapshot_id"]
    assert _durable_state(stratified_repo) != before
