"""C-2: a stratified composition must be able to stage the NEXT rung.

Three individually-correct constraints were jointly UNSATISFIABLE, so once any
rung was registered no attested stratified composition could ever stage a
successor:

#. the staging door's widen rule -- a successor derives from the registry tip
   and STRICTLY WIDENS it (2026-08-16 red-team round 2, break 1);
#. the composer's dedup -- ``ladder-bulk`` excludes every prior rung member by
   design, so a composition was a DELTA, never a cumulative membership; and
#. B-03 -- the staged set must EQUAL the composed set, so the operator may not
   hand-union the tip's members back in.

The pilot's refusal was the proof: ``stage-rung refused: cannot stage
'step-200': a successor must strictly widen the lineage tip (10 -> 200
members)``. The fix is on the COMPOSER: it carries the lineage tip explicitly as
a disclosed ``carried-tip`` stratum, so widening holds by construction, B-03
stays exact against a tip-carrying attestation, and the carrying is VISIBLE in
the report instead of implicit in door logic.

The permanent tripwire this module exists to be: the exact two-generation
sequence the operator flagged as never exercised -- compose -> stage/register a
rung, then compose -> stage its SUCCESSOR. Everything else here pins that the
fix STRENGTHENED the doors rather than loosening them:

* strict widening stays strict, and a delta-only member list still refuses;
* the AW-2 award-authority demand EXTENDS to carried runs-award holders (the
  tip's converted models are operator property, carried or not);
* an attestation may not claim ``carried-tip`` provenance for a member the
  registry tip does not contain, nor claim it at all with no tip registered;
* the carried stratum is never backfilled from ``ladder-bulk``; and
* a composition that adds nothing beyond the tip refuses at composition time
  instead of producing a rung the door must reject.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.rung_staging import (
    CARRIED_TIP_STRATUM,
    RUNG_CLASS_SUCCESSOR,
    RungStagingError,
    build_composition_attestation,
    load_rung_registrations,
    stage_rung,
)
from menagerie.crawler.rung_stratification import stratify_rung_members
from menagerie.crawler.tests.test_rung_stratification import (
    _CAMPAIGN,
    _PRIOR_RUNG_NAMES,
    _TEST_SHARES,
    _ids_by_name,
    _records_root,
    build_stratified_repo,
)

pytestmark = pytest.mark.smoke

#: Extra fresh c1-mech rows so TWO generations have room: ladder bulk needs a
#: fresh prefix per rung, and ``unsampled-zoo`` needs a zoo no earlier rung
#: sampled. The shared fixture is sized for ONE composition, where an exhausted
#: partition (a rung-sizing refusal) or a vanished tail class would mask what
#: this module is about. Plain zoo names on purpose: they derive no oddball
#: routing intent, so the salting classes stay as the shared fixture set them.
_EXTRA_ROSTER_ROWS: tuple[dict[str, Any], ...] = tuple(
    {"name": f"wide-{zoo}-{index:02d}", "zoo": f"zoo-wide-{zoo}", "variant": ""}
    for zoo in range(4)
    for index in range(6)
)

_RUN_ID = "pytest-close"


@pytest.fixture()
def succession_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return the shared stratification world, widened for two generations.

    Returns
    -------
    pathlib.Path
        Repository root whose c1-mech records hold the registered, CLOSED
        step-100 tip plus enough fresh partition rows for two successors.
    """

    return build_stratified_repo(tmp_path, monkeypatch, extra_roster_rows=_EXTRA_ROSTER_ROWS)


def _compose(
    repo_root: Path,
    *,
    campaign_id: str = _CAMPAIGN,
    size: int = 10,
    seed: int = 7,
    shares: Optional[Mapping[str, float]] = None,
) -> Any:
    """Compose one stratified rung against the fixture's records state."""

    from menagerie.crawler.tests.test_rung_stratification import _TEST_PD7_DISPOSITION

    return stratify_rung_members(
        _records_root(repo_root),
        campaign_id=campaign_id,
        size=size,
        seed=seed,
        shares=dict(_TEST_SHARES if shares is None else shares),
        # SOL-02: successor stages demand the PD-7-bearing v2 attestation.
        pd7_disposition=_TEST_PD7_DISPOSITION,
    )


def _stage(
    repo_root: Path,
    plan: Any,
    *,
    label: str,
    member_ids: Optional[Sequence[str]] = None,
    award_authorities: Optional[Mapping[str, str]] = None,
    attestation: Any = "plan",
    rung_class: Optional[str] = RUNG_CLASS_SUCCESSOR,
) -> Any:
    """Stage one composed rung through the real door, attestation and all.

    This module's scenario IS succession -- staging the strict widening of a
    CLOSED lineage tip -- so the staging-door class declaration (W16/S-2)
    defaults to ``successor_rung``. The one root exemption case (a first rung
    with no lineage and no live close) passes ``rung_class=None`` explicitly.
    """

    return stage_rung(
        _records_root(repo_root),
        campaign_id=plan.campaign_id,
        stable_ids=tuple(plan.member_ids() if member_ids is None else member_ids),
        count=None,
        derived_from_snapshot_id=None,
        label=label,
        registered_by="pytest",
        created_at="2026-08-19T00:00:00Z",
        rung_class=rung_class,
        award_authorities=dict(award_authorities or {}) or None,
        composition_attestation=(
            plan.composition_attestation if attestation == "plan" else attestation
        ),
        # EN-2 exemption, stated explicitly: the stratified fixture universe
        # composes lockless-routing members by design (oddball-intent stratum).
        allow_lockless=("jax-flax-arm64", "tf-keras-arm64"),
    )


def _members_of(repo_root: Path, snapshot_id: str) -> frozenset[str]:
    """Return one registered rung's durable membership."""

    snapshot = load_intake_snapshot(_records_root(repo_root) / "intake" / snapshot_id)
    return frozenset(item.stable_id for item in snapshot.items)


def _model_heads(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Return the fixture's current model heads, keyed by stable id."""

    path = _records_root(repo_root) / "models" / "current-shard.jsonl"
    heads: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            heads[str(row["stable_id"])] = row
    return heads


def _award_authorities_for(repo_root: Path, member_ids: Sequence[str]) -> dict[str, str]:
    """Return the AW-2 authorities the door demands, derived independently.

    Read straight from the ledger bytes here rather than through the staging
    helper: a disclosure that agrees with itself proves nothing.
    """

    heads = _model_heads(repo_root)
    demanded: dict[str, str] = {}
    for stable_id in member_ids:
        head = heads.get(str(stable_id))
        if head is None:
            continue
        if dict(head.get("status") or {}).get("kind") == "runs":
            demanded[str(stable_id)] = str(head["record_revision"])
    return demanded


def _close_rung(repo_root: Path, staged: Any, *, closed_at: str) -> None:
    """Close one staged rung HONESTLY, admitting exactly what it introduces.

    A cumulative rung's members split in two: the ones it INTRODUCES (which the
    leg works, so they get in-window heads recorded under this snapshot) and the
    inherited ones, which the ancestor close that adjudicated them CARRIES. That
    is the same split ``close_scale_support.close_rung_honestly`` performs for
    the ``--count`` ladder -- a stratified rung is cumulative in exactly the
    same way, which is the whole point of the carried stratum.
    """

    from menagerie.crawler.leg_generation import record_leg_plan
    from menagerie.crawler.rung_staging import introduced_member_ids
    from menagerie.crawler.tests import close_object_support as support

    records_root = _records_root(repo_root)
    registration = staged.registration
    members = sorted(_members_of(repo_root, registration.snapshot_id))
    introduced = sorted(introduced_member_ids(records_root, registration))
    generation = support.open_fixture_leg(
        records_root,
        campaign_id=registration.campaign_id,
        intake_snapshot_id=registration.snapshot_id,
        intake_snapshot_sha256=registration.snapshot_sha256,
        run_id=_RUN_ID,
    )
    support._seed_member_heads(records_root, introduced, registration.snapshot_id, _RUN_ID)
    record_leg_plan(
        records_root,
        campaign_id=registration.campaign_id,
        generation_id=generation.generation_id,
        run_id=_RUN_ID,
        admitted_ids=introduced,
        plan_payload={"fixture": True, "admitted_ids": introduced},
        created_at=closed_at,
    )
    support.record_fixture_close(
        records_root,
        campaign_id=registration.campaign_id,
        intake_snapshot_id=registration.snapshot_id,
        intake_snapshot_sha256=registration.snapshot_sha256,
        member_ids=members,
        seed_member_heads=False,
        completion_status="complete-with-deferrals",
        closed_at=closed_at,
        run_id=_RUN_ID,
    )


# ---------------------------------------------------------------------------
# The tripwire: two generations, both composed by the stratifier.
# ---------------------------------------------------------------------------


def test_two_composed_generations_stage_in_sequence(succession_repo: Path) -> None:
    """compose -> stage -> close -> compose -> stage the SUCCESSOR.

    The sequence the operator flagged as never exercised, and the one that was
    unsatisfiable: the step-200 dry-run gate had no registered stratified
    predecessor, so nothing ever asked the composer for a rung that had to widen
    one of its own compositions.
    """

    records_root = _records_root(succession_repo)
    tip_members = _members_of(succession_repo, _registered_tip(succession_repo).snapshot_id)

    first = _compose(succession_repo)
    staged_first = _stage(
        succession_repo,
        first,
        label="step-200",
        award_authorities=_award_authorities_for(succession_repo, first.member_ids()),
    )
    first_members = _members_of(succession_repo, staged_first.registration.snapshot_id)
    assert staged_first.registration_created is True
    assert staged_first.registration.derived_from_snapshot_id == _tip_before(first)
    assert tip_members < first_members

    _close_rung(succession_repo, staged_first, closed_at="2026-08-19T01:00:00Z")

    second = _compose(succession_repo, seed=8)
    staged_second = _stage(
        succession_repo,
        second,
        label="step-300",
        award_authorities=_award_authorities_for(succession_repo, second.member_ids()),
    )
    second_members = _members_of(succession_repo, staged_second.registration.snapshot_id)
    assert staged_second.registration_created is True
    assert (
        staged_second.registration.derived_from_snapshot_id == staged_first.registration.snapshot_id
    )
    assert first_members < second_members

    lineage = [row.label for row in load_rung_registrations(records_root)]
    assert lineage == ["step-100", "step-200", "step-300"]
    # Each generation crossed the boundary on its predecessor's close, not on a
    # skipped gate: a licensed successor names the close object that licensed it.
    assert staged_first.registration.licensed_by_close_sha256
    assert staged_second.registration.licensed_by_close_sha256
    assert (
        staged_second.registration.licensed_by_close_sha256
        != staged_first.registration.licensed_by_close_sha256
    )


def _registered_tip(repo_root: Path) -> Any:
    """Return the campaign's current registry tip registration."""

    return [
        row
        for row in load_rung_registrations(_records_root(repo_root))
        if row.campaign_id == _CAMPAIGN
    ][-1]


def _tip_before(plan: Any) -> str:
    """Return the snapshot id the plan says it carried."""

    return str(plan.to_report()["carried_tip"]["snapshot_id"])


# ---------------------------------------------------------------------------
# The carried stratum is honest.
# ---------------------------------------------------------------------------


def test_report_discloses_the_carried_lineage_tip(succession_repo: Path) -> None:
    """Member ids, count, and ``lineage-tip`` provenance are all in the report."""

    ids = _ids_by_name(succession_repo)
    tip = _registered_tip(succession_repo)
    tip_members = _members_of(succession_repo, tip.snapshot_id)

    plan = _compose(succession_repo)
    block = plan.to_report()["carried_tip"]
    assert block["provenance"] == "lineage-tip"
    assert block["snapshot_id"] == tip.snapshot_id
    assert block["snapshot_sha256"] == tip.snapshot_sha256
    assert block["label"] == tip.label
    assert block["member_count"] == len(tip_members) == len(_PRIOR_RUNG_NAMES)
    assert set(block["member_ids"]) == tip_members
    # Every tip member is in the composition, whatever stratum owns its label.
    assert tip_members <= set(plan.member_ids())
    # The carried stratum is disclosed member by member, and its members are
    # exactly the tip members no other stratum selected on its own evidence.
    labels = {member.stable_id: member.stratum for member in plan.members}
    carried = {stable_id for stable_id, name in labels.items() if name == CARRIED_TIP_STRATUM}
    assert carried == set(block["carried_forward"])
    assert carried <= tip_members
    assert carried | set(block["also_selected_on_own_evidence"]) == tip_members
    for stable_id, stratum in block["also_selected_on_own_evidence"].items():
        assert labels[stable_id] == stratum != CARRIED_TIP_STRATUM
    # A converted model is carried because membership is cumulative, and it is
    # never re-selected as fresh work.
    assert labels[ids["conv-done"]] == CARRIED_TIP_STRATUM


def test_carried_members_are_never_backfilled_from_bulk(succession_repo: Path) -> None:
    """``ladder-bulk`` means "next FRESH partition row" and still does.

    A carried member laundered as bulk would be the silent backfill the
    disclosure exists to prevent: the rung would look like fresh work.
    """

    plan = _compose(succession_repo)
    tip_members = _members_of(succession_repo, _registered_tip(succession_repo).snapshot_id)
    bulk = {member.stable_id for member in plan.members if member.stratum == "ladder-bulk"}
    assert bulk and not bulk & tip_members
    # The composed width is what --size bought; carrying is disclosed on top of
    # it, never taken out of it.
    composed = {
        stratum: count
        for stratum, count in plan.selected_counts.items()
        if stratum != CARRIED_TIP_STRATUM
    }
    assert sum(composed.values()) == plan.size == 10
    assert len(plan.members) == plan.size + len(plan.to_report()["carried_tip"]["carried_forward"])


def test_carried_stratum_is_not_share_bearing(succession_repo: Path) -> None:
    """Carrying is mandatory, so it takes no share and no acknowledgement."""

    from menagerie.crawler.rung_stratification import StratificationError

    shares = dict(_TEST_SHARES)
    shares[CARRIED_TIP_STRATUM] = 0.0
    with pytest.raises(StratificationError, match="unknown strata"):
        _compose(succession_repo, shares=shares)
    with pytest.raises(StratificationError, match="acknowledge"):
        stratify_rung_members(
            _records_root(succession_repo),
            campaign_id=_CAMPAIGN,
            size=10,
            seed=7,
            shares=dict(_TEST_SHARES),
            acknowledged_empty_strata=(CARRIED_TIP_STRATUM,),
        )


# ---------------------------------------------------------------------------
# The doors got STRONGER, not looser.
# ---------------------------------------------------------------------------


def test_carried_runs_award_holder_demands_its_authority(succession_repo: Path) -> None:
    """A carried award is still operator property (AW-2).

    Carrying a settled award without its authority is exactly what the staging
    door exists to refuse, so the demand extends to the carried stratum: the
    tip's converted models reach the successor's membership by construction now,
    and every one of them must be named with its EXACT current revision.
    """

    ids = _ids_by_name(succession_repo)
    plan = _compose(succession_repo)
    demanded = _award_authorities_for(succession_repo, plan.member_ids())
    assert set(demanded) == {ids["conv-done"]}, "the fixture tip must hold one runs award"
    # The composition discloses the demand it creates, exactly as the door makes
    # it -- so the operator never hand-tallies the authority list.
    assert plan.to_report()["award_authorities_required"] == demanded

    with pytest.raises(RungStagingError, match="award authority") as excinfo:
        _stage(succession_repo, plan, label="step-200-unauthorized")
    assert ids["conv-done"] in str(excinfo.value)
    assert not [
        row
        for row in load_rung_registrations(_records_root(succession_repo))
        if row.label == "step-200-unauthorized"
    ]

    # A revision that is not the head's exact current one licenses nothing.
    with pytest.raises(RungStagingError, match="award authority"):
        _stage(
            succession_repo,
            plan,
            label="step-200-stale-authority",
            award_authorities={ids["conv-done"]: "sha256:" + "0" * 64},
        )

    staged = _stage(succession_repo, plan, label="step-200", award_authorities=demanded)
    assert staged.registration_created is True


def test_delta_only_member_list_still_refuses(succession_repo: Path) -> None:
    """Stripping the carried members back out is still not a successor.

    The round-2 break-1 property, unchanged: a delta cannot masquerade as a
    cumulative successor. With the composer's own attestation the B-03 member
    check fires first (the list is not what was composed); with no attestation
    the widen rule refuses it.
    """

    from menagerie.crawler.rung_staging import CompositionAttestationError

    plan = _compose(succession_repo)
    tip_members = _members_of(succession_repo, _registered_tip(succession_repo).snapshot_id)
    delta = tuple(stable_id for stable_id in plan.member_ids() if stable_id not in tip_members)
    assert delta and len(delta) < len(plan.member_ids())

    with pytest.raises(CompositionAttestationError, match="attested-but-absent"):
        _stage(succession_repo, plan, label="step-200-delta", member_ids=delta)
    # SOL-02: a successor stage without any attestation now refuses at the
    # PD-7 consumption door, so the widen half is reached with a delta-shaped
    # attestation (its member set matches; only the widen rule can refuse).
    from menagerie.crawler.tests.rt3_support import fixture_composition_attestation

    with pytest.raises(RungStagingError, match="strictly widen"):
        _stage(
            succession_repo,
            plan,
            label="step-200-delta",
            member_ids=delta,
            attestation=fixture_composition_attestation(
                _records_root(succession_repo), plan.campaign_id, stable_ids=delta
            ),
            award_authorities=_award_authorities_for(succession_repo, delta),
        )
    # The no-attestation shape refuses too — at the PD-7 door, typed.
    with pytest.raises(CompositionAttestationError, match="pd7"):
        _stage(
            succession_repo,
            plan,
            label="step-200-delta",
            member_ids=delta,
            attestation=None,
            award_authorities=_award_authorities_for(succession_repo, delta),
        )


def test_carried_tip_claim_must_be_true(succession_repo: Path) -> None:
    """No member may claim ``lineage-tip`` provenance it does not have.

    The label lives inside the composition digest, so a re-digested attestation
    is the only way to forge one -- and the door recomputes the tip's membership
    from the registry, so the forged claim refuses. Disposition-negative: the
    check can only refuse a stage.
    """

    plan = _compose(succession_repo)
    tip_members = _members_of(succession_repo, _registered_tip(succession_repo).snapshot_id)
    strata = dict(plan.composition_attestation.member_strata)
    impostor = next(
        stable_id
        for stable_id, stratum in sorted(strata.items())
        if stable_id not in tip_members and stratum != CARRIED_TIP_STRATUM
    )
    strata[impostor] = CARRIED_TIP_STRATUM
    forged = build_composition_attestation(
        campaign_id=plan.campaign_id,
        size=plan.size,
        seed=plan.seed,
        member_strata=strata,
        snapshot_id=plan.composition_attestation.snapshot_id,
        snapshot_sha256=plan.composition_attestation.snapshot_sha256,
        pd7_disposition=plan.composition_attestation.pd7_disposition,
    )
    before = sorted(path.name for path in (_records_root(succession_repo) / "intake").iterdir())
    with pytest.raises(RungStagingError, match="lineage tip") as excinfo:
        _stage(
            succession_repo,
            plan,
            label="step-200-forged-carry",
            award_authorities=_award_authorities_for(succession_repo, plan.member_ids()),
            attestation=forged,
        )
    assert impostor in str(excinfo.value)
    assert (
        sorted(path.name for path in (_records_root(succession_repo) / "intake").iterdir())
        == before
    )


def test_register_door_walks_the_carry_check_too(succession_repo: Path) -> None:
    """The carry check binds the EVIDENCE, not the entry point.

    2026-08-16 red-team break 1, applied to the new check before it can become
    the next instance: a gate that lives only on ``stage_rung`` is a gate the
    ``register`` door adopts its way around. Here the adopted snapshot is real
    and valid; only the carry claim is forged, and the sibling door refuses it
    exactly as ``stage_rung`` does.
    """

    from menagerie.crawler.rung_staging import (
        _write_subset_snapshot,
        register_existing_snapshot,
        resolve_verified_root,
    )

    records_root = _records_root(succession_repo)
    plan = _compose(succession_repo)
    tip_members = _members_of(succession_repo, _registered_tip(succession_repo).snapshot_id)
    strata = dict(plan.composition_attestation.member_strata)
    impostor = next(
        stable_id
        for stable_id, stratum in sorted(strata.items())
        if stable_id not in tip_members and stratum != CARRIED_TIP_STRATUM
    )
    strata[impostor] = CARRIED_TIP_STRATUM
    forged = build_composition_attestation(
        campaign_id=plan.campaign_id,
        size=plan.size,
        seed=plan.seed,
        member_strata=strata,
        snapshot_id=plan.composition_attestation.snapshot_id,
        snapshot_sha256=plan.composition_attestation.snapshot_sha256,
        pd7_disposition=plan.composition_attestation.pd7_disposition,
    )
    _root_binding, root_snapshot = resolve_verified_root(records_root, plan.campaign_id)
    adopted, _created = _write_subset_snapshot(
        root_snapshot.root, plan.member_ids(), records_root / "intake"
    )
    with pytest.raises(RungStagingError, match="lineage tip") as excinfo:
        register_existing_snapshot(
            records_root,
            adopted.root,
            campaign_id=plan.campaign_id,
            label="step-200-readopt",
            registered_by="pytest",
            created_at="2026-08-19T00:00:00Z",
            historical_only=False,
            award_authorities=_award_authorities_for(succession_repo, plan.member_ids()),
            composition_attestation=forged,
        )
    assert impostor in str(excinfo.value)
    assert not [
        row for row in load_rung_registrations(records_root) if row.label == "step-200-readopt"
    ]


def test_first_rung_carries_nothing_and_may_claim_nothing(succession_repo: Path) -> None:
    """With no registered rung there is no tip, so nothing may claim one.

    ``c3-classics`` shares the records root and has zero registered rungs -- the
    root exemption case. Carrying is not "always add seven members"; it is
    "carry exactly what the registry tip holds", which here is nothing.
    """

    campaign = "c3-classics"
    shares = {name: 0.0 for name in _TEST_SHARES}
    shares["ladder-bulk"] = 1.0
    plan = _compose(succession_repo, campaign_id=campaign, size=1, shares=shares)
    block = plan.to_report()["carried_tip"]
    assert block["provenance"] == "no-registered-lineage-tip"
    assert block["snapshot_id"] is None
    assert block["member_count"] == 0
    assert block["member_ids"] == [] and block["carried_forward"] == []
    assert CARRIED_TIP_STRATUM not in {member.stratum for member in plan.members}

    forged = build_composition_attestation(
        campaign_id=campaign,
        size=plan.size,
        seed=plan.seed,
        member_strata={plan.member_ids()[0]: CARRIED_TIP_STRATUM},
        snapshot_id=plan.composition_attestation.snapshot_id,
        snapshot_sha256=plan.composition_attestation.snapshot_sha256,
    )
    # The root exemption: c3-classics has no registered lineage and no live
    # licensing close, so this first rung is NOT a successor and declares no
    # class (the W16/S-2 declaration is required only under a live close).
    with pytest.raises(RungStagingError, match="no registered lineage tip"):
        _stage(succession_repo, plan, label="c3-first", attestation=forged, rung_class=None)

    # The honest first rung stages: carrying nothing is the correct behaviour,
    # not a missing feature.
    staged = _stage(succession_repo, plan, label="c3-first", rung_class=None)
    assert staged.registration.member_count == 1


def test_composition_adding_nothing_beyond_the_tip_refuses(succession_repo: Path) -> None:
    """A rung that is only its predecessor is not a rung.

    Strict widening is the door's rule; the composer must not hand it a rung it
    is bound to reject. Refusing here names the actual problem (the shares bought
    nothing new) instead of a member-count mismatch two commands later.
    """

    from menagerie.crawler.rung_stratification import StratificationError

    shares = {name: 0.0 for name in _TEST_SHARES}
    shares["cuda-deferred"] = 1.0
    with pytest.raises(StratificationError, match="beyond the carried lineage tip"):
        _compose(succession_repo, size=1, shares=shares)
