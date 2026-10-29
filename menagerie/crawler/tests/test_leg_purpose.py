"""Shape 4: the licensing_close leg purpose and its enforced capability forfeit.

The tri-lab ladder ruling's six-point safety contract (Sol recon
ACCEPT-SHAPE-4), each point pinned:

1. ``leg_purpose`` has a closed vocabulary, is content-addressed in the
   durable ``leg_open`` row before the first mutation, and is immutable per
   generation (a resume declaring a different or absent purpose refuses).
2. Only the CONSUMER computes the effective certificate kind: a
   ``successor_rung`` snapshot with no (or any other) purpose keeps demanding
   ``rung_launch`` — the residual RED — and only ``licensing_close`` derives
   ``canary_admission`` there.
3. A licensing-close generation seals an EMPTY WS-1 plan; the replayed union
   must be empty on every invocation and again at close.
4. Every paid admission path refuses typed under the purpose.
5. Only orphan-citation adoption may mutate model rows pre-plan, and the
   adoption rules are not widened (the close-side family check IS the
   pre-append law read post-append).
6. Close-time validation refuses any non-empty admitted union, paid-session
   delta, positive disposition change, or out-of-family mutation.

Plus: no bare ``--leg-class`` laundering flag exists anywhere; the
close-drift probe names the recurrence class; and the door-B catalog row's
retroactive-reclass WARN fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.leg_generation import (
    LegGenerationError,
    open_leg_generation,
    record_leg_plan,
    validate_leg_open_row,
)
from menagerie.crawler.leg_purpose import (
    LEG_PURPOSE_LICENSING_CLOSE,
    LEG_PURPOSES,
    LegPurposeError,
    licensing_close_forfeit_refusals,
    validate_leg_purpose,
)

CAMPAIGN = "c1-mech"
SNAPSHOT_ID = "intake-" + "ab" * 10
SNAPSHOT_SHA = "sha256:" + "ab" * 32
HOLD_QUEUE = "crawler-human-review"


def _open_purpose_leg(
    records_root: Path,
    *,
    run_id: str = "purpose-run",
    leg_purpose: Optional[str] = LEG_PURPOSE_LICENSING_CLOSE,
):
    """Open (or continue) the fixture generation under one declared purpose."""

    return open_leg_generation(
        records_root,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=SNAPSHOT_ID,
        intake_snapshot_sha256=SNAPSHOT_SHA,
        run_id=run_id,
        opened_at="2026-08-31T00:00:00Z",
        leg_purpose=leg_purpose,
    )


def _hold_row(stable_id: str, parent: Optional[dict] = None) -> dict[str, Any]:
    """Build one review-held failure terminal in the containment shape."""

    body: dict[str, Any] = {
        "stable_id": stable_id,
        "parent_revision": None if parent is None else parent["record_revision"],
        "status": {
            "kind": "failed",
            "code": "failed:import",
            "human_review": {
                "required": True,
                "reason": None,
                "queue": HOLD_QUEUE,
                "requested_at": "2026-08-31T04:28:48Z",
            },
        },
        "source_resolution": {"rung": "R2_VENDOR", "attempted_rungs": []},
        "provenance": {"producer_run_id": "crawler-run", "machine_id": "old-host"},
    }
    if parent is not None:
        body["status"]["supersedes_revision"] = parent["record_revision"]
    body["record_seq"] = 1 if parent is None else int(parent.get("record_seq") or 1) + 1
    body["record_revision"] = stable_hash(body)
    return body


def _adoption_row(orphan: dict[str, Any], *, run_id: str) -> dict[str, Any]:
    """Re-append one orphan hold verbatim modulo the adoption re-citation set."""

    body = json.loads(json.dumps(orphan))
    body.pop("record_revision")
    body["parent_revision"] = orphan["record_revision"]
    body["status"]["supersedes_revision"] = orphan["record_revision"]
    body["provenance"] = {"producer_run_id": run_id, "machine_id": "purpose-host"}
    body["record_seq"] = int(orphan.get("record_seq") or 1) + 1
    body["record_revision"] = stable_hash(body)
    return body


def _write_shard(records_root: Path, rows: list[dict[str, Any]]) -> Path:
    """Write the sharded models ledger below one records root."""

    path = records_root / "models" / "current-shard.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    return path


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to one JSONL ledger."""

    with path.open("ab") as handle:
        handle.write(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


@pytest.fixture()
def purpose_leg(tmp_path: Path):
    """A licensing_close generation opened over one below-floor orphan hold."""

    records_root = tmp_path / "records"
    base = _hold_row("m1")
    base["status"] = {"kind": "failed", "code": "failed:import"}
    base["record_revision"] = stable_hash(
        {key: value for key, value in base.items() if key != "record_revision"}
    )
    orphan = _hold_row("m1", parent=base)
    shard = _write_shard(records_root, [base, orphan])
    generation = _open_purpose_leg(records_root)
    record_leg_plan(
        records_root,
        campaign_id=CAMPAIGN,
        generation_id=generation.generation_id,
        run_id="purpose-run",
        admitted_ids=(),
        plan_payload={"leg_purpose": LEG_PURPOSE_LICENSING_CLOSE},
        created_at="2026-08-31T00:00:01Z",
    )
    return SimpleNamespace(
        records_root=records_root, shard=shard, orphan=orphan, generation=generation
    )


# -- Point 1: closed vocabulary, content-addressed, immutable per generation --


@pytest.mark.smoke
def test_leg_purpose_vocabulary_is_closed() -> None:
    """One purpose value; anything else refuses typed."""

    assert LEG_PURPOSES == (LEG_PURPOSE_LICENSING_CLOSE,)
    assert validate_leg_purpose("licensing_close") == "licensing_close"
    for junk in ("canary", "", None, "LICENSING_CLOSE", "production"):
        with pytest.raises(LegPurposeError, match="closed vocabulary"):
            validate_leg_purpose(junk)


@pytest.mark.smoke
def test_purpose_is_stamped_content_addressed_and_immutable(tmp_path: Path) -> None:
    """The open row carries the purpose inside its content address; resumes
    replaying a different or absent purpose refuse (Sol point 1)."""

    records_root = tmp_path / "records"
    generation = _open_purpose_leg(records_root)
    assert generation.leg_purpose == LEG_PURPOSE_LICENSING_CLOSE
    from menagerie.crawler.leg_generation import load_leg_open_rows

    (open_row,) = load_leg_open_rows(records_root)
    assert open_row["leg_purpose"] == LEG_PURPOSE_LICENSING_CLOSE
    # Content-addressed: relabeling the persisted purpose breaks the digest.
    forged = dict(open_row)
    forged.pop("chain", None)
    del forged["leg_purpose"]
    with pytest.raises(LegGenerationError, match="digest"):
        validate_leg_open_row(forged)
    # Same purpose continues the SAME epoch (kill/resume law unchanged).
    continued = _open_purpose_leg(records_root, run_id="purpose-run-2")
    assert continued.generation_id == generation.generation_id
    assert continued.leg_purpose == LEG_PURPOSE_LICENSING_CLOSE
    # An absent purpose can never reclassify the open generation...
    with pytest.raises(LegGenerationError, match="immutable per generation"):
        _open_purpose_leg(records_root, run_id="launder-run", leg_purpose=None)
    # ...and a declared purpose can never reclassify an open ORDINARY one
    # (the mis-declared production resume, refused at the door).
    other_root = tmp_path / "records-ordinary"
    ordinary = _open_purpose_leg(other_root, run_id="prod-run", leg_purpose=None)
    assert ordinary.leg_purpose is None
    with pytest.raises(LegGenerationError, match="immutable per generation"):
        _open_purpose_leg(other_root, run_id="launder-run")


@pytest.mark.smoke
def test_junk_purpose_refuses_at_open_and_at_row_validation(tmp_path: Path) -> None:
    """The vocabulary is enforced at the door and on every replay."""

    with pytest.raises(LegPurposeError, match="closed vocabulary"):
        _open_purpose_leg(tmp_path / "records", leg_purpose="production")
    records_root = tmp_path / "records-forged"
    generation = _open_purpose_leg(records_root)
    from menagerie.crawler.identity import stamp_row_digest
    from menagerie.crawler.leg_generation import load_leg_open_rows

    (open_row,) = load_leg_open_rows(records_root)
    forged = {
        key: value for key, value in open_row.items() if key not in ("chain", "row_sha256")
    }
    forged["leg_purpose"] = "production"
    forged["generation_id"] = generation.generation_id
    with pytest.raises(LegGenerationError, match="unknown leg purpose"):
        validate_leg_open_row(stamp_row_digest(forged))


@pytest.mark.smoke
def test_no_bare_leg_class_flag_exists_anywhere() -> None:
    """A bare --leg-class flag is the laundering channel and must refuse.

    Neither the crawler CLI, the supervisor, nor the launch preflight accepts
    ``--leg-class``; the only declaration is ``--leg-purpose`` over the closed
    vocabulary, and argparse refuses anything outside it.
    """

    from menagerie.crawler.cli import build_parser as build_cli_parser
    from menagerie.crawler.supervisor import build_parser as build_supervisor_parser
    from menagerie.crawler.tools.launch_preflight import main as preflight_main

    cli_parser = build_cli_parser()
    with pytest.raises(SystemExit):
        cli_parser.parse_args(["resume", "--leg-class", "licensing_close"])
    with pytest.raises(SystemExit):
        cli_parser.parse_args(["resume", "--intake", "x", "--leg-purpose", "canary"])
    with pytest.raises(SystemExit):
        build_supervisor_parser().parse_args(
            ["run", "--campaign-id", "c1-mech", "--repo-root", ".",
             "--campaign-config", "x", "--leg-class", "licensing_close"]
        )
    with pytest.raises(SystemExit):
        preflight_main(["--leg-class", "licensing_close"])


@pytest.mark.smoke
def test_config_refuses_purpose_composed_with_any_selection_mode() -> None:
    """A mis-declared production launch refuses at the config door."""

    from menagerie.crawler.driver_contracts import DriverConfig

    assert DriverConfig(campaign_id=CAMPAIGN, leg_purpose="licensing_close").leg_purpose
    with pytest.raises(ValueError, match="closed vocabulary"):
        DriverConfig(campaign_id=CAMPAIGN, leg_purpose="production")
    for kwargs in (
        {"only_unsettled": True},
        {"only_status": "deferred:needs-cuda", "target": "linux-x86_64-cuda"},
        {"redrive_targets": ("m1",), "redrive_reason": "test"},
        {"gate_catchup_targets": ("m1",)},
    ):
        with pytest.raises(ValueError, match="forfeits all admission"):
            DriverConfig(campaign_id=CAMPAIGN, leg_purpose="licensing_close", **kwargs)


# -- Points 3 + 6: the empty plan and the close-time forfeit ------------------


@pytest.mark.smoke
def test_clean_licensing_close_window_passes_the_forfeit(purpose_leg) -> None:
    """Empty plan + adoption-only window + zero paid rows: no refusals."""

    adoption = _adoption_row(purpose_leg.orphan, run_id="purpose-run")
    _append_rows(purpose_leg.shard, [adoption])
    assert (
        licensing_close_forfeit_refusals(
            purpose_leg.records_root,
            generation_id=purpose_leg.generation.generation_id,
            baselines=purpose_leg.generation.baselines,
        )
        == ()
    )


@pytest.mark.smoke
def test_forfeit_refuses_a_non_empty_admitted_union(purpose_leg) -> None:
    """Sol point 3: a resumed non-empty plan row can never launder through."""

    record_leg_plan(
        purpose_leg.records_root,
        campaign_id=CAMPAIGN,
        generation_id=purpose_leg.generation.generation_id,
        run_id="purpose-run",
        admitted_ids=("m1",),
        plan_payload={"laundered": True},
        created_at="2026-08-31T00:00:02Z",
    )
    refusals = licensing_close_forfeit_refusals(
        purpose_leg.records_root,
        generation_id=purpose_leg.generation.generation_id,
        baselines=purpose_leg.generation.baselines,
    )
    assert any("NON-EMPTY admitted union" in refusal for refusal in refusals)


@pytest.mark.smoke
def test_forfeit_refuses_a_paid_session_bound_to_the_generation(purpose_leg) -> None:
    """Zero paid spend, replay-derived from the money ledger's own rows."""

    foreign = [{"generation_id": "some-other-generation", "idempotency_key": "k0"}]
    assert (
        licensing_close_forfeit_refusals(
            purpose_leg.records_root,
            generation_id=purpose_leg.generation.generation_id,
            baselines=purpose_leg.generation.baselines,
            paid_rows=foreign,
        )
        == ()
    )
    mine = [
        {"generation_id": purpose_leg.generation.generation_id, "idempotency_key": "k1"}
    ]
    refusals = licensing_close_forfeit_refusals(
        purpose_leg.records_root,
        generation_id=purpose_leg.generation.generation_id,
        baselines=purpose_leg.generation.baselines,
        paid_rows=mine,
    )
    assert any("paid session reservation" in refusal for refusal in refusals)


@pytest.mark.smoke
def test_forfeit_refuses_a_positive_disposition_change(purpose_leg) -> None:
    """A window row moving status is refused BY NAME, never as generic drift."""

    promoted = _adoption_row(purpose_leg.orphan, run_id="purpose-run")
    promoted["status"]["kind"] = "runs"
    promoted["status"]["human_review"]["required"] = True
    promoted["record_revision"] = stable_hash(
        {key: value for key, value in promoted.items() if key != "record_revision"}
    )
    _append_rows(purpose_leg.shard, [promoted])
    refusals = licensing_close_forfeit_refusals(
        purpose_leg.records_root,
        generation_id=purpose_leg.generation.generation_id,
        baselines=purpose_leg.generation.baselines,
    )
    assert any("never a positive disposition" in refusal for refusal in refusals)


@pytest.mark.smoke
def test_forfeit_refuses_content_beyond_the_adoption_recitation_set(purpose_leg) -> None:
    """Point 5's non-widening arm: the post-append family rule is the
    pre-append adoption law — content drift refuses byte-precisely."""

    drifted = _adoption_row(purpose_leg.orphan, run_id="purpose-run")
    drifted["source_resolution"] = {"rung": "R1_LIBRARY", "attempted_rungs": []}
    drifted["record_revision"] = stable_hash(
        {key: value for key, value in drifted.items() if key != "record_revision"}
    )
    _append_rows(purpose_leg.shard, [drifted])
    refusals = licensing_close_forfeit_refusals(
        purpose_leg.records_root,
        generation_id=purpose_leg.generation.generation_id,
        baselines=purpose_leg.generation.baselines,
    )
    assert any("adoption re-citation set" in refusal for refusal in refusals)


@pytest.mark.smoke
def test_forfeit_refuses_an_unjoined_citing_run_and_in_window_parents(purpose_leg) -> None:
    """Position is not provenance, and adoption needs sealed pre-window history."""

    unjoined = _adoption_row(purpose_leg.orphan, run_id="never-joined-run")
    _append_rows(purpose_leg.shard, [unjoined])
    refusals = licensing_close_forfeit_refusals(
        purpose_leg.records_root,
        generation_id=purpose_leg.generation.generation_id,
        baselines=purpose_leg.generation.baselines,
    )
    assert any("never joined this leg generation" in refusal for refusal in refusals)
    # A chain of two in-window rows: the second's parent is IN-window.
    stacked = _adoption_row(unjoined, run_id="purpose-run")
    _append_rows(purpose_leg.shard, [stacked])
    refusals = licensing_close_forfeit_refusals(
        purpose_leg.records_root,
        generation_id=purpose_leg.generation.generation_id,
        baselines=purpose_leg.generation.baselines,
    )
    assert any("strictly below the window floor" in refusal for refusal in refusals)


@pytest.mark.smoke
def test_forfeit_refuses_every_out_of_family_source_mutation(purpose_leg) -> None:
    """Attempts/gates/valves may not move; parks admit release voids ONLY."""

    records_root = purpose_leg.records_root
    gates = records_root / "gates" / "current-shard.jsonl"
    gates.parent.mkdir(parents=True, exist_ok=True)
    with gates.open("ab") as handle:
        handle.write(b'{"gate": "smuggled"}\n')
    from menagerie.crawler.grant_parks import grant_park_path

    park_path = grant_park_path(records_root)
    park_path.parent.mkdir(parents=True, exist_ok=True)
    with park_path.open("ab") as handle:
        handle.write(
            canonical_json_bytes({"stable_id": "m9", "disposition": "selection-exclusion"})
            + b"\n"
        )
    refusals = licensing_close_forfeit_refusals(
        records_root,
        generation_id=purpose_leg.generation.generation_id,
        baselines=purpose_leg.generation.baselines,
    )
    assert any("gates.shard moved" in refusal for refusal in refusals)
    assert any("may not mint new park decisions" in refusal for refusal in refusals)
    # A bare void-SHAPED row is NOT the machine release-on-record family: the
    # close replays typed schema, actor, target, and released-ness (Sol point
    # 6), so a hand-built four-field object refuses instead of passing.
    park_path.write_bytes(
        canonical_json_bytes(
            {
                "void_of": "sha256:" + "33" * 32,
                "reason": "release-on-record",
                "voided_by": "close",
                "created_at": "2026-08-31T00:00:03Z",
            }
        )
        + b"\n"
    )
    refusals = licensing_close_forfeit_refusals(
        records_root,
        generation_id=purpose_leg.generation.generation_id,
        baselines=purpose_leg.generation.baselines,
    )
    assert any(
        "park.grant" in refusal and "not a typed park-void row" in refusal
        for refusal in refusals
    )


@pytest.mark.smoke
def test_forfeit_refuses_a_REAL_paid_reservation_row(tmp_path: Path) -> None:
    """Point 6's paid arm, pinned against the shape the ledger actually emits."""

    records_root = tmp_path / "records"
    shard = records_root / "models" / "current-shard.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "stable_id": "m1",
        "parent_revision": None,
        "status": {"kind": "failed", "code": "failed:import"},
        "source_resolution": {"rung": "R2_VENDOR", "attempted_rungs": []},
        "provenance": {"producer_run_id": "crawler-run", "machine_id": "old-host"},
    }
    base["record_revision"] = stable_hash(base)
    shard.write_bytes(canonical_json_bytes(base) + b"\n")

    generation = _open_purpose_leg(records_root)
    record_leg_plan(
        records_root,
        campaign_id=CAMPAIGN,
        generation_id=generation.generation_id,
        run_id="purpose-run",
        admitted_ids=(),
        plan_payload={"leg_purpose": LEG_PURPOSE_LICENSING_CLOSE},
        created_at="2026-08-31T00:00:01Z",
    )

    from menagerie.crawler.paid_session_accounting import BudgetPolicy
    from menagerie.crawler.paid_session_lifecycle import PaidSessionAuthority

    authority = PaidSessionAuthority(
        tmp_path / "paid-sessions.jsonl",
        runtime_root=tmp_path / "paid-runtime",
        clock=lambda: "2026-08-31T00:00:02Z",
    )
    authority.reserve(
        scope={
            "campaign_id": CAMPAIGN,
            "generation_id": generation.generation_id,
            "lane": "checker",
            "dispatch_kind": "gate-request",
            "work_generations": (("m1", "work-1"),),
            "machine_id": "probe-host",
        },
        policy=BudgetPolicy(
            quota_sessions=1,
            quota_wall_seconds=1000.0,
            runaway_sessions=1,
            runaway_wall_seconds=1000.0,
            reservation_wall_seconds=1000.0,
        ),
    )

    refusals = licensing_close_forfeit_refusals(
        records_root,
        generation_id=generation.generation_id,
        baselines=generation.baselines,
        paid_rows=authority.rows(),
    )
    assert any("paid session reservation" in refusal for refusal in refusals)


# -- Point 6: the park slots admit ONLY the machine release-on-record family --


def _purpose_leg_root(tmp_path: Path) -> SimpleNamespace:
    """One records root holding the two-row m1 history, no generation yet."""

    records_root = tmp_path / "records"
    base = _hold_row("m1")
    base["status"] = {"kind": "failed", "code": "failed:import"}
    base["record_revision"] = stable_hash(
        {key: value for key, value in base.items() if key != "record_revision"}
    )
    orphan = _hold_row("m1", parent=base)
    shard = _write_shard(records_root, [base, orphan])
    return SimpleNamespace(records_root=records_root, shard=shard, base=base, orphan=orphan)


def _open_and_plan(records_root: Path):
    """Open the licensing_close generation and seal its empty plan."""

    generation = _open_purpose_leg(records_root)
    record_leg_plan(
        records_root,
        campaign_id=CAMPAIGN,
        generation_id=generation.generation_id,
        run_id="purpose-run",
        admitted_ids=(),
        plan_payload={"leg_purpose": LEG_PURPOSE_LICENSING_CLOSE},
        created_at="2026-08-31T00:00:01Z",
    )
    return generation


@pytest.mark.smoke
def test_forfeit_refuses_an_operator_void_of_a_live_exclusion_park(tmp_path: Path) -> None:
    """Sol's counterexample: a valid operator void of a pre-window
    ``review-held:selection-exclusion`` park lifts a live negative decision —
    an out-of-family positive capability change, never the machine-derived
    release-on-record disclosure — and the close forfeit must refuse it."""

    from menagerie.crawler.grant_parks import (
        append_grant_park_row,
        append_grant_park_void,
        build_grant_park_void,
        build_selection_exclusion_park_row,
        grant_park_path,
    )

    leg = _purpose_leg_root(tmp_path)
    park_path = grant_park_path(leg.records_root)
    exclusion = append_grant_park_row(
        park_path,
        build_selection_exclusion_park_row(
            stable_id="m1",
            name="model-one",
            campaign_id=CAMPAIGN,
            machine_id="ops-host",
            created_at="2026-08-30T00:00:00Z",
            held_revision=leg.orphan["record_revision"],
            held_status_code="failed:import",
            decision="PD-7-slop-family-routes",
            reason="frozen decision excludes this family from selection",
        ),
    )
    generation = _open_and_plan(leg.records_root)
    append_grant_park_void(
        park_path,
        build_grant_park_void(
            void_of=exclusion["row_sha256"],
            reason="operator chooses to lift the exclusion during the weak leg",
            voided_by="operator:forged-positive-release",
            created_at="2026-08-31T00:00:02Z",
        ),
    )
    refusals = licensing_close_forfeit_refusals(
        leg.records_root,
        generation_id=generation.generation_id,
        baselines=generation.baselines,
    )
    assert any("machine release actor" in refusal for refusal in refusals)


@pytest.mark.smoke
def test_forfeit_refuses_forged_machine_voids_out_of_family(tmp_path: Path) -> None:
    """Even the machine actor cannot launder a void whose replay fails.

    Four forgeries, each refused by name: voiding a live selection-exclusion
    park (outside the machine-voidable disposition), naming a target no park
    row carries, voiding a park whose release condition the ledger head does
    not satisfy, and carrying a hand-typed reason instead of the derived
    release-on-record disclosure."""

    from menagerie.crawler.grant_parks import (
        append_grant_park_row,
        build_grant_park_row,
        build_grant_park_void,
        build_selection_exclusion_park_row,
        grant_park_path,
    )
    from menagerie.crawler.park_release import PARK_RELEASE_VOID_ACTOR, _void_reason

    leg = _purpose_leg_root(tmp_path)
    park_path = grant_park_path(leg.records_root)
    exclusion = append_grant_park_row(
        park_path,
        build_selection_exclusion_park_row(
            stable_id="m1",
            name="model-one",
            campaign_id=CAMPAIGN,
            machine_id="ops-host",
            created_at="2026-08-30T00:00:00Z",
            held_revision=leg.orphan["record_revision"],
            held_status_code="failed:import",
            decision="PD-7-slop-family-routes",
            reason="frozen decision excludes this family from selection",
        ),
    )
    unreleased = append_grant_park_row(
        park_path,
        build_grant_park_row(
            stable_id="m1",
            work_id="work-live",
            name="model-one",
            campaign_id=CAMPAIGN,
            run_id="crawler-run",
            machine_id="old-host",
            created_at="2026-08-30T00:00:01Z",
            blocked_parent_revision=leg.orphan["record_revision"],
            blocked_stage="stage2-author",
            pending_model={"stable_id": "m1", "status": {"code": "failed:import"}},
        ),
    )
    released = append_grant_park_row(
        park_path,
        build_grant_park_row(
            stable_id="m1",
            work_id="work-consumed",
            name="model-one",
            campaign_id=CAMPAIGN,
            run_id="crawler-run",
            machine_id="old-host",
            created_at="2026-08-30T00:00:02Z",
            blocked_parent_revision=leg.base["record_revision"],
            blocked_stage="stage2-author",
            pending_model={"stable_id": "m1", "status": {"code": "failed:import"}},
        ),
    )
    generation = _open_and_plan(leg.records_root)
    forged = [
        build_grant_park_void(
            void_of=exclusion["row_sha256"],
            reason=_void_reason(leg.orphan["record_revision"]),
            voided_by=PARK_RELEASE_VOID_ACTOR,
            created_at="2026-08-31T00:00:02Z",
        ),
        build_grant_park_void(
            void_of="sha256:" + "44" * 32,
            reason=_void_reason(leg.orphan["record_revision"]),
            voided_by=PARK_RELEASE_VOID_ACTOR,
            created_at="2026-08-31T00:00:03Z",
        ),
        build_grant_park_void(
            void_of=unreleased["row_sha256"],
            reason=_void_reason(leg.base["record_revision"]),
            voided_by=PARK_RELEASE_VOID_ACTOR,
            created_at="2026-08-31T00:00:04Z",
        ),
        build_grant_park_void(
            void_of=released["row_sha256"],
            reason="the head moved on, trust me",
            voided_by=PARK_RELEASE_VOID_ACTOR,
            created_at="2026-08-31T00:00:05Z",
        ),
    ]
    # Written raw, past the append door: the close-time replay is exactly the
    # backstop that must not trust that every write used the appender.
    _append_rows(park_path, forged)
    refusals = licensing_close_forfeit_refusals(
        leg.records_root,
        generation_id=generation.generation_id,
        baselines=generation.baselines,
    )
    assert any("machine release lane voids only" in refusal for refusal in refusals)
    assert any("names no park row" in refusal for refusal in refusals)
    assert any("ledger head does not satisfy" in refusal for refusal in refusals)
    assert any("derived release-on-record reason" in refusal for refusal in refusals)


@pytest.mark.smoke
def test_machine_release_on_record_void_passes_the_forfeit(tmp_path: Path) -> None:
    """The genuine void-on-record disclosure (the production writer itself,
    :func:`menagerie.crawler.park_release.void_released_parks`) stays in
    family: zero refusals."""

    from menagerie.crawler.grant_parks import (
        append_grant_park_row,
        build_grant_park_row,
        grant_park_path,
    )
    from menagerie.crawler.park_release import PARK_RELEASE_VOID_ACTOR, void_released_parks

    leg = _purpose_leg_root(tmp_path)
    park_path = grant_park_path(leg.records_root)
    consumed = append_grant_park_row(
        park_path,
        build_grant_park_row(
            stable_id="m1",
            work_id="work-consumed",
            name="model-one",
            campaign_id=CAMPAIGN,
            run_id="crawler-run",
            machine_id="old-host",
            created_at="2026-08-30T00:00:00Z",
            blocked_parent_revision=leg.base["record_revision"],
            blocked_stage="stage2-author",
            pending_model={"stable_id": "m1", "status": {"code": "failed:import"}},
        ),
    )
    generation = _open_and_plan(leg.records_root)
    (void,) = void_released_parks(
        leg.records_root,
        {"m1": [consumed]},
        {"m1": leg.orphan},
        created_at="2026-08-31T00:00:02Z",
    )
    assert void["voided_by"] == PARK_RELEASE_VOID_ACTOR
    assert (
        licensing_close_forfeit_refusals(
            leg.records_root,
            generation_id=generation.generation_id,
            baselines=generation.baselines,
        )
        == ()
    )


# -- Point 4: paid paths refuse typed -----------------------------------------


@pytest.mark.smoke
def test_paid_admission_chokepoints_refuse_under_the_purpose() -> None:
    """Both money chokepoints refuse typed before any reservation row exists."""

    from menagerie.crawler.driver import CrawlerDriver

    stub = SimpleNamespace(
        config=SimpleNamespace(leg_purpose=LEG_PURPOSE_LICENSING_CLOSE),
        _leg_generation=None,
    )
    with pytest.raises(LegPurposeError, match="forfeits all paid admission"):
        CrawlerDriver._refuse_paid_admission_under_leg_purpose(stub, "checker-admission")
    # The generation's stamped purpose refuses even if config were laundered.
    stub = SimpleNamespace(
        config=SimpleNamespace(leg_purpose=None),
        _leg_generation=SimpleNamespace(leg_purpose=LEG_PURPOSE_LICENSING_CLOSE),
    )
    with pytest.raises(LegPurposeError, match="forfeits all paid admission"):
        CrawlerDriver._refuse_paid_admission_under_leg_purpose(stub, "author-admission")
    # Ordinary legs pass through untouched.
    stub = SimpleNamespace(
        config=SimpleNamespace(leg_purpose=None),
        _leg_generation=SimpleNamespace(leg_purpose=None),
    )
    CrawlerDriver._refuse_paid_admission_under_leg_purpose(stub, "checker-admission")


# -- The close row repeats the purpose ----------------------------------------


@pytest.mark.smoke
def test_close_row_carries_and_validates_the_purpose(tmp_path: Path) -> None:
    """The close repeats the purpose; junk and non-leg kinds refuse."""

    from menagerie.crawler.close_object import (
        CloseObjectError,
        close_object_path,
        load_close_objects,
        validate_close_object,
    )
    from menagerie.crawler.identity import stamp_row_digest
    from menagerie.crawler.tests.close_object_support import record_fixture_close

    records_root = tmp_path / "records"
    close = record_fixture_close(
        records_root,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=SNAPSHOT_ID,
        intake_snapshot_sha256=SNAPSHOT_SHA,
        member_ids=("m1",),
        seed_member_heads=True,
        completion_status="terminal-partition-complete",
    )
    assert "leg_purpose" not in close  # ordinary legs stay purpose-free
    base = {
        key: value
        for key, value in load_close_objects([close_object_path(records_root)])[0].items()
        if key not in ("chain", "close_sha256")
    }
    stamped = stamp_row_digest(
        {**base, "leg_purpose": LEG_PURPOSE_LICENSING_CLOSE}, field="close_sha256"
    )
    validated = validate_close_object(stamped)
    assert validated["leg_purpose"] == LEG_PURPOSE_LICENSING_CLOSE
    with pytest.raises(CloseObjectError, match="unknown leg purpose"):
        validate_close_object(
            stamp_row_digest({**base, "leg_purpose": "production"}, field="close_sha256")
        )
    with pytest.raises(CloseObjectError, match="only ride a leg-kind close"):
        validate_close_object(
            stamp_row_digest(
                {
                    **base,
                    "close_kind": "backfill",
                    "leg_open_row_sha256": None,
                    "leg_purpose": LEG_PURPOSE_LICENSING_CLOSE,
                },
                field="close_sha256",
            )
        )


# -- Continuation contract -----------------------------------------------------


@pytest.mark.smoke
def test_purpose_rides_the_continuation_contract_versioned_monotone() -> None:
    """The wake callback replays the declaration; None-configs keep their
    exact pre-purpose continuation identity (era safety)."""

    from menagerie.crawler.continuation_manifest import render_continuation_flags
    from menagerie.crawler.driver_contracts import DriverConfig
    from menagerie.crawler.work_selection import selection_continuation_identity

    plain = DriverConfig(campaign_id=CAMPAIGN)
    declared = DriverConfig(campaign_id=CAMPAIGN, leg_purpose=LEG_PURPOSE_LICENSING_CLOSE)
    plain_flags = render_continuation_flags(plain)
    declared_flags = render_continuation_flags(declared)
    assert "--leg-purpose" not in plain_flags
    index = declared_flags.index("--leg-purpose")
    assert declared_flags[index + 1] == LEG_PURPOSE_LICENSING_CLOSE
    # Versioned-monotone identity: a config with no purpose attribute at all
    # (the pre-purpose era) hashes identically to purpose=None.
    class _EraConfig:
        pass

    era = _EraConfig()
    for field in (
        "scheduler",
        "target",
        "phase",
        "only_status",
        "only_unsettled",
        "ack_stale_excluded",
        "acknowledge_award_holds",
        "award_holds_digest",
        "stale_excluded_digest",
        "author_concurrency",
        "checker_concurrency",
        "redrive_targets",
        "gate_catchup_targets",
    ):
        setattr(era, field, getattr(plain, field))
    assert selection_continuation_identity(era) == selection_continuation_identity(plain)
    assert selection_continuation_identity(declared) != selection_continuation_identity(plain)


# -- The recurrence tripwire ---------------------------------------------------


@pytest.mark.smoke
def test_close_drift_classifier_names_the_recurrence_classes(tmp_path: Path) -> None:
    """identical -> none; hold appends -> review-hold-append; anything else
    (non-models append, truncation, non-hold append) -> out-of-band-drift."""

    from menagerie.crawler.close_drift import (
        DELTA_CLASS_NONE,
        DELTA_CLASS_OUT_OF_BAND,
        DELTA_CLASS_REVIEW_HOLD_APPEND,
        close_source_drift,
        drift_delta_class,
    )
    from menagerie.crawler.tests.close_object_support import record_fixture_close

    records_root = tmp_path / "records"
    close = record_fixture_close(
        records_root,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=SNAPSHOT_ID,
        intake_snapshot_sha256=SNAPSHOT_SHA,
        member_ids=("m1",),
        seed_member_heads=True,
        completion_status="terminal-partition-complete",
    )
    assert drift_delta_class(close_source_drift(records_root, close)) == DELTA_CLASS_NONE
    # A hold-shaped models append names the adoption-remediable class.
    shard = records_root / "models" / "current-shard.jsonl"
    head = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
    _append_rows(shard, [_hold_row("m1", parent=head)])
    assert (
        drift_delta_class(close_source_drift(records_root, close))
        == DELTA_CLASS_REVIEW_HOLD_APPEND
    )
    # Any non-models drift joins it -> out-of-band.
    gates = records_root / "gates" / "current-shard.jsonl"
    gates.parent.mkdir(parents=True, exist_ok=True)
    with gates.open("ab") as handle:
        handle.write(b'{"gate": "hand-act"}\n')
    assert (
        drift_delta_class(close_source_drift(records_root, close)) == DELTA_CLASS_OUT_OF_BAND
    )


@pytest.mark.smoke
def test_close_drift_probe_cli_exit_codes(tmp_path: Path) -> None:
    """0 clean, 3 drift, 2 typed refusal; the report names the delta class."""

    import io
    from contextlib import redirect_stdout

    from menagerie.crawler.tests.close_object_support import record_fixture_close
    from menagerie.crawler.tools.close_drift_probe import main as probe_main

    records_root = tmp_path / "records"
    assert (
        probe_main(["--records-root", str(records_root), "--campaign", CAMPAIGN]) == 2
    )
    record_fixture_close(
        records_root,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=SNAPSHOT_ID,
        intake_snapshot_sha256=SNAPSHOT_SHA,
        member_ids=("m1",),
        seed_member_heads=True,
        completion_status="terminal-partition-complete",
    )
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = probe_main(
            [
                "--records-root",
                str(records_root),
                "--campaign",
                CAMPAIGN,
                "--intake",
                SNAPSHOT_ID,
            ]
        )
    assert code == 0
    assert json.loads(buffer.getvalue())["delta_class"] == "none"
    shard = records_root / "models" / "current-shard.jsonl"
    head = json.loads(shard.read_text(encoding="utf-8").splitlines()[0])
    _append_rows(shard, [_hold_row("m1", parent=head)])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = probe_main(
            [
                "--records-root",
                str(records_root),
                "--campaign",
                CAMPAIGN,
                "--intake",
                SNAPSHOT_ID,
            ]
        )
    assert code == 3
    report = json.loads(buffer.getvalue())
    assert report["delta_class"] == "review-hold-append"
    assert report["sources"][0]["name"] == "models.shard"


@pytest.mark.smoke
def test_retroactive_reclass_warning_fires_only_over_live_history(tmp_path: Path) -> None:
    """Door B's catalog row: refining a snapshot with durable leg closes WARNs."""

    from menagerie.crawler.close_drift import retroactive_reclass_warnings
    from menagerie.crawler.rung_staging import (
        RUNG_CLASS_SUCCESSOR,
        append_rung_class_refinement,
        build_rung_registration,
        rung_registry_path,
    )
    from menagerie.crawler.tests.close_object_support import record_fixture_close

    records_root = tmp_path / "records"
    # The leg close lands FIRST (an unregistered snapshot closes with its
    # own in-scope membership), so the later refinement is provably
    # retroactive over live history.
    record_fixture_close(
        records_root,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=SNAPSHOT_ID,
        intake_snapshot_sha256=SNAPSHOT_SHA,
        member_ids=("m1",),
        seed_member_heads=True,
        completion_status="terminal-partition-complete",
    )
    assert retroactive_reclass_warnings(records_root) == ()
    row = build_rung_registration(
        campaign_id=CAMPAIGN,
        snapshot_id=SNAPSHOT_ID,
        snapshot_sha256=SNAPSHOT_SHA,
        derived_from_snapshot_id="intake-" + "00" * 10,
        member_count=1,
        label="step-200-like",
        registered_by="tests",
        created_at="2026-08-25T00:00:00Z",
        licensed_by_close_sha256=None,
        license_kind="historical_bootstrap",
        closes_chain_head=None,
        rung_class=None,
    ).to_row()
    path = rung_registry_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    assert retroactive_reclass_warnings(records_root) == ()
    append_rung_class_refinement(
        records_root,
        snapshot_id=SNAPSHOT_ID,
        snapshot_sha256=SNAPSHOT_SHA,
        rung_class=RUNG_CLASS_SUCCESSOR,
        refined_by="tests",
        created_at="2026-08-31T00:00:00Z",
    )
    (warning,) = retroactive_reclass_warnings(records_root)
    assert "retroactive over live history" in warning
    assert SNAPSHOT_ID in warning
