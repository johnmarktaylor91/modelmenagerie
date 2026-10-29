"""Backfill migration matrix (RED-first): honest historical closes, no vacuity.

The pilot's mini-rung and step-100 closed BEFORE the leg-generation machinery
existed: no leg-open row, no baselines, no partition proof. The adjudicated
contract's migration clause: a backfill tool DERIVES an honest
``PartitionProofV2`` for each from the CLOSED ledger evidence -- members with a
head recorded under that leg cite it, the never-authored and
host-capacity-deferred members cite their deferral rows, a later rung carries
what an earlier backfilled close resolved -- and nothing vacuous passes. The
era-honest predicate substitutes for the window the era cannot supply:
ADJUDICATED-backfill := current head recorded-under EXACTLY this snapshot id
(the same discriminator ``_require_recorded_history`` already trusts).

The operator supplies no arm, ever; the one operator input is the bounded
``operator_attributed`` authority for a pre-machinery head lacking its
recorded-under field, naming the EXACT current revision, dead the moment the
head moves. The tool is fixture-tested only -- the live pilot campaign records
stay READ-ONLY until the P-sequence executes it deliberately.

The ERA GATE (rt4 break 1, RED-first): "pre-machinery" is derived from the
records, never asserted by a caller. The backfill refuses unless the rung is
provably (a) already terminally closed in the historical records -- a
``leg-close.v1`` row attests this exact rung identity -- AND (b) free of leg
machinery of its own: no machinery-era ``leg`` close and no ``leg``-kind
generation, live or closed. A live/in-flight/unrun rung is a typed refusal
(``BackfillEraError``), and the staging gate recomputes the same predicate on
every replay of a backfill close.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, cast

import pytest

from menagerie.crawler.close_object import CloseObjectError
from menagerie.crawler.rung_staging import RungStagingError
from menagerie.crawler.tests.rt3_support import (
    append_capacity_valve,
    append_refusal_park,
    append_v1_terminal_close,
    build_repo,
    campaign_member_ids,
    close_leg_honestly,
    members_of,
    records_root_of,
    seed_heads,
    stage,
    supersede_head,
)

pytestmark = pytest.mark.smoke

CAMPAIGN = "c3-classics"


def _backfill(repo: Path, snapshot_id: str, **kwargs):
    """Run the backfill minting path for one registered rung."""

    from menagerie.crawler.tools.backfill_close import backfill_close

    return backfill_close(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=snapshot_id,
        minted_by="pytest",
        minted_at="2026-08-16T00:00:00Z",
        **kwargs,
    )


def _pre_machinery_leg(repo: Path, member_ids, *, deferred=(), parked=()):
    """Simulate one pre-machinery leg: heads + valve/park rows + the era's own
    ``leg-close.v1`` terminal-close record (the pilot legs really closed; that
    v1 row is the historical proof the era gate demands). No v2 close."""

    rr = records_root_of(repo)
    staged = stage(repo, CAMPAIGN, stable_ids=member_ids, label="historical")
    ran = [
        m
        for m in members_of(repo, staged.registration.snapshot_id)
        if m not in set(deferred) | set(parked)
    ]
    seed_heads(rr, ran, intake_snapshot_id=staged.registration.snapshot_id)
    for member in deferred:
        append_capacity_valve(rr, member)
    for member in parked:
        append_refusal_park(rr, member)
    append_v1_terminal_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
    )
    return staged


def test_backfill_mints_an_honest_close_for_the_step_100_shape(tmp_path: Path) -> None:
    """The step-100 shape scaled down: run terminals + capacity-deferred +
    review-held members. The backfill derives every arm from the pinned bytes,
    mints a v2 close, and the next rung stages on it."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = _pre_machinery_leg(repo, ids[:3], deferred=(ids[1],), parked=(ids[2],))
    close = _backfill(repo, staged.registration.snapshot_id)
    assert close["close_kind"] == "backfill"
    assert close["schema_version"] == "menagerie.crawler.leg-close.v2"
    following = stage(repo, CAMPAIGN, stable_ids=ids[:4], label="step-next")
    assert following.registration_created


def test_backfill_refuses_the_exhibit_a_shape_per_member(tmp_path: Path) -> None:
    """Exhibit (a) cannot be laundered through the backfill: rung-2's members'
    heads are recorded-under rung-1, rung-1 has no close to carry from, so the
    tool refuses and NAMES the members."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-1")
    seed_heads(
        rr,
        members_of(repo, first.registration.snapshot_id),
        intake_snapshot_id=first.registration.snapshot_id,
    )
    # No close exists for rung-1, so registering the subset through either
    # door would refuse at the lineage gate. The attack therefore hand-writes
    # the snapshot bytes and the registry row -- exactly what a hostile third
    # writer does -- and asks the backfill to bless it.
    from menagerie.crawler.identity import atomic_replace_bytes
    from menagerie.crawler.rung_staging import (
        _plan_subset_snapshot,  # noqa: PLC2701 -- the attack forges what staging plans
        append_rung_registration,
        build_rung_registration,
        rung_registry_path,
    )
    from menagerie.crawler.tests.rt3_support import root_binding

    root_snapshot_dir = rr / root_binding(repo, CAMPAIGN).intake_path
    snapshot_id, snapshot_sha256, manifest_bytes, items_bytes, source_names = _plan_subset_snapshot(
        root_snapshot_dir, ids[:2]
    )
    snapshot_root = rr / "intake" / snapshot_id
    for name in source_names:
        atomic_replace_bytes(
            snapshot_root / "sources" / name,
            (root_snapshot_dir / "sources" / name).read_bytes(),
        )
    atomic_replace_bytes(snapshot_root / "items.jsonl", items_bytes)
    atomic_replace_bytes(snapshot_root / "manifest.json", manifest_bytes)
    append_rung_registration(
        rung_registry_path(rr),
        build_rung_registration(
            campaign_id=CAMPAIGN,
            snapshot_id=snapshot_id,
            snapshot_sha256=snapshot_sha256,
            derived_from_snapshot_id=first.registration.snapshot_id,
            member_count=2,
            label="hostile-rung-2",
            registered_by="pytest",
            created_at="2026-08-16T00:00:00Z",
        ),
    )
    # The attacker forges the era proof too -- the per-member proof must
    # STILL refuse, naming the members (defense in depth behind the era gate).
    append_v1_terminal_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=snapshot_id,
        intake_snapshot_sha256=snapshot_sha256,
    )

    with pytest.raises(Exception) as excinfo:
        _backfill(repo, snapshot_id)
    message = str(excinfo.value)
    assert any(member in message for member in ids[:2])


def test_backfill_carries_members_settled_by_an_earlier_backfilled_close(
    tmp_path: Path,
) -> None:
    """step-100 carrying from mini-rung: members settled under the earlier rung
    and untouched since are CARRIED, citing the earlier BACKFILLED close, and
    the chain bottoms out at a close that adjudicated them in its own era."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    mini = _pre_machinery_leg(repo, ids[:2])
    _backfill(repo, mini.registration.snapshot_id)

    step = stage(repo, CAMPAIGN, stable_ids=ids[:4], label="step-100")
    step_members = members_of(repo, step.registration.snapshot_id)
    fresh = [
        m for m in step_members if m not in set(members_of(repo, mini.registration.snapshot_id))
    ]
    seed_heads(rr, fresh, intake_snapshot_id=step.registration.snapshot_id)
    append_v1_terminal_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=step.registration.snapshot_id,
        intake_snapshot_sha256=step.registration.snapshot_sha256,
    )
    close = _backfill(repo, step.registration.snapshot_id)
    assert close["partition"]["carried_count"] == 2
    from menagerie.crawler.partition_proof import licence_verdict, load_partition_proof
    from menagerie.crawler.rung_staging import introduced_member_ids

    proof = load_partition_proof(rr, close["partition_proof_sha256"])
    assert (
        licence_verdict(
            rr,
            proof,
            lineage_member_ids=step_members,
            introduced_member_ids=cast(
                Sequence[str], introduced_member_ids(rr, step.registration)
            ),
        )
        == "COMPLETE_AT_WIDTH"
    )


def test_backfill_never_authored_members_cite_their_deferral_rows(
    tmp_path: Path,
) -> None:
    """The never-authored / host-capacity-deferred members are accounted by
    their own valve rows -- withheld, never silently absorbed, never carried."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = _pre_machinery_leg(repo, ids[:3], deferred=(ids[1], ids[2]))
    close = _backfill(repo, staged.registration.snapshot_id)
    assert len(close["partition"]["withheld_here"]) == 2

    from menagerie.crawler.partition_proof import load_partition_proof

    proof = load_partition_proof(records_root_of(repo), close["partition_proof_sha256"])
    withheld = {
        member for member, entry in proof["members"].items() if entry["disposition"] == "withheld"
    }
    assert withheld == {ids[1], ids[2]}


def test_backfill_refuses_a_member_with_no_evidence_at_all(tmp_path: Path) -> None:
    """R25: you cannot backfill a leg you cannot prove. One member with no
    head, no valve row, no park refuses the ENTIRE close, named."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = _pre_machinery_leg(repo, ids[:3], deferred=(ids[1],))
    # ids[2] ran; strip its head to fabricate the unprovable member.
    rr = records_root_of(repo)
    shard = rr / "models" / "current-shard.jsonl"
    rows = [
        line
        for line in shard.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["stable_id"] != ids[2]
    ]
    shard.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    with pytest.raises(Exception) as excinfo:
        _backfill(repo, staged.registration.snapshot_id)
    assert ids[2] in str(excinfo.value)


def test_backfill_unattributable_head_needs_an_exact_operator_authority(
    tmp_path: Path,
) -> None:
    """T15: a pre-machinery head lacking recorded-under resolves only through
    an explicit operator authority binding the EXACT current revision --
    absent refuses, exact licenses, stale (head moved) refuses."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="historical")
    members = members_of(repo, staged.registration.snapshot_id)
    seed_heads(rr, members[:1], intake_snapshot_id=staged.registration.snapshot_id)
    seed_heads(rr, members[1:], intake_snapshot_id=None)  # unattributable
    append_v1_terminal_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
    )

    with pytest.raises(Exception) as excinfo:
        _backfill(repo, staged.registration.snapshot_id)
    assert members[1] in str(excinfo.value)

    shard_rows = [
        json.loads(line)
        for line in (rr / "models" / "current-shard.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    revision = next(row["record_revision"] for row in shard_rows if row["stable_id"] == members[1])
    close = _backfill(
        repo,
        staged.registration.snapshot_id,
        operator_attributions={members[1]: revision},
    )
    assert close["close_kind"] == "backfill"

    # Head moves after attribution: a re-mint with the stale authority refuses.
    supersede_head(rr, members[1], intake_snapshot_id=None)
    with pytest.raises(Exception):
        _backfill(
            repo,
            staged.registration.snapshot_id,
            operator_attributions={members[1]: revision},
        )


def test_backfill_is_idempotent_and_a_conflicting_remint_refuses(
    tmp_path: Path,
) -> None:
    """Re-minting over unchanged books returns the identical close; re-minting
    over moved books is a NEW attestation, and the stale one no longer
    licenses (the census re-observation refuses it)."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = _pre_machinery_leg(repo, ids[:2])
    first_close = _backfill(repo, staged.registration.snapshot_id)
    again = _backfill(repo, staged.registration.snapshot_id)
    assert again["close_sha256"] == first_close["close_sha256"]

    append_capacity_valve(rr, ids[0])
    with pytest.raises((CloseObjectError, RungStagingError)):
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="next")


def test_register_door_accepts_a_historical_rung_with_deferred_members(
    tmp_path: Path,
) -> None:
    """I7 seam: proving 'already-run history' must accept the step-100 shape.
    A member with no head but a durable valve/park row IS that leg's record;
    demanding a head for every member made the honest pilot backfill
    unregisterable (a parked member has no head by definition)."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    mini = _pre_machinery_leg(repo, ids[:2])
    _backfill(repo, mini.registration.snapshot_id)

    # Simulate the pilot's step-100 living OUTSIDE the record store: seed the
    # evidence first, then adopt through register_existing_snapshot.
    from menagerie.crawler.intake import load_intake_snapshot
    from menagerie.crawler.rung_staging import register_existing_snapshot

    step = stage(repo, CAMPAIGN, stable_ids=ids[:4], label="step-100")
    step_members = members_of(repo, step.registration.snapshot_id)
    mini_members = set(members_of(repo, mini.registration.snapshot_id))
    ran = [m for m in step_members if m not in mini_members][:1]
    deferred = [m for m in step_members if m not in mini_members and m not in ran]
    seed_heads(rr, ran, intake_snapshot_id=step.registration.snapshot_id)
    for member in deferred:
        append_capacity_valve(rr, member)
    append_v1_terminal_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=step.registration.snapshot_id,
        intake_snapshot_sha256=step.registration.snapshot_sha256,
    )
    _backfill(repo, step.registration.snapshot_id)

    snapshot_root = rr / "intake" / step.registration.snapshot_id
    load_intake_snapshot(snapshot_root)
    adopted = register_existing_snapshot(
        rr,
        snapshot_root,
        campaign_id=CAMPAIGN,
        label="step-100-readopt",
        registered_by="pytest",
        created_at="2026-08-16T01:00:00Z",
        historical_only=True,
    )
    assert not adopted.registration_created  # idempotent re-adoption, no rewind


# ---------------------------------------------------------------------------
# rt4 break 1 -- the era gate: backfill is for provably closed pre-machinery
# history ONLY, derived from the records, never a caller flag.
# ---------------------------------------------------------------------------


def _era_refusal(excinfo) -> str:
    """Assert the refusal is the TYPED era refusal and return its message."""

    from menagerie.crawler.partition_proof import BackfillEraError

    assert isinstance(excinfo.value, BackfillEraError)
    return str(excinfo.value)


def test_backfill_refuses_a_machinery_era_rung_that_ran_nothing(
    tmp_path: Path,
) -> None:
    """rt4 probe 1: rung-1 really ran and closed through the v2 leg machinery;
    a capacity-valve row for a third member landed in rung-1's era; rung-2
    widens by that member and records ZERO bytes. The live gate refuses that
    close -- the backfill must refuse it too, typed, before writing a byte:
    no ``leg-close.v1`` row attests rung-2 ever terminally closed."""

    from menagerie.crawler.close_object import close_object_path
    from menagerie.crawler.leg_generation import leg_events_path
    from menagerie.crawler.partition_proof import PartitionProofError

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    append_capacity_valve(rr, ids[2], created_at="2026-08-01T00:00:00Z")
    close_leg_honestly(repo, first, CAMPAIGN)
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")

    def _ledger_bytes(path: Path) -> bytes:
        return path.read_bytes() if path.is_file() else b""

    closes_before = _ledger_bytes(close_object_path(rr))
    events_before = _ledger_bytes(leg_events_path(rr))
    with pytest.raises(PartitionProofError) as excinfo:
        _backfill(repo, second.registration.snapshot_id)
    message = _era_refusal(excinfo)
    assert "leg-close.v1" in message
    assert second.registration.snapshot_id in message
    # The refusal wrote NOTHING: no close, no backfill generation.
    assert _ledger_bytes(close_object_path(rr)) == closes_before
    assert _ledger_bytes(leg_events_path(rr)) == events_before
    # And the unrun rung still licenses nothing.
    with pytest.raises((CloseObjectError, RungStagingError)):
        stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-3")


def test_backfill_refuses_to_override_a_machinery_leg_close(tmp_path: Path) -> None:
    """rt4 probe 7a: rung-1 closed honestly through the leg machinery, then
    the books moved and its close went stale. A backfill re-mint would
    supersede the leg close (``find_close_object`` returns the newest row);
    the era gate refuses -- a rung with leg machinery of its own is never
    pre-machinery, even when a forged v1 terminal-close row exists."""

    from menagerie.crawler.partition_proof import PartitionProofError

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)
    # Books move; the honest leg close is now stale and inconvenient.
    append_capacity_valve(rr, ids[0])
    # The attacker forges the pre-machinery era proof outright.
    append_v1_terminal_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=first.registration.snapshot_id,
        intake_snapshot_sha256=first.registration.snapshot_sha256,
    )
    with pytest.raises(PartitionProofError) as excinfo:
        _backfill(repo, first.registration.snapshot_id)
    message = _era_refusal(excinfo)
    assert "leg machinery of its own" in message


def test_backfill_refuses_an_in_flight_leg_generation(tmp_path: Path) -> None:
    """rt4 probe 7b: a live, unclosed leg generation is open for the rung.
    The backfill must not mint under a different generation and bypass the
    window that live leg is accumulating -- typed refusal, even when a forged
    v1 terminal-close row exists."""

    from menagerie.crawler.leg_generation import open_leg_generation
    from menagerie.crawler.partition_proof import PartitionProofError

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    seed_heads(
        rr,
        members_of(repo, staged.registration.snapshot_id),
        intake_snapshot_id=staged.registration.snapshot_id,
    )
    append_v1_terminal_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
    )
    generation = open_leg_generation(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        run_id="live-leg",
        opened_at="2026-08-16T00:00:00Z",
    )
    with pytest.raises(PartitionProofError) as excinfo:
        _backfill(repo, staged.registration.snapshot_id)
    message = _era_refusal(excinfo)
    assert generation.generation_id in message
    assert "live" in message


def test_the_staging_gate_recomputes_the_backfill_era_from_the_records(
    tmp_path: Path,
) -> None:
    """The era gate is replay-derived with zero producer input: an honestly
    minted backfill close stops licensing the moment the records stop proving
    the rung pre-machinery (here the v1 terminal-close row is gone), because
    ``proof_refusals`` recomputes the same predicate the mint path applied."""

    from menagerie.crawler.close_object import close_object_path

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = _pre_machinery_leg(repo, ids[:2])
    _backfill(repo, staged.registration.snapshot_id)
    from menagerie.crawler.ledger_chain import verify_records_integrity

    verify_records_integrity(rr)

    # The producer-writable ledger loses the v1 row (rt4 breaks 2/4 territory,
    # but the era predicate must still be RECOMPUTED, not trusted from mint).
    path = close_object_path(rr)
    survivors = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("schema_version") != "menagerie.crawler.leg-close.v1"
    ]
    path.write_text("\n".join(survivors) + "\n", encoding="utf-8")

    with pytest.raises(Exception, match="ledger chain"):
        verify_records_integrity(rr)

    with pytest.raises((CloseObjectError, RungStagingError)) as excinfo:
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="next")
    assert any(
        token in str(excinfo.value) for token in ("leg-close.v1", "ledger chain")
    )


# ---------------------------------------------------------------------------
# FINDING A-1 -- the v1 HISTORY mint chains through the real predecessor.
# ---------------------------------------------------------------------------


def _mint_v1(repo: Path, snapshot_id: str, *, closed_at: str = "2026-05-01T00:00:00Z"):
    """Run the PRODUCTION v1 history-attestation mint for one registered rung."""

    from menagerie.crawler.tools.backfill_close import (
        V1_HISTORY_ATTESTED_ERA,
        mint_v1_history_attestation,
    )

    return mint_v1_history_attestation(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=snapshot_id,
        attested_era=V1_HISTORY_ATTESTED_ERA,
        attested_by="pytest",
        closed_at=closed_at,
    )


def _forged_v1_row(
    *,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    predecessor,
) -> dict:
    """Hand-build one v1 history row naming an ARBITRARY predecessor."""

    from menagerie.crawler.close_object import verifier_identity
    from menagerie.crawler.identity import stable_hash

    body = {
        "schema_version": "menagerie.crawler.leg-close.v1",
        "campaign_id": CAMPAIGN,
        "intake_snapshot_id": intake_snapshot_id,
        "intake_snapshot_sha256": intake_snapshot_sha256,
        "phase": None,
        "run_id": "forged-historian",
        "machine_id": "forged-host",
        "closed_at": "2026-05-01T00:00:00Z",
        "completion_status": "complete",
        "predecessor_close_sha256": predecessor,
        "ledgers": {
            name: {"rows": 0, "ledger_sha256": stable_hash({}), "tail_sha256": None}
            for name in ("models", "attempts", "gates", "artifacts")
        },
        "partition": {
            "intake": 0,
            "valid": True,
            "missing": [],
            "extra": [],
            "duplicates": [],
            "invalid": [],
            "buckets": {},
            "machine_deferred": {},
            "grant_pending": [],
            "refusal_parked": [],
        },
        "currency": {"stale_by_axis": {}},
        "verifier_identity": verifier_identity({}),
        "authority_identities": {},
    }
    row = dict(body)
    row["close_sha256"] = stable_hash(body)
    return row


def test_v1_history_mint_chains_through_the_real_predecessor(tmp_path: Path) -> None:
    """FINDING A-1 (the pilot P-seq operator run): v1 A, v2 A, then v1 B. The
    v1 history mint built its row with NO predecessor field, so it could only
    ever append onto a genesis (None) tip: after the campaign's FIRST v2 close,
    every remaining v1 history attestation refused at append forever ('close
    predecessor None does not equal the durable same-campaign tip'), which
    made the mini-rung's backfill unmintable and bricked step-200 staging. The
    mint must stamp the LEDGER-DERIVED same-campaign v2 tip, exactly as the v2
    mint does -- derived, never a caller argument."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="mini-rung")
    seed_heads(
        rr,
        members_of(repo, first.registration.snapshot_id),
        intake_snapshot_id=first.registration.snapshot_id,
    )
    v1_first = _mint_v1(repo, first.registration.snapshot_id)
    assert v1_first.get("predecessor_close_sha256") is None  # genesis mint
    v2_first = _backfill(repo, first.registration.snapshot_id)

    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="step-100")
    fresh = [
        m
        for m in members_of(repo, second.registration.snapshot_id)
        if m not in set(members_of(repo, first.registration.snapshot_id))
    ]
    seed_heads(rr, fresh, intake_snapshot_id=second.registration.snapshot_id)
    # The pilot's brick: this exact mint refused at append before the fix.
    v1_second = _mint_v1(repo, second.registration.snapshot_id)
    assert v1_second["predecessor_close_sha256"] == v2_first["close_sha256"]
    v2_second = _backfill(repo, second.registration.snapshot_id)
    assert v2_second["predecessor_close_sha256"] == v2_first["close_sha256"]
    # Identity idempotency survives a moved tip: the re-mint returns the SAME
    # row, never a refusal and never a re-chained duplicate.
    assert (
        _mint_v1(repo, second.registration.snapshot_id)["close_sha256"]
        == v1_second["close_sha256"]
    )
    # The boundary the pilot was bricked at now crosses.
    third = stage(repo, CAMPAIGN, stable_ids=ids[:4], label="step-200")
    assert third.registration_created
    # And a later v1 mint keeps chaining as the v2 tip advances.
    v1_third = _mint_v1(repo, third.registration.snapshot_id)
    assert v1_third["predecessor_close_sha256"] == v2_second["close_sha256"]


def test_a_v1_row_naming_a_wrong_predecessor_refuses_at_append(tmp_path: Path) -> None:
    """Tamper half of A-1: the append tripwire keeps refusing a v1 row whose
    predecessor is anything but the durable same-campaign v2 tip -- a claimed
    genesis (None) after a close exists, and a fabricated digest, both refuse
    and write nothing."""

    from menagerie.crawler.close_object import append_close_object, close_object_path

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = _pre_machinery_leg(repo, ids[:2])
    _backfill(repo, staged.registration.snapshot_id)
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="step-100")
    path = close_object_path(rr)
    before = path.read_bytes()
    for wrong in (None, "sha256:" + "f" * 64):
        row = _forged_v1_row(
            intake_snapshot_id=second.registration.snapshot_id,
            intake_snapshot_sha256=second.registration.snapshot_sha256,
            predecessor=wrong,
        )
        with pytest.raises(
            CloseObjectError, match="does not equal the durable same-campaign tip"
        ):
            append_close_object(path, row)
    assert path.read_bytes() == before


def test_a_tampered_v1_predecessor_refuses_on_replay(tmp_path: Path) -> None:
    """Tamper half of A-1, replay side: a persisted v1 row whose predecessor is
    rewritten refuses on every later read -- in place it breaks its own content
    address, and re-stamped it breaks the position-bound ledger chain."""

    from menagerie.crawler.close_object import close_object_path, load_close_objects
    from menagerie.crawler.identity import canonical_json_bytes, row_content_address

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="mini-rung")
    seed_heads(
        rr,
        members_of(repo, first.registration.snapshot_id),
        intake_snapshot_id=first.registration.snapshot_id,
    )
    _mint_v1(repo, first.registration.snapshot_id)
    _backfill(repo, first.registration.snapshot_id)
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="step-100")
    fresh = [
        m
        for m in members_of(repo, second.registration.snapshot_id)
        if m not in set(members_of(repo, first.registration.snapshot_id))
    ]
    seed_heads(rr, fresh, intake_snapshot_id=second.registration.snapshot_id)
    v1_second = _mint_v1(repo, second.registration.snapshot_id)
    path = close_object_path(rr)
    original = path.read_bytes()

    def _rewrite(mutate) -> None:
        out = []
        for line in original.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("close_sha256") == v1_second["close_sha256"]:
                out.append(canonical_json_bytes(mutate(dict(row))))
            else:
                out.append(line)
        path.write_bytes(b"\n".join(out) + b"\n")

    def _flip(row: dict) -> dict:
        row["predecessor_close_sha256"] = "sha256:" + "0" * 64
        return row

    _rewrite(_flip)
    with pytest.raises(CloseObjectError, match="digest does not match"):
        load_close_objects([path])

    def _flip_and_restamp(row: dict) -> dict:
        chain = row.pop("chain", None)
        row.pop("close_sha256", None)
        row["predecessor_close_sha256"] = "sha256:" + "0" * 64
        row["close_sha256"] = row_content_address(row)
        if chain is not None:
            row["chain"] = chain
        return row

    path.write_bytes(original)
    _rewrite(_flip_and_restamp)
    with pytest.raises(CloseObjectError, match="chain"):
        load_close_objects([path])


def test_v1_history_mint_refuses_a_machinery_era_registered_rung(tmp_path: Path) -> None:
    """Unbricking post-close v1 mints must not WIDEN the era gate: before A-1
    the first v2 close incidentally blocked every later v1 mint, machinery-era
    rungs included. The honest replacement predicate: a rung whose REGISTRATION
    was licensed by a machinery-era ``leg`` close is not pre-machinery history,
    however clean its own machinery census is -- and the licence kind is
    re-derived from the close ledger, never read off the registry string. The
    same predicate replays inside the backfill era gate, so a forged v1 row
    cannot unlock a backfill for a machinery-era registration either."""

    from menagerie.crawler.partition_proof import BackfillEraError, PartitionProofError

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert second.registration.license_kind == "live_origin_custody"
    with pytest.raises(BackfillEraError) as excinfo:
        _mint_v1(repo, second.registration.snapshot_id)
    assert "machinery-era licence" in str(excinfo.value)
    # Defense in depth, replay side: forge the v1 era proof outright -- the
    # backfill era gate re-derives the registration licence from the ledgers
    # and still refuses.
    append_v1_terminal_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=second.registration.snapshot_id,
        intake_snapshot_sha256=second.registration.snapshot_sha256,
    )
    with pytest.raises(PartitionProofError) as excinfo2:
        _backfill(repo, second.registration.snapshot_id)
    assert isinstance(excinfo2.value, BackfillEraError)
    assert "machinery-era licence" in str(excinfo2.value)
