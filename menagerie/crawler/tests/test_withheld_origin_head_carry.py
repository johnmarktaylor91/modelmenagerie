"""The withheld-origin head-backed carry (the leg-19 calibration incident).

A member can lawfully hold BOTH a recorded head and a newest settlement that
WITHHELD it: BL-8's carve closes an in-window ordinary-valve row over the
member's own pre-window head as WITHHELD, because that row is the leg's only
truthful statement about the member (pinned in
``test_close_contradiction_pins``).  The NEXT close that leaves the member
untouched then resolves it through the head-backed carry arm -- and the arm's
revision pin compared the head against a value a withholding never records
(``origin.record_revision`` is ``None`` for every WITHHELD origin), so every
honest no-op close after a BL-8 withheld settlement refused forever.

That is exactly how the 200b calibration close (leg 19) bricked: leg 8 closed
GREEN withholding 11 members on in-window ``valve.checker-lane`` deferral rows
over their pre-window failed terminals, and the very next close derivation
refused all 11 with "the head moved from None to <rev> ... CARRIED pins the
revision its named ancestor adjudicated" -- over records nothing had touched.

These tests pin the fixed law (``_carry_from_withheld_origin``) and its
non-weakening anchors:

* the untouched close CARRIES the member, inheriting the withheld settlement
  with the origin verbatim and the resident head bound in the evidence;
* the carry is a fixed point across further untouched closes (the chained
  origin re-proves from the settling close's sealed window every time);
* the D-27 carried-authority preflight rides the immutable record for the
  head-riding shape instead of demanding liveness of the superseded row;
* a head that MOVED between generations without adjudication still refuses
  (the ADJUDICATED-origin revision pin is untouched);
* a WITHHELD origin whose park row is dead still refuses (a released or
  voided park carries nothing -- deadness on a release-tracked source is
  never presumed to be head supersession);
* a STILL-LIVE park over the unmoved head carries on the row's authority
  (no head binding; the preflight keeps adjudicating its liveness);
* a live VOID-tracked row never rides unadjudicated head movement -- the
  resident-head witness (the origin close's sealed window floor) refuses a
  stray terminal from an abandoned generation even though the row is live;
* a withheld origin whose settling close is not resident still refuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.partition_proof import (
    PartitionProofError,
    carried_authority_preflight,
    load_partition_proof,
)
from menagerie.crawler.tests.rt3_support import (
    append_capacity_valve,
    append_refusal_park,
    build_repo,
    campaign_member_ids,
    records_root_of,
    stage,
    supersede_head,
    seed_heads,
)

pytestmark = pytest.mark.smoke

CAMPAIGN = "c3-classics"


def _v2_support():
    """Import the v2 fixture surface lazily so collection survives RED."""

    from menagerie.crawler.tests import close_object_support as support

    return support


def _open_leg(repo: Path, staged):
    """Open (or continue) the leg generation under the fixture's one run id."""

    support = _v2_support()
    return support.open_fixture_leg(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
    )


def _close(repo: Path, staged, *, completion_status: str = "complete-with-deferrals"):
    """Close one fixture rung honestly through the production derivation."""

    support = _v2_support()
    return support.record_fixture_close(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_ids=_members(repo, staged),
        completion_status=completion_status,
    )


def _members(repo: Path, staged) -> tuple[str, ...]:
    from menagerie.crawler.tests.rt3_support import members_of

    return members_of(repo, staged.registration.snapshot_id)


def _head_revision(records_root: Path, stable_id: str) -> str:
    """Return one member's current head revision straight from the shard."""

    path = records_root / "models" / "current-shard.jsonl"
    revision = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stable_id") == stable_id:
            revision = str(row["record_revision"])
    assert revision is not None, f"no head recorded for {stable_id}"
    return revision


def _bl8_withheld_state(repo: Path):
    """Drive the exact leg-8 shape: adjudicated head, then a BL-8 withhold.

    Generation A records and adjudicates the member's head.  Generation B
    appends an in-window ordinary-valve deferral over that now-pre-window
    head and closes WITHHELD on it (the BL-8 carve, pinned in
    ``test_close_contradiction_pins``).
    """

    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    first = _close(repo, staged, completion_status="complete")
    assert member in first["partition"]["adjudicated_here"]

    _open_leg(repo, staged)
    valve = append_capacity_valve(rr, member)
    second = _close(repo, staged)
    assert member in second["partition"]["withheld_here"]
    return rr, staged, member, valve, second


def test_an_untouched_close_after_a_bl8_withheld_settlement_carries(
    tmp_path: Path,
) -> None:
    """The leg-19 incident: the honest no-op close derives, CARRIED.

    Nothing touched the member since the BL-8 withheld settlement, so the
    close inherits it: origin verbatim (the withholding pins no revision) and
    the resident head bound in the evidence.
    """

    repo = build_repo(tmp_path)
    rr, staged, member, valve, second = _bl8_withheld_state(repo)

    _open_leg(repo, staged)
    third = _close(repo, staged)

    assert member in third["partition"]["carried"]
    proof = load_partition_proof(rr, str(third["partition_proof_sha256"]))
    entry = proof["members"][member]
    assert entry["disposition"] == "carried"
    origin = entry["evidence"]["origin"]
    assert origin["row_sha256"] == str(valve["row_sha256"])
    assert origin["record_revision"] is None
    assert origin["close_sha256"] == str(second["close_sha256"])
    assert entry["evidence"]["head"]["record_revision"] == _head_revision(rr, member)


def test_the_carry_is_a_fixed_point_across_further_untouched_closes(
    tmp_path: Path,
) -> None:
    """A fourth close re-proves the chained origin from the sealed window."""

    repo = build_repo(tmp_path)
    rr, staged, member, valve, second = _bl8_withheld_state(repo)

    _open_leg(repo, staged)
    _close(repo, staged)
    _open_leg(repo, staged)
    fourth = _close(repo, staged)

    assert member in fourth["partition"]["carried"]
    proof = load_partition_proof(rr, str(fourth["partition_proof_sha256"]))
    origin = proof["members"][member]["evidence"]["origin"]
    assert origin["row_sha256"] == str(valve["row_sha256"])
    assert origin["close_sha256"] == str(second["close_sha256"])


def test_the_carried_authority_preflight_rides_the_head_not_the_dead_row(
    tmp_path: Path,
) -> None:
    """D-27: the head-riding carry has nothing voidable, so no finding.

    Without the discriminator the preflight demanded liveness of the
    head-superseded valve row and refused every subsequent leg START over
    the same lawful carry the close law just sealed.
    """

    repo = build_repo(tmp_path)
    rr, staged, member, _valve, _second = _bl8_withheld_state(repo)
    _open_leg(repo, staged)
    _close(repo, staged)

    assert carried_authority_preflight(rr, campaign_id=CAMPAIGN) == ()


def test_a_moved_head_over_an_adjudicated_origin_still_refuses(
    tmp_path: Path,
) -> None:
    """NON-WEAKENING ANCHOR: the ADJUDICATED-origin revision pin holds.

    A head revision appended BETWEEN generations (below the next floor, so
    never in any window) is unadjudicated custody movement; the carry must
    keep refusing it exactly as before.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _close(repo, staged, completion_status="complete")

    supersede_head(rr, member, intake_snapshot_id=staged.registration.snapshot_id)

    _open_leg(repo, staged)
    with pytest.raises(PartitionProofError, match="CARRIED pins the revision"):
        _close(repo, staged)


def test_a_dead_park_backed_withheld_origin_still_refuses(tmp_path: Path) -> None:
    """NON-WEAKENING ANCHOR: a voided park carries nothing, head or not.

    The member's newest settlement withheld it on a verdict-bearing
    quarantine row; an operator VOIDED the row (the head never moved, so the
    resident-head witness holds).  Deadness on a release/void-tracked source
    is never presumed to be head supersession, so the untouched close still
    refuses to a human.
    """

    from menagerie.crawler.scheduler_parks import (
        append_scheduler_quarantine_row,
        append_scheduler_quarantine_void,
        build_scheduler_quarantine_row,
        build_scheduler_quarantine_void,
        scheduler_quarantine_path,
    )

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _close(repo, staged, completion_status="complete")

    _open_leg(repo, staged)
    row = build_scheduler_quarantine_row(
        stable_id=member,
        work_id="w-quarantine",
        name=member,
        campaign_id=CAMPAIGN,
        run_id="pytest-close",
        machine_id="pytest-host",
        created_at="2026-08-01T00:00:10Z",
        site="scheduler-round",
        reason_code="terminal-proof-generation-exhausted",
        fault_evidence="fixture scheduler fault",
    )
    persisted = append_scheduler_quarantine_row(scheduler_quarantine_path(rr), row)
    second = _close(repo, staged)
    assert member in second["partition"]["withheld_here"]

    # An operator drain voids the row; the head never moves.
    append_scheduler_quarantine_void(
        scheduler_quarantine_path(rr),
        build_scheduler_quarantine_void(
            void_of=str(persisted["row_sha256"]),
            reason="fixture operator drain",
            voided_by="pytest-operator",
            created_at="2026-08-01T00:00:20Z",
        ),
    )

    _open_leg(repo, staged)
    with pytest.raises(
        PartitionProofError, match="released or voided park carries nothing"
    ):
        _close(repo, staged)


def test_a_live_park_over_the_unmoved_head_carries_on_the_row(
    tmp_path: Path,
) -> None:
    """Case A: a STILL-LIVE park beside the unmoved head carries lawfully.

    The row is the authority (headless law verbatim), the resident-head
    witness holds, and the evidence binds NO head -- the D-27 preflight keeps
    adjudicating the row's liveness pre-spend.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _close(repo, staged, completion_status="complete")
    blocked = _head_revision(rr, member)

    _open_leg(repo, staged)
    park = append_refusal_park(rr, member, blocked_parent_revision=blocked)
    second = _close(repo, staged)
    assert member in second["partition"]["withheld_here"]

    _open_leg(repo, staged)
    third = _close(repo, staged)

    assert member in third["partition"]["carried"]
    proof = load_partition_proof(rr, str(third["partition_proof_sha256"]))
    entry = proof["members"][member]
    assert entry["evidence"]["origin"]["row_sha256"] == str(park["row_sha256"])
    assert "head" not in entry["evidence"]
    assert carried_authority_preflight(rr, campaign_id=CAMPAIGN) == ()


def test_a_moved_head_under_a_live_withholding_row_still_refuses(
    tmp_path: Path,
) -> None:
    """NON-WEAKENING ANCHOR: a live row never rides unadjudicated movement.

    A void-tracked quarantine row stays live across an abandoned generation's
    stray terminal, so the row's liveness alone would launder the moved head
    into CARRIED (and ``proved_here``).  The resident-head witness refuses:
    the head's current revision sits past the withheld origin's sealed floor,
    exactly the movement the old revision pin refused.
    """

    from menagerie.crawler.scheduler_parks import (
        append_scheduler_quarantine_row,
        build_scheduler_quarantine_row,
        scheduler_quarantine_path,
    )

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _close(repo, staged, completion_status="complete")

    # Generation B withholds the member on a VERDICT-BEARING quarantine row.
    _open_leg(repo, staged)
    row = build_scheduler_quarantine_row(
        stable_id=member,
        work_id="w-quarantine",
        name=member,
        campaign_id=CAMPAIGN,
        run_id="pytest-close",
        machine_id="pytest-host",
        created_at="2026-08-01T00:00:10Z",
        site="scheduler-round",
        reason_code="terminal-proof-generation-exhausted",
        fault_evidence="fixture scheduler fault",
    )
    append_scheduler_quarantine_row(scheduler_quarantine_path(rr), row)
    second = _close(repo, staged)
    assert member in second["partition"]["withheld_here"]

    # BETWEEN generations an unadjudicated head lands (an abandoned
    # generation's stray terminal).  The quarantine row is void-tracked, not
    # head-tracked, so it stays LIVE -- and the close must still refuse.
    supersede_head(rr, member, intake_snapshot_id=staged.registration.snapshot_id)

    _open_leg(repo, staged)
    with pytest.raises(
        PartitionProofError, match="unadjudicated custody movement never rides a carry"
    ):
        _close(repo, staged)


def test_a_withheld_origin_naming_a_non_resident_close_refuses() -> None:
    """The dead-row path re-proves from a RESIDENT settling close, or refuses."""

    # Lazy import so collection survives RED (the pre-fix tree has no helper).
    from menagerie.crawler.partition_proof import _carry_from_withheld_origin

    member_reasons: dict[str, str] = {}
    entry = _carry_from_withheld_origin(
        Path("/nonexistent"),
        "m0",
        origin={
            "snapshot_id": "intake-x",
            "generation_id": "sha256:gen",
            "close_sha256": "sha256:absent",
            "disposition": "withheld",
            "record_revision": None,
            "row_sha256": "sha256:row",
        },
        ancestor_snapshot_id="intake-x",
        ancestor_close_sha256="sha256:ancestor",
        head_entry={"source": "models.shard", "row_index": 0},
        head={"record_revision": "sha256:rev"},
        live_rows=(),
        authorities={"intake-x": ()},
        member_reasons=member_reasons,
    )
    assert entry is None
    assert "not a resident authority" in member_reasons["m0"]
