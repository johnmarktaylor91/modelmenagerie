"""Close-object v2 contract matrix (RED-first, 2026-08-16 tri-lab synthesis).

Every test here pins one clause of the adjudicated CLOSE-OBJECT CONTRACT v2:
a generation-scoped sealed ``PartitionProofV2`` the close binds by digest;
per-member CITATION (head-in-window / CARRIED-from-a-named-ancestor-close /
withholding-in-window), never existence-over-history; a half-open leg window
with a floor anchored against the campaign's previous close; attested-set =
read-set through ONE shared source census; and a single coverage authority
(the sealed proof -- never the live heads projection, never an all-time
valve/park scan).

Committed FAILING first: the rt3 red-team round proved both live breaks with
UNTRACKED scratch tests that never entered this suite, and the round-2 fix
shipped green beside a RED test that certified the wrong derivation
(``recorded_heads=()`` handed to the unit as a literal while the production
caller derived it campaign-globally). Every licensing pin in this file
therefore drives the PRODUCTION callers -- ``stage_rung`` /
``require_closed_lineage`` over a real records root -- never the unit function
with caller-supplied literals.

Section map:
  A  -- rt3 break A (TIME): zero-byte rungs, stale parks, carried discipline.
  B  -- rt3 break B (SPACE): shadow layouts, census, presence flips.
  C  -- the v2 machinery's own attack surface (forged floors, proofs, opens).
  D  -- prior-door insurance that is NEW here (registry duplicate refusal).
Positive controls sit beside each door so over-fixing refuses too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from menagerie.crawler.close_object import (
    CloseObjectError,
    close_object_path,
    require_closed_lineage,
    verifier_identity,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.rung_staging import RungStagingError, introduced_member_ids
from menagerie.crawler.tests.rt3_support import (
    append_capacity_valve,
    append_refusal_park,
    build_repo,
    campaign_member_ids,
    close_leg_honestly,
    head_record,
    members_of,
    models_ledger_digest,
    records_root_of,
    seed_heads,
    stage,
    supersede_head,
    void_refusal_park,
)

pytestmark = pytest.mark.smoke

CAMPAIGN = "c3-classics"


def _v2_support():
    """Import the v2 fixture surface lazily so collection survives RED."""

    from menagerie.crawler.tests import close_object_support as support

    return support


def _open_leg(repo: Path, staged, *, run_id: str = "pytest-leg"):
    """Open (or continue) the leg generation for one staged rung."""

    support = _v2_support()
    return support.open_fixture_leg(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        run_id=run_id,
    )


def _forge_close(repo: Path, staged, member_citations, **kwargs):
    """Forge a structurally valid v2 close row with attacker-chosen citations."""

    support = _v2_support()
    return support.forge_fixture_close(
        records_root_of(repo),
        campaign_id=kwargs.pop("campaign_id", CAMPAIGN),
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_citations=member_citations,
        **kwargs,
    )


def _shard_rows(repo: Path) -> list[dict]:
    path = records_root_of(repo) / "models" / "current-shard.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _head_citation(repo: Path, stable_id: str, *, source: str = "models.shard") -> dict:
    """Cite one member's current shard head exactly (row index + revision)."""

    rows = _shard_rows(repo)
    for index in range(len(rows) - 1, -1, -1):
        if rows[index]["stable_id"] == stable_id:
            return {
                "disposition": "adjudicated",
                "evidence": {
                    "source": source,
                    "row_index": index,
                    "record_revision": rows[index]["record_revision"],
                },
            }
    raise AssertionError(f"no shard head for {stable_id}")


def _force_register_subset(
    repo: Path, predecessor: Any, stable_ids: Sequence[str], *, label: str
) -> Any:
    """Force-register a non-widening attack exhibit behind the public door."""

    from menagerie.crawler.rung_staging import (
        _write_subset_snapshot,
        append_rung_registration,
        build_rung_registration,
        resolve_verified_root,
        rung_registry_path,
    )

    records_root = records_root_of(repo)
    binding, _root = resolve_verified_root(records_root, CAMPAIGN)
    snapshot, snapshot_created = _write_subset_snapshot(
        records_root / binding.intake_path,
        tuple(stable_ids),
        records_root / "intake",
    )
    staged = append_rung_registration(
        rung_registry_path(records_root),
        build_rung_registration(
            campaign_id=CAMPAIGN,
            snapshot_id=snapshot.snapshot_id,
            snapshot_sha256=snapshot.snapshot_sha256,
            derived_from_snapshot_id=predecessor.registration.snapshot_id,
            member_count=len(snapshot.items),
            label=label,
            registered_by="pytest-force-register",
            created_at="2026-08-16T00:00:01Z",
        ),
    )
    return type(staged)(
        registration=staged.registration,
        snapshot_root=staged.snapshot_root,
        snapshot_created=snapshot_created,
        registration_created=staged.registration_created,
    )


def _demand_tip_close(repo: Path, staged: Any, *, label: str) -> dict[str, Any]:
    """Run the production licensing gate for one force-registered tip."""

    records_root = records_root_of(repo)
    members = frozenset(members_of(repo, staged.registration.snapshot_id))
    return require_closed_lineage(
        records_root,
        intake_snapshot_id=staged.registration.snapshot_id,
        campaign_id=CAMPAIGN,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        lineage_member_ids=members,
        introduced_member_ids=introduced_member_ids(records_root, staged.registration),
        label=label,
    )


# ---------------------------------------------------------------------------
# Section A -- rt3 break A (TIME): the evidence pool must be THIS leg's window.
# ---------------------------------------------------------------------------


def test_honest_leg_close_licenses_the_next_rung_end_to_end(tmp_path: Path) -> None:
    """Positive control: a leg that opened, recorded every member in-window
    (recorded-under its own snapshot), and closed, licenses the next rung."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close = close_leg_honestly(repo, first, CAMPAIGN)
    assert close["schema_version"] == "menagerie.crawler.leg-close.v2"
    assert close["close_kind"] == "leg"
    from menagerie.crawler.partition_proof import licence_verdict, load_partition_proof

    proof = load_partition_proof(records_root_of(repo), close["partition_proof_sha256"])
    assert (
        licence_verdict(
            records_root_of(repo),
            proof,
            lineage_member_ids=ids[:2],
            introduced_member_ids=ids[:2],
        )
        == "LICENSING"
    )
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert second.registration_created
    _open_leg(repo, second)
    seed_heads(
        records_root_of(repo),
        ids[2:3],
        intake_snapshot_id=second.registration.snapshot_id,
    )
    _v2_support().record_fixture_close(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=second.registration.snapshot_id,
        intake_snapshot_sha256=second.registration.snapshot_sha256,
        member_ids=members_of(repo, second.registration.snapshot_id),
    )
    supersede_head(
        records_root_of(repo),
        ids[0],
        intake_snapshot_id=first.registration.snapshot_id,
    )
    with pytest.raises(Exception, match="head moved"):
        close_leg_honestly(repo, second, CAMPAIGN, run_and_record=False)


def test_a_zero_byte_rung_cannot_close_on_its_predecessors_heads(tmp_path: Path) -> None:
    """rt3 exhibit (a), the exact captured shape: rung-2 (a subset of rung-1's
    members) appends ZERO bytes, and a close claiming its members as run
    OUTCOMES -- citing the predecessor-era heads -- must refuse staging.

    The heads are real, current, and pre-window; recorded-under names rung-1.
    Both signals contradict the ADJUDICATED claim, and the forged floor-zero
    window is refused by continuity against rung-1's own close."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)

    with pytest.raises(RungStagingError, match="strictly widen"):
        stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-2")
    second = _force_register_subset(repo, first, ids[:2], label="rung-2-forced")
    before = models_ledger_digest(rr)
    generation = _open_leg(repo, second)
    assert models_ledger_digest(rr) == before  # rung-2 recorded NOTHING

    members = members_of(repo, second.registration.snapshot_id)
    _forge_close(
        repo,
        second,
        {member: _head_citation(repo, member) for member in members},
        generation=generation,
        window_floor_zero=True,
    )
    with pytest.raises((CloseObjectError, RungStagingError)) as excinfo:
        _demand_tip_close(repo, second, label="rung-3")
    assert "window" in str(excinfo.value) or "floor" in str(excinfo.value)


def test_a_zero_byte_rung_may_close_honestly_as_carried_and_disclose_it(
    tmp_path: Path,
) -> None:
    """An honest all-CARRIED close attests, but licenses no successor.

    This is the F1 impossibility pin: a zero-byte rung may close honestly, but
    ``proved_here`` is empty because every origin belongs to an ancestor rung.
    It therefore cannot advance the ladder even when a caller attempts another
    non-widening stage.
    """

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)

    with pytest.raises(RungStagingError, match="strictly widen"):
        stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-2")
    second = _force_register_subset(repo, first, ids[:2], label="rung-2-forced")
    close = close_leg_honestly(repo, second, CAMPAIGN, run_and_record=False)
    assert close["partition"]["adjudicated_here"] == []
    assert close["partition"]["carried_count"] == 2
    from menagerie.crawler.partition_proof import licence_verdict, load_partition_proof

    proof = load_partition_proof(records_root_of(repo), close["partition_proof_sha256"])
    assert (
        licence_verdict(
            records_root_of(repo),
            proof,
            lineage_member_ids=members_of(repo, second.registration.snapshot_id),
            introduced_member_ids=(),
        )
        == "HONEST_NO_OP"
    )
    with pytest.raises((CloseObjectError, RungStagingError)) as excinfo:
        _demand_tip_close(repo, second, label="rung-3")
    assert "HONEST NO-OP" in str(excinfo.value)


def test_a_widening_rung_cannot_carry_a_member_nothing_ever_resolved(
    tmp_path: Path,
) -> None:
    """The practically important half of break A: a rung that ADDS a member and
    runs nothing cannot close -- the new member has no ancestor resolution to
    carry and no in-window evidence, so the close refuses at BUILD."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)

    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    with pytest.raises(Exception) as excinfo:
        close_leg_honestly(repo, second, CAMPAIGN, run_and_record=False)
    assert ids[2] in str(excinfo.value)


def test_a_close_without_its_leg_open_row_licenses_nothing(tmp_path: Path) -> None:
    """Non-vacuity is structural: a close must point at a real, durable
    leg-open row for its generation. A close nobody opened a leg for -- a
    fabricated open-row digest -- cannot exist for the gate."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    generation = _open_leg(repo, first)
    seed_heads(
        records_root_of(repo),
        members_of(repo, first.registration.snapshot_id),
        intake_snapshot_id=first.registration.snapshot_id,
    )
    members = members_of(repo, first.registration.snapshot_id)
    _forge_close(
        repo,
        first,
        {member: _head_citation(repo, member) for member in members},
        generation=generation,
        leg_open_row_sha256="sha256:" + "0" * 64,
    )
    with pytest.raises((CloseObjectError, RungStagingError)) as excinfo:
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert "open" in str(excinfo.value).lower()


def test_a_stale_park_row_cannot_corroborate_a_withholding(tmp_path: Path) -> None:
    """rt3 exhibit (b): a refusal-park row appended in an EARLIER rung's era is
    below this leg's window floor and proves nothing, however live it looks."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    never_ran = ids[3]

    first = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-1")
    park_row = append_refusal_park(rr, never_ran)  # rung-1 era, pre-close
    close_leg_honestly(repo, first, CAMPAIGN)

    second = stage(repo, CAMPAIGN, stable_ids=ids[:4], label="rung-2")
    generation = _open_leg(repo, second)
    citations: dict[str, dict[str, Any]] = {
        member: {"disposition": "carried", "evidence": None}
        for member in ids[:3]
    }
    citations[never_ran] = {
            "disposition": "withheld",
            "evidence": {
                "source": "park.refusal",
                "row_index": 0,
                "row_sha256": park_row["row_sha256"],
                "kind": park_row["disposition"],
            },
    }
    _forge_close(repo, second, citations, generation=generation)
    with pytest.raises((CloseObjectError, RungStagingError)):
        _demand_tip_close(repo, second, label="rung-3")


def test_an_in_window_live_park_is_honest_withholding(tmp_path: Path) -> None:
    """Positive control: an in-window host-capacity deferral licenses."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    parked = ids[1]

    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    generation = _open_leg(repo, first)
    seed_heads(rr, ids[:1], intake_snapshot_id=first.registration.snapshot_id)
    append_capacity_valve(rr, parked)
    support = _v2_support()
    close = support.record_fixture_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=first.registration.snapshot_id,
        intake_snapshot_sha256=first.registration.snapshot_sha256,
        member_ids=members_of(repo, first.registration.snapshot_id),
        completion_status="terminal-partition-complete-with-refusal-parks",
    )
    assert generation.generation_id == close["generation_id"]
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert second.registration_created


def test_a_parked_member_is_carried_only_while_the_park_stays_live(
    tmp_path: Path,
) -> None:
    """A headless member an ancestor close resolved as WITHHELD carries forward
    exactly while that withholding row stays live; voiding the park kills the
    carry and the next close refuses at build."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    parked = ids[1]

    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    generation = _open_leg(repo, first)
    del generation
    seed_heads(rr, ids[:1], intake_snapshot_id=first.registration.snapshot_id)
    park_row = append_refusal_park(rr, parked)
    support = _v2_support()
    support.record_fixture_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=first.registration.snapshot_id,
        intake_snapshot_sha256=first.registration.snapshot_sha256,
        member_ids=members_of(repo, first.registration.snapshot_id),
        completion_status="terminal-partition-complete-with-refusal-parks",
    )

    # Carry the still-live park across rung-2: licenses.
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    _open_leg(repo, second)
    seed_heads(rr, ids[2:3], intake_snapshot_id=second.registration.snapshot_id)
    support.record_fixture_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=second.registration.snapshot_id,
        intake_snapshot_sha256=second.registration.snapshot_sha256,
        member_ids=members_of(repo, second.registration.snapshot_id),
        completion_status="terminal-partition-complete-with-refusal-parks",
    )
    third = stage(repo, CAMPAIGN, stable_ids=ids[:4], label="rung-3")
    _open_leg(repo, third)
    seed_heads(rr, ids[3:4], intake_snapshot_id=third.registration.snapshot_id)

    # Void the park: the carry authority is gone, so rung-3's no-op close
    # cannot account the member and refuses at build.
    void_refusal_park(rr, park_row)
    with pytest.raises(Exception) as excinfo:
        support.record_fixture_close(
            rr,
            campaign_id=CAMPAIGN,
            intake_snapshot_id=third.registration.snapshot_id,
            intake_snapshot_sha256=third.registration.snapshot_sha256,
            member_ids=members_of(repo, third.registration.snapshot_id),
            completion_status="terminal-partition-complete-with-refusal-parks",
        )
    assert parked in str(excinfo.value)


def test_an_in_window_attempt_forbids_closing_the_member_as_carried(
    tmp_path: Path,
) -> None:
    """R9 anti-laundering: a member the leg WORKED (an in-window attempt row
    names it) but failed to record may not be closed as carried -- selected and
    unrecorded is a hole, not an inheritance."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)

    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    _open_leg(repo, second)
    seed_heads(rr, ids[2:3], intake_snapshot_id=second.registration.snapshot_id)
    attempts = rr / "attempts" / "local.jsonl"
    attempts.parent.mkdir(parents=True, exist_ok=True)
    with attempts.open("ab") as handle:
        handle.write(canonical_json_bytes({"stable_id": ids[0], "work_id": "w-laundered"}) + b"\n")
    support = _v2_support()
    with pytest.raises(Exception) as excinfo:
        support.record_fixture_close(
            rr,
            campaign_id=CAMPAIGN,
            intake_snapshot_id=second.registration.snapshot_id,
            intake_snapshot_sha256=second.registration.snapshot_sha256,
            member_ids=members_of(repo, second.registration.snapshot_id),
        )
    assert ids[0] in str(excinfo.value)

    # FB-4 / R9 quantifier residue: a PRE-window park for the selected member
    # is live against its unchanged ancestor head, but it is not THIS leg's
    # resume short-circuit.  It may not suppress the selected-member refusal
    # and launder the member into CARRIED.
    stray_root = tmp_path / "stray-withholding"
    stray_root.mkdir()
    stray_repo = build_repo(stray_root)
    stray_records = records_root_of(stray_repo)
    stray_ids = campaign_member_ids(stray_repo, CAMPAIGN)
    stray_first = stage(
        stray_repo, CAMPAIGN, stable_ids=stray_ids[:2], label="stray-rung-1"
    )
    close_leg_honestly(stray_repo, stray_first, CAMPAIGN)
    blocked_revision = _shard_rows(stray_repo)[0]["record_revision"]
    append_refusal_park(
        stray_records,
        stray_ids[0],
        work_id="work-unrelated-stray-park",
        blocked_parent_revision=blocked_revision,
    )
    # Re-close the same rung so the immutable predecessor census honestly
    # attests the pre-window park; staging may never proceed over moved bytes.
    _v2_support().record_fixture_close(
        stray_records,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=stray_first.registration.snapshot_id,
        intake_snapshot_sha256=stray_first.registration.snapshot_sha256,
        member_ids=members_of(stray_repo, stray_first.registration.snapshot_id),
        completion_status="terminal-partition-complete-with-refusal-parks",
        run_id="stray-park-attestation",
    )
    stray_second = stage(
        stray_repo, CAMPAIGN, stable_ids=stray_ids[:3], label="stray-rung-2"
    )
    generation = _open_leg(stray_repo, stray_second, run_id="stray-selected-run")
    from menagerie.crawler.leg_generation import record_leg_plan

    record_leg_plan(
        stray_records,
        campaign_id=CAMPAIGN,
        generation_id=generation.generation_id,
        run_id="stray-selected-run",
        admitted_ids=(stray_ids[0],),
        plan_payload={"fb4": "selected-with-unrelated-live-park"},
        created_at="2026-08-18T00:00:00Z",
    )
    seed_heads(
        stray_records,
        stray_ids[2:3],
        intake_snapshot_id=stray_second.registration.snapshot_id,
        run_id="stray-selected-run",
    )
    with pytest.raises(Exception, match="selected member may not be closed as carried"):
        _v2_support().record_fixture_close(
            stray_records,
            campaign_id=CAMPAIGN,
            intake_snapshot_id=stray_second.registration.snapshot_id,
            intake_snapshot_sha256=stray_second.registration.snapshot_sha256,
            member_ids=members_of(stray_repo, stray_second.registration.snapshot_id),
            run_id="stray-selected-run",
        )


def test_a_forged_in_window_head_recorded_under_an_ancestor_contradicts(
    tmp_path: Path,
) -> None:
    """The ADJUDICATED dual signal: an in-window row whose recorded-under names
    an ancestor snapshot is a contradiction, not a pass -- and the mirrored
    forgery (pre-window row claiming this snapshot) refuses on position."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)

    # A strict WIDENING so rung-2 is a distinct, admissible snapshot identity.
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert second.registration.snapshot_id != first.registration.snapshot_id
    generation = _open_leg(repo, second)
    # Forge an IN-WINDOW superseding head whose dependency vector names rung-1:
    # position says "this leg wrote it", the vector says an ancestor did.
    forged = supersede_head(rr, ids[0], intake_snapshot_id=first.registration.snapshot_id)
    with pytest.raises(Exception) as build_excinfo:
        close_leg_honestly(repo, second, CAMPAIGN, run_and_record=False)
    assert "contradict" in str(build_excinfo.value)
    seed_heads(rr, ids[2:3], intake_snapshot_id=second.registration.snapshot_id)
    citations = {
        ids[0]: _head_citation(repo, ids[0]),
        ids[1]: {"disposition": "carried", "evidence": None},
        ids[2]: _head_citation(repo, ids[2]),
    }
    assert citations[ids[0]]["evidence"]["record_revision"] == forged["record_revision"]
    _forge_close(repo, second, citations, generation=generation)
    with pytest.raises((CloseObjectError, RungStagingError)):
        stage(repo, CAMPAIGN, stable_ids=ids[:4], label="rung-3")

    provenance_root = tmp_path / "provenance"
    provenance_root.mkdir()
    provenance_repo = build_repo(provenance_root)
    provenance_ids = campaign_member_ids(provenance_repo, CAMPAIGN)
    provenance_rung = stage(
        provenance_repo,
        CAMPAIGN,
        stable_ids=provenance_ids[:2],
        label="provenance-rung",
    )
    _open_leg(provenance_repo, provenance_rung, run_id="joined-run")
    seed_heads(
        records_root_of(provenance_repo),
        provenance_ids[:2],
        intake_snapshot_id=provenance_rung.registration.snapshot_id,
        run_id="foreign-run",
    )
    with pytest.raises(Exception, match="did not join generation"):
        _v2_support().record_fixture_close(
            records_root_of(provenance_repo),
            campaign_id=CAMPAIGN,
            intake_snapshot_id=provenance_rung.registration.snapshot_id,
            intake_snapshot_sha256=provenance_rung.registration.snapshot_sha256,
            member_ids=members_of(
                provenance_repo, provenance_rung.registration.snapshot_id
            ),
            run_id="joined-run",
        )


def test_history_rewritten_under_an_open_leg_refuses_at_close(tmp_path: Path) -> None:
    """I-TIME: rewriting a pre-window byte between open and close refuses at build.

    TWO authorities cover this, and each arm names the one that fires:

    * a rewritten row BODY is now caught by the chain's content address, which is
      derived from that body -- one authority earlier than the window floor
      (globalized Track A address; the fact ledgers are chain-verified through
      their census-source scopes);
    * a byte-level rewrite that leaves every PARSED row identical is invisible to
      any row digest, and is caught by the window floor's prefix hash. That arm
      keeps the floor's own authority tested rather than shadowed.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)

    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    _open_leg(repo, second)
    shard = rr / "models" / "current-shard.jsonl"
    honest_bytes = shard.read_bytes()
    rows = _shard_rows(repo)
    rows[0]["status"]["code"] = "failed:rewritten-history"
    shard.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    support = _v2_support()

    def close() -> Any:
        """Build the close over the current bytes."""

        return support.record_fixture_close(
            rr,
            campaign_id=CAMPAIGN,
            intake_snapshot_id=second.registration.snapshot_id,
            intake_snapshot_sha256=second.registration.snapshot_sha256,
            member_ids=members_of(repo, second.registration.snapshot_id),
        )

    with pytest.raises(Exception) as excinfo:
        close()
    assert "is not its own body content address" in str(excinfo.value), excinfo.value

    # PARSE-INVARIANT rewrite: the rows decode identically, so every declared
    # digest still re-derives and the chain address is unchanged. Only the floor's
    # prefix hash sees it.
    lines = honest_bytes.splitlines(keepends=True)
    padded = json.dumps(json.loads(lines[0]), sort_keys=True, separators=(", ", ": "))
    shard.write_bytes(padded.encode("utf-8") + b"\n" + b"".join(lines[1:]))
    with pytest.raises(Exception) as floor:
        close()
    assert "prefix" in str(floor.value) or "rewrit" in str(floor.value), floor.value


# ---------------------------------------------------------------------------
# Section B -- rt3 break B (SPACE): attested set == read set, one census.
# ---------------------------------------------------------------------------


def test_a_flat_models_ledger_created_after_the_close_refuses_staging(
    tmp_path: Path,
) -> None:
    """rt3 exhibit B, first rung, no history: heads materialized into
    ``models.jsonl`` while the pinned shard stays empty must flip the census
    and refuse -- an unattested layout may never steer the decision."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)

    members = members_of(repo, first.registration.snapshot_id)
    (rr / "models.jsonl").write_bytes(
        b"".join(
            canonical_json_bytes(head_record(m, intake_snapshot_id=first.registration.snapshot_id))
            + b"\n"
            for m in members
        )
    )
    with pytest.raises((CloseObjectError, RungStagingError)) as excinfo:
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert "models" in str(excinfo.value)


def test_a_flat_layout_present_at_open_is_attested_and_usable(tmp_path: Path) -> None:
    """Positive control: the flat driver-rig layout is a first-class census
    slot. Present at leg open, recorded into in-window, it adjudicates."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    (rr / "models.jsonl").write_bytes(b"")
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    _open_leg(repo, first)
    members = members_of(repo, first.registration.snapshot_id)
    seed_heads(
        rr,
        members,
        intake_snapshot_id=first.registration.snapshot_id,
        layout="models.jsonl",
    )
    support = _v2_support()
    close = support.record_fixture_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=first.registration.snapshot_id,
        intake_snapshot_sha256=first.registration.snapshot_sha256,
        member_ids=members,
    )
    assert close["sources"]["models.flat"]["present"] is True
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert second.registration_created


def test_a_layout_may_not_change_presence_during_a_leg(tmp_path: Path) -> None:
    """All slots have genesis files, and any present-to-absent flip refuses."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    generation = _open_leg(repo, first)
    assert all(
        bool(generation.baselines[name]["present"])
        for name in generation.baselines
    ), "all 15 source slots must have a materialized genesis before the window opens"
    members = members_of(repo, first.registration.snapshot_id)
    (rr / "attempts" / "local.jsonl").unlink()
    seed_heads(
        rr,
        members,
        intake_snapshot_id=first.registration.snapshot_id,
        layout="models/current-shard.jsonl",
    )
    support = _v2_support()
    with pytest.raises(Exception) as excinfo:
        support.record_fixture_close(
            rr,
            campaign_id=CAMPAIGN,
            intake_snapshot_id=first.registration.snapshot_id,
            intake_snapshot_sha256=first.registration.snapshot_sha256,
            member_ids=members,
        )
    assert "presen" in str(excinfo.value).lower()


def test_any_census_source_moved_between_close_and_staging_refuses(
    tmp_path: Path,
) -> None:
    """Every census slot is load-bearing: a valve ledger that moved since the
    close means the close no longer describes the state being staged from."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)
    append_capacity_valve(rr, ids[0])
    with pytest.raises((CloseObjectError, RungStagingError)) as excinfo:
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert "capacity" in str(excinfo.value)


def test_a_first_rung_close_cannot_pass_vacuously_with_zero_history(
    tmp_path: Path,
) -> None:
    """First rung, zero appended bytes: ADJUDICATED has nothing in-window to
    cite, CARRIED has no ancestor close, WITHHELD has no in-window row. The
    forged all-outcomes close refuses; there is no 'no history so nothing to
    check' hole."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    generation = _open_leg(repo, first)
    members = members_of(repo, first.registration.snapshot_id)
    _forge_close(
        repo,
        first,
        {
            member: {
                "disposition": "adjudicated",
                "evidence": {
                    "source": "models.shard",
                    "row_index": index,
                    "record_revision": "sha256:" + "1" * 64,
                },
            }
            for index, member in enumerate(members)
        },
        generation=generation,
    )
    with pytest.raises((CloseObjectError, RungStagingError)):
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")


def test_a_v1_close_licenses_nothing_and_names_the_backfill_remedy(
    tmp_path: Path,
) -> None:
    """Migration rule 1: ``leg-close.v1`` rows stay readable history and
    license nothing; the refusal points at the backfill tool."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    members = members_of(repo, first.registration.snapshot_id)
    seed_heads(rr, members, intake_snapshot_id=first.registration.snapshot_id)

    body = {
        "schema_version": "menagerie.crawler.leg-close.v1",
        "campaign_id": CAMPAIGN,
        "intake_snapshot_id": first.registration.snapshot_id,
        "intake_snapshot_sha256": first.registration.snapshot_sha256,
        "phase": None,
        "run_id": "pytest-v1",
        "machine_id": "pytest-host",
        "closed_at": "2026-08-16T00:00:00Z",
        "completion_status": "complete",
        "ledgers": {
            name: {"rows": 0, "ledger_sha256": stable_hash({}), "tail_sha256": None}
            for name in ("models", "attempts", "gates", "artifacts")
        },
        "partition": {
            "intake": len(members),
            "valid": True,
            "missing": [],
            "extra": [],
            "duplicates": [],
            "invalid": [],
            "buckets": {"runs": list(members)},
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
    path = close_object_path(rr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")

    with pytest.raises((CloseObjectError, RungStagingError)) as excinfo:
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert "backfill" in str(excinfo.value)


def test_close_and_staging_resolve_models_layouts_through_one_census() -> None:
    """B' / I-SPACE: exactly one module names the models layout paths. The
    prefer-first fork in ``tools/discharge_generation`` and the private layout
    tuple in ``rung_staging`` were how the read set and the attested set
    diverged; both must resolve through the shared census."""

    import ast
    import inspect

    from menagerie.crawler import close_object, rung_staging
    from menagerie.crawler.tools import discharge_generation, scheduler_shadow_audit

    def _code_strings(module) -> set[str]:
        tree = ast.parse(inspect.getsource(module))
        docstrings: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    docstrings.add(id(body[0].value))
        return {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        }

    for module in (rung_staging, close_object, discharge_generation, scheduler_shadow_audit):
        strings = _code_strings(module)
        assert "current-shard.jsonl" not in strings, module.__name__
        assert "models.jsonl" not in strings, module.__name__

    from menagerie.crawler.leg_generation import close_ledger_sources

    sources = close_ledger_sources(Path("/nonexistent"))
    assert {"models.shard", "models.flat"} <= set(sources)

    # The T4 scale contract used to be invoked from HERE, because round-2 read
    # the node-ID lock as a reason to keep its own clauses uncollectable
    # (``__test__ = False`` plus one wrapper). That made the whole scale contract
    # deletable with the lock still green (round-3 R3F-5), so its clauses are now
    # ordinary collected nodes in ``test_close_read_scale.py`` and the lock sees
    # every one of them.


# ---------------------------------------------------------------------------
# Section C -- the v2 machinery's own attack surface.
# ---------------------------------------------------------------------------


def test_a_tampered_or_absent_partition_proof_refuses(tmp_path: Path) -> None:
    """R11: the binding is only as good as the resident proof. A missing or
    byte-flipped proof file refuses staging."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close = close_leg_honestly(repo, first, CAMPAIGN)

    from menagerie.crawler.partition_proof import leg_proofs_dir

    proof_path = leg_proofs_dir(rr) / (
        close["partition_proof_sha256"].removeprefix("sha256:") + ".json"
    )
    original = proof_path.read_bytes()
    proof_path.write_bytes(original.replace(b"adjudicated", b"adjudicatex", 1))
    with pytest.raises((CloseObjectError, RungStagingError)):
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    proof_path.unlink()
    with pytest.raises((CloseObjectError, RungStagingError)):
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")


def test_a_generation_is_continued_on_resume_and_new_after_close(
    tmp_path: Path,
) -> None:
    """I8 / resume continuity: re-opening the same (campaign, snapshot, phase)
    before any close continues the SAME generation -- a kill/resume leg spans
    runs. After a close, the next open mints a NEW generation."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    open_a = _open_leg(repo, first, run_id="run-a")
    open_b = _open_leg(repo, first, run_id="run-b")
    assert open_a.generation_id == open_b.generation_id
    assert open_a.baselines == open_b.baselines
    close_leg_honestly(repo, first, CAMPAIGN)
    open_c = _open_leg(repo, first, run_id="run-c")
    assert open_c.generation_id != open_a.generation_id


def test_the_recomputation_diff_names_the_member_and_the_axis(tmp_path: Path) -> None:
    """Refusal quality is part of the contract: the per-member diff must name
    the member so the operator can see WHAT was misclaimed."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    close_leg_honestly(repo, first, CAMPAIGN)
    second = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    generation = _open_leg(repo, second)
    members = members_of(repo, second.registration.snapshot_id)
    head_ids = {str(row["stable_id"]) for row in _shard_rows(repo)}
    citations = {
        member: _head_citation(repo, member)
        if member in head_ids
        else {
            "disposition": "adjudicated",
            "evidence": {
                "source": "models.shard",
                "row_index": 99,
                "record_revision": "sha256:" + "2" * 64,
            },
        }
        for member in members
    }
    _forge_close(repo, second, citations, generation=generation)
    with pytest.raises((CloseObjectError, RungStagingError)) as excinfo:
        stage(repo, CAMPAIGN, stable_ids=ids[:4], label="rung-3")
    assert any(member in str(excinfo.value) for member in members)


# ---------------------------------------------------------------------------
# Section D -- prior-door insurance new to this wave.
# ---------------------------------------------------------------------------


def test_duplicate_registry_snapshot_ids_refuse_outright(tmp_path: Path) -> None:
    """I-TIP / R23: the uniform-lineage gate's soundness rested on the two
    registry writers never appending a duplicate. The READER now enforces what
    the writers promise: a duplicate snapshot id -- even byte-identical -- is a
    third-writer signature and refuses, because a tolerated duplicate moves
    ``registrations[-1]`` and silently rewinds the tip."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")

    from menagerie.crawler.rung_staging import load_rung_registrations, rung_registry_path

    registry = rung_registry_path(rr)
    row = json.loads(registry.read_text(encoding="utf-8").splitlines()[0])
    with registry.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
    del first
    with pytest.raises(RungStagingError) as excinfo:
        load_rung_registrations(rr)
    assert any(
        token in str(excinfo.value).lower()
        for token in ("duplicate", "twice", "ledger chain")
    )
