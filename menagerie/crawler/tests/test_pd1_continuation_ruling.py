"""RULING W7 (PD-1) pins: continuation-marker semantics at close, N1-N14 + attacks.

The KERNEL_DECISION PD-1 of 2026-08-24 flips exactly two LC-1 pins
(``test_close_contradiction_pins``) and rules that the flips are NOT lawful
without the RED half of every arm the ruling newly relies on. This module is
that half, one test per required pin:

* N1  both-arm refusing half (verdict-bearing rows keep every refusal);
* N2  custody corroboration, both directions (OPEN-3 default);
* N3  unknown/unclassified reason refuses at build and at read;
* N4  marker-monotonicity property (a marker never converts a refusal);
* N5  held-marker cell (OPEN-5 default: the close completes, the member holds);
* N6  per-row park liveness over ALL THREE park families (O A3, clause 9);
* N7  proof era v3 pinned byte-identical forever (clause 8);
* N8  the v4 differential is exactly the marker-only cell (O R-8);
* N9  introduced marker-only member refuses INCOMPLETE_LADDER at staging;
* N10 backfill-era continuation (clause 10);
* N11 exact close-status grammar (D-24);
* N12 closed reason-census set-equality, no default arm;
* N13 production write-site census (the out-of-ledger laundering owner);
* N14 production-manifestation rehearsal, zero human acts.

Plus the section-4 laundering attacks, each a test proving STRUCTURAL
impossibility: producer-declared kind (A1/S-i), marker multiplication (S-ii),
liveness collapse (A3), row self-contradiction (A4 template strike), era
laundering (attack D / the era door), and licence-side laundering (attack F).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.identity import canonical_json_bytes, stamp_row_digest
from menagerie.crawler.partition_proof import (
    PartitionProofError,
    derive_partition_proof,
    load_partition_proof,
    proof_refusals,
    proved_here,
    write_partition_proof,
)
from menagerie.crawler.scheduler_parks import (
    SchedulerQuarantineError,
    append_scheduler_quarantine_row,
    append_scheduler_quarantine_void,
    build_scheduler_quarantine_row,
    build_scheduler_quarantine_void,
    scheduler_quarantine_path,
    validate_scheduler_quarantine_row,
)
from menagerie.crawler.tests.rt3_support import (
    append_refusal_park,
    build_repo,
    campaign_member_ids,
    members_of,
    records_root_of,
    seed_heads,
    stage,
)

pytestmark = pytest.mark.smoke

CAMPAIGN = "c3-classics"
RUN_ID = "pytest-close"


# ---------------------------------------------------------------------------
# fixture plumbing (mirrors the LC-1 pin file exactly)
# ---------------------------------------------------------------------------


def _support():
    from menagerie.crawler.tests import close_object_support as support

    return support


def _open_leg(repo: Path, staged):
    support = _support()
    return support.open_fixture_leg(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
    )


def _build_marker(
    stable_id: str,
    *,
    work_id: str = "w-quarantine",
    created_at: str = "2026-08-01T00:00:10Z",
    reason_code: str = "no-durable-routing-progress",
    run_id: str = RUN_ID,
):
    return build_scheduler_quarantine_row(
        stable_id=stable_id,
        work_id=work_id,
        name=stable_id,
        campaign_id=CAMPAIGN,
        run_id=run_id,
        machine_id="pytest-host",
        created_at=created_at,
        site="scheduler-round",
        reason_code=reason_code,
        fault_evidence="fixture scheduler fault",
    )


def _append_quarantine(records_root: Path, stable_id: str, **kwargs):
    row = _build_marker(stable_id, **kwargs)
    return append_scheduler_quarantine_row(scheduler_quarantine_path(records_root), row)


def _append_quarantine_raw(records_root: Path, stable_id: str, **kwargs):
    """Append one built row RAW, as ledger merges interleave rows (O A3 shape)."""

    row = _build_marker(stable_id, **kwargs)
    path = scheduler_quarantine_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
    return row


def _close(repo: Path, staged, **kwargs):
    support = _support()
    return support.record_fixture_close(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_ids=members_of(repo, staged.registration.snapshot_id),
        completion_status=kwargs.pop("completion_status", "complete-with-deferrals"),
        **kwargs,
    )


def _head_revision(records_root: Path, stable_id: str) -> str:
    path = records_root / "models" / "current-shard.jsonl"
    revision: Optional[str] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("stable_id") == stable_id:
            revision = str(row["record_revision"])
    assert revision is not None, f"no head recorded for {stable_id}"
    return revision


def _append_custody_artifact(records_root: Path, stable_id: str) -> None:
    """Append one in-window artifact row naming the member (custody noise)."""

    path = records_root / "artifacts" / "current-shard.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(
            canonical_json_bytes({"stable_id": stable_id, "kind": "fixture-artifact-transition"})
            + b"\n"
        )


def _append_charged_park_event(
    records_root: Path, stable_id: str, work_id: str, created_at: str
) -> None:
    """Append one budget-charging park event in the driver's exact key shape."""

    from menagerie.crawler.recordio import operational_events_path
    from menagerie.crawler.scheduler_parks import (
        SCHEDULER_QUARANTINE_PARK_EVENT_DISPOSITION,
    )

    path = operational_events_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "created_at": created_at,
        "details": {
            "disposition": SCHEDULER_QUARANTINE_PARK_EVENT_DISPOSITION,
            "stable_id": stable_id,
            "work_id": work_id,
        },
    }
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(event) + b"\n")


def _marker_only_rung(tmp_path: Path):
    """One staged rung: head for member 0, ONLY a live routing marker for 1."""

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    marker = _append_quarantine(rr, ids[1])
    return repo, rr, ids, staged, marker


def _derive(repo: Path, staged, *, proof_schema_version: Optional[str] = None):
    from menagerie.crawler.leg_generation import record_leg_plan

    rr = records_root_of(repo)
    generation = _open_leg(repo, staged)
    record_leg_plan(
        rr,
        campaign_id=CAMPAIGN,
        generation_id=generation.generation_id,
        run_id=RUN_ID,
        admitted_ids=(),
        plan_payload={"fixture": "derive"},
        created_at="2026-08-01T00:00:05Z",
    )
    kwargs: dict[str, Any] = {}
    if proof_schema_version is not None:
        kwargs["proof_schema_version"] = proof_schema_version
    return derive_partition_proof(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        phase=None,
        close_kind="leg",
        generation_id=generation.generation_id,
        baselines=generation.baselines,
        member_ids=members_of(repo, staged.registration.snapshot_id),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# N12 + N3: the closed reason census and the future-kind fail-closed guard
# ---------------------------------------------------------------------------


def test_n12_reason_census_partitions_exactly_with_no_default_arm() -> None:
    """N12: ROUTING + VERDICT + AMBIGUOUS == SCHEDULER_QUARANTINE_REASONS.

    Set equality with NO default arm: adding a reason without classifying it
    turns this test RED, never a silent inheritance (ruling clause 1;
    F property 1, closing O C-2's copy-paste class for this ledger).
    """

    from menagerie.crawler.scheduler_parks import (
        AMBIGUOUS_QUARANTINE_REASONS,
        ROUTING_MARKER_REASONS,
        SCHEDULER_QUARANTINE_REASONS,
        VERDICT_BEARING_QUARANTINE_REASONS,
    )

    assert ROUTING_MARKER_REASONS == frozenset(
        {"non-model-fault", "no-durable-routing-progress", "invalidation-nonconvergent"}
    )
    assert VERDICT_BEARING_QUARANTINE_REASONS == frozenset(
        {"award-blocked:spent-chain", "terminal-proof-generation-exhausted"}
    )
    assert AMBIGUOUS_QUARANTINE_REASONS == frozenset({"member-fault", "award-unroutable"})
    assert (
        ROUTING_MARKER_REASONS | VERDICT_BEARING_QUARANTINE_REASONS | AMBIGUOUS_QUARANTINE_REASONS
    ) == SCHEDULER_QUARANTINE_REASONS
    assert not ROUTING_MARKER_REASONS & VERDICT_BEARING_QUARANTINE_REASONS
    assert not ROUTING_MARKER_REASONS & AMBIGUOUS_QUARANTINE_REASONS
    assert not VERDICT_BEARING_QUARANTINE_REASONS & AMBIGUOUS_QUARANTINE_REASONS


def test_n3_unknown_or_unclassified_reason_refuses_at_build_and_at_read(
    tmp_path: Path,
) -> None:
    """N3: an unknown reason refuses at BUILD and at READ, fail-closed."""

    from menagerie.crawler.scheduler_parks import quarantine_reason_kind

    with pytest.raises(SchedulerQuarantineError):
        _build_marker("m-x", reason_code="reason-nobody-classified")

    # A future reason added to the vocabulary WITHOUT a census class refuses
    # in the kind function itself -- classification is total or it refuses.
    with pytest.raises(SchedulerQuarantineError):
        quarantine_reason_kind("reason-nobody-classified")

    # At read: a digest-consistent row smuggling an unknown reason refuses.
    row = _build_marker("m-x")
    body = {key: value for key, value in row.items() if key != "row_sha256"}
    body["scheduler_quarantine"] = dict(body["scheduler_quarantine"])
    body["scheduler_quarantine"]["reason_code"] = "reason-nobody-classified"
    forged = stamp_row_digest(body)
    with pytest.raises(SchedulerQuarantineError):
        validate_scheduler_quarantine_row(forged)


# ---------------------------------------------------------------------------
# clause 11 + attack S(i): settlement_class is a validated-redundant tripwire
# ---------------------------------------------------------------------------


def test_settlement_class_is_derived_from_the_registry_and_cannot_steer() -> None:
    """Clause 11: v3 rows carry settlement_class; the registry stays authority.

    Attack S(i): copying a verdict-bearing reason into a row that DECLARES
    kind routing is dead -- no row field is kind authority, and a
    settlement_class disagreeing with the clause-1 registry refuses at read.
    """

    from menagerie.crawler.scheduler_parks import (
        SCHEDULER_QUARANTINE_SCHEMA_VERSION,
        expected_settlement_class,
    )

    routing = _build_marker("m-r", reason_code="non-model-fault")
    assert routing["schema_version"] == SCHEDULER_QUARANTINE_SCHEMA_VERSION
    assert routing["scheduler_quarantine"]["settlement_class"] == "routing-marker"

    verdict = _build_marker("m-v", reason_code="award-blocked:spent-chain")
    assert verdict["scheduler_quarantine"]["settlement_class"] == "verdict-bearing"

    # AMBIGUOUS reasons carry the fail-closed static class (O R-2): the
    # verdict-bearing default; only clause-3 custody corroboration at close
    # may treat the row as routing, and no field can claim it.
    ambiguous = _build_marker("m-a", reason_code="member-fault")
    assert ambiguous["scheduler_quarantine"]["settlement_class"] == "verdict-bearing"
    assert expected_settlement_class("member-fault") == "verdict-bearing"

    # The laundering direction: spent-chain wearing routing-marker refuses.
    body = {key: value for key, value in verdict.items() if key != "row_sha256"}
    body["scheduler_quarantine"] = dict(body["scheduler_quarantine"])
    body["scheduler_quarantine"]["settlement_class"] = "routing-marker"
    forged = stamp_row_digest(body)
    with pytest.raises(SchedulerQuarantineError) as excinfo:
        validate_scheduler_quarantine_row(forged)
    assert "settlement_class" in str(excinfo.value)


def test_historical_v2_rows_stay_readable_and_may_not_carry_settlement_class() -> None:
    """History stays valid; a v2 row smuggling the v3 field refuses (closed shape)."""

    from menagerie.crawler.scheduler_parks import (
        SCHEDULER_QUARANTINE_DISPOSITION,
        SCHEDULER_QUARANTINE_REMEDY,
        SCHEDULER_QUARANTINE_SCHEMA_VERSION_V2,
    )

    v2_body: dict[str, Any] = {
        "schema_version": SCHEDULER_QUARANTINE_SCHEMA_VERSION_V2,
        "stable_id": "m-h",
        "work_id": "w-h",
        "name": "m-h",
        "campaign_id": CAMPAIGN,
        "run_id": "run-h",
        "machine_id": "host-h",
        "created_at": "2026-08-01T00:00:00Z",
        "disposition": SCHEDULER_QUARANTINE_DISPOSITION,
        "scheduler_quarantine": {
            "site": "scheduler-round",
            "reason_code": "non-model-fault",
            "fault_evidence": "historical row",
            "exception_type": None,
            "remedy": SCHEDULER_QUARANTINE_REMEDY,
        },
        "recheck_hint": "historical remedy text; no verdict was rendered",
    }
    validated = validate_scheduler_quarantine_row(stamp_row_digest(dict(v2_body)))
    assert validated["schema_version"] == SCHEDULER_QUARANTINE_SCHEMA_VERSION_V2

    smuggled = dict(v2_body)
    smuggled["scheduler_quarantine"] = dict(smuggled["scheduler_quarantine"])
    smuggled["scheduler_quarantine"]["settlement_class"] = "routing-marker"
    with pytest.raises(SchedulerQuarantineError):
        validate_scheduler_quarantine_row(stamp_row_digest(smuggled))


def test_a4_the_settled_prose_is_struck_from_quarantine_templates() -> None:
    """A4/clause 11: no builder or emission template may say "settled".

    The real surviving 200a row m_ec7b said both "no verdict was rendered"
    and "member settled for this leg" (O E-7). Prose can never again disagree
    with the closed vocabulary: the builder's recheck hint and the crawler's
    quarantine emission templates carry no "settled" claim.
    """

    row = _build_marker("m-t")
    assert "settled" not in row["recheck_hint"]
    assert "no verdict was rendered" in row["recheck_hint"]

    crawler_root = Path(__file__).resolve().parent.parent
    offenders = []
    for source in ("driver_receipts.py", "driver_admission.py", "scheduler_parks.py"):
        text = (crawler_root / source).read_text(encoding="utf-8")
        if "member settled" in text:
            offenders.append(source)
    assert not offenders, f"quarantine templates still claim settlement: {offenders}"


# ---------------------------------------------------------------------------
# N1: the both-arm refusing half (verdict-bearing rows keep every refusal)
# ---------------------------------------------------------------------------


def test_n1a_a_head_beside_a_live_refusal_park_still_refuses(tmp_path: Path) -> None:
    """N1 (F's half): H2 + live in-window park.refusal refuses, retyped.

    Previously implied, never pinned. Ruling clause 4: a live in-window
    verdict-bearing row beside an in-window head refuses exactly as today,
    retyped ``head-beside-live-custody-withholding`` naming the row.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    # Blocked ON the current head: the park stays live beside it (a refused
    # supersession whose blocked parent still governs).
    append_refusal_park(rr, member, blocked_parent_revision=_head_revision(rr, member))

    with pytest.raises(PartitionProofError) as excinfo:
        _close(repo, staged)
    reason = excinfo.value.member_reasons.get(member, "")
    assert "head-beside-live-custody-withholding" in reason
    assert "park.refusal" in reason


def test_n1b_a_head_beside_a_spent_chain_quarantine_still_refuses(
    tmp_path: Path,
) -> None:
    """N1 (O's half, OPEN-2 default): spent-chain is VERDICT-BEARING.

    ``award-blocked:spent-chain`` guards post-spend custody (O E-3: the real
    spent-chain fault); it may never enter the privileged exclusion set, so a
    head beside it refuses ``head-beside-live-custody-withholding``.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _append_quarantine(rr, member, reason_code="award-blocked:spent-chain")

    with pytest.raises(PartitionProofError) as excinfo:
        _close(repo, staged)
    reason = excinfo.value.member_reasons.get(member, "")
    assert "head-beside-live-custody-withholding" in reason
    assert "award-blocked:spent-chain" in reason


# ---------------------------------------------------------------------------
# N2: custody corroboration, both directions (OPEN-3 default)
# ---------------------------------------------------------------------------


def test_n2_member_fault_over_live_custody_refuses_marker_contradicts_custody(
    tmp_path: Path,
) -> None:
    """N2 (the 200a row-37 shape): a routing-labelled row over live custody.

    Exclusion requires custody SILENCE (ruling clause 3, O R-3): the realised
    A1 attack put the most routing-looking label on a custody dead end 3 of
    22 times. A member-fault marker beside an in-window artifact transition
    is a NEW typed refusal, never an exclusion.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _append_custody_artifact(rr, member)
    _append_quarantine(rr, member, reason_code="member-fault")

    with pytest.raises(PartitionProofError) as excinfo:
        _close(repo, staged)
    reason = excinfo.value.member_reasons.get(member, "")
    assert "marker-contradicts-custody" in reason
    assert "artifacts" in reason


def test_n2_member_fault_custody_silent_is_excluded_and_adjudicates(
    tmp_path: Path,
) -> None:
    """N2 (the 200a row-5 / m5717 shape): custody-silent member-fault excludes.

    The corroboration conjunct resolves an AMBIGUOUS reason per row: custody
    replay-silent means the marker proves the fault was machinery-side, the
    validated head is the window's rendered truth, and the member adjudicates
    with the marker superseded and disclosed.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _append_quarantine(rr, member, reason_code="member-fault")

    close = _close(repo, staged)
    assert member in close["partition"]["adjudicated_here"]
    proof = load_partition_proof(rr, str(close["partition_proof_sha256"]))
    superseded = proof["members"][member]["evidence"]["superseded_markers"]
    assert superseded[0]["reason_code"] == "member-fault"


def test_n2_a_headless_ambiguous_reason_over_live_custody_withholds_and_proves(
    tmp_path: Path,
) -> None:
    """N2 third arm (clause 1, the T16 cell): ambiguous resolves VERDICT-BEARING.

    An AMBIGUOUS reason (member-fault) over live custody with NO head is
    "verdict-bearing absent corroboration, fail-closed" -- the row STANDS as
    the member's withholding exactly as pre-ruling (it proves; the close
    completes). Only a ROUTING-classified label over live custody is the
    ``marker-contradicts-custody`` laundering refusal, and only beside a head
    does the ambiguous case contradict (the 200a row-37 shape). This is the
    real driver cell: a publication failure quarantined member-fault AFTER its
    artifact chain was authorized must keep closing and resuming with zero
    human acts.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _append_custody_artifact(rr, ids[1])
    _append_quarantine(rr, ids[1], reason_code="member-fault")

    close = _close(repo, staged)
    assert ids[1] in close["partition"]["withheld_here"]
    assert ids[1] in close["partition"]["proved_here"]
    proof = load_partition_proof(rr, str(close["partition_proof_sha256"]))
    assert proof["members"][ids[1]]["disposition"] == "withheld"


# ---------------------------------------------------------------------------
# N4 + attack S(ii): marker monotonicity -- a marker never converts a refusal
# ---------------------------------------------------------------------------


def test_n4_markers_never_convert_the_park_refusal_even_multiplied(
    tmp_path: Path,
) -> None:
    """N4 + S(ii): adding routing markers to a refusing cell changes nothing.

    Marker multiplication hoping latest-wins is dead: verdict-bearing rows
    DOMINATE (ruling clause 9), and the N1a refusal survives any number of
    valid routing markers (F property 2; S monotone law).
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    append_refusal_park(rr, member, blocked_parent_revision=_head_revision(rr, member))
    for ordinal in range(3):
        _append_quarantine_raw(
            rr,
            member,
            work_id=f"w-mult-{ordinal}",
            created_at=f"2026-08-01T00:00:2{ordinal}Z",
        )

    with pytest.raises(PartitionProofError) as excinfo:
        _close(repo, staged)
    reason = excinfo.value.member_reasons.get(member, "")
    assert "head-beside-live-custody-withholding" in reason
    assert "park.refusal" in reason


def test_n4_markers_never_clear_an_identity_conjunct_refusal(tmp_path: Path) -> None:
    """N4: the snapshot/run/lineage conjuncts are independent (T8).

    A head recorded-under a foreign snapshot refuses; a routing marker beside
    it clears nothing.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id="snapshot-that-is-not-this-one")
    _append_quarantine(rr, member)

    with pytest.raises(PartitionProofError) as excinfo:
        _close(repo, staged)
    reason = excinfo.value.member_reasons.get(member, "")
    assert "contradict" in reason


# ---------------------------------------------------------------------------
# N5: the held-marker cell (OPEN-5 default)
# ---------------------------------------------------------------------------


def test_n5_an_exhausted_marker_holds_disclosed_while_the_close_completes(
    tmp_path: Path,
) -> None:
    """N5 (OPEN-5 default): charged >= limit means HELD, never a close refusal.

    Two charged park events with no restoring boundary exhaust the budget
    (PD-11 law, replay-derived). The member HOLDS as operator-owed debt --
    disclosed in the close's continuing_members currency block -- and a third
    marker buys nothing: the append is idempotent per work generation, so
    marker manufacture cannot even mint a row.
    """

    repo, rr, ids, staged, marker = _marker_only_rung(tmp_path)
    work_id = str(marker["work_id"])
    _append_charged_park_event(rr, ids[1], work_id, "2026-08-01T00:00:11Z")
    _append_charged_park_event(rr, ids[1], work_id, "2026-08-01T00:00:12Z")

    # A third park attempt is a no-op returning the incumbent row.
    again = _append_quarantine(rr, ids[1], created_at="2026-08-01T00:00:13Z")
    assert str(again["row_sha256"]) == str(marker["row_sha256"])

    close = _close(repo, staged)
    assert ids[1] in close["partition"]["continuing"]
    disclosures = {entry["stable_id"]: entry for entry in close["currency"]["continuing_members"]}
    assert disclosures[ids[1]]["held"] is True
    assert disclosures[ids[1]]["charged_parks"] == 2
    assert disclosures[ids[1]]["reason_code"] == "no-durable-routing-progress"


# ---------------------------------------------------------------------------
# N6 + attack A3: per-row liveness over ALL THREE park families (clause 9)
# ---------------------------------------------------------------------------


def test_n6_scheduler_voiding_the_later_park_does_not_erase_the_earlier(
    tmp_path: Path,
) -> None:
    """N6(i)/A3: per-row liveness -- a void releases exactly the row it names.

    VERIFIED RED on pre-ruling main by execution (O A3): with latest-per-key
    liveness, lawfully voiding the LATER of two unvoided parks for one work
    generation silently erased the EARLIER one too. Ruled (clause 9): every
    unreleased park is live.
    """

    from menagerie.crawler.scheduler_parks import current_scheduler_quarantine_rows

    rr = records_root_of(build_repo(tmp_path))
    earlier = _append_quarantine_raw(rr, "m-a3", created_at="2026-08-01T00:00:01Z")
    later = _append_quarantine_raw(rr, "m-a3", created_at="2026-08-01T00:00:02Z")

    append_scheduler_quarantine_void(
        scheduler_quarantine_path(rr),
        build_scheduler_quarantine_void(
            void_of=str(later["row_sha256"]),
            reason="fixture drain of the later row",
            voided_by="pytest-operator",
            created_at="2026-08-01T00:00:03Z",
        ),
    )

    from menagerie.crawler.scheduler_parks import load_scheduler_quarantine_ledger

    live = current_scheduler_quarantine_rows(
        load_scheduler_quarantine_ledger([scheduler_quarantine_path(rr)])
    )
    live_digests = {str(row["row_sha256"]) for row in live}
    assert str(earlier["row_sha256"]) in live_digests
    assert str(later["row_sha256"]) not in live_digests


def test_n6_scheduler_a_second_release_against_a_released_row_refuses(
    tmp_path: Path,
) -> None:
    """N6(ii): a release must name a currently-live row (closing O C-6)."""

    rr = records_root_of(build_repo(tmp_path))
    row = _append_quarantine(rr, "m-c6")
    path = scheduler_quarantine_path(rr)
    append_scheduler_quarantine_void(
        path,
        build_scheduler_quarantine_void(
            void_of=str(row["row_sha256"]),
            reason="first drain",
            voided_by="pytest-operator",
            created_at="2026-08-01T00:00:03Z",
        ),
    )
    with pytest.raises(SchedulerQuarantineError):
        append_scheduler_quarantine_void(
            path,
            build_scheduler_quarantine_void(
                void_of=str(row["row_sha256"]),
                reason="second, different drain",
                voided_by="pytest-operator",
                created_at="2026-08-01T00:00:04Z",
            ),
        )
    with pytest.raises(SchedulerQuarantineError):
        append_scheduler_quarantine_void(
            path,
            build_scheduler_quarantine_void(
                void_of="sha256:" + "0" * 64,
                reason="void of nothing",
                voided_by="pytest-operator",
                created_at="2026-08-01T00:00:05Z",
            ),
        )


def test_n6_grant_parks_are_per_row_live_and_refuse_stale_releases(
    tmp_path: Path,
) -> None:
    """N6 generic half, grant-park family (O pin 5 + remainder 3)."""

    from menagerie.crawler.grant_parks import (
        GrantParkError,
        append_grant_park_row,
        append_grant_park_void,
        build_grant_park_row,
        build_grant_park_void,
        current_grant_park_rows,
        grant_park_path,
        load_grant_park_rows,
    )

    rr = records_root_of(build_repo(tmp_path))
    path = grant_park_path(rr)

    def _park(revision: str, created_at: str):
        return append_grant_park_row(
            path,
            build_grant_park_row(
                stable_id="m-g",
                work_id="w-g",
                name="m-g",
                campaign_id=CAMPAIGN,
                run_id="run-g",
                machine_id="host-g",
                created_at=created_at,
                blocked_parent_revision=revision,
                blocked_stage="accuracy-gate",
                pending_model={
                    "stable_id": "m-g",
                    "status": {"code": "runs", "reason_code": None, "stage": None},
                },
            ),
        )

    earlier = _park("rev-earlier", "2026-08-01T00:00:01Z")
    later = _park("rev-later", "2026-08-01T00:00:02Z")
    assert str(earlier["row_sha256"]) != str(later["row_sha256"])

    void = build_grant_park_void(
        void_of=str(later["row_sha256"]),
        reason="fixture void of the later park",
        voided_by="pytest-operator",
        created_at="2026-08-01T00:00:03Z",
    )
    append_grant_park_void(path, void)

    live = {str(row["row_sha256"]) for row in load_grant_park_rows([path])}
    assert str(earlier["row_sha256"]) in live
    assert str(later["row_sha256"]) not in live
    assert current_grant_park_rows

    with pytest.raises(GrantParkError):
        append_grant_park_void(
            path,
            build_grant_park_void(
                void_of=str(later["row_sha256"]),
                reason="second, different void",
                voided_by="pytest-operator",
                created_at="2026-08-01T00:00:04Z",
            ),
        )


def test_n6_refusal_parks_are_per_row_live_and_refuse_stale_releases(
    tmp_path: Path,
) -> None:
    """N6 generic half, refusal-park family (O pin 5 + remainder 3)."""

    from menagerie.crawler.refusal_parks import (
        RefusalParkError,
        append_refusal_park_void,
        build_refusal_park_void,
        load_refusal_park_rows,
        refusal_park_path,
    )

    rr = records_root_of(build_repo(tmp_path))
    earlier = append_refusal_park(
        rr, "m-r", created_at="2026-08-01T00:00:01Z", blocked_parent_revision="rev-a"
    )
    later = append_refusal_park(
        rr, "m-r", created_at="2026-08-01T00:00:02Z", blocked_parent_revision="rev-b"
    )
    path = refusal_park_path(rr)
    append_refusal_park_void(
        path,
        build_refusal_park_void(
            void_of=str(later["row_sha256"]),
            reason="fixture void of the later park",
            voided_by="pytest-operator",
            created_at="2026-08-01T00:00:03Z",
        ),
    )

    live = {str(row["row_sha256"]) for row in load_refusal_park_rows([path])}
    assert str(earlier["row_sha256"]) in live
    assert str(later["row_sha256"]) not in live

    with pytest.raises(RefusalParkError):
        append_refusal_park_void(
            path,
            build_refusal_park_void(
                void_of=str(later["row_sha256"]),
                reason="second, different void",
                voided_by="pytest-operator",
                created_at="2026-08-01T00:00:04Z",
            ),
        )


# ---------------------------------------------------------------------------
# N7 + N8: the era clause (v3 pinned byte-identical forever; v4 differential)
# ---------------------------------------------------------------------------


def test_n7_proof_era_v3_re_derives_byte_identically_forever(tmp_path: Path) -> None:
    """N7 (ruling clause 8): sealed v3 semantics are PINNED, not migrated.

    The exact pre-ruling law -- marker-only member closes WITHHELD, enters
    ``proved_here``, licenses -- re-derives under proof schema v3 forever.
    A resident v3-bound close (all of 200a) recomputes hash-for-hash and its
    licence stands, disclosed per PD-13, forward-only.
    """

    from menagerie.crawler.partition_proof import (
        PARTITION_PROOF_SCHEMA_VERSION_V3,
        withheld_here,
    )

    repo, rr, ids, staged, _marker = _marker_only_rung(tmp_path)
    proof = _derive(repo, staged, proof_schema_version=PARTITION_PROOF_SCHEMA_VERSION_V3)
    assert proof["schema_version"] == PARTITION_PROOF_SCHEMA_VERSION_V3
    assert proof["members"][ids[1]]["disposition"] == "withheld"
    assert ids[1] in withheld_here(proof)
    assert ids[1] in proved_here(proof)

    # Determinism of the pinned era: a second derivation is byte-identical.
    again = _derive(repo, staged, proof_schema_version=PARTITION_PROOF_SCHEMA_VERSION_V3)
    assert canonical_json_bytes(proof) == canonical_json_bytes(again)

    # A RESIDENT v3-bound close replays clean: the gate derives under the
    # proof's OWN era and the digest matches hash-for-hash.
    write_partition_proof(rr, proof)
    support = _support()
    close = support.append_close_as_resident_history(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        proof=proof,
        generation=_open_leg(repo, staged),
    )
    assert (
        proof_refusals(
            rr, close, lineage_member_ids=members_of(repo, staged.registration.snapshot_id)
        )
        == ()
    )

    # And the v3 licence STANDS: the successor stages over it (era clause 8,
    # forward-only -- exactly what keeps all of 200a's ladder lawful).
    successor = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert successor.registration_created


def test_n7_pin_the_v3_proof_body_never_carries_the_dx23_citation_key(
    tmp_path: Path,
) -> None:
    """N7 pin (W7 acceptance replay, item b): DX-23's key is v4-era ONLY.

    Wave-2's DX-23 added ``leg_citations`` to the hashed proof body without
    era-keying, which broke clause 8's byte-identity for every sealed v3
    proof (dropping the key reproduced the sealed 200a digest exactly). The
    field is era-keyed: a v3-era derivation over real records emits NO
    ``leg_citations`` key at all. The pin is the EXACT v3 top-level key set,
    so no future body key can join the pinned era unconditionally again.
    """

    from menagerie.crawler.partition_proof import (
        PARTITION_PROOF_SCHEMA_VERSION_V3,
        PARTITION_PROOF_SCHEMA_VERSION_V4,
    )

    repo, _rr, _ids, staged, _marker = _marker_only_rung(tmp_path)
    v3 = _derive(repo, staged, proof_schema_version=PARTITION_PROOF_SCHEMA_VERSION_V3)
    assert "leg_citations" not in v3
    # Clause 8 pins the era's whole hashed shape, forever: any key added to
    # (or dropped from) the v3 body moves every sealed 200a digest.
    assert set(v3) == {
        "schema_version",
        "campaign_id",
        "intake_snapshot_id",
        "intake_snapshot_sha256",
        "phase",
        "close_kind",
        "generation_id",
        "window",
        "selection_plan_sha256",
        "award_transition_census",
        "members",
        "partition_proof_sha256",
    }

    # The current era KEEPS the DX-23 citation: a leg close cites a real,
    # run_id-sorted list (era-honest empty invocation_ids allowed).
    v4 = _derive(repo, staged, proof_schema_version=PARTITION_PROOF_SCHEMA_VERSION_V4)
    assert isinstance(v4["leg_citations"], list)
    assert set(v4) == set(v3) | {"leg_citations"}


def test_n8_the_v4_differential_is_exactly_the_marker_only_cell(
    tmp_path: Path,
) -> None:
    """N8 (O R-8): v3 vs v4 over the same bytes differ in ONE member cell.

    Any second differing cell is a defect in the ruling's build, not in the
    record.
    """

    from menagerie.crawler.partition_proof import (
        PARTITION_PROOF_SCHEMA_VERSION_V3,
        PARTITION_PROOF_SCHEMA_VERSION_V4,
    )

    repo, _rr, ids, staged, _marker = _marker_only_rung(tmp_path)
    v3 = _derive(repo, staged, proof_schema_version=PARTITION_PROOF_SCHEMA_VERSION_V3)
    v4 = _derive(repo, staged, proof_schema_version=PARTITION_PROOF_SCHEMA_VERSION_V4)

    differing = [
        member
        for member in sorted(set(v3["members"]) | set(v4["members"]))
        if v3["members"].get(member) != v4["members"].get(member)
    ]
    assert differing == [ids[1]]
    assert v3["members"][ids[1]]["disposition"] == "withheld"
    assert v4["members"][ids[1]]["disposition"] == "continuing"
    for field in ("window", "generation_id", "close_kind", "phase", "selection_plan_sha256"):
        assert v3[field] == v4[field]


def test_attack_d_the_era_is_not_producer_choosable_at_the_door(
    tmp_path: Path,
) -> None:
    """Attack D (era laundering): a NEW close may not opt into the v3 era.

    Resident history re-derives under its own sealed era forever; the append
    door is where the era is law for NEW attestations. A fresh close binding
    a v3 proof refuses at append -- marker manufacture cannot buy back the
    pre-ruling licence by minting yesterday's schema.
    """

    from menagerie.crawler.close_object import CloseObjectError
    from menagerie.crawler.partition_proof import PARTITION_PROOF_SCHEMA_VERSION_V3

    repo, rr, ids, staged, _marker = _marker_only_rung(tmp_path)
    with pytest.raises(CloseObjectError) as excinfo:
        _close(repo, staged, proof_schema_version=PARTITION_PROOF_SCHEMA_VERSION_V3)
    assert "proof era" in str(excinfo.value)


# ---------------------------------------------------------------------------
# N9: the licensing law at the staging boundary
# ---------------------------------------------------------------------------


def test_n9_an_introduced_marker_only_member_refuses_incomplete_ladder(
    tmp_path: Path,
) -> None:
    """N9 (S probe 4, post-fix arm): introduced-and-continuing never licenses.

    A rung that INTRODUCES a member whose only evidence is a routing marker
    closes (clause 6) but licenses nothing for that member: successor staging
    refuses INCOMPLETE_LADDER naming it (clause 5). The licensing law applies
    at the staging boundary only -- members already proved by an ancestor
    rung block nothing.
    """

    from menagerie.crawler.close_object import CloseObjectError

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)

    first = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, first)
    seed_heads(rr, ids[:1], intake_snapshot_id=first.registration.snapshot_id)
    _close(repo, first, completion_status="complete")

    second = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-2")
    _open_leg(repo, second)
    _append_quarantine(rr, ids[1])  # the rung-2 INTRODUCED member, marker-only
    close = _close(repo, second)
    assert ids[1] in close["partition"]["continuing"]

    with pytest.raises(CloseObjectError) as excinfo:
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-3")
    assert "INCOMPLETE_LADDER" in str(excinfo.value)
    assert ids[1] in str(excinfo.value)


def test_a_selected_member_with_only_a_marker_continues_not_carries(
    tmp_path: Path,
) -> None:
    """The R9 seam under v4: selected + marker-only is a continuation.

    The hole->continuation cell is N4's ONE disclosed exception: a selected
    member with NO evidence refuses (the hole), and adding a live in-window
    routing marker opens a disclosed continuation instead -- never a carry,
    never a proof.
    """

    from menagerie.crawler.leg_generation import record_leg_plan

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    generation = _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    record_leg_plan(
        rr,
        campaign_id=CAMPAIGN,
        generation_id=generation.generation_id,
        run_id=RUN_ID,
        admitted_ids=[ids[1]],
        plan_payload={"fixture": "selected-continuation"},
        created_at="2026-08-01T00:00:05Z",
    )

    # The hole first: selected, nothing recorded -> refusal (T17).
    with pytest.raises(PartitionProofError):
        _close(repo, staged)

    # The marker opens the continuation -- the one disclosed exception.
    _append_quarantine(rr, ids[1])
    close = _close(repo, staged)
    assert ids[1] in close["partition"]["continuing"]
    assert ids[1] not in close["partition"]["proved_here"]


# ---------------------------------------------------------------------------
# N10: era totality -- the backfill close obeys the same law
# ---------------------------------------------------------------------------


def test_n10_a_backfill_close_classifies_a_live_routing_marker_as_continuing(
    tmp_path: Path,
) -> None:
    """N10 (ruling clause 10, O F4/R-7): the backfill era gains the same law.

    Under v4 a live routing marker proves nothing in the backfill era either:
    the member is a disclosed continuation, excluded from proof, while the
    same bytes under v3 stay WITHHELD forever (era totality).
    """

    from menagerie.crawler.partition_proof import (
        PARTITION_PROOF_SCHEMA_VERSION_V3,
        PARTITION_PROOF_SCHEMA_VERSION_V4,
    )

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _append_quarantine(rr, ids[0])

    def _backfill(version: str):
        return derive_partition_proof(
            rr,
            campaign_id=CAMPAIGN,
            intake_snapshot_id=staged.registration.snapshot_id,
            intake_snapshot_sha256=staged.registration.snapshot_sha256,
            phase=None,
            close_kind="backfill",
            generation_id="backfill-fixture",
            baselines=None,
            member_ids=members_of(repo, staged.registration.snapshot_id),
            proof_schema_version=version,
        )

    v3 = _backfill(PARTITION_PROOF_SCHEMA_VERSION_V3)
    assert v3["members"][ids[0]]["disposition"] == "withheld"
    assert ids[0] in proved_here(v3)

    v4 = _backfill(PARTITION_PROOF_SCHEMA_VERSION_V4)
    assert v4["members"][ids[0]]["disposition"] == "continuing"
    assert ids[0] not in proved_here(v4)


# ---------------------------------------------------------------------------
# N11: the exact close-status grammar (D-24)
# ---------------------------------------------------------------------------


def test_n11_the_close_status_grammar_is_exact_and_closed() -> None:
    """N11/D-24: only enumerated spellings are closes; no prefix match.

    ``is_close_status("complete-with-continuations-typo")`` is False -- and so
    is every unenumerated spelling, including composites in the wrong suffix
    order that the old prefix match silently accepted.
    """

    from menagerie.crawler.close_object import CLOSE_STATUS_SPELLINGS, is_close_status

    expected = frozenset(
        {
            "complete",
            "complete-with-deferrals",
            "terminal-partition-complete",
            "terminal-partition-complete-with-deferrals",
            "terminal-partition-complete-with-review-holds",
            "terminal-partition-complete-with-refusal-parks",
            "terminal-partition-complete-with-deferrals-with-review-holds",
            "terminal-partition-complete-with-deferrals-with-refusal-parks",
            "terminal-partition-complete-with-review-holds-with-refusal-parks",
            "terminal-partition-complete-with-deferrals-with-review-holds-with-refusal-parks",
        }
    )
    assert CLOSE_STATUS_SPELLINGS == expected
    for core in sorted(expected):
        assert is_close_status(core), core
        assert is_close_status(f"phase-{core}:metadata"), core

    for spelling in (
        "complete-with-continuations",
        "complete-with-continuations-typo",
        "completed",
        "complete-with-deferralsx",
        "terminal-partition-complete-with-refusal-parks-with-review-holds",
        "terminal-partition-completely",
        "phase-complete:",
        "phase-:metadata",
        "phase-completed:metadata",
        "interrupted:shutdown",
        "",
    ):
        assert not is_close_status(spelling), spelling


# ---------------------------------------------------------------------------
# N13: the write-site census (owner of the out-of-ledger laundering residual)
# ---------------------------------------------------------------------------


def test_n13_every_production_quarantine_write_site_is_enumerated_no_verdict(
    tmp_path: Path,
) -> None:
    """N13: the closed write-site census, BL-12 made mechanical.

    The honest residual all three labs named: a seam that OWED a park.refusal
    writing a quarantine row (or nothing) is detectable by no close predicate.
    It is owned HERE: every production call site of the quarantine append is
    enumerated, and each site's surrounding text must state the no-verdict
    contract. A new site fails this census until it is classified in the same
    reviewed change.
    """

    crawler_root = Path(__file__).resolve().parent.parent
    sites: dict[str, int] = {}
    for source in sorted(crawler_root.glob("*.py")) + sorted((crawler_root / "tools").glob("*.py")):
        text = source.read_text(encoding="utf-8")
        count = text.count("append_scheduler_quarantine_row(")
        if source.name == "scheduler_parks.py":
            count -= 1  # its own definition
        if count:
            sites[source.name] = count
    assert sites == {"driver_admission.py": 1}, (
        "the production quarantine write-site census moved; classify the new "
        f"site against BL-12 in the same reviewed change: {sites}"
    )

    # The one production site sits on a no-model-verdict code path and says so.
    text = (crawler_root / "driver_admission.py").read_text(encoding="utf-8")
    anchor = text.index("append_scheduler_quarantine_row(")
    window = text[max(0, anchor - 4000) : anchor + 4000]
    assert "no verdict" in window


# ---------------------------------------------------------------------------
# N14: production-manifestation rehearsal -- zero human acts
# ---------------------------------------------------------------------------


def test_n14_a_scheduler_fault_closes_and_re_drives_with_zero_human_acts(
    tmp_path: Path,
) -> None:
    """N14: fault -> typed event -> disclosed continuation -> automatic re-drive.

    The PLAN W7 acceptance line: a production manifestation emits ONE typed
    operational event, the close completes with the continuation disclosed,
    and the member is retry-eligible next leg -- no operator void, no human
    ruling, anywhere on the path (ruling clause 7).
    """

    from menagerie.crawler.scheduler_parks import (
        load_scheduler_quarantine_ledger,
        scheduler_quarantine_released,
    )

    repo, rr, ids, staged, marker = _marker_only_rung(tmp_path)
    _append_charged_park_event(rr, ids[1], str(marker["work_id"]), "2026-08-01T00:00:11Z")

    close = _close(repo, staged)
    assert ids[1] in close["partition"]["continuing"]

    # Zero human acts: the ledger holds exactly one row and no release.
    ledger = load_scheduler_quarantine_ledger([scheduler_quarantine_path(rr)])
    assert [str(row["row_sha256"]) for row in ledger] == [str(marker["row_sha256"])]

    # The automatic re-drive: one charged park is under the recidivism limit,
    # so the unchanged work generation is retry-eligible next leg.
    from menagerie.crawler.recordio import operational_events_path, scan_jsonl

    events = scan_jsonl(operational_events_path(rr))
    assert scheduler_quarantine_released(marker, events, ledger_rows=ledger)
    disclosures = {entry["stable_id"]: entry for entry in close["currency"]["continuing_members"]}
    assert disclosures[ids[1]]["held"] is False
    assert disclosures[ids[1]]["charged_parks"] == 1


# ---------------------------------------------------------------------------
# attack F: licence-side laundering dies at the gate recomputation
# ---------------------------------------------------------------------------


def test_attack_f_a_forged_withheld_citation_for_a_marker_only_member_refuses(
    tmp_path: Path,
) -> None:
    """Attack F: a v4 proof CLAIMING the pre-ruling withheld cell is refused.

    The exclusion re-derives from the sealed proof's own recorded evidence:
    a forged close whose proof claims WITHHELD (the licensing disposition)
    for a member the derivation classifies CONTINUING cannot recompute, and
    the staging gate refuses it with the per-member diff.
    """

    from menagerie.crawler.close_object import CloseObjectError
    from menagerie.crawler.partition_proof import PARTITION_PROOF_SCHEMA_VERSION_V4

    repo, rr, ids, staged, marker = _marker_only_rung(tmp_path)
    support = _support()
    honest = _derive(repo, staged, proof_schema_version=PARTITION_PROOF_SCHEMA_VERSION_V4)
    citations = {member: dict(entry) for member, entry in honest["members"].items()}
    citations[ids[1]] = {
        "disposition": "withheld",
        "evidence": dict(citations[ids[1]]["evidence"]),
    }
    support.forge_fixture_close(
        rr,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_citations=citations,
        generation=_open_leg(repo, staged),
        proof_schema_version=PARTITION_PROOF_SCHEMA_VERSION_V4,
    )

    with pytest.raises(CloseObjectError) as excinfo:
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    message = str(excinfo.value)
    assert "does not equal the gate's recomputation" in message or "withheld" in message
