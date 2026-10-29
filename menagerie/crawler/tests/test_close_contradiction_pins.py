"""LC-1 pins: the head-vs-withholding coexistence matrix at close (post-ruling).

The ``partition_proof`` contradiction arm (an in-window head beside a live
in-window ``valve.scheduler-quarantine`` row refuses the close, `_resolve_member`)
killed 8 confirmed legs and bought 21 operator voids in the step-200 rung, yet
carried ZERO test coverage until these pins froze the pre-ruling semantics.

RULING W7 (PD-1) -- the KERNEL_DECISION of 2026-08-24 -- has now landed, and
exactly TWO pins flip in this same reviewed change, each citing the ruling
clause that makes the flip lawful. The other four pins HOLD, assertions
byte-untouched; they are the non-weakening anchors the ruling itself names
(clause 13: the ruling narrows exactly one refusal -- the routing-marker case
of the head-contradiction arm, which guarded no verdict -- and adds refusals
everywhere else).

Flipped cells (see each test's docstring for the clause citation):

* pin 1 -- head + live in-window ROUTING marker now closes ADJUDICATED on the
  head with a marker-superseded disclosure entry (ruling clause 2; the
  refusing sibling arm is pinned in ``test_pd1_continuation_ruling`` N1);
* pin 6 -- the marker-only member (the m_ec7b204ff3c04dc8853b shape) now
  closes as a disclosed CONTINUING member under proof schema v4: no proof, no
  licence, ever (ruling clause 5; OPEN-1 default).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.scheduler_parks import (
    append_scheduler_quarantine_row,
    append_scheduler_quarantine_void,
    build_scheduler_quarantine_row,
    build_scheduler_quarantine_void,
    scheduler_quarantine_path,
)
from menagerie.crawler.tests.rt3_support import (
    append_capacity_valve,
    build_repo,
    campaign_member_ids,
    members_of,
    records_root_of,
    seed_heads,
    stage,
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


def _append_quarantine(
    records_root: Path,
    stable_id: str,
    *,
    work_id: str = "w-quarantine",
    created_at: str = "2026-08-01T00:00:10Z",
    run_id: str = "pytest-close",
    reason_code: str = "no-durable-routing-progress",
):
    """Append one live scheduler-quarantine row through the real builder."""

    row = build_scheduler_quarantine_row(
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
    return append_scheduler_quarantine_row(scheduler_quarantine_path(records_root), row)


def _void_quarantine(records_root: Path, row) -> None:
    """Drain one quarantine row exactly like the 200a operator remedy."""

    append_scheduler_quarantine_void(
        scheduler_quarantine_path(records_root),
        build_scheduler_quarantine_void(
            void_of=str(row["row_sha256"]),
            reason="fixture operator drain",
            voided_by="pytest-operator",
            created_at="2026-08-01T00:00:20Z",
        ),
    )


def _close(repo: Path, staged, *, completion_status: str = "complete-with-deferrals"):
    """Close one fixture rung honestly through the production derivation."""

    support = _v2_support()
    return support.record_fixture_close(
        records_root_of(repo),
        campaign_id=CAMPAIGN,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_ids=members_of(repo, staged.registration.snapshot_id),
        completion_status=completion_status,
    )


def test_an_in_window_head_with_a_live_in_window_quarantine_refuses_the_close(
    tmp_path: Path,
) -> None:
    """FLIPPED (RULING W7 clause 2 -- the 200a leg killer, disarmed).

    Pre-ruling this cell refused the close to a human 21 times. The refusal
    guarded NO verdict: the excluded row is verdict-free by closed-set
    definition (``no-durable-routing-progress`` is in ROUTING_MARKER_REASONS)
    and the validated in-window head IS the window's rendered truth. The flip
    automates the operator's own rule, hand-applied 21 times and written into
    6 void texts ("superseded by a fully adjudicated head"). Four-part lens:
    no disposition upgraded, closed enumerated predicate, replay-derived,
    disclosed. The refusing sibling arm (a VERDICT-BEARING row beside the same
    head still refuses) is pinned in the same change -- N1 in
    ``test_pd1_continuation_ruling`` -- because the flip is not lawful
    without it [KERNEL_DECISION PD-1 section 5, pin 1].
    """

    from menagerie.crawler.partition_proof import load_partition_proof

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    contradicted = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:2], intake_snapshot_id=staged.registration.snapshot_id)
    marker = _append_quarantine(rr, contradicted)

    close = _close(repo, staged)
    assert contradicted in close["partition"]["adjudicated_here"]

    # The exclusion is DISCLOSED, never silent: the sealed proof's own
    # ADJUDICATED evidence carries the marker-superseded entry naming the
    # exact excluded row (ruling clause 2's disclosure requirement).
    proof = load_partition_proof(rr, str(close["partition_proof_sha256"]))
    entry = proof["members"][contradicted]
    assert entry["disposition"] == "adjudicated"
    superseded = entry["evidence"]["superseded_markers"]
    assert [item["row_sha256"] for item in superseded] == [str(marker["row_sha256"])]
    assert superseded[0]["reason_code"] == "no-durable-routing-progress"


def test_a_quarantine_below_the_window_floor_does_not_contradict_an_in_window_head(
    tmp_path: Path,
) -> None:
    """PIN (positive control): only IN-WINDOW rows reach the contradiction arm.

    A quarantine row appended below this leg's window floor is prior-era
    evidence; the in-window head stands and the member closes ADJUDICATED.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _append_quarantine(rr, member)  # pre-open: below the generation floor
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)

    close = _close(repo, staged)
    assert member in close["partition"]["adjudicated_here"]


def test_a_voided_quarantine_restores_the_close(tmp_path: Path) -> None:
    """PIN (the operator remedy, applied 21 times in 200a): a drain closes.

    Voiding the quarantine row removes it from the live set and the member
    closes ADJUDICATED on its head. Assertions byte-untouched from the
    pre-ruling pin. What changed is the drain's STANDING, not its effect:
    under RULING W7 clause 7 no close path requires a human ruling any more
    (marker liveness and retry budget are replay-derived, an exhausted member
    HOLDS while the close completes), so the drain is now a lawful operator
    act -- one exit among several -- rather than the forced exit the
    machinery used to extort 21 times.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    row = _append_quarantine(rr, member)
    _void_quarantine(rr, row)

    close = _close(repo, staged)
    assert member in close["partition"]["adjudicated_here"]


def test_an_in_window_ordinary_valve_is_superseded_by_an_in_window_head(
    tmp_path: Path,
) -> None:
    """PIN (the vocabulary asymmetry): ordinary valves never contradict a head.

    The five ordinary valves carry a head-supersession predicate ("the record
    wins"), so head + in-window capacity row closes ADJUDICATED. Only the
    scheduler-quarantine kind lacked that predicate -- the asymmetry all three
    labs root-caused as the whole defect. This pin is the ruling's own
    NON-WEAKENING ANCHOR (KERNEL_DECISION PD-1 section 5, pin 4): the ruling
    generalises exactly this cell to the quarantine kind, and a change that
    moves THIS pin proves the generalisation unsound.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    append_capacity_valve(rr, member)

    close = _close(repo, staged)
    assert member in close["partition"]["adjudicated_here"]


def test_an_in_window_ordinary_valve_over_a_pre_window_head_withholds(
    tmp_path: Path,
) -> None:
    """PIN (the BL-8 carve-out): valve over a PRE-window head is the accounting.

    A member whose head landed in an earlier rung and whose re-drive THIS leg
    ended parked closes WITHHELD citing the in-window valve row -- the row is
    the leg's only truthful statement about the member. The ruling leaves this
    carve-out intact; the fixture's member is CARRIED-not-introduced, so the
    pin holds under BOTH sides of OPEN-4 (KERNEL_DECISION PD-1 section 5,
    pin 5).
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    member = ids[0]

    first = stage(repo, CAMPAIGN, stable_ids=ids[:1], label="rung-1")
    _open_leg(repo, first)
    seed_heads(rr, ids[:1], intake_snapshot_id=first.registration.snapshot_id)
    _close(repo, first, completion_status="complete")

    second = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-2")
    _open_leg(repo, second)
    seed_heads(rr, ids[1:2], intake_snapshot_id=second.registration.snapshot_id)
    append_capacity_valve(rr, member)

    close = _close(repo, second)
    assert member in close["partition"]["withheld_here"]


def test_a_marker_only_member_closes_withheld_and_licenses_the_successor(
    tmp_path: Path,
) -> None:
    """FLIPPED (RULING W7 clause 5 -- the m_ec7b shape: no proof, no licence).

    The cited row's own recheck_hint says "no verdict was rendered" (the real
    200a row, verbatim). A row disclaiming a verdict cannot be settlement
    authority; pre-ruling this cell licensed rung width on a member nothing
    adjudicated -- the Sol B15 finding, confirmed against the final 200a
    proof JSON. Post-ruling (proof schema v4, OPEN-1 default): the member
    closes as a disclosed CONTINUING member -- it LEAVES ``withheld_here``,
    never enters ``proved_here``, and a licence boundary whose introduced set
    contains it refuses INCOMPLETE_LADDER naming the member. The flip ADDS a
    refusal and removes none [KERNEL_DECISION PD-1 section 5, pin 6].

    The pre-ruling semantics are not deleted -- they are PINNED FOREVER under
    proof schema v3 by ``test_pd1_continuation_ruling`` N7 (era clause 8).
    """

    from menagerie.crawler.close_object import CloseObjectError

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    marker_only = ids[1]

    staged = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    _open_leg(repo, staged)
    seed_heads(rr, ids[:1], intake_snapshot_id=staged.registration.snapshot_id)
    _append_quarantine(rr, marker_only)

    close = _close(repo, staged)
    assert marker_only not in close["partition"]["withheld_here"]
    assert marker_only not in close["partition"]["proved_here"]
    assert marker_only in close["partition"]["continuing"]

    # The no-verdict row carries NO licensing authority: the successor refuses
    # INCOMPLETE_LADDER naming the continuing member (ruling clause 5; the Sol
    # B15 finding, unanimously adopted).
    with pytest.raises(CloseObjectError) as excinfo:
        stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-2")
    assert "INCOMPLETE_LADDER" in str(excinfo.value)
    assert marker_only in str(excinfo.value)
