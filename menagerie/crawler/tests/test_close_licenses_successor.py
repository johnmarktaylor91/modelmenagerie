"""DX-22: close licenses its own successor -- parks are DISCLOSURE, not accounting.

R2 (PLAN_v6): stage_rung refuses on the sealed 200a close with a 212-of-210
accounting-arm overlap. The producer double-counts every LIVE park member: the
member's coverage lands in ``buckets`` (recorded head) or ``machine_deferred``
(recordless claimed row), and the SAME id is listed again in the
``grant_pending``/``refusal_parked`` arm. The multiset discipline ("every id in
exactly one recorded accounting arm", round-2 break 2b) is correct and STAYS --
the defect is that the park arms were ever counted as accounting. Under DX-22
they are disclosure arms: a census a human reads, ghost-checked against the
lineage, and excluded from the accounting multiset.

Both directions hold the line:

* the 200a shape (a live park coexisting with its member's coverage) LICENSES;
* a ghost id in a park arm, a double-listed member across ACCOUNTING arms, and
  a lying census at close BUILD all still refuse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.close_object import (
    CloseObjectError,
    accounted_members,
    assert_close_discloses_the_census,
    close_object_refusals,
    require_closed_lineage,
)
from menagerie.crawler.reducer import default_ledger_paths
from menagerie.crawler.tests.close_object_support import record_fixture_close


def _seed_ledger(records_root: Path) -> None:
    ledgers = default_ledger_paths(records_root)
    for name in ("models", "attempts", "gates", "artifacts"):
        path = getattr(ledgers, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(b"")


def _demand(records_root: Path, *, intake_snapshot_id: str, members: tuple[str, ...]):
    return require_closed_lineage(
        records_root,
        intake_snapshot_id=intake_snapshot_id,
        campaign_id="c1-mech",
        intake_snapshot_sha256=f"sha256:{intake_snapshot_id}",
        lineage_member_ids=frozenset(members),
        introduced_member_ids=frozenset(members),
        label="next-rung",
    )


@pytest.mark.smoke
def test_close_licenses_its_own_successor_with_live_parks(tmp_path: Path) -> None:
    """The 200a 212-of-210 shape: park disclosure never double-counts (DX-22).

    Every member is accounted exactly once by buckets; two of them ALSO carry a
    live park disclosure (one grant-pending review hold coexisting with its
    still-current review-held terminal, one refusal park) -- the exact producer
    shape that sealed the real 200a close. Pre-DX-22 the park arms were counted
    into the accounting multiset, the census read N+2 over an intake of N, and
    staging refused the truthful close. Disclosure arms account nothing: the
    close licenses.
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    members = ("m_bucketed", "m_held", "m_refparked")
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-parks",
        intake_snapshot_sha256="sha256:intake-parks",
        member_ids=members,
        seed_member_heads=True,
        completion_status="terminal-partition-complete-with-review-holds-with-refusal-parks",
        partition_overrides={
            "grant_pending": ["m_held"],
            "refusal_parked": ["m_refparked"],
        },
    )
    close = _demand(
        records_root,
        intake_snapshot_id="intake-parks",
        members=members,
    )
    # The multiset stays a multiset over the ACCOUNTING arms only: each member
    # exactly once, park disclosures contributing nothing.
    assert sorted(accounted_members(close)) == sorted(members)
    assert close_object_refusals(close) == ()


@pytest.mark.smoke
def test_a_ghost_in_a_park_disclosure_arm_still_refuses(tmp_path: Path) -> None:
    """Leaving the accounting multiset never exempts the arms from the census.

    The ghost tripwire used to ride the accounting multiset by accident (a
    park ghost inflated the count). Now it is explicit: a park arm naming an
    id the lineage snapshot does not contain is a lying census and refuses.
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    members = ("m_one", "m_two")
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-park-ghost",
        intake_snapshot_sha256="sha256:intake-park-ghost",
        member_ids=members,
        seed_member_heads=True,
        partition_overrides={"grant_pending": ["ghost-1"]},
    )
    with pytest.raises(CloseObjectError, match="ghost|not contain"):
        _demand(
            records_root,
            intake_snapshot_id="intake-park-ghost",
            members=members,
        )


@pytest.mark.smoke
def test_a_member_in_two_accounting_arms_still_refuses(tmp_path: Path) -> None:
    """The multiset check stays (never set-union): accounting doubles refuse."""

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    members = ("m_dupa", "m_dupb")
    record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-acct-double",
        intake_snapshot_sha256="sha256:intake-acct-double",
        member_ids=members,
        seed_member_heads=True,
        partition_overrides={
            "buckets": {"runs": ["m_dupa", "m_dupb"], "failed": ["m_dupa"]},
        },
    )
    with pytest.raises(CloseObjectError, match="exactly one"):
        _demand(
            records_root,
            intake_snapshot_id="intake-acct-double",
            members=members,
        )


@pytest.mark.smoke
def test_close_build_asserts_its_own_census(tmp_path: Path) -> None:
    """DX-22: refusals == () is asserted at close BUILD, not hours later.

    A lying census is a producer bug; the close build refuses to seal it
    instead of leaving the lie for successor staging to trip over at the end
    of the next leg. Disposition-negative: the assertion can only refuse.
    """

    records_root = tmp_path / "records"
    _seed_ledger(records_root)
    members = ("m_hona", "m_honb")
    close = record_fixture_close(
        records_root,
        campaign_id="c1-mech",
        intake_snapshot_id="intake-build-assert",
        intake_snapshot_sha256="sha256:intake-build-assert",
        member_ids=members,
        seed_member_heads=True,
    )
    # The honest census passes.
    assert_close_discloses_the_census(
        close,
        records_root=records_root,
        lineage_member_ids=frozenset(members),
        introduced_member_ids=frozenset(members),
    )
    # A lying one refuses with the disclosure findings, typed.
    lying = dict(close)
    lying["partition"] = {
        **dict(close["partition"]),
        "buckets": {"runs": ["m_hona"]},
    }
    with pytest.raises(CloseObjectError, match="lying census|accounts nothing"):
        assert_close_discloses_the_census(
            lying,
            records_root=records_root,
            lineage_member_ids=frozenset(members),
            introduced_member_ids=frozenset(members),
        )
