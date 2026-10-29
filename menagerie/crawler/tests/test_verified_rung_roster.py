"""D-34 (OB-M6): the rung roster reads fail-closed, never one member short.

The launch monitor's tolerant roster read (raw ``items.jsonl`` lines,
unreadable lines skipped) silently yields 209 of 210 members on a torn file
-- a wrong denominator in every downstream census, with no signal.
:func:`menagerie.crawler.status.verified_rung_roster` is the shared
fail-closed reader: the snapshot loads through the verifying intake reader
and must agree with its registration's ``snapshot_sha256`` AND
``member_count``, or the read refuses typed. WS9's monitor consumes this
derivation instead of its tolerant scan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.status import RungRosterError, verified_rung_roster
from menagerie.crawler.tests.rt3_support import (
    build_repo,
    campaign_member_ids,
    records_root_of,
    stage,
)

pytestmark = pytest.mark.smoke

CAMPAIGN = "c3-classics"


def test_a_registered_roster_verifies_with_its_attested_count(tmp_path: Path) -> None:
    """Positive control: registration digest and count both corroborate."""

    repo = build_repo(tmp_path)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-1")

    roster = verified_rung_roster(records_root_of(repo), label="rung-1")
    assert roster.campaign_id == CAMPAIGN
    assert roster.snapshot_id == staged.registration.snapshot_id
    assert roster.snapshot_sha256 == staged.registration.snapshot_sha256
    assert roster.member_count == 3
    assert roster.member_ids == frozenset(ids[:3])


def test_a_torn_items_file_refuses_instead_of_yielding_one_less(tmp_path: Path) -> None:
    """The 209-of-210 shape: a torn roster is a refusal, never a shrink.

    The tolerant monitor scan drops the torn tail and hands back a smaller
    roster with no signal; the verified reader refuses with the snapshot
    named, because a wrong denominator poisons every census downstream.
    """

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-1")

    items_path = rr / "intake" / staged.registration.snapshot_id / "items.jsonl"
    payload = items_path.read_bytes()
    items_path.write_bytes(payload[: len(payload) - 20])  # tear the final row

    with pytest.raises(RungRosterError) as excinfo:
        verified_rung_roster(rr, label="rung-1")
    assert staged.registration.snapshot_id in str(excinfo.value)

    # The defect this reader replaces, documented while it exists: the
    # monitor's tolerant scan hands back a silently smaller roster from the
    # same bytes. When WS9 rewires the monitor onto verified_rung_roster,
    # the tolerant read disappears and this contrast arm goes with it.
    from menagerie.crawler.tools.launch_monitor import load_rung_roster

    try:
        tolerant = load_rung_roster(rr, label="rung-1")
    except Exception:
        tolerant = None  # the monitor already fails closed: nothing to contrast
    if tolerant is not None:
        assert len(tolerant.member_ids) < 3


def test_a_count_contradicting_registration_refuses(tmp_path: Path) -> None:
    """A registration attesting the wrong member_count refuses the roster."""

    from menagerie.crawler.rung_staging import (
        _write_subset_snapshot,
        append_rung_registration,
        build_rung_registration,
        resolve_verified_root,
        rung_registry_path,
    )

    repo = build_repo(tmp_path)
    rr = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    staged = stage(repo, CAMPAIGN, stable_ids=ids[:3], label="rung-1")

    # A fresh, internally valid subset snapshot whose REGISTRATION attests a
    # census it does not hold (the registry's rebind guard forbids forging a
    # second count onto an already-registered snapshot -- good -- so the
    # contradiction is minted on first registration).
    binding, _root = resolve_verified_root(rr, CAMPAIGN)
    snapshot, _created = _write_subset_snapshot(
        rr / binding.intake_path, tuple(ids[:2]), rr / "intake"
    )
    append_rung_registration(
        rung_registry_path(rr),
        build_rung_registration(
            campaign_id=CAMPAIGN,
            snapshot_id=snapshot.snapshot_id,
            snapshot_sha256=snapshot.snapshot_sha256,
            derived_from_snapshot_id=staged.registration.snapshot_id,
            member_count=210,  # attests a census the snapshot does not hold
            label="rung-1-forged",
            registered_by="pytest-forge",
            created_at="2026-08-16T00:00:01Z",
        ),
    )

    with pytest.raises(RungRosterError) as excinfo:
        verified_rung_roster(rr, label="rung-1-forged")
    assert "210" in str(excinfo.value)


def test_an_unknown_label_refuses_typed(tmp_path: Path) -> None:
    """Selecting a rung that was never registered is a typed refusal."""

    repo = build_repo(tmp_path)
    with pytest.raises(RungRosterError):
        verified_rung_roster(records_root_of(repo), label="never-registered")
