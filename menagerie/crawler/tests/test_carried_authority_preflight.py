"""D-27 (LC-16): carried-authority preflight + require_links hardening.

Twelve 200a carries ride live park rows; ONE well-meant void bricks the next
close -- at leg END, after the spend. The LC-1 carry pin already proves the
close-side refusal; this preflight moves detection to LEG START through the
SAME live-withholding derivation the sealed proof uses. The B2-X-O10
park-void FREEZE stands exactly until this arm; with it merged, a void is
caught before the next leg spends anything.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.partition_proof import carried_authority_preflight
from menagerie.crawler.tests.rt3_support import (
    append_refusal_park,
    build_repo,
    campaign_member_ids,
    members_of,
    records_root_of,
    seed_heads,
    stage,
    void_refusal_park,
)
from menagerie.crawler.tests.test_close_v2_contract import (
    CAMPAIGN,
    _open_leg,
    _v2_support,
)


def _carried_park_root(tmp_path: Path) -> tuple[Path, Path, str, dict]:
    """One rung closed WITHHELD on a live refusal park, next rung staged."""

    repo = build_repo(tmp_path)
    records_root = records_root_of(repo)
    ids = campaign_member_ids(repo, CAMPAIGN)
    parked = ids[1]
    first = stage(repo, CAMPAIGN, stable_ids=ids[:2], label="rung-1")
    _open_leg(repo, first)
    seed_heads(records_root, ids[:1], intake_snapshot_id=first.registration.snapshot_id)
    park_row = append_refusal_park(records_root, parked)
    _v2_support().record_fixture_close(
        records_root,
        campaign_id=CAMPAIGN,
        intake_snapshot_id=first.registration.snapshot_id,
        intake_snapshot_sha256=first.registration.snapshot_sha256,
        member_ids=members_of(repo, first.registration.snapshot_id),
        completion_status="terminal-partition-complete-with-refusal-parks",
    )
    return repo, records_root, parked, park_row


@pytest.mark.smoke
def test_preflight_passes_while_the_carried_park_stays_live(tmp_path: Path) -> None:
    """A live carried authority is not a finding: the next leg opens freely."""

    _repo, records_root, _parked, _park_row = _carried_park_root(tmp_path)
    assert carried_authority_preflight(records_root, campaign_id=CAMPAIGN) == ()


@pytest.mark.smoke
def test_preflight_refuses_when_the_carried_park_is_voided(tmp_path: Path) -> None:
    """The D-27 acceptance: detection moves from leg END to PRE-SPEND.

    Voiding the park the predecessor close's WITHHELD member cites breaks the
    carry. The close-side refusal (LC-1 carry pin) fires hours later at close
    build; the preflight names the same member and the exact dead row NOW,
    with the RELEASE-never-VOID instruction in the finding.
    """

    _repo, records_root, parked, park_row = _carried_park_root(tmp_path)
    void_refusal_park(records_root, park_row)
    findings = carried_authority_preflight(records_root, campaign_id=CAMPAIGN)
    assert len(findings) == 1
    assert parked in findings[0]
    assert park_row["row_sha256"] in findings[0]
    assert "RELEASE" in findings[0]
    # The sanctioned drain: a void over a member THIS leg can select and
    # re-drive is the release path (WS-1 clause 5), never a brick.
    assert (
        carried_authority_preflight(
            records_root, campaign_id=CAMPAIGN, selectable_member_ids=(parked,)
        )
        == ()
    )


@pytest.mark.smoke
def test_preflight_is_vacuous_without_a_close(tmp_path: Path) -> None:
    """No predecessor close, nothing carried: a fresh campaign owes nothing."""

    repo = build_repo(tmp_path)
    assert (
        carried_authority_preflight(records_root_of(repo), campaign_id=CAMPAIGN) == ()
    )


@pytest.mark.smoke
def test_the_driver_arms_the_preflight_before_the_generation_opens() -> None:
    """Wiring pin: pre-spend means BEFORE open_leg_generation, typed."""

    source = (Path(__file__).resolve().parents[1] / "driver.py").read_text(
        encoding="utf-8"
    )
    preflight_at = source.index("carried_authority_preflight(")
    open_at = source.index("self._leg_generation = open_leg_generation(")
    assert preflight_at < open_at, "the D-27 arm must fire before the generation opens"
    refusal_at = source.index("StopClass.PARTITION_CONTRADICTION.value", preflight_at)
    assert refusal_at < open_at, "the refusal must be typed at the leg-start seam"


@pytest.mark.smoke
def test_a_linked_first_row_requires_links_everywhere(tmp_path: Path) -> None:
    """require_links hardening (O B-9): no link-free prefix on a linked root.

    Every close ledger this code mints links from row one, so an UNLINKED row
    appended to it must refuse -- the link-free historical-prefix tolerance
    exists for genuinely old roots only, and a linked first row proves this
    root is not one.
    """

    import json

    from menagerie.crawler.close_object import (
        CloseObjectError,
        close_object_path,
        load_close_objects,
    )
    from menagerie.crawler.identity import canonical_json_bytes

    _repo, records_root, _parked, _park_row = _carried_park_root(tmp_path)
    path = close_object_path(records_root)
    rows = load_close_objects([path])
    assert rows and rows[0].get("chain") is not None
    forged = {
        key: value for key, value in dict(rows[-1]).items() if key != "chain"
    }
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(forged) + b"\n")
    with pytest.raises(CloseObjectError, match="chain"):
        load_close_objects([path])
    # And the preflight consumes the same fail-closed read.
    with pytest.raises(CloseObjectError, match="chain"):
        carried_authority_preflight(records_root, campaign_id=CAMPAIGN)
    del json
