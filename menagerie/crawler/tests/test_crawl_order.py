"""RC-1 behavior 2: the persisted, content-addressed, append-once crawl order."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.crawl_order import (
    CrawlOrderError,
    PARTITION_ORDER_STRATUM,
    build_crawl_order,
    crawl_scope,
    ensure_crawl_order,
    load_crawl_order,
    order_path,
)
from menagerie.crawler.rung_staging import CARRIED_TIP_STRATUM
from menagerie.crawler.tests.crawl_support import CAMPAIGN, build_crawl_repo, records_root_of, root_snapshot
from menagerie.crawler.tests.test_rung_stratification import (
    _PRIOR_RUNG_NAMES,
    _ids_by_name,
    _partition_order,
    build_stratified_repo,
)

NOW = "2026-09-01T12:00:00Z"


@pytest.mark.smoke
def test_order_is_total_over_the_partition_and_content_addressed(tmp_path: Path) -> None:
    """Every partition member appears exactly once; the seal binds header + rows."""

    repo_root = build_crawl_repo(tmp_path, members=6, green_verdicts=False)
    records_root = records_root_of(repo_root)
    snapshot = root_snapshot(repo_root)
    order, composed = ensure_crawl_order(
        records_root,
        campaign_id=CAMPAIGN,
        created_at=NOW,
        expected_member_ids=frozenset(item.stable_id for item in snapshot.items),
        expected_snapshot_sha256=snapshot.snapshot_sha256,
    )
    assert composed is True
    assert sorted(order.stable_ids) == sorted(item.stable_id for item in snapshot.items)
    assert len(set(order.stable_ids)) == len(order.stable_ids)
    assert order.header["campaign_id"] == CAMPAIGN
    assert order.header["intake_snapshot_sha256"] == snapshot.snapshot_sha256
    assert order.header["seed"] == 20260901
    assert order.order_sha256.startswith("sha256:")
    path = order_path(records_root, CAMPAIGN)
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert lines[0]["kind"] == "header"
    assert lines[-1] == {"kind": "seal", "order_sha256": order.order_sha256}
    assert [row["ordinal"] for row in lines[1:-1]] == list(range(1, len(order.members) + 1))


@pytest.mark.smoke
def test_order_is_append_once_and_never_re_derived(tmp_path: Path) -> None:
    """A second use loads the SAME order; the file is never rewritten."""

    repo_root = build_crawl_repo(tmp_path, members=5, green_verdicts=False)
    records_root = records_root_of(repo_root)
    first, composed = ensure_crawl_order(records_root, campaign_id=CAMPAIGN, created_at=NOW)
    assert composed
    before = order_path(records_root, CAMPAIGN).read_bytes()
    second, composed_again = ensure_crawl_order(
        records_root, campaign_id=CAMPAIGN, created_at="2026-09-02T00:00:00Z", seed=1
    )
    assert composed_again is False
    assert second.order_sha256 == first.order_sha256
    assert second.stable_ids == first.stable_ids
    assert order_path(records_root, CAMPAIGN).read_bytes() == before


@pytest.mark.smoke
def test_tampered_order_refuses_typed(tmp_path: Path) -> None:
    """A rewritten member, a broken ordinal chain, or a foreign seal refuses."""

    repo_root = build_crawl_repo(tmp_path, members=4, green_verdicts=False)
    records_root = records_root_of(repo_root)
    ensure_crawl_order(records_root, campaign_id=CAMPAIGN, created_at=NOW)
    path = order_path(records_root, CAMPAIGN)
    lines = path.read_text().splitlines()
    member = json.loads(lines[1])
    member["stable_id"] = "m_forged"
    lines[1] = json.dumps(member, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(CrawlOrderError, match="seal"):
        load_crawl_order(records_root, CAMPAIGN)


@pytest.mark.smoke
def test_order_universe_is_checked_against_the_bound_snapshot(tmp_path: Path) -> None:
    """An order over a different member universe is a records-state repair."""

    repo_root = build_crawl_repo(tmp_path, members=4, green_verdicts=False)
    records_root = records_root_of(repo_root)
    snapshot = root_snapshot(repo_root)
    ensure_crawl_order(records_root, campaign_id=CAMPAIGN, created_at=NOW)
    with pytest.raises(CrawlOrderError, match="member universe"):
        load_crawl_order(
            records_root,
            CAMPAIGN,
            expected_member_ids=frozenset({*(i.stable_id for i in snapshot.items), "m_extra"}),
        )
    with pytest.raises(CrawlOrderError, match="composed over snapshot"):
        load_crawl_order(records_root, CAMPAIGN, expected_snapshot_sha256="sha256:" + "0" * 64)


@pytest.mark.smoke
def test_crawl_scope_is_the_order_prefix() -> None:
    """``--limit N`` is the first N order rows; None is the whole order."""

    from menagerie.crawler.crawl_order import CrawlOrder

    order = CrawlOrder(
        header={"campaign_id": CAMPAIGN},
        members=(("a", "x", ""), ("b", "x", ""), ("c", "x", "")),
        order_sha256="sha256:x",
    )
    assert crawl_scope(order, 2) == ("a", "b")
    assert crawl_scope(order, None) == ("a", "b", "c")
    assert crawl_scope(order, 99) == ("a", "b", "c")
    with pytest.raises(CrawlOrderError):
        crawl_scope(order, 0)


def test_lineage_tip_is_carried_first_then_the_stratified_step_then_partition_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over a campaign with a registered rung, the order = tip, composition, rest."""

    repo_root = build_stratified_repo(tmp_path, monkeypatch)
    records_root = records_root_of(repo_root)
    ids = _ids_by_name(repo_root)
    order = build_crawl_order(
        records_root, campaign_id=CAMPAIGN, stratified_width=4, created_at=NOW
    )
    tip_ids = {ids[name] for name in _PRIOR_RUNG_NAMES}
    partition = _partition_order(repo_root)
    # The tip leads the order, in partition order, whatever stratum evidence
    # each tip member carries (a tail stratum may have picked it on its own).
    leading = list(order.stable_ids[: len(tip_ids)])
    assert set(leading) == tip_ids
    assert leading == [sid for sid in partition if sid in tip_ids]
    assert any(stratum == CARRIED_TIP_STRATUM for _sid, stratum, _ev in order.members)
    # Then the stratified step (fresh bulk rows), then the partition remainder.
    after_tip = [stratum for _sid, stratum, _ev in order.members[len(tip_ids) :]]
    assert after_tip[0] == "ladder-bulk"
    assert sorted(order.stable_ids) == sorted(partition)
    remainder = [sid for sid, stratum, _ev in order.members if stratum == PARTITION_ORDER_STRATUM]
    assert remainder == [sid for sid in partition if sid in set(remainder)]
    assert order.header["composition"] == "stratified"
    assert order.header["stratified_width_effective"] >= 1
    assert order.header["lineage_tip"]["members"] == len(tip_ids)
