"""RC-1 behaviors 2-5 on the fake-lane driver: the ``crawl`` invocation path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.configuration_canary import (
    PACED_AUTHOR_WIDTH,
    PACED_TERMINALS,
    canary_ledger_path,
)
from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.crawl_order import load_crawl_order
from menagerie.crawler.crawl_preflight import InvocationRefused
from menagerie.crawler.driver import CrawlerDriver
from menagerie.crawler.leg_generation import load_leg_events
from menagerie.crawler.pause_authority import sprint_pause_path
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.crawl_support import (
    CAMPAIGN,
    build_crawl_repo,
    crawl_driver,
    plenty_of_disk,
    records_root_of,
    root_snapshot,
)


@pytest.fixture(autouse=True)
def _plenty_of_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The disk arm is exercised explicitly; every other test has room."""

    monkeypatch.setattr(CrawlerDriver, "_crawl_disk_free", staticmethod(plenty_of_disk))
    monkeypatch.delenv("MENAGERIE_ENVIRONMENT_STATE_ROOT", raising=False)
    monkeypatch.delenv("MENAGERIE_TMP_ROOT", raising=False)


def _events(driver: CrawlerDriver, kind: OperationalEventKind) -> list[dict[str, Any]]:
    return [
        row
        for row in scan_jsonl(driver.paths.operational_ledger)
        if row.get("event_kind") == kind.value
    ]


def _leg_events(repo_root: Path, event: str) -> list[dict[str, Any]]:
    return [row for row in load_leg_events(records_root_of(repo_root)) if row["event"] == event]


def _model_rows(driver: CrawlerDriver) -> list[dict[str, Any]]:
    return list(scan_jsonl(driver.paths.ledgers.models))


# ---------------------------------------------------------------------------
# Behavior 3: the in-process preflight refuses typed, before any spend.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_missing_gate_verdict_refuses_typed_before_any_span_or_spend(tmp_path: Path) -> None:
    """closure-green refuses; an invocation-refused event lands; no leg_started."""

    repo_root = build_crawl_repo(tmp_path, members=3, green_verdicts=False)
    driver = crawl_driver(tmp_path, repo_root, limit=3)
    with pytest.raises(InvocationRefused) as refused:
        driver.run()
    assert refused.value.arm == "closure-green"
    events = _events(driver, OperationalEventKind.INVOCATION_REFUSED)
    assert len(events) == 1
    details = events[0]["details"]
    assert details["arm"] == "closure-green"
    assert details["campaign_id"] == CAMPAIGN
    assert details["preflight"]["refusing_arm"] == "closure-green"
    arm_names = [arm["name"] for arm in details["preflight"]["arms"]]
    assert arm_names[:2] == ["driver-lock", "campaign-binding"]
    assert events[0]["status"] == "refused:invocation"
    # Nothing was bought, no span opened, no leg_open row, no campaign failure.
    assert _leg_events(repo_root, "leg_started") == []
    assert _leg_events(repo_root, "leg_open") == []
    assert _model_rows(driver) == []
    assert _events(driver, OperationalEventKind.CAMPAIGN_HEALTH) == []


@pytest.mark.smoke
def test_pause_sentinel_refuses_typed(tmp_path: Path) -> None:
    """The sprint-pause sentinel is an invocation arm, evaluated before the lifecycle."""

    repo_root = build_crawl_repo(tmp_path, members=3)
    sentinel = sprint_pause_path(records_root_of(repo_root))
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(
        json.dumps(
            {
                "schema_version": "menagerie.crawler.sprint-pause.v1",
                "held": True,
                "reason": "operator hold",
                "expires_at": "2099-01-01T00:00:00Z",
                "default_action": "stay-paused",
            }
        )
    )
    driver = crawl_driver(tmp_path, repo_root, limit=3)
    with pytest.raises(InvocationRefused) as refused:
        driver.run()
    assert refused.value.arm == "pause-sentinel"
    assert _leg_events(repo_root, "leg_started") == []


@pytest.mark.smoke
def test_disk_floor_arm_carries_the_five_gib_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free space above the production floor but inside the margin refuses."""

    from menagerie.crawler.crawl_preflight import DISK_FLOOR_MARGIN_BYTES
    from menagerie.crawler.envs import DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES

    repo_root = build_crawl_repo(tmp_path, members=3)
    inside_margin = DEFAULT_ENVIRONMENT_DISK_FLOOR_BYTES + DISK_FLOOR_MARGIN_BYTES // 2
    monkeypatch.setattr(
        CrawlerDriver, "_crawl_disk_free", staticmethod(lambda _path: inside_margin)
    )
    driver = crawl_driver(tmp_path, repo_root, limit=3)
    with pytest.raises(InvocationRefused) as refused:
        driver.run()
    assert refused.value.arm == "disk-floor"
    assert str(DISK_FLOOR_MARGIN_BYTES) in refused.value.detail


# ---------------------------------------------------------------------------
# Behaviors 2 and 4: order + limit, leg_open as the record, typed exit.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_crawl_admits_the_order_prefix_and_exits_work_exhausted(tmp_path: Path) -> None:
    """Fresh members inside --limit are authored to terminals; the exit is typed."""

    repo_root = build_crawl_repo(tmp_path, members=6)
    driver = crawl_driver(tmp_path, repo_root, limit=4)
    result = driver.run()
    assert result.status == "work_exhausted"
    assert result.paused_reason is None
    records_root = records_root_of(repo_root)
    order = load_crawl_order(records_root, CAMPAIGN)
    assert order is not None
    scope = set(order.stable_ids[:4])
    recorded = {row["stable_id"] for row in _model_rows(driver)}
    assert recorded == scope
    # The leg_open row is the invocation record: root snapshot, no purpose.
    opens = _leg_events(repo_root, "leg_open")
    assert len(opens) == 1
    snapshot = root_snapshot(repo_root)
    assert opens[0]["intake_snapshot_sha256"] == snapshot.snapshot_sha256
    assert "leg_purpose" not in opens[0]
    ended = _leg_events(repo_root, "leg_ended")
    assert [row["exit_class"] for row in ended] == ["work_exhausted"]
    # No close, no partition proof.
    assert not (records_root / "intake" / "leg-closes.jsonl").exists()


@pytest.mark.smoke
def test_crawl_records_its_limit_and_preflight_census(tmp_path: Path) -> None:
    """The limit is an operational event bound to the order; the census is durable."""

    repo_root = build_crawl_repo(tmp_path, members=5)
    driver = crawl_driver(tmp_path, repo_root, limit=4)
    assert driver.run().status == "work_exhausted"
    order = load_crawl_order(records_root_of(repo_root), CAMPAIGN)
    assert order is not None
    limits = _events(driver, OperationalEventKind.CRAWL_LIMIT)
    assert len(limits) == 1
    assert limits[0]["details"]["limit"] == 4
    assert limits[0]["details"]["first"] is True
    assert limits[0]["details"]["order_sha256"] == order.order_sha256
    state = json.loads(driver.paths.driver_state.read_text())
    assert state["status"] == "work_exhausted"
    assert state["crawl"]["limit"] == 4
    assert state["crawl"]["preflight"]["refused"] is False
    arm_names = [a["name"] for a in state["crawl"]["preflight"]["arms"]]
    assert arm_names == [
        "driver-lock",
        "campaign-binding",
        "secrets",
        "disk-floor",
        "pause-sentinel",
        "environment-rotation",
        "cas-completeness",
        "lockless-intent",
        "cli-pin",
        "closure-green",
    ]
    unarmed = {a["name"] for a in state["crawl"]["preflight"]["arms"] if not a["armed"]}
    assert unarmed == {"secrets", "cas-completeness", "cli-pin"}


@pytest.mark.smoke
def test_raising_the_limit_admits_only_the_newly_widened_rows(tmp_path: Path) -> None:
    """WS-1 settles recorded members; widening admits exactly the next rows."""

    repo_root = build_crawl_repo(tmp_path, members=6)
    first = crawl_driver(tmp_path, repo_root, limit=3, run_id="run-one")
    assert first.run().status == "work_exhausted"
    rows_after_first = len(_model_rows(first))
    second = crawl_driver(tmp_path, repo_root, limit=5, run_id="run-two")
    assert second.run().status == "work_exhausted"
    order = load_crawl_order(records_root_of(repo_root), CAMPAIGN)
    assert order is not None
    recorded = {row["stable_id"] for row in _model_rows(second)}
    assert recorded == set(order.stable_ids[:5])
    assert len(_model_rows(second)) - rows_after_first == 2
    limits = _events(second, OperationalEventKind.CRAWL_LIMIT)
    assert [row["details"]["limit"] for row in limits] == [3, 5]
    assert limits[1]["details"]["previous_limit"] == 3
    # The same generation continued (one leg_open, two leg_run rows).
    assert len(_leg_events(repo_root, "leg_open")) == 1
    assert len(_leg_events(repo_root, "leg_run")) == 2


@pytest.mark.smoke
def test_narrowing_the_limit_refuses_typed(tmp_path: Path) -> None:
    """Widening only raises N: a lower limit is an invocation refusal."""

    repo_root = build_crawl_repo(tmp_path, members=6)
    assert crawl_driver(tmp_path, repo_root, limit=4, run_id="run-one").run().status == (
        "work_exhausted"
    )
    narrower = crawl_driver(tmp_path, repo_root, limit=2, run_id="run-two")
    with pytest.raises(InvocationRefused) as refused:
        narrower.run()
    assert refused.value.arm == "crawl-limit"
    refusals = _events(narrower, OperationalEventKind.INVOCATION_REFUSED)
    assert refusals[-1]["details"]["arm"] == "crawl-limit"
    # Repeating the standing limit is not a change and records nothing new.
    same = crawl_driver(tmp_path, repo_root, limit=4, run_id="run-three")
    assert same.run().status == "work_exhausted"
    assert len(_events(same, OperationalEventKind.CRAWL_LIMIT)) == 1


@pytest.mark.smoke
def test_no_limit_means_the_whole_order(tmp_path: Path) -> None:
    """Omitting --limit crawls every order row."""

    repo_root = build_crawl_repo(tmp_path, members=4)
    driver = crawl_driver(tmp_path, repo_root, limit=None)
    assert driver.run().status == "work_exhausted"
    snapshot = root_snapshot(repo_root)
    assert {row["stable_id"] for row in _model_rows(driver)} == {
        item.stable_id for item in snapshot.items
    }


# ---------------------------------------------------------------------------
# Behavior 5: paced start.
# ---------------------------------------------------------------------------


def test_paced_start_caps_width_then_emits_the_census_and_lifts(tmp_path: Path) -> None:
    """A new identity caps at 4 until 20 terminals exist; the census row then lands."""

    repo_root = build_crawl_repo(tmp_path, members=PACED_TERMINALS + 6)
    requested = PACED_AUTHOR_WIDTH + 3
    driver = crawl_driver(
        tmp_path, repo_root, limit=PACED_TERMINALS + 4, author_concurrency=requested
    )
    observed_widths: list[int] = []
    original_tick = CrawlerDriver._paced_start_tick

    def recording_tick(self: CrawlerDriver) -> None:
        observed_widths.append(int(self.config.author_concurrency))
        original_tick(self)

    CrawlerDriver._paced_start_tick = recording_tick  # type: ignore[method-assign]
    try:
        result = driver.run()
    finally:
        CrawlerDriver._paced_start_tick = original_tick  # type: ignore[method-assign]
    assert result.status == "work_exhausted"
    # The first rounds ran capped; the cap lifted back to the requested width.
    assert observed_widths[0] == PACED_AUTHOR_WIDTH
    assert driver.config.author_concurrency == requested
    assert getattr(driver, "_paced_start") is None
    rows = [
        json.loads(line)
        for line in canary_ledger_path(records_root_of(repo_root)).read_text().splitlines()
        if line.strip()
    ]
    assert [row["event"] for row in rows] == ["paced_start", "configuration_canary"]
    assert rows[0]["identity_sha256"] == rows[1]["identity_sha256"]
    assert rows[0]["cap_author_width"] == PACED_AUTHOR_WIDTH
    census = rows[1]["census"]
    assert census["new_terminals"] >= PACED_TERMINALS
    assert set(census) >= {
        "publication_rate",
        "accept_rate",
        "storm_fuse",
        "mean_session_wall_seconds",
    }
    events = _events(driver, OperationalEventKind.CONFIGURATION_CANARY)
    assert len(events) == 1
    assert events[0]["details"]["author_width_capped"] == PACED_AUTHOR_WIDTH
    assert events[0]["details"]["author_width_lifted_to"] == requested
    assert events[0]["details"]["canary_row_sha256"] == rows[1]["row_sha256"]


def test_a_canaried_identity_runs_at_full_width_and_a_new_one_paces_again(
    tmp_path: Path,
) -> None:
    """The newest canary row for the identity lifts the cap; a model change re-paces."""

    repo_root = build_crawl_repo(tmp_path, members=PACED_TERMINALS + 8)
    requested = PACED_AUTHOR_WIDTH + 2
    first = crawl_driver(
        tmp_path, repo_root, limit=PACED_TERMINALS + 2, author_concurrency=requested, run_id="run-a"
    )
    assert first.run().status == "work_exhausted"
    # Same identity: the invocation is not paced at all.
    second = crawl_driver(
        tmp_path, repo_root, limit=PACED_TERMINALS + 4, author_concurrency=requested, run_id="run-b"
    )
    second.run()
    assert getattr(second, "_paced_start") is None
    rows = [
        json.loads(line)
        for line in canary_ledger_path(records_root_of(repo_root)).read_text().splitlines()
        if line.strip()
    ]
    assert [row["event"] for row in rows] == ["paced_start", "configuration_canary"]
    # A different author model is a new identity: a fresh paced_start opens.
    third = crawl_driver(
        tmp_path, repo_root, limit=PACED_TERMINALS + 6, author_concurrency=requested, run_id="run-c"
    )
    third.config = replace(third.config, author_model="claude-opus-5")
    third.run()
    rows = [
        json.loads(line)
        for line in canary_ledger_path(records_root_of(repo_root)).read_text().splitlines()
        if line.strip()
    ]
    assert [row["event"] for row in rows][:3] == ["paced_start", "configuration_canary", "paced_start"]
    assert rows[2]["identity"]["author_model"] == "claude-opus-5"
