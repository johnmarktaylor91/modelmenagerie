"""W6: the supervised leg lifecycle -- every exit typed, every span paired.

R11/R12 (~28% of 200a wall burned by identical leg deaths, 39/43 legs
manually resumed): no leg_end event existed, no repeat-cause fuse, and an
orphaned death was invisible. Pinned here:

- every driver invocation that passes its leg-start refusals emits ONE
  durable ``leg_started`` and EVERY ``_run_locked`` exit path pairs it with
  exactly one typed ``leg_ended`` (returns and raises alike, BaseException
  included) carrying the closed :class:`LegExitClass`, PM-5's
  ``cause_fingerprint``, and ``terminal_delta``;
- the OB-M12 census: ``count(leg-config) == count(leg-outcome) + running``,
  with ``fault_uncontained`` the zero-in-a-healthy-rung tripwire;
- the boot reconciler closes kill-9 orphans ``orphaned_uncontained`` before
  a new leg opens (synth item 10's rehearsal), and is the ONLY lawful
  producer of that class;
- the PM-5 repeat-cause fuse: K=3 identical-cause zero-delta leg deaths
  REFUSE the next leg without ``--acknowledge-repeat-cause`` + rationale (a
  durable operator row); progress or a different cause resets it, a restart
  alone never does.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.constants import InvocationOrigin
from menagerie.crawler.driver_contracts import DriverIntegrationError, LegStartRefused
from menagerie.crawler.leg_exit_vocabulary import (
    RESTARTABLE_EXIT_CLASSES,
    STOP_CLASS_EVALUATOR_COVERAGE,
    LegExitClass,
    StopClass,
    cause_fingerprint,
    classify_exception,
    classify_result_status,
    leg_outcome_census,
    repeat_cause_fuse_tripped,
)
from menagerie.crawler.leg_generation import (
    LegGenerationError,
    load_leg_events,
    orphaned_leg_started_rows,
    record_leg_ended,
    record_leg_started,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

pytestmark = pytest.mark.smoke


def _records_root(tmp_path: Path, snapshot: Any) -> Path:
    return _paths(tmp_path, snapshot).ledgers.models.parent.parent


def _lifecycle_rows(tmp_path: Path, snapshot: Any) -> list[Mapping[str, Any]]:
    return [
        row
        for row in load_leg_events(_records_root(tmp_path, snapshot))
        if row.get("event") in ("leg_started", "leg_ended")
    ]


def _operational_events(tmp_path: Path, snapshot: Any) -> list[Mapping[str, Any]]:
    path = _paths(tmp_path, snapshot).operational_ledger
    return list(scan_jsonl(path)) if path.is_file() else []


# ---------------------------------------------------------------------------
# The closed vocabulary itself (FT-a).
# ---------------------------------------------------------------------------


def test_the_exit_vocabulary_is_closed_at_exactly_the_nine_classes() -> None:
    """Adding or removing an exit class is a reviewed vocabulary change."""

    assert {member.value for member in LegExitClass} == {
        "close_minted",
        "paused_typed",
        "operator_stop",
        "shutdown_typed",
        "fault_machinery",
        "fault_host",
        "fault_uncontained",
        "orphaned_uncontained",
        # Overhaul RC-1 (reviewed vocabulary change): the crawl invocation's
        # lawful typed end under its limit; never restartable, never fuse food.
        "work_exhausted",
    }
    assert {member.value for member in StopClass} == {
        "calibration_receipt_stale_or_absent",
        "certificate_stale_vs_tree",
        "partition_contradiction",
        "unrecordable_streak",
        "disk_floor_breach",
        "roots_unmounted_or_readonly",
        "secrets_absent_at_wake",
        "repeated_cause_leg_death",
        "paid_fuse_headroom_exhausted",
        "hold_sentinel_present",
        # SD-2 (reviewed vocabulary change): leg open refuses on outstanding
        # licensing debt / stale rotated axes without a durable operator ack.
        "licensing_debt_unacked",
    }
    # The evaluator's coverage is TOTAL over the vocabulary and disclosed:
    # an unevaluated class is visible, never silently assumed green.
    assert set(STOP_CLASS_EVALUATOR_COVERAGE) == {member.value for member in StopClass}
    # paused/close/operator exits never restart; the fault family does.
    assert "paused_typed" not in RESTARTABLE_EXIT_CLASSES
    assert "close_minted" not in RESTARTABLE_EXIT_CLASSES
    assert "operator_stop" not in RESTARTABLE_EXIT_CLASSES
    assert "work_exhausted" not in RESTARTABLE_EXIT_CLASSES


def test_cause_fingerprints_bind_types_never_message_text() -> None:
    """PM-5's identity is stable across message churn, distinct across causes."""

    first = cause_fingerprint(
        "fault_uncontained", exception_types=("builtins.RuntimeError",), evidence=""
    )
    second = cause_fingerprint(
        "fault_uncontained", exception_types=("builtins.RuntimeError",), evidence=""
    )
    other = cause_fingerprint(
        "fault_host", exception_types=("builtins.RuntimeError",), evidence=""
    )
    assert first == second
    assert first != other
    assert first.startswith("sha256:")


def test_result_and_exception_classification_is_total() -> None:
    """Every status family and exception shape lands in the closed set."""

    assert classify_result_status("complete") == "close_minted"
    assert classify_result_status("complete-with-deferrals") == "close_minted"
    assert classify_result_status("terminal-partition-complete") == "close_minted"
    assert classify_result_status("paused:usage-limit") == "paused_typed"
    assert classify_result_status("paused:review-checkpoint") == "paused_typed"
    assert classify_result_status("interrupted:shutdown") == "shutdown_typed"
    assert classify_result_status("work_exhausted") == "work_exhausted"
    assert classify_result_status("something-novel") == "fault_uncontained"

    assert classify_exception(RuntimeError("boom"))[0] == "fault_uncontained"
    assert classify_exception(OSError("disk"))[0] == "fault_host"
    assert classify_exception(KeyboardInterrupt())[0] == "operator_stop"
    refused = LegStartRefused(StopClass.HOLD_SENTINEL_PRESENT.value, "held")
    assert classify_exception(refused) == ("operator_stop", "hold_sentinel_present")
    secrets = LegStartRefused(StopClass.SECRETS_ABSENT_AT_WAKE.value, "absent")
    assert classify_exception(secrets)[0] == "fault_host"


# ---------------------------------------------------------------------------
# The lifecycle pairing on real legs.
# ---------------------------------------------------------------------------


def test_a_completing_leg_pairs_started_with_close_minted(tmp_path: Path) -> None:
    """The happy path: one span, one typed close, an exact OB-M12 census."""

    snapshot = _snapshot(tmp_path, count=1)
    result = _driver(tmp_path, snapshot).run()
    assert result.status == "complete"

    rows = _lifecycle_rows(tmp_path, snapshot)
    assert [row["event"] for row in rows] == ["leg_started", "leg_ended"]
    started, ended = rows
    assert ended["leg_started_sha256"] == started["row_sha256"]
    assert ended["exit_class"] == LegExitClass.CLOSE_MINTED.value
    assert ended["synthesized"] is False
    assert isinstance(ended["terminal_delta"], int) and ended["terminal_delta"] >= 1
    assert started["invocation_origin"] == InvocationOrigin.ORDINARY_RUN.value
    assert started["invocation_ordinal"] == 0

    census = leg_outcome_census(_operational_events(tmp_path, snapshot))
    assert census["leg_config"] == 1
    assert census["leg_outcome"] == 1
    assert census["running"] == 0
    assert census["by_exit_class"] == {"close_minted": 1}


def test_a_pausing_leg_ends_paused_typed(tmp_path: Path) -> None:
    """A typed autonomous pause preserves 'one leg' (B3-XQ-12): paused_typed."""

    snapshot = _snapshot(tmp_path, count=1)
    result = _driver(tmp_path, snapshot, review_at=1).run()
    assert result.status == "paused:review-checkpoint"

    rows = _lifecycle_rows(tmp_path, snapshot)
    assert [row["event"] for row in rows] == ["leg_started", "leg_ended"]
    assert rows[1]["exit_class"] == LegExitClass.PAUSED_TYPED.value


def test_an_uncontained_exception_still_writes_its_typed_exit(tmp_path: Path) -> None:
    """The catch-all: an untyped raise is fault_uncontained, never silence."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)

    def _explode(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("wholly untyped scheduler defect")

    driver._select_work = _explode  # noqa: SLF001 -- fault injection
    with pytest.raises(RuntimeError, match="wholly untyped"):
        driver.run()

    rows = _lifecycle_rows(tmp_path, snapshot)
    assert [row["event"] for row in rows] == ["leg_started", "leg_ended"]
    assert rows[1]["exit_class"] == LegExitClass.FAULT_UNCONTAINED.value
    census = leg_outcome_census(_operational_events(tmp_path, snapshot))
    assert census["running"] == 0
    assert census["by_exit_class"] == {"fault_uncontained": 1}


def test_a_leg_start_refusal_starts_no_span(tmp_path: Path) -> None:
    """A refused leg never started: no span, no census entry."""

    snapshot = _snapshot(tmp_path, count=1)
    records_root = _records_root(tmp_path, snapshot)
    control = records_root / "control"
    control.mkdir(parents=True, exist_ok=True)
    import json as _json

    (control / "sprint-pause.json").write_text(
        _json.dumps(
            {
                "format": "menagerie.crawler.sprint-pause.v1",
                "reason": "orchestrator hold",
                "paused_by": "orchestrator",
                "created_at": "2026-08-24T00:00:00Z",
                "expires_at": None,
                "default_action": "stay-paused",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DriverIntegrationError, match="sprint-pause hold"):
        _driver(tmp_path, snapshot).run()
    assert _lifecycle_rows(tmp_path, snapshot) == []


# ---------------------------------------------------------------------------
# The kill-9 rehearsal: orphans close before a new leg opens.
# ---------------------------------------------------------------------------


def test_kill_nine_rehearsal_closes_the_orphan_before_the_new_leg(
    tmp_path: Path,
) -> None:
    """Synth item 10: a span with no exit write is closed orphaned_uncontained."""

    snapshot = _snapshot(tmp_path, count=1)
    records_root = _records_root(tmp_path, snapshot)
    orphan = record_leg_started(
        records_root,
        campaign_id="c1-mech",
        generation_id="sha256:" + "a" * 64,
        run_id="run-killed",
        created_at="2026-08-23T00:00:00Z",
        invocation_origin="ordinary-run",
        invocation_ordinal=0,
        pid=424242,
    )

    result = _driver(tmp_path, snapshot).run()
    assert result.status == "complete"

    events = load_leg_events(records_root)
    orphan_closes = [
        row
        for row in events
        if row.get("event") == "leg_ended"
        and row.get("leg_started_sha256") == orphan["row_sha256"]
    ]
    assert len(orphan_closes) == 1
    close = orphan_closes[0]
    assert close["exit_class"] == LegExitClass.ORPHANED_UNCONTAINED.value
    assert close["synthesized"] is True
    assert close["run_id"] == "run-killed"
    # Nothing is left orphaned, and the live leg paired itself normally.
    assert orphaned_leg_started_rows(load_leg_events(records_root)) == ()


def test_only_the_reconciler_may_synthesize_orphan_closes(tmp_path: Path) -> None:
    """The validator refuses synthesized XOR orphaned violations."""

    records_root = tmp_path / "records"
    started = record_leg_started(
        records_root,
        campaign_id="c1-mech",
        generation_id="sha256:" + "b" * 64,
        run_id="run-x",
        created_at="2026-08-23T01:00:00Z",
        invocation_origin="ordinary-run",
        invocation_ordinal=0,
        pid=1,
    )
    with pytest.raises(LegGenerationError, match="synthesized"):
        record_leg_ended(
            records_root,
            campaign_id="c1-mech",
            generation_id="sha256:" + "b" * 64,
            run_id="run-x",
            created_at="2026-08-23T01:01:00Z",
            leg_started_sha256=str(started["row_sha256"]),
            exit_class=LegExitClass.ORPHANED_UNCONTAINED.value,
            cause_fingerprint=cause_fingerprint("orphaned_uncontained"),
            terminal_delta=0,
            invocation_origin="ordinary-run",
            synthesized=False,
        )
    with pytest.raises(LegGenerationError, match="outside the closed vocabulary"):
        record_leg_ended(
            records_root,
            campaign_id="c1-mech",
            generation_id="sha256:" + "b" * 64,
            run_id="run-x",
            created_at="2026-08-23T01:02:00Z",
            leg_started_sha256=str(started["row_sha256"]),
            exit_class="graceful-ish",
            cause_fingerprint=cause_fingerprint("fault_host"),
            terminal_delta=0,
            invocation_origin="ordinary-run",
        )


# ---------------------------------------------------------------------------
# The PM-5 repeat-cause fuse.
# ---------------------------------------------------------------------------


def _dead_leg(
    records_root: Path,
    index: int,
    *,
    fingerprint: str,
    terminal_delta: int = 0,
) -> None:
    started = record_leg_started(
        records_root,
        campaign_id="c1-mech",
        generation_id="sha256:" + "c" * 64,
        run_id=f"run-dead-{index}",
        created_at=f"2026-08-23T02:0{index}:00Z",
        invocation_origin="ordinary-run",
        invocation_ordinal=0,
        pid=100 + index,
    )
    record_leg_ended(
        records_root,
        campaign_id="c1-mech",
        generation_id="sha256:" + "c" * 64,
        run_id=f"run-dead-{index}",
        created_at=f"2026-08-23T02:0{index}:30Z",
        leg_started_sha256=str(started["row_sha256"]),
        exit_class=LegExitClass.FAULT_MACHINERY.value,
        cause_fingerprint=fingerprint,
        terminal_delta=terminal_delta,
        invocation_origin="ordinary-run",
    )


def test_the_repeat_cause_fuse_refuses_the_fourth_identical_death(
    tmp_path: Path,
) -> None:
    """K=3 identical-cause zero-delta deaths refuse the next leg typed."""

    snapshot = _snapshot(tmp_path, count=1)
    records_root = _records_root(tmp_path, snapshot)
    fingerprint = cause_fingerprint(
        "fault_machinery", exception_types=("builtins.TimeoutError",)
    )
    for index in range(3):
        _dead_leg(records_root, index, fingerprint=fingerprint)

    with pytest.raises(DriverIntegrationError, match="repeat-cause fuse") as refusal:
        _driver(tmp_path, snapshot).run()
    assert fingerprint in str(refusal.value)

    # The acknowledgment funds exactly one retry, durably.
    driver = _driver(tmp_path, snapshot)
    driver.config = replace(
        driver.config, acknowledge_repeat_cause="operator: provider outage confirmed over"
    )
    assert driver.run().status == "complete"
    acks = [
        event
        for event in _operational_events(tmp_path, snapshot)
        if event.get("details", {}).get("disposition") == "repeat-cause-fuse-acknowledged"
    ]
    assert len(acks) == 1
    assert acks[0]["details"]["rationale"].startswith("operator:")
    assert acks[0]["details"]["cause_fingerprint"] == fingerprint


def test_progress_or_a_different_cause_resets_the_fuse(tmp_path: Path) -> None:
    """Zero-delta discipline: minted terminals or a new cause reset the streak."""

    records_root = tmp_path / "records"
    fingerprint = cause_fingerprint(
        "fault_machinery", exception_types=("builtins.TimeoutError",)
    )
    _dead_leg(records_root, 0, fingerprint=fingerprint)
    _dead_leg(records_root, 1, fingerprint=fingerprint, terminal_delta=2)
    _dead_leg(records_root, 2, fingerprint=fingerprint)
    events = load_leg_events(records_root)
    assert repeat_cause_fuse_tripped(events) is None

    other = cause_fingerprint("fault_host", exception_types=("builtins.OSError",))
    _dead_leg(records_root, 3, fingerprint=fingerprint)
    _dead_leg(records_root, 4, fingerprint=other)
    events = load_leg_events(records_root)
    assert repeat_cause_fuse_tripped(events) is None

    _dead_leg(records_root, 5, fingerprint=other)
    _dead_leg(records_root, 6, fingerprint=other)
    events = load_leg_events(records_root)
    assert repeat_cause_fuse_tripped(events) == other


def test_supervised_restart_origin_is_attested_on_the_span(tmp_path: Path) -> None:
    """The ladder's continuation is durable: origin + ordinal on leg_started."""

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    driver.config = replace(
        driver.config,
        invocation_origin=InvocationOrigin.SUPERVISED_RESTART,
        invocation_ordinal=2,
    )
    assert driver.run().status == "complete"
    rows = _lifecycle_rows(tmp_path, snapshot)
    assert rows[0]["invocation_origin"] == "supervised-restart"
    assert rows[0]["invocation_ordinal"] == 2
    assert rows[1]["invocation_origin"] == "supervised-restart"
