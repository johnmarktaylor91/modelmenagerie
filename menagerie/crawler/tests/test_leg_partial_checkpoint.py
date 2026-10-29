"""DX-26 (RATIFIED, RULING W7 clause 15): partial close on every leg exit.

Under the OPEN-6 default a close's EXISTENCE asserts the leg ended honestly.
Every other exit -- a pause, a typed shutdown, a fault -- leaves the immutable
nonterminal ``leg-partial.v1`` checkpoint instead: the D-20 preview payload at
exit, riding the append-only operational ledger. Structurally it can never be
a close: it never satisfies the closed status grammar, never enters the close
ledger, never joins the predecessor chain, settles nothing, and licenses
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.close_object import CloseObjectError, is_close_status
from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeChecker,
    FakeForward,
    FakePauseScheduler,
    _driver,
    _paths,
    _snapshot,
)


def _checkpoints(operational: Path) -> list[dict]:
    events = [
        json.loads(line)
        for line in operational.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.CLOSE_PREVIEW.value
        and dict(event.get("details") or {}).get("checkpoint") == "leg-partial.v1"
    ]


@pytest.mark.smoke
def test_a_paused_leg_exit_leaves_the_partial_checkpoint(tmp_path: Path) -> None:
    """The load-bearing arm: a NON-close exit still checkpoints, typed.

    A quota pause ends the leg with no close object; DX-26's ratified premise
    ("a close exists => the leg ended honestly") only holds if every such
    exit leaves the durable partial instead of silence -- the step-100 audit
    was planned from hand-tallied raw heads for exactly this gap.
    """

    snapshot = _snapshot(tmp_path)
    scheduler = FakePauseScheduler(_paths(tmp_path, snapshot).wakeup_root)
    result = _driver(
        tmp_path,
        snapshot,
        checker=FakeChecker(quota=True),
        forward=FakeForward(),
        pause_scheduler=scheduler,
    ).run()
    assert result.status == "paused:usage-limit"
    paths = _paths(tmp_path, snapshot)
    checkpoints = _checkpoints(paths.operational_ledger)
    assert len(checkpoints) == 1
    details = checkpoints[0]["details"]
    assert details["at"] == "leg-exit"
    assert details["exit_class"] == "paused_typed"
    assert details["leg_ended_sha256"].startswith("sha256:")
    assert details["generation_id"]
    assert details["verdict"] in {"would-refuse", "would-close", "preview-error"}


@pytest.mark.smoke
def test_a_closed_leg_carries_the_checkpoint_too(tmp_path: Path) -> None:
    """"Every leg exit" includes the honest close (close_minted)."""

    snapshot = _snapshot(tmp_path, count=2)
    bind_snapshot_to_campaign(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    assert driver.run().status == "complete"
    checkpoints = _checkpoints(driver.paths.operational_ledger)
    assert len(checkpoints) == 1
    assert checkpoints[0]["details"]["exit_class"] == "close_minted"


@pytest.mark.smoke
def test_the_checkpoint_can_never_be_a_close() -> None:
    """Structural nonterminality: outside the closed grammar, refused at build."""

    from menagerie.crawler.leg_generation import census_identities

    assert not is_close_status("leg-partial.v1")
    with pytest.raises(CloseObjectError, match="not a close"):
        from menagerie.crawler.close_object import build_close_object

        build_close_object(
            campaign_id="c1-mech",
            intake_snapshot_id="intake-partial",
            intake_snapshot_sha256="sha256:partial",
            phase=None,
            close_kind="leg",
            generation_id="sha256:" + "1" * 64,
            leg_open_row_sha256="sha256:" + "2" * 64,
            sources=census_identities(Path("/nonexistent")),
            partition_proof_sha256="sha256:" + "3" * 64,
            predecessor_close_sha256=None,
            run_id="pytest",
            machine_id="pytest",
            closed_at="2026-08-25T00:00:00Z",
            completion_status="leg-partial.v1",
            partition_arms={},
            currency_census={},
            authority_identities={},
        )
