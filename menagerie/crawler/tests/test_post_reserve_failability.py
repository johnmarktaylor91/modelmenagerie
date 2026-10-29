"""PM-22/B2-X-V41: every paid lane FAILABLE post-reserve, inventoried and fired.

Launch-night lesson (2026-08-19) generalized: the author lane's post-reserve
death rehearsal exists (``test_session_death_dry_run_rehearsal``), but a paid
stage without a rehearsal is exactly the structural blindness that let the
reserve -> die -> retry composition first execute in production. This module
carries the CLOSED inventory tripwire -- the set of post-reserve paid-session
death stages is derived from the PRODUCTION vocabulary (``GateKind`` plus the
author attempt's two guard sessions), and every stage must name a collectable
rehearsal test that fires it -- and the checker-lane and source-guard firing
rehearsals themselves, each driving the verbatim production
``_retry_infrastructure_call`` against a real anchored spend ledger.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from menagerie.crawler.constants import GateKind
from menagerie.crawler.driver_admission import _CheckerSpendScope
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    Reservation,
    load_paid_session_rows,
)
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.tests.dry_run_support import (
    SessionDeathAuthor,
    SessionDeathChecker,
    parse_session_death_plan,
)
from menagerie.crawler.tests.test_session_death_dry_run_rehearsal import (
    _InjectionLadderRig,
)

_INVENTORY_PATH = Path(__file__).resolve().parent / "rehearsal_stage_inventory.json"
_TESTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _load_inventory() -> dict[str, Any]:
    """Load the committed closed stage inventory."""

    payload = json.loads(_INVENTORY_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "menagerie.crawler.rehearsal-stage-inventory.v1"
    return payload


@pytest.mark.smoke
def test_stage_inventory_is_closed_over_production_paid_stages() -> None:
    """The inventory covers EXACTLY the production paid stages, no more, no less.

    The required set is derived from production vocabulary, not restated by
    hand: a new ``GateKind`` member (a new paid checker stage) turns this RED
    until the stage gains a firing rehearsal and an inventory row. A vanished
    stage turns it RED the other way, so the inventory can never pad itself
    with rows nothing adjudicates.
    """

    required = {f"checker:{kind.value}" for kind in GateKind} | {
        "author:source-request-guard",
        "author:author-guard",
    }
    inventory = _load_inventory()
    stages = [str(row["stage"]) for row in inventory["stages"]]
    assert len(stages) == len(set(stages)), "duplicate stage rows"
    assert set(stages) == required


@pytest.mark.smoke
def test_every_stage_names_a_collectable_firing_rehearsal() -> None:
    """Each inventory row's ``fired_by`` nodes exist as real test functions.

    A row naming a deleted or renamed rehearsal is a disarmed tripwire: the
    stage would read as covered while nothing fires it.
    """

    for row in _load_inventory()["stages"]:
        fired_by = row["fired_by"]
        assert fired_by, f"stage {row['stage']} names no firing rehearsal"
        for node_id in fired_by:
            path_text, _, function = str(node_id).partition("::")
            function = function.split("[", 1)[0]
            test_path = _TESTS_ROOT / path_text
            assert test_path.is_file(), f"missing rehearsal module: {path_text}"
            tree = ast.parse(test_path.read_text(encoding="utf-8"))
            names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert function in names, (
                f"stage {row['stage']} names a rehearsal that does not exist: {node_id}"
            )


class _CheckerInjectionLadderRig(_InjectionLadderRig):
    """The author injection rig extended with real checker-session admission."""

    def _admit_checker_spend(
        self, _operational: Any, scope: _CheckerSpendScope
    ) -> Reservation:
        """Reserve one checker session exactly as production admission does."""

        return self._authority.reserve(
            scope={
                "campaign_id": "c1-mech",
                "generation_id": "generation-open",
                "lane": "checker",
                "dispatch_kind": scope.gate_kind,
                "work_generations": scope.work_generations,
                "gate_kind": scope.gate_kind,
                "checker_identity": None,
                "run_id": "run-one",
                "machine_id": "host-one",
            },
            policy=BudgetPolicy(
                quota_sessions=100,
                quota_wall_seconds=100_000.0,
                runaway_sessions=100,
                runaway_wall_seconds=100_000.0,
                reservation_wall_seconds=100.0,
            ),
        )


def _reserved_and_settled(authority: PaidSessionAuthority) -> tuple[list[str], dict]:
    """Return reserved spend ids in order and settled rows indexed by spend id."""

    rows = load_paid_session_rows(authority.path)
    reserved = [
        str(row["spend_id"]) for row in rows if row["event"] == "session_reserved"
    ]
    settled = {
        str(row["spend_id"]): row for row in rows if row["event"] == "session_settled"
    }
    return reserved, settled


@pytest.mark.smoke
@pytest.mark.parametrize("gate_kind", [kind.value for kind in GateKind])
def test_checker_session_death_fires_post_reserve(tmp_path: Path, gate_kind: str) -> None:
    """An injected checker death takes the real classified post-reserve path.

    The reservation exists before the guard dies, the death classifies through
    the production checker exit classifier into the bounded retry ladder, the
    dead session settles failed on its own spend id, and the retry buys a
    FRESH reservation -- for every paid checker stage, not just the author
    lane. This is B2-X-V41's checker half: before this rehearsal existed, no
    test could fail a paid checker session after reserving.
    """

    authority = PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-20T03:45:26Z",
    )
    rig = _CheckerInjectionLadderRig(authority)
    item = cast(
        WorkItem,
        SimpleNamespace(
            stable_id=f"m_checker_death_{gate_kind}",
            active_work_id="work-1",
            granted_attempts=None,
            intake=SimpleNamespace(name=f"m_checker_death_{gate_kind}"),
        ),
    )
    checker = SessionDeathChecker(parse_session_death_plan(f"{gate_kind}:1"))
    scope = _CheckerSpendScope(
        gate_kind=gate_kind,
        work_generations=((item.stable_id, item.active_work_id),),
        checker_identity=None,
    )

    def operation() -> str:
        """One checker session through the injection seams, minus the gate."""

        if checker._consume_death(gate_kind):  # noqa: SLF001 -- the seam under test
            checker._die_in_checker_guard_session(gate_kind, tmp_path)  # noqa: SLF001
        return "checked"

    result = rig._retry_infrastructure_call(  # noqa: SLF001
        operation,
        admission=("checker", item),
        operational=cast(JsonlLedger, SimpleNamespace(records=())),
        checker_scope=scope,
    )
    assert result == "checked"
    assert rig.waited, "the injected death must go through the ladder's bounded backoff"
    reserved, settled = _reserved_and_settled(authority)
    assert len(reserved) == 2 == len(set(reserved)), (
        "expected the dying checker session and one fresh retry reservation"
    )
    dead, retry = reserved
    assert settled[dead]["local_outcome"] == "failed"
    assert settled[dead]["custody_outcome"] == "exited-error"
    assert settled[retry]["local_outcome"] == "completed"
    for row in load_paid_session_rows(authority.path):
        if row["event"] == "session_reserved":
            assert row["dispatch_kind"] == gate_kind
            assert row["lane"] == "checker"


@pytest.mark.smoke
def test_source_guard_session_death_fires_post_reserve(tmp_path: Path) -> None:
    """The author attempt's FIRST guard session (source-request) is failable.

    The existing author rehearsal dies in the second (author) guard; the
    source-request guard was the one paid production stage of the author
    attempt no rehearsal could kill. Here it dies with the reservation live,
    classifies through ``classify_author_exit`` as retryable transport, and
    the retry heads a fresh spend id.
    """

    authority = PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-20T03:45:26Z",
    )
    rig = _InjectionLadderRig(authority)
    item = cast(
        WorkItem,
        SimpleNamespace(
            stable_id="m_source_guard_death",
            active_work_id="work-1",
            granted_attempts=None,
            intake=SimpleNamespace(name="m_source_guard_death"),
        ),
    )
    author = SessionDeathAuthor(
        parse_session_death_plan(f"{item.stable_id}:1"), stage="source-request"
    )

    def operation() -> str:
        """One author attempt through the injection seams, minus the artifact."""

        if author._consume_death(item):  # noqa: SLF001 -- the seam under test
            author._die_in_author_guard_session(item)  # noqa: SLF001
        return "authored"

    result = rig._retry_infrastructure_call(  # noqa: SLF001
        operation,
        admission=("author", item),
        operational=cast(JsonlLedger, SimpleNamespace(records=())),
    )
    assert result == "authored"
    assert rig.waited, "the injected death must go through the ladder's bounded backoff"
    reserved, settled = _reserved_and_settled(authority)
    assert len(reserved) == 2 == len(set(reserved)), (
        "expected the dying source guard and one fresh retry reservation"
    )
    dead, retry = reserved
    assert settled[dead]["local_outcome"] == "failed"
    assert settled[dead]["custody_outcome"] == "exited-error"
    assert settled[retry]["local_outcome"] == "completed"
