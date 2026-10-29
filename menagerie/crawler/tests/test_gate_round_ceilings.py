"""AG-6: every gate lane carries the metadata mechanism's absolute round ceiling.

200a evidence: the metadata lane's ceiling (BL-10 (b)) bounded churn, but the
terminal-disposition and fidelity lanes shipped uncapped -- m4257 burned 12
all-rejected terminal-disposition rounds across 14 days, and m1359/m5445 ran
7 fidelity rounds, because chargeability bounds author incorrigibility only:
a checker that keeps disclosing fresh complaints is never chargeable and
holds any lane open forever. These tests pin the closed-inventory extension
(PLAN_v6 AG-6):

* :data:`GATE_ROUND_CEILINGS` is CLOSED over :class:`GateKind` -- a new gate
  kind cannot ship uncapped, and an unknown kind fails closed;
* only NON-ACCEPTED rounds count, so a re-gate sweep that re-confirms
  acceptances (the 08-23 sweep re-adjudicated 37 accepted arms) can never
  park a healthy lineage;
* a fidelity lineage at the ceiling parks as ``checker-disclosure-debt``
  BEFORE buying another session and never mints a ``failed:*`` terminal;
* a terminal-disposition lineage at the ceiling parks at the shared lane
  front half before the reuse probe or any spend admission.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import menagerie.crawler.author_spend as author_spend_module
from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    gate_round_ceiling_state,
)
from menagerie.crawler.constants import (
    GATE_MAX_REPAIRS,
    GATE_ROUND_CEILINGS,
    METADATA_GATE_ROUND_CEILING,
    GateKind,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.tests.test_slice_f_driver import (
    CheckerOutcome,
    FidelityAuthor,
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)
from menagerie.crawler.tests.test_checker_lane_failure_scoping import _valve_rows

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# The closed inventory: a new GateKind cannot ship uncapped.
# ---------------------------------------------------------------------------


def test_every_gate_kind_has_a_round_ceiling() -> None:
    """The ceiling inventory is exactly the closed GateKind vocabulary."""

    assert set(GATE_ROUND_CEILINGS) == {kind.value for kind in GateKind}
    assert GATE_ROUND_CEILINGS[GateKind.METADATA_BATCH.value] == METADATA_GATE_ROUND_CEILING
    for kind, ceiling in GATE_ROUND_CEILINGS.items():
        # Initial round + the chargeable cap + margin: the ceiling backs the
        # cap up and must never preempt honest repair rounds.
        assert ceiling >= 1 + GATE_MAX_REPAIRS + 1, kind


def test_unknown_gate_kind_fails_closed() -> None:
    """An unregistered kind can never be counted against a guessed ceiling."""

    with pytest.raises((CheckerDispatchError, KeyError)):
        gate_round_ceiling_state((), gate_kind="brand-new-kind", stable_id="m1")


# ---------------------------------------------------------------------------
# The counting kernel: non-accepted rounds only, lineage-scoped.
# ---------------------------------------------------------------------------


def _td_row(verdict: str, *, stable_id: str = "m1", lineage: str = "work-m1") -> dict[str, Any]:
    return {
        "gate_kind": "terminal_disposition",
        "items": [
            {
                "stable_id": stable_id,
                "campaign_root_work_id": lineage,
                "terminal_disposition": {"verdict": verdict},
            }
        ],
    }


def test_ceiling_counts_only_non_accepted_rounds() -> None:
    """Accepted re-gates never count: a re-adjudication sweep parks nobody."""

    rows = [
        *[_td_row("accepted") for _ in range(10)],
        _td_row("rejected"),
        _td_row("rejected", stable_id="m-other"),
        _td_row("rejected", lineage="work-older-generation"),
    ]
    history, ceiling = gate_round_ceiling_state(
        rows,
        gate_kind="terminal_disposition",
        stable_id="m1",
        campaign_root_work_id="work-m1",
    )
    assert ceiling == GATE_ROUND_CEILINGS["terminal_disposition"]
    assert len(history) == 1  # only m1's own-lineage rejection counts


# ---------------------------------------------------------------------------
# Fidelity: a churning lineage parks at the ceiling, blame-free.
# ---------------------------------------------------------------------------


class RejectingFidelityChecker(ScriptedChecker):
    """Reject every fidelity round on drift plus a fresh field complaint."""

    def check_fidelity(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        outcome = super().check_fidelity(artifact, work_root, config)
        assert outcome.gate is not None
        item = outcome.gate["items"][0]
        fresh = f"claim_{self.fidelity_calls}"
        item["verdict"] = "inaccurate"
        item["fidelity"]["verdict"] = "major-drift"
        item["required_repairs"] = [f"repair {fresh}"]
        item["field_checks"][0]["field"] = fresh
        item["field_checks"][0]["verdict"] = "inaccurate"
        item["field_checks"][0]["reason"] = f"{fresh} is unsupported"
        item["field_checks"][0]["required_repair"] = f"repair {fresh}"
        return outcome


def test_rejecting_fidelity_lineage_parks_at_the_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop stops at the ceiling, parks as disclosure debt, mints no failed:*.

    The ceiling is lowered below the chargeable cap for this test so the
    ceiling (not the cap terminal) is what fires: the WIRING under test is
    the pre-purchase check and its park shape; the production VALUE is pinned
    by the closed-inventory test above.
    """

    ceiling = 2
    monkeypatch.setitem(GATE_ROUND_CEILINGS, "fidelity", ceiling)  # type: ignore[arg-type]
    monkeypatch.setattr(author_spend_module, "AUTHOR_SESSIONS_MAX", 6)
    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    checker = RejectingFidelityChecker()
    result = _driver(tmp_path, snapshot, author=FidelityAuthor(), checker=checker).run()

    assert result.status == "complete-with-deferrals"
    assert checker.fidelity_calls == ceiling, (
        "the ceiling must stop the fidelity lineage before buying another session"
    )
    rows = _valve_rows(tmp_path, snapshot)
    assert len(rows) == 1
    row = rows[0]
    assert row["stable_id"] == stable_id
    assert row["checker_lane"]["failure_class"] == "checker-disclosure-debt"
    assert row["checker_lane"]["gate_kind"] == "fidelity"
    paths = _paths(tmp_path, snapshot)
    models = list(scan_jsonl(paths.ledgers.models)) if paths.ledgers.models.exists() else []
    assert all(not str(record["status"]["code"]).startswith("failed:") for record in models), (
        "disclosure debt must never be minted as a model-blaming terminal"
    )


# ---------------------------------------------------------------------------
# Terminal disposition: at the ceiling, the shared lane front half parks
# before the reuse probe and before any session admission.
# ---------------------------------------------------------------------------


def test_terminal_lineage_at_ceiling_parks_without_buying(tmp_path: Path) -> None:
    """Six persisted rejections park the lineage; check_terminal must never fire."""

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id

    class NeverBuyChecker(ScriptedChecker):
        def check_terminal(
            self, artifact: Any, work_root: Path, config: DriverConfig
        ) -> CheckerOutcome:
            raise AssertionError("a lineage at the ceiling bought a terminal session")

    driver = _driver(tmp_path, snapshot, checker=NeverBuyChecker())
    from menagerie.crawler.driver_contracts import WorkItem  # local: mirror driver route
    from menagerie.crawler.routing import route_model

    intake = snapshot.items[0]
    item = WorkItem(
        intake,
        route_model(intake.to_model_requirements("torch")),
    )
    lineage = f"work-{stable_id}"
    ceiling = GATE_ROUND_CEILINGS["terminal_disposition"]
    persisted = tuple(
        _td_row("rejected", stable_id=stable_id, lineage=lineage) for _ in range(ceiling)
    )
    reducer = SimpleNamespace(gate_records=persisted)
    artifact = SimpleNamespace(campaign_root_work_id=lineage, proposal={"work_id": lineage})
    paths = _paths(tmp_path, snapshot)
    with JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        gate, pause = driver._obtain_terminal_disposition_gate(  # noqa: SLF001
            item, artifact, reducer, operational
        )
    assert gate is None and pause is None
    rows = _valve_rows(tmp_path, snapshot)
    assert len(rows) == 1
    assert rows[0]["stable_id"] == stable_id
    assert rows[0]["checker_lane"]["failure_class"] == "checker-disclosure-debt"
    assert rows[0]["checker_lane"]["gate_kind"] == "terminal_disposition"
