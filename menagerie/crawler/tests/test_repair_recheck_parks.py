"""BL-10 (CH-F4): the fidelity-repair metadata re-check parks, never terminalizes.

The batch metadata lane was explicitly rebuilt so a checker LANE failure can
never blame a model (rung 10: 41 innocents terminalized), but the single-model
metadata re-check inside the fidelity-repair loop bypassed that valve: its
failure arms caught the lane exception and minted ``failed:accuracy-gate /
checker-contract-invalid`` (or ``failed:runner``) -- the exact wrong-blame
class that consumed three dockets of re-litigation for m10979/m1115. A lane
failure PARKS on the checker-lane valve; only a validated adverse VERDICT
terminalizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from menagerie.crawler.driver_contracts import CheckerOutcome, DriverConfig, DriverIntegrationError
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_checker_lane_failure_scoping import _valve_rows
from menagerie.crawler.tests.test_slice_f_driver import (
    FidelityAuthor,
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke


class RecheckFailingChecker(ScriptedChecker):
    """Reject the first fidelity round, then fail the metadata re-check as a LANE.

    The exception is the wrapper's exit-64 shape (a contract failure carrying
    no verdict about the model), raised on every single-item metadata re-check
    after the fidelity repair.
    """

    def check_fidelity(
        self, artifact: Any, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        outcome = super().check_fidelity(artifact, work_root, config)
        assert outcome.gate is not None
        if self.fidelity_calls == 1:
            item = outcome.gate["items"][0]
            item["verdict"] = "inaccurate"
            item["fidelity"]["verdict"] = "major-drift"
            item["fidelity"]["contradictions"] = ["initial topology mismatch"]
        return outcome

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        # The repair RE-CHECK is the first metadata session bought after the
        # fidelity lane rejected (the H-5 duplicate-accept confirmation of the
        # ORIGINAL acceptance happens earlier, before any fidelity session,
        # and must succeed for the flow to reach the fidelity repair at all).
        if self.fidelity_calls >= 1:
            self.metadata_calls += 1
            raise DriverIntegrationError(
                "checker violated the gate contract and its result was refused by the "
                "operator wrapper (exit 64): checker result is partial or has extra items "
                "[permanent-contract-rejection]"
            )
        return super().check_metadata(artifacts, work_root, config)


def test_repair_recheck_lane_failure_parks_and_never_terminalizes(tmp_path: Path) -> None:
    """A checker dispatch failure in the repair re-check produces a park row.

    Pre-fix this minted ``failed:accuracy-gate / checker-contract-invalid``
    with human review -- a canonical terminal asserting a failure the model
    never earned. Now the model parks on the checker-lane valve exactly like
    the batch lane's floor, and requeues when the lane recovers.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    checker = RecheckFailingChecker()
    result = _driver(tmp_path, snapshot, author=FidelityAuthor(), checker=checker).run()

    assert result.status == "complete-with-deferrals"
    rows = _valve_rows(tmp_path, snapshot)
    assert [row["stable_id"] for row in rows] == [stable_id]
    assert rows[0]["checker_lane"]["failure_class"] == "permanent-contract-rejection"
    paths = _paths(tmp_path, snapshot)
    models = scan_jsonl(paths.ledgers.models) if paths.ledgers.models.exists() else []
    assert all(
        record["status"]["code"] != "failed:accuracy-gate" for record in models
    ), "a lane failure must never be recorded as an accuracy-gate model failure"
    assert models == [], "the parked model holds no canonical terminal at all"
    events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "checker-lane-parked"
    ]
    assert events and events[0]["details"]["failure_site"].startswith("fidelity-repair-recheck")
