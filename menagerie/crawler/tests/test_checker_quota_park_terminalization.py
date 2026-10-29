"""Quota-exhausted checker parks must terminalize (the no-exit-park gap).

Pilot evidence (2026-08-21): five executed, artifact-valid models (m3417 m4893
m5656 m5703 m7646) hit ``QUOTA: <model>/<work> has no remaining cumulative
capacity`` through the checker batch-halving ladder and parked
``checker-lane-parked``. Checker-lane parks are RUN-scoped ("requeue on the
next run") -- designed for transient lane weather -- but generation-scoped
QUOTA exhaustion is permanent for the generation, so those models re-entered
and re-parked on EVERY leg forever, and requeue grants could not free them:
grant binding requires a review-held terminal head, and a parked model has no
head at all.

The behavior under test:

1. a floor-one park cause in the closed machine-classified QUOTA-exhaustion
   class (``CheckerWallBudgetExceeded``, the ``PaidSessionBudgetExceeded``
   QUOTA remap -- never message-text matching) does NOT park: it terminalizes
   ``failed:accuracy-gate / checker-session-exhaustion`` with a witnessing
   attempt and mandatory human review;
2. that terminal is grant-re-drivable exactly like every other review-held
   terminal: an operator ``--stage accuracy-gate`` grant re-opens it and the
   next leg completes the model;
3. a TRANSIENT floor-one cause keeps the run-scoped park semantics unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from menagerie.crawler.checker_dispatch import CheckerWallBudgetExceeded
from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    canonical_requeue_grants_path,
)
from menagerie.crawler.constants import (
    CHECKER_SESSION_EXHAUSTION_REASON_CODE,
    FAILURE_REASON_CODES,
)
from menagerie.crawler.driver_contracts import CheckerOutcome, DriverConfig
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_checker_lane_failure_scoping import (
    AlwaysPartialChecker,
    _valve_rows,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    ScriptedChecker,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke


class QuotaExhaustedChecker(ScriptedChecker):
    """Refuse every metadata batch with the typed QUOTA-exhaustion remap.

    Production raise site: the checker admission's ``PaidSessionBudgetExceeded``
    QUOTA remap in ``_admit_checker_spend`` -- before a session is bought. The
    double raises from the metadata dispatch itself, which lands in the same
    lane-failure route the admission raise travels.
    """

    def __init__(self) -> None:
        """Track the exact batch sizes the driver asked for."""

        super().__init__()
        self.batch_sizes: list[int] = []

    def check_metadata(
        self, artifacts: Sequence[Any], work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        """Refuse the batch naming every member's exhausted cumulative quota."""

        del work_root, config
        self.batch_sizes.append(len(artifacts))
        stable_ids = tuple(str(artifact.proposal["stable_id"]) for artifact in artifacts)
        raise CheckerWallBudgetExceeded(
            f"QUOTA: {stable_ids[0]}/work has no remaining cumulative capacity",
            stable_ids=stable_ids,
        )


def test_quota_exhausted_floor_park_terminalizes_reviewable(tmp_path: Path) -> None:
    """A quota-classified park cause produces the review-held terminal, not a park.

    The reason code is closed for the accuracy-gate stage, the terminal carries
    its witnessing driver attempt, human review is mandatory (grant-bindable),
    and NO checker-lane valve row or park event is minted.
    """

    assert CHECKER_SESSION_EXHAUSTION_REASON_CODE in FAILURE_REASON_CODES["accuracy-gate"]
    snapshot = _snapshot(tmp_path, count=2)
    checker = QuotaExhaustedChecker()
    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "terminal-partition-complete"
    paths = _paths(tmp_path, snapshot)
    models = {record["stable_id"]: record for record in scan_jsonl(paths.ledgers.models)}
    assert set(models) == {item.stable_id for item in snapshot.items}
    attempts = scan_jsonl(paths.ledgers.attempts)
    for stable_id, record in models.items():
        status = record["status"]
        assert status["code"] == "failed:accuracy-gate"
        assert status["stage"] == "accuracy-gate"
        assert status["reason_code"] == CHECKER_SESSION_EXHAUSTION_REASON_CODE
        assert status["human_review"]["required"] is True
        witnesses = [
            attempt
            for attempt in attempts
            if attempt["stable_id"] == stable_id
            and attempt.get("stage") == "accuracy-gate"
            and attempt.get("result") == "failed"
            and attempt.get("error", {}).get("reason_code")
            == CHECKER_SESSION_EXHAUSTION_REASON_CODE
        ]
        assert len(witnesses) == 1, "the terminal must carry its witnessing attempt"
        assert witnesses[0]["attempt_id"] in status["attempt_ids"]
    # No park: no valve row, no parked event, and the exhausted models never
    # ride a next-run requeue.
    assert _valve_rows(tmp_path, snapshot) == []
    events = scan_jsonl(paths.operational_ledger)
    assert not [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "checker-lane-parked"
    ]


def test_quota_terminal_is_grant_re_drivable(tmp_path: Path) -> None:
    """An accuracy-gate grant re-opens the exhaustion terminal; the next leg runs.

    Exactly the recovery the run-scoped park could never offer: the terminal is
    a review-held head, so the ordinary ``--stage accuracy-gate`` grant binds it
    and a healthy checker completes the model.
    """

    snapshot = _snapshot(tmp_path, count=1)
    first = _driver(tmp_path, snapshot, checker=QuotaExhaustedChecker()).run()
    assert first.status == "terminal-partition-complete"

    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id
    grants_path = canonical_requeue_grants_path(paths.ledgers.models)
    payload = build_canonical_requeue_grant(
        grants_path,
        stable_id=stable_id,
        stage="accuracy-gate",
        reason="reviewed: checker quota exhaustion, adjudication re-authorized",
        attempts=1,
        granted_by="operator",
        target_intent=None,
    )
    append_canonical_requeue_grant(grants_path, payload)

    second = _driver(tmp_path, snapshot).run()
    assert second.status == "complete"
    records = [
        record for record in scan_jsonl(paths.ledgers.models) if record["stable_id"] == stable_id
    ]
    assert records[-1]["status"]["code"] == "runs"
    assert str(payload["grant_id"]) in records[-1]["budget"]["explicit_grants"]


def test_transient_floor_park_still_parks_run_scoped(tmp_path: Path) -> None:
    """The control: a transient lane failure keeps the park semantics unchanged."""

    snapshot = _snapshot(tmp_path, count=1)
    result = _driver(tmp_path, snapshot, checker=AlwaysPartialChecker()).run()

    assert result.status == "complete-with-deferrals"
    paths = _paths(tmp_path, snapshot)
    assert scan_jsonl(paths.ledgers.models) == []
    rows = _valve_rows(tmp_path, snapshot)
    assert [row["stable_id"] for row in rows] == [snapshot.items[0].stable_id]
    assert rows[0]["checker_lane"]["failure_class"] == "partial-result"
