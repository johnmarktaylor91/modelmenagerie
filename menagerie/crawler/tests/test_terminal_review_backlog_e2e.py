"""Terminal-review backlog sweep: full-driver-leg pins.

The requeue-grant lane holds a live grant over an UNREVIEWED failed terminal
(``unreviewed-terminal``), the review-held marking is produced only by the
terminal-disposition lane, and that lane's fresh caller runs only when an
author session in the CURRENT leg returns a typed recommendation -- so a
settled failed head's owed review could never be scheduled and its grant sat
held forever. ``CrawlerDriver._review_terminal_backlog`` breaks that cycle by
adjudicating the backlog BEFORE the WS-1 selection. This suite pins the sweep
end to end through real driver legs:

* a refused backlog claim is re-marked review-held in the containment shape
  (ORIGINAL status code kept, ``human_review.required`` flipped) and the
  outstanding grant binds and is consumed IN THE SAME LEG;
* the sweep buys sessions through the exact fresh-terminal front half, so a
  receiptless production leg (real ``CommandCheckerLane``, no calibration
  receipt) refuses typed BEFORE any checker subprocess is spawned;
* a driver-minted failed terminal with NO typed author claim is disclosed
  (``no_consumer``) and never handed to the checker -- fabricating a claim to
  adjudicate is forbidden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler.checker_calibration import (
    CHECKER_CALIBRATION_RECEIPT_ENV,
    CHECKER_CALIBRATION_RUN_ENV,
    CheckerCalibrationMissingError,
)
from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    canonical_requeue_grants_path,
)
from menagerie.crawler.driver import CommandAuthorLane, CommandCheckerLane
from menagerie.crawler.driver_contracts import (
    AuthorArtifact,
    CheckerOutcome,
    DriverConfig,
    DriverPaths,
    IntakeSnapshot,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.terminal_review import (
    TERMINAL_REVIEW_CENSUS_DISPOSITION,
    TERMINAL_UNADJUDICATED_DECISION,
    unreviewed_failed_terminal_heads,
)
from menagerie.crawler.tests.test_gate_verdict_reuse import _reseal
from menagerie.crawler.tests.test_slice_f_driver import (
    _BLOCKED_SOURCE_PAYLOAD,
    NOW,
    AuthorScript,
    CheckerScript,
    FakeAuthor,
    ScriptedAuthor,
    ScriptedChecker,
    _driver,
    _install_executor_lane,
    _paths,
    _snapshot,
)


class RecordingTerminalChecker(ScriptedChecker):
    """Scripted checker that records terminal dispatches and tags gate IDs.

    Two deliberate deviations from the plain fake, both test plumbing rather
    than behavior under test:

    * every ``check_terminal`` call records the dispatched stable ID, so a
      test can assert exactly which backlog heads bought a live session;
    * the terminal ``gate_id`` gets a per-leg tag (the ``TaggedTerminalChecker``
      pattern from test_c2_gate_rehearsal): the fake derives its gate ID from
      the stable ID alone, so two legs adjudicating the SAME claim to
      DIFFERENT verdicts would otherwise collide on one ID -- an artifact of
      the fake, not of the production ID, which mixes in a request nonce.

    It deliberately does NOT advertise ``adjudication_identity``: the sweep's
    ``already_adjudicated`` pre-filter keys on the live lane identity, and an
    identity-less lane disables the filter (fail-open to a live review), which
    is exactly what lets leg 2 re-adjudicate a claim leg 1 already gated.
    """

    def __init__(self, tag: str, *, script: Optional[CheckerScript] = None) -> None:
        super().__init__(script=script)
        self.tag = tag
        self.terminal_stable_ids: list[str] = []

    def check_terminal(
        self, artifact: AuthorArtifact, work_root: Path, config: DriverConfig
    ) -> CheckerOutcome:
        self.terminal_stable_ids.append(str(artifact.author_result.binding.stable_id))
        outcome = super().check_terminal(artifact, work_root, config)
        assert outcome.gate is not None
        gate = dict(outcome.gate)
        gate["gate_id"] = f"{gate['gate_id']}-{self.tag}"
        return CheckerOutcome(gate=_reseal(gate))


def _backlog_census_events(paths: DriverPaths) -> list[Mapping[str, Any]]:
    """Return every durable terminal-review-backlog census disclosure."""

    return [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == TERMINAL_REVIEW_CENSUS_DISPOSITION
    ]


def _model_revisions(paths: DriverPaths, stable_id: str) -> list[Mapping[str, Any]]:
    """Return one model's persisted revisions in ledger order."""

    return [record for record in scan_jsonl(paths.ledgers.models) if record["stable_id"] == stable_id]


def _settle_unreviewed_blocked_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[IntakeSnapshot, str, DriverPaths, Mapping[str, Any]]:
    """Leg 1: an ACCEPTED checked BLOCKED terminal settles review-free.

    The executor lane publishes the shared ``BLOCKED(source)`` payload, the
    default scripted checker accepts the terminal disposition, and the head
    settles ``failed:source`` with ``human_review.required`` False -- exactly
    the unreviewed failed terminal (with a rehydratable finalized claim) the
    backlog sweep exists to adjudicate.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    _install_executor_lane(monkeypatch, _BLOCKED_SOURCE_PAYLOAD)
    result = _driver(tmp_path, snapshot, author=CommandAuthorLane(("fake-author",))).run()
    assert result.status == "complete"
    revisions = _model_revisions(paths, stable_id)
    assert len(revisions) == 1
    head = revisions[0]
    assert head["status"]["kind"] == "failed"
    assert head["status"]["code"] == "failed:source"
    assert head["status"]["human_review"]["required"] is not True
    # Leg 1 had no backlog before its own selection, so no census exists yet.
    assert _backlog_census_events(paths) == []
    return snapshot, stable_id, paths, head


def _append_terminal_requeue_grant(
    paths: DriverPaths, stable_id: str, stage: str
) -> Mapping[str, Any]:
    """Append one canonical operator grant over the settled failed stage."""

    grants_path = canonical_requeue_grants_path(paths.ledgers.models)
    grant = build_canonical_requeue_grant(
        grants_path,
        stable_id=stable_id,
        stage=stage,
        reason="operator: terminal predicate disputed, re-drive approved",
        attempts=1,
        granted_by="operator",
    )
    assert append_canonical_requeue_grant(grants_path, grant)
    return grant


# ---------------------------------------------------------------------------
# 1. End to end: the backlog review releases a held grant in the same leg.
# ---------------------------------------------------------------------------


def test_backlog_review_releases_a_held_grant_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused backlog claim re-marks the head and its grant binds this leg.

    Leg 1 settles an accepted BLOCKED terminal (``failed:source``, review-free).
    An operator grant then lands over that exact stage. Leg 2 confirms the
    deadlock this sweep breaks: the grant is HELD ``unreviewed-terminal`` while
    the leg's own sweep re-adjudicates the claim (confirmed -- nothing changes).
    Leg 3's checker REFUSES the terminal claim: the sweep re-appends the head
    verbatim with the containment ladder's review-held marking, the census
    discloses it, and -- because the sweep runs BEFORE the WS-1 selection --
    the SAME leg binds and consumes the grant and re-drives the model.
    """

    snapshot, stable_id, paths, head_one = _settle_unreviewed_blocked_terminal(
        tmp_path, monkeypatch
    )
    grant = _append_terminal_requeue_grant(paths, stable_id, head_one["status"]["stage"])

    # -- Leg 2: an accepting review CONFIRMS; the grant stays held. ----------
    checker_mid = RecordingTerminalChecker("mid")
    _driver(tmp_path, snapshot, checker=checker_mid).run()
    assert checker_mid.terminal_stable_ids == [stable_id]
    census_mid = _backlog_census_events(paths)
    assert len(census_mid) == 1
    details_mid = census_mid[0]["details"]
    assert details_mid["eligible"] == 1
    assert details_mid["dispatched"] == 1
    assert details_mid["classes"]["confirmed"]["stable_ids"] == [stable_id]
    assert details_mid["classes"]["review_held"]["count"] == 0
    # A confirmed claim revises nothing: the settled record was already true.
    assert len(_model_revisions(paths, stable_id)) == 1
    # The grant is durably disclosed as HELD against the unreviewed terminal.
    holds = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("details", {}).get("disposition") == "requeue-grant-review-hold"
    ]
    assert len(holds) == 1
    assert holds[0]["details"]["stable_id"] == stable_id
    assert holds[0]["details"]["grant_id"] == str(grant["grant_id"])
    assert holds[0]["details"]["hold_reason"] == "unreviewed-terminal"
    assert not [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("event_kind") == "requeue-grant-consumed"
    ]

    # -- Leg 3: the review REFUSES; hold, bind, consume, re-drive -- one leg. -
    author_redrive = FakeAuthor()
    checker_reject = RecordingTerminalChecker(
        "reject", script=CheckerScript(terminal_verdict="rejected")
    )
    _driver(tmp_path, snapshot, author=author_redrive, checker=checker_reject).run()

    # (a) The census discloses the refused head as review_held.
    census = _backlog_census_events(paths)
    assert len(census) == 2
    details = census[-1]["details"]
    assert details["eligible"] == 1
    assert details["dispatched"] == 1
    assert details["classes"]["review_held"]["stable_ids"] == [stable_id]
    assert details["classes"]["confirmed"]["count"] == 0

    # (b) The head lineage shows the containment-shape supersession: ORIGINAL
    # failed:source code kept, review-held marking applied, exact parentage.
    revisions = _model_revisions(paths, stable_id)
    assert len(revisions) >= 3, "review-held revision plus the granted re-drive must exist"
    held = revisions[1]
    assert held["status"]["code"] == head_one["status"]["code"]
    assert held["status"]["stage"] == head_one["status"]["stage"]
    assert held["status"]["reason_code"] == head_one["status"]["reason_code"]
    assert held["status"]["human_review"] == {
        "required": True,
        "reason": None,
        "queue": "crawler-human-review",
        "requested_at": NOW,
    }
    assert held["parent_revision"] == head_one["record_revision"]
    assert held["status"]["supersedes_revision"] == head_one["record_revision"]

    # (c) The SAME leg consumed the grant: the sweep runs before selection, so
    # the census row provably precedes the consumption row on one ledger.
    events = scan_jsonl(paths.operational_ledger)
    consumed = [
        index
        for index, event in enumerate(events)
        if event.get("event_kind") == "requeue-grant-consumed"
        and event.get("details", {}).get("grant_id") == str(grant["grant_id"])
    ]
    assert len(consumed) == 1
    census_index = max(
        index
        for index, event in enumerate(events)
        if event.get("details", {}).get("disposition") == TERMINAL_REVIEW_CENSUS_DISPOSITION
    )
    assert census_index < consumed[0]

    # (d) The model was re-driven through the released grant in that leg.
    assert author_redrive.calls == {stable_id: 1}
    redriven = revisions[2]
    assert grant["grant_id"] in redriven["budget"]["explicit_grants"]

    # Idempotence of eligibility: the persisted review-held revision itself is
    # no longer an unreviewed failed terminal, so a later sweep excludes it.
    assert unreviewed_failed_terminal_heads({stable_id: held}) == ()


# ---------------------------------------------------------------------------
# 2. The sweep buys sessions through the fresh lane's calibration gate.
# ---------------------------------------------------------------------------


def test_receiptless_production_leg_refuses_sweep_sessions_before_any_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An uncalibrated real checker leg refuses the backlog review typed.

    The leg's only checker demand is the sweep's: the roster is fully settled
    and no grant exists, so without the backlog review nothing would consult
    checker readiness at all and the leg would complete silently. The lane's
    command is a poison argv -- executing it would fail the test with a
    different error -- so the typed ``CheckerCalibrationMissingError`` proves
    the sweep hit the exact fresh-path readiness gate strictly BEFORE any
    session subprocess was spawned.
    """

    snapshot, _stable_id, paths, _head = _settle_unreviewed_blocked_terminal(
        tmp_path, monkeypatch
    )
    gates_before = scan_jsonl(paths.ledgers.gates)

    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, str(tmp_path / "missing-receipt.json"))
    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    lane = CommandCheckerLane(("/nonexistent-checker-wrapper-must-never-run",))
    driver = _driver(tmp_path, snapshot, checker=lane)

    with pytest.raises(CheckerCalibrationMissingError):
        driver.run()

    # No session ran and no verdict landed: the gate ledger is untouched and
    # the refusal struck before the sweep could even publish its census.
    assert scan_jsonl(paths.ledgers.gates) == gates_before
    assert _backlog_census_events(paths) == []


# ---------------------------------------------------------------------------
# 3. No typed claim, no checker call: disclosed, never fabricated.
# ---------------------------------------------------------------------------


def test_no_claim_backlog_head_is_disclosed_never_fabricated(tmp_path: Path) -> None:
    """A driver-minted failed terminal is censused, not handed to the checker.

    Leg 1's author session crashes, so the driver mints ``failed:author`` from
    infrastructure facts: no typed terminal recommendation exists anywhere.
    The terminal-disposition lane therefore has no claim to adjudicate, and
    the sweep must disclose the head under ``no_consumer`` without buying a
    session or inventing a claim -- the recording checker proves no terminal
    dispatch happened for it.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    paths = _paths(tmp_path, snapshot)
    result_one = _driver(
        tmp_path, snapshot, author=ScriptedAuthor(AuthorScript(failed_id=stable_id))
    ).run()
    assert result_one.status == "terminal-partition-complete"
    head = _model_revisions(paths, stable_id)[0]
    assert head["status"]["kind"] == "failed"
    assert head["status"]["code"] == "failed:author"
    assert head["status"]["human_review"]["required"] is not True
    # The crash minted no typed verdict, so the head does not carry the exact
    # unadjudicated producer wording -- the closed rule classifies no_consumer
    # (claim_unrehydratable is reserved for unrehydratable TYPED claims).
    assert head["source_resolution"]["decision"] != TERMINAL_UNADJUDICATED_DECISION

    recorder = RecordingTerminalChecker("sweep")
    _driver(tmp_path, snapshot, checker=recorder).run()

    census = _backlog_census_events(paths)
    assert len(census) == 1
    details = census[0]["details"]
    assert details["eligible"] == 1
    assert details["dispatched"] == 0
    classes = details["classes"]
    assert classes["no_consumer"]["stable_ids"] == [stable_id]
    assert classes["claim_unrehydratable"]["count"] == 0
    assert classes["confirmed"]["count"] == 0
    assert classes["review_held"]["count"] == 0
    # The tripwire this test exists for: NO terminal session was bought for a
    # head that has no claim -- fabricating one is forbidden.
    assert recorder.terminal_stable_ids == []
    # The head is untouched: disclosed, still operator property.
    assert [record["status"]["code"] for record in _model_revisions(paths, stable_id)] == [
        "failed:author"
    ]


def test_unrehydratable_authority_is_censused_not_leg_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A head whose recorded authority refuses rehydration never aborts the leg.

    Leg 16 (2026-08-21): the sweep called ``_rehydrate_final_authority`` on a
    claim-bearing backlog head whose private-mirror object had an ambiguous
    intrinsic inventory; the typed ``DriverIntegrationError`` propagated and
    killed the whole campaign leg at startup. Blast radius is the head: the
    sweep must disclose it ``claim_unrehydratable`` and keep going.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    monkeypatch:
        Pytest monkeypatch fixture.
    """

    from menagerie.crawler.driver import CrawlerDriver
    from menagerie.crawler.driver_contracts import DriverIntegrationError

    snapshot, stable_id, paths, _head_one = _settle_unreviewed_blocked_terminal(
        tmp_path, monkeypatch
    )

    def refusing_rehydrate(self, item, reducer, operational):
        raise DriverIntegrationError(
            "canonical artifact authority cannot be rehydrated: physical mirror "
            "object has ambiguous intrinsic inventory"
        )

    monkeypatch.setattr(CrawlerDriver, "_rehydrate_final_authority", refusing_rehydrate)
    recorder = RecordingTerminalChecker("sweep")
    result_two = _driver(tmp_path, snapshot, checker=recorder).run()
    assert "complete" in str(result_two.status)

    census = _backlog_census_events(paths)
    assert census, "the sweep census never landed"
    classes = census[-1]["details"]["classes"]
    assert stable_id in classes["claim_unrehydratable"]["stable_ids"]


def test_gate_routing_defect_parks_the_member_never_the_leg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded claim that refuses gate validation parks typed; the leg lives.

    Calibration leg 6 (2026-08-27, m3423): a settled BLOCKED head's FINAL
    evidence pack carried a ``supports`` claim outside the arm's closed
    predicate vocabulary, so ``validate_terminal_disposition_gate`` raised
    ``GateRoutingError`` at the sweep's post-gate validation site and the
    unguarded raise killed the whole campaign leg (rc=1). W5 member-fault
    containment: the defect is deterministic replay of THAT head's recorded
    artifact -- machinery, never a model verdict and never the leg. The sweep
    must park the member on the lane's own typed brake (durable deferral row,
    ``parked`` census class), keep sweeping, and leave the head untouched so
    it stays eligible on the next run.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    monkeypatch:
        Pytest monkeypatch fixture.
    """

    import menagerie.crawler.driver as driver_module
    from menagerie.crawler.checker_lane import checker_lane_deferral_path
    from menagerie.crawler.gates import GateRoutingError

    snapshot, stable_id, paths, head_one = _settle_unreviewed_blocked_terminal(
        tmp_path, monkeypatch
    )

    def refusing_validate(*args: Any, **kwargs: Any) -> None:
        raise GateRoutingError(
            "terminal evidence ev-upstream-tf-imports does not support its typed predicate"
        )

    monkeypatch.setattr(
        driver_module, "validate_terminal_disposition_gate", refusing_validate
    )
    recorder = RecordingTerminalChecker("sweep")
    result_two = _driver(tmp_path, snapshot, checker=recorder).run()
    assert "complete" in str(result_two.status)

    # A session WAS bought (the defect fires strictly after the gate minted),
    # and the census discloses the head as parked -- no silent outcome.
    assert recorder.terminal_stable_ids == [stable_id]
    census = _backlog_census_events(paths)
    assert census, "the sweep census never landed"
    classes = census[-1]["details"]["classes"]
    assert stable_id in classes["parked"]["stable_ids"]
    assert classes["confirmed"]["count"] == 0
    assert classes["review_held"]["count"] == 0

    # The park is the lane's durable typed brake: one deferral row naming the
    # closed failure class, never counted as lane weather.
    records_root = paths.ledgers.models.parent.parent
    rows = [
        row
        for row in scan_jsonl(checker_lane_deferral_path(records_root))
        if row["stable_id"] == stable_id
    ]
    assert len(rows) == 1
    assert rows[0]["checker_lane"]["gate_kind"] == "terminal_disposition"
    assert rows[0]["checker_lane"]["failure_class"] == "invalid-gate-result"

    # The head itself is untouched -- no verdict, no revision -- and remains
    # exactly the unreviewed failed terminal the next run's sweep re-selects.
    revisions = _model_revisions(paths, stable_id)
    assert len(revisions) == 1
    assert revisions[0]["status"]["code"] == head_one["status"]["code"]
    assert revisions[0]["status"]["human_review"]["required"] is not True
    assert unreviewed_failed_terminal_heads({stable_id: revisions[0]}) != ()
