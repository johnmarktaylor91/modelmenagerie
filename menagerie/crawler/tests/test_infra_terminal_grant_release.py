"""Infra-terminal grant release: the second four-part waiver (KERNEL_DECISION 2026-08-21).

Grants bind only review-held terminals; the terminal-review lane adjudicates
AUTHOR CLAIMS. A claim-less machine-infrastructure death (storm 529, worker
crash before any receipt, exhaustion from censored sessions) has nothing to
adjudicate, so its ``unreviewed-terminal`` hold had no consumer and could never
release. The kernel decision authorizes binding without review on a PROVABLY
machine-infrastructure terminal, under the four-part waiver doctrine. The
mandated RED matrix:

- an all-infra-witnessed terminal's grant binds and re-drives;
- a mixed-witness terminal stays held (one non-infra witnessing attempt = NOT
  eligible), including a failing generation row the record never referenced;
- a claim-bearing terminal still requires review (claim reason codes AND an
  adjudicated terminal-disposition gate);
- the derived census segregates ``infra-waiver-released`` from
  ``review-released``, recomputed with zero producer input.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence, cast

import pytest

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    canonical_requeue_grants_path,
)
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
)
from menagerie.crawler.driver_admission import AdmissionEnvironmentMixin
from menagerie.crawler.driver_contracts import DriverPaths
from menagerie.crawler.identity import canonical_json_bytes, payload_hash, stable_hash
from menagerie.crawler.infra_release import (
    RELEASE_AUTHORITY_HUMAN_REVIEW,
    RELEASE_AUTHORITY_INFRA_WAIVER,
    derive_infra_terminal_release,
    machine_infrastructure_witness_basis,
)
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.reducer import default_ledger_paths
from menagerie.crawler.status import grant_release_census
from menagerie.crawler.tests.conftest import (
    make_failed_attempt,
    rebind_nonaward_parent_proof,
)
from menagerie.crawler.work_selection import (
    MemberSelection,
    SelectionDecision,
    SelectionMode,
    SelectionPlan,
)

#: Load-bearing kernel-change tests: sub-second, in the mandatory smoke gate.
pytestmark = pytest.mark.smoke

NOW = "2026-08-21T08:00:00Z"
STABLE_ID = "m_infra"
OTHER_ID = "m_reviewed"
WORK_ID = f"work-{STABLE_ID}"


def _paths(tmp_path: Path) -> DriverPaths:
    """Return isolated driver paths below one temporary root."""

    return DriverPaths(
        runtime_root=tmp_path / ".crawl-local",
        intake_root=tmp_path / "intake",
        ledgers=default_ledger_paths(tmp_path / "records"),
    )


def _stub_driver(paths: DriverPaths) -> Any:
    """Return the minimal driver facade ``_consume_requeue_grants`` reads."""

    return SimpleNamespace(
        paths=paths,
        registry=SimpleNamespace(intents={}),
        dependencies=SimpleNamespace(
            clock=lambda: NOW,
            boundary_hook=lambda *args, **kwargs: None,
        ),
        config=SimpleNamespace(run_id="test-run", machine_id="test-machine"),
    )


def _terminal_record(
    stable_id: str,
    *,
    stage: Optional[str] = "runner",
    kind: str = "failed",
    code: Optional[str] = None,
    reason_code: Optional[str] = "missing-receipt",
    attempt_ids: tuple[str, ...] = (),
    review_required: bool = False,
    revision: str = "sha256:" + "a" * 64,
    record_seq: int = 1,
) -> dict[str, Any]:
    """Return one minimal persisted terminal model revision.

    Parameters
    ----------
    stable_id:
        Model identity.
    stage, kind, code, reason_code, attempt_ids, review_required:
        Recorded terminal status facts, witnessing attempts, and review flag.
    revision, record_seq:
        Persisted revision identity and sequence.
    """

    return {
        "stable_id": stable_id,
        "record_revision": revision,
        "parent_revision": None,
        "record_seq": record_seq,
        "status": {
            "kind": kind,
            "code": code if code is not None else f"failed:{stage}",
            "stage": stage,
            "reason_code": reason_code,
            "attempt_ids": list(attempt_ids),
            "human_review": {
                "queue": "crawler-human-review" if review_required else None,
                "reason": None,
                "requested_at": NOW if review_required else None,
                "required": review_required,
            },
        },
        "accuracy_gate": {"gate_id": None},
        "budget": {"explicit_grants": []},
    }


def _write_models(paths: DriverPaths, records: Sequence[Mapping[str, Any]]) -> None:
    """Write raw model revisions to the harness models ledger."""

    paths.ledgers.models.parent.mkdir(parents=True, exist_ok=True)
    with paths.ledgers.models.open("wb") as handle:
        for record in records:
            handle.write(canonical_json_bytes(dict(record)) + b"\n")


def _write_attempts(paths: DriverPaths, rows: list[dict[str, Any]]) -> None:
    """Write canonical attempt rows with exact ledger bindings stamped."""

    paths.ledgers.attempts.parent.mkdir(parents=True, exist_ok=True)
    with paths.ledgers.attempts.open("wb") as handle:
        for index, row in enumerate(rows):
            row["ledger_seq"] = index + 1
            row["payload_sha256"] = payload_hash(row)
            handle.write(canonical_json_bytes(row) + b"\n")


def _worker_crash_attempt(
    stable_id: str,
    *,
    attempt_id: str,
    attempt_no: int = 1,
    work_id: str = WORK_ID,
) -> dict[str, Any]:
    """Build one parent-attested worker crash that died before any receipt."""

    attempt = make_failed_attempt(
        stable_id, attempt_id=attempt_id, stage="runner", reason_code="missing-receipt"
    )
    attempt["work_id"] = work_id
    attempt["attempt_no"] = attempt_no
    attempt["actor"] = "worker"
    attempt["error"]["exception_type"] = None
    attempt["supervisor_observation"]["exit_code"] = 1
    return rebind_nonaward_parent_proof(attempt)


def _exhaustion_attempt(
    stable_id: str,
    *,
    attempt_id: str,
    attempt_no: int = 1,
    work_id: str = WORK_ID,
) -> dict[str, Any]:
    """Build one driver-observed effort-cap exhaustion witness."""

    attempt = make_failed_attempt(
        stable_id,
        attempt_id=attempt_id,
        stage="author",
        reason_code="effort-exhausted:author-sessions",
    )
    attempt["work_id"] = work_id
    attempt["attempt_no"] = attempt_no
    attempt["error"]["exception_type"] = (
        "menagerie.crawler.driver_contracts.AuthorEffortCapExceeded"
    )
    attempt["error"]["details"] = {"driver_observed": True, "environment": None}
    return rebind_nonaward_parent_proof(attempt)


def _forward_failure_attempt(
    stable_id: str,
    *,
    attempt_id: str,
    attempt_no: int = 2,
    work_id: str = WORK_ID,
) -> dict[str, Any]:
    """Build one NON-infra failing generation row (a model-conduct forward death)."""

    attempt = make_failed_attempt(
        stable_id, attempt_id=attempt_id, stage="forward", reason_code="mode-run"
    )
    attempt["work_id"] = work_id
    attempt["attempt_no"] = attempt_no
    attempt["actor"] = "worker"
    attempt["supervisor_observation"]["exit_code"] = 1
    return rebind_nonaward_parent_proof(attempt)


def _append_grant(
    paths: DriverPaths,
    stable_id: str,
    *,
    stage: str = "runner",
    reason: str = "operator approved infra re-drive",
) -> Mapping[str, Any]:
    """Append one tool-style grant to the canonical grant ledger."""

    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    grant = build_canonical_requeue_grant(
        canonical,
        stable_id=stable_id,
        stage=stage,
        reason=reason,
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(canonical, grant)
    return grant


def _consume(
    paths: DriverPaths, *, intake_ids: Optional[frozenset[str]] = None
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    """Run the real consumer against the harness ledgers."""

    stub = _stub_driver(paths)
    with JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        result = AdmissionEnvironmentMixin._consume_requeue_grants(
            stub,
            operational,
            intake_ids if intake_ids is not None else frozenset({STABLE_ID}),
        )
        events = list(operational.records)
    return result, cast(list[Mapping[str, Any]], events)


def _hold_events(events: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return the durable per-model grant review-hold disclosures."""

    return [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.CAMPAIGN_HEALTH.value
        and event.get("details", {}).get("disposition") == "requeue-grant-review-hold"
    ]


def _consumption_events(events: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return the durable grant consumption receipts."""

    return [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]


# ---------------------------------------------------------------------------
# RED 1 -- an all-infra-witnessed terminal's grant binds and re-drives


def test_all_infra_witnessed_worker_crash_terminal_binds_and_redrives(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    witnesses = [
        _worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-1", attempt_no=1),
        _worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-2", attempt_no=2),
    ]
    _write_attempts(paths, witnesses)
    record = _terminal_record(
        STABLE_ID, attempt_ids=("attempt-crash-1", "attempt-crash-2")
    )
    _write_models(paths, [record])
    grant = _append_grant(paths, STABLE_ID, stage="runner")

    result, events = _consume(paths)

    assert _hold_events(events) == []
    consumptions = _consumption_events(events)
    assert len(consumptions) == 1
    details = consumptions[0]["details"]
    expected_work_id = stable_hash(
        {
            "stable_id": STABLE_ID,
            "grant_id": grant["grant_id"],
            "parent_revision": record["record_revision"],
            "generation": 1,
        }
    )
    assert details["new_work_id"] == expected_work_id
    assert details["release_authority"] == RELEASE_AUTHORITY_INFRA_WAIVER
    assert details["infra_release_witness_bases"] == [
        "worker-crash-missing-receipt:exit:1",
        "worker-crash-missing-receipt:exit:1",
    ]
    assert result[STABLE_ID]["work_id"] == expected_work_id
    assert result[STABLE_ID]["active"] is True


def test_all_infra_witnessed_exhaustion_terminal_binds(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_attempts(paths, [_exhaustion_attempt(STABLE_ID, attempt_id="attempt-cap-1")])
    record = _terminal_record(
        STABLE_ID,
        stage="author",
        reason_code="effort-exhausted:author-sessions",
        attempt_ids=("attempt-cap-1",),
    )
    _write_models(paths, [record])
    _append_grant(paths, STABLE_ID, stage="author")

    result, events = _consume(paths)

    assert _hold_events(events) == []
    assert len(_consumption_events(events)) == 1
    assert result[STABLE_ID]["active"] is True


# ---------------------------------------------------------------------------
# RED 2 -- one non-infra witnessing attempt = NOT eligible = the hold stands


def test_mixed_witness_terminal_stays_held(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_attempts(
        paths,
        [
            _worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-1", attempt_no=1),
            _forward_failure_attempt(STABLE_ID, attempt_id="attempt-forward-2", attempt_no=2),
        ],
    )
    record = _terminal_record(
        STABLE_ID, attempt_ids=("attempt-crash-1", "attempt-forward-2")
    )
    _write_models(paths, [record])
    _append_grant(paths, STABLE_ID, stage="runner")

    result, events = _consume(paths)

    assert _consumption_events(events) == []
    holds = _hold_events(events)
    assert len(holds) == 1
    assert holds[0]["details"]["hold_reason"] == "unreviewed-terminal"
    assert result == {}


def test_unreferenced_failing_generation_row_still_holds(tmp_path: Path) -> None:
    """Part 2 quantifies the WHOLE generation: an omitted failing row still holds."""

    paths = _paths(tmp_path)
    _write_attempts(
        paths,
        [
            _worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-1", attempt_no=1),
            _forward_failure_attempt(STABLE_ID, attempt_id="attempt-forward-2", attempt_no=2),
        ],
    )
    record = _terminal_record(STABLE_ID, attempt_ids=("attempt-crash-1",))
    _write_models(paths, [record])
    _append_grant(paths, STABLE_ID, stage="runner")

    result, events = _consume(paths)

    assert _consumption_events(events) == []
    assert len(_hold_events(events)) == 1
    assert result == {}


def test_dangling_or_foreign_witness_holds(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    foreign = _worker_crash_attempt("m_foreign", attempt_id="attempt-foreign-1")
    foreign["work_id"] = "work-m_foreign"
    _write_attempts(paths, [foreign])
    records = [
        _terminal_record(STABLE_ID, attempt_ids=("attempt-missing-1",)),
        _terminal_record(
            OTHER_ID,
            attempt_ids=("attempt-foreign-1",),
            revision="sha256:" + "b" * 64,
        ),
    ]
    _write_models(paths, records)
    _append_grant(paths, STABLE_ID, stage="runner")
    _append_grant(paths, OTHER_ID, stage="runner")

    result, events = _consume(paths, intake_ids=frozenset({STABLE_ID, OTHER_ID}))

    assert _consumption_events(events) == []
    assert len(_hold_events(events)) == 2
    assert result == {}


# ---------------------------------------------------------------------------
# RED 3 -- a claim-bearing terminal still requires review


def test_claim_bearing_terminal_still_requires_review(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_attempts(paths, [_exhaustion_attempt(STABLE_ID, attempt_id="attempt-cap-1")])
    record = _terminal_record(
        STABLE_ID,
        stage="author",
        reason_code="malformed-result",
        attempt_ids=("attempt-cap-1",),
    )
    _write_models(paths, [record])
    _append_grant(paths, STABLE_ID, stage="author")

    result, events = _consume(paths)

    assert _consumption_events(events) == []
    holds = _hold_events(events)
    assert len(holds) == 1
    assert holds[0]["details"]["hold_reason"] == "unreviewed-terminal"
    assert result == {}


def test_adjudicated_terminal_disposition_gate_refuses_the_waiver() -> None:
    """An adjudicated author claim at the generation is a review consumer."""

    record = _terminal_record(STABLE_ID, attempt_ids=("attempt-crash-1",))
    witness = _worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-1")
    witness["ledger_seq"] = 1
    witness["payload_sha256"] = payload_hash(witness)
    claim_gate = {
        "gate_kind": "terminal_disposition",
        "items": [{"stable_id": STABLE_ID, "work_id": WORK_ID}],
    }
    assert derive_infra_terminal_release(record, [witness], []) is not None
    assert derive_infra_terminal_release(record, [witness], [claim_gate]) is None


def test_defer_evidence_on_a_generation_row_refuses_the_waiver() -> None:
    record = _terminal_record(STABLE_ID, attempt_ids=("attempt-crash-1",))
    witness = _worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-1")
    witness["defer_evidence"] = {
        "target_status": "deferred:needs-cuda",
        "source_ids": [],
        "probe_attempt_ids": [],
        "explanation": "author claims a platform handoff",
    }
    witness["ledger_seq"] = 1
    witness["payload_sha256"] = payload_hash(witness)
    assert derive_infra_terminal_release(record, [witness], []) is None


# ---------------------------------------------------------------------------
# The closed classification rows refuse producer-steerable shapes


@pytest.mark.parametrize(
    "mutate",
    [
        # A diagnostic-text basis is quarantined producer prose, never machine.
        lambda error: error.update(
            failure_class="provider-overloaded",
            failure_class_basis="diagnostic-text:overload-marker",
        ),
        # A mismatched class/basis pair is outside the closed set.
        lambda error: error.update(
            failure_class="content-refusal",
            failure_class_basis="api_error_status:529",
        ),
    ],
)
def test_non_machine_session_classifications_refuse(mutate: Any) -> None:
    attempt = make_failed_attempt(
        STABLE_ID, attempt_id="attempt-session-1", stage="author", reason_code="session-crashed"
    )
    attempt["error"]["details"] = {"driver_observed": True, "environment": None}
    mutate(attempt["error"])
    assert machine_infrastructure_witness_basis(attempt) is None


def test_machine_classified_session_death_qualifies() -> None:
    attempt = make_failed_attempt(
        STABLE_ID, attempt_id="attempt-session-1", stage="author", reason_code="session-crashed"
    )
    attempt["error"]["details"] = {"driver_observed": True, "environment": None}
    attempt["error"].update(
        failure_class="provider-overloaded",
        failure_class_basis="api_error_status:529",
    )
    basis = machine_infrastructure_witness_basis(attempt)
    assert basis == "machine-session-failure:api_error_status:529"


# ---------------------------------------------------------------------------
# RED 4 -- the derived census segregates the release classes


def test_census_segregates_infra_waiver_release_from_review_release(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    witnesses = [_worker_crash_attempt(STABLE_ID, attempt_id="attempt-crash-1")]
    _write_attempts(paths, witnesses)
    infra_record = _terminal_record(STABLE_ID, attempt_ids=("attempt-crash-1",))
    reviewed_record = _terminal_record(
        OTHER_ID,
        stage="accuracy-gate",
        reason_code="terminal-disposition-rejected",
        review_required=True,
        revision="sha256:" + "b" * 64,
    )
    _write_models(paths, [infra_record, reviewed_record])
    infra_grant = _append_grant(paths, STABLE_ID, stage="runner")
    review_grant = _append_grant(paths, OTHER_ID, stage="accuracy-gate")

    _, events = _consume(paths, intake_ids=frozenset({STABLE_ID, OTHER_ID}))
    consumptions = _consumption_events(events)
    assert len(consumptions) == 2

    census = grant_release_census(
        model_revisions=[infra_record, reviewed_record],
        operational_events=events,
        attempt_records=witnesses,
        gate_records=[],
    )
    assert census["infra-waiver-released"] == (str(infra_grant["grant_id"]),)
    assert census["review-released"] == (str(review_grant["grant_id"]),)
    assert census["unattributed"] == ()


def test_persisted_release_authority_never_steers_the_census(tmp_path: Path) -> None:
    """Part 3: the census recomputes; a forged persisted flag changes nothing."""

    paths = _paths(tmp_path)
    reviewed_record = _terminal_record(
        OTHER_ID,
        stage="accuracy-gate",
        reason_code="terminal-disposition-rejected",
        review_required=True,
        revision="sha256:" + "b" * 64,
    )
    _write_models(paths, [reviewed_record])
    review_grant = _append_grant(paths, OTHER_ID, stage="accuracy-gate")

    _, events = _consume(paths, intake_ids=frozenset({OTHER_ID}))
    forged = []
    for event in events:
        event = dict(event)
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value:
            details = dict(event["details"])
            details["release_authority"] = RELEASE_AUTHORITY_INFRA_WAIVER
            event["details"] = details
        forged.append(event)

    census = grant_release_census(
        model_revisions=[reviewed_record],
        operational_events=forged,
        attempt_records=[],
        gate_records=[],
    )
    assert census["infra-waiver-released"] == ()
    assert census["review-released"] == (str(review_grant["grant_id"]),)


def test_reviewed_terminal_binds_with_review_authority(tmp_path: Path) -> None:
    """The historical lane is untouched and discloses its own authority class."""

    paths = _paths(tmp_path)
    record = _terminal_record(
        OTHER_ID,
        stage="accuracy-gate",
        reason_code="terminal-disposition-rejected",
        review_required=True,
        revision="sha256:" + "b" * 64,
    )
    _write_models(paths, [record])
    _append_grant(paths, OTHER_ID, stage="accuracy-gate")

    result, events = _consume(paths, intake_ids=frozenset({OTHER_ID}))

    consumptions = _consumption_events(events)
    assert len(consumptions) == 1
    assert consumptions[0]["details"]["release_authority"] == RELEASE_AUTHORITY_HUMAN_REVIEW
    assert "infra_release_witness_bases" not in consumptions[0]["details"]
    assert result[OTHER_ID]["active"] is True


def test_selection_census_counts_the_infra_waiver_class() -> None:
    plan = SelectionPlan(
        mode=SelectionMode.ORDINARY,
        members=(
            MemberSelection(
                STABLE_ID,
                True,
                SelectionDecision.GRANTED_REDRIVE,
                work_authority="grant-1",
                release_authority=RELEASE_AUTHORITY_INFRA_WAIVER,
            ),
            MemberSelection(
                OTHER_ID,
                True,
                SelectionDecision.GRANTED_REDRIVE,
                work_authority="grant-2",
                release_authority=RELEASE_AUTHORITY_HUMAN_REVIEW,
            ),
        ),
    )
    census = plan.census()
    assert census["granted"] == 2
    assert census["granted_infra_waiver"] == 1
