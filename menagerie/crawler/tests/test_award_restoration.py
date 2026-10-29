"""AW-3 — operator restoration of an eaten award (AUDIT100 SYNAUD_A section 2.3).

The one outstanding restoration at the step-100 close is m5273: runs award at
seq 187 superseded by a ``failed:accuracy-gate`` head at seq 206 carrying the
typed ``award-superseded-without-grant:<seq-187>`` scar. These tests drive a
chain of exactly that shape through ``append_award_restoration`` and pin the
acceptance conditions the reducer proves from ledger bytes alone -- including
sol's S2 strengthenings: evidence disjointness instead of bare
work-generation inequality (different work ids are cheap to mint and can
describe the same defective artifact), the no-soundness-revocation check, the
dry-run that appends nothing, and the derived restored-vs-earned census
segregation.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import (
    AWARD_RESTORATION_EVENT_DISPOSITION,
    AWARD_RESTORATION_SCAR_PREFIX,
    AWARD_SUPERSESSION_SCAR_PREFIX,
    ReductionError,
    project_dependency_current,
)
from menagerie.crawler.status import funnel_counts
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_attempt,
    make_authority_context,
    make_failed_attempt,
    make_gate,
    make_model,
)
from menagerie.crawler.reducer import CanonicalReducer as ProductionCanonicalReducer
from menagerie.crawler.tests.test_reducer import CanonicalReducer

pytestmark = pytest.mark.smoke

_EVENT_ID = "award-restored-by-operator-testevent0001"


def _paths(tmp_path: Path) -> LedgerPaths:
    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _eaten_award_chain(
    tmp_path: Path, reducer: ProductionCanonicalReducer, *, share_evidence: bool = False
) -> tuple[dict, dict]:
    """Build the m5273 chain shape: runs award then a scarred review-held eater."""

    del tmp_path
    reducer.append_gate(make_gate(["m_example"]))
    reducer.append_attempt(make_attempt())
    award = reducer.append_model(make_model(accepted=True)).record

    failed_attempt = make_failed_attempt(attempt_id="attempt-2")
    failed_attempt["ledger_seq"] = 2
    failed_attempt["work_id"] = "work-m_example-redrive"
    eater = bind_terminal_attempts(
        make_model(status_code="failed:source", attempt_id="attempt-2"),
        [failed_attempt],
    )
    eater["parent_revision"] = award["record_revision"]
    eater["status"]["supersedes_revision"] = award["record_revision"]
    eater["record_seq"] = int(award["record_seq"]) + 1
    eater["scar_history"] = [f"{AWARD_SUPERSESSION_SCAR_PREFIX}{award['record_revision']}"]
    eater["status"]["human_review"] = {
        "required": True,
        "reason": None,
        "queue": "crawler-human-review",
        "requested_at": eater["created_at"],
    }
    reducer.append_attempt(failed_attempt)
    prepared = reducer.prepare_model(eater)
    if share_evidence:
        # Sol's counterexample, as synthetic history: the adverse child cites
        # the target's accepted attempt while carrying a different work id.
        # (Live admission itself refuses this shape at _model_work_id, so it
        # is written directly as ledger bytes.)
        prepared = deepcopy(prepared)
        prepared["status"]["attempt_ids"] = [
            *prepared["status"]["attempt_ids"],
            *award["execution"]["accepted_attempt_ids"],
        ]
    persisted = reducer._models.append(prepared).record  # noqa: SLF001 -- historical bytes
    reducer._current["m_example"] = deepcopy(persisted)  # noqa: SLF001
    reducer._projection_cache = None  # noqa: SLF001
    return award, persisted


def _append_restoration_event(
    tmp_path: Path, award: dict, eater: dict, *, disposition: str, event_id: str
) -> None:
    operational = tmp_path / "operational"
    operational.mkdir(parents=True, exist_ok=True)
    with JsonlLedger(operational / "events.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        ledger.append(
            {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "event_id": event_id,
                "created_at": "2026-08-16T00:00:00Z",
                "event_kind": "campaign-health",
                "status": "degraded",
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 1},
                "current_environment": None,
                "run_id": "restore-superseded-award",
                "machine_id": "operator",
                "details": {
                    "disposition": disposition,
                    "stable_id": "m_example",
                    "restored_award_revision": str(award["record_revision"]),
                    "superseding_revision": str(eater["record_revision"]),
                    "reason": "generational disagreement adjudicated",
                    "granted_by": "operator-test",
                },
            }
        )


def test_restoration_restores_the_eaten_award_append_only(tmp_path: Path) -> None:
    """The m5273 path: event, then a restored head equal to the award's content."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        award, eater = _eaten_award_chain(tmp_path, reducer)
        _append_restoration_event(
            tmp_path,
            award,
            eater,
            disposition=AWARD_RESTORATION_EVENT_DISPOSITION,
            event_id=_EVENT_ID,
        )
        report = reducer.append_award_restoration(
            stable_id="m_example",
            restored_award_revision=str(award["record_revision"]),
            superseding_revision=str(eater["record_revision"]),
            restoration_event_id=_EVENT_ID,
        )
        assert report["appended"] is True
        head = reducer.raw_head_view["m_example"]
        assert head["status"]["kind"] == "runs"
        assert head["parent_revision"] == eater["record_revision"]
        assert f"{AWARD_RESTORATION_SCAR_PREFIX}{_EVENT_ID}" in head["scar_history"]
        # Content equals the target exactly outside parentage/scar bookkeeping.
        for field in ("execution", "implementation", "accuracy_gate", "dependency_vector"):
            assert head[field] == award[field]
        # The failure revision stays in history: award -> failure(scar) ->
        # award(restoration scar), append-only.
        chain = [row for row in scan_jsonl(paths.models) if row["stable_id"] == "m_example"]
        assert [row["status"]["kind"] for row in chain] == ["runs", "failed", "runs"]
        # An identical replay is idempotent, never a conflict.
        replayed = reducer.append_award_restoration(
            stable_id="m_example",
            restored_award_revision=str(award["record_revision"]),
            superseding_revision=str(eater["record_revision"]),
            restoration_event_id=_EVENT_ID,
        )
        assert replayed["appended"] is False

    # The restored head projects CURRENT under the same authority...
    projection = project_dependency_current(paths, context=make_authority_context(["m_example"]))
    assert "m_example" in projection.current_records, projection.stale_reasons.get("m_example")
    # ...and the derived census segregates it from earned awards.
    counts = funnel_counts(projection.current_records)
    assert counts["runs:restored"] == 1


def test_restoration_appends_at_the_ledger_tail_not_after_the_head(tmp_path: Path) -> None:
    """The live pilot shape: rows from OTHER models land after the eater.

    ``record_seq`` is ledger-local, and the appender derives the next local
    sequence itself when the row omits the field. Stamping the restoration
    candidate with ``head.record_seq + 1`` instead collides with the
    appender's next-local-sequence tripwire the moment any other model
    appended after the supersession -- exactly the state the m5273 restore
    found on the pilot (head at seq 206, ledger tail at seq 298; the dry-run
    reported success because it never reaches the append).
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example", "m_later"]) as reducer:
        award, eater = _eaten_award_chain(tmp_path, reducer)
        # The ledger tail moves past the eater: an unrelated model lands.
        later_model = make_model("m_later", accepted=True, attempt_id="attempt-3")
        later_model.pop("record_seq", None)
        later_model["accuracy_gate"]["gate_id"] = "gate-later"
        later_gate = make_gate(
            ["m_later"],
            gate_id="gate-later",
            vet_identity=later_model["accuracy_gate"]["vet_identity"],
        )
        later_gate["ledger_seq"] = 2
        reducer.append_gate(later_gate)
        later_attempt = make_attempt("m_later", attempt_id="attempt-3")
        later_attempt["ledger_seq"] = 3
        reducer.append_attempt(later_attempt)
        later = reducer.append_model(later_model).record
        assert later["record_seq"] > eater["record_seq"]
        _append_restoration_event(
            tmp_path,
            award,
            eater,
            disposition=AWARD_RESTORATION_EVENT_DISPOSITION,
            event_id=_EVENT_ID,
        )
        report = reducer.append_award_restoration(
            stable_id="m_example",
            restored_award_revision=str(award["record_revision"]),
            superseding_revision=str(eater["record_revision"]),
            restoration_event_id=_EVENT_ID,
        )
        assert report["appended"] is True
        head = reducer.raw_head_view["m_example"]
        assert head["status"]["kind"] == "runs"
        assert head["record_seq"] == later["record_seq"] + 1


def test_dry_run_proves_everything_and_appends_nothing(tmp_path: Path) -> None:
    """Sol correction 4: the dry-run reports exactly one intended head change."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        award, eater = _eaten_award_chain(tmp_path, reducer)
        before = len(tuple(scan_jsonl(paths.models)))
        report = reducer.append_award_restoration(
            stable_id="m_example",
            restored_award_revision=str(award["record_revision"]),
            superseding_revision=str(eater["record_revision"]),
            restoration_event_id=_EVENT_ID,
            dry_run=True,
        )
        assert report["dry_run"] is True
        assert report["appended"] is False
        assert report["restored_status_code"] == "runs"
        assert len(tuple(scan_jsonl(paths.models))) == before
        assert reducer.raw_head_view["m_example"]["record_revision"] == (eater["record_revision"])


def test_a_scarless_downgrade_never_qualifies_for_restoration(tmp_path: Path) -> None:
    """Condition (2) is not widenable: a scarless downgrade is indistinguishable
    from a legitimate one (the eleven 2026-08-08 wipe rows are exactly that
    shape, and ten of the eleven re-won runs heads organically)."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        award, eater = _eaten_award_chain(tmp_path, reducer)
        # Strip the scar from the persisted eater bytes and rebuild state.
        rows = [dict(row) for row in scan_jsonl(paths.models)]
        rows[-1]["scar_history"] = []
        # A hand-tampered ledger row fails its content digest downstream, so
        # instead assert on the reducer's own check with the head's scar list
        # emptied in the raw view (the check reads the head, not the file).
        reducer._current["m_example"] = {**eater, "scar_history": []}  # noqa: SLF001
        with pytest.raises(ReductionError, match="scarless downgrade never qualifies"):
            reducer.append_award_restoration(
                stable_id="m_example",
                restored_award_revision=str(award["record_revision"]),
                superseding_revision=str(eater["record_revision"]),
                restoration_event_id=_EVENT_ID,
            )


def test_an_evidence_sharing_adverse_child_refuses_restoration(tmp_path: Path) -> None:
    """Sol's counterexample closed: a child citing the target's accepted
    attempt is evidence ABOUT the awarded artifact even under a fresh work id,
    so restoring over it would discard a real adverse verdict."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        award, eater = _eaten_award_chain(tmp_path, reducer, share_evidence=True)
        with pytest.raises(ReductionError, match="accepted attempts"):
            reducer.append_award_restoration(
                stable_id="m_example",
                restored_award_revision=str(award["record_revision"]),
                superseding_revision=str(eater["record_revision"]),
                restoration_event_id=_EVENT_ID,
            )


def test_a_missing_decision_event_refuses_a_live_restoration(tmp_path: Path) -> None:
    """The decision event is part of the proof chain; absent means dry-run only."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        award, eater = _eaten_award_chain(tmp_path, reducer)
        with pytest.raises(ReductionError, match="decision event is absent"):
            reducer.append_award_restoration(
                stable_id="m_example",
                restored_award_revision=str(award["record_revision"]),
                superseding_revision=str(eater["record_revision"]),
                restoration_event_id="award-restored-by-operator-neverappended",
            )


def test_a_decision_event_bound_to_the_wrong_head_refuses(tmp_path: Path) -> None:
    """P5-SOL-03/07 RED: the event must bind the EXACT superseding head.

    The round-2 probe: a durable ``award-restored-by-operator`` event whose
    ``superseding_revision`` names a head that never existed (``ffff...``)
    still discharged a restoration over the REAL head, because the live event
    lookup checked only event id, disposition, stable id, and the restored
    award revision. The event's head binding is now part of the closed lookup,
    so a wrong-head (or fabricated-head) decision event authorizes nothing.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        award, eater = _eaten_award_chain(tmp_path, reducer)
        wrong_head = dict(eater)
        wrong_head["record_revision"] = "sha256:" + "ff" * 32
        _append_restoration_event(
            tmp_path,
            award,
            wrong_head,
            disposition=AWARD_RESTORATION_EVENT_DISPOSITION,
            event_id=_EVENT_ID,
        )
        head_before = deepcopy(reducer._current["m_example"])  # noqa: SLF001
        with pytest.raises(ReductionError, match="binds a different superseding head"):
            reducer.append_award_restoration(
                stable_id="m_example",
                restored_award_revision=str(award["record_revision"]),
                superseding_revision=str(eater["record_revision"]),
                restoration_event_id=_EVENT_ID,
            )
        # Nothing appended: the eater still governs.
        assert reducer._current["m_example"] == head_before  # noqa: SLF001
        rows = list(scan_jsonl(paths.models))
        assert str(rows[-1]["record_revision"]) == str(eater["record_revision"])


def test_an_award_revocation_covering_the_target_refuses(tmp_path: Path) -> None:
    """Sol correction 2: an authenticated revocation outranks restoration."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        award, eater = _eaten_award_chain(tmp_path, reducer)
        operational = tmp_path / "operational"
        operational.mkdir(parents=True, exist_ok=True)
        with JsonlLedger(operational / "events.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
            ledger.append(
                {
                    "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                    "event_id": "award-revocation-test0001",
                    "created_at": "2026-08-16T00:00:00Z",
                    "event_kind": "campaign-health",
                    "status": "degraded",
                    "provider": None,
                    "observed_response": None,
                    "reset_at": None,
                    "queued_work_counts": {"models": 1},
                    "current_environment": None,
                    "run_id": "operator",
                    "machine_id": "operator",
                    "details": {
                        "disposition": "award-revocation",
                        "stable_id": "m_example",
                        "award_revision": str(award["record_revision"]),
                        "reason_code": "soundness-proof-invalidated",
                    },
                }
            )
        with pytest.raises(ReductionError, match="revocation covers"):
            reducer.append_award_restoration(
                stable_id="m_example",
                restored_award_revision=str(award["record_revision"]),
                superseding_revision=str(eater["record_revision"]),
                restoration_event_id=_EVENT_ID,
            )
