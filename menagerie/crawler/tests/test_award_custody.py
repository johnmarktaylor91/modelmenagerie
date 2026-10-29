"""AW-1 — the award-custody invariant (AUDIT100 SYNAUD_A section 2.1).

A model revision with ``status.kind == "runs"`` is operator property from the
moment it is appended. The step-100 books hold the refutation of the old
escapes: three verified awards were destroyed by re-adjudication failures
whose candidates carried a self-stamped scar (the producer waiving its own
tripwire), and the grant escape released on ANY strict superset of
``budget.explicit_grants`` -- retry permission and downgrade permission were
one authority. These tests pin the narrowed contract:

- a candidate-carried scar releases nothing at live append (replay of
  historical scarred rows stays acceptable -- the m5273 chain regression,
  fable's S2 rider);
- only a grant naming the exact blocked award revision releases a downgrade;
- a refused downgrade parks typed with the reducer-returned RAW award head
  bound (the close books hold 10 AwardSupersessionError refusal parks, 10/10
  with ``blocked_parent_revision: null``);
- award parks never advance the generic unrecordable streak, and a park
  APPEND failure advances its own typed breaker instead of disappearing;
- every leg-appended ``runs -> non-runs`` transition must reconcile to an
  exact award action at close, and ``runs -> runs`` remints are disclosed
  (63 silent award-over-award supersessions at close, zero events).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.grant_identity import tool_grant_identity
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import JsonlLedger, canonical_json_bytes, scan_jsonl
from menagerie.crawler.reducer import (
    AWARD_SUPERSESSION_SCAR_PREFIX,
    AwardSupersessionError,
    project_dependency_current,
)
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_attempt,
    make_authority_context,
    make_failed_attempt,
    make_gate,
    make_model,
)
from menagerie.crawler.tests.test_reducer import CanonicalReducer

pytestmark = pytest.mark.smoke


def _paths(tmp_path: Path) -> LedgerPaths:
    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _runs_head(reducer: CanonicalReducer) -> dict:
    reducer.append_gate(make_gate(["m_example"]))
    reducer.append_attempt(make_attempt())
    return reducer.append_model(make_model(accepted=True)).record


def _grantless_failure_over(
    persisted: dict, *, work_id: str = "work-m_example-redrive"
) -> tuple[dict, dict]:
    """Return an otherwise-valid grant-free failure superseding one runs head."""

    failed_attempt = make_failed_attempt(attempt_id="attempt-2")
    failed_attempt["ledger_seq"] = 2
    failed_attempt["work_id"] = work_id
    candidate = bind_terminal_attempts(
        make_model(status_code="failed:source", attempt_id="attempt-2"),
        [failed_attempt],
    )
    candidate["parent_revision"] = persisted["record_revision"]
    candidate["status"]["supersedes_revision"] = persisted["record_revision"]
    candidate["record_seq"] = int(persisted["record_seq"]) + 1
    return candidate, failed_attempt


def _durable_grant(
    tmp_path: Path,
    persisted: dict,
    *,
    blocked_award_revision: str | None,
    created_at: str,
) -> tuple[str, str]:
    """Persist one tool-minted grant row plus its consumption event.

    Returns the grant id and the granted new-work id, both canonically bound
    to ``persisted`` as the exact parent, so the reducer's requeue-lineage
    validator accepts the supersession that consumes the grant.
    """

    reason = "operator adjudicated the award supersession"
    identity_kwargs: dict = {
        "generation": 1,
        "stable_id": "m_example",
        "stage": "source",
        "reason": reason,
        "attempts": 1,
        "granted_by": "operator-test",
    }
    if blocked_award_revision is not None:
        identity_kwargs["blocked_award_revision"] = blocked_award_revision
    grant_id = tool_grant_identity(**identity_kwargs)
    grant = {
        "schema_version": "menagerie.crawler.requeue-grant.v1",
        "grant_id": grant_id,
        "stable_id": "m_example",
        "stage": "source",
        "reason": reason,
        "attempts": 1,
        "granted_by": "operator-test",
        "new_work_generation": 1,
        "created_at": created_at,
    }
    if blocked_award_revision is not None:
        grant["blocked_award_revision"] = blocked_award_revision
    operational = tmp_path / "operational"
    operational.mkdir(parents=True, exist_ok=True)
    with (operational / "requeue-grants.jsonl").open("ab") as handle:
        handle.write(canonical_json_bytes(grant) + b"\n")
    work_id = stable_hash(
        {
            "stable_id": "m_example",
            "grant_id": grant_id,
            "parent_revision": persisted["record_revision"],
            "generation": 1,
        }
    )
    details = {
        "grant_id": grant_id,
        "stable_id": "m_example",
        "stage": "source",
        "reason": reason,
        "attempts": 1,
        "source_record_revision": persisted["record_revision"],
        "new_work_generation": 1,
        "new_work_id": work_id,
    }
    if blocked_award_revision is not None:
        details["blocked_award_revision"] = blocked_award_revision
    event = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_id": f"requeue-consumed-{grant_id[7:31]}",
        "created_at": created_at,
        "event_kind": "requeue-grant-consumed",
        "status": "requeue-grant-consumed",
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": "run-test",
        "machine_id": "machine-test",
        "details": details,
    }
    with JsonlLedger(operational / "events.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        ledger.append(event)
    return grant_id, work_id


def _granted_failure_over(
    tmp_path: Path,
    reducer: CanonicalReducer,
    persisted: dict,
    *,
    blocked_award_revision: str | None,
) -> dict:
    """Build a failure candidate consuming a freshly persisted durable grant."""

    grant_id, work_id = _durable_grant(
        tmp_path,
        persisted,
        blocked_award_revision=blocked_award_revision,
        created_at=persisted["created_at"],
    )
    attempt = make_failed_attempt(attempt_id="attempt-granted")
    attempt.pop("ledger_seq", None)
    attempt.pop("payload_sha256", None)
    attempt["work_id"] = work_id
    reducer.append_attempt(attempt)
    candidate = bind_terminal_attempts(
        make_model(status_code="failed:source", attempt_id="attempt-granted"),
        [attempt],
    )
    candidate["parent_revision"] = persisted["record_revision"]
    candidate["status"]["supersedes_revision"] = persisted["record_revision"]
    candidate["record_seq"] = int(persisted["record_seq"]) + 1
    candidate["budget"]["explicit_grants"] = [grant_id]
    return candidate


def test_a_generic_new_grant_no_longer_releases_an_award_downgrade(
    tmp_path: Path,
) -> None:
    """Retry permission is not downgrade permission (delta D4, S2 all labs).

    RED at 2f22635f: ``award_blocked_supersession`` released on ANY strict
    superset of ``budget.explicit_grants``.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        candidate = _granted_failure_over(tmp_path, reducer, persisted, blocked_award_revision=None)
        # The pre-check refuses...
        assert reducer.award_blocked_supersession(candidate) is not None
        # ...and so does the append: an ordinary requeue grant authorizes
        # WORK, never the destruction of a verified award.
        with pytest.raises(AwardSupersessionError):
            reducer.append_model(candidate)
        head = [row for row in scan_jsonl(paths.models) if row["stable_id"] == "m_example"][-1]
        assert head["record_revision"] == persisted["record_revision"]
        assert head["status"]["kind"] == "runs"


def test_a_candidate_carried_scar_no_longer_releases_a_live_award_downgrade(
    tmp_path: Path,
) -> None:
    """The producer can never waive its own tripwire at live append.

    RED at 2f22635f: the reducer accepted any candidate-carried scar -- the
    exact arm resume-4 used to replace three verified awards with failures.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        candidate, failed_attempt = _grantless_failure_over(persisted)
        reducer.append_attempt(failed_attempt)
        scarred = deepcopy(candidate)
        scarred["scar_history"] = [
            f"{AWARD_SUPERSESSION_SCAR_PREFIX}{persisted['record_revision']}"
        ]
        assert reducer.award_blocked_supersession(scarred) is not None
        with pytest.raises(AwardSupersessionError):
            reducer.append_model(scarred)


def test_an_award_naming_grant_releases_the_downgrade(tmp_path: Path) -> None:
    """The one legitimate downgrade authority: a grant naming the exact award."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        candidate = _granted_failure_over(
            tmp_path,
            reducer,
            persisted,
            blocked_award_revision=str(persisted["record_revision"]),
        )
        # The receipt of the decision rides the record: the deliberate scar.
        candidate["scar_history"] = [
            f"{AWARD_SUPERSESSION_SCAR_PREFIX}{persisted['record_revision']}"
        ]
        assert reducer.award_blocked_supersession(candidate) is None
        from menagerie.crawler.driver import (
            _award_naming_grant_id,
            _reconcile_award_custody,
        )

        grant_id = candidate["budget"]["explicit_grants"][0]
        assert _award_naming_grant_id(paths, candidate, persisted) == grant_id
        prepared = reducer.prepare_model(candidate)
        census = _reconcile_award_custody(
            (persisted, prepared), baseline_row_count=1, records_root=tmp_path
        )
        assert census["award_downgrades"][0]["authority"] == "award-naming-grant"
        result = reducer.append_model(candidate)
        assert result.record["status"]["kind"] == "failed"
        assert (
            f"{AWARD_SUPERSESSION_SCAR_PREFIX}{persisted['record_revision']}"
            in result.record["scar_history"]
        )


def test_a_grant_naming_a_different_award_revision_releases_nothing(
    tmp_path: Path,
) -> None:
    """Exact identity binding: a wrong-revision naming grant is no authority."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        candidate = _granted_failure_over(
            tmp_path, reducer, persisted, blocked_award_revision="sha256:" + "0" * 64
        )
        assert reducer.award_blocked_supersession(candidate) is not None
        with pytest.raises(AwardSupersessionError):
            reducer.append_model(candidate)


def test_historical_scarred_supersessions_stay_replay_acceptable(tmp_path: Path) -> None:
    """The m5273 chain regression (fable's S2 rider, mandatory).

    m5273's CURRENT head (seq 206) is accepted on replay precisely via its
    self-stamped scar; removing the scar escape from replay would project that
    chain stale forever. Historical scarred rows are ledger bytes -- the
    projection must keep accepting them while live appends refuse.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        candidate, failed_attempt = _grantless_failure_over(persisted)
        reducer.append_attempt(failed_attempt)
        scarred = deepcopy(candidate)
        scarred["scar_history"] = [
            f"{AWARD_SUPERSESSION_SCAR_PREFIX}{persisted['record_revision']}"
        ]
        # m5273's head is review-held: the eater carried human_review.required.
        scarred["status"]["human_review"] = {
            "required": True,
            "reason": None,
            "queue": "crawler-human-review",
            "requested_at": scarred["created_at"],
        }
        # Write the scarred supersession as HISTORY (ledger bytes, exactly the
        # m5273 shape) without the live admission path.
        prepared = reducer.prepare_model(scarred)
        reducer._models.append(prepared)  # noqa: SLF001 -- historical bytes fixture
    context = make_authority_context(["m_example"])
    projection = project_dependency_current(paths, context=context)
    assert "m_example" in projection.current_records, projection.stale_reasons.get("m_example")
    head = projection.current_records["m_example"]
    assert head["status"]["kind"] == "failed"
    assert any(
        scar.startswith(AWARD_SUPERSESSION_SCAR_PREFIX) for scar in head.get("scar_history", ())
    )


def test_award_park_binds_the_reducer_returned_raw_award_head(tmp_path: Path) -> None:
    """A refused downgrade parks with the RAW blocked head bound, non-null.

    RED at 2f22635f: all ten close-state AwardSupersessionError park rows
    carry ``blocked_parent_revision: null`` because the driver read the
    dependency-currency projection (a stale award reads as nonexistent); the
    reducer's exception carried the raw head all along.
    """

    from menagerie.crawler.driver_admission import _park_award_supersession
    from menagerie.crawler.grant_parks import (
        AWARD_PARK_DISPOSITION,
        grant_park_path,
        load_grant_park_rows,
    )

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        candidate, failed_attempt = _grantless_failure_over(persisted)
        reducer.append_attempt(failed_attempt)
        prepared = reducer.prepare_model(candidate)
        with pytest.raises(AwardSupersessionError) as refused:
            reducer.append_model(prepared)

    events = tmp_path / "operational" / "events.jsonl"
    appended: list[dict] = []

    class _Ledger:
        def append(self, event):  # noqa: ANN001
            appended.append(dict(event))

    driver = SimpleNamespace(
        dependencies=SimpleNamespace(clock=lambda: "2026-08-16T00:00:00Z"),
        paths=SimpleNamespace(ledgers=paths),
        config=SimpleNamespace(
            run_id="run-test", machine_id="machine-test", campaign_id="campaign-test"
        ),
        _machine_deferral_rows={},
    )
    item = SimpleNamespace(
        stable_id="m_example",
        active_work_id="work-m_example-redrive",
        intake=SimpleNamespace(name="Example"),
    )
    del events
    row = _park_award_supersession(driver, item, prepared, refused.value, _Ledger())
    assert row["pending"]["blocked_parent_revision"] == persisted["record_revision"]
    assert row["disposition"] == AWARD_PARK_DISPOSITION
    assert row["pending"]["pending_model_sha256"] == stable_hash(prepared)
    persisted_rows = load_grant_park_rows([grant_park_path(tmp_path)])
    assert len(persisted_rows) == 1
    assert persisted_rows[0]["row_sha256"] == row["row_sha256"]
    assert len(appended) == 1
    assert appended[0]["status"] == "degraded"
    assert appended[0]["details"]["disposition"] == "award-supersession-parked"
    assert appended[0]["details"]["blocked_parent_revision"] == persisted["record_revision"]


def test_a_voided_award_naming_grant_releases_nothing(tmp_path: Path) -> None:
    """Only a LIVE grant releases custody (red-team break 6, opus #7).

    ``_introduced_grant_names_award`` read the grants and never the voids, so
    an operator revocation (the step-100 remedy: 23 grants voided against a
    running/killed leg) was silently ineffective at the ONE guard that
    matters -- the terminal append still superseded the award on an authority
    that no longer existed.
    """

    from menagerie.crawler.checkpoint import requeue_void_identity
    from menagerie.crawler.constants import REQUEUE_GRANT_VOID_SCHEMA_VERSION

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        candidate = _granted_failure_over(
            tmp_path,
            reducer,
            persisted,
            blocked_award_revision=str(persisted["record_revision"]),
        )
        grant_id = candidate["budget"]["explicit_grants"][0]
        # The operator voids the grant while the leg is in flight (the void
        # landing beside a consumption that raced it is exactly the killed-leg
        # window); the revocation must hold at the custody guard.
        reason = "operator withdrew the award adjudication"
        void = {
            "void_id": requeue_void_identity(
                void_of=grant_id, reason=reason, voided_by="operator-test"
            ),
            "schema_version": REQUEUE_GRANT_VOID_SCHEMA_VERSION,
            "void_of": grant_id,
            "reason": reason,
            "voided_by": "operator-test",
            "created_at": persisted["created_at"],
        }
        with (tmp_path / "operational" / "requeue-grants.jsonl").open("ab") as handle:
            handle.write(canonical_json_bytes(void) + b"\n")
        assert reducer.award_blocked_supersession(candidate) is not None, (
            "a VOIDED award-naming grant released the award-custody tripwire"
        )
        from menagerie.crawler.driver import (
            AwardCustodyError,
            _award_naming_grant_id,
            _reconcile_award_custody,
        )

        assert _award_naming_grant_id(paths, candidate, persisted) is None
        scarred = deepcopy(candidate)
        scarred["scar_history"] = [
            f"{AWARD_SUPERSESSION_SCAR_PREFIX}{persisted['record_revision']}"
        ]
        prepared = reducer.prepare_model(scarred)
        with pytest.raises(AwardCustodyError):
            _reconcile_award_custody(
                (persisted, prepared), baseline_row_count=1, records_root=tmp_path
            )
        with pytest.raises(AwardSupersessionError):
            reducer.append_model(candidate)
        head = [row for row in scan_jsonl(paths.models) if row["stable_id"] == "m_example"][-1]
        assert head["status"]["kind"] == "runs"


def test_a_foreign_members_award_naming_grant_releases_nothing(tmp_path: Path) -> None:
    """The release binds the grant's member, not just the revision string.

    Defense-in-depth from the same red-team pass: the lookup never checked the
    grant row's ``stable_id``, so a grant issued for member X would release
    member Y's custody if the revisions coincided.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        candidate, failed_attempt = _grantless_failure_over(persisted)
        reducer.append_attempt(failed_attempt)
        foreign = {
            "schema_version": "menagerie.crawler.requeue-grant.v1",
            "grant_id": "sha256:" + "f" * 64,
            "stable_id": "m_other",
            "stage": "source",
            "reason": "granted for a different member entirely",
            "attempts": 1,
            "granted_by": "operator-test",
            "new_work_generation": 1,
            "created_at": persisted["created_at"],
            "blocked_award_revision": str(persisted["record_revision"]),
        }
        operational = tmp_path / "operational"
        operational.mkdir(parents=True, exist_ok=True)
        with (operational / "requeue-grants.jsonl").open("ab") as handle:
            handle.write(canonical_json_bytes(foreign) + b"\n")
        candidate["budget"]["explicit_grants"] = [foreign["grant_id"]]
        assert reducer.award_blocked_supersession(candidate) is not None, (
            "another member's award-naming grant released this member's custody"
        )
        from menagerie.crawler.driver import (
            AwardCustodyError,
            _award_naming_grant_id,
            _reconcile_award_custody,
        )

        assert _award_naming_grant_id(paths, candidate, persisted) is None
        scarred = deepcopy(candidate)
        scarred["scar_history"] = [
            f"{AWARD_SUPERSESSION_SCAR_PREFIX}{persisted['record_revision']}"
        ]
        prepared = reducer.prepare_model(scarred)
        with pytest.raises(AwardCustodyError):
            _reconcile_award_custody(
                (persisted, prepared), baseline_row_count=1, records_root=tmp_path
            )


def test_close_time_custody_reconciles_every_raw_head_transition(tmp_path: Path) -> None:
    """Every leg-appended ``runs -> non-runs`` transition needs an exact action.

    Scalar award counts cannot see one eating balanced by one remint (sol's
    counterexample); the assertion reconciles transitions, and ``runs ->
    runs`` remints are censused for disclosure.
    """

    from menagerie.crawler.driver import AwardCustodyError, _reconcile_award_custody

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        candidate = _granted_failure_over(
            tmp_path,
            reducer,
            persisted,
            blocked_award_revision=str(persisted["record_revision"]),
        )
        candidate["scar_history"] = [
            f"{AWARD_SUPERSESSION_SCAR_PREFIX}{persisted['record_revision']}"
        ]
        reducer.append_model(candidate)

    rows = tuple(scan_jsonl(paths.models))
    # The granted, scarred downgrade reconciles.
    census = _reconcile_award_custody(rows, baseline_row_count=0, records_root=tmp_path)
    assert census["award_downgrades"] == [
        {
            "stable_id": "m_example",
            "superseded_revision": persisted["record_revision"],
            "authority": "award-naming-grant",
        }
    ]
    assert census["award_remints"] == []
    from menagerie.crawler.partition_proof import award_transition_census

    from menagerie.crawler.leg_generation import MODELS_SOURCE_NAMES
    from menagerie.crawler.partition_proof import PartitionProofError

    zero_floor = {name: {"rows": 0} for name in MODELS_SOURCE_NAMES}
    bound = award_transition_census(tmp_path, zero_floor)
    assert bound["instrumentation"] == "instrumented"
    assert [row["stable_id"] for row in bound["runs_to_non_runs"]] == ["m_example"]
    assert bound["runs_to_runs"] == []
    uninstrumented = award_transition_census(tmp_path, None)
    assert uninstrumented["instrumentation"] == "uninstrumented"
    assert uninstrumented["runs_to_non_runs"] == []
    assert uninstrumented["runs_to_runs"] == []
    with pytest.raises(PartitionProofError, match="partial award-transition baseline"):
        award_transition_census(tmp_path, {"models.flat": {"rows": 0}})

    # A transition appended after the previous close but before this leg opens
    # belongs to the next instrumented census. The ordinary leg baseline sits
    # at the current extent and would otherwise produce a false observed zero.
    current_floor = {
        name: {"rows": 2 if name == "models.flat" else 0}
        for name in MODELS_SOURCE_NAMES
    }
    gap_closed = award_transition_census(
        tmp_path,
        current_floor,
        coverage_baselines=zero_floor,
    )
    assert [row["stable_id"] for row in gap_closed["runs_to_non_runs"]] == ["m_example"]

    # Strip the scar from the persisted bytes: the same transition without its
    # receipt must refuse at close.
    tampered = []
    for row in rows:
        row = dict(row)
        if row["status"]["kind"] == "failed":
            row["scar_history"] = []
            row["budget"] = dict(row["budget"], explicit_grants=[])
        tampered.append(row)
    with pytest.raises(AwardCustodyError):
        _reconcile_award_custody(tuple(tampered), baseline_row_count=0, records_root=tmp_path)

    # Transitions already in the ledger BEFORE the leg are history, not this
    # leg's custody debt.
    census = _reconcile_award_custody(
        tuple(tampered), baseline_row_count=len(tampered), records_root=tmp_path
    )
    assert census["award_downgrades"] == []


def test_a_remint_is_censused_not_refused(tmp_path: Path) -> None:
    """runs -> runs supersession stays admissible and becomes disclosure.

    Adjudicated (S2 opus correction 3 vs sol): the reducer keeps accepting
    award-over-award -- admission control belongs to the work-selection
    policy -- but the close census now names every remint (the 63 silent
    ones at close are the incident evidence).
    """

    from menagerie.crawler.driver import _reconcile_award_custody

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        persisted = _runs_head(reducer)
        successor = make_model(accepted=True)
        successor["parent_revision"] = persisted["record_revision"]
        successor["status"]["supersedes_revision"] = persisted["record_revision"]
        successor["record_seq"] = int(persisted["record_seq"]) + 1
        successor["notes"] = "fresh re-award under the current dependency tree"
        result = reducer.append_model(successor)
        assert result.record["status"]["kind"] == "runs"

    rows = tuple(scan_jsonl(paths.models))
    census = _reconcile_award_custody(rows, baseline_row_count=0, records_root=tmp_path)
    assert census["award_remints"] == [
        {
            "stable_id": "m_example",
            "superseded_revision": persisted["record_revision"],
            "reminted_revision": result.record["record_revision"],
        }
    ]
    assert census["award_downgrades"] == []
    from menagerie.crawler.partition_proof import award_transition_census

    from menagerie.crawler.leg_generation import MODELS_SOURCE_NAMES

    bound = award_transition_census(
        tmp_path, {name: {"rows": 0} for name in MODELS_SOURCE_NAMES}
    )
    assert bound["instrumentation"] == "instrumented"
    assert bound["runs_to_non_runs"] == []
    assert [row["stable_id"] for row in bound["runs_to_runs"]] == ["m_example"]
