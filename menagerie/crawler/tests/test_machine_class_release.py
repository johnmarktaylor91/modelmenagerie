"""W14 machine-classified release: the four signed class widenings of waiver #2.

Each class is a JMT-signed C.8 four-part arm extension (recorded 2026-08-24) and
must independently prove: (1) disposition-invariance -- eligibility never
upgrades an outcome; (2) a CLOSED proof rule with a near-miss matrix; (3)
replay-derivation with zero producer input (B2-X-V29 keyword injection: producer
text cannot self-earn a retry); (4) a per-class segregated census arm. The O-B2
HARD CONSTRAINT runs everywhere: one model-owned failing attempt in the
witnessed generation refuses every class.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
)
from menagerie.crawler.identity import payload_hash, stamp_row_digest
from menagerie.crawler.infra_release import (
    derive_infra_terminal_release,
    derive_machine_class_release,
)
from menagerie.crawler.quota_deferral import (
    QUOTA_DEFERRAL_MAX_GRANTS,
    mint_quota_deferral,
)
from menagerie.crawler.release_vocabulary import (
    CACHE_INVALIDATION_FAULT_EVENT_KIND,
    MACHINE_RELEASE_CLASSES,
    MACHINE_RELEASE_CLASS_BLOCKED_PREREQUISITE,
    MACHINE_RELEASE_CLASS_CORPUS_CONFLICT,
    MACHINE_RELEASE_CLASS_ENGINE_CACHE,
    MACHINE_RELEASE_CLASS_QUOTA_DEFERRAL,
    PREREQUISITE_PROVISION_SCHEMA_VERSION,
    SIGNED_MACHINE_RELEASE_CLASSES,
    build_blocked_prerequisite_record,
)
from menagerie.crawler.status import GRANT_RELEASE_CENSUS_CLASSES, grant_release_census
from menagerie.crawler.tests.conftest import (
    make_failed_attempt,
    rebind_nonaward_parent_proof,
)

#: Load-bearing kernel-change tests: sub-second, in the mandatory smoke gate.
pytestmark = pytest.mark.smoke

STABLE_ID = "m_class_release"
WORK_ID = "work-gen-1"
NOW = "2026-08-24T12:00:00Z"
LATER = "2026-08-25T12:00:00+00:00"
EARLIER = "2026-08-23T12:00:00+00:00"
GATE_ID = "gate-1234"
RESULT_SHA = "sha256:" + "c" * 64


def _sealed(attempt: dict[str, Any], *, seq: int = 1) -> dict[str, Any]:
    """Stamp the exact ledger binding a canonical attempt row carries on disk."""

    attempt["ledger_seq"] = seq
    attempt.pop("payload_sha256", None)
    attempt["payload_sha256"] = payload_hash(attempt)
    return attempt


def _terminal_record(
    stable_id: str = STABLE_ID,
    *,
    stage: Optional[str] = "author",
    code: Optional[str] = None,
    reason_code: Optional[str] = "effort-exhausted:author-sessions",
    attempt_ids: tuple[str, ...] = ("attempt-1",),
    review_required: bool = False,
    revision: str = "sha256:" + "a" * 64,
) -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "record_revision": revision,
        "parent_revision": None,
        "record_seq": 1,
        "status": {
            "kind": "failed",
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


def _exhaustion_attempt(attempt_id: str = "attempt-1", *, attempt_no: int = 1) -> dict[str, Any]:
    attempt = make_failed_attempt(
        STABLE_ID,
        attempt_id=attempt_id,
        stage="author",
        reason_code="effort-exhausted:author-sessions",
    )
    attempt["work_id"] = WORK_ID
    attempt["attempt_no"] = attempt_no
    attempt["error"]["exception_type"] = (
        "menagerie.crawler.driver_contracts.AuthorEffortCapExceeded"
    )
    attempt["error"]["details"] = {"driver_observed": True, "environment": None}
    return _sealed(rebind_nonaward_parent_proof(attempt))


def _corpus_conflict_attempt(
    attempt_id: str = "attempt-1", *, attempt_no: int = 1, reason: str = "internal-error"
) -> dict[str, Any]:
    attempt = make_failed_attempt(
        STABLE_ID, attempt_id=attempt_id, stage="runner", reason_code=reason
    )
    attempt["work_id"] = WORK_ID
    attempt["attempt_no"] = attempt_no
    attempt["actor"] = "driver"
    attempt["error"]["exception_type"] = "menagerie.crawler.recordio.LedgerConflictError"
    attempt["error"]["details"] = {"driver_observed": True, "environment": None}
    return _sealed(rebind_nonaward_parent_proof(attempt))


def _blocked_claim_attempt(attempt_id: str = "attempt-1", *, attempt_no: int = 1) -> dict[str, Any]:
    attempt = make_failed_attempt(
        STABLE_ID, attempt_id=attempt_id, stage="author", reason_code="malformed-result"
    )
    attempt["work_id"] = WORK_ID
    attempt["attempt_no"] = attempt_no
    attempt["error"]["exception_type"] = (
        "menagerie.crawler.driver_contracts.AuthorBlockedPrerequisite"
    )
    attempt["error"]["details"] = {"driver_observed": True, "environment": None}
    return _sealed(rebind_nonaward_parent_proof(attempt))


def _engine_fault_attempt(attempt_id: str = "attempt-1", *, attempt_no: int = 1) -> dict[str, Any]:
    attempt = make_failed_attempt(
        STABLE_ID, attempt_id=attempt_id, stage="environment", reason_code="build-failed"
    )
    attempt["work_id"] = WORK_ID
    attempt["attempt_no"] = attempt_no
    attempt["actor"] = "driver"
    attempt["error"]["exception_type"] = "menagerie.crawler.env_lifecycle.DiskRecoveryError"
    attempt["error"]["details"] = {"driver_observed": True, "environment": None}
    return _sealed(rebind_nonaward_parent_proof(attempt))


def _model_owned_attempt(attempt_id: str = "attempt-owned", *, attempt_no: int = 2) -> dict[str, Any]:
    attempt = make_failed_attempt(
        STABLE_ID, attempt_id=attempt_id, stage="forward", reason_code="mode-run"
    )
    attempt["work_id"] = WORK_ID
    attempt["attempt_no"] = attempt_no
    attempt["actor"] = "worker"
    attempt["supervisor_observation"]["exit_code"] = 1
    return _sealed(rebind_nonaward_parent_proof(attempt))


def _seam_record(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = dict(
        stable_id=STABLE_ID,
        work_id=WORK_ID,
        stage="environment",
        prerequisite_tokens=("runtime-dependency:jsonschema",),
        gate_id=GATE_ID,
        author_result_sha256=RESULT_SHA,
        campaign_id="campaign-test",
        run_id="run-1",
        machine_id="machine-1",
        created_at=NOW,
    )
    fields.update(overrides)
    return build_blocked_prerequisite_record(**fields)


def _blocked_gate(
    *, verdict: str = "accepted", author_result_sha256: str = RESULT_SHA, gate_id: str = GATE_ID
) -> dict[str, Any]:
    return {
        "gate_kind": "terminal_disposition",
        "gate_id": gate_id,
        "items": [
            {
                "stable_id": STABLE_ID,
                "work_id": WORK_ID,
                "verdict": "accurate",
                "terminal_disposition": {
                    "kind": "BLOCKED",
                    "predicate": "blocked-prerequisite",
                    "verdict": verdict,
                    "author_result_sha256": author_result_sha256,
                },
            }
        ],
    }


def _provision_witness(
    *, token: str = "runtime-dependency:jsonschema", registered_at: str = LATER
) -> dict[str, Any]:
    return stamp_row_digest(
        {
            "schema_version": PREREQUISITE_PROVISION_SCHEMA_VERSION,
            "token": token,
            "provided_by_rotation": "rotation-2026-08-25-graph",
            "environment_intent": "graph",
            "registered_at": registered_at,
        }
    )


def _fault_event(*, reason: str = "engine-fault", work_id: str = WORK_ID) -> dict[str, Any]:
    return {
        "event_kind": CACHE_INVALIDATION_FAULT_EVENT_KIND,
        "details": {
            "stable_id": STABLE_ID,
            "work_id": work_id,
            "cache_identity": "solve-cache",
            "artifact_transaction_id": "atx-1",
            "reason": reason,
            "fault_exception_type": "menagerie.crawler.env_lifecycle.DiskRecoveryError",
        },
    }


# ---------------------------------------------------------------------------
# The signed included-class set is pinned exactly (the merge receipt binds it).


def test_signed_class_set_is_exactly_the_four_signed_arms() -> None:
    assert SIGNED_MACHINE_RELEASE_CLASSES == MACHINE_RELEASE_CLASSES
    assert MACHINE_RELEASE_CLASSES == frozenset(
        {
            "quota-deferral",
            "corpus-conflict",
            "blocked-prerequisite-post-rotation",
            "engine-invalidated-artifact-cache",
        }
    )
    machine_arms = [name for name in GRANT_RELEASE_CENSUS_CLASSES if name.startswith("machine-released:")]
    assert sorted(machine_arms) == sorted(
        f"machine-released:{name}" for name in MACHINE_RELEASE_CLASSES
    )


# ---------------------------------------------------------------------------
# quota-deferral


def test_quota_deferral_class_binds_an_all_machine_exhaustion_terminal() -> None:
    release = derive_machine_class_release(
        _terminal_record(), [_exhaustion_attempt()], []
    )
    assert release is not None
    assert release.release_class == MACHINE_RELEASE_CLASS_QUOTA_DEFERRAL
    assert release.work_id == WORK_ID
    assert any(b.startswith("driver-effort-exhaustion:") for b in release.witness_bases)


def test_quota_deferral_near_misses_refuse() -> None:
    attempts = [_exhaustion_attempt()]
    # Review-held: the review lane owns it.
    assert (
        derive_machine_class_release(
            _terminal_record(review_required=True), attempts, []
        )
        is None
    )
    # Non-exhaustion head with no valve row: not this class (and no other).
    assert (
        derive_machine_class_release(
            _terminal_record(stage="runner", reason_code="missing-receipt"),
            attempts,
            [],
        )
        is None
    )
    # One model-owned failing attempt in the generation: O-B2 refuses.
    record = _terminal_record(attempt_ids=("attempt-1",))
    assert (
        derive_machine_class_release(
            record, [_exhaustion_attempt(), _model_owned_attempt()], []
        )
        is None
    )


def test_quota_deferral_recidivist_park_holds_for_a_human(tmp_path: Any) -> None:
    ledger = tmp_path / "quota-deferrals.jsonl"
    for ordinal in range(QUOTA_DEFERRAL_MAX_GRANTS + 1):
        row = mint_quota_deferral(
            ledger,
            stable_id=STABLE_ID,
            work_id=WORK_ID,
            campaign_id="campaign-test",
            run_id=f"run-{ordinal}",
            machine_id="machine-1",
            created_at=NOW,
            evidence={"witness": "author-sessions-exhaustion-classifier"},
        )
    assert row["disposition"] == "review-held:storm-recidivist"
    release = derive_machine_class_release(
        _terminal_record(),
        [_exhaustion_attempt()],
        [],
        valve_rows=[row],
    )
    assert release is None


def test_quota_deferral_valve_row_binds_a_non_exhaustion_head(tmp_path: Any) -> None:
    ledger = tmp_path / "quota-deferrals.jsonl"
    valve = mint_quota_deferral(
        ledger,
        stable_id=STABLE_ID,
        work_id=WORK_ID,
        campaign_id="campaign-test",
        run_id="run-1",
        machine_id="machine-1",
        created_at=NOW,
        evidence={"witness": "cross-model-storm-fuse"},
    )
    assert valve["disposition"] == "machine-deferred:quota-storm"
    record = _terminal_record(stage="runner", reason_code="missing-receipt")
    release = derive_machine_class_release(
        record, [_exhaustion_attempt()], [], valve_rows=[valve]
    )
    assert release is not None
    assert release.release_class == MACHINE_RELEASE_CLASS_QUOTA_DEFERRAL


# ---------------------------------------------------------------------------
# corpus-conflict


def test_corpus_conflict_class_binds_a_typed_ledger_conflict_death() -> None:
    record = _terminal_record(stage="runner", reason_code="internal-error")
    release = derive_machine_class_release(record, [_corpus_conflict_attempt()], [])
    assert release is not None
    assert release.release_class == MACHINE_RELEASE_CLASS_CORPUS_CONFLICT
    assert any(b.startswith("corpus-conflict:") for b in release.witness_bases)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.__setitem__("actor", "worker"),
        lambda a: a["error"].__setitem__("reason_code", "build-failed"),
        lambda a: a["error"].__setitem__(
            "exception_type", "menagerie.crawler.reducer.ReductionError"
        ),
        lambda a: a["error"].__setitem__("native_crash", True),
        lambda a: a["error"]["details"].__setitem__("driver_observed", False),
    ],
)
def test_corpus_conflict_near_misses_refuse(mutate: Any) -> None:
    attempt = _corpus_conflict_attempt()
    mutate(attempt)
    attempt = _sealed(rebind_nonaward_parent_proof(attempt))
    record = _terminal_record(stage="runner", reason_code="internal-error")
    assert derive_machine_class_release(record, [attempt], []) is None


def test_corpus_conflict_with_a_model_owned_sibling_refuses() -> None:
    record = _terminal_record(stage="runner", reason_code="internal-error")
    attempts = [_corpus_conflict_attempt(), _model_owned_attempt()]
    assert derive_machine_class_release(record, attempts, []) is None


# ---------------------------------------------------------------------------
# blocked-prerequisite-post-rotation


def _blocked_record() -> dict[str, Any]:
    return _terminal_record(
        stage="environment",
        code="failed:environment",
        reason_code="blocked-prerequisite:environment",
    )


def test_blocked_prerequisite_class_binds_with_full_evidence_chain() -> None:
    release = derive_machine_class_release(
        _blocked_record(),
        [_blocked_claim_attempt()],
        [_blocked_gate()],
        blocked_prerequisite_rows=[_seam_record()],
        provision_witnesses=[_provision_witness()],
    )
    assert release is not None
    assert release.release_class == MACHINE_RELEASE_CLASS_BLOCKED_PREREQUISITE
    assert "blocked-prerequisite-claim" in release.witness_bases


def test_blocked_prerequisite_near_miss_matrix() -> None:
    record = _blocked_record()
    attempts = [_blocked_claim_attempt()]
    seam = _seam_record()
    gate = _blocked_gate()
    witness = _provision_witness()
    # (a) Era rule: no typed seam record (every 200a terminal) -> human lane.
    assert (
        derive_machine_class_release(
            record, attempts, [gate], provision_witnesses=[witness]
        )
        is None
    )
    # (b) The claim was never adjudicated accepted.
    assert (
        derive_machine_class_release(
            record,
            attempts,
            [_blocked_gate(verdict="rejected")],
            blocked_prerequisite_rows=[seam],
            provision_witnesses=[witness],
        )
        is None
    )
    # (c) The gate adjudicated a DIFFERENT author result than the seam binds.
    assert (
        derive_machine_class_release(
            record,
            attempts,
            [_blocked_gate(author_result_sha256="sha256:" + "d" * 64)],
            blocked_prerequisite_rows=[seam],
            provision_witnesses=[witness],
        )
        is None
    )
    # (d) Provision registered BEFORE the claim: the claim was false when made
    #     (or the witness is a fix-provided narrative) -- refuse.
    assert (
        derive_machine_class_release(
            record,
            attempts,
            [gate],
            blocked_prerequisite_rows=[seam],
            provision_witnesses=[_provision_witness(registered_at=EARLIER)],
        )
        is None
    )
    # (e) A token with NO witness at all.
    assert (
        derive_machine_class_release(
            record,
            attempts,
            [gate],
            blocked_prerequisite_rows=[seam],
            provision_witnesses=[_provision_witness(token="runtime-dependency:numpy")],
        )
        is None
    )
    # (f) Ambiguous typed naming: two seam rows for one generation.
    second = _seam_record(created_at=LATER)
    assert (
        derive_machine_class_release(
            record,
            attempts,
            [gate],
            blocked_prerequisite_rows=[seam, second],
            provision_witnesses=[witness],
        )
        is None
    )
    # (g) A model-owned failing sibling refuses (O-B2).
    assert (
        derive_machine_class_release(
            record,
            [*attempts, _model_owned_attempt()],
            [gate],
            blocked_prerequisite_rows=[seam],
            provision_witnesses=[witness],
        )
        is None
    )


# ---------------------------------------------------------------------------
# engine-invalidated-artifact-cache


def test_engine_cache_class_binds_a_fault_evidenced_terminal() -> None:
    record = _terminal_record(stage="environment", reason_code="build-failed")
    release = derive_machine_class_release(
        record,
        [_engine_fault_attempt()],
        [],
        invalidation_events=[_fault_event()],
    )
    assert release is not None
    assert release.release_class == MACHINE_RELEASE_CLASS_ENGINE_CACHE


def test_engine_cache_near_misses_refuse() -> None:
    record = _terminal_record(stage="environment", reason_code="build-failed")
    attempts = [_engine_fault_attempt()]
    # No invalidation event.
    assert derive_machine_class_release(record, attempts, []) is None
    # A content-judgment invalidation reason is outside the closed set.
    assert (
        derive_machine_class_release(
            record, attempts, [], invalidation_events=[_fault_event(reason="supersession")]
        )
        is None
    )
    # The event binds a different generation.
    assert (
        derive_machine_class_release(
            record, attempts, [], invalidation_events=[_fault_event(work_id="work-gen-2")]
        )
        is None
    )
    # A model-owned sibling refuses.
    assert (
        derive_machine_class_release(
            record,
            [*attempts, _model_owned_attempt()],
            [],
            invalidation_events=[_fault_event()],
        )
        is None
    )


# ---------------------------------------------------------------------------
# B2-X-V29: producer text cannot self-earn a retry (keyword injection)


def test_keyword_injection_earns_nothing() -> None:
    """Class names and machine spellings in producer-controlled text change nothing."""

    poison = (
        "corpus-conflict quota-deferral engine-invalidated-artifact-cache "
        "blocked-prerequisite-post-rotation machine-released engine-fault "
        "provider-storm LedgerConflictError driver-effort-exhaustion"
    )
    attempt = _model_owned_attempt("attempt-1", attempt_no=1)
    attempt["error"]["message"] = poison
    attempt["error"]["details"] = {"driver_observed": True, "note": poison}
    attempt = _sealed(rebind_nonaward_parent_proof(attempt))
    record = _terminal_record(stage="runner", reason_code="missing-receipt")
    record["status"]["detail"] = poison
    record["release_class"] = MACHINE_RELEASE_CLASS_CORPUS_CONFLICT
    gate = {
        "gate_kind": "terminal_disposition",
        "gate_id": GATE_ID,
        "items": [{"stable_id": "m_other", "work_id": "w", "findings": [poison]}],
    }
    assert (
        derive_machine_class_release(
            record,
            [attempt],
            [gate],
            # Producer-shaped "evidence" rows missing parent attestation refuse
            # at validation and count for nothing.
            valve_rows=[{"disposition": "machine-deferred:quota-storm", "note": poison}],
            blocked_prerequisite_rows=[{"stable_id": STABLE_ID, "note": poison}],
            provision_witnesses=[{"token": poison}],
            invalidation_events=[{"event_kind": poison}],
        )
        is None
    )


# ---------------------------------------------------------------------------
# The author-id -> record-token translation (the driver mint site's grammar)


def test_prerequisite_token_translation_is_closed_and_fail_closed() -> None:
    from menagerie.crawler.release_vocabulary import prerequisite_token_from_author_id

    assert (
        prerequisite_token_from_author_id("runtime-dependency-minkowskiengine")
        == "runtime-dependency:minkowskiengine"
    )
    assert (
        prerequisite_token_from_author_id("MinkowskiEngine-package")
        == "runtime-dependency:minkowskiengine"
    )
    assert (
        prerequisite_token_from_author_id("torch_scatter-distribution")
        == "runtime-dependency:torch-scatter"
    )
    # The bare class marker names no distribution; CUDA and free-form ids are
    # not runtime deps: all fail closed toward the human lane.
    for untranslatable in ("runtime-dependency", "cuda-device", "impl-mmcv-carafe-op", ""):
        assert prerequisite_token_from_author_id(untranslatable) is None


# ---------------------------------------------------------------------------
# Census: per-class segregated arms, conservative precedence


def _consumed_event(grant_id: str, revision: str) -> dict[str, Any]:
    return {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
        "details": {
            "grant_id": grant_id,
            "stable_id": STABLE_ID,
            "source_record_revision": revision,
        },
    }


def test_census_segregates_machine_class_release_per_class() -> None:
    record = _terminal_record()
    census = grant_release_census(
        model_revisions=[record],
        operational_events=[_consumed_event("grant-q", record["record_revision"])],
        attempt_records=[_exhaustion_attempt()],
        gate_records=[],
    )
    assert census["machine-released:quota-deferral"] == ("grant-q",)
    assert census["review-released"] == ()
    assert census["infra-waiver-released"] == ()


def test_census_review_release_survives_when_no_class_derives() -> None:
    record = _terminal_record(
        stage="accuracy-gate",
        reason_code="terminal-disposition-rejected",
        review_required=True,
    )
    census = grant_release_census(
        model_revisions=[record],
        operational_events=[_consumed_event("grant-r", record["record_revision"])],
        attempt_records=[],
        gate_records=[],
    )
    assert census["review-released"] == ("grant-r",)
    for name in MACHINE_RELEASE_CLASSES:
        assert census[f"machine-released:{name}"] == ()


def test_base_infra_waiver_is_untouched_by_the_class_engine() -> None:
    """The registered instance's own rule still derives exactly as before."""

    release = derive_infra_terminal_release(
        _terminal_record(), [_exhaustion_attempt()], []
    )
    assert release is not None
    assert release.work_id == WORK_ID


# ---------------------------------------------------------------------------
# Consumption e2e: the REAL grant consumer binds through the class lane


def test_consumption_binds_a_corpus_conflict_terminal_via_the_class_lane(
    tmp_path: Any,
) -> None:
    """A runner-stage ledger-conflict death is outside the BASE waiver's rows
    (row 3 is author-stage), so before W14 its grant held ``unreviewed-terminal``
    forever; the signed corpus-conflict class now binds it, disclosing the
    machine-class authority and its witness bases on the consumption event."""

    from types import SimpleNamespace

    from menagerie.crawler.checkpoint import (
        append_canonical_requeue_grant,
        build_canonical_requeue_grant,
        canonical_requeue_grants_path,
    )
    from menagerie.crawler.driver_admission import AdmissionEnvironmentMixin
    from menagerie.crawler.driver_contracts import DriverPaths
    from menagerie.crawler.identity import canonical_json_bytes
    from menagerie.crawler.recordio import JsonlLedger
    from menagerie.crawler.reducer import default_ledger_paths

    paths = DriverPaths(
        runtime_root=tmp_path / ".crawl-local",
        intake_root=tmp_path / "intake",
        ledgers=default_ledger_paths(tmp_path / "records"),
    )
    stub = SimpleNamespace(
        paths=paths,
        registry=SimpleNamespace(intents={}),
        dependencies=SimpleNamespace(
            clock=lambda: NOW, boundary_hook=lambda *args, **kwargs: None
        ),
        config=SimpleNamespace(run_id="test-run", machine_id="test-machine"),
    )
    witness = _corpus_conflict_attempt()
    paths.ledgers.attempts.parent.mkdir(parents=True, exist_ok=True)
    with paths.ledgers.attempts.open("wb") as handle:
        handle.write(canonical_json_bytes(witness) + b"\n")
    record = _terminal_record(stage="runner", reason_code="internal-error")
    paths.ledgers.models.parent.mkdir(parents=True, exist_ok=True)
    with paths.ledgers.models.open("wb") as handle:
        handle.write(canonical_json_bytes(record) + b"\n")
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    grant = build_canonical_requeue_grant(
        canonical,
        stable_id=STABLE_ID,
        stage="runner",
        reason="operator approved corpus-conflict re-drive",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(canonical, grant)

    with JsonlLedger(
        paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION
    ) as operational:
        result = AdmissionEnvironmentMixin._consume_requeue_grants(
            stub, operational, frozenset({STABLE_ID})
        )
        events = list(operational.records)

    assert result[STABLE_ID]["active"] is True
    consumptions = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    assert len(consumptions) == 1
    details = consumptions[0]["details"]
    assert details["release_authority"] == "machine-class:corpus-conflict"
    assert any(
        basis.startswith("corpus-conflict:")
        for basis in details["machine_class_witness_bases"]
    )
    assert "infra_release_witness_bases" not in details
