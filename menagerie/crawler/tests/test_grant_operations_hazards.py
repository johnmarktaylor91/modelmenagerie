"""Grant-operations hazards: voidability, idempotent issuance, scoped refusals.

Docket C-GRANT-OPS (T1-13 + T2-05). Four operator hazards, each pinned RED
before its fix:

- An out-of-intake grant used to be unvoidable: the validator's single
  interleaved pass raised on the grant row before any void row could neutralize
  it (and a void row scanned first referenced a not-yet-seen grant), so one
  foreign grant bricked every future admission. A void must neutralize it.
- A retried issuance minted a DOUBLE grant: ``generation`` is the ledger row
  count, so the identical command after a durable append hashed to a fresh
  ``grant_id`` and both grants stayed independently consumable. Issuance must
  refuse an outstanding logical twin (void-before-reissue).
- One model's unbindable outstanding grant (stage mismatch, unreviewed
  terminal) aborted the campaign globally at admission -- T2-01's leg-killer
  shape. The refusal must scope to the model: durable disclosure, no work
  bound, every other model's grant consumes.
- A review-held ``deferred:*`` terminal with ``status.stage=null`` was
  structurally un-grantable (T2-05): no member of the closed failure-stage
  vocabulary equals ``None``, so the operator's only move was a void. The
  typed ``deferred-terminal`` stage expresses exactly that class and nothing
  else.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    append_canonical_requeue_void,
    build_canonical_requeue_grant,
    build_canonical_requeue_void,
    canonical_requeue_grants_path,
)
from menagerie.crawler.constants import (
    DEFERRED_TERMINAL_GRANT_STAGE,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
)
from menagerie.crawler.driver_admission import (
    AdmissionEnvironmentMixin,
    _validated_requeue_grants,
)
from menagerie.crawler.driver_contracts import DriverIntegrationError, DriverPaths
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.reducer import default_ledger_paths

NOW = "2026-08-12T08:00:00Z"
STABLE_ID = "m_reviewed"
OTHER_ID = "m_other"


def _paths(tmp_path: Path) -> DriverPaths:
    """Return isolated driver paths below one temporary root.

    Parameters
    ----------
    tmp_path:
        Test-owned temporary directory.

    Returns
    -------
    DriverPaths
        Runtime and canonical ledger paths for the harness.
    """

    return DriverPaths(
        runtime_root=tmp_path / ".crawl-local",
        intake_root=tmp_path / "intake",
        ledgers=default_ledger_paths(tmp_path / "records"),
    )


def _stub_driver(paths: DriverPaths) -> Any:
    """Return the minimal driver facade ``_consume_requeue_grants`` reads.

    Parameters
    ----------
    paths:
        Harness driver paths.

    Returns
    -------
    Any
        Stub carrying paths, registry, dependencies, and config.
    """

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
    stage: Optional[str] = "accuracy-gate",
    kind: str = "failed",
    code: Optional[str] = None,
    review_required: bool = True,
    revision: str = "sha256:" + "a" * 64,
    record_seq: int = 1,
) -> Mapping[str, Any]:
    """Return one minimal persisted terminal model revision.

    Parameters
    ----------
    stable_id:
        Model identity.
    stage, kind, code, review_required:
        Recorded terminal status facts and human-review flag.
    revision, record_seq:
        Persisted revision identity and sequence.

    Returns
    -------
    Mapping[str, Any]
        Ledger-shaped model record.
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
            "human_review": {
                "queue": "crawler-human-review" if review_required else None,
                "reason": None,
                "requested_at": NOW if review_required else None,
                "required": review_required,
            },
        },
        "budget": {"explicit_grants": []},
    }


def _write_models(paths: DriverPaths, records: list[Mapping[str, Any]]) -> None:
    """Write raw model revisions to the harness models ledger.

    Parameters
    ----------
    paths:
        Harness driver paths.
    records:
        Ledger rows in append order.
    """

    paths.ledgers.models.parent.mkdir(parents=True, exist_ok=True)
    with paths.ledgers.models.open("wb") as handle:
        for record in records:
            handle.write(canonical_json_bytes(dict(record)) + b"\n")


def _append_grant(
    paths: DriverPaths,
    stable_id: str,
    *,
    stage: str = "accuracy-gate",
    reason: str = "operator approved re-adjudication",
) -> Mapping[str, Any]:
    """Append one tool-style grant to the canonical grant ledger.

    Parameters
    ----------
    paths:
        Harness driver paths.
    stable_id, stage, reason:
        Grant facts.

    Returns
    -------
    Mapping[str, Any]
        Appended grant payload.
    """

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


def _append_void(paths: DriverPaths, grant_id: str, *, reason: str) -> Mapping[str, Any]:
    """Append one operator void for an existing grant.

    Parameters
    ----------
    paths:
        Harness driver paths.
    grant_id:
        Exact grant being voided.
    reason:
        Operator attribution.

    Returns
    -------
    Mapping[str, Any]
        Appended void payload.
    """

    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    void = build_canonical_requeue_void(
        canonical,
        void_of=grant_id,
        reason=reason,
        voided_by="operator",
        created_at=NOW,
    )
    append_canonical_requeue_void(canonical, void)
    return void


def _consume(
    paths: DriverPaths, *, intake_ids: Optional[frozenset[str]] = None
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    """Run the real consumer against the harness ledgers.

    Parameters
    ----------
    paths:
        Harness driver paths.
    intake_ids:
        Trusted intake membership; defaults to the single harness model.

    Returns
    -------
    tuple[dict[str, Any], list[Mapping[str, Any]]]
        Consumer result map and persisted runtime operational events.
    """

    stub = _stub_driver(paths)
    with JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        result = AdmissionEnvironmentMixin._consume_requeue_grants(
            stub,
            operational,
            intake_ids if intake_ids is not None else frozenset({STABLE_ID}),
        )
        events = list(operational.records)
    return result, events


def _hold_events(events: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return the durable per-model grant review-hold disclosures.

    Parameters
    ----------
    events:
        Persisted operational events.

    Returns
    -------
    list[Mapping[str, Any]]
        Campaign-health events disclosing an unbindable outstanding grant.
    """

    return [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.CAMPAIGN_HEALTH.value
        and event.get("details", {}).get("disposition") == "requeue-grant-review-hold"
    ]


# ---------------------------------------------------------------------------
# T1-13a -- a void must neutralize an out-of-intake grant


@pytest.mark.smoke
def test_void_neutralizes_an_out_of_intake_grant(tmp_path: Path) -> None:
    """A voided foreign grant validates as inert history instead of raising.

    The pre-fix validator raised ``requeue grant values are invalid`` on the
    foreign grant row before the void row was ever scanned, in EITHER ledger
    order, so the campaign was bricked with no operator lever.
    """

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID)])
    foreign = _append_grant(paths, "m_never_in_any_intake")
    _append_void(paths, str(foreign["grant_id"]), reason="issued outside the intake")

    ledger = _validated_requeue_grants(
        canonical_requeue_grants_path(paths.ledgers.models),
        frozenset({STABLE_ID}),
    )
    assert [str(void["void_of"]) for void in ledger.voids] == [str(foreign["grant_id"])]

    result, events = _consume(paths)
    assert result == {}
    assert _hold_events(events) == []


@pytest.mark.smoke
def test_voided_foreign_grant_never_blocks_other_models(tmp_path: Path) -> None:
    """Another model's good grant consumes past voided foreign history."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID)])
    foreign = _append_grant(paths, "m_never_in_any_intake")
    _append_void(paths, str(foreign["grant_id"]), reason="issued outside the intake")
    good = _append_grant(paths, STABLE_ID)

    result, _events = _consume(paths)
    assert result[STABLE_ID]["active"] is True
    assert result[STABLE_ID]["grant_ids"] == [str(good["grant_id"])]


@pytest.mark.smoke
def test_unvoided_out_of_intake_grant_still_refuses(tmp_path: Path) -> None:
    """The membership tripwire stands: an un-voided foreign grant hard-raises."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID)])
    _append_grant(paths, "m_never_in_any_intake")

    with pytest.raises(DriverIntegrationError, match="requeue grant values are invalid"):
        _consume(paths)


# ---------------------------------------------------------------------------
# T1-13b -- issuance is idempotent: no double grant from a retried command


@pytest.mark.smoke
def test_retried_issuance_refuses_an_outstanding_logical_twin(tmp_path: Path) -> None:
    """Re-running the exact issuance command must not mint a second live grant."""

    paths = _paths(tmp_path)
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    first = _append_grant(paths, STABLE_ID, reason="operator approved retry")

    with pytest.raises(ValueError, match=str(first["grant_id"])):
        build_canonical_requeue_grant(
            canonical,
            stable_id=STABLE_ID,
            stage="accuracy-gate",
            reason="operator approved retry",
            attempts=1,
            granted_by="operator",
        )


@pytest.mark.smoke
def test_reissue_after_void_mints_a_new_grant(tmp_path: Path) -> None:
    """Void-before-reissue stays the sanctioned second-grant path."""

    paths = _paths(tmp_path)
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    first = _append_grant(paths, STABLE_ID, reason="operator approved retry")
    _append_void(paths, str(first["grant_id"]), reason="superseded by re-issue")

    second = build_canonical_requeue_grant(
        canonical,
        stable_id=STABLE_ID,
        stage="accuracy-gate",
        reason="operator approved retry",
        attempts=1,
        granted_by="operator",
    )
    assert second["grant_id"] != first["grant_id"]


@pytest.mark.smoke
def test_reissue_after_consumption_mints_a_new_generation(tmp_path: Path) -> None:
    """A consumed twin is spent history, not outstanding: re-issue is legal."""

    paths = _paths(tmp_path)
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    first = _append_grant(paths, STABLE_ID, reason="operator approved retry")
    events_path = canonical.parent / "events.jsonl"
    consumption = {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
        "details": {"grant_id": str(first["grant_id"]), "stable_id": STABLE_ID},
    }
    with events_path.open("wb") as handle:
        handle.write(canonical_json_bytes(consumption) + b"\n")

    second = build_canonical_requeue_grant(
        canonical,
        stable_id=STABLE_ID,
        stage="accuracy-gate",
        reason="operator approved retry",
        attempts=1,
        granted_by="operator",
    )
    assert second["grant_id"] != first["grant_id"]


# ---------------------------------------------------------------------------
# T1-13c -- an unbindable outstanding grant holds the model, not the campaign


@pytest.mark.smoke
def test_stage_mismatch_holds_the_model_not_the_campaign(tmp_path: Path) -> None:
    """One model's mismatched grant discloses and skips; the sibling consumes."""

    paths = _paths(tmp_path)
    _write_models(
        paths,
        [
            _terminal_record(STABLE_ID),
            _terminal_record(OTHER_ID, stage="runner", revision="sha256:" + "b" * 64),
        ],
    )
    mismatched = _append_grant(paths, STABLE_ID, stage="forward")
    good = _append_grant(paths, OTHER_ID, stage="runner")

    result, events = _consume(paths, intake_ids=frozenset({STABLE_ID, OTHER_ID}))

    assert STABLE_ID not in result
    assert result[OTHER_ID]["active"] is True
    assert result[OTHER_ID]["grant_ids"] == [str(good["grant_id"])]
    holds = _hold_events(events)
    assert len(holds) == 1
    details = holds[0]["details"]
    assert details["stable_id"] == STABLE_ID
    assert details["grant_id"] == str(mismatched["grant_id"])
    assert details["hold_reason"] == "stage-mismatch"


@pytest.mark.smoke
def test_unreviewed_terminal_grant_holds_not_aborts(tmp_path: Path) -> None:
    """A grant against an unreviewed terminal discloses instead of unwinding."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID, review_required=False)])
    grant = _append_grant(paths, STABLE_ID)

    result, events = _consume(paths)

    assert STABLE_ID not in result
    holds = _hold_events(events)
    assert len(holds) == 1
    assert holds[0]["details"]["hold_reason"] == "unreviewed-terminal"
    assert holds[0]["details"]["grant_id"] == str(grant["grant_id"])


@pytest.mark.smoke
def test_held_grant_stays_outstanding_and_voidable(tmp_path: Path) -> None:
    """The held grant is never consumed; the operator void remains the exit."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID)])
    mismatched = _append_grant(paths, STABLE_ID, stage="forward")

    first_result, first_events = _consume(paths)
    assert first_result == {}
    consumptions = [
        event
        for event in first_events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    assert consumptions == []

    _append_void(paths, str(mismatched["grant_id"]), reason="wrong stage; re-issuing")
    second_result, second_events = _consume(paths)
    assert second_result == {}
    # The runtime operational ledger persists across runs: the first run's
    # disclosure remains history, and the voided grant discloses nothing new.
    assert _hold_events(second_events) == _hold_events(first_events)


# ---------------------------------------------------------------------------
# T2-05 -- the deferred-terminal stage expresses the null-stage review-held class


@pytest.mark.smoke
def test_deferred_terminal_grant_expresses_and_consumes(tmp_path: Path) -> None:
    """The m4256 class: a review-held null-stage deferral is grantable, typed.

    Before the fix the closed failure-stage vocabulary had no member matching
    ``status.stage=null``, so issuance itself refused and the model was frozen
    behind a void-only lever.
    """

    paths = _paths(tmp_path)
    _write_models(
        paths,
        [
            _terminal_record(
                STABLE_ID,
                kind="deferred",
                code="deferred:needs-cuda",
                stage=None,
            )
        ],
    )
    grant = _append_grant(
        paths,
        STABLE_ID,
        stage=DEFERRED_TERMINAL_GRANT_STAGE,
        reason="operator re-adjudicates the platform deferral",
    )

    result, events = _consume(paths)

    assert result[STABLE_ID]["active"] is True
    assert result[STABLE_ID]["grant_ids"] == [str(grant["grant_id"])]
    assert _hold_events(events) == []


@pytest.mark.smoke
def test_deferred_terminal_stage_never_matches_a_staged_head(tmp_path: Path) -> None:
    """Narrowness: the sentinel binds only ``kind=deferred`` + ``stage=null``."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID)])
    _append_grant(paths, STABLE_ID, stage=DEFERRED_TERMINAL_GRANT_STAGE)

    result, events = _consume(paths)

    assert STABLE_ID not in result
    holds = _hold_events(events)
    assert len(holds) == 1
    assert holds[0]["details"]["hold_reason"] == "stage-mismatch"


@pytest.mark.smoke
def test_regular_stage_grant_never_matches_a_null_stage_head(tmp_path: Path) -> None:
    """Narrowness: an ordinary failure stage still never binds a null stage."""

    paths = _paths(tmp_path)
    _write_models(
        paths,
        [
            _terminal_record(
                STABLE_ID,
                kind="deferred",
                code="deferred:needs-cuda",
                stage=None,
            )
        ],
    )
    _append_grant(paths, STABLE_ID, stage="accuracy-gate")

    result, events = _consume(paths)

    assert STABLE_ID not in result
    holds = _hold_events(events)
    assert len(holds) == 1
    assert holds[0]["details"]["hold_reason"] == "stage-mismatch"


# ---------------------------------------------------------------------------
# T2-12 (grants half) -- newly minted rows carry a schema identity


@pytest.mark.smoke
def test_new_grant_and_void_rows_carry_schema_identity(tmp_path: Path) -> None:
    """Fresh grant and void rows self-describe; legacy rows stay readable."""

    paths = _paths(tmp_path)
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    grant = _append_grant(paths, STABLE_ID)
    void = _append_void(paths, str(grant["grant_id"]), reason="operator withdrew")

    # W31: freshly minted grants ride the v2 rung; voids stay on their v1.
    assert grant["schema_version"] == "menagerie.crawler.requeue-grant.v2"
    assert void["schema_version"] == "menagerie.crawler.requeue-grant-void.v1"

    legacy = {
        "grant_id": "sha256:" + "c" * 64,
        "stable_id": STABLE_ID,
        "stage": "accuracy-gate",
        "reason": "handwritten legacy row",
        "attempts": 1,
        "created_at": NOW,
    }
    rows = [json.loads(line) for line in canonical.read_bytes().splitlines()]
    assert all("schema_version" in row for row in rows)
    with canonical.open("ab") as handle:
        handle.write(canonical_json_bytes(legacy) + b"\n")
    # The legacy row has no canonical identity here (wrong hash), so validation
    # must refuse it for its HASH, never for lacking a schema identity.
    with pytest.raises(DriverIntegrationError, match="hash mismatch"):
        _validated_requeue_grants(canonical, frozenset({STABLE_ID}))


@pytest.mark.smoke
def test_wrong_grant_schema_identity_refuses(tmp_path: Path) -> None:
    """A row claiming a foreign schema identity is refused typed."""

    paths = _paths(tmp_path)
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    grant = dict(_append_grant(paths, STABLE_ID))
    forged = dict(grant)
    forged["schema_version"] = "menagerie.crawler.requeue-grant.v999"
    with canonical.open("wb") as handle:
        handle.write(canonical_json_bytes(forged) + b"\n")

    with pytest.raises(DriverIntegrationError):
        _validated_requeue_grants(canonical, frozenset({STABLE_ID}))
