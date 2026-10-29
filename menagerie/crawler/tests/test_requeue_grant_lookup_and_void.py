"""Requeue-grant consumption keys on the ledger's terminal revision, with voids.

Regression suite for the rung-11 leg-5 campaign halt: three operator grants were
appended for review-held terminals, and ``_consume_requeue_grants`` crashed with
``requeue grant for m4256 has no reviewed terminal record`` because it looked the
model up in the DEPENDENCY-CURRENT projection. m4256's reviewed terminal was
dependency-stale under the post-upgrade tree (exactly the situation a grant
exists to resolve), so the projection had dropped it while the raw models ledger
plainly carried the reviewed terminal. Consumption must key on the persisted
supersession-lineage head of the models ledger, never on dependency currency,
while every fail-closed raise (absent record, unreviewed terminal, stage
mismatch) stands. Operator voids are the append-only escape for a stale grant.
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
    requeue_void_identity,
)
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    REQUEUE_GRANT_VOID_SCHEMA_VERSION,
    OperationalEventKind,
)
from menagerie.crawler.driver_admission import (
    AdmissionEnvironmentMixin,
    _validated_requeue_grants,
)
from menagerie.crawler.driver_contracts import DriverIntegrationError, DriverPaths
from menagerie.crawler.grant_identity import expected_grant_id
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import default_ledger_paths

NOW = "2026-08-07T12:00:00Z"
STABLE_ID = "m_reviewed"


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
    stage: str = "accuracy-gate",
    review_required: bool = True,
    revision: str = "sha256:" + "a" * 64,
    record_seq: int = 1,
) -> Mapping[str, Any]:
    """Return one minimal persisted terminal model revision.

    Parameters
    ----------
    stable_id:
        Model identity.
    stage, review_required:
        Recorded failure stage and human-review flag.
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
            "kind": "failed",
            "code": f"failed:{stage}",
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


@pytest.mark.smoke
def test_grant_consumes_against_ledger_terminal_not_dependency_projection(
    tmp_path: Path,
) -> None:
    """The m4256 regression: a reviewed ledger terminal consumes its grant.

    The consumer receives no dependency-current projection at all, so a terminal
    that is dependency-stale under the current tree (absent from the projection)
    still consumes from its persisted reviewed revision.
    """

    paths = _paths(tmp_path)
    record = _terminal_record(STABLE_ID)
    _write_models(paths, [record])
    grant = _append_grant(paths, STABLE_ID)

    result, events = _consume(paths)

    expected_work_id = stable_hash(
        {
            "stable_id": STABLE_ID,
            "grant_id": grant["grant_id"],
            "parent_revision": record["record_revision"],
            "generation": 1,
        }
    )
    assert result == {
        STABLE_ID: {
            "grant_ids": [grant["grant_id"]],
            "work_id": expected_work_id,
            "active": True,
            "target_intent": None,
            "attempts": 1,
        }
    }
    consumed = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    assert len(consumed) == 1
    details = consumed[0]["details"]
    assert details["source_record_revision"] == record["record_revision"]
    assert details["new_work_id"] == expected_work_id
    assert details["new_work_generation"] == 1


@pytest.mark.smoke
def test_grant_without_any_terminal_record_holds_the_model(tmp_path: Path) -> None:
    """A grant for a model with no ledger history stays fail-closed, model-local.

    Pre-T1-13 this raised campaign-global at admission; the grant is
    model-local by construction, so it now discloses a durable review-hold,
    binds no work, and stays outstanding for the operator void.
    """

    paths = _paths(tmp_path)
    _write_models(paths, [])
    grant = _append_grant(paths, STABLE_ID)
    result, events = _consume(paths)
    assert result == {}
    holds = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "requeue-grant-review-hold"
    ]
    assert len(holds) == 1
    assert holds[0]["details"]["hold_reason"] == "unreviewed-terminal"
    assert holds[0]["details"]["grant_id"] == grant["grant_id"]


@pytest.mark.smoke
def test_grant_for_unreviewed_terminal_holds_the_model(tmp_path: Path) -> None:
    """A grant whose terminal was never review-flagged holds, never consumes."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID, review_required=False)])
    _append_grant(paths, STABLE_ID)
    result, events = _consume(paths)
    assert result == {}
    assert not any(
        event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
        for event in events
    )
    holds = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "requeue-grant-review-hold"
    ]
    assert [hold["details"]["hold_reason"] for hold in holds] == ["unreviewed-terminal"]


@pytest.mark.smoke
def test_grant_stage_mismatch_holds_the_model(tmp_path: Path) -> None:
    """A grant staged against a superseded stage holds, never consumes."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID, stage="accuracy-gate")])
    _append_grant(paths, STABLE_ID, stage="runner")
    result, events = _consume(paths)
    assert result == {}
    assert not any(
        event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
        for event in events
    )
    holds = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "requeue-grant-review-hold"
    ]
    assert [hold["details"]["hold_reason"] for hold in holds] == ["stage-mismatch"]


def test_consumption_binds_latest_terminal_revision(tmp_path: Path) -> None:
    """A superseding reviewed revision is the one the grant consumes against."""

    paths = _paths(tmp_path)
    first = dict(_terminal_record(STABLE_ID, stage="runner", review_required=False))
    second = dict(
        _terminal_record(
            STABLE_ID,
            stage="accuracy-gate",
            revision="sha256:" + "b" * 64,
            record_seq=2,
        )
    )
    second["parent_revision"] = first["record_revision"]
    second["status"] = {**second["status"], "supersedes_revision": first["record_revision"]}
    _write_models(paths, [first, second])
    grant = _append_grant(paths, STABLE_ID, stage="accuracy-gate")

    result, events = _consume(paths)

    assert result[STABLE_ID]["active"] is True
    details = events[-1]["details"]
    assert details["grant_id"] == grant["grant_id"]
    assert details["source_record_revision"] == second["record_revision"]


@pytest.mark.smoke
def test_voided_grant_skips_and_terminal_stands(tmp_path: Path) -> None:
    """The m4257 shape: a voided stale grant neither consumes nor crashes.

    Without the void this grant would refuse the run at the stage-mismatch
    raise. With the void it is skipped entirely: no consumption event, no
    result binding, and the current terminal stands untouched.
    """

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID, stage="accuracy-gate")])
    grant = _append_grant(paths, STABLE_ID, stage="runner")
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    void = build_canonical_requeue_void(
        canonical,
        void_of=str(grant["grant_id"]),
        reason="grant stage no longer matches the superseded terminal",
        voided_by="operator",
    )
    append_canonical_requeue_void(canonical, void)

    result, events = _consume(paths)

    assert result == {}
    assert [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ] == []


def test_void_after_consumption_keeps_recorded_history(tmp_path: Path) -> None:
    """Voiding an already-consumed grant never rewrites its consumption.

    C3-B narrowed the PRODUCER: ``build_canonical_requeue_void`` still refuses
    to mint this void. The READ side changed with the 2026-08-16 red-team
    round-2 break 5: a VOIDED grant now discharges every generation it minted
    (``derive_generation_discharge``), so the withdrawn authority stops
    re-admitting its member -- the operator lever finally does what it says.
    What this cell keeps pinning is the HISTORY half: the persisted
    consumption event is never rewritten, deleted, or re-consumed. The
    binding surfaces as settled spent history (``active=False`` with the
    consumed generation's ``work_id``), exactly like a recorded spent grant.
    """

    paths = _paths(tmp_path)
    record = _terminal_record(STABLE_ID)
    _write_models(paths, [record])
    grant = _append_grant(paths, STABLE_ID)
    first_result, first_events = _consume(paths)
    assert first_result[STABLE_ID]["active"] is True

    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    with pytest.raises(ValueError, match="already consumed"):
        build_canonical_requeue_void(
            canonical,
            void_of=str(grant["grant_id"]),
            reason="operator retraction after consumption",
            voided_by="operator",
        )
    void = {
        "void_id": requeue_void_identity(
            void_of=str(grant["grant_id"]),
            reason="operator retraction after consumption",
            voided_by="operator",
        ),
        "schema_version": REQUEUE_GRANT_VOID_SCHEMA_VERSION,
        "void_of": str(grant["grant_id"]),
        "reason": "operator retraction after consumption",
        "voided_by": "operator",
        "created_at": NOW,
    }
    append_canonical_requeue_void(canonical, void)

    second_result, second_events = _consume(paths)
    assert second_result == {
        STABLE_ID: {
            "grant_ids": first_result[STABLE_ID]["grant_ids"],
            "work_id": first_result[STABLE_ID]["work_id"],
            "active": False,
        }
    }
    consumed = [
        event
        for event in second_events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    assert len(consumed) == 1


def test_void_referencing_unknown_grant_refuses(tmp_path: Path) -> None:
    """A hand-written void for an absent grant is a typed integrity failure."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID)])
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    unknown = "sha256:" + "f" * 64
    row = {
        "void_id": requeue_void_identity(void_of=unknown, reason="typo", voided_by="operator"),
        "void_of": unknown,
        "reason": "typo",
        "voided_by": "operator",
        "created_at": NOW,
    }
    with canonical.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
    with pytest.raises(DriverIntegrationError, match="unknown grant"):
        _consume(paths)


def test_forged_void_identity_refuses(tmp_path: Path) -> None:
    """A void whose identity hash does not match its facts is refused."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID)])
    grant = _append_grant(paths, STABLE_ID)
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    row = {
        "void_id": "sha256:" + "0" * 64,
        "void_of": str(grant["grant_id"]),
        "reason": "forged",
        "voided_by": "operator",
        "created_at": NOW,
    }
    with canonical.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
    with pytest.raises(DriverIntegrationError, match="requeue void hash mismatch"):
        _consume(paths)


def test_runtime_ledger_void_refuses(tmp_path: Path) -> None:
    """Voids are canonical-only: a runtime-ledger void is a typed refusal."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID)])
    grant = _append_grant(paths, STABLE_ID)
    paths.requeue_grants.parent.mkdir(parents=True, exist_ok=True)
    runtime_rows = [
        dict(grant),
        {
            "void_id": requeue_void_identity(
                void_of=str(grant["grant_id"]),
                reason="misplaced",
                voided_by="operator",
            ),
            "void_of": str(grant["grant_id"]),
            "reason": "misplaced",
            "voided_by": "operator",
            "created_at": NOW,
        },
    ]
    with paths.requeue_grants.open("wb") as handle:
        for row in runtime_rows:
            handle.write(canonical_json_bytes(row) + b"\n")
    with pytest.raises(DriverIntegrationError, match="canonical grant ledger"):
        _consume(paths)


def test_validator_returns_grants_and_voids_in_append_order(tmp_path: Path) -> None:
    """The ledger view separates grants from voids without dropping either."""

    ledger = tmp_path / "requeue-grants.jsonl"
    grant = build_canonical_requeue_grant(
        ledger,
        stable_id=STABLE_ID,
        stage="accuracy-gate",
        reason="first",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(ledger, grant)
    void = build_canonical_requeue_void(
        ledger,
        void_of=str(grant["grant_id"]),
        reason="stale",
        voided_by="operator",
    )
    append_canonical_requeue_void(ledger, void)

    view = _validated_requeue_grants(ledger, frozenset({STABLE_ID}))
    assert [row["grant_id"] for row in view.grants] == [grant["grant_id"]]
    assert [row["void_of"] for row in view.voids] == [grant["grant_id"]]
    persisted = [json.loads(line) for line in ledger.read_bytes().splitlines()]
    assert persisted[0]["grant_id"] == grant["grant_id"]
    assert persisted[1]["void_id"] == void["void_id"]


@pytest.mark.smoke
def test_award_naming_grant_binds_and_consumes_against_a_runs_head(tmp_path: Path) -> None:
    """AW-1's award-naming lane binds: naming the runs revision IS the authority.

    THE selection policy (``work_selection_predicate`` clause 3) admits a
    ``runs`` head whose live grant names its exact blocked award revision;
    the grant-state derivation must agree, or every operator award-naming
    grant crashes the leg at ``_apply_selection_grants`` with
    "selection admitted ... on an unbindable grant". A runs head is not a
    review-held terminal, so the pre-AW reviewed-terminal bindability rule
    alone can never admit it.
    """

    paths = _paths(tmp_path)
    revision = "sha256:" + "a" * 64
    record = {
        "stable_id": STABLE_ID,
        "record_revision": revision,
        "parent_revision": None,
        "record_seq": 1,
        "status": {
            "kind": "runs",
            "code": "runs",
            "stage": None,
            "human_review": {
                "queue": None,
                "reason": None,
                "requested_at": None,
                "required": False,
            },
        },
        "budget": {"explicit_grants": []},
    }
    _write_models(paths, [record])
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    grant = build_canonical_requeue_grant(
        canonical,
        stable_id=STABLE_ID,
        stage="deferred-terminal",
        reason="operator: supersession of the named award approved",
        attempts=1,
        granted_by="operator",
        blocked_award_revision=revision,
    )
    append_canonical_requeue_grant(canonical, grant)

    result, events = _consume(paths)

    assert STABLE_ID in result
    assert result[STABLE_ID]["grant_ids"] == [grant["grant_id"]]
    assert result[STABLE_ID]["active"] is True
    holds = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "requeue-grant-review-hold"
    ]
    assert holds == []
    consumed = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    assert len(consumed) == 1
    assert consumed[0]["details"]["source_record_revision"] == revision

    # A GENERIC grant on a runs head stays fail-closed: no binding, no
    # consumption (WS-1 fixture row (h) -- a grant that does not name the
    # award reopens nothing).
    other_paths = _paths(tmp_path / "generic")
    _write_models(other_paths, [record])
    generic_canonical = canonical_requeue_grants_path(other_paths.ledgers.models)
    generic_canonical.parent.mkdir(parents=True, exist_ok=True)
    generic = build_canonical_requeue_grant(
        generic_canonical,
        stable_id=STABLE_ID,
        stage="deferred-terminal",
        reason="operator: generic re-drive",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(generic_canonical, generic)
    generic_result, generic_events = _consume(other_paths)
    assert generic_result == {}
    assert [
        event["details"]["hold_reason"]
        for event in generic_events
        if event.get("details", {}).get("disposition") == "requeue-grant-review-hold"
    ] == ["unreviewed-terminal"]


def test_validator_admits_award_naming_and_capability_grants(tmp_path: Path) -> None:
    """The canonical AW-1/WS-2 grant fields pass the admission field contract.

    ``blocked_award_revision`` is AW-1's award-naming seam -- the single
    authority that releases the award-custody tripwire -- and ``capability``
    is WS-2's typed host-capability re-open. Both are minted by the ONE
    canonical builder and folded into the grant identity, so the admission
    validator must admit them; refusing them bricks the only sanctioned
    re-drive lever for a custody-held runs award.
    """

    ledger = tmp_path / "requeue-grants.jsonl"
    award_naming = build_canonical_requeue_grant(
        ledger,
        stable_id=STABLE_ID,
        stage="accuracy-gate",
        reason="operator: supersession of the named award approved",
        attempts=1,
        granted_by="operator",
        blocked_award_revision="sha256:" + "a" * 64,
    )
    append_canonical_requeue_grant(ledger, award_naming)
    capability = build_canonical_requeue_grant(
        ledger,
        stable_id="m_other",
        stage="accuracy-gate",
        reason="operator: host capability routing re-opened",
        attempts=1,
        granted_by="operator",
        capability="host-capability-routing",
    )
    append_canonical_requeue_grant(ledger, capability)

    view = _validated_requeue_grants(ledger, frozenset({STABLE_ID, "m_other"}))
    assert [row["grant_id"] for row in view.grants] == [
        award_naming["grant_id"],
        capability["grant_id"],
    ]
    assert view.grants[0]["blocked_award_revision"] == "sha256:" + "a" * 64
    assert view.grants[1]["capability"] == "host-capability-routing"

    # The value contract stays fail-closed: a non-string award revision is an
    # integrity failure even under a self-consistent identity hash, not
    # tolerated free-form data.
    forged = dict(award_naming, blocked_award_revision=7)
    forged["grant_id"] = expected_grant_id(forged)
    with ledger.open("ab") as handle:
        handle.write(canonical_json_bytes(forged) + b"\n")
    with pytest.raises(DriverIntegrationError, match="requeue grant values are invalid"):
        _validated_requeue_grants(ledger, frozenset({STABLE_ID, "m_other"}))


def test_void_append_is_idempotent_and_conflicts_refuse(tmp_path: Path) -> None:
    """Identical voids dedupe; a fact-conflicting duplicate identity refuses."""

    ledger = tmp_path / "requeue-grants.jsonl"
    grant = build_canonical_requeue_grant(
        ledger,
        stable_id=STABLE_ID,
        stage="accuracy-gate",
        reason="first",
        attempts=1,
        granted_by="operator",
    )
    append_canonical_requeue_grant(ledger, grant)
    void = build_canonical_requeue_void(
        ledger,
        void_of=str(grant["grant_id"]),
        reason="stale",
        voided_by="operator",
    )
    assert append_canonical_requeue_void(ledger, void) is True
    later = dict(void)
    later["created_at"] = "2026-08-07T13:00:00Z"
    assert append_canonical_requeue_void(ledger, later) is False
    assert len(scan_jsonl(ledger, validate=False)) == 2

    forged = dict(void)
    forged["void_of"] = "sha256:" + "e" * 64
    from menagerie.crawler.checkpoint import CheckpointValidationError

    with pytest.raises(CheckpointValidationError, match="conflicting"):
        append_canonical_requeue_void(ledger, forged)


def test_a_consumed_grant_is_never_consumed_again(tmp_path: Path) -> None:
    """A4: one grant funds exactly one generation, across driver legs.

    The consumption event is durable on the canonical operational ledger, so a
    later leg (a fresh consumer session over the same canonical state) must
    read the existing binding back -- identical work identity, generation
    still 1, and NO second ``REQUEUE_GRANT_CONSUMED`` event. Once the
    re-adjudication terminal records the grant in its budget, the grant is
    spent history: a further leg neither re-activates nor re-consumes it.

    Parameters
    ----------
    tmp_path:
        Isolated harness root.
    """

    paths = _paths(tmp_path)
    record = _terminal_record(STABLE_ID)
    _write_models(paths, [record])
    grant = _append_grant(paths, STABLE_ID)

    def _consumed(events: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """Filter the durable consumption events from one leg's view."""

        return [
            event
            for event in events
            if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
        ]

    first_result, first_events = _consume(paths)
    assert len(_consumed(first_events)) == 1
    binding = first_result[STABLE_ID]
    assert binding["active"] is True
    assert _consumed(first_events)[0]["details"]["new_work_generation"] == 1

    # Leg 2: a fresh consumer session over the same canonical ledgers reads
    # the binding back instead of funding a second generation.
    second_result, second_events = _consume(paths)
    assert second_result == first_result
    assert len(_consumed(second_events)) == 1
    assert _consumed(second_events)[0]["details"] == _consumed(first_events)[0]["details"]

    # Leg 3: the re-adjudication terminal recorded the grant in its budget,
    # so the grant is spent -- not active, and still not re-consumed.
    superseding = dict(_terminal_record(STABLE_ID, revision="sha256:" + "b" * 64, record_seq=2))
    superseding["parent_revision"] = record["record_revision"]
    superseding["status"] = {
        **superseding["status"],
        "supersedes_revision": record["record_revision"],
    }
    superseding["budget"] = {"explicit_grants": [grant["grant_id"]]}
    _write_models(paths, [record, superseding])
    third_result, third_events = _consume(paths)
    assert len(_consumed(third_events)) == 1
    spent = third_result[STABLE_ID]
    assert spent["active"] is False
    assert spent["grant_ids"] == [grant["grant_id"]]
    assert spent["work_id"] == binding["work_id"]


@pytest.mark.smoke
def test_requeue_tool_refuses_voiding_a_consumed_grant_and_says_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Voiding an already-consumed grant refuses typed and names the consumption.

    A consumed grant's generation stays bound whether or not a void lands
    (pinned by ``test_void_after_consumption_keeps_recorded_history``), so an
    operator issuing ``requeue --void`` to CANCEL a re-drive gets nothing --
    and used to get SILENCE. This is the CLI-level pin that the operator is no
    longer left in silence.

    SYN-B adjudication (see ``SYN_B_DOCKET.md``, C3 ruling 1): lab c3a appended
    the void row and warned on stderr that it cancels nothing; lab c3b refuses
    at the producer. Refusal wins -- an append-only row that reads as a
    revocation while provably revoking nothing is a disclosure hazard, and the
    tool's own ``--void`` help already promised "an existing unconsumed grant".
    No coverage is lost: this test keeps c3a's operator-disclosure assertion and
    additionally pins that NO misleading row is minted.

    A void of an UNCONSUMED grant still succeeds quietly: that void genuinely
    deactivates the grant.
    """

    from menagerie.crawler.tools.requeue import main as requeue_main

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(STABLE_ID)])
    grant = _append_grant(paths, STABLE_ID)
    _consume(paths)

    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    exit_code = requeue_main(
        [
            "--void",
            str(grant["grant_id"]),
            "--reason",
            "operator wants to cancel the re-drive",
            "--repo-root",
            str(tmp_path),
            "--ledger",
            str(canonical),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "already consumed" in captured.err
    # The refusal must not mint the misleading revocation row.
    voids = [row for row in scan_jsonl(canonical, validate=False) if "void_of" in row]
    assert voids == []

    # Control: voiding an UNCONSUMED grant is a real deactivation -- it lands.
    fresh = _append_grant(paths, STABLE_ID, stage="runner")
    exit_code = requeue_main(
        [
            "--void",
            str(fresh["grant_id"]),
            "--reason",
            "stale stage",
            "--repo-root",
            str(tmp_path),
            "--ledger",
            str(canonical),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "already consumed" not in captured.err


def test_build_void_refuses_absent_grant(tmp_path: Path) -> None:
    """The builder refuses to void a grant the ledger does not carry."""

    ledger = tmp_path / "requeue-grants.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(b"")
    with pytest.raises(ValueError, match="absent from the ledger"):
        build_canonical_requeue_void(
            ledger,
            void_of="sha256:" + "d" * 64,
            reason="nothing to void",
            voided_by="operator",
        )
