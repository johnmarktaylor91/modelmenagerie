"""Subset-rung requeue grants: prior-lineage grants are DORMANT, never invalid.

Regression suite for the wave-2 mini-rung relaunch instant exit (LIVE blocker
#2): ``_validated_requeue_grants`` invalidated the WHOLE canonical grant ledger
("requeue grant values are invalid") because the pilot ledger legitimately holds
step-100 pre-grants and voids for models outside the 10-member REGISTERED SUBSET
intake. Same class as the RC6 reducer-membership fix: a grant whose stable ID
belongs to a REGISTERED PRIOR same-campaign snapshot (the driver-read lineage
union K2/RC6 already plumb) is DORMANT -- it validates, stays on the canonical
ledger, is never consumed this leg, and never binds work. A stable ID in NEITHER
the intake NOR the lineage keeps the byte-identical hard refusal, and an
in-intake grant keeps consuming exactly as before (exclusion happens via the
intake, never via grant validation).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    append_canonical_requeue_void,
    build_canonical_requeue_grant,
    build_canonical_requeue_void,
    canonical_requeue_grants_path,
)
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
)
from menagerie.crawler.driver_admission import (
    AdmissionEnvironmentMixin,
    _validated_requeue_grants,
)
from menagerie.crawler.driver_contracts import DriverIntegrationError, DriverPaths
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import default_ledger_paths

NOW = "2026-08-11T12:00:00Z"
IN_INTAKE_ID = "m_subset_member"
PRIOR_MEMBER_ID = "m_step100_prior"
FOREIGN_ID = "m_forged_foreign"


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
    revision_seed: str = "a",
    record_seq: int = 1,
) -> Mapping[str, Any]:
    """Return one minimal persisted reviewed terminal model revision.

    Parameters
    ----------
    stable_id:
        Model identity.
    stage, review_required:
        Recorded failure stage and human-review flag.
    revision_seed, record_seq:
        Distinct persisted revision identity seed and sequence.

    Returns
    -------
    Mapping[str, Any]
        Ledger-shaped model record.
    """

    return {
        "stable_id": stable_id,
        "record_revision": "sha256:" + revision_seed * 64,
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
    paths: DriverPaths,
    *,
    intake_ids: frozenset[str],
    prior_member_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], list[Mapping[str, Any]]]:
    """Run the real consumer against the harness ledgers.

    Parameters
    ----------
    paths:
        Harness driver paths.
    intake_ids:
        Trusted ACTIVE intake membership.
    prior_member_ids:
        Registered prior same-campaign snapshot member union.

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
            intake_ids,
            prior_member_ids=prior_member_ids,
        )
        events = list(operational.records)
    return result, events


def _canonical_rows(paths: DriverPaths) -> list[Mapping[str, Any]]:
    """Return raw canonical grant-ledger rows in append order.

    Parameters
    ----------
    paths:
        Harness driver paths.

    Returns
    -------
    list[Mapping[str, Any]]
        Persisted canonical grant/void rows.
    """

    return scan_jsonl(canonical_requeue_grants_path(paths.ledgers.models), validate=False)


@pytest.mark.smoke
def test_dormant_prior_member_grant_is_retained_and_not_consumed(tmp_path: Path) -> None:
    """A registered prior-lineage grant validates, stays durable, and is not consumed.

    The exact mini-rung shape: the canonical ledger carries a step-100
    pre-grant for a model outside the 10-member subset intake. The run must
    proceed (no ledger-wide invalidation), consume nothing for the prior
    member, bind no work to it, and leave the grant byte-for-byte on the
    canonical ledger for the full-campaign context.
    """

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(PRIOR_MEMBER_ID, revision_seed="b")])
    grant = _append_grant(paths, PRIOR_MEMBER_ID)

    result, events = _consume(
        paths,
        intake_ids=frozenset({IN_INTAKE_ID}),
        prior_member_ids=frozenset({PRIOR_MEMBER_ID}),
    )

    assert result == {}
    assert not [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    persisted = _canonical_rows(paths)
    assert [row["grant_id"] for row in persisted] == [grant["grant_id"]]


@pytest.mark.smoke
def test_foreign_grant_still_hard_refuses_with_the_exact_message(tmp_path: Path) -> None:
    """A stable ID in NEITHER the intake NOR the lineage keeps the hard refusal."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(FOREIGN_ID, revision_seed="c")])
    _append_grant(paths, FOREIGN_ID)

    with pytest.raises(DriverIntegrationError, match="^requeue grant values are invalid$"):
        _consume(
            paths,
            intake_ids=frozenset({IN_INTAKE_ID}),
            prior_member_ids=frozenset({PRIOR_MEMBER_ID}),
        )


@pytest.mark.smoke
def test_in_intake_grant_consumes_alongside_dormant_history(tmp_path: Path) -> None:
    """An active-intake grant consumes normally while prior-lineage grants sleep.

    The m2336-class discipline: a model is excluded from a leg via the INTAKE,
    never via grant validation. The subset member's grant mints its generation
    and binds work exactly as on a full rung, with the dormant grant coexisting
    on the same canonical ledger.
    """

    paths = _paths(tmp_path)
    _write_models(
        paths,
        [
            _terminal_record(PRIOR_MEMBER_ID, revision_seed="b"),
            _terminal_record(IN_INTAKE_ID, revision_seed="d", record_seq=2),
        ],
    )
    dormant = _append_grant(paths, PRIOR_MEMBER_ID)
    active = _append_grant(paths, IN_INTAKE_ID)

    result, events = _consume(
        paths,
        intake_ids=frozenset({IN_INTAKE_ID}),
        prior_member_ids=frozenset({PRIOR_MEMBER_ID}),
    )

    assert set(result) == {IN_INTAKE_ID}
    assert result[IN_INTAKE_ID]["active"] is True
    assert result[IN_INTAKE_ID]["grant_ids"] == [active["grant_id"]]
    consumed = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    assert [event["details"]["grant_id"] for event in consumed] == [active["grant_id"]]
    assert {row["grant_id"] for row in _canonical_rows(paths)} == {
        dormant["grant_id"],
        active["grant_id"],
    }


@pytest.mark.smoke
def test_void_of_a_dormant_grant_keeps_existing_semantics(tmp_path: Path) -> None:
    """An operator void of a prior-lineage grant still resolves and validates.

    Void rows key on ``by_id`` inside the validator, so dormant grants must be
    RETAINED (not filtered) for the pilot ledger's step-100 voids to keep
    resolving. Nothing is consumed and the pair stays durable.
    """

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(PRIOR_MEMBER_ID, revision_seed="b")])
    grant = _append_grant(paths, PRIOR_MEMBER_ID)
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    void = build_canonical_requeue_void(
        canonical,
        void_of=str(grant["grant_id"]),
        reason="stale after step-100 park",
        voided_by="operator",
    )
    append_canonical_requeue_void(canonical, void)

    result, events = _consume(
        paths,
        intake_ids=frozenset({IN_INTAKE_ID}),
        prior_member_ids=frozenset({PRIOR_MEMBER_ID}),
    )

    assert result == {}
    assert not [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    view = _validated_requeue_grants(
        canonical,
        frozenset({IN_INTAKE_ID}),
        prior_member_ids=frozenset({PRIOR_MEMBER_ID}),
    )
    assert [row["grant_id"] for row in view.grants] == [grant["grant_id"]]
    assert [row["void_of"] for row in view.voids] == [grant["grant_id"]]


@pytest.mark.smoke
def test_validator_retains_dormant_grants_without_loosening_integrity(tmp_path: Path) -> None:
    """The validator returns dormant grants yet still refuses a forged dormant row.

    The lineage carve-out licenses MEMBERSHIP only: every other integrity check
    (identity hash, field contract, duplicate conflicts) applies to a dormant
    grant unchanged, so a tampered prior-lineage row can never ride the
    carve-out onto the ledger view.
    """

    paths = _paths(tmp_path)
    grant = _append_grant(paths, PRIOR_MEMBER_ID)
    canonical = canonical_requeue_grants_path(paths.ledgers.models)

    view = _validated_requeue_grants(
        canonical,
        frozenset({IN_INTAKE_ID}),
        prior_member_ids=frozenset({PRIOR_MEMBER_ID}),
    )
    assert [row["grant_id"] for row in view.grants] == [grant["grant_id"]]

    forged = dict(grant)
    forged["reason"] = "tampered reason keeps the stale grant_id"
    with canonical.open("ab") as handle:
        handle.write(json.dumps(forged, sort_keys=True).encode("utf-8") + b"\n")
    with pytest.raises(DriverIntegrationError, match="requeue grant hash mismatch"):
        _validated_requeue_grants(
            canonical,
            frozenset({IN_INTAKE_ID}),
            prior_member_ids=frozenset({PRIOR_MEMBER_ID}),
        )


@pytest.mark.smoke
def test_empty_lineage_keeps_the_exact_pre_carveout_world(tmp_path: Path) -> None:
    """With no registered lineage, any out-of-intake grant refuses exactly as before."""

    paths = _paths(tmp_path)
    _write_models(paths, [_terminal_record(PRIOR_MEMBER_ID, revision_seed="b")])
    _append_grant(paths, PRIOR_MEMBER_ID)

    with pytest.raises(DriverIntegrationError, match="^requeue grant values are invalid$"):
        _consume(paths, intake_ids=frozenset({IN_INTAKE_ID}))
