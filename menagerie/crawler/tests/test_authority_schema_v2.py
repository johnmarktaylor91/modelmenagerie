"""W31 grant/authority schema-v2 wave: era ladder, adjudication identity, aging.

The requeue-grant family climbs one monotone schema rung: a NEW grant row is
``menagerie.crawler.requeue-grant.v2`` and carries ``created_at`` (timezone-aware
UTC), a recomputable ``adjudication_key`` (the identity of the adjudicated
operator decision, excluding attribution and allowance size), and the MC-Q4
``effective_session_ceiling`` disclosure. Historical rows -- legacy handwritten
and v1 tool rows -- stay readable and valid forever; nothing backfills them.
Voids gain the idempotent double-void refusal (a SECOND, differently-attributed
void of one grant refuses naming the first; an identical re-void stays a no-op
success), and the append path takes the ledger flock so the B2-X-V7 grant-race
arm has exactly one winner.
"""

from __future__ import annotations

import json
import multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import pytest

from menagerie.crawler.checkpoint import (
    CheckpointValidationError,
    append_canonical_requeue_grant,
    append_canonical_requeue_void,
    build_canonical_requeue_grant,
    build_canonical_requeue_void,
    canonical_requeue_grants_path,
    requeue_void_identity,
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
from menagerie.crawler.grant_identity import legacy_grant_identity, tool_grant_identity
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.reducer import default_ledger_paths

NOW = "2026-08-24T12:00:00Z"
STABLE_ID = "m_schema_v2"


def _paths(tmp_path: Path) -> DriverPaths:
    """Return isolated driver paths below one temporary root."""

    return DriverPaths(
        runtime_root=tmp_path / ".crawl-local",
        intake_root=tmp_path / "intake",
        ledgers=default_ledger_paths(tmp_path / "records"),
    )


def _ledger(tmp_path: Path) -> Path:
    """Return the canonical requeue-grant ledger path for the harness root."""

    path = canonical_requeue_grants_path(_paths(tmp_path).ledgers.models)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _grant(
    ledger: Path,
    *,
    stable_id: str = STABLE_ID,
    stage: str = "accuracy-gate",
    reason: str = "operator approved re-adjudication",
    attempts: int = 1,
    granted_by: str = "operator",
) -> Mapping[str, Any]:
    """Build and durably append one tool grant."""

    grant = build_canonical_requeue_grant(
        ledger,
        stable_id=stable_id,
        stage=stage,
        reason=reason,
        attempts=attempts,
        granted_by=granted_by,
    )
    append_canonical_requeue_grant(ledger, grant)
    return grant


def _write_rows(path: Path, rows: list[Mapping[str, Any]]) -> None:
    """Hand-write raw ledger rows, bypassing every append-side check."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(dict(row)) + b"\n")


def _v1_tool_row(
    *,
    stable_id: str = STABLE_ID,
    stage: str = "accuracy-gate",
    reason: str = "historical v1 authorization",
    attempts: int = 1,
    granted_by: str = "operator",
    generation: int = 1,
) -> dict[str, Any]:
    """Return one historical v1 tool grant row (no created_at, no key)."""

    return {
        "grant_id": tool_grant_identity(
            generation=generation,
            stable_id=stable_id,
            stage=stage,
            reason=reason,
            attempts=attempts,
            granted_by=granted_by,
        ),
        "schema_version": "menagerie.crawler.requeue-grant.v1",
        "stable_id": stable_id,
        "stage": stage,
        "reason": reason,
        "attempts": attempts,
        "granted_by": granted_by,
        "new_work_generation": generation,
    }


def _legacy_row(
    *,
    stable_id: str = STABLE_ID,
    stage: str = "accuracy-gate",
    reason: str = "historical handwritten authorization",
    attempts: int = 1,
) -> dict[str, Any]:
    """Return one pre-tool handwritten grant row (created_at, no generation)."""

    return {
        "grant_id": legacy_grant_identity(
            stable_id=stable_id, stage=stage, reason=reason, attempts=attempts
        ),
        "stable_id": stable_id,
        "stage": stage,
        "reason": reason,
        "attempts": attempts,
        "created_at": "2026-07-01T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# 1. created_at + v2 identity on every NEW grant row
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_new_grant_row_is_v2_with_created_at_key_and_ceiling(tmp_path: Path) -> None:
    """A freshly built grant carries the full v2 field set, recomputable."""

    from menagerie.crawler.constants import (
        EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT,
        REQUEUE_GRANT_SCHEMA_VERSION,
    )
    from menagerie.crawler.grant_identity import requeue_adjudication_key

    ledger = _ledger(tmp_path)
    grant = _grant(ledger, attempts=3)

    assert grant["schema_version"] == REQUEUE_GRANT_SCHEMA_VERSION
    assert REQUEUE_GRANT_SCHEMA_VERSION.endswith(".v2")
    stamped = datetime.fromisoformat(str(grant["created_at"]).replace("Z", "+00:00"))
    assert stamped.tzinfo is not None, "created_at must be timezone-aware UTC"
    assert grant["adjudication_key"] == requeue_adjudication_key(
        stable_id=STABLE_ID,
        stage="accuracy-gate",
        reason="operator approved re-adjudication",
    )
    assert (
        grant["effective_session_ceiling"]
        == 3 * EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT
    )
    # The row round-trips the strict reader.
    validated = _validated_requeue_grants(ledger, frozenset({STABLE_ID}))
    assert [row["grant_id"] for row in validated.grants] == [grant["grant_id"]]


@pytest.mark.smoke
def test_grant_id_preimage_is_untouched_by_v2_fields(tmp_path: Path) -> None:
    """Schema v2 adds fields OUTSIDE the identity preimage: hashes stay stable."""

    ledger = _ledger(tmp_path)
    grant = _grant(ledger)
    assert grant["grant_id"] == tool_grant_identity(
        generation=grant["new_work_generation"],
        stable_id=STABLE_ID,
        stage="accuracy-gate",
        reason="operator approved re-adjudication",
        attempts=1,
        granted_by="operator",
    )


# ---------------------------------------------------------------------------
# 2. Monotone era ladder: history valid, future refused
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_historical_v1_and_legacy_rows_stay_valid_beside_v2(tmp_path: Path) -> None:
    """The ladder is monotone: v1/legacy rows validate untouched, forever."""

    ledger = _ledger(tmp_path)
    v1_row = _v1_tool_row()
    legacy = _legacy_row(stable_id="m_legacy")
    _write_rows(ledger, [v1_row, legacy])

    validated = _validated_requeue_grants(
        ledger, frozenset({STABLE_ID, "m_legacy", "m_next"})
    )
    assert {row["grant_id"] for row in validated.grants} == {
        v1_row["grant_id"],
        legacy["grant_id"],
    }
    # A NEW grant appends beside the history without rewriting a byte.
    before = ledger.read_bytes()
    fresh = _grant(ledger, stable_id="m_next", reason="post-ladder authorization")
    assert ledger.read_bytes().startswith(before), "history was rewritten"
    revalidated = _validated_requeue_grants(
        ledger, frozenset({STABLE_ID, "m_legacy", "m_next"})
    )
    assert {row["grant_id"] for row in revalidated.grants} == {
        v1_row["grant_id"],
        legacy["grant_id"],
        fresh["grant_id"],
    }


@pytest.mark.smoke
def test_reader_refuses_unknown_future_grant_schema(tmp_path: Path) -> None:
    """A row claiming a version above the ladder refuses typed, never defaults."""

    ledger = _ledger(tmp_path)
    row = _v1_tool_row()
    row["schema_version"] = "menagerie.crawler.requeue-grant.v3"
    _write_rows(ledger, [row])
    with pytest.raises(DriverIntegrationError, match="foreign schema identity"):
        _validated_requeue_grants(ledger, frozenset({STABLE_ID}))


@pytest.mark.smoke
def test_append_refuses_new_current_row_missing_created_at(tmp_path: Path) -> None:
    """A current-version row without created_at refuses at append."""

    from menagerie.crawler.constants import REQUEUE_GRANT_SCHEMA_VERSION
    from menagerie.crawler.grant_identity import requeue_adjudication_key

    ledger = _ledger(tmp_path)
    row = _v1_tool_row()
    row["schema_version"] = REQUEUE_GRANT_SCHEMA_VERSION
    row["adjudication_key"] = requeue_adjudication_key(
        stable_id=STABLE_ID,
        stage="accuracy-gate",
        reason="historical v1 authorization",
    )
    row["effective_session_ceiling"] = 4
    assert "created_at" not in row
    with pytest.raises(CheckpointValidationError, match="created_at"):
        append_canonical_requeue_grant(ledger, row)
    assert not ledger.is_file() or not ledger.read_bytes()


@pytest.mark.smoke
def test_append_refuses_unknown_grant_schema(tmp_path: Path) -> None:
    """The append door itself refuses a version outside the ladder."""

    ledger = _ledger(tmp_path)
    row = _v1_tool_row()
    row["schema_version"] = "menagerie.crawler.requeue-grant.v99"
    with pytest.raises(CheckpointValidationError, match="schema"):
        append_canonical_requeue_grant(ledger, row)


# ---------------------------------------------------------------------------
# 3. Idempotent double-void refusal
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_second_differently_attributed_void_refuses_naming_first(
    tmp_path: Path,
) -> None:
    """Voiding an already-voided grant refuses typed and names the first void."""

    ledger = _ledger(tmp_path)
    grant = _grant(ledger)
    first = build_canonical_requeue_void(
        ledger,
        void_of=str(grant["grant_id"]),
        reason="stale after environment fix",
        voided_by="operator",
    )
    append_canonical_requeue_void(ledger, first)
    with pytest.raises(ValueError, match=str(first["void_id"])):
        build_canonical_requeue_void(
            ledger,
            void_of=str(grant["grant_id"]),
            reason="a different withdrawal story",
            voided_by="second-operator",
        )


@pytest.mark.smoke
def test_double_void_refuses_at_append_when_build_is_bypassed(
    tmp_path: Path,
) -> None:
    """The append layer sees the ledger too: a bypassed second void refuses."""

    ledger = _ledger(tmp_path)
    grant = _grant(ledger)
    first = build_canonical_requeue_void(
        ledger,
        void_of=str(grant["grant_id"]),
        reason="stale after environment fix",
        voided_by="operator",
    )
    append_canonical_requeue_void(ledger, first)
    second = {
        "void_id": requeue_void_identity(
            void_of=str(grant["grant_id"]),
            reason="bypassing build",
            voided_by="second-operator",
        ),
        "schema_version": str(first["schema_version"]),
        "void_of": str(grant["grant_id"]),
        "reason": "bypassing build",
        "voided_by": "second-operator",
        "created_at": NOW,
    }
    with pytest.raises(CheckpointValidationError, match=str(first["void_id"])):
        append_canonical_requeue_void(ledger, second)


@pytest.mark.smoke
def test_identical_revoid_stays_idempotent_no_op_success(tmp_path: Path) -> None:
    """Re-running the SAME void command is a no-op success, never a refusal."""

    ledger = _ledger(tmp_path)
    grant = _grant(ledger)
    first = build_canonical_requeue_void(
        ledger,
        void_of=str(grant["grant_id"]),
        reason="stale after environment fix",
        voided_by="operator",
    )
    assert append_canonical_requeue_void(ledger, first) is True
    # An identical rebuild still succeeds (same identity), and the re-append
    # is the historical idempotent no-op.
    rebuilt = build_canonical_requeue_void(
        ledger,
        void_of=str(grant["grant_id"]),
        reason="stale after environment fix",
        voided_by="operator",
    )
    assert rebuilt["void_id"] == first["void_id"]
    assert append_canonical_requeue_void(ledger, rebuilt) is False


# ---------------------------------------------------------------------------
# 4. Unique adjudication key
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_two_live_grants_for_one_adjudication_refuse_at_build(
    tmp_path: Path,
) -> None:
    """Same decision facts with a different allowance is still ONE adjudication."""

    ledger = _ledger(tmp_path)
    first = _grant(ledger, attempts=1)
    # attempts differs, so the T1-13 logical-twin refusal does NOT fire; the
    # adjudication key must catch the overlap (the Sol storm-grant class).
    with pytest.raises(ValueError, match="adjudicated decision"):
        _grant(ledger, attempts=2)
    with pytest.raises(ValueError, match=str(first["grant_id"])):
        _grant(ledger, attempts=2)
    validated = _validated_requeue_grants(ledger, frozenset({STABLE_ID}))
    assert [row["grant_id"] for row in validated.grants] == [first["grant_id"]]


@pytest.mark.smoke
def test_adjudication_key_frees_after_void(tmp_path: Path) -> None:
    """Void-before-reissue stays the sanctioned second-grant path."""

    ledger = _ledger(tmp_path)
    first = _grant(ledger, attempts=1)
    void = build_canonical_requeue_void(
        ledger,
        void_of=str(first["grant_id"]),
        reason="withdrawn for re-issue",
        voided_by="operator",
    )
    append_canonical_requeue_void(ledger, void)
    second = _grant(ledger, attempts=2)
    assert second["adjudication_key"] == first["adjudication_key"]
    assert second["grant_id"] != first["grant_id"]


@pytest.mark.smoke
def test_adjudication_uniqueness_ignores_keyless_history(tmp_path: Path) -> None:
    """Era rule: the uniqueness check quantifies only over rows that HAVE a key."""

    ledger = _ledger(tmp_path)
    _write_rows(ledger, [_v1_tool_row(attempts=2)])
    # Same decision facts as the keyless historical row; a keyed grant with a
    # different allowance must still issue.
    fresh = _grant(ledger, reason="historical v1 authorization", attempts=1)
    assert "adjudication_key" in fresh


@pytest.mark.smoke
def test_distinct_decisions_get_distinct_adjudication_keys(tmp_path: Path) -> None:
    """A capability grant is a DIFFERENT authorization from a generic grant."""

    from menagerie.crawler.grant_identity import requeue_adjudication_key

    generic = requeue_adjudication_key(
        stable_id=STABLE_ID, stage="accuracy-gate", reason="same words"
    )
    routed = requeue_adjudication_key(
        stable_id=STABLE_ID,
        stage="accuracy-gate",
        reason="same words",
        capability="host-capability-routing",
    )
    assert generic != routed


# ---------------------------------------------------------------------------
# 5. Two-process flock race (B2-X-V7 grant-race arm)
# ---------------------------------------------------------------------------


def _race_worker(
    ledger: str, attempts: int, barrier: Any, results: Any
) -> None:
    """Build against the pre-race ledger, sync on the barrier, then append."""

    try:
        path = Path(ledger)
        grant = build_canonical_requeue_grant(
            path,
            stable_id=STABLE_ID,
            stage="accuracy-gate",
            reason="raced authorization",
            attempts=attempts,
            granted_by="operator",
        )
        barrier.wait(timeout=60)
        append_canonical_requeue_grant(path, grant)
        results.put(("won", attempts, str(grant["grant_id"])))
    except Exception as exc:  # noqa: BLE001 -- the refusal type IS the assertion
        results.put(("refused", attempts, f"{type(exc).__name__}: {exc}"))


@pytest.mark.heavy
def test_two_process_grant_race_has_exactly_one_winner(tmp_path: Path) -> None:
    """Two processes appending one adjudication concurrently: one wins, typed."""

    ledger = _ledger(tmp_path)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results: Any = context.Queue()
    workers = [
        context.Process(
            target=_race_worker, args=(str(ledger), attempts, barrier, results)
        )
        for attempts in (1, 2)
    ]
    for worker in workers:
        worker.start()
    outcomes = [results.get(timeout=120) for _ in workers]
    for worker in workers:
        worker.join(timeout=120)

    by_kind = {kind: detail for kind, _, detail in outcomes}
    assert sorted(kind for kind, _, _ in outcomes) == ["refused", "won"], outcomes
    assert (
        "CheckpointValidationError" in by_kind["refused"]
        or "ValueError" in by_kind["refused"]
    ), outcomes
    # Exactly one grant row landed, and it is the winner's.
    validated = _validated_requeue_grants(ledger, frozenset({STABLE_ID}))
    assert [str(row["grant_id"]) for row in validated.grants] == [by_kind["won"]]


# ---------------------------------------------------------------------------
# 6. Single consumption per adjudicated decision
# ---------------------------------------------------------------------------


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


@pytest.mark.smoke
def test_second_consumption_of_one_adjudication_refuses_typed(
    tmp_path: Path,
) -> None:
    """Two consumptions of one (decision, superseded head) pair refuse typed.

    The residue the append-side uniqueness cannot reach: two same-key grants
    already on the ledger (pre-fix race residue), each with a durable
    consumption event binding the SAME superseded parent revision. That is one
    adjudicated decision spent twice; replay must refuse it, not average it.
    """

    from menagerie.crawler.constants import (
        EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT,
        REQUEUE_GRANT_SCHEMA_VERSION,
    )
    from menagerie.crawler.grant_identity import requeue_adjudication_key

    paths = _paths(tmp_path)
    revision = "sha256:" + "a" * 64
    paths.ledgers.models.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "stable_id": STABLE_ID,
        "record_revision": revision,
        "parent_revision": None,
        "record_seq": 1,
        "status": {
            "kind": "failed",
            "code": "failed:accuracy-gate",
            "stage": "accuracy-gate",
            "human_review": {
                "queue": "crawler-human-review",
                "reason": None,
                "requested_at": NOW,
                "required": True,
            },
        },
        "budget": {"explicit_grants": []},
    }
    with paths.ledgers.models.open("wb") as handle:
        handle.write(canonical_json_bytes(record) + b"\n")

    ledger = canonical_requeue_grants_path(paths.ledgers.models)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    key = requeue_adjudication_key(
        stable_id=STABLE_ID, stage="accuracy-gate", reason="raced authorization"
    )
    grants = []
    for generation, attempts in ((1, 1), (2, 2)):
        grants.append(
            {
                "grant_id": tool_grant_identity(
                    generation=generation,
                    stable_id=STABLE_ID,
                    stage="accuracy-gate",
                    reason="raced authorization",
                    attempts=attempts,
                    granted_by="operator",
                ),
                "schema_version": REQUEUE_GRANT_SCHEMA_VERSION,
                "stable_id": STABLE_ID,
                "stage": "accuracy-gate",
                "reason": "raced authorization",
                "attempts": attempts,
                "granted_by": "operator",
                "new_work_generation": generation,
                "created_at": NOW,
                "adjudication_key": key,
                "effective_session_ceiling": attempts
                * EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT,
            }
        )
    _write_rows(ledger, grants)

    from menagerie.crawler.checkpoint import append_canonical_operational_event
    from menagerie.crawler.constants import OperationalEventStatus
    from menagerie.crawler.operational_identity import mint_operational_event

    events_path = ledger.parent / "events.jsonl"
    for generation, grant in enumerate(grants, start=1):
        details = {
            "grant_id": grant["grant_id"],
            "stable_id": STABLE_ID,
            "stage": "accuracy-gate",
            "reason": "raced authorization",
            "attempts": grant["attempts"],
            "new_work_generation": generation,
            "source_record_revision": revision,
            "new_work_id": stable_hash(
                {
                    "stable_id": STABLE_ID,
                    "grant_id": grant["grant_id"],
                    "parent_revision": revision,
                    "generation": generation,
                }
            ),
        }
        event = mint_operational_event(
            {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "created_at": NOW,
                "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
                "status": OperationalEventStatus.REQUEUE_GRANT_CONSUMED.value,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 1},
                "current_environment": None,
                "run_id": "test-run",
                "machine_id": "test-machine",
                "details": details,
            },
            identity_payload={"grant_id": str(grant["grant_id"])},
        )
        append_canonical_operational_event(events_path, event)

    stub = _stub_driver(paths)
    with JsonlLedger(
        paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION
    ) as operational:
        with pytest.raises(
            DriverIntegrationError, match="one adjudicated decision"
        ):
            AdmissionEnvironmentMixin._consume_requeue_grants(
                stub, operational, frozenset({STABLE_ID})
            )


# ---------------------------------------------------------------------------
# 7. Grant-aging alarm (pure derivation, never a daemon)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_grant_aging_census_classes_live_unconsumed_grants(tmp_path: Path) -> None:
    """Aged, fresh, and age-unknown live grants disclose; spent history does not."""

    from menagerie.crawler.status import (
        GRANT_AGING_ALARM_SECONDS,
        grant_aging_census,
    )

    now = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)
    old = (now - timedelta(seconds=GRANT_AGING_ALARM_SECONDS + 3600)).isoformat()
    fresh_stamp = (now - timedelta(hours=1)).isoformat()

    def keyed(grant_id: str, created_at: Optional[str]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "grant_id": grant_id,
            "stable_id": STABLE_ID,
            "stage": "accuracy-gate",
            "reason": grant_id,
            "attempts": 1,
            "granted_by": "operator",
            "new_work_generation": 1,
        }
        if created_at is not None:
            row["created_at"] = created_at
        return row

    rows = [
        keyed("sha256:" + "1" * 64, old),
        keyed("sha256:" + "2" * 64, fresh_stamp),
        keyed("sha256:" + "3" * 64, None),  # keyless history: age-unknown
        keyed("sha256:" + "4" * 64, old),  # voided below: excluded
        keyed("sha256:" + "5" * 64, old),  # consumed below: excluded
        {
            "void_id": "sha256:" + "b" * 64,
            "void_of": "sha256:" + "4" * 64,
            "reason": "withdrawn",
            "voided_by": "operator",
            "created_at": fresh_stamp,
        },
    ]
    events = [
        {
            "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
            "details": {"grant_id": "sha256:" + "5" * 64},
        }
    ]
    census = grant_aging_census(rows, events, now=now)
    assert census["aged"] == ("sha256:" + "1" * 64,)
    assert census["fresh"] == ("sha256:" + "2" * 64,)
    assert census["age-unknown"] == ("sha256:" + "3" * 64,)


# ---------------------------------------------------------------------------
# 8. MC-Q4 disclosure at issuance (keep attempts:N, disclose the ceiling)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_cli_disclosure_prints_effective_session_ceiling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The requeue tool discloses the effective guard-session ceiling at issuance."""

    from menagerie.crawler.tools.requeue import main

    ledger = tmp_path / "requeue-grants.jsonl"
    exit_code = main(
        [
            STABLE_ID,
            "--grant",
            "2",
            "--stage",
            "accuracy-gate",
            "--reason",
            "operator approved re-adjudication",
            "--repo-root",
            str(tmp_path),
            "--ledger",
            str(ledger),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    payload = json.loads(captured.out.strip().splitlines()[-1])
    from menagerie.crawler.constants import EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT

    expected = 2 * EFFECTIVE_SESSIONS_PER_GRANTED_ATTEMPT
    assert payload["effective_session_ceiling"] == expected
    assert payload["attempts"] == 2, "attempts:N semantics must not change"
    assert "effective guard-session ceiling" in captured.err
    assert str(expected) in captured.err


# ---------------------------------------------------------------------------
# 9. Sibling authority families already carry created_at (era tripwires)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_requeue_void_and_park_void_rows_require_created_at() -> None:
    """The void families' created_at requirement is a permanent tripwire."""

    from menagerie.crawler.grant_parks import (
        GrantParkError,
        build_grant_park_void,
    )

    with pytest.raises(GrantParkError):
        build_grant_park_void(
            void_of="sha256:" + "c" * 64,
            reason="drain",
            voided_by="operator",
            created_at="",
        )
