"""B3-XV-4: replay-derived owner authority for driver-lane attempt failures.

The 200a attempts ledger carries 267 failure rows whose reasons are non-worker
driver/admission-lane spellings (``effort-exhausted:author-sessions``,
``malformed-result``, ``build-failed``, ``session-crashed``,
``blocked-prerequisite``, ``disk-floor``, ``effort-exhausted:wall-seconds``,
``solve-failed``, ``coverage-incomplete``). The sealed
:data:`~menagerie.crawler.driver_receipts.WORKER_FAILURE_REASON_OWNERS`
vocabulary is deliberately worker-scoped and never covered them, so the
B3-XV-4 attempts-owner census read them as silent owner-less residue.

The fix is a SECOND closed authority
(:mod:`menagerie.crawler.attempt_owner_derivation`) that derives each row's
owner at replay time from the row's own recorded ``(stage, reason_code,
exception_type)`` evidence -- never from a persisted owner field a producer
could backfill. A triple whose owner is genuinely unknowable from the row
alone gets a typed unowned-disposition the census must DISCLOSE (segregated,
never counted as owned); a triple outside both closed sets stays silent
owner-less and keeps the census RED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.attempt_owner_derivation import (
    ATTEMPT_TRIPLE_OWNERS,
    SESSION_ALLOWANCE_EXHAUSTION_TRIPLE,
    UNOWNED_ATTEMPT_DISPOSITIONS,
    UNOWNED_DISPOSITION_VOCABULARY,
    derive_attempt_failure_owner,
)
from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    WORKER_FAILURE_REASON_OWNERS,
)

#: Every actor an owner may name (the closed five-actor vocabulary of
#: test_terminal_owner_census).
_CLOSED_OWNER_VOCABULARY = frozenset({"model", "author", "checker", "host", "machinery"})


def _redacted(key: str) -> dict[str, Any]:
    return {
        "content_sha256": "sha256:" + "a" * 64,
        "diagnostic_key": key,
        "local_path": ".crawl-local/diagnostics/sha256:" + "a" * 64 + ".json",
        "redaction": "externally-controlled-text-v1",
    }


def _error(stage: str, reason: str, exception_type: str, **extra: Any) -> dict[str, Any]:
    """One attempt-ledger error mapping in the full recorded v3 shape."""

    return {
        "stage": stage,
        "reason_code": reason,
        "exception_type": exception_type,
        "details": {"driver_observed": True, "environment": None},
        "native_crash": False,
        "message": _redacted("$.error.message"),
        "traceback": _redacted("$.error.traceback"),
        "no_traceback_reason": None,
        "root_cause_fingerprint": "sha256:" + "b" * 64,
        **extra,
    }


# --------------------------------------------------------------------------- #
# Closure of the vocabulary
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_derivation_owner_vocabulary_is_closed() -> None:
    """Every derived owner is one of the five closed actors."""

    assert set(ATTEMPT_TRIPLE_OWNERS.values()) <= _CLOSED_OWNER_VOCABULARY
    assert set(UNOWNED_ATTEMPT_DISPOSITIONS.values()) == set(UNOWNED_DISPOSITION_VOCABULARY)
    for disposition in UNOWNED_DISPOSITION_VOCABULARY:
        assert disposition.startswith("unowned-historical:")


@pytest.mark.smoke
def test_derivation_authorities_are_disjoint_and_never_shadow_the_worker_map() -> None:
    """The three closed key sets are pairwise disjoint, and no derived reason
    also lives in the sealed worker vocabulary (this module extends coverage,
    it never re-adjudicates a worker-owned reason)."""

    owner_keys = set(ATTEMPT_TRIPLE_OWNERS)
    unowned_keys = set(UNOWNED_ATTEMPT_DISPOSITIONS)
    assert owner_keys.isdisjoint(unowned_keys)
    assert SESSION_ALLOWANCE_EXHAUSTION_TRIPLE not in owner_keys
    assert SESSION_ALLOWANCE_EXHAUSTION_TRIPLE not in unowned_keys
    derived_reasons = {
        reason
        for _stage, reason, _exception_type in (
            *owner_keys,
            *unowned_keys,
            SESSION_ALLOWANCE_EXHAUSTION_TRIPLE,
        )
    }
    assert derived_reasons.isdisjoint(WORKER_FAILURE_REASON_OWNERS)


# --------------------------------------------------------------------------- #
# The closed static table
# --------------------------------------------------------------------------- #

#: Expected owner per static triple: the exact 200a evidence census
#: (15 triples over 267 rows), minus the settlement-derived exhaustion triple
#: and the two typed unowned dispositions.
_EXPECTED_STATIC_OWNERS = {
    (
        "author",
        "effort-exhausted:wall-seconds",
        "menagerie.crawler.author_dispatch.AuthorEffortExhaustionClaim",
    ): "author",
    (
        "author",
        "malformed-result",
        "menagerie.crawler.artifact_transactions.ArtifactBindingError",
    ): "author",
    (
        "author",
        "malformed-result",
        "menagerie.crawler.author_dispatch.AuthorExecutorContractError",
    ): "author",
    (
        "author",
        "malformed-result",
        "menagerie.crawler.driver_contracts.AuthorBlockedPrerequisite",
    ): "author",
    (
        "environment",
        "blocked-prerequisite",
        "menagerie.crawler.driver_contracts.AuthorBlockedPrerequisite",
    ): "author",
    (
        "evidence",
        "coverage-incomplete",
        "menagerie.crawler.driver_contracts.DriverIntegrationError",
    ): "author",
    (
        "author",
        "session-crashed",
        "menagerie.crawler.recordio.LedgerConflictError",
    ): "machinery",
    (
        "author",
        "session-crashed",
        "menagerie.crawler.driver_contracts.DriverIntegrationError",
    ): "machinery",
    (
        "environment",
        "build-failed",
        "menagerie.crawler.driver_contracts.DriverIntegrationError",
    ): "machinery",
    (
        "environment",
        "build-failed",
        "menagerie.crawler.env_lifecycle.EnvironmentExactnessError",
    ): "machinery",
    (
        "environment",
        "solve-failed",
        "menagerie.crawler.env_lifecycle.EnvironmentSolveError",
    ): "machinery",
    (
        "resource",
        "disk-floor",
        "menagerie.crawler.env_lifecycle.DiskRecoveryError",
    ): "host",
}


@pytest.mark.smoke
def test_static_triple_table_is_exactly_the_adjudicated_set() -> None:
    """The closed owner table matches the adjudicated triple-by-triple census;
    widening it is a NEW deliberated change, never a drive-by edit."""

    assert dict(ATTEMPT_TRIPLE_OWNERS) == _EXPECTED_STATIC_OWNERS


@pytest.mark.smoke
def test_derivation_matches_the_closed_table() -> None:
    """Each static triple derives its adjudicated owner with the closed rule."""

    for (stage, reason, exception_type), owner in _EXPECTED_STATIC_OWNERS.items():
        derivation = derive_attempt_failure_owner(
            _error(stage, reason, exception_type),
            stable_id="m1",
            work_id="work-m1",
            spend_events=(),
        )
        assert derivation is not None, (stage, reason, exception_type)
        assert derivation.owner == owner
        assert derivation.disposition is None
        assert derivation.rule == "closed-triple"


@pytest.mark.smoke
def test_unknown_or_malformed_triples_stay_silent_ownerless() -> None:
    """A triple outside both closed sets -- or a row missing any leg of the
    triple -- derives NOTHING: it must keep the census red, never guess."""

    unknown = _error("author", "session-crashed", "builtins.KeyError")
    assert (
        derive_attempt_failure_owner(
            unknown, stable_id="m1", work_id="work-m1", spend_events=()
        )
        is None
    )
    missing_type = {"stage": "author", "reason_code": "session-crashed"}
    assert (
        derive_attempt_failure_owner(
            missing_type, stable_id="m1", work_id="work-m1", spend_events=()
        )
        is None
    )
    non_string = _error("author", "session-crashed", "builtins.ValueError")
    non_string["stage"] = 7
    assert (
        derive_attempt_failure_owner(
            non_string, stable_id="m1", work_id="work-m1", spend_events=()
        )
        is None
    )


# --------------------------------------------------------------------------- #
# Typed unowned dispositions
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_unowned_dispositions_disclose_without_owning() -> None:
    """The two genuinely unknowable session-crash triples get a typed
    disposition and NO owner."""

    expected = {
        (
            "author",
            "session-crashed",
            "menagerie.crawler.driver_contracts.AuthorSessionRetryExhausted",
        ): "unowned-historical:no-publication-session-death",
        (
            "author",
            "session-crashed",
            "builtins.ValueError",
        ): "unowned-historical:untyped-session-crash",
    }
    assert dict(UNOWNED_ATTEMPT_DISPOSITIONS) == expected
    for (stage, reason, exception_type), disposition in expected.items():
        derivation = derive_attempt_failure_owner(
            _error(stage, reason, exception_type),
            stable_id="m1",
            work_id="work-m1",
            spend_events=(),
        )
        assert derivation is not None
        assert derivation.owner is None
        assert derivation.disposition == disposition
        assert derivation.rule == "unowned-disposition"


# --------------------------------------------------------------------------- #
# The settlement-derived exhaustion triple (T2-07/T1-13 rule)
# --------------------------------------------------------------------------- #


def _reserved(spend_id: str, stable_id: str = "m1", work_id: str = "work-m1") -> dict[str, Any]:
    return {
        "event": "session_reserved",
        "lane": "author",
        "spend_id": spend_id,
        "axis": {"work_generations": [{"stable_id": stable_id, "work_id": work_id}]},
    }


def _settled(spend_id: str, local_outcome: str) -> dict[str, Any]:
    from menagerie.crawler.paid_session_accounting import PAID_SESSION_SETTLEMENT_SCHEMA

    return {
        "schema_version": PAID_SESSION_SETTLEMENT_SCHEMA,
        "event": "session_settled",
        "spend_id": spend_id,
        "observed": {"wall_seconds": 5.0},
        "evidence": "provider-query",
        "custody_outcome": "exited",
        "local_outcome": local_outcome,
        "provider": {
            "accepted": True,
            "attested_wall_seconds": 5.0,
            "receipt_sha256": "sha256:" + "d" * 64,
            "adapter_identity": "provider-reconcile-evidence/2",
        },
        "machine": {"wall_seconds": 5.0, "source": "writer-monotonic"},
    }


@pytest.mark.smoke
def test_session_allowance_owner_follows_the_settlement_dimension() -> None:
    """The exhaustion triple's owner is the T2-07/T1-13 settlement replay:
    genuine exhaustion is the author's, a storm-burned allowance is host
    weather, and zero evidence fails toward the author (never toward the
    restorable class)."""

    stage, reason, exception_type = SESSION_ALLOWANCE_EXHAUSTION_TRIPLE
    error = _error(stage, reason, exception_type)

    no_evidence = derive_attempt_failure_owner(
        error, stable_id="m1", work_id="work-m1", spend_events=()
    )
    assert no_evidence is not None
    assert no_evidence.owner == "author"
    assert no_evidence.rule == "session-allowance-dimension:author-sessions"

    storm_events = (
        _reserved("s1"),
        _reserved("s2"),
        _reserved("s3"),
        _settled("s1", "failed-harness-api-error"),
        _settled("s2", "failed-machine-spawn"),
        _settled("s3", "completed"),
    )
    storm = derive_attempt_failure_owner(
        error, stable_id="m1", work_id="work-m1", spend_events=storm_events
    )
    assert storm is not None
    assert storm.owner == "host"
    assert storm.rule == "session-allowance-dimension:provider-storm"
    assert storm.owner in NEVER_MODEL_TERMINAL_OWNERS


# --------------------------------------------------------------------------- #
# Census integration: GREEN-or-disclosed on owned rows, RED on silent residue
# --------------------------------------------------------------------------- #


def _write_records_root(root: Path, attempt_errors: list[dict[str, Any]]) -> None:
    """Materialize a minimal flat-layout records root with the given failures."""

    from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3
    from menagerie.crawler.recordio import JsonlLedger
    from menagerie.crawler.tests.conftest import make_failed_attempt

    root.mkdir(parents=True, exist_ok=True)
    (root / "models.jsonl").write_text(
        json.dumps({"stable_id": "m1", "status": {"code": "ok", "kind": "resolved"}}) + "\n",
        encoding="utf-8",
    )
    (root / "gates.jsonl").write_text("", encoding="utf-8")
    with JsonlLedger(root / "attempts.jsonl", ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        for index, error in enumerate(attempt_errors):
            attempt = make_failed_attempt(
                f"m{index + 1}",
                attempt_id=f"attempt-{index + 1}",
                stage=str(error["stage"]),
                reason_code=str(error["reason_code"]),
            )
            attempt["error"] = error
            attempt.pop("ledger_seq", None)
            attempt.pop("payload_sha256", None)
            ledger.append(attempt)


def _census(tmp_path: Path, attempt_errors: list[dict[str, Any]], tag: str) -> dict[str, Any]:
    from menagerie.crawler.tools.shadow_ledger_replay import attempts_owners, snapshot

    records = tmp_path / f"records-{tag}"
    shadow = tmp_path / f"shadow-{tag}"
    _write_records_root(records, attempt_errors)
    snapshot(records, shadow)
    return attempts_owners(shadow)


@pytest.mark.smoke
def test_census_derives_discloses_and_segregates(tmp_path: Path) -> None:
    """Derived owners and disclosed unowned rows are counted in their own
    segregated buckets; the census is green-with-disclosure, never silently
    green and never red."""

    report = _census(
        tmp_path,
        [
            # Worker-vocabulary owned (direct authority).
            _error("policy", "timeout", "builtins.TimeoutError"),
            # Closed-triple derived owner.
            _error(
                "resource",
                "disk-floor",
                "menagerie.crawler.env_lifecycle.DiskRecoveryError",
            ),
            # Typed unowned disposition (disclosed).
            _error("author", "session-crashed", "builtins.ValueError"),
        ],
        "disclosed",
    )
    assert report["owners"] == {"model": 1}
    assert report["derived_owners"] == {"host": 1}
    assert report["derivation_rules"] == {
        "closed-triple": 1,
        "unowned-disposition": 1,
    }
    assert report["unowned_disclosed"] == [
        {
            "stable_id": "m3",
            "reason_code": "session-crashed",
            "disposition": "unowned-historical:untyped-session-crash",
        }
    ]
    assert report["unowned_disclosed_count"] == 1
    assert report["ownerless"] == []
    assert report["verdict"] == "green-with-disclosed-unowned"
    assert report["green"] is True


@pytest.mark.smoke
def test_census_is_plain_green_with_zero_disclosures(tmp_path: Path) -> None:
    report = _census(
        tmp_path,
        [
            _error(
                "environment",
                "build-failed",
                "menagerie.crawler.env_lifecycle.EnvironmentExactnessError",
            )
        ],
        "green",
    )
    assert report["verdict"] == "green"
    assert report["green"] is True
    assert report["unowned_disclosed"] == []


@pytest.mark.smoke
def test_census_stays_red_on_silent_residue(tmp_path: Path) -> None:
    """A failure outside every closed authority is silent residue: RED."""

    report = _census(
        tmp_path,
        [_error("author", "session-crashed", "builtins.KeyError")],
        "red",
    )
    assert report["verdict"] == "red"
    assert report["green"] is False
    assert report["ownerless"] == [
        {"stable_id": "m1", "reason_code": "session-crashed"}
    ]


# --------------------------------------------------------------------------- #
# The sweep-facing derivation packet (read-only records; packet is the only write)
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_derivation_packet_names_every_failed_attempt(tmp_path: Path) -> None:
    from menagerie.crawler.tools.shadow_ledger_replay import attempts_owners, snapshot

    records = tmp_path / "records-packet"
    shadow = tmp_path / "shadow-packet"
    _write_records_root(
        records,
        [
            _error("policy", "timeout", "builtins.TimeoutError"),
            _error(
                "resource",
                "disk-floor",
                "menagerie.crawler.env_lifecycle.DiskRecoveryError",
            ),
            _error("author", "session-crashed", "builtins.ValueError"),
            _error("author", "session-crashed", "builtins.KeyError"),
        ],
    )
    snapshot(records, shadow)
    packet_path = tmp_path / "packet.json"
    report = attempts_owners(shadow, derivation_packet=packet_path)
    assert report["verdict"] == "red"  # the unknown row keeps the census honest
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["kind"] == "attempts-owner-derivation-packet"
    rows = packet["rows"]
    assert [row["source"] for row in rows] == [
        "worker-vocabulary",
        "derived-owner",
        "unowned-disclosed",
        "ownerless",
    ]
    by_source = {row["source"]: row for row in rows}
    assert by_source["worker-vocabulary"]["owner"] == "model"
    assert by_source["worker-vocabulary"]["never_model_terminal"] is False
    assert by_source["derived-owner"]["owner"] == "host"
    assert by_source["derived-owner"]["never_model_terminal"] is True
    assert by_source["derived-owner"]["rule"] == "closed-triple"
    assert (
        by_source["unowned-disclosed"]["disposition"]
        == "unowned-historical:untyped-session-crash"
    )
    assert by_source["unowned-disclosed"]["owner"] is None
    assert by_source["ownerless"]["owner"] is None
    assert by_source["ownerless"]["rule"] is None
    for row in rows:
        for field in (
            "attempt_id",
            "stable_id",
            "work_id",
            "stage",
            "reason_code",
            "exception_type",
        ):
            assert field in row
