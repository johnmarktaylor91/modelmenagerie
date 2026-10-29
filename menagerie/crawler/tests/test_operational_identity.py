"""Closed operational-event identity policy and mint interdiction tests."""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pytest

from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
)
from menagerie.crawler.operational_identity import (
    OPERATIONAL_EVENT_IDENTITY_REGISTRY,
    OperationalEventIdentityClass,
    mint_operational_event,
)
from menagerie.crawler.recordio import JsonlLedger

pytestmark = pytest.mark.smoke


def _event(event_kind: str, *, created_at: str) -> dict[str, object]:
    """Build one schema-complete operational event without its identity.

    Parameters
    ----------
    event_kind:
        Closed or deliberately invalid kind under test.
    created_at:
        UTC occurrence clock.

    Returns
    -------
    dict[str, object]
        Logical event ready for a typed identity constructor.
    """

    return {
        "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
        "created_at": created_at,
        "event_kind": event_kind,
        "status": "healthy",
        "provider": None,
        "observed_response": None,
        "reset_at": None,
        "queued_work_counts": {"models": 1},
        "current_environment": None,
        "run_id": "run-test",
        "machine_id": "machine-test",
        "details": {"kind": "identity-regression", "stable_id": "m1"},
    }


def _contains_stable_hash(node: ast.AST) -> bool:
    """Return whether an AST subtree invokes ``stable_hash``.

    Parameters
    ----------
    node:
        Candidate syntax subtree.

    Returns
    -------
    bool
        True when the subtree contains a raw hash call.
    """

    return any(
        isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == "stable_hash"
        for candidate in ast.walk(node)
    )


def _event_id_expressions(tree: ast.AST) -> Iterable[tuple[int, ast.AST]]:
    """Yield durable ``event_id`` producer expressions from one syntax tree.

    Parameters
    ----------
    tree:
        Parsed Python module.

    Yields
    ------
    tuple[int, ast.AST]
        Source line and value expression for each producer.
    """

    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "event_id" and value is not None:
                    yield node.lineno, value
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "event_id"
            ):
                yield node.lineno, value


def _raw_event_hash_mints(source: str) -> tuple[int, ...]:
    """Return lines where a raw stable hash flows into a durable event ID.

    Parameters
    ----------
    source:
        Python source text.

    Returns
    -------
    tuple[int, ...]
        Sorted violating producer lines.
    """

    tree = ast.parse(source)
    hashed_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
        and _contains_stable_hash(node.value)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    violations: set[int] = set()
    for line, expression in _event_id_expressions(tree):
        referenced = {
            candidate.id for candidate in ast.walk(expression) if isinstance(candidate, ast.Name)
        }
        if _contains_stable_hash(expression) or referenced & hashed_names:
            violations.add(line)
    return tuple(sorted(violations))


def test_operational_event_identity_registry_is_exhaustive_and_pinned() -> None:
    """Every kind has one reviewed replay class, including the new disclosure."""

    idempotent = {
        OperationalEventKind.USAGE_PAUSE,
        OperationalEventKind.WAKEUP,
        OperationalEventKind.CHECKPOINT,
        OperationalEventKind.CHECKPOINT_REVIEW,
        OperationalEventKind.REVIEW_SIGNOFF,
        OperationalEventKind.PROGRESS_NOTIFICATION,
        OperationalEventKind.NOTIFICATION_DELIVERY,
        OperationalEventKind.REQUEUE_GRANT_CONSUMED,
        OperationalEventKind.WORKER_LEASE_OPENED,
        OperationalEventKind.WORKER_LEASE_STARTED,
        OperationalEventKind.WORKER_LEASE_CLOSED,
        OperationalEventKind.WORKER_LEASE_REAPED,
        OperationalEventKind.ORPHAN_WORKER_RECOVERED,
        OperationalEventKind.WAKEUP_INSTALLED,
        OperationalEventKind.WAKEUP_DEACTIVATED,
        OperationalEventKind.WAKEUP_HEALTH_DEGRADED,
        OperationalEventKind.CAMPAIGN_COMPLETED,
        OperationalEventKind.OPERATOR_CANCELLED,
        OperationalEventKind.WORKER_SHUTDOWN_INTERRUPTED,
        # WS-1 clause 6 x ID-1: a generation discharge is ONE semantic fact
        # about one (stable_id, work_id) generation -- a crash-replayed
        # discharge dedups on identity; a conflicting one collides typed.
        OperationalEventKind.REQUEUE_GENERATION_DISCHARGED,
    }
    assert set(OPERATIONAL_EVENT_IDENTITY_REGISTRY) == set(OperationalEventKind)
    assert {
        kind
        for kind, spec in OPERATIONAL_EVENT_IDENTITY_REGISTRY.items()
        if spec.identity_class is OperationalEventIdentityClass.IDEMPOTENT_FIRST_CLOCK
    } == idempotent


def test_idempotent_event_constructor_reuses_identity_and_first_clock(tmp_path: Path) -> None:
    """Class A replays retain the first wall-clock observation.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    first = mint_operational_event(
        _event(
            OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
            created_at="2026-08-16T10:00:00Z",
        ),
        identity_payload={"grant_id": "requeue-test"},
    )
    replay = mint_operational_event(
        _event(
            OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
            created_at="2026-08-16T10:05:00Z",
        ),
        identity_payload={"grant_id": "requeue-test"},
    )
    assert first["event_id"] == replay["event_id"]
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        assert ledger.append(first).appended
        repeated = ledger.append(replay)
    assert not repeated.appended
    assert repeated.record["created_at"] == "2026-08-16T10:00:00Z"


def test_per_occurrence_constructor_changes_identity_with_clock(tmp_path: Path) -> None:
    """Class B timestamps mint distinct identities and append normal rows.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    first = mint_operational_event(
        _event(OperationalEventKind.CAMPAIGN_HEALTH.value, created_at="2026-08-16T10:00:00Z")
    )
    later_event = deepcopy(first)
    later_event.pop("event_id")
    later_event["created_at"] = "2026-08-16T10:05:00Z"
    later = mint_operational_event(later_event)
    assert first["event_id"] != later["event_id"]
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        assert ledger.append(first).appended
        assert ledger.append(later).appended
        assert len(ledger.records) == 2


def test_typed_constructor_fails_closed_on_unknown_or_wrong_class() -> None:
    """Unknown kinds and caller-selected class changes are refused."""

    with pytest.raises(ValueError, match="unknown operational event kind"):
        mint_operational_event(
            _event("future-unregistered-kind", created_at="2026-08-16T10:00:00Z")
        )
    with pytest.raises(ValueError, match="requires a non-empty idempotent identity"):
        mint_operational_event(
            _event(
                OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
                created_at="2026-08-16T10:00:00Z",
            )
        )
    with pytest.raises(ValueError, match="forbids an idempotent identity payload"):
        mint_operational_event(
            _event(
                OperationalEventKind.CAMPAIGN_HEALTH.value,
                created_at="2026-08-16T10:00:00Z",
            ),
            identity_payload={"fact": "wrong-class"},
        )


def test_raw_operational_stable_hash_mints_are_interdicted() -> None:
    """Production durable event IDs can only come from typed constructors."""

    crawler_root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for path in sorted(crawler_root.rglob("*.py")):
        if "tests" in path.parts or path.name == "operational_identity.py":
            continue
        for line in _raw_event_hash_mints(path.read_text(encoding="utf-8")):
            violations.append(f"{path.relative_to(crawler_root)}:{line}")
    assert violations == []
    synthetic = "identity = stable_hash(event)\nrow = {'event_id': f'event-{identity}'}\n"
    assert _raw_event_hash_mints(synthetic) == (2,)
