"""One canonical requeue grant identity across builder, driver, and reducer.

T1-08 (docket C-GRANT-ID): routed ``--intent`` requeue grants carried TWO
incompatible identity preimages. The checkpoint builder and driver rehydration
fold ``target_intent`` into the ``grant_id`` hash; the reducer's
consumption-side validator hashed six fields WITHOUT it, so every routed grant
was refused with ``explicit requeue grant identity is invalid`` at the
canonical append -- AFTER the consumption event was durable. The spend was
burned, the grant dangled active forever (a consumed grant cannot be voided
out), and the routed model was permanently stuck. OPS-9 interdicted ``--intent``
grants as the stopgap.

This suite pins the unified contract: one preimage function owned by
``menagerie.crawler.grant_identity`` and imported by all three authorities, a
typed reducer refusal for legacy rows that smuggle an intent under the
intent-blind legacy hash (never silent acceptance), consumption facts checked
against the SAME identity fields at the reducer that the driver enforces, and
the end-to-end routed-grant lifecycle: issue -> consume -> canonical append
accepted -> the re-drive honors the granted intent.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import pytest

from menagerie.crawler.checkpoint import (
    append_canonical_requeue_grant,
    build_canonical_requeue_grant,
    canonical_operational_ledger_path,
    canonical_requeue_grants_path,
)
from menagerie.crawler.constants import (
    OPERATIONAL_EVENT_SCHEMA_VERSION,
    OperationalEventKind,
)
from menagerie.crawler.driver_admission import _validated_requeue_grants
from menagerie.crawler.driver_contracts import DriverIntegrationError, DriverPaths
from menagerie.crawler.grant_identity import (
    LEGACY_INTENT_REFUSAL,
    expected_grant_id,
    legacy_grant_identity,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.reducer import (
    ReductionError,
    _validate_persisted_requeue_lineage,
    default_ledger_paths,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeForward,
    InaccurateChecker,
    _driver,
    _paths as _driver_paths,
    _snapshot,
)

NOW = "2026-08-11T12:00:00Z"
STABLE_ID = "m_routed"
PARENT_REVISION = "sha256:" + "a" * 64
CHILD_REVISION = "sha256:" + "b" * 64


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


def _tool_grant(paths: DriverPaths, *, target_intent: Optional[str] = None) -> Mapping[str, Any]:
    """Build and append one tool-minted grant through the real builder.

    Parameters
    ----------
    paths:
        Harness driver paths.
    target_intent:
        Optional routed environment correction.

    Returns
    -------
    Mapping[str, Any]
        Appended grant payload.
    """

    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    grant = build_canonical_requeue_grant(
        canonical,
        stable_id=STABLE_ID,
        stage="environment",
        reason="wrong environment routing",
        attempts=1,
        granted_by="operator",
        target_intent=target_intent,
    )
    append_canonical_requeue_grant(canonical, grant)
    return grant


def test_grant_builder_discovers_snapshot_inside_legacy_intake_container(
    tmp_path: Path,
) -> None:
    """Legacy container directories do not masquerade as malformed snapshots."""

    snapshot = _snapshot(tmp_path, count=1)
    container = tmp_path / "records" / "intake" / "intake_head50"
    container.mkdir(parents=True)
    snapshot.root.rename(container / snapshot.root.name)
    ledger = tmp_path / "records" / "operational" / "requeue-grants.jsonl"

    grant = build_canonical_requeue_grant(
        ledger,
        stable_id=snapshot.items[0].stable_id,
        stage="author",
        reason="reviewed retry",
        attempts=1,
        granted_by="operator",
    )

    assert grant["stable_id"] == snapshot.items[0].stable_id


def _append_consumption(
    paths: DriverPaths,
    grant: Mapping[str, Any],
    *,
    details_overrides: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Append the canonical consumption event exactly as the driver mints it.

    Parameters
    ----------
    paths:
        Harness driver paths.
    grant:
        Persisted grant row being consumed.
    details_overrides:
        Optional field overrides simulating divergent consumption facts.

    Returns
    -------
    Mapping[str, Any]
        Consumption event details.
    """

    generation = int(grant.get("new_work_generation", 1))
    details: dict[str, Any] = {
        "grant_id": grant["grant_id"],
        "stable_id": grant["stable_id"],
        "stage": grant["stage"],
        "reason": grant["reason"],
        "attempts": grant["attempts"],
        "source_record_revision": PARENT_REVISION,
        "new_work_generation": generation,
        "new_work_id": stable_hash(
            {
                "stable_id": grant["stable_id"],
                "grant_id": grant["grant_id"],
                "parent_revision": PARENT_REVISION,
                "generation": generation,
            }
        ),
    }
    if grant.get("target_intent") is not None:
        details["target_intent"] = grant["target_intent"]
    if details_overrides:
        details.update(details_overrides)
        details = {key: value for key, value in details.items() if value is not None}
    events_path = canonical_operational_ledger_path(paths.ledgers.models)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with JsonlLedger(events_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        ledger.append(
            {
                "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
                "event_id": f"requeue-consumed-{str(grant['grant_id'])[7:31]}",
                "created_at": NOW,
                "event_kind": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
                "status": OperationalEventKind.REQUEUE_GRANT_CONSUMED.value,
                "provider": None,
                "observed_response": None,
                "reset_at": None,
                "queued_work_counts": {"models": 1},
                "current_environment": None,
                "run_id": "run-test",
                "machine_id": "machine-test",
                "details": details,
            }
        )
    return details


def _validate_lineage(
    paths: DriverPaths, grant: Mapping[str, Any], details: Mapping[str, Any]
) -> None:
    """Run the reducer's persisted-lineage validator over one granted supersession.

    Parameters
    ----------
    paths:
        Harness driver paths.
    grant:
        Persisted grant row.
    details:
        Durable consumption event details.
    """

    parent = {
        "stable_id": STABLE_ID,
        "record_revision": PARENT_REVISION,
        "parent_revision": None,
        "budget": {"explicit_grants": []},
    }
    child = {
        "stable_id": STABLE_ID,
        "record_revision": CHILD_REVISION,
        "parent_revision": PARENT_REVISION,
        "budget": {"explicit_grants": [grant["grant_id"]]},
        "status": {"attempt_ids": ["att-1"]},
    }
    attempt = {
        "stable_id": STABLE_ID,
        "work_id": details["new_work_id"],
        "attempt_id": "att-1",
    }
    _validate_persisted_requeue_lineage([parent, child], paths.ledgers, [attempt], [], [])


@pytest.mark.smoke
def test_routed_grant_lineage_survives_the_canonical_append_validator(
    tmp_path: Path,
) -> None:
    """The T1-08 flip: a routed tool grant validates exactly like an unrouted one.

    Before the unified preimage the reducer hashed six fields without
    ``target_intent`` and refused every routed grant at the canonical append
    with ``explicit requeue grant identity is invalid``.
    """

    control_paths = _paths(tmp_path / "control")
    control = _tool_grant(control_paths, target_intent=None)
    control_details = _append_consumption(control_paths, control)
    _validate_lineage(control_paths, control, control_details)

    routed_paths = _paths(tmp_path / "routed")
    routed = _tool_grant(routed_paths, target_intent="graph")
    routed_details = _append_consumption(routed_paths, routed)
    _validate_lineage(routed_paths, routed, routed_details)


@pytest.mark.smoke
def test_legacy_grant_carrying_an_intent_is_refused_typed_at_the_reducer(
    tmp_path: Path,
) -> None:
    """A legacy two-identity row refuses loudly; it is never silently accepted.

    The legacy preimage is intent-blind, so a legacy row that smuggles a
    ``target_intent`` field under a matching legacy hash used to validate as if
    the routing did not exist. The reducer must refuse it as a typed
    two-identity row instead.
    """

    paths = _paths(tmp_path)
    reason = "human reviewed corrected evidence"
    grant_id = stable_hash(
        {
            "stable_id": STABLE_ID,
            "stage": "environment",
            "reason": reason,
            "grant": 1,
        }
    )
    grant = {
        "grant_id": grant_id,
        "stable_id": STABLE_ID,
        "stage": "environment",
        "reason": reason,
        "attempts": 1,
        "created_at": NOW,
        "target_intent": "graph",
    }
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with canonical.open("wb") as handle:
        handle.write(canonical_json_bytes(dict(grant)) + b"\n")
    details = _append_consumption(paths, grant)

    with pytest.raises(
        ReductionError, match="legacy requeue grant cannot carry an environment intent"
    ):
        _validate_lineage(paths, grant, details)


@pytest.mark.smoke
def test_consumption_facts_must_match_the_grant_intent(tmp_path: Path) -> None:
    """The reducer holds consumption to the same identity fields as the driver.

    A durable consumption event whose ``target_intent`` diverges from (or
    omits) the grant's routing is a fact conflict, exactly as
    ``_consume_requeue_grants`` already enforces on the driver side.
    """

    omitted_paths = _paths(tmp_path / "omitted")
    grant = _tool_grant(omitted_paths, target_intent="graph")
    details = _append_consumption(omitted_paths, grant, details_overrides={"target_intent": None})
    with pytest.raises(
        ReductionError, match="explicit requeue grant facts conflict with consumption"
    ):
        _validate_lineage(omitted_paths, grant, details)

    diverged_paths = _paths(tmp_path / "diverged")
    grant = _tool_grant(diverged_paths, target_intent="graph")
    details = _append_consumption(
        diverged_paths, grant, details_overrides={"target_intent": "vision"}
    )
    with pytest.raises(
        ReductionError, match="explicit requeue grant facts conflict with consumption"
    ):
        _validate_lineage(diverged_paths, grant, details)


@pytest.mark.smoke
def test_builder_driver_and_reducer_share_one_grant_preimage(tmp_path: Path) -> None:
    """All three authorities agree on the identity of routed and unrouted grants.

    The builder mints through :func:`tool_grant_identity`, driver rehydration
    re-verifies through :func:`expected_grant_id`, and the reducer's lineage
    validator verifies the same function -- so one grant can never carry two
    identities again.
    """

    for index, target_intent in enumerate((None, "graph")):
        paths = _paths(tmp_path / f"case-{index}")
        grant = _tool_grant(paths, target_intent=target_intent)
        assert expected_grant_id(grant) == grant["grant_id"]
        ledger = _validated_requeue_grants(
            canonical_requeue_grants_path(paths.ledgers.models),
            frozenset({STABLE_ID}),
        )
        assert [row["grant_id"] for row in ledger.grants] == [grant["grant_id"]]


@pytest.mark.smoke
def test_driver_rehydration_refuses_a_legacy_two_identity_row(tmp_path: Path) -> None:
    """Driver rehydration shares the reducer's typed legacy-intent refusal."""

    paths = _paths(tmp_path)
    reason = "human reviewed corrected evidence"
    grant = {
        "grant_id": legacy_grant_identity(
            stable_id=STABLE_ID, stage="environment", reason=reason, attempts=1
        ),
        "stable_id": STABLE_ID,
        "stage": "environment",
        "reason": reason,
        "attempts": 1,
        "created_at": NOW,
        "target_intent": "graph",
    }
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with canonical.open("wb") as handle:
        handle.write(canonical_json_bytes(dict(grant)) + b"\n")

    with pytest.raises(DriverIntegrationError, match=LEGACY_INTENT_REFUSAL):
        _validated_requeue_grants(canonical, frozenset({STABLE_ID}))


#: The two preimage key sets only ``grant_identity.py`` may spell as a dict
#: literal. A re-spelled inline preimage is exactly how the fork shipped.
_PREIMAGE_KEY_SETS = (
    frozenset({"generation", "stable_id", "stage", "reason", "attempts", "granted_by"}),
    frozenset({"stable_id", "stage", "reason", "grant"}),
)

_CRAWLER_ROOT = Path(__file__).parents[1]


def _production_modules() -> Iterator[Path]:
    """Yield every production crawler module except the identity owner."""

    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        relative = path.relative_to(_CRAWLER_ROOT)
        if relative.parts[0] in {"tests", "__pycache__"}:
            continue
        if relative.name == "grant_identity.py":
            continue
        yield path


@pytest.mark.smoke
def test_grant_identity_preimage_has_one_authority() -> None:
    """No production module re-spells a grant-identity preimage dict inline.

    The T1-08 fork existed because three modules each restated the preimage;
    this audit makes a silent fourth spelling impossible: any dict literal
    whose constant key set equals a grant preimage, outside
    ``grant_identity.py``, is a failure naming file and line.
    """

    violations: list[str] = []
    for path in _production_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if len(keys) != len(node.keys):
                continue
            if any(keys == preimage for preimage in _PREIMAGE_KEY_SETS):
                violations.append(f"{path.relative_to(_CRAWLER_ROOT)}:{node.lineno}")
    assert not violations, (
        f"grant-identity preimage re-spelled outside grant_identity.py: {violations}"
    )


@pytest.mark.heavy
def test_routed_grant_end_to_end_redrive_honors_the_granted_intent(
    tmp_path: Path,
) -> None:
    """Issue routed grant -> consume -> canonical append accepted -> intent honored.

    The full lifecycle that OPS-9 interdicted: an operator issues a tool grant
    with ``--intent`` against a reviewed failed terminal, the next drive
    durably consumes it, the superseding revision is ACCEPTED at the canonical
    append carrying the grant, and the re-driven work is routed to the granted
    environment intent.
    """

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    paths = _driver_paths(tmp_path, snapshot)
    assert (
        _driver(tmp_path, snapshot, checker=InaccurateChecker()).run().status
        == "terminal-partition-complete"
    )
    terminal = scan_jsonl(paths.ledgers.models)[-1]
    assert terminal["stable_id"] == stable_id
    assert terminal["status"]["code"] == "failed:accuracy-gate"

    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    grant = build_canonical_requeue_grant(
        canonical,
        stable_id=stable_id,
        stage="accuracy-gate",
        reason="human reviewed: reroute to the graph environment",
        attempts=1,
        granted_by="operator",
        target_intent="graph",
    )
    append_canonical_requeue_grant(canonical, grant)
    grant_id = str(grant["grant_id"])

    forward = FakeForward()
    result = _driver(tmp_path, snapshot, forward=forward).run()
    assert result.status == "complete"
    assert forward.calls == {stable_id: 1}

    revisions = [
        record for record in scan_jsonl(paths.ledgers.models) if record["stable_id"] == stable_id
    ]
    assert [record["status"]["code"] for record in revisions] == [
        "failed:accuracy-gate",
        "runs",
    ]
    assert revisions[-1]["budget"]["explicit_grants"] == [grant_id]

    consumed = [
        event
        for event in scan_jsonl(canonical_operational_ledger_path(paths.ledgers.models))
        if event["event_kind"] == OperationalEventKind.REQUEUE_GRANT_CONSUMED.value
    ]
    assert len(consumed) == 1
    assert consumed[0]["details"]["grant_id"] == grant_id
    assert consumed[0]["details"]["target_intent"] == "graph"

    granted_work_id = consumed[0]["details"]["new_work_id"]
    granted_attempts = [
        attempt
        for attempt in scan_jsonl(paths.ledgers.attempts)
        if attempt.get("work_id") == granted_work_id
    ]
    assert granted_attempts
    assert {(attempt.get("environment") or {}).get("family") for attempt in granted_attempts} == {
        "graph"
    }
