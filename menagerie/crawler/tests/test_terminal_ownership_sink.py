"""DX-1/RE-1: terminal ownership is enforced at the mint sink, not only at arms.

200a minted 19 false ``failed:runner`` and 3 false ``failed:resource`` verdicts
because four ``_terminalize`` call sites classified host/machinery faults
correctly and then terminalized anyway (m12446: a machinery internal-error via
the bounded repair path settled a model's head). The rule is now structural:

* the SINK (``driver._terminalize``) refuses every ``failed:*`` disposition
  whose classified owner is in ``NEVER_MODEL_TERMINAL_OWNERS`` -- a fifth,
  unguarded call site cannot leak a false verdict;
* the four synthesis-confirmed catch arms consult the owner and park loudly
  BEFORE reaching the sink (the sink event's ``sink_guard`` stays ``false`` on
  their parks, so a ``true`` is always a defect signal);
* every leg close runs the standing ledger-scan clause: no terminal head with
  a host- or machinery-owned reason in the leg's own append window.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.constants import EnvironmentPhase
from menagerie.crawler.driver_contracts import WorkItem
from menagerie.crawler.routing import IntentRoute
from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    _non_model_terminal_error,
)
from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
from menagerie.crawler.scheduler_parks import (
    load_scheduler_quarantine_rows,
    scheduler_quarantine_path,
)
from menagerie.crawler.schema import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.terminal_ownership import (
    non_model_owned_failed_heads,
    non_model_owned_failed_heads_in_window,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    DetectedModeRepairAuthor,
    ForwardScript,
    ScriptedForward,
    _driver,
    _paths,
    _snapshot,
)

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]


def _parked_events(operational_rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return the non-model-fault park events from an operational ledger scan."""

    return [
        event
        for event in operational_rows
        if event.get("details", {}).get("disposition") == "non-model-fault-parked"
    ]


def test_m12446_machinery_repair_fault_parks_and_never_terminalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machinery internal-error escaping the repair path parks; no head mints.

    The exact m12446 mechanism: the worker's dual-mode observation triggers the
    bounded ``run_modes`` repair, the repair MACHINERY dies on its own defect
    (not an author answer, not host weather the transport ladder absorbs), and
    the blanket arm classifies ``internal-error``/machinery. 200a terminalized
    ``failed:runner`` on this; the classified owner now parks it.
    """

    from menagerie.crawler.driver import CrawlerDriver

    def _machinery_fault(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic repair machinery defect (m12446 shape)")

    monkeypatch.setattr(
        CrawlerDriver, "_repair_author_for_detected_modes", _machinery_fault, raising=True
    )
    snapshot = _snapshot(tmp_path, count=1)
    expanded_id = snapshot.items[0].stable_id
    result = _driver(
        tmp_path,
        snapshot,
        author=DetectedModeRepairAuthor(),
        forward=ScriptedForward(ForwardScript(expanded_id=expanded_id)),
    ).run()

    paths = _paths(tmp_path, snapshot)
    # The member parks as machine-deferred: no verdict about the model exists.
    assert result.status == "complete-with-deferrals"
    assert scan_jsonl(paths.ledgers.models) == []
    records_root = paths.ledgers.models.parent.parent
    quarantines = load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)])
    assert {row["stable_id"] for row in quarantines} == {expanded_id}
    assert {row["scheduler_quarantine"]["reason_code"] for row in quarantines} == {
        "non-model-fault"
    }
    parked = _parked_events(scan_jsonl(paths.operational_ledger))
    assert {event["details"]["stable_id"] for event in parked} == {expanded_id}
    assert {event["details"]["owner"] for event in parked} == {"machinery"}
    # The classification arm parked this, not the sink backstop.
    assert {event["details"]["sink_guard"] for event in parked} == {False}
    # The typed evidence stays on the ledger: the fault is recorded, not hidden.
    machinery_attempts = [
        attempt
        for attempt in scan_jsonl(paths.ledgers.attempts)
        if isinstance(attempt.get("error"), Mapping)
        and attempt["error"].get("reason_code") == "internal-error"
    ]
    assert machinery_attempts, "the machinery fault must persist as attempt evidence"
    assert {
        attempt["error"]["details"]["owner"] for attempt in machinery_attempts
    } == {"machinery"}


def test_sink_refuses_machinery_owned_failed_terminal_directly(tmp_path: Path) -> None:
    """``_terminalize`` itself parks a machinery-owned ``failed:*`` disposition.

    ``reducer`` and ``state`` are deliberately unusable sentinels: a sink that
    reaches any bookkeeping with this disposition is the defect this guard
    exists to stop, and the test then fails on the sentinel instead of minting.
    """

    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    intake = snapshot.items[0]
    item = WorkItem(
        intake=intake,
        route=IntentRoute(
            stable_id=intake.stable_id, intent="core", phase=EnvironmentPhase.PYTORCH
        ),
    )
    paths = _paths(tmp_path, snapshot)
    with JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        driver._terminalize(  # noqa: SLF001 -- the sink contract under test
            item,
            None,
            "failed:runner",
            "internal-error",
            "synthetic unguarded call site",
            (),
            None,  # type: ignore[arg-type]
            operational,
            {},
        )
    assert scan_jsonl(paths.ledgers.models) == []
    parked = _parked_events(scan_jsonl(paths.operational_ledger))
    assert len(parked) == 1
    assert parked[0]["details"]["sink_guard"] is True
    assert parked[0]["details"]["owner"] == "machinery"
    records_root = paths.ledgers.models.parent.parent
    quarantines = load_scheduler_quarantine_rows([scheduler_quarantine_path(records_root)])
    assert {row["stable_id"] for row in quarantines} == {item.stable_id}


@pytest.mark.smoke
def test_non_model_terminal_error_classification() -> None:
    """The sink-guard classifier binds owners exactly, with attempt-stamp priority."""

    # A machinery-owned reason refuses even with no attempt evidence.
    assert _non_model_terminal_error("failed:runner", "internal-error", (), None) is not None
    # A model-owned reason passes.
    assert _non_model_terminal_error("failed:runner", "protocol-violation", (), None) is None
    # A reason outside the worker vocabulary passes (accuracy-gate lane etc.).
    assert (
        _non_model_terminal_error(
            "failed:accuracy-gate", "terminal-disposition-unverifiable", (), None
        )
        is None
    )
    # Non-failed dispositions and reasonless terminals pass.
    assert _non_model_terminal_error("deferred:needs-cuda", "internal-error", (), None) is None
    assert _non_model_terminal_error("failed:runner", None, (), None) is None
    # The decisive attempt's explicit owner stamp overrides the map: a
    # model-owned reason with a host stamp refuses...
    stamped_host = {
        "result": "failed",
        "error": {
            "stage": "runner",
            "reason_code": "protocol-violation",
            "details": {"owner": "host"},
        },
    }
    assert (
        _non_model_terminal_error("failed:runner", "protocol-violation", (stamped_host,), None)
        is not None
    )
    # ...and a machinery-mapped reason explicitly stamped model-owned passes.
    stamped_model = {
        "result": "failed",
        "error": {
            "stage": "runner",
            "reason_code": "internal-error",
            "details": {"owner": "model"},
        },
    }
    assert (
        _non_model_terminal_error("failed:runner", "internal-error", (stamped_model,), None)
        is None
    )


@pytest.mark.smoke
def test_close_clause_flags_window_heads_and_ignores_carried_history(tmp_path: Path) -> None:
    """The per-leg clause fires on window rows only; pre-DX-1 history never bricks."""

    def _head(stable_id: str, kind: str, code: str, reason: Any) -> dict[str, Any]:
        return {
            "stable_id": stable_id,
            "record_revision": f"rev-{stable_id}",
            "status": {"kind": kind, "code": code, "reason_code": reason},
        }

    machinery = _head("m-machinery", "failed", "failed:runner", "internal-error")
    host = _head("m-host", "failed", "failed:runner", "worker-unreapable")
    model = _head("m-model", "failed", "failed:import", "import-exception")
    award = _head("m-award", "runs", "runs", None)
    findings = non_model_owned_failed_heads([machinery, host, model, award])
    assert {(finding["stable_id"], finding["owner"]) for finding in findings} == {
        ("m-machinery", "machinery"),
        ("m-host", "host"),
    }

    # Windowed scan: one carried (pre-window) machinery head, one appended in
    # this leg. Only the in-window head is a finding.
    records_root = tmp_path / "records"
    shard = records_root / "models" / "current-shard.jsonl"
    shard.parent.mkdir(parents=True)
    carried = _head("m-carried", "failed", "failed:runner", "internal-error")
    minted_here = _head("m-fresh", "failed", "failed:runner", "internal-error")
    with shard.open("w") as handle:
        for row in (carried, award, minted_here):
            handle.write(json.dumps(row) + "\n")
    baselines = {"models.shard": {"rows": 1}, "models.flat": {"rows": 0}}
    window_findings = non_model_owned_failed_heads_in_window(
        records_root, baselines=baselines
    )
    assert [finding["stable_id"] for finding in window_findings] == ["m-fresh"]
    assert window_findings[0]["row_index"] == 2
    # A window past the carried history with no fresh offender stays clean.
    clean = non_model_owned_failed_heads_in_window(
        records_root, baselines={"models.shard": {"rows": 3}, "models.flat": {"rows": 0}}
    )
    assert clean == ()


@pytest.mark.smoke
def test_close_clause_flags_rss_resource_heads_in_window(tmp_path: Path) -> None:
    """DX-6 clause: an in-window failed:resource/rss-cap head is a valve bypass."""

    from menagerie.crawler.terminal_ownership import unverified_resource_head_findings

    records_root = tmp_path / "records"
    shard = records_root / "models" / "current-shard.jsonl"
    shard.parent.mkdir(parents=True)
    rows = [
        {
            "stable_id": "m-rss",
            "record_revision": "rev-rss",
            "status": {"kind": "failed", "code": "failed:resource", "reason_code": "rss-cap"},
        },
        # Timeouts keep their model-owned resource terminal.
        {
            "stable_id": "m-timeout",
            "record_revision": "rev-timeout",
            "status": {"kind": "failed", "code": "failed:resource", "reason_code": "timeout"},
        },
    ]
    with shard.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    findings = unverified_resource_head_findings(
        records_root, baselines={"models.shard": {"rows": 0}, "models.flat": {"rows": 0}}
    )
    assert [finding["stable_id"] for finding in findings] == ["m-rss"]
    # Carried pre-window history never bricks a close it did not write.
    assert (
        unverified_resource_head_findings(
            records_root,
            baselines={"models.shard": {"rows": 2}, "models.flat": {"rows": 0}},
        )
        == ()
    )


@pytest.mark.smoke
def test_terminalize_sink_guard_dominates_every_append() -> None:
    """AST census: the sink consults the owner guard before any terminal append.

    The dominance property that makes a fifth unguarded call site harmless:
    ``_terminalize``'s body must evaluate ``_non_model_terminal_error`` (and be
    able to return without appending) BEFORE the first statement that can reach
    ``_append_terminal_revision``.
    """

    tree = ast.parse((_CRAWLER_ROOT / "driver.py").read_text())
    terminalize = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_terminalize"
    )
    guard_index: int | None = None
    append_index: int | None = None
    for index, statement in enumerate(terminalize.body):
        names = {
            child.id for child in ast.walk(statement) if isinstance(child, ast.Name)
        } | {
            child.attr for child in ast.walk(statement) if isinstance(child, ast.Attribute)
        }
        if guard_index is None and "_non_model_terminal_error" in names:
            guard_index = index
        if append_index is None and "_append_terminal_revision" in names:
            append_index = index
    assert guard_index is not None, "_terminalize lost its DX-1 sink guard"
    assert append_index is not None, "_terminalize no longer appends terminals?"
    assert guard_index < append_index, (
        "the DX-1 sink guard must dominate the terminal append"
    )


@pytest.mark.smoke
def test_terminal_append_caller_inventory_is_closed() -> None:
    """AST census: ``_append_terminal_revision`` has exactly the audited callers.

    A new caller is a new mint sink that bypasses the DX-1 guard; it must be
    added here in the same reviewed change that proves its own guard.
    """

    expected = {
        ("driver.py", "_terminalize"),
        ("driver.py", "_contain_terminal_finalization_refusal"),
        # The WS-2 host-capability head mints ``deferred:*`` only, behind its
        # own four-precondition eligibility check -- never a ``failed:*``.
        ("driver_admission.py", "_mint_host_capability_head"),
    }
    observed: set[tuple[str, str]] = set()
    for module in ("driver.py", "driver_admission.py", "driver_receipts.py"):
        tree = ast.parse((_CRAWLER_ROOT / module).read_text())
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_append_terminal_revision"
                ):
                    observed.add((module, function.name))
    # Nested functions attribute the call to the innermost enclosing def; the
    # census only needs a CLOSED inventory, so exact-match both directions.
    assert observed == expected


@pytest.mark.smoke
def test_never_model_terminal_owner_set_is_pinned() -> None:
    """The guard binds exactly host and machinery; widening is a reviewed change."""

    assert NEVER_MODEL_TERMINAL_OWNERS == frozenset({"host", "machinery"})
