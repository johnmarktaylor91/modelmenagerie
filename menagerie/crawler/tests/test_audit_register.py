"""PM-6/W20 (+AB-1, AB-3): audit register, zero-GAP preflight arm, docket
guard, regression-pin registry, incident-pin revert census.

The audit register is the machine-readable failure-pattern catalog: every row
carries EXACTLY ONE closed-vocabulary disposition ({FIXED@sha | DEFERRED +
trigger | REJECTED + reason} for finding rows; {BLOCKED-ON-JMT-SIGNATURE |
DEAD-WITH-EVIDENCE} for item-resolution rows, per PLAN §4.2). A GAP is a row
lacking a lawful disposition; the launch preflight's ``audit-register`` arm
refuses launch while any exists, and machine-checks every FIXED row's pytest
tripwires against the committed smoke inventory lock (a falsely-FIXED row is
worse than an honest GAP). The regression-pin registry maps every code
failure class to collected tripwire nodes (covers-every-leg-death, PM-Q7/O);
the revert census (B3-XV-18) proves each recent production diff reddens the
blocking gate when reverted. The docket guard is P-07's mechanical closure.
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import pytest

from menagerie.crawler.tools.audit_register import (
    AUDIT_REGISTER_PATH,
    AUDIT_REGISTER_SCHEMA_VERSION,
    REGRESSION_PINS_PATH,
    AuditRegisterError,
    evaluate_register,
    evaluate_regression_pins,
    load_register,
    load_regression_pins,
    resolve_pytest_ref,
)
from menagerie.crawler.tools.docket_guard import scan_docket
from menagerie.crawler.tools.incident_pin_revert_census import (
    REVERT_CENSUS_ARTIFACT_PATH,
    classify_revert,
)
from menagerie.crawler.tools import collection_gate

pytestmark = pytest.mark.smoke

_REPO_ROOT = Path(__file__).resolve().parents[3]


@functools.lru_cache(maxsize=1)
def _corpus_universe() -> tuple[frozenset[str], frozenset[str]]:
    """Measure the live corpus once per session: (collected node ids, smoke node ids).

    The exact-inventory lock retired with RC-3; the pin web resolves against
    a fresh collection instead, so it can never be satisfied by a stale
    committed manifest.
    """

    report = collection_gate.collect(_REPO_ROOT, python=sys.executable)
    assert report.ok, report.failures
    smoke = collection_gate.collect(
        _REPO_ROOT, python=sys.executable, tracked_modules=report.tracked_modules, marker="smoke"
    )
    return frozenset(report.node_ids), frozenset(smoke.node_ids)


def _lock() -> tuple[frozenset[str], frozenset[str]]:
    return _corpus_universe()


def _register_row(row_id: str = "C-99", disposition: dict | None = None) -> dict:
    return {
        "row_id": row_id,
        "name": "synthetic",
        "kind": "finding",
        "disposition": disposition,
    }


def _register(rows: list[dict]) -> dict:
    return {
        "schema_version": AUDIT_REGISTER_SCHEMA_VERSION,
        "staleness": {
            "catalog_source": "synthetic",
            "refreshed_tree": "sha-test",
            "refreshed_at": "2026-08-25",
        },
        "rows": rows,
    }


# --- register schema + GAP semantics -----------------------------------------


def test_gap_row_refuses() -> None:
    collected, _ = _lock()
    census = evaluate_register(
        _register([_register_row(disposition=None)]),
        collected_nodes=collected,
        repo_root=_REPO_ROOT,
    )
    assert census["status"] == "failed"
    assert census["gaps"] == ["C-99"]


def test_disposition_outside_closed_vocabulary_refuses() -> None:
    collected, _ = _lock()
    census = evaluate_register(
        _register([_register_row(disposition={"kind": "SORT-OF-FIXED"})]),
        collected_nodes=collected,
        repo_root=_REPO_ROOT,
    )
    assert census["status"] == "failed"
    assert any("closed" in refusal for refusal in census["refusals"])


def test_fixed_row_with_unresolvable_tripwire_refuses() -> None:
    collected, _ = _lock()
    census = evaluate_register(
        _register(
            [
                _register_row(
                    disposition={
                        "kind": "FIXED",
                        "sha": "sha-test",
                        "tripwires": [
                            {
                                "kind": "pytest-node",
                                "ref": "menagerie/crawler/tests/test_zzz_missing.py::test_nope",
                            }
                        ],
                    }
                )
            ]
        ),
        collected_nodes=collected,
        repo_root=_REPO_ROOT,
    )
    assert census["status"] == "failed"
    assert any("test_zzz_missing" in refusal for refusal in census["refusals"])


def test_deferred_requires_trigger_and_rejected_requires_reason() -> None:
    collected, _ = _lock()
    census = evaluate_register(
        _register(
            [
                _register_row("P-90", {"kind": "DEFERRED"}),
                _register_row("P-91", {"kind": "REJECTED", "reason": "  "}),
            ]
        ),
        collected_nodes=collected,
        repo_root=_REPO_ROOT,
    )
    assert census["status"] == "failed"
    assert len(census["refusals"]) == 2


def test_item_resolution_rows_are_segregated_operationally_resolved() -> None:
    collected, _ = _lock()
    census = evaluate_register(
        _register(
            [
                {
                    "row_id": "W99",
                    "name": "synthetic item",
                    "kind": "item-resolution",
                    "disposition": {
                        "kind": "DEAD-WITH-EVIDENCE",
                        "census_entry": "dead-census-row-1",
                    },
                }
            ]
        ),
        collected_nodes=collected,
        repo_root=_REPO_ROOT,
    )
    assert census["status"] == "passed"
    assert census["operationally_resolved"] == ["W99"]


def test_process_only_rows_are_disclosed_never_counted_machine_tripwired() -> None:
    collected, _ = _lock()
    census = evaluate_register(
        _register(
            [
                _register_row(
                    "P-92",
                    {
                        "kind": "FIXED",
                        "sha": "sha-test",
                        "tripwires": [{"kind": "process", "ref": "some/offtree.md"}],
                    },
                )
            ]
        ),
        collected_nodes=collected,
        repo_root=_REPO_ROOT,
    )
    assert census["status"] == "passed"
    assert census["process_enforced"] == ["P-92"]


def test_resolve_pytest_ref_handles_class_param_and_module_refs() -> None:
    collected, _ = _lock()
    class_hosted = resolve_pytest_ref(
        "menagerie/crawler/tests/test_deadman_watchdog.py::test_stale_heartbeat_pages",
        collected,
    )
    assert class_hosted, "class-hosted function refs must resolve"
    module_ref = resolve_pytest_ref(
        "menagerie/crawler/tests/test_close_v2_contract.py", collected
    )
    assert len(module_ref) > 10
    assert resolve_pytest_ref("menagerie/crawler/tests/test_nope.py::test_x", collected) == []


# --- the committed register is current, lawful, and GAP-free ------------------


def test_committed_register_is_gap_free_and_machine_checked() -> None:
    register = load_register(AUDIT_REGISTER_PATH)
    collected, _ = _lock()
    census = evaluate_register(register, collected_nodes=collected, repo_root=_REPO_ROOT)
    assert census["gaps"] == []
    assert census["refusals"] == []
    assert census["status"] == "passed"
    assert census["rows_total"] == 44
    # Deferred/rejected/process rows stay DISCLOSED, never silently green.
    assert set(census["deferred"]) == {"C-24", "C-29", "P-03"}
    assert census["rejected"] == ["P-04"]
    assert set(census["process_enforced"]) == {"P-06", "P-11"}


def test_legacy_or_malformed_register_refuses(tmp_path: Path) -> None:
    stale = tmp_path / "register.json"
    stale.write_text(json.dumps({"schema_version": "v0", "rows": []}), encoding="utf-8")
    with pytest.raises(AuditRegisterError, match="schema"):
        load_register(stale)


# --- regression pins ----------------------------------------------------------


def test_regression_pins_cover_every_leg_death_class() -> None:
    pins = load_regression_pins(REGRESSION_PINS_PATH)
    register = load_register(AUDIT_REGISTER_PATH)
    collected, smoke = _lock()
    census = evaluate_regression_pins(
        pins, register, collected_nodes=collected, smoke_nodes=smoke
    )
    assert census["status"] == "passed"
    assert census["uncovered_incidents"] == []
    assert census["unresolvable_nodes"] == []
    # Every code failure class C-01..C-30 is pinned.
    assert len(census["covered_incidents"]) == 30


def test_regression_pins_never_lower_against_register() -> None:
    """Dropping a FIXED code row's tripwire from the pin registry refuses."""

    pins = load_regression_pins(REGRESSION_PINS_PATH)
    register = load_register(AUDIT_REGISTER_PATH)
    collected, smoke = _lock()
    lowered = json.loads(json.dumps(pins))
    lowered["pins"]["C-01"]["nodes"] = lowered["pins"]["C-01"]["nodes"][:1]
    census = evaluate_regression_pins(
        lowered, register, collected_nodes=collected, smoke_nodes=smoke
    )
    assert census["status"] == "failed"
    assert any("C-01" in refusal for refusal in census["refusals"])


def test_regression_pins_unknown_node_refuses() -> None:
    pins = load_regression_pins(REGRESSION_PINS_PATH)
    register = load_register(AUDIT_REGISTER_PATH)
    collected, smoke = _lock()
    widened = json.loads(json.dumps(pins))
    widened["pins"]["C-01"]["nodes"].append("menagerie/crawler/tests/test_gone.py::test_x")
    census = evaluate_regression_pins(
        widened, register, collected_nodes=collected, smoke_nodes=smoke
    )
    assert census["status"] == "failed"
    assert census["unresolvable_nodes"] == ["menagerie/crawler/tests/test_gone.py::test_x"]


# --- docket guard (P-07) ------------------------------------------------------


def test_docket_guard_refuses_fill_markers() -> None:
    refusals = scan_docket(
        "| item | result |\n| D-1 | [FILL] |\n| D-2 | TBD |\n| D-3 | done |\n"
    )
    assert len(refusals) == 2
    assert all("line" in refusal for refusal in refusals)


def test_docket_guard_requires_pasted_gate_output(tmp_path: Path) -> None:
    from menagerie.crawler.tools.docket_guard import main as docket_main

    docket = tmp_path / "docket.md"
    docket.write_text("all done, suite pending\n", encoding="utf-8")
    assert docket_main(["--docket", str(docket), "--require-gate-output"]) == 1
    docket.write_text("gate:\n3750 passed in 100s\n", encoding="utf-8")
    assert docket_main(["--docket", str(docket), "--require-gate-output"]) == 0


def test_docket_guard_accepts_a_clean_docket() -> None:
    assert scan_docket("| D-1 | merged abc123, 12 passed |\n") == []


# --- revert census ------------------------------------------------------------


def test_classify_revert_ladder() -> None:
    """Inventory pins first, then pin nodes, then the full blocking gate."""

    assert (
        classify_revert(inventory_matches=False, pin_run_passed=None)["verdict"]
        == "pinned-by-inventory"
    )
    assert (
        classify_revert(inventory_matches=True, pin_run_passed=False)["verdict"]
        == "pinned-by-node"
    )
    assert (
        classify_revert(
            inventory_matches=True, pin_run_passed=True, blocking_gate_passed=False
        )["verdict"]
        == "pinned-by-blocking-gate"
    )
    assert (
        classify_revert(
            inventory_matches=True, pin_run_passed=True, blocking_gate_passed=True
        )["verdict"]
        == "UNPINNED-GREEN"
    )


def test_committed_revert_census_artifact_covers_thirty_and_is_clean() -> None:
    """B3-XV-18 acceptance: the committed artifact exists, covers the last ~30
    production diffs, and names ZERO unpinned-green reverts."""

    payload = json.loads(REVERT_CENSUS_ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"].startswith("menagerie.crawler.incident-pin-revert-census")
    assert len(payload["rows"]) >= 30
    unpinned = [row for row in payload["rows"] if row["verdict"] == "UNPINNED-GREEN"]
    assert unpinned == [], f"unpinned production diffs (block launch until pinned): {unpinned}"
