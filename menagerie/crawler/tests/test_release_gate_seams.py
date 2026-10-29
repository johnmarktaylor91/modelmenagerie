"""Release-gate composition tripwires: floor x attestation x invocation.

The 200a-era latent bug: ``pytest_sessionfinish`` enforced the execution floor
(which mutates ``session.exitstatus``) and then composed the release
attestation from the STALE ``exitstatus`` parameter — armed the moment both
environment variables were co-set. These tests drive the real conftest seam
with both variables co-set and prove the attestation reads the post-hook
verdict, carries the floor verdict as an explicit conjunct, and binds the
producing invocation (nonce/PID/expiry) against stale reuse.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import json

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_NODE = "menagerie/crawler/tests/test_fixture_node.py::test_green"
_SKIPPED_NODE = "menagerie/crawler/tests/test_fixture_node.py::test_confined_skip"


def _fresh_conftest() -> ModuleType:
    """Import the crawler conftest under a private name (isolated globals)."""

    spec = importlib.util.spec_from_file_location(
        "ws11_release_gate_composition_conftest", _TESTS_DIR / "conftest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_session(lines: list[str] | None = None) -> SimpleNamespace:
    """Build the minimal session object the sessionfinish seam consumes.

    Pass ``lines`` to capture the floor's announce output (the census lines)
    through a fake terminal reporter.
    """

    reporter = None if lines is None else SimpleNamespace(write_line=lines.append)
    pluginmanager = SimpleNamespace(get_plugin=lambda name: reporter)
    return SimpleNamespace(
        config=SimpleNamespace(pluginmanager=pluginmanager),
        exitstatus=pytest.ExitCode.OK,
    )


def _isolate_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    conftest: ModuleType,
    waivers: list[dict[str, str]] | None = None,
) -> None:
    """Point the fresh conftest at a fixture waiver registry.

    The committed registry names real width-gated nodes; the fake sessions
    here select only fixture nodes, so without isolation the real rows would
    read as stale waivers and change what each seam test is proving.
    """

    registry = tmp_path / "smoke_execution_waivers.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "menagerie.crawler.smoke-execution-waivers.v1",
                "waivers": [] if waivers is None else waivers,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(conftest, "_EXECUTION_WAIVERS_PATH", registry)


def _arm_floor(
    monkeypatch: pytest.MonkeyPatch,
    conftest: ModuleType,
    *,
    selected: list[str],
    passed: set[str] | None = None,
    skipped: set[str] | None = None,
    failed: set[str] | None = None,
) -> None:
    """Arm the execution floor over an explicit fake outcome partition."""

    monkeypatch.setenv("MENAGERIE_SMOKE_EXECUTION_FLOOR", "1")
    monkeypatch.setattr(conftest, "_EXEC_SELECTED", selected)
    monkeypatch.setattr(conftest, "_EXEC_PASSED", passed or set())
    monkeypatch.setattr(conftest, "_EXEC_SKIPPED", skipped or set())
    monkeypatch.setattr(conftest, "_EXEC_XFAILED", set())
    monkeypatch.setattr(conftest, "_EXEC_FAILED", failed or set())


def _arm_release_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conftest: ModuleType
) -> Path:
    """Co-set BOTH env vars and make every non-floor conjunct green-shaped."""

    lock = tmp_path / "round19-osx-arm64.lock"
    lock.write_text("fixture-lock\n", encoding="utf-8")
    lock.with_suffix(".resolved.json").write_text("{}\n", encoding="utf-8")
    lock.with_suffix(".provenance.json").write_text("{}\n", encoding="utf-8")
    attestation_path = tmp_path / "attestation.json"
    _isolate_registry(tmp_path, monkeypatch, conftest)
    monkeypatch.setenv("MENAGERIE_RELEASE_ATTESTATION", str(attestation_path))
    monkeypatch.setenv("MENAGERIE_RELEASE_GATE", "1")
    monkeypatch.setenv("MENAGERIE_PLATFORM_LOCK", str(lock))
    monkeypatch.setenv("MENAGERIE_REAL_ENV_PREFIX", str(tmp_path))
    monkeypatch.setenv("MENAGERIE_SMOKE_EXECUTION_FLOOR", "1")
    target = "osx-arm64"
    monkeypatch.setattr(conftest, "_round21_release_nodes", lambda _target: (_NODE,))
    monkeypatch.setattr(conftest, "_round21_release_commit", lambda: "f" * 40)
    monkeypatch.setattr(conftest, "_round21_other_platform_expectations", lambda _t: {})
    monkeypatch.setattr(conftest, "_ROUND21_RELEASE_CONTENT_DIGEST", "sha256:" + "0" * 64)
    monkeypatch.setattr(conftest, "_ROUND21_RELEASE_PROBE_RESULTS", ())
    for name, value in (
        ("_ROUND21_RELEASE_COLLECTED", {target: {_NODE}, "linux-64": set()}),
        ("_ROUND21_RELEASE_PASSED", {target: {_NODE}, "linux-64": set()}),
        ("_ROUND21_RELEASE_SKIPPED", {target: set(), "linux-64": set()}),
        ("_ROUND21_RELEASE_XFAILED", {target: set(), "linux-64": set()}),
        ("_ROUND21_RELEASE_FAILED", {target: set(), "linux-64": set()}),
    ):
        monkeypatch.setattr(conftest, name, value)
    # Floor state: one selected node that ran and passed.
    monkeypatch.setattr(conftest, "_EXEC_SELECTED", [_NODE])
    monkeypatch.setattr(conftest, "_EXEC_PASSED", {_NODE})
    monkeypatch.setattr(conftest, "_EXEC_SKIPPED", set())
    monkeypatch.setattr(conftest, "_EXEC_XFAILED", set())
    monkeypatch.setattr(conftest, "_EXEC_FAILED", set())
    return attestation_path


@pytest.mark.smoke
def test_deliberate_unwaived_skip_fails_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE NAMED TRIPWIRE: an unwaived skip under the armed floor fails closed."""

    conftest = _fresh_conftest()
    _isolate_registry(tmp_path, monkeypatch, conftest)
    _arm_floor(
        monkeypatch,
        conftest,
        selected=[_NODE, _SKIPPED_NODE],
        passed={_NODE},
        skipped={_SKIPPED_NODE},
    )
    session = _fake_session()
    verdict = conftest._enforce_smoke_execution_floor(session, pytest.ExitCode.OK)
    assert verdict is False
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    # The clean arm returns True and leaves the exit status alone.
    monkeypatch.setattr(conftest, "_EXEC_SELECTED", [_NODE])
    monkeypatch.setattr(conftest, "_EXEC_SKIPPED", set())
    session = _fake_session()
    assert conftest._enforce_smoke_execution_floor(session, pytest.ExitCode.OK) is True
    assert session.exitstatus == pytest.ExitCode.OK
    # Unarmed floor abstains (None), never blesses.
    monkeypatch.delenv("MENAGERIE_SMOKE_EXECUTION_FLOOR")
    assert conftest._enforce_smoke_execution_floor(_fake_session(), 0) is None


@pytest.mark.smoke
def test_attestation_reads_posthook_exitstatus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Co-set both env vars: a floor-red session can never attest 'passed'.

    Every non-floor conjunct is green-shaped and the stale ``exitstatus``
    parameter says OK — under the pre-fix composition the attestation would
    read 'passed'. It must read 'unmet-release-gate' from the post-hook
    session state and the explicit floor conjunct.
    """

    conftest = _fresh_conftest()
    attestation_path = _arm_release_environment(tmp_path, monkeypatch, conftest)
    # Make the floor RED: an unwaived skip joins the selection.
    monkeypatch.setattr(conftest, "_EXEC_SELECTED", [_NODE, _SKIPPED_NODE])
    monkeypatch.setattr(conftest, "_EXEC_SKIPPED", {_SKIPPED_NODE})
    session = _fake_session()
    conftest.pytest_sessionfinish(session, pytest.ExitCode.OK)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert attestation["status"] == "unmet-release-gate"
    assert attestation["execution_floor"] == {"armed": True, "ok": False}
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    # Control arm: the same composition with a green floor attests passed —
    # proving the red arm above failed BECAUSE of the floor conjuncts.
    conftest2 = _fresh_conftest()
    attestation_path = _arm_release_environment(tmp_path, monkeypatch, conftest2)
    session = _fake_session()
    conftest2.pytest_sessionfinish(session, pytest.ExitCode.OK)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert attestation["status"] == "passed"
    assert attestation["execution_floor"] == {"armed": True, "ok": True}
    assert session.exitstatus == pytest.ExitCode.OK


@pytest.mark.smoke
def test_attestation_invocation_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The attestation binds its producing invocation: nonce echo, PID, expiry."""

    conftest = _fresh_conftest()
    attestation_path = _arm_release_environment(tmp_path, monkeypatch, conftest)
    monkeypatch.setenv("MENAGERIE_RELEASE_ATTESTATION_NONCE", "ws11-fixture-nonce")
    session = _fake_session()
    conftest.pytest_sessionfinish(session, pytest.ExitCode.OK)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    invocation = attestation["invocation"]
    assert invocation["nonce"] == "ws11-fixture-nonce"
    assert invocation["pid"] == os.getpid()
    assert invocation["expires_at"] > invocation["minted_at"]


# --------------------------------------------------------------------------
# Four-part pins for the no-consumer-at-production-width waiver code
# (widened 2026-08-29). Each test drives the real floor seam with a fixture
# registry; the last one recomputes the COMMITTED registry's eligibility
# proof from the live width constant.
# --------------------------------------------------------------------------

_WIDTH_WAIVER = {
    "node_id": _SKIPPED_NODE,
    "reason_code": "no-consumer-at-production-width",
    "detail": "fixture: the adjudicated shape is unconstructible at width 1",
}


@pytest.mark.smoke
def test_waived_skip_disclosed_and_never_counted_as_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parts 1 and 4: a registered skip is censused, segregated, never passed."""

    conftest = _fresh_conftest()
    _isolate_registry(tmp_path, monkeypatch, conftest, waivers=[_WIDTH_WAIVER])
    _arm_floor(
        monkeypatch,
        conftest,
        selected=[_NODE, _SKIPPED_NODE],
        passed={_NODE},
        skipped={_SKIPPED_NODE},
    )
    lines: list[str] = []
    session = _fake_session(lines)
    assert conftest._enforce_smoke_execution_floor(session, pytest.ExitCode.OK) is True
    assert session.exitstatus == pytest.ExitCode.OK
    census = [line for line in lines if "NOT counted as passed" in line]
    assert len(census) == 1
    assert _SKIPPED_NODE in census[0]
    assert "[no-consumer-at-production-width]" in census[0]
    # Segregated: the waived node never joins the pass count.
    assert any("1 passed of 2 selected (1 waived, 0 failed)" in line for line in lines)


@pytest.mark.smoke
def test_reason_code_outside_closed_vocabulary_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Part 2: the strict loader refuses unknown codes, and the floor fails closed."""

    conftest = _fresh_conftest()
    forged = dict(_WIDTH_WAIVER, reason_code="width-is-basically-one")
    _isolate_registry(tmp_path, monkeypatch, conftest, waivers=[forged])
    _arm_floor(
        monkeypatch,
        conftest,
        selected=[_NODE, _SKIPPED_NODE],
        passed={_NODE},
        skipped={_SKIPPED_NODE},
    )
    lines: list[str] = []
    session = _fake_session(lines)
    assert conftest._enforce_smoke_execution_floor(session, pytest.ExitCode.OK) is False
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("FAIL-CLOSED" in line for line in lines)


@pytest.mark.smoke
def test_waiver_over_running_node_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A width re-arm revives the node: the now-stale row REFUSES, never blesses.

    Part 3 recomputes eligibility from the live session every run, and part
    2's exactness demands the registry row disappear in its own reviewed
    change once the adjudicated behavior exists again — a waived node that
    RAN (here: PASSED) is a floor violation, not an inert leftover.
    """

    conftest = _fresh_conftest()
    _isolate_registry(tmp_path, monkeypatch, conftest, waivers=[dict(_WIDTH_WAIVER, node_id=_NODE)])
    _arm_floor(monkeypatch, conftest, selected=[_NODE], passed={_NODE})
    lines: list[str] = []
    session = _fake_session(lines)
    assert conftest._enforce_smoke_execution_floor(session, pytest.ExitCode.OK) is False
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
    assert any("prune the registry row" in line and _NODE in line for line in lines)


@pytest.mark.smoke
def test_waived_node_that_fails_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Part 1: a waiver grants nothing positive — a registered FAILURE stays red."""

    conftest = _fresh_conftest()
    _isolate_registry(tmp_path, monkeypatch, conftest, waivers=[_WIDTH_WAIVER])
    _arm_floor(
        monkeypatch,
        conftest,
        selected=[_NODE, _SKIPPED_NODE],
        passed={_NODE},
        failed={_SKIPPED_NODE},
    )
    session = _fake_session()
    assert conftest._enforce_smoke_execution_floor(session, pytest.ExitCode.OK) is False
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED


@pytest.mark.smoke
def test_committed_width_waivers_match_the_live_width_proof() -> None:
    """The COMMITTED registry rows, recomputed against the live width constant.

    Pins three things: the strict loader accepts the committed file, the
    no-consumer-at-production-width rows are EXACTLY the three width-gated
    nodes (part 2 exactness — silent growth fails here), and each row's
    eligibility proof still holds at the live METADATA_BATCH_MAX (a width
    bump fails here with pruning instructions, alongside the floor's own
    runtime refusal once the revived nodes run).
    """

    from menagerie.crawler.constants import METADATA_BATCH_MAX

    conftest = _fresh_conftest()
    waivers = conftest._load_execution_waivers()
    width_rows = {
        node for node, code in waivers.items() if code == "no-consumer-at-production-width"
    }
    golden = "menagerie/crawler/tests/test_checker_envelope_golden.py"
    scoping = "menagerie/crawler/tests/test_checker_lane_failure_scoping.py"
    assert width_rows == {
        f"{golden}::test_golden_cap_width_batch_routes_per_item_not_per_batch",
        f"{golden}::test_golden_mixed_batch_routes_each_item_independently",
        f"{scoping}::test_healthy_checker_after_downshift_gates_every_member",
    }
    assert waivers.keys() == width_rows, "only the three width rows are committed"
    assert METADATA_BATCH_MAX < 2, (
        "width re-armed: the two golden-batch tests run again — prune their "
        "waiver rows in this reviewed change"
    )
    assert METADATA_BATCH_MAX < 4, (
        "width re-armed: the downshift-ladder test runs again — prune its "
        "waiver row in this reviewed change"
    )
