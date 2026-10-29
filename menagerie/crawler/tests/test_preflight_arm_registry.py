"""W4 + TC-7: the preflight arm registry, roots contract, and consumed verdict.

W4 (registry rule, O-M1): every preflight assertion lands through ONE registry
API — ``register_preflight_arm(name, armed, evidence, refusal_shape)`` — and
the report carries the armed-guard census, so a skipped precondition is a
machine-readable unarmed guard instead of a stderr warning. A new guard cannot
exist outside the census.

W4 (roots contract): every DECLARED root must exist as a directory on a
mounted, writable filesystem above the floor, proven by a
create+fsync+readback+unlink sentinel IN THE ACTING PROCESS plus a real
allocation probe (statvfs lies); a missing directory refuses — the preflight
never walks up to the nearest existing ancestor.

The TC-7 verdict FILE (nonce, freshness, single-use consumption) retired with
RC-3: the invocation preflight calls these arms as functions in process.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from menagerie.crawler.envs import DEFAULT_ENVS_ROOT
from menagerie.crawler.tests.test_launch_preflight import _ALLOWANCES
from menagerie.crawler.tools import launch_preflight
from menagerie.crawler.tools.launch_preflight import (
    EXPECTED_PREFLIGHT_ARMS,
    ROOTS_CONTRACT_UNMET,
    PreflightArmRegistry,
    evaluate_roots_contract,
    main,
)

pytestmark = pytest.mark.smoke


def _trivial_campaign(tmp_path: Path) -> tuple[Path, Path]:
    """Return a records/envs pair whose preflight passes trivially."""

    envs_root = tmp_path / "envs"
    shutil.copytree(DEFAULT_ENVS_ROOT, envs_root)
    records_root = tmp_path / "records"
    (records_root / "models").mkdir(parents=True)
    return records_root, envs_root


def _green_arguments(tmp_path: Path) -> list[str]:
    """Return main() arguments that pass on the trivial campaign."""

    records_root, envs_root = _trivial_campaign(tmp_path)
    return [
        "--records-root",
        str(records_root),
        "--envs-root",
        str(envs_root),
        # Fixture parameterization (host-independent): a modest floor, a
        # small real allocation probe, and a basetemp-local placement anchor;
        # production keeps the defaults.
        "--disk-floor-bytes",
        str(1024**3),
        "--allocation-probe-bytes",
        str(1024**2),
        "--placement-anchor",
        str(tmp_path),
        *_ALLOWANCES,
    ]


# --- registry API ---


def test_registry_records_arms_and_refuses_duplicates() -> None:
    """One arm name registers once; a second registration is a defect."""

    registry = PreflightArmRegistry()
    arm = registry.register_preflight_arm(
        "store-mount",
        armed=True,
        evidence={"env_root": "/store"},
        refusal_shape="store-mount-unavailable",
    )
    assert arm.name == "store-mount"
    assert registry.census() == [
        {
            "name": "store-mount",
            "armed": True,
            "evidence": {"env_root": "/store"},
            "refusal_shape": "store-mount-unavailable",
        }
    ]
    with pytest.raises(ValueError, match="already registered"):
        registry.register_preflight_arm(
            "store-mount", armed=True, evidence={}, refusal_shape="x"
        )


def test_main_report_carries_the_armed_guard_census(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every expected arm appears; omitted preconditions read armed=False."""

    exit_code = main(_green_arguments(tmp_path))
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    census = {row["name"]: row for row in report["armed_guard_census"]}
    assert set(census) == set(EXPECTED_PREFLIGHT_ARMS)
    # --env-root/--state-root were omitted: their guards are UNARMED, disclosed.
    assert census["store-mount"]["armed"] is False
    assert census["disk-floor"]["armed"] is False
    assert census["cas-completeness"]["armed"] is False
    # Always-on guards are armed.
    assert census["rotation-parks"]["armed"] is True
    assert census["roots-contract"]["armed"] is True


def test_every_evaluate_function_is_registered_in_the_census() -> None:
    """A new guard cannot exist outside the census (DX-62's consumer)."""

    import ast

    source = Path(launch_preflight.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    evaluators = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("evaluate_")
    }
    main_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    called_in_main = {
        node.func.id
        for node in ast.walk(main_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("evaluate_")
    }
    assert evaluators == called_in_main
    # The closed census names at least one arm per evaluator, so an evaluator
    # added without a census row fails the arm-count invariant.
    assert len(EXPECTED_PREFLIGHT_ARMS) >= len(evaluators)
    registrations = sum(
        1
        for node in ast.walk(main_node)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "register_preflight_arm")
            or (isinstance(node.func, ast.Name) and node.func.id == "register_preflight_arm")
        )
    )
    assert registrations >= len(EXPECTED_PREFLIGHT_ARMS)


# --- roots contract ---


def test_healthy_roots_pass_with_sentinel_and_device_evidence(tmp_path: Path) -> None:
    """Writable roots carry sentinel, device, and free-space evidence."""

    records = tmp_path / "records"
    scratch = tmp_path / "scratch"
    records.mkdir()
    scratch.mkdir()

    result = evaluate_roots_contract(
        {"records-root": records, "tmp-root": scratch, "state-root": None},
        minimum_free_bytes=1,
        allocation_probe_bytes=1024,
    )

    assert result["ok"] is True
    rows = {row["root"]: row for row in result["rows"]}
    assert rows["records-root"]["sentinel_ok"] is True
    assert isinstance(rows["records-root"]["st_dev"], int)
    assert rows["records-root"]["read_only"] is False
    assert rows["state-root"]["declared"] is False
    # Same device: exactly one real allocation probe ran.
    probes = [
        row.get("allocation_probe")
        for row in result["rows"]
        if row.get("declared", False)
    ]
    assert probes.count("probed") == 1
    assert probes.count("same-device-already-probed") == 1


def test_missing_root_refuses_instead_of_walking_up(tmp_path: Path) -> None:
    """An unmounted/missing TMPDIR refuses; the parent existing is irrelevant."""

    missing = tmp_path / "Volumes" / "gone" / "crawler-tmp"

    result = evaluate_roots_contract(
        {"tmp-root": missing},
        minimum_free_bytes=1,
        allocation_probe_bytes=0,
    )

    assert result["ok"] is False
    assert any("tmp-root" in refusal for refusal in result["refusals"])
    assert any(
        refusal.startswith(ROOTS_CONTRACT_UNMET) for refusal in result["refusals"]
    )


def test_read_only_root_fails_the_acting_process_sentinel(tmp_path: Path) -> None:
    """A read-only mount is caught by the sentinel, not trusted from flags."""

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    frozen.chmod(0o555)
    try:
        result = evaluate_roots_contract(
            {"records-root": frozen},
            minimum_free_bytes=1,
            allocation_probe_bytes=0,
        )
    finally:
        frozen.chmod(0o755)

    assert result["ok"] is False
    assert any("records-root" in refusal for refusal in result["refusals"])


def test_unmeetable_floor_refuses(tmp_path: Path) -> None:
    """A root below the free-bytes floor is a typed refusal."""

    root = tmp_path / "records"
    root.mkdir()

    result = evaluate_roots_contract(
        {"records-root": root},
        minimum_free_bytes=2**62,
        allocation_probe_bytes=0,
    )

    assert result["ok"] is False
    assert any("floor" in refusal for refusal in result["refusals"])


def test_main_refuses_on_a_missing_declared_tmp_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The roots contract is wired into the enforcing default path."""

    arguments = _green_arguments(tmp_path) + [
        "--tmp-root",
        str(tmp_path / "not-mounted" / "tmp"),
    ]
    exit_code = main(arguments)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert ROOTS_CONTRACT_UNMET in captured.err


# --- TC-7 consumed verdict ---
