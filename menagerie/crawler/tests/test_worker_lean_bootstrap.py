"""Worker bootstrap must stay importable inside lean routed model environments.

Step-200 leg 7 (2026-08-20): every model that reached the runner died with a
uniform ``failed:runner missing-receipt`` because the worker subprocess crashed
at bootstrap -- ``worker -> authority -> host_capacity -> recordio -> schema``
reached a module-level ``import jsonschema`` and the routed model environments
are deliberately sealed without driver-side dependencies. The wave that added
the chain was green on every driver-side suite because nothing imported the
worker graph under a lean interpreter.

These pins hold both halves of the fix permanently:

* the worker's import graph must succeed with ``jsonschema``/``referencing``
  blocked (routed-env-clean bootstrap), and
* payload validation without the engine must raise the TYPED fail-closed
  refusal -- absence may never read as a skipped check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_BLOCKED_BOOTSTRAP = """
import builtins
import sys

_BLOCKED = ("jsonschema", "referencing")
_real_import = builtins.__import__


def _blocking_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in _BLOCKED:
        raise ModuleNotFoundError(f"No module named {root!r}")
    return _real_import(name, *args, **kwargs)


for _name in list(sys.modules):
    if _name.split(".")[0] in _BLOCKED:
        del sys.modules[_name]
builtins.__import__ = _blocking_import

# The exact chain that killed leg-7 workers, plus the worker module itself.
import menagerie.crawler.schema
import menagerie.crawler.recordio
import menagerie.crawler.host_capacity
import menagerie.crawler.authority
import menagerie.crawler.worker

from menagerie.crawler.schema import PayloadValidationError, validate_payload

try:
    validate_payload({"schema_version": "model-v3"})
except PayloadValidationError as exc:
    if "schema-engine-unavailable" not in str(exc):
        raise SystemExit(f"WRONG-REFUSAL: {exc}")
else:
    raise SystemExit("VALIDATION-SILENTLY-PASSED-WITHOUT-ENGINE")

print("LEAN-BOOTSTRAP-OK")
"""


def test_worker_import_graph_needs_no_schema_engine() -> None:
    """The worker's bootstrap chain imports with jsonschema/referencing blocked."""

    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_BOOTSTRAP],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        "worker bootstrap graph is no longer routed-env-clean "
        f"(the leg-7 missing-receipt class):\n{result.stderr[-2000:]}"
    )
    assert "LEAN-BOOTSTRAP-OK" in result.stdout


_CLOSURE_PROBE = """
import sys

import menagerie.crawler.worker

closure = sorted(name for name in sys.modules if name.startswith("menagerie"))
forbidden = [
    name
    for name in (
        "jsonschema",
        "referencing",
        "menagerie.crawler.authority",
        "menagerie.crawler.host_capacity",
        "menagerie.crawler.recordio",
        "menagerie.crawler.schema",
    )
    if name in sys.modules
]
print("CLOSURE=" + ",".join(closure))
if forbidden:
    raise SystemExit("FORBIDDEN-IN-CLOSURE: " + ", ".join(forbidden))
print("CLOSURE-DIET-OK")
"""


def _worker_closure_modules() -> list[str]:
    """Import the worker in a clean subprocess and return its menagerie closure."""

    result = subprocess.run(
        [sys.executable, "-c", _CLOSURE_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        "worker bootstrap closure regressed (DX-3/RE-3 diet: the driver-side "
        f"authority chain re-entered the routed-env import graph):\n{result.stderr[-2000:]}"
        f"\n{result.stdout[-500:]}"
    )
    assert "CLOSURE-DIET-OK" in result.stdout
    for line in result.stdout.splitlines():
        if line.startswith("CLOSURE="):
            return [name for name in line.removeprefix("CLOSURE=").split(",") if name]
    raise AssertionError("closure probe printed no CLOSURE line")


def test_worker_closure_stays_on_the_diet() -> None:
    """Importing ONLY the worker pulls no authority/recordio/schema/jsonschema.

    The leg-7 fix made the chain import-safe in lean envs; the DX-3 diet
    removes it from the worker closure outright (completion-line leaf +
    policy's parent-side lazy imports). This pin fails on the first re-added
    module-level edge from any closure module back into the driver tree.
    """

    closure = _worker_closure_modules()
    assert "menagerie.crawler.worker" in closure
    assert "menagerie.crawler.completion_line" in closure


def test_worker_closure_has_no_unguarded_strenum_import() -> None:
    """No worker-closure module may import ``enum.StrEnum`` outside a guard.

    ``from enum import StrEnum`` at module level is python>=3.11; the oldest
    routed-env interpreter is the legacy-torch lock's py3.10, so an unguarded
    import is a deterministic bootstrap death for every member routed there
    (the 2026-08-22 recurrence, DX-3/RE-3). A guarded import (inside ``try``)
    falling back to ``constants.StrEnum`` is the sanctioned spelling.
    """

    import ast

    crawler_root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for module_name in _worker_closure_modules():
        if not module_name.startswith("menagerie.crawler."):
            continue
        leaf = module_name.rsplit(".", 1)[1]
        path = crawler_root / f"{leaf}.py"
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guarded: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for child in ast.walk(node):
                    guarded.add(getattr(child, "lineno", -1))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "enum"
                and any(alias.name == "StrEnum" for alias in node.names)
                and node.lineno not in guarded
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], (
        "unguarded `from enum import StrEnum` in worker-closure modules "
        f"(kills every legacy-torch member at bootstrap): {offenders}"
    )


def test_completion_line_reexport_is_the_same_object() -> None:
    """The authority re-export and the leaf are ONE function -- no protocol fork."""

    from menagerie.crawler import authority, completion_line

    assert (
        authority.completion_line_for_raw_award_receipt
        is completion_line.completion_line_for_raw_award_receipt
    )
    assert authority.raw_award_receipt_sha256 is completion_line.raw_award_receipt_sha256
    assert authority.AuthorityDerivationError is completion_line.AuthorityDerivationError


@pytest.mark.round21_macos_real
def test_routed_env_interpreter_imports_worker_closure(
    request: pytest.FixtureRequest,
) -> None:
    """D-46/OS-2: the committed routed-env interpreter imports the worker closure.

    The leg-7 jsonschema deaths and the 8/22 StrEnum deaths shared one cause:
    the worker import graph was never exercised under a ROUTED interpreter
    before models were charged. This release-registry probe spawns the real
    lock-built prefix python on the worker closure; red means the environment
    is unfit and no model may be charged in it.
    """

    import os

    if sys.platform != "darwin" or os.uname().machine != "arm64":
        pytest.skip("hosted Darwin arm64 release host is unavailable")
    fixture = request.getfixturevalue("real_environment_fixture")
    env_python = fixture.prefix / "bin" / "python"
    assert env_python.exists()
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        [str(env_python), "-c", _CLOSURE_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=env,
    )
    assert result.returncode == 0, (
        "routed-env interpreter cannot import the worker closure "
        f"({env_python}):\n{result.stderr[-2000:]}\n{result.stdout[-500:]}"
    )
    assert "CLOSURE-DIET-OK" in result.stdout


def test_validation_without_engine_is_fail_closed() -> None:
    """With the engine present, the typed refusal still guards absence exactly."""

    from menagerie.crawler import schema

    # Simulate absence without unloading the real engine: the guard consults
    # only the recorded import error.
    original = schema._SCHEMA_ENGINE_IMPORT_ERROR  # noqa: SLF001
    schema._SCHEMA_ENGINE_IMPORT_ERROR = ModuleNotFoundError(  # noqa: SLF001
        "No module named 'jsonschema'"
    )
    try:
        with pytest.raises(schema.PayloadValidationError, match="schema-engine-unavailable"):
            schema._require_schema_engine()  # noqa: SLF001
    finally:
        schema._SCHEMA_ENGINE_IMPORT_ERROR = original  # noqa: SLF001
