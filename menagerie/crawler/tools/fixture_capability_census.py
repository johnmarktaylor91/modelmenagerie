"""Fixture-fidelity capability census over the driver test compositions (AB-14).

The 200a launch shipped a dry-run composition missing ``provider_reconciler``
-- a capability the production driver requires fail-closed -- because nothing
enumerated what each test composition binds against the production
``DriverDependencies`` contract. This tool derives that census mechanically:

- The CONTRACT: the exact field list of the production ``DriverDependencies``
  dataclass, in declaration order.
- The COMPOSITIONS: the binding expression each test composition passes for
  each field, extracted from source via AST (``ast.unparse``, so formatting
  cannot fake a drift). Three compositions are registered: the unit fake
  builder (``test_slice_f_driver._driver``), the acceptance dry-run builder
  (``dry_run_support.build_dry_run_driver``), and the permanent rehearsal
  fixture (``rehearsal_support.fake_driver_factory``, which composes through
  ``_driver``).

``python -m menagerie.crawler.tools.fixture_capability_census write``
regenerates the committed manifest
(``menagerie/crawler/tests/fixture_capability_manifest.json``);
``test_fixture_capability_census.py`` fails whenever the derived census
drifts from the committed manifest, so ANY dependency-wiring change -- a new
contract field, a rebound capability, a dropped one -- must be consciously
reviewed in the same change. The manifest's ``plain_leg`` classification per
field is enforced by the runtime mutation-delete test in the same module.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from menagerie.crawler.driver_contracts import DriverDependencies

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TESTS = _REPO_ROOT / "menagerie" / "crawler" / "tests"
MANIFEST_PATH = _TESTS / "fixture_capability_manifest.json"
SCHEMA_VERSION = "menagerie.crawler.fixture-capability-manifest.v1"

#: The three registered driver test compositions: display name -> (source
#: file, function owning the composition, how DriverDependencies is built).
COMPOSITIONS = {
    "unit-fake-driver": (_TESTS / "test_slice_f_driver.py", "_driver"),
    "acceptance-dry-run": (_TESTS / "dry_run_support.py", "build_dry_run_driver"),
    "rehearsal-fixture": (_TESTS / "rehearsal_support.py", "fake_driver_factory"),
}


def contract_fields() -> tuple[str, ...]:
    """Return the production dependency contract's exact field order."""

    return tuple(field.name for field in dataclasses.fields(DriverDependencies))


def _function_node(path: Path, name: str) -> ast.FunctionDef:
    """Return the named top-level function's AST node."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise LookupError(f"{path.name} has no top-level function {name!r}")


def _dependency_call(function: ast.FunctionDef, callee: str) -> ast.Call:
    """Return the single ``callee(...)`` construction inside one function."""

    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == callee)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == callee)
        )
    ]
    if len(calls) != 1:
        raise LookupError(
            f"{function.name} constructs {callee} {len(calls)} times; expected exactly 1"
        )
    return calls[0]


def _binding_expressions(call: ast.Call, fields: tuple[str, ...]) -> dict[str, str]:
    """Map every contract field to the composition's binding expression.

    Positional arguments follow dataclass field order; a field never passed
    reads ``<dataclass-default>`` so contract growth is loudly visible in the
    census of EVERY composition.
    """

    bindings: dict[str, str] = {}
    for index, arg in enumerate(call.args):
        bindings[fields[index]] = ast.unparse(arg)
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("dependency construction must not use **kwargs")
        bindings[keyword.arg] = ast.unparse(keyword.value)
    return {field: bindings.get(field, "<dataclass-default>") for field in fields}


def derive_census() -> dict[str, Any]:
    """Derive the current capability census for all registered compositions."""

    fields = contract_fields()
    compositions: dict[str, dict[str, str]] = {}
    unit_path, unit_function = COMPOSITIONS["unit-fake-driver"]
    unit_call = _dependency_call(
        _function_node(unit_path, unit_function), "DriverDependencies"
    )
    compositions["unit-fake-driver"] = _binding_expressions(unit_call, fields)

    dry_path, dry_function = COMPOSITIONS["acceptance-dry-run"]
    dry_call = _dependency_call(
        _function_node(dry_path, dry_function), "DriverDependencies"
    )
    compositions["acceptance-dry-run"] = _binding_expressions(dry_call, fields)

    # The rehearsal fixture composes THROUGH the unit builder: its census is
    # the ``_driver`` keyword set it overrides, resolved against the unit
    # composition for everything it leaves alone.
    rehearsal_path, rehearsal_function = COMPOSITIONS["rehearsal-fixture"]
    rehearsal_call = _dependency_call(
        _function_node(rehearsal_path, rehearsal_function), "_driver"
    )
    driver_param_to_field = {
        "author": "author",
        "checker": "checker",
        "forward": "forward",
        "environments": "environments",
        "notifier": "notifier",
        "boundary": "boundary_hook",
        "pause_scheduler": "usage_pause_scheduler",
        "sleeper": "sleeper",
        "provider_reconciler": "provider_reconciler",
    }
    rehearsal: dict[str, str] = {
        field: f"<unit-fake-driver:{expression}>"
        for field, expression in compositions["unit-fake-driver"].items()
    }
    for keyword in rehearsal_call.keywords:
        field = driver_param_to_field.get(keyword.arg or "")
        if field is not None:
            rehearsal[field] = ast.unparse(keyword.value)
    compositions["rehearsal-fixture"] = rehearsal

    return {
        "schema_version": SCHEMA_VERSION,
        "contract_fields": list(fields),
        "compositions": compositions,
    }


def main(argv: list[str]) -> int:
    """Write or print the derived census."""

    census = derive_census()
    if argv[1:] == ["write"]:
        committed = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        committed["contract_fields"] = census["contract_fields"]
        committed["compositions"] = census["compositions"]
        MANIFEST_PATH.write_text(
            json.dumps(committed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"fixture_capability_census: wrote {MANIFEST_PATH}")
        return 0
    print(json.dumps(census, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
