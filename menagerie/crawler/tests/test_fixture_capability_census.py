"""AB-14: fixture-fidelity capability census enforcement.

Three tripwires over ``fixture_capability_manifest.json``:

1. CENSUS DRIFT -- the derived contract-field list and per-composition binding
   expressions must equal the committed manifest, so a new production
   dependency capability, a rebound fixture, or a silently dropped one cannot
   land without the manifest changing in the same reviewed diff. This is the
   instrument that would have caught the launch composition missing
   ``provider_reconciler``.
2. MUTATION-DELETE -- poisoning each capability in a plain unit fake leg must
   produce EXACTLY the committed behavior class: a loud raise, a classified
   non-complete leg, or (for capabilities a plain leg never consults) an
   untouched complete leg. The poison records consultation, so a dormant row
   that quietly starts firing -- or a firing row that quietly goes dormant --
   turns RED instead of drifting.
3. DORMANT COVERAGE -- every dormant capability must name a collectable
   rehearsal that fires it; those rehearsals are the only evidence the
   capability works at all.
"""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.crawler.tools.fixture_capability_census import (
    MANIFEST_PATH,
    SCHEMA_VERSION,
    contract_fields,
    derive_census,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot

_TESTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _manifest() -> dict:
    """Load the committed capability manifest."""

    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    return payload


@pytest.mark.smoke
def test_capability_census_matches_the_committed_manifest() -> None:
    """Derived contract fields and composition bindings equal the manifest.

    Regenerate deliberately via
    ``python -m menagerie.crawler.tools.fixture_capability_census write``
    in the same change that rewires a composition or grows the contract.
    """

    manifest = _manifest()
    census = derive_census()
    assert census["contract_fields"] == manifest["contract_fields"]
    assert census["compositions"] == manifest["compositions"]


@pytest.mark.smoke
def test_every_contract_field_has_a_plain_leg_classification() -> None:
    """The mutation-behavior table is closed over the exact contract."""

    manifest = _manifest()
    assert set(manifest["plain_leg"]) == set(contract_fields())
    for field, row in manifest["plain_leg"].items():
        assert row["class"] in ("fires-loudly", "fires-classified", "dormant"), field
        if row["class"] == "dormant":
            assert row.get("fired_by"), f"dormant capability {field} names no rehearsal"


@pytest.mark.smoke
def test_dormant_capabilities_name_collectable_firing_rehearsals() -> None:
    """Each dormant capability's ``fired_by`` nodes exist as test functions."""

    for field, row in _manifest()["plain_leg"].items():
        for node_id in row.get("fired_by", ()):
            path_text, _, function = str(node_id).partition("::")
            function = function.split("[", 1)[0]
            test_path = _TESTS_ROOT / path_text
            assert test_path.is_file(), f"{field}: missing rehearsal module {path_text}"
            tree = ast.parse(test_path.read_text(encoding="utf-8"))
            names = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert function in names, (
                f"dormant capability {field} names a rehearsal that does not exist: {node_id}"
            )


class _PoisonCapability:
    """A capability double that fails loudly and records every consultation."""

    def __init__(self, name: str) -> None:
        """Bind the poisoned capability's name."""

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "consulted", False)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Record the consultation and refuse."""

        object.__setattr__(self, "consulted", True)
        raise RuntimeError(f"poisoned-capability-consulted:{self.name}")

    def __getattr__(self, attribute: str) -> object:
        """Record the consultation and refuse (protocol-shaped capabilities)."""

        if attribute.startswith("_"):
            raise AttributeError(attribute)
        object.__setattr__(self, "consulted", True)
        raise RuntimeError(f"poisoned-capability-consulted:{self.name}.{attribute}")


@pytest.mark.heavy
@pytest.mark.parametrize("field", contract_fields())
def test_mutation_deleting_each_capability_matches_its_committed_class(
    tmp_path: Path, field: str
) -> None:
    """Poisoning one capability produces exactly the committed behavior class.

    ``fires-loudly``: the run raises. ``fires-classified``: the capability was
    consulted and the leg ends non-complete (production classification
    absorbed the failure into typed model/leg outcomes -- meaningfully
    different from the baseline ``complete``). ``dormant``: the plain leg
    completes without ever consulting the capability. Any drift between the
    observed behavior and the committed class is a fixture-fidelity change
    that must be root-caused, never reclassified to pass.
    """

    expected = _manifest()["plain_leg"][field]["class"]
    snapshot = _snapshot(tmp_path, count=1)
    driver = _driver(tmp_path, snapshot)
    poison = _PoisonCapability(field)
    driver.dependencies = replace(driver.dependencies, **{field: poison})
    if expected == "fires-loudly":
        with pytest.raises(BaseException):
            driver.run()
        assert poison.consulted
        return
    result = driver.run()
    if expected == "fires-classified":
        assert poison.consulted, f"{field} was never consulted by a plain leg"
        assert result.status != "complete", (
            f"poisoning {field} left the leg indistinguishable from healthy"
        )
        return
    assert not poison.consulted, f"dormant capability {field} was consulted"
    assert result.status == "complete"
