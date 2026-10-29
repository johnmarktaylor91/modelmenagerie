"""D-19 (LC-6): every registered waiver has a PRODUCTION part-4 disclosure.

The four-part waiver doctrine's part 4 says a waived record is visible in a
derived census, segregated from records that passed the check. That sentence
was FALSE in production for the infra-terminal grant-release waiver:
``status.grant_release_census`` was correct, replay-derived -- and called
from exactly one place, a test. Calling a census from a test is exactly what
makes it LOOK disclosed while nothing an operator runs ever computes it.

This module is the reachability compiler: it recompiles, from source, the
binding between the CLOSED waiver registry (:data:`WAIVER_DISCLOSURES`) and
the production entry points that compute each disclosure. A future change
that deletes a production caller, registers a third waiver without a
disclosure, or points a binding at a callable that no longer exists fails
here with the waiver named.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from typing import Any

import pytest

from menagerie.crawler.status import (
    GRANT_RELEASE_CENSUS_CLASSES,
    WAIVER_DISCLOSURES,
    grant_release_census_from_ledgers,
)

pytestmark = pytest.mark.smoke


def _resolve(dotted: str) -> Any:
    """Resolve one dotted path through its longest importable module prefix."""

    parts = dotted.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            value: Any = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[split:]:
            value = getattr(value, attribute)
        return value
    raise AssertionError(f"no importable module prefix in {dotted!r}")


def _calls_in(source: str) -> set[str]:
    """Return the terminal names of every call expression in one source blob."""

    names: set[str] = set()
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def test_the_waiver_registry_is_closed_at_exactly_the_three_doctrine_instances() -> None:
    """The doctrine names exactly three registered waivers; a fourth fails here.

    Registering another four-part waiver is a deliberate reviewed change that
    must add its disclosure binding IN THE SAME CHANGE -- this set-equality pin
    is what makes forgetting that impossible. The third instance is W12's
    scheduler-quarantine machine drain (Wave-2 rotation batch, 2026-08-24):
    budget restoration is replay-derived from the quarantine ledger and
    disclosed through the segregated drain census.
    """

    assert {row.waiver_id for row in WAIVER_DISCLOSURES} == {
        "fidelity-gate-no-consumer",
        "infra-terminal-grant-release",
        "scheduler-quarantine-machine-drain",
    }
    assert len(WAIVER_DISCLOSURES) == 3


def test_every_registered_waiver_binding_resolves() -> None:
    """Every proof rule, census, and entry point in the registry is real code."""

    for row in WAIVER_DISCLOSURES:
        assert callable(_resolve(row.proof_rule)), row.waiver_id
        assert callable(_resolve(row.census_callable)), row.waiver_id
        assert row.production_entry_points, row.waiver_id
        for entry in row.production_entry_points:
            assert callable(_resolve(entry)), f"{row.waiver_id}: {entry}"


def test_every_registered_waiver_has_a_production_disclosure() -> None:
    """The census is CALLED from every named entry point, none of them a test.

    This is the arm that was red for a full rung: the grant-release census
    existed, validated, and was computed by nothing outside the suite.
    """

    for row in WAIVER_DISCLOSURES:
        census_name = row.census_callable.rsplit(".", 1)[-1]
        for entry in row.production_entry_points:
            module_path = entry.rsplit(".", 2)[0] if entry.count(".") > 1 else entry
            assert ".tests." not in entry and not module_path.endswith("tests"), (
                f"{row.waiver_id}: {entry} is not a production entry point"
            )
            source = inspect.getsource(_resolve(entry))
            assert census_name in _calls_in(source), (
                f"{row.waiver_id}: {entry} never computes {census_name} -- the "
                "waiver has no production disclosure"
            )


def test_the_census_keys_are_the_censuses_own_closed_vocabulary() -> None:
    """Each binding's keys are derived by its census, never invented here."""

    grant_release = next(
        row for row in WAIVER_DISCLOSURES if row.waiver_id == "infra-terminal-grant-release"
    )
    assert tuple(grant_release.census_keys) == tuple(GRANT_RELEASE_CENSUS_CLASSES)

    fidelity = next(
        row for row in WAIVER_DISCLOSURES if row.waiver_id == "fidelity-gate-no-consumer"
    )
    census_source = inspect.getsource(_resolve(fidelity.census_callable))
    for key in fidelity.census_keys:
        assert f'"{key}"' in census_source, key


def test_the_ledger_assembly_returns_every_census_class_on_an_empty_campaign(
    tmp_path,
) -> None:
    """The one production assembly is total: every class present, possibly empty."""

    from types import SimpleNamespace

    ledgers = SimpleNamespace(
        models=tmp_path / "models" / "current-shard.jsonl",
        attempts=tmp_path / "attempts" / "local.jsonl",
        gates=tmp_path / "gates" / "current-shard.jsonl",
    )
    census = grant_release_census_from_ledgers(ledgers)
    assert set(census) == set(GRANT_RELEASE_CENSUS_CLASSES)
    assert all(members == () for members in census.values())
