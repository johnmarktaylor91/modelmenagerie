"""Every driver composition must carry every fail-closed driver dependency.

Launch-day incident (2026-08-19): the paid-session hardening made
``DriverDependencies.provider_reconciler`` a REQUIRED runtime capability -- admission
and settlement both raise when it is absent -- but only two of the three shipped
compositions were updated. The unit builder and the production CLI factory carried
it; the acceptance dry-run composition did not, so the gap could surface only inside
the round21 real-environment gate, on the ordinary paid path, late.

This module kills the CLASS rather than that instance. It DERIVES the fail-closed
dependency set from the driver's own source (a defaulted-``None`` field whose
``is None`` guard raises unconditionally) and then proves every composition supplies
every derived member. Adding a new fail-closed dependency, or a new production
composition, fails here -- which is the entire point, so the derived set is never
frozen into a snapshot that can silently go stale.

Scope note: the covered inventory is the SHARED compositions (the production CLI
factory plus the two shared test builders every end-to-end rehearsal routes
through) and every PRODUCTION construction site. Ad-hoc per-test
``DriverDependencies(...)`` literals are the owning test's business; a paid path
they cannot reach is not a launch risk.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import pytest

from menagerie.crawler import cli as cli_module
from menagerie.crawler.driver_contracts import DriverDependencies
from menagerie.crawler.env_lifecycle import LifecycleResult
from menagerie.crawler.tests import dry_run_support
from menagerie.crawler.tests import test_slice_f_driver as unit_composition

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]

#: Driver-side modules that consume ``DriverDependencies``. Every module whose name
#: starts with ``driver`` is scanned, so a new ``driver_*.py`` split is covered the
#: day it lands instead of quietly narrowing the derivation.
_DRIVER_SOURCE_GLOB = "driver*.py"

#: Composition inventory: label -> builder. Each builder returns the exact
#: ``DriverDependencies`` that composition hands the driver.
_COMPOSITION_LABELS = (
    "unit-builder:test_slice_f_driver._driver",
    "dry-run:dry_run_support.build_dry_run_driver",
    "production-cli:cli._default_driver_factory",
)


def _driver_sources() -> tuple[tuple[str, ast.Module], ...]:
    """Parse every driver-side module that reads injected dependencies.

    Returns
    -------
    tuple[tuple[str, ast.Module], ...]
        Module name and parsed tree for each ``driver*.py`` source.
    """

    parsed: list[tuple[str, ast.Module]] = []
    for path in sorted(_CRAWLER_ROOT.glob(_DRIVER_SOURCE_GLOB)):
        parsed.append((path.name, ast.parse(path.read_bytes(), filename=str(path))))
    return tuple(parsed)


def _defaulted_dependency_fields() -> tuple[str, ...]:
    """Return dependency fields that default to ``None``.

    These are the only at-risk fields: a mandatory field without a default is
    enforced by Python itself, and a field defaulting to a real callable cannot be
    silently absent.

    Returns
    -------
    tuple[str, ...]
        Field names whose declared default is ``None``.
    """

    return tuple(
        field.name for field in dataclasses.fields(DriverDependencies) if field.default is None
    )


def _dependency_attribute(node: ast.AST, fields: frozenset[str]) -> Optional[str]:
    """Return the dependency field named by ``self.dependencies.<field>``.

    Parameters
    ----------
    node:
        Candidate expression node.
    fields:
        At-risk dependency field names.

    Returns
    -------
    str | None
        Field name, or ``None`` when the expression is not a dependency read.
    """

    if not isinstance(node, ast.Attribute) or node.attr not in fields:
        return None
    owner = node.value
    if isinstance(owner, ast.Attribute) and owner.attr == "dependencies":
        return node.attr
    return None


def _dependency_reads(tree: ast.Module, fields: frozenset[str]) -> frozenset[str]:
    """Return every at-risk field the module reads off ``self.dependencies``.

    Parameters
    ----------
    tree, fields:
        Parsed module and at-risk field names.

    Returns
    -------
    frozenset[str]
        Fields observably consumed by this module.
    """

    return frozenset(
        field
        for node in ast.walk(tree)
        if (field := _dependency_attribute(node, fields)) is not None
    )


def _local_aliases(function: ast.AST, fields: frozenset[str]) -> Mapping[str, str]:
    """Map local names bound directly from a dependency read to their field.

    The production shape is ``reconciler = self.dependencies.provider_reconciler``
    followed by ``if reconciler is None: raise ...``, so the guard must be resolved
    through the alias.

    Parameters
    ----------
    function, fields:
        Function subtree and at-risk field names.

    Returns
    -------
    Mapping[str, str]
        Local variable name to dependency field name.
    """

    aliases: dict[str, str] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        field = _dependency_attribute(node.value, fields)
        if isinstance(target, ast.Name) and field is not None:
            aliases[target.id] = field
    return aliases


def _none_guard(
    test: ast.expr, fields: frozenset[str], aliases: Mapping[str, str]
) -> Optional[tuple[str, bool]]:
    """Return the field and polarity of a ``<dependency> is [not] None`` test.

    Parameters
    ----------
    test, fields, aliases:
        Condition expression, at-risk field names, and local alias map.

    Returns
    -------
    tuple[str, bool] | None
        Field name and whether the tested polarity is ``is None``.
    """

    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return None
    comparator = test.comparators[0]
    if not (isinstance(comparator, ast.Constant) and comparator.value is None):
        return None
    if isinstance(test.ops[0], ast.Is):
        polarity = True
    elif isinstance(test.ops[0], ast.IsNot):
        polarity = False
    else:
        return None
    field = _dependency_attribute(test.left, fields)
    if field is None and isinstance(test.left, ast.Name):
        field = aliases.get(test.left.id)
    if field is None:
        return None
    return field, polarity


def derive_fail_closed_dependencies() -> Mapping[str, tuple[str, ...]]:
    """Derive the dependencies that are REQUIRED at runtime, with their evidence.

    A field qualifies when the driver's own source refuses unconditionally on its
    absence: an ``is None`` branch (or the ``else`` of an ``is not None`` branch)
    whose body raises directly. An absence with a fallback -- ``sleeper``'s own
    shutdown-event wait, ``usage_pause_scheduler``'s inline ``WakeupManager`` -- is
    deliberately NOT derived, because those compositions are correct while absent.

    Returns
    -------
    Mapping[str, tuple[str, ...]]
        Required field name to sorted ``module:lineno`` refusal sites.
    """

    fields = frozenset(_defaulted_dependency_fields())
    evidence: dict[str, set[str]] = {}
    for name, tree in _driver_sources():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            aliases = _local_aliases(node, fields)
            for statement in ast.walk(node):
                if not isinstance(statement, ast.If):
                    continue
                guard = _none_guard(statement.test, fields, aliases)
                if guard is None:
                    continue
                field, absent_branch = guard
                branch = statement.body if absent_branch else statement.orelse
                if any(isinstance(child, ast.Raise) for child in branch):
                    evidence.setdefault(field, set()).add(f"{name}:{statement.lineno}")
    return {field: tuple(sorted(sites)) for field, sites in sorted(evidence.items())}


def _production_composition_sites() -> Mapping[str, int]:
    """Count non-test ``DriverDependencies(...)`` construction sites per module.

    Counted per FILE rather than pinned to a line number: unrelated edits above a
    composition would otherwise churn this tripwire, while a second composition in
    the same module -- the thing worth failing on -- is still caught.

    Returns
    -------
    Mapping[str, int]
        Module path relative to ``menagerie/crawler`` to construction-site count.
    """

    sites: list[str] = []
    for path in sorted(_CRAWLER_ROOT.rglob("*.py")):
        if "tests" in path.relative_to(_CRAWLER_ROOT).parts:
            continue
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "DriverDependencies"
            ):
                sites.append(str(path.relative_to(_CRAWLER_ROOT)))
    return {module: sites.count(module) for module in sorted(set(sites))}


class _StubEnvironments:
    """Stand in for the materialized-prefix lifecycle lane.

    The dry-run composition's environment lane needs a real conda prefix, which is
    irrelevant to -- and would gate -- a dependency-closure check. Only the
    composed ``DriverDependencies`` is under test here; the real prefix path is
    exercised by the acceptance dry-run itself.
    """

    def run(self, intent: Any, *, use: Any) -> LifecycleResult:
        """Refuse to run: this stub exists to be composed, never executed.

        Parameters
        ----------
        intent, use:
            Routed intent and driver callback, both unused.

        Raises
        ------
        AssertionError
            Always.
        """

        del intent, use
        raise AssertionError("the closure stub environment lane must never run")


def _unit_dependencies(tmp_path: Path) -> DriverDependencies:
    """Compose dependencies through the shared unit driver builder.

    Parameters
    ----------
    tmp_path:
        Per-test scratch root.

    Returns
    -------
    DriverDependencies
        Exactly what ``test_slice_f_driver._driver`` injects.
    """

    snapshot = unit_composition._snapshot(tmp_path, count=1)  # noqa: SLF001
    return unit_composition._driver(tmp_path, snapshot).dependencies  # noqa: SLF001


def _dry_run_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DriverDependencies:
    """Compose dependencies through the acceptance dry-run builder.

    Parameters
    ----------
    tmp_path, monkeypatch:
        Scratch root and patcher used to stub only the real-prefix lane.

    Returns
    -------
    DriverDependencies
        Exactly what ``dry_run_support.build_dry_run_driver`` injects.
    """

    monkeypatch.setattr(
        dry_run_support,
        "_materialized_environment",
        lambda _prefix: (_StubEnvironments(), None),
    )
    driver = dry_run_support.build_dry_run_driver(
        tmp_path / "repo",
        tmp_path / "campaign",
        tmp_path / "prefix",
        review_checkpoint_at=None,
        progress_milestones=(),
        run_id="closure-dry-run",
    )
    return driver.dependencies


def _production_cli_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DriverDependencies:
    """Compose dependencies through the production CLI driver factory.

    The factory is invoked exactly as ``run`` invokes it -- parsed argv, bound
    campaign, resolved lane commands -- and nothing is launched: only the returned
    composition is inspected.

    Parameters
    ----------
    tmp_path, monkeypatch:
        Scratch root and patcher for the mirror/certificate host preconditions.

    Returns
    -------
    DriverDependencies
        Exactly what ``cli._default_driver_factory`` injects.
    """

    from menagerie.crawler.tests.campaign_binding_support import bind_snapshot_to_campaign

    snapshot = unit_composition._snapshot(tmp_path, count=1)  # noqa: SLF001
    bind_snapshot_to_campaign(tmp_path, snapshot)
    commands: dict[str, Path] = {}
    for lane in ("author", "checker", "provider-reconcile", "environment", "notify"):
        path = tmp_path / "bin" / lane
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        commands[lane] = path
    for mirror in ("MENAGERIE_PUBLIC_MIRROR", "MENAGERIE_PRIVATE_MIRROR"):
        root = tmp_path / mirror.lower()
        root.mkdir()
        monkeypatch.setenv(mirror, str(root))
    args = cli_module.build_parser().parse_args(
        [
            "--repo-root",
            str(tmp_path),
            "run",
            "--intake",
            str(snapshot.root),
            "--run-id",
            "closure-production",
            "--author-command",
            str(commands["author"]),
            "--checker-command",
            str(commands["checker"]),
            "--provider-reconcile-command",
            str(commands["provider-reconcile"]),
            "--environment-command",
            str(commands["environment"]),
            "--notify-command",
            str(commands["notify"]),
        ]
    )
    assert isinstance(args, argparse.Namespace)
    return cli_module._default_driver_factory(args).dependencies  # noqa: SLF001


def _compositions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, DriverDependencies]]:
    """Yield every covered composition's label and injected dependencies.

    Parameters
    ----------
    tmp_path, monkeypatch:
        Scratch root and patcher shared by the composition builders.

    Yields
    ------
    tuple[str, DriverDependencies]
        Composition label and the dependencies it composes.
    """

    roots: dict[str, Path] = {}
    for name in ("unit", "dry-run", "cli"):
        root = tmp_path / name
        root.mkdir(parents=True)
        roots[name] = root
    yield _COMPOSITION_LABELS[0], _unit_dependencies(roots["unit"])
    yield _COMPOSITION_LABELS[1], _dry_run_dependencies(roots["dry-run"], monkeypatch)
    yield _COMPOSITION_LABELS[2], _production_cli_dependencies(roots["cli"], monkeypatch)


@pytest.mark.smoke
def test_fail_closed_dependency_derivation_is_live() -> None:
    """The derivation must still understand the driver's dependency-guard shape."""

    fields = _defaulted_dependency_fields()
    assert fields, "DriverDependencies has no defaulted-None field: the contract moved"
    read = frozenset().union(
        *(_dependency_reads(tree, frozenset(fields)) for _name, tree in _driver_sources())
    )
    unread = sorted(set(fields) - read)
    # A field nobody reads cannot be derived either way, so an unread field means
    # the scan lost the driver's ``self.dependencies.<field>`` shape (a rename, a
    # new indirection) and the whole derivation is silently empty-ish.
    assert not unread, f"defaulted dependencies are never read off self.dependencies: {unread}"
    derived = derive_fail_closed_dependencies()
    assert derived, "no fail-closed dependency was derived: the refusal shape moved"
    for field, sites in derived.items():
        assert field in fields
        assert sites


@pytest.mark.smoke
def test_provider_reconciler_is_derived_as_fail_closed() -> None:
    """The launch-day regression's own dependency must be inside the derived set.

    Not the contract -- the derivation above is -- but its health check: a
    derivation that stops seeing the paid path's unconditional refusals would pass
    every composition check vacuously.
    """

    derived = derive_fail_closed_dependencies()
    assert "provider_reconciler" in derived
    assert len(derived["provider_reconciler"]) >= 2
    assert all(site.startswith("driver") for site in derived["provider_reconciler"])


@pytest.mark.smoke
def test_fallback_dependencies_are_not_derived_as_required() -> None:
    """A dependency with a real absent-path fallback must stay optional."""

    derived = derive_fail_closed_dependencies()
    for optional_field in ("sleeper", "usage_pause_scheduler"):
        assert optional_field in _defaulted_dependency_fields()
        assert optional_field not in derived


@pytest.mark.smoke
def test_every_composition_provides_every_fail_closed_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each shipped composition must supply every derived required dependency."""

    required = tuple(derive_fail_closed_dependencies())
    missing: dict[str, list[str]] = {}
    for label, dependencies in _compositions(tmp_path, monkeypatch):
        absent = [field for field in required if getattr(dependencies, field) is None]
        if absent:
            missing[label] = absent
    assert not missing, (
        "compositions omit fail-closed driver dependencies "
        f"{missing}; every derived requirement {required} must be injected by "
        "EVERY composition, not only the ones a unit test happens to exercise"
    )


@pytest.mark.smoke
def test_covered_compositions_include_every_production_site() -> None:
    """A new production ``DriverDependencies`` composition must be covered here."""

    sites = _production_composition_sites()
    assert sites == {"cli.py": 1}, (
        "production DriverDependencies construction sites changed: "
        f"{sites}. Add the new composition to this module's inventory "
        f"{_COMPOSITION_LABELS} in the same change."
    )
