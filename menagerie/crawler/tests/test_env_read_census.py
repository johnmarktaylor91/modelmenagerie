"""Every environment read in the crawler is classified, or the census fails (W15).

The wake environment vocabulary was a CLASS of partial hand-enumerations: 32
names read as literals, ~14 more through module constants, six persisted, and
the unpersisted remainder silently reshaping campaigns (a bare-PATH ``claude``
reading as an author storm; an absent ``EXA_API_KEY`` reading as a phantom
campaign pause). This census walks the tree's ACTUAL ``os.environ`` /
``os.getenv`` read sites by AST and refuses any name that
``menagerie.crawler.env_census`` does not classify -- in BOTH directions, so
a stale census row fails exactly like an unclassified new read.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from menagerie.crawler.env_census import (
    CLASSIFIED_ENVIRONMENT_READS,
    CONSTANT_CARRIED_READS,
    WAKE_REQUIRED_SECRETS,
    missing_wake_secrets,
)

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]


def _environ_read_names() -> tuple[set[str], set[str], list[str]]:
    """Return (literal names, constant-carried names, dynamic sites)."""

    paths = sorted(_CRAWLER_ROOT.glob("*.py")) + sorted(
        (_CRAWLER_ROOT / "tools").glob("*.py")
    )
    trees = {path: ast.parse(path.read_text(encoding="utf-8")) for path in paths}
    # Pass 1: one GLOBAL constant map -- cross-module imports (TIER_CAMPAIGN_ENV
    # and friends live in constants.py, read elsewhere) resolve through it.
    module_constants: dict[str, str] = {}
    for tree in trees.values():
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                module_constants.setdefault(node.targets[0].id, node.value.value)

    literal: set[str] = set()
    constant_carried: set[str] = set()
    unresolvable: list[str] = []
    for path, tree in trees.items():
        for node in ast.walk(tree):
            argument: ast.expr | None = None
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in {"get", "getenv", "pop", "setdefault"}
                    and node.args
                ):
                    owner = func.value
                    is_environ = (
                        isinstance(owner, ast.Attribute) and owner.attr == "environ"
                    ) or (isinstance(owner, ast.Name) and owner.id == "environ")
                    is_os_getenv = (
                        isinstance(owner, ast.Name)
                        and owner.id == "os"
                        and func.attr == "getenv"
                    )
                    if is_environ or is_os_getenv:
                        argument = node.args[0]
            elif isinstance(node, ast.Subscript):
                owner = node.value
                if isinstance(owner, ast.Attribute) and owner.attr == "environ":
                    argument = node.slice
            if argument is None:
                continue
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                literal.add(argument.value)
            elif isinstance(argument, ast.Name):
                resolved = module_constants.get(argument.id)
                if resolved is not None:
                    constant_carried.add(resolved)
                else:
                    unresolvable.append(f"{path.name}: {argument.id}")
            else:
                unresolvable.append(f"{path.name}:{getattr(node, 'lineno', '?')}")
    return literal, constant_carried, unresolvable


@pytest.mark.smoke
def test_every_environment_read_is_classified() -> None:
    """The census is exact in both directions; no read escapes classification.

    Constant-carried reads resolve within their own module; a read through a
    name this walker cannot resolve to a module-level string constant is
    reported as unresolvable and must be either simplified or classified via
    an explicit vocabulary entry -- silence is the failure mode this census
    retires.
    """

    literal, constant_carried, unresolvable = _environ_read_names()
    classified = set(CLASSIFIED_ENVIRONMENT_READS)
    constant_classified = set(CONSTANT_CARRIED_READS)

    unclassified_literals = sorted(literal - classified - constant_classified)
    assert unclassified_literals == [], (
        "environment reads with no census classification (add them to "
        f"menagerie.crawler.env_census): {unclassified_literals}"
    )
    stale_rows = sorted(classified - literal)
    assert stale_rows == [], (
        f"census rows no longer read anywhere as literals: {stale_rows}"
    )
    unclassified_constants = sorted(
        constant_carried - constant_classified - classified
    )
    assert unclassified_constants == [], (
        "constant-carried environment reads with no census classification: "
        f"{unclassified_constants}"
    )
    # The only legal dynamic sites are loop-variable passthroughs that iterate
    # an already-classified vocabulary (never a specific hidden name). This
    # list is CLOSED: a new dynamic read site must either resolve to a
    # module-level constant or be reviewed onto this list.
    known_dynamic = {
        "author_attempts.py: name",
        "cli.py: name",
        "doctor.py",
        "env_census.py: name",
        "operator_environment.py: environment_key",
        "operator_environment.py: key",
        "source_broker.py: name",
    }
    normalized = {
        site.split(":")[0] if site.startswith("doctor.py") else site
        for site in unresolvable
    }
    unexpected = sorted(normalized - known_dynamic)
    assert unexpected == [], (
        f"unreviewed dynamic environment-read sites: {unexpected}"
    )


@pytest.mark.smoke
def test_wake_required_secrets_are_classified_secrets() -> None:
    """The refuse-loudly set is a subset of census-classified secrets."""

    all_reads = {**CLASSIFIED_ENVIRONMENT_READS, **CONSTANT_CARRIED_READS}
    for name in WAKE_REQUIRED_SECRETS:
        assert all_reads.get(name) == "secret", name


@pytest.mark.smoke
def test_production_leg_refuses_loudly_on_absent_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W15/B3-XO-8: an absent secret refuses at leg open, naming the variable.

    The historical shape was a phantom campaign-wide pause: every author
    session died as infra and read as a storm. A production launch shape (one
    carrying the environment-readiness receipt -- every 200b leg and every
    replayed wake does) must instead refuse before any admission or spend.
    """

    from dataclasses import replace

    from menagerie.crawler.driver_contracts import DriverIntegrationError
    from menagerie.crawler.tests.test_slice_f_driver import _driver, _snapshot

    snapshot = _snapshot(tmp_path, count=1)
    receipt = tmp_path / "environment-readiness-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    driver = _driver(tmp_path, snapshot)
    driver.config = replace(driver.config, environment_readiness_receipt_path=receipt)
    with pytest.raises(DriverIntegrationError, match="EXA_API_KEY"):
        driver.run()

    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    fresh = _driver(tmp_path, snapshot)
    fresh.config = replace(fresh.config, environment_readiness_receipt_path=receipt)
    assert fresh.run().status == "complete"


@pytest.mark.smoke
def test_missing_wake_secrets_reports_absent_and_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent and whitespace-only secrets both count as missing."""

    monkeypatch.delenv("EXA_API_KEY", raising=False)
    assert "EXA_API_KEY" in missing_wake_secrets()
    monkeypatch.setenv("EXA_API_KEY", "   ")
    assert "EXA_API_KEY" in missing_wake_secrets()
    monkeypatch.setenv("EXA_API_KEY", "exa-key")
    assert "EXA_API_KEY" not in missing_wake_secrets()


def _environ_write_names() -> tuple[set[str], list[str]]:
    """Return (written names, unresolvable write sites) across production code.

    Same scope as the read walker (crawler root + tools). A subscript
    assignment keyed by a loop variable resolves through the enclosing
    ``for name in <module-level tuple constant>`` (cli.py's launch-environment
    restore); any other dynamic key is unresolvable and fails the census.
    """

    paths = sorted(_CRAWLER_ROOT.glob("*.py")) + sorted(
        (_CRAWLER_ROOT / "tools").glob("*.py")
    )
    written: set[str] = set()
    unresolvable: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        tuple_constants: dict[str, tuple[str, ...]] = {}
        for node in tree.body:
            target: str | None = None
            value: ast.expr | None = None
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                target, value = node.targets[0].id, node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                target, value = node.target.id, node.value
            if target is None or not isinstance(value, ast.Tuple):
                continue
            members = [
                element.value
                for element in value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            if len(members) == len(value.elts):
                tuple_constants[target] = tuple(members)
        loop_vocabularies: dict[str, tuple[str, ...]] = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Name)
                and node.iter.id in tuple_constants
            ):
                loop_vocabularies[node.target.id] = tuple_constants[node.iter.id]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for assign_target in node.targets:
                if not isinstance(assign_target, ast.Subscript):
                    continue
                owner = assign_target.value
                is_environ = (
                    isinstance(owner, ast.Attribute) and owner.attr == "environ"
                ) or (isinstance(owner, ast.Name) and owner.id == "environ")
                if not is_environ:
                    continue
                key = assign_target.slice
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    written.add(key.value)
                elif isinstance(key, ast.Name) and key.id in loop_vocabularies:
                    written.update(loop_vocabularies[key.id])
                else:
                    unresolvable.append(f"{path.name}:{node.lineno}")
    return written, unresolvable


@pytest.mark.smoke
def test_every_environment_write_is_isolation_classified() -> None:
    """Every production ``os.environ`` write belongs to the isolation vocabulary.

    The restore-write leak class (D-63 launch restore; the mirror exports that
    cost the 200b presweep gate nine preflight tests on a leaked
    ``/mirrors/private``) is closed by construction only if the test-session
    isolation fixture snapshots EVERY name production code writes. This census
    walks the actual write sites and refuses, in both directions, any drift
    between them and ``LAUNCH_ENVIRONMENT_VARIABLES`` +
    ``CAMPAIGN_IDENTITY_ENVIRONMENT_WRITES`` -- a new production environment
    write cannot ship without joining the vocabulary the fixture restores.
    """

    from menagerie.crawler.cli import LAUNCH_ENVIRONMENT_VARIABLES
    from menagerie.crawler.env_census import CAMPAIGN_IDENTITY_ENVIRONMENT_WRITES

    written, unresolvable = _environ_write_names()
    assert unresolvable == [], (
        f"unresolvable dynamic environment-write sites: {unresolvable}"
    )
    vocabulary = set(LAUNCH_ENVIRONMENT_VARIABLES) | set(
        CAMPAIGN_IDENTITY_ENVIRONMENT_WRITES
    )
    unclassified = sorted(written - vocabulary)
    assert unclassified == [], (
        "production environment writes outside the isolation vocabulary "
        "(add them to cli.LAUNCH_ENVIRONMENT_VARIABLES or "
        f"env_census.CAMPAIGN_IDENTITY_ENVIRONMENT_WRITES): {unclassified}"
    )
    stale = sorted(vocabulary - written)
    assert stale == [], (
        f"isolation vocabulary rows no production code writes anymore: {stale}"
    )
