"""Award/runner closure walkers must follow same-package re-exports.

Regression pin for the DX-3 closure-diet extraction: ``AuthorityDerivationError``
moved from ``authority.py`` into the dependency-free leaf ``completion_line.py``,
with ``authority.py`` re-exporting it for every historical parent-side caller.
The award-closure AST walker resolved cross-module edges only to real
DEFINITIONS, so the re-export read as "award-closure source symbol is missing"
and every driver leg quarantined every member at the env turn
(``deferred:scheduler-fault-quarantined``), turning every close into
``*-with-deferrals``.

The walkers stay fail-closed: an unresolvable symbol still refuses. A symbol
that resolves through a same-package ``from menagerie.crawler.X import Y``
re-export chain hashes the REAL definition in its defining module, so the
anti-tamper closure keeps covering the moved class body.
"""

from __future__ import annotations

import ast

import pytest

from menagerie.crawler import driver_admission
from menagerie.crawler.driver_contracts import DriverIntegrationError

pytestmark = pytest.mark.smoke


def test_award_closure_identity_resolves_reexported_symbols() -> None:
    """The live tree computes an award closure despite the completion-line move."""

    identity = driver_admission._award_closure_identity()
    assert identity.startswith("sha256:")


def test_award_closure_hashes_the_real_definition_behind_a_reexport() -> None:
    """A re-exported name binds the defining module's class body, fail-closed."""

    seed_module = "menagerie/crawler/user.py".removeprefix("menagerie/crawler/")
    source_items = (
        (
            seed_module,
            b"from menagerie.crawler.facade import MovedError\n"
            b"def seed():\n"
            b"    raise MovedError('x')\n",
        ),
        (
            "facade.py",
            b"from menagerie.crawler.leaf import MovedError as MovedError\n",
        ),
        (
            "leaf.py",
            b"class MovedError(ValueError):\n"
            b"    pass\n",
        ),
    )
    symbols = {seed_module: ("seed",)}
    components = driver_admission._award_closure_components(source_items, symbols)
    assert "leaf.py:MovedError" in components
    assert f"{seed_module}:seed" in components


def test_award_closure_still_refuses_a_symbol_with_no_definition_anywhere() -> None:
    """Fail-closed is preserved: a dangling re-export still refuses typed."""

    source_items = (
        ("user.py", b"from menagerie.crawler.facade import Ghost\ndef seed():\n    return Ghost\n"),
        ("facade.py", b"from menagerie.crawler.nowhere import Ghost as Ghost\n"),
    )
    with pytest.raises(DriverIntegrationError, match="award-closure source symbol is missing"):
        driver_admission._award_closure_components(source_items, {"user.py": ("seed",)})


def test_runner_identity_resolves_reexported_symbols() -> None:
    """The runner execution closure computes on the live tree too."""

    identity = driver_admission._runner_identity()
    assert identity.startswith("sha256:")


def test_authority_reexport_is_walkable_from_the_live_tree() -> None:
    """The concrete regression shape: authority.py re-exports the moved error."""

    root = driver_admission.Path(driver_admission.__file__).parent
    tree = ast.parse((root / "authority.py").read_text(encoding="utf-8"))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reexported = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "menagerie.crawler.completion_line"
        for alias in node.names
    }
    # The pin only bites while the re-export shape exists; if the class moves
    # back to a real definition this guard keeps the test honest instead of
    # vacuously green.
    assert "AuthorityDerivationError" in (defined | reexported)
