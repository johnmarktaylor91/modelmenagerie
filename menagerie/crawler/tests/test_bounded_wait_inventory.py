"""Bounded-wait inventory: no wait in the worker area may lack a deadline.

BL-12(b) / catalog C-19, applied as a permanent grep-class AST census over the
worker-area owner modules. One D-state worker, one contended flock, or one
unbounded reap used to hang an unattended run forever; every such wait now has
a bound, and this census fails the build the moment a NEW unbounded wait is
written -- the sprint's I2 DEADLINE invariant enforced in code, not memory.

Rules enforced per owner module:

* every ``.wait(`` / ``.join(`` call passes a timeout (positional or keyword);
* every ``fcntl.flock`` call is non-blocking (``LOCK_NB``) or an unlock
  (``LOCK_UN``) -- blocking exclusive acquisition is forbidden at the call
  site; bounded acquisition loops poll ``LOCK_NB`` under a deadline;
* every ``while True:`` loop body references a bounding construct (a deadline,
  timeout, shutdown event, or bounded select) -- a spin with no exit signal
  is exactly the silent-stall shape this census exists to forbid.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]

#: The BL-12 owner modules: the worker execution + containment surface.
_OWNER_MODULES = (
    "worker.py",
    "worker_supervisor.py",
    "driver_receipts.py",
    "execution_lock.py",
    "policy.py",
    "standard_inputs.py",
    "modes.py",
)

#: Identifier fragments that prove a ``while True`` loop is bounded: it either
#: measures time against a deadline, polls a shutdown signal, or blocks in a
#: timeout-carrying select. Matching is on names REFERENCED INSIDE the loop
#: body, so a bound cannot be claimed by a comment.
_BOUNDING_NAME_FRAGMENTS = (
    "deadline",
    "timeout",
    "elapsed",
    "shutdown",
    "monotonic",
    "select",
)


def _call_has_timeout(node: ast.Call) -> bool:
    """Return whether a wait/join call carries an explicit timeout argument."""

    if any(keyword.arg == "timeout" for keyword in node.keywords):
        return True
    # subprocess.Popen.wait / threading.Thread.join accept the timeout as the
    # sole positional argument.
    return len(node.args) >= 1


def _flock_flags_names(node: ast.Call) -> set[str]:
    """Collect attribute/name identifiers inside a flock flags expression."""

    names: set[str] = set()
    if len(node.args) < 2:
        return names
    for descendant in ast.walk(node.args[1]):
        if isinstance(descendant, ast.Attribute):
            names.add(descendant.attr)
        elif isinstance(descendant, ast.Name):
            names.add(descendant.id)
    return names


def _loop_referenced_names(node: ast.While) -> set[str]:
    """Collect every identifier referenced inside one loop body."""

    names: set[str] = set()
    for statement in node.body:
        for descendant in ast.walk(statement):
            if isinstance(descendant, ast.Name):
                names.add(descendant.id)
            elif isinstance(descendant, ast.Attribute):
                names.add(descendant.attr)
            elif isinstance(descendant, ast.arg):
                names.add(descendant.arg)
    return names


def _census() -> list[str]:
    """Return every unbounded-wait violation across the owner modules."""

    violations: list[str] = []
    for module_name in _OWNER_MODULES:
        path = _CRAWLER_ROOT / module_name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"wait", "join"} and not _call_has_timeout(node):
                    violations.append(
                        f"{module_name}:{node.lineno} .{node.func.attr}() without a timeout"
                    )
                if node.func.attr == "flock":
                    flags = _flock_flags_names(node)
                    if flags and "LOCK_NB" not in flags and "LOCK_UN" not in flags:
                        violations.append(
                            f"{module_name}:{node.lineno} blocking flock without LOCK_NB"
                        )
            if isinstance(node, ast.While):
                test = node.test
                is_while_true = isinstance(test, ast.Constant) and test.value is True
                if not is_while_true:
                    continue
                referenced = _loop_referenced_names(node)
                bounded = any(
                    any(fragment in name.lower() for fragment in _BOUNDING_NAME_FRAGMENTS)
                    for name in referenced
                )
                if not bounded:
                    violations.append(
                        f"{module_name}:{node.lineno} while True without a bounding reference"
                    )
    return violations


@pytest.mark.smoke
def test_no_unbounded_waits_in_worker_area_owner_modules() -> None:
    """CENSUS: every wait, join, flock, and spin loop in the area is bounded.

    A new entry in this list is a recurrence of catalog C-19 (the unbounded
    ``process.wait()``, the timeout-free SIGTERM-immune global flock): fix the
    wait, never this census.
    """

    assert _census() == []


@pytest.mark.smoke
def test_census_actually_detects_an_unbounded_wait(tmp_path: Path) -> None:
    """Self-test: the census rules flag each forbidden construct when present."""

    sample = tmp_path / "sample.py"
    sample.write_text(
        "import fcntl\n"
        "def bad(process, descriptor):\n"
        "    process.wait()\n"
        "    fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
        "    while True:\n"
        "        pass\n",
        encoding="utf-8",
    )
    tree = ast.parse(sample.read_text(encoding="utf-8"))
    findings = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"wait", "join"} and not _call_has_timeout(node):
                findings += 1
            if node.func.attr == "flock":
                flags = _flock_flags_names(node)
                if flags and "LOCK_NB" not in flags and "LOCK_UN" not in flags:
                    findings += 1
        if isinstance(node, ast.While):
            test = node.test
            if isinstance(test, ast.Constant) and test.value is True:
                if not any(
                    any(fragment in name.lower() for fragment in _BOUNDING_NAME_FRAGMENTS)
                    for name in _loop_referenced_names(node)
                ):
                    findings += 1
    assert findings == 3
