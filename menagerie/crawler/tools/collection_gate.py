"""Collection gate: the corpus must collect cleanly and every module must test.

Replaces the exact-inventory floor (``smoke_inventory.lock`` set equality),
retired by the overhaul ruling (D.3 item 4). The floor caught exactly one
incident in its life: a merge-conflict marker left in a test module made that
module unparseable and silently dropped 70 nodes. Zero collection errors
catches that same class at zero maintenance, so this gate demands:

- pytest ``--collect-only`` over both corpus roots exits 0 (a collection
  error -- SyntaxError from a conflict marker, an ImportError, a typo'd
  marker under ``--strict-markers`` -- exits 2 and fails the gate);
- every tracked ``test_*.py`` module under the roots collected at least one
  node (a module that collects nothing has been emptied, skipped out, or
  renamed away from pytest's discovery, and the gate names it);
- per-module collected counts are REPORTED, never gated on an exact total.

The ``addopts`` override keeps ``--strict-markers`` and drops only the
``-m 'not rare'`` selection so the whole corpus is measured.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

CORPUS_ROOTS = ("menagerie/crawler/tests", "tests/crawler")
_TEST_MODULE_PREFIX = "test_"


class CollectionGateError(RuntimeError):
    """Raised when collection cannot be measured at all."""


@dataclass(frozen=True)
class CollectionReport:
    """One measured collection over the corpus roots."""

    node_ids: tuple[str, ...]
    returncode: int
    error_lines: tuple[str, ...]
    tracked_modules: tuple[str, ...]
    output_tail: tuple[str, ...] = field(default_factory=tuple)

    @property
    def per_module(self) -> dict[str, int]:
        counts: Counter[str] = Counter(node.split("::", 1)[0] for node in self.node_ids)
        return dict(sorted(counts.items()))

    @property
    def empty_modules(self) -> tuple[str, ...]:
        collected = set(self.per_module)
        return tuple(module for module in self.tracked_modules if module not in collected)

    @property
    def failures(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.returncode != 0:
            failures.append(
                f"collection exited {self.returncode} (0 required): "
                + ("; ".join(self.error_lines) if self.error_lines else "see output tail")
            )
        if not self.node_ids:
            failures.append("collection produced ZERO node ids")
        for module in self.empty_modules:
            failures.append(f"module collected zero tests: {module}")
        return tuple(failures)

    @property
    def ok(self) -> bool:
        return not self.failures


def tracked_test_modules(repo_root: Path, roots: Sequence[str] = CORPUS_ROOTS) -> tuple[str, ...]:
    """Return every tracked ``test_*.py`` module under ``roots`` (git ls-files)."""

    completed = subprocess.run(
        ("git", "ls-files", "--", *roots),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CollectionGateError(
            f"git ls-files failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    modules = [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().endswith(".py") and Path(line.strip()).name.startswith(_TEST_MODULE_PREFIX)
    ]
    return tuple(sorted(modules))


def _is_node_line(line: str) -> bool:
    return "::" in line and not line.startswith(("=", "<", " ", "ERROR", "error"))


def collect(
    repo_root: Path,
    *,
    python: str = sys.executable,
    roots: Sequence[str] = CORPUS_ROOTS,
    tracked_modules: Sequence[str] | None = None,
    marker: str | None = None,
) -> CollectionReport:
    """Measure one collection over ``roots`` in a subprocess.

    ``marker`` narrows the collection with ``-m`` (a selection view such as
    the smoke tier); the gate itself always measures the whole corpus.
    """

    argv = [
        python,
        "-m",
        "pytest",
        *roots,
        "-q",
        "--collect-only",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=--strict-markers",
    ]
    if marker is not None:
        argv.extend(["-m", marker])
    # The collection gate adjudicates COLLECTION only. The smoke execution
    # floor is an execution-gate arm: inherited into a --collect-only child it
    # would refuse every selected node as "no terminal outcome" and turn a
    # clean collection into exit 1 (seen when the smoke gate's own tests
    # measure the corpus universe from inside the armed gate run).
    child_env = {
        key: value for key, value in os.environ.items() if key != "MENAGERIE_SMOKE_EXECUTION_FLOOR"
    }
    child_env["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        argv,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=child_env,
    )
    lines = (completed.stdout + completed.stderr).splitlines()
    nodes = tuple(sorted({line.strip() for line in lines if _is_node_line(line.strip())}))
    errors = tuple(
        line.strip()
        for line in lines
        if line.startswith(("ERROR", "E   ")) or "error" in line.lower() and "collecting" in line
    )
    modules = (
        tracked_test_modules(repo_root, roots)
        if tracked_modules is None
        else tuple(tracked_modules)
    )
    return CollectionReport(
        node_ids=nodes,
        returncode=completed.returncode,
        error_lines=errors,
        tracked_modules=modules,
        output_tail=tuple(lines[-25:]),
    )


def render_report(report: CollectionReport) -> str:
    """Render the per-module census plus verdict for humans and CI logs."""

    lines = ["collection-gate: per-module collected counts"]
    for module, count in report.per_module.items():
        lines.append(f"  {count:5d}  {module}")
    lines.append(
        f"collection-gate: {len(report.node_ids)} nodes across "
        f"{len(report.per_module)} modules ({len(report.tracked_modules)} tracked)"
    )
    if report.ok:
        lines.append("collection-gate: OK (zero collection errors, every module tests)")
    else:
        lines.append("collection-gate: FAILED")
        lines.extend(f"  - {failure}" for failure in report.failures)
        if report.returncode != 0:
            lines.append("collection-gate: output tail")
            lines.extend(f"  | {line}" for line in report.output_tail)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="corpus root to collect (repeatable; default: the two crawler corpus roots)",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    roots = tuple(args.root) if args.root else CORPUS_ROOTS
    try:
        report = collect(repo_root, python=args.python, roots=roots)
    except CollectionGateError as exc:
        print(f"collection-gate: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "returncode": report.returncode,
                    "nodes": len(report.node_ids),
                    "per_module": report.per_module,
                    "empty_modules": list(report.empty_modules),
                    "failures": list(report.failures),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
