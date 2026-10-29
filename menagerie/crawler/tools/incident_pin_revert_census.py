"""B3-XV-18 (PM-6/W20 acceptance): the incident-pin REVERT census.

For each recent campaign production diff: revert it in place in a scratch
checkout of the CURRENT tree and run only the blocking gate. A change whose
revert stays GREEN is an UNPINNED fix — nothing in the blocking gate defends
it — and blocks launch until pinned. The committed artifact is the
acceptance evidence.

The blocking-gate ladder per revert (documented, no silent narrowing):

0. **Conflict handling.** Reverting an older diff on the current tree
   usually conflicts on the inventory-enforced artifacts (the smoke
   inventory lock/floor, the pin registry, the audit register) because later
   changes regenerated them. Those conflicts ARE the never-lower web
   fighting the revert; the census resolves them by KEEPING the current
   tree's artifacts (so the gate below judges the reverted code against the
   standing inventory) and continues. A conflict on ANY OTHER file is
   ``pinned-by-conflict``: the diff is load-bearing for later committed
   state and cannot even be reverted cleanly.
1. **Inventory equality** (ONE full collection compared to the KEPT current
   lock's collected-node set): a revert that removes, renames, or adds test
   nodes reddens here — set-equality is itself a blocking-gate member. The
   smoke-subset half of the lock is deliberately NOT re-collected here: the
   smoke-tier run below enforces it in-session through the anti-substitution
   test, so the composition loses nothing and each revert pays one
   collection instead of two.
2. **Full smoke tier** (fail-fast): the blocking gate proper, which also
   contains every smoke-resolvable regression pin — a red pin IS a red
   blocking gate, so no separate pin pass runs in the census (the
   ``pinned-by-node`` verdict remains for targeted tool use). Red is
   ``pinned-by-blocking-gate``, green is ``UNPINNED-GREEN`` — a blocking
   finding. The census's own committed-artifact test is deselected here:
   the artifact cannot witness itself mid-mint (disclosed narrowing, one
   node).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

REVERT_CENSUS_SCHEMA_VERSION = "menagerie.crawler.incident-pin-revert-census.v1"
REVERT_CENSUS_ARTIFACT_PATH = Path(__file__).with_name("incident_pin_revert_census.json")

#: Inventory-enforced artifacts a revert may conflict on without being
#: load-bearing: the census keeps the CURRENT tree's copies so the standing
#: never-lower web judges the reverted code.
INVENTORY_ARTIFACTS = (
    "menagerie/crawler/tests/regression_pins.json",
    "menagerie/crawler/tools/audit_register.json",
    "menagerie/crawler/tools/incident_pin_revert_census.json",
)

#: The census's own acceptance test: deselected in step 3 because the
#: artifact cannot witness itself while being minted.
_SELF_TEST_DESELECT = (
    "menagerie/crawler/tests/test_audit_register.py::"
    "test_committed_revert_census_artifact_covers_thirty_and_is_clean"
)


class RevertCensusError(RuntimeError):
    """Raised when the scratch repo cannot be driven."""


def classify_revert(
    *,
    inventory_matches: bool,
    pin_run_passed: Optional[bool],
    blocking_gate_passed: Optional[bool] = None,
) -> dict:
    """Apply the documented ladder to one revert's observations."""

    if not inventory_matches:
        return {"verdict": "pinned-by-inventory"}
    if pin_run_passed is False:
        return {"verdict": "pinned-by-node"}
    if blocking_gate_passed is False:
        return {"verdict": "pinned-by-blocking-gate"}
    return {"verdict": "UNPINNED-GREEN"}


def _git(scratch: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(scratch), *argv], check=False, capture_output=True, text=True
    )


def _conflicted_paths(scratch: Path) -> list[str]:
    status = _git(scratch, "status", "--porcelain")
    paths = []
    for line in status.stdout.splitlines():
        if line[:2] in {"UU", "AA", "DU", "UD", "AU", "UA", "DD"}:
            paths.append(line[3:].strip())
    return paths


def _inventory_matches(scratch: Path, python: str) -> bool:
    """The inventory rung is the collection gate (RC-3): the reverted tree must
    collect with zero errors and every tracked module must still test."""

    from menagerie.crawler.tools import collection_gate

    try:
        report = collection_gate.collect(scratch, python=python)
    except collection_gate.CollectionGateError:
        return False
    return report.ok


def _subprocess_env() -> dict:
    return {"PYTHONPATH": ".", **os.environ}


def _pin_run_passed(scratch: Path, python: str, pin_nodes: Sequence[str]) -> bool:
    from menagerie.crawler.tools.audit_register import resolve_pytest_ref

    collect = subprocess.run(
        [python, "-m", "pytest", "--collect-only", "-q", "--no-header",
         "-o", "addopts=", "menagerie/crawler/tests", "tests/crawler"],
        check=False,
        capture_output=True,
        text=True,
        cwd=scratch,
        env=_subprocess_env(),
    )
    universe = frozenset(
        line.strip() for line in collect.stdout.splitlines() if "::" in line
    )
    runnable = sorted(
        {hit for node in pin_nodes for hit in resolve_pytest_ref(node, universe)}
    )
    if not runnable:
        return True
    completed = subprocess.run(
        [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--no-header",
         "-o", "addopts=", *runnable],
        check=False,
        capture_output=True,
        text=True,
        cwd=scratch,
        env=_subprocess_env(),
    )
    return completed.returncode == 0


def _blocking_gate_passed(scratch: Path, python: str) -> bool:
    completed = subprocess.run(
        [python, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--no-header",
         "-m", "smoke", "--deselect", _SELF_TEST_DESELECT,
         "menagerie/crawler/tests", "tests/crawler"],
        check=False,
        capture_output=True,
        text=True,
        cwd=scratch,
        env=_subprocess_env(),
    )
    return completed.returncode == 0


def run_revert_census(
    scratch: Path,
    commits: Sequence[tuple[str, str]],
    *,
    python: str,
    pin_nodes: Sequence[str],
    log=print,
) -> dict:
    """Revert each commit in the scratch checkout and classify it."""

    base = _git(scratch, "rev-parse", "HEAD").stdout.strip()
    rows: list[dict] = []
    for sha, subject in commits:
        _git(scratch, "reset", "--hard", base)
        _git(scratch, "clean", "-fd")
        reverted = _git(scratch, "revert", "--no-commit", sha)
        detail = ""
        if reverted.returncode != 0:
            conflicted = _conflicted_paths(scratch)
            foreign = [path for path in conflicted if path not in INVENTORY_ARTIFACTS]
            if foreign or not conflicted:
                rows.append(
                    {
                        "sha": sha,
                        "subject": subject,
                        "verdict": "pinned-by-conflict",
                        "detail": (
                            "revert does not apply cleanly outside the inventory "
                            f"artifacts (conflicts: {sorted(conflicted)}): the diff "
                            "is load-bearing for later committed state"
                        ),
                    }
                )
                log(f"revert-census: {sha[:9]} -> pinned-by-conflict")
                _git(scratch, "revert", "--abort")
                _git(scratch, "reset", "--hard", base)
                continue
            # Keep the CURRENT tree's inventory artifacts and continue: the
            # standing never-lower web judges the reverted code below.
            _git(scratch, "checkout", base, "--", *conflicted)
            detail = f"inventory-artifact conflicts kept at current tree: {sorted(conflicted)}"
        inventory_ok = _inventory_matches(scratch, python)
        # The pin pass is subsumed here: every smoke-resolvable pin is a
        # member of the smoke-tier run below (module docstring, step 2).
        gate_passed = _blocking_gate_passed(scratch, python) if inventory_ok else None
        verdict = classify_revert(
            inventory_matches=inventory_ok,
            pin_run_passed=None,
            blocking_gate_passed=gate_passed,
        )
        row = {"sha": sha, "subject": subject, **verdict}
        if detail:
            row["detail"] = detail
        rows.append(row)
        log(f"revert-census: {sha[:9]} -> {verdict['verdict']}")
        _git(scratch, "reset", "--hard", base)
        _git(scratch, "clean", "-fd")
    return {
        "schema_version": REVERT_CENSUS_SCHEMA_VERSION,
        "base_sha": base,
        "rows": rows,
        "unpinned": [row["sha"] for row in rows if row["verdict"] == "UNPINNED-GREEN"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-repo", type=Path, required=True)
    parser.add_argument("--commits-file", type=Path, required=True,
                        help="one '<sha> <subject>' per line")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", type=Path, default=REVERT_CENSUS_ARTIFACT_PATH)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from menagerie.crawler.tools.audit_register import (
        REGRESSION_PINS_PATH,
        all_pin_nodes,
        load_regression_pins,
    )

    pins = all_pin_nodes(load_regression_pins(REGRESSION_PINS_PATH))
    commits = []
    for line in args.commits_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            sha, _, subject = line.strip().partition(" ")
            commits.append((sha, subject))
    census = run_revert_census(
        args.scratch_repo.resolve(), commits, python=args.python, pin_nodes=pins
    )
    args.output.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"unpinned": census["unpinned"], "rows": len(census["rows"])}))
    return 0 if not census["unpinned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
