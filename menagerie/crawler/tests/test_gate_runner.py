"""Direct tests for the closure-keyed gate runner (RC-3).

The runner is the ONE path a gate command takes into the verdict ledger, so
its own behavior is pinned here from a real committed fixture repo: a run
appends exactly one closure-keyed row, an unchanged closure since ``--base``
runs nothing and records nothing, a dirty closure refuses, junit outcomes map
onto the four exact node sets, and an import recomputes every closure and
re-hashes every evidence blob before accepting a row.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from menagerie.crawler.tests.test_gate_verdicts import (
    _TWO_GATES,
    _commit_file,
    _fixture_manifest,
    _fixture_repo,
)
from menagerie.crawler.tools import gate_closure, gate_runner, gate_verdicts

pytestmark = pytest.mark.smoke


def _run(repo: Path, gate: str, *, base_ref: str | None = None, force: bool = False) -> int:
    return gate_runner.run_gate(
        repo, gate_name=gate, python=sys.executable, base_ref=base_ref, force=force
    )


def test_run_appends_one_closure_keyed_green_row(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, _fixture_manifest(_TWO_GATES))
    manifest = gate_closure.load_manifest(repo)

    assert _run(repo, "tests-gate") == 0

    rows, legacy = gate_verdicts.load_ledger(repo)
    assert legacy == 0 and len(rows) == 1
    (row,) = rows
    assert row["gate"] == "tests-gate" and row["status"] == "green"
    assert row["closure_sha"] == gate_closure.closure_sha(repo, manifest, "tests-gate")
    assert row["commit_sha"] == gate_closure.commit_sha(repo)
    assert row["env_identity"]["conda_prefix"] in {"present", "absent"}
    evidence = repo / gate_verdicts.GATE_LOGS_RELDIR / f"{row['log_sha256']}.log"
    assert evidence.is_file()
    assert gate_verdicts._sha256_hex(evidence.read_bytes()) == row["log_sha256"]


def test_unchanged_closure_since_base_runs_nothing_and_records_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _fixture_repo(tmp_path, _fixture_manifest(_TWO_GATES))
    assert _run(repo, "tests-gate") == 0
    _commit_file(repo, "docs/other.md", "# docs only\n")

    assert _run(repo, "tests-gate", base_ref="HEAD~1") == 0
    assert "nothing to re-run, nothing recorded" in capsys.readouterr().out
    rows, _ = gate_verdicts.load_ledger(repo)
    assert len(rows) == 1

    # A commit INSIDE the closure moves it: the gate runs and appends a new row.
    _commit_file(repo, "tests/test_b.py", "def test_b():\n    pass\n")
    assert _run(repo, "tests-gate", base_ref="HEAD~1") == 0
    rows, _ = gate_verdicts.load_ledger(repo)
    assert len(rows) == 2 and rows[0]["closure_sha"] != rows[1]["closure_sha"]

    # --force runs even when the closure is unchanged.
    assert _run(repo, "tests-gate", base_ref="HEAD", force=True) == 0
    rows, _ = gate_verdicts.load_ledger(repo)
    assert len(rows) == 3


def test_dirty_closure_and_attestation_gate_refuse(tmp_path: Path) -> None:
    gates = dict(_TWO_GATES)
    gates["attest"] = {"input_closure": ["docs/"], "command": None}
    repo = _fixture_repo(tmp_path, _fixture_manifest(gates))

    assert _run(repo, "attest") == gate_runner.EXIT_REFUSED

    (repo / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    assert _run(repo, "tests-gate") == gate_runner.EXIT_REFUSED
    assert not (repo / gate_verdicts.VERDICTS_RELPATH).exists()


def test_missing_junit_is_disclosed_and_recorded_red(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A junit gate whose process dies before session finish (the faulthandler backstop
    ending a native-level hang, or an external kill) records a red verdict with empty
    node sets and names the condition on stderr instead of crashing or waiting."""

    manifest = _fixture_manifest(_TWO_GATES)
    manifest["gates"]["tests-gate"]["junit"] = True
    manifest["gates"]["tests-gate"]["command"] = ["{python}", "-c", "import os; os._exit(1)"]
    repo = _fixture_repo(tmp_path, manifest)

    assert _run(repo, "tests-gate") == 1

    assert "WARNING no junit report was written" in capsys.readouterr().err
    rows, _legacy = gate_verdicts.load_ledger(repo)
    assert rows[-1]["gate"] == "tests-gate" and rows[-1]["status"] == "red"


def test_junit_outcomes_map_onto_the_four_exact_node_sets(tmp_path: Path) -> None:
    junit = tmp_path / "r.xml"
    junit.write_text(
        """<testsuites><testsuite>
        <testcase classname="pkg.test_m" name="test_pass"/>
        <testcase classname="pkg.test_m" name="test_fail"><failure/></testcase>
        <testcase classname="pkg.test_m" name="test_err"><error/></testcase>
        <testcase classname="pkg.test_m" name="test_skip"><skipped type="pytest.skip"/></testcase>
        <testcase classname="pkg.test_m" name="test_xf"><skipped type="pytest.xfail"/></testcase>
        </testsuite></testsuites>""",
        encoding="utf-8",
    )
    sets = gate_runner._junit_node_sets(junit)
    assert sets == {
        "passed": ["pkg/test_m.py::test_pass"],
        "failed": ["pkg/test_m.py::test_err", "pkg/test_m.py::test_fail"],
        "skipped": ["pkg/test_m.py::test_skip"],
        "xfailed": ["pkg/test_m.py::test_xf"],
    }
    assert gate_runner._junit_node_sets(tmp_path / "missing.xml") == {
        "passed": [],
        "failed": [],
        "skipped": [],
        "xfailed": [],
    }


def test_import_recomputes_closures_rehashes_evidence_and_dedupes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _fixture_repo(tmp_path, _fixture_manifest(_TWO_GATES))
    assert _run(repo, "tests-gate") == 0
    exported = tmp_path / "exported"
    shutil.copytree(repo / ".crawl-local", exported)
    shutil.rmtree(repo / ".crawl-local")

    assert gate_runner.import_verdicts(repo, from_dir=exported) == 0
    assert "imported 1 row(s), skipped 0 duplicate(s), refused 0" in capsys.readouterr().out
    rows, _ = gate_verdicts.load_ledger(repo)
    assert len(rows) == 1

    assert gate_runner.import_verdicts(repo, from_dir=exported) == 0
    assert "imported 0 row(s), skipped 1 duplicate(s), refused 0" in capsys.readouterr().out

    # A corrupt evidence blob refuses the row (exit 1) and appends nothing.
    shutil.rmtree(repo / ".crawl-local")
    (blob,) = (exported / "gate-logs").glob("*.log")
    blob.write_bytes(blob.read_bytes() + b"tampered\n")
    assert gate_runner.import_verdicts(repo, from_dir=exported) == 1
    assert "missing or corrupt" in capsys.readouterr().err
    assert not (repo / gate_verdicts.VERDICTS_RELPATH).exists()

    # A row whose closure does not recompute from its commit refuses too.
    blob.write_bytes(blob.read_bytes()[: -len(b"tampered\n")])
    ledger = exported / "gate-verdicts.jsonl"
    ledger.write_text(
        ledger.read_text(encoding="utf-8").replace('"gate":"tests-gate"', '"gate":"src-gate"')
    )
    assert gate_runner.import_verdicts(repo, from_dir=exported) == 1
    assert "does not recompute" in capsys.readouterr().err

    assert (
        gate_runner.import_verdicts(repo, from_dir=tmp_path / "empty") == gate_runner.EXIT_REFUSED
    )


def test_main_dispatches_and_refuses_typed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = _fixture_repo(tmp_path, _fixture_manifest(_TWO_GATES))
    manifest = gate_closure.load_manifest(repo)

    assert gate_runner.main(["--repo-root", str(repo), "closure", "--gate", "tests-gate"]) == 0
    assert capsys.readouterr().out.strip() == gate_closure.closure_sha(repo, manifest, "tests-gate")

    _commit_file(repo, "src/b.py", "B = 2\n")
    assert gate_runner.main(["--repo-root", str(repo), "plan", "--base", "HEAD~1"]) == 0
    assert '"src-gate"' in capsys.readouterr().out

    assert gate_runner.main(["--repo-root", str(repo), "closure", "--gate", "no-such-gate"]) == (
        gate_runner.EXIT_REFUSED
    )
    assert "REFUSED" in capsys.readouterr().err
