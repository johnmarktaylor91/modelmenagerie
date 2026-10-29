"""Gate-verdict ledger v3: closure-keyed rows, full evidence, typed refusals.

RC-3 rekeyed the ledger from the whole tree to each gate's input closure. The
tests here drive the real closure computation over a synthetic git repository
with a fixture manifest: a commit outside a gate's closure keeps its verdict
consumable, a commit inside it refuses as stale, and the census over source
paths refuses a manifest that leaves any tracked source uncovered.

Also carries the FORBIDDEN-OVERRIDE census: the operator-attestation class of
override (any spelling) may never appear in an argv parser, config schema, or
environment read anywhere in the crawler, its tools, or its workflows.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.tools import gate_closure, gate_verdicts

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TREE = "1" * 40
_CLOSURE = "a" * 64


def _git(repo: Path, *arguments: str) -> str:
    """Run one git command in a synthetic test repository."""

    return subprocess.run(
        ("git", *arguments), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _row(repo: Path, *, gate: str = "crawler-smoke", status: str = "green", **overrides):
    """Build one valid verdict row bound to a fixture closure."""

    log_sha256 = gate_verdicts.store_gate_log(repo, f"output of {gate} {status}\n")
    row = gate_verdicts.build_verdict(
        gate=gate,
        status=status,
        exit_code=0 if status == "green" else 1,
        closure_sha=_CLOSURE,
        tree_sha_start=_TREE,
        tree_sha_end=_TREE,
        commit_sha="2" * 40,
        env_identity={"fixture": "x"},
        argv=["pytest"],
        execution_floor_armed=True,
        node_sets={
            "passed": ["t::a"],
            "failed": [] if status == "green" else ["t::red"],
            "skipped": [],
            "xfailed": [],
        },
        log_sha256=log_sha256,
    )
    row.update(overrides)
    return row


def _fixture_manifest(gates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Build a v3 manifest document over fixture gates (defaults filled in)."""

    rendered: dict[str, dict[str, Any]] = {}
    for name, spec in gates.items():
        rendered[name] = {
            "description": f"fixture gate {name}",
            "always": False,
            "input_closure": spec["input_closure"],
            "exclude": spec.get("exclude", []),
            "command": spec.get("command", ["{python}", "-c", "pass"]),
            "env": {},
            "junit": False,
            "requires_armed_floor": spec.get("requires_armed_floor", False),
        }
    return {
        "schema_version": gate_closure.MANIFEST_SCHEMA_VERSION,
        "comment": "fixture",
        "census_roots": ["src/", "tests/", "pyproject.toml"],
        "implicit_closure": ["menagerie/crawler/tools/required_gate_set.json"],
        "gates": rendered,
        "required_for_invocation": list(gates),
    }


def _fixture_repo(tmp_path: Path, manifest: Mapping[str, Any]) -> Path:
    """Create a committed repo carrying ``manifest`` plus two source files."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "docs").mkdir()
    (repo / "src" / "a.py").write_text("A = 1\n", encoding="utf-8")
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    (repo / "docs" / "guide.md").write_text("# guide\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    manifest_path = repo / gate_closure.MANIFEST_RELPATH
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    return repo


def _commit_file(repo: Path, relative: str, content: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", relative)
    _git(repo, "commit", "-m", f"edit {relative}")


_TWO_GATES: dict[str, dict[str, Any]] = {
    "src-gate": {"input_closure": ["src/"], "requires_armed_floor": True},
    "tests-gate": {"input_closure": ["tests/"]},
}


@pytest.mark.smoke
def test_append_load_and_consume_roundtrip(tmp_path: Path) -> None:
    """A recorded green verdict with intact evidence is consumable by closure."""

    row = _row(tmp_path)
    gate_verdicts.append_verdict(tmp_path, row)
    rows = gate_verdicts.load_verdicts(tmp_path)
    assert rows == [row]
    consumed = gate_verdicts.consume_green(
        tmp_path, gate="crawler-smoke", closure_sha=_CLOSURE, require_armed_floor=True
    )
    assert consumed["log_sha256"] == row["log_sha256"]


@pytest.mark.smoke
def test_closed_schema_refuses_unknown_producer_status_and_legacy(tmp_path: Path) -> None:
    """Unknown fields, foreign statuses, green-with-failures, and v1 rows refuse."""

    with pytest.raises(gate_verdicts.GateVerdictError, match="unknown"):
        gate_verdicts.append_verdict(tmp_path, {**_row(tmp_path), "operator_note": "ok"})
    with pytest.raises(gate_verdicts.GateVerdictError, match="closed vocabulary"):
        gate_verdicts.append_verdict(tmp_path, {**_row(tmp_path), "status": "cached"})
    bad = _row(tmp_path)
    bad["node_sets"] = {**bad["node_sets"], "failed": ["t::red"]}
    with pytest.raises(gate_verdicts.GateVerdictError, match="green verdict cannot"):
        gate_verdicts.append_verdict(tmp_path, bad)
    legacy = {**_row(tmp_path), "schema_version": "menagerie.crawler.gate-verdict.v1"}
    with pytest.raises(gate_verdicts.GateVerdictError, match="gate-verdict-legacy"):
        gate_verdicts.append_verdict(tmp_path, legacy)
    with pytest.raises(gate_verdicts.GateVerdictError, match="closure_sha"):
        gate_verdicts.append_verdict(tmp_path, {**_row(tmp_path), "closure_sha": "short"})


@pytest.mark.smoke
def test_prior_era_whole_tree_rows_are_segregated_never_consumed(tmp_path: Path) -> None:
    """A v2 (tree-keyed) row is counted as legacy, returned to no consumer."""

    gate_verdicts.append_verdict(tmp_path, _row(tmp_path))
    ledger = tmp_path / gate_verdicts.VERDICTS_RELPATH
    v2 = {
        "schema_version": "menagerie.crawler.gate-verdict.v2",
        "gate": "crawler-smoke",
        "status": "green",
        "tree_sha_start": _TREE,
        "tree_sha_end": _TREE,
    }
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(v2) + "\n")
    rows, legacy = gate_verdicts.load_ledger(tmp_path)
    assert len(rows) == 1
    assert legacy == 1
    assert rows[0]["schema_version"] == gate_verdicts.GATE_VERDICT_SCHEMA_VERSION
    # A malformed non-legacy row still refuses the whole load typed.
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema_version": "made-up"}) + "\n")
    with pytest.raises(gate_verdicts.GateVerdictError, match="gate-verdict-invalid"):
        gate_verdicts.load_ledger(tmp_path)


@pytest.mark.smoke
def test_truncated_gate_log_refuses_verdict(tmp_path: Path) -> None:
    """THE NAMED TRIPWIRE: truncated or lost full evidence refuses the verdict."""

    row = _row(tmp_path)
    gate_verdicts.append_verdict(tmp_path, row)
    blob = tmp_path / gate_verdicts.GATE_LOGS_RELDIR / f"{row['log_sha256']}.log"
    blob.write_bytes(blob.read_bytes()[:3])
    with pytest.raises(gate_verdicts.GateVerdictError, match="gate-log-truncated"):
        gate_verdicts.consume_green(tmp_path, gate="crawler-smoke", closure_sha=_CLOSURE)
    blob.unlink()
    with pytest.raises(gate_verdicts.GateVerdictError, match="gate-log-missing"):
        gate_verdicts.consume_green(tmp_path, gate="crawler-smoke", closure_sha=_CLOSURE)


@pytest.mark.smoke
def test_newest_row_wins_and_red_refuses(tmp_path: Path) -> None:
    """A red recorded AFTER a green refuses until a fresh green supersedes it."""

    gate_verdicts.append_verdict(tmp_path, _row(tmp_path, status="green"))
    gate_verdicts.append_verdict(tmp_path, _row(tmp_path, status="red"))
    with pytest.raises(gate_verdicts.GateVerdictError, match="gate-verdict-red"):
        gate_verdicts.consume_green(tmp_path, gate="crawler-smoke", closure_sha=_CLOSURE)
    gate_verdicts.append_verdict(tmp_path, _row(tmp_path, status="green"))
    assert (
        gate_verdicts.consume_green(tmp_path, gate="crawler-smoke", closure_sha=_CLOSURE)["status"]
        == "green"
    )


@pytest.mark.smoke
def test_stale_closure_foreign_gate_and_drift_refuse(tmp_path: Path) -> None:
    """Another closure is stale; an unknown gate is missing; tree drift refuses."""

    gate_verdicts.append_verdict(tmp_path, _row(tmp_path))
    with pytest.raises(gate_verdicts.GateVerdictError, match="gate-verdict-stale"):
        gate_verdicts.consume_green(tmp_path, gate="crawler-smoke", closure_sha="b" * 64)
    with pytest.raises(gate_verdicts.GateVerdictError, match="gate-verdict-missing"):
        gate_verdicts.consume_green(tmp_path, gate="never-ran", closure_sha=_CLOSURE)
    gate_verdicts.append_verdict(tmp_path, _row(tmp_path, gate="drifted", tree_sha_end="9" * 40))
    with pytest.raises(gate_verdicts.GateVerdictError, match="tree drift"):
        gate_verdicts.consume_green(tmp_path, gate="drifted", closure_sha=_CLOSURE)


@pytest.mark.smoke
def test_unarmed_floor_refuses_when_required(tmp_path: Path) -> None:
    """An unarmed-floor green cannot satisfy an armed-floor requirement."""

    gate_verdicts.append_verdict(tmp_path, _row(tmp_path, execution_floor_armed=False))
    with pytest.raises(gate_verdicts.GateVerdictError, match="floor-unarmed"):
        gate_verdicts.consume_green(
            tmp_path, gate="crawler-smoke", closure_sha=_CLOSURE, require_armed_floor=True
        )


@pytest.mark.smoke
def test_secret_redaction_in_stored_logs(tmp_path: Path) -> None:
    """Secret-shaped substrings never reach the durable evidence store."""

    output = (
        "ANTHROPIC_API_KEY=" + "sk-" + "ant-" + "abc123def456ghi789" + "\n"
        "token " + "AK" + "IA" + "1234567890ABCDEF" + " used\n"
        "Authorization: Bearer abcdefghijklmnop1234\nplain line\n"
    )
    digest = gate_verdicts.store_gate_log(tmp_path, output)
    stored = (tmp_path / gate_verdicts.GATE_LOGS_RELDIR / f"{digest}.log").read_text(
        encoding="utf-8"
    )
    assert "sk-ant-abc123" not in stored
    assert "AK" + "IA" + "1234567890ABCDEF" not in stored
    assert "Bearer abcdefghijklmnop1234" not in stored
    assert "plain line" in stored


# --- closure keying over a real git tree ---------------------------------------


@pytest.mark.smoke
def test_closure_sha_moves_only_when_the_gate_inputs_move(tmp_path: Path) -> None:
    """A commit outside a closure keeps its sha; a commit inside changes it."""

    repo = _fixture_repo(tmp_path, _fixture_manifest(_TWO_GATES))
    manifest = gate_closure.load_manifest(repo)
    before = gate_closure.all_closure_shas(repo, manifest)
    _commit_file(repo, "docs/guide.md", "# guide v2\n")
    after_docs = gate_closure.all_closure_shas(repo, manifest)
    assert after_docs == before, "a docs-only commit moved a closure"
    _commit_file(repo, "tests/test_b.py", "def test_b():\n    pass\n")
    after_tests = gate_closure.all_closure_shas(repo, manifest)
    assert after_tests["src-gate"] == before["src-gate"]
    assert after_tests["tests-gate"] != before["tests-gate"]
    moved = gate_closure.moved_gates(repo, manifest, base_ref="HEAD~2")
    assert moved == {"src-gate": False, "tests-gate": True}
    # Editing the manifest itself (implicit closure) moves EVERY gate.
    payload = json.loads((repo / gate_closure.MANIFEST_RELPATH).read_text(encoding="utf-8"))
    payload["comment"] = "changed"
    _commit_file(
        repo, gate_closure.MANIFEST_RELPATH.as_posix(), json.dumps(payload, indent=2) + "\n"
    )
    after_manifest = gate_closure.all_closure_shas(repo, manifest)
    assert all(after_manifest[name] != after_tests[name] for name in after_tests)


@pytest.mark.smoke
def test_required_gates_consume_by_closure_across_unrelated_commits(tmp_path: Path) -> None:
    """The invocation arm accepts CI-shaped rows for the SAME bytes on a later commit."""

    repo = _fixture_repo(tmp_path, _fixture_manifest(_TWO_GATES))
    manifest = gate_closure.load_manifest(repo)
    with pytest.raises(gate_verdicts.GateVerdictError, match="gate-verdict-missing"):
        gate_verdicts.consume_required_gates_green(repo)
    for name in manifest.required_for_invocation:
        log = gate_verdicts.store_gate_log(repo, f"{name} green\n")
        gate_verdicts.append_verdict(
            repo,
            gate_verdicts.build_verdict(
                gate=name,
                status="green",
                exit_code=0,
                closure_sha=gate_closure.closure_sha(repo, manifest, name),
                tree_sha_start=gate_closure.tree_sha(repo),
                tree_sha_end=gate_closure.tree_sha(repo),
                commit_sha=gate_closure.commit_sha(repo),
                env_identity={"host": "ci-runner"},
                argv=["fixture"],
                execution_floor_armed=True,
                node_sets={"passed": ["t::ok"], "failed": [], "skipped": [], "xfailed": []},
                log_sha256=log,
            ),
        )
    consumed = gate_verdicts.consume_required_gates_green(repo)
    assert set(consumed) == {"src-gate", "tests-gate"}
    # A docs-only commit on top: every verdict stays consumable (no re-run).
    _commit_file(repo, "docs/guide.md", "# guide v2\n")
    assert set(gate_verdicts.consume_required_gates_green(repo)) == {"src-gate", "tests-gate"}
    # A src commit: ONLY src-gate refuses, typed stale, naming the moved inputs.
    _commit_file(repo, "src/a.py", "A = 2\n")
    with pytest.raises(gate_verdicts.GateVerdictError, match="gate-verdict-stale.*src-gate"):
        gate_verdicts.consume_required_gates_green(repo)
    status = {entry["gate"]: entry["consumable"] for entry in gate_verdicts.verdict_status(repo)}
    assert status == {"src-gate": False, "tests-gate": True}


@pytest.mark.smoke
def test_closure_census_refuses_an_uncovered_source_path(tmp_path: Path) -> None:
    """A tracked source file outside every closure is a census failure."""

    repo = _fixture_repo(tmp_path, _fixture_manifest(_TWO_GATES))
    manifest = gate_closure.load_manifest(repo)
    blobs = gate_closure.tracked_blobs(repo)
    assert gate_closure.uncovered_source_paths(manifest, blobs) == ("pyproject.toml",)
    narrowed = _fixture_manifest({"src-gate": {"input_closure": ["src/a.py"]}})
    assert gate_closure.uncovered_source_paths(gate_closure.parse_manifest(narrowed), blobs) == (
        "pyproject.toml",
        "tests/test_a.py",
    )
    excluded = _fixture_manifest(
        {"all": {"input_closure": ["src/", "tests/", "pyproject.toml"], "exclude": ["tests/"]}}
    )
    assert gate_closure.uncovered_source_paths(gate_closure.parse_manifest(excluded), blobs) == (
        "tests/test_a.py",
    )


@pytest.mark.smoke
def test_manifest_schema_is_closed_and_prior_era_refuses_typed(tmp_path: Path) -> None:
    """v2 manifests, unknown fields, unknown required gates, empty closures refuse."""

    with pytest.raises(gate_closure.GateClosureError, match="required-gate-set-legacy"):
        gate_closure.parse_manifest({"schema_version": "menagerie.crawler.required-gate-set.v2"})
    good = _fixture_manifest(_TWO_GATES)
    gate_closure.parse_manifest(good)
    with pytest.raises(gate_closure.GateClosureError, match="unknown="):
        gate_closure.parse_manifest({**good, "operator_override": True})
    with pytest.raises(gate_closure.GateClosureError, match="unknown gates"):
        gate_closure.parse_manifest({**good, "required_for_invocation": ["ghost"]})
    with pytest.raises(gate_closure.GateClosureError, match="empty closure"):
        gate_closure.parse_manifest(_fixture_manifest({"g": {"input_closure": []}}))
    repo = _fixture_repo(tmp_path, good)
    manifest = gate_closure.load_manifest(repo)
    with pytest.raises(gate_closure.GateClosureError, match="unknown gate"):
        gate_closure.closure_sha(repo, manifest, "ghost")


@pytest.mark.smoke
def test_dirty_closure_paths_name_edits_and_untracked_files(tmp_path: Path) -> None:
    """The runner's dirty check sees edits AND untracked files inside a closure only."""

    repo = _fixture_repo(tmp_path, _fixture_manifest(_TWO_GATES))
    manifest = gate_closure.load_manifest(repo)
    assert gate_closure.closure_dirty_paths(repo, manifest, "src-gate") == ()
    (repo / "docs" / "scratch.md").write_text("x\n", encoding="utf-8")
    assert gate_closure.closure_dirty_paths(repo, manifest, "src-gate") == ()
    (repo / "src" / "untracked.py").write_text("B = 2\n", encoding="utf-8")
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n", encoding="utf-8")
    assert gate_closure.closure_dirty_paths(repo, manifest, "src-gate") == ("src/untracked.py",)
    assert gate_closure.closure_dirty_paths(repo, manifest, "tests-gate") == ("tests/test_a.py",)


@pytest.mark.smoke
def test_committed_manifest_covers_every_tracked_source_path() -> None:
    """THE CLOSURE CENSUS: nothing escapes certification by living outside every closure."""

    manifest = gate_closure.load_manifest(_REPO_ROOT)
    blobs = gate_closure.tracked_blobs(_REPO_ROOT)
    uncovered = gate_closure.uncovered_source_paths(manifest, blobs)
    assert uncovered == (), (
        "tracked source paths inside NO required gate's input_closure (extend a "
        f"closure in required_gate_set.json in the same change): {uncovered}"
    )
    # Every declared closure selects real files, every required gate exists,
    # and the crawler-smoke gate keeps its armed-floor requirement.
    for name in manifest.gates:
        assert gate_closure.closure_paths(manifest, manifest.gate(name), blobs)
    assert set(manifest.required_for_invocation) >= {
        "collection",
        "crawler-smoke",
        "crawler-full",
        "crawler-rigs",
        "fast-tier",
    }
    assert manifest.gate("crawler-smoke").requires_armed_floor is True
    assert manifest.gate("collection").always is True


@pytest.mark.smoke
def test_every_pytest_gate_bounds_each_test_twice() -> None:
    """A hung test fails fast and named; it never wedges a gate (fast-tier wedge, 2026-09-02).

    Both layers are required on every pytest-run gate: pytest-timeout's signal method
    (names a Python-level hang as a failed node and lets the run continue to a complete
    junit) and the faulthandler exit backstop at a strictly later deadline (names a
    native-level hang the signal cannot interrupt, then ends the process so the runner
    records a red verdict instead of waiting for hours).
    """

    manifest = gate_closure.load_manifest(_REPO_ROOT)
    pytest_gates = [
        gate for gate in manifest.gates.values() if gate.command and "pytest" in gate.command
    ]
    assert {gate.name for gate in pytest_gates} >= {"crawler-smoke", "crawler-full", "fast-tier"}
    for gate in pytest_gates:
        argv = list(gate.command or ())
        per_test = [token for token in argv if token.startswith("--timeout=")]
        assert len(per_test) == 1, f"gate {gate.name!r} lacks exactly one --timeout=<seconds>"
        per_test_seconds = float(per_test[0].split("=", 1)[1])
        assert per_test_seconds > 0, gate.name
        assert "--timeout-method=signal" in argv, gate.name
        overrides = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "-o"]
        backstop = [item for item in overrides if item.startswith("faulthandler_timeout=")]
        assert len(backstop) == 1, f"gate {gate.name!r} lacks the faulthandler backstop"
        assert float(backstop[0].split("=", 1)[1]) > per_test_seconds, gate.name
        assert "faulthandler_exit_on_timeout=true" in overrides, gate.name


@pytest.mark.smoke
def test_forbidden_override_census() -> None:
    """No operator-attestation override spelling exists anywhere reachable.

    Greps every argv parser, config schema, env read, and workflow under the
    crawler, its tools, and CI for the deleted override class. The census is
    spelling-based on purpose: reintroducing the flag under its own or an
    obvious variant name fails this node.
    """

    forbidden = re.compile(
        r"operator[-_]attest|attest[-_]green|OPERATOR_ATTEST|force[-_]green|"
        r"skip[-_]gates|override[-_]verdict",
        re.IGNORECASE,
    )
    roots = [
        _REPO_ROOT / "menagerie" / "crawler",
        _REPO_ROOT / "tools",
        _REPO_ROOT / ".github",
    ]
    hits: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".yml", ".yaml", ".json", ".sh"}:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if forbidden.search(line):
                    hits.append(f"{path.relative_to(_REPO_ROOT)}:{line_number}: {line.strip()}")
    assert hits == [], "forbidden override spellings reachable:\n" + "\n".join(hits)
