"""Run one required gate and append its closure-keyed verdict (RC-3).

The one runner CI and the campaign host share. It replaces the worktree train
runner (``tools/crawler_train_runner.py``, retired with the launch
certificate): there is no train, no certificate kind, and no cache -- a gate
either has a consumable green verdict for the running tree's closure or it
does not, and the verdict ledger is the only state.

Subcommands
-----------
``run --gate G``
    Refuse if any path inside G's closure is dirty (edited, staged, deleted,
    or untracked -- the verdict would not describe what ran). With
    ``--base REF``, skip (exit 0, nothing recorded) when G's closure sha at
    HEAD equals the sha at REF and G is not ``always``; that is how CI re-runs
    only the gates whose inputs moved. Otherwise run G's manifest command
    with its declared environment, tee the FULL output to the terminal and to
    a content-addressed evidence blob, read the junit node sets, observe the
    tree again, and append one verdict row. Exit with the command's status.
``record-attestation --gate G --attestation PATH``
    Record a verdict for an attestation-recorded gate (round21) from the
    attestation JSON the release conftest minted.
``closure [--gate G] [--tree REF]``
    Print closure shas.
``plan --base REF``
    Print which gates moved between REF and HEAD.
``status [--tree REF]``
    Per-gate consumability for one tree.
``import --from-dir DIR``
    Merge verdict rows minted elsewhere (CI artifacts downloaded with
    ``gh run download``): every row's closure sha is RECOMPUTED from its
    commit in the local object store and must match, its evidence blob must
    be present with a matching digest, and duplicates are skipped. A row
    whose commit is not fetched locally refuses typed; nothing a producer
    wrote steers acceptance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from menagerie.crawler.tools import gate_closure, gate_verdicts

EXIT_REFUSED = 78
EXIT_USAGE = 64
JUNIT_RELDIR = Path(".crawl-local/junit")


def _junit_outcome(case: ElementTree.Element) -> str:
    """Classify one junit testcase as passed, failed, skipped, or xfailed."""

    if case.find("failure") is not None or case.find("error") is not None:
        return "failed"
    skipped = case.find("skipped")
    if skipped is None:
        return "passed"
    return "xfailed" if skipped.get("type") == "pytest.xfail" else "skipped"


def _junit_node_sets(path: Path) -> dict[str, list[str]]:
    """Parse exact per-outcome node sets from one junit-xml report."""

    node_sets: dict[str, set[str]] = {
        key: set() for key in ("passed", "failed", "skipped", "xfailed")
    }
    if path.is_file():
        for case in ElementTree.parse(path).getroot().iter("testcase"):
            classname = case.get("classname", "")
            module = classname.replace(".", "/") + ".py" if classname else ""
            node_sets[_junit_outcome(case)].add(f"{module}::{case.get('name', '')}")
    return {key: sorted(value) for key, value in node_sets.items()}


def _refuse(message: str) -> int:
    print(f"gate-runner: REFUSED: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _resolvable(repo_root: Path, ref: str) -> bool:
    completed = subprocess.run(
        ("git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _run_streaming(argv: Sequence[str], *, cwd: Path, env: dict[str, str]) -> tuple[int, str]:
    """Run ``argv`` teeing combined output to stdout; return (exit, full output)."""

    chunks: list[str] = []
    with subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            chunks.append(line)
        process.wait()
        return process.returncode, "".join(chunks)


def _closure_unchanged_since(
    repo_root: Path,
    manifest: gate_closure.GateManifest,
    gate_name: str,
    *,
    head_closure: str,
    base_ref: str,
) -> bool:
    """True when the gate's closure sha at ``base_ref`` equals ``head_closure``."""

    if not _resolvable(repo_root, base_ref):
        print(
            f"gate-runner: base {base_ref!r} is not resolvable locally; running "
            f"{gate_name} unconditionally"
        )
        return False
    base_closure = gate_closure.closure_sha(repo_root, manifest, gate_name, tree_ref=base_ref)
    if base_closure != head_closure:
        return False
    print(
        f"gate-runner: {gate_name} closure unchanged since {base_ref} "
        f"({head_closure[:16]}); nothing to re-run, nothing recorded"
    )
    return True


def _junit_target(repo_root: Path, gate: gate_closure.GateSpec) -> Path | None:
    """Return a fresh junit path for a junit gate (a stale report is removed first)."""

    if not gate.junit:
        return None
    junit_path = repo_root / JUNIT_RELDIR / f"{gate.name}.xml"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    if junit_path.exists():
        junit_path.unlink()
    return junit_path


def _presence(env: Mapping[str, str], name: str) -> str:
    return "present" if env.get(name) else "absent"


def run_gate(
    repo_root: Path,
    *,
    gate_name: str,
    python: str,
    base_ref: str | None,
    force: bool,
) -> int:
    """Run one gate and append its verdict; see the module docstring."""

    manifest = gate_closure.load_manifest(repo_root)
    gate = manifest.gate(gate_name)
    if gate.is_attestation:
        return _refuse(f"gate {gate_name!r} is attestation-recorded; use record-attestation")
    dirty = gate_closure.closure_dirty_paths(repo_root, manifest, gate_name)
    if dirty:
        return _refuse(
            f"working tree is dirty inside gate {gate_name!r}'s closure; a verdict "
            "must describe committed bytes. Commit or clean: " + ", ".join(dirty[:20])
        )
    head_closure = gate_closure.closure_sha(repo_root, manifest, gate_name)
    if (
        base_ref is not None
        and not force
        and not gate.always
        and _closure_unchanged_since(
            repo_root, manifest, gate_name, head_closure=head_closure, base_ref=base_ref
        )
    ):
        return 0
    junit_path = _junit_target(repo_root, gate)
    argv = gate_closure.render_command(gate, python=python, junit=junit_path)
    env = {**os.environ, "PYTHONPATH": str(repo_root), **gate.env}
    tree_start = gate_closure.tree_sha(repo_root)
    commit = gate_closure.commit_sha(repo_root)
    print(f"gate-runner: running {gate_name} on closure {head_closure[:16]} (commit {commit[:12]})")
    print("gate-runner: $ " + " ".join(argv))
    # Disclose the environment triple every crawler red is adjudicated against:
    # an unset prefix makes the real-environment fixture fall back to the
    # ambient conda prefix (a different family, with its own deterministic reds).
    print(
        "gate-runner: env "
        f"MENAGERIE_REAL_ENV_PREFIX={_presence(env, 'MENAGERIE_REAL_ENV_PREFIX')} "
        f"CONDA_PREFIX={_presence(env, 'CONDA_PREFIX')} "
        f"MENAGERIE_SMOKE_EXECUTION_FLOOR={env.get('MENAGERIE_SMOKE_EXECUTION_FLOOR', 'unset')}"
    )
    exit_code, output = _run_streaming(argv, cwd=repo_root, env=env)
    tree_end = gate_closure.tree_sha(repo_root)
    if junit_path is not None and not junit_path.is_file():
        print(
            "gate-runner: WARNING no junit report was written: the test process ended "
            "before session finish (a native-level hang terminated by the faulthandler "
            "backstop, or an external kill); the hung node is named in the stack dump "
            "inside the stored log",
            file=sys.stderr,
        )
    node_sets = (
        _junit_node_sets(junit_path)
        if junit_path is not None
        else {key: [] for key in ("passed", "failed", "skipped", "xfailed")}
    )
    status = "green" if exit_code == 0 and not node_sets["failed"] else "red"
    log_sha256 = gate_verdicts.store_gate_log(repo_root, output)
    row = gate_verdicts.build_verdict(
        gate=gate_name,
        status=status,
        exit_code=exit_code,
        closure_sha=head_closure,
        tree_sha_start=tree_start,
        tree_sha_end=tree_end,
        commit_sha=commit,
        env_identity=gate_verdicts.environment_identity(python),
        argv=argv,
        execution_floor_armed=env.get("MENAGERIE_SMOKE_EXECUTION_FLOOR") == "1",
        node_sets=node_sets,
        log_sha256=log_sha256,
    )
    gate_verdicts.append_verdict(repo_root, row)
    print(
        f"gate-runner: recorded {status} verdict for {gate_name} "
        f"(passed={len(node_sets['passed'])} failed={len(node_sets['failed'])} "
        f"skipped={len(node_sets['skipped'])} xfailed={len(node_sets['xfailed'])}, "
        f"evidence {log_sha256[:16]})"
    )
    if tree_start != tree_end:
        print(
            "gate-runner: WARNING tree drifted during the run; the recorded verdict "
            "refuses at consumption",
            file=sys.stderr,
        )
    return exit_code


def record_attestation(repo_root: Path, *, gate_name: str, attestation: Path) -> int:
    """Record one verdict row from a release attestation artifact."""

    manifest = gate_closure.load_manifest(repo_root)
    gate = manifest.gate(gate_name)
    if not gate.is_attestation:
        return _refuse(f"gate {gate_name!r} is command-run; use run")
    dirty = gate_closure.closure_dirty_paths(repo_root, manifest, gate_name)
    if dirty:
        return _refuse(
            f"working tree is dirty inside gate {gate_name!r}'s closure: " + ", ".join(dirty[:20])
        )
    try:
        payload = json.loads(attestation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _refuse(f"attestation unreadable: {exc}")
    if not isinstance(payload, dict):
        return _refuse("attestation is not a JSON object")
    status = "green" if payload.get("status") == "passed" else "red"
    tree = gate_closure.tree_sha(repo_root)
    log_sha256 = gate_verdicts.store_gate_log(
        repo_root, json.dumps(payload, indent=2, sort_keys=True)
    )
    row = gate_verdicts.build_verdict(
        gate=gate_name,
        status=status,
        exit_code=0 if status == "green" else 1,
        closure_sha=gate_closure.closure_sha(repo_root, manifest, gate_name),
        tree_sha_start=tree,
        tree_sha_end=tree,
        commit_sha=gate_closure.commit_sha(repo_root),
        env_identity={
            **gate_verdicts.environment_identity(),
            "environment_content_digest": str(payload.get("environment_content_digest")),
            "lock_sha256": str(payload.get("lock_sha256")),
        },
        argv=["record-attestation", attestation.as_posix()],
        execution_floor_armed=bool((payload.get("execution_floor") or {}).get("armed")),
        node_sets={
            "passed": list(payload.get("passed_nodes", [])),
            "failed": list(payload.get("failed_nodes", [])),
            "skipped": list(payload.get("skipped_nodes", [])),
            "xfailed": list(payload.get("xfailed_nodes", [])),
        },
        log_sha256=log_sha256,
    )
    gate_verdicts.append_verdict(repo_root, row)
    print(f"gate-runner: recorded {status} verdict for {gate_name} from {attestation}")
    return 0 if status == "green" else 1


def _refuse_import(ledger: Path, line_number: int, message: str) -> str:
    print(f"gate-runner: import refused {ledger}:{line_number}: {message}", file=sys.stderr)
    return "refused"


def _import_ledger_row(
    repo_root: Path,
    manifest: gate_closure.GateManifest,
    *,
    ledger: Path,
    line_number: int,
    line: str,
    existing_keys: set[tuple[Any, ...]],
) -> str:
    """Import one ledger line; return ``imported``, ``skipped``, or ``refused``."""

    try:
        row = json.loads(line)
        gate_verdicts._validate_row(row, context=f"{ledger}:{line_number}")
    except (json.JSONDecodeError, gate_verdicts.GateVerdictError) as exc:
        return _refuse_import(ledger, line_number, str(exc))
    key = (row["gate"], row["closure_sha"], row["log_sha256"], row["minted_at"])
    if key in existing_keys:
        return "skipped"
    if not _resolvable(repo_root, row["commit_sha"]):
        return _refuse_import(
            ledger,
            line_number,
            f"commit {row['commit_sha']} is not in the local object store (fetch it first)",
        )
    try:
        recomputed = gate_closure.closure_sha(
            repo_root, manifest, row["gate"], tree_ref=row["commit_sha"]
        )
    except gate_closure.GateClosureError as exc:
        return _refuse_import(ledger, line_number, str(exc))
    if recomputed != row["closure_sha"]:
        return _refuse_import(
            ledger,
            line_number,
            f"closure_sha {row['closure_sha']} does not recompute from commit "
            f"{row['commit_sha']} ({recomputed}); the manifest or the closure moved",
        )
    logs_dir = ledger.parent / "gate-logs"
    blob = logs_dir / f"{row['log_sha256']}.log"
    if not blob.is_file() or gate_verdicts._sha256_hex(blob.read_bytes()) != row["log_sha256"]:
        return _refuse_import(
            ledger,
            line_number,
            f"evidence blob {row['log_sha256']} missing or corrupt under {logs_dir}",
        )
    destination = repo_root / gate_verdicts.GATE_LOGS_RELDIR / blob.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(blob, destination)
    gate_verdicts.append_verdict(repo_root, row)
    existing_keys.add(key)
    return "imported"


def import_verdicts(repo_root: Path, *, from_dir: Path) -> int:
    """Merge externally minted rows after recomputing each closure locally."""

    manifest = gate_closure.load_manifest(repo_root)
    existing, _legacy = gate_verdicts.load_ledger(repo_root)
    existing_keys: set[tuple[Any, ...]] = {
        (row["gate"], row["closure_sha"], row["log_sha256"], row["minted_at"]) for row in existing
    }
    ledgers = sorted(from_dir.rglob("gate-verdicts.jsonl"))
    if not ledgers:
        return _refuse(f"no gate-verdicts.jsonl under {from_dir}")
    outcomes: Counter[str] = Counter()
    for ledger in ledgers:
        lines = ledger.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            outcome = _import_ledger_row(
                repo_root,
                manifest,
                ledger=ledger,
                line_number=line_number,
                line=line,
                existing_keys=existing_keys,
            )
            outcomes[outcome] += 1
    print(
        f"gate-runner: imported {outcomes['imported']} row(s), skipped "
        f"{outcomes['skipped']} duplicate(s), refused {outcomes['refused']}"
    )
    return 0 if outcomes["refused"] == 0 else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run one gate and record its verdict")
    run.add_argument("--gate", required=True)
    run.add_argument("--python", default=sys.executable)
    run.add_argument(
        "--base",
        default=None,
        help="skip (exit 0, nothing recorded) when the gate's closure is unchanged since this ref",
    )
    run.add_argument("--force", action="store_true", help="run even when the closure is unchanged")

    record = subparsers.add_parser("record-attestation", help="record an attestation-gate verdict")
    record.add_argument("--gate", required=True)
    record.add_argument("--attestation", type=Path, required=True)

    closure = subparsers.add_parser("closure", help="print closure shas")
    closure.add_argument("--gate", default=None)
    closure.add_argument("--tree", default="HEAD")

    plan = subparsers.add_parser("plan", help="which gates moved between --base and HEAD")
    plan.add_argument("--base", required=True)

    status = subparsers.add_parser("status", help="per-gate consumability for one tree")
    status.add_argument("--tree", default="HEAD")

    imp = subparsers.add_parser("import", help="merge verdict rows minted elsewhere")
    imp.add_argument("--from-dir", type=Path, required=True)
    return parser


def _cmd_run(repo_root: Path, args: argparse.Namespace) -> int:
    return run_gate(
        repo_root,
        gate_name=args.gate,
        python=args.python,
        base_ref=args.base,
        force=args.force,
    )


def _cmd_record_attestation(repo_root: Path, args: argparse.Namespace) -> int:
    return record_attestation(repo_root, gate_name=args.gate, attestation=args.attestation)


def _cmd_closure(repo_root: Path, args: argparse.Namespace) -> int:
    manifest = gate_closure.load_manifest(repo_root)
    if args.gate:
        print(gate_closure.closure_sha(repo_root, manifest, args.gate, tree_ref=args.tree))
    else:
        shas = gate_closure.all_closure_shas(repo_root, manifest, tree_ref=args.tree)
        print(json.dumps(shas, indent=2))
    return 0


def _cmd_plan(repo_root: Path, args: argparse.Namespace) -> int:
    manifest = gate_closure.load_manifest(repo_root)
    if not _resolvable(repo_root, args.base):
        return _refuse(f"base {args.base!r} is not resolvable")
    moved = gate_closure.moved_gates(repo_root, manifest, base_ref=args.base)
    print(json.dumps(moved, indent=2, sort_keys=True))
    return 0


def _cmd_status(repo_root: Path, args: argparse.Namespace) -> int:
    report: list[dict[str, Any]] = gate_verdicts.verdict_status(repo_root, tree_ref=args.tree)
    print(json.dumps(report, indent=2, sort_keys=True))
    required = [entry for entry in report if entry["required_for_invocation"]]
    return 0 if all(entry["consumable"] for entry in required) else 1


def _cmd_import(repo_root: Path, args: argparse.Namespace) -> int:
    return import_verdicts(repo_root, from_dir=args.from_dir.resolve())


_COMMANDS: dict[str, Callable[[Path, argparse.Namespace], int]] = {
    "run": _cmd_run,
    "record-attestation": _cmd_record_attestation,
    "closure": _cmd_closure,
    "plan": _cmd_plan,
    "status": _cmd_status,
    "import": _cmd_import,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    handler = _COMMANDS.get(args.command)
    if handler is None:
        return EXIT_USAGE
    try:
        return handler(repo_root, args)
    except (gate_closure.GateClosureError, gate_verdicts.GateVerdictError) as exc:
        return _refuse(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
