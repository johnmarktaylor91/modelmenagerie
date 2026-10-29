"""Cross-cutting static boundaries over the complete crawler source tree."""

from __future__ import annotations

import subprocess
import shutil
import sys
from pathlib import Path

import pytest

from menagerie.crawler.checkpoint import (
    NonAllowlistedPath,
    WrongCheckpointBranch,
    create_checkpoint_set,
)
from menagerie.crawler.mirrors import MirrorStore
from menagerie.crawler.policy import static_source_check
from menagerie.crawler.tests.test_slice_e_checkpoint_retro import RecordingGit
from .support import (
    fabricated_crawler_locks,
    repository_root,
    tracked_paths,
)


def _mirror_store(root: Path) -> MirrorStore:
    """Return isolated empty mirror classes for checkpoint refusal tests."""

    return MirrorStore(root / "public", root / "private", root / "local")


@pytest.mark.smoke
def test_worker_execution_sources_hold_the_torchlens_import_ban() -> None:
    """Every worker/execution source passes the shared static policy check."""

    crawler_root = repository_root() / "menagerie" / "crawler"
    execution_paths = [
        crawler_root / name
        for name in (
            "worker.py",
            "worker_supervisor.py",
            "frameworks.py",
            "recipe.py",
            "standard_inputs.py",
        )
    ]
    for optional_root in ("adapters", "ports", "patches"):
        execution_paths.extend((crawler_root / optional_root).rglob("*.py"))
    for path in execution_paths:
        static_source_check(path)


@pytest.mark.smoke
def test_authority_static_import_graph_never_reaches_catalog_side() -> None:
    """authority.py's transitive static import graph stays worker-sealed.

    ``authority.py`` is a sealed worker-closure member: the release gate's
    ``_validate_static_import_closure_v3`` refuses any static import (the
    scanner follows function-scoped imports too) that escapes the sealed
    environment plus the exact member inventory. The K2 regression (RC4) was a
    lazy ``from menagerie.crawler.rung_staging import ...`` inside authority.py
    whose import graph reaches catalog-side top-level ``menagerie.identity``.

    This pin re-walks authority.py's static import graph with the SAME
    production scanner helpers the validator uses and asserts it never reaches
    ``menagerie.crawler.rung_staging`` nor ANY top-level catalog-side module
    (anything under ``menagerie/`` outside ``menagerie/crawler/``, except the
    ``menagerie/__init__.py`` namespace initializer) -- without needing the
    real sealed environment.
    """

    from menagerie.crawler.authority import _import_names, _module_files

    repo_root = repository_root()
    crawler_root = repo_root / "menagerie" / "crawler"
    roots = (repo_root,)
    namespace_init = (repo_root / "menagerie" / "__init__.py").resolve()

    def allowed(path: Path) -> bool:
        """Return whether one resolved import target is worker-closure-eligible."""

        if path == namespace_init:
            return True
        if not path.is_relative_to(repo_root / "menagerie"):
            return True  # outside the menagerie tree entirely (stdlib/site deps)
        return path.is_relative_to(crawler_root) and path.name != "rung_staging.py"

    start = (crawler_root / "authority.py").resolve()
    chain: dict[Path, Path] = {}
    pending = [start]
    seen = {start}
    violations: list[str] = []
    while pending:
        source = pending.pop()
        if source.suffix != ".py":
            continue
        for module_name in _import_names(source, roots):
            for resolved in _module_files(module_name, roots):
                if resolved in seen:
                    continue
                seen.add(resolved)
                chain[resolved] = source
                if not allowed(resolved):
                    hops = [resolved]
                    while hops[-1] in chain:
                        hops.append(chain[hops[-1]])
                    rendered = " <- ".join(str(hop.relative_to(repo_root)) for hop in hops)
                    violations.append(f"{module_name}: {rendered}")
                    continue
                if resolved.is_relative_to(repo_root / "menagerie"):
                    pending.append(resolved)
    assert not violations, (
        "authority.py's static import graph escaped the sealed worker closure "
        "(fix the dependency direction, never the validator): " + "; ".join(violations)
    )


@pytest.mark.smoke
def test_checkpoint_refuses_wrong_branch_and_nonallowlisted_paths(tmp_path: Path) -> None:
    """The cross-cutting staging boundary rejects both independent authority failures."""

    common = {
        "ledger_paths": [],
        "derived_view_checks": [],
        "public_artifacts": [],
        "mirrors": _mirror_store(tmp_path),
        "git_runner": RecordingGit(),
    }
    with pytest.raises(WrongCheckpointBranch):
        create_checkpoint_set(tmp_path, [], branch="main", **common)
    with pytest.raises(NonAllowlistedPath):
        create_checkpoint_set(
            tmp_path,
            [Path(".crawl-local/runtime.json")],
            branch="menagerie/crawler-pipeline",
            **common,
        )


@pytest.mark.rig
def test_git_tree_ships_no_runtime_solve_or_secret_artifacts(tmp_path: Path) -> None:
    """Only source/spec facts are tracked; runtime, fabricated locks, and secrets stay local."""

    repo_root = repository_root()
    tracked = tracked_paths(repo_root)
    assert all(".crawl-local" not in path.parts for path in tracked)
    # Runtime solve output never ships; the genuine target-solved release families ship on
    # purpose, because a release proof must consume fixed committed bytes. The helper draws
    # that line by attestation, so an unprovenanced or incomplete family still fails here.
    assert fabricated_crawler_locks(repo_root) == ()
    crawler_paths = [path for path in tracked if path.parts[:2] == ("menagerie", "crawler")]
    secret_parts = {"credential", "credentials", "secret", "secrets", "token"}
    assert all(
        not secret_parts.intersection(part.lower() for part in path.parts) for path in crawler_paths
    )
    baseline = tmp_path / ".secrets.baseline"
    shutil.copyfile(repo_root / ".secrets.baseline", baseline)
    # Invoke the scanner through this interpreter rather than a bare PATH name: the tool is
    # a declared dev dependency, so an installed environment must be enough to run the
    # check. A PATH lookup turns "the scanner is missing" into a host accident.
    secret_scan = subprocess.run(
        [sys.executable, "-m", "detect_secrets", "scan", "--baseline", str(baseline)],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert secret_scan.returncode == 0, secret_scan.stdout + secret_scan.stderr
