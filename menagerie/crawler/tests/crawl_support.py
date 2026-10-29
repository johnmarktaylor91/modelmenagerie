"""Test support for the RC-1 ``crawl`` invocation: a clean campaign repo with
green gate verdicts on its tree, and a fake-lane crawl driver over it.

Real producers throughout (no hand-built ledger shapes): the campaign
partition comes from ``emit_campaign_partitions``, the closure shas from
``gate_closure`` over the fixture repository's own git tree, the verdict rows
from ``gate_verdicts.build_verdict``/``append_verdict``, and the driver from
the canonical fake-lane builder of ``test_slice_f_driver``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.driver import CrawlerDriver, DriverPaths
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.intake import IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.partitioner import emit_campaign_partitions, load_campaign_bindings
from menagerie.crawler.tools import gate_closure, gate_verdicts

CAMPAIGN = "c1-mech"

#: The committed required-gate manifest the closure-green arm reads.
REQUIRED_GATE_SET = Path(__file__).resolve().parents[1] / "tools" / "required_gate_set.json"

#: A generous free-disk measurement for fixture runs (the arm is exercised
#: separately with an injected scarcity).
PLENTY_OF_DISK = 10**15


def records_root_of(repo_root: Path) -> Path:
    """Return the records root of one fixture repository.

    ``build_crawl_repo(layout="production")`` places the records inside the
    repository (``menagerie/crawler/records``, the layout the CLI requires);
    ``layout="fixture"`` places them BESIDE the repository (``../records``) so
    the reducer's production-only committed-registry validation stays out of
    the way of in-process fake environment lanes, exactly like every other
    fake-lane driver fixture in this corpus.
    """

    inside = repo_root / "menagerie" / "crawler" / "records"
    if inside.is_dir():
        return inside
    return repo_root.parent / "records"


def root_snapshot(repo_root: Path, campaign_id: str = CAMPAIGN) -> IntakeSnapshot:
    """Load the campaign's bound full-partition intake snapshot."""

    bindings = load_campaign_bindings(records_root_of(repo_root) / "campaigns.json")
    binding = next(b for b in bindings if b.spec.campaign_id == campaign_id)
    return load_intake_snapshot(records_root_of(repo_root) / binding.intake_path)


def _git(repo_root: Path, *argv: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo_root), *argv),
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "fixture",
            "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
            "GIT_COMMITTER_NAME": "fixture",
            "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        },
    )
    return completed.stdout.strip()


def head_tree_sha(repo_root: Path) -> str:
    """Return ``HEAD^{tree}`` of the fixture repository."""

    return _git(repo_root, "rev-parse", "HEAD^{tree}")


def mint_green_verdicts(repo_root: Path, *, gates: Optional[Sequence[str]] = None) -> str:
    """Append a consumable green verdict for every invocation-required gate on HEAD.

    Closure-keyed (RC-3): each row carries the gate's ``closure_sha`` computed
    by the real ``gate_closure`` machinery over the fixture repository's HEAD
    tree, exactly as ``gate_runner`` would record it.
    """

    manifest = gate_closure.load_manifest(repo_root)
    closures = gate_closure.all_closure_shas(repo_root, manifest)
    tree = head_tree_sha(repo_root)
    commit = _git(repo_root, "rev-parse", "HEAD")
    for gate in gates if gates is not None else manifest.required_for_invocation:
        log_sha256 = gate_verdicts.store_gate_log(repo_root, f"{gate}: fixture green run\n")
        row = gate_verdicts.build_verdict(
            gate=gate,
            status="green",
            exit_code=0,
            closure_sha=closures[gate],
            tree_sha_start=tree,
            tree_sha_end=tree,
            commit_sha=commit,
            env_identity={"fixture": gate},
            argv=["pytest", "-m", gate],
            execution_floor_armed=True,
            node_sets={"passed": [f"t::{gate}"], "failed": [], "skipped": [], "xfailed": []},
            log_sha256=log_sha256,
        )
        gate_verdicts.append_verdict(repo_root, row)
    return tree


def build_crawl_repo(
    tmp_path: Path,
    *,
    members: int = 8,
    zoo: str = "fixtures",
    extra_rows: Sequence[Mapping[str, Any]] = (),
    green_verdicts: bool = True,
    layout: str = "fixture",
) -> Path:
    """Materialize a clean campaign repository ready for a ``crawl`` invocation.

    ``members`` roster rows land in ``c1-mech`` (their zoo is none of the
    reserved literals); one filler row per sibling campaign keeps every frozen
    partition non-empty. The repository is a real git tree (the closure-green
    arm recomputes every gate's input closure from HEAD) carrying the committed
    required-gate manifest plus a ``pyproject.toml`` stub (the manifest's
    implicit closure; the ``fast-tier`` gate excludes ``menagerie/crawler/`` so
    without it that gate's closure would select zero tracked files) and, by
    default, a green verdict per invocation-required gate.
    """

    if layout not in {"fixture", "production"}:
        raise ValueError(f"unknown fixture layout {layout!r}")
    repo_root = tmp_path / "repo"
    records_root = (
        repo_root / "menagerie" / "crawler" / "records"
        if layout == "production"
        else tmp_path / "records"
    )
    records_root.mkdir(parents=True)
    rows = [{"name": f"Crawl{index}", "zoo": zoo, "variant": "base"} for index in range(members)]
    rows.extend(dict(row) for row in extra_rows)
    rows.extend(
        (
            {"name": "disco-one", "zoo": "discovered-pytorch", "variant": ""},
            {"name": "classic-one", "zoo": "unregistered-classics-pytorch", "variant": ""},
            {"name": "native-one", "zoo": "discovered-keras", "variant": ""},
        )
    )
    roster = tmp_path / "roster.jsonl"
    roster.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    emit_campaign_partitions(roster, records_root)
    (records_root / "models").mkdir(exist_ok=True)
    (records_root / "models" / "current-shard.jsonl").write_bytes(b"")
    tools_dir = repo_root / "menagerie" / "crawler" / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REQUIRED_GATE_SET, tools_dir / "required_gate_set.json")
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "crawl-fixture"\n', encoding="utf-8"
    )
    (repo_root / ".gitignore").write_text(".crawl-local/\n", encoding="utf-8")
    _git(repo_root, "init", "-q")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "-q", "-m", "fixture campaign")
    if green_verdicts:
        mint_green_verdicts(repo_root)
    return repo_root


def crawl_paths(repo_root: Path, snapshot: IntakeSnapshot) -> DriverPaths:
    """Return the production-shaped paths for one fixture repository."""

    records_root = records_root_of(repo_root)
    return DriverPaths(
        repo_root / ".crawl-local",
        snapshot.root,
        LedgerPaths(
            records_root / "models" / "current-shard.jsonl",
            records_root / "attempts" / "local.jsonl",
            records_root / "gates" / "current-shard.jsonl",
        ),
    )


def crawl_driver(
    tmp_path: Path,
    repo_root: Path,
    *,
    limit: Optional[int],
    author_concurrency: int = 2,
    run_id: str = "run-crawl-test",
    **driver_kwargs: Any,
) -> CrawlerDriver:
    """Build the fake-lane driver in the ``crawl`` shape over a fixture repo."""

    from menagerie.crawler.tests.test_slice_f_driver import _driver

    snapshot = root_snapshot(repo_root)
    driver = _driver(
        tmp_path,
        snapshot,
        paths_override=crawl_paths(repo_root, snapshot),
        scheduler="pipelined",
        author_concurrency=author_concurrency,
        **driver_kwargs,
    )
    driver.config = replace(
        driver.config,
        run_id=run_id,
        invocation_shape="crawl",
        scheduler="pipelined",
        crawl_limit=limit,
    )
    return driver


def plenty_of_disk(_path: Path) -> int:
    """Disk measurement seam: never below the floor."""

    return PLENTY_OF_DISK
