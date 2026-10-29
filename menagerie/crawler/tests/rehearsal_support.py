"""Deterministic paid-lane replacements for the permanent rehearsal fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from menagerie.crawler import cli
from menagerie.crawler.driver import CrawlerDriver, default_driver_paths
from menagerie.crawler.env_lifecycle import (
    CommittedLockFamilyError,
    parse_probe_receipt_bytes,
)
from menagerie.crawler.envs import EnvironmentIntent, load_environment_registry
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.tests.test_slice_f_driver import _driver


class CommittedFakeEnvironments:
    """Materialize fake metadata matching each committed lock family exactly."""

    def __init__(self, root: Path) -> None:
        """Bind the disposable fake-prefix root."""

        self.root = root

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Invoke the driver with committed inventory and probe evidence."""

        if intent.lock.export_bytes is None:
            raise CommittedLockFamilyError(
                f"intent {intent.name!r} has no proven committed lock family"
            )
        prefix = self.root / intent.name
        metadata_root = prefix / "conda-meta"
        metadata_root.mkdir(parents=True, exist_ok=True)
        export = json.loads(intent.lock.export_bytes)
        packages = export.get("packages")
        if not isinstance(packages, list) or not packages:
            raise ValueError(f"fake rehearsal found no committed packages for {intent.name}")
        for index, package in enumerate(packages):
            if not isinstance(package, dict):
                raise ValueError(f"fake rehearsal found a malformed package for {intent.name}")
            name = str(package.get("name", "package")).replace("/", "-")
            (metadata_root / f"{index:04d}-{name}.json").write_bytes(
                canonical_json_bytes(package) + b"\n"
            )
        probe_path = intent.lock.lock_path.with_name(f"{intent.lock.target}.probes.json")
        probes = parse_probe_receipt_bytes(intent.probes, probe_path.read_bytes())
        use(prefix, probes)
        return object()


def fake_driver_factory(args: argparse.Namespace) -> CrawlerDriver:
    """Build the production driver around the established test-suite fake lanes."""

    if args.intake is None:
        raise ValueError("rehearsal fake requires --intake")
    repo_root = args.repo_root.resolve()
    snapshot = load_intake_snapshot(args.intake.resolve())
    binding = cli._required_campaign_binding(repo_root, snapshot)
    registry = load_environment_registry(target=args.target)
    return _driver(
        repo_root,
        snapshot,
        paths_override=default_driver_paths(repo_root, snapshot.root),
        environments=CommittedFakeEnvironments(repo_root / ".crawl-local" / "fake-envs"),
        registry=registry,
        scheduler=args.scheduler,
        phase=args.phase,
        review_at=args.review_checkpoint_at,
        milestones=tuple(args.progress_milestones),
        campaign_id=binding.spec.campaign_id,
        author_model=binding.spec.author_model,
        checker_model=binding.spec.checker_model,
        author_concurrency=args.author_concurrency,
        checker_concurrency=args.checker_concurrency,
    )
