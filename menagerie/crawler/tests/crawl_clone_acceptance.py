"""RC-1 clone acceptance: ``crawl --campaign c1-mech --limit 230`` with fake providers.

Runs the REAL ``crawl`` CLI path (``menagerie.crawler.cli.main``) over a
byte-identical copy of a campaign's records with the corpus's canonical
fake-provider lanes (``FakeAuthor``/``FakeChecker``/``FakeForward``/
``FakeEnvironments``) injected through the CLI's driver-factory seam, then
prints a JSON transcript: the exit code, the selection census, the admitted
members, the terminals appended, the leg lifecycle rows, and the crawl-limit
and paced-start facts.

The records copy must sit at a NON-production path (not
``<repo>/menagerie/crawler/records``): the reducer validates attempts against
the committed environment registry only under the production layout, and fake
environment lanes cannot satisfy it (exactly like every fake-lane fixture in
this corpus). The repository root must be a clean checkout of the tree under
test, because the closure-green arm consumes that tree's gate verdicts.

    PYTHONPATH=. python -m menagerie.crawler.tests.crawl_clone_acceptance \\
        --repo-root ~/campaigns/rc1-clone \\
        --records-root ~/campaigns/rc1-clone-records/records \\
        --campaign c1-mech --limit 230 --author-width 12 \\
        --transcript /path/to/transcript.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from menagerie.crawler import cli
from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.driver import CrawlerDriver, DriverPaths
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.invocation_identity import mint_default_run_id
from menagerie.crawler.leg_generation import load_leg_events
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.partitioner import load_campaign_bindings
from menagerie.crawler.recordio import scan_jsonl


def _count_rows(path: Path) -> int:
    return sum(1 for _ in scan_jsonl(path)) if path.exists() else 0


def _run_events(paths: DriverPaths, kind: OperationalEventKind, run_id: str) -> list[dict[str, Any]]:
    if not paths.operational_ledger.exists():
        return []
    return [
        row
        for row in scan_jsonl(paths.operational_ledger)
        if row.get("event_kind") == kind.value and row.get("run_id") == run_id
    ]


def _canary_rows(records_root: Path) -> list[dict[str, Any]]:
    path = records_root / "control" / "configuration-canary.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _transcript(
    *,
    code: int,
    elapsed: float,
    args: argparse.Namespace,
    repo_root: Path,
    records_root: Path,
    run_id: str,
    snapshot: Any,
    baseline: dict[str, int],
    heads_before: set[str],
    paths: DriverPaths,
    ledgers: LedgerPaths,
) -> dict[str, Any]:
    """Assemble the acceptance transcript from the ledgers after the run."""

    model_rows = list(scan_jsonl(ledgers.models)) if ledgers.models.exists() else []
    new_rows = model_rows[baseline["models_rows"] :]
    heads_after = {row["stable_id"] for row in model_rows}
    leg_rows = (
        load_leg_events(records_root)
        if (records_root / "intake" / "leg-events.jsonl").exists()
        else []
    )
    keys = ("event", "generation_id", "exit_class", "terminal_delta", "leg_purpose")
    return {
        "exit_code": code,
        "elapsed_seconds": round(elapsed, 1),
        "repo_root": str(repo_root),
        "records_root": str(records_root),
        "campaign": args.campaign,
        "limit": args.limit,
        "author_width": args.author_width,
        "run_id": run_id,
        "intake_snapshot_id": snapshot.snapshot_id,
        "intake_members": len(snapshot.items),
        "baseline": baseline,
        "selection_census": [
            row["details"] for row in _run_events(paths, OperationalEventKind.WORK_SELECTION_CENSUS, run_id)
        ],
        "invocation_refused": [
            row["details"] for row in _run_events(paths, OperationalEventKind.INVOCATION_REFUSED, run_id)
        ],
        "crawl_limit_events": [
            row["details"] for row in _run_events(paths, OperationalEventKind.CRAWL_LIMIT, run_id)
        ],
        "configuration_canary_events": [
            row["details"]
            for row in _run_events(paths, OperationalEventKind.CONFIGURATION_CANARY, run_id)
        ],
        "leg_events_this_run": [
            {k: row.get(k) for k in keys if k in row}
            for row in leg_rows
            if row.get("run_id") == run_id
        ],
        "new_model_rows": len(new_rows),
        "new_terminal_members": sorted(heads_after - heads_before),
        "new_row_status_codes": sorted({str(r.get("status", {}).get("code")) for r in new_rows}),
        "driver_state": (
            json.loads(paths.driver_state.read_text()) if paths.driver_state.exists() else None
        ),
        "paced_start": _canary_rows(records_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m menagerie.crawler.tests.crawl_clone_acceptance")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--campaign", default="c1-mech")
    parser.add_argument("--limit", type=int, default=230)
    parser.add_argument("--author-width", type=int, default=12)
    parser.add_argument("--transcript", type=Path, required=True)
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    records_root = args.records_root.resolve()
    from dataclasses import replace

    from menagerie.crawler.tests.test_slice_f_driver import _driver

    bindings = load_campaign_bindings(records_root / "campaigns.json")
    binding = next(b for b in bindings if b.spec.campaign_id == args.campaign)
    snapshot = load_intake_snapshot(records_root / binding.intake_path)
    ledgers = LedgerPaths(
        records_root / "models" / "current-shard.jsonl",
        records_root / "attempts" / "local.jsonl",
        records_root / "gates" / "current-shard.jsonl",
    )
    paths = DriverPaths(repo_root / ".crawl-local", snapshot.root, ledgers)
    scratch = Path(tempfile.mkdtemp(prefix="rc1-accept-"))
    baseline = {
        "models_rows": _count_rows(ledgers.models),
        "attempts_rows": _count_rows(ledgers.attempts),
        "gates_rows": _count_rows(ledgers.gates),
        "operational_rows": _count_rows(paths.operational_ledger),
        "leg_events": len(load_leg_events(records_root)) if (records_root / "intake" / "leg-events.jsonl").exists() else 0,
    }
    heads_before = {row["stable_id"] for row in scan_jsonl(ledgers.models)} if ledgers.models.exists() else set()
    run_id = mint_default_run_id()
    built: dict[str, CrawlerDriver] = {}

    def factory(namespace: argparse.Namespace) -> CrawlerDriver:
        driver = _driver(
            scratch,
            snapshot,
            paths_override=paths,
            scheduler="pipelined",
            author_concurrency=int(namespace.author_concurrency),
            campaign_id=args.campaign,
            author_model=binding.spec.author_model,
            checker_model=binding.spec.checker_model,
        )
        driver.config = replace(driver.config, run_id=run_id, machine_id="rc1-acceptance")
        built["driver"] = driver
        return driver

    started = time.time()
    code = cli.main(
        [
            "--repo-root",
            str(repo_root),
            "crawl",
            "--records-root",
            str(records_root),
            "--campaign",
            args.campaign,
            "--limit",
            str(args.limit),
            "--author-width",
            str(args.author_width),
        ],
        driver_factory=factory,
    )
    elapsed = time.time() - started

    transcript = _transcript(
        code=code,
        elapsed=elapsed,
        args=args,
        repo_root=repo_root,
        records_root=records_root,
        run_id=run_id,
        snapshot=snapshot,
        baseline=baseline,
        heads_before=heads_before,
        paths=paths,
        ledgers=ledgers,
    )
    args.transcript.parent.mkdir(parents=True, exist_ok=True)
    args.transcript.write_text(json.dumps(transcript, indent=2, sort_keys=True, default=str))
    print(json.dumps({k: transcript[k] for k in ("exit_code", "elapsed_seconds", "new_model_rows", "invocation_refused", "crawl_limit_events")}, default=str))
    return 0 if code in (cli.EXIT_OK, cli.EXIT_OPERATOR_OUTAGE) else 1


if __name__ == "__main__":
    sys.exit(main())
