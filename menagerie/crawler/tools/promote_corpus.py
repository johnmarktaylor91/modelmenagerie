"""Operator backfill: promote work-dir custody trees into the durable text corpus.

The driver promotes at award time; this tool covers everything the hook never
saw -- work dirs from before the corpus existed (the pilot backfill), and
terminal models whose fetched documents the fetch-once policy still retains.
Promotion is idempotent and digest-checked, so re-running over the same work
root is always safe.

Usage:

    PYTHONPATH=. python -m menagerie.crawler.tools.promote_corpus \\
        --work-root .crawl-local/work \\
        --records-root menagerie/crawler/records
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from menagerie.crawler.driver_contracts import DriverLock, DriverLockError
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.text_corpus import (
    TextCorpusError,
    license_dispositions_from_artifact_events,
    promote_author_root,
    verify_corpus,
    verify_corpus_launch_gate,
)


def _license_maps(records_root: Path) -> dict[str, dict[str, str]]:
    """Build per-model license mirrors from the artifact-authority ledger."""

    ledger = records_root / "artifacts" / "current-shard.jsonl"
    if not ledger.is_file():
        return {}
    events = scan_jsonl(ledger, validate=False)
    stable_ids = {
        str(claim.get("stable_id"))
        for event in events
        for claim in (event.get("claims") or [])
        if isinstance(claim, dict) and claim.get("stable_id")
    }
    return {
        stable_id: license_dispositions_from_artifact_events(events, stable_id)
        for stable_id in stable_ids
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument(
        "--stable-id",
        action="append",
        default=None,
        help="Limit promotion to these work dirs (repeatable); default is every one.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Backfill every selected author root and require a clean verification."""

    args = build_parser().parse_args(argv)
    work_root: Path = args.work_root
    records_root: Path = args.records_root
    if not work_root.is_dir():
        print(f"work root does not exist: {work_root}", file=sys.stderr)
        return 2

    runtime_root = work_root.parent
    owner = {
        "tool": "menagerie.crawler.tools.promote_corpus",
        "pid": os.getpid(),
        "work_root": str(work_root.resolve()),
        "records_root": str(records_root.resolve()),
    }
    try:
        with DriverLock(runtime_root / "locks" / "driver.lock", owner):
            author_roots = sorted(
                candidate / "author"
                for candidate in work_root.iterdir()
                if candidate.is_dir()
                and (candidate / "author").is_dir()
                and (args.stable_id is None or candidate.name in set(args.stable_id))
            )
            print(
                f"plan: promote {len(author_roots)} author custody trees from {work_root} "
                f"into {records_root}"
            )

            licenses = _license_maps(records_root)
            totals = {"promoted": 0, "replayed": 0, "refused": 0, "missing": 0}
            refusal_models: list[str] = []
            for author_root in author_roots:
                stable_id = author_root.parent.name
                report = promote_author_root(
                    author_root,
                    records_root,
                    license_dispositions=licenses.get(stable_id, {}),
                )
                promoted = len(report.promoted)
                replayed = len(report.replayed)
                refused = sum(
                    1
                    for row in report.refusals
                    if row["disposition"] in ("digest-mismatch-refused", "index-busy-refused")
                )
                missing = sum(
                    1 for row in report.refusals if row["disposition"] == "evidence-missing"
                )
                totals["promoted"] += promoted
                totals["replayed"] += replayed
                totals["refused"] += refused
                totals["missing"] += missing
                if report.refusals:
                    refusal_models.append(report.stable_id)
                print(
                    f"{report.stable_id}: promoted={promoted} replayed={replayed} "
                    f"refused={refused} missing={missing}"
                )

            try:
                verification = verify_corpus_launch_gate(records_root, work_root=work_root)
            except TextCorpusError:
                # Preserve the complete typed report in the operator output;
                # the gate's exception is the fail-closed control signal.
                verification = verify_corpus(records_root, work_roots=(work_root,))
    except DriverLockError as exc:
        print(f"driver-active: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "totals": totals,
                "models_with_refusals": refusal_models,
                "verification": {
                    "passed": verification.passed,
                    "row_count": verification.row_count,
                    "payload_count": verification.payload_count,
                    "excerpt_count": verification.excerpt_count,
                    "orphan_payload_count": verification.orphan_payload_count,
                    "finding_count": len(verification.findings),
                },
            },
            indent=2,
        )
    )
    return 1 if refusal_models or not verification.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
