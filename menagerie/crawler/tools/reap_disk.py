"""Tiered, report-only disk reaper for crawler scratch (C-DISK-GC).

Prints the full reap plan -- per-tier candidate counts and byte totals plus
the protected-class summary -- and deletes NOTHING unless ``--apply`` is
passed explicitly. Apply takes the driver lock (refusing while a driver is
live), recomputes the whole plan, and deletes only candidates both plans
admit; every deletion passes the corpus guard on its resolved location.

The binding contract is ``procedures/TEXT_CORPUS.md``: the durable text
corpus is never reapable, work-dir scratch is reapable only once its corpus
clearance passes, and GC stays report-only until explicitly applied.

Usage:

    PYTHONPATH=. python -m menagerie.crawler.tools.reap_disk \\
        --work-root .crawl-local/work \\
        --records-root menagerie/crawler/records \\
        --runtime-root .crawl-local \\
        [--env-state-root .crawl-local/environment-operator] \\
        [--tier promoted-attempt-scratch ...] [--json plan.json] [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from menagerie.crawler.reaper import (
    REAP_TIERS,
    apply_reap_plan,
    build_reap_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=None,
        help="Runtime root (.crawl-local); default is the work root's parent.",
    )
    parser.add_argument(
        "--env-state-root",
        type=Path,
        default=None,
        help=(
            "Environment-operator state root for the env-orphans tier; default is "
            "<runtime-root>/environment-operator when that directory exists."
        ),
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=REAP_TIERS,
        default=None,
        help="Limit the plan to these tiers (repeatable); default is every tier.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the plan JSON here.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the planned candidates. Default is report-only.",
    )
    return parser


def _human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024.0
    return f"{int(size)} B"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime_root = args.runtime_root or args.work_root.parent
    env_state_root = args.env_state_root
    if env_state_root is None:
        default_env = runtime_root / "environment-operator"
        env_state_root = default_env if default_env.is_dir() else None

    plan = build_reap_plan(
        work_root=args.work_root,
        records_root=args.records_root,
        runtime_root=runtime_root,
        env_state_root=env_state_root,
        tiers=tuple(args.tier) if args.tier else REAP_TIERS,
    )

    totals = plan.tier_totals()
    print(f"reap plan ({'APPLY' if args.apply else 'report-only'})")
    print(f"  work root:    {plan.work_root}")
    print(f"  records root: {plan.records_root}")
    if plan.env_state_root is not None:
        print(f"  env state:    {plan.env_state_root}")
    total_bytes = 0
    for tier in plan.tiers:
        count, size = totals.get(tier, (0, 0))
        total_bytes += size
        print(f"  {tier:<26} {count:>6} candidates  {_human_bytes(size):>12}")
    print(
        f"  {'TOTAL reapable':<26} {len(plan.candidates):>6} candidates  {_human_bytes(total_bytes):>12}"
    )
    protected = plan.protected_totals()
    print("  protected classes: " + ", ".join(f"{k}={v}" for k, v in sorted(protected.items())))
    for note in plan.notes:
        print(f"  note: {note}")

    payload = plan.to_json()
    if args.apply:
        report = apply_reap_plan(
            plan,
            work_root=args.work_root,
            records_root=args.records_root,
            runtime_root=runtime_root,
            env_state_root=env_state_root,
        )
        payload["applied"] = True
        payload["apply_report"] = {
            "removed": [str(path) for path in report.removed],
            "skipped": [{"path": str(path), "reason": reason} for path, reason in report.skipped],
            "bytes_removed": report.bytes_removed,
        }
        print(
            f"  applied: removed {len(report.removed)} candidates, "
            f"{_human_bytes(report.bytes_removed)}; skipped {len(report.skipped)}"
        )
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
