"""CLI for tool-minted checker calibration receipts (W2).

``mint`` derives every measured number from the campaign's own ledgers plus
the blind-replay matrix report and writes the receipt (refusing, and removing
the file, if the written artifact fails its own activation loader). ``verify``
recomputes every measured field from the same ledgers and refuses divergence
-- the R6 stage's "receipt verifies" EXIT instrument.

Examples
--------
::

    PYTHONPATH=. python -m menagerie.crawler.tools.calibration_receipt mint \
        --records-root CAMPAIGN/menagerie/crawler/records \
        --work-root CAMPAIGN/.crawl-local/work \
        --checker-model gpt-5.6-sol \
        --blind-replay-report replay-report.json \
        --source "R6 calibration leg 2026-08-xx" \
        --next-rung-models 200 \
        --projected-wall-hours 60 --wall-budget-hours 120 \
        --projected-sessions 800 --session-budget 2000 \
        --projected-dollars 150 --dollar-budget 300 \
        --out menagerie/crawler/assets/checker-calibration-receipt.json

    PYTHONPATH=. python -m menagerie.crawler.tools.calibration_receipt verify \
        --receipt menagerie/crawler/assets/checker-calibration-receipt.json \
        --records-root ... --work-root ... --blind-replay-report replay-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.checker_calibration import CheckerCalibrationError
from menagerie.crawler.checker_calibration_mint import (
    mint_calibration_receipt,
    verify_calibration_receipt,
)


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        action="append",
        default=[],
        help="root searched recursively for operator-telemetry.jsonl; repeatable",
    )
    parser.add_argument("--blind-replay-report", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    mint = subparsers.add_parser("mint", help="derive and write one receipt")
    _add_shared(mint)
    mint.add_argument("--checker-model", required=True)
    mint.add_argument("--source", required=True)
    mint.add_argument("--captured-at", default=None)
    mint.add_argument("--next-rung-models", type=float, required=True)
    mint.add_argument("--projected-wall-hours", type=float, required=True)
    mint.add_argument("--wall-budget-hours", type=float, required=True)
    mint.add_argument("--projected-sessions", type=float, required=True)
    mint.add_argument("--session-budget", type=float, required=True)
    mint.add_argument("--projected-dollars", type=float, required=True)
    mint.add_argument("--dollar-budget", type=float, required=True)
    mint.add_argument("--projected-output-tokens", type=float, default=None)
    mint.add_argument("--token-budget", type=float, default=None)
    mint.add_argument("--out", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="recompute and refuse divergence")
    _add_shared(verify)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "golden corpus directory; when given, the locator-attestation zero "
            "bar is recomputed from the corpus bytes themselves (fable A-3)"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "mint":
            if args.blind_replay_report is None:
                print(
                    "calibration-receipt mint requires --blind-replay-report "
                    "(B1-XV2 is the receipt's acceptance instrument)",
                    file=sys.stderr,
                )
                return 2
            receipt = mint_calibration_receipt(
                args.records_root,
                work_roots=tuple(args.work_root),
                checker_model=args.checker_model,
                blind_replay_report_path=args.blind_replay_report,
                captured_at=(
                    args.captured_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                ),
                source=args.source,
                next_rung_models=args.next_rung_models,
                projected_wall_hours=args.projected_wall_hours,
                wall_budget_hours=args.wall_budget_hours,
                projected_sessions=args.projected_sessions,
                session_budget=args.session_budget,
                projected_dollars=args.projected_dollars,
                dollar_budget=args.dollar_budget,
                projected_output_tokens=args.projected_output_tokens,
                token_budget=args.token_budget,
                out_path=args.out,
            )
            print(json.dumps({"minted": str(args.out), "receipt": receipt}, indent=2))
        else:
            receipt = verify_calibration_receipt(
                args.receipt,
                args.records_root,
                work_roots=tuple(args.work_root),
                blind_replay_report_path=args.blind_replay_report,
                corpus_dir=args.corpus,
            )
            print(json.dumps({"verified": str(args.receipt), "receipt": receipt}, indent=2))
    except CheckerCalibrationError as exc:
        print(f"calibration-receipt refusal: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
