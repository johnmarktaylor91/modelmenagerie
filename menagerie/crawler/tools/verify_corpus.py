"""Verify the durable text corpus fail-closed.

Every index row's payload must be physically present and re-hash to its
content address, and every excerpt found below the given work roots must
resolve through the index to a corpus document its locator verifies against.
Report-only: nothing is written.

Usage:

    PYTHONPATH=. python -m menagerie.crawler.tools.verify_corpus \\
        --records-root menagerie/crawler/records \\
        --work-root .crawl-local/work
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from menagerie.crawler.text_corpus import verify_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        action="append",
        default=None,
        help="Work root to sweep for excerpt resolution (repeatable, optional).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_corpus(args.records_root, work_roots=tuple(args.work_root or ()))
    print(
        json.dumps(
            {
                "passed": report.passed,
                "row_count": report.row_count,
                "payload_count": report.payload_count,
                "excerpt_count": report.excerpt_count,
                "orphan_payload_count": report.orphan_payload_count,
                "findings": list(report.findings),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
