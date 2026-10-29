"""P-07 mechanical closure (PM-6/W20): the docket refusal guard.

The process lesson: [FILL]-bearing deliverables and "suite pending" claims
read as done and ship. The guard is the cheap machine gate the catalog
demanded — refuse any docket carrying an unfilled placeholder, and (opt-in)
require a pasted gate output line before a docket may claim completion.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

#: Placeholder tokens that mark an unfilled required cell. Closed set;
#: widening it can only make the guard refuse MORE dockets.
FILL_MARKERS = ("[FILL]", "TBD", "<pending>", "[TODO]", "XXX-FILL")

#: A pasted gate output: a pytest tail ("N passed"), the census "status":
#: "passed", or an explicit exit-code line.
_GATE_OUTPUT_RE = re.compile(r"(\b\d+ passed\b|\"status\": \"passed\"|exit code 0)")

#: A completion claim resting on an unrun suite.
_PENDING_SUITE_RE = re.compile(r"\b(suite|gate|tests?)\s+(pending|not\s+run|skipped)\b", re.I)


def scan_docket(text: str) -> list[dict]:
    """Return one refusal per placeholder-bearing or pending-suite line."""

    refusals: list[dict] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for marker in FILL_MARKERS:
            if marker in line:
                refusals.append(
                    {"line": line_no, "marker": marker, "content": line.strip()}
                )
                break
        else:
            if _PENDING_SUITE_RE.search(line):
                refusals.append(
                    {"line": line_no, "marker": "pending-suite", "content": line.strip()}
                )
    return refusals


def has_gate_output(text: str) -> bool:
    """Return whether a pasted gate output line is present."""

    return _GATE_OUTPUT_RE.search(text) is not None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docket", type=Path, required=True)
    parser.add_argument(
        "--require-gate-output",
        action="store_true",
        help="also refuse a docket with no pasted gate output line",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        text = args.docket.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"docket-guard failed: {exc}", file=sys.stderr)
        return 1
    refusals = scan_docket(text)
    for refusal in refusals:
        print(
            f"docket-guard: line {refusal['line']}: {refusal['marker']}: "
            f"{refusal['content']}",
            file=sys.stderr,
        )
    if args.require_gate_output and not has_gate_output(text):
        print("docket-guard: no pasted gate output found", file=sys.stderr)
        return 1
    return 1 if refusals else 0


if __name__ == "__main__":
    raise SystemExit(main())
