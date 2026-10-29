"""Shadow-close probe: run the REAL close partition predicate, read-only (DX-24).

The close-killing classes of 200a -- head+live-quarantine contradictions,
unaccounted stale heads, coverage holes, double-counted accounting arms -- were
all discovered AT LEG END, hours after the row that would refuse the close had
already landed. This probe is the in-flight detector: it runs the exact
production boundary predicate (``status --verify-partition
--require-valid-partition``, the same ``assert_partition`` the driver's close
uses -- one implementation, never a reimplementation that could drift) against
the live records root and reports every member the close WOULD refuse, in
minutes instead of at leg end.

Read-only by construction: the status path consumes the pure
``project_dependency_current`` projection and scans ledgers without taking the
driver lock, so the probe can run beside a live leg on any cadence (the
monitor lane owns the 30-60 minute schedule; this tool is one invocation).

Exit codes: 0 = the partition would close today; 2 = at least one would-refuse
member (the refusal text names them); 3 = the probe itself could not run.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.cli import EXIT_OK
from menagerie.crawler.cli import main as crawler_main

PROBE_OK = 0
PROBE_WOULD_REFUSE = 2
PROBE_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    """Return the probe's argument parser."""

    parser = argparse.ArgumentParser(
        prog="shadow-close-probe",
        description=(
            "run the production close partition predicate read-only and name "
            "every member the close would refuse"
        ),
    )
    parser.add_argument("--intake", type=Path, required=True, help="intake snapshot root")
    parser.add_argument(
        "--records-root", type=Path, required=True, help="canonical records root"
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path.cwd(), help="repository root (default: cwd)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full status payload plus the probe verdict as JSON",
    )
    return parser


def run_probe(
    *, intake: Path, records_root: Path, repo_root: Path
) -> tuple[bool, Optional[str], dict]:
    """Run the read-only close predicate once and return the structured verdict.

    The ONE implementation both consumers ride: the DX-24 cadence probe (this
    module's CLI) and the D-20 per-leg close-preview event the driver emits at
    leg start. Read-only by construction -- the status path takes no driver
    lock and triggers no recovery writes.

    Returns
    -------
    tuple[bool, Optional[str], dict]
        ``(would_refuse, refusal_text, status_payload)``. A probe that cannot
        run raises; callers own their degradation shape.
    """

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status_exit = crawler_main(
            [
                "--repo-root",
                str(repo_root),
                "status",
                "--intake",
                str(intake),
                "--records-root",
                str(records_root),
                "--verify-partition",
                "--require-valid-partition",
            ]
        )
    refusal = stderr.getvalue().strip() or None
    payload_text = stdout.getvalue().strip().splitlines()
    try:
        payload = json.loads(payload_text[-1]) if payload_text else {}
    except json.JSONDecodeError:
        payload = {}
    return status_exit != EXIT_OK, refusal, payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one read-only shadow-close probe invocation."""

    args = build_parser().parse_args(argv)
    try:
        would_refuse, refusal_or_none, payload = run_probe(
            intake=args.intake,
            records_root=args.records_root,
            repo_root=args.repo_root,
        )
    except Exception as exc:  # noqa: BLE001 -- a broken probe is a fact, never a clean exit
        print(f"shadow-close-probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return PROBE_ERROR
    refusal = refusal_or_none or ""
    if args.json:
        print(
            json.dumps(
                {
                    "probe": "shadow-close",
                    "would_refuse": would_refuse,
                    "refusal": refusal or None,
                    "status": payload,
                },
                sort_keys=True,
            )
        )
    if not would_refuse:
        if not args.json:
            terminal = payload.get("terminal")
            print(f"shadow-close probe OK: partition would close ({terminal} terminal)")
        return PROBE_OK
    # The refusal IS the production close's refusal text -- the exact member
    # list an operator would otherwise first read at leg end.
    if not args.json:
        print(refusal or "shadow-close probe: partition invalid (no refusal text)")
    print(
        "shadow-close probe: the close WOULD REFUSE today -- act before leg end",
        file=sys.stderr,
    )
    return PROBE_WOULD_REFUSE


if __name__ == "__main__":  # pragma: no cover - CLI shim
    raise SystemExit(main())
