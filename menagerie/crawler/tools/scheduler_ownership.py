"""D-59 (OC-12) CLI: host scheduler ownership census + decommission verb.

``census`` reverse-sweeps the host's torchlens scheduler entries (crontab
wake blocks + launchd agent-root labels), joins them against the machine-
scoped ownership registry and each entry's owning operational ledger, and
prints the classified JSON census (exit 1 on anything needing attention).

``decommission`` removes ONE eligible entry (unregistered / owner-lost /
stale / residue) with a byte-exact provenance snapshot appended to the
registry; it refuses entries a reachable owner still holds active.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

from menagerie.crawler.identity import utc_now
from menagerie.crawler.scheduler_ownership import (
    SchedulerOwnershipError,
    SchedulerOwnershipRegistry,
    census_host_entries,
    decommission_entry,
    default_registry_path,
    evaluate_ownership_census,
)
from menagerie.crawler.wakeup import OperationalContext, launchd_agent_root


def _read_crontab_host() -> str:
    result = subprocess.run(["crontab", "-l"], check=False, capture_output=True, text=True)
    if result.returncode not in {0, 1}:
        raise SchedulerOwnershipError(f"cannot inspect crontab: {result.stderr.strip()}")
    return result.stdout if result.returncode == 0 else ""


def _write_crontab_host(content: str) -> None:
    result = subprocess.run(
        ["crontab", "-"], input=content, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SchedulerOwnershipError(f"cannot rewrite crontab: {result.stderr.strip()}")


def _bootout_host(label: str) -> None:
    import os

    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
        check=False,
        capture_output=True,
        text=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=("census", "decommission"))
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument(
        "--crontab-file",
        type=Path,
        default=None,
        help="read the crontab from a file instead of `crontab -l` (audits, tests)",
    )
    parser.add_argument("--agent-root", type=Path, default=None)
    parser.add_argument("--entry", help="entry id to decommission")
    parser.add_argument("--reason", default="", help="operator reason (decommission)")
    parser.add_argument("--machine-id", default="operator-cli")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    registry = SchedulerOwnershipRegistry(args.registry or default_registry_path())
    agent_root = args.agent_root if args.agent_root is not None else launchd_agent_root()
    try:
        if args.crontab_file is not None:
            crontab_text = args.crontab_file.read_text(encoding="utf-8")
        else:
            crontab_text = _read_crontab_host()
        entries = census_host_entries(crontab_text=crontab_text, agent_root=agent_root)
        if args.verb == "census":
            census = evaluate_ownership_census(registry, entries)
            print(json.dumps(census, sort_keys=True))
            return 0 if census["status"] == "clean" else 1
        if not args.entry:
            raise SchedulerOwnershipError("decommission requires --entry")
        matches = [entry for entry in entries if entry.entry_id == args.entry]
        if not matches:
            raise SchedulerOwnershipError(f"no host entry named {args.entry!r}")
        if args.crontab_file is not None:
            crontab_path = args.crontab_file

            def _read() -> str:
                return crontab_path.read_text(encoding="utf-8")

            def _write(content: str) -> None:
                crontab_path.write_text(content, encoding="utf-8")

            reader, writer = _read, _write
        else:
            reader, writer = _read_crontab_host, _write_crontab_host
        context = OperationalContext("operator-cli", args.machine_id, {}, None)
        row = decommission_entry(
            registry,
            matches[0],
            reason=args.reason,
            context=context,
            created_at=utc_now(),
            crontab_reader=reader,
            crontab_writer=writer,
            launchd_bootout=_bootout_host,
        )
        print(json.dumps(row, sort_keys=True))
        return 0
    except SchedulerOwnershipError as exc:
        print(f"scheduler-ownership failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
