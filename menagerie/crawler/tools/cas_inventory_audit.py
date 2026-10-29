"""Read-only CAS inventory audit + fail-closed preflight arm (IC-2).

Classifies every ledger-attested or on-disk public/private mirror key as
``attested-present``, ``attested-absent``, ``present-unattested-litter``
(non-addressable-shaped, IC-3), or ``present-unattested-foreign``.

The m8635 shape motivates the preflight: a torn publication left 2 of 23
attested public objects unwritten, shielded only by that transaction's
staleness verdict -- one un-staling action away from a campaign-wide
admission brick. ``--preflight`` refuses (exit 1, keys named) on any
attested-absent or foreign key so the hole is caught BEFORE a campaign runs.

    python -m menagerie.crawler.tools.cas_inventory_audit \\
        --artifact-shard menagerie/crawler/records/artifacts/current-shard.jsonl \\
        --mirrors-root .crawl-local/mirrors [--preflight]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from menagerie.crawler.artifact_transactions import (
    ArtifactCheckpointError,
    cas_inventory_audit,
)
from menagerie.crawler.mirrors import MirrorStore


def main(argv: list[str] | None = None) -> int:
    """Run the audit and print one JSON row per key."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-shard",
        action="append",
        required=True,
        type=Path,
        help="Artifact ledger shard path (repeatable).",
    )
    parser.add_argument(
        "--mirrors-root",
        required=True,
        type=Path,
        help="Mirror store root containing public/, private/, local/.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Refuse (exit 1) on any attested-absent or foreign key.",
    )
    arguments = parser.parse_args(argv)
    mirrors = MirrorStore(
        arguments.mirrors_root / "public",
        arguments.mirrors_root / "private",
        arguments.mirrors_root / "local",
    )
    try:
        rows = cas_inventory_audit(
            tuple(arguments.artifact_shard),
            mirrors=mirrors,
            preflight=arguments.preflight,
        )
    except ArtifactCheckpointError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
        print(json.dumps(row, sort_keys=True))
    print(json.dumps({"summary": counts}, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
