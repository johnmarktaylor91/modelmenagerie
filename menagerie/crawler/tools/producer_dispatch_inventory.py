"""Regenerate the disclosed C-22 producer-dispatch inventory from the Python authority.

``menagerie.crawler.backend.PRODUCER_DISPATCH_INVENTORY`` is the authority; the committed
JSON beside the tests is its DISCLOSURE, so the inventory (and in particular the two
code-only closures) is reviewable as data in a diff rather than only as Python. The census
set-compares the two, so hand-editing the JSON to make a run pass fails immediately -- and
regenerating it is a deliberate, reviewable act, which is the point.

Usage::

    python -m menagerie.crawler.tools.producer_dispatch_inventory check
    python -m menagerie.crawler.tools.producer_dispatch_inventory write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from menagerie.crawler.backend import (
    CODE_ONLY_CLOSURE_FIELDS,
    DISPATCH_AND_IDENTITY_SINKS,
    PRODUCER_DISPATCH_INVENTORY,
)

INVENTORY_RELPATH = Path("menagerie/crawler/tests/producer_dispatch_inventory.json")
SCHEMA_VERSION = "menagerie.crawler.producer-dispatch-inventory.v1"

_COMMENT = (
    "Derived DISCLOSURE of menagerie.crawler.backend.PRODUCER_DISPATCH_INVENTORY: every "
    "producer-written string that reaches a machine dispatch or an identity preimage, bound "
    "to the closed vocabulary that refuses free text. Catalog C-22. The Python tuple is the "
    "authority; this file exists so the inventory -- especially the two code-only closures "
    "-- is reviewable as data and cannot grow silently. Regenerate with "
    "python -m menagerie.crawler.tools.producer_dispatch_inventory write."
)


def render() -> Mapping[str, Any]:
    """Return the disclosure document for the live inventory.

    Returns
    -------
    Mapping[str, Any]
        JSON-ready document mirroring every inventory row.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "comment": _COMMENT,
        "sinks": sorted(DISPATCH_AND_IDENTITY_SINKS),
        "code_only_closure": sorted(CODE_ONLY_CLOSURE_FIELDS),
        "fields": [
            {
                "field": row.field,
                "schema_paths": [
                    {"schema_version": version, "path": path} for version, path in row.schema_paths
                ],
                "schema_closure": row.schema_closure.value,
                "closure_note": row.closure_note,
                "sinks": list(row.sinks),
                "validator": row.validator,
                "refusal": row.refusal,
                "canonical_sample": row.canonical_sample,
                "free_text_sample": row.free_text_sample,
                "finding": row.finding,
            }
            for row in PRODUCER_DISPATCH_INVENTORY
        ],
    }


def _repository_root() -> Path:
    """Return the repository root holding the committed inventory."""

    return Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> int:
    """Check or rewrite the committed disclosure.

    Parameters
    ----------
    argv:
        Command-line arguments.

    Returns
    -------
    int
        ``0`` when the committed file matches (or was written), ``1`` on drift.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "write"))
    arguments = parser.parse_args(argv)
    path = _repository_root() / INVENTORY_RELPATH
    rendered = json.dumps(render(), indent=2) + "\n"
    if arguments.command == "write":
        path.write_text(rendered, encoding="utf-8")
        print(f"wrote {INVENTORY_RELPATH}")
        return 0
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current != rendered:
        print(f"{INVENTORY_RELPATH} is stale; run `... producer_dispatch_inventory write`")
        return 1
    print(f"{INVENTORY_RELPATH} is current")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
