"""Schema-migration census over recorded author results (IC-9, census half).

The carried solve cache is the finalized artifact transactions' RECORDED
author results, and each re-validates against the LIVE
``AUTHOR_RESULT_SCHEMA_VERSION`` at reuse time: any schema tightening
silently converts recorded results into per-member schema-stale cache misses
-- each individually disclosed, the STORM undisclosed (453
``author-cache-discarded`` events already in 200a; a 210-member storm is
100+ paid re-author sessions).

This census makes the storm a PRE-LAUNCH fact instead of a paid surprise: it
validates every finalized transaction's recorded author result against the
candidate tree's live schema and reports the would-be-stale count. Consumed
at every wave boundary, in the Wave-2 rotation-batch procedure, and at the
first-generation Wave-1 boundary run (PLAN_v6 SS4.1/SS4.2); its pairing
consumer is AB-30's mixed-era reader matrix. ``--gate`` exits nonzero on ANY
schema-stale row: a nonzero census is an explicit migration decision, never
an ambient cost.

    python -m menagerie.crawler.tools.schema_migration_census \\
        --artifact-shard <records>/artifacts/current-shard.jsonl \\
        --repository-root <campaign clone root> [--gate]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from menagerie.crawler.artifact_transactions import ArtifactEventKind, _load_events
from menagerie.crawler.constants import AUTHOR_RESULT_SCHEMA_VERSION
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.models import JsonObject
from menagerie.crawler.schema import PayloadValidationError, validate_payload

_FINAL_KINDS = {
    ArtifactEventKind.PUBLISHED.value,
    ArtifactEventKind.PRIVATE_COMMITTED.value,
}


def schema_migration_census(
    artifact_ledger_paths: Iterable[Path],
    *,
    repository_root: Path,
    schema_version: str = AUTHOR_RESULT_SCHEMA_VERSION,
) -> tuple[JsonObject, ...]:
    """Validate every finalized transaction's recorded author result.

    Parameters
    ----------
    artifact_ledger_paths:
        Complete artifact ledger shards.
    repository_root:
        Root the reconstruction anchors' relative paths resolve against.
    schema_version:
        Schema to validate against; defaults to the LIVE author-result schema
        so the census reports exactly the reuse-eligibility the candidate
        tree would apply.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One row per finalized transaction: ``schema-valid``, ``schema-stale``
        (with the exact refusal), or ``reconstruction-unreadable``.
    """

    events = _load_events(tuple(artifact_ledger_paths), memo=None)
    rows: list[JsonObject] = []
    for event in events:
        if str(event.get("event_kind")) not in _FINAL_KINDS:
            continue
        row: JsonObject = {
            "stable_id": str(event.get("stable_id")),
            "work_id": str(event.get("work_id")),
            "transaction_id": str(event.get("transaction_id")),
            "schema_version": schema_version,
        }
        anchor = event.get("reconstruction")
        try:
            if not isinstance(anchor, Mapping):
                raise OSError("final event lacks a reconstruction anchor")
            document_path = repository_root / str(anchor.get("path"))
            document_bytes = document_path.read_bytes()
            if hash_bytes(document_bytes) != anchor.get("sha256"):
                raise OSError("reconstruction bytes contradict their anchor")
            document: Any = json.loads(document_bytes)
            author_result = document.get("author_result")
            if not isinstance(author_result, Mapping):
                raise OSError("reconstruction document lacks its author result")
        except (OSError, json.JSONDecodeError) as exc:
            row["disposition"] = "reconstruction-unreadable"
            row["error"] = str(exc)[:200]
            rows.append(row)
            continue
        try:
            validate_payload(author_result, schema_version)
        except PayloadValidationError as exc:
            row["disposition"] = "schema-stale"
            row["error"] = str(exc)[:200]
        else:
            row["disposition"] = "schema-valid"
        rows.append(row)
    return tuple(sorted(rows, key=lambda value: str(value["transaction_id"])))


def main(argv: list[str] | None = None) -> int:
    """Run the census; ``--gate`` exits nonzero on any non-valid row."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-shard", action="append", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument(
        "--schema-version",
        default=AUTHOR_RESULT_SCHEMA_VERSION,
        help="Schema identity to validate against (default: the live schema).",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit 2 when any recorded result is schema-stale or unreadable: "
        "a nonzero census is an explicit migration decision.",
    )
    arguments = parser.parse_args(argv)
    rows = schema_migration_census(
        tuple(arguments.artifact_shard),
        repository_root=arguments.repository_root,
        schema_version=arguments.schema_version,
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["disposition"])] = counts.get(str(row["disposition"]), 0) + 1
        print(json.dumps(row, sort_keys=True))
    print(json.dumps({"summary": counts}, sort_keys=True), file=sys.stderr)
    if arguments.gate and (set(counts) - {"schema-valid"}):
        print(
            "GATE: nonzero schema-migration census -- an explicit migration "
            "decision is required before this tree gates a wave boundary",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
