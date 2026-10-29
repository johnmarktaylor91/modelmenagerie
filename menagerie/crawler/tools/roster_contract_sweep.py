"""Roster-contract conformance sweep (AB-9): materialize every declared input
contract OFFLINE, before staging admits the member.

An unhonorable input contract (a shape the materializer refuses, a dtype the
converter cannot build, a declared distribution outside the vocabulary) is a
member-fault discovered today at author-spend time. This sweep drives the
REAL worker materialization seam — ``worker._materialize_declarative_call``
for rich contracts, ``materialize_standard_input`` for plain specs — with no
model import and no spend, and DISCLOSES every member whose declared inputs
cannot be honored. Staging consumes the report; ``--refuse-unhonorable``
makes it a refusing preflight arm.

The sweep deliberately reuses production code paths (never a reimplementation
of the contract semantics): what it proves honorable is exactly what the
worker will build.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

SWEEP_SCHEMA_VERSION = "menagerie.crawler.roster-contract-sweep.v1"


class RosterContractError(RuntimeError):
    """Raised when member rows cannot be loaded for the sweep."""


def _materialize_member(row: Mapping[str, Any], scratch_root: Path) -> None:
    """Materialize one member's declared inputs through the worker seam.

    Raises whatever typed error the production seam raises for an
    unhonorable contract; returning means the contract is honorable.
    """

    from menagerie.crawler.standard_inputs import InputSpec
    from menagerie.crawler.worker import WorkerRequest, _materialize_declarative_call

    input_spec = row.get("input_spec")
    request = WorkerRequest(
        stable_id=str(row.get("stable_id", "unbound")),
        recipe={},
        modality=row.get("modality"),
        input_spec=InputSpec.from_value(input_spec) if input_spec is not None else None,
        scratch_root=scratch_root,
        receipt_path=scratch_root / "unused-receipt.json",
        input_contract=row.get("input_contract"),
        seed=0,
        input_seed=int(row.get("input_seed", 0)),
        device="cpu",
        framework=str(row.get("framework", "pytorch")),
    )
    _materialize_declarative_call(request)


def sweep(rows: Iterable[Mapping[str, Any]], scratch_root: Path) -> dict[str, Any]:
    """Materialize every declared contract; disclose the unhonorable ones."""

    honored: list[str] = []
    contractless: list[str] = []
    refused: list[dict[str, str]] = []
    for row in rows:
        stable_id = str(row.get("stable_id", "<missing-stable-id>"))
        if row.get("input_spec") is None and row.get("input_contract") is None:
            contractless.append(stable_id)
            continue
        try:
            _materialize_member(row, scratch_root)
        except Exception as exc:  # noqa: BLE001 — every refusal class is disclosure
            refused.append(
                {
                    "stable_id": stable_id,
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:300],
                }
            )
        else:
            honored.append(stable_id)
    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "honored": sorted(honored),
        "contractless": sorted(contractless),
        "refused": sorted(refused, key=lambda entry: entry["stable_id"]),
        "counts": {
            "honored": len(honored),
            "contractless": len(contractless),
            "refused": len(refused),
        },
        "status": "all-honorable" if not refused else "unhonorable-disclosed",
    }


def _load_members_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load generic member rows: stable_id, modality, input_spec/input_contract."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise RosterContractError(f"{path}:{line_number} is not a JSON object")
        rows.append(dict(value))
    return rows


def _load_catalog_rows(path: Path, stable_ids: Optional[frozenset[str]]) -> list[dict[str, Any]]:
    """Adapt master-catalog rows (input_shape/input_dtype) into sweep rows."""

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, Mapping):
            continue
        stable_id = str(record.get("stable_id") or record.get("name") or "")
        if stable_ids is not None and stable_id not in stable_ids:
            continue
        shape = record.get("input_shape")
        row: dict[str, Any] = {
            "stable_id": stable_id,
            "modality": record.get("domain") or record.get("modality"),
        }
        if isinstance(shape, (list, tuple)):
            row["input_spec"] = {
                "shape": shape,
                "dtype": record.get("input_dtype") or "float32",
            }
        rows.append(row)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one roster-contract sweep and print the disclosure report."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--members-jsonl", type=Path)
    source.add_argument("--catalog-jsonl", type=Path)
    parser.add_argument("--scratch-root", type=Path, default=Path(".crawl-local/contract-sweep"))
    parser.add_argument("--refuse-unhonorable", action="store_true")
    args = parser.parse_args(argv)
    if args.members_jsonl is not None:
        rows = _load_members_jsonl(args.members_jsonl)
    else:
        rows = _load_catalog_rows(args.catalog_jsonl, None)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    report = sweep(rows, args.scratch_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.refuse_unhonorable and report["refused"]:
        print(
            f"roster-contract-sweep: {len(report['refused'])} unhonorable declared "
            "contracts (staging must not admit them silently)",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
