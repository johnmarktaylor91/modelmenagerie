"""Ordered library work lists. Planning never imports or executes June code."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout
from menagerie.pipeline.records import RecordError

ORDER = ("needs_adapter", "needs_adapter:inputs", "standin:kwargs",
         "missing_dependency", "x_standin", "rung3-env", "discovered-paddle")
FIELDS = ("id", "name", "zoo", "variant", "family", "wave", "state", "class", "list")


def _current(data: DataLayout) -> list[dict[str, str]]:
    from menagerie.pipeline import read_json
    from menagerie.pipeline.records import ModelRecord

    rows = []
    for path in sorted(data.root.glob("models/*/*.json")):
        model = read_json(ModelRecord, path)
        cls = model.give_up.cls if model.give_up else ""
        label = _label(cls)
        if cls:
            rows.append(dict(id=model.id, name=model.name, zoo=model.zoo, variant=model.variant,
                             family=model.family, wave="", state="typed", **{
                                 "class": cls, "list": label or cls}))
    return rows


def _label(cls: str) -> str | None:
    if cls.startswith("missing_dependency"):
        return "missing_dependency"
    if cls in ("standin:compact", "standin:exec"):
        return "x_standin"
    if cls.startswith("needs_env"):
        return "rung3-env"
    return cls if cls in ORDER else None


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise RecordError(f"residual rows missing columns: {sorted(missing)}")
        rows = list(reader)
    ids = [(r["id"], r["list"]) for r in rows if r["id"]]
    if len(ids) != len(set(ids)):
        raise RecordError("residual rows contain duplicate stable ids within a work list")
    return rows


def plan(*, data: DataLayout, rows: Path | None = None,
         out: Path | None = None) -> dict[str, Any]:
    """Group a frozen TSV (or current typed records), without changing records or queues.

    ``out`` is an optional JSON manifest path. Anonymous discovery rows are retained
    and explicitly held, rather than silently losing them from the denominator.
    """

    source = _read(rows) if rows is not None else _current(data)
    lists = {label: [dict(r, held="no_stable_id" if not r["id"] else "")
                     for r in source if r["list"] == label] for label in ORDER}
    result = {"schema": "residual/1", "rows": len(source),
              "snapshot_sha256": hashlib.sha256(rows.read_bytes()).hexdigest() if rows else None,
              "counts": dict(Counter(r["list"] for r in source)),
              "order": list(ORDER), "work_lists": lists,
              "held_lists": sorted({r["list"] for r in source} - set(ORDER)),
              "queueing": "held; stage3 queue --committed only after acceptance"}
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        temporary = out.with_suffix(out.suffix + ".part")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(out)
    return result
