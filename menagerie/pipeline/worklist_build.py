"""Build the tagged worklist from frozen rows and a committed census draw.

Coverage rule (decisions.md 2026-10-08 9:56 PM PT): a row a script typed is owed one agent
attempt. So a frozen row leaves the list only when ``drive`` settles it (an agent typed its
give-up, or the class is not about implementing the model), never because a script retyped it
since the freeze; and rows outside the frozen set that a scripted give-up left owed an attempt
(earlier rungs' rows) are appended after the frozen rows, in the ``r3`` cohort.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, read_jsonl
from menagerie.pipeline.records import STABLE_ID, CensusRow

R3_ORDER = ("c2_tail", "needs_adapter", "needs_adapter:inputs", "standin:kwargs",
            "missing_dependency", "rung3-env", "x_standin", "discovered-paddle",
            "codex_rows", "repin")
HELD = {"rung3-env", "repin"}


def _r3_line(row: dict[str, str], files: Any) -> str | None:
    from menagerie.pipeline import drive, ledger

    model_id = row["id"]
    if not model_id:
        return None
    if not re.fullmatch(STABLE_ID, model_id):
        raise ValueError(f"invalid frozen id: {model_id!r}")
    record = files.record(model_id)
    if record and record.recipe and ledger.validated(ledger.deciding_trace(
            model_id, files.rows.get(model_id, []), recipe_rev=record.recipe.rev, tl=files.tl)):
        return None
    settled = drive._step(model_id, files)[0] == "done"
    if settled and (record and record.give_up or row.get("list") in ("c2_tail", "repin")):
        return None
    return f"{'#' if row.get('list') in HELD else ''}{model_id} r3"


def _reopened_lines(data: DataLayout, files: Any, seen: set[str]) -> list[str]:
    """Rows outside the frozen set whose scripted give-up leaves them owed an agent attempt."""
    from menagerie.pipeline import drive

    ids = sorted((p.stem for p in data.root.glob("models/*/*.json")), key=lambda i: int(i[1:]))
    lines = []
    for model_id in ids:
        record = None if model_id in seen else files.record(model_id)
        if record is None or record.give_up is None:
            continue
        if drive.agent_tried(record, files.rows.get(model_id, [])):
            continue
        if drive._step(model_id, files)[0] in ("2a", "2b"):
            lines.append(f"{model_id} r3")
            seen.add(model_id)
    return lines


def _draw_lines(draw: Path | None, frozen: list[dict[str, str]], listed: set[str],
                listed_families: set[str]) -> list[str]:
    seen = {row["id"] for row in frozen if row["id"]} | listed
    families = {row.get("family") for row in frozen if row.get("family")} | listed_families
    lines = []
    for row in read_jsonl(CensusRow, draw) if draw is not None else ():
        if not row.eligible or not row.alive or row.archived or row.id in seen or row.family in families:
            continue
        lines.append(f"{row.id} r4-github")
        seen.add(row.id)
        if row.family:
            families.add(row.family)
    return lines


def build(*, data: DataLayout, rows: Path, draw: Path | None = None,
          out: Path | None = None) -> dict[str, Any]:
    """Keep frozen priority/holds, then rows reopened outside the frozen set, then eligible
    new families in draw order.

    Output must be explicit: building a preview must never replace the live worklist.
    The caller supplies a census draw already committed to its source revision.
    """
    from menagerie.pipeline import drive

    if out is None:
        raise ValueError("worklist build requires --out (use a preview before the switch)")
    with rows.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError("frozen rows need an id column")
        frozen = list(reader)
    rank = {name: n for n, name in enumerate(R3_ORDER)}
    frozen.sort(key=lambda row: rank.get(row.get("list", ""), len(rank)))
    files = drive._load(data)
    lines, emitted = [], set()
    for row in frozen:
        line = _r3_line(row, files)
        if line is not None and row["id"] not in emitted:
            lines.append(line)
            emitted.add(row["id"])
    reopened = _reopened_lines(data, files, emitted)
    lines.extend(reopened)
    reopened_ids = {line.split()[0] for line in reopened}
    families = {files.record(i).family for i in reopened_ids}
    lines.extend(_draw_lines(draw, frozen, reopened_ids, families))
    out = out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(f".{out.name}.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(out)
    counts = Counter(line.split()[1] for line in lines if not line.startswith("#"))
    return {"out": str(out), "cohorts": dict(counts), "reopened": len(reopened),
            "held": sum(line.startswith("#") for line in lines)}
