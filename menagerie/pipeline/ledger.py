"""The append-only ledger: rows, current state, ``publishable()`` and the June legacy shard.

Owned by the data lane. Current state is computed from the rows, never stored. A pass counts
only at the model's current ``recipe_rev`` and the pinned TorchLens version; an ``infra`` row
never overrides a pass; a render failure never touches a trace verdict.

The rules, applied to one model's rows in ledger order:

- The deciding trace row is the latest full-capture, eval-mode ``trace`` row at the current
  ``recipe_rev`` and TorchLens version that is not ``infra``. It validates the model when it
  passed with ``validation_state: passed`` and non-null ``n_ops`` and ``graph_hash``. A newer
  genuine failure therefore blocks publication until a newer pass resolves it.
- The latest non-``infra`` review row at the current ``recipe_rev`` blocks when its verdict is
  ``fabrication``. Whether a model needs a review at all is the caller's (its source case).
- Renders count only from the latest non-``infra`` render row keyed on the deciding trace's
  sha and the current ``renderer_rev``. The model is ``publishable`` when that row holds an
  SVG and a WebP thumbnail of one eval variant, else ``publishable_pending_render``.
- Legacy June rows carry June's 64-hex recipe revision, so they never validate a model.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from menagerie.pipeline import DataLayout, append_jsonl, month_of, read_jsonl
from menagerie.pipeline.records import (
    LedgerLine,
    LedgerRow,
    LegacyRow,
    RecordError,
    RenderRow,
    dumps,
    parse_ledger_line,
)

ModelState = Literal["publishable", "publishable_pending_render", "not_publishable"]
LEGACY_MONTH = "2026-06"
# June tracebacks name the worker's home directory; the committed shard keeps them as `~/`.
HOME_PREFIX = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")


def append(rows: Iterable[LedgerRow | RenderRow], *, data: DataLayout) -> list[Path]:
    """Append rows to their month shards; return the shard files written."""

    by_month: dict[str, list[LedgerRow | RenderRow]] = {}
    for row in rows:
        if not isinstance(row, (LedgerRow, RenderRow)):
            raise RecordError(f"ledger.append takes ledger and render rows, not {type(row)}")
        by_month.setdefault(month_of(row.at), []).append(row)
    paths = []
    for month, batch in sorted(by_month.items()):
        append_jsonl(batch, data.ledger(month))
        paths.append(data.ledger(month))
    return paths


def read_all(*, data: DataLayout) -> list[LedgerLine]:
    """Read every ledger shard, oldest first.

    Shards are parsed through ``drive``'s memo, so one process (a tick pass) parses only the
    lines appended since its last read, and shares the parse with ``drive.next_steps``. Rows are
    frozen records; the returned list is the caller's own."""

    from menagerie.pipeline import drive

    memo = drive._memo(data)
    shards = sorted((data.root / "ledger").glob("*.jsonl"))
    return [line for shard in shards for line in drive.shard_rows(memo, shard)]


def read_containing(text: str, *, data: DataLayout) -> list[LedgerLine]:
    """The rows whose ledger line contains ``text``, oldest first: a prefilter for a caller that
    needs a few rows, since a full parse takes seconds; the caller still checks the fields. An
    unfinished last line (an append in progress) is left out."""

    from menagerie.pipeline import drive

    rows = []
    for shard in sorted((data.root / "ledger").glob("*.jsonl")):
        whole = shard.read_text(encoding="utf-8").split("\n")[:-1]
        rows += [drive._ledger_line(json.loads(line)) for line in whole if text in line]
    return rows


def _latest(rows: Iterable[LedgerRow | RenderRow]) -> LedgerRow | RenderRow | None:
    """The last row by timestamp; ledger order breaks ties (``max`` keeps the first maximum)."""

    ordered = list(rows)
    if not ordered:
        return None
    return max(reversed(ordered), key=lambda row: row.at)


def deciding_trace(model_id: str, rows: Sequence[LedgerLine], *, recipe_rev: str,
                   tl: str) -> LedgerRow | None:
    """The trace row that decides a model's verdict at the current revisions, if any."""

    return _latest(
        r for r in rows
        if isinstance(r, LedgerRow) and r.id == model_id and r.stage == "trace"
        and r.recipe_rev == recipe_rev and r.tl == tl and r.outcome != "infra"
        and r.capture in (None, "full") and r.mode in (None, "eval")
    )


def validated(row: LedgerRow | None) -> bool:
    """Whether a deciding trace row validates its model."""

    return (row is not None and row.outcome == "pass" and row.validation_state == "passed"
            and row.n_ops is not None and row.graph_hash is not None)


def _fabricated(model_id: str, rows: Sequence[LedgerLine], recipe_rev: str) -> bool:
    review = _latest(
        r for r in rows
        if isinstance(r, LedgerRow) and r.id == model_id and r.stage == "review"
        and r.recipe_rev == recipe_rev and r.outcome != "infra"
    )
    return review is not None and review.verdict == "fabrication"


def _rendered(model_id: str, rows: Sequence[LedgerLine], trace: str | None,
              renderer_rev: str) -> bool:
    render = _latest(
        r for r in rows
        if isinstance(r, RenderRow) and r.id == model_id and r.trace == trace
        and r.renderer_rev == renderer_rev and r.outcome != "infra"
    )
    if render is None or render.outcome == "fail":
        return False
    kinds: dict[str, set[str]] = {}
    for art in render.renders:
        if art.variant and not art.variant.endswith("-train"):
            kinds.setdefault(art.variant, set()).add(art.kind)
    return any({"svg", "thumb"} <= found for found in kinds.values())


def state(model_id: str, rows: Sequence[LedgerLine], *, recipe_rev: str, tl: str,
          renderer_rev: str) -> ModelState:
    """Return a model's publication state from its rows at the current revisions."""

    trace = deciding_trace(model_id, rows, recipe_rev=recipe_rev, tl=tl)
    if not validated(trace) or _fabricated(model_id, rows, recipe_rev):
        return "not_publishable"
    if _rendered(model_id, rows, trace.trace, renderer_rev):
        return "publishable"
    return "publishable_pending_render"


def publishable(model_id: str, rows: Sequence[LedgerLine], *, recipe_rev: str, tl: str,
                renderer_rev: str) -> bool:
    """Return whether a model's latest rows pass at the current revisions (the one predicate)."""

    return state(model_id, rows, recipe_rev=recipe_rev, tl=tl,
                 renderer_rev=renderer_rev) == "publishable"


def import_legacy(manifest: Path, *, data: DataLayout) -> int:
    """Load ``routing_manifest.tsv`` as ``ledger/2026-06.jsonl``; return the row count.

    Home directories inside June error messages are written as ``~/``; nothing else changes.
    """

    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        rows = [LegacyRow.from_manifest(row) for row in csv.DictReader(handle, delimiter="\t")]
    rows = [dataclasses.replace(r, error=HOME_PREFIX.sub("~/", r.error)) for r in rows]
    shard = data.ledger(LEGACY_MONTH)
    if shard.exists() and any(not isinstance(r, LegacyRow)
                              for r in read_jsonl(parse_ledger_line, shard)):
        raise RecordError(f"{shard} holds rows that are not legacy; refusing to rewrite it")
    shard.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard.with_name(f".{shard.name}.tmp")
    tmp.write_text("".join(dumps(r) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, shard)
    return len(rows)


def status_counts(*, data: DataLayout) -> dict[str, int]:
    """Count models by current state and failure class, for ``menagerie-pipeline status``.

    Legacy rows count as ``legacy:<June status>``. For new rows, each model's latest
    non-``infra`` row per stage counts as ``<stage>:<outcome>`` and, when it has one, as
    ``<stage>:<outcome>:<class>``; models whose only rows are ``infra`` count as ``<stage>:infra``.
    """

    counts: Counter[str] = Counter()
    latest: dict[tuple[str, str], LedgerRow | RenderRow] = {}
    infra: set[tuple[str, str]] = set()
    for row in read_all(data=data):
        if isinstance(row, LegacyRow):
            counts[f"legacy:{row.status}"] += 1
            continue
        key = (row.id, row.stage)
        if row.outcome == "infra":
            infra.add(key)
        elif key not in latest or row.at >= latest[key].at:
            latest[key] = row
    for (_, stage), row in latest.items():
        counts[f"{stage}:{row.outcome}"] += 1
        cls = getattr(row, "cls", None)
        if cls:
            counts[f"{stage}:{row.outcome}:{cls}"] += 1
    for _, stage in infra - set(latest):
        counts[f"{stage}:infra"] += 1
    return dict(sorted(counts.items()))
