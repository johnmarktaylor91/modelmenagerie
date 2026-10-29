"""The roster: rung 0 has the provenance repair only.

Owned by the data lane. ``repair`` stamps every row with ``stable_id``, ``found_by``,
``hint_urls`` and the other hint fields, joining the S-C discovery plan; stable ids never
change.

The joins, all by exact name and then by punctuation-insensitive name:

- ``stable_id`` from ``stable_ids.jsonl`` by ``(name, zoo, variant)``. The June catalog
  moved intake rows of ``unregistered-classics-pytorch`` to the classics, vendored, ported
  and reimplementation zoos, so such a row takes the one id its name and variant hold in
  those zoos. Writing repair mints missing active-row ids under the packet writer's
  roster lock; read-only repair only joins existing ids.
- ``S001:axis=<cell>`` from the July sweep's candidate list (committed under ``research/``).
- ``S-C:<cand_id>`` from the June 2026-06-28 discovery plan (``sweeps/S-C/IMPL_PLAN.tsv``),
  which also gives ``hint_urls`` (its repository URL, after the row's own ``source_url``),
  ``hint_framework`` and ``hint_paper``. Hints are untrusted until 2a confirms them.
- ``legacy:<partition>`` from the frozen July intake partitions, else ``legacy:unknown``.

``crawl_history.json`` lists sweep summaries, not names, so it gives no per-row join.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from menagerie.pipeline import DataLayout
from menagerie.pipeline.records import RosterRow, from_dict

PLACEHOLDERS = frozenset({"", "-", "na", "n/a", "none", "unknown", "?"})
PAPER = re.compile(r"arxiv:\s*\d{4}\.\d{4,5}|doi:\s*\S+|10\.\d{4,9}/\S+|https?://arxiv\.org/\S+",
                   re.IGNORECASE)
AXIS = re.compile(r"axis=([^;]+)")
REPO = Path(__file__).resolve().parents[2]
JULY = REPO / "research" / "fable-discovery-2026-07-03" / "discovery-complete" / (
    "candidates_ALL_dedup_clean.tsv")
PARTITIONS = REPO / "menagerie" / "crawler" / "records" / "partitions"


def read(*, data: DataLayout) -> list[RosterRow]:
    """Read ``crawl_roster.jsonl``; rows written before the repair have no ``schema`` key.

    The parse goes through ``drive``'s per-file memo, so one process (a tick pass, a 2a brief
    seeding several families) parses the roster again only when its size, mtime or inode
    changed. Rows are frozen records; the returned list is the caller's own."""

    from menagerie.pipeline import drive

    rows = drive._memo(data).get(data.roster, _parse)
    if rows is None:  # the memo reads a missing file as None; callers expect the open error
        raise FileNotFoundError(data.roster)
    return list(rows)


def _parse(path: Path) -> list[RosterRow]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [from_dict(RosterRow, {"schema": RosterRow.SCHEMA, **json.loads(line)})
            for line in lines if line.strip()]


def norm(name: str) -> str:
    """The punctuation-insensitive form of a name: lower case, letters and digits only."""

    return re.sub(r"[\W_]+", "", name.lower())


def _tsv(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [[c.strip() for c in row] for row in csv.reader(handle, delimiter="\t",
                                                               quoting=csv.QUOTE_NONE)]


def _value(text: str) -> str | None:
    return None if text.lower() in PLACEHOLDERS else text


INTAKE_ZOO = "unregistered-classics-pytorch"
CATALOG_ZOOS = ("classics-pytorch", "vendored-pytorch", "ported-pytorch", "reimpl-pytorch")


def _stable_ids(data: DataLayout) -> dict[tuple[str, str, str], str]:
    """The stable-id table, plus the intake-zoo keys of rows the catalog re-zoned."""

    from menagerie.identity import load_stable_ids

    table = load_stable_ids(data.root / "stable_ids.jsonl")
    rezoned: dict[tuple[str, str, str], list[str]] = {}
    for (name, zoo, variant), stable_id in table.items():
        if zoo in CATALOG_ZOOS:
            rezoned.setdefault((name, INTAKE_ZOO, variant), []).append(stable_id)
    extra = {key: ids[0] for key, ids in rezoned.items() if len(ids) == 1 and key not in table}
    return {**table, **extra}


def _july(path: Path) -> dict[str, str]:
    """July candidate name to its ``S001:axis=<cell>`` provenance."""

    out = {}
    for row in _tsv(path):
        axis = AXIS.search(row[4]) if len(row) > 4 else None
        out[row[0]] = f"S001:axis={axis.group(1).strip()}" if axis else "S001"
    return out


@dataclasses.dataclass(frozen=True)
class PlanRow:
    """One S-C candidate: its id and the hints it carries."""

    cand: str
    url: str | None
    framework: str | None
    paper: str | None


def _plan(path: Path) -> dict[str, list[PlanRow]]:
    """S-C candidate name to its rows. Rows with tabs inside fields are ragged, so the URL is
    the fourth field only when it is a URL, the framework the sixth only on regular rows,
    and the paper the first field that looks like an arXiv id or DOI."""

    out: dict[str, list[PlanRow]] = {}
    for row in _tsv(path):
        url = row[3] if len(row) > 3 and row[3].startswith(("http://", "https://")) else None
        framework = _value(row[5]) if len(row) >= 11 else None
        paper = next((m.group(0) for c in row[3:] if (m := PAPER.search(c))), None)
        out.setdefault(row[1], []).append(PlanRow(row[0], url, framework, paper))
    return out


def _partitions(folder: Path) -> dict[str, str]:
    out = {}
    for shard in sorted(folder.glob("c*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.setdefault(json.loads(line)["name"], f"legacy:{shard.stem}")
    return out


def _matcher(names: Iterable[str]) -> dict[str, list[str]]:
    by_norm: dict[str, list[str]] = {}
    for name in names:
        by_norm.setdefault(norm(name), []).append(name)
    return by_norm


def _lookup(name: str, table: dict, by_norm: dict[str, list[str]]) -> list:
    """Values for ``name``: the exact key, else every key with the same normalized form."""

    if name in table:
        return [table[name]]
    return [table[key] for key in by_norm.get(norm(name), ())]


def _merge(seq: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in seq if item))


def _repair_row(row: RosterRow, ids: dict, july: tuple, plan: tuple, partitions: dict,
                counts: Counter) -> RosterRow:
    stable_id = row.stable_id or ids.get((row.name, row.zoo, row.variant))
    found = list(_lookup(row.name, *july))
    hits: list[PlanRow] = [p for group in _lookup(row.name, *plan) for p in group]
    found += [f"S-C:{p.cand}" for p in hits]
    known = next((f for f in row.found_by if f.startswith("legacy:")), "legacy:unknown")
    found.append(partitions.get(row.name, known))
    urls = _merge([*row.hint_urls, row.source_url or "", *(p.url or "" for p in hits)])
    counts.update(stable_id=bool(stable_id), s001=any(f.startswith("S001") for f in found),
                  s_c=bool(hits), url_gained=not row.source_url and any(p.url for p in hits),
                  **{found[-1]: 1})
    return dataclasses.replace(
        row, stable_id=stable_id, found_by=_merge([*row.found_by, *found]), hint_urls=urls,
        hint_framework=row.hint_framework or next((p.framework for p in hits if p.framework),
                                                  None),
        hint_paper=row.hint_paper or next((p.paper for p in hits if p.paper), None),
    )


def _repair_rows(rows: list[RosterRow], data: DataLayout, sc_dir: Path | None,
                 counts: Counter, *, mint: bool) -> list[RosterRow]:
    from menagerie.identity import ensure_stable_ids

    july = _july(JULY)
    plan = _plan((sc_dir or data.root / "sweeps" / "S-C") / "IMPL_PLAN.tsv")
    roster_norms = {norm(r.name) for r in rows}
    counts.update(
        rows=len(rows), sc_candidates=sum(map(len, plan.values())),
        sc_candidates_matched=sum(len(g) for name, g in plan.items() if norm(name) in roster_norms),
    )
    ids, partitions = _stable_ids(data), _partitions(PARTITIONS)
    # Ambiguous catalog aliases cannot choose an identity for a distinct intake key.
    missing = {(r.name, r.zoo, r.variant) for r in rows
               if r.state == "active" and not r.stable_id
               and (r.name, r.zoo, r.variant) not in ids}
    counts["ids_missing"] = len(missing)
    counts["ids_minted"] = len(missing) if mint else 0
    if mint and missing:
        ensure_stable_ids(missing, data.root / "stable_ids.jsonl")
        ids = _stable_ids(data)
    july_match, plan_match = (july, _matcher(july)), (plan, _matcher(plan))
    return [_repair_row(r, ids, july_match, plan_match, partitions, counts) for r in rows]


def repair(*, data: DataLayout, sc_dir: Path | None = None, write: bool = False) -> dict[str, int]:
    """Repair provenance; writing also mints missing active ids under the roster lock."""

    counts: Counter[str] = Counter()
    if write:
        from menagerie.pipeline.packet import _locked_roster

        def change(rows: list[RosterRow]) -> list[RosterRow]:
            return _repair_rows(rows, data, sc_dir, counts, mint=True)

        _locked_roster(data, change)
    else:
        _repair_rows(read(data=data), data, sc_dir, counts, mint=False)
    return dict(sorted((k, int(v)) for k, v in counts.items()))
