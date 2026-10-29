"""Research hypotheses from commit-pinned harvested documents, never row outcomes."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    bib_harvest,
    census,
    census_harvest,
    fetch,
    find_paper,
    now,
    read_jsonl,
    roster,
)
from menagerie.pipeline.citecheck import norm_title
from menagerie.pipeline.records import CensusRow, Dossier, dumps


def path(data: DataLayout) -> Path:
    return data.root / "sweeps" / census.SWEEP / "dossier.jsonl"


def name_agrees(title: str | None, name: str, repo: str) -> bool:
    """Both names must occur as complete normalized tokens; short names stay held."""
    text = f" {norm_title(title or '')} "
    names = (norm_title(name), norm_title(repo.split('/')[-1]))
    extra = set(text.split()) - {token for name in names for token in name.split()}
    return bool(extra) and all(len(n.replace(' ', '')) >= 3 and f" {n} " in text for n in names)




def _agreement(row: CensusRow, name: str, ref: str, title: str | None,
               store: StoreLayout, data: DataLayout) -> str | None:
    _, records, keyed = bib_harvest._gather(row.family or census._family(row.repo),
                                           ref, title, store=store, data=data)
    titles = bib_harvest._record_titles(records)
    own = "bibtex" if ref.startswith("arXiv:") else "crossref"
    agreed, apis = bib_harvest._agreeing(title, titles, own, keyed)
    paired = any(norm_title(titles.get(k, "")) == norm_title(agreed or "")
                 for k in (own, *keyed))
    return agreed if agreed and apis and paired and name_agrees(agreed, name, row.repo) else None


def _checked_candidate(row: CensusRow, name: str, ref: str, title: str | None, paper: bool,
                       store: StoreLayout, data: DataLayout) -> tuple[str | None, str | None]:
    if not paper:
        return None, None
    try:
        return _agreement(row, name, ref, title, store, data), None
    except (fetch.FetchError, OSError, ValueError) as exc:
        return None, f"{ref}: {type(exc).__name__}"


def _candidate(row: CensusRow, name: str, root: Path, store: StoreLayout,
               data: DataLayout) -> tuple[str | None, str | None, str | None, list[str]]:
    evidence = json.loads((root / "harvest-files" / f"{row.id}.json").read_text())
    point = census_harvest.checkpoint(root, row.repo)
    if evidence.get("repo") != row.repo or evidence.get("commit") != point["commit"]:
        raise ValueError("harvest file provenance does not match its checkpoint")
    candidates = [(ref, next(iter(doc.get("titles", [])), None), doc.get("paper_candidate", True))
                  for doc in evidence["documents"] for ref in doc.get("refs", [])]
    held = candidates[0][:2] if candidates else (None, None)
    warnings = []
    for ref, title, paper in list(dict.fromkeys(candidates))[:find_paper.MAX_CANDIDATES]:
        agreed, warning = _checked_candidate(row, name, ref, title, paper, store, data)
        warnings.extend([warning] if warning else [])
        if agreed:
            return ref, agreed, "agrees", warnings
    return (*held, "held" if held[0] else None, warnings)


def _record(row: CensusRow, name: str, root: Path, store: StoreLayout,
            data: DataLayout) -> tuple[Dossier, list[str]]:
    point = census_harvest.checkpoint(root, row.repo)
    paper, title, verdict, warnings = _candidate(row, name, root, store, data)
    value = Dossier(id=row.id, family=row.family or census._family(row.repo),
                   repo=row.repo, commit=point["commit"], branch=point.get("branch"),
                   built=now(), paper=paper, paper_title=title, paper_verdict=verdict,
                   license=row.license, readme_sha=row.readme_sha)
    return value, warnings


def _attempt(row: CensusRow, name: str, root: Path, store: StoreLayout,
             data: DataLayout) -> tuple[Dossier | None, dict[str, Any]]:
    if not row.eligible or row.harvested is None:
        return None, {"id": row.id, "status": "not_harvested"}
    try:
        value, warnings = _record(row, name, root, store, data)
    except (OSError, ValueError, RuntimeError) as exc:
        return None, {"id": row.id, "status": "retry", "reason": type(exc).__name__}
    return value, {"id": row.id, "status": "built", "paper_verdict": value.paper_verdict,
                   "warnings": warnings}

def build(ids: Sequence[str] | None, *, data: DataLayout, store: StoreLayout,
          harvest: Path | None = None, write: bool = True) -> list[dict[str, Any]]:
    """Bibliographic I/O runs on the coordinator, through the existing API wrapper."""
    file = harvest or data.root / "sweeps" / census.SWEEP / "harvest.jsonl"
    if not write:
        return _build(ids, data=data, store=store, file=file, write=False)
    file.parent.mkdir(parents=True, exist_ok=True)
    with (file.parent / ".dossier.lock").open("a") as lock:
        census._lock(lock.fileno())
        return _build(ids, data=data, store=store, file=file, write=True)


def _build(ids: Sequence[str] | None, *, data: DataLayout, store: StoreLayout,
           file: Path, write: bool) -> list[dict[str, Any]]:
    root = file.parent
    wanted = set(ids) if ids is not None else None
    names = {r.stable_id: r.name for r in roster.read(data=data)}
    target = root / "dossier.jsonl"
    existing = {d.id: d for d in read_jsonl(Dossier, target)} if target.exists() else {}
    results = []
    for row in read_jsonl(CensusRow, file):
        if wanted is not None and row.id not in wanted:
            continue
        value, result = _attempt(row, names.get(row.id, ""), root, store, data)
        results.append(result)
        if value is not None:
            existing[row.id] = value
    if write:
        target = root / "dossier.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".jsonl.tmp")
        temp.write_text("".join(dumps(existing[k]) + "\n" for k in sorted(existing)), encoding="utf-8")
        temp.replace(target)
    return results


def marker(value: Dossier) -> str:
    return f"dossier-confirmed:{value.id}:{value.commit}:{value.paper}"


def template(prompt: str) -> tuple[str, str]:
    before, begin, rest = prompt.partition("<!-- DOSSIER -->\n")
    section, end, after = rest.partition("<!-- /DOSSIER -->")
    if not begin or not end:
        raise ValueError("2a prompt needs its dossier section boundaries")
    return before + after, section


def brief(ids: Sequence[str], *, data: DataLayout, enabled: bool,
          records: Sequence[Dossier] | None = None, section: str | None = None) -> str:
    """No dossier or an omitted opt-in leaves the ordinary 2a brief unchanged."""
    if not enabled or not path(data).exists():
        return ""
    records = read_jsonl(Dossier, path(data)) if records is None else records
    values = [d for d in records if d.id in ids]
    if not values:
        return ""
    if section is None:
        prompt = Path(__file__).parent / "prompts" / "2a.md"
        _, section = template(prompt.read_text(encoding="utf-8"))
    proposals = "".join(f"```json\n{dumps(d)}\n```\nConfirmation marker: `{marker(d)}`\n"
                        for d in values)
    return section.replace("{DOSSIER}", "\n\n" + proposals)
