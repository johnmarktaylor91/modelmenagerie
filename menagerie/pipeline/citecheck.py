"""Quality checks on scraped metadata: citation formats, SPDX ids, titles against API records.

Owned by the rung 2 metadata lane. Every check returns typed notes (``<kind>: <detail>``) and
never raises on data already written, so ``packet write`` can refuse a draft with them and a
publish can report what it finds on existing packets without crashing.

- ``citation_notes``: ``first_public`` and ``published`` are ``YYYY``, ``YYYY-MM`` or
  ``YYYY-MM-DD`` with a real month and day, between ``EARLIEST_YEAR`` and next year; any year
  ``basis`` names is in that range too; ``arxiv``, ``doi`` and ``openreview`` match their patterns.
- ``spdx_notes``: a license source's ``spdx`` is ``unknown`` or an expression of ids on the
  pinned SPDX license list (``assets/spdx-licenses.json``); a deprecated id is noted.
- ``title_note``: the citation title against the titles the bibliographic records give for its
  identifier (the rung 0 path's normalization); a mismatch is a typed note, never a fill.
- ``duplicate_notes``: two model records with one natural key (name, zoo, variant) or one id.
"""

from __future__ import annotations

import datetime
import functools
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout
from menagerie.pipeline.records import Citation, ModelRecord, Packet, RosterRow

EARLIEST_YEAR = 1940
DATE = re.compile(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?")
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?")
DOI_ID = re.compile(r"10\.\d{4,9}/\S+")
OPENREVIEW_ID = re.compile(r"[A-Za-z0-9_-]{6,20}")
SPDX_LIST = Path(__file__).with_name("assets") / "spdx-licenses.json"
SPDX_SENTINELS = frozenset({"unknown"})
SPDX_OPERATORS = re.compile(r"\s+(?:AND|OR)\s+")
DOCUMENT_TYPES = ("application/pdf", "text/html", "application/xhtml", "application/octet-stream")


def norm_title(title: str) -> str:
    """Lower-case words of letters and digits: the title comparison of the rung 0 path."""

    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def bare(text: str) -> str:
    """Letters and digits only, lower-cased: for finding a family name inside a title."""

    return re.sub(r"[^a-z0-9]", "", text.lower())


def date_note(field: str, value: str | None) -> str | None:
    """Return why a citation date is malformed or out of range, or ``None`` when it is fine."""

    if value is None:
        return None
    match = DATE.fullmatch(value)
    if not match:
        return f"date-format: {field} {value!r} is not YYYY, YYYY-MM or YYYY-MM-DD"
    year, month, day = (int(g) if g else None for g in match.groups())
    if not EARLIEST_YEAR <= year <= datetime.date.today().year + 1:  # type: ignore[operator]
        return f"date-range: {field} {value!r} is outside {EARLIEST_YEAR} to next year"
    try:
        datetime.date(year, month or 1, day or 1)  # type: ignore[arg-type]
    except ValueError:
        return f"date-format: {field} {value!r} is not a calendar date"
    return None


def year_of(value: str | None) -> int | None:
    """The year of a citation date that passes ``date_note``, else ``None`` (never raises)."""

    return int(value[:4]) if value and date_note("date", value) is None else None


def citation_notes(citation: Citation | None) -> list[str]:
    """Format and range problems of one packet citation."""

    if citation is None:
        return []
    notes = [n for n in (date_note("first_public", citation.first_public),
                         date_note("published", citation.published)) if n]
    for year in re.findall(r"(?<!\d)(\d{4})(?!\d)", citation.basis or ""):
        if not EARLIEST_YEAR <= int(year) <= datetime.date.today().year + 1:
            notes.append(f"date-range: basis names {year}")
    patterns = (("arxiv", ARXIV_ID), ("doi", DOI_ID), ("openreview", OPENREVIEW_ID))
    for name, pattern in patterns:
        value = getattr(citation, name)
        if value is not None and not pattern.fullmatch(value):
            notes.append(f"id-format: {name} {value!r} does not match {pattern.pattern}")
    return notes


@functools.cache
def spdx_ids() -> tuple[frozenset[str], frozenset[str]]:
    """The pinned SPDX license list: current ids and deprecated ids."""

    listed = json.loads(SPDX_LIST.read_text(encoding="utf-8"))
    return frozenset(listed["licenses"]), frozenset(listed["deprecated"])


def spdx_note(value: str | None) -> str | None:
    """Return why an ``spdx`` value is not an SPDX expression, or note a deprecated id.

    An expression joins ids with ``AND`` and ``OR`` (parentheses allowed); ``WITH`` names an
    exception, which is not a license id and is not checked; ``LicenseRef-`` ids are local.
    """

    if value is None or value in SPDX_SENTINELS:
        return None
    current, deprecated = spdx_ids()
    terms = [t.strip() for t in SPDX_OPERATORS.split(value.replace("(", " ").replace(")", " "))]
    ids = [t.split(" WITH ")[0].strip().removesuffix("+") for t in terms if t]
    unknown = [t for t in ids if t not in current and t not in deprecated
               and not t.startswith("LicenseRef-")]
    if unknown or not ids:
        return f"spdx-invalid: {value!r} names {unknown or 'nothing'} not on the SPDX list"
    old = [t for t in ids if t in deprecated]
    return f"spdx-deprecated: {value!r} uses deprecated {old}" if old else None


def spdx_notes(packet: Packet) -> list[str]:
    """SPDX problems of a packet's license sources, each with its source id."""

    return [f"{note} (source {s.sid})" for s in packet.sources
            if s.role == "license" and (note := spdx_note(s.spdx))]


def api_titles(records: Mapping[str, Any]) -> dict[str, str]:
    """The title each bibliographic record gives: S2, OpenAlex, Crossref and the BibTeX text."""

    titles = {
        "s2": (records.get("s2") or {}).get("title"),
        "openalex": (records.get("openalex") or {}).get("title")
        or (records.get("openalex") or {}).get("display_name"),
        "crossref": next(iter(((records.get("crossref") or {}).get("message") or {})
                              .get("title") or []), None),
        "bibtex": bibtex_title(records.get("bibtex") or ""),
    }
    return {api: title for api, title in titles.items() if isinstance(title, str) and title}


def bibtex_title(text: str) -> str | None:
    """The ``title`` field of a BibTeX entry, braces removed."""

    match = re.search(r"\btitle\s*=\s*[{\"](.+?)[}\"]\s*,?\s*\n", text, re.IGNORECASE | re.DOTALL)
    return re.sub(r"[{}]", "", " ".join(match.group(1).split())) if match else None


def title_note(title: str, records: Mapping[str, Any]) -> str | None:
    """A typed note when no record agrees with the citation title; ``None`` when one does.

    No record with a title gives no note: there is nothing to compare against.
    """

    titles = api_titles(records)
    if not titles or any(norm_title(t) == norm_title(title) for t in titles.values()):
        return None
    shown = ", ".join(f"{api} {t[:80]!r}" for api, t in sorted(titles.items()))
    return f"title-mismatch: citation {title[:80]!r}; records give {shown}"


def duplicate_notes(models: Iterable[ModelRecord], roster: Iterable[RosterRow] = ()) -> list[str]:
    """Stable ids or natural keys (name, zoo, variant) held twice: among model records, and
    among active roster rows (June's catalog check; a ``duplicate_of`` row is a typed duplicate)."""

    def notes_for(what: str, rows: list[tuple[str | None, tuple[str, str, str]]]) -> list[str]:
        ids = Counter(i for i, _ in rows if i)
        keys = Counter(key for _, key in rows)
        found = [f"duplicate-id: {i} in {n} {what}" for i, n in sorted(ids.items()) if n > 1]
        found += [f"duplicate-key: {key} in {n} {what} ("
                  + ", ".join(sorted(str(i) for i, k in rows if k == key)) + ")"
                  for key, n in sorted(keys.items()) if n > 1]
        return found

    return (notes_for("model records", [(m.id, (m.name, m.zoo, m.variant)) for m in models])
            + notes_for("roster rows", [(r.stable_id, (r.name, r.zoo, r.variant))
                                        for r in roster if r.state == "active"]))


def _holders(data: DataLayout) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """The families whose packets hold each blob, and each source URL."""

    from menagerie.pipeline import read_json  # noqa: PLC0415

    by_blob: dict[str, set[str]] = {}
    by_url: dict[str, set[str]] = {}
    for path in sorted(data.root.glob("research/*/*.json")):
        packet = read_json(Packet, path)
        for source in packet.sources:
            for key, table in ((source.blob, by_blob), (source.url, by_url)):
                if key:
                    table.setdefault(key, set()).add(packet.family)
    return by_blob, by_url


def _document_problem(line: Any, store: StoreLayout) -> str | None:
    """Why one saved document is not what was fetched, or ``None``."""

    from menagerie.pipeline import fetch  # noqa: PLC0415

    blob = store.blob(line.sha256)
    if not blob.is_file():
        return "missing-blob"
    url = line.url or ""
    pdf = ("arxiv.org/pdf/" in url or url.lower().endswith(".pdf")
           or line.mime.startswith("application/pdf"))
    problem = fetch.paper_problem(blob.read_bytes(), url=url, mime=line.mime, pdf=pdf)
    return f"page-inaccessible: {problem}" if problem else None


def audit_documents(*, store: StoreLayout, data: DataLayout) -> dict[str, Any]:
    """Check every saved PDF and web page in the store as a fresh paper download is checked.

    Each document of the store index (not derived text) goes through ``fetch.paper_problem``; a
    blob fetched as a PDF (an ``arxiv.org/pdf`` or ``.pdf`` URL, or a PDF type) must parse as one.
    Returns: counts, plus one entry per bad or missing blob with the families whose packets hold
    it (by blob, else by URL).
    """

    from menagerie.pipeline import read_jsonl  # noqa: PLC0415
    from menagerie.pipeline.records import StoreIndexLine  # noqa: PLC0415

    by_blob, by_url = _holders(data)
    documents = {line.sha256: line for shard in sorted(data.root.glob("store-index/*.jsonl"))
                 for line in read_jsonl(StoreIndexLine, shard)
                 if not line.derived_from and line.mime.startswith(DOCUMENT_TYPES)}
    found: list[dict[str, Any]] = []
    counts = Counter[str]()
    for line in documents.values():
        problem = _document_problem(line, store)
        counts["missing" if problem == "missing-blob" else "bad" if problem else "ok"] += 1
        if problem:
            found.append({"sha256": line.sha256, "url": line.url, "mime": line.mime,
                          "problem": problem, "families": sorted(
                              by_blob.get(line.sha256) or by_url.get(line.url or "") or ())})
    return {"documents": len(documents), **counts, "found": found}


def check_packets(*, data: DataLayout) -> dict[str, Any]:
    """Every packet's citation and SPDX notes, and duplicates across records and the roster."""

    from menagerie.pipeline import read_json, roster  # noqa: PLC0415

    notes: dict[str, list[str]] = {}
    for path in sorted(data.root.glob("research/*/*.json")):
        packet = read_json(Packet, path)
        found = citation_notes(packet.citation) + spdx_notes(packet)
        if found:
            notes[packet.family] = found
    models = [read_json(ModelRecord, p) for p in sorted(data.root.glob("models/*/*.json"))]
    duplicates = duplicate_notes(models, roster.read(data=data))
    return {"packets": notes, "duplicates": duplicates}
