"""Fill a harvester packet's authors, dates and venue from the bibliographic records of its paper.

Owned by the rung 2 metadata lane. A harvester packet carries only what a library docstring names:
a title and an arXiv id, or just the id. ``fill`` fetches the records of that id (``fetch.
bib_records``: Semantic Scholar, OpenAlex, Crossref, BibTeX, with the day cache and the S2
cooldown) and fills ``authors``, ``first_public``, ``venue`` and ``published`` only from records
that agree on the title:

- a packet with a title: the S2, OpenAlex or Crossref records whose normalized title equals it;
- a packet with only an id: the id's own record (arXiv's BibTeX, else Crossref for a DOI) and an
  S2 or OpenAlex record that agree on the title; the citation then takes that title, basis
  ``arxiv id; title agreed across records``. Agreement proves the id resolves cleanly, not that
  the docstring linked the right paper, so when the family name is in neither the title nor the
  abstract the fill is marked for a spot check (Crawler, 2026-10-05).

A field already set is never overwritten. Each fill is a packet amendment (``by: bib-harvest``)
naming the records as evidence; a disagreement is reported with the titles, never filled.
"""

from __future__ import annotations

import calendar
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout, append_jsonl, fetch, month_of, now
from menagerie.pipeline import read_json, write_json
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.citecheck import api_titles, bare, date_note, norm_title
from menagerie.pipeline.records import Amendment, Citation, ModelRecord, Packet, RosterRow, Source

BY = "bib-harvest"
SPOT_CHECK = "spot-check"  # the amendment author that confirms a held titleless fill
RETIRED = "paper-rejected"  # retained historical evidence that must never ground new claims
TITLELESS_BASIS = "arxiv id; title agreed across records"
FILLED = ("authors", "first_public", "venue", "published")
# Venue names that are a preprint server, not a venue.
PREPRINT = ("arxiv", "corr", "biorxiv", "medrxiv", "ssrn", "research square")


@dataclass(frozen=True)
class Outcome:
    """What ``fill`` did for one packet."""

    family: str
    status: str  # filled, unmatched, refused, complete, skipped
    ref: str | None = None
    title: str | None = None  # the packet's title before the fill
    fetched_title: str | None = None
    matched: tuple[str, ...] = ()
    name_match: bool | None = None
    spot_check: bool = False
    fields: tuple[str, ...] = ()
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, (), "")}


def _ids(url: str | None) -> str | None:
    """``arXiv:<id>`` or the DOI a URL names (with or without its scheme), else ``None``."""

    url = f"https://{url}" if url and re.match(r"(www\.)?(arxiv|doi)\.org/", url) else url
    arxiv, doi = fetch._arxiv_id(url or ""), fetch._doi(url or "")  # noqa: SLF001
    return f"arXiv:{arxiv}" if arxiv else doi


def _ref(packet: Packet) -> str | None:
    """The paper's arXiv id or DOI: the citation's, else the first paper source URL's."""

    cite = packet.citation
    if cite and (cite.arxiv or cite.doi):
        return f"arXiv:{cite.arxiv}" if cite.arxiv else cite.doi
    excluded = held_sources(packet)
    return next((ref for s in packet.sources if s.role == "paper" and s.sid not in excluded
                 and (ref := _ids(s.url))), None)


def _complete(cite: Citation | None) -> bool:
    return bool(cite and cite.authors and cite.first_public and cite.venue and cite.bib)


def _records(got: fetch.BibResult, store: StoreLayout) -> dict[str, Any]:
    """The JSON records plus the BibTeX text, read back from its saved blob."""

    records = dict(got.records)
    bib = next((s for s in got.sources if s.api == "bibtex" and s.blob), None)
    text = _saved_text(bib.blob, store) if bib else None
    if text is not None:
        records["bibtex"] = text
    return records


def _saved_text(sha: str, store: StoreLayout, limit: int = 2**20) -> str | None:
    """Absent retained text gives no evidence; callers may use the remaining sources."""

    try:
        path = store_mod.get(sha, store=store)
    except FileNotFoundError:
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.read(limit)


def _record_titles(records: Mapping[str, Any]) -> dict[str, str]:
    """Include an exact-title search work only when its response actually has a title."""

    titles = api_titles(records)
    if (work := records.get("openalex_search")) and work.get("title"):
        titles["openalex_search"] = work["title"]
    return titles


def _agreeing(title: str | None, titles: Mapping[str, str], own: str,
              keyed: Sequence[str] = ("s2", "openalex")) -> tuple[str | None, list[str]]:
    """The title to cite and the records that agree on it (S2, OpenAlex, Crossref only).

    With a packet title, every record whose title equals it; without one, the id's own record's
    title when a record keyed by the same id (``keyed``) agrees with it, and then every record
    that agrees.
    """

    def same(api: str, want: str) -> bool:
        return api in titles and norm_title(titles[api]) == norm_title(want)

    if title:
        return title, [a for a in (*FILLERS, "crossref") if same(a, title)]
    if own not in titles or not any(same(a, titles[own]) for a in keyed):
        return None, []
    agreeing = [a for a in FILLERS if same(a, titles[own])]
    return titles[own], agreeing + ([own] if own == "crossref" else [])


def _names_id(work: Mapping[str, Any], ref: str) -> bool:
    """Whether an OpenAlex work lists the arXiv id or DOI among its locations or ids."""

    wanted = ref.removeprefix("arXiv:").lower()
    places = [str(v) for v in (work.get("ids") or {}).values()] + [
        str(loc.get(k)) for loc in work.get("locations") or [] for k in ("landing_page_url",
                                                                       "pdf_url")]
    return any(wanted in place.lower() for place in places)


# What each record gives: authors, dates (the earliest public one is found by sorting) and venue
# names. Crossref's dates are `date-parts` lists; a preprint server is never a venue.


def _crossref_date(message: Mapping[str, Any], key: str) -> str | None:
    parts = ((message.get(key) or {}).get("date-parts") or [[]])[0]
    return "-".join(f"{int(p):02d}" for p in parts) if parts and parts[0] else None


def _message(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("message") or {}


# The records a titleless packet's id record must agree with: S2, the OpenAlex work its id names,
# and the OpenAlex work found by title (`openalex_search`, the rung 0 path's exact-title match).
FILLERS = ("s2", "openalex", "openalex_search")

AUTHORS = {
    "s2": lambda r: [a.get("name") for a in r.get("authors") or []],
    "openalex": lambda r: [(a.get("author") or {}).get("display_name")
                           for a in r.get("authorships") or []],
    "crossref": lambda r: [" ".join(filter(None, (a.get("given"), a.get("family"))))
                           for a in _message(r).get("author") or []],
}
DATES = {
    "s2": lambda r: [r.get("publicationDate"), str(r.get("year") or "")],
    "openalex": lambda r: [r.get("publication_date")],
    "crossref": lambda r: [_crossref_date(_message(r), k)
                           for k in ("published-online", "published-print", "issued")],
}
VENUES = {
    "s2": lambda r: [r.get("venue"), (r.get("publicationVenue") or {}).get("name")],
    "openalex": lambda r: [((r.get("primary_location") or {}).get("source") or {})
                           .get("display_name")],
    "crossref": lambda r: list(_message(r).get("container-title") or []),
}
for _table in (AUTHORS, DATES, VENUES):
    _table["openalex_search"] = _table["openalex"]


def _given(table: Mapping[str, Any], records: Mapping[str, Any], apis: Sequence[str]) -> list[Any]:
    """Every non-empty value the agreeing records give, in ``apis`` order."""

    return [v for api in apis for v in table[api](records.get(api) or {}) if v]


def _authors(records: Mapping[str, Any], apis: Sequence[str]) -> tuple[str, ...]:
    """The first agreeing record's author list."""

    return next((tuple(n for n in AUTHORS[a](records.get(a) or {}) if n) for a in apis
                 if any(AUTHORS[a](records.get(a) or {}))), ())


def _first_public(records: Mapping[str, Any], apis: Sequence[str]) -> str | None:
    dates = [d for d in _given(DATES, records, apis) if date_note("date", d) is None]
    return min(dates, key=_date_order) if dates else None


def _date_order(value: str) -> tuple[int, int, int, int]:
    """An imprecise date cannot precede a known earlier date in the same interval."""

    parts = [int(p) for p in value.split("-")]
    year, month = parts[0], parts[1] if len(parts) > 1 else 12
    day = parts[2] if len(parts) > 2 else calendar.monthrange(year, month)[1]
    return year, month, day, -len(parts)


def _venue(records: Mapping[str, Any], apis: Sequence[str]) -> str | None:
    return next((n for n in _given(VENUES, records, apis)
                 if not any(p in n.lower() for p in PREPRINT)), None)


def _abstract(records: Mapping[str, Any]) -> str:
    s2 = (records.get("s2") or {}).get("abstract") or ""
    index = (records.get("openalex") or {}).get("abstract_inverted_index") or {}
    words = sorted((pos, word) for word, places in index.items() for pos in places)
    return f"{s2} {' '.join(w for _, w in words)}"


def _name_match(packet: Packet, title: str, records: Mapping[str, Any]) -> bool:
    names = {bare(packet.identity.canonical), bare(packet.family[2:]),
             *(bare(a) for a in packet.identity.aliases)}
    text = bare(f"{title} {_abstract(records)}")
    return any(len(n) >= 3 and n in text for n in names)


def _base(packet: Packet, title: str) -> Citation:
    """The packet's citation, or a new one for a titleless packet from its paper source."""

    if packet.citation is not None:
        return packet.citation
    ref = _ref(packet) or ""
    return Citation(title=title, arxiv=fetch._arxiv_id(ref), doi=fetch._doi(ref),  # noqa: SLF001
                    from_=next((s.sid for s in packet.sources if s.role == "paper"), None))


def _filled(base: Citation, records: Mapping[str, Any], apis: Sequence[str],
            basis: str) -> Citation:
    """The citation with every empty field the agreeing records give; set fields are kept."""

    from menagerie.pipeline.packet import _published  # noqa: PLC0415 -- packet imports fetch

    # Crossref's date, else the OpenAlex venue record's: the title search's work before the id's.
    published = _published({a.removesuffix("_search"): records[a]
                            for a in sorted(apis, key=lambda a: a.endswith("_search"))
                            if a in records})
    agreed = _agreed_records(records, apis)
    work_id = str((agreed.get("openalex") or {}).get("id") or "").rsplit("/", 1)[-1]
    first = _first_public(records, apis)
    given = {
        "authors": _authors(records, apis), "first_public": first,
        "basis": basis if first else None, "venue": _venue(records, apis),
        "published": published if date_note("published", published) is None else None,
        "s2": (agreed.get("s2") or {}).get("paperId"), "openalex": work_id or None,
    }
    return replace(base, **{k: getattr(base, k) or v for k, v in given.items()})


def _agreed_records(records: Mapping[str, Any], apis: Sequence[str]) -> dict[str, Any]:
    """Only agreeing records may seed counts or identifier fields; prefer the title work."""

    return {a.removesuffix("_search"): records[a]
            for a in sorted(apis, key=lambda a: a.endswith("_search")) if a in records}


def _matching_sources(sources: Sequence[Source], apis: Sequence[str]) -> tuple[Source, ...]:
    """Keep the exact-title OpenAlex search distinct from the work looked up by identifier."""

    def key(source: Source) -> str | None:
        if source.api == "openalex" and urllib.parse.urlsplit(source.url or "").query:
            return "openalex_search"
        return source.api
    return tuple(s for s in sources if key(s) in apis)


def _gather(family: str, ref: str, title: str | None, *, store: StoreLayout,
            data: DataLayout) -> tuple[fetch.BibResult, dict[str, Any], list[str]]:
    """The id's records, plus the exact-title OpenAlex work; and the records keyed by the id."""

    got = fetch.bib_records(ref, family=family, store=store, data=data)
    records = _records(got, store)
    own = "bibtex" if ref.startswith("arXiv:") else "crossref"
    found, work = _title_search(title or api_titles(records).get(own), family=family,
                                store=store, data=data)
    keyed = ["s2", "openalex"]
    if work is not None:
        records["openalex_search"] = work
        got = replace(got, sources=(*got.sources, *found))
        keyed += ["openalex_search"] if _names_id(work, ref) else []
    return got, records, keyed


def _title_search(title: str | None, *, family: str, store: StoreLayout,
                  data: DataLayout) -> tuple[tuple[Source, ...], dict[str, Any] | None]:
    """The most-cited OpenAlex work whose normalized title equals ``title`` (the rung 0 path's
    search, ``research.openalex``), saved as a ``bib`` source; nothing when none matches."""

    if not title:
        return (), None
    query = urllib.parse.urlencode({"filter": f"title.search:{title}",
                                    "sort": "cited_by_count:desc", "per-page": "10"})
    shown = f"https://api.openalex.org/works?{query}"
    bib = fetch._Bib(family, store, data)  # noqa: SLF001 -- the fetch lane's saving path
    bib.call("openalex", shown)
    results = (bib.records.get("openalex") or {}).get("results") or []
    exact = [w for w in results if norm_title(w.get("title") or "") == norm_title(title)]
    return (tuple(bib.sources), exact[0]) if exact else ((), None)


def _unmatched(family: str, ref: str, title: str | None, titles: Mapping[str, str],
               shown: str | None, notes: Sequence[str]) -> Outcome:
    if not titles:
        return Outcome(family, "refused", ref, title, note="; ".join(notes) or "no record")
    listed = "; ".join(f"{a} {t[:90]!r}" for a, t in sorted(titles.items()))
    return Outcome(family, "unmatched", ref, title, shown, note=f"titles: {listed}")


def _shared_link(packet: Packet, data: DataLayout) -> bool:
    """Whether a classic packet's paper link is one its module's docs no longer give this family
    (``harvest.classic_paper``: a docstring naming several papers names none of them for one)."""

    from menagerie.pipeline import harvest  # noqa: PLC0415

    paper = next((_ids(s.url) for s in packet.sources if s.role == "paper"), None)
    for member in packet.members:
        path = data.model(member.id)
        recipe = read_json(ModelRecord, path).recipe if path.exists() else None
        if recipe is None or recipe.kind != "classic" or not recipe.adapter:
            continue
        module, _, build = recipe.adapter.partition(":")
        link, _ = harvest.classic_paper(module, build, family=packet.family)
        if paper and _ids(link) != paper:
            return True
    return False


def _skip(packet: Packet, data: DataLayout, *, force: bool = False) -> Outcome | None:
    """The outcome for a packet the fill leaves alone, else ``None``."""

    cite = packet.citation
    if citation_held(packet):
        return Outcome(packet.family, "skipped", note="held:spot-check")
    if _complete(cite) and not force:
        return Outcome(packet.family, "complete", title=cite.title if cite else None)
    if _ref(packet) is None:
        return Outcome(packet.family, "refused", title=cite.title if cite else None,
                       note="no arXiv id or DOI")
    if packet.by.startswith("harvester") and _shared_link(packet, data):
        return Outcome(packet.family, "refused", _ref(packet), cite.title if cite else None,
                       note="shared docstring: the module names several papers")
    return None


def fill(family: str, *, store: StoreLayout, data: DataLayout, write: bool = True,
         force: bool = False) -> Outcome:
    """Fill one harvester packet's citation from its paper's records; see the module docstring.

    Returns: the outcome. Side effects (when ``write`` and something was filled): the packet
    with new ``bib`` sources, the filled citation and an amendment; one citation seed line.
    """

    packet = read_json(Packet, data.packet(family))
    skipped = _skip(packet, data, force=force)
    if skipped is not None:
        return skipped
    ref, title = _ref(packet) or "", packet.citation.title if packet.citation else None
    try:
        got, records, keyed = _gather(family, ref, title, store=store, data=data)
    except (ValueError, fetch.FetchError) as exc:
        return Outcome(family, "refused", ref, title, note=str(exc)[:200])
    titles = _record_titles(records)
    own = "bibtex" if ref.startswith("arXiv:") else "crossref"
    cited, apis = _agreeing(title, titles, own, keyed)
    if not apis:
        return _unmatched(family, ref, title, titles, titles.get(own), got.notes)
    basis = f"{' and '.join(apis)} matched on title" if title else TITLELESS_BASIS
    new = _filled(_base(packet, cited or ""), records, apis, basis)
    fields = _fill_fields(packet.citation, new)
    name_match = _name_match(packet, cited or "", records)
    if not fields:
        if write:
            _seed(family, _agreed_records(records, apis), data)
        return Outcome(family, "unmatched", ref, title, cited, tuple(apis), name_match,
                       note="agreeing records give no missing field")
    spot = not title and not name_match
    if write:
        used = [*apis, *(["bibtex"] if norm_title(titles.get("bibtex", ""))
                         == norm_title(cited or "") else [])]
        _write(packet, new, _matching_sources(got.sources, used), used, fields, spot, data=data)
        _seed(family, _agreed_records(records, apis), data)
    return Outcome(family, "filled", ref, title, cited, tuple(apis), name_match, spot, fields)


def _fill_fields(old: Citation | None, new: Citation) -> tuple[str, ...]:
    """Metadata additions, including attachment of agreeing records to an existing citation."""

    fields = (() if old else ("title",)) + tuple(
        f for f in FILLED if getattr(new, f) and not (old and getattr(old, f)))
    return fields or (("bib",) if old is not None and not old.bib else ())


def _seed(family: str, records: Mapping[str, Any], data: DataLayout) -> None:
    from menagerie.pipeline.packet import _seed_line  # noqa: PLC0415

    seed = _seed_line(family, records)
    if seed is not None:
        if seed.openalex is not None:
            seed = replace(seed, from_="openalex")
        append_jsonl([seed], data.cite(month_of(seed.at)))


def _write(packet: Packet, citation: Citation, fetched: Sequence[Source], apis: Sequence[str],
           fields: Sequence[str], spot: bool, *, data: DataLayout) -> None:
    known = {s.blob for s in packet.sources if s.blob} | {s.sid for s in packet.sources}
    added = [s for s in fetched if s.blob not in known and s.sid not in known]
    sources = (*packet.sources, *added)
    fetched_blobs = {f.blob for f in fetched}
    used = {a.removesuffix("_search") for a in apis}
    evidence = tuple(s.sid for s in sources if s.role == "bib" and s.blob in fetched_blobs
                     and s.api in used)
    bibs = evidence
    why = (f"bib harvest: {', '.join(fields)} from the {' and '.join(apis)} records"
           + (" (title agreed across records keyed by the arXiv id)" if "title" in fields else
              " (title matched the docstring title)")
           + ("; family name in neither the title nor the abstract: spot check" if spot else ""))
    amendment = Amendment(at=now(), why=why, evidence=evidence or bibs,
                          changed=("citation", "sources"), by=BY)
    filled = replace(packet, sources=sources, citation=replace(citation, bib=bibs),
                     amendments=(*packet.amendments, amendment))
    write_json(filled, data.packet(packet.family))


def citation_held(packet: Packet) -> bool:
    """Whether the packet's citation is a titleless fill still waiting for its spot check.

    A ``bib-harvest`` amendment whose ``why`` says spot check holds it; a later amendment by
    ``spot-check`` (someone read the paper against the family) releases it.
    """

    held = False
    for amendment in packet.amendments:
        if amendment.by == BY and "spot check" in amendment.why:
            held = True
        elif amendment.by == SPOT_CHECK:
            held = False
    return held


def no_paper(packet: Packet) -> bool:
    """The latest explicit paper decision wins, including a later successful discovery."""

    decision = next((a for a in reversed(packet.amendments)
                     if a.by == "paper-find" and "citation" in a.changed), None)
    return bool(decision and decision.why.startswith("no_paper:"))


HELD = "held:spot-check"  # the type a held source carries where a lane would read it


def held_sources(packet: Packet) -> frozenset[str]:
    """The sources of a held fill: its paper source (the possibly wrong docstring link), the
    records fetched for it, and so any saved text and claims drawn from them."""

    retired = {sid for a in packet.amendments if a.by == RETIRED for sid in a.evidence}
    held = {s.sid for s in packet.sources if s.role in ("paper", "bib")} if citation_held(packet) else set()
    return frozenset(retired | held)


def confirm(verdicts: Sequence[Mapping[str, str]], *, data: DataLayout) -> list[str]:
    """Release held fills a check found right: a ``spot-check`` amendment per confirmed family.

    ``verdicts``: rows with ``family``, ``verdict`` (``right``, ``wrong`` or ``unsure``) and
    ``reason``. Wrong and unsure rows stay held. Returns: the families released.
    """

    released = []
    for row in verdicts:
        path = data.packet(row["family"])
        packet = read_json(Packet, path) if path.exists() else None
        if row.get("verdict") != "right" or packet is None or not citation_held(packet):
            continue
        paper = tuple(s.sid for s in packet.sources if s.role == "paper")
        amendment = Amendment(at=now(), why=f"spot check: right; {row.get('reason', '')}"[:300],
                              evidence=paper[:1], changed=("citation",), by=SPOT_CHECK)
        write_json(replace(packet, amendments=(*packet.amendments, amendment)), path)
        released.append(row["family"])
    return released


def fill_all(*, store: StoreLayout, data: DataLayout, families: Sequence[str] | None = None,
             write: bool = True) -> list[Outcome]:
    """Run ``fill`` over every harvester packet (or the named families); return the outcomes."""

    paths = sorted(data.root.glob("research/*/*.json"))
    names = families or [p.stem for p in paths]
    out = []
    for family in names:
        try:
            out.append(fill(family, store=store, data=data, write=write))
        except Exception as exc:  # noqa: BLE001 -- one bad packet must not stop the pass
            out.append(Outcome(family, "refused", note=f"{type(exc).__name__}: {str(exc)[:200]}"))
    return out



def june_notes(*, data: DataLayout, write: bool = True) -> list[str]:
    """Copy the June catalog ``notes`` into every harvester packet that has none, by amendment.

    Returns: the families changed. The harvester writes them itself from now on
    (``harvest.june_notes``); this carries them into packets written before it did.
    """

    from menagerie.pipeline import harvest  # noqa: PLC0415

    changed = []
    for path in sorted(data.root.glob("research/*/*.json")):
        packet = read_json(Packet, path)
        if not packet.by.startswith("harvester") or packet.notes_for_implementer:
            continue
        notes = harvest.june_notes([m.id for m in packet.members], data)
        if not notes:
            continue
        changed.append(packet.family)
        if write:
            amendment = Amendment(at=now(), why="June catalog notes carried into the packet",
                                  evidence=(), changed=("notes_for_implementer",), by=BY)
            write_json(replace(packet, notes_for_implementer=notes,
                               amendments=(*packet.amendments, amendment)), path)
    return changed


def set_domains(domains: Mapping[str, str], *, data: DataLayout) -> list[str]:
    """Set an empty roster ``domain`` for each stable id, with an amendment on its family packet.

    The interim domain for validated rows the June catalog never described, read from the
    packet's saved citation title (``domains``: stable id to a domain of the roster's vocabulary).
    Returns: the ids changed. A row that already has a domain, or has no roster row, is left.
    """

    from menagerie.pipeline.packet import _locked_roster  # noqa: PLC0415

    changed: list[str] = []

    def change(rows: list[RosterRow]) -> list[RosterRow]:
        out = []
        for row in rows:
            if row.stable_id in domains and not row.domain:
                changed.append(row.stable_id)
                row = replace(row, domain=domains[row.stable_id])
            out.append(row)
        return out

    _locked_roster(data, change)
    _amend_domains(changed, domains, data=data)
    return changed


def _amend_domains(changed: Sequence[str], domains: Mapping[str, str], *,
                   data: DataLayout) -> None:
    """One amendment per family packet naming the domains its members got."""

    for path in sorted(data.root.glob("research/*/*.json")):
        packet = read_json(Packet, path)
        hit = [m.id for m in packet.members if m.id in changed]
        if not hit:
            continue
        paper = tuple(s.sid for s in packet.sources if s.role == "paper")
        amendment = Amendment(
            at=now(), why="interim roster domain from the packet's citation title: "
            + ", ".join(f"{i} {domains[i]}" for i in hit),
            evidence=paper[:1], changed=tuple(f"roster:{i}.domain" for i in hit), by=BY)
        write_json(replace(packet, amendments=(*packet.amendments, amendment)), path)
