"""Paper discovery from retained evidence and exact titles, then bibliographic trickles.

Discovery never guesses a paper from a model name. Saved identifiers require title agreement
under bib-harvest's rule. Unresolved families become needs_2a; find never starts agents.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from menagerie.pipeline import (
    DataLayout, StoreLayout, bib_harvest, fetch, now, read_json, read_jsonl, write_json,
)
from menagerie.pipeline.citecheck import api_titles, norm_title
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import (
    FAMILY_ID, Amendment, Citation, CitationLine, Dossier, Packet, RecordError, Source, from_dict,
    to_dict,
)

BY = "paper-find"
HARVEST_BY = "paper-harvest"
# Harvest retries (coverage rule: a packet is never dropped from harvesting for good). An attempt
# that found nothing new waits RETRY_DAYS plus 0 to RETRY_SPREAD_DAYS - 1 days by family unless the
# packet's inputs change; a passing API refusal or failure waits REFUSED_RETRY_DAYS.
RETRY_DAYS = 14
RETRY_SPREAD_DAYS = 7
REFUSED_RETRY_DAYS = 2
LASTING_REFUSALS = ("shared docstring", "no arXiv id or DOI")
INPUTS = re.compile(r"; inputs ([0-9a-f]{16})(?:;|$)")
PROMPT = Path(__file__).with_name("prompts") / "2a-paper.md"
BATCH_SIZE = 15
TEXT_LIMIT = 2**20
MAX_CANDIDATES = 8
MAX_TITLES = 3
IDENTIFIER = re.compile(
    r"(?:arxiv(?:\.org/(?:abs|pdf)/|\s*:\s*))"
    r"(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
    r"|(?:https?://(?:dx\.)?doi\.org/|\bdoi\s*:\s*)(10\.\d{4,9}/[^\s<>\"}]+)", re.I)


def _packets(families: Sequence[str] | None, data: DataLayout) -> list[Packet]:
    paths = ([_packet_path(f, data) for f in dict.fromkeys(families)] if families is not None
             else sorted(data.root.glob("research/*/*.json")))
    return [read_json(Packet, p) for p in paths]


def _packet_path(family: str, data: DataLayout) -> Path:
    if not isinstance(family, str) or not re.fullmatch(FAMILY_ID, family):
        raise RecordError("paper family must be a canonical family id")
    return data.packet(family)


def _text(packet: Packet, store: StoreLayout) -> list[tuple[str, str]]:
    """Retained text and explicit source URLs, excluding held paper material."""

    held = bib_harvest.held_sources(packet)
    texts = []
    for source in packet.sources:
        if source.sid in held:
            continue
        texts.append((source.sid, source.url or ""))
        local = _local_code(source)
        if local is not None:
            texts.append((source.sid, local))
        sha = source.text or (source.blob if source.role in ("page", "bib") else None)
        text = bib_harvest._saved_text(sha, store, TEXT_LIMIT) if sha else None  # noqa: SLF001
        if text is not None:
            texts.append((source.sid, text))
    return texts


def _local_code(source: Source) -> str | None:
    """Read bounded repository code evidence without following paths outside the checkout."""

    if source.role != "code" or not source.url or not source.url.startswith("menagerie/"):
        return None
    root = Path(__file__).resolve().parents[2]
    path = (root / source.url).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.read(TEXT_LIMIT)


def _refs(texts: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    found: dict[str, str] = {}
    for sid, text in texts:
        for match in IDENTIFIER.finditer(text):
            arxiv, doi = match.groups()
            ref = f"arXiv:{arxiv}" if arxiv else (doi or "").rstrip(".,;)]")
            found.setdefault(ref, sid)
    return list(found.items())[:MAX_CANDIDATES]


def _cff_paper(text: str) -> str:
    preferred = re.search(r"(?m)^preferred-citation:\s*\n((?:[ \t]+.*\n?)*)", text)
    if not preferred:
        return ""
    kind = re.search(r"(?m)^\s*type:\s*([^\n]+)$", preferred[1])
    papers = {"article", "conference-paper", "proceedings", "report", "thesis", "book"}
    return preferred[1] if kind and kind[1].strip().strip("\"'") in papers else ""


def _titles(packet: Packet | None, texts: Sequence[tuple[str, str]]) -> list[str]:
    if packet is not None and packet.citation and packet.citation.title:
        return [packet.citation.title]
    titles = []
    for _, text in texts:
        cff = _cff_paper(text)
        for match in re.finditer(r"(?m)^\s*title:\s*([^\n]+)$", cff):
            value = match[1].strip()
            if value and value[0] not in "|>":
                titles.append(value.strip("\"'"))
        titles.extend(re.findall(r'\btitle\s*=\s*[{"]([^\n]+?)[}"]\s*,', text, re.I))
        titles.extend(re.findall(r'"([^"\n]{15,200})"\s*[,\n]\s*(?:arXiv|https?://)', text))
    return list(dict.fromkeys(titles))[:MAX_TITLES]


def _arxiv_title(title: str, *, family: str, store: StoreLayout,
                  data: DataLayout) -> list[str]:
    query = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "max_results": 10})
    url = f"https://export.arxiv.org/api/query?{query}"
    got, _ = fetch._bib_get("arxiv", url, family=family)  # noqa: SLF001
    if got is None:
        return []
    fetch._put_bytes(got.body, store=store, data=data, mime="application/atom+xml",  # noqa: SLF001
                     url=url)
    try:
        feed = ET.fromstring(got.body)
    except ET.ParseError:
        raise fetch.FetchError("arXiv title search: invalid Atom") from None
    ns = {"a": "http://www.w3.org/2005/Atom"}
    return [f"arXiv:{ref}" for entry in feed.findall("a:entry", ns)
            if norm_title(entry.findtext("a:title", "", ns)) == norm_title(title)
            and (ref := fetch._arxiv_id(entry.findtext("a:id", "", ns)))]  # noqa: SLF001


def _search_refs(title: str, packet: Packet, store: StoreLayout, data: DataLayout
                  ) -> list[tuple[str, str | None]]:
    try:
        refs = [(r, None) for r in _arxiv_title(title, family=packet.family, store=store, data=data)]
    except fetch.FetchError:
        refs = []  # An unavailable arXiv endpoint still permits exact-title OpenAlex search.
    if refs:
        return refs
    _, work = bib_harvest._title_search(title, family=packet.family, store=store,  # noqa: SLF001
                                       data=data)
    if work:
        urls = [work.get("doi"), *((work.get("ids") or {}).values())]
        urls += [loc.get(k) for loc in work.get("locations") or []
                 for k in ("landing_page_url", "pdf_url")]
        refs += [(ref, None) for u in urls if isinstance(u, str)
                 and (ref := bib_harvest._ids(u))]  # noqa: SLF001
    return list(dict.fromkeys(refs))[:MAX_CANDIDATES]


def _agree(packet: Packet, ref: str, sid: str | None, title: str | None,
           store: StoreLayout, data: DataLayout) -> tuple[Citation, tuple, bool] | None:
    got, records, keyed = bib_harvest._gather(packet.family, ref, title,  # noqa: SLF001
                                             store=store, data=data)
    titles = api_titles(records)
    if (work := records.get("openalex_search")) and work.get("title"):
        titles["openalex_search"] = work["title"]
    own = "bibtex" if ref.startswith("arXiv:") else "crossref"
    cited, apis = bib_harvest._agreeing(title, titles, own, keyed)  # noqa: SLF001
    if not cited or not apis:
        return None
    if not any(norm_title(titles.get(api, "")) == norm_title(cited) for api in (own, *keyed)):
        return None
    arxiv, doi = fetch._arxiv_id(ref), fetch._doi(ref)  # noqa: SLF001
    base = Citation(title=cited, arxiv=arxiv, doi=doi, from_=sid, basis="exact-title discovery")
    citation = bib_harvest._filled(base, records, apis, "exact-title discovery")  # noqa: SLF001
    used = set(apis) | ({"bibtex"} if norm_title(titles.get("bibtex", "")) == norm_title(cited)
                       else set())
    sources = bib_harvest._matching_sources(got.sources, used)  # noqa: SLF001
    citation = replace(citation, bib=tuple(s.sid for s in sources))
    trusted_title = packet.citation and norm_title(packet.citation.title) == norm_title(title or "")
    spot = not trusted_title and not bib_harvest._name_match(packet, cited, records)  # noqa: SLF001
    return citation, sources, spot


def _amend(packet: Packet, citation: Citation, sources: tuple, spot: bool,
            data: DataLayout, *, verified: str | None = None,
            why: str = "paper find: identifier accepted under agreeing-title rule") -> None:
    known = {s.sid for s in packet.sources}
    added = tuple(s for s in sources if s.sid not in known)
    amendment = Amendment(at=now(), why=why,
                          evidence=citation.bib, changed=("citation", "sources"), by=BY)
    amendments = (*packet.amendments, amendment)
    retired = tuple(s.sid for s in packet.sources if s.role in ("paper", "bib")
                    and s.sid not in citation.bib and s.sid != citation.from_)
    if retired:
        amendments += (Amendment(at=now(), why="paper replacement: prior paper evidence excluded",
                                  evidence=retired, changed=("sources",),
                                  by=bib_harvest.RETIRED),)
    if spot:
        amendments += (Amendment(at=now(), why="titleless discovery: spot check",
                                 evidence=citation.bib, changed=("citation",),
                                 by=bib_harvest.BY),)
    if verified:
        amendments += (Amendment(at=now(), why=f"spot check: right; {verified}"[:300],
                                  evidence=tuple(filter(None, (citation.from_,))),
                                  changed=("citation",), by=bib_harvest.SPOT_CHECK),)
    write_json(replace(packet, citation=citation, sources=(*packet.sources, *added),
                       availability=replace(packet.availability, paper="found"),
                       amendments=amendments), data.packet(packet.family))


def _one(packet: Packet, store: StoreLayout, data: DataLayout, write: bool) -> dict[str, Any]:
    row = {"family": packet.family}
    if bib_harvest.no_paper(packet):
        return {**row, "status": "no_paper"}
    if bib_harvest.citation_held(packet):
        return {**row, "status": "held"}
    if packet.citation and (packet.citation.arxiv or packet.citation.doi):
        return {**row, "status": "complete", "ref": bib_harvest._ref(packet)}  # noqa: SLF001
    agreed = _discover(packet, store, data)
    if len(agreed) != 1:
        return {**row, "status": "needs_2a", "reason": "ambiguous" if agreed else "no_agreement"}
    citation, sources, spot = agreed[0]
    if write:
        _amend(packet, citation, sources, spot, data)
    return {**row, "status": "held" if spot else "found",
            "ref": f"arXiv:{citation.arxiv}" if citation.arxiv else citation.doi,
            "title": citation.title, "evidence": list(citation.bib)}


def _discover(packet: Packet, store: StoreLayout, data: DataLayout) -> list[tuple]:
    """Bounded evidence candidates and exact-title matches; ambiguous hits remain separate."""

    texts = _text(packet, store)
    titles = _titles(packet, texts)
    paper_ids = _refs([(s.sid, s.url or "") for s in packet.sources if s.role == "paper"
                       and s.sid not in bib_harvest.held_sources(packet)])
    candidates = paper_ids or _refs(texts)
    unique = {}
    for ref, sid in candidates:
        unique.setdefault(ref, sid)
    agreed = []
    for ref, sid in list(unique.items())[:MAX_CANDIDATES]:
        hit = _agree(packet, ref, sid, titles[0] if titles else None, store, data)
        if hit:
            agreed.append(hit)
    agreed.extend(_title_matches(packet, titles, store, data))
    return list({(c.openalex or c.arxiv or c.doi): (c, s, spot) for c, s, spot in agreed}.values())


def _title_matches(packet: Packet, titles: Sequence[str], store: StoreLayout,
                   data: DataLayout) -> list[tuple]:
    """Keep each exact-title query paired with the candidates it returned."""

    matches = []
    for title in titles:
        for ref, sid in _search_refs(title, packet, store, data):
            hit = _agree(packet, ref, sid, title, store, data)
            if hit:
                matches.append(hit)
    return matches


def find(families: list[str] | None, *, data: DataLayout, store: StoreLayout,
         write: bool = True, from_dossier: bool = False) -> list[dict[str, Any]]:
    """Discover identifiers; dry-run saves API evidence but does not edit packets or launch."""

    rows = []
    index = _dossier_index(data) if from_dossier else {}
    for packet in _packets(families, data):
        try:
            rows.append(_dossier_one(packet, store, data, write,
                                     [index[m.id] for m in packet.members if m.id in index])
                        if from_dossier
                        else _one(packet, store, data, write))
        except (fetch.FetchError, OSError, ValueError) as exc:
            rows.append({"family": packet.family, "status": "needs_2a",
                         "reason": f"{type(exc).__name__}: {str(exc)[:160]}"})
    return rows



def _dossier_index(data: DataLayout) -> dict[str, Dossier]:
    from menagerie.pipeline import dossier  # noqa: PLC0415

    file = dossier.path(data)
    return {d.id: d for d in read_jsonl(Dossier, file)} if file.exists() else {}



def _dossier_source(packet: Packet, value: Dossier) -> Source | None:
    return next((s for s in packet.sources if s.role == "code" and s.blob and s.rev == value.commit and
                 (s.url or "").rstrip("/").lower() == f"https://github.com/{value.repo}".lower()), None)


def _dossier_confirmed(packet: Packet, value: Dossier) -> bool:
    from menagerie.pipeline import dossier  # noqa: PLC0415

    pinned = _dossier_source(packet, value) is not None
    return (value.paper_verdict == "agrees" and pinned and
            dossier.marker(value) in packet.search.note.splitlines())

def _dossier_hit(packet: Packet, value: Dossier, store: StoreLayout,
                 data: DataLayout) -> tuple[Citation, tuple] | None:
    """Revalidate a confirmed candidate's identifier and title against saved code evidence."""
    source = _dossier_source(packet, value)
    if source is None:
        return None
    got = _agree(packet, value.paper or "", source.sid, value.paper_title, store, data)
    if got is None:
        return None
    citation, sources, spot = got
    if spot or norm_title(citation.title or "") != norm_title(value.paper_title or ""):
        return None
    return citation, sources


def _dossier_one(packet: Packet, store: StoreLayout, data: DataLayout,
                 write: bool, values: Sequence[Dossier]) -> dict[str, Any]:
    result = {"family": packet.family, "status": "held"}
    if bib_harvest.no_paper(packet):
        return {"family": packet.family, "status": "no_paper"}
    if bib_harvest.citation_held(packet):
        return result
    if packet.citation and (packet.citation.arxiv or packet.citation.doi):
        return {"family": packet.family, "status": "complete"}
    for value in values:
        if not _dossier_confirmed(packet, value):
            continue
        got = _dossier_hit(packet, value, store, data)
        if got is None:
            continue
        citation, sources = got
        if write:
            from menagerie.pipeline import dossier  # noqa: PLC0415

            _amend(packet, citation, sources, False, data,
                   why=f"paper find: confirmed {dossier.marker(value)}; README {value.readme_sha}")
        return {"family": packet.family, "status": "found", "ref": value.paper}
    return {**result, "reason": "no confirmed dossier with passing title evidence"}

def harvest(families: list[str] | None, *, data: DataLayout, store: StoreLayout,
            limit: int | None = None, write: bool = True) -> list[dict[str, Any]]:
    """Idempotent bibliographic trickle; limit counts eligible IDs, not examined packets.

    Without ``write`` it only lists the packets it would try now (status ``due``): no API call,
    no store blob, no index or citation line, no packet change.
    """

    if limit is not None and limit < 0:
        raise RecordError("paper harvest limit must be nonnegative")
    seeded = {line.family for p in data.root.glob("cite/*.jsonl")
              for line in read_jsonl(CitationLine, p) if line.openalex is not None}
    due = _due(_packets(families, data), seeded, limit)
    if not write:
        return [{"family": p.family, "status": "due", "ref": bib_harvest._ref(p)}  # noqa: SLF001
                for p in due]
    return [_harvest_one(p, seeded, data, store, write).to_json() for p in due]


def _due(packets: Sequence[Packet], seeded: set[str], limit: int | None) -> list[Packet]:
    """The packets a harvest tries now, in order, at most ``limit``."""

    due: list[Packet] = []
    for packet in packets:
        if limit is not None and len(due) >= limit:
            break
        if not bib_harvest._ref(packet) or bib_harvest.citation_held(packet):  # noqa: SLF001
            continue
        if not _harvested(packet, seeded):
            due.append(packet)
    return due


def _harvest_one(packet: Packet, seeded: set[str], data: DataLayout,
                 store: StoreLayout, write: bool) -> bib_harvest.Outcome:
    """Isolate one packet's API failure and record the attempt before advancing the trickle."""

    try:
        got = bib_harvest.fill(packet.family, data=data, store=store, write=write,
                               force=packet.family not in seeded)
    except (OSError, ValueError, fetch.FetchError) as exc:
        got = bib_harvest.Outcome(packet.family, "refused", note=type(exc).__name__)
    if write:
        _mark_harvest(packet.family, got, data)
    return got


def _harvested(packet: Packet, seeded: set[str]) -> bool:
    """Complete records stop; an attempt waits for its retry time unless its inputs changed
    (``_recently_tried``); preprints without a venue retry only on a later day."""

    cite = packet.citation
    if cite is not None and packet.family in seeded and bib_harvest._complete(cite):  # noqa: SLF001
        return True
    if _recently_tried(packet):
        return True
    if cite is None or packet.family not in seeded:
        return False
    return bool(cite.bib and cite.authors and cite.first_public and any(
        s.sid in cite.bib and (s.fetched or "")[:10] == now()[:10] for s in packet.sources))


def harvest_inputs(packet: Packet) -> str:
    """A digest of what a harvest's answer depends on: the paper id it asks about, the citation
    it would fill (title, ids, every filled field, its bib sources), the unheld paper links and
    the spot-check hold. Any change to them makes the packet due again at once."""

    held = bib_harvest.held_sources(packet)
    body = {"ref": bib_harvest._ref(packet),  # noqa: SLF001
            "citation": to_dict(packet.citation) if packet.citation else None,
            "papers": sorted([s.sid, s.url or ""] for s in packet.sources
                             if s.role == "paper" and s.sid not in held),
            "held": bib_harvest.citation_held(packet)}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _retry_days(family: str, why: str) -> int:
    """Days until a packet's next attempt: soon after a passing API refusal or failure, else
    ``RETRY_DAYS`` spread by family so one round's packets do not all come due on one day."""

    status = why.removeprefix("paper harvest: ").split(";", 1)[0]
    if status == "refused" and not any(x in why for x in LASTING_REFUSALS):
        return REFUSED_RETRY_DAYS
    return RETRY_DAYS + int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % RETRY_SPREAD_DAYS


def _days_since(at: str) -> float:
    then = datetime.fromisoformat(at.replace("Z", "+00:00"))
    return (datetime.fromisoformat(now().replace("Z", "+00:00")) - then).total_seconds() / 86400


def _recently_tried(packet: Packet) -> bool:
    """Whether the last harvest attempt saw the packet's current inputs and its retry time has
    not come. A marker names its inputs (``harvest_inputs`` after the attempt); an older marker
    without them counts on its own day, and later only while no other amendment followed it.
    Never final: every packet comes due again (coverage rule, decisions 2026-10-08)."""

    marks = [i for i, a in enumerate(packet.amendments) if a.by == HARVEST_BY]
    if not marks:
        return False
    last = packet.amendments[marks[-1]]
    key = INPUTS.search(last.why)
    if key is not None and key.group(1) != harvest_inputs(packet):
        return False
    if key is None and last.at[:10] != now()[:10] and any(
            a.by != HARVEST_BY for a in packet.amendments[marks[-1] + 1:]):
        return False
    return _days_since(last.at) < _retry_days(packet.family, last.why)


def _mark_harvest(family: str, outcome: bib_harvest.Outcome, data: DataLayout) -> None:
    """Attempt markers prevent repeated API refusals from starving the rest of a trickle; each
    names the packet's inputs after the attempt, so an unchanged packet waits its retry time."""

    packet = read_json(Packet, data.packet(family))
    why = f"paper harvest: {outcome.status}; inputs {harvest_inputs(packet)}; {outcome.note}"
    amendment = Amendment(at=now(), why=why.rstrip("; ")[:300],
                          evidence=packet.citation.bib if packet.citation else (),
                          changed=("paper_harvest",), by=HARVEST_BY)
    write_json(replace(packet, amendments=(*packet.amendments, amendment)), data.packet(family))


def batches(families: Sequence[str], *, data: DataLayout,
             parent: str | None = None) -> dict[str, Any]:
    """A ready research-launch manifest, at most 15 families per paper-only lane."""

    packets = _packets(families, data)
    return {"parent": parent, "batches": [
        {"label": f"paper-{group[0].family}", "role": "model-implementation", "paper_only": True,
         "web_search_usd": 0, "families": [{"family": p.family,
                                            "ids": [m.id for m in p.members]} for p in group]}
        for start in range(0, len(packets), BATCH_SIZE)
        if (group := packets[start:start + BATCH_SIZE])]}


def brief(batch: Mapping[str, Any], *, data: DataLayout, lane_dir: Path) -> str:
    """Paper-only brief: no re-research of model implementations."""

    from menagerie.pipeline.research import _pipeline_command, _result  # noqa: PLC0415

    items = batch["families"]
    if len(items) > BATCH_SIZE:
        raise RecordError("paper-only batch exceeds 15 families")
    command = _pipeline_command(data, lane_dir / "drafts")
    prompt = PROMPT.read_text(encoding="utf-8").replace("{PIPE}", command)
    entries = [f"{item['family']}: {data.packet(item['family'])}" for item in items]
    return (prompt + "\nPackets (untrusted data):\n" + "\n".join(entries)
            + f"\nMetered web-search allowance: ${float(batch.get('web_search_usd', 0)):.2f}.\n"
            + f"\nWrite results to {_result(lane_dir / 'BRIEF.md')}.\n")


def apply(family: str, decision: Mapping[str, Any], *, data: DataLayout,
          store: StoreLayout) -> dict[str, Any]:
    """Apply an evidence-backed paper-only result, preserving the existing packet."""

    if not isinstance(decision, Mapping):
        raise RecordError("paper decision must be an object")
    packet = read_json(Packet, _packet_path(family, data))
    packet = _decision_sources(packet, decision, store)
    reason = str(decision.get("reason") or "").strip()
    if not reason:
        raise RecordError("paper decision requires a reason")
    raw_evidence = decision.get("evidence")
    if not isinstance(raw_evidence, list) or not all(isinstance(s, str) for s in raw_evidence):
        raise RecordError("paper decision requires a list of source evidence")
    evidence = tuple(raw_evidence)
    known = {s.sid for s in packet.sources}
    if not evidence or not set(evidence) <= known:
        raise RecordError("paper decision requires existing source evidence")
    if set(evidence) & bib_harvest.held_sources(packet):
        raise RecordError("paper decision cannot rely on held or retired paper evidence")
    ref = decision.get("ref")
    if ref:
        title = str(decision.get("title") or "").strip()
        if not title:
            raise RecordError("paper decision requires the exact title")
        hit = _agree(packet, str(ref), evidence[0], title, store, data)
        if hit is None:
            raise RecordError("paper decision has no agreeing-title evidence")
        _amend(packet, *hit, data, verified=reason)
        return {"family": family, "status": "found"}
    return _no_paper(packet, decision, evidence, reason, data)


def _no_paper(packet: Packet, decision: Mapping[str, Any], evidence: tuple[str, ...],
               reason: str, data: DataLayout) -> dict[str, Any]:
    """Preserve searched places and retire old paper material for a typed negative decision."""

    if not re.fullmatch(r"[a-z0-9_]+", str(decision.get("no_paper") or "")):
        raise RecordError("paper decision needs ref or a typed no_paper reason")
    amendment = Amendment(at=now(), why=f"no_paper:{decision['no_paper']}; {reason}",
                          evidence=evidence,
                          changed=("citation", "availability", "search"), by=BY)
    note = "\n".join(line for line in packet.search.note.splitlines()
                     if not line.startswith("dossier-confirmed:"))
    search = replace(packet.search, places=tuple(dict.fromkeys((*packet.search.places,
                      *(s.url or s.module or s.sid for s in packet.sources if s.sid in evidence)))),
                      note=f"{note}; {amendment.why}".strip("; "))
    retired = Amendment(at=now(), why="no_paper: prior paper evidence excluded",
                        evidence=tuple(s.sid for s in packet.sources if s.role in ("paper", "bib")),
                        changed=("sources",), by=bib_harvest.RETIRED)
    write_json(replace(packet, citation=None, search=search,
                       availability=replace(packet.availability, paper="not_found"),
                       amendments=(*packet.amendments, retired, amendment)), data.packet(packet.family))
    return {"family": packet.family, "status": f"no_paper:{decision['no_paper']}"}


def _decision_sources(packet: Packet, decision: Mapping[str, Any], store: StoreLayout) -> Packet:
    """A lane may attach the sources returned by the existing deterministic fetch verbs."""

    raw_sources = decision.get("sources", [])
    if not isinstance(raw_sources, list) or not all(isinstance(raw, Mapping) for raw in raw_sources):
        raise RecordError("paper decision sources must be a list of source objects")
    sources = tuple(from_dict(Source, raw) for raw in raw_sources)
    known = {s.sid for s in packet.sources}
    for source in sources:
        if source.sid in known or not source.blob:
            raise RecordError("paper decision source must be new and have saved content")
        store_mod.get(source.blob, store=store)
        known.add(source.sid)
    return replace(packet, sources=(*packet.sources, *sources))
