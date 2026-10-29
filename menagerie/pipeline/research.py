"""Research: rung 0's paper completion and rung 1's 2a verbs (family seeds, packet write, lanes).

Rung 1 (the research lane) adds ``seed_family``, ``packet_write``, ``launch`` and
``second_search``; each is documented where it is defined. Rung 0's part follows, owned by
integration until the 2a agents existed. For each family in ``RUNG0_PAPERS``
(the papers the libraries' docstrings do not link) or with a paper or license source that has
no blob yet, ``complete`` fetches the document into the store through ``fetch``, fills the
source's ``blob`` and ``text``, takes the bibliographic record from OpenAlex (the most-cited
work with the paper's exact title, so a published version wins over its arXiv copy), and writes
the packet back with a visible amendment, plus one dated citation-count line. Nothing here
guesses: a ref that fetches nothing, or a title OpenAlex does not hold exactly, is reported and
left out.
"""

from __future__ import annotations

import html.parser
import json
import os
import re
import shlex
import tempfile
import time
import urllib.parse
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    append_jsonl,
    fetch,
    month_of,
    now,
    read_json,
    roster,
    write_json,
)
from menagerie.pipeline import citecheck
from menagerie.pipeline.records import (
    Amendment,
    Citation,
    CitationLine,
    Packet,
    RecordError,
    RosterRow,
    Source,
)

BY = "integration@rung0"
# Family: (ref, title when the ref is a plain URL, why it is this paper).
RUNG0_PAPERS: dict[str, tuple[str, str | None, str]] = {
    "f-resnet": ("1512.03385", None, "ResNet, the paper timm's resnet module implements"),
    "f-coatnext": ("2106.04803", None,
                   "CoAtNet; timm's coatnext is CoAtNet with ConvNeXt blocks, unpublished"),
    "f-botnet": ("2101.11605", None, "Bottleneck Transformers, timm's byoanet botnet"),
    "f-convnextv2": ("2301.00808", None, "ConvNeXt V2"),
    "f-bert": ("1810.04805", None, "BERT"),
    "f-gpt": ("https://cdn.openai.com/better-language-models/"
              "language_models_are_unsupervised_multitask_learners.pdf",
              "Language Models are Unsupervised Multitask Learners",
              "GPT-2's technical report (no arXiv version exists)"),
    "f-fpn-segmentation": ("1612.03144", None, "Feature Pyramid Networks, smp's FPN decoder"),
    "f-full-unrestricted-boltzmann-machine": (
        "https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf",
        "A Learning Algorithm for Boltzmann Machines",
        "Ackley, Hinton and Sejnowski 1985 (doi 10.1207/s15516709cog0901_7, whose landing page "
        "refuses scripts), from Hinton's own page"),
    "f-lenet5-classic": ("http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf",
                         "Gradient-Based Learning Applied to Document Recognition",
                         "LeNet-5's paper, as the classic's packet links it"),
    "f-original-lstm-1997-no-forget-gate": (
        "10.1162/neco.1997.9.8.1735", None, "Hochreiter and Schmidhuber 1997"),
}


def _get_json(url: str) -> dict[str, Any] | None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc.json"
        try:
            fetch._download(url, path)  # noqa: SLF001 -- the fetch lane's one HTTP path
        except fetch.FetchError:
            return None
        return json.loads(path.read_text(encoding="utf-8"))


_norm = citecheck.norm_title


def openalex(ref: str, title: str | None = None) -> list[dict[str, Any]]:
    """Return the OpenAlex works for a paper: the most-cited work with exactly its title first,
    then the work its arXiv id or DOI names when that is another one (often the preprint)."""

    arxiv, doi = fetch._arxiv_id(ref), fetch._doi(ref)  # noqa: SLF001
    direct = None
    if doi or arxiv:
        key = doi or f"10.48550/arXiv.{arxiv}"
        direct = _get_json(f"https://api.openalex.org/works/doi:{urllib.parse.quote(key, safe='/')}")
        title = title or (direct or {}).get("title")
    works = [direct] if direct else []
    if title:
        query = urllib.parse.urlencode({"filter": f"title.search:{title}",
                                        "sort": "cited_by_count:desc", "per-page": "10"})
        found = _get_json(f"https://api.openalex.org/works?{query}") or {}
        exact = [w for w in found.get("results", [])
                 if _norm(w.get("title") or "") == _norm(title)]
        works = exact[:1] + [w for w in works if not exact or w["id"] != exact[0]["id"]]
    return works


def citation_of(works: Sequence[dict[str, Any]], ref: str, sid: str) -> Citation:
    """A packet citation from OpenAlex works: the most-cited one's record and venue, and the
    earliest public date of any of them (an arXiv preprint usually precedes the venue)."""

    best = works[0]
    authors = tuple(a["author"]["display_name"] for a in best.get("authorships", [])
                    if a.get("author", {}).get("display_name"))
    dates = sorted(w["publication_date"] for w in works if w.get("publication_date"))
    return Citation(
        title=best["title"], authors=authors, first_public=dates[0] if dates else None,
        basis="openalex", published=best.get("publication_date"),
        venue=((best.get("primary_location") or {}).get("source") or {}).get("display_name"),
        arxiv=fetch._arxiv_id(ref), doi=fetch._doi(ref),  # noqa: SLF001
        openalex=str(best["id"]).rsplit("/", 1)[-1], from_=sid,
    )


class _Meta(html.parser.HTMLParser):
    """Collect the Highwire ``citation_*`` meta tags an abstract or landing page carries."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        named = dict(attrs)
        name, content = named.get("name") or "", named.get("content")
        if tag == "meta" and name.startswith("citation_") and content:
            self.tags.setdefault(name, []).append(content.strip())


def page_citation(paper: Source, title: str | None, ref: str, *,
                  store: StoreLayout) -> Citation | None:
    """A citation read from the stored document when no bibliographic API answers.

    The title comes from the page's ``citation_title`` tag, else from ``RUNG0_PAPERS``; either
    way it must occur in the document's extracted text, or nothing is cited.
    """

    raw = store.blob(paper.blob).read_bytes() if paper.blob else b""
    meta = _Meta()
    if raw[:4] != b"%PDF":
        meta.feed(raw.decode("utf-8", errors="replace"))
    tags = meta.tags
    title = (tags.get("citation_title") or [title])[0]
    text = store.blob(paper.text).read_text(encoding="utf-8", errors="replace") if (
        paper.text and store.blob(paper.text).is_file()) else ""
    if not title or _norm(title) not in _norm(text):
        return None
    date = (tags.get("citation_date") or tags.get("citation_online_date") or [None])[0]
    authors = tuple(", ".join(reversed(a.split(", ", 1))) if ", " in a else a
                    for a in tags.get("citation_author", []))
    return Citation(title=title, authors=authors,
                    first_public=date.replace("/", "-") if date else None,
                    basis="page metadata" if tags else "document text",
                    arxiv=fetch._arxiv_id(ref), doi=fetch._doi(ref),  # noqa: SLF001
                    from_=paper.sid)


def _next_sid(sources: Sequence[Source]) -> str:
    return f"s{max((int(s.sid[1:]) for s in sources if s.sid[1:].isdigit()), default=0) + 1}"


def _paper(packet: Packet, ref: str, *, store: StoreLayout, data: DataLayout,
           report: dict[str, Any]) -> tuple[list[Source], list[str]]:
    sources, changed = list(packet.sources), []
    paper = next((s for s in sources if s.role == "paper"), None)
    if paper is not None and paper.blob:
        return sources, changed
    url = paper.url if paper is not None else ref
    try:
        lines = fetch.fetch_paper(url, store=store, data=data)
    except (fetch.FetchError, ValueError) as exc:
        report["paper"] = f"not fetched: {exc}"
        return sources, changed
    text = next((line.sha256 for line in lines if line.derived_from == lines[0].sha256), None)
    filled = Source(sid=paper.sid if paper else _next_sid(sources), role="paper",
                    url=lines[0].url or url, blob=lines[0].sha256, text=text,
                    official=True, fetched=now())
    sources = [filled if s.sid == filled.sid else s for s in sources]
    if paper is None:
        sources.append(filled)
    report["paper"] = f"{filled.sid} {len(lines)} documents"
    return sources, changed + [f"sources.{filled.sid}"]


def _licenses(sources: list[Source], *, store: StoreLayout, data: DataLayout,
              report: dict[str, Any]) -> tuple[list[Source], list[str]]:
    out, changed = [], []
    for source in sources:
        if source.role == "license" and source.url and not source.blob:
            try:
                line = fetch.fetch_license(source.url, store=store, data=data)
                source = replace(source, blob=line.sha256, fetched=now())
                changed.append(f"sources.{source.sid}")
            except (fetch.FetchError, ValueError) as exc:
                report.setdefault("license_errors", []).append(f"{source.url}: {exc}")
        out.append(source)
    return out, changed


def _cite(family: str, citation: Citation | None, paper: Source, ref: str, title: str | None, *,
          store: StoreLayout, report: dict[str, Any]) -> tuple[Citation | None, CitationLine | None]:
    """OpenAlex's record and count when it answers, else the document's own metadata."""

    title = title or (citation.title if citation else None)
    works = openalex(ref, title)
    if works:
        counts = CitationLine(family=family, at=now(), openalex=int(works[0]["cited_by_count"]),
                              from_="openalex")
        return citation_of(works, ref, paper.sid), counts
    if citation and citation.from_:
        return citation, None
    found = page_citation(paper, title, ref, store=store)
    report["citation"] = (f"from the document ({found.basis}); no citation count" if found
                          else "no API record and no title found in the document")
    return found, None


def _amended(packet: Packet, sources: list[Source], citation: Citation | None,
             changed: list[str], why: str) -> Packet:
    sids = tuple(dict.fromkeys(c.split(".", 1)[1] for c in changed if c.startswith("sources.")))
    paper = next((s.sid for s in sources if s.role == "paper"), None)
    amendment = Amendment(at=now(), why=f"rung 0 integration: {why}; fetched into the store",
                          evidence=sids or ((paper,) if paper else ()), changed=tuple(changed),
                          by=BY)
    availability = replace(packet.availability,
                           paper="found" if paper else packet.availability.paper)
    return replace(packet, sources=tuple(sources), citation=citation, availability=availability,
                   amendments=packet.amendments + (amendment,))


def complete(family: str, *, store: StoreLayout, data: DataLayout) -> dict[str, Any]:
    """Fetch a family's paper and licenses, cite it, amend the packet; return a report."""

    path = data.packet(family)
    packet = read_json(Packet, path)
    ref, title, why = RUNG0_PAPERS.get(family, (None, None, "the library's own paper link"))
    report: dict[str, Any] = {"family": family}
    sources, changed = list(packet.sources), []
    if ref is not None or any(s.role == "paper" for s in sources):
        sources, changed = _paper(packet, ref or "", store=store, data=data, report=report)
    sources, licensed = _licenses(sources, store=store, data=data, report=report)
    paper = next((s for s in sources if s.role == "paper" and s.blob), None)
    citation, counts = packet.citation, None
    if paper is not None and not (citation and citation.openalex):
        citation, counts = _cite(family, citation, paper, ref or paper.url or "", title,
                                 store=store, report=report)
    changed += licensed
    if citation != packet.citation:
        changed.append("citation")
    if changed:
        write_json(_amended(packet, sources, citation, changed, why), path)
    if counts is not None:
        append_jsonl([counts], data.cite(month_of(counts.at)))
    return {**report, "changed": changed, "citations": counts.openalex if counts else None,
            "title": citation.title if citation else None}


# --- Rung 1: family seeds ---------------------------------------------------------------------
# ``seed_family`` proposes, the 2a agent decides: a candidate is any roster row that shares a
# normalized name, an alias or acronym, a name stem, a repository, an S-C candidate or a paper id
# with an input row. Stems drop size and dataset words and checkpoint codes, so
# ``dino_r50_4scale`` and ``dino_swin_l_4scale`` both stem from ``dino``.

VARIANT_WORDS = frozenset({
    "tiny", "small", "base", "large", "huge", "giant", "mini", "nano", "micro", "medium", "xl",
    "xxl", "xs", "official", "unofficial", "pretrained", "pytorch", "torch", "tf", "keras",
    "cifar", "cifar10", "cifar100", "imagenet", "in1k", "in22k", "in21k", "coco", "mnist", "voc",
    "ade20k", "cityscapes", "ft", "finetuned", "l", "s", "m", "b", "t", "x", "h", "g",
})
VARIANT_TOKEN = re.compile(r"\d{3,}|\d+[a-z]+\d*[a-z]*|[a-z]\d+[a-z]?|patch\d+|\d+x\d+[a-z]*")
CANDIDATE_CAP = 30
CLASSIC_ZOOS = frozenset({"historical-classics", "unregistered-classics-pytorch",
                          "classics-pytorch", "base-pytorch-local"})


def _bare(name: str) -> str:
    """A name without its parenthetical notes: ``KVAE (Kalman ...)`` -> ``KVAE``."""

    return re.sub(r"\s*\([^)]*\)", "", name).strip() or name


def _stem(name: str) -> tuple[str, ...]:
    """Name tokens with variant words and checkpoint codes dropped after the first token."""

    raw = [t for t in re.split(r"[\W_]+", _bare(name).lower()) if t]
    tokens: list[str] = []
    for token in raw:
        if tokens and re.fullmatch(r"v?\d{1,2}", token):
            tokens[-1] += token  # BEiT-3 -> beit3, YOLO v5 -> yolov5
        elif not tokens or not (token in VARIANT_WORDS or VARIANT_TOKEN.fullmatch(token)):
            tokens.append(token)
    if tokens and len(tokens[-1]) > 4 and tokens[-1].endswith("s") and not tokens[-1].endswith("ss"):
        tokens[-1] = tokens[-1][:-1]
    return tuple(tokens)


def _acronym(name: str) -> str | None:
    """Initials of a spaced or CamelCase name (``NetworkInNetwork`` -> ``nin``); identifiers with
    underscores (checkpoint names) have none."""

    if "_" in name:
        return None
    words = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", _bare(name))
    words = [w for w in words if w.lower() not in VARIANT_WORDS and not w.isdigit()]
    acronym = "".join(w[0] for w in words).lower()
    return acronym if len(words) >= 2 and len(acronym) >= 3 and all(len(w) >= 2 for w in words) \
        else None


def _aliases(row: RosterRow) -> set[str]:
    """Normalized short names: the row's aliases and a one-word parenthetical."""

    notes = [n.strip() for n in re.findall(r"\(([^)]*)\)", row.name)]
    names = list(row.aliases) + [n for n in notes if n and " " not in n and len(n) <= 20]
    return {roster.norm(n) for n in names if len(roster.norm(n)) >= 3}


def _repos(row: RosterRow) -> set[str]:
    out = set()
    for url in (row.source_url, *row.hint_urls):
        match = re.search(r"(github\.com|gitee\.com|gitlab\.com|huggingface\.co)/([^/\s#?]+)/"
                          r"([^/\s#?]+)", url or "", re.IGNORECASE)
        if match:
            out.add("/".join(match.groups()).lower().removesuffix(".git"))
    return out


def _papers(row: RosterRow) -> set[str]:
    out = set()
    for ref in (row.source_url, row.hint_paper, *row.hint_urls):
        for piece in re.split(r"[\s;,]+", ref or ""):
            arxiv, doi = fetch._arxiv_id(piece), fetch._doi(piece)  # noqa: SLF001
            if arxiv or doi:
                out.add(f"arXiv:{arxiv}" if arxiv else f"doi:{(doi or '').lower()}")
    return out


def _keys(row: RosterRow) -> dict[str, set[str]]:
    return {"same normalized name": {roster.norm(_bare(row.name)), roster.norm(row.name)},
            "alias": _aliases(row), "acronym": {a for a in [_acronym(row.name)] if a},
            "same repository": _repos(row),
            "same S-C candidate": {f for f in row.found_by if f.startswith("S-C:")},
            "same paper id": _papers(row), "stem": {_stem(row.name)[0]} if _stem(row.name) else set()}


def _packet_of(data: DataLayout) -> dict[str, str]:
    """Which family packet already lists each stable id (packets parsed through ``drive``'s
    per-file memo, so seeding many families reads each packet once). The memo is shared by
    path, so a packet is read as the ``Packet`` record every other memo reader expects."""

    from menagerie.pipeline import drive

    memo = drive._memo(data)
    out = {}
    for path in sorted((data.root / "research").glob("*/f-*.json")):
        packet = memo.get(path, lambda p: read_json(Packet, p))
        out.update({m.id: packet.family for m in packet.members})
    return out


def _match(row: RosterRow, other: RosterRow, keys: dict[str, set[str]],
           other_keys: dict[str, set[str]]) -> list[str]:
    why = [k for k in ("same normalized name", "same repository", "same S-C candidate",
                       "same paper id") if keys[k] & other_keys[k]]
    my_short = keys["alias"] | keys["acronym"]
    their_short = other_keys["alias"] | other_keys["acronym"]
    if (my_short & other_keys["same normalized name"]) or (
            their_short & keys["same normalized name"]) \
            or (keys["alias"] & other_keys["alias"]):  # two equal acronyms are no evidence
        why.append("alias")
    else:
        inside = [a for a in keys["alias"] if len(a) >= 4
                  and any(a in n for n in other_keys["same normalized name"])]
        why += [f"alias {inside[0]!r} inside the name"] if inside else []
    mine, theirs = _stem(row.name), _stem(other.name)
    short = min(len(mine), len(theirs))
    if short and mine[:short] == theirs[:short] and len("".join(mine[:short])) >= 3:
        why.append(f"name stem {' '.join(mine[:short])!r}")
    return why


NAME_KINDS = ("same normalized name", "alias", "acronym")


def _index(keyed: Sequence[dict[str, set[str]]]) -> dict[tuple[str, str], set[int]]:
    index: dict[tuple[str, str], set[int]] = {}
    for n, keys in enumerate(keyed):
        for kind, values in keys.items():
            for value in values:
                index.setdefault((kind, value), set()).add(n)
    return index


def _near(keys: dict[str, set[str]], index: dict[tuple[str, str], set[int]],
          keyed: Sequence[dict[str, set[str]]]) -> set[int]:
    """Rows sharing any key with ``keys``, or holding one of its aliases in their name: the
    shortlist ``_match`` then scores."""

    names = set().union(*(keys[kind] for kind in NAME_KINDS))
    lookups = [(kind, v) for kind, values in keys.items() if kind not in NAME_KINDS
               for v in values] + [(kind, v) for v in names for kind in NAME_KINDS]
    near = set().union(*(index.get(key, set()) for key in lookups))
    return near | _alias_inside(keys["alias"], keyed)


def _alias_inside(aliases: set[str], keyed: Sequence[dict[str, set[str]]]) -> set[int]:
    """Rows whose name contains an alias of four or more letters (``kvae`` in ``autoencoderklkvae``)."""

    long = [a for a in aliases if len(a) >= 4]
    if not long:
        return set()
    return {n for n, other in enumerate(keyed)
            if any(a in name for a in long for name in other["same normalized name"])}


# The last roster's ``_keys`` and their ``_index``: a 2a batch seeds every family twice (brief,
# launch), and keying the whole roster takes seconds; reused while the rows are equal.
_KEYED: dict[str, Any] = {}


def _roster_keys(rows: Sequence[RosterRow]) -> tuple[list[dict[str, set[str]]],
                                                     dict[tuple[str, str], set[int]]]:
    if _KEYED.get("rows") != list(rows):
        keyed = [_keys(row) for row in rows]
        _KEYED.update(rows=list(rows), keyed=keyed, index=_index(keyed))
    return _KEYED["keyed"], _KEYED["index"]


def _seed_matches(inputs: Sequence[RosterRow], rows: Sequence[RosterRow],
                  ids: Sequence[str]) -> dict[int, list[str]]:
    """Each candidate row's reasons, merged over the input rows."""

    keyed, index = _roster_keys(rows)
    found: dict[int, list[str]] = {}
    for row in inputs:
        keys = _keys(row)
        for n in _near(keys, index, keyed):
            why = [] if rows[n].stable_id in ids else _match(row, rows[n], keys, keyed[n])
            if why:
                found[n] = found.get(n, []) + [w for w in why if w not in found.get(n, [])]
    return found


def _row_view(row: RosterRow) -> dict[str, Any]:
    return {"id": row.stable_id, "name": row.name, "zoo": row.zoo, "variant": row.variant,
            "hint_urls": list(row.hint_urls), "hint_paper": row.hint_paper,
            "hint_framework": row.hint_framework, "found_by": list(row.found_by),
            "state": row.state}


def seed_family(ids: Sequence[str], *, data: DataLayout) -> dict[str, Any]:
    """Propose a family for roster rows from deterministic seeds, without writing anything.

    Inputs: stable ids. Returns: ``{"family": <proposed id>, "existing": <family of a packet that
    already lists an input, or null>, "members": [ids], "rows": [input rows with their hints],
    "candidates": [{"id", "name", "zoo", "variant", "state", "packet", "why"}], "more": <count
    not shown>}``, where a candidate shares a normalized name, an alias or acronym, a name stem,
    a repository, an S-C candidate or a paper id with an input row (duplicates, June stand-ins,
    name collisions). The agent accepts or rejects each in the packet; ``id`` is null for a row
    with no stable id yet (``packet write`` mints one if the row becomes a member). Over 30
    matches, the 30 with the most reasons are listed. Side effects: none.
    """

    rows = roster.read(data=data)
    by_id = {r.stable_id: r for r in rows if r.stable_id}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise RecordError(f"no roster row has stable id {missing}")
    inputs = [by_id[i] for i in ids]
    found = _seed_matches(inputs, rows, ids)
    packets = _packet_of(data)
    ranked = sorted(found.items(), key=lambda item: (-len(item[1]), rows[item[0]].name))
    candidates = [{"id": rows[n].stable_id, "name": rows[n].name, "zoo": rows[n].zoo,
                   "variant": rows[n].variant, "state": rows[n].state,
                   "packet": packets.get(rows[n].stable_id or ""), "why": why}
                  for n, why in ranked[:CANDIDATE_CAP]]
    existing = next((packets[i] for i in ids if i in packets), None)
    stem = _stem(inputs[0].name) or (roster.norm(inputs[0].name),)
    proposed = existing or "f-" + re.sub(r"[^a-z0-9._-]+", "-", "-".join(stem)).strip("-._")
    return {"family": proposed, "existing": existing, "members": list(ids),
            "rows": [_row_view(r) for r in inputs], "candidates": candidates,
            "more": max(0, len(ranked) - CANDIDATE_CAP)}


def packet_write(family: str, path: Path, *, store: StoreLayout, data: DataLayout) -> Packet:
    """Check a draft packet and write it (``packet.packet_write``, split out at rung 2)."""

    from menagerie.pipeline import packet

    return packet.packet_write(family, path, store=store, data=data)


# --- Rung 1: 2a lanes and the second code search ------------------------------------------------

PROMPT = Path(__file__).resolve().parent / "prompts" / "2a.md"
BUDGET = {"fetches": 12, "tool_calls": 25, "minutes": 15}
SECOND_SEARCH_ROLE = "model-implementation"
SECOND_SEARCH_WAIT_S = 300
REPO = Path(__file__).resolve().parents[2]


def _result(brief_path: Path) -> Path:
    """The file a lane must write: ``drive.launch`` expects ``<brief stem>.result.md``."""

    return brief_path.with_name(f"{brief_path.stem}.result.md")


def _pipeline_command(data: DataLayout, drafts: Path) -> str:
    """The shell prefix a lane runs verbs with: this checkout, its data, store and drafts."""

    env = {"MENAGERIE_DATA": str(data.root), "MENAGERIE_DRAFTS": str(drafts)}
    env |= {k: os.environ[k] for k in ("MENAGERIE_STORE", "MENAGERIE_STORE_REMOTE",
                                     "MENAGERIE_OPENALEX", "MENAGERIE_OPENALEX_COMMAND")
            if os.environ.get(k)}
    pairs = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"cd {shlex.quote(str(REPO))} && env {pairs} python3 -m menagerie.pipeline"


def _sc_notes(row_ids: Sequence[str], seed: dict[str, Any], data: DataLayout) -> list[str]:
    cands = {f.split(":", 1)[1] for r in seed["rows"] for f in r["found_by"] if f.startswith("S-C:")}
    plan = data.root / "sweeps" / "S-C" / "IMPL_PLAN.tsv"
    if not cands or not plan.exists():
        return []
    with plan.open(encoding="utf-8") as handle:
        return [" | ".join(c for c in line.rstrip("\n").split("\t") if c)
                for line in handle if line.split("\t", 1)[0] in cands]


def _open_gaps(row_ids: Sequence[str], data: DataLayout) -> list[str]:
    """Each row's ``research_gap`` give-up, for the brief: the amendment must name its class.

    ``drive._gap_step`` counts only an amendment whose ``why`` holds the give-up's class as the
    answer that earns the 2b retry; without it the row would come back to 2a after the write.
    """

    from menagerie.pipeline.records import ModelRecord  # noqa: PLC0415

    lines = []
    for model_id in row_ids:
        path = data.model(model_id)
        give = read_json(ModelRecord, path).give_up if path.exists() else None
        if give is not None and give.cls.split(":", 1)[0] == "research_gap":
            lines.append(f"- {model_id} `{give.cls}`: {(give.missing or give.reason)[:300]}")
    return lines


# The metered search as a lane must run it: the broker records the lane's run id itself, and the
# purpose names the family, so spend splits by lane and family (fix 14).
WEB_SEARCH = "web-search --caller menagerie-2a --purpose <family>"


def brief(batch: Mapping[str, Any], *, data: DataLayout, lane_dir: Path) -> str:
    """The 2a brief for one batch: the prompt, the verbs, and each family's seeds and hints."""

    if batch.get("paper_only"):
        from menagerie.pipeline.find_paper import brief as paper_brief  # noqa: PLC0415

        return paper_brief(batch, data=data, lane_dir=lane_dir)
    command = _pipeline_command(data, lane_dir / "drafts")
    from menagerie.pipeline import dossier  # noqa: PLC0415

    prompt, dossier_section = dossier.template(PROMPT.read_text(encoding="utf-8"))
    parts = [prompt.replace("{PIPE}", command),
             f"\n## This lane\n\nLabel `{batch['label']}`. Budget per family: "
             f"{BUDGET['fetches']} fetches, {BUDGET['tool_calls']} tool calls, "
             f"{BUDGET['minutes']} minutes (a stop, not a failure). Metered `web-search` spend "
             f"allowed for the whole lane: ${float(batch.get('web_search_usd', 0)):.2f}, always run "
             f"as `{WEB_SEARCH} \"<query>\"` so the meter records this lane and family. Drafts go "
             f"in `{lane_dir / 'drafts'}/<family>.json`. When every family is done or stopped, "
             f"write `{_result(lane_dir / 'BRIEF.md')}`: one line per family (packet written or "
             "the last refusal, minutes, fetches, tool calls, web-search spend).\n"]
    from menagerie.pipeline.records import Dossier  # noqa: PLC0415
    from menagerie.pipeline import read_jsonl  # noqa: PLC0415

    enabled = batch.get("dossiers", False)
    dossiers = read_jsonl(Dossier, dossier.path(data)) if enabled and dossier.path(data).exists() else []
    proposals = []
    for item in batch["families"]:
        seed = seed_family(item["ids"], data=data)
        family = item.get("family") or seed["family"]
        lines = [json.dumps(r, ensure_ascii=False) for r in seed["rows"]]
        cands = [json.dumps(c, ensure_ascii=False) for c in seed["candidates"]]
        parts.append(f"\n## Family `{family}`\n\nRows (members to confirm):\n```\n"
                     + "\n".join(lines) + "\n```\nCandidates from `family seed` (accept or reject"
                     f"; {seed['more']} more not shown; existing packet: {seed['existing']}):\n"
                     "```\n" + "\n".join(cands or ["none"]) + "\n```\n")
        proposals.append(dossier.brief(item["ids"], data=data, enabled=enabled, records=dossiers,
                                   section=dossier_section))
        from menagerie.pipeline.resolve_confirm import hint_block

        parts.append(hint_block(item["ids"], data=data))
        if item.get("hints"):
            parts.append(f"Hints (untrusted): {item['hints']}\n")
        gaps = _open_gaps(item["ids"], data)
        if gaps:
            parts.append("Open 2b give-ups to answer (a 2b lane found the saved sources lacking; "
                         "save what is missing, and the packet's new amendment's `why` names each "
                         "class exactly, as in `research_gap:code: saved the repository`):\n"
                         + "\n".join(gaps) + "\n")
        notes = _sc_notes(item["ids"], seed, data)
        if notes:
            parts.append("S-C notes (June discovery plan, untrusted):\n" +
                         "\n".join(f"- {n}" for n in notes) + "\n")
    return "".join(proposals + parts)


def launch(batch_file: Path, *, data: DataLayout, role: str | None = None) -> list[str]:
    """Start one 2a lane per family batch listed in ``batch_file``, each on the first admitted lab.

    Inputs: a JSON file ``{"parent": <run id or null>, "batches": [{"label", "role" (default
    model-default), "web_search_usd" (default 0), "tier" (optional), "families": [{"ids":
    [stable ids], "family" (optional; else the seed's proposal), "hints" (optional text),
    "tier" (optional; else the batch's)}]}]}``; ``role``, when given, is tried first for every
    batch. 2a is lab-agnostic: a refused role hands the batch to the other main lab, and a batch
    neither admits is queued for ``requeue`` (BUILD-PLAN r2 section 5). Each batch's brief is
    written to ``briefs/<label>/BRIEF.md`` under the work root (``~/scratch/menagerie``) and the
    lane writes ``BRIEF.result.md`` beside it. Returns: the lane run ids (``queued:<label>`` for
    a queued batch). Side effects: briefs, lanes, their leases, their launch events and one
    ``tier`` event per family with a tier (the plan's case, read by ``measures``).
    """

    from menagerie.pipeline import admission, drive

    spec = json.loads(Path(batch_file).read_text(encoding="utf-8"))
    runs = []
    for batch in spec["batches"]:
        lane_dir = work_dir("briefs") / batch["label"]
        (lane_dir / "drafts").mkdir(parents=True, exist_ok=True)
        path = lane_dir / "BRIEF.md"
        path.write_text(brief(batch, data=data, lane_dir=lane_dir), encoding="utf-8")
        keys = [f.get("family") or seed_family(f["ids"], data=data)["family"]
                for f in batch["families"]]
        first = role or batch.get("role", "model-default")
        roles = [first, *filter(None, [admission.other_role(first)])]
        runs.append(drive.launch_first("2a", roles, path, keys=keys, parent=spec.get("parent"),
                                       label=batch["label"], web=True,
                                       on_start=_tier_events(keys, batch)))
    return runs


def _tier_events(families: Sequence[str], batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One ``tier`` event per family with a tier (its own, else the batch's), for ``measures``."""

    out = []
    for family, item in zip(families, batch["families"], strict=True):
        tier = item.get("tier") or batch.get("tier")
        if tier:
            out.append({"family": family, "verb": "tier", "tier": tier})
    return out


def _search_subject(family: str, data: DataLayout) -> dict[str, Any]:
    """What the second search looks for: the packet if written, else the lane's draft."""

    path = data.packet(family)
    if not path.exists():
        drafts = os.environ.get("MENAGERIE_DRAFTS")
        path = Path(drafts) / f"{family}.json" if drafts else path
    if not path.exists():
        raise FileNotFoundError(f"no packet or draft for {family} (drafts: $MENAGERIE_DRAFTS)")
    raw = json.loads(path.read_text(encoding="utf-8"))
    cite = raw.get("citation") or {}
    return {"name": (raw.get("identity") or {}).get("canonical") or family,
            "aliases": (raw.get("identity") or {}).get("aliases") or [],
            "title": cite.get("title"), "arxiv": cite.get("arxiv"), "doi": cite.get("doi"),
            "authors": (cite.get("authors") or [])[:3],
            "searched": (raw.get("search") or {}).get("places") or []}


def _answer(path: Path) -> dict[str, Any]:
    """The second search's JSON answer, tolerating a fence or prose around the object."""

    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def second_search(family: str, *, data: DataLayout) -> dict[str, Any]:
    """Run the one-minute second code search on the other lab before ``sufficient`` with no code.

    Starts a short web-enabled lane by role ``model-implementation`` (the other lab from the 2a
    lanes' ``model-default``) with the family's name, aliases and citation from its packet or
    draft, and waits up to five minutes for its answer. Returns: ``{"family", "role", "run",
    "found": [urls], "queries": [...], "status": "done|timeout"}``; the 2a agent adds each found
    repository as a source (or an amendment to a written packet) and names ``second-search`` in
    ``search.places``. Side effects: one lane and a ``second_search`` drive event.
    """

    from menagerie.pipeline import drive

    subject = _search_subject(family, data)
    label = f"ss-{family[2:34]}-{uuid.uuid4().hex[:8]}"
    lane_dir = work_dir("briefs") / label
    lane_dir.mkdir(parents=True, exist_ok=True)
    path = lane_dir / "BRIEF.md"
    answer = _result(path)
    path.write_text(
        "# Second code search (one minute)\n\nFind any public implementation, in any framework, "
        f"of this model:\n```json\n{json.dumps(subject, indent=1)}\n```\nSearch GitHub (code and "
        "repository search), the Hugging Face papers page for its arXiv id, Papers with Code, "
        "the authors' pages and one open-web search. Spend at most one minute; do not clone or "
        f"download anything. Then write `{answer}` as JSON: "
        '{"found": [repository URLs that implement this model], "queries": [what you searched]} '
        "and stop. An empty `found` is a valid answer.\n", encoding="utf-8")
    run = drive.launch("2a", SECOND_SEARCH_ROLE, path, keys=(), label=label, web=True)
    deadline = time.monotonic() + SECOND_SEARCH_WAIT_S
    while not answer.exists() and time.monotonic() < deadline:
        time.sleep(5)
    got = _answer(answer)
    result = {"family": family, "role": SECOND_SEARCH_ROLE, "run": run,
              "found": list(got.get("found") or []), "queries": list(got.get("queries") or []),
              "status": "done" if answer.exists() else "timeout"}
    drive.event(os.environ.get("FLEET_RUN_ID") or "manual", family, "second_search",
                run=run, found=len(result["found"]), status=result["status"])
    return result
