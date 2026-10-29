"""Resumable source searches for the 2a router. Candidates never settle a row.

Run OpenAlex locally through its broker, then carry its checkpoint to the worker
for the other sources. Each source is serial and paced; a failed query stays
incomplete and is retried on the next invocation. Evidence is content addressed.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import html
import json
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlencode, urlsplit, urlunsplit

import requests

from menagerie.pipeline import DataLayout, fetch, now, roster
from menagerie.pipeline.admission import _write_atomic
from menagerie.pipeline.records import RecordError, RosterRow

SOURCES = ("openalex", "arxiv", "github", "hf")
GAPS = {"openalex": 1.0, "arxiv": 3.1, "github": 6.2, "hf": 1.0}
KINDS = {"openalex": "paper", "arxiv": "paper", "github": "code", "hf": "weights"}
GAPS.update(crossref=1.0, wikipedia=1.0)
KINDS.update(crossref="paper", wikipedia="page")
TOP = 3
ATTEMPTS = 3
MAX_WAIT = 300
MAX_BYTES = 8 * 1024 * 1024
SCHEMA = "resolve/1"


def _save_response(path: Path, body: bytes) -> None:
    """Publish complete bytes, repairing an interrupted or corrupt existing response."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        pending = Path(handle.name)
        try:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)


def _row(row: RosterRow) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(row)))


def folder(data: DataLayout) -> Path:
    return data.root / "resolve" / now()[:7]


def normalized(name: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name.split("(")[0])
    return re.sub(r"[^a-zA-Z0-9]+", " ", spaced).strip().lower()


def url_key(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username:
        raise RecordError("resolver candidate must be a public HTTP URL")
    host = parts.hostname.lower()
    path = parts.path.rstrip("/")
    if host in ("arxiv.org", "www.arxiv.org"):
        host = "arxiv.org"
        path = re.sub(r"v\d+$", "", path.replace("/pdf/", "/abs/").removesuffix(".pdf"))
    if host in ("github.com", "doi.org"):
        path = path.lower().removesuffix(".git")
    return urlunsplit(("https", host, path, parts.query, ""))


def candidate(title: str, url: str, year: Any, text: str, source: str) -> dict[str, Any]:
    url_key(url)
    return {"title": title, "url": url, "year": year, "matching_text": text[:1800],
            "kind": KINDS[source], "via": source, "evidence": []}


def _openalex_work(work: dict[str, Any]) -> dict[str, Any]:
    words = sorted((i, w) for w, ids in (work.get("abstract_inverted_index") or {}).items()
                   for i in ids)
    url = work.get("doi") or (work.get("primary_location") or {}).get("landing_page_url")
    return candidate(work["title"], url or work["id"], work.get("publication_year"),
                     " ".join(w for _, w in words), "openalex")


def _arxiv(body: bytes) -> list[dict[str, Any]]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(body)
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise ValueError("not an Atom feed")
    return [candidate(e.findtext("a:title", "", ns), e.findtext("a:id", "", ns),
                      e.findtext("a:published", "", ns)[:4], e.findtext("a:summary", "", ns), "arxiv")
            for e in root.findall("a:entry", ns)]


def _crossref(record: dict) -> list[dict[str, Any]]:
    return [candidate(w["title"][0], "https://doi.org/" + w["DOI"],
                      w.get("published", {}).get("date-parts"), w.get("abstract", ""), "crossref")
            for w in record["message"]["items"]]


def _wikipedia(record: dict) -> list[dict[str, Any]]:
    page = record["parse"]
    markup = page["text"]["*"]
    # References alone verify the hypothesis; a title merely in the body does not.
    refs = re.findall(r'<ol\b[^>]*class=[\'"][^\'"]*\breferences\b[^\'"]*[\'"][^>]*>(.*?)</ol>',
                      markup, flags=re.DOTALL)
    text = html.unescape(re.sub(r"<[^>]+>", " ", " ".join(refs)))
    return [{**candidate(page["title"], "https://en.wikipedia.org/wiki/" +
                         quote(page["title"].replace(" ", "_")), None, text, "wikipedia"),
             "references": text}]


def _github(record: dict) -> list[dict[str, Any]]:
    if record.get("incomplete_results") or not isinstance(record.get("items"), list):
        raise ValueError("incomplete GitHub search")
    return [candidate(w["full_name"], w["html_url"], w["created_at"][:4],
                      (w.get("description") or "") + " " + " ".join(
                          m["fragment"] for m in w.get("text_matches", [])), "github")
            for w in record["items"]]


def _hf(record: list) -> list[dict[str, Any]]:
    if not isinstance(record, list):
        raise TypeError("not a model list")
    return [candidate(w["id"], "https://huggingface.co/" + w["id"],
                      (w.get("createdAt") or "")[:4],
                      json.dumps(w.get("cardData") or w.get("tags", [])), "hf") for w in record]


def _openalex(record: dict) -> list[dict[str, Any]]:
    if not isinstance(record.get("results"), list):
        raise TypeError("not an OpenAlex works list")
    return [_openalex_work(w) for w in record["results"]]


def parse(source: str, body: bytes) -> list[dict[str, Any]]:
    """Project each API's public schema; malformed envelopes are errors, not no hits."""
    if source == "arxiv":
        return _arxiv(body)
    parsers = {"crossref": _crossref, "wikipedia": _wikipedia, "github": _github,
               "hf": _hf, "openalex": _openalex}
    if source not in parsers:
        raise ValueError("unknown source")
    return parsers[source](json.loads(body))


def top_three(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One combined cap after URL de-duplication, retaining every response sid."""
    unique: dict[str, dict[str, Any]] = {}
    for item in candidates:
        key = url_key(item["url"])
        prior = unique.get(key, {**item, "evidence": []})
        unique[key] = {**prior, "evidence": list(dict.fromkeys(
            [*prior["evidence"], *item.get("evidence", [])]))}
    return list(unique.values())[:TOP]


def _endpoint(source: str, query: str) -> str:
    endpoints = {
        "openalex": ("https://api.openalex.org/works", {"search": query, "per_page": TOP}),
        "arxiv": ("https://export.arxiv.org/api/query", {"search_query": "all:" + query,
                                                      "max_results": TOP}),
        "github": ("https://api.github.com/search/repositories",
                   {"q": query + " in:name,description,readme", "per_page": TOP}),
        "hf": ("https://huggingface.co/api/models",
               {"search": query.strip('"'), "limit": TOP, "full": "true"}),
        "crossref": ("https://api.crossref.org/works", {"query.title": query, "rows": TOP}),
        "wikipedia": ("https://en.wikipedia.org/w/api.php",
                      {"action": "parse", "page": query, "prop": "text", "format": "json"}),
    }
    url, params = endpoints[source]
    return url + "?" + urlencode(params)


def _request(source: str, query: str) -> tuple[int, bytes, dict[str, str]]:
    url = _endpoint(source, query)
    if source == "openalex":
        response = fetch._openalex(url)  # The credential-safe local broker path.
        return response.status, response.body, {}
    with requests.get(url, timeout=25, stream=True, headers={
        "User-Agent": "model-source-router/1.0",
        "Accept": "application/vnd.github.text-match+json" if source == "github" else "*/*",
    }) as response:
        chunks, size = [], 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError("source response exceeds byte cap")
            chunks.append(chunk)
        headers = {k.lower(): v for k, v in response.headers.items()
                   if k.lower() in ("retry-after", "x-ratelimit-reset", "x-ratelimit-remaining")}
        return response.status_code, b"".join(chunks), headers


def _cooldown(source: str, status: int, headers: dict[str, str], attempt: int) -> float:
    reset = max(0, float(headers.get("x-ratelimit-reset", "0")) - time.time() + 1)
    wait = GAPS[source]
    if headers.get("x-ratelimit-remaining") == "0":
        wait = max(wait, reset)
    if status in (403, 429):
        wait = max(wait, 60, float(headers.get("retry-after", "0")), reset)
    if status >= 500:
        wait = max(wait, 10 * 2**attempt)
    return wait


def _query(source: str, query: str, *, root: Path, clock: dict[str, float]) -> dict[str, Any]:
    calls = []
    for attempt in range(ATTEMPTS):
        wait = max(0, clock.get(source, 0) - time.monotonic())
        if wait > MAX_WAIT:
            calls.append({"query": query, "error": "cooldown", "retry_after_seconds": wait})
            break
        time.sleep(wait)
        try:
            status, body, headers = _request(source, query)
            clock[source] = time.monotonic() + _cooldown(source, status, headers, attempt)
            call = {"query": query, "status": status, "parsed": False}
            calls.append(call)
            if status == 200:
                sha = hashlib.sha256(body).hexdigest()
                responses = root / "responses"
                responses.mkdir(exist_ok=True)
                path = responses / sha
                _save_response(path, body)
                found = parse(source, body)
                call.update(parsed=True, sid="s-resolve-" + sha, sha256=sha)
                return {"complete": True, "candidates": [
                    {**c, "evidence": [call["sid"]]} for c in found], "calls": calls}
            if status not in (403, 429) and status < 500:
                break
        except (requests.RequestException, fetch.FetchError, ValueError, TypeError,
                KeyError, AttributeError, ET.ParseError) as exc:
            calls.append({"query": query, "error": type(exc).__name__})
            clock[source] = time.monotonic() + 10 * 2**attempt
    return {"complete": False, "candidates": [], "calls": calls}


def search(source: str, name: str, *, root: Path, clock: dict[str, float]) -> dict[str, Any]:
    names = list(dict.fromkeys([name, normalized(name)]))
    # Broker policy refuses quoted searches; the other sources use phrase queries.
    queries = [n.replace('"', '') if source == "openalex" else '"' + n.replace('"', '') + '"'
               for n in names if n.strip()]
    got = [_query(source, q, root=root, clock=clock) for q in queries]
    return {"complete": bool(got) and all(g["complete"] for g in got),
            "candidates": top_three([c for g in got for c in g["candidates"]]),
            "calls": [c for g in got for c in g["calls"]]}


def _dead_ids(data: DataLayout) -> set[str]:
    path = data.root / "sweeps/census-2026-10/census.jsonl"
    if not path.exists():
        return set()
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"] for r in records if r.get("alive") is False or r.get("archived") is True}


def eligible(row: RosterRow) -> bool:
    """Dry-run name-only rule, excluding inline sources and library config names."""
    if row.source_url or row.hint_urls or row.hint_paper:
        return False
    if row.zoo == "discovered-pytorch":
        return "legacy:c2-disco" in row.found_by and not re.search(r"https?://|arxiv|10\.\d{4,9}/",
                             " ".join((row.name, *row.found_by)), re.IGNORECASE)
    if "legacy:c3-classics" not in row.found_by:
        return False
    return not (re.match(r"^(timm|torchvision|mmpose|mmcls|nemo|keras|smp|openke|dyglib)_",
                         row.name, re.IGNORECASE)
                or len(row.name.split("_")) >= 3)


def _cohort_inputs(rows: Sequence[RosterRow], data: DataLayout) -> list[RosterRow]:
    """Select the complete default cohort; missing identities never filter rows away."""
    dead = _dead_ids(data)
    selected = [r for r in rows if eligible(r) or r.stable_id in dead]
    if any(not r.stable_id for r in selected):
        raise RecordError("resolver input needs stable identities; refusing to silently skip name-only rows")
    return selected


def inputs(ids: Sequence[str] | None, *, data: DataLayout) -> list[RosterRow]:
    rows = [r for r in roster.read(data=data) if r.state == "active"]
    if ids is None:
        return _cohort_inputs(rows, data)
    by_id = {r.stable_id: r for r in rows}
    if any(not isinstance(i, str) or not re.fullmatch(r"m\d+", i) for i in ids):
        raise RecordError("resolver ids must be stable model identifiers")
    unknown = set(ids) - by_id.keys()
    if unknown:
        raise RecordError(f"unknown or inactive resolver ids: {sorted(unknown)}")
    return [by_id[i] for i in dict.fromkeys(ids)]


def _aggregate(rows: Sequence[RosterRow], root: Path) -> list[dict[str, Any]]:
    out = []
    hypotheses = _hypotheses(root)
    for row in rows:
        records = []
        for source in SOURCES:
            path = root / "checkpoints" / f"{row.stable_id}-{source}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record.get("row") == _row(row):
                    records.append(record)
        out.append({"schema": SCHEMA, "id": row.stable_id, "row": _row(row),
                    # Three-source title checks precede broad name hits within the same cap.
                    "candidates": top_three(_introduced(row.stable_id, root, hypotheses=hypotheses) +
                                            [c for r in records for c in r["candidates"]]),
                    "sources": {r["source"]: r["complete"] for r in records},
                    "complete": len(records) == len(SOURCES) and all(r["complete"] for r in records)})
    return out


def _hypotheses(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "introductions.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] \
        if path.exists() else []
    return {r["id"]: r for r in records if r.get("title") and r.get("wikipedia")}


def _intro_records(stable_id: str, root: Path, hypothesis: dict) -> dict:
    """Read the complete three-source checkpoint set for this exact hypothesis."""
    results = {}
    for source in ("openalex", "crossref", "wikipedia"):
        path = root / "checkpoints" / f"{stable_id}-intro-{source}.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        if not record.get("complete") or record.get("hypothesis") != hypothesis:
            return {}
        results[source] = record["candidates"]
    return results


def _introduced(stable_id: str, root: Path, *, hypotheses: dict | None = None) -> list[dict[str, Any]]:
    hypothesis = (hypotheses if hypotheses is not None else _hypotheses(root)).get(stable_id)
    if not hypothesis:
        return []
    results = _intro_records(stable_id, root, hypothesis)
    if not results:
        return []
    title = roster.norm(hypothesis["title"])
    works = {s: [c for c in results[s] if roster.norm(c["title"]) == title]
             for s in ("openalex", "crossref")}
    phrase = re.sub(r"[\W_]+", " ", hypothesis["title"].casefold()).strip()
    wiki = [c for c in results["wikipedia"] if " " + phrase + " " in
            " " + re.sub(r"[\W_]+", " ", c["references"].casefold()).strip() + " "]
    if not works["openalex"] or not works["crossref"] or not wiki:
        return []
    evidence = list(dict.fromkeys(sid for source in (*works.values(), wiki)
                                 for c in source for sid in c["evidence"]))
    return [{**c, "via": "introduced-by", "evidence": evidence} for c in works["openalex"]]


def _intro_source(rows: Sequence[RosterRow], source: str, root: Path) -> tuple[int, int]:
    hypotheses = _hypotheses(root)
    searched, incomplete = 0, 0
    clock: dict[str, float] = {}
    for row in rows:
        hypothesis = hypotheses.get(row.stable_id)
        if not hypothesis:
            continue
        path = root / "checkpoints" / f"{row.stable_id}-intro-{source}.json"
        prior = json.loads(path.read_text()) if path.exists() else {}
        if prior.get("complete") and prior.get("hypothesis") == hypothesis:
            continue
        query = hypothesis["title"]
        if source == "wikipedia":
            url = urlsplit(hypothesis["wikipedia"])
            if url.hostname != "en.wikipedia.org" or not url.path.startswith("/wiki/"):
                raise RecordError("introduction hypothesis needs a Wikipedia article URL")
            query = unquote(url.path.removeprefix("/wiki/")).replace("_", " ")
        got = _query(source, query, root=root, clock=clock)
        record = {"id": row.stable_id, "hypothesis": hypothesis, "at": now(), **got}
        _write_atomic(path, json.dumps(record, ensure_ascii=False) + "\n")
        searched += 1
        incomplete += not got["complete"]
    return searched, incomplete


def _source(rows: Sequence[RosterRow], source: str, root: Path) -> tuple[int, int]:
    clock: dict[str, float] = {}
    searched, incomplete = 0, 0
    for row in rows:
        path = root / "checkpoints" / f"{row.stable_id}-{source}.json"
        prior = json.loads(path.read_text()) if path.exists() else {}
        if prior.get("complete") and prior.get("row") == _row(row):
            continue
        got = search(source, row.name, root=root, clock=clock)
        record = {"schema": SCHEMA, "id": row.stable_id, "row": _row(row),
                  "source": source, "at": now(), **got}
        _write_atomic(path, json.dumps(record, ensure_ascii=False) + "\n")
        searched += 1
        incomplete += not got["complete"]
    return searched, incomplete


def run(ids: Sequence[str] | None, *, data: DataLayout, out: Path | None = None,
        limit: int | None = None) -> dict[str, Any]:
    """Write checkpoints and combined candidate rows; explicit sources split hub/worker I/O."""
    if limit is not None and limit < 0:
        raise RecordError("resolver limit must be nonnegative")
    rows = inputs(ids, data=data)[:limit]
    root = Path(out) if out else folder(data)
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(exist_ok=True)
    sources = tuple(dict.fromkeys(os.environ.get("MENAGERIE_RESOLVE_SOURCES", ",".join(SOURCES)).split(",")))
    if not sources or set(sources) - set(SOURCES):
        raise RecordError("resolver sources must be openalex,arxiv,github,hf")
    with (root / "search.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # One serial, paced stream per source; the threads spend their time in network I/O.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as pool:
            futures = [pool.submit(_source, rows, source, root) for source in sources]
            counts = [f.result() for f in futures]
            # Title hypotheses come from a judgment lane, never from the search script.
            intro_sources = ["openalex"] if sources == ("openalex",) else ["crossref", "wikipedia"]
            if "openalex" in sources and sources != ("openalex",):
                intro_sources.append("openalex")
            intro = [pool.submit(_intro_source, rows, s, root) for s in intro_sources]
            counts += [f.result() for f in intro]
        records = _aggregate(rows, root)
        candidate_path = root / "candidates.jsonl"
        _write_atomic(candidate_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return {"rows": len(rows), "searched": sum(c[0] for c in counts),
            "incomplete": sum(c[1] for c in counts), "candidates": str(candidate_path),
            "nothing_typed": True}
