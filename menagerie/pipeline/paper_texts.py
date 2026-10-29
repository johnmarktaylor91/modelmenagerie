"""Open-access full texts for paper sources never fetched, as describe's ``full`` text.

A packet's ``paper`` source with an arXiv id, a DOI or a URL but no saved blob gets its document
here, open access only: for arXiv the abstract page into the source itself and the PDF as a second
paper source ``<sid>-pdf``; for a DOI only the open-access PDF OpenAlex names, as ``<sid>-pdf`` (a
DOI with no open-access location is not fetched); for any other URL that document into the source.
``fetch`` stores each document's extracted text beside it, so ``describe_text.candidates`` reads it
as kind ``full`` and a record grounded on an abstract or library docs re-opens.

Each fetched paper's license is recorded the way the packet licenses step records a code tree's: a
``license`` source ``of`` the paper (sid ``rights-<sid>``; ``license-`` sids are the licenses
step's own), ``spdx`` from arXiv's rights link or OpenAlex's location license, else ``unknown``.
Texts are third-party store blobs read only for grounding quotes; nothing republishes them.
Failures rest ``RETRY_DAYS`` in ``work_dir("paper-texts")/tried.json``.

The tick's trickle fetches a few a pass (``paper_texts``); a backlog runs as its own unit,
``python -m menagerie.pipeline.paper_texts`` (``main``), which commits exactly what it wrote
(``tick_commit``) every ``--commit-every`` packets. Never run both at once: set the trickle's
``paper_texts`` to 0 while the unit runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import re
import tempfile
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from menagerie.paths import work_dir
from menagerie.pipeline import DataLayout, StoreLayout, fetch, now, read_json, write_json
from menagerie.pipeline import store as store_mod
from menagerie.pipeline import tick_commit, written
from menagerie.pipeline.records import Amendment, Packet, Source, StoreIndexLine

BY = "paper-texts"
RETRY_DAYS = 14
RIGHTS = re.compile(r'href="(https?://(?:creativecommons\.org/(?:licenses|publicdomain)/[^"]+'
                    r'|arxiv\.org/licenses/[^"]+))"', re.I)
OPENALEX_SPDX = {"cc-by": "CC-BY-4.0", "cc-by-sa": "CC-BY-SA-4.0", "cc-by-nc": "CC-BY-NC-4.0",
                 "cc-by-nc-sa": "CC-BY-NC-SA-4.0", "cc-by-nd": "CC-BY-ND-4.0",
                 "cc-by-nc-nd": "CC-BY-NC-ND-4.0", "cc0": "CC0-1.0", "public-domain": "CC0-1.0"}


def _pdf_sid(sid: str) -> str:
    return f"{sid}-pdf"


def _wanting(packet: Packet) -> list[Source]:
    """Paper sources with a URL, no saved blob and no fetched ``<sid>-pdf`` sibling."""

    sids = {s.sid for s in packet.sources}
    return [s for s in packet.sources if s.role == "paper" and s.url and not s.blob
            and _pdf_sid(s.sid) not in sids]


def due(packet: Packet) -> bool:
    """A paper source names a document that was never fetched and the packet has no full text."""

    texts = any(s.role in ("paper", "page") and s.text for s in packet.sources)
    return not texts and bool(_wanting(packet))


def _tried_path() -> Path:
    return work_dir("paper-texts") / "tried.json"


def _read_tried() -> dict[str, dict[str, str]]:
    path = _tried_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_tried(tried: Mapping[str, Any]) -> None:
    path = _tried_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(tried, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _resting(entry: Mapping[str, str] | None, at: datetime) -> bool:
    if not entry:
        return False
    when = datetime.strptime(entry["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return at - when < timedelta(days=RETRY_DAYS)


def _due_packets(families: Sequence[str] | None, data: DataLayout,
                 tried: Mapping[str, Mapping[str, str]]) -> list[Packet]:
    """Due packets not resting after a recent failure; families describe has already seen (so
    validated) first."""

    paths = ([data.packet(f) for f in dict.fromkeys(families)] if families is not None
             else sorted(data.root.glob("research/*/*.json")))
    at = datetime.now(timezone.utc)
    found = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if not any(s.get("role") == "paper" and not s.get("blob") for s in raw.get("sources") or ()):
            continue
        if _resting(tried.get(str(raw.get("family"))), at):
            continue
        packet = read_json(Packet, path)
        if due(packet):
            found.append(packet)
    return sorted(found, key=lambda p: (not data.describe(p.family).is_file(), p.family))


# --- licenses ------------------------------------------------------------------------------

def spdx_of(url: str) -> str:
    """An SPDX id for a Creative Commons or arXiv rights URL (``LicenseRef-`` for arXiv's own)."""

    parts = [p for p in urllib.parse.urlsplit(url).path.split("/") if p]
    if "arxiv.org" in url:
        return "LicenseRef-arXiv-" + "-".join(parts[1:]) if len(parts) > 1 else "unknown"
    if parts[:2] == ["publicdomain", "zero"]:
        return f"CC0-{parts[2]}" if len(parts) > 2 else "CC0-1.0"
    if parts and parts[0] == "licenses" and len(parts) >= 3:
        return f"CC-{parts[1].upper()}-{parts[2]}"
    return "unknown"


def _arxiv_rights(abs_sha: str, store: StoreLayout, data: DataLayout) -> str | None:
    """The rights URL an arXiv abstract page links (``None`` when it links none)."""

    html = store_mod.get(abs_sha, store=store, data=data).read_text("utf-8", errors="replace")
    found = RIGHTS.search(html)
    return found.group(1) if found else None


def _rights(paper: Source, url: str | None, spdx: str, taken: set[str]) -> Source:
    sid = f"rights-{paper.sid}"
    return Source(sid=sid if sid not in taken else f"{sid}-2", role="license", of=paper.sid,
                  url=url, spdx=spdx, fetched=now())


# --- fetching ------------------------------------------------------------------------------

def _oa_location(doi: str) -> tuple[str, str | None] | None:
    """OpenAlex's open-access PDF for a DOI and that location's license; ``None`` when closed."""

    url = f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='/')}"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            fetch._download(url, Path(tmp) / "work.json")  # noqa: SLF001
        except fetch.FetchError:
            return None
        work = json.loads((Path(tmp) / "work.json").read_text(encoding="utf-8"))
    for location in (work.get("best_oa_location"), *(work.get("locations") or ())):
        if location and location.get("pdf_url") and location.get("is_oa", True):
            return location["pdf_url"], location.get("license")
    return None


def _doc(url: str, *, pdf: bool, store: StoreLayout, data: DataLayout) -> tuple[str, str | None]:
    """One document and its extracted text's sha (``None`` when it has none)."""

    lines: list[StoreIndexLine] = fetch._fetch_into_store(  # noqa: SLF001
        url, store=store, data=data, paper=True, pdf=pdf)
    text = next((x.sha256 for x in lines[1:] if x.derived_from == lines[0].sha256), None)
    return lines[0].sha256, text


def _arxiv(paper: Source, arxiv: str, *, store: StoreLayout, data: DataLayout,
           taken: set[str]) -> list[Source]:
    abs_url, pdf_url = f"https://arxiv.org/abs/{arxiv}", f"https://arxiv.org/pdf/{arxiv}"
    blob, text = _doc(abs_url, pdf=False, store=store, data=data)
    out = [replace(paper, url=abs_url, blob=blob, text=text, fetched=now())]
    rights = _arxiv_rights(blob, store, data)
    out.append(_rights(paper, rights, spdx_of(rights) if rights else "unknown", taken))
    try:
        pdf, pdf_text = _doc(pdf_url, pdf=True, store=store, data=data)
    except fetch.FetchError:
        return out  # the abstract page still stands; the PDF is tried with the next text
    return [*out, Source(sid=_pdf_sid(paper.sid), role="paper", url=pdf_url, blob=pdf,
                         text=pdf_text, official=paper.official, fetched=now())]


def _paper(paper: Source, *, store: StoreLayout, data: DataLayout, taken: set[str]
           ) -> tuple[list[Source], str]:
    """The sources one paper becomes (empty when nothing open was found) and what happened."""

    ref = str(paper.url)
    arxiv, doi = fetch._arxiv_id(ref), fetch._doi(ref)  # noqa: SLF001
    if arxiv:
        got = _arxiv(paper, arxiv, store=store, data=data, taken=taken)
        return got, f"arXiv {arxiv}: {len(got)} sources"
    if doi:
        location = _oa_location(doi)
        if location is None:
            return [], f"DOI {doi}: no open-access location"
        pdf_url, license_ = location
        blob, text = _doc(pdf_url, pdf=True, store=store, data=data)
        pdf = Source(sid=_pdf_sid(paper.sid), role="paper", url=pdf_url, blob=blob, text=text,
                     official=paper.official, fetched=now())
        spdx = OPENALEX_SPDX.get(license_ or "", f"LicenseRef-{license_}" if license_ else "unknown")
        return [pdf, _rights(pdf, None, spdx, taken)], f"DOI {doi}: open-access PDF"
    if not re.match(r"https?://", ref):
        return [], f"{ref[:80]}: not an arXiv id, a DOI or a URL"
    blob, text = _doc(ref, pdf=urllib.parse.urlsplit(ref).path.lower().endswith(".pdf"),
                      store=store, data=data)
    return ([replace(paper, blob=blob, text=text, fetched=now()), _rights(paper, None, "unknown", taken)],
            f"{ref[:80]}: fetched")


def _one(packet: Packet, *, store: StoreLayout, data: DataLayout) -> dict[str, Any]:
    taken = {s.sid for s in packet.sources}
    replaced: dict[str, Source] = {}
    added: list[Source] = []
    notes = []
    for paper in _wanting(packet):
        try:
            got, note = _paper(paper, store=store, data=data, taken=taken)
        except (fetch.FetchError, ValueError, OSError) as exc:
            got, note = [], f"{paper.sid}: {type(exc).__name__}: {exc}"
        notes.append(note)
        for src in got:
            if src.sid == paper.sid:
                replaced[src.sid] = src
            else:
                added.append(src)
                taken.add(src.sid)
    texts = [s.sid for s in [*replaced.values(), *added] if s.role == "paper" and s.text]
    if not texts:
        return {"family": packet.family, "status": "none", "why": "; ".join(notes)[:400]}
    # Fetching takes minutes; the packet is read again just before the write so another writer's
    # edit since (the bib trickle) is kept.
    fresh = read_json(Packet, data.packet(packet.family))
    have = {s.sid for s in fresh.sources}
    sources = (*(replaced.get(s.sid, s) for s in fresh.sources),
               *(s for s in added if s.sid not in have))
    amendment = Amendment(at=now(), why=f"paper texts: {'; '.join(notes)}"[:300],
                          evidence=tuple(texts), changed=("sources",), by=BY)
    write_json(replace(fresh, sources=sources, amendments=(*fresh.amendments, amendment)),
               data.packet(packet.family))
    return {"family": packet.family, "status": "text", "sids": texts}


def texts(families: Sequence[str] | None, *, data: DataLayout, store: StoreLayout,
          limit: int | None = None, write: bool = True, seconds: float | None = None
          ) -> list[dict[str, Any]]:
    """Fetch the open-access text of each due packet's paper, at most ``limit`` packets and no new
    one after ``seconds``. Without ``write`` it lists the due packets (status ``due``) and fetches
    and writes nothing. A family that gained no text rests ``RETRY_DAYS``."""

    if limit is not None and limit < 0:
        raise ValueError("paper texts limit must be nonnegative")
    tried = _read_tried()
    todo = _due_packets(families, data, tried)[:limit]
    if not write:
        return [{"family": p.family, "status": "due"} for p in todo]
    began, out = time.monotonic(), []
    for packet in todo:
        if seconds is not None and time.monotonic() - began > seconds:
            break
        try:
            row = _one(packet, store=store, data=data)
        except Exception as exc:  # noqa: BLE001 -- one family's failure never stops the run
            row = {"family": packet.family, "status": "error",
                   "why": f"{type(exc).__name__}: {exc}"[:400]}
        out.append(row)
        if row["status"] == "text":
            tried.pop(packet.family, None)
        else:
            tried[packet.family] = {"at": now(), "why": row["why"]}
        _write_tried(tried)
    return out


def _commit(repo: Path, keep: str, rec: written.Writes, held: tick_commit.Owned,
            rows: Sequence[Mapping[str, Any]]) -> tuple[tick_commit.Owned, dict[str, Any]]:
    owned = held.merged(tick_commit.Owned.from_writes(rec, repo, keep=lambda r: r.startswith(keep)))
    if not owned:
        return owned, {}
    got = sum(r.get("status") == "text" for r in rows)
    message = (f"data: paper texts ({len(rows)} packets tried, {got} gained a text; packets and "
               "store-index lines it wrote)")
    out = tick_commit.commit_owned(repo, owned, message, time.monotonic() + 120)
    return out.pending, out.to_json()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the backlog: ``--commit-every`` packets at a time, each batch committed, until none is
    due or ``--seconds`` have passed. One JSON line per batch on stdout and in
    ``work_dir("paper-texts")/runs.jsonl``."""

    parser = argparse.ArgumentParser(prog="paper_texts")
    parser.add_argument("--commit-every", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=6 * 3600)
    args = parser.parse_args(argv)
    data, store = DataLayout.from_env(), StoreLayout.from_env()
    repo = Path(subprocess.run(["git", "-C", str(data.root), "rev-parse", "--show-toplevel"],
                               check=True, capture_output=True, text=True).stdout.strip())
    keep = f"{Path(data.root).resolve().relative_to(repo.resolve()).as_posix()}/"
    began, held = time.monotonic(), tick_commit.Owned()
    log = work_dir("paper-texts") / "runs.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    while time.monotonic() - began < args.seconds:
        with written.recording() as rec:
            rows = texts(None, data=data, store=store, limit=args.commit_every)
        held, commit = _commit(repo, keep, rec, held, rows)
        line = json.dumps({"at": now(), "rows": rows, "commit": commit})
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if not rows:
            break
    return 1 if held else 0


if __name__ == "__main__":
    sys.exit(main())
