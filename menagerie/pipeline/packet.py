"""Packets: the 2a draft's checks, ``packet write`` and ``bib fill``.

Split from ``research`` at rung 2 (``research.packet_write`` delegates here). ``packet write``
checks a 2a draft mechanically, fetches its bibliographic records and writes the family's packet
with what follows from it; ``bib fill`` completes a packet whose records were left pending when
an API refused. Imports neither torch nor psutil.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
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
    read_jsonl,
    roster,
    write_json,
)
from menagerie.pipeline import citecheck
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.records import (
    Amendment,
    CitationLine,
    HFRef,
    ModelRecord,
    Packet,
    RecordError,
    RosterRow,
    Source,
    StoreIndexLine,
    dumps,
    from_dict,
    to_dict,
)
from menagerie.pipeline.research import CLASSIC_ZOOS

# --- Rung 1: packet write -------------------------------------------------------------------
# The draft a 2a agent writes is the packet's JSON (``family``, ``written``, ``by`` and
# ``schema`` may be left out) whose members may carry four extra keys that ``packet write``
# consumes: ``source_case`` (else derived from ``code_order``), ``hf`` (the Hub repo id),
# ``delta_fields`` (the model record's ``delta``) and, for a roster row with no stable id yet,
# ``name``, ``zoo`` and ``variant`` in place of ``id``. Roster findings are claims whose field is
# ``roster:<row>=duplicate_of:<id>`` or ``roster:<row>=not_architecture``, ``<row>`` a stable id
# or ``name@zoo``, so every state change is a quoted, checked finding inside the packet.

PROMPT_VERSION = "2a@prompt-v2"
PENDING = "bib: pending"  # the 2a prompt's mark for records an API refused
MEMBER_EXTRAS = ("source_case", "hf", "delta_fields", "name", "zoo", "variant")
FINDING = re.compile(r"roster:(.+)=(duplicate_of:m\d+|not_architecture)")
PLACE_GROUPS = (  # every not_found names one place from each group
    ("GitHub", re.compile(r"github", re.I)),
    ("Hugging Face papers", re.compile(r"hf[- ]?papers|hugging ?face.*papers|huggingface\.co/papers",
                                       re.I)),
    ("arXiv or OpenAlex", re.compile(r"arxiv|openalex", re.I)),
    ("an open-web search", re.compile(r"web|google|bing|duckduckgo|search engine", re.I)),
)
NATIVE = frozenset({"tensorflow", "jax", "paddle", "mlx", "tinygrad"})


def _norm_text(text: str) -> str:
    """Whitespace- and ligature-normalized text for verbatim quote checks."""

    return " ".join(unicodedata.normalize("NFKC", text).split())


class _Blobs:
    """Store reads for one packet check: index lines once, blob bytes from staging or the worker."""

    def __init__(self, store: StoreLayout, data: DataLayout) -> None:
        self.store = store
        folder = data.root / "store-index"
        self.lines = [line for shard in sorted(folder.glob("*.jsonl"))
                      for line in read_jsonl(StoreIndexLine, shard)]
        self.index = {line.sha256: line for line in self.lines}
        self._text: dict[str, str] = {}

    def text(self, sha: str) -> str:
        """Readable text of a blob: a snapshot's text files, else the bytes decoded."""

        if sha not in self._text:
            with tempfile.TemporaryDirectory() as tmp:
                path = store_mod.materialize(sha, Path(tmp) / "blob", store=self.store) \
                    if not self.store.blob(sha).is_file() else self.store.blob(sha)
                if path.is_dir():
                    body = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                                     for p in sorted(path.rglob("*")) if p.is_file())
                elif store_mod.snapshot_manifest(path) is not None:
                    body = "\n".join(b.decode("utf-8", "replace")
                                     for b in store_mod.snapshot_files(path).values())
                else:
                    body = path.read_bytes().decode("utf-8", errors="replace")
            self._text[sha] = _norm_text(body)
        return self._text[sha]


def _author(drafted: str | None) -> str:
    """Who wrote the packet: the lane's own launch record (role, and the harness and model that
    ran), so a 2a lane on either lab is recorded as what ran; else the draft's ``by``."""

    lane = os.environ.get("FLEET_RUN_ID")
    log = work_dir("lanes") / lane / "events.jsonl" if lane else None
    lines = log.read_text(encoding="utf-8").splitlines() if log and log.is_file() else []
    started = next((e for e in map(json.loads, filter(None, lines)) if e.get("verb") == "launch"),
                   None)
    if started:
        return f"{PROMPT_VERSION} {started.get('role')} ({started.get('harness')} " \
               f"{started.get('model')})"
    return drafted or PROMPT_VERSION


def _split_draft(raw: dict[str, Any], family: str, rows: dict[str, RosterRow],
                 keyed: dict[tuple[str, str, str], RosterRow], errors: list[str]
                 ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate the member extras from the packet JSON; placeholder ids for rows to mint."""

    if raw.get("family", family) != family:
        errors.append(f"draft family {raw.get('family')!r} is not {family!r}")
    packet = {**raw, "schema": Packet.SCHEMA, "family": family, "written": now(),
              "by": _author(raw.get("by"))}
    extras, members = [], []
    for n, member in enumerate(raw.get("members") or []):
        extra = {k: member[k] for k in MEMBER_EXTRAS if k in member}
        if "id" not in member:
            key = (member.get("name", ""), member.get("zoo", ""), member.get("variant", ""))
            row = keyed.get(key)
            if row is None:
                errors.append(f"member {n}: no roster row {key}")
            elif row.stable_id:
                member = {**member, "id": row.stable_id}
            extra["mint"] = key
        elif member["id"] not in rows:
            errors.append(f"member {member['id']}: not in the roster")
        members.append({"id": member.get("id", f"m0{n}"), "delta": member.get("delta", "")})
        extras.append(extra)
    packet["members"] = members
    return packet, extras


def _check_store(packet: Packet, blobs: _Blobs, errors: list[str]) -> None:
    for source in packet.sources:
        for sha in filter(None, (source.blob, source.text)):
            if sha not in blobs.index:
                errors.append(f"source {source.sid}: {sha[:12]} is not in the store index")


def _check_claims(packet: Packet, blobs: _Blobs, errors: list[str]) -> None:
    by_sid = {s.sid: s for s in packet.sources}
    for n, claim in enumerate(packet.claims):
        source = by_sid.get(claim.sid)
        sha = source and (source.text or source.blob)
        if not sha or sha not in blobs.index:
            errors.append(f"claim {n} ({claim.field}): source {claim.sid} has no saved text")
            continue
        try:
            text = blobs.text(sha)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            errors.append(f"claim {n} ({claim.field}): cannot read {claim.sid}: {exc}")
            continue
        if _norm_text(claim.quote) not in text:
            errors.append(f"claim {n} ({claim.field}): quote not found verbatim in {claim.sid}: "
                          f"{claim.quote[:80]!r}")


FOUND_NEEDS = {  # slot: (source roles with saved content that back "found", what is missing)
    "paper": ({"paper", "page"}, "no paper source has saved content"),
    "code": ({"code", "library"}, "no code or library source has saved content"),
    "license": ({"license"}, "no license blob is saved"),
}


def _check_found(packet: Packet, errors: list[str]) -> None:
    """``found`` needs saved content: a URL without retained readable content does not count."""

    held = {s.role for s in packet.sources if s.blob or s.text}
    held |= {"library" for s in packet.sources if s.role == "library"}
    held |= {"license" for s in packet.sources if s.role == "code" and s.blob and s.spdx}
    for slot, (roles, missing) in FOUND_NEEDS.items():
        if getattr(packet.availability, slot) == "found" and not roles & held:
            errors.append(f"{slot} is found but {missing}")


def _check_places(packet: Packet, errors: list[str]) -> None:
    """A ``not_found`` names where it searched; ``sufficient`` with no code had a second search."""

    avail = packet.availability
    places = " | ".join(packet.search.places)
    unfound = [slot for slot in ("paper", "code", "license", "description")
               if getattr(avail, slot) == "not_found"]
    lacking = [name for name, pattern in PLACE_GROUPS if not pattern.search(places)]
    if unfound and lacking:
        errors.append(f"{', '.join(unfound)} not_found, yet search.places lacks "
                      f"{', '.join(lacking)}")
    if avail.description == "sufficient" and avail.code != "found" \
            and "second-search" not in places:
        errors.append("description is sufficient with no code found: run `research second-search` "
                      "and name second-search in search.places")


def _check_citation(packet: Packet, errors: list[str]) -> None:
    if packet.availability.paper != "found":
        return
    cite = packet.citation
    if cite is None:
        errors.append("paper is found but there is no citation")
    elif not (cite.arxiv or cite.doi) and "no_identifier" not in packet.search.note:
        errors.append("the citation has no arXiv id or DOI: add one, or write no_identifier "
                      "in search.note")


def _check_formats(packet: Packet, errors: list[str]) -> None:
    """Citation dates, ids and license SPDX ids well formed (a deprecated SPDX id is allowed)."""

    errors += citecheck.citation_notes(packet.citation)
    errors += [n for n in citecheck.spdx_notes(packet) if n.startswith("spdx-invalid")]


def _findings(packet: Packet, rows: dict[str, RosterRow],
              keyed_name: dict[tuple[str, str], RosterRow], errors: list[str]
              ) -> list[tuple[RosterRow, str]]:
    out = []
    for claim in packet.claims:
        if not claim.field.startswith("roster:"):
            continue
        match = FINDING.fullmatch(claim.field)
        if not match:
            errors.append(f"finding {claim.field!r}: want roster:<id or name@zoo>="
                          "duplicate_of:<id>|not_architecture")
            continue
        ref, state = match.groups()
        name, _, zoo = ref.rpartition("@")
        row = rows.get(ref) or keyed_name.get((name, zoo))
        target = state.split(":", 1)[1] if state.startswith("duplicate_of") else None
        if row is None:
            errors.append(f"finding {claim.field!r}: no roster row {ref!r}")
        elif target and target not in rows:
            errors.append(f"finding {claim.field!r}: {target} is not a roster id")
        else:
            out.append((row, state))
    return out


def _hub_refs(packet: Packet, extras: list[dict[str, Any]], blobs: _Blobs,
              errors: list[str]) -> dict[str, HFRef]:
    """The model record's ``hf`` per Hub repo id: the saved card and ``config.json``."""

    refs = {}
    for repo in {e["hf"] for e in extras if e.get("hf")}:
        url = f"https://huggingface.co/{repo}"
        card = next((s.blob for s in packet.sources if s.role == "page" and s.url == url
                     and s.blob), None)
        # Every index line, not one per sha: two repos may ship byte-identical config.json.
        config = next((line.sha256 for line in blobs.lines if line.url and re.fullmatch(
            rf"{re.escape(url)}/resolve/[0-9a-f]{{40}}/config\.json", line.url)), None)
        if not (card and config):
            errors.append(f"hub model {repo}: card and config.json must be saved "
                          f"(`fetch hf {repo}`), and its card source kept in the packet")
        refs[repo] = HFRef(id=repo, card=card, config=config)
    return refs


def _source_case(packet: Packet, extra: dict[str, Any], zoo: str) -> str:
    if extra.get("source_case"):
        return extra["source_case"]
    by_sid = {s.sid: s for s in packet.sources}
    first = by_sid.get(packet.code_order[0]) if packet.code_order else None
    if first is None:
        return "classic" if zoo in CLASSIC_ZOOS else "reimplement"
    if first.role == "library":
        return "library"
    return framework_case(first.framework or "")


def framework_case(framework: str) -> str:
    """Vendor, native or port from a source's framework, which may carry a version or a note.

    The 2a lanes write e.g. 'pytorch 1.0' or 'pytorch (torch_geometric)', so the first word
    decides. TensorFlow and Paddle before 2.0 (graph mode, fluid) do not run as-is in the
    native environments and stay ports; a 2b lane may still set the case its recipe needs.
    """

    text = framework.strip().lower()
    word = re.match(r"([a-z][a-z0-9_-]*)\s*(\d+)?", text)
    name, major = (word.group(1), word.group(2)) if word else ("", None)
    if name in ("pytorch", "torch"):
        return "vendor"
    if name in ("tensorflow", "paddle") and major is not None and int(major) < 2:
        return "port"
    return "native" if name in NATIVE else "port"


def _bib_sources(packet: Packet, refs: Sequence[str], *, family: str, store: StoreLayout,
                 data: DataLayout) -> tuple[list[Source], list[str], dict[str, Any]]:
    """The packet's sources plus every new bibliographic record, the notes and the records."""

    sources, notes, records = list(packet.sources), [], {}
    known = {s.blob for s in sources if s.blob}
    for ref in refs:
        got = fetch.bib_records(ref, family=family, store=store, data=data)
        sources += [s for s in got.sources if s.blob not in known]
        known |= {s.blob for s in got.sources}
        notes += got.notes
        records = {**got.records, **records}
    return sources, notes, records


def _published(records: Mapping[str, Any]) -> str | None:
    """The proceedings or journal date: Crossref's, else a venue record's from OpenAlex."""

    message = (records.get("crossref") or {}).get("message") or {}
    parts = (message.get("published-print") or message.get("published") or {}).get("date-parts")
    if parts and parts[0]:
        return "-".join(f"{int(p):02d}" for p in parts[0])
    work = records.get("openalex") or {}
    venue = work.get("publication_date") and "10.48550" not in str(work.get("doi"))
    return work["publication_date"] if venue else None  # not the arXiv preprint's date


def _seed_line(family: str, records: Mapping[str, Any]) -> CitationLine | None:
    s2 = (records.get("s2") or {}).get("citationCount")
    openalex = (records.get("openalex") or {}).get("cited_by_count")
    if s2 is None and openalex is None:
        return None
    return CitationLine(family=family, at=now(), s2=s2, openalex=openalex,
                        from_="s2" if s2 is not None else "openalex")


def _with_bib(packet: Packet, *, family: str, store: StoreLayout, data: DataLayout
              ) -> tuple[Packet, CitationLine | None]:
    """Fetch the bibliographic records of every citation identifier into the packet."""

    cite = packet.citation
    refs = [r for r in ((f"arXiv:{cite.arxiv}" if cite.arxiv else None), cite.doi) if r] \
        if cite else []
    if not cite or not refs:
        return packet, None
    sources, notes, records = _bib_sources(packet, refs, family=family, store=store, data=data)
    # The records must be of the cited paper: on a title mismatch the note says so and no
    # identifier or date is taken from them (the rung 0 path's title match).
    mismatch = citecheck.title_note(cite.title, records)
    if mismatch:
        notes, records = [*notes, mismatch], {}
    s2, work = records.get("s2") or {}, records.get("openalex") or {}
    citation = replace(cite, bib=tuple(s.sid for s in sources if s.role == "bib"),
                       s2=s2.get("paperId") or cite.s2,
                       openalex=str(work.get("id") or "").rsplit("/", 1)[-1] or cite.openalex,
                       published=_published(records) or cite.published)
    fresh = [n for n in dict.fromkeys(notes) if n not in packet.search.note]
    search = replace(packet.search, note="; ".join(filter(None, (packet.search.note, *fresh))))
    return (replace(packet, sources=tuple(sources), citation=citation, search=search),
            _seed_line(family, records))


def _license_line(code: Source, blobs: _Blobs) -> StoreIndexLine | None:
    """The LICENSE a fetched code tree saved: the first index line derived from its blob.

    Every line, not one per sha: one stock LICENSE text (Apache-2.0) is saved by many trees, and
    the sha-keyed index keeps only the tree that saved it last."""

    return next((x for x in blobs.lines if x.derived_from == code.blob
                 and x.mime.startswith("text/plain")), None)


def _license_source(code: Source, line: StoreIndexLine, taken: set[str]) -> Source:
    sid = f"license-{line.sha256[:8]}"
    return Source(sid=sid if sid not in taken else f"{sid}-{code.sid}", role="license",
                  of=code.sid, url=line.url, blob=line.sha256, spdx=code.spdx or "unknown",
                  fetched=line.fetched)


def _with_licenses(packet: Packet, blobs: _Blobs) -> Packet:
    """Add the ``license`` source of every fetched code tree that saved a LICENSE blob."""

    covered = {s.of for s in packet.sources if s.role == "license"}
    taken = {s.sid for s in packet.sources}
    added: list[Source] = []
    for source in packet.sources:
        if source.role != "code" or not source.blob or source.sid in covered:
            continue
        line = _license_line(source, blobs)
        if line is not None:
            added.append(_license_source(source, line, taken))
            taken.add(added[-1].sid)
    return replace(packet, sources=packet.sources + tuple(added)) if added else packet


def _relicensed(packet: Packet, blobs: _Blobs) -> tuple[Packet, list[str]]:
    """The packet with each derived ``license-`` source pointing at its own tree's LICENSE copy,
    and any missing one added; with the source ids changed or added."""

    by_sid = {s.sid: s for s in packet.sources}
    sources, changed = [], []
    for source in packet.sources:
        code = by_sid.get(source.of or "") if source.role == "license" \
            and source.sid.startswith("license-") else None
        line = _license_line(code, blobs) if code is not None and code.blob else None
        if line is not None and (line.url, line.sha256) != (source.url, source.blob):
            source = replace(source, url=line.url, blob=line.sha256, fetched=line.fetched)
            changed.append(source.sid)
        sources.append(source)
    fixed = _with_licenses(replace(packet, sources=tuple(sources)), blobs)
    return fixed, changed + [s.sid for s in fixed.sources[len(sources):]]


def relicense(families: Sequence[str] | None, *, store: StoreLayout, data: DataLayout,
              write: bool = True) -> list[str]:
    """Re-resolve written packets' derived license sources by each code tree's own saved copy.

    Inputs: families (default every written packet); ``write`` false reports only. A
    ``license-`` source whose URL or blob is not its tree's own LICENSE line is corrected, and a
    tree's missing one added, under a new amendment (``by`` ``packet-licenses``). Returns: one
    line per corrected packet. Side effects: the packets and a ``packet_licenses`` drive event
    each.
    """

    from menagerie.pipeline import drive

    blobs = _Blobs(store, data)
    paths = [data.packet(f) for f in families] if families else \
        sorted((data.root / "research").glob("*/*.json"))
    out = []
    for path in paths:
        packet = read_json(Packet, path)
        fixed, sids = _relicensed(packet, blobs)
        if not sids:
            continue
        out.append(f"{packet.family}: {', '.join(sids)}")
        if not write:
            continue
        amendment = Amendment(
            at=now(), why="licenses: each code tree's LICENSE re-resolved from its own saved "
            "copy (sha256 and derived_from), not the last tree that saved the same text",
            evidence=tuple(sids), changed=("sources",), by="packet-licenses")
        write_json(replace(fixed, amendments=(*fixed.amendments, amendment)), path)
        drive.event(os.environ.get("FLEET_RUN_ID") or "manual", packet.family,
                    "packet_licenses", sources=sids)
    return out


def _locked_roster(data: DataLayout, change: Callable[[list[RosterRow]], list[RosterRow]]) -> None:
    """Rewrite changed roster lines in place under an exclusive lock; other lines stay as bytes."""

    with data.roster.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        lines = handle.read().splitlines(keepends=True)
        rows = [from_dict(RosterRow, {"schema": RosterRow.SCHEMA, **json.loads(line)})
                for line in lines]
        new = change(rows)
        out = [line if old == row else dumps(row) + "\n"
               for line, old, row in zip(lines, rows, new)]
        if out != lines:
            handle.seek(0)
            handle.write("".join(out))
            handle.truncate()


def _restamp(rows: list[RosterRow], minted: Mapping[tuple[str, str, str], str],
             states: Mapping[tuple[str, str, str], str]) -> list[RosterRow]:
    out = []
    for row in rows:
        key = (row.name, row.zoo, row.variant)
        if key in minted and not row.stable_id:
            row = replace(row, stable_id=minted[key])
        out.append(replace(row, state=states[key]) if key in states else row)
    return out


def _apply_roster(data: DataLayout, mint: set[tuple[str, str, str]],
                  findings: list[tuple[RosterRow, str]]) -> dict[tuple[str, str, str], str]:
    """Mint ids for new members and set ``state`` for findings; return the minted ids."""

    from menagerie.identity import ensure_stable_ids

    minted: dict[tuple[str, str, str], str] = {}
    states = {(r.name, r.zoo, r.variant): state for r, state in findings}

    def change(rows: list[RosterRow]) -> list[RosterRow]:
        if mint:  # under the roster lock, so two lanes never mint the same ordinal
            table = ensure_stable_ids(mint, data.root / "stable_ids.jsonl")
            minted.update({key: table[key] for key in mint})
        return _restamp(rows, minted, states)

    _locked_roster(data, change)
    return minted


def _model_records(packet: Packet, extras: list[dict[str, Any]], rows: dict[str, RosterRow],
                   hub: dict[str, HFRef], data: DataLayout) -> None:
    """Write each member's identity fields. An existing record keeps its ``delta`` and ``hf``
    unless the draft gives that member new ones (``delta_fields``, ``hf``): 2b records the delta,
    and a packet amendment is not a rebuild. ``source_case`` always follows the packet."""

    for member, extra in zip(packet.members, extras):
        row = rows[member.id]
        path = data.model(member.id)
        identity = {"name": row.name, "zoo": row.zoo, "variant": row.variant,
                    "family": packet.family, "source_case": _source_case(packet, extra, row.zoo)}
        given = {"delta": dict(extra.get("delta_fields") or {}),
                 "hf": hub.get(extra.get("hf", ""))}
        if path.exists():
            named = {"delta": "delta_fields", "hf": "hf"}
            record = replace(read_json(ModelRecord, path), **identity,
                             **{k: v for k, v in given.items() if named[k] in extra})
        else:
            record = ModelRecord(id=member.id, **identity, **given)
        write_json(record, path)


def _check_amendments(packet: Packet, data: DataLayout, errors: list[str]) -> None:
    path = data.packet(packet.family)
    if not path.exists():
        return
    old = read_json(Packet, path).amendments
    if packet.amendments[:len(old)] != old or len(packet.amendments) <= len(old):
        errors.append(f"{packet.family} already has a packet: keep its {len(old)} amendments and "
                      "add one saying what changed and why")


@dataclass(frozen=True)
class _Checked:
    """A draft that passed every check, with what writing it needs."""

    packet: Packet
    extras: list[dict[str, Any]]
    rows: dict[str, RosterRow]
    findings: list[tuple[RosterRow, str]]
    hub: dict[str, HFRef]


def _checked(family: str, path: Path, *, store: StoreLayout, data: DataLayout) -> _Checked:
    """Run every mechanical check on a draft; raise one ``RecordError`` listing all failures."""

    errors: list[str] = []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    all_rows = roster.read(data=data)
    rows = {r.stable_id: r for r in all_rows if r.stable_id}
    keyed = {(r.name, r.zoo, r.variant): r for r in all_rows}
    draft, extras = _split_draft(raw, family, rows, keyed, errors)
    try:
        packet = from_dict(Packet, draft)
    except RecordError as exc:
        raise RecordError(f"packet {family} refused:\n- " + "\n- ".join([*errors, str(exc)])) \
            from exc
    blobs = _Blobs(store, data)
    for check in (_check_store, _check_claims):
        check(packet, blobs, errors)
    for rule in (_check_found, _check_places, _check_citation, _check_formats):
        rule(packet, errors)
    findings = _findings(packet, rows, {(r.name, r.zoo): r for r in all_rows}, errors)
    hub = _hub_refs(packet, extras, blobs, errors)
    _check_amendments(packet, data, errors)
    if errors:
        raise RecordError(f"packet {family} refused:\n- " + "\n- ".join(errors))
    return _Checked(packet, extras, rows, findings, hub)


def _with_minted(packet: Packet, extras: list[dict[str, Any]],
                 minted: Mapping[tuple[str, str, str], str]) -> Packet:
    members = tuple(replace(m, id=minted[e["mint"]]) if e.get("mint") in minted else m
                    for m, e in zip(packet.members, extras))
    return replace(packet, members=members)


def packet_write(family: str, path: Path, *, store: StoreLayout, data: DataLayout) -> Packet:
    """Check a draft packet and write it as the family's packet, with what follows from it.

    Inputs: the family id and a draft packet JSON file (see the section comment for the member
    extras and roster findings). Checks, all reported together as one ``RecordError``: the
    schema; every ``blob`` and ``text`` in the store index; every quote verbatim in its source's
    text (whitespace and ligatures normalized); availability ``found`` backed by saved content;
    every ``not_found`` naming GitHub, Hugging Face papers, arXiv or OpenAlex and an open-web
    search in ``search.places``; ``description: sufficient`` with no code only after a second
    search; a found paper cited with an arXiv id or DOI, or ``no_identifier`` in ``search.note``;
    every Hub member's card and config saved; roster findings that resolve; an existing packet
    changed only by a new amendment. Then ``fetch_bib`` runs for each citation identifier
    (records added as ``bib`` sources, ``openalex``, ``s2`` and ``published`` filled, refusals
    noted in ``search.note``) and each fetched code tree's LICENSE becomes a ``license`` source.
    Returns: the written packet. Side effects: ``research/<xx>/<family>.json``; each member's
    model record (identity fields only; a recipe or give-up is kept); minted ids and roster
    ``state`` for findings; one citation seed line; a ``packet_write`` drive event.
    """

    checked = _checked(family, path, store=store, data=data)
    packet, seed = _with_bib(checked.packet, family=family, store=store, data=data)
    packet = _with_licenses(packet, _Blobs(store, data))
    mint = {e["mint"] for e in checked.extras if "mint" in e}
    minted = _apply_roster(data, mint, checked.findings)
    rows = checked.rows
    if minted:
        packet = _with_minted(packet, checked.extras, minted)
        rows = {r.stable_id: r for r in roster.read(data=data) if r.stable_id}
    write_json(packet, data.packet(family))
    _model_records(packet, checked.extras, rows, checked.hub, data)
    if seed is not None:
        append_jsonl([seed], data.cite(month_of(seed.at)))
    from menagerie.pipeline import drive

    drive.event(os.environ.get("FLEET_RUN_ID") or "manual", family, "packet_write",
                members=[m.id for m in packet.members], sources=len(packet.sources),
                availability=to_dict(packet.availability), findings=len(checked.findings))
    return packet


def _unpended(note: str, fresh: Sequence[str]) -> str:
    """The note without ``bib: pending`` once no API refused or failed in this fill."""

    if any(word in n for n in fresh for word in ("refused", "error", "http_")):
        return note
    parts = [p.strip() for p in note.split(";") if p.strip() and p.strip() != PENDING]
    return "; ".join(p.replace(PENDING, "").strip(" ,") for p in parts)


def bib_fill(family: str, *, store: StoreLayout, data: DataLayout) -> Packet:
    """Fetch the bibliographic records a packet left pending (``bib: pending`` in its note).

    Inputs: a family whose packet's ``search.note`` holds ``bib: pending`` (``packet write``
    wrote it when an API refused). Runs the same fetch as ``packet write``; new ``bib`` sources,
    ``openalex``, ``s2`` and ``published`` are added with an amendment naming them, and the mark
    is dropped once no API refused or failed. Returns: the packet (unchanged when nothing new
    came back). Side effects: the packet, blobs and index lines, a citation seed line, a
    ``bib_fill`` drive event.
    """

    from menagerie.pipeline import drive

    path = data.packet(family)
    packet = read_json(Packet, path)
    if PENDING not in packet.search.note:
        raise RecordError(f"{family}: its packet has no {PENDING!r} in search.note")
    filled, seed = _with_bib(packet, family=family, store=store, data=data)
    known = {s.sid for s in packet.sources}
    new = tuple(s.sid for s in filled.sources if s.role == "bib" and s.sid not in known)
    fresh = [n.strip() for n in filled.search.note.split(";")
             if n.strip() and n.strip() not in packet.search.note]
    drive.event(os.environ.get("FLEET_RUN_ID") or "manual", family, "bib_fill", added=list(new),
                notes=fresh)
    if not new:
        return packet
    note = _unpended(filled.search.note, fresh)
    amendment = Amendment(at=now(), why="bib fill: records left pending after an API refusal",
                          evidence=new, changed=("sources", "citation", "search.note"),
                          by="bib-fill")
    packet = replace(filled, search=replace(filled.search, note=note),
                     amendments=(*filled.amendments, amendment))
    write_json(packet, path)
    if seed is not None:
        append_jsonl([seed], data.cite(month_of(seed.at)))
    return packet
