"""What the describe pass reads: a validated family's saved text and the facts code can settle.

Lane r3-describe. Runs on the hub (no torch). ``gather`` lays one family's folder out for a
prose lane and the jev classifier: the packet, each member's roster row and model record, every
saved text (paper, page, Hub card), a library's own documentation saved as a code source's text,
each code snapshot's top-level README and, where switched on and no full text exists, the abstract
of the saved S2 or OpenAlex record of the cited paper, plus ``facts.json``
with the deterministic fields: ``era`` from the citation's first public date, ``institutions``
from the saved OpenAlex record's authorships, and the stage 3 output fields of each member's
deciding trace. Nothing here touches the web.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout, ledger, roster
from menagerie.pipeline import store as store_mod
from menagerie.pipeline.describe import _bib_abstract
from menagerie.pipeline.records import LedgerRow, to_dict

TEXT_ROLES = ("paper", "page")
ABSTRACT_APIS = ("s2", "openalex")
# What describe can read, by kind, and what each kind of saved text file is (``<sid>-<x>.txt``).
# A record's ``basis`` is the strongest kind it read; a stronger kind it did not read re-opens it.
RANK = {"none": 0, "readme": 1, "docs": 2, "abstract": 2, "full": 3}
FILE_KIND = {"paper": "full", "page": "full", "docs": "docs", "abstract": "abstract",
             "readme": "readme"}
MODES = (("full", "full"), ("abstract", "abstract"), ("docs", "docs"), ("readme", "readme-only"))
README = re.compile(r"readme(\.(md|rst|txt|markdown))?", re.I)
SAFE_SID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")  # a sid names a file: no separators


def readable_sid(sid: str) -> bool:
    """Whether a source id may name a file under the family folder (never a path)."""

    return bool(SAFE_SID.fullmatch(sid)) and ".." not in sid
OUTPUT_KEYS = ("output_shape", "n_output_tensors", "output_container_kind")
MAX_TEXT_CHARS = 400_000  # a saved full text beyond this is cut for the lane folder
# Summaries are pulled from the worker store one blob at a time, so a wide family (f-vit has about
# 240 members) reads the output fields of its first members only; they decide the family's output
# type, and a member block is written only where members differ.
STAGE3_MEMBERS = 12


@dataclass(frozen=True)
class Inputs:
    """One family's describe inputs, laid out under ``folder``."""

    family: str
    folder: Path
    packet: dict[str, Any]
    members: list[dict[str, Any]]
    texts: dict[str, Path]  # sid -> saved text file (papers, pages, docs, abstracts, READMEs)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        """``full`` with a paper or page text, else ``abstract`` with a saved abstract, else
        ``docs`` with a library's documentation, ``readme-only`` with READMEs alone, else
        ``none``."""

        kinds = set(self.kinds.values())
        return next((mode for kind, mode in MODES if kind in kinds), "none")

    @property
    def kinds(self) -> dict[str, str]:
        """sid -> the kind of each saved text, from its file name (``<sid>-<x>.txt``)."""

        return {sid: FILE_KIND.get(path.stem.rsplit("-", 1)[-1], "full")
                for sid, path in self.texts.items()}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validated_ids(rows: Sequence[Any], *, tl: str) -> dict[str, str]:
    """Each model id whose latest deciding full trace at ``tl`` validates it, with its recipe
    revision (one pass over the ledger; ``ledger.deciding_trace`` per id would be quadratic)."""

    latest: dict[tuple[str, str], LedgerRow] = {}
    for row in rows:
        if not (isinstance(row, LedgerRow) and row.stage == "trace" and row.tl == tl
                and row.outcome != "infra" and row.capture in (None, "full")
                and row.mode in (None, "eval")):
            continue
        key = (row.id, row.recipe_rev)
        if key not in latest or row.at >= latest[key].at:
            latest[key] = row
    return {mid: rev for (mid, rev), row in latest.items() if ledger.validated(row)}


def validated_families(*, data: DataLayout, tl: str, rows: Sequence[Any] | None = None
                       ) -> dict[str, list[str]]:
    """Families with a validated member at its current recipe revision: family -> member ids.
    ``rows`` are the ledger's lines when the caller already read them."""

    valid = validated_ids(ledger.read_all(data=data) if rows is None else rows, tl=tl)
    found: dict[str, list[str]] = {}
    for mid, rev in valid.items():
        path = data.model(mid)
        if not path.is_file():
            continue
        record = _read_json(path)
        if (record.get("recipe") or {}).get("rev") == rev:
            found.setdefault(record["family"], []).append(mid)
    return {f: sorted(ids) for f, ids in found.items()}


def _version(pin: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", pin))


def validated_pins(*, data: DataLayout, tl: str, earlier: bool = False,
                   rows: Sequence[Any] | None = None) -> dict[str, str]:
    """Each family to describe, with the TorchLens pin its stage 3 facts are read at: the families
    validated at ``tl`` first; with ``earlier``, then each family validated only at an earlier
    pin (newest first), at that pin. A family re-validated at ``tl`` later keeps its record."""

    rows = ledger.read_all(data=data) if rows is None else rows
    found = dict.fromkeys(validated_families(data=data, tl=tl, rows=rows), tl)
    if not earlier:
        return found
    pins = {r.tl for r in rows if isinstance(r, LedgerRow) and r.stage == "trace" and r.tl}
    for pin in sorted((p for p in pins if _version(p) < _version(tl)), key=_version, reverse=True):
        for family in validated_families(data=data, tl=pin, rows=rows):
            found.setdefault(family, pin)
    return found


def _members(packet: Mapping[str, Any], rows: Mapping[str, Any], data: DataLayout
             ) -> list[dict[str, Any]]:
    members = []
    for member in packet.get("members", []):
        mid = member["id"]
        path = data.model(mid)
        model = _read_json(path) if path.is_file() else None
        row = rows.get(mid)
        members.append({"id": mid, "delta": member.get("delta", ""),
                        "name": (model or {}).get("name") or (row or {}).get("name") or mid,
                        "roster": row, "model": model})
    return members


def _save_text(sha: str, dest: Path, store: StoreLayout) -> Path:
    store_mod.materialize(sha, dest, store=store)
    text = dest.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_TEXT_CHARS:
        dest.write_text(text[:MAX_TEXT_CHARS] + "\n[cut]\n", encoding="utf-8")
    return dest


def _readme(sha: str, dest: Path, store: StoreLayout) -> Path | None:
    """A code snapshot's top-level README, saved as ``dest``; ``None`` when it has none."""

    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "tree"
        try:
            store_mod.materialize(sha, tree, store=store)
        except Exception:  # noqa: BLE001  (a blob the store no longer holds: no README)
            return None
        if not tree.is_dir():
            return None
        hit = next((p for p in sorted(tree.iterdir()) if p.is_file() and README.fullmatch(p.name)),
                   None)
        if hit is None:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(hit.read_text(encoding="utf-8", errors="replace")[:MAX_TEXT_CHARS],
                        encoding="utf-8")
    return dest


def _abstract(sha: str, title: str, dest: Path, store: StoreLayout) -> Path | None:
    """The abstract a saved S2 or OpenAlex record holds for the cited title, saved as ``dest``;
    ``None`` when it holds none."""

    try:
        text = _bib_abstract(sha, title, store, dest.parent.parent) if title else None
    except Exception:  # noqa: BLE001  (a record the store no longer holds or that does not parse)
        return None
    if not text:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text[:MAX_TEXT_CHARS], encoding="utf-8")
    return dest


def candidates(packet: Mapping[str, Any], *, abstracts: bool = False) -> dict[str, str]:
    """Each source describe could read, sid -> kind, from the packet alone (nothing fetched):
    ``full`` (a paper or page text), ``docs`` (a code source's text: a library's documentation at
    the validated version), ``readme`` (a code snapshot, which may hold a README) and, with
    ``abstracts`` and no full text, ``abstract`` (a saved S2 or OpenAlex record). A sid that is
    not a plain file name is left unread, never a path."""

    found: dict[str, str] = {}
    for src in packet.get("sources", []):
        sid, role = src.get("sid", ""), src.get("role")
        if not readable_sid(sid):
            continue
        if role in TEXT_ROLES and src.get("text"):
            found[sid] = "full"
        elif role == "code" and (src.get("text") or src.get("blob")):
            found[sid] = "docs" if src.get("text") else "readme"
        elif role == "bib" and abstracts and src.get("api") in ABSTRACT_APIS and src.get("blob"):
            found[sid] = "abstract"
    if "full" in found.values():
        found = {sid: kind for sid, kind in found.items() if kind != "abstract"}
    return found


def _texts(packet: Mapping[str, Any], folder: Path, store: StoreLayout, *,
           abstracts: bool = False) -> dict[str, Path]:
    sources = {s.get("sid"): s for s in packet.get("sources", [])}
    title = (packet.get("citation") or {}).get("title") or ""
    texts: dict[str, Path] = {}
    for sid, kind in candidates(packet, abstracts=abstracts).items():
        src, dest = sources[sid], folder / "source"
        if kind == "full":
            got = _save_text(src["text"], dest / f"{sid}-{src['role']}.txt", store)
        elif kind == "docs":
            got = _save_text(src["text"], dest / f"{sid}-docs.txt", store)
        elif kind == "readme":
            got = _readme(src["blob"], dest / f"{sid}-readme.txt", store)
        else:
            got = _abstract(src["blob"], title, dest / f"{sid}-abstract.txt", store)
        if got is not None:
            texts[sid] = got
    return texts


def institutions(packet: Mapping[str, Any], folder: Path, store: StoreLayout
                 ) -> tuple[list[str], str | None]:
    """Institutions named by the saved OpenAlex record's authorships, in order, and its sid."""

    bib = next((s for s in packet.get("sources", []) if s.get("role") == "bib"
                and s.get("api") == "openalex" and s.get("blob") and readable_sid(s["sid"])), None)
    if bib is None:
        return [], None
    try:
        path = store_mod.materialize(bib["blob"], folder / "source" / f"{bib['sid']}.json",
                                     store=store)
        record = _read_json(path)
    except Exception:  # noqa: BLE001  (an unreadable record adds nothing)
        return [], None
    names: list[str] = []
    for authorship in record.get("authorships") or []:
        for inst in authorship.get("institutions") or []:
            name = inst.get("display_name")
            if name and name not in names:
                names.append(name)
    return names, bib["sid"]


def _era(packet: Mapping[str, Any], members: Sequence[Mapping[str, Any]]) -> str | None:
    first = ((packet.get("citation") or {}).get("first_public") or "")[:4]
    if first.isdigit():
        return first
    return next((m["roster"].get("era") for m in members if m.get("roster")
                 and m["roster"].get("era")), None)


def stage3_outputs(members: Sequence[Mapping[str, Any]], rows: Sequence[Any], *, tl: str,
                   folder: Path, store: StoreLayout) -> dict[str, dict[str, Any]]:
    """The stage 3 output fields of each member's deciding trace (from its summary blob)."""

    out: dict[str, dict[str, Any]] = {}
    for member in members[:STAGE3_MEMBERS]:
        model = member.get("model") or {}
        rev = (model.get("recipe") or {}).get("rev")
        if not rev:
            continue
        trace = ledger.deciding_trace(member["id"], rows, recipe_rev=rev, tl=tl)
        if trace is None or trace.summary is None:
            continue
        try:
            path = store_mod.materialize(trace.summary, folder / "stage3" / f"{member['id']}.json",
                                         store=store)
            summary = _read_json(path)
        except Exception:  # noqa: BLE001  (no summary reachable: the field stays open)
            continue
        out[member["id"]] = {k: summary.get(k) for k in OUTPUT_KEYS}
    return out


def gather(family: str, *, data: DataLayout, store: StoreLayout, folder: Path, tl: str,
           rows: Sequence[Any] | None = None, vocab: Mapping[str, Sequence[str]] | None = None,
           jev: Mapping[str, Any] | None = None, abstracts: bool = False) -> Inputs:
    """Lay out ``folder`` for one family and return its inputs.

    Writes ``packet.json``, ``members.json``, ``source/<sid>-<kind>.txt`` per saved text, docs,
    README and (with ``abstracts``) abstract, ``stage3/<id>.json`` per summary read, and
    ``facts.json`` (``mode``, ``sids``, ``considered``: every candidate sid looked at, ``era``,
    ``institutions``, ``output`` per member, ``vocab``, ``jev``). ``rows`` are the ledger's lines
    when the caller already read them.
    """

    packet = _read_json(data.packet(family))
    rows = ledger.read_all(data=data) if rows is None else rows
    roster_rows = {r.stable_id: to_dict(r) for r in roster.read(data=data) if r.stable_id}
    members = _members(packet, roster_rows, data)
    folder.mkdir(parents=True, exist_ok=True)
    texts = _texts(packet, folder, store, abstracts=abstracts)
    names, bib_sid = institutions(packet, folder, store)
    inputs = Inputs(family, folder, packet, members, texts)
    facts = {
        "family": family, "mode": inputs.mode, "sids": sorted(texts),
        "considered": sorted(candidates(packet, abstracts=abstracts)),
        "era": _era(packet, members), "institutions": names, "institutions_sid": bib_sid,
        "output": stage3_outputs(members, rows, tl=tl, folder=folder, store=store),
        "vocab": dict(vocab or {}), "jev": dict(jev or {}),
    }
    (folder / "packet.json").write_text(json.dumps(packet, indent=1) + "\n", encoding="utf-8")
    (folder / "members.json").write_text(json.dumps(members, indent=1) + "\n", encoding="utf-8")
    (folder / "facts.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")
    return Inputs(family, folder, packet, members, texts, facts)


def excerpts(inputs: Inputs, *, chars: int = 3000) -> dict[str, str]:
    """The opening of each saved text by sid (the jev state and the drive's label lane)."""

    return {sid: path.read_text(encoding="utf-8", errors="replace")[:chars]
            for sid, path in sorted(inputs.texts.items())}
