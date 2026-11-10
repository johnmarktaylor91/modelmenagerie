"""The site export's side lists: shared-diagram counts, the roster index and the families file.

``publish.export_site`` calls these after it has built the records.

- ``mark_diagrams`` sets ``own_diagram`` (every exported record's default SVG comes from its own
  deciding trace: stand-ins are refused before a record is built) and ``diagram_shared_with``
  (how many other exported records' default SVG has the same hash: the same graph drawn from a
  different record's own trace).
- ``roster_index`` lists every active roster row that is not exported, as a text entry with a
  ``listed_status`` and a short public phrase (``listed_reason``) derived from the model record's
  typed give-up class. The raw class never reaches the export.
- ``families`` lists every family with at least one exported member.
- ``drop_missing_blobs`` takes out every record whose default SVG is not in the store (the site
  would show a broken picture) and lists each missing blob of the export, with what was done.

Field types and allowed values: ``docs/site-export-schema.md``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from menagerie.pipeline import publish_names
from menagerie.pipeline.records import ModelRecord, Packet, RosterRow

LISTED_STATUSES = ("not_yet_traced", "source_unavailable", "paper_only")
# Give-up heads that mean the source code is missing or cannot be reached.
SOURCE_UNAVAILABLE = frozenset({"not_found", "inaccessible", "ambiguous"})
# Give-up heads that mean the model exists only as a paper description, with no code.
PAPER_ONLY = frozenset({"insufficient_description"})
# Give-up heads that say the row is not a model of its own; such rows are not listed at all.
NOT_LISTED = frozenset({"not_architecture", "duplicate_of"})
# The public phrase per give-up head. Plain words only: the site's copy rules refuse internal
# vocabulary, and a head missing here gives no phrase.
REASONS = {
    "not_found": "No public source code was found",
    "inaccessible": "The source code could not be reached",
    "ambiguous": "The source code could not be identified with confidence",
    "insufficient_description": "Described in a paper with no public code",
    "oversized": "Too large to trace on the machines available",
    "memory_cap": "Too large to trace on the machines available",
    "needs_gpu": "Needs a GPU to trace",
    "missing_dependency": "A dependency could not be installed",
    "env_unresolved": "A dependency could not be installed",
    "needs_env": "A dependency could not be installed",
    "compiled_ext": "Needs compiled extensions that could not be built",
    "needs_weights": "Needs pretrained weights to build",
    "not_traceable_framework": "Written in a framework that cannot be traced yet",
    "timeout": "Tracing did not finish in the time allowed",
    "budget": "Tracing did not finish in the time allowed",
    "construct_error": "The model could not be built from its code",
    "forward_error": "The forward pass failed",
    "capture_error": "Tracing has not succeeded yet",
    "torchlens_gap": "Tracing has not succeeded yet",
    "torchlens_unverified": "Traced, but the trace could not be checked",
    "degenerate_forward": "The forward pass gave no usable output",
    "standin": "No faithful build of the original code yet",
    "june_code": "No faithful build of the original code yet",
    "unpublishable": "No faithful build of the original code yet",
    "fabrication": "No faithful build of the original code yet",
}
ROSTER_PAPER_ONLY = "Described in a paper with no public code"
ROSTER_NAME_ONLY = "Known by name only, with no public code"
# Links that point at a paper, not at code.
_PAPER_LINK = re.compile(r"arxiv\.org|doi\.org|openreview\.net|aclanthology|\.pdf$|^arxiv:"
                         r"|^doi:|semanticscholar|papers\.nips|proceedings\.", re.IGNORECASE)
_ARXIV_ID = re.compile(r"^arxiv:\s*(\d{4}\.\d{4,5}(v\d+)?)$", re.IGNORECASE)


def mark_diagrams(records: Sequence[dict[str, Any]]) -> None:
    """Set ``own_diagram`` and ``diagram_shared_with`` on every record, in place."""

    default_svg = {r["stable_id"]: _default_svg(r) for r in records}
    users: dict[str, int] = defaultdict(int)
    for sha in default_svg.values():
        users[sha] += 1
    for record in records:
        record["own_diagram"] = True
        record["diagram_shared_with"] = users[default_svg[record["stable_id"]]] - 1


def mark_shared_summaries(records: Sequence[dict[str, Any]]) -> None:
    """Set ``summary_shared_with`` on every record, in place: how many other exported models
    carry the same summary text. A describe record is written per family, so its summary
    shows on every member's page; the site says so when the count is above zero."""

    users: dict[str, int] = defaultdict(int)
    for record in records:
        if record.get("summary_markdown"):
            users[record["summary_markdown"]] += 1
    for record in records:
        record["summary_shared_with"] = max(users[record.get("summary_markdown") or ""] - 1, 0)


def _default_svg(record: Mapping[str, Any]) -> str:
    return next(v["svg"]["sha256"] for v in record["variants"]
                if v["key"] == record["default_variant"])


def listed_status(model: ModelRecord | None, row: RosterRow) -> tuple[str, str | None]:
    """Return ``(listed_status, listed_reason)`` for a roster row that is not exported."""

    head = model.give_up.cls.split(":")[0] if model and model.give_up else None
    if head in SOURCE_UNAVAILABLE:
        return "source_unavailable", REASONS[head]
    if head in PAPER_ONLY:
        return "paper_only", REASONS[head]
    if model is None or model.recipe is None and head is None:
        if row.triage_class == "REIMPLEMENT" and not _code_links(row):
            paper = _paper_link(row)
            return "paper_only", ROSTER_PAPER_ONLY if paper else ROSTER_NAME_ONLY
    return "not_yet_traced", REASONS.get(head or "")


def roster_index(roster_rows: Iterable[RosterRow], *, models: Mapping[str, ModelRecord],
                 exported: set[str], packets: Mapping[str, Packet],
                 source_url: Any, paper_url: Any, year: Any) -> list[dict[str, Any]]:
    """Return one text entry per active roster row whose model is not exported.

    ``source_url(model, packet)``, ``paper_url(packet)`` and ``year(packet, row)`` are the
    export's own readers, so a listed row and a record agree on every shared field.
    """

    entries: dict[str, dict[str, Any]] = {}
    for row in roster_rows:
        sid = row.stable_id
        if row.state != "active" or not sid or sid in exported or sid in entries:
            continue
        model = models.get(sid)
        if model and model.give_up and model.give_up.cls.split(":")[0] in NOT_LISTED:
            continue
        packet = packets.get(model.family) if model else None
        status, reason = listed_status(model, row)
        entries[sid] = {
            "stable_id": sid,
            "display_name": model.name if model else row.name,
            "family_normalized": _family(model, packet, row),
            "zoo": publish_names.bare_zoo(model.zoo if model else row.zoo),
            "source_url": (source_url(model, packet) if model and packet else None)
            or _first(_code_links(row)),
            "paper_url": (paper_url(packet) if packet else None) or _paper_link(row),
            "year": year(packet, row),
            "listed_status": status,
            "listed_reason": reason,
        }
    return [entries[k] for k in sorted(entries, key=_id_order)]


def _id_order(sid: str) -> tuple[int, str]:
    digits = re.sub(r"\D", "", sid)
    return (int(digits) if digits else 0, sid)


def _family(model: ModelRecord | None, packet: Packet | None, row: RosterRow) -> str | None:
    if packet is not None:
        return packet.identity.canonical
    if model is not None:
        return model.family[2:]
    return row.family or None


def _links(row: RosterRow) -> list[str]:
    seen: list[str] = []
    for url in (row.source_url, *row.hint_urls):
        if url and url not in seen:
            seen.append(url)
    return seen


def _code_links(row: RosterRow) -> list[str]:
    return [u for u in _links(row) if u.startswith(("http://", "https://"))
            and not _PAPER_LINK.search(u)]


def _paper_link(row: RosterRow) -> str | None:
    """The row's paper as a public URL: its paper hint, else a paper-looking link."""

    for value in (row.hint_paper, *_links(row)):
        url = _paper_url_of(value)
        if url:
            return url
    return None


def _paper_url_of(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    arxiv = _ARXIV_ID.match(value)
    if arxiv:
        return f"https://arxiv.org/abs/{arxiv[1]}"
    if value.lower().startswith("doi:"):
        return f"https://doi.org/{value[4:].strip()}"
    if value.startswith(("http://", "https://")) and _PAPER_LINK.search(value):
        return value
    return None


def _first(values: Sequence[str]) -> str | None:
    return values[0] if values else None


def families(records: Sequence[Mapping[str, Any]], *, models: Mapping[str, ModelRecord],
             describes: Mapping[str, Any], packets: Mapping[str, Packet],
             citation: Any, paper_url: Any) -> list[dict[str, Any]]:
    """Return one entry per family (``family_normalized``) with at least one exported member.

    ``summary`` is the family's grounded describe summary; ``paper_url`` and ``paper_title`` come
    from the first member family whose packet cites a paper. ``citation(packet)`` and
    ``paper_url(packet)`` are the export's own readers.
    """

    members: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["family_normalized"]:
            members[record["family_normalized"]].append(record)
    out = []
    for name in sorted(members, key=str.lower):
        group = sorted(members[name], key=lambda r: r["slug"])
        family_ids = sorted({models[r["stable_id"]].family for r in group
                             if r["stable_id"] in models})
        describe = next((describes[f] for f in family_ids if f in describes
                         and describes[f].grounding_state != "ungrounded"
                         and describes[f].summary_markdown), None)
        cited = next(((packets[f], c) for f in family_ids if f in packets
                      and (c := citation(packets[f])) is not None), None)
        out.append({
            "family": name,
            "display_name": publish_names.family_name(name),
            "member_slugs": [r["slug"] for r in group],
            "summary": describe.summary_markdown if describe else None,
            "paper_url": paper_url(cited[0]) if cited else None,
            "paper_title": cited[1].title if cited else None,
        })
    return out


# --- Blobs the store does not hold ----------------------------------------------------------

MISSING_BLOB_FIELDS = ("stable_id", "slug", "variant", "kind", "sha256", "action")
# The blobs a record's page shows from the store: the diagram, its thumbnail and the PDF.
_VARIANT_BLOBS = ("svg", "thumb", "pdf")


def drop_missing_blobs(records: Sequence[dict[str, Any]], present: Callable[[str], bool],
                       ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return ``(kept records, missing-blob rows)``.

    A record whose default variant's SVG is missing is dropped (``action`` ``dropped_record``,
    on every row of that record). Any other missing blob keeps its record (``kept``): the site
    hides a file that is not there. Each blob is checked once."""

    known: dict[str, bool] = {}

    def here(sha: str) -> bool:
        if sha not in known:
            known[sha] = present(sha)
        return known[sha]

    kept, rows = [], []
    for record in records:
        found = [(variant["key"], kind, variant[kind]["sha256"])
                 for variant in record["variants"] for kind in _VARIANT_BLOBS
                 if variant.get(kind) and not here(variant[kind]["sha256"])]
        dropped = any(key == record["default_variant"] and kind == "svg"
                      for key, kind, _ in found)
        rows += [{"stable_id": record["stable_id"], "slug": record["slug"], "variant": key,
                  "kind": kind, "sha256": sha,
                  "action": "dropped_record" if dropped else "kept"}
                 for key, kind, sha in found]
        if not dropped:
            kept.append(record)
    return kept, rows
