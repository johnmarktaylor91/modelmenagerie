"""Atlas-prose store, schema, and facts-traceability gate.

Every menagerie model gets a reader-facing prose summary derived from verified
facts only. This module owns the durable side of that lane:

- the per-model store files under ``menagerie/data/atlas_prose/<stable_id>.json``
  (schema ``menagerie.atlas-prose.v1``, append-only entries, newest supersedes);
- structural row validation (required provenance fields, length bounds);
- the traceability gate that binds each entry to the exact record revision it
  describes and refuses any fact path or evidence id the record cannot support.

Two source kinds are recognized. ``crawler-record`` entries derive from a
metadata-verified crawler model record (accuracy gate ``accurate`` and authored
metadata ``accepted``); their evidence ids resolve against the record's own
captured excerpts. ``catalog-row`` entries derive from a trace-verified master
catalog row (classics); the row itself is the record, it carries no excerpt
store, and the entry's ``record_revision`` is the canonical digest of the row.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA_FORMAT = "menagerie.atlas-prose.v1"
SOURCE_KINDS = ("crawler-record", "catalog-row")

MIN_SUMMARY_CHARS = 250
MAX_SUMMARY_CHARS = 4000
MIN_SUMMARY_PARAGRAPHS = 2
MAX_SUMMARY_PARAGRAPHS = 4

#: Explicit trace-verification stamps a catalog row's notes may carry. A row
#: without one of these is not eligible for prose (the atlas only narrates
#: models whose capture actually ran).
TRACE_VERIFIED_MARKERS = (
    "verified instantiate+trace",
    "trace+draw verified",
    "trace-verified",
    "trace verified",
)

GATE_CODES = frozenset(
    {
        "schema-invalid",
        "length-out-of-bounds",
        "paragraphs-out-of-bounds",
        "facts-path-unresolved",
        "evidence-id-unresolved",
        "record-not-verified",
        "row-not-trace-verified",
        "revision-mismatch",
        "stable-id-mismatch",
        "store-malformed",
    }
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CREATED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_REQUIRED_STR_FIELDS = (
    "stable_id",
    "summary_markdown",
    "author_model",
    "prompt_sha256",
    "record_revision",
    "created_at",
    "source_kind",
)


class AtlasProseGateError(ValueError):
    """Typed gate failure with a closed-vocabulary code.

    Parameters
    ----------
    code:
        One of :data:`GATE_CODES`.
    detail:
        Human-readable failure description.
    """

    def __init__(self, code: str, detail: str) -> None:
        if code not in GATE_CODES:
            raise ValueError(f"unknown atlas-prose gate code: {code}")
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def canonical_catalog_revision(row: Mapping[str, Any]) -> str:
    """Compute the canonical revision digest of a catalog row.

    Parameters
    ----------
    row:
        Master-catalog row object.

    Returns
    -------
    str
        ``sha256:`` digest of the canonical (sorted-keys, compact) JSON row.
    """

    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def crawler_record_is_verified(record: Mapping[str, Any]) -> bool:
    """Report whether a crawler model record is metadata-verified.

    Parameters
    ----------
    record:
        Crawler model record.

    Returns
    -------
    bool
        True when the accuracy gate verdict is ``accurate`` and the authored
        metadata state is ``accepted``.
    """

    gate = record.get("accuracy_gate")
    verdict = gate.get("verdict") if isinstance(gate, Mapping) else None
    return verdict == "accurate" and record.get("authored_metadata_state") == "accepted"


def catalog_row_is_trace_verified(row: Mapping[str, Any]) -> bool:
    """Report whether a catalog row carries an explicit trace-verification stamp.

    Parameters
    ----------
    row:
        Master-catalog row object.

    Returns
    -------
    bool
        True when the row's notes contain one of :data:`TRACE_VERIFIED_MARKERS`.
    """

    notes = row.get("notes")
    if not isinstance(notes, str):
        return False
    lowered = notes.lower()
    return any(marker in lowered for marker in TRACE_VERIFIED_MARKERS)


def _paragraphs(summary: str) -> list[str]:
    """Split a summary into blank-line-separated paragraphs.

    Parameters
    ----------
    summary:
        Markdown summary text.

    Returns
    -------
    list[str]
        Non-empty paragraph blocks.
    """

    return [block for block in re.split(r"\n\s*\n", summary.strip()) if block.strip()]


def validate_row_schema(row: Mapping[str, Any]) -> None:
    """Validate one prose row structurally.

    Parameters
    ----------
    row:
        Candidate atlas-prose row.

    Raises
    ------
    AtlasProseGateError
        ``schema-invalid`` for structural problems, ``length-out-of-bounds`` /
        ``paragraphs-out-of-bounds`` for summaries outside the editorial band.
    """

    if not isinstance(row, Mapping):
        raise AtlasProseGateError("schema-invalid", "row is not an object")
    for field in _REQUIRED_STR_FIELDS:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AtlasProseGateError("schema-invalid", f"missing or empty field {field!r}")
    if row["source_kind"] not in SOURCE_KINDS:
        raise AtlasProseGateError(
            "schema-invalid", f"source_kind {row['source_kind']!r} not in {SOURCE_KINDS}"
        )
    if not _SHA256_RE.match(row["prompt_sha256"]):
        raise AtlasProseGateError("schema-invalid", "prompt_sha256 is not a sha256: hex literal")
    if not _SHA256_RE.match(row["record_revision"]):
        raise AtlasProseGateError("schema-invalid", "record_revision is not a sha256: hex literal")
    if not _CREATED_AT_RE.match(row["created_at"]):
        raise AtlasProseGateError("schema-invalid", "created_at is not an ISO-8601 UTC Z timestamp")

    facts = row.get("facts_used")
    if (
        not isinstance(facts, list)
        or not facts
        or not all(isinstance(path, str) and path for path in facts)
    ):
        raise AtlasProseGateError("schema-invalid", "facts_used must be a non-empty list of paths")
    evidence = row.get("evidence_ids")
    if not isinstance(evidence, list) or not all(
        isinstance(evidence_id, str) and evidence_id for evidence_id in evidence
    ):
        raise AtlasProseGateError("schema-invalid", "evidence_ids must be a list of ids")
    if row["source_kind"] == "catalog-row":
        source_name = row.get("source_name")
        if not isinstance(source_name, str) or not source_name.strip():
            raise AtlasProseGateError("schema-invalid", "catalog-row entries require a source_name")

    summary = row["summary_markdown"]
    if not MIN_SUMMARY_CHARS <= len(summary) <= MAX_SUMMARY_CHARS:
        raise AtlasProseGateError(
            "length-out-of-bounds",
            f"summary is {len(summary)} chars; bounds are "
            f"[{MIN_SUMMARY_CHARS}, {MAX_SUMMARY_CHARS}]",
        )
    paragraph_count = len(_paragraphs(summary))
    if not MIN_SUMMARY_PARAGRAPHS <= paragraph_count <= MAX_SUMMARY_PARAGRAPHS:
        raise AtlasProseGateError(
            "paragraphs-out-of-bounds",
            f"summary has {paragraph_count} paragraphs; bounds are "
            f"[{MIN_SUMMARY_PARAGRAPHS}, {MAX_SUMMARY_PARAGRAPHS}]",
        )


def resolve_fact_path(record: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted fact path against a record.

    Parameters
    ----------
    record:
        Source record object.
    path:
        Dot-separated path; integer segments index into lists.

    Returns
    -------
    Any
        The resolved, non-null value.

    Raises
    ------
    AtlasProseGateError
        ``facts-path-unresolved`` when any segment fails to resolve or the
        final value is null.
    """

    value: Any = record
    for segment in path.split("."):
        if isinstance(value, Mapping) and segment in value:
            value = value[segment]
        elif isinstance(value, list) and segment.isdigit() and int(segment) < len(value):
            value = value[int(segment)]
        else:
            raise AtlasProseGateError(
                "facts-path-unresolved", f"fact path {path!r} does not resolve in the record"
            )
    if value is None:
        raise AtlasProseGateError(
            "facts-path-unresolved", f"fact path {path!r} resolves to null; a null is not a fact"
        )
    return value


def _record_evidence_ids(record: Mapping[str, Any]) -> set[str]:
    """Collect the evidence excerpt ids a crawler record carries.

    Parameters
    ----------
    record:
        Crawler model record.

    Returns
    -------
    set[str]
        Declared excerpt evidence ids.
    """

    evidence = record.get("evidence")
    excerpts = evidence.get("excerpts") if isinstance(evidence, Mapping) else None
    if not isinstance(excerpts, list):
        return set()
    return {
        excerpt["evidence_id"]
        for excerpt in excerpts
        if isinstance(excerpt, Mapping) and isinstance(excerpt.get("evidence_id"), str)
    }


def validate_entry_against_record(row: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """Run the full traceability gate for one entry against its source record.

    Parameters
    ----------
    row:
        Candidate atlas-prose row.
    record:
        The source it claims to describe: a crawler model record for
        ``crawler-record`` rows, a master-catalog row for ``catalog-row`` rows.

    Raises
    ------
    AtlasProseGateError
        With the specific closed-vocabulary code for the first failure found.
    """

    validate_row_schema(row)
    if row["source_kind"] == "crawler-record":
        if record.get("stable_id") != row["stable_id"]:
            raise AtlasProseGateError(
                "stable-id-mismatch",
                f"row describes {row['stable_id']!r} but the record is {record.get('stable_id')!r}",
            )
        if not crawler_record_is_verified(record):
            raise AtlasProseGateError(
                "record-not-verified",
                "prose derives from verified facts only; this record's metadata "
                "is not gate-accepted",
            )
        if row["record_revision"] != record.get("record_revision"):
            raise AtlasProseGateError(
                "revision-mismatch",
                "row binds a different record revision than the record presents",
            )
        known_evidence = _record_evidence_ids(record)
        for evidence_id in row["evidence_ids"]:
            if evidence_id not in known_evidence:
                raise AtlasProseGateError(
                    "evidence-id-unresolved",
                    f"evidence id {evidence_id!r} is not among the record's excerpts",
                )
    else:
        if record.get("name") != row["source_name"]:
            raise AtlasProseGateError(
                "stable-id-mismatch",
                f"row names {row['source_name']!r} but the catalog row is {record.get('name')!r}",
            )
        if not catalog_row_is_trace_verified(record):
            raise AtlasProseGateError(
                "row-not-trace-verified",
                "catalog row carries no trace-verification stamp in its notes",
            )
        if row["record_revision"] != canonical_catalog_revision(record):
            raise AtlasProseGateError(
                "revision-mismatch",
                "row binds a different canonical catalog-row digest",
            )
        if row["evidence_ids"]:
            raise AtlasProseGateError(
                "evidence-id-unresolved",
                "catalog rows carry no excerpt store; evidence_ids must be empty",
            )
    for path in row["facts_used"]:
        resolve_fact_path(record, path)


def store_path(store_dir: Path | str, stable_id: str) -> Path:
    """Compute the store file path for one model.

    Parameters
    ----------
    store_dir:
        Atlas-prose store directory.
    stable_id:
        Model stable id.

    Returns
    -------
    Path
        ``<store_dir>/<stable_id>.json``.
    """

    return Path(store_dir) / f"{stable_id}.json"


def load_store_file(path: Path | str) -> dict[str, Any]:
    """Load and structurally check one store file.

    Parameters
    ----------
    path:
        Store file path.

    Returns
    -------
    dict
        Parsed store object with ``format``, ``stable_id``, and ``entries``.

    Raises
    ------
    AtlasProseGateError
        ``store-malformed`` for unparsable or foreign-format files.
    """

    try:
        raw = Path(path).read_text(encoding="utf-8")
        store = json.loads(raw)
    except (OSError, json.JSONDecodeError) as err:
        raise AtlasProseGateError("store-malformed", f"cannot parse {path}: {err}") from err
    if (
        not isinstance(store, dict)
        or store.get("format") != SCHEMA_FORMAT
        or not isinstance(store.get("stable_id"), str)
        or not isinstance(store.get("entries"), list)
    ):
        raise AtlasProseGateError("store-malformed", f"{path} is not a {SCHEMA_FORMAT} store file")
    return store


def current_entry(store: Mapping[str, Any]) -> dict[str, Any]:
    """Return the current (newest) entry of a store object.

    Parameters
    ----------
    store:
        Parsed store object.

    Returns
    -------
    dict
        The last entry; newest supersedes.

    Raises
    ------
    AtlasProseGateError
        ``store-malformed`` when the store holds no entries.
    """

    entries = store["entries"]
    if not entries:
        raise AtlasProseGateError("store-malformed", "store file holds no entries")
    return entries[-1]


def append_entry(store_dir: Path | str, row: Mapping[str, Any]) -> Path:
    """Append one validated row to its model's store file.

    Prior entries are preserved verbatim; the write is atomic (temp file plus
    rename). The row is schema-validated before anything touches disk.

    Parameters
    ----------
    store_dir:
        Atlas-prose store directory (created if absent).
    row:
        Schema-valid atlas-prose row.

    Returns
    -------
    Path
        The store file written.
    """

    validate_row_schema(row)
    directory = Path(store_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = store_path(directory, row["stable_id"])
    if path.exists():
        store = load_store_file(path)
        if store["stable_id"] != row["stable_id"]:
            raise AtlasProseGateError(
                "store-malformed",
                f"{path} declares stable_id {store['stable_id']!r}",
            )
    else:
        store = {"format": SCHEMA_FORMAT, "stable_id": row["stable_id"], "entries": []}
    store["entries"].append(dict(row))
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(store, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    return path
