"""The ONE terminal/prerequisite vocabulary seam for machine-classified release (W14 base).

This module is the single owner of three closed vocabularies that were previously
scattered claims (the W14/W38 join defect): the machine-release CLASS set, the
prerequisite TOKEN grammar, and the blocked-prerequisite REASON grammar -- plus the
typed evidence contracts the per-class proof rules consume. Consumers:

* :mod:`menagerie.crawler.infra_release` -- the class engine
  (:func:`~menagerie.crawler.infra_release.derive_machine_class_release`) binds
  releases to these classes and evidence shapes.
* :mod:`menagerie.crawler.status` -- the per-class segregated census arms
  (``machine-released:<class>``) speak exactly :data:`MACHINE_RELEASE_CLASSES`.
* The quota deferral valve (W1, :mod:`menagerie.crawler.quota_deferral`) mints the
  valve rows the ``quota-deferral`` class releases.
* The blocked-prerequisite terminal routing (AL-6, ``driver.py``) mints the typed
  :data:`BLOCKED_PREREQUISITE_RECORD_SCHEMA_VERSION` seam rows -- the FIRST
  machine-readable prerequisite naming in the record system (every 200a
  prerequisite name lives only in redacted externally-controlled text, which
  doctrine part 3 forbids reading; era rule below).
* The environment lane (EN-6) emits :data:`PREREQUISITE_PROVISION_SCHEMA_VERSION`
  witnesses from its rotation/coverage aggregation, speaking this token grammar.
* The artifact lane (W9/AT-14) emits cache-invalidation fault events matching
  :func:`validate_cache_invalidation_fault_event`.

SIGNATURE RECORD: all four release classes and the AL-6 terminal-vocabulary change
are JMT-signed C.8 four-part arm extensions (recorded 2026-08-24, sprint STATE
"JMT APPROVALS RECORDED" rows 1 and 4; dossier v1 sections 1.2/1.4). A signature
authorizes an arm; it never substitutes for the arm's four-part proof, which rides
the class engine in :mod:`menagerie.crawler.infra_release`.

ERA RULE (uniform for every family here): historical rows are never rewritten and
never retro-qualify. A 200a blocked-prerequisite terminal has no typed seam row,
so the ``blocked-prerequisite-post-rotation`` class refuses it and it stays on the
human-review lane -- the deliberate resolution of the class-audit width question
(fix-provided prerequisites have no closed machine witness; rotation-provided ones
gain one only from the moment this schema exists). New mints REQUIRE the typed
rows; absence fails closed.
"""

from __future__ import annotations

import fcntl
import os
import re
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Optional

from menagerie.crawler.constants import BLOCKED_ARM_STAGES
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    row_content_address,
    stamp_row_digest,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import read_with_tail_recovery

__all__ = [
    "BLOCKED_PREREQUISITE_RECORD_RELATIVE_PATH",
    "BLOCKED_PREREQUISITE_RECORD_SCHEMA_VERSION",
    "BLOCKED_PREREQUISITE_REASON_PREFIX",
    "BLOCKED_PREREQUISITE_STAGES",
    "CACHE_INVALIDATION_FAULT_EVENT_KIND",
    "CACHE_INVALIDATION_FAULT_EXCEPTION_TYPES",
    "CACHE_INVALIDATION_FAULT_REASONS",
    "MACHINE_RELEASE_CLASSES",
    "MACHINE_RELEASE_CLASS_BLOCKED_PREREQUISITE",
    "MACHINE_RELEASE_CLASS_CORPUS_CONFLICT",
    "MACHINE_RELEASE_CLASS_ENGINE_CACHE",
    "MACHINE_RELEASE_CLASS_QUOTA_DEFERRAL",
    "PLATFORM_PREREQUISITE_TOKENS",
    "PREREQUISITE_PROVISION_SCHEMA_VERSION",
    "PREREQUISITE_TOKEN_RUNTIME_DEPENDENCY_PREFIX",
    "PrerequisiteTokenError",
    "ReleaseVocabularyError",
    "SIGNED_MACHINE_RELEASE_CLASSES",
    "append_blocked_prerequisite_record",
    "blocked_prerequisite_reason",
    "blocked_prerequisite_records",
    "blocked_prerequisite_record_path",
    "build_blocked_prerequisite_record",
    "is_blocked_prerequisite_reason",
    "normalize_pep503",
    "parse_blocked_prerequisite_reason",
    "parse_prerequisite_token",
    "prerequisite_token_from_author_id",
    "validate_blocked_prerequisite_record",
    "validate_cache_invalidation_fault_event",
    "validate_prerequisite_provision_witness",
]


class ReleaseVocabularyError(ValueError):
    """A seam-family row or evidence record violates its closed contract."""


class PrerequisiteTokenError(ReleaseVocabularyError):
    """A prerequisite token falls outside the closed grammar."""


# --------------------------------------------------------------------------- #
# Machine-release classes (the closed class set the signed C.8 arms extend)
# --------------------------------------------------------------------------- #

MACHINE_RELEASE_CLASS_QUOTA_DEFERRAL = "quota-deferral"
MACHINE_RELEASE_CLASS_CORPUS_CONFLICT = "corpus-conflict"
MACHINE_RELEASE_CLASS_BLOCKED_PREREQUISITE = "blocked-prerequisite-post-rotation"
MACHINE_RELEASE_CLASS_ENGINE_CACHE = "engine-invalidated-artifact-cache"

#: The closed machine-release class vocabulary. Adding a member is a NEW C.8
#: four-part arm extension: it needs its own JMT signature and its own per-class
#: four-part proof in one reviewed change, never an edit here alone.
MACHINE_RELEASE_CLASSES = frozenset(
    {
        MACHINE_RELEASE_CLASS_QUOTA_DEFERRAL,
        MACHINE_RELEASE_CLASS_CORPUS_CONFLICT,
        MACHINE_RELEASE_CLASS_BLOCKED_PREREQUISITE,
        MACHINE_RELEASE_CLASS_ENGINE_CACHE,
    }
)

#: The exact signed included-class set this change binds (JMT, 2026-08-24, direct
#: in-session word; sprint STATE "JMT APPROVALS RECORDED" row 1: "W14 all four
#: classes"). The Wave-2 merge receipt binds THIS set. A class removed from a
#: future signature record shrinks this set in its own reviewed change; the class
#: engine refuses to derive a release for any class outside it.
SIGNED_MACHINE_RELEASE_CLASSES = frozenset(MACHINE_RELEASE_CLASSES)


# --------------------------------------------------------------------------- #
# Prerequisite token grammar (EN-6 seam)
# --------------------------------------------------------------------------- #

PREREQUISITE_TOKEN_RUNTIME_DEPENDENCY_PREFIX = "runtime-dependency:"

#: Closed platform-capability prerequisite tokens. These route to the existing
#: capability deferral valves (``deferred:needs-cuda`` / ``deferred:needs-x86``),
#: never to a ``failed:`` terminal; they are members of the grammar so the ONE
#: parser adjudicates every prerequisite spelling.
PLATFORM_PREREQUISITE_TOKENS = frozenset({"platform:needs-cuda", "platform:needs-x86"})

_PEP503_CANONICAL_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def normalize_pep503(name: str) -> str:
    """Return the PEP 503 canonical form of one distribution name.

    Parameters
    ----------
    name:
        Raw distribution name.

    Returns
    -------
    str
        Lower-cased name with every run of ``-``, ``_``, ``.`` collapsed to one
        ``-`` (the PyPI simple-index rule).
    """

    return re.sub(r"[-_.]+", "-", name).lower()


def prerequisite_token_from_author_id(prerequisite_id: str) -> Optional[str]:
    """Translate one author-grammar prerequisite ID into a record-grammar token.

    The author's schema-validated ``prerequisite_ids`` cannot spell the record
    token grammar (``:`` is outside the id pattern), so the driver's routing
    site translates at mint time. The translation is closed: exactly the
    runtime-dependency spellings the AL-6 classifier recognizes map to
    ``runtime-dependency:<pep503>``; every other id returns ``None`` and the
    seam record is simply not minted (fail-closed toward the human lane --
    an untranslatable prerequisite is not post-rotation-releasable evidence).

    Parameters
    ----------
    prerequisite_id:
        One schema-validated blocked-arm prerequisite ID.

    Returns
    -------
    str | None
        Grammar-valid token, or ``None`` for an untranslatable id (including
        the bare ``runtime-dependency`` class marker, which names no
        distribution).
    """

    name: Optional[str] = None
    if prerequisite_id.startswith("runtime-dependency-"):
        name = prerequisite_id[len("runtime-dependency-") :]
    elif prerequisite_id.endswith("-distribution"):
        name = prerequisite_id[: -len("-distribution")]
    elif prerequisite_id.endswith("-package"):
        name = prerequisite_id[: -len("-package")]
    if not name:
        return None
    token = PREREQUISITE_TOKEN_RUNTIME_DEPENDENCY_PREFIX + normalize_pep503(name)
    try:
        parse_prerequisite_token(token)
    except PrerequisiteTokenError:
        return None
    return token


def parse_prerequisite_token(token: Any) -> tuple[str, str]:
    """Parse one prerequisite token against the closed grammar.

    The runtime-dependency payload must ALREADY be PEP 503 canonical: accepting
    aliases here would let two spellings of one distribution defeat set
    comparisons between claimed and provided prerequisites, which is exactly the
    laundering surface a closed grammar exists to remove.

    Parameters
    ----------
    token:
        Candidate token string.

    Returns
    -------
    tuple[str, str]
        ``("runtime-dependency", <pep503-name>)`` or ``("platform", <token>)``.

    Raises
    ------
    PrerequisiteTokenError
        On any spelling outside the closed grammar.
    """

    if not isinstance(token, str) or not token:
        raise PrerequisiteTokenError(f"prerequisite token must be a non-empty string, got {token!r}")
    if token in PLATFORM_PREREQUISITE_TOKENS:
        return ("platform", token)
    if token.startswith(PREREQUISITE_TOKEN_RUNTIME_DEPENDENCY_PREFIX):
        payload = token[len(PREREQUISITE_TOKEN_RUNTIME_DEPENDENCY_PREFIX) :]
        if not payload or not _PEP503_CANONICAL_RE.fullmatch(payload):
            raise PrerequisiteTokenError(
                f"runtime-dependency token payload {payload!r} is not PEP 503 canonical"
            )
        return ("runtime-dependency", payload)
    raise PrerequisiteTokenError(f"prerequisite token {token!r} is outside the closed grammar")


# --------------------------------------------------------------------------- #
# Blocked-prerequisite reason grammar (AL-6 seam)
# --------------------------------------------------------------------------- #

BLOCKED_PREREQUISITE_REASON_PREFIX = "blocked-prerequisite:"

#: Closed stages a blocked-prerequisite reason may name: exactly the blocked-arm
#: stages of the author-result schema (``blocked_payload.stage`` in v3/v4), the
#: ONE authority :data:`~menagerie.crawler.constants.BLOCKED_ARM_STAGES` also
#: pins into ``FAILURE_REASON_CODES`` and the attempt schema (DX-15 cross-pin).
#: ``environment-adjacent`` is deliberately NOT a member: it is AL-6's routing
#: STATUS stage (``failed:environment-adjacent``), never a reason suffix -- the
#: suffix always preserves the stage the AUTHOR named, which is what keeps the
#: ``(status_code, reason_code)`` pair injective back to the blocking stage.
BLOCKED_PREREQUISITE_STAGES = frozenset(BLOCKED_ARM_STAGES)


def blocked_prerequisite_reason(stage: str) -> str:
    """Return the stage-suffixed closed reason for one blocked prerequisite.

    Parameters
    ----------
    stage:
        Blocking stage; must be a :data:`BLOCKED_PREREQUISITE_STAGES` member.

    Raises
    ------
    ReleaseVocabularyError
        On a stage outside the closed set.
    """

    if stage not in BLOCKED_PREREQUISITE_STAGES:
        raise ReleaseVocabularyError(
            f"blocked-prerequisite stage {stage!r} is outside the closed stage set"
        )
    return f"{BLOCKED_PREREQUISITE_REASON_PREFIX}{stage}"


def parse_blocked_prerequisite_reason(reason: Any) -> Optional[str]:
    """Return the stage of one blocked-prerequisite reason, or ``None`` for legacy.

    Both eras parse: the bare legacy ``blocked-prerequisite`` (200a rows, stage
    unknown -> ``None``) and the stage-suffixed new-era spelling. A non-member
    raises; callers asking "is this the family at all" use
    :func:`is_blocked_prerequisite_reason` instead.

    Raises
    ------
    ReleaseVocabularyError
        When ``reason`` is not a blocked-prerequisite family member.
    """

    if reason == "blocked-prerequisite":
        return None
    if isinstance(reason, str) and reason.startswith(BLOCKED_PREREQUISITE_REASON_PREFIX):
        stage = reason[len(BLOCKED_PREREQUISITE_REASON_PREFIX) :]
        if stage in BLOCKED_PREREQUISITE_STAGES:
            return stage
        raise ReleaseVocabularyError(
            f"blocked-prerequisite reason names unknown stage {stage!r}"
        )
    raise ReleaseVocabularyError(f"{reason!r} is not a blocked-prerequisite reason")


def is_blocked_prerequisite_reason(reason: Any) -> bool:
    """Return whether one reason code is a blocked-prerequisite family member."""

    try:
        parse_blocked_prerequisite_reason(reason)
    except ReleaseVocabularyError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Blocked-prerequisite seam records (minted by the driver's BLOCKED routing)
# --------------------------------------------------------------------------- #

BLOCKED_PREREQUISITE_RECORD_SCHEMA_VERSION = "menagerie.crawler.blocked-prerequisite.v1"
BLOCKED_PREREQUISITE_RECORD_RELATIVE_PATH = Path("intake-extensions") / "blocked-prerequisites.jsonl"


def blocked_prerequisite_record_path(records_root: Path) -> Path:
    """Return the canonical seam-record ledger path under one records root."""

    return records_root / BLOCKED_PREREQUISITE_RECORD_RELATIVE_PATH


def build_blocked_prerequisite_record(
    *,
    stable_id: str,
    work_id: str,
    stage: str,
    prerequisite_tokens: tuple[str, ...] | list[str],
    gate_id: str,
    author_result_sha256: str,
    campaign_id: str,
    run_id: str,
    machine_id: str,
    created_at: str,
) -> JsonObject:
    """Build the typed seam record for one checked-accurate BLOCKED terminal.

    This row is DRIVER-written at the terminal routing site, after the checker
    adjudicated the BLOCKED recommendation accurate: the token spellings are
    grammar-validated here, and the row binds the exact adjudicating gate and
    author result, so the release rule later re-derives everything from
    parent-attested records without reading any producer text.

    Raises
    ------
    ReleaseVocabularyError
        On an empty identity field or a non-adjudicated binding.
    PrerequisiteTokenError
        On any token outside the closed grammar.
    """

    for field_name, value in (
        ("stable_id", stable_id),
        ("work_id", work_id),
        ("gate_id", gate_id),
        ("author_result_sha256", author_result_sha256),
        ("campaign_id", campaign_id),
        ("run_id", run_id),
        ("machine_id", machine_id),
        ("created_at", created_at),
    ):
        if not isinstance(value, str) or not value or value == str(None):
            raise ReleaseVocabularyError(
                f"blocked-prerequisite record requires a real {field_name}, got {value!r}"
            )
    tokens = tuple(prerequisite_tokens)
    if not tokens:
        raise ReleaseVocabularyError(
            "a blocked-prerequisite record names at least one prerequisite token"
        )
    if len(set(tokens)) != len(tokens):
        raise ReleaseVocabularyError("blocked-prerequisite tokens must be distinct")
    for token in tokens:
        parse_prerequisite_token(token)
    row: JsonObject = {
        "schema_version": BLOCKED_PREREQUISITE_RECORD_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "reason_code": blocked_prerequisite_reason(stage),
        "prerequisite_tokens": sorted(tokens),
        "gate_id": gate_id,
        "author_result_sha256": author_result_sha256,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": machine_id,
        "created_at": created_at,
    }
    return stamp_row_digest(row)


def validate_blocked_prerequisite_record(row: Mapping[str, Any]) -> JsonObject:
    """Validate one seam record and return its canonical copy.

    Raises
    ------
    ReleaseVocabularyError
        If a required field is missing, mistyped, outside a closed vocabulary,
        or the content address disagrees.
    """

    required = (
        "schema_version",
        "stable_id",
        "work_id",
        "reason_code",
        "prerequisite_tokens",
        "gate_id",
        "author_result_sha256",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ReleaseVocabularyError(
            f"blocked-prerequisite record is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != BLOCKED_PREREQUISITE_RECORD_SCHEMA_VERSION:
        raise ReleaseVocabularyError(
            f"blocked-prerequisite record declares unknown schema {row['schema_version']!r}"
        )
    for field in (
        "stable_id",
        "work_id",
        "gate_id",
        "author_result_sha256",
        "campaign_id",
        "run_id",
        "machine_id",
        "created_at",
    ):
        if not isinstance(row[field], str) or not row[field]:
            raise ReleaseVocabularyError(f"blocked-prerequisite record has an empty {field}")
    stage = parse_blocked_prerequisite_reason(row["reason_code"])
    if stage is None:
        raise ReleaseVocabularyError(
            "blocked-prerequisite record reason must be stage-suffixed (new-era rows only)"
        )
    tokens = row["prerequisite_tokens"]
    if (
        not isinstance(tokens, list)
        or not tokens
        or len(set(tokens)) != len(tokens)
        or tokens != sorted(tokens)
    ):
        raise ReleaseVocabularyError(
            "blocked-prerequisite record tokens must be a sorted, distinct, non-empty list"
        )
    for token in tokens:
        parse_prerequisite_token(token)
    validated = dict(row)
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise ReleaseVocabularyError(
            f"blocked-prerequisite record carries a foreign content-address field "
            f"{', '.join(foreign)}"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise ReleaseVocabularyError(
            "blocked-prerequisite record digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


def _seam_ledger_bytes(path: Path, handle: Optional[BinaryIO] = None) -> bytes:
    """Read the seam ledger, recovering only a torn final line with evidence."""

    return read_with_tail_recovery(
        path,
        prefix_check=lambda retained: _decode_seam_lines(retained.read_bytes(), path),
        locked_handle=handle,
    )


def _decode_seam_lines(payload: bytes, path: Path) -> list[JsonObject]:
    """Decode and validate every complete seam-record line."""

    import json as _json

    rows: list[JsonObject] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ReleaseVocabularyError(
            f"{path} is not a UTF-8 blocked-prerequisite ledger: {exc}"
        ) from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            candidate = _json.loads(line)
        except _json.JSONDecodeError as exc:
            raise ReleaseVocabularyError(
                f"{path}:{index} is not a decodable blocked-prerequisite record: {exc}"
            ) from exc
        rows.append(validate_blocked_prerequisite_record(candidate))
    return rows


def blocked_prerequisite_records(path: Path) -> tuple[JsonObject, ...]:
    """Return every validated seam record, empty when the ledger is absent."""

    if not path.exists():
        return ()
    return tuple(_decode_seam_lines(_seam_ledger_bytes(path), path))


def append_blocked_prerequisite_record(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one seam record durably and idempotently (first mint wins).

    A later run that re-derives the same ``(stable_id, work_id)`` routing is a
    no-op returning the persisted row, exactly like the sibling valve appenders:
    ``created_at``/``run_id`` necessarily differ on a resume and must not abort
    the driver.
    """

    validated = validate_blocked_prerequisite_record(row)
    key = (validated["stable_id"], validated["work_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = _decode_seam_lines(_seam_ledger_bytes(path, handle), path)
        matching = [item for item in existing if (item["stable_id"], item["work_id"]) == key]
        if matching:
            return matching[0]
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


# --------------------------------------------------------------------------- #
# Prerequisite provision witnesses (EN-6 emits; the release class consumes)
# --------------------------------------------------------------------------- #

PREREQUISITE_PROVISION_SCHEMA_VERSION = "menagerie.crawler.prerequisite-provision.v1"


def validate_prerequisite_provision_witness(row: Mapping[str, Any]) -> JsonObject:
    """Validate one rotation-provision witness against the closed contract.

    A witness asserts: an ATTESTED environment rotation now provides one exact
    prerequisite token. It is emitted by the environment lane's aggregation from
    its own rotation/coverage records (parent-attested; a producer session never
    writes one) and is the ONLY evidence the blocked-prerequisite-post-rotation
    class accepts for "the prerequisite is provided now". A code-fix narrative is
    deliberately NOT a witness shape: "fixed in commit X provides jsonschema" has
    no closed machine predicate, so those terminals stay on the human lane (the
    class-audit width decision, disclosed).

    Raises
    ------
    ReleaseVocabularyError
        On a field outside the closed contract.
    PrerequisiteTokenError
        On a token outside the grammar.
    """

    required = (
        "schema_version",
        "token",
        "provided_by_rotation",
        "environment_intent",
        "registered_at",
        "row_sha256",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ReleaseVocabularyError(
            f"prerequisite-provision witness is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != PREREQUISITE_PROVISION_SCHEMA_VERSION:
        raise ReleaseVocabularyError(
            f"prerequisite-provision witness declares unknown schema {row['schema_version']!r}"
        )
    parse_prerequisite_token(row["token"])
    for field in ("provided_by_rotation", "environment_intent", "registered_at"):
        if not isinstance(row[field], str) or not row[field]:
            raise ReleaseVocabularyError(f"prerequisite-provision witness has an empty {field}")
    validated = dict(row)
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise ReleaseVocabularyError(
            "prerequisite-provision witness digest does not match its payload"
        )
    validated["row_sha256"] = digest
    return validated


# --------------------------------------------------------------------------- #
# Cache-invalidation fault evidence (W9 emits; AT-14 class consumes)
# --------------------------------------------------------------------------- #

CACHE_INVALIDATION_FAULT_EVENT_KIND = "artifact-cache-invalidated"

#: Closed machine reasons an invalidation event may carry for the AT-14 class.
#: ``engine-fault`` is the ONLY member: the engine's own machinery destroyed or
#: corrupted a cache/artifact it had committed. Content-judgment invalidations
#: (supersession, discard, operator sweep) are deliberately outside this set.
CACHE_INVALIDATION_FAULT_REASONS = frozenset({"engine-fault"})

#: Closed TorchLens-internal exception identities that may witness an engine
#: fault: the disk-recovery machinery and the ledger conflict class -- both
#: crawler infrastructure by construction (the session's content never
#: participates in either raise site).
CACHE_INVALIDATION_FAULT_EXCEPTION_TYPES = frozenset(
    {
        "menagerie.crawler.env_lifecycle.DiskRecoveryError",
        "menagerie.crawler.recordio.LedgerConflictError",
    }
)


def validate_cache_invalidation_fault_event(event: Mapping[str, Any]) -> JsonObject:
    """Validate one engine-fault cache-invalidation event for the AT-14 class.

    The event is written by the artifact-transaction machinery (parent-attested),
    never by a session. Every field the class rule keys on is closed: the kind,
    the reason, the fault's typed exception identity, and the exact
    ``(stable_id, work_id)`` binding.

    Raises
    ------
    ReleaseVocabularyError
        On any field outside the closed contract.
    """

    if event.get("event_kind") != CACHE_INVALIDATION_FAULT_EVENT_KIND:
        raise ReleaseVocabularyError(
            f"cache-invalidation event kind {event.get('event_kind')!r} is not "
            f"{CACHE_INVALIDATION_FAULT_EVENT_KIND!r}"
        )
    details = event.get("details")
    if not isinstance(details, Mapping):
        raise ReleaseVocabularyError("cache-invalidation event carries no details mapping")
    for field in ("stable_id", "work_id", "cache_identity", "artifact_transaction_id"):
        if not isinstance(details.get(field), str) or not details[field]:
            raise ReleaseVocabularyError(f"cache-invalidation event has an empty details.{field}")
    if details.get("reason") not in CACHE_INVALIDATION_FAULT_REASONS:
        raise ReleaseVocabularyError(
            f"cache-invalidation reason {details.get('reason')!r} is outside the closed set"
        )
    if details.get("fault_exception_type") not in CACHE_INVALIDATION_FAULT_EXCEPTION_TYPES:
        raise ReleaseVocabularyError(
            f"cache-invalidation fault exception {details.get('fault_exception_type')!r} "
            "is outside the closed set"
        )
    return dict(event)
