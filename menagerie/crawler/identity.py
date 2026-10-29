"""Canonical identities and exact dependency-staleness propagation."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import SplitResult, urlsplit, urlunsplit

from menagerie.crawler.constants import STABLE_ID_DIGEST_CHARS
from menagerie.crawler.models import StalenessReport

_HASH_PREFIX = "sha256:"

#: Permanent scale-cost instrumentation at the UNBYPASSABLE hashing boundary.
#: Every canonical digest in the tree is computed by :func:`stable_hash`, so a
#: change that re-digests history inside a hot path grows this counter no matter
#: which module spells the loop -- including a re-digest that reads no bytes and
#: decodes no row, which every consumer-side counter is blind to (PSLRE5 P5-D-2
#: arm B2).  Counting is the only mutable module state here, it is write-only
#: from this module, and no decision anywhere reads it.
IDENTITY_STATS: dict[str, int] = {"stable_hashes": 0}


def reset_identity_stats() -> None:
    """Zero the canonical-hash counter before one measurement window."""

    for key in IDENTITY_STATS:
        IDENTITY_STATS[key] = 0


def utc_now() -> str:
    """Return the current RFC 3339 UTC timestamp.

    Returns
    -------
    str
        Current timestamp ending in ``Z``.
    """

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_sha256(value: str) -> bool:
    """Return whether a value is a canonical prefixed SHA-256 digest.

    Parameters
    ----------
    value:
        Candidate digest.

    Returns
    -------
    bool
        Whether the digest uses lowercase ``sha256:<64 hex>`` syntax.
    """

    return (
        len(value) == 71
        and value.startswith(_HASH_PREFIX)
        and all(character in "0123456789abcdef" for character in value[7:])
    )


DARWIN_F_FULLFSYNC = 51
"""Darwin ``F_FULLFSYNC`` command from ``<sys/fcntl.h>``."""


def durable_fsync(descriptor: int) -> None:
    """Flush one descriptor through the platform's power-loss barrier.

    THE one durability primitive for the whole crawler (BL-7, propagated to
    every ledger in r2fix). On Darwin ``os.fsync`` only evicts the write from
    the kernel page cache: the bytes may still sit in the drive's volatile
    cache, so a power loss can erase a row the campaign already ACKNOWLEDGED.
    ``F_FULLFSYNC`` is the only call that asks the device to flush that cache.

    It lives in this leaf module, not in :mod:`menagerie.crawler.recordio`,
    precisely so that every writer in the tree can reach it without an import
    cycle -- ``recordio`` re-exports it for its historical importers. The AST
    census in ``tests/test_durable_fsync_census.py`` holds the invariant that
    this function is the ONLY ``os.fsync`` call site in production code.

    Parameters
    ----------
    descriptor:
        Open file descriptor whose preceding writes must be durable before an
        acknowledgement may be returned.
    """

    if sys.platform == "darwin":
        fcntl.fcntl(descriptor, DARWIN_F_FULLFSYNC)
        return
    os.fsync(descriptor)


def fsync_directory(path: Path) -> None:
    """Synchronize a directory after changing one of its entries.

    Parameters
    ----------
    path:
        Directory to synchronize.
    """

    descriptor = os.open(path, os.O_RDONLY)
    try:
        durable_fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_replace_bytes(path: Path, data: bytes) -> None:
    """Replace a file with fsynced bytes and then fsync its parent directory.

    Parameters
    ----------
    path:
        Destination path.
    data:
        Exact bytes to persist.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        durable_fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value into stable UTF-8 bytes.

    Parameters
    ----------
    value:
        JSON-compatible value.

    Returns
    -------
    bytes
        Deterministic serialization with sorted object keys.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_bytes(data: bytes) -> str:
    """Return a prefixed SHA-256 digest for bytes.

    Parameters
    ----------
    data:
        Bytes to hash.

    Returns
    -------
    str
        ``sha256:`` followed by 64 lowercase hexadecimal characters.
    """

    return f"{_HASH_PREFIX}{hashlib.sha256(data).hexdigest()}"


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible value using canonical serialization.

    Parameters
    ----------
    value:
        JSON-compatible value.

    Returns
    -------
    str
        Canonical prefixed SHA-256 digest.
    """

    IDENTITY_STATS["stable_hashes"] += 1
    return hash_bytes(canonical_json_bytes(value))


#: The ONE exclusion set behind every chained ledger row's content address, and
#: behind the self-digest each writer stamps over its own body.  It holds the
#: position-bound ``chain`` envelope (attached after the body is stamped) and the
#: four self-referential digest names, so that stamping a digest into a row can
#: never move the row's address (:func:`stamp_row_digest` is idempotent under
#: :func:`row_content_address`).
#:
#: Membership is NEVER additive without re-deriving every persisted chain head: a
#: new name here is a new UNHASHED region on every Track A ledger at once.  The
#: contents are pinned by an anti-substitution test
#: (``tests/test_ledger_chain_content_address.py``).
ROW_CONTENT_ADDRESS_EXCLUDED = frozenset(
    {"chain", "close_sha256", "row_sha256", "record_revision", "payload_sha256"}
)

#: The names the recordio fact families' NARROWER exclusion deliberately does NOT
#: drop, spelled as the DIFFERENCE from the one shared set rather than by restating
#: a second literal (PSLRE6 D6-3: the narrower set WAS a second literal inside
#: :func:`payload_hash`, so nothing pinned the relationship between the two and they
#: agreed only while no chained record family happened to carry one of these names --
#: the drift risk this module's own docstring names, materialized).
#:
#: The two sets must differ, and that is a decision rather than an oversight:
#: :func:`payload_hash` also digests payloads that are never chain-verified
#: (partition proofs, driver receipts, checkpoints), some of which legitimately carry
#: a top-level REFERENCE spelled like a digest name, and widening the exclusion there
#: would silently move those digests.  So the difference is NAMED here, and the
#: coupling it implies is enforced at the writer:
#: :func:`stamp_payload_digest` refuses a body carrying one of these names instead of
#: letting the payload digest and the chain content address diverge silently.
PAYLOAD_HASH_TOLERATED_ADDRESS_FIELDS = frozenset({"chain", "close_sha256", "row_sha256"})

#: The recordio record families' exclusion, DERIVED from the one shared set.  Because
#: it is a derivation, a name added to :data:`ROW_CONTENT_ADDRESS_EXCLUDED` lands in
#: exactly one of the two halves by construction, and the halves cannot drift.
PAYLOAD_HASH_EXCLUDED = ROW_CONTENT_ADDRESS_EXCLUDED - PAYLOAD_HASH_TOLERATED_ADDRESS_FIELDS


def row_content_address(row: Mapping[str, Any]) -> str:
    """Return one ledger row's content address, DERIVED from its body.

    THE derivation the tamper-evident chain uses and every chained writer stamps
    against, in one place, so the exclusion set cannot drift between a writer and
    the chain (PSLRE5's flail class: the chain SELECTED its address from the first
    self-declared digest field a row happened to carry, so one extra field name
    hijacked the address of any body an attacker liked).  The address is a
    function of the body: a reader believes nothing the row says about itself.

    Parameters
    ----------
    row:
        Decoded ledger row, with or without its digest and chain envelope.

    Returns
    -------
    str
        Canonical prefixed SHA-256 digest of the row body.
    """

    return stable_hash(
        {
            str(key): value
            for key, value in row.items()
            if str(key) not in ROW_CONTENT_ADDRESS_EXCLUDED
        }
    )


def stamp_row_digest(body: Mapping[str, Any], *, field: str = "row_sha256") -> dict[str, Any]:
    """Return one row body carrying its own derived content address.

    The writer half of :func:`row_content_address`: a writer that cannot spell its
    own exclusion set cannot drift from the chain's.  ``field`` must itself be
    excluded from the address, so stamping is idempotent -- the stamped row and
    the bare body have the identical address.

    Parameters
    ----------
    body:
        Row body without its self-digest and without a chain envelope.
    field:
        The self-digest field name this family stamps.

    Returns
    -------
    dict[str, Any]
        The row with its derived content address stamped under ``field``.
    """

    if field not in ROW_CONTENT_ADDRESS_EXCLUDED:
        raise ValueError(
            f"a row self-digest field must be excluded from the content address: {field!r} "
            "is not, so stamping it would move the address it is supposed to name"
        )
    plain = {str(key): value for key, value in body.items()}
    return {**plain, field: row_content_address(plain)}


def foreign_content_address_fields(row: Mapping[str, Any], *, own: str) -> tuple[str, ...]:
    """Return every self-digest field a row declares that is not its own.

    A chained row family owns exactly ONE content-address field (``row_sha256``
    for money, leg, registry, valve and park rows; ``close_sha256`` for closes;
    the recordio pair for fact records).  A row carrying a SECOND one has no
    honest producer, and under the deleted preference ladder that extra field WAS
    the hijack tool (PSLRE5).  Refusing it in each family's own vocabulary retires
    the tool as well as the mechanism.

    Membership is by KEY PRESENCE, not by non-null value (PSLRE6 D6-5): a
    ``"close_sha256": null`` was skipped as "absent" and was therefore unhashed
    padding no authority looked at, so a present key with no address is as foreign as
    a present key with one.

    Parameters
    ----------
    row:
        Decoded ledger row.
    own:
        The one content-address field this family stamps.

    Returns
    -------
    tuple[str, ...]
        Sorted foreign content-address field names, empty when there are none.
    """

    return tuple(
        sorted(
            name
            for name in ROW_CONTENT_ADDRESS_EXCLUDED
            if name not in {"chain", own} and name in row
        )
    )


def stable_leaf_hash(value: Any) -> str:
    """Hash one observed payload leaf, tolerating non-JSON-serializable objects.

    Graph inputs (DGLGraph, PyG Data, arbitrary adapter-constructed objects)
    reach the worker's input-signature builder as ``python``-kind leaves that
    canonical JSON cannot serialize; hashing them killed every graph-model
    receipt AFTER the forward itself succeeded (m538, step-100 B2). A leaf
    canonical JSON accepts keeps its exact :func:`stable_hash` digest; any
    other leaf (``TypeError``) and any non-finite float (``ValueError`` under
    ``allow_nan=False``) gets a TYPE-TAGGED digest over the object's
    module-qualified type name.

    Determinism bounds: the fallback digest is deterministic across replays,
    processes, and hosts for the same leaf TYPE, but it deliberately does NOT
    bind the object's CONTENTS -- two structurally different graphs of the
    same type share one digest. That is sufficient for its only consumer, the
    per-mode receipt input signature, which needs a total, reproducible
    description, not content addressing. Contract-literal comparisons
    (``metadata.input_signature_matches_contract``) hash author-declared JSON
    literals with plain ``stable_hash``, so both sides keep byte-identical
    semantics for every value that could match before.

    Parameters
    ----------
    value:
        Observed leaf of any Python type.

    Returns
    -------
    str
        Canonical prefixed SHA-256 digest.
    """

    try:
        return stable_hash(value)
    except (TypeError, ValueError):
        return stable_hash(
            {
                "non_json_leaf_type_v1": {
                    "type_module": type(value).__module__,
                    "type_qualname": type(value).__qualname__,
                }
            }
        )


def payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash a ledger payload without its self-referential hash fields.

    The recordio record families' own convention, kept deliberately NARROWER than
    :data:`ROW_CONTENT_ADDRESS_EXCLUDED`: this function also hashes payloads that
    are never chain-verified (partition proofs, driver receipts, checkpoints), some
    of which legitimately carry a top-level REFERENCE spelled like a digest name, and
    widening the exclusion here would silently move those digests.

    The exclusion is :data:`PAYLOAD_HASH_EXCLUDED` -- DERIVED from the one shared set
    rather than a second literal, so the narrower convention cannot drift away from
    the chain's (PSLRE6 D6-3).  For a chain-bearing record the two derivations agree
    exactly, because :func:`stamp_payload_digest` refuses to stamp a body carrying any
    :data:`PAYLOAD_HASH_TOLERATED_ADDRESS_FIELDS` name at all.

    Parameters
    ----------
    payload:
        Ledger payload.

    Returns
    -------
    str
        Canonical payload digest.
    """

    unhashed = {
        key: value for key, value in payload.items() if key not in PAYLOAD_HASH_EXCLUDED
    }
    return stable_hash(unhashed)


def unpinned_address_fields(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every top-level name that decouples a payload digest from the address.

    THE coupling predicate behind :data:`PAYLOAD_HASH_EXCLUDED`: a payload carrying
    one of these names hashes it into its :func:`payload_hash` preimage while
    :func:`row_content_address` does not, so the record's own stamp is not its chain
    content address and the chain's fused re-certification would refuse the row
    forever after it is persisted.  Naming the condition where the row is MINTED
    turns a permanent, unexplained chain refusal into a typed refusal at the writer.

    Parameters
    ----------
    payload:
        Record body, without its own payload digest.

    Returns
    -------
    tuple[str, ...]
        Sorted offending names, empty when the two derivations agree.
    """

    return tuple(
        sorted(PAYLOAD_HASH_TOLERATED_ADDRESS_FIELDS & {str(key) for key in payload})
    )


def stamp_payload_digest(
    payload: Mapping[str, Any], *, field: str
) -> dict[str, Any]:
    """Return one recordio record carrying its own payload digest.

    The recordio half of :func:`stamp_row_digest`: ONE helper both fact-ledger digest
    fields route through, so neither writer spells its own preimage and neither can
    mint a record whose payload digest is not also its chain content address.

    Parameters
    ----------
    payload:
        Record body without its own payload digest.
    field:
        ``record_revision`` or ``payload_sha256``.

    Returns
    -------
    dict[str, Any]
        The record with its payload digest stamped under ``field``.

    Raises
    ------
    ValueError
        If ``field`` is not a payload-digest name, or the body carries a name that
        would decouple its payload digest from its chain content address.
    """

    if field not in PAYLOAD_HASH_EXCLUDED:
        raise ValueError(
            f"a record payload digest field must be excluded from the payload hash: "
            f"{field!r} is not, so stamping it would move the digest it names"
        )
    unpinned = unpinned_address_fields(payload)
    if unpinned:
        raise ValueError(
            "a chain-bearing record may not carry the top-level content-address "
            f"name(s) {list(unpinned)}: the payload digest hashes them and the chain "
            "content address does not, so the record's own stamp would not be its "
            "chain address and every later chain read would refuse the row"
        )
    plain = {str(key): value for key, value in payload.items()}
    return {**plain, field: payload_hash(plain)}


def assign_stable_id(namespace: str, natural_key: str) -> tuple[str, str]:
    """Assign a deterministic stable ID and retain its full digest.

    Parameters
    ----------
    namespace:
        Immutable identity namespace.
    natural_key:
        Immutable normalized natural key.

    Returns
    -------
    tuple[str, str]
        ``m_`` ID using the first 20 base32 characters and full SHA-256 digest.
    """

    digest = hashlib.sha256(canonical_json_bytes([namespace, natural_key])).digest()
    base32 = base64.b32encode(digest).decode("ascii").lower().rstrip("=")
    return f"m_{base32[:STABLE_ID_DIGEST_CHARS]}", f"{_HASH_PREFIX}{digest.hex()}"


def normalize_url(url: str) -> str:
    """Normalize a source URL without changing path/query semantics.

    Parameters
    ----------
    url:
        Absolute public source URL.

    Returns
    -------
    str
        URL with lowercase scheme/host, no fragment, and normalized default port.
    """

    split = urlsplit(url.strip())
    hostname = (split.hostname or "").lower()
    port = split.port
    if port is not None and not (
        (split.scheme.lower() == "http" and port == 80)
        or (split.scheme.lower() == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    normalized = SplitResult(
        split.scheme.lower(),
        hostname,
        split.path or "/",
        split.query,
        "",
    )
    return urlunsplit(normalized)


def compute_source_identity(
    sources: Sequence[Mapping[str, Any]], search_report: Mapping[str, Any]
) -> str:
    """Compute source identity over normalized sources and search report.

    Parameters
    ----------
    sources:
        Source rows containing URLs, revisions, locators, and content hashes.
    search_report:
        Complete source-search report.

    Returns
    -------
    str
        Source identity.
    """

    normalized_sources = []
    for source in sources:
        selected = {
            "url": normalize_url(str(source["url"])),
            "revision_kind": source.get("revision_kind"),
            "revision": source.get("revision"),
            "locator": source.get("locator"),
            "content_sha256": source.get("content_sha256"),
        }
        normalized_sources.append(selected)
    normalized_sources.sort(key=lambda item: canonical_json_bytes(item))
    return stable_hash({"sources": normalized_sources, "search_report": search_report})


def compute_evidence_identity(excerpts: Sequence[Mapping[str, Any]]) -> str:
    """Compute ordered literal-evidence identity.

    Parameters
    ----------
    excerpts:
        Ordered excerpts with locators, supports, and content hashes.

    Returns
    -------
    str
        Evidence identity.
    """

    selected = [
        {
            "source_id": excerpt.get("source_id"),
            "locator": excerpt.get("locator"),
            "text": excerpt.get("text"),
            "text_sha256": excerpt.get("text_sha256"),
            "supports": excerpt.get("supports"),
            "source_content_sha256": excerpt.get("source_content_sha256"),
        }
        for excerpt in excerpts
    ]
    return stable_hash(selected)


def compute_recipe_revision(
    recipe: Mapping[str, Any], source_identity: str, *, adapter_bytes: Optional[bytes] = None
) -> str:
    """Compute a recipe revision bound to exact source identity.

    Parameters
    ----------
    recipe:
        Declarative constructor/adapter choices and policies.
    source_identity:
        Current source identity.
    adapter_bytes:
        Optional exact typed-adapter module bytes.

    Returns
    -------
    str
        Recipe revision.
    """

    return stable_hash(
        {
            "recipe": recipe,
            "adapter_sha256": hash_bytes(adapter_bytes) if adapter_bytes is not None else None,
            "source_identity": source_identity,
        }
    )


def compute_env_generation(
    intent: Mapping[str, Any],
    exact_lock: Any,
    resolved_export: Any,
    platform_facts: Mapping[str, Any],
    probe_results: Sequence[Mapping[str, Any]],
) -> str:
    """Compute an exact target environment generation.

    Parameters
    ----------
    intent:
        Environment intent.
    exact_lock:
        Exact lock data including artifact hashes.
    resolved_export:
        Actual resolved export.
    platform_facts:
        Compiler and SDK facts.
    probe_results:
        Ordered probe results.

    Returns
    -------
    str
        Environment generation.
    """

    return stable_hash(
        {
            "intent": intent,
            "exact_lock": exact_lock,
            "resolved_export": resolved_export,
            "platform_facts": platform_facts,
            "probe_results": probe_results,
        }
    )


def compute_fidelity_identity(
    *,
    source_identity: str,
    evidence_identity: str,
    implementation_hash: Optional[str],
    source_to_code_map: Sequence[Mapping[str, Any]],
    prompt_hash: str,
    checker_model: str,
    checker_version: str,
) -> str:
    """Compute a fidelity-gate identity.

    Parameters
    ----------
    source_identity, evidence_identity:
        Current source and evidence identities.
    implementation_hash:
        Exact accepted implementation/code hash.
    source_to_code_map:
        Material source-to-code mapping.
    prompt_hash:
        Frozen fidelity prompt hash.
    checker_model, checker_version:
        Exact checker identity.

    Returns
    -------
    str
        Fidelity identity.
    """

    return stable_hash(locals())


def compute_vet_identity(
    *,
    authored_metadata: Mapping[str, Any],
    evidence_references: Mapping[str, Any],
    source_identity: str,
    evidence_identity: str,
    prompt_hash: str,
    checker_model: str,
    checker_version: str,
) -> str:
    """Compute an authored-metadata vet identity.

    Parameters
    ----------
    authored_metadata:
        Every authored metadata field.
    evidence_references:
        Field-to-evidence references.
    source_identity, evidence_identity:
        Current source and evidence identities.
    prompt_hash:
        Frozen vet prompt hash.
    checker_model, checker_version:
        Exact checker identity.

    Returns
    -------
    str
        Vet identity.
    """

    return stable_hash(locals())


def compute_execution_identity(
    *,
    stable_id: str,
    recipe_revision: str,
    env_generation: str,
    runner_version: str,
    target: str,
    machine_class: str,
    seed_policy: Mapping[str, Any],
    framework_adapter: Mapping[str, Any],
    device: str,
) -> str:
    """Compute a controlled execution identity.

    Parameters
    ----------
    stable_id, recipe_revision, env_generation:
        Model, recipe, and environment identities.
    runner_version, target, machine_class:
        Exact runner and platform facts.
    seed_policy, framework_adapter, device:
        Execution policy inputs.

    Returns
    -------
    str
        Execution identity.
    """

    return stable_hash(locals())


def stale_dependencies(changed_inputs: Iterable[str]) -> StalenessReport:
    """Return dependent products stale for named byte-level input changes.

    Parameters
    ----------
    changed_inputs:
        Names such as source, evidence, code, lock, runner, or checker_prompt.

    Returns
    -------
    StalenessReport
        Transitive dependency invalidations.
    """

    graph = {
        "source": {"source", "recipe", "fidelity", "vet"},
        "evidence": {"evidence", "fidelity", "vet"},
        "code": {"recipe", "fidelity"},
        "source_to_code_map": {"fidelity"},
        "authored_metadata": {"vet"},
        "lock": {"environment"},
        "resolved_export": {"environment"},
        "probe": {"environment"},
        "runner": {"execution"},
        "adapter": {"recipe", "execution"},
        "input": {"recipe", "execution"},
        "author_prompt": {"vet"},
        "checker_prompt": {"fidelity", "vet"},
    }
    stale: set[str] = set()
    for changed in changed_inputs:
        stale.update(graph.get(changed, {changed}))
    if stale & {"recipe", "environment"}:
        stale.add("execution")
    if "execution" in stale:
        stale.update({"attempts", "run_status"})
    if "fidelity" in stale:
        stale.add("fidelity_verdict")
    if "vet" in stale:
        stale.add("accuracy_gate")
    return StalenessReport(frozenset(stale))


#: W44 SHADOW-EXCLUSION RIDER: the ONE structural marker naming a record as
#: written by a shadow-accuracy rehearsal (R6b / W3's third mode). The marker
#: lives ON THE RECORD -- never an environment or mode flag -- and every
#: reuse/authority seam keys its exclusion on :func:`is_shadow_rehearsal_record`
#: alone: the verdict-reuse predicate (``gate_reuse``), the artifact checkpoint
#: fold (``artifact_transactions._fold_transaction_chain``), the reducer's
#: current-gate accumulator, and every cache registered in
#: ``cache_inventory.json`` (each row declares its ``shadow_exclusion``).
#: SEAM (Wave-2 rotation batch): the schema field itself is W3's half (WS1);
#: it must bind exactly this name. Until that schema lands, no persisted
#: record can carry the marker (schemas refuse unknown fields), so this
#: read-side exclusion is inert-by-construction on historical stores.
SHADOW_REHEARSAL_MARKER_FIELD = "shadow_rehearsal"


def is_shadow_rehearsal_record(record: Mapping[str, Any]) -> bool:
    """Return whether one record carries the structural shadow-rehearsal marker.

    Fail-closed: ANY present truthy value marks the record (a boolean, a
    rehearsal generation-namespace string, a future enum member). Only an
    absent, ``False``, or ``None`` marker reads as a normal record.

    Parameters
    ----------
    record:
        Any persisted or proposed record mapping (gate, artifact event,
        attempt, model, cache row).
    """

    return bool(record.get(SHADOW_REHEARSAL_MARKER_FIELD))


class ShadowRehearsalStampError(ValueError):
    """A record's existing shadow marker contradicts the declared namespace.

    Fail-closed: the write-side producer never overwrites or launders a
    marker. A record arriving at a stamped seam already carrying a DIFFERENT
    truthy marker value is evidence of cross-generation or cross-mode
    contamination and refuses before anything durable happens.
    """


def stamp_shadow_rehearsal(record: dict) -> dict:
    """Stamp the active rehearsal namespace onto one record about to persist.

    W3's write-side producer half (the disclosure layer over W44's read-side
    exclusion): while the shadow-rehearsal mode is declared
    (``MENAGERIE_SHADOW_REHEARSAL_RUN=<namespace>``), EVERY record a leg
    writes carries the typed structural marker
    :data:`SHADOW_REHEARSAL_MARKER_FIELD` whose value is the declared
    rehearsal generation namespace string -- never a bare ``True``. Applied
    at the shared append/mint seams (the ``JsonlLedger`` durable write door,
    the reducer's live model door, the checker gate scaffold seal, the
    artifact-event mint, and the paid-session row mint), each BEFORE the
    seam's own content hash so the marker is inside every derived identity
    and cannot be stripped without changing it.

    LIVE MODE ONLY by construction: semantic replay routes through the
    in-memory replay ledgers and rehydration only reads, so neither ever
    reaches a stamped seam. Outside the mode this is an exact no-op.

    Parameters
    ----------
    record:
        Mutable record mapping the calling seam owns (already copied from any
        caller-held payload). Mutated in place and returned.

    Returns
    -------
    dict
        The same record, marker-stamped when the mode is active.

    Raises
    ------
    ShadowRehearsalStampError
        If the record already carries a different truthy marker value.
    menagerie.crawler.checker_calibration.AmbiguousLegModeError
        If calibration and shadow-rehearsal are both declared.
    """

    # Runtime import: identity is the low-level module every write seam
    # already imports, while the mode vocabulary lives with the calibration
    # mode in checker_calibration (which transitively imports identity).
    from menagerie.crawler.checker_calibration import shadow_rehearsal_namespace

    namespace = shadow_rehearsal_namespace()
    if namespace is None:
        return record
    existing = record.get(SHADOW_REHEARSAL_MARKER_FIELD)
    if existing is not None and existing is not False and existing != namespace:
        raise ShadowRehearsalStampError(
            f"record already carries shadow_rehearsal={existing!r}, which contradicts "
            f"the declared rehearsal namespace {namespace!r}; refusing to overwrite "
            "a foreign marker"
        )
    record[SHADOW_REHEARSAL_MARKER_FIELD] = namespace
    return record
