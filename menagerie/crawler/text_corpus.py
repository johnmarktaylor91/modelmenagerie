"""Durable text corpus: fetch-once retention of every document a crawl holds.

The crawl's purpose statement is fetch-once: make the model runnable and keep
all text and info a later pass could need, so the crawl never has to be re-run.
Fetched document bytes already survive inside ``.crawl-local/work/<id>/author``
custody trees, but that scratch tree is the disposable half of the store and is
on the future reaper's path. This module is the durable half: a
content-addressed payload store plus an append-only index ledger under
``records/text-corpus/``, populated by an idempotent, digest-checked promotion
at the point a model's author result is accepted (and by the operator backfill
tool for existing work dirs).

Contracts, in the order the tests pin them:

* Promotion admits bytes only from the author root's own evidence directories
  and CAS -- a producer-written path is provenance, never an authority -- and
  only when the full recomputed digest matches the recorded one.
* The corpus tree is NEVER reapable; author scratch is reapable only once
  :func:`work_dir_reap_clearance` recomputes, from corpus bytes alone, that
  every retainable document is durably present.
* ``verify-corpus`` re-hashes every payload and resolves every excerpt back to
  the corpus document its locator names.
* Multi-host union is a content-addressed union: identical digests are one
  payload, differing bytes at one address refuse loudly, and index rows dedup
  on their deterministic ``corpus_row_id``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from menagerie.crawler.constants import TEXT_CORPUS_INDEX_SCHEMA_VERSION
from menagerie.crawler.evidence import EvidenceValidationError, _validate_locator
from menagerie.crawler.fetcher import cas_path, publish_cas_object
from menagerie.crawler.identity import hash_bytes, is_sha256, stable_hash
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import JsonlLedger, SingleWriterError, scan_jsonl
from menagerie.crawler.source_broker import broker_evidence_dirs

TEXT_CORPUS_DIRNAME = "text-corpus"
CORPUS_INDEX_FILENAME = "corpus-index.jsonl"

#: Closed retained-role vocabulary. The three broker manifest roles carry over
#: verbatim; the last three name retention classes the manifest never covered.
CORPUS_ROLES = frozenset(
    {
        "implementation",
        "introducing-paper",
        "documentation",
        "probe",
        "discovery",
        "machine-inventory",
        "uncatalogued",
    }
)

#: Closed origin vocabulary naming which retention class produced the bytes.
CORPUS_ORIGINS = frozenset(
    {"custody-promotion", "broker-evidence", "discovery-evidence", "source-cas"}
)

#: Closed license vocabulary mirrored from ``licenses.RedistributionClass``.
_LICENSE_DISPOSITIONS = frozenset(
    {"public-compatible", "restricted-private", "manifest-only", "not-applicable"}
)

_ROW_ID_TAG = "menagerie.crawler.text-corpus-row.v1"

#: Serializes in-process promotions so two driver threads never race the
#: index ledger's single-writer flock (which is per file handle, not per
#: thread). Cross-process exclusion stays with the flock itself.
_PROMOTION_LOCK = threading.Lock()

#: Cross-process corpus-writer contention is waited out, but never without a
#: deadline. A writer surviving this bound becomes a typed promotion refusal;
#: it cannot abort the driver leg.
CORPUS_WRITER_LOCK_TIMEOUT_SECONDS = 10.0


class TextCorpusError(Exception):
    """Base error for durable text-corpus failures."""


class CorpusIntegrityError(TextCorpusError):
    """The corpus itself holds bytes that do not match their content address."""


class CorpusReapRefusal(TextCorpusError):
    """A reaper asked to remove a path the retention contract protects."""


@dataclass(frozen=True)
class CorpusGrowthSurface:
    """One BL14-owned directory whose growth and lifecycle must stay explicit."""

    key: str
    relative_path: str
    bytes_per_model: int
    measurement_basis: str
    capacity_authority: str
    reclaimer: Optional[str]
    durable_retention: bool = False


#: Closed growth/reclaimer inventory. Rates are the 2026-08-16 pilot
#: measurements cited by ROUND1 synthesis; they are planning evidence, not a
#: quota. Work custody is reclaimed by the ordinary reaper after corpus
#: clearance. Corpus payloads/index are canonical fetch-once state and are
#: therefore capacity-governed rather than eligible for ordinary GC.
CORPUS_GROWTH_SURFACES = (
    CorpusGrowthSurface(
        key="author-broker-evidence",
        relative_path=".crawl-local/work/<stable_id>/author/**/broker/evidence",
        bytes_per_model=11_750_000,
        measurement_basis="751 MiB / 67 pilot author roots",
        capacity_authority="campaign work-root disk authority",
        reclaimer="reaper.apply_reap_plan after work_dir_reap_clearance",
    ),
    CorpusGrowthSurface(
        key="author-source-cas",
        relative_path=".crawl-local/work/<stable_id>/author/source-cas",
        bytes_per_model=1_633_068,
        measurement_basis="109,415,540 bytes / 67 pilot author roots",
        capacity_authority="campaign work-root disk authority",
        reclaimer="reaper.apply_reap_plan after work_dir_reap_clearance",
    ),
    CorpusGrowthSurface(
        key="corpus-index",
        relative_path="menagerie/crawler/records/text-corpus/corpus-index.jsonl",
        bytes_per_model=34_925,
        measurement_basis="2.34 MB / 67 pilot members",
        capacity_authority="campaign records-root disk authority",
        reclaimer=None,
        durable_retention=True,
    ),
    CorpusGrowthSurface(
        key="corpus-payloads",
        relative_path="menagerie/crawler/records/text-corpus/sha256",
        bytes_per_model=723_880,
        measurement_basis="48.5 MB / 67 pilot members",
        capacity_authority="campaign records-root disk authority",
        reclaimer=None,
        durable_retention=True,
    ),
)


def corpus_root(records_root: Union[str, Path]) -> Path:
    """Return the corpus root below one records store."""

    return Path(records_root) / TEXT_CORPUS_DIRNAME


def corpus_index_path(records_root: Union[str, Path]) -> Path:
    """Return the append-only corpus index ledger path."""

    return corpus_root(records_root) / CORPUS_INDEX_FILENAME


def corpus_payload_path(records_root: Union[str, Path], content_sha256: str) -> Path:
    """Return the canonical payload path for one document digest."""

    return cas_path(corpus_root(records_root), content_sha256)


#: W10 (+AL-5): the DECLARED closed volatile-field list of one corpus index
#: row. These are advisory observation metadata a lawful WITHIN-generation
#: re-promotion may re-observe (a second fetch of the same bytes moves its
#: clock, redirect target, or advisory dispositions without minting new
#: provenance). On an existing ``corpus_row_id`` the promotion retains the
#: existing row's values for exactly these fields -- the immutable FIRST
#: observation -- so the replay is byte-identical and the ledger's
#: idempotent-replay rule lands it as ``replayed`` instead of a
#: conflicting-replay driver abort (R19: 11 members, two aborts).
#: CLOSED VOCABULARY: widening this tuple is a Wave-2-class reviewed change
#: (it changes row-replay semantics) and must update the pin test in the same
#: change. It must NEVER overlap :func:`corpus_row_id`'s identity inputs --
#: the m10949 lesson stands: never widen corpus_row_id.
_CORPUS_ROW_VOLATILE_FIELDS: tuple[str, ...] = (
    "license_disposition",
    "retrieved_at",
    "final_url",
    "locator",
    "custody_disposition",
)


def _retain_volatile_first_observation(
    index_row: JsonObject, existing: Optional[Mapping[str, Any]]
) -> JsonObject:
    """Return ``index_row`` with volatile fields pinned to the first observation.

    When ``existing`` holds the already-persisted row for the same
    ``corpus_row_id``, every declared volatile field is forced to the existing
    row's value -- present iff present there -- so a re-observation replays
    byte-identically. Identity and content fields are never touched: any
    non-volatile divergence still reaches the ledger and refuses as a
    conflicting replay (the tripwire stays armed).
    """

    if existing is None:
        return index_row
    for field in _CORPUS_ROW_VOLATILE_FIELDS:
        if field in existing:
            index_row[field] = existing[field]
        else:
            index_row.pop(field, None)
    return index_row


def corpus_row_id(
    *,
    stable_id: str,
    content_sha256: str,
    source_id: str,
    role: str,
    origin: str,
    work_id: str,
) -> str:
    """Return the deterministic identity of one corpus index row.

    Every field is derived from work-dir inputs, never from a promotion-time
    clock, so the same document promoted for the same model on any host mints
    the same row and the ledger's idempotent-replay rule applies unchanged.

    The work generation is part of the identity: a RE-DRIVEN model re-fetches
    its sources under a new ``work_id`` with a new ``retrieved_at``, and that
    is new provenance -- a new row, never a "conflicting replay" of the prior
    generation's row (m10949's step-100 award collided with its own mini-rung
    promotion, 2026-08-13). Payload dedup is unaffected: bytes are stored once
    per ``content_sha256`` in the CAS regardless of how many generations index
    them.
    """

    return stable_hash([_ROW_ID_TAG, stable_id, content_sha256, source_id, role, origin, work_id])


def open_corpus_index(records_root: Union[str, Path]) -> JsonlLedger:
    """Open the corpus index ledger for appending (single-writer flock)."""

    return JsonlLedger(
        corpus_index_path(records_root),
        TEXT_CORPUS_INDEX_SCHEMA_VERSION,
        lock_timeout=CORPUS_WRITER_LOCK_TIMEOUT_SECONDS,
    )


# -- retention candidates -----------------------------------------------------


@dataclass(frozen=True)
class RetentionCandidate:
    """One document the fetch-once policy requires the corpus to hold."""

    source_id: str
    content_sha256: str
    role: str
    origin: str
    work_id: str
    requested_role: str
    media_type: str
    url: str
    final_url: str
    retrieved_at: str
    locator: Optional[JsonObject]
    custody_disposition: str


def _read_json(path: Path) -> Optional[JsonObject]:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _pack_paths(author_root: Path) -> list[Path]:
    """Return every broker pack / receipts document below one author root."""

    paths = [author_root / "source-targets.json"]
    for evidence_dir in broker_evidence_dirs(author_root):
        paths.append(evidence_dir.parent / "receipts.json")
    return [path for path in paths if path.is_file()]


def _outcome_rows(author_root: Path) -> list[JsonObject]:
    rows: list[JsonObject] = []
    seen: set[str] = set()
    for path in _pack_paths(author_root):
        pack = _read_json(path)
        if pack is None:
            continue
        broker = pack.get("broker")
        outcomes = broker.get("outcomes") if isinstance(broker, Mapping) else None
        if not isinstance(outcomes, list):
            continue
        for raw in outcomes:
            if not isinstance(raw, Mapping):
                continue
            digest = raw.get("sha256")
            source_id = str(raw.get("source_id", ""))
            if not isinstance(digest, str) or not is_sha256(digest) or not source_id:
                continue
            key = f"{source_id}\n{digest}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(raw))
    return rows


def _role_for_outcome(outcome: Mapping[str, Any]) -> str:
    bound = outcome.get("bound_role")
    if isinstance(bound, str) and bound in CORPUS_ROLES:
        return bound
    return "uncatalogued"


def _manifest_rows(author_root: Path) -> list[JsonObject]:
    """Return the frozen source-manifest rows recorded below one author root.

    The author request envelope's ``source_manifest`` is the authoritative
    merged manifest (broker rows, controlled-fetch rows, and the machine
    inventory row). The broker packs' ``sources`` lists and the cached
    ``routed-inventory-row.json`` cover roots captured before the envelope was
    written. Rows dedup on ``(source_id, content_sha256)``.
    """

    rows: list[JsonObject] = []
    seen: set[str] = set()

    def _admit(raw: Any) -> None:
        if not isinstance(raw, Mapping):
            return
        digest = raw.get("content_sha256") or raw.get("expected_sha256")
        source_id = str(raw.get("source_id", ""))
        if not isinstance(digest, str) or not is_sha256(digest) or not source_id:
            return
        key = f"{source_id}\n{digest}"
        if key in seen:
            return
        seen.add(key)
        rows.append({**dict(raw), "content_sha256": digest})

    request = _read_json(author_root / "request.json") or {}
    manifest = request.get("source_manifest")
    if isinstance(manifest, Mapping):
        for field in ("sources", "supplementary_sources"):
            entries = manifest.get(field)
            if isinstance(entries, list):
                for raw in entries:
                    _admit(raw)
    for path in _pack_paths(author_root):
        pack = _read_json(path)
        if pack is None:
            continue
        entries = pack.get("sources")
        if isinstance(entries, list):
            for raw in entries:
                _admit(raw)
    # The accepted result is itself a frozen binding surface. It matters for
    # crash-resume fixtures and legacy roots whose disposable request envelope
    # is absent: both proposal and terminal handoff arms carry the exact source
    # list that grounded their excerpts.
    result = _read_json(author_root / "result.json") or {}
    for raw in _accepted_result_source_rows(result):
        _admit(raw)
    _admit(_read_json(author_root / "routed-inventory-row.json"))
    return rows


def _role_for_manifest_row(row: Mapping[str, Any]) -> str:
    role = row.get("broker_role") or row.get("role")
    if isinstance(role, str) and role in CORPUS_ROLES:
        return role
    if str(row.get("retrieval_status", "")) == "machine-derived":
        return "machine-inventory"
    return "uncatalogued"


def _accepted_result_source_rows(document: Mapping[str, Any]) -> list[JsonObject]:
    """Return the exact source rows frozen into one accepted author result."""

    payload = document.get("payload")
    if not isinstance(payload, Mapping):
        return []
    proposals: list[Mapping[str, Any]] = []
    proposal = payload.get("proposal")
    if isinstance(proposal, Mapping):
        proposals.append(proposal)
    handoff = payload.get("handoff_execution")
    if isinstance(handoff, Mapping) and isinstance(handoff.get("proposal"), Mapping):
        proposals.append(handoff["proposal"])

    rows: list[JsonObject] = []
    for bound_proposal in proposals:
        proposed_facts = bound_proposal.get("proposed_facts")
        source_resolution = (
            proposed_facts.get("source_resolution")
            if isinstance(proposed_facts, Mapping)
            else None
        )
        entries = (
            source_resolution.get("sources")
            if isinstance(source_resolution, Mapping)
            else None
        )
        if isinstance(entries, list):
            rows.extend(dict(row) for row in entries if isinstance(row, Mapping))
    return rows


def _locator_for_outcome(outcome: Mapping[str, Any]) -> Optional[JsonObject]:
    receipt = outcome.get("resolver_receipt")
    if not isinstance(receipt, Mapping):
        return None
    locator = {
        key: str(receipt[key])
        for key in ("forge", "repo", "ref", "resolved_sha")
        if isinstance(receipt.get(key), str) and receipt[key]
    }
    return locator or None


def _custody_promotions(author_root: Path) -> tuple[JsonObject, dict[str, str]]:
    custody = _read_json(author_root / "source-custody.json") or {}
    dispositions: dict[str, str] = {}
    for field in ("promotions", "supplement_promotions"):
        rows = custody.get(field)
        if not isinstance(rows, list):
            continue
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            digest = raw.get("content_sha256")
            source_id = str(raw.get("source_id", ""))
            disposition = str(raw.get("disposition", ""))
            if not isinstance(digest, str) or not is_sha256(digest) or not source_id:
                continue
            dispositions[f"{source_id}\n{digest}"] = disposition
    return dict(custody), dispositions


def retention_candidates(author_root: Union[str, Path]) -> tuple[RetentionCandidate, ...]:
    """Enumerate every document one author root obliges the corpus to hold.

    Three retention classes, recomputed from the work dir's own machine-written
    envelopes with zero reliance on any producer-supplied path:

    * every broker outcome that recorded a content digest (manifest rows AND
      the fetched-but-unmanifested probe bodies the grounding path discards),
    * the frozen discovery envelope named by ``discovery-provenance.json``,
    * every object physically present in the author's ``source-cas`` tree that
      no outcome row already covers.
    """

    root = Path(author_root)
    custody, custody_dispositions = _custody_promotions(root)
    work_id = str(custody.get("work_id", ""))

    candidates: list[RetentionCandidate] = []
    covered_digests: set[str] = set()
    covered_keys: set[str] = set()

    for row in _manifest_rows(root):
        digest = str(row["content_sha256"])
        source_id = str(row["source_id"])
        key = f"{source_id}\n{digest}"
        covered_digests.add(digest)
        covered_keys.add(key)
        candidates.append(
            RetentionCandidate(
                source_id=source_id,
                content_sha256=digest,
                role=_role_for_manifest_row(row),
                origin="custody-promotion",
                work_id=work_id,
                requested_role=str(row.get("requested_role", "")),
                media_type=str(row.get("media_type") or "application/octet-stream"),
                url=str(row.get("url") or ""),
                final_url=str(row.get("final_url") or ""),
                retrieved_at=str(row.get("retrieved_at") or ""),
                locator=None,
                custody_disposition=custody_dispositions.get(key, ""),
            )
        )

    for outcome in _outcome_rows(root):
        digest = str(outcome["sha256"])
        source_id = str(outcome["source_id"])
        key = f"{source_id}\n{digest}"
        if key in covered_keys:
            continue
        custody_disposition = custody_dispositions.get(key, "")
        origin = "custody-promotion" if custody_disposition else "broker-evidence"
        covered_digests.add(digest)
        covered_keys.add(key)
        candidates.append(
            RetentionCandidate(
                source_id=source_id,
                content_sha256=digest,
                role=_role_for_outcome(outcome),
                origin=origin,
                work_id=work_id,
                requested_role=str(outcome.get("requested_role", "")),
                media_type=str(outcome.get("media_type") or "application/octet-stream"),
                url=str(outcome.get("url") or ""),
                final_url=str(outcome.get("final_url") or ""),
                retrieved_at=str(outcome.get("retrieved_at") or ""),
                locator=_locator_for_outcome(outcome),
                custody_disposition=custody_disposition,
            )
        )

    provenance = _read_json(root / "discovery-provenance.json") or {}
    evidence = provenance.get("discovery_evidence")
    if isinstance(evidence, Mapping):
        discovery_digest = evidence.get("content_sha256")
        if isinstance(discovery_digest, str) and is_sha256(discovery_digest):
            covered_digests.add(discovery_digest)
            candidates.append(
                RetentionCandidate(
                    source_id="menagerie-source-discovery",
                    content_sha256=discovery_digest,
                    role="discovery",
                    origin="discovery-evidence",
                    work_id=str(provenance.get("work_id", "")) or work_id,
                    requested_role="",
                    media_type=str(evidence.get("media_type") or "application/json"),
                    url=str(evidence.get("url") or ""),
                    final_url="",
                    retrieved_at="",
                    locator=None,
                    custody_disposition="",
                )
            )

    cas_tree = root / "source-cas" / "sha256"
    if cas_tree.is_dir():
        for path in sorted(cas_tree.rglob("*")):
            if not path.is_file():
                continue
            digest = f"sha256:{path.name}"
            if not is_sha256(digest) or digest in covered_digests:
                continue
            covered_digests.add(digest)
            candidates.append(
                RetentionCandidate(
                    source_id=f"source-cas-{path.name[:16]}",
                    content_sha256=digest,
                    role="uncatalogued",
                    origin="source-cas",
                    work_id=work_id,
                    requested_role="",
                    media_type="application/octet-stream",
                    url="",
                    final_url="",
                    retrieved_at="",
                    locator=None,
                    custody_disposition="",
                )
            )

    return tuple(candidates)


def _resolve_candidate_bytes(author_root: Path, digest: str) -> tuple[Optional[bytes], bool]:
    """Resolve one digest's bytes from the author root's own stores.

    Returns ``(bytes, mismatched)``. Every candidate location is re-hashed in
    full before its bytes are admitted; a location whose bytes do not hash to
    the digest is refused and only reported when no location matches at all.
    Producer-recorded paths (custody ``evidence_path``, provenance
    ``cas_path``) are deliberately never read.
    """

    bare = digest.removeprefix("sha256:")
    locations = [
        cas_path(author_root / "source-cas", digest),
        author_root / "source-cas" / f"{bare}.source",
    ]
    for evidence_dir in broker_evidence_dirs(author_root):
        locations.append(evidence_dir / f"{bare[:16]}.bin")
    mismatched = False
    for location in locations:
        if not location.is_file():
            continue
        body = location.read_bytes()
        if hash_bytes(body) == digest:
            return body, mismatched
        mismatched = True
    return None, mismatched


# -- promotion ----------------------------------------------------------------


@dataclass(frozen=True)
class PromotionReport:
    """Typed outcome of one author-root promotion."""

    author_root: Path
    stable_id: str
    dispositions: tuple[JsonObject, ...]

    @property
    def promoted(self) -> tuple[JsonObject, ...]:
        return tuple(row for row in self.dispositions if row["disposition"] == "promoted")

    @property
    def replayed(self) -> tuple[JsonObject, ...]:
        return tuple(row for row in self.dispositions if row["disposition"] == "replayed")

    @property
    def refusals(self) -> tuple[JsonObject, ...]:
        return tuple(
            row
            for row in self.dispositions
            if row["disposition"]
            in ("digest-mismatch-refused", "evidence-missing", "index-busy-refused")
        )


def _publish_payload(records_root: Path, digest: str, body: bytes) -> bool:
    """Publish one payload, returning ``True`` when bytes were written.

    An existing payload is re-hashed rather than trusted. When freshly
    verified custody bytes are in hand, a corrupt shared destination is
    atomically replaced from those exact bytes instead of aborting the leg.
    """

    destination = corpus_payload_path(records_root, digest)
    if destination.is_file():
        existing = destination.read_bytes()
        actual = hash_bytes(existing)
        if actual != digest:
            publish_cas_object(corpus_root(records_root), digest, body)
            return True
        return False
    publish_cas_object(corpus_root(records_root), digest, body)
    return True


def promote_author_root(
    author_root: Union[str, Path],
    records_root: Union[str, Path],
    *,
    license_dispositions: Optional[Mapping[str, str]] = None,
    stable_id_override: Optional[str] = None,
    work_id_override: Optional[str] = None,
) -> PromotionReport:
    """Promote one author root's retained documents into the durable corpus.

    Idempotent, resumable, and digest-checked end to end: payloads publish
    before index rows, so a crash can only ever leave benign payload-first
    intermediate state, and a replay converges on the identical rows.

    Parameters
    ----------
    author_root:
        Per-model author custody root (``.../work/<stable_id>/author``).
    records_root:
        Durable records store receiving the corpus.
    license_dispositions:
        Optional digest-to-redistribution-class mapping mirrored from the
        artifact-authority license decisions. Documents without a mapped
        decision record the conservative ``manifest-only`` default. Advisory
        metadata only; no publication authority derives from it.
    stable_id_override, work_id_override:
        Trusted driver bindings at the author-accept chokepoint. Operator
        backfills omit them and replay the machine-written custody bindings.

    Returns
    -------
    PromotionReport
        One typed disposition per retention candidate.
    """

    root = Path(author_root)
    records = Path(records_root)
    candidates = list(retention_candidates(root))
    custody, _ = _custody_promotions(root)
    stable_id = stable_id_override or str(custody.get("stable_id", "")) or root.parent.name
    accepted_result = _read_json(root / "result.json") or {}
    accepted_work_id = str(accepted_result.get("work_id") or "")
    accepted_stable_id = str(accepted_result.get("stable_id") or "")
    candidate_keys = {
        (candidate.source_id, candidate.content_sha256, candidate.work_id)
        for candidate in candidates
    }
    if accepted_work_id and accepted_stable_id == stable_id:
        _, custody_dispositions = _custody_promotions(root)
        for row in _accepted_result_source_rows(accepted_result):
            digest = row.get("content_sha256")
            source_id = row.get("source_id")
            if (
                not isinstance(digest, str)
                or not is_sha256(digest)
                or not isinstance(source_id, str)
                or not source_id
            ):
                continue
            key = (source_id, digest, accepted_work_id)
            if key in candidate_keys:
                continue
            candidate_keys.add(key)
            candidates.append(
                RetentionCandidate(
                    source_id=source_id,
                    content_sha256=digest,
                    role=_role_for_manifest_row(row),
                    origin="custody-promotion",
                    work_id=accepted_work_id,
                    requested_role=str(row.get("requested_role", "")),
                    media_type=str(row.get("media_type") or "application/octet-stream"),
                    url=str(row.get("url") or ""),
                    final_url=str(row.get("final_url") or ""),
                    retrieved_at=str(row.get("retrieved_at") or ""),
                    locator=None,
                    custody_disposition=custody_dispositions.get(
                        f"{source_id}\n{digest}", ""
                    ),
                )
            )

    licenses = dict(license_dispositions or {})
    for digest, disposition in licenses.items():
        if disposition not in _LICENSE_DISPOSITIONS:
            raise TextCorpusError(
                f"license disposition {disposition!r} for {digest} is outside the "
                f"closed vocabulary {sorted(_LICENSE_DISPOSITIONS)}"
            )

    dispositions: list[JsonObject] = []
    with _PROMOTION_LOCK:
        try:
            ledger = open_corpus_index(records)
        except SingleWriterError:
            return PromotionReport(
                author_root=root,
                stable_id=stable_id,
                dispositions=tuple(
                    {
                        "source_id": candidate.source_id,
                        "content_sha256": candidate.content_sha256,
                        "role": candidate.role,
                        "origin": candidate.origin,
                        "disposition": "index-busy-refused",
                    }
                    for candidate in candidates
                ),
            )
        try:
            existing_rows = {str(row.get("corpus_row_id")): row for row in ledger.records}
            for candidate in candidates:
                candidate_work_id = work_id_override or candidate.work_id
                summary: JsonObject = {
                    "source_id": candidate.source_id,
                    "content_sha256": candidate.content_sha256,
                    "role": candidate.role,
                    "origin": candidate.origin,
                }
                body, mismatched = _resolve_candidate_bytes(root, candidate.content_sha256)
                if body is None:
                    summary["disposition"] = (
                        "digest-mismatch-refused" if mismatched else "evidence-missing"
                    )
                    dispositions.append(summary)
                    continue
                payload_written = _publish_payload(records, candidate.content_sha256, body)
                row_id = corpus_row_id(
                    stable_id=stable_id,
                    content_sha256=candidate.content_sha256,
                    source_id=candidate.source_id,
                    role=candidate.role,
                    origin=candidate.origin,
                    work_id=candidate_work_id,
                )
                index_row: JsonObject = {
                    "schema_version": TEXT_CORPUS_INDEX_SCHEMA_VERSION,
                    "corpus_row_id": row_id,
                    "stable_id": stable_id,
                    "work_id": candidate_work_id,
                    "source_id": candidate.source_id,
                    "role": candidate.role,
                    "origin": candidate.origin,
                    "content_sha256": candidate.content_sha256,
                    "bytes_len": len(body),
                    "media_type": candidate.media_type,
                    "url": candidate.url,
                    "retrieved_at": candidate.retrieved_at,
                    "license_disposition": licenses.get(candidate.content_sha256, "manifest-only"),
                }
                if candidate.requested_role:
                    index_row["requested_role"] = candidate.requested_role
                if candidate.final_url:
                    index_row["final_url"] = candidate.final_url
                if candidate.locator is not None:
                    index_row["locator"] = candidate.locator
                if candidate.custody_disposition:
                    index_row["custody_disposition"] = candidate.custody_disposition
                # Volatile fields are advisory snapshots, not row identity
                # (W10 generalizes the historical license-only carve-out): a
                # within-generation re-promotion retains the immutable first
                # observation instead of turning the same provenance row into
                # a ledger conflict.
                index_row = _retain_volatile_first_observation(
                    index_row, existing_rows.get(row_id)
                )
                appended = ledger.append(index_row)
                summary["disposition"] = "promoted" if appended.appended else "replayed"
                if payload_written:
                    summary["payload_written"] = True
                dispositions.append(summary)
        finally:
            ledger.close()

    return PromotionReport(author_root=root, stable_id=stable_id, dispositions=tuple(dispositions))


# -- GC contract ---------------------------------------------------------------


@dataclass(frozen=True)
class ReapClearance:
    """Replay-derived verdict on whether one work dir may be reaped."""

    work_dir: Path
    cleared: bool
    reasons: tuple[str, ...]


def assert_not_corpus_path(path: Union[str, Path], records_root: Union[str, Path]) -> None:
    """Refuse any reap target inside the durable corpus tree.

    Every future reaper MUST route candidate deletions through this guard: the
    corpus (payloads and index alike) is never reapable, under any policy.
    """

    resolved = Path(path).resolve()
    protected = corpus_root(records_root).resolve()
    if resolved == protected or protected in resolved.parents:
        raise CorpusReapRefusal(
            f"{resolved} is inside the durable text corpus {protected}; the corpus "
            "is never reapable"
        )


def _index_rows(records_root: Path) -> list[JsonObject]:
    path = corpus_index_path(records_root)
    if not path.is_file():
        return []
    return scan_jsonl(path, validate=False)


def work_dir_reap_clearance(
    work_dir: Union[str, Path], records_root: Union[str, Path]
) -> ReapClearance:
    """Recompute, from corpus bytes alone, whether one work dir may be reaped.

    Fail-closed and producer-blind: clearance re-enumerates the work dir's
    retention candidates and requires each one's payload to be physically
    present in the corpus with matching bytes, plus a matching index row. No
    persisted flag or prior report steers the verdict. A work dir with no
    author custody tree has nothing the corpus adjudicates and clears vacuously.
    """

    target = Path(work_dir)
    records = Path(records_root)
    author_root = target / "author"
    if not author_root.is_dir():
        return ReapClearance(work_dir=target, cleared=True, reasons=())

    durable_bindings = {
        (
            str(row.get("stable_id")),
            str(row.get("source_id")),
            str(row.get("content_sha256")),
            str(row.get("role")),
            str(row.get("origin")),
        )
        for row in _index_rows(records)
    }
    custody, _ = _custody_promotions(author_root)
    stable_id = str(custody.get("stable_id", "")) or author_root.parent.name

    reasons: list[str] = []
    for candidate in retention_candidates(author_root):
        label = f"{candidate.source_id} ({candidate.content_sha256})"
        payload = corpus_payload_path(records, candidate.content_sha256)
        if not payload.is_file():
            reasons.append(f"payload absent from corpus: {label}")
            continue
        if hash_bytes(payload.read_bytes()) != candidate.content_sha256:
            reasons.append(f"corpus payload fails its content address: {label}")
            continue
        binding = (
            stable_id,
            candidate.source_id,
            candidate.content_sha256,
            candidate.role,
            candidate.origin,
        )
        if binding not in durable_bindings:
            reasons.append(f"index row absent from corpus: {label}")

    return ReapClearance(work_dir=target, cleared=not reasons, reasons=tuple(reasons))


# -- verification ----------------------------------------------------------------


@dataclass(frozen=True)
class VerifyReport:
    """Typed result of one full corpus verification pass."""

    passed: bool
    findings: tuple[JsonObject, ...]
    row_count: int
    payload_count: int
    excerpt_count: int
    orphan_payload_count: int


def _iter_excerpts(document: Any) -> Iterable[JsonObject]:
    if isinstance(document, Mapping):
        excerpts = document.get("excerpts")
        if isinstance(excerpts, list):
            for entry in excerpts:
                if isinstance(entry, Mapping) and {
                    "source_id",
                    "text",
                    "text_sha256",
                    "locator",
                } <= set(entry):
                    yield dict(entry)
        for value in document.values():
            yield from _iter_excerpts(value)
    elif isinstance(document, list):
        for value in document:
            yield from _iter_excerpts(value)


def _excerpt_documents(
    work_root: Path,
) -> Iterable[tuple[str, str, Path, JsonObject]]:
    """Yield durable excerpt documents with their exact work binding."""

    for name in ("result.json", "evidence-pack.json"):
        for path in sorted(work_root.rglob(name)):
            # Per-attempt scratch is superseded history: each attempt's excerpts
            # were adjudicated against that attempt's own frozen manifest, whose
            # superseded machine rows are not durably recoverable. The durable
            # grounding surfaces are the author-level result and terminal packs.
            if "attempts" in path.relative_to(work_root).parts:
                continue
            document = _read_json(path)
            if document is None:
                continue
            author_root = path.parent
            while author_root != work_root and author_root.name != "author":
                author_root = author_root.parent
            stable_id = (
                author_root.parent.name if author_root.name == "author" else path.parent.name
            )
            yield stable_id, str(document.get("work_id") or ""), author_root, document


def _excerpt_source_digests(
    document: Mapping[str, Any], author_root: Path
) -> dict[str, set[str]]:
    """Return the exact accepted source bindings for one excerpt document.

    The accepted result's own frozen proposal wins when present. Historical
    broker packs may reuse a ``source_id`` across work generations, so unioning
    them would manufacture ambiguity and reintroduce generation aliasing. The
    author-root manifest scan is only the legacy/simple-document fallback.
    """

    rows: Sequence[Mapping[str, Any]] = _accepted_result_source_rows(document)
    if not rows:
        rows = _manifest_rows(author_root)

    digests: dict[str, set[str]] = {}
    for row in rows:
        source_id = row.get("source_id")
        digest = row.get("content_sha256") or row.get("expected_sha256")
        if isinstance(source_id, str) and source_id and isinstance(digest, str) and is_sha256(digest):
            digests.setdefault(source_id, set()).add(digest)
    return digests


def verify_corpus(
    records_root: Union[str, Path],
    *,
    work_roots: Sequence[Union[str, Path]] = (),
) -> VerifyReport:
    """Verify the corpus fail-closed: payload integrity and excerpt resolution.

    Every index row's payload must be present and re-hash to its address, and
    every excerpt found below ``work_roots`` must resolve through its
    ``source_id``'s index rows to a corpus document its locator verifies
    against. Payloads with no index row yet are the benign payload-first
    intermediate state of a resumable promotion; they are counted, never
    failed.
    """

    records = Path(records_root)
    findings: list[JsonObject] = []
    rows = scan_jsonl(corpus_index_path(records)) if corpus_index_path(records).is_file() else []

    referenced: set[str] = set()
    for row in rows:
        digest = str(row.get("content_sha256"))
        referenced.add(digest)
        payload = corpus_payload_path(records, digest)
        if not payload.is_file():
            findings.append(
                {
                    "kind": "payload-missing",
                    "content_sha256": digest,
                    "stable_id": row.get("stable_id"),
                    "source_id": row.get("source_id"),
                }
            )
            continue
        actual = hash_bytes(payload.read_bytes())
        if actual != digest:
            findings.append(
                {
                    "kind": "payload-digest-mismatch",
                    "content_sha256": digest,
                    "actual_sha256": actual,
                    "stable_id": row.get("stable_id"),
                    "source_id": row.get("source_id"),
                }
            )

    payload_count = 0
    orphans = 0
    tree = corpus_root(records) / "sha256"
    if tree.is_dir():
        for path in tree.rglob("*"):
            if not path.is_file():
                continue
            payload_count += 1
            if f"sha256:{path.name}" not in referenced:
                orphans += 1

    rows_by_stable: dict[str, list[JsonObject]] = {}
    for row in rows:
        rows_by_stable.setdefault(str(row.get("stable_id")), []).append(row)

    excerpt_count = 0
    for work_root in work_roots:
        for stable_id, work_id, author_root, document in _excerpt_documents(Path(work_root)):
            manifest_digests = _excerpt_source_digests(document, author_root)
            for excerpt in _iter_excerpts(document):
                excerpt_count += 1
                identity = {
                    "stable_id": stable_id,
                    "evidence_id": excerpt.get("evidence_id"),
                    "source_id": excerpt.get("source_id"),
                }
                text = str(excerpt["text"]).encode("utf-8")
                if not work_id:
                    findings.append({"kind": "excerpt-work-id-missing", **identity})
                    continue
                if hash_bytes(text) != excerpt["text_sha256"]:
                    findings.append({"kind": "excerpt-text-digest-mismatch", **identity})
                    continue
                digests = manifest_digests.get(str(excerpt["source_id"]), set())
                if not digests:
                    findings.append({"kind": "excerpt-source-binding-missing", **identity})
                    continue
                if len(digests) != 1:
                    findings.append({"kind": "excerpt-source-binding-ambiguous", **identity})
                    continue
                content_sha256 = next(iter(digests))
                candidates = [
                    row
                    for row in rows_by_stable.get(stable_id, [])
                    if row.get("work_id") == work_id
                    and row.get("source_id") == excerpt["source_id"]
                    and row.get("content_sha256") == content_sha256
                ]
                if not candidates:
                    findings.append({"kind": "excerpt-source-unpromoted", **identity})
                    continue
                resolved = False
                for row in candidates:
                    payload = corpus_payload_path(records, str(row["content_sha256"]))
                    if not payload.is_file():
                        continue
                    try:
                        _validate_locator(
                            str(excerpt.get("evidence_id") or excerpt["source_id"]),
                            str(excerpt["locator"]),
                            text,
                            payload.read_bytes(),
                        )
                    except EvidenceValidationError:
                        continue
                    resolved = True
                    break
                if not resolved:
                    findings.append({"kind": "excerpt-locator-unresolved", **identity})

    return VerifyReport(
        passed=not findings,
        findings=tuple(findings),
        row_count=len(rows),
        payload_count=payload_count,
        excerpt_count=excerpt_count,
        orphan_payload_count=orphans,
    )


def verify_corpus_launch_gate(
    records_root: Union[str, Path], *, work_root: Union[str, Path]
) -> VerifyReport:
    """Require a zero-finding corpus verification before driver admission.

    Parameters
    ----------
    records_root:
        Canonical campaign records root.
    work_root:
        Campaign work root whose durable excerpt surfaces must resolve.

    Returns
    -------
    VerifyReport
        The clean report, for launch attestation and tests.

    Raises
    ------
    TextCorpusError
        If any payload, binding, generation, or excerpt finding remains.
    """

    report = verify_corpus(records_root, work_roots=(work_root,))
    if report.passed:
        return report
    kinds: dict[str, int] = {}
    for finding in report.findings:
        kind = str(finding.get("kind", "unknown"))
        kinds[kind] = kinds.get(kind, 0) + 1
    summary = ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items()))
    raise TextCorpusError(
        f"text-corpus launch gate found {len(report.findings)} finding(s): {summary}"
    )


# -- multi-host union ------------------------------------------------------------


@dataclass(frozen=True)
class UnionReport:
    """Typed result of one content-addressed corpus union."""

    payloads_copied: int
    payloads_shared: int
    rows_appended: int
    rows_replayed: int


def union_corpus(
    sources: Sequence[Union[str, Path]],
    output_records_root: Union[str, Path],
) -> UnionReport:
    """Union shard corpora into one records store, content-addressed.

    Payloads first, rows second, so interruption leaves only the benign
    payload-first state. Identical digests unify to one payload; differing
    bytes at one address raise :class:`CorpusIntegrityError`; index rows dedup
    on ``corpus_row_id`` through the ledger's idempotent-replay rule.
    """

    output = Path(output_records_root)
    payloads_copied = 0
    payloads_shared = 0
    rows_appended = 0
    rows_replayed = 0

    for source in sources:
        tree = corpus_root(Path(source)) / "sha256"
        if not tree.is_dir():
            continue
        for path in sorted(tree.rglob("*")):
            if not path.is_file():
                continue
            digest = f"sha256:{path.name}"
            if not is_sha256(digest):
                continue
            body = path.read_bytes()
            actual = hash_bytes(body)
            if actual != digest:
                raise CorpusIntegrityError(
                    f"shard payload {path} holds bytes hashing to {actual}, not its "
                    f"address {digest}"
                )
            if _publish_payload(output, digest, body):
                payloads_copied += 1
            else:
                payloads_shared += 1

    with _PROMOTION_LOCK:
        ledger: Optional[JsonlLedger] = None
        existing_rows: dict[str, JsonObject] = {}
        try:
            for source in sources:
                path = corpus_index_path(Path(source))
                if not path.is_file():
                    continue
                for row in scan_jsonl(path):
                    replay = {
                        key: value
                        for key, value in row.items()
                        if key not in ("ledger_seq", "payload_sha256")
                    }
                    if ledger is None:
                        ledger = open_corpus_index(output)
                        existing_rows = {
                            str(existing.get("corpus_row_id")): existing
                            for existing in ledger.records
                        }
                    # Same volatile carve-out as promotion (W10): the union
                    # output's first observation wins for advisory fields;
                    # non-volatile divergence still refuses as a conflict.
                    replay = _retain_volatile_first_observation(
                        replay, existing_rows.get(str(replay.get("corpus_row_id")))
                    )
                    appended = ledger.append(replay)
                    if appended.appended:
                        existing_rows[str(replay.get("corpus_row_id"))] = replay
                        rows_appended += 1
                    else:
                        rows_replayed += 1
        finally:
            if ledger is not None:
                ledger.close()

    return UnionReport(
        payloads_copied=payloads_copied,
        payloads_shared=payloads_shared,
        rows_appended=rows_appended,
        rows_replayed=rows_replayed,
    )


# -- award-path helpers ------------------------------------------------------------


def license_dispositions_from_artifact_events(
    events: Iterable[Mapping[str, Any]], stable_id: str
) -> dict[str, str]:
    """Mirror artifact-authority license decisions for one model's digests.

    Later events supersede earlier ones for the same digest, matching the
    ledger's append order. The result is advisory corpus metadata; the
    artifact ledger remains the sole publication authority.
    """

    dispositions: dict[str, str] = {}
    for event in events:
        claims = event.get("claims")
        if not isinstance(claims, list):
            continue
        for claim in claims:
            if not isinstance(claim, Mapping):
                continue
            if str(claim.get("stable_id", "")) != stable_id:
                continue
            digest = claim.get("revision")
            disposition = claim.get("license_disposition")
            if (
                isinstance(digest, str)
                and is_sha256(digest)
                and isinstance(disposition, str)
                and disposition in _LICENSE_DISPOSITIONS
            ):
                dispositions[digest] = disposition
    return dispositions
