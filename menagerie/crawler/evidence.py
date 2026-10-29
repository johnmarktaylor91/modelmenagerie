"""Literal evidence, locator, and claim-coverage validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Any, Iterable, Mapping, Optional, Sequence, Union

from menagerie.crawler.fetcher import UnpinnedTargetError, cas_path
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.source_broker import (
    OUTCOME_PAPER_DERIVATION_ONLY,
    OUTCOME_UNFETCHABLE_BY_POLICY,
)


CHECKED_LINK_RECEIPT_OUTCOMES_BY_DISPOSITION = {
    OUTCOME_UNFETCHABLE_BY_POLICY: frozenset({OUTCOME_UNFETCHABLE_BY_POLICY, "redirect-refused"}),
    OUTCOME_PAPER_DERIVATION_ONLY: frozenset({OUTCOME_PAPER_DERIVATION_ONLY}),
}

# ---------------------------------------------------------------------------
# Locator authority (KERNEL_DECISION 2026-08-29: excerpt-locator adjudication
# is machine-decidable and is withdrawn from model judgment)
# ---------------------------------------------------------------------------

#: Historical locator semantics: ``bytes:`` locators strict, every other form
#: whitespace-folded MEMBERSHIP only -- the line number of a textual locator
#: was never checked by any machine. Recorded author results rehydrate at this
#: basis forever (append-only history is never reinterpreted).
LOCATOR_AUTHORITY_LEGACY_MEMBERSHIP = "legacy-membership-v1"

#: The mechanical locator authority (rule-2 resolution, 2026-08-29): textual
#: locators carrying a line claim are POSITIONALLY adjudicated against the
#: frozen CAS bytes -- 1-based inclusive lines, universal-newline splitting,
#: zero tolerance -- under the pipeline's one documented byte equivalence
#: (:func:`_fold_space` plus outer-whitespace accommodation). Activated by the
#: live envelope's machine stamp, never by a producer field.
LOCATOR_AUTHORITY_TEXT_LINE = "text-line-locator-v1"

#: Typed per-excerpt attestation row schema (Sol H-2.1).
TEXT_LINE_LOCATOR_SCHEMA = "text-line-locator.v1"
TEXT_LINE_LOCATOR_BASIS = "one-based-inclusive-universal-newline-v1"
TEXT_LINE_LOCATOR_EQUIVALENCE = "unicode-whitespace-run-v1"

#: Closed locator grammar (fable A-2). Every locator is exactly one form:
#: ``bytes`` (strict byte range, unchanged), ``line`` (a textual locator
#: asserting a line position -- the new strict check), or ``membership``
#: (anchor/element/content-addressed forms carrying NO positional claim, for
#: which whitespace-folded membership is the correct semantics, not a gap).
LOCATOR_FORM_BYTES = "bytes"
LOCATOR_FORM_LINE = "line"
LOCATOR_FORM_MEMBERSHIP = "membership"

#: Closed attestation statuses. ``valid``: the excerpt verifies at its locator
#: under the authority. ``invalid``: the frozen bytes resolve and REFUTE the
#: locator (mislocated, absent, out-of-range, or ambiguous positional claim).
#: ``unattestable``: the claim cannot be adjudicated on this host (unknown
#: source row, no digest, or CAS bytes unavailable) -- disclosed, never
#: silently demoted to membership semantics.
LOCATOR_ATTESTATION_VALID = "valid"
LOCATOR_ATTESTATION_INVALID = "invalid"
LOCATOR_ATTESTATION_UNATTESTABLE = "unattestable"

#: A textual locator asserts a line position iff this matches (the Opus sweep
#: script's derivation basis, corpus-measured over 222 observed shapes:
#: ``"t5.md line 21"``, ``"lines 167-169"``, ``"line 60 (author affiliation
#: block)"``). A locator with NO match makes no positional claim and keeps
#: membership semantics; more than one match is an ambiguous positional claim
#: and is refused under the strict authority, never guessed at.
_LINE_LOCATOR_PATTERN = re.compile(r"\blines?\s+(\d+)(?:\s*[-–]\s*(\d+))?\b")

#: Universal-newline splitting over the exact CAS bytes: CRLF, bare CR, and LF
#: all end a line; a final unterminated line still counts; a BOM stays inside
#: line 1's bytes (it folds away only if it is whitespace, which U+FEFF is
#: not). Deliberately NOT ``str.splitlines`` -- that also splits on VT, FF,
#: U+2028, and friends, which no author line-count ever meant.
_UNIVERSAL_NEWLINE_PATTERN = re.compile("\r\n|\r|\n")


class EvidenceValidationError(ValueError):
    """Raised when literal evidence is altered, missing, or uncovered."""


SMP_ZOO_PREFIX = "segmentation_models_pytorch-"
SMP_LEGACY_NAME_PREFIX = "smp_"


@dataclass(frozen=True)
class EvidenceValidationReport:
    """Validated evidence coverage.

    Parameters
    ----------
    excerpt_count:
        Number of validated literal excerpts.
    supported_claims:
        Claim categories named by at least one valid excerpt.
    family_grounded:
        Whether at least one excerpt is explicitly family-level.
    absence_covered_claims:
        Claim categories discharged by a typed absence state rather than by an excerpt.
        Kept separate from ``supported_claims`` so a declared absence can never be
        mistaken for positive literal support when reading the report.
    """

    excerpt_count: int
    supported_claims: frozenset[str]
    family_grounded: bool
    absence_covered_claims: frozenset[str] = frozenset()


def trusted_intake_identity_mismatches(
    proposed_identity: Mapping[str, Any],
    trusted_identity: Mapping[str, Any],
    *,
    intake_name: str,
    intake_zoo: str,
) -> dict[str, dict[str, Any]]:
    """Return trusted-intake identity leaves contradicted by proposed facts.

    The trusted-intake tripwire stays exact for every field except one legacy SMP
    spelling: roster names use ``smp_<model>_<encoder>`` as their full natural-key
    token, while source-grounded R1 proposals use the encoder token as
    ``identity.variant``. That equivalence is accepted only when the intake row is
    from segmentation_models_pytorch and the trusted full token decomposes exactly
    around the proposed encoder token.

    Parameters
    ----------
    proposed_identity:
        Author-proposed ``identity`` object.
    trusted_identity:
        Trusted-intake projection for the same item.
    intake_name:
        Raw trusted intake ``name`` token.
    intake_zoo:
        Raw trusted intake ``zoo`` token.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mismatched trusted fields, keyed by field name.
    """

    return {
        field: {"proposed": proposed_identity.get(field), "trusted": trusted_value}
        for field, trusted_value in trusted_identity.items()
        if not _trusted_identity_field_matches(
            field,
            proposed_identity.get(field),
            trusted_value,
            intake_name=intake_name,
            intake_zoo=intake_zoo,
        )
    }


def _trusted_identity_field_matches(
    field: str,
    proposed_value: Any,
    trusted_value: Any,
    *,
    intake_name: str,
    intake_zoo: str,
) -> bool:
    """Return whether one trusted identity field is satisfied."""

    if proposed_value == trusted_value:
        return True
    if (
        field != "variant"
        or not isinstance(proposed_value, str)
        or not isinstance(trusted_value, str)
    ):
        return False
    return _smp_legacy_variant_matches(
        proposed_variant=proposed_value,
        trusted_variant=trusted_value,
        intake_name=intake_name,
        intake_zoo=intake_zoo,
    )


def _smp_legacy_variant_matches(
    *,
    proposed_variant: str,
    trusted_variant: str,
    intake_name: str,
    intake_zoo: str,
) -> bool:
    """Return whether an SMP full roster token proves the proposed encoder variant."""

    if not intake_zoo.startswith(SMP_ZOO_PREFIX):
        return False
    if trusted_variant != intake_name or not trusted_variant.startswith(SMP_LEGACY_NAME_PREFIX):
        return False
    if proposed_variant != proposed_variant.strip() or not proposed_variant:
        return False
    encoder = _smp_encoder_token(trusted_variant, proposed_variant)
    return encoder == proposed_variant


def _smp_encoder_token(trusted_variant: str, proposed_variant: str) -> Optional[str]:
    """Extract the SMP encoder suffix from a trusted roster token, if it is exact."""

    prefix = SMP_LEGACY_NAME_PREFIX
    suffix = f"_{proposed_variant}"
    if not trusted_variant.startswith(prefix) or not trusted_variant.endswith(suffix):
        return None
    model_token = trusted_variant[len(prefix) : -len(suffix)]
    if not model_token:
        return None
    return proposed_variant


def validate_evidence(
    evidence: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    required_claims: Iterable[str],
    *,
    cas_root: Union[str, Path, None] = None,
    require_family_grounding: bool = False,
    declared_absences: Optional[Mapping[str, Sequence[str]]] = None,
    locator_authority: str = LOCATOR_AUTHORITY_LEGACY_MEMBERSHIP,
) -> EvidenceValidationReport:
    """Validate verbatim excerpts and exhaustive claim-category coverage.

    Coverage is nominal: a claim category is covered when an excerpt names it, or when
    the proposal declares a typed absence state for it. Whether an excerpt's *text*
    entails the claim is the Codex accuracy checker's judgment, not this function's.

    Claim-category strings are matched EXACTLY. A leaf path ``X.y`` never satisfies the
    aggregate ``X``: an excerpt supporting only ``external_metadata.citation.arxiv_id``
    genuinely does not support the citation as a whole, and rolling leaves up into their
    parent would silently launder that partial support into full coverage.

    Parameters
    ----------
    evidence:
        Model ``evidence`` block.
    source_manifest:
        A manifest wrapper containing ``sources`` or a direct source list.
    required_claims:
        Every gated claim category that must have literal support.
    cas_root:
        Optional governing CAS root; when provided every source read resolves
        digest-derived beneath it and recorded ``cas_path`` values are ignored.
    require_family_grounding:
        Whether a family-level grounding excerpt is mandatory.
    declared_absences:
        Claim categories carrying a typed, structurally validated absence state, mapped
        to the evidence IDs that state cites. An absence state is a positive,
        evidence-carrying assertion that a fact is not there, so it discharges coverage
        for its own claim; every ID it cites must be a real validated excerpt. Absence
        of the record never discharges anything -- a bare empty value still fails.
    locator_authority:
        Locator-adjudication era for this validation pass (see
        :func:`_validate_locator`). The default is the recorded legacy basis;
        the LIVE author envelope stamps :data:`LOCATOR_AUTHORITY_TEXT_LINE`
        from the introducing epoch onward, and recorded results keep
        rehydrating at their own basis -- history is never reinterpreted.

    Returns
    -------
    EvidenceValidationReport
        Validated support summary.

    Raises
    ------
    EvidenceValidationError
        If an excerpt, locator, source, hash, or coverage assertion is invalid.
    """

    excerpts = evidence.get("excerpts")
    if not isinstance(excerpts, list):
        raise EvidenceValidationError("evidence.excerpts must be a list")
    sources = _source_index(source_manifest)
    supported: set[str] = set()
    evidence_ids: set[str] = set()
    family_grounded = False
    for excerpt in excerpts:
        if not isinstance(excerpt, Mapping):
            raise EvidenceValidationError("every evidence excerpt must be an object")
        evidence_id = _nonempty(excerpt.get("evidence_id"), "evidence_id")
        if evidence_id in evidence_ids:
            raise EvidenceValidationError(f"duplicate evidence_id: {evidence_id}")
        evidence_ids.add(evidence_id)
        source_id = _nonempty(excerpt.get("source_id"), f"{evidence_id}.source_id")
        source = sources.get(source_id)
        if source is None:
            raise EvidenceValidationError(f"{evidence_id} references unknown source {source_id}")
        locator = _nonempty(excerpt.get("locator"), f"{evidence_id}.locator")
        text = _nonempty(excerpt.get("text"), f"{evidence_id}.text")
        actual_text_hash = hash_bytes(text.encode("utf-8"))
        if excerpt.get("text_sha256") != actual_text_hash:
            raise EvidenceValidationError(
                f"{evidence_id} text_sha256 does not match the verbatim UTF-8 bytes"
            )
        supports = excerpt.get("supports")
        if (
            not isinstance(supports, list)
            or not supports
            or not all(isinstance(value, str) and value.strip() for value in supports)
        ):
            raise EvidenceValidationError(f"{evidence_id}.supports must be non-empty strings")
        content = _read_source(source, cas_root)
        _validate_locator(
            evidence_id, locator, text.encode("utf-8"), content, authority=locator_authority
        )
        supported.update(supports)
        family_level = excerpt.get("family_level")
        if not isinstance(family_level, bool):
            raise EvidenceValidationError(f"{evidence_id}.family_level must be boolean")
        family_grounded = family_grounded or family_level

    absence_covered = _absence_coverage(declared_absences, evidence_ids)
    required = {claim for claim in required_claims if claim}
    missing = required - supported - absence_covered
    if missing:
        raise EvidenceValidationError(f"ungrounded claim categories: {sorted(missing)}")
    coverage = evidence.get("coverage")
    if not isinstance(coverage, Mapping):
        raise EvidenceValidationError("evidence.coverage must be an object")
    declared_missing = coverage.get("missing_support")
    if declared_missing != [] or coverage.get("all_agent_fields_have_support") is not True:
        raise EvidenceValidationError("evidence coverage does not declare complete support")
    if require_family_grounding and not family_grounded:
        raise EvidenceValidationError("family-level grounding is required")
    if coverage.get("family_grounding_complete") is True and not family_grounded:
        raise EvidenceValidationError(
            "family grounding is declared complete without a family excerpt"
        )
    return EvidenceValidationReport(
        len(excerpts), frozenset(supported), family_grounded, absence_covered
    )


def _absence_coverage(
    declared_absences: Optional[Mapping[str, Sequence[str]]],
    known_evidence_ids: AbstractSet[str],
) -> frozenset[str]:
    """Return claim categories discharged by a typed absence state.

    Parameters
    ----------
    declared_absences:
        Claim category to the evidence IDs its typed absence state cites.
    known_evidence_ids:
        Evidence IDs of excerpts that validated in this same pass.

    Returns
    -------
    frozenset[str]
        Claim categories a typed absence state covers.

    Raises
    ------
    EvidenceValidationError
        If a declared absence is malformed or cites an evidence ID that no validated
        excerpt provides. A fabricated corroboration must never buy coverage.
    """

    if not declared_absences:
        return frozenset()
    covered: set[str] = set()
    for claim, cited in declared_absences.items():
        if not isinstance(claim, str) or not claim.strip():
            raise EvidenceValidationError("declared absence claim names must be non-empty strings")
        # A bare string is a Sequence whose iteration yields characters; require a real
        # list/tuple so a single ID never silently decomposes into per-character IDs.
        if isinstance(cited, (str, bytes)) or not isinstance(cited, Sequence):
            raise EvidenceValidationError(
                f"declared absence for {claim} must carry a list of evidence IDs"
            )
        unknown = [
            evidence_id
            for evidence_id in cited
            if not isinstance(evidence_id, str) or evidence_id not in known_evidence_ids
        ]
        if unknown:
            raise EvidenceValidationError(
                f"declared absence for {claim} cites missing or fabricated evidence: "
                f"{sorted(map(str, unknown))}"
            )
        covered.add(claim)
    return frozenset(covered)


def evidence_ids(evidence: Mapping[str, Any]) -> frozenset[str]:
    """Return all unique evidence identifiers.

    Parameters
    ----------
    evidence:
        Model evidence block.

    Returns
    -------
    frozenset[str]
        Exact excerpt identifiers.

    Raises
    ------
    EvidenceValidationError
        If an identifier is missing or duplicated.
    """

    excerpts = evidence.get("excerpts", [])
    if not isinstance(excerpts, list):
        raise EvidenceValidationError("evidence.excerpts must be a list")
    result: set[str] = set()
    for excerpt in excerpts:
        if not isinstance(excerpt, Mapping):
            raise EvidenceValidationError("every evidence excerpt must be an object")
        value = _nonempty(excerpt.get("evidence_id"), "evidence_id")
        if value in result:
            raise EvidenceValidationError(f"duplicate evidence_id: {value}")
        result.add(value)
    return frozenset(result)


def fetched_sources_for_checked_links(
    search_report: Mapping[str, Any],
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> tuple[Mapping[str, Any], ...]:
    """Bind every checked search-report link to one fetched CAS source.

    Parameters
    ----------
    search_report:
        Bounded source search whose ``links_checked`` entries claim inspected content.
    source_manifest:
        Exact controlled-fetch manifest for the author session.

    Returns
    -------
    tuple[Mapping[str, Any], ...]
        Manifest rows in checked-link order, ready for deterministic inventory.

    Raises
    ------
    EvidenceValidationError
        If a checked link was withheld from controlled fetch, is ambiguous, or did
        not produce an inspectable fetched/already-present CAS object.
    """

    links = search_report.get("links_checked")
    if (
        not isinstance(links, list)
        or not links
        or not all(_checked_link_url(link) is not None for link in links)
    ):
        raise EvidenceValidationError("search_report.links_checked must be non-empty URLs")
    link_urls = tuple(str(_checked_link_url(link)) for link in links)
    if len(link_urls) != len(set(link_urls)):
        raise EvidenceValidationError("search_report.links_checked contains duplicate URLs")

    sources_by_url: dict[str, list[Mapping[str, Any]]] = {}
    for source in _source_index(source_manifest).values():
        url = source.get("url")
        if isinstance(url, str) and url:
            sources_by_url.setdefault(url, []).append(source)
    receipts_by_url = _broker_receipts_by_url(source_manifest)

    bound: list[Mapping[str, Any]] = []
    for link in links:
        if isinstance(link, Mapping):
            _validate_receipt_bound_checked_link(link, receipts_by_url)
            continue
        assert isinstance(link, str)
        matches = sources_by_url.get(link, [])
        if len(matches) != 1:
            raise EvidenceValidationError(
                f"checked search link must bind to exactly one controlled-fetch source: {link}"
            )
        source = matches[0]
        if source.get("retrieval_status") not in {"fetched", "already-present"}:
            raise EvidenceValidationError(
                f"checked search link has no fetched CAS inventory: {link}"
            )
        if not isinstance(source.get("content_sha256"), str):
            raise EvidenceValidationError(
                f"checked search link has no hash-bound fetched bytes: {link}"
            )
        bound.append(source)
    return tuple(bound)


def _checked_link_url(link: object) -> Optional[str]:
    """Return the URL named by one ``links_checked`` entry, if well-shaped.

    Parameters
    ----------
    link:
        Raw checked-link entry from an author search report.

    Returns
    -------
    str | None
        Non-empty checked URL, or ``None`` when the entry is structurally invalid.
    """

    if isinstance(link, str) and link.strip():
        return link
    if not isinstance(link, Mapping):
        return None
    if set(link) != {"url", "disposition"}:
        return None
    url = link.get("url")
    disposition = link.get("disposition")
    if (
        isinstance(url, str)
        and url.strip()
        and isinstance(disposition, str)
        and disposition in CHECKED_LINK_RECEIPT_OUTCOMES_BY_DISPOSITION
    ):
        return url
    return None


def _broker_receipts_by_url(
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Index broker outcome receipts by their requested or final URL.

    Parameters
    ----------
    source_manifest:
        Manifest wrapper that may carry ``broker.outcomes``.

    Returns
    -------
    dict[str, list[Mapping[str, Any]]]
        Broker outcome rows keyed by URL.
    """

    if not isinstance(source_manifest, Mapping):
        return {}
    broker = source_manifest.get("broker")
    outcomes = broker.get("outcomes") if isinstance(broker, Mapping) else None
    if not isinstance(outcomes, list):
        return {}
    indexed: dict[str, list[Mapping[str, Any]]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            continue
        for field in ("url", "final_url"):
            url = outcome.get(field)
            if isinstance(url, str) and url:
                indexed.setdefault(url, []).append(outcome)
    return indexed


def _validate_receipt_bound_checked_link(
    link: Mapping[str, Any],
    receipts_by_url: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Require a typed checked link disposition to be backed by a broker receipt.

    Parameters
    ----------
    link:
        Structured ``links_checked`` entry.
    receipts_by_url:
        Broker outcomes indexed by URL.

    Raises
    ------
    EvidenceValidationError
        If no matching broker receipt proves the non-byte checked-link outcome.
    """

    url = str(link["url"])
    disposition = str(link["disposition"])
    receipt_outcomes = CHECKED_LINK_RECEIPT_OUTCOMES_BY_DISPOSITION.get(disposition)
    if receipt_outcomes is None:
        raise EvidenceValidationError(f"checked search link disposition is unsupported: {url}")
    matches = [
        receipt
        for receipt in receipts_by_url.get(url, ())
        if receipt.get("outcome") in receipt_outcomes
    ]
    if len(matches) != 1:
        raise EvidenceValidationError(
            f"checked search link disposition lacks a matching broker receipt: {url}"
        )


def _source_index(
    source_manifest: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
) -> dict[str, Mapping[str, Any]]:
    """Index source manifests by source identifier.

    Parameters
    ----------
    source_manifest:
        Wrapper or direct manifest sequence.

    Returns
    -------
    dict[str, Mapping[str, Any]]
        Source identifier lookup.
    """

    raw_sources: object
    if isinstance(source_manifest, Mapping):
        raw_sources = source_manifest.get("sources")
        if raw_sources is None and "source_id" in source_manifest:
            raw_sources = [source_manifest]
    else:
        raw_sources = source_manifest
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise EvidenceValidationError("source manifest must contain a source list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for source in raw_sources:
        if not isinstance(source, Mapping):
            raise EvidenceValidationError("every source manifest must be an object")
        source_id = _nonempty(source.get("source_id"), "source.source_id")
        if source_id in indexed:
            raise EvidenceValidationError(f"duplicate source_id: {source_id}")
        indexed[source_id] = source
    return indexed


def _read_source(source: Mapping[str, Any], cas_root: Union[str, Path, None]) -> bytes:
    """Read and re-verify exact source bytes from the CAS.

    Parameters
    ----------
    source:
        Source manifest row.
    cas_root:
        Governing CAS root. When provided, resolution is digest-derived under
        this root and the row's recorded ``cas_path`` is provenance metadata
        only -- it is never dereferenced, so a manifest frozen on another host
        (or against a since-pruned working tree) stays readable wherever its
        CAS objects were relocated. When absent, the recorded path is the only
        available authority and current behavior stands.

    Returns
    -------
    bytes
        Verified source bytes.

    Raises
    ------
    EvidenceValidationError
        If content is absent or no longer hash-bound.
    """

    digest = source.get("content_sha256")
    if not isinstance(digest, str):
        raise EvidenceValidationError("source content_sha256 is missing")
    override = source.get("unpromoted_read_path")
    if isinstance(override, str) and override:
        # In-process grounding channel for bytes a trusted caller holds OUTSIDE
        # the CAS: the executor's pre-publication replay reads supplement bytes
        # from the attempt's broker evidence directory before the driver
        # promotes them. The key is constructed per-run and never persisted in
        # artifacts, and the rehash below refuses wrong bytes regardless.
        path = Path(override)
    elif cas_root is not None:
        try:
            path = cas_path(cas_root, digest)
        except UnpinnedTargetError as exc:
            raise EvidenceValidationError(
                f"source content_sha256 is not a canonical digest: {digest}"
            ) from exc
    else:
        path_value = source.get("cas_path")
        if isinstance(path_value, str) and path_value:
            path = Path(path_value)
        else:
            raise EvidenceValidationError("source manifest has no CAS path")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise EvidenceValidationError(f"cannot read source CAS object {path}: {exc}") from exc
    if hash_bytes(content) != digest:
        raise EvidenceValidationError(f"source CAS object does not match {digest}")
    return content


def classify_locator(locator: str) -> str:
    """Classify one locator string into the closed three-form grammar.

    Parameters
    ----------
    locator:
        Verbatim locator string.

    Returns
    -------
    str
        :data:`LOCATOR_FORM_BYTES`, :data:`LOCATOR_FORM_LINE`, or
        :data:`LOCATOR_FORM_MEMBERSHIP`. The grammar is closed by enumeration
        (fable A-2): the only positional-claim-bearing forms are ``bytes:``
        and a textual line claim; everything else asserts membership only.
    """

    if locator.startswith("bytes:"):
        return LOCATOR_FORM_BYTES
    if _LINE_LOCATOR_PATTERN.search(locator):
        return LOCATOR_FORM_LINE
    return LOCATOR_FORM_MEMBERSHIP


def parse_line_locator(locator: str) -> Optional[tuple[int, int]]:
    """Parse one textual line locator's positional claim, fail-closed.

    Parameters
    ----------
    locator:
        Verbatim locator string whose form is :data:`LOCATOR_FORM_LINE`.

    Returns
    -------
    Optional[tuple[int, int]]
        ``(start_line, end_line)`` -- 1-based inclusive -- or ``None`` when
        the positional claim is ambiguous (more than one line claim in one
        locator) or degenerate (``end < start`` or ``start < 1``). ``None``
        is a REFUSAL under the strict authority, never a demotion to
        membership semantics (Sol H-2.1).
    """

    matches = _LINE_LOCATOR_PATTERN.findall(locator)
    if len(matches) != 1:
        return None
    start_text, end_text = matches[0]
    start = int(start_text)
    end = int(end_text) if end_text else start
    if start < 1 or end < start:
        return None
    return start, end


def split_universal_lines(content: bytes) -> list[str]:
    """Split exact source bytes into lines under the declared basis.

    Parameters
    ----------
    content:
        Exact fetched source bytes (any byte sequence; invalid UTF-8 is
        carried through surrogateescape, never dropped).

    Returns
    -------
    list[str]
        Lines under :data:`TEXT_LINE_LOCATOR_BASIS`: CRLF, bare CR, and LF
        all terminate a line; the final unterminated line counts; no other
        codepoint splits.
    """

    return _UNIVERSAL_NEWLINE_PATTERN.split(content.decode("utf-8", "surrogateescape"))


def _line_window(content: bytes, start: int, end: int) -> Optional[bytes]:
    """Return the exact declared 1-based inclusive line window, or ``None``.

    ``None`` means the declared range does not exist in the source (past the
    last line), which refutes the positional claim outright.
    """

    lines = split_universal_lines(content)
    if end > len(lines):
        return None
    return "\n".join(lines[start - 1 : end]).encode("utf-8", "surrogateescape")


def excerpt_in_line_window(text: bytes, content: bytes, start: int, end: int) -> bool:
    """Adjudicate one line-form positional claim, zero tolerance.

    The claim holds iff the excerpt -- folded under the pipeline's one
    documented equivalence (:func:`_fold_space`, then outer-whitespace
    accommodation) -- is contained in EXACTLY the declared window's folded
    bytes. No adjacent-line slack: a heading one line above the content
    (m10470's fabricated ``line 19``) fails, while a CRLF/LF newline-flavor
    difference (m10759 ``ev-requirements-base``) passes, because whitespace
    codepoint identity is inside the fold by design.

    Parameters
    ----------
    text:
        Claimed verbatim excerpt bytes.
    content:
        Exact fetched source bytes.
    start, end:
        Declared 1-based inclusive line range.

    Returns
    -------
    bool
        Whether the excerpt sits inside exactly the declared window.
    """

    window = _line_window(content, start, end)
    if window is None:
        return False
    folded = _fold_space(text).strip()
    return bool(folded) and folded in _fold_space(window)


def _validate_locator(
    evidence_id: str,
    locator: str,
    text: bytes,
    content: bytes,
    *,
    authority: str = LOCATOR_AUTHORITY_LEGACY_MEMBERSHIP,
) -> None:
    """Validate an excerpt against an exact byte locator or source membership.

    Parameters
    ----------
    evidence_id:
        Excerpt identifier used in errors.
    locator:
        Locator string. ``bytes:start-end`` receives strict byte-range checking.
    text:
        Claimed verbatim excerpt bytes.
    content:
        Exact fetched source bytes.
    authority:
        Locator-authority era. Under
        :data:`LOCATOR_AUTHORITY_LEGACY_MEMBERSHIP` (the default, and the
        recorded basis every historical author result rehydrates at) a
        textual locator's line number is not checked -- membership only.
        Under :data:`LOCATOR_AUTHORITY_TEXT_LINE` (stamped by the LIVE
        author envelope from the introducing epoch onward) a line-form
        locator is positionally adjudicated with zero tolerance, and an
        unparseable or ambiguous positional claim is refused rather than
        demoted to membership.

    Raises
    ------
    EvidenceValidationError
        If the excerpt is absent, differs at the byte locator, or (strict
        authority) fails its positional line claim.
    """

    if authority == LOCATOR_AUTHORITY_TEXT_LINE and classify_locator(locator) == LOCATOR_FORM_LINE:
        parsed = parse_line_locator(locator)
        if parsed is None:
            raise EvidenceValidationError(
                f"{evidence_id} locator {locator!r} carries an ambiguous or degenerate "
                "line claim; a positional locator that cannot be parsed is refused, "
                "never read as membership"
            )
        start, end = parsed
        if not excerpt_in_line_window(text, content, start, end):
            raise EvidenceValidationError(
                f"{evidence_id} excerpt is not inside its declared line window "
                f"{start}-{end} ({TEXT_LINE_LOCATOR_BASIS}); a perfect excerpt at a "
                "stale line is a locator failure"
            )
        return
    if locator.startswith("bytes:"):
        try:
            start_text, end_text = locator.removeprefix("bytes:").split("-", 1)
            start, end = int(start_text), int(end_text)
        except (ValueError, TypeError) as exc:
            raise EvidenceValidationError(f"{evidence_id} has an invalid byte locator") from exc
        if start < 0 or end < start or not _same_text(content[start:end], text):
            raise EvidenceValidationError(f"{evidence_id} does not exist verbatim at {locator}")
    elif text not in content:
        # The fold erases which whitespace character sits inside the excerpt; this
        # strip erases whether whitespace sits AROUND it. Line-based extraction
        # tooling appends a trailing newline to whatever it prints (``sed -n 'Np'``
        # ends every emission with one), so an author quoting a 500-byte run out of
        # the middle of a single-line page reproduces every visible byte exactly and
        # still carries one invisible trailing ``\n`` the source does not have at
        # that position. That refusal burned the model's one authoring attempt
        # (menagerie campaign ``pilot``, model ``m4066``, Implicit Q-Learning:
        # excerpt ``ev-arxiv-meta``, byte-identical for its whole visible length).
        #
        # Only the OUTER whitespace of the whole excerpt is dropped, after the fold
        # has already run on both sides. Every visible character, its order, and all
        # interior spacing (modulo the fold's existing run-identity rule) still
        # must match, so an excerpt whose quoted content differs from the source
        # anywhere something is displayed is refused exactly as before. An excerpt
        # that strips to nothing quotes nothing and is refused rather than matched
        # against every source trivially.
        folded = _fold_space(text).strip()
        if not folded or folded not in _fold_space(content):
            raise EvidenceValidationError(
                f"{evidence_id} excerpt is not verbatim in its fetched source"
            )


def _attest_read_source(source: Mapping[str, Any], cas_root: Union[str, Path, None]) -> Optional[bytes]:
    """Best-effort hash-verified source read for attestation derivation.

    Resolution order: digest-derived under ``cas_root`` when given, else the
    row's recorded ``cas_path``. Every read is rehashed; wrong bytes are as
    unavailable as no bytes. Returns ``None`` instead of raising because the
    attestation's job is to DISCLOSE unattestable rows, not to refuse the
    envelope that carries them (the component-drift adverse golden cites
    sources its manifest never froze -- the attestation is the machine
    evidence of exactly that).
    """

    digest = source.get("content_sha256")
    if not isinstance(digest, str) or not digest:
        return None
    candidates: list[Path] = []
    if cas_root is not None:
        try:
            candidates.append(cas_path(cas_root, digest))
        except UnpinnedTargetError:
            return None
    recorded = source.get("cas_path")
    if isinstance(recorded, str) and recorded:
        candidates.append(Path(recorded))
    for candidate in candidates:
        try:
            content = candidate.read_bytes()
        except OSError:
            continue
        if hash_bytes(content) == digest:
            return content
    return None


def attest_excerpt_locators(
    excerpts: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    *,
    cas_root: Union[str, Path, None] = None,
) -> list[dict[str, Any]]:
    """Derive the typed per-excerpt machine locator attestation (Sol H-2.1).

    One :data:`TEXT_LINE_LOCATOR_SCHEMA` row per excerpt, in excerpt order,
    coordinator-derived and never authored: the same idiom as
    ``env-routed-inventory``. The row is the ONLY admissible basis for a
    checker locator finding -- a checker disputing a ``valid`` row is a typed
    authority conflict, never a member verdict.

    Parameters
    ----------
    excerpts:
        Excerpt rows carrying ``evidence_id``, ``source_id``, ``locator``,
        and ``text`` (proposal evidence excerpts and terminal evidence-pack
        excerpts share this projection).
    source_manifest:
        Manifest wrapper containing ``sources`` or a direct source list.
    cas_root:
        Preferred digest-derived CAS root; the recorded ``cas_path`` is the
        fallback. Every read is rehashed.

    Returns
    -------
    list[dict[str, Any]]
        Attestation rows with closed ``status`` values
        (:data:`LOCATOR_ATTESTATION_VALID` / ``invalid`` / ``unattestable``)
        and a human-readable ``detail`` naming every non-valid cause.
    """

    sources = _source_index(source_manifest)
    rows: list[dict[str, Any]] = []
    content_cache: dict[str, Optional[bytes]] = {}
    for excerpt in excerpts:
        if not isinstance(excerpt, Mapping):
            continue
        evidence_id = str(excerpt.get("evidence_id") or "")
        source_id = str(excerpt.get("source_id") or "")
        locator = str(excerpt.get("locator") or "")
        text = excerpt.get("text")
        form = classify_locator(locator)
        row: dict[str, Any] = {
            "schema": TEXT_LINE_LOCATOR_SCHEMA,
            "evidence_id": evidence_id,
            "source_id": source_id,
            "locator": locator,
            "form": form,
            "basis": TEXT_LINE_LOCATOR_BASIS if form == LOCATOR_FORM_LINE else None,
            "equivalence": TEXT_LINE_LOCATOR_EQUIVALENCE,
            "start_line": None,
            "end_line": None,
            "source_content_sha256": None,
            "excerpt_text_sha256": None,
            "status": LOCATOR_ATTESTATION_UNATTESTABLE,
            "detail": None,
        }
        rows.append(row)
        if not evidence_id or not locator or not isinstance(text, str) or not text:
            row["detail"] = "excerpt row is structurally incomplete"
            continue
        text_bytes = text.encode("utf-8")
        row["excerpt_text_sha256"] = hash_bytes(text_bytes)
        source = sources.get(source_id)
        if source is None:
            row["detail"] = f"excerpt cites source {source_id!r} absent from the frozen manifest"
            continue
        digest = source.get("content_sha256")
        row["source_content_sha256"] = digest if isinstance(digest, str) else None
        cache_key = source_id
        if cache_key not in content_cache:
            content_cache[cache_key] = _attest_read_source(source, cas_root)
        content = content_cache[cache_key]
        if content is None:
            row["detail"] = "frozen source bytes are unavailable or no longer hash-bound"
            continue
        if form == LOCATOR_FORM_LINE:
            parsed = parse_line_locator(locator)
            if parsed is None:
                row["status"] = LOCATOR_ATTESTATION_INVALID
                row["detail"] = "ambiguous or degenerate line claim"
                continue
            row["start_line"], row["end_line"] = parsed
            if excerpt_in_line_window(text_bytes, content, *parsed):
                row["status"] = LOCATOR_ATTESTATION_VALID
            else:
                row["status"] = LOCATOR_ATTESTATION_INVALID
                row["detail"] = (
                    f"excerpt is not inside declared line window {parsed[0]}-{parsed[1]}"
                )
            continue
        try:
            _validate_locator(evidence_id, locator, text_bytes, content)
        except EvidenceValidationError as exc:
            row["status"] = LOCATOR_ATTESTATION_INVALID
            row["detail"] = str(exc)
        else:
            row["status"] = LOCATOR_ATTESTATION_VALID
    return rows


#: One run of whitespace. Python's ``\s`` over ``str`` is Unicode-aware: it covers every
#: character in category ``Zs`` (U+0020, U+00A0, U+2002-U+200A, U+202F, U+3000, ...) plus
#: the ASCII whitespace controls, and deliberately does NOT cover U+200B ZERO WIDTH SPACE,
#: which is a format character (``Cf``) rather than a separator. That exclusion matters: a
#: zero-width character is invisible but occupies no rendered gap, so folding it away would
#: let an excerpt differ from its source somewhere nothing is displayed. Compiled over
#: decoded text rather than raw bytes because the distinction erased is codepoint-level.
_SPACE_RUN_PATTERN = re.compile(r"\s+")


def _fold_space(value: bytes) -> bytes:
    """Canonicalize every run of Unicode whitespace to one ASCII space.

    A fetched page and the excerpt an author transcribes out of it can differ in
    exactly one respect that carries no meaning: *which* space character sits between
    two runs of visible text. ar5iv renders LaTeX inter-author spacing as U+2003 EM
    SPACE and non-breaking gaps as U+00A0, and an author reading that page writes the
    ordinary U+0020 it renders as. Comparing raw bytes then refuses a transcription
    that reproduced 150 characters of markup -- including exact generated ``id``
    attribute values no one could invent -- over a single invisible codepoint, and
    burns the model's one authoring attempt (menagerie campaign ``pilot``, model
    ``m8245``, MetaFormer/PoolFormer, excerpt ``ev-ar5iv-affiliation``).

    The fold is applied identically to both sides and erases *only* the identity of
    whitespace characters. It can never join two tokens, because a whitespace run
    always folds to one space rather than to nothing, and it can never split one,
    because no space is ever introduced. Every visible character, its order, and every
    token boundary therefore survive untouched: an excerpt naming text the source does
    not contain is refused exactly as before. This is the same equivalence
    :func:`menagerie.crawler.proposal._normalize_support_text` already applies through
    its NFKD fold, which maps these separators to U+0020 -- so the raw-byte check was
    the one place in the pipeline that still treated them as distinct.

    Parameters
    ----------
    value:
        Excerpt or source bytes to canonicalize.

    Returns
    -------
    bytes
        UTF-8 bytes with every whitespace run replaced by a single ASCII space.
    """

    # Deliberately NOT a full NFKC/NFKD pass. Those forms also fold ligatures, full-width
    # forms, and compatibility digits, which are visible-character differences an excerpt
    # must still reproduce exactly. Only the whitespace class is canonicalized here.
    decoded = value.decode("utf-8", "surrogateescape")
    return _SPACE_RUN_PATTERN.sub(" ", decoded).encode("utf-8", "surrogateescape")


def _same_text(content: bytes, text: bytes) -> bool:
    """Return whether two byte runs agree exactly or differ only in space characters."""

    return content == text or _fold_space(content) == _fold_space(text)


def _nonempty(value: object, field: str) -> str:
    """Return a stripped non-empty string.

    Parameters
    ----------
    value:
        Candidate field value.
    field:
        Field name used in errors.

    Returns
    -------
    str
        Original non-empty string.

    Raises
    ------
    EvidenceValidationError
        If the value is not a non-empty string.
    """

    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{field} must be a non-empty string")
    return value
