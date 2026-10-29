"""Track A tamper-evident hash chains for close-relevant ledgers.

The chain is derived from each row's BODY: the content address of a row is
``stable_hash`` over the row minus its position-bound ``chain`` envelope and
minus the self-referential digest fields a writer stamps over that same body
(:data:`menagerie.crawler.identity.ROW_CONTENT_ADDRESS_EXCLUDED`).  New machinery
rows additionally persist their link, while historical rows remain a verified
legacy prefix.  Durable close, leg-event, and registry rows anchor the heads used
by the close-licensing gate.

The address is DERIVED, never SELECTED
--------------------------------------
Until PSLRE5 this module read the address out of the first self-declared digest
field a row happened to carry -- ``close_sha256``, then ``row_sha256``, then
``record_revision``, then ``payload_sha256``.  That made the address a string the
row asserts about itself: adding ONE foreign digest field pinned the chain to an
arbitrary value while the rest of the body was rewritten freely, on every one of
the twenty chain scopes (money rows, close rows, leg events, the rung registry,
and every census source).  The invariant now has exactly one spelling:

    A row's chain content address must be computable by a reader that believes
    nothing the row says about itself.

Equivalently: for any row, any field name ``k`` and any value ``v``, adding
``k=v`` may never preserve the address across a change to any other field.  That
holds for ALL names at once because no field is read -- the body is hashed, and
``k`` is part of the body.

Fused re-certification
----------------------
Because the four digest names are excluded from the preimage, editing one of them
does not move the address -- and consumers DO key on those fields (the close-head
index, the same-campaign predecessor spine, award revision references).  So the
same chokepoint re-certifies them: any PRESENT member of
:data:`FUSED_ROW_DIGEST_FIELDS` that does not equal the body derivation refuses
typed.  Honest rows are unaffected (every family already stamps exactly the body
hash), so the declared digest is evidence again without reintroducing selection.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.close_reads import (
    DECODED_ROW_WEIGHT_FACTOR,
    HEAD_DIGEST_WEIGHT,
    PROJECTION_BASE_WEIGHT,
    LedgerRow,
    derive,
    read_epoch,
)
from menagerie.crawler.identity import (
    ROW_CONTENT_ADDRESS_EXCLUDED,
    canonical_json_bytes,
    is_sha256,
    row_content_address,
    stable_hash,
)
from menagerie.crawler.models import JsonObject

LEDGER_CHAIN_VERSION = "menagerie-ledger-chain-v1"

#: The self-digest fields the chain RE-CERTIFIES against its own derivation.
#: DERIVED from the one exclusion set rather than restated, so the certified set
#: and the unhashed region can never drift apart: every name excluded from the
#: address (other than the position-bound envelope itself) is a name a consumer
#: might key on as "this row's digest", and is therefore either equal to the body
#: hash or a typed refusal.  Adding a name to the exclusion set automatically
#: brings it under certification.
FUSED_ROW_DIGEST_FIELDS: tuple[str, ...] = tuple(
    sorted(ROW_CONTENT_ADDRESS_EXCLUDED - {"chain"})
)

#: The CLOSED key vocabulary of a persisted chain envelope.  ``_verified_link``
#: checks ``seq``/``prev``/``sha256``; anything else inside the envelope used to be
#: covered by NEITHER the row digest (which excludes ``chain``) NOR the link check
#: -- an unhashed, unchecked free-form region on every chained ledger, i.e. a
#: laundering channel (PSLRE5 acct observation).  The envelope is machinery, so its
#: vocabulary is exact.
CHAIN_ENVELOPE_FIELDS = frozenset({"seq", "prev", "sha256"})


def chain_envelope_value_findings(envelope: Mapping[str, Any]) -> tuple[str, ...]:
    """Return why one chain envelope's VALUES are not machinery-shaped, if they are not.

    Closing the envelope's KEY vocabulary retired the extra-KEY spelling of the
    laundering channel, not the extra-BYTES property (PSLRE6 D6-5). ``chain`` is
    excluded from the content address, so nothing inside it is hashed, and the link
    check compares its members NUMERICALLY/by equality rather than by shape -- so
    ``"seq": 0.000...0`` with 240 zeros, ``-0.0``, ``0e0`` and even ``false`` all
    satisfy ``chain["seq"] == 0`` while carrying unbounded unauthenticated bytes.

    The envelope is machinery, so its values have a SHAPE: ``seq`` is a non-negative
    ``int`` (never a ``bool``, which Python counts as one) and the two links are
    canonical digests. Asserting the shape closes the channel for every row whether or
    not an equality check also applies to it, so the residual's "no consumer today"
    cannot quietly become a consumer tomorrow.

    Parameters
    ----------
    envelope:
        Persisted ``chain`` envelope, already key-closed.

    Returns
    -------
    tuple[str, ...]
        Findings, empty when every value conforms.
    """

    findings: list[str] = []
    sequence = envelope.get("seq")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        findings.append("seq must be a non-negative integer row index")
    for name in ("prev", "sha256"):
        value = envelope.get(name)
        if not isinstance(value, str) or not is_sha256(value):
            findings.append(f"{name} must be one canonical sha256: digest")
    return tuple(findings)


#: Permanent scale-cost instrumentation at the UNBYPASSABLE boundary: every
#: spelling of "chain-verify a row" runs through :func:`_verified_link`, and every
#: spelling of "decode a ledger row" runs through :func:`decode_payload` or
#: :func:`_decode_row`.  A cost census built on a consumer's own incremental
#: counters can only measure the paths that consumer instrumented, so an
#: uninstrumented whole-history re-verify inside a money append shipped O(history)
#: work per append with the census GREEN (PSLRE4 D-3).  Counted here, that
#: regression is measured no matter which entry point spells it.  Counting is the
#: only mutable module state, it is write-only from this module, and no decision
#: anywhere reads it.
LEDGER_CHAIN_STATS: dict[str, int] = {"rows_linked": 0, "rows_decoded": 0}


def reset_ledger_chain_stats() -> None:
    """Zero the chain-verification counters before one measurement window."""

    for key in LEDGER_CHAIN_STATS:
        LEDGER_CHAIN_STATS[key] = 0


class LedgerChainError(ValueError):
    """Raised when a ledger does not reproduce its committed history."""


class _DuplicateLedgerKey(ValueError):
    """Raised while decoding a ledger line that repeats an object key."""


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    """Build one decoded JSON object, refusing a repeated key.

    ``json.loads`` keeps the LAST occurrence of a repeated key, so eight extra copies
    of ``"campaign_id"`` decode to the identical row -- identical content address,
    identical chain head -- while adding unbounded unauthenticated bytes to the line
    (PSLRE6 D6-5, third spelling). The canonical writer never emits a duplicate, so
    refusing one costs nothing honest and removes the byte donor.
    """

    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateLedgerKey(f"duplicate object key {key!r}")
        seen[key] = value
    return seen


def _loads_strict(line: bytes | str) -> object:
    """Decode one ledger line with the duplicate-key refusal installed."""

    return json.loads(line, object_pairs_hook=_no_duplicate_keys)


def _strict_object(value: object, *, path: Path, line_number: int) -> JsonObject:
    """Return one decoded ledger row, or refuse TYPED for anything else.

    THE strict decoder's whole judgement, in one place, so the streaming reader
    (:func:`_decode_row`) and the whole-bytes reader (:func:`decode_payload`) cannot
    drift in what they accept or how they name a refusal.

    ``json.loads`` is more permissive than this module's canonicalization: it accepts
    the ``NaN``/``Infinity``/``-Infinity`` literals, any overflowing exponent
    (``1e400`` decodes to ``inf``), and ``\\ud800``-style lone surrogates, none of
    which :func:`menagerie.crawler.identity.canonical_json_bytes` can serialize.
    Before PSLRE6 D6-2 such a line was ACCEPTED here and then died deep inside the
    address derivation as a bare ``ValueError``/``UnicodeEncodeError``, so one
    appended line turned every Track A integrity check -- the chain, the close
    verifier, the whole ``verify_records_integrity`` spine -- into an UNTYPED
    exception that no ``except LedgerChainError`` handler catches. The declared
    contract is that a complete malformed line is hard corruption and refuses typed,
    so the decoder is made actually strict: a row this decoder returns is a row the
    chain can address, proven by round-tripping it through the one canonicalization
    rather than by enumerating the spellings that break it.

    Parameters
    ----------
    value:
        Result of ``json.loads`` on one ledger line.
    path:
        Ledger path, for diagnostics.
    line_number:
        One-based line number, for diagnostics.

    Returns
    -------
    JsonObject
        The decoded row.

    Raises
    ------
    LedgerChainError
        If the line is not a JSON object, or is not canonically serializable.
    """

    if not isinstance(value, Mapping):
        raise LedgerChainError(f"{path}:{line_number} is not a JSON object")
    row = dict(value)
    try:
        canonical_json_bytes(row)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise LedgerChainError(
            f"{path}:{line_number} is not canonically serializable and therefore has "
            f"no content address: {type(exc).__name__}: {exc}"
        ) from exc
    return row


@dataclass(frozen=True)
class ChainState:
    """One verified ledger-chain extent.

    Parameters
    ----------
    rows:
        Number of non-empty rows covered.
    head_sha256:
        Chain head after those rows.
    heads:
        Head after each row, in ledger order.
    """

    rows: int
    head_sha256: str
    heads: tuple[str, ...]


def _genesis(scope: str) -> str:
    """Return the deterministic genesis head for one canonical scope."""

    return stable_hash({"chain_version": LEDGER_CHAIN_VERSION, "scope": scope, "genesis": True})


def row_content_digest(row: Mapping[str, Any]) -> str:
    """Return the BODY-DERIVED content address of one ledger row.

    One derivation for every scope: ``stable_hash`` of the row minus
    :data:`menagerie.crawler.identity.ROW_CONTENT_ADDRESS_EXCLUDED`.  Nothing is
    read from the row about the row, so no field of any name can hold an address
    across a body edit -- the property the deleted preference ladder could only
    ever have for the names someone remembered to ban.

    Every honest family already stamped exactly this value (each writer hashes its
    own body minus its own digest, with one canonicalization tree-wide), so the
    derivation reproduces every historical chain head, close pin, registry anchor,
    and paid anchor byte-for-byte.  It diverges on exactly one shape: a row whose
    declared digest is not the hash of its own body.

    Parameters
    ----------
    row:
        Decoded ledger row.

    Returns
    -------
    str
        The row's content address.
    """

    return row_content_address(row)


def _certified_content_address(row: Mapping[str, Any], *, scope: str, index: int) -> str:
    """Return one row's derived address, refusing any digest that contradicts it.

    THE chain chokepoint's single act: derive the address from the body, then
    re-certify every self-digest field the row declares against it
    (:data:`FUSED_ROW_DIGEST_FIELDS`).  Both halves live here so the exclusion set
    exists in exactly one place and no writer can drift from it -- a stamp that
    disagrees with the derivation refuses at its own append, not silently later.

    Parameters
    ----------
    row:
        Decoded ledger row.
    scope:
        Canonical chain scope, for the typed refusal.
    index:
        Zero-based row position, for the typed refusal.

    Returns
    -------
    str
        The row's body-derived content address.

    Raises
    ------
    LedgerChainError
        When a declared digest field is not the row's own body derivation.
    """

    address = row_content_address(row)
    for field in FUSED_ROW_DIGEST_FIELDS:
        if field not in row:
            continue
        declared = row[field]
        if declared is None:
            # PSLRE6 D6-5: a null used to read as "absent" and was skipped, so
            # ``"close_sha256": null, "record_revision": null, "payload_sha256": null``
            # was 65 bytes of unhashed, unchecked, address-invariant padding on any
            # chained row. A row DECLARES its own digest or OMITS the key; a present
            # key holding no address is neither.
            raise LedgerChainError(
                f"{scope} row {index} declares a null {field}: a chained row states its "
                "own content address or omits the key, and a present key holding no "
                "address is an unhashed region the link check never sees"
            )
        if declared != address:
            raise LedgerChainError(
                f"{scope} row {index} declares a {field} that is not its own body "
                "content address: a digest consumers key on is evidence, and this "
                "row's does not re-derive from the row body"
            )
    return address


def _extend(scope: str, previous: str, index: int, content: str) -> str:
    """Extend one chain by a position-bound logical row digest."""

    return stable_hash(
        {
            "chain_version": LEDGER_CHAIN_VERSION,
            "scope": scope,
            "index": index,
            "previous": previous,
            "content": content,
        }
    )


def _verified_link(
    row: Mapping[str, Any],
    *,
    scope: str,
    index: int,
    previous: str,
    linked: bool,
    require_links: bool = False,
) -> tuple[str, bool]:
    """Verify one row's position-bound link and return the extended head.

    THE per-row chain rule, shared by the whole-sequence verifier and the
    incremental derivation so the two can never disagree about what a valid
    link is or how a violation is named.

    Parameters
    ----------
    row:
        Decoded ledger row.
    scope:
        Canonical chain scope.
    index:
        Zero-based row position.
    previous:
        Verified head after the preceding row.
    linked:
        Whether a persisted link has already been seen (the link-free
        historical prefix closes for good at the first one).
    require_links:
        Whether even the first row must carry a persisted link.

    Returns
    -------
    tuple[str, bool]
        Extended head and the updated ``linked`` state.
    """

    LEDGER_CHAIN_STATS["rows_linked"] += 1
    expected = _extend(
        scope, previous, index, _certified_content_address(row, scope=scope, index=index)
    )
    chain = row.get("chain")
    if chain is None:
        if linked or require_links:
            raise LedgerChainError(
                f"{scope} fails its ledger chain at row {index}: a post-genesis row "
                "has no persisted chain link"
            )
    else:
        linked = True
        if not isinstance(chain, Mapping):
            raise LedgerChainError(f"{scope} row {index} has a malformed chain envelope")
        if set(map(str, chain)) != CHAIN_ENVELOPE_FIELDS:
            raise LedgerChainError(
                f"{scope} row {index} has a chain envelope outside its closed "
                "vocabulary: the envelope is machinery, and a key the digest does "
                "not cover and the link does not check is an unchecked region"
            )
        value_findings = chain_envelope_value_findings(chain)
        if value_findings:
            raise LedgerChainError(
                f"{scope} row {index} has a chain envelope whose values are not "
                "machinery-shaped (" + "; ".join(value_findings) + "): the envelope is "
                "unhashed, so a free-form value is an unauthenticated region"
            )
        if (
            chain.get("seq") != index
            or chain.get("prev") != previous
            or chain.get("sha256") != expected
        ):
            raise LedgerChainError(
                f"{scope} fails its ledger chain at row {index}: the stored link does "
                "not extend the preceding committed row"
            )
    return expected, linked


def verify_rows(
    rows: Sequence[Mapping[str, Any]], *, scope: str, require_links: bool = False
) -> ChainState:
    """Verify and derive the chain over decoded rows.

    Historical rows may form one link-free prefix.  Once a persisted link is
    encountered, every later row must carry the exact next link.
    """

    previous = _genesis(scope)
    heads: list[str] = []
    linked = False
    for index, row in enumerate(rows):
        previous, linked = _verified_link(
            row,
            scope=scope,
            index=index,
            previous=previous,
            linked=linked,
            require_links=require_links,
        )
        heads.append(previous)
    return ChainState(rows=len(rows), head_sha256=previous, heads=tuple(heads))


def extend_chain_state(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    state: Optional[ChainState] = None,
    linked: bool = False,
    require_links: bool = False,
) -> ChainState:
    """Verify an APPENDED SUFFIX against an already-verified chain state.

    :func:`verify_rows` is the whole-sequence spelling of the same per-row rule
    (:func:`_verified_link`); this is the incremental one, for a consumer that
    holds a verified prefix and must not re-verify it on every append.  Passing
    ``state=None`` derives from the scope's genesis, so this is also the way to
    ask for the empty state without constructing one by hand.

    Parameters
    ----------
    rows:
        Decoded rows that immediately follow ``state``, in ledger order.
    scope:
        Canonical chain scope.
    state:
        Verified state covering the preceding rows, or ``None`` for genesis.
    linked:
        Whether a persisted link has already been seen in the prefix.
    require_links:
        Whether even the first row must carry a persisted link.

    Returns
    -------
    ChainState
        The state extended over ``rows``.
    """

    previous = _genesis(scope) if state is None else state.head_sha256
    heads = [] if state is None else list(state.heads)
    index = 0 if state is None else state.rows
    seen = linked
    for row in rows:
        previous, seen = _verified_link(
            row,
            scope=scope,
            index=index,
            previous=previous,
            linked=seen,
            require_links=require_links,
        )
        heads.append(previous)
        index += 1
    return ChainState(rows=index, head_sha256=previous, heads=tuple(heads))


def link_row(
    row: Mapping[str, Any],
    *,
    scope: str,
    existing_rows: Sequence[Mapping[str, Any]],
    existing_state: Optional[ChainState] = None,
) -> JsonObject:
    """Attach the next persisted chain link without changing logical identity.

    Parameters
    ----------
    row:
        Logical row to link.
    scope:
        Canonical chain scope.
    existing_rows:
        Already decoded predecessor rows.
    existing_state:
        Optional chain state produced while validating ``existing_rows``. Passing
        it avoids a second whole-history verification inside an append critical
        section.

    Returns
    -------
    JsonObject
        Row with its exact position-bound chain envelope.
    """

    state = existing_state or verify_rows(existing_rows, scope=scope)
    if state.rows != len(existing_rows):
        raise LedgerChainError("provided chain state does not cover the existing rows")
    index = len(existing_rows)
    if "chain" in row:
        raise LedgerChainError(
            f"{scope} row {index} already carries a chain envelope: a link is attached "
            "exactly once, by this function, at the position the row is appended to"
        )
    # The SAME certification the reader runs, on the write path: a writer whose
    # stamp disagrees with the shared derivation refuses at its own append instead
    # of persisting a row every later reader must refuse.
    digest = _extend(
        scope, state.head_sha256, index, _certified_content_address(row, scope=scope, index=index)
    )
    linked = dict(row)
    linked["chain"] = {"seq": index, "prev": state.head_sha256, "sha256": digest}
    return linked


def decode_payload(payload: bytes, path: Path) -> tuple[JsonObject, ...]:
    """Decode every non-empty JSON object in one ledger's bytes fail-closed.

    THE strict decoder for every chain-bearing ledger: a complete malformed
    line is hard corruption and refuses. It is also the ``prefix_check`` that
    :func:`decode_ledger` hands to torn-tail recovery, so the bytes recovery is
    allowed to retain are exactly the bytes this decoder accepts.

    Parameters
    ----------
    payload:
        Complete ledger bytes.
    path:
        Ledger path, for diagnostics.
    """

    decoded: list[JsonObject] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = _loads_strict(line)
        except ValueError as exc:
            raise LedgerChainError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
        decoded.append(_strict_object(value, path=path, line_number=line_number))
    LEDGER_CHAIN_STATS["rows_decoded"] += len(decoded)
    return tuple(decoded)


def tail_recovery(path: Path, *, locked_handle: Optional[BinaryIO] = None) -> None:
    """Recover a torn final line on one chain-bearing ledger, or refuse typed.

    THE torn-tail policy of every Track A ledger and every census source: a
    non-newline-terminated final line (SIGKILL, power loss, or ENOSPC
    mid-append) is hashed into a durable recovery ledger and truncated by
    :func:`menagerie.crawler.recordio.recover_torn_tail`; a COMPLETE malformed
    line is untouched hard corruption that still refuses.

    Parameters
    ----------
    path:
        Ledger whose tail may be torn.
    locked_handle:
        Open handle on ``path`` whose exclusive ``flock`` the caller already
        holds. Recovery must never try to take that lock a second time from
        the same process, so writers pass their handle through.

    Raises
    ------
    LedgerChainError
        If a live writer owns the ledger's data inode while its tail is torn
        (retry after the writer closes).
    """

    from menagerie.crawler.recordio import SingleWriterError, read_with_tail_recovery

    try:
        read_with_tail_recovery(
            path,
            prefix_check=lambda retained: decode_payload(retained.read_bytes(), path),
            locked_handle=locked_handle,
        )
    except SingleWriterError as exc:
        # Keep this module's typed refusal vocabulary: an in-flight append by a
        # live writer is a retry condition, never a chain violation.
        raise LedgerChainError(str(exc)) from exc


def _decode_row(path: Path, line_number: int, line: bytes) -> JsonObject:
    """Decode one chain-bearing ledger row fail-closed.

    The single-row half of :func:`decode_payload`, so the streaming and
    whole-bytes readers cannot drift in what they accept or how they name a
    refusal.  Both halves settle through :func:`_strict_object`.
    """

    LEDGER_CHAIN_STATS["rows_decoded"] += 1
    try:
        value = _loads_strict(line)
    except ValueError as exc:
        raise LedgerChainError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return _strict_object(value, path=path, line_number=line_number)


def decode_ledger(
    path: Path, *, locked_handle: Optional[BinaryIO] = None
) -> tuple[JsonObject, ...]:
    """Decode one chain-bearing ledger, recovering only a torn final line first.

    Every Track A ledger -- the leg-event ledger, the close ledger, the rung
    registry -- resolves through here, so this is the ONE place their torn-tail
    contract lives (:func:`tail_recovery`).

    Before r2fix this read was a bare ``read_bytes().splitlines()``: one torn
    byte on any of those ledgers refused every close and every staging decision
    campaign-wide, permanently, with no recovery path -- the step-200 blocker.
    Round-2 T4 then made it INCREMENTAL: the decode is held per path and
    extended over the appended suffix once the held prefix re-hashes, so a
    staging gate that consults the close ledger a dozen times decodes each row
    once instead of once per consultation.

    Parameters
    ----------
    path:
        Ledger to decode. A missing file is an empty ledger.
    locked_handle:
        Open handle on ``path`` whose exclusive ``flock`` the caller already
        holds (every chain-linking append path).

    Raises
    ------
    LedgerChainError
        If a complete line is malformed, or a live writer owns the ledger's
        data inode while its tail is torn (retry after the writer closes).
    """

    if locked_handle is None and not path.is_file():
        return ()

    def extend(
        value: tuple[JsonObject, ...], rows: Iterable[LedgerRow]
    ) -> tuple[JsonObject, ...]:
        return (*value, *(_decode_row(path, number, line) for number, _offset, line in rows))

    return derive(
        path,
        projection="chain.rows",
        seed=tuple,
        extend=extend,
        recover=lambda target: tail_recovery(target, locked_handle=locked_handle),
        locked_handle=locked_handle,
        # The decoded rows themselves: every meta ledger this reads (closes,
        # leg events, the registry) is O(#closes), never O(campaign-history).
        weigh=lambda _value, byte_count: (
            PROJECTION_BASE_WEIGHT + DECODED_ROW_WEIGHT_FACTOR * byte_count
        ),
    )


@dataclass(frozen=True)
class _ChainProjection:
    """One held chain derivation: verified heads WITHOUT the decoded rows.

    A census source's chain is O(campaign-history) ROWS but only ever consumed
    as heads, so retaining the rows would put the whole models ledger in memory
    to answer "what was the head at row N?". The projection keeps the heads and
    the once-linked flag, and drops every decoded row.
    """

    state: ChainState
    linked: bool


def _extend_chain(
    path: Path, scope: str, value: _ChainProjection, rows: Iterable[LedgerRow]
) -> _ChainProjection:
    """Extend one held chain by the ledger's newly appended rows, STREAMING.

    Refusal order is the historical one: a ledger holding both a malformed
    complete line and a broken link must keep naming the MALFORMED LINE, which
    is what ``decode_ledger`` then ``verify_rows`` did by decoding every row
    before linking any. Round-2 reproduced that ordering by materializing the
    whole appended suffix as decoded rows first -- the campaign's entire models
    history resident as dicts (~2.65x its bytes) to derive a chain consumed only
    as heads (round-3 B1).

    So the ordering is preserved WITHOUT the materialization: one row is resident
    at a time, and a link violation is DEFERRED rather than raised, while the
    remaining rows keep decoding. A malformed line found afterwards wins, exactly
    as it did when decoding ran to completion first; if none appears, the held
    link violation is raised. Identical refusal, identical wording, O(1) rows
    resident.
    """

    previous = value.state.head_sha256
    heads = list(value.state.heads)
    linked = value.linked
    index = value.state.rows
    deferred: Optional[LedgerChainError] = None
    for number, _offset, line in rows:
        row = _decode_row(path, number, line)
        if deferred is not None:
            # The chain state is already broken; keep DECODING so a malformed
            # line further down the ledger still gets named first.
            continue
        try:
            previous, linked = _verified_link(
                row, scope=scope, index=index, previous=previous, linked=linked
            )
        except LedgerChainError as exc:
            deferred = exc
            continue
        heads.append(previous)
        index += 1
    if deferred is not None:
        raise deferred
    if index == value.state.rows:
        return value
    return _ChainProjection(
        state=ChainState(rows=index, head_sha256=previous, heads=tuple(heads)),
        linked=linked,
    )


def verified_chain(path: Path, *, scope: str) -> ChainState:
    """Return one ledger's verified chain state, derived incrementally.

    Parameters
    ----------
    path:
        Ledger path. A missing file is the scope's genesis state.
    scope:
        Canonical chain scope name.
    """

    def extend(value: _ChainProjection, rows: Iterable[LedgerRow]) -> _ChainProjection:
        return _extend_chain(path, scope, value, rows)

    projection: _ChainProjection = derive(
        path,
        projection=f"chain.state:{scope}",
        seed=lambda: _ChainProjection(
            state=ChainState(rows=0, head_sha256=_genesis(scope), heads=()), linked=False
        ),
        extend=extend,
        recover=tail_recovery,
        # One head per row and nothing else: the decoded rows are dropped as
        # they stream past. ``chain_head_at`` answers any committed row extent
        # from these, so the per-row head is what the projection exists to keep.
        weigh=lambda value, _byte_count: (
            PROJECTION_BASE_WEIGHT + HEAD_DIGEST_WEIGHT * value.state.rows
        ),
    )
    return projection.state


def chain_state(path: Path, *, scope: str) -> ChainState:
    """Return the verified chain state of one ledger path."""

    return verified_chain(path, scope=scope)


def chain_head_at(path: Path, *, scope: str, rows: int) -> str:
    """Return the verified chain head at an exact committed row extent."""

    state = verified_chain(path, scope=scope)
    if rows < 0 or rows > state.rows:
        raise LedgerChainError(
            f"a committed chain head for {scope} covers {rows} rows, but only "
            f"{state.rows} remain"
        )
    if rows == 0:
        return _genesis(scope)
    return state.heads[rows - 1]


def close_chain_head(records_root: Path, close_sha256: str) -> str:
    """Return the close-ledger chain head committed by one resident close."""

    from menagerie.crawler.close_object import close_object_path

    path = close_object_path(records_root)
    index = _close_head_index(path)
    head = index.get(str(close_sha256))
    if head is None:
        raise LedgerChainError(f"close {close_sha256} is not resident on the close ledger")
    return head


def _close_head_index(path: Path) -> Mapping[str, str]:
    """Return every resident close's content address to its verified head.

    Built once per observed close-ledger state instead of once per lookup: the
    registry anchor check asks this question per registration, which was
    O(#rungs) x O(#closes) whole-ledger verifications per staging gate.
    """

    rows = decode_ledger(path)
    state = verified_chain(path, scope="meta.closes")
    index: dict[str, str] = {}
    for position, row in enumerate(rows):
        digest = row.get("close_sha256")
        if isinstance(digest, str):
            # FIRST occurrence wins, matching the per-lookup scan this replaced.
            index.setdefault(digest, state.heads[position])
    return index


def ledger_chain_refusals(records_root: Path) -> tuple[str, ...]:
    """Return every Track A chain or anchor violation below one records root.

    Round-2 T4 (scale): every read below is a HELD derivation. Before it, this
    one function was the campaign's dominant cost -- it resolved a committed
    chain head per close row per census source, each resolution a whole-ledger
    decode plus a whole-history chain verification, so a single staging gate
    walked O(#closes) x O(15 sources) x O(campaign-history) rows. The
    per-source chain is now derived once per observed ledger state and indexed
    by row extent, and the registry anchor check reads one close-head index
    instead of re-verifying the close ledger per registration. The refusals,
    their order, and their wording are unchanged.

    Round-3 B1 bounded the same walk in BYTES: this verdict is ONE decision, so
    it runs inside one :func:`menagerie.crawler.close_reads.read_epoch` and each
    ledger extent it consults is proven from its bytes once instead of once per
    close per source.
    """

    with read_epoch():
        return _ledger_chain_refusals(records_root)


def _ledger_chain_refusals(records_root: Path) -> tuple[str, ...]:
    """Derive every Track A refusal inside the caller's open read epoch."""

    from menagerie.crawler.close_object import close_object_path
    from menagerie.crawler.leg_generation import close_ledger_sources, leg_events_path
    from menagerie.crawler.paid_session_lifecycle import load_paid_session_rows
    from menagerie.crawler.rung_staging import (
        rung_class_refinement_path,
        rung_registry_path,
    )

    findings: list[str] = []
    meta = (
        ("meta.leg-events", leg_events_path(records_root)),
        ("meta.closes", close_object_path(records_root)),
        ("meta.registry", rung_registry_path(records_root)),
        # R2-4/SOL-01: the class-refinement ledger steers which certificate
        # kind a leg demands, so it joins the same tamper-evidence spine.
        ("meta.class-refinements", rung_class_refinement_path(records_root)),
    )
    for scope, path in meta:
        try:
            verified_chain(path, scope=scope)
        except LedgerChainError as exc:
            findings.append(str(exc))

    # Paid-session money is not a close-content source, so it cannot perturb
    # unrelated close/licensing content addresses. Its separate anchored
    # identity and extent nevertheless join the same fail-closed integrity
    # spine: a valid-prefix rollback must refuse before any close-consuming or
    # paid-admission decision proceeds.
    paid_session_path = records_root / "operational" / "spend.jsonl"
    try:
        load_paid_session_rows(paid_session_path)
    except ValueError as exc:
        findings.append(str(exc))

    close_rows = decode_ledger(close_object_path(records_root))
    resident = {
        str(row.get("close_sha256")): row
        for row in close_rows
        if isinstance(row.get("close_sha256"), str)
    }
    previous_by_campaign: dict[str, str] = {}
    sources = close_ledger_sources(records_root)
    for row in close_rows:
        if row.get("close_kind") not in ("leg", "backfill"):
            continue
        campaign = str(row.get("campaign_id"))
        claimed = row.get("predecessor_close_sha256")
        expected = previous_by_campaign.get(campaign)
        if claimed != expected:
            findings.append(
                f"close {row.get('close_sha256')} names predecessor {claimed!r}, but the "
                f"verified same-campaign close spine yields {expected!r}"
            )
        previous_by_campaign[campaign] = str(row.get("close_sha256"))
        for scope, identity_value in dict(row.get("sources") or {}).items():
            identity = dict(identity_value or {})
            if scope not in sources:
                continue
            if "chain_head" not in identity:
                findings.append(
                    f"{scope}'s committed close identity names no chain_head: "
                    "the Track A tamper-evidence authority may not be omitted"
                )
                continue
            try:
                observed = chain_head_at(
                    sources[scope], scope=scope, rows=int(identity.get("rows") or 0)
                )
            except LedgerChainError as exc:
                findings.append(str(exc))
                continue
            if observed != identity.get("chain_head"):
                findings.append(
                    f"{scope}'s committed close head {identity.get('chain_head')} is not "
                    f"resolvable on the current ledger ({observed})"
                )

    # One close-head index for every registration anchor, not one whole-ledger
    # chain verification per registration. A broken close chain still reports
    # once per licensed registration, exactly as the per-registration
    # ``close_chain_head`` call did.
    close_heads: Mapping[str, str] = {}
    close_head_failure: Optional[str] = None
    try:
        close_heads = _close_head_index(close_object_path(records_root))
    except LedgerChainError as exc:
        close_head_failure = str(exc)
    for registration in decode_ledger(rung_registry_path(records_root)):
        licensed = registration.get("licensed_by_close_sha256")
        if licensed is not None and str(licensed) not in resident:
            findings.append(
                f"the registry says {registration.get('snapshot_id')} was staged on close "
                f"{licensed}, which is no longer resident on the close ledger"
            )
            continue
        if licensed is not None:
            if close_head_failure is not None:
                findings.append(close_head_failure)
                continue
            expected_head = close_heads.get(str(licensed))
            if expected_head is None:
                findings.append(
                    f"close {licensed} is not resident on the close ledger"
                )
                continue
            if registration.get("closes_chain_head") != expected_head:
                findings.append(
                    f"the registry anchor for {registration.get('snapshot_id')} is "
                    f"{registration.get('closes_chain_head')!r}, not the licensed "
                    f"close's verified chain head {expected_head!r}"
                )
    return tuple(findings)


def verify_records_integrity(records_root: Path) -> None:
    """Verify the Track A ledger chain and every committed close anchor."""

    findings = ledger_chain_refusals(records_root)
    if findings:
        raise LedgerChainError("; ".join(findings))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only Track A verifier from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records_root", type=Path)
    args = parser.parse_args(argv)
    try:
        verify_records_integrity(args.records_root)
    except LedgerChainError as exc:
        parser.exit(1, f"REFUSED: {exc}\n")
    print("TRACK_A_CHAIN_OK")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
