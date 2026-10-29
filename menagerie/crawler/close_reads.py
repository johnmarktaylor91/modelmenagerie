"""The generation-bound incremental read substrate of the close/licensing path.

BL-7 closed the O(campaign-history) read class on the four core fact ledgers.
BL-5 / Track A then re-opened it one layer up, in the same wave: every close
derivation and every staging-gate evaluation re-materialized whole ledger
histories -- JSON-decoding, digest-verifying, and chain-verifying every row --
and did so once per PRIOR CLOSE, because ``ledger_chain_refusals`` resolves one
committed chain head per close row per census source, and ``_accounting_close_for``
walks every prior close per member. One licensing decision therefore cost
O(campaign-history) x O(#closes), measured at ``(#closes + 2) x history`` rows
decoded per staging gate: survivable at step-200, a hard wall at the 500 rung.

The doctrine here is BL-7's, verbatim: **hashing bytes is cheap, decoding them
is not.** A whole-file sha256 of a 20k-model ledger is tens of milliseconds; a
whole-file JSON decode with per-row ``payload_sha256`` re-derivation is tens of
seconds. So nothing here is ever served on trust:

* held state is reused ONLY over bytes that reproduce their recorded digest.
  Every call re-hashes the whole held prefix in bounded chunks -- there is no
  mtime fast path, no inode fast path, and no append-only assumption anywhere
  in the decision. This is the licensing gate's read path; its tripwire value
  comes from actually looking at the bytes;
* a ledger that GREW past a prefix that still hashes extends the derivation
  over the appended suffix alone;
* anything else -- a shrink, a replacement, a rewritten prefix -- discards the
  held state and re-derives from byte zero through byte-for-byte the same code
  path a cold process runs.

Nothing is memoized across a failure: a derivation that raises stores nothing,
so a corrupt ledger refuses identically on every call. Rows stream through the
``extend`` callback, so a cold derivation's peak resident bytes are one chunk
plus whatever the projection itself retains -- never the whole ledger. The
projections are SHARED, read-only values; a caller that needs to mutate one
must copy it.

Round-3 closed the two halves round-2 left open, both measured on the same
ladder rig:

* **BYTES, not decodes.** The decode was incremental, but the PROOF was not:
  every held derivation re-hashed its whole prefix on every call, so one
  licensing decision that resolves a committed chain head per close per census
  source paid ``O(#closes) x O(campaign-history)`` BYTES even while decoding each
  row exactly once. A prefix proof can never be cheaper than the bytes it covers
  -- that is what makes it a proof -- so what is bounded instead is the number of
  TIMES one decision performs it. :func:`read_epoch` is a decision's explicitly
  scoped read window: inside it each ``(path, physical extent, prefix length)`` is
  proven from the real bytes ONCE, and every later reader of that same unchanged
  extent reuses the verified hash state. This is the sibling of
  :func:`menagerie.crawler.partition_proof.proof_read_epoch` and grants exactly
  what that grants: the repetition goes, the verification stays. Outside an epoch
  every call still proves from byte zero; inside one, ANY change to the observed
  extent (device, inode, size, or mtime) misses the memo and re-proves from byte
  zero. A same-size, same-mtime in-place rewrite landing inside a single
  decision's own window is therefore the one mutation an epoch does not
  re-observe: disclosed here, pinned by the fail-closed suite, and the deliberate
  price of a decision whose evidence set is self-consistent.
* **A byte budget, not an entry count.** ``_MAX_HELD`` bounds how MANY
  derivations are held, which is not a memory bound at all -- one models-ledger
  projection at the 20k rung retains more than a thousand small identities do.
  Every projection now declares what it retains through ``weigh``, and the held
  set is evicted least-recently-derived first to stay inside
  :data:`_MAX_HELD_BYTES`. A derivation heavier than the whole budget is returned
  to its caller but never held.

This is deliberately a sibling of
:class:`menagerie.crawler.recordio.IncrementalJsonlScan` rather than a user of
it. That class is an advisory row cache whose invalidation rests on the stat
signature alone; the licensing gate's reads are the tripwire that catches a
rewritten history, so their fast path is gated on a real prefix digest instead.

The observed physical extent is still recorded, but only as a diagnostic and as
the cheap bound on how many bytes a prefix proof has to cover -- never as
permission to skip the proof.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Optional

#: Streaming chunk for prefix verification and row decoding. Peak resident
#: bytes for a held prefix proof are one chunk, never the whole ledger.
_CHUNK_BYTES = 1 << 20

#: Bound on concurrently held derivations. The close path touches ~20 ledger
#: paths across a handful of projections; the cap keeps a long-lived driver
#: from accumulating derivations for ledgers it will never read again.
_MAX_HELD = 256

#: Byte budget for the whole held set -- the REAL memory bound (round-3 B2).
#: A 256-ENTRY cap over entries of unbounded size bounded nothing that matters:
#: ONE models projection at the 20k rung retains ~55 MiB (measured 6.2x its
#: ledger's bytes) and is charged ~72 MiB, so 256 entries of it is a resident set
#: that grows with the campaign until the host dies. This budget holds one live
#: campaign's whole close-path working set (~100 MiB at the 20k rung) with
#: headroom, and evicts rather than grows when a long run walks further.
_MAX_HELD_BYTES = 192 << 20

#: What an undeclared projection is charged. Deliberately pessimistic -- above
#: the heaviest inflation any declared projection measures -- so forgetting
#: ``weigh`` over-evicts (slow) instead of under-accounting (an unbounded
#: resident set, the very defect the budget exists to close).
_UNDECLARED_WEIGHT_FACTOR = 8

#: Cap on one epoch's memoized prefix proofs. A licensing decision touches ~20
#: paths at a handful of lengths each; past the cap the epoch simply stops
#: memoizing and every further reader proves from byte zero again.
_MAX_EPOCH_PROOFS = 4096

# The declared per-unit retention of every projection this substrate and its
# consumers hold. Each is an UPPER estimate of real CPython footprint, measured
# on a 64-bit interpreter against a real ladder and pinned by
# ``test_close_read_scale.test_declared_weights_bound_what_projections_retain``
# -- a weight that UNDER-states what a projection retains would turn the byte
# budget back into the entry count it replaced.

#: Per-projection fixed overhead: the dataclass, its containers, and the small
#: strings every projection carries regardless of the ledger's size.
PROJECTION_BASE_WEIGHT = 4096

#: A four-field identity with two digest strings, whatever the ledger's size.
CONTENT_IDENTITY_WEIGHT = 1024

#: One ``(start_offset, length)`` row span: a 2-tuple, two ints, one pointer.
LINE_SPAN_WEIGHT = 136

#: One ``sha256:``-spelled chain head string plus its container pointer.
HEAD_DIGEST_WEIGHT = 136

#: What a projection retaining DECODED rows holds against the bytes they were
#: decoded from. Measured 2.8x-3.1x for the meta ledgers and 6.2x for the models
#: head fold (nested status mappings, one dict and its value strings per row);
#: charged above the worst of those so the budget errs toward evicting.
DECODED_ROW_WEIGHT_FACTOR = 8

#: One decoded ledger row as the substrate hands it to a projection:
#: ``(physical_line_number, start_offset, line_bytes)``.
LedgerRow = tuple[int, int, bytes]


@dataclass(frozen=True)
class LedgerExtent:
    """One ledger's observed physical extent.

    Parameters
    ----------
    device, inode:
        Identity of the file the bytes live in.
    size:
        Byte length.
    mtime_ns:
        Nanosecond modification time.
    """

    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class _Held:
    """One held derivation and the exact bytes it was derived from.

    Parameters
    ----------
    extent:
        Physical extent the derivation covers.
    byte_count, line_count:
        Prefix length and non-empty row count covered.
    prefix_sha256:
        Digest over that prefix; the fast path is served only against it.
    value:
        The shared projection value.
    weight:
        Declared retained bytes, the unit :data:`_MAX_HELD_BYTES` is spent in.
    """

    extent: LedgerExtent
    byte_count: int
    line_count: int
    prefix_sha256: str
    value: Any
    weight: int


_HELD: dict[tuple[str, str], _Held] = {}

#: Declared weight of everything in ``_HELD``, maintained on every store and
#: eviction so the budget never has to walk the held set to answer "how much?".
_HELD_WEIGHT = 0


@dataclass(frozen=True)
class ReadStatistics:
    """Physical cost of the reads this substrate has served since the last reset.

    The scale tripwire's instrument, and PRODUCTION-side on purpose: a census
    that has to monkeypatch its way in measures whatever the patch happens to
    cover, while these counters sit on the one code path every ledger byte of
    the close/licensing spine flows through.

    Parameters
    ----------
    bytes_read:
        Ledger bytes physically read, whether to hash them or to decode them.
    prefix_proof_bytes, prefix_proofs:
        Bytes covered by, and the number of, executed prefix proofs.
    epoch_proof_reuses:
        Prefix proofs served from an open :func:`read_epoch` instead of
        re-hashing an unchanged extent.
    cold_derivations, extended_derivations:
        Derivations that started from byte zero, and those that extended a held
        prefix over an appended suffix.
    held_evictions:
        Held derivations dropped to stay inside the count or byte budget.
    """

    bytes_read: int
    prefix_proof_bytes: int
    prefix_proofs: int
    epoch_proof_reuses: int
    cold_derivations: int
    extended_derivations: int
    held_evictions: int


_STATS: dict[str, int] = {
    "bytes_read": 0,
    "prefix_proof_bytes": 0,
    "prefix_proofs": 0,
    "epoch_proof_reuses": 0,
    "cold_derivations": 0,
    "extended_derivations": 0,
    "held_evictions": 0,
}


def read_statistics() -> ReadStatistics:
    """Return the physical read cost accumulated since the last reset."""

    return ReadStatistics(**_STATS)


def reset_read_statistics() -> None:
    """Zero the physical read counters, holding every derivation."""

    for key in _STATS:
        _STATS[key] = 0


@dataclass
class _ReadEpoch:
    """One decision's read window: prefix proofs, keyed by exact extent.

    Parameters
    ----------
    proofs:
        ``(path, device, inode, size, mtime_ns, length) -> hash state`` over
        that prefix. The extent is part of the KEY, so nothing about a file
        that moved, grew, shrank, or was rewritten can be answered from here.
    """

    proofs: dict[tuple[str, int, int, int, int, int], "hashlib._Hash"] = field(
        default_factory=dict
    )


_EPOCH: Optional[_ReadEpoch] = None


@contextmanager
def read_epoch() -> Iterator[Any]:
    """Open one decision's read epoch, proving each unchanged extent once.

    Re-entrant: a nested epoch joins the outer one, so an entry point can open
    one without knowing whether its caller already did. Nothing survives the
    outermost exit, and nothing is trusted on entry.
    """

    global _EPOCH  # noqa: PLW0603 -- module-scoped epoch, one owner
    if _EPOCH is not None:
        yield _EPOCH
        return
    _EPOCH = _ReadEpoch()
    try:
        yield _EPOCH
    finally:
        _EPOCH = None


def reset_incremental_reads() -> None:
    """Drop every held derivation and abandon any open read epoch.

    For tests that rewrite a ledger's bytes behind the substrate's back, and
    for any caller that has forked. Correctness never depends on it: a changed
    prefix is caught by its digest, not by this.
    """

    global _HELD_WEIGHT, _EPOCH  # noqa: PLW0603 -- module-scoped, one owner
    _HELD.clear()
    _HELD_WEIGHT = 0
    _EPOCH = None


def held_derivation_count() -> int:
    """Return how many derivations are currently held (test observability)."""

    return len(_HELD)


def held_derivation_bytes() -> int:
    """Return the declared retained bytes of every currently held derivation."""

    return _HELD_WEIGHT


def ledger_extent(
    path: Path, *, locked_handle: Optional[BinaryIO] = None
) -> Optional[LedgerExtent]:
    """Return one ledger's physical extent, or ``None`` when it is absent.

    Parameters
    ----------
    path:
        Ledger path.
    locked_handle:
        Open handle whose ``flock`` the caller holds; stat'ed through
        ``fstat`` so a writer never has to re-resolve its own path.
    """

    try:
        stat = os.fstat(locked_handle.fileno()) if locked_handle is not None else path.stat()
    except OSError:
        return None
    return LedgerExtent(
        device=stat.st_dev, inode=stat.st_ino, size=stat.st_size, mtime_ns=stat.st_mtime_ns
    )


def _read_range(
    path: Path, start: int, stop: int, *, locked_handle: Optional[BinaryIO]
) -> Iterator[bytes]:
    """Yield ``[start, stop)`` in bounded chunks, never materializing the whole."""

    if stop <= start:
        return
    if locked_handle is not None:
        position = locked_handle.tell()
        try:
            locked_handle.seek(start)
            remaining = stop - start
            while remaining > 0:
                chunk = locked_handle.read(min(_CHUNK_BYTES, remaining))
                if not chunk:
                    return
                remaining -= len(chunk)
                _STATS["bytes_read"] += len(chunk)
                yield chunk
        finally:
            locked_handle.seek(position)
        return
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = stop - start
        while remaining > 0:
            chunk = handle.read(min(_CHUNK_BYTES, remaining))
            if not chunk:
                return
            remaining -= len(chunk)
            _STATS["bytes_read"] += len(chunk)
            yield chunk


def _sha256_spelling(digest: "hashlib._Hash") -> str:
    """Return the tree's canonical ``sha256:`` spelling of one hash state."""

    return f"sha256:{digest.hexdigest()}"


def prefix_digest(
    path: Path, length: int, *, locked_handle: Optional[BinaryIO] = None
) -> Optional[str]:
    """Return ``sha256:`` over the ledger's first ``length`` bytes.

    THE bounded-memory spelling of "does this file still start with the bytes
    somebody pinned?". Returns ``None`` when the file is shorter than
    ``length`` -- a shrink can never supply a prefix, and the caller must
    refuse rather than compare a truncated digest.

    Parameters
    ----------
    path:
        Ledger path. A missing file supplies only the empty prefix.
    length:
        Prefix byte length to digest.
    locked_handle:
        Open handle whose ``flock`` the caller already holds.
    """

    if length < 0:
        return None
    if length == 0:
        return _sha256_spelling(hashlib.sha256())
    extent = ledger_extent(path, locked_handle=locked_handle)
    if extent is None or extent.size < length:
        return None
    digest = _proven_prefix_hash(path, extent, length, locked_handle=locked_handle)
    return None if digest is None else _sha256_spelling(digest)


def _prefix_hash(
    path: Path, length: int, *, locked_handle: Optional[BinaryIO]
) -> Optional["hashlib._Hash"]:
    """Return the live hash state over ``[0, length)``, or ``None`` if short."""

    if length < 0:
        return None
    digest = hashlib.sha256()
    if length == 0:
        return digest
    extent = ledger_extent(path, locked_handle=locked_handle)
    if extent is None or extent.size < length:
        return None
    seen = 0
    for chunk in _read_range(path, 0, length, locked_handle=locked_handle):
        digest.update(chunk)
        seen += len(chunk)
    if seen != length:
        return None
    _STATS["prefix_proofs"] += 1
    _STATS["prefix_proof_bytes"] += length
    return digest


def _epoch_key(
    path: Path, extent: LedgerExtent, length: int
) -> tuple[str, int, int, int, int, int]:
    """Return the epoch memo key binding one prefix proof to exact bytes."""

    return (
        str(path),
        extent.device,
        extent.inode,
        extent.size,
        extent.mtime_ns,
        length,
    )


def _proven_prefix_hash(
    path: Path,
    extent: LedgerExtent,
    length: int,
    *,
    locked_handle: Optional[BinaryIO],
) -> Optional["hashlib._Hash"]:
    """Return the proven hash state over ``[0, length)`` of one observed extent.

    Identical to :func:`_prefix_hash` outside a :func:`read_epoch`. Inside one,
    the proof over an extent that has not moved is performed once and its hash
    state is copied for every later reader; a different device, inode, size, or
    mtime is a different key and re-proves from byte zero.
    """

    epoch = _EPOCH
    if epoch is None:
        return _prefix_hash(path, length, locked_handle=locked_handle)
    key = _epoch_key(path, extent, length)
    held = epoch.proofs.get(key)
    if held is not None:
        _STATS["epoch_proof_reuses"] += 1
        return held.copy()
    digest = _prefix_hash(path, length, locked_handle=locked_handle)
    if digest is not None:
        _publish_epoch_proof(key, digest)
    return digest


def _publish_epoch_proof(
    key: tuple[str, int, int, int, int, int], digest: "hashlib._Hash"
) -> None:
    """Memoize one completed prefix proof in the open epoch, if there is room."""

    epoch = _EPOCH
    if epoch is None or len(epoch.proofs) >= _MAX_EPOCH_PROOFS:
        return
    epoch.proofs.setdefault(key, digest.copy())


def _tail_is_terminated(
    path: Path, extent: LedgerExtent, *, locked_handle: Optional[BinaryIO]
) -> bool:
    """Return whether the ledger's final byte is a newline (or it is empty).

    One seek, one byte: the cheap pre-check that keeps the ordinary read off
    the whole-file torn-tail recovery path entirely.
    """

    if extent.size == 0:
        return True
    for chunk in _read_range(path, extent.size - 1, extent.size, locked_handle=locked_handle):
        return chunk.endswith(b"\n")
    return False


class _RowReader:
    """Streaming non-empty-row reader that also carries the running digest.

    Blank (whitespace-only) terminated lines are skipped but counted, matching
    every reader in the tree. A non-newline-terminated final line raises
    ``torn_error(start_offset)`` when a refusing policy is supplied; callers
    that recover the tail first pass ``None`` and never meet one.
    """

    def __init__(
        self,
        path: Path,
        *,
        start_byte: int,
        stop_byte: int,
        start_line: int,
        digest: "hashlib._Hash",
        locked_handle: Optional[BinaryIO],
        torn_error: Optional[Callable[[int], Exception]],
    ) -> None:
        self.path = path
        self.start_byte = start_byte
        self.stop_byte = stop_byte
        self.line_count = start_line
        self.digest = digest
        self.locked_handle = locked_handle
        self.torn_error = torn_error
        #: Bytes this reader has actually pulled in. The digest covers exactly
        #: ``[0, start_byte + covered)``, so a short read (a ledger truncated
        #: under a live reader) can never be published as a whole-extent proof.
        self.covered = 0
        self._iterator = self._iterate()

    def _iterate(self) -> Iterator[LedgerRow]:
        buffer = b""
        offset = self.start_byte
        for chunk in _read_range(
            self.path, self.start_byte, self.stop_byte, locked_handle=self.locked_handle
        ):
            self.digest.update(chunk)
            self.covered += len(chunk)
            buffer += chunk
            while True:
                index = buffer.find(b"\n")
                if index < 0:
                    break
                line = buffer[: index + 1]
                buffer = buffer[index + 1 :]
                self.line_count += 1
                if line.strip():
                    yield (self.line_count, offset, line)
                offset += len(line)
        if buffer:
            if self.torn_error is not None:
                raise self.torn_error(offset)
            self.line_count += 1
            if buffer.strip():
                yield (self.line_count, offset, buffer)

    def __iter__(self) -> Iterator[LedgerRow]:
        return self._iterator

    def drain(self) -> None:
        """Consume any rows the projection left unread, completing the digest."""

        for _row in self._iterator:
            pass


def derive(
    path: Path,
    *,
    projection: str,
    seed: Callable[[], Any],
    extend: Callable[[Any, Iterable[LedgerRow]], Any],
    finalize: Optional[Callable[[Any, str, int], Any]] = None,
    recover: Optional[Callable[[Path], None]] = None,
    torn_error: Optional[Callable[[int], Exception]] = None,
    locked_handle: Optional[BinaryIO] = None,
    weigh: Optional[Callable[[Any, int], int]] = None,
) -> Any:
    """Derive one named projection over an append-only ledger incrementally.

    Parameters
    ----------
    path:
        Ledger path. A missing file derives from ``seed()`` over zero rows.
    projection:
        Stable name of the derivation, so two consumers of the same ledger
        never share each other's held value.
    seed:
        Zero-row projection factory.
    extend:
        ``(value, rows) -> value``, where ``rows`` streams
        ``(physical_line_number, start_offset, line_bytes)`` in ledger order.
        Called with the WHOLE ledger on a cold or invalidated derivation and
        with only the appended suffix on a prefix-proven one, so it must be
        purely additive.
    finalize:
        Optional ``(value, whole_file_sha256, byte_count) -> value`` applied
        once the digest over EVERY byte is complete, for projections (the
        source census identity) whose value includes it.
    recover:
        Torn-tail recovery for this ledger family, invoked only when the final
        byte is not a newline. Mutually exclusive with ``torn_error``.
    torn_error:
        Factory for the consumer's own torn-tail refusal, for readers whose
        contract is to REFUSE a torn tail rather than recover it (the
        ``scan_jsonl``-shaped reads).
    locked_handle:
        Open handle on ``path`` whose exclusive ``flock`` the caller holds.
    weigh:
        ``(value, byte_count) -> retained bytes`` for this projection, the unit
        the held-set byte budget is spent in. It must be an UPPER estimate of
        what the value actually retains; an omitted one is charged
        :data:`_UNDECLARED_WEIGHT_FACTOR` times the covered prefix.

    Returns
    -------
    Any
        The SHARED projection value. Callers must not mutate it.
    """

    extent = ledger_extent(path, locked_handle=locked_handle)
    if extent is None:
        return seed()
    if recover is not None and not _tail_is_terminated(
        path, extent, locked_handle=locked_handle
    ):
        recover(path)
        extent = ledger_extent(path, locked_handle=locked_handle)
        if extent is None:
            return seed()
    key = (str(path), projection)
    held = _HELD.get(key)
    base: Optional[_Held] = None
    digest: Optional["hashlib._Hash"] = None
    if held is not None and extent.size >= held.byte_count:
        # The held prefix is served ONLY against a digest over its bytes: no
        # mtime fast path, no inode fast path, no append-only assumption -- the
        # fast path is a proof, not a guess. Inside a read epoch that proof is
        # performed once per unchanged extent instead of once per consultation
        # (round-3 B1); outside one it runs on every call, as it always did.
        candidate = _proven_prefix_hash(
            path, extent, held.byte_count, locked_handle=locked_handle
        )
        if candidate is not None and _sha256_spelling(candidate) == held.prefix_sha256:
            base = held
            digest = candidate
    if base is None or digest is None:
        base = None
        digest = hashlib.sha256()
    _STATS["cold_derivations" if base is None else "extended_derivations"] += 1
    reader = _RowReader(
        path,
        start_byte=0 if base is None else base.byte_count,
        stop_byte=extent.size,
        start_line=0 if base is None else base.line_count,
        digest=digest,
        locked_handle=locked_handle,
        torn_error=torn_error,
    )
    value = extend(seed() if base is None else base.value, reader)
    reader.drain()
    whole = _sha256_spelling(reader.digest)
    covered = (0 if base is None else base.byte_count) + reader.covered
    if covered == extent.size:
        # A completed proof over the WHOLE observed extent, published so the
        # next projection of the same ledger in this decision reuses it. A
        # short read (truncated under a live reader) publishes nothing.
        _publish_epoch_proof(_epoch_key(path, extent, extent.size), reader.digest)
    if finalize is not None:
        value = finalize(value, whole, extent.size)
    _store(
        key,
        extent=extent,
        byte_count=extent.size,
        line_count=reader.line_count,
        prefix_sha256=whole,
        value=value,
        weight=(
            _UNDECLARED_WEIGHT_FACTOR * extent.size
            if weigh is None
            else max(0, int(weigh(value, extent.size)))
        ),
    )
    return value


def _store(
    key: tuple[str, str],
    *,
    extent: LedgerExtent,
    byte_count: int,
    line_count: int,
    prefix_sha256: str,
    value: Any,
    weight: int,
) -> None:
    """Hold one successful derivation inside the count and byte budgets.

    Eviction is least-recently-derived first: every successful derivation
    re-inserts its key at the end, so insertion order IS recency order. A
    derivation whose declared weight exceeds the whole budget is never held --
    its caller still has the value, and the driver's resident set stays bounded.
    """

    global _HELD_WEIGHT  # noqa: PLW0603 -- module-scoped held set, one owner
    displaced = _HELD.pop(key, None)
    if displaced is not None:
        _HELD_WEIGHT -= displaced.weight
    if weight > _MAX_HELD_BYTES:
        return
    while _HELD and (len(_HELD) >= _MAX_HELD or _HELD_WEIGHT + weight > _MAX_HELD_BYTES):
        evicted = _HELD.pop(next(iter(_HELD)))
        _HELD_WEIGHT -= evicted.weight
        _STATS["held_evictions"] += 1
    _HELD[key] = _Held(
        extent=extent,
        byte_count=byte_count,
        line_count=line_count,
        prefix_sha256=prefix_sha256,
        value=value,
        weight=weight,
    )
    _HELD_WEIGHT += weight


@dataclass(frozen=True)
class LedgerContentIdentity:
    """One ledger's observed content identity, derived by streaming.

    THE shared shape behind every "what exactly is on disk right now" claim the
    close spine attests. The whole-file digest stays a REAL digest over every
    byte -- it is what a close's re-observation is compared against, and hashing
    bytes is the cheap half of the read. What no longer happens is materializing
    the ledger to split it into lines.

    Parameters
    ----------
    rows:
        Non-empty row count.
    byte_count:
        Byte length covered.
    ledger_sha256:
        Digest over EVERY byte.
    tail_sha256:
        Digest over the last non-empty line, newline excluded; ``None`` when
        there is no such line.
    """

    rows: int
    byte_count: int
    ledger_sha256: str
    tail_sha256: Optional[str]


def _extend_content_identity(
    value: LedgerContentIdentity, rows: Iterable[LedgerRow]
) -> LedgerContentIdentity:
    """Fold newly appended rows into a held content identity."""

    count = value.rows
    tail = value.tail_sha256
    for _number, _offset, line in rows:
        count += 1
        tail = _sha256_spelling(hashlib.sha256(line.rstrip(b"\n")))
    if count == value.rows:
        return value
    return LedgerContentIdentity(
        rows=count,
        byte_count=value.byte_count,
        ledger_sha256=value.ledger_sha256,
        tail_sha256=tail,
    )


def content_identity(
    path: Path, *, projection: str, recover: Optional[Callable[[Path], None]] = None
) -> LedgerContentIdentity:
    """Return one ledger's held content identity.

    Parameters
    ----------
    path:
        Ledger path. A missing file is the empty identity.
    projection:
        Held-derivation name, so two consumers with different torn-tail
        policies over the same ledger never share a value.
    recover:
        Torn-tail recovery, or ``None`` for readers whose contract is to hash
        whatever bytes are there.
    """

    return derive(
        path,
        projection=projection,
        seed=lambda: LedgerContentIdentity(
            rows=0,
            byte_count=0,
            ledger_sha256=_sha256_spelling(hashlib.sha256()),
            tail_sha256=None,
        ),
        extend=_extend_content_identity,
        finalize=lambda value, digest, byte_count: LedgerContentIdentity(
            rows=value.rows,
            byte_count=byte_count,
            ledger_sha256=digest,
            tail_sha256=value.tail_sha256,
        ),
        recover=recover,
        # Four fields and two digest strings, whatever the ledger's size.
        weigh=lambda _value, _byte_count: CONTENT_IDENTITY_WEIGHT,
    )


@dataclass(frozen=True)
class LedgerLineIndex:
    """Byte spans of every non-empty row in one ledger, in ledger order.

    The windowed-read authority: a leg window ``[floor, close)`` reads exactly
    its own rows by seeking to ``spans[floor]`` instead of materializing the
    campaign's whole history to slice the tail off it.

    Parameters
    ----------
    spans:
        ``(start_offset, length)`` per non-empty row.
    """

    spans: tuple[tuple[int, int], ...]

    @property
    def rows(self) -> int:
        """Return the non-empty row count."""

        return len(self.spans)


def _extend_line_index(
    value: LedgerLineIndex, rows: Iterable[LedgerRow]
) -> LedgerLineIndex:
    """Append newly observed row spans to a held line index."""

    appended = tuple((offset, len(line)) for _number, offset, line in rows)
    if not appended:
        return value
    return LedgerLineIndex(spans=(*value.spans, *appended))


def line_index(path: Path, *, recover: Callable[[Path], None]) -> LedgerLineIndex:
    """Return the held byte-span index of one ledger's non-empty rows.

    Parameters
    ----------
    path:
        Ledger path.
    recover:
        The ledger family's torn-tail recovery, invoked only for a torn tail.
    """

    return derive(
        path,
        projection="line-index",
        seed=lambda: LedgerLineIndex(spans=()),
        extend=_extend_line_index,
        recover=recover,
        weigh=lambda value, _byte_count: (
            PROJECTION_BASE_WEIGHT + LINE_SPAN_WEIGHT * len(value.spans)
        ),
    )


def read_rows(
    path: Path, index: LedgerLineIndex, start: int, stop: int
) -> tuple[bytes, ...]:
    """Read rows ``[start, stop)`` of one ledger by byte span.

    Parameters
    ----------
    path:
        Ledger path.
    index:
        Held line index over the same bytes.
    start, stop:
        Half-open row window, clamped to the index.

    Returns
    -------
    tuple[bytes, ...]
        The window's raw lines, newline included, in ledger order.
    """

    first = max(0, int(start))
    last = min(int(stop), index.rows)
    if last <= first:
        return ()
    begin = index.spans[first][0]
    end = index.spans[last - 1][0] + index.spans[last - 1][1]
    payload = b"".join(_read_range(path, begin, end, locked_handle=None))
    return tuple(
        payload[offset - begin : offset - begin + length]
        for offset, length in index.spans[first:last]
    )
