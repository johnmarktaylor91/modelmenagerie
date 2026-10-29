"""CONCERN D -- paid-session DURABILITY: the anchored append-only substrate.

This module knows how bytes reach disk and nothing about what they mean.  It
imports no budget symbol and no custody symbol; its whole contract is "an
append-only chain-linked ledger whose identity and extent are pinned by an
external anchor, recoverable across a crash and refusing every rollback,
truncation, replacement, or out-of-band append".  The money meaning of a row is
supplied by the caller as a pure fold (concern A), so a durability change can
never alter a charge and an accounting change can never weaken durability.

The anchor protocol
-------------------
The failure that kept re-opening across rounds is that a crash between the row's
``F_FULLFSYNC`` and the anchor update is BYTE-IDENTICAL to a deliberate
out-of-band append, so any comparison performed *afterwards* must either refuse
both (bricking a crashed campaign) or adopt both (blessing a forgery).  The
distinguishing evidence therefore has to be written BEFORE the row:

1. Under the ledger's exclusive ``flock``, the writer computes the exact next
   row -- its chain head and its byte extent -- and records it as the anchor's
   ``intent`` beside the still-current ``committed`` pin.
2. The row is appended and ``F_FULLFSYNC``-ed.
3. The anchor is rewritten with ``committed`` advanced and ``intent`` cleared.

A reader accepts exactly two states: the ledger matches ``committed`` (the row
never landed -- crash in window 1-2, and the declared intent proves it), or the
ledger matches ``intent`` (the row landed and the anchor lagged -- crash in
window 2-3).  Anything else refuses.  An out-of-band append leaves the ledger
matching neither, because ``intent`` is null in steady state and pins the exact
``(rows, bytes, chain_head)`` of one intended row.  That pin is only as strong as
the content address the chain head is derived from, and that address is now a
function of the ROW BODY: :func:`menagerie.crawler.ledger_chain.row_content_digest`
hashes the row minus its chain envelope and minus the self-digest fields, reading
nothing the row declares about itself.  So satisfying an open intent with a
substituted row requires a PREIMAGE, full stop -- not because a second concern
happens to re-derive one field.  Twice before, that claim was false: the chain read
the row's own ``row_sha256`` and nothing recomputed it (PSLRE4 D-1/D-2), and then
it read a foreign ``close_sha256`` a forger could simply add (PSLRE5 P5-D-1).  The
accounting concern's own re-derivation
(:func:`menagerie.crawler.paid_session_accounting.validate_row_digest`) remains, as
an independent authority over the same body through the same shared derivation.
A truncation likewise refuses, because ``committed`` is ahead of the shortened
ledger.  Both directions are closed by an ordering, not by a tolerance.

The anchor is ONE atomically replaced object, so reading or writing it is O(1)
in campaign history.  The ledger itself is read through the incremental
:func:`menagerie.crawler.close_reads.derive` substrate, so each row is decoded,
chain-verified, and folded exactly once per process no matter how many
consultations or appends follow.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.close_reads import derive, ledger_extent
from menagerie.crawler.identity import canonical_json_bytes, fsync_directory, stable_hash
from menagerie.crawler.ledger_chain import (
    ChainState,
    _decode_row,
    extend_chain_state,
    link_row,
    tail_recovery,
)
from menagerie.crawler.recordio import durable_fsync

ANCHOR_SCHEMA_V1 = "menagerie.crawler.paid-session-anchor.v1"
ANCHOR_SCHEMA_V2 = "menagerie.crawler.paid-session-anchor.v2"
ANCHOR_SCHEMA = "menagerie.crawler.paid-session-anchor.v3"

#: Permanent scale-cost instrumentation for the durability concern.  The scale
#: census asserts these stay LINEAR in the number of appends; the regression
#: this exists to prevent was an anchor that was decoded and chain-verified
#: three times per money append and never compacted.
DURABILITY_STATS: dict[str, int] = {
    "ledger_rows_verified": 0,
    "anchor_reads": 0,
    "anchor_writes": 0,
    "ledger_appends": 0,
}


def reset_durability_stats() -> None:
    """Zero the scale-cost counters before one measurement window."""

    for key in DURABILITY_STATS:
        DURABILITY_STATS[key] = 0


@dataclass(frozen=True)
class ExtentPin:
    """The exact external identity and extent of one ledger prefix."""

    ledger_identity: str
    rows: int
    bytes: int
    chain_head: str

    def as_json(self) -> dict[str, Any]:
        """Return the canonical persisted spelling of this pin."""

        return {
            "ledger_identity": self.ledger_identity,
            "rows": self.rows,
            "bytes": self.bytes,
            "chain_head": self.chain_head,
        }

    @classmethod
    def from_json(cls, value: object, *, subject: str) -> ExtentPin:
        """Parse and fail-closed validate one persisted pin."""

        if not isinstance(value, Mapping) or set(value) != {
            "ledger_identity",
            "rows",
            "bytes",
            "chain_head",
        }:
            raise ValueError(f"{subject} anchor pin field contract is invalid")
        for name in ("rows", "bytes"):
            number = value.get(name)
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{subject} anchor {name} must be non-negative")
        for name in ("ledger_identity", "chain_head"):
            text = value.get(name)
            if not isinstance(text, str) or not text:
                raise ValueError(f"{subject} anchor requires {name}")
        return cls(
            ledger_identity=str(value["ledger_identity"]),
            rows=int(value["rows"]),
            bytes=int(value["bytes"]),
            chain_head=str(value["chain_head"]),
        )


@dataclass(frozen=True)
class AnchorState:
    """One anchor: the committed pin plus an optional in-flight intent."""

    committed: ExtentPin
    intent: Optional[ExtentPin]


@dataclass(frozen=True)
class LedgerSnapshot:
    """One verified read: decoded rows, chain state, and the caller's fold."""

    rows: tuple[dict[str, Any], ...]
    chain_state: ChainState
    value: Any


def read_anchor_state(
    anchor_path: Path, *, ledger_name: str, subject: str
) -> Optional[AnchorState]:
    """Read and fail-closed validate one anchor object at an explicit path.

    The parsing that used to live on :class:`AnchoredLedger` alone, factored so
    a PRESERVED anchor (a ``.poisoned-*`` sibling that still names its original
    ledger) authenticates through the exact same code path as the live one.

    Parameters
    ----------
    anchor_path:
        The anchor file to read.
    ledger_name:
        The ledger filename the anchor must name (its original canonical name,
        even when the anchor file itself has been renamed alongside a preserved
        sibling).
    subject:
        Refusal-message subject.

    Returns
    -------
    Optional[AnchorState]
        The validated anchor state, or ``None`` when no anchor file exists.
    """

    if not anchor_path.is_file():
        return None
    DURABILITY_STATS["anchor_reads"] += 1
    payload = anchor_path.read_bytes()
    lines = payload.split(b"\n")
    # An atomic replace can never leave a partial object, so a torn final
    # line only ever comes from a legacy append-only anchor.  Dropping it is
    # the same torn-tail contract every Track A ledger already has.
    if payload and not payload.endswith(b"\n"):
        lines = lines[:-1]
    rows: list[Mapping[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{subject} anchor is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{subject} anchor row is not a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{subject} anchor has no complete durable row")
    return _parse_anchor_rows(rows, ledger_name=ledger_name, subject=subject)


def _parse_anchor_rows(
    rows: Sequence[Mapping[str, Any]], *, ledger_name: str, subject: str
) -> AnchorState:
    """Validate one current anchor object or a legacy anchor prefix."""

    latest = rows[-1]
    schema = latest.get("schema_version")
    if schema == ANCHOR_SCHEMA:
        if len(rows) != 1:
            raise ValueError(f"{subject} anchor must be one atomic object")
        return _parse_current_anchor(latest, ledger_name=ledger_name, subject=subject)
    if schema in {ANCHOR_SCHEMA_V1, ANCHOR_SCHEMA_V2}:
        return AnchorState(
            committed=_parse_legacy_anchor(latest, subject=subject), intent=None
        )
    raise ValueError(f"{subject} anchor schema mismatch")


def _parse_current_anchor(
    row: Mapping[str, Any], *, ledger_name: str, subject: str
) -> AnchorState:
    """Validate the current committed-plus-intent anchor object."""

    expected = {"schema_version", "ledger_name", "committed", "intent", "anchor_sha256"}
    if set(row) != expected:
        raise ValueError(f"{subject} anchor field contract is invalid")
    body = {key: row[key] for key in expected if key != "anchor_sha256"}
    if row.get("anchor_sha256") != stable_hash(body):
        raise ValueError(f"{subject} anchor digest is invalid")
    if row.get("ledger_name") != ledger_name:
        raise ValueError(f"{subject} anchor names a different ledger")
    committed = ExtentPin.from_json(row["committed"], subject=subject)
    intent = (
        None
        if row["intent"] is None
        else ExtentPin.from_json(row["intent"], subject=subject)
    )
    if intent is not None and (
        intent.rows != committed.rows + 1 or intent.bytes <= committed.bytes
    ):
        raise ValueError(f"{subject} anchor intent is not the exact next row")
    return AnchorState(committed=committed, intent=intent)


def _parse_legacy_anchor(row: Mapping[str, Any], *, subject: str) -> ExtentPin:
    """Validate one historical singleton or chained anchor row."""

    expected = {
        "schema_version",
        "ledger_name",
        "ledger_identity",
        "rows",
        "bytes",
        "chain_head",
        "anchor_sha256",
    }
    if row.get("schema_version") == ANCHOR_SCHEMA_V2:
        expected.add("chain")
    if set(row) != expected:
        raise ValueError(f"{subject} anchor field contract is invalid")
    body = {key: row[key] for key in expected if key not in {"anchor_sha256", "chain"}}
    if row.get("anchor_sha256") != stable_hash(body):
        raise ValueError(f"{subject} anchor digest is invalid")
    return ExtentPin.from_json(
        {
            "ledger_identity": row.get("ledger_identity"),
            "rows": row.get("rows"),
            "bytes": row.get("bytes"),
            "chain_head": row.get("chain_head"),
        },
        subject=subject,
    )


def preserved_lineage_verifier(
    directory: Path, *, ledger_name: str, subject: str
) -> Callable[..., None]:
    """Return the replay-time authenticator for a rolled-over predecessor.

    The accounting fold holds this callable (injected through its state seed)
    and calls it on EVERY replay of an ``accounting_rollover`` row, so the
    predecessor's identity is ANCHOR-DERIVED rather than producer-declared: a
    rollover is admissible only while the preserved ``.poisoned-*`` sibling
    still sits beside the ledger with the named whole-file digest AND its
    preserved commit/intent anchor confirms the same identity, extent, and
    chain head.  A synthetic chain-valid row sequence has no anchored on-disk
    predecessor and therefore cannot mint (or replay) a rollover through any
    fold equipped with this verifier; forging one would require the same
    rewrite-and-re-pin filesystem capability the ledger contract already
    discloses as its global residual, never less.

    Parameters
    ----------
    directory:
        The canonical ledger's own directory (where preservation renames land).
    ledger_name:
        The canonical ledger filename (``spend.jsonl``): the preserved sibling
        namespace prefix, and the name the preserved anchor must still declare.
    subject:
        Refusal-message subject.
    """

    def verify(
        *,
        sibling_name: str,
        ledger_sha256: str,
        ledger_bytes: int,
        rows: int,
        chain_head: str,
        ledger_identity: str,
    ) -> None:
        """Authenticate one named predecessor against its on-disk preservation."""

        prefix = f"{ledger_name}.poisoned-"
        if (
            "/" in sibling_name
            or "\\" in sibling_name
            or not sibling_name.startswith(prefix)
            or len(sibling_name) <= len(prefix)
        ):
            raise ValueError(
                f"{subject} rollover names {sibling_name!r}, outside the preserved "
                f"predecessor namespace {prefix}*"
            )
        sibling = directory / sibling_name
        if not sibling.is_file():
            raise ValueError(
                f"{subject} preserved predecessor {sibling_name} is not on disk "
                "beside the ledger"
            )
        payload = sibling.read_bytes()
        if len(payload) != ledger_bytes or stable_hash(
            {"bytes_hex": payload.hex()}
        ) != ledger_sha256:
            raise ValueError(
                f"{subject} preserved predecessor {sibling_name} does not match "
                "the named whole-file digest and byte count"
            )
        anchor_state = read_anchor_state(
            directory / f"{sibling_name}.anchor.json",
            ledger_name=ledger_name,
            subject=subject,
        )
        if anchor_state is None:
            raise ValueError(
                f"{subject} preserved predecessor {sibling_name} has no preserved "
                "anchor: an unanchored predecessor cannot license a rollover"
            )
        expected = ExtentPin(
            ledger_identity=ledger_identity,
            rows=rows,
            bytes=ledger_bytes,
            chain_head=chain_head,
        )
        if anchor_state.committed != expected and anchor_state.intent != expected:
            raise ValueError(
                f"{subject} preserved predecessor anchor does not confirm the "
                "rolled-over identity, extent, and chain head"
            )

    return verify


@dataclass(frozen=True)
class _Projection:
    """The incrementally derived value held per ledger path."""

    rows: tuple[dict[str, Any], ...]
    chain: ChainState
    linked: bool
    value: Any


class AnchoredLedger:
    """An append-only chain-linked ledger with an externally pinned extent.

    Parameters
    ----------
    path:
        Canonical ledger path.
    chain_scope:
        Canonical chain scope of the ledger's own rows.
    subject:
        Refusal-message subject, so a consumer's typed vocabulary survives.
    seed:
        Zero-row factory for the caller's fold value.
    fold:
        ``(value, appended_rows) -> value``, purely additive.
    projection:
        Stable derivation name for the incremental read cache.
    """

    def __init__(
        self,
        path: Path,
        *,
        chain_scope: str,
        subject: str,
        seed: Callable[[], Any],
        fold: Callable[[Any, Sequence[Mapping[str, Any]]], Any],
        projection: str,
    ) -> None:
        """Bind one ledger path to its chain scope and caller-supplied fold."""

        self.path = path
        self.chain_scope = chain_scope
        self.subject = subject
        self._seed = seed
        self._fold = fold
        self._projection = projection

    # -- anchor -----------------------------------------------------------

    @property
    def anchor_path(self) -> Path:
        """Return the external identity-and-extent anchor for this ledger."""

        return self.path.with_name(f"{self.path.name}.anchor.json")

    def _observed_pin(self, chain: ChainState, size: int) -> ExtentPin:
        """Return the pin the ledger's own verified bytes currently assert."""

        identity = chain.heads[0] if chain.heads else chain.head_sha256
        return ExtentPin(
            ledger_identity=identity,
            rows=chain.rows,
            bytes=size,
            chain_head=chain.head_sha256,
        )

    def read_anchor(self) -> Optional[AnchorState]:
        """Return the durable anchor state, or ``None`` when none exists.

        Raises
        ------
        ValueError
            If the anchor exists but does not authenticate.
        """

        return read_anchor_state(
            self.anchor_path, ledger_name=self.path.name, subject=self.subject
        )

    def _write_anchor(self, state: AnchorState) -> None:
        """Atomically replace and power-loss-flush the anchor object."""

        anchor = self.anchor_path
        body = {
            "schema_version": ANCHOR_SCHEMA,
            "ledger_name": self.path.name,
            "committed": state.committed.as_json(),
            "intent": None if state.intent is None else state.intent.as_json(),
        }
        row = {**body, "anchor_sha256": stable_hash(body)}
        anchor.parent.mkdir(parents=True, exist_ok=True)
        temporary = anchor.with_name(f".{anchor.name}.tmp")
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(row) + b"\n")
            handle.flush()
            durable_fsync(handle.fileno())
        os.replace(temporary, anchor)
        fsync_directory(anchor.parent)
        DURABILITY_STATS["anchor_writes"] += 1

    def _resolve_anchor(self, observed: ExtentPin, *, materialized: bool) -> ExtentPin:
        """Return the authoritative pin, or refuse an unexplained ledger extent."""

        anchor = self.read_anchor()
        if anchor is None:
            if observed.rows:
                raise ValueError(
                    f"{self.subject} anchor is missing for a materialized ledger extent"
                )
            return observed
        if observed == anchor.committed:
            return anchor.committed
        if anchor.intent is not None and observed == anchor.intent:
            return anchor.intent
        if not materialized and anchor.committed.rows == 0:
            return anchor.committed
        raise ValueError(
            f"{self.subject} anchor identity or extent does not match {self.path.name}; "
            "the ledger may be missing, truncated, replaced, or appended to out of band"
        )

    # -- reading ----------------------------------------------------------

    def _extend(
        self, value: _Projection, rows: Iterable[tuple[int, int, bytes]]
    ) -> _Projection:
        """Decode, chain-verify, and fold one appended ledger suffix.

        The decode routes through the SHARED row decoder rather than an inline
        ``json.loads``, so the money path's decode work is counted at the boundary
        the scale census reads (PSLRE5 P5-D-2: with the money path decoding its own
        suffix inline, ``rows_decoded`` was a permanent 0.00 and a history re-read
        spelled that way was invisible to every counter).  The refusal vocabulary
        is identical, because it is the same decoder.
        """

        decoded: list[dict[str, Any]] = [
            _decode_row(self.path, number, line) for number, _offset, line in rows
        ]
        if not decoded:
            return value
        DURABILITY_STATS["ledger_rows_verified"] += len(decoded)
        chain = extend_chain_state(
            decoded,
            scope=self.chain_scope,
            state=value.chain,
            linked=value.linked,
            require_links=True,
        )
        return _Projection(
            rows=(*value.rows, *decoded),
            chain=chain,
            linked=True,
            value=self._fold(value.value, decoded),
        )

    def _derive(self, *, locked_handle: Optional[Any] = None) -> _Projection:
        """Derive the incremental verified projection over the whole ledger."""

        return derive(
            self.path,
            projection=self._projection,
            seed=lambda: _Projection(
                rows=(),
                chain=extend_chain_state((), scope=self.chain_scope, state=None),
                linked=False,
                value=self._seed(),
            ),
            extend=self._extend,
            recover=lambda target: tail_recovery(target, locked_handle=locked_handle),
            locked_handle=locked_handle,
            weigh=lambda _value, byte_count: 4096 + 4 * byte_count,
        )

    def read(self, *, locked_handle: Optional[Any] = None) -> LedgerSnapshot:
        """Return one anchor-verified snapshot of the ledger.

        Parameters
        ----------
        locked_handle:
            Open handle whose ``flock`` the caller already holds.

        Returns
        -------
        LedgerSnapshot
            Decoded rows, verified chain state, and the folded projection.
        """

        # The extent is read AFTER the derivation: torn-tail recovery truncates
        # the ledger, so a size captured beforehand would name bytes the reader
        # no longer sees and the anchor would refuse a recovery it just performed.
        projection = self._derive(locked_handle=locked_handle)
        extent = ledger_extent(self.path, locked_handle=locked_handle)
        observed = self._observed_pin(projection.chain, 0 if extent is None else extent.size)
        self._resolve_anchor(observed, materialized=extent is not None)
        return LedgerSnapshot(
            rows=projection.rows, chain_state=projection.chain, value=projection.value
        )

    def read_locked(self) -> LedgerSnapshot:
        """Return one snapshot taken under a shared lock, or the empty ledger."""

        if not self.path.exists():
            anchor = self.read_anchor()
            if anchor is not None and anchor.committed.rows:
                raise ValueError(
                    f"{self.subject} ledger is missing while its anchored extent remains"
                )
            return LedgerSnapshot(
                rows=(),
                chain_state=extend_chain_state((), scope=self.chain_scope, state=None),
                value=self._seed(),
            )
        with self.path.open("r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            return self.read(locked_handle=handle)

    # -- appending --------------------------------------------------------

    def append(
        self,
        build: Callable[[LedgerSnapshot], Mapping[str, Any]],
    ) -> tuple[dict[str, Any], LedgerSnapshot]:
        """Append one row under the anchor's commit-then-intent protocol.

        Parameters
        ----------
        build:
            Called with the locked verified snapshot; returns the logical row to
            append, or raises to refuse the transaction without minting a row.

        Returns
        -------
        tuple[dict[str, Any], LedgerSnapshot]
            The linked row and the snapshot after it.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            projection = self._derive(locked_handle=handle)
            extent = ledger_extent(self.path, locked_handle=handle)
            size = 0 if extent is None else extent.size
            observed = self._observed_pin(projection.chain, size)
            committed = self._resolve_anchor(observed, materialized=True)
            snapshot = LedgerSnapshot(
                rows=projection.rows,
                chain_state=projection.chain,
                value=projection.value,
            )
            row = build(snapshot)
            linked = link_row(
                row,
                scope=self.chain_scope,
                existing_rows=projection.rows,
                existing_state=projection.chain,
            )
            line = canonical_json_bytes(linked) + b"\n"
            next_chain = ChainState(
                rows=projection.chain.rows + 1,
                head_sha256=str(linked["chain"]["sha256"]),
                heads=(*projection.chain.heads, str(linked["chain"]["sha256"])),
            )
            next_pin = self._observed_pin(next_chain, size + len(line))
            # Phase 0: fold the row BEFORE any durable byte.  The caller's fold
            # is the row's own validating authority, and it re-runs on every
            # later read -- so a row the fold refuses must refuse HERE, inside
            # the same exclusive lock, while nothing has landed.  Folding after
            # the write let one writer bug mint a durable row every subsequent
            # read of the honest ledger refuses (the 2026-08-20 step-200
            # duplicate-custody row bricked the whole money authority at the
            # first terminal recording).  The fold is pure over (value, rows),
            # so the successful value doubles as the returned snapshot's.
            next_value = self._fold(projection.value, (dict(linked),))
            # Phase 1: declare the exact next row BEFORE it exists.
            self._write_anchor(AnchorState(committed=committed, intent=next_pin))
            # Phase 2: the row itself, power-loss durable.
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            durable_fsync(handle.fileno())
            DURABILITY_STATS["ledger_appends"] += 1
            # Phase 3: promote the intent to committed.
            self._write_anchor(AnchorState(committed=next_pin, intent=None))
            next_snapshot = LedgerSnapshot(
                rows=(*projection.rows, dict(linked)),
                chain_state=next_chain,
                value=next_value,
            )
        if not existed:
            fsync_directory(self.path.parent)
        return dict(linked), next_snapshot
