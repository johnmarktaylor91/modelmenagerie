"""The close-relevant source census and the durable leg generation (v2 core).

Both rt3 breaks were one structural defect on two axes (2026-08-16 tri-lab
adjudication):

* **SPACE** -- the set of byte sources close-consuming decisions READ was a
  superset of the set any close ATTESTED. Three resolvers named the models
  ledger three different ways, so every read-but-unpinned byte was steerable.
  RD-1: the attested set IS the read set. :func:`close_ledger_sources` is the
  ONE enumerator of close-relevant byte sources -- a fixed, closed universe of
  layout SLOTS, each attested present-or-absent -- and every close-consuming
  reader resolves through it.
* **TIME** -- the gate's evidence pool was campaign-global and all-time;
  nothing answered "did THIS leg produce this evidence?". RD-2: every leg has
  a durable GENERATION -- an epoch opened as an append-only ``leg_open`` row
  BEFORE work, whose per-source baselines anchor a half-open evidence window
  ``[floor, close)`` verified by prefix hash. Evidence outside the window
  proves nothing.

The generation is a LOGICAL epoch, not a process: a killed-and-resumed leg
continues its open generation (same floor), and a new generation may open for
a rung only once the prior one is closed. Naming note: this codebase already
spends "generation" on work generations and repair generations; this third
thing is spelled ``leg generation`` everywhere.
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - annotation only, breaks no import cycle
    from menagerie.crawler.state import CurrentRevisionFold

from menagerie.crawler.close_reads import (
    DECODED_ROW_WEIGHT_FACTOR,
    HEAD_DIGEST_WEIGHT,
    LINE_SPAN_WEIGHT,
    PROJECTION_BASE_WEIGHT,
    LedgerRow,
    content_identity,
    derive,
    line_index,
    read_rows,
)
from menagerie.crawler.identity import (
    canonical_json_bytes,
    durable_fsync,
    foreign_content_address_fields,
    fsync_directory,
    hash_bytes,
    row_content_address,
    stable_hash,
    stamp_row_digest,
)
from menagerie.crawler.models import JsonObject

LEG_EVENT_SCHEMA_VERSION = "menagerie.crawler.leg-event.v1"

#: Versioned preimage tag for leg generation ids.
LEG_GENERATION_VERSION = "menagerie-leg-generation-v1"

LEG_EVENTS_RELATIVE_PATH = ("intake", "leg-events.jsonl")

#: The FIXED, CLOSED universe of close-relevant byte sources: every layout of
#: the four fact ledgers the tree can construct (``LedgerPaths`` derives the
#: sharded production layout and the flat driver-rig layout, and the artifact
#: ledger name follows the model shard name), plus the eight valve/park
#: ledgers. A slot that is not on disk is attested ABSENT, never omitted --
#: ``present: false`` is a positive claim, which is what makes materializing a
#: shadow layout a detectable census delta instead of a free steering input.
_FACT_SOURCE_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("models.shard", ("models", "current-shard.jsonl")),
    ("models.flat", ("models.jsonl",)),
    ("attempts.local", ("attempts", "local.jsonl")),
    ("attempts.flat", ("attempts.jsonl",)),
    ("gates.shard", ("gates", "current-shard.jsonl")),
    ("gates.flat", ("gates.jsonl",)),
    ("artifacts.shard", ("artifacts", "current-shard.jsonl")),
    ("artifacts.flat", ("artifacts", "models.jsonl")),
)

CLOSE_SOURCE_NAMES: tuple[str, ...] = (
    *(name for name, _ in _FACT_SOURCE_PATHS),
    "valve.capacity",
    "valve.coverage",
    "valve.checker-lane",
    "valve.rebuild",
    "valve.disk-floor",
    # BL-8 (a)/(b): the scheduler-fault quarantine valve joined the closed
    # universe in the same reviewed change that introduced its ledger; a
    # quarantined member's WITHHELD citation corroborates from its row.
    "valve.scheduler-quarantine",
    "park.grant",
    "park.refusal",
)

#: The models-layout slots, in authority order (sharded production first).
MODELS_SOURCE_NAMES: tuple[str, ...] = ("models.shard", "models.flat")

#: The valve/park slots whose rows corroborate a WITHHELD citation.
WITHHOLDING_SOURCE_NAMES: tuple[str, ...] = (
    "valve.capacity",
    "valve.coverage",
    "valve.checker-lane",
    "valve.rebuild",
    "valve.disk-floor",
    "valve.scheduler-quarantine",
    "park.grant",
    "park.refusal",
)

#: The attempt-ledger slots whose in-window rows mark a member as WORKED this
#: leg (the anti-laundering signal for CARRIED).
ATTEMPT_SOURCE_NAMES: tuple[str, ...] = ("attempts.local", "attempts.flat")


class LegGenerationError(ValueError):
    """Raised when the census, a leg-open row, or a window cannot be trusted."""


def _relative_source_paths() -> dict[str, tuple[str, ...]]:
    """Return the universe's relative paths, resolving the eight valve slots.

    The valve/park relative paths are owned by their modules; the census reads
    them from those modules (lazily -- they are leaf modules but keep imports
    light) so this table can never drift from the writers.
    """

    from menagerie.crawler.checker_lane import checker_lane_deferral_path
    from menagerie.crawler.disk_floor import disk_floor_deferral_path
    from menagerie.crawler.environment_coverage import coverage_deferral_path
    from menagerie.crawler.environment_rebuild import rebuild_deferral_path
    from menagerie.crawler.grant_parks import grant_park_path
    from menagerie.crawler.host_capacity import capacity_deferral_path
    from menagerie.crawler.refusal_parks import refusal_park_path
    from menagerie.crawler.scheduler_parks import scheduler_quarantine_path

    probe = Path("")
    table: dict[str, tuple[str, ...]] = dict(_FACT_SOURCE_PATHS)
    table.update(
        {
            "valve.capacity": tuple(capacity_deferral_path(probe).parts),
            "valve.coverage": tuple(coverage_deferral_path(probe).parts),
            "valve.checker-lane": tuple(checker_lane_deferral_path(probe).parts),
            "valve.rebuild": tuple(rebuild_deferral_path(probe).parts),
            "valve.disk-floor": tuple(disk_floor_deferral_path(probe).parts),
            "valve.scheduler-quarantine": tuple(scheduler_quarantine_path(probe).parts),
            "park.grant": tuple(grant_park_path(probe).parts),
            "park.refusal": tuple(refusal_park_path(probe).parts),
        }
    )
    return table


def close_ledger_sources(records_root: Path) -> dict[str, Path]:
    """Return THE census: every close-relevant source slot below one root.

    The single enumerator of close-relevant byte sources (RD-1). Every slot in
    the fixed universe is returned whether or not it exists on disk; presence
    is a fact :func:`source_identity` attests, never a filter applied here.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    dict[str, pathlib.Path]
        Source name to path, in fixed universe order.
    """

    table = _relative_source_paths()
    return {name: records_root.joinpath(*table[name]) for name in CLOSE_SOURCE_NAMES}


def census_source_bytes(path: Path) -> bytes:
    """Return one census source's bytes, recovering only a torn final line.

    The census attests ``rows``, ``bytes``, and ``ledger_sha256`` over whatever
    is on disk. A torn, non-newline-terminated final line -- a crash mid-append
    on ANY of the census sources -- therefore used to be COUNTED and HASHED into
    the attested identity, so the close's own re-observation could never agree
    with it again and licensing refused permanently with nothing to recover.

    The read routes through recordio's evidence-then-truncate recovery (the
    same one every core ledger and every valve uses) with the chain module's
    strict decoder as the prefix check: only a torn FINAL line is ever removed,
    its bytes are durably evidenced first, and a COMPLETE malformed line stays
    untouched hard corruption.

    Parameters
    ----------
    path:
        Census source JSONL path. A missing file is an empty ledger.
    """

    from menagerie.crawler.ledger_chain import decode_payload
    from menagerie.crawler.recordio import read_with_tail_recovery

    if not path.is_file():
        return b""
    return read_with_tail_recovery(
        path,
        prefix_check=lambda retained: decode_payload(retained.read_bytes(), path),
    )


def _recover_census_tail(path: Path) -> None:
    """Recover a torn final line on one census source before anything reads it.

    The whole-file recovery read is the crash path only: every streaming reader
    below first proves the final byte is a newline with one seek, and calls this
    exclusively when it is not.
    """

    census_source_bytes(path)


def source_identity(path: Path, *, scope: Optional[str] = None) -> JsonObject:
    """Return the exact observed identity of one census source.

    Parameters
    ----------
    path:
        Source JSONL path.

    Returns
    -------
    dict[str, Any]
        ``present``, ``rows`` (non-empty lines), ``bytes``, ``ledger_sha256``
        over the whole file, and ``tail_sha256`` over the last non-empty line.
        An absent source is the honest empty identity with ``present: false``
        -- a positive claim, distinct from an existing zero-byte file.

    Notes
    -----
    Round-2 T4 (scale): the whole-file digest stays a REAL digest over every
    byte -- it is the attestation the close re-observation compares against, and
    hashing bytes is the cheap half. What no longer happens is materializing the
    file to split it (``read_bytes().split(b"\\n")`` on a 20k-model ledger, five
    times per close): rows stream through in bounded chunks and the identity is
    held per observed extent.
    """

    if not path.is_file():
        identity: JsonObject = {
            "present": False,
            "rows": 0,
            "bytes": 0,
            "ledger_sha256": hash_bytes(b""),
            "tail_sha256": None,
        }
        if scope is not None:
            from menagerie.crawler.ledger_chain import chain_state

            identity["chain_head"] = chain_state(path, scope=scope).head_sha256
        return identity
    held = content_identity(
        path, projection="census.identity", recover=_recover_census_tail
    )
    identity = {
        "present": True,
        "rows": held.rows,
        "bytes": held.byte_count,
        "ledger_sha256": held.ledger_sha256,
        "tail_sha256": held.tail_sha256,
    }
    if scope is not None:
        from menagerie.crawler.ledger_chain import chain_state

        identity["chain_head"] = chain_state(path, scope=scope).head_sha256
    return identity


def census_identities(records_root: Path) -> dict[str, JsonObject]:
    """Return every census source's observed identity, in universe order."""

    return {
        name: source_identity(path, scope=name)
        for name, path in close_ledger_sources(records_root).items()
    }


def materialize_close_sources(records_root: Path) -> None:
    """Materialize an empty genesis file for every one of the 15 source slots.

    Presence is consequently fixed before a generation opens.  An honest first
    append advances an existing empty ledger; it never flips an absent slot to
    present inside the window.
    """

    for path in close_ledger_sources(records_root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
            with path.open("rb") as handle:
                durable_fsync(handle.fileno())
            fsync_directory(path.parent)


def source_row_count(path: Path) -> int:
    """Return one source's non-empty row count without decoding a row.

    The window-bound authority for callers that need only the extent.
    """

    if not path.is_file():
        return 0
    return line_index(path, recover=_recover_census_tail).rows


def source_row_window(path: Path, start: int, stop: int) -> tuple[bytes, ...]:
    """Return rows ``[start, stop)`` of one source, newline stripped.

    THE generation-bound row read (round-2 T4): a leg window reads exactly its
    own rows by seeking to row ``start``'s byte offset, instead of
    materializing the campaign's whole history in order to slice the tail off
    it. Row indices are :func:`source_rows`'s, so a citation derived through
    either spelling names the same row.

    Parameters
    ----------
    path:
        Source JSONL path. A missing file has no rows.
    start, stop:
        Half-open row window, clamped to the source's row count.
    """

    if not path.is_file():
        return ()
    index = line_index(path, recover=_recover_census_tail)
    return tuple(
        line.rstrip(b"\n") for line in read_rows(path, index, start, stop)
    )


def source_rows(path: Path) -> tuple[bytes, ...]:
    """Return one source's non-empty lines in file order.

    THE row-index authority: every citation's ``row_index`` and every window
    bound counts these lines, so derivation and verification can never
    disagree about what row N is -- which is exactly why its torn-tail contract
    is :func:`census_source_bytes`'s: a torn tail must not be able to occupy
    row N.

    Prefer :func:`source_row_window` on any path that only needs a leg's own
    window; this spelling is O(campaign-history) BY DEFINITION and exists for
    the readers that genuinely fold all of history.
    """

    if not path.is_file():
        return ()
    return source_row_window(path, 0, source_row_count(path))


def models_source_paths(records_root: Path) -> tuple[Path, ...]:
    """Return every PRESENT models layout, sharded production layout first."""

    sources = close_ledger_sources(records_root)
    return tuple(sources[name] for name in MODELS_SOURCE_NAMES if sources[name].is_file())


@dataclass(frozen=True)
class ModelsSourceProjection:
    """One models layout's close-relevant derivation, held per observed extent.

    THE single fold of a models ledger that the close path needs, replacing
    three whole-history passes per close (a ``scan_jsonl`` for the head
    selection, a ``source_rows`` re-parse for row indices, and a third
    ``source_rows`` re-parse for the award-transition census). Only the
    projection is retained; the decoded rows are not.

    Parameters
    ----------
    rows:
        Non-empty row count folded so far.
    fold:
        Current-revision selection state (heads plus revision digests).
    row_index:
        ``(stable_id, record_revision)`` to its LAST row index, exactly the
        ``source_rows`` enumeration every citation counts in.
    revision_status:
        ``record_revision`` to that row's raw ``status`` value, for the
        award-transition census's predecessor lookup. Presence of the key is
        the census's "this revision exists" predicate.
    """

    rows: int
    fold: "CurrentRevisionFold"
    row_index: Mapping[tuple[str, Any], int]
    revision_status: Mapping[str, Any]


def _extend_models_projection(
    path: Path, value: ModelsSourceProjection, rows: Iterable[LedgerRow]
) -> ModelsSourceProjection:
    """Fold newly appended models rows into a held projection.

    Reproduces the historical read exactly: every row decodes through
    ``recordio``'s validated decoder with ``validate=False`` (so a torn tail
    still refuses and a persisted digest is still re-verified live), the lenient
    shape filter still decides what may select a head, and the row index still
    counts EVERY non-empty row.
    """

    from menagerie.crawler.recordio import (  # noqa: PLC2701 -- the ONE validated row decoder
        _decode_validated_line,
    )
    from menagerie.crawler.state import CurrentRevisionFold, fold_current_revisions

    index = value.rows
    row_index = dict(value.row_index)
    revision_status = dict(value.revision_status)
    candidates: list[JsonObject] = []
    for number, _offset, line in rows:
        row = _decode_validated_line(
            line, ledger_path=path, line_number=number, validate=False
        )
        row_index[(str(row.get("stable_id")), row.get("record_revision"))] = index
        revision = row.get("record_revision")
        if isinstance(revision, str):
            revision_status[revision] = row.get("status")
        # Lenient shape filter, matching the validate=False read: a line that
        # is not even a model row (no stable id, no revision) selects nothing.
        # Chain integrity of the REAL rows is still the current-revision fold's
        # to enforce, and it still raises on a corrupted revision chain.
        if isinstance(row.get("stable_id"), str) and row.get("record_revision") is not None:
            candidates.append(row)
        index += 1
    if index == value.rows:
        return value
    # The fold advances a COPY: a refusal midway through the appended suffix
    # must never leave the held prefix state partially advanced, or the next
    # call would fold the same rows twice.
    advanced = fold_current_revisions(
        CurrentRevisionFold(
            current=dict(value.fold.current),
            seen_revisions=dict(value.fold.seen_revisions),
        ),
        candidates,
    )
    return ModelsSourceProjection(
        rows=index, fold=advanced, row_index=row_index, revision_status=revision_status
    )


def models_source_projection(path: Path) -> ModelsSourceProjection:
    """Return one models layout's held close-relevant projection.

    Parameters
    ----------
    path:
        Models ledger JSONL path. A missing file projects zero rows.

    Raises
    ------
    menagerie.crawler.recordio.TornTailError
        If the final line is not newline terminated -- the historical
        ``scan_jsonl`` contract, deliberately NOT recovered here, because the
        head selection has always refused a torn models ledger rather than
        truncating one behind the caller's back.
    menagerie.crawler.recordio.LedgerCorruptionError
        If a complete row is malformed, digest-inconsistent, or forks the
        revision chain.
    """

    from menagerie.crawler.recordio import TornTailError
    from menagerie.crawler.state import CurrentRevisionFold

    def extend(
        value: ModelsSourceProjection, rows: Iterable[LedgerRow]
    ) -> ModelsSourceProjection:
        return _extend_models_projection(path, value, rows)

    return derive(
        path,
        projection="models.heads",
        seed=lambda: ModelsSourceProjection(
            rows=0,
            fold=CurrentRevisionFold(current={}, seen_revisions={}),
            row_index={},
            revision_status={},
        ),
        extend=extend,
        # Deliberately NOT auto-recovered (fail-closed: award decisions never
        # read a truncated-behind-the-caller's-back models ledger), but the
        # refusal is RECOVERABLE: the named operator tool quarantines the torn
        # tail bytes to the evidence sidecar and discloses the repair on the
        # operational ledger (H1 F6).
        torn_error=lambda offset: TornTailError(
            f"torn final line at byte {offset} in {path}; run "
            "`python -m menagerie.crawler.tools.repair_models_ledger` to "
            "quarantine the torn tail with evidence"
        ),
        # THE heaviest derivation on the close path, and the one round-3 B2 named:
        # it retains a decoded row per CURRENT head plus a row-index and a
        # revision-status entry per row, so at the 20k rung it is ~1.2x the models
        # ledger held for the driver's lifetime. The bound charged here is the
        # decoded weight of every row it could be holding plus its two per-row
        # index entries, which is what makes the held-set budget a real memory
        # budget rather than an entry count.
        weigh=lambda value, byte_count: (
            PROJECTION_BASE_WEIGHT
            + DECODED_ROW_WEIGHT_FACTOR * byte_count
            + 2 * LINE_SPAN_WEIGHT * value.rows
        ),
    )


def current_model_heads(
    records_root: Path, *, require_ledger: bool = True
) -> dict[str, JsonObject]:
    """Return the raw current model heads, unioned across attested layouts.

    Every present models slot contributes its heads; a member whose layouts
    disagree about its current revision is AMBIGUOUS authority and refuses.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    require_ledger:
        Whether the absence of every models layout refuses (the AW-2
        fail-closed posture) rather than returning no heads.

    Raises
    ------
    LegGenerationError
        If no models ledger exists while one is required, or two layouts
        disagree about a member's current head.
    """

    citations = model_head_citations(records_root, require_ledger=require_ledger)
    return {stable_id: entry["record"] for stable_id, entry in citations.items()}


def model_head_citations(
    records_root: Path, *, require_ledger: bool = True
) -> dict[str, JsonObject]:
    """Return each member's current head WITH its exact citation.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    require_ledger:
        Whether the absence of every models layout refuses.

    Returns
    -------
    dict[str, dict[str, Any]]
        Stable id to ``{"source", "row_index", "record"}`` where ``row_index``
        is the cited row's position among the source's non-empty lines.

    Raises
    ------
    LegGenerationError
        Absent ledgers (when required) or cross-layout revision disagreement.
    """

    sources = close_ledger_sources(records_root)
    present = [(name, sources[name]) for name in MODELS_SOURCE_NAMES if sources[name].is_file()]
    if not present and require_ledger:
        raise LegGenerationError(
            f"no models ledger exists under {records_root} (checked "
            "models/current-shard.jsonl and models.jsonl): an absent ledger cannot "
            "prove the absence of runs holders, so staging fails CLOSED (AW-2). "
            "Materialize the campaign's models ledger first -- an EMPTY ledger file "
            "is the explicit statement that no awards exist, and EVERY layout "
            "present is read, so an empty one can never shadow a populated one."
        )
    merged: dict[str, JsonObject] = {}
    for name, path in present:
        projection = models_source_projection(path)
        current = projection.fold.current
        indices = projection.row_index
        for stable_id, head in current.items():
            held = merged.get(stable_id)
            if held is not None and held["record"].get("record_revision") != head.get(
                "record_revision"
            ):
                raise LegGenerationError(
                    f"the models ledger layouts disagree about {stable_id}: "
                    f"{held['source']} holds {held['record'].get('record_revision')} "
                    f"while {name} holds {head.get('record_revision')}. Staging "
                    "fails CLOSED on ambiguous award authority -- reconcile the "
                    "layouts (tools/merge_host_shards.py) before staging."
                )
            if held is None:
                merged[stable_id] = {
                    "source": name,
                    "row_index": indices.get((stable_id, head.get("record_revision")), -1),
                    "record": dict(head),
                }
    return merged


def leg_events_path(records_root: Path) -> Path:
    """Return the append-only leg-event ledger below one records root."""

    return records_root.joinpath(*LEG_EVENTS_RELATIVE_PATH)


@dataclass(frozen=True)
class LegGeneration:
    """One durable leg generation: the evidence epoch of one adjudication pass.

    Parameters
    ----------
    campaign_id, intake_snapshot_id, intake_snapshot_sha256, phase:
        Exactly which rung adjudication this epoch belongs to.
    close_kind:
        ``leg`` for a live driver leg, ``backfill`` for the migration mint.
    generation_id, ordinal:
        Content-derived epoch identity and its 1-based ordinal per
        ``(campaign, snapshot digest, phase, kind)``.
    run_id, opened_at:
        Provenance of the OPENING run; resumed runs continue this epoch.
    baselines:
        Per-source identity at open -- the window floor.
    row_sha256:
        Content address of the durable ``leg_open`` row.
    """

    campaign_id: str
    intake_snapshot_id: str
    intake_snapshot_sha256: str
    phase: Optional[str]
    close_kind: str
    generation_id: str
    ordinal: int
    run_id: str
    opened_at: str
    baselines: Mapping[str, JsonObject]
    row_sha256: str
    #: Declared closed leg purpose (:mod:`menagerie.crawler.leg_purpose`), or
    #: ``None`` for an ordinary leg. Content-addressed in the leg-open row and
    #: immutable for the generation: a resume declaring a different (or
    #: absent) purpose refuses rather than reclassifying the open epoch.
    leg_purpose: Optional[str] = None


def _leg_generation_id(
    *,
    campaign_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str],
    close_kind: str,
    ordinal: int,
) -> str:
    """Return the content-derived identity of one leg generation."""

    return stable_hash(
        {
            "generation_version": LEG_GENERATION_VERSION,
            "campaign_id": campaign_id,
            "intake_snapshot_sha256": intake_snapshot_sha256,
            "phase": phase,
            "close_kind": close_kind,
            "ordinal": ordinal,
        }
    )


def _generation_from_row(row: Mapping[str, Any]) -> LegGeneration:
    """Type one validated leg-open row."""

    return LegGeneration(
        campaign_id=str(row["campaign_id"]),
        intake_snapshot_id=str(row["intake_snapshot_id"]),
        intake_snapshot_sha256=str(row["intake_snapshot_sha256"]),
        phase=row.get("phase"),
        close_kind=str(row["close_kind"]),
        generation_id=str(row["generation_id"]),
        ordinal=int(row["ordinal"]),
        run_id=str(row["run_id"]),
        opened_at=str(row["opened_at"]),
        baselines=dict(row["baselines"]),
        row_sha256=str(row["row_sha256"]),
        leg_purpose=row.get("leg_purpose"),
    )


def _validate_declared_purpose(row: Mapping[str, Any]) -> None:
    """Validate one leg-open row's optional declared purpose.

    Versioned-monotone: rows minted before the purpose vocabulary existed
    carry no field and stay valid ordinary legs; a present value must belong
    to the closed vocabulary.
    """

    if "leg_purpose" not in row:
        return
    from menagerie.crawler.leg_purpose import LEG_PURPOSES

    if row["leg_purpose"] not in LEG_PURPOSES:
        raise LegGenerationError(
            f"leg-open row declares unknown leg purpose {row['leg_purpose']!r}"
        )


def _normalized_declared_purpose(leg_purpose: Optional[str]) -> Optional[str]:
    """Normalize a DECLARED purpose; an ordinary invocation touches no vocabulary.

    RC-1: the crawl invocation declares no purpose, and the leg-purpose module
    is a retired door -- only a declared purpose imports and normalizes it.
    """

    if leg_purpose is None:
        return None
    from menagerie.crawler.leg_purpose import normalized_leg_purpose

    return normalized_leg_purpose(leg_purpose)


def _require_matching_purpose(open_row: Mapping[str, Any], leg_purpose: Optional[str]) -> None:
    """Refuse continuing an open generation under a different declared purpose."""

    if open_row.get("leg_purpose") == leg_purpose:
        return
    raise LegGenerationError(
        f"leg generation {open_row['generation_id']} is open with "
        f"declared purpose {open_row.get('leg_purpose')!r}; this "
        f"invocation declares {leg_purpose!r}. The purpose is "
        "immutable per generation — a resume never reclassifies "
        "an open epoch. Close the open generation (or resume it "
        "under its own declared purpose) instead"
    )


def validate_leg_open_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one leg-open row and return its canonical copy.

    Raises
    ------
    LegGenerationError
        If a field is missing/mistyped, the baselines do not cover the closed
        source universe exactly, or the content address disagrees.
    """

    if not isinstance(row, Mapping):
        raise LegGenerationError("leg-open row is not an object")
    if row.get("schema_version") != LEG_EVENT_SCHEMA_VERSION:
        raise LegGenerationError(
            f"leg-open row declares unknown schema {row.get('schema_version')!r}"
        )
    if row.get("event") != "leg_open":
        raise LegGenerationError(f"leg event {row.get('event')!r} is not a leg_open")
    for field in (
        "campaign_id",
        "intake_snapshot_id",
        "intake_snapshot_sha256",
        "close_kind",
        "generation_id",
        "run_id",
        "opened_at",
        "row_sha256",
    ):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise LegGenerationError(f"leg-open row has an empty {field}")
    if row["close_kind"] not in ("leg", "backfill"):
        raise LegGenerationError(f"leg-open row declares unknown close kind {row['close_kind']!r}")
    if not (row.get("phase") is None or isinstance(row.get("phase"), str)):
        raise LegGenerationError("leg-open row phase must be a string or null")
    # D-22/DX-23 invocation attribution, versioned-monotone: rows minted before
    # the invocation ledger existed carry no field and stay valid.
    if "invocation_id" in row and (
        not isinstance(row["invocation_id"], str) or not row["invocation_id"].strip()
    ):
        raise LegGenerationError(
            "leg-open row invocation_id must be a non-empty string when present"
        )
    ordinal = row.get("ordinal")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1:
        raise LegGenerationError("leg-open row ordinal must be a positive integer")
    _validate_declared_purpose(row)
    baselines = row.get("baselines")
    if not isinstance(baselines, Mapping) or set(baselines) != set(CLOSE_SOURCE_NAMES):
        raise LegGenerationError(
            "leg-open row baselines must cover the closed source universe exactly"
        )
    for name, identity in baselines.items():
        if not isinstance(identity, Mapping) or not isinstance(identity.get("ledger_sha256"), str):
            raise LegGenerationError(f"leg-open row baseline {name} is malformed")
    validated = dict(json.loads(canonical_json_bytes(dict(row)).decode("utf-8")))
    chain = validated.pop("chain", None)
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise LegGenerationError(
            f"leg-event row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise LegGenerationError("leg-open row digest does not match its payload")
    validated["row_sha256"] = digest
    if chain is not None:
        validated["chain"] = chain
    return validated


def validate_leg_event_row(row: Mapping[str, Any]) -> JsonObject:
    """Validate one leg-event row under the closed event vocabulary."""

    event = row.get("event")
    if event == "leg_open":
        return validate_leg_open_row(row)
    if event not in ("leg_run", "leg_plan", "leg_withholding", "leg_started", "leg_ended"):
        raise LegGenerationError(f"leg event {event!r} is outside the closed vocabulary")
    required = (
        "schema_version",
        "event",
        "campaign_id",
        "generation_id",
        "run_id",
        "created_at",
        "row_sha256",
    )
    for field in required:
        if not isinstance(row.get(field), str) or not str(row[field]).strip():
            raise LegGenerationError(f"{event} row has an empty {field}")
    if row.get("schema_version") != LEG_EVENT_SCHEMA_VERSION:
        raise LegGenerationError(f"{event} row declares an unknown schema")
    # D-22/DX-23 invocation attribution, versioned-monotone (see leg_open).
    if "invocation_id" in row and (
        not isinstance(row["invocation_id"], str) or not row["invocation_id"].strip()
    ):
        raise LegGenerationError(
            f"{event} row invocation_id must be a non-empty string when present"
        )
    if event == "leg_plan":
        admitted = row.get("admitted_ids")
        if (
            not isinstance(admitted, Sequence)
            or isinstance(admitted, (str, bytes))
            or list(admitted) != sorted({str(value) for value in admitted})
        ):
            raise LegGenerationError("leg_plan admitted_ids must be a sorted unique sequence")
        if not isinstance(row.get("plan_sha256"), str):
            raise LegGenerationError("leg_plan has no plan_sha256")
    if event == "leg_withholding":
        for field in ("stable_id", "withholding_row_sha256"):
            if not isinstance(row.get(field), str) or not str(row[field]).strip():
                raise LegGenerationError(f"leg_withholding row has an empty {field}")
    if event == "leg_started":
        for field in ("invocation_origin",):
            if not isinstance(row.get(field), str) or not str(row[field]).strip():
                raise LegGenerationError(f"leg_started row has an empty {field}")
        ordinal = row.get("invocation_ordinal")
        if not isinstance(ordinal, int) or ordinal < 0:
            raise LegGenerationError("leg_started invocation_ordinal must be a non-negative int")
    if event == "leg_ended":
        from menagerie.crawler.leg_exit_vocabulary import LegExitClass

        exit_class = row.get("exit_class")
        if exit_class not in {member.value for member in LegExitClass}:
            raise LegGenerationError(
                f"leg_ended exit_class {exit_class!r} is outside the closed vocabulary"
            )
        for field in ("cause_fingerprint", "leg_started_sha256", "invocation_origin"):
            if not isinstance(row.get(field), str) or not str(row[field]).strip():
                raise LegGenerationError(f"leg_ended row has an empty {field}")
        delta = row.get("terminal_delta")
        if delta is not None and (not isinstance(delta, int) or delta < 0):
            raise LegGenerationError(
                "leg_ended terminal_delta must be a non-negative int or null"
            )
        if not isinstance(row.get("synthesized"), bool):
            raise LegGenerationError("leg_ended synthesized must be a bool")
        if (
            row["synthesized"]
            and exit_class != LegExitClass.ORPHANED_UNCONTAINED.value
        ) or (
            not row["synthesized"]
            and exit_class == LegExitClass.ORPHANED_UNCONTAINED.value
        ):
            raise LegGenerationError(
                "orphaned_uncontained is synthesized by the boot reconciler XOR "
                "a live leg types its own exit"
            )
    validated = dict(json.loads(canonical_json_bytes(dict(row)).decode("utf-8")))
    chain = validated.pop("chain", None)
    foreign = foreign_content_address_fields(validated, own="row_sha256")
    if foreign:
        raise LegGenerationError(
            f"leg-event row carries a foreign content-address field "
            f"{', '.join(foreign)}: a row of this family owns exactly one content address"
        )
    digest = validated.pop("row_sha256")
    if row_content_address(validated) != digest:
        raise LegGenerationError(f"{event} row digest does not match its payload")
    validated["row_sha256"] = digest
    if chain is not None:
        validated["chain"] = chain
    return validated


def load_leg_events(records_root: Path) -> tuple[JsonObject, ...]:
    """Load and chain-verify every durable event in ledger order.

    Round-2 T4 (scale): the chain verification and the per-row validation (a
    canonical re-serialization plus a digest per row) are HELD, so the four
    close-path consumers -- ``generation_run_ids``, ``sealed_selection_plan``,
    ``find_leg_open_row``, ``load_leg_open_rows`` -- validate each event once
    per observed ledger state instead of once per consumer per close.
    """

    from menagerie.crawler.ledger_chain import decode_ledger, verified_chain

    path = leg_events_path(records_root)
    decode_ledger(path)
    verified_chain(path, scope="meta.leg-events")
    return _validated_leg_events(path)


def _validated_leg_events(path: Path) -> tuple[JsonObject, ...]:
    """Return the held per-row validation of one leg-event ledger."""

    from menagerie.crawler.ledger_chain import (  # noqa: PLC2701 -- the ONE chain row decoder
        _decode_row,
    )

    def extend(
        value: tuple[JsonObject, ...], observed: Iterable[LedgerRow]
    ) -> tuple[JsonObject, ...]:
        decoded = tuple(
            _decode_row(path, number, line) for number, _offset, line in observed
        )
        return (*value, *(validate_leg_event_row(row) for row in decoded))

    validated: tuple[JsonObject, ...] = derive(
        path,
        projection="leg-events.validated",
        seed=tuple,
        extend=extend,
        recover=lambda target: _recover_chain_tail(target),
        # Every validated leg-event row, retained. O(#closes), never O(history).
        weigh=lambda _value, byte_count: (
            PROJECTION_BASE_WEIGHT + DECODED_ROW_WEIGHT_FACTOR * byte_count
        ),
    )
    return validated


def _recover_chain_tail(path: Path) -> None:
    """Recover a torn final line on one Track A meta ledger."""

    from menagerie.crawler.ledger_chain import tail_recovery

    tail_recovery(path)


def load_leg_open_rows(records_root: Path) -> tuple[JsonObject, ...]:
    """Load every persisted leg-open row in ledger order.

    Malformed lines refuse: the leg-event ledger is close machinery, and a
    row nothing can validate is a row nothing can trust a floor to.
    """

    return tuple(row for row in load_leg_events(records_root) if row["event"] == "leg_open")


def generation_run_ids(records_root: Path, generation_id: str) -> frozenset[str]:
    """Return run ids durably joined to one leg generation."""

    return frozenset(
        str(row["run_id"])
        for row in load_leg_events(records_root)
        if row["event"] == "leg_run" and row["generation_id"] == generation_id
    )


def sealed_selection_plan(
    records_root: Path, generation_id: str
) -> tuple[frozenset[str], str]:
    """Return the monotone admitted union and its replay-derived digest."""

    rows = [
        row
        for row in load_leg_events(records_root)
        if row["event"] == "leg_plan" and row["generation_id"] == generation_id
    ]
    if not rows:
        raise LegGenerationError(
            f"leg generation {generation_id} sealed no selection plan: R9 cannot "
            "be waived by producer omission"
        )
    admitted = frozenset(
        str(member) for row in rows for member in row.get("admitted_ids", ())
    )
    digest = stable_hash(
        {
            "generation_id": generation_id,
            "plan_rows": [str(row["row_sha256"]) for row in rows],
            "admitted_ids": sorted(admitted),
        }
    )
    return admitted, digest


def _append_leg_event(records_root: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one validated leg event with a Track A physical chain link."""

    from menagerie.crawler.ledger_chain import decode_ledger, link_row

    path = leg_events_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing = decode_ledger(path, locked_handle=handle)
        for candidate in existing:
            if candidate.get("row_sha256") == row.get("row_sha256"):
                return validate_leg_event_row(candidate)
        linked = link_row(row, scope="meta.leg-events", existing_rows=existing)
        validated = validate_leg_event_row(linked)
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def record_leg_plan(
    records_root: Path,
    *,
    campaign_id: str,
    generation_id: str,
    run_id: str,
    admitted_ids: Sequence[str],
    plan_payload: Mapping[str, Any],
    created_at: str,
) -> JsonObject:
    """Seal one WS-1 selection plan before its admitted work has side effects."""

    body: JsonObject = {
        "schema_version": LEG_EVENT_SCHEMA_VERSION,
        "event": "leg_plan",
        "campaign_id": campaign_id,
        "generation_id": generation_id,
        "run_id": run_id,
        "created_at": created_at,
        "admitted_ids": sorted({str(value) for value in admitted_ids}),
        "plan_sha256": stable_hash(dict(plan_payload)),
    }
    row = stamp_row_digest(body)
    return _append_leg_event(records_root, row)


def record_leg_withholding(
    records_root: Path,
    *,
    campaign_id: str,
    generation_id: str,
    run_id: str,
    stable_id: str,
    withholding_row_sha256: str,
    created_at: str,
) -> JsonObject:
    """Bind one resume short-circuit to its exact durable withholding row.

    Parameters
    ----------
    records_root:
        Canonical campaign records directory.
    campaign_id, generation_id, run_id:
        Current leg and joined-run provenance.
    stable_id:
        Selected member withheld by the resume decision.
    withholding_row_sha256:
        Exact live valve row reused instead of re-running the member.
    created_at:
        UTC timestamp of the decision.

    Returns
    -------
    dict[str, Any]
        Chained, content-addressed leg-withholding receipt.
    """

    body: JsonObject = {
        "schema_version": LEG_EVENT_SCHEMA_VERSION,
        "event": "leg_withholding",
        "campaign_id": campaign_id,
        "generation_id": generation_id,
        "run_id": run_id,
        "created_at": created_at,
        "stable_id": stable_id,
        "withholding_row_sha256": withholding_row_sha256,
    }
    row = stamp_row_digest(body)
    return _append_leg_event(records_root, row)


def record_leg_started(
    records_root: Path,
    *,
    campaign_id: str,
    generation_id: str,
    run_id: str,
    created_at: str,
    invocation_origin: str,
    invocation_ordinal: int,
    pid: int,
) -> JsonObject:
    """Open one leg's lifecycle span (W6): a durable "a leg is live" claim.

    Emitted once per process invocation right after the generation opens.
    A ``leg_started`` whose ``row_sha256`` no ``leg_ended`` ever references
    is an ORPHAN the boot reconciler must close ``orphaned_uncontained``
    before any new leg opens -- that pairing is the whole census.
    """

    body: JsonObject = {
        "schema_version": LEG_EVENT_SCHEMA_VERSION,
        "event": "leg_started",
        "campaign_id": campaign_id,
        "generation_id": generation_id,
        "run_id": run_id,
        "created_at": created_at,
        "invocation_origin": invocation_origin,
        "invocation_ordinal": int(invocation_ordinal),
        "pid": int(pid),
    }
    row = stamp_row_digest(body)
    return _append_leg_event(records_root, row)


def record_leg_ended(
    records_root: Path,
    *,
    campaign_id: str,
    generation_id: str,
    run_id: str,
    created_at: str,
    leg_started_sha256: str,
    exit_class: str,
    cause_fingerprint: str,
    terminal_delta: Optional[int],
    invocation_origin: str,
    detail: str = "",
    synthesized: bool = False,
) -> JsonObject:
    """Close one leg's lifecycle span with its typed outcome (W6/FT-a).

    Every ``_run_locked`` exit path lands here with exactly one closed
    :class:`~menagerie.crawler.leg_exit_vocabulary.LegExitClass`, the PM-5
    cause fingerprint, and the terminals-minted-this-leg delta. Only the
    boot reconciler may pass ``synthesized=True`` (and only for orphans).
    """

    body: JsonObject = {
        "schema_version": LEG_EVENT_SCHEMA_VERSION,
        "event": "leg_ended",
        "campaign_id": campaign_id,
        "generation_id": generation_id,
        "run_id": run_id,
        "created_at": created_at,
        "leg_started_sha256": leg_started_sha256,
        "exit_class": exit_class,
        "cause_fingerprint": cause_fingerprint,
        "terminal_delta": terminal_delta,
        "invocation_origin": invocation_origin,
        "detail": str(detail)[:1_500],
        "synthesized": bool(synthesized),
    }
    row = stamp_row_digest(body)
    return _append_leg_event(records_root, row)


def orphaned_leg_started_rows(
    leg_events: Sequence[Mapping[str, Any]],
) -> tuple[JsonObject, ...]:
    """Return every ``leg_started`` no ``leg_ended`` references, in order."""

    ended = {
        str(row.get("leg_started_sha256"))
        for row in leg_events
        if row.get("event") == "leg_ended"
    }
    return tuple(
        dict(row)
        for row in leg_events
        if row.get("event") == "leg_started" and str(row.get("row_sha256")) not in ended
    )


def generation_withholding_receipts(
    records_root: Path, generation_id: str
) -> dict[str, tuple[JsonObject, ...]]:
    """Return exact resume-withholding receipts for one leg generation.

    Parameters
    ----------
    records_root:
        Canonical campaign records directory.
    generation_id:
        Generation whose chained receipts are requested.

    Returns
    -------
    dict[str, tuple[dict[str, Any], ...]]
        Stable-ID keyed receipts in ledger order.
    """

    receipts: dict[str, list[JsonObject]] = {}
    for row in load_leg_events(records_root):
        if row["event"] != "leg_withholding" or row["generation_id"] != generation_id:
            continue
        receipts.setdefault(str(row["stable_id"]), []).append(dict(row))
    return {stable_id: tuple(rows) for stable_id, rows in receipts.items()}


def find_leg_open_row(records_root: Path, row_sha256: str) -> Optional[JsonObject]:
    """Return the leg-open row with one content address, if durable."""

    for row in load_leg_open_rows(records_root):
        if row["row_sha256"] == row_sha256:
            return row
    return None


def _closed_generation_ids(records_root: Path) -> frozenset[str]:
    """Return every generation id any persisted close row names.

    Read LENIENTLY: this only decides whether a new epoch must be minted, and
    a malformed close row can therefore only force a fresh generation, never
    resurrect an old one.
    """

    from menagerie.crawler.close_object import close_object_path

    path = close_object_path(records_root)
    if not path.is_file():
        return frozenset()

    def extend(value: frozenset[str], rows: Iterable[LedgerRow]) -> frozenset[str]:
        named = set(value)
        for _number, _offset, line in rows:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, Mapping):
                generation_id = row.get("generation_id")
                if isinstance(generation_id, str) and generation_id:
                    named.add(generation_id)
        return frozenset(named)

    closed: frozenset[str] = derive(
        path,
        projection="close.generations",
        seed=frozenset,
        extend=extend,
        # One generation-id digest string per close row.
        weigh=lambda value, _byte_count: (
            PROJECTION_BASE_WEIGHT + HEAD_DIGEST_WEIGHT * len(value)
        ),
    )
    return closed


def open_leg_generation(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str] = None,
    run_id: str,
    opened_at: str,
    close_kind: str = "leg",
    invocation_id: Optional[str] = None,
    leg_purpose: Optional[str] = None,
) -> LegGeneration:
    """Open the leg generation for one rung adjudication, or continue it.

    The epoch is durable EVIDENCE written before any work: baselines are the
    census identities at open, and they are the window floor everything this
    leg cites is verified against. A generation already open for the same
    ``(campaign, snapshot digest, phase, kind)`` and not yet named by any
    close is CONTINUED -- kill/resume legs span runs -- with its original
    floor. A new epoch begins only after the prior one closed.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id, intake_snapshot_id, intake_snapshot_sha256, phase:
        Exactly which rung adjudication opens.
    run_id, opened_at:
        Opening-run provenance.
    close_kind:
        ``leg`` or ``backfill``.
    invocation_id:
        Durable invocation-ledger row this leg start is attributed to
        (D-22/DX-23). Versioned-monotone: omitted, the rows stay era-legacy.
    leg_purpose:
        Declared closed leg purpose (:mod:`menagerie.crawler.leg_purpose`),
        or ``None`` for an ordinary leg. Immutable per generation: continuing
        an open generation with a DIFFERENT declared purpose (either
        direction) refuses — a resume can never reclassify the open epoch,
        which is what blocks laundering an earlier production generation
        under the weaker purpose (Sol contract point 1).

    Returns
    -------
    LegGeneration
        The open (possibly continued) generation.
    """

    from menagerie.crawler.ledger_chain import decode_ledger, link_row
    from menagerie.crawler.records_root_guard import (  # noqa: PLC0415 -- lazy: guard is leaf-light
        assert_and_stamp_records_root_writer,
        assert_records_root_flock_safe,
    )

    # LC-22/UU-9 interim refusals: every ledger guarantee below assumes this
    # root's flock actually excludes writers, so a leg may not open on a
    # filesystem where that is false, and a root last written by another
    # machine must be claimed deliberately before this host writes it.
    assert_records_root_flock_safe(records_root)
    assert_and_stamp_records_root_writer(records_root)

    path = leg_events_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        existing_events = decode_ledger(path, locked_handle=handle)
        from menagerie.crawler.ledger_chain import verify_rows

        # D-27 require_links hardening (O B-9): a leg-events ledger whose
        # first row is linked has no historical link-free prefix to tolerate.
        verify_rows(
            existing_events,
            scope="meta.leg-events",
            require_links=bool(
                existing_events and existing_events[0].get("chain") is not None
            ),
        )
        existing = [
            validate_leg_open_row(row) for row in existing_events if row.get("event") == "leg_open"
        ]
        mine = [
            row
            for row in existing
            if row["campaign_id"] == campaign_id
            and row["intake_snapshot_sha256"] == intake_snapshot_sha256
            and row.get("phase") == phase
            and row["close_kind"] == close_kind
        ]
        closed = _closed_generation_ids(records_root)
        open_rows = [row for row in mine if row["generation_id"] not in closed]
        leg_purpose = _normalized_declared_purpose(leg_purpose)
        if open_rows:
            open_row = open_rows[-1]
            _require_matching_purpose(open_row, leg_purpose)
            generation = _generation_from_row(open_row)
            joined = {
                str(row.get("run_id"))
                for row in existing_events
                if row.get("event") == "leg_run"
                and row.get("generation_id") == generation.generation_id
            }
            if run_id not in joined:
                run_body: JsonObject = {
                    "schema_version": LEG_EVENT_SCHEMA_VERSION,
                    "event": "leg_run",
                    "campaign_id": campaign_id,
                    "generation_id": generation.generation_id,
                    "run_id": run_id,
                    "created_at": opened_at,
                }
                if invocation_id is not None:
                    run_body["invocation_id"] = invocation_id
                run_row = stamp_row_digest(run_body)
                linked_run = link_row(
                    run_row, scope="meta.leg-events", existing_rows=existing_events
                )
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json_bytes(validate_leg_event_row(linked_run)) + b"\n")
                handle.flush()
                durable_fsync(handle.fileno())
            return generation
        materialize_close_sources(records_root)
        ordinal = len(mine) + 1
        body: JsonObject = {
            "schema_version": LEG_EVENT_SCHEMA_VERSION,
            "event": "leg_open",
            "campaign_id": campaign_id,
            "intake_snapshot_id": intake_snapshot_id,
            "intake_snapshot_sha256": intake_snapshot_sha256,
            "phase": phase,
            "close_kind": close_kind,
            "generation_id": _leg_generation_id(
                campaign_id=campaign_id,
                intake_snapshot_sha256=intake_snapshot_sha256,
                phase=phase,
                close_kind=close_kind,
                ordinal=ordinal,
            ),
            "ordinal": ordinal,
            "run_id": run_id,
            "opened_at": opened_at,
            "baselines": census_identities(records_root),
        }
        if leg_purpose is not None:
            body["leg_purpose"] = leg_purpose
        if invocation_id is not None:
            body["invocation_id"] = invocation_id
        row = stamp_row_digest(body)
        linked = link_row(row, scope="meta.leg-events", existing_rows=existing_events)
        validated = validate_leg_open_row(linked)
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        run_body = {
            "schema_version": LEG_EVENT_SCHEMA_VERSION,
            "event": "leg_run",
            "campaign_id": campaign_id,
            "generation_id": validated["generation_id"],
            "run_id": run_id,
            "created_at": opened_at,
        }
        if invocation_id is not None:
            run_body["invocation_id"] = invocation_id
        run_row = stamp_row_digest(run_body)
        linked_run = link_row(
            run_row,
            scope="meta.leg-events",
            existing_rows=(*existing_events, validated),
        )
        handle.write(canonical_json_bytes(validate_leg_event_row(linked_run)) + b"\n")
        handle.flush()
        durable_fsync(handle.fileno())
    fsync_directory(path.parent)
    return _generation_from_row(validated)


def window_floor_refusals(records_root: Path, baselines: Mapping[str, Any]) -> tuple[str, ...]:
    """Verify every window floor by prefix hash against the bytes on disk.

    I-TIME: a floor is believed only when the file's first ``baseline.bytes``
    bytes still hash to the identity recorded at open. A rewritten history
    cannot reproduce the prefix; a shrunk ledger cannot supply it.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    baselines:
        Per-source identities recorded at leg open.

    Returns
    -------
    tuple[str, ...]
        Human-readable refusals; empty when every floor verifies.
    """

    from menagerie.crawler.close_reads import ledger_extent, prefix_digest

    findings: list[str] = []
    sources = close_ledger_sources(records_root)
    for name in CLOSE_SOURCE_NAMES:
        baseline = dict(baselines.get(name) or {})
        if not baseline:
            findings.append(f"the leg open recorded no {name} baseline")
            continue
        path = sources[name]
        extent = ledger_extent(path)
        observed_bytes = 0 if extent is None else extent.size
        prefix_length = int(baseline.get("bytes") or 0)
        if observed_bytes < prefix_length:
            findings.append(
                f"{name} shrank below its window floor ({observed_bytes} bytes on "
                f"disk, floor at {prefix_length}): append-only was violated"
            )
            continue
        # The floor is proven by hashing exactly the pinned prefix in bounded
        # chunks; the old ``read_bytes()`` pulled the whole growing ledger into
        # memory for all 15 slots to hash a leading slice of each.
        if prefix_digest(path, prefix_length) != baseline.get("ledger_sha256"):
            findings.append(
                f"{name}'s pre-window bytes no longer reproduce the floor prefix "
                "recorded at leg open: history was rewritten under the leg"
            )
    return tuple(findings)
