"""Close-vs-books drift classification: the recurrence-class tripwire engine.

Failure-catalog class (R7 canary three-door deadlock, 2026-08-31): "out-of-band
operator append to a close-attested source (models/attempts/gates/artifacts,
valves, park.grant/park.refusal) between a leg close and the next boundary
stage." The PD-7 arm (c) hold appends staled the step-200 close, and the
staleness surfaced at the STAGING door — after the acts had landed, when the
remedy leg was already door-B-blocked. This module names the delta class per
census slot so an operator (or a stage preflight) can ask BEFORE an act lands:
"does the books' state still re-observe the licensing close, and if not, what
class of delta is it?"

The classifier is DIAGNOSTIC ONLY. No gate consumes its verdict and no class
it names licenses anything: the shipped remedy for a staled close is a fresh
close minted by the next leg — for a launched successor rung, the
zero-admission ``licensing_close`` leg (:mod:`menagerie.crawler.leg_purpose`,
shape 4) — never a census carve-out (the tri-lab ruling rejected every
gate-side readmission shape as a tripwire disarm).

Second catalog row, from door B (Opus ruling §6): "a rung-class refinement
that retroactively re-classes a snapshot on which pre-launch legs must still
run." :func:`retroactive_reclass_warnings` is that row's WARN tripwire.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from menagerie.crawler.close_reads import prefix_digest
from menagerie.crawler.identity import row_content_address
from menagerie.crawler.leg_generation import (
    CLOSE_SOURCE_NAMES,
    MODELS_SOURCE_NAMES,
    close_ledger_sources,
    source_identity,
    source_row_window,
)

#: Closed per-slot drift vocabulary. ``identical`` is the only class the close
#: gate's re-observation accepts; everything else means the books moved and
#: the next boundary stage will demand a FRESH close.
DRIFT_IDENTICAL = "identical"
DRIFT_PRESENCE_FLIP = "presence-flip"
DRIFT_TRUNCATED = "truncated"
DRIFT_REWRITTEN = "rewritten"
DRIFT_APPENDED = "appended"

#: Closed whole-census delta classes the probe names. ``review-hold-append``
#: is the adoption-remediable class (every drifted slot is an append-only
#: models extent whose every appended row is hold-shaped — the licensing_close
#: leg's adoption pass re-cites exactly these); ``out-of-band-drift`` is
#: everything else, which no leg remedies and an operator must adjudicate.
DELTA_CLASS_NONE = "none"
DELTA_CLASS_REVIEW_HOLD_APPEND = "review-hold-append"
DELTA_CLASS_OUT_OF_BAND = "out-of-band-drift"


@dataclass(frozen=True)
class SourceDrift:
    """One census slot's classified drift since a close.

    Parameters
    ----------
    name:
        Census slot name (:data:`menagerie.crawler.leg_generation.CLOSE_SOURCE_NAMES`).
    drift:
        Closed drift class (``identical`` .. ``appended``).
    attested:
        The close's recorded identity for the slot.
    observed:
        The slot's live identity.
    appended_rows:
        For an ``appended`` models-layout slot: ``(row_index, stable_id,
        content_address, is_review_hold)`` per appended row, in file order.
        ``None`` outside that case (including undecodable appends, which
        classify the slot as ``rewritten`` — an append nothing can read is
        not evidence of anything).
    """

    name: str
    drift: str
    attested: Mapping[str, Any]
    observed: Mapping[str, Any]
    appended_rows: Optional[tuple[tuple[int, str, str, bool], ...]] = None


def is_review_hold_row(row: Mapping[str, Any]) -> bool:
    """Return whether one model row is a review hold over a failure terminal.

    The structural half of the adoption family: ``status.kind == "failed"``
    carrying ``status.human_review.required is True`` (a hold on a non-failed
    head is unadoptable and would keep refusing at the partition proof).
    """

    status = row.get("status")
    if not isinstance(status, Mapping) or status.get("kind") != "failed":
        return False
    human_review = status.get("human_review")
    return isinstance(human_review, Mapping) and human_review.get("required") is True


def _classify_appended_rows(
    path: Path, attested_rows: int, observed_rows: int
) -> Optional[tuple[tuple[int, str, str, bool], ...]]:
    """Decode and classify one slot's appended rows; ``None`` if undecodable."""

    classified: list[tuple[int, str, str, bool]] = []
    for offset, raw in enumerate(source_row_window(path, attested_rows, observed_rows)):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, Mapping) or not isinstance(row.get("stable_id"), str):
            return None
        classified.append(
            (
                attested_rows + offset,
                str(row["stable_id"]),
                row_content_address(row),
                is_review_hold_row(row),
            )
        )
    return tuple(classified)


def close_source_drift(records_root: Path, close: Mapping[str, Any]) -> tuple[SourceDrift, ...]:
    """Classify every census slot's drift between one close and the books.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    close:
        Validated v2 close object whose ``sources`` census anchors the diff.

    Returns
    -------
    tuple[SourceDrift, ...]
        One classified entry per census slot, in fixed universe order.
    """

    sources = close_ledger_sources(records_root)
    attested_census = dict(close.get("sources") or {})
    drifts: list[SourceDrift] = []
    for name in CLOSE_SOURCE_NAMES:
        attested = dict(attested_census.get(name) or {})
        observed = source_identity(sources[name])
        drifts.append(_classify_slot(name, sources[name], attested, observed))
    return tuple(drifts)


def _classify_slot(
    name: str, path: Path, attested: Mapping[str, Any], observed: Mapping[str, Any]
) -> SourceDrift:
    """Classify one slot's drift into the closed vocabulary."""

    if bool(attested.get("present")) != bool(observed.get("present")):
        return SourceDrift(name, DRIFT_PRESENCE_FLIP, attested, observed)
    if attested.get("ledger_sha256") == observed.get("ledger_sha256"):
        return SourceDrift(name, DRIFT_IDENTICAL, attested, observed)
    attested_bytes = int(attested.get("bytes") or 0)
    attested_rows = int(attested.get("rows") or 0)
    if int(observed.get("bytes") or 0) < attested_bytes:
        return SourceDrift(name, DRIFT_TRUNCATED, attested, observed)
    if prefix_digest(path, attested_bytes) != attested.get("ledger_sha256"):
        return SourceDrift(name, DRIFT_REWRITTEN, attested, observed)
    appended_rows: Optional[tuple[tuple[int, str, str, bool], ...]] = None
    if name in MODELS_SOURCE_NAMES:
        appended_rows = _classify_appended_rows(
            path, attested_rows, int(observed.get("rows") or 0)
        )
        if appended_rows is None:
            # An appended extent nothing can decode is hard drift.
            return SourceDrift(name, DRIFT_REWRITTEN, attested, observed)
    return SourceDrift(name, DRIFT_APPENDED, attested, observed, appended_rows)


def drift_delta_class(drifts: tuple[SourceDrift, ...]) -> str:
    """Name the whole-census delta class for one close-vs-books diff.

    ``review-hold-append`` is claimed ONLY when every drifted slot is an
    append-only models-layout extent whose every appended row is hold-shaped —
    the class the next leg's orphan-citation adoption pass remedies (for a
    launched successor rung, a ``licensing_close`` leg). The name is advisory:
    no gate consumes it, and the adoption pass recomputes its own closed rule
    from the bytes.
    """

    moved = [drift for drift in drifts if drift.drift != DRIFT_IDENTICAL]
    if not moved:
        return DELTA_CLASS_NONE
    for drift in moved:
        if drift.drift != DRIFT_APPENDED or drift.name not in MODELS_SOURCE_NAMES:
            return DELTA_CLASS_OUT_OF_BAND
        if not drift.appended_rows or not all(hold for _, _, _, hold in drift.appended_rows):
            return DELTA_CLASS_OUT_OF_BAND
    return DELTA_CLASS_REVIEW_HOLD_APPEND


def retroactive_reclass_warnings(records_root: Path) -> tuple[str, ...]:
    """WARN on rung-class refinements landing over live leg history (door B).

    The second catalog row: a refinement (or any append-once registry
    classing) that retroactively re-classes a snapshot which already has
    durable leg-kind closes changes what every LATER leg on that snapshot
    must present — the exact cross-lane JOIN that sealed the ladder (R2-4
    classed the step-200 tip ``successor_rung`` four legs after it launched).
    Advisory only: refinements are lawful and append-once; the warning names
    the composition so the operator sequences the next boundary act
    deliberately.

    Returns
    -------
    tuple[str, ...]
        One warning per refined snapshot that already had leg-kind closes.
    """

    from menagerie.crawler.close_object import close_object_path, load_close_objects
    from menagerie.crawler.rung_staging import (
        load_rung_class_refinements,
        load_rung_registrations,
    )

    refinements = load_rung_class_refinements(records_root)
    if not refinements:
        return ()
    registrations = {row.snapshot_sha256: row for row in load_rung_registrations(records_root)}
    closes = load_close_objects([close_object_path(records_root)])
    warnings: list[str] = []
    for refinement in refinements:
        snapshot_sha256 = str(refinement.snapshot_sha256)
        registration = registrations.get(snapshot_sha256)
        if registration is None:
            continue
        leg_closes = [
            row
            for row in closes
            if row.get("intake_snapshot_sha256") == snapshot_sha256
            and row.get("close_kind") == "leg"
        ]
        if leg_closes:
            warnings.append(
                f"rung-class refinement declares {refinement.rung_class!r} "
                f"for snapshot {registration.snapshot_id}, which already holds "
                f"{len(leg_closes)} durable leg-kind close(s): the classing is "
                "retroactive over live history, and every later leg on this "
                "snapshot inherits the refined class's certificate demand "
                "(pre-launch legs need the licensing_close purpose lane)"
            )
    return tuple(warnings)
