"""The closed leg-purpose vocabulary and the licensing-close capability forfeit.

Shape 4 of the R7 pre-launch ladder unblock (tri-lab KERNEL_DECISION,
2026-08-31: Opus verdict `.ladder-verdict.md` shape 4, Sol recon
ACCEPT-SHAPE-4 with the six-point safety contract). Door B (the retired
``leg_class`` certificate-kind derivation; RC-3 removed certificate kinds with
the launch certificate) keyed the demanded certificate kind on the
bound RUNG's class, so once a rung is refined ``successor_rung`` every later
leg demands ``rung_launch`` — including the zero-admission licensing-close leg
that must run BEFORE R7 can exist (the licensing guard's own contract names
the next leg as the remedy for a staled close). The class vocabulary has no
value meaning "a launched rung whose remaining production work is gated on
R7"; the R2-4 refinement overloaded ``successor_rung`` to carry a gating
policy, and that overload deadlocked the ladder.

The correction is a DECLARED, CLOSED leg purpose whose only effect is a
strictly weaker leg:

* ``licensing_close`` buys ``canary_admission`` on a ``successor_rung``
  snapshot at the price of admitting ZERO members. The declaration is not
  authority — the weakening is enforced, never asserted: the WS-1 selection
  plan seals EMPTY, every paid admission path refuses typed (including the
  pre-plan terminal-review backlog sweep, which buys checker sessions), and
  the close refuses any non-empty admitted union, any paid-session
  reservation bound to the generation, any positive disposition change, and
  any in-window mutation outside the enumerated adoption / release-void /
  close families.

The purpose is content-addressed into the durable ``leg_open`` row BEFORE the
first ledger mutation and is immutable for the generation: every ``run`` /
``resume`` replays the open row, and a different or absent purpose refuses
rather than reclassifying the open generation
(:func:`menagerie.crawler.leg_generation.open_leg_generation`). The close row
repeats the purpose and binds the exact ``leg_open_row_sha256``, so the leg's
own replay is checkable and the census shows ``adjudicated_here == []``.

Unsafe formulations this module deliberately is NOT (killed in the recon —
each turns a producer-controlled fact into weaker licensing):

* a bare ``--leg-class`` flag selecting the certificate kind directly;
* "the rung already launched / has a prior leg close" relaxing the kind
  (permanently weakens every later production leg on the rung);
* "the newest close is stale" relaxing the kind (an out-of-band append can
  manufacture the predicate that weakens the gate);
* any generic "continuation leg" class (a temporal description, not a
  bounded capability).

Four-part doctrine check (this is a correction to a mis-keyed derivation, not
a waiver, but it passes anyway): (1) disposition-invariant — the one unlocked
capability publishes nothing positive and is strictly weaker than what the
tip's prior legs lawfully did under ``canary_admission``; (2) closed proof
rule — one enumerated purpose value, the kind map stays total over a closed
vocabulary; (3) replay-derived — the declaration steers only DOWNWARD and
every consequence is recomputed from ledger bytes on replay, no persisted
boolean steers a verdict; (4) disclosed — the preflight's
``expected_kind_basis`` names the purpose, the close carries it, and the
census shows the leg adjudicated nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Iterable, Mapping

#: The one purpose value in the closed vocabulary. Widening this tuple is a
#: new leg-purpose law and needs its own deliberated, reviewed change under
#: the same bar as the four-part waiver doctrine's closed proof rules.
LEG_PURPOSE_LICENSING_CLOSE: Final[str] = "licensing_close"

LEG_PURPOSES: Final[tuple[str, ...]] = (LEG_PURPOSE_LICENSING_CLOSE,)


class LegPurposeError(ValueError):
    """Raised when a declared leg purpose is unknown or its forfeit is violated."""


def validate_leg_purpose(value: Any) -> str:
    """Return one validated purpose value from the closed vocabulary.

    Parameters
    ----------
    value:
        Candidate purpose.

    Raises
    ------
    LegPurposeError
        If the value is not a member of :data:`LEG_PURPOSES`.
    """

    if value not in LEG_PURPOSES:
        raise LegPurposeError(
            f"leg purpose {value!r} is outside the closed vocabulary "
            f"{sorted(LEG_PURPOSES)}"
        )
    return str(value)


def normalized_leg_purpose(value: Any) -> str | None:
    """Return ``None`` unchanged, else the validated closed-vocabulary purpose."""

    return None if value is None else validate_leg_purpose(value)


def _window_rows(path: Path, start: int, stop: int) -> tuple[tuple[int, bytes], ...]:
    """Return ``(row_index, raw_line)`` for one source's in-window rows."""

    from menagerie.crawler.leg_generation import source_row_window

    return tuple(
        (start + offset, raw)
        for offset, raw in enumerate(source_row_window(path, start, stop))
    )


def _decoded_pre_window_revisions(
    path: Path, floor_rows: int
) -> dict[str, tuple[int, Mapping[str, Any]]]:
    """Return ``record_revision`` -> (row index, row) strictly below the floor."""

    from menagerie.crawler.leg_generation import source_row_window

    revisions: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, raw in enumerate(source_row_window(path, 0, floor_rows)):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping) and isinstance(row.get("record_revision"), str):
            revisions[str(row["record_revision"])] = (index, row)
    return revisions


def _adoption_family_refusal(
    row: Mapping[str, Any],
    *,
    row_index: int,
    source: str,
    floor_rows: int,
    pre_window: Mapping[str, tuple[int, Mapping[str, Any]]],
    joined_run_ids: frozenset[str],
) -> str | None:
    """Classify one in-window models row against the adoption family.

    The rule is the shipped orphan-citation adoption law read POST-append: a
    lawful adoption re-append is its below-floor parent verbatim modulo the
    enumerated re-citation set
    (:data:`menagerie.crawler.terminal_review.ORPHAN_ADOPTION_ALLOWED_DIFF`
    plus the rotatable identity axes), carries the containment hold shape,
    and is cited by a run durably joined to THIS generation. Nothing here
    widens either adoption rule — a row the pre-append predicates would have
    refused fails the same byte comparison here.

    Returns
    -------
    str | None
        ``None`` when the row is a lawful adoption re-append, else the typed
        refusal reason.
    """

    from menagerie.crawler.terminal_review import (
        ORPHAN_ADOPTION_ALLOWED_DIFF,
        ORPHAN_ROTATABLE_VECTOR_AXES,
        _disallowed_diff_paths,
    )

    prefix = f"{source}[{row_index}]"
    stable_id = row.get("stable_id")
    if not isinstance(stable_id, str) or not stable_id:
        return f"{prefix} is not a model row (no stable_id)"
    status = row.get("status")
    if not isinstance(status, Mapping) or status.get("kind") != "failed":
        return (
            f"{prefix} ({stable_id}) has status kind "
            f"{None if not isinstance(status, Mapping) else status.get('kind')!r}: "
            "a licensing_close leg may append only review-held failure "
            "terminals (the adoption family), never a positive disposition"
        )
    review = status.get("human_review")
    if not isinstance(review, Mapping) or review.get("required") is not True:
        return (
            f"{prefix} ({stable_id}) is not review-held: outside the adoption "
            "family"
        )
    parent_revision = row.get("parent_revision")
    if not isinstance(parent_revision, str) or parent_revision not in pre_window:
        return (
            f"{prefix} ({stable_id}) parent_revision does not resolve strictly "
            f"below the window floor ({floor_rows}): adoption requires sealed "
            "pre-generation history"
        )
    parent_index, parent = pre_window[parent_revision]
    if parent_index >= floor_rows:  # pragma: no cover - resolver is floor-bounded
        return f"{prefix} ({stable_id}) parent sits inside the open window"
    parent_status = parent.get("status")
    parent_kind = (
        parent_status.get("kind") if isinstance(parent_status, Mapping) else None
    )
    if parent_kind != status.get("kind"):
        return (
            f"{prefix} ({stable_id}) changes status kind {parent_kind!r} -> "
            f"{status.get('kind')!r}: a licensing_close leg may never move a "
            "disposition"
        )
    offending = _disallowed_diff_paths(
        row, parent, ORPHAN_ADOPTION_ALLOWED_DIFF | ORPHAN_ROTATABLE_VECTOR_AXES
    )
    if offending:
        shown = ", ".join(offending[:5])
        suffix = "" if len(offending) <= 5 else f" (+{len(offending) - 5} more)"
        return (
            f"{prefix} ({stable_id}) differs from its parent outside the "
            f"adoption re-citation set: {shown}{suffix}"
        )
    provenance = row.get("provenance")
    producer_run = (
        provenance.get("producer_run_id") if isinstance(provenance, Mapping) else None
    )
    if producer_run not in joined_run_ids:
        return (
            f"{prefix} ({stable_id}) cites run {producer_run!r}, which never "
            "joined this leg generation: position is not provenance"
        )
    return None


def _row_generation(row: Mapping[str, Any]) -> Any:
    """Return a paid row's bound generation from its accounting axis.

    Durable reservation rows carry the axis block
    (``paid_session_lifecycle.reserve``); the flat spelling stays accepted
    for pre-axis era rows.
    """

    axis = row.get("axis")
    if isinstance(axis, Mapping) and "generation_id" in axis:
        return axis.get("generation_id")
    return row.get("generation_id")


def _paid_spend_refusal(
    generation_id: str, paid_rows: Iterable[Mapping[str, Any]]
) -> str | None:
    """Refuse when any durable paid-session row binds this generation.

    The named paid-spend arm of the forfeit (Sol contract point 6's
    "paid-session delta"), keyed on the shape the money ledger actually
    emits: the generation nested in the accounting ``axis`` block and the
    ``provider_idempotency_key`` reservation identity, with the flat era
    spellings still accepted.
    """

    reserved = sorted(
        {
            str(
                row.get("provider_idempotency_key")
                or row.get("idempotency_key")
                or row.get("row_sha256")
                or "<reservation>"
            )
            for row in paid_rows
            if _row_generation(row) == generation_id
        }
    )
    if not reserved:
        return None
    return (
        f"licensing_close generation {generation_id} bought "
        f"{len(reserved)} paid session reservation(s): the purpose "
        "forfeits all paid spend"
    )


def _machine_release_reason_revision(reason: Any) -> str | None:
    """Return the revision a derived release-on-record reason names, or None.

    The genuine disclosure text is byte-reconstructible from the revision it
    names (:func:`menagerie.crawler.park_release._void_reason`); anything
    that does not round-trip through the writer's own derivation is a
    hand-authored reason, not the machine disclosure.
    """

    from menagerie.crawler.park_release import _void_reason

    if not isinstance(reason, str):
        return None
    prefix = "released by a landed canonical revision: the model's ledger head is "
    marker = ", which the park row's own release predicate reports"
    if not reason.startswith(prefix):
        return None
    end = reason.find(marker)
    if end <= len(prefix):
        return None
    revision = reason[len(prefix) : end]
    return revision if _void_reason(revision) == reason else None


def _park_void_family(source: str) -> tuple[Any, Any, Any, str, type[Exception]]:
    """Return one park slot's (validate_void, validate_park, released,
    machine-voidable disposition, typed error) replay surface.

    Exactly the per-ledger vocabulary
    :func:`menagerie.crawler.park_release.void_released_parks` writes with:
    the machine lane voids grant-PENDING rows on the grant ledger and
    terminal-refusal rows on the refusal ledger, and nothing else — award and
    selection-exclusion parks have no machine void, ever.
    """

    from menagerie.crawler.grant_parks import (
        GRANT_PENDING_DISPOSITION,
        GrantParkError,
        grant_park_released,
        validate_grant_park_row,
        validate_grant_park_void,
    )
    from menagerie.crawler.refusal_parks import (
        REFUSAL_PARK_DISPOSITION,
        RefusalParkError,
        refusal_park_released,
        validate_refusal_park_row,
        validate_refusal_park_void,
    )

    if source == "park.grant":
        return (
            validate_grant_park_void,
            validate_grant_park_row,
            grant_park_released,
            GRANT_PENDING_DISPOSITION,
            GrantParkError,
        )
    return (
        validate_refusal_park_void,
        validate_refusal_park_row,
        refusal_park_released,
        REFUSAL_PARK_DISPOSITION,
        RefusalParkError,
    )


def _park_void_refusal(
    row: Mapping[str, Any],
    *,
    row_index: int,
    source: str,
    ledger_rows: Mapping[str, Mapping[str, Any]],
    model_heads: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Allow only the machine release-on-record void family inside the window.

    The close path's void-on-record disclosure
    (:func:`menagerie.crawler.park_release.void_released_parks`) appends a
    void ONLY under the fixed machine actor, with the derived disclosure
    reason, targeting an existing park row of the machine-voidable
    disposition whose own release predicate the model's raw ledger head
    satisfies — disposition-invariant by construction. Every one of those
    elements is replayed here from ledger bytes, so an operator void (a
    fresh park decision lifting a live claim) refuses exactly like a new
    park row, and no writer-side guard is trusted.
    """

    from menagerie.crawler.park_release import PARK_RELEASE_VOID_ACTOR

    prefix = f"{source}[{row_index}]"
    if not (isinstance(row.get("void_of"), str) and row["void_of"].strip()):
        return (
            f"{prefix} is not a release-void row: a licensing_close "
            "leg may not mint new park decisions"
        )
    validate_void, validate_park, released, machine_disposition, park_error = (
        _park_void_family(source)
    )
    try:
        void = validate_void(row)
    except park_error as exc:
        return f"{prefix} is not a typed park-void row: {exc}"
    if void["voided_by"] != PARK_RELEASE_VOID_ACTOR:
        return (
            f"{prefix} was voided by {void['voided_by']!r}, not the machine "
            f"release actor {PARK_RELEASE_VOID_ACTOR!r}: an operator void is "
            "a fresh park decision, outside the licensing_close close family"
        )
    return _park_void_target_refusal(
        void,
        prefix=prefix,
        validate_park=validate_park,
        released=released,
        machine_disposition=machine_disposition,
        park_error=park_error,
        ledger_rows=ledger_rows,
        model_heads=model_heads,
    )


def _park_void_target_refusal(
    void: Mapping[str, Any],
    *,
    prefix: str,
    validate_park: Any,
    released: Any,
    machine_disposition: str,
    park_error: type[Exception],
    ledger_rows: Mapping[str, Mapping[str, Any]],
    model_heads: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Replay one machine void's TARGET half: park row, disposition, release.

    The void row itself already validated typed under the machine actor; this
    half re-derives what it claims — an existing machine-voidable park row
    whose own release predicate both the disclosed reason's revision and the
    model's raw ledger head satisfy.
    """

    target = ledger_rows.get(void["void_of"])
    if target is None:
        return f"{prefix} voids {void['void_of']!r}, which names no park row in this ledger"
    try:
        park = validate_park(target)
    except park_error as exc:
        return f"{prefix} voids a row that is not a typed park row: {exc}"
    if park["disposition"] != machine_disposition:
        return (
            f"{prefix} voids a {park['disposition']!r} park: the machine "
            f"release lane voids only {machine_disposition!r} rows, and "
            "lifting any other park is an operator decision the purpose forfeits"
        )
    revision = _machine_release_reason_revision(void["reason"])
    if revision is None or not released(park, {"record_revision": revision}):
        return (
            f"{prefix} does not carry the derived release-on-record reason: "
            "the machine disclosure names the exact consuming revision, and "
            "this reason does not replay from one"
        )
    if not released(park, model_heads.get(str(park.get("stable_id")))):
        return (
            f"{prefix} voids a park whose release condition the ledger head "
            "does not satisfy: the machine lane discloses releases, it never "
            "lifts a live decision"
        )
    return None


def _decoded_ledger_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    """Return every decodable ledger row keyed by its own ``row_sha256``."""

    rows: dict[str, Mapping[str, Any]] = {}
    if not path.is_file():
        return rows
    for raw in path.read_bytes().splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping) and isinstance(row.get("row_sha256"), str):
            rows[str(row["row_sha256"])] = row
    return rows


def licensing_close_forfeit_refusals(
    records_root: Path,
    *,
    generation_id: str,
    baselines: Mapping[str, Any],
    paid_rows: Iterable[Mapping[str, Any]] = (),
) -> tuple[str, ...]:
    """Derive every licensing-close forfeit violation from ledger bytes.

    The close-time half of the six-point contract (points 3, 4, and 6),
    recomputed on every invocation with zero producer input:

    * the replayed WS-1 admitted union must be EMPTY;
    * no paid-session reservation may be bound to this generation;
    * per census slot, the in-window delta must stay inside the enumerated
      families — models layouts: adoption re-appends only; park ledgers:
      release voids only; everything else: zero rows.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    generation_id:
        The open licensing-close generation.
    baselines:
        The generation's leg-open per-source baselines (the window floor).
    paid_rows:
        Durable paid-session ledger rows (reservations carry the generation
        they were admitted under).

    Returns
    -------
    tuple[str, ...]
        Typed refusals; empty when every forfeit holds.
    """

    from menagerie.crawler.leg_generation import (
        CLOSE_SOURCE_NAMES,
        MODELS_SOURCE_NAMES,
        LegGenerationError,
        close_ledger_sources,
        current_model_heads,
        generation_run_ids,
        sealed_selection_plan,
        source_row_count,
    )

    park_sources = ("park.grant", "park.refusal")
    refusals: list[str] = []
    try:
        admitted, _digest = sealed_selection_plan(records_root, generation_id)
    except LegGenerationError as exc:
        refusals.append(str(exc))
        admitted = frozenset()
    if admitted:
        refusals.append(
            f"licensing_close generation {generation_id} sealed a NON-EMPTY "
            f"admitted union ({len(admitted)} member(s): "
            f"{', '.join(sorted(admitted)[:5])}"
            f"{'' if len(admitted) <= 5 else ' ...'}): the purpose forfeits "
            "all admission"
        )
    paid_refusal = _paid_spend_refusal(generation_id, paid_rows)
    if paid_refusal is not None:
        refusals.append(paid_refusal)
    joined_runs = generation_run_ids(records_root, generation_id)
    model_heads: dict[str, Mapping[str, Any]] | None = None
    sources = close_ledger_sources(records_root)
    for name in CLOSE_SOURCE_NAMES:
        baseline = dict(baselines.get(name) or {})
        floor_rows = int(baseline.get("rows") or 0)
        path = sources[name]
        observed_rows = source_row_count(path)
        if observed_rows < floor_rows:
            refusals.append(
                f"{name} shrank below its window floor during a "
                "licensing_close leg"
            )
            continue
        if observed_rows == floor_rows:
            continue
        if name in MODELS_SOURCE_NAMES:
            pre_window = _decoded_pre_window_revisions(path, floor_rows)
            for row_index, raw in _window_rows(path, floor_rows, observed_rows):
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    refusals.append(
                        f"{name}[{row_index}] is undecodable inside a "
                        "licensing_close window"
                    )
                    continue
                if not isinstance(row, Mapping):
                    refusals.append(
                        f"{name}[{row_index}] is not an object inside a "
                        "licensing_close window"
                    )
                    continue
                refusal = _adoption_family_refusal(
                    row,
                    row_index=row_index,
                    source=name,
                    floor_rows=floor_rows,
                    pre_window=pre_window,
                    joined_run_ids=joined_runs,
                )
                if refusal is not None:
                    refusals.append(refusal)
        elif name in park_sources:
            if model_heads is None:
                # The replay needs the raw heads the release predicates key
                # on; an underivable models census fails CLOSED (no head can
                # vouch for any in-window void).
                from menagerie.crawler.recordio import LedgerError

                try:
                    model_heads = current_model_heads(records_root, require_ledger=False)
                except (LegGenerationError, LedgerError) as exc:
                    refusals.append(
                        "park release replay has no derivable model heads: " f"{exc}"
                    )
                    model_heads = {}
            ledger_rows = _decoded_ledger_rows(path)
            for row_index, raw in _window_rows(path, floor_rows, observed_rows):
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    refusals.append(
                        f"{name}[{row_index}] is undecodable inside a "
                        "licensing_close window"
                    )
                    continue
                if not isinstance(row, Mapping):
                    refusals.append(
                        f"{name}[{row_index}] is not an object inside a "
                        "licensing_close window"
                    )
                    continue
                refusal = _park_void_refusal(
                    row,
                    row_index=row_index,
                    source=name,
                    ledger_rows=ledger_rows,
                    model_heads=model_heads,
                )
                if refusal is not None:
                    refusals.append(refusal)
        else:
            refusals.append(
                f"{name} moved during a licensing_close leg "
                f"({floor_rows} -> {observed_rows} rows): outside every "
                "enumerated family"
            )
    return tuple(refusals)
