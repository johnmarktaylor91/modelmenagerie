"""Recheck census (RC-2): the ONE derived view a rotation class feeds.

Currency and class are separate facts. Since RC-2 the dependency-currency
projection replays every settled head at its recorded policy basis
unconditionally -- a code or prompt change never stales a recorded proof and
never holds an award. What a rotation's CLASS decides is this census, whose
vocabulary is the ruling's and nothing more (RULING D.2.3, B.4):

* for every in-scope RAW ledger head and every recorded-basis policy axis
  whose recorded value differs from the LIVE authority-context value, the
  move is classified against ``records/control/rotations.jsonl``;
* ``refinement`` owes nothing;
* ``soundness`` -- or NO connecting row at all, the UNMEASURED move, which
  takes the conservative soundness class -- makes the head ``needs_recheck``:
  excluded from release views, disclosed in ``status``, admitted to work
  under the named ``RECHECK`` selection mode. Every head kind: a failed,
  skipped, or deferred terminal recorded before the edge re-enters exactly
  like an award ("the 35 soundness-stale terminals and the 7 held awards
  become needs_recheck rows"). The award stays an award until the re-check
  says otherwise; a failed re-check appends a superseding terminal exactly
  as today (AW-1 award custody unchanged: the census prints the
  award-naming grant lever).

Replay-derived on every read from the raw ledger heads, the rotations ledger,
and the live authority context; nothing persisted steers it, and no ledger
row ever supplies the live identity.

The predicate replay probe (:func:`predicate_replay_probe`) is the code-axis
MEASUREMENT: every head replays at its recorded basis under the RUNNING
reducer, which re-derives the terminal disposition from recorded evidence;
the probe compares the derived and recorded dispositions FIELD BY FIELD and
also runs the full replay (evidence re-admission, artifact and family
cross-checks). A flip is any disposition difference, or a full-replay refusal
whose reason is not a changed input of the record. Zero flips over a
non-empty head set is ``refinement``; any flip is ``soundness``; zero heads
measures nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.authority import AuthorityContext, DependencyState
from menagerie.crawler.models import JsonObject, LedgerPaths
from menagerie.crawler.reducer import (
    RECORDED_BASIS_AXES,
    _select_current,
    default_ledger_paths,
    project_dependency_current,
    scan_jsonl,
)
from menagerie.crawler.rotations import (
    CODE_AXES,
    load_rotations,
    move_class,
    soundness_rotation_ids,
)
from menagerie.crawler.status import stale_axis

__all__ = [
    "CHANGED_INPUT_STALE_CLASSES",
    "DISPOSITION_FIELDS",
    "PREDICATE_REPLAY_PROBE_SCHEMA_VERSION",
    "RECHECK_CENSUS_SCHEMA_VERSION",
    "REMEDY_BY_AXIS",
    "RecheckRow",
    "award_release_lever",
    "predicate_replay_probe",
    "recheck_authorities",
    "recheck_census",
    "recheck_census_from_records",
    "recheck_review_held",
    "recheck_targets",
]

RECHECK_CENSUS_SCHEMA_VERSION = "menagerie.crawler.recheck-census.v2"
PREDICATE_REPLAY_PROBE_SCHEMA_VERSION = "menagerie.crawler.predicate-replay-probe.v2"

#: The remedy a soundness move on each axis names (D.2.3): checker axes
#: re-run the checker under the live epoch; author axes re-author; code axes
#: re-derive (after the probe has measured the move when it has not).
REMEDY_BY_AXIS: Mapping[str, str] = {
    "checker_prompt_identity": "checker-rerun",
    "author_prompt_identity": "re-author",
    "author_result_schema_identity": "re-author",
    "author_dispatcher_identity": "re-author",
    "terminal_rule_identity": "re-derive",
    "publication_policy_identity": "re-derive",
    "runner_identity": "re-execute",
    "award_closure_identity": "re-derive",
}

#: Stale classes that name a CHANGED INPUT of the record (its intake row, its
#: source bytes, its recipe, its evidence rows, its environment generation) --
#: real staleness the projection keeps, and never a code flip. Everything
#: else a replay refuses under new code is attributed to the code (including
#: ``unclassified``: an unexplained refusal is a flip until explained).
CHANGED_INPUT_STALE_CLASSES = frozenset(
    {
        "intake",
        "intake_snapshot_id",
        "intake_snapshot_sha256",
        "intake_item_sha256",
        "source_manifest_identity",
        "recipe_revision",
        "environment_generation",
        "accepted_attempt_ids",
        "artifact_transaction_id",
        "artifact_claim_ids",
        "representative_revision",
        "family-authority",
        "legacy-untrusted",
    }
)
_CHANGED_INPUT_REASON_MARKERS = (
    "referenced attempt is missing",
    "referenced gate is missing",
    "attested nowhere",
    # A head outside the replaying intake's membership (``--all-heads`` over
    # a multi-intake ledger): intake membership is an input of the record,
    # and the projection refuses it before any predicate runs.
    "is outside intake",
)
#: The reducer's derived-vector contradiction: same evidence bytes, different
#: derivation. Whatever axis it names, it is the code disagreeing with the
#: recorded proof -- a flip, never a changed input.
_DERIVATION_CONTRADICTION_MARKER = "contradicts reducer derivation"

#: The recorded disposition fields the probe compares exactly against the
#: running code's derivation: status kind and code plus the failed-status
#: proof fields. Evidence identity (which attempts and gates the proof names)
#: is compared by the full replay's derived-vector check, whose contradiction
#: refusal is ALWAYS a flip below (the evidence bytes did not change; the
#: derivation did).
DISPOSITION_FIELDS: tuple[str, ...] = (
    "kind",
    "code",
    "stage",
    "reason_code",
    "root_cause_fingerprint",
)


@dataclass(frozen=True)
class RecheckRow:
    """One (member, axis) soundness move the census discloses."""

    stable_id: str
    axis: str
    recorded: str
    live: str
    soundness_basis: str
    rotation_ids: tuple[str, ...]
    head_kind: str
    status_code: str
    record_revision: str
    remedy: str
    #: The head is held for explicit human review: its re-check enters work
    #: through the existing review grant lane (a held terminal's supersession
    #: requires a new explicit grant; admitting it directly recreates the
    #: rung-11 leg-4 unrecordable-terminal crash chain). Disclosure only.
    review_held: bool = False

    def to_json(self) -> JsonObject:
        """Return the census row."""

        return {
            "stable_id": self.stable_id,
            "axis": self.axis,
            "recorded": self.recorded,
            "live": self.live,
            "class": "soundness",
            "soundness_basis": self.soundness_basis,
            "rotation_ids": list(self.rotation_ids),
            "head_kind": self.head_kind,
            "status_code": self.status_code,
            "record_revision": self.record_revision,
            "remedy": self.remedy,
            "review_held": self.review_held,
        }


def _review_held(head: Mapping[str, Any]) -> bool:
    status = head.get("status")
    if not isinstance(status, Mapping):
        return False
    review = status.get("human_review")
    return isinstance(review, Mapping) and bool(review.get("required"))


def _head_kind(head: Mapping[str, Any]) -> tuple[str, str]:
    status = head.get("status")
    if not isinstance(status, Mapping):
        return ("", "")
    return (str(status.get("kind", "")), str(status.get("code", "")))


def _moved_axes(
    head: Mapping[str, Any], context: AuthorityContext
) -> list[tuple[str, str, str]]:
    """Return ``(axis, recorded, live)`` for every policy axis whose value moved.

    ``live`` is the authority-context field paired with the axis in
    :data:`RECORDED_BASIS_AXES` -- never a ledger row.
    """

    vector = head.get("dependency_vector")
    if not isinstance(vector, Mapping):
        return []
    moves: list[tuple[str, str, str]] = []
    for axis, field in RECORDED_BASIS_AXES:
        recorded = vector.get(axis)
        if (
            not isinstance(recorded, str)
            or not recorded
            or recorded == DependencyState.NOT_APPLICABLE.value
        ):
            continue
        live = str(getattr(context, field))
        if recorded != live:
            moves.append((axis, recorded, live))
    return moves


def _head_recheck_rows(
    stable_id: str,
    head: Mapping[str, Any],
    rows: tuple[Mapping[str, Any], ...],
    context: AuthorityContext,
) -> list[RecheckRow]:
    """Return one census row per soundness (or unmeasured) axis move of one head."""

    kind, code = _head_kind(head)
    out: list[RecheckRow] = []
    for axis, recorded, live in _moved_axes(head, context):
        resolved = move_class(rows, axis=axis, recorded=recorded, live=live)
        if resolved == "refinement":
            continue
        measured = resolved == "soundness"
        out.append(
            RecheckRow(
                stable_id=stable_id,
                axis=axis,
                recorded=recorded,
                live=live,
                soundness_basis="measured" if measured else "unmeasured",
                rotation_ids=(
                    soundness_rotation_ids(rows, axis=axis, recorded=recorded) if measured else ()
                ),
                head_kind=kind,
                status_code=code,
                record_revision=str(head.get("record_revision", "")),
                remedy=REMEDY_BY_AXIS[axis],
                review_held=_review_held(head),
            )
        )
    return out


def recheck_census(
    raw_heads: Mapping[str, Mapping[str, Any]],
    rotations: Iterable[Mapping[str, Any]],
    context: AuthorityContext,
    *,
    scope: Optional[Iterable[str]] = None,
) -> JsonObject:
    """Derive the recheck census over raw ledger heads.

    Parameters
    ----------
    raw_heads:
        Stable-ID keyed RAW model-ledger heads (the settlement authority).
    rotations:
        Validated rows of the measured-rotations ledger.
    context:
        Live authority context (its policy identities are the ``live`` side).
    scope:
        Optional in-scope membership (campaign intake); ``None`` is ledger-wide.

    Returns
    -------
    dict[str, Any]
        ``needs_recheck`` rows (one per member per soundness axis), the
        distinct member count, the member count by head kind, the unmeasured
        count, and the per-axis member lists.
    """

    rows = tuple(rotations)
    in_scope = None if scope is None else frozenset(str(item) for item in scope)
    needs: list[RecheckRow] = []
    for stable_id in sorted(raw_heads):
        if in_scope is None or stable_id in in_scope:
            needs.extend(_head_recheck_rows(str(stable_id), raw_heads[stable_id], rows, context))
    by_axis: dict[str, set[str]] = {}
    for row in needs:
        by_axis.setdefault(row.axis, set()).add(row.stable_id)
    return {
        "schema_version": RECHECK_CENSUS_SCHEMA_VERSION,
        "needs_recheck": [row.to_json() for row in needs],
        "counts": _census_counts(needs, raw_heads),
        "by_axis": {axis: sorted(ids) for axis, ids in sorted(by_axis.items())},
    }


def _census_counts(
    needs: Sequence[RecheckRow], raw_heads: Mapping[str, Mapping[str, Any]]
) -> JsonObject:
    """Distinct-member counts: total, by head kind, unmeasured, review-held."""

    members = {row.stable_id for row in needs}
    kinds: dict[str, int] = {}
    for stable_id in members:
        kind, _ = _head_kind(raw_heads[stable_id])
        kinds[kind] = kinds.get(kind, 0) + 1
    return {
        "needs_recheck": len(members),
        "by_head_kind": {kind: kinds[kind] for kind in sorted(kinds)},
        "unmeasured": len({row.stable_id for row in needs if row.soundness_basis == "unmeasured"}),
        "review_held": len({row.stable_id for row in needs if row.review_held}),
    }


def recheck_targets(census: Mapping[str, Any]) -> frozenset[str]:
    """Return the members the RECHECK selection mode admits (needs_recheck)."""

    return frozenset(str(row["stable_id"]) for row in census.get("needs_recheck", ()))


def recheck_review_held(census: Mapping[str, Any]) -> frozenset[str]:
    """Return the needs_recheck members held for human review (grant lane only)."""

    return frozenset(
        str(row["stable_id"]) for row in census.get("needs_recheck", ()) if row.get("review_held")
    )


def recheck_authorities(census: Mapping[str, Any]) -> dict[str, str]:
    """Return the per-member admission authority string for RECHECK plans."""

    authorities: dict[str, str] = {}
    for row in census.get("needs_recheck", ()):
        stable_id = str(row["stable_id"])
        ids = [str(value) for value in row.get("rotation_ids", ())]
        label = ",".join(ids) if ids else f"unmeasured:{row.get('axis')}"
        authorities[stable_id] = (
            label if stable_id not in authorities else f"{authorities[stable_id]};{label}"
        )
    return authorities


def award_release_lever(row: Mapping[str, Any]) -> str:
    """Spell the AW-1 award-naming grant that reopens one needs_recheck award."""

    authority = ",".join(row.get("rotation_ids", ())) or f"unmeasured:{row.get('axis')}"
    return (
        f"python -m menagerie.crawler.tools.requeue {row['stable_id']} --grant 1 "
        f"--stage <closed failure stage> --blocked-award-revision {row['record_revision']} "
        f"--reason 'recheck: {row.get('remedy')} after soundness rotation on "
        f"{row.get('axis')} ({authority})'"
    )


def recheck_census_from_records(
    records_root: Path,
    context: AuthorityContext,
    *,
    scope: Optional[Iterable[str]] = None,
    ledgers: Optional[LedgerPaths] = None,
) -> JsonObject:
    """Derive the census from one records root (raw heads + rotations ledger)."""

    paths = ledgers or default_ledger_paths(records_root)
    heads = _select_current(scan_jsonl(paths.models))
    return recheck_census(heads, load_rotations(Path(records_root)), context, scope=scope)


def _recorded_disposition(head: Mapping[str, Any]) -> JsonObject:
    status = head.get("status") if isinstance(head.get("status"), Mapping) else {}
    return {field: status.get(field) for field in DISPOSITION_FIELDS}


_FAILED_ONLY_FIELDS = frozenset({"stage", "reason_code", "root_cause_fingerprint"})


def _disposition_diff(recorded: Mapping[str, Any], derived: Mapping[str, Any]) -> JsonObject:
    """Field-by-field diff, mirroring the reducer's own terminal-proof check.

    The three failure fields are decisive only for a ``failed:`` code (the
    proof carries the ``not-applicable`` sentinel where a non-failed status
    stores ``None``); they are compared when EITHER side is a failure, so a
    failure the new code derives as ``runs`` -- or the reverse -- still diffs
    on ``kind``/``code`` and, when both are failures, on every proof field.
    """

    failed = any(str(side.get("code", "")).startswith("failed:") for side in (recorded, derived))
    return {
        field: {"recorded": recorded.get(field), "derived": derived.get(field)}
        for field in DISPOSITION_FIELDS
        if (failed or field not in _FAILED_ONLY_FIELDS)
        and recorded.get(field) != derived.get(field)
    }


def _is_changed_input(axis: str, reason: str) -> bool:
    if _DERIVATION_CONTRADICTION_MARKER in reason:
        return False
    return axis in CHANGED_INPUT_STALE_CLASSES or any(
        marker in reason for marker in _CHANGED_INPUT_REASON_MARKERS
    )


def _classify_replayed_head(
    stable_id: str,
    head: Mapping[str, Any],
    derived: Optional[Mapping[str, Any]],
    reason: Optional[str],
) -> tuple[str, JsonObject]:
    """Classify one replayed head: ``current``, ``flip``, or ``changed-input``.

    The exact disposition diff decides first; a full-replay refusal decides
    next (a derivation contradiction is always a flip; a changed-input class
    is segregated; anything else is a flip).
    """

    kind, code = _head_kind(head)
    entry: JsonObject = {"stable_id": stable_id, "head_kind": kind, "status_code": code}
    if isinstance(derived, Mapping) and "disposition" in derived:
        diff = _disposition_diff(_recorded_disposition(head), derived["disposition"])
        if diff:
            return "flip", {**entry, "disposition_diff": diff}
    if reason is None:
        return "current", entry
    axis = stale_axis(reason)
    entry.update({"reason": str(reason), "axis": axis})
    return ("changed-input" if _is_changed_input(axis, str(reason)) else "flip"), entry


def predicate_replay_probe(
    ledgers: LedgerPaths,
    context: AuthorityContext,
    *,
    scope: Optional[Iterable[str]] = None,
) -> JsonObject:
    """Re-derive every head under the running code and report the flips exactly.

    The probe IS the production projection plus an explicit comparison: every
    settled head replays at its recorded policy basis under the live reducer,
    validator, and gate code. For each head the running code's derived
    disposition (:data:`DISPOSITION_FIELDS`) is compared with the recorded
    one field by field; any difference is a FLIP -- including a prior
    non-acceptance the new code would now accept, or one failure disposition
    the new code derives as another. A head the full replay refuses for a
    CHANGED-INPUT reason (its intake row, source bytes, recipe, evidence rows,
    or environment generation moved) is real staleness and not attributed to
    the code; every other refusal is a flip too. One flip makes the code move
    ``soundness``; zero flips over a non-empty head set is ``refinement``;
    zero heads measures nothing (``class = None``).

    Returns
    -------
    dict[str, Any]
        Heads replayed, awards among them, flips (member, kind, code, the
        exact disposition diff or the refusal reason and axis), changed-input
        stale heads (segregated), the code axes the probe exercises, and the
        measured class.
    """

    in_scope = None if scope is None else frozenset(str(item) for item in scope)
    heads = _select_current(scan_jsonl(ledgers.models))
    derived_by_id: dict[str, JsonObject] = {}
    projection = project_dependency_current(
        ledgers, context=context, disposition_sink=derived_by_id
    )
    flips: list[JsonObject] = []
    changed_input: list[JsonObject] = []
    replayed = 0
    awards = 0
    for stable_id in sorted(heads):
        if in_scope is not None and stable_id not in in_scope:
            continue
        replayed += 1
        head = heads[stable_id]
        awards += _head_kind(head)[0] == "runs"
        verdict, entry = _classify_replayed_head(
            str(stable_id),
            head,
            derived_by_id.get(stable_id),
            projection.stale_reasons.get(stable_id),
        )
        if verdict == "flip":
            flips.append(entry)
        elif verdict == "changed-input":
            changed_input.append(entry)
    return {
        "schema_version": PREDICATE_REPLAY_PROBE_SCHEMA_VERSION,
        "heads": replayed,
        "awards": awards,
        "current": replayed - len(flips) - len(changed_input),
        "flips": flips,
        "changed_input_stale": changed_input,
        "measures_axes": sorted(CODE_AXES),
        "class": None if replayed == 0 else ("soundness" if flips else "refinement"),
    }
