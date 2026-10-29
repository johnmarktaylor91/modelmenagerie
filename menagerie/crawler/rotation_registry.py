"""Registered authority rotations — the declared refinement/soundness ledger (AW-2).

The step-100 award-eating loop was crash -> hotfix -> closure rotation -> every
settled head stale -> mass re-drive: any commit touching one of the ~11 byte-
hashed authority files re-litigated the whole campaign, and the eaten awards'
rotating axis (``publication_policy_identity``, 51cdae77) was a REFINEMENT --
new work should run under it, but nothing about the already-executed awards
changed. Rotation class cannot be inferred from a byte hash, so it is DECLARED
here: one operator-durable row per rotating commit, naming the axis, the exact
from/to identities, and whether the rotation is a ``refinement`` (recorded
proofs stand at their recorded basis) or a ``soundness`` fix (recorded proofs
may be wrong; affected awards enter a release hold for targeted re-verification
and are never silently honored). Unregistered rotations read as soundness --
fail closed.

The registry is a records ledger, deliberately OUTSIDE every hashed authority
surface: registration is post-commit by construction (the ``to`` identity is
only computable after the commit whose bytes it hashes), so the sequence is
commit -> compute identities -> append rows -> launch.

SAME-EDGE CLASS CONFLICTS (P5 fix wave, 2026-08-31; fable-F3 / sol-04): two
rows registering the SAME ``(axis, from_identity, to_identity)`` edge under
DIFFERENT rotation classes are a contradiction the v1 registry silently
resolved refinement-first, letting a later parallel ``refinement`` row
defeat a standing ``soundness`` registration with no review gate (the
2026-08-29/30 checker_prompt pair that dissolved 46 designed review holds).
The law is now three-part and fail-closed:

* APPEND refuses a row that would put a second class on an existing edge
  unless the candidate is an explicit RESOLUTION row: it must carry
  ``supersedes`` naming a prior same-edge registration_id AND
  ``supersession_evidence`` binding the re-verification evidence or ruling
  that authorizes the reclassification.
* RESOLUTION (:func:`registered_move_class`) treats an edge whose LIVE
  (non-superseded) rows still disagree as SOUNDNESS -- the conflict reads
  fail-closed pending an explicit resolution row, so a historical
  contradictory append is inert, never a silent downgrade. Rows are never
  deleted; a superseded row stays in the ledger as a disclosed dead edge.
* DISCLOSURE: :func:`rotation_edge_census` derives every edge's registered
  classes, live classes, superseded rows, and conflict state from the
  ledger alone, so a reclassified or contested edge is visible to every
  consumer and to ``register_authority_rotation --list``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import read_with_tail_recovery

__all__ = [
    "ROTATION_AXES",
    "ROTATION_CLASSES",
    "ROTATION_REGISTRATION_SCHEMA_VERSION",
    "RotationRegistrationError",
    "append_rotation_registration",
    "build_rotation_registration",
    "load_rotation_registrations",
    "registered_move_class",
    "rotation_edge_census",
    "rotation_registry_path",
]

ROTATION_REGISTRATION_SCHEMA_VERSION = "menagerie.crawler.authority-rotation-registration.v1"

#: Closed dependency-vector axes a rotation registration may govern: exactly
#: the shipped-policy identity axes the recorded-basis carve-out pins (the
#: dependency-vector spelling, not the AuthorityContext field). Environment
#: generations are deliberately NOT here -- they resolve against the
#: ledger-attested generation set, never against declarations.
ROTATION_AXES = frozenset(
    {
        "author_result_schema_identity",
        "author_dispatcher_identity",
        "author_prompt_identity",
        "checker_prompt_identity",
        "terminal_rule_identity",
        "publication_policy_identity",
        "runner_identity",
        "award_closure_identity",
        # AL-22: the author STEERING-SURFACE identity -- a content hash over
        # exactly the prompt files that steer one author attempt (executor
        # stage briefs + the campaign fragment + the pinned top-level
        # prompt), stamped per attempt. Registered rotations of this axis
        # use the disclosed-non-staling class: the surface's history is a
        # first-class disclosed identity, but it is NOT a staleness axis --
        # it appears in no dependency-vector currency check and no
        # record-time pinned set, so a prompt edit rotates it without
        # staling any carried head.
        "author_steering_surface_identity",
        # W32: the TOOLCHAIN identity -- a content hash over the reviewed CLI
        # version pins (Claude author harness, codex checker) and the
        # tier->author-model table (F-O23 rider). A pin bump or a model-tier
        # change moves this identity, so it is a registerable rotation with
        # storm-envelope re-attestation, never a silent drift. Computed by
        # menagerie.crawler.toolchain_pins.toolchain_identity().
        "toolchain_identity",
    }
)

#: Closed rotation classes. A ``refinement`` changes what NEW work would do
#: and leaves recorded proofs valid at their recorded basis; a ``soundness``
#: rotation invalidates a class of prior acceptances and forces the release
#: hold on every record-anchored acceptance it touches; a
#: ``disclosed-non-staling`` rotation (AL-22) records that an identity moved
#: -- fully disclosed, auditable history -- while by CONSTRUCTION staling
#: nothing: recorded proofs stay valid exactly as under a refinement.
ROTATION_CLASSES = frozenset({"refinement", "soundness", "disclosed-non-staling"})

ROTATION_REGISTRY_RELATIVE_PATH = Path("intake") / "registered-authority-rotations.jsonl"

_REQUIRED_FIELDS = (
    "schema_version",
    "registration_id",
    "axis",
    "from_identity",
    "to_identity",
    "rotation_class",
    "rationale",
    "registered_by",
    "created_at",
)


class RotationRegistrationError(ValueError):
    """A rotation registration row is malformed, unknown, or conflicting."""


def rotation_registry_path(records_root: Path) -> Path:
    """Return the canonical rotation-registry ledger below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only authority-rotation registration JSONL path.
    """

    return records_root / ROTATION_REGISTRY_RELATIVE_PATH


def build_rotation_registration(
    *,
    axis: str,
    from_identity: str,
    to_identity: str,
    rotation_class: str,
    rationale: str,
    registered_by: str,
    created_at: str,
    supersedes: Optional[str] = None,
    supersession_evidence: Optional[str] = None,
) -> JsonObject:
    """Assemble one validated rotation-registration row.

    Parameters
    ----------
    axis:
        Rotated dependency-vector axis (member of :data:`ROTATION_AXES`).
    from_identity, to_identity:
        Exact recorded and post-commit identity values.
    rotation_class:
        ``refinement`` or ``soundness``.
    rationale, registered_by, created_at:
        Operator provenance.
    supersedes, supersession_evidence:
        Present together or not at all. A RESOLUTION row reclassifying an
        already-registered edge must name the superseded prior
        ``registration_id`` and bind the re-verification evidence or ruling
        that authorizes the reclassification -- the only lawful path to a
        second class on one edge.

    Returns
    -------
    dict[str, Any]
        Content-addressed registration row.

    Raises
    ------
    RotationRegistrationError
        If any field is outside its closed vocabulary or empty.
    """

    row: JsonObject = {
        "schema_version": ROTATION_REGISTRATION_SCHEMA_VERSION,
        "axis": axis,
        "from_identity": from_identity,
        "to_identity": to_identity,
        "rotation_class": rotation_class,
        "rationale": rationale,
        "registered_by": registered_by,
        "created_at": created_at,
    }
    if supersedes is not None or supersession_evidence is not None:
        row["supersedes"] = supersedes
        row["supersession_evidence"] = supersession_evidence
    row["registration_id"] = "rotation-" + stable_hash(row)[7:31]
    return _validate_registration(row, source="build")


def _validate_registration(row: Mapping[str, Any], *, source: str) -> JsonObject:
    """Validate one registration row fail-closed and return its dict copy."""

    missing = [field for field in _REQUIRED_FIELDS if field not in row]
    if missing:
        raise RotationRegistrationError(
            f"rotation registration ({source}) is missing {', '.join(sorted(missing))}"
        )
    if row["schema_version"] != ROTATION_REGISTRATION_SCHEMA_VERSION:
        raise RotationRegistrationError(
            f"rotation registration ({source}) declares unknown schema {row['schema_version']!r}"
        )
    if row["axis"] not in ROTATION_AXES:
        raise RotationRegistrationError(
            f"rotation registration ({source}) names an axis outside the closed "
            f"set: {row['axis']!r}"
        )
    if row["rotation_class"] not in ROTATION_CLASSES:
        raise RotationRegistrationError(
            f"rotation registration ({source}) declares an unknown rotation class: "
            f"{row['rotation_class']!r}"
        )
    for field in ("from_identity", "to_identity", "rationale", "registered_by", "created_at"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise RotationRegistrationError(
                f"rotation registration ({source}) has an empty {field}"
            )
    if row["from_identity"] == row["to_identity"]:
        raise RotationRegistrationError(
            f"rotation registration ({source}) does not rotate: from and to are equal"
        )
    has_supersedes = "supersedes" in row or "supersession_evidence" in row
    if has_supersedes:
        for field in ("supersedes", "supersession_evidence"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise RotationRegistrationError(
                    f"rotation registration ({source}) resolution rows require both "
                    f"supersedes and supersession_evidence as non-empty strings; "
                    f"{field} is {value!r}"
                )
        if row["supersedes"] == row.get("registration_id"):
            raise RotationRegistrationError(
                f"rotation registration ({source}) cannot supersede itself"
            )
    return dict(row)


def append_rotation_registration(path: Path, row: Mapping[str, Any]) -> bool:
    """Append one registration row idempotently to the append-only registry.

    Parameters
    ----------
    path:
        Registry ledger path (created with parents when absent).
    row:
        Validated registration row.

    Returns
    -------
    bool
        Whether new bytes were appended (``False`` for an identical replay).

    Raises
    ------
    RotationRegistrationError
        If an existing row holds the same ``registration_id`` with different
        bytes, the candidate row is malformed, or the candidate would put a
        SECOND rotation class on an already-registered edge without being an
        explicit resolution row (``supersedes`` naming a prior same-edge
        registration plus ``supersession_evidence``) -- the silent parallel
        reclassification channel this door closes (fable-F3 / sol-04).
    """

    candidate = _validate_registration(row, source="append")
    existing = load_rotation_registrations(path)
    same_edge: list[JsonObject] = []
    for prior in existing:
        if prior["registration_id"] == candidate["registration_id"]:
            if prior == candidate:
                return False
            raise RotationRegistrationError(
                f"rotation registry holds conflicting bytes for {candidate['registration_id']!r}"
            )
        if (
            prior["axis"] == candidate["axis"]
            and prior["from_identity"] == candidate["from_identity"]
            and prior["to_identity"] == candidate["to_identity"]
        ):
            same_edge.append(prior)
    supersedes = candidate.get("supersedes")
    if supersedes is not None:
        if not any(prior["registration_id"] == supersedes for prior in same_edge):
            raise RotationRegistrationError(
                f"rotation registration (append) supersedes {supersedes!r}, which is "
                "not a registered row on the same (axis, from, to) edge; a resolution "
                "row may only reclassify the exact edge it names"
            )
    elif any(prior["rotation_class"] != candidate["rotation_class"] for prior in same_edge):
        conflicting = sorted(
            {
                str(prior["rotation_class"])
                for prior in same_edge
                if prior["rotation_class"] != candidate["rotation_class"]
            }
        )
        raise RotationRegistrationError(
            f"rotation registration (append) would put class "
            f"{candidate['rotation_class']!r} on an edge already registered as "
            f"{', '.join(conflicting)}; a reclassification is a resolution row -- it "
            "must carry supersedes naming the prior registration_id and "
            "supersession_evidence binding the re-verification evidence or ruling "
            "(until one lands, consumers read the conflicted edge as soundness)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(candidate, sort_keys=True) + "\n")
    return True


def load_rotation_registrations(path: Path) -> tuple[JsonObject, ...]:
    """Load and validate every registration row; a missing file is empty.

    Parameters
    ----------
    path:
        Registry ledger path.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Validated rows in ledger order.

    Raises
    ------
    RotationRegistrationError
        If any persisted row fails validation -- an unreadable registry must
        never silently degrade to "nothing registered" (that would flip every
        registered refinement back to fail-closed soundness without a trace).
    """

    if not path.is_file():
        return ()
    data = read_with_tail_recovery(
        path, prefix_check=lambda retained: _decode_registry_lines(retained.read_bytes())
    )
    return tuple(_decode_registry_lines(data))


def _decode_registry_lines(data: bytes) -> list[JsonObject]:
    """Strictly decode and validate complete registry bytes."""

    rows: list[JsonObject] = []
    for index, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RotationRegistrationError(
                f"rotation registry line {index} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise RotationRegistrationError(f"rotation registry line {index} is not an object")
        rows.append(_validate_registration(payload, source=f"line {index}"))
    return rows


def _resolved_edge_classes(
    registrations: Iterable[Mapping[str, Any]],
    *,
    axis: Optional[str] = None,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Resolve every registered edge to its governing class, fail-closed.

    Liveness: a row named by ANY row's ``supersedes`` is dead (a disclosed
    dead edge -- never deleted, never consulted for classification). An edge
    whose LIVE rows still carry more than one class is CONFLICTED and
    resolves to ``soundness``: the contradiction reads fail-closed pending an
    explicit resolution row, so a contradictory historical append (the
    2026-08-30 checker_prompt corrective row) is inert rather than a silent
    downgrade of the standing soundness registration.

    Returns
    -------
    dict[tuple[str, str, str], dict[str, Any]]
        ``(axis, from, to) -> {"resolved_class", "conflicted",
        "live_classes", "registered_classes", "registration_ids",
        "superseded_registration_ids"}``.
    """

    rows = [
        dict(row)
        for row in registrations
        if axis is None or row.get("axis") == axis
    ]

    def _edge(row: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(row.get("axis")),
            str(row.get("from_identity")),
            str(row.get("to_identity")),
        )

    # A supersession is honored ONLY within its own edge: the append door
    # enforces that, and a hand-written cross-edge supersedes must never kill
    # a soundness row somewhere else (that would be an upgrade channel).
    edge_by_id = {str(row.get("registration_id")): _edge(row) for row in rows}
    superseded: set[str] = set()
    for row in rows:
        named = row.get("supersedes")
        if (
            isinstance(named, str)
            and named
            and edge_by_id.get(named) == _edge(row)
        ):
            superseded.add(named)
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        entry = edges.setdefault(
            _edge(row),
            {
                "registered_classes": set(),
                "live_classes": set(),
                "registration_ids": [],
                "superseded_registration_ids": [],
            },
        )
        registration_id = str(row.get("registration_id"))
        rotation_class = str(row.get("rotation_class"))
        entry["registration_ids"].append(registration_id)
        entry["registered_classes"].add(rotation_class)
        if registration_id in superseded:
            entry["superseded_registration_ids"].append(registration_id)
        else:
            entry["live_classes"].add(rotation_class)
    for entry in edges.values():
        live_classes = entry["live_classes"] or entry["registered_classes"]
        entry["conflicted"] = len(live_classes) != 1
        entry["resolved_class"] = (
            "soundness" if entry["conflicted"] else next(iter(live_classes))
        )
    return edges


def rotation_edge_census(
    registrations: Iterable[Mapping[str, Any]],
) -> tuple[JsonObject, ...]:
    """Disclose every registered edge's class resolution, ledger-derived.

    The disclosure half of the same-edge conflict law: every edge carrying a
    superseded row or a live class conflict is visible here (and through
    ``register_authority_rotation --list``), so a reclassification can never
    be silent. Consumers of :func:`registered_move_class` follow exactly the
    ``resolved_class`` this census reports.

    Returns
    -------
    tuple[dict[str, Any], ...]
        One row per registered edge, sorted by (axis, from, to).
    """

    edges = _resolved_edge_classes(registrations)
    census: list[JsonObject] = []
    for (edge_axis, source, target) in sorted(edges):
        entry = edges[(edge_axis, source, target)]
        census.append(
            {
                "axis": edge_axis,
                "from_identity": source,
                "to_identity": target,
                "registered_classes": sorted(entry["registered_classes"]),
                "live_classes": sorted(entry["live_classes"]),
                "resolved_class": entry["resolved_class"],
                "conflicted": entry["conflicted"],
                "registration_ids": list(entry["registration_ids"]),
                "superseded_registration_ids": list(
                    entry["superseded_registration_ids"]
                ),
            }
        )
    return tuple(census)


def registered_move_class(
    registrations: Iterable[Mapping[str, Any]],
    *,
    axis: str,
    recorded: str,
    live: str,
) -> Optional[str]:
    """Classify one recorded->live identity move against the registry.

    A move is a ``refinement`` exactly when a chain of registered refinement
    rotations connects the recorded identity to the live one (rotations
    compose across successive commits). Any connection that needs a
    soundness-registered edge classifies as ``soundness`` -- a soundness link
    anywhere in the chain poisons the whole move. No connection at all returns
    ``None``: the unregistered rotation, which every consumer must read as
    soundness (fail closed).

    Each edge is first RESOLVED (:func:`_resolved_edge_classes`): superseded
    rows are dead, and an edge whose live rows disagree on class is
    CONFLICTED and reads as soundness -- fail-closed pending an explicit
    resolution row, never refinement-first (the v1 resolver's parallel-edge
    hole, fable-F3 / sol-04).

    Parameters
    ----------
    registrations:
        Validated registry rows.
    axis:
        Dependency-vector axis of the move.
    recorded, live:
        Recorded identity value and the live authority value.

    Returns
    -------
    str | None
        ``"refinement"``, ``"soundness"``, or ``None`` for unregistered.
    """

    if recorded == live:
        return "refinement"
    refinement_edges: dict[str, set[str]] = {}
    non_staling_edges: dict[str, set[str]] = {}
    any_edges: dict[str, set[str]] = {}
    for (_axis, source, target), entry in _resolved_edge_classes(
        registrations, axis=axis
    ).items():
        resolved = entry["resolved_class"]
        any_edges.setdefault(source, set()).add(target)
        if resolved == "refinement":
            refinement_edges.setdefault(source, set()).add(target)
        if resolved in ("refinement", "disclosed-non-staling"):
            # AL-22: both classes leave recorded proofs valid at their
            # recorded basis, so a chain mixing them still stales nothing.
            non_staling_edges.setdefault(source, set()).add(target)

    def _reaches(edges: Mapping[str, set[str]]) -> bool:
        seen = {recorded}
        frontier = [recorded]
        while frontier:
            node = frontier.pop()
            for successor in edges.get(node, ()):
                if successor == live:
                    return True
                if successor not in seen:
                    seen.add(successor)
                    frontier.append(successor)
        return False

    if _reaches(refinement_edges):
        return "refinement"
    if _reaches(non_staling_edges):
        return "disclosed-non-staling"
    if _reaches(any_edges):
        return "soundness"
    return None
