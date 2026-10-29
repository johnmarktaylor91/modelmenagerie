"""Measured authority rotations: ``records/control/rotations.jsonl`` (RC-2).

One row per axis move: ``{axis, from, to, class, reason, measured_by, tree_sha,
at}``. The class is MEASURED by the one instrument that exercises the axis
(:data:`MEASUREMENT_AUTHORITY`), never operator-declared:

* ``checker_prompt_identity`` -- the checker blind-replay rotation arm replays
  the golden corpus under the new prompt; a recorded ACCEPT that flips (either
  direction) makes the edge ``soundness`` (:func:`classify_golden_replay`);
* the three reducer-owned code axes (``terminal_rule_identity``,
  ``publication_policy_identity``, ``award_closure_identity``) -- the predicate
  replay probe re-derives every existing head under the new code and compares
  the recorded and derived dispositions field by field
  (:func:`menagerie.crawler.recheck_census.predicate_replay_probe`); any flip
  is ``soundness``, zero flips over a non-empty head set is ``refinement``;
* the four PRODUCER axes (``author_prompt_identity``,
  ``author_result_schema_identity``, ``author_dispatcher_identity``,
  ``runner_identity``) have NO replay instrument in this tree: recorded
  receipts disclose what the old producer did, not what the new one would do.
  No measured row can be written for them; the census reads an unconnected
  move on any axis as the CONSERVATIVE soundness class.

``measured_by = declared`` exists for exactly one thing: the one-time seed of
the frozen operator registry (``intake/registered-authority-rotations.jsonl``)
into this ledger with its recorded classes (RULING B.4). It is enforced
structurally, not by convention: :func:`append_rotation` refuses a declared
row unless the seed importer passes its content-addressed allowlist, and
:func:`load_rotations` recomputes that allowlist from the frozen registry at
the same records root and refuses the ledger if any declared row is outside
it. There is no operator write path to a positive class.

Consumers: the projection does NOT read this ledger (every settled head
replays at its recorded policy basis unconditionally since RC-2). The class
feeds exactly one derived view, the recheck census
(:mod:`menagerie.crawler.recheck_census`).

Class resolution is fail-closed and SIMPLE: soundness is sticky per
``(axis, from, to)`` edge -- any soundness row on an edge makes the edge
soundness, a later contradictory row is disclosed in the history and never a
silent downgrade, and there is no supersedes mechanism. Resolution is a
commutative OR over rows plus set-based reachability, so row append order can
never change a verdict; the LIVE identity is always the caller's authority
context, never a ledger row.

Every row is chain-linked (:mod:`menagerie.crawler.ledger_chain`, scope
``control.rotations``); links are REQUIRED on every row because the ledger
was born chained. The writer holds an exclusive ``fcntl.flock`` on the data
inode for the read-verify-link-append critical section.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.identity import durable_fsync, stable_hash
from menagerie.crawler.ledger_chain import (
    LedgerChainError,
    decode_ledger,
    link_row,
    verify_rows,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.rotation_registry import (
    ROTATION_AXES,
    RotationRegistrationError,
    load_rotation_registrations,
    rotation_registry_path,
)

__all__ = [
    "CODE_AXES",
    "MEASUREMENT_AUTHORITY",
    "MEASUREMENT_METHODS",
    "PRODUCER_AXES",
    "ROTATIONS_CHAIN_SCOPE",
    "ROTATION_AXES",
    "ROTATION_CLASSES",
    "ROTATION_SCHEMA_VERSION",
    "RotationLedgerError",
    "append_rotation",
    "build_rotation",
    "classify_golden_replay",
    "edge_classes",
    "load_rotations",
    "move_class",
    "rotation_history",
    "rotations_path",
    "seed_from_registry",
    "seed_rows_from_registry",
    "soundness_rotation_ids",
]

ROTATION_SCHEMA_VERSION = "menagerie.crawler.rotation.v1"
ROTATIONS_RELATIVE_PATH = Path("control") / "rotations.jsonl"
ROTATIONS_CHAIN_SCOPE = "control.rotations"

#: Closed measured classes. ``refinement``: recorded proofs stand at their
#: recorded basis and nothing is owed. ``soundness``: heads recorded before
#: the edge are ``needs_recheck`` until re-checked under the live authority.
#: (The old registry's ``disclosed-non-staling`` class folds into
#: ``refinement`` at seed: non-staling by construction IS refinement here.)
ROTATION_CLASSES = frozenset({"refinement", "soundness"})

#: Closed measurement methods (RULING D.2.1). ``declared`` is legal for the
#: seed only; the two replay methods are bound to axes below.
MEASUREMENT_METHODS = frozenset({"golden_replay", "predicate_replay", "declared"})

#: Reducer-owned code axes the predicate replay probe measures: the running
#: reducer re-derives every recorded disposition from recorded evidence, so
#: the comparison is exact and the instrument exercises exactly these files.
CODE_AXES = frozenset(
    {"terminal_rule_identity", "publication_policy_identity", "award_closure_identity"}
)

#: Producer axes: what the author or the worker DID. Recorded receipts cannot
#: say what a changed producer would do, so no instrument in this tree can
#: measure them and no measured row may be written; an unconnected move reads
#: soundness in the census (the conservative class).
PRODUCER_AXES = frozenset(
    {
        "author_prompt_identity",
        "author_result_schema_identity",
        "author_dispatcher_identity",
        "runner_identity",
    }
)

#: The ONE instrument allowed to write a measured row per axis (``None``: no
#: instrument exists; seed rows only). Registry-only axes
#: (``author_steering_surface_identity``, ``toolchain_identity``) are not
#: recorded-basis axes and carry seed rows only.
MEASUREMENT_AUTHORITY: Mapping[str, Optional[str]] = {
    "checker_prompt_identity": "golden_replay",
    "terminal_rule_identity": "predicate_replay",
    "publication_policy_identity": "predicate_replay",
    "award_closure_identity": "predicate_replay",
    "author_prompt_identity": None,
    "author_result_schema_identity": None,
    "author_dispatcher_identity": None,
    "runner_identity": None,
    "author_steering_surface_identity": None,
    "toolchain_identity": None,
}

_REQUIRED_FIELDS = (
    "schema_version",
    "rotation_id",
    "axis",
    "from",
    "to",
    "class",
    "reason",
    "measured_by",
    "tree_sha",
    "at",
)
_ALLOWED_FIELDS = frozenset(_REQUIRED_FIELDS) | {"chain"}


class RotationLedgerError(ValueError):
    """A rotation row is malformed, the ledger is unreadable, or its chain broke."""


def rotations_path(records_root: Path) -> Path:
    """Return the measured-rotations ledger below one records root."""

    return Path(records_root) / ROTATIONS_RELATIVE_PATH


def _rotation_id(row: Mapping[str, Any]) -> str:
    body = {key: value for key, value in row.items() if key not in ("rotation_id", "chain")}
    return "rot-" + stable_hash(body)[7:31]


def build_rotation(
    *,
    axis: str,
    from_identity: str,
    to_identity: str,
    rotation_class: str,
    reason: str,
    measured_by: str,
    tree_sha: str,
    at: str,
) -> JsonObject:
    """Assemble one validated, content-addressed rotation row.

    Parameters
    ----------
    axis:
        Rotated axis (member of :data:`ROTATION_AXES`).
    from_identity, to_identity:
        Exact recorded and live identity values (the JSON keys are ``from``
        and ``to``, exactly as the ruling spells them).
    rotation_class:
        ``refinement`` or ``soundness``.
    reason:
        What moved and what the measurement found (or the registration it
        seeds from).
    measured_by:
        Member of :data:`MEASUREMENT_METHODS`; a measured method must be the
        axis's :data:`MEASUREMENT_AUTHORITY`.
    tree_sha:
        Git tree the measurement ran on (or the seed tree).
    at:
        UTC instant of the measurement.

    Raises
    ------
    RotationLedgerError
        If any field is outside its closed vocabulary or empty, or the method
        is not the axis's authority.
    """

    row: JsonObject = {
        "schema_version": ROTATION_SCHEMA_VERSION,
        "axis": axis,
        "from": from_identity,
        "to": to_identity,
        "class": rotation_class,
        "reason": reason,
        "measured_by": measured_by,
        "tree_sha": tree_sha,
        "at": at,
    }
    row["rotation_id"] = _rotation_id(row)
    return _validate_rotation(row, source="build")


def _require_closed_vocabulary(row: Mapping[str, Any], *, source: str) -> None:
    """Refuse a row whose schema, axis, class, or method is outside its closed set."""

    closed = (
        ("schema_version", {ROTATION_SCHEMA_VERSION}, "declares unknown schema"),
        ("axis", ROTATION_AXES, "names an axis outside the closed set:"),
        ("class", ROTATION_CLASSES, "declares an unknown class:"),
        ("measured_by", MEASUREMENT_METHODS, "declares an unknown measurement method:"),
    )
    for field, vocabulary, message in closed:
        if row[field] not in vocabulary:
            raise RotationLedgerError(f"rotation ({source}) {message} {row[field]!r}")


def _require_axis_authority(axis: str, measured_by: str, *, source: str) -> None:
    """The axis-to-instrument binding (RC-2 Q4).

    A checker blind replay cannot vouch for an author axis, a predicate replay
    cannot vouch for the runner, and a producer axis has no instrument at all;
    ``declared`` is checked separately (seed-only) and passes here.
    """

    authority = MEASUREMENT_AUTHORITY.get(axis)
    if measured_by == "declared" or measured_by == authority:
        return
    detail = (
        f"that axis is measurable only by {authority!r}"
        if authority
        else "that axis has no replay instrument in this tree (an unconnected move reads "
        "soundness in the census; seed rows only)"
    )
    raise RotationLedgerError(
        f"rotation ({source}) on {axis} claims measured_by={measured_by!r}; {detail}"
    )


def _validate_rotation(row: Mapping[str, Any], *, source: str) -> JsonObject:
    """Validate one rotation row fail-closed and return a dict copy."""

    if not isinstance(row, Mapping):
        raise RotationLedgerError(f"rotation ({source}) is not an object")
    missing = [field for field in _REQUIRED_FIELDS if field not in row]
    if missing:
        raise RotationLedgerError(
            f"rotation ({source}) is missing {', '.join(sorted(missing))}"
        )
    unknown = sorted(set(map(str, row)) - _ALLOWED_FIELDS)
    if unknown:
        raise RotationLedgerError(
            f"rotation ({source}) carries fields outside the closed row: {unknown}"
        )
    _require_closed_vocabulary(row, source=source)
    _require_axis_authority(str(row["axis"]), str(row["measured_by"]), source=source)
    for field in ("from", "to", "reason", "tree_sha", "at", "rotation_id"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise RotationLedgerError(f"rotation ({source}) has an empty {field}")
    if row["from"] == row["to"]:
        raise RotationLedgerError(f"rotation ({source}) does not rotate: from and to are equal")
    if row["rotation_id"] != _rotation_id(row):
        raise RotationLedgerError(
            f"rotation ({source}) rotation_id does not match its body (rewritten row)"
        )
    return dict(row)


def _logical(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "chain"}


def _seed_allowlist_for(
    records_root: Path, tree_shas: Iterable[str], *, registry_path: Optional[Path] = None
) -> frozenset[str]:
    """Return the seed rotation_ids the frozen registry derives for each tree.

    Raises
    ------
    RotationLedgerError
        If the registry is absent or unreadable while a declared row exists:
        a declared row with no registry to derive it from is a forgery.
    """

    registry = registry_path or rotation_registry_path(records_root)
    if not registry.is_file():
        raise RotationLedgerError(
            f"declared rotation rows need the frozen registry at {registry}; none exists"
        )
    try:
        rows = load_rotation_registrations(registry)
    except (OSError, RotationRegistrationError, ValueError) as exc:
        raise RotationLedgerError(f"frozen registry {registry} refuses: {exc}") from exc
    allow: set[str] = set()
    for tree_sha in sorted(set(tree_shas)):
        allow.update(row["rotation_id"] for row in seed_rows_from_registry(rows, tree_sha=tree_sha))
    return frozenset(allow)


def _require_declared_rows_seeded(
    records_root: Path, rows: Sequence[Mapping[str, Any]], *, registry_path: Optional[Path]
) -> None:
    """Refuse any ``declared`` row the frozen registry does not derive (Q5)."""

    declared = [row for row in rows if row.get("measured_by") == "declared"]
    if not declared:
        return
    allow = _seed_allowlist_for(
        records_root, (str(row["tree_sha"]) for row in declared), registry_path=registry_path
    )
    foreign = sorted(str(row["rotation_id"]) for row in declared if row["rotation_id"] not in allow)
    if foreign:
        raise RotationLedgerError(
            "declared rotation rows outside the frozen-registry seed: "
            f"{foreign}; `declared` is legal for the seed only -- a positive class needs "
            "the axis's replay authority"
        )


def load_rotations(
    records_root: Path, *, registry_path: Optional[Path] = None
) -> tuple[JsonObject, ...]:
    """Load, chain-verify, and validate every rotation row; a missing file is empty.

    Parameters
    ----------
    records_root:
        Records root holding ``control/rotations.jsonl`` and the frozen
        registry the seed rows derive from.
    registry_path:
        Override for the frozen registry location (tests, foreign roots).

    Raises
    ------
    RotationLedgerError
        If any row fails validation, the chain does not verify, or a
        ``declared`` row is not a seed row the frozen registry derives -- an
        unreadable ledger must never degrade to "no rotations" (that would
        read every edge as absent).
    """

    path = rotations_path(records_root)
    if not path.is_file():
        return ()
    try:
        raw = decode_ledger(path)
        verify_rows(raw, scope=ROTATIONS_CHAIN_SCOPE, require_links=True)
    except LedgerChainError as exc:
        raise RotationLedgerError(f"rotations ledger {path} refuses: {exc}") from exc
    rows = tuple(_validate_rotation(row, source=f"row {index}") for index, row in enumerate(raw))
    _require_declared_rows_seeded(Path(records_root), rows, registry_path=registry_path)
    return rows


def append_rotation(
    records_root: Path,
    row: Mapping[str, Any],
    *,
    seed_allowlist: frozenset[str] = frozenset(),
) -> bool:
    """Append one rotation row idempotently, chain-linked, under the writer lock.

    Parameters
    ----------
    records_root:
        Records root.
    row:
        Validated rotation row.
    seed_allowlist:
        Content-addressed rotation_ids the seed importer derived from the
        frozen registry. A ``declared`` row is refused unless its id is here;
        every other caller leaves it empty and cannot write a declared row.

    Returns
    -------
    bool
        Whether new bytes were appended (``False`` for an identical replay).

    Raises
    ------
    RotationLedgerError
        If the row is malformed, a declared row is not a seed row, the
        ledger's chain does not verify, or an existing row holds the same
        ``rotation_id`` with different bytes.
    """

    candidate = _validate_rotation(row, source="append")
    if candidate["measured_by"] == "declared" and candidate["rotation_id"] not in seed_allowlist:
        raise RotationLedgerError(
            f"rotation {candidate['rotation_id']} claims measured_by=declared outside the "
            "frozen-registry seed; a positive class needs the axis's replay authority"
        )
    path = rotations_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    with os.fdopen(descriptor, "r+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            try:
                existing = decode_ledger(path, locked_handle=handle)
                state = verify_rows(existing, scope=ROTATIONS_CHAIN_SCOPE, require_links=True)
            except LedgerChainError as exc:
                raise RotationLedgerError(f"rotations ledger {path} refuses: {exc}") from exc
            for prior in existing:
                if prior.get("rotation_id") == candidate["rotation_id"]:
                    if _logical(prior) == candidate:
                        return False
                    raise RotationLedgerError(
                        f"rotations ledger holds conflicting bytes for {candidate['rotation_id']!r}"
                    )
            try:
                linked = link_row(
                    candidate,
                    scope=ROTATIONS_CHAIN_SCOPE,
                    existing_rows=existing,
                    existing_state=state,
                )
            except LedgerChainError as exc:
                raise RotationLedgerError(str(exc)) from exc
            handle.seek(0, os.SEEK_END)
            handle.write((json.dumps(linked, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            durable_fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return True


def edge_classes(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    """Resolve every ``(axis, from, to)`` edge to its class; soundness is sticky.

    A commutative OR over the rows: no append order can change a result.
    """

    classes: dict[tuple[str, str, str], str] = {}
    for row in rows:
        edge = (str(row.get("axis")), str(row.get("from")), str(row.get("to")))
        if str(row.get("class")) == "soundness" or classes.get(edge) == "soundness":
            classes[edge] = "soundness"
        else:
            classes[edge] = "refinement"
    return classes


def _reaches(edges: Mapping[str, set[str]], *, start: str, goal: str) -> bool:
    seen = {start}
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for successor in edges.get(node, ()):
            if successor == goal:
                return True
            if successor not in seen:
                seen.add(successor)
                frontier.append(successor)
    return False


def move_class(
    rows: Iterable[Mapping[str, Any]],
    *,
    axis: str,
    recorded: str,
    live: str,
) -> Optional[str]:
    """Classify one recorded->live identity move against the ledger.

    ``refinement`` when the identities are equal or a chain of refinement
    edges connects them; ``soundness`` when a connection exists but every
    connection needs a soundness edge; ``None`` when nothing connects them
    (an UNMEASURED move -- the census reads it as soundness, the conservative
    class, and discloses that no row measured it).

    ``live`` is the caller's authority-context value: this function never
    infers the live identity from the ledger.
    """

    if recorded == live:
        return "refinement"
    refinement_edges: dict[str, set[str]] = {}
    any_edges: dict[str, set[str]] = {}
    for (edge_axis, source, target), resolved in edge_classes(rows).items():
        if edge_axis != axis:
            continue
        any_edges.setdefault(source, set()).add(target)
        if resolved == "refinement":
            refinement_edges.setdefault(source, set()).add(target)
    if _reaches(refinement_edges, start=recorded, goal=live):
        return "refinement"
    if _reaches(any_edges, start=recorded, goal=live):
        return "soundness"
    return None


def soundness_rotation_ids(
    rows: Iterable[Mapping[str, Any]], *, axis: str, recorded: str
) -> tuple[str, ...]:
    """Return the soundness rows on ``axis`` reachable from ``recorded`` (the authority)."""

    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("axis")) == axis:
            by_source.setdefault(str(row.get("from")), []).append(row)
    ids: set[str] = set()
    seen = {recorded}
    frontier = [recorded]
    while frontier:
        node = frontier.pop()
        for row in by_source.get(node, ()):
            if str(row.get("class")) == "soundness":
                ids.add(str(row.get("rotation_id")))
            target = str(row.get("to"))
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    return tuple(sorted(ids))


def rotation_history(rows: Iterable[Mapping[str, Any]]) -> list[JsonObject]:
    """Render the ledger as ``status`` disclosure (``configuration_history``)."""

    history: list[JsonObject] = []
    for row in rows:
        chain = row.get("chain") if isinstance(row.get("chain"), Mapping) else {}
        history.append(
            {
                "rotation_id": row.get("rotation_id"),
                "axis": row.get("axis"),
                "from": row.get("from"),
                "to": row.get("to"),
                "class": row.get("class"),
                "measured_by": row.get("measured_by"),
                "reason": row.get("reason"),
                "tree_sha": row.get("tree_sha"),
                "at": row.get("at"),
                "chain_sha256": chain.get("sha256"),
            }
        )
    return history


def seed_rows_from_registry(
    registry_rows: Iterable[Mapping[str, Any]], *, tree_sha: str
) -> list[JsonObject]:
    """Derive the one-time seed rows from the frozen operator-declared registry.

    One row per registered row, with its recorded class (the non-staling
    class folds into ``refinement``), ``measured_by = declared``, the original
    registration instant as ``at``, and a reason naming the registration id
    (plus any supersession the row carried) so the history stays auditable.
    Soundness stickiness then reproduces the measured 2026-08-31 result on
    the disputed checker_prompt edge without any special case. The output is
    a pure function of (registry rows, tree_sha): it is also the allowlist
    that makes ``declared`` seed-only.
    """

    seeds: list[JsonObject] = []
    for row in registry_rows:
        recorded_class = str(row.get("rotation_class"))
        rotation_class = "soundness" if recorded_class == "soundness" else "refinement"
        reason = (
            f"seeded from {row.get('registration_id')} "
            f"({row.get('registered_by')}, class {recorded_class}): {row.get('rationale')}"
        )
        if row.get("supersedes"):
            reason += (
                f" [supersedes {row.get('supersedes')}: {row.get('supersession_evidence')}]"
            )
        seeds.append(
            build_rotation(
                axis=str(row.get("axis")),
                from_identity=str(row.get("from_identity")),
                to_identity=str(row.get("to_identity")),
                rotation_class=rotation_class,
                reason=reason,
                measured_by="declared",
                tree_sha=tree_sha,
                at=str(row.get("created_at")),
            )
        )
    return seeds


def seed_from_registry(
    records_root: Path, *, tree_sha: str, registry_path: Optional[Path] = None
) -> dict[str, int]:
    """Import the frozen registry into the measured ledger once (idempotent replay).

    Returns
    -------
    dict[str, int]
        ``{"registry_rows": n, "appended": a, "already_present": p}``.
    """

    registry = registry_path or rotation_registry_path(records_root)
    rows = load_rotation_registrations(registry)
    seeds = seed_rows_from_registry(rows, tree_sha=tree_sha)
    allow = frozenset(seed["rotation_id"] for seed in seeds)
    appended = 0
    for seed in seeds:
        if append_rotation(records_root, seed, seed_allowlist=allow):
            appended += 1
    return {
        "registry_rows": len(rows),
        "appended": appended,
        "already_present": len(rows) - appended,
    }


_ACCEPT_VERDICT_BY_KIND: Mapping[str, tuple[str, str]] = {
    "fidelity": ("fidelity", "match"),
    "terminal_disposition": ("terminal_disposition", "accepted"),
    "metadata_batch": ("verdict", "accurate"),
}


def _accepts(kind: str, item: Any) -> bool:
    """Return whether one verdict projection item is a checker ACCEPT.

    The blind-replay matrix projects each golden to ``{stable_id: verdict}``
    with the verdict as a bare string; a structured item carries the verdict
    under the kind's own key.
    """

    key, accept = _ACCEPT_VERDICT_BY_KIND.get(kind, ("verdict", "accurate"))
    value = item.get(key) if isinstance(item, Mapping) else item
    return value == accept


def _golden_accept_flips(
    golden: Mapping[str, Any], rotation: Mapping[str, Any]
) -> list[JsonObject]:
    """Return the per-item accept flips between a golden's first same-epoch and rotation projections."""

    name = str(golden.get("name"))
    kind = str(golden.get("kind"))
    same_epoch = (golden.get("same_epoch") or {}).get("projections") or []
    if not same_epoch:
        raise RotationLedgerError(f"golden {name!r} carries no same-epoch projection")
    baseline = same_epoch[0] if isinstance(same_epoch[0], Mapping) else {}
    rotated = rotation.get("projection") if isinstance(rotation.get("projection"), Mapping) else {}
    flips: list[JsonObject] = []
    for stable_id in sorted(set(baseline) | set(rotated)):
        before = baseline.get(stable_id)
        after = rotated.get(stable_id)
        if before != after and (_accepts(kind, before) or _accepts(kind, after)):
            flips.append({"golden": name, "stable_id": stable_id, "before": before, "after": after})
    return flips


def classify_golden_replay(report: Mapping[str, Any], *, axis: str) -> JsonObject:
    """Classify a checker-prompt move from a blind-replay report's rotation arm.

    Every golden's rotation replay is compared item by item with its first
    same-epoch projection (exact old verdict vs new verdict); a difference
    where EITHER side is an accept is a recorded accept that flipped, and one
    such flip makes the move ``soundness``. A report whose rotation arm did
    not run for every golden cannot classify anything and refuses.

    Parameters
    ----------
    report:
        A checker blind-replay matrix report.
    axis:
        The axis being classified. Only ``checker_prompt_identity`` is
        accepted: the instrument re-submits frozen CHECKER requests, so it
        exercises no author axis (an author prompt change alters the authored
        proposal while the frozen checker request stays byte-identical).

    Returns
    -------
    dict[str, Any]
        ``{"class", "goldens", "flipped": [...]}``.
    """

    if MEASUREMENT_AUTHORITY.get(axis) != "golden_replay":
        raise RotationLedgerError(
            f"a checker blind-replay report measures checker_prompt_identity only; it does "
            f"not exercise {axis} and cannot classify its move"
        )
    goldens = report.get("goldens")
    if not isinstance(goldens, Sequence) or not goldens:
        raise RotationLedgerError("blind-replay report carries no goldens; nothing to classify")
    flipped: list[JsonObject] = []
    unran: list[str] = []
    for golden in goldens:
        rotation = golden.get("rotation") or {}
        if not isinstance(rotation, Mapping) or not rotation.get("ran"):
            unran.append(str(golden.get("name")))
            continue
        flipped.extend(_golden_accept_flips(golden, rotation))
    if unran:
        raise RotationLedgerError(
            "the rotation arm did not run for every golden; a partial replay classifies "
            f"nothing: {sorted(unran)}"
        )
    return {
        "class": "soundness" if flipped else "refinement",
        "goldens": len(goldens),
        "flipped": flipped,
    }
