"""Park release: void-on-record disclosure plus D-48's machine release lane.

BD-1's second half. Park release is keyed on the RAW LEDGER HEAD everywhere it
is derived, so a park whose model later recorded a revision is inert for every
census by construction. This module makes the LEDGER say so as well: at close,
every park row whose own release predicate reports it consumed gets one durable
void row naming the revision that consumed it.

Why it is not an operator decision (and not a bypass of one):

- The void is derived from the SAME predicate every census consults
  (:func:`~menagerie.crawler.refusal_parks.refusal_park_released` /
  :func:`~menagerie.crawler.grant_parks.grant_park_released`), so it voids only
  rows that already account for nothing. It cannot release a live decision, it
  cannot claim coverage, and it cannot move a close toward ``complete``.
- It is authored under its own machine actor, never an operator identity, so
  ledger readers can tell a disclosure from a decision.
- Voiding never swallows a future fire: park append idempotency consults
  UNVOIDED rows only, so a genuinely new refusal for the same work generation
  appends fresh.

Without it, park ledgers accumulate rows asserting review-held claims that
resolved hours earlier -- 5 such rows at the step-100 close (T2-opus-07),
growing to 17 rows / 14 models during the finishing legs, every one of them a
human decision request about a model that was already done.

D-48 (the second half of this module): the park exit made mechanically
REACHABLE. Clause 5 of the ONE work-selection policy makes every live park row
absolute -- a parked member is not-work in every mode, BEFORE the grant arm is
consulted. For grant-pending and award parks that is honest: their own remedy
text names a grant the selection layer releases against
(:func:`~menagerie.crawler.driver._park_released_for_selection`). For the
TERMINAL-REFUSAL park it was not: its module contract promised "the next leg
attempts the model normally and the ladder re-parks idempotently", and its
recheck hint told operators a requeue would land a releasing revision -- but a
revision can only land through selection admission, admission is exactly what
clause 5 refuses, and ``_park_released_for_selection`` returns ``False`` for
the refusal disposition. The release mechanism was therefore mechanically
unreachable (D1-OS confirmed; the no-head-park fixture in
``tests/test_park_release_lane.py`` decides it definitively), and 200a
accumulated parks whose only real exit was an operator void.

The adjudicated fix (PLAN_v6 §2 D-48, preferred option) is a bounded machine
release lane for the MACHINERY-FAULT park class only:

* a live terminal-refusal park -- evidence the RECORDING PATH, not the member,
  is broken -- readmits its member for ONE fresh work generation per change of
  the running code tree (``tree_sha``), and the recording ladder re-parks
  idempotently if the refusal still holds;
* the re-park row records the tree it re-parked under (the parking tree joins
  the park FACT identity), so the member is machine-withheld again until the
  NEXT tree change -- one attempt per tree, never a per-leg retry loop;
* grant-pending and award parks keep their existing grant release lever and
  gain no machine lane; a member claimed by BOTH a machinery-fault park and an
  operator park stays absolutely parked (the strictest claim governs);
* every live park row names its COMPUTED release path at close time
  (:func:`park_release_path_census`); a row whose release path cannot be
  computed is a typed close refusal, never a silent void-only dead end.

Wave placement: Wave 2 by law (the lane extends the park/release vocabulary
and the refusal-park row schema). JMT signature record: STATE.md "JMT APPROVALS
RECORDED (2026-08-24)" -- the signed §6 C.8 dispositions (the five fidelity-park
re-drives, the duplicate-custody canary) execute THROUGH this lane, never via
blanket voids (§2.2 OS-Q3); the signature authorizes those dispositions, and
this change still carries its own proofs.

Fail-open boundary, stated: a leg that admits a member through this lane and
dies before the recording ladder re-parks leaves no durable spend marker, so
the next leg under the SAME tree re-buys one attempt. That is the documented
valve doctrine (fail-open toward attempting, bounded by the durable re-park on
every completed attempt), not an unbounded loop: every COMPLETED attempt under
an unchanged tree ends in the idempotent re-park that closes the lane.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.grant_parks import (
    AWARD_PARK_DISPOSITION,
    GRANT_PENDING_DISPOSITION,
    append_grant_park_void,
    build_grant_park_void,
    grant_park_path,
    grant_park_released,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.refusal_parks import (
    REFUSAL_PARK_DISPOSITION,
    append_refusal_park_void,
    build_refusal_park_void,
    refusal_park_path,
    refusal_park_released,
)

#: Actor recorded on a void this module appends. It is deliberately NOT an
#: operator identity: the row discloses that a park's own release condition
#: materialized, which is a machine observation about landed records, and an
#: auditor must be able to separate it from the operator voids that ARE
#: decisions.
PARK_RELEASE_VOID_ACTOR = "machine:park-released-on-record"

#: The closed machinery-fault park class: park dispositions whose row is
#: evidence the MACHINE (the recording path) refused, so a code-tree change is
#: the causal remedy their own doctrine names. Extending this set is a
#: park/release vocabulary change and merges only in a rotation batch.
MACHINERY_FAULT_PARK_DISPOSITIONS = frozenset({REFUSAL_PARK_DISPOSITION})

#: Closed release-path vocabulary (the close's per-row disclosure). Every live
#: park row computes to exactly one of these; anything else refuses typed.
RELEASE_PATH_MACHINE_RETRY_NOW = "machine-retry:fresh-generation-under-current-tree"
RELEASE_PATH_MACHINE_RETRY_ON_TREE_CHANGE = "machine-retry:on-next-tree-change"
RELEASE_PATH_OPERATOR_GRANT_STAGE = "operator-grant:requeue-stage"
RELEASE_PATH_OPERATOR_GRANT_AWARD = "operator-grant:requeue-blocked-award-revision"

RELEASE_PATH_VOCABULARY = frozenset(
    {
        RELEASE_PATH_MACHINE_RETRY_NOW,
        RELEASE_PATH_MACHINE_RETRY_ON_TREE_CHANGE,
        RELEASE_PATH_OPERATOR_GRANT_STAGE,
        RELEASE_PATH_OPERATOR_GRANT_AWARD,
    }
)


class ParkReleasePathError(ValueError):
    """Raised when a live park row's release path cannot be computed.

    The close treats this as a refusal: a park with no computable exit is the
    exact mechanically-unreachable dead end D-48 exists to remove, so it may
    never be silently accounted as an ordinary withheld member.
    """


def _void_reason(revision: str) -> str:
    """Return the disclosure text for one released park row.

    Parameters
    ----------
    revision:
        ``record_revision`` of the ledger head that consumed the park.

    Returns
    -------
    str
        Reason naming the consuming revision.
    """

    return (
        "released by a landed canonical revision: the model's ledger head is "
        f"{revision}, which the park row's own release predicate reports as "
        "consuming it. Disclosure only -- this void grants nothing and decides "
        "nothing; a genuinely new refusal for this work generation still "
        "appends a fresh governing row."
    )


def void_released_parks(
    records_root: Path,
    persisted_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    ledger_heads: Mapping[str, Mapping[str, Any]],
    *,
    created_at: str,
    scoped_ids: Optional[Iterable[str]] = None,
) -> tuple[JsonObject, ...]:
    """Append one void per park row the model's ledger head already released.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory holding both park ledgers.
    persisted_rows:
        Governing durable valve rows per stable ID.
    ledger_heads:
        Stable-ID keyed RAW ledger heads.
    created_at:
        UTC timestamp of the close appending the disclosures.
    scoped_ids:
        Optional membership restriction (the close's in-scope members).

    Returns
    -------
    tuple[dict[str, Any], ...]
        Appended (or already-present, idempotent) void rows in append order.
    """

    from menagerie.crawler.grant_parks import existing_grant_park_release
    from menagerie.crawler.refusal_parks import existing_refusal_park_release

    kinds = {
        GRANT_PENDING_DISPOSITION: (
            grant_park_released,
            grant_park_path(records_root),
            build_grant_park_void,
            append_grant_park_void,
            existing_grant_park_release,
        ),
        REFUSAL_PARK_DISPOSITION: (
            refusal_park_released,
            refusal_park_path(records_root),
            build_refusal_park_void,
            append_refusal_park_void,
            existing_refusal_park_release,
        ),
    }
    in_scope = None if scoped_ids is None else frozenset(str(value) for value in scoped_ids)
    voided: list[JsonObject] = []
    for stable_id, rows in sorted(persisted_rows.items()):
        if in_scope is not None and stable_id not in in_scope:
            continue
        head = ledger_heads.get(stable_id)
        if head is None:
            continue
        revision = head.get("record_revision")
        if not isinstance(revision, str) or not revision:
            continue
        for row in rows:
            kind = kinds.get(str(row.get("disposition")))
            if kind is None:
                continue
            is_released, path, build, append, existing_release = kind
            if not is_released(row, head):
                continue
            digest = row.get("row_sha256")
            if not isinstance(digest, str) or not digest:
                continue
            # Re-close idempotency under the strict release door (RULING W7
            # clause 9): a release must name a LIVE row, so a row this lane
            # already released on an earlier close keeps its standing release
            # instead of appending (and being refused) a second, different one.
            prior = existing_release(path, digest)
            if prior is not None:
                voided.append(prior)
                continue
            voided.append(
                append(
                    path,
                    build(
                        void_of=digest,
                        reason=_void_reason(revision),
                        voided_by=PARK_RELEASE_VOID_ACTOR,
                        created_at=created_at,
                    ),
                )
            )
    return tuple(voided)


def current_park_tree_identity(repo_root: Optional[Path] = None) -> Optional[str]:
    """Derive the running code tree's identity for the park release lane.

    ``HEAD^{tree}`` of the repository enclosing the running ``menagerie``
    package (or ``repo_root`` when given), with a ``-dirty`` suffix when the
    working tree carries uncommitted changes. ``None`` when no identity is
    derivable -- underivable fails CLOSED for readmission (the member stays
    parked; the operator void lever is untouched), because a lane that cannot
    name the tree it runs cannot prove the causal remedy happened.

    Parameters
    ----------
    repo_root:
        Repository to describe; defaults to the tree enclosing this module.

    Returns
    -------
    str | None
        Tree identity, or ``None`` when underivable.
    """

    root = repo_root if repo_root is not None else Path(__file__).resolve().parent

    def _git(*arguments: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ("git", "-C", str(root), *arguments),
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip() or None

    tree = _git("rev-parse", "HEAD^{tree}")
    if tree is None:
        return None
    porcelain = _git("status", "--porcelain")
    return f"{tree}-dirty" if porcelain else tree


def derive_park_machine_readmission(
    member_rows: Sequence[Mapping[str, Any]],
    *,
    tree_identity: Optional[str],
) -> Optional[JsonObject]:
    """Replay-derive whether one member's live parks machine-readmit it now.

    Pure and replay-derived from the governing durable rows plus the running
    tree identity -- no persisted readmission bit, zero producer input. The
    member readmits exactly when EVERY live park row claiming it is in the
    closed machinery-fault class AND every such row's recorded parking tree
    provably differs from the running tree:

    * a row recording parking tree T readmits only under a tree that is not T;
    * a pre-D-48 row (no ``tree_sha`` field, or an explicit ``None`` recorded
      where the tree was underivable at park time) readmits once -- its
      refusal provably predates a decidable tree binding, and the re-park it
      earns records the current tree, closing the loop;
    * an underivable running tree (``tree_identity is None``) never readmits
      (fail-closed: the causal remedy cannot be proven);
    * any non-machinery-fault live park row (grant-pending, award) vetoes the
      lane outright -- the strictest live claim governs.

    Parameters
    ----------
    member_rows:
        Every LIVE (governing, unvoided, unreleased) park row claiming one
        member, across all park dispositions.
    tree_identity:
        The running code tree identity, or ``None`` when underivable.

    Returns
    -------
    dict[str, Any] | None
        The readmission fact -- governing row digest, its recorded parking
        tree, and the running tree -- or ``None`` when the member stays parked.
    """

    if not member_rows or tree_identity is None:
        return None
    machinery: list[Mapping[str, Any]] = []
    for row in member_rows:
        if str(row.get("disposition")) not in MACHINERY_FAULT_PARK_DISPOSITIONS:
            return None
        machinery.append(row)
    for row in machinery:
        parked_tree = row.get("tree_sha")
        if parked_tree is not None and str(parked_tree) == tree_identity:
            return None
    governing = machinery[-1]
    return {
        "row_sha256": str(governing.get("row_sha256")),
        "parked_tree_sha": governing.get("tree_sha"),
        "current_tree_sha": tree_identity,
    }


def compute_release_path(
    row: Mapping[str, Any],
    *,
    tree_identity: Optional[str],
) -> str:
    """Compute one live park row's release path from the closed vocabulary.

    Parameters
    ----------
    row:
        One live durable park row.
    tree_identity:
        The running code tree identity, or ``None`` when underivable.

    Returns
    -------
    str
        One member of :data:`RELEASE_PATH_VOCABULARY`.

    Raises
    ------
    ParkReleasePathError
        When the row's disposition is outside the closed park vocabulary --
        the mechanically-unreachable dead end the close must refuse on.
    """

    disposition = str(row.get("disposition"))
    if disposition == GRANT_PENDING_DISPOSITION:
        return RELEASE_PATH_OPERATOR_GRANT_STAGE
    if disposition == AWARD_PARK_DISPOSITION:
        return RELEASE_PATH_OPERATOR_GRANT_AWARD
    if disposition in MACHINERY_FAULT_PARK_DISPOSITIONS:
        if derive_park_machine_readmission([row], tree_identity=tree_identity) is not None:
            return RELEASE_PATH_MACHINE_RETRY_NOW
        return RELEASE_PATH_MACHINE_RETRY_ON_TREE_CHANGE
    raise ParkReleasePathError(
        f"live park row {row.get('row_sha256')!r} for {row.get('stable_id')!r} has "
        f"disposition {disposition!r} outside the closed park vocabulary: no release "
        "path is computable, which is exactly the mechanically-unreachable exit the "
        "close refuses to account silently"
    )


def park_release_path_census(
    live_parks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    tree_identity: Optional[str],
) -> dict[str, dict[str, str]]:
    """Compute the close's per-row release-path disclosure over every live park.

    Disclosure only (waiver doctrine part 4 shape): recomputed from the durable
    rows on every close, never persisted as authority, and it can never improve
    a close decision -- an uncomputable path REFUSES the close via
    :class:`ParkReleasePathError`.

    Parameters
    ----------
    live_parks:
        Live park rows keyed by disposition then stable ID (the
        :func:`~menagerie.crawler.driver._live_park_rows` shape).
    tree_identity:
        The running code tree identity, or ``None`` when underivable.

    Returns
    -------
    dict[str, dict[str, str]]
        Per-disposition, per-member computed release path.

    Raises
    ------
    ParkReleasePathError
        When any live row's release path is uncomputable.
    """

    census: dict[str, dict[str, str]] = {}
    for disposition in sorted(live_parks):
        rows = live_parks[disposition]
        census[disposition] = {
            stable_id: compute_release_path(rows[stable_id], tree_identity=tree_identity)
            for stable_id in sorted(rows)
        }
    return census
