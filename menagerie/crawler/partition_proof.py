"""``PartitionProofV2``: the sealed, generation-scoped partition proof (v2 core).

The single coverage authority of a leg close. Every intake member cites ONE
exact evidence item by content address -- never "evidence exists somewhere in
campaign history":

* **adjudicated** -- a model-record row IN THIS LEG'S WINDOW that is the
  member's current head AND is recorded-under this snapshot. The two signals
  must agree; a contradiction (an in-window row claiming an ancestor, a
  pre-window row claiming this snapshot) refuses rather than passing either.
* **carried** -- a member a NAMED ancestor close already resolved, untouched
  by this leg (no in-window row, no in-window attempt, not plan-selected),
  pinned to the unchanged revision and delegated BY DIGEST to a close that
  still resides byte-identically.
* **withheld** -- a valve/park row appended IN THIS LEG'S WINDOW, valid per
  the same gauntlet every partition census applies, and LIVE at close
  (unvoided, unreleased against the member's raw head).

The proof is REPLAY-DERIVED: :func:`derive_partition_proof` is the one
derivation, computed from pinned bytes + the registry + the snapshot's own
membership, used identically by the producer (driver / backfill tool) and by
the staging gate's recomputation. A producer hash that does not equal the
gate's recomputation refuses with a per-member diff. No producer-written arm
feeds the derivation (four-part waiver doctrine, part 3).

``close_kind="backfill"`` selects the era-honest predicates for pre-machinery
legs (no window existed): adjudicated-backfill is "current head recorded-under
EXACTLY this snapshot id" -- the discriminator the register door already
trusts -- and an unattributable pre-machinery head resolves only through an
explicit operator authority naming the exact current revision.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Optional, Sequence

from menagerie.crawler.close_reads import (
    DECODED_ROW_WEIGHT_FACTOR,
    LINE_SPAN_WEIGHT,
    PROJECTION_BASE_WEIGHT,
    LedgerRow,
    derive,
    read_epoch,
)
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    stable_hash,
)
from menagerie.crawler.leg_generation import (
    ATTEMPT_SOURCE_NAMES,
    CLOSE_SOURCE_NAMES,
    WITHHOLDING_SOURCE_NAMES,
    LegGenerationError,
    MODELS_SOURCE_NAMES,
    census_identities,
    close_ledger_sources,
    model_head_citations,
    models_source_projection,
    source_row_window,
)
from menagerie.crawler.invocation_identity import close_leg_citations
from menagerie.crawler.models import JsonObject

PARTITION_PROOF_SCHEMA_VERSION_V3 = "menagerie.crawler.partition-proof.v3"
"""The pre-PD-1 proof era, PINNED FOREVER (RULING W7 clause 8).

Every sealed v3 proof (all of 200a) re-derives and re-validates
BYTE-IDENTICALLY under v3 semantics -- marker-only members close WITHHELD and
prove -- and its licence stands, disclosed per PD-13, forward-only. The era is
keyed by the proof's OWN schema version at replay; it is NOT producer-choosable
for new attestations (the close append door refuses a fresh close binding a
non-current era)."""

PARTITION_PROOF_SCHEMA_VERSION_V4 = "menagerie.crawler.partition-proof.v4"
"""The post-PD-1 proof era: RULING W7's continuation-marker semantics.

Routing markers are excluded from the head-contradiction predicate by closed
census membership AND replay-derived custody silence; marker-only members close
as disclosed CONTINUING members that never prove and never license."""

PARTITION_PROOF_SCHEMA_VERSION = PARTITION_PROOF_SCHEMA_VERSION_V4

PARTITION_PROOF_SCHEMA_VERSIONS = frozenset(
    {PARTITION_PROOF_SCHEMA_VERSION_V3, PARTITION_PROOF_SCHEMA_VERSION_V4}
)
"""Every replayable proof era (append-only history stays valid)."""

LEG_PROOFS_RELATIVE_PATH = ("intake", "leg-proofs")

ADJUDICATED = "adjudicated"
CARRIED = "carried"
WITHHELD = "withheld"
OPERATOR_ATTRIBUTED = "operator_attributed"
CONTINUING = "continuing"

PROOF_DISPOSITIONS = (ADJUDICATED, CARRIED, WITHHELD, OPERATOR_ATTRIBUTED, CONTINUING)

LicenceVerdict = Literal[
    "LICENSING", "HONEST_NO_OP", "INCOMPLETE_LADDER", "COMPLETE_AT_WIDTH"
]

#: The historical stringified absence T2-12 froze into 62 pilot valve-row
#: digests. Backfill reads tolerate it (the rows are real history); live leg
#: windows never do (new-era rows bind a real campaign at build time).
_PILOT_ABSENT_CAMPAIGN = "None"


class PartitionProofError(ValueError):
    """Raised when a partition proof cannot be derived, stored, or verified."""

    def __init__(self, message: str, *, member_reasons: Optional[Mapping[str, str]] = None):
        super().__init__(message)
        self.member_reasons = dict(member_reasons or {})


def award_transition_census(
    records_root: Path,
    baselines: Optional[Mapping[str, Any]],
    *,
    coverage_baselines: Optional[Mapping[str, Any]] = None,
) -> JsonObject:
    """Derive separate in-window award destruction and remint arms.

    Parameters
    ----------
    records_root:
        Canonical campaign records directory.
    baselines:
        Leg-open source identities, or ``None`` for pre-machinery backfill.
    coverage_baselines:
        Optional earlier, fully enumerated award-census floor. The live close
        path supplies the predecessor close's model-source extents so appends
        between that close and this leg's open belong to this census.

    Returns
    -------
    dict[str, Any]
        Instrumentation status, exact covered windows, and transition citations.

    Raises
    ------
    PartitionProofError
        If a purported instrumented census omits a model source or names an
        impossible coverage floor.
    """

    census: JsonObject = {
        "instrumentation": "uninstrumented" if baselines is None else "instrumented",
        "coverage_windows": None,
        "runs_to_non_runs": [],
        "runs_to_runs": [],
    }
    if baselines is None:
        return census
    missing = sorted(set(MODELS_SOURCE_NAMES) - set(baselines))
    if missing:
        raise PartitionProofError(
            "partial award-transition baseline omits model source(s): "
            + ", ".join(missing)
        )
    effective = baselines if coverage_baselines is None else coverage_baselines
    missing_coverage = sorted(set(MODELS_SOURCE_NAMES) - set(effective))
    if missing_coverage:
        raise PartitionProofError(
            "partial award-transition baseline omits model source(s): "
            + ", ".join(missing_coverage)
        )
    sources = close_ledger_sources(records_root)
    projections = {
        source_name: models_source_projection(sources[source_name])
        for source_name in MODELS_SOURCE_NAMES
    }
    # The predecessor lookup is campaign-global BY DEFINITION -- an in-window
    # successor's parent normally lives BELOW the floor -- so it comes from the
    # held models projection (revision to that row's raw ``status``), which is
    # folded once per observed ledger state. The SUCCESSOR scan is bound to the
    # generation window: only rows at or above the floor can be transitions,
    # and only those rows are read off disk (round-2 T4 / N4).
    by_revision: dict[str, Any] = {}
    for source_name in MODELS_SOURCE_NAMES:
        by_revision.update(projections[source_name].revision_status)
    for source_name in MODELS_SOURCE_NAMES:
        leg_floor = int(dict(baselines.get(source_name) or {}).get("rows") or 0)
        floor = int(dict(effective.get(source_name) or {}).get("rows") or 0)
        if floor > leg_floor:
            raise PartitionProofError(
                f"award-transition coverage floor for {source_name} ({floor}) lies "
                f"above the leg-open baseline ({leg_floor})"
            )
        if floor > projections[source_name].rows:
            raise PartitionProofError(
                f"award-transition coverage floor for {source_name} ({floor}) lies "
                f"above the observed close extent ({projections[source_name].rows})"
            )
        window = source_row_window(
            sources[source_name], floor, projections[source_name].rows
        )
        for offset, payload in enumerate(window):
            row_index = floor + offset
            row = json.loads(payload)
            if not isinstance(row, Mapping):
                continue
            parent_revision = row.get("parent_revision")
            if not parent_revision or str(parent_revision) not in by_revision:
                continue
            parent_status = by_revision[str(parent_revision)]
            if not isinstance(parent_status, Mapping) or parent_status.get("kind") != "runs":
                continue
            status = row.get("status")
            successor_kind = status.get("kind") if isinstance(status, Mapping) else None
            arm = "runs_to_runs" if successor_kind == "runs" else "runs_to_non_runs"
            transition: JsonObject = {
                "stable_id": str(row.get("stable_id")),
                "source": source_name,
                "row_index": row_index,
                "predecessor_revision": str(parent_revision),
                "successor_revision": str(row.get("record_revision")),
            }
            cast_arm = census[arm]
            if isinstance(cast_arm, list):
                cast_arm.append(transition)
    census["coverage_windows"] = {
        source_name: [
            int(dict(effective.get(source_name) or {}).get("rows") or 0),
            projections[source_name].rows,
        ]
        for source_name in MODELS_SOURCE_NAMES
    }
    return census


def _award_transition_coverage_baselines(
    records_root: Path,
    *,
    campaign_id: str,
    generation_id: str,
    leg_baselines: Mapping[str, Any],
) -> dict[str, JsonObject]:
    """Return the predecessor close's model extents, or genesis floors.

    Producer derivation runs before its close exists, so the latest resident
    same-campaign close is its predecessor. Replay derivation may run after
    later closes; when this generation is resident, its bound predecessor ID
    selects the same floor. This closes the between-window append gap without
    making later history alter an older proof.

    Parameters
    ----------
    records_root:
        Canonical campaign records directory.
    campaign_id:
        Campaign whose close spine owns the proof.
    generation_id:
        Generation currently being derived or replayed.
    leg_baselines:
        Current generation's complete source floors. A copied or deliberately
        moved ledger can reopen below the predecessor's row extent; census
        starts from the earlier extent so the moved bytes are scanned again.

    Returns
    -------
    dict[str, dict[str, Any]]
        Complete model-source row floors.
    """

    from menagerie.crawler.close_object import close_object_path, load_close_objects

    closes = [
        row
        for row in load_close_objects([close_object_path(records_root)])
        if row.get("schema_version") != "menagerie.crawler.leg-close.v1"
        and str(row.get("campaign_id")) == campaign_id
    ]
    current = next(
        (row for row in closes if str(row.get("generation_id")) == generation_id),
        None,
    )
    predecessor: Optional[Mapping[str, Any]] = None
    if current is not None:
        predecessor_id = current.get("predecessor_close_sha256")
        predecessor = next(
            (
                row
                for row in closes
                if predecessor_id is not None
                and str(row.get("close_sha256")) == str(predecessor_id)
            ),
            None,
        )
    elif closes:
        predecessor = closes[-1]
    sources = dict(predecessor.get("sources") or {}) if predecessor is not None else {}
    return {
        name: {
            "rows": min(
                int(dict(sources.get(name) or {}).get("rows") or 0),
                int(dict(leg_baselines.get(name) or {}).get("rows") or 0),
            )
        }
        for name in MODELS_SOURCE_NAMES
    }


class BackfillEraError(PartitionProofError):
    """Raised when the backfill mint path targets a rung outside its era.

    The backfill exists ONLY for legs that terminally closed before the leg
    machinery -- a rung the records do not prove terminally closed and
    machinery-free gets a typed refusal, never a licensing close (rt4 break 1).
    """


def leg_proofs_dir(records_root: Path) -> Path:
    """Return the content-addressed proof store below one records root."""

    return records_root.joinpath(*LEG_PROOFS_RELATIVE_PATH)


def _proof_path(records_root: Path, proof_sha256: str) -> Path:
    return leg_proofs_dir(records_root) / (str(proof_sha256).removeprefix("sha256:") + ".json")


def write_partition_proof(records_root: Path, proof: Mapping[str, Any]) -> Path:
    """Persist one proof immutably under its own content address."""

    body = {key: value for key, value in proof.items() if key != "partition_proof_sha256"}
    digest = stable_hash(body)
    if proof.get("partition_proof_sha256") not in (None, digest):
        raise PartitionProofError("partition proof digest does not match its payload at write time")
    payload = canonical_json_bytes({**body, "partition_proof_sha256": digest}) + b"\n"
    path = _proof_path(records_root, digest)
    if path.is_file():
        if path.read_bytes() != payload:
            raise PartitionProofError(f"proof store already holds different bytes for {digest}")
        return path
    atomic_replace_bytes(path, payload)
    return path


#: The active proof READ EPOCH, or ``None`` outside one. An epoch is a single
#: derivation's explicitly scoped consistency window: the first load of a given
#: proof inside it does the full residency + content-address verification on the
#: real bytes, and later loads of the SAME address in the same derivation reuse
#: that verified parse instead of re-reading and re-hashing an O(#members)
#: document once per member.
#:
#: Nothing is trusted here that the pre-fix code did not already assume. The
#: verification still happens, on real bytes, inside every derivation; what is
#: removed is only the per-member REPETITION of it, which made the carry
#: resolver O(#members) x O(#closes) whole-proof reads and hashes -- the ladder's
#: single largest quadratic term. Outside an epoch every call verifies from
#: bytes, so no state survives a derivation.
_PROOF_EPOCH: Optional[dict[str, JsonObject]] = None


def reset_partition_proof_cache() -> None:
    """Abandon any active proof read epoch (tests, and error recovery)."""

    global _PROOF_EPOCH  # noqa: PLW0603 -- module-scoped epoch, one owner
    _PROOF_EPOCH = None


@contextmanager
def proof_read_epoch() -> Iterator[None]:
    """Open one derivation's proof read epoch, verifying each proof once.

    Re-entrant: a nested epoch joins the outer one, so a caller can open one
    without knowing whether its own caller already did.
    """

    global _PROOF_EPOCH  # noqa: PLW0603 -- module-scoped epoch, one owner
    if _PROOF_EPOCH is not None:
        yield
        return
    _PROOF_EPOCH = {}
    try:
        yield
    finally:
        _PROOF_EPOCH = None


def load_partition_proof(records_root: Path, proof_sha256: str) -> JsonObject:
    """Load one proof by content address, verifying the address on the way in.

    Round-2 T4 (scale): inside a :func:`proof_read_epoch` the verified parse of
    one address is reused for the rest of that derivation. The carry resolver
    asks this question once per CARRIED member per candidate ancestor close, and
    every answer used to re-read the whole proof and re-derive its content
    address -- an O(#members) canonical re-serialization inside an
    O(#members) x O(#closes) loop, which is O(#members**2) work for one rung.
    Residency and the content address are still proven from the bytes on disk;
    only the repetition is gone, and outside an epoch nothing is reused at all.

    Raises
    ------
    PartitionProofError
        If the proof is absent, unreadable, or its bytes do not reproduce the
        requested digest.
    """

    epoch = _PROOF_EPOCH
    if epoch is not None:
        held = epoch.get(str(proof_sha256))
        if held is not None:
            return dict(held)
    path = _proof_path(records_root, proof_sha256)
    if not path.is_file():
        raise PartitionProofError(
            f"partition proof {proof_sha256} is not resident at {path}: a close "
            "binding an absent proof licenses nothing"
        )
    try:
        proof = json.loads(path.read_bytes())
    except ValueError as exc:
        raise PartitionProofError(f"partition proof {path} is unreadable: {exc}") from exc
    if not isinstance(proof, Mapping):
        raise PartitionProofError(f"partition proof {path} is not an object")
    body = {key: value for key, value in proof.items() if key != "partition_proof_sha256"}
    if stable_hash(body) != proof_sha256 or proof.get("partition_proof_sha256") != proof_sha256:
        raise PartitionProofError(
            f"partition proof at {path} does not reproduce its content address "
            f"{proof_sha256}: the proof bytes moved since the close bound them"
        )
    if epoch is not None:
        epoch[str(proof_sha256)] = dict(proof)
    return dict(proof)


def _ancestor_chain(records_root: Path, campaign_id: str, snapshot_id: str) -> tuple[str, ...]:
    """Return the campaign's lineage chain from one snapshot back to its root.

    The chain, not "any registered rung": a sibling branch's evidence is
    foreign. The snapshot itself leads the chain so a prior GENERATION's close
    of the same rung is a valid carry authority (the honest re-close case).
    """

    from menagerie.crawler.rung_staging import load_rung_registrations

    derived_from = {
        registration.snapshot_id: registration.derived_from_snapshot_id
        for registration in load_rung_registrations(records_root)
        if registration.campaign_id == campaign_id
    }
    chain = [snapshot_id]
    cursor = snapshot_id
    while cursor in derived_from:
        cursor = derived_from[cursor]
        if cursor in chain:
            break
        chain.append(cursor)
    return tuple(chain)


def _carry_authorities(
    records_root: Path, campaign_id: str, chain: Sequence[str]
) -> dict[str, tuple[JsonObject, ...]]:
    """Return each chain snapshot's validated v2 closes, oldest first."""

    from menagerie.crawler.close_object import (
        CLOSE_OBJECT_SCHEMA_VERSION,
        close_object_path,
        load_close_objects,
    )

    by_snapshot: dict[str, list[JsonObject]] = {snapshot: [] for snapshot in chain}
    for row in load_close_objects([close_object_path(records_root)]):
        if row.get("schema_version") != CLOSE_OBJECT_SCHEMA_VERSION:
            continue
        if str(row.get("campaign_id")) != campaign_id:
            continue
        snapshot = str(row.get("intake_snapshot_id"))
        if snapshot in by_snapshot:
            by_snapshot[snapshot].append(row)
    return {snapshot: tuple(rows) for snapshot, rows in by_snapshot.items()}


def _accounting_close_for(
    records_root: Path,
    authorities: Mapping[str, tuple[JsonObject, ...]],
    snapshots: Sequence[str],
    stable_id: str,
    *,
    exclude_generation_id: Optional[str],
) -> Optional[tuple[str, JsonObject, JsonObject]]:
    """Find the latest resident close whose own proof accounts one member.

    Returns ``(snapshot_id, close, member_entry)`` or ``None``. Delegation is
    BY DIGEST: the close must still resolve its proof byte-identically, and
    the proof must name the member -- a close that no longer proves anything
    carries nothing.
    """

    fallback: Optional[tuple[str, JsonObject, JsonObject]] = None
    for snapshot in snapshots:
        # A granted re-drive can legitimately move a head and directly
        # adjudicate the new revision in a later close of the same snapshot.
        # Carry must start from that newest custody statement.  Choosing the
        # first close permanently pinned the pre-grant revision and made every
        # subsequent honest no-op close fail N1 despite the newer adjudication.
        for close in reversed(authorities.get(snapshot, ())):
            if (
                exclude_generation_id is not None
                and close.get("generation_id") == exclude_generation_id
            ):
                continue
            try:
                proof = load_partition_proof(records_root, str(close["partition_proof_sha256"]))
            except PartitionProofError:
                continue
            # One key lookup, never a copy of the whole member table: this runs
            # once per member per candidate close, so copying an O(#members)
            # mapping here was quadratic in the rung's own width.
            entry = (proof.get("members") or {}).get(stable_id)
            if isinstance(entry, Mapping):
                if str(entry.get("disposition")) == CONTINUING:
                    # RULING W7 clause 5: a continuation settles nothing, so it
                    # never OUTRANKS a real settlement deeper in the chain --
                    # exclusion is not supersession. It surfaces only when no
                    # close settled the member at all (the persistence cell).
                    if fallback is None:
                        fallback = (snapshot, close, dict(entry))
                    continue
                return snapshot, close, dict(entry)
    return fallback


def adjudicated_here(proof: Mapping[str, Any]) -> frozenset[str]:
    """Return members directly adjudicated by this proof's own window or era."""

    return frozenset(
        str(member)
        for member, entry in dict(proof.get("members") or {}).items()
        if dict(entry).get("disposition") == ADJUDICATED
    )


def withheld_here(proof: Mapping[str, Any]) -> frozenset[str]:
    """Return members directly withheld by this proof's own window or era."""

    return frozenset(
        str(member)
        for member, entry in dict(proof.get("members") or {}).items()
        if dict(entry).get("disposition") == WITHHELD
    )


def attributed_here(proof: Mapping[str, Any]) -> frozenset[str]:
    """Return members directly settled by a backfill operator authority."""

    return frozenset(
        str(member)
        for member, entry in dict(proof.get("members") or {}).items()
        if dict(entry).get("disposition") == OPERATOR_ATTRIBUTED
    )


def carried(proof: Mapping[str, Any]) -> frozenset[str]:
    """Return members whose resolution is inherited from an origin close."""

    return frozenset(
        str(member)
        for member, entry in dict(proof.get("members") or {}).items()
        if dict(entry).get("disposition") == CARRIED
    )


def continuing_here(proof: Mapping[str, Any]) -> frozenset[str]:
    """Return members whose only in-window evidence is a routing marker.

    RULING W7 clause 5 (OPEN-1 default): a CONTINUING member is a disclosed
    continuation, EXCLUDED from ``settled_here``, ``proved_here``, origins, and
    introduced-member coverage. A row that says "no verdict was rendered"
    proves nothing and licenses nothing.
    """

    return frozenset(
        str(member)
        for member, entry in dict(proof.get("members") or {}).items()
        if dict(entry).get("disposition") == CONTINUING
    )


def settled_here(proof: Mapping[str, Any]) -> frozenset[str]:
    """Return the union of this proof's three direct settlement dispositions.

    CONTINUING is deliberately absent (RULING W7 clause 5, OPEN-1 default): a
    continuation settles nothing.
    """

    return adjudicated_here(proof) | withheld_here(proof) | attributed_here(proof)


def _entry_origin(
    proof: Mapping[str, Any], entry: Mapping[str, Any], close_sha256: str
) -> Optional[JsonObject]:
    """Resolve one proof entry to its direct origin adjudication."""

    disposition = entry.get("disposition")
    evidence = dict(entry.get("evidence") or {})
    if disposition == CARRIED:
        origin_value = evidence.get("origin")
        return dict(origin_value) if isinstance(origin_value, Mapping) else None
    if disposition not in (ADJUDICATED, WITHHELD, OPERATOR_ATTRIBUTED):
        return None
    return {
        "snapshot_id": str(proof.get("intake_snapshot_id")),
        "generation_id": str(proof.get("generation_id")),
        "close_sha256": close_sha256,
        "disposition": str(disposition),
        "record_revision": evidence.get("record_revision"),
        "row_sha256": evidence.get("row_sha256"),
    }


def proved_here(proof: Mapping[str, Any]) -> frozenset[str]:
    """Return members whose transitive origin belongs to this rung snapshot.

    The member table is resolved ONCE (round-2 T4): copying it per carried
    member made the disclosure recomputation O(#members**2) in the rung's own
    width -- 20k members meant 400M dict copies inside one staging gate.
    """

    snapshot_id = str(proof.get("intake_snapshot_id"))
    members = proof.get("members") or {}
    proved = set(settled_here(proof))
    for member in carried(proof):
        entry = members[member]
        origin_value = (entry.get("evidence") or {}).get("origin")
        if isinstance(origin_value, Mapping) and origin_value.get("snapshot_id") == snapshot_id:
            proved.add(member)
    return frozenset(proved)


#: D-27: withholding kinds (row DISPOSITIONS -- the ``kind`` a sealed proof's
#: evidence records) whose liveness is release/void-only, so the preflight
#: can adjudicate them at leg start with zero window ambiguity. Ordinary
#: valve kinds are deliberately OUTSIDE this set: their supersession is
#: window-aware (BL-8), so only the close's own windowed derivation may
#: judge them -- a preflight that guessed would mint false refusals.
_PARK_AUTHORITY_KINDS = frozenset(
    {
        "review-held:supersession-pending-grant",
        "review-held:award-supersession-pending-grant",
        "review-held:terminal-recording-refused",
        "deferred:scheduler-fault-quarantined",
    }
)


def carried_authority_preflight(
    records_root: Path,
    *,
    campaign_id: str,
    selectable_member_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """D-27 (LC-16): verify every carried authority is still live, PRE-SPEND.

    The close-side law already refuses at build when a predecessor close's
    withheld/carried member lost its exact origin row (a released or voided
    park carries nothing) -- but that refusal fires at leg END, after the
    spend. Twelve 200a carries ride live park rows; ONE well-meant void
    bricks the next close. This preflight moves detection to leg START by
    re-deriving, through the SAME live-withholding derivation the sealed
    proof uses, that each cited origin row is still live.

    Scope, fail-safe by construction: ROW-backed CARRIED members (headless --
    a live withholding row IS the authority; head-backed carries ride the
    immutable record and have nothing voidable), plus WITHHELD/CONTINUING
    members whose cited row kind is a park or scheduler marker
    (release/void-only liveness). Ordinary-valve WITHHELD members are
    excluded: their supersession is window-aware (BL-8) and belongs to the
    close's own derivation. Members in ``selectable_member_ids`` are also
    excluded: voiding a park over a member THIS leg can re-drive is the
    sanctioned operator drain (WS-1 clause 5 -- the void exists to re-admit
    the member), not a brick; the close's own law remains the final authority
    if the leg then fails to resolve it. Disposition-negative: findings can
    only refuse a leg start, never upgrade, settle, or license anything.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id:
        Campaign whose newest close's carried authorities are verified.
    selectable_member_ids:
        Members the opening leg's intake can select and re-drive.

    Returns
    -------
    tuple[str, ...]
        One finding per broken carry; empty when every authority is live.
    """

    from menagerie.crawler.close_object import (  # noqa: PLC0415 -- lazy: breaks the proof<->close import cycle
        close_object_path,
        load_close_objects,
    )

    close_path = close_object_path(records_root)
    if not close_path.is_file():
        return ()
    newest: Optional[Mapping[str, Any]] = None
    for row in load_close_objects([close_path]):
        if str(row.get("campaign_id")) == campaign_id:
            newest = row
    if newest is None:
        return ()
    if newest.get("schema_version") == "menagerie.crawler.leg-close.v1":
        # A v1 close predates sealed proofs entirely; there is no carried
        # authority to verify (and nothing it could license anyway).
        return ()
    try:
        proof = load_partition_proof(records_root, str(newest["partition_proof_sha256"]))
    except (KeyError, ValueError) as exc:
        return (f"the predecessor close's sealed proof cannot load: {exc}",)
    members = dict(proof.get("members") or {})
    selectable = frozenset(str(value) for value in selectable_member_ids)
    cited: dict[str, tuple[str, Optional[str]]] = {}
    findings: list[str] = []
    for stable_id, raw_entry in members.items():
        if str(stable_id) in selectable:
            continue
        entry = dict(raw_entry)
        disposition = str(entry.get("disposition"))
        evidence = dict(entry.get("evidence") or {})
        if disposition == CARRIED:
            origin = dict(evidence.get("origin") or {})
            row_sha = origin.get("row_sha256")
            # A head-backed carry (origin pinned by record_revision, no row)
            # rides the immutable record -- nothing voidable to preflight.
            # A HEAD-RIDING withheld carry (evidence binds the resident head;
            # the origin's ordinary-valve row is head-superseded, BL-8's
            # aftermath) likewise rides the record: its witness is the sealed
            # origin window plus the immutable head, which the close law
            # re-derives itself, and there is no live row whose void could
            # brick the close.  Only ROW-backed carries (headless: a live
            # withholding row IS the authority) are adjudicated here.
            if isinstance(row_sha, str) and not isinstance(evidence.get("head"), Mapping):
                cited[str(stable_id)] = (row_sha, disposition)
        elif disposition in (WITHHELD, CONTINUING):
            row_sha = evidence.get("row_sha256")
            kind = str(evidence.get("kind"))
            if isinstance(row_sha, str) and kind in _PARK_AUTHORITY_KINDS:
                cited[str(stable_id)] = (row_sha, disposition)
    if not cited:
        return tuple(findings)
    heads = model_head_citations(records_root, require_ledger=False)
    live = _live_withholding_rows(
        records_root, campaign_id, heads, tolerate_pilot_campaign=False
    )
    for stable_id in sorted(cited):
        row_sha, disposition = cited[stable_id]
        rows = live.get(stable_id) or []
        if not any(str(row.get("row_sha256")) == row_sha for row in rows):
            findings.append(
                f"{stable_id}: the predecessor close's {disposition} authority "
                f"({row_sha}) is no longer live -- a released or voided row "
                "carries nothing, and the next close will refuse at build. "
                "Dispose carries by RELEASE, never VOID (B2-X-O10)"
            )
    return tuple(findings)


def _recorded_snapshot_of(record: Mapping[str, Any]) -> Optional[str]:
    """Return the intake snapshot one raw head was recorded under, if declared."""

    vector = record.get("dependency_vector")
    if not isinstance(vector, Mapping):
        return None
    recorded = vector.get("intake_snapshot_id")
    return recorded if isinstance(recorded, str) else None


def _window_bounds(
    baselines: Mapping[str, Any], closing: Mapping[str, Any]
) -> dict[str, tuple[int, int]]:
    """Return each source's half-open row window ``[floor, close)``."""

    bounds: dict[str, tuple[int, int]] = {}
    for name in CLOSE_SOURCE_NAMES:
        floor = int(dict(baselines.get(name) or {}).get("rows") or 0)
        close_rows = int(dict(closing.get(name) or {}).get("rows") or 0)
        bounds[name] = (floor, close_rows)
    return bounds


def _live_withholding_rows(
    records_root: Path,
    campaign_id: str,
    heads: Mapping[str, Mapping[str, Any]],
    *,
    tolerate_pilot_campaign: bool,
    bounds: Optional[Mapping[str, tuple[int, int]]] = None,
) -> dict[str, list[JsonObject]]:
    """Return each member's LIVE withholding rows with exact citations.

    Live means: valid per the shared census gauntlet, unvoided (parks),
    unreleased against the member's raw head (parks), and unsuperseded by a
    recorded head (ordinary valves). Rows are returned with their source name
    and row index so citations are exact.

    BL-8: head supersession of an ordinary valve row is WINDOW-AWARE when
    ``bounds`` is supplied. A member admitted WITH a raw head (a granted
    re-drive) whose re-drive ends PARKED mints its valve row strictly after
    that head; discarding the row because "a head exists" erased the leg's
    only truthful accounting and refused the close on a phantom hole. An
    IN-WINDOW valve row over a PRE-window head therefore stays live; every
    other head-coexisting ordinary valve row remains superseded exactly as
    before, and an in-window head beside an in-window row still refuses
    loudly in the resolver (the leg contradicted itself).
    """

    from menagerie.crawler.grant_parks import (
        current_grant_park_rows,
        grant_park_released,
    )
    from menagerie.crawler.refusal_parks import (
        current_refusal_park_rows,
        refusal_park_released,
    )
    from menagerie.crawler.status import machine_deferral_rejection
    from menagerie.crawler.scheduler_parks import current_scheduler_quarantine_rows

    sources = close_ledger_sources(records_root)
    live: dict[str, list[JsonObject]] = {}
    for name in WITHHOLDING_SOURCE_NAMES:
        decoded = _valve_rows(sources[name])
        current_park_digests: Optional[frozenset[str]] = None
        if name == "park.refusal":
            current_park_digests = frozenset(
                str(row.get("row_sha256"))
                for row in current_refusal_park_rows([row for _, row in decoded])
            )
        elif name == "park.grant":
            current_park_digests = frozenset(
                str(row.get("row_sha256"))
                for row in current_grant_park_rows([row for _, row in decoded])
            )
        elif name == "valve.scheduler-quarantine":
            current_park_digests = frozenset(
                str(row.get("row_sha256"))
                for row in current_scheduler_quarantine_rows(
                    [row for _, row in decoded]
                )
            )
        for index, row in decoded:
            stable_id = row.get("stable_id")
            if not isinstance(stable_id, str) or not stable_id:
                continue
            row_campaign = str(row.get("campaign_id"))
            if row_campaign != campaign_id and not (
                tolerate_pilot_campaign and row_campaign == _PILOT_ABSENT_CAMPAIGN
            ):
                continue
            if machine_deferral_rejection(stable_id, row) is not None:
                continue
            head_entry = heads.get(stable_id)
            head = None if head_entry is None else head_entry.get("record")
            if name == "park.refusal":
                if row.get("row_sha256") not in (current_park_digests or frozenset()):
                    continue
                if refusal_park_released(row, head):
                    continue
            elif name == "park.grant":
                if row.get("row_sha256") not in (current_park_digests or frozenset()):
                    continue
                if grant_park_released(row, head):
                    continue
            elif name == "valve.scheduler-quarantine":
                if row.get("row_sha256") not in (current_park_digests or frozenset()):
                    continue
            elif head is not None:
                # An ordinary valve is superseded once a head landed: the
                # record wins, and the valve keeps only its disclosure half.
                # EXCEPT (BL-8) an in-window row over a pre-window head: the
                # re-drive this leg admitted ended parked, and that row is
                # the leg's only truthful accounting for the member.
                row_in_window = False
                head_in_window = True
                if bounds is not None:
                    floor, close_rows = bounds[name]
                    row_in_window = floor <= index < close_rows
                    if head_entry is not None:
                        head_floor, head_close = bounds[str(head_entry["source"])]
                        head_in_window = (
                            head_floor <= int(head_entry["row_index"]) < head_close
                        )
                if not (row_in_window and not head_in_window):
                    continue
            entry: JsonObject = {
                "source": name,
                "row_index": index,
                "row_sha256": str(row.get("row_sha256")),
                "kind": str(row.get("disposition")),
                "campaign_id": row_campaign,
                "run_id": row.get("run_id"),
            }
            if name == "valve.scheduler-quarantine":
                # The clause-1 kind partition classifies quarantine rows by
                # reason; carry the reason so the resolver never re-reads the
                # ledger per member (D-21 read-substrate interaction).
                quarantine = row.get("scheduler_quarantine")
                if isinstance(quarantine, Mapping):
                    entry["reason_code"] = quarantine.get("reason_code")
            live.setdefault(stable_id, []).append(entry)
    return live


def _valve_rows(path: Path) -> tuple[tuple[int, JsonObject], ...]:
    """Return one valve/park ledger's lenient rows with their row indices.

    A withholding row stays LIVE only against the whole ledger (a void or a
    release may sit anywhere after it), so this read is history-wide by
    definition. What it no longer does is re-decode that history on every
    close: the decode is held per observed ledger state, and an unreadable row
    still marks nothing.
    """

    from menagerie.crawler.leg_generation import _recover_census_tail  # noqa: PLC2701

    if not path.is_file():
        return ()

    def extend(
        value: tuple[int, tuple[tuple[int, JsonObject], ...]], rows: Iterable[LedgerRow]
    ) -> tuple[int, tuple[tuple[int, JsonObject], ...]]:
        # The row index counts EVERY non-empty row, decodable or not, because
        # that is the index every citation carries; only decodable mappings are
        # retained.
        index, decoded = value
        appended: list[tuple[int, JsonObject]] = []
        for _number, _offset, line in rows:
            try:
                row = json.loads(line)
            except ValueError:
                index += 1
                continue
            if isinstance(row, Mapping):
                appended.append((index, dict(row)))
            index += 1
        return (index, (*decoded, *appended))

    held: tuple[int, tuple[tuple[int, JsonObject], ...]] = derive(
        path,
        projection="valve.rows",
        seed=lambda: (0, ()),
        extend=extend,
        recover=_recover_census_tail,
        # Every decodable row of a valve/park ledger, retained with its index.
        weigh=lambda value, byte_count: (
            PROJECTION_BASE_WEIGHT
            + DECODED_ROW_WEIGHT_FACTOR * byte_count
            + LINE_SPAN_WEIGHT * len(value[1])
        ),
    )
    return held[1]


def _model_row_run_id(record: Mapping[str, Any]) -> Optional[str]:
    """Return the run id carried by one model row's logical provenance."""

    direct = record.get("run_id")
    if isinstance(direct, str) and direct:
        return direct
    provenance = record.get("provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("producer_run_id")
        if isinstance(value, str) and value:
            return value
    return None


def _attempted_member_ids(
    records_root: Path, bounds: Mapping[str, tuple[int, int]]
) -> frozenset[str]:
    """Return every stable id an IN-WINDOW attempt row names.

    The anti-laundering signal: a member the leg worked cannot be carried.
    Read leniently -- an unreadable attempt row marks nothing, and marking is
    the only power this census has (it can only forbid a carry).
    """

    sources = close_ledger_sources(records_root)
    attempted: set[str] = set()
    for name in ATTEMPT_SOURCE_NAMES:
        floor, close_rows = bounds[name]
        # Generation-bound: the window's own rows are read by seeking to row
        # ``floor``'s byte offset, never by materializing the campaign's whole
        # attempt history in order to slice its tail (round-2 T4 / B2).
        for payload in source_row_window(sources[name], floor, close_rows):
            try:
                row = json.loads(payload)
            except ValueError:
                continue
            if isinstance(row, Mapping):
                stable_id = row.get("stable_id")
                if isinstance(stable_id, str) and stable_id:
                    attempted.add(stable_id)
    return frozenset(attempted)


#: The custody sources of RULING W7 clause 3: an in-window row in any of these
#: naming a member is custody evidence -- an artifact transition/publication or
#: a gate verdict -- that contradicts a routing marker's "no verdict was
#: rendered" claim. Live park rows are the third custody signal and come from
#: the live-withholding census. Model sources are deliberately absent: the
#: adjudicating head is the custody a marker defers TO, and every other head
#: shape refuses through its own independent conjunct (T8).
_CUSTODY_SOURCE_NAMES = ("gates.shard", "gates.flat", "artifacts.shard", "artifacts.flat")


def _custody_index(
    records_root: Path, bounds: Mapping[str, tuple[int, int]]
) -> dict[str, tuple[JsonObject, ...]]:
    """Derive each member's in-window custody citations, ONCE per close.

    Replay-derived from bytes the producer does not control (RULING W7
    clause 3, O R-3 -- "the attacker must make the WORLD silent, not the row
    quiet"): the gates and artifacts windows are scanned once per derivation
    and indexed by member, never re-read per member (the D-21 read-substrate
    interaction O's remainder 1 measured; this IS the per-window custody
    index fallback, built in from the start). Only decodable rows naming a
    ``stable_id`` are custody evidence; an unreadable row attributes to no
    member, exactly like the attempt census (T18 fails toward the member's
    own hole refusals, never toward exclusion).
    """

    sources = close_ledger_sources(records_root)
    custody: dict[str, list[JsonObject]] = {}
    for name in _CUSTODY_SOURCE_NAMES:
        floor, close_rows = bounds[name]
        for offset, payload in enumerate(source_row_window(sources[name], floor, close_rows)):
            try:
                row = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(row, Mapping):
                continue
            if name.startswith("artifacts.") and (
                row.get("event_kind") == "staged-private"
            ):
                # Staging is PRE-custody by the artifact state machine's own
                # frozen vocabulary: every member's inputs stage before any
                # authoring, and AT-5's whole point is parking BEFORE the
                # chain is spent. Custody begins at the first TRANSITION
                # (terminal/publication authorization, commit, publish); an
                # unknown or absent event kind fails toward custody, never
                # toward exclusion.
                continue
            stable_id = row.get("stable_id")
            if isinstance(stable_id, str) and stable_id:
                custody.setdefault(stable_id, []).append(
                    {"source": name, "row_index": floor + offset}
                )
    return {member: tuple(rows) for member, rows in custody.items()}


def _member_custody_facts(
    stable_id: str,
    custody: Mapping[str, Sequence[Mapping[str, Any]]],
    live_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return every custody fact contradicting a routing marker's exclusion.

    Clause 3's conjunct, spelled as citations: in-window gate/artifact rows
    naming the member, plus any LIVE park.grant or park.refusal row (a live
    park is a standing custody statement whatever window minted it).
    """

    facts = [
        f"{item['source']}[{item['row_index']}]" for item in custody.get(stable_id, ())
    ]
    for row in live_rows:
        if str(row.get("source")) in ("park.grant", "park.refusal"):
            facts.append(f"live {row['source']}[{row['row_index']}]")
    return tuple(facts)


def _marker_settlement_kind(row: Mapping[str, Any]) -> Optional[str]:
    """Classify one live withholding citation under the clause-1 registry.

    Returns ``"routing"``, ``"verdict-bearing"``, or ``"ambiguous"`` for a
    scheduler-quarantine row, and ``None`` for every other withholding source
    (parks are always verdict-bearing and ordinary valves keep their own
    source-registry semantics untouched -- OPEN-4 default). An unclassifiable
    reason refuses fail-closed through the registry's own typed error.
    """

    if str(row.get("source")) != "valve.scheduler-quarantine":
        return None
    from menagerie.crawler.scheduler_parks import quarantine_reason_kind

    return quarantine_reason_kind(str(row.get("reason_code")))


def _marker_citation(row: Mapping[str, Any]) -> JsonObject:
    """Return one routing marker's disclosure citation (clause 2's entry)."""

    return {
        "source": str(row["source"]),
        "row_index": int(row["row_index"]),
        "row_sha256": str(row["row_sha256"]),
        "reason_code": str(row.get("reason_code")),
    }


def derive_partition_proof(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str],
    close_kind: str,
    generation_id: str,
    baselines: Optional[Mapping[str, Any]],
    member_ids: Sequence[str],
    selected_ids: Sequence[str] = (),
    operator_attributions: Optional[Mapping[str, Any]] = None,
    proof_schema_version: str = PARTITION_PROOF_SCHEMA_VERSION,
) -> JsonObject:
    """Derive the canonical partition proof from pinned bytes alone.

    THE one derivation (I-PROOF): the producer computes and seals it, the
    staging gate recomputes it bitwise over the same frozen bytes. Nothing a
    producer writes feeds it -- membership comes from the snapshot, evidence
    from the census sources, carry authority from resident ancestor closes.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id, intake_snapshot_id, intake_snapshot_sha256, phase:
        Exactly which rung adjudication is being proven.
    close_kind:
        ``leg`` (window predicates) or ``backfill`` (era-honest predicates).
    generation_id:
        The leg generation being sealed.
    baselines:
        The generation's per-source floor identities; required for ``leg``,
        forbidden for ``backfill``.
    member_ids:
        The membership to partition (the snapshot's own stable ids).
    selected_ids:
        Members the WS-1 selection plan admitted to work this leg; a selected
        member may not be closed as carried.
    operator_attributions:
        Backfill-only operator authorities, two closed shapes: ``stable_id ->
        exact current record_revision`` for a pre-machinery head lacking
        recorded-under, or ``stable_id -> {"recorded_under": snapshot_id,
        "record_revision": revision}`` for a head recorded-under a NAMED
        non-chain snapshot (the pilot's re-adjudicated history).
    Returns
    -------
    dict[str, Any]
        Sealed proof carrying ``partition_proof_sha256``.

    Raises
    ------
    PartitionProofError
        If any member cannot be resolved by an exact citation -- the error
        names every unresolved member and why.
    """

    # ONE derivation is ONE decision, so it is also ONE close-read epoch
    # (round-3 B1): the census sources, the models projections, the Track A
    # chains, and the window floors it consults are proven from their bytes once
    # each here instead of once per close per source.
    with read_epoch():
        return _derive_partition_proof(
            records_root,
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            intake_snapshot_sha256=intake_snapshot_sha256,
            phase=phase,
            close_kind=close_kind,
            generation_id=generation_id,
            baselines=baselines,
            member_ids=member_ids,
            selected_ids=selected_ids,
            operator_attributions=operator_attributions,
            proof_schema_version=proof_schema_version,
        )


def _derive_partition_proof(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str],
    close_kind: str,
    generation_id: str,
    baselines: Optional[Mapping[str, Any]],
    member_ids: Sequence[str],
    selected_ids: Sequence[str] = (),
    operator_attributions: Optional[Mapping[str, Any]] = None,
    proof_schema_version: str = PARTITION_PROOF_SCHEMA_VERSION,
) -> JsonObject:
    """Derive one sealed proof inside the caller's open close-read epoch."""

    if proof_schema_version not in PARTITION_PROOF_SCHEMA_VERSIONS:
        raise PartitionProofError(
            f"unknown partition-proof era {proof_schema_version!r}: the era is a "
            "closed set, and an unknown one refuses fail-closed (E13)"
        )
    if close_kind not in ("leg", "backfill"):
        raise PartitionProofError(f"unknown close kind {close_kind!r}")
    if close_kind == "leg" and baselines is None:
        raise PartitionProofError("a leg proof requires its generation baselines")
    if close_kind == "backfill" and baselines is not None:
        raise PartitionProofError("a backfill proof has no window to baseline")
    attributions = dict(operator_attributions or {})
    if attributions and close_kind != "backfill":
        raise PartitionProofError(
            "operator attributions exist only for the backfill era; a live leg "
            "proves every member from its own window"
        )
    for stable_id, authority in attributions.items():
        if isinstance(authority, str) and authority:
            continue
        if (
            isinstance(authority, Mapping)
            and set(authority) == {"recorded_under", "record_revision"}
            and all(isinstance(authority[key], str) and authority[key] for key in authority)
        ):
            continue
        raise PartitionProofError(
            f"operator attribution for {stable_id} has an unknown shape: an "
            "authority is either the exact current record_revision (a "
            "pre-machinery head declaring no recorded-under snapshot) or "
            "{'recorded_under': <snapshot_id>, 'record_revision': <revision>} "
            "(a head recorded-under a NAMED non-chain snapshot); the closed "
            "shapes are the disclosure the close carries"
        )

    closing = census_identities(records_root)
    member_reasons: dict[str, str] = {}

    selection_plan_sha256: Optional[str] = None
    generation_runs: frozenset[str] = frozenset()
    if close_kind == "leg":
        # Sol's materialized genesis makes presence equality correct for the
        # complete 15-slot universe: honest first appends no longer flip a slot.
        for name in CLOSE_SOURCE_NAMES:
            base = dict((baselines or {}).get(name) or {})
            now = closing[name]
            if bool(base.get("present")) != bool(now.get("present")):
                raise PartitionProofError(
                    f"source {name} changed presence during the leg (open: "
                    f"present={base.get('present')}, close: present="
                    f"{now.get('present')}): all source genesi are materialized "
                    "before the generation opens, so any presence flip is a "
                    "ledger replacement"
                )
        from menagerie.crawler.leg_generation import window_floor_refusals

        floor_findings = window_floor_refusals(records_root, baselines or {})
        if floor_findings:
            raise PartitionProofError(
                "the leg window floor no longer verifies: " + "; ".join(floor_findings)
            )
        bounds = _window_bounds(baselines or {}, closing)
        from menagerie.crawler.leg_generation import (
            generation_run_ids,
            generation_withholding_receipts,
            sealed_selection_plan,
        )

        generation_runs = generation_run_ids(records_root, generation_id)
        if not generation_runs:
            raise PartitionProofError(
                f"leg generation {generation_id} has no durable leg_run event"
            )
        sealed_selected, selection_plan_sha256 = sealed_selection_plan(
            records_root, generation_id
        )
        supplied_selected = frozenset(str(value) for value in selected_ids)
        if supplied_selected and supplied_selected != sealed_selected:
            raise PartitionProofError(
                "the producer-selected ids differ from the sealed WS-1 plan; "
                "only the replay-derived plan may steer R9"
            )
        selected = sealed_selected
        resume_withholding = generation_withholding_receipts(records_root, generation_id)
    else:
        bounds = {name: (0, 0) for name in CLOSE_SOURCE_NAMES}
        selected = frozenset()
        resume_withholding = {}

    try:
        heads = model_head_citations(records_root, require_ledger=False)
    except LegGenerationError as exc:
        raise PartitionProofError(str(exc)) from exc

    chain = _ancestor_chain(records_root, campaign_id, intake_snapshot_id)
    authorities = _carry_authorities(records_root, campaign_id, chain)
    live_withholding = _live_withholding_rows(
        records_root,
        campaign_id,
        heads,
        tolerate_pilot_campaign=close_kind == "backfill",
        bounds=bounds,
    )
    attempted = _attempted_member_ids(records_root, bounds) if close_kind == "leg" else frozenset()
    # RULING W7 clause 3: custody silence is replay-derived over the pinned
    # CLOSE_SOURCE_NAMES windows, indexed ONCE per derivation (never a
    # per-member ledger read -- the D-21 hot-path interaction). Backfill
    # closes have no window floor, so their custody scan covers the whole
    # observed extent (clause 10's era totality).
    if proof_schema_version == PARTITION_PROOF_SCHEMA_VERSION_V3:
        custody: Mapping[str, tuple[JsonObject, ...]] = {}
    elif close_kind == "leg":
        custody = _custody_index(records_root, bounds)
    else:
        custody = _custody_index(
            records_root,
            {
                name: (0, int(closing[name].get("rows") or 0))
                for name in CLOSE_SOURCE_NAMES
            },
        )

    members: dict[str, JsonObject] = {}
    # One proof read epoch for the whole membership: each candidate ancestor
    # proof is read and content-address-verified once for this derivation, not
    # once per carried member (round-2 T4).
    with proof_read_epoch():
        for stable_id in sorted(str(value) for value in member_ids):
            entry = _resolve_member(
                records_root,
                stable_id,
                close_kind=close_kind,
                campaign_id=campaign_id,
                intake_snapshot_id=intake_snapshot_id,
                generation_id=generation_id,
                heads=heads,
                bounds=bounds,
                chain=chain,
                authorities=authorities,
                live_withholding=live_withholding,
                attempted=attempted,
                selected=selected,
                resume_withholding=resume_withholding,
                attributions=attributions,
                member_reasons=member_reasons,
                generation_runs=generation_runs,
                proof_schema_version=proof_schema_version,
                custody=custody,
            )
            if entry is not None:
                members[stable_id] = entry

    consumed_attributions = {
        stable_id
        for stable_id, entry in members.items()
        if entry["disposition"] == OPERATOR_ATTRIBUTED
    }
    for stable_id in sorted(set(attributions) - consumed_attributions):
        # A member the resolver already refused keeps its more specific
        # reason; the unconsumed-authority message covers only members whose
        # evidence resolved WITHOUT the authority.
        member_reasons.setdefault(
            stable_id,
            "an operator attribution names this member but no unattributable "
            "head consumed it: an authority may never override evidence",
        )

    if member_reasons:
        raise PartitionProofError(
            f"{len(member_reasons)} member(s) of {intake_snapshot_id} cannot be "
            "resolved by exact citation: "
            + "; ".join(
                f"{stable_id} ({reason})" for stable_id, reason in sorted(member_reasons.items())
            ),
            member_reasons=member_reasons,
        )

    body: JsonObject = {
        "schema_version": proof_schema_version,
        "campaign_id": campaign_id,
        "intake_snapshot_id": intake_snapshot_id,
        "intake_snapshot_sha256": intake_snapshot_sha256,
        "phase": phase,
        "close_kind": close_kind,
        "generation_id": generation_id,
        "window": (
            {name: list(bounds[name]) for name in CLOSE_SOURCE_NAMES}
            if close_kind == "leg"
            else None
        ),
        "selection_plan_sha256": selection_plan_sha256,
        "award_transition_census": award_transition_census(
            records_root,
            baselines,
            coverage_baselines=(
                _award_transition_coverage_baselines(
                    records_root,
                    campaign_id=campaign_id,
                    generation_id=generation_id,
                    leg_baselines=baselines or {},
                )
                if close_kind == "leg"
                else None
            ),
        ),
        "members": members,
    }
    if proof_schema_version != PARTITION_PROOF_SCHEMA_VERSION_V3:
        # DX-23: the close cites the invocation-attributed legs it spans
        # (era-honest: pre-invocation-ledger runs cite with empty
        # invocation_ids). The tuple is sorted by run_id, so the close digest
        # stays deterministic. ERA-KEYED (RULING W7 clause 8): the citation
        # key joins the hashed body only from the v4 era on -- sealed v3
        # proofs predate DX-23 and must re-derive byte-identically, so the
        # v3-era derivation emits NO leg_citations key at all (canonical
        # hashing sorts keys, so conditional insertion is digest-safe).
        body["leg_citations"] = (
            [dict(citation) for citation in close_leg_citations(records_root, generation_id)]
            if close_kind == "leg"
            else None
        )
    proof = dict(body)
    proof["partition_proof_sha256"] = stable_hash(body)
    return proof


#: Withholding sources whose row liveness is release/void-tracked (the two
#: parks and the scheduler quarantine): the exact row's absence from the live
#: set means a RELEASE or VOID happened, and a released or voided row carries
#: nothing.  The remaining (ordinary valve) sources lose liveness by HEAD
#: SUPERSESSION alone, which the withheld-origin carry re-proves from sealed
#: windows instead of presuming.
_RELEASE_TRACKED_WITHHOLDING_SOURCES = frozenset(
    {"park.grant", "park.refusal", "valve.scheduler-quarantine"}
)


def _carry_from_withheld_origin(
    records_root: Path,
    stable_id: str,
    *,
    origin: JsonObject,
    ancestor_snapshot_id: str,
    ancestor_close_sha256: str,
    head_entry: Mapping[str, Any],
    head: Mapping[str, Any],
    live_rows: Sequence[Mapping[str, Any]],
    authorities: Mapping[str, tuple[JsonObject, ...]],
    member_reasons: dict[str, str],
) -> Optional[JsonObject]:
    """Carry one member whose newest settlement WITHHELD it beside its head.

    The head-backed carry's revision pin cannot adjudicate this cell: a
    withholding records no ``record_revision``, so the pin compared the real
    head against ``None`` and refused every honest no-op close that followed
    a BL-8 withheld settlement (an in-window ordinary-valve row over the
    member's own pre-window head -- the leg-19 calibration incident, 11
    checker-lane deferrals over failed terminals).  The lawful law mirrors
    the headless withheld carry, with the head's immutability re-proved from
    sealed evidence:

    * ALWAYS, the head-immutability witness first: the settling close's own
      SEALED window must show the CURRENT head row PRE-window there, so the
      withholding was rendered over exactly this resident head and the head
      cannot have moved since (a moved head lands in a later window and
      never reaches the carry arm; an unclosed-generation stray lands past
      the sealed floor and fails the pre-window conjunct).  Without this
      conjunct a LIVE row laundered unadjudicated custody movement the old
      pin refused: a void-tracked quarantine row stays live across an
      abandoned generation's stray terminal, and the row's authority must
      not ride over it;
    * the witness held and the origin's EXACT row still live -> the row is
      the authority, carry (no head binding: the D-27 preflight keeps
      adjudicating the row's liveness pre-spend);
    * the row dead on a release/void-tracked source -> a released or voided
      park carries nothing, refuse;
    * the row dead on an ordinary valve -> deadness on those sources means
      head supersession (their only liveness rule), lawful to inherit ONLY
      in BL-8's re-proved aftermath: the row was IN the settling close's
      sealed window.  The carry rides the immutable record plus the sealed
      proof and binds the resident head in its evidence for the D-27
      preflight.

    Purely a refusal-point widening: every input this function carries
    previously RAISED at proof build, and a refused proof never sealed, so
    all sealed proofs re-derive byte-identically and no era key is needed
    (RULING W7 clause 8 is satisfied vacuously, the same argument as the
    newest-first ``_accounting_close_for`` fix).
    """

    evidence: JsonObject = {
        "ancestor_snapshot_id": ancestor_snapshot_id,
        "ancestor_close_sha256": ancestor_close_sha256,
        "origin": origin,
    }
    row_sha = origin.get("row_sha256")
    if not isinstance(row_sha, str):
        member_reasons[stable_id] = "the ancestor withholding carries no exact origin row"
        return None
    origin_close = next(
        (
            candidate
            for candidate in authorities.get(str(origin.get("snapshot_id")), ())
            if str(candidate.get("close_sha256")) == str(origin.get("close_sha256"))
        ),
        None,
    )
    if origin_close is None:
        member_reasons[stable_id] = (
            f"the withheld origin's settling close ({origin.get('close_sha256')}) "
            f"is not a resident authority of {origin.get('snapshot_id')}: "
            "nothing re-proves the withholding"
        )
        return None
    origin_proof = load_partition_proof(
        records_root, str(origin_close["partition_proof_sha256"])
    )
    origin_entry = dict(dict(origin_proof.get("members") or {}).get(stable_id) or {})
    origin_evidence = dict(origin_entry.get("evidence") or {})
    if (
        str(origin_entry.get("disposition")) != WITHHELD
        or origin_evidence.get("row_sha256") != row_sha
    ):
        member_reasons[stable_id] = (
            "the settling close's sealed proof does not withhold this member "
            "on the exact origin row: nothing re-proves the withholding"
        )
        return None
    window = origin_proof.get("window")
    if not isinstance(window, Mapping):
        member_reasons[stable_id] = (
            "the withheld origin's settling close seals no window: the "
            "resident-head witness cannot be re-proved"
        )
        return None
    head_floor, _head_close = (
        int(value) for value in (window.get(str(head_entry["source"])) or (0, 0))
    )
    if int(head_entry["row_index"]) >= head_floor:
        member_reasons[stable_id] = (
            f"the head's current revision ({head['record_revision']}) sits at "
            f"or past the withheld origin's sealed window floor: the "
            "withholding was rendered over a DIFFERENT resident head, and "
            "unadjudicated custody movement never rides a carry"
        )
        return None
    if any(row.get("row_sha256") == row_sha for row in live_rows):
        # The live row IS the authority, exactly as the headless carry law.
        return {"disposition": CARRIED, "evidence": evidence}
    source = str(origin_evidence.get("source"))
    if source in _RELEASE_TRACKED_WITHHOLDING_SOURCES:
        member_reasons[stable_id] = (
            f"the withheld origin's {source} row is no longer live: a released "
            "or voided park carries nothing"
        )
        return None
    row_floor, row_close = (int(value) for value in (window.get(source) or (0, 0)))
    row_index = int(origin_evidence.get("row_index", -1))
    if not row_floor <= row_index < row_close:
        member_reasons[stable_id] = (
            "the withheld origin's valve row is no longer live and sits outside "
            "the settling close's own sealed window: head supersession (the "
            "BL-8 aftermath) cannot be presumed"
        )
        return None
    evidence["head"] = {
        "source": str(head_entry["source"]),
        "row_index": int(head_entry["row_index"]),
        "record_revision": str(head["record_revision"]),
    }
    return {"disposition": CARRIED, "evidence": evidence}


def _resolve_member(
    records_root: Path,
    stable_id: str,
    *,
    close_kind: str,
    campaign_id: str,
    intake_snapshot_id: str,
    generation_id: str,
    heads: Mapping[str, Mapping[str, Any]],
    bounds: Mapping[str, tuple[int, int]],
    chain: Sequence[str],
    authorities: Mapping[str, tuple[JsonObject, ...]],
    live_withholding: Mapping[str, Sequence[Mapping[str, Any]]],
    attempted: frozenset[str],
    selected: frozenset[str],
    resume_withholding: Mapping[str, Sequence[Mapping[str, Any]]],
    attributions: Mapping[str, Any],
    member_reasons: dict[str, str],
    generation_runs: frozenset[str],
    proof_schema_version: str = PARTITION_PROOF_SCHEMA_VERSION,
    custody: Mapping[str, Sequence[Mapping[str, Any]]] = {},
) -> Optional[JsonObject]:
    """Classify one member, or record why it is unresolved.

    The classification is ERA-KEYED (RULING W7 clause 8): proof schema v3 is
    the pre-PD-1 law, byte-identical forever; v4 applies the ruling's kind
    partition, custody-silence conjunct, and continuation classification.
    """

    v4 = proof_schema_version != PARTITION_PROOF_SCHEMA_VERSION_V3

    head_entry = heads.get(stable_id)
    head = None if head_entry is None else dict(head_entry["record"])
    recorded_under = None if head is None else _recorded_snapshot_of(head)

    if close_kind == "backfill":
        return _resolve_member_backfill(
            records_root,
            stable_id,
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            generation_id=generation_id,
            head_entry=head_entry,
            recorded_under=recorded_under,
            chain=chain,
            authorities=authorities,
            live_withholding=live_withholding,
            attributions=attributions,
            member_reasons=member_reasons,
            proof_schema_version=proof_schema_version,
            custody=custody,
        )

    in_window_head = False
    if head_entry is not None:
        floor, close_rows = bounds[str(head_entry["source"])]
        index = int(head_entry["row_index"])
        in_window_head = floor <= index < close_rows

    if in_window_head:
        assert head_entry is not None and head is not None  # in_window_head implies a head
        if recorded_under != intake_snapshot_id:
            member_reasons[stable_id] = (
                f"in-window head at {head_entry['source']}[{head_entry['row_index']}] "
                f"is recorded-under {recorded_under!r}, not this snapshot: the "
                "window and the dependency vector contradict"
            )
            return None
        run_id = _model_row_run_id(head)
        if run_id not in generation_runs:
            member_reasons[stable_id] = (
                f"in-window adjudicated citation at {head_entry['source']}"
                f"[{head_entry['row_index']}] names run {run_id!r}, which did not "
                f"join generation {generation_id}: position is not provenance"
            )
            return None
        live_rows = list(live_withholding.get(stable_id) or ())
        in_window_live = [
            row
            for row in live_rows
            if bounds[str(row["source"])][0]
            <= int(row["row_index"])
            < bounds[str(row["source"])][1]
        ]
        if in_window_live and not v4:
            member_reasons[stable_id] = (
                "holds both an in-window head and a live in-window withholding "
                f"row ({in_window_live[-1]['source']}[{in_window_live[-1]['row_index']}]): "
                "the leg contradicted itself, and a human must say which is true"
            )
            return None
        superseded_markers: list[JsonObject] = []
        if in_window_live:
            # RULING W7 clause 2: exclusion is MEMBERSHIP (the clause-1 kind
            # registry) AND custody silence (clause 3), never judgement.
            custody_facts = _member_custody_facts(stable_id, custody, live_rows)
            verdicts: list[Mapping[str, Any]] = []
            markers: list[Mapping[str, Any]] = []
            contradicted: list[Mapping[str, Any]] = []
            for row in in_window_live:
                kind = _marker_settlement_kind(row)
                if kind in ("routing", "ambiguous"):
                    if custody_facts:
                        contradicted.append(row)
                    else:
                        markers.append(row)
                else:
                    # Parks and verdict-classified quarantine reasons; a
                    # future reason explicitly classified verdict-bearing
                    # slots in here with zero predicate change (F's Q-V arm).
                    verdicts.append(row)
            if verdicts:
                cited_row = verdicts[-1]
                member_reasons[stable_id] = (
                    "head-beside-live-custody-withholding: holds an in-window "
                    "head and the live in-window verdict-bearing row "
                    f"{cited_row['source']}[{cited_row['row_index']}] "
                    f"({cited_row.get('reason_code') or cited_row['kind']}); the "
                    "custody fact stands and the close refuses (RULING W7 "
                    "clause 4); exit: a replay-derived release of the row "
                    "(remedied void / machine drain / park release), never a "
                    "bare marker"
                )
                return None
            if contradicted:
                cited_row = contradicted[-1]
                member_reasons[stable_id] = (
                    "marker-contradicts-custody: the routing-labelled row "
                    f"{cited_row['source']}[{cited_row['row_index']}] (reason "
                    f"{cited_row.get('reason_code')!r}) claims no verdict was "
                    "rendered, but the member's custody window is not silent "
                    f"({', '.join(custody_facts)}): exclusion requires replay-"
                    "derived custody silence (RULING W7 clause 3)"
                )
                return None
            superseded_markers = [_marker_citation(row) for row in markers]
        evidence: JsonObject = {
            "source": str(head_entry["source"]),
            "row_index": int(head_entry["row_index"]),
            "record_revision": str(head["record_revision"]),
            "provenance": {"run_id": run_id, "generation_id": generation_id},
        }
        if superseded_markers:
            # Clause 2's disclosure entry: the excluded routing markers ride
            # in the sealed evidence, never silently dropped.
            evidence["superseded_markers"] = superseded_markers
        return {"disposition": ADJUDICATED, "evidence": evidence}

    live_rows = list(live_withholding.get(stable_id) or ())
    in_window_live = [
        row
        for row in live_rows
        if bounds[str(row["source"])][0] <= int(row["row_index"]) < bounds[str(row["source"])][1]
        and row.get("run_id") in generation_runs
    ]
    if in_window_live:
        if not v4:
            cited = in_window_live[-1]
            return {
                "disposition": WITHHELD,
                "evidence": {
                    "source": str(cited["source"]),
                    "row_index": int(cited["row_index"]),
                    "row_sha256": str(cited["row_sha256"]),
                    "kind": str(cited["kind"]),
                    "provenance": {
                        "run_id": str(cited["run_id"]),
                        "generation_id": generation_id,
                    },
                },
            }
        # RULING W7 clauses 5 and 9: verdict-bearing rows DOMINATE and keep
        # the pre-ruling withheld law; a ROUTING marker over live custody is
        # a typed refusal; an AMBIGUOUS reason over live custody resolves
        # VERDICT-BEARING (clause 1: "verdict-bearing absent corroboration",
        # the T16 cell -- it withholds and proves, exactly as pre-ruling); a
        # marker-only member is a disclosed CONTINUATION that proves nothing
        # and licenses nothing.
        custody_facts = _member_custody_facts(stable_id, custody, live_rows)
        verdicts = []
        valves = []
        markers = []
        contradicted = []
        for row in in_window_live:
            kind = _marker_settlement_kind(row)
            if kind in ("routing", "ambiguous"):
                if not custody_facts:
                    markers.append(row)
                elif kind == "routing":
                    contradicted.append(row)
                else:
                    verdicts.append(row)
            elif kind == "verdict-bearing" or str(row["source"]) in (
                "park.grant",
                "park.refusal",
            ):
                verdicts.append(row)
            else:
                # Ordinary deferral valves keep their proving power untouched
                # (OPEN-4 default, F+S): a closed negative machine fact.
                valves.append(row)
        governing: Optional[Mapping[str, Any]] = None
        if verdicts:
            governing = verdicts[-1]
        elif contradicted:
            cited_row = contradicted[-1]
            member_reasons[stable_id] = (
                "marker-contradicts-custody: the routing-labelled row "
                f"{cited_row['source']}[{cited_row['row_index']}] (reason "
                f"{cited_row.get('reason_code')!r}) claims no verdict was "
                "rendered, but the member's custody window is not silent "
                f"({', '.join(custody_facts)}): exclusion requires replay-"
                "derived custody silence (RULING W7 clause 3)"
            )
            return None
        elif valves:
            governing = valves[-1]
        if governing is not None:
            return {
                "disposition": WITHHELD,
                "evidence": {
                    "source": str(governing["source"]),
                    "row_index": int(governing["row_index"]),
                    "row_sha256": str(governing["row_sha256"]),
                    "kind": str(governing["kind"]),
                    "provenance": {
                        "run_id": str(governing["run_id"]),
                        "generation_id": generation_id,
                    },
                },
            }
        governing = markers[-1]
        return {
            "disposition": CONTINUING,
            "evidence": {
                "source": str(governing["source"]),
                "row_index": int(governing["row_index"]),
                "row_sha256": str(governing["row_sha256"]),
                "kind": str(governing["kind"]),
                "reason_code": str(governing.get("reason_code")),
                "settlement_class": "routing-marker",
                "provenance": {
                    "run_id": str(governing["run_id"]),
                    "generation_id": generation_id,
                },
            },
        }

    # The sole R9 exemption: the selected member was short-circuited THIS
    # generation by a chained leg_withholding receipt naming one exact live
    # row. The receipt's run must have joined the generation. A stray live row
    # with no receipt, a copied digest, or a foreign run reaches the refusal
    # below and can never launder the member into CARRIED.
    receipts = [
        receipt
        for receipt in resume_withholding.get(stable_id, ())
        if receipt.get("run_id") in generation_runs
    ]
    resume_citations = [
        (row, receipt)
        for row in live_rows
        for receipt in receipts
        if receipt.get("withholding_row_sha256") == row.get("row_sha256")
    ]
    if stable_id in selected and resume_citations:
        cited, receipt = resume_citations[-1]
        disposition = WITHHELD
        extra: JsonObject = {}
        if v4 and _marker_settlement_kind(cited) in ("routing", "ambiguous"):
            custody_facts = _member_custody_facts(stable_id, custody, live_rows)
            if custody_facts and _marker_settlement_kind(cited) == "routing":
                member_reasons[stable_id] = (
                    "marker-contradicts-custody: the resume receipt cites the "
                    f"routing-labelled row {cited['source']}[{cited['row_index']}] "
                    f"(reason {cited.get('reason_code')!r}) but the member's "
                    f"custody window is not silent ({', '.join(custody_facts)}): "
                    "exclusion requires replay-derived custody silence "
                    "(RULING W7 clause 3)"
                )
                return None
            if not custody_facts:
                # A resume short-circuit onto a routing marker is a
                # continuation, not a settlement (RULING W7 clause 5): no
                # proof, no licence. An AMBIGUOUS reason over live custody
                # falls through as verdict-bearing (clause 1) and keeps the
                # pre-ruling withheld shape.
                disposition = CONTINUING
                extra = {
                    "reason_code": str(cited.get("reason_code")),
                    "settlement_class": "routing-marker",
                }
        return {
            "disposition": disposition,
            "evidence": {
                "source": str(cited["source"]),
                "row_index": int(cited["row_index"]),
                "row_sha256": str(cited["row_sha256"]),
                "kind": str(cited["kind"]),
                **extra,
                "resume_receipt": {
                    "row_sha256": str(receipt["row_sha256"]),
                    "run_id": str(receipt["run_id"]),
                    "generation_id": generation_id,
                },
            },
        }

    # CARRIED: the leg must not have touched the member at all.
    if stable_id in attempted:
        member_reasons[stable_id] = (
            "an in-window attempt row names this member and no in-window head "
            "or live withholding row accounts it: selected-and-unrecorded is a "
            "hole, not a carry"
        )
        return None
    if stable_id in selected:
        # R9's one exemption was already consumed above: an IN-WINDOW live
        # withholding row whose run joined this generation returns WITHHELD.
        # Any row left here is pre-window, foreign-run, or otherwise stray and
        # may not launder a selected member into CARRIED merely by existing.
        member_reasons[stable_id] = (
            "the WS-1 selection plan admitted this member to work this leg and "
            "nothing in-window resolves it: a selected member may not be closed "
            "as carried"
        )
        return None
    if head_entry is not None:
        assert head is not None
        if recorded_under is None or recorded_under not in chain:
            member_reasons[stable_id] = (
                f"current head is recorded-under {recorded_under!r}, which is "
                "not on this campaign's lineage chain: foreign evidence carries "
                "nothing"
            )
            return None
        accounting = _accounting_close_for(
            records_root,
            authorities,
            [recorded_under],
            stable_id,
            exclude_generation_id=generation_id,
        )
        if accounting is None:
            member_reasons[stable_id] = (
                f"head recorded-under {recorded_under} but no resident close of "
                "that rung accounts this member: nothing to carry from"
            )
            return None
        snapshot, close, ancestor_entry = accounting
        ancestor_proof = load_partition_proof(
            records_root, str(close["partition_proof_sha256"])
        )
        origin = _entry_origin(ancestor_proof, ancestor_entry, str(close["close_sha256"]))
        if origin is None:
            member_reasons[stable_id] = "the ancestor entry carries no well-formed origin"
            return None
        if origin.get("disposition") == WITHHELD:
            # The revision pin below is vacuous against a WITHHELD origin: a
            # withholding adjudicates no record revision, so it pins ``None``
            # forever and the first untouched close after a BL-8 withheld
            # settlement bricked every honest no-op close (the leg-19
            # calibration incident).  The withheld-origin carry has its own
            # closed law.
            return _carry_from_withheld_origin(
                records_root,
                stable_id,
                origin=origin,
                ancestor_snapshot_id=snapshot,
                ancestor_close_sha256=str(close["close_sha256"]),
                head_entry=head_entry,
                head=head,
                live_rows=live_rows,
                authorities=authorities,
                member_reasons=member_reasons,
            )
        if origin.get("record_revision") != str(head["record_revision"]):
            member_reasons[stable_id] = (
                f"the head moved from {origin.get('record_revision')} to "
                f"{head['record_revision']} since the origin adjudication "
                f"({origin.get('snapshot_id')}/{origin.get('generation_id')}): "
                "CARRIED pins the revision its named ancestor adjudicated"
            )
            return None
        return {
            "disposition": CARRIED,
            "evidence": {
                "ancestor_snapshot_id": snapshot,
                "ancestor_close_sha256": str(close["close_sha256"]),
                "origin": origin,
            },
        }
    # Headless carry: an ancestor resolved the member as withheld and the
    # withholding is STILL live -- a released or voided park carries nothing.
    accounting = _accounting_close_for(
        records_root,
        authorities,
        chain,
        stable_id,
        exclude_generation_id=generation_id,
    )
    if accounting is None:
        member_reasons[stable_id] = (
            "no head, no in-window withholding row, and no resident ancestor "
            "close accounts this member"
        )
        return None
    if not live_rows:
        member_reasons[stable_id] = (
            "an ancestor close accounts this member but no LIVE withholding row "
            "backs the inherited disposition: a released or voided park carries "
            "nothing"
        )
        return None
    snapshot, close, ancestor_entry = accounting
    if str(ancestor_entry.get("disposition")) == CONTINUING:
        # Continuation persistence (RULING W7 clauses 5-6): an untouched member
        # whose ancestor close left it CONTINUING stays a disclosed, live-row-
        # backed continuation -- it never blocks the close, never becomes a
        # carry (a continuation is not an origin), and never proves.
        cited_sha = str(dict(ancestor_entry.get("evidence") or {}).get("row_sha256"))
        persisted: Optional[Mapping[str, Any]] = next(
            (row for row in live_rows if str(row.get("row_sha256")) == cited_sha), None
        )
        if not v4 or persisted is None:
            member_reasons[stable_id] = (
                "an ancestor close left this member CONTINUING but its cited "
                "routing marker is no longer live: a released or voided marker "
                "carries nothing, and a continuation is never an origin"
            )
            return None
        return {
            "disposition": CONTINUING,
            "evidence": {
                "source": str(persisted["source"]),
                "row_index": int(persisted["row_index"]),
                "row_sha256": str(persisted["row_sha256"]),
                "kind": str(persisted["kind"]),
                "reason_code": str(persisted.get("reason_code")),
                "settlement_class": "routing-marker",
                "continued_from": {
                    "ancestor_snapshot_id": snapshot,
                    "ancestor_close_sha256": str(close["close_sha256"]),
                },
            },
        }
    ancestor_proof = load_partition_proof(records_root, str(close["partition_proof_sha256"]))
    origin = _entry_origin(ancestor_proof, ancestor_entry, str(close["close_sha256"]))
    if origin is None or not isinstance(origin.get("row_sha256"), str):
        member_reasons[stable_id] = "the ancestor withholding carries no exact origin row"
        return None
    if not any(row.get("row_sha256") == origin["row_sha256"] for row in live_rows):
        member_reasons[stable_id] = (
            "the live withholding row is not the exact row cited by the origin "
            f"({origin['row_sha256']}): a copy or re-issued park is new evidence"
        )
        return None
    return {
        "disposition": CARRIED,
        "evidence": {
            "ancestor_snapshot_id": snapshot,
            "ancestor_close_sha256": str(close["close_sha256"]),
            "origin": origin,
        },
    }


def _resolve_member_backfill(
    records_root: Path,
    stable_id: str,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    generation_id: str,
    head_entry: Optional[Mapping[str, Any]],
    recorded_under: Optional[str],
    chain: Sequence[str],
    authorities: Mapping[str, tuple[JsonObject, ...]],
    live_withholding: Mapping[str, Sequence[Mapping[str, Any]]],
    attributions: Mapping[str, Any],
    member_reasons: dict[str, str],
    proof_schema_version: str = PARTITION_PROOF_SCHEMA_VERSION,
    custody: Mapping[str, Sequence[Mapping[str, Any]]] = {},
) -> Optional[JsonObject]:
    """Classify one member under the era-honest backfill predicates.

    RULING W7 clause 10 (era totality): under proof schema v4 the backfill era
    obeys the same kind partition, custody-silence conjunct, and continuation
    classification as a live leg. The backfill has no generation window or run
    census (nothing pre-machinery minted them), so its custody scan covers the
    whole observed source extent -- the widest, most refusal-prone bound.
    Proof schema v3 keeps the pre-ruling law byte-identical forever.
    """

    if head_entry is not None:
        head = dict(head_entry["record"])
        revision = str(head["record_revision"])
        if recorded_under == intake_snapshot_id:
            return {
                "disposition": ADJUDICATED,
                "evidence": {
                    "source": str(head_entry["source"]),
                    "row_index": int(head_entry["row_index"]),
                    "record_revision": revision,
                },
            }
        if recorded_under is None:
            authority = attributions.get(stable_id)
            if authority is None:
                member_reasons[stable_id] = (
                    "pre-machinery head declares no recorded-under snapshot; "
                    "resolving it requires an explicit operator authority naming "
                    "the exact current revision "
                    "(--generation-authority stable_id=revision)"
                )
                return None
            if isinstance(authority, Mapping):
                member_reasons[stable_id] = (
                    "operator authority names recorded-under snapshot "
                    f"{authority.get('recorded_under')!r}, but the head declares "
                    "none: the authority does not describe this head"
                )
                return None
            if authority != revision:
                member_reasons[stable_id] = (
                    f"operator authority names revision {authority}, but the "
                    f"current head is {revision}: the head moved and the "
                    "authority died with it"
                )
                return None
            return {
                "disposition": OPERATOR_ATTRIBUTED,
                "evidence": {
                    "source": str(head_entry["source"]),
                    "row_index": int(head_entry["row_index"]),
                    "record_revision": revision,
                },
            }
        if recorded_under in chain and recorded_under != intake_snapshot_id:
            from_close = _accounting_close_for(
                records_root,
                authorities,
                [recorded_under],
                stable_id,
                exclude_generation_id=generation_id,
            )
            if from_close is None:
                member_reasons[stable_id] = (
                    f"head recorded-under {recorded_under}, not this snapshot, "
                    "and no resident close of that rung accounts this member: "
                    "honest history is admissible, fabricated history is not"
                )
                return None
            snapshot, close, ancestor_entry = from_close
            ancestor_proof = load_partition_proof(
                records_root, str(close["partition_proof_sha256"])
            )
            origin = _entry_origin(
                ancestor_proof, ancestor_entry, str(close["close_sha256"])
            )
            if origin is None or origin.get("record_revision") != revision:
                member_reasons[stable_id] = (
                    "the current head no longer equals the revision its ancestor "
                    "origin adjudicated"
                )
                return None
            return {
                "disposition": CARRIED,
                "evidence": {
                    "ancestor_snapshot_id": snapshot,
                    "ancestor_close_sha256": str(close["close_sha256"]),
                    "origin": origin,
                },
            }
        # A NAMED non-chain snapshot (rt4/OPS-10 dead-end (b)): the pilot's
        # heads were re-adjudicated AFTER their leg by later runs (the
        # mini-rung smoke, the ad-hoc head50 run), so their current heads are
        # recorded-under foreign or DESCENDANT snapshots -- real history, but
        # nothing this chain can delegate to. The only honest resolution is an
        # explicit operator authority that BINDS both facts: the exact named
        # snapshot the head is recorded-under and the exact current revision.
        # Either moving kills the authority, and the whole binding is
        # disclosed in the close and re-verified on every gate replay.
        authority = attributions.get(stable_id)
        if isinstance(authority, Mapping):
            named = str(authority.get("recorded_under"))
            claimed_revision = str(authority.get("record_revision"))
            if named != recorded_under:
                member_reasons[stable_id] = (
                    f"operator authority names recorded-under {named!r}, but the "
                    f"head is recorded-under {recorded_under!r}: an authority "
                    "binds the exact snapshot it discloses"
                )
                return None
            if claimed_revision != revision:
                member_reasons[stable_id] = (
                    f"operator authority names revision {claimed_revision}, but "
                    f"the current head is {revision}: the head moved and the "
                    "authority died with it"
                )
                return None
            return {
                "disposition": OPERATOR_ATTRIBUTED,
                "evidence": {
                    "source": str(head_entry["source"]),
                    "row_index": int(head_entry["row_index"]),
                    "record_revision": revision,
                    "recorded_under": recorded_under,
                },
            }
        if isinstance(authority, str):
            member_reasons[stable_id] = (
                f"operator authority supplies only a revision, but the head is "
                f"recorded-under the non-chain snapshot {recorded_under!r}: a "
                "non-chain attribution must NAME that snapshot "
                "(--recorded-under-authority stable_id=snapshot:revision)"
            )
            return None
        member_reasons[stable_id] = (
            f"head recorded-under {recorded_under!r}, which is neither this "
            "snapshot nor an ancestor on this campaign's lineage chain; "
            "resolving named non-chain history requires an explicit operator "
            "authority binding that snapshot and the exact current revision "
            "(--recorded-under-authority stable_id=snapshot:revision)"
        )
        return None
    live_rows = list(live_withholding.get(stable_id) or ())
    if live_rows:
        if proof_schema_version == PARTITION_PROOF_SCHEMA_VERSION_V3:
            cited = live_rows[-1]
            return {
                "disposition": WITHHELD,
                "evidence": {
                    "source": str(cited["source"]),
                    "row_index": int(cited["row_index"]),
                    "row_sha256": str(cited["row_sha256"]),
                    "kind": str(cited["kind"]),
                },
            }
        # Era totality (clause 10): the same law as the live leg's resolver --
        # routing over live custody refuses; ambiguous over live custody is
        # verdict-bearing (clause 1) and keeps the pre-ruling withheld shape.
        custody_facts = _member_custody_facts(stable_id, custody, live_rows)
        verdicts = []
        valves = []
        markers = []
        contradicted = []
        for row in live_rows:
            kind = _marker_settlement_kind(row)
            if kind in ("routing", "ambiguous"):
                if not custody_facts:
                    markers.append(row)
                elif kind == "routing":
                    contradicted.append(row)
                else:
                    verdicts.append(row)
            elif kind == "verdict-bearing" or str(row["source"]) in (
                "park.grant",
                "park.refusal",
            ):
                verdicts.append(row)
            else:
                valves.append(row)
        governing: Optional[Mapping[str, Any]] = (
            verdicts[-1] if verdicts else (valves[-1] if valves else None)
        )
        if governing is None and contradicted:
            cited_row = contradicted[-1]
            member_reasons[stable_id] = (
                "marker-contradicts-custody: the routing-labelled row "
                f"{cited_row['source']}[{cited_row['row_index']}] (reason "
                f"{cited_row.get('reason_code')!r}) claims no verdict was "
                "rendered, but the member's custody history is not silent "
                f"({', '.join(custody_facts)}): exclusion requires replay-"
                "derived custody silence (RULING W7 clauses 3 and 10)"
            )
            return None
        if governing is not None:
            return {
                "disposition": WITHHELD,
                "evidence": {
                    "source": str(governing["source"]),
                    "row_index": int(governing["row_index"]),
                    "row_sha256": str(governing["row_sha256"]),
                    "kind": str(governing["kind"]),
                },
            }
        governing = markers[-1]
        return {
            "disposition": CONTINUING,
            "evidence": {
                "source": str(governing["source"]),
                "row_index": int(governing["row_index"]),
                "row_sha256": str(governing["row_sha256"]),
                "kind": str(governing["kind"]),
                "reason_code": str(governing.get("reason_code")),
                "settlement_class": "routing-marker",
            },
        }
    accounting = _accounting_close_for(
        records_root,
        authorities,
        chain,
        stable_id,
        exclude_generation_id=generation_id,
    )
    if accounting is not None:
        snapshot, close, ancestor_entry = accounting
        ancestor_proof = load_partition_proof(
            records_root, str(close["partition_proof_sha256"])
        )
        origin = _entry_origin(ancestor_proof, ancestor_entry, str(close["close_sha256"]))
        if origin is None or not isinstance(origin.get("row_sha256"), str):
            member_reasons[stable_id] = "the ancestor withholding has no exact origin row"
            return None
        if not any(row.get("row_sha256") == origin["row_sha256"] for row in live_rows):
            member_reasons[stable_id] = "the exact origin withholding row is no longer live"
            return None
        return {
            "disposition": CARRIED,
            "evidence": {
                "ancestor_snapshot_id": snapshot,
                "ancestor_close_sha256": str(close["close_sha256"]),
                "origin": origin,
            },
        }
    member_reasons[stable_id] = (
        "no current head, no live valve or park row, and no resident ancestor "
        "close accounts this member: you cannot backfill a leg you cannot prove"
    )
    return None


def _member_diff(claimed: Mapping[str, Any], derived: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the per-member differences between two proofs' member tables."""

    findings: list[str] = []
    claimed_members = dict(claimed.get("members") or {})
    derived_members = dict(derived.get("members") or {})
    for stable_id in sorted(set(claimed_members) | set(derived_members)):
        theirs = claimed_members.get(stable_id)
        ours = derived_members.get(stable_id)
        if theirs == ours:
            continue
        described = (
            "unaccounted"
            if theirs is None
            else f"claimed {theirs.get('disposition')} {theirs.get('evidence')}"
        )
        recomputed = (
            "unaccounted"
            if ours is None
            else f"derived {ours.get('disposition')} {ours.get('evidence')}"
        )
        findings.append(f"{stable_id}: {described}, {recomputed}")
    if not findings:
        for field in ("window", "generation_id", "close_kind", "phase"):
            if claimed.get(field) != derived.get(field):
                findings.append(
                    f"proof {field} differs: claimed {claimed.get(field)!r}, "
                    f"derived {derived.get(field)!r}"
                )
    return tuple(findings)


def _previous_close_sources(
    records_root: Path, close: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    """Return the campaign's newest v2 close BEFORE this one, by ledger order."""

    from menagerie.crawler.close_object import (
        CLOSE_OBJECT_SCHEMA_VERSION,
        close_object_path,
        load_close_objects,
    )

    previous: Optional[Mapping[str, Any]] = None
    for row in load_close_objects([close_object_path(records_root)]):
        if row.get("close_sha256") == close.get("close_sha256"):
            break
        if row.get("schema_version") != CLOSE_OBJECT_SCHEMA_VERSION:
            continue
        if str(row.get("campaign_id")) != str(close.get("campaign_id")):
            continue
        previous = row
    return previous


def campaign_has_v2_close(records_root: Path, campaign_id: str) -> bool:
    """Return whether ANY v2 close exists for one campaign (producer side)."""

    from menagerie.crawler.close_object import (
        CLOSE_OBJECT_SCHEMA_VERSION,
        close_object_path,
        load_close_objects,
    )

    return any(
        row.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION
        and str(row.get("campaign_id")) == campaign_id
        for row in load_close_objects([close_object_path(records_root)])
    )


def backfill_era_refusals(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str],
) -> tuple[str, ...]:
    """Return every way one rung is NOT provably pre-machinery closed history.

    rt4 break 1: the backfill is the ONLY mint path for legs that terminally
    closed BEFORE the leg machinery existed, but nothing derived that era from
    the records -- so it could mint a licensing close for a live, in-flight,
    or never-run rung, and a re-mint could supersede a leg close that had
    honestly gone stale. The era is three record-derived predicates, never a
    caller flag:

    * TERMINALLY CLOSED IN HISTORY: a ``leg-close.v1`` row (readable history
      that licenses nothing itself) attests this exact rung identity --
      campaign, snapshot id, snapshot digest, and phase -- closed in the
      pre-machinery era; and
    * NO LEG MACHINERY OF ITS OWN: no machinery-era ``leg`` close and no
      ``leg``-kind leg-open row (live or closed generation) names this rung
      under either identity handle, any phase. Backfill generations are the
      tool's own idempotent re-mint epochs and do not count; and
    * REGISTERED BEFORE THE MACHINERY ERA: the rung's registry row is not
      licensed by a machinery-era ``leg`` close
      (:func:`machinery_licensed_registration_refusals`), the licence kind
      re-derived from the close ledger, never the persisted registry string.

    The mint path refuses on these findings before writing a byte, and
    :func:`proof_refusals` recomputes them on EVERY replay of a ``backfill``
    close -- the predicate is derived from the ledgers each time, with zero
    persisted producer input steering it.
    """

    from menagerie.crawler.close_object import (  # noqa: PLC0415 -- lazy: breaks the proof<->close import cycle
        CLOSE_OBJECT_SCHEMA_VERSION_V1,
        close_object_path,
        load_close_objects,
    )

    findings: list[str] = []
    closes = load_close_objects([close_object_path(records_root)])
    if not any(
        row.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION_V1
        and str(row.get("campaign_id")) == campaign_id
        and str(row.get("intake_snapshot_id")) == intake_snapshot_id
        and str(row.get("intake_snapshot_sha256")) == intake_snapshot_sha256
        and row.get("phase") == phase
        for row in closes
    ):
        findings.append(
            f"no leg-close.v1 row attests {intake_snapshot_id} terminally "
            "closed in the pre-machinery era: the backfill mints closes only "
            "for provably closed history, never a live, in-flight, or unrun "
            "rung"
        )
    findings.extend(
        leg_machinery_refusals(
            records_root,
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            intake_snapshot_sha256=intake_snapshot_sha256,
        )
    )
    findings.extend(
        machinery_licensed_registration_refusals(
            records_root,
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            intake_snapshot_sha256=intake_snapshot_sha256,
        )
    )
    return tuple(findings)


def machinery_licensed_registration_refusals(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
) -> tuple[str, ...]:
    """Return every way one rung was REGISTERED inside the machinery era.

    The third era predicate (A-1 follow-through). Chaining v1 history
    attestations through the durable same-campaign v2 tip unbricked minting
    them AFTER a campaign's first v2 close, so "pre-machinery" must now be
    proven about the rung's REGISTRATION too, not only about its own
    machinery census: a rung whose registry row was licensed by a
    machinery-era ``leg`` close entered the books through the live pipeline
    and can never be re-described as pre-machinery history.

    The registry row supplies only the licence's content address; the licence
    KIND is replay-derived by resolving that address against the close ledger
    as it is NOW -- never read off the persisted ``license_kind`` string a
    producer wrote. The eligibility rule is closed: an ABSENT licence (the
    campaign's first rung, or a registration that predates the licence
    fields) or one resolving to a v2 ``backfill`` close (the
    historical-bootstrap ladder adopting pre-machinery history in lineage
    order) is pre-machinery; everything else -- a ``leg`` licence, a licence
    that no longer resolves, an identity the registry does not bind --
    refuses, fail-closed.
    """

    from menagerie.crawler.close_object import (  # noqa: PLC0415 -- lazy: breaks the proof<->close import cycle
        CLOSE_OBJECT_SCHEMA_VERSION,
        close_object_path,
        load_close_objects,
    )
    from menagerie.crawler.rung_staging import load_rung_registrations

    registration = next(
        (
            row
            for row in load_rung_registrations(records_root)
            if row.campaign_id == campaign_id
            and row.snapshot_id == intake_snapshot_id
            and row.snapshot_sha256 == intake_snapshot_sha256
        ),
        None,
    )
    if registration is None:
        return (
            f"no registry row binds {intake_snapshot_id} to campaign "
            f"{campaign_id} at digest {intake_snapshot_sha256}: the era of an "
            "unregistered identity cannot be proven",
        )
    licence = registration.licensed_by_close_sha256
    if licence is None:
        return ()
    licensing = next(
        (
            row
            for row in load_close_objects([close_object_path(records_root)])
            if str(row.get("close_sha256")) == licence
        ),
        None,
    )
    if licensing is None:
        return (
            f"{intake_snapshot_id} was registered under licence {licence}, "
            "which is not resident on the close ledger: an era that cannot be "
            "re-derived is never assumed pre-machinery",
        )
    if (
        licensing.get("schema_version") != CLOSE_OBJECT_SCHEMA_VERSION
        or licensing.get("close_kind") != "backfill"
    ):
        return (
            f"{intake_snapshot_id} was registered under machinery-era licence "
            f"{licence} (close kind {licensing.get('close_kind')!r}): a rung "
            "whose registration was licensed by a live leg close entered the "
            "books through the leg pipeline and is not pre-machinery history",
        )
    return ()


def leg_machinery_refusals(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
) -> tuple[str, ...]:
    """Return every way one rung has LEG MACHINERY of its own.

    The second era predicate, shared by the backfill mint gate
    (:func:`backfill_era_refusals`) and the v1-history attestation mint: a
    rung with a machinery-era ``leg`` close or a ``leg``-kind generation
    (live or closed) under either identity handle is NOT pre-machinery
    history, so neither migration door may touch it (C-08: a migration path
    may only mint for the era it exists to serve and may never outrank a
    live-machinery close). Backfill generations are the tool's own idempotent
    re-mint epochs and do not count.
    """

    from menagerie.crawler.close_object import (  # noqa: PLC0415 -- lazy: breaks the proof<->close import cycle
        CLOSE_OBJECT_SCHEMA_VERSION,
        close_object_path,
        load_close_objects,
    )
    from menagerie.crawler.leg_generation import (
        _closed_generation_ids,  # noqa: PLC2701 -- the one open-vs-closed generation reader
        load_leg_open_rows,
    )

    def _names_this_rung(row: Mapping[str, Any]) -> bool:
        return str(row.get("campaign_id")) == campaign_id and (
            str(row.get("intake_snapshot_id")) == intake_snapshot_id
            or str(row.get("intake_snapshot_sha256")) == intake_snapshot_sha256
        )

    findings: list[str] = []
    for row in load_close_objects([close_object_path(records_root)]):
        if (
            row.get("schema_version") == CLOSE_OBJECT_SCHEMA_VERSION
            and row.get("close_kind") == "leg"
            and _names_this_rung(row)
        ):
            findings.append(
                f"machinery-era leg close {row['close_sha256']} already "
                f"adjudicated {intake_snapshot_id}: a rung with leg machinery "
                "of its own is not pre-machinery, and a backfill re-mint may "
                "never supersede a leg close"
            )
    closed_generations = _closed_generation_ids(records_root)
    for row in load_leg_open_rows(records_root):
        if row["close_kind"] != "leg" or not _names_this_rung(row):
            continue
        if row["generation_id"] in closed_generations:
            state = "closed"
            remedy = ""
        else:
            state = "live, unclosed"
            remedy = (
                " -- the window that live leg is accumulating cannot be "
                "bypassed by a mint under a different generation"
            )
        findings.append(
            f"leg generation {row['generation_id']} ({state}) was opened for "
            f"{intake_snapshot_id}: a rung with leg machinery of its own is "
            f"not pre-machinery{remedy}"
        )
    return tuple(findings)


def proof_refusals(
    records_root: Path,
    close: Mapping[str, Any],
    *,
    lineage_member_ids: Sequence[str],
) -> tuple[str, ...]:
    """Recompute a close's sealed proof and return every disagreement.

    The licensing decision: load the bound proof by digest, anchor its window
    (leg-open row, floor prefixes, continuity against the campaign's previous
    close), and recompute the whole classification from the pinned bytes. The
    producer's ``partition_proof_sha256`` must equal the recomputation
    hash-for-hash; a mismatch refuses with the per-member diff.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    close:
        Validated v2 close object.
    lineage_member_ids:
        The lineage snapshot's own membership, derived by the caller from
        durable, registry-verified bytes.

    Returns
    -------
    tuple[str, ...]
        Human-readable refusals; empty when the proof verifies.
    """

    from menagerie.crawler.leg_generation import (
        find_leg_open_row,
        window_floor_refusals,
    )

    members = frozenset(str(member) for member in lineage_member_ids)
    if not members:
        return ("the lineage snapshot names no members: a close over nothing licenses nothing",)
    proof_digest = str(close.get("partition_proof_sha256"))
    try:
        proof = load_partition_proof(records_root, proof_digest)
    except PartitionProofError as exc:
        return (str(exc),)

    findings: list[str] = []
    if proof.get("schema_version") not in PARTITION_PROOF_SCHEMA_VERSIONS:
        return (
            f"partition proof schema {proof.get('schema_version')!r} cannot license; "
            f"re-mint as {PARTITION_PROOF_SCHEMA_VERSION}",
        )
    for field in ("campaign_id", "intake_snapshot_id", "intake_snapshot_sha256", "phase"):
        if proof.get(field) != close.get(field):
            findings.append(
                f"the proof's {field} ({proof.get(field)!r}) contradicts the "
                f"close's ({close.get(field)!r})"
            )
    if proof.get("generation_id") != close.get("generation_id"):
        findings.append("the proof and the close name different leg generations")
    if proof.get("close_kind") != close.get("close_kind"):
        findings.append("the proof and the close disagree about their kind")
    if findings:
        return tuple(findings)

    proof_members = frozenset(dict(proof.get("members") or {}))
    if proof_members != members:
        missing = sorted(members - proof_members)
        foreign = sorted(proof_members - members)
        return (
            "the proof does not partition the lineage snapshot's exact "
            f"membership ({len(missing)} unaccounted {missing[:5]}, "
            f"{len(foreign)} foreign {foreign[:5]})",
        )

    close_kind = str(close.get("close_kind"))
    baselines: Optional[Mapping[str, Any]] = None
    previous = _previous_close_sources(records_root, close)
    expected_predecessor = None if previous is None else previous.get("close_sha256")
    if close.get("predecessor_close_sha256") != expected_predecessor:
        return (
            f"the close names predecessor {close.get('predecessor_close_sha256')!r}, "
            f"but the durable same-campaign spine yields {expected_predecessor!r}",
        )
    if close_kind == "leg":
        open_digest = close.get("leg_open_row_sha256")
        open_row = (
            None
            if not isinstance(open_digest, str)
            else find_leg_open_row(records_root, open_digest)
        )
        if open_row is None:
            return (
                "no durable leg-open row matches the close's "
                f"leg_open_row_sha256 ({open_digest!r}): a close nobody opened "
                "a leg for cannot exist",
            )
        for field, close_field in (
            ("campaign_id", "campaign_id"),
            ("intake_snapshot_id", "intake_snapshot_id"),
            ("intake_snapshot_sha256", "intake_snapshot_sha256"),
            ("phase", "phase"),
            ("close_kind", "close_kind"),
            ("generation_id", "generation_id"),
        ):
            if open_row.get(field) != close.get(close_field):
                findings.append(
                    f"the leg-open row's {field} ({open_row.get(field)!r}) is not "
                    f"the close's ({close.get(close_field)!r})"
                )
        if findings:
            return tuple(findings)
        baselines = dict(open_row["baselines"])
        window = dict(proof.get("window") or {})
        for name in CLOSE_SOURCE_NAMES:
            base_rows = int(dict(baselines.get(name) or {}).get("rows") or 0)
            bounds = window.get(name) or [None, None]
            if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] != base_rows:
                findings.append(
                    f"the proof's {name} window floor ({bounds!r}) is not the "
                    f"leg-open baseline ({base_rows} rows): the floor is the "
                    "open row's, never the producer's"
                )
        if findings:
            return tuple(findings)
        findings.extend(window_floor_refusals(records_root, baselines))
        if previous is not None:
            from menagerie.crawler.close_reads import prefix_digest

            sources = close_ledger_sources(records_root)
            for name in CLOSE_SOURCE_NAMES:
                prior = dict(dict(previous.get("sources") or {}).get(name) or {})
                prior_rows = int(prior.get("rows") or 0)
                prior_bytes = int(prior.get("bytes") or 0)
                floor_rows = int(dict(baselines.get(name) or {}).get("rows") or 0)
                if floor_rows < prior_rows:
                    findings.append(
                        f"the leg window floor for {name} ({floor_rows} rows) lies "
                        f"below the campaign's previous close ({prior_rows} rows): "
                        "evidence from a prior leg's era proves nothing here"
                    )
                    continue
                # The prior close's prefix is proven by hashing exactly those
                # bytes in bounded chunks; ``read_bytes()`` pulled the whole
                # growing ledger into memory for all 15 slots to hash a slice.
                if prefix_digest(sources[name], prior_bytes) != prior.get("ledger_sha256"):
                    findings.append(
                        f"{name} no longer extends the campaign's previous close "
                        "append-only: pre-window history moved"
                    )
        if findings:
            return tuple(findings)
    else:
        # A backfill close's era anchor is what the leg-open row is to a leg
        # close: recomputed from the ledgers on every replay, never trusted
        # from mint time (rt4 break 1).
        findings.extend(
            backfill_era_refusals(
                records_root,
                campaign_id=str(close["campaign_id"]),
                intake_snapshot_id=str(close["intake_snapshot_id"]),
                intake_snapshot_sha256=str(close["intake_snapshot_sha256"]),
                phase=close.get("phase"),
            )
        )
        if findings:
            return tuple(findings)

    try:
        derived = derive_partition_proof(
            records_root,
            campaign_id=str(close["campaign_id"]),
            intake_snapshot_id=str(close["intake_snapshot_id"]),
            intake_snapshot_sha256=str(close["intake_snapshot_sha256"]),
            phase=close.get("phase"),
            close_kind=close_kind,
            generation_id=str(close["generation_id"]),
            baselines=baselines,
            member_ids=sorted(members),
            operator_attributions=(
                dict(close.get("operator_attributions") or {}) if close_kind == "backfill" else None
            ),
            # RULING W7 clause 8: the era is keyed by the sealed proof's OWN
            # schema version, so every resident v3 close (all of 200a)
            # re-derives byte-identically under v3 semantics forever. New
            # attestations cannot choose the era -- the close APPEND door
            # refuses a fresh close binding a non-current proof schema.
            proof_schema_version=str(proof["schema_version"]),
        )
    except PartitionProofError as exc:
        return (f"the proof does not recompute from the pinned bytes: {exc}",)

    if derived["partition_proof_sha256"] != proof_digest:
        diff = _member_diff(proof, derived)
        return (
            "the sealed proof does not equal the gate's recomputation over the "
            "pinned bytes: " + ("; ".join(diff) if diff else "digest mismatch"),
        )
    return ()


def licence_refusals(
    records_root: Path,
    close: Mapping[str, Any],
    *,
    lineage_member_ids: Sequence[str],
    introduced_member_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return successor-licence refusals over an already verified v3 proof.

    Closing and licensing are deliberately separate.  This function reads no
    producer authorization flag: ``proved_here`` comes from the sealed proof,
    while ``introduced`` is supplied by the registry consumer after it verifies
    every ancestor snapshot.
    """

    del lineage_member_ids  # exact membership was already checked by proof_refusals
    try:
        proof = load_partition_proof(records_root, str(close["partition_proof_sha256"]))
    except (KeyError, PartitionProofError) as exc:
        return (str(exc),)
    if proof.get("schema_version") not in PARTITION_PROOF_SCHEMA_VERSIONS:
        return (
            "the close binds a pre-v3 proof with no origin-traced custody; re-mint "
            f"the close under {PARTITION_PROOF_SCHEMA_VERSION}",
        )
    required_origin_fields = {
        "snapshot_id",
        "generation_id",
        "close_sha256",
        "disposition",
        "record_revision",
        "row_sha256",
    }
    # The member table is resolved ONCE for both loops below (round-2 T4):
    # copying it per carried member was O(#members**2) in the rung's own width.
    proof_members = proof.get("members") or {}
    malformed: list[str] = []
    for member in sorted(carried(proof)):
        evidence = proof_members[member].get("evidence") or {}
        origin_value = evidence.get("origin")
        if not isinstance(origin_value, Mapping) or set(origin_value) != required_origin_fields:
            malformed.append(member)
    if malformed:
        return (
            f"{len(malformed)} CARRIED member(s) have no well-formed origin "
            f"({malformed[:5]}): inheritance may copy an origin, never mint one",
        )
    verdict = licence_verdict(
        records_root,
        proof,
        lineage_member_ids=dict(proof.get("members") or {}),
        introduced_member_ids=introduced_member_ids,
    )
    proved = proved_here(proof)
    if verdict == "HONEST_NO_OP":
        origin_snapshots = sorted(
            {
                str(proof_members[member]["evidence"]["origin"]["snapshot_id"])
                for member in carried(proof)
            }
        )
        return (
            "the close is an HONEST NO-OP: every member's resolution traces to "
            f"an ancestor rung ({origin_snapshots}), so this rung proved nothing "
            "of its own and licenses no boundary",
        )
    introduced = frozenset(str(value) for value in introduced_member_ids)
    missing = sorted(introduced - proved)
    if verdict == "INCOMPLETE_LADDER":
        continuing = sorted(set(missing) & continuing_here(proof))
        if continuing:
            # RULING W7 clause 5 (the Sol B15 finding): a licence boundary
            # whose introduced set contains a marker-only member refuses,
            # naming the member -- a row that says "no verdict was rendered"
            # proves nothing and licenses nothing.
            return (
                f"{len(missing)} member(s) this rung is the first to contain "
                f"are open continuations or inherited ({missing[:5]}; "
                f"continuing: {continuing[:5]}): INCOMPLETE_LADDER",
            )
        return (
            f"{len(missing)} member(s) this rung is the first to contain are "
            f"accounted by inheritance ({missing[:5]}): INCOMPLETE_LADDER",
        )
    if close.get("close_kind") == "backfill":
        from menagerie.crawler.rung_staging import load_rung_registrations

        consumed = [
            row
            for row in load_rung_registrations(records_root)
            if row.license_kind == "historical_bootstrap"
            and row.licensed_by_close_sha256 == close.get("close_sha256")
        ]
        if consumed:
            return (
                "the historical-bootstrap licence is one-use and was already "
                f"consumed by {consumed[0].snapshot_id}",
            )
    return ()


def licence_verdict(
    records_root: Path,
    proof: Mapping[str, Any],
    *,
    lineage_member_ids: Sequence[str] | Mapping[str, Any],
    introduced_member_ids: Sequence[str],
) -> LicenceVerdict:
    """Derive the closed four-verdict licensing taxonomy.

    ``COMPLETE_AT_WIDTH`` is disclosure only.  It follows the same proved-here
    predicate as ``LICENSING`` and never creates an additional authorization.
    """

    proved = proved_here(proof)
    introduced = frozenset(str(value) for value in introduced_member_ids)
    if introduced & continuing_here(proof):
        # RULING W7 clause 5: an introduced member left CONTINUING is the
        # richer, member-naming refusal -- never masked by the no-op arm.
        # v3 proofs carry no CONTINUING disposition, so the pinned pre-ruling
        # taxonomy is untouched.
        return "INCOMPLETE_LADDER"
    if not proved:
        return "HONEST_NO_OP"
    if introduced - proved:
        return "INCOMPLETE_LADDER"
    members = frozenset(str(value) for value in lineage_member_ids)
    try:
        from menagerie.crawler.rung_staging import resolve_verified_root

        _binding, root_snapshot = resolve_verified_root(
            records_root, str(proof["campaign_id"])
        )
    except (KeyError, OSError, ValueError):
        return "LICENSING"
    full_membership = frozenset(item.stable_id for item in root_snapshot.items)
    if members == full_membership:
        return "COMPLETE_AT_WIDTH"
    return "LICENSING"
