"""Fixture support for tests whose subject sits DOWNSTREAM of a leg close.

Since BD-1, staging a rung that derives from a registered prior rung requires
that rung's immutable close object; since the v2 contract, the close binds a
generation-scoped sealed ``PartitionProofV2``. Tests about lineage chaining,
registry append-only semantics, or CLI projection are not tests about closing
a leg, so they synthesize the precondition here instead of driving a whole leg.

Two distinct fixtures, deliberately separated:

* :func:`record_fixture_close` -- the HONEST path. Opens (or continues) a real
  leg generation, optionally records every member in-window recorded-under the
  snapshot, DERIVES the proof through the production derivation, and appends a
  close structurally identical to a driver-produced one. If the evidence on
  disk cannot resolve a member, this fixture refuses exactly like the driver.
* :func:`forge_fixture_close` -- the ATTACK path. Writes a structurally valid
  v2 close over attacker-chosen citations WITHOUT deriving them, so gate pins
  can prove the staging door refuses what the builder would never produce.

Nothing in production may ever construct a close for a leg it did not run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.close_object import (
    append_close_object,
    build_close_object,
    close_object_path,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.leg_generation import (
    CLOSE_SOURCE_NAMES,
    MODELS_SOURCE_NAMES,
    LegGeneration,
    census_identities,
    close_ledger_sources,
    hash_bytes,
    open_leg_generation,
    source_identity,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.partition_proof import (
    ADJUDICATED,
    CARRIED,
    CONTINUING,
    OPERATOR_ATTRIBUTED,
    PARTITION_PROOF_SCHEMA_VERSION,
    WITHHELD,
    award_transition_census,
    derive_partition_proof,
    write_partition_proof,
)

_FIXTURE_AUTHORITY_IDENTITIES = {
    "author_prompt_identity": "sha256:fixture-author-prompt",
    "reducer_policy_identity": "sha256:fixture-reducer-policy",
}

_PARK_KIND_ARMS = {
    "review-held:supersession-pending-grant": "grant_pending",
    "review-held:award-supersession-pending-grant": "grant_pending",
    "review-held:terminal-recording-refused": "refusal_parked",
}


def open_fixture_leg(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str] = None,
    run_id: str = "pytest-close",
    opened_at: str = "2026-08-11T00:00:00Z",
) -> LegGeneration:
    """Open (or continue) the durable leg generation for one fixture rung."""

    return open_leg_generation(
        records_root,
        campaign_id=campaign_id,
        intake_snapshot_id=intake_snapshot_id,
        intake_snapshot_sha256=intake_snapshot_sha256,
        phase=phase,
        run_id=run_id,
        opened_at=opened_at,
    )


def _fixture_head(
    stable_id: str,
    intake_snapshot_id: str,
    run_id: str,
    *,
    status_kind: str = "failed",
    status_code: str = "failed:import",
    parent_revision: Optional[str] = None,
    record_seq: int = 1,
) -> JsonObject:
    """Build one minimal snapshot-scoped current model revision.

    ``status_kind`` and ``parent_revision`` are the two fields the
    award-transition census reads (a transition is a row whose recorded parent
    holds a ``runs`` kind). Defaulting them to a parentless ``failed`` head is
    what every close-object fixture wants; :func:`seed_award_head` and
    :func:`append_award_downgrade` vary them so the census has something to
    find.
    """

    body: JsonObject = {
        "stable_id": stable_id,
        "record_seq": record_seq,
        "parent_revision": parent_revision,
        "status": {"kind": status_kind, "code": status_code},
        "source_resolution": {"rung": "R2_VENDOR", "attempted_rungs": []},
        "budget": {
            "author_sessions_used": 1,
            "author_sessions_max": 3,
            "gate_rounds_used": 0,
            "run_revisions_used": 1,
            "explicit_grants": [],
        },
        "dependency_vector": {"intake_snapshot_id": intake_snapshot_id},
        "provenance": {"producer_run_id": run_id},
    }
    body["record_revision"] = stable_hash(body)
    return body


def _append_head(records_root: Path, row: JsonObject) -> str:
    """Append one head to the first present models layout; return its revision."""

    sources = close_ledger_sources(records_root)
    target = next(
        (sources[name] for name in MODELS_SOURCE_NAMES if sources[name].is_file()),
        None,
    )
    if target is None:
        raise AssertionError(
            "no models layout is present: materialize one BEFORE appending a "
            "fixture head (a layout may not change presence during a leg)"
        )
    with target.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
    return str(row["record_revision"])


def seed_award_head(
    records_root: Path,
    *,
    stable_id: str,
    intake_snapshot_id: str,
    run_id: str = "pytest-close",
    status_kind: str = "runs",
) -> str:
    """Append a PRE-leg ``runs`` award head and return its exact revision.

    Every other close-object fixture head is a parentless ``failed`` row, so
    the award-transition census (which fires only on a row whose recorded
    PARENT holds a ``runs`` kind) had nothing to find anywhere in this corpus
    and both its arms were structurally ``[]``. The disagreement gate could
    therefore only ever catch the close that INVENTS a transition, never the
    close that HIDES one. Call this before opening the leg so the award sits
    below the generation floor, exactly like a real prior-leg award.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory, with a models layout present.
    stable_id, intake_snapshot_id, run_id:
        Identity and provenance of the award head.
    status_kind:
        The recorded kind. Only ``runs`` is an award; the parameter exists so a
        test can BREAK the fixture deliberately and watch the census go silent.

    Returns
    -------
    str
        Exact ``record_revision`` a successor must name as its parent.
    """

    return _append_head(
        records_root,
        _fixture_head(
            stable_id,
            intake_snapshot_id,
            run_id,
            status_kind=status_kind,
            status_code=status_kind if status_kind == "runs" else f"{status_kind}:import",
        ),
    )


def append_award_downgrade(
    records_root: Path,
    *,
    stable_id: str,
    parent_revision: str,
    intake_snapshot_id: str,
    run_id: str = "pytest-close",
) -> str:
    """Append an IN-window ``failed`` successor of one award head.

    The ``runs -> non-runs`` transition the AW-1 census exists to disclose.
    Call this after the leg is open so the row lands above the generation
    floor, which is the only window the census scans.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    stable_id, parent_revision, intake_snapshot_id, run_id:
        Identity, the exact superseded award revision, and provenance.

    Returns
    -------
    str
        The successor's ``record_revision``.
    """

    return _append_head(
        records_root,
        _fixture_head(
            stable_id,
            intake_snapshot_id,
            run_id,
            status_kind="failed",
            status_code="failed:accuracy-gate",
            parent_revision=parent_revision,
            record_seq=2,
        ),
    )


def _seed_member_heads(
    records_root: Path, member_ids: Sequence[str], intake_snapshot_id: str, run_id: str
) -> None:
    """Record a head for every member lacking one, in the first present layout."""

    sources = close_ledger_sources(records_root)
    target = next(
        (sources[name] for name in MODELS_SOURCE_NAMES if sources[name].is_file()),
        None,
    )
    if target is None:
        raise AssertionError(
            "no models layout is present: materialize one BEFORE opening the "
            "fixture leg (a layout may not change presence during a leg)"
        )
    existing = {
        json.loads(line)["stable_id"]
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    rows = [
        _fixture_head(stable_id, intake_snapshot_id, run_id)
        for stable_id in member_ids
        if stable_id not in existing
    ]
    if not rows:
        return
    with target.open("ab") as handle:
        handle.write(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _introduced_for_snapshot(
    records_root: Path, intake_snapshot_id: str, member_ids: Sequence[str]
) -> list[str]:
    """Recompute introduced members from the registry when the rung is registered."""

    from menagerie.crawler.rung_staging import introduced_member_ids, load_rung_registrations

    registration = next(
        (
            row
            for row in load_rung_registrations(records_root)
            if row.snapshot_id == intake_snapshot_id
        ),
        None,
    )
    if registration is None:
        return sorted(str(value) for value in member_ids)
    return sorted(introduced_member_ids(records_root, registration))


def _disclosure_arms(
    proof: Mapping[str, Any],
    member_ids: Sequence[str],
    *,
    records_root: Path,
    intake_snapshot_id: str,
) -> dict[str, Any]:
    """Project one derived proof into honest close disclosure arms."""

    members = dict(proof.get("members") or {})
    buckets: dict[str, list[str]] = {}
    machine_deferred: dict[str, list[str]] = {}
    grant_pending: list[str] = []
    refusal_parked: list[str] = []
    for stable_id in sorted(members):
        entry = members[stable_id]
        disposition = entry.get("disposition")
        if disposition in (WITHHELD, CONTINUING):
            kind = str(dict(entry.get("evidence") or {}).get("kind"))
            arm = _PARK_KIND_ARMS.get(kind)
            # DX-22: the park arms are DISCLOSURE, not accounting. A parked
            # member's ACCOUNTING is its machine-deferral claim (the producer
            # shape for a recordless park member), and the park arm discloses
            # the live operator decision on top of it.
            machine_deferred.setdefault(kind, []).append(stable_id)
            if arm == "grant_pending":
                grant_pending.append(stable_id)
            elif arm == "refusal_parked":
                refusal_parked.append(stable_id)
        else:
            buckets.setdefault("runs", []).append(stable_id)
    from menagerie.crawler.partition_proof import (
        PARTITION_PROOF_SCHEMA_VERSION_V3,
        adjudicated_here,
        attributed_here,
        carried,
        continuing_here,
        proved_here,
        withheld_here,
    )

    dispositions = [str(entry.get("disposition")) for entry in members.values()]
    arms: dict[str, Any] = {
        "intake": len(tuple(member_ids)),
        "valid": True,
        "missing": [],
        "extra": [],
        "duplicates": [],
        "invalid": [],
        "buckets": buckets,
        "machine_deferred": machine_deferred,
        "superseded_deferral_rows": [],
        "grant_pending": grant_pending,
        "refusal_parked": refusal_parked,
        "adjudicated_here": sorted(adjudicated_here(proof)),
        "carried_count": dispositions.count(CARRIED),
        "withheld_here": sorted(withheld_here(proof)),
        "attributed_here": sorted(attributed_here(proof)),
        "carried": sorted(carried(proof)),
        "proved_here": sorted(proved_here(proof)),
        "introduced": _introduced_for_snapshot(
            records_root, intake_snapshot_id, member_ids
        ),
    }
    if proof.get("schema_version") != PARTITION_PROOF_SCHEMA_VERSION_V3:
        # RULING W7 clause 6: v4 closes disclose the first-class continuation
        # arm exactly as the driver does (the gate recomputes it).
        arms["continuing"] = sorted(continuing_here(proof))
    return arms


def record_fixture_close(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    member_ids: Sequence[str] = (),
    seed_member_heads: bool = False,
    completion_status: str = "complete",
    phase: Optional[str] = None,
    closed_at: str = "2026-08-11T00:00:00Z",
    run_id: str = "pytest-close",
    partition_overrides: Optional[Mapping[str, Any]] = None,
    currency_overrides: Optional[Mapping[str, Any]] = None,
    proof_schema_version: Optional[str] = None,
) -> JsonObject:
    """Close one fixture rung HONESTLY through the production derivation.

    Opens (or continues) the rung's leg generation, optionally records every
    member in-window recorded-under the snapshot, derives the sealed proof
    from the bytes on disk, and appends the v2 close. Raises exactly like the
    driver when the evidence cannot resolve a member.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id, intake_snapshot_id, intake_snapshot_sha256:
        Identity of the rung that closes.
    member_ids:
        The rung's membership.
    seed_member_heads:
        Whether to record an in-window head for every member lacking one.
    completion_status:
        Driver completion status to attest (must be a close).
    phase, closed_at, run_id:
        Provenance.
    partition_overrides, currency_overrides:
        DISCLOSURE overrides, for tests that need an unclean attested census
        (an attested coverage hole, a stale head). These never touch the
        sealed proof.

    Returns
    -------
    dict[str, Any]
        The persisted close object.
    """

    members = sorted(str(stable_id) for stable_id in member_ids)
    generation = open_fixture_leg(
        records_root,
        campaign_id=campaign_id,
        intake_snapshot_id=intake_snapshot_id,
        intake_snapshot_sha256=intake_snapshot_sha256,
        phase=phase,
        run_id=run_id,
    )
    if seed_member_heads:
        _seed_member_heads(records_root, members, intake_snapshot_id, run_id)
    from menagerie.crawler.leg_generation import record_leg_plan

    record_leg_plan(
        records_root,
        campaign_id=campaign_id,
        generation_id=generation.generation_id,
        run_id=run_id,
        admitted_ids=members if seed_member_heads else (),
        plan_payload={"fixture": True, "admitted_ids": members if seed_member_heads else []},
        created_at=closed_at,
    )

    proof = derive_partition_proof(
        records_root,
        campaign_id=campaign_id,
        intake_snapshot_id=intake_snapshot_id,
        intake_snapshot_sha256=intake_snapshot_sha256,
        phase=phase,
        close_kind="leg",
        generation_id=generation.generation_id,
        baselines=generation.baselines,
        member_ids=members,
        proof_schema_version=proof_schema_version or PARTITION_PROOF_SCHEMA_VERSION,
    )
    write_partition_proof(records_root, proof)
    # D-49 parity with the driver: an in-window operator row post-dating the
    # close refuses here exactly like the production close path.
    from menagerie.crawler.operator_provenance import (
        OperatorProvenanceError,
        post_dated_window_rows,
    )

    time_findings = post_dated_window_rows(
        records_root, baselines=generation.baselines, as_of=closed_at
    )
    if time_findings:
        raise OperatorProvenanceError(
            "close refuses counted post-dated operator rows: " + "; ".join(time_findings)
        )
    # D-48 parity with the driver: every live park row names its computed
    # release path in the close currency block, and an uncomputable path
    # refuses this fixture close exactly like the production close path.
    from menagerie.crawler.grant_parks import grant_park_path, load_grant_park_rows
    from menagerie.crawler.park_release import (
        current_park_tree_identity,
        park_release_path_census,
    )
    from menagerie.crawler.refusal_parks import load_refusal_park_rows, refusal_park_path

    fixture_parks: dict[str, dict[str, Any]] = {}
    for park_row in (
        *load_grant_park_rows([grant_park_path(records_root)]),
        *load_refusal_park_rows([refusal_park_path(records_root)]),
    ):
        fixture_parks.setdefault(str(park_row.get("disposition")), {})[
            str(park_row["stable_id"])
        ] = dict(park_row)
    park_release_paths = park_release_path_census(
        fixture_parks, tree_identity=current_park_tree_identity()
    )
    # W42 parity with the driver: a member serviced by an in-window VOID while
    # its requeue grant stays LIVE refuses this fixture close exactly like the
    # production close path.
    from menagerie.crawler.operator_provenance import void_only_with_live_grant_findings

    void_grant_findings = void_only_with_live_grant_findings(
        records_root, baselines=generation.baselines, as_of=closed_at
    )
    if void_grant_findings:
        raise OperatorProvenanceError(
            "close refuses operator-void-only members with live grants: "
            + "; ".join(void_grant_findings)
        )
    partition = _disclosure_arms(
        proof,
        members,
        records_root=records_root,
        intake_snapshot_id=intake_snapshot_id,
    )
    partition.update(dict(partition_overrides or {}))
    currency: dict[str, Any] = {
        "ledger_heads": len(members),
        "dependency_current": len(members),
        "stale": 0,
        "stale_by_axis": {},
        "stale_reasons": {},
        "awards_stale_on_live_basis": [],
        "award_runs_to_non_runs": list(
            dict(proof["award_transition_census"])["runs_to_non_runs"]
        ),
        "award_runs_to_runs": list(
            dict(proof["award_transition_census"])["runs_to_runs"]
        ),
        "award_transition_instrumentation": dict(proof["award_transition_census"])[
            "instrumentation"
        ],
        "award_transition_windows": dict(proof["award_transition_census"])[
            "coverage_windows"
        ],
        # D-48: the per-row release-path disclosure, driver-close parity.
        "park_release_paths": park_release_paths,
    }
    from menagerie.crawler.close_object import continuing_member_disclosures

    # RULING W7 clause 6, driver-close parity through the ONE derivation.
    currency["continuing_members"] = [
        dict(entry) for entry in continuing_member_disclosures(records_root, proof)
    ]
    currency.update(dict(currency_overrides or {}))
    return append_close_object(
        close_object_path(records_root),
        build_close_object(
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            intake_snapshot_sha256=intake_snapshot_sha256,
            phase=phase,
            close_kind="leg",
            generation_id=generation.generation_id,
            leg_open_row_sha256=generation.row_sha256,
            sources=census_identities(records_root),
            partition_proof_sha256=proof["partition_proof_sha256"],
            predecessor_close_sha256=__import__(
                "menagerie.crawler.close_object", fromlist=["predecessor_close_sha256"]
            ).predecessor_close_sha256(records_root, campaign_id),
            run_id=run_id,
            machine_id="pytest-host",
            closed_at=closed_at,
            completion_status=completion_status,
            partition_arms=partition,
            currency_census=currency,
            authority_identities=_FIXTURE_AUTHORITY_IDENTITIES,
        ),
    )


def append_close_as_resident_history(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    proof: Mapping[str, Any],
    generation: LegGeneration,
    completion_status: str = "complete-with-deferrals",
    run_id: str = "pytest-close",
    closed_at: str = "2026-08-11T00:00:00Z",
) -> JsonObject:
    """Append one close binding an ALREADY-SEALED proof as resident history.

    The proof-era door in :func:`append_close_object` refuses a NEW close
    binding a non-current proof era (RULING W7 clause 8, attack D). Era pins
    (N7) need the opposite: a RESIDENT v3-bound close -- the shape all of 200a
    holds -- to prove replay derives under the proof's own sealed era forever.
    This helper simulates that residency by writing the validated, chained row
    directly, exactly as the ledger bytes would exist had the close been
    minted before the ruling. Tests only; production has exactly one door.
    """

    from menagerie.crawler.ledger_chain import link_row
    from menagerie.crawler.close_object import (
        close_ledger_rows,
        predecessor_close_sha256,
        validate_close_object,
    )

    partition = _disclosure_arms(
        proof,
        sorted(dict(proof.get("members") or {})),
        records_root=records_root,
        intake_snapshot_id=intake_snapshot_id,
    )
    currency: dict[str, Any] = {
        "ledger_heads": len(dict(proof.get("members") or {})),
        "dependency_current": len(dict(proof.get("members") or {})),
        "stale": 0,
        "stale_by_axis": {},
        "stale_reasons": {},
        "awards_stale_on_live_basis": [],
        "award_runs_to_non_runs": list(
            dict(proof["award_transition_census"])["runs_to_non_runs"]
        ),
        "award_runs_to_runs": list(dict(proof["award_transition_census"])["runs_to_runs"]),
        "award_transition_instrumentation": dict(proof["award_transition_census"])[
            "instrumentation"
        ],
        "award_transition_windows": dict(proof["award_transition_census"])[
            "coverage_windows"
        ],
    }
    row = build_close_object(
        campaign_id=campaign_id,
        intake_snapshot_id=intake_snapshot_id,
        intake_snapshot_sha256=intake_snapshot_sha256,
        phase=None,
        close_kind="leg",
        generation_id=generation.generation_id,
        leg_open_row_sha256=generation.row_sha256,
        sources=census_identities(records_root),
        partition_proof_sha256=str(proof["partition_proof_sha256"]),
        predecessor_close_sha256=predecessor_close_sha256(records_root, campaign_id),
        run_id=run_id,
        machine_id="pytest-host",
        closed_at=closed_at,
        completion_status=completion_status,
        partition_arms=partition,
        currency_census=currency,
        authority_identities=_FIXTURE_AUTHORITY_IDENTITIES,
    )
    path = close_object_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = close_ledger_rows(path)
    linked = validate_close_object(link_row(row, scope="meta.closes", existing_rows=existing))
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(linked) + b"\n")
    return linked


def _forged_open_row(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str],
    generation_id: str,
    baselines: Mapping[str, Any],
    run_id: str,
) -> str:
    """Append an attacker-authored (digest-consistent) leg-open row."""

    from menagerie.crawler.leg_generation import (
        LEG_EVENT_SCHEMA_VERSION,
        _append_leg_event,
        record_leg_plan,
    )

    body: JsonObject = {
        "schema_version": LEG_EVENT_SCHEMA_VERSION,
        "event": "leg_open",
        "campaign_id": campaign_id,
        "intake_snapshot_id": intake_snapshot_id,
        "intake_snapshot_sha256": intake_snapshot_sha256,
        "phase": phase,
        "close_kind": "leg",
        "generation_id": generation_id,
        "ordinal": 1,
        "run_id": run_id,
        "opened_at": "2026-08-11T00:00:00Z",
        "baselines": {name: dict(identity) for name, identity in baselines.items()},
    }
    row = dict(body)
    row["row_sha256"] = stable_hash(body)
    _append_leg_event(records_root, row)
    run_body: JsonObject = {
        "schema_version": LEG_EVENT_SCHEMA_VERSION,
        "event": "leg_run",
        "campaign_id": campaign_id,
        "generation_id": generation_id,
        "run_id": run_id,
        "created_at": "2026-08-11T00:00:00Z",
    }
    _append_leg_event(records_root, {**run_body, "row_sha256": stable_hash(run_body)})
    record_leg_plan(
        records_root,
        campaign_id=campaign_id,
        generation_id=generation_id,
        run_id=run_id,
        admitted_ids=(),
        plan_payload={"fixture_forgery": True},
        created_at="2026-08-11T00:00:00Z",
    )
    return str(row["row_sha256"])


def _auto_carried_evidence(
    records_root: Path,
    campaign_id: str,
    intake_snapshot_id: str,
    stable_id: str,
    *,
    exclude_generation_id: Optional[str],
) -> JsonObject:
    """Resolve one member's honest carried citation, for mixed forgeries."""

    from menagerie.crawler.partition_proof import (
        _accounting_close_for,  # noqa: PLC2701 -- fixture reuses the production resolver
        _ancestor_chain,  # noqa: PLC2701
        _carry_authorities,  # noqa: PLC2701
    )

    chain = _ancestor_chain(records_root, campaign_id, intake_snapshot_id)
    authorities = _carry_authorities(records_root, campaign_id, chain)
    accounting = _accounting_close_for(
        records_root,
        authorities,
        chain,
        stable_id,
        exclude_generation_id=exclude_generation_id,
    )
    if accounting is None:
        raise AssertionError(f"no ancestor close accounts {stable_id} to carry from")
    snapshot, close, entry = accounting
    from menagerie.crawler.partition_proof import _entry_origin, load_partition_proof

    proof = load_partition_proof(records_root, str(close["partition_proof_sha256"]))
    origin = _entry_origin(proof, entry, str(close["close_sha256"]))
    return {
        "ancestor_snapshot_id": snapshot,
        "ancestor_close_sha256": str(close["close_sha256"]),
        "origin": origin,
    }


def forge_fixture_close(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    member_citations: Mapping[str, Mapping[str, Any]],
    generation: Optional[LegGeneration] = None,
    leg_open_row_sha256: Optional[str] = None,
    window_floor_zero: bool = False,
    phase: Optional[str] = None,
    completion_status: str = "complete",
    run_id: str = "pytest-forge",
    partition_overrides: Optional[Mapping[str, Any]] = None,
    proof_schema_version: str = PARTITION_PROOF_SCHEMA_VERSION,
) -> JsonObject:
    """Forge a structurally valid v2 close over attacker-chosen citations.

    The proof is written VERBATIM from ``member_citations`` -- no derivation,
    no refusal -- so gate pins can prove the staging door recomputes and
    refuses. A citation with ``evidence: None`` and disposition ``carried`` is
    auto-filled with the member's honest carry evidence, for forgeries that
    mix one lie into an otherwise honest close.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.
    campaign_id, intake_snapshot_id, intake_snapshot_sha256, phase:
        Identity the forged close claims.
    member_citations:
        ``stable_id -> {"disposition", "evidence"}`` exactly as the proof will
        carry them.
    generation:
        A REAL open generation to bind (its floor and open row), when the
        forgery targets the citation checks rather than the window anchors.
    leg_open_row_sha256:
        Explicit open-row digest override (e.g. a fabricated one).
    window_floor_zero:
        Forge a floor-zero window AND a matching attacker-authored leg-open
        row, targeting the continuity anchor.
    completion_status, run_id:
        Provenance the forgery claims.
    partition_overrides:
        Disclosure-arm overrides.

    Returns
    -------
    dict[str, Any]
        The persisted forged close row.
    """

    closing = census_identities(records_root)
    generation_id = (
        generation.generation_id
        if generation is not None
        else stable_hash({"forged-generation": intake_snapshot_sha256})
    )
    if generation is not None:
        from menagerie.crawler.leg_generation import record_leg_plan

        record_leg_plan(
            records_root,
            campaign_id=campaign_id,
            generation_id=generation_id,
            run_id=run_id,
            admitted_ids=(),
            plan_payload={"fixture_forgery": True},
            created_at="2026-08-11T00:00:00Z",
        )
    citations: dict[str, JsonObject] = {}
    for stable_id, entry in member_citations.items():
        disposition = str(entry["disposition"])
        evidence = entry.get("evidence")
        if evidence is None and disposition == CARRIED:
            evidence = _auto_carried_evidence(
                records_root,
                campaign_id,
                intake_snapshot_id,
                stable_id,
                exclude_generation_id=generation_id,
            )
        citations[str(stable_id)] = {
            "disposition": disposition,
            "evidence": None if evidence is None else dict(evidence),
        }

    if window_floor_zero:
        floors: dict[str, JsonObject] = {}
        for name in CLOSE_SOURCE_NAMES:
            floors[name] = {
                "present": bool(closing[name]["present"]),
                "rows": 0,
                "bytes": 0,
                "ledger_sha256": hash_bytes(b""),
                "tail_sha256": None,
            }
        baselines: Mapping[str, Any] = floors
        open_digest = _forged_open_row(
            records_root,
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            intake_snapshot_sha256=intake_snapshot_sha256,
            phase=phase,
            generation_id=generation_id,
            baselines=floors,
            run_id=run_id,
        )
    elif generation is not None:
        baselines = dict(generation.baselines)
        open_digest = generation.row_sha256
    else:
        baselines = {
            name: source_identity(close_ledger_sources(records_root)[name])
            for name in CLOSE_SOURCE_NAMES
        }
        open_digest = leg_open_row_sha256 or ("sha256:" + "0" * 64)
    if leg_open_row_sha256 is not None:
        open_digest = leg_open_row_sha256

    window = {
        name: [
            int(dict(baselines.get(name) or {}).get("rows") or 0),
            int(closing[name]["rows"]),
        ]
        for name in CLOSE_SOURCE_NAMES
    }
    proof_body: JsonObject = {
        "schema_version": proof_schema_version,
        "campaign_id": campaign_id,
        "intake_snapshot_id": intake_snapshot_id,
        "intake_snapshot_sha256": intake_snapshot_sha256,
        "phase": phase,
        "close_kind": "leg",
        "generation_id": generation_id,
        "window": window,
        "selection_plan_sha256": __import__(
            "menagerie.crawler.leg_generation", fromlist=["sealed_selection_plan"]
        ).sealed_selection_plan(records_root, generation_id)[1],
        "award_transition_census": award_transition_census(records_root, baselines),
        "members": citations,
    }
    proof = dict(proof_body)
    proof["partition_proof_sha256"] = stable_hash(proof_body)
    write_partition_proof(records_root, proof)

    members = sorted(citations)
    partition: dict[str, Any] = {
        "intake": len(members),
        "valid": True,
        "missing": [],
        "extra": [],
        "duplicates": [],
        "invalid": [],
        "buckets": {"runs": members},
        "machine_deferred": {},
        "superseded_deferral_rows": [],
        "grant_pending": [],
        "refusal_parked": [],
        "adjudicated_here": sorted(
            member for member, entry in citations.items() if entry["disposition"] == ADJUDICATED
        ),
        "withheld_here": sorted(
            member for member, entry in citations.items() if entry["disposition"] == WITHHELD
        ),
        "attributed_here": sorted(
            member
            for member, entry in citations.items()
            if entry["disposition"] == OPERATOR_ATTRIBUTED
        ),
        "carried": sorted(
            member for member, entry in citations.items() if entry["disposition"] == CARRIED
        ),
        "proved_here": sorted(
            member
            for member, entry in citations.items()
            if entry["disposition"] not in (CARRIED, CONTINUING)
        ),
        "introduced": _introduced_for_snapshot(
            records_root, intake_snapshot_id, members
        ),
    }
    from menagerie.crawler.partition_proof import PARTITION_PROOF_SCHEMA_VERSION_V3

    if proof_schema_version != PARTITION_PROOF_SCHEMA_VERSION_V3:
        partition["continuing"] = sorted(
            member
            for member, entry in citations.items()
            if entry["disposition"] == CONTINUING
        )
    partition.update(dict(partition_overrides or {}))
    currency: dict[str, Any] = {
        "ledger_heads": len(members),
        "dependency_current": len(members),
        "stale": 0,
        "stale_by_axis": {},
        "stale_reasons": {},
        "awards_stale_on_live_basis": [],
        "award_runs_to_non_runs": list(
            dict(proof["award_transition_census"])["runs_to_non_runs"]
        ),
        "award_runs_to_runs": list(
            dict(proof["award_transition_census"])["runs_to_runs"]
        ),
        "award_transition_instrumentation": dict(proof["award_transition_census"])[
            "instrumentation"
        ],
        "award_transition_windows": dict(proof["award_transition_census"])[
            "coverage_windows"
        ],
    }
    return append_close_object(
        close_object_path(records_root),
        build_close_object(
            campaign_id=campaign_id,
            intake_snapshot_id=intake_snapshot_id,
            intake_snapshot_sha256=intake_snapshot_sha256,
            phase=phase,
            close_kind="leg",
            generation_id=generation_id,
            leg_open_row_sha256=open_digest,
            sources=closing,
            partition_proof_sha256=proof["partition_proof_sha256"],
            predecessor_close_sha256=__import__(
                "menagerie.crawler.close_object", fromlist=["predecessor_close_sha256"]
            ).predecessor_close_sha256(records_root, campaign_id),
            run_id=run_id,
            machine_id="pytest-host",
            closed_at="2026-08-11T00:00:00Z",
            completion_status=completion_status,
            partition_arms=partition,
            currency_census=currency,
            authority_identities=_FIXTURE_AUTHORITY_IDENTITIES,
        ),
    )
