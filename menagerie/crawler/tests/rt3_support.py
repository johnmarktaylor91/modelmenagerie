"""Round-3 red-team rig, extracted from the untracked scratch tree and committed.

The 2026-08-16 red-team round 3 (opus lab) reproduced its two breaks in
``.scratch-redteam3-opus/repo/menagerie/crawler/tests/{rt3_support,test_redteam3_opus}.py``
-- UNTRACKED files that never entered the suite, which is exactly how the
round-2 fix shipped with a RED test that certified the wrong derivation. This
module is that rig, extracted verbatim where possible and extended where the
close-object v2 contract needs it:

* every head can be seeded RECORDED-UNDER a snapshot
  (``dependency_vector.intake_snapshot_id``), because v2's ADJUDICATED and
  CARRIED citations both read that field;
* park and valve rows are built through the REAL builders so positive controls
  exercise the same validation the production censuses apply; and
* closing a leg goes through :mod:`close_object_support`, the one fixture
  spelling of "close this rung honestly", so the rig tracks the production
  close shape instead of freezing a private copy of it.

Mirrors ``test_rung_staging``'s repository fixture exactly (same roster, same
emitted partitions, same empty models shard) so every pin runs on the tree the
gate suites pass on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.intake import IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.partitioner import emit_campaign_partitions, load_campaign_bindings
from menagerie.crawler.tests.close_object_support import record_fixture_close

ROSTER_ROWS: tuple[dict[str, Any], ...] = (
    {"name": "mech-alpha", "zoo": "timm", "variant": ""},
    {"name": "mech-beta", "zoo": "timm", "variant": ""},
    {"name": "mech-gamma", "zoo": "smp", "variant": ""},
    {"name": "mech-delta", "zoo": "smp", "variant": ""},
    {"name": "disco-one", "zoo": "discovered-pytorch", "variant": ""},
    {"name": "classic-one", "zoo": "unregistered-classics-pytorch", "variant": ""},
    {"name": "classic-two", "zoo": "unregistered-classics-pytorch", "variant": ""},
    {"name": "classic-three", "zoo": "unregistered-classics-pytorch", "variant": ""},
    {"name": "classic-four", "zoo": "unregistered-classics-pytorch", "variant": ""},
    {"name": "native-one", "zoo": "discovered-keras", "variant": ""},
)


def build_repo(tmp_path: Path) -> Path:
    """Materialize the four-campaign repository with an EMPTY models shard."""

    repo_root = tmp_path / "repo"
    records_root = repo_root / "menagerie" / "crawler" / "records"
    roster_path = tmp_path / "roster.jsonl"
    roster_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in ROSTER_ROWS))
    emit_campaign_partitions(roster_path, records_root)
    write_models_shard(records_root, [])
    return repo_root


def records_root_of(repo_root: Path) -> Path:
    """Return the campaign records root for one repository."""

    return repo_root / "menagerie" / "crawler" / "records"


def root_binding(repo_root: Path, campaign_id: str):
    """Return the manifest root binding for one campaign."""

    bindings = load_campaign_bindings(records_root_of(repo_root) / "campaigns.json")
    return next(b for b in bindings if b.spec.campaign_id == campaign_id)


def root_snapshot(repo_root: Path, campaign_id: str) -> IntakeSnapshot:
    """Load one campaign's bound full-partition intake snapshot."""

    binding = root_binding(repo_root, campaign_id)
    return load_intake_snapshot(records_root_of(repo_root) / binding.intake_path)


def campaign_member_ids(repo_root: Path, campaign_id: str) -> tuple[str, ...]:
    """Return one campaign's partition members in frozen scheduling order."""

    from menagerie.crawler.rung_staging import partition_member_stable_ids

    return partition_member_stable_ids(
        records_root_of(repo_root),
        root_binding(repo_root, campaign_id),
        root_snapshot(repo_root, campaign_id),
    )


def write_models_shard(records_root: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the sharded models ledger below one records root."""

    path = records_root / "models" / "current-shard.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def head_record(
    stable_id: str,
    status_kind: str = "failed",
    status_code: str = "failed:import",
    *,
    intake_snapshot_id: Optional[str] = None,
    parent_revision: Optional[str] = None,
    record_seq: int = 1,
    run_id: str = "pytest-close",
) -> dict[str, Any]:
    """Build one minimal current model revision, optionally snapshot-scoped."""

    body: dict[str, Any] = {
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
    }
    if intake_snapshot_id is not None:
        body["dependency_vector"] = {"intake_snapshot_id": intake_snapshot_id}
    body["provenance"] = {"producer_run_id": run_id}
    body["record_revision"] = stable_hash(body)
    return body


def seed_heads(
    records_root: Path,
    member_ids: Sequence[str],
    *,
    status_kind: str = "failed",
    status_code: str = "failed:import",
    intake_snapshot_id: Optional[str] = None,
    layout: str = "models/current-shard.jsonl",
    run_id: str = "pytest-close",
) -> None:
    """Append a current head for every member that does not already hold one."""

    path = records_root / layout
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        {
            json.loads(line)["stable_id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if path.is_file()
        else set()
    )
    rows = [
        head_record(
            stable_id,
            status_kind,
            status_code,
            intake_snapshot_id=intake_snapshot_id,
            run_id=run_id,
        )
        for stable_id in member_ids
        if stable_id not in existing
    ]
    if not rows:
        return
    with path.open("ab") as handle:
        handle.write(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def supersede_head(
    records_root: Path,
    stable_id: str,
    *,
    intake_snapshot_id: Optional[str] = None,
    status_code: str = "failed:import",
    layout: str = "models/current-shard.jsonl",
    run_id: str = "pytest-close",
) -> dict[str, Any]:
    """Append a chained superseding revision for one member's current head."""

    path = records_root / layout
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    prior = next(row for row in reversed(rows) if row["stable_id"] == stable_id)
    child = head_record(
        stable_id,
        "failed",
        status_code,
        intake_snapshot_id=intake_snapshot_id,
        parent_revision=prior["record_revision"],
        record_seq=int(prior.get("record_seq", 1)) + 1,
        run_id=run_id,
    )
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(child) + b"\n")
    return child


def append_refusal_park(
    records_root: Path,
    stable_id: str,
    *,
    work_id: str = "w-park",
    campaign_id: str = "c3-classics",
    run_id: str = "pytest-close",
    created_at: str = "2026-08-01T00:00:00Z",
    blocked_parent_revision: Optional[str] = None,
) -> dict[str, Any]:
    """Append one VALID terminal-refusal park row through the real builder."""

    from menagerie.crawler.refusal_parks import build_refusal_park_row, refusal_park_path

    row = build_refusal_park_row(
        stable_id=stable_id,
        work_id=work_id,
        name=stable_id,
        campaign_id=campaign_id,
        run_id=run_id,
        machine_id="pytest-host",
        created_at=created_at,
        refusal_class="RecordAppendRefused",
        record_error_type="ValueError",
        record_error="fixture refusal",
        fallback_error_type=None,
        fallback_error=None,
        intended_status_code="failed:import",
        intended_reason_code=None,
        blocked_parent_revision=blocked_parent_revision,
        candidate_sha256=None,
    )
    path = refusal_park_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
    return row


def append_capacity_valve(
    records_root: Path,
    stable_id: str,
    *,
    work_id: str = "w-valve",
    campaign_id: str = "c3-classics",
    run_id: str = "pytest-close",
    created_at: str = "2026-08-01T00:00:00Z",
) -> dict[str, Any]:
    """Append one minimal host-capacity deferral row the census accepts.

    Shaped to pass :func:`menagerie.crawler.status.machine_deferral_rejection`
    (identity, closed disposition, recheck hint, content digest) -- the same
    gauntlet every partition census applies -- without dragging the full
    ``CapacityAssessment`` machinery into a fixture that is about the CLOSE.
    """

    from menagerie.crawler.host_capacity import (
        CAPACITY_DEFERRAL_SCHEMA_VERSION,
        capacity_deferral_path,
    )
    from menagerie.crawler.status import CAPACITY_DEFERRAL_DISPOSITION

    row: dict[str, Any] = {
        "schema_version": CAPACITY_DEFERRAL_SCHEMA_VERSION,
        "stable_id": stable_id,
        "work_id": work_id,
        "name": stable_id,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "machine_id": "pytest-host",
        "created_at": created_at,
        "disposition": CAPACITY_DEFERRAL_DISPOSITION,
        "recheck_hint": "fixture deferral: re-run on a larger host",
    }
    row["row_sha256"] = stable_hash(row)
    path = capacity_deferral_path(records_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row) + b"\n")
    return row


def append_v1_terminal_close(
    records_root: Path,
    *,
    campaign_id: str,
    intake_snapshot_id: str,
    intake_snapshot_sha256: str,
    phase: Optional[str] = None,
    closed_at: str = "2026-05-01T00:00:00Z",
) -> dict[str, Any]:
    """Append one pre-machinery ``leg-close.v1`` terminal-close record.

    The pilot legs that predate leg generations closed under ``leg-close.v1``;
    that row licenses nothing itself (it predates the window, the census, and
    the sealed proof) but it IS the historical record that the rung terminally
    closed -- the existence proof the backfill era gate demands before minting
    a v2 close for it.
    """

    from menagerie.crawler.close_object import (
        append_close_object,
        close_object_path,
        predecessor_close_sha256,
        verifier_identity,
    )

    body: dict[str, Any] = {
        "schema_version": "menagerie.crawler.leg-close.v1",
        "campaign_id": campaign_id,
        "intake_snapshot_id": intake_snapshot_id,
        "intake_snapshot_sha256": intake_snapshot_sha256,
        "phase": phase,
        "run_id": "pilot-historical",
        "machine_id": "pilot-host",
        "closed_at": closed_at,
        "completion_status": "complete",
        "predecessor_close_sha256": predecessor_close_sha256(
            records_root, campaign_id
        ),
        "ledgers": {
            name: {"rows": 0, "ledger_sha256": stable_hash({}), "tail_sha256": None}
            for name in ("models", "attempts", "gates", "artifacts")
        },
        "partition": {
            "intake": 0,
            "valid": True,
            "missing": [],
            "extra": [],
            "duplicates": [],
            "invalid": [],
            "buckets": {},
            "machine_deferred": {},
            "grant_pending": [],
            "refusal_parked": [],
        },
        "currency": {"stale_by_axis": {}},
        "verifier_identity": verifier_identity({}),
        "authority_identities": {},
    }
    row = dict(body)
    row["close_sha256"] = stable_hash(body)
    return append_close_object(close_object_path(records_root), row)


def void_refusal_park(records_root: Path, row: dict[str, Any]) -> None:
    """Void one refusal-park row through the real void builder."""

    from menagerie.crawler.refusal_parks import build_refusal_park_void, refusal_park_path

    void = build_refusal_park_void(
        void_of=row["row_sha256"],
        reason="fixture void",
        voided_by="pytest",
        created_at="2026-08-02T00:00:00Z",
    )
    with refusal_park_path(records_root).open("ab") as handle:
        handle.write(canonical_json_bytes(void) + b"\n")


def fixture_composition_attestation(
    records_root: Path,
    campaign_id: str,
    *,
    stable_ids: Optional[Sequence[str]] = None,
    count: Optional[int] = None,
    label: str = "fixture",
    snapshot_id: Optional[str] = None,
    snapshot_sha256: Optional[str] = None,
):
    """Synthesize one lawful PD-7-bearing composition attestation for a stage.

    SOL-02 made the PD-7 consumption non-optional for a ``successor_rung``
    stage, so every fixture staging the successor shape presents a v2
    attestation whose digest binds ``pd7.status == SATISFIED``. The member set
    and planned identity are computed through the SAME production derivations
    the door recomputes, so the attestation always describes the stage.
    """

    from menagerie.crawler.rung_staging import (
        _partition_prefix_ids,
        build_composition_attestation,
        planned_subset_identity,
        resolve_verified_root,
    )
    from menagerie.crawler.rung_stratification import LADDER_BULK_STRATUM

    if (stable_ids is None) == (count is None):
        raise ValueError(f"{label}: exactly one of stable_ids/count selects the members")
    if count is not None:
        root_binding, root_snapshot = resolve_verified_root(records_root, campaign_id)
        member_ids: tuple[str, ...] = _partition_prefix_ids(
            records_root, root_binding, root_snapshot, count
        )
    else:
        member_ids = tuple(str(value) for value in stable_ids or ())
    if snapshot_id is None or snapshot_sha256 is None:
        # The staged planned identity; a register-door fixture passes the
        # adopted snapshot's own verified identity instead.
        snapshot_id, snapshot_sha256 = planned_subset_identity(
            records_root, campaign_id, member_ids
        )
    return build_composition_attestation(
        campaign_id=campaign_id,
        size=len(member_ids),
        seed=0,
        member_strata={stable_id: LADDER_BULK_STRATUM for stable_id in member_ids},
        snapshot_id=snapshot_id,
        snapshot_sha256=snapshot_sha256,
        pd7_disposition={
            "status": "SATISFIED",
            "source_attestation_sha256": "sha256:" + "7d" * 32,
        },
    )


def stage(
    repo_root: Path,
    campaign_id: str,
    *,
    stable_ids: Optional[Sequence[str]] = None,
    count: Optional[int] = None,
    derived_from: Optional[str] = None,
    label: str = "rt3-rung",
    award_authorities: Optional[dict[str, str]] = None,
    rung_class: Optional[str] = None,
    composition_attestation: Optional[object] = None,
):
    """Stage one rung through the library API with fixed provenance.

    Fixtures model the composed next-rung shape, so the staging-door class
    declaration defaults to ``successor_rung`` here (the production door has
    no default); pass ``rung_class`` to model another class. A successor stage
    synthesizes the SOL-02 PD-7-bearing attestation unless the caller presents
    its own.
    """

    from menagerie.crawler.rung_staging import RUNG_CLASS_SUCCESSOR, stage_rung

    resolved_class = rung_class if rung_class is not None else RUNG_CLASS_SUCCESSOR
    attestation = composition_attestation
    if attestation is None and resolved_class == RUNG_CLASS_SUCCESSOR:
        attestation = fixture_composition_attestation(
            records_root_of(repo_root),
            campaign_id,
            stable_ids=stable_ids,
            count=count,
            label=label,
        )
    return stage_rung(
        records_root_of(repo_root),
        campaign_id=campaign_id,
        stable_ids=tuple(stable_ids) if stable_ids is not None else None,
        count=count,
        derived_from_snapshot_id=derived_from,
        label=label,
        registered_by="pytest",
        created_at="2026-08-16T00:00:00Z",
        award_authorities=award_authorities,
        rung_class=resolved_class,
        composition_attestation=attestation,
    )


def members_of(repo_root: Path, snapshot_id: str) -> tuple[str, ...]:
    """Return the durable membership of one registered rung snapshot."""

    snapshot = load_intake_snapshot(records_root_of(repo_root) / "intake" / snapshot_id)
    return tuple(sorted(item.stable_id for item in snapshot.items))


def close_leg_honestly(
    repo_root: Path,
    staged: Any,
    campaign_id: str,
    *,
    run_and_record: bool = True,
) -> Any:
    """Open a leg for one staged rung, optionally record every member, close.

    THE one spelling of "this rung's leg really ran and closed" for every pin
    in the matrix. ``run_and_record=False`` opens a real leg and closes it
    WITHOUT recording a byte -- under the v2 contract that close must be
    honest about it (all-CARRIED where a real ancestor resolution exists,
    refused where none does), never a claim of run outcomes.
    """

    members = members_of(repo_root, staged.registration.snapshot_id)
    return record_fixture_close(
        records_root_of(repo_root),
        campaign_id=campaign_id,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_ids=members,
        seed_member_heads=run_and_record,
    )


def models_ledger_digest(records_root: Path) -> str:
    """Return the sha256 of the sharded models ledger, for before/after proof."""

    import hashlib

    path = records_root / "models" / "current-shard.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"
