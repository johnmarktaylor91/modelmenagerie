"""Real two-shard merge over copies of the live pilot corpus (C-SHARD-MERGE gate).

These tests run the FULL production merge path -- reducer replay, valve-row
dereference, review-hold disclosure, corpus union, transactional views -- over
byte-copies of the actual pilot records root. Nothing is monkeypatched. The
pilot root is read-only input; every write lands in the test's tmp tree. The
expectations are DERIVED from the copied bytes (which models hold records,
which are valve-claimed, which are review-held), so the tests stay truthful as
the live pilot advances; they skip on hosts without the pilot checkout.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from menagerie.local_config import local_value
from menagerie.crawler.campaign_merge import CampaignMergeError
from menagerie.crawler.driver_admission import _persisted_deferral_rows
from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.intake import _snapshot_id, load_intake_snapshot
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import default_ledger_paths
from menagerie.crawler.rung_staging import (
    append_rung_registration,
    build_rung_registration,
    rung_registry_path,
)
from menagerie.crawler.status import machine_deferral_rejection
from menagerie.crawler.tools.handoff_bundle import build_handoff_manifest
from menagerie.crawler.tools.merge_host_shards import (
    HOST_SHARD_MERGE_MANIFEST_NAME,
    HostShardSource,
    merge_host_shards,
)

_PILOT_RECORDS = Path(
    local_value("PILOT_RECORDS", "~/.local/share/menagerie/pilot/crawler/records")
).expanduser()
_CAMPAIGN_ID = "c1-mech"


def _require_pilot_records() -> Path:
    """Return the live pilot records root, or skip when absent.

    Returns
    -------
    pathlib.Path
        Existing pilot records root.
    """

    if not _PILOT_RECORDS.is_dir():
        pytest.skip("live pilot records root is not present on this host")
    return _PILOT_RECORDS


def _active_snapshot_dir(records_root: Path) -> Path:
    """Return the pilot's newest registered full-rung snapshot directory.

    Parameters
    ----------
    records_root:
        Pilot records root copy.

    Returns
    -------
    pathlib.Path
        Snapshot directory every current model chain is accounted under.
    """

    rows = [
        json.loads(line)
        for line in rung_registry_path(records_root).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    covering = [
        row
        for row in rows
        if row["campaign_id"] == _CAMPAIGN_ID
        and (records_root / "intake" / row["snapshot_id"]).is_dir()
    ]
    assert covering, "pilot registry names no materialized c1 rung snapshot"
    best = max(covering, key=lambda row: row["member_count"])
    return records_root / "intake" / str(best["snapshot_id"])


def _copy_pilot(destination: Path) -> Path:
    """Copy the live pilot records tree into a clone-shaped root.

    Parameters
    ----------
    destination:
        Temporary clone root.

    Returns
    -------
    pathlib.Path
        Copied records root.
    """

    records_root = destination / "menagerie" / "crawler" / "records"
    shutil.copytree(
        _PILOT_RECORDS,
        records_root,
        ignore=shutil.ignore_patterns("*.lock"),
    )
    return records_root


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic JSONL bytes.

    Parameters
    ----------
    path:
        Output path.
    rows:
        JSON object rows.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _filter_shard_records(records_root: Path, keep: set[str]) -> None:
    """Restrict one pilot copy to the models a host shard would own.

    Rows are kept or dropped WHOLE (per-row content digests stay valid); gate
    rows survive when any adjudicated item belongs to the shard.

    Parameters
    ----------
    records_root:
        Copied records root.
    keep:
        Shard-assigned stable IDs.
    """

    ledgers = default_ledger_paths(records_root)
    # Artifact rows are content-addressed and may be recorded once for bytes
    # several models share; they carry no ownership claim, so the shard copy
    # keeps the whole ledger exactly as the full single-source merge does.
    for path in (ledgers.models, ledgers.attempts):
        rows = [
            row for row in scan_jsonl(path, validate=False) if str(row.get("stable_id")) in keep
        ]
        _write_jsonl(path, rows)
    gate_rows = []
    for row in scan_jsonl(ledgers.gates, validate=False):
        items = row.get("items")
        if isinstance(items, list) and any(str(item.get("stable_id")) in keep for item in items):
            gate_rows.append(row)
    _write_jsonl(ledgers.gates, gate_rows)
    for extension in sorted((records_root / "intake-extensions").glob("*.jsonl")):
        rows = [
            row
            for row in (
                json.loads(line)
                for line in extension.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if str(row.get("stable_id")) in keep
        ]
        _write_jsonl(extension, rows)
    grants = records_root / "operational" / "requeue-grants.jsonl"
    if grants.is_file():
        rows = [
            row
            for row in (
                json.loads(line)
                for line in grants.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if str(row.get("stable_id")) in keep
        ]
        _write_jsonl(grants, rows)


def _subset_snapshot(source_intake: Path, ids: set[str], out_parent: Path) -> Path:
    """Emit a validated subset snapshot using the production snapshot basis.

    Parameters
    ----------
    source_intake:
        Source intake snapshot directory.
    ids:
        Stable IDs to keep.
    out_parent:
        Snapshot parent directory.

    Returns
    -------
    pathlib.Path
        New subset snapshot root.
    """

    source = load_intake_snapshot(source_intake)
    manifest = json.loads((source.root / "manifest.json").read_text(encoding="utf-8"))
    items = [item for item in manifest["items"] if item["stable_id"] in ids]
    basis = {
        "format": manifest["format"],
        "sources": manifest["sources"],
        "stable_ids": manifest["stable_ids"],
        "items": items,
    }
    snapshot_sha256 = stable_hash(basis)
    snapshot_id = _snapshot_id(snapshot_sha256)
    root = out_parent / snapshot_id
    for name in manifest["sources"]:
        data = (source.root / "sources" / name).read_bytes()
        (root / "sources" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "sources" / name).write_bytes(data)
    out_manifest = {
        **basis,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "item_count": len(items),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_bytes(canonical_json_bytes(out_manifest) + b"\n")
    _write_jsonl(root / "items.jsonl", items)
    loaded = load_intake_snapshot(root)
    assert {item.stable_id for item in loaded.items} == ids
    return root


def _register_shard_rung(records_root: Path, shard_intake: Path, parent_snapshot_id: str) -> None:
    """Register a shard subset snapshot in the shard copy's rung registry.

    A real two-host leg stages and registers its shard rung on each host; the
    registration is what licenses the T4-T1 prior-lineage currency carve-out.

    Parameters
    ----------
    records_root:
        Shard-local records root.
    shard_intake:
        Shard subset snapshot directory (already under ``records/intake``).
    parent_snapshot_id:
        Snapshot the shard subset was drawn from.
    """

    snapshot = load_intake_snapshot(shard_intake)
    registration = build_rung_registration(
        campaign_id=_CAMPAIGN_ID,
        snapshot_id=snapshot.snapshot_id,
        snapshot_sha256=snapshot.snapshot_sha256,
        derived_from_snapshot_id=parent_snapshot_id,
        member_count=len(snapshot.items),
        label="two-host pilot shard (TW-G composition test)",
        registered_by="test_merge_pilot_shards",
        created_at="2026-08-12T00:00:00Z",
    )
    append_rung_registration(rung_registry_path(records_root), registration)


def _gate_bound_components(records_root: Path, assigned: set[str]) -> list[list[str]]:
    """Group assigned models into shard-splittable connected components.

    Two production constraints bind models to one shard:

    * Adjudication gate rows batch several models and are content-addressed
      (never editable), so a post-hoc shard split must keep every gate batch
      on one shard -- exactly as a real two-host leg would, where each host
      runs and records its own batches.
    * The merge's environment axis (``_assert_environment_axis``) demands
      that shards agree on every environment family they both carry: a
      shard's latest durable generation for a family is the generation of
      the family's LAST attempt row it holds. The pilot's history spans
      generation rotations, so a member whose last family attempt predates
      the pilot's own current generation (a "laggard") must ride with the
      member holding the family's final attempt -- a real lagging host owes
      a re-validation leg before merging, which a paper split of a single
      host's honest history cannot run.

    Parameters
    ----------
    records_root:
        Pilot records root copy.
    assigned:
        Assigned stable IDs.

    Returns
    -------
    list[list[str]]
        Components, largest first.
    """

    parent = {stable_id: stable_id for stable_id in assigned}

    def find(stable_id: str) -> str:
        while parent[stable_id] != stable_id:
            parent[stable_id] = parent[parent[stable_id]]
            stable_id = parent[stable_id]
        return stable_id

    for row in scan_jsonl(default_ledger_paths(records_root).gates, validate=False):
        items = row.get("items")
        members = [
            str(item.get("stable_id"))
            for item in (items if isinstance(items, list) else [])
            if str(item.get("stable_id")) in parent
        ]
        for other in members[1:]:
            parent[find(other)] = find(members[0])
    family_rows: dict[str, list[tuple[str, str]]] = {}
    for attempt in scan_jsonl(default_ledger_paths(records_root).attempts, validate=False):
        stable_id = str(attempt.get("stable_id"))
        if stable_id not in parent:
            continue
        environment = attempt.get("environment")
        identities = attempt.get("identities")
        family = environment.get("family") if isinstance(environment, Mapping) else None
        generation = identities.get("environment") if isinstance(identities, Mapping) else None
        if isinstance(family, str) and family and isinstance(generation, str) and generation:
            family_rows.setdefault(family, []).append((stable_id, generation))
    for rows in family_rows.values():
        last_generation_by_model = dict(rows)
        anchor, final_generation = rows[-1]
        for stable_id, generation in last_generation_by_model.items():
            if generation != final_generation:
                parent[find(stable_id)] = find(anchor)
    groups: dict[str, list[str]] = {}
    for stable_id in sorted(assigned):
        groups.setdefault(find(stable_id), []).append(stable_id)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def _pilot_census(records_root: Path, snapshot_dir: Path) -> dict[str, Any]:
    """Derive the accounted/unaccounted membership of one pilot copy.

    The derivation mirrors the ownership rule from the OUTSIDE: a model is
    accounted when it holds any model-ledger row (current OR stale history) or
    a durable valve row that survives the completion partition's fail-closed
    validation.

    Parameters
    ----------
    records_root:
        Copied pilot records root.
    snapshot_dir:
        Active rung snapshot directory.

    Returns
    -------
    dict[str, Any]
        Assigned, processed, valve-claimed, and unaccounted stable IDs.
    """

    snapshot = load_intake_snapshot(snapshot_dir)
    assigned = {item.stable_id for item in snapshot.items}
    ledgers = default_ledger_paths(records_root)
    processed = {str(row.get("stable_id")) for row in scan_jsonl(ledgers.models, validate=False)}
    shim = SimpleNamespace(paths=SimpleNamespace(ledgers=ledgers))
    claimed = set()
    for stable_id, rows in _persisted_deferral_rows(shim).items():
        if any(machine_deferral_rejection(stable_id, row) is None for row in rows):
            claimed.add(stable_id)
    return {
        "snapshot": snapshot,
        "assigned": assigned,
        "processed": processed & assigned,
        "claimed": claimed & assigned,
        "unaccounted": assigned - processed - claimed,
    }


# RC-3: the two real-pilot merges are a rig (about 99 s together); they run
# under the crawler-rigs gate when their inputs move, not in every tier.
pytestmark = pytest.mark.rig


def test_real_pilot_two_shard_merge_passes_end_to_end(tmp_path: Path) -> None:
    """Two shards cut from the real pilot corpus merge through the full path."""

    _require_pilot_records()
    reference_records = _copy_pilot(tmp_path / "reference")
    snapshot_dir = _active_snapshot_dir(reference_records)
    census = _pilot_census(reference_records, snapshot_dir)
    accounted = sorted(census["assigned"] - census["unaccounted"])
    assert len(accounted) >= 4, "pilot corpus too small to split"
    shard_ids: dict[str, set[str]] = {"host-a": set(), "host-b": set()}
    for component in _gate_bound_components(reference_records, set(accounted)):
        smaller = min(shard_ids, key=lambda shard: len(shard_ids[shard]))
        shard_ids[smaller].update(component)
    assert shard_ids["host-a"] and shard_ids["host-b"], "gate batches span the whole corpus"

    sources = []
    for shard_id, members in shard_ids.items():
        clone = tmp_path / shard_id
        records_root = _copy_pilot(clone)
        _filter_shard_records(records_root, members)
        shard_intake = _subset_snapshot(snapshot_dir, members, records_root / "intake")
        _register_shard_rung(records_root, shard_intake, snapshot_dir.name)
        manifest_path = clone / "handoff-manifest.json"
        build_handoff_manifest(
            records_root,
            shard_id=shard_id,
            campaign_id=_CAMPAIGN_ID,
            manifest_path=manifest_path,
        )
        sources.append(
            HostShardSource(
                shard_id,
                records_root,
                clone / ".crawl-local",
                shard_intake,
                handoff_manifest=manifest_path,
            )
        )
    expected_intake = _subset_snapshot(snapshot_dir, set(accounted), tmp_path / "intake")

    before = {
        source.shard_id: {
            path.relative_to(source.records_root).as_posix(): hash_bytes(path.read_bytes())
            for path in sorted(source.records_root.rglob("*"))
            if path.is_file()
        }
        for source in sources
    }

    result = merge_host_shards(
        reference_records / "campaigns.json",
        _CAMPAIGN_ID,
        tuple(sources),
        tmp_path / "merged",
        expected_intake_root=expected_intake,
    )

    report = result.report
    assert report["merge_status"].startswith("complete")
    assert report["proof"]["assigned_ids_terminal_or_deferral_claimed_per_shard"] is True
    assert report["proof"]["environment_axis_consistent_across_shards"] is True
    assert report["proof"]["source_ledgers_read_only"] is True

    merged_ids = {record["stable_id"] for record in result.current_records}
    claimed_ids: set[str] = set()
    review_held: set[str] = set()
    stale_ids: set[str] = set()
    for shard_report in report["sources"].values():
        for stable_ids in shard_report["machine_deferred"].values():
            claimed_ids.update(stable_ids)
        review_held.update(shard_report["review_held_ids"])
        stale_ids.update(shard_report["stale_reasons"])
    # Every accounted model is covered exactly once: a current terminal, a
    # machine-deferral claim, or a carried stale record awaiting its re-drive.
    assert merged_ids | claimed_ids | stale_ids == set(accounted)
    assert not merged_ids & claimed_ids
    assert not merged_ids & stale_ids
    # The pilot's valve-only models (deferral rows, no model row BY DESIGN) are owned.
    valve_only = census["assigned"] - census["processed"] - census["unaccounted"]
    assert valve_only <= claimed_ids
    # Review-held terminals merged and are disclosed, not refused.
    derived_review_held = {
        stable_id
        for stable_id, record in (
            (record["stable_id"], record) for record in result.current_records
        )
        if record.get("status", {}).get("human_review", {}).get("required") is True
    }
    assert review_held == derived_review_held
    # Carried stale records are disclosed in the merged partition census.
    assert set(report["partition"]["stale_carried"]) == stale_ids

    # Transactional commit manifest covers every view byte-for-byte.
    commit_manifest = json.loads(
        (tmp_path / "merged" / HOST_SHARD_MERGE_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert commit_manifest["committed"] is True
    for relative, digest in commit_manifest["views"].items():
        assert hash_bytes((tmp_path / "merged" / relative).read_bytes()) == digest

    # The shard trees the merge read are byte-identical afterwards.
    after = {
        source.shard_id: {
            path.relative_to(source.records_root).as_posix(): hash_bytes(path.read_bytes())
            for path in sorted(source.records_root.rglob("*"))
            if path.is_file()
        }
        for source in sources
    }
    assert after == before


def test_real_pilot_unaccounted_model_still_refuses(tmp_path: Path) -> None:
    """A model with neither a record nor a valid valve row keeps refusing.

    The hole is synthesized on the copy by discarding one valve-only model's
    deferral rows: its ONLY truthful accounting disappears, and the merge must
    name it rather than read the shard as owned.
    """

    _require_pilot_records()
    clone = tmp_path / "single"
    records_root = _copy_pilot(clone)
    snapshot_dir = _active_snapshot_dir(records_root)
    census = _pilot_census(records_root, snapshot_dir)
    valve_only = sorted(census["claimed"] - census["processed"])
    if not valve_only:
        pytest.skip("pilot corpus currently has no valve-only models")
    hole = valve_only[0]
    for extension in sorted((records_root / "intake-extensions").glob("*.jsonl")):
        rows = [
            row
            for row in (
                json.loads(line)
                for line in extension.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if str(row.get("stable_id")) != hole
        ]
        _write_jsonl(extension, rows)

    source = HostShardSource(
        "pilot",
        records_root,
        clone / ".crawl-local",
        snapshot_dir,
    )
    with pytest.raises(CampaignMergeError, match=hole):
        merge_host_shards(
            records_root / "campaigns.json",
            _CAMPAIGN_ID,
            (source,),
            tmp_path / "merged",
            expected_intake_root=snapshot_dir,
        )
