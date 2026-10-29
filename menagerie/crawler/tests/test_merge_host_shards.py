"""Tests for same-campaign host-shard projection merges."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import pytest

from menagerie.local_config import local_value
from menagerie.crawler.authority import build_authority_context
from menagerie.crawler.campaign_merge import CampaignMergeError
from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.environment_rebuild import (
    append_rebuild_deferral_row,
    build_rebuild_deferral_row,
    rebuild_deferral_path,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.intake import _snapshot_id, load_intake_snapshot
from menagerie.crawler.models import CompletenessReport
from menagerie.crawler.partitioner import (
    CampaignBinding,
    emit_campaign_partitions,
    refresh_campaign_manifest,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import (
    CanonicalReducer,
    DependencyCurrencyProjection,
    default_ledger_paths,
)
from menagerie.crawler.status import funnel_counts, partition_report
from menagerie.crawler.constants import TEXT_CORPUS_INDEX_SCHEMA_VERSION
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_failed_attempt,
    make_model,
    make_operational_event,
)
from menagerie.crawler.text_corpus import (
    corpus_payload_path,
    corpus_root,
    corpus_row_id,
)
from menagerie.crawler.fetcher import publish_cas_object
from menagerie.crawler.tools.merge_host_shards import (
    ENV_AXIS_DIVERGENT_ERROR,
    HOST_SHARD_COLLISION_ERROR,
    HOST_SHARD_MERGE_MANIFEST_NAME,
    HostShardSource,
    merge_host_shards,
)


def _seed_corpus(records_root: Path, stable_id: str, document: bytes) -> None:
    """Publish one document plus its index row into a shard corpus."""

    digest = hash_bytes(document)
    publish_cas_object(corpus_root(records_root), digest, document)
    ledger = JsonlLedger(
        records_root / "text-corpus" / "corpus-index.jsonl", TEXT_CORPUS_INDEX_SCHEMA_VERSION
    )
    try:
        ledger.append(
            {
                "schema_version": TEXT_CORPUS_INDEX_SCHEMA_VERSION,
                "corpus_row_id": corpus_row_id(
                    stable_id=stable_id,
                    content_sha256=digest,
                    source_id="impl-seeded",
                    role="implementation",
                    origin="custody-promotion",
                    work_id=f"work-{stable_id}",
                ),
                "stable_id": stable_id,
                "work_id": f"work-{stable_id}",
                "source_id": "impl-seeded",
                "role": "implementation",
                "origin": "custody-promotion",
                "content_sha256": digest,
                "bytes_len": len(document),
                "media_type": "text/x-python",
                "url": "https://example.test/impl.py",
                "retrieved_at": "2026-08-11T00:00:00.000000Z",
                "license_disposition": "manifest-only",
            }
        )
    finally:
        ledger.close()


_RUNG8_RECORDS = (
    Path(local_value("DURABLE_RUNG_ARCHIVE", "~/.local/share/menagerie/rung-archive")).expanduser()
    / "rung8-19a72e4a-complete/records"
)


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


def _copy_archive_records(destination: Path) -> Path:
    """Copy the real rung-8 records archive into a clone-shaped root.

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
    shutil.copytree(_RUNG8_RECORDS, records_root)
    return records_root


def _refreshed_archive_manifest(out_parent: Path) -> Path:
    """Return the rung-8 manifest re-projected onto the live frozen specs.

    The archive is byte-frozen history: its ``campaigns.json`` records the
    spec projection as of rung 8 (c1-mech authored by claude-sonnet), which the
    loader correctly refuses after any frozen-spec change. The sanctioned
    remediation -- exactly what the pilot runs before its next launch
    (``stage_rung refresh-manifest``, OPS-10) -- re-emits the spec-projection
    fields while preserving every partition and intake binding, so this fixture
    derives from the live specs instead of pinning a model string that can go
    stale again.

    Parameters
    ----------
    out_parent:
        Temporary directory owning the refreshed copy.

    Returns
    -------
    pathlib.Path
        Refreshed ``campaigns.json`` path.
    """

    manifest_path = out_parent / "campaigns.json"
    shutil.copyfile(_RUNG8_RECORDS / "campaigns.json", manifest_path)
    refresh_campaign_manifest(manifest_path)
    return manifest_path


def _subset_snapshot(source_intake: Path, ids: set[str], out_parent: Path) -> Path:
    """Emit a validated subset snapshot using the production snapshot basis.

    Parameters
    ----------
    source_intake:
        Source c1 intake snapshot.
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


def _filter_records(records_root: Path, ids: set[str]) -> None:
    """Filter canonical rung-8 ledgers to the selected stable IDs.

    Parameters
    ----------
    records_root:
        Copied records root.
    ids:
        Stable IDs to keep.
    """

    ledgers = default_ledger_paths(records_root)
    for path in (ledgers.models, ledgers.attempts, ledgers.artifacts):
        rows = [row for row in scan_jsonl(path, validate=False) if str(row.get("stable_id")) in ids]
        _write_jsonl(path, rows)
    gate_rows = []
    for row in scan_jsonl(ledgers.gates, validate=False):
        items = row.get("items")
        if not isinstance(items, list):
            continue
        kept = [item for item in items if str(item.get("stable_id")) in ids]
        if kept:
            edited = dict(row)
            edited["items"] = kept
            gate_rows.append(edited)
    _write_jsonl(ledgers.gates, gate_rows)


def _archive_ids() -> list[str]:
    """Return stable IDs from the real rung-8 model ledger.

    Returns
    -------
    list[str]
        Ledger-order stable IDs.
    """

    return [
        str(row["stable_id"])
        for row in scan_jsonl(_RUNG8_RECORDS / "models" / "current-shard.jsonl", validate=False)
    ]


def _source_bytes(source: HostShardSource) -> Mapping[str, bytes]:
    """Snapshot source bytes for read-only assertions.

    Parameters
    ----------
    source:
        Host shard source.

    Returns
    -------
    Mapping[str, bytes]
        Relative file path to exact bytes.
    """

    return {
        path.relative_to(source.records_root).as_posix(): path.read_bytes()
        for path in source.records_root.rglob("*")
        if path.is_file()
    }


def _fake_projection(
    rows_by_root: Mapping[Path, Sequence[Mapping[str, Any]]],
) -> Callable[..., DependencyCurrencyProjection]:
    """Build a policy-drift-neutral projection stub over archived rows.

    Parameters
    ----------
    rows_by_root:
        Records root to rows that the real archive contributed.

    Returns
    -------
    Callable[..., DependencyCurrencyProjection]
        Replacement projector for tests that do not rewrite archived dependency vectors.
    """

    def project(ledgers: Any, **_kwargs: Any) -> DependencyCurrencyProjection:
        """Return current rows for one copied archive shard."""

        rows = rows_by_root[Path(ledgers.models).parents[1]]
        return DependencyCurrencyProjection({str(row["stable_id"]): dict(row) for row in rows}, {})

    return project


def _archived_completion(
    intake_ids: Sequence[str],
    current_records: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    machine_deferrals: Mapping[str, Mapping[str, Any]] | None = None,
) -> CompletenessReport:
    """Return partition-only completeness for archived policy-drift fixtures.

    Parameters
    ----------
    intake_ids:
        Stable IDs assigned to the shard or full subset.
    current_records:
        Projected current records.

    Returns
    -------
    CompletenessReport
        Complete report when the stable-ID partition itself is exact.
    """

    partition = partition_report(intake_ids, current_records)
    records = current_records.values() if isinstance(current_records, Mapping) else current_records
    return CompletenessReport(
        partition=partition,
        incomplete_by_issue={},
        workflow_counts={},
        funnel_counts=funnel_counts(records),
        complete=partition.valid,
    )


def test_real_archive_shards_union_and_preserve_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real rung-8 rows merge as a stable-ID union with source fingerprints."""

    ids = _archive_ids()
    shard_a_ids = set(ids[:10])
    shard_b_ids = set(ids[10:])
    clone_a = tmp_path / "host-a"
    clone_b = tmp_path / "host-b"
    records_a = _copy_archive_records(clone_a)
    records_b = _copy_archive_records(clone_b)
    _filter_records(records_a, shard_a_ids)
    _filter_records(records_b, shard_b_ids)
    source_intake = _RUNG8_RECORDS / "intake" / "intake-61d2ae3aed9ea0cbb3f3"
    expected = _subset_snapshot(source_intake, set(ids), tmp_path / "intake")
    intake_a = _subset_snapshot(source_intake, shard_a_ids, tmp_path / "intake")
    intake_b = _subset_snapshot(source_intake, shard_b_ids, tmp_path / "intake")
    sources = (
        HostShardSource("host-a", records_a, clone_a / ".crawl-local", intake_a),
        HostShardSource("host-b", records_b, clone_b / ".crawl-local", intake_b),
    )
    rows_by_root = {
        records_a: scan_jsonl(records_a / "models" / "current-shard.jsonl", validate=False),
        records_b: scan_jsonl(records_b / "models" / "current-shard.jsonl", validate=False),
    }
    monkeypatch.setattr(
        "menagerie.crawler.tools.merge_host_shards.project_dependency_current",
        _fake_projection(rows_by_root),
    )
    monkeypatch.setattr(
        "menagerie.crawler.tools.merge_host_shards.checkpoint_consistency_report",
        _archived_completion,
    )
    monkeypatch.setattr(
        "menagerie.crawler.tools.merge_host_shards.completeness_report",
        _archived_completion,
    )
    shared_document = b"shared corpus document bytes\n"
    _seed_corpus(records_a, "m-shard-a", shared_document)
    _seed_corpus(records_b, "m-shard-b", shared_document)
    before = {source.shard_id: _source_bytes(source) for source in sources}

    result = merge_host_shards(
        _refreshed_archive_manifest(tmp_path),
        "c1-mech",
        sources,
        tmp_path / "merged",
        expected_intake_root=expected,
    )

    corpus = result.report["text_corpus"]
    assert corpus["payloads_copied"] == 1
    assert corpus["payloads_shared"] == 1
    assert corpus["rows_appended"] == 2
    merged_payload = corpus_payload_path(tmp_path / "merged", hash_bytes(shared_document))
    assert merged_payload.read_bytes() == shared_document
    assert [record["stable_id"] for record in result.current_records] == sorted(ids)
    assert result.report["proof"]["assigned_stable_ids_pairwise_disjoint"] is True
    assert result.report["proof"]["source_ledgers_read_only"] is True
    assert result.report["sources"]["host-a"]["assigned_count"] == 10
    assert result.report["sources"]["host-b"]["assigned_count"] == 10
    assert result.report["total"]["terminal_count"] == 20
    assert (tmp_path / "merged" / "host-shard-merge-report.json").is_file()
    assert {source.shard_id: _source_bytes(source) for source in sources} == before


def test_real_archive_overlapping_stable_ids_refuse_with_named_collision(
    tmp_path: Path,
) -> None:
    """An overlapping shard assignment fails before projection."""

    ids = _archive_ids()
    overlap = set(ids[:3])
    clone_a = tmp_path / "host-a"
    clone_b = tmp_path / "host-b"
    records_a = _copy_archive_records(clone_a)
    records_b = _copy_archive_records(clone_b)
    _filter_records(records_a, set(ids))
    _filter_records(records_b, overlap)
    source_intake = _RUNG8_RECORDS / "intake" / "intake-61d2ae3aed9ea0cbb3f3"
    expected = _subset_snapshot(source_intake, set(ids), tmp_path / "intake")
    intake_a = _subset_snapshot(source_intake, set(ids), tmp_path / "intake")
    intake_b = _subset_snapshot(source_intake, overlap, tmp_path / "intake")

    with pytest.raises(CampaignMergeError, match=HOST_SHARD_COLLISION_ERROR):
        merge_host_shards(
            _refreshed_archive_manifest(tmp_path),
            "c1-mech",
            (
                HostShardSource("host-a", records_a, clone_a / ".crawl-local", intake_a),
                HostShardSource("host-b", records_b, clone_b / ".crawl-local", intake_b),
            ),
            tmp_path / "merged",
            expected_intake_root=expected,
        )
    assert not (tmp_path / "merged").exists()


_REPO_ROOT = Path(__file__).parents[3]
_ROSTER = _REPO_ROOT / "menagerie" / "data" / "crawl_roster.jsonl"
_NOW = "2026-08-12T00:00:00.000000Z"


@dataclass(frozen=True)
class _ShardFixture:
    """Synthetic single-campaign, two-host-shard merge fixture."""

    manifest: Path
    campaign_id: str
    binding: CampaignBinding
    expected_intake: Path
    sources: tuple[HostShardSource, ...]
    shard_ids: Mapping[str, tuple[str, ...]]
    output_root: Path


def _fixture_roster_rows() -> list[Mapping[str, Any]]:
    """Return six c1 rows plus one row for each other campaign classifier.

    Returns
    -------
    list[Mapping[str, Any]]
        Production-shaped roster rows.
    """

    rows = [json.loads(line) for line in _ROSTER.read_text(encoding="utf-8").splitlines() if line]
    picked = [row for row in rows if row["zoo"] == "torchvision"][:6]
    for zoo in ("discovered-pytorch", "unregistered-classics-pytorch", "discovered-tensorflow"):
        picked.append(next(row for row in rows if row["zoo"] == zoo))
    return picked


def _mint_terminal(
    records_root: Path,
    binding: CampaignBinding,
    snapshot: Any,
    stable_id: str,
    *,
    review_held: bool = False,
) -> None:
    """Append one production-reduced failed:source terminal for one model.

    Parameters
    ----------
    records_root:
        Shard-local records root.
    binding:
        Frozen campaign binding.
    snapshot:
        Authority intake snapshot the record is minted under.
    stable_id:
        Model identity.
    review_held:
        Whether the terminal carries a live human-review hold.
    """

    context = build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=binding.spec.author_model,
        author_version="current",
        checker_model=binding.spec.checker_model,
        checker_version="current",
    )
    attempt_id = f"attempt-{stable_id}"
    attempt = make_failed_attempt(stable_id, attempt_id=attempt_id)
    attempt.pop("ledger_seq", None)
    model = bind_terminal_attempts(
        make_model(stable_id, status_code="failed:source", attempt_id=attempt_id),
        [attempt],
    )
    model.pop("record_seq", None)
    model["provenance"].update(
        {
            "author_model": binding.spec.author_model,
            "author_version": "current",
            "checker_model": binding.spec.checker_model,
            "checker_version": "current",
        }
    )
    if review_held:
        model["status"]["human_review"] = {
            "required": True,
            "reason": None,
            "queue": "crawler-human-review",
            "requested_at": _NOW,
        }
    with CanonicalReducer(default_ledger_paths(records_root), context) as reducer:
        reducer.append_attempt(attempt)
        reducer.append_model(reducer.prepare_model(model))


def _record_environment_axis(records_root: Path, generation: str) -> None:
    """Durably record one shard's latest ``core`` environment generation.

    The generation lands exactly the way a real campaign records one the
    environment axis can replay: a canonical ``campaign-health`` quarantine
    event naming the intent and its environment generation.

    Parameters
    ----------
    records_root:
        Shard-local records root.
    generation:
        Environment generation content digest.
    """

    event = make_operational_event()
    event.pop("ledger_seq", None)
    event.update(
        {
            "event_id": f"event-env-{generation[:16]}",
            "event_kind": "campaign-health",
            "status": "healthy",
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 0},
            "current_environment": None,
            "details": {
                "disposition": "environment-integrity-quarantined",
                "intent": "core",
                "env_generation": generation,
            },
        }
    )
    ledger_path = canonical_operational_ledger_path(default_ledger_paths(records_root).models)
    with JsonlLedger(ledger_path, OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        ledger.append(event)


def _claim_with_rebuild_row(records_root: Path, stable_id: str) -> None:
    """Persist one valid environment-rebuild valve row claiming a model.

    Parameters
    ----------
    records_root:
        Shard-local records root.
    stable_id:
        Withheld model identity.
    """

    row = build_rebuild_deferral_row(
        stable_id=stable_id,
        work_id=f"work-{stable_id}",
        name=stable_id,
        campaign_id="c1-mech",
        run_id="run-fixture",
        machine_id="machine-fixture",
        created_at=_NOW,
        quarantine_event_id="event-fixture-1",
        intent="core",
        target="test",
        artifact_identity="sha256:" + "0" * 64,
        artifact_state={"conda-lock": {"state": "absent"}},
        env_generation=None,
        failure_type="FixtureQuarantine",
    )
    append_rebuild_deferral_row(rebuild_deferral_path(records_root), row)


def _build_shard_fixture(
    tmp_path: Path,
    *,
    valve_only: Mapping[str, Sequence[int]] = (),
    review_held: Mapping[str, Sequence[int]] = (),
    stale: Mapping[str, Sequence[int]] = (),
    unclaimed_stale: Mapping[str, Sequence[int]] = (),
    environment_generation: Mapping[str, str] = (),
) -> _ShardFixture:
    """Build one synthetic c1 campaign split across two production host shards.

    Index-based selectors pick members from each shard's ordered three-model
    slice: ``valve_only`` members get a rebuild valve row instead of a model
    record, ``stale``/``unclaimed_stale`` members are minted under the CAMPAIGN
    ROOT snapshot so shard-authority replay reports them dependency-stale
    (claimed by a valve row only for ``stale``), ``review_held`` members carry a
    live human-review hold, and ``environment_generation`` stamps every shard
    attempt's environment identity.

    Parameters
    ----------
    tmp_path:
        Test-owned scratch root.
    valve_only, review_held, stale, unclaimed_stale:
        Shard-ID-keyed member indexes.
    environment_generation:
        Shard-ID-keyed environment identity.

    Returns
    -------
    _ShardFixture
        Physical two-shard fixture.
    """

    primary_records = tmp_path / "primary" / "menagerie" / "crawler" / "records"
    roster_path = tmp_path / "primary" / "menagerie" / "data" / "crawl_roster.jsonl"
    _write_jsonl(roster_path, _fixture_roster_rows())
    bindings = emit_campaign_partitions(roster_path, primary_records)
    binding = next(b for b in bindings if b.spec.campaign_id == "c1-mech")
    root_snapshot = load_intake_snapshot(primary_records / binding.intake_path)
    ids = sorted(item.stable_id for item in root_snapshot.items)
    assert len(ids) == 6
    shard_slices = {"host-a": tuple(ids[:3]), "host-b": tuple(ids[3:])}
    expected_intake = _subset_snapshot(
        primary_records / binding.intake_path, set(ids), tmp_path / "intake"
    )

    sources: list[HostShardSource] = []
    for shard_id, members in shard_slices.items():
        clone = tmp_path / shard_id
        records_root = clone / "menagerie" / "crawler" / "records"
        shard_intake = _subset_snapshot(
            primary_records / binding.intake_path, set(members), records_root / "intake"
        )
        shard_snapshot = load_intake_snapshot(shard_intake)
        valve_members = {members[i] for i in dict(valve_only).get(shard_id, ())}
        stale_members = {members[i] for i in dict(stale).get(shard_id, ())}
        unclaimed_members = {members[i] for i in dict(unclaimed_stale).get(shard_id, ())}
        review_members = {members[i] for i in dict(review_held).get(shard_id, ())}
        generation = dict(environment_generation).get(shard_id)
        if generation is not None:
            _record_environment_axis(records_root, generation)
        for stable_id in members:
            if stable_id in valve_members:
                _claim_with_rebuild_row(records_root, stable_id)
                continue
            minted_under = (
                root_snapshot if stable_id in stale_members | unclaimed_members else shard_snapshot
            )
            _mint_terminal(
                records_root,
                binding,
                minted_under,
                stable_id,
                review_held=stable_id in review_members,
            )
            if stable_id in stale_members:
                _claim_with_rebuild_row(records_root, stable_id)
        sources.append(
            HostShardSource(shard_id, records_root, clone / ".crawl-local", shard_intake)
        )
    return _ShardFixture(
        manifest=primary_records / "campaigns.json",
        campaign_id="c1-mech",
        binding=binding,
        expected_intake=expected_intake,
        sources=tuple(sources),
        shard_ids=shard_slices,
        output_root=tmp_path / "merged",
    )


@pytest.mark.smoke
def test_valve_claimed_deferral_is_owned_not_missing(tmp_path: Path) -> None:
    """A machine-deferral valve row accounts an assigned model with no record."""

    fixture = _build_shard_fixture(tmp_path, valve_only={"host-a": (0,)})
    deferred_id = fixture.shard_ids["host-a"][0]

    result = merge_host_shards(
        fixture.manifest,
        fixture.campaign_id,
        fixture.sources,
        fixture.output_root,
        expected_intake_root=fixture.expected_intake,
    )

    merged_ids = {record["stable_id"] for record in result.current_records}
    assert deferred_id not in merged_ids
    assert len(merged_ids) == 5
    source_report = result.report["sources"]["host-a"]
    claimed = {
        stable_id
        for stable_ids in source_report["machine_deferred"].values()
        for stable_id in stable_ids
    }
    assert deferred_id in claimed
    assert result.report["proof"]["assigned_ids_terminal_or_deferral_claimed_per_shard"] is True
    assert result.report["merge_status"] == "complete-with-deferrals"


@pytest.mark.smoke
def test_deferral_claim_never_relaxes_a_real_hole(tmp_path: Path) -> None:
    """A model with neither a record nor a valid valve row still refuses."""

    fixture = _build_shard_fixture(tmp_path, valve_only={"host-a": (0,)})
    deferred_id = fixture.shard_ids["host-a"][0]
    records_root = fixture.sources[0].records_root
    valve_path = rebuild_deferral_path(records_root)
    valve_path.unlink()

    with pytest.raises(CampaignMergeError, match=deferred_id):
        merge_host_shards(
            fixture.manifest,
            fixture.campaign_id,
            fixture.sources,
            fixture.output_root,
            expected_intake_root=fixture.expected_intake,
        )


@pytest.mark.smoke
def test_review_held_terminals_merge_with_disclosure(tmp_path: Path) -> None:
    """A live human-review hold merges as a disclosed review-held terminal."""

    fixture = _build_shard_fixture(tmp_path, review_held={"host-b": (1,)})
    held_id = fixture.shard_ids["host-b"][1]

    result = merge_host_shards(
        fixture.manifest,
        fixture.campaign_id,
        fixture.sources,
        fixture.output_root,
        expected_intake_root=fixture.expected_intake,
    )

    merged_ids = {record["stable_id"] for record in result.current_records}
    assert held_id in merged_ids
    assert tuple(result.report["sources"]["host-b"]["review_held_ids"]) == (held_id,)
    assert result.report["merge_status"] == "complete-with-review-holds"


@pytest.mark.smoke
def test_stale_reasons_are_carried_when_valve_claimed(tmp_path: Path) -> None:
    """A dependency-stale, valve-claimed model is carried with its reason."""

    fixture = _build_shard_fixture(tmp_path, stale={"host-a": (2,)})
    stale_id = fixture.shard_ids["host-a"][2]

    result = merge_host_shards(
        fixture.manifest,
        fixture.campaign_id,
        fixture.sources,
        fixture.output_root,
        expected_intake_root=fixture.expected_intake,
    )

    merged_ids = {record["stable_id"] for record in result.current_records}
    assert stale_id not in merged_ids
    stale_reasons = result.report["sources"]["host-a"]["stale_reasons"]
    assert set(stale_reasons) == {stale_id}
    assert stale_reasons[stale_id]
    claimed = {
        stable_id
        for stable_ids in result.report["sources"]["host-a"]["machine_deferred"].values()
        for stable_id in stable_ids
    }
    assert stale_id in claimed
    assert result.report["merge_status"] == "complete-with-deferrals-and-stale-records"


@pytest.mark.smoke
def test_unclaimed_stale_record_carries_but_never_reads_current(tmp_path: Path) -> None:
    """A stale record with no valve row still covers its model, disclosed.

    The record IS the coverage: an honest terminal awaiting its re-drive under
    current dependencies (the driver-close tolerated mid-flight state). It
    merges carried -- excluded from the current view, reason disclosed, status
    never plain ``complete`` -- rather than bricking the whole shard merge.
    """

    fixture = _build_shard_fixture(tmp_path, unclaimed_stale={"host-a": (2,)})
    stale_id = fixture.shard_ids["host-a"][2]

    result = merge_host_shards(
        fixture.manifest,
        fixture.campaign_id,
        fixture.sources,
        fixture.output_root,
        expected_intake_root=fixture.expected_intake,
    )

    merged_ids = {record["stable_id"] for record in result.current_records}
    assert stale_id not in merged_ids
    assert stale_id in result.report["partition"]["stale_carried"]
    assert result.report["merge_status"] == "complete-with-stale-records"


@pytest.mark.smoke
def test_divergent_environment_axes_refuse_to_merge_as_current(tmp_path: Path) -> None:
    """Shards whose shared environment family diverges must not merge silently."""

    fixture = _build_shard_fixture(
        tmp_path,
        environment_generation={
            "host-a": "sha256:" + "a" * 64,
            "host-b": "sha256:" + "b" * 64,
        },
    )

    with pytest.raises(CampaignMergeError, match=ENV_AXIS_DIVERGENT_ERROR):
        merge_host_shards(
            fixture.manifest,
            fixture.campaign_id,
            fixture.sources,
            fixture.output_root,
            expected_intake_root=fixture.expected_intake,
        )


@pytest.mark.smoke
def test_matching_environment_axes_merge_and_report_the_axis(tmp_path: Path) -> None:
    """Shards on one environment generation merge with the axis disclosed."""

    generation = "sha256:" + "c" * 64
    fixture = _build_shard_fixture(
        tmp_path,
        environment_generation={"host-a": generation, "host-b": generation},
    )

    result = merge_host_shards(
        fixture.manifest,
        fixture.campaign_id,
        fixture.sources,
        fixture.output_root,
        expected_intake_root=fixture.expected_intake,
    )

    assert result.report["proof"]["environment_axis_consistent_across_shards"] is True
    assert result.report["sources"]["host-a"]["environment_generations"] == {"core": generation}
    assert result.report["merge_status"] == "complete"


@pytest.mark.smoke
def test_views_commit_through_a_single_manifest(tmp_path: Path) -> None:
    """All derived views commit through one manifest written last."""

    fixture = _build_shard_fixture(tmp_path)

    result = merge_host_shards(
        fixture.manifest,
        fixture.campaign_id,
        fixture.sources,
        fixture.output_root,
        expected_intake_root=fixture.expected_intake,
    )

    manifest_path = fixture.output_root / HOST_SHARD_MERGE_MANIFEST_NAME
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["committed"] is True
    assert set(manifest["views"]) == set(result.view_digests)
    for relative, digest in manifest["views"].items():
        assert hash_bytes((fixture.output_root / relative).read_bytes()) == digest


@pytest.mark.smoke
def test_interrupted_view_write_leaves_no_commit_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid view-write leaves the output visibly uncommitted."""

    fixture = _build_shard_fixture(tmp_path)
    from menagerie.crawler.tools import merge_host_shards as module

    real_replace = module.atomic_replace_bytes
    written = {"count": 0}

    def failing_replace(path: Path, payload: bytes) -> None:
        written["count"] += 1
        if written["count"] >= 2:
            raise OSError("injected crash between view writes")
        real_replace(path, payload)

    monkeypatch.setattr(module, "atomic_replace_bytes", failing_replace)
    with pytest.raises(OSError, match="injected crash"):
        merge_host_shards(
            fixture.manifest,
            fixture.campaign_id,
            fixture.sources,
            fixture.output_root,
            expected_intake_root=fixture.expected_intake,
        )

    assert not (fixture.output_root / HOST_SHARD_MERGE_MANIFEST_NAME).exists()


@pytest.mark.smoke
def test_torn_shard_ledger_refuses_typed_read_only_and_cli_graceful(
    tmp_path: Path,
) -> None:
    """A SIGKILLed shard (torn models ledger) refuses typed and stays unmutated.

    The merge is read-only: it must never evidence-and-truncate a source shard
    (recovery belongs to a writer open on the shard's own host), and the CLI
    must exit typed instead of surfacing a raw ``TornTailError`` traceback.
    """

    from menagerie.crawler.recordio import TornTailError
    from menagerie.crawler.tools import merge_host_shards as module

    fixture = _build_shard_fixture(tmp_path)
    models = default_ledger_paths(fixture.sources[0].records_root).models
    models.write_bytes(models.read_bytes() + b'{"torn": ')
    torn_bytes = models.read_bytes()

    with pytest.raises(TornTailError):
        merge_host_shards(
            fixture.manifest,
            fixture.campaign_id,
            fixture.sources,
            fixture.output_root,
            expected_intake_root=fixture.expected_intake,
        )
    assert models.read_bytes() == torn_bytes

    argv = ["--manifest", str(fixture.manifest), "--campaign-id", fixture.campaign_id]
    for source in fixture.sources:
        argv += [
            "--source",
            f"{source.shard_id}={source.records_root.parents[2]}={source.intake_root}",
        ]
    argv += [
        "--output-root",
        str(fixture.output_root),
        "--expected-intake-root",
        str(fixture.expected_intake),
    ]
    assert module.main(argv) == 1
    assert models.read_bytes() == torn_bytes


def test_no_source_carve_is_a_closed_enumeration() -> None:
    """P5 fable-F12: the carve keys on exactly "failed", never an open complement.

    RED at bd5c7837b: ``_self_disclosed_no_source`` accepted ANY non-"ok"
    ``mandatory_link_status`` -- an unknown future status value would have
    passed the carve as a self-disclosure. Production writes exactly
    "ok"/"failed"; anything else refuses the merge until the set is widened
    in its own reviewed change.
    """

    from menagerie.crawler.tools.merge_host_shards import _self_disclosed_no_source

    def record(status: object) -> dict:
        return {
            "status": {"kind": "failed"},
            "source_resolution": {"mandatory_link_status": status},
        }

    assert _self_disclosed_no_source(record("failed")) is True
    assert _self_disclosed_no_source(record("ok")) is False
    # The open complement is closed: unknown values are not self-disclosures.
    assert _self_disclosed_no_source(record("pending")) is False
    assert _self_disclosed_no_source(record(None)) is False
    assert _self_disclosed_no_source({"status": {"kind": "runs"}}) is False
