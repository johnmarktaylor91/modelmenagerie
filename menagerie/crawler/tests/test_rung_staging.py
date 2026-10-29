"""RED suite for C-RUNG-STAGING: durable rung snapshots and binding refusal.

Pins the DD-6 cluster (MASTER_DOCKET C-RUNG-STAGING):

* T4-T2 -- an intake snapshot with no campaign binding must REFUSE with a typed
  error instead of silently falling back to the default ``DriverConfig`` (wrong
  author model, wrong review checkpoint).
* T4-T1 (staging half) -- next-rung intakes are built in-repo from the catalog
  plus prior-rung state, with explicit ``--campaign`` membership validation and
  ``derived_from_snapshot_id`` lineage; staging refuses while the K2
  prior-snapshot currency carve-out is absent from ``authority``.
* T2-03 (code half) -- staged snapshots and their registrations are durable
  under ``records/intake/``, never only in disposable scratch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pytest

from menagerie.crawler import authority as authority_module
from menagerie.crawler import cli as cli_module
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.identity import canonical_json_bytes
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot, load_intake_snapshot
from menagerie.crawler.partitioner import emit_campaign_partitions, load_campaign_bindings
from menagerie.crawler.tests.close_object_support import record_fixture_close

pytestmark = pytest.mark.smoke

_ROSTER_ROWS: tuple[dict[str, Any], ...] = (
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


@pytest.fixture()
def campaign_repo(tmp_path: Path) -> Path:
    """Materialize a tiny four-campaign repository with a real manifest.

    The models ledger is seeded EMPTY rather than left absent: an empty ledger
    is the explicit statement that no awards exist, while an absent one cannot
    prove anything and the AW-2 staging door fails CLOSED on it.

    Returns
    -------
    pathlib.Path
        Repository root whose ``menagerie/crawler/records`` holds the emitted
        partitions, intake snapshots, and ``campaigns.json``.
    """

    repo_root = tmp_path / "repo"
    records_root = repo_root / "menagerie" / "crawler" / "records"
    roster_path = tmp_path / "roster.jsonl"
    roster_path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in _ROSTER_ROWS))
    emit_campaign_partitions(roster_path, records_root)
    _write_models_shard(records_root, [])
    return repo_root


def _records_root(repo_root: Path) -> Path:
    """Return the campaign records root for one repository."""

    return repo_root / "menagerie" / "crawler" / "records"


def _root_binding(repo_root: Path, campaign_id: str):
    """Return the manifest root binding for one campaign."""

    bindings = load_campaign_bindings(_records_root(repo_root) / "campaigns.json")
    return next(b for b in bindings if b.spec.campaign_id == campaign_id)


def _root_snapshot(repo_root: Path, campaign_id: str) -> IntakeSnapshot:
    """Load one campaign's bound full-partition intake snapshot."""

    binding = _root_binding(repo_root, campaign_id)
    return load_intake_snapshot(_records_root(repo_root) / binding.intake_path)


def _scratch_subset_snapshot(
    repo_root: Path,
    scratch: Path,
    campaign_id: str,
    take: int,
) -> IntakeSnapshot:
    """Build an UNREGISTERED subset snapshot outside the durable record store.

    This reproduces today's out-of-repo ladder staging: a content-addressed
    subset whose new snapshot id no manifest row owns (T4-T2's trigger).
    """

    binding = _root_binding(repo_root, campaign_id)
    partition_rows = [
        json.loads(line)
        for line in (_records_root(repo_root) / binding.partition_path)
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    scratch.mkdir(parents=True, exist_ok=True)
    subset_partition = scratch / "subset.jsonl"
    subset_partition.write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in partition_rows[:take])
    )
    empty = scratch / "empty.jsonl"
    empty.write_bytes(b"\n")
    return create_intake_snapshot(subset_partition, empty, scratch / "intake")


def _write_models_shard(records_root: Path, rows: list[dict[str, Any]]) -> None:
    """Write the sharded models ledger below one records root."""

    path = records_root / "models" / "current-shard.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _head_record(
    stable_id: str,
    status_kind: str,
    status_code: str,
    *,
    intake_snapshot_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build one minimal current model revision with the given status kind.

    ``intake_snapshot_id`` records which snapshot the revision was recorded
    under (``dependency_vector.intake_snapshot_id``) -- the evidence the
    provable-history register path reads.
    """

    from menagerie.crawler.identity import stable_hash

    body: dict[str, Any] = {
        "stable_id": stable_id,
        "record_seq": 1,
        "parent_revision": None,
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
    body["record_revision"] = stable_hash(body)
    return body


def _seed_member_heads(
    records_root: Path,
    member_ids: Sequence[str],
    *,
    status_kind: str = "failed",
    status_code: str = "failed:import",
) -> None:
    """Append a current head for every member that does not already hold one.

    The models shard is append-only history, so members already recorded keep
    the head they have; this only fills the gap a fixture leg would have filled
    by running.
    """

    path = records_root / "models" / "current-shard.jsonl"
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
        _head_record(stable_id, status_kind, status_code)
        for stable_id in member_ids
        if stable_id not in existing
    ]
    if not rows:
        return
    with path.open("ab") as handle:
        handle.write(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _close_staged_rung(repo_root: Path, staged: Any, campaign_id: str) -> None:
    """Attest a close for one staged rung so a NEXT rung may derive from it.

    Since BD-1, ``stage_rung`` requires an immutable close object for a lineage
    that is a registered rung (``rung_staging._require_closed_lineage``): a
    boundary is crossed on a durable attestation, never on an interruption
    record or a hand tally (T1-sol-02). These lineage/registry tests are not
    tests about closing a leg, so the precondition is synthesized through the
    real builder; the close object's own honesty is pinned in
    ``test_close_object.py``.
    """

    members = {
        item.stable_id
        for item in load_intake_snapshot(
            _records_root(repo_root) / "intake" / staged.registration.snapshot_id
        ).items
    }
    # A close over a leg that recorded NOTHING licenses nothing (2026-08-16
    # red-team round 2, break 2; v2 contract): the fixture leg opens a real
    # generation and records every member IN-WINDOW, recorded-under the
    # snapshot, so the derived proof adjudicates them. They settle as
    # non-award terminals, which is what a fixture leg honestly is -- seeding
    # awards would make every successor rung an AW-2 refusal.
    record_fixture_close(
        _records_root(repo_root),
        campaign_id=campaign_id,
        intake_snapshot_id=staged.registration.snapshot_id,
        intake_snapshot_sha256=staged.registration.snapshot_sha256,
        member_ids=sorted(members),
        seed_member_heads=True,
    )


def _enable_carve_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate the K2 authority carve-out having landed."""

    from menagerie.crawler.rung_staging import CURRENCY_CARVE_OUT_ATTRIBUTE

    monkeypatch.setattr(authority_module, CURRENCY_CARVE_OUT_ATTRIBUTE, True, raising=False)


def _remove_carve_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a world where the K2 authority carve-out has not landed.

    The carve-out shipped with C-M1-KERNEL (K2), so the refusal arm these
    tests pin is now reachable only by deleting the capability attribute.
    """

    from menagerie.crawler.rung_staging import CURRENCY_CARVE_OUT_ATTRIBUTE

    monkeypatch.delattr(authority_module, CURRENCY_CARVE_OUT_ATTRIBUTE, raising=False)


def _stage(
    repo_root: Path,
    campaign_id: str,
    *,
    stable_ids: Optional[tuple[str, ...]] = None,
    count: Optional[int] = None,
    derived_from: Optional[str] = None,
    label: str = "test-rung",
):
    """Stage one rung through the library API with fixed provenance fields."""

    from menagerie.crawler.rung_staging import RUNG_CLASS_SUCCESSOR, stage_rung
    from menagerie.crawler.tests.rt3_support import fixture_composition_attestation

    return stage_rung(
        _records_root(repo_root),
        campaign_id=campaign_id,
        stable_ids=stable_ids,
        count=count,
        derived_from_snapshot_id=derived_from,
        label=label,
        registered_by="pytest",
        created_at="2026-08-11T00:00:00Z",
        # Fixtures model the composed next-rung shape (W16/S-2 declaration),
        # so they present the SOL-02 PD-7-bearing attestation the class demands.
        rung_class=RUNG_CLASS_SUCCESSOR,
        composition_attestation=fixture_composition_attestation(
            _records_root(repo_root),
            campaign_id,
            stable_ids=stable_ids,
            count=count,
            label=label,
        ),
    )


# ---------------------------------------------------------------------------
# T4-T2: unbound snapshots refuse; they never default the driver config.
# ---------------------------------------------------------------------------


def test_unbound_snapshot_is_refused_not_defaulted(campaign_repo: Path, tmp_path: Path) -> None:
    """An unregistered ladder subset must refuse instead of defaulting models.

    Today ``_snapshot_driver_config`` returns a bare ``DriverConfig()`` for any
    snapshot the manifest does not own: author claude-fable-5-1, checker
    gpt-5.6-terra, review checkpoint 1000 -- silently wrong for c3-classics.
    """

    subset = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c3-classics", take=2)
    with pytest.raises(ValueError, match=subset.snapshot_id):
        cli_module._snapshot_driver_config(campaign_repo, subset)


def test_unbound_snapshot_refusal_is_typed(campaign_repo: Path, tmp_path: Path) -> None:
    """The refusal is the typed rung-staging error and names the remediation."""

    from menagerie.crawler.rung_staging import UnboundIntakeSnapshotError

    subset = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c1-mech", take=1)
    with pytest.raises(UnboundIntakeSnapshotError, match="stage_rung"):
        cli_module._snapshot_driver_config(campaign_repo, subset)


def test_driver_factory_refuses_unbound_snapshot(
    campaign_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run/resume cannot launch an unbound snapshot with defaulted models."""

    from menagerie.crawler.rung_staging import UnboundIntakeSnapshotError

    subset = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c3-classics", take=2)
    public_mirror = tmp_path / "public-mirror"
    private_mirror = tmp_path / "private-mirror"
    public_mirror.mkdir()
    private_mirror.mkdir()
    monkeypatch.setenv("MENAGERIE_PUBLIC_MIRROR", str(public_mirror))
    monkeypatch.setenv("MENAGERIE_PRIVATE_MIRROR", str(private_mirror))
    args = cli_module.build_parser().parse_args(
        [
            "--repo-root",
            str(campaign_repo),
            "run",
            "--intake",
            str(subset.root),
            "--target",
            "osx-arm64",
            "--author-command",
            "/usr/bin/true",
            "--checker-command",
            "/usr/bin/true",
            "--provider-reconcile-command",
            "/usr/bin/true",
            "--environment-command",
            "/usr/bin/true",
        ]
    )
    with pytest.raises(UnboundIntakeSnapshotError, match=subset.snapshot_id):
        cli_module._default_driver_factory(args)


def test_missing_manifest_refuses_snapshot(tmp_path: Path) -> None:
    """A repository with no campaign manifest cannot bind, so it refuses."""

    from menagerie.crawler.rung_staging import UnboundIntakeSnapshotError

    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(canonical_json_bytes({"name": "solo", "zoo": "timm"}) + b"\n")
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"\n")
    snapshot = create_intake_snapshot(rows, empty, tmp_path / "intake")
    with pytest.raises(UnboundIntakeSnapshotError, match="campaign manifest"):
        cli_module._snapshot_driver_config(tmp_path / "bare-repo", snapshot)


# ---------------------------------------------------------------------------
# Staging: carve-out gate, campaign validation, lineage, durability.
# ---------------------------------------------------------------------------


def test_stage_refuses_when_currency_carve_out_absent(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Staging refuses with a typed error while the K2 carve-out is missing.

    The refusal must name the exact ``authority`` capability attribute so the
    operator knows precisely which fix (C-M1-KERNEL / K2) unblocks staging, and
    it must leave the durable store untouched. K2 landed the carve-out, so the
    missing-capability world is simulated by deleting the attribute.
    """

    from menagerie.crawler.rung_staging import (
        CURRENCY_CARVE_OUT_ATTRIBUTE,
        CurrencyCarveOutMissingError,
        rung_registry_path,
    )

    _remove_carve_out(monkeypatch)
    assert not hasattr(authority_module, CURRENCY_CARVE_OUT_ATTRIBUTE)
    root = _root_snapshot(campaign_repo, "c3-classics")
    ids = tuple(item.stable_id for item in root.items[:2])
    intake_before = sorted(
        path.name for path in (_records_root(campaign_repo) / "intake").iterdir()
    )
    with pytest.raises(CurrencyCarveOutMissingError, match=CURRENCY_CARVE_OUT_ATTRIBUTE):
        _stage(campaign_repo, "c3-classics", stable_ids=ids)
    assert not rung_registry_path(_records_root(campaign_repo)).exists()
    intake_after = sorted(path.name for path in (_records_root(campaign_repo) / "intake").iterdir())
    assert intake_after == intake_before


def test_registered_mini_rung_binds_correct_models(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merge gate: a staged mini rung loads with its campaign's exact binding.

    c3-classics is the campaign whose binding differs from every default, so a
    silent fallback cannot masquerade as success here.
    """

    _enable_carve_out(monkeypatch)
    root = _root_snapshot(campaign_repo, "c3-classics")
    ids = tuple(item.stable_id for item in root.items[:3])
    staged = _stage(campaign_repo, "c3-classics", stable_ids=ids, label="mini-3")

    loaded = load_intake_snapshot(staged.snapshot_root)
    assert loaded.snapshot_id == staged.registration.snapshot_id
    assert {item.stable_id for item in loaded.items} == set(ids)

    config = cli_module._snapshot_driver_config(campaign_repo, loaded)
    assert config.author_model == "claude-opus-5"
    assert config.checker_model == "claude-fable-5-1"
    assert config.review_checkpoint_at == 0
    assert (config.author_model, config.checker_model) != (
        DriverConfig().author_model,
        DriverConfig().checker_model,
    )


def test_staged_snapshot_is_durable_under_records_intake(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2-03: the staged snapshot and its lineage row live in records/intake."""

    from menagerie.crawler.rung_staging import (
        RUNG_REGISTRATION_SCHEMA_VERSION,
        load_rung_registrations,
        rung_registry_path,
    )

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    root_binding = _root_binding(campaign_repo, "c3-classics")
    root = _root_snapshot(campaign_repo, "c3-classics")
    ids = tuple(item.stable_id for item in root.items[:2])
    staged = _stage(campaign_repo, "c3-classics", stable_ids=ids, label="mini-2")

    assert staged.snapshot_root == records_root / "intake" / staged.registration.snapshot_id
    assert (staged.snapshot_root / "manifest.json").is_file()
    assert (staged.snapshot_root / "items.jsonl").is_file()
    reloaded = load_intake_snapshot(staged.snapshot_root)
    assert reloaded.snapshot_sha256 == staged.registration.snapshot_sha256

    registry = rung_registry_path(records_root)
    assert registry == records_root / "intake" / "registered-rungs.jsonl"
    rows = load_rung_registrations(records_root)
    assert len(rows) == 1
    registration = rows[0]
    assert registration.campaign_id == "c3-classics"
    assert registration.derived_from_snapshot_id == root_binding.intake_snapshot_id
    assert registration.member_count == 2
    raw = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line]
    assert raw[0]["schema_version"] == RUNG_REGISTRATION_SCHEMA_VERSION
    assert raw[0]["row_sha256"].startswith("sha256:")


def test_registry_is_append_only_and_contradiction_refused(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restaging is idempotent; a conflicting claim over one snapshot refuses."""

    from menagerie.crawler.rung_staging import (
        RungStagingError,
        build_rung_registration,
        append_rung_registration,
        load_rung_registrations,
        rung_registry_path,
    )

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    root = _root_snapshot(campaign_repo, "c3-classics")
    ids = tuple(item.stable_id for item in root.items[:2])
    first = _stage(campaign_repo, "c3-classics", stable_ids=ids, label="mini-2")
    assert first.registration_created

    again = _stage(campaign_repo, "c3-classics", stable_ids=ids, label="mini-2")
    assert not again.registration_created
    assert again.registration.snapshot_id == first.registration.snapshot_id
    assert len(load_rung_registrations(records_root)) == 1

    _close_staged_rung(campaign_repo, first, "c3-classics")
    wider = _stage(
        campaign_repo,
        "c3-classics",
        stable_ids=tuple(item.stable_id for item in root.items[:3]),
        label="mini-3",
    )
    rows = load_rung_registrations(records_root)
    assert len(rows) == 2
    assert rows[0].snapshot_id == first.registration.snapshot_id
    assert rows[1].snapshot_id == wider.registration.snapshot_id

    conflicting = build_rung_registration(
        campaign_id="c1-mech",
        snapshot_id=first.registration.snapshot_id,
        snapshot_sha256=first.registration.snapshot_sha256,
        derived_from_snapshot_id=first.registration.derived_from_snapshot_id,
        member_count=first.registration.member_count,
        label="conflict",
        registered_by="pytest",
        created_at="2026-08-11T00:00:00Z",
    )
    with pytest.raises(RungStagingError, match="already registered"):
        append_rung_registration(rung_registry_path(records_root), conflicting)


def test_stage_validates_campaign_and_membership(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown campaigns and out-of-campaign members refuse with named ids."""

    from menagerie.crawler.rung_staging import RungStagingError

    _enable_carve_out(monkeypatch)
    with pytest.raises(RungStagingError, match="c9-unknown"):
        _stage(campaign_repo, "c9-unknown", stable_ids=("m_0",))

    foreign = _root_snapshot(campaign_repo, "c1-mech").items[0].stable_id
    with pytest.raises(RungStagingError, match=foreign):
        _stage(campaign_repo, "c3-classics", stable_ids=(foreign,))


def test_stage_count_prefix_expresses_step_rungs(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``count`` stages the partition-order prefix (the step-200 shape)."""

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    binding = _root_binding(campaign_repo, "c1-mech")
    partition_rows = [
        json.loads(line)
        for line in (records_root / binding.partition_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    root = _root_snapshot(campaign_repo, "c1-mech")
    by_key = {item.natural_key: item.stable_id for item in root.items}
    expected = {
        by_key[(str(row["name"]), str(row["zoo"]), str(row.get("variant", "")))]
        for row in partition_rows[:2]
    }

    staged = _stage(campaign_repo, "c1-mech", count=2, label="step-2")
    loaded = load_intake_snapshot(staged.snapshot_root)
    assert {item.stable_id for item in loaded.items} == expected


def test_stage_lineage_chains_from_prior_rung(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The next rung derives from the prior registered rung, once it closed."""

    from menagerie.crawler.close_object import CloseObjectError
    from menagerie.crawler.rung_staging import RungStagingError

    _enable_carve_out(monkeypatch)
    root = _root_snapshot(campaign_repo, "c3-classics")
    first = _stage(
        campaign_repo,
        "c3-classics",
        stable_ids=tuple(item.stable_id for item in root.items[:2]),
        label="rung-1",
    )
    # BD-1: the next rung derives from a rung, so that rung must have CLOSED.
    with pytest.raises(CloseObjectError, match="no immutable close object"):
        _stage(
            campaign_repo,
            "c3-classics",
            stable_ids=tuple(item.stable_id for item in root.items[:3]),
            label="rung-2",
        )
    _close_staged_rung(campaign_repo, first, "c3-classics")
    second = _stage(
        campaign_repo,
        "c3-classics",
        stable_ids=tuple(item.stable_id for item in root.items[:3]),
        label="rung-2",
    )
    assert second.registration.derived_from_snapshot_id == first.registration.snapshot_id

    with pytest.raises(RungStagingError, match="intake-does-not-exist"):
        _stage(
            campaign_repo,
            "c3-classics",
            stable_ids=tuple(item.stable_id for item in root.items[:1]),
            derived_from="intake-does-not-exist",
            label="rung-broken",
        )


def test_register_adopts_existing_snapshot_without_carve_out(
    campaign_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OPS-2 contract: an already-run scratch snapshot is copied in and bound.

    Registration of an EXISTING snapshot is deliberately not carve-out gated:
    it is the durability copy that protects already-recorded history, and the
    pilot must be able to register its two live rungs even in a world without
    K2's carve-out (simulated here by deleting the capability attribute).
    "Already-run" is evidence, not a claim (2026-08-16 red-team break 3), so
    the fixture records each member's head under the snapshot being adopted.
    """

    from menagerie.crawler.rung_staging import (
        CURRENCY_CARVE_OUT_ATTRIBUTE,
        register_existing_snapshot,
    )

    _remove_carve_out(monkeypatch)
    assert not hasattr(authority_module, CURRENCY_CARVE_OUT_ATTRIBUTE)
    records_root = _records_root(campaign_repo)
    root_binding = _root_binding(campaign_repo, "c1-mech")
    root = _root_snapshot(campaign_repo, "c1-mech")
    scratch = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c1-mech", take=3)
    assert {item.stable_id for item in scratch.items} <= {item.stable_id for item in root.items}
    _write_models_shard(
        records_root,
        [
            _head_record(item.stable_id, "runs", "runs", intake_snapshot_id=scratch.snapshot_id)
            for item in scratch.items
        ],
    )

    adopted = register_existing_snapshot(
        records_root,
        scratch.root,
        campaign_id="c1-mech",
        derived_from_snapshot_id=None,
        label="pilot-adopted",
        registered_by="pytest",
        created_at="2026-08-11T00:00:00Z",
    )
    assert adopted.snapshot_root == records_root / "intake" / scratch.snapshot_id
    durable = load_intake_snapshot(adopted.snapshot_root)
    assert durable.snapshot_sha256 == scratch.snapshot_sha256
    assert adopted.registration.derived_from_snapshot_id == root_binding.intake_snapshot_id

    config = cli_module._snapshot_driver_config(campaign_repo, durable)
    assert config.author_model == "claude-fable-5-1"
    assert config.review_checkpoint_at == root_binding.spec.review_checkpoint_at


def test_stage_tool_cli_stages_and_lists(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The in-repo tool stages a rung and reports lineage as JSON."""

    from menagerie.crawler.tools.stage_rung import main as stage_rung_main

    _enable_carve_out(monkeypatch)
    root = _root_snapshot(campaign_repo, "c3-classics")
    ids = [item.stable_id for item in root.items[:3]]
    argv = [
        "stage",
        "--repo-root",
        str(campaign_repo),
        "--campaign",
        "c3-classics",
        "--label",
        "mini-3",
    ]
    for stable_id in ids:
        argv.extend(["--stable-id", stable_id])
    assert stage_rung_main(argv) == 0
    staged_report = json.loads(capsys.readouterr().out)
    assert staged_report["campaign_id"] == "c3-classics"
    assert staged_report["member_count"] == 3
    snapshot_id = staged_report["snapshot_id"]

    assert stage_rung_main(["list", "--repo-root", str(campaign_repo)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert [row["snapshot_id"] for row in listing["registered_rungs"]] == [snapshot_id]

    loaded = load_intake_snapshot(_records_root(campaign_repo) / "intake" / snapshot_id)
    config = cli_module._snapshot_driver_config(campaign_repo, loaded)
    assert config.author_model == "claude-opus-5"


def test_stage_tool_cli_refuses_without_carve_out(
    campaign_repo: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool exits non-zero and names the missing carve-out capability."""

    from menagerie.crawler.rung_staging import CURRENCY_CARVE_OUT_ATTRIBUTE
    from menagerie.crawler.tools.stage_rung import main as stage_rung_main

    _remove_carve_out(monkeypatch)

    root = _root_snapshot(campaign_repo, "c3-classics")
    exit_code = stage_rung_main(
        [
            "stage",
            "--repo-root",
            str(campaign_repo),
            "--campaign",
            "c3-classics",
            "--stable-id",
            root.items[0].stable_id,
            "--label",
            "mini-1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code != 0
    assert CURRENCY_CARVE_OUT_ATTRIBUTE in captured.err


def test_stage_tool_cli_refuses_an_unclosed_lineage(
    campaign_repo: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BD-1: the tool prints the close-object refusal, it never tracebacks.

    The refusal an operator would have hit at step-200: the prior rung is
    registered but never produced an immutable close, so its successor cannot be
    staged. The message must reach stderr with a non-zero exit like every other
    typed staging refusal (T1-sol-02).

    Parameters
    ----------
    campaign_repo:
        Tiny four-campaign repository with a real manifest.
    capsys:
        Captured CLI output.
    monkeypatch:
        Enables the K2 carve-out so this arm is the only refusal in play.
    """

    from menagerie.crawler.tools.stage_rung import main as stage_rung_main

    _enable_carve_out(monkeypatch)
    root = _root_snapshot(campaign_repo, "c3-classics")
    first = _stage(
        campaign_repo,
        "c3-classics",
        stable_ids=(root.items[0].stable_id,),
        label="rung-1",
    )
    exit_code = stage_rung_main(
        [
            "stage",
            "--repo-root",
            str(campaign_repo),
            "--campaign",
            "c3-classics",
            "--stable-id",
            root.items[0].stable_id,
            "--stable-id",
            root.items[1].stable_id,
            "--label",
            "rung-2",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "stage-rung refused" in captured.err
    assert "no immutable close object" in captured.err
    assert first.registration.snapshot_id in captured.err


def test_cli_projection_carries_the_registered_prior_lineage(campaign_repo: Path) -> None:
    """The CLI must hand the projection the SAME lineage facts the driver does.

    ``CrawlerDriver`` passes ``prior_intake_snapshots`` and
    ``prior_intake_member_ids`` into ``build_authority_context`` (``driver.py``);
    ``cli._snapshot_authority_context`` passed neither, so every CLI projection
    silently disagreed with the driver that produced the ledgers it reads. The K2
    currency carve-out and its T4-T1 membership half exist precisely so a revision
    recorded under a PRIOR registered snapshot of the same campaign stays
    dependency-current -- facts the CLI dropped on the floor.

    Measured on a real mini-rung the pipelined driver closed as
    ``terminal-partition-complete``: ``status --verify-partition`` resolved 1 of 127
    revisions as current instead of 3 and called that COMPLETE partition invalid,
    naming the two members whose terminals were recorded under the prior snapshot.

    Parameters
    ----------
    campaign_repo:
        Tiny four-campaign repository with a real manifest.
    """

    from menagerie.crawler.cli import _snapshot_authority_context, _snapshot_driver_config

    records_root = _records_root(campaign_repo)
    first = _stage(campaign_repo, "c1-mech", count=2, label="step-2")
    _close_staged_rung(campaign_repo, first, "c1-mech")
    second = _stage(
        campaign_repo,
        "c1-mech",
        count=3,
        derived_from=first.registration.snapshot_id,
        label="step-3",
    )
    assert second.registration.snapshot_id != first.registration.snapshot_id
    snapshot = load_intake_snapshot(records_root / "intake" / second.registration.snapshot_id)

    context = _snapshot_authority_context(
        snapshot, _snapshot_driver_config(campaign_repo, snapshot)
    )

    assert first.registration.snapshot_id in context.prior_intake_snapshots
    prior_members = {
        item.stable_id
        for item in load_intake_snapshot(
            records_root / "intake" / first.registration.snapshot_id
        ).items
    }
    assert prior_members
    assert prior_members <= set(context.prior_intake_member_ids)


# ---------------------------------------------------------------------------
# Stage-gate hardening (2026-08-16 red-team): BOTH staging doors are gated on
# the LINEAGE and the EVIDENCE, never on which CLI subcommand ran.
# ---------------------------------------------------------------------------


def test_register_new_work_requires_the_lineage_close(
    campaign_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rung boundary is crossed on a close object through EITHER door.

    Red-team break 1 (opus #1 / fable vector 4): ``register_existing_snapshot``
    never called ``_require_closed_lineage``, so the exact lineage ``stage_rung``
    refused was adoptable via the sibling subcommand.
    """

    from menagerie.crawler.close_object import CloseObjectError
    from menagerie.crawler.rung_staging import register_existing_snapshot

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    prior = _stage(campaign_repo, "c1-mech", count=2, label="prior-rung")
    scratch = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c1-mech", take=3)

    with pytest.raises(CloseObjectError, match="no immutable close object"):
        register_existing_snapshot(
            records_root,
            scratch.root,
            campaign_id="c1-mech",
            derived_from_snapshot_id=prior.registration.snapshot_id,
            label="escape-attempt",
            registered_by="pytest",
            created_at="2026-08-16T00:00:00Z",
            historical_only=False,
        )

    _close_staged_rung(campaign_repo, prior, "c1-mech")
    from menagerie.crawler.intake import load_intake_snapshot
    from menagerie.crawler.tests.rt3_support import fixture_composition_attestation

    scratch_snapshot = load_intake_snapshot(scratch.root)
    scratch_members = tuple(item.stable_id for item in scratch_snapshot.items)
    adopted = register_existing_snapshot(
        records_root,
        scratch.root,
        campaign_id="c1-mech",
        derived_from_snapshot_id=prior.registration.snapshot_id,
        label="gated-adoption",
        registered_by="pytest",
        created_at="2026-08-16T00:00:00Z",
        historical_only=False,
        # A new-work adoption under a live close declares its class (W16/S-2)
        # and, as a successor, presents the SOL-02 PD-7-bearing attestation.
        rung_class="successor_rung",
        composition_attestation=fixture_composition_attestation(
            records_root,
            "c1-mech",
            stable_ids=scratch_members,
            label="gated-adoption",
            snapshot_id=scratch_snapshot.snapshot_id,
            snapshot_sha256=scratch_snapshot.snapshot_sha256,
        ),
    )
    assert adopted.registration.derived_from_snapshot_id == prior.registration.snapshot_id


def test_register_historical_requires_the_lineage_close_too(
    campaign_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The close gate binds the lineage, not the door: history adoption included."""

    from menagerie.crawler.close_object import CloseObjectError
    from menagerie.crawler.rung_staging import register_existing_snapshot

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    prior = _stage(campaign_repo, "c1-mech", count=2, label="prior-rung")
    scratch = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c1-mech", take=3)
    _write_models_shard(
        records_root,
        [
            _head_record(item.stable_id, "runs", "runs", intake_snapshot_id=scratch.snapshot_id)
            for item in scratch.items
        ],
    )
    with pytest.raises(CloseObjectError, match="no immutable close object"):
        register_existing_snapshot(
            records_root,
            scratch.root,
            campaign_id="c1-mech",
            derived_from_snapshot_id=prior.registration.snapshot_id,
            label="history-over-open-boundary",
            registered_by="pytest",
            created_at="2026-08-16T00:00:00Z",
        )


def test_register_refuses_an_unproven_historical_claim(
    campaign_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``historical_only`` is EVIDENCE, not a caller boolean (red-team break 3).

    A snapshot is history only if every member holds a head recorded under
    that exact snapshot. A freshly composed snapshot (no heads at all, or
    heads recorded under a different snapshot) refuses the historical door.
    """

    from menagerie.crawler.rung_staging import RungStagingError, register_existing_snapshot

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    scratch = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c1-mech", take=2)

    # No member holds any head: nothing proves this snapshot ever ran.
    with pytest.raises(RungStagingError, match="historical"):
        register_existing_snapshot(
            records_root,
            scratch.root,
            campaign_id="c1-mech",
            label="fresh-composed-claiming-history",
            registered_by="pytest",
            created_at="2026-08-16T00:00:00Z",
        )

    # Heads exist but were recorded under a DIFFERENT snapshot: still no proof.
    _write_models_shard(
        records_root,
        [
            _head_record(item.stable_id, "runs", "runs", intake_snapshot_id="intake-somewhere-else")
            for item in scratch.items
        ],
    )
    with pytest.raises(RungStagingError, match="historical"):
        register_existing_snapshot(
            records_root,
            scratch.root,
            campaign_id="c1-mech",
            label="foreign-history-claim",
            registered_by="pytest",
            created_at="2026-08-16T00:00:00Z",
        )


def test_register_proven_history_adopts_without_award_authority(
    campaign_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proven history adopts: the awards it holds are that snapshot's own run."""

    from menagerie.crawler.rung_staging import register_existing_snapshot

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    scratch = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c1-mech", take=2)
    _write_models_shard(
        records_root,
        [
            _head_record(item.stable_id, "runs", "runs", intake_snapshot_id=scratch.snapshot_id)
            for item in scratch.items
        ],
    )
    adopted = register_existing_snapshot(
        records_root,
        scratch.root,
        campaign_id="c1-mech",
        label="proven-history",
        registered_by="pytest",
        created_at="2026-08-16T00:00:00Z",
    )
    durable = load_intake_snapshot(adopted.snapshot_root)
    assert durable.snapshot_sha256 == scratch.snapshot_sha256


def test_register_new_work_runs_holder_requires_award_authority(
    campaign_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AW-2 fires on the register door exactly as on stage (red-team break 1/2)."""

    from menagerie.crawler.rung_staging import RungStagingError, register_existing_snapshot

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    scratch = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c1-mech", take=2)
    holder = scratch.items[0].stable_id
    head = _head_record(holder, "runs", "runs", intake_snapshot_id="intake-earlier-rung")
    _write_models_shard(records_root, [head])

    with pytest.raises(RungStagingError, match="runs awards"):
        register_existing_snapshot(
            records_root,
            scratch.root,
            campaign_id="c1-mech",
            label="new-work-with-award-holder",
            registered_by="pytest",
            created_at="2026-08-16T00:00:00Z",
            historical_only=False,
        )

    adopted = register_existing_snapshot(
        records_root,
        scratch.root,
        campaign_id="c1-mech",
        label="new-work-authorized",
        registered_by="pytest",
        created_at="2026-08-16T00:00:00Z",
        historical_only=False,
        award_authorities={holder: str(head["record_revision"])},
    )
    assert holder in {item.stable_id for item in load_intake_snapshot(adopted.snapshot_root).items}


def test_a_refused_stage_writes_no_snapshot(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused stage leaves NOTHING durable for the other door to adopt.

    Red-team break 2 (opus #1 contributing defect): ``_write_subset_snapshot``
    ran before the close gate, so a refused ``stage`` parked a ready-made
    orphan snapshot under ``records/intake/`` for ``register`` to pick up.
    """

    from menagerie.crawler.close_object import CloseObjectError

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    _stage(campaign_repo, "c1-mech", count=2, label="prior-rung")
    intake_before = sorted(path.name for path in (records_root / "intake").iterdir())
    with pytest.raises(CloseObjectError, match="no immutable close object"):
        _stage(campaign_repo, "c1-mech", count=3, label="refused-successor")
    intake_after = sorted(path.name for path in (records_root / "intake").iterdir())
    assert intake_after == intake_before, (
        "a refused stage left durable snapshot bytes behind: "
        f"{sorted(set(intake_after) - set(intake_before))}"
    )


def test_aw2_reads_the_flat_models_ledger_layout(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AW-2 resolves the layout: a flat-ledger award refuses staging too.

    Red-team break 4 (opus #3): only ``models/current-shard.jsonl`` was
    consulted, so a runs head in the flat ``models.jsonl`` layout -- which
    ``resolve_ledger_paths`` supports everywhere else -- staged unchecked.
    """

    from menagerie.crawler.rung_staging import RungStagingError

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    (records_root / "models" / "current-shard.jsonl").unlink()
    root = _root_snapshot(campaign_repo, "c1-mech")
    holder = root.items[0].stable_id
    (records_root / "models.jsonl").write_bytes(
        canonical_json_bytes(_head_record(holder, "runs", "runs")) + b"\n"
    )
    with pytest.raises(RungStagingError, match="runs awards"):
        _stage(campaign_repo, "c1-mech", stable_ids=(holder,), label="flat-layout-award")


def test_aw2_fails_closed_when_no_models_ledger_exists(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent models ledger cannot prove the absence of runs holders.

    Red-team break 4 (opus #3): a missing shard read as "no awards exist" and
    the door failed OPEN -- exactly the freshly composed staging-root shape
    (P-9) the next operation produces.
    """

    from menagerie.crawler.rung_staging import RungStagingError

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    (records_root / "models" / "current-shard.jsonl").unlink()
    root = _root_snapshot(campaign_repo, "c1-mech")
    with pytest.raises(RungStagingError, match="models ledger"):
        _stage(
            campaign_repo,
            "c1-mech",
            stable_ids=(root.items[0].stable_id,),
            label="ledgerless-stage",
        )


def test_register_tool_cli_expresses_award_authority(
    campaign_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both doors of the shipped tool express award authority (break 8).

    The ``register`` subcommand takes ``--new-work`` (the un-provable-history
    escape hatch becomes an explicit operator claim) and ``--award-authority``,
    and the AW-2 refusal is reachable and satisfiable from the CLI.
    """

    from menagerie.crawler.tools.stage_rung import main as stage_rung_main

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    scratch = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c1-mech", take=2)
    holder = scratch.items[0].stable_id
    head = _head_record(holder, "runs", "runs", intake_snapshot_id="intake-earlier-rung")
    _write_models_shard(records_root, [head])

    common = [
        "register",
        "--repo-root",
        str(campaign_repo),
        "--campaign",
        "c1-mech",
        "--snapshot-root",
        str(scratch.root),
        "--label",
        "cli-register",
    ]

    refused = stage_rung_main([*common, "--new-work"])
    captured = capsys.readouterr()
    assert refused == 2
    assert "runs awards" in captured.err

    authorized = stage_rung_main(
        [
            *common,
            "--new-work",
            "--award-authority",
            f"{holder}={head['record_revision']}",
        ]
    )
    captured = capsys.readouterr()
    assert authorized == 0
    report = json.loads(captured.out)
    assert report["member_count"] == 2


def test_register_tool_cli_award_authority_requires_new_work(
    campaign_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An authority the historical door would silently ignore refuses loudly."""

    from menagerie.crawler.tools.stage_rung import main as stage_rung_main

    _enable_carve_out(monkeypatch)
    scratch = _scratch_subset_snapshot(campaign_repo, tmp_path / "scratch", "c1-mech", take=1)
    exit_code = stage_rung_main(
        [
            "register",
            "--repo-root",
            str(campaign_repo),
            "--campaign",
            "c1-mech",
            "--snapshot-root",
            str(scratch.root),
            "--label",
            "cli-misuse",
            "--award-authority",
            "m_x=sha256:0",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--new-work" in captured.err


# ---------------------------------------------------------------------------
# Staging hardening (2026-08-16 red-team ROUND 2, breaks 1 and 4): round 1
# closed the ungated DOOR; these close the ungated ARGUMENT. The gate binds the
# lineage the STATE names, and "no awards" is proved by the ledger that holds
# heads -- never by whichever layout was created first.
# ---------------------------------------------------------------------------


def test_derived_from_cannot_choose_the_lineage_the_gate_inspects(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--derived-from <root>`` was a THIRD staging door (round-2 break 1).

    The close gate exempted the campaign ROOT (a root never ran as a rung, so
    nothing can have closed it) -- and then let the caller NAME the lineage it
    inspected. Re-deriving from the root with a live, unclosed rung on the
    registry skipped the gate entirely: one flag instead of one subcommand.
    The exemption written for rung 1 is only sound while the campaign has ZERO
    registered rungs, so the gate now reads the tip from the STATE.
    """

    from menagerie.crawler.close_object import CloseObjectError
    from menagerie.crawler.rung_staging import RungStagingError

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    root_binding = _root_binding(campaign_repo, "c1-mech")
    prior = _stage(campaign_repo, "c1-mech", count=2, label="live-unclosed-rung")
    intake_before = sorted(path.name for path in (records_root / "intake").iterdir())

    # The named-root escape refuses on the SAME authority as the default
    # lineage, and leaves nothing durable behind.
    with pytest.raises((CloseObjectError, RungStagingError)) as excinfo:
        _stage(
            campaign_repo,
            "c1-mech",
            count=3,
            derived_from=root_binding.intake_snapshot_id,
            label="root-rederivation-escape",
        )
    assert prior.registration.snapshot_id in str(excinfo.value)
    assert sorted(path.name for path in (records_root / "intake").iterdir()) == intake_before

    # The register door is the same door: it may not name the root either.
    from menagerie.crawler.rung_staging import register_existing_snapshot

    scratch_root = campaign_repo.parent / "scratch-root-escape"
    scratch = _scratch_subset_snapshot(campaign_repo, scratch_root, "c1-mech", take=3)
    with pytest.raises((CloseObjectError, RungStagingError)):
        register_existing_snapshot(
            records_root,
            scratch.root,
            campaign_id="c1-mech",
            derived_from_snapshot_id=root_binding.intake_snapshot_id,
            label="root-rederivation-register",
            registered_by="pytest",
            created_at="2026-08-16T00:00:00Z",
            historical_only=False,
        )

    # Closing the live rung is the ONE way through, and then the lineage the
    # registry records is the rung that actually closed.
    _close_staged_rung(campaign_repo, prior, "c1-mech")
    successor = _stage(campaign_repo, "c1-mech", count=3, label="successor")
    assert successor.registration.derived_from_snapshot_id == prior.registration.snapshot_id


def test_an_empty_shard_cannot_shadow_a_populated_flat_ledger(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The absent-shard fix's own remedy became the escape (round-2 break 4).

    ``_resolve_models_ledger`` returned the FIRST layout that exists, so
    following the refusal's instruction -- "materialize the campaign's models
    ledger first; an EMPTY ledger file is the explicit statement that no awards
    exist" -- in a flat-layout root put a zero-byte shard in front of the
    ledger that actually holds the awards, and AW-2 went silent. Every layout
    present is consulted; absence of awards is proved, never assumed.
    """

    from menagerie.crawler.rung_staging import RungStagingError

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    root = _root_snapshot(campaign_repo, "c1-mech")
    holder = root.items[0].stable_id
    head = _head_record(holder, "runs", "runs")
    (records_root / "models" / "current-shard.jsonl").unlink()
    (records_root / "models.jsonl").write_bytes(canonical_json_bytes(head) + b"\n")
    with pytest.raises(RungStagingError, match="runs awards"):
        _stage(campaign_repo, "c1-mech", stable_ids=(holder,), label="flat-only")

    # The documented remedy, applied on top: an EMPTY shard beside the
    # populated flat ledger must not read as innocence.
    _write_models_shard(records_root, [])
    with pytest.raises(RungStagingError, match="runs awards"):
        _stage(campaign_repo, "c1-mech", stable_ids=(holder,), label="shadowed-by-empty-shard")

    # And the authority is satisfiable: the exact revision the FLAT ledger
    # holds is the authority the door accepts.
    from menagerie.crawler.rung_staging import stage_rung

    staged = stage_rung(
        records_root,
        campaign_id="c1-mech",
        stable_ids=(holder,),
        label="authorized-over-both-layouts",
        registered_by="pytest",
        created_at="2026-08-16T00:00:00Z",
        award_authorities={holder: str(head["record_revision"])},
    )
    assert staged.registration.member_count == 1


def test_two_populated_layouts_that_disagree_fail_closed(
    campaign_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Divergent layouts are ambiguous authority, and ambiguity never opens.

    Round-2 break 4's other half: with both layouts populated there is no
    ledger the door can call THE ledger. Refusing is the only fail-closed
    answer -- picking one silently is how an award-holding head disappears.
    """

    from menagerie.crawler.rung_staging import RungStagingError

    _enable_carve_out(monkeypatch)
    records_root = _records_root(campaign_repo)
    root = _root_snapshot(campaign_repo, "c1-mech")
    holder = root.items[0].stable_id
    _write_models_shard(records_root, [_head_record(holder, "failed", "failed:import")])
    (records_root / "models.jsonl").write_bytes(
        canonical_json_bytes(_head_record(holder, "runs", "runs")) + b"\n"
    )
    with pytest.raises(RungStagingError, match="disagree"):
        _stage(campaign_repo, "c1-mech", stable_ids=(holder,), label="divergent-layouts")
