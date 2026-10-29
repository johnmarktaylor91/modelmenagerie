"""Environment-rebuild deferral ledger and quarantine-park freshness tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler import cli
from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.constants import OperationalEventKind
from menagerie.crawler.driver_admission import (
    ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION,
    _deferral_row_fact_still_holds,
    _intent_setup_artifact_identity,
    _intent_setup_artifact_state,
    _quarantine_code_identity_drift,
    _quarantine_is_cleared,
)
from menagerie.crawler.environment_rebuild import (
    REBUILD_DEFERRAL_DISPOSITION,
    REBUILD_DEFERRAL_SCHEMA_VERSION,
    REBUILD_DEFERRAL_SCHEMA_VERSION_V1,
    RebuildDeferralError,
    append_rebuild_deferral_row,
    build_rebuild_deferral_row,
    current_rebuild_rows,
    load_rebuild_deferral_rows,
    rebuild_deferral_path,
    validate_rebuild_deferral_row,
)
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.intake import IntakeItem
from menagerie.crawler.status import MACHINE_DEFERRAL_DISPOSITIONS, machine_deferral_rejection

_STATE = {
    "osx-arm64.lock": {"state": "present", "sha256": "sha256:" + "aa" * 32},
    "osx-arm64.resolved.json": {"state": "present", "sha256": "sha256:" + "bb" * 32},
    "osx-arm64.resolved.sha256": {"state": "present", "sha256": "sha256:" + "cc" * 32},
    "osx-arm64.probes.json": {"state": "present", "sha256": "sha256:" + "dd" * 32},
}


def _row(**overrides: object) -> dict:
    """Build one valid rebuild deferral row with optional field overrides."""

    values = {
        "stable_id": "m_deadbeef",
        "work_id": "work-m_deadbeef",
        "name": "First",
        "campaign_id": "campaign-1",
        "run_id": "run-1",
        "machine_id": "machine-1",
        "created_at": "2026-08-08T00:00:00Z",
        "quarantine_event_id": "environment-cleanup-abc123",
        "intent": "core",
        "target": "osx-arm64",
        "artifact_identity": "sha256:" + "11" * 32,
        "artifact_state": _STATE,
        "env_generation": "sha256:" + "22" * 32,
        "failure_type": "menagerie.crawler.env_lifecycle.DiskRecoveryError",
    }
    values.update(overrides)
    return build_rebuild_deferral_row(**values)  # type: ignore[arg-type]


def test_rebuild_row_roundtrips_and_names_the_remedy() -> None:
    """A built row validates, carries the disposition, and names the re-drive."""

    row = _row()
    assert row["disposition"] == REBUILD_DEFERRAL_DISPOSITION
    assert REBUILD_DEFERRAL_DISPOSITION in MACHINE_DEFERRAL_DISPOSITIONS
    assert validate_rebuild_deferral_row(row) == row
    hint = row["recheck_hint"]
    assert "environment-cleanup-abc123" in hint
    assert "sha256:" + "22" * 32 in hint
    assert "re-drive" in hint
    assert "no build was attempted" in hint
    assert "environment-quarantine-cleared" in hint
    # The partition's fail-closed claim validation accepts the persisted row.
    assert machine_deferral_rejection("m_deadbeef", row) is None
    assert machine_deferral_rejection("m_other", row) is not None


def test_rebuild_row_rejects_tampered_digest_and_missing_provenance() -> None:
    """Digest, quarantine-provenance, and state-vector validation stay fail-closed."""

    row = _row()
    tampered = dict(row)
    tampered["name"] = "Renamed"
    with pytest.raises(RebuildDeferralError, match="digest"):
        validate_rebuild_deferral_row(tampered)
    stripped = {key: value for key, value in row.items() if key != "quarantine"}
    with pytest.raises(RebuildDeferralError, match="missing"):
        validate_rebuild_deferral_row(stripped)
    unnamed = dict(row, quarantine=dict(row["quarantine"], event_id=""))
    with pytest.raises(RebuildDeferralError, match="event_id"):
        validate_rebuild_deferral_row(unnamed)
    with pytest.raises(RebuildDeferralError, match="artifact_identity"):
        _row(artifact_identity="")
    with pytest.raises(RebuildDeferralError, match="unknown state"):
        _row(artifact_state={"osx-arm64.lock": {"state": "maybe"}})
    with pytest.raises(RebuildDeferralError, match="without its byte digest"):
        _row(artifact_state={"osx-arm64.lock": {"state": "present"}})
    with pytest.raises(RebuildDeferralError, match="extra fields"):
        _row(
            artifact_state={"osx-arm64.lock": {"state": "absent", "sha256": "sha256:" + "ee" * 32}}
        )


def test_rebuild_append_is_idempotent_per_quarantine_fact(tmp_path: Path) -> None:
    """The same park replays as a no-op; a NEW quarantine fact appends a new row.

    Rung-11 review finding 2: keying idempotency on the work generation alone
    resurrected the released first row forever -- a second quarantine (after a
    rotation) for the same work ID was silently discarded, so parking and
    releasing cycled unboundedly. The key is now the quarantine FACT identity.
    """

    ledger = rebuild_deferral_path(tmp_path)
    first = append_rebuild_deferral_row(ledger, _row())
    replay = append_rebuild_deferral_row(
        ledger, _row(created_at="2026-08-08T01:00:00Z", run_id="run-2")
    )
    assert replay == first
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    # Q1 -> rotation -> Q2: SAME work generation, new quarantine event and new
    # setup identity. The second fact persists instead of returning Q1.
    second_fact = append_rebuild_deferral_row(
        ledger,
        _row(
            quarantine_event_id="environment-cleanup-def456",
            artifact_identity="sha256:" + "99" * 32,
            env_generation="sha256:" + "88" * 32,
        ),
    )
    assert second_fact["quarantine"]["event_id"] == "environment-cleanup-def456"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2
    # Replaying Q2 stays idempotent, and the load surface returns the LATEST
    # fact per work generation (the current-row model).
    assert (
        append_rebuild_deferral_row(
            ledger,
            _row(
                quarantine_event_id="environment-cleanup-def456",
                artifact_identity="sha256:" + "99" * 32,
                env_generation="sha256:" + "88" * 32,
            ),
        )
        == second_fact
    )
    current = load_rebuild_deferral_rows([ledger])
    assert [row["quarantine"]["event_id"] for row in current] == ["environment-cleanup-def456"]
    # A NEW work generation remains a distinct park.
    other = append_rebuild_deferral_row(ledger, _row(work_id="work-m_deadbeef-2"))
    assert other != second_fact
    assert [row["work_id"] for row in load_rebuild_deferral_rows([ledger])] == [
        "work-m_deadbeef",
        "work-m_deadbeef-2",
    ]


def _fake_intent(tmp_path: Path) -> SimpleNamespace:
    """Build a duck-typed intent whose four setup artifacts exist on disk."""

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "osx-arm64.lock"
    export_path = lock_dir / "osx-arm64.resolved.json"
    export_hash_path = lock_dir / "osx-arm64.resolved.sha256"
    receipt_path = lock_dir / "osx-arm64.probes.json"
    lock_path.write_bytes(b"url#hash\n")
    export_path.write_bytes(b'{"packages": []}\n')
    export_hash_path.write_bytes(b"sha256:00\n")
    receipt_path.write_bytes(b"[]\n")
    return SimpleNamespace(
        name="core",
        lock=SimpleNamespace(
            lock_path=lock_path,
            export_path=export_path,
            export_hash_path=export_hash_path,
            target="osx-arm64",
        ),
    )


def _fake_driver(tmp_path: Path, intent: SimpleNamespace) -> SimpleNamespace:
    """Build a duck-typed driver exposing the registry and ledger paths."""

    models = tmp_path / "records" / "models" / "current-shard.jsonl"
    return SimpleNamespace(
        registry=SimpleNamespace(intents={"core": intent}),
        paths=SimpleNamespace(ledgers=SimpleNamespace(models=models)),
    )


def test_rebuild_row_holds_while_the_quarantined_artifacts_are_unchanged(
    tmp_path: Path,
) -> None:
    """The fact check mirrors the live valve: identity unchanged -> still parked."""

    intent = _fake_intent(tmp_path)
    identity = _intent_setup_artifact_identity(intent)  # type: ignore[arg-type]
    state = _intent_setup_artifact_state(intent)  # type: ignore[arg-type]
    driver = _fake_driver(tmp_path, intent)
    item = SimpleNamespace(route=SimpleNamespace(intent="core"))
    row = _row(artifact_identity=identity, artifact_state=state)
    assert _deferral_row_fact_still_holds(driver, item, row) is True  # type: ignore[arg-type]

    # A recovery/rotation re-solves the intent: the identity changes and the
    # row releases so the model is attempted normally.
    intent.lock.lock_path.write_bytes(b"url#rotated\n")
    assert _deferral_row_fact_still_holds(driver, item, row) is False  # type: ignore[arg-type]

    # A different routed intent or target can never be claimed by this row.
    intent.lock.lock_path.write_bytes(b"url#hash\n")
    other_item = SimpleNamespace(route=SimpleNamespace(intent="graph"))
    assert _deferral_row_fact_still_holds(driver, other_item, row) is False  # type: ignore[arg-type]


def test_status_partition_ignores_a_released_rebuild_deferral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public boundary accounting must re-check rebuild facts before claiming them."""

    records_root = tmp_path / "records"
    row = _row()
    append_rebuild_deferral_row(rebuild_deferral_path(records_root), row)
    item = IntakeItem(
        stable_id="m_deadbeef",
        name="Example",
        zoo="fixtures",
        variant="base",
        discovery_source="master_catalog",
        legacy_row_sha256="sha256:" + "ab" * 32,
        preserved_legacy_flags=(),
    )

    def released(_driver: Any, _item: Any, _row: Any) -> bool:
        """Report that the persisted quarantine fact no longer holds."""

        return False

    monkeypatch.setattr(cli, "_deferral_row_fact_still_holds", released, raising=False)
    monkeypatch.setattr(
        cli,
        "load_environment_registry",
        lambda *_args, **_kwargs: SimpleNamespace(intents={}),
    )

    selected = cli._status_machine_deferrals(
        records_root,
        repo_root=tmp_path,
        snapshot=SimpleNamespace(items=(item,)),
    )

    assert "m_deadbeef" not in selected


def test_partial_setup_families_are_typed_distinct_states(tmp_path: Path) -> None:
    """Distinct incomplete families never collapse to one shared identity.

    Rung-11 review finding 6: mapping every incomplete four-file family to
    ``None`` made shared absence read as positive identity, so different
    partial states -- and mutations to surviving members -- stayed parked
    together. Absence is now a typed per-member state inside the identity.
    """

    intent = _fake_intent(tmp_path)
    complete_identity = _intent_setup_artifact_identity(intent)  # type: ignore[arg-type]

    intent.lock.lock_path.unlink()
    missing_lock_identity = _intent_setup_artifact_identity(intent)  # type: ignore[arg-type]
    missing_lock_state = _intent_setup_artifact_state(intent)  # type: ignore[arg-type]
    assert isinstance(missing_lock_identity, str) and missing_lock_identity
    assert missing_lock_identity != complete_identity
    assert missing_lock_state["osx-arm64.lock"] == {"state": "absent"}

    # A DIFFERENT partial family is a DIFFERENT identity, not a shared match.
    intent.lock.lock_path.write_bytes(b"url#hash\n")
    intent.lock.export_path.unlink()
    missing_export_identity = _intent_setup_artifact_identity(intent)  # type: ignore[arg-type]
    assert missing_export_identity != missing_lock_identity
    assert missing_export_identity != complete_identity

    # Mutating a SURVIVING member of the same partial family also moves it.
    intent.lock.lock_path.write_bytes(b"url#mutated\n")
    assert _intent_setup_artifact_identity(intent) != missing_export_identity  # type: ignore[arg-type]

    # A row parked on one partial state releases on any other partial state.
    driver = _fake_driver(tmp_path, intent)
    item = SimpleNamespace(route=SimpleNamespace(intent="core"))
    row = _row(artifact_identity=missing_lock_identity, artifact_state=missing_lock_state)
    assert _deferral_row_fact_still_holds(driver, item, row) is False  # type: ignore[arg-type]


def test_clearance_receipt_releases_row_and_live_valve(tmp_path: Path) -> None:
    """An explicit operator clearance releases byte-identical setup artifacts.

    Rung-11 review finding 6: an out-of-band backend/tool repair leaves the
    setup artifacts unchanged, so identity comparison alone can never release
    the park. The typed ``environment-quarantine-cleared`` receipt is the
    honest release signal for that case.
    """

    intent = _fake_intent(tmp_path)
    identity = _intent_setup_artifact_identity(intent)  # type: ignore[arg-type]
    state = _intent_setup_artifact_state(intent)  # type: ignore[arg-type]
    driver = _fake_driver(tmp_path, intent)
    item = SimpleNamespace(route=SimpleNamespace(intent="core"))
    row = _row(artifact_identity=identity, artifact_state=state)
    assert _deferral_row_fact_still_holds(driver, item, row) is True  # type: ignore[arg-type]

    events_path = canonical_operational_ledger_path(driver.paths.ledgers.models)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    clearance = {
        "event_kind": OperationalEventKind.CAMPAIGN_HEALTH.value,
        "details": {
            "disposition": ENVIRONMENT_QUARANTINE_CLEARED_DISPOSITION,
            "quarantine_event_id": "environment-cleanup-abc123",
        },
    }
    events_path.write_text(json.dumps(clearance) + "\n", encoding="utf-8")

    assert _quarantine_is_cleared([clearance], "environment-cleanup-abc123") is True
    assert _quarantine_is_cleared([clearance], "environment-cleanup-def456") is False
    # The persisted row releases through the receipt with unchanged bytes.
    assert _deferral_row_fact_still_holds(driver, item, row) is False  # type: ignore[arg-type]
    # A row parked on a DIFFERENT (uncleared) quarantine keeps holding.
    other = _row(
        quarantine_event_id="environment-cleanup-def456",
        artifact_identity=identity,
        artifact_state=state,
    )
    assert _deferral_row_fact_still_holds(driver, item, other) is True  # type: ignore[arg-type]


def test_quarantine_code_identity_drift_names_moved_components() -> None:
    """Only non-record components may move; any record drift refuses acceptance."""

    recorded = {
        "stable_id": "m_deadbeef",
        "intake_item_sha256": "sha256:aa",
        "proposal_sha256": "sha256:bb",
        "source_identity": "sha256:cc",
        "evidence_identity": "sha256:dd",
        "recipe_revision": "sha256:ee",
        "vet_identity": "sha256:ff",
        "fidelity_identity": None,
        "author_prompt": "sha256:11",
        "template_source_revision": None,
        "work_id": "work-1",
        "checker_prompt": "sha256:22",
        "runner_identity": "sha256:33",
        "award_closure": "sha256:44",
        "campaign_root_work_id": "campaign-1",
    }
    live = dict(recorded, award_closure="sha256:55", checker_prompt="sha256:66")
    assert _quarantine_code_identity_drift(recorded, live) == (
        "award_closure",
        "checker_prompt",
    )
    assert _quarantine_code_identity_drift(recorded, dict(recorded)) == ()
    # Record drift refuses: the completed work is genuinely stale.
    stale = dict(live, proposal_sha256="sha256:other")
    assert _quarantine_code_identity_drift(recorded, stale) is None


def _v1_row(**overrides: object) -> dict:
    """Reconstruct the EXACT row shape the v1 implementation persisted."""

    row: dict = {
        "schema_version": REBUILD_DEFERRAL_SCHEMA_VERSION_V1,
        "stable_id": "m_deadbeef",
        "work_id": "work-m_deadbeef",
        "name": "First",
        "campaign_id": "campaign-1",
        "run_id": "run-legacy",
        "machine_id": "machine-1",
        "created_at": "2026-08-07T00:00:00Z",
        "disposition": REBUILD_DEFERRAL_DISPOSITION,
        "quarantine": {
            "event_id": "environment-cleanup-abc123",
            "intent": "core",
            "target": "osx-arm64",
            "artifact_identity": None,
            "env_generation": "sha256:" + "22" * 32,
            "failure_type": "menagerie.crawler.env_lifecycle.DiskRecoveryError",
        },
        "recheck_hint": (
            "environment generation is quarantined; recover or rotate the "
            "environment, then re-drive this work item."
        ),
    }
    row.update(overrides)
    row.pop("row_sha256", None)
    row["row_sha256"] = stable_hash(row)
    return row


def test_legacy_v1_rows_stay_readable_and_v2_governs(tmp_path: Path) -> None:
    """A durable v1 row never aborts a load, and a v2 row supersedes it.

    Sol round-2 review finding 2: v2-only validation made a digest-valid v1
    row abort every ledger read, so an upgraded campaign holding round-1 parks
    could not even report status. v1 rows are read-only history: loads accept
    them, appends refuse them, and the v2 row governs its work generation.
    """

    ledger = rebuild_deferral_path(tmp_path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    legacy = _v1_row()
    ledger.write_bytes(canonical_json_bytes(legacy) + b"\n")

    # The exact former v1 shape validates and loads as the governing row
    # while no v2 row exists, and the partition claim validation accepts it.
    assert validate_rebuild_deferral_row(legacy) == legacy
    assert load_rebuild_deferral_rows([ledger]) == (legacy,)
    assert machine_deferral_rejection("m_deadbeef", legacy) is None

    # A fresh derivation of the SAME quarantine fact persists the v2 upgrade
    # instead of resurrecting the v1 row, and the v2 row then governs.
    upgraded = append_rebuild_deferral_row(ledger, _row())
    assert upgraded["schema_version"] == REBUILD_DEFERRAL_SCHEMA_VERSION
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2
    governed = load_rebuild_deferral_rows([ledger])
    assert [row["schema_version"] for row in governed] == [REBUILD_DEFERRAL_SCHEMA_VERSION]
    # The governs rule is deterministic regardless of interleaving: latest v2
    # wins over any v1 for the same work generation.
    assert current_rebuild_rows([upgraded, legacy]) == (upgraded,)

    # A DIFFERENT work generation with only v1 history keeps its v1 row.
    other_legacy = _v1_row(work_id="work-m_deadbeef-legacy-only")
    with ledger.open("ab") as handle:
        handle.write(canonical_json_bytes(other_legacy) + b"\n")
    governed = load_rebuild_deferral_rows([ledger])
    assert {row["work_id"]: row["schema_version"] for row in governed} == {
        "work-m_deadbeef": REBUILD_DEFERRAL_SCHEMA_VERSION,
        "work-m_deadbeef-legacy-only": REBUILD_DEFERRAL_SCHEMA_VERSION_V1,
    }


def test_rebuild_appends_refuse_legacy_v1_candidates(tmp_path: Path) -> None:
    """Appends are v2-only; v1 is a read-only historical schema."""

    ledger = rebuild_deferral_path(tmp_path)
    with pytest.raises(RebuildDeferralError, match="v2-only"):
        append_rebuild_deferral_row(ledger, _v1_row())
    assert not ledger.exists()
    # A truly unknown schema stays refused on both paths.
    with pytest.raises(RebuildDeferralError, match="unknown schema"):
        validate_rebuild_deferral_row(_v1_row(schema_version="rebuild.v0"))


def test_rebuild_ledger_rejects_unreadable_rows(tmp_path: Path) -> None:
    """A corrupt ledger line fails loudly instead of silently dropping claims."""

    ledger = rebuild_deferral_path(tmp_path)
    append_rebuild_deferral_row(ledger, _row())
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("{not-json\n")
    with pytest.raises(RebuildDeferralError, match="unreadable"):
        load_rebuild_deferral_rows([ledger])
    assert json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
