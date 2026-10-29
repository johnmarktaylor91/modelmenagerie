"""D-59 (OC-12): scheduler ownership registry + decommission verb.

The 200a incident this forecloses: a legitimately installed */15min wake cron
(episode b1e1e9ee...) outlived its owning campaign ledger and ran
``crawler run --resume`` silently for days — nothing on the HOST said who
owned the entry, so nothing could lawfully find or remove it. The registry is
a machine-scoped append-only ledger written in the same transaction as every
scheduler install/deactivation; the census reverse-sweeps the host (crontab
wake blocks, launchd agent-root labels) and joins it against the registry and
the owning operational ledgers; the decommission verb removes orphans with a
provenance snapshot and refuses entries a live owner still holds active.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
from menagerie.crawler.recordio import JsonlLedger
from menagerie.crawler.scheduler_ownership import (
    CRON_WAKE_MARKER_PREFIX,
    SCHEDULER_OWNERSHIP_SCHEMA_VERSION,
    SchedulerOwnershipError,
    SchedulerOwnershipRegistry,
    census_host_entries,
    decommission_entry,
    evaluate_ownership_census,
    parse_cron_wake_blocks,
)
from menagerie.crawler.tools.scheduler_ownership import main as tool_main
from menagerie.crawler.wakeup import OperationalContext, WakeupBackend, WakeupManager

pytestmark = pytest.mark.smoke

NOW = "2026-08-25T12:00:00Z"
RESET = "2026-08-25T13:00:00Z"

_B1E1E9EE_ID = "wake-episode-b1e1e9ee7e5393005c1fce01aef8eb1c"
#: The exact stale-cron shape WS0 found on the host (wave-0 provenance
#: snapshots, 2026-08-24): a well-formed wakeup block whose owning ledger no
#: longer existed to deactivate it.
_B1E1E9EE_BLOCK = (
    f"{CRON_WAKE_MARKER_PREFIX}{_B1E1E9EE_ID} begin\n"
    "CRON_TZ=UTC\n"
    f"*/15 * * * * python -m menagerie.crawler run --resume --wake-episode-id {_B1E1E9EE_ID}\n"
    f"{CRON_WAKE_MARKER_PREFIX}{_B1E1E9EE_ID} end\n"
)


class FakeScheduler:
    """In-memory verified scheduler backend (wakeup-test idiom)."""

    def __init__(self) -> None:
        self.registrations: dict[str, str] = {}

    def install(self, definition) -> None:  # noqa: ANN001 - wakeup callback shape
        self.registrations[definition.episode.episode_id] = definition.scheduler_definition

    def verify(self, definition) -> bool:  # noqa: ANN001
        return (
            self.registrations.get(definition.episode.episode_id) == definition.scheduler_definition
        )

    def deactivate(self, definition) -> None:  # noqa: ANN001
        self.registrations.pop(definition.episode.episode_id, None)


def _context() -> OperationalContext:
    return OperationalContext("run-test", "machine-test", {"author": 1}, None)


def _registry(tmp_path: Path) -> SchedulerOwnershipRegistry:
    return SchedulerOwnershipRegistry(tmp_path / "state" / "scheduler-ownership.jsonl")


def _paused_manager(
    tmp_path: Path, ledger: JsonlLedger, registry: SchedulerOwnershipRegistry
) -> WakeupManager:
    scheduler = FakeScheduler()
    manager = WakeupManager(
        tmp_path / "wakeup",
        ledger,
        ["python", "-m", "menagerie.crawler", "run", "--resume"],
        backend=WakeupBackend.CRON,
        installer=scheduler.install,
        verifier=scheduler.verify,
        deactivator=scheduler.deactivate,
        ownership_registry=registry,
    )
    manager.record_pause_and_schedule(
        provider="anthropic",
        observed_response="429",
        reset_at=RESET,
        context=_context(),
        created_at=NOW,
    )
    return manager


def test_parse_cron_wake_blocks_recovers_the_incident_shape() -> None:
    blocks = parse_cron_wake_blocks("MAILTO=x\n" + _B1E1E9EE_BLOCK + "5 4 * * * true\n")
    assert [entry_id for entry_id, _ in blocks] == [_B1E1E9EE_ID]
    assert blocks[0][1] == _B1E1E9EE_BLOCK


def test_unregistered_host_entry_is_flagged(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    entries = census_host_entries(crontab_text=_B1E1E9EE_BLOCK, agent_root=tmp_path / "agents")
    census = evaluate_ownership_census(registry, entries)
    assert census["status"] == "attention"
    (row,) = census["entries"]
    assert row["entry_id"] == _B1E1E9EE_ID
    assert row["classification"] == "unregistered"
    assert row["decommission_eligible"] is True


def test_install_registers_ownership_in_the_same_transaction(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        _paused_manager(tmp_path, ledger, registry)
    rows = registry.rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == SCHEDULER_OWNERSHIP_SCHEMA_VERSION
    assert row["kind"] == "registered"
    assert row["backend"] == "cron"
    assert row["machine_id"] == "machine-test"
    assert row["run_id"] == "run-test"
    assert row["owning_ledger"].endswith("operational.jsonl")
    assert row["entry_id"].startswith("wake-episode-")
    assert row["row_sha256"].startswith("sha256:")


def test_registered_live_entry_census_and_decommission_refusal(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _paused_manager(tmp_path, ledger, registry)
        episode_id = manager.projection.active_episode_ids[0]
        block = (
            f"{CRON_WAKE_MARKER_PREFIX}{episode_id} begin\n"
            "* * * * * true\n"
            f"{CRON_WAKE_MARKER_PREFIX}{episode_id} end\n"
        )
        entries = census_host_entries(crontab_text=block, agent_root=tmp_path / "agents")
        census = evaluate_ownership_census(registry, entries)
        (row,) = census["entries"]
        assert row["classification"] == "registered-live"
        assert row["decommission_eligible"] is False

        crontab = {"text": block}
        with pytest.raises(SchedulerOwnershipError, match="live"):
            decommission_entry(
                registry,
                entries[0],
                reason="operator cleanup",
                context=_context(),
                created_at=NOW,
                crontab_reader=lambda: crontab["text"],
                crontab_writer=lambda text: crontab.__setitem__("text", text),
            )


def test_deactivation_records_decommission_row(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _paused_manager(tmp_path, ledger, registry)
        episode_id = manager.projection.active_episode_ids[0]
        manager.resolve_episode(
            episode_id,
            resolution="operator-cancelled",
            context=_context(),
            created_at=NOW,
        )
        manager.reconcile(context=_context(), created_at=NOW)
    kinds = [row["kind"] for row in registry.rows()]
    assert kinds == ["registered", "decommissioned"]
    decommissioned = registry.rows()[-1]
    assert decommissioned["entry_id"] == episode_id
    assert decommissioned["verb"] == "wakeup-deactivation"


def test_decommission_verb_removes_orphan_with_provenance(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    crontab = {"text": "MAILTO=x\n" + _B1E1E9EE_BLOCK}
    entries = census_host_entries(crontab_text=crontab["text"], agent_root=tmp_path / "agents")
    row = decommission_entry(
        registry,
        entries[0],
        reason="stale wake cron: owning ledger lost (wave-0 incident class)",
        context=_context(),
        created_at=NOW,
        crontab_reader=lambda: crontab["text"],
        crontab_writer=lambda text: crontab.__setitem__("text", text),
    )
    assert row["kind"] == "decommissioned"
    assert row["verb"] == "decommission-tool"
    assert row["removed_content"] == _B1E1E9EE_BLOCK
    assert row["reason"].startswith("stale wake cron")
    assert _B1E1E9EE_ID not in crontab["text"]
    assert "MAILTO=x" in crontab["text"]

    # The host is now clean: a fresh census over the rewritten crontab is green.
    after = census_host_entries(crontab_text=crontab["text"], agent_root=tmp_path / "agents")
    assert evaluate_ownership_census(registry, after)["status"] == "clean"


def test_decommission_refuses_when_residue_survives(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    crontab = {"text": _B1E1E9EE_BLOCK}
    entries = census_host_entries(crontab_text=crontab["text"], agent_root=tmp_path / "agents")
    with pytest.raises(SchedulerOwnershipError, match="residue"):
        decommission_entry(
            registry,
            entries[0],
            reason="cleanup",
            context=_context(),
            created_at=NOW,
            crontab_reader=lambda: crontab["text"],
            # A writer that silently drops the rewrite: residue must refuse.
            crontab_writer=lambda text: None,
        )
    # No decommission row may stand over surviving residue.
    assert [row["kind"] for row in registry.rows()] == []


def test_owner_lost_registered_entry_is_orphaned(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with JsonlLedger(tmp_path / "operational.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
        manager = _paused_manager(tmp_path, ledger, registry)
        episode_id = manager.projection.active_episode_ids[0]
    (tmp_path / "operational.jsonl").unlink()
    block = (
        f"{CRON_WAKE_MARKER_PREFIX}{episode_id} begin\n"
        "* * * * * true\n"
        f"{CRON_WAKE_MARKER_PREFIX}{episode_id} end\n"
    )
    entries = census_host_entries(crontab_text=block, agent_root=tmp_path / "agents")
    census = evaluate_ownership_census(registry, entries)
    (row,) = census["entries"]
    assert row["classification"] == "orphaned-owner-lost"
    assert row["decommission_eligible"] is True


def test_decommissioned_residue_is_flagged(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    crontab = {"text": _B1E1E9EE_BLOCK}
    entries = census_host_entries(crontab_text=crontab["text"], agent_root=tmp_path / "agents")
    decommission_entry(
        registry,
        entries[0],
        reason="cleanup",
        context=_context(),
        created_at=NOW,
        crontab_reader=lambda: crontab["text"],
        crontab_writer=lambda text: crontab.__setitem__("text", text),
    )
    # The same block reappears (e.g. restored from a crontab backup).
    reappeared = census_host_entries(
        crontab_text=_B1E1E9EE_BLOCK, agent_root=tmp_path / "agents"
    )
    census = evaluate_ownership_census(registry, reappeared)
    (row,) = census["entries"]
    assert row["classification"] == "decommissioned-residue"
    assert census["status"] == "attention"


def test_launchd_agent_root_entries_join_the_census(tmp_path: Path) -> None:
    agent_root = tmp_path / "agents"
    agent_root.mkdir()
    label = "org.menagerie.crawler-wake.wake-episode-feedbeeffeedbeeffeedbeeffeedbeef"
    (agent_root / f"{label}.plist").write_text("<plist/>", encoding="utf-8")
    entries = census_host_entries(crontab_text="", agent_root=agent_root)
    assert [entry.backend for entry in entries] == ["launchd"]
    assert entries[0].entry_id == "wake-episode-feedbeeffeedbeeffeedbeeffeedbeef"
    census = evaluate_ownership_census(_registry(tmp_path), entries)
    assert census["entries"][0]["classification"] == "unregistered"


def test_census_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry_path = tmp_path / "scheduler-ownership.jsonl"
    crontab_file = tmp_path / "crontab.txt"
    crontab_file.write_text(_B1E1E9EE_BLOCK, encoding="utf-8")
    agent_root = tmp_path / "agents"
    agent_root.mkdir()
    assert (
        tool_main(
            [
                "census",
                "--registry",
                str(registry_path),
                "--crontab-file",
                str(crontab_file),
                "--agent-root",
                str(agent_root),
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "attention"

    crontab_file.write_text("", encoding="utf-8")
    assert (
        tool_main(
            [
                "census",
                "--registry",
                str(registry_path),
                "--crontab-file",
                str(crontab_file),
                "--agent-root",
                str(agent_root),
            ]
        )
        == 0
    )
