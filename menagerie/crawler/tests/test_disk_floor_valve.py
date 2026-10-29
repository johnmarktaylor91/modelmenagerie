"""Disk-floor machine-deferral valve: host disk state never terminalizes a model.

Rung-12 census, class A: when the host fell below the env lane's 30-GiB disk
floor, six models were TERMINALIZED ``failed:resource/disk-floor`` -- two of
them THROUGH consumed review-held supersession grants, and one burying its own
honest ``failed:environment/solve-failed`` negative. A full disk is a MACHINE
condition, not a fact about any model: no solve, build, or probe ran. The
adjudicated fix is the same shape as the rung-11 quarantine valve: the refusal
parks the affected work identities in a durable machine-deferral ledger
(withheld, recoverable, re-drivable on the next leg or another host) and mints
NO model terminal. Genuine per-model resource failures keep terminalizing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.disk_floor import (
    DISK_FLOOR_DEFERRAL_DISPOSITION,
    DiskFloorDeferralError,
    append_disk_floor_deferral_row,
    build_disk_floor_deferral_row,
    disk_floor_deferral_path,
    load_disk_floor_deferral_rows,
    validate_disk_floor_deferral_row,
)
from menagerie.crawler.driver_admission import _deferral_row_fact_still_holds
from menagerie.crawler.driver_progress import _environment_failure
from menagerie.crawler.env_lifecycle import DiskFloorError, DiskRecoveryError
from menagerie.crawler.environment_rebuild import (
    load_rebuild_deferral_rows,
    rebuild_deferral_path,
)
from menagerie.crawler.envs import EnvironmentIntent
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.status import (
    MACHINE_DEFERRAL_DISPOSITIONS,
    machine_deferral_rejection,
    partition_report,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeEnvironments,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke

_FLOOR = 30 * 1024**3


def _row(**overrides: object) -> dict:
    """Build one valid disk-floor deferral row with optional field overrides."""

    values = {
        "stable_id": "m_deadbeef",
        "work_id": "work-m_deadbeef",
        "name": "First",
        "campaign_id": "campaign-1",
        "run_id": "run-1",
        "machine_id": "machine-1",
        "created_at": "2026-08-11T00:00:00Z",
        "intent": "core",
        "target": "osx-arm64",
        "env_root": "/tmp/envs",
        "floor_bytes": _FLOOR,
        "free_bytes": 5 * 1024**3,
    }
    values.update(overrides)
    return build_disk_floor_deferral_row(**values)  # type: ignore[arg-type]


def test_disk_floor_row_roundtrips_and_names_the_remedy() -> None:
    """A built row validates, is partition-claimable, and names the re-drive."""

    row = _row()
    assert row["disposition"] == DISK_FLOOR_DEFERRAL_DISPOSITION
    assert DISK_FLOOR_DEFERRAL_DISPOSITION in MACHINE_DEFERRAL_DISPOSITIONS
    assert validate_disk_floor_deferral_row(row) == row
    hint = row["recheck_hint"]
    assert "machine-1" in hint
    assert str(_FLOOR) in hint
    assert "re-drive" in hint
    assert "never" in hint and "model" in hint
    # The partition's fail-closed claim validation accepts the persisted row.
    assert machine_deferral_rejection("m_deadbeef", row) is None
    assert machine_deferral_rejection("m_other", row) is not None


def test_disk_floor_row_rejects_tampering_and_non_refusing_measurements() -> None:
    """Digest, provenance, and measurement validation stay fail-closed."""

    row = _row()
    tampered = dict(row)
    tampered["name"] = "Renamed"
    with pytest.raises(DiskFloorDeferralError, match="digest"):
        validate_disk_floor_deferral_row(tampered)
    stripped = {key: value for key, value in row.items() if key != "disk"}
    with pytest.raises(DiskFloorDeferralError, match="missing"):
        validate_disk_floor_deferral_row(stripped)
    with pytest.raises(DiskFloorDeferralError, match="intent"):
        _row(intent="")
    with pytest.raises(DiskFloorDeferralError, match="floor"):
        _row(floor_bytes=0)
    with pytest.raises(DiskFloorDeferralError, match="floor"):
        _row(floor_bytes=True)
    # Only an actual refusal may be recorded: a measurement at or above the
    # floor is not a park fact, exactly as only a refusing capacity assessment
    # may be recorded by the host-capacity valve.
    with pytest.raises(DiskFloorDeferralError, match="refus"):
        _row(free_bytes=_FLOOR)
    with pytest.raises(DiskFloorDeferralError, match="free_bytes"):
        _row(free_bytes=-1)


def test_disk_floor_append_is_idempotent_per_host_fact(tmp_path: Path) -> None:
    """The same starved-host park replays as a no-op; a new host appends anew."""

    ledger = disk_floor_deferral_path(tmp_path)
    first = append_disk_floor_deferral_row(ledger, _row())
    # A resume on the same starved machine re-measures different free bytes and
    # stamps a new created_at/run_id: the SAME host fact, so a no-op.
    replay = append_disk_floor_deferral_row(
        ledger,
        _row(
            created_at="2026-08-11T01:00:00Z",
            run_id="run-2",
            free_bytes=6 * 1024**3,
        ),
    )
    assert replay == first
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    # A refusal observed on a DIFFERENT machine is a new fact and governs.
    second = append_disk_floor_deferral_row(ledger, _row(machine_id="machine-2"))
    assert second != first
    current = load_disk_floor_deferral_rows([ledger])
    assert [row["machine_id"] for row in current] == ["machine-2"]
    # A new work generation remains a distinct park.
    other = append_disk_floor_deferral_row(ledger, _row(work_id="work-m_deadbeef-2"))
    assert other != second
    assert [row["work_id"] for row in load_disk_floor_deferral_rows([ledger])] == [
        "work-m_deadbeef",
        "work-m_deadbeef-2",
    ]


def test_environment_failure_refuses_to_terminalize_a_disk_floor_refusal() -> None:
    """The routing registry can never map a disk condition to a model terminal."""

    error = DiskFloorError("floor", floor_bytes=_FLOOR, free_bytes=1, env_root="/tmp/envs")
    assert _environment_failure(error).kind == "park-disk-floor"
    # A NON-floor disk-recovery failure (teardown residue) is a host condition:
    # it parks as a machine deferral instead of re-minting the abolished
    # ``failed:resource/disk-floor`` terminal (T3-T2c). Pre-existing terminal
    # rows of that shape stay readable history; no new ones can be minted.
    assert _environment_failure(DiskRecoveryError("residue")).kind == ("park-host-condition")


class _DiskFloorEnvironments(FakeEnvironments):
    """Refuse every environment build at the host disk floor, pre-use."""

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Raise the typed floor refusal before any model callback begins."""

        del use
        raise DiskFloorError(
            f"less than the required {_FLOOR} bytes of free disk is available",
            floor_bytes=_FLOOR,
            free_bytes=5 * 1024**3,
            env_root=str(self.root),
        )


def test_disk_floor_refusal_defers_models_instead_of_terminalizing(
    tmp_path: Path,
) -> None:
    """A tripped floor parks every affected model recoverably; no terminal mints."""

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    result = _driver(
        tmp_path,
        snapshot,
        environments=_DiskFloorEnvironments(tmp_path / "fake-envs"),
    ).run()

    assert result.status == "complete-with-deferrals"
    # THE fix: host disk state mints no model terminal.
    assert scan_jsonl(paths.ledgers.models) == []
    records_root = paths.ledgers.models.parent.parent
    rows = load_disk_floor_deferral_rows([disk_floor_deferral_path(records_root)])
    assert {row["stable_id"] for row in rows} == {item.stable_id for item in snapshot.items}
    for row in rows:
        assert row["disposition"] == DISK_FLOOR_DEFERRAL_DISPOSITION
        assert row["machine_id"] == "machine-test"
        assert row["disk"]["floor_bytes"] == _FLOOR
        assert row["disk"]["free_bytes"] == 5 * 1024**3
        assert machine_deferral_rejection(str(row["stable_id"]), row) is None
    events = scan_jsonl(canonical_operational_ledger_path(paths.ledgers.models))
    deferred_events = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "host-disk-floor-deferred"
    ]
    assert {event["details"]["stable_id"] for event in deferred_events} == {
        item.stable_id for item in snapshot.items
    }

    # A resume against the still-starved ledger is idempotent: same parks, no
    # duplicate rows, still no terminals.
    second = _driver(
        tmp_path,
        snapshot,
        environments=_DiskFloorEnvironments(tmp_path / "fake-envs"),
    ).run()
    assert second.status == "complete-with-deferrals"
    assert scan_jsonl(paths.ledgers.models) == []
    ledger = disk_floor_deferral_path(records_root)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == len(snapshot.items)


class _PreUseDiskRecoveryEnvironments(FakeEnvironments):
    """Raise the non-floor disk-recovery failure before any model callback."""

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Fail with the base class the floor no longer raises."""

        del use
        raise DiskRecoveryError("synthetic pre-use disk recovery failure")


def test_pre_use_generic_disk_recovery_error_parks_as_a_host_condition(
    tmp_path: Path,
) -> None:
    """Non-floor disk recovery is a HOST condition: it defers, never terminalizes.

    The old table row ``DiskRecoveryError -> ("resource", "disk-floor")``
    re-minted the abolished rung-12 terminal for teardown residue (T3-T2c).
    Residue is host state -- cleanup, then re-drive -- so the models park in
    the environment-failure machine-deferral ledger instead.
    """

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    result = _driver(
        tmp_path,
        snapshot,
        environments=_PreUseDiskRecoveryEnvironments(tmp_path / "fake-envs"),
    ).run()

    assert result.status == "complete-with-deferrals"
    assert scan_jsonl(paths.ledgers.models) == []
    records_root = paths.ledgers.models.parent.parent
    assert load_disk_floor_deferral_rows([disk_floor_deferral_path(records_root)]) == ()
    parked = load_rebuild_deferral_rows([rebuild_deferral_path(records_root)])
    assert {row["stable_id"] for row in parked} == {item.stable_id for item in snapshot.items}
    assert {row["quarantine"]["condition_scope"] for row in parked} == {"host-condition"}


def test_disk_floor_row_holds_only_on_the_recording_starved_host(
    tmp_path: Path,
) -> None:
    """The fact re-check keys on the recording host and the live measurement."""

    env_root = tmp_path / "envs"
    env_root.mkdir()
    driver = SimpleNamespace(config=SimpleNamespace(machine_id="machine-1"))
    item = SimpleNamespace(route=SimpleNamespace(intent="core"))

    # An impossibly high recorded floor is still unmet here and now: the row
    # keeps claiming the model on the SAME machine.
    starved = _row(env_root=str(env_root), floor_bytes=2**62, free_bytes=1)
    assert _deferral_row_fact_still_holds(driver, item, starved) is True  # type: ignore[arg-type]

    # Any other machine attempts normally -- that is the re-drive path.
    other_host = SimpleNamespace(config=SimpleNamespace(machine_id="machine-2"))
    assert _deferral_row_fact_still_holds(other_host, item, starved) is False  # type: ignore[arg-type]

    # A recovered disk (live free bytes above the recorded floor) releases it.
    recovered = _row(env_root=str(env_root), floor_bytes=2, free_bytes=1)
    assert _deferral_row_fact_still_holds(driver, item, recovered) is False  # type: ignore[arg-type]

    # A vanished environment root is unknown, never "still provably starved".
    missing = _row(env_root=str(tmp_path / "gone"), floor_bytes=2**62, free_bytes=1)
    assert _deferral_row_fact_still_holds(driver, item, missing) is False  # type: ignore[arg-type]


def test_partition_accounts_disk_floor_rows_and_terminals_supersede() -> None:
    """A valve row claims its model; a later real terminal wins over the park."""

    row = _row(stable_id="m4334", work_id="work-m4334")
    report = partition_report(["m4334"], [], machine_deferrals={"m4334": row})
    assert report.valid
    assert "m4334" in report.machine_deferred.get(DISK_FLOOR_DEFERRAL_DISPOSITION, ())
    terminal = {"stable_id": "m4334", "status": {"code": "failed:environment"}}
    superseded = partition_report(["m4334"], [terminal], machine_deferrals={"m4334": row})
    assert superseded.valid
    assert "m4334" in superseded.superseded_deferral_ids


def test_tmp_role_row_releases_when_tmp_destination_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tmp-role starvation fact is configuration-bound: moving TMPDIR releases.

    The recorded path may still be starved, but tmp is chosen by configuration;
    a moved destination means the row describes a path no longer in use. The
    live per-destination floor check re-derives the park against the NEW tmp if
    that one is genuinely starved, so failing open here can never skip a real
    refusal -- it only stops a cured condition from claiming models forever.
    """

    from menagerie.crawler.driver_admission import _deferral_row_fact_still_holds
    from menagerie.crawler.disk_floor import DISK_FLOOR_DEFERRAL_DISPOSITION

    recorded = tmp_path / "old-tmp"
    recorded.mkdir()
    row = {
        "disposition": DISK_FLOOR_DEFERRAL_DISPOSITION,
        "machine_id": "test-host",
        "disk": {
            "condition": "floor",
            "destination_role": "tmp",
            "env_root": str(recorded),
            "floor_bytes": 1 << 60,  # absurd floor: recorded path IS starved
            "free_bytes": 0,
            "intent": "graph",
            "target": "osx-arm64",
        },
    }

    class _Config:
        machine_id = "test-host"

    class _Driver:
        config = _Config()

    import tempfile as _tempfile

    # Same tmp destination as recorded -> the fact holds (still starved).
    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(recorded))
    assert _deferral_row_fact_still_holds(_Driver(), object(), row) is True

    # TMPDIR moved elsewhere -> the recorded fact is about a retired path.
    moved = tmp_path / "new-tmp"
    moved.mkdir()
    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(moved))
    assert _deferral_row_fact_still_holds(_Driver(), object(), row) is False
