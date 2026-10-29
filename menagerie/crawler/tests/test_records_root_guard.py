"""LC-22/UU-9 interim refusals: leg open guards the records root's substrate.

Every torn-tail recovery, append lock, and chain verification in the ledger
spine assumes ``fcntl.flock`` on the records root actually excludes writers.
These pins make the assumption a REFUSAL instead of a hope: a leg cannot
open on a filesystem outside the closed local-flock allowlist (or one whose
flock probe fails), and a root last written by another machine refuses until
claimed deliberately. The full multi-host lease/fencing story is the
step-500 charter; the foreign-live-``leg_run`` arm needs Wave-2 invocation
identity and is deliberately absent here.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from menagerie.crawler import leg_generation as leg_generation_module
from menagerie.crawler import records_root_guard as guard
from menagerie.crawler.records_root_guard import (
    FLOCK_SAFE_FSTYPES,
    RecordsRootGuardError,
    assert_and_stamp_records_root_writer,
    assert_records_root_flock_safe,
    claim_records_root,
    last_writer_stamp,
)

pytestmark = pytest.mark.smoke


def test_a_local_filesystem_passes_and_reports_its_type(tmp_path: Path) -> None:
    """Positive control: the suite's own tmp roots are flock-safe."""

    observed = assert_records_root_flock_safe(tmp_path / "records")
    assert observed in FLOCK_SAFE_FSTYPES


def test_a_network_filesystem_refuses_the_open_with_the_type_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interim LC-22 refusal: flock on NFS is a lie, so the open refuses."""

    monkeypatch.setattr(guard, "resolve_records_fstype", lambda _path: "nfs")
    with pytest.raises(RecordsRootGuardError) as excinfo:
        assert_records_root_flock_safe(tmp_path / "records")
    assert "'nfs'" in str(excinfo.value)
    assert "filesystem_capability" in str(excinfo.value)


def test_an_undeterminable_filesystem_refuses_rather_than_passing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown is refused, never silently trusted (fail-closed)."""

    monkeypatch.setattr(guard, "resolve_records_fstype", lambda _path: None)
    with pytest.raises(RecordsRootGuardError):
        assert_records_root_flock_safe(tmp_path / "records")
    # The capability names EXACTLY what it accepts.
    observed = assert_records_root_flock_safe(
        tmp_path / "records", filesystem_capability="unknown"
    )
    assert observed == "unknown"


def test_the_capability_accepts_only_the_exact_observed_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capability for one filesystem never blankets another."""

    monkeypatch.setattr(guard, "resolve_records_fstype", lambda _path: "nfs")
    with pytest.raises(RecordsRootGuardError):
        assert_records_root_flock_safe(tmp_path / "records", filesystem_capability="smbfs")
    assert (
        assert_records_root_flock_safe(tmp_path / "records", filesystem_capability="nfs")
        == "nfs"
    )


def test_a_failing_flock_probe_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A filesystem that LOOKS safe but refuses the probe still refuses."""

    def broken(_root: Path) -> None:
        raise OSError("flock not supported")

    monkeypatch.setattr(guard, "_probe_flock", broken)
    with pytest.raises(RecordsRootGuardError) as excinfo:
        assert_records_root_flock_safe(tmp_path / "records")
    assert "probe" in str(excinfo.value)


def test_a_foreign_last_writer_refuses_until_claimed(tmp_path: Path) -> None:
    """The moved-root arm: another machine's stamp refuses; a claim transfers.

    flock cannot see across hosts, so a root carried between machines may
    still have a live orchestrator on the other side; only the deliberate,
    reasoned claim transfers ownership.
    """

    root = tmp_path / "records"
    first = assert_and_stamp_records_root_writer(root, machine_id="host-a")
    assert first["machine_id"] == "host-a"

    with pytest.raises(RecordsRootGuardError) as excinfo:
        assert_and_stamp_records_root_writer(root, machine_id="host-b")
    assert "host-a" in str(excinfo.value) and "host-b" in str(excinfo.value)
    assert "claim_records_root" in str(excinfo.value)

    with pytest.raises(RecordsRootGuardError):
        claim_records_root(root, reason="   ", machine_id="host-b")
    claimed = claim_records_root(
        root, reason="host-a decommissioned 2026-08-24", machine_id="host-b"
    )
    assert claimed["machine_id"] == "host-b"
    assert claimed["claimed_reason"] == "host-a decommissioned 2026-08-24"
    again = assert_and_stamp_records_root_writer(root, machine_id="host-b")
    assert again["machine_id"] == "host-b"


def test_a_pre_guard_root_is_stamped_and_admitted(tmp_path: Path) -> None:
    """Era honesty: every existing root has no stamp; first write stamps it."""

    root = tmp_path / "records"
    root.mkdir()
    assert last_writer_stamp(root) is None
    stamp = assert_and_stamp_records_root_writer(root, machine_id="host-a")
    assert stamp["machine_id"] == "host-a"
    assert last_writer_stamp(root) == stamp


def test_a_corrupt_stamp_refuses_rather_than_guessing(tmp_path: Path) -> None:
    """A corrupt ownership stamp is a human question, never a default."""

    root = tmp_path / "records"
    root.mkdir()
    (root / ".records-root-writer.json").write_text("not json", encoding="utf-8")
    with pytest.raises(RecordsRootGuardError):
        assert_and_stamp_records_root_writer(root, machine_id="host-a")


def test_leg_open_runs_both_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Source binding + behavior: opening a leg refuses on an unsafe root."""

    source = inspect.getsource(leg_generation_module.open_leg_generation)
    assert "assert_records_root_flock_safe(" in source
    assert "assert_and_stamp_records_root_writer(" in source

    monkeypatch.setattr(guard, "resolve_records_fstype", lambda _path: "smbfs")
    with pytest.raises(RecordsRootGuardError):
        leg_generation_module.open_leg_generation(
            tmp_path / "records",
            campaign_id="c3-classics",
            intake_snapshot_id="intake-guard",
            intake_snapshot_sha256="sha256:" + "ab" * 32,
            run_id="pytest-guard",
            opened_at="2026-08-24T00:00:00Z",
        )


def test_the_stamp_is_a_plain_document_not_a_ledger(tmp_path: Path) -> None:
    """The stamp is one atomically replaced JSON object -- auditable at a glance."""

    root = tmp_path / "records"
    assert_and_stamp_records_root_writer(root, machine_id="host-a")
    payload = json.loads((root / ".records-root-writer.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "menagerie.crawler.records-root-writer.v1"
    assert payload["machine_id"] == "host-a"
