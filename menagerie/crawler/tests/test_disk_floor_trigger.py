"""Disk-floor TRIGGER coverage: real measurements, every write destination.

T5-8 (mutation-verified): deleting the one real floor check left 85/85
lifecycle tests green, because every construction passed
``minimum_free_bytes=0`` and the valve suite injected ``DiskFloorError`` from
a fake lane. T3-T3 (3-lab): the floor statvfs'd env_root -- a symlink to the
931-GiB SSD -- while the artifact CAS, package cache, wheelhouse, and TMPDIR
all landed on the ~23-GiB internal disk, so a real ENOSPC never tripped the
floor and fell into the build-failed catch-all.

These tests pin the trigger itself: the floor measures EVERY write
destination, re-checks after the build input is validated (the post-solve
point of the committed-family flow), converts a mid-create ENOSPC into the
typed floor condition, and uses byte -- never block -- arithmetic.
"""

from __future__ import annotations

import errno
import shutil
from pathlib import Path
from typing import Any, Sequence

import pytest

from menagerie.crawler.disk_floor import (
    DiskFloorDeferralError,
    build_disk_floor_deferral_row,
    disk_floor_deferral_path,
    load_disk_floor_deferral_rows,
)
from menagerie.crawler.effort import EffortTracker, StageCap
from menagerie.crawler.env_lifecycle import (
    DiskFloorError,
    SequentialEnvironmentLifecycle,
    disk_free_bytes,
)
from menagerie.crawler.envs import EnvironmentIntent, load_environment_registry
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.tests.test_slice_c_envs_lifecycle import (
    FakeEnvironmentBackend,
    _copy_env_specs,
    _swap_in_fake_family,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeEnvironments,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke

_FLOOR = 512 * 1024**2


def _lifecycle(
    tmp_path: Path,
    backend: FakeEnvironmentBackend,
    *,
    disk_free: Any = None,
    minimum_free_bytes: int = _FLOOR,
) -> tuple[SequentialEnvironmentLifecycle, EnvironmentIntent, EffortTracker]:
    """Build one lifecycle over the fake core family with a real-ish floor."""

    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    tracker = EffortTracker({"environment": StageCap(attempts=4, seconds=50, bytes=1000)})
    keyword_arguments = {} if disk_free is None else {"disk_free": disk_free}
    lifecycle = SequentialEnvironmentLifecycle(
        backend,
        tracker,
        env_root=tmp_path / "active",
        minimum_free_bytes=minimum_free_bytes,
        **keyword_arguments,
    )
    return lifecycle, intent, tracker


def test_disk_free_bytes_matches_shutil_disk_usage(tmp_path: Path) -> None:
    """The floor measurement is bytes, never blocks.

    A blocks-vs-bytes bug in ``disk_free_bytes`` (``f_bavail`` alone, or
    ``f_bavail * f_bsize`` on a filesystem where the fragment size differs)
    mis-sets every configured threshold by roughly the 4096x block factor.
    ``shutil.disk_usage().free`` is the standard-library byte answer; any
    block-arithmetic bug lands orders of magnitude away from it.
    """

    measured = disk_free_bytes(tmp_path)
    reference = shutil.disk_usage(tmp_path).free
    assert measured > 0
    assert abs(measured - reference) < 64 * 1024**2


def test_floor_refuses_with_the_real_measurement_before_any_effort(
    tmp_path: Path,
) -> None:
    """The REAL statvfs trigger: an unmeetable floor refuses typed, spend-free.

    This is the mutation killer T5-8 demanded: no fake ``disk_free``, no
    ``minimum_free_bytes=0``. Deleting the floor check makes this test fail.
    """

    events: list[str] = []
    lifecycle, intent, tracker = _lifecycle(
        tmp_path, FakeEnvironmentBackend(events), minimum_free_bytes=2**62
    )
    with pytest.raises(DiskFloorError) as refused:
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert refused.value.floor_bytes == 2**62
    assert 0 < refused.value.free_bytes < 2**62
    assert refused.value.condition == "floor"
    # Refused BEFORE spending any effort or invoking any backend action.
    assert events == []
    assert tracker.usage("environment").attempts == 0


def test_floor_probes_every_write_destination(tmp_path: Path) -> None:
    """A starved backend state root refuses even when env_root has disk.

    T3-T3's live hole: the floor watched the SSD at env_root while the CAS,
    package cache, and wheelhouse filled the internal disk. A backend that
    enumerates its write destinations gets each of them floored.
    """

    state_root = tmp_path / "operator-state"
    state_root.mkdir()

    class DestinationBackend(FakeEnvironmentBackend):
        """Enumerate one dedicated backend write destination."""

        def write_destinations(self) -> Sequence[Path]:
            return (state_root,)

    def disk_free(path: Path) -> int:
        return _FLOOR // 2 if Path(path) == state_root else _FLOOR * 4

    events: list[str] = []
    lifecycle, intent, tracker = _lifecycle(
        tmp_path, DestinationBackend(events), disk_free=disk_free
    )
    with pytest.raises(DiskFloorError) as refused:
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert refused.value.env_root == str(state_root)
    assert refused.value.destination_role == "backend-state"
    assert refused.value.free_bytes == _FLOOR // 2
    assert events == []
    assert tracker.usage("environment").attempts == 0


def test_floor_probes_the_temporary_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TMPDIR is a write destination too: conda-lock and staging land there."""

    scratch = tmp_path / "scratch-tmp"
    scratch.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(scratch))

    def disk_free(path: Path) -> int:
        return _FLOOR // 4 if Path(path) == scratch else _FLOOR * 4

    events: list[str] = []
    lifecycle, intent, _tracker = _lifecycle(
        tmp_path, FakeEnvironmentBackend(events), disk_free=disk_free
    )
    with pytest.raises(DiskFloorError) as refused:
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert refused.value.env_root == str(scratch)
    assert refused.value.destination_role == "tmp"
    assert events == []


def test_floor_uses_the_explicit_lane_tmp_root(tmp_path: Path) -> None:
    """The lifecycle authority does not silently substitute ambient TMPDIR."""

    scratch = tmp_path / "explicit-tmp"
    scratch.mkdir()

    def disk_free(path: Path) -> int:
        return _FLOOR // 4 if Path(path) == scratch else _FLOOR * 4

    events: list[str] = []
    root = _copy_env_specs(tmp_path)
    _swap_in_fake_family(root, "core")
    intent = load_environment_registry(root).intents["core"]
    tracker = EffortTracker({"environment": StageCap(attempts=4, seconds=50, bytes=1000)})
    lifecycle = SequentialEnvironmentLifecycle(
        FakeEnvironmentBackend(events),
        tracker,
        env_root=tmp_path / "active",
        disk_free=disk_free,
        minimum_free_bytes=_FLOOR,
        tmp_root=scratch,
    )
    with pytest.raises(DiskFloorError) as refused:
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert lifecycle.tmp_root == scratch
    assert refused.value.env_root == str(scratch)
    assert refused.value.destination_role == "tmp"
    assert events == []


def test_floor_is_rechecked_after_the_build_input_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disk that collapses between admission and create still refuses typed.

    The committed-family flow's post-solve point is "build input validated,
    nothing written yet": the floor measures again there, so a filesystem
    that filled up in the window refuses BEFORE the backend writes anything.
    """

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "active"))
    measurements = iter((_FLOOR * 4, _FLOOR // 8))

    def collapsing_disk_free(_path: Path) -> int:
        return next(measurements, _FLOOR // 8)

    events: list[str] = []
    lifecycle, intent, _tracker = _lifecycle(
        tmp_path, FakeEnvironmentBackend(events), disk_free=collapsing_disk_free
    )
    with pytest.raises(DiskFloorError) as refused:
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert refused.value.condition == "floor"
    # The re-check fired before any backend write: create never ran.
    assert events == []


def test_enospc_during_create_is_typed_as_the_disk_floor_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-create ENOSPC becomes the typed floor refusal, never build-failed.

    T5-10's documented real conda failure: create dies mid-build with ENOSPC
    and a partial prefix. The refusal must be the machine condition (so the
    driver parks it) and the partial prefix must be torn down.
    """

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "active"))

    class EnospcBackend(FakeEnvironmentBackend):
        """Materialize a partial prefix, then die exactly like conda does."""

        def create(self, lock_file: Path, prefix: Path) -> bytes:
            del lock_file
            self.events.append("create")
            prefix.mkdir(parents=True, exist_ok=True)
            (prefix / "conda-meta.partial").write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "No space left on device")

        def remove(self, prefix: Path) -> None:
            self.events.append("remove")
            shutil.rmtree(prefix, ignore_errors=True)

    events: list[str] = []
    lifecycle, intent, _tracker = _lifecycle(
        tmp_path, EnospcBackend(events), disk_free=lambda _path: _FLOOR * 4
    )
    with pytest.raises(DiskFloorError) as refused:
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert refused.value.condition == "enospc"
    assert isinstance(refused.value.__cause__, OSError)
    assert refused.value.__cause__.errno == errno.ENOSPC
    # The partial prefix was torn down: mid-create death never leaves state
    # that a later run could mistake for "built".
    assert events == ["create", "remove"]
    prefix_root = tmp_path / "active"
    assert not any(prefix_root.glob("core-*"))


def test_a_recovered_disk_builds_normally_after_a_mid_create_death(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mid-create partial state is repairable: the next attempt succeeds."""

    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "active"))

    class OneShotEnospcBackend(FakeEnvironmentBackend):
        """Die with ENOSPC once, then create normally."""

        def __init__(self, events: list[str]) -> None:
            super().__init__(events)
            self.fail = True

        def create(self, lock_file: Path, prefix: Path) -> bytes:
            if self.fail:
                self.fail = False
                prefix.mkdir(parents=True, exist_ok=True)
                (prefix / "conda-meta.partial").write_bytes(b"partial")
                raise OSError(errno.ENOSPC, "No space left on device")
            return super().create(lock_file, prefix)

        def remove(self, prefix: Path) -> None:
            super().remove(prefix)
            shutil.rmtree(prefix, ignore_errors=True)

    events: list[str] = []
    lifecycle, intent, _tracker = _lifecycle(
        tmp_path, OneShotEnospcBackend(events), disk_free=lambda _path: _FLOOR * 4
    )
    with pytest.raises(DiskFloorError):
        lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    result = lifecycle.run(intent, use=lambda _prefix, _probes: events.append("use"))
    assert result.disk_recovery_checked
    assert events[-3:] == ["probe", "use", "remove"]


class _EnospcParkEnvironments(FakeEnvironments):
    """Surface the lifecycle's typed ENOSPC floor condition to the driver."""

    def run(self, intent: EnvironmentIntent, *, use: Any) -> object:
        """Raise the mid-create disk-exhaustion refusal, pre-use."""

        del use
        raise DiskFloorError(
            "environment write failed with ENOSPC during create",
            floor_bytes=_FLOOR,
            free_bytes=_FLOOR * 2,
            env_root=str(self.root),
            condition="enospc",
            destination_role="backend-state",
        )


def test_disk_exhaustion_during_create_defers_and_never_terminalizes(
    tmp_path: Path,
) -> None:
    """Driver-level: the ENOSPC floor condition parks; zero model terminals."""

    snapshot = _snapshot(tmp_path, count=2)
    paths = _paths(tmp_path, snapshot)
    result = _driver(
        tmp_path,
        snapshot,
        environments=_EnospcParkEnvironments(tmp_path / "fake-envs"),
    ).run()
    assert result.status == "complete-with-deferrals"
    assert scan_jsonl(paths.ledgers.models) == []
    records_root = paths.ledgers.models.parent.parent
    rows = load_disk_floor_deferral_rows([disk_floor_deferral_path(records_root)])
    assert {row["stable_id"] for row in rows} == {item.stable_id for item in snapshot.items}
    for row in rows:
        assert row["disk"]["condition"] == "enospc"
        # An ENOSPC refusal is real evidence even when the measurement sits
        # above the configured floor (quota, another writer freeing space).
        assert row["disk"]["free_bytes"] >= row["disk"]["floor_bytes"]


def test_disk_floor_row_condition_vocabulary_is_closed() -> None:
    """Only the two refusal conditions exist; floor rows still demand a refusal."""

    common = {
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
    }
    # An enospc row may record a measurement at or above the floor: the failed
    # write itself is the refusal evidence.
    row = build_disk_floor_deferral_row(**common, free_bytes=_FLOOR * 2, condition="enospc")
    assert row["disk"]["condition"] == "enospc"
    # A floor row still requires an actual refusing measurement.
    with pytest.raises(DiskFloorDeferralError, match="refus"):
        build_disk_floor_deferral_row(**common, free_bytes=_FLOOR, condition="floor")
    with pytest.raises(DiskFloorDeferralError, match="condition"):
        build_disk_floor_deferral_row(**common, free_bytes=1, condition="quota-exceeded")
