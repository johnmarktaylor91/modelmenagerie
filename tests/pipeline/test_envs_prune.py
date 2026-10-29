"""Overlay eviction: age, size cap, recent use, half-built leftovers, and reuse marking last use."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from menagerie.pipeline import envs

NOW = 1_800_000_000.0
DAY = 86400.0


def _overlay(root: Path, name: str, *, used: float, size: int = 1000, built: bool = True) -> Path:
    path = root / "overlays" / name
    (path / "lib").mkdir(parents=True)
    (path / "lib" / "extra.py").write_bytes(b"x" * size)
    if built:
        marker = path / envs.BUILT_MARKER
        marker.write_text("torch-main + extra\n")
        os.utime(marker, (used, used))
    os.utime(path, (used, used))
    return path


def test_age_removes_overlays_unused_for_keep_days(tmp_path: Path) -> None:
    old = _overlay(tmp_path, "old", used=NOW - 4 * DAY)
    fresh = _overlay(tmp_path, "fresh", used=NOW - 1 * DAY)
    report = envs.prune_overlays(tmp_path, keep_days=3, cap_gib=100, now=NOW)
    assert not old.exists() and fresh.exists()
    assert report["removed_age"] == 1 and report["kept"] == 1 and report["freed_bytes"] > 0


def test_cap_removes_least_recently_used_first(tmp_path: Path) -> None:
    mib = 2**20
    oldest = _overlay(tmp_path, "a", used=NOW - 2 * DAY, size=mib)
    middle = _overlay(tmp_path, "b", used=NOW - 1 * DAY, size=mib)
    newest = _overlay(tmp_path, "c", used=NOW - 0.5 * DAY, size=mib)
    report = envs.prune_overlays(tmp_path, keep_days=3, cap_gib=2.5 * mib / 2**30, now=NOW)
    assert not oldest.exists() and middle.exists() and newest.exists()
    assert report["removed_cap"] == 1 and report["kept_bytes"] <= 2.5 * mib


def test_recent_use_is_never_removed(tmp_path: Path) -> None:
    busy = _overlay(tmp_path, "busy", used=NOW - 60, size=2**20)
    report = envs.prune_overlays(tmp_path, keep_days=0, cap_gib=0, now=NOW)
    assert busy.exists() and report["kept"] == 1


def test_the_in_use_window_outlasts_a_build_and_no_more(tmp_path: Path) -> None:
    assert envs.INSTALL_TIMEOUT_S < envs.OVERLAY_IN_USE_S <= 3600
    idle = _overlay(tmp_path, "idle", used=NOW - 90 * 60, size=2**20)
    recent = _overlay(tmp_path, "recent", used=NOW - 40 * 60, size=2**20)
    building = _overlay(tmp_path, "building", used=NOW - 40 * 60, built=False)
    envs.prune_overlays(tmp_path, keep_days=3, cap_gib=0, now=NOW)
    assert not idle.exists() and recent.exists() and building.exists()


def test_half_built_leftovers_go_but_a_build_in_progress_stays(tmp_path: Path) -> None:
    stale = _overlay(tmp_path, "stale", used=NOW - 3 * 3600, built=False)
    building = _overlay(tmp_path, "building", used=NOW - 60, built=False)
    report = envs.prune_overlays(tmp_path, keep_days=3, cap_gib=100, now=NOW)
    assert not stale.exists() and building.exists()
    assert report["removed_halfbuilt"] == 1
    assert not (building / envs.OVERLAY_COST).exists()


def _cached_extra(root: Path, overlay: Path, archive: str, size: int) -> Path:
    """An extra installed the way uv does it: the overlay's file is a hardlink of a cache copy,
    and uv's entry for the package (named like the archive here) is a symlink to the archive."""

    cached = root / "overlay-uv-cache" / "archive-v0" / archive / "pkg" / "mod.so"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"w" * size)
    os.link(cached, overlay / "lib" / f"{archive}.so")
    entry = root / "overlay-uv-cache" / "wheels-v7" / "pypi" / archive / "1.0-py3-none-any"
    entry.parent.mkdir(parents=True)
    entry.symlink_to(Path("../../../archive-v0") / archive)
    return cached


@pytest.fixture
def fake_uv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """uv's ``cache clean PKG..`` (drops the packages' entries) and ``cache prune`` (drops archives
    no entry links to), recorded."""

    import shutil

    from menagerie.pipeline import overlay_cache

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        cache = Path(command[command.index("--cache-dir") + 1])
        if command[2] == "clean":
            for name in command[3:command.index("--cache-dir")]:
                shutil.rmtree(cache / "wheels-v7" / "pypi" / name, ignore_errors=True)
        else:
            linked = {os.readlink(e).rsplit("/", 1)[-1] for e in cache.glob("wheels-v7/*/*/*") if e.is_symlink()}
            for archive in cache.glob("archive-v0/*"):
                if archive.name not in linked:
                    shutil.rmtree(archive)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(overlay_cache.subprocess, "run", run)
    return calls


def test_size_counts_extras_but_not_files_shared_with_the_base(tmp_path: Path) -> None:
    base = tmp_path / "torch-main-abc"
    base.mkdir()
    (base / "torch.so").write_bytes(b"y" * 2**20)
    path = _overlay(tmp_path, "torch-main-abc-1234abcd", used=NOW - DAY, size=10)
    os.link(base / "torch.so", path / "lib" / "torch.so")
    _cached_extra(tmp_path, path, "a1", 2**19)  # linked from the overlay cache: still the overlay's
    size = envs._overlay_bytes(path)
    assert 2**19 <= size < 2**20
    assert (path / envs.OVERLAY_COST).read_text() == str(size)


def test_cap_eviction_then_trim_frees_the_evicted_overlays_cached_wheels(tmp_path: Path,
                                                                        fake_uv: list[list[str]]) -> None:
    (tmp_path / "torch-main-abc").mkdir()
    old = _overlay(tmp_path, "torch-main-abc-0000000a", used=NOW - 2 * DAY, size=10)
    new = _overlay(tmp_path, "torch-main-abc-0000000b", used=NOW - 0.5 * DAY, size=10)
    gone = _cached_extra(tmp_path, old, "old", 2**20)
    stays = _cached_extra(tmp_path, new, "new", 2**20)
    report = envs.prune_overlays(tmp_path, keep_days=3, cap_gib=1.5 * 2**20 / 2**30, now=NOW)
    assert not old.exists() and new.exists()
    assert not gone.parent.parent.exists() and stays.is_file()
    assert report["cache_archives_removed"] == 1 and report["freed_bytes"] >= 2**20
    # through uv's own commands, so no entry is left pointing at a removed archive
    assert [c[2:4] for c in fake_uv] == [["clean", "old"], ["prune", "--cache-dir"]]


def test_trim_cleans_a_package_whose_entry_already_dangles(tmp_path: Path, fake_uv: list[list[str]]) -> None:
    from menagerie.pipeline import overlay_cache

    overlay = _overlay(tmp_path, "torch-main-abc-0000000f", used=NOW, size=10)
    archive = _cached_extra(tmp_path, overlay, "torchaudio", 10).parent.parent
    shutil.rmtree(archive)  # an archive removed by hand, as trim did before 2026-10-26
    report = overlay_cache.trim(tmp_path)
    assert fake_uv[0][2:4] == ["clean", "torchaudio"] and report["packages_cleaned"] == 1
    assert not list((tmp_path / "overlay-uv-cache").glob("wheels-v7/pypi/*"))


def test_trim_reports_a_failed_uv_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import overlay_cache

    overlay = _overlay(tmp_path, "torch-main-abc-00000010", used=NOW, size=10)
    archive = _cached_extra(tmp_path, overlay, "x", 10).parent.parent
    (overlay / "lib" / "x.so").unlink()
    monkeypatch.setattr(overlay_cache.subprocess, "run",
                        lambda command, **k: subprocess.CompletedProcess(command, 2, "", "boom"))
    report = overlay_cache.trim(tmp_path)
    assert report["uv_failed"] == 1 and report["archives_removed"] == 0 and archive.is_dir()


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_trim_with_real_uv_leaves_a_cache_that_installs_again(tmp_path: Path) -> None:
    """A local wheel installed through the overlay cache, its venv gone, trimmed, then installed
    again from the same cache: uv must not fail on an entry whose archive the trim removed."""

    import sys
    import zipfile

    from menagerie.pipeline import overlay_cache

    wheel = tmp_path / "tinypkg-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("tinypkg/__init__.py", "VALUE = 1\n")
        zf.writestr("tinypkg-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: tinypkg\nVersion: 1.0\n")
        zf.writestr("tinypkg-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        zf.writestr("tinypkg-1.0.dist-info/RECORD", "")
    cache = overlay_cache.cache_dir(tmp_path)

    def install(venv: Path) -> subprocess.CompletedProcess:
        subprocess.run(["uv", "venv", "-q", "--python", sys.executable, str(venv)], check=True)
        return subprocess.run(["uv", "pip", "install", "--offline", "--cache-dir", str(cache),
                               "--python", str(venv / "bin" / "python"), str(wheel)],
                              capture_output=True, text=True)

    assert install(tmp_path / "v1").returncode == 0
    archives = list(cache.glob("archive-v*/*"))
    assert archives
    shutil.rmtree(tmp_path / "v1")
    report = overlay_cache.trim(tmp_path)
    assert report["uv_failed"] == 0 and report["archives_removed"] == len(archives)
    again = install(tmp_path / "v2")
    assert again.returncode == 0, again.stderr


def test_trim_waits_for_a_build_and_reports_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 fake_uv: list[list[str]]) -> None:
    from menagerie.pipeline import overlay_cache

    overlay = _overlay(tmp_path, "torch-main-abc-0000000c", used=NOW, size=10)
    archive = _cached_extra(tmp_path, overlay, "x", 10).parent.parent
    (overlay / "lib" / "x.so").unlink()  # nothing links it any more
    monkeypatch.setattr(overlay_cache, "LOCK_WAIT_S", 0.0)
    with overlay_cache.locked(tmp_path):  # a build installing from the cache (flock is per open file)
        assert overlay_cache.trim(tmp_path)["busy"] == 1
    assert archive.is_dir()
    assert overlay_cache.trim(tmp_path)["archives_removed"] == 1 and not archive.exists()


def test_an_overlay_a_spool_request_names_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    monkeypatch.setattr(envs, "lock_id", lambda name: f"{name}@abc")
    needed = _overlay(tmp_path, "torch-main-abc-0000000d", used=NOW - 9 * DAY)
    (needed / envs.BUILT_MARKER).write_text("torch-main@abc + a b\n")
    os.utime(needed / envs.BUILT_MARKER, (NOW - 9 * DAY, NOW - 9 * DAY))
    other = _overlay(tmp_path, "torch-main-abc-0000000e", used=NOW - 9 * DAY)
    request = tmp_path / "spool" / "claimed" / "r.json"
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps({"recipe": {"env": "torch-main", "extras": ["b"]}, "extras": ["a"]}))
    report = envs.prune_overlays(tmp_path, keep_days=3, cap_gib=0, now=NOW, keep_for=[request])
    assert needed.exists() and not other.exists()
    assert report["kept_needed"] == 1


def test_missing_overlays_dir_is_empty_report(tmp_path: Path) -> None:
    assert envs.prune_overlays(tmp_path, keep_days=3, cap_gib=8, now=NOW)["overlays"] == 0


def test_reuse_touches_the_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    from menagerie.pipeline import sdist

    monkeypatch.setattr(envs, "python", lambda *a, **k: tmp_path / "base" / "bin" / "python")
    monkeypatch.setattr(sdist, "wheels", lambda *a, **k: [])
    monkeypatch.setattr(envs.subprocess, "run",
                        lambda command, **k: subprocess.CompletedProcess(command, 0, "torch==2.6.0\n", ""))
    monkeypatch.setattr(envs, "lock_id", lambda name: "torch-main@abc")
    key = hashlib.sha256(b"extra").hexdigest()[:8]
    built = _overlay(tmp_path, f"torch-main-abc-{key}", used=time.time() - 5 * DAY)
    assert envs.overlay("torch-main", ["extra"], root=tmp_path) == built / "bin" / "python"
    assert (built / envs.BUILT_MARKER).stat().st_mtime > time.time() - 60
