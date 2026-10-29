"""The overlays' own uv cache, and what an overlay costs on disk.

Overlay extras install from ``<envs root>/overlay-uv-cache``, never from the shared ``~/.cache/uv``:
uv hardlinks each installed file to its cache copy, so with a shared cache an evicted overlay freed
nothing (2026-10-22). Builds hold a shared lock on the cache while uv installs; ``trim`` takes it
exclusively and drops, with uv's own ``cache clean`` and ``cache prune``, every package that has a
wheel archive no overlay links any more. uv's entries are symlinks into ``archive-v*/``: removing an
archive alone leaves a dangling entry, which uv refetches for an index wheel but fails on for a
direct-URL wheel (``failed to read directory``, 2026-10-25), so archives are never removed by hand.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import subprocess
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

CACHE_DIR = "overlay-uv-cache"
LOCK_FILE = "overlay-uv-cache.lock"
LOCK_WAIT_S = 120.0  # trim gives up (and says so) when builds hold the cache this long
UV_TIMEOUT_S = 600
_PACKAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def cache_dir(root: Path) -> Path:
    return Path(root) / CACHE_DIR


@contextlib.contextmanager
def locked(root: Path, *, exclusive: bool = False, wait_s: float | None = None) -> Iterator[bool]:
    """Hold the cache lock (shared for a build, exclusive for a trim); yield False when ``wait_s``
    passed without getting it."""

    Path(root).mkdir(parents=True, exist_ok=True)
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    with open(Path(root) / LOCK_FILE, "a") as handle:
        deadline = None if wait_s is None else time.monotonic() + wait_s
        while True:
            try:
                fcntl.flock(handle, mode | (0 if deadline is None else fcntl.LOCK_NB))
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    yield False
                    return
                time.sleep(1.0)
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def trim(root: Path, *, env: dict[str, str] | None = None) -> dict[str, int]:
    """Drop every package with a wheel archive no overlay links (and every package whose entry
    already points at a missing archive) through ``uv cache clean``, then ``uv cache prune``.

    ``busy`` is 1 when builds held the cache past ``LOCK_WAIT_S`` and nothing was looked at;
    ``uv_failed`` is 1 when a uv command failed, which leaves the cache as uv left it.
    """

    report = {"archives_removed": 0, "freed_bytes": 0, "packages_cleaned": 0, "busy": 0,
              "uv_failed": 0}
    cache = cache_dir(root)
    if not cache.is_dir():
        return report
    with locked(root, exclusive=True, wait_s=LOCK_WAIT_S) as held:
        if not held:
            report["busy"] = 1
            return report
        unused = {a: size for a in sorted(cache.glob("archive-v*/*"))
                  if (size := _unlinked_bytes(a)) is not None}
        packages = _packages(cache, {a.name for a in unused})
        for command in ([["clean", *sorted(packages)]] if packages else []) + [["prune"]]:
            done = subprocess.run(["uv", "cache", *command, "--cache-dir", str(cache)], check=False,
                                  capture_output=True, text=True, env=env, timeout=UV_TIMEOUT_S)
            if done.returncode != 0:
                report["uv_failed"] = 1
                break
        gone = [size for archive, size in unused.items() if not archive.exists()]
        report.update(archives_removed=len(gone), freed_bytes=sum(gone),
                      packages_cleaned=len(packages))
    return report


def _packages(cache: Path, unused: set[str]) -> set[str]:
    """Names of the packages whose uv entries link to an unused archive or to none at all."""

    names = set()
    for bucket in cache.glob("wheels-v*"):
        for top, dirs, files in os.walk(bucket):
            for name in dirs + files:
                entry = Path(top) / name
                if not entry.is_symlink():
                    continue
                target = Path(os.readlink(entry))
                dangling = not (entry.parent / target).exists()
                if (dangling or target.name in unused) and _PACKAGE.fullmatch(entry.parent.name):
                    names.add(entry.parent.name)
    return names


def _unlinked_bytes(archive: Path) -> int | None:
    """The archive's allocated bytes when no file of it has another link, else None."""

    if archive.is_symlink() or not archive.is_dir():
        return None
    total = 0
    for top, _dirs, files in os.walk(archive):
        for name in files:
            info = os.lstat(os.path.join(top, name))
            if info.st_nlink > 1:
                return None
            total += info.st_blocks * 512
    return total


def cost(overlay: Path, bases: dict[Path, set[tuple[int, int]]] | None = None) -> int:
    """Bytes the overlay holds beyond its base prefix: its extras, whether or not the overlay
    cache also links them (removing the overlay lets ``trim`` free those).

    The base is the built prefix the overlay was copied from (``cp -al``), named by the overlay
    directory without its final ``-<key>``. With the base gone every file counts once. ``bases``
    keeps each base's inodes across the overlays of one prune.
    """

    bases = {} if bases is None else bases
    base = _base_prefix(overlay)
    if base and base not in bases:
        bases[base] = _inodes(base)
    shared = bases[base] if base else set()
    seen, total = set(), 0
    for top, _dirs, files in os.walk(overlay):
        for name in files:
            try:
                info = os.lstat(os.path.join(top, name))
            except FileNotFoundError:  # removed while counting
                continue
            key = (info.st_dev, info.st_ino)
            if key not in shared and key not in seen:
                seen.add(key)
                total += info.st_blocks * 512
    return total


def _base_prefix(overlay: Path) -> Path | None:
    workspace = overlay.parent.parent / overlay.name.rsplit("-", 1)[0]
    if "-" not in overlay.name or not workspace.is_dir():
        return None
    pixi = workspace / ".pixi" / "envs" / "default"
    return pixi if pixi.is_dir() else workspace


def _inodes(base: Path) -> set[tuple[int, int]]:
    found = set()
    for top, _dirs, files in os.walk(base):
        for name in files:
            with contextlib.suppress(FileNotFoundError):
                info = os.lstat(os.path.join(top, name))
                found.add((info.st_dev, info.st_ino))
    return found


def needed_markers(requests: Iterable[Path], lock_id) -> set[str]:
    """Built-marker texts of the overlays that queued or running spool requests will use.

    ``lock_id`` maps an environment name to its ``<name>@<lock8>``; a request that cannot be read
    is skipped (it names no overlay this prune can match).
    """

    needed = set()
    for path in requests:
        try:
            data = json.loads(Path(path).read_text())
            recipe = data["recipe"]
            extras = sorted({*recipe.get("extras", ()), *data.get("extras", ())})
            if extras:
                needed.add(f"{lock_id(recipe['env'])} + {' '.join(extras)}")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return needed
