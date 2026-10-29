"""The in-attempt disk watchdog: an attempt's output cap and the worker's free-space floor.

The disk guard acts between attempts, so one attempt could still fill the worker's disk: m1618
(18 Oct 2026) wrote 26 GB of saves in 14 minutes. ``runner._supervise`` calls ``breach`` every
``POLL_S`` while the sandbox runs and kills the attempt when it returns a reason:

- ``output_cap``: the attempt directory is over ``MENAGERIE_ATTEMPT_MAX_GIB`` (default 8; 0 turns
  the cap off). The runner types it ``oversized:output``, a fail the model keeps, so flow takes a
  structure-only capture instead and never goes on to the train trace.
- ``disk_low``: free space is under the fleet floor plus ``MARGIN_GIB``, an ``infra`` result. Only
  in a runner job, which sets ``MENAGERIE_DISK_PAUSE_GIB``; ``runner serve`` also claims nothing
  below that pause line, so the requeued row waits for space instead of looping.

``shed`` then deletes the killed attempt's saves and keeps its small files (log, request, result).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from menagerie.pipeline import disk_guard

MAX_ENV = "MENAGERIE_ATTEMPT_MAX_GIB"
# Past passes (13,202 saves in the store index, 18 Oct 2026): median 3.4 MiB, max 1.9 GiB packed,
# none over 2 GiB; an attempt writes two saves (runnable and portable), so 8 GiB loses none.
DEFAULT_MAX_GIB = 8.0
POLL_S = 10.0
MARGIN_GIB = 4.0
KEEP_BYTES = 16 * 2**20
OUTPUT_CAP = "output_cap"
DISK_LOW = "disk_low"


def max_bytes() -> int | None:
    """The attempt directory's cap in bytes, or ``None`` when the cap is off."""

    raw = os.environ.get(MAX_ENV)
    gib = float(raw) if raw else DEFAULT_MAX_GIB
    return int(gib * disk_guard.GIB) if gib > 0 else None


def tree_bytes(root: Path) -> int:
    """Bytes the files under ``root`` occupy on disk (allocated blocks, so sparse files count
    what they use)."""

    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
            except OSError:  # removed while walking (a save's staging renamed into place)
                continue
    return total


def breach(attempt_dir: Path, *, statvfs: Callable[[str], Any] = os.statvfs) -> str | None:
    """Why a running attempt must stop now (``output_cap``, ``disk_low``), else ``None``."""

    cap = max_bytes()
    if cap is not None and tree_bytes(attempt_dir) > cap:
        return OUTPUT_CAP
    if os.environ.get(disk_guard.PAUSE_ENV):
        free, floor = disk_guard.free_and_floor(attempt_dir, statvfs=statvfs)
        if free < floor + MARGIN_GIB:
            return DISK_LOW
    return None


def shed(attempt_dir: Path) -> None:
    """Delete a killed attempt's directories (its saves and their staging) and large files."""

    for child in attempt_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        elif child.is_file() and child.stat().st_size > KEEP_BYTES:
            child.unlink(missing_ok=True)
