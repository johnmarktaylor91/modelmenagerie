"""Default working locations for Menagerie outputs.

Rendered galleries, validation runs and smoke runs are regenerable, so they default to
``~/scratch/menagerie/<name>`` on the machine doing the work. Set ``MENAGERIE_WORK_ROOT``
to put them on a large volume instead; every CLI also takes an explicit ``--out-dir``.
"""

from __future__ import annotations

import os
from pathlib import Path

WORK_ROOT_ENV = "MENAGERIE_WORK_ROOT"


def work_root() -> Path:
    """Return the root directory for regenerable Menagerie outputs."""
    configured = os.environ.get(WORK_ROOT_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "scratch" / "menagerie"


def work_dir(name: str) -> Path:
    """Return the default output directory ``name`` under :func:`work_root`."""
    return work_root() / name
