"""Durable frozen-archive resolution for real-bytes replay tests (A2).

The rung-9 and rung-11 frozen-archive replays are among the strongest pins in
the corpus -- they re-run the actual campaign bytes through the shipped fixes.
They used to ``skipif`` on per-test environment variables nothing set, so they
silently never ran anywhere. This module gives them one resolution order and a
fail-closed mode, in the release-attestation style:

1. An explicit per-archive environment variable (``MENAGERIE_RUNG9_ARCHIVE``,
   ``MENAGERIE_RUNG11_ARCHIVE``) wins. An explicitly configured archive that
   does not resolve is a FAILURE, never a skip: pointing a run at a missing or
   torn archive must be loud.
2. ``MENAGERIE_RUNG_ARCHIVE_ROOT`` names a root holding the canonical archive
   directory names.
3. The durable operator archive root where the sprint keeps the frozen rungs
   (present on the campaign fleet; absent on CI and contributor machines).

When no source resolves, the test skips with a typed reason -- unless
``MENAGERIE_FROZEN_ARCHIVE_GATE=1``, which makes absence fail closed so a
fleet or mid-tier run whose host MUST carry the archives cannot silently lose
these pins the way the old env-var skips did.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import pytest

from menagerie.local_config import local_value

FROZEN_ARCHIVE_GATE_ENV = "MENAGERIE_FROZEN_ARCHIVE_GATE"
FROZEN_ARCHIVE_ROOT_ENV = "MENAGERIE_RUNG_ARCHIVE_ROOT"

#: Durable operator archive root (campaign fleet). CI and contributor machines
#: without it skip typed; hosts that must have it set the gate env variable.
DURABLE_ARCHIVE_ROOT = Path(
    local_value("DURABLE_RUNG_ARCHIVE", "~/.local/share/menagerie/rung-archive")
).expanduser()


@dataclass(frozen=True)
class FrozenArchiveSpec:
    """One frozen campaign archive a replay test consumes."""

    #: Per-archive override environment variable.
    env_var: str
    #: Canonical directory name beneath an archive root.
    directory_name: str
    #: Relative path that must exist for the archive to count as intact.
    sentinel: str


FROZEN_ARCHIVES: dict[str, FrozenArchiveSpec] = {
    "rung9": FrozenArchiveSpec(
        env_var="MENAGERIE_RUNG9_ARCHIVE",
        directory_name="rung9a-8fa88386-recording-halt",
        sentinel="work/checker/metadata-143c9aa7dfcc0bb1/request.json",
    ),
    "rung11": FrozenArchiveSpec(
        env_var="MENAGERIE_RUNG11_ARCHIVE",
        directory_name="rung11-c15c2753-3runs-reducer-crash",
        sentinel="records/attempts/local.jsonl",
    ),
}


def _intact(root: Path, spec: FrozenArchiveSpec) -> bool:
    """Return whether one candidate archive root is present and not torn.

    Parameters
    ----------
    root:
        Candidate archive directory.
    spec:
        Archive contract naming the sentinel member.

    Returns
    -------
    bool
        True when the directory and its sentinel member both exist.
    """

    return root.is_dir() and (root / spec.sentinel).is_file()


def require_frozen_archive(name: str) -> Path:
    """Return one intact frozen archive root, or skip/fail per the contract.

    Parameters
    ----------
    name:
        Registered archive key (``rung9`` or ``rung11``).

    Returns
    -------
    Path
        Intact archive root.

    Raises
    ------
    pytest.Failed
        If an explicitly configured archive does not resolve, or no archive
        resolves while ``MENAGERIE_FROZEN_ARCHIVE_GATE=1``.
    pytest.Skipped
        If no archive source resolves and the gate is not armed.
    """

    spec = FROZEN_ARCHIVES[name]
    explicit = os.environ.get(spec.env_var, "")
    if explicit:
        root = Path(explicit)
        if not _intact(root, spec):
            pytest.fail(
                f"unmet-frozen-archive-gate: {spec.env_var}={explicit!r} does not"
                f" resolve to an intact {name} archive (missing {spec.sentinel})"
            )
        return root
    for base in (os.environ.get(FROZEN_ARCHIVE_ROOT_ENV, ""), DURABLE_ARCHIVE_ROOT):
        if not base:
            continue
        root = Path(base) / spec.directory_name
        if _intact(root, spec):
            return root
    if os.environ.get(FROZEN_ARCHIVE_GATE_ENV) == "1":
        pytest.fail(
            f"unmet-frozen-archive-gate: frozen {name} archive"
            f" {spec.directory_name!r} is required on this host and was not found"
        )
    pytest.skip(
        f"frozen-archive-unavailable: {name} archive {spec.directory_name!r} is not"
        f" on this host (set {spec.env_var} or {FROZEN_ARCHIVE_ROOT_ENV};"
        f" fleet hosts arm {FROZEN_ARCHIVE_GATE_ENV}=1 to fail closed)"
    )
    raise AssertionError("unreachable")  # pragma: no cover -- skip() raises
