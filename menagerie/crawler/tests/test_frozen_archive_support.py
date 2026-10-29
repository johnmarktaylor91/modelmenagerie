"""Contract tests for the durable frozen-archive resolution (A2).

These pin the resolution order and, more importantly, the fail-closed edges:
an explicitly configured archive that does not resolve is always a failure,
and an armed ``MENAGERIE_FROZEN_ARCHIVE_GATE`` turns absence into a failure so
a fleet run can never silently lose the real-bytes replay pins the way the
old set-by-nothing env-var skips did.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.tests import frozen_archive_support
from menagerie.crawler.tests.frozen_archive_support import (
    FROZEN_ARCHIVE_GATE_ENV,
    FROZEN_ARCHIVE_ROOT_ENV,
    FROZEN_ARCHIVES,
    require_frozen_archive,
)


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point every resolution source at controlled empty locations.

    Parameters
    ----------
    monkeypatch:
        Active pytest monkeypatch context.
    tmp_path:
        Isolated scratch root.
    """

    for spec in FROZEN_ARCHIVES.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(FROZEN_ARCHIVE_ROOT_ENV, raising=False)
    monkeypatch.delenv(FROZEN_ARCHIVE_GATE_ENV, raising=False)
    monkeypatch.setattr(frozen_archive_support, "DURABLE_ARCHIVE_ROOT", tmp_path / "no-archives")


def _materialize(root: Path, name: str) -> Path:
    """Create one intact fake archive for a registered spec.

    Parameters
    ----------
    root:
        Parent directory receiving the canonical archive directory name.
    name:
        Registered archive key.

    Returns
    -------
    Path
        The intact archive directory.
    """

    spec = FROZEN_ARCHIVES[name]
    archive = root / spec.directory_name
    sentinel = archive / spec.sentinel
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("{}", encoding="utf-8")
    return archive


@pytest.mark.smoke
def test_explicit_env_archive_that_does_not_resolve_fails_never_skips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A configured-but-missing archive is a loud failure, not a silent skip."""

    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv(FROZEN_ARCHIVES["rung9"].env_var, str(tmp_path / "missing-archive"))
    with pytest.raises(pytest.fail.Exception, match="unmet-frozen-archive-gate"):
        require_frozen_archive("rung9")


@pytest.mark.smoke
def test_torn_explicit_archive_missing_its_sentinel_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An archive directory without its sentinel member counts as torn."""

    _isolate(monkeypatch, tmp_path)
    torn = tmp_path / "torn-archive"
    torn.mkdir()
    monkeypatch.setenv(FROZEN_ARCHIVES["rung11"].env_var, str(torn))
    with pytest.raises(pytest.fail.Exception, match="unmet-frozen-archive-gate"):
        require_frozen_archive("rung11")


@pytest.mark.smoke
def test_absent_archive_skips_typed_without_the_gate_and_fails_with_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absence skips with the typed reason; the armed gate makes it fail closed."""

    _isolate(monkeypatch, tmp_path)
    with pytest.raises(pytest.skip.Exception, match="frozen-archive-unavailable"):
        require_frozen_archive("rung9")
    monkeypatch.setenv(FROZEN_ARCHIVE_GATE_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match="unmet-frozen-archive-gate"):
        require_frozen_archive("rung9")


@pytest.mark.smoke
def test_resolution_prefers_explicit_env_then_root_env_then_durable_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The three sources resolve in the documented priority order."""

    _isolate(monkeypatch, tmp_path)
    durable = _materialize(tmp_path / "durable", "rung9")
    monkeypatch.setattr(frozen_archive_support, "DURABLE_ARCHIVE_ROOT", tmp_path / "durable")
    assert require_frozen_archive("rung9") == durable

    from_root_env = _materialize(tmp_path / "root-env", "rung9")
    monkeypatch.setenv(FROZEN_ARCHIVE_ROOT_ENV, str(tmp_path / "root-env"))
    assert require_frozen_archive("rung9") == from_root_env

    explicit = _materialize(tmp_path / "explicit", "rung9")
    monkeypatch.setenv(FROZEN_ARCHIVES["rung9"].env_var, str(explicit))
    assert require_frozen_archive("rung9") == explicit
