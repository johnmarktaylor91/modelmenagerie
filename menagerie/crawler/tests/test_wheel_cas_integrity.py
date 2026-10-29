"""EN-3 -- the wheel CAS byte-hashes at EVERY lookup and quarantines violations.

Reproduced 200a probe (sol, `wheel_cas_corruption_probe.txt`): `_wheel_cas_file`
returned the sole file in the digest directory WITHOUT hashing its bytes, so a
wrong-bytes wheel planted under a right digest was admitted
(``admitted_despite_digest_mismatch=True``) and the readiness receipt sealed green
over a corrupt store -- an integrity hole in a lane whose whole story is
byte-exactness. Every lookup now streams the candidate's bytes against the digest,
refuses symlinks and multi-file entries, and QUARANTINES the violating entry aside so
the store self-heals: the entry stops satisfying ``is_dir`` presence checks and the
next materialization re-downloads it from the pinned manifest.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from menagerie.crawler.operator_environment import (
    PermanentEnvironmentOperatorError,
    _wheel_cas_entry,
    _wheel_cas_file,
)


def _plant(wheel_cas: Path, content: bytes, *, digest: str | None = None) -> tuple[str, Path]:
    """Plant one wheel-CAS entry and return (digest, artifact path)."""

    digest = digest or hashlib.sha256(content).hexdigest()
    entry = wheel_cas / "sha256" / digest
    entry.mkdir(parents=True)
    artifact = entry / "pkg-1.0-py3-none-any.whl"
    artifact.write_bytes(content)
    return digest, artifact


@pytest.mark.smoke
def test_correct_bytes_pass_and_the_entry_is_untouched(tmp_path: Path) -> None:
    wheel_cas = tmp_path / "wheel-cas"
    digest, artifact = _plant(wheel_cas, b"good wheel bytes")
    selected = _wheel_cas_file(wheel_cas, digest)
    assert selected == artifact
    assert artifact.is_file()
    assert not (wheel_cas / "quarantine").exists()


@pytest.mark.smoke
def test_wrong_bytes_under_right_digest_are_refused_and_quarantined(tmp_path: Path) -> None:
    """The reproduced probe: a wrong-digest wheel must never be admitted."""

    wheel_cas = tmp_path / "wheel-cas"
    pinned = hashlib.sha256(b"the bytes the manifest pinned").hexdigest()
    _plant(wheel_cas, b"attacker or corruption bytes", digest=pinned)

    assert _wheel_cas_file(wheel_cas, pinned) is None
    # Quarantined aside: presence checks stop being satisfied, so the next
    # materialization re-downloads instead of wedging on the corrupt object.
    assert not _wheel_cas_entry(wheel_cas, pinned).exists()
    quarantined = list((wheel_cas / "quarantine").iterdir())
    assert len(quarantined) == 1
    assert pinned in quarantined[0].name
    assert "digest-mismatch" in quarantined[0].name


@pytest.mark.smoke
def test_multiple_files_in_one_entry_are_refused_and_quarantined(tmp_path: Path) -> None:
    wheel_cas = tmp_path / "wheel-cas"
    digest, artifact = _plant(wheel_cas, b"real bytes")
    (artifact.parent / "second-file.whl").write_bytes(b"real bytes")

    assert _wheel_cas_file(wheel_cas, digest) is None
    assert not _wheel_cas_entry(wheel_cas, digest).exists()
    assert any(
        "not-exactly-one-regular-file" in path.name
        for path in (wheel_cas / "quarantine").iterdir()
    )


@pytest.mark.smoke
def test_symlinked_artifact_is_refused_and_quarantined(tmp_path: Path) -> None:
    """A symlink can swap content after verification; only regular files count."""

    wheel_cas = tmp_path / "wheel-cas"
    content = b"linked bytes"
    digest = hashlib.sha256(content).hexdigest()
    outside = tmp_path / "outside.whl"
    outside.write_bytes(content)
    entry = wheel_cas / "sha256" / digest
    entry.mkdir(parents=True)
    os.symlink(outside, entry / "pkg-1.0-py3-none-any.whl")

    assert _wheel_cas_file(wheel_cas, digest) is None
    assert not entry.exists()
    assert any(
        "not-exactly-one-regular-file" in path.name
        for path in (wheel_cas / "quarantine").iterdir()
    )


@pytest.mark.smoke
def test_symlinked_entry_directory_is_refused_and_quarantined(tmp_path: Path) -> None:
    wheel_cas = tmp_path / "wheel-cas"
    content = b"entry-linked bytes"
    digest = hashlib.sha256(content).hexdigest()
    real_entry = tmp_path / "elsewhere" / digest
    real_entry.mkdir(parents=True)
    (real_entry / "pkg-1.0-py3-none-any.whl").write_bytes(content)
    (wheel_cas / "sha256").mkdir(parents=True)
    os.symlink(real_entry, wheel_cas / "sha256" / digest)

    assert _wheel_cas_file(wheel_cas, digest) is None
    assert not (wheel_cas / "sha256" / digest).exists()
    # The real directory behind the link is untouched; only the link moved.
    assert (real_entry / "pkg-1.0-py3-none-any.whl").is_file()


@pytest.mark.smoke
def test_missing_entry_is_still_a_plain_miss(tmp_path: Path) -> None:
    wheel_cas = tmp_path / "wheel-cas"
    digest = hashlib.sha256(b"never stored").hexdigest()
    assert _wheel_cas_file(wheel_cas, digest) is None
    assert not (wheel_cas / "quarantine").exists()


@pytest.mark.smoke
def test_unquarantinable_violation_raises_typed_never_admits(tmp_path: Path) -> None:
    """If the corrupt entry cannot be moved aside, the lookup fails closed, typed."""

    wheel_cas = tmp_path / "wheel-cas"
    pinned = hashlib.sha256(b"pinned bytes").hexdigest()
    _plant(wheel_cas, b"corrupt bytes", digest=pinned)
    # A FILE occupying the quarantine root makes the aside-rename impossible.
    (wheel_cas / "quarantine").write_bytes(b"")

    with pytest.raises(PermanentEnvironmentOperatorError):
        _wheel_cas_file(wheel_cas, pinned)
