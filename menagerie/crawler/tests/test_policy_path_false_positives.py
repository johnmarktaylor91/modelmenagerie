"""B1 (step-100 triage): the checkpoint-read classifier and benign system/package files.

Three false-positive families killed honest models on the 2026-08-12 step-100 leg:

* m4066: ``/usr/share/icu/icudt76l.dat`` -- host ICU locale data flagged by the
  large-file weight heuristic;
* m7960: ``site-packages/astor/VERSION`` -- extensionless package metadata; the landed
  installed-files.txt fix covered one filename, not the category;
* m10759: ``/etc/os-release`` -- host OS metadata.

The fix allows by PREFIX + CATEGORY (host system data trees, closed package-metadata
basenames under site-packages, any bounded-text non-weight-suffixed file inside a
``.dist-info``/``.egg-info`` directory) and requires the extensionless weight heuristic
to see real bulk or a tensor-format magic before firing.

LOCKED validation principle, proven in BOTH directions below: the m1115-style refusal of
real undeclared model-data reads stays armed -- weight-suffixed names anywhere, tensor
magic or >= 1 MiB bulk in extensionless files, and model-data payloads planted inside a
metadata directory are all still refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.policy import (
    ExecutionPolicy,
    _runtime_import_metadata_path_allowed,
    _runtime_model_data_path,
    _runtime_static_path_allowed,
    generate_macos_sandbox_profile,
)
from menagerie.crawler.worker_supervisor import _read_path_is_allowed


def _parent_allows(path: str, tmp_path: Path, runtime_roots: tuple[Path, ...] = ()) -> bool:
    """Run the parent-side kernel-read classifier over one path."""

    return _read_path_is_allowed(path, tmp_path, (), (), runtime_roots)


def _worker_allows(path: str, tmp_path: Path) -> bool:
    """Run the in-process worker classifier over one path without arming tripwires."""

    return ExecutionPolicy(tmp_path / "scratch")._read_path_allowed(path)


# ---------------------------------------------------------------------------
# B1(a): host system data trees are never checkpoint reads
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "path",
    [
        "/usr/share/icu/icudt76l.dat",  # m4066: large ICU payload, benign
        "/etc/os-release",  # m10759: OS metadata
        "/usr/share/misc/magic.mgc",
    ],
)
def test_system_data_reads_are_never_checkpoint_reads(tmp_path: Path, path: str) -> None:
    """Both classifiers accept reads beneath /usr/share and /etc by prefix."""

    assert _parent_allows(path, tmp_path) is True
    assert _worker_allows(path, tmp_path) is True


@pytest.mark.smoke
def test_dot_dot_escape_cannot_launder_through_system_prefixes(tmp_path: Path) -> None:
    """A lexical /usr/share/../.. spelling never grants the resolved target."""

    hidden = (tmp_path / "hidden.pt").resolve()
    escape = "/usr/share/../.." + str(hidden)
    assert _parent_allows(escape, tmp_path) is False
    assert _worker_allows(escape, tmp_path) is False


@pytest.mark.smoke
def test_macos_profile_grants_the_same_system_data_authority(tmp_path: Path) -> None:
    """The Seatbelt layer may not stay stricter than the classifiers it backs."""

    profile = generate_macos_sandbox_profile([tmp_path])
    assert '(allow file-read-data (subpath "/usr/share"))' in profile
    assert '(allow file-read-data (subpath "/private/etc"))' in profile


# ---------------------------------------------------------------------------
# B1(b): the package-metadata CATEGORY, not one filename
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_site_packages_version_family_is_package_metadata(tmp_path: Path) -> None:
    """m7960: an installed package's small textual VERSION file is metadata, not weights."""

    site_packages = tmp_path / "env" / "lib" / "python3.11" / "site-packages"
    version_file = site_packages / "astor" / "VERSION"
    version_file.parent.mkdir(parents=True)
    version_file.write_text("0.8.1\n", encoding="utf-8")

    assert _runtime_model_data_path(version_file) is False
    assert _runtime_static_path_allowed(version_file) is True
    assert _parent_allows(str(version_file), tmp_path, (tmp_path / "env",)) is True

    # Direction 2: the family is no smuggling channel. The same basename with
    # tensor-format magic, and dist-info-confined names relocated into the
    # package directory, both stay refused.
    smuggled_version = site_packages / "pkg" / "VERSION"
    smuggled_version.parent.mkdir(parents=True)
    smuggled_version.write_bytes(b"\x80\x04undeclared pickled weights")
    assert _runtime_static_path_allowed(smuggled_version) is False
    assert _parent_allows(str(smuggled_version), tmp_path) is False
    relocated_record = site_packages / "pkg" / "installed-files.txt"
    relocated_record.write_text("model.py\n", encoding="utf-8")
    assert _runtime_static_path_allowed(relocated_record) is False


@pytest.mark.smoke
def test_dist_info_category_reads_are_package_metadata(tmp_path: Path) -> None:
    """Any bounded-text, non-weight-suffixed file inside dist-info/egg-info is metadata."""

    dist_info = tmp_path / "site-packages" / "demo-1.0.dist-info"
    dist_info.mkdir(parents=True)
    for name in ("LICENSE", "REQUESTED", "AUTHORS.rst"):
        member = dist_info / name
        member.write_text("package metadata text\n", encoding="utf-8")
        assert _runtime_import_metadata_path_allowed(member) is True, name
        assert _parent_allows(str(member), tmp_path) is True, name


@pytest.mark.smoke
def test_missing_closed_metadata_probe_stays_allowed(tmp_path: Path) -> None:
    """importlib probes for absent closed names keep working (no existence proof needed)."""

    probe = tmp_path / "site-packages" / "demo-1.0.dist-info" / "RECORD"
    assert _runtime_import_metadata_path_allowed(probe) is True


# ---------------------------------------------------------------------------
# B1(c): the extensionless weight heuristic requires bulk or tensor magic
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_small_non_tensor_extensionless_file_is_not_model_data(tmp_path: Path) -> None:
    """A small textual extensionless file cannot trip the weight heuristic."""

    benign = tmp_path / "os-release"
    benign.write_text('NAME="Debian GNU/Linux"\nVERSION_ID="12"\n', encoding="utf-8")
    assert _runtime_model_data_path(benign) is False


@pytest.mark.smoke
def test_bulky_extensionless_file_is_still_model_data(tmp_path: Path) -> None:
    """Direction 2: real bulk keeps firing the heuristic regardless of the name."""

    bulky = tmp_path / "VERSION"
    bulky.write_bytes(b"\x00" * (1024**2 + 1))
    assert _runtime_model_data_path(bulky) is True


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("label", "head"),
    [
        ("zip-container", b"PK\x03\x04rest-of-a-torch-zipfile"),
        ("numpy", b"\x93NUMPY\x01\x00rest"),
        ("pickle-protocol-2", b"\x80\x02}q\x00."),
        ("hdf5", b"\x89HDF\r\n\x1a\nrest"),
        ("safetensors", b"\x10\x00\x00\x00\x00\x00\x00\x00" + b'{"a":{"dtype"'),
    ],
)
def test_small_extensionless_tensor_magic_is_still_model_data(
    tmp_path: Path, label: str, head: bytes
) -> None:
    """Direction 2: a tiny extensionless file with tensor-format magic stays refused."""

    payload = tmp_path / "weights"
    payload.write_bytes(head)
    assert _runtime_model_data_path(payload) is True, label


# ---------------------------------------------------------------------------
# The m1115-style tripwire stays armed (LOCKED principle, refusal direction)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_weight_suffixed_names_stay_model_data_everywhere(tmp_path: Path) -> None:
    """Author-shaped weight names are refused by NAME, even tiny and textual."""

    outside = tmp_path / "site-packages" / "pkg" / "weights.npz"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"undeclared model data")
    assert _runtime_model_data_path(outside) is True
    assert _parent_allows(str(outside), tmp_path) is False


@pytest.mark.smoke
def test_model_data_planted_inside_dist_info_is_still_refused(tmp_path: Path) -> None:
    """The dist-info category grant cannot launder weight-suffixed or binary payloads."""

    dist_info = tmp_path / "site-packages" / "demo-1.0.dist-info"
    dist_info.mkdir(parents=True)
    planted = dist_info / "payload.pt"
    planted.write_bytes(b"PK\x03\x04hidden weights")
    binary = dist_info / "blob"
    binary.write_bytes(b"\x00\x01\x02 binary payload")
    assert _runtime_import_metadata_path_allowed(planted) is False
    assert _parent_allows(str(planted), tmp_path) is False
    assert _runtime_import_metadata_path_allowed(binary) is False


@pytest.mark.smoke
def test_large_data_outside_system_trees_is_still_refused(tmp_path: Path) -> None:
    """A >=1 MiB .dat outside /usr/share (the m4066 shape, relocated) stays refused."""

    blob = tmp_path / "site-packages" / "pkg" / "table.dat"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\x00" * (1024**2 + 1))
    assert _runtime_model_data_path(blob) is True
    assert _parent_allows(str(blob), tmp_path) is False


# ---------------------------------------------------------------------------
# Item 12 (F51 DECIDED_PLAN; author 2.3/R-5): a file the classifier cannot STAT
# is not "proven model data". The R6b re-run recorded 15/52 `failed:policy /
# checkpoint-read` rows whose only "proof" was the clone's own 50-byte
# `menagerie/__init__.py` -- a .py the clone worker could not stat because its
# package root fell outside the sandbox's read grants. Positive proof (a weight
# suffix, a tensor magic, or observed bulk) stays the ONLY poisoning path; a
# read the classifier cannot measure is refused as a benign probe, never a
# poisoning checkpoint read.
# ---------------------------------------------------------------------------


def _unstattable_file(tmp_path: Path, name: str) -> "Path | None":
    """Return a path under a permission-denied dir, or None if unreproducible."""

    import os

    locked = tmp_path / "locked-origin"
    locked.mkdir()
    target = locked / name
    target.write_bytes(b"x" * 50)  # the 50-byte clone __init__.py shape
    os.chmod(locked, 0o000)
    try:
        target.exists()
    except OSError:
        return target  # stat is genuinely denied: the R6b condition holds
    # Running as root, or a filesystem that ignores the mode bits: cannot
    # reproduce the stat refusal, so this environment cannot exercise the arm.
    os.chmod(locked, 0o755)
    return None


@pytest.mark.smoke
def test_unstattable_py_is_not_proven_model_data(tmp_path: Path) -> None:
    """A `.py` whose stat is refused is not model data (the R6b __init__.py)."""

    import os

    target = _unstattable_file(tmp_path, "__init__.py")
    if target is None:
        pytest.skip("stat refusal not reproducible on this host")
    try:
        # No suffix, magic, or bulk can be observed, so the classifier cannot
        # PROVE model data -- and an unprovable read is not a checkpoint read.
        assert _runtime_model_data_path(target) is False
    finally:
        os.chmod(target.parent, 0o755)


@pytest.mark.smoke
def test_unstattable_read_is_refused_but_never_poisons(tmp_path: Path) -> None:
    """The stat-failure arm records a non-poisoning refused-read probe, not a poison."""

    import os

    from menagerie.crawler.policy import PolicyViolation

    target = _unstattable_file(tmp_path, "__init__.py")
    if target is None:
        pytest.skip("stat refusal not reproducible on this host")
    policy = ExecutionPolicy(tmp_path / "scratch")
    try:
        with pytest.raises((PermissionError, FileNotFoundError)) as raised:
            policy._audit_path(str(target), writing=False)
        # A benign probe -- refused by the OS-shaped error a library's own
        # try/except absorbs, NEVER the checkpoint-read PolicyViolation.
        assert not isinstance(raised.value, PolicyViolation)
        assert policy.observation.checkpoint_or_weight_read_attempted is False
        assert not policy.observation.checkpoint_paths
        assert str(target) in policy.observation.refused_read_probe_paths
    finally:
        os.chmod(target.parent, 0o755)


@pytest.mark.smoke
def test_unstattable_weight_suffix_still_poisons(tmp_path: Path) -> None:
    """Positive proof is untouched: an unstattable .pt still poisons the receipt.

    The LOCKED direction. A weight-suffixed read is proven model data by its
    suffix alone -- checked before any stat -- so refusing to stat it can never
    downgrade a real checkpoint read to a benign probe.
    """

    import os

    from menagerie.crawler.policy import PolicyViolation

    target = _unstattable_file(tmp_path, "weights.pt")
    if target is None:
        pytest.skip("stat refusal not reproducible on this host")
    policy = ExecutionPolicy(tmp_path / "scratch")
    try:
        assert _runtime_model_data_path(target) is True
        with pytest.raises(PolicyViolation) as raised:
            policy._audit_path(str(target), writing=False)
        assert raised.value.reason_code == "checkpoint-read"
        assert policy.observation.checkpoint_or_weight_read_attempted is True
        assert str(target) in policy.observation.checkpoint_paths
    finally:
        os.chmod(target.parent, 0o755)
