"""Environment-operator orphan GC: the T1-19 crash window and the sweeper.

A SIGKILL between the package-cache ``mkdir`` and the registration write used
to brick the ``(prefix, lock)`` pair permanently: the cache name is a pure
function of the pair, the retry died on ``mkdir(exist_ok=False)``, and the
removal path could not name the orphan. These tests pin the repaired create
ordering (registration is durable before any cache bytes exist), the
self-healing retry over historical residue, and the sweeper that makes every
remaining orphan class nameable — report-only enumeration, explicit removal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from menagerie.crawler import operator_environment
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.operator_environment import (
    EnvironmentCasReachability,
    EnvironmentOperator,
    PROTOCOL_VERSION,
    _prefix_identifier,
    remove_environment_orphan,
    sweep_environment_orphans,
)

pytestmark = pytest.mark.smoke


class _CreateRunner:
    """Minimal runner satisfying solve/create/probe command shapes."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.registration_seen_before_cache: list[bool] = []

    def __call__(self, command, environment, timeout_seconds):
        self.commands.append(tuple(command))
        argv = tuple(command)
        if "create" in argv:
            prefix = Path(argv[argv.index("--prefix") + 1])
            meta = prefix / "conda-meta"
            meta.mkdir(parents=True, exist_ok=True)
            (meta / "history").write_text("created\n")
        return operator_environment.CommandObservation(returncode=0, stdout="", stderr="")


def _lock_bytes() -> bytes:
    body = b"@EXPLICIT\n"
    digest = hash_bytes(b"pkg-bytes").removeprefix("sha256:")
    return body + (f"https://conda.test/noarch/pkg-1.0-0.conda#{digest}\n").encode()


def _operator(tmp_path: Path) -> tuple[EnvironmentOperator, Path, _CreateRunner]:
    state_root = tmp_path / "operator-state"
    runner = _CreateRunner()
    mamba = tmp_path / "bin" / "mamba"
    mamba.parent.mkdir(parents=True, exist_ok=True)
    mamba.write_text("#!/bin/sh\nexit 0\n")
    mamba.chmod(0o755)
    operator = EnvironmentOperator(
        mamba_command=(str(mamba),),
        conda_command=("/fake/conda",),
        conda_lock_command=("/fake/conda-lock",),
        state_root=state_root,
        runner=runner,
    )
    return operator, state_root, runner


def _staged_lock(tmp_path: Path, state_root: Path) -> Path:
    lock_file = tmp_path / "explicit.lock"
    lock_bytes = _lock_bytes()
    lock_file.write_bytes(lock_bytes)
    digest = hash_bytes(b"pkg-bytes").removeprefix("sha256:")
    cas_dir = state_root / "artifact-cas" / "sha256"
    cas_dir.mkdir(parents=True, exist_ok=True)
    (cas_dir / digest).write_bytes(b"pkg-bytes")
    return lock_file


def _expected_cache(state_root: Path, prefix: Path, lock_file: Path) -> Path:
    lock_sha = hash_bytes(lock_file.read_bytes())
    return state_root / "package-caches" / f"{_prefix_identifier(prefix)}-{lock_sha[7:19]}"


# -- the T1-19 crash window ------------------------------------------------------


def test_create_survives_a_registrationless_orphan_cache(tmp_path):
    """The historical SIGKILL residue no longer bricks the (prefix, lock) pair."""

    operator, state_root, runner = _operator(tmp_path)
    lock_file = _staged_lock(tmp_path, state_root)
    prefix = tmp_path / "envs" / "core-abc"

    orphan = _expected_cache(state_root, prefix, lock_file)
    orphan.mkdir(parents=True)
    (orphan / "leftover.partial").write_bytes(b"crash residue")

    result = operator.create(lock_file, prefix)
    assert result["prefix"] == str(prefix)
    assert not (orphan / "leftover.partial").exists()
    registration = state_root / "prefix-registrations" / f"{_prefix_identifier(prefix)}.json"
    assert registration.is_file()


def test_create_registers_before_any_cache_bytes_exist(tmp_path):
    """Ordering pin: the registration write precedes the cache mkdir."""

    operator, state_root, runner = _operator(tmp_path)
    lock_file = _staged_lock(tmp_path, state_root)
    prefix = tmp_path / "envs" / "core-order"
    cache = _expected_cache(state_root, prefix, lock_file)

    observed: list[bool] = []
    real_replace = operator_environment.atomic_replace_bytes

    def _spy(path: Path, payload: bytes) -> None:
        if path.parent.name == "prefix-registrations":
            observed.append(cache.exists())
        real_replace(path, payload)

    original = operator_environment.atomic_replace_bytes
    operator_environment.atomic_replace_bytes = _spy
    try:
        operator.create(lock_file, prefix)
    finally:
        operator_environment.atomic_replace_bytes = original
    assert observed == [False]


def test_create_crash_after_registration_write_heals_on_retry(tmp_path):
    """New-order residue (registration, no cache) converges on the next create."""

    operator, state_root, runner = _operator(tmp_path)
    lock_file = _staged_lock(tmp_path, state_root)
    prefix = tmp_path / "envs" / "core-heal"
    cache = _expected_cache(state_root, prefix, lock_file)
    registration_path = state_root / "prefix-registrations" / f"{_prefix_identifier(prefix)}.json"
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "prefix": str(prefix),
                "package_cache": str(cache),
                "lock_sha256": hash_bytes(lock_file.read_bytes()),
            },
            sort_keys=True,
        )
        + "\n"
    )

    result = operator.create(lock_file, prefix)
    assert result["prefix"] == str(prefix)
    assert cache.is_dir()


# -- sweeper: enumeration is report-only, orphans become nameable -----------------


def _sweep_fixture(tmp_path: Path):
    """A state root holding one live registration and every orphan class."""

    operator, state_root, runner = _operator(tmp_path)
    lock_file = _staged_lock(tmp_path, state_root)
    prefix = tmp_path / "envs" / "core-live"
    operator.create(lock_file, prefix)

    caches = state_root / "package-caches"
    wheelhouses = state_root / "wheelhouses"
    wheelhouses.mkdir(exist_ok=True)
    registrations = state_root / "prefix-registrations"

    orphan_cache = caches / f"{'ab' * 12}-{'cd' * 6}"
    orphan_cache.mkdir()
    (orphan_cache / "pkg.conda").write_bytes(b"orphan cache bytes")

    orphan_wheelhouse = wheelhouses / f"{'ef' * 12}-{'12' * 6}"
    orphan_wheelhouse.mkdir()
    (orphan_wheelhouse / "wheel.whl").write_bytes(b"orphan wheel")

    dangling_prefix = tmp_path / "envs" / "core-gone"
    dangling_registration = registrations / f"{_prefix_identifier(dangling_prefix)}.json"
    dangling_cache = caches / f"{_prefix_identifier(dangling_prefix)}-{'34' * 6}"
    dangling_cache.mkdir()
    dangling_registration.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "prefix": str(dangling_prefix),
                "package_cache": str(dangling_cache),
                "lock_sha256": hash_bytes(b"whatever"),
            },
            sort_keys=True,
        )
        + "\n"
    )

    stale_temp = state_root / "tmp" / "overlay-download-deadbeef0123"
    stale_temp.mkdir(parents=True)
    (stale_temp / "wheel.part").write_bytes(b"partial download")
    fresh_temp = state_root / "tmp" / "overlay-download-fresh0000000"
    fresh_temp.mkdir()

    cas_partial = state_root / "artifact-cas" / "sha256" / f".{'aa' * 32}.12345.feedface.tmp"
    cas_partial.write_bytes(b"partial artifact")

    staging = state_root / "wheel-cas" / "sha256" / f"{'bb' * 32}.staging-77"
    staging.mkdir(parents=True)

    solve_temp = state_root / "solves" / "core" / "osx-arm64" / ".solve-zzz"
    solve_temp.mkdir(parents=True)

    registration_temp = registrations / f".{'cc' * 12}.json.4242.tmp"
    registration_temp.write_bytes(b"{}")

    orphan_lock = registrations / f"{'dd' * 12}.lock"
    orphan_lock.write_bytes(b"")

    old = 1_000_000_000.0
    for path in (
        stale_temp,
        cas_partial,
        staging,
        solve_temp,
        registration_temp,
        orphan_lock,
    ):
        os.utime(path, (old, old))

    return (
        state_root,
        prefix,
        {
            "unregistered-package-cache": orphan_cache,
            "unregistered-wheelhouse": orphan_wheelhouse,
            "dangling-registration": dangling_registration,
            "stale-temp": stale_temp,
            "cas-partial": cas_partial,
            "wheel-cas-staging": staging,
            "solve-temp": solve_temp,
            "registration-temp": registration_temp,
            "orphan-pair-lock": orphan_lock,
        },
        fresh_temp,
    )


def test_sweep_names_every_orphan_class_and_deletes_nothing(tmp_path):
    state_root, prefix, expected, fresh_temp = _sweep_fixture(tmp_path)
    before = sorted(str(path) for path in state_root.rglob("*"))
    orphans = sweep_environment_orphans(state_root)
    after = sorted(str(path) for path in state_root.rglob("*"))
    assert after == before, "sweep enumeration must be read-only"

    by_kind = {orphan.kind: orphan for orphan in orphans}
    for kind, path in expected.items():
        assert kind in by_kind, f"orphan class {kind} not named"
        matching = [o for o in orphans if Path(o.path).resolve() == path.resolve()]
        assert matching, f"{kind}: {path} not named"

    named = {str(Path(orphan.path).resolve()) for orphan in orphans}
    live_registration = state_root / "prefix-registrations" / f"{_prefix_identifier(prefix)}.json"
    live_cache = json.loads(live_registration.read_text())["package_cache"]
    assert str(Path(live_cache).resolve()) not in named
    assert str(live_registration.resolve()) not in named
    assert str(fresh_temp.resolve()) not in named
    digest = hash_bytes(b"pkg-bytes").removeprefix("sha256:")
    assert str((state_root / "artifact-cas" / "sha256" / digest).resolve()) not in named


def test_remove_environment_orphan_reclaims_and_respects_liveness(tmp_path):
    state_root, prefix, expected, fresh_temp = _sweep_fixture(tmp_path)
    orphans = sweep_environment_orphans(state_root)

    for orphan in orphans:
        assert remove_environment_orphan(state_root, orphan) is True
    for path in expected.values():
        assert not path.exists(), path

    remaining = sweep_environment_orphans(state_root)
    assert remaining == ()

    live_registration = state_root / "prefix-registrations" / f"{_prefix_identifier(prefix)}.json"
    assert live_registration.is_file()


def test_remove_skips_a_cache_that_became_registered(tmp_path):
    """Replay-derived removal: adoption between sweep and remove wins."""

    state_root, prefix, expected, fresh_temp = _sweep_fixture(tmp_path)
    orphans = sweep_environment_orphans(state_root)
    cache_orphan = next(o for o in orphans if o.kind == "unregistered-package-cache")
    adopter_prefix = tmp_path / "envs" / "core-adopter"
    registration = (
        state_root / "prefix-registrations" / f"{_prefix_identifier(adopter_prefix)}.json"
    )
    registration.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "prefix": str(adopter_prefix),
                "package_cache": str(cache_orphan.path),
                "lock_sha256": hash_bytes(b"adopted"),
            },
            sort_keys=True,
        )
        + "\n"
    )

    assert remove_environment_orphan(state_root, cache_orphan) is False
    assert Path(cache_orphan.path).is_dir()


def test_cas_gc_is_rooted_in_committed_digest_reachability(tmp_path: Path) -> None:
    """Only immutable CAS objects outside the committed closure are reclaimable."""

    state_root = tmp_path / "operator-state"
    cas_root = state_root / "artifact-cas" / "sha256"
    cas_root.mkdir(parents=True)
    reachable_digest = hash_bytes(b"reachable").removeprefix("sha256:")
    unreachable_digest = hash_bytes(b"superseded").removeprefix("sha256:")
    reachable = cas_root / reachable_digest
    unreachable = cas_root / unreachable_digest
    reachable.write_bytes(b"reachable")
    unreachable.write_bytes(b"superseded")
    closure = EnvironmentCasReachability(
        artifact_digests=frozenset({reachable_digest}),
        wheel_digests=frozenset(),
    )

    orphans = sweep_environment_orphans(state_root, cas_reachability=closure)

    assert {orphan.path for orphan in orphans if orphan.kind == "cas-unreachable"} == {unreachable}
    orphan = next(orphan for orphan in orphans if orphan.kind == "cas-unreachable")
    assert remove_environment_orphan(state_root, orphan) is False
    assert remove_environment_orphan(state_root, orphan, cas_reachability=closure) is True
    assert reachable.read_bytes() == b"reachable"
    assert not unreachable.exists()
