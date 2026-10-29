"""EN-7: rotation safety -- disk authority, whitespace guard, publication serialization, driver-lock exclusion.

Rotation cost 3 of 4 GO-day attempts. The failure classes pinned here:

* **Whitespace roots** -- conda/mamba tooling breaks on paths with whitespace
  (the space-in-volume-name shape: ``/Volumes/Devmini SSD``); the rotation must
  refuse such solve roots TYPED and BEFORE any backend work.
* **Single-root disk authority** -- only ``env_root`` was floor-checked; a full
  state-root or publication filesystem failed mid-flight instead. Every write
  destination (state root, env root, tmp root, publication lock dir, campaign
  state roots) is measured INDEPENDENTLY through the runtime
  ``DiskSpaceAuthority`` shape, refusal names the breached root, and the check
  re-runs before publication so no single-root breach can publish.
* **Publication collision** (sol's reproduced OSError 66) -- two concurrent
  rotations publishing the SAME proven family must converge (identical winner);
  a DIFFERENT concurrent family must refuse under compare-and-swap, and the
  whole publication runs under an exclusive per-intent lock.
* **Campaign driver-lock exclusion** -- a rotation never runs concurrent with a
  leg: it takes the account-global execution flock the worker supervisor holds
  during legs, and refuses typed when a leg holds it.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Optional

import pytest

from menagerie.crawler.envs import family_manifest_path, load_environment_registry
from menagerie.crawler.execution_lock import ExecutionLockTimeout
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.tests.rotation_transaction_support import (
    INTENT,
    TARGET,
    ScriptedRotationBackend,
    family_members,
    mini_envs_root,
    rotation_provenance,
    solve_fixture,
)
from menagerie.crawler.tools.rotate_setup_artifacts import (
    RotationError,
    publish_rotation_family,
    rotate_intent,
    rotation_publication_lock_path,
    rotate_intent_write_destinations,
)

pytestmark = pytest.mark.smoke

_OLD = solve_fixture("old")
_NEW = solve_fixture("new")


def _rotate(roots, *, backend=None, **overrides):
    """Run rotate_intent against the mini universe with test-friendly defaults."""

    backend = backend or ScriptedRotationBackend(roots["state"], _NEW)
    defaults = dict(
        envs_root=roots["envs"],
        target=TARGET,
        environment_command=("scripted-test-operator",),
        state_root=roots["state"],
        env_root=roots["env"],
        minimum_free_bytes=1,
        backend=backend,
        execution_lock_path=roots["flock"],
    )
    defaults.update(overrides)
    return rotate_intent(INTENT, **defaults)


@pytest.fixture()
def roots(tmp_path: Path) -> dict:
    """Standard mini universe: envs root, fresh state/env roots, private flock."""

    envs_root = mini_envs_root(tmp_path, old=_OLD)
    state_root = tmp_path / "rotation-state"
    state_root.mkdir()
    env_root = tmp_path / "env-store"
    env_root.mkdir()
    return {
        "envs": envs_root,
        "state": state_root,
        "env": env_root,
        "flock": tmp_path / "execution.flock",
        "tmp": tmp_path,
    }


# ---------------------------------------------------------------------------
# (b) whitespace guard
# ---------------------------------------------------------------------------


def test_whitespace_state_root_refuses_before_any_backend_work(roots, tmp_path: Path) -> None:
    """The space-in-volume-name shape refuses typed; the backend never runs."""

    spaced = tmp_path / "Devmini SSD" / "rotation-state"
    spaced.mkdir(parents=True)

    class Untouchable:
        def __getattr__(self, name):  # pragma: no cover - the guard must fire first
            raise AssertionError("backend must not be touched behind a whitespace root")

    with pytest.raises(RotationError, match="whitespace"):
        _rotate(roots, backend=Untouchable(), state_root=spaced)


def test_whitespace_env_root_refuses(roots, tmp_path: Path) -> None:
    """The transient-prefix root is a conda write destination too."""

    spaced = tmp_path / "Env Store"
    spaced.mkdir(parents=True)
    with pytest.raises(RotationError, match="whitespace"):
        _rotate(roots, env_root=spaced)


# ---------------------------------------------------------------------------
# (a) multi-destination disk authority
# ---------------------------------------------------------------------------


def test_every_write_destination_is_measured_independently(roots) -> None:
    """A breach on ANY write root refuses typed and names the root."""

    destinations = rotate_intent_write_destinations(
        envs_root=roots["envs"],
        intent=INTENT,
        state_root=roots["state"],
        env_root=roots["env"],
        campaign_state_roots=(roots["tmp"] / "campaign-state",),
    )
    roles = {role for role, _ in destinations}
    assert {"state-root", "env-root", "tmp-root", "publication", "campaign-state"} <= roles

    breached = roots["state"].resolve()

    def low_only_on_state(path: Path) -> int:
        probe = Path(path).resolve()
        if probe == breached or str(probe).startswith(str(breached) + os.sep):
            return 0
        return 1 << 40

    # The state root's filesystem is full; env_root alone passing must not admit.
    with pytest.raises(RotationError, match="state-root"):
        _rotate(roots, disk_free=low_only_on_state, minimum_free_bytes=1024)


def test_no_publication_on_a_breach_that_appears_after_the_solve(roots) -> None:
    """The authority re-measures BEFORE publication; the old family survives."""

    breached = {"now": False}

    def breach_after_proof() -> None:
        breached["now"] = True

    def shrinking(path: Path) -> int:
        # Plenty of space at admission, none by publication time.
        return 0 if breached["now"] else (1 << 40)

    backend = ScriptedRotationBackend(roots["state"], _NEW, after_probe_proof=breach_after_proof)
    with pytest.raises(RotationError, match="free"):
        _rotate(roots, backend=backend, disk_free=shrinking, minimum_free_bytes=1024)
    registry = load_environment_registry(roots["envs"], target=TARGET)
    assert registry.intents[INTENT].lock.status == "locked"
    assert (
        hash_bytes(registry.intents[INTENT].lock.lock_bytes or b"") == hash_bytes(_OLD.lock_bytes)
    )


# ---------------------------------------------------------------------------
# (c) publication serialization
# ---------------------------------------------------------------------------


def _members(fixture, envs_root):
    return family_members(fixture, rotation_provenance(envs_root, INTENT, fixture))


def test_publication_holds_an_exclusive_per_intent_lock(roots) -> None:
    """During the manifest switch the per-intent publication lock is held."""

    lock_dir = roots["envs"] / INTENT / "locks"
    observed: dict[str, Optional[bool]] = {"held": None}

    def probe_lock(phase: str) -> None:
        if phase != "manifest-switch":
            return
        path = rotation_publication_lock_path(lock_dir, TARGET)
        with path.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                observed["held"] = True
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                observed["held"] = False

    publish_rotation_family(
        lock_dir, TARGET, _members(_NEW, roots["envs"]), phase_observer=probe_lock
    )
    assert observed["held"] is True


def test_identical_concurrent_winner_converges(roots, monkeypatch) -> None:
    """Sol's OSError 66: the loser of an identical-bytes race converges."""

    lock_dir = roots["envs"] / INTENT / "locks"
    members = _members(_NEW, roots["envs"])
    real_rename = os.rename

    def racing_rename(src, dst, *args, **kwargs):
        dst_p = Path(dst)
        if dst_p.parent.name == "generations" and not dst_p.exists():
            # The winner publishes the identical generation first.
            dst_p.mkdir(parents=True)
            for name, content in members.items():
                (dst_p / name).write_bytes(content)
            raise OSError(66, "Directory not empty", str(dst))
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", racing_rename)
    publish_rotation_family(lock_dir, TARGET, members)
    manifest = json.loads(family_manifest_path(lock_dir, TARGET).read_text())
    expected_generation = stable_hash(
        {"lock_family_generation": {name: hash_bytes(content) for name, content in members.items()}}
    ).removeprefix("sha256:")
    assert manifest["generation"] == expected_generation


def test_divergent_concurrent_winner_refuses(roots, monkeypatch) -> None:
    """A rename collision against DIFFERENT bytes is tampering, never converged."""

    lock_dir = roots["envs"] / INTENT / "locks"
    members = _members(_NEW, roots["envs"])
    real_rename = os.rename

    def racing_rename(src, dst, *args, **kwargs):
        dst_p = Path(dst)
        if dst_p.parent.name == "generations" and not dst_p.exists():
            dst_p.mkdir(parents=True)
            (dst_p / f"{TARGET}.lock").write_bytes(b"attacker bytes\n")
            raise OSError(66, "Directory not empty", str(dst))
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", racing_rename)
    with pytest.raises(RotationError, match="does not match|immutable"):
        publish_rotation_family(lock_dir, TARGET, members)


def test_compare_and_swap_refuses_a_third_family_published_meanwhile(roots) -> None:
    """CAS: the manifest moved to a THIRD family since the rotation observed old."""

    lock_dir = roots["envs"] / INTENT / "locks"
    third = solve_fixture("third")
    publish_rotation_family(lock_dir, TARGET, _members(third, roots["envs"]))
    observed_old_generation = "0" * 16  # what this rotation believes is committed
    with pytest.raises(RotationError, match="another rotation|compare-and-swap"):
        publish_rotation_family(
            lock_dir,
            TARGET,
            _members(_NEW, roots["envs"]),
            expected_prior_generation=observed_old_generation,
        )
    # The third family stays authoritative: no overwrite happened.
    manifest = json.loads(family_manifest_path(lock_dir, TARGET).read_text())
    third_generation = stable_hash(
        {
            "lock_family_generation": {
                name: hash_bytes(content)
                for name, content in _members(third, roots["envs"]).items()
            }
        }
    ).removeprefix("sha256:")
    assert manifest["generation"] == third_generation


def test_rotate_intent_passes_its_observed_prior_through_cas(roots) -> None:
    """End to end: a family switched mid-rotation refuses at publication."""

    def switch_family() -> None:
        third = solve_fixture("third")
        publish_rotation_family(
            roots["envs"] / INTENT / "locks", TARGET, _members(third, roots["envs"])
        )

    backend = ScriptedRotationBackend(roots["state"], _NEW, after_probe_proof=switch_family)
    with pytest.raises(RotationError, match="another rotation|compare-and-swap"):
        _rotate(roots, backend=backend)


# ---------------------------------------------------------------------------
# (d) campaign driver-lock exclusion
# ---------------------------------------------------------------------------


def test_rotation_refuses_while_a_leg_holds_the_execution_lock(roots) -> None:
    """A rotation never runs concurrent with a leg; the refusal is typed and fast."""

    flock_path = roots["flock"]
    flock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = os.open(flock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        with pytest.raises((RotationError, ExecutionLockTimeout)):
            _rotate(roots, execution_lock_timeout_seconds=0.2)
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)


def test_rotation_acquires_and_releases_the_execution_lock(roots) -> None:
    """A normal rotation takes the lock, rotates, and releases it."""

    old_identity, new_identity = _rotate(roots)
    assert old_identity != new_identity
    # Released: the test can take it immediately.
    descriptor = os.open(roots["flock"], os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
