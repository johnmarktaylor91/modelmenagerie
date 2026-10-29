"""Environments: committed ``envs/<name>/`` locks, built on the worker, with a canary each.

Owned by the environment and runner lane. Each environment is ``envs/<name>/pyproject.toml``
plus its ``uv.lock``, or ``pixi.toml`` plus ``pixi.lock`` for mixed Conda/PyPI runtimes.
Builds use frozen uv or locked Pixi installation in ``<root>/<name>-<lock8>``, so a lock change builds a new prefix beside the old one and the ledger's
``env`` value (``<name>@<lock8>``) always names the exact lock that ran. Built environments live
under ``root`` (the CLI reads ``MENAGERIE_ENVS``); uv's cache for base builds stays at its default
under ``~/.cache``, never under the data directory, and overlays use their own cache under ``root``
(``overlay_cache``) so evicting them frees disk. Per-recipe extras go into an overlay of a built
environment (``overlay``), installed outside the sandbox where PyPI is reachable, from wheels
only: no third-party build script ever runs with network or credentials. A compiled extension can
only be built inside the sandbox, without network, where a failure is ``compiled_ext``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from menagerie.pipeline import overlay_cache

SPECS = Path(__file__).resolve().parents[2] / "envs"
BUILT_MARKER = ".menagerie-lock"
CANARY_TIMEOUT_S = 900
INSTALL_TIMEOUT_S = 1800
OVERLAY_COST = ".overlay-cost"  # cached bytes an overlay holds beyond its base (overlay_cache.cost)
# An overlay used or started this recently may be mid-build (``INSTALL_TIMEOUT_S``) or wanted by its
# lane's next try; a running attempt's overlay is kept by its claimed request (``keep_for``), so the
# window only has to outlast a build. Steady-state overlay disk is about the build rate times this.
OVERLAY_IN_USE_S = 3600
# uv's words when a requirement has no wheel and building from source is off.
_SDIST_ONLY = re.compile(r"no usable wheels|building from source is disabled|only-binary", re.I)


class CanaryError(RuntimeError):
    """A freshly built environment whose canary failed; it is not marked built."""


class ExtrasError(Exception):
    """Extras that cannot be installed; ``cls`` is ``missing_dependency`` or its ``:sdist`` form."""

    def __init__(self, cls: str, msg: str) -> None:
        super().__init__(msg)
        self.cls = cls


def lock_path(name: str) -> Path:
    """Return the committed lock of environment ``name``."""

    path = SPECS / name / ("pixi.lock" if (SPECS / name / "pixi.toml").is_file() else "uv.lock")
    if not path.is_file():
        raise FileNotFoundError(f"environment {name!r} has no committed lock at {path}")
    return path


def lock_id(name: str) -> str:
    """Return ``<name>@<first 8 hex of the lock's sha256>``, the ledger's ``env`` value."""

    return f"{name}@{hashlib.sha256(lock_path(name).read_bytes()).hexdigest()[:8]}"


def prefix(name: str, *, root: Path) -> Path:
    """Return where the current lock of ``name`` is (or would be) built under ``root``."""

    workspace = Path(root) / lock_id(name).replace("@", "-")
    return workspace / ".pixi" / "envs" / "default" if lock_path(name).name == "pixi.lock" else workspace


def python(name: str, *, root: Path) -> Path:
    """Return the interpreter of the built environment; raise if it is not built."""

    target = prefix(name, root=root)
    marker = target / BUILT_MARKER
    if not marker.is_file() or marker.read_text().strip() != lock_id(name):
        raise FileNotFoundError(f"{lock_id(name)} is not built under {root}")
    return target / "bin" / "python"


def package_version(name: str, package: str) -> str | None:
    """Return the locked version of ``package`` in environment ``name``, if it is locked."""

    lock = lock_path(name)
    if lock.name == "pixi.lock":
        return _pixi_version(lock, package)
    text = lock.read_text()
    found = re.search(rf'^name = "{re.escape(package)}"\nversion = "([^"]+)"', text, re.MULTILINE)
    return found.group(1) if found else None


def _pixi_version(lock: Path, package: str) -> str | None:
    import yaml
    from packaging.utils import canonicalize_name

    data = yaml.safe_load(lock.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('packages'), list):
        raise ValueError('invalid Pixi package table')
    for entry in data['packages']:
        dist, version = _pixi_package(entry)
        if canonicalize_name(dist) == canonicalize_name(package):
            return version
    return None


def _pixi_package(entry: dict) -> tuple[str, str]:
    """Read an artifact identity without assuming PyPI and Conda share a schema; Conda's
    ``pytorch`` is reported as ``torch``, the name the inventory's key packages use."""

    dist, version = pixi_identity(entry)
    return ('torch' if 'conda' in entry and dist == 'pytorch' else dist), version


def pixi_identity(entry: dict) -> tuple[str, str]:
    """Return the ``(name, version)`` a Pixi lock entry records, under the lock's own name."""
    from urllib.parse import unquote, urlparse
    from packaging.utils import parse_wheel_filename

    if not isinstance(entry, dict):
        raise ValueError('invalid Pixi package entry')
    if 'name' in entry:
        return entry['name'], str(entry['version'])
    if 'conda' in entry:
        filename = Path(urlparse(entry['conda']).path).name
        stem = filename.removesuffix('.conda').removesuffix('.tar.bz2')
        dist, version, _ = stem.rsplit('-', 2)
        return dist, version
    if 'pypi' in entry:
        filename = unquote(Path(urlparse(entry['pypi']).path).name)
        if filename.endswith('.whl'):
            dist, version, _, _ = parse_wheel_filename(filename)
            return dist, str(version)
    raise ValueError('Pixi package has no supported locked artifact')


def build(name: str, *, root: Path) -> Path:
    """Build environment ``name`` from its committed lock under ``root``; return its prefix.

    A fresh build runs the environment's canary before it is marked built, so every built
    environment has passed its canary; a failing canary raises ``CanaryError`` and leaves the
    prefix unmarked (the next build retries it).
    """

    target = prefix(name, root=root)
    try:
        python(name, root=root)
        return target
    except FileNotFoundError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    _install(name, target)
    if not _run_canary(name, target / "bin" / "python"):
        raise CanaryError(f"{lock_id(name)} built under {root} but its canary failed")
    (target / BUILT_MARKER).write_text(lock_id(name) + "\n")
    return target


def _install(name: str, target: Path) -> None:
    if lock_path(name).name == "pixi.lock":
        workspace = target.parents[2]
        workspace.mkdir(parents=True, exist_ok=True)
        for filename in ("pixi.toml", "pixi.lock"):
            shutil.copyfile(SPECS / name / filename, workspace / filename)
        command = ["pixi", "install", "--locked", "--manifest-path", str(workspace / "pixi.toml")]
        env = {**_environ(), "PIXI_NO_CONFIG": "true"}
        env.pop("PIXI_CONFIG_FILE", None)
    else:
        command = ["uv", "sync", "--frozen", "--no-install-project", "--managed-python",
                   "--project", str(SPECS / name)]
        env = {**_environ(), "UV_PROJECT_ENVIRONMENT": str(target)}
    subprocess.run(command, check=True, env=env, timeout=INSTALL_TIMEOUT_S)


def canary(name: str, *, root: Path) -> bool:
    """Run the built environment's import-and-forward canary; return whether it passed."""

    return _run_canary(name, python(name, root=root))


def _run_canary(name: str, interpreter: Path) -> bool:
    script = SPECS / name / "canary.py"
    env = {**_environ(), "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "YOLO_OFFLINE": "true"}
    done = subprocess.run([str(interpreter), str(script)], env=env,
                          timeout=CANARY_TIMEOUT_S, check=False)
    return done.returncode == 0


def inventory(*, root: Path) -> dict[str, str]:
    """Return the built environments as ``{name: lock_id}``."""

    built = {}
    for lock in sorted([*SPECS.glob("*/uv.lock"), *SPECS.glob("*/pixi.lock")]):
        name = lock.parent.name
        try:
            python(name, root=root)
        except FileNotFoundError:
            continue
        built[name] = lock_id(name)
    return built


def missing(*, root: Path) -> list[str]:
    """Return the committed environments whose current lock is not built under ``root``."""

    names = {lock.parent.name for lock in [*SPECS.glob("*/uv.lock"), *SPECS.glob("*/pixi.lock")]}
    return sorted(names - set(inventory(root=root)))


def overlay(name: str, extras: Sequence[str], *, root: Path) -> Path:
    """Return an interpreter with ``extras`` added to the built environment, never moving a pin.

    The overlay is a hardlinked copy of the built prefix (seconds, almost no disk) with the
    extras installed from wheels only (``--only-binary :all:``) under the base environment's
    pins as constraints, so an extra that needs a different version of a locked package fails
    to install instead of changing it. Raises ``ExtrasError``: ``missing_dependency:sdist`` when
    a requirement has no wheel, else ``missing_dependency``. Overlays are kept under
    ``<root>/overlays/`` per lock and extras set until ``prune_overlays`` removes them.
    """

    from menagerie.pipeline import sdist

    base = python(name, root=root)
    pins = subprocess.run(["uv", "pip", "freeze", "--python", str(base)], check=True,
                          capture_output=True, text=True, env=_environ()).stdout
    wheels = sdist.wheels(name, list(extras), root=root,
                          protected=[line for line in pins.splitlines() if line and not line.startswith("#")])
    wheel_hashes = [sdist.digest(w) for w in wheels]
    key = hashlib.sha256("\n".join([*sorted(extras), *wheel_hashes]).encode()).hexdigest()[:8]
    target = Path(root) / "overlays" / f"{lock_id(name).replace('@', '-')}-{key}"
    if (target / BUILT_MARKER).is_file():
        os.utime(target / BUILT_MARKER)  # the marker's mtime is the overlay's last use (prune_overlays)
        return target / "bin" / "python"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-al", str(prefix(name, root=root)), str(target)], check=True)
    (target / BUILT_MARKER).unlink()
    constraints = target / ".base-pins.txt"
    constraints.write_text(pins)
    with overlay_cache.locked(root):  # a trim never removes an archive mid-install
        done = subprocess.run(["uv", "pip", "install", "--python", str(target / "bin" / "python"),
                               "--cache-dir", str(overlay_cache.cache_dir(root)),
                               "--only-binary", ":all:", "--constraint", str(constraints),
                               *[str(wheel) for wheel in wheels],
                               *sorted(extras)], check=False, capture_output=True, text=True,
                              env=_environ(), timeout=INSTALL_TIMEOUT_S)
    if done.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        err = done.stderr.strip()
        cls = "missing_dependency:sdist" if _SDIST_ONLY.search(err) else "missing_dependency"
        cause = err[err.find("error:"):] if "error:" in err else err  # rows are committed
        cause = re.sub(r"/(?:home|Users)/[^/\s]+/", "~/", cause)[:400]
        raise ExtrasError(cls, f"extras {sorted(extras)} did not install: {cause}")
    (target / BUILT_MARKER).write_text(f"{lock_id(name)} + {' '.join(sorted(extras))}\n")
    return target / "bin" / "python"


def prune_overlays(root: Path, *, keep_days: float, cap_gib: float, now: float | None = None,
                   keep_for: Iterable[Path] = ()) -> dict[str, int]:
    """Remove overlays unused for ``keep_days``, then the least recently used until the rest
    hold no more than ``cap_gib``, then the overlay cache's wheels no overlay links any more; an
    overlay is rebuilt (fetching its wheels again if trimmed) on its next use.

    Last use is the marker's mtime (``overlay`` touches it on reuse); an overlay with no marker
    is a half-built leftover. Nothing used or started within ``OVERLAY_IN_USE_S`` is removed, so
    a running job keeps its interpreter, and nothing a spool request in ``keep_for`` names. An
    overlay's size is what it holds beyond its base prefix (``overlay_cache.cost``): ``cp -al``
    shares every base file, and its extras are freed with the cache trim.
    """

    now = time.time() if now is None else now
    needed = overlay_cache.needed_markers(keep_for, lock_id)
    report = dict.fromkeys(("overlays", "removed_age", "removed_halfbuilt", "removed_cap",
                            "kept", "kept_needed", "kept_bytes", "freed_bytes"), 0)
    kept, bases = [], {}
    for used, path, built in _overlays(Path(root) / "overlays"):
        report["overlays"] += 1
        idle = now - used
        if built and (path / BUILT_MARKER).read_text().strip() in needed:
            report["kept_needed"] += 1  # outside the cap too
        elif idle >= OVERLAY_IN_USE_S and (not built or idle > keep_days * 86400):
            report["freed_bytes"] += _remove_overlay(path)
            report["removed_age" if built else "removed_halfbuilt"] += 1
        else:
            kept.append((used, path, _overlay_bytes(path, bases) if built else 0))  # never walk a build
    removed, freed, total = _evict_to_cap(kept, int(cap_gib * 2**30), now)
    report.update(removed_cap=removed, kept=len(kept) - removed, kept_bytes=total)
    trimmed = overlay_cache.trim(root, env=_environ())
    report.update(cache_archives_removed=trimmed["archives_removed"], cache_busy=trimmed["busy"],
                  cache_packages_cleaned=trimmed["packages_cleaned"], cache_uv_failed=trimmed["uv_failed"])
    report["freed_bytes"] += freed + trimmed["freed_bytes"]
    return report


def _evict_to_cap(kept: list[tuple[float, Path, int]], cap: int, now: float) -> tuple[int, int, int]:
    """Remove the least recently used overlays not in use until the rest total ``cap`` bytes or
    less; return (removed, bytes freed, bytes kept)."""

    removed = freed = 0
    total = sum(size for _, _, size in kept)
    for used, path, size in sorted(kept):
        if total <= cap:
            break
        if now - used >= OVERLAY_IN_USE_S:
            total -= size
            freed += _remove_overlay(path)
            removed += 1
    return removed, freed, total


def _overlays(parent: Path) -> list[tuple[float, Path, bool]]:
    """Every overlay under ``parent`` as (last use, path, built)."""

    found = []
    for path in sorted(parent.iterdir()) if parent.is_dir() else []:
        if path.is_dir() and not path.is_symlink():
            marker = path / BUILT_MARKER
            built = marker.is_file()
            found.append(((marker if built else path).stat().st_mtime, path, built))
    return found


def _overlay_bytes(path: Path, bases: dict | None = None) -> int:
    """What the overlay holds beyond its base, cached in the overlay after the first count."""

    cached = path / OVERLAY_COST
    try:
        return int(cached.read_text())
    except (OSError, ValueError):
        size = overlay_cache.cost(path, bases)
    cached.write_text(str(size))
    return size


def _unique_bytes(path: Path) -> int:
    total = 0
    for top, _dirs, files in os.walk(path):
        for name in files:
            try:
                info = os.lstat(os.path.join(top, name))
            except FileNotFoundError:  # removed while counting
                continue
            total += info.st_blocks * 512 if info.st_nlink == 1 else 0
    return total


def _remove_overlay(path: Path) -> int:
    """Remove the overlay; return what that alone frees (its extras go with the cache trim)."""

    size = _unique_bytes(path)
    shutil.rmtree(path)
    return size


def _environ() -> dict[str, str]:
    # uv must never be steered into an unrelated project or virtualenv by the caller's shell.
    drop = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "CONDA_PREFIX", "PYTHONPATH")
    return {k: v for k, v in os.environ.items() if k not in drop}
