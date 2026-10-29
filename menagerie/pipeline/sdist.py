"""Build saved source archives offline; expose verified wheels to pinned overlays.

The caller provisions build/backend dependencies in the interpreter beforehand.
No source build runs with network, credentials, a writable prefix or host home.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from pathlib import Path

from menagerie import host_budget
from menagerie.pipeline import envs

TIMEOUT_S = 1800
SOURCE_BYTES = 1024 ** 3
SOURCE_FILES = 100_000


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        hashed = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hashed.update(chunk)
        return hashed.hexdigest()


def _trees(interpreter: Path) -> list[Path]:
    trees, link = [interpreter.parents[1]], interpreter
    while link.is_symlink():
        link = link.parent / os.readlink(link)
        trees.append(link.parents[1])
    return [*trees, interpreter.resolve().parents[1]]


def _command(work: Path, interpreter: Path, source: Path) -> list[str]:
    argv = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session',
            '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp', '--dir', '/etc']
    for name in ('/bin', '/sbin', '/lib', '/lib32', '/lib64', '/libx32'):
        if os.path.islink(name):
            argv += ['--symlink', os.readlink(name), name]
        elif os.path.isdir(name):
            argv += ['--ro-bind', name, name]
    for path in dict.fromkeys([Path('/usr'), *_trees(interpreter)]):
        argv += ['--ro-bind', str(path), str(path)]
    argv += ['--bind', str(work), str(work), '--chdir', str(work), str(interpreter),
             '-m', 'build', '--no-isolation', '--wheel', '--outdir', str(work / 'wheels'),
             str(source)]
    return argv


def _unpack(archive: Path, target: Path) -> Path:
    if not hasattr(tarfile, 'data_filter'):
        raise envs.ExtrasError('compiled_ext', 'saved source builds require tar data-filter support')
    with tarfile.open(archive) as saved:
        members = saved.getmembers()
        if len(members) > SOURCE_FILES or sum(m.size for m in members) > SOURCE_BYTES:
            raise envs.ExtrasError('compiled_ext', 'saved source archive exceeds build limits')
        if any(not (m.isfile() or m.isdir()) for m in members):
            raise envs.ExtrasError('compiled_ext', 'source archive has links or special files')
        saved.extractall(target, filter='data')
    children = list(target.iterdir())
    source = children[0] if len(children) == 1 and children[0].is_dir() else target
    if not (source / 'pyproject.toml').is_file() and not (source / 'setup.py').is_file():
        raise envs.ExtrasError('compiled_ext', 'saved archive has no Python build metadata')
    return source


def _identity(wheel: Path, distribution: str, version: str) -> None:
    """The artifact, metadata and independently requested identity must agree."""

    try:
        package, released, _, _ = parse_wheel_filename(wheel.name)
        if package != canonicalize_name(distribution) or released != Version(version):
            raise ValueError('wheel filename differs from expected distribution/version')
        with zipfile.ZipFile(wheel) as saved:
            metadata = [m for m in saved.infolist() if m.filename.endswith('.dist-info/METADATA')]
            if len(metadata) != 1 or metadata[0].file_size > 2 * 1024 * 1024:
                raise ValueError('wheel metadata missing, duplicated or oversized')
            info = BytesParser().parsebytes(saved.read(metadata[0]))
            if any(len(info.get_all(field, [])) != 1 for field in ('Name', 'Version')):
                raise ValueError('wheel metadata has ambiguous identity')
        if (canonicalize_name(info['Name'] or '') != package or
                Version(info['Version'] or '') != released):
            raise ValueError('wheel metadata differs from filename')
    except (ValueError, zipfile.BadZipFile, KeyError) as exc:
        raise envs.ExtrasError('compiled_ext', f'invalid wheel identity: {exc}') from exc


def _requirements(values: list[str]) -> list[Requirement]:
    """Reject unsupported freeze records at the installation boundary."""

    try:
        return [Requirement(value) for value in values]
    except InvalidRequirement as exc:
        raise envs.ExtrasError('compiled_ext', 'unsupported requirement or base pin record') from exc


def wheels(name: str, extras: list[str], *, root: Path,
           protected: list[str] = ()) -> list[Path]:
    """Select verified wheels for requested extras, excluding installed base packages."""

    requested = {canonicalize_name(r.name): r for r in _requirements(extras)}
    pinned = {canonicalize_name(r.name) for r in _requirements(protected)}
    cache = Path(root) / 'sdist-wheels' / envs.lock_id(name).replace('@', '-')
    result = []
    for manifest in sorted(cache.glob('*/manifest.json')):
        try:
            info = json.loads(manifest.read_text())
            package = canonicalize_name(info['distribution'])
            if package not in requested or package in pinned:
                continue
            wheel = manifest.parent / info['wheel']
            if (wheel.parent != manifest.parent or wheel.suffix != '.whl' or wheel.is_symlink()
                    or info['base_lock'] != envs.lock_id(name)):
                raise ValueError('invalid cached wheel path or base lock')
            if digest(wheel) != info['wheel_sha256']:
                raise ValueError('cached wheel hash differs from build manifest')
            _identity(wheel, info['distribution'], info['version'])
            if requested[package].specifier.contains(info['version'], prereleases=True):
                result.append(wheel)
        except (OSError, KeyError, ValueError) as exc:
            raise envs.ExtrasError('compiled_ext', f'invalid cached wheel: {exc}') from exc
    return result


def _run_build(work: Path, interpreter: Path, source: Path, jobs: int,
               distribution: str, version: str) -> Path:
    """Run the sandbox and validate its sole wheel before publishing anything."""

    result = subprocess.run(_command(work, interpreter, source), check=False,
                            capture_output=True, timeout=TIMEOUT_S,
                            env={'PATH': '/usr/bin:/bin', 'HOME': '/tmp',
                                 'LANG': 'C.UTF-8', 'TMPDIR': '/tmp',
                                 'PYTHONDONTWRITEBYTECODE': '1',
                                 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                 'MAX_JOBS': str(jobs), 'CMAKE_BUILD_PARALLEL_LEVEL': str(jobs)})
    output = work / 'wheels'
    if output.is_symlink() or not output.resolve().is_relative_to(work.resolve()):
        raise envs.ExtrasError('compiled_ext', 'source build replaced the wheel directory')
    found = list(output.glob('*.whl'))
    if result.returncode or len(found) != 1:
        raise envs.ExtrasError('compiled_ext',
                              f'offline source build failed (exit {result.returncode})')
    if found[0].is_symlink() or not found[0].is_file():
        raise envs.ExtrasError('compiled_ext', 'source build produced an unsafe wheel path')
    _identity(found[0], distribution, version)
    return found[0]


def build(archive: Path, *, name: str, root: Path, interpreter: Path,
          distribution: str, version: str) -> Path:
    """Build one saved tar sdist, then atomically cache its wheel and provenance.

    Build frontends/backends are preinstalled, never bootstrapped by source code.
    A failure remains compiled_ext; callers do not retry outside the sandbox.
    """

    archive, interpreter = archive.resolve(), interpreter.absolute()
    if shutil.which('bwrap') is None:
        raise envs.ExtrasError('compiled_ext', 'source builds require bubblewrap')
    sha = digest(archive)
    # The containing run declares its cap; never let an extension builder choose nproc.
    jobs = host_budget.declared_cpus(affinity=1, total=1)
    cache = Path(root).absolute() / 'sdist-wheels' / envs.lock_id(name).replace('@', '-')
    cache.mkdir(parents=True, exist_ok=True)
    existing = cache / sha
    if (existing / 'manifest.json').is_file():
        valid = wheels(name, [f'{distribution}=={version}'], root=root)
        matching = [wheel for wheel in valid if wheel.parent == existing]
        if not matching:
            raise envs.ExtrasError('compiled_ext', 'cached source belongs to a different distribution')
        return matching[0]
    with tempfile.TemporaryDirectory(prefix='.build-', dir=cache) as temporary:
        work = Path(temporary)
        try:
            source = _unpack(archive, work / 'source')
            wheel = _run_build(work, interpreter, source, jobs, distribution, version)
        except (subprocess.TimeoutExpired, OSError, tarfile.TarError) as exc:
            raise envs.ExtrasError('compiled_ext', f'source build failed: {type(exc).__name__}') from exc
        _publish(wheel, existing, name, sha, distribution, version)
    return existing / wheel.name


def _publish(wheel: Path, target: Path, name: str, sha: str,
             distribution: str, version: str) -> None:
    """Publish only copied, revalidated bytes into a fresh host-owned directory."""

    try:
        with tempfile.TemporaryDirectory(prefix='.publish-', dir=target.parent) as temporary:
            output = Path(temporary)
            copied = output / wheel.name
            descriptor = os.open(wheel, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, 'rb') as source, copied.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            _identity(copied, distribution, version)
            info = {'schema': 'sdist-wheel/1', 'source_sha256': sha,
                    'base_lock': envs.lock_id(name), 'distribution': distribution,
                    'version': version, 'wheel': copied.name, 'wheel_sha256': digest(copied)}
            descriptor = os.open(output / 'manifest.json',
                                 os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, 'w') as manifest:
                manifest.write(json.dumps(info, indent=2) + '\n')
            output.rename(target)
    except OSError as exc:
        raise envs.ExtrasError('compiled_ext', 'wheel cache publication failed') from exc
