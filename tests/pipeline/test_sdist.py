"""Saved source builds are offline and their cached wheels are hash checked."""

from __future__ import annotations

import io
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from menagerie.pipeline import envs, sdist


def _archive(path: Path, files: dict[str, str]) -> Path:
    with tarfile.open(path, 'w:gz') as output:
        for name, text in files.items():
            content = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
    return path


def test_extract_refuses_escape_and_links(tmp_path: Path) -> None:
    escape = _archive(tmp_path / 'escape.tar.gz', {'../escaped': 'no'})
    with pytest.raises(tarfile.FilterError):
        sdist._unpack(escape, tmp_path / 'source')
    assert not (tmp_path / 'escaped').exists()
    with tarfile.open(tmp_path / 'link.tar.gz', 'w:gz') as output:
        member = tarfile.TarInfo('link')
        member.type, member.linkname = tarfile.SYMTYPE, '/etc/passwd'
        output.addfile(member)
    with pytest.raises(envs.ExtrasError, match='links'):
        sdist._unpack(tmp_path / 'link.tar.gz', tmp_path / 'source')


def test_cache_refuses_tampered_wheels(tmp_path: Path) -> None:
    cache = tmp_path / 'sdist-wheels' / envs.lock_id('torch-main').replace('@', '-') / 'digest'
    cache.mkdir(parents=True)
    (cache / 'sample.whl').write_bytes(b'changed')
    (cache / 'manifest.json').write_text(json.dumps({'wheel': 'sample.whl', 'wheel_sha256': 'bad', 'distribution': 'sample',
                                                   'version': '1.0', 'base_lock': envs.lock_id('torch-main')}))
    with pytest.raises(envs.ExtrasError, match='hash differs'):
        sdist.wheels('torch-main', ['sample'], root=tmp_path)


@pytest.mark.skipif(not shutil.which('bwrap'), reason='requires the source-build sandbox')
def test_saved_source_build_is_offline_and_credentials_are_absent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('build')
    pytest.importorskip('setuptools')
    monkeypatch.setenv('RESIDUAL_TEST_CREDENTIAL', 'must-not-be-inherited')
    sentinel = tmp_path / 'host-side-marker'
    sentinel.write_text('preserved\n')
    setup = '''import os, socket
assert 'RESIDUAL_TEST_CREDENTIAL' not in os.environ
assert os.environ['HOME'] == '/tmp'
try:
    socket.create_connection(('1.1.1.1', 443), timeout=0.1)
except OSError:
    pass
else:
    raise RuntimeError('source build had network')
from pathlib import Path
wheels = Path(__file__).resolve().parents[2] / 'wheels'
wheels.mkdir(exist_ok=True)
manifest = wheels / 'manifest.json'
if not manifest.is_symlink():
    manifest.symlink_to(SENTINEL)
(wheels / 'injected.txt').write_text('untrusted extra')
from setuptools import setup
setup(name='residual-smoke', version='1.0', py_modules=['residual_smoke'])
'''.replace('SENTINEL', repr(str(sentinel)))
    source = _archive(tmp_path / 'smoke.tar.gz', {
        'smoke/setup.py': setup, 'smoke/residual_smoke.py': 'VALUE = 42\n',
        'smoke/pyproject.toml': '[build-system]\nrequires=["setuptools", "wheel"]\n'
                              'build-backend="setuptools.build_meta"\n'})
    wheel = sdist.build(source, name='torch-main', root=tmp_path / 'envs',
                        interpreter=Path(sys.executable), distribution='residual-smoke', version='1.0')
    assert sentinel.read_text() == 'preserved\n'
    assert {p.name for p in wheel.parent.iterdir()} == {wheel.name, 'manifest.json'}
    assert wheel.suffix == '.whl'
    assert sdist.wheels('torch-main', ['residual-smoke==1.0'], root=tmp_path / 'envs') == [wheel]
    assert sdist.build(source, name='torch-main', root=tmp_path / 'envs',
                       interpreter=Path(sys.executable), distribution='residual-smoke', version='1.0') == wheel


def _wheel(path: Path, name: str, version: str) -> Path:
    with zipfile.ZipFile(path, 'w') as output:
        output.writestr(f'{name}-{version}.dist-info/METADATA',
                        f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
    return path


def test_wheel_identity_cannot_claim_a_pinned_package(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path / 'torch-2.6.0-py3-none-any.whl', 'torch', '2.6.0')
    with pytest.raises(envs.ExtrasError, match='filename differs'):
        sdist._identity(wheel, 'residual-smoke', '1.0')
    lying = _wheel(tmp_path / 'residual_smoke-1.0-py3-none-any.whl', 'torch', '2.6.0')
    with pytest.raises(envs.ExtrasError, match='metadata differs'):
        sdist._identity(lying, 'residual-smoke', '1.0')


def test_cache_exposes_only_requested_unpinned_packages(tmp_path: Path) -> None:
    cache = tmp_path / 'sdist-wheels' / envs.lock_id('torch-main').replace('@', '-')
    for package in ('extra', 'torch', 'unrelated'):
        folder = cache / package
        folder.mkdir(parents=True)
        wheel = _wheel(folder / f'{package}-1.0-py3-none-any.whl', package, '1.0')
        (folder / 'manifest.json').write_text(json.dumps({
            'distribution': package, 'version': '1.0', 'base_lock': envs.lock_id('torch-main'),
            'wheel': wheel.name, 'wheel_sha256': sdist.digest(wheel)}))
    exposed = sdist.wheels('torch-main', ['extra==1.0', 'torch==1.0'], root=tmp_path,
                           protected=['torch==1.0'])
    assert [wheel.parent.name for wheel in exposed] == ['extra']


def test_old_archive_runtime_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _archive(tmp_path / 'source.tar.gz', {'setup.py': 'pass'})
    monkeypatch.delattr(tarfile, 'data_filter')
    with pytest.raises(envs.ExtrasError, match='data-filter support'):
        sdist._unpack(source, tmp_path / 'source')
