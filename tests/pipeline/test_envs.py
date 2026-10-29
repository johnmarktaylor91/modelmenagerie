"""The committed locks (``torch-main``, the native ``tf``, ``jax``, ``paddle`` and ``mlx`` and the
CUDA ``torch-gpu``), and their build and canary on the worker.

The lock tests read only committed files. The build tests need ``MENAGERIE_ENVS`` and run on the
worker, where a rebuild of an already built lock returns at once.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

from menagerie.pipeline import envs

ROOT = os.environ.get("MENAGERIE_ENVS")
PROMISED = {
    "torch": "2.6.0+cpu", "torchlens": "2.36.0", "timm": "1.0.30", "torchvision": "0.21.0+cpu",
    "transformers": None, "segmentation-models-pytorch": None, "ultralytics": None,
    "einops": None, "diffusers": None, "pillow": None, "psutil": None, "pyyaml": None,
    "pyvips": "3.2.0", "pyvips-binary": "8.18.7",
}


def test_lock_holds_every_promised_package() -> None:
    for package, version in PROMISED.items():
        locked = envs.package_version("torch-main", package)
        assert locked is not None, f"{package} is not in the torch-main lock"
        if version:
            assert locked == version
    text = envs.lock_path("torch-main").read_text()
    assert re.search(r'^requires-python = "==3\.11\.\*"', text, re.MULTILINE)
    assert "nvidia-cudnn" not in text and "+cu" not in text, "the lock must stay CPU-only"


def test_lock_id_names_the_exact_lock() -> None:
    digest = hashlib.sha256(envs.lock_path("torch-main").read_bytes()).hexdigest()
    assert envs.lock_id("torch-main") == f"torch-main@{digest[:8]}"
    assert envs.prefix("torch-main", root=Path("/r")) == Path(f"/r/torch-main-{digest[:8]}")
    with pytest.raises(FileNotFoundError):
        envs.lock_id("no-such-env")


@pytest.mark.skipif(not ROOT, reason="needs MENAGERIE_ENVS on the worker")
def test_build_canary_and_inventory_on_the_worker() -> None:
    root = Path(ROOT)
    prefix = envs.build("torch-main", root=root)
    assert prefix == envs.prefix("torch-main", root=root)
    assert envs.python("torch-main", root=root).is_file()
    assert envs.canary("torch-main", root=root)
    assert envs.inventory(root=root)["torch-main"] == envs.lock_id("torch-main")
    done = subprocess.run(["uv", "cache", "dir"], check=True, capture_output=True, text=True)
    cache = Path(done.stdout.strip()).resolve()
    assert not cache.is_relative_to(Path.home() / "data"), f"uv cache under ~/data: {cache}"


# Each native environment: its framework at the TorchLens extra's range, CPU torch 2.6 for the
# worker's records, TorchLens at the pin. torch-gpu: CUDA 12.4 torch with the extension builders.
NATIVE = {
    "tf": {"tensorflow": r"2\.(1[6-9]|[2-9]\d)\..*", "keras": r"3\..*"},
    "jax": {"jax": r".+", "jaxlib": r".+", "flax": r".+", "equinox": r".+"},
    "paddle": {"paddlepaddle": r"3\.3\..*", "setuptools": r".+"},
    "mlx": {"mlx": r"0\.26\.(?!5$).*"},
}


@pytest.mark.parametrize("name", sorted(NATIVE))
def test_native_locks_hold_their_framework(name: str) -> None:
    for package, version in {**NATIVE[name], "torch": r"2\.6\.\d\+cpu",
                             "torchlens": r"2\.36\.0", "pyvips": r"3\.2\.0"}.items():
        locked = envs.package_version(name, package)
        assert locked is not None and re.fullmatch(version, locked), (name, package, locked)
    text = envs.lock_path(name).read_text()
    assert "nvidia-cudnn" not in text and "+cu" not in text, f"{name} must stay CPU-only"
    assert (envs.SPECS / name / "canary.py").is_file()


def test_torch_gpu_lock_is_cuda_12_4() -> None:
    assert envs.package_version("torch-gpu", "torch") == "2.6.0+cu124"
    assert envs.package_version("torch-gpu", "torchlens") == "2.36.0"
    for package in ("ninja", "setuptools", "timm", "transformers"):
        assert envs.package_version("torch-gpu", package), package


def test_a_failing_canary_leaves_the_build_unmarked(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        ran.append(command)
        if command[0] == "uv":  # the sync makes the prefix and its interpreter
            target = Path(str(kwargs["env"]["UV_PROJECT_ENVIRONMENT"]))  # type: ignore[index]
            (target / "bin").mkdir(parents=True)
            (target / "bin" / "python").write_text("")
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 1 if verdict["fail"] else 0)

    verdict = {"fail": True}
    monkeypatch.setattr(envs.subprocess, "run", fake_run)
    with pytest.raises(envs.CanaryError):
        envs.build("tf", root=tmp_path)
    assert not (envs.prefix("tf", root=tmp_path) / envs.BUILT_MARKER).exists()
    assert ran[-1][-1].endswith("envs/tf/canary.py")
    verdict["fail"] = False
    import shutil

    shutil.rmtree(envs.prefix("tf", root=tmp_path))
    assert envs.build("tf", root=tmp_path) == envs.prefix("tf", root=tmp_path)
    assert envs.python("tf", root=tmp_path).is_file()
    assert envs.inventory(root=tmp_path) == {"tf": envs.lock_id("tf")}


def test_pixi_lock_identifies_conda_and_pypi_packages(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    specs = tmp_path / 'specs' / 'mixed'
    specs.mkdir(parents=True)
    (specs / 'pixi.toml').write_text('[workspace]\nname="mixed"\n')
    (specs / 'pixi.lock').write_text('version: 6\npackages:\n'
        '- conda: https://example.invalid/numpy-1.26.4-py311h64a7726_0.conda\n'
        '- pypi: https://example.invalid/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl\n')
    monkeypatch.setattr(envs, 'SPECS', specs.parent)
    assert envs.package_version('mixed', 'numpy') == '1.26.4'
    assert envs.package_version('mixed', 'torch') == '2.6.0+cpu'
    assert envs.package_version('mixed', 'absent') is None
    assert envs.prefix('mixed', root=tmp_path).parts[-3:] == ('.pixi', 'envs', 'default')


def test_pixi_build_requires_canary_and_isolated_configuration(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    specs = tmp_path / 'specs' / 'mixed'
    specs.mkdir(parents=True)
    (specs / 'pixi.toml').write_text('[workspace]\nname="mixed"\n')
    (specs / 'pixi.lock').write_text('version: 6\npackages: []\n')
    monkeypatch.setattr(envs, 'SPECS', specs.parent)
    monkeypatch.setenv('PIXI_CONFIG_FILE', '/unrelated/config.toml')
    verdict = {'fail': True}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[0] == 'pixi':
            assert '--locked' in command
            config = kwargs['env']
            assert config['PIXI_NO_CONFIG'] == 'true' and 'PIXI_CONFIG_FILE' not in config
            target = envs.prefix('mixed', root=tmp_path / 'built')
            (target / 'bin').mkdir(parents=True, exist_ok=True)
            (target / 'bin' / 'python').write_text('')
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 1 if verdict['fail'] else 0)

    monkeypatch.setattr(envs.subprocess, 'run', fake_run)
    with pytest.raises(envs.CanaryError):
        envs.build('mixed', root=tmp_path / 'built')
    with pytest.raises(FileNotFoundError):
        envs.python('mixed', root=tmp_path / 'built')
    verdict['fail'] = False
    envs.build('mixed', root=tmp_path / 'built')
    assert envs.inventory(root=tmp_path / 'built') == {'mixed': envs.lock_id('mixed')}


def test_conda_pytorch_reports_the_torch_distribution(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    specs = tmp_path / 'specs' / 'mixed'
    specs.mkdir(parents=True)
    (specs / 'pixi.toml').write_text('[workspace]\nname="mixed"\n')
    (specs / 'pixi.lock').write_text('version: 6\npackages:\n'
        '- conda: https://example.invalid/pytorch-2.6.0-cpu_py311_0.conda\n')
    monkeypatch.setattr(envs, 'SPECS', specs.parent)
    assert envs.package_version('mixed', 'torch') == '2.6.0'


def test_overlay_installs_the_verified_wheel_explicitly(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import sdist

    base = tmp_path / 'base'
    (base / 'bin').mkdir(parents=True)
    (base / 'bin' / 'python').write_text('')
    (base / envs.BUILT_MARKER).write_text('built')
    wheel = tmp_path / 'extra-1.0-py3-none-any.whl'
    wheel.write_bytes(b'verified artifact')
    monkeypatch.setattr(envs, 'python', lambda *a, **k: base / 'bin' / 'python')
    monkeypatch.setattr(envs, 'prefix', lambda *a, **k: base)
    monkeypatch.setattr(sdist, 'wheels', lambda *a, **k: [wheel])
    original = envs.subprocess.run
    installs = []

    def run(command, **kwargs):
        if command[0] == 'cp':
            return original(command, **kwargs)
        if 'freeze' in command:
            return subprocess.CompletedProcess(command, 0, 'torch==2.6.0\n', '')
        installs.append(command)
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(envs.subprocess, 'run', run)
    envs.overlay('torch-main', ['extra>=1.0'], root=tmp_path)
    assert str(wheel) in installs[0]
    assert '--find-links' not in installs[0]
    # the overlays' own cache, never the shared ~/.cache/uv, so prune-overlays can free the extras
    assert installs[0][installs[0].index('--cache-dir') + 1] == str(tmp_path / 'overlay-uv-cache')


def test_missing_names_every_committed_lock_not_built_at_its_lock(tmp_path: Path) -> None:
    locks = [*envs.SPECS.glob("*/uv.lock"), *envs.SPECS.glob("*/pixi.lock")]
    committed = {lock.parent.name for lock in locks}
    assert envs.missing(root=tmp_path) == sorted(committed)
    for name in ("tf", "detectron2"):  # a uv and a pixi prefix
        target = envs.prefix(name, root=tmp_path)
        target.mkdir(parents=True)
        (target / envs.BUILT_MARKER).write_text(envs.lock_id(name) + "\n")
    (tmp_path / "jax-00000000").mkdir()  # a superseded revision does not count
    assert envs.missing(root=tmp_path) == sorted(committed - {"tf", "detectron2"})
