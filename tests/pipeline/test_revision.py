"""Recipe revisions hash only trace-relevant packages; the renderer's packages move renderer_rev.

Edits are made to the committed ``torch-main`` lock as text, so the walk is checked on the real
dependency graph: the renderer's packages (``NON_TRACE_ROOTS``) and what only they reach are
left out of every revision, while library versions remain in it and TorchLens has separate trace currency.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pytest

from menagerie.pipeline import NON_TRACE_ROOTS, read_json, recipe_revision, trace_lock_digest
from menagerie.pipeline.records import ModelRecord, to_dict

REPO = Path(__file__).parents[2]
LOCK = (REPO / "envs" / "torch-main" / "uv.lock").read_text(encoding="utf-8")
RECIPES = [to_dict(r.recipe) for r in (read_json(ModelRecord, p) for p in
           sorted((REPO / "menagerie" / "data" / "models").rglob("*.json"))) if r.recipe]
HEADER, *BLOCKS = LOCK.split("\n[[package]]\n")


@pytest.fixture(autouse=True)
def reuse_identical_lock_walks(monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie import pipeline

    # Every recipe still runs through recipe_revision; only the same immutable
    # lock's dependency walk is reused within a test. Direct digest tests stay uncached.
    monkeypatch.setattr(pipeline, "trace_lock_digest", lru_cache(maxsize=None)(trace_lock_digest))


def _lock(blocks: list[str]) -> bytes:
    return "\n[[package]]\n".join([HEADER, *blocks]).encode()


def _name(block: str) -> str:
    return re.match(r'name = "([^"]+)"', block).group(1)


def _block(name: str) -> str:
    return next(b for b in BLOCKS if _name(b) == name)


def _replace(old: str, new: str) -> bytes:
    assert LOCK.count(old) == 1, old
    return LOCK.replace(old, new).encode()


def _version(name: str, version: str) -> bytes:
    found = re.search(rf'name = "{re.escape(name)}"\nversion = "([^"]+)"', LOCK)
    return _replace(found.group(0), f'name = "{name}"\nversion = "{version}"')


def _revisions(lock: bytes) -> list[str]:
    return [recipe_revision(recipe, None, lock) for recipe in RECIPES]


def test_the_renderer_packages_are_in_the_lock_as_non_trace_roots() -> None:
    names = {_name(b) for b in BLOCKS}
    assert set(NON_TRACE_ROOTS) <= names
    assert {"torch", "torchlens", "timm"} <= names
    assert len(RECIPES) > 200


def test_non_trace_packages_leave_every_revision_unchanged() -> None:
    base = _revisions(LOCK.encode())
    without = [b for b in BLOCKS if _name(b) not in (*NON_TRACE_ROOTS, "cffi", "pycparser")]
    root = next(b for b in without if 'source = { virtual = "." }' in b)
    stripped = re.sub(r'    \{ name = "pyvips[^\n]*\n', "", root.split("[package.metadata]")[0])
    removed = [b if b is not root else root.replace(root.split("[package.metadata]")[0], stripped)
               for b in without]
    assert _lock(removed) != LOCK.encode()
    pyvips = _block("pyvips")
    added_block = pyvips.replace("dependencies = [\n",
                                 'dependencies = [\n    { name = "fakedep" },\n')
    added = [*(added_block if b is pyvips else b for b in BLOCKS),
             'name = "fakedep"\nversion = "1.0"\n'
             'source = { registry = "https://pypi.org/simple" }\n']
    for lock in (_lock(removed), _lock(added), _version("pyvips", "3.2.1"),
                 _version("pyvips-binary", "8.19.0"), _version("cffi", "2.2.0")):
        assert _revisions(lock) == base


@pytest.mark.parametrize("name", ["timm", "torch"])
def test_trace_packages_change_every_revision(name: str) -> None:
    base = _revisions(LOCK.encode())
    changed = _revisions(_version(name, "99.0.0"))
    assert all(a != b for a, b in zip(base, changed))


def test_torchlens_only_repin_preserves_existing_recipe_keys() -> None:
    baseline = _revisions(_version("torchlens", "2.35.3"))
    assert _revisions(_version("torchlens", "2.36.0")) == baseline


def test_a_package_shared_with_a_trace_root_stays_in() -> None:
    torchlens = _block("torchlens")
    shared = torchlens.replace("dependencies = [\n", 'dependencies = [\n    { name = "cffi" },\n')
    lock = _lock([shared if b is torchlens else b for b in BLOCKS])
    assert trace_lock_digest(lock) != trace_lock_digest(LOCK.encode())
    bumped = lock.replace(b'name = "cffi"\nversion = "2.1.1"', b'name = "cffi"\nversion = "2.2.0"')
    assert trace_lock_digest(bumped) != trace_lock_digest(lock)


def test_the_digest_ignores_lock_ordering_and_hashes() -> None:
    base = trace_lock_digest(LOCK.encode())
    assert trace_lock_digest(_lock(BLOCKS[::-1])) == base
    rehashed = re.sub(r'hash = "sha256:[0-9a-f]{4}', 'hash = "sha256:0000', LOCK)
    assert trace_lock_digest(rehashed.encode()) == base


def test_renderer_packages_move_renderer_rev(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pyvips")
    import importlib.metadata

    from menagerie.pipeline import render

    monkeypatch.setattr(render, "_graphviz_version", lambda: "2.43.0")
    base = render.renderer_rev()
    assert re.fullmatch(r"graphviz 2\.43\.0\+pyvips \S+\+libvips \S+\+render-opts@[0-9a-f]{8}",
                        base)
    real = importlib.metadata.version
    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version",
                      lambda name: "8.19.0" if name == "pyvips-binary" else real(name))
        assert render.renderer_rev() != base
    with monkeypatch.context() as patch:
        patch.setattr(render, "EXTRAS", (*render.EXTRAS, "cffi"))
        assert render.renderer_rev() != base
    assert render.renderer_rev() == base


# --- Pixi locks: the same contract over a YAML lock with no root project -------------------

PIXI = (REPO / "tests" / "pipeline" / "fixtures" / "small.pixi.lock").read_text(encoding="utf-8")
# The fixture's trace pairs: pyvips, pyvips-binary and what only they reach (cffi, libvips) are
# out, an entry outside the default environment (stray) is out, TorchLens sits at its anchor.
PIXI_PAIRS = "numpy==1.26.4\npython==3.11.14\npytorch==2.6.0\ntorchlens==2.35.3"
PIXI_DIGEST = "3f8613238a9ad06f288b89db5a59e30d5b256e0e8215db450f2e06d63a324e02"


def _pixi(old: str, new: str, *, count: int = 1) -> bytes:
    assert PIXI.count(old) == count, old
    return PIXI.replace(old, new).encode()


def test_a_uv_digest_is_unchanged() -> None:
    import hashlib

    uv = ('version = 1\nrevision = 5\nrequires-python = "==3.11.*"\n\n'
          '[[package]]\nname = "env"\nversion = "0.0.0"\nsource = { virtual = "." }\n'
          'dependencies = [\n    { name = "pyvips" },\n    { name = "torch" },\n'
          '    { name = "torchlens" },\n]\n\n'
          '[[package]]\nname = "cffi"\nversion = "2.1.1"\n'
          'source = { registry = "https://pypi.org/simple" }\n\n'
          '[[package]]\nname = "pyvips"\nversion = "3.2.0"\n'
          'source = { registry = "https://pypi.org/simple" }\n'
          'dependencies = [\n    { name = "cffi" },\n]\n\n'
          '[[package]]\nname = "torch"\nversion = "2.6.0"\n'
          'source = { registry = "https://pypi.org/simple" }\n\n'
          '[[package]]\nname = "torchlens"\nversion = "2.36.0"\n'
          'source = { registry = "https://pypi.org/simple" }\n'
          'dependencies = [\n    { name = "torch" },\n]\n')
    pairs = "env==0.0.0\ntorch==2.6.0\ntorchlens==2.35.3"
    assert trace_lock_digest(uv.encode()) == hashlib.sha256(pairs.encode()).hexdigest()
    assert trace_lock_digest(uv.encode()) == (
        "ab99b6a21a2016753126865cde485309b415451efa4092ff9a34e18a6e987c18")


def test_a_pixi_lock_digests_its_trace_pairs() -> None:
    import hashlib

    assert hashlib.sha256(PIXI_PAIRS.encode()).hexdigest() == PIXI_DIGEST
    assert trace_lock_digest(PIXI.encode()) == PIXI_DIGEST


def test_a_pixi_digest_ignores_hashes_urls_and_ordering() -> None:
    import yaml

    rehashed = re.sub(r"sha256: [0-9a-f]{4}", "sha256: 0000", PIXI).encode()
    moved = _pixi("https://files.pythonhosted.org/packages/0e/4c/7610d3b5/",
                  "https://mirror.example/pypi/", count=3)
    data = yaml.safe_load(PIXI)
    data["packages"].reverse()
    data["environments"]["default"]["packages"]["linux-64"].reverse()
    reordered = yaml.safe_dump(data, sort_keys=False).encode()
    for lock in (rehashed, moved, reordered):
        assert trace_lock_digest(lock) == PIXI_DIGEST


@pytest.mark.parametrize("old,new", [
    ("pyvips-3.2.0-py311h5733de1_2", "pyvips-3.2.1-py311h5733de1_0"),
    ("libvips-8.18.6-h1a2b3c4_0", "libvips-8.19.0-h1a2b3c4_0"),
    ("cffi-2.1.1-py311h904a5e5_3", "cffi-2.2.0-py311h904a5e5_0"),
    ("version: 8.18.7", "version: 8.19.0"),
    ("version: 2.36.0", "version: 2.37.0"),
])
def test_pixi_non_trace_packages_and_torchlens_leave_the_digest(old: str, new: str) -> None:
    count = PIXI.count(old)
    assert count >= 1, old
    assert trace_lock_digest(_pixi(old, new, count=count)) == PIXI_DIGEST


@pytest.mark.parametrize("old,new", [
    ("pytorch-2.6.0-cpu_mkl_py311h1d7a3a5_100", "pytorch-2.7.0-cpu_mkl_py311h1d7a3a5_100"),
    ("numpy-1.26.4-py311h7125741_0", "numpy-1.26.5-py311h7125741_0"),
    ("python-3.11.14-h9e4cc4f_2_cpython", "python-3.11.15-h9e4cc4f_2_cpython"),
])
def test_pixi_trace_packages_change_the_digest(old: str, new: str) -> None:
    assert trace_lock_digest(_pixi(old, new, count=PIXI.count(old))) != PIXI_DIGEST


def test_a_pixi_package_shared_with_a_trace_package_stays_in() -> None:
    shared = _pixi("  - numpy>=1.26.4\n", "  - numpy>=1.26.4\n  - cffi>=1.0\n")
    assert trace_lock_digest(shared) != PIXI_DIGEST
    bumped = shared.replace(b"cffi-2.1.1-py311h904a5e5_3", b"cffi-2.2.0-py311h904a5e5_0")
    assert trace_lock_digest(bumped) != trace_lock_digest(shared)


def test_pypi_packages_keep_the_interpreter_in() -> None:
    import yaml

    data = yaml.safe_load(PIXI)
    for entry in data["packages"]:
        if "conda" in entry and "/pyvips-" not in entry["conda"]:
            entry["depends"] = []  # only pyvips and the PyPI wheels still lead to python
    alone = yaml.safe_dump(data, sort_keys=False)
    bumped = alone.replace("python-3.11.14-", "python-3.11.15-")
    assert trace_lock_digest(bumped.encode()) != trace_lock_digest(alone.encode())


def test_an_unreadable_pixi_lock_is_a_record_error() -> None:
    from menagerie.pipeline.records import RecordError

    with pytest.raises(RecordError):
        trace_lock_digest(b"version: 7\nplatforms: []\n")
    with pytest.raises(RecordError, match="version 6"):
        trace_lock_digest(_pixi("version: 7\n", "version: 6\n"))
    with pytest.raises(RecordError, match="matches no locked package"):
        moved = re.sub(r"(?m)^- (conda|pypi): ", "- \\1: https://elsewhere/", PIXI)
        trace_lock_digest(moved.encode())


@pytest.mark.parametrize("env", ["detectron2", "mmlab"])
def test_committed_pixi_environments_have_a_trace_digest(env: str) -> None:
    from menagerie.pipeline import envs

    path = envs.lock_path(env)
    assert path.name == "pixi.lock"
    lock = path.read_text(encoding="utf-8")
    base = trace_lock_digest(lock.encode())
    assert re.fullmatch(r"[0-9a-f]{64}", base)
    revisions = [recipe_revision({**r, "env": env}, None, lock.encode()) for r in RECIPES[:3]]
    assert all(re.fullmatch(r"[0-9a-f]{12}", rev) for rev in revisions)
    pyvips = re.search(r"/pyvips-(\d[^-/]*)-", lock).group(0)
    assert trace_lock_digest(lock.replace(pyvips, "/pyvips-99.0.0-").encode()) == base
    pypi = re.search(r"\n  name: torch\n  version: [^\n]+", lock)  # mmlab takes torch's wheel
    conda = re.search(r"/pytorch-\d[^-/]*-", lock)
    bumped = (lock.replace(pypi.group(0), "\n  name: torch\n  version: 99.0.0") if pypi
              else lock.replace(conda.group(0), "/pytorch-99.0.0-"))
    assert trace_lock_digest(bumped.encode()) != base
    python = re.search(r"/python-3\.[^-/]*-", lock).group(0)
    assert trace_lock_digest(lock.replace(python, "/python-3.99.0-").encode()) != base
