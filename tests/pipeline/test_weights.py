"""A recipe's optional ``weights``: a stored state dict loaded after the seeded random build.

Only a pretrained-weights recipe (decisions.md 2026-11-05, m7354) sets it; every other recipe's
serialized form and revision must stay exactly as before.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.pipeline import (
    DataLayout,
    StoreLayout,
    append_jsonl,
    month_of,
    read_json,
    recipe_revision,
)
from menagerie.pipeline import cluster, publish
from menagerie.pipeline.harvest_build import construct
from menagerie.pipeline.records import (
    ModelRecord,
    Recipe,
    RecordError,
    StoreIndexLine,
    from_dict,
    to_dict,
)

REPO = Path(__file__).parents[2]
DATA = DataLayout(REPO / "menagerie" / "data")
SHA = "ab" * 32
LINEAR = {"rev": "0" * 12, "kind": "library", "env": "torch-main", "backend": "torch",
          "input": {"preset": "image", "kind": "random-fallback", "shape": [1, 4]},
          "call": "torch.nn.Linear", "args": [4, 2], "kwargs": {}}


def _m7354() -> ModelRecord:
    return read_json(ModelRecord, DATA.model("m7354"))


def test_a_recipe_without_weights_serializes_and_revises_as_before() -> None:
    recipe = _m7354().recipe
    raw = to_dict(recipe)
    assert "weights" not in raw
    lock = (REPO / "envs" / "torch-main" / "uv.lock").read_bytes()
    assert recipe_revision(raw, None, lock) == recipe_revision({**raw, "weights": None}, None, lock)
    assert recipe_revision({**raw, "weights": SHA}, None, lock) != recipe_revision(raw, None, lock)


def test_weights_round_trip_and_must_be_a_sha() -> None:
    recipe = from_dict(Recipe, {**LINEAR, "weights": SHA})
    assert recipe.weights == SHA
    assert from_dict(Recipe, to_dict(recipe)) == recipe
    with pytest.raises(RecordError):
        from_dict(Recipe, {**LINEAR, "weights": "model.safetensors"})


def _stored_linear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, out: int = 2) -> str:
    import torch
    from safetensors.torch import save_file

    weights = {"weight": torch.arange(4 * out, dtype=torch.float32).reshape(out, 4),
               "bias": torch.full((out,), 7.0)}
    path = tmp_path / "model.safetensors"
    save_file(weights, str(path))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    store = StoreLayout(tmp_path / "store")
    store.blob(sha).parent.mkdir(parents=True, exist_ok=True)
    store.blob(sha).write_bytes(path.read_bytes())
    monkeypatch.setenv("MENAGERIE_STORE", str(store.root))
    return sha


def test_construct_loads_the_stored_weights_over_the_random_build(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    sha = _stored_linear(tmp_path, monkeypatch)
    model = construct(from_dict(Recipe, {**LINEAR, "weights": sha}))
    assert torch.equal(model.weight, torch.arange(8, dtype=torch.float32).reshape(2, 4))
    assert torch.equal(model.bias, torch.full((2,), 7.0))
    random = construct(from_dict(Recipe, LINEAR))
    assert not torch.equal(random.weight, model.weight)
    meta = construct(from_dict(Recipe, {**LINEAR, "weights": sha}), device="meta")
    assert meta.weight.is_meta  # a structure-only build never reads the blob


def test_weights_that_do_not_fit_the_model_are_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = _stored_linear(tmp_path, monkeypatch, out=3)
    with pytest.raises(RecordError, match="do not fit the model"):
        construct(from_dict(Recipe, {**LINEAR, "weights": sha}))


def test_the_site_recipe_line_names_the_pretrained_weights(tmp_path: Path) -> None:
    record = _m7354()
    plain = publish.recipe_text(record)
    assert "pretrained weights" not in plain
    pretrained = replace(record, recipe=replace(record.recipe, weights=SHA))
    data = DataLayout(tmp_path)
    url = "https://huggingface.co/timm/x/resolve/abc/model.safetensors"
    line = StoreIndexLine(sha256=SHA, bytes=1, mime="application/octet-stream",
                          fetched="2026-11-05T00:00:00Z", license_class="third-party", url=url)
    append_jsonl([line], data.store_index(month_of(line.fetched)))
    source = publish._weights_from(pretrained, SimpleNamespace(data=data))
    assert source == url
    assert publish.recipe_text(pretrained, source) == f"{plain}  # then loads pretrained weights: {url}"
    assert publish.recipe_text(pretrained).endswith(f"pretrained weights: sha256 {SHA}")
    assert publish._weights_from(record, SimpleNamespace(data=data)) is None


def test_a_cluster_batch_stages_the_weights_blob(tmp_path: Path) -> None:
    record = _m7354()
    pretrained = replace(record, recipe=replace(record.recipe, weights=SHA))
    assert SHA in cluster._copy_inputs([pretrained], DATA, tmp_path)
    assert SHA not in cluster._copy_inputs([record], DATA, tmp_path / "plain")


def test_the_bridge_drains_the_weights_blob_before_forwarding(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from menagerie.pipeline import bridge, write_json
    from menagerie.pipeline.records import Caps, RunRequest

    record = _m7354()
    recipe = replace(record.recipe, weights=SHA)
    request = RunRequest(request="m7354-forward-eval-x-0001", id="m7354", mode="forward",
                         recipe=recipe, caps=Caps(rss_mb=4096, timeout_s=600, threads=1))
    local = tmp_path / "hubspool"
    for name in ("new", "claimed", "done", "rejected"):
        (local / name).mkdir(parents=True)
    path = local / "new" / f"{request.request}.json"
    write_json(request, path)
    drained: list[str | None] = []
    monkeypatch.setattr(bridge, "_drain", drained.append)
    sent: list[str] = []
    link = SimpleNamespace(run=lambda command, *, data=None: sent.append(command) or b"")
    bridge._forward(link, path, local, "remote")
    assert SHA in drained and sent
    assert (local / "claimed" / path.name).is_file()
