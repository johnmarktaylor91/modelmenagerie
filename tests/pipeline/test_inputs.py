"""Input presets on the committed canonical assets (needs torch and Pillow)."""

from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from menagerie.pipeline.inputs import (  # noqa: E402
    ASSETS,
    asset_path,
    contract,
    make_input,
    presets,
)
from menagerie.pipeline.records import RecipeInput, RecordError  # noqa: E402


def test_assets_are_the_recorded_bytes_with_source_and_license() -> None:
    for asset_id, asset in ASSETS.items():
        assert hashlib.sha256(asset_path(asset_id).read_bytes()).hexdigest() == asset.sha256
        assert asset.source.startswith("https://") and "Public domain" in asset.license
    assert asset_path("std-image-01").read_bytes()[:3] == b"\xff\xd8\xff"  # a JPEG


@pytest.mark.parametrize("shape", [(1, 3, 224, 224), (2, 3, 256, 128), (1, 1, 28, 28),
                                   (3, 64, 64), (1, 4, 32, 32), (1, 3, 4, 32, 32)])
def test_image_preset_fills_every_layout(shape: tuple[int, ...]) -> None:
    spec = contract("image", shape)
    assert (spec.kind, spec.asset) == ("standard-image", "std-image-01")
    (image,), kwargs = make_input(spec)
    assert kwargs == {} and tuple(image.shape) == shape and image.dtype == torch.float32
    # ImageNet-normalized real pixels: varied, inside the normalized range of 0..255.
    assert image.std() > 0.3 and -2.2 < float(image.min()) < float(image.max()) < 2.7
    assert torch.equal(image, make_input(spec, seed=5)[0][0])  # the photo ignores the seed


def test_image_preset_is_the_photo_resized() -> None:
    from PIL import Image

    (image,), _ = make_input(contract("image", (1, 3, 768, 1024)))  # the photo's own size
    with Image.open(asset_path("std-image-01")) as raw:
        pixels = torch.tensor(bytearray(raw.convert("RGB").tobytes()), dtype=torch.uint8)
    rgb = pixels.view(768, 1024, 3).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    assert torch.allclose(image[0], (rgb - mean) / std, atol=1e-5)


def test_text_preset_is_the_prompt_as_byte_ids() -> None:
    spec = contract("text", (2, 16))
    assert (spec.kind, spec.asset) == ("standard-text", "std-text-01")
    (ids,), _ = make_input(spec)
    assert ids.dtype == torch.int64 and tuple(ids.shape) == (2, 16)
    prompt = asset_path("std-text-01").read_text(encoding="utf-8").strip()
    assert bytes((ids.flatten() - 1).tolist()).decode() == prompt[:32]


def test_random_presets_are_seeded_random_fallback() -> None:
    for preset, dtype in (("randn", torch.float32), ("randint:33", torch.int64)):
        spec = contract(preset, (1, 50))
        assert (spec.kind, spec.asset) == ("random-fallback", None)
        first, second = make_input(spec, seed=1)[0][0], make_input(spec, seed=1)[0][0]
        assert first.dtype == dtype and torch.equal(first, second)
        assert not torch.equal(first, make_input(spec, seed=2)[0][0])
    assert int(make_input(contract("randint:33", (1, 500)))[0][0].max()) < 33


def test_example_preset_calls_the_classic_under_the_seed() -> None:
    spec = contract("example:menagerie.classics.orig_lstm_1997:example_input", None)
    (x,), _ = make_input(spec, seed=3)
    assert tuple(x.shape) == (2, 3, 4) and torch.equal(x, make_input(spec, seed=3)[0][0])
    state = torch.random.get_rng_state()
    make_input(spec, seed=4)
    assert torch.equal(state, torch.random.get_rng_state())  # the caller's RNG is untouched


def test_bad_presets_are_refused() -> None:
    assert presets() == ["image", "text", "randn", "randint", "example"]
    for preset in ("noise", "randint", "randint:x", "example", "image:2"):
        with pytest.raises(RecordError):
            make_input(RecipeInput(preset=preset, kind="random-fallback", shape=(1, 2)))
    with pytest.raises(RecordError, match="C,H,W"):
        make_input(contract("image", (1, 16)))
