"""Model inputs: presets and the license-clean canonical assets, each with source and license.

Owned by the harvest lane. Rung 0 commits one public-domain image and one text prompt; every
other modality is ``random-fallback``.

A recipe's ``input.preset`` is a preset name, optionally with one argument after a colon:

- ``image``: the canonical photograph, resized and center-cropped to the shape's height and
  width, ImageNet-normalized float32 (``N,C,H,W``, ``C,H,W`` or ``N,C,T,H,W`` with C 1, 3 or 4).
  ``image:nhwc`` is the same picture channels-last (``N,H,W,C`` or ``H,W,C``), the layout most
  TensorFlow, Keras and JAX models read.
- ``text``: the canonical prompt's UTF-8 bytes plus one as int64 token ids, repeated to fill
  the shape. There is no tokenizer at random initialization, so ids run 1 to 256 and the
  model's vocabulary must hold at least 257 entries.
- ``randn``: seeded standard-normal float32.
- ``randint:<high>``: seeded int64 ids in ``[0, high)``.
- ``example:<module>:<function>``: the inputs a module's own example function returns
  (a classic's ``example_input``), called under the seed.

Inputs are built as torch tensors; ``native`` converts them for a model of another backend
(TensorFlow, JAX, Paddle, MLX), leaving values that are already native untouched.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from menagerie.pipeline.records import RecipeInput, RecordError

ASSET_DIR = Path(__file__).with_name("assets")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Asset:
    """A committed canonical input with where it came from and why we may use it."""

    file: str
    kind: str
    sha256: str
    source: str
    license: str
    note: str


ASSETS: dict[str, Asset] = {
    "std-image-01": Asset(
        file="std-image-01.jpg", kind="standard-image",
        sha256="7a7d156fcb6fe815670924619cf4bb2e310232a93e6b7fea64505d1dac486ec5",
        source="https://commons.wikimedia.org/wiki/File:Golden_Retriever_Dukedestiny01.jpg",
        license="Public domain (released by its author, Janneke Vreugdenhil; Commons: "
                "LicenseShortName 'Public domain', AttributionRequired false)",
        note="golden retriever photograph, 1024x768 JPEG, original bytes (Commons sha1 "
             "49016af022f65a34de894d0e6446f90a2c2ae338); resized and center-cropped, "
             "ImageNet-normalized",
    ),
    "std-text-01": Asset(
        file="std-text-01.txt", kind="standard-text",
        sha256="3cad24a80129246d96cae89342cff8e6ead9f064be35896a6eff1b2e1c4b64b2",
        source="https://www.gutenberg.org/ebooks/1342 (Jane Austen, Pride and Prejudice, "
               "1813, chapter 1, first sentence)",
        license="Public domain (published 1813)",
        note="first sentence of Pride and Prejudice; UTF-8 bytes plus one as token ids, "
             "no tokenizer",
    ),
}

PRESETS = ("image", "text", "randn", "randint", "example")
_PRESET_ASSET = {"image": "std-image-01", "text": "std-text-01"}


def presets() -> list[str]:
    """Return the preset names a recipe's ``input.preset`` may use."""

    return list(PRESETS)


def asset_path(asset_id: str) -> Path:
    """Return the committed file of a canonical asset (``std-image-01``, ...)."""

    if asset_id not in ASSETS:
        raise RecordError(f"unknown input asset {asset_id!r}; known: {sorted(ASSETS)}")
    return ASSET_DIR / ASSETS[asset_id].file


def contract(preset: str, shape: Sequence[int] | None, note: str = "") -> RecipeInput:
    """Return the input contract for a preset: its REQ 14 kind, asset and a one-line note."""

    name = _split(preset)[0]
    asset = _PRESET_ASSET.get(name)
    if asset:
        kind, note = ASSETS[asset].kind, note or ASSETS[asset].note
    else:
        kind = "random-fallback"
    return RecipeInput(preset=preset, kind=kind, shape=tuple(shape) if shape else None,
                       asset=asset, note=note)


def make_input(spec: RecipeInput, *, seed: int = 0) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build the model's positional and keyword inputs for a recipe's input contract."""

    import torch

    name, arg = _split(spec.preset)
    if name == "example":
        return _example(arg, seed)
    if spec.shape is None:
        raise RecordError(f"input preset {spec.preset!r} needs a shape")
    shape = tuple(spec.shape)
    generator = torch.Generator().manual_seed(seed)
    if name == "image" and arg == "nhwc":
        tensor = _channels_last(shape)
    elif name == "image":
        tensor = _image(shape)
    elif name == "text":
        tensor = _text(shape)
    elif name == "randn":
        tensor = torch.randn(shape, generator=generator)
    else:
        tensor = torch.randint(0, int(arg), shape, generator=generator)
    return (tensor,), {}


def _split(preset: str) -> tuple[str, str]:
    name, _, arg = preset.partition(":")
    if name not in PRESETS:
        raise RecordError(f"unknown input preset {preset!r}; known: {PRESETS}")
    if name == "image" and arg not in ("", "nhwc"):
        raise RecordError(f"input preset {preset!r}: image takes no argument or `nhwc`")
    if name != "image" and (name in ("randint", "example")) != bool(arg):
        raise RecordError(f"input preset {preset!r}: randint and example take one argument, "
                          f"text and randn none")
    if name == "randint" and not (arg.isdigit() and int(arg) > 0):
        raise RecordError(f"input preset {preset!r}: randint needs a positive bound")
    return name, arg


def _example(target: str, seed: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    import torch

    module, _, function = target.rpartition(":")
    if not module:
        raise RecordError(f"example preset {target!r} needs <module>:<function>")
    make = getattr(import_module(module), function)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        value = make()
    if isinstance(value, dict):
        return (), dict(value)
    if isinstance(value, (tuple, list)):
        return tuple(value), {}
    return (value,), {}


def import_module(name: str) -> Any:
    """Import a module; a classic loads without running the registry that imports them all.

    ``menagerie.classics``'s ``__init__`` imports all 2,700 classics (about 20 s), so a classic
    is imported under a bare package object that has the right search path and no registry.
    """

    package = "menagerie.classics"
    if name.startswith(f"{package}.") and package not in sys.modules:
        root = Path(importlib.util.find_spec("menagerie").origin).parent / "classics"
        bare = importlib.util.spec_from_file_location(
            package, root / "__init__.py", submodule_search_locations=[str(root)])
        sys.modules[package] = importlib.util.module_from_spec(bare)
    return importlib.import_module(name)


def _image(shape: tuple[int, ...]) -> Any:
    import torch
    from PIL import Image

    if len(shape) not in (3, 4, 5) or shape[-4 if len(shape) == 5 else -3] not in (1, 3, 4):
        raise RecordError(f"image preset: shape {shape} has no C,H,W layout with C 1, 3 or 4")
    channels, (height, width) = shape[1 if len(shape) >= 4 else 0], shape[-2:]
    with Image.open(asset_path("std-image-01")) as raw:
        image = raw.convert("RGB")
    scale = max(height / image.height, width / image.width)
    size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.BICUBIC)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    image = image.crop((left, top, left + width, top + height))
    pixels = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    rgb = pixels.view(height, width, 3).permute(2, 0, 1).float() / 255.0
    rgb = (rgb - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    if channels == 1:
        chw = rgb.mean(dim=0, keepdim=True)
    elif channels == 4:
        chw = torch.cat([rgb, torch.ones(1, height, width)])
    else:
        chw = rgb
    if len(shape) == 3:
        return chw.contiguous()
    if len(shape) == 5:
        chw = chw.unsqueeze(1).expand(channels, shape[2], height, width)
    return chw.unsqueeze(0).expand(shape[0], *chw.shape).contiguous()


def _channels_last(shape: tuple[int, ...]) -> Any:
    """The canonical image in ``N,H,W,C`` or ``H,W,C`` order."""

    if len(shape) not in (3, 4):
        raise RecordError(f"image:nhwc preset: shape {shape} is not N,H,W,C or H,W,C")
    first = (shape[-1], *shape[-3:-1])
    if len(shape) == 3:
        return _image(first).permute(1, 2, 0).contiguous()
    return _image((shape[0], *first)).permute(0, 2, 3, 1).contiguous()


# How each preview backend makes its own array from a NumPy one.
_NATIVE = {
    "tf": ("tensorflow", "convert_to_tensor"),
    "jax": ("jax.numpy", "asarray"),
    "paddle": ("paddle", "to_tensor"),
    "mlx": ("mlx.core", "array"),
}


def map_leaves(value: Any, leaf: Callable[[Any], bool], fn: Callable[[Any], Any]) -> Any:
    """Apply ``fn`` to every leaf of nested dicts, lists, tuples and named tuples; containers
    keep their type and values that are neither leaves nor containers are unchanged."""

    if leaf(value):
        return fn(value)
    if isinstance(value, dict):
        return type(value)({k: map_leaves(v, leaf, fn) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        items = [map_leaves(v, leaf, fn) for v in value]
        return type(value)(*items) if hasattr(value, "_fields") else type(value)(items)
    return value


def native_array(backend: str) -> Callable[[Any], Any]:
    """The backend's constructor of its own array from a NumPy one."""

    if backend not in _NATIVE:
        raise RecordError(f"no input conversion for backend {backend!r}; known: {sorted(_NATIVE)}")
    module, function = _NATIVE[backend]
    return getattr(importlib.import_module(module), function)


def native(value: Any, backend: str) -> Any:
    """Return ``value`` with every torch tensor replaced by the ``backend``'s array (same values,
    shape and dtype), through NumPy; containers keep their type and other values are unchanged.
    """

    if backend == "torch":
        return value
    make = native_array(backend)
    import torch

    return map_leaves(value, lambda item: isinstance(item, torch.Tensor),
                      lambda tensor: make(tensor.detach().cpu().numpy()))


def _text(shape: tuple[int, ...]) -> Any:
    import torch

    data = asset_path("std-text-01").read_text(encoding="utf-8").strip().encode("utf-8")
    ids = torch.tensor(list(data), dtype=torch.int64) + 1
    count = 1
    for dim in shape:
        count *= dim
    return ids.repeat(count // ids.numel() + 1)[:count].view(shape)
