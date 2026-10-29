"""Building a recipe's model: library calls, classics and adapters, on the CPU or the meta device.

Split from ``harvest`` (rung 2), which re-exports these names: the worker calls
``harvest.construct`` and adapters call ``harvest.vendor_dir`` and ``harvest.hub_model``.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import types
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout
from menagerie.pipeline.inputs import contract, import_module
from menagerie.pipeline.records import Recipe, RecipeInput, RecordError, to_dict

REPO = Path(__file__).resolve().parents[2]
# A recipe argument naming an object (``{"$ref": "<dotted name>"}``), resolved at build time.
REF = "$ref"
# A recipe argument calling one (``{"$call": "<dotted name>", "args": [...], "kwargs": {...}}``).
CALL = "$call"


class MissingLibrary(RecordError):
    """No module of a dotted name imports here: an environment fact, not a property of the code."""

# An encoder-decoder (t5, bart) also needs decoder input: the canonical text feeds both sides.
SEQ2SEQ_SHAPE = (1, 16)
SEQ2SEQ_INPUT = RecipeInput(
    preset="example:menagerie.pipeline.harvest:seq2seq_text", kind="standard-text",
    shape=SEQ2SEQ_SHAPE, asset="std-text-01",
    note="the canonical prompt's token ids (1, 16) as input_ids and decoder_input_ids")


def _deref(value: Any) -> Any:
    """A recipe argument, with a ``{"$ref": name}`` resolved to the object it names and a
    ``{"$call": name, "args", "kwargs"}`` to the result of calling it (arguments resolved first);
    lists and tuples are resolved item by item."""

    if isinstance(value, dict) and set(value) == {REF}:
        return _resolve(value[REF])
    if isinstance(value, dict) and set(value) == {CALL, "args", "kwargs"}:
        args = [_deref(a) for a in value["args"]]
        return _resolve(value[CALL])(*args, **{k: _deref(v) for k, v in value["kwargs"].items()})
    if isinstance(value, (list, tuple)):
        return type(value)(_deref(v) for v in value)
    return value


def names_class(target: str) -> bool:
    """Whether a dotted name resolves to a class (a layer, a neuron), not a constant or helper."""

    try:
        return isinstance(_resolve(target), type)
    except (ImportError, AttributeError, RecordError):
        return False


def component(value: Any) -> bool:
    """Whether a recipe argument is an object reference or a call (``$ref`` or ``$call``)."""

    return isinstance(value, dict) and (set(value) == {REF} or set(value) == {CALL, "args",
                                                                              "kwargs"})


# --- construction ----------------------------------------------------------------------


def construct(recipe: Recipe, *, device: str = "cpu", data: DataLayout | None = None) -> Any:
    """Build the model a recipe names, with random weights, on ``device`` (``cpu`` or ``meta``).

    Weights are drawn under seed 0 inside a forked RNG, so two constructions are identical
    and the caller's random state is untouched.
    """

    import torch

    build = _builder(to_dict(recipe), data)
    try:
        model = _seeded(build, device)
    except Exception as exc:
        from menagerie.pipeline.meta_build import is_meta_error, probe_mode

        if device != "meta" or not is_meta_error(exc):
            raise
        # A constructor that reads a tensor value (Swin's drop-path `linspace(...).item()`)
        # builds with its small factory tensors on the CPU; `.to("meta")` below moves them.
        model = _seeded(build, device, probe_mode())
    if device == "meta" and isinstance(model, torch.nn.Module):
        # Legacy constructors (`nn.Parameter(torch.Tensor(n, d))`, DINO's `level_embed`) ignore
        # the default device; a meta build holds only meta tensors, so structure-only accepts it.
        model = model.to("meta")
    return model


def _seeded(build: Callable[[], Any], device: str, mode: Any = None) -> Any:
    import torch

    with torch.random.fork_rng(devices=[]), on_device(device), mode or nullcontext():
        torch.manual_seed(0)
        return build()


# Modules whose first import runs tensor operations, which fail under the meta device (torch's
# dynamo, which transformers' model base imports): imported on the CPU first (rung 1 fix 11).
META_UNSAFE_IMPORTS = ("torch._dynamo", "transformers.modeling_utils")


@contextmanager
def on_device(device: str) -> Iterator[None]:
    """``torch.device(device)``, with the meta-unsafe imports done beforehand on the CPU."""

    import torch

    if device == "meta":
        for name in META_UNSAFE_IMPORTS:
            if importlib.util.find_spec(name.partition(".")[0]) is not None:
                importlib.import_module(name)
    with torch.device(device):
        yield


def _library_build(call: str, config: Mapping[str, Any], args: list, kwargs: dict
                   ) -> Callable[[], Any]:
    fn = _resolve(call)
    first = [_hf_config(config) if call.startswith("transformers.") else dict(config)] \
        if config else []

    def build() -> Any:
        # A ``$call`` argument builds a component: inside build, under its seed and device.
        return fn(*first, *(_deref(v) for v in args), **{k: _deref(v) for k, v in kwargs.items()})

    return build


def _builder(recipe: Mapping[str, Any], data: DataLayout | None) -> Callable[[], Any]:
    raw_args = list(recipe.get("args", ()))
    raw_kwargs = dict(recipe.get("kwargs") or {})
    config = recipe.get("config") or {}
    if recipe["kind"] == "library":
        build = _library_build(recipe["call"], config, raw_args, raw_kwargs)
        preset = (recipe.get("input") or {}).get("preset", "")
        if preset.startswith("example:menagerie_library_inputs_"):
            from menagerie.pipeline import library_inputs

            if preset != library_inputs.preset(dict(recipe)):
                raise RecordError("library input binding differs from its recipe")
            alias = types.ModuleType(preset.split(':')[1])
            alias.example_input = functools.partial(library_inputs.example_input, dict(recipe))
            sys.modules[alias.__name__] = alias
        return build
    if recipe["kind"] == "classic":
        module, _, name = recipe["adapter"].partition(":")
        fn = getattr(import_module(module), name or "build")
        kwargs = {k: _deref(v) for k, v in raw_kwargs.items()}
        return lambda: fn(*raw_args, **kwargs)
    module = load_adapter(recipe["adapter"], data=data, config=config)
    return lambda: module.build(dict(config))


ADAPTER_MODULE = "menagerie_adapter"


def adapter_file(adapter: str, *, data: DataLayout | None = None) -> Path:
    """Return the file an adapter recipe loads: the attempt's ``adapter_src`` copy, if any."""

    override = os.environ.get("MENAGERIE_ADAPTER")
    if override:
        return Path(override)
    return (data or DataLayout.from_env()).root / adapter


def load_adapter(adapter: str, *, data: DataLayout | None = None,
                 config: Mapping[str, Any] | None = None) -> Any:
    """Import an adapter (``adapters/<family>.py``), also as ``menagerie_adapter``.

    Under that name an ``example_input(cfg)`` taking a config is bound to ``config``, so the
    ``example:menagerie_adapter:example_input`` preset makes the inputs of the variant built.
    """

    module = _load_file(adapter_file(adapter, data=data), name=Path(adapter).stem)
    alias = types.ModuleType(ADAPTER_MODULE)
    alias.__dict__.update(vars(module))
    example = getattr(module, "example_input", None)
    if callable(example) and inspect.signature(example).parameters:
        alias.example_input = functools.partial(example, dict(config or {}))
    sys.modules[ADAPTER_MODULE] = alias
    return module


def vendor_dir(*parts: str) -> Path:
    """Return a path in the recipe's vendor snapshot (``MENAGERIE_VENDOR``, read-only)."""

    root = os.environ.get("MENAGERIE_VENDOR")
    if not root:
        raise RecordError("no vendor snapshot in this attempt: set the recipe's `vendor`")
    return Path(root, *parts)


def hub_model(path: Path | str, *, auto: str = "AutoModel", **overrides: Any) -> Any:
    """Build a Hub custom-code model from a saved snapshot directory, offline, never weights.

    The config and its code come from ``path`` with ``trust_remote_code=True`` and the model
    from ``<auto>.from_config``, so no weight file is read. ``overrides`` set config fields (a
    variant's size). ``auto_map`` entries naming another repository (``repo--module.Class``)
    are pointed at the same module in ``path`` when it is there; otherwise loading fails, which
    is a research gap (the other repository's code was not saved).
    """

    import torch
    import transformers

    with _local_auto_map(Path(path)) as local:
        # Config and code load on the CPU device: under a caller's meta device (a structure-only
        # capture), transformers' lazy imports of the snapshot's code fail ("Could not import
        # module 'PreTrainedModel'", rung 1 fix 11). Only the construction follows the caller.
        with torch.device("cpu"):
            config = transformers.AutoConfig.from_pretrained(
                str(local), trust_remote_code=True, local_files_only=True, **overrides)
            _import_hub_class(config, local, auto)
        return getattr(transformers, auto).from_config(config, trust_remote_code=True)


def _import_hub_class(config: Any, local: Path, auto: str) -> None:
    """Import ``transformers``' model base and the snapshot's model class, so later calls hit
    the module cache."""

    import transformers.modeling_utils  # noqa: F401
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    ref = (getattr(config, "auto_map", None) or {}).get(auto)
    if ref:
        get_class_from_dynamic_module(ref if isinstance(ref, str) else ref[0], str(local),
                                      local_files_only=True)


@contextmanager
def _local_auto_map(path: Path) -> Iterator[Path]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    auto_map = config.get("auto_map") or {}
    if not any("--" in str(ref) for ref in auto_map.values()):
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="hub-") as tmp:
        local = Path(tmp) / path.name
        shutil.copytree(path, local)
        config["auto_map"] = {k: _local_ref(v, local) for k, v in auto_map.items()}
        (local / "config.json").write_text(json.dumps(config), encoding="utf-8")
        yield local


def _local_ref(ref: Any, local: Path) -> Any:
    if isinstance(ref, list):
        return [_local_ref(r, local) for r in ref]
    if not isinstance(ref, str) or "--" not in ref:
        return ref
    bare = ref.split("--", 1)[1]
    module = bare.rsplit(".", 1)[0]
    return bare if (local / f"{module.replace('.', '/')}.py").is_file() else ref


def _hf_config(config: Mapping[str, Any]) -> Any:
    import transformers

    return transformers.AutoConfig.for_model(**config)


def _resolve(call: str) -> Any:
    """Import the longest module prefix of a dotted name and walk the rest as attributes."""

    parts = call.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise MissingLibrary(f"cannot import any module of {call!r}")


def _module_file(module: str) -> Path:
    base = REPO / Path(*module.split("."))
    return base / "__init__.py" if base.is_dir() else base.with_suffix(".py")


def _module_doc(module: str) -> str:
    return ast.get_docstring(ast.parse(_module_file(module).read_text(encoding="utf-8"))) or ""


def _load_file(path: Path, *, name: str | None = None) -> Any:
    # Registered before it runs: dataclasses and pickling look a module up in sys.modules.
    name = "menagerie_adapter_" + re.sub(r"\W", "_", name or path.stem)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RecordError(f"cannot load adapter {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _encoder_decoder(recipe: Mapping[str, Any]) -> bool:
    """Whether a recipe builds a Hugging Face encoder-decoder (t5, bart), which needs decoder ids."""

    config = recipe.get("config") or {}
    return (recipe["kind"] == "library" and recipe["call"].startswith("transformers.")
            and bool(config) and bool(getattr(_hf_config(config), "is_encoder_decoder", False)))


def seq2seq_text() -> dict[str, Any]:
    """The canonical prompt as both ``input_ids`` and ``decoder_input_ids``, shape (1, 16)."""

    from menagerie.pipeline.inputs import make_input

    ids = make_input(contract("text", SEQ2SEQ_SHAPE))[0][0]
    return {"input_ids": ids, "decoder_input_ids": ids.clone()}
