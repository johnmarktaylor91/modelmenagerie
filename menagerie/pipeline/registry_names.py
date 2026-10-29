"""A row's name looked up in the model registries this environment installs.

Rung 3 (r3-coverage, after the drafts review of 2026-10-09): when June's code is a stand-in or
fails identity, the named model may still be a registry entry, and the registry's default is
then its published configuration. Names match exactly after normalizing case and ``-``/``_``,
with the zoo or library prefix (``monai.``, ``ganzoo_``), an ``_hf`` suffix and a ``:<task>``
part dropped (a ``:`` part that is not a task is the variant, and stays). A registry whose library does not import here is skipped. Each hit is a parsed
call ``(call, args, kwargs, config, imported)`` like ``harvest_tiers.parse_model_call``'s.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
import re
from collections.abc import Callable
from pathlib import Path

Parsed = tuple[str, list, dict, dict, dict]
# Ultralytics scale letters a model yaml stem takes after its version (yolo11 -> yolo11n).
SCALES = "nslmxtbce"
# A ``name:part`` whose part is a task (``image-classification``, ``ctc-asr``, ``base-vlm``).
TASK_PART = re.compile(r"-(classification|detection|segmentation|estimation|answering|text|vlm|"
                       r"asr|seq2seq|xvector|waveform|spectrogram|encoder|codec|fusion|"
                       r"generation)$", re.IGNORECASE)


def norm(name: str) -> str:
    return name.casefold().replace("-", "_")


def candidates(name: str, zoo: str, prefixes: tuple[str, ...] = ()) -> list[str]:
    """The row name's normalized forms: as is, without a ``:<task>`` part, an ``_hf`` suffix
    and a zoo or library prefix (the zoo label's first word, or any of ``prefixes``). A ``:``
    part is dropped only when it names a task (``udop:base-document-vlm``); otherwise it is the
    variant (``midas:MiDaS_small``, ``mmdet:convnext``) and the namespace alone would name
    another model."""

    head, _, part = name.partition(":")
    forms = [name, head] if TASK_PART.search(part) else [name]
    heads = sorted(h for h in {re.split(r"[\s.\-_(]", zoo, maxsplit=1)[0], *prefixes} if h)
    forms += [rest for base in list(forms) for rest in _unprefixed(base, heads)]
    forms += [f[:-3] for f in list(forms) if norm(f).endswith("_hf")]
    return list(dict.fromkeys(norm(f) for f in forms if f))


def _unprefixed(name: str, heads: list[str]) -> list[str]:
    """``name`` without each zoo or library prefix it starts with (``monai.``, ``ganzoo_``)."""

    return [name[len(head) + 1:] for head in heads for sep in ("_", ".", "-")
            if norm(name).startswith(norm(head + sep))]


def _installed(top: str) -> bool:
    return importlib.util.find_spec(top) is not None


def _torchvision(names: list[str]) -> Parsed | None:
    from torchvision.models import list_models

    found = {norm(n): n for n in list_models()}
    hit = next((found[n] for n in names if n in found), None)
    return ("torchvision.models.get_model", [hit], {"weights": None}, {}, {}) if hit else None


def _timm(names: list[str]) -> Parsed | None:
    import timm

    found = {norm(n): n for n in timm.list_models()}
    hit = next((found[n] for n in names if n in found), None)
    return ("timm.create_model", [hit], {"pretrained": False}, {}, {}) if hit else None


def _open_clip(names: list[str]) -> Parsed | None:
    import open_clip

    found = {norm(n): n for n in open_clip.list_models()}
    hit = next((found[n] for n in names if n in found), None)
    return ("open_clip.create_model", [hit], {}, {}, {}) if hit else None


def _transformers(names: list[str]) -> Parsed | None:
    from transformers import CONFIG_MAPPING

    found = {norm(n): n for n in CONFIG_MAPPING.keys()}
    hit = next((found[n] for n in names if n in found), None)
    return ("transformers.AutoModel.from_config", [], {}, {"model_type": hit}, {}) if hit else None


def _classes(module: str, base: str) -> Callable[[list[str]], Parsed | None]:
    """A model class ``module`` exports (a subclass of ``base``) whose name is the row name.
    Only a matching name is imported: the libraries export lazily."""

    def lookup(names: list[str]) -> Parsed | None:
        mod = importlib.import_module(module)
        head, _, attr = base.rpartition(".")
        parent = getattr(importlib.import_module(head), attr)
        exported = {norm(a): a for a in dir(mod) if not a.startswith("_")}
        for name in (exported[n] for n in names if n in exported):
            try:
                obj = getattr(mod, name)
            except Exception:  # a lazy export whose own dependency is missing here
                continue
            if isinstance(obj, type) and issubclass(obj, parent):
                return f"{module}.{name}", [], {}, {}, {}
        return None

    return lookup


def _ultralytics(names: list[str]) -> Parsed | None:
    from menagerie.pipeline.harvest import YOLO_TASKS

    for stem in _yaml_stems():
        for n in names:
            if re.fullmatch(_scaled(stem), n):
                task = stem.rpartition("_")[2]  # stems are normalized: yolov8_seg
                call = f"ultralytics.nn.tasks.{YOLO_TASKS.get(task, 'DetectionModel')}"
                return call, [f"{n.replace('_', '-')}.yaml"], {"verbose": False}, {}, {}
    return None


@functools.lru_cache(maxsize=1)
def _yaml_stems() -> tuple[str, ...]:
    spec = importlib.util.find_spec("ultralytics")
    root = Path(spec.origin).parent / "cfg" / "models" if spec and spec.origin else None
    if root is None or not root.is_dir():
        return ()
    return tuple(sorted({norm(p.stem) for p in root.rglob("*.yaml")}, key=lambda s: (-len(s), s)))


def _scaled(stem: str) -> str:
    """A stem as a pattern that also takes a scale letter after its first number."""

    digits = re.search(r"\d+", stem)
    if digits is None:
        return re.escape(stem)
    at = digits.end()
    return re.escape(stem[:at]) + f"[{SCALES}]?" + re.escape(stem[at:])


# (import name, lookup), tried in order after the zoo's own library; a library that does not
# import is skipped. TorchVision before timm: its classic models are the reference builds.
REGISTRIES: tuple[tuple[str, Callable[[list[str]], Parsed | None]], ...] = (
    ("torchvision", _torchvision),
    ("timm", _timm),
    ("open_clip", _open_clip),
    ("ultralytics", _ultralytics),
    ("monai", _classes("monai.networks.nets", "torch.nn.Module")),
    ("diffusers", _classes("diffusers", "diffusers.ModelMixin")),
    ("transformers", _transformers),
)


def lookup(name: str, zoo: str, prefixes: tuple[str, ...] = ()) -> Parsed | None:
    """The first registry entry the row's name is, as a parsed default call, else ``None``."""

    names = candidates(name, zoo, prefixes)
    own = norm(re.split(r"[\s.\-(]", zoo, maxsplit=1)[0])
    ordered = sorted(REGISTRIES, key=lambda entry: norm(entry[0]) != own)  # stable
    for top, find in ordered:
        if not _installed(top):
            continue
        try:
            hit = find(names)
        except ImportError:  # the registry's own import needs something missing here
            continue
        if hit is not None:
            return hit
    return None

