"""The catalog pass: its plan, and recipes for the library tier from library defaults.

Owned by the harvest lane (rung 2, fix 11; BUILD-PLAN r2 section 4). The 7,699 June catalog rows
that join to the roster (by name, zoo and variant, to rows carrying ``era`` and ``domain``) fall
into three groups by their June recipe text:

- X: exec-string recipes and the rest of the ``base-pytorch-compact`` zoo, all hand-written
  stand-ins: typed ``standin:exec`` or ``standin:compact``, never traced.
- B: recipes importing ``menagerie.classics``: traced as ``kind: classic``, review pending, never
  counted as the named model (a timm or Hugging Face name still builds the library's model).
- A: named library models, harvested from library defaults or a named published config.

A library call keeps an argument when it is the library default, a weight switch turned off, a
registry selector (a model name, an smp encoder) or task configuration (class count, input
channels, recorded in the member delta). A required argument with no default makes the record a
configured instance of a template class, with the library docstring's example values when it
has one, else June's. Any other argument (depth, width, heads, embedding size) types the row
``standin:kwargs``, and ``catalog/standin-kwargs.tsv`` lists it as a rung 3 candidate for a
rebuild at library defaults. June code that is not one library call is ``needs_adapter``;
a model whose June wrapper made its inputs is ``needs_adapter:inputs``.

Rows whose library is not in ``torch-main`` get ``extras`` from ``OVERLAYS`` (wheels, pinned), and
``harvest tier`` reruns itself inside that overlay. ``catalog plan`` imports no torch.
"""

from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout
from menagerie.pipeline.records import ModelRecord, RecordError, from_dict

COMPACT_ZOO = "base-pytorch-compact"
CLASSICS = "menagerie.classics"
# Environments cut 2 (mmlab, detectron2, PaddleClas and PaddleOCR configs): rung 3's rows.
DEFERRED = re.compile(r"mmdet|mmseg|mmpretrain|mmengine|mmcv|mmocr|mmpose|open-mmlab|detectron2"
                      r"|paddle", re.IGNORECASE)
FAST = frozenset({"validated", "failed:exception", "failed:replay", "failed:trace_summary"})
# Registry order of the pass: timm first; classics (group B) come after every A row of a wave.
REGISTRIES = ("timm", "torchvision", "segmentation_models_pytorch", "transformers",
              "huggingface", "diffusers", "ultralytics", "torch_geometric", "torchaudio")
WAVES = ("C1", "C2", "skip", "typed", "rung3-env")
PLAN_FIELDS = ("order", "id", "name", "zoo", "variant", "family", "group", "wave", "class",
               "june", "june_s")
OVERRIDE_FIELDS = ("id", "name", "zoo", "call", "overridden", "default_gib")

TORCHAUDIO = ("torchaudio @ https://download.pytorch.org/whl/cpu/"
              "torchaudio-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl")
# Top-level import name -> the wheels its overlay installs (resolved under torch-main's pins on
# the worker, 2026-10-05). PyPI's torchaudio is a CUDA build, so it comes from the CPU index.
OVERLAYS: dict[str, tuple[str, ...]] = {
    "torch_geometric": ("torch_geometric==2.8.0.post1",),
    "effdet": ("effdet==0.4.1",),
    "monai": ("monai==1.5.2",),
    # norse 1.1.0 is sdist only on PyPI (0.0.4 lacks the 1.x API June's code uses); its wheel comes
    # from the saved-sdist cache (`implement sdist-wheel`, gotchas.md "norse 1.1.0").
    "norse": ("norse==1.1.0",),
    "spikingjelly": ("spikingjelly==0.0.0.0.14",),
    "deepctr_torch": ("deepctr-torch==0.3.0",),
    "recbole": ("recbole==1.2.1",),
    "e3nn": ("e3nn==0.6.0",),
    "neuralforecast": ("neuralforecast==3.1.9",),
    "torchaudio": (TORCHAUDIO,),
    "speechbrain": ("speechbrain==1.0.3", TORCHAUDIO),
    "snntorch": ("snntorch==1.0.0",),
    "zuko": ("zuko==1.6.0",),
    "neuralop": ("neuraloperator==2.0.0",),
    "demucs": ("demucs==4.1.0",),
    "torchdyn": ("torchdyn==1.0.6",),
    "super_image": ("super-image==0.2.0",),
    "torchsr": ("torchsr==1.0.4",),
    "pykeen": ("pykeen==1.11.1",),
    "open_clip": ("open_clip_torch==3.3.0",),
    "mambapy": ("mambapy==1.2.0",),
    "torchdiffeq": ("torchdiffeq==0.2.5",),
    "egnn_pytorch": ("egnn-pytorch==0.2.8",),
    "se3_transformer_pytorch": ("se3-transformer-pytorch==0.9.0",),
    "toolformer_pytorch": ("toolformer-pytorch==0.0.30",),
    "gigagan_pytorch": ("gigagan-pytorch==0.0.84",),
    "siren_pytorch": ("siren-pytorch==0.1.7",),
    "ogb": ("ogb==1.3.6",),
}
OVERLAY_FLAG = "MENAGERIE_HARVEST_OVERLAY"
# Libraries whose PyPI releases have no wheel (overlays install wheels only), 2026-10-05.
NO_WHEEL = frozenset({"pytorchvideo", "torch_geometric_temporal", "normflows", "nflows", "FrEIA",
                      "encodec"})

# Arguments that never change the architecture, or name a published configuration.
WEIGHT_SWITCHES = frozenset({"pretrained", "weights", "encoder_weights", "pretrained_backbone",
                             "weights_backbone", "progress", "pretrained_hf", "checkpoint_path",
                             "verbose", "init_weights"})
SELECTORS = frozenset({"encoder_name", "model_name", "name", "arch"})
TASK = frozenset({"num_classes", "classes", "num_labels", "in_channels", "in_chans",
                  "out_channels"})
EMPTY = inspect.Parameter.empty
# Calls in June code that set things up rather than build the model.
# A Hugging Face config built on its own is a stand-in's sizes, not a model.
NOT_MODELS = re.compile(r"^(torch\.(?!nn\.[A-Z])|builtins\.|types\.|warnings\.|functools\.|__"
                        r"|tempfile\.|os\.|exec$)|.*Config$")


# --- the plan ---------------------------------------------------------------------------


def joined(data: DataLayout) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(catalog row, roster row)`` for the June catalog rows the roster dates."""

    roster = {}
    with data.roster.open(encoding="utf-8") as handle:
        for row in map(json.loads, handle):
            if row.get("era") and row.get("domain") and row.get("stable_id"):
                roster[(row["name"], row["zoo"], row["variant"])] = row
    with (data.root / "master_catalog.jsonl").open(encoding="utf-8") as handle:
        for cat in map(json.loads, handle):
            found = roster.get((cat["name"], cat["zoo"], cat["variant"]))
            if found is not None:
                yield cat, found


def code(cat: Mapping[str, Any]) -> str:
    recipe = cat.get("recipe") or {}
    return recipe.get("expr") or recipe.get("code") or recipe.get("body") or ""


def group(cat: Mapping[str, Any]) -> str:
    """X (a stand-in by its recipe), B (built from our classics) or A (a library model)."""

    if (cat.get("recipe") or {}).get("type") == "exec-string" or cat["zoo"] == COMPACT_ZOO:
        return "X"
    return "B" if CLASSICS in code(cat) else "A"


def standin_class(cat: Mapping[str, Any]) -> str:
    return "standin:compact" if cat["zoo"] == COMPACT_ZOO else "standin:exec"


def catalog_plan(*, data: DataLayout, skip: Path | None = None,
                 write: bool = True) -> dict[str, Any]:
    """Write the catalog pass's ordered plan: every joined row with its group and wave.

    Inputs: the data root and ``skip``, a TSV whose ``id`` column lists rows the pass leaves out
    (rung 2's frozen rows, ``data/catalog/r2-rows.tsv``, by default). Wave C1 is every A and B
    row that validated, failed fast or was skipped in June, by registry with timm first and
    classics last; wave C2 the June-timeout and unrun rows under a 15-minute trace cap; ``typed``
    the X rows; ``rung3-env`` rows whose environment is cut; ``skip`` the frozen rows. Returns:
    counts by group, wave and both. Side effects: with ``write``, ``data/catalog/plan.tsv``.
    """

    skip = skip or data.root / "catalog" / "r2-rows.tsv"
    skipped = _ids(skip) if skip.exists() else set()
    june = _june(data)
    rows = [_plan_row(cat, ros, june.get(ros["stable_id"], {}), skipped)
            for cat, ros in joined(data)]
    rows.sort(key=_order)
    for n, row in enumerate(rows, 1):
        row["order"] = n
    if write:
        _write_tsv(data.root / "catalog" / "plan.tsv", PLAN_FIELDS, rows)
    return {"rows": len(rows), "group": dict(Counter(r["group"] for r in rows)),
            "wave": dict(Counter(r["wave"] for r in rows)),
            "group_wave": {f"{g}/{w}": n for (g, w), n in
                           sorted(Counter((r["group"], r["wave"]) for r in rows).items())}}


def _plan_row(cat: Mapping[str, Any], ros: Mapping[str, Any], june: Mapping[str, str],
              skipped: set[str]) -> dict[str, Any]:
    sid, grp, status = ros["stable_id"], group(cat), june.get("status") or "unrun"
    cls = standin_class(cat) if grp == "X" else ""
    if sid in skipped:
        wave = "skip"
    elif grp == "X":
        wave = "typed"
    elif grp == "A" and DEFERRED.search(cat["zoo"] + " " + code(cat)[:400]):
        wave = "rung3-env"
    elif status in FAST or status.startswith("skipped"):
        wave = "C1"
    else:
        wave = "C2"
    return {"id": sid, "name": cat["name"], "zoo": cat["zoo"], "variant": cat["variant"],
            "family": cat["family"], "group": grp, "wave": wave, "class": cls, "june": status,
            "june_s": june.get("elapsed") or ""}


def _order(row: Mapping[str, Any]) -> tuple:
    zoo = row["zoo"].lower()
    rank = next((n for n, r in enumerate(REGISTRIES) if zoo.startswith(r)), len(REGISTRIES))
    seconds = float(row["june_s"] or 0)
    return (WAVES.index(row["wave"]), row["group"] == "B", rank, zoo, seconds, row["id"])


def _june(data: DataLayout) -> dict[str, dict[str, str]]:
    with data.routing_manifest.open(encoding="utf-8") as handle:
        return {r["stable_id"]: r for r in csv.DictReader(handle, delimiter="\t")}


def _ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {r["id"] for r in csv.DictReader(handle, delimiter="\t")}


def _write_tsv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        out = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n",
                             extrasaction="ignore")
        out.writeheader()
        out.writerows(rows)


# --- the tier ---------------------------------------------------------------------------


def harvest_tier(zoo: str, names: Sequence[str] | None = None, *, data: DataLayout,
                 write: bool = True) -> list[ModelRecord]:
    """Write packets and model records for a zoo's catalog rows from library defaults.

    Inputs: the zoo, optionally roster names or stable ids within it (a named row is harvested
    even when its environment is cut). Rows another writer owns (a 2a packet, a 2b record) are
    left alone, and so is an existing record the draft would change (named on stderr). Returns:
    the records written or unchanged. Side effects: with ``write``, packets, model
    records and the zoo's lines of ``catalog/standin-kwargs.tsv``; when the zoo needs libraries
    this interpreter lacks, the work runs again inside their overlay.
    """

    from menagerie.pipeline import harvest

    rows = _tier_rows(zoo, names, data)
    extras = _missing_extras(rows)
    if extras and not os.environ.get(OVERLAY_FLAG):
        return _in_overlay(zoo, names, extras, write=write)
    drafts = [tier_draft(row, data) for row in rows if not _owned_elsewhere(row, data)]
    records = harvest.write_drafts(drafts, data=data, write=write, rewrite=False)
    if write:
        _write_overrides(zoo, drafts, data)
    return records


def _tier_rows(zoo: str, names: Sequence[str] | None, data: DataLayout) -> list[Any]:
    from menagerie.pipeline.harvest import Row

    rows = []
    for cat, ros in joined(data):
        if cat["zoo"] != zoo:
            continue
        sid = ros["stable_id"]
        if names and sid not in names and cat["name"] not in names:
            continue
        if not names and group(cat) == "A" and DEFERRED.search(zoo + " " + code(cat)[:400]):
            continue
        rows.append(Row(sid, cat["name"], zoo, cat["variant"], cat["family"], catalog=cat))
    if not rows:
        raise RecordError(f"no catalog rows for zoo {zoo!r} and names {list(names or [])}")
    return rows


def _owned_elsewhere(row: Any, data: DataLayout) -> bool:
    from menagerie.pipeline import read_json
    from menagerie.pipeline.harvest import _family_id
    from menagerie.pipeline.records import Packet

    families = {_family_id(row.family)}
    if data.model(row.id).exists():
        families.add(read_json(ModelRecord, data.model(row.id)).family)
    for family in families:
        path = data.packet(family)
        if path.exists() and not read_json(Packet, path).by.startswith("harvester@"):
            return True
    return False


def tier_draft(row: Any, data: DataLayout) -> Any:
    """The library tier's draft for one catalog row: a recipe, or a typed give-up."""

    from menagerie.pipeline import harvest

    if group(row.catalog) == "X":
        return _give_up(row, standin_class(row.catalog), "June's recipe is a hand-written "
                        "stand-in, not the named model; rung 3's 2b queue",
                        "code that builds the named model")
    try:
        draft = harvest._translate(row, data, parse=parse_model_call, vet=vet)
    except harvest.NoRecipe as typed:
        return _give_up(row, typed.cls, typed.reason, typed.missing)
    except ModuleNotFoundError as exc:
        return _give_up(row, f"missing_dependency:{exc.name}", f"{exc.name} does not import "
                        "in torch-main or its overlay", f"a wheel of {exc.name}")
    except RecordError:  # June code that is not one library call, and no adapter
        return _give_up(row, "needs_adapter", "June's recipe is code, not one library call with "
                        "literal arguments, and no adapter exists", "an adapter for the named model")
    except Exception as exc:  # the library refuses the call: the row needs a written recipe
        return _give_up(row, "needs_adapter", f"harvest could not build a library recipe: "
                        f"{type(exc).__name__}: {str(exc)[:200]}", "an adapter for the named model")
    if draft.recipe.get("kind") != "library":
        return draft
    if draft.recipe["input"].note.startswith("sentinel"):  # a library model reads its input
        return _give_up(row, "needs_adapter:inputs", f"June gave {draft.recipe['call']} no input "
                        "(its wrapper made them); the library model needs an input builder",
                        "an input builder for the library model")
    draft.recipe["extras"] = list(_extras(draft.recipe["call"]))
    return draft


def _give_up(row: Any, cls: str, reason: str, missing: str | None) -> Any:
    from menagerie.pipeline.harvest import Draft

    source_case = "library" if group(row.catalog) == "A" else "vendor"
    reason = re.sub(r"/(?:home|Users|tmp)/\S*", "<path>", reason)  # records are committed
    return Draft(row, source_case, {}, give_up=(cls, reason), missing=missing)


# --- June code to one library call ------------------------------------------------------


def parse_model_call(source: str) -> tuple[str, list, dict, dict, dict[str, list[str]]] | None:
    """The model's library call in June code: the strict form, else the first library constructor
    among its statements (setup calls and June's wrappers dropped). ``None`` when there is none.

    A callee must be a library name: one bound in the code itself (a wrapper class, ``_m``) is
    not, and a bare name from ``from M import *`` is qualified as ``M.name``. A row whose wrapper
    is dropped must not have needed it for its inputs: ``vet`` types those.
    """

    from menagerie.pipeline.harvest import _parse_call

    try:
        tree = ast.parse(source.strip())
    except SyntaxError:
        return None
    local, stars = _bound_names(tree), _star_modules(tree)
    strict = _parse_call(source, refs=True)
    candidates = [(strict, False)] if strict else [(c, True) for c in _statement_calls(tree)]
    for parsed, wrapped in candidates:
        call = _qualify(parsed[0], local, stars)
        if call is not None and not NOT_MODELS.match(call):
            return (call, *parsed[1:4], {**parsed[4], **({"_wrapped": ["1"]} if wrapped else {})})
    return None


def _statement_calls(tree: ast.Module) -> Iterator[tuple[str, list, dict, dict, dict]]:
    """Each top-level statement's call with literal arguments, in order (``.eval()`` dropped)."""

    from menagerie.pipeline.harvest import _parse_call

    for node in tree.body:
        value = node.value if isinstance(node, (ast.Assign, ast.Expr)) else None
        while (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
               and value.func.attr == "eval" and not value.args):
            value = value.func.value  # `f(...).eval()`
        if isinstance(value, ast.Call):
            parsed = _parse_call(_imports(tree) + f"model = {ast.unparse(value)}", refs=True)
            if parsed:
                yield parsed


def _bound_names(tree: ast.Module) -> set[str]:
    names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def _star_modules(tree: ast.Module) -> list[str]:
    return [n.module for n in tree.body if isinstance(n, ast.ImportFrom) and n.module
            and any(a.name == "*" for a in n.names)]


def _qualify(call: str, local: set[str], stars: Sequence[str]) -> str | None:
    """``call`` as a library name, or ``None`` when its head is bound in the June code."""

    head = call.split(".")[0]
    if head in local:
        return None
    if "." in call or not stars or importlib.util.find_spec(head) is not None:
        return call
    for module in stars:
        try:
            if hasattr(importlib.import_module(module), head):
                return f"{module}.{call}"
        except ImportError:
            continue
    return call


def _imports(tree: ast.Module) -> str:
    lines = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    return "".join(f"{line}\n" for line in lines)


# --- the arguments of a library call ----------------------------------------------------


def vet(row: Any, call: str, args: list, kwargs: dict) -> tuple[list, dict, dict[str, Any]]:
    """Check a library call's arguments against the library; return what to build and the delta.

    Raises ``NoRecipe``: ``standin:kwargs`` for an argument that overrides a default and is not a
    weight switch, selector or task configuration; ``needs_adapter:inputs`` when June's dropped
    wrapper made the model's inputs.
    """

    from menagerie.pipeline.harvest import NoRecipe, _resolve

    top = call.split(".")[0]
    if importlib.util.find_spec(top) is None:
        cls = "missing_dependency" + (":sdist" if top in NO_WHEEL else "")
        raise NoRecipe(f"{cls}", f"{top} is not in torch-main and has no overlay here"
                       + (" (no wheel on PyPI)" if top in NO_WHEEL else ""), f"a wheel of {top}")
    fn = _resolve(call)
    params = _parameters(fn)
    bound = _bind(params, args, kwargs)
    if call == "torchvision.models.get_model":  # its builder's signature has the defaults
        from torchvision.models import get_model_builder

        params = {**params, **_parameters(get_model_builder(args[0]))}
    split: dict[str, dict[str, Any]] = {"overridden": {}, "task": {}, "required": {}, "kept": {}}
    for name, value in bound.items():
        split[_kind(params.get(name), name, value)][name] = value
    overridden, task, required = split["overridden"], split["task"], split["required"]
    # A config with sizes under a name that is not the model type: the default would substitute
    # the reference model for the named one (the model-type rows took the rung 0 rule earlier).
    june = parse_model_call(code(row.catalog))
    sizes = {k: v for k, v in (june[3] if june and june[0] == call else {}).items()
             if k != "model_type"}
    overridden.update(sizes)
    if overridden:
        raise NoRecipe("standin:kwargs", f"{call} with {_show(overridden)}, which override the "
                       "library's defaults; rung 3 candidate for a rebuild at library defaults",
                       f"{call} at its library defaults (catalog/standin-kwargs.tsv)")
    spec = row.catalog.get("input") or {}
    if _wrapped(row) and spec.get("kind") in ("none", "multi", "callable", "kwargs", None):
        raise NoRecipe("needs_adapter:inputs", f"June's wrapper around {call} made its inputs "
                       f"(input kind {spec.get('kind')!r}); the bare model needs an input adapter",
                       "an input builder for the bare library model")
    delta: dict[str, Any] = dict(task)
    if required and not _is_registry(call):
        example = _doc_example(fn, required)
        source = "the library docstring's example" if example else "June's recipe"
        if example:
            args, kwargs = _rebuild(params, args, kwargs, example)
            required = example
        delta.update(required)
        delta["template"] = (f"configured instance of the template class {call.rpartition('.')[2]}; "
                             f"sizes from {source}")
    return args, kwargs, delta


def _kind(param: inspect.Parameter | None, name: str, value: Any) -> str:
    """``required`` (no default), ``task``, ``kept`` (a default, switch or selector) or
    ``overridden``; an argument absorbed by ``**kwargs`` has no default to compare with."""

    if param is not None and param.default is EMPTY:
        return "required"
    default = EMPTY if param is None else param.default
    from menagerie.pipeline.harvest_build import component

    if default is None and component(value):
        return "required"  # a component the library leaves to the caller (a neuron, a layer)
    if name in TASK:
        return "kept" if _same(value, default) else "task"
    if isinstance(value, dict) and set(value) == {"$ref"}:
        from menagerie.pipeline.harvest import _deref

        value = _deref(value)
    if name in WEIGHT_SWITCHES or name in SELECTORS or _same(value, default):
        return "kept"
    return "overridden"


def _same(value: Any, default: Any) -> bool:
    try:
        return default is not EMPTY and bool(value == default)
    except Exception:  # an array-like default compares elementwise
        return False


def _wrapped(row: Any) -> bool:
    parsed = parse_model_call(code(row.catalog))
    return bool(parsed and parsed[4].get("_wrapped"))


def _is_registry(call: str) -> bool:
    return call in ("timm.create_model", "torchvision.models.get_model") or call.endswith(
        ".create_model")


def _parameters(fn: Any) -> dict[str, inspect.Parameter]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}
    return {n: p for n, p in sig.parameters.items() if p.kind not in (p.VAR_POSITIONAL,
                                                                       p.VAR_KEYWORD)}


def _bind(params: Mapping[str, inspect.Parameter], args: list, kwargs: dict) -> dict[str, Any]:
    positional = [n for n, p in params.items() if p.kind != p.KEYWORD_ONLY]
    bound = dict(zip(positional, args))
    for extra in args[len(positional):]:
        bound[f"<arg {len(bound)}>"] = extra
    return {**bound, **kwargs}


def _rebuild(params: Mapping[str, inspect.Parameter], args: list, kwargs: dict,
             example: Mapping[str, Any]) -> tuple[list, dict]:
    positional = [n for n, p in params.items() if p.kind != p.KEYWORD_ONLY]
    named = {**dict(zip(positional, args)), **kwargs, **example}
    return [], named


def _doc_example(fn: Any, required: Mapping[str, Any]) -> dict[str, Any] | None:
    """The literal keyword values of a ``Name(...)`` example in the class or function docstring,
    when they cover every required argument."""

    name = getattr(fn, "__name__", "")
    doc = "\n".join(filter(None, [inspect.getdoc(fn) or "",
                                  inspect.getdoc(getattr(fn, "__init__", None)) or ""]))
    doc = re.sub(r"(?m)^\s*(>>>|\.\.\.)", "", doc)  # doctest prompts
    for match in re.finditer(rf"\b{re.escape(name)}\(", doc):
        found = _literal_call(doc[match.start():match.start() + 600])
        if found is not None and set(required) <= set(found):
            return found
    return None


def _literal_call(text: str) -> dict[str, Any] | None:
    """The keywords of the shortest ``f(...)`` prefix of ``text`` with only literal keywords."""

    ends = [n + 1 for n, char in enumerate(text) if char == ")"]
    for end in ends:
        try:
            node = ast.parse(" ".join(text[:end].split()), mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, ast.Call) or node.args:
            return None
        try:
            return {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
        except ValueError:
            return None
    return None


def _show(kwargs: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in kwargs.items())[:200]


# --- overlays ---------------------------------------------------------------------------


def _extras(call: str) -> tuple[str, ...]:
    top = call.split(".")[0]
    return OVERLAYS.get(top, ()) if not _in_base(top) else ()


def _in_base(top: str) -> bool:
    """Whether ``torch-main`` itself has the module (an overlay's extras are not counted)."""

    if not os.environ.get(OVERLAY_FLAG):
        return importlib.util.find_spec(top) is not None
    return top not in OVERLAYS


def _missing_extras(rows: Sequence[Any]) -> list[str]:
    tops: set[str] = set()
    for row in rows:
        imports = (row.catalog.get("recipe") or {}).get("imports") or []
        parsed = parse_model_call(code(row.catalog))  # an expression names its module only
        tops |= {i.split(".")[0] for i in imports} | set(_code_imports(code(row.catalog)))
        tops |= {parsed[0].split(".")[0]} if parsed else set()
    missing = sorted(t for t in tops if t in OVERLAYS and importlib.util.find_spec(t) is None)
    return sorted({req for t in missing for req in OVERLAYS[t]})


def _code_imports(source: str) -> Iterator[str]:
    try:
        tree = ast.parse(source.strip())
    except SyntaxError:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.module.split(".")[0]


def _in_overlay(zoo: str, names: Sequence[str] | None, extras: Sequence[str], *,
                write: bool) -> list[ModelRecord]:
    from menagerie.pipeline import envs
    from menagerie.pipeline.harvest import ENV

    root = os.environ.get("MENAGERIE_ENVS")
    if not root:
        raise RecordError(f"zoo {zoo!r} needs {list(extras)}: set MENAGERIE_ENVS for the overlay")
    python = envs.overlay(ENV, extras, root=Path(root))
    argv = [str(python), "-m", "menagerie.pipeline", "harvest", "tier", zoo, *(names or ())]
    done = subprocess.run([*argv, *([] if write else ["--dry-run"])], capture_output=True,
                          text=True, check=False, env={**os.environ, OVERLAY_FLAG: "1"})
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-4000:])
        raise RecordError(f"harvest tier {zoo!r} in its overlay exited {done.returncode}")
    return [from_dict(ModelRecord, json.loads(line)) for line in done.stdout.splitlines()
            if line.startswith("{")]


# --- rung 3 rebuild candidates ----------------------------------------------------------


def _write_overrides(zoo: str, drafts: Sequence[Any], data: DataLayout) -> None:
    """Replace the zoo's lines of ``catalog/standin-kwargs.tsv`` with this run's ``standin:kwargs``
    rows: the call, the overridden arguments and the fp32 weight size at library defaults."""

    path = data.root / "catalog" / "standin-kwargs.tsv"
    kept = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            kept = [r for r in csv.DictReader(handle, delimiter="\t") if r["zoo"] != zoo]
    for draft in drafts:
        if not draft.give_up or draft.give_up[0] != "standin:kwargs":
            continue
        call, _, rest = draft.give_up[1].partition(" with ")
        kept.append({"id": draft.row.id, "name": draft.row.name, "zoo": zoo, "call": call,
                     "overridden": rest.partition(", which override")[0],
                     "default_gib": _default_gib(call, draft.row)})
    _write_tsv(path, OVERRIDE_FIELDS, sorted(kept, key=lambda r: (r["zoo"], r["id"])))


def _default_gib(call: str, row: Any) -> str:
    """The fp32 weights of ``call`` with only its selectors and required arguments, on meta."""

    from menagerie.pipeline.harvest import _resolve, on_device

    parsed = parse_model_call(code(row.catalog))
    params = _parameters(_resolve(call))
    keep = {k: v for k, v in _bind(params, list(parsed[1]), dict(parsed[2])).items()
            if not k.startswith("<") and (k in SELECTORS or k in TASK or k in WEIGHT_SWITCHES
                                          or k in params and params[k].default is EMPTY)}
    try:
        with on_device("meta"):
            model = _resolve(call)(**keep)
        return f"{sum(p.numel() for p in model.parameters()) * 4 / 2**30:.3f}"
    except Exception:  # sizing is best effort; the rebuild itself happens at rung 3
        return ""
