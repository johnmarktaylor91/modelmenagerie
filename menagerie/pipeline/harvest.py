"""The library tier: packets and model records for library rows, without agents.

Owned by the harvest lane. ``construct`` is the one way a recipe becomes a model, used by the
worker; giants are built on the meta device for a structure-only capture. An adapter recipe's
file is the attempt's copy of the request's ``adapter_src`` when the runner sets
``MENAGERIE_ADAPTER``, else ``<data>/<recipe.adapter>``; the module is also importable as
``menagerie_adapter`` (so an input preset can name ``example:menagerie_adapter:example_input``).
``vendor_dir`` and ``hub_model`` are for adapters: the recipe's vendor snapshot, and a Hub
custom-code model built offline from a saved snapshot with random weights.

Rows come from the June catalog (``master_catalog.jsonl``) and the classics registry, keyed by
their stable ids. A timm row is always ``timm.create_model(name)``, the registry's own model;
any other catalog recipe is translated into a data recipe: a library call with literal
arguments, a Hugging Face model built from a config, a function of a ``menagerie.classics``
module, or, where the June recipe was inline code, an adapter file under ``adapters/``. Facts
come from the installed library: its version and license metadata, the paper its docstrings
link, the timm Hub repository. A family packet gains the members and sources of every
harvested row; a model record keeps the ``modes`` earlier runs observed at the same revision.
Measured peaks from the Slice A table and the smoke manifest's cluster rows set ``memory_gib``
and ``placement: axon-cpu``; the driver gives cluster rows a structure-only capture at once,
built on the meta device.
"""

from __future__ import annotations

import ast
import csv
import functools
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, now, read_json, recipe_revision, write_json
from menagerie.pipeline.inputs import contract
from menagerie.pipeline.harvest_facts import (
    ARXIV,
    _code_source,
    _library_facts,
    _license,
    _paper_source,
    _paper_sources,
    _timm_cfg,
    _version,
)
from menagerie.pipeline.harvest_build import (  # noqa: F401 (re-exported: worker, adapters)
    ADAPTER_MODULE,
    CALL,
    REF,
    _builder,
    _deref,
    _hf_config,
    _load_file,
    _module_doc,
    _module_file,
    _resolve,
    adapter_file,
    construct,
    hub_model,
    load_adapter,
    SEQ2SEQ_INPUT,
    _encoder_decoder,
    on_device,
    seq2seq_text,
    vendor_dir,
)
from menagerie.pipeline.records import (
    Availability,
    Citation,
    GiveUp,
    HFRef,
    Identity,
    Member,
    ModelRecord,
    Packet,
    Recipe,
    RecipeInput,
    RecordError,
    Search,
    Source,
    from_dict,
    to_dict,
)

ENV = "torch-main"
REPO = Path(__file__).resolve().parents[2]
CLASSICS = "menagerie.classics"
# Placements set by the plan, not by a measurement: beit_large_patch16_512 is the 128 GiB
# reviewed job (BUILD-PLAN r2 sections 3 and 5; the giants lane, 2026-10-05).
PLACED = {"m3635": "axon-reviewed"}

# The design's twenty (DESIGN section 7, rung 0) plus the two rows the worker lane's
# acceptance names: the YOLOv8n detection head and the beit_large structure-only giant.
RUNG0_NAMED = (
    "m8839", "m8850", "m7341", "m3882", "m3825",  # timm, passed Slice A or June
    "m3631", "m3711", "m3910", "m3798", "m3877",  # timm: sdpa, batch_norm, addcmul, shallow conv, undiagnosed
    "m3638", "m5228", "m10919",                   # Hugging Face from config: bert-tiny, gpt2, vit-base
    "m10921", "m7325",                            # torchvision vit_b_16, mobilenet_v3_small
    "m10036", "m9427",                            # smp Unet(resnet34), FPN(resnet18)
    "m1574", "m1064", "m2061",                    # classics: LeNet-5, Boltzmann machine, LSTM 1997
    "m11497", "m3635",                            # yolov8n, beit_large_patch16_512
)
# Hub repositories of the named Hugging Face rows, whose library default config is the
# published model (bert-base-uncased, gpt2, vit-base-patch16-224).
HF_NAMED = {
    "m3638": "google-bert/bert-base-uncased",
    "m5228": "openai-community/gpt2",
    "m10919": "google/vit-base-patch16-224",
}

YOLO = re.compile(r"YOLO\(['\"]([\w.-]+\.yaml)['\"]\)")
YOLO_TASKS = {"cls": "ClassificationModel", "seg": "SegmentationModel", "pose": "PoseModel",
              "obb": "OBBModel"}
INTEGER_DTYPES = ("long", "int64", "int32", "int")
FETCHED_FIELDS = ("sid", "blob", "text", "fetched")


@dataclass(frozen=True)
class Row:
    """One roster row to harvest: its identity plus the June catalog row or classics entry."""

    id: str
    name: str
    zoo: str
    variant: str
    family: str
    catalog: dict[str, Any] | None = None
    classic: tuple[str, str, str] | None = None  # module, build function, example function


@dataclass
class Draft:
    """A model record before its family packet assigns source ids."""

    row: Row
    source_case: str
    recipe: dict[str, Any]
    sources: list[dict[str, Any]] = field(default_factory=list)
    hf: str | None = None
    delta: dict[str, Any] = field(default_factory=dict)
    cites: str | None = None  # a title the code names without a link
    same_as_june: bool = True  # whether June's measured peak describes this recipe
    give_up: tuple[str, str] | None = None  # (class, reason) when no faithful recipe exists
    missing: str | None = None  # what a give-up lacks


class NoRecipe(RecordError):
    """The installed libraries cannot build the real model; the record gets a typed give-up."""

    def __init__(self, cls: str, reason: str, missing: str | None = None) -> None:
        super().__init__(f"{cls}: {reason}")
        self.cls, self.reason, self.missing = cls, reason, missing


# --- selection -------------------------------------------------------------------------


def harvest(zoo: str, names: Sequence[str] | None = None, *, data: DataLayout,
            write: bool = True) -> list[ModelRecord]:
    """Write packets and model records for a zoo's rows (all, or ``names``); return the records.

    ``zoo`` is a June catalog zoo (``timm``, ``torchvision``, ``classics-pytorch``, ...) or
    ``rung0``: the named models, the smoke rows that need only ``torch-main`` and the Slice A
    200. ``names`` may be roster names or stable ids.
    """

    return write_drafts([_draft(row, data) for row in _select(zoo, names, data)], data=data,
                        write=write)


def write_drafts(drafts: Sequence[Draft], *, data: DataLayout, write: bool = True,
                 rewrite: bool = True) -> list[ModelRecord]:
    """Group drafts by family, merge each family's packet and write packets and records.

    With ``rewrite`` false (``harvest tier``) an existing record whose content would change is
    kept as it is, left out of the result and named on stderr: a record changed at its revision
    would let stage 3 take a trace of the old content as current, so a deliberate change goes
    through a new revision instead.
    """

    measured = _Measured(data)
    families: dict[str, list[Draft]] = {}
    for draft in drafts:
        families.setdefault(_family_id(draft.row.family), []).append(draft)
    records: list[ModelRecord] = []
    for family, drafts in sorted(families.items()):
        packet, sids = _packet(family, drafts, data)
        family_records = [measured.apply(_record(d, family, code, data))
                          for d, code in zip(drafts, sids)]
        if not rewrite:
            family_records = [r for r in family_records if not _would_rewrite(r, data)]
        if write:
            _write_if_changed(packet, data.packet(family), keep=("written",))
            for record in family_records:
                _write_if_changed(record, data.model(record.id))
        records += family_records
    return records


def _would_rewrite(record: ModelRecord, data: DataLayout) -> bool:
    """Whether ``record`` would change the committed record of its id (named on stderr if so)."""

    path = data.model(record.id)
    if not path.exists():
        return False
    old, new = json.loads(path.read_text(encoding="utf-8")), to_dict(record)
    if old == new:
        return False
    changed = sorted(k for k in {*old, *new} if old.get(k) != new.get(k))
    sys.stderr.write(f"harvest: kept {record.id} as committed; the draft changes {changed} "
                     f"(a deliberate change needs a new revision)\n")
    return True


def _select(zoo: str, names: Sequence[str] | None, data: DataLayout) -> list[Row]:
    from menagerie.identity import load_stable_ids

    ids = load_stable_ids(data.root / "stable_ids.jsonl")
    keys = {sid: key for key, sid in ids.items()}
    with (data.root / "master_catalog.jsonl").open(encoding="utf-8") as handle:
        catalog = {(r["name"], r["zoo"], r["variant"]): r for r in map(json.loads, handle)}
    if zoo == "rung0":
        wanted = list(RUNG0_NAMED) + _smoke_ids(data) + list(_slice_a(data))
    else:
        wanted = [sid for key, sid in ids.items() if key[1] == zoo]
    if names:
        wanted = [sid for sid in wanted if sid in names or keys[sid][0] in names]
    rows, classics = [], None
    for sid in dict.fromkeys(wanted):
        name, row_zoo, variant = keys[sid]
        found = catalog.get((name, row_zoo, variant))
        if found is not None:
            rows.append(Row(sid, name, row_zoo, variant, found["family"], catalog=found))
            continue
        classics = classics if classics is not None else _classics_registry()
        if name not in classics:
            raise RecordError(f"{sid} {name!r} ({row_zoo}) is in neither the catalog nor classics")
        module, build, example, family = classics[name]
        rows.append(Row(sid, name, row_zoo, variant, family, classic=(module, build, example)))
    if not rows:
        raise RecordError(f"no rows for zoo {zoo!r} and names {list(names or [])}")
    return rows


def _smoke_ids(data: DataLayout) -> list[str]:
    with (data.root / "smoke_manifest.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return [r["stable_id"] for r in rows if r["expected_env"] == "base" and not r["synthetic"]]


def _slice_a(data: DataLayout) -> dict[str, dict[str, str]]:
    path = data.root / "regression" / "slice_a_200.tsv"
    if not path.exists():
        raise RecordError(f"{path} is missing; the environment lane freezes the Slice A 200 there")
    with path.open(encoding="utf-8") as handle:
        return {row["stable_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


class _Measured:
    """What the ledger measured, applied to a record: ``memory_gib`` is the forward's peak at
    the record's own revision (fix 8), and ``placement`` is stage 3's rule on the measured peaks
    (fix 10, ``stage3.sizing``; never a weight estimate), or the plan's ``PLACED``. A row with no
    measurement has neither; stage 3 measures its forward first."""

    def __init__(self, data: DataLayout) -> None:
        from menagerie.pipeline import stage3

        self.data, self.peaks, self.families = data, stage3.Peaks.read(data), None
        self.forwards: dict[tuple[str, str], float] = {}
        for path in sorted((data.root / "ledger").glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for row in map(json.loads, handle):
                    peak, key = row.get("forward_peak_rss_mb"), (row["id"], row.get("recipe_rev"))
                    if peak and key[1]:
                        self.forwards[key] = max(self.forwards.get(key, 0.0), round(peak / 1024, 2))

    def apply(self, record: ModelRecord) -> ModelRecord:
        if record.recipe is None:
            return record
        gib = self.forwards.get((record.id, record.recipe.rev))
        placement = PLACED.get(record.id) or self._placement(record)
        return replace(record, recipe=replace(record.recipe, memory_gib=gib, placement=placement))

    def _placement(self, record: ModelRecord) -> str | None:
        from menagerie.pipeline import flow, stage3

        peaks = self.peaks
        if not any(record.id in seen for seen in (peaks.forward, peaks.passed, peaks.capped)):
            return None
        if self.families is None:
            self.families = stage3._families(self.data)
        forward = peaks.forward.get(record.id)
        projected = stage3.sizing(record, peaks, self.families).project(
            forward / 1024 if forward else None)
        found = flow.placement_for(projected) if projected else None
        return None if found == "worker" else found


def _classics_registry() -> dict[str, tuple[str, str, str, str]]:
    """Return ``{canonical name: (module, build, example, family)}`` (imports every classic)."""

    package = sys.modules.get(CLASSICS)
    if package is not None and not hasattr(package, "CLASSICS"):
        package.__spec__.loader.exec_module(package)  # the bare package from import_module
    from menagerie.classics import CLASSICS as registry

    return {name: (e["build"].__module__, e["build"].__name__, e["example_input"].__name__,
                   e["family"]) for name, e in registry.items()}


# --- recipes ---------------------------------------------------------------------------


def _draft(row: Row, data: DataLayout) -> Draft:
    try:
        return _translate(row, data)
    except NoRecipe as missing:
        call = (_parse_call(_code(row)) or ("",))[0]
        _, library = _library_facts(call, [], {}, row.family) if call else (None, None)
        return Draft(row, "library", {}, [library] if library else [],
                     give_up=(missing.cls, missing.reason), missing=missing.missing)


def _code(row: Row) -> str:
    recipe = row.catalog["recipe"] if row.catalog else {}
    return recipe.get("expr") or recipe.get("code") or recipe.get("body") or ""


# A parser of June recipe code, and a check of a library call's arguments that returns the
# arguments to build with and the member's delta (the library tier's, ``harvest_tiers``).
Parse = Callable[[str], "tuple[str, list, dict, dict, dict[str, list[str]]] | None"]
Vet = Callable[[Row, str, list, dict], "tuple[list, dict, dict[str, Any]]"]


def _translate(row: Row, data: DataLayout, parse: Parse | None = None,
               vet: Vet | None = None) -> Draft:
    if row.classic:
        return _classic_draft(row, data, *row.classic)
    code = _code(row)
    spec = row.catalog.get("input") or {}
    cfg = _timm_cfg(row.name) if row.zoo == "timm" else None
    if cfg is not None:  # the registry's own model at its default input size
        spec = {"kind": "tensor", "spec": {"shape": [1, *cfg.input_size], "dtype": "float32"}}
        draft = _library_draft(row, data, "timm.create_model", [row.name], {"pretrained": False},
                               {}, spec)
        draft.hf, draft.same_as_june = cfg.hf_hub_id, "create_model" in code
        return draft
    model_type = _hf_model_type(row)
    parsed = (parse or _parse_call)(code)
    if model_type:  # the library's reference configuration for the row's model type
        auto = row.zoo.partition(":")[2] or (
            parsed[0].split(".")[1] if parsed and parsed[0].startswith("transformers.Auto")
            else "AutoModel")
        config = {"model_type": model_type}
        draft = _library_draft(row, data, f"transformers.{auto}.from_config", [], {}, config, spec)
        draft.same_as_june = bool(parsed) and parsed[3] == config
        return draft
    yolo = YOLO.search(code)
    if yolo:
        task = yolo.group(1).removesuffix(".yaml").rpartition("-")[2]
        call = f"ultralytics.nn.tasks.{YOLO_TASKS.get(task, 'DetectionModel')}"
        return _library_draft(row, data, call, [yolo.group(1)], {"verbose": False}, {}, spec)
    if parsed is None:
        return _adapter_draft(row, data, spec)
    call, args, kwargs, config, imported = parsed
    if call.startswith(f"{CLASSICS}."):
        module, _, build = call.rpartition(".")
        example = next((n for n in imported.get(module, ()) if n.startswith("example")), None)
        return _classic_draft(row, data, module, build, example, args, spec)
    args, kwargs, delta = vet(row, call, args, kwargs) if vet else (args, kwargs, {})
    draft = _library_draft(row, data, call, args, kwargs, config, spec)
    draft.delta = delta
    return draft


def _classic_draft(row: Row, data: DataLayout, module: str, build: str, example: str | None,
                   args: Sequence[Any] = (), spec: Mapping[str, Any] | None = None) -> Draft:
    link, title = classic_paper(module, build, family=_family_id(row.family))
    sources = [_code_source(_module_file(module))]
    if link:
        sources.append(_paper_source(link, title))
    recipe = {"kind": "classic", "adapter": f"{module}:{build}", "args": list(args)}
    if example:
        recipe["input"] = contract(f"example:{module}:{example}", None, "the module's example input")
    else:
        recipe["input"] = _input(spec or {}, recipe, data)
    return Draft(row, "classic", recipe, sources, cites=title)


PAPER_LINK = re.compile(r"https?://\S+?\.pdf|arxiv\.org/abs/\d{4}\.\d{4,5}")
QUOTED_TITLE = re.compile(r"[\"“]([^\"”]{8,}?)\.?[\"”]")


def classic_paper(module: str, build: str, *, family: str | None = None
                  ) -> tuple[str | None, str | None]:
    """The paper link and quoted title a classic's own docs name, or ``(None, None)``.

    The build function's docstring first, else the module docstring, each only when it names
    exactly one paper: a module several families share names each of their papers, and its
    first link is not this family's (ri_ts_graph_hash's four hashing methods all took DPSH's).
    """

    override = _classic_papers().get(family or "")
    if override:
        return override
    tree = ast.parse(_module_file(module).read_text(encoding="utf-8"))
    builder = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                    and n.name == build), None)
    for text in ((ast.get_docstring(builder) or "") if builder else "", _module_doc(module)):
        links = list(dict.fromkeys(PAPER_LINK.findall(text)))
        if len(links) == 1:
            title = QUOTED_TITLE.search(text)
            return links[0], title.group(1) if title else None
        if links:
            return None, None
    return None, None


@functools.cache
def _classic_papers() -> dict[str, tuple[str, str]]:
    """Family-specific corrections are metadata, independent of a recipe's adapter revision."""

    path = Path(__file__).resolve().parents[1] / "data" / "classics-papers.tsv"
    with path.open(encoding="utf-8") as handle:
        return {r["family"]: (r["url"], r["title"]) for r in csv.DictReader(handle, delimiter="\t")}


def _library_draft(row: Row, data: DataLayout, call: str, args: list, kwargs: dict,
                   config: dict, spec: Mapping[str, Any]) -> Draft:
    if config and set(config) != {"model_type"}:
        # A June config with sizes is a shrunken stand-in (JMT, 2026-10-03): build the
        # library default, and give up when that is not the published model.
        config = {"model_type": config["model_type"]}
        try:
            _weights_gib({"kind": "library", "call": call, "config": config}, data, strict=True)
        except Exception as exc:
            raise NoRecipe("research_gap:hf-config", f"the June recipe shrank the config and "
                           f"{call}(default {config['model_type']} config) fails: "
                           f"{type(exc).__name__}: {str(exc)[:120]}",
                           "the model's Hub config.json") from exc
    docs, library = _library_facts(call, args, config, row.family)
    recipe = {"kind": "library", "call": call, "args": args, "kwargs": kwargs, "config": config}
    recipe["input"] = _input(spec, recipe, data)
    return Draft(row, "library", recipe, [library, *_paper_sources(docs)], hf=HF_NAMED.get(row.id))


def _adapter_draft(row: Row, data: DataLayout, spec: Mapping[str, Any]) -> Draft:
    path = data.adapter(_family_id(row.family))
    if not path.exists():
        raise RecordError(f"{row.id} {row.name!r}: the June recipe is code, not a call; "
                          f"write {path} first")
    meta = _load_file(path).META
    sources = []
    if meta.get("library"):
        lib = meta["library"]
        sources.append({"role": "library", "dist": lib["dist"], "version": _version(lib["dist"]),
                        "module": lib["module"], "symbol": lib["symbol"]})
    sources.append({"role": "code", "framework": "pytorch", **meta["code"]} if meta.get("code")
                   else _code_source(path))
    if meta.get("paper"):
        sources.append(_paper_source(meta["paper"]["url"], meta["paper"]["title"]))
    recipe = {"kind": "adapter", "adapter": path.relative_to(data.root).as_posix()}
    recipe["input"] = _input(spec, recipe, data)
    return Draft(row, meta["source_case"], recipe, sources)


def _parse_call(code: str, refs: bool = False) -> tuple[str, list, dict, dict,
                                                       dict[str, list[str]]] | None:
    """Translate ``[imports;] model = f(literals)[; model.eval()]`` into a call, or ``None``.

    With ``refs``, an argument may also name an imported object (``spiking_neuron=neuron.LIFNode``),
    kept as ``{"$ref": "<dotted name>"}``, or call one with such arguments
    (``surrogate_function=surrogate.ATan()``), kept as ``{"$call": name, "args", "kwargs"}``; both
    are resolved when the model is built. ``dict(k=v)`` and list or tuple displays of such
    values are kept as their values. The audit (``june_audit``) still proves every nested call.
    """

    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    aliases: dict[str, str] = {}
    imported: dict[str, list[str]] = {}
    target: ast.expr | None = None
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                aliases[alias.asname or head] = alias.name if alias.asname else head
        elif isinstance(node, ast.ImportFrom) and node.module:
            aliases.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names})
            imported.setdefault(node.module, []).extend(a.name for a in node.names)
        elif isinstance(node, (ast.Assign, ast.Expr)) and target is None:
            target = node.value
        elif not (isinstance(node, ast.Expr) and _is_eval(node.value)):
            return None
    if not isinstance(target, ast.Call) or any(k.arg is None for k in target.keywords):
        return None
    try:
        call = _dotted(target.func, aliases)
        args = [_literal(a, aliases if refs else None) for a in target.args if not _is_config(a)]
        kwargs = {k.arg: _literal(k.value, aliases if refs else None) for k in target.keywords}
        configs = [_config(a, aliases) for a in target.args if _is_config(a)]
    except (ValueError, TypeError, AttributeError):
        return None
    if len(configs) > 1:
        return None
    return call, args, kwargs, configs[0] if configs else {}, imported


def _literal(node: ast.expr, aliases: dict[str, str] | None) -> Any:
    try:
        return ast.literal_eval(node)
    except ValueError:
        if aliases is None:
            raise
    if isinstance(node, (ast.Name, ast.Attribute)):
        return {REF: _imported_name(node, aliases)}
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_literal(e, aliases) for e in node.elts]
        return items if isinstance(node, ast.List) else tuple(items)
    if isinstance(node, ast.Call) and not any(k.arg is None for k in node.keywords):
        return _call_value(node, aliases)
    raise ValueError(f"not a literal or an imported object: {ast.unparse(node)[:80]}")


def _imported_name(node: ast.expr, aliases: dict[str, str]) -> str:
    """The dotted name of an imported object; a name the code binds itself is refused."""

    head = node
    while isinstance(head, ast.Attribute):
        head = head.value
    if not isinstance(head, ast.Name) or head.id not in aliases:
        raise ValueError(f"not an imported object: {ast.unparse(node)[:80]}")
    return _dotted(node, aliases)


def _call_value(node: ast.Call, aliases: dict[str, str]) -> Any:
    """``dict(k=v)`` as its dict; a call of an imported library name as ``$call``."""

    keywords = {k.arg: _literal(k.value, aliases) for k in node.keywords}
    if isinstance(node.func, ast.Name) and node.func.id == "dict" and "dict" not in aliases:
        if node.args:
            raise ValueError("dict(...) with positional arguments")
        return keywords
    called = _imported_name(node.func, aliases)
    if called.split(".")[0] == "torch":  # a hand-written layer is a stand-in, not a component
        raise ValueError(f"nested torch call: {called}")
    return {CALL: called, "args": [_literal(a, aliases) for a in node.args], "kwargs": keywords}


def _is_eval(node: ast.expr) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "eval" and not node.args)


def _is_config(node: ast.expr) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name == "for_model" or name.endswith("Config")


def _dotted(node: ast.expr, aliases: dict[str, str]) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        raise TypeError("callee is not a dotted name")
    # Bare Auto* names were in scope in the June runner without an import.
    head = aliases.get(node.id) or (f"transformers.{node.id}" if node.id.startswith("Auto")
                                    else node.id)
    return ".".join([head, *reversed(parts)])


def _config(node: ast.Call, aliases: dict[str, str]) -> dict[str, Any]:
    """Return a Hugging Face config call as ``{"model_type": ..., **kwargs}``."""

    kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
    name = _dotted(node.func, aliases)
    if name.endswith(".for_model"):
        return {"model_type": ast.literal_eval(node.args[0]), **kwargs}
    return {"model_type": _resolve(name).model_type, **kwargs}


def _input(spec: Mapping[str, Any], recipe: dict[str, Any], data: DataLayout) -> RecipeInput:
    """Choose a preset for a June input spec: the canonical image or text where they fit."""

    tensor = spec.get("spec") if spec.get("kind") == "tensor" else None
    if not tensor:
        shape = spec.get("sentinel_shape") or [1]
        return contract("randn", shape, "sentinel input; the model ignores its argument")
    shape, dtype = list(tensor["shape"]), str(tensor.get("dtype", "float32"))
    if dtype in INTEGER_DTYPES:
        vocab = _vocab(recipe, data)
        if vocab and vocab >= 257 and len(shape) == 2:
            return SEQ2SEQ_INPUT if _encoder_decoder(recipe) else contract("text", shape)
        bound = vocab or 2
        return contract(f"randint:{bound}", shape, f"random token ids below {bound}")
    if len(shape) == 4 and shape[1] == 3 and min(shape[2:]) >= 32:
        return contract("image", shape)
    return contract("randn", shape, "random float32 at the June input shape")


def _hf_model_type(row: Row) -> str | None:
    if not row.zoo.startswith(("huggingface", "transformers")):
        return None
    from transformers import CONFIG_MAPPING

    name = row.name.partition(":")[0]
    return name if name in CONFIG_MAPPING else None


def _weights_gib(recipe: dict[str, Any], data: DataLayout, strict: bool = False) -> float:
    """Return the fp32 weight size of a recipe's model, built on the meta device.

    Weights alone above the worker cap make a row ``cluster`` without a measured peak. A model
    that does not build here (a missing optional dependency) counts as 0 unless ``strict``;
    the runner then types its failure.
    """

    from menagerie.pipeline.meta_build import is_meta_error, probe_gib

    try:
        build = _builder(recipe, data)
        try:
            with on_device("meta"):
                model = build()
        except Exception as exc:
            if not is_meta_error(exc):
                raise
            # A constructor that reads tensor values is sized by the probe (meta_build).
            return probe_gib(build)
        return sum(p.numel() for p in model.parameters()) * 4 / 2**30
    except Exception:
        if strict:
            raise
        return 0.0


def _vocab(recipe: dict[str, Any], data: DataLayout) -> int | None:
    """Return the size of the model's first embedding table, built on the meta device."""

    import torch

    with on_device("meta"):
        model = _builder(recipe, data)()
    if not isinstance(model, torch.nn.Module):
        return None
    sizes = [m.num_embeddings for m in model.modules() if isinstance(m, torch.nn.Embedding)]
    return sizes[0] if sizes else None


# --- packets and records ---------------------------------------------------------------


def _family_id(family: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", family.lower()).strip("-._")
    return f"f-{slug or 'unnamed'}"


def _packet(family: str, drafts: list[Draft], data: DataLayout) -> tuple[Packet, list[list[str]]]:
    """Merge the drafts into the family's harvester packet; return it and each draft's code sids."""

    path = data.packet(family)
    old = read_json(Packet, path) if path.exists() else None
    if old and not old.by.startswith("harvester@"):
        raise RecordError(f"{path} was written by {old.by}; the harvester does not edit it")
    sources = [to_dict(s) for s in old.sources] if old else []

    def key(source: dict[str, Any]) -> str:
        # What a fetch adds to a source (its blobs) does not make it another source.
        return json.dumps({k: v for k, v in source.items()
                           if k not in FETCHED_FIELDS and k[0] != "_"}, sort_keys=True)

    index = {key(s): s["sid"] for s in sources}
    by_sid = {s["sid"]: s for s in sources}
    licensed = {by_sid[s["of"]].get("dist") for s in sources if s["role"] == "license"}
    titles: dict[str, str] = {}

    def add(source: dict[str, Any]) -> str:
        if key(source) not in index:
            index[key(source)] = f"s{len(sources) + 1}"
            sources.append({"sid": index[key(source)],
                            **{k: v for k, v in source.items() if k[0] != "_"}})
        sid = index[key(source)]
        if source.get("_title"):
            titles.setdefault(sid, source["_title"])
        if source["role"] == "library" and source["dist"] not in licensed:
            licensed.add(source["dist"])
            add({"role": "license", "of": sid, **_license(source["dist"])})
        return sid

    code_sids = [[add(s) for s in d.sources if s["role"] in ("library", "code")] for d in drafts]
    for draft in drafts:
        for source in _admitted_sources(old, draft.sources):
            add(source)
    built = [from_dict(Source, s) for s in sources]
    members = {m.id: m for m in old.members} if old else {}
    for draft in drafts:
        delta = ", ".join(f"{k} {v}" for k, v in draft.delta.items())
        members.setdefault(draft.row.id, Member(id=draft.row.id, delta=delta))
    availability = _packet_availability(old, built)
    unfound = availability.paper != "found" or availability.license != "found"
    places = dict.fromkeys(f"{s.module or s.url} docstrings" for s in built
                           if s.role in ("library", "code")) or {"the June catalog recipe": None}
    search = Search(places=tuple(places), note="harvester: what the library or module names; "
                    "2a searches the rest") if unfound else Search()
    packet = Packet(
        family=family, written=now(), by=_by(built), members=tuple(members.values()),
        identity=Identity(canonical=drafts[0].row.family), sources=tuple(built),
        availability=availability,
        **_carried(old, _citation(built, titles, next((d.cites for d in drafts if d.cites), None)),
                   search),
        code_order=tuple(s.sid for s in built if s.role in ("library", "code")),
        notes_for_implementer=june_notes(list(members), data),
    )
    return packet, code_sids


def _packet_availability(old: Packet | None, sources: list[Source]) -> Availability:
    from menagerie.pipeline.bib_harvest import no_paper  # noqa: PLC0415

    availability = _availability(sources)
    if old and no_paper(old):
        return replace(availability, paper="not_found")
    return availability


def _admitted_sources(old: Packet | None, sources: Sequence[dict]) -> list[dict]:
    """Admit draft evidence while respecting paper decisions from earlier amendments."""

    return [s for s in sources if not (s["role"] == "paper" and _refused_paper(old, s))]


def _refused_paper(old: Packet | None, source: Mapping[str, Any]) -> bool:
    """A metadata amendment is authoritative over later extraction of the same stale docs."""

    if old is None or not any("citation" in a.changed for a in old.amendments):
        return False
    from menagerie.pipeline.bib_harvest import _ids, citation_held, held_sources  # noqa: PLC0415, SLF001

    url = source.get("url")
    accepted = {s.url for s in old.sources if s.role == "paper" and s.sid not in held_sources(old)}
    if old.citation and not citation_held(old):
        accepted_ids = {old.citation.doi}
        if old.citation.arxiv:
            accepted_ids.add(f"arXiv:{old.citation.arxiv}")
    else:
        accepted_ids = set()
    accepted_ids = (accepted_ids | {_ids(u) for u in accepted}) - {None}
    return url not in accepted and (_ids(url) is None or _ids(url) not in accepted_ids)


def june_notes(ids: Sequence[str], data: DataLayout) -> str:
    """The June catalog ``notes`` of these members, each distinct note once with its ids."""

    by_note: dict[str, list[str]] = {}
    for sid in ids:
        note = (_june_rows(data.root).get(sid) or {}).get("notes") or ""
        if note.strip():
            by_note.setdefault(" ".join(note.split()), []).append(sid)
    if not by_note:
        return ""
    if len(by_note) == 1:
        return f"June catalog notes: {next(iter(by_note))}"
    return "June catalog notes: " + "; ".join(f"{', '.join(i)}: {n}" for n, i in by_note.items())


@functools.cache
def _june_rows(root: Path) -> dict[str, dict[str, Any]]:
    """The June catalog rows by stable id (``stable_ids.jsonl`` maps the natural keys)."""

    from menagerie.identity import load_stable_ids

    ids = load_stable_ids(root / "stable_ids.jsonl")
    with (root / "master_catalog.jsonl").open(encoding="utf-8") as handle:
        rows = {(r["name"], r["zoo"], r["variant"]): r for r in map(json.loads, handle)}
    return {sid: rows[key] for key, sid in ids.items() if key in rows}


def _carried(old: Packet | None, citation: Citation | None, search: Search) -> dict[str, Any]:
    """What a rerun keeps from the old packet: its amendments, a citation one of them filled
    (from a bibliographic API or the paper's own page) over the harvester's docstring title, its
    citation when this run found none (a rerun replaces a citation, never drops one), and
    its `search`, which outlives the paper and license research then found (new places join it)."""

    if old is None:
        return {"citation": citation, "amendments": (), "search": search}
    amended = any("citation" in a.changed for a in old.amendments)
    if old.search != Search():
        search = Search(queries=old.search.queries,
                        places=tuple(dict.fromkeys(old.search.places + search.places)),
                        note=old.search.note or search.note)
    return {"citation": old.citation if amended or citation is None else citation,
            "amendments": old.amendments, "search": search}


def _by(sources: list[Source]) -> str:
    libs = sorted({f"{s.dist}-{s.version}" for s in sources if s.role == "library"})
    return "harvester@" + ("+".join(libs) if libs else "menagerie")


def _availability(sources: list[Source]) -> Availability:
    roles = {s.role for s in sources}
    licensed = any(s.role == "license" and s.spdx != "unknown" for s in sources)
    return Availability(
        paper="found" if "paper" in roles else "not_found",
        code="found" if roles & {"library", "code"} else "not_found",
        license="found" if licensed else "not_found",
        description="sufficient" if roles & {"library", "code"} else "insufficient",
    )


def _citation(sources: list[Source], titles: dict[str, str], cites: str | None) -> Citation | None:
    paper = next((s for s in sources if s.role == "paper"), None)
    if paper is None or paper.sid not in titles:
        return Citation(title=cites) if cites else None
    arxiv = ARXIV.search(paper.url or "")
    return Citation(title=titles[paper.sid], arxiv=arxiv.group(1) if arxiv else None,
                    from_=paper.sid)


def _record(draft: Draft, family: str, code_sids: list[str], data: DataLayout) -> ModelRecord:
    path = data.model(draft.row.id)
    old = read_json(ModelRecord, path) if path.exists() else None
    if draft.give_up:
        cls, reason = draft.give_up
        same = old is not None and old.give_up is not None and old.give_up.cls == cls
        # An unchanged class keeps its time and author; an agent's give-up is kept whole.
        at, by = (old.give_up.at, old.give_up.by) if same else (now(), "script")
        if same and by == "agent":
            return replace(old, name=draft.row.name, zoo=draft.row.zoo, variant=draft.row.variant,
                           family=family, source_case=draft.source_case)
        return ModelRecord(
            id=draft.row.id, name=draft.row.name, zoo=draft.row.zoo, variant=draft.row.variant,
            family=family, source_case=draft.source_case,
            give_up=GiveUp(cls=cls, reason=reason, at=at, missing=draft.missing, by=by),
        )
    fields = dict(draft.recipe, env=ENV, backend="torch", sources=code_sids, rev="0" * 12)
    recipe = to_dict(from_dict(Recipe, {k: v for k, v in fields.items() if v is not None}))
    recipe["rev"] = recipe_revision(recipe, _adapter_bytes(recipe, data), _lock(ENV))
    if old and old.recipe and old.recipe.modes and old.recipe.rev == recipe["rev"]:
        recipe["modes"] = to_dict(old.recipe.modes)
    return ModelRecord(
        id=draft.row.id, name=draft.row.name, zoo=draft.row.zoo, variant=draft.row.variant,
        family=family, source_case=draft.source_case, delta=dict(draft.delta),
        hf=HFRef(id=draft.hf) if draft.hf else None, recipe=from_dict(Recipe, recipe),
    )


def _lock(env: str) -> bytes:
    path = REPO / "envs" / env / "uv.lock"
    if not path.exists():
        raise RecordError(f"{path} is missing; recipe revisions need the environment lock")
    return path.read_bytes()


def _adapter_bytes(recipe: Mapping[str, Any], data: DataLayout) -> bytes | None:
    if recipe["kind"] == "classic":
        return _module_file(recipe["adapter"].partition(":")[0]).read_bytes()
    if recipe["kind"] == "adapter":
        return (data.root / recipe["adapter"]).read_bytes()
    return None


def _write_if_changed(record: Any, path: Path, keep: tuple[str, ...] = ()) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        new = to_dict(record)
        if {k: v for k, v in old.items() if k not in keep} == \
                {k: v for k, v in new.items() if k not in keep}:
            return
    write_json(record, path)
