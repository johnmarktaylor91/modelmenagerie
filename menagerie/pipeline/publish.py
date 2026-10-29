"""The site export: one record per publishable model, with variants, viewer refs and dated counts.

Owned by the publish lane. Rung 0 writes the export only; nothing is uploaded or deployed.

The export is the five files ``site/scripts/sync_site_data.py`` copies into ``site/site-data/``
(``models.jsonl``, ``facets.json``, ``funnel.json``, ``export_manifest.json``, ``catalog.csv``)
plus ``assets.tsv``, the store blobs the records point at, which ``stage_assets`` copies out
under their public keys (``<sha256><ext>``). Records follow ``ModelRecord`` in
``site/src/lib/catalog.ts``.

Selection: by default a model is exported when its research packet exists (JMT 2026-10-03:
research finishes before publishing) and either ``ledger.state`` is ``publishable`` or its
current trace is a passing structure-only capture with renders (the labelled hypothesis
tier). Explicit ``ids`` export those models whatever their validation state, which is how a
non-public preview is built; each still needs a current trace with a rendered default variant.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from menagerie.pipeline import (
    DataLayout, StoreLayout, citecheck, hash_harness, now, read_json, read_jsonl,
)
from menagerie.pipeline import bib_harvest, standin, trace_metrics
from menagerie.pipeline.records import (
    Citation,
    CitationLine,
    DescribeRecord,
    LedgerLine,
    LedgerRow,
    ModelRecord,
    Packet,
    RecordError,
    RenderArtifact,
    RenderRow,
    RosterRow,
    StoreIndexLine,
)

EXPORT_SCHEMA = "site-export/1"
DATA_FILES = ("models.jsonl", "facets.json", "funnel.json", "export_manifest.json", "catalog.csv")
# The 23 live categories (`site/site-data/facets.json`), kept so existing URLs and facets survive.
CATEGORIES = (
    "history", "vision_classification-backbone", "vision_segmentation",
    "vision_detection-tracking", "generative_diffusion-GAN-flow", "audio_speech",
    "graph_geometric", "NLP_LLM_text", "recsys_tabular", "video_action", "exotic_other",
    "time-series_dynamics", "vision_pose", "3D_geometry_NeRF_point-cloud",
    "scientific_molecular_protein_genomics", "multimodal", "OCR_document",
    "spiking_neuromorphic", "RL_robotics_control", "vision_depth-geometry",
    "scientific_physics_weather_PDE", "neuro_scientific-signals", "medical-imaging",
)
# Keyword to category, first match wins, so specific tasks come before their broad domain.
# A keyword matches a token that starts with it; keywords of three letters or fewer match exactly.
CATEGORY_KEYWORDS = (
    ("medical", "medical-imaging"), ("ocr", "OCR_document"), ("document", "OCR_document"),
    ("pose", "vision_pose"), ("depth", "vision_depth-geometry"),
    ("segment", "vision_segmentation"), ("detect", "vision_detection-tracking"),
    ("track", "vision_detection-tracking"), ("video", "video_action"),
    ("action", "video_action"), ("spiking", "spiking_neuromorphic"),
    ("neuromorphic", "spiking_neuromorphic"), ("neuro", "neuro_scientific-signals"),
    ("molecul", "scientific_molecular_protein_genomics"),
    ("protein", "scientific_molecular_protein_genomics"),
    ("genom", "scientific_molecular_protein_genomics"),
    ("physics", "scientific_physics_weather_PDE"), ("weather", "scientific_physics_weather_PDE"),
    ("pde", "scientific_physics_weather_PDE"), ("point", "3D_geometry_NeRF_point-cloud"),
    ("nerf", "3D_geometry_NeRF_point-cloud"), ("3d", "3D_geometry_NeRF_point-cloud"),
    ("graph", "graph_geometric"), ("multimodal", "multimodal"),
    ("diffusion", "generative_diffusion-GAN-flow"), ("gan", "generative_diffusion-GAN-flow"),
    ("generative", "generative_diffusion-GAN-flow"), ("audio", "audio_speech"),
    ("speech", "audio_speech"), ("series", "time-series_dynamics"),
    ("dynamics", "time-series_dynamics"), ("tabular", "recsys_tabular"),
    ("recsys", "recsys_tabular"), ("robot", "RL_robotics_control"),
    ("control", "RL_robotics_control"), ("rl", "RL_robotics_control"),
    ("text", "NLP_LLM_text"), ("nlp", "NLP_LLM_text"), ("llm", "NLP_LLM_text"),
    ("language", "NLP_LLM_text"), ("vision", "vision_classification-backbone"),
    ("image", "vision_classification-backbone"),
    ("classification", "vision_classification-backbone"),
)
HISTORY_BEFORE = 1990
# Public object-key suffix per render kind; tiles stay in the store until the site shows them.
EXTENSIONS = {
    "svg": ".svg", "pdf": ".pdf", "thumb": ".webp", "html": ".html",
    "netron": ".netron.json", "model_explorer": ".model-explorer.json",
}
MONSTER_SVG_BYTES = 5 * 1024 * 1024
MONSTER_OPS = 2000
VIS_LABELS = {"unrolled": "Unrolled", "rolled": "Rolled"}
COLLAPSE_LABELS = {"none": "full", "auto": "collapsed (auto)", "max": "collapsed (max)"}


@dataclass(frozen=True)
class Current:
    """The rows one site record is built from: the current trace, its render and the state."""

    trace: LedgerRow
    render: RenderRow
    state: str
    review: LedgerRow | None

    @property
    def validated(self) -> bool:
        return self.state != "not_publishable"


def export_site(*, data: DataLayout, out_dir: Path, ids: Sequence[str] | None = None,
                as_of: str | None = None) -> Path:
    """Write the site export for ``ids`` (default: every publishable model); return its path."""

    from menagerie.pipeline import ledger, render, roster

    as_of = as_of or now()[:10]
    tl, renderer = torchlens_pin(), render.renderer_rev()
    models = {m.id: m for m in _model_records(data)}
    rows_by_id: dict[str, list[LedgerLine]] = defaultdict(list)
    for row in ledger.read_all(data=data):
        rows_by_id[row.id].append(row)
    roster_rows = roster.read(data=data)
    duplicates = citecheck.duplicate_notes(models.values(), roster_rows)
    if duplicates:
        raise RecordError(f"cannot export: {duplicates}")

    # A recipe the stand-in check calls a substitute is refused even before a sweep types it.
    standins = standin.known(models.values(), (r for rs in rows_by_id.values() for r in rs),
                             data=data)

    def current(model: ModelRecord) -> Current | str:
        hit = standin.detect(model, standins)
        if hit is not None:
            return f"gave up: standin ({hit[1]})"
        return _current(model, rows_by_id.get(model.id, []), tl=tl, renderer=renderer)

    selected = _explicit(ids, models, current) if ids else {
        m.id: found for m in models.values()
        if data.packet(m.family).exists() and isinstance(found := current(m), Current)
        and (found.state == "publishable" or _structure_only(found.trace))
    }
    context = _Context.load(data, models=models,
                           roster={r.stable_id: r for r in roster_rows if r.stable_id})
    slugs = assign_slugs(models.values())
    records = [
        site_record(models[i], selected[i], slug=slugs[i], context=context,
                    default=render.default_variant)
        for i in sorted(selected, key=lambda i: slugs[i])
    ]
    roster_names = sum(1 for r in roster_rows if r.state == "active")
    funnel, manifest = _counts(records, roster_names=roster_names, as_of=as_of, tl=tl,
                               renderer=renderer, basis=hash_harness.headline_basis(data))
    _write_export(Path(out_dir), records, funnel, manifest)
    trace_metrics.write_table(Path(out_dir), [
        trace_metrics.metrics_row(i, selected[i].trace, found) for i in sorted(selected)
        if (found := trace_metrics.summary(selected[i].trace, context.store)) is not None])
    return Path(out_dir)


def _current(model: ModelRecord, rows: Sequence[LedgerLine], *, tl: str,
             renderer: str) -> Current | str:
    """Return the rows a model's record is built from, or why it has none."""

    from menagerie.pipeline import ledger

    if model.recipe is None:
        return "no recipe"
    rev = model.recipe.rev
    # A substitute is never published under the real name (decisions.md, 2026-10-03): a recipe
    # once typed `standin:*` stays refused, whatever give-up a later lane records.
    if (model.give_up is not None and model.give_up.cls.startswith("standin")) or any(
            isinstance(r, LedgerRow) and r.outcome == "giveup" and r.recipe_rev == rev
            and (r.cls or "").startswith("standin") for r in rows):
        return "gave up: standin (a stand-in recipe)"
    # The full capture that decides the verdict; without a saved one, a structure-only one. A
    # structure-only giant that no save level accepts has no `.tlspec`: its renders were drawn in
    # process and are keyed on its summary sha.
    trace = ledger.deciding_trace(model.id, rows, recipe_rev=rev, tl=tl)
    if trace is None or trace.trace is None:
        trace = _latest(rows, lambda r: isinstance(r, LedgerRow) and r.stage == "trace"
                        and r.capture == "structure_only" and r.recipe_rev == rev
                        and r.tl == tl and (r.trace or r.summary) is not None)
    if trace is None:
        return "no current trace"
    key = trace.trace or trace.summary
    render_row = _latest(rows, lambda r: isinstance(r, RenderRow) and r.trace == key
                         and r.outcome in ("pass", "partial"))
    if render_row is None:
        return "no render of the current trace"
    state = ledger.state(model.id, rows, recipe_rev=rev, tl=tl, renderer_rev=renderer)
    review = _latest(rows, lambda r: isinstance(r, LedgerRow) and r.stage == "review"
                     and r.recipe_rev == rev and r.outcome != "infra")
    return Current(trace, render_row, state, review)


def _explicit(ids: Sequence[str], models: Mapping[str, ModelRecord],
              current: Any) -> dict[str, Current]:
    """Return the named models' rows; refuse the export if any of them cannot be built."""

    found = {i: current(models[i]) if i in models else "no model record" for i in ids}
    problems = {i: why for i, why in found.items() if isinstance(why, str)}
    if problems:
        raise RecordError(f"cannot export {problems}")
    return found


def _counts(records: Sequence[Mapping[str, Any]], *, roster_names: int, as_of: str, tl: str,
            renderer: str, basis: Mapping[str, Any] | None = None,
            ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the dated funnel counts and the export manifest.

    Distinct architectures count on the headline hash ``basis`` (``hash_harness.headline_basis``):
    the legacy ``graph_shape_hash`` until the hash harness has passed.
    """

    basis = basis or {"basis": "graph_shape_hash", "schema": None, "backends": []}
    validated = [r for r in records if r["forward_validated"]]
    structure_only = sum(1 for r in records if r["capture"] == "structure_only")
    keys = {hash_harness.dedup_key(r["hashes"], r["graph_shape_hash"], basis,
                                   r.get("backend", "torch")) for r in validated}
    funnel = {
        "as_of": as_of,
        "roster_names": roster_names,
        "validated_records": len(validated),
        "structure_only_records": structure_only,
        "distinct_architectures": len(keys - {""}),
        "distinct_basis": hash_harness.basis_label(basis),
    }
    if "%" in json.dumps(funnel):
        raise RecordError("the funnel carries a percent sign; counts carry their date only")
    manifest = {
        "schema_version": EXPORT_SCHEMA,
        "export_version": as_of,
        "as_of": as_of,
        "counts": {
            "models": len(records),
            "validated": len(validated),
            "structure_only": structure_only,
            "preview_unvalidated": len(records) - len(validated) - structure_only,
            "thumbnail_fallbacks": sum(v["thumb"]["url"].endswith(".svg")
                                       for r in records for v in r["variants"]),
            "monster_graphs": sum(r["monster_graph"] for r in records),
        },
        "torchlens_version": tl,
        "renderer_version": renderer,
    }
    return funnel, manifest


def torchlens_pin() -> str:
    """Return the TorchLens version pinned in the repository's ``pyproject.toml``."""

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    pins = re.findall(r'^\s*"torchlens==([^"]+)"', text, re.MULTILINE)
    if len(pins) != 1:
        raise RecordError(f"{pyproject}: expected one torchlens== pin, found {pins}")
    return pins[0]


def _model_records(data: DataLayout) -> list[ModelRecord]:
    return [read_json(ModelRecord, p) for p in sorted(data.root.glob("models/*/*.json"))]


def _latest(rows: Iterable[LedgerLine], keep: Any) -> Any:
    """Return the kept row with the latest ``at``, the later one in ledger order on a tie."""

    kept = [r for r in rows if keep(r)]
    return max(reversed(kept), key=lambda r: r.at) if kept else None


def _structure_only(trace: LedgerRow) -> bool:
    return trace.capture == "structure_only" and trace.outcome == "pass"


@dataclass(frozen=True)
class _Context:
    """Everything a site record reads besides its model's own rows."""

    data: DataLayout
    roster: Mapping[str, RosterRow]
    packets: Mapping[str, Packet]
    describes: Mapping[str, DescribeRecord]
    citations: Mapping[str, CitationLine]
    blob_bytes: Mapping[str, int]
    store: StoreLayout | None

    @classmethod
    def load(cls, data: DataLayout, *, models: Mapping[str, ModelRecord],
             roster: Mapping[str, RosterRow]) -> _Context:
        families = {m.family for m in models.values()}
        packets = {f: read_json(Packet, data.packet(f))
                   for f in families if data.packet(f).exists()}
        describes = {f: read_json(DescribeRecord, data.describe(f))
                     for f in families if data.describe(f).exists()}
        citations: dict[str, CitationLine] = {}
        for shard in sorted(data.root.glob("cite/*.jsonl")):
            citations.update((line.family, line) for line in read_jsonl(CitationLine, shard))
        blob_bytes = {line.sha256: line.bytes
                      for shard in sorted(data.root.glob("store-index/*.jsonl"))
                      for line in read_jsonl(StoreIndexLine, shard)}
        try:
            store = StoreLayout.from_env()
        except RecordError:
            store = None
        return cls(data, roster, packets, describes, citations, blob_bytes, store)


def assign_slugs(models: Iterable[ModelRecord]) -> dict[str, str]:
    """Return a slug per model id, decided over every model record so a subset never moves one.

    The slug is the name; a name shared by several records adds the zoo, then the stable id.
    """

    models = list(models)
    bases = {m.id: _slugify(m.name) for m in models}
    with_zoo = {m.id: f"{bases[m.id]}-{_slugify(m.zoo)}" for m in models}
    base_counts, zoo_counts = Counter(bases.values()), Counter(with_zoo.values())
    return {
        m.id: bases[m.id] if base_counts[bases[m.id]] == 1
        else with_zoo[m.id] if zoo_counts[with_zoo[m.id]] == 1
        else f"{with_zoo[m.id]}-{m.id}"
        for m in models
    }


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "model"


def category_for(describe: DescribeRecord | None, roster_row: RosterRow | None) -> str:
    """Map the describe domain and field (else the roster's domain hint) onto a live category."""

    era = (describe.era if describe else None) or (roster_row.era if roster_row else None)
    year = re.search(r"\d{4}", era or "")
    if year and int(year.group()) < HISTORY_BEFORE:
        return "history"
    keys = ("domain", "field", "subfield", "task", "modality")
    texts = [t for t in [*(getattr(describe, k) for k in keys if describe),
                         roster_row.domain if roster_row else None] if t]
    exact = [t.replace("/", "_") for t in texts if t.replace("/", "_") in CATEGORIES]
    return exact[0] if exact else _keyword_category(" ".join(texts))


def _keyword_category(text: str) -> str:
    tokens = re.split(r"[^a-z0-9]+", text.lower())

    def hit(keyword: str) -> bool:
        return any(t == keyword if len(keyword) <= 3 else t.startswith(keyword) for t in tokens)

    return next((category for keyword, category in CATEGORY_KEYWORDS if hit(keyword)),
                "exotic_other")


def asset(sha256: str, kind: str, size: int) -> dict[str, Any]:
    """Return an ``AssetRef``; the site rewrites the URL to R2 or ``/assets/<key>``."""

    return {"url": f"/export/assets/{sha256}{EXTENSIONS[kind]}", "sha256": sha256, "bytes": size}


def variants_of(render_row: RenderRow, *, n_ops: int | None) -> list[dict[str, Any]]:
    """Return one ``ModelVariant`` per variant with an SVG; a missing thumbnail falls back to it."""

    by_variant: dict[str, dict[str, RenderArtifact]] = defaultdict(dict)
    for item in render_row.renders:
        if item.variant:
            by_variant[item.variant][item.kind] = item
    variants = []
    for key, kinds in sorted(by_variant.items(), key=lambda kv: _variant_order(kv[0])):
        if "svg" not in kinds:
            continue
        svg = kinds["svg"]
        thumb = kinds.get("thumb", svg)
        vis_mode, collapse, *train = key.split("-")
        label = f"{VIS_LABELS[vis_mode]}, {COLLAPSE_LABELS[collapse]}"
        label += ", train mode" if train else ""
        variants.append({
            "key": key, "vis_mode": vis_mode, "collapse": collapse, "label": label,
            "mode": "train" if train else "eval",
            "svg": asset(svg.sha256, "svg", svg.bytes),
            "thumb": asset(thumb.sha256, thumb.kind, thumb.bytes),
            "pdf": asset(kinds["pdf"].sha256, "pdf", kinds["pdf"].bytes)
            if "pdf" in kinds else None,
            "monster": "tiles" in kinds or svg.bytes > MONSTER_SVG_BYTES
            or (n_ops or 0) > MONSTER_OPS,
        })
    return variants


def _variant_order(key: str) -> tuple[int, int, int]:
    vis_mode, collapse, *train = key.split("-")
    return (bool(train), ("unrolled", "rolled").index(vis_mode),
            ("none", "auto", "max").index(collapse))


def recipe_text(model: ModelRecord) -> str:
    """Render the recipe call as one line of Python."""

    recipe = model.recipe
    if recipe is None:
        return ""
    if recipe.kind == "library":
        args = [repr(a) for a in recipe.args] + [f"{k}={v!r}" for k, v in recipe.kwargs.items()]
        return f"{recipe.call}({', '.join(args)})"
    config = f"VARIANTS[{model.variant!r}]" if model.variant else "{}"
    return f"{recipe.adapter}: build({config})"


def site_record(model: ModelRecord, current: Current, *, slug: str, context: _Context,
                default: Any) -> dict[str, Any]:
    """Build one site ``ModelRecord`` from the model's record, packet, describe and current rows."""

    packet = context.packets.get(model.family)
    # A family not yet described reads as an empty describe record: every field unknown.
    describe = context.describes.get(model.family) or DescribeRecord(family=model.family,
                                                                     by="none")
    roster_row = context.roster.get(model.id)
    return {
        "stable_id": model.id,
        "slug": slug,
        "display_name": model.name,
        "category": category_for(describe, roster_row),
        "family_normalized": packet.identity.canonical if packet else model.family[2:],
        "domain": describe.domain or (roster_row.domain if roster_row else None) or "unknown",
        "zoo": model.zoo,
        "source_zoo": _library_label(packet) or model.zoo,
        "source_license": _license(model, packet),
        "source_url": _source_url(model, packet),
        "paper_url": _paper_url(packet),
        "render_status": "rendered",
        "quarantine": False,
        "recipe_text": recipe_text(model),
        "rung": model.source_case,
        "forward_validated": current.validated,
        "fidelity": current.review.verdict if current.review else None,
        **_trace_fields(model, current, context, default=default),
        **_paper_fields(model, packet, context),
        **_era_fields(packet, roster_row),
        **_describe_fields(model, describe),
        "trace_stats": trace_metrics.flagship(trace_metrics.summary(current.trace, context.store)),
    }


def _trace_fields(model: ModelRecord, current: Current, context: _Context, *,
                  default: Any) -> dict[str, Any]:
    """The fields the current trace and its render row give."""

    trace, render_row = current.trace, current.render
    variants = variants_of(render_row, n_ops=trace.n_ops)
    is_recurrent = any(v["vis_mode"] == "rolled" for v in variants)
    keys = [v["key"] for v in variants]
    default_key = default(keys, is_recurrent=is_recurrent, n_ops=trace.n_ops or 0) if keys else None
    if default_key not in keys:
        raise RecordError(f"{model.id}: default variant {default_key!r} has no SVG in {keys}")
    trace_level = {a.kind: asset(a.sha256, a.kind, a.bytes)
                   for a in render_row.renders if a.variant is None}
    hashes = trace.hashes
    return {
        "validation_kind": trace.validation,
        "validation_state": trace.validation_state,
        "capture": trace.capture or "full",
        "input_tier": trace.input_boundary or "none",
        "n_ops": trace.n_ops,
        "graph_shape_hash": trace.graph_hash or "",
        "hashes": hashes and {k: v for k, v in (("skeleton", hashes.skeleton),
                                                ("architecture", hashes.architecture),
                                                ("model", hashes.model),
                                                ("schema", hashes.schema)) if v},
        "param_count": trace.params,
        "params_source": trace.params_source,
        "is_recurrent": is_recurrent,
        "default_variant": default_key,
        "variants": variants,
        "monster_graph": any(v["monster"] for v in variants),
        "interactive": trace_level.get("html"),
        "netron": trace_level.get("netron"),
        "model_explorer": trace_level.get("model_explorer"),
        "tlspec": _tlspec(trace, context),
        "backend": trace.backend or (model.recipe.backend if model.recipe else "torch"),
        "torchlens_version": trace.tl or "unknown",
        "renderer_version": render_row.renderer_rev,
    }


def _paper_fields(model: ModelRecord, packet: Packet | None,
                  context: _Context) -> dict[str, Any]:
    """The introducing paper's fields, from the packet's citation and the citation snapshot."""

    citation = _citation(packet)
    cited = context.citations.get(model.family)
    count = (cited.s2 if cited.from_ == "s2" else cited.openalex) if cited else None
    if citation is None:
        return {"paper_title": None, "year": None, "authors": [], "venue": None,
                "bibtex": None, "citations": None}
    return {
        "paper_title": citation.title,
        "year": citecheck.year_of(citation.first_public),
        "authors": list(citation.authors),
        "venue": citation.venue,
        "bibtex": _bibtex(model, packet, context),
        "citations": {"count": count, "as_of": cited.at[:10]} if cited and count is not None
        else None,
    }


def _citation(packet: Packet | None) -> Citation | None:
    """The packet's citation, unless it is a titleless fill waiting for its spot check."""

    if packet is None or bib_harvest.citation_held(packet):
        return None
    return packet.citation


def _era_fields(packet: Packet | None, roster_row: RosterRow | None) -> dict[str, Any]:
    """The roster's June ``era``, and ``year`` from it when it is a four-digit year and no
    citation gives one (``year_source`` says which)."""

    era = roster_row.era if roster_row else None
    citation = _citation(packet)
    cited = citecheck.year_of(citation.first_public) if citation else None
    if cited is not None:
        return {"era": era, "year": cited, "year_source": "citation"}
    june = int(era) if era and re.fullmatch(r"\d{4}", era) else None
    return {"era": era, "year": june, "year_source": "june_era" if june else None}


def _describe_fields(model: ModelRecord, describe: DescribeRecord) -> dict[str, Any]:
    member = describe.members.get(model.id)
    # Prose the mechanical check could not ground (`describe:ungrounded`) never reaches the site;
    # the taxonomy and keywords of such a record still do.
    prose = describe.grounding_state != "ungrounded"
    return {
        "tagline": describe.tagline if prose else None,
        "summary_markdown": describe.summary_markdown if prose else None,
        "key_contribution": describe.key_contribution if prose else None,
        "keywords": list(describe.keywords),
        "field": describe.field,
        "output_type": (member.output_type if member else None) or describe.output_type,
        "labs": list(describe.labs),
    }


def _sources(packet: Packet | None) -> dict[str, Any]:
    return {s.sid: s for s in packet.sources} if packet else {}


def _library_label(packet: Packet | None) -> str | None:
    lib = next((s for s in _sources(packet).values() if s.role == "library"), None)
    return f"{lib.dist} {lib.version}" if lib else None


def _code_order(model: ModelRecord, packet: Packet | None) -> list[str]:
    """The packet's code sources, the model's own first: a family packet can hold several
    libraries (ViT: timm, transformers and torchvision), each with its own license."""

    order = list(packet.code_order) if packet else []
    own = [sid for sid in (model.recipe.sources if model.recipe else ()) if sid in order]
    return own + [sid for sid in order if sid not in own]


def _source_url(model: ModelRecord, packet: Packet | None) -> str | None:
    """Return the model's first code source: its URL, or a library's PyPI release page."""

    sources = _sources(packet)
    for sid in _code_order(model, packet):
        source = sources[sid]
        if source.url:
            return source.url
        if source.role == "library":
            return f"https://pypi.org/project/{source.dist}/{source.version}/"
    return None


def _license(model: ModelRecord, packet: Packet | None) -> str:
    """Return the SPDX of the model's first code source's license, else of any, else unknown."""

    licenses = [s for s in _sources(packet).values() if s.role == "license" and s.spdx]
    first_code = next(iter(_code_order(model, packet)), None)
    chosen = next((s for s in licenses if s.of == first_code), licenses[0] if licenses else None)
    return chosen.spdx if chosen else "unknown"


def _paper_url(packet: Packet | None) -> str | None:
    citation = _citation(packet)
    if citation and citation.arxiv:
        return f"https://arxiv.org/abs/{citation.arxiv}"
    if citation and citation.doi:
        return f"https://doi.org/{citation.doi}"
    held = bib_harvest.held_sources(packet) if packet else frozenset()
    paper = next((s for s in _sources(packet).values() if s.role == "paper"
                  and s.sid not in held), None)
    return paper.url if paper else None


def _tlspec(trace: LedgerRow, context: _Context) -> dict[str, Any] | None:
    """Return the saved trace as a download, when the store index knows its size."""

    size = context.blob_bytes.get(trace.trace or "")
    if size is None:
        return None
    return {"url": f"/export/assets/{trace.trace}.tlspec", "sha256": trace.trace, "bytes": size,
            "runnable": trace.runnable == "ready"}


def _bibtex(model: ModelRecord, packet: Packet | None, context: _Context) -> str | None:
    """Copy the saved BibTeX blob when the store is at hand, else build it from the citation."""

    citation = _citation(packet)
    if citation is None:
        return None
    bib = next((s for sid, s in _sources(packet).items() if sid in citation.bib
                and s.role == "bib" and s.api == "bibtex"
                and s.blob), None)
    if bib and context.store:
        saved = bib_harvest._saved_text(bib.blob, context.store)  # noqa: SLF001
        if saved is not None:
            return saved.strip()
    fields = {
        "title": citation.title,
        "author": " and ".join(citation.authors) or None,
        "year": citation.first_public[:4] if citation.first_public else None,
        "journal": citation.venue or (f"arXiv:{citation.arxiv}" if citation.arxiv else None),
        "doi": citation.doi,
        "eprint": citation.arxiv,
    }
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    kind = "article" if citation.venue or citation.arxiv else "misc"
    key = _slugify(model.family[2:]).replace("-", "") + (fields["year"] or "")
    return f"@{kind}{{{key},\n{body}\n}}"


CSV_FIELDS = (
    "stable_id", "slug", "display_name", "category", "family_normalized", "domain", "zoo",
    "source_zoo", "source_license", "source_url", "paper_url", "year", "year_source", "era",
    "forward_validated",
    "validation_state", "capture", "backend", "input_tier", "n_ops", "graph_shape_hash",
    "param_count", "is_recurrent", "default_variant", "monster_graph", "torchlens_version",
)


def _count_facet(values: Iterable[Any]) -> list[dict[str, Any]]:
    counts = Counter(str(v or "unknown") for v in values)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"value": v, "count": n} for v, n in ordered]


def _size_bucket(n_ops: int | None) -> str:
    if n_ops is None:
        return "unknown"
    bounds = ((50, "small"), (250, "medium"), (1000, "large"))
    return next((name for limit, name in bounds if n_ops < limit), "monster")


def _facets(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    keys = {"categories": "category", "families": "family_normalized", "domains": "domain",
            "zoos": "zoo", "tiers": "input_tier"}
    facets = {name: _count_facet(r[key] for r in records) for name, key in keys.items()}
    facets["sizes"] = _count_facet(_size_bucket(r["n_ops"]) for r in records)
    return facets


def _assets(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, int]]:
    """Return ``(key, sha256, bytes)`` for every store blob the records point at."""

    refs = []
    for record in records:
        refs += [record[k] for k in ("interactive", "netron", "model_explorer", "tlspec")
                 if record[k]]
        for variant in record["variants"]:
            refs += [variant[k] for k in ("svg", "thumb", "pdf") if variant[k]]
    return sorted({(r["url"].rsplit("/", 1)[1], r["sha256"], r["bytes"]) for r in refs})


def _write_export(out: Path, records: list[dict[str, Any]], funnel: dict[str, Any],
                  manifest: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, value: Any) -> None:
        text = json.dumps(value, indent=2, sort_keys=True)
        (out / name).write_text(text + "\n", encoding="utf-8")

    with (out / "models.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write(line + "\n")
    dump("funnel.json", funnel)
    dump("export_manifest.json", manifest)
    dump("facets.json", _facets(records))
    with (out / "catalog.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (out / "assets.tsv").open("w", encoding="utf-8") as handle:
        handle.write("key\tsha256\tbytes\n")
        handle.writelines(f"{key}\t{sha}\t{size}\n" for key, sha, size in _assets(records))


def stage_assets(export_dir: Path, *, store: StoreLayout, dest: Path) -> list[str]:
    """Copy every blob ``assets.tsv`` names from the store to ``dest/<key>``; return missing keys.

    A ``.tlspec`` blob is a directory in the store and is copied as one; a packed blob is read
    through its packed form. A blob only the box holds is missing (``store_box restore`` first).
    """

    from menagerie.pipeline import store as store_mod

    dest.mkdir(parents=True, exist_ok=True)
    missing = []
    with (export_dir / "assets.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                store_mod.read_blob(row["sha256"], partial(_copy_blob, dest / row["key"]),
                                    store=store)
            except FileNotFoundError:
                missing.append(row["key"])
    return missing


def _copy_blob(target: Path, source: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    elif not target.exists():
        shutil.copyfile(source, target)
