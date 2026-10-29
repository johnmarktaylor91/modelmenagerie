"""SCHEMA v2 local releases from committed records and saved summaries, without model execution.
The catalog includes untraced and typed rows. Trust columns and artifact quality flags keep
those rows distinct from current validated captures; only the separate site export is public.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from menagerie import csv_dictionary as dictionary
from menagerie.op_taxonomy import OP_TAXONOMY_VERSION
from menagerie.pipeline import DataLayout, StoreLayout, hash_harness, ledger, now, publish, render
from menagerie.pipeline.records import DescribeRecord, LedgerRow, ModelRecord, RecordError, RenderRow, to_dict

TABLES = {"menagerie.csv": (), **dictionary.SIDE_TABLE_COLUMNS}
PUBLIC_SITE = "https://modelmenagerie.ai"
PRESET_DTYPES = {"image": "float32", "randn": "float32", "text": "int64", "randint": "int64"}


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _commit() -> str:
    source = os.environ.get("FLEET_SOURCE_SHA")
    if source:
        if len(source) != 40 or any(c not in "0123456789abcdef" for c in source):
            raise RecordError("FLEET_SOURCE_SHA must be a full git commit hash")
        return source
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                   cwd=Path(__file__).resolve().parents[2]).strip()


def _trace(model: ModelRecord, rows: Sequence[Any], tl: str) -> LedgerRow | None:
    if model.recipe is None:
        return None
    full = ledger.deciding_trace(model.id, rows, recipe_rev=model.recipe.rev, tl=tl)
    if full is not None:
        return full if ledger.validated(full) else None
    return publish._latest(rows, lambda r: isinstance(r, LedgerRow) and r.stage == "trace"
                           and r.recipe_rev == model.recipe.rev and r.tl == tl
                           and r.capture == "structure_only" and r.mode in (None, "eval")
                           and r.outcome != "infra")


def _summary(trace: LedgerRow | None, store: StoreLayout | None) -> dict[str, Any]:
    if trace is None or trace.summary is None:
        return {}
    if store is None:
        raise RecordError("a saved summary requires MENAGERIE_STORE")
    from menagerie.pipeline import store as blobs
    payload = blobs.get(trace.summary, store=store).read_bytes()
    if hashlib.sha256(payload).hexdigest() != trace.summary:
        raise RecordError(f"{trace.id}: summary checksum mismatch")
    def reject_constant(value: str) -> None:
        raise RecordError(f"{trace.id}: non-finite summary value {value}")
    value = json.loads(payload, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise RecordError(f"{trace.id}: summary must be an object")
    return value


def _validation_status(model: ModelRecord, latest: LedgerRow | None) -> str:
    if ledger.validated(latest):
        return "passed"
    if latest and latest.outcome == "fail":
        return "failed"
    if model.give_up or latest and latest.outcome == "giveup":
        return "not_applicable"
    return "not_verified"


def _trust(model: ModelRecord, rows: Sequence[Any], tl: str, renderer: str) -> dict[str, Any]:
    from menagerie.csv_export import _compatible_version
    latest = publish._latest(rows, lambda r: isinstance(r, LedgerRow) and r.stage == "trace"
                             and r.outcome != "infra" and r.capture in (None, "full")
                             and r.mode in (None, "eval"))
    trusted = ledger.validated(latest) and _compatible_version(latest.tl, tl)
    if latest and ledger._fabricated(model.id, rows, latest.recipe_rev):
        trusted = False
    current = bool(model.recipe and ledger.state(model.id, rows, recipe_rev=model.recipe.rev,
                   tl=tl, renderer_rev=renderer) != "not_publishable")
    status = _validation_status(model, latest)
    tier = {"failed": "failed", "not_applicable": "unsupported"}.get(status, "not_verified")
    if trusted:
        tier = "current_verified" if current else "stale_recipe"
    return {"is_trustworthy": trusted, "validated_on_current_release": trusted and current,
            "trust_tier": tier, "validation_status": status, "catalog_verified_hint": False}


def _installation(recipe: Any) -> str:
    if recipe is None:
        return "unavailable"
    if recipe.vendor:
        return "source-build"
    if recipe.extras or recipe.env != "torch-main":
        return "pip"
    return "core"


def _input(model: ModelRecord, summary: Mapping[str, Any]) -> dict[str, Any]:
    inp = model.recipe.input if model.recipe else None
    shape = inp.shape if inp else None
    return {"input_shape": str(tuple(shape)) if shape else None,
            "input_dtype": summary.get("input_dtype") or (
                PRESET_DTYPES.get(inp.preset.split(":", 1)[0]) if inp else None),
            "reference_batch_size": shape[0] if shape else None}


def _taxonomy(model: ModelRecord, ctx: Any) -> dict[str, Any]:
    roster = ctx.roster.get(model.id)
    desc = ctx.describes.get(model.family) or DescribeRecord(family=model.family, by="none")
    member = desc.members.get(model.id)
    return {"domain": desc.domain or (roster.domain if roster else None) or "unknown",
        "era": desc.era or (roster.era if roster else None),
        "task_primary": desc.task, "modality_primary": desc.modality,
        "architecture_paradigm": desc.paradigm or desc.arch_class,
        "architecture_subclass": desc.arch_subclass, "is_generative": desc.is_generative,
        "is_multimodal": len(desc.modalities) > 1 if desc.modalities else None,
        "modality_output": (member.modality_output if member else None) or desc.modality_output,
        "output_type": (member.output_type if member else None) or desc.output_type}


def _execution(model: ModelRecord, summary: Mapping[str, Any], trace: LedgerRow | None,
               basis: Mapping[str, Any]) -> dict[str, Any]:
    from menagerie.csv_export import _model_scale_bucket, _size_variant
    params = summary.get("n_params", trace.params if trace else None)
    recipe = model.recipe
    source = summary.get("n_params_source")
    if not source and trace and params is not None:
        source = "static_count" if trace.params_source == "geometry" else "traced"
    return {"n_params": params, "n_params_source": source,
        "model_scale_bucket": _model_scale_bucket(params) or None, "size_variant": _size_variant(model) or None,
        "n_ops": trace.n_ops if trace else None,
        "dedup_architecture_key": hash_harness.dedup_key(
            to_dict(trace.hashes) if trace.hashes else None, trace.graph_hash,
            basis, trace.backend or "torch") if trace else None,
        "install_difficulty": _installation(recipe), **_input(model, summary)}


def _flagship(model: ModelRecord, rows: Sequence[Any], ctx: Any, summary: Mapping[str, Any],
              trace: LedgerRow | None, *, tl: str, renderer: str, basis: Mapping[str, Any],
              svg: Mapping[str, Any] | None) -> dict[str, Any]:
    packet = ctx.packets.get(model.family)
    citation = publish._citation(packet)
    values = {k: summary.get(k) for k in publish.trace_metrics.FLAGSHIP}
    values.update({"stable_id": model.id, "display_name": model.name.replace("_", " "),
        "family_normalized": packet.identity.canonical if packet else model.family[2:],
        "family": model.family, "variant": model.variant or None, "zoo": model.zoo,
        "publication_year": publish.citecheck.year_of(citation.first_public) if citation else None,
        "paper_url": publish._paper_url(packet), "code_url": publish._source_url(model, packet),
        "license": publish._license(model, packet), **_taxonomy(model, ctx),
        **_execution(model, summary, trace, basis), **_trust(model, rows, tl, renderer),
        "svg_url": svg["url"] if svg else None,
    })
    return values


def _paper(model: ModelRecord, ctx: Any) -> dict[str, Any]:
    packet = ctx.packets.get(model.family)
    citation = publish._citation(packet)
    snap = ctx.citations.get(model.family)
    values = {"stable_id": model.id, "paper_url": publish._paper_url(packet),
              "huggingface_id": model.hf.id if model.hf else None,
              "model_card_url": f"https://huggingface.co/{model.hf.id}" if model.hf else None}
    if citation is None:
        return values
    authors = list(citation.authors)
    values.update({"paper_title": citation.title, "arxiv_id": citation.arxiv,
            "doi": citation.doi, "publication_venue": citation.venue,
            "authors_json": authors or None, "first_author": authors[0] if authors else None,
            "first_preprint_date": citation.first_public if citation.arxiv else None,
            "first_publication_date": citation.published, "semantic_scholar_paper_id": citation.s2,
            "n_citations": (snap.s2 if snap.from_ == "s2" else snap.openalex) if snap else None,
            "citations_snapshot_date": snap.at[:10] if snap else None})
    return values


def _artifacts(model: ModelRecord, current: Any, trace: LedgerRow | None, *, slug: str, svg: Any,
               as_of: str, commit: str, quality: Mapping[str, Any]) -> dict[str, Any]:
    return {"stable_id": model.id, "model_page_url": f"{PUBLIC_SITE}/models/{slug}/" if svg else None,
            "svg_path": svg["url"] if svg else None, "svg_url": svg["url"] if svg else None,
            "svg_sha256": svg["sha256"] if svg else None, "has_svg": svg is not None,
            "tlspec_path": f"/export/assets/{trace.trace}.tlspec" if trace and trace.trace else None,
            "model_card_url": f"https://huggingface.co/{model.hf.id}" if model.hf else None,
            "added_wave": model.source_case,
            "row_quality_flags_json": ([f"capture:{trace.capture or 'full'}"] if trace else []) + [
                current if isinstance(current, str) else f"state:{current.state}"] + [flag for flag, missing in
                (("stale_recipe", quality["trust_tier"] == "stale_recipe"), ("missing_paper", not quality["paper_url"]),
                 ("no_artifact", not svg and (trace is None or not trace.trace))) if missing],
            "dataset_schema_version": dictionary.SCHEMA_VERSION, "dataset_as_of_date": as_of,
            "csv_generation_commit_sha": commit, "op_taxonomy_version": OP_TAXONOMY_VERSION}


def _evidence(model: ModelRecord, rows: Sequence[Any], ctx: Any, standins: Any, *,
              tl: str, renderer: str) -> tuple[Any, Any, dict, dict]:
    """Refuse known wrong architectures before reading their saved trace evidence."""
    current = publish._current(model, rows, tl=tl, renderer=renderer)
    hit = publish.standin.detect(model, standins)
    fabricated = bool(model.recipe and ledger._fabricated(model.id, rows, model.recipe.rev))
    if fabricated:
        return "fabrication", None, {}, dict(is_trustworthy=False,
            validated_on_current_release=False, trust_tier="failed", validation_status="failed")
    if hit is not None or isinstance(current, str) and "standin" in current:
        return "gave up: standin", None, {}, dict(is_trustworthy=False,
            validated_on_current_release=False, trust_tier="unsupported", validation_status="not_applicable")
    trace = _trace(model, rows, tl)
    return current, trace, _summary(trace, ctx.store), {}


def _tables(data: DataLayout, *, as_of: str, commit: str) -> tuple[dict[str, list], Any]:
    from menagerie.pipeline import roster
    models = {m.id: m for m in publish._model_records(data)}
    ctx = publish._Context.load(data, models=models,
                               roster={r.stable_id: r for r in roster.read(data=data) if r.stable_id})
    rows_by_id: dict[str, list] = defaultdict(list)
    for row in ledger.read_all(data=data):
        rows_by_id[row.id].append(row)
    slugs, basis = publish.assign_slugs(models.values()), hash_harness.headline_basis(data)
    standins = publish.standin.known(models.values(), (r for rs in rows_by_id.values() for r in rs),
                                    data=data)
    tables: dict[str, list] = {name: [] for name in TABLES}
    tl, renderer = publish.torchlens_pin(), render.renderer_rev()
    for model in sorted(models.values(), key=lambda m: m.id):
        rows = rows_by_id[model.id]
        current, trace, summary, overrides = _evidence(model, rows, ctx, standins,
                                                      tl=tl, renderer=renderer)
        svg = _svg(current, rows, renderer, trace, summary)
        row = _flagship(model, rows, ctx, summary, trace, tl=tl,
                        renderer=renderer, basis=basis, svg=svg)
        row.update(overrides)
        tables["menagerie.csv"].append(row)
        tables["trace_metrics.parquet"].append({**summary, "stable_id": model.id})
        tables["trace_histograms.jsonl"].append({**summary, "stable_id": model.id})
        tables["papers.csv"].append(_paper(model, ctx))
        tables["artifacts.csv"].append(_artifacts(model, current, trace, slug=slugs[model.id], svg=svg,
                                                as_of=as_of, commit=commit, quality=row))
        desc = ctx.describes.get(model.family)
        for item in desc.lineage if desc else ():
            tables["lineage.csv"].append({"stable_id": model.id, "relation": item.relation,
                "related_stable_id": item.target_id, "related_name": item.target,
                "variant_dimension": item.variant_dimension, "is_canonical_variant": None,
                "source_note": item.quote})
    return tables, basis


def _svg(current: Any, rows: Sequence[Any], renderer: str, trace: LedgerRow | None,
         summary: Mapping[str, Any]) -> Any:
    if not isinstance(current, publish.Current) or trace is None:
        return None
    latest = publish._latest(rows, lambda r: isinstance(r, RenderRow)
                             and r.trace == (trace.trace or trace.summary)
                             and r.renderer_rev == renderer and r.outcome != "infra")
    if latest is None or latest.outcome == "fail":
        return None
    if current.state != "publishable" and not publish._structure_only(trace):
        return None
    variants = publish.variants_of(latest, n_ops=trace.n_ops)
    if not variants:
        return None
    default = render.default_variant([v["key"] for v in variants],
                                     is_recurrent=bool(summary.get("is_recurrent")),
                                     n_ops=trace.n_ops or 0)
    return next(v["svg"] for v in variants if v["key"] == default)


def _cell(value: Any) -> Any:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _columns() -> dict[str, tuple[str, ...]]:
    return {**TABLES, "menagerie.csv": tuple(c.name for c in dictionary.parse_flagship_schema())}


def _project(tables: Mapping[str, list], columns: Mapping[str, Sequence[str]]) -> dict[str, list]:
    return {name: [{k: row.get(k) for k in columns[name]} for row in rows]
            for name, rows in tables.items()}


def _nulls(tables: Mapping[str, list], columns: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    return {name: {k: {"nulls": sum(r[k] is None or r[k] == "" for r in rows),
                       "rows": len(rows),
                       "rate": sum(r[k] is None or r[k] == "" for r in rows) / len(rows)
                       if rows else None} for k in columns[name]} for name, rows in tables.items()}


def _write_table(path: Path, rows: Sequence[dict], columns: Sequence[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from menagerie.csv_export import _pa_type_for, _pa_value
    if path.suffix == ".parquet":
        arrays = {k: pa.array([_pa_value(r[k], _pa_type_for(k)) for r in rows],
                               type=_pa_type_for(k)) for k in columns}
        pq.write_table(pa.table(arrays), path)
    elif path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n"
                                for r in rows), encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: _cell(v) for k, v in r.items()} for r in rows)


def _validate(tables: Mapping[str, list]) -> None:
    for column in dictionary.parse_flagship_schema():
        if column.nullable == "N" and any(r[column.name] in (None, "")
                                          for r in tables["menagerie.csv"]):
            raise RecordError(f"required column {column.name} is null")
    ids = {r["stable_id"] for r in tables["menagerie.csv"]}
    if len(ids) != len(tables["menagerie.csv"]):
        raise RecordError("duplicate stable ids")
    for name, rows in tables.items():
        if any(r["stable_id"] not in ids for r in rows):
            raise RecordError(f"{name}: dangling stable_id")


def _manifest(out: Path, report: Mapping[str, Any], basis: Any) -> dict[str, Any]:
    files = {p.relative_to(out).as_posix(): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
             "bytes": p.stat().st_size} for p in sorted(out.rglob("*")) if p.is_file()}
    return {**report, "files": files, "torchlens_version": publish.torchlens_pin(),
            "hash_basis": basis, "renderer_revision": render.renderer_rev(),
            "schema_version": dictionary.SCHEMA_VERSION, "local_candidate": True}


def build(*, data: DataLayout, out: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    """Write an immutable local ``v1`` and atomically move ``latest.json`` last.

    ``out`` is the dataset root. Dry run reads the same sources and returns coverage only.
    Failure before installation cleans staging. The previous latest pointer survives failure;
    a complete installed version is retained if the final pointer replacement fails.
    """
    root = Path(out or "dataset")
    commit, as_of = _commit(), now()[:10]
    raw, basis = _tables(data, as_of=as_of, commit=commit)
    columns = _columns()
    tables = _project(raw, columns)
    _validate(tables)
    report = {"release": "v1", "as_of": as_of, "commit": commit, "dry_run": dry_run,
              "counts": {k: len(v) for k, v in tables.items()}, "null_rates": _nulls(tables, columns)}
    if dry_run:
        return report
    root.mkdir(parents=True, exist_ok=True)
    if (root / "v1").exists():
        raise RecordError(f"{root / 'v1'} already exists; releases are immutable")
    from menagerie import bundle
    with tempfile.TemporaryDirectory(prefix=".v1-", dir=root) as staging:
        stage = Path(staging)
        for name, rows in tables.items():
            _write_table(stage / name, rows, columns[name])
        dictionary.write_data_dictionary(stage / "DATA_DICTIONARY.md")
        _json(stage / "null-rates.json", report)
        (stage / "RELEASE.md").write_text(
            f"# Local release candidate v1\n\nMeasured {as_of}: {len(tables['menagerie.csv'])} catalog "
            f"records. TorchLens {publish.torchlens_pin()}, schema {dictionary.SCHEMA_VERSION}.\n"
            "Trust columns distinguish validated captures, structure-only hypotheses and untraced rows.\n"
            "Null rates are in null-rates.json. These bundles contain metadata only; trace and "
            "visual assets remain referenced externally. No pretrained weights.\n",
            encoding="utf-8")
        bundle.build_bundles(dist_dir=stage / "bundles", csv_dir=stage,
                             catalog_files=(), as_of=date.fromisoformat(as_of))
        _json(stage / "manifest.json", _manifest(stage, report, basis))
        # Serialize publication and atomically install the complete directory before latest.
        with (root / ".build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (root / "v1").exists():
                raise RecordError("v1 already exists; releases are immutable")
            stage.rename(root / "v1")
            latest = root / ".latest.json.tmp"
            _json(latest, {"release": "v1", "manifest": "v1/manifest.json"})
            os.replace(latest, root / "latest.json")
    return report


def site_count(build_dir: Path, *, out: Path | None = None) -> dict[str, Any]:
    """Count the actual static build, including search shards, against both Pages tiers."""
    from importlib.util import module_from_spec, spec_from_file_location
    script = Path(__file__).resolve().parents[2] / "site/scripts/count_build.py"
    spec = spec_from_file_location("menagerie_site_count", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.count_build(build_dir, out=out)
