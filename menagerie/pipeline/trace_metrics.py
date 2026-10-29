"""Trace statistics for publishing: each model's saved summary blob as SCHEMA v2 columns.

Owned by the rung 2 metadata lane. Every trace attempt saves June's 1.1.0 summary
(``menagerie.trace_summary.summarize_trace``, about 60 statistics: FLOPs, memory, depth, op mix,
module and parameter counts) as the ``summary`` blob of its ledger row. ``flagship`` picks the
SCHEMA v2 flagship trace columns for a site record; ``metrics_row`` gives one ``trace_metrics``
table row; ``write_table`` writes that table beside the export; ``store_summaries`` reads the
summary of every published model for ``csv_export``. Column lists are copied here so publishing
never imports TorchLens; ``test_trace_metrics`` holds them equal to ``csv_export`` and
``csv_dictionary``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from menagerie.pipeline import DataLayout, StoreLayout
from menagerie.pipeline.records import LedgerRow, ModelRecord

# The summary keys behind SCHEMA v2's flagship trace columns (`csv_export._FLAGSHIP_TRACE_COLUMNS`).
FLAGSHIP = (
    "n_params", "n_params_source", "is_recurrent", "is_branching", "is_dynamic_graph",
    "n_compute_ops", "n_unique_op_types", "dominant_op_type", "top_op_types_json", "graph_depth",
    "graph_max_width", "max_fan_out", "max_fan_in", "max_recurrence_iters", "n_conditionals",
    "pct_conv", "pct_linear", "pct_attention", "pct_norm", "pct_elementwise", "has_attention",
    "has_conv", "has_residual", "has_embedding", "norm_type", "activation_fn_type", "n_modules",
    "n_unique_module_types", "module_max_depth", "model_class_name", "structural_barcode_json",
    "total_flops_forward", "total_macs_forward", "flops_coverage_pct", "param_memory_bytes",
    "param_memory_mb", "activation_memory_bytes", "activation_memory_mb",
    "forward_peak_memory_mb", "n_params_trainable", "primary_param_dtype", "n_buffers",
    "output_shape", "n_output_tensors",
)
# The `trace_metrics` side table's columns after `stable_id` (`csv_dictionary.TRACE_METRICS_COLUMNS`).
METRICS = (
    "n_compute_ops", "n_unique_op_types", "n_inplace_ops", "has_custom_op", "dominant_op_type",
    "pct_conv", "pct_linear", "pct_attention", "pct_norm", "pct_elementwise", "pct_reduction",
    "pct_reshape", "pct_embedding", "pct_pooling", "graph_depth", "graph_max_width",
    "branching_factor", "max_fan_out", "max_fan_in", "is_branching", "is_recurrent",
    "max_recurrence_iters", "n_recurrent_layers", "has_conditional_branching", "n_conditionals",
    "is_dynamic_graph", "n_modules", "n_module_calls", "module_max_depth", "n_top_level_modules",
    "n_unique_module_types", "model_class_name", "model_class_qualname", "n_params",
    "n_params_trainable", "n_params_frozen", "n_param_tensors", "param_memory_bytes",
    "primary_param_dtype", "param_dtype_set_json", "quantized_param_tensor_count",
    "has_frozen_params", "n_buffers", "buffer_memory_bytes", "buffer_overwrite_count",
    "total_flops_forward", "total_macs_forward", "total_flops_backward", "total_macs_backward",
    "flops_coverage_pct", "n_unknown_flops_ops", "activation_memory_bytes",
    "forward_peak_memory_bytes", "largest_activation_bytes", "param_memory_mb",
    "activation_memory_mb", "forward_peak_memory_mb",
)
# Which capture a table row's statistics come from: a structure-only (meta-device) capture has
# no real activations, so its memory and FLOPs columns describe the graph, not a run.
PROVENANCE = ("stable_id", "capture", "summary_sha256", "torchlens_version",
              "trace_summary_version")
TABLE = "trace_metrics"


def summary(trace: LedgerRow, store: StoreLayout | None) -> dict[str, Any] | None:
    """The trace's saved summary, or ``None`` with no store, no summary sha or no blob."""

    from menagerie.pipeline import store as store_mod

    if store is None or trace.summary is None:
        return None
    try:  # the raw blob, else its gzipped form
        path = store_mod.get(trace.summary, store=store)
    except FileNotFoundError:
        return None
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def flagship(found: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The flagship trace columns of one summary (``None`` when there is no summary)."""

    return None if found is None else {k: found.get(k) for k in FLAGSHIP}


def metrics_row(stable_id: str, trace: LedgerRow, found: Mapping[str, Any]) -> dict[str, Any]:
    """One ``trace_metrics`` row: provenance, then every metrics column of the summary."""

    return {"stable_id": stable_id, "capture": trace.capture or "full",
            "summary_sha256": trace.summary, "torchlens_version": trace.tl,
            "trace_summary_version": found.get("trace_summary_version"),
            **{k: found.get(k) for k in METRICS}}


def write_table(out: Path, rows: Sequence[Mapping[str, Any]]) -> list[Path]:
    """Write ``trace_metrics.jsonl`` and ``trace_metrics.csv`` (lists and maps as JSON text)."""

    out.mkdir(parents=True, exist_ok=True)
    jsonl, table = out / f"{TABLE}.jsonl", out / f"{TABLE}.csv"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*PROVENANCE, *METRICS))
        writer.writeheader()
        writer.writerows({k: _cell(v) for k, v in row.items()} for row in rows)
    return [jsonl, table]


def _cell(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "" if value is None else value


def store_summaries(*, data: DataLayout, store: StoreLayout) -> dict[str, dict[str, Any]]:
    """The summary of each model's current published trace (``publish``'s choice), by stable id.

    Each row is marked ``summary_source: store``: its recipe and TorchLens version were already
    decided by the rebuild ledger, which ``csv_export`` trusts in place of June's recipe hash.
    """

    from menagerie.pipeline import ledger, publish, read_json, render  # noqa: PLC0415

    tl, renderer = publish.torchlens_pin(), render.renderer_rev()
    rows_by_id: dict[str, list[Any]] = {}
    for row in ledger.read_all(data=data):
        rows_by_id.setdefault(row.id, []).append(row)
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(data.root.glob("models/*/*.json")):
        model = read_json(ModelRecord, path)
        current = publish._current(model, rows_by_id.get(model.id, []), tl=tl,  # noqa: SLF001
                                   renderer=renderer)
        got = None if isinstance(current, str) else summary(current.trace, store)
        if got is not None:
            found[model.id] = {**got, "stable_id": model.id, "summary_source": "store"}
    return found
