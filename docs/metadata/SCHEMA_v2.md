# Menagerie Public Metadata -- SCHEMA v2 (FINAL, implementation-ready)

**One row per model, ~11,670 models. A FLAGSHIP one-stop CSV + joinable SIDE-TABLES, all keyed
on `stable_id`.** This supersedes SCHEMA.md (v1). It applies the reconciled MUST-CHANGE findings
from the two adversarial reviews; it does not re-litigate them.

- Date: 2026-06-25. Schema version: `2.0`.
- Verified against live code: `menagerie/ledger.py`, `menagerie/catalog.py`,
  `menagerie/structural_digest.py`, `torchlens/data_classes/{trace,op,param,module,buffer}.py`.

---

## 0. THE HEADLINE CORRECTION (why v2 exists)

v1 claimed ~103 columns were "in hand". **They are not.** The verification ledger
(`verification_runs` table, `current_verification` view) persists only these trace-derived facts
**per model, today**:

> `n_ops`, `graph_shape_hash`, `svg_sha256`, `recipe_revision_sha256`, `name`, `zoo`, `variant`,
> `scope`, `status`, `forward_pass`, `backward_pass`, `backward_na_reason`, `metadata_ok`,
> `torchlens_version`, `torch_version`, `python_version`, `device_requested`, `device_actual`,
> `env_hash`, `runner_host`, `started_at`, `finished_at`, `duration_sec`, `peak_rss_mb`,
> `error_class`, `error_message`.

That is the WHOLE persisted trace footprint: **`n_ops` + `graph_shape_hash` + `svg_sha256`** are
the only structural facts. Everything else the "moat" advertises -- params, FLOPs/MACs, memory,
depth, width, module stats, op/module histograms, motifs, the architecture hashes -- is a **live
`Trace` property that EXISTS but is NEVER serialized**. Filling those columns requires a NEW
re-trace pass (the TRACE-SUMMARY EXPORTER, S2). This is the single most important fact for anyone
implementing the build.

### 0.1 AVAILABILITY TIER (orthogonal to priority) -- every column carries one

| tier | meaning | cost to fill |
|---|---|---|
| **`persisted`** | In the ledger or catalog row **today**; a build can read it with zero model execution. | free (SQL/JSONL read) |
| **`retrace_required`** | A `Trace`/`Op`/`Param`/`Module` property that EXISTS but is NOT serialized. Needs the TRACE-SUMMARY EXPORTER (S2) to re-trace each model and persist it. | one re-trace per model |
| **`external`** | Outside lookup (paper / license / citations / authors). Not in any TorchLens artifact. | curation / API |

The EXACT `persisted` set (the only zero-cost columns):
`stable_id`, `name`, `family`, `family_normalized`, `domain`, `zoo`, `variant`, `era`,
`constructor_call`, `input_shape`, `input_dtype`, `input_is_real`, `verified` (catalog
`CatalogRow`); `source`/`quarantine` (catalog); `status`, `forward_pass`, `backward_pass`,
`backward_na_reason`, `metadata_ok`, `n_ops`, `graph_shape_hash`, `svg_sha256`,
`recipe_revision_sha256`, `torchlens_version`, `torch_version`, `python_version`, `device_actual`,
`duration_sec`, `peak_rss_mb`, `finished_at`, `error_class`, `error_message` (ledger
`current_verification`). Anything `computed` purely from those is also effectively `persisted`
(noted as `persisted (computed)`). **Everything else is `retrace_required` or `external`.**

> **Availability != priority.** `priority` (core/nice/stretch) = "do we ship it in v1?".
> `availability` = "what does it cost to fill?". A `core` flagship column can be `retrace_required`
> (most of the moat is). Both are columns in every table below.

---

## 1. PACKAGING (JMT's locked decision)

One flagship CSV plus joinable side-tables, **all keyed on `stable_id`** (the durable public join
key; survives recipe edits and renames).

| file | format | role | rows |
|---|---|---|---|
| **`menagerie.csv`** | CSV | **FLAGSHIP** -- 78 curated core columns: identity + at-a-glance + taxonomy + a POPULATED core moat summary + key provenance + validation/trust. The browsable one-stop reference. | one per model |
| `trace_metrics.parquet` | Parquet | The full heavy numeric moat: params / FLOPs / MACs / memory / depth / width / module + param + buffer stats. | one per model |
| `trace_histograms.jsonl` | JSONL | `op_type_histogram`, `module_type_histogram`, FLOPs-by-op-type, top-N blobs, structural barcode. The big variable-length JSON. | one per model |
| `papers.csv` | CSV | External/curated paper provenance (mostly-null-for-now lookups; grows over time). | one per model that has a paper |
| `lineage.csv` | CSV | Curated genealogy relationships (parent / improves-on / successors / variant axis). | one per model with known lineage |
| `artifacts.csv` | CSV | SVG / page / tlspec paths + URLs + content hashes + dataset-release bookkeeping. | one per model |

### 1.1 Join model

```
menagerie.csv          (stable_id PK)
   |-- 1:1  trace_metrics.parquet   on stable_id   (LEFT; null block when not retraced)
   |-- 1:1  trace_histograms.jsonl  on stable_id   (LEFT; absent line == no histograms)
   |-- 1:1  papers.csv              on stable_id   (LEFT; absent when no paper curated)
   |-- 1:N  lineage.csv             on stable_id   (0..N relationship rows per model)
   |-- 1:1  artifacts.csv           on stable_id   (LEFT; one artifact row per model)
```

- `stable_id` is the PK of `menagerie.csv` and the FK of every side-table.
- Every side-table is a LEFT join from the flagship: a model with no retrace yet simply has an
  absent / all-null side-row, never a dropped flagship row.
- `lineage.csv` is the only 1:N table (a model may improve on several predecessors); the rest are
  1:1. The data dictionary documents the cardinality and the absent-row convention per table.
- `menagerie.csv` ALSO carries a populated *summary* of the moat (e.g. `n_params`,
  `total_macs_forward`, `graph_depth`, `pct_conv`, `has_attention`, ...) so the flagship is useful
  standalone; `trace_metrics.parquet` is the exhaustive superset for analysts who join.

---

## 2. CONVENTIONS (LOCKED)

- **Naming:** `snake_case`, lowercase. Established acronyms kept (`arxiv`, `doi`, `url`, `sha256`, `id`).
- **Counts:** `n_*` prefix, no unit suffix, int. `n_ops`, `n_params`, `n_modules`.
- **Booleans (RESOLVED OQ #2 / fix #5):** `is_*` / `has_*` prefix.
  - **Non-nullable bool -> `0`/`1` int**, always populated.
  - **Nullable bool -> tri-state:** `1` = true, `0` = false, **empty string = unknown/not-determined**.
    The data dictionary annotates EVERY boolean with its nullable flag and what empty means. No
    `true`/`false` literals anywhere.
- **Memory (RESOLVED OQ #1):** raw **bytes** with `_bytes` suffix (int) is canonical for all
  trace-derived memory. PLUS three convenience floats in MB for spreadsheet humans:
  `param_memory_mb`, `activation_memory_mb`, `forward_peak_memory_mb` (float, = bytes / 1048576).
  `peak_rss_mb` stays MB int (ledger-native). `min_inference_ram_mb` stays MB int (practitioner).
- **Durations:** `_sec` suffix (float seconds).
- **Percentages:** `pct_*` prefix, **0-100 float** (NOT 0-1). Fractions use `_frac` (0-1 float).
- **Compute:** ship BOTH `total_macs_forward` and `total_flops_forward` (1 MAC = 2 FLOPs);
  practitioners think in both, and the `Trace` exposes both directly.
- **Hashes:** lowercase hex. SHA-256 = 64 chars.
- **Lists / histograms (RESOLVED OQ #3):** JSON-array / JSON-object strings, `_json` suffix,
  deterministic key order. Short top-N lists may live in the flagship CSV; full histograms live in
  `trace_histograms.jsonl`. One serialization convention (JSON), never pipe-delimited.
- **Dates:** ISO-8601. `_utc` suffix for full timestamps; year-only is int (`YYYY`).
- **Nullability:** "Nullable Y" = legitimately absent for some rows; rendered **empty string** in
  CSV (never `"NA"`/`"null"`). "Nullable N" = always populated.
- **Priority:** `core` = ship in v1; `nice` = high-value, realistically fillable; `stretch` =
  aspirational / external-API / new-artifact dependent.
- **Availability:** `persisted` / `retrace_required` / `external` per Section 0.1.

---

## 3. THE FLAGSHIP TABLE -- `menagerie.csv` (78 columns, ordered, at-a-glance first)

Each column: `name | type+unit | description | availability | source | priority | nullable`.
The first 16 are the AT-A-GLANCE default view.

### 3a. At-a-glance identity + browse (cols 1-16)

| # | name | type+unit | description | availability | source | priority | nullable |
|---|---|---|---|---|---|---|---|
| 1 | `stable_id` | str (opaque) | Durable per-model identity; the public join key. Survives recipe edits/renames. | persisted | catalog | core | N |
| 2 | `display_name` | str | Human-facing model name for tables/search (e.g. "ResNet-50"). | persisted (computed) | catalog `name`, cleaned | core | N |
| 3 | `family_normalized` | str | Canonical family for grouping (e.g. "ResNet", "ViT"). | persisted | catalog | core | N |
| 4 | `variant` | str | Discriminator for same-name/same-zoo rows; empty when not needed. | persisted | catalog | core | Y |
| 5 | `zoo` | str | Source zoo/library (e.g. "timm", "torchvision", "classics"). | persisted | catalog | core | N |
| 6 | `domain` | str | Canonical macro-domain facet (e.g. "vision/classification-backbone"). | persisted | catalog | core | N |
| 7 | `task_primary` | str | One canonical task for quick filtering (e.g. "image-classification"). | external | computed/curated | core | Y |
| 8 | `modality_primary` | str | Main input modality (image/text/audio/video/graph/...). | retrace_required | computed (catalog+trace) | core | Y |
| 9 | `architecture_paradigm` | str | Coarse class: CNN/Transformer/RNN/GNN/Diffusion/GAN/MLP/SSM/Hybrid/Other. | retrace_required | computed (family+op-taxonomy) | core | Y |
| 10 | `n_params` | int | Total scalar parameter count. Traced value when retraced, else static fallback. | retrace_required | trace / static (see `n_params_source`) | core | Y |
| 11 | `n_params_source` | enum str | Provenance of `n_params`: `traced` / `static_count`. | persisted (computed) | computed | core | Y |
| 12 | `publication_year` | int (YYYY) | First public-release year (preprint or pub, earliest). | external | external (classics: seeded) | core | Y |
| 13 | `era` | str | Source era / year string or classics era code. | persisted | catalog | core | Y |
| 14 | `validation_status` | enum str | Latest verification status (raw ledger). One of the real ledger `STATUS_VALUES`, or `not_verified`. | persisted | ledger `status` | core | Y |
| 15 | `is_trustworthy` | bool | Passed forward + metadata validation on a COMPATIBLE TorchLens version, with `n_ops` + `graph_shape_hash` present. (Decoupled from current-release match -- see `validated_on_current_release`.) | persisted (computed) | computed (ledger) | core | N |
| 16 | `n_ops` | int | Total traced forward op count. | persisted | ledger / trace | core | Y |

*`validation_status` enum (fix #5) -- ALL real ledger STATUS_VALUES plus the never-run sentinel:*
`passed` / `failed` / `skipped` / `timeout` / `not_applicable` / `deferred` / `error` /
`install_failed` / `env_unavailable` / `oom` / `native_crash` / `killed` / `not_verified`.

### 3b. Taxonomy + classification (cols 17-26) -- faceted-browse axes

| # | name | type+unit | description | availability | source | priority | nullable |
|---|---|---|---|---|---|---|---|
| 17 | `family` | str | Raw source family string before normalization. | persisted | catalog | core | N |
| 18 | `architecture_subclass` | str | Finer subtype (encoder-only/decoder-only/encoder-decoder/deformable/sparse/...). | external | computed/curated | nice | Y |
| 19 | `modality_output` | str | Primary output modality (logits/embeddings/mask/bbox/text/audio/image/...). | retrace_required | computed (task+trace) | nice | Y |
| 20 | `model_scale_bucket` | enum str | `<1M`/`1-10M`/`10-100M`/`100M-1B`/`1B+`. | retrace_required | computed (n_params) | core | Y |
| 21 | `size_variant` | str | Natural size token (tiny/small/base/large/xl) parsed from name/variant. | persisted (computed) | computed | nice | Y |
| 22 | `is_multimodal` | bool | Operates over multiple modalities at once. | retrace_required | computed | nice | Y |
| 23 | `is_generative` | bool | Produces content (image/text/audio) vs classify/regress. | external | computed/curated | nice | Y |
| 24 | `is_recurrent` | bool | Any layer executed more than once (loops/shared weights). | retrace_required | trace `is_recurrent` | core | Y |
| 25 | `is_branching` | bool | Any op has >1 child (residual/multi-head/fork-merge topology). | retrace_required | trace `is_branching` | core | Y |
| 26 | `is_dynamic_graph` | bool | Graph topology depends on runtime tensor values (conditional flow present). | retrace_required | trace (`has_conditional_branching`) | core | Y |

### 3c. Core moat summary (cols 27-52) -- populated structural signature

*Populated subset of `trace_metrics.parquet`. ALL `retrace_required` unless noted.*

| # | name | type+unit | description | availability | source | priority | nullable |
|---|---|---|---|---|---|---|---|
| 27 | `n_compute_ops` | int | Non-boundary compute op count. | retrace_required | trace `num_compute_ops` | core | Y |
| 28 | `n_unique_op_types` | int | Distinct torch func types (functional vocabulary breadth). | retrace_required | trace | core | Y |
| 29 | `dominant_op_type` | str | Single most-frequent op type by count. | retrace_required | trace (op-taxonomy) | nice | Y |
| 30 | `top_op_types_json` | JSON array str | Top-5 most-frequent func_names. | retrace_required | trace | core | Y |
| 31 | `graph_depth` | int | Longest input->output path in op hops. **Computed `max(Op.max_distance_from_input)`** -- NOT a Trace property. | retrace_required | exporter (op distances) | core | Y |
| 32 | `graph_max_width` | int | Max ops at any single BFS depth level (parallelism capacity). | retrace_required | exporter (op distances) | core | Y |
| 33 | `max_fan_out` | int | Max children of any op (`Trace.max_out_degree`). | retrace_required | trace `max_out_degree` | nice | Y |
| 34 | `max_fan_in` | int | Max parents of any op (`Trace.max_in_degree`). | retrace_required | trace `max_in_degree` | nice | Y |
| 35 | `max_recurrence_iters` | int | Max passes any single layer executed (`Trace.max_layer_op_count`). | retrace_required | trace `max_layer_op_count` | core | Y |
| 36 | `n_conditionals` | int | Count of conditional-flow records (`Trace.num_conditionals`). | retrace_required | trace | core | Y |
| 37 | `pct_conv` | float (0-100) | Percent of ops that are convolution primitives. | retrace_required | trace + op-taxonomy | core | Y |
| 38 | `pct_linear` | float (0-100) | Percent linear/matmul primitives. | retrace_required | trace + op-taxonomy | core | Y |
| 39 | `pct_attention` | float (0-100) | Percent attention primitives. | retrace_required | trace + op-taxonomy | core | Y |
| 40 | `pct_norm` | float (0-100) | Percent normalization primitives. | retrace_required | trace + op-taxonomy | core | Y |
| 41 | `pct_elementwise` | float (0-100) | Percent elementwise ops. | retrace_required | trace + op-taxonomy | core | Y |
| 42 | `has_attention` | bool | Uses attention mechanisms (`pct_attention>0`). | retrace_required | trace + op-taxonomy | core | Y |
| 43 | `has_conv` | bool | Uses any convolution primitive. | retrace_required | trace + op-taxonomy | core | Y |
| 44 | `has_residual` | bool | Uses residual/skip connections (add op with skip-connected parents). | retrace_required | trace (topology) | core | Y |
| 45 | `has_embedding` | bool | Embedding lookup ops present. | retrace_required | trace + op-taxonomy | core | Y |
| 46 | `norm_type` | enum str | Dominant norm: batch_norm/layer_norm/group_norm/instance_norm/rms_norm/none/mixed. | retrace_required | trace + op-taxonomy | core | Y |
| 47 | `activation_fn_type` | enum str | Dominant activation: relu/gelu/silu/elu/leaky_relu/tanh/sigmoid/mish/.../mixed. | retrace_required | trace + op-taxonomy | core | Y |
| 48 | `n_modules` | int | Total nn.Module instances (`Trace.num_modules`). | retrace_required | trace | core | Y |
| 49 | `n_unique_module_types` | int | Distinct nn.Module subclass types. | retrace_required | trace | core | Y |
| 50 | `module_max_depth` | int | Max nesting depth of the module tree (`max(Op.module_call_depth)`). | retrace_required | exporter | core | Y |
| 51 | `model_class_name` | str | Captured root model class name (`Trace.model_class_name`). | retrace_required | trace | core | Y |
| 52 | `structural_barcode_json` | JSON object str | Compact "structural DNA": n_ops, graph_depth, max_width, recurrent/branching/dynamic flags, has_attention/conv/residual, norm_type, activation_fn, n_params, n_modules, top_op. Single-column quick characterizer. | retrace_required | exporter (trace fields) | nice | Y |

### 3d. Compute + memory summary (cols 53-60)

| # | name | type+unit | description | availability | source | priority | nullable |
|---|---|---|---|---|---|---|---|
| 53 | `total_flops_forward` | int (FLOPs) | Forward FLOPs at reference input (`Trace.total_flops_forward`; unknown ops skipped). | retrace_required | trace | core | Y |
| 54 | `total_macs_forward` | int (MACs) | Forward MACs (`Trace.total_macs_forward`; 1 MAC = 2 FLOPs). | retrace_required | trace | core | Y |
| 55 | `flops_coverage_pct` | float (0-100) | Percent of ops with a non-null FLOP estimate. | retrace_required | exporter | core | Y |
| 56 | `param_memory_bytes` | int (bytes) | Total parameter memory (`Trace.total_param_memory`). | retrace_required | trace | core | Y |
| 57 | `param_memory_mb` | float (MB) | Convenience MB view of `param_memory_bytes`. | retrace_required | computed | nice | Y |
| 58 | `activation_memory_bytes` | int (bytes) | Total activation tensor volume (`Trace.total_activation_memory`). | retrace_required | trace | core | Y |
| 59 | `activation_memory_mb` | float (MB) | Convenience MB view of `activation_memory_bytes`. | retrace_required | computed | nice | Y |
| 60 | `forward_peak_memory_mb` | float (MB) | Convenience MB view of `forward_peak_memory_bytes` (in `trace_metrics`). | retrace_required | computed | nice | Y |

### 3e. Parameters + I/O contract (cols 61-69)

| # | name | type+unit | description | availability | source | priority | nullable |
|---|---|---|---|---|---|---|---|
| 61 | `n_params_trainable` | int | Trainable scalar parameters. | retrace_required | trace (param accessor) | core | Y |
| 62 | `primary_param_dtype` | str | Most common param dtype by scalar count. | retrace_required | trace (`Param.dtype`) | core | Y |
| 63 | `n_buffers` | int | Registered persistent buffer count. | retrace_required | trace `buffers` | nice | Y |
| 64 | `input_shape` | str | Example input tensor shape (e.g. "(1,3,224,224)"); multi-input lists all. | persisted | catalog | core | Y |
| 65 | `input_dtype` | str | Example input dtype. | persisted | catalog | core | Y |
| 66 | `reference_batch_size` | int | Batch size from first dim of primary tensor input. | persisted (computed) | computed (input_shape) | core | Y |
| 67 | `output_shape` | str | Shape of primary output at reference input. | retrace_required | trace (output op) | core | Y |
| 68 | `n_output_tensors` | int | Number of tensor leaves in model output. | retrace_required | trace | core | Y |
| 69 | `output_type` | enum str | Output semantics: logits/boxes/mask/waveform/token-logits/embedding/policy/reconstruction. | external | computed/curated | core | Y |

### 3f. Provenance + install (cols 70-73)

| # | name | type+unit | description | availability | source | priority | nullable |
|---|---|---|---|---|---|---|---|
| 70 | `paper_url` | str | Primary paper URL (arXiv preferred). Joins to `papers.csv` for the rest. | external | external (classics: docstring) | core | Y |
| 71 | `code_url` | str | Official implementation repository URL. | external | external | nice | Y |
| 72 | `license` | str (SPDX-ish) | Model/code/weights license. | external | external | core | Y |
| 73 | `install_difficulty` | enum str | core/pip/pinned-env/source-build/unavailable. | persisted (computed) | computed (`menagerie.runtime`) | core | Y |

### 3g. Validation + trust + artifacts (cols 74-78)

| # | name | type+unit | description | availability | source | priority | nullable |
|---|---|---|---|---|---|---|---|
| 74 | `trust_tier` | enum str | Graded trust: `current_verified`/`stale_recipe`/`legacy_seed`/`failed`/`deferred`/`unsupported`/`not_verified`. | persisted (computed) | computed (ledger+catalog) | core | N |
| 75 | `validated_on_current_release` | bool | Latest passing run used the CURRENT TorchLens release AND current recipe hash. (The strict-version signal, decoupled from `is_trustworthy`.) | persisted (computed) | computed (ledger) | core | N |
| 76 | `catalog_verified_hint` | bool | Legacy catalog-level instantiable hint (looser than `is_trustworthy`; renamed from `verified`). | persisted | catalog `verified` | nice | N |
| 77 | `dedup_architecture_key` | str | "Same architecture" grouping key: `architecture_distinctness_hash` (= `graph_shape_hash`) when retraced, else empty. | retrace_required | structural_digest | core | Y |
| 78 | `svg_url` | str | Public URL for the rendered SVG graph (joins to `artifacts.csv` for paths/hashes). | persisted (computed) | computed (`svg_sha256`) | core | Y |

**Flagship total: 78 columns.** By availability (exact): `persisted` 12 + `persisted (computed)` 9
= 21 zero-cost, `retrace_required` 49, `external` 8. The single largest fact this schema
communicates: **49 of 78 flagship columns -- and nearly the entire moat -- cannot be filled without
the new TRACE-SUMMARY EXPORTER (S2).** By priority: `core` ~62, `nice` ~16.

> **Trust columns are exactly 3 + 1 rename (fix #6):** `validation_status` (col 14, raw),
> `is_trustworthy` (col 15, gate, decoupled from strict version match),
> `trust_tier` (col 74, graded), with the strict-version match split out as
> `validated_on_current_release` (col 75) and catalog `verified` renamed to
> `catalog_verified_hint` (col 76). No four-way ambiguity.

---

## 4. SIDE-TABLES

### 4.1 `trace_metrics.parquet` -- the full heavy numeric moat (57 cols + key)

Key: `stable_id`. ALL `retrace_required` (produced by the TRACE-SUMMARY EXPORTER, S2) unless
marked computed. This is the exhaustive superset; the flagship reproduces ~26 of these as its core
moat summary. The per-section counts below sum to 57 (14+12+7+12+9+3). Variable-length JSON
distributions (`op_type_histogram`, `module_type_histogram`, `flops_by_op_type`, ...) live in
`trace_histograms.jsonl`, NOT here. `name | type+unit | description | source`.

**Op composition (14):**
`n_compute_ops` int (`num_compute_ops`); `n_unique_op_types` int; `n_inplace_ops` int;
`has_custom_op` bool (op outside std torch namespace); `dominant_op_type` str (op-taxonomy);
`pct_conv` / `pct_linear` / `pct_attention` / `pct_norm` / `pct_elementwise` / `pct_reduction` /
`pct_reshape` / `pct_embedding` / `pct_pooling` float 0-100 (op-taxonomy classifier).

**Graph topology (12):**
`graph_depth` int (`max(Op.max_distance_from_input)` -- COMPUTED, not a Trace prop);
`graph_max_width` int (max ops per distance level -- computed); `branching_factor` float
(`Trace.branching_factor`); `max_fan_out` int (`max_out_degree`); `max_fan_in` int
(`max_in_degree`); `is_branching` bool; `is_recurrent` bool; `max_recurrence_iters` int
(`max_layer_op_count`); `n_recurrent_layers` int (`len(recurrent_layers)`);
`has_conditional_branching` bool; `n_conditionals` int (`num_conditionals`); `is_dynamic_graph` bool.

**Module hierarchy (7):**
`n_modules` int (`num_modules`); `n_module_calls` int (`num_module_calls`); `module_max_depth` int
(`max(Op.module_call_depth)` -- computed); `n_top_level_modules` int; `n_unique_module_types` int;
`model_class_name` str; `model_class_qualname` str (`Trace.model_class_qualname`).

**Parameters + buffers (12):**
`n_params` int (sum of `Param.num_params`); `n_params_trainable` int
(`num_param_tensors_trainable` scalars); `n_params_frozen` int; `n_param_tensors` int;
`param_memory_bytes` int (`total_param_memory`); `primary_param_dtype` str; `param_dtype_set_json`
JSON array (mixed-precision detector); `quantized_param_tensor_count` int (`Param.is_quantized`);
`has_frozen_params` bool (`Trace.has_frozen_params`); `n_buffers` int; `buffer_memory_bytes` int;
`buffer_overwrite_count` int (`num_buffer_write_ops`).

**Compute + memory (9):**
`total_flops_forward` int; `total_macs_forward` int; `total_flops_backward` int
(analytic TorchLens estimate when available; often non-zero even for inference-only summaries);
`total_macs_backward` int (analytic TorchLens estimate when available); `flops_coverage_pct` float
0-100; `n_unknown_flops_ops` int; `activation_memory_bytes` int (`total_activation_memory`);
`forward_peak_memory_bytes` int (high-water estimate); `largest_activation_bytes` int.

**Convenience MB floats (3):** `param_memory_mb`, `activation_memory_mb`, `forward_peak_memory_mb`.

### 4.2 `trace_histograms.jsonl` -- variable-length JSON blobs (1 line per model)

Key: `stable_id`. Each line is a JSON object. ALL `retrace_required`. Fields:
- `stable_id` str (key)
- `op_type_histogram` object `{func_name: count}` desc -- full primitive distribution
- `module_type_histogram` object `{module_class_name: count}` -- full module distribution
- `top_module_types_json` array -- top-5 module class names
- `top_level_block_sequence_json` array -- ordered top-level module class names (macro recipe)
- `flops_by_op_type` object `{op_type: flops}` (`Trace.flops_by_op_type`)
- `macs_by_op_type` object `{op_type: macs}` (`Trace.macs_by_op_type`)
- `structural_barcode` object -- the same blob as flagship `structural_barcode_json`, expanded

### 4.3 `papers.csv` -- external paper provenance (mostly null today)

Key: `stable_id`. ALL `external` unless computed. Columns:
`stable_id` (key); `paper_title`; `paper_url`; `arxiv_id`; `doi`; `publication_venue`;
`venue_type` enum (computed); `authors_json` array; `first_author` str (computed);
`first_preprint_date` (YYYY-MM-DD); `first_publication_date` (YYYY-MM-DD); `huggingface_id`;
`model_card_url`; `semantic_scholar_paper_id`; `n_citations` int; `citations_snapshot_date`;
`citation_tier` enum (computed); `is_milestone` bool (curated). **17 data cols + `stable_id` key.**

### 4.4 `lineage.csv` -- curated genealogy (1:N; sparse)

Key: `stable_id` (NOT unique -- a model may have multiple lineage rows). Columns:
`stable_id` (FK); `relation` enum (`parent`/`improves_on`/`inspired_by`/`successor`/`variant_of`);
`related_stable_id` str (target, when in-catalog); `related_name` str (free-text target when not);
`variant_dimension` str (scale/depth/width/resolution/efficiency/training/language/hardware, for
`variant_of`); `is_canonical_variant` bool; `source_note` str (where the relation was asserted).
ALL `external`/curated. **6 data cols + `stable_id` key.**

### 4.5 `artifacts.csv` -- artifacts + dataset bookkeeping (1:1)

Key: `stable_id`. Columns:
`stable_id` (FK); `model_page_url` (computed); `svg_path` (artifact manifest); `svg_url`;
`svg_sha256` (persisted, ledger); `has_svg` bool; `tlspec_path` (stretch); `model_card_url`;
`added_wave` str (catalog); `row_quality_flags_json` array (stale_recipe/missing_paper/no_artifact);
`dataset_schema_version` (build); `dataset_as_of_date` (build); `csv_generation_commit_sha`
(build). **12 data cols + `stable_id` key.**

### 4.6 Side-table counts

| table | data cols (excl. `stable_id` key) | key | cardinality |
|---|---|---|---|
| `trace_metrics.parquet` | 57 | stable_id | 1:1 |
| `trace_histograms.jsonl` | 6 JSON fields | stable_id | 1:1 |
| `papers.csv` | 17 | stable_id | 1:1 |
| `lineage.csv` | 6 | stable_id | 1:N |
| `artifacts.csv` | 12 | stable_id | 1:1 |

---

## 5. REQUIRED BUILD ARTIFACTS (the two new dependencies, fix #1/#3)

The flagship moat does not exist on disk. Building this dataset REQUIRES two new, versioned
artifacts. Neither exists today; both must be built before `menagerie.csv` can be populated past
its `persisted` columns.

### 5.1 OP-TAXONOMY CLASSIFIER (`op_taxonomy`, versioned)

**Why:** every `pct_*`, `has_*`, `dominant_op_type`, `norm_type`, `activation_fn_type`,
`architecture_paradigm`, and the FLOPs-by-category splits depend on mapping a raw op to a
semantic category. That mapping does not exist; v1 hand-waved it.

**What it must be:** a deterministic, versioned function
`classify_op(op) -> category` keyed on **`Op.func_name` + the op's owning module class**, returning
one canonical category from a CLOSED set:

> `conv` | `linear` | `attention` | `norm` | `elementwise` | `reduction` | `reshape` |
> `embedding` | `pooling` | `activation` | `other`

**Requirements:**
- Versioned artifact; its version string is recorded per build as **`op_taxonomy_version`**
  (add this column to `artifacts.csv` build-metadata, and stamp it in `trace_metrics` provenance).
- Module-class tokens intentionally take precedence over function-name tokens when both are present;
  e.g. primitives inside a semantic attention/convolution wrapper inherit the wrapper category.
- Sub-classifiers it must also expose (used by `norm_type` / `activation_fn_type`):
  norm subtype (batch/layer/group/instance/rms) and activation subtype
  (relu/gelu/silu/elu/leaky_relu/tanh/sigmoid/mish). `mixed`/`none` are valid outputs.
- Deterministic and total (every op maps to exactly one category; unknown -> `other`).
- Lives in `menagerie/` (e.g. `menagerie/op_taxonomy.py`), unit-tested against a fixture set.

**Field list it powers (downstream):** `pct_conv`, `pct_linear`, `pct_attention`, `pct_norm`,
`pct_elementwise`, `pct_reduction`, `pct_reshape`, `pct_embedding`, `pct_pooling`,
`dominant_op_type`, `has_attention/conv/embedding`, `norm_type`, `activation_fn_type`,
`architecture_paradigm`, `flops_by_op_type` category rollups, `pct_flops_from_{attention,conv,linear}`.

### 5.2 TRACE-SUMMARY EXPORTER (`trace_summary`, versioned)

**Why:** the headline fix. Every `retrace_required` column is a live `Trace` property that the
ledger never persisted. The exporter is the new per-model pass that RE-TRACES each model (CPU,
inference-only, exactly as `menagerie/structural_digest.py` already does) and writes a durable
summary record so the CSV build never needs to re-run models.

**What it must produce, per model (the persisted field list):**

```
stable_id, recipe_revision_sha256, torchlens_version (provenance)

# op composition
n_compute_ops, n_unique_op_types, n_inplace_ops, has_custom_op,
op_type_histogram (full), dominant_op_type,
pct_{conv,linear,attention,norm,elementwise,reduction,reshape,embedding,pooling}   # via op_taxonomy

# topology  (graph_depth / graph_max_width / module_max_depth are COMPUTED from Op distances)
graph_depth                 = max(op.max_distance_from_input for op in compute ops)
graph_max_width             = max count of ops sharing one max_distance_from_input level
branching_factor, max_fan_out (max_out_degree), max_fan_in (max_in_degree),
is_branching, is_recurrent, max_recurrence_iters (max_layer_op_count),
n_recurrent_layers, has_conditional_branching, n_conditionals (num_conditionals), is_dynamic_graph

# modules
n_modules (num_modules), n_module_calls (num_module_calls),
module_max_depth = max(op.module_call_depth), n_top_level_modules,
n_unique_module_types, module_type_histogram (full), top_module_types,
top_level_block_sequence, model_class_name, model_class_qualname

# parameters + buffers
n_params (sum Param.num_params), n_params_trainable, n_params_frozen, n_param_tensors,
param_memory_bytes (total_param_memory), primary_param_dtype, param_dtype_set,
quantized_param_tensor_count (Param.is_quantized), has_frozen_params,
n_buffers, buffer_memory_bytes, buffer_overwrite_count (num_buffer_write_ops)

# compute + memory
total_flops_forward, total_macs_forward, total_flops_backward, total_macs_backward,
flops_coverage_pct, n_unknown_flops_ops, flops_by_op_type, macs_by_op_type,
activation_memory_bytes (total_activation_memory), forward_peak_memory_bytes,
largest_activation_bytes

# i/o
output_shape, n_output_tensors, output_container_kind

# motifs (topology-derived, not a single property)
has_residual, has_self_attention, has_cross_attention, norm_type, activation_fn_type, pooling_type

# identity hashes (the REAL entry points -- menagerie/structural_digest.py)
graph_shape_hash               = architecture_distinctness_hash(model, x)   # shape-blind
structural_fingerprint_hash    = structural_fingerprint(model, x)           # raw-event shape-aware
```

**Requirements:**
- Reuse `menagerie/structural_digest._trace_for_digest` (CPU, `inference_only=True`) so the
  summary trace matches the digest trace exactly (same input scaling, same device).
- Versioned: stamp **`trace_summary_version`** + the `op_taxonomy_version` it consumed into each
  record's provenance.
- `graph_depth`, `graph_max_width`, `module_max_depth` are **computed in the exporter** from
  `Op.max_distance_from_input` / `Op.module_call_depth` (both `FieldPolicy.KEEP`, portable) -- they
  are NOT Trace properties (fix #3). Do not invent a `Trace.graph_depth`.
- Output feeds `trace_metrics.parquet` + `trace_histograms.jsonl` directly; the flagship moat
  summary columns are a projection of this record.
- Failure handling: a model that fails the re-trace gets a null metrics block (LEFT-join absence),
  never blocks the flagship row.

### 5.3 STATIC n_params FALLBACK (fix #4)

`n_params` must not be blank for the exotic, never-traced rows people will click. When a model is
constructible but not traced (failed / too-big / deferred), the exporter (or a lighter sidecar
pass) computes a static count from `sum(p.numel() for p in model.named_parameters())` -- **no trace
required, no forward pass** -- and sets `n_params_source = static_count`. Traced rows set
`n_params_source = traced`. `n_params` stays empty only when even construction fails.

### 5.4 Demoted / recomputed (fix #3 tail)

- `raw_event_shape_hash` (was a v1 column from `Trace._raw_event_shape_hash`, a private postprocessor
  attr with `FieldPolicy.DROP`, non-portable) -> **demoted to stretch**, exposed only via the
  exporter as `structural_fingerprint_hash` using `structural_digest.structural_fingerprint`
  (recompute-in-exporter, RESOLVED OQ #5). Not a flagship column.
- `architecture_param_shape_hash` (v1 claimed; not found in code) -> dropped; the real two hashes
  are `graph_shape_hash` (= `architecture_distinctness_hash`) and `structural_fingerprint_hash`.
- Benchmark family (latency/throughput/VRAM/energy) -> OUT of v1 entirely (RESOLVED OQ #4); the
  data dictionary carries a one-line roadmap note, no placeholder columns.
- `description`/`key_innovation` -> `nice`, not core (RESOLVED OQ #6; ~96% null today); they live in
  `papers.csv`/curation, not the flagship.

---

## 6. DATA DICTIONARY REQUIREMENT (companion deliverable)

The CSV ships with a human-readable DATA DICTIONARY on the website. The schema rows above are the
SEED. The dictionary MUST carry, per column:
- name, type + unit, plain-language meaning, nullability + what empty means, example value;
- **`source` and `availability` tier** (persisted / retrace_required / external) so a consumer
  knows whether a null means "absent for this model" vs "not yet exported";
- the JOIN KEY (`stable_id`) and the cardinality + absent-row convention of each side-table;
- for EVERY `computed` or `motif` column, an **operational definition** -- the exact formula or
  predicate. Examples that MUST appear verbatim-equivalent:
  - `graph_depth` = `max(op.max_distance_from_input)` over compute ops (op hops).
  - `pct_conv` = `100 * (#ops classified conv by op_taxonomy v<X>) / n_compute_ops`.
  - `is_trustworthy` = `forward_pass=1 AND metadata_ok=1 AND n_ops NOT NULL AND graph_shape_hash
    NOT NULL` on a COMPATIBLE torchlens version (NOT requiring the current release).
  - `validated_on_current_release` = above AND `torchlens_version = <current>` AND recipe hash matches.
  - `model_scale_bucket` = bucketed `n_params` with the documented edges.
  - `has_residual` = exists an add-op whose parents include a skip-connected ancestor (topology).
- for every BOOLEAN: nullable flag + the meaning of `0` / `1` / empty.
- the `op_taxonomy_version` and `trace_summary_version` stamped into the build, so dictionary
  consumers can pin the exact category definitions used.

---

## 7. CHANGE LOG vs v1 (what this fixes)

1. **Availability tier added to every column** (fix #1). Headline: ledger persists only
   ~3 structural facts (`n_ops`, `graph_shape_hash`, `svg_sha256`); the moat is `retrace_required`.
2. **Count reconciled** (fix #2): flagship = exactly 78 columns (listed, ordered); side-tables sum
   explicitly, data cols excluding the `stable_id` key (trace_metrics 57, histograms 6 JSON fields,
   papers 17, lineage 6, artifacts 12).
3. **Non-existent / mis-sourced columns fixed** (fix #3): `graph_depth` computed from
   `Op.max_distance_from_input`; motif layer routed through the new OP-TAXONOMY CLASSIFIER +
   `op_taxonomy_version`; `raw_event_shape_hash` demoted/recomputed; bogus
   `architecture_param_shape_hash` replaced with the real `structural_digest` entry points.
4. **Static n_params fallback** (fix #4): `n_params` + `n_params_source ∈ {traced, static_count}`.
5. **Enums/booleans fixed** (fix #5): `validation_status` includes all ledger `STATUS_VALUES` +
   `not_verified`; tri-state nullable booleans documented; 0/1 for non-nullable.
6. **Trust columns bounded to 3 + rename** (fix #6): `validation_status` / `is_trustworthy`
   (decoupled from strict version) / `trust_tier`, plus `validated_on_current_release` and
   `verified` -> `catalog_verified_hint`.
7. **Packaging**: flagship CSV + 5 side-tables, all keyed on `stable_id`.
