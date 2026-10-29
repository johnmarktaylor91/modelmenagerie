# Menagerie Public Metadata CSV -- UNIFIED SCHEMA

**One row per model. ~11,670 models. The definitive one-stop metadata reference.**

This is the synthesized, deduplicated, ordered column schema for the public per-model
metadata CSV that ships on modelmenagerie.ai. It is the union of four brainstorm proposals
(provenance, practitioner/scale, structural/TorchLens-unique, dataset/identity), aggressively
deduped to **one canonical name per concept**, organized into 16 sections ordered
at-a-glance-first.

- **Total distinct columns: 138**
- Synthesized from: `claude_provenance.md` (P), `codex_practitioner.md` (Pr),
  `claude_structural.md` (S), `codex_dataset.md` (D).
- Date: 2026-06-25.

---

## Conventions (LOCKED)

- **Naming:** `snake_case`, lowercase. Established acronyms kept (`arxiv`, `doi`, `url`, `sha256`).
- **Counts:** `n_*` prefix, no unit suffix (int). `n_ops`, `n_params`, `n_modules`.
- **Booleans:** `is_*` / `has_*` prefix; serialized as `0`/`1` integers in CSV (not `true`/`false`).
- **Memory:** raw bytes with `_bytes` suffix (int) as the canonical store; consumers scale. We
  add `_mb` (float) ONLY for the two columns the ledger already records in MB
  (`peak_rss_mb`). Trace-derived memory is always `_bytes`.
- **Durations:** `_sec` suffix (float seconds).
- **Percentages:** `pct_*` prefix, 0-100 float (NOT 0-1 fraction). Fractions use `_frac` (0-1).
- **Compute:** `total_macs_*` is canonical (1 MAC = 2 FLOPs); `total_flops_*` kept where the
  Trace exposes FLOPs directly. We ship BOTH for forward (practitioners think in both).
- **Hashes:** lowercase hex strings. SHA-256 = 64 chars; opaque IDs shorter.
- **Histograms / distributions:** JSON-object strings with deterministic key order, `_json` suffix.
- **Ordered short lists** (top-N): JSON-array strings, `_json` suffix (NOT pipe-delimited -- one
  serialization convention only; resolves the S/D disagreement in favor of JSON arrays).
- **Dates:** ISO-8601. `_utc` suffix for full timestamps; year-only is int (`YYYY`).
- **Nullability:** "Nullable Y" = field is legitimately absent for some rows; rendered as **empty
  string** in CSV (not `"NA"`/`"null"`). "Nullable N" = always populated.
- **Priority:** `core` = ship in v1; `nice` = high-value, realistically fillable; `stretch` =
  aspirational / external-API / new-benchmark dependent.

## Source legend

- **catalog** -- `menagerie/catalog.py` `CatalogRow` / `master_catalog.jsonl` (in hand)
- **ledger** -- `menagerie/ledger.py` `VerificationRun` / `current_verification` (in hand)
- **trace** -- TorchLens `Trace` properties / `menagerie.structural_digest` (in hand, MENAGERIE-UNIQUE)
- **computed** -- derived from other columns at CSV-build time (in hand)
- **external** -- requires paper/license/HF/GitHub lookup (NOT in hand)

## MENAGERIE-UNIQUE flag

A column marked **[U]** exists nowhere else: only this dataset has it because we ran TorchLens
end-to-end on every model. These are the moat. A column marked **[X]** requires an external
lookup we do not currently hold (paper/license/citations/benchmarks).

---

## SECTION 1: At-a-glance identity + browse  (15 cols) -- DEFAULT VIEW

*The first columns a browser sees. Lead with these. All 15 are the AT-A-GLANCE set.*

| # | name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|---|
| 1 | `stable_id` | str (opaque hex) | **[AT-A-GLANCE]** Durable per-model identity; the public join key. Survives recipe edits/renames. | catalog | core | N |
| 2 | `display_name` | str | **[AT-A-GLANCE]** Human-facing model name for tables/search (e.g. "ResNet-50"). | catalog (`name`, cleaned) | core | N |
| 3 | `family_normalized` | str | **[AT-A-GLANCE]** Canonical family for grouping (e.g. "ResNet", "ViT"). | catalog | core | N |
| 4 | `variant` | str | **[AT-A-GLANCE]** Discriminator for same-name/same-zoo rows; empty when not needed. | catalog | core | Y |
| 5 | `zoo` | str | **[AT-A-GLANCE]** Source zoo/library (e.g. "timm", "torchvision", "classics"). | catalog | core | N |
| 6 | `domain` | str | **[AT-A-GLANCE]** Canonical macro-domain facet (e.g. "vision/classification-backbone"). | catalog | core | N |
| 7 | `task_primary` | str | **[AT-A-GLANCE]** One canonical task for quick filtering (e.g. "image-classification"). | computed/external | core | Y |
| 8 | `modality_primary` | str | **[AT-A-GLANCE]** Main input modality (image/text/audio/video/graph/...). | computed (catalog+trace) | core | Y |
| 9 | `architecture_paradigm` | str | **[AT-A-GLANCE]** Coarse class: CNN/Transformer/RNN/GNN/Diffusion/GAN/MLP/SSM/Hybrid/Other. | computed (family+trace) | core | Y |
| 10 | `n_params` | int | **[AT-A-GLANCE][U]** Total scalar parameter count (live-traced). | trace | core | Y |
| 11 | `publication_year` | int (YYYY) | **[AT-A-GLANCE]** First public-release year (preprint or pub, earliest). | external (classics: seeded) | core | Y |
| 12 | `era` | str | **[AT-A-GLANCE]** Source era/year string or classics era code. | catalog | core | Y |
| 13 | `validation_status` | enum str | **[AT-A-GLANCE][U]** Latest verification status: `passed`/`failed`/`deferred`/`timeout`/`error`/`install_failed`/`env_unavailable`/`not_verified`. | ledger (`status`) | core | Y |
| 14 | `is_trustworthy` | bool | **[AT-A-GLANCE][U]** Current recipe passed forward+metadata validation, has n_ops + graph_shape_hash, matches current TorchLens release. | computed (catalog+ledger) | core | N |
| 15 | `n_ops` | int | **[AT-A-GLANCE][U]** Total traced forward op count (live execution). | ledger/trace | core | Y |

*Naming note: `display_name` (D's pick) over `name` (P/Pr/S). `task_primary`/`modality_primary`
(D's singular-canonical naming) over P's `task`/`modality_input`; multi-valued versions live in
Section 4. `architecture_paradigm` (D) chosen over `architecture_class` (P) -- same concept.
`validation_status` (D) over `verification_status` (P/S) and `current_verification_status` (Pr).
`is_trustworthy` (D) is the single strong trust gate; `verified` (catalog-level) kept separately
in Section 5.*

---

## SECTION 2: Extended identity + display  (8 cols)

*Secondary names and human-facing display. Catalog facts + light curation.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `name` | str | Raw catalog name (pre-display cleanup); preserved for provenance. | catalog | core | N |
| `family` | str | Raw source family string before normalization. | catalog | core | N |
| `family_slug` | str | URL-safe family key. | computed | nice | N |
| `display_index` | int | Human-facing sort position (1-based, family then name). Not stable across rebuilds. | catalog | core | N |
| `sort_name` | str | Lowercased normalized name for stable UI sorting. | computed | nice | N |
| `common_aliases_json` | JSON array str | Known alternative names/abbreviations/spellings. | external | nice | Y |
| `short_name` | str | Abbreviated label (<=20 chars) for space-constrained UI. | computed | nice | Y |
| `search_keywords_json` | JSON array str | Precomputed search tokens (names, family, zoo, aliases, tags). | computed | nice | Y |

---

## SECTION 3: Taxonomy + classification  (16 cols)

*Family, domain, task, modality, paradigm at multiple granularities. The faceted-browse axes.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `domain_raw` | str | Pre-collapse source domain string (audit trail). | catalog (retain JSONL field) | nice | Y |
| `tasks_json` | JSON array str | All ML task facets (multi-task models). | computed/external | nice | Y |
| `modalities_json` | JSON array str | All input/output modalities involved. | computed (catalog+trace) | nice | Y |
| `modality_output` | str | Primary output modality (logits/embeddings/mask/bbox/text/audio/image/...). | computed (task+trace) | nice | Y |
| `architecture_subclass` | str | Finer architectural subtype (encoder-only/decoder-only/encoder-decoder/deformable/sparse/...). | computed/curated | nice | Y |
| `backbone_type` | str | Common backbone family when identifiable. | computed (name+module hist) | nice | Y |
| `generator_type` | str | For generative rows: diffusion/GAN/VAE/flow/autoregressive. | computed/external | nice | Y |
| `learning_setting` | str | supervised/self-supervised/contrastive/RL/generative/... | external (sometimes inferable) | nice | Y |
| `data_structure` | str | tensor/sequence/graph/set/point-cloud/mesh/molecule/PDE-grid. | computed (domain+input) | nice | Y |
| `spatial_dimensionality` | str | 1D/2D/3D/4D/non-spatial. | computed (input_shape) | nice | Y |
| `temporal_extent` | str | static/sequence/recurrent/video/time-series/continuous-time. | computed (domain+trace) | nice | Y |
| `model_scale_bucket` | enum str | `<1M`/`1-10M`/`10-100M`/`100M-1B`/`1B+`. | computed (n_params) | core | Y |
| `size_variant` | str | Natural size token (tiny/small/base/large/xl) parsed from name/variant. | computed | nice | Y |
| `is_multimodal` | bool | Operates over multiple modalities at once. | computed | nice | Y |
| `is_generative` | bool | Produces content (image/text/audio) vs classify/regress. | computed | nice | Y |
| `is_foundation_model` | bool | Large-scale pretrained foundation-model family. | external/curated | stretch | Y |

*Dedup note: this section absorbs P-Section5, S has none here, D-taxonomy, Pr modality fields.
`task_primary`/`modality_primary`/`architecture_paradigm` promoted to Section 1; the multi-valued
and finer-grained siblings live here. `model_scale_bucket` (D) == Pr `param_scale_bucket` -- one name.*

---

## SECTION 4: Paper provenance + external references  (19 cols)

*The provenance lens. Almost entirely **[X]** external lookup (classics rows can seed from docstrings).*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `paper_title` | str | **[X]** Full paper title as published. | external (classics: docstring) | core | Y |
| `paper_url` | str | **[X]** Primary paper URL (arXiv preferred, DOI/proceedings fallback). | external | core | Y |
| `arxiv_id` | str | **[X]** arXiv id, canonical form (e.g. "1706.03762", no prefix). | external/catalog `source_url` | core | Y |
| `doi` | str | **[X]** Digital Object Identifier. | external | nice | Y |
| `publication_venue` | str | **[X]** Venue: conference acronym or journal (e.g. "NeurIPS 2017"). | external | nice | Y |
| `venue_type` | enum str | **[X]** conference/journal/workshop/preprint/tech_report/thesis/unknown. | computed (venue) | nice | Y |
| `authors_json` | JSON array str | **[X]** Author list in publication order. | external | nice | Y |
| `first_author` | str | First-author surname for concise sort/display. | computed (authors) | nice | Y |
| `author_institution_json` | JSON array str | **[X]** Primary author affiliation(s). | external | stretch | Y |
| `first_preprint_date` | str (YYYY-MM-DD) | **[X]** Date of first arXiv posting if known. | external | stretch | Y |
| `first_publication_date` | str (YYYY-MM-DD) | **[X]** Date of first conference/journal publication. | external | stretch | Y |
| `code_url` | str | **[X]** Official implementation repository URL. | external (sometimes by zoo) | core | Y |
| `huggingface_id` | str | **[X]** HuggingFace model/repo id. | external/catalog | nice | Y |
| `model_card_url` | str | **[X]** Public model card / documentation page. | external | nice | Y |
| `semantic_scholar_paper_id` | str | **[X]** Semantic Scholar corpus id for citation-graph linkage. | external | stretch | Y |
| `n_citations` | int | **[X]** Approximate citation count at snapshot date. | external (Semantic Scholar) | nice | Y |
| `citations_snapshot_date` | str (YYYY-MM-DD) | Date `n_citations` was fetched. | external metadata | nice | Y |
| `citation_tier` | enum str | landmark(>10k)/highly_cited(1k-10k)/cited(100-1k)/emerging(<100)/unknown. | computed (n_citations) | nice | Y |
| `is_milestone` | bool | Curated field-milestone shortlist (AlexNet/VGG/ResNet/Transformer/BERT/...). | curated | nice | Y |

*Dedup note: `n_citations` appeared in P-Sec3 AND P-Sec11 -- single canonical column here.
`paper_url`/`code_url`/`model_card_url`/`huggingface_id` consolidate P/Pr/D's overlapping URL
fields. `publication_year` lives in Section 1 (at-a-glance); the precise dates live here.*

---

## SECTION 5: Lineage + genealogy  (7 cols)

*The unique provenance relationships. Mostly curated + paper-mined.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `lineage_parent` | str (stable_id/name) | **[X]** Most direct architectural predecessor. | curated/external | nice | Y |
| `lineage_improves_on_json` | JSON array str | **[X]** Architectures this explicitly improves on (paper intro). | external | nice | Y |
| `lineage_inspired_by_json` | JSON array str | **[X]** Non-direct inspirations cited. | external | stretch | Y |
| `lineage_successor_json` | JSON array str | **[X]** Known direct successors (filled when known). | external/computed | stretch | Y |
| `is_canonical_variant` | bool | Whether this is the main/original variant (vs scale/ablation/adaptation). | curated | nice | Y |
| `variant_of` | str (stable_id/family) | Which canonical model this varies from. | curated | nice | Y |
| `variant_dimension` | str | What varies: scale/depth/width/resolution/efficiency/training/language/hardware. | curated | stretch | Y |

---

## SECTION 6: Key innovation + description  (3 cols)

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `key_innovation` | str (<=120 chars) | One-line "what's novel" from the paper's own framing. | curated/external | core | Y |
| `description` | str (<=500 chars) | 2-4 sentence layperson description of what it does and why it matters. | curated | nice | Y |
| `notes` | str | Sanitized public-facing catalog notes (private notes stripped). | catalog | nice | Y |

---

## SECTION 7: TorchLens structural signature -- the MOAT  (28 cols)

*Direct outputs of running TorchLens on every model. **Every column here is [U].** No static
analysis can recover taken-branch counts, recurrence iterations, live op composition, or
graph topology. This is the section nobody else can fill.*

### 7a. Op count + op-type composition

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `n_compute_ops` | int | **[U]** Non-boundary compute op count (excludes input/output/buffer boundary ops). | trace (`num_compute_ops`) | core | Y |
| `n_unique_op_types` | int | **[U]** Distinct torch func types across all ops (functional vocabulary breadth). | trace | core | Y |
| `op_type_histogram_json` | JSON object str | **[U]** `{func_name: count}` desc -- full primitive distribution. | trace | core | Y |
| `top_op_types_json` | JSON array str | **[U]** Top-5 most-frequent func_names (quick-scan composition). | trace | core | Y |
| `dominant_op_type` | str | **[U]** Single most-frequent op type by count. | trace | nice | Y |
| `pct_conv` | float (0-100) | **[U]** Percent of ops that are convolution primitives. | trace | core | Y |
| `pct_linear` | float (0-100) | **[U]** Percent linear/matmul primitives. | trace | core | Y |
| `pct_attention` | float (0-100) | **[U]** Percent attention primitives (sdpa, flash, softmax+matmul). | trace | core | Y |
| `pct_norm` | float (0-100) | **[U]** Percent normalization primitives. | trace | core | Y |
| `pct_elementwise` | float (0-100) | **[U]** Percent elementwise ops (activations, add, mul, ...). | trace | core | Y |
| `pct_reduction` | float (0-100) | **[U]** Percent reduction ops (mean/sum/max/softmax/...). | trace | core | Y |
| `pct_reshape` | float (0-100) | **[U]** Percent reshape/view/permute ops. | trace | core | Y |
| `pct_embedding` | float (0-100) | **[U]** Percent embedding/lookup primitives. | trace | nice | Y |
| `pct_pooling` | float (0-100) | **[U]** Percent pooling primitives. | trace | nice | Y |
| `n_inplace_ops` | int | **[U]** Count of in-place ops (mutation-heavy indicator). | trace | nice | Y |
| `has_custom_op` | bool | **[U]** Any op outside the standard torch namespace (custom kernels/registered ops). | trace | nice | Y |

### 7b. Graph topology -- depth, width, branching, recurrence

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `graph_depth` | int | **[U]** Longest input->output path in op hops (true topological depth). | trace (`graph_depth_max_from_input`) | core | Y |
| `graph_max_width` | int | **[U]** Max ops at any single BFS depth level (parallelism capacity). | trace | core | Y |
| `branching_factor` | float | **[U]** Mean fan-out (children) per compute op. | trace | nice | Y |
| `max_fan_out` | int | **[U]** Max children of any single op (2=residual, >2=multi-fork). | trace (`max_out_degree`) | nice | Y |
| `max_fan_in` | int | **[U]** Max parents of any single op (>1 = merge/concat/add). | trace (`max_in_degree`) | nice | Y |
| `is_branching` | bool | **[U]** Any op has >1 child (residual/multi-head/fork-merge topology). | trace | core | Y |
| `is_recurrent` | bool | **[U]** Any layer executed more than once (loops/shared weights). | trace | core | Y |
| `max_recurrence_iters` | int | **[U]** Max passes any single layer was executed. | trace (`max_layer_op_count`) | core | Y |
| `n_recurrent_layers` | int | **[U]** Number of layers with >1 pass. | trace | nice | Y |
| `has_conditional_branching` | bool | **[U]** Any data-dependent conditional branch (if/else on tensor values). | trace | core | Y |
| `n_conditionals` | int | **[U]** Count of conditional-flow records. | trace | core | Y |
| `is_dynamic_graph` | bool | **[U]** Graph topology depends on runtime tensor values. | trace | core | Y |

*Dedup note: this section merges S-Sec4/5/6/7 and the trace-signature half of D. `graph_depth`
canonicalizes D's `graph_depth_max_from_input`. `max_fan_out`/`max_fan_in` (S) ==
`max_out_degree`/`max_in_degree` (D) -- S names kept (clearer). The verbose D edge-count family
(`num_edges`/`num_compute_edges`/`num_layer_edges`/...) is dropped as low-value; `branching_factor`
captures the useful summary. Op/module histograms kept ONCE.*

---

## SECTION 8: Structural motifs  (16 cols)

*High-level pattern flags derived from the op/module composition. **All [U]** -- they require the
actual computation graph. The "does this model USE X?" search axis.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `has_attention` | bool | **[U]** Uses attention mechanisms. | trace (`pct_attention>0`) | core | Y |
| `has_self_attention` | bool | **[U]** Self-attention (Q/K/V from same input). | trace (QKV pattern) | nice | Y |
| `has_cross_attention` | bool | **[U]** Cross-attention (encoder-decoder, two input streams). | trace | nice | Y |
| `has_conv` | bool | **[U]** Uses any convolution primitive. | trace (`pct_conv>0`) | core | Y |
| `has_residual` | bool | **[U]** Uses residual/skip connections (add op with skip-connected parents). | trace (topology) | core | Y |
| `has_recurrent_primitive` | bool | **[U]** Uses explicit RNN/LSTM/GRU cell primitives. | trace | core | Y |
| `has_moe` | bool | **[U]** Mixture-of-Experts gating (router + parallel experts). | trace | nice | Y |
| `has_embedding` | bool | **[U]** Embedding lookup ops present (LM/multimodal). | trace (`pct_embedding>0`) | core | Y |
| `has_graph_conv` | bool | **[U]** GNN convolution primitives (torch_geometric patterns). | trace | nice | Y |
| `has_sparse_ops` | bool | **[U]** Sparse tensor ops present. | trace | nice | Y |
| `has_dropout` | bool | **[U]** Dropout ops present in traced graph. | trace | nice | Y |
| `norm_type` | enum str | **[U]** Dominant norm: batch_norm/layer_norm/group_norm/instance_norm/rms_norm/none/mixed. | trace | core | Y |
| `activation_fn_type` | enum str | **[U]** Dominant activation: relu/gelu/silu/elu/leaky_relu/tanh/sigmoid/mish/.../mixed. | trace | core | Y |
| `pooling_type` | enum str | **[U]** Primary pooling: global_avg/global_max/adaptive_avg/max_pool/avg_pool/attention_pool/none/mixed. | trace | nice | Y |
| `has_weight_sharing` | bool | **[U]** Any module/weights reused (Siamese/tied). | trace | nice | Y |
| `structural_barcode_json` | JSON object str | **[U]** Compact "structural DNA" blob: n_ops, graph_depth, max_width, is_recurrent/branching/dynamic, has_attention/conv/residual, norm_type, activation_fn, n_params, n_modules, top_op. Single-column quick characterizer for ML-over-the-catalog. | computed (trace fields) | nice | Y |

---

## SECTION 9: Module hierarchy  (9 cols)

*The nn.Module tree -- invisible to static op analysis. **All [U].***

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `n_modules` | int | **[U]** Total nn.Module instances (all levels). | trace | core | Y |
| `n_module_calls` | int | **[U]** Dynamic module invocation count. | trace | nice | Y |
| `module_max_depth` | int | **[U]** Max nesting depth of the module tree. | trace (`max_module_call_depth`) | core | Y |
| `n_top_level_modules` | int | **[U]** Number of direct children of the root module. | trace | nice | Y |
| `n_unique_module_types` | int | **[U]** Distinct nn.Module subclass types. | trace | core | Y |
| `module_type_histogram_json` | JSON object str | **[U]** `{module_class_name: count}` -- full module distribution. | trace | core | Y |
| `top_module_types_json` | JSON array str | **[U]** Top-5 most-frequent module class names. | trace | core | Y |
| `top_level_block_sequence_json` | JSON array str | **[U]** Ordered top-level module class names (macro architecture recipe). | trace | nice | Y |
| `model_class_name` | str | **[U]** Captured root model class name. | trace | core | Y |

---

## SECTION 10: Parameters + buffers  (15 cols)

*Exact counts/memory from live execution (not `named_parameters()` estimates). **All [U]** except
where computed.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `n_params_trainable` | int | **[U]** Trainable scalar parameters. | trace | core | Y |
| `n_params_frozen` | int | **[U]** Frozen scalar parameters. | trace | core | Y |
| `n_param_tensors` | int | **[U]** Number of distinct parameter tensors. | trace | nice | Y |
| `trainable_param_frac` | float (0-1) | Fraction of scalar params that are trainable. | computed | nice | Y |
| `param_memory_bytes` | int (bytes) | **[U]** Total parameter memory. | trace (`total_param_memory`) | core | Y |
| `primary_param_dtype` | str | **[U]** Most common param dtype by scalar count. | trace | core | Y |
| `param_dtype_set_json` | JSON array str | **[U]** Distinct parameter dtypes (mixed-precision detector). | trace | nice | Y |
| `quantized_param_tensor_count` | int | **[U]** Count of quantized parameter tensors. | trace | nice | Y |
| `has_frozen_params` | bool | **[U]** Any frozen params (pretrained/fine-tune pattern). | trace | core | Y |
| `shared_param_tensor_count` | int | **[U]** Param tensors registered at multiple addresses (weight tying). | trace | nice | Y |
| `n_buffers` | int | **[U]** Registered persistent buffer count. | trace | core | Y |
| `buffer_memory_bytes` | int (bytes) | **[U]** Memory held by registered buffers. | trace | nice | Y |
| `buffer_overwrite_count` | int | **[U]** Total buffer write events during forward (stateful-forward signal). | trace | nice | Y |
| `nontrainable_state_bytes` | int (bytes) | Frozen-param bytes + buffer bytes. | computed | nice | Y |
| `embedding_param_count` | int | **[U]** Scalar params in embedding-like modules (heuristic). | trace | nice | Y |

---

## SECTION 11: Compute + memory at reference input  (18 cols)

*FLOPs/MACs and activation memory from the reference forward pass. **All [U]** trace facts or
computed from them.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `total_flops_forward` | int (FLOPs) | **[U]** Forward FLOPs at reference input (unknown ops skipped). | trace | core | Y |
| `total_macs_forward` | int (MACs) | **[U]** Forward MACs (1 MAC = 2 FLOPs). | trace | core | Y |
| `per_sample_macs_forward` | float (MACs/sample) | Forward MACs normalized by reference batch size. | computed | nice | Y |
| `flops_coverage_pct` | float (0-100) | **[U]** Percent of ops with a non-null FLOP estimate. | trace | core | Y |
| `n_unknown_flops_ops` | int | **[U]** Count of ops without a FLOP estimate. | trace | nice | Y |
| `flops_by_op_type_json` | JSON object str | **[U]** Forward FLOPs grouped by op type. | trace | nice | Y |
| `dominant_compute_op_type` | str | **[U]** Op type contributing the largest forward FLOP total. | trace | nice | Y |
| `pct_flops_from_attention` | float (0-100) | **[U]** Percent of FLOPs from attention primitives. | trace | nice | Y |
| `pct_flops_from_conv` | float (0-100) | **[U]** Percent of FLOPs from convolution primitives. | trace | nice | Y |
| `pct_flops_from_linear` | float (0-100) | **[U]** Percent of FLOPs from linear/matmul primitives. | trace | nice | Y |
| `flops_per_param` | float | Forward FLOPs / scalar param count (compute intensity). | computed | nice | Y |
| `total_activation_memory_bytes` | int (bytes) | **[U]** Total activation tensor volume across forward ops. | trace | core | Y |
| `forward_peak_memory_bytes` | int (bytes) | **[U]** Estimated forward peak memory (high-water mark). | trace | core | Y |
| `largest_activation_bytes` | int (bytes) | **[U]** Largest single-op output activation (bottleneck op). | trace | nice | Y |
| `largest_activation_op_type` | str | **[U]** Op type producing the largest activation. | trace | nice | Y |
| `activation_to_param_memory_frac` | float | Activation volume / parameter memory (>1 = activation-dominated). | computed | nice | Y |
| `forward_working_set_bytes` | int (bytes) | Practical forward-memory proxy: param bytes + forward peak. | computed | core | Y |
| `total_macs_backward` | int (MACs) | **[U]** Backward MACs when backward capture exists (usually null for menagerie). | trace | nice | Y |

---

## SECTION 12: Input / output contract  (16 cols)

*The reference input used for capture, and the model's I/O shape. Catalog + trace.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `input_shape` | str | Example input tensor shape string (e.g. "(1,3,224,224)"); multi-input lists all. | catalog | core | Y |
| `input_dtype` | str | Example input dtype (e.g. "float32", "long"). | catalog | core | Y |
| `input_kind` | enum str | Typed input builder kind: tensor/multi/kwargs/none/callable. | catalog (JSONL) | core | Y |
| `input_is_real` | bool | Runtime input is a real example, not an ignored sentinel. | catalog | core | N |
| `input_rank` | int | Rank of primary tensor input. | computed | nice | Y |
| `reference_batch_size` | int | Batch size from first dim of primary tensor input. | computed | core | Y |
| `input_resolution` | str | Primary spatial resolution (e.g. "224x224") for image/video. | computed | nice | Y |
| `input_sequence_length` | int | Sequence length for NLP/audio/time-series models. | computed | nice | Y |
| `input_scale` | float | Validation input scale when too-large models were down-scaled (1.0 = unscaled). | catalog/validator | core | Y |
| `output_shape` | str | **[U]** Shape of primary output at reference input. | trace | core | Y |
| `output_dtype` | str | **[U]** Dtype of primary output (long=classification, float=regression). | trace | nice | Y |
| `n_output_tensors` | int | **[U]** Number of tensor leaves in model output. | trace | core | Y |
| `output_container_kind` | str | **[U]** Output container: tensor/tuple/list/dict/dataclass/model-output. | trace | nice | Y |
| `output_num_classes` | int | **[U]** Number of output classes when resolved. | trace | nice | Y |
| `output_type` | enum str | Output semantics: logits/boxes/mask/waveform/token-logits/embedding/policy/reconstruction. | computed/external | core | Y |
| `is_multi_output` | bool | **[U]** Output has multiple tensor leaves. | trace | nice | Y |

---

## SECTION 13: Validation + trust  (16 cols)

*The honesty layer -- only we have it. Mostly **[U]** ledger facts. `is_trustworthy` and
`validation_status` are promoted to Section 1.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `verified` | bool | Catalog-level instantiable signal (legacy, looser than `is_trustworthy`). | catalog | core | N |
| `forward_pass` | bool | **[U]** Forward validation passed (replay within tolerance). | ledger | core | Y |
| `backward_pass` | bool | **[U]** Backward validation passed when attempted. | ledger | nice | Y |
| `backward_na_reason` | str | **[U]** Why backward was N/A (no_loss/inference_only/non_differentiable_output). | ledger | nice | Y |
| `metadata_ok` | bool | **[U]** TorchLens metadata-invariant validation passed. | ledger | core | Y |
| `trust_tier` | enum str | Trust bucket: current_verified/stale_recipe/legacy_seed/failed/deferred/unsupported. | computed | core | N |
| `current_recipe_matches_catalog` | bool | Latest run used the row's current recipe hash. | computed | core | Y |
| `verification_expectation` | enum str | Declared posture: forward_required/deferred/unsupported. | catalog | core | N |
| `is_classics` | bool | Hand-built historical reimplementation from `menagerie/classics/`. | catalog (`source=="classics"`) | core | N |
| `validation_error_class` | str | **[U]** Error/status class for failed/skipped validations. | ledger | nice | Y |
| `validation_error_message` | str | **[U]** Sanitized error message / skip reason (public-safe). | ledger | nice | Y |
| `validation_duration_sec` | float (sec) | **[U]** Wall-clock duration of the latest verification run. | ledger | nice | Y |
| `peak_rss_mb` | int (MB) | **[U]** Peak RSS of the isolated validation worker (ledger stores MB). | ledger | nice | Y |
| `validation_finished_at_utc` | datetime UTC | **[U]** Timestamp of the latest verification run. | ledger | nice | Y |
| `validation_attempt_count` | int | Number of ledger attempts for this stable_id. | computed | nice | N |
| `trace_completeness` | enum str | full-trace/partial-trace/validation-only/catalog-only. | computed | core | N |

*Dedup note: this merges P-Sec9, Pr-Sec02/14, S-Sec3/14, D-validation. `validation_duration_sec`
== Pr `validation_duration_sec` == S `forward_duration_sec` -- one name. `peak_rss_mb` is the ONE
MB column (ledger-native). The verbose D timestamp set (`started_at`/`last_success_at`/
`last_failure_at`) collapses to `validation_finished_at_utc` + `validation_attempt_count`.*

---

## SECTION 14: Identity hashes + deduplication  (13 cols)

*The fingerprints that power "how many DISTINCT architectures?" Mix of **[U]** trace hashes and
computed dedup keys.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `recipe_revision_sha256` | str (64 hex) | Constructor-only recipe fingerprint (input EXCLUDED by sprint decision). | catalog | core | N |
| `graph_shape_hash` | str (hex) | **[U]** Shape-blind graph-topology hash (exact-design dedup). | ledger/trace | core | Y |
| `architecture_param_shape_hash` | str (hex) | **[U]** Architecture hash with param shapes IN, input-activation shapes OUT (sprint refinement). | trace (computed) | core | Y |
| `raw_event_shape_hash` | str (hex) | **[U]** Shape-aware capture-event hash (catches width/shape drift; replay/migration fidelity). | trace (`_raw_event_shape_hash`) | nice | Y |
| `param_hash_full` | str (hex) | **[U]** Full parameter-structure fingerprint. | trace | nice | Y |
| `input_signature_hash` | str (hex) | **[U]** Input shape/dtype/device fingerprint (explicitly NOT recipe identity). | trace | nice | Y |
| `svg_sha256` | str (64 hex) | **[U]** Content hash of the rendered SVG artifact (cache-busting). | ledger | nice | Y |
| `dedup_architecture_key` | str | Recommended "same architecture" grouping key: prefer `architecture_param_shape_hash`, else `graph_shape_hash`. | computed | core | Y |
| `is_architecture_representative` | bool | Whether this is the row to keep when collapsing an architecture duplicate group. | computed | core | Y |
| `canonical_representative_stable_id` | str | stable_id chosen to represent the architecture duplicate group. | computed | nice | Y |
| `architecture_duplicate_count` | int | Number of rows sharing this `dedup_architecture_key` (1 = unique design). | computed | nice | Y |
| `dedup_recipe_key` | str | Key collapsing rows with identical current construction recipe. | computed (recipe hash) | nice | N |
| `natural_key` | str | Human-auditable identity key (`name|zoo|variant`). | computed | nice | N |

*Dedup note: `architecture_hash` (S) == `graph_shape_hash` (everyone) -- ONE name, `graph_shape_hash`.
`structural_fingerprint` (S) == `raw_event_shape_hash` (Pr/D) -- ONE name, `raw_event_shape_hash`.
`is_architecture_unique`/`architecture_duplicate_of` (S) folded into the D dedup-key family.
`param_hash_quick` dropped in favor of `param_hash_full`. `env_hash`, `deterministic_barcode_hash`,
`near_duplicate_group_id` dropped as low-value/speculative for a public CSV.*

---

## SECTION 15: Runtime + capture provenance  (12 cols)

*Reproducibility metadata: versions, device, backend used for the reference capture. Ledger + trace.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `constructor_call` | str | Public construction expression (random-init). | catalog | core | N |
| `recipe_uses_code_execution` | bool | Recipe uses eval/exec/statement execution. | catalog | core | Y |
| `quarantine` | bool | Recipe uses a quarantined executable-string form. | catalog | core | N |
| `random_init_recipe` | bool | Recipe constructs random-init weights (no pretrained download). | computed/catalog | core | Y |
| `pretrained_weights_available` | bool | **[X]** Pretrained weights publicly available for this architecture. | external | core | Y |
| `torchlens_version` | str | **[U]** TorchLens version used for the current validation. | ledger | core | Y |
| `torch_version` | str | PyTorch version used for the current validation. | ledger | core | Y |
| `python_version` | str | Python version used for the current validation. | ledger | nice | Y |
| `backend` | str | TorchLens capture backend (e.g. "torch"). | trace | core | Y |
| `device_actual` | str | Device actually used for verification (cpu/cuda/mps). | ledger | core | Y |
| `inference_only` | bool | Forward capture ran under inference/no-grad mode. | trace | nice | Y |
| `dependency_cluster` | str | Validation environment / dependency cluster for install grouping. | computed (`menagerie.runtime`) | nice | Y |

*Dedup note: `device_requested` dropped in favor of `device_actual` (what was used matters);
`runner_host`, `env_hash`, `backend_runtime_*`, `capture_mode`, `save_mode`, `random_seed`,
`tlspec_version` dropped as internal reproducibility minutiae not worth a public column. The
license/weights enrichment lives mostly in Section 16.*

---

## SECTION 16: Installability + practitioner enrichment  (17 cols)

*Practitioner sizing and usability. Mix of computed install signals (in hand) and **[X]** external
license/hardware/benchmark fields.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `required_packages_json` | JSON array str | Pip packages required by the dependency plan. | computed (`menagerie.runtime`) | core | Y |
| `install_difficulty` | enum str | core/pip/pinned-env/source-build/unavailable. | computed | core | Y |
| `requires_custom_ops` | bool | Likely needs compiled/custom ops (mmcv/detectron2/dgl/triton/...). | computed (heuristic) | nice | Y |
| `requires_network_download` | bool | Construction/pretrained load needs network access. | computed (static audit) | core | Y |
| `license` | str (SPDX-ish) | **[X]** Model/code/weights license. | external | core | Y |
| `commercial_use_ok` | bool | **[X]** License permits commercial use. | computed (license) | core | Y |
| `training_data` | str | **[X]** Known training dataset/corpus for pretrained weights. | external | stretch | Y |
| `training_regime` | str | **[X]** supervised/self-supervised/RL/diffusion-pretraining/instruction-tuning. | external | stretch | Y |
| `typical_hardware_tier` | enum str | **[X]** cpu/laptop_gpu/single_consumer_gpu/single_datacenter_gpu/multi_gpu. | computed/external | nice | Y |
| `min_inference_ram_mb` | int (MB) | Minimum host RAM for a single reference inference. | computed (peak_rss/working-set) | nice | Y |
| `mobile_edge_suitable` | bool | Plausibly deployable on mobile/edge (params/memory/FLOPs/license rubric). | computed (heuristic) | nice | Y |
| `max_context_length` | int (tokens) | **[X]** Max usable context/window for text/audio/time-series. | external (sometimes ctor) | nice | Y |
| `autoregressive` | bool | Inference is autoregressive/sequential. | computed/external | nice | Y |
| `generative_sampling_required` | bool | Normal use needs an external sampler/decoder loop beyond one forward. | computed/external | core | Y |
| `is_stateful_forward` | bool | **[U]** Forward mutates persistent state/buffers (`buffer_overwrite_count>0`). | trace | nice | Y |
| `inference_notes` | str | Short practitioner caveat (sentinel input / sampler needed / output not decoded). | computed | core | Y |
| `metric_confidence` | enum str | Confidence for practitioner metrics: observed/trace-derived/heuristic/external/unknown. | computed | core | N |

*Dedup note: this consolidates Pr-Sec10/11/12/13 and the stretch tail of P. The full latency/
throughput/energy benchmark family (`cpu_latency_ms`, `gpu_throughput_samples_per_sec`,
`energy_mj_per_inference`, `benchmark_*`) is DROPPED for v1 -- it requires a benchmark harness we
do not have; see OPEN QUESTION 4. VRAM columns dropped (CPU-only validation ledger can't fill them).*

---

## SECTION 17: Artifacts + dataset admin  (10 cols)

*Links to rendered artifacts and dataset-release bookkeeping. Computed/build-time.*

| name | type+unit | description | source | priority | nullable |
|---|---|---|---|---|---|
| `model_page_url` | str | Public modelmenagerie.ai page for this row. | computed (stable_id) | core | N |
| `svg_path` | str | Repo/site-relative SVG graph artifact path. | computed (artifact manifest) | core | Y |
| `svg_url` | str | Public URL for the SVG graph. | computed | nice | Y |
| `has_svg` | bool | Whether a rendered SVG exists. | computed (`svg_sha256` present) | nice | N |
| `tlspec_path` | str | Portable TorchLens trace bundle path, if shipped. | computed (artifact manifest) | stretch | Y |
| `added_wave` | str | Harvest/discovery sweep that added this model. | catalog | nice | Y |
| `dataset_schema_version` | str (semver) | Version of THIS CSV schema. | build metadata | core | N |
| `dataset_as_of_date` | str (YYYY-MM-DD) | Public CSV as-of date. | build metadata | core | N |
| `csv_generation_commit_sha` | str (git sha) | Git commit used to generate the CSV. | build metadata | core | Y |
| `row_quality_flags_json` | JSON array str | Machine-readable warnings: stale_recipe/missing_paper/no_artifact/hash_collision_check. | computed | core | N |

*Dedup note: collapses P-Sec12, D-links, D-dataset-ops. The verbose D build-metadata set
(`dataset_row_version`/`row_created_at`/`row_updated_at`/`csv_generation_command`/
`csv_generator_version`/`deprecation_status`/`replacement_stable_id`/`png_path`/`thumbnail_path`/
`dagua_json_path`/`facet_tokens`) is dropped or folded -- `row_quality_flags_json` carries the
machine-readable warnings; `dataset_schema_version` + `dataset_as_of_date` + `csv_generation_commit_sha`
are the minimal reproducibility triple.*

---

## Section index + counts

| # | Section | Cols | Notes |
|---|---|---|---|
| 1 | At-a-glance identity + browse | 15 | the DEFAULT VIEW / first columns |
| 2 | Extended identity + display | 8 | |
| 3 | Taxonomy + classification | 16 | faceted-browse axes |
| 4 | Paper provenance + external references | 19 | mostly [X] external |
| 5 | Lineage + genealogy | 7 | [X] curated |
| 6 | Key innovation + description | 3 | |
| 7 | TorchLens structural signature (THE MOAT) | 28 | all [U] |
| 8 | Structural motifs | 16 | all [U] |
| 9 | Module hierarchy | 9 | all [U] |
| 10 | Parameters + buffers | 15 | mostly [U] |
| 11 | Compute + memory at reference input | 18 | all [U] |
| 12 | Input / output contract | 16 | catalog + [U] trace |
| 13 | Validation + trust | 16 | mostly [U] ledger |
| 14 | Identity hashes + deduplication | 13 | [U] hashes + computed keys |
| 15 | Runtime + capture provenance | 12 | reproducibility |
| 16 | Installability + practitioner enrichment | 17 | computed + [X] external |
| 17 | Artifacts + dataset admin | 10 | build-time |
| | **TOTAL** | **138** | |

### By priority
- **core:** ~74
- **nice:** ~50
- **stretch:** ~14

### By source (where the data comes from)
- **In hand** (catalog + ledger + trace + computed): ~103
- **MENAGERIE-UNIQUE [U]** (trace-derived, the moat): ~78 of those
- **External lookup required [X]** (paper/license/citations/hardware): ~35

### The AT-A-GLANCE set (15 columns, Section 1, in order)
`stable_id`, `display_name`, `family_normalized`, `variant`, `zoo`, `domain`, `task_primary`,
`modality_primary`, `architecture_paradigm`, `n_params`, `publication_year`, `era`,
`validation_status`, `is_trustworthy`, `n_ops`.

---

## OPEN QUESTIONS for JMT (genuine cross-proposal disagreements / judgment calls)

1. **Memory units: bytes vs MB.** S and D say store raw `_bytes` for ALL trace memory; P and Pr
   want `_mb` floats for spreadsheet sorting. I chose **`_bytes` everywhere for trace-derived
   memory** (lossless, one convention) and kept `_mb` ONLY for `peak_rss_mb` (ledger-native) and
   the practitioner sizing field `min_inference_ram_mb`. Want a parallel `_mb` view for the big
   three (`param_memory_mb`, `total_activation_memory_mb`, `forward_peak_memory_mb`) for
   spreadsheet humans, accepting the redundancy? My call: NO -- the website can format bytes->MB
   on display; the CSV stays canonical.

2. **Boolean serialization: 0/1 vs true/false.** S/D say `0`/`1` ints; nothing says `true`/`false`.
   I locked **0/1 ints**. Confirm -- some pandas/R users prefer literal booleans. (Easy to flip
   globally; just needs to be ONE way.)

3. **List serialization: JSON arrays vs pipe-delimited.** S used pipe-delimited (`conv2d|linear`),
   D/Pr used JSON arrays. I locked **JSON arrays everywhere** (`_json` suffix) for one consistent
   parse path. Pipe-delimited is friendlier to a human eyeballing the raw CSV; JSON is friendlier
   to programmatic load. My call: JSON. Confirm.

4. **Benchmark columns (latency/throughput/VRAM/energy).** Pr proposed a large hardware-benchmark
   family (`cpu_latency_ms`, `gpu_latency_ms`, `*_throughput_samples_per_sec`, `min_inference_vram_mb`,
   `energy_mj_per_inference`, `benchmark_*`). I **DROPPED all of these for v1** -- they require a
   real benchmark harness across hardware we do not have, and the menagerie validation ledger is
   CPU-RSS-only. Keep them out (my recommendation), or reserve placeholder stretch columns to
   signal intent?

5. **`raw_event_shape_hash` portability.** Pr/D flag that this hash exists on the Trace
   (`_raw_event_shape_hash`) but is NOT in `Trace.PORTABLE_STATE_SPEC` today, so the exporter must
   persist it separately. It's high-value for migration-fidelity dedup. Confirm we're willing to
   add the export plumbing, or demote it to stretch.

6. **`description` / `key_innovation` authoring cost.** Both are `core`/`nice` but require
   per-model curation across 11,670 rows. Realistically these get filled for classics + milestones
   first and stay null for the long tail. Accept a mostly-null `core` column, or demote to `nice`?
   (I left `key_innovation`=core, `description`=nice.)

7. **Trust column naming.** I have THREE trust-ish columns: `validation_status` (raw ledger enum,
   at-a-glance), `is_trustworthy` (strict computed gate, at-a-glance), `trust_tier` (graded bucket,
   Section 13), plus catalog `verified`. This is intentional (raw / boolean-gate / graded / legacy)
   but it's four columns about "is this good." Confirm you want all four, or collapse `verified`
   (the loosest, legacy) into a `trust_tier` value.
