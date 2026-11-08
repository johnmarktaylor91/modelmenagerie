# TorchLens boundary

Menagerie depends on TorchLens as a pinned PyPI release (`pyproject.toml`, `torchlens==3.0.2`).
Almost all calls use the public API: `tl.trace`, `tl.options`, `torchlens.options.CaptureOptions`,
`Trace.save`, `tl.load`, `tl.validate`, `tl.validation`, `tl.__version__`.

Private TorchLens calls (decision 002: Menagerie adapts when TorchLens refactors). Recheck these
whenever the pin moves; status checked 2026-11-06 against 2.36.0, 3.0.1 and 3.0.2:

| Private symbol | Used in | At 3.0.2 |
| --- | --- | --- |
| `torchlens.user_funcs._validate_forward_pass_torch`, only through `menagerie/tl_validation.py` | the worker, the June validator, the trace-equivalence audit and the repin probe all call `tl_validation.validate_forward`; tests patch `tl_validation._legacy_validator` | gone: the module is private (`torchlens._user_funcs`), and Menagerie refuses it (below) |
| `torchlens.backends.torch.wrappers.wrap_torch` (the path `tl.wrap_torch`'s removal note names) | `menagerie/pipeline/worker.py` (`_wrap_torch`, before every build at every pin since 2026-10-27); `repin_checks.private_calls` checks it imports | present, same path |
| `torchlens._io.bundle._prescan_metadata_pickle_opcodes` (saved metadata above 8 MiB; reader limit 256 MiB) | `menagerie/pipeline/replay_table.py`, only for a save with `metadata.pkl` (2.37 and earlier) | gone (the bundle unpickler prescans every load itself) |
| `torchlens._io.bundle._RenameAwareUnpickler` (reads slim-trace metadata without `tl.load`) | `menagerie/pipeline/replay_table.py`, only for a save with `metadata.pkl` (2.37 and earlier) | present at 2.38.1, 3.0.1 and 3.0.2, but 3.0 refuses the 2.x record classes a real save pickles |

`replay_table` picks its reader by the save: a 2.37-and-earlier save (`metadata.pkl`) goes through
the two private calls above, which only a TorchLens before 2.38 has in full, so under a later
TorchLens its row records `OldSaveError` naming the `torchlens<2.38` reader environment; a
2.38-and-later save (typed records, no pickle) goes through public `tl.load`. A table at an old pin
is therefore run in the reader environment, a table at a new pin in the main one.

The failure readers are public: `tl.validation.get_validation_failure` and
`tl.validation.last_validation_failure` (both exported at 2.36.0 and 2.38.1), reached through
`tl_validation.failure_for` and `tl_validation.last_failure`. A failure names its check as `check`
through 2.37 and `check_name` from 2.38; `tl_validation.failure_check` reads either.

`menagerie/tl_validation.py` is the one validator door. TorchLens's public `tl.validate` returns a
bool and frees its trace, and takes no thread count; the worker needs the validation trace's op
count, graph hash and replay status, read before cleanup, and a single-thread retry. The door uses
public `tl.validate(scope="forward", trace_observer=...)` once TorchLens's `tl.validate` accepts
`trace_observer`, which TorchLens 3.0.1 does with `num_threads`, `output_device` and
`save_budget` (2.38.1 did not; where `num_threads` is not a keyword the door pins
`torch.set_num_threads` around the call, which TorchLens confirmed is equivalent). Before that it uses the validator in `torchlens.user_funcs` (through 2.37), and on a
TorchLens with neither it raises `TorchLensBoundaryError`, which the worker records as
`capture_error`. Menagerie never imports a `torchlens._*` module for validation. The public bool
plus `last_validation_failure()` alone gives the verdict and failure check, but not the op count,
graph hash or failed node count of the validated trace.

Names that differ between the pin and 2.38 go through `menagerie/tl_compat.py`, which reads the
installed TorchLens and uses whichever name it offers: `num_ops` (`trace.num_ops`, then
`len(trace.ops)`), `graph_hash` (`trace.graph_shape_hash`, then `trace.provenance.fingerprint`),
`skeleton` (`tl.hash.trace`, then `tl.fingerprint.trace`), `draw` (the `vis_*` keywords of
`Trace.draw` through 2.37; for 3.0's `Trace.draw`, which refuses them, `rolled=`, `path=`,
`file_format=` and `layout=`, with `vis_save_only` dropped because 3.0 opens no viewer by default;
`render.DRAW_OPTIONS` stays in the pinned names, so `renderer_rev` does not move), `status_count` (`failed_node_count` and
the other `*_node_count` fields, then `num_failed_ops` and the other `num_*_ops`), `capture_options`
(`random_seed=`, `intervention_ready=` and `save_rng_states=` become `seed=`, `replayable=` and
`save_rng=`; `layers_to_save=` is dropped where `tl.trace(save=...)` carries it) and `save`
(`tl.save(trace, path)`, then `trace.save(path)`), `layer_list` (`trace.layer_list`, then iterating
the trace), `layers` and `layer_calls` (`trace.layer_logs` and `trace.layer_num_calls`, then
`trace.layers`), `layer_has_params` (`uses_params`, then `has_params`), `op_type` (`op.layer_type`,
then `op.layer.layer_type`) and `trace_field` (trace-level totals and graph facts, which 2.38 moved
into `trace.cost` and `trace.graph`, `is_dynamic_graph` and `max_layer_op_count` renamed
`is_dynamic` and `max_ops_per_layer`; the module identity mode a preview row records, which 2.38
moved to `trace.options`). Nothing else names a version-specific TorchLens attribute.
From 3.0 the summary's remaining trace fields have homes too: `source.class_name` and
`class_qualname`, `layers.recurrent`, `cost.macs_by_kind`, and counts read as the length of
`ops.compute`, `graph.conditionals`, `modules`, `module_calls`, `params` and `buffer_write_ops`
(`TRACE_FIELD_COUNTS`); `op_field` reads the per-op fields 3.0 moved into groups (`cost`,
`graph`, `module_context`, `source`, `call`; `OP_FIELD_HOMES`, from TorchLens's removed-spellings
ledger `docs/reference/deprecations.md`). Checked against 2.36.0 on the same models (report
`2026-11-06-tl301-readiness`). The CPU forward peak, which 2.36 measured on every capture, 3.0
measures only under `measure_host_peak_memory=True`, so the summary captures ask for it
(`tl_compat.CAPTURE_WHERE_ACCEPTED` drops the keyword at the pin). Two summary values differ at 3.0.1
on those models and stay unmapped until their meaning is settled: `total_flops_backward` (0 without a
backward pass) and the graph degree facts (`branching_factor`, `max_out_degree`, `is_branching`). Positional `tl.trace(model, inputs)` and `input_kwargs=` work on both.

The op count, the graph hash, the skeleton hash and Menagerie's own `architecture` and `model`
digests change meaning at 2.38 (the fingerprint is address-free, and the per-op records under the
digests changed). Each stored value sits beside the TorchLens version that made it (the ledger
row's `tl`, a published row's `torchlens_version`), and `tl_compat.same_value` refuses a pair from
different value families (`MixedPinError`, families in `tl_compat.FAMILY_BREAKS`) rather than call
them equal or different. The regression pin check, the repin slice comparison and the published
distinct-architecture count use it; a pair inside one family (2.35.3 against 2.36.0) still compares.

Moving the pin past 2.37 needs one thing the adapters do not give: files saved by 2.37 and
earlier do not load from 2.38 on (`ArtifactTooOldError`, and TorchLens ships no migration), so the
store's saved traces need a `torchlens<2.38` reader environment or a re-capture (research reports
`2026-11-05-tl238-readiness` and `2026-11-06-tl301-readiness`).

The reader environment is the torch-main environment built from the last 2.36.0 lock,
`~/.cache/menagerie/envs/torch-main-cd598953` on the trace host; base environments are keyed by
their lock and never pruned (only overlays are), so it stays after the pin moved to 3.0.2
(2026-11-06, `decisions.md`). Run old-pin readers with its interpreter from the current checkout,
whose adapters run at either pin, for example `replay table --pin 2.36.0`. Keep it until every
published and validated model has a 3.0.2 trace, then archive the 2.35.3 and 2.36.0 saves with a
verified restore and remove it. A 3.0 save made by another backend (TensorFlow, JAX) loads only
where that backend is installed: `tl.load` raises `BackendRuntimeCompatibilityError` in torch-main,
which `replay_table` reports as `BackendSaveError` naming the environment to read it in.

Since 2.35.2, the validator calls the observer with `None` when it refuses a
precondition before building a trace, and `get_validation_failure(None)` returns that refusal's
reason. Menagerie's observers skip trace summarization on `None` and still read the reason.

The worker's observer also reads the validation trace's public `validation_replay_status`
(`state`, `reason`): `unverified` (at 2.35.3 the retrace check's downgrade, reason
`trace_retrace_structure_mismatch`) is typed `torchlens_unverified`, any other non-pass
`torchlens_gap:replay`. The validation and save captures follow one warm-up forward
(`worker._warm`); the hash family's captures do not.

On a GPU (`gpu` mode), the digest and save captures keep their saves in host memory
(`CaptureOptions(output_device="cpu")`) under an absolute `save_budget` of half the attempt's
memory cap (`worker.gpu_save_options`; TorchLens's own `"auto"` reads the whole node's free memory
on a CPU device). The validation capture takes the same two options through the door
(`tl_validation.validate_forward(save_options=...)`, from `worker._validate`): TorchLens's
validator accepts `output_device` and `save_budget` since 2.36.0, and public `tl.validate` too, so
the validation keeps its saves in host memory under the attempt's budget instead of `"auto"` on the
card, the cause of m4934's `SaveBudgetExceededError` (2026-10-05; not yet re-run on a GPU). On the
CPU nothing is passed. A JAX try (`worker_preview._jax_contract`) runs one
`tl.trace(..., backend="jax")` and fails `capture_error:jax_contract` on TorchLens's
`BackendMismatchError` or its closed-jaxpr-constants refusal, matching that refusal's text.

The crawler drives TorchLens in subprocess workers and imports none of it directly; comment-only
references (`torchlens._io.payload_codec`, `torchlens/backends/...`) are not dependencies.

Moving the pin: pick a released TorchLens version from PyPI, update `pyproject.toml`, run
`pytest` on a worker, and fix Menagerie (never TorchLens) where it breaks.

At 2.36.0 the stored-GELU reference repair is upstream, and the repin day check exercises BERT,
ViT and depth_pro twice in each of two fresh processes through the normal hash-capture path. The
worker still pre-wraps torch before every build (`worker._wrap_torch`) for stale aliases the
validator does not rescue (m19908 `bmm`, m20502 `cat`). The private-call probe exercises the
validator door, its observer and the diagnostic reader, and checks the wrapper imports.

`repin check VERSION --part day` runs the hash harness and private/GELU probes. The Slice A
part uses the frozen regression batches and `--deadline` for a bounded shutdown. `--smoke`
uses the declared smaller allocation and never qualifies a flip. `repin flip VERSION --dry-run`
lists the carried and replay rows before old-pin passes; the real flip requires complete matching
check evidence and a separately supplied drain/approval record. Its queue reuses settled forward
rows at the same recipe revision. TorchLens is anchored out of recipe-version identity and remains
an independent trace-currency field, so changing only its pin invalidates traces, not forwards.
