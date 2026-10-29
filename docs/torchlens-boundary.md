# TorchLens boundary

Menagerie depends on TorchLens as a pinned PyPI release (`pyproject.toml`, `torchlens==2.36.0`).
Almost all calls use the public API: `tl.trace`, `tl.options`, `torchlens.options.CaptureOptions`,
`tl.save`, `tl.load`, `tl.validate`, `tl.validation`, `tl.__version__`.

Private TorchLens calls (decision 002: Menagerie adapts when TorchLens refactors). Recheck these
whenever the pin moves:

| Private symbol | Used in |
| --- | --- |
| `torchlens.user_funcs._validate_forward_pass_torch` | `menagerie/pipeline/worker.py`, `menagerie/validate_menagerie.py`, `menagerie/trace_equivalence_audit.py`, `tests/test_menagerie_trace_equivalence.py`; patched by name in `tests/test_menagerie_validation_scheduler.py` |
| `torchlens.validation.diagnostics.get_validation_failure` | `menagerie/validate_menagerie.py` |
| `torchlens.backends.torch.wrappers.wrap_torch` (the path `tl.wrap_torch`'s removal note names) | `menagerie/pipeline/worker.py` (`_wrap_torch`, legacy pins only; disabled at 2.36.0) |
| `torchlens._io.bundle._prescan_metadata_pickle_opcodes` (saved metadata above 8 MiB; reader limit 256 MiB) | `menagerie/pipeline/replay_table.py` |
| `torchlens._io.bundle._RenameAwareUnpickler` (reads slim-trace metadata without `tl.load`) | `menagerie/pipeline/replay_table.py` |

Since 2.35.2, `_validate_forward_pass_torch` calls `_trace_observer(None)` when it refuses a
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
on a CPU device). The validation capture cannot: at 2.35.3 `_validate_forward_pass_torch` takes no
capture options, so it saves every activation and argument on the card under `"auto"` (half the
card's free memory at its first save), and a model whose validation retains more stops with
`SaveBudgetExceededError` (m4934, 2026-10-05). That limit lifts only when TorchLens's validator
accepts capture options. A JAX try (`worker_preview._jax_contract`) runs one
`tl.trace(..., backend="jax")` and fails `capture_error:jax_contract` on TorchLens's
`BackendMismatchError` or its closed-jaxpr-constants refusal, matching that refusal's text.

The crawler drives TorchLens in subprocess workers and imports none of it directly; comment-only
references (`torchlens._io.payload_codec`, `torchlens/backends/...`) are not dependencies.

Moving the pin: pick a released TorchLens version from PyPI, update `pyproject.toml`, run
`pytest` on a worker, and fix Menagerie (never TorchLens) where it breaks.

At 2.36.0 the stored-GELU reference repair is upstream. `worker._wrap_torch` returns without
wrapping, and the repin day check exercises BERT, ViT and depth_pro twice in each of two fresh
processes through the normal hash-capture path. The private-call probe exercises the validator,
its observer and the diagnostic reader; importing the old wrapper only verifies compatibility.

`repin check VERSION --part day` runs the hash harness and private/GELU probes. The Slice A
part uses the frozen regression batches and `--deadline` for a bounded shutdown. `--smoke`
uses the declared smaller allocation and never qualifies a flip. `repin flip VERSION --dry-run`
lists the carried and replay rows before old-pin passes; the real flip requires complete matching
check evidence and a separately supplied drain/approval record. Its queue reuses settled forward
rows at the same recipe revision. TorchLens is anchored out of recipe-version identity and remains
an independent trace-currency field, so changing only its pin invalidates traces, not forwards.
