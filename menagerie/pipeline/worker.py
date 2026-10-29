"""The in-sandbox worker: build, forward, trace, validate, save and summarize one model.

Owned by the worker lane. Modes: ``forward`` runs eval then train on freshly built models and
classifies the divergence; ``trace`` times a plain forward, takes the verdict, op count and graph
hash from the June validator's entry, the summary and hash family from June's digest capture, and
saves a third capture made ``intervention_ready`` (unless the request turns it off) runnable with
a round-trip check; ``trace-structure-only`` captures on the meta device; ``render`` hands the
saved trace to ``render.render_set``; ``gpu`` is the trace mode on a CUDA device, only inside the
GPU window. Each capture is of a freshly built model. Torch is imported inside functions.

A recipe whose ``backend`` is not ``torch`` (TensorFlow, JAX, Paddle, MLX) runs its forward and
trace modes in ``worker_preview``, through TorchLens's public backend surface.

Validation goes through TorchLens's private ``_validate_forward_pass_torch``, exactly as the June
validator calls it (``docs/torchlens-boundary.md``), so verdicts are the June validator's at the
pin. The public ``validate_forward_pass`` returns ``unverified`` unless the capture keeps every
activation and argument; with those it agreed with the private entry on the eleven models checked
at 2.35.2, but it skips the private entry's retrace check and witness mode. A failed verdict is
retried once on a fresh model with one thread, as the June validator does.

The validation and the save capture a freshly built model warmed by one plain forward
(``_warm``; Lead, 2026-10-05): timm fills an attention-bias cache on the first eval call, so a
never-run model's pristine re-trace differs and TorchLens reports it ``unverified``. The hash
family's captures (digest, structure-only, the hash harness) stay on a never-run build: a warmed
cache enters the graph as a tensor with an input-sized shape (ultralytics' anchors), which would
make the architecture hash depend on the input and every cache model's hashes incomparable with
rows traced before. A verdict TorchLens reports as
``unverified`` is typed ``torchlens_unverified``, apart from a replay failure
(``torchlens_gap:replay``), from the trace's own replay status, never its message.

A forward whose output holds NaN or inf is typed ``degenerate_forward:nonfinite`` before any
capture (JMT via Lead, 2026-10-28): it can never be validated, so the forward and trace modes stop
there. In forward mode a non-finite train-mode output fails the row's ``train`` mode only.

A recipe's vendor tree (``MENAGERIE_VENDOR``) goes on ``sys.path`` when ``main`` starts, not on
``PYTHONPATH``: a tree with a top-level folder named like a stdlib module the interpreter loads
while starting (``importlib``) would otherwise replace it before any code runs.
"""

from __future__ import annotations

import gc
import os
import re
import resource
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from menagerie.pipeline import now, read_json, tree_sha256, write_json
from menagerie.pipeline.records import (
    LedgerRow, Modes, Offense, OutputFile, Recipe, RunRequest, RunResult,
)

SEED = 0
TRACE_FILE = "trace.tlspec"
# A portable save without activations redraws the live diagram exactly (contracts probe); the
# runnable save alone draws a poorer one. Render mode reads this file when it is staged. It is
# written only beside a runnable ``trace.tlspec``: a refused one is already the portable save.
RENDER_FILE = "render.tlspec"
SUMMARY_FILE = "summary.json"
TAIL_LINES = 40
_MEMORY = re.compile(r"(?i)out of memory|can't allocate memory|DefaultCPUAllocator|std::bad_alloc")
# A compiled extension built in the sandbox (torch.utils.cpp_extension, setuptools, ninja) failed.
# A forward whose output holds NaN or inf: never validated, so typed before any capture.
NONFINITE = "degenerate_forward:nonfinite"
_COMPILE = re.compile(r"(?i)error building extension|ninja: build stopped|CompileError|"
                      r"command '[^']*(gcc|g\+\+|cc|c\+\+|nvcc)' failed|"
                      r"CUDA_HOME environment variable is not set|Ninja is required")


class Failure(Exception):
    """A typed model failure: ``cls`` is a class from the closed taxonomy."""

    def __init__(self, cls: str, msg: str, tail: tuple[str, ...] = (),
                 offenses: tuple[Offense, ...] = ()) -> None:
        super().__init__(msg)
        self.cls, self.msg, self.tail, self.offenses = cls, msg, tail, offenses


@dataclass(frozen=True)
class Subject:
    """A freshly built model with its inputs, ready for one forward or capture."""

    model: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def _rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


_HOME = re.compile(r"/(?:home|Users)/[^/\s\"']+/")


def _first_line(exc: BaseException) -> str:
    """The exception's first line, with home directories written ``~/`` (rows are committed)."""

    text = str(exc).strip().splitlines()
    line = f"{type(exc).__name__}: {text[0][:300] if text else ''}"
    fields = getattr(exc, "fields", None)  # a TorchLens error's structured payload
    if isinstance(fields, dict) and fields.get("budget_bytes"):  # SaveBudgetExceededError
        line += (f" ({fields.get('accounted_bytes')} bytes accounted, budget "
                 f"{fields['budget_bytes']} on {fields.get('device')}, {fields.get('num_saved')} "
                 f"saves, at {fields.get('label')})")
    return _HOME.sub("~/", line)


MAX_OFFENSES = 8


def _offenses(exc: BaseException) -> tuple[Offense, ...]:
    """The value escapes a TorchLens refusal names in ``fields["offenses"]`` (first eight), home
    directories written ``~/``; empty for any other exception or a malformed entry."""

    fields = getattr(exc, "fields", None)
    raw = fields.get("offenses") if isinstance(fields, Mapping) else None
    if not isinstance(raw, (list, tuple)):
        return ()
    out = []
    for item in raw[:MAX_OFFENSES]:
        if not isinstance(item, Mapping) or not isinstance(item.get("file"), str):
            continue
        line = item.get("line")
        text = {k: item[k] for k in ("consumer_kind", "tensor_label", "escape_method", "substrate")
                if isinstance(item.get(k), str)}
        out.append(Offense(file=_HOME.sub("~/", item["file"]),
                           line=line if isinstance(line, int) and not isinstance(line, bool)
                           else None, **text))
    return tuple(out)


@contextmanager
def _stage(cls: str) -> Iterator[None]:
    """Turn any exception inside a stage into a typed ``Failure`` (memory first)."""

    try:
        yield
    except Failure:
        raise
    except BaseException as exc:  # noqa: BLE001 -- model code raises anything, all are typed here
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        tail = tuple(traceback.format_exc().splitlines()[-TAIL_LINES:])
        if isinstance(exc, MemoryError) or _MEMORY.search(str(exc)):
            cls = "memory_cap"
        elif cls == "construct_error" and (_COMPILE.search(f"{type(exc).__name__}: {exc}")
                                           or any("cpp_extension" in line for line in tail)):
            cls = "compiled_ext"  # torch's extension builder, or what it needs, failed
        elif isinstance(exc, ImportError) and cls == "construct_error":
            cls = "missing_dependency"
        raise Failure(cls, _first_line(exc), tail, _offenses(exc)) from exc


def _wrap_torch() -> None:
    """Wrap torch before the model is built, at every TorchLens pin.

    A model that binds a torch function at import or construction (``from torch import bmm``,
    transformers' ``GELUActivation`` storing ``F.gelu``) holds the unwrapped original when it is
    built before TorchLens first wraps torch. TorchLens 2.36.0 rescues such stale references for
    ``tl.trace`` but not inside the validator, whose completeness check then fails (m19908
    ``bmm``, m20502 ``cat``; TorchLens diagnosis 2026-10-26, G1). Wrapping first gives every
    build the wrapped functions.
    """

    from torchlens.backends.torch.wrappers import wrap_torch

    wrap_torch()


def _build(recipe: Recipe, *, train: bool = False, device: str = "cpu") -> Subject:
    import torch

    from menagerie.pipeline import harvest, inputs

    if recipe.backend == "torch":
        _wrap_torch()
    with _stage("construct_error"):
        torch.manual_seed(SEED)
        model = harvest.construct(recipe, device=device)
        if hasattr(model, "train"):
            model.train(train)
        args, kwargs = inputs.make_input(recipe.input, seed=SEED)
    if device != "cpu":
        args, kwargs = _map_tensors((args, kwargs), lambda t: t.to(device))
    return Subject(model, tuple(args), dict(kwargs))


def _map_tensors(value: Any, fn: Callable[[Any], Any]) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, Mapping):
        return type(value)({k: _map_tensors(v, fn) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        mapped = [_map_tensors(v, fn) for v in value]
        return type(value)(*mapped) if hasattr(value, "_fields") else type(value)(mapped)
    return value


def tensors(value: Any) -> list[Any]:
    """Return the tensors in a nested output, in order (mappings by insertion order)."""

    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [t for v in value.values() for t in tensors(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensors(v)]
    return []


def nonfinite(value: Any) -> str | None:
    """Where a forward's output is not finite: the first floating or complex output tensor that
    holds NaN or inf, with its counts, or ``None`` when every one is finite. Integer and boolean
    outputs are finite by type; sparse and meta tensors hold no values to read here."""

    import torch

    for i, t in enumerate(tensors(value)):
        if not (t.is_floating_point() or t.is_complex()) or t.is_sparse or t.is_meta:
            continue
        data = t.detach()
        if bool(torch.isfinite(data).all()):
            continue
        nan, inf = int(torch.isnan(data).sum()), int(torch.isinf(data).sum())
        return (f"output {i} of shape {tuple(data.shape)} holds {nan} NaN and {inf} inf of "
                f"{data.numel()} values")
    return None


def _finite(out: Any, mode: str) -> None:
    """Raise ``NONFINITE`` when a plain forward's output holds NaN or inf."""

    why = nonfinite(out)
    if why is not None:
        raise Failure(NONFINITE, f"{mode}-mode forward is not finite: {why}")


def _shapes(value: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(d) for d in t.shape) for t in tensors(value))


def _forward(subject: Subject) -> Any:
    import torch

    torch.manual_seed(SEED)
    return subject.model(*subject.args, **subject.kwargs)


def _warm(subject: Subject, threads: int | None = None) -> str | None:
    """One plain forward before a capture: eval mode, no grad, the capture's thread count.

    A model that fills a cache on its first eval call (timm's attention biases in LeViT, TinyViT,
    EfficientFormer) then shows every capture its steady graph, and TorchLens's retrace check
    compares like with like. The thread count is restored, and a train-mode model is switched to
    eval and back; an eval model is never switched, because timm's ``train()`` (``eval()`` too)
    clears the cache the warm-up filled. No grad matters:
    a cache filled with gradients on holds a graph, and TorchLens then refuses the deep copy it
    validates (``precondition_refusal``, probed on those timm families at 2.35.3). Only a forward
    that raises without autograd (a physics-informed net's ``torch.autograd.grad``) is warmed
    with gradients on.
    Memory: the model is the capture's own build and the forward keeps no graph, so the peak is
    at most the measured forward's, which the trace's projection already covers. Returns why the
    warm-up raised, or ``None``; that is recorded on the row, never the outcome, except that
    running out of memory raises (it is the capture's ``memory_cap`` too).
    """

    import torch

    model, prior = subject.model, torch.get_num_threads()
    training = bool(getattr(model, "training", False))
    if threads:
        torch.set_num_threads(threads)
    try:
        if training:
            model.eval()
        why = None
        for grad in (False, True):
            try:
                with torch.set_grad_enabled(grad):
                    _forward(subject)
                return None
            except Exception as exc:  # noqa: BLE001 -- model code raises anything; recorded
                if isinstance(exc, MemoryError) or _MEMORY.search(str(exc)):
                    raise
                why = _first_line(exc)
        return why
    finally:
        if training:
            model.train(True)
        torch.set_num_threads(prior)


def _perturbed(value: Any) -> Any:
    """Return the inputs with every tensor replaced by different values of its shape and dtype."""

    import torch

    def change(t: Any) -> Any:
        if t.dtype == torch.bool:
            return ~t
        if t.is_floating_point() or t.is_complex():
            return torch.randn_like(t) * 3 + 1
        high = int(t.max()) + 1 if t.numel() else 2
        return (t + 1) % max(high, 2)

    torch.manual_seed(SEED + 1)
    return _map_tensors(value, change)


def _digest_of(value: Any) -> tuple[tuple[tuple[int, ...], float, float], ...]:
    """Shape, sum and absolute sum of every tensor in a value: cheap to keep for every module."""

    return tuple((tuple(t.shape), float(t.detach().double().sum()),
                  float(t.detach().double().abs().sum())) for t in tensors(value))


def _module_outputs(subject: Subject, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    """One no-grad forward; a digest of every module call's output, in call order."""

    import torch

    seen: list[Any] = []

    def hook(_module: Any, _inputs: Any, out: Any) -> None:
        seen.append(_digest_of(out))

    modules = list(subject.model.modules()) if hasattr(subject.model, "modules") else []
    handles = [m.register_forward_hook(hook) for m in modules]
    try:
        with torch.no_grad():
            seen.append(_digest_of(_forward(Subject(subject.model, args, kwargs))))
    finally:
        for handle in handles:
            handle.remove()
    return seen


def input_boundary(subject: Subject, reference: Any) -> str:
    """Classify the model's use of its input: ``real``, ``wrapper`` (ignored) or ``none``.

    Every module's output is compared between the real inputs and changed ones, not only the
    model's output: a zero-initialised head (torchvision's ViT) makes the output constant at
    random initialisation although the body reads the input. ``reference`` is unused (kept for
    callers that already hold the output).
    """

    del reference
    if not tensors((subject.args, subject.kwargs)):
        return "none"
    args, kwargs = _perturbed((subject.args, subject.kwargs))
    try:
        before = _module_outputs(subject, subject.args, subject.kwargs)
        after = _module_outputs(subject, args, kwargs)
    except Exception:  # noqa: BLE001 -- a model that breaks on other values reads them
        return "real"
    return "wrapper" if before == after else "real"


# Shape and layout queries are not graph ops; whether they reach the recorder depends on how torch
# functions are wrapped (TorchLens decorates them once imported), so they are left out.
METADATA_CALLS = frozenset({
    "size", "dim", "ndimension", "numel", "nelement", "stride", "is_contiguous", "__len__",
    "__get__", "element_size", "data_ptr", "storage_offset", "is_floating_point", "is_complex",
})


class _OpRecorder:
    """Record the torch function names one forward calls (cheap; no capture).

    In-place updates of the model's own buffers are bookkeeping, not graph: BatchNorm's
    ``num_batches_tracked.add_(1)`` runs only in train mode, and counting it would call every
    batch-norm model structurally divergent.
    """

    def __init__(self, buffers: frozenset[int] = frozenset()) -> None:
        from torch.overrides import TorchFunctionMode

        names: list[str] = []

        class Mode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
                name = getattr(func, "__name__", repr(func))
                bookkeeping = name.endswith("_") and args and id(args[0]) in buffers
                if not bookkeeping and name not in METADATA_CALLS:
                    names.append(name)
                return func(*args, **(kwargs or {}))

        self.names, self.mode = names, Mode()


def _timed_forward(subject: Subject, *, record: bool = False) -> tuple[Any, float, list[str]]:
    from contextlib import nullcontext

    recorder = None
    if record:
        model = subject.model
        buffers = frozenset(map(id, model.buffers())) if hasattr(model, "buffers") else frozenset()
        recorder = _OpRecorder(buffers)
    with _stage("forward_error"), (recorder.mode if recorder else nullcontext()):
        started = time.perf_counter()
        out = _forward(subject)
        secs = time.perf_counter() - started
    return out, secs, recorder.names if recorder else []


def divergence(eval_out: Any, train_out: Any, eval_ops: list[str], train_ops: list[str]) -> str:
    """``structural`` if the graph or output shapes differ, ``statistical`` if only values do."""

    import torch

    if eval_ops != train_ops or _shapes(eval_out) != _shapes(train_out):
        return "structural"
    pairs = zip(tensors(eval_out), tensors(train_out))
    same = all(torch.equal(a.detach(), b.detach()) for a, b in pairs)
    return "none" if same else "statistical"


# --- modes ---------------------------------------------------------------------------


def _base_row(request: RunRequest, **fields: Any) -> LedgerRow:
    import torch
    import torchlens as tl

    from menagerie.pipeline import envs, worker_preview

    recipe = request.recipe
    fw = f"torch {torch.__version__}" if recipe.backend == "torch" else (
        worker_preview.framework(recipe.backend))
    try:
        env = envs.lock_id(recipe.env)
    except FileNotFoundError:  # an environment without a committed lock (a probe environment)
        env = recipe.env
    return LedgerRow(
        id=request.id, at=now(), host=os.environ.get("MENAGERIE_HOST", "worker"), env=env,
        recipe_rev=recipe.rev, backend=recipe.backend, tl=tl.__version__, fw=fw, **fields,
    )


def _forward_mode(request: RunRequest, attempt_dir: Path) -> RunResult:
    """Eval then train on fresh models; the eval outcome is the result, train goes in ``modes``."""

    if request.recipe.backend != "torch":
        from menagerie.pipeline import worker_preview

        return worker_preview.forward_mode(request, attempt_dir)
    subject = _build(request.recipe)
    eval_out, secs, eval_ops = _timed_forward(subject, record=True)
    _finite(eval_out, "eval")
    peak = _rss_mb()
    boundary = input_boundary(subject, eval_out)
    params = sum(int(p.numel()) for p in subject.model.parameters())
    del subject
    train, div = "pass", None
    try:
        train_out, _, train_ops = _timed_forward(_build(request.recipe, train=True), record=True)
        _finite(train_out, "train")
        div = divergence(eval_out, train_out, eval_ops, train_ops)
    except Failure as failure:
        train = f"fail:{failure.cls}"
    modes = Modes(eval="pass", train=train, divergence=div)
    row = _base_row(request, stage="forward", outcome="pass", mode="eval", params=params,
                    params_source="counted", forward_secs=round(secs, 3),
                    forward_peak_rss_mb=peak, input_boundary=boundary)
    return _result(request, "pass", rows=(row,), output_shapes=_shapes(eval_out),
                   params=params, modes=modes, input_boundary=boundary)


@dataclass(frozen=True)
class Validation:
    """The June validator's verdict and what was read off its trace before TorchLens freed it."""

    passed: bool
    msg: str | None
    secs: float
    n_ops: int | None = None
    graph_hash: str | None = None
    # TorchLens's replay status for the validation trace (``passed``, ``failed``, ``unverified``),
    # its reason code (``trace_retrace_structure_mismatch`` when the retrace check stopped it) and
    # failed node count, and the check of the first recorded failure (``retrace_mismatch``).
    state: str | None = None
    reason: str | None = None
    failed: int | None = None
    check: str | None = None
    warm: str | None = None  # why the warm-up raised, or None


def _replay_status(trace: Any) -> dict[str, Any]:
    """The state, reason and failed node count of a validation trace's public
    ``validation_replay_status``."""

    status = getattr(trace, "validation_replay_status", None)
    return {"state": getattr(status, "state", None), "reason": getattr(status, "reason", None),
            "failed": getattr(status, "failed_node_count", None)}


def _validate(subject: Subject, threads: int | None) -> Validation:
    """Validate through the June validator's private entry: its own ground truth, one capture of
    every activation, the retrace check and per-op replay (``docs/torchlens-boundary.md``)."""

    import torchlens as tl
    from torchlens.user_funcs import _validate_forward_pass_torch

    seen: dict[str, Any] = {}

    def observe(trace: Any) -> None:
        if trace is not None:  # None: a precondition refusal before any capture (2.35.2)
            seen.update(n_ops=int(trace.num_ops), graph_hash=trace.graph_shape_hash or None,
                        **_replay_status(trace))

    with _stage("capture_error"):
        seen["warm"] = _warm(subject, threads)
        started = time.perf_counter()
        passed = bool(_validate_forward_pass_torch(
            subject.model, list(subject.args), subject.kwargs, random_seed=SEED,
            validate_metadata=True, num_threads=threads, _trace_observer=observe))
        secs = time.perf_counter() - started
    failure = None if passed else tl.validation.last_validation_failure()
    msg = None if passed else "validation returned False with no failure record"
    if failure is not None:
        msg = failure.summary() if hasattr(failure, "summary") else str(failure)
        seen["check"] = getattr(failure, "check", None)
    return Validation(passed, msg and _HOME.sub("~/", msg[:300]), secs, **seen)


@dataclass(frozen=True)
class Saved:
    """What ``_save_runnable`` wrote: the ledger's runnable state (``ready`` or
    ``refused:<code>``) and the level ``trace.tlspec`` holds (``runnable``, ``portable`` after a
    refusal, ``None`` when no level saved). The level is what was written, so a round-trip
    refusal of a runnable save is still a runnable save."""

    state: str
    level: str | None


def _save_runnable(trace: Any, path: Path, skeleton: str) -> Saved:
    """Save runnable when the preflight passes, else the portable level; check the round trip.

    A save problem is a state of the trace, never a verdict: the state is ``ready`` or
    ``refused:<code>``, and the file may be missing when even the portable save fails.
    """

    import shutil

    import torchlens as tl

    shutil.rmtree(path, ignore_errors=True)  # a retry saves over the first attempt's file
    try:
        tl.save(trace, path, level="runnable")
        runnable, level = "ready", "runnable"
    except Exception as exc:  # noqa: BLE001 -- the refusal code is recorded, not raised
        code = re.search(r"\[([a-z0-9_]+)\]", str(exc))
        preflight = type(exc).__name__ == "RunnablePreflightError"
        runnable = f"refused:{code.group(1) if preflight and code else 'save_error'}"
        if not _portable(trace, path):
            return Saved(runnable, None)
        level = "portable"
    try:
        same = tl.hash.trace(tl.load(path)) == skeleton
    except Exception:  # noqa: BLE001 -- a save that does not load is not runnable
        return Saved("refused:round_trip_load", level)
    return Saved(runnable if same else "refused:round_trip_hash", level)


def _portable(trace: Any, path: Path) -> bool:
    import shutil

    import torchlens as tl

    shutil.rmtree(path, ignore_errors=True)
    try:
        tl.save(trace, path, level="portable", include_outs=False)
        return True
    except Exception:  # noqa: BLE001 -- e.g. TorchLensIOError on beit's module state
        shutil.rmtree(path, ignore_errors=True)
        for partial in path.parent.glob(f"{path.name}.tmp.*"):  # a failed save leaves its staging
            shutil.rmtree(partial, ignore_errors=True)
        return False


def _files(attempt_dir: Path, *names: tuple[str, str]) -> tuple[OutputFile, ...]:
    out = []
    for kind, name in names:
        path = attempt_dir / name
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*")) if path.is_dir() else (
                path.stat().st_size)
            out.append(OutputFile(kind=kind, path=name, sha256=tree_sha256(path), bytes=size))
    return tuple(out)


def _write_summary(row: dict[str, Any], attempt_dir: Path) -> None:
    import json

    text = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
    (attempt_dir / SUMMARY_FILE).write_text(text + "\n", encoding="utf-8")


SAVE_BUDGET_FRACTION = 0.5  # TorchLens's documented "auto" fraction (LIMITATIONS.md, 2.35.3)


def gpu_save_options(device: str, rss_mb: int) -> dict[str, Any]:
    """Capture options for a capture on ``device``: none on the CPU; on a GPU, saves go to host
    memory (``output_device="cpu"``) under an absolute ``save_budget`` of TorchLens's own default
    fraction of the attempt's memory cap (``rss_mb``, sized from the job's declaration), not of
    the node's free memory, which TorchLens's ``"auto"`` reads on a CPU device.

    TorchLens 2.35.3's validator entry takes no capture options, so its capture keeps
    ``save_budget="auto"`` on the card (``docs/torchlens-boundary.md``).
    """

    if device == "cpu":
        return {}
    return {"output_device": "cpu", "save_budget": int(rss_mb * 2**20 * SAVE_BUDGET_FRACTION)}


def _hash_capture(subject: Subject, *, structure_only: bool = False, device: str = "cpu",
                  rss_mb: int = 0) -> Any:
    """The capture the hash family is read from.

    June's digest capture (inference-only, seed 0, host RNG restored), or with ``structure_only``
    the meta-device capture. Trace mode, structure-only mode and the hash harness all hash here,
    so the harness checks the hashes the ledger records. Never warmed (module docstring). On a
    GPU its saves go to host memory (``gpu_save_options``).
    """

    import torchlens as tl
    from torchlens.options import CaptureOptions

    from menagerie.structural_digest import _restored_host_rng_state

    options = (CaptureOptions(structure_only=True) if structure_only
               else CaptureOptions(inference_only=True, random_seed=SEED,
                                   **gpu_save_options(device, rss_mb)))
    with _stage("capture_error"), _restored_host_rng_state():
        return tl.trace(subject.model, list(subject.args), input_kwargs=subject.kwargs,
                        capture=options)


def _digest(request: RunRequest, device: str = "cpu") -> tuple[dict[str, Any], dict[str, Any]]:
    """June's digest capture of a fresh build: the 1.1.0 summary columns exactly as June
    computed them, and the hash family."""

    from menagerie import trace_summary

    trace = _hash_capture(_build(request.recipe, train=request.model_mode == "train",
                                 device=device), device=device, rss_mb=request.caps.rss_mb)
    hashes = trace_summary.hash_family(trace)
    return trace_summary.summarize_trace(request.id, trace, request.recipe.rev,
                                         hashes=hashes), hashes


def _save(request: RunRequest, attempt_dir: Path,
          device: str = "cpu") -> tuple[str, float, Any | None]:
    """Capture a fresh build with ``intervention_ready`` as the request says and save it.

    Its own capture because TorchLens refuses ``intervention_ready`` with June's
    ``inference_only``, and the June validator's private entry takes no capture options.
    Returns the runnable state, the capture's seconds and, when neither save was written, the
    live trace: render mode would have nothing to draw (fasterrcnn m4851 at 2.35.3).

    The portable save for drawing (``render.tlspec``) is written only when ``trace.tlspec`` is not
    already one: after a runnable refusal ``trace.tlspec`` is that same portable save, and render
    reads it (``flow.render_request`` stages it; ``_render_mode`` falls back to it).
    """

    import shutil

    import torchlens as tl
    from torchlens.options import CaptureOptions

    subject = _build(request.recipe, train=request.model_mode == "train", device=device)
    with _stage("capture_error"):
        _warm(subject)  # a raise is recorded from the validation's warm-up of the same build
        started = time.perf_counter()
        trace = tl.trace(subject.model, list(subject.args), input_kwargs=subject.kwargs,
                         capture=CaptureOptions(intervention_ready=request.intervention_ready,
                                                random_seed=SEED,
                                                **gpu_save_options(device, request.caps.rss_mb)))
        secs = time.perf_counter() - started
    saved = _save_runnable(trace, attempt_dir / TRACE_FILE, tl.hash.trace(trace))
    shutil.rmtree(attempt_dir / RENDER_FILE, ignore_errors=True)  # a retry's twin goes too
    twin = saved.level != "portable" and _portable(trace, attempt_dir / RENDER_FILE)
    return saved.state, secs, None if (saved.level or twin) else trace


def _trace_mode(request: RunRequest, attempt_dir: Path, device: str = "cpu") -> RunResult:
    """Timed plain forward; the June validator (retried once on one thread) for the verdict,
    op count and graph hash; June's digest capture for the summary and hashes; the save. A
    preview backend's recipe goes to ``worker_preview``."""

    from menagerie import trace_summary

    if request.recipe.backend != "torch":
        from menagerie.pipeline import worker_preview

        return worker_preview.trace_mode(request, attempt_dir)
    train = request.model_mode == "train"
    subject = _build(request.recipe, train=train, device=device)
    truth, forward_secs, _ = _timed_forward(subject)
    _finite(truth, request.model_mode)
    forward_peak = _rss_mb()
    boundary = input_boundary(subject, truth)
    del subject
    gc.collect()
    # The verdict is taken on a fresh build warmed by exactly one forward (``_warm``), never on
    # the timed model: its forward ran with gradients and in the request's mode, and restoring
    # the state dict would not undo what it cached (ConViT's relative position indices).
    check = _validate(_build(request.recipe, train=train, device=device), None)
    msg = check.msg
    gc.collect()
    if not check.passed:
        check = _validate(_build(request.recipe, train=train, device=device), 1)
        msg = f"single-thread retry: {check.msg}" if not check.passed else (
            f"passed on the single-thread retry after: {msg}")
    if check.n_ops is None:
        raise Failure("capture_error", f"validation refused before capture: {check.msg}")
    cls, state = verdict_class(check)
    if state == "unverified" and check.reason:
        msg = f"unverified ({check.reason}): {msg}"
    if check.warm:
        msg = "; ".join(filter(None, (msg, f"warm-up raised: {check.warm}")))
    summary, hashes = _digest(request, device)
    runnable, save_secs, unsaved = _save(request, attempt_dir, device)
    timings = {"forward_secs": round(forward_secs, 3), "capture_secs": round(check.secs, 3),
               "forward_peak_rss_mb": forward_peak}
    summary = {**summary, **timings, "save_capture_secs": round(save_secs, 3)}
    if device != "cpu":
        msg = "; ".join(filter(None, (msg, _cuda_note())))
        summary = {**summary, "device": _cuda_note()}
    _write_summary(summary, attempt_dir)
    files = _files(attempt_dir, ("tlspec", TRACE_FILE), ("tlspec-portable", RENDER_FILE),
                   ("summary", SUMMARY_FILE))
    sha = {f.kind: f.sha256 for f in files}
    renders: tuple[Any, ...] = ()
    render_files: tuple[OutputFile, ...] = ()
    if unsaved is not None and check.passed:
        # Nothing saved to draw from later, so draw now, keyed on this capture's summary.
        renders, render_files, note = _render_live(request, unsaved, sha["summary"], attempt_dir)
        msg = "; ".join(filter(None, (msg, note)))
    del unsaved
    params = int(summary.get("n_params") or 0) or None
    row = _base_row(
        request, stage="trace", outcome="pass" if check.passed else "fail",
        cls=cls, mode=request.model_mode,
        capture="full", n_ops=check.n_ops, hashes=trace_summary.ledger_hashes(hashes),
        graph_hash=check.graph_hash, params=params, params_source="counted",
        peak_rss_mb=_rss_mb(), input_boundary=boundary, validation="torch-replay",
        validation_state=state, runnable=runnable, warmed=check.warm is None,
        trace=sha.get("tlspec"), summary=sha.get("summary"),
        msg=msg, **timings,
    )
    fields = {"rows": (row, *renders), "files": files + render_files, "params": params,
              "input_boundary": boundary, "output_shapes": _shapes(truth)}
    if check.passed:
        return _result(request, "pass", **fields)
    return _result(request, "fail", cls=row.cls, msg=msg, **fields)


# The failure check TorchLens records when its pristine re-trace diverged after a replay that
# otherwise passed (``torchlens.validation.diagnostics.CHECK_RETRACE_MISMATCH`` at 2.35.3).
RETRACE_MISMATCH = "retrace_mismatch"


def verdict_class(check: Validation) -> tuple[str | None, str]:
    """The failure class and ledger validation state of a verdict: ``passed``; ``unverified``
    (``torchlens_unverified``) when TorchLens's replay status is ``unverified`` with no failed node
    and no recorded failure but the retrace check's; else a replay failure
    (``torchlens_gap:replay``). TorchLens downgrades to ``unverified`` after a retrace mismatch
    even when the replay also failed (LeViT at 2.35.3: a perturbation mismatch), which is a
    replay failure."""

    if check.passed:
        return None, "passed"
    if check.state == "unverified" and not check.failed and check.check in (None, RETRACE_MISMATCH):
        return "torchlens_unverified", "unverified"
    return "torchlens_gap:replay", "failed"


def _structure_only_mode(request: RunRequest, attempt_dir: Path) -> RunResult:
    """Capture a meta-device build: graph, hashes and geometry, never validated."""

    if request.recipe.backend != "torch":
        raise Failure("capture_error:backend",
                      f"structure-only capture builds on torch's meta device; a "
                      f"{request.recipe.backend} model has none")
    from menagerie import trace_summary

    subject = _build(request.recipe, train=request.model_mode == "train", device="meta")
    started = time.perf_counter()
    trace = _hash_capture(subject, structure_only=True)
    secs = round(time.perf_counter() - started, 3)
    hashes = trace_summary.hash_family(trace)
    _write_summary(trace_summary.summarize_trace(
        request.id, trace, request.recipe.rev, digest_capture_policy="structure_only",
        hashes=hashes, timings={"capture_secs": secs}), attempt_dir)
    saved = _portable(trace, attempt_dir / TRACE_FILE)
    files = _files(attempt_dir, ("tlspec", TRACE_FILE), ("summary", SUMMARY_FILE))
    sha = {f.kind: f.sha256 for f in files}
    renders, render_files, msg = (), (), None
    if not saved:
        # No save level accepts some structure-only traces (beit at 2.35.2), so render mode could
        # never draw them: render now, keyed on the summary of this capture.
        renders, render_files, msg = _render_live(request, trace, sha["summary"], attempt_dir)
    params = sum(int(p.numel()) for p in subject.model.parameters())
    row = _base_row(
        request, stage="trace", outcome="pass", mode=request.model_mode,
        capture="structure_only", n_ops=int(trace.num_ops),
        hashes=trace_summary.ledger_hashes(hashes),
        graph_hash=trace.graph_shape_hash or None,
        params=params, params_source="geometry", peak_rss_mb=_rss_mb(), capture_secs=secs,
        validation_state="structure_only", runnable="n/a", trace=sha.get("tlspec"),
        summary=sha.get("summary"), msg=msg,
    )
    return _result(request, "pass", rows=(row, *renders), files=files + render_files,
                   params=params)


def _render_live(request: RunRequest, trace: Any, key: str,
                 attempt_dir: Path) -> tuple[tuple[Any, ...], tuple[OutputFile, ...], str]:
    """Render a live trace in process; a render failure is a note, never the trace's outcome."""

    from menagerie.pipeline import render

    try:
        row, files = render.render_set(trace, model_id=request.id, trace_sha=key,
                                       out_dir=attempt_dir / "render")
    except Exception as exc:  # noqa: BLE001 -- recorded on the trace row, retried by renderer_rev
        return (), (), f"trace not saveable; render in process failed: {_first_line(exc)}"
    return (row,), _under(files, "render"), "trace not saveable; rendered in process"


def _under(files: Any, folder: str) -> tuple[OutputFile, ...]:
    """``render_set`` lists paths under its ``out_dir``; results list them under the attempt."""

    from dataclasses import replace

    return tuple(replace(f, path=f"{folder}/{f.path}") for f in files)


def _render_mode(request: RunRequest, attempt_dir: Path) -> RunResult:
    """Render a saved trace: ``render.tlspec`` or ``trace.tlspec`` in the attempt directory, else
    the store blob the request names (name the portable save's sha: it draws like the live
    trace, while a runnable save alone draws a poorer diagram)."""

    import torchlens as tl

    from menagerie.pipeline import StoreLayout, render, store

    staged = [p for p in (attempt_dir / RENDER_FILE, attempt_dir / TRACE_FILE) if p.exists()]
    try:
        if staged:
            trace = tl.load(staged[0])
        elif os.environ.get("MENAGERIE_STORE"):  # the raw blob, else its packed form
            trace = store.read_blob(str(request.trace), tl.load, store=StoreLayout.from_env())
        else:
            raise FileNotFoundError("MENAGERIE_STORE is not set")
    except FileNotFoundError as exc:
        if staged:
            raise
        return _result(request, "infra", cls="infra:trace_not_staged",
                       msg=f"trace {request.trace} is neither in the attempt nor in the store: "
                           f"{_first_line(exc)}")
    # A train-mode trace (structural divergence only) gets the ``-train`` variant keys.
    variants = render.variants_for(trace, train=True) if request.model_mode == "train" else None
    row, files = render.render_set(trace, model_id=request.id, trace_sha=str(request.trace),
                                   out_dir=attempt_dir / "render", variants=variants)
    if row.outcome in ("pass", "partial"):
        return _result(request, "pass", rows=(row,), files=_under(files, "render"))
    why = row.skipped[0].why if row.skipped else "render:fail"
    return _result(request, "fail" if row.outcome == "fail" else "infra", cls=why,
                   rows=(row,), files=_under(files, "render"))


def in_gpu_window(at: datetime | None = None) -> bool:
    """Apply machine_load.md rule 4 using the fleet's current jmt-tz setting."""

    import json
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    setting = Path(os.environ.get("JMT_TZ_FILE", str(Path.home() / ".config/fleet/jmt-timezone.json")))
    try:
        zone = json.loads(setting.read_text())["zone"] if setting.exists() else "America/Los_Angeles"
        local = (at or datetime.now(UTC)).astimezone(ZoneInfo(zone))
    except (OSError, ValueError, KeyError, TypeError, ZoneInfoNotFoundError):
        return False
    return local.hour >= 22 or local.hour < 8


def _cuda_note() -> str:
    import torch

    peak = torch.cuda.max_memory_allocated() // 2**20
    return f"cuda: {torch.cuda.get_device_name(0)}, peak {peak} MiB allocated"


def _gpu_mode(request: RunRequest, attempt_dir: Path) -> RunResult:
    """The trace mode on a CUDA device (timed forward, the June validator, digest, save), only
    inside the GPU window on the worker (a cluster batch has its own allocation, so no window).
    Outside it, or with no CUDA device in the attempt (the runner binds the GPU only for this
    mode), the result is ``infra``: never the model's failure."""

    import torch

    if request.recipe.backend != "torch":
        raise Failure("capture_error:backend", "gpu mode runs torch recipes only")
    if os.environ.get("MENAGERIE_HOST") != "cluster" and not in_gpu_window():
        return _result(request, "infra", cls="infra:gpu_window",
                       msg="outside the quiet GPU window (machine_load.md rule 4, JMT's current timezone)")
    if not torch.cuda.is_available():
        return _result(request, "infra", cls="infra:no_gpu",
                       msg=f"no CUDA device in the attempt (torch {torch.__version__}, "
                           f"cuda {torch.version.cuda})")
    torch.cuda.reset_peak_memory_stats()
    return _trace_mode(request, attempt_dir, device="cuda")


MODES: dict[str, Callable[[RunRequest, Path], RunResult]] = {
    "forward": _forward_mode, "trace": _trace_mode,
    "trace-structure-only": _structure_only_mode, "render": _render_mode, "gpu": _gpu_mode,
}
STARTED: dict[str, str] = {}


def _result(request: RunRequest, outcome: str, **fields: Any) -> RunResult:
    return RunResult(request=request.request, id=request.id, mode=request.mode, outcome=outcome,
                     started=STARTED.get(request.request, now()), ended=now(),
                     host=os.environ.get("MENAGERIE_HOST", "worker"), **fields)


def run_request(request: RunRequest, *, attempt_dir: Path) -> RunResult:
    """Execute one request in ``attempt_dir`` and return its result (not yet written)."""

    import torch

    STARTED[request.request] = now()
    torch.set_num_threads(max(1, request.caps.threads))
    attempt_dir.mkdir(parents=True, exist_ok=True)
    # A warning is reported once and never decides an outcome, whatever the caller's filters.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        try:
            return MODES[request.mode](request, attempt_dir)
        except Failure as failure:
            return _failed(request, failure)


def _failed(request: RunRequest, failure: Failure) -> RunResult:
    """The result and ledger row of an attempt that ended in a typed failure."""

    if request.mode == "render":
        return _result(request, "fail", cls=failure.cls, msg=failure.msg, tail=failure.tail)
    structure_only = request.mode == "trace-structure-only"
    row = _base_row(
        request, stage="forward" if request.mode == "forward" else "trace", outcome="fail",
        cls=failure.cls, mode=request.model_mode, msg=failure.msg,
        offenses=failure.offenses or None,
        capture=None if request.mode == "forward" else (
            "structure_only" if structure_only else "full"),
        validation_state="structure_only" if structure_only else None,
    )
    modes = Modes(eval=f"fail:{failure.cls}") if request.mode == "forward" else None
    return _result(request, "fail", cls=failure.cls, msg=failure.msg, tail=failure.tail,
                   rows=(row,), modes=modes)


def vendor_on_path(path: list[str], environ: MutableMapping[str, str]) -> None:
    """Put the vendor root (``MENAGERIE_VENDOR``) in ``path`` just ahead of the pipeline's code
    root, where ``PYTHONPATH`` used to put it, and on the ``PYTHONPATH`` a child interpreter gets.

    Done after the interpreter has started, the stdlib modules it already loaded stay the stdlib's
    even when the tree has a folder of the same name (a repo's old ``importlib`` copy), as when the
    authors run a script from their repository; every other name the tree holds still comes before
    the stdlib's and the installed packages'."""

    vendor = environ.get("MENAGERIE_VENDOR")
    if not vendor or vendor in path:
        return
    code = str(Path(__file__).resolve().parents[2])
    path.insert(path.index(code) if code in path else 0, vendor)
    environ["PYTHONPATH"] = os.pathsep.join(filter(None, (vendor, environ.get("PYTHONPATH"))))


def main(attempt_dir: Path) -> int:
    """Run ``attempt_dir/request.json``, write ``result.json`` last; return the exit code."""

    vendor_on_path(sys.path, os.environ)
    attempt_dir = Path(attempt_dir)
    request = read_json(RunRequest, attempt_dir / "request.json")
    try:
        result = run_request(request, attempt_dir=attempt_dir)
    except Exception as exc:  # noqa: BLE001 -- a worker defect is infrastructure, never the model's
        tail = tuple(traceback.format_exc().splitlines()[-TAIL_LINES:])
        result = _result(request, "infra", cls="infra:worker", msg=_first_line(exc), tail=tail)
    write_json(result, attempt_dir / "result.json")
    return 0 if result.outcome == "pass" else 1
