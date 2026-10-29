"""The worker's preview-backend branch: TensorFlow, JAX, Paddle and MLX models through TorchLens.

Owned by the backends lane. ``worker`` hands a forward or trace request here when its recipe's
``backend`` is not ``torch``; the request runs in the recipe's own environment (``envs/<env>``),
which carries torch 2.6 CPU beside the framework, so the worker's records still import torch.

The model is built by ``harvest.construct`` under the framework's own seed, and its inputs are the
preset's tensors made native (``inputs.native``). Capture goes through TorchLens's public backend
surface only, with the backend pinned from the recipe, never guessed: ``tl.trace(..., backend=)``,
then the live capture's own ``validate_forward_pass``. That is what the forward scope of
``tl.validate(..., backend=)`` runs for a preview (capture, then replay the captured ops), but at
2.35.3 ``tl.validate`` hands ``validate_metadata`` to the TensorFlow capture, which refuses it, so
TensorFlow could never validate through it; and the capture validated is then the one summarized.
No preview accepts ``random_seed`` (none replays torch's RNG): models are seeded by their own
framework before they are built. Preview validation ends ``passed``,
``failed``, ``unverified`` or ``unavailable``; only ``passed`` passes, and the state is recorded as
TorchLens gives it, never coerced to a bool (``unverified`` is partial validation). The trace's
module mode (``object_module``, ``pytree_module`` or ``function_root``) goes on the ledger row so
publishing can label it. Structure-only capture and the GPU mode are torch-only.

Build contract (2b prompt, native section): ``build(cfg)`` returns the framework's model object
(a Keras model or ``tf.Module``, a Paddle ``Layer``, an MLX module, an Equinox or Flax NNX module,
or a plain function), or for JAX a pair ``(module or function, parameters)``, called as
``fn(params, *inputs)`` (a Flax Linen module through its ``apply``), so the parameters are the
first pytree argument TorchLens attributes. Train mode: Keras models get ``training=True``;
Paddle, MLX and Flax NNX modules switch with ``train()``; Equinox through
``eqx.nn.inference_mode``. A model with no switch (a ``tf.Module``, a Linen pair, a function)
has no train result.
"""

from __future__ import annotations

import gc
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from menagerie.pipeline import worker
from menagerie.pipeline.records import Modes, RunRequest, RunResult

BACKENDS = ("tf", "jax", "paddle", "mlx")
# The distribution whose version names the framework on a ledger row (``fw``).
FRAMEWORKS = {"tf": "tensorflow", "jax": "jax", "paddle": "paddlepaddle", "mlx": "mlx"}
VALIDATION = "preview-replay"
# Whether the metadata invariants run with the replay. MLX's invariant pass fails every
# function_root capture at 2.35.3 ("No output layers found"), so it is off there.
VALIDATE_METADATA = {"tf": True, "jax": True, "paddle": True, "mlx": False}
# Previews that refuse an ``intervention_ready`` capture at 2.35.3 (it needs torch autograd).
NO_INTERVENTION_READY = frozenset({"mlx"})
# Previews whose validated trace no save level accepts at 2.35.3: a fresh capture is saved.
SAVE_FRESH = frozenset({"mlx"})
# A validation state other than ``passed`` as a failure class. ``failed`` is a replay mismatch, as
# for torch; ``unverified`` and ``unavailable`` mean TorchLens could not check the capture, which
# the 2b prompt turns into a port (``not_traceable_framework:unverified``).
STATE_CLASS = {
    "failed": "torchlens_gap:replay",
    "unverified": "not_traceable_framework:unverified",
    "unavailable": "not_traceable_framework:unavailable",
}


@dataclass(frozen=True)
class Native:
    """A freshly built native model: what to call, its parameters (a JAX pair), its inputs."""

    model: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    params: Any = None
    root: Any = None  # the built object, for counting parameters
    switched: bool = False  # the model has a train/eval switch and it was set

    @property
    def inputs(self) -> list[Any]:
        """The positional arguments TorchLens sees: the parameters first, for a JAX pair."""

        return [*([self.params] if self.params is not None else []), *self.args]

    def __call__(self) -> Any:
        return self.model(*self.inputs, **self.kwargs)


def framework(backend: str) -> str:
    """``<distribution> <version>`` of the backend's framework, for the ledger's ``fw``."""

    from importlib.metadata import PackageNotFoundError, version

    name = FRAMEWORKS.get(backend, backend)
    try:
        return f"{name} {version(name)}"
    except PackageNotFoundError:
        return name


def prepare(backend: str, threads: int) -> None:
    """Hold the framework to the request's threads; JAX reads its flags before it is imported."""

    threads = max(1, threads)
    if backend == "jax":
        os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                              f"intra_op_parallelism_threads={threads}")
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif backend == "tf":
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:  # already initialized in this process: the first setting holds
            pass
    elif backend == "paddle":
        import paddle

        paddle.set_device("cpu")


def _seed(backend: str) -> None:
    import random

    import numpy as np

    random.seed(worker.SEED)
    np.random.seed(worker.SEED)
    if backend == "tf":
        import keras

        keras.utils.set_random_seed(worker.SEED)
    elif backend == "paddle":
        import paddle

        paddle.seed(worker.SEED)
    elif backend == "mlx":
        import mlx.core as mx

        mx.random.seed(worker.SEED)


def _unpack(backend: str, built: Any) -> tuple[Any, Any]:
    """The callable and the parameters of what ``build`` returned (a pair only for JAX)."""

    if backend == "jax" and isinstance(built, tuple) and len(built) == 2:
        fn, params = built
        return (fn.apply if hasattr(fn, "apply") and hasattr(fn, "init") else fn), params
    return built, None


def _set_mode(backend: str, model: Any, train: bool) -> tuple[Any, dict[str, Any], bool]:
    """The model in train or eval mode, extra call keywords, and whether a switch existed."""

    if backend == "tf":
        import keras

        if isinstance(model, keras.Model | keras.layers.Layer):
            return model, ({"training": True} if train else {}), True
        return model, {}, False
    if backend == "jax":
        return _set_jax_mode(model, train)
    if backend in ("paddle", "mlx") and hasattr(model, "train") and hasattr(model, "eval"):
        model.train() if train else model.eval()
        return model, {}, True
    return model, {}, False


def _set_jax_mode(model: Any, train: bool) -> tuple[Any, dict[str, Any], bool]:
    try:
        import equinox as eqx

        if isinstance(model, eqx.Module):
            return eqx.nn.inference_mode(model, value=not train), {}, True
    except ImportError:
        pass
    try:
        from flax import nnx

        if isinstance(model, nnx.Module):
            model.train() if train else model.eval()
            return model, {}, True
    except ImportError:
        pass
    return model, {}, False


def build(request: RunRequest, *, train: bool = False) -> Native:
    """Build the recipe's model natively with its inputs, in train or eval mode."""

    from menagerie.pipeline import harvest, inputs

    recipe = request.recipe
    with worker._stage("construct_error"):
        _seed(recipe.backend)
        built = harvest.construct(recipe)
        model, params = _unpack(recipe.backend, built)
        model, extra, switched = _set_mode(recipe.backend, model, train)
        args, kwargs = inputs.make_input(recipe.input, seed=worker.SEED)
        args, kwargs = inputs.native((tuple(args), dict(kwargs)), recipe.backend)
    native = Native(model, tuple(args), {**kwargs, **extra}, params, built, switched)
    if recipe.backend == "mlx":
        _materialize(native)
    return native


def _materialize(native: Native) -> None:
    """One forward so MLX evaluates a fresh module's lazy parameters: at mlx 0.26.3 on Linux a
    capture (or ``mx.eval`` of the parameters) of a never-run module raises "Attempting to eval
    an array without a primitive", while after one forward it captures and validates."""

    import mlx.core as mx

    with worker._stage("forward_error"):
        mx.eval(arrays(native()))


# --- arrays ---------------------------------------------------------------------------


def _is_array(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and not isinstance(value, str | bytes)


def arrays(value: Any) -> list[Any]:
    """The arrays in a nested output, in order (mappings by insertion order)."""

    if _is_array(value):
        return [value]
    if isinstance(value, dict):
        return [a for v in value.values() for a in arrays(v)]
    if isinstance(value, list | tuple):
        return [a for v in value for a in arrays(v)]
    return []


def to_numpy(value: Any) -> Any:
    import numpy as np

    return value.numpy() if callable(getattr(value, "numpy", None)) else np.asarray(value)


def shapes(value: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(d) for d in a.shape) for a in arrays(value))


def _same(left: Any, right: Any) -> bool:
    """Whether two outputs hold exactly the same arrays. Whole arrays, not sums: every softmax
    row sums to one, so a sum digest reads a classifier's output as unchanged."""

    import numpy as np

    a, b = arrays(left), arrays(right)
    return len(a) == len(b) and all(np.array_equal(to_numpy(x), to_numpy(y))
                                    for x, y in zip(a, b, strict=True))


def _changed(x: Any, rng: Any) -> Any:
    """Different values of the same shape and dtype (NumPy in, NumPy out)."""

    import numpy as np

    if x.dtype == np.bool_:
        return ~x
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.complexfloating):
        return (rng.standard_normal(x.shape) * 3 + 1).astype(x.dtype)
    high = int(x.max()) + 1 if x.size else 2
    return ((x + 1) % max(high, 2)).astype(x.dtype)


def _perturbed(native: Native, backend: str) -> Native:
    """The same model with every input array replaced by different values of its shape and dtype
    (the parameters of a JAX pair are left alone)."""

    import numpy as np

    from menagerie.pipeline import inputs

    rng, make = np.random.default_rng(worker.SEED + 1), inputs.native_array(backend)

    def change(a: Any) -> Any:
        return make(_changed(to_numpy(a), rng))

    return Native(native.model, inputs.map_leaves(native.args, _is_array, change),
                  inputs.map_leaves(native.kwargs, _is_array, change), native.params,
                  native.root, native.switched)


def input_boundary(native: Native, backend: str, reference: Any) -> str:
    """``real`` if changing the inputs changes the output, ``wrapper`` if not, ``none`` without
    input arrays. When the output does not change, every layer's output is compared where the
    framework exposes them (Keras, Paddle), as the torch path does: a model whose output is
    constant at random initialization (a collapsed head) still reads its input in its body."""

    if not arrays((native.args, native.kwargs)):
        return "none"
    changed = _perturbed(native, backend)
    try:
        if not _same(changed(), reference):
            return "real"
        inner, inner_changed = _layer_outputs(native, backend), _layer_outputs(changed, backend)
    except Exception:  # noqa: BLE001 -- a model that breaks on other values reads them
        return "real"
    if inner is None or inner_changed is None:
        return "wrapper"
    return "wrapper" if _same(inner, inner_changed) else "real"


def _layer_outputs(native: Native, backend: str) -> list[Any] | None:
    """Every layer's output in one forward (Keras functional models, Paddle layers), else None."""

    if backend == "tf":
        import keras

        model = native.model
        if not isinstance(model, keras.Model) or not getattr(model, "inputs", None):
            return None
        layers = [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]
        return list(keras.Model(model.inputs, [layer.output for layer in layers])(*native.args))
    if backend == "paddle" and hasattr(native.model, "sublayers"):
        seen: list[Any] = []
        hooks = [layer.register_forward_post_hook(lambda _l, _i, out: seen.append(out))
                 for layer in native.model.sublayers()]
        try:
            native()
        finally:
            for hook in hooks:
                hook.remove()
        return seen
    return None


def count_params(native: Native, backend: str) -> int | None:
    """The model's parameter count from the framework's own parameter list."""

    target = native.params if native.params is not None else native.root
    try:
        if backend == "tf":
            return sum(int(to_numpy(v).size) for v in target.variables)
        if backend == "paddle":
            return sum(int(p.size) for p in target.parameters())
        if backend == "mlx":
            from mlx.utils import tree_flatten

            return sum(int(v.size) for _, v in tree_flatten(target.parameters()))
        return _count_jax(target)
    except Exception:  # noqa: BLE001 -- an unusual model object; the trace's count still exists
        return None


def _count_jax(target: Any) -> int | None:
    import jax

    try:
        from flax import nnx

        if isinstance(target, nnx.Module):
            target = nnx.state(target, nnx.Param)
    except ImportError:
        pass
    leaves = [x for x in jax.tree_util.tree_leaves(target) if _is_array(x)]
    return sum(int(x.size) for x in leaves) if leaves else None


def _divergence(eval_out: Any, train_out: Any) -> str:
    """Native models have no op recorder: shapes decide ``structural``, values the rest."""

    if shapes(eval_out) != shapes(train_out):
        return "structural"
    return "none" if _same(eval_out, train_out) else "statistical"


def _timed(native: Native) -> tuple[Any, float]:
    with worker._stage("forward_error"):
        started = time.perf_counter()
        out = native()
        for a in arrays(out):  # JAX and MLX are lazy: the forward ends when its values exist
            to_numpy(a)
        return out, time.perf_counter() - started


# --- modes ----------------------------------------------------------------------------


def forward_mode(request: RunRequest, attempt_dir: Path) -> RunResult:
    """Eval then train on fresh native models; the eval outcome is the result."""

    del attempt_dir
    backend = request.recipe.backend
    prepare(backend, request.caps.threads)
    native = build(request)
    eval_out, secs = _timed(native)
    peak = worker._rss_mb()
    contract = _jax_contract(native) if backend == "jax" else None
    boundary = input_boundary(native, backend, eval_out)
    params = count_params(native, backend)
    train, div, msg = None, None, contract
    if native.switched:
        try:
            train_out, _ = _timed(build(request, train=True))
            train, div = "pass", _divergence(eval_out, train_out)
        except worker.Failure as failure:
            train = f"fail:{failure.cls}"
    else:
        msg = "; ".join(filter(None, (
            msg, f"no train mode: a {type(native.root).__name__} has no train/eval switch")))
    row = worker._base_row(request, stage="forward", outcome="pass", mode="eval", params=params,
                           params_source="counted" if params is not None else None,
                           forward_secs=round(secs, 3), forward_peak_rss_mb=peak,
                           input_boundary=boundary, msg=msg)
    return worker._result(request, "pass", rows=(row,), output_shapes=shapes(eval_out),
                          params=params, modes=Modes(eval="pass", train=train, divergence=div),
                          input_boundary=boundary, msg=msg)


CONTRACT = "capture_error:jax_contract"
CLOSED_CONSTANTS = "closed-jaxpr constants"  # TorchLens 2.35.3's refusal text


def _jax_contract(native: Native) -> str | None:
    """The JAX trace contract, checked at the 2b try exactly as the trace checks it: one
    ``tl.trace(..., backend="jax")`` of the built model. TorchLens must accept what ``build``
    returned (the model itself, or a ``(function, parameters)`` pair, not a wrapper object) and
    the function must close over no constants (parameters and captured arrays are explicit pytree
    leaves). Either refusal fails the try as ``capture_error:jax_contract``; any other capture
    problem is the trace's to type, and its first line is returned as a note for the row."""

    import torchlens as tl
    from torchlens.errors import BackendMismatchError

    fix = ("build must return the model itself or a (function, parameters) pair called as "
           "fn(params, *inputs), with every array it uses passed in, none closed over")
    try:
        tl.trace(native.model, native.inputs, input_kwargs=native.kwargs, backend="jax")
    except BackendMismatchError as exc:
        raise worker.Failure(CONTRACT, f"{worker._first_line(exc)} ({fix})") from exc
    except ValueError as exc:
        if CLOSED_CONSTANTS in str(exc):
            raise worker.Failure(CONTRACT, f"{worker._first_line(exc)} ({fix})") from exc
        return f"trace contract check: {worker._first_line(exc)}"
    except Exception as exc:  # noqa: BLE001 -- the trace types it; the try only notes it
        if isinstance(exc, MemoryError) or worker._MEMORY.search(str(exc)):
            raise worker.Failure("memory_cap", worker._first_line(exc)) from exc
        return f"trace contract check: {worker._first_line(exc)}"
    return None


@contextmanager
def _capture() -> Iterator[None]:
    """A capture stage; TorchLens refusing the model's construct is ``not_traceable_framework``."""

    try:
        with worker._stage("capture_error"):
            yield
    except worker.Failure as failure:
        cause = failure.__cause__
        if failure.cls == "capture_error" and cause is not None and (
                "Unsupported" in type(cause).__name__ or "Conformance" in type(cause).__name__):
            raise worker.Failure("not_traceable_framework:unsupported", failure.msg,
                                 failure.tail) from cause
        raise


def validation_state(result: Any) -> str:
    """A ``tl.validate`` result's state; a plain bool is ``passed`` or ``failed``."""

    if isinstance(result, bool):
        return "passed" if result else "failed"
    state = str(getattr(result, "state", "unavailable"))
    return state if state in ("passed", "failed", "unverified", "unavailable") else "unavailable"


@dataclass(frozen=True)
class Check:
    """A capture and its validation: the live trace, the state, a failure message, seconds, and
    why the warm-up raised (``None`` when it ran)."""

    trace: Any
    state: str
    msg: str | None
    secs: float
    warm: str | None = None


def _warm(native: Native) -> str | None:
    """One plain forward before a capture, as the torch worker warms (``worker._warm``): in
    inference (a train-mode model with a switch goes to eval and back, an eval one is never
    switched, which can clear a cache; Keras drops ``training``). Returns why it raised, or
    ``None``; recorded, never the outcome."""

    model = native.model
    kwargs = {k: v for k, v in native.kwargs.items() if k != "training"}
    switch = native.switched and bool(getattr(model, "training", False)) and callable(
        getattr(model, "eval", None)) and callable(getattr(model, "train", None))
    try:
        if switch:
            model.eval()
        model(*native.inputs, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- model code raises anything; recorded on the row
        if isinstance(exc, MemoryError) or worker._MEMORY.search(str(exc)):
            raise
        return worker._first_line(exc)
    finally:
        if switch:
            model.train()
    return None


def _validate(request: RunRequest, train: bool, truth: Any) -> Check:
    """Capture a fresh build and validate that live capture (forward scope, replay of the
    captured ops); ``truth`` is the reference forward's output, for backends that compare it."""

    import torchlens as tl

    backend = request.recipe.backend
    native = build(request, train=train)
    with _capture():
        warm = _warm(native)
        started = time.perf_counter()
        trace = tl.trace(native.model, native.inputs, input_kwargs=native.kwargs, backend=backend)
        result = trace.validate_forward_pass(arrays(truth),
                                             validate_metadata=VALIDATE_METADATA[backend])
        secs = time.perf_counter() - started
    state = validation_state(result)
    msg = None
    if state != "passed":
        status = trace.validation_replay_status  # the counts behind a bare bool verdict
        msg = (f"{status.message} ({status.replayed_node_count} replayed, "
               f"{status.failed_node_count} failed, {status.effect_region_node_count} effect "
               f"regions, {status.pure_unverified_node_count} pure unverified)")
        failure = tl.validation.last_validation_failure() if state == "failed" else None
        if failure is not None:
            msg = failure.summary() if hasattr(failure, "summary") else str(failure)
        msg = worker._HOME.sub("~/", msg[:300])
    return Check(trace, state, msg, secs, warm)


def _ready_capture(request: RunRequest, train: bool, check: Check) -> tuple[Any, float, str | None]:
    """The trace to save: a fresh ``intervention_ready`` capture when the request asks and the
    preview accepts it, else the validated capture (with a note when the preview refuses)."""

    import torchlens as tl
    from torchlens.options import CaptureOptions

    backend = request.recipe.backend
    if not request.intervention_ready:
        return check.trace, check.secs, None
    ready = backend not in NO_INTERVENTION_READY
    note = None if ready else f"intervention_ready refused by the {backend} preview"
    if not ready and backend not in SAVE_FRESH:
        return check.trace, check.secs, note
    native = build(request, train=train)
    options = {"capture": CaptureOptions(intervention_ready=True)} if ready else {}
    with _capture():
        _warm(native)  # a raise is recorded from the validation's warm-up of the same build
        started = time.perf_counter()
        trace = tl.trace(native.model, native.inputs, input_kwargs=native.kwargs, backend=backend,
                         **options)
        return trace, time.perf_counter() - started, note


def _summarize(request: RunRequest, trace: Any) -> tuple[dict[str, Any], Any, str | None]:
    """The hash family and the trace summary; each that the torch-shaped summary code cannot
    compute for a preview trace is left out (a minimal summary instead), with the reason."""

    from menagerie import trace_summary

    notes, hashes = [], None
    try:
        hashes = trace_summary.hash_family(trace)
    except Exception as exc:  # noqa: BLE001 -- recorded on the row; the capture still counts
        notes.append(f"hashes: {worker._first_line(exc)}")
    try:
        summary = trace_summary.summarize_trace(request.id, trace, request.recipe.rev,
                                                hashes=hashes)
    except Exception as exc:  # noqa: BLE001 -- as above
        notes.append(f"summary: {worker._first_line(exc)}")
        summary = {"id": request.id, "recipe_rev": request.recipe.rev,
                   "n_ops": int(trace.num_ops), "graph_hash": trace.graph_shape_hash or None,
                   **({"hashes": hashes} if hashes else {})}
    return summary, hashes, "; ".join(notes) or None


def _save(trace: Any, attempt_dir: Path) -> tuple[worker.Saved, str | None]:
    """The runnable save when TorchLens allows it for this backend, else the portable one, with
    the round-trip check (``worker.Saved``: the state and the level written); and why no file
    was written, if none was. At 2.35.3 every preview refuses the runnable level ("requires the
    persistent-buffer state universe"): ``refused:runnable_state``, and the portable save stays."""

    import shutil

    import torchlens as tl

    path = attempt_dir / worker.TRACE_FILE
    shutil.rmtree(path, ignore_errors=True)
    try:
        tl.save(trace, path, level="runnable")
        runnable, level = "ready", "runnable"
    except Exception as exc:  # noqa: BLE001 -- the refusal is recorded, never raised
        state = "persistent-buffer state" in str(exc)
        runnable = "refused:runnable_state" if state else "refused:save_error"
        why = _portable(trace, path)
        if why:
            return worker.Saved(runnable, None), f"no save: {why}"
        level = "portable"
    try:
        same = tl.hash.trace(tl.load(path)) == tl.hash.trace(trace)
    except Exception:  # noqa: BLE001 -- a save that does not load is not usable
        return worker.Saved("refused:round_trip_load", level), None
    return worker.Saved(runnable if same else "refused:round_trip_hash", level), None


def _portable(trace: Any, path: Path) -> str | None:
    """Save portable without activations; return the failure, or ``None`` when saved."""

    import shutil

    import torchlens as tl

    shutil.rmtree(path, ignore_errors=True)
    try:
        tl.save(trace, path, level="portable", include_outs=False)
        return None
    except Exception as exc:  # noqa: BLE001 -- recorded on the row
        shutil.rmtree(path, ignore_errors=True)
        for partial in path.parent.glob(f"{path.name}.tmp.*"):
            shutil.rmtree(partial, ignore_errors=True)
        return worker._first_line(exc)


def trace_mode(request: RunRequest, attempt_dir: Path) -> RunResult:
    """Timed native forward; ``tl.validate`` (forward scope) for the verdict; one capture for the
    op count, graph hash, module mode, summary and save."""

    import shutil

    from menagerie import trace_summary

    backend, train = request.recipe.backend, request.model_mode == "train"
    prepare(backend, request.caps.threads)
    native = build(request, train=train)
    truth, forward_secs = _timed(native)
    forward_peak = worker._rss_mb()
    boundary = input_boundary(native, backend, truth)
    params = count_params(native, backend)
    del native
    gc.collect()
    check = _validate(request, train, truth)
    trace = check.trace
    summary, hashes, summary_note = _summarize(request, trace)
    ready, save_secs, note = _ready_capture(request, train, check)
    saved, save_note = _save(ready, attempt_dir)
    runnable = saved.state
    shutil.rmtree(attempt_dir / worker.RENDER_FILE, ignore_errors=True)  # a retry's twin goes too
    if saved.level != "portable":  # a refused trace.tlspec is already the portable save
        _portable(ready, attempt_dir / worker.RENDER_FILE)
    module_mode = str(trace.module_identity_mode)
    timings = {"forward_secs": round(forward_secs, 3), "capture_secs": round(check.secs, 3),
               "forward_peak_rss_mb": forward_peak}
    summary = {**summary, **timings, "save_capture_secs": round(save_secs, 3),
               "backend": backend, "module_mode": module_mode, "validation_state": check.state}
    worker._write_summary(summary, attempt_dir)
    files = worker._files(attempt_dir, ("tlspec", worker.TRACE_FILE),
                          ("tlspec-portable", worker.RENDER_FILE), ("summary", worker.SUMMARY_FILE))
    sha = {f.kind: f.sha256 for f in files}
    passed = check.state == "passed"
    cls = None if passed else STATE_CLASS[check.state]
    warm_note = check.warm and f"warm-up raised: {check.warm}"
    msg = "; ".join(m for m in (check.msg, note, summary_note, save_note, warm_note) if m) or None
    row = worker._base_row(
        request, stage="trace", outcome="pass" if passed else "fail", cls=cls,
        mode=request.model_mode, capture="full", n_ops=int(trace.num_ops),
        hashes=trace_summary.ledger_hashes(hashes) if hashes else None,
        graph_hash=trace.graph_shape_hash or None, params=params,
        params_source="counted" if params is not None else None, peak_rss_mb=worker._rss_mb(),
        input_boundary=boundary, validation=VALIDATION, validation_state=check.state,
        module_mode=module_mode, runnable=runnable, warmed=check.warm is None,
        trace=sha.get("tlspec"), summary=sha.get("summary"), msg=msg, **timings,
    )
    fields = {"rows": (row,), "files": files, "params": params, "input_boundary": boundary,
              "output_shapes": shapes(truth)}
    if passed:
        return worker._result(request, "pass", **fields)
    return worker._result(request, "fail", cls=cls, msg=msg, **fields)
