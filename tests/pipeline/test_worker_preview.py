"""The worker's preview branch: TensorFlow, JAX, Paddle and MLX models through TorchLens, and the
GPU window.

A backend's tests run only where its framework is installed, so run this file in each native
environment (``uv run --project envs/<env> --frozen --with pytest python -m pytest
tests/pipeline/test_worker_preview.py``); in torch-main only the framework-free checks run. The
models are the small fixtures in ``native_fixtures/`` (no real names), built through the adapter
seam exactly as a 2b recipe is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from menagerie.pipeline import inputs, worker, worker_preview
from menagerie.pipeline.records import (
    Caps, LedgerRow, Recipe, RecordError, RunRequest, RunResult, from_dict, to_dict,
)

FIXTURES = Path(__file__).parent / "native_fixtures"
ENV = {"tf": "tf", "jax": "jax", "paddle": "paddle", "mlx": "mlx"}
MODULE = {"tf": "tensorflow", "jax": "jax", "paddle": "paddle", "mlx": "mlx.core"}


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def needs(backend: str) -> Any:
    return pytest.mark.skipif(not _has(MODULE[backend]), reason=f"{backend} is not installed here")


# --- framework-free --------------------------------------------------------------------


class _Status:
    def __init__(self, state: str) -> None:
        self.state = state


@pytest.mark.parametrize(("result", "state"), [
    (True, "passed"), (False, "failed"), (_Status("passed"), "passed"),
    (_Status("unverified"), "unverified"), (_Status("unavailable"), "unavailable"),
    (_Status("something-new"), "unavailable"), (object(), "unavailable"),
])
def test_validation_state_is_never_coerced(result: Any, state: str) -> None:
    assert worker_preview.validation_state(result) == state


def test_every_non_passing_state_has_a_class() -> None:
    assert set(worker_preview.STATE_CLASS) == {"failed", "unverified", "unavailable"}
    assert worker_preview.STATE_CLASS["unverified"] == "not_traceable_framework:unverified"


def test_native_inputs_refuse_an_unknown_backend() -> None:
    assert inputs.native(((1,), {}), "torch") == ((1,), {})
    with pytest.raises(RecordError):
        inputs.native(((1,), {}), "theano")


def test_module_mode_is_a_ledger_field() -> None:
    row = LedgerRow(id="m1", stage="trace", at="2026-10-05T00:00:00Z", host="w", env="jax",
                    recipe_rev="0123456789ab", outcome="pass", module_mode="pytree_module")
    assert from_dict(LedgerRow, to_dict(row)).module_mode == "pytree_module"
    with pytest.raises(RecordError):
        LedgerRow(id="m1", stage="trace", at="2026-10-05T00:00:00Z", host="w", env="jax",
                  recipe_rev="0123456789ab", outcome="pass", module_mode="guessed")


@pytest.mark.parametrize(("level", "twin"), [("runnable", True), ("portable", False)])
def test_a_preview_writes_its_drawing_save_only_beside_a_runnable_save(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, level: str, twin: bool) -> None:
    """Every preview refuses the runnable level at 2.35.3, so its ``trace.tlspec`` is the
    portable save and no ``render.tlspec`` twin follows; a retry's stale one goes."""

    from types import SimpleNamespace

    trace = SimpleNamespace(num_ops=3, graph_shape_hash="", module_identity_mode="object_module")
    check = SimpleNamespace(trace=trace, state="passed", msg=None, warm=None, secs=0.1)
    stand_ins = {"prepare": lambda backend, threads: None, "build": lambda r, train: object(),
                 "_timed": lambda native: ((), 0.1), "count_params": lambda native, b: 7,
                 "input_boundary": lambda native, backend, truth: "real",
                 "_validate": lambda r, train, truth: check, "shapes": lambda truth: (),
                 "_summarize": lambda r, t: ({}, None, None),
                 "_ready_capture": lambda r, train, c: (trace, 0.1, None)}
    for name, value in stand_ins.items():
        monkeypatch.setattr(worker_preview, name, value)
    written: list[str] = []

    def save(ready: Any, attempt_dir: Path) -> tuple[worker.Saved, None]:
        (attempt_dir / worker.TRACE_FILE).mkdir()
        (attempt_dir / worker.TRACE_FILE / "manifest.json").write_text("{}")
        state = "ready" if level == "runnable" else "refused:runnable_state"
        return worker.Saved(state, level), None

    def portable(ready: Any, path: Path) -> None:
        written.append(path.name)
        path.mkdir()
        (path / "manifest.json").write_text("{}")

    monkeypatch.setattr(worker_preview, "_save", save)
    monkeypatch.setattr(worker_preview, "_portable", portable)
    stale = tmp_path / worker.RENDER_FILE
    stale.mkdir()
    (stale / "stale").write_text("an earlier try's save")
    request = RunRequest(request="t-trace", id="m900001", mode="trace",
                         recipe=recipe_free("tf"), caps=Caps(rss_mb=8192, timeout_s=60, threads=1))
    result = worker_preview.trace_mode(request, tmp_path)
    kinds = {f.kind for f in result.files}
    assert written == ([worker.RENDER_FILE] if twin else [])
    assert ("tlspec-portable" in kinds) is twin and "tlspec" in kinds
    assert not (stale / "stale").exists()


def recipe_free(backend: str) -> Recipe:
    """A preview recipe that is never built (its builders are stood in)."""

    return Recipe(rev="0123456789ab", kind="adapter", env=ENV[backend], backend=backend,
                  input=inputs.contract("randn", (2, 8)), adapter=f"adapters/{backend}_models.py",
                  config={"model": "dropout_mlp"})


# --- native models ---------------------------------------------------------------------


def recipe(backend: str, config: dict[str, Any], preset: str, shape: tuple[int, ...]) -> Recipe:
    return Recipe(rev="0123456789ab", kind="adapter", env=ENV[backend], backend=backend,
                  input=inputs.contract(preset, shape), adapter=f"adapters/{backend}_models.py",
                  config=config)


def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rec: Recipe, mode: str,
        **fields: Any) -> RunResult:
    monkeypatch.setenv("MENAGERIE_ADAPTER", str(FIXTURES / f"{rec.backend}_models.py"))
    request = RunRequest(request=f"t-{mode}", id="m900001", mode=mode, recipe=rec,
                         caps=Caps(rss_mb=8192, timeout_s=1800, threads=1), **fields)
    result = worker.run_request(request, attempt_dir=tmp_path)
    from_dict(RunResult, to_dict(result))  # the result survives its own schema
    return result


def only_row(result: RunResult) -> LedgerRow:
    (row,) = result.rows
    assert isinstance(row, LedgerRow)
    return row


# At TorchLens 2.35.3 the TF preview leaves every variable read as an unverified effect region, so
# a Keras model validates `unverified` (a port, by the 2b rule), and the Paddle coverage oracle
# fails any op reading a layer's parameters, so a Paddle Layer validates `failed`; functions
# without parameters pass on both. These rows pin that behaviour until the pin moves.
CASES = [  # backend, config, preset, shape, module mode, divergence, validation state
    pytest.param("tf", {"model": "dropout_mlp"}, "randn", (2, 8), "object_module", "statistical",
                 "unverified", marks=needs("tf"), id="tf-dropout-mlp"),
    pytest.param("tf", {"model": "cnn"}, "image:nhwc", (1, 32, 32, 3), "object_module",
                 "statistical", "unverified", marks=needs("tf"), id="tf-cnn-nhwc"),
    pytest.param("tf", {"model": "function"}, "randn", (2, 8), "function_root", None, "passed",
                 marks=needs("tf"), id="tf-function"),
    pytest.param("jax", {"model": "eqx_mlp"}, "randn", (8,), "pytree_module", "none", "passed",
                 marks=needs("jax"), id="jax-equinox"),
    pytest.param("jax", {"model": "nnx_dropout"}, "randn", (2, 8), "pytree_module",
                 "statistical", "passed", marks=needs("jax"), id="jax-nnx"),
    pytest.param("jax", {"model": "linen_pair"}, "randn", (2, 8), "function_root", None,
                 "passed", marks=needs("jax"), id="jax-linen-pair"),
    pytest.param("paddle", {}, "image", (1, 3, 16, 16), "object_module", "statistical",
                 "failed", marks=needs("paddle"), id="paddle-cnn"),
    pytest.param("paddle", {"model": "function"}, "randn", (2, 8), "function_root", None,
                 "passed", marks=needs("paddle"), id="paddle-function"),
    pytest.param("mlx", {}, "randn", (2, 8), "object_module", "none", "passed",
                 marks=needs("mlx"), id="mlx-mlp"),
]
COLUMNS = ("backend", "config", "preset", "shape", "mode", "divergence", "state")


@pytest.mark.parametrize(COLUMNS, CASES)
def test_forward_mode_runs_eval_then_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           backend: str, config: dict[str, Any], preset: str,
                                           shape: tuple[int, ...], mode: str,
                                           divergence: str | None, state: str) -> None:
    del state
    result = run(tmp_path, monkeypatch, recipe(backend, config, preset, shape), "forward")
    assert result.outcome == "pass", (result.cls, result.msg, result.tail[-5:])
    assert result.output_shapes
    if mode != "function_root":  # a raw function has no parameter list of its own
        assert result.params and result.params > 0
    assert result.input_boundary == "real"
    assert result.modes.eval == "pass"
    if divergence is None:  # no train/eval switch: no train result, said on the row
        assert result.modes.train is None and "no train mode" in (result.msg or "")
    else:
        assert result.modes.train == "pass" and result.modes.divergence == divergence
    row = only_row(result)
    assert row.backend == backend and row.fw.startswith(worker_preview.FRAMEWORKS[backend])


@pytest.mark.parametrize(COLUMNS, CASES)
def test_trace_mode_passes_only_on_a_passed_state(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch, backend: str,
                                                  config: dict[str, Any], preset: str,
                                                  shape: tuple[int, ...], mode: str,
                                                  divergence: str | None, state: str) -> None:
    del divergence
    result = run(tmp_path, monkeypatch, recipe(backend, config, preset, shape), "trace")
    row = only_row(result)
    assert row.validation == "preview-replay" and row.module_mode == mode and row.warmed is True
    assert row.validation_state == state, (row.validation_state, row.msg, result.tail[-5:])
    if state == "passed":
        assert result.outcome == "pass" and row.outcome == "pass" and row.cls is None
    else:
        assert result.outcome == "fail" and row.cls == worker_preview.STATE_CLASS[state]
    assert row.n_ops and row.n_ops > 0 and row.capture == "full" and row.backend == backend
    assert row.runnable in ("ready", "refused:runnable_state") and row.summary and row.hashes
    kinds = {f.kind for f in result.files}
    assert {"summary", "tlspec"} <= kinds  # a portable twin only beside a runnable save
    assert ("tlspec-portable" in kinds) is (row.runnable == "ready")


@needs("jax")
@pytest.mark.parametrize("model", ["wrapper", "closed_constants"])
def test_a_jax_try_refuses_a_build_that_breaks_the_trace_contract(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    result = run(tmp_path, monkeypatch, recipe("jax", {"model": model}, "randn", (2, 8)), "forward")
    assert (result.outcome, result.cls) == ("fail", worker_preview.CONTRACT), result.msg
    assert "fn(params, *inputs)" in result.msg
    assert only_row(result).cls == worker_preview.CONTRACT  # the try's row says why, before done


@needs("tf")
def test_a_softmax_or_collapsed_head_still_reads_its_input(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    for model in ("softmax", "collapsed_head"):
        result = run(tmp_path / model, monkeypatch, recipe("tf", {"model": model}, "randn", (2, 8)),
                     "forward")
        assert result.outcome == "pass" and result.input_boundary == "real", model


@needs("tf")
def test_trace_mode_types_a_non_passing_state(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker_preview, "validation_state", lambda result: "unverified")
    result = run(tmp_path, monkeypatch, recipe("tf", {"model": "dropout_mlp"}, "randn", (2, 8)),
                 "trace")
    row = only_row(result)
    assert result.outcome == "fail" and row.cls == "not_traceable_framework:unverified"
    assert row.validation_state == "unverified"


@needs("tf")
def test_structure_only_and_gpu_refuse_a_preview_backend(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    rec = recipe("tf", {"model": "dropout_mlp"}, "randn", (2, 8))
    for mode in ("trace-structure-only", "gpu"):
        result = run(tmp_path / mode, monkeypatch, rec, mode)
        assert result.outcome == "fail" and result.cls == "capture_error:backend", mode


@needs("tf")
def test_native_inputs_keep_values_shape_and_dtype() -> None:
    import numpy as np
    import tensorflow as tf
    import torch

    t = torch.arange(6, dtype=torch.float32).view(2, 3)
    (args, kwargs) = inputs.native(((t,), {"ids": torch.tensor([1, 2])}), "tf")
    assert isinstance(args[0], tf.Tensor) and args[0].shape == (2, 3)
    assert np.array_equal(args[0].numpy(), t.numpy()) and kwargs["ids"].dtype == tf.int64


# --- gpu mode (torch) -----------------------------------------------------------------


def _torch_request(mode: str = "gpu") -> RunRequest:
    rec = Recipe(rev="0123456789ab", kind="library", env="torch-gpu", backend="torch",
                 input=inputs.contract("randn", (1, 4)), call="torch.nn.Linear", args=(4, 2))
    return RunRequest(request="t-gpu", id="m900041", mode=mode, recipe=rec,
                      caps=Caps(rss_mb=8192, timeout_s=2400, threads=1))


def test_gpu_mode_outside_the_window_is_infra(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setattr(worker, "in_gpu_window", lambda at=None: False)
    result = worker.run_request(_torch_request(), attempt_dir=tmp_path)
    assert result.outcome == "infra" and result.cls == "infra:gpu_window" and not result.rows


def test_a_cluster_gpu_attempt_ignores_the_worker_window(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is visible here")
    monkeypatch.setattr(worker, "in_gpu_window", lambda at=None: False)
    monkeypatch.setenv("MENAGERIE_HOST", "cluster")
    result = worker.run_request(_torch_request(), attempt_dir=tmp_path)
    assert result.outcome == "infra" and result.cls == "infra:no_gpu"


def test_gpu_mode_without_a_device_is_infra(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("a CUDA device is visible here")
    monkeypatch.setattr(worker, "in_gpu_window", lambda at=None: True)
    result = worker.run_request(_torch_request(), attempt_dir=tmp_path)
    assert result.outcome == "infra" and result.cls == "infra:no_gpu"


def test_only_gpu_requests_see_the_nvidia_devices(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("psutil")
    from menagerie.pipeline import runner

    monkeypatch.setattr(runner.glob, "glob", lambda pattern: ["/dev/nvidia0", "/dev/nvidiactl"])
    plain = runner._sandbox_argv(tmp_path, Path("/x/python"), [])
    gpu = runner._sandbox_argv(tmp_path, Path("/x/python"), [], gpu=True)
    assert "/dev/nvidia0" not in plain
    assert gpu[gpu.index("/dev/nvidia0") - 1] == "--dev-bind" and "/dev/nvidiactl" in gpu
