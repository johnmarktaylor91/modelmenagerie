"""The worker on real models: modes, divergence, input boundary, runnable save, hashes, summary.

Torch, TorchLens and timm are required; YOLOv8n needs ultralytics. Library recipes are built by
``harvest.construct`` and inputs by ``inputs.make_input``; until the harvest lane's versions
exist, a minimal importer and a seeded random tensor stand in. The toy models are built through
the same seam so the worker is exercised end to end.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("torchlens")
pytest.importorskip("timm")
from torch import nn  # noqa: E402

from menagerie import trace_summary  # noqa: E402
from menagerie.pipeline import harvest, inputs, read_json, tree_sha256, write_json  # noqa: E402
from menagerie.pipeline import worker  # noqa: E402
from menagerie.pipeline.records import (  # noqa: E402
    Caps, LedgerRow, Recipe, RecipeInput, RenderRow, RunRequest, RunResult, from_dict, to_dict,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "menagerie" / "data" / "routing_manifest.tsv"


class ModeGated(nn.Module):
    """A detection-style head: train returns raw features, eval decodes them (structural)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)

    def forward(self, x: Any) -> Any:
        y = self.conv(x)
        return y if self.training else y.flatten(1).softmax(-1)


class Dropped(nn.Module):
    """Dropout between two layers: same graph and shapes, different numbers in train."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(27, 16), nn.Dropout(0.5), nn.Linear(16, 4))

    def forward(self, x: Any) -> Any:
        return self.net(x)


class Plain(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(27, width), nn.ReLU(), nn.Linear(width, 4))

    def forward(self, x: Any) -> Any:
        return self.net(x)


class IgnoresInput(nn.Module):
    """A sentinel wrapper: runs a real layer on a constant and never reads its input."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 4)

    def forward(self, x: Any) -> Any:
        return self.lin(torch.ones(1, 8))


class Explodes(nn.Module):
    def forward(self, x: Any) -> Any:
        raise ValueError("shape mismatch in block 3")


class NeedsMissing(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import no_such_package  # noqa: F401


class LazyCache(nn.Module):
    """Fills a cache on its first forward, as ConViT's relative position indices do."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.cache: Any = None
        self.calls = 0

    def forward(self, x: Any) -> Any:
        self.calls += 1
        if self.cache is None:
            self.cache = torch.zeros(3)
        return self.lin(x) + self.cache


class BiasCache(nn.Module):
    """timm's LeViT attention biases: an eval call fills a cache on its first call, then reads it,
    so a never-run model's re-trace differs from its first capture."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.biases = nn.Parameter(torch.randn(3, 4))
        self.register_buffer("idx", torch.tensor([0, 2, 1]), persistent=False)
        self.cache: dict[str, Any] = {}

    def _bias(self, device: Any) -> Any:
        if self.training:
            return self.biases[:, self.idx]
        if str(device) not in self.cache:
            self.cache[str(device)] = self.biases[:, self.idx]
        return self.cache[str(device)]

    def train(self, mode: bool = True) -> BiasCache:
        super().train(mode)
        self.cache = {}  # timm clears the cache on any train() or eval()
        return self

    def forward(self, x: Any) -> Any:
        return self.lin(x) + self._bias(x.device)  # as timm adds the biases to the attention


class EvalRaises(nn.Module):
    """Runs only in train mode, so its (eval-mode) warm-up raises."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: Any) -> Any:
        if not self.training:
            raise RuntimeError("no eval path")
        return self.lin(x)


class ValueBranch(nn.Module):
    """Branches on a tensor's value, which a structure-only capture refuses by design."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: Any) -> Any:
        y = self.lin(x)
        if y.sum() > 0:
            return y.relu()
        return y


TOYS = {f"toys.{c.__name__}": c
        for c in (ModeGated, Dropped, Plain, IgnoresInput, Explodes, NeedsMissing, LazyCache,
                  BiasCache, EvalRaises, ValueBranch)}


def _import(call: str) -> Any:
    import importlib

    module, _, name = call.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _stand_in_input(spec: RecipeInput, *, seed: int = 0) -> tuple[tuple[Any, ...], dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(*spec.shape, generator=generator),), {}


def install_seam(mp: pytest.MonkeyPatch) -> dict[str, int]:
    """Route toys through ``harvest.construct``; stand in for stubs the harvest lane owns."""

    real_construct, calls = harvest.construct, {"construct": 0}
    try:
        inputs.make_input(recipe("toys.Plain", (1, 3, 3, 3)).input)
    except NotImplementedError:
        mp.setattr(inputs, "make_input", _stand_in_input)

    def construct(rec: Recipe, *, device: str = "cpu", data: Any = None) -> Any:
        calls["construct"] += 1
        if rec.call in TOYS:
            with torch.device(device):
                return TOYS[rec.call](*rec.args, **rec.kwargs)
        try:
            return real_construct(rec, device=device, data=data)
        except NotImplementedError:
            with torch.device(device):
                return _import(str(rec.call))(*rec.args, **rec.kwargs)

    mp.setattr(harvest, "construct", construct)
    return calls


@pytest.fixture(autouse=True)
def seam(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    return install_seam(monkeypatch)


def recipe(call: str, shape: tuple[int, ...], *args: Any, preset: str = "randn",
           **kwargs: Any) -> Recipe:
    return Recipe(rev="0123456789ab", kind="library", env="torch-main", backend="torch",
                  input=RecipeInput(preset=preset, kind="random-fallback", shape=shape),
                  call=call, args=args, kwargs=kwargs)


def timm_recipe(name: str, px: int = 224, **kwargs: Any) -> Recipe:
    return recipe("timm.create_model", (1, 3, px, px), name, preset="image", pretrained=False,
                  **kwargs)


def run(tmp_path: Path, rec: Recipe, mode: str = "trace", **fields: Any) -> RunResult:
    request = RunRequest(request=f"t-{mode}", id="m8839", mode=mode, recipe=rec,
                         caps=Caps(rss_mb=8192, timeout_s=1800, threads=2), **fields)
    result = worker.run_request(request, attempt_dir=tmp_path)
    from_dict(RunResult, to_dict(result))  # the result survives its own schema
    return result


def only_row(result: RunResult) -> LedgerRow:
    (row,) = result.rows
    assert isinstance(row, LedgerRow)
    return row


def only_row_of(result: RunResult, cls: type) -> Any:
    (row,) = [r for r in result.rows if isinstance(r, cls)]
    return row


# --- forward mode: divergence and input boundary -------------------------------------


@pytest.mark.parametrize(("call", "expected"), [
    ("toys.ModeGated", "structural"), ("toys.Dropped", "statistical"), ("toys.Plain", "none"),
])
def test_forward_classifies_train_eval_divergence(tmp_path: Path, seam: dict[str, int],
                                                  call: str, expected: str) -> None:
    result = run(tmp_path, recipe(call, (2, 3, 3, 3)), mode="forward")
    assert result.outcome == "pass" and result.modes is not None
    assert (result.modes.eval, result.modes.train, result.modes.divergence) == (
        "pass", "pass", expected)
    assert seam["construct"] == 2, "eval and train each run on a freshly built model"
    row = only_row(result)
    assert row.stage == "forward" and row.forward_secs is not None and row.forward_peak_rss_mb


def test_yolov8n_from_yaml_is_structural(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    rec = recipe("ultralytics.nn.tasks.DetectionModel", (1, 3, 640, 640), "yolov8n.yaml",
                 verbose=False)
    result = run(tmp_path, rec, mode="forward")
    assert result.modes is not None and result.modes.divergence == "structural"
    assert result.output_shapes[0] == (1, 84, 8400)


def test_input_boundary_wrapper_real_and_none(tmp_path: Path) -> None:
    assert run(tmp_path, recipe("toys.IgnoresInput", (1, 8)), mode="forward"
               ).input_boundary == "wrapper"
    resnet = run(tmp_path, timm_recipe("resnet18"), mode="forward")
    assert resnet.input_boundary == "real"
    assert worker.input_boundary(worker.Subject(Plain(), (), {}), None) == "none"
    # torchvision's ViT has a zero-initialised head: a constant output, a body that reads the input.
    vit = recipe("torchvision.models.get_model", (1, 3, 224, 224), "vit_b_16", weights=None)
    assert run(tmp_path, vit, mode="forward").input_boundary == "real"


@pytest.mark.parametrize("model_mode", ["eval", "train"])
def test_validation_and_save_warm_once_and_hashes_stay_cold(tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch,
                                                             model_mode: str) -> None:
    seen: list[tuple[int, int, bool, bool, int]] = []
    hashed: list[int] = []
    real, real_hash = worker._warm, worker._hash_capture

    def hash_spy(subject: worker.Subject, *, structure_only: bool = False, **kw: Any) -> Any:
        hashed.append(subject.model.calls)
        return real_hash(subject, structure_only=structure_only, **kw)

    def spy(subject: worker.Subject, threads: int | None = None) -> str | None:
        before, mode = subject.model.calls, subject.model.training
        why = real(subject, threads)
        seen.append((before, subject.model.calls, mode, subject.model.training,
                     torch.get_num_threads()))
        return why

    monkeypatch.setattr(worker, "_warm", spy)
    monkeypatch.setattr(worker, "_hash_capture", hash_spy)
    result = run(tmp_path, recipe("toys.LazyCache", (2, 3)), model_mode=model_mode)
    train = model_mode == "train"
    # Validation and save each build afresh and warm once (a retry adds one), in the request's
    # mode before and after, at the worker's thread count; the hash family's capture never runs
    # a warm-up (a warmed cache can put the input size into the architecture hash).
    assert len(seen) >= 2, seen
    assert all(s == (0, 1, train, train, 2) for s in seen), seen
    assert hashed == [0]
    assert only_row(result).warmed is True


def test_a_warm_up_that_raises_is_recorded_not_the_outcome(tmp_path: Path) -> None:
    result = run(tmp_path, recipe("toys.EvalRaises", (2, 3)), model_mode="train")
    row = only_row(result)
    assert row.warmed is False, row.msg
    assert "warm-up raised: RuntimeError: no eval path" in (row.msg or "")
    assert row.n_ops and row.validation_state in ("passed", "failed")


# levit_conv_128s, not levit_128s: at 2.35.3 the latter also fails a perturbation check at
# ``reshape_as`` (a TorchLens validation bug fixed upstream), a replay failure warm or cold.
@pytest.mark.parametrize("rec", [recipe("toys.BiasCache", (3, 3)), timm_recipe("levit_conv_128s")],
                         ids=["toy", "levit_conv_128s"])
def test_warming_lets_a_first_call_cache_validate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  rec: Recipe) -> None:
    warm = only_row(run(tmp_path / "warm", rec))
    assert (warm.outcome, warm.validation_state, warm.warmed) == ("pass", "passed", True), warm.msg
    # Unwarmed, TorchLens's retrace check stops the never-run model by design: its replay
    # status is unverified, typed apart from a replay failure.
    monkeypatch.setattr(worker, "_warm", lambda subject, threads=None: None)
    cold = only_row(run(tmp_path / "cold", rec))
    assert (cold.outcome, cold.cls, cold.validation_state) == (
        "fail", "torchlens_unverified", "unverified"), cold.msg
    assert "trace_retrace_structure_mismatch" in (cold.msg or "")


@pytest.mark.parametrize(("check", "expected"), [
    (worker.Validation(True, None, 1.0, state="passed"), (None, "passed")),
    (worker.Validation(False, "m", 1.0, state="unverified", failed=0, check="retrace_mismatch",
                       reason="trace_retrace_structure_mismatch"),
     ("torchlens_unverified", "unverified")),
    # Downgraded after a retrace mismatch, but the replay failed too: a replay failure.
    (worker.Validation(False, "m", 1.0, state="unverified", failed=1, check="retrace_mismatch"),
     ("torchlens_gap:replay", "failed")),
    (worker.Validation(False, "m", 1.0, state="unverified", failed=0, check="perturbation"),
     ("torchlens_gap:replay", "failed")),
    (worker.Validation(False, "m", 1.0, state="failed"), ("torchlens_gap:replay", "failed")),
    (worker.Validation(False, "m", 1.0), ("torchlens_gap:replay", "failed")),
])
def test_verdict_class_keys_off_the_replay_state(check: worker.Validation,
                                                 expected: tuple[str | None, str]) -> None:
    assert worker.verdict_class(check) == expected


def test_failures_are_typed_rows(tmp_path: Path) -> None:
    missing = run(tmp_path, recipe("toys.NeedsMissing", (1, 3)), mode="forward")
    assert (missing.outcome, missing.cls) == ("fail", "missing_dependency")
    assert only_row(missing).cls == "missing_dependency"
    assert missing.modes is not None and missing.modes.eval == "fail:missing_dependency"
    broken = run(tmp_path, recipe("toys.Explodes", (1, 3)))
    assert (broken.cls, broken.msg) == ("forward_error", "ValueError: shape mismatch in block 3")
    assert broken.tail and only_row(broken).capture == "full"


def test_main_writes_result_json(tmp_path: Path) -> None:
    request = RunRequest(request="t-main", id="m1", mode="forward",
                         recipe=recipe("toys.Plain", (1, 3, 3, 3)),
                         caps=Caps(rss_mb=2048, timeout_s=60))
    write_json(request, tmp_path / "request.json")
    assert worker.main(tmp_path) == 0
    assert read_json(RunResult, tmp_path / "result.json").outcome == "pass"


# --- trace mode: runnable save, refusal, hashes, timings -----------------------------


@pytest.fixture(scope="module")
def resnet18_trace(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, RunResult]:
    attempt = tmp_path_factory.mktemp("resnet18")
    mp = pytest.MonkeyPatch()
    try:
        install_seam(mp)
        return attempt, run(attempt, timm_recipe("resnet18"))
    finally:
        mp.undo()


def test_resnet18_trace_row(resnet18_trace: tuple[Path, RunResult]) -> None:
    attempt, result = resnet18_trace
    row = only_row(result)
    assert (row.stage, row.capture, row.n_ops, row.params) == ("trace", "full", 85, 11689512)
    assert row.validation == "torch-replay" and row.validation_state in ("passed", "failed")
    assert (row.outcome == "pass") == (row.validation_state == "passed")
    assert row.runnable == "ready" and row.input_boundary == "real"
    assert row.forward_secs and row.capture_secs and row.forward_peak_rss_mb and row.peak_rss_mb
    assert row.hashes is not None and row.hashes.architecture and row.hashes.model
    assert row.trace == tree_sha256(attempt / worker.TRACE_FILE)
    summary = json.loads((attempt / worker.SUMMARY_FILE).read_text())
    assert summary["trace_summary_version"] == "1.2.0"
    assert summary["skeleton_hash"] == row.hashes.skeleton
    assert summary["capture_secs"] == row.capture_secs and summary["save_capture_secs"]


def test_resnet18_runnable_round_trip_runs(resnet18_trace: tuple[Path, RunResult]) -> None:
    attempt, result = resnet18_trace
    loaded = tl.load(attempt / worker.TRACE_FILE)
    skeleton = only_row(result).hashes.skeleton  # type: ignore[union-attr]
    assert tl.hash.trace(loaded) == skeleton
    x, _ = _stand_in_input(RecipeInput(preset="image", kind="random-fallback",
                                       shape=(1, 3, 224, 224)))
    with pytest.warns(Warning, match="carries no model weights"):
        rerun = loaded.run(inputs=x[0], seed=0)
    assert tl.hash.trace(rerun.trace) == skeleton


def test_render_mode_reads_the_store_blob(resnet18_trace: tuple[Path, RunResult], tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """A render request names the portable save's sha; the worker finds it in the store."""

    import shutil

    from menagerie.pipeline import render
    from menagerie.pipeline.records import RenderRow

    attempt, result = resnet18_trace
    (portable,) = [f for f in result.files if f.kind == "tlspec-portable"]
    blob = tmp_path / "store" / "sha256" / portable.sha256[:2] / portable.sha256
    shutil.copytree(attempt / worker.RENDER_FILE, blob)
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    seen: dict[str, Any] = {}

    def draw_none(trace: Any, *, model_id: str, trace_sha: str, out_dir: Path,
                  variants: Any = None) -> tuple[RenderRow, list[Any]]:
        seen["n_ops"] = trace.num_ops
        return RenderRow(id=model_id, at="2026-10-04T00:00:00Z", trace=trace_sha,
                         renderer_rev="stand-in", tl=tl.__version__, outcome="pass"), []

    monkeypatch.setattr(render, "render_set", draw_none)
    out = run(tmp_path / "render", timm_recipe("resnet18"), mode="render", trace=portable.sha256)
    assert out.outcome == "pass" and seen["n_ops"] == 85
    assert out.rows[0].trace == portable.sha256


@pytest.fixture(scope="module")
def refused_trace(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, RunResult]:
    """resnet18 without ``intervention_ready``: the runnable preflight refuses it."""

    attempt = tmp_path_factory.mktemp("resnet18-refused")
    mp = pytest.MonkeyPatch()
    try:
        install_seam(mp)
        return attempt, run(attempt, timm_recipe("resnet18"), intervention_ready=False)
    finally:
        mp.undo()


def test_plain_capture_records_preflight_refusal(refused_trace: tuple[Path, RunResult]) -> None:
    attempt, result = refused_trace
    row = only_row(result)
    assert row.runnable == "refused:missing_callable_ref"
    assert row.trace == tree_sha256(attempt / worker.TRACE_FILE)


def test_a_refused_runnable_save_writes_no_portable_twin(
        refused_trace: tuple[Path, RunResult]) -> None:
    """``trace.tlspec`` is already the portable save, so ``render.tlspec`` is not written again
    (m20100 on 2026-10-28: two 1.8 GB portable saves of one capture)."""

    attempt, result = refused_trace
    manifest = json.loads((attempt / worker.TRACE_FILE / "manifest.json").read_text())
    assert manifest["save_level"] == "portable"
    assert not (attempt / worker.RENDER_FILE).exists()
    assert {f.kind for f in result.files} == {"tlspec", "summary"}


def _draw_stand_in(seen: dict[str, Any]) -> Any:
    def draw(trace: Any, *, model_id: str, trace_sha: str, out_dir: Path,
             variants: Any = None) -> tuple[RenderRow, list[Any]]:
        seen["n_ops"], seen["key"] = trace.num_ops, trace_sha
        return RenderRow(id=model_id, at="2026-10-28T00:00:00Z", trace=trace_sha,
                         renderer_rev="stand-in", tl=tl.__version__, outcome="pass"), []
    return draw


def test_render_mode_draws_a_refused_trace_save_staged_or_from_the_store(
        refused_trace: tuple[Path, RunResult], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """With no ``tlspec-portable``, render reads ``trace.tlspec``: staged in the attempt
    (``flow.render_request``), else the store blob the request names (the ledger ``trace``)."""

    import shutil

    from menagerie.pipeline import render

    attempt, result = refused_trace
    (save,) = [f for f in result.files if f.kind == "tlspec"]
    seen: dict[str, Any] = {}
    monkeypatch.setattr(render, "render_set", _draw_stand_in(seen))
    staged = tmp_path / "staged"
    shutil.copytree(attempt / worker.TRACE_FILE, staged / worker.TRACE_FILE)
    out = run(staged, timm_recipe("resnet18"), mode="render", trace=save.sha256)
    assert out.outcome == "pass" and seen == {"n_ops": 85, "key": save.sha256}

    seen.clear()
    blob = tmp_path / "store" / "sha256" / save.sha256[:2] / save.sha256
    shutil.copytree(attempt / worker.TRACE_FILE, blob)
    monkeypatch.setenv("MENAGERIE_STORE", str(tmp_path / "store"))
    out = run(tmp_path / "from-store", timm_recipe("resnet18"), mode="render", trace=save.sha256)
    assert out.outcome == "pass" and seen == {"n_ops": 85, "key": save.sha256}


@pytest.mark.parametrize(("level", "twin"), [("runnable", True), ("portable", False),
                                             (None, True)])
def test_the_drawing_save_is_written_only_beside_a_non_portable_trace_save(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, level: str | None, twin: bool) -> None:
    """The level ``_save_runnable`` reports decides, never its refusal text; a retry's stale
    ``render.tlspec`` never outlives a save that needs none."""

    written: list[str] = []

    def save_runnable(trace: Any, path: Path, skeleton: str) -> worker.Saved:
        if level is not None:
            path.mkdir()
            (path / "manifest.json").write_text(json.dumps({"save_level": level}))
        return worker.Saved("ready" if level == "runnable" else "refused:test", level)

    def portable(trace: Any, path: Path) -> bool:
        written.append(path.name)
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"save_level": "portable"}))
        return True

    stale = tmp_path / worker.RENDER_FILE
    stale.mkdir(parents=True)
    (stale / "stale").write_text("an earlier try's save")
    monkeypatch.setattr(worker, "_save_runnable", save_runnable)
    monkeypatch.setattr(worker, "_portable", portable)
    result = run(tmp_path, recipe("toys.Plain", (1, 3, 3, 3)))
    kinds = {f.kind for f in result.files}
    assert written == ([worker.RENDER_FILE] if twin else [])
    assert ("tlspec-portable" in kinds) is twin and ("tlspec" in kinds) is (level is not None)
    assert not (stale / "stale").exists()
    assert not any(isinstance(r, RenderRow) for r in result.rows)  # a save: render mode draws


def test_hash_family_is_input_invariant_and_width_sensitive() -> None:
    import timm

    def family(model: Any, px: int) -> dict[str, Any]:
        return trace_summary.hash_family(tl.trace(model, torch.randn(1, 3, px, px)))

    torch.manual_seed(0)
    base = family(timm.create_model("resnet18").eval(), 224)
    assert family(timm.create_model("resnet18").eval(), 256) == base
    narrow = family(timm.create_model("resnet18", num_classes=10).eval(), 224)
    assert narrow["skeleton"] == base["skeleton"]
    assert narrow["architecture"] != base["architecture"] and narrow["model"] != base["model"]
    wide = trace_summary.hash_family(tl.trace(Plain(32).eval(), torch.randn(1, 3, 3, 3)))
    thin = trace_summary.hash_family(tl.trace(Plain(16).eval(), torch.randn(1, 3, 3, 3)))
    assert wide["architecture"] != thin["architecture"]


# --- the June comparison and the structure-only giant --------------------------------


# Columns that measure the run rather than the model: the worker's timings, and the forward peak
# memory, a runtime measurement that varies run to run (as tests/test_menagerie_trace_summary.py
# also treats it).
CAPTURE_DEPENDENT = {"forward_secs", "capture_secs", "forward_peak_rss_mb",
                     "save_capture_secs", "forward_peak_memory_bytes", "forward_peak_memory_mb"}


def manifest_row(stable_id: str) -> dict[str, str]:
    with MANIFEST.open(encoding="utf-8") as handle:
        return next(r for r in csv.DictReader(handle, delimiter="\t")
                    if r["stable_id"] == stable_id)


@pytest.mark.parametrize(("stable_id", "name"), [
    ("m8839", "resnet18"), ("m8850", "resnet34"), ("m7341", "mobilenetv3_small_050"),
])
def test_summary_and_legacy_hash_match_june(tmp_path: Path, stable_id: str, name: str) -> None:
    """Op counts equal the June manifest; the 1.1.0 columns equal the June digest path's.

    The manifest's ``graph_shape_hash`` came from TorchLens 2.27.0, whose traces list buffer
    layers in a different order; installing 2.27.0 reproduces it exactly, so the legacy hash is
    compared with the June code path at the pin, which reads the same TorchLens attribute.
    """

    import timm

    result = run(tmp_path, timm_recipe(name))
    row, june = only_row(result), manifest_row(stable_id)
    assert row.n_ops == int(june["n_ops"])
    ours = json.loads((tmp_path / worker.SUMMARY_FILE).read_text())
    torch.manual_seed(0)
    model = timm.create_model(name).eval()
    (x,), _ = inputs.make_input(timm_recipe(name).input)
    theirs = trace_summary.summarize_model("m8839", model, x, "0123456789ab")
    assert row.graph_hash == ours["graph_shape_hash"] == theirs["graph_shape_hash"]
    differ = {c: (ours[c], theirs[c]) for c in trace_summary.TRACE_SUMMARY_COLUMNS
              if c not in CAPTURE_DEPENDENT and ours[c] != theirs[c]}
    assert not differ


def test_structure_only_giant_with_banner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """beit_large_patch16_512 on the meta device: geometry params, never validated, bannered.

    No save level accepts this trace at 2.35.2, so the worker renders it in process; the
    renderer is the render lane's, so a one-variant stand-in draws the default diagram here.
    """

    from menagerie.pipeline import render
    from menagerie.pipeline.records import RenderArtifact, RenderRow

    def draw_one(trace: Any, *, model_id: str, trace_sha: str, out_dir: Path,
                 variants: Any = None) -> tuple[RenderRow, list[Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        trace.draw(vis_outpath=str(out_dir / "unrolled-none"), vis_fileformat="svg",
                   vis_save_only=True)
        svg = out_dir / "unrolled-none.svg"
        art = RenderArtifact(kind="svg", variant="unrolled-none", sha256=tree_sha256(svg),
                             bytes=svg.stat().st_size)
        return RenderRow(id=model_id, at="2026-10-04T00:00:00Z", trace=trace_sha,
                         renderer_rev="stand-in", tl=tl.__version__, outcome="pass",
                         renders=(art,)), []

    monkeypatch.setattr(render, "render_set", draw_one)
    result = run(tmp_path, timm_recipe("beit_large_patch16_512", 512),
                 mode="trace-structure-only")
    row = only_row_of(result, LedgerRow)
    assert (row.capture, row.validation_state, row.params_source) == (
        "structure_only", "structure_only", "geometry")
    assert (row.params, row.runnable, row.outcome) == (305674728, "n/a", "pass")
    assert row.warmed is None, row.msg  # structure-only, like the hash family, is never warmed
    assert row.hashes is not None and row.n_ops and row.capture_secs is not None
    assert "hypothes" in (tmp_path / "render" / "unrolled-none.svg").read_text().lower()
    assert only_row_of(result, RenderRow).trace == row.summary


def test_a_full_capture_that_saved_nothing_is_drawn_in_process(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both saves refused (fasterrcnn m4851 at 2.35.3): render mode would have nothing to load,
    so the trace attempt draws the live trace, keyed on its summary like a structure-only one."""

    from menagerie.pipeline import render
    from menagerie.pipeline.records import RenderRow

    drawn: list[str] = []

    def draw_one(trace: Any, *, model_id: str, trace_sha: str, out_dir: Path,
                 variants: Any = None) -> tuple[RenderRow, list[Any]]:
        drawn.append(trace_sha)
        return RenderRow(id=model_id, at="2026-10-06T00:00:00Z", trace=trace_sha,
                         renderer_rev="stand-in", tl=tl.__version__, outcome="pass"), []

    monkeypatch.setattr(render, "render_set", draw_one)
    monkeypatch.setattr(worker, "_portable", lambda trace, path: False)
    monkeypatch.setattr(worker, "_save_runnable",
                        lambda trace, path, skeleton: worker.Saved("refused:test", None))
    result = run(tmp_path, recipe("toys.Plain", (1, 3, 3, 3)))
    row = only_row_of(result, LedgerRow)
    assert row.outcome == "pass" and row.trace is None and row.summary is not None
    assert drawn == [row.summary] and only_row_of(result, RenderRow).trace == row.summary
    assert "rendered in process" in (row.msg or "")

    drawn.clear()
    monkeypatch.setattr(worker, "_portable", lambda trace, path: True)  # a save: render mode draws
    (tmp_path / "saved").mkdir()
    only_row_of(run(tmp_path / "saved", recipe("toys.Plain", (1, 3, 3, 3))), LedgerRow)
    assert drawn == []


def test_a_value_branch_refusal_records_its_offending_line(tmp_path: Path) -> None:
    """TorchLens's message runs past the 300-character first line before naming the line; the
    row carries ``fields["offenses"]`` instead, and survives its own schema."""

    from menagerie.pipeline.records import dumps

    result = run(tmp_path, recipe("toys.ValueBranch", (1, 3)), mode="trace-structure-only")
    row = only_row(result)
    assert (row.outcome, row.cls, row.capture) == ("fail", "capture_error", "structure_only")
    assert "ValueDependentBranchError" in str(row.msg)
    assert row.offenses, row.msg
    (offense,) = row.offenses
    assert offense.file.endswith("test_worker.py") and not offense.file.startswith(
        str(Path.home()))
    source = Path(__file__).read_text().splitlines()
    assert isinstance(offense.line, int) and "if y.sum() > 0" in source[offense.line - 1]
    assert from_dict(LedgerRow, json.loads(dumps(row))) == row


def test_offenses_are_read_defensively() -> None:
    class Refusal(Exception):
        fields = {"offenses": (
            {"file": str(Path.home() / "x.py"), "line": 3, "consumer_kind": "if", "extra": 1},
            {"line": 4}, "junk", {"file": "y.py", "line": True}) + ({"file": "z.py"},) * 20}

    offenses = worker._offenses(Refusal())
    assert offenses[0].file == "~/x.py" and offenses[0].line == 3
    assert offenses[0].consumer_kind == "if" and offenses[1].line is None
    assert len(offenses) == worker.MAX_OFFENSES - 2  # malformed entries are dropped
    assert worker._offenses(ValueError("plain")) == ()
    assert to_dict(only_row(worker._failed(
        RunRequest(request="t", id="m8839", mode="trace", recipe=recipe("toys.Plain", (1, 3)),
                   caps=Caps(rss_mb=8192, timeout_s=1800, threads=2)),
        worker.Failure("capture_error", "x")))).get("offenses") is None


def test_batch_norm_reads_as_statistical_in_a_fresh_worker(tmp_path: Path) -> None:
    """Batch statistics change values only; the batch counter's in-place update is not graph.

    Run as the runner runs it, in a fresh process: earlier tests in this process change global
    torch state that the op recorder sees.
    """

    import subprocess
    import sys

    pytest.importorskip("timm")
    request = RunRequest(request="t-bn", id="m8839", mode="forward", recipe=timm_recipe("resnet18"),
                         caps=Caps(rss_mb=8192, timeout_s=600, threads=2))
    write_json(request, tmp_path / "request.json")
    subprocess.run([sys.executable, "-m", "menagerie.pipeline", "worker", str(tmp_path)],
                   check=False, timeout=600, cwd=Path(__file__).parents[2])
    result = read_json(RunResult, tmp_path / "result.json")
    assert result.modes is not None and result.modes.divergence == "statistical"



class Wide(nn.Module):
    """One megabyte of output: over a one-megabyte attempt's half-cap save budget."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(27, 2**18)

    def forward(self, x: Any) -> Any:
        return self.lin(x.flatten(1))


def test_gpu_captures_save_to_host_under_the_declared_budget() -> None:
    from torchlens.options import CaptureOptions

    assert worker.gpu_save_options("cpu", 8192) == {}
    options = worker.gpu_save_options("cuda", 8192)
    assert options == {"output_device": "cpu", "save_budget": 4 * 2**30}
    x = torch.randn(1, 3, 3, 3)
    torch.manual_seed(0)
    cold = trace_summary.hash_family(tl.trace(Plain().eval(), x, capture=CaptureOptions(
        inference_only=True, random_seed=worker.SEED)))
    torch.manual_seed(0)
    host = trace_summary.hash_family(tl.trace(Plain().eval(), x, capture=CaptureOptions(
        inference_only=True, random_seed=worker.SEED, **options)))
    assert host == cold  # where the saves live never enters the hashes
    trace = tl.trace(Plain().eval(), x, capture=CaptureOptions(
        intervention_ready=True, random_seed=worker.SEED, **options))  # the save capture's set
    assert trace.num_ops > 0
    with pytest.raises(worker.Failure) as caught, worker._stage("capture_error"):
        tl.trace(Wide().eval(), x, capture=CaptureOptions(**worker.gpu_save_options("cuda", 1)))
    assert caught.value.cls == "capture_error"
    assert caught.value.msg.startswith("SaveBudgetExceededError")
    assert "budget 524288 on cpu" in caught.value.msg


_STALE_ALIAS = """
import sys, types, torch
from menagerie.pipeline import worker
from torchlens.user_funcs import _validate_forward_pass_torch

original = torch.bmm
worker._wrap_torch()
# The model's module is imported after the pre-wrap, as harvest.construct imports it in _build.
module = types.ModuleType("stale_alias")
exec(
    "import torch\\nfrom torch import bmm\\n"
    "class Model(torch.nn.Module):\\n"
    "    def __init__(self):\\n"
    "        super().__init__()\\n"
    "        self.proj = torch.nn.Linear(4, 4)\\n"
    "    def forward(self, h):\\n"
    "        h = self.proj(h)\\n"
    "        return bmm(h, h.transpose(1, 2)).relu().sum(-1)\\n",
    module.__dict__)
sys.modules["stale_alias"] = module
torch.manual_seed(0)
model = module.Model().eval()
passed = _validate_forward_pass_torch(model, [torch.randn(1, 3, 4)], {}, random_seed=0)
print(module.bmm is not original, bool(passed))
"""


def test_a_torch_alias_bound_at_import_is_the_wrapped_function() -> None:
    """m19908 (``from torch import bmm``) and m20502 (``cat``) failed validation at TorchLens
    2.36.0 once the pre-wrap was skipped: the validator does not rescue a stale alias."""

    pytest.importorskip("torchlens")
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-c", _STALE_ALIAS], capture_output=True, text=True,
                          check=True, timeout=300)
    assert done.stdout.strip().splitlines()[-1] == "True True"
