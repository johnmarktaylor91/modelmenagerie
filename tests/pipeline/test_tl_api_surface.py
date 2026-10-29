"""TorchLens API probe at the pin: every call the worker and renderer use, live and loaded.

Re-run whenever the TorchLens pin moves. Each call prints one ``PROBE {json}`` line (run with
``-s``). Calls the pipeline depends on fail the test when they raise; the open questions
(whether a loaded trace validates, draws and exports) are recorded and asserted only in the
summary line, because their answer decides whether ``render`` is its own runner mode.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("torchlens")
timm = pytest.importorskip("timm")
from torch import nn  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402

PIN = "2.36.0"
ANSWERS: dict[str, bool] = {}


class TwoLayerLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(8, 16, num_layers=2, batch_first=True)
        self.head = nn.Linear(16, 4)

    def forward(self, x: Any) -> Any:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


BUILDERS: dict[str, tuple[Callable[[], Any], tuple[int, ...]]] = {
    "resnet18": (lambda: timm.create_model("resnet18", pretrained=False), (1, 3, 224, 224)),
    "lstm2": (TwoLayerLSTM, (2, 5, 8)),
}


def _brief(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value if not isinstance(value, str) else value[:240]
    if hasattr(value, "state"):
        fields = ("state", "reason", "replayed_node_count", "unverified_node_count",
                  "failed_node_count", "message")
        return {k: str(getattr(value, k, None))[:160] for k in fields}
    if isinstance(value, Path):
        return {"path": value.name, "bytes": value.stat().st_size if value.exists() else None}
    return type(value).__name__


def probe(model: str, call: str, which: str, fn: Callable[[], Any], *,
          required: bool = True) -> Any:
    """Run one call, print its PROBE line, and re-raise a failure when the call is required."""

    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value, error = fn(), None
        except Exception as exc:  # noqa: BLE001 -- the probe records every failure type
            first = (str(exc).splitlines() or [""])[0][:200]
            value, error = None, f"{type(exc).__name__}: {first}"
    line = {
        "model": model, "call": call, "trace": which, "ok": error is None,
        "secs": round(time.perf_counter() - started, 3), "value": _brief(value), "error": error,
        "warnings": sorted({str(w.message).splitlines()[0][:100] for w in caught})[:3],
    }
    print("PROBE", json.dumps(line), flush=True)
    ANSWERS[f"{model}:{call}:{which}"] = error is None
    if error and required:
        pytest.fail(f"{model} {call} ({which}): {error}")
    return value


class Captured:
    """One model's live trace, ground truth and saved-and-loaded runnable trace."""

    def __init__(self, name: str, root: Path) -> None:
        build, shape = BUILDERS[name]
        torch.manual_seed(0)
        self.name, self.root, self.model = name, root, build().eval()
        self.x = torch.randn(*shape)
        with torch.no_grad():
            self.truth = self.model(self.x)
        options = CaptureOptions(intervention_ready=True)
        self.live = probe(name, "trace(intervention_ready=True)", "live",
                          lambda: tl.trace(self.model, self.x, capture=options))
        self.path = root / f"{name}.tlspec"
        probe(name, "save(level=runnable)", "live",
              lambda: tl.save(self.live, self.path, level="runnable"))
        self.loaded = probe(name, "load", "loaded", lambda: tl.load(self.path))

    def trace(self, which: str) -> Any:
        return self.live if which == "live" else self.loaded


@pytest.fixture(scope="module", params=sorted(BUILDERS))
def captured(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Captured:
    return Captured(request.param, tmp_path_factory.mktemp(request.param))


def test_pin_is_installed() -> None:
    assert tl.__version__ == PIN, f"probe is for {PIN}, installed {tl.__version__}"


def _why_not_valid() -> str:
    return str(tl.validation.last_validation_failure())[:240]


def test_plain_capture_is_refused_runnable(captured: Captured) -> None:
    plain = probe(captured.name, "trace(plain)", "live",
                  lambda: tl.trace(captured.model, captured.x))
    probe(captured.name, "validate_forward_pass(plain capture)", "live",
          lambda: plain.validate_forward_pass(captured.truth), required=False)
    probe(captured.name, "last_validation_failure", "live", _why_not_valid, required=False)
    target = captured.root / "plain.tlspec"
    probe(captured.name, "save(plain, level=runnable) refuses", "live",
          lambda: tl.save(plain, target, level="runnable"), required=False)
    assert not ANSWERS[f"{captured.name}:save(plain, level=runnable) refuses:live"], (
        "a capture without intervention_ready saved runnable; the preflight contract changed"
    )


def test_validate_forward_pass_and_scope(captured: Captured) -> None:
    status = probe(captured.name, "validate_forward_pass", "live",
                   lambda: captured.live.validate_forward_pass(captured.truth))
    assert status is not None
    probe(captured.name, "last_validation_failure", "live", _why_not_valid, required=False)
    probe(captured.name, "validate_forward_pass", "loaded",
          lambda: captured.loaded.validate_forward_pass(captured.truth), required=False)
    verdict = probe(captured.name, "validate(scope=forward)", "model",
                    lambda: tl.validate(captured.model, captured.x, scope="forward"))
    assert isinstance(verdict, bool)


def test_hash_round_trip(captured: Captured) -> None:
    live = probe(captured.name, "hash.trace", "live", lambda: tl.hash.trace(captured.live))
    loaded = probe(captured.name, "hash.trace", "loaded", lambda: tl.hash.trace(captured.loaded))
    assert live == loaded and len(live) == 64


def test_loaded_runnable_runs(captured: Captured) -> None:
    result = probe(captured.name, "run(inputs, seed)", "loaded",
                   lambda: captured.loaded.run(inputs=captured.x, seed=0))
    rerun_hash = probe(captured.name, "hash.trace(run result)", "loaded",
                       lambda: tl.hash.trace(result.trace))
    assert rerun_hash == tl.hash.trace(captured.live)
    probe(captured.name, "run report", "loaded", lambda: str(result.report), required=False)


@pytest.mark.parametrize("which", ["live", "loaded"])
def test_draw_and_exports(captured: Captured, which: str) -> None:
    trace, out = captured.trace(which), captured.root / which
    out.mkdir(exist_ok=True)
    required = which == "live"
    views = [("unrolled", "none"), ("unrolled", "auto")]
    if captured.name == "lstm2":
        views.append(("rolled", "none"))
    for vis_mode, collapse in views:
        stem = out / f"{vis_mode}-{collapse}"
        probe(captured.name, f"draw({vis_mode},{collapse},svg)", which, lambda: trace.draw(
            vis_mode=vis_mode, collapse=collapse, vis_outpath=str(stem), vis_fileformat="svg",
            vis_save_only=True), required=required)
    probe(captured.name, "draw(pdf)", which, lambda: trace.draw(
        vis_outpath=str(out / "pdf"), vis_fileformat="pdf", vis_save_only=True), required=required)
    for kind, suffix in (("html", "html"), ("netron", "json"), ("model_explorer", "json")):
        exporter = getattr(tl.export, kind)
        path = probe(captured.name, f"export.{kind}", which,
                     lambda: exporter(trace, out / f"{kind}.{suffix}"), required=required)
        if path is not None:
            assert Path(path).stat().st_size > 0
    if which == "live":
        written = sorted(p.name for p in out.iterdir())
        assert any(n.endswith(".svg") for n in written), written


def test_facts_for_fixtures(captured: Captured) -> None:
    """Print the values the fixture rows are built from: op count, hashes and written files."""

    from menagerie.pipeline import tree_sha256

    facts: dict[str, Any] = {
        "model": captured.name, "call": "FACTS", "num_ops": getattr(captured.live, "num_ops", None),
        "loaded_num_ops": getattr(captured.loaded, "num_ops", None),
        "skeleton": tl.hash.trace(captured.live), "tlspec_sha256": tree_sha256(captured.path),
        "tlspec_files": sum(1 for p in captured.path.rglob("*") if p.is_file()),
        "params": sum(p.numel() for p in captured.model.parameters()),
    }
    for which in ("live", "loaded"):
        for item in sorted((captured.root / which).iterdir()):
            digest = hashlib.sha256(item.read_bytes()).hexdigest()
            facts[f"{which}/{item.name}"] = [digest, item.stat().st_size]
    print("PROBE", json.dumps(facts))
    for name in ("unrolled-none.svg", "html.html"):
        live, loaded = ((captured.root / w / name).read_text().splitlines()
                        for w in ("live", "loaded"))
        changed = [line[:160] for line in difflib.unified_diff(live, loaded, n=0, lineterm="")
                   if line[:1] in "+-" and line[:3] not in ("+++", "---")]
        labels = [line for line in changed if "<text" in line]
        print("PROBE", json.dumps({"model": captured.name, "call": f"diff live vs loaded {name}",
                                   "changed_lines": len(changed), "first": changed[:6],
                                   "labels": labels[:4] + labels[len(labels) // 2:][:4]}))


@pytest.mark.parametrize("include_outs", [False, True])
def test_portable_save_draws_like_live(captured: Captured, include_outs: bool) -> None:
    """Informational: does a portable-level save redraw the live diagram byte for byte?"""

    from menagerie.pipeline import tree_sha256

    path = captured.root / f"portable-{include_outs}.tlspec"
    tl.save(captured.live, path, level="portable", include_outs=include_outs)
    stem = captured.root / f"portable-{include_outs}"
    tl.load(path).draw(vis_mode="unrolled", collapse="none", vis_outpath=str(stem),
                       vis_fileformat="svg", vis_save_only=True)
    live = (captured.root / "live" / "unrolled-none.svg").read_bytes()
    redrawn = Path(f"{stem}.svg").read_bytes()
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print("PROBE", json.dumps({"model": captured.name, "call": "portable save redraw",
                               "include_outs": include_outs, "svg_identical": live == redrawn,
                               "tlspec_bytes": size, "tlspec_sha256": tree_sha256(path)}))


def test_capture_overhead_of_intervention_ready() -> None:
    """Informational: alternate plain and intervention-ready captures of resnet18, three each."""

    build, shape = BUILDERS["resnet18"]
    torch.manual_seed(0)
    model, x = build().eval(), torch.randn(*shape)
    tl.trace(model, x)  # warm-up: the first capture of a session wraps torch once
    times: dict[str, list[float]] = {"plain": [], "intervention_ready": []}
    for _ in range(3):
        for kind, capture in (("plain", None), ("intervention_ready", CaptureOptions(
                intervention_ready=True))):
            started = time.perf_counter()
            tl.trace(model, x, capture=capture)
            times[kind].append(round(time.perf_counter() - started, 3))
    ratio = sorted(times["intervention_ready"])[1] / sorted(times["plain"])[1]
    print("PROBE", json.dumps({"model": "resnet18", "call": "capture overhead", "secs": times,
                               "median_ratio": round(ratio, 3)}))


def test_structure_only_on_meta(tmp_path: Path) -> None:
    for name, (build, shape) in sorted(BUILDERS.items()):
        with torch.device("meta"):
            model = probe(name, "build on meta", "meta", build)
            x = torch.randn(*shape)
        trace = probe(name, "trace(structure_only=True)", "meta", lambda: tl.trace(
            model, x, capture=CaptureOptions(structure_only=True)))
        probe(name, "hash.trace", "meta", lambda: tl.hash.trace(trace))
        probe(name, "draw(svg)", "meta", lambda: trace.draw(
            vis_outpath=str(tmp_path / name), vis_fileformat="svg", vis_save_only=True))
        params = sum(p.numel() for p in model.parameters())
        print("PROBE", json.dumps({"model": name, "call": "geometry params", "value": params}))
        assert params > 0


def test_zz_answer_loaded_render_question(captured: Captured) -> None:
    """Print whether a loaded trace drew and exported: the answer that sets render's mode."""

    keys = [k for k in ANSWERS if k.startswith(f"{captured.name}:") and k.endswith(":loaded")
            and (":draw(" in k or ":export." in k)]
    answer = bool(keys) and all(ANSWERS[k] for k in keys)
    failed = sorted(k for k in keys if not ANSWERS[k])
    same = {name: (captured.root / "live" / name).read_bytes()
            == (captured.root / "loaded" / name).read_bytes()
            for name in ("unrolled-none.svg", "netron.json", "model_explorer.json")}
    print("PROBE", json.dumps({"model": captured.name, "call": "ANSWER loaded draws and exports",
                               "value": answer, "failed": failed, "identical_to_live": same}))
    assert keys, "no loaded draw or export call ran"
