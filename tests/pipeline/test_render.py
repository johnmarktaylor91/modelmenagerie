"""Render tests on real models: variants, artifacts, viewers, tiles and the June collapse set.

The quick tests run in the inner tier. Setting ``MENAGERIE_RENDER_OUT`` to a directory also runs
the acceptance set: ten models captured, saved at the ``portable`` level without outputs, loaded
and rendered the way the runner's ``render`` mode does, with every artifact checked and one line
per model appended to ``table.jsonl`` there (op counts, bytes and seconds per artifact).
``MENAGERIE_CHROMIUM`` names a Chromium binary for the HTML check (default: Playwright's own).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
tl = pytest.importorskip("torchlens")
pytest.importorskip("pyvips")
from PIL import Image  # noqa: E402

from menagerie.pipeline import render, tree_sha256  # noqa: E402
from menagerie.pipeline.records import RenderRow, dumps, from_dict, to_dict  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = os.environ.get("MENAGERIE_RENDER_OUT")
acceptance = pytest.mark.skipif(not OUT, reason="MENAGERIE_RENDER_OUT is not set")
Built = tuple[Any, Any]


def _catalog(name: str) -> Callable[[], Built]:
    """Build a ``master_catalog.jsonl`` statement recipe (the smoke manifest's own rows)."""

    def build() -> Built:
        with (REPO / "menagerie/data/master_catalog.jsonl").open() as rows:
            row = next(r for r in map(json.loads, rows) if r["name"] == name)
        scope: dict[str, Any] = {}
        exec(row["recipe"]["code"], scope)  # noqa: S102 -- the repo's own catalog recipe
        return scope[row["recipe"].get("output_name", "model")], torch.randn(
            *row["input"]["spec"]["shape"])

    return build


def _classic(name: str) -> Callable[[], Built]:
    def build() -> Built:
        import importlib

        module = importlib.import_module(f"menagerie.classics.{name}")
        return module.build(), module.example_input()

    return build


def _torchvision(name: str) -> Callable[[], Built]:
    def build() -> Built:
        import torchvision

        return getattr(torchvision.models, name)(weights=None), torch.randn(1, 3, 224, 224)

    return build


def _resnet18() -> Built:
    import timm

    return timm.create_model("resnet18", pretrained=False), torch.randn(1, 3, 224, 224)


def _gpt2() -> Built:
    from transformers import GPT2Config, GPT2Model

    return GPT2Model(GPT2Config()), torch.randint(0, 50257, (1, 16))


BUILDERS: dict[str, Callable[[], Built]] = {
    "resnet18": _resnet18,
    "lenet5_classic": _classic("lenet5_classic"),
    "orig_lstm_1997": _classic("orig_lstm_1997"),
    "m413_ConvLSTM": _catalog("ConvLSTM"),
    "m1481_Kandinsky3UNet": _catalog("Kandinsky3UNet"),
    "mobilenet_v3_small": _torchvision("mobilenet_v3_small"),
    "vit_b_16": _torchvision("vit_b_16"),
    "densenet121": _torchvision("densenet121"),
    "maxvit_t": _torchvision("maxvit_t"),
    "gpt2": _gpt2,
}


def capture(name: str, tmp: Path) -> tuple[Any, Any, dict[str, float]]:
    """Return the live trace, the trace loaded from its portable save, and the timings."""

    torch.manual_seed(0)
    model, x = BUILDERS[name]()
    model.eval()
    started = time.monotonic()
    live = tl.trace(model, x)
    traced = time.monotonic()
    path = tmp / f"{name}.tlspec"
    tl.save(live, path, level="portable", include_outs=False)
    loaded = tl.load(path)
    times = {"capture_secs": round(traced - started, 2),
             "save_load_secs": round(time.monotonic() - traced, 2),
             "tlspec_bytes": sum(p.stat().st_size for p in path.rglob("*") if p.is_file())}
    return live, loaded, times


def check_artifacts(row: RenderRow, files: list[Any], out: Path) -> None:
    """Every artifact in the row exists with its sha and size, and decodes."""

    assert from_dict(RenderRow, json.loads(dumps(row))) == row
    on_disk = {f.sha256: out / f.path for f in files}
    for item in row.renders:
        path = on_disk[item.sha256]
        assert tree_sha256(path) == item.sha256
        if item.kind == "svg":
            assert path.read_text().lstrip().startswith(("<?xml", "<svg"))
        elif item.kind == "pdf":
            assert path.read_bytes()[:5] == b"%PDF-"
        elif item.kind == "thumb":
            with Image.open(path) as image:
                assert image.format == "WEBP"
                assert image.width <= render.THUMB_BOX[0] and image.height <= render.THUMB_BOX[1]
    assert (out / "render.log").exists() and row.evidence == tree_sha256(out / "render.log")


def drawn_variants(row: RenderRow) -> set[str]:
    return {r.variant for r in row.renders if r.kind == "svg"}


def console_errors(html: Path) -> tuple[list[str], int]:
    """Open the HTML viewer in headless Chromium; return console errors and drawn elements."""

    sync_api = pytest.importorskip("playwright.sync_api")
    errors: list[str] = []
    with sync_api.sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=os.environ.get("MENAGERIE_CHROMIUM"))
        page = browser.new_page()
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
        page.set_content(html.read_text(), wait_until="load")
        page.wait_for_timeout(1500)
        drawn = page.locator("svg, canvas").count()
        browser.close()
    return errors, drawn


def test_renderer_rev_names_graphviz_vips_and_options() -> None:
    import re

    assert re.fullmatch(r"graphviz \S+\+pyvips \S+\+libvips \S+\+render-opts@[0-9a-f]{8}",
                        render.renderer_rev())


def test_resnet18_renders_every_artifact_and_viewers_parse(tmp_path: Path) -> None:
    live, loaded, _ = capture("resnet18", tmp_path)
    assert render.variants_for(loaded) == ["unrolled-none", "unrolled-auto", "unrolled-max"]
    out = tmp_path / "loaded"
    row, files = render.render_set(loaded, model_id="m8839", trace_sha="0" * 64, out_dir=out)
    assert row.outcome == "pass", [to_dict(s) for s in row.skipped]
    assert drawn_variants(row) == {"unrolled-none", "unrolled-auto", "unrolled-max"}
    check_artifacts(row, files, out)
    kinds = {(r.variant, r.kind) for r in row.renders}
    assert {(v, k) for v in drawn_variants(row) for k in ("svg", "pdf", "thumb")} <= kinds
    assert {(None, "html"), (None, "netron"), (None, "model_explorer")} <= kinds

    live_row, _ = render.render_set(live, model_id="m8839", trace_sha="0" * 64,
                                    out_dir=tmp_path / "live", variants=["unrolled-none"])
    assert live_row.renders[0].sha256 == row.renders[0].sha256, "loaded SVG differs from live"

    onnx = pytest.importorskip("onnx")
    from google.protobuf import json_format

    model = json_format.Parse((out / "netron.json").read_text(), onnx.ModelProto())
    assert len(model.graph.node) > 0
    explorer = json.loads((out / "model_explorer.json").read_text())
    assert explorer["graphs"] and explorer["graphs"][0]["nodes"]
    errors, drawn = console_errors(out / "interactive.html")
    assert errors == [] and drawn > 0, (errors, drawn)


def test_recurrent_classic_gets_rolled_default(tmp_path: Path) -> None:
    _, loaded, _ = capture("orig_lstm_1997", tmp_path)
    assert loaded.is_recurrent
    keys = render.variants_for(loaded)
    assert "rolled-none" in keys
    row, files = render.render_set(loaded, model_id="m0001", trace_sha="1" * 64,
                                   out_dir=tmp_path / "out")
    assert row.outcome == "pass", [to_dict(s) for s in row.skipped]
    check_artifacts(row, files, tmp_path / "out")
    accounted = drawn_variants(row) | {s.variant for s in row.skipped if s.kind == "svg"}
    assert accounted == set(keys)
    assert render.default_variant(sorted(drawn_variants(row)), is_recurrent=True,
                                  n_ops=loaded.num_ops) == "rolled-none"


def test_identical_render_is_deduplicated(tmp_path: Path) -> None:
    _, loaded, _ = capture("lenet5_classic", tmp_path)
    assert loaded.num_ops <= render.READABLE_OPS
    assert render.variants_for(loaded) == ["unrolled-none"]
    row, _ = render.render_set(loaded, model_id="m0002", trace_sha="2" * 64,
                               out_dir=tmp_path / "out", variants=["unrolled-none", "unrolled-auto"])
    assert row.outcome == "pass"
    assert drawn_variants(row) == {"unrolled-none"}
    assert [(s.variant, s.why) for s in row.skipped] == [
        ("unrolled-auto", "render:same-as-unrolled-none")]


def test_graphviz_failure_and_missing_graphviz_are_typed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    _, loaded, _ = capture("lenet5_classic", tmp_path)
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "dot").write_text(f"""#!/bin/sh
case "$*" in *-Tpdf*) echo "simulated pdf failure" >&2; exit 3;; esac
exec {shutil.which("dot")} "$@"
""")
    (shim / "dot").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")
    row, _ = render.render_set(loaded, model_id="m0002", trace_sha="2" * 64,
                               out_dir=tmp_path / "nopdf")
    assert row.outcome == "partial"
    assert [(s.variant, s.kind, s.why) for s in row.skipped] == [
        ("unrolled-none", "pdf", "render:draw")]

    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    render._graphviz_version.cache_clear()
    try:
        row, files = render.render_set(loaded, model_id="m0002", trace_sha="2" * 64,
                                       out_dir=tmp_path / "nodot")
    finally:
        render._graphviz_version.cache_clear()
    assert (row.outcome, row.renders, row.renderer_rev) == ("infra", (), "unavailable")
    assert [f.kind for f in files] == ["log"]


QUALIFIED = ("unrolled-none", "unrolled-auto", "unrolled-max")
RECURRENT = ("rolled-none", "rolled-auto", "rolled-max")


@acceptance
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_acceptance_model(name: str, tmp_path: Path) -> None:
    """Render the full matrix from a loaded portable save, check it, and record the table row."""

    root = Path(str(OUT))
    out = root / name
    _, loaded, times = capture(name, tmp_path)
    recurrent = bool(loaded.is_recurrent)
    requested = list(QUALIFIED + (RECURRENT if render.reuses_weights(loaded) else ()))
    row, files = render.render_set(loaded, model_id="m0003", trace_sha="3" * 64, out_dir=out,
                                   variants=requested)
    (out / "row.json").write_text(dumps(row) + "\n")
    check_artifacts(row, files, out)
    drawn = drawn_variants(row)
    deduped = {s.variant: s.why for s in row.skipped if s.why.startswith("render:same-as-")}
    assert set(render.variants_for(loaded)) <= drawn | set(deduped)
    assert row.outcome == "pass", [to_dict(s) for s in row.skipped]
    if name in ("m413_ConvLSTM", "orig_lstm_1997"):
        assert "rolled-none" in drawn
    tiles = [r for r in row.renders if r.kind == "tiles"]
    for item in tiles:
        tile_dir = out / f"{item.variant}.tiles" / f"{item.variant}_files"
        deepest = max((p for p in tile_dir.iterdir() if p.is_dir()), key=lambda p: int(p.name))
        with Image.open(min(deepest.iterdir())) as tile:
            tile.load()
            assert tile.size[0] > 0
    log = [json.loads(line) for line in (out / "render.log").read_text().splitlines()]
    line = {"model": name, "n_ops": loaded.num_ops, "num_layers": loaded.num_layers,
            "is_recurrent": recurrent,
            "reuses_weights": render.reuses_weights(loaded),
            "rolled_units": render.rolled_units(loaded), **times, "render_secs": row.secs, "outcome": row.outcome,
            "renderer_rev": row.renderer_rev, "qualified": render.variants_for(loaded),
            "drawn": sorted(drawn), "deduped": deduped,
            "default": render.default_variant(sorted(drawn), is_recurrent=recurrent,
                                              n_ops=loaded.num_ops),
            "artifacts": [{k: e.get(k) for k in ("variant", "kind", "bytes", "secs", "why")}
                          for e in log],
            "warnings": sorted({w for e in log for w in e.get("warnings", [])})}
    with (root / "table.jsonl").open("a") as table:
        table.write(json.dumps(line) + "\n")
    if name == "m1481_Kandinsky3UNet":
        assert tiles, "the large-graph row made no tiles"


class _SlowDot:
    """A trace whose dot layout times out the way TorchLens reports it (its fixed 120 s)."""

    num_ops = 1393

    def __init__(self, slow: tuple[str, ...]) -> None:
        self.slow, self.placements = slow, []

    def draw(self, *, vis_outpath: str, vis_fileformat: str, vis_node_placement: str,
             **_: Any) -> None:
        import subprocess

        self.placements.append(vis_node_placement)
        if vis_node_placement in self.slow:
            try:
                raise subprocess.TimeoutExpired(["dot"], 120)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Graphviz render timed out after 120s") from exc
        Path(f"{vis_outpath}.{vis_fileformat}").write_text("<svg/>")


def test_a_dot_timeout_is_drawn_again_with_rank_placement(tmp_path: Path) -> None:
    """RGN m14408 (1,393 nodes) timed out in dot at TorchLens's 120 s and drew in rank placement
    in about a second; a rank timeout stays a typed skip."""

    trace, log = _SlowDot(("auto",)), {}
    render._draw(trace, "unrolled-none", tmp_path / "unrolled-none", "svg", log)
    assert trace.placements == ["auto", "rank"]
    assert (log["placement"], log["timed_out"]) == ("rank", "auto")
    assert (tmp_path / "unrolled-none.svg").exists()
    stuck = _SlowDot(("auto", "rank"))
    with pytest.raises(render._Skip) as skip:
        render._draw(stuck, "unrolled-auto", tmp_path / "unrolled-auto", "svg", {})
    assert skip.value.why == "render:timeout" and stuck.placements == ["auto", "rank"]
