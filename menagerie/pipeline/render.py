"""Rendering from one trace: diagram variants, PDF, thumbnail, DZI tiles and three exports.

Owned by the render lane. ``trace`` is a live or loaded TorchLens ``Trace``; a loaded trace
draws like the live one when the worker saved it at the ``portable`` level without outputs
(step zero probe), so ``render`` can run as its own runner mode on that save. A render failure
is a render row; it never touches the trace verdict.

Per variant: TorchLens draws the SVG and the PDF through Graphviz with the June options (its own
120 s bound applies to each); libvips rasterises the SVG into the WebP thumbnail and, for a
monster, the DZI tile pyramid. Identical SVGs are deduplicated: a variant whose SVG equals an
earlier one is skipped as ``render:same-as-<variant>``. Output paths are relative to
``out_dir``; ``render.log`` (the row's ``evidence``) holds one JSON line per artifact with its
seconds and the warnings TorchLens raised.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import warnings
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from menagerie.pipeline import NON_TRACE_ROOTS, now, tree_sha256
from menagerie.pipeline.records import (
    VARIANT,
    OutputFile,
    RenderArtifact,
    RenderRow,
    RenderSkip,
)

# The June draw options (``generate_menagerie.py`` ``render_one``); the variant adds the view.
DRAW_OPTIONS: dict[str, Any] = {"vis_save_only": True, "show_legend": False}
# The renderer's packages. They are in the ``torch-main`` lock as declared non-trace roots, so
# recipe revisions leave them out and ``renderer_rev`` carries their versions instead: a renderer
# change re-renders without retiring a trace. pyvips's binary wheel bundles libvips.
EXTRAS = NON_TRACE_ROOTS
# Collapsed variants qualify above TorchLens's readable band for small graphs (40 visible
# nodes, ``auto_collapse._readable_band_high``); measured at rung 0 (lane-render REPORT.md).
READABLE_OPS = 40
# The monster predicate (DESIGN section 5): tiles when an SVG has over 2,000 nodes or 5 MB. An
# uncollapsed view over 2,000 units uses TorchLens's rank placement: Graphviz dot timed out
# (120 s) on the 2,841-op large-graph smoke row m1481 (lane-render REPORT.md).
MONSTER_NODES = 2000
MONSTER_SVG_BYTES = 5_000_000
ROLLED_FRACTION = 0.75  # see ``variants_for``
THUMB_BOX = (400, 2400)  # June width 400 (site/scripts/build_catalog.py), height capped
THUMB_QUALITY = 78  # June cwebp quality
TILE_SIZE, TILE_OVERLAP = 254, 1
TILE_MAX_PIXELS = 400_000_000  # the base level is one pixel per point, scaled down to this
# The HTML viewer carries no embedded tensor values (``share_safe``) and no timestamp
# (``deterministic``); its op timings, like the Netron and Model Explorer files', still differ
# per capture, and so do PDF creation dates. SVGs and thumbnails are byte-stable.
EXPORTS = (("html", "interactive.html"), ("netron", "netron.json"),
           ("model_explorer", "model_explorer.json"))


class _Skip(Exception):
    """One artifact could not be made; ``why`` is the ``render:<detail>`` reason."""

    def __init__(self, why: str, detail: str = "") -> None:
        super().__init__(detail or why)
        self.why = why


@cache
def _graphviz_version() -> str:
    proc = subprocess.run(["dot", "-V"], capture_output=True, text=True, timeout=30, check=True)
    found = re.search(r"version (\S+)", proc.stderr + proc.stdout)
    if not found:
        raise RuntimeError(f"cannot read the Graphviz version from {proc.stderr!r}")
    return found.group(1)


def _render_options() -> dict[str, Any]:
    return {
        "draw": DRAW_OPTIONS, "rank_above": MONSTER_NODES, "html": "deterministic+share_safe",
        "readable_ops": READABLE_OPS, "rolled": ROLLED_FRACTION,
        "monster": [MONSTER_NODES, MONSTER_SVG_BYTES], "thumb": [*THUMB_BOX, THUMB_QUALITY],
        "tiles": [TILE_SIZE, TILE_OVERLAP, TILE_MAX_PIXELS, "webp"],
    }


def _renderer_packages() -> dict[str, str]:
    """Installed versions of the renderer's packages (``EXTRAS``) and of the libvips they load."""

    from importlib.metadata import version

    import pyvips

    found = {name: version(name) for name in EXTRAS}
    found["libvips"] = ".".join(str(pyvips.version(i)) for i in range(3))
    return found


def renderer_rev() -> str:
    """Return ``graphviz <v>+pyvips <v>+libvips <v>+render-opts@<8 hex>``.

    The hash covers the draw options and every renderer package's version; any change re-arms
    rendering and never a trace.
    """

    packages = _renderer_packages()
    blob = json.dumps({**_render_options(), "packages": packages}, sort_keys=True).encode()
    return (f"graphviz {_graphviz_version()}+pyvips {packages['pyvips']}"
            f"+libvips {packages['libvips']}+render-opts@{hashlib.sha256(blob).hexdigest()[:8]}")


def rolled_units(trace: Any) -> int:
    """Return the op count of the rolled view: each multi-pass layer drawn once."""

    return int(trace.num_ops) - sum(n - 1 for n in trace.layer_num_calls.values() if n > 1)


def reuses_weights(trace: Any) -> bool:
    """Whether a layer with parameters ran more than once: TorchLens's true recurrence."""

    return any(layer.num_passes > 1 and layer.uses_params for layer in trace.layer_logs.values())


def variants_for(trace: Any, *, train: bool = False) -> list[str]:
    """Return the variant keys a trace qualifies for (``unrolled-none`` always).

    Rolled views qualify when the same parameters run more than once and rolling hides at least
    a quarter of the ops. ``is_recurrent`` alone also counts a parameterless module called more
    than once (the 2,841-op Kandinsky U-Net qualified that way), whose rolled view is a tangle.
    """

    keys = ["unrolled-none"]
    if trace.num_ops > READABLE_OPS:
        keys += ["unrolled-auto", "unrolled-max"]
    rolled = rolled_units(trace)
    if reuses_weights(trace) and rolled <= ROLLED_FRACTION * trace.num_ops:
        keys.append("rolled-none")
        if rolled > READABLE_OPS:
            keys += ["rolled-auto", "rolled-max"]
    return [f"{key}-train" if train else key for key in keys]


def default_variant(variants: Sequence[str], *, is_recurrent: bool, n_ops: int) -> str:
    """Return the default variant: rolled if recurrent, collapsed above the readable band."""

    preferred = (["rolled-none"] if is_recurrent else []) + (
        ["unrolled-auto"] if n_ops > READABLE_OPS else []) + ["unrolled-none"]
    for key in preferred:
        if key in variants:
            return key
    eval_variants = [v for v in variants if not v.endswith("-train")]
    return (eval_variants or list(variants))[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _draw(trace: Any, key: str, stem: Path, fmt: str, log: dict[str, Any]) -> None:
    """Draw one variant to ``<stem>.<fmt>`` through TorchLens with the June options."""

    match = re.fullmatch(VARIANT, key)
    if match is None:
        raise ValueError(f"not a variant key: {key!r}")
    units = rolled_units(trace) if match[1] == "rolled" else trace.num_ops
    placement = "rank" if match[2] == "none" and units > MONSTER_NODES else "auto"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            try:
                _draw_at(trace, match, stem, fmt, placement, log)
            except _Timeout:
                if placement == "rank":
                    raise
                # dot can pass TorchLens's fixed 120 s well under MONSTER_NODES (RGN m14408,
                # 1,393 nodes, at 2.35.3) while rank placement draws it in seconds. Only drawings
                # that would have failed change, so the fallback is not in the renderer's hash.
                log["timed_out"] = "auto"
                _draw_at(trace, match, stem, fmt, "rank", log)
        except Exception as exc:  # every draw failure becomes a typed skip
            stem.unlink(missing_ok=True)  # TorchLens leaves the DOT source at the stem
            error = exc.error if isinstance(exc, _Timeout) else exc
            timed_out = isinstance(error.__cause__, subprocess.TimeoutExpired)
            raise _Skip("render:timeout" if timed_out else "render:draw",
                        f"{type(error).__name__}: {error}") from error
        finally:
            log["warnings"] = _first_lines(caught)


class _Timeout(Exception):
    """A draw that Graphviz timed out, carrying TorchLens's error."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


def _draw_at(trace: Any, match: re.Match[str], stem: Path, fmt: str, placement: str,
             log: dict[str, Any]) -> None:
    log["placement"] = placement
    try:
        trace.draw(vis_mode=match[1], collapse=match[2], vis_outpath=str(stem),
                   vis_fileformat=fmt, vis_node_placement=placement, **DRAW_OPTIONS)
    except Exception as exc:
        if isinstance(exc.__cause__, subprocess.TimeoutExpired):
            stem.unlink(missing_ok=True)
            raise _Timeout(exc) from exc
        raise


def _first_lines(caught: list[warnings.WarningMessage]) -> list[str]:
    return sorted({(str(w.message).splitlines() or [""])[0][:200] for w in caught})


def _svg_points(svg: Path) -> tuple[float, float]:
    """Return the SVG's width and height in points (one pixel each at libvips's 72 dpi)."""

    size = re.search(r'width="([\d.]+)pt" height="([\d.]+)pt"', svg.read_text()[:4000])
    if size is None:
        raise _Skip("render:svg-size", "no width and height in points in the SVG header")
    return float(size[1]), float(size[2])


def _thumbnail(svg: Path, target: Path) -> None:
    """Rasterise straight at thumbnail scale (libvips's thumbnail would load full size first)."""

    import pyvips

    width, height = _svg_points(svg)
    scale = min(THUMB_BOX[0] / width, THUMB_BOX[1] / height)
    pyvips.Image.svgload(str(svg), unlimited=True, scale=scale).flatten(
        background=[255, 255, 255]).webpsave(str(target), Q=THUMB_QUALITY, strip=True)


def is_monster(svg: Path) -> bool:
    """Return whether an SVG needs DZI tiles: over 2,000 drawn nodes or over 5 MB."""

    size = svg.stat().st_size
    return size > MONSTER_SVG_BYTES or svg.read_text(errors="replace").count(
        'class="node"') > MONSTER_NODES


def _tiles(svg: Path, target: Path) -> None:
    import pyvips

    width, height = _svg_points(svg)
    scale = min(1.0, (TILE_MAX_PIXELS / (width * height)) ** 0.5)
    target.mkdir(parents=True, exist_ok=True)
    image = pyvips.Image.svgload(str(svg), unlimited=True, scale=scale,
                                 access="sequential").flatten(background=[255, 255, 255])
    image.dzsave(str(target / target.stem), tile_size=TILE_SIZE, overlap=TILE_OVERLAP,
                 suffix=".webp[Q=80]")


class _Collector:
    """Accumulates the row's artifacts, skips, output files and log lines for one render."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.renders: list[RenderArtifact] = []
        self.skipped: list[RenderSkip] = []
        self.files: list[OutputFile] = []
        self.log: list[dict[str, Any]] = []
        self.failed: set[tuple[str | None, str]] = set()

    def make(self, kind: str, variant: str | None, path: Path,
             build: Callable[[dict[str, Any]], object]) -> bool:
        """Run ``build`` for one artifact at ``path``; record it or its typed skip."""

        entry: dict[str, Any] = {"variant": variant, "kind": kind}
        started = time.monotonic()
        try:
            build(entry)
            if not path.exists():
                raise _Skip("render:missing", f"{path.name} was not written")
        except _Skip as skip:
            self._skip(kind, variant, skip.why, entry, str(skip))
            return False
        except Exception as exc:  # noqa: BLE001 -- an artifact failure never ends the render
            self._skip(kind, variant, "render:error", entry, f"{type(exc).__name__}: {exc}")
            return False
        finally:
            entry["secs"] = round(time.monotonic() - started, 3)
            self.log.append(entry)
        sha, size = tree_sha256(path), _size(path)
        entry.update(sha256=sha, bytes=size)
        self.renders.append(RenderArtifact(kind=kind, sha256=sha, bytes=size, variant=variant))
        self.files.append(OutputFile(kind=kind, path=path.relative_to(self.out_dir).as_posix(),
                                     sha256=sha, bytes=size))
        return True

    def _skip(self, kind: str, variant: str | None, why: str, entry: dict[str, Any],
              detail: str) -> None:
        entry.update(why=why, detail=detail[:300])
        self.skipped.append(RenderSkip(kind=kind, why=why, variant=variant))
        if not why.startswith("render:same-as-"):
            self.failed.add((variant, kind))


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _render_variant(trace: Any, key: str, out: _Collector, seen: dict[str, str]) -> bool:
    """Render one variant's SVG, PDF, thumbnail and (for a monster) tiles; False if deduped."""

    stem = out.out_dir / key
    svg = stem.with_name(f"{key}.svg")

    def draw(entry: dict[str, Any]) -> None:
        _draw(trace, key, stem, "svg", entry)
        digest = _sha256(svg)
        if digest in seen:
            svg.unlink()
            raise _Skip(f"render:same-as-{seen[digest]}")
        seen[digest] = key

    if not out.make("svg", key, svg, draw):
        return False
    pdf = stem.with_name(f"{key}.pdf")
    out.make("pdf", key, pdf, lambda entry: _draw(trace, key, stem, "pdf", entry))
    thumb = stem.with_name(f"{key}.webp")
    out.make("thumb", key, thumb, lambda _: _thumbnail(svg, thumb))
    if is_monster(svg):
        tiles = stem.with_name(f"{key}.tiles")
        out.make("tiles", key, tiles, lambda _: _tiles(svg, tiles))
    return True


def render_set(trace: Any, *, model_id: str, trace_sha: str, out_dir: Path,
               variants: Sequence[str] | None = None) -> tuple[RenderRow, list[OutputFile]]:
    """Render every artifact for the variants into ``out_dir``; return the render row and files.

    The outcome is ``fail`` when the default variant has no SVG or thumbnail (the model is then
    ``publishable_pending_render``), ``partial`` when anything else failed, else ``pass``;
    ``infra`` when Graphviz or libvips is missing. Deduplicated variants are not failures.
    """

    import torchlens as tl

    started = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(variants) if variants is not None else variants_for(trace)
    out, seen = _Collector(out_dir), {}
    try:
        rev = renderer_rev()
    except (OSError, ImportError, RuntimeError, subprocess.SubprocessError) as exc:
        out.log.append({"kind": "renderer", "why": "render:infra",
                        "detail": f"{type(exc).__name__}: {exc}"[:300]})
        return _row(out, model_id, trace_sha, "unavailable", tl.__version__, "infra", started)
    drawn = [key for key in keys if _render_variant(trace, key, out, seen)]
    for kind, name in EXPORTS:
        path = out_dir / name
        out.make(kind, None, path, lambda e, k=kind, p=path: _export(k, trace, p, e))
    default = default_variant(drawn or keys, is_recurrent=bool(trace.is_recurrent),
                              n_ops=int(trace.num_ops))
    if not drawn or {(default, "svg"), (default, "thumb")} & out.failed:
        outcome = "fail"
    else:
        outcome = "partial" if out.failed else "pass"
    return _row(out, model_id, trace_sha, rev, tl.__version__, outcome, started)


def _export(kind: str, trace: Any, path: Path, log: dict[str, Any]) -> None:
    """Write one trace-level export; the HTML viewer carries no tensor values."""

    import torchlens as tl

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if kind == "html":
                options = tl.export.ReportOptions(deterministic=True, share_safe=True)
                tl.export.html(trace, path, options=options)
            else:
                getattr(tl.export, kind)(trace, path)
        finally:
            log["warnings"] = _first_lines(caught)


def _row(out: _Collector, model_id: str, trace_sha: str, rev: str, tl_version: str,
         outcome: str, started: float) -> tuple[RenderRow, list[OutputFile]]:
    """Write ``render.log`` as the evidence and build the row."""

    log_path = out.out_dir / "render.log"
    log_path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in out.log))
    evidence = _sha256(log_path)
    files = [*out.files, OutputFile(kind="log", path=log_path.name, sha256=evidence,
                                    bytes=log_path.stat().st_size)]
    row = RenderRow(id=model_id, at=now(), trace=trace_sha, renderer_rev=rev, tl=tl_version,
                    outcome=outcome, renders=tuple(out.renders),  # type: ignore[arg-type]
                    skipped=tuple(out.skipped), secs=round(time.monotonic() - started, 3),
                    evidence=evidence)
    return row, files
