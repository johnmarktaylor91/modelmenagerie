"""Site asset derivatives and staging: SVG sizes, top crops, share images, and the upload folder.

Runs where the store is (a worker), after ``publish export``:

- ``derive`` reads each variant SVG from the store once, records its own width and height in CSS
  px, rasterises the top crop (``crop_plan``: the graph fitted to 800 px wide and its top 600 px,
  or, for a very wide graph that would be a thin strip, its top-left corner at a readable scale;
  WebP) and, for default variants, a 1200 by 630 share image (JPEG, the top of the graph on
  white). Images are named by content hash in the derivatives folder; a per-SVG cache entry
  (``by-svg/<svg sha>.json``) makes a second run incremental, and a share image made under an
  earlier revision with the same share settings is kept. It then rewrites ``models.jsonl``,
  ``assets.tsv`` and the manifest with the new fields. No model is re-rendered.
- ``stage`` copies the asset kinds asked for (store blobs and derivatives) to ``DEST/<key>`` and
  writes ``DEST/MANIFEST.tsv`` (key, sha256, bytes, content type) for the upload step. A missing
  blob is reported by key and never stops the rest.

A monster SVG (over 5 MB) is rasterised alone in a child process under a memory cap taken from
``MENAGERIE_MEMORY_GB``; when it dies or times out, its crop is left out (null) and the site
falls back to the whole-graph thumbnail. Field meanings: ``docs/site-export-schema.md``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
import re
import resource
import shutil
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

from menagerie.pipeline import StoreLayout, publish, tree_sha256
from menagerie.pipeline.records import RecordError

CROP_W, CROP_H, CROP_QUALITY = 800, 600, 78
# A crop whose fitted height would fall under CORNER_BELOW_H px takes the top-left corner instead,
# drawn at the scale a READABLE_W px wide graph gets when fitted (node text stays legible).
CORNER_BELOW_H, READABLE_W = 150, 1200
# The plan's card budget is 40 KB a crop; a busier crop steps down in quality to meet it.
CROP_MAX_BYTES = 40_000
CROP_FALLBACK_QUALITIES = (65, 50)
SHARE_W, SHARE_H, SHARE_QUALITY = 1200, 630, 85
# The revision names every raster parameter; a change to one moves it, and cached entries are
# redone. A share image whose SHARE_SPEC is unchanged is carried over (``_carried_share``).
CROP_SPEC = (f"crop={CROP_W}x{CROP_H} webp q{CROP_QUALITY} "
             f"corner<{CORNER_BELOW_H}px@{READABLE_W}px")
SHARE_SPEC = f"share={SHARE_W}x{SHARE_H} jpeg q{SHARE_QUALITY}"
DERIVE_REV = f"derive/2 {CROP_SPEC} {SHARE_SPEC}"
PX_PER_PT = 96 / 72  # CSS px per point; libvips draws one pixel per point at its 72 dpi
MONSTER_SVG_BYTES = publish.MONSTER_SVG_BYTES
MONSTER_TIMEOUT_S = 600
CHUNK_PER_WORKER = 8
WHITE = [255, 255, 255]
DEFAULT_KINDS = ("svg", "thumb", "crop", "pdf", "share")
EXTRA_KINDS = ("html", "netron", "model_explorer", "tiles")
DERIVED_KINDS = frozenset({"crop", "share"})
CONTENT_TYPES = {
    ".svg": "image/svg+xml", ".webp": "image/webp", ".pdf": "application/pdf",
    ".html": "text/html; charset=utf-8", ".json": "application/json", ".jpg": "image/jpeg",
    ".dzi": "application/xml", ".csv": "text/csv; charset=utf-8",
}


# --- SVG and WebP sizes ---------------------------------------------------------------------

_SVG_TAG = re.compile(r"<svg\b[^>]*>", re.DOTALL)
_LENGTH = re.compile(r"^\s*([\d.]+)\s*(px|pt|in|cm|mm|pc)?\s*$")
_UNIT_PX = {None: 1.0, "px": 1.0, "pt": PX_PER_PT, "in": 96.0, "cm": 96 / 2.54,
            "mm": 96 / 25.4, "pc": 16.0}


def svg_size(svg: Path) -> tuple[float, float, float]:
    """Return ``(width_px, height_px, raster_scale_1)``: the SVG's own size in CSS px, and its
    width in pixels when libvips loads it at scale 1 (points and px both draw one pixel each).

    Graphviz SVGs give width and height in points (``render._svg_points``); anything else is
    read from the root element's width and height, else its viewBox."""

    from menagerie.pipeline import render

    try:
        width_pt, height_pt = render._svg_points(svg)  # noqa: SLF001 -- the renderer's own parse
        return width_pt * PX_PER_PT, height_pt * PX_PER_PT, width_pt
    except Exception:  # noqa: BLE001 -- not a points header: read the root element instead
        pass
    with svg.open("rb") as handle:
        head = handle.read(16_384).decode("utf-8", errors="replace")
    return svg_size_of_text(head)


def svg_size_of_text(head: str) -> tuple[float, float, float]:
    """``svg_size`` on the start of an SVG's text (see there)."""

    tag = _SVG_TAG.search(head)
    if tag is None:
        raise RecordError("no <svg> root element in the SVG's first 16 KB")
    attrs = dict(re.findall(r'\s([\w:-]+)\s*=\s*"([^"]*)"', tag.group()))
    width, height = (_length(attrs.get(k)) for k in ("width", "height"))
    box = [float(v) for v in re.split(r"[\s,]+", attrs.get("viewBox", "").strip()) if v]
    if (width is None or height is None) and len(box) == 4 and box[2] > 0 and box[3] > 0:
        width = width or (box[2], None)
        height = height or (box[3], None)
    if width is None or height is None:
        raise RecordError("the SVG gives neither a width and height nor a viewBox")
    width_px = width[0] * _UNIT_PX[width[1]]
    # libvips draws physical units at 72 dpi and user units and px at one pixel each.
    raster = width[0] * (72 / 96) * _UNIT_PX[width[1]] if width[1] not in (None, "px", "pt") \
        else width[0]
    return width_px, height[0] * _UNIT_PX[height[1]], raster


def _length(value: str | None) -> tuple[float, str | None] | None:
    match = _LENGTH.match(value or "")
    return (float(match[1]), match[2]) if match and float(match[1]) > 0 else None


def webp_size(path: Path) -> tuple[int, int] | None:
    """Return a WebP file's pixel width and height from its header, or ``None``."""

    with path.open("rb") as handle:
        head = handle.read(30)
    if len(head) < 30 or head[:4] != b"RIFF" or head[8:12] != b"WEBP":
        return None
    chunk = head[12:16]
    if chunk == b"VP8X":
        return (1 + int.from_bytes(head[24:27], "little"),
                1 + int.from_bytes(head[27:30], "little"))
    if chunk == b"VP8 ":
        return (int.from_bytes(head[26:28], "little") & 0x3FFF,
                int.from_bytes(head[28:30], "little") & 0x3FFF)
    if chunk == b"VP8L":
        bits = int.from_bytes(head[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


# --- One SVG: sizes and rasters (runs in a worker process) -----------------------------------


def _init_worker(cap_bytes: int) -> None:
    """Cap the process's data size so a runaway SVG fails here, not on the machine."""

    if cap_bytes > 0:
        resource.setrlimit(resource.RLIMIT_DATA, (cap_bytes, cap_bytes))
    try:
        import pyvips

        pyvips.cache_set_max(0)
    except Exception:  # noqa: BLE001 -- no libvips here: sizes only
        pass


def _have_pyvips() -> bool:
    try:
        import pyvips  # noqa: F401
    except Exception:  # noqa: BLE001 -- an import failure of the binding or of libvips
        return False
    return True


def derive_one(job: Mapping[str, Any], store_root: str, derived: str) -> dict[str, Any]:
    """Measure one SVG and make the rasters ``job`` asks for; write and return its cache entry."""

    from menagerie.pipeline import store as store_mod

    store, out = StoreLayout(Path(store_root)), Path(derived)

    def read(sha: str, ext: str, fn: Any) -> Any:
        # A cleaned copy (``publish clean``) lives only in the derivatives folder.
        local = out / f"{sha}{ext}"
        return fn(local) if local.is_file() else store_mod.read_blob(sha, fn, store=store)

    entry: dict[str, Any] = {"rev": DERIVE_REV, "svg": job["svg"]}
    try:
        width_px, height_px, raster_w = read(job["svg"], ".svg", svg_size)
        entry.update(svg_width_px=round(width_px), svg_height_px=round(height_px))
    except FileNotFoundError:
        entry["missing"] = True
        return _write_entry(out, entry)
    except Exception as exc:  # noqa: BLE001 -- a size the SVG does not state is left null
        entry["size_error"] = f"{type(exc).__name__}: {exc}"[:300]
        raster_w = None
    if job.get("thumb"):
        try:
            size = read(job["thumb"], ".webp", webp_size)
            entry["thumb"] = {"sha256": job["thumb"], "width": size[0], "height": size[1]} \
                if size else None
        except FileNotFoundError:
            entry["thumb"] = None
    if job.get("share_done"):  # made under an earlier revision with the same share settings
        entry["share"] = job["share_done"]
    wanted = [k for k in ("crop", "share") if job.get(k)
              and not (k == "share" and job.get("share_done"))]
    if wanted and raster_w is None:
        entry["crop_skip"] = "no-size"
    elif wanted and not _have_pyvips():
        entry["crop_skip"] = "no-libvips"
    elif wanted:
        try:
            sizes = (width_px, height_px, raster_w)
            read(job["svg"], ".svg", lambda p: _rasters(p, sizes, wanted, out, entry))
        except Exception as exc:  # noqa: BLE001 -- one bad SVG never ends the run
            entry["crop_skip"] = f"error: {type(exc).__name__}: {exc}"[:300]
    return _write_entry(out, entry)


def crop_plan(width_px: float, height_px: float, raster_w: float) -> tuple[str, float]:
    """Return ``(rule, libvips scale)`` for an SVG's crop.

    ``fit``: the whole width fitted to ``CROP_W``. ``corner``: when the fitted height would fall
    under ``CORNER_BELOW_H`` (a graph more than about five times wider than tall), the top-left
    corner at the scale a ``READABLE_W`` px wide graph gets when fitted, so node text is legible.
    A graph no wider than ``READABLE_W`` is already drawn at that scale or larger, so it fits."""

    fitted_h = height_px * CROP_W / width_px
    if fitted_h < CORNER_BELOW_H and width_px > READABLE_W:
        return "corner", CROP_W * width_px / (READABLE_W * raster_w)
    return "fit", CROP_W / raster_w


def _rasters(svg: Path, sizes: tuple[float, float, float], wanted: Sequence[str], out: Path,
             entry: dict[str, Any]) -> None:
    import pyvips

    width_px, height_px, raster_w = sizes
    if "crop" in wanted:
        rule, scale = crop_plan(width_px, height_px, raster_w)
        top = _top(pyvips, svg, scale, CROP_W, CROP_H)
        data = top.webpsave_buffer(Q=CROP_QUALITY, strip=True)
        for quality in CROP_FALLBACK_QUALITIES:
            if len(data) <= CROP_MAX_BYTES:
                break
            data = top.webpsave_buffer(Q=quality, strip=True)
        entry["crop"] = {**_save(out, data, ".webp", top.width, top.height), "rule": rule}
    if "share" in wanted:
        top = _top(pyvips, svg, SHARE_W / raster_w, SHARE_W, SHARE_H)
        canvas = top.embed((SHARE_W - top.width) // 2, (SHARE_H - top.height) // 2,
                           SHARE_W, SHARE_H, extend="background", background=WHITE)
        entry["share"] = _save(out, canvas.jpegsave_buffer(Q=SHARE_QUALITY, strip=True),
                               ".jpg", SHARE_W, SHARE_H)


def _top(pyvips: Any, svg: Path, scale: float, width: int, height: int) -> Any:
    """The top ``width`` by ``height`` pixels of the SVG drawn at ``scale``, flattened on white.
    libvips draws only the region the crop asks for."""

    image = pyvips.Image.svgload(str(svg), unlimited=True, scale=scale)
    top = image.crop(0, 0, min(width, image.width), min(height, image.height))
    return top.flatten(background=WHITE).copy_memory()


def _save(out: Path, data: bytes, ext: str, width: int, height: int) -> dict[str, Any]:
    sha = hashlib.sha256(data).hexdigest()
    target = out / f"{sha}{ext}"
    if not target.exists():
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
    return {"sha256": sha, "bytes": len(data), "width": width, "height": height}


def _entry_path(out: Path, svg_sha: str) -> Path:
    return out / "by-svg" / f"{svg_sha}.json"


def _write_entry(out: Path, entry: dict[str, Any]) -> dict[str, Any]:
    path = _entry_path(out, entry["svg"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return entry


def _read_entry(out: Path, svg_sha: str) -> dict[str, Any] | None:
    path = _entry_path(out, svg_sha)
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return entry if entry.get("rev") == DERIVE_REV else None


def _carried_share(out: Path, svg_sha: str) -> dict[str, Any] | None:
    """The share image of an earlier revision's entry made with today's ``SHARE_SPEC``, when its
    file is still in the derivatives folder; else ``None``."""

    try:
        entry = json.loads(_entry_path(out, svg_sha).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    share = entry.get("share")
    if (entry.get("rev") == DERIVE_REV or SHARE_SPEC not in str(entry.get("rev"))
            or not share or not (out / f"{share['sha256']}.jpg").is_file()):
        return None
    return share


def _isolated_main(job: Mapping[str, Any], store_root: str, derived: str, cap: int) -> None:
    _init_worker(cap)
    derive_one(job, store_root, derived)


# --- The run: jobs, pools and the rewrite ----------------------------------------------------


def _jobs(records: Sequence[Mapping[str, Any]], *, variants: str,
          share: bool) -> list[dict[str, Any]]:
    """One job per distinct SVG, default variants first; flags merged over every use."""

    jobs: dict[str, dict[str, Any]] = {}
    ordered = sorted(((v["key"] != r["default_variant"], r, v) for r in records
                      for v in r["variants"]), key=lambda t: t[0])
    for not_default, record, variant in ordered:
        svg = variant["svg"]
        thumb = variant["thumb"]["sha256"] if variant["thumb"]["url"].endswith(".webp") else None
        job = jobs.setdefault(svg["sha256"], {"svg": svg["sha256"], "bytes": svg["bytes"],
                                              "thumb": thumb, "crop": False, "share": False})
        job["crop"] |= variants == "all" or not not_default
        job["share"] |= share and not not_default
    return list(jobs.values())


def _needs_work(entry: Mapping[str, Any] | None, job: Mapping[str, Any]) -> bool:
    if entry is None or entry.get("missing") or entry.get("crop_skip") == "no-libvips":
        return True
    if job["thumb"] and "thumb" not in entry:
        return True
    if entry.get("crop_skip"):  # a memory cap, a timeout or a bad SVG: kept until DERIVE_REV moves
        return False
    return any(job[k] and k not in entry for k in ("crop", "share"))


class _Run:
    """Settings and counters of one derive run."""

    def __init__(self, store: StoreLayout, out: Path, workers: int, memory_gb: float,
                 deadline: float | None) -> None:
        self.store_root, self.out = str(store.root), out
        self.workers = max(1, workers)
        total = int(memory_gb * 2**30)
        self.worker_cap = max(int(1.5 * 2**30), int(0.8 * total / self.workers))
        self.monster_cap = max(self.worker_cap, int(0.75 * total))
        self.deadline = deadline
        self.ctx = multiprocessing.get_context("spawn")
        self.entries: dict[str, dict[str, Any]] = {}
        self.isolated: Counter[str] = Counter()
        self.unstarted = 0

    def late(self) -> bool:
        return self.deadline is not None and time.monotonic() > self.deadline

    def pool_phase(self, jobs: Sequence[dict[str, Any]]) -> None:
        """Run jobs in a capped process pool; a crash retries its chunk's unfinished jobs alone."""

        chunk = self.workers * CHUNK_PER_WORKER
        pool: ProcessPoolExecutor | None = None
        try:
            for start in range(0, len(jobs), chunk):
                if self.late():
                    self.unstarted += len(jobs) - start
                    break
                pool = pool or ProcessPoolExecutor(self.workers, mp_context=self.ctx,
                                                   initializer=_init_worker,
                                                   initargs=(self.worker_cap,))
                futures = [(pool.submit(derive_one, job, self.store_root, str(self.out)), job)
                           for job in jobs[start:start + chunk]]
                retry = []
                for future, job in futures:
                    try:
                        self.entries[job["svg"]] = future.result()
                    except BrokenProcessPool:
                        retry.append(job)
                if retry:
                    pool.shutdown(wait=True, cancel_futures=True)
                    pool = None
                    for job in retry:
                        self.isolate(job, self.worker_cap, MONSTER_TIMEOUT_S)
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

    def monster_phase(self, jobs: Sequence[dict[str, Any]]) -> None:
        for index, job in enumerate(jobs):
            if self.late():
                self.unstarted += len(jobs) - index
                break
            self.isolate(job, self.monster_cap, MONSTER_TIMEOUT_S)

    def isolate(self, job: dict[str, Any], cap: int, timeout: float) -> None:
        """Run one job alone under ``cap``; when it dies, record the sizes without rasters."""

        child = self.ctx.Process(target=_isolated_main,
                                 args=(job, self.store_root, str(self.out), cap))
        child.start()
        child.join(timeout)
        why = None
        if child.is_alive():
            child.kill()
            child.join()
            why = "timeout"
        elif child.exitcode != 0:
            why = f"memory-cap (exit {child.exitcode})"
        entry = _read_entry(self.out, job["svg"]) if why is None else None
        if entry is None:
            why = why or "no entry written"
            self.isolated[why.split(" ")[0]] += 1
            entry = derive_one({**job, "crop": False, "share": False}, self.store_root,
                               str(self.out))
            entry["crop_skip"] = why
            _write_entry(self.out, entry)
        self.entries[job["svg"]] = entry


def derive(export_dir: Path, *, store: StoreLayout, derived: Path | None = None,
           workers: int | None = None, memory_gb: float | None = None,
           deadline_minutes: float | None = None, variants: str = "all",
           share: bool = True) -> dict[str, Any]:
    """Add SVG sizes, top crops and share images to the export at ``export_dir``; return counts.

    Work is cached per SVG in ``derived/by-svg/``, so a second run only does what is new. With
    ``deadline_minutes`` no raster starts after that long; the export is rewritten with what is
    done, and the rest stays null until the next run.
    """

    started = time.monotonic()
    export_dir = Path(export_dir)
    out = Path(derived) if derived else export_dir / "derived"
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or int(os.environ.get("MENAGERIE_CPUS") or 2)
    memory_gb = memory_gb or float(os.environ.get("MENAGERIE_MEMORY_GB") or 4)
    # Spawned workers read these at start: one libvips thread each, few malloc arenas.
    os.environ.setdefault("VIPS_CONCURRENCY", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    records = _read_records(export_dir)
    jobs = _jobs(records, variants=variants, share=share)
    run = _Run(store, out, workers, memory_gb,
               started + deadline_minutes * 60 if deadline_minutes else None)
    todo = []
    for job in jobs:
        entry = _read_entry(out, job["svg"])
        if _needs_work(entry, job):
            if entry is None and job["share"]:
                job["share_done"] = _carried_share(out, job["svg"])
            todo.append(job)
        else:
            run.entries[job["svg"]] = entry
    cached = len(run.entries)
    run.pool_phase([j for j in todo if j["bytes"] <= MONSTER_SVG_BYTES])
    run.monster_phase([j for j in todo if j["bytes"] > MONSTER_SVG_BYTES])
    counts = rewrite(export_dir, records, run.entries)
    report = {
        "export": str(export_dir), "derived": str(out), "rev": DERIVE_REV,
        "svgs": len(jobs), "cached": cached, "worked": len(todo) - run.unstarted,
        "unstarted": run.unstarted, "isolated_failures": dict(run.isolated),
        **_entry_counts(run.entries.values()), **counts,
        "workers": run.workers, "memory_gb": memory_gb,
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    # Replaced, never written in place: a day folder seeded by hard links shares the old file.
    _replace(out / "derive-report.json", lambda tmp: tmp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"))
    return report


def _entry_counts(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    entries = list(entries)
    skipped = Counter(str(e["crop_skip"]).split(":")[0].split(" ")[0]
                      for e in entries if e.get("crop_skip"))
    crops = [e["crop"]["bytes"] for e in entries if e.get("crop")]
    heights = sorted(e["crop"]["height"] for e in entries if e.get("crop"))
    return {
        "crops_by_rule": dict(Counter(e["crop"].get("rule", "fit")
                                      for e in entries if e.get("crop"))),
        "crop_heights": _spread(heights),
        "missing_svgs": sum(1 for e in entries if e.get("missing")),
        "size_errors": sum(1 for e in entries if e.get("size_error")),
        "crop_skipped_by_reason": dict(skipped),
        "crops_over_budget": sum(1 for b in crops if b > CROP_MAX_BYTES),
        "crop_bytes_total": sum(crops),
    }


def _spread(values: Sequence[int]) -> dict[str, Any]:
    """Minimum, quartiles, maximum and the counts under 150 and 300 of sorted ``values``."""

    if not values:
        return {}
    at = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {"min": values[0], "p25": at(0.25), "median": at(0.5), "p75": at(0.75),
            "max": values[-1], "under_150": sum(v < 150 for v in values),
            "under_300": sum(v < 300 for v in values), "at_600": sum(v == 600 for v in values)}


def _read_records(export_dir: Path) -> list[dict[str, Any]]:
    with (export_dir / "models.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sized(entry: Mapping[str, Any] | None, key: str, kind: str) -> dict[str, Any] | None:
    found = entry.get(key) if entry else None
    if not found:
        return None
    sized = {**publish.asset(found["sha256"], kind, found["bytes"]),
             "width": found["width"], "height": found["height"]}
    if kind == "crop":  # which crop rule drew it (``crop_plan``)
        sized["rule"] = found.get("rule", "fit")
    return sized


def rewrite(export_dir: Path, records: list[dict[str, Any]],
            entries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Write the derived fields into the export's records, assets list and manifest."""

    for record in records:
        share = None
        for variant in record["variants"]:
            entry = entries.get(variant["svg"]["sha256"])
            variant["svg_width_px"] = entry.get("svg_width_px") if entry else None
            variant["svg_height_px"] = entry.get("svg_height_px") if entry else None
            variant["thumb_crop"] = _sized(entry, "crop", "crop")
            thumb = entry.get("thumb") if entry else None
            if thumb and thumb["sha256"] == variant["thumb"]["sha256"]:
                variant["thumb"].update(width=thumb["width"], height=thumb["height"])
            if variant["key"] == record["default_variant"]:
                share = _sized(entry, "share", "share")
        record["share_image"] = share
    counts = {
        "thumb_crops": sum(1 for r in records for v in r["variants"] if v["thumb_crop"]),
        "default_thumb_crops": sum(1 for r in records for v in r["variants"]
                                   if v["thumb_crop"] and v["key"] == r["default_variant"]),
        "svg_dimensions": sum(1 for r in records for v in r["variants"] if v["svg_width_px"]),
        "share_images": sum(1 for r in records if r["share_image"]),
        "default_thumb_crops_by_rule": dict(sorted(Counter(
            v["thumb_crop"]["rule"] for r in records for v in r["variants"]
            if v["thumb_crop"] and v["key"] == r["default_variant"]).items())),
    }
    _replace(export_dir / "models.jsonl",
             lambda tmp: publish._write_jsonl(tmp, records))  # noqa: SLF001
    _replace(export_dir / "assets.tsv", lambda tmp: publish.write_assets(tmp, records))
    manifest_path = export_dir / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"].update(counts)
    _replace(manifest_path, lambda tmp: tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"))
    return counts


def _replace(path: Path, write: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    write(tmp)
    os.replace(tmp, path)


# --- Staging --------------------------------------------------------------------------------


def parse_kinds(kinds: str | Sequence[str] | None) -> frozenset[str] | None:
    """``None`` or ``all`` means every kind ``assets.tsv`` lists; else the named kinds."""

    if kinds is None:
        return None
    names = [k.strip() for k in (kinds.split(",") if isinstance(kinds, str) else kinds)
             if k.strip()]
    if "all" in names:
        return frozenset({*DEFAULT_KINDS, *EXTRA_KINDS, *(n for n in names if n != "all")})
    unknown = set(names) - set(publish.EXTENSIONS)
    if unknown:
        raise RecordError(f"unknown asset kinds {sorted(unknown)}; known: "
                          f"{sorted(publish.EXTENSIONS)}")
    return frozenset(names)


def asset_rows(export_dir: Path) -> list[dict[str, Any]]:
    """The rows of ``assets.tsv``; a file from before the ``kind`` column reads it off the key."""

    with (Path(export_dir) / "assets.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["kind"] = row.get("kind") or publish.asset_kind(row["key"])
        row["bytes"] = int(row["bytes"])
    return rows


def stage(export_dir: Path, dest: Path, *, store: StoreLayout,
          kinds: str | Sequence[str] | None = DEFAULT_KINDS, derived: Path | None = None,
          workers: int | None = None) -> dict[str, Any]:
    """Copy the named asset kinds to ``dest/<key>`` and write ``dest/MANIFEST.tsv``.

    Store blobs are read raw, gunzipped from their gzipped form, or through their packed form;
    derivatives come from ``derived`` (default ``export_dir/derived``). A file already at its
    key with the right size is kept. A missing blob goes to ``dest/MISSING.tsv`` and the result.
    """

    export_dir, dest = Path(export_dir), Path(dest)
    derived = Path(derived) if derived else export_dir / "derived"
    wanted = parse_kinds(kinds)
    rows = [r for r in asset_rows(export_dir) if wanted is None or r["kind"] in wanted]
    dest.mkdir(parents=True, exist_ok=True)
    workers = workers or int(os.environ.get("MENAGERIE_CPUS") or 2)
    with ThreadPoolExecutor(max(1, workers)) as pool:
        results = list(pool.map(lambda r: _stage_one(r, dest, store, derived), rows))
    lines, missing = [], []
    by_kind: dict[str, dict[str, int]] = {}
    for row, (files, why) in zip(rows, results):
        if why is not None:
            missing.append((row["key"], row["kind"], why))
            continue
        tally = by_kind.setdefault(row["kind"], {"objects": 0, "files": 0, "bytes": 0})
        tally["objects"] += 1
        tally["files"] += len(files)
        tally["bytes"] += sum(size for _, _, size, _ in files)
        lines += files
    # Replaced, never written in place: a day folder seeded by hard links shares the old files.
    _replace(dest / "MANIFEST.tsv", lambda tmp: tmp.write_text(
        "key\tsha256\tbytes\tcontent_type\n"
        + "".join(f"{k}\t{s}\t{b}\t{t}\n" for k, s, b, t in sorted(lines)), encoding="utf-8"))
    _replace(dest / "MISSING.tsv", lambda tmp: tmp.write_text(
        "key\tkind\twhy\n"
        + "".join(f"{k}\t{kind}\t{why}\n" for k, kind, why in sorted(missing)),
        encoding="utf-8"))
    return {"dest": str(dest), "kinds": sorted(wanted) if wanted else "all",
            "objects": len(rows) - len(missing), "files": len(lines),
            "bytes": sum(t["bytes"] for t in by_kind.values()), "by_kind": by_kind,
            "missing": [k for k, _, _ in sorted(missing)]}


def stage_summary(export_dir: Path, dest: Path, **kwargs: Any) -> dict[str, Any]:
    """``stage`` with the missing keys cut to the first 20 (all of them are in MISSING.tsv)."""

    report = stage(export_dir, dest, **kwargs)
    report["missing_count"] = len(report["missing"])
    report["missing"] = report["missing"][:20]
    return report


def _stage_one(row: Mapping[str, Any], dest: Path, store: StoreLayout,
               derived: Path) -> tuple[list[tuple[str, str, int, str]], str | None]:
    """Copy one asset; return its manifest lines, or why it is missing."""

    key, sha, kind = row["key"], row["sha256"], row["kind"]
    target = dest / key
    try:
        if kind in DERIVED_KINDS or (derived / key).is_file():  # cleaned copies too
            source = derived / key
            if not source.is_file():
                return [], "not derived"
            _copy_file(source, target, row["bytes"])
        else:
            _copy_blob(sha, target, store, row["bytes"])
    except FileNotFoundError:
        return [], "not in the store"
    except (OSError, ValueError) as exc:
        return [], f"{type(exc).__name__}: {exc}"[:200]
    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.is_file())
        return [(f"{key}/{p.relative_to(target).as_posix()}", tree_sha256(p), p.stat().st_size,
                 _content_type(p.name)) for p in files], None
    return [(key, sha, target.stat().st_size, _content_type(key))], None


def _copy_blob(sha: str, target: Path, store: StoreLayout, size: int) -> None:
    from menagerie.pipeline import store as store_mod

    if target.is_file() and target.stat().st_size == size:
        return
    raw, gzipped = store.blob(sha), store_mod.gzipped_path(sha, store=store)
    if raw.is_file():
        _copy_file(raw, target, size)
    elif not raw.exists() and gzipped.is_file():
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.unlink(missing_ok=True)
        store_mod.gunzip_file(gzipped, tmp)
        tmp.chmod(0o644)
        _check_size(tmp, size)
        os.replace(tmp, target)
    else:
        store_mod.read_blob(sha, lambda p: _copy_tree_or_file(p, target, size), store=store)


def _copy_tree_or_file(source: Path, target: Path, size: int) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        _copy_file(source, target, size)


def _copy_file(source: Path, target: Path, size: int) -> None:
    if target.is_file() and target.stat().st_size == size:
        return
    tmp = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, tmp)
    _check_size(tmp, size)
    os.replace(tmp, target)


def _check_size(path: Path, size: int) -> None:
    actual = path.stat().st_size
    if actual != size:
        path.unlink(missing_ok=True)
        raise ValueError(f"copied {actual} bytes, assets.tsv says {size}")


def _content_type(name: str) -> str:
    for ext, kind in CONTENT_TYPES.items():
        if name.endswith(ext):
            return kind
    return "application/octet-stream"
