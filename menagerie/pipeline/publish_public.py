"""The public form of a site export: diagnostics out of the drawings, references only to published files.

Runs where the store is (a worker), around ``publish derive`` and ``publish stage``:

- ``clean`` (before derive) finds variant SVGs whose graph label carries a renderer diagnostic (the
  ``collapse fell back to the floor plan`` line) and writes a copy without that line into the
  derivatives folder, named by its own hash. The record then points at the copy, its small picture
  is redrawn from the copy, and its PDF (drawn with the same line) is left out. ``derive`` and
  ``stage`` read a derivatives-folder file before the store, so crops, share images and uploads
  all come from the copy. A per-SVG cache (``clean/<svg sha>.json``) makes a second run read only
  new SVGs. The renderer fix that stops printing the line is in gotchas.md.
- ``restrict`` (after stage) keeps a reference only when its object staged: a reference of a kind
  that is not uploaded (the HTML viewer, Netron, Model Explorer, the TorchLens spec, tile sets) is
  removed, and so is a missing object of a kind named in ``null_missing`` (``pdf``). Any other
  missing object stays referenced, so the publish loop's sanity step refuses the export.
- ``release`` (after restrict) writes the full records as gzipped JSON Lines with public file
  addresses into the staged folder (``dataset/<version>/models-<sha12>.jsonl.gz``) and names it in
  the manifest's ``dataset_release``, for the Dataset page's download and Python example.

Field meanings: ``docs/site-export-schema.md``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from menagerie.pipeline import StoreLayout, publish
from menagerie.pipeline.records import RecordError

CLEAN_REV = "clean-1"
# A renderer diagnostic printed into the graph label (render-opts@06c1f0d3): one <text> line.
DIAGNOSTIC = re.compile(rb'<text\b[^>]*>collapse fell back to the floor plan[^<]*</text>\r?\n?')
HEAD_BYTES = 65_536  # the graph label sits at the top of the file
THUMB_QUALITY = 80
REQUIRED_KINDS = frozenset({"svg", "thumb"})
EXPORT_ASSET = re.compile(r"/export/assets/([A-Za-z0-9._-]+)")
RELEASE_DIR = "dataset"


# --- clean ------------------------------------------------------------------------------------


def _read_records(export_dir: Path) -> list[dict[str, Any]]:
    with (export_dir / "models.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _replace(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_records(export_dir: Path, records: list[dict[str, Any]]) -> None:
    tmp = export_dir / ".models.jsonl.tmp"
    publish._write_jsonl(tmp, records)  # noqa: SLF001
    os.replace(tmp, export_dir / "models.jsonl")
    tmp = export_dir / ".assets.tsv.tmp"
    publish.write_assets(tmp, records)
    os.replace(tmp, export_dir / "assets.tsv")


def _update_manifest(export_dir: Path, change: Callable[[dict[str, Any]], None]) -> None:
    path = export_dir / "export_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    _replace(path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _reader(store: StoreLayout, hint: Path | None) -> Callable[[str, str, Callable[[Path], Any]], Any]:
    """``read(sha, ext, fn)``: ``fn`` on a local copy (``hint/<sha><ext>``) when there is one, else
    on the store's blob."""

    from menagerie.pipeline import store as store_mod

    def read(sha: str, ext: str, fn: Callable[[Path], Any]) -> Any:
        local = hint / f"{sha}{ext}" if hint else None
        if local is not None and local.is_file():
            return fn(local)
        return store_mod.read_blob(sha, fn, store=store)

    return read


def strip_diagnostic(svg: bytes) -> bytes | None:
    """The SVG without its diagnostic line, or None when its header has none."""

    if not DIAGNOSTIC.search(svg[:HEAD_BYTES]):
        return None
    head, tail = svg[:HEAD_BYTES], svg[HEAD_BYTES:]
    return DIAGNOSTIC.sub(b"", head) + tail


def _thumb(clean: Path, width: int, out: Path) -> dict[str, Any] | None:
    """A small picture of the cleaned SVG at the old one's width (WebP), or None without libvips."""

    try:
        import pyvips
    except Exception:  # noqa: BLE001 -- the binding or libvips is not installed
        return None
    image = pyvips.Image.thumbnail(str(clean), max(1, width), height=10_000_000)
    image = image.flatten(background=[255, 255, 255]) if image.hasalpha() else image
    data = image.webpsave_buffer(Q=THUMB_QUALITY, strip=True)
    sha = hashlib.sha256(data).hexdigest()
    target = out / f"{sha}.webp"
    if not target.exists():
        _replace(target, data)
    return {"sha256": sha, "bytes": len(data), "width": image.width, "height": image.height}


def _clean_one(job: Mapping[str, Any], read: Any, out: Path) -> dict[str, Any]:
    """Check one SVG; write its cleaned copy and small picture when it has a diagnostic."""

    entry: dict[str, Any] = {"rev": CLEAN_REV, "svg": job["svg"], "diagnostic": False}
    try:
        head = read(job["svg"], ".svg", lambda p: p.open("rb").read(HEAD_BYTES))
    except FileNotFoundError:
        entry["missing"] = True  # stage reports it; nothing to clean
        return entry
    if not DIAGNOSTIC.search(head):
        return entry
    cleaned = strip_diagnostic(read(job["svg"], ".svg", lambda p: p.read_bytes()))
    if cleaned is None:  # pragma: no cover - the header said otherwise
        return entry
    sha = hashlib.sha256(cleaned).hexdigest()
    target = out / f"{sha}.svg"
    if not target.exists():
        _replace(target, cleaned)
    entry.update(diagnostic=True, clean={"sha256": sha, "bytes": len(cleaned)}, thumb=None)
    if job.get("thumb_width"):
        try:
            entry["thumb"] = _thumb(target, int(job["thumb_width"]), out)
        except Exception as exc:  # noqa: BLE001 -- the record falls back to the SVG as its picture
            entry["thumb_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return entry


def _cache_path(out: Path, sha: str) -> Path:
    return out / "clean" / f"{sha}.json"


def _cached(out: Path, sha: str) -> dict[str, Any] | None:
    try:
        entry = json.loads(_cache_path(out, sha).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if entry.get("rev") != CLEAN_REV or entry.get("missing"):
        return None
    files = [entry["clean"]["sha256"] + ".svg"] if entry.get("diagnostic") else []
    files += [entry["thumb"]["sha256"] + ".webp"] if entry.get("thumb") else []
    return entry if all((out / name).is_file() for name in files) else None


def _clean_jobs(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    for record in records:
        for variant in record["variants"]:
            thumb = variant.get("thumb") or {}
            width = thumb.get("width") if str(thumb.get("url", "")).endswith(".webp") else None
            job = jobs.setdefault(variant["svg"]["sha256"], {"svg": variant["svg"]["sha256"],
                                                             "thumb_width": None})
            job["thumb_width"] = job["thumb_width"] or width
    return list(jobs.values())


def clean(export_dir: Path, *, store: StoreLayout, derived: Path | None = None,
          assets_hint: Path | None = None, workers: int | None = None) -> dict[str, Any]:
    """Point every record at diagnostic-free copies of its SVGs; return the counts."""

    export_dir = Path(export_dir)
    out = Path(derived) if derived else export_dir / "derived"
    (out / "clean").mkdir(parents=True, exist_ok=True)
    records = _read_records(export_dir)
    read = _reader(store, Path(assets_hint) if assets_hint else None)
    jobs = _clean_jobs(records)
    entries = {j["svg"]: e for j in jobs if (e := _cached(out, j["svg"])) is not None}
    todo = [j for j in jobs if j["svg"] not in entries]
    workers = workers or int(os.environ.get("MENAGERIE_CPUS") or 2)
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for entry in pool.map(lambda j: _clean_one(j, read, out), todo):
            entries[entry["svg"]] = entry
            if not entry.get("missing"):
                _replace(_cache_path(out, entry["svg"]),
                         (json.dumps(entry, sort_keys=True) + "\n").encode("utf-8"))
    counts = apply_clean(records, entries)
    _write_records(export_dir, records)
    _update_manifest(export_dir, lambda m: m["counts"].update(counts))
    return {**counts, "checked": len(jobs), "read": len(todo)}


def apply_clean(records: Sequence[dict[str, Any]],
                entries: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Point the variants of every cleaned SVG at its copy, in place; return the counts."""

    variants = pdfs = 0
    for record in records:
        for variant in record["variants"]:
            entry = entries.get(variant["svg"]["sha256"]) or {}
            if not entry.get("diagnostic"):
                continue
            clean_ref = publish.asset(entry["clean"]["sha256"], "svg", entry["clean"]["bytes"])
            thumb = entry.get("thumb")
            variant["svg"] = clean_ref
            variant["thumb"] = ({**publish.asset(thumb["sha256"], "thumb", thumb["bytes"]),
                                 "width": thumb["width"], "height": thumb["height"]}
                                if thumb else dict(clean_ref))
            pdfs += variant.get("pdf") is not None
            variant["pdf"] = None  # drawn with the same diagnostic line
            variants += 1
    svgs = sum(1 for e in entries.values() if e.get("diagnostic"))
    return {"diagnostic_svgs_cleaned": svgs, "diagnostic_variants_cleaned": variants,
            "diagnostic_pdfs_removed": pdfs}


# --- restrict ---------------------------------------------------------------------------------


def staged_keys(assets: Path) -> set[str]:
    """Object keys ``MANIFEST.tsv`` lists (a tile set's files count as its folder key)."""

    keys = set()
    with (Path(assets) / "MANIFEST.tsv").open(encoding="utf-8") as handle:
        for line in handle:
            key = line.split("\t", 1)[0]
            if key and key != "key":
                keys.add(key)
                if ".tiles/" in key:
                    keys.add(key.split("/", 1)[0])
    return keys


def asset_refs(record: Mapping[str, Any]) -> Iterable[tuple[dict[str, Any], str, str]]:
    """``(holder, field, kind)`` for every asset reference of a record, variants included."""

    for field, kind in publish.RECORD_ASSETS.items():
        if record.get(field):
            yield record, field, kind  # type: ignore[misc]
    for variant in record.get("variants") or []:
        for field in publish.VARIANT_ASSETS:
            ref = variant.get(field)
            if ref:
                yield variant, field, _variant_kind(field, ref)


def _variant_kind(field: str, ref: Mapping[str, Any]) -> str:
    """The upload kind of a variant's file: a small picture that is the SVG itself is an SVG."""

    if field == "thumb_crop":
        return "crop"
    if field == "thumb":
        return "svg" if str(ref.get("url", "")).endswith(".svg") else "thumb"
    return field


def key_of(ref: Mapping[str, Any]) -> str:
    return str(ref.get("url", "")).split("?", 1)[0].rsplit("/", 1)[-1]


def unstaged_refs(records: Iterable[Mapping[str, Any]], staged: set[str]) -> list[tuple[str, str, str]]:
    """``(stable_id, kind, key)`` for every reference whose object did not stage."""

    return [(r.get("stable_id", "?"), kind, key_of(holder[field]))
            for r in records for holder, field, kind in asset_refs(r)
            if key_of(holder[field]) not in staged]


def restrict(export_dir: Path, assets: Path, *, kinds: str | Sequence[str],
             null_missing: Sequence[str] = ()) -> dict[str, Any]:
    """Remove references the public site cannot serve; return what was removed and what is left."""

    from menagerie.pipeline import publish_assets

    wanted = publish_assets.parse_kinds(kinds) or frozenset(publish.EXTENSIONS)
    forbidden = REQUIRED_KINDS & set(null_missing)
    if forbidden:
        raise RecordError(f"cannot leave out required kinds {sorted(forbidden)}")
    export_dir = Path(export_dir)
    staged = staged_keys(Path(assets))
    records = _read_records(export_dir)
    removed: Counter[str] = Counter()
    for record in records:
        for holder, field, kind in list(asset_refs(record)):
            if key_of(holder[field]) in staged:
                continue
            if kind not in wanted:
                removed[f"unpublished:{kind}"] += 1
            elif kind in null_missing:
                removed[f"missing:{kind}"] += 1
            else:
                continue  # stays referenced; the sanity step refuses the export
            holder[field] = None
    left = unstaged_refs(records, staged)
    _write_records(export_dir, records)
    counts = {"refs_removed": dict(sorted(removed.items())), "refs_unstaged": len(left),
              "public_kinds": sorted(wanted)}
    _update_manifest(export_dir, lambda m: m["counts"].update(counts))
    # Stable ids and kinds only: object keys are content hashes, which leak scans read as secrets.
    return {**counts, "unstaged": [[sid, kind] for sid, kind, _ in left[:20]]}


# --- release ----------------------------------------------------------------------------------


def public_href(base: str, cache_key: str) -> Callable[[str], str]:
    """``site/src/lib/asset-href.mjs`` (``makeAssetHref``) for one asset base and cache key."""

    root, query = base.rstrip("/"), cache_key.lstrip("?&")

    def href(key: str) -> str:
        return f"{root}/{key}" + (f"?{query}" if query else "")

    return href


def release(export_dir: Path, assets: Path, *, base: str, cache_key: str = "?v=1",
            version: str | None = None) -> dict[str, Any]:
    """Write the gzipped public JSON Lines into ``assets`` and name it in the manifest."""

    export_dir, assets = Path(export_dir), Path(assets)
    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    version = version or str(manifest.get("export_version") or manifest.get("as_of"))
    text = (export_dir / "models.jsonl").read_text(encoding="utf-8")
    href = public_href(base, cache_key)
    public = EXPORT_ASSET.sub(lambda m: href(m.group(1)), text).encode("utf-8")
    data = gzip.compress(public, compresslevel=9, mtime=0)
    sha = hashlib.sha256(data).hexdigest()
    key = f"{RELEASE_DIR}/{version}/models-{sha[:12]}.jsonl.gz"
    target = assets / key
    target.parent.mkdir(parents=True, exist_ok=True)
    _replace(target, data)
    for old in (assets / RELEASE_DIR).rglob("*"):
        if old.is_file() and old != target:
            old.unlink()
    rows = [line for line in (assets / "MANIFEST.tsv").read_text(encoding="utf-8").splitlines(True)
            if not line.startswith(f"{RELEASE_DIR}/")]
    rows.append(f"{key}\t{sha}\t{len(data)}\tapplication/gzip\n")
    _replace(assets / "MANIFEST.tsv", "".join(rows).encode("utf-8"))
    entry = {"name": "models.jsonl.gz", "path": key, "bytes": len(data), "sha256": sha,
             "format": "application/gzip", "rows": sum(1 for line in text.splitlines() if line.strip()),
             "uncompressed_bytes": len(public)}
    _update_manifest(export_dir, lambda m: m.update(dataset_release={"version": version,
                                                                     "files": [entry]}))
    return {"path": key, "bytes": len(data), "rows": entry["rows"]}
