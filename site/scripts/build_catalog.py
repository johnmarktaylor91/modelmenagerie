#!/usr/bin/env python3
"""Build the preview catalog export from the rendered menagerie gallery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


SVG_MONSTER_BYTES = 5 * 1024 * 1024
OPS_MONSTER_THRESHOLD = 2000
THUMB_WIDTH = 400
RASTER_TIMEOUT_SECONDS = 20
SCHEMA_VERSION = "phase0-target-schema"
EXPORT_VERSION = "phase0-preview"
PREDICATE_NOTE = "preview: render coverage only, validation pending"


@dataclass(frozen=True)
class ManifestRow:
    """A rendered manifest row selected for the preview export."""

    name: str
    model_id: str
    n_nodes: int | None
    render_path: Path
    graph_shape_hash: str
    category: str
    stable_id: str
    recipe_revision_sha256: str


@dataclass(frozen=True)
class AssetRef:
    """A content-addressed generated asset reference."""

    url: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class BuiltModel:
    """A built model record plus thumbnail generation status."""

    record: dict[str, Any]
    thumbnail_generated: bool
    thumbnail_placeholder: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", required=True, type=Path, help="Rendered gallery root directory.")
    parser.add_argument("--manifest", required=True, type=Path, help="Manifest TSV path.")
    parser.add_argument("--out", required=True, type=Path, help="Output export directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional model limit for local preview builds.")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel asset build workers.")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help=(
            "Optional verification ledger JSONL export (menagerie.ledger rows, "
            "e.g. from menagerie/export_ledger_jsonl.py). When given, a "
            "manifest row whose (stable_id, recipe_revision_sha256) has a "
            "passed, forward_pass=1, metadata_ok=1 ledger row is marked "
            "forward_validated with the ledger's real torchlens_version and "
            "n_ops; everything else keeps the prior render-only preview "
            "values (forward_validated=False, torchlens_version='unknown')."
        ),
    )
    return parser.parse_args()


def load_ledger(ledger_path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Load passed verification ledger rows keyed by (stable_id, recipe revision).

    Parameters
    ----------
    ledger_path:
        Path to a verification ledger JSONL export, or ``None``.

    Returns
    -------
    dict[tuple[str, str], dict[str, Any]]
        Passed rows keyed by ``(stable_id, recipe_revision_sha256)``. A
        manifest row only matches when both its current stable ID and its
        current recipe revision hash are present in the ledger with a
        passing status, so a stale or superseded recipe never inherits an
        old pass.
    """
    if ledger_path is None:
        return {}
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "passed" or row.get("forward_pass") != 1 or row.get("metadata_ok") != 1:
                continue
            key = (row.get("stable_id"), row.get("recipe_revision_sha256"))
            entries[key] = row
    return entries


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_ref(path: Path, out_root: Path) -> AssetRef:
    """Return an asset reference for a generated file."""
    digest = sha256_file(path)
    return AssetRef(
        url=f"/export/{path.relative_to(out_root).as_posix()}",
        sha256=digest,
        bytes=path.stat().st_size,
    )


def int_or_none(value: str) -> int | None:
    """Parse an integer-like field, returning None for blanks."""
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def slugify(value: str) -> str:
    """Return a deterministic URL slug base."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "model"


def derive_family(name: str) -> str:
    """Return a conservative best-effort model family label."""
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", name).strip()
    if not normalized:
        return ""
    words = normalized.split()
    return " ".join(words[:2]).lower()


def derive_category(gallery_root: Path, render_path: Path) -> str:
    """Return the gallery category from an SVG render path."""
    try:
        return render_path.resolve().relative_to(gallery_root.resolve()).parts[0]
    except (ValueError, IndexError):
        return "unknown"


def read_rendered_rows(gallery_root: Path, manifest_path: Path) -> tuple[list[ManifestRow], int]:
    """Read rendered manifest rows that point to existing SVG files."""
    rows: list[ManifestRow] = []
    cataloged = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            cataloged += 1
            render_path = Path(raw["render_path"])
            if raw["status"] != "rendered" or render_path.suffix.lower() != ".svg" or not render_path.exists():
                continue
            rows.append(
                ManifestRow(
                    name=raw["name"],
                    model_id=raw["model_id"],
                    n_nodes=int_or_none(raw["n_nodes"]),
                    render_path=render_path,
                    graph_shape_hash=raw["graph_shape_hash"],
                    category=derive_category(gallery_root, render_path),
                    stable_id=raw.get("stable_id", ""),
                    recipe_revision_sha256=raw.get("recipe_revision_sha256", ""),
                )
            )
    rows.sort(key=lambda row: (row.category, row.name.lower(), int_or_none(row.model_id) or 0, row.render_path.as_posix()))
    return rows, cataloged


def select_rows(rows: list[ManifestRow], limit: int | None) -> list[ManifestRow]:
    """Select a deterministic category-spread subset when a limit is requested."""
    if limit is None or limit >= len(rows):
        return rows
    buckets: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        buckets[row.category].append(row)
    selected: list[ManifestRow] = []
    categories = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for category in categories:
            bucket = buckets[category]
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    selected.sort(key=lambda row: (row.category, row.name.lower(), int_or_none(row.model_id) or 0, row.render_path.as_posix()))
    return selected


def assign_slugs(rows: list[ManifestRow]) -> list[str]:
    """Assign deterministic de-duplicated slugs to selected rows."""
    bases = [slugify(row.name) for row in rows]
    base_counts = Counter(bases)
    seen: Counter[str] = Counter()
    slugs: list[str] = []
    for row, base in zip(rows, bases, strict=True):
        candidate = base if base_counts[base] == 1 else f"{base}-{row.model_id}"
        seen[candidate] += 1
        slug = candidate if seen[candidate] == 1 else f"{candidate}-{seen[candidate]}"
        slugs.append(slug)
    return slugs


def unique_tokens(rows: list[ManifestRow], slugs: list[str]) -> list[str]:
    """Return deterministic unique build tokens for selected rows."""
    seen: Counter[str] = Counter()
    tokens: list[str] = []
    for row, slug in zip(rows, slugs, strict=True):
        base = f"{slug}-{row.model_id}-{sha256_file(row.render_path)[:12]}"
        seen[base] += 1
        tokens.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return tokens


def copy_content_addressed(source: Path, assets_dir: Path, suffix: str) -> AssetRef:
    """Copy a file to assets/<sha256><suffix> and return its reference."""
    digest = sha256_file(source)
    target = assets_dir / f"{digest}{suffix}"
    if not target.exists():
        shutil.copy2(source, target)
    return AssetRef(url=f"/export/assets/{target.name}", sha256=digest, bytes=target.stat().st_size)


def write_placeholder_thumb(target: Path, label: str) -> None:
    """Write a small WebP placeholder thumbnail."""
    image = Image.new("RGB", (THUMB_WIDTH, 225), "#f7f7f2")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, THUMB_WIDTH - 1, 224), outline="#d1d5db")
    draw.text((18, 92), "Thumbnail unavailable", fill="#111827")
    draw.text((18, 116), label[:44], fill="#4b5563")
    image.save(target, "WEBP", quality=78, method=6)


def generate_webp_thumb(source_svg: Path, target_webp: Path, label: str) -> bool:
    """Rasterize an SVG to a WebP thumbnail, falling back to a placeholder on failure."""
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_png = Path(temp_dir_name) / "thumb.png"
        try:
            subprocess.run(
                ["rsvg-convert", "--width", str(THUMB_WIDTH), "--keep-aspect-ratio", "--format", "png", "--output", str(temp_png), str(source_svg)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=RASTER_TIMEOUT_SECONDS,
            )
            subprocess.run(
                ["cwebp", "-quiet", "-q", "78", str(temp_png), "-o", str(target_webp)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=RASTER_TIMEOUT_SECONDS,
            )
            if not target_webp.exists():
                write_placeholder_thumb(target_webp, label)
                return False
            return True
        except (subprocess.SubprocessError, OSError):
            write_placeholder_thumb(target_webp, label)
            return False


def content_address_generated(path: Path, assets_dir: Path, suffix: str) -> AssetRef:
    """Move a generated file to assets/<sha256><suffix> and return its reference."""
    digest = sha256_file(path)
    target = assets_dir / f"{digest}{suffix}"
    if not target.exists():
        path.replace(target)
    else:
        path.unlink()
    return AssetRef(url=f"/export/assets/{target.name}", sha256=digest, bytes=target.stat().st_size)


def build_model(
    row: ManifestRow,
    slug: str,
    token: str,
    out_root: Path,
    ledger: dict[tuple[str, str], dict[str, Any]],
) -> BuiltModel:
    """Build one model JSON record and its content-addressed assets."""
    ledger_row = ledger.get((row.stable_id, row.recipe_revision_sha256))
    assets_dir = out_root / "assets"
    svg = copy_content_addressed(row.render_path, assets_dir, ".svg")
    thumb_working_path = assets_dir / f".{token}.webp"
    thumbnail_generated = generate_webp_thumb(row.render_path, thumb_working_path, row.name)
    thumb = content_address_generated(thumb_working_path, assets_dir, ".webp")
    monster = svg.bytes > SVG_MONSTER_BYTES or (row.n_nodes is not None and row.n_nodes > OPS_MONSTER_THRESHOLD)
    variant = {
        "key": "unrolled-none",
        "vis_mode": "unrolled",
        "collapse": "none",
        "label": "Unrolled - full",
        "svg": svg.__dict__,
        "thumb": thumb.__dict__,
        "pdf": None,
        "monster": monster,
    }
    record = {
        "stable_id": row.model_id,
        "slug": slug,
        "display_name": row.name,
        "category": row.category,
        "family_normalized": derive_family(row.name),
        "domain": "unknown",
        "zoo": "unknown",
        "source_zoo": "unknown",
        "source_license": "unknown",
        "source_url": None,
        "paper_url": None,
        "render_status": "rendered",
        "forward_validated": ledger_row is not None,
        "input_tier": "none",
        "quarantine": False,
        "n_ops": row.n_nodes if ledger_row is None or ledger_row.get("n_ops") is None else ledger_row["n_ops"],
        "graph_shape_hash": row.graph_shape_hash,
        "param_count": None,
        "is_recurrent": None,
        "default_variant": variant["key"],
        "variants": [variant],
        "tlspec": None,
        "recipe_text": "",
        "torchlens_version": (ledger_row.get("torchlens_version") if ledger_row else None) or "unknown",
        "renderer_version": "unknown",
        "monster_graph": monster,
    }
    return BuiltModel(record=record, thumbnail_generated=thumbnail_generated, thumbnail_placeholder=not thumbnail_generated)


def size_bucket(n_ops: int | None) -> str:
    """Return a stable operation-count size bucket."""
    if n_ops is None:
        return "unknown"
    if n_ops < 50:
        return "small"
    if n_ops < 250:
        return "medium"
    if n_ops < 1000:
        return "large"
    return "monster"


def count_facets(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Return sorted facet counts for one record key."""
    counts = Counter(str(record.get(key) or "unknown") for record in records)
    return [{"value": value, "count": count} for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def build_facets(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build gallery facet metadata."""
    size_counts = Counter(size_bucket(record["n_ops"]) for record in records)
    return {
        "categories": count_facets(records, "category"),
        "families": count_facets(records, "family_normalized"),
        "domains": count_facets(records, "domain"),
        "zoos": count_facets(records, "zoo"),
        "tiers": count_facets(records, "input_tier"),
        "sizes": [{"value": value, "count": count} for value, count in sorted(size_counts.items(), key=lambda item: (-item[1], item[0]))],
    }


def write_json(path: Path, value: Any) -> None:
    """Write stable pretty JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write stable JSON Lines records."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def write_catalog_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write the aggregate flat catalog CSV."""
    fields = [
        "stable_id",
        "slug",
        "display_name",
        "category",
        "family_normalized",
        "domain",
        "zoo",
        "source_zoo",
        "source_license",
        "render_status",
        "forward_validated",
        "input_tier",
        "quarantine",
        "n_ops",
        "graph_shape_hash",
        "param_count",
        "is_recurrent",
        "default_variant",
        "monster_graph",
        "svg_url",
        "svg_sha256",
        "svg_bytes",
        "thumb_url",
        "thumb_sha256",
        "thumb_bytes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            variant = record["variants"][0]
            writer.writerow(
                {
                    **{field: record.get(field) for field in fields if field in record},
                    "svg_url": variant["svg"]["url"],
                    "svg_sha256": variant["svg"]["sha256"],
                    "svg_bytes": variant["svg"]["bytes"],
                    "thumb_url": variant["thumb"]["url"],
                    "thumb_sha256": variant["thumb"]["sha256"],
                    "thumb_bytes": variant["thumb"]["bytes"],
                }
            )


def clean_output(out_root: Path) -> None:
    """Reset generated output directories."""
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "assets").mkdir(parents=True)
    (out_root / "models").mkdir(parents=True)


def real_torchlens_version(records: list[dict[str, Any]]) -> str:
    """Return the most common non-'unknown' torchlens_version among records."""
    counts = Counter(
        record["torchlens_version"] for record in records if record["torchlens_version"] != "unknown"
    )
    if not counts:
        return "unknown"
    return counts.most_common(1)[0][0]


def write_export(out_root: Path, records: list[dict[str, Any]], cataloged: int, rendered_total: int, stats: Counter[str]) -> None:
    """Write all aggregate export files."""
    records.sort(key=lambda record: record["slug"])
    for record in records:
        write_json(out_root / "models" / f"{record['slug']}.json", record)
    write_jsonl(out_root / "models.jsonl", records)
    forward_validated_real = sum(1 for record in records if record["forward_validated"])
    predicate_note = PREDICATE_NOTE if forward_validated_real == 0 else "render coverage plus committed TorchLens validation ledger"
    write_json(
        out_root / "funnel.json",
        {
            "cataloged": cataloged,
            "build_verified": rendered_total,
            "forward_validated_real": forward_validated_real,
            "forward_validated_wrapper": 0,
            "rendered": rendered_total,
            "quarantined": 0,
            "deferred": max(cataloged - rendered_total, 0),
            "predicate_note": predicate_note,
        },
    )
    write_json(
        out_root / "export_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "export_version": EXPORT_VERSION,
            "counts": {
                "models": len(records),
                "cataloged": cataloged,
                "rendered_available": rendered_total,
                "thumbnails_generated": stats["thumbnails_generated"],
                "thumbnail_placeholders": stats["thumbnail_placeholders"],
                "monster_graphs": sum(1 for record in records if record["monster_graph"]),
            },
            "torchlens_version": real_torchlens_version(records),
            "renderer_version": "unknown",
        },
    )
    write_json(out_root / "facets.json", build_facets(records))
    write_catalog_csv(out_root / "catalog.csv", records)


def main() -> None:
    """Build the catalog export."""
    args = parse_args()
    gallery_root = args.gallery.resolve()
    manifest_path = args.manifest.resolve()
    out_root = args.out.resolve()
    ledger = load_ledger(args.ledger)
    rows, cataloged = read_rendered_rows(gallery_root, manifest_path)
    selected = select_rows(rows, args.limit)
    slugs = assign_slugs(selected)
    tokens = unique_tokens(selected, slugs)
    clean_output(out_root)

    stats: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as executor:
        futures = [
            executor.submit(build_model, row, slug, token, out_root, ledger)
            for row, slug, token in zip(selected, slugs, tokens, strict=True)
        ]
        for future in as_completed(futures):
            built = future.result()
            records.append(built.record)
            stats["thumbnails_generated"] += int(built.thumbnail_generated)
            stats["thumbnail_placeholders"] += int(built.thumbnail_placeholder)

    write_export(out_root, records, cataloged, len(rows), stats)
    print(
        json.dumps(
            {
                "models_built": len(records),
                "rendered_available": len(rows),
                "cataloged": cataloged,
                "thumbnails_generated": stats["thumbnails_generated"],
                "thumbnail_placeholders": stats["thumbnail_placeholders"],
                "monster_graphs": sum(1 for record in records if record["monster_graph"]),
                "out": str(out_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
