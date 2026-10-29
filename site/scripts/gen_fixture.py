#!/usr/bin/env python3
"""Generate deterministic synthetic TorchLens Menagerie fixture data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEED = 20260622
ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "fixture" / "export" / "v1"
FAMILIES = [
    "ResNet",
    "ViT",
    "U-Net",
    "Swin",
    "YOLO",
    "DiT",
    "BERT",
    "ConvNeXt",
    "EfficientNet",
    "Whisper",
    "GraphSAGE",
    "CLIP",
]
DOMAINS = ["vision", "generative", "nlp", "audio", "graph", "multimodal", "medical"]
ZOOS = ["timm", "torchvision", "classics", "huggingface", "research", "segmentation_models"]
TIERS = ["none", "wrapper", "real"]
LICENSES = ["MIT", "Apache-2.0", "BSD-3-Clause", "CC-BY-4.0", "custom research license"]


@dataclass(frozen=True)
class AssetInfo:
    """Metadata for one generated asset."""

    url: str
    sha256: str
    bytes: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed fixture generation options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60, help="Number of models to emit.")
    return parser.parse_args()


def slugify(value: str) -> str:
    """Convert a display name to a stable URL slug.

    Parameters
    ----------
    value
        Human-readable model name.

    Returns
    -------
    str
        Lowercase slug containing letters, digits, and dashes.
    """
    normalized = value.lower().replace("+", " plus ").replace("/", " ")
    chars = [char if char.isalnum() else "-" for char in normalized]
    slug = "-".join(part for part in "".join(chars).split("-") if part)
    return slug


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    """Create a PNG chunk with CRC.

    Parameters
    ----------
    kind
        Four-byte PNG chunk kind.
    payload
        Chunk payload bytes.

    Returns
    -------
    bytes
        Encoded PNG chunk.
    """
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def write_png(path: Path, width: int, height: int, seed: int, node_count: int) -> AssetInfo:
    """Write a deterministic diagram-like PNG thumbnail.

    Parameters
    ----------
    path
        Destination PNG path.
    width
        Image width in pixels.
    height
        Image height in pixels.
    seed
        Local random seed for visual variation.
    node_count
        Approximate graph size represented by the thumbnail.

    Returns
    -------
    AssetInfo
        Metadata for the written PNG.
    """
    rng = random.Random(seed)
    bg = (13, 17, 23)
    stroke = (88, 166, 255)
    fill = (22, 27, 34)
    grid = (48, 54, 61)
    rows: list[bytes] = []
    pixels = [[bg for _ in range(width)] for _ in range(height)]
    for x in range(0, width, 24):
        for y in range(height):
            pixels[y][x] = grid
    for y in range(0, height, 24):
        for x in range(width):
            pixels[y][x] = grid
    boxes = min(max(node_count, 4), 48)
    for i in range(boxes):
        x0 = 10 + (i * 37 + rng.randrange(12)) % max(width - 42, 1)
        y0 = 10 + (i * 23 + rng.randrange(12)) % max(height - 28, 1)
        box_w = 28 + (i % 3) * 8
        box_h = 14
        for y in range(y0, min(y0 + box_h, height)):
            for x in range(x0, min(x0 + box_w, width)):
                edge = y in {y0, y0 + box_h - 1} or x in {x0, x0 + box_w - 1}
                pixels[y][x] = stroke if edge else fill
        if i > 0:
            x1 = max(0, x0 - 20)
            y1 = min(height - 1, y0 + box_h // 2)
            for x in range(x1, x0):
                pixels[y1][x] = stroke
    for row in pixels:
        rows.append(b"\x00" + bytes(channel for pixel in row for channel in pixel))
    raw = b"".join(rows)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += png_chunk(b"IDAT", zlib.compress(raw, 9))
    payload += png_chunk(b"IEND", b"")
    path.write_bytes(payload)
    return asset_info(path)


def write_svg(path: Path, name: str, stable_id: str, node_count: int, monster: bool) -> AssetInfo:
    """Write a deterministic graphviz-style SVG diagram.

    Parameters
    ----------
    path
        Destination SVG path.
    name
        Model display name.
    stable_id
        Stable model identifier.
    node_count
        Number of synthetic graph nodes.
    monster
        Whether to use dense monster-graph styling.

    Returns
    -------
    AssetInfo
        Metadata for the written SVG.
    """
    cols = 16 if monster else 6
    box_w = 116 if monster else 128
    box_h = 34
    gap_x = 28
    gap_y = 26
    rows = (node_count + cols - 1) // cols
    width = cols * (box_w + gap_x) + 40
    height = rows * (box_h + gap_y) + 92
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{escape_xml(name)} TorchLens architecture diagram</title>",
        "<defs>",
        '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L8,4 L0,8 Z" fill="#58a6ff"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        f'<text x="20" y="30" fill="#e6edf3" font-family="JetBrains Mono, monospace" font-size="18">{escape_xml(name)}</text>',
    ]
    centers: list[tuple[int, int]] = []
    for index in range(node_count):
        col = index % cols
        row = index // cols
        x = 20 + col * (box_w + gap_x)
        y = 58 + row * (box_h + gap_y)
        centers.append((x + box_w // 2, y + box_h // 2))
        label = f"{stable_id}.op{index:03d}" if monster else f"op{index:02d}"
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="6" fill="#161b22" stroke="#30363d"/>'
        )
        parts.append(
            f'<text x="{x + 12}" y="{y + 22}" fill="#e6edf3" font-family="JetBrains Mono, monospace" font-size="11">{label}</text>'
        )
    for index in range(1, node_count):
        x1, y1 = centers[index - 1]
        x2, y2 = centers[index]
        parts.append(
            f'<path d="M{x1 + box_w // 2 - 4},{y1} C{x1 + 64},{y1} {x2 - 64},{y2} {x2 - box_w // 2 + 4},{y2}" '
            'fill="none" stroke="#58a6ff" stroke-width="1.4" marker-end="url(#arrow)"/>'
        )
        if monster and index > 7 and index % 9 == 0:
            bx, by = centers[index - 7]
            parts.append(
                f'<path d="M{bx},{by + 17} C{bx},{by + 44} {x2},{y2 - 44} {x2},{y2 - 17}" '
                'fill="none" stroke="#3fb950" stroke-width="0.8" opacity="0.55"/>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return asset_info(path)


def escape_xml(value: str) -> str:
    """Escape text for XML element content.

    Parameters
    ----------
    value
        Text to escape.

    Returns
    -------
    str
        XML-safe text.
    """
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def asset_info(path: Path) -> AssetInfo:
    """Return metadata for an existing asset.

    Parameters
    ----------
    path
        Asset path under the export root.

    Returns
    -------
    AssetInfo
        URL, sha256, and byte size metadata.
    """
    payload = path.read_bytes()
    rel = path.relative_to(ROOT).as_posix()
    return AssetInfo(url=f"/{rel}", sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def weighted_choice(rng: random.Random, values: list[str], weights: list[int]) -> str:
    """Select a deterministic weighted value.

    Parameters
    ----------
    rng
        Random generator.
    values
        Candidate values.
    weights
        Integer weights aligned with values.

    Returns
    -------
    str
        Selected value.
    """
    return rng.choices(values, weights=weights, k=1)[0]


def model_record(index: int, n_models: int, rng: random.Random) -> tuple[dict[str, Any], int, bool]:
    """Create deterministic metadata for one synthetic model.

    Parameters
    ----------
    index
        Zero-based model index.
    n_models
        Total number of fixture models.
    rng
        Random generator.

    Returns
    -------
    tuple[dict[str, Any], int, bool]
        Metadata record, node count, and monster flag.
    """
    family = weighted_choice(rng, FAMILIES, [18, 14, 10, 8, 8, 6, 7, 8, 6, 5, 4, 6])
    domain_map = {
        "ResNet": "vision",
        "ViT": "vision",
        "U-Net": "medical",
        "Swin": "vision",
        "YOLO": "vision",
        "DiT": "generative",
        "BERT": "nlp",
        "ConvNeXt": "vision",
        "EfficientNet": "vision",
        "Whisper": "audio",
        "GraphSAGE": "graph",
        "CLIP": "multimodal",
    }
    domain = domain_map[family] if rng.random() < 0.82 else rng.choice(DOMAINS)
    zoo = weighted_choice(rng, ZOOS, [24, 18, 12, 20, 14, 12])
    year = rng.choice([None, *range(2015, 2026)])
    tier = weighted_choice(rng, TIERS, [74, 18, 8])
    forward = tier in {"real", "wrapper"} and rng.random() < (0.85 if tier == "real" else 0.65)
    stable_id = f"m{index + 1}"
    variant = rng.choice(["Tiny", "Small", "Base", "Large", "XL", "V2", "Hybrid", "Lite"])
    display_name = f"{family}-{variant}-{index + 1:04d}"
    slug = slugify(display_name)
    monster = index in {7, 31, 53} if n_models <= 100 else index % 997 == 0
    node_count = rng.randint(8, 38)
    if monster:
        node_count = rng.randint(180, 340) if n_models <= 100 else 96
    if n_models > 1000:
        node_count = 5 if not monster else 16
    n_ops = node_count + rng.randint(4, 140)
    params = rng.choice([None, rng.randint(1_000_000, 800_000_000)])
    record: dict[str, Any] = {
        "stable_id": stable_id,
        "slug": slug,
        "display_name": display_name,
        "family_normalized": family,
        "domain": domain,
        "zoo": zoo,
        "era_raw": "unknown" if year is None else str(year),
        "year": year,
        "year_confidence": "none" if year is None else rng.choice(["high", "medium", "inferred"]),
        "source_zoo": zoo,
        "source_license": rng.choice(LICENSES),
        "source_url": f"https://example.invalid/{zoo}/{slug}",
        "paper_url": None if rng.random() < 0.38 else f"https://arxiv.org/abs/{rng.randint(1501, 2605)}.{rng.randint(1000, 9999)}",
        "render_status": "rendered",
        "forward_validated": forward,
        "input_tier": tier,
        "quarantine": rng.random() < 0.045,
        "n_ops": n_ops,
        "graph_shape_hash": hashlib.sha256(f"{stable_id}:{family}:{node_count}".encode()).hexdigest()[:16],
        "param_count": params,
        "is_recurrent": rng.choice([False, False, False, True, None]),
        "input_shape_label": rng.choice(["1x3x224x224", "1x3x512x512", "1x77", "1x80x3000", "synthetic"]),
        "input_dtype_label": rng.choice(["float32", "float16", "int64"]),
        "recipe_text": f"tl.trace({slug.replace('-', '_')}, x).log.draw()",
        "torchlens_version": "0.1.18",
        "renderer_version": "fixture-renderer-0.3",
        "monster_graph": monster,
    }
    return record, node_count, monster


def write_catalog(records: list[dict[str, Any]]) -> None:
    """Write catalog JSONL and CSV files.

    Parameters
    ----------
    records
        Fixture model records.
    """
    jsonl_path = EXPORT_ROOT / "models.jsonl"
    csv_path = EXPORT_ROOT / "catalog.csv"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for record in records:
            jsonl.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    scalar_fields = [key for key in records[0] if key != "assets"]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[*scalar_fields, "assets"])
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in scalar_fields}
            row["assets"] = json.dumps(record["assets"], sort_keys=True)
            writer.writerow(row)


def build_funnel(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build honest-looking funnel counts from generated records.

    Parameters
    ----------
    records
        Fixture model records.

    Returns
    -------
    dict[str, Any]
        Funnel summary.
    """
    cataloged = len(records)
    rendered = sum(1 for record in records if record["render_status"] == "rendered")
    real = sum(1 for record in records if record["forward_validated"] and record["input_tier"] == "real")
    wrapper = sum(1 for record in records if record["forward_validated"] and record["input_tier"] == "wrapper")
    quarantined = sum(1 for record in records if record["quarantine"])
    return {
        "cataloged": cataloged,
        "build_verified": rendered,
        "forward_validated_real": real,
        "forward_validated_wrapper": wrapper,
        "rendered": rendered,
        "quarantined": quarantined,
        "deferred": max(cataloged - real - wrapper - quarantined, 0),
        "predicate_note": "Synthetic fixture counts: rendering is broad; real forward validation is intentionally sparse.",
    }


def write_support_files(records: list[dict[str, Any]]) -> None:
    """Write non-catalog fixture support files.

    Parameters
    ----------
    records
        Fixture model records.
    """
    funnel = build_funnel(records)
    assets = []
    for record in records:
        for kind, info in record["assets"].items():
            if info is not None:
                assets.append({"stable_id": record["stable_id"], "kind": kind, **info})
    (EXPORT_ROOT / "funnel.json").write_text(json.dumps(funnel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (EXPORT_ROOT / "assets_index.json").write_text(json.dumps(assets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    retired = {
        "legacy-resnet-50-a": records[0]["stable_id"],
        "draft-vit-observatory": records[min(4, len(records) - 1)]["stable_id"],
    }
    (EXPORT_ROOT / "retired_ids.json").write_text(json.dumps(retired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "synthetic-seam-0.1",
        "export_version": "v1",
        "counts": {"models": len(records), "assets": len(assets), **{f"funnel_{key}": value for key, value in funnel.items() if isinstance(value, int)}},
        "torchlens_version": "0.1.18",
        "renderer_version": "fixture-renderer-0.3",
    }
    (EXPORT_ROOT / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate(n_models: int) -> None:
    """Generate a full fixture export.

    Parameters
    ----------
    n_models
        Number of model records to generate.
    """
    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)
    (EXPORT_ROOT / "assets" / "svg").mkdir(parents=True)
    (EXPORT_ROOT / "assets" / "thumb").mkdir(parents=True)
    (EXPORT_ROOT / "assets" / "tlspec").mkdir(parents=True)
    (EXPORT_ROOT / "models").mkdir(parents=True)
    rng = random.Random(SEED + n_models)
    records = []
    for index in range(n_models):
        record, node_count, monster = model_record(index, n_models, rng)
        svg_path = EXPORT_ROOT / "assets" / "svg" / f"{record['slug']}.svg"
        thumb_path = EXPORT_ROOT / "assets" / "thumb" / f"{record['slug']}.png"
        tlspec_path = EXPORT_ROOT / "assets" / "tlspec" / f"{record['slug']}.json"
        svg = write_svg(svg_path, record["display_name"], record["stable_id"], node_count, monster)
        thumb = write_png(thumb_path, 480 if n_models <= 1000 else 160, 270 if n_models <= 1000 else 90, SEED + index, node_count)
        tlspec = {
            "stable_id": record["stable_id"],
            "slug": record["slug"],
            "ops": [f"op_{op:03d}" for op in range(min(node_count, 48 if n_models > 1000 else node_count))],
            "truncated": n_models > 1000 and node_count > 48,
        }
        tlspec_path.write_text(json.dumps(tlspec, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        record["assets"] = {
            "svg": svg.__dict__,
            "thumb": thumb.__dict__,
            "pdf": None,
            "tlspec": asset_info(tlspec_path).__dict__,
        }
        (EXPORT_ROOT / "models" / f"{record['slug']}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        records.append(record)
    write_catalog(records)
    write_support_files(records)


def main() -> None:
    """Run fixture generation from the command line."""
    args = parse_args()
    generate(args.n)
    print(f"Generated {args.n} synthetic models in {EXPORT_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
