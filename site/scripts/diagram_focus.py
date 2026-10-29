#!/usr/bin/env python3
"""Find where each wide diagram's drawing starts, so the viewer can open on it.

The viewer opens a diagram at reading scale at the top of the graph. A graph wider than the
screen at that scale shows only part of its top row, and for some graphs (gpt2, for example)
the middle of the top row is blank paper. For every SVG wider than WIDE_PX this script finds
the topmost nodes and records the horizontal position of their median as a fraction of the
diagram width. Only positions away from the centre are stored; the viewer centres the rest.

Reads SVGs from an export's assets directory, or fetches them from the public asset base.
Writes `site-data/diagram_focus.json`: {"version": 1, "focus": {"<svg sha256>": fraction}}.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FOCUS_VERSION = 1
# Natural width (CSS px) above which a phone at the viewer's smallest reading scale scrolls sideways.
WIDE_PX = 560
# Above this width the viewer opens on the whole graph instead (WIDE_GRAPH_READABLE_PX / 0.6).
WHOLE_VIEW_PX = 6667
# How many of the topmost nodes vote on the position.
TOP_NODES = 8
# Positions closer to the centre than this (fraction of width) are left to the viewer's default.
MIN_OFF_CENTRE = 0.05
HEADER_BYTES = 4096
PT_TO_PX = 4 / 3
USER_AGENT = "menagerie-site-diagram-focus/1"
FETCH_ATTEMPTS = 6

SVG_SIZE = re.compile(r'<svg[^>]*\swidth="([\d.]+)(pt|px)?"[^>]*\sheight="([\d.]+)(pt|px)?"[^>]*viewBox="([-\d.\s]+)"', re.S)
GRAPH_TRANSLATE = re.compile(r'class="graph"[^>]*transform="[^"]*translate\(([-\d.]+)[ ,]+([-\d.]+)\)')
NODE_GROUP = re.compile(r'<g id="[^"]*" class="node">(.*?)</g>', re.S)
SHAPE = re.compile(r"<(ellipse|polygon|polyline|path)\b([^>]*)>")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site-data", type=Path, default=Path("site-data"), help="Tracked site data directory.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--assets-dir", type=Path, help="Export assets directory holding <sha256>.svg files.")
    source.add_argument("--asset-base", help="Public asset base URL (PUBLIC_R2_BASE).")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel reads or downloads.")
    return parser.parse_args()


def svg_hashes(site_data: Path) -> list[str]:
    """Return every distinct SVG hash referenced by the catalog."""
    hashes: set[str] = set()
    with (site_data / "models.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                for variant in json.loads(line)["variants"]:
                    hashes.add(variant["svg"]["sha256"])
    return sorted(hashes)


def natural_width_px(header: str) -> float | None:
    """Return the browser's natural width of an SVG from its opening tag, or None."""
    match = SVG_SIZE.search(header)
    if not match:
        return None
    width, unit = float(match.group(1)), match.group(2) or "px"
    return width * PT_TO_PX if unit == "pt" else width


def shape_centre(kind: str, attrs: str) -> tuple[float, float] | None:
    """Return the centre of one Graphviz node shape, in graph coordinates."""
    if kind == "ellipse":
        cx = re.search(r'\scx="([-\d.]+)"', attrs)
        cy = re.search(r'\scy="([-\d.]+)"', attrs)
        return (float(cx.group(1)), float(cy.group(1))) if cx and cy else None
    field = re.search(r'\s(?:points|d)="([^"]*)"', attrs)
    if not field:
        return None
    values = [float(value) for value in NUMBER.findall(field.group(1))]
    xs, ys = values[0::2], values[1::2]
    if not xs or len(xs) != len(ys):
        return None
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)


def focus_fraction(svg: str) -> float | None:
    """Return the median x of the topmost nodes as a fraction of the diagram width, or None."""
    size = SVG_SIZE.search(svg)
    if not size:
        return None
    view_box = [float(value) for value in size.group(5).split()]
    view_width = view_box[2] if len(view_box) == 4 else 0.0
    translate = GRAPH_TRANSLATE.search(svg)
    offset_x = float(translate.group(1)) if translate else 0.0
    centres: list[tuple[float, float]] = []
    for group in NODE_GROUP.finditer(svg):
        shape = SHAPE.search(group.group(1))
        centre = shape_centre(shape.group(1), shape.group(2)) if shape else None
        if centre:
            centres.append(centre)
    if view_width <= 0 or not centres:
        return None
    centres.sort(key=lambda centre: centre[1])
    median_x = statistics.median(x for x, _ in centres[:TOP_NODES])
    return min(1.0, max(0.0, (median_x + offset_x) / view_width))


def read_local(assets_dir: Path, sha: str, limit: int | None) -> str | None:
    """Read an SVG (or its first `limit` bytes) from an export assets directory."""
    path = assets_dir / f"{sha}.svg"
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return handle.read(limit if limit else -1).decode("utf-8", "replace")


def read_remote(asset_base: str, sha: str, limit: int | None) -> str | None:
    """Fetch an SVG (or its first `limit` bytes, by range request) from the public asset base."""
    # Cloudflare refuses urllib's default user agent.
    request = urllib.request.Request(f"{asset_base.rstrip('/')}/{sha}.svg", headers={"User-Agent": USER_AGENT})
    if limit:
        request.add_header("Range", f"bytes=0-{limit - 1}")
    for attempt_number in range(1, FETCH_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read().decode("utf-8", "replace")
        except OSError as error:
            print(f"read {sha} attempt {attempt_number}: {error}", file=sys.stderr)
            time.sleep(3 * attempt_number)  # r2.dev answers 429 under load
    return None


class ReadError(Exception):
    """An SVG could not be read."""


def measure(args: argparse.Namespace, sha: str) -> tuple[str, float | None]:
    """Return the stored focus for one SVG, or None when the viewer default is right."""
    read = (lambda limit: read_local(args.assets_dir, sha, limit)) if args.assets_dir else (lambda limit: read_remote(args.asset_base, sha, limit))
    header = read(HEADER_BYTES)
    if header is None:
        raise ReadError(sha)
    width = natural_width_px(header)
    if width is None or not WIDE_PX < width <= WHOLE_VIEW_PX:
        return sha, None
    svg = read(None)
    if svg is None:
        raise ReadError(sha)
    fraction = focus_fraction(svg)
    if fraction is None or abs(fraction - 0.5) < MIN_OFF_CENTRE:
        return sha, None
    return sha, round(fraction, 3)


def attempt(args: argparse.Namespace, sha: str) -> float | None | ReadError:
    """Measure one SVG, returning the read error instead of raising it."""
    try:
        return measure(args, sha)[1]
    except ReadError as error:
        return error


def main() -> None:
    """Measure every catalog SVG and write the focus file."""
    args = parse_args()
    hashes = svg_hashes(args.site_data)
    results: dict[str, float | None] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for sha, outcome in zip(hashes, pool.map(lambda sha: attempt(args, sha), hashes)):
            if isinstance(outcome, ReadError):
                failed.append(sha)
            else:
                results[sha] = outcome
    if failed:
        sys.exit(f"{len(failed)} of {len(hashes)} diagrams could not be read (first: {failed[0]}); nothing written")
    focus = {sha: value for sha, value in sorted(results.items()) if value is not None}
    target = args.site_data / "diagram_focus.json"
    target.write_text(json.dumps({"version": FOCUS_VERSION, "focus": focus}, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"measured {len(hashes)} diagrams; {len(focus)} open off centre; wrote {target}")


if __name__ == "__main__":
    main()
