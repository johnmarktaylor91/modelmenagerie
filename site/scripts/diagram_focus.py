#!/usr/bin/env python3
"""Find where each wide diagram's drawing starts, so the viewer can open on it.

The viewer opens a diagram at reading scale at the top of the graph. A graph wider than the
screen at that scale shows only part of its top row, and for some graphs (gpt2, for example)
the middle of the top row is blank paper. For every SVG wider than WIDE_PX this script finds
the topmost nodes and records the horizontal position of their median as a fraction of the
diagram width (for graphs too wide to show whole, the median of the busiest stretch of the top
band). Only positions away from the centre are stored; the viewer centres the rest.

A graph's top row can also sit far from where the reader lands: gpt2's grouped drawing has one output
node at its far right top corner, so the window the viewer opens on (centred at the focus) starts with
a band of blank paper. For those the script also records where the first node inside that window
begins, as a fraction of the diagram height, and the viewer opens scrolled down to it.

Reads SVGs from an export's assets directory, or fetches them from the public asset base.
Writes `site-data/diagram_focus.json`:
{"version": 2, "focus": {"<svg sha256>": x fraction}, "focus_y": {"<svg sha256>": y fraction}}.
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

FOCUS_VERSION = 2
# Natural width (CSS px) above which a phone at the viewer's smallest reading scale scrolls sideways.
WIDE_PX = 560
# How many of the topmost nodes vote on the position.
TOP_NODES = 8
# Positions closer to the centre than this (fraction of width) are left to the viewer's default.
MIN_OFF_CENTRE = 0.05
# Natural width (CSS px) above which the viewer cannot show the whole graph legibly. For these the
# topmost nodes can be a long row of inputs and constants far from where the operations start (ACDNet's
# inputs sit 27,000 px right of its first convolutions), so the focus is the densest stretch of the top
# band instead: the median x of the nodes in the WINDOW_PT-wide stretch of the band holding the most.
DENSE_FROM_PX = 6667
BAND_PT = 900.0
WINDOW_PT = 900.0
# Half the width (pt) a phone shows at the viewer's smallest reading scale: a 358 px stage at 60% is
# about 447 pt of diagram. The vertical start is the top of the first node inside this window.
WINDOW_HALF_PT = 224.0
# Space kept above that node (pt), and the least blank band (pt) worth scrolling past.
TOP_MARGIN_PT = 12.0
MIN_BLANK_PT = 60.0
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


def shape_box(kind: str, attrs: str) -> tuple[float, float, float, float] | None:
    """Return (left, top, right, bottom) of one Graphviz node shape, in graph coordinates."""
    if kind == "ellipse":
        values = {name: re.search(rf'\s{name}="([-\d.]+)"', attrs) for name in ("cx", "cy", "rx", "ry")}
        if not (values["cx"] and values["cy"]):
            return None
        cx, cy = float(values["cx"].group(1)), float(values["cy"].group(1))
        rx = float(values["rx"].group(1)) if values["rx"] else 0.0
        ry = float(values["ry"].group(1)) if values["ry"] else 0.0
        return (cx - rx, cy - ry, cx + rx, cy + ry)
    field = re.search(r'\s(?:points|d)="([^"]*)"', attrs)
    if not field:
        return None
    values = [float(value) for value in NUMBER.findall(field.group(1))]
    xs, ys = values[0::2], values[1::2]
    if not xs or len(xs) != len(ys):
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def shape_centre(kind: str, attrs: str) -> tuple[float, float] | None:
    """Return the centre of one Graphviz node shape, in graph coordinates."""
    box = shape_box(kind, attrs)
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) if box else None


class Layout:
    """Node boxes of one SVG, with the transform from graph coordinates to the viewBox."""

    def __init__(self, svg: str) -> None:
        size = SVG_SIZE.search(svg)
        view_box = [float(value) for value in size.group(5).split()] if size else []
        self.view_width = view_box[2] if len(view_box) == 4 else 0.0
        self.view_height = view_box[3] if len(view_box) == 4 else 0.0
        translate = GRAPH_TRANSLATE.search(svg)
        self.offset_x = float(translate.group(1)) if translate else 0.0
        self.offset_y = float(translate.group(2)) if translate else 0.0
        self.wide_px = natural_width_px(svg[:HEADER_BYTES]) or 0.0
        self.boxes: list[tuple[float, float, float, float]] = []
        for group in NODE_GROUP.finditer(svg):
            shape = SHAPE.search(group.group(1))
            box = shape_box(shape.group(1), shape.group(2)) if shape else None
            if box:
                self.boxes.append(box)

    @property
    def usable(self) -> bool:
        """True when the SVG has a size and at least one node."""
        return self.view_width > 0 and self.view_height > 0 and bool(self.boxes)


def focus_fraction(svg: str) -> float | None:
    """Return the median x of the topmost nodes as a fraction of the diagram width, or None."""
    layout = Layout(svg)
    return layout_focus_x(layout) if layout.usable else None


def layout_focus_x(layout: Layout) -> float:
    """Return the horizontal focus of a usable layout as a fraction of its width."""
    centres = sorted((((left + right) / 2, (top + bottom) / 2) for left, top, right, bottom in layout.boxes), key=lambda centre: centre[1])
    if layout.wide_px > DENSE_FROM_PX:
        median_x = densest_top_x(centres)
    else:
        median_x = statistics.median(x for x, _ in centres[:TOP_NODES])
    return min(1.0, max(0.0, (median_x + layout.offset_x) / layout.view_width))


def layout_focus_y(layout: Layout, focus_x: float) -> float | None:
    """Return where the viewer's opening window should start, as a fraction of the height, or None.

    The window is WINDOW_HALF_PT either side of `focus_x` (a width fraction, the point the viewer
    centres on). None when the first node in it is already near the top.
    """
    centre = focus_x * layout.view_width - layout.offset_x
    tops = [top for left, top, right, _ in layout.boxes if right >= centre - WINDOW_HALF_PT and left <= centre + WINDOW_HALF_PT]
    if not tops:
        return None
    start = min(tops) + layout.offset_y - TOP_MARGIN_PT
    if start < MIN_BLANK_PT:
        return None
    return min(1.0, start / layout.view_height)


def densest_top_x(centres: list[tuple[float, float]]) -> float:
    """Return the median x of the nodes in the busiest WINDOW_PT stretch of the top BAND_PT band."""
    top = centres[0][1]
    xs = sorted(x for x, y in centres if y - top <= BAND_PT)
    best_start, best_count, end = 0, 0, 0
    for start, x in enumerate(xs):
        while end < len(xs) and xs[end] - x <= WINDOW_PT:
            end += 1
        if end - start > best_count:
            best_start, best_count = start, end - start
    return statistics.median(xs[best_start : best_start + best_count])


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


def measure(args: argparse.Namespace, sha: str) -> tuple[str, tuple[float | None, float | None]]:
    """Return the stored (x, y) focus for one SVG; each is None when the viewer default is right."""
    read = (lambda limit: read_local(args.assets_dir, sha, limit)) if args.assets_dir else (lambda limit: read_remote(args.asset_base, sha, limit))
    header = read(HEADER_BYTES)
    if header is None:
        raise ReadError(sha)
    width = natural_width_px(header)
    # Every wide graph, however wide: the viewer opens even the widest at reading scale (its whole-graph
    # fit would be too small to read), so it needs to know where the drawing starts.
    if width is None or width <= WIDE_PX:
        return sha, (None, None)
    svg = read(None)
    if svg is None:
        raise ReadError(sha)
    layout = Layout(svg)
    if not layout.usable:
        return sha, (None, None)
    fraction = layout_focus_x(layout)
    stored_x = None if abs(fraction - 0.5) < MIN_OFF_CENTRE else round(fraction, 3)
    # The window the viewer opens on is centred at the stored focus, or the middle when none is stored.
    start = layout_focus_y(layout, 0.5 if stored_x is None else stored_x)
    return sha, (stored_x, None if start is None else round(start, 3))


def attempt(args: argparse.Namespace, sha: str) -> tuple[float | None, float | None] | ReadError:
    """Measure one SVG, returning the read error instead of raising it."""
    try:
        return measure(args, sha)[1]
    except ReadError as error:
        return error


def main() -> None:
    """Measure every catalog SVG and write the focus file."""
    args = parse_args()
    hashes = svg_hashes(args.site_data)
    results: dict[str, tuple[float | None, float | None]] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for sha, outcome in zip(hashes, pool.map(lambda sha: attempt(args, sha), hashes)):
            if isinstance(outcome, ReadError):
                failed.append(sha)
            else:
                results[sha] = outcome
    if failed:
        sys.exit(f"{len(failed)} of {len(hashes)} diagrams could not be read (first: {failed[0]}); nothing written")
    focus = {sha: xy[0] for sha, xy in sorted(results.items()) if xy[0] is not None}
    focus_y = {sha: xy[1] for sha, xy in sorted(results.items()) if xy[1] is not None}
    target = args.site_data / "diagram_focus.json"
    payload = {"version": FOCUS_VERSION, "focus": focus, "focus_y": focus_y}
    target.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"measured {len(hashes)} diagrams; {len(focus)} open off centre; {len(focus_y)} open below the top; wrote {target}")


if __name__ == "__main__":
    main()
