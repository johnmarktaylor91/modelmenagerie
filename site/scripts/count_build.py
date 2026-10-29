#!/usr/bin/env python3
"""Count a built Pages tree; no network, upload, deployment or server is started."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

FREE_FILES = 20_000
PAID_FILES = 100_000
MAX_ASSET_BYTES = 25 * 1024 * 1024
LIMITS_SOURCE = "https://developers.cloudflare.com/pages/platform/limits/"


def _kind(path: Path) -> str:
    if "pagefind" in path.parts or path.suffix in (".pf_fragment", ".pf_index", ".pf_meta"):
        return "search_fragments"
    return "html" if path.suffix == ".html" else "assets"


def _scan(root: Path) -> tuple[Counter, dict, list]:
    counts: Counter[str] = Counter()
    largest: dict[str, Any] = {"path": None, "bytes": 0}
    oversized = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in build: {path.relative_to(root)}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        counts[_kind(path.relative_to(root))] += 1
        if size > largest["bytes"] or largest["path"] is None:
            largest = {"path": relative, "bytes": size}
        if size > MAX_ASSET_BYTES:
            oversized.append({"path": relative, "bytes": size})
    return counts, largest, oversized


def count_build(build_dir: Path, *, out: Path | None = None) -> dict[str, Any]:
    """Report all files and sizes; refuse symlinks, empty builds and a report inside the tree."""
    root = Path(build_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("build_dir must be a directory")
    if out and Path(out).resolve().is_relative_to(root):
        raise ValueError("the report must be outside the build tree")
    counts, largest, oversized = _scan(root)
    total = sum(counts.values())
    if total == 0 or counts["html"] == 0:
        raise ValueError("no HTML build files found")
    report = {
        "files": total, "html": counts["html"], "assets": counts["assets"],
        "search_fragments": counts["search_fragments"], "largest_asset": largest,
        "oversized_assets": oversized,
        "limits": {"free_files": FREE_FILES, "paid_files": PAID_FILES,
                   "max_asset_bytes": MAX_ASSET_BYTES, "source": LIMITS_SOURCE},
        "free": {"fits": total <= FREE_FILES and not oversized, "headroom": FREE_FILES - total},
        "paid": {"fits": total <= PAID_FILES and not oversized, "headroom": PAID_FILES - total,
                 "requires": "PAGES_WRANGLER_MAJOR_VERSION=4"},
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    print(json.dumps(count_build(args.build_dir, out=args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
