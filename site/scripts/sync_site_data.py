#!/usr/bin/env python3
"""Sync lightweight catalog data from a generated export into site-data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DATA_FILES = (
    "models.jsonl",
    "facets.json",
    "funnel.json",
    "export_manifest.json",
    "catalog.csv",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=Path("export"), help="Generated catalog export directory.")
    parser.add_argument("--site-data", type=Path, default=Path("site-data"), help="Tracked site data output directory.")
    return parser.parse_args()


def require_source_files(export_dir: Path) -> None:
    """Raise if any required aggregate data file is missing."""
    missing = [name for name in DATA_FILES if not (export_dir / name).is_file()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing required export data file(s): {missing_list}")


def sync_site_data(export_dir: Path, site_data_dir: Path) -> list[Path]:
    """Copy aggregate text data files into the tracked site data directory."""
    export_dir = export_dir.resolve()
    site_data_dir = site_data_dir.resolve()
    require_source_files(export_dir)
    site_data_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for name in DATA_FILES:
        source = export_dir / name
        target = site_data_dir / name
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def main() -> None:
    """Run the site data sync."""
    args = parse_args()
    copied = sync_site_data(args.export_dir, args.site_data)
    for path in copied:
        print(f"synced {path}")


if __name__ == "__main__":
    main()
