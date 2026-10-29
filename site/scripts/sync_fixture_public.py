#!/usr/bin/env python3
"""Mirror generated catalog exports into Astro's public directory."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORT_SOURCE = ROOT / "export"
EXPORT_TARGET = ROOT / "public" / "export"
FIXTURE_SOURCE = ROOT / "fixture"
FIXTURE_TARGET = ROOT / "public" / "fixture"


def sync_tree(source: Path, target: Path) -> bool:
    """Copy one generated tree into public/ if the source exists."""
    if not source.exists():
        if target.exists():
            shutil.rmtree(target)
        return False
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return True


def sync_public_exports() -> None:
    """Copy generated export files into public/ for Astro dev and static builds."""
    if os.environ.get("TORCHLENS_SKIP_PUBLIC_FIXTURE") == "1":
        for target in (EXPORT_TARGET, FIXTURE_TARGET):
            if target.exists():
                shutil.rmtree(target)
        print("Skipped public export sync; assets are treated as external URLs.")
        return
    synced = []
    if sync_tree(EXPORT_SOURCE, EXPORT_TARGET):
        synced.append((EXPORT_SOURCE, EXPORT_TARGET))
    if sync_tree(FIXTURE_SOURCE, FIXTURE_TARGET):
        synced.append((FIXTURE_SOURCE, FIXTURE_TARGET))
    for source, target in synced:
        print(f"Synced {source.relative_to(ROOT)} to {target.relative_to(ROOT)}")


def main() -> None:
    """Run the public export sync."""
    sync_public_exports()


if __name__ == "__main__":
    main()
