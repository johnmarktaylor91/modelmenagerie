"""Packaging checks: runtime data ships with non-editable installs."""

from __future__ import annotations

import fnmatch
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DATA_DIRS = ("menagerie/data", "menagerie/locks", "menagerie/pipeline/assets")


def _tracked_files(directory: str) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", directory], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line]


def test_package_data_covers_runtime_data() -> None:
    """Every tracked file under menagerie/data and menagerie/locks matches a package-data glob."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = pyproject["tool"]["setuptools"]["package-data"]["menagerie"]
    missing = [
        path
        for directory in RUNTIME_DATA_DIRS
        for path in _tracked_files(directory)
        if not any(fnmatch.fnmatch(path.removeprefix("menagerie/"), p) for p in patterns)
    ]
    assert not missing, f"not shipped in wheels: {missing[:5]}"
