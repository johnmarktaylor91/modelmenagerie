"""End-to-end ownership test for the permanent stateful rehearsal fixture."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_FIXTURE_ARCHIVE = Path(__file__).parent / "fixtures" / "p1_rehearsal" / "pilot-records.tar.gz"

# The archive is 4.9MB of campaign records and stays OUT of the public repo
# (only its digest manifest is committed). Operators materialize it with:
#   python tools/build_rehearsal_fixture.py --campaign <pilot-root>
pytestmark = pytest.mark.skipif(
    not _FIXTURE_ARCHIVE.is_file(),
    reason="rehearsal fixture archive not materialized; build with tools/build_rehearsal_fixture.py",
)


@pytest.mark.slow
def test_permanent_rehearsal_smallest_slice_end_to_end(tmp_path: Path) -> None:
    """The frozen fixture validates and closes one wave-scheduler slice."""

    repo_root = Path(__file__).parents[3]
    attestation = tmp_path / "attestation.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        (
            sys.executable,
            str(repo_root / "tools" / "run_rehearsal.py"),
            "--repo-root",
            str(repo_root),
            "--python",
            sys.executable,
            "--scheduler",
            "waves",
            "--attestation",
            str(attestation),
        ),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(attestation.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["schedulers"] == ["waves"]
    assert payload["outside_temp_writes"] == []
    assert payload["scheduler_results"][0]["partition"] == {
        "missing": [],
        "partition_valid": True,
        "terminal": 0,
    }
