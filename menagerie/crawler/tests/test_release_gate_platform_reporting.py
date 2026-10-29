"""The release gate reports foreign-platform registry rows as documented-expected.

A release-gate run collects only its own platform's registry, so the other
platform's rows are absent from every count. These tests pin the gate's own
reporting of that fact -- in the attestation JSON and the terminal summary --
so a reader on macOS can tell "these rows need a Linux host" (documented,
expected, actionable) from "these rows are broken".
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from menagerie.crawler.tests.conftest import (
    _REPOSITORY_ROOT,
    _round21_other_platform_expectations,
    _round21_release_nodes,
)


@pytest.mark.smoke
def test_each_gate_target_documents_every_other_platforms_rows() -> None:
    """Every non-active registry appears in full with the typed expectation."""

    for active, other in (("osx-arm64", "linux-64"), ("linux-64", "osx-arm64")):
        expectations = _round21_other_platform_expectations(active)
        assert set(expectations) == {other}
        expectation = expectations[other]
        assert expectation["status"] == "documented-expected-absent"
        assert f"require a {other} host" in expectation["reason"]
        assert "not a failure" in expectation["reason"]
        # Release for the other platform is explicitly NOT granted by this run.
        assert "requires its own gate run" in expectation["reason"]
        assert expectation["nodes"] == sorted(_round21_release_nodes(other))
        assert expectation["nodes"], "an empty registry would document nothing"


def test_attestation_and_terminal_summary_carry_the_expected_absence(
    tmp_path: Path,
) -> None:
    """A real gate-armed session writes and prints the documented expectation.

    Runs one nested single-test pytest session with the attestation armed and
    the macOS platform lock selected, then reads the gate's two reporting
    surfaces. The nested gate is intentionally unmet (one arbitrary test is not
    the registry): the expectation must be present in exactly that failure
    shape, because that is the run a confused reader would be looking at.

    Parameters
    ----------
    tmp_path:
        Scratch root for the nested attestation JSON.
    """

    attestation_path = tmp_path / "release-attestation.json"
    environment = dict(os.environ)
    environment.update(
        {
            "MENAGERIE_RELEASE_ATTESTATION": str(attestation_path),
            "MENAGERIE_PLATFORM_LOCK": "round19-osx-arm64.lock",
            "PYTHONPATH": str(_REPOSITORY_ROOT),
        }
    )
    environment.pop("MENAGERIE_RELEASE_GATE", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "menagerie/crawler/tests/test_wakeup_os_scheduler.py::"
            "test_cron_projection_pins_marked_block_shape",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=_REPOSITORY_ROOT,
        env=environment,
    )

    # The armed-but-unmet gate fails closed; that is the reader's scenario.
    assert completed.returncode != 0, completed.stdout
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    assert attestation["status"] == "unmet-release-gate"
    assert attestation["target"] == "osx-arm64"
    foreign = attestation["other_platform_registries"]
    assert set(foreign) == {"linux-64"}
    assert foreign["linux-64"]["status"] == "documented-expected-absent"
    assert foreign["linux-64"]["nodes"] == sorted(_round21_release_nodes("linux-64"))
    row_count = len(foreign["linux-64"]["nodes"])
    assert (
        f"release gate [osx-arm64]: {row_count} linux-64 registry rows are"
        " DOCUMENTED-EXPECTED absent on this host" in completed.stdout
    )
