"""Repo hygiene censuses that must fire INSIDE the suite.

Pre-commit hooks guard only hosts that can run hooks; confined workers and CI
checkouts commit through other paths. These censuses run in the mandatory
crawler-smoke gate, so the guarantees hold everywhere:

- No TRACKED text file carries a merge-conflict marker (two 200a-era
  incidents reached main; one landed inside an unparsed gate authority).
- No tracked script, workflow, or crawler module identifies the driver by
  ``pgrep`` — process-name matching killed the wrong process and misread
  liveness in 200a; the lock-based ``tools/whoami_driver.py`` is the
  liveness authority.
- Every test file NAMED by a policy hook or a procedures doc EXISTS (the
  doc-phantom class: LADDER.md rows and a hook message named tripwires that
  had never existed, and two of them hid real coverage holes). Pending
  entries are a closed, disclosed allowlist with a firing trigger — never a
  silent pass.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SCRATCH_PREFIXES = (".gitlocal", ".compose2-scratch", ".scratch-", ".ws2-scratch")

#: Doc-named test files allowed to be ABSENT, each with its firing trigger.
#: An entry here is a disclosed pending obligation, not an exemption: remove
#: the entry (making the census enforce existence) in the same change that
#: lands the file.
_PENDING_DOC_NAMED_TESTS = {
    "test_no_tracked_scratch.py": (
        "lands with the scratch/.gitlocal untracking operator act; the census "
        "enforces existence the moment that sweep merges"
    ),
}


def _tracked_text_files() -> list[Path]:
    """Return tracked text-suffixed files outside the scratch overlays."""

    listing = subprocess.run(
        ("git", "ls-files", "*.py", "*.md", "*.json", "*.yml", "*.yaml", "*.toml", "*.sh"),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    kept: list[Path] = []
    for relative in listing:
        if relative.startswith(_SCRATCH_PREFIXES) or "-scratch/" in relative:
            continue
        kept.append(_REPO_ROOT / relative)
    return kept


@pytest.mark.smoke
def test_no_tracked_file_carries_a_conflict_marker() -> None:
    """Merge-conflict markers must never reach a tracked file."""

    marker = re.compile(r"^(<{7} |>{7} )")
    own_path = Path(__file__).resolve()
    offenders: list[str] = []
    for path in _tracked_text_files():
        if path == own_path:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if marker.match(line):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{line_number}")
    assert not offenders, f"tracked files carry conflict markers: {offenders}"


@pytest.mark.smoke
def test_no_pgrep_driver_identification() -> None:
    """Driver liveness is lock-based (tools/whoami_driver.py), never pgrep."""

    own_path = Path(__file__).resolve()
    offenders: list[str] = []
    for path in _tracked_text_files():
        relative = path.relative_to(_REPO_ROOT).as_posix()
        if path == own_path:
            continue
        if relative == "menagerie/crawler/tests/test_crawl_doctor.py":
            # WS9's own pgrep-ban test: names the pattern to ban it.
            continue
        if relative == "menagerie/crawler/tools/whoami_driver.py":
            # The anti-pgrep authority itself: its docstrings name the banned
            # pattern to document why lock-based liveness replaces it.
            continue
        if not (relative.startswith(("scripts/", "tools/", ".github/", "menagerie/crawler/"))):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "pgrep" in line:
                offenders.append(f"{relative}:{line_number}: {line.strip()[:100]}")
    assert not offenders, (
        "pgrep-based process identification is banned (lock-based liveness via "
        f"tools/whoami_driver.py instead): {offenders}"
    )


@pytest.mark.smoke
def test_every_doc_named_test_file_exists() -> None:
    """Every test file named by procedures docs or policy hooks exists."""

    documents = [
        *sorted((_REPO_ROOT / "menagerie/crawler/procedures").glob("*.md")),
        _REPO_ROOT / ".pre-commit-config.yaml",
    ]
    search_roots = (
        _REPO_ROOT / "menagerie/crawler/tests",
        _REPO_ROOT / "tests",
        _REPO_ROOT / "tools/tests",
    )
    known = {path.name for root in search_roots for path in root.rglob("test_*.py")}
    phantoms: list[str] = []
    disclosed: list[str] = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for name in sorted(set(re.findall(r"test_[a-z0-9_]+\.py", text))):
            if name in known:
                continue
            if name in _PENDING_DOC_NAMED_TESTS:
                disclosed.append(
                    f"{document.name}:{name} (pending: {_PENDING_DOC_NAMED_TESTS[name]})"
                )
                continue
            phantoms.append(f"{document.relative_to(_REPO_ROOT)} names {name}")
    assert not phantoms, (
        "doc-phantom census: documents name test files that do not exist "
        f"(fix the doc or land the test): {phantoms}; disclosed pending: {disclosed}"
    )


@pytest.mark.smoke
def test_gated_release_jobs_are_ungated_or_runbook_named() -> None:
    """A dormancy-gated release job must have a recorded runbook equivalent.

    Dormant-on-a-nonexistent-variable was the disguise "green" wore: the
    round21 release jobs never ran anywhere. Until real runners are
    provisioned, every job gated on MENAGERIE_RELEASE_RUNNERS must be NAMED
    by the procedures runbook with an artifact-producing, ledger-recorded
    hand-run — or lose its gate.
    """

    workflow = (_REPO_ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    runbook = (_REPO_ROOT / "menagerie/crawler/procedures/RUN.md").read_text(encoding="utf-8")
    gated_jobs: list[str] = []
    current_job: str | None = None
    for line in workflow.splitlines():
        job_match = re.match(r"^  ([a-z0-9-]+):\s*$", line)
        if job_match:
            current_job = job_match.group(1)
        if "vars.MENAGERIE_RELEASE_RUNNERS" in line and current_job:
            gated_jobs.append(current_job)
    assert gated_jobs, "expected at least one gated release job (census wiring check)"
    unnamed = [job for job in gated_jobs if job not in runbook]
    assert not unnamed, (
        "gated release jobs without a recorded runbook equivalent (un-dark them "
        f"or name them in procedures/RUN.md): {unnamed}"
    )
    # The runbook step must be artifact-producing AND ledger-recorded (the
    # gate runner's record-attestation verb appends the closure-keyed row).
    assert "gate_runner record-attestation" in runbook
    assert "MENAGERIE_RELEASE_ATTESTATION" in runbook


@pytest.mark.smoke
def test_macos_scope_disclosure_matches_the_registry() -> None:
    """The published launch-platform coverage counts track the registry exactly.

    "Documented-expected" must never launder authorship as necessity: the
    conformance registry's linux-only majority is an authorship fact, and the
    disclosure that publishes it has to move in lockstep with the registry so
    the count can neither rot nor be quietly widened without the doc noticing.
    """

    import json

    registry = json.loads(
        (_REPO_ROOT / "menagerie/crawler/conformance-round21.json").read_text(encoding="utf-8")
    )
    host_index = registry["record_fields"].index("host_alias")
    host_aliases = registry["aliases"]["hosts"]
    counts = {"both": 0, "linux": 0, "macos": 0}
    for record in registry["records"]:
        counts[host_aliases[record[host_index]]] += 1
    total = len(registry["records"])
    macos_covered = counts["both"] + counts["macos"]
    disclosure = (_REPO_ROOT / "menagerie/crawler/CONFORMANCE.md").read_text(encoding="utf-8")
    assert "Launch-platform scope disclosure" in disclosure
    assert "authored-linux-only" in disclosure, (
        "the disclosure must name the closed reason vocabulary (authorship, "
        "never implied necessity)"
    )
    for expected in (
        f"{macos_covered}/{total} macos-covered",
        f"{counts['linux']}/{total} linux-only",
    ):
        assert expected in disclosure, (
            f"CONFORMANCE.md scope disclosure is stale: expected {expected!r}; "
            "update the published counts in the same change that moved the registry"
        )
