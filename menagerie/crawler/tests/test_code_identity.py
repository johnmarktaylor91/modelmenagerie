"""DX-45 (TC-25+UU-10): leg-start code identity.

The pilot launched through a ``.venv-crawler`` symlink and nothing recorded
WHICH code adjudicated a leg: the editable finder could resolve torchlens or
menagerie outside the pilot tree (mid-leg dev edits are routine during fix
waves) and the launch certificate proved only the git tree, not the runner.
The certificate now binds the runner identity (interpreter path, python
version, resolved package origins) and refuses on drift; the leg-open host
pack refuses an editable finder that resolves outside the declared tree; and
the driver-lock owner record carries the identity durably.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from menagerie.crawler.code_identity import (
    runner_code_identity,
    runner_identity_drift,
)
from menagerie.crawler.tools.host_preflight import (
    RUNNER_IDENTITY_OUTSIDE_TREE,
    evaluate_host_preflight,
)

pytestmark = pytest.mark.smoke


def test_runner_identity_names_this_exact_runner() -> None:
    """Interpreter, python version, and package origins are all resolved."""

    identity = runner_code_identity()
    assert identity["interpreter"] == str(Path(sys.executable).resolve())
    assert identity["python_version"] == ".".join(
        str(part) for part in sys.version_info[:3]
    )
    assert identity["menagerie_origin"] is not None
    assert Path(identity["menagerie_origin"]).is_file()


def test_drift_comparator_names_exactly_the_moved_facts() -> None:
    """Equal identities have zero drift; each moved fact is named."""

    recorded = runner_code_identity()
    assert runner_identity_drift(recorded, runner_code_identity()) == []
    moved = dict(recorded)
    moved["interpreter"] = "/opt/other/bin/python3"
    moved["python_version"] = "0.0.0"
    drift = runner_identity_drift(moved, runner_code_identity())
    assert "interpreter" in drift
    assert "python_version" in drift


def _committed_repo(tmp_path: Path) -> Path:
    """Return a minimal committed git repository."""

    repo = tmp_path / "repo"
    repo.mkdir()
    for arguments in (
        ("init", "-b", "main"),
        ("config", "user.email", "crawler-tests@example.invalid"),
        ("config", "user.name", "Crawler Tests"),
        ("config", "commit.gpgsign", "false"),
    ):
        subprocess.run(("git", *arguments), cwd=repo, check=True, capture_output=True)
    (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=repo, check=True, capture_output=True)
    subprocess.run(("git", "commit", "-m", "initial"), cwd=repo, check=True, capture_output=True)
    return repo


def _host_pack(tmp_path: Path, **overrides: object) -> dict:
    """Run the host pack over a healthy fixture."""

    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    arguments: dict = {
        "records_root": records,
        "roots": {"records-root": records},
        "minimum_nofile": 1,
    }
    arguments.update(overrides)
    return evaluate_host_preflight(**arguments)


def test_editable_finder_outside_the_declared_tree_refuses(tmp_path: Path) -> None:
    """A leg whose code resolves outside the launch tree is a typed refusal."""

    pack = _host_pack(tmp_path, code_tree_root=tmp_path / "not-the-pilot-tree")
    assert pack["ok"] is False
    assert any(
        RUNNER_IDENTITY_OUTSIDE_TREE in refusal for refusal in pack["refusals"]
    )


def test_editable_finder_inside_the_declared_tree_passes(tmp_path: Path) -> None:
    """The real worktree contains its own resolved packages."""

    tree = Path(runner_code_identity()["menagerie_origin"]).parents[1]
    pack = _host_pack(tmp_path, code_tree_root=tree)
    census = {row["name"]: row for row in pack["armed_guard_census"]}
    assert census["runner-identity"]["armed"] is True
    assert not any(
        RUNNER_IDENTITY_OUTSIDE_TREE in refusal for refusal in pack["refusals"]
    )


def test_driver_lock_owner_records_the_runner_identity() -> None:
    """The leg-start record carries the code identity durably (census hook)."""

    source = (
        Path(runner_code_identity()["menagerie_origin"]).parent / "crawler" / "driver.py"
    ).read_text(encoding="utf-8")
    assert '"runner_identity": ' in source
    assert "runner_code_identity()" in source
