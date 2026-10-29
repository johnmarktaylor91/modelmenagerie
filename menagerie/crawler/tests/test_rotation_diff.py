"""Rotation-control toolchain: closure diff, stale-head census, batch seal.

The census and seal fixtures run in smoke; the nodes that compute the REAL
award-closure identity over the full crawler source are marked heavy (a few
seconds of AST walking each) — they are the merge-queue instrument's own
proof, not per-step gates.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from menagerie.crawler.rotation_registry import (
    ROTATION_AXES,
    build_rotation_registration,
    append_rotation_registration,
    rotation_registry_path,
)
from menagerie.crawler.tools.rotation_diff import (
    RotationDiffError,
    mint_seal,
    run_self_test,
    stale_head_census,
    tw_closure_diff,
    verify_seal,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _fresh_identities() -> dict[str, str]:
    """Return a full current-identity map (award closure supplied too, so
    fixture tests never pay the native closure computation)."""

    return {axis: f"sha256:{axis}-head" for axis in sorted(ROTATION_AXES)}


def _register_all(records_root: Path, identities: dict[str, str]) -> None:
    """Register every axis's head so the census reads fresh."""

    path = rotation_registry_path(records_root)
    for axis, identity in identities.items():
        row = build_rotation_registration(
            axis=axis,
            from_identity="sha256:genesis",
            to_identity=identity,
            rotation_class="refinement",
            rationale="fixture registration",
            registered_by="tests",
            created_at="2026-08-24T00:00:00Z",
        )
        append_rotation_registration(path, row)


@pytest.mark.smoke
def test_stale_head_census_fixture(tmp_path: Path) -> None:
    """Fresh heads pass; drifted or unregistered heads fail the census."""

    identities = _fresh_identities()
    _register_all(tmp_path, identities)
    census = stale_head_census(_REPO_ROOT, tmp_path, identities)
    assert census["status"] == "passed"
    assert census["stale_axes"] == []
    # Drift one axis: the census refuses and names it.
    drifted = dict(identities)
    drifted["runner_identity"] = "sha256:runner-DRIFTED"
    census = stale_head_census(_REPO_ROOT, tmp_path, drifted)
    assert census["status"] == "failed"
    assert census["stale_axes"] == ["runner_identity"]
    # An empty registry with current identities reads every axis stale.
    census = stale_head_census(_REPO_ROOT, tmp_path / "empty", identities)
    assert census["status"] == "failed"
    assert set(census["stale_axes"]) == set(identities)
    # Axes outside the closed vocabulary refuse.
    with pytest.raises(RotationDiffError, match="closed vocabulary"):
        stale_head_census(_REPO_ROOT, tmp_path, {"new_axis_identity": "sha256:x"})


@pytest.mark.smoke
def test_seal_refuses_stale_heads(tmp_path: Path) -> None:
    """The batch seal is mintable only over a clean census."""

    with pytest.raises(RotationDiffError, match="seal-refused"):
        mint_seal(_REPO_ROOT, tmp_path, _fresh_identities())


@pytest.mark.heavy
def test_seal_and_post_seal_mutation_refusal(tmp_path: Path) -> None:
    """A minted seal verifies; ANY artifact or registry mutation refuses."""

    identities = _fresh_identities()
    del identities["award_closure_identity"]  # computed natively for the seal
    _register_all(tmp_path, identities)
    # Register the natively computed closure head so the census is clean.
    census = stale_head_census(_REPO_ROOT, tmp_path, dict(identities))
    closure_row = next(row for row in census["axes"] if row["axis"] == "award_closure_identity")
    append_rotation_registration(
        rotation_registry_path(tmp_path),
        build_rotation_registration(
            axis="award_closure_identity",
            from_identity="sha256:genesis",
            to_identity=str(closure_row["current_identity"]),
            rotation_class="refinement",
            rationale="fixture registration",
            registered_by="tests",
            created_at="2026-08-24T00:00:00Z",
        ),
    )
    seal = mint_seal(_REPO_ROOT, tmp_path, identities)
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", "utf-8")
    ok, message = verify_seal(_REPO_ROOT, tmp_path, seal_path)
    assert ok, message
    # Artifact mutation refuses.
    mutated = dict(seal)
    mutated["heads"] = {**seal["heads"], "runner_identity": "sha256:relabel"}
    seal_path.write_text(json.dumps(mutated, indent=2, sort_keys=True) + "\n", "utf-8")
    ok, message = verify_seal(_REPO_ROOT, tmp_path, seal_path)
    assert not ok and message.startswith("seal-mutated")
    # Registry mutation after the seal refuses.
    seal_path.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", "utf-8")
    append_rotation_registration(
        rotation_registry_path(tmp_path),
        build_rotation_registration(
            axis="runner_identity",
            from_identity=identities["runner_identity"],
            to_identity="sha256:post-seal-rotation",
            rotation_class="refinement",
            rationale="post-seal mutation",
            registered_by="tests",
            created_at="2026-08-24T00:00:00Z",
        ),
    )
    ok, message = verify_seal(_REPO_ROOT, tmp_path, seal_path)
    assert not ok and "registry changed" in message


@pytest.mark.heavy
def test_self_test_detects_planted_rotations() -> None:
    """The tool proves it can detect planted rotations (never vacuous)."""

    assert run_self_test(_REPO_ROOT) == []


@pytest.mark.heavy
def test_closure_diff_is_deterministic_and_shaped() -> None:
    """Same-tree diff reads unrotated with both identities equal."""

    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = tw_closure_diff(_REPO_ROOT, head, head)
    assert report["rotated"] is False
    assert report["base_closure_identity"] == report["head_closure_identity"]
    assert report["base_closure_identity"].startswith("sha256:")
