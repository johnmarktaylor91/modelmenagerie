"""AL-22: the author steering-surface identity -- disclosed, never staling.

The gap this closes: ``author_prompt_identity`` hashes exactly ONE file
(``prompts/claude_crawler_author_v2.txt``) that the headless executor never
renders, so the prompts that actually steered every 200a attempt (the
executor stage briefs, the campaign fragment) were absent from every
identity axis and every attempt record. Pinned here:

- the per-attempt digest covers exactly the file-resident steering surfaces
  and moves iff their bytes move;
- ``author_steering_surface_identity`` joins the CLOSED rotation-axis set
  with the new CLOSED ``disclosed-non-staling`` rotation class: registered
  rotations of the surface are first-class disclosed history that stales
  NOTHING -- the axis appears in no dependency-vector currency check, no
  review-held recorded-axis set, and no record-time pinned set (the
  non-staling proof is a source census over all three);
- the executor stamps the digest on both stage records of every attempt.

Merges via the prompt-freeze slot with ZERO prompt byte changes: machinery
only (the commit's diff of ``prompts/`` is empty, asserted by the lane).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from menagerie.crawler.author_executor import (
    AuthorExecutorError,
    author_steering_surface_digest,
)
from menagerie.crawler.rotation_registry import (
    ROTATION_AXES,
    ROTATION_CLASSES,
    registered_move_class,
)

pytestmark = pytest.mark.smoke


def _rotation_row(axis: str, source: str, target: str, rotation_class: str) -> dict:
    return {
        "axis": axis,
        "from_identity": source,
        "to_identity": target,
        "rotation_class": rotation_class,
    }


def test_the_axis_and_class_vocabularies_extend_closed() -> None:
    """One new axis, one new class -- both closed-set extensions (Wave 2)."""

    assert "author_steering_surface_identity" in ROTATION_AXES
    assert ROTATION_CLASSES == {"refinement", "soundness", "disclosed-non-staling"}


def test_the_digest_moves_with_the_surface_bytes_and_only_those(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Identity follows the steering bytes: stable, campaign-scoped, typed."""

    import menagerie.crawler.author_executor as executor_module

    prompt_root = tmp_path / "prompts"
    (prompt_root / "executor").mkdir(parents=True)
    (prompt_root / "pool").mkdir()
    (prompt_root / "executor" / "stage1_discovery.md").write_text("s1", encoding="utf-8")
    (prompt_root / "executor" / "stage2_author.md").write_text("s2", encoding="utf-8")
    (prompt_root / "claude_crawler_author_v2.txt").write_text("top", encoding="utf-8")
    (prompt_root / "pool" / "campaign_c1-mech.md").write_text("c1", encoding="utf-8")
    (prompt_root / "pool" / "campaign_c2-disco.md").write_text("c2", encoding="utf-8")
    monkeypatch.setattr(executor_module, "_PROMPT_ROOT", prompt_root)
    monkeypatch.setattr(executor_module, "_EXECUTOR_PROMPTS", prompt_root / "executor")
    monkeypatch.setattr(executor_module, "_POOL_PROMPTS", prompt_root / "pool")

    first = author_steering_surface_digest("c1-mech")
    assert first == author_steering_surface_digest("c1-mech")
    assert first.startswith("sha256:")
    # A different campaign fragment is a different steering surface.
    assert first != author_steering_surface_digest("c2-disco")
    # Editing ANY surface byte rotates the identity.
    (prompt_root / "executor" / "stage2_author.md").write_text("s2 edited", encoding="utf-8")
    assert first != author_steering_surface_digest("c1-mech")
    # An unreadable surface refuses typed, exactly like a render would.
    (prompt_root / "executor" / "stage1_discovery.md").unlink()
    with pytest.raises(AuthorExecutorError, match="steering surface"):
        author_steering_surface_digest("c1-mech")


def test_the_shipped_surfaces_produce_one_stable_identity() -> None:
    """Against the real shipped prompts the digest is derivable and stable."""

    assert author_steering_surface_digest("c1-mech") == author_steering_surface_digest(
        "c1-mech"
    )
    assert author_steering_surface_digest(None).startswith("sha256:")


def test_disclosed_non_staling_moves_classify_without_poisoning() -> None:
    """The move classifier keeps the class lattice honest.

    refinement-only chains stay refinement; chains needing a
    disclosed-non-staling edge classify as disclosed-non-staling (recorded
    proofs stay valid); any chain needing a soundness edge stays soundness;
    unregistered stays None (fail closed).
    """

    axis = "author_steering_surface_identity"
    rows = [
        _rotation_row(axis, "sha256:a", "sha256:b", "disclosed-non-staling"),
        _rotation_row(axis, "sha256:b", "sha256:c", "refinement"),
        _rotation_row(axis, "sha256:c", "sha256:d", "soundness"),
    ]
    assert (
        registered_move_class(rows, axis=axis, recorded="sha256:b", live="sha256:c")
        == "refinement"
    )
    assert (
        registered_move_class(rows, axis=axis, recorded="sha256:a", live="sha256:c")
        == "disclosed-non-staling"
    )
    assert (
        registered_move_class(rows, axis=axis, recorded="sha256:a", live="sha256:d")
        == "soundness"
    )
    assert (
        registered_move_class(rows, axis=axis, recorded="sha256:x", live="sha256:c")
        is None
    )


def test_the_axis_is_provably_non_staling() -> None:
    """Source census: the new axis joins NO staleness or pinning authority.

    Non-staling is a structural property, not a promise: the axis must be
    absent from the dependency-vector currency check, the review-held
    recorded-axis set, the record-time pinned axes, and the coarse
    staleness graph. Adding it to any of these is a soundness-class change
    that must fail here first.
    """

    from menagerie.crawler import artifact_transactions, authority, identity, reducer

    currency_source = inspect.getsource(authority.validate_currency)
    assert "author_steering_surface_identity" not in currency_source

    review_axes = {axis for axis, _field in reducer._REVIEW_HELD_RECORDED_AXES}  # noqa: SLF001
    assert "author_steering_surface_identity" not in review_axes

    pinned_source = inspect.getsource(artifact_transactions)
    pinned_block = pinned_source.split("_RECORD_TIME_PINNED_AXES")[1][:2000]
    assert "author_steering_surface_identity" not in pinned_block

    assert "author_steering_surface_identity" not in inspect.getsource(
        identity
    ).split("stale_dependencies")[-1][:4000]


def test_the_executor_stamps_the_digest_on_both_stage_records() -> None:
    """Source census: both stage-running updates carry the identity.

    The full executor session rig is heavy; the stamp sites are pinned
    structurally -- each ``attempt.update(status="stageN-running", ...)``
    block must stamp ``steering_surface_digest``.
    """

    import menagerie.crawler.author_executor as executor_module

    source = inspect.getsource(executor_module)
    for status in ("stage1-running", "stage2-running"):
        block = source.split(f'status="{status}"')[1][:600]
        assert "steering_surface_digest" in block, (
            f"the {status} attempt record no longer stamps its steering surface"
        )
