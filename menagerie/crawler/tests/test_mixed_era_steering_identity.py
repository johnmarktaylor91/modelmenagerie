"""AB-30 matrix: steering-identity seam (AL-22, owed-by-batch-lane discharge).

The Wave-2 batch stamps ``author_steering_surface_digest(campaign_id)`` -- a
per-file sha256 map over exactly the file-resident surfaces steering one
author attempt (both executor stage briefs, the campaign fragment, the pinned
top-level prompt) -- on BOTH stage records of every fresh attempt, and
registers the NEW rotation axis ``author_steering_surface_identity`` under
the NEW closed rotation class ``disclosed-non-staling``. Prior-era attempt
records carry only the author binding's ``prompt_sha256`` and no
``steering_surface_digest`` anywhere.

The mixed-era obligation this module discharges (registry seam
``steering-identity``, ``matrix_disposition: owed-by-batch-lane``,
``owed_by: AL-22``): every declared reader of the attempt-record seam keeps
parsing prior-era rows era-scoped -- an ABSENT digest never refuses, never
crashes unclassified, and can never retro-stale a prior-era attempt, because
the axis is DISCLOSED-NON-STALING by construction: absent from
``authority.validate_currency``, from ``reducer._REVIEW_HELD_RECORDED_AXES``,
from ``artifact_transactions._RECORD_TIME_PINNED_AXES``, and from the coarse
staleness graph in ``identity.stale_dependencies``. The census here uses the
ACTUAL set objects where they are importable objects, and source text only
for the two consumers whose closed sets live as function-local literals.

Era artifacts are minted through the REAL producers at this tree
(``author_attempts.new_attempt`` / ``AttemptHandle.update``): the prior-era
record differs from a fresh one by exactly the absent key plus the
prompt_sha256-only author binding, which is precisely the Wave-1 shape --
no hand-forged record shapes. The digest-determinism cell exercises the real
``author_steering_surface_digest`` over a minimal surface set laid out in the
function's own expected prompt-root layout under ``tmp_path``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Iterator

import pytest

from menagerie.crawler.author_attempts import (
    latest_attempt,
    list_attempts,
    new_attempt,
    prior_attempts_summary,
)
from menagerie.crawler.author_executor import (
    AuthorExecutorError,
    author_steering_surface_digest,
)
from menagerie.crawler.rotation_registry import (
    ROTATION_AXES,
    ROTATION_CLASSES,
    RotationRegistrationError,
    append_rotation_registration,
    build_rotation_registration,
    load_rotation_registrations,
    registered_move_class,
)
from menagerie.crawler.tests.mixed_era import classify_reader

pytestmark = pytest.mark.smoke

AXIS = "author_steering_surface_identity"
NEW_CLASS = "disclosed-non-staling"


# -- deterministic producers ------------------------------------------------


def _clock() -> Iterator[str]:
    tick = 0
    while True:
        tick += 1
        yield f"2026-08-24T00:00:{tick:02d}Z"


def _nonces() -> Iterator[str]:
    count = 0
    while True:
        count += 1
        yield f"{count:08d}abcdef00"


@pytest.fixture()
def author_root(tmp_path: Path) -> Path:
    root = tmp_path / "author-root"
    root.mkdir()
    return root


def _mint_prior_era_attempt(author_root: Path, clock, nonces) -> None:
    """Mint a Wave-1-shaped attempt via the real producers: no digest anywhere.

    Prior-era attempts carry only the author binding's ``prompt_sha256``; the
    stage dicts have every field the Wave-1 executor wrote and NO
    ``steering_surface_digest`` key.
    """

    handle = new_attempt(
        author_root,
        stable_id="m0001",
        campaign_id="c1-mech",
        kind="author",
        clock=lambda: next(clock),
        nonce_factory=lambda: next(nonces),
    )
    handle.update(
        status="stage2-running",
        stage2={
            "started_at": next(clock),
            "resumed_from": None,
            "cold_start_reason": "no-recoverable-stage1-session",
        },
        author={
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "version": "1.0.0",
            "prompt_sha256": "sha256:" + "a" * 64,
        },
    )
    handle.update(
        status="failed",
        outcome={
            "failure_stage": "stage2",
            "failure_reason": "provider-timeout",
            "detail": None,
        },
    )


# -- cell 1: closed-set membership ------------------------------------------


def test_axis_and_class_are_closed_set_members() -> None:
    """The new axis and class are members of the CLOSED vocabularies.

    Membership is asserted against the actual frozensets, and round-tripped
    through the real registration constructor + ledger reader: a row naming
    the new axis under the new class builds, appends, and loads back; the
    same axis under an unknown class refuses typed (the set stayed closed).
    """

    assert AXIS in ROTATION_AXES
    assert NEW_CLASS in ROTATION_CLASSES

    row = build_rotation_registration(
        axis=AXIS,
        from_identity="sha256:" + "1" * 64,
        to_identity="sha256:" + "2" * 64,
        rotation_class=NEW_CLASS,
        rationale="AB-30 matrix round-trip",
        registered_by="test_mixed_era_steering_identity",
        created_at="2026-08-24T00:00:00Z",
    )
    assert row["axis"] == AXIS
    assert row["rotation_class"] == NEW_CLASS

    with pytest.raises(RotationRegistrationError, match="rotation class"):
        build_rotation_registration(
            axis=AXIS,
            from_identity="sha256:" + "1" * 64,
            to_identity="sha256:" + "2" * 64,
            rotation_class="disclosed-but-unregistered",
            rationale="must refuse",
            registered_by="test_mixed_era_steering_identity",
            created_at="2026-08-24T00:00:00Z",
        )


def test_registered_rotation_of_the_axis_round_trips_the_real_ledger(
    tmp_path: Path,
) -> None:
    """The real append/load/classify readers accept the new axis+class row."""

    ledger = tmp_path / "records" / "intake" / "registered-authority-rotations.jsonl"
    row = build_rotation_registration(
        axis=AXIS,
        from_identity="sha256:" + "1" * 64,
        to_identity="sha256:" + "2" * 64,
        rotation_class=NEW_CLASS,
        rationale="AB-30 matrix round-trip",
        registered_by="test_mixed_era_steering_identity",
        created_at="2026-08-24T00:00:00Z",
    )
    assert append_rotation_registration(ledger, row) is True
    kind, loaded = classify_reader(lambda: load_rotation_registrations(ledger))
    assert kind == "parsed", f"ledger reader refused the new axis row: {loaded!r}"
    assert [entry["axis"] for entry in loaded] == [AXIS]
    assert (
        registered_move_class(
            loaded,
            axis=AXIS,
            recorded="sha256:" + "1" * 64,
            live="sha256:" + "2" * 64,
        )
        == NEW_CLASS
    )


# -- cell 2: the axis joins NO staling consumer ------------------------------


def test_axis_is_absent_from_every_staling_consumer() -> None:
    """Source census: disclosed-non-staling is structural, not a promise.

    The axis must be absent from the review-held recorded-axis set and the
    record-time pinned axes (asserted on the ACTUAL set objects), and from
    ``validate_currency`` and the ``stale_dependencies`` graph (function-local
    closed literals, asserted on their source). Adding the axis to any of
    these is a soundness-class change that must fail here first -- that WOULD
    retro-stale prior-era attempts, exactly what the seam declaration forbids.
    """

    from menagerie.crawler import artifact_transactions, authority, identity, reducer

    review_axes = {axis for axis, _field in reducer._REVIEW_HELD_RECORDED_AXES}  # noqa: SLF001
    assert AXIS not in review_axes

    pinned_axes = set(artifact_transactions._RECORD_TIME_PINNED_AXES)  # noqa: SLF001
    assert AXIS not in pinned_axes

    assert AXIS not in inspect.getsource(authority.validate_currency)
    assert AXIS not in inspect.getsource(identity.stale_dependencies)


# -- cell 3: prior-era attempt records parse era-scoped ----------------------


def test_prior_era_attempt_record_parses_without_digest(author_root: Path) -> None:
    """Every attempt-record reader parses the digest-free prior-era row.

    Absence of ``steering_surface_digest`` never refuses, never crashes
    unclassified, and no reader invents the key: the prior-era record stays
    exactly the record the Wave-1 executor wrote.
    """

    clock, nonces = _clock(), _nonces()
    _mint_prior_era_attempt(author_root, clock, nonces)

    kind, handles = classify_reader(lambda: list_attempts(author_root))
    assert kind == "parsed", f"list_attempts refused a prior-era record: {handles!r}"
    assert len(handles) == 1
    record = handles[0].record
    assert record["author"]["prompt_sha256"].startswith("sha256:")
    assert "steering_surface_digest" not in record["stage2"]
    assert "steering_surface_digest" not in record

    kind, newest = classify_reader(
        lambda: latest_attempt(author_root, stable_id="m0001")
    )
    assert kind == "parsed" and newest is not None
    assert newest.status == "failed"

    kind, summaries = classify_reader(lambda: prior_attempts_summary(author_root))
    assert kind == "parsed", f"prior_attempts_summary refused: {summaries!r}"
    assert len(summaries) == 1
    assert summaries[0]["failure_reason"] == "provider-timeout"
    assert "steering_surface_digest" not in summaries[0]


def test_mixed_era_attempts_coexist_under_one_author_root(author_root: Path) -> None:
    """One lane, both eras: the digest-free prior attempt and a fresh stamped
    attempt are parsed side by side, each keeping its own era's shape."""

    clock, nonces = _clock(), _nonces()
    _mint_prior_era_attempt(author_root, clock, nonces)

    fresh_digest = author_steering_surface_digest(None)
    fresh = new_attempt(
        author_root,
        stable_id="m0001",
        campaign_id=None,
        kind="author",
        clock=lambda: next(clock),
        nonce_factory=lambda: next(nonces),
    )
    fresh.update(
        status="stage2-running",
        stage2={
            "started_at": next(clock),
            "resumed_from": None,
            "cold_start_reason": "no-recoverable-stage1-session",
            "steering_surface_digest": fresh_digest,
        },
    )

    kind, handles = classify_reader(lambda: list_attempts(author_root))
    assert kind == "parsed"
    assert [handle.number for handle in handles] == [1, 2]
    prior_record, fresh_record = handles[0].record, handles[1].record
    # The prior-era row is untouched by the fresh era: absent stays absent.
    assert "steering_surface_digest" not in prior_record["stage2"]
    assert fresh_record["stage2"]["steering_surface_digest"] == fresh_digest
    assert fresh_digest.startswith("sha256:")

    kind, summaries = classify_reader(lambda: prior_attempts_summary(author_root))
    assert kind == "parsed"
    assert [summary["attempt_number"] for summary in summaries] == [1, 2]
    # The prior attempt ended terminal ("failed") before the fresh one opened;
    # its status is its own recorded history, never a retro-staled verdict.
    assert summaries[0]["status"] == "failed"


# -- cell 4: digest determinism over the surface bytes -----------------------


def test_digest_is_deterministic_and_moves_with_surface_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same surface bytes -> same identity; any changed byte -> new identity.

    A minimal surface set is laid out in the real function's expected prompt
    layout (executor/ briefs, top-level pinned prompt, pool/ campaign
    fragment) under ``tmp_path``; the real function reads it unmodified.
    """

    import menagerie.crawler.author_executor as executor_module

    prompt_root = tmp_path / "prompts"
    (prompt_root / "executor").mkdir(parents=True)
    (prompt_root / "pool").mkdir()
    (prompt_root / "executor" / "stage1_discovery.md").write_text("s1", encoding="utf-8")
    (prompt_root / "executor" / "stage2_author.md").write_text("s2", encoding="utf-8")
    (prompt_root / "claude_crawler_author_v2.txt").write_text("top", encoding="utf-8")
    (prompt_root / "pool" / "campaign_c1-mech.md").write_text("frag", encoding="utf-8")
    monkeypatch.setattr(executor_module, "_PROMPT_ROOT", prompt_root)
    monkeypatch.setattr(executor_module, "_EXECUTOR_PROMPTS", prompt_root / "executor")
    monkeypatch.setattr(executor_module, "_POOL_PROMPTS", prompt_root / "pool")

    first = author_steering_surface_digest("c1-mech")
    assert first.startswith("sha256:")
    assert first == author_steering_surface_digest("c1-mech")

    # The campaign fragment is part of the surface: no fragment, new identity.
    campaign_free = author_steering_surface_digest(None)
    assert campaign_free != first
    assert campaign_free == author_steering_surface_digest(None)

    # One changed byte in any steering surface rotates the identity.
    (prompt_root / "pool" / "campaign_c1-mech.md").write_text("frag!", encoding="utf-8")
    assert author_steering_surface_digest("c1-mech") != first
    # ... and only the touched campaign's identity moved.
    assert author_steering_surface_digest(None) == campaign_free

    # An unreadable surface refuses typed, exactly like a brief render would.
    (prompt_root / "executor" / "stage2_author.md").unlink()
    with pytest.raises(AuthorExecutorError, match="steering surface"):
        author_steering_surface_digest("c1-mech")
