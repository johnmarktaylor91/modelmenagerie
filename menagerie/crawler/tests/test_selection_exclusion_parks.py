"""P5 round-2 R2-1: review holds un-bar, selection-exclusion parks re-bar.

The regression pinned here is the arm-(c) composition defect: a review hold is
a grant RELEASE authority (``status.human_review.required`` is exactly what
lets a stage-matching requeue grant bind), so "excluded by review holds" was
never true — appending the PD-7 holds mechanically UN-BARRED the four members
whose only structural bar was ``unreviewed-terminal``. Both halves are pinned:

* the un-barring mechanism itself, at the selection predicate (a hold flips
  ``GRANT_UNBINDABLE`` into ``GRANTED_REDRIVE`` for a stage-matching grant) —
  documented behavior nobody may "fix" silently, because the requeue lane
  depends on it; and
* the exclusion instrument that restores the bar: a live SELECTION-EXCLUSION
  park row reads the hold (live while the exact review-held revision governs),
  is absolute in every selection mode, is released by NO grant of any spelling
  (its stage sentinel is outside the closed grant-stage vocabulary), vetoes the
  D-48 machine-readmission lane, and yields only to an explicit operator void
  or the head lawfully moving through an operator lane.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.grant_parks import (
    GRANT_PENDING_DISPOSITION,
    SELECTION_EXCLUSION_DISPOSITION,
    SELECTION_EXCLUSION_STAGE,
    GrantParkError,
    build_selection_exclusion_park_row,
    grant_park_path,
    grant_park_released,
    load_grant_park_rows,
    validate_grant_park_row,
)
from menagerie.crawler.work_selection import (
    MemberFacts,
    SelectionDecision,
    SelectionMode,
    work_selection_predicate,
)

pytestmark = pytest.mark.smoke

_HELD_REVISION = "sha256:" + "5e" * 32


def _held_facts(**overrides: Any) -> MemberFacts:
    """Facts for a review-held failure terminal with a stage-matching grant."""

    base: dict[str, Any] = {
        "stable_id": "m2129",
        "raw_head": {
            "status": {
                "kind": "failed",
                "code": "failed:author",
                "human_review": {"required": True},
            },
            "record_revision": _HELD_REVISION,
            "authored_metadata_state": "pending",
        },
        "dependency_current": True,
        "stale_reason": None,
        "parked": False,
        "deferral_live": False,
        "live_grant": {"grant_id": "grant-1", "stage": "author"},
        # The derived admission fact: reviewed==True + stage match binds
        # (driver_admission's reviewed-terminal rule).
        "grant_bindable": True,
        "grant_release_authority": "human-review",
        "active_generation": None,
        "active_generation_discharged": False,
    }
    base.update(overrides)
    return MemberFacts(**base)


def _exclusion_row(**overrides: Any) -> dict[str, Any]:
    """One valid selection-exclusion park row."""

    fields: dict[str, Any] = {
        "stable_id": "m2129",
        "name": "highway-mlp",
        "campaign_id": "c1-mech",
        "machine_id": "worker-a",
        "created_at": "2026-08-31T00:00:00Z",
        "held_revision": _HELD_REVISION,
        "held_status_code": "failed:author",
        "decision": "PD-7-slop-family-routes",
        "reason": "frozen decision: slop-family routes stay excluded from authoring",
    }
    fields.update(overrides)
    return build_selection_exclusion_park_row(**fields)


def test_a_review_hold_alone_is_a_release_authority_not_an_exclusion() -> None:
    """The un-barring mechanism, pinned as documented behavior (R2-1).

    Pre-hold, the member's structural bar is ``unreviewed-terminal``
    (``grant_bindable=False`` -> GRANT_UNBINDABLE). Appending the hold flips
    the derived bindability, so the SAME stage-matching grant now admits. This
    is the requeue lane working as designed — which is exactly why a hold can
    never BE an exclusion, and why arm (c) needed the park below.
    """

    pre_hold = work_selection_predicate(
        _held_facts(grant_bindable=False, grant_release_authority=None),
        mode=SelectionMode.ORDINARY,
    )
    assert not pre_hold.admitted
    assert pre_hold.decision is SelectionDecision.GRANT_UNBINDABLE

    post_hold = work_selection_predicate(_held_facts(), mode=SelectionMode.ORDINARY)
    assert post_hold.admitted, "a review hold IS a grant release authority"
    assert post_hold.decision is SelectionDecision.GRANTED_REDRIVE
    assert post_hold.release_authority == "human-review"


def test_a_live_exclusion_park_outranks_the_binding_grant_in_every_mode() -> None:
    """Clause 5 restores the bar: parked beats granted, named, and catch-up."""

    for mode, targets in (
        (SelectionMode.ORDINARY, frozenset()),
        (SelectionMode.REDRIVE, frozenset({"m2129"})),
        (SelectionMode.GATE_CATCHUP, frozenset({"m2129"})),
        (SelectionMode.ONLY_UNSETTLED, frozenset()),
    ):
        decision = work_selection_predicate(
            _held_facts(parked=True), mode=mode, mode_targets=targets
        )
        assert not decision.admitted, mode
        assert decision.decision is SelectionDecision.PARKED, mode


def test_exclusion_row_builds_validates_and_names_its_release() -> None:
    """The row shape: hold-bound, decision-named, void-only release."""

    row = _exclusion_row()
    assert row["disposition"] == SELECTION_EXCLUSION_DISPOSITION
    assert row["pending"]["blocked_parent_revision"] == _HELD_REVISION
    assert row["pending"]["blocked_stage"] == SELECTION_EXCLUSION_STAGE
    assert row["pending"]["exclusion_decision"] == "PD-7-slop-family-routes"
    assert "--void" in row["recheck_hint"]
    assert validate_grant_park_row(row) == row

    with pytest.raises(GrantParkError, match="decision"):
        _exclusion_row(decision="  ")
    with pytest.raises(GrantParkError, match="reason"):
        _exclusion_row(reason="")
    # An exclusion may not smuggle a publishable pending revision.
    smuggled = dict(_exclusion_row())
    smuggled["pending"] = {**smuggled["pending"], "pending_model": {"stable_id": "m2129"}}
    with pytest.raises(GrantParkError, match="may not embed"):
        validate_grant_park_row(smuggled)
    # The stage sentinel is content-bound.
    relabeled = dict(_exclusion_row())
    relabeled["pending"] = {**relabeled["pending"], "blocked_stage": "author"}
    with pytest.raises(GrantParkError, match="stage sentinel|row_sha256|digest"):
        validate_grant_park_row(relabeled)


def test_no_grant_releases_an_exclusion_park() -> None:
    """Release is head-movement or operator void — never a grant.

    Two independent walls: the selection-side release predicate returns False
    for the exclusion disposition regardless of the offered grant, and the
    grant-stage vocabulary is closed so no lawful grant can even carry the
    sentinel stage.
    """

    from menagerie.crawler.driver import _park_released_for_selection

    row = _exclusion_row()
    forged_grant = {"grant_id": "grant-2", "stage": SELECTION_EXCLUSION_STAGE}
    assert not _park_released_for_selection(row, forged_grant)
    assert not _park_released_for_selection(row, {"grant_id": "grant-3", "stage": "author"})
    # The grant builder refuses the sentinel stage outright (closed vocabulary).
    from menagerie.crawler.checkpoint import build_canonical_requeue_grant

    with pytest.raises(ValueError, match="unknown failure stage"):
        build_canonical_requeue_grant(
            Path("/nonexistent/operational/grants.jsonl"),
            stable_id="m2129",
            stage=SELECTION_EXCLUSION_STAGE,
            attempts=1,
            reason="try to grant against the sentinel",
            granted_by="tests",
        )
    # Liveness is the head-bound fact: live while the held revision governs,
    # released when the head lawfully moved.
    held_head = {"record_revision": _HELD_REVISION}
    moved_head = {"record_revision": "sha256:" + "6f" * 32}
    assert not grant_park_released(row, held_head)
    assert grant_park_released(row, moved_head)
    assert not grant_park_released(row, None)


def test_exclusion_parks_veto_the_machine_readmission_lane() -> None:
    """D-48 never re-admits over an operator exclusion (non-machinery park)."""

    from menagerie.crawler.park_release import derive_park_machine_readmission

    row = _exclusion_row()
    assert (
        derive_park_machine_readmission([row], tree_identity="sha256:" + "77" * 32)
        is None
    )


def test_park_exclusion_tool_derives_the_held_head_and_refuses_unheld(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The operator tool: live-ledger identity derivation, fail-closed targets.

    The held revision is derived from the LIVE models ledger (never typed by
    hand — the registry-identity-derivation rule), and the tool refuses a
    member with no head, a non-failed head, and a hold-less head.
    """

    from menagerie.crawler.tools.grant_parks import main as parks_main

    repo = tmp_path / "repo"
    records_root = repo / "menagerie" / "crawler" / "records"
    models = records_root / "models" / "current-shard.jsonl"
    models.parent.mkdir(parents=True)
    rows = [
        {
            "stable_id": "m2129",
            "name": "highway-mlp",
            "parent_revision": None,
            "record_revision": _HELD_REVISION,
            "status": {
                "kind": "failed",
                "code": "failed:author",
                "human_review": {"required": True},
            },
        },
        {
            "stable_id": "m9999",
            "name": "no-hold",
            "parent_revision": None,
            "record_revision": "sha256:" + "aa" * 32,
            "status": {"kind": "failed", "code": "failed:author"},
        },
        {
            "stable_id": "m1115",
            "name": "runs-award",
            "parent_revision": None,
            "record_revision": "sha256:" + "bb" * 32,
            "status": {"kind": "runs", "code": "runs"},
        },
    ]
    models.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    def park(stable_id: str) -> int:
        return parks_main(
            [
                "--repo-root",
                str(repo),
                "--park-exclusion",
                stable_id,
                "--decision",
                "PD-7-slop-family-routes",
                "--campaign",
                "c1-mech",
                "--reason",
                "frozen decision exclusion",
            ]
        )

    # Missing operator inputs refuse.
    assert (
        parks_main(["--repo-root", str(repo), "--park-exclusion", "m2129"]) == 1
    )
    capsys.readouterr()
    # The three fail-closed target classes.
    assert park("m-absent") == 1
    assert "holds no revision" in capsys.readouterr().err
    assert park("m1115") == 1
    assert "not a failure" in capsys.readouterr().err
    assert park("m9999") == 1
    assert "no review hold" in capsys.readouterr().err
    # The held member parks, binding the exact live revision.
    assert park("m2129") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["disposition"] == SELECTION_EXCLUSION_DISPOSITION
    assert payload["pending"]["blocked_parent_revision"] == _HELD_REVISION
    # Idempotent: the same fact returns the same row without a second append.
    assert park("m2129") == 0
    assert json.loads(capsys.readouterr().out)["row_sha256"] == payload["row_sha256"]
    live = load_grant_park_rows([grant_park_path(records_root)])
    assert [row["row_sha256"] for row in live] == [payload["row_sha256"]]


def test_exclusion_disposition_is_threaded_through_the_liveness_maps() -> None:
    """The closed-set widening is complete: every liveness consumer knows it.

    An unknown disposition is SKIPPED by ``_live_park_rows`` (it feeds no
    ``facts.parked``), so a disposition that exists in the ledger but not in
    the maps would be an exclusion that excludes nothing — the silent-discard
    class again, one seam over.
    """

    from menagerie.crawler.driver import _live_park_rows
    from menagerie.crawler.status import MACHINE_DEFERRAL_DISPOSITIONS

    assert SELECTION_EXCLUSION_DISPOSITION in MACHINE_DEFERRAL_DISPOSITIONS
    row = _exclusion_row()
    live = _live_park_rows(
        {"m2129": [row]},
        frozenset({"m2129"}),
        {"m2129": {"record_revision": _HELD_REVISION}},
    )
    assert set(live[SELECTION_EXCLUSION_DISPOSITION]) == {"m2129"}
    assert GRANT_PENDING_DISPOSITION in live
    # A moved head releases it in the same derivation.
    released = _live_park_rows(
        {"m2129": [row]},
        frozenset({"m2129"}),
        {"m2129": {"record_revision": "sha256:" + "6f" * 32}},
    )
    assert released[SELECTION_EXCLUSION_DISPOSITION] == {}
