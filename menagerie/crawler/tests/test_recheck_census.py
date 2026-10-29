"""RC-2: the recheck census, the RECHECK mode, and the predicate replay probe.

Heads are appended through the REAL reducer (never hand-built dicts), so the
recorded vectors, statuses, and projection reasons are production-shaped.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import pytest

from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recheck_census import (
    award_release_lever,
    predicate_replay_probe,
    recheck_authorities,
    recheck_census,
    recheck_census_from_records,
    recheck_targets,
)
from menagerie.crawler.reducer import (
    _award_recorded_basis_context,
    project_dependency_current,
)
from menagerie.crawler.rotations import append_rotation, build_rotation
from menagerie.crawler.tests.conftest import make_authority_context
from menagerie.crawler.tests.test_award_recorded_basis import _award_head
from menagerie.crawler.tests.test_reducer import CanonicalReducer
from menagerie.crawler.work_selection import (
    MemberFacts,
    SelectionDecision,
    SelectionMode,
    build_selection_plan,
)

pytestmark = pytest.mark.smoke

_ROTATED = "sha256:" + "b" * 64
_AXIS = "publication_policy_identity"


def _paths(tmp_path: Path) -> LedgerPaths:
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    return LedgerPaths(
        models=records / "models.jsonl",
        attempts=records / "attempts.jsonl",
        gates=records / "gates.jsonl",
    )


def _two_heads(tmp_path: Path) -> tuple[LedgerPaths, dict, dict]:
    """One released award (m_example) and one failed:forward terminal (m_other)."""

    paths = _paths(tmp_path)
    award = _award_head(paths)
    attempt_paths = paths  # same ledgers
    from menagerie.crawler.tests.conftest import bind_terminal_attempts, make_failed_attempt, make_model

    attempt = make_failed_attempt(stable_id="m_other", attempt_id="attempt-other", stage="forward", reason_code="exception")
    attempt["ledger_seq"] = 2  # the award's attempt holds local sequence 1
    model = make_model(stable_id="m_other", status_code="failed:forward", attempt_id="attempt-other")
    model["status"]["reason_code"] = "exception"
    model.pop("record_seq", None)  # the reducer assigns the ledger sequence
    bind_terminal_attempts(model, [attempt])
    with CanonicalReducer(attempt_paths, ["m_example", "m_other"]) as reducer:
        reducer.append_attempt(attempt)
        failed = reducer.append_model(model).record
    return paths, award, failed


def _held_head(paths: LedgerPaths, stable_id: str = "m_held") -> dict:
    """Append one review-held failed:source terminal (the leg-4 crash-chain shape)."""

    from menagerie.crawler.tests.conftest import bind_terminal_attempts, make_failed_attempt, make_model

    attempt = make_failed_attempt(stable_id=stable_id, attempt_id=f"attempt-{stable_id}")
    attempt["ledger_seq"] = 3
    model = bind_terminal_attempts(make_model(stable_id=stable_id, status_code="failed:source", attempt_id=f"attempt-{stable_id}"), [attempt])
    model.pop("record_seq", None)
    model["status"]["human_review"] = {
        "required": True, "reason": None, "queue": "crawler-human-review", "requested_at": model["created_at"],
    }
    with CanonicalReducer(paths, ["m_example", "m_other", stable_id]) as reducer:
        reducer.append_attempt(attempt)
        return reducer.append_model(model).record


def _heads(paths: LedgerPaths) -> dict:
    from menagerie.crawler.reducer import _select_current, scan_jsonl

    return _select_current(scan_jsonl(paths.models))


def _measured(*, from_identity: str, to_identity: str, rotation_class: str, at: str = "2026-09-01T00:00:00Z") -> dict:
    return build_rotation(
        axis=_AXIS,
        from_identity=from_identity,
        to_identity=to_identity,
        rotation_class=rotation_class,
        reason="probe: 2 heads (1 award) replayed; 0 flips",
        measured_by="predicate_replay",
        tree_sha="deadbeef",
        at=at,
    )


def test_every_head_kind_projects_current_under_a_rotation_and_lands_in_the_census(
    tmp_path: Path,
) -> None:
    """D.2.3/B.4: currency is unconditional; the class feeds the census only.

    RED against the pre-RC-2 projection, which raised the award-release-hold
    for the award and (after tl-soundstale) vouched the failed terminal.
    """

    paths, award, failed = _two_heads(tmp_path)
    context = make_authority_context(["m_example", "m_other"])
    rotated = replace(context, publication_policy_identity=_ROTATED)

    projection = project_dependency_current(paths, context=rotated)
    assert set(projection.current_records) == {"m_example", "m_other"}, projection.stale_reasons
    assert projection.stale_reasons == {}

    # An UNMEASURED move (no row connects recorded -> live) reads soundness:
    # both heads, award and terminal alike, are needs_recheck.
    census = recheck_census(_heads(paths), (), rotated, scope=["m_example", "m_other"])
    assert census["counts"] == {
        "needs_recheck": 2,
        "by_head_kind": {"failed": 1, "runs": 1},
        "unmeasured": 2,
        "review_held": 0,
    }
    assert census["by_axis"] == {_AXIS: ["m_example", "m_other"]}
    rows = {row["stable_id"]: row for row in census["needs_recheck"]}
    assert rows["m_example"]["class"] == "soundness"
    assert rows["m_example"]["soundness_basis"] == "unmeasured"
    assert rows["m_example"]["rotation_ids"] == []
    assert rows["m_example"]["remedy"] == "re-derive"
    assert rows["m_example"]["recorded"] == award["dependency_vector"][_AXIS]
    assert rows["m_example"]["live"] == _ROTATED
    assert rows["m_other"]["head_kind"] == "failed"
    assert set(rows["m_example"]) == {
        "stable_id", "axis", "recorded", "live", "class", "soundness_basis", "rotation_ids",
        "head_kind", "status_code", "record_revision", "remedy", "review_held",
    }
    assert recheck_targets(census) == {"m_example", "m_other"}
    assert recheck_authorities(census) == {
        "m_example": f"unmeasured:{_AXIS}",
        "m_other": f"unmeasured:{_AXIS}",
    }
    assert "--blocked-award-revision " + award["record_revision"] in award_release_lever(rows["m_example"])

    # The unrotated context owes nothing; scope excludes out-of-intake heads.
    assert recheck_census(_heads(paths), (), context)["counts"]["needs_recheck"] == 0
    assert recheck_census(_heads(paths), (), rotated, scope=["m_other"])["by_axis"] == {_AXIS: ["m_other"]}


def test_measured_refinement_clears_and_measured_soundness_names_its_rows(tmp_path: Path) -> None:
    paths, award, _failed = _two_heads(tmp_path)
    context = make_authority_context(["m_example", "m_other"])
    rotated = replace(context, publication_policy_identity=_ROTATED)
    recorded = str(award["dependency_vector"][_AXIS])
    records_root = paths.models.parent

    refinement = _measured(from_identity=recorded, to_identity=_ROTATED, rotation_class="refinement")
    append_rotation(records_root, refinement)
    census = recheck_census_from_records(records_root, rotated, ledgers=paths)
    assert census["counts"]["needs_recheck"] == 0

    soundness = _measured(
        from_identity=recorded, to_identity=_ROTATED, rotation_class="soundness", at="2026-09-02T00:00:00Z"
    )
    append_rotation(records_root, soundness)
    census = recheck_census_from_records(records_root, rotated, ledgers=paths)
    assert census["counts"] == {"needs_recheck": 2, "by_head_kind": {"failed": 1, "runs": 1}, "unmeasured": 0, "review_held": 0}
    for row in census["needs_recheck"]:
        assert row["soundness_basis"] == "measured"
        assert row["rotation_ids"] == [soundness["rotation_id"]]
    assert recheck_authorities(census)["m_example"] == soundness["rotation_id"]


def test_census_is_append_order_independent_and_live_comes_from_the_context(tmp_path: Path) -> None:
    """Q1: two forked append orders, one census; a stray row never redirects live."""

    paths, award, _failed = _two_heads(tmp_path)
    context = make_authority_context(["m_example", "m_other"])
    live = _ROTATED
    recorded = str(award["dependency_vector"][_AXIS])
    hop = "sha256:" + "c" * 64
    rows = (
        _measured(from_identity=recorded, to_identity=hop, rotation_class="refinement"),
        _measured(from_identity=hop, to_identity=live, rotation_class="soundness", at="2026-09-02T00:00:00Z"),
        _measured(from_identity=hop, to_identity=live, rotation_class="refinement", at="2026-09-03T00:00:00Z"),
        # A later row past the live identity: order-sensitive "newest" logic
        # would have chased it; the census asks about the CONTEXT's live value.
        _measured(from_identity=live, to_identity="sha256:" + "e" * 64, rotation_class="refinement", at="2026-09-04T00:00:00Z"),
    )
    heads = _heads(paths)
    rotated = replace(context, publication_policy_identity=live)
    expected = recheck_census(heads, rows, rotated)
    assert expected["counts"]["needs_recheck"] == 2
    assert expected["needs_recheck"][0]["rotation_ids"] == [rows[1]["rotation_id"]]
    for order in itertools.permutations(rows):
        assert recheck_census(heads, order, rotated) == expected
    # Two forked ledgers with different append orders read identically.
    for name, order in (("fork-a", rows), ("fork-b", tuple(reversed(rows)))):
        root = tmp_path / name
        for row in order:
            append_rotation(root, row)
    from menagerie.crawler.rotations import load_rotations

    assert recheck_census(heads, load_rotations(tmp_path / "fork-a"), rotated) == recheck_census(
        heads, load_rotations(tmp_path / "fork-b"), rotated
    )


def test_recorded_basis_context_pins_unconditionally_and_still_refuses_unattested_generations(
    tmp_path: Path,
) -> None:
    paths, award, _failed = _two_heads(tmp_path)
    context = make_authority_context(["m_example", "m_other"])
    rotated = replace(context, publication_policy_identity=_ROTATED)
    pinned = _award_recorded_basis_context(rotated, award, attested_bindings={"env-test-gen": "core"} if False else {award["dependency_vector"]["environment_generation"]: "env-test"})
    assert pinned.publication_policy_identity == award["dependency_vector"][_AXIS]
    rebuilt = replace(context, environment_generations={"env-test": "sha256:" + "e" * 64})
    with pytest.raises(Exception, match="attested"):
        _award_recorded_basis_context(rebuilt, award, attested_bindings={})


def _facts(head: dict, *, dependency_current: bool = True) -> MemberFacts:
    return MemberFacts(
        stable_id=str(head["stable_id"]),
        raw_head=head,
        dependency_current=dependency_current,
        stale_reason=None,
        parked=False,
        deferral_live=False,
        live_grant=None,
        grant_bindable=False,
        active_generation=None,
        active_generation_discharged=False,
    )


def test_recheck_mode_admits_census_targets_and_keeps_award_custody(tmp_path: Path) -> None:
    paths, award, failed = _two_heads(tmp_path)
    context = make_authority_context(["m_example", "m_other"])
    rotated = replace(context, publication_policy_identity=_ROTATED)
    census = recheck_census(_heads(paths), (), rotated)
    plan = build_selection_plan(
        [_facts(award), _facts(failed)],
        mode=SelectionMode.RECHECK,
        mode_targets=recheck_targets(census),
        recheck_authorities=recheck_authorities(census),
    )
    decisions = {member.stable_id: member for member in plan.members}
    assert decisions["m_other"].admitted is True
    assert decisions["m_other"].decision is SelectionDecision.RECHECK
    assert decisions["m_other"].work_authority == f"unmeasured:{_AXIS}"
    # AW-1 custody unchanged: the award waits for its named grant.
    assert decisions["m_example"].admitted is False
    assert decisions["m_example"].decision is SelectionDecision.AWARD_CUSTODY
    # A review-held target is disclosed in the census (review_held) but the
    # census alone never admits it: its supersession needs the review grant.
    held = _held_head(paths)
    census_held = recheck_census(_heads(paths), (), rotated)
    assert census_held["counts"]["review_held"] == 1
    held_row = next(row for row in census_held["needs_recheck"] if row["stable_id"] == "m_held")
    assert held_row["review_held"] is True and held_row["head_kind"] == "failed"
    held_plan = build_selection_plan(
        [_facts(held)], mode=SelectionMode.RECHECK, mode_targets=recheck_targets(census_held)
    )
    (held_decision,) = held_plan.members
    assert held_decision.admitted is False
    assert held_decision.decision is SelectionDecision.REVIEW_CUSTODY
    # Outside RECHECK mode the same census admits nothing.
    ordinary = build_selection_plan([_facts(award), _facts(failed)], mode=SelectionMode.ORDINARY)
    assert not ordinary.admitted_ids
    # A non-target settled head in RECHECK mode stays settled.
    partial = build_selection_plan(
        [_facts(award), _facts(failed)], mode=SelectionMode.RECHECK, mode_targets=frozenset({"m_other"})
    )
    assert {m.stable_id for m in partial.members if m.admitted} == {"m_other"}


def test_predicate_replay_probe_reports_zero_flips_then_an_exact_disposition_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q4: the probe compares recorded vs derived dispositions, never "replay didn't fail"."""

    import dataclasses

    from menagerie.crawler import reducer as reducer_module

    paths, _award, _failed = _two_heads(tmp_path)
    context = make_authority_context(["m_example", "m_other"])
    report = predicate_replay_probe(paths, context)
    assert report["heads"] == 2 and report["awards"] == 1
    assert report["flips"] == [] and report["changed_input_stale"] == []
    assert report["class"] == "refinement"
    assert report["measures_axes"] == [
        "award_closure_identity", "publication_policy_identity", "terminal_rule_identity",
    ]

    # A predicate edit that re-derives the failed terminal as a different
    # failure disposition: the probe names both values.
    real = reducer_module.CanonicalReducer._derive_model_authority

    def edited(self, model):  # type: ignore[no-untyped-def]
        proof, family, vector, artifacts = real(self, model)
        if proof.status_code == "failed:forward":
            proof = dataclasses.replace(proof, status_code="failed:import", failure_stage="import")
        return proof, family, vector, artifacts

    monkeypatch.setattr(reducer_module.CanonicalReducer, "_derive_model_authority", edited)
    flipped = predicate_replay_probe(paths, context)
    assert flipped["class"] == "soundness"
    (flip,) = flipped["flips"]
    assert flip["stable_id"] == "m_other"
    assert flip["disposition_diff"]["code"] == {"recorded": "failed:forward", "derived": "failed:import"}
    assert flip["disposition_diff"]["stage"] == {"recorded": "forward", "derived": "import"}

    # A predicate edit that turns the prior non-acceptance into an acceptance
    # is a flip too (the kind moved), with the award untouched.
    def promoted(self, model):  # type: ignore[no-untyped-def]
        proof, family, vector, artifacts = real(self, model)
        if proof.status_code == "failed:forward":
            proof = dataclasses.replace(proof, status_code="runs", failure_stage=None, reason_code=None, root_cause_fingerprint=None)
        return proof, family, vector, artifacts

    monkeypatch.setattr(reducer_module.CanonicalReducer, "_derive_model_authority", promoted)
    (flip,) = predicate_replay_probe(paths, context)["flips"]
    assert flip["disposition_diff"]["kind"] == {"recorded": "failed", "derived": "runs"}


def test_predicate_replay_probe_segregates_changed_inputs(tmp_path: Path) -> None:
    paths, _award, _failed = _two_heads(tmp_path)
    context = make_authority_context(["m_example", "m_other"])
    paths.attempts.write_text("")
    report = predicate_replay_probe(paths, context)
    assert report["flips"] == [], report["flips"]
    assert {entry["stable_id"] for entry in report["changed_input_stale"]} == {"m_example", "m_other"}
    assert report["class"] == "refinement"
    empty = predicate_replay_probe(paths, context, scope=[])
    assert empty["heads"] == 0 and empty["class"] is None


def test_predicate_replay_probe_segregates_heads_outside_the_intake(tmp_path: Path) -> None:
    """A ledger head the replaying intake does not contain is a changed input, not a flip."""

    paths, _award, _failed = _two_heads(tmp_path)
    narrow = make_authority_context(["m_example"])
    report = predicate_replay_probe(paths, narrow)
    assert report["heads"] == 2 and report["flips"] == []
    (outside,) = report["changed_input_stale"]
    assert outside["stable_id"] == "m_other" and "outside intake" in outside["reason"]
    assert report["class"] == "refinement"


def test_kernel_closure_declaration_names_the_live_predicate_replay_identities() -> None:
    """CI obligation (D.2.2): a kernel edit cannot ship without acknowledging the move."""

    from menagerie.crawler.tools.register_authority_rotation import (
        ACKNOWLEDGE_COMMAND,
        KERNEL_CLOSURE_DECLARATION_PATH,
        _live_identities,
        load_kernel_closure_declaration,
        undeclared_code_axes,
    )

    declaration = load_kernel_closure_declaration(KERNEL_CLOSURE_DECLARATION_PATH)
    moved = undeclared_code_axes(declaration, _live_identities())
    assert not moved, (
        "the kernel closure moved without a declaration -- run "
        f"`{ACKNOWLEDGE_COMMAND}` in the same change: {json.dumps(moved, indent=1)}"
    )
    for axis, entry in declaration["axes"].items():
        measurement = entry.get("measurement")
        if measurement is not None:
            assert measurement["class"] in {"refinement", "soundness", None}


def test_release_views_exclude_needs_recheck_members_without_touching_currency(tmp_path: Path) -> None:
    """D.2.3: needs_recheck is release-INELIGIBLE, current, covered, still an award."""

    from menagerie.crawler.status import (
        completeness_report,
        record_is_release_eligible,
        release_view_members,
    )

    paths, award, _failed = _two_heads(tmp_path)
    context = make_authority_context(["m_example", "m_other"])
    rotated = replace(context, publication_policy_identity=_ROTATED)
    current = project_dependency_current(paths, context=rotated).current_records
    assert record_is_release_eligible(award, current) is True
    released = release_view_members(current.values(), current)
    assert [record["stable_id"] for record in released] == ["m_example"]
    assert release_view_members(current.values(), current, needs_recheck={"m_example"}) == []
    report = completeness_report(
        ["m_example", "m_other"], current, needs_recheck={"m_example"}
    )
    assert list(report.incomplete_by_issue.get("needs_recheck", ())) == ["m_example"]
    assert report.complete is False
    # The award stays an award: nothing rewrote the head.
    assert current["m_example"]["status"]["kind"] == "runs"
