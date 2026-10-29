"""RC-2: the measured-rotations ledger (``records/control/rotations.jsonl``).

Pins the closed row vocabulary, the axis-to-instrument binding, the seed-only
``declared`` method (structural, at append AND at load), sticky soundness,
append-order independence (Q1: the live identity is the caller's authority
context, never a ledger row), the seed from real pilot registry rows, and
golden-replay classification from a REAL blind-replay report.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.rotation_registry import (
    append_rotation_registration,
    build_rotation_registration,
    load_rotation_registrations,
    rotation_registry_path,
)
from menagerie.crawler.rotations import (
    CODE_AXES,
    MEASUREMENT_AUTHORITY,
    PRODUCER_AXES,
    RotationLedgerError,
    append_rotation,
    build_rotation,
    classify_golden_replay,
    edge_classes,
    load_rotations,
    move_class,
    rotations_path,
    seed_from_registry,
    seed_rows_from_registry,
)

pytestmark = pytest.mark.smoke

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_A, _B, _C = ("sha256:" + ch * 64 for ch in "abc")


def _row(
    *,
    axis: str = "award_closure_identity",
    from_identity: str = _A,
    to_identity: str = _B,
    rotation_class: str = "refinement",
    measured_by: str = "predicate_replay",
    at: str = "2026-09-01T00:00:00Z",
    reason: str = "probe: 2 heads (1 award) replayed; 0 flips",
) -> dict[str, Any]:
    return build_rotation(
        axis=axis,
        from_identity=from_identity,
        to_identity=to_identity,
        rotation_class=rotation_class,
        reason=reason,
        measured_by=measured_by,
        tree_sha="deadbeef",
        at=at,
    )


def test_rows_are_closed_content_addressed_and_axis_bound() -> None:
    """Closed vocabularies, content-addressed ids, one instrument per axis."""

    row = _row()
    assert row["rotation_id"].startswith("rot-")
    assert set(row) == {
        "schema_version", "rotation_id", "axis", "from", "to", "class", "reason",
        "measured_by", "tree_sha", "at",
    }
    with pytest.raises(RotationLedgerError, match="unknown class"):
        _row(rotation_class="cosmetic")
    with pytest.raises(RotationLedgerError, match="does not rotate"):
        _row(from_identity=_A, to_identity=_A)
    # Q4: a checker blind replay cannot vouch for a code axis, a predicate
    # replay cannot vouch for the checker prompt, and a producer axis has no
    # instrument at all.
    with pytest.raises(RotationLedgerError, match="measurable only by 'predicate_replay'"):
        _row(axis="award_closure_identity", measured_by="golden_replay")
    with pytest.raises(RotationLedgerError, match="measurable only by 'golden_replay'"):
        _row(axis="checker_prompt_identity", measured_by="predicate_replay")
    for axis in sorted(PRODUCER_AXES):
        assert MEASUREMENT_AUTHORITY[axis] is None
        for method in ("golden_replay", "predicate_replay"):
            with pytest.raises(RotationLedgerError, match="no replay instrument"):
                _row(axis=axis, measured_by=method)
    assert CODE_AXES == {
        "terminal_rule_identity", "publication_policy_identity", "award_closure_identity"
    }


def test_append_is_idempotent_chain_linked_and_conflict_refusing(tmp_path: Path) -> None:
    row = _row()
    assert append_rotation(tmp_path, row) is True
    assert append_rotation(tmp_path, row) is False
    persisted = [json.loads(line) for line in rotations_path(tmp_path).read_text().splitlines()]
    assert len(persisted) == 1 and "chain" in persisted[0]
    forged = dict(row)
    forged["reason"] = "rewritten"
    with pytest.raises(RotationLedgerError, match="does not match its body"):
        append_rotation(tmp_path, forged)
    loaded = load_rotations(tmp_path)
    assert [entry["rotation_id"] for entry in loaded] == [row["rotation_id"]]
    # A torn or edited ledger refuses at load rather than degrading to "no rotations".
    rotations_path(tmp_path).write_text(rotations_path(tmp_path).read_text().replace("deadbeef", "d3adbeef"))
    with pytest.raises(RotationLedgerError, match="refuses"):
        load_rotations(tmp_path)


def _registry_rows(records_root: Path, *rows: Mapping[str, Any]) -> None:
    for row in rows:
        append_rotation_registration(rotation_registry_path(records_root), row)


def _registration(*, from_identity: str, to_identity: str, rotation_class: str, at: str) -> dict:
    return build_rotation_registration(
        axis="award_closure_identity",
        from_identity=from_identity,
        to_identity=to_identity,
        rotation_class=rotation_class,
        rationale="test registration",
        registered_by="operator-test",
        created_at=at,
    )


def test_declared_is_seed_only_at_append_and_at_load(tmp_path: Path) -> None:
    """Q5: no operator write path to a positive class."""

    declared = _row(measured_by="declared", reason="hand-declared refinement")
    with pytest.raises(RotationLedgerError, match="outside the frozen-registry seed"):
        append_rotation(tmp_path, declared)
    assert not rotations_path(tmp_path).exists()

    # The seed importer is the one lawful source of declared rows.
    _registry_rows(
        tmp_path,
        _registration(from_identity=_A, to_identity=_B, rotation_class="refinement", at="2026-08-20T00:00:00Z"),
        _registration(from_identity=_B, to_identity=_C, rotation_class="soundness", at="2026-08-21T00:00:00Z"),
    )
    result = seed_from_registry(tmp_path, tree_sha="deadbeef")
    assert result == {"registry_rows": 2, "appended": 2, "already_present": 0}
    assert seed_from_registry(tmp_path, tree_sha="deadbeef")["appended"] == 0
    rows = load_rotations(tmp_path)
    assert [row["measured_by"] for row in rows] == ["declared", "declared"]
    assert [row["class"] for row in rows] == ["refinement", "soundness"]

    # A forger who smuggles the allowlist past append still fails at LOAD:
    # the frozen registry does not derive the row.
    forgery = _row(
        from_identity=_C, to_identity="sha256:" + "d" * 64, measured_by="declared",
        reason="forged declared refinement",
    )
    assert append_rotation(tmp_path, forgery, seed_allowlist=frozenset({forgery["rotation_id"]}))
    with pytest.raises(RotationLedgerError, match="outside the frozen-registry seed"):
        load_rotations(tmp_path)


def test_declared_rows_need_the_registry_present(tmp_path: Path) -> None:
    declared = _row(measured_by="declared")
    append_rotation(tmp_path, declared, seed_allowlist=frozenset({declared["rotation_id"]}))
    with pytest.raises(RotationLedgerError, match="need the frozen registry"):
        load_rotations(tmp_path)


def test_soundness_is_sticky_and_move_class_is_order_independent() -> None:
    """Q1: a commutative OR plus set reachability; append order never matters."""

    refinement_ab = _row(from_identity=_A, to_identity=_B, rotation_class="refinement")
    soundness_ab = _row(from_identity=_A, to_identity=_B, rotation_class="soundness", at="2026-09-02T00:00:00Z")
    refinement_bc = _row(from_identity=_B, to_identity=_C, rotation_class="refinement")
    key = {"axis": "award_closure_identity"}
    rows = (refinement_ab, soundness_ab, refinement_bc)
    for order in itertools.permutations(rows):
        assert edge_classes(order)[("award_closure_identity", _A, _B)] == "soundness"
        assert move_class(order, recorded=_A, live=_C, **key) == "soundness"
        assert move_class(order, recorded=_B, live=_C, **key) == "refinement"
    assert move_class((refinement_ab, refinement_bc), recorded=_A, live=_C, **key) == "refinement"
    assert move_class((), recorded=_A, live=_C, **key) is None
    assert move_class((), recorded=_A, live=_A, **key) == "refinement"
    # The live side is whatever the caller's authority context says; a row
    # pointing at some other "newest" identity never redirects the question.
    stray = _row(from_identity=_C, to_identity="sha256:" + "e" * 64, rotation_class="refinement")
    assert move_class((refinement_ab, refinement_bc, stray), recorded=_A, live=_C, **key) == "refinement"


def test_seed_from_real_pilot_checker_prompt_rows_resolves_the_disputed_edge() -> None:
    """Q2 on real bytes: the pilot's checker_prompt rows (6 refinement, 2 soundness)."""

    registry = _FIXTURES / "pilot_checker_prompt_registrations.jsonl"
    rows = load_rotation_registrations(registry)
    assert len(rows) == 8
    seeds = seed_rows_from_registry(rows, tree_sha="deadbeef")
    assert len(seeds) == 8 and all(seed["measured_by"] == "declared" for seed in seeds)
    supersessions = [row for row in rows if row.get("supersedes")]
    assert len(supersessions) == 2
    live = "sha256:d7d0043f"
    disputed_from = "sha256:21c694bb"
    (edge,) = [
        (source, target)
        for (axis, source, target) in edge_classes(seeds)
        if source.startswith(disputed_from) and target.startswith(live)
    ]
    key = {"axis": "checker_prompt_identity"}
    assert move_class(seeds, recorded=edge[0], live=edge[1], **key) == "soundness"
    # The a032320a heads traverse the same soundness edge through a refinement hop.
    (older,) = [source for (axis, source, target) in edge_classes(seeds) if target == edge[0] and source.startswith("sha256:a032320a")]
    assert move_class(seeds, recorded=older, live=edge[1], **key) == "soundness"
    assert move_class(seeds, recorded=older, live=edge[0], **key) == "refinement"
    for order in itertools.permutations(seeds[:4]):
        assert move_class((*order, *seeds[4:]), recorded=older, live=edge[1], **key) == "soundness"


def _real_blind_replay_report(rotated_verdict: str) -> dict[str, Any]:
    """Run the REAL blind-replay matrix harness with fake adjudicators."""

    from menagerie.crawler.tools.blind_replay_matrix import run_blind_replay_matrix

    goldens = [
        {"name": "metadata_accept", "kind": "metadata_batch", "width": 1, "request": "req-m"},
        {"name": "terminal_accept", "kind": "terminal_disposition", "width": 1, "request": "req-t"},
    ]

    def same_epoch(golden: Mapping[str, Any]) -> dict[str, Any]:
        verdicts = {"m1": "accurate"} if golden["kind"] == "metadata_batch" else {"m9": "accepted"}
        return {"verdicts": verdicts, "latency_seconds": 1.0}

    def rotated(golden: Mapping[str, Any]) -> dict[str, Any]:
        if golden["kind"] == "metadata_batch":
            return {"verdicts": {"m1": rotated_verdict}, "latency_seconds": 1.0}
        return same_epoch(golden)

    return run_blind_replay_matrix(
        goldens,
        adjudicate=same_epoch,
        rotation_adjudicate=rotated,
        replays_per_golden=3,
        checker_model="fixture-checker",
        captured_at="2026-09-01T00:00:00Z",
        production_batch_width=1,
    )


def test_golden_replay_classifies_exact_verdict_flips_for_the_checker_axis_only() -> None:
    stable = _real_blind_replay_report("accurate")
    assert classify_golden_replay(stable, axis="checker_prompt_identity")["class"] == "refinement"
    flipped = _real_blind_replay_report("inaccurate")
    verdict = classify_golden_replay(flipped, axis="checker_prompt_identity")
    assert verdict["class"] == "soundness"
    assert [entry["stable_id"] for entry in verdict["flipped"]] == ["m1"]
    # Q4: the instrument never runs the author; it cannot classify an author axis.
    with pytest.raises(RotationLedgerError, match="does not exercise author_prompt_identity"):
        classify_golden_replay(stable, axis="author_prompt_identity")
    partial = json.loads(json.dumps(stable))
    partial["goldens"][0]["rotation"] = {"ran": False}
    with pytest.raises(RotationLedgerError, match="did not run for every golden"):
        classify_golden_replay(partial, axis="checker_prompt_identity")
