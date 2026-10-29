"""H-5: the metadata/fidelity duplicate-accept write door (rule-2, 2026-08-29).

The calibration receipt's ``acceptance_witness`` block binds production
acceptance depth K=2 UNCONDITIONALLY for every gate kind (fable A-4 / Sol
H-5), and the terminal lane has enforced it since AG-4 -- while metadata and
fidelity acceptances stood on one session, so the receipt lied at rung time.
These tests pin the production door that makes it true:

- the v2 conflict-ledger row family: question-bound (``question_sha256``),
  carrying both sessions' FULL verdict projections, ``status`` derived from
  projection equality plus the lane's acceptance kernel -- a forged status,
  tampered projection, or terminal-kind v2 row refuses on read;
- the mixed-era reader: v1 (terminal) rows validate under v1 rules forever;
- witness semantics: a distinct persisted same-question accept is the second
  blind witness; reused rows and the fresh row itself never witness;
- the projection is byte-for-byte the blind-replay matrix's comparison
  surface, so the receipt bar and the production door adjudicate ONE fact;
- the lane vocabulary carries the receipt's bound outcomes
  (``checker-verdict-conflict`` disagreement park,
  ``checker-accept-unconfirmed`` budget-fuse park), and the driver wires the
  door on both accuracy lanes plus the publication belt.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

from menagerie.crawler import checker_conflicts as cc
from menagerie.crawler.checker_calibration import (
    ACCEPTANCE_UNFUNDED_OUTCOME,
    ACCEPTANCE_WITNESS_MINIMUM,
)
from menagerie.crawler.checker_lane import (
    CHECKER_LANE_FAILURE_CLASSES,
    CHECKER_LANE_NON_LADDER_FAILURE_CLASSES,
)

pytestmark = pytest.mark.smoke


def _metadata_gate(gate_id: str = "g1", **item_overrides: object) -> dict:
    item = {
        "stable_id": "m1",
        "work_id": "w1",
        "campaign_root_work_id": "c1",
        "family_representative_id": "m1",
        "vet_identity": "vet-1",
        "fidelity_identity": None,
        "verified_hashes": {"proposal": "a" * 64, "source_manifest": "b" * 64},
        "verdict": "accurate",
        "integrity": {"verdict": "accurate"},
        "rung_check": {"verdict": "accurate"},
    }
    item.update(item_overrides)
    return {
        "gate_kind": "metadata_batch",
        "gate_id": gate_id,
        "gate_identity": f"env-{gate_id}",
        "result_envelope_sha256": "e" * 64,
        "checker": {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "version": "1",
            "prompt_sha256": "p" * 64,
        },
        "items": [item],
    }


def _fidelity_gate(gate_id: str = "f1", **item_overrides: object) -> dict:
    gate = _metadata_gate(gate_id)
    gate["gate_kind"] = "fidelity"
    item = gate["items"][0]
    del item["verdict"]
    item["fidelity"] = {"verdict": "match"}
    item.update(item_overrides)
    return gate


def _build(gate: dict, confirmation: dict, stable_id: str = "m1") -> dict:
    return cc.build_accuracy_duplicate_accept_row(
        gate=gate,
        confirmation=confirmation,
        stable_id=stable_id,
        campaign_id="camp",
        run_id="run",
        machine_id="mach",
        created_at="2026-08-29T00:00:00Z",
        author_result_sha256="d" * 64,
    )


# ---------------------------------------------------------------------------
# Build + derived status.
# ---------------------------------------------------------------------------


def test_agreeing_confirmation_settles_confirmed() -> None:
    gate = _metadata_gate("g1")
    confirmation = _metadata_gate("g2")
    row = _build(gate, confirmation)
    assert row["schema_version"] == cc.CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V2
    assert row["status"] == "confirmed"
    assert row["first_projection"] == {
        "verdict": "accurate",
        "rung_check": "accurate",
        "integrity": "accurate",
    }
    validated = cc.validate_checker_verdict_conflict_row(row)
    assert validated["question_sha256"] == row["question_sha256"]


def test_full_projection_divergence_settles_conflict_even_when_both_accept() -> None:
    """The comparison is the FULL projection, not the acceptance bit."""

    gate = _fidelity_gate("f1")
    confirmation = _fidelity_gate("f2", fidelity={"verdict": "minor-drift"})
    row = _build(gate, confirmation)
    # Both projections pass the fidelity acceptance kernel...
    assert cc.accuracy_projection_accepted("fidelity", row["confirmation_projection"])
    # ...but the projections differ, so the settlement is a conflict.
    assert row["status"] == "conflict"


def test_rejecting_confirmation_settles_conflict() -> None:
    gate = _metadata_gate("g1")
    confirmation = _metadata_gate("g2", rung_check={"verdict": "cannot-verify"})
    assert _build(gate, confirmation)["status"] == "conflict"


def test_rejections_are_never_duplicated() -> None:
    gate = _metadata_gate("g1", verdict="inaccurate")
    with pytest.raises(cc.CheckerVerdictConflictError, match="only for acceptances"):
        _build(gate, _metadata_gate("g2"))


def test_cross_question_confirmation_refuses() -> None:
    gate = _metadata_gate("g1")
    for divergent in (
        _metadata_gate("g2", work_id="w2"),
        _metadata_gate("g2", verified_hashes={"proposal": "f" * 64, "source_manifest": "b" * 64}),
    ):
        with pytest.raises(cc.CheckerVerdictConflictError, match="different question"):
            _build(gate, divergent)
    rotated = _metadata_gate("g2")
    rotated["checker"]["prompt_sha256"] = "q" * 64
    with pytest.raises(cc.CheckerVerdictConflictError, match="different question"):
        _build(gate, rotated)


def test_terminal_rows_stay_v1() -> None:
    gate = _metadata_gate("g1")
    gate["gate_kind"] = "terminal_disposition"
    with pytest.raises(cc.CheckerVerdictConflictError, match="settles only"):
        _build(gate, gate)


# ---------------------------------------------------------------------------
# Validation: forged fields refuse on read.
# ---------------------------------------------------------------------------


def test_forged_status_refuses_on_read() -> None:
    row = copy.deepcopy(_build(_metadata_gate("g1"), _metadata_gate("g2")))
    row["status"] = "conflict"
    with pytest.raises(cc.CheckerVerdictConflictError, match="must re-derive"):
        cc.validate_checker_verdict_conflict_row(row)


def test_tampered_projection_refuses_on_read() -> None:
    row = copy.deepcopy(_build(_metadata_gate("g1"), _metadata_gate("g2")))
    row["confirmation_projection"]["rung_check"] = "cannot-verify"
    with pytest.raises(cc.CheckerVerdictConflictError, match="does not bind"):
        cc.validate_checker_verdict_conflict_row(row)


def test_wrong_projection_keyset_refuses_on_read() -> None:
    row = copy.deepcopy(_build(_fidelity_gate("f1"), _fidelity_gate("f2")))
    row["first_projection"] = {"verdict": "accurate", "rung_check": "accurate", "integrity": "accurate"}
    with pytest.raises(cc.CheckerVerdictConflictError, match="full\nverdict projection|full verdict projection"):
        cc.validate_checker_verdict_conflict_row(row)


def test_v2_terminal_kind_refuses_on_read() -> None:
    row = copy.deepcopy(_build(_metadata_gate("g1"), _metadata_gate("g2")))
    row["gate_kind"] = "terminal_disposition"
    with pytest.raises(cc.CheckerVerdictConflictError):
        cc.validate_checker_verdict_conflict_row(row)


def test_v1_terminal_rows_keep_validating_beside_v2(tmp_path: Path) -> None:
    ledger = tmp_path / "conflicts.jsonl"
    v1 = cc.build_duplicate_accept_row(
        stable_id="m9",
        work_id="w9",
        campaign_id="camp",
        run_id="run",
        machine_id="mach",
        created_at="2026-08-29T00:00:00Z",
        gate_kind="terminal_disposition",
        author_result_sha256="a" * 64,
        gate_id="t1",
        gate_identity="env-t1",
        confirmation_gate_identity="env-t2",
        confirmation_verdict="accepted",
        confirmation_identity={
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "version": "1",
            "prompt_sha256": "p" * 64,
        },
        confirmation_result_envelope_sha256="e" * 64,
    )
    v2 = _build(_metadata_gate("g1"), _metadata_gate("g2"))
    cc.append_checker_verdict_conflict_row(ledger, v1)
    cc.append_checker_verdict_conflict_row(ledger, v2)
    rows = cc.load_checker_verdict_conflict_rows((ledger,))
    assert [row["schema_version"] for row in rows] == [
        cc.CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION,
        cc.CHECKER_VERDICT_CONFLICT_SCHEMA_VERSION_V2,
    ]
    # v1 settlement lookups never see v2 rows and vice versa.
    assert cc.duplicate_accept_state("t1", rows) == "confirmed"
    assert cc.accuracy_duplicate_accept_state("t1", "m9", rows) is None
    assert cc.accuracy_duplicate_accept_state("g1", "m1", rows) == "confirmed"


def test_append_is_idempotent_per_gate_and_item(tmp_path: Path) -> None:
    ledger = tmp_path / "conflicts.jsonl"
    first = cc.append_checker_verdict_conflict_row(
        ledger, _build(_metadata_gate("g1"), _metadata_gate("g2"))
    )
    replay = _build(_metadata_gate("g1"), _metadata_gate("g3"))
    settled = cc.append_checker_verdict_conflict_row(ledger, replay)
    assert settled == first
    assert len(cc.load_checker_verdict_conflict_rows((ledger,))) == 1


# ---------------------------------------------------------------------------
# Witness semantics.
# ---------------------------------------------------------------------------


def test_distinct_same_question_accept_is_the_second_witness() -> None:
    gate = _metadata_gate("g1")
    witness = _metadata_gate("g2")
    assert len(cc.accuracy_acceptance_witnesses(gate, "m1", [witness])) == 1
    # K=2 exactly: the fresh accept plus one distinct witness.
    assert 1 + 1 == ACCEPTANCE_WITNESS_MINIMUM


def test_reused_self_rejecting_and_cross_question_rows_never_witness() -> None:
    gate = _metadata_gate("g1")
    reused = _metadata_gate("g2")
    reused["verdict_source"] = "reused"
    rejecting = _metadata_gate("g3", verdict="inaccurate")
    cross_identity = _metadata_gate("g4")
    cross_identity["checker"]["prompt_sha256"] = "q" * 64
    other_member = _metadata_gate("g5")
    other_member["items"][0]["stable_id"] = "m2"
    assert (
        cc.accuracy_acceptance_witnesses(
            gate, "m1", [gate, reused, rejecting, cross_identity, other_member]
        )
        == []
    )


def test_rejected_fresh_gate_has_no_witnesses() -> None:
    gate = _metadata_gate("g1", verdict="inaccurate")
    assert cc.accuracy_acceptance_witnesses(gate, "m1", [_metadata_gate("g2")]) == []


# ---------------------------------------------------------------------------
# The projection IS the matrix's comparison surface (anti-drift mirror).
# ---------------------------------------------------------------------------


def test_projection_matches_the_blind_replay_matrix_surface() -> None:
    from menagerie.crawler.tools.checker_calibration_runner import verdict_projection

    metadata_gate = _metadata_gate("g1")
    fidelity_gate = _fidelity_gate("f1")
    for gate in (metadata_gate, fidelity_gate):
        kind = str(gate["gate_kind"])
        item = gate["items"][0]
        assert cc.accuracy_item_projection(kind, item) == verdict_projection(kind, gate)["m1"]


def test_projection_acceptance_rederivation_matches_the_item_kernels() -> None:
    cases = [
        ("metadata_batch", _metadata_gate("g", verdict="accurate")["items"][0], True),
        ("metadata_batch", _metadata_gate("g", verdict="cannot-verify")["items"][0], False),
        (
            "metadata_batch",
            _metadata_gate("g", integrity={"verdict": "inaccurate"})["items"][0],
            False,
        ),
        ("fidelity", _fidelity_gate("f")["items"][0], True),
        ("fidelity", _fidelity_gate("f", fidelity={"verdict": "minor-drift"})["items"][0], True),
        ("fidelity", _fidelity_gate("f", fidelity={"verdict": "cannot-verify"})["items"][0], False),
        (
            "fidelity",
            _fidelity_gate("f", rung_check={"verdict": "cannot-verify"})["items"][0],
            False,
        ),
    ]
    for kind, item, expected in cases:
        assert cc.accuracy_item_accepted(kind, item) is expected
        projection = cc.accuracy_item_projection(kind, item)
        assert cc.accuracy_projection_accepted(kind, projection) is expected


# ---------------------------------------------------------------------------
# Vocabulary + wiring.
# ---------------------------------------------------------------------------


def test_lane_vocabulary_carries_the_receipt_bound_outcomes() -> None:
    assert ACCEPTANCE_UNFUNDED_OUTCOME == "checker-accept-unconfirmed"
    assert "checker-accept-unconfirmed" in CHECKER_LANE_FAILURE_CLASSES
    assert "checker-accept-unconfirmed" in CHECKER_LANE_NON_LADDER_FAILURE_CLASSES
    assert "checker-verdict-conflict" in CHECKER_LANE_FAILURE_CLASSES


def test_gate_lane_wires_every_accuracy_acceptance_site() -> None:
    from menagerie.crawler.driver_admission import AdmissionEnvironmentMixin

    source = inspect.getsource(AdmissionEnvironmentMixin._run_gate_rounds)
    # The three acceptance sites: the post-round metadata door, the fidelity
    # acceptance break, and the fidelity-repair metadata RE-CHECK (the second
    # metadata dispatch site -- an acceptance minted there without the bound
    # confirmation would bypass K=2).
    assert source.count("_require_confirmed_accuracy_acceptance(") == 3
    assert '"metadata_batch", reducer, operational' in source
    door = inspect.getsource(
        AdmissionEnvironmentMixin._require_confirmed_accuracy_acceptance
    )
    # Disagreement parks as the conflict class; the budget fuse parks
    # unconfirmed; the confirmation is a live session, never a pool capsule.
    assert door.count('failure_class="checker-verdict-conflict"') >= 2
    assert 'failure_class="checker-accept-unconfirmed"' in door
    assert "except CheckerWallBudgetExceeded" in door
    assert "pool.claim" not in door


def test_publication_belt_requires_the_witnessed_acceptance() -> None:
    from menagerie.crawler.driver import CrawlerDriver

    source = inspect.getsource(CrawlerDriver._authorize_terminal_artifact)
    assert "accuracy_duplicate_accept_state" in source
    assert "accuracy_acceptance_witnesses" in source
    assert "no settled K=2" in source
