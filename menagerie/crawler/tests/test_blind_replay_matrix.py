"""B1-XV2 blind-replay matrix harness: raw facts in, receipt block re-derived.

The harness records per-golden RAW comparison facts (no producer-written
aggregates); the receipt mint re-derives every number and the receipt loader
enforces the acceptance floors (>= 3 same-epoch replays, zero same-epoch and
zero width-invariance disagreement, rotation replay ran). These tests pin the
comparison semantics -- same-epoch equality, rotation disclosure, shared-item
width invariance, production-width latency collection -- and the end-to-end
pairing with the mint's ``derive_blind_replay_block``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.checker_calibration_mint import (
    derive_blind_replay_block,
    load_blind_replay_report,
)
from menagerie.crawler.tools.blind_replay_matrix import (
    BlindReplayMatrixError,
    load_golden_corpus_entries,
    run_blind_replay_matrix,
    write_report,
)

pytestmark = pytest.mark.smoke


def _goldens() -> list[dict[str, Any]]:
    return [
        {
            "name": "metadata_accept",
            "kind": "metadata_batch",
            "width": 5,
            "request": "req-metadata",
            "width_variants": {1: "req-metadata-w1"},
        },
        {
            "name": "terminal_accept",
            "kind": "terminal_disposition",
            "width": 1,
            "request": "req-terminal",
        },
    ]


def _stable(golden: Mapping[str, Any]) -> dict[str, Any]:
    verdicts = (
        {"m1": "accurate", "m2": "accurate"}
        if golden["kind"] == "metadata_batch"
        else {"m9": "accepted"}
    )
    if golden.get("width") == 1 and golden["kind"] == "metadata_batch":
        verdicts = {"m1": "accurate"}
    return {"verdicts": verdicts, "latency_seconds": 80.0}


def test_stable_matrix_yields_a_receipt_grade_block(tmp_path: Path) -> None:
    # The harness is width-agnostic machinery: this test pins its behavior on
    # a width-5-shaped synthetic corpus, so the production width is passed
    # EXPLICITLY rather than inherited from the (now width-1) constant.
    report = run_blind_replay_matrix(
        _goldens(),
        _stable,
        rotation_adjudicate=_stable,
        production_batch_width=5,
        checker_model="gpt-5.6-sol",
        captured_at="2026-08-24T12:00:00Z",
    )
    path = write_report(report, tmp_path / "report.json")
    block = derive_blind_replay_block(load_blind_replay_report(path), report_path=path)
    assert block["goldens"] == 2
    assert block["replays_per_golden"] == 3
    assert block["same_epoch_disagreements"] == 0
    assert block["rotation_replay_ran"] is True
    assert block["rotation_disagreements"] == 0
    assert block["width_invariance"] == {
        "widths": [1, 5],
        "disagreements": 0,
        # A real second width exists in this synthetic corpus, so the arm is
        # NOT degenerate and nothing folded.
        "degenerate_at_production_width": False,
        "folded_into_same_epoch": False,
    }
    assert block["p99_at_production_width_seconds"] == pytest.approx(80.0)
    # 3 same-epoch replays + 1 width-1 variant: only the width-5 calls count.
    assert len(report["latency_seconds_at_production_width"]) == 3


def test_same_epoch_disagreement_is_per_golden_fact() -> None:
    answers = iter(
        [
            {"verdicts": {"m1": "accurate"}, "latency_seconds": 10.0},
            {"verdicts": {"m1": "inaccurate"}, "latency_seconds": 10.0},
            {"verdicts": {"m1": "accurate"}, "latency_seconds": 10.0},
        ]
    )
    report = run_blind_replay_matrix(
        [{"name": "g", "kind": "metadata_batch", "width": 5, "request": "r"}],
        lambda golden: next(answers),
        rotation_adjudicate=lambda golden: {
            "verdicts": {"m1": "accurate"},
            "latency_seconds": 10.0,
        },
        checker_model="gpt-5.6-sol",
        captured_at="2026-08-24T12:00:00Z",
    )
    row = report["goldens"][0]
    assert row["same_epoch"]["disagreement"] is True
    assert row["rotation"]["disagreement"] is False


def test_rotation_flip_is_recorded_not_constrained() -> None:
    report = run_blind_replay_matrix(
        [{"name": "g", "kind": "terminal_disposition", "width": 1, "request": "r"}],
        lambda golden: {"verdicts": {"m9": "accepted"}, "latency_seconds": 5.0},
        rotation_adjudicate=lambda golden: {
            "verdicts": {"m9": "rejected"},
            "latency_seconds": 5.0,
        },
        checker_model="gpt-5.6-sol",
        captured_at="2026-08-24T12:00:00Z",
    )
    row = report["goldens"][0]
    assert row["same_epoch"]["disagreement"] is False
    assert row["rotation"] == {
        "ran": True,
        "projection": {"m9": "rejected"},
        "disagreement": True,
    }


def test_width_invariance_compares_only_shared_items() -> None:
    def adjudicate(golden: Mapping[str, Any]) -> dict[str, Any]:
        if golden.get("width") == 1:
            # The width-1 replay carries only m1; its batch-mate m2 is not
            # part of the question. m1 flips -> disagreement.
            return {"verdicts": {"m1": "inaccurate"}, "latency_seconds": 7.0}
        return {"verdicts": {"m1": "accurate", "m2": "accurate"}, "latency_seconds": 7.0}

    report = run_blind_replay_matrix(
        [
            {
                "name": "g",
                "kind": "metadata_batch",
                "width": 5,
                "request": "r",
                "width_variants": {1: "r-w1"},
            }
        ],
        adjudicate,
        rotation_adjudicate=adjudicate,
        checker_model="gpt-5.6-sol",
        captured_at="2026-08-24T12:00:00Z",
    )
    assert report["goldens"][0]["width_invariance"]["disagreement"] is True


def test_missing_rotation_leg_is_recorded_and_refused_downstream(
    tmp_path: Path,
) -> None:
    report = run_blind_replay_matrix(
        [{"name": "g", "kind": "terminal_disposition", "width": 1, "request": "r"}],
        lambda golden: {"verdicts": {"m9": "accepted"}, "latency_seconds": 5.0},
        rotation_adjudicate=None,
        checker_model="gpt-5.6-sol",
        captured_at="2026-08-24T12:00:00Z",
    )
    assert report["goldens"][0]["rotation"] == {"ran": False}
    path = write_report(report, tmp_path / "report.json")
    block = derive_blind_replay_block(load_blind_replay_report(path), report_path=path)
    assert block["rotation_replay_ran"] is False


def test_fail_closed_inputs() -> None:
    with pytest.raises(BlindReplayMatrixError, match="zero goldens"):
        run_blind_replay_matrix(
            [],
            lambda golden: {"verdicts": {"m": "accepted"}, "latency_seconds": 1.0},
            checker_model="gpt-5.6-sol",
            captured_at="2026-08-24T12:00:00Z",
        )
    with pytest.raises(BlindReplayMatrixError, match="verdict projection"):
        run_blind_replay_matrix(
            [{"name": "g", "kind": "fidelity", "width": 1, "request": "r"}],
            lambda golden: {"verdicts": {}, "latency_seconds": 1.0},
            checker_model="gpt-5.6-sol",
            captured_at="2026-08-24T12:00:00Z",
        )


def test_corpus_loader_reads_the_committed_transcripts() -> None:
    corpus = Path(__file__).parent / "golden_checker_transcripts"
    entries = load_golden_corpus_entries(corpus)
    names = {entry["name"] for entry in entries}
    assert "fidelity_accept" in names and "terminal_accept" in names
    widths = {entry["name"]: entry["width"] for entry in entries}
    assert widths["metadata_accept"] >= 1
    kinds = {entry["kind"] for entry in entries}
    assert kinds <= {"metadata_batch", "fidelity", "terminal_disposition"}
