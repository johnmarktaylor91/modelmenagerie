"""AB-30 matrix: terminal-status vocabulary seam (W13/W1/AL-6/W14/W5).

Readers x eras over frozen pre-batch artifacts captured at the Wave-2 freeze
tree. The Wave-2 batch extends the closed terminal vocabulary
(``blocked:disposition-review``, deferral valve rows, deferred-family parks)
and re-scopes ``failed:accuracy-gate`` to RENDERED adverse verdicts. Prior-era
terminal heads must keep replaying era-scoped (PD-13 forward-only), future
vocabulary members must refuse typed until the sealed batch extends the enum,
and the pre-batch permissive behaviors W13 retires are pinned as goldens so
the flip is measured, never ambient.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from menagerie.crawler.authority import derive_terminal_proof
from menagerie.crawler.reducer import CanonicalReducer, LedgerPaths
from menagerie.crawler.tests.conftest import make_authority_context
from menagerie.crawler.tests.mixed_era import (
    assert_era_outcome,
    classify_reader,
    read_fixture_bytes,
    read_fixture_json,
)

pytestmark = pytest.mark.smoke

_PROOF_RULE = "sha256:" + "a" * 64


def _replay_heads(
    tmp_path: Path,
    models_lines: Sequence[Mapping[str, Any]],
    attempts_bytes: bytes,
) -> dict[str, str]:
    """Replay raw ledger bytes through the production reducer."""

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    with paths.models.open("w") as handle:
        for row in models_lines:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    paths.attempts.write_bytes(attempts_bytes)
    paths.gates.touch()
    stable_ids = sorted({str(row["stable_id"]) for row in models_lines})
    with CanonicalReducer(paths, make_authority_context(stable_ids)) as reducer:
        return {
            stable_id: str(record["status"]["code"])
            for stable_id, record in reducer.raw_head_view.items()
        }


def _frozen_models() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in read_fixture_bytes("terminal_models.jsonl").decode().splitlines()
    ]


def test_prior_era_terminal_heads_replay_era_scoped(tmp_path: Path) -> None:
    """Frozen pre-batch terminal heads replay to their exact recorded codes."""

    outcome = classify_reader(
        lambda: _replay_heads(
            tmp_path, _frozen_models(), read_fixture_bytes("terminal_attempts.jsonl")
        )
    )
    assert_era_outcome(
        reader_id="canonical-reducer-replay",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    if outcome[0] == "parsed":
        assert outcome[1] == {
            "m_forward": "failed:forward",
            "m_author": "failed:author",
        }


@pytest.mark.parametrize(
    ("mutation_field", "mutation_value"),
    [
        ("code", "failed:not-a-stage"),
        ("reason_code", "reason-from-the-future"),
    ],
)
def test_unknown_vocabulary_members_refuse_typed(
    tmp_path: Path, mutation_field: str, mutation_value: str
) -> None:
    """An unknown terminal code or reason NEVER parses silently, in any era."""

    models = _frozen_models()
    models[0]["status"][mutation_field] = mutation_value
    outcome = classify_reader(
        lambda: _replay_heads(
            tmp_path, models, read_fixture_bytes("terminal_attempts.jsonl")
        )
    )
    assert outcome[0] == "refused-typed", (
        f"unknown vocabulary member {mutation_value!r} parsed silently: a reader "
        "accepted a row from outside the closed set"
    )


def test_w13_future_member_refuses_until_the_sealed_batch(tmp_path: Path) -> None:
    """``blocked:disposition-review`` refuses on the closed enum pre-batch.

    W13 extends the closed set inside the sealed rotation batch; at batch
    assembly this cell may flip to parsed, but it must land classified.
    """

    models = _frozen_models()
    models[0]["status"]["code"] = "blocked:disposition-review"
    outcome = classify_reader(
        lambda: _replay_heads(
            tmp_path, models, read_fixture_bytes("terminal_attempts.jsonl")
        )
    )
    assert_era_outcome(
        reader_id="model-v3-schema-closed-enum",
        outcome=outcome,
        prebatch_expected="refused-typed",
        expected_flip="W13",
    )


def _proof(gate: Mapping[str, Any]) -> tuple[str, Any, Any]:
    proof = derive_terminal_proof(
        "m_example",
        "work-1",
        "failed:accuracy-gate",
        attempts=[],
        gates=[dict(gate)],
        proof_rule_identity=_PROOF_RULE,
    )
    return (str(proof.status_code), proof.reason_code, proof.gate_id)


def test_rendered_adverse_verdict_keeps_minting_accuracy_gate() -> None:
    """A RENDERED rejection stays failed:accuracy-gate in every era (W13 keep-arm)."""

    outcome = classify_reader(lambda: _proof(read_fixture_json("terminal_gate_rendered.json")))
    assert_era_outcome(
        reader_id="derive-terminal-proof-rendered-verdict",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    assert outcome[1] == (
        "failed:accuracy-gate",
        "terminal-disposition-rejected",
        "gate-terminal-1",
    )


def test_unrendered_verdict_pin_the_prebatch_permissive_golden() -> None:
    """PRE-BATCH DEFECT PIN: a verdict=None disposition still mints a terminal.

    This is the 27-row 200a census shape. Today the proof derivation mints
    ``failed:accuracy-gate`` with ``terminal-disposition-unverifiable``; W13
    declares the reduction of such heads refuses or reclassifies to
    ``blocked:disposition-review``. Pinning today's exact output makes the
    flip a measured fact at batch assembly instead of an ambient drift.
    """

    outcome = classify_reader(
        lambda: _proof(read_fixture_json("terminal_gate_unrendered.json"))
    )
    assert_era_outcome(
        reader_id="derive-terminal-proof-unrendered-verdict",
        outcome=outcome,
        prebatch_expected="parsed",
        expected_flip="W13",
    )
    if outcome[0] == "parsed":
        assert outcome[1] == (
            "failed:accuracy-gate",
            "terminal-disposition-unverifiable",
            "gate-terminal-1",
        )


def test_missing_disposition_block_refuses_typed() -> None:
    """A terminal_disposition gate lacking its block refuses typed, every era."""

    outcome = classify_reader(
        lambda: _proof(read_fixture_json("terminal_gate_missing_block.json"))
    )
    assert outcome[0] == "refused-typed"


def test_append_sink_refuses_verdictless_accuracy_gate_terminals(tmp_path: Path) -> None:
    """The mint sink refuses failed:accuracy-gate without rejected gate evidence.

    The frozen replay arms above cover READ-side era honesty; this arm pins
    the WRITE side: no new verdictless accuracy-gate terminal can be minted,
    so the prior-era permissive population (the 27 rows) can never grow.
    """

    from menagerie.crawler.tests.conftest import (
        bind_terminal_attempts,
        make_failed_attempt,
        make_model,
    )
    from menagerie.crawler.tests.test_reducer import CanonicalReducer as FixtureReducer

    paths = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )

    def _mint() -> None:
        with FixtureReducer(paths, ["m_example"]) as reducer:
            attempt = make_failed_attempt(
                "m_example", stage="accuracy-gate", reason_code="terminal-disposition-rejected"
            )
            reducer.append_attempt(attempt)
            model = make_model("m_example", status_code="failed:accuracy-gate")
            model["status"]["reason_code"] = "terminal-disposition-rejected"
            bind_terminal_attempts(model, [attempt])
            reducer.append_model(model)

    outcome = classify_reader(_mint)
    assert outcome[0] == "refused-typed"
    assert "gate evidence" in str(outcome[1])
