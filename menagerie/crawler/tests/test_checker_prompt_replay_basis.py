"""FT-A (A13/IC-D7): semantic replay validates at the RECORDED prompt basis.

``_checker_prompt_hash()`` reads the LIVE checker-prompt file bytes. The
write-time doors correctly bind the currently shipped prompt, and the
append_gate replay carve-out already honors sealed gates at their recorded
basis -- but the MODEL-validation pipeline consulted the live bytes during
semantic replay too: ``_derive_rung_gate_current`` compared the sealed gate's
recorded ``prompt_sha256`` against the live hash, and
``recompute_accepted_identities`` folded the live hash into the recomputed
vet identity. After any checker-prompt re-pin (23 rotations in 60 days;
m7305 paid six re-authors for prompt churn) EVERY held runs/accepted head
refused on replay -- the rung-11/rung-12 history-vs-live family at the model
layer, and exactly the crash mode the ArtifactTransactionProjection docstring
already documents for its own consumers.

The pin: a settled runs head stays current in a fresh dependency projection
after the live prompt bytes rotate; the identity checks stay armed at the
recorded basis (a record contradicting its OWN recorded identities still
refuses); live appends keep the live-bytes binding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import ReductionError, project_dependency_current
from menagerie.crawler.tests.conftest import (
    make_attempt,
    make_authority_context,
    make_gate,
    make_model,
)
from menagerie.crawler.tests.test_reducer import CanonicalReducer

pytestmark = pytest.mark.smoke

_ROTATED_PROMPT = "sha256:" + "d" * 64


def _paths(tmp_path: Path) -> LedgerPaths:
    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _award_head(paths: LedgerPaths) -> dict:
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_gate(make_gate(["m_example"]))
        reducer.append_attempt(make_attempt())
        return reducer.append_model(make_model(accepted=True)).record


def test_prompt_re_pin_does_not_stale_settled_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checker-prompt rotation must never refuse held records on replay:
    the sealed gate row is the adjudication authority at its recorded basis."""

    paths = _paths(tmp_path)
    head = _award_head(paths)
    assert head["status"]["kind"] == "runs"
    context = make_authority_context(["m_example"])
    baseline = project_dependency_current(paths, context=context)
    assert "m_example" in baseline.current_records

    monkeypatch.setattr(reducer_module, "_checker_prompt_hash", lambda: _ROTATED_PROMPT)
    after = project_dependency_current(paths, context=context)
    assert "m_example" in after.current_records, (
        "a live checker-prompt re-pin staled a settled runs head on replay: "
        f"{after.stale_reasons.get('m_example')}"
    )


def test_replay_still_refuses_a_record_contradicting_its_own_basis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The armed direction: record-basis replay is not a bypass. A persisted
    head whose recorded vet identity contradicts its own recorded facts and
    basis still refuses on replay after the prompt rotates."""

    paths = _paths(tmp_path)
    _award_head(paths)
    monkeypatch.setattr(reducer_module, "_checker_prompt_hash", lambda: _ROTATED_PROMPT)
    context = make_authority_context(["m_example"])
    projection = project_dependency_current(paths, context=context)
    record = dict(projection.current_records["m_example"])
    # Contradict the record's own recorded identity, then replay it directly:
    # the recorded-basis recompute must still catch it.
    record["evidence"] = {**record["evidence"], "evidence_identity": "sha256:" + "e" * 64}
    with pytest.raises(ReductionError):
        with CanonicalReducer(paths, ["m_example"]) as reducer:
            replayed = reducer_module._replay_reducer(  # noqa: SLF001
                paths,
                context=reducer.context,
                prior_model_records=(),
                attempt_records=tuple(reducer._attempts.iter_records()),  # noqa: SLF001
                gate_records=tuple(reducer._gates.iter_records()),  # noqa: SLF001
                artifact_records=tuple(reducer._artifacts.iter_events()),  # noqa: SLF001
                raw_prior_current={},
                dependency_current={},
                attempt_index=reducer._attempt_index(),  # noqa: SLF001
                gate_index=reducer._gate_index(),  # noqa: SLF001
            )
            replayed.append_model(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"record_seq", "record_revision"}
                }
            )


def test_live_appends_keep_the_live_prompt_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write-time door is untouched: sealing a NEW gate that binds a
    stale prompt still refuses against the live bytes."""

    paths = _paths(tmp_path)
    monkeypatch.setattr(reducer_module, "_checker_prompt_hash", lambda: _ROTATED_PROMPT)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        with pytest.raises(ReductionError, match="does not bind the current prompt"):
            reducer.append_gate(make_gate(["m_example"]))
