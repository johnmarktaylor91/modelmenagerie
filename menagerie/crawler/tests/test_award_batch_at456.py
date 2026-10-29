"""AT-4/AT-5/AT-6 award-closure batch: pre-flight, coherent proofs, one classifier.

The m12646 shape: the award tail published (spending the artifact chain,
events 2-4) and THEN the canonical append refused ("terminal proof spans
multiple environment generations"), leaving a spent chain under a failed
terminal that every later leg re-bought. This module pins the batch's
bankable, design-fork-neutral halves:

- AT-4 pre-flight (B1-XQ28): the reducer's ENTIRE append-validation chain is
  runnable as a dry run (``append_model(..., preflight=True)``) -- every
  refusal class raises pre-spend and NOTHING persists.
- AT-5: per-mode decisive selection is generation-coherent -- the newest
  single-generation satisfying set wins, replay-identical on every recorded
  model, and a split with no coherent set raises
  :class:`GenerationExhaustedError` BEFORE any publication.
- AT-6 TW-EXHAUSTIVE-AWARD: ONE closed exception -> disposition map for both
  publication lanes; an unmapped class fails here, and the disclosure
  builder stamps the AT-Q9 ``publication_lane`` field with honest DEGRADED
  status (DX-36).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.artifact_transactions import (
    ArtifactChainSpentError,
    ArtifactCheckpointError,
    ArtifactPublicationError,
    ArtifactTransitionError,
)
from menagerie.crawler.authority import (
    GenerationExhaustedError,
    derive_per_mode_attempt_ids,
    per_mode_generation_span,
)
from menagerie.crawler.award_seam import (
    AWARD_SEAM_DISPOSITIONS,
    AWARD_SEAM_LANES,
    AWARD_SEAM_UNCLASSIFIED,
    AwardSeamClassificationError,
    award_seam_refusal_event,
    classify_award_seam_exception,
    disclosed_award_seam_classification,
)
from menagerie.crawler.driver_contracts import PublicationAuthorizationRefused
from menagerie.crawler.mirrors import MirrorHashMismatchError
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.tests.conftest import make_authority_context

pytestmark = pytest.mark.smoke

_SID = "m_at5"
_WID = "work-m_at5"


def _attempt(
    attempt_id: str,
    mode: str,
    *,
    seq: int,
    generation: str | None,
    result: str = "succeeded",
) -> dict:
    attempt = {
        "attempt_id": attempt_id,
        "stable_id": _SID,
        "work_id": _WID,
        "mode": mode,
        "stage": "forward",
        "result": result,
        "attempt_no": seq,
        "ledger_seq": seq,
    }
    if generation is not None:
        attempt["identities"] = {"environment": generation}
    return attempt


# -- AT-5: generation-coherent decisive selection --------------------------------


def test_single_generation_selection_is_replay_identical_to_the_historical_rule():
    """No split: the independent newest-per-mode picks are returned unchanged
    -- every recorded model derives byte-identically (no recorded model can
    span, because the append-time span refusal never admitted one)."""

    attempts = [
        _attempt("a-train-1", "train", seq=1, generation="G1"),
        _attempt("a-train-2", "train", seq=3, generation="G1"),
        _attempt("a-eval-1", "eval", seq=2, generation="G1"),
    ]
    assert derive_per_mode_attempt_ids(attempts, stable_id=_SID, work_id=_WID) == (
        ("train", "a-train-2"),
        ("eval", "a-eval-1"),
    )


def test_split_selection_chooses_the_newest_coherent_generation():
    """m12646's arming shape: eval re-ran under a rebuilt generation with no
    matching train. The historical rule minted a spanning proof (train G1 +
    eval G2) that refused at append AFTER the chain was spent; the coherent
    rule selects G1's complete set and the span stays disclosed."""

    attempts = [
        _attempt("a-train-1", "train", seq=1, generation="G1"),
        _attempt("a-eval-1", "eval", seq=2, generation="G1"),
        _attempt("a-eval-2", "eval", seq=5, generation="G2"),
    ]
    assert derive_per_mode_attempt_ids(attempts, stable_id=_SID, work_id=_WID) == (
        ("train", "a-train-1"),
        ("eval", "a-eval-1"),
    )
    assert per_mode_generation_span(attempts, stable_id=_SID, work_id=_WID) == ("G1", "G2")


def test_newer_complete_generation_wins_over_an_older_complete_one():
    attempts = [
        _attempt("a-train-1", "train", seq=1, generation="G1"),
        _attempt("a-eval-1", "eval", seq=2, generation="G1"),
        _attempt("a-train-2", "train", seq=4, generation="G2"),
        _attempt("a-eval-2", "eval", seq=5, generation="G2"),
        # A stale straggler in G1 newer than nothing in G2: G2 still wins on
        # its newest member.
        _attempt("a-eval-3", "eval", seq=3, generation="G1"),
    ]
    assert derive_per_mode_attempt_ids(attempts, stable_id=_SID, work_id=_WID) == (
        ("train", "a-train-2"),
        ("eval", "a-eval-2"),
    )


def test_unidentified_attempts_never_constrain_coherence():
    """An attempt with no attested generation joins any generation -- exactly
    the validator's span predicate (it counts only identified generations)."""

    attempts = [
        _attempt("a-train-1", "train", seq=1, generation=None),
        _attempt("a-eval-1", "eval", seq=2, generation="G2"),
    ]
    assert derive_per_mode_attempt_ids(attempts, stable_id=_SID, work_id=_WID) == (
        ("train", "a-train-1"),
        ("eval", "a-eval-1"),
    )


def test_no_coherent_generation_raises_generation_exhausted_pre_spend():
    attempts = [
        _attempt("a-train-1", "train", seq=1, generation="G1"),
        _attempt("a-eval-2", "eval", seq=5, generation="G2"),
    ]
    with pytest.raises(GenerationExhaustedError, match="generation exhausted"):
        derive_per_mode_attempt_ids(attempts, stable_id=_SID, work_id=_WID)
    # The exhausted class is an AuthorityDerivationError: prepare_model and
    # assembly surface it typed, never as a blanket protocol violation.
    from menagerie.crawler.authority import AuthorityDerivationError

    assert issubclass(GenerationExhaustedError, AuthorityDerivationError)


# -- AT-6: TW-EXHAUSTIVE-AWARD ---------------------------------------------------


def test_award_seam_classifier_is_total_over_the_closed_inventory():
    """Every exception class the publication seams raise maps to exactly one
    closed disposition; subclass precedence keeps the spent split typed."""

    spent = ArtifactChainSpentError(
        transaction_id="tx-m12646",
        stable_id=_SID,
        work_id=_WID,
        final_event_kind="published",
        attempted_event_kind="publication-authorized",
    )
    expected = {
        spent: "spent-chain-cache-miss",
        ArtifactTransitionError("lifecycle"): "spent-chain-cache-miss",
        ArtifactCheckpointError("store"): "store-integrity-campaign-stop",
        ArtifactPublicationError("contradictory materialization"): (
            "store-integrity-campaign-stop"
        ),
        PublicationAuthorizationRefused(ValueError("stale vector")): (
            "pre-commit-refusal-park"
        ),
        MirrorHashMismatchError("mirror bytes"): "store-integrity-campaign-stop",
        ReductionError("terminal proof spans multiple environment generations"): (
            "append-refusal-award-blocked"
        ),
        OSError("disk full mid-publish"): "post-commit-resume",
    }
    for exc, disposition in expected.items():
        assert classify_award_seam_exception(exc) == disposition
        assert disposition in AWARD_SEAM_DISPOSITIONS


def test_unmapped_exception_class_fails_closed():
    with pytest.raises(AwardSeamClassificationError):
        classify_award_seam_exception(ValueError("novel class"))
    # Disclosure never guesses: it stamps the loud unclassified value.
    assert (
        disclosed_award_seam_classification(ValueError("novel class"))
        == AWARD_SEAM_UNCLASSIFIED
    )


def test_refusal_event_stamps_lane_and_honest_status():
    event = award_seam_refusal_event(
        publication_lane="award-worker",
        stable_id=_SID,
        work_id=_WID,
        failure=ArtifactChainSpentError(
            transaction_id="tx-m12646",
            stable_id=_SID,
            work_id=_WID,
            final_event_kind="published",
            attempted_event_kind="publication-authorized",
        ),
        run_id="run-test",
        machine_id="machine-test",
        created_at="2026-08-24T00:00:00.000000Z",
    )
    assert event["status"] == "degraded"
    details = event["details"]
    assert details["publication_lane"] in AWARD_SEAM_LANES
    assert details["award_seam_classification"] == "spent-chain-cache-miss"
    assert details["disposition"] == "award-seam-refusal"
    with pytest.raises(ValueError, match="unknown publication lane"):
        award_seam_refusal_event(
            publication_lane="not-a-lane",
            stable_id=_SID,
            work_id=_WID,
            failure=OSError("x"),
            run_id="run-test",
            machine_id="machine-test",
            created_at="2026-08-24T00:00:00.000000Z",
        )


# -- AT-4: refusal AFTER publication parks typed, never a failed terminal --------


def test_append_refusal_after_publication_parks_spent_chain_typed(tmp_path, monkeypatch):
    """m12646's residual shape, contained: a deterministic reducer REFUSAL that
    only manifests on the real append (the pre-flight passed, the chain is
    spent) parks the member typed ``award-blocked:spent-chain`` with a
    ``chain_spent`` seam disclosure -- never a failed terminal under a
    published final event (the ``published_with_failure_head`` incoherence)."""

    from menagerie.crawler.recordio import scan_jsonl
    from menagerie.crawler.scheduler_parks import SCHEDULER_QUARANTINE_RELATIVE_PATH
    from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

    snapshot = _snapshot(tmp_path, count=1)
    paths = _paths(tmp_path, snapshot)
    stable_id = snapshot.items[0].stable_id

    def refused_append(self, model):
        # The pre-flight path (preflight_model) is untouched; only the REAL
        # append refuses -- a refusal that manifests solely against
        # final-event authority.
        raise ReductionError("final-event-authority refusal (fixture)")

    monkeypatch.setattr(CanonicalReducer, "append_model", refused_append)

    driver = _driver(tmp_path, snapshot)
    driver.run()

    # The chain WAS spent: exactly one final artifact event is durable.
    finals = [
        event
        for event in scan_jsonl(paths.ledgers.artifacts)
        if event.get("event_kind") in ("published", "private-committed")
    ]
    assert len(finals) == 1
    # ...and NO model row rides under it: no award, and no failed terminal
    # pairing a published final event with a failure head.
    assert scan_jsonl(paths.ledgers.models) == []

    seam_events = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if isinstance(event.get("details"), dict)
        and event["details"].get("disposition") == "award-seam-refusal"
        and event["details"].get("seam") == "append"
    ]
    assert seam_events, "the spent-chain refusal must disclose at the award seam"
    details = seam_events[-1]["details"]
    assert details["publication_lane"] == "award-worker"
    assert details["chain_spent"] is True
    assert details["award_seam_classification"] == "append-refusal-award-blocked"
    assert seam_events[-1]["status"] == "degraded"

    records_root = paths.ledgers.models.parent.parent
    quarantines = [
        row
        for row in scan_jsonl(records_root / SCHEDULER_QUARANTINE_RELATIVE_PATH)
        if row.get("stable_id") == stable_id
    ]
    assert quarantines
    assert (
        quarantines[-1]["scheduler_quarantine"]["reason_code"]
        == "award-blocked:spent-chain"
    )


# -- AT-4: the append pre-flight persists nothing --------------------------------


def test_append_preflight_raises_the_real_refusal_and_persists_nothing(tmp_path: Path):
    """The dry run IS the real validation chain: the same typed refusal, zero
    ledger movement, zero reducer-state movement."""

    ledgers = LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )
    with CanonicalReducer(ledgers, make_authority_context([_SID])) as reducer:
        from menagerie.crawler.constants import MODEL_SCHEMA_VERSION_V3

        candidate = {
            "schema_version": MODEL_SCHEMA_VERSION_V3,
            "stable_id": _SID,
            "parent_revision": "sha256:" + "e" * 64,
            "status": {"kind": "runs", "code": "runs"},
        }
        with pytest.raises(ReductionError, match="bad parentage"):
            reducer.preflight_model(candidate)
        assert reducer.current_view.get(_SID) is None
    assert not (tmp_path / "models.jsonl").exists()