"""AB-30 matrix: fidelity no-consumer waiver arm widening (W8, signed PD-2).

Seam: the reducer's fidelity no-consumer waiver -- registered four-part waiver
instance #1, ``CanonicalReducer._fidelity_gate_has_no_consumer``
(``menagerie/crawler/reducer.py``) -- widened by lane w2-ledger under the
JMT-signed PD-2 conjunction. This seam row is ADDED to the declaration
registry at batch assembly (the W8 arm extension was not in the pre-batch
registry), so unlike the frozen-fixture matrices this module carries no
pre-batch fixture: the prior-era shape is rebuilt byte-honestly through the
same production entry points the lane used, and every cell is adjudicated
through :func:`menagerie.crawler.tests.mixed_era.classify_reader` so an
unclassified crash can never read as a lawful outcome.

THE DECLARED FLIP (cell 1). Pre-batch, the dispatched-dead bootstrap shape --
environment bound and ``identities.environment``/``identities.execution``
pre-declared (dispatch INTENT), but NO worker receipt, NO started forward, no
``accepted_attempt_ids``, no ``execution.current``, and no execution evidence
on any referenced or same-work ledger attempt -- REFUSED reduction with
"required fidelity is missing its gate": dispatch intent alone disqualified
the arm, so the already-failed terminal could not record and the member
parked (the five-park R18 shape). Post-batch the same shape reduces: the
already-negative ``failed:*`` terminal stands unchanged, disclosed segregated
via the ``fidelity_no_consumer_pre_execution`` census key in
:func:`menagerie.crawler.status.funnel_counts`.

The matrix here is the READER side of the four-part doctrine (torchlens
CLAUDE.md, LOCKED), reader = the REAL production reducer:

* cell 1 -- the flip lands and is DISCLOSED (part 4);
* cells 2/3 -- EXECUTION EVIDENCE in any form disqualifies: a present worker
  receipt, or a started forward even with ``present`` forged false, never
  waives (part 2's closed conjunction, adjudicated through full reduction);
* cell 4 -- disposition invariance: the waived outcome is exactly the
  already-recorded negative terminal, never an upgrade, never
  release-eligible, never counted audited (parts 1 and 4);
* cell 5 -- an accuracy-gate terminal WITH same-work execution evidence
  refuses TYPED naming ``fidelity-consumer-exists`` (the AG-10 route);
* cell 6 -- the assembler-declared post-batch surface exists on the reader:
  the ``PRE_FIDELITY_FAILURE_STAGES`` module constant and the shared
  ``_work_execution_evidence`` evidence rule.

Shape builders are imported from the owning lane's proof module
(``test_fidelity_no_consumer_widening``); importing it has no side effects
beyond its own imports of the same production modules used here.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from menagerie.crawler.reducer import (
    PRE_FIDELITY_FAILURE_STAGES,
    CanonicalReducer as ProductionCanonicalReducer,
)
from menagerie.crawler.status import (
    WAIVER_DISCLOSURES,
    funnel_counts,
    record_is_release_eligible,
)
from menagerie.crawler.tests.conftest import rebind_nonaward_parent_proof
from menagerie.crawler.tests.mixed_era import classify_reader
from menagerie.crawler.tests.test_fidelity_no_consumer_widening import (
    _bootstrap_terminal,
    _dispatched_dead_attempt,
)
from menagerie.crawler.tests.test_fidelity_waiver_kernel import (
    CanonicalReducer,
    _append,
    _paths,
    _succeeded_forward_attempt,
    _waiver_terminal,
)

pytestmark = pytest.mark.smoke

_CENSUS_KEY = "fidelity_no_consumer_pre_execution"


def _reduce(
    tmp_path: Path,
    attempts: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reduce one attempt stream plus terminal through the production reducer.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The reduced current record for ``m_example`` and the derived
        funnel-census counts. Any refusal propagates for
        :func:`classify_reader` to classify.
    """

    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        for attempt in attempts:
            reducer.append_attempt(dict(deepcopy(attempt)))
        _append(reducer, dict(deepcopy(model)))
        record = deepcopy(reducer.current_records["m_example"])
        counts = dict(funnel_counts(reducer.current_records))
    return record, counts


def _assert_refusal_left_no_waived_record(tmp_path: Path) -> None:
    """Replay the refused ledger: no record landed, no waiver was disclosed.

    Part 3 shadow: the census is replay-derived, so a refused append must
    leave NOTHING a rebuild could count as waived.
    """

    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        assert "m_example" not in reducer.current_records
        assert _CENSUS_KEY not in funnel_counts(reducer.current_records)


def test_prior_era_dispatched_dead_shape_flips_to_disclosed_waiver(tmp_path: Path) -> None:
    """Cell 1, the FLIP: the prior-era bootstrap-death shape now reduces.

    Pre-batch this exact shape -- dispatch intent (environment binding plus
    pre-declared environment/execution identities) with zero execution
    evidence -- refused with "required fidelity is missing its gate" and
    parked the member. Post-W8 it parses: the reduction lands the exact
    already-negative terminal and discloses it segregated under
    ``fidelity_no_consumer_pre_execution``.
    """

    outcome = classify_reader(
        lambda: _reduce(tmp_path, [_dispatched_dead_attempt()], _bootstrap_terminal())
    )
    assert outcome[0] == "parsed", (
        "the signed W8 flip did not land: the dispatched-dead shape still "
        f"refuses reduction ({outcome[1]!r})"
    )
    record, counts = outcome[1]
    assert record["status"]["code"] == "failed:runner"
    assert counts[_CENSUS_KEY] == 1


def test_worker_receipt_present_disqualifies_the_arm(tmp_path: Path) -> None:
    """Cell 2: the SAME shape with a worker receipt present never waives.

    A present receipt is execution evidence -- the executed work has a
    fidelity consumer -- so full reduction refuses TYPED (the real arm routes
    it to ``fidelity-consumer-exists``); it must never land a waived record.
    """

    receipted = _dispatched_dead_attempt()
    receipted["worker_receipt"]["present"] = True
    receipted["worker_receipt"]["input_note"] = "worker died after producing a receipt"
    rebind_nonaward_parent_proof(receipted)

    outcome = classify_reader(lambda: _reduce(tmp_path, [receipted], _bootstrap_terminal()))
    assert outcome[0] == "refused-typed", (
        f"a receipted attempt WAIVED the fidelity gate: {outcome[1]!r}"
    )
    assert "fidelity-consumer-exists" in str(outcome[1])
    _assert_refusal_left_no_waived_record(tmp_path)


def test_forward_started_with_forged_present_false_never_waives(tmp_path: Path) -> None:
    """Cell 3, the HOSTILE arm: a started forward refuses even forged.

    ``forward_started`` is execution evidence on its own; forging
    ``present=False`` alongside it must not launder the attempt back into the
    no-consumer shape. The real reader routes it through the same-work
    evidence rule (``fidelity-consumer-exists``); it never parses into a
    waiver.
    """

    forged = _dispatched_dead_attempt()
    forged["worker_receipt"]["forward_started"] = True
    assert forged["worker_receipt"]["present"] is False
    rebind_nonaward_parent_proof(forged)

    outcome = classify_reader(lambda: _reduce(tmp_path, [forged], _bootstrap_terminal()))
    assert outcome[0] == "refused-typed", (
        f"a forged forward-started attempt WAIVED the fidelity gate: {outcome[1]!r}"
    )
    assert "fidelity-consumer-exists" in str(outcome[1])
    _assert_refusal_left_no_waived_record(tmp_path)


def test_waived_disposition_is_invariant_segregated_and_never_audited(tmp_path: Path) -> None:
    """Cell 4: parts 1 and 4 over the waived outcome.

    The recorded terminal is byte-for-byte the already-negative disposition
    the terminal submitted -- same code, same failed kind, same reason --
    never an upgrade; it stays release-ineligible; and the census counts it
    ONLY under the segregated key, never beside audited outcomes.
    """

    model = _bootstrap_terminal()
    submitted_code = model["status"]["code"]
    submitted_reason = model["status"]["reason_code"]
    assert submitted_code.startswith("failed:")

    outcome = classify_reader(
        lambda: _reduce(tmp_path, [_dispatched_dead_attempt()], model)
    )
    assert outcome[0] == "parsed"
    record, counts = outcome[1]

    # Part 1: exactly the recorded negative disposition, never an upgrade.
    assert record["status"]["code"] == submitted_code
    assert record["status"]["kind"] == "failed"
    assert record["status"]["reason_code"] == submitted_reason
    # Never release-eligible: both the recorded flag and the derived authority.
    assert record["completeness"]["release_eligible"] is False
    assert record_is_release_eligible(record, {"m_example": record}) is False

    # Part 4: disclosed segregated, never counted audited.
    assert counts[_CENSUS_KEY] == 1
    assert "fidelity_audited_failure" not in counts
    assert "fidelity_waived_machine_failure" not in counts

    # The disclosure rides the REGISTERED instance-#1 row, not an ad-hoc key.
    row = next(row for row in WAIVER_DISCLOSURES if row.waiver_id == "fidelity-gate-no-consumer")
    assert _CENSUS_KEY in row.census_keys


def test_accuracy_gate_with_execution_evidence_refuses_naming_the_route(
    tmp_path: Path,
) -> None:
    """Cell 5 (AG-10): same-work execution evidence names a REAL fidelity gate.

    An accuracy-gate terminal minted after a completed forward has a fidelity
    consumer; the reader must refuse TYPED with ``fidelity-consumer-exists``,
    never waive and never fall through to the generic dead end.
    """

    executed = _succeeded_forward_attempt(attempt_id="attempt-fwd-1", attempt_no=1)
    gate_death = _dispatched_dead_attempt(
        attempt_id="attempt-gate-1",
        stage="accuracy-gate",
        reason_code="checker-session-exhaustion",
    )
    gate_death["attempt_no"] = 2
    model = _waiver_terminal(
        stage="accuracy-gate",
        reason_code="checker-session-exhaustion",
        attempt_ids=("attempt-gate-1",),
    )
    model["status"]["human_review"] = {
        "required": True,
        "reviewed_at": None,
        "reviewed_by": None,
        "notes": None,
    }

    outcome = classify_reader(lambda: _reduce(tmp_path, [executed, gate_death], model))
    assert outcome[0] == "refused-typed", (
        f"an accuracy-gate terminal with same-work execution evidence WAIVED: {outcome[1]!r}"
    )
    assert "fidelity-consumer-exists" in str(outcome[1])
    _assert_refusal_left_no_waived_record(tmp_path)


def test_declared_post_batch_surface_exists_on_the_reader() -> None:
    """Cell 6: the assembler-declared W8 surface is present on the reader.

    The registry row being added at assembly names the new module constant
    ``PRE_FIDELITY_FAILURE_STAGES`` and the shared evidence rule
    ``_work_execution_evidence``; both must exist on the production reducer,
    with the two execute-first-orderable stages marked CONDITIONAL (their
    membership is sound only under the execution-evidence sweep).
    """

    assert isinstance(PRE_FIDELITY_FAILURE_STAGES, Mapping)
    assert "runner" in PRE_FIDELITY_FAILURE_STAGES
    assert "accuracy-gate" in PRE_FIDELITY_FAILURE_STAGES
    for stage in ("runner", "accuracy-gate"):
        assert "CONDITIONAL" in PRE_FIDELITY_FAILURE_STAGES[stage]
    assert callable(getattr(ProductionCanonicalReducer, "_work_execution_evidence"))
    assert callable(getattr(ProductionCanonicalReducer, "_fidelity_gate_has_no_consumer"))
