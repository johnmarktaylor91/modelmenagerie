"""W8/PD-2: the fidelity no-consumer arm disqualifies on EXECUTION EVIDENCE.

The JMT-SIGNED widening of registered four-part waiver instance #1
(``CanonicalReducer._fidelity_gate_has_no_consumer``; signature record:
sprint STATE.md "JMT APPROVALS RECORDED (2026-08-24)", §6 C.8 -- the signature
authorizes the arm; THIS change carries the proof). Pre-W8 the arm disqualified
on DISPATCH INTENT -- any referenced attempt with an environment binding or a
pre-declared ``identities.environment``/``identities.execution`` -- so a worker
that provably died before executing (the DX-3 bootstrap-death shape) could not
reduce its failure terminal: the recording path refused, the member parked, and
the shape recurs on every pre-execution death (R18; the five 200a fidelity
parks m2783, m2426, m3416, m1163, m1221).

THE FOUR-PART PROOF (doctrine: torchlens CLAUDE.md, LOCKED):

1. DISPOSITION-INVARIANT. The arm can only let an already-NEGATIVE failure
   terminal record; it upgrades no outcome, publishes no record, and unlocks
   no capability (release eligibility keeps requiring ``runs``; a waived
   record is never counted fidelity-audited). Per-park: each of m2783, m2426,
   m3416, m1163, m1221 is a terminal-refusal park whose durable row records
   the exact ``intended_status_code`` -- a ``failed:*`` terminal -- that the
   recording path refused; post-W8 each park's re-drive (through D-48's
   release lane, under the signed C.8 disposition) can land AT MOST that same
   already-failed terminal class with its fidelity DISCLOSED as
   no-consumer-waived: never a ``runs``, never a publication, never an audit
   claim. The exact per-park intended codes are replay-read from the
   production park rows at re-drive time (this worktree holds no campaign
   ledgers; the per-park verification run is the orchestrator-level
   acceptance, the same boundary as D-17's B2-X-V2).
2. CLOSED PROOF RULE. The enumerated conjunction -- no
   ``worker_receipt.present`` AND no ``worker_receipt.forward_started`` on any
   referenced or same-work ledger attempt, AND no ``execution.current`` AND no
   ``accepted_attempt_ids`` on the record -- with the near-miss matrix below:
   every single-flag flip refuses.
3. REPLAY-DERIVED, ZERO PRODUCER INPUT. Recomputed on every append/replay from
   ledger-bound attempt records at the exact stable/work identity; the
   same-work sweep reads the CURRENT attempt ledger, so omitting an attempt
   from ``status.attempt_ids`` cannot hide its receipt; no persisted waiver
   bit exists.
4. DISCLOSED, SEGREGATED. The ``fidelity_no_consumer_pre_execution`` census
   arm in :func:`menagerie.crawler.status.funnel_counts`, registered on the
   instance's ``WAIVER_DISCLOSURES`` row, never counted beside audited
   outcomes; release eligibility keeps excluding it.

Also here: the AG-10 detail (opus E7) -- an accuracy-gate terminal minted after
a completed forward routes to a REAL fidelity gate (typed
``fidelity-consumer-exists`` refusal), never the generic dead-end fall-through
-- and the CLOSED INVENTORY proving every ``PRE_FIDELITY_FAILURE_STAGES``
member is pre-execution (unconditionally, or guarded by the evidence sweep).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.reducer import (
    PRE_FIDELITY_FAILURE_STAGES,
    CanonicalReducer as ProductionCanonicalReducer,
    ReductionError,
)
from menagerie.crawler.status import WAIVER_DISCLOSURES, funnel_counts
from menagerie.crawler.tests.conftest import (
    HASH,
    make_attempt,
    make_failed_attempt,
    rebind_nonaward_parent_proof,
)
from menagerie.crawler.tests.test_fidelity_waiver_kernel import (
    CanonicalReducer,
    _append,
    _paths,
    _succeeded_forward_attempt,
    _waiver_terminal,
)

pytestmark = pytest.mark.smoke


def _dispatched_dead_attempt(
    stable_id: str = "m_example",
    *,
    attempt_id: str = "attempt-exec-1",
    stage: str = "runner",
    reason_code: str = "internal-error",
) -> dict:
    """Build the bootstrap-death shape: dispatched (intent), died pre-execution.

    The worker was really dispatched -- environment bound, environment and
    execution identities pre-declared (exactly what disqualified the arm
    pre-W8) -- and left NO execution evidence: no receipt, no started forward.
    """

    attempt = make_failed_attempt(
        stable_id, attempt_id=attempt_id, stage=stage, reason_code=reason_code
    )
    attempt["environment"] = make_attempt(stable_id)["environment"]
    attempt["identities"]["environment"] = HASH
    attempt["identities"]["execution"] = HASH
    attempt["attempted_rungs"] = ["R3_PORT"]
    attempt["identities"]["recipe"] = HASH
    attempt["supervisor_observation"].update(
        {
            "exit_code": 1,
            "stdout_sha256": hash_bytes(b""),
            "stderr_sha256": hash_bytes(b""),
        }
    )
    attempt.pop("ledger_seq", None)
    attempt.pop("payload_sha256", None)
    return rebind_nonaward_parent_proof(attempt)


def _bootstrap_terminal(*, attempt_ids: tuple[str, ...] = ("attempt-exec-1",)) -> dict:
    """Build the R3 failure terminal for a runner-stage bootstrap death."""

    return _waiver_terminal(
        stage="runner", reason_code="internal-error", attempt_ids=attempt_ids
    )


def test_bootstrap_refusal_shape_reduces_without_park(tmp_path: Path) -> None:
    """The W8 RED core: dispatch intent no longer disqualifies the arm.

    A worker that was dispatched (environment bound, identities pre-declared)
    and provably died before executing -- no receipt, no started forward, no
    accepted attempts -- reduces its failure terminal instead of refusing into
    a terminal-refusal park (the five-park R18 shape).
    """

    attempt = _dispatched_dead_attempt()
    model = _bootstrap_terminal()
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert _append(reducer, model).appended
        record = reducer.current_records["m_example"]
        # Part 1, disposition invariance: the recorded outcome is exactly the
        # already-negative failure terminal -- never upgraded, never
        # release-eligible, never counted audited.
        assert record["status"]["code"] == "failed:runner"
        assert record["completeness"]["release_eligible"] is False
        counts = funnel_counts(reducer.current_records)
        assert counts["fidelity_no_consumer_pre_execution"] == 1
        assert "fidelity_audited_failure" not in counts
        assert "fidelity_waived_machine_failure" not in counts


def _arm_duck(attempts: list[dict]) -> SimpleNamespace:
    """Duck-typed reducer exposing exactly what the pure arm consumes."""

    index = {str(attempt["attempt_id"]): attempt for attempt in attempts}
    return SimpleNamespace(
        _attempt_index=lambda: index,
        _current_attempt_records=lambda: tuple(attempts),
        _model_work_id=lambda model, inputs: "work-m_example",
        _artifact_authority_inputs=lambda model: None,
    )


def _arm_model(attempt_ids: tuple[str, ...], **execution: object) -> dict:
    """Minimal pre-execution failure terminal for the pure arm."""

    return {
        "stable_id": "m_example",
        "status": {"kind": "failed", "stage": "runner", "attempt_ids": list(attempt_ids)},
        "fidelity": {},
        "execution": {"current": False, "accepted_attempt_ids": [], **execution},
    }


def _arm(duck: SimpleNamespace, model: dict) -> bool:
    """Invoke the production arm over the duck."""

    return ProductionCanonicalReducer._fidelity_gate_has_no_consumer(duck, model)  # noqa: SLF001


def test_near_miss_matrix_every_single_flag_flip_refuses() -> None:
    """Part 2's near-miss matrix over the exact closed conjunction.

    The dispatched-dead base PASSES (environment bound, identities
    pre-declared, zero execution evidence); every one-flag flip REFUSES.
    """

    base = _dispatched_dead_attempt()
    assert _arm(_arm_duck([base]), _arm_model(("attempt-exec-1",))) is True

    receipted = _dispatched_dead_attempt()
    receipted["worker_receipt"]["present"] = True
    assert _arm(_arm_duck([receipted]), _arm_model(("attempt-exec-1",))) is False

    # The HOSTILE arm: a started forward refuses even with ``present`` forged
    # false -- an executed forward always has a fidelity consumer.
    started = _dispatched_dead_attempt()
    started["worker_receipt"]["forward_started"] = True
    assert started["worker_receipt"]["present"] is False
    assert _arm(_arm_duck([started]), _arm_model(("attempt-exec-1",))) is False

    assert (
        _arm(
            _arm_duck([base]),
            _arm_model(("attempt-exec-1",), accepted_attempt_ids=["attempt-exec-1"]),
        )
        is False
    )
    assert _arm(_arm_duck([base]), _arm_model(("attempt-exec-1",), current=True)) is False
    # A referenced attempt missing from the ledger fails closed.
    assert _arm(_arm_duck([base]), _arm_model(("attempt-never-persisted",))) is False


def test_hidden_same_work_receipt_cannot_be_omitted() -> None:
    """Part 3: the sweep reads the LEDGER, not the producer's attempt list.

    A same-work attempt carrying a receipt refuses the arm even when the
    terminal's ``status.attempt_ids`` omits it -- reference omission can never
    improve the verdict.
    """

    dead = _dispatched_dead_attempt()
    hidden = _dispatched_dead_attempt(attempt_id="attempt-exec-2")
    hidden["attempt_no"] = 2
    hidden["worker_receipt"]["present"] = True
    duck = _arm_duck([dead, hidden])
    # Referenced: only the dead attempt. The hidden receipt still refuses.
    assert _arm(duck, _arm_model(("attempt-exec-1",))) is False
    # A hidden FOREIGN-work receipt proves nothing about this work.
    foreign = _dispatched_dead_attempt(attempt_id="attempt-exec-3")
    foreign["work_id"] = "work-other"
    foreign["worker_receipt"]["present"] = True
    assert _arm(_arm_duck([dead, foreign]), _arm_model(("attempt-exec-1",))) is True


def test_accuracy_gate_after_forward_routes_to_a_real_fidelity_gate(tmp_path: Path) -> None:
    """AG-10 (opus E7): the shape names its route, never a generic dead end."""

    executed = _succeeded_forward_attempt(attempt_id="attempt-fwd-1", attempt_no=1)
    gate_death = _dispatched_dead_attempt(
        attempt_id="attempt-gate-1", stage="accuracy-gate", reason_code="checker-session-exhaustion"
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
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(executed)
        reducer.append_attempt(gate_death)
        with pytest.raises(
            ReductionError, match="fidelity-consumer-exists.*routes to a REAL fidelity gate"
        ):
            _append(reducer, model)


def test_pre_fidelity_stage_inventory_is_closed_and_proven() -> None:
    """The closed inventory: every admitted stage carries its pre-execution proof."""

    assert set(PRE_FIDELITY_FAILURE_STAGES) == {
        "intake",
        "source",
        "fetch",
        "author",
        "evidence",
        "accuracy-gate",
        "runner",
    }
    for stage, proof in PRE_FIDELITY_FAILURE_STAGES.items():
        assert isinstance(proof, str) and proof, stage
    # The two conditional members are exactly the execute-first-orderable pair,
    # and their proofs SAY so -- the execution-evidence sweep is load-bearing.
    for stage in ("accuracy-gate", "runner"):
        assert "CONDITIONAL" in PRE_FIDELITY_FAILURE_STAGES[stage]
    # Source binding: the arm consumes THIS inventory, not a private set, and
    # runs the same-work ledger sweep that guards the conditional members.
    source = inspect.getsource(
        ProductionCanonicalReducer._fidelity_gate_has_no_consumer  # noqa: SLF001
    )
    assert "PRE_FIDELITY_FAILURE_STAGES" in source
    assert "_current_attempt_records()" in source


def test_widening_census_key_is_registered_on_the_instance() -> None:
    """Part 4: the segregated arm rides the registered instance's disclosure row."""

    row = next(row for row in WAIVER_DISCLOSURES if row.waiver_id == "fidelity-gate-no-consumer")
    assert "fidelity_no_consumer_pre_execution" in row.census_keys
    assert "fidelity_audited_failure" in row.census_keys
