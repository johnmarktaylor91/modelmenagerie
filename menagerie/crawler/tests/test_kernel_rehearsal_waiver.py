"""C1 rehearsal: adversarial forgeries against the K2 execution-failure waiver.

``test_fidelity_waiver_kernel`` pins the KERNEL_DECISION's mandated matrix. This
module is the rehearsal layer BEYOND it: forge every field a producer can reach
and require the waiver to stay refused, so the only way to a waived terminal
remains a genuine replay-proven machine death.

Each test names the forgery it attempts. A waiver granted here would publish an
R3/R4 terminal as fidelity-exempt on evidence a rebuild cannot confirm -- the
tripwire-disarming failure the LOCKED validation-integrity principle forbids.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.identity import stable_hash
from menagerie.crawler.recordio import LedgerConflictError
from menagerie.crawler.reducer import ReductionError
from menagerie.crawler.schema import PayloadValidationError
from menagerie.crawler.tests.conftest import HASH, rebind_nonaward_parent_proof
from menagerie.crawler.tests.test_fidelity_waiver_kernel import (
    CanonicalReducer,
    _append,
    _execution_failed_attempt,
    _paths,
    _succeeded_forward_attempt,
    _waiver_terminal,
)

pytestmark = pytest.mark.smoke


def _waived(reducer: CanonicalReducer, terminal: dict[str, Any]) -> bool:
    """Return whether one terminal is admitted AND disclosed as fidelity-waived."""

    _append(reducer, terminal)
    return terminal["stable_id"] in reducer.fidelity_waived_terminals()


def _refuses(reducer: CanonicalReducer, terminal: dict[str, Any]) -> str:
    """Require the terminal to be refused and return the refusal message."""

    with pytest.raises(ReductionError) as refused:
        _append(reducer, terminal)
    return str(refused.value)


# --- forging the bound evidence set -------------------------------------------


def test_a_terminal_binding_no_attempt_ids_cannot_launder_a_waiver(tmp_path: Path) -> None:
    """Forgery: drop ``status.attempt_ids`` so nothing can be disqualified.

    The referenced-attempt loop is a fail-closed DISQUALIFIER, so an empty list
    disqualifies nothing. Admission must still refuse the terminal outright --
    a failure terminal that binds no evidence has no decisive proof.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(_execution_failed_attempt())
        terminal = _waiver_terminal(attempt_ids=())
        message = _refuses(reducer, terminal)
        assert message, "an evidence-free failure terminal must refuse with a legible reason"


def test_a_duplicate_attempt_id_cannot_shadow_the_decisive_attempt(tmp_path: Path) -> None:
    """Forgery: re-append the decisive attempt ID carrying weaker evidence.

    Shadowing the decisive row would let a forger rewrite the death stage after
    the fact. The attempt ledger must refuse the duplicate identity.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(_execution_failed_attempt())
        shadow = _execution_failed_attempt(stage="environment", reason_code="build-failed")
        shadow["attempt_id"] = "attempt-exec-1"
        with pytest.raises((ReductionError, LedgerConflictError)) as refused:
            reducer.append_attempt(rebind_nonaward_parent_proof(shadow))
        assert "attempt-exec-1" in str(refused.value), (
            "a duplicate attempt identity must refuse naming the identity it collides with"
        )


def test_a_later_shallower_failure_cannot_relabel_a_deep_death(tmp_path: Path) -> None:
    """Forgery: die at ``forward``, then append an ``environment`` failure and claim it.

    The decisive attempt is the MAXIMAL-order execution attempt, so the appended
    shallow failure becomes decisive -- but the deep sibling still carries
    receipt evidence beyond ``environment``, so the sibling scan must refuse.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        deep = _execution_failed_attempt(
            attempt_id="attempt-exec-1", attempt_no=1, stage="forward", reason_code="exception"
        )
        reducer.append_attempt(deep)
        shallow = _execution_failed_attempt(
            attempt_id="attempt-exec-2",
            attempt_no=2,
            stage="environment",
            reason_code="build-failed",
            env_generation=None,
        )
        reducer.append_attempt(shallow)
        terminal = _waiver_terminal(
            stage="environment",
            reason_code="build-failed",
            attempt_ids=("attempt-exec-1", "attempt-exec-2"),
            env_generation=None,
        )
        with pytest.raises(ReductionError):
            _append(reducer, terminal)


def test_an_attempt_no_tie_stays_deterministic_and_refuses(tmp_path: Path) -> None:
    """Forgery: collide ``attempt_no`` with the honest death to contest decisiveness.

    ``_attempt_order`` breaks ties on ledger sequence then attempt ID, so the
    LATER row wins deterministically; the forged shallow claim must then refuse
    on the honest deep sibling's evidence rather than winning silently.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        honest = _execution_failed_attempt(
            attempt_id="attempt-exec-1",
            attempt_no=1,
            stage="input",
            reason_code="generation-exception",
        )
        reducer.append_attempt(honest)
        tie = _execution_failed_attempt(
            attempt_id="attempt-exec-0",
            attempt_no=1,
            stage="import",
            reason_code="import-exception",
        )
        reducer.append_attempt(tie)
        terminal = _waiver_terminal(
            stage="import",
            reason_code="import-exception",
            attempt_ids=("attempt-exec-0", "attempt-exec-1"),
        )
        with pytest.raises(ReductionError):
            _append(reducer, terminal)


# --- forging the stage/reason vocabulary --------------------------------------


def test_a_reason_code_closed_for_another_stage_refuses(tmp_path: Path) -> None:
    """Forgery: pair a real reason code with the wrong execution stage."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        attempt = _execution_failed_attempt(stage="import", reason_code="build-failed")
        with pytest.raises(ReductionError):
            reducer.append_attempt(attempt)


@pytest.mark.parametrize(
    "reason_code",
    # Every forward-stage reason code that can only arise AFTER the forward
    # returned, i.e. after execution produced the outcome fidelity adjudicates.
    ("invalid-output-signature", "confirmation-mismatch", "incomplete-receipt"),
)
def test_a_completed_forward_is_evidence_the_fidelity_gate_has_a_consumer(
    tmp_path: Path, reason_code: str
) -> None:
    """Forgery: claim a ``forward`` death on an attempt whose forward COMPLETED.

    A completed forward produced an execution outcome, which is exactly what the
    fidelity gate adjudicates, so the no-consumer contract cannot hold. The
    driver copies the worker's ``forward_completed`` onto failed attempts
    verbatim, so this is a live shape, not a synthetic one: before the C1
    rehearsal fix the ``forward`` allowance admitted every milestone and these
    terminals were WAIVED despite having run.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        attempt = _execution_failed_attempt(stage="forward", reason_code=reason_code)
        attempt["worker_receipt"]["forward_completed"] = True
        reducer.append_attempt(rebind_nonaward_parent_proof(attempt))
        terminal = _waiver_terminal(stage="forward", reason_code=reason_code)
        message = _refuses(reducer, terminal)
        assert "fidelity" in message, (
            "the refusal must name the missing fidelity authority an operator must obtain"
        )


def test_a_forward_death_before_completion_stays_waiver_eligible(tmp_path: Path) -> None:
    """The tightening is exactly one milestone wide: a real forward death still waives.

    Guards the fix against over-reach -- ``forward_started`` without
    ``forward_completed`` is the genuine machine death the waiver exists for.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        attempt = _execution_failed_attempt(stage="forward", reason_code="exception")
        assert attempt["worker_receipt"]["forward_started"] is True
        assert attempt["worker_receipt"]["forward_completed"] is False
        reducer.append_attempt(attempt)
        terminal = _waiver_terminal(stage="forward", reason_code="exception")
        assert _waived(reducer, terminal), (
            "an incomplete forward is a machine death and must stay waiver-eligible"
        )


# --- forging the environment-generation binding -------------------------------


def test_a_non_string_environment_identity_cannot_launder_not_applicable(
    tmp_path: Path,
) -> None:
    """Forgery: make ``identities.environment`` non-string to dodge the str filter.

    The generation set comprehends only string identities, so a non-string would
    read as "no observed environment" and collapse to the NOT_APPLICABLE
    sentinel while the attempt really ran inside a sealed environment.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        attempt = _execution_failed_attempt()
        attempt["identities"]["environment"] = 1234
        attempt = rebind_nonaward_parent_proof(attempt)
        try:
            reducer.append_attempt(attempt)
        except (ReductionError, TypeError, ValueError):
            return  # Schema/admission closes the laundering at the boundary.
        terminal = _waiver_terminal(
            env_generation=None,
        )
        terminal["execution"]["env_generation"] = stable_hash({"environment": "not-applicable"})
        with pytest.raises(ReductionError):
            _append(reducer, terminal)


def test_an_environment_death_binds_to_its_proofs_same_generation_success(
    tmp_path: Path,
) -> None:
    """Characterization: condition 7 binds the PROOF's generation, not the death's.

    An ``environment``-stage death never seals a generation of its own, so the
    binding comes from the proof's evidence set -- the decisive attempt UNIONED
    with same-generation successes. A model that ran in generation ``G`` and then
    died rebuilding therefore records ``G``, and the docstring's
    "no observed environment -> NOT_APPLICABLE sentinel" clause applies only when
    the whole proof observed none.

    Pinned deliberately: this is load-bearing for the majority fork that keeps
    partial-success-then-required-failure waiver-eligible, so any future change
    to the union must be a deliberated one rather than a silent edit.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        # Ledger order agrees with attempt_no order: the success lands first,
        # the decisive death last. (The dual-order waiver guard refuses any
        # interleaving where the two independent orders disagree, so an
        # out-of-order append here would be the AMBIGUITY case, not this
        # binding characterization.)
        success = _succeeded_forward_attempt(attempt_id="attempt-ok-1", attempt_no=1)
        assert success["identities"]["environment"] == HASH
        reducer.append_attempt(success)
        death = _execution_failed_attempt(
            attempt_id="attempt-exec-9",
            attempt_no=9,
            stage="environment",
            reason_code="build-failed",
            env_generation=None,
        )
        reducer.append_attempt(death)
        terminal = _waiver_terminal(
            stage="environment",
            reason_code="build-failed",
            attempt_ids=("attempt-exec-9",),
            env_generation=HASH,
        )
        assert _waived(reducer, terminal)

        # The negative half that makes the binding load-bearing: a generation the
        # proof never observed still refuses.
        _refuses(
            reducer,
            _waiver_terminal(
                stable_id="m_example",
                stage="environment",
                reason_code="build-failed",
                attempt_ids=("attempt-exec-9",),
                env_generation="sha256:" + "9" * 64,
            ),
        )


# --- forging the disclosure surface -------------------------------------------


def test_a_persisted_waiver_field_cannot_steer_the_disclosure(tmp_path: Path) -> None:
    """Forgery: persist waiver-looking fields and expect them to be ignored.

    Condition 3: the decision is replay-derived with ZERO producer input. A
    stamped ``fidelity.waived`` (or a census-shaped marker) must neither create
    a waiver nor suppress a real one.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(_execution_failed_attempt())
        forged = _waiver_terminal()
        forged["fidelity"]["waived"] = True
        forged["fidelity"]["waiver_reason"] = "operator asserted machine death"
        # A foreign field must not be silently accepted into the record: the
        # closed model schema refuses it before any waiver logic runs, so no
        # persisted bit can ever reach the derivation.
        with pytest.raises((ReductionError, PayloadValidationError)) as refused:
            _append(reducer, forged)
        assert "waived" in str(refused.value), (
            "the refusal must name the forged field an operator has to remove"
        )


def test_the_waiver_is_recomputed_not_remembered_across_reopen(tmp_path: Path) -> None:
    """Condition 3/8: reopening recomputes the same disclosure from bytes alone."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(_execution_failed_attempt())
        terminal = _waiver_terminal()
        assert _waived(reducer, terminal), "the honest control arm must be waived"
        live = reducer.fidelity_waived_terminals()
    with CanonicalReducer(paths, ["m_example"]) as reopened:
        assert reopened.fidelity_waived_terminals() == live, (
            "the waiver disclosure must be replay-derived, not process-local"
        )


def test_a_waived_terminal_never_reads_as_fidelity_audited(tmp_path: Path) -> None:
    """Condition 4: waived is segregated from audited in the derived census."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(_execution_failed_attempt())
        terminal = _waiver_terminal()
        assert _waived(reducer, terminal)
        record = reducer.current_records["m_example"]
        assert record["fidelity"].get("current") is not True, (
            "a waived terminal must never carry a current fidelity gate"
        )
        assert record["completeness"].get("required_fidelity_current") is not True, (
            "a waived terminal must never read as having its required fidelity satisfied"
        )


def test_a_waiver_cannot_upgrade_the_disposition(tmp_path: Path) -> None:
    """Condition 1: the waiver is disposition-invariant -- the terminal stays failed."""

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(_execution_failed_attempt())
        terminal = _waiver_terminal()
        assert _waived(reducer, terminal)
        record = reducer.current_records["m_example"]
        assert record["status"]["kind"] == "failed"
        assert record["status"]["code"] == "failed:import"


def test_a_waived_stable_id_is_not_carried_across_work_generations(tmp_path: Path) -> None:
    """Forgery: re-drive under a new work generation and expect the waiver to lapse.

    The waiver binds to the CURRENT work's evidence. A fresh work generation
    with no execution evidence must not inherit the prior generation's waiver.
    """

    paths = _paths(tmp_path)
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(_execution_failed_attempt())
        terminal = _waiver_terminal()
        assert _waived(reducer, terminal)
        redriven = deepcopy(terminal)
        redriven["status"]["attempt_ids"] = []
        with pytest.raises(ReductionError):
            _append(reducer, redriven)
