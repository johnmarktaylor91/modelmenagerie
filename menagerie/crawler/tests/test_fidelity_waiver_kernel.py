"""M1 validation-kernel change: the execution-failure fidelity waiver (KERNEL_DECISION).

The tri-lab deliberation of 2026-08-11 approved extending the reducer's fidelity
no-consumer predicate to R3/R4 FAILURE terminals whose death is replay-proven at an
execution stage, under ten binding conditions. This suite is the mandated full
adversarial matrix (condition 10): forged claim, wrong environment generation,
cross-model evidence, wrong stage / error stage, open reason, exhaustion reason,
legacy proof, unbound decisive attempt, receipt beyond stage, recovery success,
and the runs-negative scope lock -- plus the majority-fork RED tests in BOTH
directions (partial-success-then-required-failure stays eligible; successes that
complete the policy refuse) and the condition-8 derived disclosure census.

Every negative arm must fall back to the unweakened refusal: the waiver is a
narrow, replay-derived carve-out, never a loosening of "required fidelity is
missing its gate".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.reducer import (
    FIDELITY_WAIVER_EXECUTION_STAGES,
    CanonicalReducer as ProductionCanonicalReducer,
    ReductionError,
)
from menagerie.crawler.status import funnel_counts, record_is_release_eligible
from menagerie.crawler.tests.conftest import (
    HASH,
    make_attempt,
    make_authority_context,
    make_failed_attempt,
    make_model,
    rebind_attempt_raw_proof,
    rebind_nonaward_parent_proof,
)

#: Kernel tests are sub-second and load-bearing for the mandatory smoke gate.
pytestmark = pytest.mark.smoke

_HASH2 = "sha256:" + "e" * 64


class CanonicalReducer(ProductionCanonicalReducer):
    """Project reducer-owned v3 fields before exercising admission invariants."""

    def __init__(self, paths: LedgerPaths, stable_ids: Any) -> None:
        super().__init__(paths, make_authority_context(stable_ids))

    def append_model(self, model: Any) -> Any:
        return super().append_model(self.prepare_model(model))


def _paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths."""

    return LedgerPaths(
        models=tmp_path / "models.jsonl",
        attempts=tmp_path / "attempts.jsonl",
        gates=tmp_path / "gates.jsonl",
    )


def _execution_failed_attempt(
    stable_id: str = "m_example",
    *,
    attempt_id: str = "attempt-exec-1",
    attempt_no: int = 1,
    stage: str = "import",
    reason_code: str = "import-exception",
    env_generation: Optional[str] = HASH,
    recipe_revision: str = HASH,
    rung: str = "R3_PORT",
) -> dict[str, Any]:
    """Build one worker-side execution-stage failed attempt with honest evidence.

    Parameters
    ----------
    stable_id, attempt_id, attempt_no:
        Exact model, attempt identity, and ledger order.
    stage, reason_code:
        Closed execution failure stage and reason.
    env_generation:
        Environment generation the worker ran under (``None`` for a death at
        ``environment``, where no generation ever sealed).
    recipe_revision, rung:
        Recipe identity the attempt executed, matching the terminal's current
        ``implementation.recipe_revision``, and the selected attempted rung.
    """

    attempt = make_failed_attempt(stable_id, attempt_id=attempt_id, stage=stage)
    attempt["attempt_no"] = attempt_no
    attempt["error"]["stage"] = stage
    attempt["error"]["reason_code"] = reason_code
    attempt["identities"]["recipe"] = recipe_revision
    attempt["attempted_rungs"] = [rung]
    if stage == "environment":
        attempt["environment"] = None
        attempt["identities"]["environment"] = None
        attempt["identities"]["execution"] = None
    else:
        attempt["actor"] = "worker"
        attempt["environment"] = make_attempt(stable_id)["environment"]
        attempt["identities"]["environment"] = env_generation
        attempt["identities"]["execution"] = HASH
        attempt["supervisor_observation"].update(
            {
                "exit_code": 1,
                "stdout_sha256": hash_bytes(b""),
                "stderr_sha256": hash_bytes(b""),
            }
        )
        receipt = attempt["worker_receipt"]
        receipt["present"] = True
        receipt["input_note"] = "worker died at the claimed execution stage"
        if stage in {"constructor", "input", "forward"}:
            receipt["constructor_started"] = True
        if stage in {"input", "forward"}:
            receipt["constructor_completed"] = True
        if stage == "forward":
            receipt["input_completed"] = True
            receipt["forward_started"] = True
    attempt.pop("ledger_seq", None)
    attempt.pop("payload_sha256", None)
    return rebind_nonaward_parent_proof(attempt)


def _succeeded_forward_attempt(
    stable_id: str = "m_example",
    *,
    attempt_id: str,
    attempt_no: int,
    mode: str = "eval",
    rung: str = "R3_PORT",
) -> dict[str, Any]:
    """Build one authenticated succeeded forward attempt at an explicit order."""

    attempt = make_attempt(stable_id, attempt_id=attempt_id, mode=mode)
    attempt["attempt_no"] = attempt_no
    attempt["identities"]["recipe"] = HASH
    attempt["worker_receipt"]["observed_recipe_revision"] = HASH
    attempt["attempted_rungs"] = [rung]
    attempt.pop("ledger_seq", None)
    attempt.pop("payload_sha256", None)
    return rebind_attempt_raw_proof(attempt)


def _waiver_terminal(
    stable_id: str = "m_example",
    *,
    stage: str = "import",
    reason_code: str = "import-exception",
    attempt_ids: tuple[str, ...] = ("attempt-exec-1",),
    env_generation: Optional[str] = HASH,
    rung: str = "R3_PORT",
) -> dict[str, Any]:
    """Build one R3/R4 execution-stage failure terminal with pending metadata."""

    model = make_model(stable_id, accepted=False, status_code=f"failed:{stage}")
    model["source_resolution"]["rung"] = rung
    model["fidelity"].update({"required": True, "reason": "R3 port requires fidelity"})
    model["status"].update(
        {
            "reason_code": reason_code,
            "attempt_ids": list(attempt_ids),
            "environment": "env-test",
        }
    )
    model["execution"].update(
        {
            "accepted_attempt_ids": [],
            # The schema mandates non-null identities: a death with no observed
            # environment carries the driver's exact NOT_APPLICABLE sentinel.
            "env_generation": (
                env_generation
                if env_generation is not None
                else stable_hash({"environment": "not-applicable"})
            ),
            "environment_id": "env-test" if env_generation is not None else "not-applicable",
        }
    )
    model["modes"]["per_mode_run"] = {}
    model["completeness"]["required_fidelity_current"] = False
    return model


def _append(reducer: CanonicalReducer, model: dict[str, Any]) -> Any:
    """Append one candidate through the production projection path."""

    return reducer.append_model(model)


# --- the waiver grants (per-stage positive arms) -------------------------------


@pytest.mark.parametrize(
    ("stage", "reason_code"),
    (
        ("environment", "build-failed"),
        ("import", "import-exception"),
        ("constructor", "exception"),
        ("input", "generation-exception"),
        ("forward", "exception"),
    ),
)
def test_execution_stage_machine_death_waives_the_fidelity_gate(
    tmp_path: Path, stage: str, reason_code: str
) -> None:
    """Every closed waiver stage admits a replay-proven R3 machine death.

    RED on the pre-M1 kernel: each append raised "required fidelity is missing
    its gate", buying a fidelity session that could adjudicate nothing.
    """

    env_generation = None if stage == "environment" else HASH
    attempt = _execution_failed_attempt(stage=stage, reason_code=reason_code)
    model = _waiver_terminal(stage=stage, reason_code=reason_code, env_generation=env_generation)
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        result = _append(reducer, model)
        assert result.appended
        record = reducer.current_records["m_example"]
        assert record["status"]["code"] == f"failed:{stage}"
        # Condition 8: the marker is DERIVED, never persisted.
        assert reducer.fidelity_waived_terminals() == ("m_example",)
        assert "fidelity_waived" not in record
        assert not record_is_release_eligible(record, reducer.current_records)


def test_r4_reimplement_is_waiver_eligible(tmp_path: Path) -> None:
    """The waiver covers both fidelity-required rungs, not just R3."""

    attempt = _execution_failed_attempt(rung="R4_REIMPLEMENT")
    model = _waiver_terminal(rung="R4_REIMPLEMENT")
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert _append(reducer, model).appended
        assert reducer.fidelity_waived_terminals() == ("m_example",)


def test_partial_success_then_required_failure_stays_waiver_eligible(
    tmp_path: Path,
) -> None:
    """The deliberation's majority fork, positive direction (Fable+Sol, 2/3).

    One cold run completed, the second required run died at environment: the
    model cannot complete its required execution policy for machine reasons, so
    the failure terminal plus condition-8 disclosure covers it.
    """

    success = _succeeded_forward_attempt(attempt_id="attempt-cold-1", attempt_no=1)
    failure = _execution_failed_attempt(
        attempt_id="attempt-env-2",
        attempt_no=2,
        stage="environment",
        reason_code="build-failed",
        env_generation=None,
    )
    # The record honestly binds the generation the completed run observed; the
    # decisive environment death itself never sealed one.
    model = _waiver_terminal(
        stage="environment",
        reason_code="build-failed",
        attempt_ids=("attempt-env-2",),
        env_generation=HASH,
    )
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(success)
        reducer.append_attempt(failure)
        assert _append(reducer, model).appended
        assert reducer.fidelity_waived_terminals() == ("m_example",)


# --- the adversarial matrix (every arm falls back to the unweakened refusal) ---


def test_recovery_success_completing_the_policy_refuses_the_waiver(
    tmp_path: Path,
) -> None:
    """The majority fork, negative direction: completed policy = gate has a consumer.

    R3 requires two cold forwards per meaningful mode. When authenticated
    same-generation successes collectively satisfy that policy, a stray machine
    failure cannot waive fidelity.
    """

    first = _succeeded_forward_attempt(attempt_id="attempt-cold-1", attempt_no=1)
    second = _succeeded_forward_attempt(attempt_id="attempt-cold-2", attempt_no=2)
    failure = _execution_failed_attempt(attempt_id="attempt-exec-3", attempt_no=3)
    model = _waiver_terminal(attempt_ids=("attempt-exec-3",))
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(first)
        reducer.append_attempt(second)
        reducer.append_attempt(failure)
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_failure_superseded_by_a_later_success_refuses_the_waiver(
    tmp_path: Path,
) -> None:
    """A later success in the required slot supersedes the failure (condition 6)."""

    failure = _execution_failed_attempt(attempt_id="attempt-exec-1", attempt_no=1)
    recovery = _succeeded_forward_attempt(attempt_id="attempt-cold-2", attempt_no=2)
    model = _waiver_terminal(attempt_ids=("attempt-exec-1",))
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(failure)
        reducer.append_attempt(recovery)
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_conflicting_attempt_and_ledger_orders_refuse_the_waiver(tmp_path: Path) -> None:
    """A concurrent-looking completion order is ambiguity, never waiver proof.

    ``attempt_no`` is allocated before work completes, whereas ``ledger_seq``
    is reducer-owned completion order. A forged high attempt number on the
    earlier failure used to keep it decisive after a later recovery append.
    """

    failure = _execution_failed_attempt(attempt_id="attempt-exec-99", attempt_no=99)
    recovery = _succeeded_forward_attempt(attempt_id="attempt-cold-2", attempt_no=2)
    model = _waiver_terminal(attempt_ids=("attempt-exec-99",))
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(failure)
        reducer.append_attempt(recovery)
        assert reducer._derive_execution_failure_waiver(model) is False
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


@pytest.mark.parametrize(
    ("later_stage", "reason_code"),
    (
        ("resource", "oom"),
        ("policy", "network-attempt"),
        ("runner", "protocol-violation"),
    ),
)
def test_later_ineligible_runner_stage_refuses_an_earlier_waiver(
    tmp_path: Path, later_stage: str, reason_code: str
) -> None:
    """Later runner-lifecycle evidence cannot be hidden by claiming an old death."""

    failure = _execution_failed_attempt(attempt_id="attempt-import-1", attempt_no=1)
    later = _execution_failed_attempt(
        attempt_id=f"attempt-{later_stage}-2",
        attempt_no=2,
        stage=later_stage,
        reason_code=reason_code,
    )
    model = _waiver_terminal(attempt_ids=("attempt-import-1",))
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(failure)
        reducer.append_attempt(later)
        assert reducer._derive_execution_failure_waiver(model) is False
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_earlier_ineligible_stage_progress_disqualifies_a_later_claim(
    tmp_path: Path,
) -> None:
    """Any sibling runner attempt past the claimed stage poisons the waiver."""

    resource = _execution_failed_attempt(
        attempt_id="attempt-resource-1",
        attempt_no=1,
        stage="resource",
        reason_code="oom",
    )
    resource["worker_receipt"]["constructor_started"] = True
    resource = rebind_nonaward_parent_proof(resource)
    failure = _execution_failed_attempt(attempt_id="attempt-import-2", attempt_no=2)
    model = _waiver_terminal(attempt_ids=("attempt-import-2",))
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(resource)
        reducer.append_attempt(failure)
        assert reducer._derive_execution_failure_waiver(model) is False
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


@pytest.mark.parametrize(
    ("stage", "reason_code", "forged_actor"),
    (
        ("environment", "build-failed", "worker"),
        ("import", "import-exception", "driver"),
        ("constructor", "exception", "driver"),
        ("input", "generation-exception", "driver"),
        ("forward", "exception", "driver"),
    ),
)
def test_stage_actor_forgery_refuses_the_waiver(
    tmp_path: Path, stage: str, reason_code: str, forged_actor: str
) -> None:
    """Only the stage's actual observing actor may prove its machine death."""

    env_generation = None if stage == "environment" else HASH
    attempt = _execution_failed_attempt(
        stage=stage,
        reason_code=reason_code,
        env_generation=env_generation,
    )
    attempt["actor"] = forged_actor
    attempt = rebind_nonaward_parent_proof(attempt)
    model = _waiver_terminal(
        stage=stage,
        reason_code=reason_code,
        env_generation=env_generation,
    )
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert reducer._derive_execution_failure_waiver(model) is False
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("network_attempted", True),
        ("checkpoint_or_weight_read_attempted", True),
        ("write_outside_scratch_attempted", True),
        ("credentials_present", True),
        ("torchlens_import_attempted", True),
        ("cache_read_attempted", True),
        ("socket_targets", ["forged.example:443"]),
        ("checkpoint_paths", ["/forged/checkpoint"]),
        ("write_paths", ["/forged/output"]),
    ),
)
def test_policy_observation_forgery_refuses_the_waiver(
    tmp_path: Path, field: str, value: Any
) -> None:
    """Every policy-observation contamination field poisons no-consumer proof."""

    attempt = _execution_failed_attempt()
    attempt["policy_observation"][field] = value
    model = _waiver_terminal()
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert reducer._derive_execution_failure_waiver(model) is False


@pytest.mark.parametrize("field", ("timed_out", "rss_exceeded", "signal"))
def test_parent_resource_or_signal_forgery_refuses_the_waiver(tmp_path: Path, field: str) -> None:
    """Parent-observed timeout, RSS, or signal evidence routes outside K2."""

    attempt = _execution_failed_attempt()
    parent = attempt["parent_attestation"]
    parent[field] = 9 if field == "signal" else True
    parent["attestation_sha256"] = stable_hash(
        {key: value for key, value in parent.items() if key != "attestation_sha256"}
    )
    if field == "signal":
        attempt["supervisor_observation"]["signal"] = 9
    model = _waiver_terminal()
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert reducer._derive_execution_failure_waiver(model) is False


@pytest.mark.parametrize(
    ("stage", "reason_code", "receipt_mutation"),
    (
        ("import", "import-exception", {"present": False}),
        ("constructor", "exception", {"constructor_started": False}),
        ("input", "generation-exception", {"constructor_completed": False}),
        ("forward", "exception", {"forward_started": False}),
        ("import", "import-exception", {"output_value_sha256": HASH}),
        ("constructor", "exception", {"input_kind": "standard-image"}),
    ),
)
def test_receipt_shape_forgery_refuses_the_waiver(
    tmp_path: Path,
    stage: str,
    reason_code: str,
    receipt_mutation: dict[str, Any],
) -> None:
    """Missing required milestones and impossible later payloads both refuse."""

    attempt = _execution_failed_attempt(stage=stage, reason_code=reason_code)
    attempt["worker_receipt"].update(receipt_mutation)
    attempt = rebind_nonaward_parent_proof(attempt)
    model = _waiver_terminal(stage=stage, reason_code=reason_code)
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert reducer._derive_execution_failure_waiver(model) is False


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong-rung",
        "not-required",
        "fidelity-current",
        "execution-current",
        "accepted-attempt",
        "wrong-status-kind",
    ),
)
def test_terminal_scope_forgery_refuses_the_waiver(tmp_path: Path, mutation: str) -> None:
    """Every caller-scope field is reasserted inside the waiver predicate."""

    attempt = _execution_failed_attempt()
    model = _waiver_terminal()
    if mutation == "wrong-rung":
        model["source_resolution"]["rung"] = "R1_LIBRARY"
    elif mutation == "not-required":
        model["fidelity"]["required"] = False
    elif mutation == "fidelity-current":
        model["fidelity"]["current"] = True
    elif mutation == "execution-current":
        model["execution"]["current"] = True
    elif mutation == "accepted-attempt":
        model["execution"]["accepted_attempt_ids"] = ["attempt-exec-1"]
    else:
        model["status"]["kind"] = "runs"
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert reducer._derive_execution_failure_waiver(model) is False


def test_environment_id_and_success_recipe_forgery_refuse_the_waiver(
    tmp_path: Path,
) -> None:
    """All same-generation environment IDs and recipe witnesses must agree."""

    failure = _execution_failed_attempt(attempt_id="attempt-import-2", attempt_no=2)
    wrong_environment = _waiver_terminal(attempt_ids=("attempt-import-2",))
    wrong_environment["execution"]["environment_id"] = "env-forged"
    with CanonicalReducer(_paths(tmp_path / "environment"), ["m_example"]) as reducer:
        reducer.append_attempt(failure)
        assert reducer._derive_execution_failure_waiver(wrong_environment) is False

    success = _succeeded_forward_attempt(attempt_id="attempt-cold-1", attempt_no=1)
    success["identities"]["recipe"] = _HASH2
    success["worker_receipt"]["observed_recipe_revision"] = _HASH2
    success = rebind_attempt_raw_proof(success)
    model = _waiver_terminal(attempt_ids=("attempt-import-2",))
    with CanonicalReducer(_paths(tmp_path / "recipe"), ["m_example"]) as reducer:
        reducer.append_attempt(success)
        reducer.append_attempt(failure)
        assert reducer._derive_execution_failure_waiver(model) is False


@pytest.mark.parametrize(
    "mutation",
    ("attempted-rung", "execution-identity", "native-crash"),
)
def test_decisive_attempt_binding_forgery_refuses_the_waiver(tmp_path: Path, mutation: str) -> None:
    """Rung, execution identity, and native-crash routing remain fail-closed."""

    attempt = _execution_failed_attempt()
    if mutation == "attempted-rung":
        attempt["attempted_rungs"] = ["R1_LIBRARY"]
    elif mutation == "execution-identity":
        attempt["identities"]["execution"] = None
    else:
        attempt["error"]["native_crash"] = True
    model = _waiver_terminal()
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert reducer._derive_execution_failure_waiver(model) is False


def test_exhaustion_reason_is_not_a_machine_death(tmp_path: Path) -> None:
    """Condition 5: effort exhaustion never waives fidelity -- unfinished, not dead."""

    attempt = _execution_failed_attempt(reason_code="effort-cap-exhausted")
    model = _waiver_terminal(reason_code="effort-cap-exhausted")
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_open_reason_refuses_at_the_predicate(tmp_path: Path) -> None:
    """Condition 5: a reason outside the closed stage vocabulary never waives.

    ``_validate_status`` refuses open reasons before the gate pipeline runs, so
    this arm pins the PREDICATE directly: the waiver itself must be closed and
    must not rely on an earlier check for its own soundness.
    """

    attempt = _execution_failed_attempt()
    attempt["error"]["reason_code"] = "novel-unclosed-reason"
    attempt = rebind_nonaward_parent_proof(attempt)
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        with pytest.raises(ReductionError, match="not allowed for its stage"):
            reducer.append_attempt(attempt)
    # A stage-valid but terminal-contradicting reason refuses at the predicate.
    mismatched = _execution_failed_attempt(reason_code="module-missing")
    model = _waiver_terminal(reason_code="import-exception")
    with CanonicalReducer(_paths(tmp_path / "mismatch"), ["m_example"]) as reducer:
        reducer.append_attempt(mismatched)
        assert reducer._derive_execution_failure_waiver(model) is False


def test_wrong_stage_and_wrong_error_stage_refuse(tmp_path: Path) -> None:
    """Condition 4: status stage, decisive attempt stage, and error stage agree."""

    constructor_death = _execution_failed_attempt(stage="constructor", reason_code="exception")
    import_claim = _waiver_terminal(stage="import", reason_code="import-exception")
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(constructor_death)
        with pytest.raises(ReductionError):
            _append(reducer, import_claim)

    inconsistent = _execution_failed_attempt(attempt_id="attempt-exec-9")
    inconsistent["error"]["stage"] = "constructor"
    inconsistent["error"]["reason_code"] = "exception"
    inconsistent = rebind_nonaward_parent_proof(inconsistent)
    with CanonicalReducer(_paths(tmp_path / "err"), ["m_example"]) as reducer:
        with pytest.raises(ReductionError, match="error stages must match"):
            reducer.append_attempt(inconsistent)


def test_receipt_beyond_stage_refuses(tmp_path: Path) -> None:
    """Condition 4 + adopted F3: decisive evidence must not exceed the claimed stage."""

    attempt = _execution_failed_attempt()
    attempt["worker_receipt"]["constructor_started"] = True
    attempt = rebind_nonaward_parent_proof(attempt)
    model = _waiver_terminal()
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_sibling_failed_attempt_beyond_stage_refuses(tmp_path: Path) -> None:
    """A non-succeeded sibling with evidence past the claimed stage disqualifies."""

    constructor_death = _execution_failed_attempt(
        attempt_id="attempt-ctor-1",
        attempt_no=1,
        stage="constructor",
        reason_code="exception",
    )
    env_death = _execution_failed_attempt(
        attempt_id="attempt-env-2",
        attempt_no=2,
        stage="environment",
        reason_code="build-failed",
        env_generation=None,
    )
    model = _waiver_terminal(
        stage="environment",
        reason_code="build-failed",
        attempt_ids=("attempt-env-2",),
        env_generation=None,
    )
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(constructor_death)
        reducer.append_attempt(env_death)
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_wrong_environment_generation_refuses(tmp_path: Path) -> None:
    """Condition 7: the derived binding must equal ``execution.env_generation``.

    The authority pipeline already refuses the contradiction at append time; the
    predicate must ALSO assert it so the waiver never depends on a sibling check.
    """

    attempt = _execution_failed_attempt(env_generation=_HASH2)
    model = _waiver_terminal(env_generation=HASH)
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        with pytest.raises(ReductionError):
            _append(reducer, model)
        assert reducer._derive_execution_failure_waiver(model) is False


def test_pre_repair_recipe_revision_refuses(tmp_path: Path) -> None:
    """Condition 7 rider: a pre-repair death can never waive a repaired recipe."""

    attempt = _execution_failed_attempt(recipe_revision=_HASH2)
    model = _waiver_terminal()
    assert model["implementation"]["recipe_revision"] == HASH
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_forged_claim_with_no_ledger_attempt_refuses(tmp_path: Path) -> None:
    """Condition 1: a terminal claim with no authenticated evidence never waives."""

    model = _waiver_terminal(attempt_ids=("attempt-forged",))
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        with pytest.raises(ReductionError):
            _append(reducer, model)
        # The forged reference cannot even resolve a work identity: fail-closed.
        with pytest.raises(ReductionError):
            reducer._derive_execution_failure_waiver(model)


def test_referenced_attempt_missing_from_the_ledger_disqualifies(
    tmp_path: Path,
) -> None:
    """Condition 4: every referenced attempt must exist, even beside real evidence."""

    attempt = _execution_failed_attempt()
    model = _waiver_terminal(attempt_ids=("attempt-exec-1", "attempt-ghost"))
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert reducer._derive_execution_failure_waiver(model) is False


def test_cross_model_evidence_refuses(tmp_path: Path) -> None:
    """Condition 1: another model's death can never prove this model's waiver."""

    other = _execution_failed_attempt("m_other", attempt_id="attempt-other-1")
    model = _waiver_terminal(attempt_ids=("attempt-other-1",))
    with CanonicalReducer(_paths(tmp_path), ["m_example", "m_other"]) as reducer:
        reducer.append_attempt(other)
        with pytest.raises(ReductionError):
            _append(reducer, model)
        # Another model's attempt resolves no work identity here: fail-closed.
        with pytest.raises(ReductionError):
            reducer._derive_execution_failure_waiver(model)


def test_waiver_evidence_flows_only_through_current_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Condition 1 + legacy-row arm: only the authenticated view is evidence.

    Legacy rows are structurally outside the evidence channel because
    ``_current_attempt_records`` admits only replay-authenticated current-v3
    rows. With that view empty, a row still visible to the raw attempt index
    can prove nothing -- pinning that the waiver never consults
    ``status.attempt_ids`` projections, ``_attempt_index()``, or any producer
    field for its evidence.
    """

    attempt = _execution_failed_attempt()
    model = _waiver_terminal()
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        assert reducer._derive_execution_failure_waiver(model) is True
        monkeypatch.setattr(
            ProductionCanonicalReducer,
            "_current_attempt_records",
            lambda self: (),
        )
        assert reducer._derive_execution_failure_waiver(model) is False


def test_unbound_decisive_attempt_fails_closed(tmp_path: Path) -> None:
    """Condition 1: a tampered proof row poisons replay loudly, never silently."""

    from menagerie.crawler.constants import ATTEMPT_SCHEMA_VERSION_V3
    from menagerie.crawler.recordio import JsonlLedger

    paths = _paths(tmp_path)
    tampered = _execution_failed_attempt()
    # Break the parent attestation self-hash while keeping the row schema-valid
    # and ledger-bound: the refusal must come from proof replay, not the codec.
    tampered["parent_attestation"] = dict(tampered["parent_attestation"])
    tampered["parent_attestation"]["exit_code"] = 99
    with JsonlLedger(paths.attempts, ATTEMPT_SCHEMA_VERSION_V3) as ledger:
        ledger.append(tampered)
    model = _waiver_terminal()
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        with pytest.raises(ReductionError):
            reducer._derive_execution_failure_waiver(model)


def test_runs_revision_with_execution_stage_failure_still_requires_fidelity(
    tmp_path: Path,
) -> None:
    """Condition 2 runs-negative RED: the failure branch scope lock holds.

    An R3 ``runs`` revision carrying an execution-stage failed attempt in its
    work still raises "required fidelity is missing its gate" -- the waiver can
    never leak into the success branch.
    """

    from menagerie.crawler.tests.conftest import _bind_model_identities, make_gate

    stable_ids = ["m_example", *[f"m_{index}" for index in range(9)]]
    success = _succeeded_forward_attempt(attempt_id="attempt-1", attempt_no=2)
    bait = _execution_failed_attempt(attempt_id="attempt-exec-1", attempt_no=1)
    model = make_model(accepted=True)
    model["source_resolution"]["rung"] = "R3_PORT"
    model["fidelity"].update({"required": True, "reason": "R3 port requires fidelity"})
    model["completeness"]["required_fidelity_current"] = False
    _bind_model_identities(model)
    gate = make_gate(stable_ids)
    item = gate["items"][0]
    assert item["stable_id"] == "m_example"
    item["vet_identity"] = model["accuracy_gate"]["vet_identity"]
    item["rung_check"]["selected_rung"] = "R3_PORT"
    item["rung_check"]["highest_applicable"] = "R3_PORT"
    gate["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in gate.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    with CanonicalReducer(_paths(tmp_path), stable_ids) as reducer:
        reducer.append_gate(gate)
        reducer.append_attempt(bait)
        reducer.append_attempt(success)
        with pytest.raises(ReductionError, match="fidelity"):
            _append(reducer, model)


def test_runner_arm_and_resource_policy_exclusions_are_unchanged(
    tmp_path: Path,
) -> None:
    """Condition 3: the closed stage set admits exactly the five execution stages."""

    assert FIDELITY_WAIVER_EXECUTION_STAGES == frozenset(
        {"environment", "import", "constructor", "input", "forward"}
    )
    for stage, reason_code in (("resource", "oom"), ("policy", "network-attempt")):
        attempt = _execution_failed_attempt(
            attempt_id=f"attempt-{stage}", stage=stage, reason_code=reason_code
        )
        model = _waiver_terminal(
            stage=stage, reason_code=reason_code, attempt_ids=(f"attempt-{stage}",)
        )
        with CanonicalReducer(_paths(tmp_path / stage), ["m_example"]) as reducer:
            reducer.append_attempt(attempt)
            with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
                _append(reducer, model)


def test_r1_pre_execution_terminal_keeps_the_historical_arm(tmp_path: Path) -> None:
    """The pre-execution arm (incl. its evidence loop) is byte-for-byte preserved."""

    failed_attempt = make_failed_attempt()
    model = make_model(accepted=False, status_code="failed:source")
    model["intake"]["preserved_legacy_flags"] = ["legacy-fidelity-claim"]
    model["status"]["attempt_ids"] = [failed_attempt["attempt_id"]]
    model["execution"]["accepted_attempt_ids"] = []
    model["completeness"]["required_fidelity_current"] = False
    model["fidelity"]["required"] = True
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(failed_attempt)
        assert _append(reducer, model).appended


# --- condition 8: derived disclosure census ------------------------------------


def test_census_segregates_waived_from_audited_failures(tmp_path: Path) -> None:
    """Waived terminals are counted, disclosed, and never read as audited."""

    attempt = _execution_failed_attempt()
    model = _waiver_terminal()
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        _append(reducer, model)
        counts = funnel_counts(reducer.current_records)
    assert counts["fidelity_waived_machine_failure"] == 1
    assert "fidelity_audited_failure" not in counts

    audited = _waiver_terminal("m_example")
    audited["status"].update(
        {
            "kind": "failed",
            "code": "failed:fidelity",
            "stage": "fidelity",
            "reason_code": "major-drift-cap-exhausted",
        }
    )
    audited["fidelity"].update({"verdict": "major-drift", "gate_id": "gate-fid-1"})
    counts = funnel_counts({"m_example": audited})
    assert counts["fidelity_audited_failure"] == 1
    assert "fidelity_waived_machine_failure" not in counts


def test_waived_terminal_is_never_release_eligible(tmp_path: Path) -> None:
    """Release eligibility keeps excluding waived R3/R4 failure terminals."""

    attempt = _execution_failed_attempt()
    model = _waiver_terminal()
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        _append(reducer, model)
        record = reducer.current_records["m_example"]
        assert record["completeness"]["release_eligible"] is False
        assert not record_is_release_eligible(record, reducer.current_records)


def test_waiver_survives_replay_with_no_persisted_field(tmp_path: Path) -> None:
    """Condition 1: the waiver is recomputed from the ledgers on a cold rebuild."""

    paths = _paths(tmp_path)
    attempt = _execution_failed_attempt()
    model = _waiver_terminal()
    with CanonicalReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        _append(reducer, model)
    with CanonicalReducer(paths, ["m_example"]) as reborn:
        assert reborn.fidelity_waived_terminals() == ("m_example",)
        record = reborn.current_records["m_example"]
        assert "waiver" not in canonical_json_bytes(record).decode().lower()


# --- C1 rehearsal tightenings (2026-08-12): forgeries beyond the shipped matrix -


def _beyond_stage_late_sibling(
    stable_id: str = "m_example",
    *,
    attempt_id: str,
    attempt_no: int,
    stage: str,
    reason_code: str,
) -> dict[str, Any]:
    """A failed NON-waiver-stage attempt whose receipt proves a forward ran.

    A resource or policy death that struck AFTER the forward completed honestly
    carries the full milestone receipt. Such an attempt is exactly the evidence
    the no-consumer contract cares about -- the code ran -- and it must disqualify
    the waiver even though its own stage sits outside the closed waiver set.
    """

    attempt = make_failed_attempt(stable_id, attempt_id=attempt_id, stage=stage)
    attempt["attempt_no"] = attempt_no
    attempt["error"]["stage"] = stage
    attempt["error"]["reason_code"] = reason_code
    attempt["identities"]["recipe"] = HASH
    attempt["actor"] = "worker"
    attempt["environment"] = make_attempt(stable_id)["environment"]
    attempt["identities"]["environment"] = HASH
    attempt["identities"]["execution"] = HASH
    attempt["supervisor_observation"].update(
        {
            "exit_code": 1,
            "stdout_sha256": hash_bytes(b""),
            "stderr_sha256": hash_bytes(b""),
        }
    )
    attempt["worker_receipt"].update(
        {
            "present": True,
            "constructor_started": True,
            "constructor_completed": True,
            "input_completed": True,
            "forward_started": True,
            "forward_completed": True,
            "input_note": "forward completed before the late-stage death",
        }
    )
    attempt.pop("ledger_seq", None)
    attempt.pop("payload_sha256", None)
    return rebind_nonaward_parent_proof(attempt)


@pytest.mark.parametrize(
    ("stage", "reason_code"),
    (("resource", "oom"), ("policy", "network-attempt")),
)
def test_beyond_stage_evidence_outside_the_waiver_stage_set_refuses(
    tmp_path: Path, stage: str, reason_code: str
) -> None:
    """Condition 4 closes over EVERY non-succeeded same-work attempt.

    RED on the shipped kernel: the beyond-stage ceiling loop filtered siblings
    to ``FIDELITY_WAIVER_EXECUTION_STAGES``, so a resource/policy death
    carrying a completed-forward receipt was invisible and the waiver granted
    over a work generation whose code provably ran.
    """

    sibling = _beyond_stage_late_sibling(
        attempt_id="attempt-late-1", attempt_no=1, stage=stage, reason_code=reason_code
    )
    decisive = _execution_failed_attempt(attempt_id="attempt-exec-2", attempt_no=2)
    model = _waiver_terminal(attempt_ids=("attempt-exec-2",))
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(sibling)
        reducer.append_attempt(decisive)
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_foreign_model_attempt_reference_refuses_the_waiver(tmp_path: Path) -> None:
    """Condition 1: referenced evidence must live at the exact stable ID and work.

    RED on the shipped kernel: the fail-closed disqualifier only required a
    referenced attempt to EXIST in the current ledger index, so a terminal
    citing ANOTHER model's attempt in ``status.attempt_ids`` still waived --
    laundered provenance passing the referenced-but-missing check.
    """

    foreign = _succeeded_forward_attempt("m_other", attempt_id="attempt-foreign-1", attempt_no=1)
    decisive = _execution_failed_attempt(attempt_id="attempt-exec-2", attempt_no=2)
    model = _waiver_terminal(attempt_ids=("attempt-exec-2", "attempt-foreign-1"))
    with CanonicalReducer(_paths(tmp_path), ["m_example", "m_other"]) as reducer:
        reducer.append_attempt(foreign)
        reducer.append_attempt(decisive)
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_forged_meaningful_modes_cannot_dodge_the_completed_policy_refusal(
    tmp_path: Path,
) -> None:
    """Condition 6 counts machine-witnessed successes, never the producer's claim.

    RED on the shipped kernel: the success census keyed on the model's own
    ``meaningful_modes``. Claiming ``train`` while two authenticated eval cold
    forwards satisfied the policy hid the successes from the refusal -- producer
    input steering the waiver, exactly what doctrine part 3 forbids. A claimed
    mode no same-work attempt ever witnessed cannot hold the waiver open.
    """

    first = _succeeded_forward_attempt(attempt_id="attempt-cold-1", attempt_no=1)
    second = _succeeded_forward_attempt(attempt_id="attempt-cold-2", attempt_no=2)
    failure = _execution_failed_attempt(attempt_id="attempt-exec-3", attempt_no=3)
    model = _waiver_terminal(attempt_ids=("attempt-exec-3",))
    model["modes"]["meaningful_modes"] = ["train"]
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(first)
        reducer.append_attempt(second)
        reducer.append_attempt(failure)
        with pytest.raises(ReductionError, match="required fidelity is missing its gate"):
            _append(reducer, model)


def test_attempt_witnessed_short_mode_keeps_the_fork_positive_arm(tmp_path: Path) -> None:
    """The binding majority fork is untouched by the witnessed-mode census.

    A mode the attempt evidence witnesses as short of its policy (one of two
    required cold forwards) keeps the model waiver-eligible exactly as the
    deliberation's positive arm requires.
    """

    success = _succeeded_forward_attempt(attempt_id="attempt-cold-1", attempt_no=1)
    failure = _execution_failed_attempt(
        attempt_id="attempt-env-2",
        attempt_no=2,
        stage="environment",
        reason_code="build-failed",
        env_generation=None,
    )
    model = _waiver_terminal(
        stage="environment",
        reason_code="build-failed",
        attempt_ids=("attempt-env-2",),
        env_generation=HASH,
    )
    with CanonicalReducer(_paths(tmp_path), ["m_example"]) as reducer:
        reducer.append_attempt(success)
        reducer.append_attempt(failure)
        assert _append(reducer, model).appended
        assert reducer.fidelity_waived_terminals() == ("m_example",)
