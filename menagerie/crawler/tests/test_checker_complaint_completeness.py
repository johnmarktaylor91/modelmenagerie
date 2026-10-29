"""Checker complaint-disclosure completeness and cap chargeability.

Rung 9b burned m10517 (tf_efficientnet_lite4) and m7362 (mobileone_s3) to
``failed:accuracy-gate/cannot-verify-cap-exhausted`` although every round's
complaints were repaired: the checker surfaced a brand-new complaint per round
on previously blessed claims (``website`` blessed in round 1, flagged in round
2; ``paradigm`` blessed in rounds 1-2, flagged in round 3) until the raw
three-rejection cap fired. Both rung-9b censuses ranked the fix: each checker
round must enumerate its COMPLETE complaint set against the machine-derived
claim inventory, and a repair round may only be CHARGED against complaints
disclosed in the round it answers.

These tests pin both halves plus the tripwires they must not weaken:

- the complaint-set and chargeability authority in ``checker_dispatch``;
- dispatch-time refusal of an adverse item with incomplete inventory coverage
  (an unenumerable round never publishes, hence never charges);
- the driver counting split (raw generations keep numbering repair artifacts;
  only chargeable rejections bound the cap);
- the frozen rung-9b trajectories reaching a decided verdict within cap;
- an incorrigible lineage still terminalizing exactly as before;
- the authority proof kernel's mirror staying in lockstep and refusing a cap
  claim built from non-chargeable rounds.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional, Sequence

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    _chargeable_metadata_rejection_count,
    _derive_gate_failure,
    _metadata_complaint_fields as authority_complaint_fields,
)
from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    build_metadata_vet_envelope,
    chargeable_metadata_rejections,
    metadata_complaint_fields,
    metadata_item_rejected,
    metadata_rejection_chargeable,
    validate_checker_result_mapping,
)
from menagerie.crawler.constants import METADATA_BATCH_MAX, GateRoute
from menagerie.crawler.driver_models import (
    _chargeable_metadata_repair_count,
    _latest_metadata_complaint_fields,
    _metadata_repair_count,
    _terminal_metadata_gate,
)
from menagerie.crawler.gates import route_metadata_gate
from menagerie.crawler.tests.conftest import make_gate
from menagerie.crawler.tests.test_slice_d_checker_gates import (
    _checker_item_pack,
    _stamped,
)

#: The exact adverse complaint trajectories from the frozen rung-9b ledger
#: (rung-archive/rung9b-dcfc1324-first-runs/records/gates/current-shard.jsonl):
#: consecutive rounds are pairwise DISJOINT, so only the first round answers a
#: disclosed complaint set and only it may charge the bounded cap.
_M10517_ROUNDS: tuple[dict[str, str], ...] = (
    {
        "external_metadata.country": "inaccurate",
        "external_metadata.institution": "inaccurate",
        "input_contract": "inaccurate",
    },
    {"website": "cannot-verify"},
    {"external_metadata.paradigm": "cannot-verify"},
)
_M7362_ROUNDS: tuple[dict[str, str], ...] = (
    {
        "external_metadata.authors": "inaccurate",
        "external_metadata.citation": "inaccurate",
    },
    {
        "external_metadata.country": "cannot-verify",
        "external_metadata.description": "cannot-verify",
        "taxonomy.architecture_tags": "cannot-verify",
        "website": "cannot-verify",
    },
    {"external_metadata.paradigm": "cannot-verify"},
)


def _adverse_gate(
    stable_id: str,
    adverse: Mapping[str, str],
    *,
    gate_id: str,
    gate_round: int = 1,
) -> dict[str, Any]:
    """Build one full-coverage metadata gate with the named adverse claims.

    Parameters
    ----------
    stable_id:
        Single gated model.
    adverse:
        Claim path to non-accurate verdict. Empty means an accepted round.
    gate_id:
        Distinct immutable gate identity.
    gate_round:
        One-based checker round for deterministic authority ordering.

    Returns
    -------
    dict[str, Any]
        Complete gate whose single item covers the whole claim inventory.
    """

    gate = make_gate([stable_id], gate_id=gate_id)
    gate["gate_round"] = gate_round
    item = gate["items"][0]
    for check in item["field_checks"]:
        verdict = adverse.get(str(check["field"]))
        if verdict is not None:
            check["verdict"] = verdict
            check["required_repair"] = f"repair {check['field']}"
    if adverse:
        item["verdict"] = "inaccurate" if "inaccurate" in adverse.values() else "cannot-verify"
        item["required_repairs"] = [f"repair {field}" for field in sorted(adverse)]
    return gate


def _lineage(stable_id: str) -> str:
    """Return the fixture lineage make_gate stamps on its items."""

    return f"work-{stable_id}"


def _history(
    gates: Sequence[Mapping[str, Any]], stable_id: str
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    """Pair each gate with its single fixture item, in ledger order."""

    return tuple(
        (gate, item) for gate in gates for item in gate["items"] if item["stable_id"] == stable_id
    )


def test_metadata_complaint_fields_names_adverse_components() -> None:
    """Complaints are the non-accurate claim rows plus integrity/rung components."""

    accepted = _adverse_gate("m_x", {}, gate_id="gate-a")["items"][0]
    assert metadata_complaint_fields(accepted) == frozenset()
    assert not metadata_item_rejected(accepted)

    adverse = _adverse_gate("m_x", {"website": "cannot-verify"}, gate_id="gate-b")["items"][0]
    assert metadata_complaint_fields(adverse) == frozenset({"website"})
    assert metadata_item_rejected(adverse)

    flipped = deepcopy(accepted)
    flipped["integrity"]["verdict"] = "inaccurate"
    flipped["rung_check"]["verdict"] = "cannot-verify"
    assert metadata_complaint_fields(flipped) == frozenset({"integrity", "rung_check"})

    prefixed = deepcopy(adverse)
    for check in prefixed["field_checks"]:
        if check["field"] == "website":
            check["field"] = "proposed_facts.website"
    assert metadata_complaint_fields(prefixed) == frozenset({"website"})


def test_first_round_rejection_always_charges() -> None:
    """A lineage's first rejection answers nothing and is always chargeable."""

    item = _adverse_gate("m_x", {"website": "cannot-verify"}, gate_id="g")["items"][0]
    assert metadata_rejection_chargeable(item, None) is True


def test_disjoint_followup_complaints_do_not_charge_the_cap() -> None:
    """A rejection made only of first disclosures never consumes the cap."""

    gates = [
        _adverse_gate("m_x", adverse, gate_id=f"gate-{index}", gate_round=index + 1)
        for index, adverse in enumerate(_M10517_ROUNDS)
    ]
    chargeable = chargeable_metadata_rejections(_history(gates, "m_x"))
    assert [gate["gate_id"] for gate, _item in chargeable] == ["gate-0"]


def test_re_raised_complaint_charges_whatever_its_verdict_class() -> None:
    """Re-raising a disclosed complaint charges, even reclassified cannot-verify."""

    gates = [
        _adverse_gate("m_x", {"website": "inaccurate"}, gate_id="gate-0", gate_round=1),
        _adverse_gate(
            "m_x",
            {"website": "cannot-verify", "external_metadata.era": "cannot-verify"},
            gate_id="gate-1",
            gate_round=2,
        ),
    ]
    chargeable = chargeable_metadata_rejections(_history(gates, "m_x"))
    assert [gate["gate_id"] for gate, _item in chargeable] == ["gate-0", "gate-1"]


def test_rejection_after_an_accepted_round_is_a_fresh_disclosure() -> None:
    """An accepted round disclosed nothing, so a following rejection cannot charge."""

    gates = [
        _adverse_gate("m_x", {}, gate_id="gate-0", gate_round=1),
        _adverse_gate("m_x", {"website": "cannot-verify"}, gate_id="gate-1", gate_round=2),
    ]
    assert chargeable_metadata_rejections(_history(gates, "m_x")) == ()


@pytest.mark.parametrize(
    ("stable_id", "rounds"),
    [("m10517", _M10517_ROUNDS), ("m7362", _M7362_ROUNDS)],
)
def test_rung9b_frozen_pair_reaches_a_decided_verdict_within_cap(
    stable_id: str, rounds: tuple[dict[str, str], ...]
) -> None:
    """The frozen rung-9b trajectories no longer exhaust the cap at round 3.

    Replay of the real ledger shapes: three rejected rounds whose complaint
    sets are pairwise disjoint charge exactly ONE repair allowance, so after
    the frozen round 3 the model routes back to bounded repair (the paradigm
    complaint finally gets the repair round the checker owed it) and a
    subsequent accepted round decides the verdict -- instead of the historical
    ``cannot-verify-cap-exhausted`` human-review terminal.
    """

    gates = [
        _adverse_gate(stable_id, adverse, gate_id=f"gate-{index}", gate_round=index + 1)
        for index, adverse in enumerate(rounds)
    ]
    lineage = _lineage(stable_id)

    # The historical defect: the raw count (3 > max_repairs=2) terminalized
    # here. Under disclosure-honest accounting the lineage is NOT terminal.
    assert _terminal_metadata_gate(gates, stable_id, lineage, max_repairs=2) is None
    assert _chargeable_metadata_repair_count(gates, stable_id, lineage) == 1
    # Raw generations stay monotone: they number repair artifacts and report
    # rounds actually consumed.
    assert _metadata_repair_count(gates, stable_id, lineage) == 3

    # Routing the frozen round 3 the way the driver now does: chargeable prior
    # counts plus the round-2 disclosure set. The model requeues for repair
    # within cap; it is not handed to human review.
    prior = gates[:2]
    decision = route_metadata_gate(
        gates[2],
        {stable_id: _chargeable_metadata_repair_count(prior, stable_id, lineage)},
        max_repairs=2,
        prior_complaint_fields={
            stable_id: _latest_metadata_complaint_fields(prior, stable_id, lineage)
        },
    )[0]
    assert decision.route is GateRoute.REQUEUE_NEXT_BATCH
    assert decision.human_review_required is False
    assert decision.repair_count == 1

    # The repair round the checker owed lands and round 4 accepts: a decided
    # verdict, with one chargeable allowance still unspent.
    accepted = _adverse_gate(stable_id, {}, gate_id="gate-3", gate_round=4)
    final = route_metadata_gate(
        accepted,
        {stable_id: _chargeable_metadata_repair_count(gates, stable_id, lineage)},
        max_repairs=2,
        prior_complaint_fields={
            stable_id: _latest_metadata_complaint_fields(gates, stable_id, lineage)
        },
    )[0]
    assert final.route is GateRoute.ACCEPT
    assert final.canonical_write_allowed is True


def test_incorrigible_lineages_still_terminalize_loudly() -> None:
    """Neither tripwire is weakened: repeats and re-raised complaints still cap.

    An exact repeat of an earlier rejected round fast-terminals exactly as
    before, and a lineage that keeps re-raising disclosed complaints exhausts
    the cap at three chargeable rejections.
    """

    repeat = [
        _adverse_gate("m_x", {"website": "inaccurate"}, gate_id="gate-0", gate_round=1),
        _adverse_gate("m_x", {"website": "inaccurate"}, gate_id="gate-1", gate_round=2),
    ]
    terminal = _terminal_metadata_gate(repeat, "m_x", _lineage("m_x"), max_repairs=2)
    assert terminal is not None and terminal["gate_id"] == "gate-1"

    chain = [
        _adverse_gate("m_y", {"website": "cannot-verify"}, gate_id="gate-0", gate_round=1),
        _adverse_gate(
            "m_y",
            {"website": "cannot-verify", "external_metadata.era": "cannot-verify"},
            gate_id="gate-1",
            gate_round=2,
        ),
        _adverse_gate(
            "m_y",
            {
                "external_metadata.era": "cannot-verify",
                "external_metadata.tags": "cannot-verify",
            },
            gate_id="gate-2",
            gate_round=3,
        ),
    ]
    assert len(chargeable_metadata_rejections(_history(chain, "m_y"))) == 3
    terminal = _terminal_metadata_gate(chain, "m_y", _lineage("m_y"), max_repairs=2)
    assert terminal is not None and terminal["gate_id"] == "gate-2"


def test_route_without_disclosure_info_keeps_the_raw_bound() -> None:
    """A caller supplying no disclosure history gets the pre-rule bound."""

    gate = _adverse_gate("m_x", {"website": "cannot-verify"}, gate_id="gate-2")
    capped = route_metadata_gate(gate, {"m_x": 2}, max_repairs=2)[0]
    assert capped.route is GateRoute.HUMAN_FAIL

    # With the disclosure history saying the prior round complained about a
    # DISJOINT set, the same failure is a first disclosure: it requeues and
    # leaves the charged count untouched.
    excused = route_metadata_gate(
        gate,
        {"m_x": 2},
        max_repairs=2,
        prior_complaint_fields={"m_x": frozenset({"external_metadata.year"})},
    )[0]
    assert excused.route is GateRoute.REQUEUE_NEXT_BATCH
    assert excused.repair_count == 2

    # No prior round known for the model: chargeable, exactly like round one.
    fresh = route_metadata_gate(
        gate,
        {"m_x": 2},
        max_repairs=2,
        prior_complaint_fields={},
    )[0]
    assert fresh.route is GateRoute.HUMAN_FAIL


def _validated_batch(
    tmp_path: Any, mutate: Optional[Any] = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one stamped calibrated-size metadata batch and its envelope."""

    # The envelope builder enforces the calibrated METADATA_BATCH_MIN..MAX
    # cardinality, so the batch must match the current calibration exactly.
    gate = make_gate([f"m_{index}" for index in range(METADATA_BATCH_MAX)])
    if mutate is not None:
        mutate(gate)
    items = [_checker_item_pack(item) for item in gate["items"]]
    envelope = build_metadata_vet_envelope(
        items,
        gate_round=1,
        output_path=tmp_path / "result.json",
        checker_model="codex",
        checker_version="test",
        request_nonce="disclosure-1",
    )
    return _stamped(gate, envelope), envelope


def _make_item_adverse(item: dict[str, Any], field: str) -> None:
    """Flip one claim row adverse and keep the item's decision consistent."""

    for check in item["field_checks"]:
        if check["field"] == field:
            check["verdict"] = "cannot-verify"
            check["required_repair"] = f"repair {field}"
    item["verdict"] = "cannot-verify"
    item["required_repairs"] = [f"repair {field}"]


def test_complete_adverse_item_publishes(tmp_path: Any) -> None:
    """Full-coverage adverse items remain publishable adverse verdicts."""

    def mutate(gate: dict[str, Any]) -> None:
        _make_item_adverse(gate["items"][0], "website")

    stamped, envelope = _validated_batch(tmp_path, mutate)
    validated = validate_checker_result_mapping(stamped, envelope)
    assert validated["items"][0]["verdict"] == "cannot-verify"


def test_adverse_item_with_unchecked_claim_is_refused_at_dispatch(
    tmp_path: Any,
) -> None:
    """An adverse item silent on part of the inventory never publishes."""

    def mutate(gate: dict[str, Any]) -> None:
        item = gate["items"][0]
        _make_item_adverse(item, "website")
        item["field_checks"] = [
            check
            for check in item["field_checks"]
            if check["field"] != "external_metadata.paradigm"
        ]

    stamped, envelope = _validated_batch(tmp_path, mutate)
    with pytest.raises(
        CheckerDispatchError,
        match=r"complaint set is incomplete.*external_metadata\.paradigm",
    ):
        validate_checker_result_mapping(stamped, envelope)


def test_adverse_item_with_duplicate_claim_is_refused_at_dispatch(
    tmp_path: Any,
) -> None:
    """Duplicated rows cannot stand in for real coverage."""

    def mutate(gate: dict[str, Any]) -> None:
        item = gate["items"][0]
        _make_item_adverse(item, "website")
        item["field_checks"].append(deepcopy(item["field_checks"][0]))

    stamped, envelope = _validated_batch(tmp_path, mutate)
    with pytest.raises(CheckerDispatchError, match="repeats field checks"):
        validate_checker_result_mapping(stamped, envelope)


def test_adverse_item_with_extraneous_claim_is_refused_at_dispatch(
    tmp_path: Any,
) -> None:
    """Section names and invented claims are refused on adverse items too."""

    def mutate(gate: dict[str, Any]) -> None:
        item = gate["items"][0]
        _make_item_adverse(item, "website")
        for check in item["field_checks"]:
            if check["field"] == "external_metadata.year":
                check["field"] = "identity"

    stamped, envelope = _validated_batch(tmp_path, mutate)
    with pytest.raises(CheckerDispatchError, match="outside the machine-derived claim inventory"):
        validate_checker_result_mapping(stamped, envelope)


def test_authority_mirror_matches_checker_dispatch_rule() -> None:
    """The sealed-closure mirror in ``authority`` stays in lockstep.

    ``authority`` may not import ``checker_dispatch`` (the worker import
    closure walks every import statement), so it carries a duplicated
    implementation -- this test is the drift tripwire binding the two.
    """

    histories = [
        [
            _adverse_gate("m_x", adverse, gate_id=f"gate-{index}", gate_round=index + 1)
            for index, adverse in enumerate(rounds)
        ]
        for rounds in (
            _M10517_ROUNDS,
            _M7362_ROUNDS,
            ({"website": "inaccurate"}, {"website": "inaccurate"}),
            ({}, {"website": "cannot-verify"}),
        )
    ]
    for gates in histories:
        history = _history(gates, "m_x")
        assert _chargeable_metadata_rejection_count(history) == len(
            chargeable_metadata_rejections(history)
        )
        for _gate, item in history:
            assert authority_complaint_fields(item) == metadata_complaint_fields(item)


def test_authority_refuses_a_cap_claim_built_from_first_disclosures() -> None:
    """The proof kernel rejects the rung-9b shape as an unproven cap claim."""

    gates = [
        _adverse_gate("m10517", adverse, gate_id=f"gate-{index}", gate_round=index + 1)
        for index, adverse in enumerate(_M10517_ROUNDS)
    ]
    with pytest.raises(AuthorityDerivationError, match="bounded cap"):
        _derive_gate_failure(
            gates,
            stable_id="m10517",
            work_id="work-m10517",
            stage="accuracy-gate",
        )


def test_authority_still_proves_chargeable_cap_exhaustion() -> None:
    """Three chargeable rejections still prove ``cannot-verify-cap-exhausted``."""

    gates = [
        _adverse_gate("m_y", {"website": "cannot-verify"}, gate_id="gate-0", gate_round=1),
        _adverse_gate(
            "m_y",
            {"website": "cannot-verify", "external_metadata.era": "cannot-verify"},
            gate_id="gate-1",
            gate_round=2,
        ),
        _adverse_gate(
            "m_y",
            {
                "external_metadata.era": "cannot-verify",
                "external_metadata.tags": "cannot-verify",
            },
            gate_id="gate-2",
            gate_round=3,
        ),
    ]
    gate_id, reason, fingerprint = _derive_gate_failure(
        gates,
        stable_id="m_y",
        work_id="work-m_y",
        stage="accuracy-gate",
    )
    assert gate_id == "gate-2"
    assert reason == "cannot-verify-cap-exhausted"
    assert fingerprint
