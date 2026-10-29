"""The complaint-disclosure repair accounting applies to the fidelity lane.

The checker prompt has promised, for EVERY ``gate_round``, that "a brand-new
complaint on a claim a prior round blessed is accounted as checker
under-disclosure, never as author failure" and grants a fresh repair round.
Only the metadata lane implemented that accounting (wave-3, rung-9b docket
#4); the fidelity lane still charged raw rejections. Pilot m1359's terminal
fidelity round raised a brand-new ``input_contract`` complaint after rounds 1
and 2 had blessed the identical text ``accurate`` -- late disclosure the raw
count spent the author's final round on.

These tests pin the fidelity twin of the metadata rule:

- fidelity complaint keys add the five-way ``fidelity`` component to the
  shared field-check/integrity/rung keys, so a pure-drift rejection is never
  an empty (cost-free) complaint set;
- a rejection made only of first disclosures does not consume the cap, while
  a re-raised complaint charges whatever its verdict class;
- m1359's ACTUAL trajectory (``source_resolution.rung`` re-raised in every
  round) still terminalizes at three rounds -- no verdict or bound is
  weakened;
- the repeated-root-cause fast terminal is unchanged;
- the sealed-closure authority mirror stays in lockstep and refuses a
  fidelity cap claim built from first disclosures.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    _chargeable_fidelity_rejection_count,
    _derive_gate_failure,
    _fidelity_complaint_fields as authority_fidelity_complaint_fields,
)
from menagerie.crawler.checker_dispatch import (
    chargeable_fidelity_rejections,
    fidelity_complaint_fields,
    fidelity_item_rejected,
    fidelity_rejection_chargeable,
)
from menagerie.crawler.driver_models import _terminal_fidelity_gate
from menagerie.crawler.tests.conftest import make_gate

_STABLE_ID = "m1359"


def _fidelity_gate(
    adverse_fields: Mapping[str, str],
    *,
    gate_id: str,
    gate_round: int = 1,
    fidelity_verdict: str = "match",
    rung_verdict: str = "accurate",
) -> dict[str, Any]:
    """Build one complete fidelity gate with the named adverse components.

    Parameters
    ----------
    adverse_fields:
        Claim path to non-accurate verdict for the item's field checks.
    gate_id:
        Distinct immutable gate identity.
    gate_round:
        One-based checker round for deterministic authority ordering.
    fidelity_verdict:
        Five-way fidelity verdict.
    rung_verdict:
        Rung-check verdict.

    Returns
    -------
    dict[str, Any]
        Complete one-item fidelity gate.
    """

    gate = make_gate([_STABLE_ID], gate_id=gate_id, gate_kind="fidelity")
    gate["gate_round"] = gate_round
    item = gate["items"][0]
    item["fidelity"]["verdict"] = fidelity_verdict
    item["rung_check"]["verdict"] = rung_verdict
    if rung_verdict == "accurate":
        item["rung_check"]["highest_applicable"] = item["rung_check"]["selected_rung"]
    for check in item["field_checks"]:
        verdict = adverse_fields.get(str(check["field"]))
        if verdict is not None:
            check["verdict"] = verdict
            check["required_repair"] = f"repair {check['field']}"
    adverse = (
        bool(adverse_fields)
        or fidelity_verdict
        not in {
            "match",
            "minor-drift",
        }
        or rung_verdict != "accurate"
    )
    if adverse:
        item["verdict"] = "cannot-verify"
        item["required_repairs"] = sorted(
            {f"repair {field}" for field in adverse_fields} | {"repair the gated defect"}
        )
    return gate


def _history(
    gates: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], Mapping[str, Any]], ...]:
    """Pair each gate with its single fixture item, in ledger order."""

    return tuple((gate, gate["items"][0]) for gate in gates)


#: m1359's ACTUAL pilot trajectory: the unsatisfiable routed-environment rung
#: complaint re-raised in every round (plus round 3's brand-new
#: ``input_contract`` late disclosure). Every round re-answers a disclosed
#: complaint, so all three charge and the terminal stands.
def _m1359_actual() -> list[dict[str, Any]]:
    return [
        _fidelity_gate(
            {"source_resolution.rung": "cannot-verify"},
            gate_id="gate-0",
            gate_round=1,
            fidelity_verdict="cannot-verify",
            rung_verdict="cannot-verify",
        ),
        _fidelity_gate(
            {"source_resolution.rung": "cannot-verify"},
            gate_id="gate-1",
            gate_round=2,
            fidelity_verdict="cannot-verify",
            rung_verdict="cannot-verify",
        ),
        _fidelity_gate(
            {
                "source_resolution.rung": "cannot-verify",
                "input_contract": "cannot-verify",
            },
            gate_id="gate-2",
            gate_round=3,
            fidelity_verdict="minor-drift",
            rung_verdict="cannot-verify",
        ),
    ]


#: A late-disclosure lineage: round 1 rejects on the rung check alone (the
#: five-way fidelity verdict is blessed ``match``), the author repairs it, and
#: round 2 first discloses an implementation-drift complaint on the very
#: component round 1 blessed. Round 2 answers nothing round 1 asked, so it is
#: checker disclosure debt; round 3 re-raises round 2's drift complaint and
#: charges normally.
def _late_disclosure_lineage() -> list[dict[str, Any]]:
    return [
        _fidelity_gate(
            {"source_resolution.rung": "cannot-verify"},
            gate_id="gate-0",
            gate_round=1,
            rung_verdict="cannot-verify",
        ),
        _fidelity_gate(
            {},
            gate_id="gate-1",
            gate_round=2,
            fidelity_verdict="major-drift",
        ),
        _fidelity_gate(
            {"input_contract": "cannot-verify"},
            gate_id="gate-2",
            gate_round=3,
            fidelity_verdict="cannot-verify",
        ),
    ]


def test_fidelity_complaint_fields_add_the_fidelity_component() -> None:
    """A pure-drift rejection discloses ``fidelity``, never an empty set."""

    accepted = _fidelity_gate({}, gate_id="gate-a")["items"][0]
    assert fidelity_complaint_fields(accepted) == frozenset()
    assert not fidelity_item_rejected(accepted)

    drift = _fidelity_gate({}, gate_id="gate-b", fidelity_verdict="major-drift")["items"][0]
    assert fidelity_complaint_fields(drift) == frozenset({"fidelity"})
    assert fidelity_item_rejected(drift)

    mixed = _fidelity_gate(
        {"website": "cannot-verify"},
        gate_id="gate-c",
        fidelity_verdict="cannot-verify",
        rung_verdict="cannot-verify",
    )["items"][0]
    assert fidelity_complaint_fields(mixed) == frozenset({"website", "fidelity", "rung_check"})


def test_first_round_rejection_always_charges() -> None:
    """A lineage's first rejection answers nothing and is always chargeable."""

    item = _fidelity_gate({}, gate_id="gate-a", fidelity_verdict="cannot-verify")["items"][0]
    assert fidelity_rejection_chargeable(item, None) is True


def test_pure_drift_rejections_re_raise_the_fidelity_component() -> None:
    """Two drift rounds are the SAME complaint re-raised, not free disclosures."""

    gates = [
        _fidelity_gate({}, gate_id="gate-0", gate_round=1, fidelity_verdict="major-drift"),
        _fidelity_gate({}, gate_id="gate-1", gate_round=2, fidelity_verdict="cannot-verify"),
    ]
    charged = chargeable_fidelity_rejections(_history(gates))
    assert [gate["gate_id"] for gate, _item in charged] == ["gate-0", "gate-1"]


def test_late_disclosures_do_not_consume_the_fidelity_cap() -> None:
    """A round made only of first disclosures grants a fresh repair round."""

    gates = _late_disclosure_lineage()
    charged = chargeable_fidelity_rejections(_history(gates))
    # Round 2 first discloses drift on the component round 1 blessed: not
    # chargeable. Round 3 re-raises round 2's drift complaint: chargeable.
    assert [gate["gate_id"] for gate, _item in charged] == ["gate-0", "gate-2"]
    # The historical rule terminalized here (3 raw rejections > max_repairs=2).
    # Under disclosure-honest accounting the model keeps the repair round the
    # checker's own late disclosure owes it.
    assert _terminal_fidelity_gate(gates, _STABLE_ID, f"work-{_STABLE_ID}", max_repairs=2) is None


def test_m1359s_actual_trajectory_still_terminalizes() -> None:
    """Re-raised complaints charge every round: the pilot terminal stands."""

    gates = _m1359_actual()
    # The real rounds carried distinct repair prose (5 repairs, then 1, then
    # 2), so distinguish the fingerprints here to prove the terminal comes
    # from the chargeable cap, not the repeated-root-cause fast arm.
    for index, gate in enumerate(gates):
        gate["items"][0]["required_repairs"] = [
            f"bind the routed-environment inventory, round {index}"
        ]
    assert len(chargeable_fidelity_rejections(_history(gates))) == 3
    terminal = _terminal_fidelity_gate(gates, _STABLE_ID, f"work-{_STABLE_ID}", max_repairs=2)
    assert terminal is not None and terminal["gate_id"] == "gate-2"


def test_the_repeated_root_cause_fast_terminal_is_unchanged() -> None:
    """An exact repeat of an earlier rejected round still terminalizes at two."""

    first = _fidelity_gate(
        {"website": "cannot-verify"},
        gate_id="gate-0",
        gate_round=1,
        fidelity_verdict="major-drift",
    )
    second = deepcopy(first)
    second["gate_id"] = "gate-1"
    second["gate_round"] = 2
    terminal = _terminal_fidelity_gate(
        [first, second], _STABLE_ID, f"work-{_STABLE_ID}", max_repairs=2
    )
    assert terminal is not None and terminal["gate_id"] == "gate-1"


def test_authority_mirror_matches_checker_dispatch_rule() -> None:
    """The sealed-closure mirror in ``authority`` stays in lockstep."""

    for gates in (_m1359_actual(), _late_disclosure_lineage()):
        history = _history(gates)
        assert _chargeable_fidelity_rejection_count(history) == len(
            chargeable_fidelity_rejections(history)
        )
        for _gate, item in history:
            assert authority_fidelity_complaint_fields(item) == fidelity_complaint_fields(item)


def test_authority_refuses_a_fidelity_cap_claim_built_from_first_disclosures() -> None:
    """The proof kernel rejects the late-disclosure shape as an unproven cap."""

    with pytest.raises(AuthorityDerivationError, match="bounded cap"):
        _derive_gate_failure(
            _late_disclosure_lineage(),
            stable_id=_STABLE_ID,
            work_id=f"work-{_STABLE_ID}",
            stage="fidelity",
        )


def test_authority_still_proves_chargeable_fidelity_cap_exhaustion() -> None:
    """Three chargeable fidelity rejections still prove the capped terminal."""

    gates = [
        _fidelity_gate(
            {"source_resolution.rung": "cannot-verify"},
            gate_id=f"gate-{index}",
            gate_round=index + 1,
            fidelity_verdict="cannot-verify",
            rung_verdict="cannot-verify",
        )
        for index in range(3)
    ]
    # Distinct repair prose per round keeps the fingerprints distinct, so the
    # proof rides the chargeable-cap arm rather than the repeat fast terminal.
    for index, gate in enumerate(gates):
        gate["items"][0]["required_repairs"] = [f"repair the rung, take {index}"]
    gate_id, reason, fingerprint = _derive_gate_failure(
        gates,
        stable_id=_STABLE_ID,
        work_id=f"work-{_STABLE_ID}",
        stage="fidelity",
    )
    assert gate_id == "gate-2"
    assert reason == "cannot-verify-cap-exhausted"
    assert fingerprint
