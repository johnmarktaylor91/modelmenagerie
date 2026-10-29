"""ONE gate-verdict authority: the m1359 gate-semantics cluster (rung-12 census).

Two independent censuses (Sol + Fable) adjudicated the same design: a single
gate-choice authority with no per-lane carve-outs. These tests pin the two
defects that adjudication closes and the tripwires it must leave armed:

1. BLOCKER -- a correctly-rejected terminal disposition could not terminalize
   whenever ANY ``metadata_batch`` item existed for the same work. The repair
   lane produces exactly that history (rejected metadata rounds, then an
   author BLOCKED answer whose disposition the checker rejects), and the
   fidelity-repair lane produces the accepted-metadata variant; both ended in
   ``AuthorityDerivationError`` -> ``PartitionError`` with NO record written.
   The proof rule is now selected by which checker gate adjudicated LAST, so
   the rejected disposition proves its own single-adjudication obligation.

2. Latent inconsistency -- ``authority._derive_gate_failure`` and
   ``gates.route_fidelity_gate`` disagreed on the reason for a rejected
   fidelity item whose CONTENT passed (``match``/``minor-drift``) but whose
   rung check came back ``cannot-verify``: the authority read it as
   ``identity-mismatch`` (a refusal on the merits that never happened), the
   router as ``cannot-verify-cap-exhausted``. m1359 is the worked example:
   fidelity content passed 6/7 with one nonmaterial drift, and the two
   evidence gaps were rung ``cannot-verify`` findings. Both call sites now
   consult one shared kernel (``constants.fidelity_gate_failure_reason``).

The general proof rule stays armed: a model must never be marked gate-failed
without gate evidence, a cap claim still requires its cap proof, and neither
rule can borrow the other's evidence.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    _derive_gate_failure,
    derive_terminal_proof,
)
from menagerie.crawler.constants import (
    TERMINAL_DISPOSITION_REJECTED_REASON_CODE,
    FAILURE_REASON_CODES,
    fidelity_gate_failure_reason,
    metadata_gate_failure_reason,
)
from menagerie.crawler.gates import route_fidelity_gate
from menagerie.crawler.tests.conftest import HASH, make_author_proposal, make_gate

_STABLE_ID = "m1359"
_WORK_ID = f"work-{_STABLE_ID}"


def _metadata_gate(
    *,
    gate_id: str,
    gate_round: int,
    ledger_seq: int,
    verdict: str = "inaccurate",
    integrity_verdict: str = "accurate",
    rung_verdict: str = "accurate",
) -> dict[str, Any]:
    """Build one minimal canonical ``metadata_batch`` gate for the proof kernel."""

    return {
        "gate_id": gate_id,
        "gate_kind": "metadata_batch",
        "gate_round": gate_round,
        "ledger_seq": ledger_seq,
        "items": [
            {
                "stable_id": _STABLE_ID,
                "work_id": _WORK_ID,
                "campaign_root_work_id": _WORK_ID,
                "verdict": verdict,
                "integrity": {"verdict": integrity_verdict},
                "rung_check": {"verdict": rung_verdict},
                "required_repairs": [f"repair round {gate_round}"],
            }
        ],
    }


def _terminal_gate(
    *,
    gate_id: str = "gate-terminal",
    gate_round: int,
    ledger_seq: int,
    verdict: str = "rejected",
    findings: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build one minimal canonical ``terminal_disposition`` gate."""

    return {
        "gate_id": gate_id,
        "gate_kind": "terminal_disposition",
        "gate_round": gate_round,
        "ledger_seq": ledger_seq,
        "items": [
            {
                "stable_id": _STABLE_ID,
                "work_id": _WORK_ID,
                "campaign_root_work_id": _WORK_ID,
                "verdict": "inaccurate",
                "integrity": {"verdict": "accurate"},
                "rung_check": {"selected_rung": "R5_SKIP", "verdict": "accurate"},
                "terminal_disposition": {
                    "kind": "BLOCKED",
                    "predicate": "blocked-prerequisite",
                    "verdict": verdict,
                    "findings": (
                        ["the resolved excerpts do not entail the typed predicate"]
                        if findings is None
                        else findings
                    ),
                    "source_ids": ["source-1"],
                    "evidence_ids": ["evidence-1"],
                },
            }
        ],
    }


def _fidelity_round(
    *,
    gate_id: str,
    gate_round: int,
    fidelity_verdict: str,
    rung_verdict: str,
) -> dict[str, Any]:
    """Build one schema-valid rejected fidelity round with a distinct repair ask."""

    gate = make_gate([_STABLE_ID], gate_id=gate_id, gate_kind="fidelity", fidelity_identity=HASH)
    gate["gate_round"] = gate_round
    item = gate["items"][0]
    item["fidelity"]["verdict"] = fidelity_verdict
    item["rung_check"]["verdict"] = rung_verdict
    item["verdict"] = "cannot-verify"
    # Distinct repair prose keeps fingerprints distinct, so the terminal proof
    # rides the chargeable-cap arm rather than the repeated-root-cause arm.
    item["required_repairs"] = [f"bind the routed-environment inventory, round {gate_round}"]
    return gate


def _accuracy_proof(gates: list[dict[str, Any]]) -> Any:
    """Derive the ``failed:accuracy-gate`` terminal proof over one gate history."""

    return derive_terminal_proof(
        _STABLE_ID,
        _WORK_ID,
        "failed:accuracy-gate",
        attempts=[],
        gates=gates,
        proof_rule_identity=HASH,
    )


@pytest.mark.smoke
def test_rejected_disposition_after_metadata_repair_lineage_terminalizes() -> None:
    """The width-1 blocker: a repair lineage plus a later rejected disposition.

    Production shape: one rejected metadata round (below the bounded cap), the
    repair author answers BLOCKED, the terminal checker rejects the claim, the
    driver stamps ``failed:accuracy-gate``. Before unification the proof kernel
    only consulted the disposition when NO ``metadata_batch`` item existed, so
    this history raised (``has not reached its bounded cap``) and the campaign
    stopped with no record written.
    """

    proof = _accuracy_proof(
        [
            _metadata_gate(gate_id="gate-md-1", gate_round=1, ledger_seq=1),
            _terminal_gate(gate_round=2, ledger_seq=2),
        ]
    )
    assert proof.status_code == "failed:accuracy-gate"
    assert proof.gate_id == "gate-terminal"
    assert proof.reason_code == TERMINAL_DISPOSITION_REJECTED_REASON_CODE
    # A single adjudication never smuggles in a repair-cap claim.
    assert "cap-exhausted" not in proof.reason_code


@pytest.mark.smoke
def test_rejected_disposition_after_accepted_metadata_terminalizes() -> None:
    """The fidelity-lane variant: accepted metadata, then a rejected disposition.

    Production shape: the metadata gate accepted, the fidelity repair loop asked
    the author again, and the author answered BLOCKED; the checker rejected that
    claim. Before unification this history raised the census's exact message
    (``lacks exact rejected gate evidence``) because the accepted metadata item
    blocked the disposition arm while proving nothing itself.
    """

    proof = _accuracy_proof(
        [
            _metadata_gate(gate_id="gate-md-1", gate_round=1, ledger_seq=1, verdict="accurate"),
            _terminal_gate(gate_round=2, ledger_seq=2),
        ]
    )
    assert proof.gate_id == "gate-terminal"
    assert proof.reason_code == TERMINAL_DISPOSITION_REJECTED_REASON_CODE


@pytest.mark.smoke
def test_later_metadata_lineage_still_governs_and_still_refuses_below_cap() -> None:
    """A repair lineage POSTDATING the disposition keeps the bounded-cap rule.

    The proof rule is selected by the LAST adjudication, so a stale rejected
    disposition cannot be borrowed to terminalize a live repair lineage that
    has not reached its cap -- the tripwire this cluster must not disarm.
    """

    with pytest.raises(AuthorityDerivationError, match="bounded cap"):
        _accuracy_proof(
            [
                _terminal_gate(gate_round=1, ledger_seq=1),
                _metadata_gate(gate_id="gate-md-1", gate_round=2, ledger_seq=2),
            ]
        )


@pytest.mark.smoke
def test_accepted_disposition_still_proves_no_gate_failure() -> None:
    """An ACCEPTED last adjudication proves no failure, whatever sits before it."""

    with pytest.raises(AuthorityDerivationError, match="adjudicated terminal disposition"):
        _accuracy_proof(
            [
                _metadata_gate(gate_id="gate-md-1", gate_round=1, ledger_seq=1),
                _terminal_gate(gate_round=2, ledger_seq=2, verdict="accepted"),
            ]
        )


@pytest.mark.smoke
def test_evidence_free_gate_failure_stays_refused() -> None:
    """No gate evidence, no gate-failure terminal: the original refusal stands."""

    with pytest.raises(AuthorityDerivationError, match="lacks exact rejected gate evidence"):
        _accuracy_proof([])
    # A findings-free refusal is an unevidenced refusal even when it is the
    # last adjudication and a repair lineage sits beside it.
    with pytest.raises(AuthorityDerivationError, match="adjudicated terminal disposition"):
        _accuracy_proof(
            [
                _metadata_gate(gate_id="gate-md-1", gate_round=1, ledger_seq=1),
                _terminal_gate(gate_round=2, ledger_seq=2, findings=[]),
            ]
        )


def _m1359_terminal_history() -> list[dict[str, Any]]:
    """m1359's shape: content passes on the terminal round, the rung cannot verify."""

    return [
        _fidelity_round(
            gate_id="gate-0",
            gate_round=1,
            fidelity_verdict="cannot-verify",
            rung_verdict="cannot-verify",
        ),
        _fidelity_round(
            gate_id="gate-1",
            gate_round=2,
            fidelity_verdict="cannot-verify",
            rung_verdict="cannot-verify",
        ),
        _fidelity_round(
            gate_id="gate-2",
            gate_round=3,
            fidelity_verdict="minor-drift",
            rung_verdict="cannot-verify",
        ),
    ]


@pytest.mark.smoke
def test_fidelity_rung_cannot_verify_reason_is_unified_across_call_sites() -> None:
    """The m1359 worked example: both call sites derive the same honest reason.

    Fidelity CONTENT passed on the terminal round (``minor-drift``); the
    rejection is the rung check's ``cannot-verify`` (evidence gaps). That is a
    could-not-determine outcome, never a refusal on the merits, so BOTH the
    routing decision and the terminal proof must say
    ``cannot-verify-cap-exhausted`` -- the authority previously said
    ``identity-mismatch`` and the two disagreed on the same gate item.
    """

    history = _m1359_terminal_history()
    gate_id, reason, fingerprint = _derive_gate_failure(
        history,
        stable_id=_STABLE_ID,
        work_id=_WORK_ID,
        stage="fidelity",
    )
    routed = route_fidelity_gate(history[-1], make_author_proposal(_STABLE_ID))
    assert gate_id == "gate-2"
    assert fingerprint
    assert routed.accepted_for_fidelity is False
    assert reason == routed.failure_reason_code == "cannot-verify-cap-exhausted"
    assert reason in FAILURE_REASON_CODES["fidelity"]


@pytest.mark.smoke
def test_fidelity_rung_inaccurate_reason_stays_identity_mismatch() -> None:
    """A rung-inaccurate finding over matching content is an identity dispute."""

    history = [
        _fidelity_round(
            gate_id=f"gate-{index}",
            gate_round=index + 1,
            fidelity_verdict="match",
            rung_verdict="inaccurate",
        )
        for index in range(3)
    ]
    _gate_id, reason, _fingerprint = _derive_gate_failure(
        history,
        stable_id=_STABLE_ID,
        work_id=_WORK_ID,
        stage="fidelity",
    )
    routed = route_fidelity_gate(history[-1], make_author_proposal(_STABLE_ID))
    assert reason == routed.failure_reason_code == "identity-mismatch"


@pytest.mark.smoke
def test_shared_reason_kernels_are_total() -> None:
    """One odd verdict string on one model must never take the campaign down.

    Unreadable inputs degrade to the WEAKER could-not-determine claim (the
    ``terminal_disposition_failure_reason`` doctrine), and every output stays
    inside its stage's closed reason vocabulary.
    """

    for item in (
        {},
        {"fidelity": None, "rung_check": None},
        {"fidelity": {"verdict": "wat"}, "rung_check": {"verdict": 7}},
        {"fidelity": {"verdict": "match"}, "rung_check": {"verdict": "wat"}},
    ):
        assert fidelity_gate_failure_reason(item) == "cannot-verify-cap-exhausted"
    assert (
        fidelity_gate_failure_reason(
            {"fidelity": {"verdict": "match"}, "rung_check": {"verdict": "inaccurate"}}
        )
        == "identity-mismatch"
    )
    for verdict, expected in (
        ("major-drift", "major-drift-cap-exhausted"),
        ("slop", "slop-cap-exhausted"),
        ("cannot-verify", "cannot-verify-cap-exhausted"),
    ):
        reason = fidelity_gate_failure_reason({"fidelity": {"verdict": verdict}})
        assert reason == expected
        assert reason in FAILURE_REASON_CODES["fidelity"]

    assert metadata_gate_failure_reason({"verdict": "cannot-verify"}) == (
        "cannot-verify-cap-exhausted"
    )
    assert (
        metadata_gate_failure_reason(
            {"verdict": "inaccurate", "integrity": {"verdict": "inaccurate"}}
        )
        == "identity-mismatch"
    )
    assert (
        metadata_gate_failure_reason(
            {"verdict": "inaccurate", "integrity": {"verdict": "accurate"}}
        )
        == "inaccurate-cap-exhausted"
    )
    for item in ({}, {"verdict": "inaccurate", "integrity": None}):
        assert metadata_gate_failure_reason(item) == "identity-mismatch"
