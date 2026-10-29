"""Gate-failure kernel recency: the LAST adjudication in ledger order governs.

The unified gate-verdict authority selects WHICH proof rule governs by the last
adjudication in per-work ledger order, but each rule then read its own kind's
history without guarding recency inside it. Three wrong-verdict shapes follow,
each reproduced against the pure kernels before this suite existed:

1. ``_derive_terminal_disposition_failure`` filtered accepted dispositions out
   and returned the last REMAINING one, so a stale rejected disposition proved a
   ``failed:accuracy-gate`` terminal even when the checker's actual last word
   ACCEPTED the same work (crash-resume re-dispatch is the production shape:
   rejected gate appended, leg dies, resumed leg re-dispatches, checker accepts).
2. The bounded-cap rule counted rejected rounds without ever asking whether the
   FINAL round of the lineage accepted, so a lineage that ended in acceptance
   still terminalized as cap-exhausted. Pilot m10949 carries the real mixed
   history: ledger sequences 3/35/36 adjudicated accurate/inaccurate/accurate
   for one work.
3. ``_terminal_gate`` collected acceptances and returned ``accepted[-1]``,
   ignoring a LATER rejection of the same claim, so a superseded acceptance
   still proved a deferral or skip.

All three must now REFUSE: a verdict may only be derived from the checker's
last word, and a claim that contradicts it is the unevidenced claim the
terminal-proof tripwire exists to catch. The same recency doctrine is why the
metadata reason kernel must carry the rung-check arms its fidelity sibling has
(a content-accepted item rejected by its rung check alone is an identity
dispute or a could-not-determine outcome, never a merits cap), and why the
driver's ``max_repairs`` 2 and the kernel's chargeable-rejection 3 must derive
from ONE linked constant instead of two unrelated literals.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import pytest

import menagerie.crawler.authority as authority
from menagerie.crawler.authority import (
    AuthorityDerivationError,
    derive_terminal_proof,
)
from menagerie.crawler.constants import (
    FAILURE_REASON_CODES,
    fidelity_gate_failure_reason,
    metadata_gate_failure_reason,
)

_STABLE_ID = "m10949"
_WORK_ID = f"work-{_STABLE_ID}"
_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64


def _metadata_gate(
    *,
    gate_id: str,
    gate_round: int,
    ledger_seq: int,
    verdict: str = "inaccurate",
    field_checks: Optional[list[dict[str, Any]]] = None,
    repairs: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build one minimal canonical ``metadata_batch`` gate for the proof kernel."""

    item: dict[str, Any] = {
        "stable_id": _STABLE_ID,
        "work_id": _WORK_ID,
        "campaign_root_work_id": _WORK_ID,
        "verdict": verdict,
        "integrity": {"verdict": "accurate"},
        "rung_check": {"verdict": "accurate"},
    }
    if field_checks is not None:
        item["field_checks"] = field_checks
    if repairs is not None:
        item["required_repairs"] = repairs
    return {
        "gate_id": gate_id,
        "gate_kind": "metadata_batch",
        "gate_round": gate_round,
        "ledger_seq": ledger_seq,
        "items": [item],
    }


def _fidelity_gate(
    *,
    gate_id: str,
    gate_round: int,
    ledger_seq: int,
    fidelity_verdict: str,
    rung_verdict: str = "accurate",
) -> dict[str, Any]:
    """Build one minimal canonical ``fidelity`` gate for the proof kernel."""

    accepted = fidelity_verdict in {"match", "minor-drift"} and rung_verdict == "accurate"
    return {
        "gate_id": gate_id,
        "gate_kind": "fidelity",
        "gate_round": gate_round,
        "ledger_seq": ledger_seq,
        "items": [
            {
                "stable_id": _STABLE_ID,
                "work_id": _WORK_ID,
                "campaign_root_work_id": _WORK_ID,
                "verdict": "accurate" if accepted else "cannot-verify",
                "integrity": {"verdict": "accurate"},
                "rung_check": {"verdict": rung_verdict},
                "fidelity": {"verdict": fidelity_verdict},
                "required_repairs": [] if accepted else ["rebind the drifted module stack"],
            }
        ],
    }


def _terminal_gate(
    *,
    gate_id: str,
    gate_round: int,
    ledger_seq: int,
    verdict: str = "rejected",
    predicate: str = "blocked-prerequisite",
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
                "verdict": "accurate" if verdict == "accepted" else "inaccurate",
                "integrity": {"verdict": "accurate"},
                "rung_check": {"selected_rung": "R5_SKIP", "verdict": "accurate"},
                "terminal_disposition": {
                    "kind": "BLOCKED",
                    "predicate": predicate,
                    "verdict": verdict,
                    "findings": ["the declared excerpts do not entail the typed predicate"],
                    "source_ids": ["source-1"],
                    "evidence_ids": ["evidence-1"],
                    "source_manifest_identity": _HASH_A,
                    "evidence_identity": _HASH_B,
                    "license_identity": _HASH_C,
                },
            }
        ],
    }


def _accuracy_proof(gates: list[dict[str, Any]]) -> Any:
    """Derive the ``failed:accuracy-gate`` terminal proof over one gate history."""

    return derive_terminal_proof(
        _STABLE_ID,
        _WORK_ID,
        "failed:accuracy-gate",
        attempts=[],
        gates=gates,
        proof_rule_identity=_HASH_A,
    )


@pytest.mark.smoke
def test_stale_rejected_disposition_never_beats_a_later_acceptance() -> None:
    """S2 fixture 1: [terminal rejected, terminal accepted] + failure claim REFUSES.

    The checker's last word on this work is an ACCEPTANCE; deriving a gate
    failure from the earlier rejection is exactly the wrong-recency verdict the
    kernel exists to refuse. Filtering accepted rounds out before choosing the
    last remaining adjudication proved the stale rejection instead.
    """

    with pytest.raises(AuthorityDerivationError, match="adjudicated terminal disposition"):
        _accuracy_proof(
            [
                _terminal_gate(gate_id="gate-t1", gate_round=1, ledger_seq=1),
                _terminal_gate(gate_id="gate-t2", gate_round=2, ledger_seq=2, verdict="accepted"),
            ]
        )


@pytest.mark.smoke
def test_unreadable_last_adjudication_never_yields_to_a_stale_rejection() -> None:
    """A last adjudication whose disposition block cannot be read fails CLOSED.

    Skipping unreadable rows hands the verdict to an earlier stale adjudication
    -- the same recency hole through the side door. The checker's last word
    being unreadable means the failure claim cannot be shown to agree with it.
    """

    unreadable = _terminal_gate(gate_id="gate-t2", gate_round=2, ledger_seq=2)
    unreadable["items"][0]["terminal_disposition"] = None
    with pytest.raises(AuthorityDerivationError, match="adjudicated terminal disposition"):
        _accuracy_proof(
            [
                _terminal_gate(gate_id="gate-t1", gate_round=1, ledger_seq=1),
                unreadable,
            ]
        )


@pytest.mark.smoke
def test_metadata_cap_proof_refuses_a_final_accepted_round() -> None:
    """S2 fixture 2: three rejected metadata rounds then an ACCEPTED fourth REFUSES.

    The cap arms counted rejections without ever asking whether the FINAL round
    of the lineage accepted, so this history terminalized as cap-exhausted over
    the checker's own later acceptance (the accepted-side mirror of the rule
    that a stale rejected disposition cannot terminalize a live repair lineage).
    """

    gates = [
        _metadata_gate(gate_id=f"gate-md-{index}", gate_round=index, ledger_seq=index)
        for index in range(1, 4)
    ]
    gates.append(
        _metadata_gate(gate_id="gate-md-4", gate_round=4, ledger_seq=4, verdict="accurate")
    )
    with pytest.raises(AuthorityDerivationError, match="contradicted by its final accepted"):
        _accuracy_proof(gates)


@pytest.mark.smoke
def test_m10949_mixed_verdict_history_refuses_for_the_honest_reason() -> None:
    """Pilot m10949's real shape (sequences 3/35/36) refuses AS a contradiction.

    accurate/inaccurate/accurate: the final round accepted, so the refusal must
    name the contradiction with the checker's last word, not read as a lineage
    that merely "has not reached its bounded cap" yet -- W4 reuse re-appends
    verdicts, and a cap-shortfall reading invites retrying a settled work.
    """

    gates = [
        _metadata_gate(gate_id="gate-md-a", gate_round=1, ledger_seq=3, verdict="accurate"),
        _metadata_gate(gate_id="gate-md-b", gate_round=2, ledger_seq=35),
        _metadata_gate(gate_id="gate-md-c", gate_round=3, ledger_seq=36, verdict="accurate"),
    ]
    with pytest.raises(AuthorityDerivationError, match="contradicted by its final accepted"):
        _accuracy_proof(gates)


@pytest.mark.smoke
def test_fidelity_cap_proof_refuses_a_final_accepted_round() -> None:
    """The fidelity mirror: a lineage ending in ``match`` + accurate rung REFUSES."""

    gates = [
        _fidelity_gate(
            gate_id=f"gate-fid-{index}",
            gate_round=index,
            ledger_seq=index,
            fidelity_verdict="major-drift",
        )
        for index in range(1, 4)
    ]
    gates.append(
        _fidelity_gate(gate_id="gate-fid-4", gate_round=4, ledger_seq=4, fidelity_verdict="match")
    )
    with pytest.raises(AuthorityDerivationError, match="contradicted by its final accepted"):
        derive_terminal_proof(
            _STABLE_ID,
            _WORK_ID,
            "failed:fidelity",
            attempts=[],
            gates=gates,
            proof_rule_identity=_HASH_A,
        )


def _deferral_proof(gates: list[dict[str, Any]]) -> Any:
    """Derive the ``deferred:needs-cuda`` terminal proof over one gate history."""

    sources = [{"source_id": "source-1", "url": "https://example.test/source"}]
    evidence = [
        {
            "evidence_id": "evidence-1",
            "source_id": "source-1",
            "supports": ["platform.needs-cuda"],
        }
    ]
    return derive_terminal_proof(
        _STABLE_ID,
        _WORK_ID,
        "deferred:needs-cuda",
        attempts=[],
        gates=gates,
        source_manifest=sources,
        evidence_excerpts=evidence,
        source_manifest_identity=_HASH_A,
        evidence_identity=_HASH_B,
        license_identity=_HASH_C,
        proof_rule_identity=_HASH_A,
    )


@pytest.mark.smoke
def test_terminal_acceptance_superseded_by_a_later_rejection_refuses() -> None:
    """S2 fixture 3: [terminal accepted, terminal rejected] + platform claim REFUSES.

    ``_terminal_gate`` returned ``accepted[-1]`` without asking whether a later
    adjudication rejected the same claim, so a superseded acceptance still
    proved the deferral. The checker's last word is the rejection.
    """

    with pytest.raises(AuthorityDerivationError, match="superseded by a later"):
        _deferral_proof(
            [
                _terminal_gate(
                    gate_id="gate-t1",
                    gate_round=1,
                    ledger_seq=1,
                    verdict="accepted",
                    predicate="needs-cuda",
                ),
                _terminal_gate(
                    gate_id="gate-t2",
                    gate_round=2,
                    ledger_seq=2,
                    verdict="rejected",
                    predicate="needs-cuda",
                ),
            ]
        )


@pytest.mark.smoke
def test_late_acceptance_after_an_earlier_rejection_still_proves() -> None:
    """The recency rule cuts one way only: a LAST acceptance proves its claim."""

    proof = _deferral_proof(
        [
            _terminal_gate(
                gate_id="gate-t1",
                gate_round=1,
                ledger_seq=1,
                verdict="rejected",
                predicate="needs-cuda",
            ),
            _terminal_gate(
                gate_id="gate-t2",
                gate_round=2,
                ledger_seq=2,
                verdict="accepted",
                predicate="needs-cuda",
            ),
        ]
    )
    assert proof.platform_claim == "needs-cuda"
    assert proof.gate_id == "gate-t2"


@pytest.mark.smoke
def test_metadata_reason_kernel_mirrors_the_fidelity_rung_check_arms() -> None:
    """A content-accepted metadata item rejected by its rung alone is never a merits cap.

    The fidelity kernel distinguishes an identity dispute (rung ``inaccurate``)
    from a could-not-determine outcome (rung ``cannot-verify`` or unreadable) on
    content-accepted items; the metadata sibling collapsed both onto
    ``inaccurate-cap-exhausted``, a refusal on the merits that never happened.
    """

    content_accepted = {"verdict": "accurate", "integrity": {"verdict": "accurate"}}
    assert (
        metadata_gate_failure_reason({**content_accepted, "rung_check": {"verdict": "inaccurate"}})
        == "identity-mismatch"
    )
    for rung_check in (
        {"verdict": "cannot-verify"},
        {"verdict": 7},
        {"verdict": "wat"},
        None,
    ):
        reason = metadata_gate_failure_reason({**content_accepted, "rung_check": rung_check})
        assert reason == "cannot-verify-cap-exhausted"
        assert reason in FAILURE_REASON_CODES["accuracy-gate"]

    # Lockstep with the fidelity sibling: the same rung verdict over accepted
    # content derives the same reason in both kernels.
    for rung_verdict in ("inaccurate", "cannot-verify", "wat"):
        metadata_item = {**content_accepted, "rung_check": {"verdict": rung_verdict}}
        fidelity_item = {
            "fidelity": {"verdict": "match"},
            "rung_check": {"verdict": rung_verdict},
        }
        assert metadata_gate_failure_reason(metadata_item) == fidelity_gate_failure_reason(
            fidelity_item
        )

    # The pre-existing arms are unchanged: an undetermined round outranks
    # integrity, an integrity dispute outranks content, and a content rejection
    # is the merits cap.
    assert metadata_gate_failure_reason({"verdict": "cannot-verify"}) == (
        "cannot-verify-cap-exhausted"
    )
    assert (
        metadata_gate_failure_reason({"verdict": "inaccurate", "integrity": None})
        == "identity-mismatch"
    )
    assert (
        metadata_gate_failure_reason(
            {"verdict": "inaccurate", "integrity": {"verdict": "accurate"}}
        )
        == "inaccurate-cap-exhausted"
    )


@pytest.mark.smoke
def test_repair_cap_literals_are_linked_not_restated() -> None:
    """ONE constant links the driver's ``max_repairs`` 2 and the kernel's 3.

    ``max_repairs=2`` at the driver call sites and ``charged < 3`` in the
    terminal-proof kernel encode the same bound (an initial round plus two
    repairs) as two unrelated literals; editing one silently breaks the other.
    """

    from menagerie.crawler.constants import (
        GATE_CAP_CHARGEABLE_REJECTIONS,
        GATE_MAX_REPAIRS,
    )

    assert GATE_MAX_REPAIRS == 2
    assert GATE_CAP_CHARGEABLE_REJECTIONS == GATE_MAX_REPAIRS + 1
    source = inspect.getsource(authority._derive_gate_failure)
    assert "GATE_CAP_CHARGEABLE_REJECTIONS" in source
    assert "charged < 3" not in source


@pytest.mark.smoke
def test_cap_threshold_tracks_the_linked_constant() -> None:
    """The bounded cap proves at exactly the linked threshold and refuses below it."""

    from menagerie.crawler.constants import GATE_CAP_CHARGEABLE_REJECTIONS

    def _rejection(round_no: int) -> dict[str, Any]:
        # The same complaint re-raised every round keeps each rejection
        # chargeable; distinct repair prose keeps fingerprints distinct so the
        # proof rides the chargeable-cap arm, not the repeated-root-cause arm.
        return _metadata_gate(
            gate_id=f"gate-md-{round_no}",
            gate_round=round_no,
            ledger_seq=round_no,
            field_checks=[{"field": "proposed_facts.family", "verdict": "inaccurate"}],
            repairs=[f"correct the family claim, round {round_no}"],
        )

    capped = [_rejection(round_no) for round_no in range(1, GATE_CAP_CHARGEABLE_REJECTIONS + 1)]
    proof = _accuracy_proof(capped)
    assert proof.reason_code == "inaccurate-cap-exhausted"

    with pytest.raises(AuthorityDerivationError, match="bounded cap"):
        _accuracy_proof(capped[:-1])
