"""AG-11 (gate-side kernels): per-adjudication-identity telemetry derivations.

200a ran blind on three signals these kernels expose from persisted gate rows
alone: 509/678 metadata rounds (75%) were bought AFTER first acceptance, 119
ACCEPT->REJECT flips occurred, and the ``slop`` verdict arm never fired once
(0/37) without anything able to say so. The monitor half lands in WS9's
surface; these tests pin the WS1 kernels it derives from, so the monitor and
the proof kernels can never disagree about what "accepted" means:

* the timeline counts rounds-after-first-acceptance and ACCEPT->REJECT flips
  in ledger order under the SHARED per-lane acceptance kernels;
* the verdict-arm census enumerates every closed arm up front -- an arm that
  never fired appears with an explicit 0 (printable as ``not-exercised``),
  never silently missing.
"""

from __future__ import annotations

from typing import Any

import pytest

from menagerie.crawler.checker_dispatch import (
    gate_acceptance_timeline,
    gate_adjudication_identity,
    gate_verdict_arm_census,
)
from menagerie.crawler.constants import AccuracyVerdict, FidelityVerdict

pytestmark = pytest.mark.smoke


def _td_gate(verdict: str, *, prompt: str = "sha256:" + "a" * 64) -> dict[str, Any]:
    return {
        "gate_kind": "terminal_disposition",
        "checker": {
            "provider": "openai",
            "model": "gpt-5.6-sol",
            "version": "current",
            "prompt_sha256": prompt,
        },
        "items": [{"stable_id": "m1", "terminal_disposition": {"verdict": verdict}}],
    }


def test_adjudication_identity_extracts_all_four_components() -> None:
    gate = _td_gate("accepted")
    assert gate_adjudication_identity(gate) == (
        "openai",
        "gpt-5.6-sol",
        "current",
        "sha256:" + "a" * 64,
    )
    assert gate_adjudication_identity({}) == ("", "", "", "")


def test_timeline_counts_regates_and_flips() -> None:
    """The m4976 shape: accept then reject on re-gate is one counted flip."""

    rows = [
        _td_gate("rejected"),
        _td_gate("accepted"),
        _td_gate("accepted"),  # re-gate: bought after first acceptance
        _td_gate("rejected"),  # ACCEPT->REJECT flip
        _td_gate("accepted"),
        _td_gate("rejected"),  # second flip
    ]
    timeline = gate_acceptance_timeline(rows, gate_kind="terminal_disposition", stable_id="m1")
    assert timeline["rounds"] == 6
    assert timeline["accepted_rounds"] == 3
    assert timeline["rounds_after_first_acceptance"] == 4
    assert timeline["accept_to_reject_flips"] == 2


def test_arm_census_prints_every_closed_arm_even_at_zero() -> None:
    """The slop/permanent-scar lesson: an unfired arm is an explicit 0 row."""

    census = gate_verdict_arm_census([_td_gate("rejected")])
    assert set(census["metadata_batch"]) == {member.value for member in AccuracyVerdict}
    assert set(census["fidelity"]) == {member.value for member in FidelityVerdict}
    assert census["fidelity"]["slop"] == 0  # present, printable as not-exercised
    assert census["terminal_disposition"] == {
        "accepted": 0,
        "rejected": 1,
        "cannot-verify": 0,
    }
