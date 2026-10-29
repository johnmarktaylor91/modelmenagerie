"""BL-10 (d): identities are the machine's -- an ACCEPT survives a flubbed echo.

Audit B-4: the rung-7 defect ("terminal disposition source IDs do not exactly
match result") was fixed only for the adverse arms; the ACCEPTED arm -- where
the author's recommendation was just VALIDATED -- still demanded a stochastic
model transcribe up to 19 machine-known identity strings exactly, and one flub
rode ``driver.py``'s catch-all into ``failed:runner / protocol-violation``
(measured flub rate: 2 of 8 terminal items on rung 7). The accepted arm now
derives its checked sets machine-side, exactly as the BLOCKED arm always did.

What must NOT loosen (and is re-pinned here): the machine-derived sets still
face manifest-membership and evidence-reference validation, and an acceptance
over degraded integrity still refuses -- the checker keeps authority over the
verdict, findings, and integrity; it never had authority over identity sets.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorResultBinding,
    DeferRecommendation,
    HandoffExecution,
)
from menagerie.crawler.constants import GATE_SCHEMA_VERSION_V3
from menagerie.crawler.gates import GateRoutingError, validate_terminal_disposition_gate
from menagerie.crawler.tests.conftest import HASH, make_author_proposal, make_gate

pytestmark = pytest.mark.smoke


def _accepted_fixture() -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    """One accepted DEFER terminal gate with two machine-known sources."""

    raw_result = {
        "result_id": "result-defer",
        "result_sha256": HASH,
        "stable_id": "m_example",
        "work_id": "work-m_example",
        "campaign_id": "work-m_example",
        "author_identity": HASH,
        "prompt_identity": HASH,
        "dispatcher_identity": HASH,
        "source_manifest_identity": HASH,
        "intake_snapshot_id": "intake-1",
        "intake_snapshot_sha256": HASH,
        "intake_item_sha256": HASH,
        "created_at": "2026-07-16T00:00:00Z",
    }
    binding = AuthorResultBinding(raw_result=raw_result, **raw_result)
    handoff_proposal = deepcopy(make_author_proposal())
    handoff_proposal["proposal_id"] = "proposal-defer-1"
    result = DeferRecommendation(
        binding=binding,
        platform="cuda",
        source_ids=("source-1", "source-2"),
        evidence_ids=("evidence-1", "evidence-2"),
        evidence_identity=HASH,
        license_identity=HASH,
        recommendation_sha256=HASH,
        handoff_execution=HandoffExecution(
            proposal=handoff_proposal,
            proposal_sha256=HASH,
            code_manifest_identity=HASH,
            source_manifest_identity=HASH,
            handoff_sha256=HASH,
        ),
    )
    gate = make_gate(["m_example"])
    gate.update(
        {
            "schema_version": GATE_SCHEMA_VERSION_V3,
            "gate_kind": "terminal_disposition",
            "batch_size": 1,
            "author_result_schema_identity": HASH,
            "dispatcher_identity": HASH,
        }
    )
    gate["items"][0]["terminal_disposition"] = {
        "author_result_id": "result-defer",
        "author_result_sha256": HASH,
        "kind": "DEFER_RECOMMENDATION",
        "predicate": "needs-cuda",
        "handoff_proposal_id": "proposal-defer-1",
        "handoff_sha256": HASH,
        "verdict": "accepted",
        "source_manifest_identity": HASH,
        "source_ids": ["source-1", "source-2"],
        "evidence_identity": HASH,
        "evidence_ids": ["evidence-1", "evidence-2"],
        "license_identity": HASH,
        "findings": [],
    }
    source_manifest = {
        "manifest_sha256": HASH,
        "sources": [{"source_id": "source-1"}, {"source_id": "source-2"}],
    }
    evidence_pack = {
        "evidence_identity": HASH,
        "excerpts": [
            {"evidence_id": "evidence-1", "source_id": "source-1", "supports": ["needs-cuda"]},
            {"evidence_id": "evidence-2", "source_id": "source-2", "supports": ["needs-cuda"]},
        ],
    }
    return gate, result, source_manifest, evidence_pack


@pytest.mark.parametrize(
    ("field", "flub"),
    (
        ("source_ids", ["source-1", "source-tw0"]),  # one mis-transcribed id
        ("source_ids", ["source-1"]),  # incomplete echo (rung-7 m7637 shape)
        ("evidence_ids", ["evidence-1"]),  # incomplete evidence echo
        ("evidence_ids", ["evidence-1", "evidence-2", "evidence-3"]),  # invented extra
    ),
    ids=("mistranscribed-source", "incomplete-source", "incomplete-evidence", "extra-evidence"),
)
def test_accepted_arm_uses_machine_ids_whatever_the_echo_says(
    field: str, flub: list[str]
) -> None:
    """Any transcription flub yields the MACHINE sets, not a wrong-blame refusal."""

    gate, result, source_manifest, evidence_pack = _accepted_fixture()
    gate["items"][0]["terminal_disposition"][field] = flub

    decision = validate_terminal_disposition_gate(
        gate,
        result,
        source_manifest=source_manifest,
        evidence_pack=evidence_pack,
        license_identity=HASH,
    )
    assert decision.accepted is True
    assert decision.source_ids == ("source-1", "source-2")
    assert decision.evidence_ids == ("evidence-1", "evidence-2")


def test_machine_ids_outside_the_manifest_still_refuse() -> None:
    """No loosening: the machine-derived set still faces manifest membership."""

    gate, result, source_manifest, evidence_pack = _accepted_fixture()
    source_manifest["sources"] = [{"source_id": "source-1"}]  # source-2 vanished

    with pytest.raises(GateRoutingError, match="outside its manifest"):
        validate_terminal_disposition_gate(
            gate,
            result,
            source_manifest=source_manifest,
            evidence_pack=evidence_pack,
            license_identity=HASH,
        )


def test_missing_evidence_excerpt_still_refuses() -> None:
    """No loosening: every machine-checked evidence reference must resolve."""

    gate, result, source_manifest, evidence_pack = _accepted_fixture()
    evidence_pack["excerpts"] = evidence_pack["excerpts"][:1]

    with pytest.raises(GateRoutingError):
        validate_terminal_disposition_gate(
            gate,
            result,
            source_manifest=source_manifest,
            evidence_pack=evidence_pack,
            license_identity=HASH,
        )


def test_acceptance_over_degraded_integrity_still_refuses() -> None:
    """The checker's real authority -- integrity -- is untouched by (d)."""

    gate, result, source_manifest, evidence_pack = _accepted_fixture()
    gate["items"][0]["integrity"]["verdict"] = "cannot-verify"

    with pytest.raises(GateRoutingError, match="accurate item integrity"):
        validate_terminal_disposition_gate(
            gate,
            result,
            source_manifest=source_manifest,
            evidence_pack=evidence_pack,
            license_identity=HASH,
        )


def test_rejected_arm_transcription_discipline_is_unchanged() -> None:
    """The adverse arms keep their existing subset validation untouched."""

    gate, result, source_manifest, evidence_pack = _accepted_fixture()
    item = gate["items"][0]
    item["verdict"] = "inaccurate"
    item["terminal_disposition"]["verdict"] = "rejected"
    item["terminal_disposition"]["source_ids"] = ["source-not-in-result"]

    with pytest.raises(GateRoutingError):
        validate_terminal_disposition_gate(
            gate,
            result,
            source_manifest=source_manifest,
            evidence_pack=evidence_pack,
            license_identity=HASH,
        )
