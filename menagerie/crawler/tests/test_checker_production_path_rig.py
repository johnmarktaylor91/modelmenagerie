"""Driver E2Es whose checker rides the PRODUCTION envelope path.

``ScriptedChecker`` hand-builds its gates and self-hashes, so the 148 driver
E2Es enforce nothing the production ``validate_checker_result`` enforces --
the closed loop T5-3 flagged. :class:`ContractCheckerLane` scripts only the
checker's decisions; envelope construction, machine-owned stamping, and full
result validation are the production code's. These tests prove (a) the driver
runs green through that lane end to end, and (b) checker content the
production path refuses RAISES here -- including the exact 2026-08-04 pilot
class (a hard-coded ``fidelity_identity`` binding the mock used to launder).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from menagerie.crawler.checker_dispatch import (
    CheckerBindingMismatchError,
    CheckerDispatchError,
    compute_result_envelope_sha256,
)
from menagerie.crawler.constants import GATE_SCHEMA_VERSION_V3
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.models import JsonObject
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.tests.checker_contract_support import (
    CheckerContractScript,
    ContractCheckerLane,
)
from menagerie.crawler.tests.conftest import (
    HASH,
    bind_author_proposal,
    make_author_proposal,
    make_bound_source_manifest,
    make_proposed_artifact,
    materialize_checker_model_dir,
)
from menagerie.crawler.tests.test_slice_f_driver import _driver, _paths, _snapshot

pytestmark = pytest.mark.smoke

CONFIG = DriverConfig(run_id="run-test", machine_id="machine-test")


def test_driver_completes_through_the_production_checker_path(tmp_path: Path) -> None:
    """A full driver run rides production envelope construction + validation."""

    snapshot = _snapshot(tmp_path, count=2)
    checker = ContractCheckerLane()
    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "complete"
    assert checker.metadata_calls >= 1
    paths = _paths(tmp_path, snapshot)
    gates = [row for row in scan_jsonl(paths.ledgers.gates) if row["gate_kind"] == "metadata_batch"]
    assert gates, "the run must have bought and recorded production-validated gates"
    for row in gates:
        validate_payload(row, GATE_SCHEMA_VERSION_V3)
        assert row["result_envelope_sha256"] == compute_result_envelope_sha256(row)
        assert row["checker"]["model"] == CONFIG.checker_model
        for item in row["items"]:
            assert item["field_checks"], "production path publishes complete claim coverage"


def test_scripted_rejection_routes_through_bounded_repair(tmp_path: Path) -> None:
    """A production-validated rejection drives the same repair lane as a live one."""

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    checker = ContractCheckerLane(
        CheckerContractScript(metadata_verdicts={stable_id: "inaccurate"})
    )
    result = _driver(tmp_path, snapshot, checker=checker).run()

    assert result.status == "terminal-partition-complete"
    paths = _paths(tmp_path, snapshot)
    models = {row["stable_id"]: row for row in scan_jsonl(paths.ledgers.models)}
    assert models[stable_id]["status"]["code"] == "failed:accuracy-gate"
    gates = [row for row in scan_jsonl(paths.ledgers.gates) if row["gate_kind"] == "metadata_batch"]
    assert gates
    for row in gates:
        item = next(entry for entry in row["items"] if entry["stable_id"] == stable_id)
        assert item["verdict"] == "inaccurate"
        assert item["required_repairs"]


class _LaunderingChecker(ContractCheckerLane):
    """Script the 2026-08-04 pilot defect: a hard-coded ``fidelity_identity``.

    The old mock hard-coded ``None`` into its items, hiding the production
    echo contract; through the production path the same content must refuse
    as a binding mismatch instead of publishing.
    """

    def _author(self, envelope: Mapping[str, Any]) -> JsonObject:
        raw = super()._author(envelope)
        for item in raw["items"]:
            item["fidelity_identity"] = None
        return raw


def test_hardcoded_fidelity_identity_refuses_instead_of_publishing(tmp_path: Path) -> None:
    """The exact pilot mock defect cannot survive the production path."""

    proposal = make_author_proposal("m_fid")
    proposal["fidelity_identity"] = HASH
    # Rebind to a row-derived manifest identity and reseal ``proposal_sha256``
    # over the mutated fidelity identity: the machine digest attestation
    # recomputes both at build, and the test is about the ECHO defect below.
    source_manifest = make_bound_source_manifest([])
    bind_author_proposal(proposal, source_manifest)
    artifact = make_proposed_artifact(
        proposal,
        source_manifest,
        Path(materialize_checker_model_dir("m_fid")),
    )
    with pytest.raises(CheckerBindingMismatchError, match="fidelity_identity"):
        _LaunderingChecker().check_fidelity(artifact, tmp_path, CONFIG)


class _ContradictoryChecker(ContractCheckerLane):
    """Fabricate an acceptance whose own field checks say otherwise."""

    def _author(self, envelope: Mapping[str, Any]) -> JsonObject:
        raw = super()._author(envelope)
        raw["items"][0]["field_checks"][0]["verdict"] = "inaccurate"
        return raw


def test_contradictory_verdict_refuses_instead_of_publishing(tmp_path: Path) -> None:
    """An acceptance over an adverse check dies in production validation."""

    source_manifest = make_bound_source_manifest([])
    proposal = bind_author_proposal(make_author_proposal("m_contradict"), source_manifest)
    artifact = make_proposed_artifact(
        proposal,
        source_manifest,
        Path(materialize_checker_model_dir("m_contradict")),
    )
    with pytest.raises(CheckerDispatchError, match="contradicts"):
        _ContradictoryChecker().check_metadata([artifact], tmp_path, CONFIG)


class _ScaffoldForgingChecker(ContractCheckerLane):
    """Supply a fabricated machine-owned identity, fixture-templating style."""

    def _author(self, envelope: Mapping[str, Any]) -> JsonObject:
        raw = super()._author(envelope)
        raw["dispatcher_identity"] = "sha256:" + "f" * 64
        return raw


def test_fabricated_machine_identity_refuses_at_the_stamp(tmp_path: Path) -> None:
    """A checker-supplied conflicting scaffold value is refused, never laundered."""

    source_manifest = make_bound_source_manifest([])
    proposal = bind_author_proposal(make_author_proposal("m_forge"), source_manifest)
    artifact = make_proposed_artifact(
        proposal,
        source_manifest,
        Path(materialize_checker_model_dir("m_forge")),
    )
    with pytest.raises(CheckerDispatchError, match="machine-owned"):
        _ScaffoldForgingChecker().check_metadata([artifact], tmp_path, CONFIG)


class _IncompleteDisclosureChecker(ContractCheckerLane):
    """Reject on one claim while silently dropping the rest of the inventory."""

    def _author(self, envelope: Mapping[str, Any]) -> JsonObject:
        raw = super()._author(envelope)
        item = raw["items"][0]
        item["verdict"] = "inaccurate"
        item["required_repairs"] = ["repair the first claim"]
        item["field_checks"] = [dict(item["field_checks"][0], verdict="inaccurate")]
        return raw


def test_partial_complaint_disclosure_refuses_instead_of_publishing(tmp_path: Path) -> None:
    """An adverse round covering one claim of many cannot publish (rung-9b class)."""

    source_manifest = make_bound_source_manifest([])
    proposal = bind_author_proposal(make_author_proposal("m_partial"), source_manifest)
    artifact = make_proposed_artifact(
        proposal,
        source_manifest,
        Path(materialize_checker_model_dir("m_partial")),
    )
    with pytest.raises(CheckerDispatchError, match="complaint set is incomplete"):
        _IncompleteDisclosureChecker().check_metadata([artifact], tmp_path, CONFIG)
