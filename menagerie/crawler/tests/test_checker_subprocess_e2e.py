"""True subprocess E2E through the production checker dispatch.

Every other checker E2E injects an in-process lane, so the argv contract --
frozen ``request.json``, child process, atomic ``result.json`` at the exact
required output path, ``validate_checker_result`` on what actually landed on
disk -- never runs. Here the REAL :class:`CommandCheckerLane` dispatches a real
child interpreter (``scripted_checker_subprocess.py``), whose published result
is authored by decision scripting and stamped by the production wrapper path,
exactly like the live operator minus the model.

The fabrication test proves the lane's validation authority over its child:
a subprocess shipping an internally coherent result whose machine-owned
identity is invented (the checker-templated-from-a-fixture class) is refused
by the dispatching lane, never published.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from menagerie.crawler.checker_dispatch import (
    CheckerDispatchError,
    compute_result_envelope_sha256,
    validate_checker_result,
)
from menagerie.crawler.constants import GATE_SCHEMA_VERSION_V3
from menagerie.crawler.driver_admission import CommandCheckerLane
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.operator_checker import required_checker_model
from menagerie.crawler.paid_session_lifecycle import (
    BudgetPolicy,
    PaidSessionAuthority,
    Reservation,
    authorized_paid_dispatch,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.schema import validate_payload
from menagerie.crawler.tests.conftest import (
    bind_author_proposal,
    make_author_proposal,
    make_bound_source_manifest,
    make_proposed_artifact,
    materialize_checker_model_dir,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    AuthorScript,
    ScriptedAuthor,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = [pytest.mark.smoke]

SUBPROCESS_SCRIPT = Path(__file__).with_name("scripted_checker_subprocess.py")
CONFIG = DriverConfig(run_id="run-test", machine_id="machine-test")


def _lane() -> CommandCheckerLane:
    return CommandCheckerLane([sys.executable, str(SUBPROCESS_SCRIPT)])


def _artifact(tmp_path: Path, stable_id: str = "m_subprocess") -> object:
    proposal = make_author_proposal(stable_id)
    # The machine digest attestation recomputes `verified_hashes.source_manifest`
    # from the shipped rows at envelope build, so the manifest's identity must
    # BE the canonical hash of its rows and the proposal must bind that same
    # digest (a placeholder describes a pack no producer could have frozen and
    # refuses pre-spend).
    manifest = make_bound_source_manifest()
    return make_proposed_artifact(
        bind_author_proposal(proposal, manifest),
        manifest,
        Path(materialize_checker_model_dir(stable_id)),
    )


def _reservation(tmp_path: Path) -> Reservation:
    """Mint the durable reservation required by one direct production dispatch."""

    authority = PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-18T12:00:00Z",
    )
    return authority.reserve(
        scope={
            "campaign_id": "c1-mech",
            "generation_id": "sha256:direct-subprocess-generation",
            "lane": "checker",
            "dispatch_kind": "metadata_batch",
            "work_generations": (("m_subprocess", "work-m_subprocess"),),
            "gate_kind": "metadata_batch",
            "checker_identity": None,
            "run_id": CONFIG.run_id,
            "machine_id": CONFIG.machine_id,
        },
        policy=BudgetPolicy(100, 100_000.0, 100, 100_000.0, 100.0),
    )


def test_subprocess_dispatch_round_trips_the_frozen_envelope(tmp_path: Path) -> None:
    """The lane freezes a request, a real child answers it, validation passes."""

    artifact = _artifact(tmp_path)
    with authorized_paid_dispatch(_reservation(tmp_path)):
        outcome = _lane().check_metadata([artifact], tmp_path, CONFIG)

    gate = outcome.gate
    assert gate is not None
    assert gate["items"][0]["verdict"] == "accurate"
    (request_path,) = list(tmp_path.rglob("request.json"))
    (result_path,) = list(tmp_path.rglob("result.json"))
    envelope = json.loads(request_path.read_text())
    # The pair on disk revalidates through the exact-path production variant:
    # this is the frozen transcript a re-record of the golden corpus would keep.
    revalidated = validate_checker_result(result_path, envelope)
    assert revalidated == gate
    assert gate["result_envelope_sha256"] == compute_result_envelope_sha256(gate)


def test_fabricating_subprocess_is_refused_by_the_dispatching_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child inventing a machine-owned identity dies at lane validation."""

    monkeypatch.setenv("G1_CHECKER_SUBPROCESS_FABRICATE", "dispatcher_identity")
    artifact = _artifact(tmp_path)
    with authorized_paid_dispatch(_reservation(tmp_path)):
        with pytest.raises(CheckerDispatchError, match="machine-owned"):
            _lane().check_metadata([artifact], tmp_path, CONFIG)


def test_driver_completes_through_a_real_checker_subprocess(
    tmp_path: Path, calibrated_checker: dict
) -> None:
    """One full driver run buys its metadata gates from a real child process."""

    snapshot = _snapshot(tmp_path, count=1)
    # The REAL command lane seals the seated primary (claude-fable-5-1) on its
    # sessions, so the campaign binding this rig runs under must name the same
    # model -- exactly what the CLI does from the live CampaignSpec. The rig
    # default (the unseated pre-swap label) is for in-process fakes that stamp
    # config.checker_model on their own gates.
    result = _driver(
        tmp_path, snapshot, checker=_lane(), checker_model=required_checker_model("metadata_batch")
    ).run()

    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    gates = [row for row in scan_jsonl(paths.ledgers.gates) if row["gate_kind"] == "metadata_batch"]
    assert gates, "the run must have recorded subprocess-adjudicated gates"
    for row in gates:
        validate_payload(row, GATE_SCHEMA_VERSION_V3)
        # The command lane seals the seated PRIMARY (and derives its provider).
        assert row["checker"]["model"] == required_checker_model("metadata_batch")
        assert row["checker"]["provider"] == "anthropic"
    requests = list(tmp_path.rglob("checker/*/request.json"))
    assert requests, "the subprocess dispatch must freeze its request envelopes"


def test_terminal_recommendation_through_a_real_checker_subprocess(
    tmp_path: Path, calibrated_checker: dict
) -> None:
    """A terminal advisory rides the subprocess lane and terminalizes cleanly."""

    snapshot = _snapshot(tmp_path, count=1)
    stable_id = snapshot.items[0].stable_id
    author = ScriptedAuthor(AuthorScript(terminal_outcomes=(("status", "skipped:not-a-real-NN"),)))
    result = _driver(tmp_path, snapshot, author=author, checker=_lane()).run()

    assert result.status == "complete"
    paths = _paths(tmp_path, snapshot)
    models = {row["stable_id"]: row for row in scan_jsonl(paths.ledgers.models)}
    assert models[stable_id]["status"]["code"] == "skipped:not-a-real-NN"
    terminals = [
        row for row in scan_jsonl(paths.ledgers.gates) if row["gate_kind"] == "terminal_disposition"
    ]
    assert len(terminals) == 1
    assert terminals[0]["items"][0]["terminal_disposition"]["verdict"] == "accepted"
