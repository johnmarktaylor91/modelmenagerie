"""Reuse semantic drift: evidence-identical gates whose ENVELOPE moved never replay.

The reuse key's dangerous failure mode is not a changed proposal -- that moves
``proposal_sha256`` and misses trivially -- it is a prior row whose EVIDENCE
still matches byte-for-byte while the adjudication context it was sealed under
(checker prompt/model/provider/version channel, shipped component bytes,
result-contract version, or the row's own integrity) has drifted. Replaying
such a verdict would assert that a rotated checker already ruled on work it
never saw. Every variant here holds the evidence identity FIXED and drifts
only the envelope; each must fail closed to a live session.

Unlike the W4 suite's mismatch matrix, the priors here are built by the
PRODUCTION envelope path (``checker_contract_support``), and the retired
identities are REAL ones lifted from the golden pilot corpus where they exist
(the pre-rotation prompt hash and the retired dispatcher component identity),
so the drift arms are recorded history, not invented strings.

This suite (with its exact-name headline test) is the C-W4-MERGE RED gate from
S1-T5 finding 3; on trees without ``gate_reuse`` it skips as not-yet-merged.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import pytest

pytest.importorskip(
    "menagerie.crawler.gate_reuse",
    reason="C-W4-MERGE gate: gate-identity verdict reuse is not on this tree",
)

from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.gate_reuse import (
    plan_metadata_gate_reuse,
    reusable_fidelity_gate,
    reusable_metadata_gate,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.recordio import scan_jsonl
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
from menagerie.crawler.tests.test_slice_f_driver import (
    AuthorScript,
    ScriptedAuthor,
    _driver,
    _paths,
    _snapshot,
)

pytestmark = pytest.mark.smoke

CONFIG = DriverConfig(run_id="run-test", machine_id="machine-test")
CORPUS_DIR = Path(__file__).parent / "golden_checker_transcripts"


def _corpus_result(name: str) -> dict[str, Any]:
    path = CORPUS_DIR / f"{name}.json"
    return json.loads(path.read_text())["result"] if path.is_file() else {}


def _retired_prompt_sha256(current: str) -> str:
    """A REAL pre-rotation pilot prompt hash where recorded, else synthetic."""

    recorded = str(
        _corpus_result("metadata_component_drift").get("checker", {}).get("prompt_sha256", "")
    )
    return recorded if recorded and recorded != current else "sha256:" + "1" * 64


def _retired_dispatcher_identity(current: str) -> str:
    """The REAL retired pilot dispatcher component identity where recorded."""

    recorded = str(_corpus_result("metadata_component_drift").get("dispatcher_identity", ""))
    return recorded if recorded and recorded != current else "sha256:" + "2" * 64


def _identity() -> Any:
    return ContractCheckerLane().adjudication_identity("metadata_batch", CONFIG)


def _artifact(tmp_path: Path, stable_id: str = "m_drift") -> Any:
    # The manifest identity is derived from its (empty) rows and the proposal
    # is rebound to it: the machine digest attestation recomputes
    # ``verified_hashes.source_manifest`` from the shipped rows at build.
    source_manifest = make_bound_source_manifest([])
    proposal = bind_author_proposal(make_author_proposal(stable_id), source_manifest)
    return make_proposed_artifact(
        proposal,
        source_manifest,
        Path(materialize_checker_model_dir(stable_id)),
    )


def _persisted_rejection(tmp_path: Path, artifact: Any) -> dict[str, Any]:
    """One production-path rejected metadata row, persisted-shaped."""

    stable_id = str(artifact.proposal["stable_id"])
    lane = ContractCheckerLane(CheckerContractScript(metadata_verdicts={stable_id: "inaccurate"}))
    gate = lane.check_metadata([artifact], tmp_path, CONFIG).gate
    assert gate is not None
    persisted = dict(deepcopy(gate))
    persisted["ledger_seq"] = 1
    persisted["payload_sha256"] = HASH
    return persisted


def _reseal(gate: dict[str, Any]) -> dict[str, Any]:
    gate["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in gate.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    return gate


def _rotate_prompt(gate: dict[str, Any], identity: Any) -> None:
    gate["checker"]["prompt_sha256"] = _retired_prompt_sha256(identity.prompt_sha256)


def _switch_model(gate: dict[str, Any], identity: Any) -> None:
    gate["checker"]["model"] = "gpt-5.6-sol" if identity.model != "gpt-5.6-sol" else "gpt-5.6-terra"


def _move_provider(gate: dict[str, Any], identity: Any) -> None:
    del identity
    gate["checker"]["provider"] = "azure-openai"


def _move_version(gate: dict[str, Any], identity: Any) -> None:
    gate["checker"]["version"] = identity.version + "-next"


def _drift_dispatcher_component(gate: dict[str, Any], identity: Any) -> None:
    del identity
    gate["dispatcher_identity"] = _retired_dispatcher_identity(str(gate["dispatcher_identity"]))


def _drift_schema_component(gate: dict[str, Any], identity: Any) -> None:
    del identity
    gate["author_result_schema_identity"] = "sha256:" + "3" * 64


def _drift_payload(gate: dict[str, Any], identity: Any) -> None:
    del identity
    gate["items"][0]["verified_hashes"]["proposal"] = "sha256:" + "4" * 64


def _regress_schema_version(gate: dict[str, Any], identity: Any) -> None:
    del identity
    gate["schema_version"] = "menagerie.crawler.gate.v2"


DRIFTS: tuple[tuple[str, Callable[[dict[str, Any], Any], None], bool], ...] = (
    ("prompt_rotated", _rotate_prompt, True),
    ("checker_model_switched", _switch_model, True),
    ("provider_moved", _move_provider, True),
    ("version_channel_moved", _move_version, True),
    ("dispatcher_component_drifted", _drift_dispatcher_component, True),
    ("author_result_schema_drifted", _drift_schema_component, True),
    ("result_contract_regressed", _regress_schema_version, True),
    # A torn row: the stored envelope no longer validates its own self-hash.
    (
        "row_no_longer_self_validates",
        lambda gate, _: gate["items"][0].update(confidence="low"),
        False,
    ),
)


@pytest.mark.parametrize(("name", "drift", "reseal"), DRIFTS, ids=[d[0] for d in DRIFTS])
def test_gate_reuse_key_rejects_semantic_drift(
    tmp_path: Path, name: str, drift: Callable[[dict[str, Any], Any], None], reseal: bool
) -> None:
    """An evidence-identical prior whose envelope drifted buys a live session."""

    artifact = _artifact(tmp_path)
    identity = _identity()
    persisted = _persisted_rejection(tmp_path, artifact)

    # Retirement (BL-10 h): the positive control -- an undrifted prior
    # replaying -- is gone; every arm refuses, undrifted included. The per-drift
    # refusals below are the regression suite for any future re-arming, at
    # which point the control assertion must be restored from history.
    control = reusable_metadata_gate([artifact], [persisted], checker=identity)
    assert control is None, "retired reuse must refuse even the undrifted prior"

    drifted = deepcopy(persisted)
    drift(drifted, identity)
    if reseal:
        _reseal(drifted)
    assert drifted["items"] == persisted["items"] or not reseal, (
        "drift arms must hold the adjudicated evidence fixed"
    )
    assert reusable_metadata_gate([artifact], [drifted], checker=identity) is None, (
        f"{name}: a drifted envelope must fail closed to a live session"
    )


@pytest.mark.parametrize(
    ("name", "drift"),
    (
        ("payload", _drift_payload),
        ("prompt", _rotate_prompt),
        ("model", _switch_model),
        ("schema", _drift_schema_component),
        ("dispatcher", _drift_dispatcher_component),
    ),
)
def test_newer_identity_mutation_never_falls_back_to_an_older_cached_verdict(
    tmp_path: Path,
    name: str,
    drift: Callable[[dict[str, Any], Any], None],
) -> None:
    """Every changed identity component forces the pending item into live work.

    The older row is a valid full-identity cache hit. The newer row adjudicates
    the same stable ID but moves exactly one of the five required reuse
    components. A lookup that filters by identity before applying recency can
    skip the newer row and replay the stale older verdict; the planner must
    instead classify the item as live.
    """

    artifact = _artifact(tmp_path)
    identity = _identity()
    older = _persisted_rejection(tmp_path, artifact)
    newer = deepcopy(older)
    newer["gate_id"] = "gate-newer-adjudication"
    newer["gate_identity"] = "sha256:" + "5" * 64
    newer["ledger_seq"] = 2
    drift(newer, identity)
    _reseal(newer)

    # Retirement: the older row no longer replays either; both spellings of
    # the history must classify the item live.
    assert reusable_metadata_gate([artifact], [older], checker=identity) is None
    assert reusable_metadata_gate([artifact], [older, newer], checker=identity) is None, name
    groups, live = plan_metadata_gate_reuse([artifact], [older, newer], checker=identity)
    assert groups == (), name
    assert live == (artifact,), name


def test_fidelity_reuse_key_rejects_prompt_rotation(tmp_path: Path) -> None:
    """The fidelity lane honors the same fail-closed envelope-drift rule."""

    proposal = make_author_proposal("m_fid_drift")
    proposal["fidelity_identity"] = HASH
    # Rebinding reseals ``proposal_sha256`` over the mutated fidelity identity.
    source_manifest = make_bound_source_manifest([])
    bind_author_proposal(proposal, source_manifest)
    artifact = make_proposed_artifact(
        proposal,
        source_manifest,
        Path(materialize_checker_model_dir("m_fid_drift")),
    )
    lane = ContractCheckerLane(
        CheckerContractScript(fidelity_verdicts={"m_fid_drift": "major-drift"})
    )
    gate = lane.check_fidelity(artifact, tmp_path, CONFIG).gate
    assert gate is not None
    persisted = _reseal({**deepcopy(gate), "ledger_seq": 1, "payload_sha256": HASH})
    identity = _identity()
    # Retirement: even the undrifted prior refuses.
    assert reusable_fidelity_gate(artifact, [persisted], checker=identity) is None

    drifted = deepcopy(persisted)
    _rotate_prompt(drifted, identity)
    _reseal(drifted)
    assert reusable_fidelity_gate(artifact, [drifted], checker=identity) is None


class _RotatedPromptLane(ContractCheckerLane):
    """A lane whose NEXT session would buy under a rotated checker prompt."""

    def adjudication_identity(self, gate_kind: str, config: DriverConfig) -> Any:
        from menagerie.crawler.gate_reuse import CheckerAdjudicationIdentity

        declared = super().adjudication_identity(gate_kind, config)
        return CheckerAdjudicationIdentity(
            provider=declared.provider,
            model=declared.model,
            version=declared.version,
            prompt_sha256="sha256:" + "9" * 64,
        )


def _clear_model_history(paths: Any) -> None:
    """Simulate a re-drive leg: models and driver state reset, gates kept."""

    for stale in (paths.ledgers.models, paths.driver_state):
        if stale.exists():
            stale.unlink()


def test_driver_buys_live_when_the_checker_prompt_rotates_between_legs(tmp_path: Path) -> None:
    """A re-drive leg after prompt rotation never replays pre-rotation verdicts."""

    snapshot = _snapshot(tmp_path, count=1)
    author = ScriptedAuthor(AuthorScript(terminal_outcomes=(("status", "skipped:not-a-real-NN"),)))
    leg_one = ContractCheckerLane()
    assert _driver(tmp_path, snapshot, author=author, checker=leg_one).run().status == "complete"
    # AG-4 duplicate-accept (Wave-2 batch): a fresh terminal ACCEPT buys one
    # blind confirmation session -- two live calls per fresh acceptance.
    assert leg_one.terminal_calls == 2

    paths = _paths(tmp_path, snapshot)
    _clear_model_history(paths)
    author_two = ScriptedAuthor(
        AuthorScript(terminal_outcomes=(("status", "skipped:not-a-real-NN"),))
    )
    leg_two = _RotatedPromptLane()
    result = _driver(tmp_path, snapshot, author=author_two, checker=leg_two).run()

    # The leg finishes either way; the claim under test is the SESSION, not the
    # completion label (a second live terminal row closes the partition, which
    # the reused-idempotent leg in the W4 suite never appends).
    assert result.status in {"complete", "terminal-partition-complete"}
    # W4's redrive test proves terminal_calls == 0 when identity holds; after a
    # rotation the same evidence-identical row MUST buy a fresh session.
    # Two calls again under AG-4: the rotated identity cannot inherit leg
    # one's accept as its blind witness (different adjudication identity), so
    # the fresh accept buys its own confirmation. Zero calls would mean the
    # pre-rotation verdict replayed -- the defect this test pins.
    assert leg_two.terminal_calls == 2, "post-rotation reuse would replay an unseen checker"
    events = scan_jsonl(paths.operational_ledger)
    reused = [
        event
        for event in events
        if event.get("details", {}).get("disposition") == "checker-verdict-reused"
    ]
    assert reused == [], "no verdict may be marked reused across a prompt rotation"
