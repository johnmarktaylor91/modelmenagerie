"""Rung-9 recording-halt regressions: the write-contract seam and the terminal fallback.

Pilot rung 9 (2026-08-05, main @ ``8fa88386``) halted with
``TerminalRecordingUnavailable`` after THREE consecutive models could not have a
terminal recorded, stranding nine. Two independent defects composed:

1. PRIMARY -- one batch checker emitted a ``metadata_batch`` gate whose nine items
   were all fully accurate while EVERY ``field_checks`` row carried EMPTY
   ``checked_source_ids``/``evidence_ids``. The gate schema's plain ``string_array``
   let it publish, admission routed it ``canonical_write_allowed`` on verdicts
   alone, and the refusal surfaced only inside
   ``validate_authored_facts_for_write`` at canonical write:
   ``authored field check lacks checked source context:
   external_metadata.architecture_class``.
2. FALLBACK -- the artifact-free minimal terminal rung passed ``attempts=()``, so
   its assembled record carried the NOT-APPLICABLE ``execution.env_generation``
   placeholder while the reducer derived the terminal proof from the LEDGER's
   environment-carrying attempts: ``model environment generation contradicts its
   attempt proof``. The rung that exists precisely to survive bookkeeping refusals
   refused deterministically for every model with a persisted environment attempt.

Frozen-artifact replay (the rung-9 archive) confirmed both failures byte-exactly at
``8fa88386`` AND at pre-wave-2 ``19a72e4a``: no wave-2 commit opened the seam; it
was latent on both sides. These tests reproduce both defects synthetically through
the real driver/operator lanes and pin the fixes; the env-gated replay test at the
bottom re-runs the frozen archive itself where it is available.
"""

from __future__ import annotations

from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import shutil
from typing import Any, Sequence

import pytest

from menagerie.crawler.checker_dispatch import (
    LEDGER_ASSIGNED_GATE_FIELDS,
    CheckerDispatchError,
    apply_machine_owned_gate_fields,
    build_metadata_vet_envelope,
    machine_owned_gate_fields,
    validate_checker_result_mapping,
)
from menagerie.crawler.constants import CheckerSeatRole, METADATA_BATCH_MAX, GateKind, OperationalEventKind
from menagerie.crawler.driver_models import _metadata_gate_accepted
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.metadata import (
    MetadataValidationError,
    validate_authored_facts_for_write,
)
from menagerie.crawler.operator_checker import (
    CodexAttempt,
    execute_checker_request,
    required_checker_model,
)
from menagerie.crawler.operator_protocol import OperatorExitCode
from menagerie.crawler.state import scan_jsonl
from menagerie.crawler.tests.conftest import _model_facts, make_gate, make_model
from menagerie.crawler.tests.test_operator_checker import (
    _checker_item_pack,
    _events,
    _native_final_message,
)
from menagerie.crawler.tests.test_slice_f_driver import (
    FakeChecker,
    _driver,
    _model_records,
    _paths,
    _snapshot,
)

from menagerie.crawler.tests.frozen_archive_support import require_frozen_archive


def _strip_source_context(gate: dict[str, Any]) -> dict[str, Any]:
    """Apply the recorded rung-9 checker emission to a gate in place.

    Every field check keeps its accurate verdict and loses its source context --
    the exact shape the batch checker emitted for all nine models.
    """

    for item in gate["items"]:
        for check in item["field_checks"]:
            check["checked_source_ids"] = []
            check["evidence_ids"] = []
    gate["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in gate.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    return gate


class Rung9PoisonedChecker(FakeChecker):
    """Emit fully accurate metadata gates whose every check lacks source context."""

    def check_metadata(self, artifacts, work_root, config):  # noqa: ANN001, ANN201
        """Return the accurate synthetic gate with the rung-9 emission applied."""

        outcome = super().check_metadata(artifacts, work_root, config)
        assert outcome.gate is not None
        _strip_source_context(outcome.gate)
        return outcome


@pytest.mark.smoke
def test_rung9_composed_failure_now_terminalizes_every_model_honestly(
    tmp_path: Path,
) -> None:
    """The exact composed failure records honest terminals instead of halting.

    The poisoned gate is injected through the driver's checker dependency --
    modeling a ledger the pre-fix dispatch validation let through -- so every
    model marches to run award, refuses at canonical write (PRIMARY), refuses
    again on the artifact-carrying terminal rung, and lands on the minimal
    fallback. Pre-fix, that fallback contradicted the attempt proof's
    environment generation (FALLBACK) and the third model raised
    ``TerminalRecordingUnavailable``; the run below would die rather than
    return. Post-fix the fallback keeps the persisted attempts, agrees with
    reducer-derived authority, and every model records a ``failed:runner``
    terminal with zero unrecordable events.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    """

    snapshot = _snapshot(tmp_path, count=3)
    result = _driver(tmp_path, snapshot, checker=Rung9PoisonedChecker()).run()

    models = _model_records(tmp_path, snapshot)
    assert len(models) == 3
    attempts = scan_jsonl(_paths(tmp_path, snapshot).ledgers.attempts)
    env_by_id = {
        str(attempt["attempt_id"]): attempt.get("identities", {}).get("environment")
        for attempt in attempts
    }
    for stable_id, record in models.items():
        assert record["status"]["kind"] == "failed", stable_id
        assert record["status"]["code"] == "failed:runner", stable_id
        assert record["status"]["reason_code"] == "protocol-violation", stable_id
        # The minimal fallback landed: no artifact-derived facts, human review
        # requested, and the record NAMES its persisted attempts.
        assert record["status"]["human_review"]["required"] is True, stable_id
        attempt_ids = record["status"]["attempt_ids"]
        assert attempt_ids, stable_id
        # The record's environment generation agrees with the attempt proof --
        # the exact contradiction that made rung 9's fallback unrecordable.
        proof_generations = {
            env_by_id[attempt_id]
            for attempt_id in attempt_ids
            if isinstance(env_by_id.get(attempt_id), str)
        }
        assert record["execution"]["env_generation"] in proof_generations, stable_id
    events = scan_jsonl(_paths(tmp_path, snapshot).operational_ledger)
    unrecordable = [
        event
        for event in events
        if event.get("event_kind") == OperationalEventKind.TERMINAL_UNRECORDABLE.value
    ]
    assert not unrecordable
    assert result.status in {"complete", "terminal-partition-complete"}


@pytest.mark.smoke
def test_rung9_poisoned_emission_is_refused_with_the_exact_write_error(
    tmp_path: Path, calibrated_checker: dict
) -> None:
    """The operator lane refuses rung-9 BEFORE publication, then repairs it.

    Attempt 1 injects the recorded shape (fully accurate, every check without
    source context); the machine validator must refuse it BEFORE publication
    with the canonical-write error named. The defect is the gate item's own
    citation bookkeeping -- rung 9's live checker emitted complete
    checked_source_ids on its repair rounds -- so the attribution is
    checker-side and the bounded fed-back re-ask runs; the honest second
    answer publishes. The unchanged honest shape is validated separately
    below, proving the tripwire still distinguishes valid output from the
    poisoned emission.

    Parameters
    ----------
    tmp_path:
        Isolated wrapper root.
    """

    # The recorded rung-9 trio, truncated to the calibrated batch width
    # (the tripwire is per-item; m10517 carries the poisoned emission).
    gate = make_gate(["m10517", "m4066", "m4334"][:METADATA_BATCH_MAX])
    packs = [_checker_item_pack(item) for item in gate["items"]]
    # The checker copies the envelope's row-derived ``verified_hashes`` verbatim,
    # so the recorded emission echoes the packs' digests, never the fixture's
    # placeholders (the machine digest attestation rebinds them at pack build).
    for gate_item, pack in zip(gate["items"], packs):
        gate_item["verified_hashes"] = deepcopy(pack["verified_hashes"])
    envelope = build_metadata_vet_envelope(
        packs,
        gate_round=1,
        output_path=tmp_path / "result.json",
        # Codex-shaped wrapper fault test: the OPENAI (confirmation) seat.
        checker_model=required_checker_model(GateKind.METADATA_BATCH, CheckerSeatRole.CONFIRMATION),
        checker_version="current",
        request_nonce="rung9-replay",
        final_tail=True,
    )
    request_path = tmp_path / "request.json"
    request_path.write_bytes(canonical_json_bytes(envelope) + b"\n")
    for field in (*machine_owned_gate_fields(envelope), *LEDGER_ASSIGNED_GATE_FIELDS):
        gate.pop(field, None)
    gate.pop("checker", None)
    gate.pop("result_envelope_sha256", None)
    poisoned = _strip_source_context(deepcopy(gate))
    prompts: list[str] = []

    def invoke(argv: Sequence[str], last_message: Path, timeout: float) -> CodexAttempt:
        """Inject the recorded poisoned emission, then the honest one."""

        del timeout
        prompts.append(str(argv[-1]))
        message = poisoned if len(prompts) == 1 else gate
        last_message.write_bytes(_native_final_message(message))
        return CodexAttempt(
            0,
            '{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
            "",
        )

    exit_code = execute_checker_request(
        request_path,
        invoke=invoke,
        sleep=lambda _seconds: None,
        diagnostic_stream=StringIO(),
    )

    assert exit_code is OperatorExitCode.SUCCESS
    assert len(prompts) == 2, "the checker-side slip earns its fed-back re-ask"
    assert "lacks checked source context" in prompts[1]
    assert (tmp_path / "result.json").exists()
    refusals = [
        event for event in _events(request_path) if event["event"] == "item-contract-refused"
    ]
    assert [event["attempt"] for event in refusals] == [1]
    assert [event["retrying"] for event in refusals] == [True]
    assert [event["author_side"] for event in refusals] == [False]
    assert refusals[0]["stable_ids"] == ["m10517"]
    assert "would refuse at canonical write" in refusals[0]["detail"]
    assert "lacks checked source context" in refusals[0]["detail"]

    stamped = apply_machine_owned_gate_fields(
        gate,
        envelope,
        started_at="2026-08-17T00:00:00Z",
        finished_at="2026-08-17T00:00:01Z",
    )
    published = validate_checker_result_mapping(stamped, envelope)
    assert all(item["verdict"] == "accurate" for item in published["items"])


def test_rung9_persisted_poisoned_gate_is_not_accepted_for_write() -> None:
    """A persisted poisoned-but-accurate gate no longer counts as accepted.

    Relaunch admission consults persisted gates through
    ``_metadata_gate_accepted``; on verdicts alone the rung-9 gate re-admitted
    the stranded models straight past the checker into a canonical write that
    deterministically refuses. Acceptance now replays the write contract, so the
    poisoned gate routes the model back to a fresh checker round while an honest
    gate stays accepted.
    """

    stable_id = "m10517"
    gate = make_gate([stable_id])
    item = gate["items"][0]
    proposal = {
        "stable_id": stable_id,
        "work_id": item["work_id"],
        "proposal_sha256": item["verified_hashes"]["proposal"],
        "vet_identity": item["vet_identity"],
        "fidelity_identity": item["fidelity_identity"],
        "verified_hashes": {
            key: value for key, value in item["verified_hashes"].items() if key != "proposal"
        },
        "proposed_facts": _model_facts(make_model(stable_id, accepted=True)),
    }
    assert _metadata_gate_accepted([gate], stable_id, proposal) is True
    poisoned = _strip_source_context(deepcopy(gate))
    assert _metadata_gate_accepted([poisoned], stable_id, proposal) is False
    # The write-time tripwire itself is untouched: it still refuses the item.
    with pytest.raises(MetadataValidationError, match="lacks checked source context"):
        validate_authored_facts_for_write(proposal["proposed_facts"], poisoned["items"][0])


def test_adverse_verdicts_are_exempt_from_the_write_contract_replay() -> None:
    """An adverse item publishes untouched: it is never write-eligible.

    The dispatch-time replay guards exactly the items canonical write will
    consult (all three verdicts accurate). Refusing adverse items over their
    field-check structure would turn honest rejections into contract failures
    and starve the repair loop.
    """

    stable_id = "m5445"
    gate = make_gate([stable_id])
    poisoned = _strip_source_context(deepcopy(gate))
    item = poisoned["items"][0]
    item["verdict"] = "inaccurate"
    item["field_checks"][0]["verdict"] = "inaccurate"
    item["field_checks"][0]["required_repair"] = "ground the claim"
    item["required_repairs"] = ["ground the claim"]
    pack = _checker_item_pack(gate["items"][0])
    from menagerie.crawler.checker_dispatch import _validate_item_write_contract

    # Adverse: returns without refusing despite the stripped source context.
    _validate_item_write_contract(item, pack)
    # Accurate: the same stripped shape refuses.
    with pytest.raises(CheckerDispatchError, match="would refuse at canonical write"):
        _validate_item_write_contract(poisoned["items"][0] | {"verdict": "accurate"}, pack)


def test_rung9_frozen_archive_replays_to_recordable_terminals(tmp_path: Path) -> None:
    """Replay the stranded models' actual frozen artifacts through the fixes.

    Requires the archived rung-9 campaign state (checker request/result, gate and
    attempt ledgers, intake snapshot), resolved through the durable archive
    contract in ``frozen_archive_support`` (typed skip off-fleet, fail-closed
    under ``MENAGERIE_FROZEN_ARCHIVE_GATE=1``). Proves, on the real bytes: the
    poisoned batch can no longer publish; no stranded model's persisted gate is
    accepted for write on relaunch; and the minimal fallback records honest
    terminals for both recorded attempt topologies.

    Parameters
    ----------
    tmp_path:
        Scratch root for the copied ledgers.
    """

    archive = require_frozen_archive("rung9")
    envelope = json.loads(
        (archive / "work/checker/metadata-143c9aa7dfcc0bb1/request.json").read_text()
    )
    frozen_result = json.loads(
        (archive / "work/checker/metadata-143c9aa7dfcc0bb1/result.json").read_text()
    )
    # 1a. Byte-exact, the frozen batch still refuses at dispatch: the archived
    #     result was stamped by the rung-9-era dispatcher, so today's
    #     machine-owned scaffold comparison fails closed before any content
    #     check. Every dispatcher change rotates this identity, so the frozen
    #     bytes can never again reach publication as-is.
    with pytest.raises(CheckerDispatchError, match="machine-owned value"):
        validate_checker_result_mapping(frozen_result, envelope)

    # 1b. Restamped with today's machine-owned scaffold -- checker-authored
    #     content untouched -- the refusal is the canonical-write contract:
    #     the defect-1 source-context seam this replay exists to pin.
    restamped = dict(frozen_result)
    restamped.update(machine_owned_gate_fields(envelope))
    with pytest.raises(CheckerDispatchError, match="lacks checked source context"):
        validate_checker_result_mapping(restamped, envelope)

    # 2. No batch member's persisted gate is accepted for canonical write.
    env_items = {item["stable_id"]: item for item in envelope["items"]}
    gates = [
        json.loads(line)
        for line in (archive / "records/gates/current-shard.jsonl").read_text().splitlines()
        if line.strip()
    ]
    round1 = next(g for g in gates if g["gate_round"] == 1 and g["batch_size"] == 9)
    for item in round1["items"]:
        stable_id = item["stable_id"]
        assert (
            _metadata_gate_accepted(gates, stable_id, env_items[stable_id]["proposal"]) is False
        ), stable_id

    # 3. The minimal fallback appends honest terminals for both recorded
    #    attempt topologies (runner failure after successful forwards; import
    #    failure with a single environment-carrying attempt).
    import menagerie.crawler.driver  # noqa: F401, PLC0415 -- configures facade deps
    from menagerie.crawler.authority import build_authority_context  # noqa: PLC0415
    from menagerie.crawler.driver_contracts import (  # noqa: PLC0415
        DriverConfig,
        WorkItem,
    )
    from menagerie.crawler.driver_models import _assemble_terminal_model  # noqa: PLC0415
    from menagerie.crawler.intake import load_intake_snapshot  # noqa: PLC0415
    from menagerie.crawler.models import LedgerPaths  # noqa: PLC0415
    from menagerie.crawler.reducer import CanonicalReducer  # noqa: PLC0415
    from menagerie.crawler.routing import EnvironmentPhase, IntentRoute  # noqa: PLC0415

    state = tmp_path / "records"
    shutil.copytree(archive / "records", state)
    snapshot = load_intake_snapshot(state / "intake/intake-61d2ae3aed9ea0cbb3f3")
    attempts = [
        json.loads(line)
        for line in (state / "attempts/local.jsonl").read_text().splitlines()
        if line.strip()
    ]
    env_generation = next(
        attempt["identities"]["environment"]
        for attempt in attempts
        if isinstance(attempt.get("identities", {}).get("environment"), str)
    )
    context = build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model="claude-sonnet",
        author_version="current",
        checker_model="gpt-5.6-sol",
        checker_version="current",
        environment_generations={"core": env_generation},
    )
    ledgers = LedgerPaths(
        models=state / "models/current-shard.jsonl",
        attempts=state / "attempts/local.jsonl",
        gates=state / "gates/current-shard.jsonl",
        artifacts=state / "artifacts/current-shard.jsonl",
    )
    config = DriverConfig(campaign_id="c1-mech", run_id="rung9-frozen-replay")
    with CanonicalReducer(ledgers, context) as reducer:
        for stable_id, status_code in (("m10517", "failed:runner"), ("m4066", "failed:import")):
            intake_item = next(i for i in snapshot.items if i.stable_id == stable_id)
            work_item = WorkItem(
                intake=intake_item,
                route=IntentRoute(stable_id, "core", EnvironmentPhase.PYTORCH),
            )
            model_attempts = tuple(
                attempt for attempt in attempts if attempt.get("stable_id") == stable_id
            )
            model = _assemble_terminal_model(
                work_item,
                None,
                status_code,
                "protocol-violation",
                "frozen-archive minimal fallback replay",
                model_attempts,
                reducer.gate_records,
                config,
                "2026-08-05T18:00:00Z",
                human_review=True,
                root_cause_fingerprint=None,
            )
            appended = reducer.append_model(reducer.prepare_model(model))
            assert appended.record["status"]["code"] == status_code, stable_id
