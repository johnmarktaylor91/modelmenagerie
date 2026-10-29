"""Rung-11 award-witness reducer-crash regressions: attempt class vs parent-proof class.

Pilot rung 11 (2026-08-07, main @ ``c15c2753``, head50, ~19h in) crashed with
``ReductionError: non-award parent proof cannot name an award receipt`` while
terminalizing m10949, stranding 35 models. The rung was the first campaign in
which a worker session produced a challenge-verified award receipt that the
DRIVER then refused on contract grounds: m10949's eval forward completed
cleanly (exit 0, receipt + completion line verified by the supervisor), but the
worker detected meaningful modes ``[train, eval]`` against a proposal declaring
only ``[eval]``, so ``_attempts_from_supervised`` classified the attempt
``failed`` at stage ``input`` (``contract-invalid``,
``recipe-and-gate-revision-required``) while honestly retaining the parent's
award-witnessing attestation and stripping the award proof material.

``_validate_nonaward_parent_projection`` conflated the two independent axes --
attempt classification (awarded or not) and parent-proof class (award-witnessing
or not) -- by demanding that every non-award attempt carry a NON-award parent
proof. The next global proof replay (``_current_attempt_records`` inside
``prepare_model``) therefore refused the honest persisted row and halted the
campaign. The proof rule itself stands: an award receipt stays nameable ONLY by
an award-witnessing parent proof (completion-line witness paired with the named
digest), no award authority can flow from a non-succeeded row
(``derive_attempt_projection`` requires ``result == "succeeded"``), and a named
digest without its completion witness still refuses.

These tests pin: the honest driver-refused-award row loads as non-award
authority; multiple award receipts plus interleaved non-award terminals reduce
side by side; genuinely contaminated proofs still trip; and the env-gated
replay at the bottom re-runs the frozen rung-11 archive itself where available.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    authenticate_accepted_attempts,
    load_current_attempt_proof,
)
from menagerie.crawler.identity import payload_hash, stable_hash
from menagerie.crawler.tests.conftest import (
    HASH,
    bind_terminal_attempts,
    make_attempt,
    make_failed_attempt,
    make_model,
)
from menagerie.crawler.tests.frozen_archive_support import require_frozen_archive
from menagerie.crawler.tests.test_reducer import CanonicalReducer, _paths


def _driver_refused_award_attempt(stable_id: str, attempt_id: str) -> dict[str, Any]:
    """Build the exact rung-11 m10949 row shape: a driver-refused worker award.

    The worker succeeded (exit 0, receipt and completion line verified, so the
    parent attestation names the award receipt), but the driver refused the
    award on the meaningful-mode contract and recorded the attempt as failed at
    stage ``input`` with the award proof material stripped.

    Parameters
    ----------
    stable_id, attempt_id:
        Exact model and immutable attempt identities.

    Returns
    -------
    dict[str, Any]
        Reducer-valid failed attempt retaining its award-witnessing parent proof.
    """

    attempt = make_attempt(stable_id, attempt_id=attempt_id)
    failure = {
        "stage": "input",
        "reason_code": "contract-invalid",
        "exception_type": None,
        "message": {
            "redaction": "externally-controlled-text-v1",
            "content_sha256": HASH,
            "local_path": f".crawl-local/diagnostics/{attempt_id}.json",
            "diagnostic_key": "$.error.message",
        },
        "traceback": None,
        "no_traceback_reason": "no Python traceback was available",
        "native_crash": False,
        "details": {
            "route": "recipe-and-gate-revision-required",
            "proposal_meaningful_modes": ["eval"],
            "detected_meaningful_modes": ["train", "eval"],
            "missing_proposal_modes": ["train"],
        },
    }
    attempt.update(
        {
            "stage": "input",
            "mode": None,
            "result": "failed",
            "raw_award_receipt": None,
            "raw_award_receipt_sha256": None,
            "error": {**failure, "root_cause_fingerprint": stable_hash(failure)},
        }
    )
    attempt["invocation"]["mode"] = None
    attempt["supervisor_observation"]["stdout_completion_line"] = None
    return _reseal_row(attempt)


def _reseal_row(attempt: dict[str, Any]) -> dict[str, Any]:
    """Re-derive the canonical ledger payload self-hash around a mutated row.

    Parameters
    ----------
    attempt:
        Attempt fixture mutated under test, so the payload self-hash clause
        cannot be the one that refuses the mutation.

    Returns
    -------
    dict[str, Any]
        The same fixture with its payload self-hash re-derived.
    """

    attempt["payload_sha256"] = payload_hash(attempt)
    return attempt


def _reseal_parent(attempt: dict[str, Any]) -> None:
    """Re-derive the parent attestation self-hash around a mutated body.

    Parameters
    ----------
    attempt:
        Attempt whose ``parent_attestation`` body was mutated under test, so the
        self-hash clause cannot be the one that refuses the mutation.
    """

    parent = dict(attempt["parent_attestation"])
    parent["attestation_sha256"] = stable_hash(
        {key: value for key, value in parent.items() if key != "attestation_sha256"}
    )
    attempt["parent_attestation"] = parent
    _reseal_row(attempt)


@pytest.mark.smoke
def test_driver_refused_award_attempt_loads_as_nonaward_authority() -> None:
    """The rung-11 crash row: an honest refused-award attempt is valid non-award proof."""

    refused = _driver_refused_award_attempt("m_example", "attempt-refused-1")
    assert refused["parent_attestation"]["named_raw_award_receipt_sha256"] is not None
    assert load_current_attempt_proof(refused) is None


@pytest.mark.smoke
def test_refused_award_attempt_can_never_be_award_counted() -> None:
    """No award authority can launder through the retained award-witnessing proof."""

    refused = _driver_refused_award_attempt("m_example", "attempt-refused-1")
    with pytest.raises(AuthorityDerivationError, match="not a v3 authenticated success"):
        authenticate_accepted_attempts([str(refused["attempt_id"])], [refused])


@pytest.mark.smoke
def test_named_receipt_without_completion_witness_still_trips() -> None:
    """A receipt digest named without the completion witness that produced it refuses."""

    refused = _driver_refused_award_attempt("m_example", "attempt-refused-1")
    refused["parent_attestation"] = dict(refused["parent_attestation"])
    refused["parent_attestation"]["completion_line_sha256"] = None
    _reseal_parent(refused)
    with pytest.raises(
        AuthorityDerivationError,
        match="witness a completion line and its named award receipt together",
    ):
        load_current_attempt_proof(refused)


@pytest.mark.smoke
def test_completion_witness_without_named_receipt_still_trips() -> None:
    """A half-stripped proof (completion witnessed, receipt naming removed) refuses."""

    refused = _driver_refused_award_attempt("m_example", "attempt-refused-1")
    refused["parent_attestation"] = dict(refused["parent_attestation"])
    refused["parent_attestation"]["named_raw_award_receipt_sha256"] = None
    _reseal_parent(refused)
    with pytest.raises(
        AuthorityDerivationError,
        match="witness a completion line and its named award receipt together",
    ):
        load_current_attempt_proof(refused)


@pytest.mark.smoke
def test_refused_award_attempt_cannot_retain_award_material() -> None:
    """A failed attempt smuggling the raw receipt bytes remains contamination."""

    refused = _driver_refused_award_attempt("m_example", "attempt-refused-1")
    award = make_attempt("m_example", attempt_id="attempt-refused-1")
    refused["raw_award_receipt"] = deepcopy(award["raw_award_receipt"])
    _reseal_row(refused)
    with pytest.raises(AuthorityDerivationError, match="cannot retain award proof material"):
        load_current_attempt_proof(refused)

    refused["raw_award_receipt"] = None
    refused["raw_award_receipt_sha256"] = award["raw_award_receipt_sha256"]
    _reseal_row(refused)
    with pytest.raises(AuthorityDerivationError, match="cannot retain award proof material"):
        load_current_attempt_proof(refused)


@pytest.mark.smoke
def test_rewritten_award_witness_proof_still_trips() -> None:
    """A rewritten award-witnessing attestation fails its self-hash replay."""

    refused = _driver_refused_award_attempt("m_example", "attempt-refused-1")
    refused["parent_attestation"] = dict(refused["parent_attestation"])
    refused["parent_attestation"]["peak_rss_bytes"] = 1
    _reseal_row(refused)
    with pytest.raises(AuthorityDerivationError, match="self hash is invalid"):
        load_current_attempt_proof(refused)


@pytest.mark.smoke
def test_refused_award_row_contradicting_supervisor_facts_still_trips() -> None:
    """The retained proof must still agree with the row's supervisor projection."""

    refused = _driver_refused_award_attempt("m_example", "attempt-refused-1")
    refused["supervisor_observation"] = dict(refused["supervisor_observation"])
    refused["supervisor_observation"]["exit_code"] = 1
    _reseal_row(refused)
    with pytest.raises(AuthorityDerivationError, match="contradicts parent proof at exit_code"):
        load_current_attempt_proof(refused)


@pytest.mark.smoke
def test_multiple_awards_and_refused_award_reduce_side_by_side(tmp_path: Path) -> None:
    """Two runs-awarded models plus interleaved non-award terminals reduce green.

    Rung 11 was the first campaign carrying several award receipts in one
    attempts ledger next to a refused-award row; the global current-proof
    replay must authenticate every row and the refused model must terminalize.

    Parameters
    ----------
    tmp_path:
        Isolated canonical ledgers.
    """

    stable_ids = ["m_award_one", "m_award_two", "m_refused", "m_plain"]

    def _unsequenced(attempt: dict[str, Any]) -> dict[str, Any]:
        attempt.pop("ledger_seq", None)
        attempt.pop("payload_sha256", None)
        return attempt

    with CanonicalReducer(_paths(tmp_path), stable_ids) as reducer:
        reducer.append_attempt(_unsequenced(make_attempt("m_award_one", attempt_id="attempt-a1")))
        refused = _driver_refused_award_attempt("m_refused", "attempt-r1")
        reducer.append_attempt(_unsequenced(deepcopy(refused)))
        reducer.append_attempt(_unsequenced(make_attempt("m_award_two", attempt_id="attempt-b1")))
        runner_failure = make_failed_attempt(
            "m_refused",
            attempt_id="attempt-r2",
            stage="runner",
            reason_code="protocol-violation",
        )
        reducer.append_attempt(_unsequenced(deepcopy(runner_failure)))
        plain_failure = make_failed_attempt("m_plain", attempt_id="attempt-p1", stage="source")
        reducer.append_attempt(_unsequenced(deepcopy(plain_failure)))

        # The exact crashed seam: the global authenticated current-attempt replay.
        current = reducer._current_attempt_records()
        assert [attempt["attempt_id"] for attempt in current] == [
            "attempt-a1",
            "attempt-r1",
            "attempt-b1",
            "attempt-r2",
            "attempt-p1",
        ]

        # The refused-award model terminalizes while award receipts sit beside it.
        model = bind_terminal_attempts(
            make_model("m_refused", status_code="failed:runner"),
            [refused, runner_failure],
        )
        appended = reducer.append_model(model)
        assert appended.record["status"]["code"] == "failed:runner"


def test_rung11_frozen_archive_replays_to_clean_reduction(tmp_path: Path) -> None:
    """Replay the crashed model's actual frozen ledgers through the fix.

    Requires the archived rung-11 campaign state (attempt/gate/model ledgers and
    intake snapshots), resolved through the durable archive contract in
    ``frozen_archive_support`` (typed skip off-fleet, fail-closed under
    ``MENAGERIE_FROZEN_ARCHIVE_GATE=1``). Proves, on the real bytes: every
    persisted attempt row -- the six award-receipt rows and m10949's
    refused-award row included -- authenticates through the exact call that
    crashed the campaign, and m10949's stranded ``failed:runner`` terminal
    records, so a live resume completes instead of crashing again.

    Parameters
    ----------
    tmp_path:
        Scratch root for the copied ledgers.
    """

    import menagerie.crawler.driver  # noqa: F401, PLC0415 -- configures facade deps
    from menagerie.crawler.authority import build_authority_context  # noqa: PLC0415
    from menagerie.crawler.driver_contracts import (  # noqa: PLC0415
        DriverConfig,
        WorkItem,
    )
    from menagerie.crawler.driver_models import _assemble_terminal_model  # noqa: PLC0415
    from menagerie.crawler.intake import load_intake_snapshot  # noqa: PLC0415
    from menagerie.crawler.models import LedgerPaths  # noqa: PLC0415
    from menagerie.crawler.reducer import (  # noqa: PLC0415
        CanonicalReducer as ProductionCanonicalReducer,
    )
    from menagerie.crawler.routing import EnvironmentPhase, IntentRoute  # noqa: PLC0415

    archive = require_frozen_archive("rung11")
    state = tmp_path / "records"
    shutil.copytree(archive / "records", state)
    attempts = [
        json.loads(line)
        for line in (state / "attempts/local.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ledger_ids = {str(attempt["stable_id"]) for attempt in attempts}
    snapshot = next(
        candidate
        for candidate in (
            load_intake_snapshot(intake_dir)
            for intake_dir in sorted((state / "intake").iterdir())
            if intake_dir.is_dir()
        )
        if ledger_ids <= {item.stable_id for item in candidate.items}
    )
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
    config = DriverConfig(campaign_id="c1-mech", run_id="rung11-frozen-replay")
    with ProductionCanonicalReducer(ledgers, context) as reducer:
        # 1. The exact crashed call authenticates every frozen row.
        current = reducer._current_attempt_records()
        assert len(current) == len(attempts)
        refused = next(
            attempt
            for attempt in current
            if attempt["stable_id"] == "m10949" and attempt["stage"] == "input"
        )
        assert refused["result"] == "failed"
        assert refused["parent_attestation"]["named_raw_award_receipt_sha256"] is not None
        assert refused["raw_award_receipt"] is None

        # 2. The stranded model's terminal records, completing the resume path.
        intake_item = next(item for item in snapshot.items if item.stable_id == "m10949")
        work_item = WorkItem(
            intake=intake_item,
            route=IntentRoute("m10949", "core", EnvironmentPhase.PYTORCH),
        )
        model_attempts = tuple(
            attempt for attempt in attempts if attempt.get("stable_id") == "m10949"
        )
        model = _assemble_terminal_model(
            work_item,
            None,
            "failed:runner",
            "protocol-violation",
            "frozen-archive rung-11 resume replay",
            model_attempts,
            reducer.gate_records,
            config,
            "2026-08-07T12:00:00Z",
            human_review=True,
            root_cause_fingerprint=None,
        )
        appended = reducer.append_model(reducer.prepare_model(model))
        assert appended.record["status"]["code"] == "failed:runner"
