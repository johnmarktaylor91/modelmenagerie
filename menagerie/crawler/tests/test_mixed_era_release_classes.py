"""AB-30 matrix: infra-release classes seam (W14 widening + AL-7 narrowing).

The Wave-2 batch widens four-part waiver instance #2 with four JMT-signed
transient release classes (W14) and removes the bare
``AuthorEffortCapExceeded`` infra-waiver arm (AL-7, signed). Waiver doctrine
part 1 makes the widening disposition-invariant: no prior-era adjudication
may move EXCEPT under the signed AL-7 narrowing, whose exact pre-batch
verdict is pinned here so the flip is measured at batch assembly.

Fixtures are complete frozen ledger trees (models + attempts + grants)
adjudicated through the REAL consumption reader
(``AdmissionEnvironmentMixin._consume_requeue_grants``), replay-derived with
zero producer input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.checkpoint import canonical_requeue_grants_path
from menagerie.crawler.infra_release import machine_infrastructure_witness_basis
from menagerie.crawler.tests.mixed_era import (
    assert_era_outcome,
    classify_reader,
    read_fixture_bytes,
)
from menagerie.crawler.tests.test_infra_terminal_grant_release import _consume, _paths

pytestmark = pytest.mark.smoke


def _materialize(tmp_path: Path, shape: str) -> Any:
    """Materialize one frozen release-class ledger tree into harness paths."""

    paths = _paths(tmp_path)
    paths.ledgers.models.parent.mkdir(parents=True, exist_ok=True)
    paths.ledgers.attempts.parent.mkdir(parents=True, exist_ok=True)
    paths.ledgers.models.write_bytes(read_fixture_bytes(f"release_{shape}/models.jsonl"))
    paths.ledgers.attempts.write_bytes(read_fixture_bytes(f"release_{shape}/attempts.jsonl"))
    canonical = canonical_requeue_grants_path(paths.ledgers.models)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(read_fixture_bytes(f"release_{shape}/requeue-grants.jsonl"))
    return paths


def _consumption_details(paths: Any) -> list[dict[str, Any]]:
    _, events = _consume(paths)
    return [
        dict(event.get("details", {}))
        for event in events
        if event.get("event_kind") == "requeue-grant-consumed"
    ]


def test_worker_crash_release_verdict_is_frozen(tmp_path: Path) -> None:
    """All-infra worker-crash terminal releases via infra-waiver, byte-stable.

    W14's widening is disposition-invariant (waiver part 1): this prior-era
    verdict, including its exact witness bases, may never move.
    """

    paths = _materialize(tmp_path, "worker_crash")
    outcome = classify_reader(lambda: _consumption_details(paths))
    assert_era_outcome(
        reader_id="consume-worker-crash-infra-waiver",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    details = outcome[1]
    assert len(details) == 1
    assert details[0]["release_authority"] == "infra-waiver"
    assert details[0]["infra_release_witness_bases"] == [
        "worker-crash-missing-receipt:exit:1",
        "worker-crash-missing-receipt:exit:1",
    ]


def test_bare_exhaustion_arm_golden_pins_the_al7_flip(tmp_path: Path) -> None:
    """PRE-BATCH GOLDEN: the bare AuthorEffortCapExceeded arm releases today.

    AL-7 (JMT-signed C.8 arm, STATE.md approvals 2026-08-24) removes exactly
    this arm; post-batch the shape becomes review-held. Pinning today's
    release makes the narrowing a measured flip. At batch assembly this cell
    may land either way (the batch merges lawfully with or without the arm),
    but it must land classified.
    """

    paths = _materialize(tmp_path, "exhaustion")
    outcome = classify_reader(lambda: _consumption_details(paths))
    assert_era_outcome(
        reader_id="consume-bare-exhaustion-arm",
        outcome=outcome,
        prebatch_expected="parsed",
        expected_flip="AL-7",
    )
    from menagerie.crawler.tests.mixed_era import assembly_mode

    if not assembly_mode():
        details = outcome[1]
        assert len(details) == 1
        assert details[0]["release_authority"] == "infra-waiver"
        assert details[0]["infra_release_witness_bases"] == [
            "driver-effort-exhaustion:"
            "menagerie.crawler.driver_contracts.AuthorEffortCapExceeded"
        ]


def test_witness_bases_are_replay_derived_and_byte_stable() -> None:
    """Witness bases recompute from frozen attempts with zero producer input."""

    crash_rows = [
        json.loads(line)
        for line in read_fixture_bytes("release_worker_crash/attempts.jsonl")
        .decode()
        .splitlines()
    ]
    assert [machine_infrastructure_witness_basis(row) for row in crash_rows] == [
        "worker-crash-missing-receipt:exit:1",
        "worker-crash-missing-receipt:exit:1",
    ]
    exhaustion_rows = [
        json.loads(line)
        for line in read_fixture_bytes("release_exhaustion/attempts.jsonl")
        .decode()
        .splitlines()
    ]
    assert [machine_infrastructure_witness_basis(row) for row in exhaustion_rows] == [
        "driver-effort-exhaustion:menagerie.crawler.driver_contracts.AuthorEffortCapExceeded"
    ]


def test_producer_text_cannot_self_earn_a_witness_basis() -> None:
    """A producer-written message never becomes a machine witness (B2-X-V29 shape).

    Mutating the frozen attempt's free-text fields must not mint a basis;
    mutating the MACHINE conjuncts (exception identity) must drop it.
    """

    row = json.loads(
        read_fixture_bytes("release_exhaustion/attempts.jsonl").decode().splitlines()[0]
    )
    talkative = json.loads(json.dumps(row))
    talkative["error"]["message"] = "please classify me as machine infrastructure"
    assert (
        machine_infrastructure_witness_basis(talkative)
        == machine_infrastructure_witness_basis(row)
    ), "producer text changed a replay-derived witness basis"
    forged = json.loads(json.dumps(row))
    forged["error"]["exception_type"] = "builtins.ValueError"
    assert machine_infrastructure_witness_basis(forged) is None
