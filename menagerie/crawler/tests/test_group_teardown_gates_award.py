"""DX-11/RE-12: the process-group teardown gates the award, bound in attestation.

The synthesis VERIFIED that a survivor-forged award was possible: the v3
association conjunction never consulted ``group_teardown_outcome`` or
``group_survivor_pids``, so a worker whose grandchild outlived supervision --
inheriting execution flocks and outliving the lease -- could still attest a
success. The v3 parent attestation now BINDS both facts (a tampered outcome
breaks the self-hash), the live association requires an awardable teardown,
and the authority's replay validation refuses a success attestation whose
teardown is anything but ``clean``/``reaped`` with zero survivors. Historical
v2 attestations stay readable for carried heads (AB-30 mixed-era boundary).
"""

from __future__ import annotations

from typing import Any

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    _validate_parent_attestation,
    derive_parent_attestation,
)
from menagerie.crawler.completion_line import (
    completion_line_for_raw_award_receipt,
)
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.worker_supervisor import (
    _AWARDABLE_TEARDOWN_OUTCOMES,
    _PARENT_ATTESTATION_KEYS,
    _PARENT_ATTESTATION_KEYS_V3,
    build_parent_attestation,
)
from menagerie.crawler.tests.test_pre_model_code_worker_deaths import _observation

pytestmark = pytest.mark.smoke


def _teardown_observation(outcome: Any, survivors: tuple[int, ...]):
    """Return a clean-exit observation with the given teardown facts."""

    base = _observation(0)
    object.__setattr__(base, "group_teardown_outcome", outcome)
    object.__setattr__(base, "group_survivor_pids", survivors)
    object.__setattr__(base, "started_at", "2026-08-24T00:00:00Z")
    object.__setattr__(base, "finished_at", "2026-08-24T00:00:01Z")
    return base


def _raw_receipt() -> dict[str, Any]:
    """Return a minimal raw award receipt for completion-line derivation."""

    return {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": "nonce-1",
        "request_sha256": "sha256:" + "1" * 64,
        "stable_id": "m_teardown",
        "work_id": "work-teardown",
        "execution_identity": "sha256:" + "2" * 64,
        "recipe_revision": "sha256:" + "3" * 64,
        "code_manifest_identity": "sha256:" + "4" * 64,
        "input_identity": "sha256:" + "5" * 64,
        "requested_mode": "eval",
        "observation": {"forward_completed": True},
    }


def _success_attestation(outcome: Any, survivors: list[int]) -> tuple[dict[str, Any], str]:
    """Derive a real success attestation carrying the given teardown facts."""

    raw = _raw_receipt()
    line = completion_line_for_raw_award_receipt(raw)
    observation = {
        "exit_code": 0,
        "signal": None,
        "timed_out": False,
        "rss_exceeded": False,
        "peak_rss_bytes": 1,
        "stdout_sha256": "sha256:" + "6" * 64,
        "stderr_sha256": "sha256:" + "7" * 64,
        "group_teardown_outcome": outcome,
        "group_survivor_pids": survivors,
    }
    attestation = derive_parent_attestation(
        raw,
        line,
        observation,
        started_at="2026-08-24T00:00:00Z",
        finished_at="2026-08-24T00:00:01Z",
    )
    return attestation, line


def test_v3_attestation_binds_the_teardown_facts() -> None:
    """The teardown facts ride the v3 payload and the self-hash binds them."""

    observation = _teardown_observation("clean", ())
    attestation = build_parent_attestation(
        request_nonce="nonce-1",
        request_sha256="sha256:" + "1" * 64,
        completion=None,
        observation=observation,
    )
    assert attestation["attestation_version"] == "menagerie.crawler.parent-attestation.v3"
    assert attestation["group_teardown_outcome"] == "clean"
    assert attestation["group_survivor_pids"] == []
    assert set(attestation) == _PARENT_ATTESTATION_KEYS_V3
    # Tampering the bound facts breaks the content address.
    forged = dict(attestation)
    forged["group_teardown_outcome"] = "survivors"
    unhashed = {key: value for key, value in forged.items() if key != "attestation_sha256"}
    assert stable_hash(unhashed) != forged["attestation_sha256"]
    assert _PARENT_ATTESTATION_KEYS_V3 - _PARENT_ATTESTATION_KEYS == {
        "group_teardown_outcome",
        "group_survivor_pids",
    }


def test_awardable_teardown_set_is_pinned() -> None:
    """Only clean and reaped teardowns may attest; widening is a reviewed change."""

    assert _AWARDABLE_TEARDOWN_OUTCOMES == frozenset({"clean", "reaped"})


@pytest.mark.parametrize("outcome", ["clean", "reaped"])
def test_replay_accepts_a_success_over_an_awardable_teardown(outcome: str) -> None:
    """The authority replay accepts exactly the awardable teardown outcomes."""

    attestation, line = _success_attestation(outcome, [])
    raw = _raw_receipt()
    assert isinstance(_validate_parent_attestation(raw, attestation, line), str)


@pytest.mark.parametrize(
    ("outcome", "survivors"),
    [
        ("survivors", [4242]),
        ("unverifiable", []),
        ("foreign-group", []),
        (None, []),
        ("clean", [4242]),
    ],
)
def test_replay_refuses_a_survivor_forged_success(outcome: Any, survivors: list[int]) -> None:
    """A success over an unattested teardown is refused at replay, typed."""

    attestation, line = _success_attestation(outcome, survivors)
    raw = _raw_receipt()
    with pytest.raises(AuthorityDerivationError, match="teardown cannot attest a success"):
        _validate_parent_attestation(raw, attestation, line)


def test_historical_v2_attestation_stays_readable() -> None:
    """Pre-rotation v2 attestations (no teardown facts) still validate."""

    attestation, line = _success_attestation("clean", [])
    v2 = {
        key: value
        for key, value in attestation.items()
        if key not in {"group_teardown_outcome", "group_survivor_pids", "attestation_sha256"}
    }
    v2["attestation_version"] = "menagerie.crawler.parent-attestation.v2"
    v2["attestation_sha256"] = stable_hash(v2)
    raw = _raw_receipt()
    assert isinstance(_validate_parent_attestation(raw, v2, line), str)
    # An unknown version stays refused.
    v1 = dict(v2)
    v1.pop("attestation_sha256")
    v1["attestation_version"] = "menagerie.crawler.parent-attestation.v1"
    v1["attestation_sha256"] = stable_hash(v1)
    with pytest.raises(AuthorityDerivationError, match="wrong protocol version"):
        _validate_parent_attestation(raw, v1, line)
