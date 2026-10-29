"""AB-30 matrix: runner-reason-vocabulary seam (DX-1/2/6..12, lane w2-runner).

The Wave-2 runner lane declares three coupled changes this matrix adjudicates
against the REAL production readers:

* the attempt failure-reason vocabulary gains ``worker-bootstrap-failed``
  (host owner), ``worker-runtime-unavailable`` (host), and
  ``blocked-by-sibling`` (model, derived, never decisive) as ONE change across
  the schema enum, ``FAILURE_REASON_CODES``, and
  ``WORKER_FAILURE_REASON_OWNERS``;
* the parent-attestation version vocabulary is the closed two-member set
  {``parent-attestation.v2`` (read-only history), ``parent-attestation.v3``
  (minted)} where v3 keys = v2 keys + {``group_teardown_outcome``,
  ``group_survivor_pids``}; a recorded v2 row re-verifies against its own
  bytes unchanged, and a v3 SUCCESS over a non-awardable teardown refuses at
  replay validation;
* the deterministic attempt-slot identity gains ``retry_generation`` where
  generation ZERO keeps the historical identity spelling byte-for-byte
  (pinned era-honesty) and generation > 0 mints a distinct bound identity.

Readers exercised: ``authority.derive_attempt_projection`` (replay-validation
authority over raw receipt + parent attestation), the three reason-vocabulary
authorities, and ``recordio.resolve_attempt_slot_state`` /
``resolve_attempt_slot``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pytest

from menagerie.crawler.authority import (
    AuthorityDerivationError,
    completion_line_for_raw_award_receipt,
    derive_attempt_projection,
    derive_parent_attestation,
    raw_award_receipt_sha256,
)
from menagerie.crawler.constants import FAILURE_REASON_CODES
from menagerie.crawler.driver_receipts import (
    NEVER_MODEL_TERMINAL_OWNERS,
    WORKER_FAILURE_REASON_OWNERS,
)
from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.recordio import (
    AttemptSlotResolutionError,
    deterministic_attempt_id,
    resolve_attempt_slot,
    resolve_attempt_slot_state,
)
from menagerie.crawler.tests.conftest import make_attempt
from menagerie.crawler.tests.mixed_era import assert_era_outcome, classify_reader

pytestmark = pytest.mark.smoke

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
HASH_D = "sha256:" + "d" * 64
STARTED = "2026-07-16T12:00:00Z"
FINISHED = "2026-07-16T12:00:01Z"

#: The closed two-member attestation version vocabulary the seam declares.
V2_VERSION = "menagerie.crawler.parent-attestation.v2"
V3_VERSION = "menagerie.crawler.parent-attestation.v3"

#: The exact recorded pre-batch (v2) attestation key set. Frozen HERE, not
#: imported from production, so a production field-set drift is a red matrix
#: cell instead of a silently moving target.
V2_KEYS = frozenset(
    {
        "attestation_version",
        "request_nonce",
        "request_sha256",
        "completion_line_sha256",
        "named_raw_award_receipt_sha256",
        "exit_code",
        "signal",
        "timed_out",
        "rss_exceeded",
        "peak_rss_bytes",
        "stdout_sha256",
        "stderr_sha256",
        "started_at",
        "finished_at",
        "attestation_sha256",
    }
)
TEARDOWN_KEYS = frozenset({"group_teardown_outcome", "group_survivor_pids"})

#: The three minted reason codes: code -> (stage, declared owner).
NEW_REASON_CODES = {
    "worker-bootstrap-failed": ("runner", "host"),
    "worker-runtime-unavailable": ("runner", "host"),
    "blocked-by-sibling": ("resource", "model"),
}

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "attempt-common.schema.json"


def _observation() -> dict[str, Any]:
    """Build one closed successful raw worker observation (attempt-v3 shape)."""

    return {
        "present": True,
        "receipt_sha256": None,
        "observed_recipe_revision": HASH_B,
        "observed_adapter_sha256": None,
        "observed_code_manifest_sha256": HASH_C,
        "observed_input_asset_sha256": None,
        "constructor_seconds": 0.1,
        "forward_seconds": 0.2,
        "constructor_started": True,
        "constructor_completed": True,
        "input_completed": True,
        "forward_started": True,
        "forward_completed": True,
        "mode": "eval",
        "input_signature": {"args": []},
        "output_signature": {"tree": {"leaf": 0}, "leaves": []},
        "input_kind": "random-fallback",
        "input_asset": None,
        "input_note": "deterministic fixture",
        "parameter_count_total": 2,
        "parameter_count_trainable": 2,
        "native_framework": "torch",
        "delegated_method": "forward",
    }


def _raw_receipt() -> dict[str, Any]:
    """Build one closed v3 raw award receipt for the eval mode."""

    return {
        "receipt_version": "menagerie.crawler.raw-award-receipt.v3",
        "request_nonce": "nonce-eval",
        "request_sha256": HASH_A,
        "stable_id": "m_example",
        "work_id": "work-1",
        "execution_identity": HASH_A,
        "recipe_revision": HASH_B,
        "code_manifest_identity": HASH_C,
        "input_identity": HASH_D,
        "requested_mode": "eval",
        "observation": _observation(),
    }


def _supervisor(
    completion_line: str,
    *,
    teardown_outcome: Optional[str] = "clean",
    survivor_pids: Sequence[int] = (),
) -> dict[str, Any]:
    """Build parent-owned supervisor facts with explicit teardown evidence.

    Parameters
    ----------
    completion_line:
        Exact child completion marker.
    teardown_outcome:
        Parent-observed process-group teardown outcome (verbatim-bound).
    survivor_pids:
        PIDs the parent observed outliving the group teardown.
    """

    return {
        "exit_code": 0,
        "signal": None,
        "wall_seconds": 1.0,
        "cpu_seconds": 0.5,
        "peak_rss_bytes": 100,
        "timed_out": False,
        "rss_exceeded": False,
        "stdout_sha256": HASH_A,
        "stdout_bytes": 10,
        "stdout_tail": "",
        "stdout_completion_line": completion_line,
        "stderr_sha256": HASH_B,
        "stderr_bytes": 0,
        "stderr_tail": "",
        "full_log_local_path": ".crawl-local/logs/worker.log",
        "full_log_retention": "campaign",
        "group_teardown_outcome": teardown_outcome,
        "group_survivor_pids": list(survivor_pids),
    }


def _recorded_v2_attestation(
    raw: Mapping[str, Any], completion_line: str, supervisor: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct one recorded pre-batch v2 attestation from the same facts.

    This is the exact row shape the pre-rotation supervisor minted: the v2
    field set with the self hash sealed over ONLY those bytes -- no teardown
    keys existed. The matrix reads it back through the live replay validator.
    """

    attestation: dict[str, Any] = {
        "attestation_version": V2_VERSION,
        "request_nonce": raw["request_nonce"],
        "request_sha256": raw["request_sha256"],
        "completion_line_sha256": hash_bytes((completion_line + "\n").encode("utf-8")),
        "named_raw_award_receipt_sha256": raw_award_receipt_sha256(raw),
        "exit_code": supervisor["exit_code"],
        "signal": supervisor["signal"],
        "timed_out": supervisor["timed_out"],
        "rss_exceeded": supervisor["rss_exceeded"],
        "peak_rss_bytes": supervisor["peak_rss_bytes"],
        "stdout_sha256": supervisor["stdout_sha256"],
        "stderr_sha256": supervisor["stderr_sha256"],
        "started_at": STARTED,
        "finished_at": FINISHED,
    }
    attestation["attestation_sha256"] = stable_hash(attestation)
    return attestation


def _minted_v3_attestation(
    *, teardown_outcome: Optional[str] = "clean", survivor_pids: Sequence[int] = ()
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Mint one v3 attestation through the REAL production deriver.

    Returns
    -------
    tuple[dict, dict, str]
        ``(raw_receipt, attestation, completion_line)``.
    """

    raw = _raw_receipt()
    line = completion_line_for_raw_award_receipt(raw)
    supervisor = _supervisor(
        line, teardown_outcome=teardown_outcome, survivor_pids=survivor_pids
    )
    attestation = derive_parent_attestation(
        raw, line, supervisor, started_at=STARTED, finished_at=FINISHED
    )
    return raw, attestation, line


# ---------------------------------------------------------------------------
# Cells 1-4: the parent-attestation version vocabulary at the replay validator
# ---------------------------------------------------------------------------


def test_recorded_v2_attestation_reverifies_against_its_own_bytes() -> None:
    """Cell 1: a recorded pre-batch v2 row stays era-scoped readable history.

    The live replay validator must re-verify the v2 self hash over exactly the
    recorded v2 bytes -- never demand the minted-era teardown keys from a row
    whose era could not have carried them.
    """

    raw = _raw_receipt()
    line = completion_line_for_raw_award_receipt(raw)
    v2 = _recorded_v2_attestation(raw, line, _supervisor(line))
    assert set(v2) == V2_KEYS

    outcome = classify_reader(
        lambda: derive_attempt_projection(raw, v2, completion_line=line)
    )
    assert_era_outcome(
        reader_id="authority-attempt-replay-validation",
        outcome=outcome,
        prebatch_expected="parsed",
    )
    if outcome[0] == "parsed":
        authority = outcome[1]
        assert authority.raw_award_receipt_sha256 == raw_award_receipt_sha256(raw)
        assert authority.parent_attestation_sha256 == v2["attestation_sha256"]


def test_recorded_v2_attestation_tampered_bytes_refuse_typed() -> None:
    """Era-scoped is not tamper-tolerant: one flipped v2 byte still refuses.

    The self hash must adjudicate the recorded v2 bytes in every era; a
    prior-era version label can never become a hash bypass.
    """

    raw = _raw_receipt()
    line = completion_line_for_raw_award_receipt(raw)
    v2 = _recorded_v2_attestation(raw, line, _supervisor(line))
    v2["peak_rss_bytes"] = 101

    outcome = classify_reader(
        lambda: derive_attempt_projection(raw, v2, completion_line=line)
    )
    assert outcome[0] == "refused-typed"
    assert isinstance(outcome[1], AuthorityDerivationError)
    assert "self hash" in str(outcome[1])


def test_minted_v3_key_set_is_exactly_v2_plus_the_teardown_pair() -> None:
    """Seam (ii) key contract: v3 keys = v2 keys + the two teardown facts."""

    _, attestation, _ = _minted_v3_attestation()
    assert attestation["attestation_version"] == V3_VERSION
    assert set(attestation) == V2_KEYS | TEARDOWN_KEYS


@pytest.mark.parametrize("teardown_outcome", ["clean", "reaped"])
def test_minted_v3_with_awardable_teardown_validates(teardown_outcome: str) -> None:
    """Cell 2: an awardable teardown with zero survivors parses as minted."""

    raw, attestation, line = _minted_v3_attestation(teardown_outcome=teardown_outcome)
    assert attestation["group_teardown_outcome"] == teardown_outcome
    assert attestation["group_survivor_pids"] == []

    outcome = classify_reader(
        lambda: derive_attempt_projection(raw, attestation, completion_line=line)
    )
    assert outcome[0] == "parsed"
    assert outcome[1].parent_attestation_sha256 == attestation["attestation_sha256"]


def test_unknown_attestation_version_refuses_typed() -> None:
    """Cell 3: the version vocabulary is closed -- v4 refuses even self-sealed."""

    raw, attestation, line = _minted_v3_attestation()
    future = {key: value for key, value in attestation.items() if key != "attestation_sha256"}
    future["attestation_version"] = "menagerie.crawler.parent-attestation.v4"
    # Re-seal so ONLY the closed version vocabulary can be the refusing
    # authority, never an incidental hash mismatch.
    future["attestation_sha256"] = stable_hash(future)

    outcome = classify_reader(
        lambda: derive_attempt_projection(raw, future, completion_line=line)
    )
    assert outcome[0] == "refused-typed"
    assert isinstance(outcome[1], AuthorityDerivationError)
    assert "protocol version" in str(outcome[1])


@pytest.mark.parametrize(
    ("teardown_outcome", "survivor_pids"),
    [
        pytest.param("escaped", (), id="non-awardable-outcome"),
        pytest.param("clean", (4242,), id="survivors-present"),
        pytest.param(None, (), id="unverified-teardown"),
    ],
)
def test_v3_success_over_non_awardable_teardown_refuses_at_replay(
    teardown_outcome: Optional[str], survivor_pids: tuple[int, ...]
) -> None:
    """Cell 4: a v3 SUCCESS cannot stand on an unattested group teardown.

    The producer binds the teardown facts verbatim; the REFUSAL authority is
    replay validation, exactly as the seam declares.
    """

    raw, attestation, line = _minted_v3_attestation(
        teardown_outcome=teardown_outcome, survivor_pids=survivor_pids
    )
    # The deriver records the facts verbatim (producer side never adjudicates).
    assert attestation["group_teardown_outcome"] == teardown_outcome
    assert attestation["group_survivor_pids"] == list(survivor_pids)

    outcome = classify_reader(
        lambda: derive_attempt_projection(raw, attestation, completion_line=line)
    )
    assert outcome[0] == "refused-typed"
    assert isinstance(outcome[1], AuthorityDerivationError)
    assert "teardown" in str(outcome[1])


# ---------------------------------------------------------------------------
# Cell 5: the three new reason codes across all three vocabulary authorities
# ---------------------------------------------------------------------------


def test_new_reason_codes_are_cross_pinned_across_all_three_authorities() -> None:
    """Cell 5: schema enum, FAILURE_REASON_CODES, and owner map moved as one.

    Each minted code must be simultaneously recordable (schema enum),
    reducer-appendable at its exact stage (``FAILURE_REASON_CODES``), and
    owner-bound (``WORKER_FAILURE_REASON_OWNERS``) -- a one-sided registration
    is the BL-12 quarantine-loop recurrence.
    """

    schema_reasons = set(
        json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))["$defs"]["reason_code"]["enum"]
    )
    for code, (stage, owner) in NEW_REASON_CODES.items():
        assert code in schema_reasons, f"{code}: unrecordable (missing from the schema enum)"
        assert code in FAILURE_REASON_CODES[stage], (
            f"{code}: not reducer-appendable at stage {stage!r}"
        )
        assert WORKER_FAILURE_REASON_OWNERS[code] == owner, (
            f"{code}: owner drifted from the declared {owner!r}"
        )
    # Ownership semantics: the two host codes can never terminalize a model;
    # the derived sibling code stays a model fact (and is never decisive on
    # its own -- adjudicated by the DX-7 award-composition tests, not here).
    assert "host" in NEVER_MODEL_TERMINAL_OWNERS
    assert "model" not in NEVER_MODEL_TERMINAL_OWNERS


# ---------------------------------------------------------------------------
# Cell 6: attempt-slot identity era-honesty across retry generations
# ---------------------------------------------------------------------------

_SLOT = {
    "work_id": "work-m_example",
    "execution_identity": "sha256:" + "e" * 64,
    "cold_index": 0,
    "mode": "eval",
}


def _non_model(error: Mapping[str, Any]) -> bool:
    """The driver-supplied DX-10 predicate, spelled from the owner vocabulary."""

    owner = WORKER_FAILURE_REASON_OWNERS.get(str(error.get("reason_code")))
    return owner in NEVER_MODEL_TERMINAL_OWNERS


def _machinery_weather_row(attempt_id: str) -> dict[str, Any]:
    """Build one schema-valid machinery-owned failed row for the exact slot."""

    attempt = make_attempt(
        attempt_id=attempt_id, execution_identity=_SLOT["execution_identity"]
    )
    attempt["result"] = "failed"
    attempt["error"] = {
        "stage": "runner",
        "reason_code": "internal-error",
        "exception_type": "builtins.RuntimeError",
        "message": "synthetic harness defect",
        "traceback": None,
        "no_traceback_reason": "synthetic fixture",
        "native_crash": False,
        "root_cause_fingerprint": "sha256:" + "d" * 64,
        "details": {"owner": "machinery"},
    }
    return attempt


def test_generation_zero_identity_is_byte_identical_to_the_legacy_spelling() -> None:
    """Cell 6a: generation zero keeps the pre-batch identity bytes exactly.

    The legacy spelling is recomputed INDEPENDENTLY here (the 4-key stable
    hash with no ``retry_generation`` member), so a producer that started
    hashing a fifth key at generation zero would orphan every recorded row
    and fail this pin.
    """

    legacy_spelling = stable_hash(
        {
            "work_id": _SLOT["work_id"],
            "execution_identity": _SLOT["execution_identity"],
            "cold_index": _SLOT["cold_index"],
            "mode": _SLOT["mode"],
        }
    )
    assert deterministic_attempt_id(**_SLOT) == legacy_spelling
    assert deterministic_attempt_id(**_SLOT, retry_generation=0) == legacy_spelling

    # A recorded generation-zero row resolves identically through the new
    # generation-aware resolver and the legacy resolver.
    recorded = make_attempt(
        attempt_id=legacy_spelling, execution_identity=_SLOT["execution_identity"]
    )
    state = resolve_attempt_slot_state((recorded,), **_SLOT, non_model_failure=_non_model)
    assert state.retry_generation == 0
    assert state.consumed_non_model_rows == ()
    assert state.record is not None
    assert state.record == resolve_attempt_slot((recorded,), **_SLOT)
    assert state.record["attempt_id"] == legacy_spelling


def test_retry_generation_mints_a_distinct_bound_identity() -> None:
    """Cell 6b: generation > 0 is a NEW deterministic identity, bound typed."""

    generation_zero = deterministic_attempt_id(**_SLOT)
    generation_one = deterministic_attempt_id(**_SLOT, retry_generation=1)
    assert generation_one != generation_zero
    assert generation_one == stable_hash({**_SLOT, "retry_generation": 1})

    # Walking past one machinery-owned weather row lands exactly there.
    weather = _machinery_weather_row(generation_zero)
    honest = make_attempt(
        attempt_id=generation_one, execution_identity=_SLOT["execution_identity"]
    )
    honest["retries"] = {**honest["retries"], "slot_retry_generation": 1}
    state = resolve_attempt_slot_state(
        (weather, honest), **_SLOT, non_model_failure=_non_model
    )
    assert state.record is not None
    assert state.record["attempt_id"] == generation_one
    assert state.retry_generation == 1
    assert state.consumed_non_model_rows == (generation_zero,)


@pytest.mark.parametrize("forge", ["counter-omitted", "counter-inflated"])
def test_forged_carried_retry_counter_refuses_typed(forge: str) -> None:
    """Cell 6c: a carried counter disagreeing with its binding identity refuses.

    Both directions: a generation-1 identity whose row omits the carried
    counter, and a generation-zero identity smuggling a nonzero counter.
    """

    generation_zero = deterministic_attempt_id(**_SLOT)
    if forge == "counter-omitted":
        weather = _machinery_weather_row(generation_zero)
        forged = make_attempt(
            attempt_id=deterministic_attempt_id(**_SLOT, retry_generation=1),
            execution_identity=_SLOT["execution_identity"],
        )
        rows: tuple[dict[str, Any], ...] = (weather, forged)
    else:
        forged = make_attempt(
            attempt_id=generation_zero, execution_identity=_SLOT["execution_identity"]
        )
        forged["retries"] = {**forged["retries"], "slot_retry_generation": 1}
        rows = (forged,)

    outcome = classify_reader(
        lambda: resolve_attempt_slot_state(rows, **_SLOT, non_model_failure=_non_model)
    )
    assert outcome[0] == "refused-typed"
    assert isinstance(outcome[1], AttemptSlotResolutionError)
    assert "contradicts its exact slot" in str(outcome[1])
