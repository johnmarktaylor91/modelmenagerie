"""W44 SHADOW-EXCLUSION RIDER: rehearsal records never confer reuse authority.

R6b's shadow-accuracy rehearsal (W3's third mode) writes REAL records through
the production machinery, each carrying the structural ``shadow_rehearsal``
marker ON THE RECORD (never an environment or mode flag). A rehearsal verdict
must never fire reuse, and a rehearsal artifact must never serve as canonical
authority: this module pins the exclusion at every WS6 reuse/authority seam --
the verdict-reuse predicate (the same predicate DX-59 would later re-arm; the
exclusion lands regardless of DX-59), the artifact checkpoint fold, the reducer
gate accumulator, and the IC-4 closed cache inventory's per-row declaration.

SEAM NOTE (Wave-2 batch): the marker FIELD itself is W3's schema half (WS1) --
today's schemas refuse unknown fields, so no persisted record can carry the
marker until that half merges in the same rotation batch. The single point of
convergence is :data:`menagerie.crawler.identity.SHADOW_REHEARSAL_MARKER_FIELD`;
W3's schema field must bind that exact name.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import menagerie.crawler.gate_reuse as gate_reuse_module
from menagerie.crawler.artifact_transactions import _fold_transaction_chain
from menagerie.crawler.checker_calibration import (
    CALIBRATION_BYPASS_CENSUS_ARM,
    CHECKER_CALIBRATION_RECEIPT_ENV,
    CHECKER_CALIBRATION_RUN_ENV,
    CHECKER_GOLDEN_CORPUS_ENV,
    SHADOW_REHEARSAL_CENSUS_ARM,
    SHADOW_REHEARSAL_RUN_ENV,
    AmbiguousLegModeError,
    CheckerCalibrationWithheldError,
    calibration_run_active,
    evaluate_checker_calibration_mode,
    shadow_rehearsal_census,
    shadow_rehearsal_namespace,
    shadow_rehearsal_run_active,
)
from menagerie.crawler.gate_reuse import reusable_metadata_gate
from menagerie.crawler.identity import (
    SHADOW_REHEARSAL_MARKER_FIELD,
    is_shadow_rehearsal_record,
)
from menagerie.crawler.reducer import CanonicalReducer, ReductionError
from menagerie.crawler.tests.conftest import make_gate, make_model
from menagerie.crawler.tests.test_artifact_chain_spent import _publish_public_chain
from menagerie.crawler.tests.test_calibration_withholding import (
    _award_scenario,
    _paths as _withholding_paths,
)
from menagerie.crawler.tests.test_reducer import CanonicalReducer as HarnessReducer
from menagerie.crawler.tests.test_checker_calibration_mode_arm import (
    PINNED,
    _calibrated_env,
)
from menagerie.crawler.tests.test_gate_verdict_reuse import (
    _artifact,
    _identity,
    _prior_metadata_gate,
    _reseal,
)

pytestmark = pytest.mark.smoke


# -- the ONE marker predicate --------------------------------------------------


def test_shadow_marker_predicate_is_the_single_closed_authority():
    """The marker is one named field read by one predicate, fail-closed on any
    present truthy value; absent or explicitly-false records are not shadow."""

    assert SHADOW_REHEARSAL_MARKER_FIELD == "shadow_rehearsal"
    assert is_shadow_rehearsal_record({}) is False
    assert is_shadow_rehearsal_record({"shadow_rehearsal": False}) is False
    assert is_shadow_rehearsal_record({"shadow_rehearsal": None}) is False
    assert is_shadow_rehearsal_record({"shadow_rehearsal": True}) is True
    # Fail-closed: a marker whose value this code does not recognize still
    # marks the record (a rehearsal namespace string, a future enum, ...).
    assert is_shadow_rehearsal_record({"shadow_rehearsal": "r6b-gen-1"}) is True


# -- artifact-identity reuse: the checkpoint fold ------------------------------


def _fold(chain, *, canonical, repo):
    registries = {
        "object_rows": {},
        "custody_claims": {},
        "accepted_claims": {},
        "transactions": {},
        "stale_transactions": {},
    }
    _fold_transaction_chain(
        str(chain[0]["transaction_id"]),
        chain,
        canonical_root=canonical,
        repository_root=repo,
        **registries,
    )
    return registries


def test_shadow_marked_transaction_quarantines_out_of_reusable_authority(tmp_path):
    """A chain whose events carry the marker folds as a DISCLOSED stale entry
    (cache miss, claims retained for record cross-checks) and never projects
    reusable authority; the unmarked control chain still projects."""

    ledger, _mirrors, canonical, repo = _publish_public_chain(tmp_path, "m_shadow")
    with ledger:
        chain = [dict(event) for event in ledger.events]
    assert [event["event_kind"] for event in chain] == [
        "staged-private",
        "publication-authorized",
        "reconstruction-committed",
        "published",
    ]

    control = _fold(deepcopy(chain), canonical=canonical, repo=repo)
    assert control["transactions"], "control chain must project reusable authority"
    assert not control["stale_transactions"]

    marked = deepcopy(chain)
    marked[-1][SHADOW_REHEARSAL_MARKER_FIELD] = True
    shadow = _fold(marked, canonical=canonical, repo=repo)
    assert not shadow["transactions"]
    (stale,) = shadow["stale_transactions"].values()
    assert "shadow-rehearsal" in stale.reason
    assert stale.claims, "accepted claims stay disclosed for record cross-checks"
    # Physical custody stays attested: rehearsal bytes are never CAS litter.
    assert shadow["object_rows"] and shadow["accepted_claims"]


# -- the reducer gate accumulator ----------------------------------------------


def test_shadow_marked_gate_never_accumulates_current_authority():
    """The current-proof gate accumulator excludes marked rows BEFORE proof
    replay; unmarked rows still reach the proof tripwire (stays armed)."""

    from menagerie.crawler.constants import GATE_SCHEMA_VERSION_V3

    class _Gates:
        def __init__(self, records):
            self._records = records

        def iter_records(self):
            return iter(self._records)

    reducer = CanonicalReducer.__new__(CanonicalReducer)
    marked = {
        "schema_version": GATE_SCHEMA_VERSION_V3,
        SHADOW_REHEARSAL_MARKER_FIELD: True,
        "gate_id": "gate-shadow",
    }
    reducer._gates = _Gates([marked])
    assert reducer._current_gate_records() == ()

    unmarked_bad_proof = {
        "schema_version": GATE_SCHEMA_VERSION_V3,
        "gate_id": "gate-live",
    }
    reducer._gates = _Gates([unmarked_bad_proof])
    with pytest.raises(ReductionError):
        reducer._current_gate_records()


# -- the verdict-reuse predicate seam -------------------------------------------


def test_shadow_marked_latest_row_buys_live_even_if_reuse_is_rearmed(
    tmp_path, monkeypatch
):
    """Under a hypothetical DX-59 re-arm, a shadow-marked latest adjudication
    is INELIGIBLE and decides live -- never reused, never walked past into
    older history. The unmarked control guards against vacuity."""

    artifacts = (_artifact(tmp_path),)
    rejection = _prior_metadata_gate(tmp_path, artifacts, verdict="inaccurate")
    monkeypatch.setattr(gate_reuse_module, "GATE_VERDICT_REUSE_RETIRED", False)

    control = reusable_metadata_gate(artifacts, (rejection,), checker=_identity())
    assert control is not None, "vacuity guard: an unmarked row must be reusable here"

    shadow = deepcopy(rejection)
    shadow[SHADOW_REHEARSAL_MARKER_FIELD] = True
    shadow = _reseal(shadow)
    assert reusable_metadata_gate(artifacts, (shadow,), checker=_identity()) is None

    # And with retirement back in force (the shipped state), still no reuse.
    monkeypatch.setattr(gate_reuse_module, "GATE_VERDICT_REUSE_RETIRED", True)
    assert reusable_metadata_gate(artifacts, (shadow,), checker=_identity()) is None


# -- the award-issuance seam (W3's rider on the AT batch) ------------------------


def test_shadow_marked_candidate_refuses_award_issuance_with_all_mode_flags_unset(
    monkeypatch,
):
    """W3 RED pair (a): a shadow-MARKED candidate refuses award issuance keyed
    on the record's own marker, with every mode flag unset; the unmarked
    control passes the guard untouched."""

    from menagerie.crawler.reducer import (
        CheckerCalibrationWithheldError,
        _withhold_award_issuance_in_calibration_mode,
    )

    monkeypatch.delenv("MENAGERIE_CHECKER_CALIBRATION_RUN", raising=False)
    runs_award = {
        "stable_id": "m_shadow_award",
        "status": {"kind": "runs", "code": "runs"},
    }
    _withhold_award_issuance_in_calibration_mode(dict(runs_award))  # control: no raise

    marked = {**runs_award, SHADOW_REHEARSAL_MARKER_FIELD: True}
    with pytest.raises(CheckerCalibrationWithheldError, match="shadow-rehearsal"):
        _withhold_award_issuance_in_calibration_mode(marked)

    eligible = {
        "stable_id": "m_shadow_award",
        SHADOW_REHEARSAL_MARKER_FIELD: "r6b-gen-1",
        "status": {"kind": "terminal", "code": "failed:runner"},
        "completeness": {"release_eligible": True},
    }
    with pytest.raises(CheckerCalibrationWithheldError, match="release-eligible"):
        _withhold_award_issuance_in_calibration_mode(eligible)


# -- the IC-4 closed cache inventory --------------------------------------------


def test_every_registered_cache_declares_its_shadow_exclusion():
    """Every row of the closed cache inventory names its shadow-exclusion
    mechanism. IC-4's census refuses unregistered caches, so a NEW cache
    cannot appear without stating how shadow-marked records are excluded."""

    import json

    inventory_path = (
        Path(__file__).resolve().parents[1] / "cache_inventory.json"
    )
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    missing = [
        row["id"]
        for row in inventory["caches"]
        if not (isinstance(row.get("shadow_exclusion"), str) and row["shadow_exclusion"].strip())
    ]
    assert not missing, f"cache rows lacking a shadow_exclusion declaration: {missing}"


# =============================================================================
# W3 THIRD MODE (shadow_rehearsal_active) + the WRITE-SIDE producer.
#
# Everything above pins the READ-side exclusion (W44). Everything below pins
# the runtime mode itself: the closed three-value mode vocabulary, the
# receipt-bearing XOR arm, the mode-level award withhold (marker or no
# marker), the per-seam marker stamping with the declared rehearsal
# generation namespace, the semantic-replay exemption, and the segregated
# disclosure census.
# =============================================================================

NAMESPACE = "r6b-gen-1"


def _shadow_mode(monkeypatch) -> None:
    """Declare the shadow-rehearsal mode alone (never alongside calibration)."""

    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    monkeypatch.setenv(SHADOW_REHEARSAL_RUN_ENV, NAMESPACE)


# -- the closed mode vocabulary ---------------------------------------------


def test_declaring_calibration_and_shadow_together_refuses_every_direct_reader(
    monkeypatch,
):
    """Mode vocabulary is CLOSED at three: both declared is ambiguous and every
    direct mode reader refuses typed before any mode-conditioned behavior."""

    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    monkeypatch.setenv(SHADOW_REHEARSAL_RUN_ENV, NAMESPACE)
    with pytest.raises(AmbiguousLegModeError, match="ambiguous leg mode"):
        calibration_run_active()
    with pytest.raises(AmbiguousLegModeError):
        shadow_rehearsal_run_active()
    with pytest.raises(AmbiguousLegModeError):
        shadow_rehearsal_namespace()


def test_shadow_accessors_read_the_declared_namespace(monkeypatch):
    _shadow_mode(monkeypatch)
    assert shadow_rehearsal_run_active() is True
    assert shadow_rehearsal_namespace() == NAMESPACE
    monkeypatch.delenv(SHADOW_REHEARSAL_RUN_ENV, raising=False)
    assert shadow_rehearsal_run_active() is False
    assert shadow_rehearsal_namespace() is None


# -- the W2 receipt-XOR preflight arm ----------------------------------------


def test_xor_arm_shadow_mode_with_covering_receipt_passes(tmp_path, monkeypatch):
    """RECEIPT-BEARING: shadow mode rides the receipt side of the XOR --
    shadow+receipt is NOT ambiguous (unlike calibration+receipt)."""

    _shadow_mode(monkeypatch)
    _calibrated_env(tmp_path, monkeypatch)
    result = evaluate_checker_calibration_mode(
        renders_gate_verdicts=True, pinned_models=PINNED
    )
    assert result["ok"] is True and result["refusals"] == []
    assert result["shadow_rehearsal_active"] is True
    assert result["environment"]["shadow_rehearsal_run"] == NAMESPACE


def test_xor_arm_shadow_mode_without_receipt_refuses_verdict_rendering(monkeypatch):
    """Shadow mode never substitutes for the receipt: a verdict-rendering leg
    under shadow mode alone is the uncalibrated launch and refuses."""

    _shadow_mode(monkeypatch)
    monkeypatch.setenv(CHECKER_CALIBRATION_RECEIPT_ENV, "/nonexistent/receipt.json")
    result = evaluate_checker_calibration_mode(
        renders_gate_verdicts=True, pinned_models=PINNED
    )
    assert result["ok"] is False
    assert "receipt-bearing" in result["refusals"][0]


def test_xor_arm_both_modes_declared_refuses_as_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setenv(CHECKER_CALIBRATION_RUN_ENV, "1")
    monkeypatch.setenv(SHADOW_REHEARSAL_RUN_ENV, NAMESPACE)
    _calibrated_env(tmp_path, monkeypatch)
    result = evaluate_checker_calibration_mode(
        renders_gate_verdicts=True, pinned_models=PINNED
    )
    assert result["ok"] is False
    assert "ambiguous leg mode" in result["refusals"][0]


# -- the mode-level award withhold (marker or NO marker) ----------------------


def test_runs_award_append_refuses_while_shadow_mode_active(tmp_path, monkeypatch):
    """W3 RED (a): a live runs-award append refuses typed under the mode.

    Belt and braces at the real seam: the model door stamps the candidate, so
    the marker-based rider fires; the mode-level withhold behind it is pinned
    separately below for a candidate that carries NO marker."""

    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    monkeypatch.delenv(SHADOW_REHEARSAL_RUN_ENV, raising=False)
    with HarnessReducer(_withholding_paths(tmp_path), ["m_example"]) as reducer:
        candidate = _award_scenario(reducer, accepted=True)
        monkeypatch.setenv(SHADOW_REHEARSAL_RUN_ENV, NAMESPACE)
        with pytest.raises(CheckerCalibrationWithheldError, match="withholds awards"):
            reducer.append_model(candidate)


def test_mode_withholds_a_runs_award_that_carries_no_marker(monkeypatch):
    """W3 RED (a, mode half): the MODE alone withholds -- the candidate carries
    no marker at all, mirroring the calibration_run_active() withhold."""

    from menagerie.crawler.reducer import _withhold_award_issuance_in_calibration_mode

    _shadow_mode(monkeypatch)
    runs_award = {
        "stable_id": "m_shadow_mode",
        "status": {"kind": "runs", "code": "runs"},
    }
    with pytest.raises(CheckerCalibrationWithheldError, match="shadow-rehearsal mode"):
        _withhold_award_issuance_in_calibration_mode(runs_award)


def test_release_eligible_marking_refuses_while_shadow_mode_active(monkeypatch):
    """W3 RED (b): release-eligible marking withholds under the mode, keyed on
    the MODE alone -- the candidate carries no marker."""

    from menagerie.crawler.reducer import _withhold_award_issuance_in_calibration_mode

    _shadow_mode(monkeypatch)
    eligible = {
        "stable_id": "m_shadow_mode",
        "status": {"kind": "terminal", "code": "failed:runner"},
        "completeness": {"release_eligible": True},
    }
    with pytest.raises(CheckerCalibrationWithheldError, match="release-eligible"):
        _withhold_award_issuance_in_calibration_mode(eligible)


def test_award_restoration_refuses_while_shadow_mode_active(tmp_path, monkeypatch):
    """W3 RED (c): the restoration mint path is withheld before any ledger scan."""

    _shadow_mode(monkeypatch)
    with HarnessReducer(_withholding_paths(tmp_path), ["m_example"]) as reducer:
        with pytest.raises(CheckerCalibrationWithheldError, match="shadow-rehearsal mode"):
            reducer.append_award_restoration(
                stable_id="m_example",
                restored_award_revision="sha256:" + "a" * 64,
                superseding_revision="sha256:" + "b" * 64,
                restoration_event_id="restoration-1",
            )


# -- the write-side producer: every stamped seam ------------------------------


def test_model_and_attempt_seams_stamp_the_namespace(tmp_path, monkeypatch):
    """W3 RED (d): a shadow leg still records non-award terminals, and both the
    model head and the durable attempt row carry the namespace string."""

    from menagerie.crawler.recordio import scan_jsonl
    from menagerie.crawler.tests.conftest import bind_terminal_attempts, make_failed_attempt

    _shadow_mode(monkeypatch)
    attempt = make_failed_attempt(stage="forward", reason_code="exception")
    model = make_model(status_code="failed:forward")
    model["status"]["reason_code"] = "exception"
    bind_terminal_attempts(model, [attempt])
    paths = _withholding_paths(tmp_path)
    with HarnessReducer(paths, ["m_example"]) as reducer:
        reducer.append_attempt(attempt)
        head = reducer.append_model(model).record
    assert head[SHADOW_REHEARSAL_MARKER_FIELD] == NAMESPACE
    (attempt_row,) = scan_jsonl(paths.attempts)
    assert attempt_row[SHADOW_REHEARSAL_MARKER_FIELD] == NAMESPACE
    (model_row,) = scan_jsonl(paths.models)
    assert model_row[SHADOW_REHEARSAL_MARKER_FIELD] == NAMESPACE


def test_gate_seal_stamps_inside_the_result_envelope_hash(tmp_path, monkeypatch):
    """The gate marker is machine-owned, sealed INSIDE result_envelope_sha256
    at the dispatch scaffold, and round-trips full validation."""

    from menagerie.crawler.checker_dispatch import (
        CheckerDispatchError,
        build_metadata_vet_envelope,
        validate_checker_result_mapping,
    )
    from menagerie.crawler.constants import METADATA_BATCH_MAX
    from menagerie.crawler.tests.test_slice_d_checker_gates import (
        _checker_item_pack,
        _stamped,
    )

    _shadow_mode(monkeypatch)
    gate = make_gate([f"m_{index}" for index in range(METADATA_BATCH_MAX)])
    items = [_checker_item_pack(item) for item in gate["items"]]
    # The checker copies the envelope's row-derived ``verified_hashes`` verbatim,
    # so the gate echo carries the packs' digests, never the fixture's
    # placeholders (the machine digest attestation rebinds them at pack build).
    for gate_item, pack in zip(gate["items"], items):
        gate_item["verified_hashes"] = deepcopy(pack["verified_hashes"])
    envelope = build_metadata_vet_envelope(
        items,
        gate_round=1,
        output_path=tmp_path / "result.json",
        checker_model="codex",
        checker_version="test",
        request_nonce="fresh-shadow-1",
    )
    sealed = _stamped(gate, envelope)
    assert sealed[SHADOW_REHEARSAL_MARKER_FIELD] == NAMESPACE
    # The self-hash covers the marker: full validation replays it exactly.
    validated = validate_checker_result_mapping(sealed, envelope)
    assert validated[SHADOW_REHEARSAL_MARKER_FIELD] == NAMESPACE

    # A checker SUPPLYING the machine-owned marker is refused on presence.
    supplied = dict(gate)
    supplied[SHADOW_REHEARSAL_MARKER_FIELD] = NAMESPACE
    with pytest.raises(CheckerDispatchError, match="machine-owned"):
        _stamped(supplied, envelope)


def test_gate_door_refuses_an_unmarked_gate_under_shadow_mode(tmp_path, monkeypatch):
    """A gate whose marker was not sealed at the scaffold refuses at the
    reducer door instead of being silently stamped after its self-hash."""

    monkeypatch.delenv(SHADOW_REHEARSAL_RUN_ENV, raising=False)
    unmarked = make_gate(["m_example"])
    with HarnessReducer(_withholding_paths(tmp_path), ["m_example"]) as reducer:
        monkeypatch.setenv(SHADOW_REHEARSAL_RUN_ENV, NAMESPACE)
        with pytest.raises(ReductionError, match="unmarked gate"):
            reducer.append_gate(unmarked)


def test_artifact_event_mint_stamps_inside_the_event_identity(tmp_path, monkeypatch):
    """Every event of a chain minted under the mode carries the namespace, and
    the chain's intrinsic id validation replays clean (the marker is inside
    artifact_event_id, stamped at the mint, before the id derivation)."""

    _shadow_mode(monkeypatch)
    ledger, _mirrors, _canonical, _repo = _publish_public_chain(tmp_path, "m_shadow_w3")
    with ledger:
        chain = [dict(event) for event in ledger.events]
    assert len(chain) == 4
    assert all(event[SHADOW_REHEARSAL_MARKER_FIELD] == NAMESPACE for event in chain)


def test_operational_seam_stamps_and_the_leg_attests_the_mode(tmp_path, monkeypatch):
    """The durable JSONL door stamps operational events, and the leg-config
    attestation (calibration_environment_snapshot) carries the shadow read."""

    from menagerie.crawler.constants import OPERATIONAL_EVENT_SCHEMA_VERSION
    from menagerie.crawler.recordio import JsonlLedger, scan_jsonl
    from menagerie.crawler.tests.test_slice_f_driver import (
        _driver,
        _paths as _driver_paths,
        _snapshot,
    )

    _shadow_mode(monkeypatch)
    monkeypatch.delenv(CHECKER_CALIBRATION_RECEIPT_ENV, raising=False)
    monkeypatch.delenv(CHECKER_GOLDEN_CORPUS_ENV, raising=False)
    snapshot = _snapshot(tmp_path)
    paths = _driver_paths(tmp_path, snapshot)
    driver = _driver(tmp_path, snapshot)
    with JsonlLedger(paths.operational_ledger, OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        driver._record_leg_config(operational)  # noqa: SLF001 -- the seam under test
    (event,) = [
        event
        for event in scan_jsonl(paths.operational_ledger)
        if event.get("event_kind") == "leg-config"
    ]
    assert event[SHADOW_REHEARSAL_MARKER_FIELD] == NAMESPACE
    assert event["details"]["checker_calibration"]["shadow_rehearsal_run"] == NAMESPACE


def test_spend_row_mint_stamps_inside_the_row_digest(tmp_path, monkeypatch):
    """Spend rows minted under the mode carry the namespace inside their own
    row_sha256 (the fold and chain replay clean on read-back)."""

    from menagerie.crawler.paid_session_lifecycle import (
        BudgetPolicy,
        PaidSessionAuthority,
        load_paid_session_rows,
    )

    _shadow_mode(monkeypatch)
    authority = PaidSessionAuthority(
        tmp_path / "records" / "operational" / "spend.jsonl",
        runtime_root=tmp_path / ".crawl-local",
        clock=lambda: "2026-08-31T00:00:00Z",
    )
    authority.reserve(
        scope={
            "campaign_id": "c1-shadow",
            "generation_id": "generation-open",
            "lane": "author",
            "dispatch_kind": "author_source",
            "attempt_group": None,
            "work_generations": (("m_shadow_spend", "work-shadow"),),
            "gate_kind": None,
            "checker_identity": None,
            "run_id": "run-shadow",
            "machine_id": "host-shadow",
        },
        policy=BudgetPolicy(
            quota_sessions=10,
            quota_wall_seconds=1_000.0,
            runaway_sessions=10,
            runaway_wall_seconds=1_000.0,
            reservation_wall_seconds=100.0,
        ),
    )
    rows = load_paid_session_rows(tmp_path / "records" / "operational" / "spend.jsonl")
    reserved = [row for row in rows if row.get("event") == "session_reserved"]
    assert reserved, "the reservation must have minted a durable spend row"
    assert all(row[SHADOW_REHEARSAL_MARKER_FIELD] == NAMESPACE for row in reserved)


def test_stamp_helper_is_inert_outside_the_mode_and_refuses_foreign_markers(
    monkeypatch,
):
    from menagerie.crawler.identity import ShadowRehearsalStampError, stamp_shadow_rehearsal

    monkeypatch.delenv(SHADOW_REHEARSAL_RUN_ENV, raising=False)
    record: dict = {"payload": 1}
    assert stamp_shadow_rehearsal(record) == {"payload": 1}
    _shadow_mode(monkeypatch)
    assert stamp_shadow_rehearsal({"payload": 1}) == {
        "payload": 1,
        SHADOW_REHEARSAL_MARKER_FIELD: NAMESPACE,
    }
    with pytest.raises(ShadowRehearsalStampError, match="foreign marker"):
        stamp_shadow_rehearsal({SHADOW_REHEARSAL_MARKER_FIELD: "some-other-generation"})


# -- semantic replay: never stamps, never withholds ---------------------------


def test_semantic_replay_never_stamps_and_readmits_awards_under_the_mode(
    tmp_path, monkeypatch
):
    """W3 RED (g): history replay is exempt -- an issued award re-admits during
    a shadow rehearsal and the replayed record carries NO marker."""

    from menagerie.crawler.reducer import project_dependency_current
    from menagerie.crawler.tests.conftest import make_authority_context

    monkeypatch.delenv(CHECKER_CALIBRATION_RUN_ENV, raising=False)
    monkeypatch.delenv(SHADOW_REHEARSAL_RUN_ENV, raising=False)
    paths = _withholding_paths(tmp_path)
    with HarnessReducer(paths, ["m_example"]) as reducer:
        head = reducer.append_model(_award_scenario(reducer, accepted=True)).record
    monkeypatch.setenv(SHADOW_REHEARSAL_RUN_ENV, NAMESPACE)
    projection = project_dependency_current(
        paths, context=make_authority_context(["m_example"])
    )
    replayed = projection.current_records["m_example"]
    assert replayed["record_revision"] == head["record_revision"]
    assert replayed["status"]["kind"] == "runs"
    assert SHADOW_REHEARSAL_MARKER_FIELD not in replayed


# -- the segregated disclosure census -----------------------------------------


def test_shadow_census_counts_marked_gate_rows_segregated_from_production():
    """W3 RED (h): the census arm is derived from ledger bytes (the rows' own
    markers), counts marked rows into its own segregated arm -- DISTINCT from
    the PD-9 bypass arm -- and never counts them as production-adjudicated."""

    assert SHADOW_REHEARSAL_CENSUS_ARM == "gates_rendered_under_shadow_rehearsal"
    assert SHADOW_REHEARSAL_CENSUS_ARM != CALIBRATION_BYPASS_CENSUS_ARM
    rows = [
        {"gate_id": "gate-live-1"},
        {"gate_id": "gate-shadow-1", SHADOW_REHEARSAL_MARKER_FIELD: NAMESPACE},
        {"gate_id": "gate-shadow-2", SHADOW_REHEARSAL_MARKER_FIELD: NAMESPACE},
    ]
    census = shadow_rehearsal_census(rows)
    assert census["census_arm"] == SHADOW_REHEARSAL_CENSUS_ARM
    assert census["gates_rendered_under_shadow_rehearsal"] == 2
    assert census["production_adjudicated"] == 1
    assert census["shadow_gate_ids"] == ["gate-shadow-1", "gate-shadow-2"]
