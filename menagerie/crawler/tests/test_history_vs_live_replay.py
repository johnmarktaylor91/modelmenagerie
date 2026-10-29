"""Rung-11 history-vs-live family regressions (leg-4 crash chain).

One family, four resume legs, four sites: after a mid-campaign upgrade moves a
shipped-policy content identity, replay and re-drive paths re-derive artifacts
under NEW code and collide with (or refuse) records written under OLD code.
The doctrine (striker-4, ``ecfd3c34``): WRITE time checks the live world;
READ/replay time validates records against their OWN recorded basis; history
is never rewritten and never re-imposed with live-context equality; genuine
contradiction at the recorded basis still refuses.

This module pins the leg-4 sites both ways:

- driver-failure attempt replay reuse (the ``f4626f26`` crash identity),
- terminal-unrecordable occurrence minting (the ``413b6f8a`` conflict),
- diagnostics sidecar first-write-wins,
- recorded-result envelope revalidation at the recorded basis,
- review-held records standing at their recorded rule basis in the
  dependency-currency projection.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import menagerie.crawler.driver as driver_module
import menagerie.crawler.reducer as reducer_module
from menagerie.crawler.author_dispatch import (
    AuthorEngineFaultError,
    build_author_envelope,
    build_recorded_result_envelope,
    validate_author_result_mapping,
)
from menagerie.crawler.constants import (
    AUTHOR_RESULT_SCHEMA_VERSION,
    OPERATIONAL_EVENT_SCHEMA_VERSION,
)
from menagerie.crawler.driver_models import _driver_failure_attempt
from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import LedgerPaths
from menagerie.crawler.recordio import (
    JsonlLedger,
    LedgerConflictError,
    driver_failure_attempt_identity,
)
from menagerie.crawler.reducer import (
    CanonicalReducer,
    ReductionError,
    _revision_work_ids,
    project_dependency_current,
)
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_attempt,
    make_authority_context,
    make_failed_attempt,
    make_model,
)

T1 = "2026-08-07T11:34:43.160429Z"
T2 = "2026-08-07T13:37:00.000000Z"


def _paths(tmp_path: Path) -> LedgerPaths:
    """Return isolated canonical ledger paths below one records root."""

    records = tmp_path / "records"
    return LedgerPaths(
        models=records / "models" / "current-shard.jsonl",
        attempts=records / "attempts" / "local.jsonl",
        gates=records / "gates" / "current-shard.jsonl",
    )


def _failure_attempt(
    tmp_path: Path, *, created_at: str, message: str = "conflicting replay fixture"
) -> dict[str, Any]:
    """Re-derive one driver-observed lane failure through the REAL builder."""

    diagnostics = tmp_path / ".crawl-local" / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    return _driver_failure_attempt(
        SimpleNamespace(  # type: ignore[arg-type]
            stable_id="m_example", active_work_id="work-m_example"
        ),
        None,
        "runner",
        "protocol-violation",
        RuntimeError(message),
        SimpleNamespace(machine_id="machine-test"),  # type: ignore[arg-type]
        diagnostics_root=diagnostics,
        environment=None,
        created_at=created_at,
    )


@pytest.mark.smoke
def test_driver_failure_attempt_replay_reuses_the_recorded_row(tmp_path: Path) -> None:
    """A resumed leg re-observing one failure reuses the frozen attempt.

    Rung-11 leg-4 crash regression: m4256's repeating supersession refusal
    re-derived attempt ``sha256:f4626f26...`` on every resume. The identity is
    time-free, so the replay differed from the frozen row ONLY at
    ``started_at``/``finished_at`` and the parent attestation recomputed over
    them -- and the append killed the campaign with ``LedgerConflictError``.
    The replay must recognize the existing identity and return the recorded
    row exactly as recorded, appending nothing.
    """

    first = _failure_attempt(tmp_path, created_at=T1)
    replay = _failure_attempt(tmp_path, created_at=T2)
    assert first["attempt_id"] == replay["attempt_id"]
    assert first["started_at"] != replay["started_at"]
    with CanonicalReducer(_paths(tmp_path), make_authority_context(["m_example"])) as reducer:
        persisted = reducer.append_attempt(first)
        assert persisted.appended
        reused = reducer.append_attempt(replay)
        assert not reused.appended
        assert reused.record["started_at"] == T1
        assert reused.record["attempt_id"] == first["attempt_id"]
        assert reducer._attempts.record_count == 1


@pytest.mark.smoke
def test_driver_failure_attempt_with_new_inputs_mints_its_own_identity(
    tmp_path: Path,
) -> None:
    """A genuinely new failure (different signature) appends a new row."""

    first = _failure_attempt(tmp_path, created_at=T1)
    different = _failure_attempt(tmp_path, created_at=T2, message="a different root cause")
    assert first["attempt_id"] != different["attempt_id"]
    with CanonicalReducer(_paths(tmp_path), make_authority_context(["m_example"])) as reducer:
        assert reducer.append_attempt(first).appended
        assert reducer.append_attempt(different).appended
        assert reducer._attempts.record_count == 2


@pytest.mark.smoke
def test_attempt_contradicting_the_recorded_basis_still_refuses(tmp_path: Path) -> None:
    """Both contradiction directions keep the ledger's conflict tripwire armed."""

    paths = _paths(tmp_path)
    first = _failure_attempt(tmp_path, created_at=T1)
    with CanonicalReducer(paths, make_authority_context(["m_example"])) as reducer:
        reducer.append_attempt(first)
        # Proposed row disagreeing with the recorded basis under the same
        # claimed identity: no reuse, the conflict refuses.
        forged = _failure_attempt(tmp_path, created_at=T2)
        forged["error"] = dict(forged["error"])
        forged["error"]["root_cause_fingerprint"] = "sha256:" + "e" * 64
        with pytest.raises(LedgerConflictError, match="conflicting replay"):
            reducer.append_attempt(forged)

    # Recorded row whose identity does not re-derive from its own recorded
    # basis: replay reuse is refused and the conflict tripwire fires.
    tampered_root = tmp_path / "tampered"
    tampered_paths = _paths(tampered_root)
    recorded = _failure_attempt(tampered_root, created_at=T1)
    recorded = {
        key: value for key, value in recorded.items() if key not in {"ledger_seq", "payload_sha256"}
    }
    recorded["error"] = dict(recorded["error"])
    recorded["error"]["root_cause_fingerprint"] = "sha256:" + "d" * 64
    assert recorded["attempt_id"] != driver_failure_attempt_identity(
        work_id=str(recorded["work_id"]),
        stage=str(recorded["stage"]),
        reason_code=str(recorded["error"]["reason_code"]),
        root_cause_fingerprint=str(recorded["error"]["root_cause_fingerprint"]),
    )
    with JsonlLedger(tampered_paths.attempts, str(recorded["schema_version"])) as raw_ledger:
        raw_ledger.append(recorded)
    with CanonicalReducer(tampered_paths, make_authority_context(["m_example"])) as reducer:
        replay = _failure_attempt(tampered_root, created_at=T2)
        with pytest.raises(LedgerConflictError, match="conflicting replay"):
            reducer.append_attempt(replay)


@pytest.mark.smoke
def test_diagnostics_sidecar_first_write_wins_for_one_identity(tmp_path: Path) -> None:
    """A replay derivation never rewrites the sidecar the recorded row references."""

    first = _failure_attempt(tmp_path, created_at=T1)
    sidecar = tmp_path / ".crawl-local" / "diagnostics" / f"{first['attempt_id']}.json"
    original_bytes = sidecar.read_bytes()
    _failure_attempt(tmp_path, created_at=T2)
    assert sidecar.read_bytes() == original_bytes
    payload = json.loads(original_bytes)
    assert payload["attempt_id"] == first["attempt_id"]


def _reporter(clock_values: list[str]) -> Any:
    """Build a minimal driver facade for ``_report_unrecordable_terminal``."""

    values = iter(clock_values)
    return SimpleNamespace(
        config=SimpleNamespace(run_id="crawler-run", machine_id="machine-test"),
        dependencies=SimpleNamespace(clock=lambda: next(values)),
        _unrecordable_diagnostics={},
        _unrecordable_streak=[],
    )


@pytest.mark.smoke
def test_unrecordable_terminal_report_mints_a_new_identity_per_occurrence(
    tmp_path: Path,
) -> None:
    """A restarted driver's report never collides with a prior leg's event.

    Rung-11 leg-4 regression (event ``terminal-unrecordable-413b6f8a...``):
    the occurrence number came from a process-local counter, so a resumed
    driver re-minted occurrence 1 for the same repeating failure and the
    fresh ``created_at`` made the replay a conflicting append. The occurrence
    is now derived from the model's durable terminal-unrecordable history, so
    each observation mints a genuinely new identity.
    """

    item = SimpleNamespace(stable_id="m_example", active_work_id="work-m_example")
    error = RuntimeError("human-review terminal supersession requires a new explicit grant")
    with JsonlLedger(tmp_path / "events.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as operational:
        driver_module.CrawlerDriver._report_unrecordable_terminal(
            _reporter([T1]),
            item,
            "failed:accuracy-gate",
            "terminal-disposition-rejected",
            error,
            error,
            operational,
            recorded=False,
        )
        # A fresh facade models the process restart that reset the old counter.
        driver_module.CrawlerDriver._report_unrecordable_terminal(
            _reporter([T2]),
            item,
            "failed:accuracy-gate",
            "terminal-disposition-rejected",
            error,
            error,
            operational,
            recorded=False,
        )
        events = [
            event
            for event in operational.records
            if event.get("event_kind") == "terminal-unrecordable"
        ]
    assert len(events) == 2
    assert events[0]["event_id"] != events[1]["event_id"]
    assert [event["details"]["occurrence"] for event in events] == [1, 2]


def _recorded_bindings() -> dict[str, str]:
    """Return one complete recorded identity binding set."""

    return {
        "author_identity": "sha256:" + "1" * 64,
        "prompt_identity": "sha256:" + "2" * 64,
        "dispatcher_identity": "sha256:" + "3" * 64,
        "intake_snapshot_id": "intake-recorded",
        "intake_snapshot_sha256": "sha256:" + "4" * 64,
        "intake_item_sha256": "sha256:" + "5" * 64,
    }


@pytest.mark.smoke
def test_recorded_result_envelope_binds_the_recorded_basis(tmp_path: Path) -> None:
    """Canonical rehydration validates history at the identities it recorded.

    The live-context envelope re-imposed the ACTIVE prompt/dispatcher/intake
    identities on recorded results, so one author-prompt re-pin refused every
    finalized transaction ("author result prompt_identity does not match its
    envelope") and aborted the campaign at
    ``_rehydrate_final_authority``. The recorded-result envelope stamps
    ``expected_result`` from the transaction's own commitments.
    """

    context = make_authority_context(["m_example"])
    bindings = _recorded_bindings()
    manifest = {"sources": [], "manifest_sha256": stable_hash([])}
    envelope = build_recorded_result_envelope(
        context=context,
        work_id="work-m_example",
        stable_id="m_example",
        campaign_id="campaign-test",
        created_at=T1,
        untrusted_hints={},
        source_manifest=manifest,
        allowed_model_dir=tmp_path,
        output_path=tmp_path / "result.json",
        recorded_bindings=bindings,
    )
    expected = envelope["expected_result"]
    assert expected["prompt_identity"] == bindings["prompt_identity"]
    assert expected["dispatcher_identity"] == bindings["dispatcher_identity"]
    assert expected["author_identity"] == bindings["author_identity"]
    assert expected["intake_snapshot_id"] == bindings["intake_snapshot_id"]
    assert expected["intake_item_sha256"] == bindings["intake_item_sha256"]
    assert expected["prompt_identity"] != context.author_prompt_identity

    # The recorded result revalidates against the recorded basis even though
    # every one of these identities has moved in the live context.
    payload = {
        "arm": "BLOCKED",
        "stage": "source",
        "reason_code": "missing-mandatory-link",
        "prerequisite_ids": ["src-upstream"],
        "evidence_ids": ["ev-source"],
        "evidence_identity": "sha256:" + "a" * 64,
        "license_identity": "sha256:" + "a" * 64,
    }
    payload["recommendation_sha256"] = stable_hash(payload)
    result = {
        "schema_version": AUTHOR_RESULT_SCHEMA_VERSION,
        "result_id": "result-m_example",
        "kind": "BLOCKED",
        "stable_id": "m_example",
        "work_id": "work-m_example",
        "campaign_id": "campaign-test",
        "created_at": T1,
        "author_identity": bindings["author_identity"],
        "prompt_identity": bindings["prompt_identity"],
        "dispatcher_identity": bindings["dispatcher_identity"],
        "source_manifest_identity": manifest["manifest_sha256"],
        "intake_snapshot_id": bindings["intake_snapshot_id"],
        "intake_snapshot_sha256": bindings["intake_snapshot_sha256"],
        "intake_item_sha256": bindings["intake_item_sha256"],
        "payload": payload,
    }
    result["result_sha256"] = stable_hash(result)
    parsed = validate_author_result_mapping(result, envelope)
    assert parsed.binding.result_id == "result-m_example"

    # The DISPATCH envelope still binds the live world for the same inputs.
    live_envelope = build_author_envelope(
        context=context,
        work_id="work-m_example",
        stable_id="m_example",
        campaign_id="campaign-test",
        created_at=T1,
        untrusted_hints={},
        source_manifest=manifest,
        allowed_model_dir=tmp_path,
        output_path=tmp_path / "result.json",
    )
    assert live_envelope["expected_result"]["prompt_identity"] == context.author_prompt_identity


@pytest.mark.smoke
def test_recorded_result_envelope_refuses_incomplete_bindings(tmp_path: Path) -> None:
    """A recorded binding set missing any identity field refuses loudly."""

    context = make_authority_context(["m_example"])
    bindings = _recorded_bindings()
    del bindings["prompt_identity"]
    with pytest.raises(AuthorEngineFaultError, match="recorded result bindings"):
        build_recorded_result_envelope(
            context=context,
            work_id="work-m_example",
            stable_id="m_example",
            campaign_id="campaign-test",
            created_at=T1,
            untrusted_hints={},
            source_manifest={"sources": [], "manifest_sha256": stable_hash([])},
            allowed_model_dir=tmp_path,
            output_path=tmp_path / "result.json",
            recorded_bindings=bindings,
        )


_MOVED_RULE_IDENTITIES = {
    "author_schema_identity": "sha256:" + "6" * 64,
    "author_dispatcher_identity": "sha256:" + "7" * 64,
    "author_prompt_identity": "sha256:" + "8" * 64,
    "checker_prompt_identity": "sha256:" + "9" * 64,
    "terminal_policy_identity": "sha256:" + "a" * 63 + "b",
    "publication_policy_identity": "sha256:" + "b" * 63 + "c",
    "reducer_policy_identity": "sha256:" + "c" * 63 + "d",
    "runner_policy_identity": "sha256:" + "d" * 63 + "e",
}


def _held_terminal(paths: LedgerPaths, context: Any, *, held: bool) -> None:
    """Append one failed:source terminal, optionally held for human review."""

    attempt = make_failed_attempt()
    model = bind_terminal_attempts(make_model(status_code="failed:source"), [attempt])
    if held:
        model["status"]["human_review"] = {
            "required": True,
            "reason": None,
            "queue": "crawler-human-review",
            "requested_at": model["created_at"],
        }
    with CanonicalReducer(paths, context) as reducer:
        reducer.append_attempt(attempt)
        reducer.append_model(reducer.prepare_model(model))


@pytest.mark.smoke
def test_review_held_record_stands_at_its_recorded_rule_basis(tmp_path: Path) -> None:
    """An upgrade never flags a review-held record for a re-drive it cannot record.

    Rung-11 leg-4 regression: batch-15 moved ``publication_policy_identity``,
    the projection flagged every record stale, and the scheduler re-drove all
    21 review-held models -- every one refused at ``human-review terminal
    supersession requires a new explicit grant`` and cascaded into the
    unrecordable-terminal crash chain. A held record's re-adjudication belongs
    to the explicit requeue grant, so it REPLAYS at the rule basis it
    recorded and stays the model's standing disposition.
    """

    context = make_authority_context(["m_example"])
    paths = _paths(tmp_path)
    _held_terminal(paths, context, held=True)
    upgraded = replace(context, **_MOVED_RULE_IDENTITIES)

    projection = project_dependency_current(paths, context=upgraded)

    assert projection.stale_reasons == {}
    assert "m_example" in projection.current_records
    status = projection.current_records["m_example"]["status"]
    assert status["human_review"]["required"] is True


@pytest.mark.smoke
def test_ordinary_record_stands_at_its_recorded_basis_and_enters_the_recheck_census(
    tmp_path: Path,
) -> None:
    """RC-2 supersedes the striker-4 currency ruling for grantable records.

    A rule-identity move never stales a settled head (currency is
    unconditional); whether it impugns the recorded disposition is the
    measured rotation class, and an UNMEASURED move reads soundness in the
    recheck census -- the head is current AND needs_recheck on every moved
    policy axis.
    """

    from menagerie.crawler.recheck_census import recheck_census
    from menagerie.crawler.reducer import _select_current
    from menagerie.crawler.recordio import scan_jsonl

    context = make_authority_context(["m_example"])
    paths = _paths(tmp_path)
    _held_terminal(paths, context, held=False)
    upgraded = replace(context, **_MOVED_RULE_IDENTITIES)

    projection = project_dependency_current(paths, context=upgraded)

    assert "m_example" in projection.current_records, projection.stale_reasons
    census = recheck_census(_select_current(scan_jsonl(paths.models)), (), upgraded)
    assert census["counts"]["needs_recheck"] == 1
    # A failed:source terminal never reached the runner or the award closure,
    # so those two axes are recorded not-applicable and cannot move.
    assert set(census["by_axis"]) == {
        "author_result_schema_identity",
        "author_dispatcher_identity",
        "author_prompt_identity",
        "checker_prompt_identity",
        "terminal_rule_identity",
        "publication_policy_identity",
    }


@pytest.mark.smoke
def test_review_held_record_contradicting_its_own_evidence_still_refuses(
    tmp_path: Path,
) -> None:
    """The record-anchored cross-checks keep full strength for held records."""

    from menagerie.crawler.recordio import payload_hash

    context = make_authority_context(["m_example"])
    paths = _paths(tmp_path)
    _held_terminal(paths, context, held=True)
    rows = [json.loads(line) for line in paths.models.read_text().splitlines()]
    assert len(rows) == 1
    tampered = rows[0]
    # A coherent ledger rewrite that contradicts the record's own decisive
    # attempt proof: the reason code no longer matches the evidence.
    tampered["status"] = dict(tampered["status"])
    tampered["status"]["reason_code"] = "source-model-mismatch"
    tampered.pop("record_revision", None)
    tampered["record_revision"] = payload_hash(tampered)
    paths.models.write_text(json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n")

    projection = project_dependency_current(
        paths, context=replace(context, **_MOVED_RULE_IDENTITIES)
    )

    assert "m_example" not in projection.current_records
    assert "m_example" in projection.stale_reasons


def _historical_mislabel_head(paths: LedgerPaths, context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write one first-mint mislabel row straight into the models ledger.

    The R24 shape: ``failed:accuracy-gate`` + ``terminal-disposition-rejected``
    + ``accuracy_gate.verdict = None`` + a mandatory review hold, proved by a
    checker-REJECTED ``terminal_disposition`` gate. ``prepare_model`` derives
    exactly this shape today (the proof rule's default claim is the historical
    code); the fixture then bypasses ``append_model`` and writes a self-hashed,
    honest-sequence row directly.

    This construction is BOTH sides of the era boundary at once. It is what
    authentic history IS to current code (parent-written rows in the
    hash-validated append-only ledger that no live door would mint fresh) --
    AND it is byte-for-byte the reconciliation's R1 tamper attack: a
    current-law row, current ``terminal_rule_identity``, valid recomputed
    self-hash, written by an actor with direct ledger write access. The only
    thing separating the two is frozen realized-bytes membership, which is
    exactly what the era arm checks: this row REFUSES on replay unless a
    reviewed change registers its recomputed digest in
    ``DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS``.

    Returns the prepared candidate and the persisted row (with its recomputed
    ``record_revision``), so callers can exercise either side.
    """

    from menagerie.crawler.constants import TERMINAL_DISPOSITION_REJECTED_REASON_CODE
    from menagerie.crawler.recordio import payload_hash
    from menagerie.crawler.tests.conftest import HASH, make_gate

    attempt = make_failed_attempt()
    model = bind_terminal_attempts(make_model(status_code="failed:accuracy-gate"), [attempt])
    model["status"]["reason_code"] = TERMINAL_DISPOSITION_REJECTED_REASON_CODE
    model["status"]["human_review"] = {
        "required": True,
        "reason": None,
        "queue": "crawler-human-review",
        "requested_at": model["created_at"],
    }
    gate = make_gate(["m_example"], gate_id="gate-terminal-1", gate_kind="terminal_disposition")
    item = gate["items"][0]
    item["work_id"] = attempt["work_id"]
    item["campaign_root_work_id"] = attempt["work_id"]
    item["terminal_disposition"] = {
        "author_result_id": "result-1",
        "author_result_sha256": HASH,
        "handoff_proposal_id": None,
        "handoff_sha256": None,
        "kind": "SKIP_RECOMMENDATION",
        "predicate": "no-description",
        "verdict": "rejected",
        "source_manifest_identity": HASH,
        "source_ids": ["source-1"],
        "evidence_identity": HASH,
        "evidence_ids": ["evidence-1"],
        "license_identity": HASH,
        "findings": ["claim contradicts the frozen source"],
    }
    gate["result_envelope_sha256"] = stable_hash(
        {
            key: value
            for key, value in gate.items()
            if key not in {"result_envelope_sha256", "payload_sha256", "ledger_seq"}
        }
    )
    with CanonicalReducer(paths, context) as reducer:
        reducer.append_attempt(attempt)
        reducer.append_gate(gate)
        prepared = reducer.prepare_model(model)
    assert prepared["status"]["code"] == "failed:accuracy-gate"
    assert prepared["status"]["reason_code"] == TERMINAL_DISPOSITION_REJECTED_REASON_CODE
    assert prepared["accuracy_gate"]["verdict"] is None
    row = dict(prepared)
    row["record_seq"] = 1
    row.pop("record_revision", None)
    row["record_revision"] = payload_hash(row)
    with paths.models.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return prepared, row


@pytest.mark.smoke
def test_ledger_resident_tamper_row_refuses_on_replay(tmp_path: Path) -> None:
    """The reconciliation's R1 kill, retired at the projection: a self-hashed,
    honest-sequence, current-law row of the historical shape written straight
    into the models ledger does NOT vouch under era passage.

    Ledger residency via replay mode is caller identity, not era authority --
    the row's recomputed bytes digest is not a member of the frozen realized
    cohort, so the projection marks it stale with the exact W13 fresh-claim
    refusal, and the boundary partition REFUSES LOUDLY (typed
    ``PartitionError``) rather than blessing it as current.
    """

    from menagerie.crawler.status import PartitionError, assert_partition

    context = make_authority_context(["m_example"])
    paths = _paths(tmp_path)
    _prepared, row = _historical_mislabel_head(paths, context)
    assert (
        row["record_revision"]
        not in reducer_module.DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS
    )

    projection = project_dependency_current(paths, context=context)

    assert "m_example" not in projection.current_records
    assert "RENDERED accuracy adjudication" in projection.stale_reasons["m_example"]
    with pytest.raises(PartitionError, match="cannot vouch"):
        assert_partition(
            ["m_example"],
            projection.current_records,
            stale_reasons=projection.stale_reasons,
        )


@pytest.mark.smoke
def test_frozen_cohort_member_vouches_on_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Era honesty is replay-symmetric for frozen realized history.

    The R24 deadlock regression: the pilot members hold 200a-era first-mint
    ``failed:accuracy-gate`` + ``terminal-disposition-rejected`` terminals; the
    fresh-claim refusal fired on their projection replay (no prior same-shape
    head exists at a FIRST mint), every close refused them as unaccounted
    stale, and the requeue grants that re-drive them were consumable only by
    the ordinary selection those closes gate. With the row's recomputed bytes
    digest registered in the frozen cohort -- the reviewed change a real era
    boundary makes -- history validates under the law of its era, and the
    boundary partition passes read-only over the vouched head. Membership is
    over the RECOMPUTED ``record_revision`` bytes-hash: the identical bytes
    with an unregistered digest refuse in the sibling test above.
    """

    from menagerie.crawler.status import assert_partition, record_is_release_eligible

    context = make_authority_context(["m_example"])
    paths = _paths(tmp_path)
    _prepared, row = _historical_mislabel_head(paths, context)
    monkeypatch.setattr(
        reducer_module,
        "DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS",
        frozenset({row["record_revision"]}),
    )

    projection = project_dependency_current(paths, context=context)

    assert projection.stale_reasons == {}
    assert "m_example" in projection.current_records
    head = projection.current_records["m_example"]
    assert head["status"]["code"] == "failed:accuracy-gate"
    assert head["status"]["human_review"]["required"] is True
    report = assert_partition(
        ["m_example"],
        projection.current_records,
        stale_reasons=projection.stale_reasons,
    )
    assert report.valid
    # Recon C4: vouching under era law unlocks NOTHING positive. The head is
    # a failed, review-held terminal: release eligibility excludes it twice
    # (kind and mandatory review) ...
    assert not record_is_release_eligible(head, projection.current_records)
    # ... and live supersession over it stays grant-gated: a candidate that
    # introduces no NEW explicit grant is exactly the review-hold tripwire's
    # blocked shape.
    with CanonicalReducer(paths, context) as reducer:
        blocked = reducer.grant_blocked_supersession(
            {"stable_id": "m_example", "budget": {"explicit_grants": []}}
        )
    assert blocked is not None
    assert blocked["record_revision"] == row["record_revision"]


@pytest.mark.smoke
def test_fresh_live_mint_of_the_mislabel_still_refuses_at_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CURRENT-law violation in a NEW row refuses at the full live doors.

    The identical prepared candidate that replays era-accepted as registered
    history is refused by ``append_model`` AND ``preflight_model`` on every
    live path -- even with its exact digest registered in the frozen cohort
    (recon C4: preflight never gains era passage; the passage requires the
    module-private semantic-replay construction, not row content).
    """

    context = make_authority_context(["m_example"])
    paths = _paths(tmp_path)
    prepared, row = _historical_mislabel_head(paths, context)
    monkeypatch.setattr(
        reducer_module,
        "DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS",
        frozenset({row["record_revision"]}),
    )
    fresh_paths = _paths(tmp_path / "fresh")
    with CanonicalReducer(fresh_paths, context) as reducer:
        with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
            reducer.append_model(dict(prepared))
        with pytest.raises(ReductionError, match="RENDERED accuracy adjudication"):
            reducer.preflight_model(dict(prepared))


@pytest.mark.smoke
def test_adoption_shaped_head_over_a_frozen_cohort_member_rides_continuity_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The era-cohort x adoption overlap (fable F14), composed for the first time.

    An adoption-shaped supersession -- the containment hold marking re-appended
    verbatim under a MOVED citation -- over a frozen-cohort mislabel head is
    both sides at once: the parent vouches through the W13 era passage (its
    recomputed digest is registered), while the adoption row itself rides the
    CONTINUITY passage (a re-append whose current head already carries the
    exact code+reason) plus the moved-citation carve on replay; its OWN digest
    is deliberately NOT in the cohort. The composed head must replay current
    with its disposition UNCHANGED -- still the failed, review-held terminal --
    and can upgrade nothing: release eligibility keeps excluding it and live
    supersession over it stays grant-gated.
    """

    from menagerie.crawler.recordio import payload_hash
    from menagerie.crawler.status import record_is_release_eligible
    from menagerie.crawler.terminal_review import terminal_review_hold_candidate

    context = make_authority_context(["m_example"])
    paths = _paths(tmp_path)
    _prepared, row = _historical_mislabel_head(paths, context)
    monkeypatch.setattr(
        reducer_module,
        "DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS",
        frozenset({row["record_revision"]}),
    )
    adoption = terminal_review_hold_candidate(
        row,
        requested_at=str(row["status"]["human_review"]["requested_at"]),
        # The adoption shape: hold marking preserved verbatim, citation MOVED
        # to the adopting run ("position is provenance").
        producer_run_id="run-adopting-era",
        machine_id="machine-adopting-era",
    )
    adopted = dict(adoption)
    adopted["record_seq"] = 2
    adopted.pop("record_revision", None)
    adopted["record_revision"] = payload_hash(adopted)
    with paths.models.open("a") as handle:
        handle.write(json.dumps(adopted, sort_keys=True, separators=(",", ":")) + "\n")
    # The composed row rides CONTINUITY, never cohort membership of its own.
    assert (
        adopted["record_revision"]
        not in reducer_module.DISPOSITION_MISLABEL_ERA_RECORD_REVISIONS
    )

    projection = project_dependency_current(paths, context=context)

    assert projection.stale_reasons == {}
    head = projection.current_records["m_example"]
    assert head["record_revision"] == adopted["record_revision"]
    # Disposition UNCHANGED: the era head's failed, review-held terminal
    # survives the passage byte-for-byte on every disposition-bearing fact.
    assert head["status"]["code"] == "failed:accuracy-gate"
    assert head["status"]["reason_code"] == row["status"]["reason_code"]
    assert head["status"]["kind"] == "failed"
    assert head["status"]["human_review"]["required"] is True
    assert head["provenance"]["producer_run_id"] == "run-adopting-era"
    # Cannot upgrade: release eligibility excludes it (kind and mandatory
    # review), and live supersession over it stays grant-gated.
    assert not record_is_release_eligible(head, projection.current_records)
    with CanonicalReducer(paths, context) as reducer:
        blocked = reducer.grant_blocked_supersession(
            {"stable_id": "m_example", "budget": {"explicit_grants": []}}
        )
    assert blocked is not None
    assert blocked["record_revision"] == adopted["record_revision"]


@pytest.mark.smoke
def test_award_restoration_runs_gate_contains_the_era_arm(tmp_path: Path) -> None:
    """The NAMED containment (recon C1): restoration cannot reach the era arm.

    ``append_award_restoration`` is the one replay-mode path that appends
    without the live ``append_model`` door, and it hands a NOVEL, non-resident
    candidate to its replay reducer. Two independent facts keep the W13 era
    arm out of its reach, both pinned here: the runs-only target gate refuses
    any non-award target (so a candidate of the mislabel shape can never be
    minted through it), and the replay construction declares
    ``source_record=None`` -- non-residency -- so even a future non-runs
    restoration sibling would carry no era authority.
    """

    import inspect

    context = make_authority_context(["m_example"])
    paths = _paths(tmp_path)
    _prepared, row = _historical_mislabel_head(paths, context)
    with CanonicalReducer(paths, context) as reducer:
        with pytest.raises(ReductionError, match="restoration target is not a runs award"):
            reducer.append_award_restoration(
                stable_id="m_example",
                restored_award_revision=row["record_revision"],
                superseding_revision=row["record_revision"],
                restoration_event_id="evt-era-containment",
                dry_run=True,
            )
    source = inspect.getsource(CanonicalReducer.append_award_restoration)
    assert "source_record=None" in source


@pytest.mark.smoke
def test_fresh_gate_write_still_recomputes_component_identities(tmp_path: Path) -> None:
    """The anti-templating recompute is armed at the WRITE door.

    ``load_current_gate_proof`` validates sealed gates at their recorded
    component identities (see ``test_authority``), so the zero-argument
    recompute against shipped bytes moved to ``reducer.append_gate``: a gate
    being sealed into the ledger NOW that claims an identity the machine
    cannot re-derive refuses at the door, and a genuinely derived gate still
    appends.
    """

    from menagerie.crawler.tests.conftest import make_gate

    def _reseal(gate: dict[str, Any]) -> dict[str, Any]:
        resealed = {
            key: value for key, value in gate.items() if key not in {"payload_sha256", "ledger_seq"}
        }
        resealed["result_envelope_sha256"] = stable_hash(
            {key: value for key, value in resealed.items() if key != "result_envelope_sha256"}
        )
        return resealed

    with CanonicalReducer(_paths(tmp_path), make_authority_context(["m_0"])) as reducer:
        genuine = _reseal(make_gate(["m_0"], gate_id="gate-genuine"))
        assert reducer.append_gate(genuine).appended

        templated = make_gate(["m_0"], gate_id="gate-templated")
        templated["dispatcher_identity"] = "sha256:" + "a" * 64
        with pytest.raises(ReductionError, match="conflicts with the identity derived"):
            reducer.append_gate(_reseal(templated))


# --------------------------------------------------------------------------
# Rung-11 leg-6: re-drive terminals must bind the ACTIVE work generation.
#
# Three refusals killed leg 6 (TerminalRecordingUnavailable, streak m4256,
# m4257, m9617), replay-verified against the frozen pilot ledgers:
#
# - ``_driver_failure_attempt`` defaulted a proposal-less failure to the
#   campaign-root ``work-{stable_id}``, filing a re-drive generation's lane
#   failure into the ROOT generation. The reducer then derived the terminal
#   proof from the root generation's env-carrying forwards while the fresh
#   attempt-free candidate carried the placeholder environment generation:
#   ``model environment generation contradicts its attempt proof`` (m9617).
# - A checker-rejected terminal recommendation references NO attempt, NO
#   accepted gate, and NO retained proposal, so its ONLY durable evidence of
#   the granted new generation is its staged artifact transaction -- which
#   ``_revision_work_ids`` did not count: ``superseding revision does not
#   bind the granted new-work identity`` (m4256/m4257).
# - ``prepare_model`` fell back to a FOREIGN generation's artifact events
#   when the candidate's resolved generation had none, re-imposing the root
#   generation's published custody on a re-drive terminal.
#
# The accept path for the granted supersession is additionally proven against
# the real frozen pilot artifacts by the leg-7 dry-verify replay (sprint
# evidence: striker-7 leg-6 replay); these regressions pin each site both
# ways on synthetic ledgers.
# --------------------------------------------------------------------------

REDRIVE_WORK_ID = stable_hash({"redrive-generation": "m_example"})


def _redrive_failure_attempt(
    tmp_path: Path,
    *,
    active_work_id: str,
    created_at: str = T2,
    message: str = "proposal source fields differ from controlled-fetch manifest",
) -> dict[str, Any]:
    """Re-derive one proposal-less lane failure through the REAL builder."""

    diagnostics = tmp_path / ".crawl-local" / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    return _driver_failure_attempt(
        SimpleNamespace(  # type: ignore[arg-type]
            stable_id="m_example", active_work_id=active_work_id
        ),
        None,
        "author",
        "malformed-result",
        RuntimeError(message),
        SimpleNamespace(machine_id="machine-test"),  # type: ignore[arg-type]
        diagnostics_root=diagnostics,
        environment=None,
        created_at=created_at,
    )


@pytest.mark.smoke
def test_driver_failure_attempt_binds_the_active_work_generation(tmp_path: Path) -> None:
    """A proposal-less lane failure is evidence of the generation it happened in.

    Rung-11 leg-6 regression (m9617): the builder defaulted to the campaign
    root ``work-{stable_id}``, so a requeue/refresh generation's failure was
    filed into the ROOT generation and inherited its terminal proof.
    """

    root = _redrive_failure_attempt(tmp_path, active_work_id="work-m_example")
    redrive = _redrive_failure_attempt(tmp_path, active_work_id=REDRIVE_WORK_ID)
    assert root["work_id"] == "work-m_example"
    assert redrive["work_id"] == REDRIVE_WORK_ID
    # Identities are minted per generation: the same failure signature in a
    # new generation is a NEW observation, never a replay of the root row.
    assert root["attempt_id"] != redrive["attempt_id"]


def _root_generation_with_environment_proof(reducer: CanonicalReducer) -> str:
    """Append a root-generation terminal whose proof pins an environment.

    Returns
    -------
    str
        The appended root revision hash.
    """

    forward_ok = make_attempt(attempt_id="attempt-forward-ok")
    failed_forward = make_failed_attempt(
        attempt_id="attempt-forward-failed", stage="forward", reason_code="exception"
    )
    failed_forward.pop("ledger_seq")
    failed_forward.pop("payload_sha256")
    root = bind_terminal_attempts(
        make_model(status_code="failed:forward"), [forward_ok, failed_forward]
    )
    root["status"]["reason_code"] = "exception"
    reducer.append_attempt(forward_ok)
    reducer.append_attempt(failed_forward)
    prepared = reducer.prepare_model(root)
    return str(reducer.append_model(prepared).record["record_revision"])


@pytest.mark.smoke
def test_redrive_terminal_records_beside_the_root_generations_environment_proof(
    tmp_path: Path,
) -> None:
    """The m9617 shape records cleanly after the fix.

    The root generation holds an env-carrying succeeded forward (its recorded
    proof). The re-drive generation's author-lane failure terminal derives its
    proof from ITS OWN generation's attempts -- which carry no environment --
    so the attempt-free placeholder environment generation is honest and the
    terminal records, superseding the stale root record.
    """

    with CanonicalReducer(_paths(tmp_path), make_authority_context(["m_example"])) as reducer:
        parent = _root_generation_with_environment_proof(reducer)
        failure = _redrive_failure_attempt(tmp_path, active_work_id=REDRIVE_WORK_ID)
        persisted = reducer.append_attempt(failure).record
        assert persisted["work_id"] == REDRIVE_WORK_ID
        terminal = bind_terminal_attempts(make_model(status_code="failed:author"), [persisted])
        terminal["status"]["reason_code"] = "malformed-result"
        terminal["record_seq"] = 2
        terminal["parent_revision"] = parent
        terminal["status"]["supersedes_revision"] = parent
        prepared = reducer.prepare_model(terminal)
        # The re-drive generation staged nothing, so it honestly has no
        # artifact authority -- the root generation's custody is NOT inherited.
        assert prepared["artifact_authority"]["state"] == "not-applicable"
        appended = reducer.append_model(prepared)
        assert appended.appended
        assert appended.record["status"]["code"] == "failed:author"


@pytest.mark.smoke
def test_terminal_contradicting_its_own_generations_environment_still_refuses(
    tmp_path: Path,
) -> None:
    """The tripwire stays armed: a record contradicting its OWN proof refuses.

    Same-generation fault injection: the decisive attempts pin one exact
    environment generation, and a candidate stamped with a different one is a
    genuine contradiction, not history-vs-live drift.
    """

    forward_ok = make_attempt(attempt_id="attempt-forward-ok")
    failure = make_failed_attempt(
        attempt_id="attempt-author-failed", stage="author", reason_code="malformed-result"
    )
    failure.pop("ledger_seq")
    failure.pop("payload_sha256")
    candidate = bind_terminal_attempts(make_model(status_code="failed:author"), [failure])
    candidate["status"]["reason_code"] = "malformed-result"
    with CanonicalReducer(_paths(tmp_path), make_authority_context(["m_example"])) as reducer:
        reducer.append_attempt(forward_ok)
        reducer.append_attempt(failure)
        tampered = reducer.prepare_model(candidate)
        # The generation's per-mode proof pins the forward's exact environment.
        tampered["execution"]["env_generation"] = stable_hash({"tampered": "generation"})
        with pytest.raises(
            ReductionError, match="environment generation contradicts its attempt proof"
        ):
            reducer.append_model(tampered)


@pytest.mark.smoke
def test_revision_work_ids_counts_the_referenced_artifact_transaction() -> None:
    """The revision's own artifact transaction binds its work generation.

    Rung-11 leg-6 regression (m4256/m4257): a checker-rejected terminal
    recommendation's staged transaction is its only durable evidence of the
    granted generation. Foreign transactions and foreign models stay excluded.
    """

    transaction_id = "sha256:" + "1" * 64
    revision = {
        "stable_id": "m_example",
        "status": {},
        "execution": {},
        "artifact_authority": {"state": "staged-private", "transaction_id": transaction_id},
    }
    events = (
        {
            "transaction_id": transaction_id,
            "stable_id": "m_example",
            "work_id": REDRIVE_WORK_ID,
        },
        {
            "transaction_id": transaction_id,
            "stable_id": "m_other",
            "work_id": "work-foreign-model",
        },
        {
            "transaction_id": "sha256:" + "2" * 64,
            "stable_id": "m_example",
            "work_id": "work-foreign-transaction",
        },
    )
    assert _revision_work_ids(revision, (), (), events) == frozenset({REDRIVE_WORK_ID})
    # Without the artifact evidence the revision binds nothing.
    assert _revision_work_ids(revision, (), (), ()) == frozenset()


@pytest.mark.smoke
def test_granted_supersession_binding_no_new_generation_evidence_still_refuses(
    tmp_path: Path,
) -> None:
    """A grant-consuming supersession whose evidence is all root-generation refuses.

    Both-ways guard for the widened evidence set: counting the artifact
    transaction must not weaken the rule -- a superseding revision that
    references only root-generation evidence (and no new-generation artifact)
    still cannot bind the granted new-work identity.
    """

    from menagerie.crawler.identity import canonical_json_bytes

    paths = _paths(tmp_path)
    failed_attempt = make_failed_attempt()
    first = bind_terminal_attempts(make_model(status_code="failed:source"), [failed_attempt])
    first["status"]["human_review"] = {
        "required": True,
        "reason": None,
        "queue": "crawler-human-review",
        "requested_at": first["created_at"],
    }
    with CanonicalReducer(paths, make_authority_context(["m_example"])) as reducer:
        reducer.append_attempt(failed_attempt)
        parent = reducer.append_model(reducer.prepare_model(first)).record["record_revision"]
        reason = "reviewed corrected source"
        grant_id = stable_hash(
            {"stable_id": "m_example", "stage": "source", "reason": reason, "grant": 1}
        )
        grant = {
            "grant_id": grant_id,
            "stable_id": "m_example",
            "stage": "source",
            "reason": reason,
            "attempts": 1,
            "created_at": first["created_at"],
        }
        operational = paths.models.parent.parent / "operational"
        operational.mkdir(parents=True, exist_ok=True)
        (operational / "requeue-grants.jsonl").write_bytes(canonical_json_bytes(grant) + b"\n")
        new_work_id = stable_hash(
            {
                "stable_id": "m_example",
                "grant_id": grant_id,
                "parent_revision": parent,
                "generation": 1,
            }
        )
        event = {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "event_id": f"requeue-consumed-{grant_id[7:31]}",
            "created_at": first["created_at"],
            "event_kind": "requeue-grant-consumed",
            "status": "requeue-grant-consumed",
            "provider": None,
            "observed_response": None,
            "reset_at": None,
            "queued_work_counts": {"models": 1},
            "current_environment": None,
            "run_id": "run-test",
            "machine_id": "machine-test",
            "details": {
                "grant_id": grant_id,
                "stable_id": "m_example",
                "stage": "source",
                "reason": reason,
                "attempts": 1,
                "source_record_revision": parent,
                "new_work_generation": 1,
                "new_work_id": new_work_id,
            },
        }
        with JsonlLedger(operational / "events.jsonl", OPERATIONAL_EVENT_SCHEMA_VERSION) as ledger:
            ledger.append(event)
        # The superseding revision references ONLY the root-generation attempt.
        second = bind_terminal_attempts(
            make_model(status_code="failed:source", attempt_id="attempt-1"),
            [failed_attempt],
        )
        second = reducer.prepare_model(second)
        second["record_seq"] = 2
        second["parent_revision"] = parent
        second["status"]["supersedes_revision"] = parent
        second["budget"]["explicit_grants"] = [grant_id]
        with pytest.raises(ReductionError, match="does not bind the granted new-work identity"):
            reducer.append_model(second)
